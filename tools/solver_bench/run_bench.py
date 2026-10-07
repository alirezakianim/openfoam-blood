#!/usr/bin/env python3
"""run_bench.py -- build and run the UMFPACK / MKL PARDISO / cuDSS benchmarks on umfdump captures.

Colab (default mode, Linux, g++):
    python3 run_bench.py --capture /content/umfdump/call1 [--capture ...] --out /content/bench
Shim only (build libumfdump.so and self-test it before the solver run):
    python3 run_bench.py --shim-only --out /content/bench
Local (macOS, clang++, the local SuiteSparse; PARDISO and cuDSS are skipped when absent):
    python3 run_bench.py --local --capture <dir>/call1 --out <dir>/bench_local

Writes <out>/results.json, <out>/results.md, <out>/build.log, <out>/logs/*.log, <out>/bin/*.
Only the standard library is used (no numpy).
"""
import argparse
import array
import datetime
import glob
import json
import os
import platform
import shutil
import site
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
IS_MAC = platform.system() == "Darwin"


def sh(cmd, env=None, timeout=None, cwd=None):
    """Run a command (list); return (rc, combined output). Never raises for a missing binary."""
    try:
        p = subprocess.run(cmd, env=env, timeout=timeout, cwd=cwd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors="replace")
        return p.returncode, p.stdout
    except FileNotFoundError as e:
        return 127, str(e)
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        return -9, out + "\n[timeout after %s s]" % timeout


class Log:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, msg):
        line = "[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


# --------------------------------------------------------------------------- hardware

def hardware():
    hw = {"platform": platform.platform(), "python": sys.version.split()[0], "nproc": os.cpu_count()}
    cmds = {
        "uname": ["uname", "-a"],
        "nproc": ["nproc"],
        "lscpu": ["lscpu"],
        "free": ["free", "-b"],
        "os_release": ["cat", "/etc/os-release"],
        "gcc": ["gcc", "--version"],
        "nvidia_smi": ["nvidia-smi"],
        "nvidia_smi_query": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                             "--format=csv,noheader"],
        "blas_alternative": ["update-alternatives", "--display", "libblas.so.3-x86_64-linux-gnu"],
        "ldconfig_blas": ["sh", "-c", "ldconfig -p 2>/dev/null | grep -E 'libblas|libopenblas|libumfpack|libmkl_rt|libcudss|libcublas' | head -20"],
    }
    if IS_MAC:
        cmds = {"uname": ["uname", "-a"], "sw_vers": ["sw_vers"],
                "sysctl": ["sysctl", "machdep.cpu.brand_string", "hw.ncpu", "hw.perflevel0.physicalcpu",
                           "hw.perflevel1.physicalcpu", "hw.memsize"]}
    for k, c in cmds.items():
        rc, out = sh(c, timeout=60)
        hw[k] = out.strip() if rc == 0 else "unavailable (rc %d)" % rc
    try:
        n = int(hw.get("nproc") if not IS_MAC else os.cpu_count())
        hw["nproc_int"] = n
    except (TypeError, ValueError):
        hw["nproc_int"] = os.cpu_count() or 1
    return hw


# --------------------------------------------------------------------------- discovery

def site_dirs():
    dirs = []
    try:
        dirs += site.getsitepackages()
    except AttributeError:
        pass
    try:
        dirs.append(site.getusersitepackages())
    except AttributeError:
        pass
    dirs += [p for p in sys.path if p and os.path.isdir(p)]
    seen, out = set(), []
    for d in dirs:
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def find_mkl(root=None):
    roots = [root] if root else []
    roots += [os.environ.get("MKLROOT"), sysconfig.get_paths().get("data"), sys.prefix, "/usr/local",
              getattr(site, "USER_BASE", None), "/opt/intel/oneapi/mkl/latest"]
    tried = []
    for r in [x for x in roots if x]:
        inc = os.path.join(r, "include", "mkl_pardiso.h")
        libs = sorted(glob.glob(os.path.join(r, "lib", "libmkl_rt.so*")) +
                      glob.glob(os.path.join(r, "lib", "intel64", "libmkl_rt.so*")) +
                      glob.glob(os.path.join(r, "lib", "libmkl_rt*.dylib")))
        tried.append(r)
        if os.path.isfile(inc) and libs:
            lib = [l for l in libs if l.endswith(".so.2")] or libs
            return {"include": os.path.dirname(inc), "lib": lib[0], "libdir": os.path.dirname(lib[0])}, None
    return None, "mkl_pardiso.h + libmkl_rt not found under: " + ", ".join(tried)


def find_cudss(root=None):
    cands = []
    if root:
        cands.append(root)
    for d in site_dirs():
        cands += [os.path.join(d, "nvidia", "cu12"), os.path.join(d, "nvidia", "cudss")]
    for c in cands:
        inc = os.path.join(c, "include", "cudss.h")
        libs = sorted(glob.glob(os.path.join(c, "lib", "libcudss.so*")))
        libs = [l for l in libs if os.path.basename(l).startswith("libcudss.so")]
        if os.path.isfile(inc) and libs:
            lib = [l for l in libs if l.endswith(".so.0")] or libs
            mt = sorted(glob.glob(os.path.join(c, "lib", "libcudss_mtlayer_gomp.so*")))
            return {"include": os.path.dirname(inc), "lib": lib[0], "libdir": os.path.dirname(lib[0]),
                    "mtlayer": mt[0] if mt else None}, None
    return None, "cudss.h + libcudss.so.0 not found in site-packages (pip install nvidia-cudss-cu12)"


def find_gpu():
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi absent (no NVIDIA GPU runtime)"
    rc, out = sh(["nvidia-smi", "-L"], timeout=60)
    if rc != 0 or "GPU" not in out:
        return False, "nvidia-smi -L found no GPU (rc %d): %s" % (rc, out.strip()[:200])
    return True, out.strip()


def find_cuda(root):
    inc = os.path.join(root, "include", "cuda_runtime_api.h")
    lib = os.path.join(root, "lib64")
    if os.path.isfile(inc) and glob.glob(os.path.join(lib, "libcudart.so*")):
        return {"root": root, "include": os.path.dirname(inc), "libdir": lib}, None
    return None, "CUDA runtime headers/libcudart not found under %s" % root


# --------------------------------------------------------------------------- build

def build(name, cmd, log):
    log("BUILD %s: %s" % (name, " ".join(cmd)))
    rc, out = sh(cmd, timeout=900)
    for line in out.strip().splitlines():
        log("  | " + line)
    log("BUILD %s: %s" % (name, "ok" if rc == 0 else "FAILED rc %d" % rc))
    return rc == 0


def umf_flags(args):
    if args.local:
        p = args.suitesparse
        return (["-I%s/include/suitesparse" % p, "-I%s/include" % p],
                ["-L%s/lib" % p, "-Wl,-rpath,%s/lib" % p, "-lumfpack", "-lsuitesparseconfig", "-lomp"])
    return ["-I/usr/include/suitesparse"], ["-lumfpack", "-ldl"]


def build_shim(args, bindir, log):
    inc, _ = umf_flags(args)
    if IS_MAC:
        out = os.path.join(bindir, "libumfdump.dylib")
        cmd = ["clang", "-O2", "-Wall", "-dynamiclib", "-fvisibility=hidden"] + inc + \
              [os.path.join(HERE, "umfdump.c"), "-o", out]
    else:
        out = os.path.join(bindir, "libumfdump.so")
        cmd = [args.cc, "-O2", "-Wall", "-shared", "-fPIC", "-fvisibility=hidden"] + inc + \
              [os.path.join(HERE, "umfdump.c"), "-o", out, "-ldl", "-lpthread"]
    return out if build("umfdump", cmd, log) else None


# --------------------------------------------------------------------------- run

def parse_result(out):
    res = None
    for line in out.splitlines():
        if line.startswith("RESULT_JSON "):
            try:
                res = json.loads(line[len("RESULT_JSON "):])
            except json.JSONDecodeError as e:
                res = {"error": "unparsable RESULT_JSON: %s" % e}
    return res


def run(tag, cmd, env, logdir, log, timeout):
    path = os.path.join(logdir, tag + ".log")
    log("RUN %s: %s" % (tag, " ".join(cmd)))
    rc, out = sh(cmd, env=env, timeout=timeout)
    with open(path, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n" + out)
    res = parse_result(out) or {"error": "no RESULT_JSON (rc %d); see %s" % (rc, path)}
    res["rc"] = rc
    res["log"] = path
    summary = "rc %d" % rc
    if "error" in res:
        summary += " ERROR " + str(res["error"])[:200]
    else:
        f = lambda k: (res.get(k) or {}).get("med")
        summary += " factor %s refactor %s solve %s relres2 %s" % (f("factor_s"), f("refactor_s"), f("solve_s"),
                                                                    res.get("relres2"))
    log("RUN %s: %s" % (tag, summary))
    return res


def thread_list(spec, nproc):
    out, skipped = [], []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        t = nproc if tok == "nproc" else int(tok)
        if t > nproc:
            skipped.append(t)
        elif t >= 1 and t not in out:
            out.append(t)
    return sorted(out), skipped


def env_threads(base, t, mkl=False):
    e = dict(base)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "GOTO_NUM_THREADS", "MKL_NUM_THREADS"):
        e[k] = str(t)
    if mkl:
        e.update({"MKL_THREADING_LAYER": "INTEL", "MKL_INTERFACE_LAYER": "LP64", "MKL_DYNAMIC": "FALSE"})
    return e


def prepend(env, key, value):
    env = dict(env)
    env[key] = value + (":" + env[key] if env.get(key) else "")
    return env


# --------------------------------------------------------------------------- shim self-test

def write_synthetic(prefix, m=40):
    """Unsymmetric 2D convection-diffusion, 5-point stencil, n = m*m, as an umfdump-style capture."""
    n = m * m
    Ap, Ai, Ax = array.array("i", [0]), array.array("i"), array.array("d")
    for j in range(n):
        gr, gc = divmod(j, m)
        rows = []
        if gr > 0:
            rows.append((j - m, -1.0 + 0.1))   # A[j-m, j]: row j-m's +m neighbour
        if gc > 0:
            rows.append((j - 1, -1.0 + 0.3))   # A[j-1, j]
        rows.append((j, 4.2))
        if gc < m - 1:
            rows.append((j + 1, -1.0 - 0.3))   # A[j+1, j]
        if gr < m - 1:
            rows.append((j + m, -1.0 - 0.1))   # A[j+m, j]
        for i, v in rows:
            Ai.append(i)
            Ax.append(v)
        Ap.append(len(Ai))
    b = array.array("d", [1.0 + (k % 7) * 0.25 for k in range(n)])
    for ext, arr in ((".Ap", Ap), (".Ai", Ai), (".Ax", Ax), (".b", b)):
        with open(prefix + ext, "wb") as f:
            arr.tofile(f)
    return n


def same_file(a, b):
    try:
        with open(a, "rb") as fa, open(b, "rb") as fb:
            return fa.read() == fb.read()
    except OSError:
        return False


def shim_selftest(args, bindir, shim, log):
    """bench_umf as the host, with and without the shim, on a synthetic matrix."""
    st = os.path.join(args.out, "shim_selftest")
    shutil.rmtree(st, ignore_errors=True)
    os.makedirs(st)
    inp = os.path.join(st, "in")
    n = write_synthetic(inp)
    inc, libs = umf_flags(args)
    host = os.path.join(bindir, "bench_umf_host")
    extra = ["-Wl,-flat_namespace"] if IS_MAC else []   # macOS: ELF-like binding (see README)
    if not build("bench_umf_host", [args.cxx, "-O2", "-std=c++17"] + inc + [os.path.join(HERE, "bench_umf.cpp")] +
                 libs + extra + ["-o", host], log):
        return {"ok": False, "error": "host build failed"}
    base = env_threads(os.environ, 1)
    for k in ("UMFDUMP_DIR", "UMFDUMP_CALLS", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"):
        base.pop(k, None)

    def host_run(tag, dump, calls):
        e = dict(base)
        if dump:
            e.update({"UMFDUMP_DIR": dump, "UMFDUMP_CALLS": calls})
            if IS_MAC:
                e.update({"DYLD_INSERT_LIBRARIES": shim, "DYLD_FORCE_FLAT_NAMESPACE": "1"})
            else:
                e["LD_PRELOAD"] = shim
        return run("selftest_" + tag, [host, inp, "1", "1"], e, st, log, 600)

    keys = ("n", "nnz", "lu_nnz", "lnz", "unz", "rcond", "flops", "relres2", "relresInf", "refactor_relres2",
            "ordering_used", "strategy_used", "status", "rc")
    ref = host_run("noshim", None, None)
    dump = os.path.join(st, "dump")
    got = host_run("shim", dump, "1,2")
    neg_dump = os.path.join(st, "dump_neg")
    neg = host_run("shim_outofrange", neg_dump, "9")
    checks = {
        "results_unchanged_by_shim": all(ref.get(k) == got.get(k) for k in keys),
        "results_unchanged_outofrange": all(ref.get(k) == neg.get(k) for k in keys),
        "DONE_written": os.path.isfile(os.path.join(dump, "DONE")),
        "Ap_Ai_Ax_b_identical": all(same_file(os.path.join(dump, "call1" + e), inp + e)
                                    for e in (".Ap", ".Ai", ".Ax", ".b")),
        "call2_identical": all(same_file(os.path.join(dump, "call2" + e), inp + e) for e in (".Ap", ".Ai", ".Ax", ".b")),
        "x_written": os.path.getsize(os.path.join(dump, "call1.x")) == 8 * n if os.path.isfile(
            os.path.join(dump, "call1.x")) else False,
        "outofrange_no_DONE": not os.path.exists(os.path.join(neg_dump, "DONE")),
        "outofrange_no_files": not glob.glob(os.path.join(neg_dump, "call*")),
    }
    ok = all(checks.values())
    for k, v in checks.items():
        log("SELFTEST %s %s" % ("PASS" if v else "FAIL", k))
    log("SELFTEST %s" % ("ALL PASS" if ok else "FAILED -- do not trust a capture from this shim build"))
    return {"ok": ok, "checks": checks, "n": n, "reference": {k: ref.get(k) for k in keys},
            "with_shim": {k: got.get(k) for k in keys}}


# --------------------------------------------------------------------------- report

def med(r, k):
    v = r.get(k)
    return v.get("med") if isinstance(v, dict) else None


def fmt(v, f="%.3f"):
    return "-" if v is None else (f % v)


def step_time(r):
    a, b = med(r, "refactor_s"), med(r, "refactor_solve_s")
    if a is None or b is None:
        return None
    return a + b + (med(r, "h2d_values_s") or 0.0) + (med(r, "d2h_x_s") or 0.0)


def write_md(path, results, meta):
    lines = ["# Sparse direct solver benchmark", "",
             "- generated %s, mode %s" % (meta["generated"], meta["mode"]),
             "- hardware: %s; nproc %s" % ("; ".join(str(meta["cpu"]).splitlines()), meta["nproc"]),
             "- GPU: %s" % meta["gpu"], "",
             "Times are medians in seconds. step = refactor + its solve (+ H2D values + D2H x for cuDSS): the "
             "per-step cost when the pattern is unchanged. speedup = UMFPACK 1-thread step / this step, same capture.",
             "",
             "| capture | solver | thr | analysis | factor | refactor | solve | step | speedup | relres2 | relresInf | xdiff | LU nnz | pivots | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    base = {}
    for r in results:
        if r.get("solver") == "umfpack" and r.get("threads") == 1 and step_time(r):
            base[r.get("capture")] = step_time(r)
    for r in results:
        st = step_time(r)
        b = base.get(r.get("capture"))
        piv = r.get("perturbed_pivots", r.get("npivots", "-"))
        status = "ERROR: " + str(r["error"])[:80] if "error" in r else str(r.get("status", "-"))
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            os.path.basename(str(r.get("capture", "-"))), r.get("solver", "-"), r.get("threads", "-"),
            fmt(med(r, "analysis_s")), fmt(med(r, "factor_s")), fmt(med(r, "refactor_s")), fmt(med(r, "solve_s")),
            fmt(st), fmt(b / st if (b and st) else None, "%.2fx"), fmt(r.get("relres2"), "%.2e"),
            fmt(r.get("relresInf"), "%.2e"), fmt(r.get("xdiff"), "%.2e"), fmt(r.get("lu_nnz"), "%.4g"), piv, status))
    for s in meta.get("skipped", []):
        lines.append("")
        lines.append("- SKIPPED %s" % s)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", action="append", default=[], help="capture prefix, e.g. /content/umfdump/call1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3, help="fresh analysis+factor+solve repetitions")
    ap.add_argument("--refacs", type=int, default=None, help="refactorisations on the kept analysis (default reps)")
    ap.add_argument("--threads", default="1,2,4,nproc")
    ap.add_argument("--local", action="store_true", help="macOS/local: clang++, --suitesparse prefix")
    ap.add_argument("--suitesparse", default=os.path.expanduser("~/.cache/openfoam_blood/suitesparse"))
    ap.add_argument("--cxx", default=None)
    ap.add_argument("--cc", default=None)
    ap.add_argument("--mkl-root", default=None)
    ap.add_argument("--cudss-root", default=None)
    ap.add_argument("--cuda-root", default="/usr/local/cuda")
    ap.add_argument("--skip-umf", action="store_true")
    ap.add_argument("--skip-pardiso", action="store_true")
    ap.add_argument("--skip-cudss", action="store_true")
    ap.add_argument("--pardiso-args", default="", help="extra tokens, e.g. 'iparm1=3 iparm23=10'")
    ap.add_argument("--cudss-args", default="", help="extra tokens, e.g. 'ir=2 reorder=4'")
    ap.add_argument("--cudss-mt", action="store_true", help="also run cuDSS with the gomp threading layer")
    ap.add_argument("--shim-only", action="store_true", help="build libumfdump + run its self-test, then exit")
    ap.add_argument("--shim-selftest", action="store_true", help="also self-test the shim before the sweep")
    ap.add_argument("--timeout", type=int, default=3600, help="per-run timeout, s")
    args = ap.parse_args()
    args.cxx = args.cxx or ("clang++" if (args.local or IS_MAC) else "g++")
    args.cc = args.cc or ("clang" if (args.local or IS_MAC) else "gcc")
    args.refacs = args.reps if args.refacs is None else args.refacs
    args.out = os.path.abspath(args.out)
    bindir, logdir = os.path.join(args.out, "bin"), os.path.join(args.out, "logs")
    os.makedirs(bindir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    log = Log(os.path.join(args.out, "build.log"))
    log("run_bench.py start: mode %s, out %s" % ("local" if args.local else "colab", args.out))

    hw = hardware()
    nproc = int(hw.get("nproc_int") or os.cpu_count() or 1)
    gpu_ok, gpu_msg = find_gpu()
    meta = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": "local" if args.local else "colab", "nproc": nproc, "gpu": gpu_msg if gpu_ok else "none",
            "cpu": hw.get("sysctl", "") if IS_MAC else next((l.split(":", 1)[1].strip() for l in
                                                             str(hw.get("lscpu", "")).splitlines()
                                                             if l.startswith("Model name")), "?"),
            "args": vars(args), "skipped": [], "builds": {}}
    out = {"meta": meta, "hardware": hw, "selftest": None, "runs": []}

    def save():
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(out, f, indent=1)
        write_md(os.path.join(args.out, "results.md"), out["runs"], meta)

    if args.shim_only or args.shim_selftest:
        shim = build_shim(args, bindir, log)
        meta["builds"]["umfdump"] = bool(shim)
        out["selftest"] = shim_selftest(args, bindir, shim, log) if shim else {"ok": False, "error": "shim build failed"}
        save()
        if args.shim_only:
            log("shim: %s | self-test %s" % (shim, "PASS" if out["selftest"].get("ok") else "FAIL"))
            return 0 if out["selftest"].get("ok") else 1

    if not args.capture:
        log("no --capture given; nothing to benchmark")
        save()
        return 2
    threads, over = thread_list(args.threads, nproc)
    if over:
        meta["skipped"].append("thread counts %s exceed nproc %d" % (over, nproc))
        log("SKIP threads %s > nproc %d" % (over, nproc))

    # ---- builds
    inc, libs = umf_flags(args)
    exe = {}
    if not args.skip_umf:
        p = os.path.join(bindir, "bench_umf")
        if build("bench_umf", [args.cxx, "-O2", "-std=c++17"] + inc + [os.path.join(HERE, "bench_umf.cpp")] + libs +
                 ["-o", p], log):
            exe["umf"] = p
        meta["builds"]["bench_umf"] = "umf" in exe
    mkl = None
    if not args.skip_pardiso:
        mkl, why = find_mkl(args.mkl_root)
        if mkl is None:
            meta["skipped"].append("PARDISO: " + why)
            log("SKIP PARDISO: " + why)
        else:
            p = os.path.join(bindir, "bench_pardiso")
            rp = ["-Wl,--disable-new-dtags", "-Wl,-rpath," + mkl["libdir"]] if not IS_MAC else ["-Wl,-rpath," + mkl["libdir"]]
            if build("bench_pardiso", [args.cxx, "-O2", "-std=c++17", "-I" + mkl["include"],
                                       os.path.join(HERE, "bench_pardiso.cpp"), "-o", p, mkl["lib"]] + rp +
                     (["-ldl", "-lpthread", "-lm"] if not IS_MAC else []), log):
                exe["pardiso"] = p
            meta["builds"]["bench_pardiso"] = "pardiso" in exe
            meta["mkl"] = mkl
    cudss = cuda = None
    if not args.skip_cudss:
        cudss, why = find_cudss(args.cudss_root)
        cuda, why2 = find_cuda(args.cuda_root)
        reason = None
        if not gpu_ok:
            reason = gpu_msg
        elif cudss is None:
            reason = why
        elif cuda is None:
            reason = why2
        if reason:
            meta["skipped"].append("cuDSS: " + reason)
            log("SKIP cuDSS: " + reason)
        else:
            p = os.path.join(bindir, "bench_cudss")
            if build("bench_cudss", [args.cxx, "-O2", "-std=c++17", "-I" + cudss["include"], "-I" + cuda["include"],
                                     os.path.join(HERE, "bench_cudss.cpp"), "-o", p, cudss["lib"],
                                     "-L" + cuda["libdir"], "-lcudart", "-Wl,--disable-new-dtags",
                                     "-Wl,-rpath," + cudss["libdir"] + ":" + cuda["libdir"],
                                     "-Wl,-rpath-link," + cuda["libdir"], "-ldl", "-lpthread"], log):
                exe["cudss"] = p
            meta["builds"]["bench_cudss"] = "cudss" in exe
            meta["cudss"] = cudss
            if "cudss" in exe:
                rc, o = sh(["ldd", p], timeout=60)
                log("ldd bench_cudss:\n" + o)
    save()

    # ---- sweep
    reps = [str(args.reps), str(args.refacs)]
    for cap in args.capture:
        tag0 = os.path.basename(cap)
        if not os.path.isfile(cap + ".Ap"):
            meta["skipped"].append("capture %s: %s.Ap missing" % (cap, cap))
            log("SKIP capture %s (no .Ap)" % cap)
            continue
        if "umf" in exe:
            for t in threads:
                r = run("%s_umf_t%d" % (tag0, t), [exe["umf"], cap] + reps, env_threads(os.environ, t), logdir, log,
                        args.timeout)
                r["threads"] = t
                out["runs"].append(r)
                save()
        if "pardiso" in exe:
            for t in threads:
                e = prepend(env_threads(os.environ, t, mkl=True), "LD_LIBRARY_PATH", mkl["libdir"])
                r = run("%s_pardiso_t%d" % (tag0, t), [exe["pardiso"], cap, str(t)] + reps + args.pardiso_args.split(),
                        e, logdir, log, args.timeout)
                r["threads"] = t
                out["runs"].append(r)
                save()
        if "cudss" in exe:
            e = prepend(dict(os.environ), "LD_LIBRARY_PATH", cudss["libdir"] + ":" + cuda["libdir"])
            variants = [("cudss", args.cudss_args.split())]
            if args.cudss_mt and cudss.get("mtlayer"):
                variants.append(("cudss_mt", args.cudss_args.split() + ["mtlib=" + cudss["mtlayer"],
                                                                         "host_nthreads=%d" % nproc]))
            for name, extra in variants:
                r = run("%s_%s" % (tag0, name), [exe["cudss"], cap] + reps + extra, e, logdir, log, args.timeout)
                r["threads"] = "gpu" if name == "cudss" else "gpu+%d" % nproc
                out["runs"].append(r)
                save()
    save()
    log("wrote %s and %s" % (os.path.join(args.out, "results.json"), os.path.join(args.out, "results.md")))
    bad = [r for r in out["runs"] if "error" in r or r.get("rc") not in (0,)]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
