#!/usr/bin/env python3
"""Quick facts of a returned run, in one call: the first thing /debug-run reads.

Per child: exit status and its meaning, end time reached or not, why it
stopped (completed / the stopAtClockTime budget / a signal / FatalError /
unexplained), steps, wall time, the assessor's admissibility and its reason,
and the head of any crash stack. Per run: the package sha the runner saw, its
verdict and the batch filter. Reads only the tail of each solver log.

It answers "did it finish, and if not, why did it stop?" -- not "why did the
physics do that". Everything it prints is read from a file it names.

Usage: python3 -B scripts/run_digest.py preflight/<stamp>-<name> [--json]
       python3 -B scripts/run_digest.py --self-test
"""
import glob
import json
import os
import re
import sys
import tempfile

SIGNALS = {4: "SIGILL", 6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE", 9: "SIGKILL",
           11: "SIGSEGV", 15: "SIGTERM"}
TAIL = 512 * 1024
CLOCK_SLACK_S = 120.0       # one step of the slowest child is far below this


def tail(path, n=TAIL):
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - n))
        return f.read().decode(errors="replace")


def read(path, default=None):
    try:
        return open(path, errors="replace").read()
    except OSError:
        return default


def child_facts(d):
    f = {"dir": d}
    spec = json.loads(read(os.path.join(d, "child.json"), "{}") or "{}")
    f["child"] = spec.get("child_id", os.path.basename(d))
    f["series"] = spec.get("series")
    f["end_time_s"] = spec.get("end_time_s")
    rc = (read(os.path.join(d, "run-exit-code.txt"), "") or "").strip()
    f["exit"] = int(rc) if rc.lstrip("-").isdigit() else None
    w = re.search(r"WALL_SECONDS=(\d+)", read(os.path.join(d, "log.wallTime"), "") or "")
    f["wall_s"] = int(w.group(1)) if w else None
    cd = read(os.path.join(d, "system", "controlDict"), "") or ""
    m = re.search(r"type\s+stopAtClockTime;.*?stopTime\s+([0-9.eE+-]+);", cd, re.S)
    f["clock_budget_s"] = float(m.group(1)) if m else None
    log = os.path.join(d, "log.phaseClotAxisymmetricFluid")
    if not os.path.isfile(log):
        f["stop"] = "NOT STARTED (no solver log)"
        return f
    t = tail(log)
    times = re.findall(r"^Time = ([0-9.eE+-]+)", t, re.M)
    f["final_time_s"] = float(times[-1]) if times else None
    clk = re.findall(r"ClockTime = ([0-9.]+) s", t)
    f["clock_s"] = float(clk[-1]) if clk else None
    f["ended"] = bool(re.search(r"^End\s*$", t, re.M))
    stack = []
    if "sigHandler" in t:
        seg = t[t.index("sigHandler"):].splitlines()[1:12]
        stack = [ln.strip() for ln in seg if ln.strip().startswith("#")][:6]
    fatal = re.search(r"--> FOAM FATAL (?:IO )?ERROR.*?\n\n", t, re.S)
    f["stack"] = stack
    f["fatal"] = " ".join(fatal.group(0).split())[:300] if fatal else None
    end, fin, ex = f["end_time_s"], f["final_time_s"], f["exit"]
    reached = end is not None and fin is not None and fin >= end * (1 - 1e-6)
    # THE STEP-1 WATCHER (design 2026-10-02-case-free-shell-throat, rev. 3,
    # S7.3b/04 #2). watcher-stop.txt is written by clot_free_watcher.py, not
    # by Allrun, so it survives run-exit-code.txt and is checked BEFORE the
    # clock-budget/crash logic below: a watcher stop is never a clean
    # finish, whatever exit code and "End" line the solver itself left.
    watcher_stop = read(os.path.join(d, "watcher-stop.txt"), None)
    if watcher_stop is not None:
        f["watcher_stop"] = watcher_stop.strip()
        f["stop"] = f"WATCHER-STOPPED: {watcher_stop.strip().splitlines()[0]}"
    elif ex is not None and ex > 128:
        f["stop"] = f"CRASHED {SIGNALS.get(ex - 128, 'signal %d' % (ex - 128))} (exit {ex})"
    elif ex not in (0, None):
        f["stop"] = f"FAILED exit {ex}" + (" (FOAM FatalError)" if fatal else "")
    elif reached:
        f["stop"] = "COMPLETED"
    elif (f["clock_budget_s"] and f["clock_s"] is not None
          and f["clock_s"] >= f["clock_budget_s"] - CLOCK_SLACK_S):
        f["stop"] = (f"CLOCK BUDGET: stopAtClockTime {f['clock_budget_s']:.0f} s reached "
                     f"(ClockTime {f['clock_s']:.0f} s) before endTime")
    elif ex is None:
        f["stop"] = "UNKNOWN (no run-exit-code.txt)"
    else:
        f["stop"] = "STOPPED EARLY, UNEXPLAINED (exit 0, endTime not reached, no clock budget hit)"
    if end and fin is not None:
        f["done_pct"] = round(100.0 * fin / end, 1)
    return f


def digest_run(run):
    out = {"run": os.path.basename(os.path.normpath(run)), "cases": []}
    rr = glob.glob(os.path.join(run, "artifacts", "*", "runner_result.json"))
    if rr:
        r = json.loads(read(rr[0]))
        out["runner"] = {k: r.get(k) for k in ("package_sha256", "source", "tag", "repo_digest",
                                               "repo_digest_recomputed", "verdict", "series_filter",
                                               "compile_exit_code", "runner_exit_code")}
    for case in sorted(glob.glob(os.path.join(run, "case-after-run-*"))):
        if not os.path.isdir(case):
            continue
        res = json.loads(read(os.path.join(case, "postProcessing", "mainCase", "result.json"), "{}") or "{}")
        why = {e.get("child_id"): e.get("why") for e in res.get("inadmissible", []) or []}
        rows = []
        for d in sorted(glob.glob(os.path.join(case, "runs", "*"))):
            if not os.path.isfile(os.path.join(d, "child.json")):
                continue
            f = child_facts(d)
            if res:
                f["admissible"] = f["child"] not in why
                f["why"] = why.get(f["child"])
            rows.append(f)
        out["cases"].append({"case": os.path.basename(case),
                             "admissible": f"{res.get('children_admissible')}/{res.get('children_total')}" if res else None,
                             "children": rows})
    return out


def source_line(r):
    """L1: where the code came from. A github package's sha is built in the
    notebook and is no release id; the tag and the recomputed digest are."""
    if r.get("source") != "github":
        return f"{r.get('source') or 'zip (no source recorded)'}  package {r.get('package_sha256')}"
    d, rc = r.get("repo_digest"), r.get("repo_digest_recomputed")
    return (f"github  tag {r.get('tag')}  repo_digest {d}  recomputed {rc}  "
            + ("MATCH" if d and d == rc else "MISMATCH"))


def show(dg):
    r = dg.get("runner", {})
    print(f"RUN {dg['run']}  package {str(r.get('package_sha256'))[:12]}  verdict {r.get('verdict')}  "
          f"compile {r.get('compile_exit_code')}  filter: {r.get('series_filter')}")
    print(f"SOURCE {source_line(r)}")
    for c in dg["cases"]:
        print(f"CASE {c['case']}  assessor admissible {c['admissible']}")
        for f in c["children"]:
            print(f"  {f['child']:<28} {f['stop']}")
            print(f"  {'':<28} t {f.get('final_time_s')} / {f.get('end_time_s')} s ({f.get('done_pct')}%)"
                  f"  wall {f.get('wall_s')} s  clock budget {f.get('clock_budget_s')}")
            if "admissible" in f:
                print(f"  {'':<28} assessor: {'admissible' if f['admissible'] else 'INADMISSIBLE: ' + str(f.get('why'))[:220]}")
            if f.get("fatal"):
                print(f"  {'':<28} fatal: {f['fatal']}")
            for s in f.get("stack", []):
                print(f"  {'':<28} {s[:150]}")


def self_test():
    ok = []
    with tempfile.TemporaryDirectory() as t:
        def child(name, exit_code, times, clock, budget, end=0.005, extra=""):
            d = f"{t}/run/case-after-run-X/runs/{name}"
            os.makedirs(f"{d}/system")
            json.dump({"child_id": name, "end_time_s": end}, open(f"{d}/child.json", "w"))
            open(f"{d}/run-exit-code.txt", "w").write(f"{exit_code}\n")
            open(f"{d}/system/controlDict", "w").write(
                "functions { wallBudget { type stopAtClockTime; stopTime %d; } }\n" % budget)
            open(f"{d}/log.phaseClotAxisymmetricFluid", "w").write(
                "".join(f"Time = {x}\n\nExecutionTime = 1 s  ClockTime = {clock} s\n" for x in times) + extra)
        child("done", 0, [0.0025, 0.005], 100, 50400)
        child("clock", 0, [0.001, 0.00278], 50410, 50400)
        child("early", 0, [0.001, 0.00278], 3000, 50400)
        child("fpe", 136, [4.16e-06], 6, 50400, extra="#0  Foam::error::printStack(Foam::Ostream&)\n#1  Foam::sigFpe::sigHandler(int)\n#2  ? in libc\n#3  Foam::phaseClot::solve()\n")
        dg = digest_run(f"{t}/run")
        got = {f["child"]: f["stop"].split()[0] for f in dg["cases"][0]["children"]}
        want = {"done": "COMPLETED", "clock": "CLOCK", "early": "STOPPED", "fpe": "CRASHED"}
        for k, v in want.items():
            good = got.get(k) == v
            ok.append(good)
            print(f"{'ok  ' if good else 'FAIL'} {k}: {got.get(k)} (want {v})")
        fpe = [f for f in dg["cases"][0]["children"] if f["child"] == "fpe"][0]
        good = "SIGFPE" in fpe["stop"] and fpe["stack"] and "phaseClot::solve" in fpe["stack"][-1]
        ok.append(bool(good))
        print(f"{'ok  ' if good else 'FAIL'} fpe names SIGFPE and keeps the frames after sigHandler")
    print("SELF-TEST", "PASS" if all(ok) else "FAIL")
    return 0 if all(ok) else 1


def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    if not argv:
        print(__doc__)
        return 2
    dg = digest_run(argv[0])
    if "--json" in argv:
        print(json.dumps(dg, indent=1, default=str))
    else:
        show(dg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
