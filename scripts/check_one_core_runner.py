#!/usr/bin/env python3
"""S25: the runner's two new launch-path pieces, offline: the MKL_CBWR pin probe (1-core batches) and the
restart staging (--restart-from).

The probe runs the solver's own CNR call order (blockCoupledKernel.H L1610: mkl_cbwr_set(MKL_CBWR_AUTO) before
any other MKL call) against a libmkl_rt and decides whether an MKL_CBWR environment pin survives it. There is no
MKL on this machine, so three FAKE libraries (compiled here from C) stand in for the three outcomes the probe must
tell apart; each is run through the probe's real subprocess path and its verdict checked, and each verdict's
launch consequence checked. This proves the probe's plumbing and decision logic. It does NOT prove what the real
libmkl_rt does: that is measured by the probe itself on Colab and recorded in environment.json (s22.mkl_cbwr).

Exit 0 when every check passes. Run: python3 -B scripts/check_one_core_runner.py
"""

from __future__ import annotations

import json

import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import check_notebook_batch_selector as nbsel   # noqa: E402

CASE_SCRIPTS = ROOT / "cases" / "AM2020_planar_mural_detachment" / "scripts"
CONTROLS = ROOT / "cases" / "AM2020_planar_mural_detachment" / "constant" / "mainCaseControls.json"

# A fake libmkl_rt. First MKL_CBWR_Get/Set call fixes the mode from the environment, as MKL does. Branch codes are the
# fake's own (3 = COMPATIBLE, 10 = the pin, 11 = what AUTO picks on this fake CPU).
FAKE = r"""
#include <stdlib.h>
#include <string.h>
static int branch = -1;
static void init(void) {
    if (branch >= 0) return;
    const char* e = getenv("MKL_CBWR");
    if (!e) branch = 0;
    else if (strstr(e, "COMPATIBLE")) branch = 3;
    else branch = 10;
}
int MKL_CBWR_Get(int what) { init(); return branch == 0 ? 11 : branch; }
int MKL_CBWR_Set(int mode) {
    init();
    if (mode != 2) return -1;
#if defined(HONORS)
    if (branch == 0) branch = 11;          /* AUTO applies only when no pin is in force */
    return 0;
#elif defined(OVERRIDES)
    branch = 11;                            /* AUTO replaces whatever the environment said */
    return 0;
#else
    return branch == 0 ? 0 : -3;            /* a pinned mode cannot be changed: MKL_CBWR_ERR_MODE_CHANGE_FAILURE */
#endif
}
"""


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_fakes(tmp: Path) -> dict:
    src = tmp / "fake_mkl.c"
    src.write_text(FAKE)
    out = {}
    for name, flag in (("honors", "-DHONORS"), ("overrides", "-DOVERRIDES"), ("fails", "-DFAILS")):
        lib = tmp / f"libfake_{name}.dylib"
        done = subprocess.run(["cc", "-shared", "-fPIC", flag, str(src), "-o", str(lib)], capture_output=True, text=True)
        assert done.returncode == 0, f"cc failed for the {name} fake: {done.stderr[-300:]}"
        out[name] = str(lib)
    return out


def main() -> int:
    ns = nbsel.runner_namespace()
    probe, refusal = ns["s25_cbwr_probe"], ns["s25_cbwr_refusal"]
    with tempfile.TemporaryDirectory(prefix="one-core-runner-check-") as t:
        tmp = Path(t)
        fakes = build_fakes(tmp)
        print("the MKL_CBWR pin probe, three outcomes (fake libmkl_rt):")
        want = {"honors": "held", "overrides": "overridden", "fails": "set_failed"}
        for name, verdict in want.items():
            r = probe(fakes[name], "AVX2,STRICT")
            assert r["verdict"] == verdict, (name, r)
            print(f"  {name:<10} -> {r['verdict']:<11} pin_only {r['pin_only']['branch']}, pin_then_auto "
                  f"{r['pin_then_auto']['branch']} (set rc {r['pin_then_auto']['set_auto_rc']}), auto_only {r['auto_only']['branch']}")
            msg = refusal("AVX2,STRICT", r)
            assert (msg is not None) == (verdict != "held"), (name, msg)
        r = probe(str(tmp / "no_such_lib.dylib"), "AVX2,STRICT")
        assert r["verdict"] == "not_run" and refusal("AVX2,STRICT", r) is None, r
        print("  a library that cannot be opened -> not_run, recorded, no refusal (the probe's own failure never stops a session)")
        assert "overridden" in refusal("AVX2,STRICT", {"verdict": "overridden"}) and "refused after" in refusal("AVX2,STRICT", {"verdict": "set_failed"})
        print("  only a MEASURED override or failure refuses the launch")

        print("restart staging (--restart-from):")
        rd = load(CASE_SCRIPTS / "restart_driver.py", "restart_driver_for_runner_check")
        chk = load(ROOT / "scripts" / "check_restart_driver.py", "check_restart_driver_helpers")
        controls_kids = ns["selected_child_ids"]("", CONTROLS)
        # Derived, not pinned (CE 2026-10-10): an empty filter selects EVERY child in the controls, in order.
        _all_ids = [c["child_id"] for c in json.loads(CONTROLS.read_text())["children"]]
        assert controls_kids == _all_ids and len(_all_ids) == len(set(_all_ids)), (len(controls_kids), len(_all_ids))
        assert ns["selected_child_ids"]("S23v_G4_r3", CONTROLS) == ["S23v_G4_r3"]
        d = tmp / "prev"
        for child in ("P73_G4_r3_40", "D75_refI"):
            c = d / child
            chk.make_case(tmp / f"mk_{child}")
            shutil.copytree(tmp / f"mk_{child}" / "case", c)
            chk.make_frame(c, "0.0002")
            chk.make_full(c, "0.01", 3)
            chk.make_frame(c, "0.0102")
        os.environ.pop("AM2020_RESTART_FROM", None)
        rec = ns["stage_restart"](str(d), ["P73_G4_r3_40", "D75_refI"], CASE_SCRIPTS / "restart_driver.py", tmp / "dest")
        assert rec["chosen"] == {"P73_G4_r3_40": "0.01", "D75_refI": "0.01"} and os.environ["AM2020_RESTART_FROM"] == str(d)
        print("  a directory source: both children restart from 0.01 (a newer frame 0.0102 is passed over); AM2020_RESTART_FROM exported")
        shutil.rmtree(d / "D75_refI" / "0.01")
        try:
            ns["stage_restart"](str(d), ["P73_G4_r3_40", "D75_refI"], CASE_SCRIPTS / "restart_driver.py", tmp / "dest")
            raise AssertionError("a child with only frames was accepted")
        except RuntimeError as error:
            assert "no full intact write" in str(error) and "0.0002" in str(error), error
        print("  a child with only frames is refused at the start of the session, naming its directories")
        try:
            ns["stage_restart"](str(d), ["P73_G4_r3_40", "S23v_G4_r3"], CASE_SCRIPTS / "restart_driver.py", tmp / "dest")
            raise AssertionError("a child absent from the source was accepted")
        except RuntimeError as error:
            assert "holds no directory for child 'S23v_G4_r3'" in str(error), error
        print("  a selected child absent from the source is refused")
        tar = tmp / "case-after-run-X.tar.gz"
        (tmp / "pk" / "case-after-run-X" / "runs").mkdir(parents=True)
        shutil.copytree(tmp / "prev" / "P73_G4_r3_40", tmp / "pk" / "case-after-run-X" / "runs" / "P73_G4_r3_40")
        with tarfile.open(tar, "w:gz") as tf:
            tf.add(tmp / "pk" / "case-after-run-X", arcname="case-after-run-X")
        rec = ns["stage_restart"](str(tar), ["P73_G4_r3_40"], CASE_SCRIPTS / "restart_driver.py", tmp / "dest_tar")
        assert rec["chosen"] == {"P73_G4_r3_40": "0.01"} and rec["extracted"]["children"] == {"P73_G4_r3_40": ["0.01"]}
        print("  a case-after-run tarball: extracted (full writes + dictionaries only), restarts from 0.01")
        try:
            ns["stage_restart"](str(tmp / "nothing"), ["P73_G4_r3_40"], CASE_SCRIPTS / "restart_driver.py", tmp / "d")
            raise AssertionError("a missing source was accepted")
        except RuntimeError as error:
            assert "neither a tarball nor a directory" in str(error)
        print("  a source that does not exist is refused")
        os.environ.pop("AM2020_RESTART_FROM", None)

        print("the runner's argparse takes the new flags:")
        argv = ["python3", "run_main_case_colab.py", "--run-id", "x"] + nbsel.FAKE_BASE + [
            "--only", "P73_G4_r3_40", "--mkl-threads", "1", "--mkl-cbwr", "AVX2,STRICT", "--restart-from", "/content/drive/prev.tar.gz"]
        args = nbsel.runner_accepts(argv, CONTROLS)
        assert (args.mkl_threads, args.mkl_cbwr, args.restart_from) == (1, "AVX2,STRICT", "/content/drive/prev.tar.gz"), args
        for bad in (["--mkl-threads", "-1"], ["--mkl-cbwr", "AVX2 STRICT"]):
            try:
                nbsel.runner_accepts(["python3", "run_main_case_colab.py", "--run-id", "x"] + nbsel.FAKE_BASE + ["--only", "P73_G4_r3_40"] + bad, CONTROLS)
                raise AssertionError(f"{bad} was accepted")
            except nbsel.LaunchError as error:
                assert "argparse rejects" in str(error), error
        print("  --mkl-threads 1 --mkl-cbwr AVX2,STRICT --restart-from parse; a negative thread count and a CBWR with a space are rejected")

        print("S25 section 12.1, the CPU identity a session records (no pin is requested, so SAME or VOID rests on these):")
        ident = ns["s25_cpu_identity"]
        two_cpus = "processor\t: 0\nmodel name\t: X\nflags\t\t: sse2 avx2 fma\n\nprocessor\t: 1\nflags\t\t: sse2 avx512f\n"
        r = ident(two_cpus, fakes["honors"], record_auto=True)
        assert r["cpu_flags"] == "avx2 fma sse2", r      # the FIRST cpu's flags, sorted
        assert r["auto_branch"] == {"set_auto_rc": 0, "branch": 11}, r      # the fake's AUTO branch, from a fresh process, no MKL_CBWR
        assert ident(two_cpus, fakes["honors"], record_auto=False) == {"cpu_flags": "avx2 fma sse2"}
        assert ident("no flags here\n", None, record_auto=False) == {"cpu_flags": None}
        r = ident("", None, record_auto=True)
        assert r["cpu_flags"] is None and "no libmkl_rt" in r["auto_branch"]["error"], r
        r = ident("", str(tmp / "no_such_lib.dylib"), record_auto=True)
        assert "error" in r["auto_branch"], r      # recorded, never raised
        os.environ["MKL_CBWR"] = "AVX2,STRICT"     # an inherited pin must not leak into the AUTO measurement
        assert ident("", fakes["honors"], record_auto=True)["auto_branch"]["branch"] == 11
        os.environ.pop("MKL_CBWR")
        print("  flags of the first CPU, sorted; the AUTO branch measured in a fresh process without MKL_CBWR; a missing library or a failed probe is recorded, not raised")
        # the wiring in s22_mkl_environment itself (pip and lscpu stubbed; no libmkl_rt on this host, so the probe records its "no libmkl_rt" error)
        import types
        real_run, real_cores = ns["subprocess"].run, ns["s22_physical_cores"]
        ok = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        ns["subprocess"].run = lambda *a, **k: ok
        ns["s22_physical_cores"] = lambda: (1, "0,0\n")
        saved_env = {k: os.environ.get(k) for k in ("MKL_CBWR", "MKL_NUM_THREADS", "AM2020_CONCURRENCY", "PHASE_CLOT_MKL_RT")}
        try:
            os.environ.pop("MKL_CBWR", None)
            os.environ["AM2020_CONCURRENCY"] = "1"
            one = ns["s22_mkl_environment"](1, "")
            assert "cpu_flags" in one and "no libmkl_rt" in one["mkl_cbwr"]["auto_branch"]["error"] and one["mkl_cbwr"]["requested"] is None, one["mkl_cbwr"]
            many = ns["s22_mkl_environment"](0, "")
            assert "auto_branch" not in many["mkl_cbwr"] and "cpu_flags" in many, many["mkl_cbwr"]      # the 6-thread batches record what they always did
            pinned = ns["s22_mkl_environment"](1, "AVX2,STRICT")
            assert "auto_branch" not in pinned["mkl_cbwr"] and pinned["mkl_cbwr"]["requested"] == "AVX2,STRICT", pinned["mkl_cbwr"]    # a pin keeps its own probe
        finally:
            ns["subprocess"].run, ns["s22_physical_cores"] = real_run, real_cores
            for k, v in saved_env.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        print("  s22_mkl_environment: the 1-core unpinned session records cpu_flags and mkl_cbwr.auto_branch; the 6-thread and the pinned sessions do not record an auto_branch")

        print("the baseline reference library (solver_library baseline, B76_OFF_A):")
        assert ns["REFERENCE_LIBRARIES"]["baseline"][0] == "PHASE_CLOT_BASELINE_LIBBIN" and ns["REFERENCE_LIBRARIES"]["baseline"][4] == "baseline_solver"
        base = ["python3", "run_main_case_colab.py", "--run-id", "x"] + nbsel.FAKE_BASE + ["--mkl-threads", "1"]
        args = nbsel.runner_accepts(base + ["--only", "B76_OFF_A", "--baseline-solver", nbsel.FAKE_SOLVER], CONTROLS)
        assert args.baseline_solver == nbsel.FAKE_SOLVER and args.mkl_threads == 1 and args.mkl_cbwr == "", args      # no CNR pin (S25 12.1)
        for argv_, words in ((base + ["--only", "B76_OFF_A"], "--baseline-solver was not passed"),
                             (base + ["--only", "B76_OFF_B", "--baseline-solver", nbsel.FAKE_SOLVER], "built for nothing")):
            try:
                nbsel.runner_accepts(argv_, CONTROLS)
                raise AssertionError(f"accepted: {argv_[-4:]}")
            except nbsel.LaunchError as error:
                assert words in str(error), (words, error)
        nbsel.runner_accepts(base + ["--only", "B76_OFF_B"], CONTROLS)
        print("  B76_OFF_A with --baseline-solver parses (and carries no --mkl-cbwr); without it the session is refused at its start; B76_OFF_B with it is refused as built for nothing")
    print("ONE-CORE RUNNER OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
