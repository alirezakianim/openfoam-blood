#!/usr/bin/env python3
"""Score the Y restart test ACROSS TWO SESSIONS (design 2026-10-14-case-symmetric-y-plug.md section 5 item 2; BATCH 80, and the confirming test BATCH 82).

    y_rt_compare.py <RT_A case dir> <RT_B case dir> [--pair Y80_RT | Y80_RT86] [--controls constant/mainCaseControls.json] [--environment <environment.json> ...]

BATCH 80: RT_A (Y80_RT_A) runs in launch 1 (every step written, stopped after step 105). RT_B's leg 1 runs in the same launch (stopped after step 100, not write-aligned); its leg 2 is
launch 2: `LAUNCH = 2` and `RESTART_FROM = <the launch-1 case-after-run tarball>` in the notebook, i.e. a second runner invocation with `--restart-from`, which stops after step 105.
BATCH 82 (the confirming test, pair Y80_RT86): Y80_RT86_B's leg 1 stops after step 86 (launch 1) and its leg 2 after step 105 (launch 2), compared against the EXISTING Y80_RT_A of BATCH 80,
whose log runs steps 1-105 and whose last 10 writes (96-105) are kept: the compared window is steps 87-105 and RT-1a's reference write is the pair's `a_reference_step` (96).
The two sessions' returned cases are separate trees, so assess_case.py (which scores the pair when both arms are in ONE tree) reports ARMS_NOT_IN_THIS_RUN in each; this
script scores the pair with the SAME function (assess_case.y_rt_pair).

THE WINDOW IS READ FROM THE CASES. The pair is chosen by RT_B's child.json child_id (--pair, when given, must agree) and the window (leg1, leg2) is RT_B's child.json stop_after_step;
the controls' pair must register the same window (else exit 2: the controls are not the ones that built the case), and y_rt_pair cross-checks both cases' child.json again (VOID).
The pair's optional a_reference_step is the controls'.

THE CPU CONTROL (docket b80-rt-restart DECISIONS 2026-10-15): the result is VOID unless every environment.json given (the run roots of RT_A and RT_B are found by walking up from each case
dir; `--environment` adds more, e.g. the launch-1 run of RT_B's leg 1) names the same s22.cpu_model and s22.cpu_flags. With none found, `cpu_identity` is NOT_CHECKED in the output
(exit status is unaffected: a synthetic pair has no environment); the CE reads that line before calling the result ESTABLISHED.

Exit 0 only on PASS (RT-1a/b/c exactly 0 at 17 digits, every precondition met); 1 on FAIL or VOID; 2 on a usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _child_json(case: Path) -> dict:
    try:
        d = json.loads((case / "child.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{case}/child.json is unreadable ({error})")
    if not isinstance(d, dict):
        raise ValueError(f"{case}/child.json is not an object")
    return d


def _environment_of(case: Path):
    """The nearest environment.json at or above a returned case directory (preflight/<run>/environment.json), or None."""
    for d in [case, *case.parents][:6]:
        f = d / "environment.json"
        if f.is_file():
            return f
    return None


def cpu_identity(paths: list) -> dict:
    seen = {}
    for p in paths:
        e = json.loads(Path(p).read_text())
        s22 = e.get("s22", e)
        seen[str(p)] = {"cpu_model": s22.get("cpu_model"), "cpu_flags": s22.get("cpu_flags")}
    kinds = {json.dumps(v, sort_keys=True) for v in seen.values()}
    if not seen:
        return {"status": "NOT_CHECKED", "reason": "no environment.json found for either case and none given"}
    if any(v["cpu_model"] is None or v["cpu_flags"] is None for v in seen.values()):
        return {"status": "VOID", "reason": "an environment.json holds no s22.cpu_model / s22.cpu_flags", "environments": seen}
    return {"status": "SAME" if len(kinds) == 1 else "VOID", "reason": None if len(kinds) == 1 else "the CPU model or flags differ between the sessions", "environments": seen}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="the reference arm's case directory (Y80_RT_A, BATCH 80 launch 1)")
    ap.add_argument("b", type=Path, help="the restarted arm's case directory (Y80_RT_B or Y80_RT86_B, the launch that restarted: leg 2)")
    ap.add_argument("--pair", default=None, help="the y_rt_pairs pair_id; default: the pair whose b is RT_B's child.json child_id")
    ap.add_argument("--controls", type=Path, default=HERE.parent / "constant" / "mainCaseControls.json")
    ap.add_argument("--environment", type=Path, action="append", default=[], help="an extra environment.json whose CPU must equal the cases' (repeatable)")
    args = ap.parse_args(argv)
    for d in (args.a, args.b):
        if not d.is_dir():
            print(f"y_rt_compare: {d} is not a directory", file=sys.stderr)
            return 2
    try:
        controls = json.loads(args.controls.read_text())
        pairs = controls["y_rt_pairs"]
        cj_b = _child_json(args.b)
        legs = cj_b.get("stop_after_step")
        if not (isinstance(legs, dict) and set(legs) == {"leg1", "leg2"} and all(isinstance(legs[k], int) and not isinstance(legs[k], bool) for k in legs)):
            raise ValueError(f"RT_B's child.json stop_after_step {legs!r} is not {{leg1: int, leg2: int}}: this is not a restart-test B case")
        cands = [e for e in pairs if e["b"] == cj_b.get("child_id")]
        if args.pair is not None:
            cands = [e for e in cands if e["pair_id"] == args.pair]
        if len(cands) != 1:
            raise ValueError(f"{len(cands)} y_rt_pairs entries have b == {cj_b.get('child_id')!r}" + (f" and pair_id {args.pair!r}" if args.pair else "") + f" (pairs: {[e['pair_id'] for e in pairs]})")
        entry = dict(cands[0])
        if (entry["leg1_stop"], entry["leg2_stop"]) != (legs["leg1"], legs["leg2"]):
            raise ValueError(f"the controls' pair {entry['pair_id']} registers the window ({entry['leg1_stop']}, {entry['leg2_stop']}) but RT_B's child.json says ({legs['leg1']}, {legs['leg2']}): "
                             "the controls are not the ones that built this case")
        # the window below is the CASE's (equal to the controls' by the check above); a_reference_step is the controls'
        entry["leg1_stop"], entry["leg2_stop"] = legs["leg1"], legs["leg2"]
    except (OSError, ValueError, KeyError) as error:
        print(f"y_rt_compare: {error}", file=sys.stderr)
        return 2
    import assess_case
    res = assess_case.y_rt_pair(args.a.resolve(), args.b.resolve(), entry)
    envs = [p for p in (_environment_of(args.a.resolve()), _environment_of(args.b.resolve())) if p is not None] + list(args.environment)
    try:
        cpu = cpu_identity(sorted({str(p) for p in envs}))
    except (OSError, ValueError) as error:
        cpu = {"status": "VOID", "reason": f"an environment.json is unreadable ({error})"}
    res["cpu_identity"] = cpu
    if cpu["status"] == "VOID":
        res.setdefault("void_reasons", []).append(f"CPU identity: {cpu['reason']}")
        res["status"] = "VOID"
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
