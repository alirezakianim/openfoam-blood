#!/usr/bin/env python3
"""Score the Y restart test ACROSS TWO SESSIONS, with the CPU identity of ALL THREE sessions required (design 2026-10-14-case-symmetric-y-plug.md section 5 item 2; BATCH 80, the confirming test BATCH 82, and its corrected retest BATCH 83).

    y_rt_compare.py <RT_A case dir> <RT_B case dir> [--pair Y80_RT | Y80_RT86] [--controls constant/mainCaseControls.json]
                    ( --env-a F | --run-a D )  ( --env-b1 F | --run-b1 D )  ( --env-b2 F | --run-b2 D )

THE THREE SESSIONS. A = the session that ran the continuous arm (RT_A, every step written); B1 = the session that ran RT_B's leg 1 (stopped after step `leg1`, not write-aligned); B2 = the session
that ran RT_B's leg 2 (a second runner invocation, `--restart-from` the leg-1 tarball). In BATCH 83 launch 1 runs RT_A and RT86_B's leg 1 together, so A and B1 are ONE session and the same
environment.json is given for both (`--env-a` and `--env-b1`, or `--run-a` and `--run-b1` naming the same run directory); launch 2 is B2. Each is given as the environment.json itself (`--env-*`) or as the
returned run directory that holds it (`--run-*` = <run>/environment.json); both for one role is a usage error (exit 2). Nothing is discovered: an identity not named here is MISSING.

THE RESULT IS VOID, WITH THE REASON, and exit 1, unless all three identities are present, readable, hold s22.cpu_model and s22.cpu_flags, and are EQUAL. `cpu_identity.status` is therefore SAME only on 3 of 3 equal;
there is no NOT_CHECKED. Docket b80-rt-restart DECISIONS 2026-10-16: BATCH 82 was scored on two environments (A's and leg 2's, the CE copied) and read SAME while leg 1 had run on an AMD EPYC 7B12 and
the other two on an Intel Xeon: a 2-of-3 check passed. Three more refusals, each VOID: one file (same path, or the same bytes) standing for B1 and B2, or for A and B2, is one session where two are needed;
B2's environment.json records no restart (leg 2 is a restart session); and, when B1 is given as a stamp-named run directory, B2's restart_from does not name that run (B2 did not restart from this B1).
`a_and_b1_one_session` reports whether A and B1 are the same file (BATCH 83 intends it; the CPU equality is the control either way).

THE WINDOW IS READ FROM THE CASES. The pair is chosen by RT_B's child.json child_id (--pair, when given, must agree) and the window (leg1, leg2) is RT_B's child.json stop_after_step; the controls' pair must
register the same window (else exit 2: the controls are not the ones that built the case), and y_rt_pair cross-checks both cases' child.json again (VOID). The pair's optional a_reference_step is the controls'.
The two sessions' returned cases are separate trees, so assess_case.py (which scores the pair when both arms and leg 2 are in ONE tree) reports ARMS_NOT_IN_THIS_RUN in each; this script scores the pair with the
SAME function (assess_case.y_rt_pair).

Exit 0 only on PASS (RT-1a/b/c exactly 0 at 17 digits, every precondition met, three equal CPU identities); 1 on FAIL or VOID; 2 on a usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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


ROLES = ("a", "b1", "b2")
ROLE_NAMES = {"a": "A (the continuous arm)", "b1": "B leg 1", "b2": "B leg 2"}
RUN_DIR = re.compile(r"^\d{8}-\d{6}-am2020-main-case$")


def _identity_of(role: str, env_path, run_dir):
    """The environment.json that stands for one session: (path or None, usage error or None)."""
    if env_path is not None and run_dir is not None:
        return None, f"--env-{role} and --run-{role} both given: name the session once"
    if env_path is not None:
        return Path(env_path), None
    if run_dir is not None:
        return Path(run_dir) / "environment.json", None
    return None, None


def cpu_identity(paths: dict, b2_restart_check: bool = True) -> dict:
    """paths: {"a": Path|None, "b1": Path|None, "b2": Path|None}. SAME only when all three sessions are present, readable, name a cpu_model and cpu_flags, and are equal;
    VOID (with every reason) otherwise. Never NOT_CHECKED: fewer than three identities is no control."""
    reasons, seen, raw = [], {}, {}
    for role in ROLES:
        path = paths.get(role)
        if path is None:
            reasons.append(f"the CPU identity of {ROLE_NAMES[role]} is MISSING (give --env-{role} or --run-{role}); the control needs all three sessions")
            continue
        try:
            e = json.loads(Path(path).read_text())
            if not isinstance(e, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as error:
            reasons.append(f"the environment.json of {ROLE_NAMES[role]} ({path}) is unreadable ({error})")
            continue
        s22 = e.get("s22", e) if isinstance(e.get("s22", e), dict) else {}
        rec = {"path": str(path), "cpu_model": s22.get("cpu_model"), "cpu_flags": s22.get("cpu_flags")}
        if rec["cpu_model"] is None or rec["cpu_flags"] is None:
            reasons.append(f"the environment.json of {ROLE_NAMES[role]} ({path}) holds no s22.cpu_model / s22.cpu_flags")
            continue
        seen[role], raw[role] = rec, e
    if len(seen) == 3:
        kinds = {json.dumps([v["cpu_model"], v["cpu_flags"]]) for v in seen.values()}
        if len(kinds) != 1:
            diff = {r: v["cpu_model"] for r, v in seen.items()}
            reasons.append(f"the CPU model or flags differ between the sessions (cpu_model: {diff})")
    # one file for two sessions is one session: B1 and B2 (or A and B2) are two different launches by construction. A and B1 are ONE session in BATCH 83 (the same file is expected).
    def same_file(r1, r2):
        if r1 not in seen or r2 not in seen:
            return False
        a, b = Path(seen[r1]["path"]), Path(seen[r2]["path"])
        return a.resolve() == b.resolve() or a.read_bytes() == b.read_bytes()
    for r1, r2 in (("b1", "b2"), ("a", "b2")):
        if same_file(r1, r2):
            reasons.append(f"one environment.json stands for {ROLE_NAMES[r1]} and {ROLE_NAMES[r2]} ({seen[r1]['path']} and {seen[r2]['path']}): they are two sessions, two files")
    prov = {"status": "NOT_CHECKED", "reason": "B1's environment.json is not in a stamp-named run directory, or B1/B2 is unreadable"}
    if b2_restart_check and "b2" in raw and "b1" in seen:
        rs = raw["b2"].get("restart")
        b1_run = Path(seen["b1"]["path"]).resolve().parent.name
        if not isinstance(rs, dict) or not rs.get("restart_from"):
            prov = {"status": "VOID", "reason": "B leg 2's environment.json records no restart (restart is null): leg 2 is a restart session"}
            reasons.append(prov["reason"])
        elif RUN_DIR.match(b1_run):
            ok = b1_run in str(rs["restart_from"]).split("/")
            prov = {"status": "MATCH" if ok else "VOID", "restart_from": rs["restart_from"], "b1_run": b1_run,
                    "reason": None if ok else f"B leg 2 restarted from {rs['restart_from']}, which is not the run {b1_run} given as B leg 1"}
            if not ok:
                reasons.append(prov["reason"])
        else:
            prov = {"status": "NOT_CHECKED", "restart_from": rs["restart_from"], "reason": f"B1's environment.json sits in {b1_run!r}, not a <stamp>-am2020-main-case run directory"}
    out = {"status": "VOID" if reasons else "SAME", "reason": "; ".join(reasons) if reasons else None, "restart_provenance": prov,
           "a_and_b1_one_session": "a" in seen and "b1" in seen and Path(seen["a"]["path"]).resolve() == Path(seen["b1"]["path"]).resolve(),
           "environments": {r: {"path": v["path"], "cpu_model": v["cpu_model"], "cpu_flags_sha256": hashlib.sha256(str(v["cpu_flags"]).encode()).hexdigest()[:16], "n_cpu_flags": len(str(v["cpu_flags"]).split())}
                            for r, v in seen.items()}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="the continuous arm's case directory (Y80_RT_A)")
    ap.add_argument("b", type=Path, help="the restarted arm's case directory (Y80_RT_B or Y80_RT86_B, the launch that restarted: leg 2)")
    ap.add_argument("--pair", default=None, help="the y_rt_pairs pair_id; default: the pair whose b is RT_B's child.json child_id")
    ap.add_argument("--controls", type=Path, default=HERE.parent / "constant" / "mainCaseControls.json")
    for role, who in (("a", "the continuous arm RT_A"), ("b1", "RT_B's leg 1"), ("b2", "RT_B's leg 2")):
        ap.add_argument(f"--env-{role}", type=Path, default=None, help=f"the environment.json of the session that ran {who}")
        ap.add_argument(f"--run-{role}", type=Path, default=None, help=f"the returned run directory of the session that ran {who} (its environment.json)")
    ap.add_argument("--environment", type=Path, action="append", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.environment is not None:
        print("y_rt_compare: --environment is replaced by --env-a / --env-b1 / --env-b2 (or --run-a / --run-b1 / --run-b2): the CPU identity of ALL THREE sessions is required "
              "(docket b80-rt-restart DECISIONS 2026-10-16: a 2-of-3 check read SAME)", file=sys.stderr)
        return 2
    sessions = {}
    for role in ROLES:
        sessions[role], usage = _identity_of(role, getattr(args, f"env_{role}"), getattr(args, f"run_{role}"))
        if usage:
            print(f"y_rt_compare: {usage}", file=sys.stderr)
            return 2
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
    cpu = cpu_identity(sessions)
    res["cpu_identity"] = cpu
    if cpu["status"] == "VOID":
        res.setdefault("void_reasons", []).append(f"CPU identity: {cpu['reason']}")
        res["status"] = "VOID"
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
