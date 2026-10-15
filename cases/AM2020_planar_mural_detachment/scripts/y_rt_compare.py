#!/usr/bin/env python3
"""Score the Y restart test ACROSS TWO SESSIONS (design 2026-10-14-case-symmetric-y-plug.md section 5 item 2; BATCH 80).

    y_rt_compare.py <Y80_RT_A case dir> <Y80_RT_B case dir> [--pair Y80_RT] [--controls constant/mainCaseControls.json]

RT_A runs in BATCH 80's launch 1 (every step written, stopped after step 105). RT_B's leg 1 runs in the same launch (stopped after step 100, not write-aligned); its leg 2 is
launch 2: `LAUNCH = 2` and `RESTART_FROM = <the launch-1 case-after-run tarball>` in the notebook, i.e. a second runner invocation with `--restart-from`, which stops after
step 105. The two sessions' returned cases are separate trees, so assess_case.py (which scores the pair when both arms are in ONE tree) reports ARMS_NOT_IN_THIS_RUN in each; this
script scores the pair from RT_A's directory of launch 1 and RT_B's directory of launch 2 with the SAME function (assess_case.y_rt_pair). Exit 0 only on PASS (RT-1a/b/c exactly 0
at 17 digits, every precondition met); 1 on FAIL or VOID; 2 on a usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="Y80_RT_A's case directory (launch 1)")
    ap.add_argument("b", type=Path, help="Y80_RT_B's case directory (launch 2, the restart)")
    ap.add_argument("--pair", default="Y80_RT")
    ap.add_argument("--controls", type=Path, default=HERE.parent / "constant" / "mainCaseControls.json")
    args = ap.parse_args(argv)
    try:
        controls = json.loads(args.controls.read_text())
        entry = next(e for e in controls["y_rt_pairs"] if e["pair_id"] == args.pair)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"y_rt_compare: no y_rt_pairs entry {args.pair!r} in {args.controls} ({error!r})", file=sys.stderr)
        return 2
    for d in (args.a, args.b):
        if not d.is_dir():
            print(f"y_rt_compare: {d} is not a directory", file=sys.stderr)
            return 2
    import assess_case
    res = assess_case.y_rt_pair(args.a.resolve(), args.b.resolve(), entry)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
