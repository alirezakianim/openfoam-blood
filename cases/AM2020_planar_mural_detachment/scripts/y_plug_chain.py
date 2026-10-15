#!/usr/bin/env python3
"""Score a Y81 plug CHAIN across its restarted sessions (design 2026-10-14-case-symmetric-y-plug.md section 6 B; BATCH 81; case-engineer 28).

    y_plug_chain.py <session-1 child dir> [<session-2 child dir> ...] [--controls constant/mainCaseControls.json] [--json OUT]

Each argument is the returned case of ONE session of ONE child (the runs/Y81_*_PLUG directory of that session's case-after-run tree), in session order. A Y81 child ends each
Colab session at its wall budget (a writeNow) and the next session restarts from that write (the notebook's RESTART_FROM = the previous session's case-after-run tarball), so no
session holds the whole run: session k holds 0/ (the initial state), the restart point it copied, and its own writes. This script reduces every session's full writes
(scripts/y_gate/y_plug_assess.py reduce_case), MERGES them in time order and scores the chain with the SAME analyse() the in-session assessor calls.

REFUSED (exit 1, status CHAIN_REFUSED, with the reason): a session that is not a restart of the previous one; a restart point that is not the previous session's last write; a write that
appears twice or two writes closer than half an interval (overlapping sessions); a gap of more than 1.5 write intervals; dictionaries that differ between sessions (one pinned package
per chain). Exit codes: 0 a final verdict (STUCK / SQUEEZE / INDETERMINATE-final); 1 the chain is refused or cannot be assessed; 3 the chain is not yet readable or its INDETERMINATE is
not yet final (run the next session); 2 a usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "y_gate"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", type=Path, nargs="+", help="each session's child case directory, in session order")
    ap.add_argument("--controls", type=Path, default=HERE.parent / "constant" / "mainCaseControls.json")
    ap.add_argument("--json", type=Path, default=None, help="also write the full result (with the per-write series) here")
    args = ap.parse_args(argv)
    for d in args.sessions:
        if not d.is_dir():
            print(f"y_plug_chain: {d} is not a directory", file=sys.stderr)
            return 2
    try:
        controls = json.loads(args.controls.read_text())
        spec = json.loads((args.sessions[0] / "child.json").read_text())
    except (OSError, ValueError) as error:
        print(f"y_plug_chain: cannot read the controls or the first session's child.json ({error!r})", file=sys.stderr)
        return 2
    import y_plug_assess
    from assess_case import s25_report                                    # F3 is read by the same function the in-session assessor uses
    texts = []
    for d in args.sessions:
        log = d / "log.phaseClotAxisymmetricFluid"
        texts.append(log.read_text(errors="replace") if log.is_file() else "")
    params = y_plug_assess.load_params(controls)
    res = y_plug_assess.assess_case_dirs([d.resolve() for d in args.sessions], spec, params, texts=texts, labels=[f"session {k + 1} ({d.parent.parent.name})" for k, d in enumerate(args.sessions)], s25_report=s25_report)
    cols = res.pop("series_columns", None)
    print(json.dumps(res, indent=2, default=str))
    if args.json:
        args.json.write_text(json.dumps(dict(res, series_columns=cols), indent=1, allow_nan=False) + "\n")
    if res.get("status") == "VERDICT" and res.get("final"):
        return 0
    if res.get("status") in ("NOT_READABLE_YET",) or (res.get("status") == "VERDICT" and not res.get("final")):
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
