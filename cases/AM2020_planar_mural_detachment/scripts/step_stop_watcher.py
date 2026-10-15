#!/usr/bin/env python3
"""S25 B76-RT (RT-1): stop a child after a given solver step, by touching the file its `stopAfterStep` function object watches.

WHY A WATCHER. RT-1's A and B must run to step S = N0 + 5 exactly, and dt is adaptive (that is what RT-1b tests), so no endTime
reaches step S reliably. OF-13's stopAtFile function object (the route clot_free_watcher.py uses, proven in returned runs) stops a
run when <case>/<its own name> exists; its action writeNow makes the NEXT step the last, and that step is written
(Time.C L898-920, L1252-1256; measured on fs_G4_kxu_v250: "stopAtFile ... stopping now" logged at timeIndex 7865, step 7866 ran and was
written).

WHAT IT DOES. `step_stop_watcher.py CASE`
  * reads CASE/child.json `stop_after_step` = S (an integer >= 2; or {leg1, leg2} of the Y restart test: leg 2 when CASE/restart.json exists);
  * tails CASE/log.phaseClotAxisymmetricFluid, complete lines only. PHASE_CLOT_DT is printed at the START of each step with the ABSOLUTE
    time index (solver .C L13170: `step=` runTime.timeIndex(), after runTime++), so a restart's first record is step N0 + 1;
  * at the first record with step >= S - 1 it creates CASE/stopAfterStep and writes CASE/step-stop.txt. The solver is then inside step
    S - 1, which takes seconds; the function object runs at the start of the next iteration, so step S is the last and is written;
  * a record with step > S - 1 on its first sight (the watcher started late) is touched all the same and noted `late`;
  * exits 0 after the touch; exits 2 on a malformed child.json; gives up after --max-wait-s (default 3600) so it cannot outlive its case.
A late watcher costs steps beyond S, never steps before it: every RT-1 gate reads steps by index, and purgeWrite keeps the last 10 writes.

Standard library only (it runs under Allrun's plain python3).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

LOG_NAME = "log.phaseClotAxisymmetricFluid"
STOP_FILE = "stopAfterStep"      # the function object's own name: stopAtFile.C L81
REASON_FILE = "step-stop.txt"
RECORD = re.compile(r"^PHASE_CLOT_DT step=(\d+) ")


class Tail:
    """Feed it text as it arrives; it returns the step numbers of the COMPLETE PHASE_CLOT_DT lines only."""

    def __init__(self) -> None:
        self.partial = ""

    def feed(self, text: str) -> list[int]:
        self.partial += text
        *lines, self.partial = self.partial.split("\n")
        return [int(m.group(1)) for m in (RECORD.match(ln) for ln in lines) if m]


def stop_step_of(case: Path) -> int:
    try:
        spec = json.loads((case / "child.json").read_text())
        s = spec["stop_after_step"]
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"step_stop_watcher: {case / 'child.json'} needs stop_after_step ({error!r})") from error
    if isinstance(s, dict):
        # THE Y RESTART TEST (design 2026-10-14-case-symmetric-y-plug.md section 5 item 2): S PER LEG, {leg1, leg2}. restart_driver compares the dictionaries but not child.json, so
        # the two legs run under identical dictionaries and differ only here. Leg 2 is the case that holds restart.json (restart_driver --apply writes it before this watcher starts).
        if set(s) != {"leg1", "leg2"}:
            raise SystemExit(f"step_stop_watcher: stop_after_step {s!r} must be an integer or exactly {{leg1, leg2}}")
        s = s["leg2"] if (case / "restart.json").is_file() else s["leg1"]
    if isinstance(s, bool) or not isinstance(s, int) or s < 2:
        raise SystemExit(f"step_stop_watcher: stop_after_step {s!r} must be an integer >= 2")
    return s


def touch(case: Path, stop: int, step: int, first_sight: bool) -> None:
    (case / REASON_FILE).write_text(
        f"STEP-STOP: touched {STOP_FILE} at PHASE_CLOT_DT step={step} (stop_after_step {stop}; the last step run is {stop} when this "
        f"is on time){' [LATE: the first record seen was already at or past the trigger]' if first_sight and step > stop - 1 else ''}\n")
    (case / STOP_FILE).touch()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--poll-s", type=float, default=0.05)
    ap.add_argument("--max-wait-s", type=float, default=3600.0)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    case = args.case.resolve()
    stop = stop_step_of(case)
    log = case / LOG_NAME
    tail, pos, seen_any = Tail(), 0, False
    deadline = time.monotonic() + args.max_wait_s
    while time.monotonic() < deadline:
        if log.is_file() and log.stat().st_size > pos:
            with log.open("r", errors="replace") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
            for step in tail.feed(chunk):
                if step >= stop - 1:
                    touch(case, stop, step, not seen_any)
                    return 0
                seen_any = True
        time.sleep(args.poll_s)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
