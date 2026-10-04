#!/usr/bin/env python3
"""The step-1 watcher for the free-shell throat series (design
2026-10-02-case-free-shell-throat, rev. 3, S7.3b; red team 03 #3, red team 04
#2).

Launched by Allrun BESIDE foamRun (background), it tails
log.phaseClotAxisymmetricFluid and, at the FIRST PHASE_CLOT_DEBOND,
PHASE_CLOT_RELEASED and PHASE_CLOT_BONDHOLD record, checks the five keys this
case's whole premise rests on:

    PHASE_CLOT_DEBOND:   bondedClotFaces, injuryFaces      (must be 0)
    PHASE_CLOT_RELEASED: supported, bondSeeds               (must be 0)
    PHASE_CLOT_BONDHOLD: seeds                              (must be 0)

If any is non-zero the clot was never free, continuing the run only burns the
wall budget on a configuration the case was not built to answer. The watcher
touches the stopAtFile watch file (default "$FOAM_CASE/<name>", i.e.
<case>/stopWatcher for the `stopWatcher` function object this series writes)
and records WHY in watcher-stop.txt -- NOT run-exit-code.txt, which Allrun
overwrites with the solver's own exit code after it stops.

A latch failure then costs the time to the next log flush, not the 13 h wall
budget (red team 03 #3).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

RECORD = re.compile(r"^(PHASE_CLOT_DEBOND|PHASE_CLOT_RELEASED|PHASE_CLOT_BONDHOLD)\b(.*)$")
KV = re.compile(r"(\w+)=([^\s]+)")

# record type -> keys that must read 0 at the FIRST record of that type.
V1_KEYS = {
    "PHASE_CLOT_DEBOND": ("bondedClotFaces", "injuryFaces"),
    "PHASE_CLOT_RELEASED": ("supported", "bondSeeds"),
    "PHASE_CLOT_BONDHOLD": ("seeds",),
}


def check_line(line: str) -> str | None:
    """Return a reason string if this line fails V1, else None."""
    m = RECORD.match(line)
    if not m:
        return None
    record, rest = m.group(1), m.group(2)
    kv = dict(KV.findall(rest))
    bad = []
    for key in V1_KEYS[record]:
        if key not in kv:
            continue
        try:
            value = float(kv[key])
        except ValueError:
            continue
        if value != 0.0:
            bad.append(f"{key}={kv[key]}")
    if bad:
        return f"{record}: {', '.join(bad)} (expected 0 at step 1; V1 failed)"
    return None


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: clot_free_watcher.py <case_dir>")
    case = Path(sys.argv[1]).resolve()
    log_path = case / "log.phaseClotAxisymmetricFluid"
    stop_path = case / "stopWatcher"
    reason_path = case / "watcher-stop.txt"

    seen_types: set[str] = set()
    pos = 0
    # Poll rather than inotify: Allrun launches this with no extra
    # dependency, and the solver's own write cadence (per step) bounds the
    # latency this needs anyway.
    while len(seen_types) < len(V1_KEYS):
        if reason_path.exists():
            return 0  # another invocation (or a restart) already decided
        if not log_path.exists():
            time.sleep(0.2)
            continue
        with log_path.open("r", errors="replace") as fh:
            fh.seek(pos)
            lines = fh.readlines()
            pos = fh.tell()
        for line in lines:
            m = RECORD.match(line)
            if not m:
                continue
            record = m.group(1)
            if record in seen_types:
                continue  # only the FIRST record of each type is V1
            seen_types.add(record)
            reason = check_line(line)
            if reason:
                reason_path.write_text(
                    f"WATCHER-STOPPED: {reason}\nline: {line.rstrip()}\n"
                )
                stop_path.touch()
                return 0
        if not log_path.exists() or pos == log_path.stat().st_size:
            # End of case: foamRun may have exited (or not started yet with
            # content). Stop politely once all three record types have been
            # seen with nothing to report, or give up after the solver ends.
            time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
