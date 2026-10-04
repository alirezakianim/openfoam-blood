#!/usr/bin/env python3
"""Refuse to ship if the stage-3 gate could fail to run on a child.

WHY THIS EXISTS. Twice, a child was added to the controls and the stage-3 gate
silently did not run on it:

  * run 20260818-081527 -- five diag_* children, the newly-added det(be) gate
    never fired, and every number had to be re-extracted from raw logs by hand.
  * run 20260823-052707 -- force_cell and force_face matched no prefix, so
    assess_stage3.py was NEVER INVOKED, no stage3_result.json was written, and
    the run reported gate_exit_code 0 / status VALID / passed True with the
    entire stage-3 gate absent. It was noticed because the notebook cell that
    reads the JSON raised -- by Alireza, not by any check.

Both times the mitigation was to add the new prefix to a hard-coded list. That
works until the next child is named something new, which happened twice. The
trigger is now derived from the child's SERIES in both files.

WHAT THIS CHECKS, AND WHY IT IS NOT VACUOUS. The first version of this script
compared "children in a stage-3 series" against "children the runner would
select", where both sides evaluated the same series test -- so the two sets were
equal by construction and the check could never fail. That is the same
never-fails-so-never-tested defect it was written to prevent, and it is why the
assertions below are about the TRIGGER MECHANISM rather than about set equality:

  1. assess_stage3.py defines EULERIAN_SERIES_PREFIX and discover_pairs() reads
     each child's series and tests it.
  2. run_main_case_colab.py's stage3 selection reads the series from
     matrix.json and tests it.
  3. Every stage-3 child's series actually starts with that prefix.
  4. Reported, not enforced: how many children the prefix fallback ALONE would
     miss, so the series trigger's load-bearing-ness is visible.

Delete either series test and this fails. That is the property worth having.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "cases" / "AM2020_planar_mural_detachment"
CONTROLS = CASE / "constant" / "mainCaseControls.json"
RUNNER = CASE / "scripts" / "run_main_case_colab.py"
ASSESSOR = CASE / "scripts" / "assess_stage3.py"


def prefixes(text: str, name: str) -> tuple[str, ...]:
    m = re.search(rf"^{re.escape(name)}\s*=\s*\((.*?)\)", text, re.S | re.M)
    return tuple(re.findall(r'"([^"]+)"', m.group(1))) if m else ()


def main() -> int:
    controls = json.loads(CONTROLS.read_text())
    children = controls["children"]
    assessor = ASSESSOR.read_text()
    runner = RUNNER.read_text()
    problems: list[str] = []

    # 1. The assessor's series trigger exists and is used.
    m = re.search(r'^EULERIAN_SERIES_PREFIX\s*=\s*"([^"]+)"', assessor, re.M)
    if not m:
        problems.append(
            "assess_stage3.py has no EULERIAN_SERIES_PREFIX -- the "
            "series-derived trigger is gone and discovery is back to a "
            "hard-coded prefix list."
        )
        prefix = "eulerian"
    else:
        prefix = m.group(1)
        disc = re.search(r"def discover_pairs.*?\n(?=\ndef |\Z)", assessor, re.S)
        body = disc.group(0) if disc else ""
        if "EULERIAN_SERIES_PREFIX" not in body or "series" not in body:
            problems.append(
                "assess_stage3.py defines EULERIAN_SERIES_PREFIX but "
                "discover_pairs() does not read a child's series with it, so "
                "the constant is decorative and discovery is prefix-only."
            )

    # 2. The runner's series trigger exists and is used.
    sel = re.search(r"stage3_series = set\(\).*?stage3 = sorted\(.*?\n            \)",
                    runner, re.S)
    if not sel:
        problems.append(
            "run_main_case_colab.py has no stage3_series block -- the runner "
            "decides whether to invoke the assessor AT ALL, so losing this "
            "means the gate can be absent while the run reports VALID."
        )
    elif 'startswith("eulerian")' not in sel.group(0) and prefix not in sel.group(0):
        problems.append(
            "run_main_case_colab.py's stage3 selection does not test the "
            f"series against {prefix!r}."
        )

    # 3. Every stage-3 child's series really starts with that prefix.
    known = prefixes(assessor, "EULERIAN_ONLY_PREFIXES")
    orphans = sorted(
        c["child_id"] for c in children
        if c["child_id"].startswith(known)
        and not str(c.get("series", "")).startswith(prefix)
    )
    if orphans:
        problems.append(
            "these children match a stage-3 prefix but their series does not "
            f"start with {prefix!r}, so they rely on the fallback alone: "
            + ", ".join(orphans)
        )

    # 4. Reported, not enforced -- is the series trigger load-bearing today?
    in_series = {
        c["child_id"] for c in children
        if str(c.get("series", "")).startswith(prefix)
    }
    by_prefix = {c for c in in_series if c.startswith(known)}
    only_series = sorted(in_series - by_prefix)

    if problems:
        print("STAGE-3 GATE TRIGGER IS NOT SAFE:")
        for p in problems:
            print(f"  {p}")
        print("\n  Do not fix this by adding a prefix. Make the child's series "
              "start with 'eulerian', or restore the series-derived trigger in "
              "BOTH files. Runs 20260818-081527 and 20260823-052707 are what "
              "this prevents.")
        return 1

    print(f"stage-3 gate trigger OK: series-derived in both files, "
          f"{len(in_series)} children in stage-3 series")
    if only_series:
        print(f"  {len(only_series)} reach the gate ONLY via the series test: "
              f"{', '.join(only_series)}")
    else:
        print("  the prefix fallback happens to cover all of them today; the "
              "series test is still what makes the next one safe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
