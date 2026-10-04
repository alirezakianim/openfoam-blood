#!/usr/bin/env python3
"""The notebook must RUN the batch it was just built for.

WHY THIS EXISTS. build_main_case_colab_package.sh re-pins EXPECTED_SHA in the
notebook every time it builds, so the notebook always downloads the NEW
package -- and it has never touched `BATCH`, the variable that selects WHICH
CHILDREN to run. On 2026-09-08 that combination shipped: package 53c4edb6
carried three new children (em26_blend_off, em26_blend_on,
em26_transport_only) and the notebook was pinned to it correctly, while
`BATCH = 25` still selected em_2vel_carrier and em_2vel_carrier_ero. It would
have downloaded the right package and re-run the PREVIOUS batch, burning a
paid Colab session to reproduce a result already on disk. Alireza spotted it
by reading the notebook; nothing in the build would have.

Note that "do the named children exist?" would NOT have caught it -- the
Batch 25 children still exist and still generate. The failure is that the
selector points at an OLDER entry than the newest one defined. That is what
this checks.

DELIBERATELY RE-RUNNING AN OLD BATCH is legitimate. This refuses it at BUILD
time only, because building a package is the act that says "there is
something new to run". Set BATCH to whatever you like to run an old batch by
hand; just do not ship a package whose notebook ignores it.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / (
    "notebooks/AM2020_main_case_colab.ipynb"
)
CONTROLS = Path(__file__).resolve().parents[1] / (
    "cases/AM2020_planar_mural_detachment/constant/mainCaseControls.json"
)


def fail(message: str) -> None:
    print(f"notebook batch selector: {message}", file=sys.stderr)
    raise SystemExit(1)


def source_of(notebook: Path) -> str:
    nb = json.loads(notebook.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def parse(text: str) -> tuple[int, dict[int, str]]:
    selected = re.search(r"^BATCH = (\d+)\s*$", text, re.M)
    if not selected:
        fail("no `BATCH = <n>` line found; the selector may have been renamed")

    start = text.find("BATCHES = {")
    if start < 0:
        fail("no `BATCHES = {` mapping found")
    depth, end = 0, None
    for i in range(text.index("{", start), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        fail("the BATCHES mapping is not brace-balanced")

    # Comment lines inside the mapping are stripped by ast; a trailing comma
    # and tuple values are both fine for literal_eval.
    mapping = ast.literal_eval(text[text.index("{", start):end])
    batches = {int(k): v[0] for k, v in mapping.items()}
    return int(selected.group(1)), batches


def main() -> int:
    text = source_of(NOTEBOOK)
    selected, batches = parse(text)
    if not batches:
        fail("BATCHES is empty")

    newest = max(batches)
    known = {
        c["child_id"]
        for c in json.loads(CONTROLS.read_text())["children"]
        if c.get("child_id")
    }

    # 1. The selector must point at the newest entry.
    if selected != newest:
        fail(
            f"BATCH = {selected} but the newest entry is {newest} "
            f"({batches[newest]!r}). The build re-pins EXPECTED_SHA and does "
            f"NOT touch BATCH, so this package would download correctly and "
            f"then run batch {selected} instead. Set BATCH = {newest}, or "
            f"remove the {newest} entry if it is not meant to run."
        )

    # 2. Every child it names must actually exist, or --only silently matches
    #    nothing and the run returns an empty artefact that reads as success.
    missing = [c for c in batches[selected].split() if c not in known]
    if missing:
        fail(
            f"BATCH {selected} names {missing}, which are not child_ids in "
            f"mainCaseControls.json. `--only` would match nothing and the run "
            f"would return no children at all."
        )

    # 3. THE HOLE THIS CHECK CLOSES. Rules 1 and 2 compare the selector with
    #    the BATCHES mapping and the mapping with controls -- neither notices
    #    a child ADDED to controls that no batch names. On 2026-09-14 that
    #    shipped: em29_stage2_baseline was added, the package built, the
    #    notebook re-pinned to it, and BATCH still selected 28
    #    (em28_content_matched em27_content_s1) -- the run already on disk.
    #    Every rule above passed. The session would have downloaded the right
    #    package and reproduced an existing result.
    #
    #    A child may be legitimately unscheduled, so the escape hatch is an
    #    explicit `not_scheduled_because` on the child, not silence.
    children = json.loads(CONTROLS.read_text())["children"]
    covered = {name for v in batches.values() for name in v.split()}
    orphans = [
        c["child_id"] for c in children
        if c.get("child_id")
        and c["child_id"] not in covered
        and not str(c.get("not_scheduled_because", "")).strip()
    ]
    if orphans:
        fail(
            f"{orphans} are defined in mainCaseControls.json and named by no "
            f"BATCHES entry, so no notebook run can reach them. Add a batch "
            f"(and point BATCH at it), or give each child a "
            f"`not_scheduled_because` saying why it is defined but not run."
        )

    named = batches[selected].split()
    unscheduled = [
        c["child_id"] for c in children
        if c.get("child_id") and c["child_id"] not in covered
    ]
    print(
        f"notebook batch selector OK: BATCH = {selected} (newest), "
        f"{len(named)} child(ren), all present in controls; "
        f"{len(unscheduled)} child(ren) declared unscheduled"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
