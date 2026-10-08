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

LAUNCHES (2026-10-08, BATCH 72 split into two parallel Colab sessions). A batch with several launches is
only as safe as EACH launch's exact command. This also executes the notebook's `# >>> launch_plan`
block (the one implementation the cell itself calls) for every launch of the selected batch and of every
batch that clones a reference library, builds the argv the cell would hand Popen (the git clone is the
only thing stubbed), parses it with the runner's OWN argparse and runs the runner's OWN start-of-session
library check on it. It refuses: a launch that clones a library none of its children names (the runner
says "built for nothing"), a launch that omits the library a child names, launches that do not partition
the batch, the G23-0 identity pair split across launches, a default LAUNCH other than 1. `--self-test`
runs a mutated copy of the notebook through each of those and requires each to be refused for its stated
reason.

DELIBERATELY RE-RUNNING AN OLD BATCH is legitimate. This refuses it at BUILD
time only, because building a package is the act that says "there is
something new to run". Set BATCH to whatever you like to run an old batch by
hand; just do not ship a package whose notebook ignores it.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
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


RUNNER = Path(__file__).resolve().parents[1] / (
    "cases/AM2020_planar_mural_detachment/scripts/run_main_case_colab.py"
)
# The arms the comparator scores bitwise (G23-0, scripts/s23_validation_compare.py ARMS "1" and "1p"):
# they must share ONE launch, i.e. one session and one host.
IDENTITY_PAIRS = {72: ("S23v_G4_r3", "S23v_G4_r3_S22")}
FAKE_SOLVER = "/content/openfoam-blood-ref/user-applications/phaseClotAxisymmetricFluid"
FAKE_BASE = ["--package", "/content/package.zip", "--repo-digest", "0" * 64]


class LaunchError(Exception):
    """A launch's command is one the runner (or the pre-registration) would refuse."""


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



# ---- the launches ------------------------------------------------------------

def launch_cell(notebook: Path = NOTEBOOK) -> str:
    """The notebook code cell that holds the `# >>> launch_plan` block (other cells are not all Python)."""
    cells = ["".join(c["source"]) for c in json.loads(notebook.read_text())["cells"] if c["cell_type"] == "code"]
    hits = [c for c in cells if "# >>> launch_plan" in c]
    if len(hits) != 1:
        raise LaunchError(f"{len(hits)} notebook code cells carry `# >>> launch_plan`; exactly one must")
    return hits[0]


NAMES = ("P2PRIME_TAG", "P2PRIME_REPO_DIGEST", "S22REF_TAG", "S22REF_REPO_DIGEST", "BATCHES", "LAUNCHES", "LAUNCH",
         "LAUNCH_CONCURRENCY", "CONCURRENCY", "REFERENCE_LIBRARY_BY_BATCH", "REFERENCE_LIBRARY_BY_LAUNCH")


def notebook_names(cell: str) -> dict:
    """The cell's top-level literal assignments of NAMES, executed in order (nothing else runs)."""
    ns: dict = {}
    for node in ast.parse(cell).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in NAMES):
            exec(compile(ast.Module([node], []), "notebook", "exec"), ns)
    missing = [n for n in NAMES if n not in ns]
    if missing:
        raise LaunchError(f"the notebook cell has no top-level assignment of {missing}")
    return ns


def launch_functions(cell: str) -> dict:
    start, end = cell.find("# >>> launch_plan\n"), cell.find("# <<< launch_plan\n")
    if start < 0 or end < start or cell.count("# >>> launch_plan") != 1:
        raise LaunchError("the notebook cell has no single `# >>> launch_plan` ... `# <<< launch_plan` block")
    ns: dict = {}
    exec(compile(cell[start:end], "launch_plan", "exec"), ns)
    for name in ("launch_plan", "library_args", "runner_argv"):
        if name not in ns:
            raise LaunchError(f"the launch_plan block defines no {name}")
    return ns


_RUNNER_NS: dict = {}


def runner_namespace() -> dict:
    """The runner module executed from source (not as __main__)."""
    if not _RUNNER_NS:
        src = RUNNER.read_text()
        # the two lines of main() this check mirrors; if they move, this must be re-read
        for pin in ('reference_args = {"p2prime": args.p2prime_solver, "s22": args.s22_solver}',
                    "reference_libraries_needed(\n                series_filter,"):
            if src.count(pin) != 1:
                raise LaunchError(f"the runner no longer contains the pinned text {pin!r} once; re-read main()")
        _RUNNER_NS.update({"__name__": "runner_for_launch_check", "__file__": str(RUNNER)})
        exec(compile(src, str(RUNNER), "exec"), _RUNNER_NS)
    return _RUNNER_NS


class _Stop(Exception):
    pass


def runner_accepts(argv: list, controls: Path = CONTROLS) -> argparse.Namespace:
    """Parse argv (the Popen command) with the runner's own argparse, then run the runner's own
    start-of-session library check on it. Raises LaunchError with the runner's words if it refuses."""
    ns = runner_namespace()
    original = argparse.ArgumentParser.parse_args

    def stop(self, args=None, namespace=None):
        raise _Stop(original(self, args, namespace))
    saved = sys.argv
    argparse.ArgumentParser.parse_args = stop
    sys.argv = ["run_main_case_colab.py"] + list(argv[2:])
    usage = io.StringIO()
    try:
        with contextlib.redirect_stderr(usage):
            ns["main"]()
        raise LaunchError("the runner's main() returned before reading its arguments")
    except _Stop as parsed:
        args = parsed.args[0]
    except SystemExit as exc:
        raise LaunchError(f"the runner's argparse rejects the argv ({exc}): {usage.getvalue().strip().splitlines()[-1:]}")
    finally:
        argparse.ArgumentParser.parse_args = original
        sys.argv = saved
    needed = ns["reference_libraries_needed"](args.only.strip(), controls)
    try:
        ns["check_reference_libraries"](needed, {"p2prime": args.p2prime_solver, "s22": args.s22_solver})
    except RuntimeError as error:
        raise LaunchError(f"the runner would refuse this argv at the start of the session: {error}")
    return args


def check_launches(cell: str, controls: Path = CONTROLS, selected: int | None = None) -> list:
    """Every launch of the selected batch and of every batch that clones a reference library: its
    printed lines and its exact runner argv, accepted by the runner. Returns one record per launch."""
    names = notebook_names(cell)
    fn = launch_functions(cell)
    batches, launches = names["BATCHES"], names["LAUNCHES"]
    by_batch, by_launch = names["REFERENCE_LIBRARY_BY_BATCH"], names["REFERENCE_LIBRARY_BY_LAUNCH"]
    known = {c["child_id"] for c in json.loads(controls.read_text())["children"] if c.get("child_id")}
    if names["LAUNCH"] != 1:
        raise LaunchError(f"the default is LAUNCH = {names['LAUNCH']}; it must be LAUNCH = 1")
    strict = sorted({b for b in by_batch} | ({selected} if selected is not None else set()))
    records = []
    # structure of every multi-launch batch: the launches partition the batch, in order 1..n
    for b, parts in sorted(launches.items()):
        if sorted(parts) != list(range(1, len(parts) + 1)):
            raise LaunchError(f"BATCH {b}: launches {sorted(parts)} are not 1..{len(parts)}")
        want = batches[b][0].split()
        got = [c for _l, (only, _h) in sorted(parts.items()) for c in only.split()]
        if sorted(got) != sorted(want) or len(set(got)) != len(got):
            raise LaunchError(f"BATCH {b}: the launches {got} do not partition the batch {want} (each child in exactly one launch)")
        for lnum, (only, hours) in sorted(parts.items()):
            bad = [c for c in only.split() if c not in known]
            if bad:
                raise LaunchError(f"BATCH {b} launch {lnum} names {bad}, which are not child_ids in mainCaseControls.json")
            if not hours > 0:
                raise LaunchError(f"BATCH {b} launch {lnum}: hours {hours}")
        if b in IDENTITY_PAIRS:
            a, c = IDENTITY_PAIRS[b]
            where = {x: [l for l, (only, _h) in parts.items() if x in only.split()] for x in (a, c)}
            if where[a] != where[c] or len(where[a]) != 1:
                raise LaunchError(f"BATCH {b}: the identity pair {a} / {c} is compared bitwise and must share ONE launch; it is in {where}")
    for b in strict:
        if b not in batches:
            raise LaunchError(f"BATCH {b} is selected or has a reference library but no BATCHES entry")
        concurrency = names["LAUNCH_CONCURRENCY"].get(b, names["CONCURRENCY"])
        for lnum in sorted(launches.get(b, {1: None})):
            lines: list = []
            try:
                plan = fn["launch_plan"](b, lnum, batches, launches, by_batch, by_launch, concurrency, say=lines.append)
            except ValueError as error:
                raise LaunchError(f"BATCH {b} launch {lnum}: the notebook refuses it: {error}")
            if not plan["only"].split():
                raise LaunchError(f"BATCH {b} launch {lnum} selects no child")
            extra = fn["library_args"](plan["library"], FAKE_SOLVER) if plan["library"] else []
            argv = fn["runner_argv"]("/content/project/cases/AM2020_planar_mural_detachment/scripts/run_main_case_colab.py",
                                     "20261008-000000-am2020-main-case", FAKE_BASE + extra, plan["only"])
            args = runner_accepts(argv, controls)
            if args.only.strip() != plan["only"]:
                raise LaunchError(f"BATCH {b} launch {lnum}: the runner reads --only {args.only!r}, the plan is {plan['only']!r}")
            passed = [n for n, v in (("p2prime", args.p2prime_solver), ("s22", args.s22_solver)) if v]
            records.append({"batch": b, "launch": lnum, "children": plan["only"].split(), "hours": plan["hours"],
                            "library_passed": passed, "argv": argv, "printed": lines})
    return records


def _replace_all(cell: str, old: str, new: str, count: int) -> str:
    if cell.count(old) != count:
        raise AssertionError(f"self-test mutation anchor {old!r} occurs {cell.count(old)} times, not {count}")
    return cell.replace(old, new)


def mutate(cell: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, cell, count=1, flags=re.M)
    if n != 1:
        raise AssertionError(f"self-test mutation anchor not found: {pattern!r}")
    return out


def negatives(cell: str, controls: Path = CONTROLS, selected: int = 72) -> list:
    """Each defect, applied to a copy of the notebook cell or to a launch's argv, must be REFUSED for
    its stated reason. Returns (label, the refusal's first line) per case; raises AssertionError if one
    is accepted or refused for another reason."""
    base = check_launches(cell, controls, selected)
    argv1 = next(r["argv"] for r in base if r["batch"] == 72 and r["launch"] == 1)
    argv2 = next(r["argv"] for r in base if r["batch"] == 72 and r["launch"] == 2)
    if "--s22-solver" not in argv1 or "--s22-solver" in argv2:
        raise AssertionError("the notebook as it stands must pass --s22-solver on launch 1 only")
    i = argv1.index("--s22-solver")
    cases = [
        ("launch 2 argv WITH --s22-solver", lambda: runner_accepts(argv2 + ["--s22-solver", FAKE_SOLVER], controls),
         "built for nothing"),
        ("launch 1 argv WITHOUT --s22-solver", lambda: runner_accepts(argv1[:i] + argv1[i + 2:], controls),
         "--s22-solver was not passed"),
        ("launch 1 argv with an unknown flag", lambda: runner_accepts(argv1 + ["--s23-solver", "/x"], controls),
         "argparse rejects"),
        ("the notebook's table says launch 2 clones", lambda: check_launches(
            mutate(cell, r"^    \(72, 2\): False,", "    (72, 2): True,"), controls, selected), "built for nothing"),
        ("the notebook's table says launch 1 clones nothing", lambda: check_launches(
            mutate(cell, r"^    \(72, 1\): True,", "    (72, 1): False,"), controls, selected), "--s22-solver was not passed"),
        ("the notebook's table has no entry for launch 2", lambda: check_launches(
            mutate(cell, r"^    \(72, 2\): False,\n", ""), controls, selected), "has no entry in REFERENCE_LIBRARY_BY_LAUNCH"),
        ("the identity pair split across the launches", lambda: check_launches(
            mutate(cell, r'72: \{1: \("S23v_G4_r3 S23v_G4_r3_S22", ([\d.]+)\), 2: \("S23v_G4_nor3 S23v_G40", ([\d.]+)\)\}',
                   r'72: {1: ("S23v_G4_r3 S23v_G4_nor3", \1), 2: ("S23v_G4_r3_S22 S23v_G40", \2)}'), controls, selected),
         "must share ONE launch"),
        ("a child in no launch", lambda: check_launches(
            mutate(cell, r'2: \("S23v_G4_nor3 S23v_G40", ([\d.]+)\)\}', r'2: ("S23v_G4_nor3", \1)}'), controls, selected),
         "do not partition the batch"),
        ("a child in both launches", lambda: check_launches(
            mutate(cell, r'2: \("S23v_G4_nor3 S23v_G40", ([\d.]+)\)\}', r'2: ("S23v_G4_nor3 S23v_G40 S23v_G4_r3", \1)}'), controls, selected),
         "do not partition the batch"),
        ("a launch names a child the controls do not define", lambda: check_launches(
            _replace_all(cell, 'S23v_G40", 12.0', 'S23v_G40x", 12.0', 2), controls, selected), "not child_ids"),
        ("the default launch is 2", lambda: check_launches(
            mutate(cell, r"^LAUNCH = 1$", "LAUNCH = 2"), controls, selected), "must be LAUNCH = 1"),
        ("CONCURRENCY 1 on BATCH 72", lambda: check_launches(
            mutate(cell, r"^CONCURRENCY = 2$", "CONCURRENCY = 1"), controls, selected), "CONCURRENCY 2"),
        ("a launch the entry does not have", lambda: check_launches(
            mutate(cell, r"^    \(72, 2\): False,", "    (72, 2): False,\n    (72, 3): False,")
            .replace('2: ("S23v_G4_nor3 S23v_G40", 12.0)}', '2: ("S23v_G4_nor3 S23v_G40", 12.0), 4: ("S23v_G40", 1.0)}'),
            controls, selected), "are not 1.."),
    ]
    seen = []
    for label, run, why in cases:
        try:
            run()
        except LaunchError as error:
            if why not in str(error):
                raise AssertionError(f"{label}: refused for the WRONG reason: {error}")
            seen.append((label, str(error)[:150]))
            continue
        raise AssertionError(f"{label}: ACCEPTED; it must be refused")
    return seen


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        try:
            for label, line in negatives(launch_cell(), CONTROLS, 72):
                print(f"   refused: {label} -> {line}")
        except (LaunchError, AssertionError) as error:
            fail(f"self-test: {error}")
        print("notebook batch selector self-test OK")
        return 0
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

    # 4. EACH LAUNCH's exact command (the BATCH 72 split, 2026-10-08).
    try:
        records = check_launches(launch_cell(), CONTROLS, selected)
    except LaunchError as error:
        fail(str(error))
    for r in records:
        print(f"  BATCH {r['batch']} launch {r['launch']}: {len(r['children'])} child(ren) {' '.join(r['children'])}; "
              f"runner library flags passed: {r['library_passed'] or 'none'}")

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
