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
# A batch that is ONE session by decision (BATCH 73, the S23 production pair, Alireza 2026-10-08: "both 40 ms. both in one
# session with 2 concurrency"; BATCH 74, the D74 passive-field diagnostic, Alireza 2026-10-09: "Approve; run it alone"; BATCH 75, the
# re-run of that measurement on rev. 3.1, Alireza 2026-10-09: "Approve; rebuild and re-run (Recommended)"): its
# filter is exactly these children, it has no LAUNCHES entry and no reference library.
ONE_SESSION = {73: ("P73_G4_r3_40", "P73_G40_40"), 74: ("D74_refI",), 75: ("D75_refI",)}
# S25 BATCH 76 (the verification) is THREE sessions on the 1-core runtime, by decision (design 2026-10-10-S25-arteriole-bundle.md rev. 2.2 section 7 and 12.1; case-engineer report 15): the
# children of each launch IN THE ORDER THEY RUN, which is the generator's matrix order (Allrun runs a launch's children in matrix order, never in the order BATCHES names them; dry_run
# launch_filters_generate asserts the built order for these batches). Launch 1 holds the baseline arm (solver_library baseline), the only launch that clones a reference library.
ONE_CORE_SESSIONS = {76: {1: ("B76_OFF_A",), 2: ("B76_OFF_B",),
                          3: ("RT_P73_A", "RT_P73_B", "RT_5a_A", "RT_5a_B", "S25_5a_tan", "S25_5a_tan_half")},
                   # S25.1 BATCH 78 (design 2026-10-13-S25.1 section C(d) and D): ONE session, sequential, both arms of every pair in it, the children in the generator's matrix order (the order they run in);
                   # its one launch holds B78_OFF_A (solver_library baseline), so it clones the baseline library. No LAUNCHES entry (ONE_CORE_ONE_SESSION below).
                   78: {1: ("B78_OFF_A", "B78_OFF_B", "B78_CTL_1", "B78_CTL_2", "B78_RT_5a_A", "B78_RT_5a_B")},
                   # BATCH 79, the Y-bifurcation geometry gate (design 2026-10-14-case-y-bifurcation-gate.md rev. 2.2): ONE session, ONE child, on the S25.1 build itself (no reference library).
                   79: {1: ("B79_Y_GATE",)}}
ONE_CORE_ONE_SESSION = (78, 79)
ONE_CORE_LIBRARY_LAUNCHES = {76: {1: ["baseline"], 2: [], 3: []}, 78: {1: ["baseline"]}, 79: {1: []}}
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


NAMES = ("P2PRIME_TAG", "P2PRIME_REPO_DIGEST", "S22REF_TAG", "S22REF_REPO_DIGEST", "BASELINE_TAG", "BASELINE_REPO_DIGEST", "BATCHES", "LAUNCHES", "LAUNCH",
         "LAUNCH_CONCURRENCY", "CONCURRENCY", "REFERENCE_LIBRARY_BY_BATCH", "REFERENCE_LIBRARY_BY_LAUNCH",
         "ONE_CORE_BATCHES", "ONE_CORE_MKL_THREADS", "RESTART_FROM")


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
        for pin in ('reference_args = {"p2prime": args.p2prime_solver, "s22": args.s22_solver, "baseline": args.baseline_solver}',
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
        ns["check_reference_libraries"](needed, {"p2prime": args.p2prime_solver, "s22": args.s22_solver, "baseline": args.baseline_solver})
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
    if names["RESTART_FROM"] != "":
        raise LaunchError(f"RESTART_FROM is {names['RESTART_FROM']!r}; the shipped notebook must be a cold start (RESTART_FROM = ''), "
                          f"a restart names its tarball by hand for one session (S25 section 2C)")
    one_core = {"batches": names["ONE_CORE_BATCHES"], "threads": names["ONE_CORE_MKL_THREADS"]}
    # the baseline release's pin (BATCH 76 launch 1): its own constants, spelt so the release re-pin never touches them
    if not re.fullmatch(r"[0-9a-f]{64}", names["BASELINE_REPO_DIGEST"]) or names["BASELINE_TAG"] != "digest-" + names["BASELINE_REPO_DIGEST"][:12]:
        raise LaunchError(f"BASELINE_TAG {names['BASELINE_TAG']!r} is not 'digest-' + the first 12 hex of BASELINE_REPO_DIGEST {names['BASELINE_REPO_DIGEST']!r}")
    for _b in (76, 78):
        if _b in by_batch and tuple(by_batch[_b][:3]) != ("baseline", names["BASELINE_TAG"], names["BASELINE_REPO_DIGEST"]):
            raise LaunchError(f"REFERENCE_LIBRARY_BY_BATCH[{_b}] must be the baseline library at BASELINE_TAG / BASELINE_REPO_DIGEST; it is {by_batch.get(_b)!r}")
    for _b in ONE_CORE_ONE_SESSION:
        if _b in launches:
            raise LaunchError(f"BATCH {_b} is ONE session (both arms of every pair share it) and must have no LAUNCHES entry; it has {sorted(launches[_b])}")
    # the newest entry is always checked, even while BATCH still selects an older one (BATCH 73 is built before it is the default)
    strict = sorted({b for b in by_batch} | ({selected} if selected is not None else set()) | {max(batches)}
                    | {b for b in ONE_SESSION if b in batches})
    for b, pair in ONE_SESSION.items():
        if b in launches:
            raise LaunchError(f"BATCH {b} is ONE session and must have no LAUNCHES entry; it has {sorted(launches[b])}")
        if b in batches and batches[b][0].split() != list(pair):
            raise LaunchError(f"BATCH {b} is ONE session of {list(pair)} (its decision); the notebook selects {batches[b][0].split()}")
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
                plan = fn["launch_plan"](b, lnum, batches, launches, by_batch, by_launch, concurrency, say=lines.append, one_core=one_core)
            except ValueError as error:
                raise LaunchError(f"BATCH {b} launch {lnum}: the notebook refuses it: {error}")
            if not plan["only"].split():
                raise LaunchError(f"BATCH {b} launch {lnum} selects no child")
            unknown = [c for c in plan["only"].split() if c not in known]
            if unknown:
                raise LaunchError(f"BATCH {b} launch {lnum} names {unknown}, which are not child_ids in mainCaseControls.json")
            extra = fn["library_args"](plan["library"], FAKE_SOLVER) if plan["library"] else []
            argv = fn["runner_argv"]("/content/project/cases/AM2020_planar_mural_detachment/scripts/run_main_case_colab.py",
                                     "20261008-000000-am2020-main-case", FAKE_BASE + extra + plan["runner_args"], plan["only"])
            args = runner_accepts(argv, controls)
            if args.only.strip() != plan["only"]:
                raise LaunchError(f"BATCH {b} launch {lnum}: the runner reads --only {args.only!r}, the plan is {plan['only']!r}")
            _want = (one_core["threads"], "") if b in one_core["batches"] else (0, "")   # S25 section 12.1: a 1-core batch pins NO MKL_CBWR
            if (args.mkl_threads, args.mkl_cbwr) != _want:
                raise LaunchError(f"BATCH {b} launch {lnum}: the runner reads --mkl-threads {args.mkl_threads} --mkl-cbwr {args.mkl_cbwr!r}; "
                                  f"{'a 1-core runtime batch' if b in one_core['batches'] else 'a multi-core batch'} must pass {_want}")
            passed = [n for n, v in (("p2prime", args.p2prime_solver), ("s22", args.s22_solver), ("baseline", args.baseline_solver)) if v]
            records.append({"batch": b, "launch": lnum, "children": plan["only"].split(), "hours": plan["hours"],
                            "library_passed": passed, "argv": argv, "printed": lines})
    # S25 BATCH 76: three sessions, their children in the order they run, the baseline library on launch 1 only
    for b, sessions in ONE_CORE_SESSIONS.items():
        if b not in batches:
            continue
        got = {r["launch"]: r for r in records if r["batch"] == b}
        if sorted(got) != sorted(sessions):
            raise LaunchError(f"BATCH {b} runs launches {sorted(got)}; its decision is the sessions {sorted(sessions)}")
        for lnum, kids in sessions.items():
            if tuple(got[lnum]["children"]) != kids:
                raise LaunchError(f"BATCH {b} launch {lnum} runs {got[lnum]['children']} but its session is {list(kids)}, in this order "
                                  f"(a pair runs A then B; the runner follows the matrix order)")
            if got[lnum]["library_passed"] != ONE_CORE_LIBRARY_LAUNCHES[b][lnum]:
                raise LaunchError(f"BATCH {b} launch {lnum} passes the reference libraries {got[lnum]['library_passed']}, "
                                  f"its session passes {ONE_CORE_LIBRARY_LAUNCHES[b][lnum]}")
            if "--mkl-cbwr" in got[lnum]["argv"]:
                raise LaunchError(f"BATCH {b} launch {lnum} passes --mkl-cbwr; a 1-core batch pins no MKL CNR branch (design section 12.1)")
        if names["LAUNCH_CONCURRENCY"].get(b) != 1:
            raise LaunchError(f"BATCH {b} must be forced to CONCURRENCY 1 by LAUNCH_CONCURRENCY")
        if b not in names["ONE_CORE_BATCHES"]:
            raise LaunchError(f"BATCH {b} is a 1-core runtime batch (ONE_CORE_SESSIONS) but ONE_CORE_BATCHES {tuple(names['ONE_CORE_BATCHES'])} lacks it: "
                              f"it would run with the multi-core thread count and a pinned MKL_CBWR")
    return records


def check_one_core(cell: str, controls: Path = CONTROLS, concurrency: int = 1) -> dict:
    """S25 (the 1-core runtime batches). BATCHES has no entry for batch 76/77 until its children exist, so the
    mechanism is exercised on a SYNTHETIC entry that names an existing child: the notebook's own launch_plan for
    each batch in ONE_CORE_BATCHES at `concurrency`, the exact runner argv it builds, parsed by the runner's own
    argparse. It must carry --mkl-threads and NO --mkl-cbwr (S25 section 12.1), and a batch outside ONE_CORE_BATCHES must not carry --mkl-threads."""
    names = notebook_names(cell)
    fn = launch_functions(cell)
    one_core = {"batches": names["ONE_CORE_BATCHES"], "threads": names["ONE_CORE_MKL_THREADS"]}
    if not one_core["batches"]:
        raise LaunchError("ONE_CORE_BATCHES is empty")
    if one_core["threads"] != 1:
        raise LaunchError(f"the 1-core runtime batches run ONE solver thread (S25 section 7) and pin NO MKL_CBWR (section 12.1: the solver's own "
                          f"mkl_cbwr_set(AUTO) overrides it): ONE_CORE_MKL_THREADS must be 1; got {one_core['threads']!r}")
    if any(names["LAUNCH_CONCURRENCY"].get(_b) != 1 for _b in (76, 77, 78, 79)):
        raise LaunchError("LAUNCH_CONCURRENCY must force batches 76, 77, 78 and 79 to 1")
    child = "P73_G4_r3_40"
    out = {}
    for b in one_core["batches"]:
        batches = {b: (child, 7.5, "S25 synthetic 1-core entry")}
        lines: list = []
        try:
            plan = fn["launch_plan"](b, 1, batches, {}, {}, {}, concurrency, say=lines.append, one_core=one_core)
        except ValueError as error:
            raise LaunchError(f"BATCH {b} (1-core): the notebook refuses it: {error}")
        argv = fn["runner_argv"]("/content/project/cases/AM2020_planar_mural_detachment/scripts/run_main_case_colab.py",
                                 "20261010-000000-am2020-main-case", FAKE_BASE + plan["runner_args"], plan["only"])
        args = runner_accepts(argv, controls)
        if (args.mkl_threads, args.mkl_cbwr) != (one_core["threads"], ""):
            raise LaunchError(f"BATCH {b} (1-core): the argv {argv} must carry --mkl-threads {one_core['threads']} and NO --mkl-cbwr "
                              f"(section 12.1; the runner reads {args.mkl_threads}, {args.mkl_cbwr!r})")
        if "--mkl-cbwr" in argv:
            raise LaunchError(f"BATCH {b} (1-core): the argv {argv} carries --mkl-cbwr; a 1-core batch pins no MKL CNR branch (section 12.1)")
        if args.restart_from != "":
            raise LaunchError(f"BATCH {b} (1-core): a restart in the default argv")
        out[b] = argv
    return out


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
    argv73 = next(r["argv"] for r in base if r["batch"] == 73 and r["launch"] == 1)
    if "--s22-solver" in argv73 or "--p2prime-solver" in argv73:
        raise AssertionError("BATCH 73 must pass no reference-library flag")
    argv74 = next(r["argv"] for r in base if r["batch"] == 74 and r["launch"] == 1)
    if "--s22-solver" in argv74 or "--p2prime-solver" in argv74:
        raise AssertionError("BATCH 74 must pass no reference-library flag")
    argv75 = next(r["argv"] for r in base if r["batch"] == 75 and r["launch"] == 1)
    if "--s22-solver" in argv75 or "--p2prime-solver" in argv75:
        raise AssertionError("BATCH 75 must pass no reference-library flag")
    cases = [
        ("BATCH 73 argv WITH --s22-solver", lambda: runner_accepts(argv73 + ["--s22-solver", FAKE_SOLVER], controls),
         "built for nothing"),
        ("BATCH 73 split into two launches", lambda: check_launches(
            mutate(cell, r"^LAUNCHES = \{", 'LAUNCHES = {\n    73: {1: ("P73_G4_r3_40", 5.0), 2: ("P73_G40_40", 10.1)},'), controls, selected),
         "ONE session and must have no LAUNCHES entry"),
        ("BATCH 73 loses a child", lambda: check_launches(
            _replace_all(cell, '73: ("P73_G4_r3_40 P73_G40_40", 10.1,', '73: ("P73_G4_r3_40", 10.1,', 1), controls, selected),
         "is ONE session of"),
        ("BATCH 73 names a child the controls do not define", lambda: check_launches(
            cell, _controls_without(controls, "P73_G40_40"), selected), "not child_ids in mainCaseControls.json"),
        ("BATCH 73 given a reference library by launch", lambda: check_launches(
            mutate(cell, r"^    \(72, 2\): False,", "    (72, 2): False,\n    (73, 1): True,"), controls, selected),
         "which REFERENCE_LIBRARY_BY_BATCH does not"),
        ("BATCH 73 at LAUNCH 2", lambda: fn_launch73(cell, 2, controls), "no launch split"),
        ("BATCH 73 at CONCURRENCY 1", lambda: fn_launch73(cell, 1, controls, concurrency=1), "CONCURRENCY 2"),
        # BATCH 74 (D74, the passive-field diagnostic): ONE session, ONE child, CONCURRENCY 2, no library
        ("BATCH 74 argv WITH --s22-solver", lambda: runner_accepts(argv74 + ["--s22-solver", FAKE_SOLVER], controls),
         "built for nothing"),
        ("BATCH 74 argv WITH --p2prime-solver", lambda: runner_accepts(argv74 + ["--p2prime-solver", FAKE_SOLVER], controls),
         "built for nothing"),
        ("BATCH 74 split into two launches", lambda: check_launches(
            mutate(cell, r"^LAUNCHES = \{", 'LAUNCHES = {\n    74: {1: ("D74_refI", 4.6), 2: ("D74_refI", 1.0)},'), controls, selected),
         "ONE session and must have no LAUNCHES entry"),
        ("BATCH 74 gains a second child", lambda: check_launches(
            _replace_all(cell, '74: ("D74_refI", 4.6,', '74: ("D74_refI P73_G4_r3_40", 4.6,', 1), controls, selected),
         "is ONE session of"),
        ("BATCH 74 loses its child", lambda: check_launches(
            _replace_all(cell, '74: ("D74_refI", 4.6,', '74: ("", 4.6,', 1), controls, selected),
         "is ONE session of"),
        ("BATCH 74 names a child the controls do not define", lambda: check_launches(
            cell, _controls_without(controls, "D74_refI"), selected), "not child_ids in mainCaseControls.json"),
        ("BATCH 74 given a reference library by launch", lambda: check_launches(
            mutate(cell, r"^    \(72, 2\): False,", "    (72, 2): False,\n    (74, 1): True,"), controls, selected),
         "which REFERENCE_LIBRARY_BY_BATCH does not"),
        ("BATCH 74 at LAUNCH 2", lambda: fn_launch73(cell, 2, controls, batch=74), "no launch split"),
        ("BATCH 74 at CONCURRENCY 1", lambda: fn_launch73(cell, 1, controls, concurrency=1, batch=74), "CONCURRENCY 2"),
        ("BATCH 74 at CONCURRENCY 3", lambda: fn_launch73(cell, 1, controls, concurrency=3, batch=74), "CONCURRENCY 2"),
        # BATCH 75 (D75, the re-run of the D74 measurement on rev. 3.1): ONE session, ONE child, CONCURRENCY 2, no library
        ("BATCH 75 argv WITH --s22-solver", lambda: runner_accepts(argv75 + ["--s22-solver", FAKE_SOLVER], controls),
         "built for nothing"),
        ("BATCH 75 argv WITH --p2prime-solver", lambda: runner_accepts(argv75 + ["--p2prime-solver", FAKE_SOLVER], controls),
         "built for nothing"),
        ("BATCH 75 split into two launches", lambda: check_launches(
            mutate(cell, r"^LAUNCHES = \{", 'LAUNCHES = {\n    75: {1: ("D75_refI", 4.5), 2: ("D75_refI", 1.0)},'), controls, selected),
         "ONE session and must have no LAUNCHES entry"),
        ("BATCH 75 gains a second child", lambda: check_launches(
            _replace_all(cell, '75: ("D75_refI", 4.5,', '75: ("D75_refI D74_refI", 4.5,', 1), controls, selected),
         "is ONE session of"),
        ("BATCH 75 loses its child", lambda: check_launches(
            _replace_all(cell, '75: ("D75_refI", 4.5,', '75: ("", 4.5,', 1), controls, selected),
         "is ONE session of"),
        ("BATCH 75 names a child the controls do not define", lambda: check_launches(
            cell, _controls_without(controls, "D75_refI"), selected), "not child_ids in mainCaseControls.json"),
        ("BATCH 75 given a reference library by launch", lambda: check_launches(
            mutate(cell, r"^    \(72, 2\): False,", "    (72, 2): False,\n    (75, 1): True,"), controls, selected),
         "which REFERENCE_LIBRARY_BY_BATCH does not"),
        ("BATCH 75 at LAUNCH 2", lambda: fn_launch73(cell, 2, controls, batch=75), "no launch split"),
        ("BATCH 75 at CONCURRENCY 1", lambda: fn_launch73(cell, 1, controls, concurrency=1, batch=75), "CONCURRENCY 2"),
        ("BATCH 75 at CONCURRENCY 3", lambda: fn_launch73(cell, 1, controls, concurrency=3, batch=75), "CONCURRENCY 2"),
        # S25, the 1-core runtime batches (76, 77): CONCURRENCY 1, ONE solver thread, NO pinned MKL_CBWR (design section 12.1); the BATCH 72-75 branches are not reused
        ("a 1-core batch at CONCURRENCY 2", lambda: check_one_core(cell, controls, concurrency=2), "CONCURRENCY must be 1"),
        ("a 1-core batch at CONCURRENCY 3", lambda: check_one_core(cell, controls, concurrency=3), "CONCURRENCY must be 1"),
        ("a 1-core batch that loses its thread pin", lambda: check_one_core(
            _replace_all(cell, "runner_args = ['--mkl-threads', str(one_core['threads'])]", "runner_args = []", 1), controls),
         "must carry --mkl-threads"),
        ("a 1-core batch that pins an MKL CNR branch again", lambda: check_one_core(
            _replace_all(cell, "runner_args = ['--mkl-threads', str(one_core['threads'])]",
                         "runner_args = ['--mkl-threads', str(one_core['threads']), '--mkl-cbwr', 'AVX2,STRICT']", 1), controls),
         "NO --mkl-cbwr"),
        ("BATCH 76 launch 3 swaps a pair (B before A)", lambda: check_launches(
            _replace_all(cell, '3: ("RT_P73_A RT_P73_B RT_5a_A RT_5a_B S25_5a_tan S25_5a_tan_half", 4.1)}',
                         '3: ("RT_P73_B RT_P73_A RT_5a_A RT_5a_B S25_5a_tan S25_5a_tan_half", 4.1)}', 1), controls, selected),
         "in this order"),
        ("BATCH 76 launch 3 runs the 5a pair before the RT pairs", lambda: check_launches(
            _replace_all(cell, '3: ("RT_P73_A RT_P73_B RT_5a_A RT_5a_B S25_5a_tan S25_5a_tan_half", 4.1)}',
                         '3: ("S25_5a_tan S25_5a_tan_half RT_P73_A RT_P73_B RT_5a_A RT_5a_B", 4.1)}', 1), controls, selected),
         "in this order"),
        ("BATCH 76 launch 2 clones the baseline library", lambda: check_launches(
            mutate(cell, r"^    \(76, 2\): False,", "    (76, 2): True,"), controls, selected), "built for nothing"),
        ("BATCH 76 launch 1 does not clone the baseline library", lambda: check_launches(
            mutate(cell, r"^    \(76, 1\): True,", "    (76, 1): False,"), controls, selected), "--baseline-solver was not passed"),
        ("BATCH 76 loses its CONCURRENCY 1 forcing", lambda: check_launches(
            _replace_all(cell, "LAUNCH_CONCURRENCY = {69: 1, 71: 1, 76: 1, 77: 1, 78: 1, 79: 1}", "LAUNCH_CONCURRENCY = {69: 1, 71: 1, 77: 1, 78: 1, 79: 1}", 1), controls, selected),
         "CONCURRENCY must be 1"),
        ("the baseline tag is not its digest's prefix", lambda: check_launches(
            mutate(cell, r"^BASELINE_TAG = 'digest-[0-9a-f]{12}'$", "BASELINE_TAG = 'digest-000000000000'"), controls, selected),
         "is not 'digest-' + the first 12 hex"),
        ("BATCH 76 clones another release as its baseline", lambda: check_launches(
            _replace_all(cell, "76: ('baseline', BASELINE_TAG, BASELINE_REPO_DIGEST,", "76: ('baseline', S22REF_TAG, S22REF_REPO_DIGEST,", 1), controls, selected),
         "must be the baseline library"),
        ("a 1-core batch with 6 solver threads", lambda: check_one_core(
            mutate(cell, r"^ONE_CORE_MKL_THREADS = 1$", "ONE_CORE_MKL_THREADS = 6"), controls),
         "ONE_CORE_MKL_THREADS must be 1"),
        # S25.1 BATCH 78: ONE session, both arms of every pair in it, sequential in the generator's matrix order, the baseline library cloned once
        ("BATCH 78 runs OFF_B before OFF_A (a pair runs A then B)", lambda: check_launches(
            _replace_all(cell, '78: ("B78_OFF_A B78_OFF_B B78_CTL_1', '78: ("B78_OFF_B B78_OFF_A B78_CTL_1', 1), controls, selected),
         "in this order"),
        ("BATCH 78 runs the 5a pair before the control pair", lambda: check_launches(
            _replace_all(cell, 'B78_CTL_1 B78_CTL_2 B78_RT_5a_A B78_RT_5a_B", 7.8,', 'B78_RT_5a_A B78_RT_5a_B B78_CTL_1 B78_CTL_2", 7.8,', 1), controls, selected),
         "in this order"),
        ("BATCH 78 loses a child", lambda: check_launches(
            _replace_all(cell, 'B78_RT_5a_A B78_RT_5a_B", 7.8,', 'B78_RT_5a_A", 7.8,', 1), controls, selected),
         "in this order"),
        ("BATCH 78 split into two launches", lambda: check_launches(
            mutate(cell, r"^LAUNCHES = \{", 'LAUNCHES = {\n    78: {1: ("B78_OFF_A B78_OFF_B", 7.5), 2: ("B78_CTL_1 B78_CTL_2 B78_RT_5a_A B78_RT_5a_B", 0.3)},'), controls, selected),
         "ONE session (both arms of every pair share it) and must have no LAUNCHES entry"),
        ("BATCH 78 at LAUNCH 2", lambda: fn_launch73(cell, 2, controls, concurrency=1, batch=78), "no launch split"),
        ("BATCH 78 does not clone the baseline library", lambda: check_launches(
            mutate(cell, r"^    \(78, 1\): True,", "    (78, 1): False,"), controls, selected), "--baseline-solver was not passed"),
        ("BATCH 78 clones another release as its baseline", lambda: check_launches(
            _replace_all(cell, "78: ('baseline', BASELINE_TAG, BASELINE_REPO_DIGEST,", "78: ('baseline', S22REF_TAG, S22REF_REPO_DIGEST,", 1), controls, selected),
         "must be the baseline library"),
        ("BATCH 78 loses its CONCURRENCY 1 forcing", lambda: check_launches(
            _replace_all(cell, "LAUNCH_CONCURRENCY = {69: 1, 71: 1, 76: 1, 77: 1, 78: 1, 79: 1}", "LAUNCH_CONCURRENCY = {69: 1, 71: 1, 76: 1, 77: 1, 79: 1}", 1), controls, selected),
         "CONCURRENCY must be 1"),
        ("BATCH 78 names a child the controls do not define", lambda: check_launches(
            cell, _controls_without(controls, "B78_CTL_2"), selected), "not child_ids in mainCaseControls.json"),
        # BATCH 79, the Y-bifurcation gate: ONE session, ONE child, the 1-core runtime, no LAUNCHES entry, no reference library
        ("BATCH 79 names a child the controls do not define", lambda: check_launches(
            cell, _controls_without(controls, "B79_Y_GATE"), selected), "not child_ids in mainCaseControls.json"),
        ("BATCH 79 runs a second child", lambda: check_launches(
            _replace_all(cell, '79: ("B79_Y_GATE", 9.4,', '79: ("B79_Y_GATE B78_CTL_1", 9.4,', 1), controls, selected), "in this order"),
        ("BATCH 79 split into two launches", lambda: check_launches(
            mutate(cell, r"^LAUNCHES = \{", 'LAUNCHES = {\n    79: {1: ("B79_Y_GATE", 9.4), 2: ("B79_Y_GATE", 0.1)},'), controls, selected),
         "must have no LAUNCHES entry"),
        ("BATCH 79 at LAUNCH 2", lambda: fn_launch73(cell, 2, controls, concurrency=1, batch=79), "no launch split"),
        ("BATCH 79 forced to CONCURRENCY 2", lambda: check_launches(
            _replace_all(cell, "LAUNCH_CONCURRENCY = {69: 1, 71: 1, 76: 1, 77: 1, 78: 1, 79: 1}", "LAUNCH_CONCURRENCY = {69: 1, 71: 1, 76: 1, 77: 1, 78: 1, 79: 2}", 1), controls, selected),
         "CONCURRENCY must be 1"),
        ("BATCH 79 given a reference library by launch", lambda: check_launches(
            mutate(cell, r"^    \(78, 1\): True,", "    (78, 1): True,\n    (79, 1): True,"), controls, selected),
         "which REFERENCE_LIBRARY_BY_BATCH does not"),
        ("BATCH 79 loses its CONCURRENCY 1 forcing", lambda: check_launches(
            _replace_all(cell, "LAUNCH_CONCURRENCY = {69: 1, 71: 1, 76: 1, 77: 1, 78: 1, 79: 1}", "LAUNCH_CONCURRENCY = {69: 1, 71: 1, 76: 1, 77: 1, 78: 1}", 1), controls, selected),
         "CONCURRENCY must be 1"),
        ("BATCH 79 dropped from the 1-core batches (it would run with six MKL threads)", lambda: check_launches(
            mutate(cell, r"^ONE_CORE_BATCHES = \(76, 77, 78, 79\)$", "ONE_CORE_BATCHES = (76, 77, 78)"), controls, selected),
         "lacks it"),
        ("BATCH 75 declared a 1-core batch (its branch needs CONCURRENCY 2)", lambda: check_launches(
            mutate(cell, r"^ONE_CORE_BATCHES = \(76, 77, 78, 79\)$", "ONE_CORE_BATCHES = (75, 76, 77, 78, 79)"), controls, selected),
         "CONCURRENCY must be 1"),
        ("a multi-core batch whose argv carries the 1-core pin", lambda: check_launches(
            _replace_all(cell, "'library': library, 'runner_args': runner_args}",
                         "'library': library, 'runner_args': ['--mkl-threads', '1']}", 1), controls, selected),
         "a multi-core batch must pass (0, '')"),
        ("the shipped notebook restarts", lambda: check_launches(
            mutate(cell, r"^RESTART_FROM = ''$", "RESTART_FROM = '/content/drive/x.tar.gz'"), controls, selected),
         "must be a cold start"),
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


def _controls_without(controls: Path, child_id: str) -> Path:
    """A scratch copy of the controls with one child removed (the file is deleted at exit)."""
    import atexit
    import os
    import tempfile
    data = json.loads(controls.read_text())
    kept = [c for c in data["children"] if c.get("child_id") != child_id]
    if len(kept) != len(data["children"]) - 1:
        raise AssertionError(f"self-test: {child_id} is not a child of the controls")
    data["children"] = kept
    handle, name = tempfile.mkstemp(prefix="controls-without-", suffix=".json")
    with os.fdopen(handle, "w") as out:
        out.write(json.dumps(data))
    atexit.register(lambda: os.path.exists(name) and os.unlink(name))
    return Path(name)


def fn_launch73(cell: str, launch: int, controls: Path, concurrency: int = 2, batch: int = 73) -> None:
    """The notebook's own launch_plan for a ONE_SESSION batch (73, or 74 / 75 with batch=74 / batch=75) at a launch / concurrency
    the decision forbids."""
    names = notebook_names(cell)
    fn = launch_functions(cell)
    try:
        fn["launch_plan"](batch, launch, names["BATCHES"], names["LAUNCHES"], names["REFERENCE_LIBRARY_BY_BATCH"],
                          names["REFERENCE_LIBRARY_BY_LAUNCH"], concurrency, say=lambda _line: None)
    except ValueError as error:
        raise LaunchError(f"BATCH {batch}: the notebook refuses it: {error}")


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        try:
            for label, line in negatives(launch_cell(), CONTROLS, 79):
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

    try:
        one_core_argv = check_one_core(launch_cell(), CONTROLS)
    except LaunchError as error:
        fail(str(error))
    for b, argv in sorted(one_core_argv.items()):
        print(f"  BATCH {b} (1-core runtime, synthetic entry): argv carries (no --mkl-cbwr) {' '.join(argv[argv.index('--mkl-threads'):argv.index('--mkl-threads') + 2])}")

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
