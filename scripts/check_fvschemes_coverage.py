#!/usr/bin/env python3
"""Fail the build if the solver can ask for a divergence scheme no case defines.

Run 20260810-032625 lost all six finiteStrain children to one line. The
finite-strain path called

    fvc::div(sigmaD_ & (I + gradUs))

with no scheme name, so OpenFOAM looked up the scheme by the EXPRESSION's own
name, "div((sigmaD&((1)+grad(Us))))". Every case carries `default none;` in
divSchemes, which turns an unlisted entry into a fatal error at the first
timestep. Nothing offline caught it: the library compiled, the dry runs pass on
synthesised fields, and the case files were all valid.

The rule enforced here is narrow on purpose:

  * every `fvc::div(` / `fvm::div(` in the solver must either carry an explicit
    quoted scheme name, or be one of the ALLOWED_UNNAMED calls below -- the
    exact set already proven to run;
  * every scheme name the solver can request must appear in the divSchemes of
    every case, unless that case's divSchemes has no `default none`.

A new unnamed div is therefore a build failure with a message that names the
fix, rather than a run that dies 40 minutes into a Colab session.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Unnamed div calls proven to run. The value is the fvSchemes entry OpenFOAM
# derives from the expression. Adding to this list means asserting a run has
# demonstrated it; naming the call in the source is the cheaper option.
ALLOWED_UNNAMED = {
    "fvc::div(sigmaD_)": "div(sigmaD)",
    "fvm::div(qPhi, w_)": "div(qPhi,w)",
    "fvm::div(phi_, U_)": "div(phi,U)",
    # NO SCHEME IS LOOKED UP AT ALL for these. fvc::div of a SURFACE field is
    # fvc::surfaceIntegrateExtrapolate (src/finiteVolume/finiteVolume/fvc/
    # fvcDiv.C:46) -- a bare sum of face values over the cell volume, with no
    # divScheme and no convectionScheme involved. `default none` cannot make it
    # fatal because nothing asks the dictionary anything. Mapped to None so the
    # scan records "checked, requires nothing" rather than silently passing.
    "fvc::div(phi)": None,
    "fvc::div(phi_)": None,
    # Same rule, one more surface field: phiHbyA is the surfaceScalarField the
    # pressure equation is built from, so fvc::div(phiHbyA) is again a bare
    # surface integral and selects nothing. It appears in
    # phaseClotAxisymmetricFluid::correctPressure(), copied from OpenFOAM-13's
    # own applications/modules/incompressibleFluid/correctPressure.cpp, where it
    # is written exactly the same way -- every standard incompressible case runs
    # `default none` with no div(phiHbyA) entry and does not fail.
    #
    # ADDED DELIBERATELY AND NARROWLY. The alternative on offer was to inject a
    # meaningless `div(phiHbyA)` entry into 66 cases' fvSchemes to silence a
    # checker, which would have made every one of those dictionaries claim a
    # scheme that nothing reads.
    "fvc::div(phiHbyA)": None,
}

DIV_CALL = re.compile(r"\bfv[cm]::div\(")

# A DIRECT LOOKUP IS EXACTLY AS BINDING AS AN fvm::div, AND THIS CHECKER COULD
# NOT SEE ONE.
#
# `phaseFieldFaceWeights()` asks for its weights with
#
#     ITstream& cStream = mesh_.schemes().div("div(phiReleased,c)");
#
# which never goes through fvc::div or fvm::div, so DIV_CALL above misses it
# structurally. Run 20260913-044404 died on exactly that entry -- all three
# arms, step one -- against a case this checker had just reported as needing
# six OTHER schemes and not that one. A checker whose whole purpose is "no
# application may ask for a scheme its case does not define" must see every
# way of asking.
SCHEMES_LOOKUP = re.compile(r"\.schemes\(\)\s*\.\s*div\(\s*\"([^\"]+)\"")


def split_args(text: str, start: int) -> tuple[str, int]:
    """Return the argument text of a call whose '(' is at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i
    raise ValueError("unbalanced parentheses in solver source")


def blank_cpp_comments(text: str) -> str:
    """Replace comment bodies with spaces, preserving every byte offset.

    The scan reports line numbers from character offsets, so comments cannot
    simply be deleted -- that would shift every line number after the first
    comment and send the reader to the wrong place. Newlines are kept so the
    line count is exact.

    This exists because the guard flagged its own documentation: a comment
    reading "matching fvc::div(sigma) and dragCoeff*U" was scanned as code and
    reported as an unnamed divergence. A checker that cannot tell code from
    prose about code will cry wolf, and a guard people learn to override is
    worse than no guard.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif two == "/*":
            while i < n and text[i:i + 2] != "*/":
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
        elif text[i] == '"':
            # A string literal is code: the scheme NAME lives in one.
            out.append(text[i])
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i])
                    i += 1
                out.append(text[i])
                i += 1
            # CONSUME THE CLOSING QUOTE. Without this, i still points at it,
            # the outer loop reads it as an OPENING quote, and the parity
            # inverts for the rest of the file -- so every real comment after
            # the first string literal is scanned as code.
            if i < n:
                out.append(text[i])
                i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def scan_solver(source: Path) -> tuple[set[str], list[str]]:
    text = blank_cpp_comments(source.read_text())
    requested: set[str] = set()
    problems: list[str] = []

    for match in SCHEMES_LOOKUP.finditer(text):
        requested.add(match.group(1))

    for match in DIV_CALL.finditer(text):
        open_paren = match.end() - 1
        args, close = split_args(text, open_paren)
        call = text[match.start() : close + 1]
        line = text.count("\n", 0, match.start()) + 1

        quoted = re.findall(r'"([^"]+)"', args)
        if quoted:
            requested.add(quoted[-1])
            continue

        normalised = re.sub(r"\s+", " ", call).strip()
        if normalised in ALLOWED_UNNAMED:
            scheme = ALLOWED_UNNAMED[normalised]
            if scheme is not None:
                requested.add(scheme)
            continue

        problems.append(
            f"{source.name}:{line}: unnamed div with no proven scheme entry:\n"
            f"    {normalised}\n"
            f"  OpenFOAM will look this up by the expression's own name and "
            f"`default none` will make it fatal.\n"
            f'  Fix: pass an explicit scheme name, e.g. ..., "div(sigmaD)").'
        )

    return requested, problems


def generated_schemes(target: Path):
    """Run a case's own generator in a scratch copy and return its fvSchemes.

    In a scratch copy on purpose. Generating into the working tree would leave
    `runs/` behind, and the package stamp hashes `cases/` -- so a check run
    before a build would silently change the content id of the thing it was
    checking.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="fvschemes-check-"))
    try:
        scratch = tmp / target.name
        shutil.copytree(
            target, scratch,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "runs",
                                          "postProcessing", "log.*"),
        )
        done = subprocess.run(
            [sys.executable, str(scratch / "scripts" / "create_cases.py")],
            capture_output=True, text=True,
        )
        if done.returncode != 0:
            print(done.stdout + done.stderr)
            return None
        found = sorted(scratch.glob("**/system/fvSchemes"))
        if not found:
            return None
        # Read them here: the scratch tree is gone by the time the caller
        # would open them, so hand back paths that still exist.
        kept = []
        stash = tmp.parent / f"{tmp.name}-kept"
        stash.mkdir(exist_ok=True)
        for i, path in enumerate(found):
            child = stash / f"{path.parent.parent.name}" / "system"
            child.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, child / "fvSchemes")
            control = path.parent / "controlDict"
            if control.is_file():
                shutil.copy2(control, child / "controlDict")
            kept.append(child / "fvSchemes")
        return kept
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def application_of(fv_schemes: Path) -> str:
    """Which executable a case runs, from its own controlDict.

    Added 2026-08-16 with the stage-1 be-transport gate, which is the first
    case in this repo that does NOT run phaseClotAxisymmetricFluid. Before
    that the solver was a global constant and the check quietly assumed it;
    applied to the new gate it demanded div(sigmaD) of a case that has no
    skeleton, no stress and no clot, and would have failed every build.

    An unreadable or absent controlDict falls back to the clot solver, which
    is the conservative direction: it asks for MORE schemes, not fewer.
    """
    control = fv_schemes.parent / "controlDict"
    if not control.is_file():
        return "phaseClotAxisymmetricFluid"
    text = re.sub(r"//[^\n]*", "", control.read_text(errors="replace"))
    app = re.search(r"^\s*application\s+(\S+)\s*;", text, re.M)
    if app and app.group(1) != "foamRun":
        return app.group(1)
    solver = re.search(r"^\s*solver\s+(\S+)\s*;", text, re.M)
    return solver.group(1) if solver else "phaseClotAxisymmetricFluid"


def requests_by_application(root: Path) -> tuple[dict, list[str]]:
    """Scan every user application once, keyed by the name a case would name.

    The key is the directory name under user-applications/, which is also the
    executable name for a standalone application and the solver name for a
    foamRun module. Both are what appears in a case's controlDict.
    """
    table, problems = {}, []
    for app_dir in sorted((root / "user-applications").iterdir()):
        if not (app_dir / "Make").is_dir():
            continue

        # HEADERS TOO, AND SHARED ONES.
        #
        # This globbed *.C only, and on the first application to put a div
        # call in a header it reported "(none)" and passed -- while the
        # application really did request div(phi,be). A checker that answers
        # "nothing to check" when it cannot see the code is worse than one
        # that errors, because it looks like coverage.
        #
        # The shared directories are read from the app's own Make/options
        # rather than guessed, so the set scanned here is exactly the set the
        # compiler sees.
        dirs = [app_dir]
        options = app_dir / "Make" / "options"
        if options.is_file():
            for rel in re.findall(r"-I\.\./(\S+)", options.read_text()):
                shared = app_dir.parent / rel
                if shared.is_dir():
                    dirs.append(shared)

        requested = set()
        for d in dirs:
            for source in sorted(list(d.glob("*.C")) + list(d.glob("*.H"))):
                found, trouble = scan_solver(source)
                requested |= found
                problems += trouble
        table[app_dir.name] = requested
    return table, problems


def scan_case(fv_schemes: Path) -> tuple[set[str], bool]:
    text = fv_schemes.read_text()
    block = re.search(r"divSchemes\s*\{(.*?)\}", text, re.S)
    if not block:
        return set(), False
    # Strip comments first, and keep the entry pattern on ONE line: `[^;]+`
    # spans newlines, so a comment line without a semicolon swallows the entry
    # underneath it and the check reports a defined scheme as missing.
    body = re.sub(r"//[^\n]*", "", block.group(1))
    entries = set(re.findall(r"^\s*(\S+)\s+[^;\n]+;", body, re.M))
    entries.discard("default")
    default_none = bool(re.search(r"^\s*default\s+none\s*;", body, re.M))
    return entries, default_none


def self_test(solver: Path) -> int:
    """Prove the guard still discriminates code from prose about code.

    Added after the comment-blanking fix: the first version of that fix looked
    like it worked, but its string branch never consumed the closing quote, so
    the parity inverted at the first string literal and NO comment after it was
    blanked. The symptom was a false positive; the same defect could equally
    have hidden a real one. A guard needs its own guard.
    """
    import tempfile

    source = solver.read_text()
    cases = []

    def run(label: str, text: str, expect: bool) -> bool:
        with tempfile.NamedTemporaryFile("w", suffix=".C", delete=False) as handle:
            handle.write(text)
            path = Path(handle.name)
        _, problems = scan_solver(path)
        path.unlink()
        ok = bool(problems) == expect
        cases.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
        return ok

    run("the shipped solver is clean", source, False)

    named = 'fvc::div(sigmaD_ & (I + gradUs), "div(sigmaD)")'
    if named in source:
        run("a real unnamed div is caught",
            source.replace(named, "fvc::div(sigmaD_ & (I + gradUs))", 1), True)
    else:
        print("  FAIL the anchor for the unnamed-div case is gone; fix this test")
        cases.append(False)

    run("an unnamed div inside a block comment is ignored",
        "/* fvc::div(ghost_); */\n" + source, False)
    run("an unnamed div inside a line comment is ignored",
        "// fvc::div(ghost_);\n" + source, False)

    # A DIRECT LOOKUP IS A REQUEST, AND THIS CHECKER USED TO BE BLIND TO IT.
    #
    # phaseFieldFaceWeights() asks with mesh_.schemes().div("div(phiReleased,c)"),
    # which is not an fvc::div or fvm::div call at all. Run 20260913-044404
    # died on that entry in all three arms at step one, against a case this
    # checker had just finished listing OTHER missing schemes for. Both
    # directions are asserted: it must be seen, and the detection must be what
    # sees it.
    def requests(text: str) -> set:
        with tempfile.NamedTemporaryFile("w", suffix=".C", delete=False) as handle:
            handle.write(text)
            path = Path(handle.name)
        requested, _ = scan_solver(path)
        path.unlink()
        return requested

    lookup = 'mesh_.schemes().div("div(phiReleased,c)")'
    if lookup in source:
        seen = "div(phiReleased,c)" in requests(source)
        cases.append(seen)
        print(f"  {'ok  ' if seen else 'FAIL'} a direct schemes().div() "
              f"lookup is counted as a request")
        # EVERY occurrence, not the first. There are two -- the phase-field
        # weights at ~1869 and ~5742 -- and replacing one left the other
        # supplying the name, which made this assertion fail for the right
        # reason and told me the file asks twice.
        blind = "div(phiReleased,c)" not in requests(
            source.replace(lookup, 'mesh_.schemes().div(cName)'))
        cases.append(blind)
        print(f"  {'ok  ' if blind else 'FAIL'} ... and it is the literal "
              f"that is seen, not something else in the file "
              f"({source.count(lookup)} call sites ask for it)")
    else:
        print("  FAIL the anchor for the direct-lookup case is gone; fix this test")
        cases.append(False)

    blanked = blank_cpp_comments(source)
    offsets = len(blanked) == len(source) and blanked.count("\n") == source.count("\n")
    cases.append(offsets)
    print(f"  {'ok  ' if offsets else 'FAIL'} line numbers are preserved")

    print("self-test:", "passed" if all(cases) else "FAILED")
    return 0 if all(cases) else 1


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    source = root / "user-applications" / "phaseClotAxisymmetricFluid" / (
        "phaseClotAxisymmetricFluid.C"
    )
    if not source.is_file():
        print(f"check_fvschemes_coverage.py: solver source not found: {source}")
        return 2

    if "--self-test" in sys.argv[1:]:
        return self_test(source)

    by_app, problems = requests_by_application(root)
    for name in sorted(by_app):
        print(f"{name} requests div schemes: {sorted(by_app[name]) or '(none)'}")

    # Coverage is checked only for the cases named on the command line -- the
    # ones a build is about to ship. A repo-wide sweep is the wrong scope:
    # most older cases never solve the skeleton (fixed-solid sweeps, diffusion
    # and flow verification), so they correctly omit div(sigmaD) and would
    # drown the real signal in 50 false alarms.
    targets = [Path(a) for a in sys.argv[1:]]
    if not targets:
        print(
            "usage: check_fvschemes_coverage.py <case-dir> [<case-dir> ...]\n"
            "  pass the cases the package ships; coverage is not repo-wide."
        )
        return 2

    schemes: list[Path] = []
    for target in targets:
        target = target if target.is_absolute() else (root / target)
        if not target.is_dir():
            print(f"check_fvschemes_coverage.py: not a directory: {target}")
            return 2
        found = sorted(target.glob("**/system/fvSchemes"))
        if not found:
            # The gate cases have no fvSchemes on disk: scripts/create_cases.py
            # writes system/ at run time. That is precisely why nothing offline
            # caught run 20260810-032625, so the generator's own source is the
            # thing to read.
            generator = target / "scripts" / "create_cases.py"
            if not generator.is_file():
                print(
                    f"check_fvschemes_coverage.py: {target} has neither a "
                    f"system/fvSchemes nor a scripts/create_cases.py"
                )
                return 2
            # The root variable is GATE in the gates and CASE in the main
            # case; match either rather than only the one that existed first.
            template = re.search(
                r'TEMPLATE\s*=\s*\w+\.parent\s*/\s*"([^"]+)"\s*/\s*"([^"]+)"',
                generator.read_text(),
            )
            if not template:
                # NO TEMPLATE AT ALL: the generator writes system/ from
                # scratch. The be-transport gate is the first case built that
                # way, and this branch used to `return 2` on it, which would
                # have made every package unbuildable.
                #
                # There is nothing to borrow, so the schemes are GENERATED and
                # scanned -- in a scratch copy, never in the working tree, so
                # the check cannot leave behind children that the package
                # would then ship and hash.
                found = generated_schemes(target)
                if found is None:
                    print(
                        f"check_fvschemes_coverage.py: {target} has no "
                        f"TEMPLATE and its generator produced no fvSchemes"
                    )
                    return 2
                print(f"{target.name}: generated {len(found)} fvSchemes to check")
                schemes += found
                continue
            borrowed = (
                target.parent / template.group(1) / template.group(2)
                / "system" / "fvSchemes"
            )
            if not borrowed.is_file():
                print(f"check_fvschemes_coverage.py: missing {borrowed}")
                return 2
            print(f"{target.name}: copies system/ from {borrowed.parent.parent}")
            found = [borrowed]
        schemes += found

    checked = 0
    unknown = set()
    for path in schemes:
        entries, default_none = scan_case(path)
        if not default_none:
            continue
        checked += 1
        app = application_of(path)
        if app not in by_app:
            unknown.add(app)
            continue
        missing = sorted(by_app[app] - entries)
        if missing:
            problems.append(
                f"{path.relative_to(root)}: runs {app}, whose divSchemes has "
                f"`default none` but does not define {missing}"
            )

    print(f"checked {checked} case fvSchemes with `default none`")
    if unknown:
        problems.append(
            f"cases name an application with no source under "
            f"user-applications/: {sorted(unknown)}. Either the name in "
            f"controlDict is wrong or the application is not in this repo, "
            f"and either way its scheme requests are unchecked."
        )

    if problems:
        print("\nFVSCHEMES COVERAGE FAILED\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("FVSCHEMES COVERAGE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
