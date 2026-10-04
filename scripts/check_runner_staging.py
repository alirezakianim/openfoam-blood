#!/usr/bin/env python3
"""Reproduce the Colab runner's staging offline and generate every child.

Four runs have now died because a generator needed something at run time that
the runner had not put on disk:

  20260810-032625  a divergence scheme no case defined
  20260810-075435  a hard-coded child list that had gone stale
  20260810-192247  the mural gate, which the finite-strain gate imports
  20260810-210833  the mural gate again, which the MAIN CASE imports

Every one of them compiled, passed its dry run, and shipped. The dry runs could
not catch them because each builds its own tidy temporary tree containing
exactly what it knows it needs -- which is precisely the assumption that was
wrong.

This does the opposite. It copies `cases/` the way the runner does, with the
same ignore pattern, and then runs each Allrun's generator step from inside
that tree. If a generator imports a sibling case, reads a control file, or
reaches for a template that staging does not provide, it fails here in two
seconds rather than forty minutes into a session.

It stops short of OpenFOAM: no mesh, no solver. What it proves is that the
cases can be BUILT from what the runner stages.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
# The runner's own pattern. Kept identical on purpose: staging that differs
# from the runner's is testing a tree nobody will ever run.
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "runs", "postProcessing", "log.*"
)


APPLICATION = re.compile(r"^\s*application\s+(\w+)\s*;", re.M)

BASE_REQUIRED = ("child.json", "system/blockMeshDict", "system/controlDict")

# ALLOWLIST, NOT A TEST FOR PHASE-CLOT-NESS. The strict requirement is the
# DEFAULT and an application must be named here to escape it.
#
# The first version of this relaxation asked the opposite question -- "is the
# application phaseClotAxisymmetricFluid?" -- and silently dropped the check
# for the 146 children of AM2020_planar_mural_detachment, because the phase
# clot is a solver MODULE and its controlDict says `application foamRun;`.
# The case the check was written for was the case it stopped checking, and it
# still printed RUNNER STAGING OK.
#
# Written this way round, an application nobody has thought about keeps the
# strict rule and fails loudly, which is a decision someone has to make on
# purpose rather than one that happens by omission.
STANDALONE_GATE_APPS = frozenset({
    "beTransportGate",         # AM2020_be_transport_gate
    "plateletTransportGate",   # AM2020_platelet_transport_gate
})


def application_of(control_dict: Path) -> str | None:
    """The application a child runs, or None if it cannot be determined."""
    if not control_dict.is_file():
        return None
    match = APPLICATION.search(control_dict.read_text(errors="replace"))
    return match.group(1) if match else None


def required_files(app: str | None) -> tuple[str, ...]:
    """What a child running `app` must have on disk before its Allrun."""
    if app in STANDALONE_GATE_APPS:
        return BASE_REQUIRED
    # Everything else -- foamRun, an unreadable controlDict, an application
    # nobody has classified yet -- keeps the strict requirement.
    return BASE_REQUIRED + ("constant/phaseClotProperties",)


def self_test() -> int:
    """The relaxation must not have relaxed the case it was written for.

    THE FIRST VERSION OF THIS SELF-TEST PASSED WHILE THE CHANGE WAS BROKEN,
    because it exercised `application_of()` -- the helper -- and not
    `required_files()`, the branch that actually decides. The main case's
    controlDict says `application foamRun;`, so the check silently stopped
    demanding phaseClotProperties of all 146 of its children and still
    printed OK. These assert the DECISION, and `foamRun` is the first row.
    """
    failures = []
    cases = [
        ("foamRun -- the phase clot is a MODULE, this is the main case",
         "foamRun", True),
        ("phaseClotAxisymmetricFluid named directly", "phaseClotAxisymmetricFluid", True),
        ("an application nobody has classified", "someNewSolver", True),
        ("no controlDict to read at all", None, True),
        ("beTransportGate, an allowlisted standalone gate", "beTransportGate", False),
        ("plateletTransportGate, an allowlisted standalone gate",
         "plateletTransportGate", False),
    ]
    for label, app, wants_props in cases:
        got = "constant/phaseClotProperties" in required_files(app)
        ok = got == wants_props
        print(("ok   " if ok else "FAIL ")
              + "%-58s phaseClotProperties %s" % (
                  label, "required" if got else "not required"))
        if not ok:
            failures.append(label)

    # And the helper, since required_files() is only as good as its input.
    with tempfile.TemporaryDirectory(prefix="am2020-staging-self-") as tmp:
        base = Path(tmp) / "system"
        base.mkdir(parents=True)
        (base / "controlDict").write_text("application     foamRun;\n")
        app = application_of(base / "controlDict")
        ok = app == "foamRun"
        print(("ok   " if ok else "FAIL ")
              + "%-58s app=%r" % ("the application is read from controlDict", app))
        if not ok:
            failures.append("application_of")

        ok = application_of(Path(tmp) / "absent" / "controlDict") is None
        print(("ok   " if ok else "FAIL ")
              + "%-58s" % "a missing controlDict yields None (strict branch)")
        if not ok:
            failures.append("missing controlDict")

    if failures:
        print("self-test FAILED (%d): %s" % (len(failures), "; ".join(failures)))
        return 1
    print("self-test: passed")
    return 0


def main() -> int:
    targets = sys.argv[1:]
    if "--self-test" in targets:
        targets.remove("--self-test")
        rc = self_test()
        if rc or not targets:
            return rc
    if not targets:
        print("usage: check_runner_staging.py <case-name> [<case-name> ...]")
        return 2

    with tempfile.TemporaryDirectory(prefix="am2020-staging-") as tmp:
        staged = Path(tmp) / "run"
        shutil.copytree(ROOT / "cases", staged, ignore=IGNORE)

        failures = []
        for name in targets:
            case = staged / name
            if not case.is_dir():
                failures.append(f"{name}: not present in the staged tree")
                continue
            generator = case / "scripts" / "create_cases.py"
            if not generator.is_file():
                failures.append(f"{name}: no scripts/create_cases.py")
                continue

            done = subprocess.run(
                [sys.executable, str(generator)],
                capture_output=True, text=True, cwd=str(case),
            )
            if done.returncode != 0:
                failures.append(
                    f"{name}: create_cases.py exited {done.returncode}\n"
                    f"      stdout: {done.stdout.strip()[-400:]}\n"
                    f"      stderr: {done.stderr.strip()[-400:]}"
                )
                continue

            matrix = case / "runs" / "matrix.json"
            if not matrix.is_file():
                failures.append(f"{name}: no runs/matrix.json after generation")
                continue

            children = sorted(
                p.name for p in (case / "runs").iterdir() if p.is_dir()
            )
            # Every child must have what its Allrun then needs -- and WHICH
            # files those are depends on the application it runs.
            #
            # This used to demand `constant/phaseClotProperties` of every child
            # of every case, unconditionally. That is right for the phase-clot
            # cases and wrong for every gate that runs its own application:
            # AM2020_be_transport_gate has reported FAILED here since August,
            # and AM2020_platelet_transport_gate would have joined it. A
            # checker that is permanently red on healthy cases teaches its
            # reader to ignore it, which costs more than the check is worth.
            #
            # The application is read from the child's own controlDict rather
            # than assumed, so a case that silently changes application is
            # visible rather than merely re-passing.
            apps = set()
            for child in children:
                root = case / "runs" / child
                app = application_of(root / "system" / "controlDict")
                apps.add(app or "?")
                for item in required_files(app):
                    if not (root / item).is_file():
                        failures.append(f"{name}/{child}: missing {item}")
            print(f"{name}: staged and generated {len(children)} children "
                  f"({', '.join(sorted(apps))})")

        if failures:
            print("\nRUNNER STAGING FAILED\n")
            for failure in failures:
                print(f"  - {failure}")
            return 1

    print("RUNNER STAGING OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
