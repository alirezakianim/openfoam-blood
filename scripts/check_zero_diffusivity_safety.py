#!/usr/bin/env python3
"""Every use of D_ in the solver must be safe when D == 0.

WHY THIS EXISTS
---------------
D = 0 is a SUPPORTED configuration: four of the 119 children set it
(gate_a_floor, gate_a_linear, gate_a_nodiff, gate_a_tol), and they are gates,
not curiosities. Under the pre-2026-09-02 gradient closure they always ran.

Adopting Capek (4.6) introduced `Vn = (alpha/D)*kEff*w`, and D = 0 then killed
all four with SIGFPE -- twice, in two separate Colab runs:

  20260902-141852  alpha_/D_ = 0.0/0.0 in updateDerivedFields(), in the
                   CONSTRUCTOR, before step 1.
  20260902-150654  after that fix, harmonic::interpolate(qEffOld*D_) divides
                   BY the field, SIGFPE at step 1 inside solvePlatelets().

Two runs, one class of defect, found one site at a time. `g++ -fsyntax-only`
cannot see either: both are well-typed. There is no local OpenFOAM runtime, so
nothing else catches them before a Colab session is spent.

WHAT THIS DOES
--------------
Pins every line mentioning `D_` to a recorded justification. A NEW use fails
the build until someone states why it is safe at D = 0. It cannot prove safety
-- it forces the question to be answered before a run pays for it.

Matching is on normalised line CONTENT, so ordinary edits that move lines do
not trip it.
"""
import re
import sys
from pathlib import Path

SOLVER = (Path(__file__).resolve().parent.parent / "user-applications"
          / "phaseClotAxisymmetricFluid" / "phaseClotAxisymmetricFluid.C")

# signature -> why it is safe at D == 0
#
# S21 (research/designs/2026-10-05-S21-clean-solver.md) deleted four recorded
# uses with the code around them, so their entries are removed: the Dfloor
# denominator in updateDerivedFields() (release-4198bb17 .C:8716), the two
# `- fvm::laplacian(qEff*D_, w_)` sites of solvePlatelets()'s legacy branches
# (.C:15757, 15771; report 06 hunk a39), and its diffusionMatrix and
# qFloorExtension forms (.C:16033, 16046, 16052). None is in the S21 solver.
ALLOWED = {
    'D_(phaseClotProperties_.lookup("D")),':
        "the lookup itself",
    "if (D_.value() == 0.0)":
        "THE GUARD: refuses alpha != 0 with D == 0 as ill-posed",
    "stoichRatio = alpha_/D_;":
        "GUARDED: only reached when alpha != 0 and D != 0",
    'dimensionedScalar("zero", D_.dimensions(), 0.0)':
        "dimensions only, no arithmetic",
    "if (D_.value() != 0.0)":
        "THE GUARD: skips the harmonic mean entirely at D == 0",
    "harmonic(mesh_).interpolate(qEffOld*D_);":
        "GUARDED: harmonic divides BY the field, so it is skipped entirely at "
        "D == 0 and a zero field is substituted. The D != 0 path still goes "
        "through the original (word, tmp) constructor so its boundary field "
        "is unchanged -- fvc::laplacian(gammaPlateletf, w_) reads it.",
    "- fvc::interpolate(q_)*D_*fvc::snGrad(w_)*mesh.magSf()":
        "multiplication only",
    "const scalar Dv = max(D_.value(), vSmall);":
        "GUARDED: max(D_, vSmall)",
    ": (D_.value() == 0.0)":
        "THE GUARD: diagnostics report nan rather than dividing",
    ": (alpha_/D_).value();":
        "GUARDED: only reached when alpha != 0 and D != 0",
    ": 2.0*(alpha_/D_).value();":
        "GUARDED: only reached when alpha != 0 and D != 0",
}


def normalise(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def main() -> int:
    if not SOLVER.is_file():
        print(f"MISSING: {SOLVER}", file=sys.stderr)
        return 1

    unknown, seen = [], set()
    for n, raw in enumerate(SOLVER.read_text(encoding="utf-8").splitlines(), 1):
        line = normalise(raw)
        if not re.search(r"\bD_\b", line):
            continue
        if line.startswith("//") or line.startswith("*"):
            continue          # prose, including the records above
        match = next((k for k in ALLOWED if normalise(k) in line), None)
        if match is None:
            unknown.append((n, line))
        else:
            seen.add(match)

    if unknown:
        print("ZERO-DIFFUSIVITY SAFETY: unrecorded use of D_\n", file=sys.stderr)
        for n, line in unknown:
            print(f"  {SOLVER.name}:{n}: {line}", file=sys.stderr)
        print(
            "\nD = 0 is a SUPPORTED configuration -- gate_a_floor,"
            "\ngate_a_linear, gate_a_nodiff and gate_a_tol all set it, and this"
            "\nexact class of defect has already cost TWO Colab runs"
            "\n(20260902-141852, 20260902-150654).\n"
            "\nState why this use is safe at D == 0 and record it in ALLOWED in"
            f"\n{Path(__file__).name}. Division by D, or by any field built"
            "\nfrom D -- harmonic interpolation above all -- is NOT safe.",
            file=sys.stderr,
        )
        return 1

    stale = sorted(set(ALLOWED) - seen)
    if stale:
        print("ZERO-DIFFUSIVITY SAFETY: recorded uses that no longer exist\n",
              file=sys.stderr)
        for k in stale:
            print(f"  {k}", file=sys.stderr)
        print("\nRemove them from ALLOWED so the list stays a real inventory.",
              file=sys.stderr)
        return 1

    print(f"ZERO-DIFFUSIVITY SAFETY OK -- {len(seen)} recorded uses of D_")
    return 0


if __name__ == "__main__":
    sys.exit(main())
