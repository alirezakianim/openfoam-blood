#!/usr/bin/env python3
"""Every gate must give a field the SAME dimensions as every other gate.

WHY THIS EXISTS. Gate 5's template was hand-written, and `muC` was given
`[0 0 0 0 0 0 0]` where every other gate in the repository has
`[0 -2 0 0 0 0 0]`. Nothing local was wrong: the file parsed, the case meshed,
the fields initialised, `blockMesh` and `writeCellCentres` succeeded. The run
reached the first timestep on Colab and died inside `solvePhaseField()` with

    --> FOAM FATAL ERROR:
    Different dimensions for =
         dimensions : [] = [length^-2]

Gate 3b-ii-c already carries `template_dimensions()`, whose docstring reads
"read a field's dimensions from the template; never restate them", and a
`verify_child_field_dimensions()` guard built on it. Both are scoped to ONE
gate: they check that a gate's CHILDREN match ITS OWN template. Neither can see
a template that is itself wrong, and gate 5's was.

So the comparison here is ACROSS gates. A field name that appears in several
gates has one right answer, and the gates that have actually run are the
evidence for it. This catches a new gate written by hand, which is exactly how
the defect arose - the tenth instance of the "restated by hand" class, and the
first in a file OpenFOAM parses rather than one this project's Python does.

A field appearing in only one gate cannot be checked this way and is reported
as unverifiable rather than passed silently.
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

# Fields the SOLVER declares, with the dimensions it declares them with
# (phaseClotAxisymmetricFluid.C). These are authoritative regardless of how
# many gates write them, and are the reason `Us`, `Dg` and `H` are checkable
# even though only gate 5 currently ships them in `0/`.
SOLVER_DIMENSIONS = {
    "Us": "[0 1 0 0 0 0 0]",     # dimLength
    "Dg": "[0 0 0 0 0 0 0]",     # dimless
    "H": "[1 -1 -2 0 0 0 0]",    # dimPressure
}

DIMENSIONS = re.compile(r"(?m)^\s*dimensions\s+(\[[^\]]*\])\s*;")


def normalise(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    seen = collections.defaultdict(dict)     # field -> dims -> [where]

    for field in sorted((root / "cases").glob("*/template/0/*")):
        if field.is_dir() or field.name.startswith("."):
            continue
        matched = DIMENSIONS.search(field.read_text(errors="replace"))
        if not matched:
            continue
        gate = field.parents[2].name
        seen[field.name].setdefault(normalise(matched.group(1)), []).append(gate)

    problems, unverifiable, checked = [], [], 0
    for name, by_dims in sorted(seen.items()):
        authority = SOLVER_DIMENSIONS.get(name)
        if authority is not None:
            authority = normalise(authority)
            for dims, gates in sorted(by_dims.items()):
                if dims != authority:
                    for gate in gates:
                        problems.append(
                            f"0/{name}: {gate} has {dims}, but the SOLVER "
                            f"declares {authority}"
                        )
            checked += 1
            continue
        if len(by_dims) == 1:
            (dims, gates), = by_dims.items()
            if len(gates) == 1:
                unverifiable.append(f"0/{name}: only {gates[0]} defines it")
            else:
                checked += 1
            continue
        # Disagreement. The majority is the evidence; name the minority.
        ranked = sorted(by_dims.items(), key=lambda kv: -len(kv[1]))
        (majority, majority_gates), *rest = ranked
        for dims, gates in rest:
            for gate in gates:
                problems.append(
                    f"0/{name}: {gate} has {dims}, but {len(majority_gates)} "
                    f"other gate(s) have {majority} "
                    f"(e.g. {majority_gates[0]})"
                )
        checked += 1

    for note in unverifiable:
        print(f"  unverifiable (single definition): {note}")
    if problems:
        print("FIELD DIMENSION MISMATCH:")
        for p in sorted(set(problems)):
            print("  " + p)
        print("\n  A dimension is not a local choice. Copy it from a gate that "
              "has run, or from the solver's own declaration; do not restate "
              "it by hand.")
        return 1
    print(f"field dimension check: {checked} field name(s) consistent across "
          f"all gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
