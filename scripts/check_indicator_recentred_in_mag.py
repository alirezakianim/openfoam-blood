#!/usr/bin/env python3
"""mag() of the phase indicator must be recentred, because ind is not c.

WHY THIS EXISTS
---------------
S6c-2 replaced the phase field c by the indicator ind at 100+ sites using the
exact identity c = 1 - 2*ind. Every site that compared c against a threshold
was migrated correctly, because a comparison visibly has to turn over:

    c < L        becomes     ind > (1 - L)/2

Two sites did not compare -- they took a MAGNITUDE:

    if (mag(cIn[celli]) < 0.9)     ->     if (mag(indIn[celli]) < 0.9)

and mag() has no direction to turn over, so nothing looked wrong. It is
wrong. c is centred on 0 at the interface and ind is centred on 0.5, so
|c| < 0.9 is the BAND 0.05 < ind < 0.95, while mag(ind) < 0.9 is simply
ind < 0.9 -- which includes the entire lumen, where ind = 0.

Measured on run 20260920-220246's own written q field, at three write times,
the broken form selected 29324/29332/29345 of 31360 cells (93.5-93.6%) where
the correct form selects 1664/1678/1711 (5.3-5.5%). The two instruments
built on that mask went vacuous rather than noisy:

  PHASE_CLOT_DAMKOHLER   DaInterfaceMax == DaMax at 272/272 steps in em46,
                         against 1/272 in em45. An "interface maximum" that
                         is identically the global maximum measures nothing.
  PHASE_CLOT_GROWTH_STOICH  VnClosureRef, the normalisation of a
                         pre-registered gate (G6, "VnClosureResidual stays
                         exactly 0"), was taken over 93.6% of the mesh.

Neither feeds physics -- both only reach Info -- which is exactly why a run
can complete, conserve content to 4.3e-22 a step, write byte-identical
fields, and still ship two broken instruments.

WHAT IT CHECKS
--------------
Every mag(...) in the compiled translation units whose argument names the
indicator must also carry a recentring (1 - 2*ind, or 2*ind - 1). A bare
mag(ind...) is a FAILURE. Comments and the argument's own nesting are
handled; string literals are kept but cannot match, since the pattern
requires the call syntax.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "user-applications" / "phaseClotAxisymmetricFluid"

# Identifiers that hold the indicator (ind in [0,1]), not the phase field c.
INDICATOR = re.compile(r"\b(ind|indIn|indHold|indicator|phaseIndicator)\b")
# A recentring back into c's frame: 1 - 2*x, or 2*x - 1, in any spacing.
RECENTRED = re.compile(r"(1\s*\)?\s*-\s*2\s*\*)|(2\s*\*[^;]*?-\s*1\b)|(scalar\(1\)\s*-\s*2\s*\*)")


def code_only(text: str) -> str:
    """Source with // and /* */ comments removed, string literals KEPT.

    The same scanner the sibling checks use. It matters here more than
    usual: the fix's own comment quotes `mag(ind)` verbatim as the thing it
    is warning about, so a scan that reads comments would fail on the
    corrected file and pass once someone deleted the warning.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:min(j + 1, n)])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            out.append(" ")
            i = n if j < 0 else j + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def translation_units(solver: Path) -> list[Path]:
    """The .C files wmake compiles, from Make/files. The program, not a file."""
    files = solver / "Make" / "files"
    if not files.is_file():
        raise SystemExit(f"FAIL: {files} does not exist; cannot enumerate units")
    units = []
    for raw in files.read_text().splitlines():
        line = raw.split("/*")[0].strip()
        if not line.endswith(".C"):
            continue
        p = (solver / line).resolve()
        if not p.is_file():
            raise SystemExit(f"FAIL: Make/files lists {line}, which does not exist")
        units.append(p)
    if not units:
        raise SystemExit(f"FAIL: no .C units parsed from {files}")
    return units


def mag_arguments(code: str):
    """(offset, argument text) for every mag( ... ) call, brace-matched."""
    for m in re.finditer(r"\bmag\s*\(", code):
        depth, i, n = 1, m.end(), len(code)
        while i < n and depth:
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
            i += 1
        yield m.start(), code[m.end():i - 1]


def run_check(solver: Path = SOLVER):
    problems: list[str] = []
    indicator_sites = 0

    for unit in translation_units(solver):
        code = code_only(unit.read_text())
        for off, arg in mag_arguments(code):
            if not INDICATOR.search(arg):
                continue
            indicator_sites += 1
            if RECENTRED.search(arg):
                continue
            line = code.count("\n", 0, off) + 1
            problems.append(
                f"{unit.name}:{line}: mag({arg.strip()[:60]}) takes the "
                f"magnitude of the INDICATOR without recentring. ind is "
                f"centred on 0.5 and c on 0, so this is not the |c| test it "
                f"replaced: it selects everything below the threshold, "
                f"including the lumen at ind = 0. Write "
                f"mag(scalar(1) - 2*<ind>) to restate |c| < L."
            )

    if indicator_sites == 0:
        problems.append(
            "no mag() of the indicator was found at all. The two recentred "
            "sites on record (the Vn closure mask and the Damkohler "
            "interface mask) should both match, so this check is measuring "
            "nothing -- the identifier set or the mag() scanner is stale."
        )

    return problems, indicator_sites


def main() -> int:
    problems, sites = run_check()
    for p in problems:
        print(f"FAIL {p}")
    if problems:
        return 1
    print(
        f"OK: {sites} mag() call(s) on the phase indicator are all recentred "
        f"into c's frame (1 - 2*ind); none takes a bare magnitude of ind."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
