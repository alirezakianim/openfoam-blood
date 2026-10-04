#!/usr/bin/env python3
"""Every per-step difference state must be primed in the constructor body.

WHY THIS EXISTS
---------------
Run 20260920-220246 completed 272 steps, reached endTime, stayed bounded, and
was graded INADMISSIBLE:

    "clot volume is not conserved: cumulative drift +100.0% of the initial
     volume (limit 5%)"

The clot was conserved to 4.3e-22 per step. The whole +100.0% came from step
ONE, because S6c-2 had deleted this line from the constructor body

    previousClotVolume_ = fvc::domainIntegrate(scalar(1) - q_).value();

while renaming the member to previousContentVolume_ and keeping its `(0)`
member-initialiser. The emitter computes

    contentDVdt = (contentVolume - previousContentVolume_)/deltaT

and writes the member only at its own end, so the first emission necessarily
reported the entire seeded clot appearing out of nothing in one step:
contentDVdt = V0/dt1 = 4.125519e-10, matching V0/dt1 to seven significant
figures. The assessor weights each residual by its interval, so that one step
contributed 0.999997 of V0 and the remaining 271 summed to -1.2e-11.

The deleted line was collateral: it read `1 - q_`, the retired legacy
indicator form, so it went out with the `c` cleanup instead of being migrated
to theta. Its SIBLING, on the adjacent line, survived --

    previousInventory_ = fvc::domainIntegrate(q_*w_).value();

-- which is what makes this a class rather than a typo: the primes live far
from the emitters that depend on them, and deleting one is silent at compile
time, silent at run time, and shows up only as a physics verdict on a run
that cost an hour of Colab.

WHAT IT CHECKS
--------------
Both sides are DERIVED, so a member added later is covered without editing
this file:

  consumers -- every `(<expr> - <member>_)/deltaT` in any translation unit
               wmake actually compiles, read out of Make/files;
  primes    -- an assignment to that member inside a CONSTRUCTOR BODY
               (a function whose name is `Class::Class`), at brace depth 1,
               which is where the surviving prime sits.

A member differenced per step with no constructor-body prime is a FAILURE.
The member-initialiser `member_(0)` does NOT count: that is exactly the state
the defect left behind.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "user-applications" / "phaseClotAxisymmetricFluid"

# The emitter differences a running scalar against stored state, per step.
DIFF = re.compile(r"-\s*([A-Za-z_][A-Za-z0-9_]*_)\s*\)\s*/\s*deltaT")

# `member_ =` but not `==`, `>=`, `!=`, `+=` ...
def assign_re(member: str) -> re.Pattern[str]:
    return re.compile(r"(?<![=!<>+\-*/%&|^])\b" + re.escape(member) + r"\s*=(?!=)")


def code_only(text: str) -> str:
    """Source with // and /* */ comments removed, string literals KEPT.

    Same scanner as check_registry_lookups_resolve.py. Comments here quote the
    deleted prime verbatim -- including this file's own docstring in the
    solver's comment block -- so a plain substring test would read the
    epitaph as the live line.
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
    """The .C files wmake compiles, read out of Make/files -- not guessed.

    The defect S6c-2 shipped before this one lived in the SECOND unit, so a
    scan that finds only the big one is not measuring.
    """
    files = solver / "Make" / "files"
    if not files.is_file():
        raise SystemExit(f"FAIL: {files} does not exist; cannot enumerate units")
    units: list[Path] = []
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


def constructor_bodies(code: str, class_name: str) -> list[tuple[int, int]]:
    """(start, end) offsets of each `Class::Class(...)` definition body.

    Located by the qualified name, then the body is taken by walking braces
    from the first `{` that follows. The member-initialiser list sits between
    the `)` and that `{`, so anything it contains is outside the returned
    range -- which is the point: `member_(0)` must not be mistaken for a
    prime.
    """
    bodies: list[tuple[int, int]] = []
    sig = re.compile(re.escape(class_name) + r"::" + re.escape(class_name) + r"\s*\(")
    for m in sig.finditer(code):
        brace = code.find("{", m.end())
        if brace < 0:
            continue
        depth, i, n = 0, brace, len(code)
        while i < n:
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    bodies.append((brace, i))
                    break
            i += 1
    return bodies


def run_check(solver: Path = SOLVER, class_name: str = "phaseClotAxisymmetricFluid"):
    problems: list[str] = []
    differenced: dict[str, str] = {}   # member -> unit name where differenced
    primed: set[str] = set()
    bodies_total = 0

    for unit in translation_units(solver):
        code = code_only(unit.read_text())
        for m in DIFF.finditer(code):
            differenced.setdefault(m.group(1), unit.name)
        bodies = constructor_bodies(code, class_name)
        bodies_total += len(bodies)
        for start, end in bodies:
            body = code[start:end]
            # collect every assignment target in the ctor body, once
            for a in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*_)\s*=(?!=)", body):
                primed.add(a.group(1))

    # ---- vacuity guards: this check must be able to FAIL -----------------
    if len(differenced) < 2:
        problems.append(
            f"only {len(differenced)} per-step difference state(s) found "
            f"(expected at least the 2 on record: previousInventory_ and "
            f"previousContentVolume_). The pattern stopped matching, so this "
            f"check is measuring nothing."
        )
    if bodies_total == 0:
        problems.append(
            f"no `{class_name}::{class_name}(` constructor body was located, "
            f"so no prime could be found and every member would be reported "
            f"unprimed. The locator is broken, not the source."
        )

    if not problems:
        for member, unit in sorted(differenced.items()):
            if member not in primed:
                problems.append(
                    f"{unit}: `{member}` is differenced per step as "
                    f"(x - {member})/deltaT but is never assigned in a "
                    f"constructor body. Its member-initialiser leaves it at 0, "
                    f"so the FIRST step reports the whole accumulated quantity "
                    f"as a one-step rate and the conservation gate reads "
                    f"~100% drift on a perfectly conserved run."
                )

    return problems, differenced, primed


def main() -> int:
    problems, differenced, primed = run_check()
    for p in problems:
        print(f"FAIL {p}")
    if problems:
        return 1
    print(
        f"OK: {len(differenced)} per-step difference state(s) "
        f"({', '.join(sorted(differenced))}) are each primed in a constructor "
        f"body; {len(primed)} constructor-body assignments scanned."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
