#!/usr/bin/env python3
"""A key written DIMENSIONED must not be read as a bare scalar.

WHY THIS EXISTS.

Run 20260920-190025 -- the whole of batch 43 -- died at construction with

    --> FOAM FATAL IO ERROR:
    wrong token type - expected Scalar, found on line 19 the punctuation
    token '[' ... constant/phaseClotProperties/mobility

The solver was healthy. The case was healthy. The defect was that S6c-2
deleted the `dimensionedScalar mobility_` member and replaced its read with

    mag(readScalar(phaseClotProperties_.lookup("mobility"))) > vSmall

to refuse a nonzero mobility -- while every generator on record writes the
entry DIMENSIONED, `mobility [0 4 -1 0 0 0 0] 0;`. `readScalar` reaches the
'[' and aborts before the guard can decide anything. A refusal that cannot
be reached is not a refusal; here it was a fatal on every case in existence.

Nothing local could see it. `typecheck_solver.sh` compiles the expression
happily -- `readScalar(ITstream&)` is a valid call whatever the tokens turn
out to be. `dry_run.py` never runs OpenFOAM. The token type is a property of
the DICTIONARY, so the check has to hold the source and the dictionary
against each other, which is what this does.

THE RULE, derived rather than listed.

  bare-scalar readers   readScalar / readLabel / lookup<T> / lookupOrDefault<T>
                        -- these parse the entry's tokens DIRECTLY and die on
                        a leading '[' .
  tolerant readers      construction of a dimensioned<T> from the entry's
                        stream: dimensioned<T>::initialise() reads an optional
                        name, optional units, then the value, so it accepts
                        both `k 1;` and `k [0 1 0 0 0 0 0] 1;`.

Every key a template writes with a `[...]` dimension set must be read the
tolerant way. This enumerates both sides -- the solver's read sites, and
every `constant/phaseClotProperties` template under cases/ -- and fails on
the intersection.

Run: python3 scripts/check_dictionary_read_forms.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "user-applications" / "phaseClotAxisymmetricFluid"

# A dimensioned entry at the start of a line: `name [d d d d d d d] value;`
DIMENSIONED_ENTRY = re.compile(
    r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s+\[[^\]]*\]\s*[^;]*;"
)


def code_only(text: str) -> str:
    """Source with // and /* */ comments removed, string literals KEPT.

    The literals have to survive: the key is inside one, `lookup("mobility")`.
    So the comment stripper is a real scanner rather than a regex -- a `//`
    inside a string literal must not truncate the line, or a read site
    sharing a line with such a string becomes invisible.

    Prose cannot be mistaken for a read site because every pattern below
    requires the call syntax around the literal, not the literal alone.
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


def bare_scalar_reads(src: str) -> dict[str, set[str]]:
    """key -> the read forms that parse its tokens directly."""
    found: dict[str, set[str]] = {}

    def add(key: str, form: str) -> None:
        found.setdefault(key, set()).add(form)

    # readScalar(dict.lookup("k")) / readLabel(...) / readBool(...)
    for fn, key in re.findall(
        r"\bread(Scalar|Label|Bool|Int|Word)\s*\(\s*[A-Za-z_][A-Za-z0-9_]*"
        r"\s*\.\s*lookup\s*\(\s*\"([^\"]+)\"",
        src,
    ):
        add(key, f"read{fn}(...lookup)")

    # dict.lookup<scalar>("k") and dict.lookupOrDefault<scalar>("k", ...)
    for call, targ, key in re.findall(
        r"\.\s*(lookup|lookupOrDefault|lookupOrAddDefault)\s*<\s*([A-Za-z:]+)"
        r"\s*>\s*\(\s*\"([^\"]+)\"",
        src,
    ):
        # A dimensioned<T> template argument is tolerant; a raw POD is not.
        if targ.startswith("dimensioned"):
            continue
        add(key, f"{call}<{targ}>")

    return found


def dimensioned_keys() -> dict[str, list[Path]]:
    """key -> the template files that write it with a dimension set."""
    out: dict[str, list[Path]] = {}
    for path in sorted((ROOT / "cases").rglob("constant/phaseClotProperties")):
        # runs/ holds generated copies; packages/ and preflight/ are excluded
        # from the tree walk by living outside cases/.
        if "/runs/" in str(path):
            continue
        for key in DIMENSIONED_ENTRY.findall(path.read_text(errors="replace")):
            out.setdefault(key, []).append(path.relative_to(ROOT))
    return out


def violations(src: str) -> list[str]:
    reads = bare_scalar_reads(src)
    written = dimensioned_keys()
    bad = []
    for key in sorted(set(reads) & set(written)):
        forms = ", ".join(sorted(reads[key]))
        where = written[key][0]
        bad.append(
            f"{key}: read as a bare scalar ({forms}) but written "
            f"dimensioned in {where} "
            f"(+{len(written[key]) - 1} more)"
        )
    return bad


def self_test() -> int:
    """The negative test: the exact line that lost run 20260920-190025."""
    failures = 0
    probe = (
        'if (phaseClotProperties_.found("mobility")\n'
        ' && mag(readScalar(phaseClotProperties_.lookup("mobility")))'
        " > vSmall)\n"
    )
    if not violations(code_only(probe)):
        print("  SELF-TEST FAIL: the readScalar form that killed batch 43 "
              "is not detected")
        failures += 1
    tolerant = (
        "const dimensionedScalar mobilitySet\n"
        '(\n    phaseClotProperties_.lookup("mobility")\n);\n'
    )
    if violations(code_only(tolerant)):
        print("  SELF-TEST FAIL: the dimensioned-construction form is "
              "reported as a violation")
        failures += 1
    commented = '// mag(readScalar(phaseClotProperties_.lookup("mobility")))\n'
    if violations(code_only(commented)):
        print("  SELF-TEST FAIL: a commented-out read is read as live code")
        failures += 1
    message = '<< "mobility is nonzero, set mobility to 0" << exit(x);\n'
    if violations(code_only(message)):
        print("  SELF-TEST FAIL: a refusal message is read as a read site")
        failures += 1
    return failures


def main() -> int:
    written = dimensioned_keys()
    if not written:
        print("FAIL: no dimensioned entries found in any template -- the "
              "enumeration is empty, so the check cannot fire")
        return 1

    failures = self_test()
    if failures:
        print("SELF-TEST FAILED -- the check is not measuring anything")
        return 1

    for path in sorted(SOLVER.glob("*.C")) + sorted(SOLVER.glob("*.H")):
        bad = violations(code_only(path.read_text(errors="replace")))
        for line in bad:
            print(f"FAIL {path.name}: {line}")
            failures += 1

    if failures:
        print(
            "\nA dimensioned entry cannot be parsed by readScalar/lookup<T>. "
            "Read it as a dimensioned<T> built from the entry's stream, or "
            "stop writing the dimension set."
        )
        return 1

    print(
        f"OK: {len(written)} dimensioned keys across the templates; no "
        f"bare-scalar read of any of them in the live solver."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
