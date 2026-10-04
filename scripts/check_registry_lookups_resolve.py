#!/usr/bin/env python3
"""Every field name looked up in the object registry must be a field the
library actually creates.

WHY THIS EXISTS.

Run 20260920-201014 -- the whole of batch 44 -- died at its first mechanics
step with

    --> FOAM FATAL ERROR:
    request for volScalarField c from objectRegistry region0 failed
    available objects of type volScalarField are 52 ( ... )
    #4  Foam::clotWallAdhesionFvPatchVectorField::updateCoeffs()

S6c-2 deleted `c` from phaseClotAxisymmetricFluid.C. It did not touch
clotWallAdhesionFvPatchVectorField.C, which is the SECOND translation unit of
the same library -- named on the next line of Make/files, compiled by the same
wmake, and the only place in the whole library that talks to the object
registry at all. Its `cName_` defaulted to "c" and fed an unguarded
`patch().lookupPatchField<volScalarField, scalar>(cName_)`.

Nothing saw it. The S6c-2 consumer map enumerated 107 sites by LINE NUMBER in
phaseClotAxisymmetricFluid.C; a site in a sibling file has no line number in
that file. `typecheck_solver.sh` compiles both TUs and passed both -- a
registry lookup is a runtime string, and the front end cannot know which
fields exist. `dry_run.py` never starts OpenFOAM. And no check script in
scripts/ mentioned clotWallAdhesion at all: every one of them scanned exactly
one file.

The lookup was also REACHABLE ONLY LATE: updateCoeffs() returns early until
`db().foundObject<volSymmTensorField>(sigmaName_)` is true, and sigmaElastic
enters the registry only at the first mechanics solve. So the run built, ran
its whole configuration banner, completed a platelet step and a permeability
step, and then died. A crash at construction would have been cheaper.

THE RULE, derived rather than listed.

  produced   every name the library creates, taken from the `IOobject("NAME"`
             literals across all of its sources.
  consumed   every name reaching a registry accessor -- lookupObject,
             lookupObjectRef, lookupPatchField, foundObject -- whether the
             name is a literal at the call site or a `word` member resolved
             back to the literals it can hold.

consumed must be a subset of produced. A name that is neither is a run that
dies the moment that line executes.

The dictionary side is checked too: a case may override a name key (the BC
takes `q` and `sigma`), so any value a case or generator supplies for one of
those keys must also be a name the library produces -- and the keys S6c-2
retired (`c`, `clotThreshold`) must appear in no dictionary at all.

Run: python3 scripts/check_registry_lookups_resolve.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "user-applications" / "phaseClotAxisymmetricFluid"
SELF = Path(__file__).resolve()

# Fields the library looks up but does not create, with the reason. Every
# entry is a claim that something OUTSIDE this library registers the name;
# an empty list is the honest state today and the list stays empty unless a
# real external producer is named here.
EXTERNALLY_PRODUCED: dict[str, str] = {}

# Keys a patch field reads as a FIELD NAME from its own dictionary entry.
# Derived below from lookupOrDefault<word>(...) in the library's sources;
# this is only the retired set that must now appear nowhere.
RETIRED_NAME_KEYS = {
    "c": "S6c-2 deleted the phase field c; the condition reads q",
    "clotThreshold": "the threshold is now clotThresholdQ, in q units",
}

ACCESSORS = ("lookupObject", "lookupObjectRef", "lookupPatchField", "foundObject")


def code_only(text: str) -> str:
    """Source with // and /* */ comments removed, string literals KEPT.

    Same scanner as check_dictionary_read_forms.py, and for the same reason:
    the names live inside literals, and a `//` inside a literal must not
    truncate the line. Prose cannot be mistaken for a call site because every
    pattern below requires the call syntax around the literal.
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


def translation_units() -> list[Path]:
    """The .C files wmake compiles, read out of Make/files -- not guessed.

    A file that is in the directory but not in Make/files is not part of the
    program; a file that is in Make/files and not on disk is a broken build.
    Both are reported rather than skipped.
    """
    files = SOLVER / "Make" / "files"
    if not files.is_file():
        raise SystemExit(f"FAIL: {files} is missing; cannot enumerate the library")
    units: list[Path] = []
    for line in files.read_text().splitlines():
        line = line.split("/*")[0].strip()
        if not line or line.startswith("LIB") or "=" in line:
            continue
        p = SOLVER / line
        if not p.is_file():
            raise SystemExit(f"FAIL: Make/files names {line}, which is not on disk")
        units.append(p)
    if len(units) < 2:
        raise SystemExit(
            "FAIL: Make/files yielded %d translation unit(s). This check exists "
            "because a defect hid in the SECOND one; a parse that finds one is "
            "not measuring." % len(units)
        )
    return units


def library_sources() -> list[Path]:
    """Every source the library compiles, plus the headers it compiles them
    against. The headers matter: a lookup can live in an inline method."""
    srcs = translation_units()
    srcs += sorted(SOLVER.glob("*.H"))
    for inc in ("beTransport", "plateletContent"):
        d = ROOT / "user-applications" / inc
        srcs += sorted(d.glob("*.H")) + sorted(d.glob("*.C"))
    return srcs


def produced_names(srcs: list[Path]) -> dict[str, Path]:
    """name -> first source that constructs an IOobject with it."""
    out: dict[str, Path] = {}
    pat = re.compile(r"IOobject\s*\(\s*\"([A-Za-z_][A-Za-z0-9_]*)\"")
    for p in srcs:
        for name in pat.findall(code_only(p.read_text())):
            out.setdefault(name, p)
    return out


def word_member_literals(src: str) -> dict[str, set[str]]:
    """`word` member -> every literal it can be initialised to.

    Two forms, both present in this library:
        xName_("lit")                                  default constructor
        xName_(dict.lookupOrDefault<word>("key","lit")) dictionary constructor
    A copy constructor `xName_(ptf.xName_)` adds nothing and is skipped.
    """
    out: dict[str, set[str]] = {}
    for member, lit in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*Name_)\s*\(\s*\"([^\"]*)\"\s*\)", src
    ):
        out.setdefault(member, set()).add(lit)
    for member, _key, lit in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*Name_)\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\.\s*"
        r"lookupOrDefault\s*<\s*word\s*>\s*\(\s*\"([^\"]+)\"\s*,\s*\"([^\"]*)\"",
        src,
    ):
        out.setdefault(member, set()).add(lit)
    return out


def name_keys(src: str) -> dict[str, str]:
    """dictionary key -> default value, for keys read as a `word`."""
    return {
        key: lit
        for _m, key, lit in re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*Name_)\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\.\s*"
            r"lookupOrDefault\s*<\s*word\s*>\s*\(\s*\"([^\"]+)\"\s*,\s*\"([^\"]*)\"",
            src,
        )
    }


def consumed_names(srcs: list[Path]) -> tuple[list[tuple], list[tuple]]:
    """(resolved, unresolved) registry accesses across the library.

    resolved   (path, accessor, name, how)
    unresolved (path, accessor, argument-text)  -- a FAILURE, never a skip:
               an argument this cannot resolve is a name this cannot check.
    """
    resolved: list[tuple] = []
    unresolved: list[tuple] = []
    # accessor < template args > ( argument )
    pat = re.compile(
        r"\b(" + "|".join(ACCESSORS) + r")\s*<[^>]*>\s*\(\s*([^),]+?)\s*\)"
    )
    for p in srcs:
        src = code_only(p.read_text())
        members = word_member_literals(src)
        for accessor, arg in pat.findall(src):
            arg = arg.strip()
            m = re.fullmatch(r'"([A-Za-z_][A-Za-z0-9_]*)"', arg)
            if m:
                resolved.append((p, accessor, m.group(1), "literal"))
                continue
            if arg in members:
                for lit in sorted(members[arg]):
                    resolved.append((p, accessor, lit, f"{arg} default"))
                continue
            unresolved.append((p, accessor, arg))
    return resolved, unresolved


# An entry inside a clotWallAdhesion block. The leading alternation lets the
# same pattern read a real dictionary line AND a generator's emitted string
# literal, `"        sigma            {sigma};\n"`, where the line begins with
# a quote rather than whitespace. Case dictionaries for this BC are all
# GENERATED -- there is no static 0/Us carrying the block anywhere in cases/ --
# so a pattern anchored to the start of a line would have had nothing to read.
BLOCK_ENTRY = re.compile(r"""(?:^|["'])[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]+([^;{}"']+?)[ \t]*;""")
PLACEHOLDER = re.compile(r"[{%]")


def adhesion_blocks(text: str) -> list[list[str]]:
    """The 25 lines after each line that declares the BC type."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if "type" in line and "clotWallAdhesion" in line:
            out.append(lines[i : min(i + 25, len(lines))])
    return out


def dictionary_side(keys: dict[str, str]) -> tuple[list[str], int, int]:
    """(problems, blocks seen, resolvable name-key entries seen).

    The counts are returned so a scan that reads nothing is VISIBLE. This arm
    read zero entries on its first draft -- the pattern was anchored to the
    start of a line and every block in the repo is emitted from inside a
    Python string literal -- and an arm that reads nothing cannot fail.
    """
    problems: list[str] = []
    blocks = 0
    entries = 0
    for base in ("cases", "scripts"):
        for p in sorted((ROOT / base).rglob("*")):
            # This file quotes the BC name in its own fixtures; scanning it
            # would report the self-tests as defects.
            if not p.is_file() or "/runs/" in str(p) or p == SELF:
                continue
            try:
                txt = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if "clotWallAdhesion" not in txt:
                continue
            for block in adhesion_blocks(txt):
                blocks += 1
                for line in block:
                    m = BLOCK_ENTRY.search(line)
                    if not m:
                        continue
                    key, value = m.group(1), m.group(2).strip()
                    if key in RETIRED_NAME_KEYS:
                        problems.append(
                            f"{p.relative_to(ROOT)}: a clotWallAdhesion block "
                            f"still sets `{key}` -- {RETIRED_NAME_KEYS[key]}"
                        )
                    elif key in keys and not PLACEHOLDER.search(value):
                        entries += 1
                        problems.append(f"NAMEKEY {p.relative_to(ROOT)} {key}={value}")
    return problems, blocks, entries


def run_check() -> list[str]:
    srcs = library_sources()
    produced = produced_names(srcs)
    if len(produced) < 20 or "q" not in produced:
        return [
            "FAIL: the produced-name scan returned %d names%s. The library "
            "creates dozens and `q` is one of them, so the parser is broken "
            "and a pass here would mean nothing."
            % (len(produced), "" if "q" in produced else " and no `q`")
        ]
    resolved, unresolved = consumed_names(srcs)
    problems: list[str] = []
    for p, accessor, arg in unresolved:
        problems.append(
            f"{p.name}: {accessor}(...) argument `{arg}` could not be resolved "
            f"to a field name; this check cannot see what it looks up"
        )
    for p, accessor, name, how in resolved:
        if name in produced or name in EXTERNALLY_PRODUCED:
            continue
        problems.append(
            f"{p.name}: {accessor} asks the registry for `{name}` ({how}), "
            f"which no source in this library creates. The run dies the first "
            f"time that line executes."
        )
    keys: dict[str, str] = {}
    for p in srcs:
        keys.update(name_keys(code_only(p.read_text())))
    items, blocks, entries = dictionary_side(keys)
    if blocks == 0:
        problems.append(
            "the dictionary scan found no clotWallAdhesion block anywhere in "
            "cases/ or scripts/. It found several when written, so either the "
            "BC was renamed or the scan is broken; either way this arm is not "
            "measuring."
        )
    for item in items:
        if item.startswith("NAMEKEY "):
            _, loc, kv = item.split(" ", 2)
            key, value = kv.split("=", 1)
            if value not in produced and value not in EXTERNALLY_PRODUCED:
                problems.append(
                    f"{loc}: `{key} {value};` names a field the library never "
                    f"creates"
                )
        else:
            problems.append(item)
    if not problems:
        print(
            "OK: %d registry accesses across %d sources resolve to the %d fields "
            "the library creates; %d clotWallAdhesion block(s) in cases/ and "
            "scripts/ name no retired key (%d literal name entr%s checked)."
            % (len(resolved), len(srcs), len(produced), blocks, entries,
               "y" if entries == 1 else "ies")
        )
    return problems


# ---------------------------------------------------------------- self-tests

PREFIX = (
    'IOobject("q"); IOobject("sigmaD"); IOobject("Us"); '
    + " ".join(f'IOobject("f{i}");' for i in range(30))
)


def selftests() -> list[str]:
    fails: list[str] = []

    def check(label: str, src: str, want_names: set[str]) -> None:
        resolved, unresolved = _scan_text(src)
        got = {n for _a, n, _h in resolved}
        if got != want_names or unresolved:
            fails.append(
                f"self-test {label}: names {sorted(got)} "
                f"unresolved {unresolved} (wanted {sorted(want_names)})"
            )

    # 1. The form that killed batch 44 is seen, through the member default.
    check(
        "pre-fix BC resolves cName_ to c",
        'cName_("c"), sigmaName_("sigmaD"),'
        ' f = db().foundObject<volSymmTensorField>(sigmaName_);'
        ' x = patch().lookupPatchField<volScalarField, scalar>(cName_);',
        {"c", "sigmaD"},
    )
    # 2. The fixed form resolves to q instead.
    check(
        "fixed BC resolves qName_ to q",
        'qName_("q"), sigmaName_("sigmaD"),'
        ' f = db().foundObject<volSymmTensorField>(sigmaName_);'
        ' x = patch().lookupPatchField<volScalarField, scalar>(qName_);',
        {"q", "sigmaD"},
    )
    # 3. A dictionary override literal is picked up as well as the default.
    check(
        "lookupOrDefault<word> default is collected",
        'qName_(dict.lookupOrDefault<word>("q", "q")),'
        ' x = db().foundObject<volScalarField>(qName_);',
        {"q"},
    )
    # 4. A commented-out lookup is not a lookup.
    check(
        "a commented lookup is not counted",
        'qName_("q"); // x = db().lookupObject<volScalarField>("cOld");',
        set(),
    )
    # 5. An unresolvable argument is reported, not silently dropped.
    _r, unres = _scan_text(
        'x = db().lookupObject<volScalarField>(someExpression(i));'
    )
    if not unres:
        fails.append("self-test 5: an unresolvable argument was silently dropped")
    # 6. A literal naming a field nothing creates is a problem.
    probs = _problems_for(PREFIX + ' x = db().lookupObject<volScalarField>("c");')
    if not any("`c`" in p for p in probs):
        fails.append("self-test 6: a lookup of an uncreated name did not fail")
    # 7. ...and the same lookup passes once the field is created.
    probs = _problems_for(
        PREFIX + ' IOobject("c"); x = db().lookupObject<volScalarField>("c");'
    )
    if probs:
        fails.append(f"self-test 7: a created name still failed: {probs}")
    # 8. The retired key is caught inside a generator's emitted string, which
    #    is the only shape a clotWallAdhesion block takes in this repo.
    gen = (
        '            "        type             clotWallAdhesion;\\n"\n'
        '            "        adhesiveStrength 0.4;\\n"\n'
        '            "        c                cOld;\\n"\n'
    )
    if not any(
        "`c`" in x for x in _retired_in(gen)
    ):
        fails.append("self-test 8: a retired key in an emitted block was missed")
    # 9. ...and the same block without it is clean.
    if _retired_in(gen.replace('            "        c                cOld;\\n"\n', "")):
        fails.append("self-test 9: a clean emitted block was flagged")
    return fails


def _retired_in(text: str) -> list[str]:
    out = []
    for block in adhesion_blocks(text):
        for line in block:
            m = BLOCK_ENTRY.search(line)
            if m and m.group(1) in RETIRED_NAME_KEYS:
                out.append(f"`{m.group(1)}`")
    return out


def _scan_text(src: str) -> tuple[list[tuple], list[tuple]]:
    src = code_only(src)
    members = word_member_literals(src)
    pat = re.compile(
        r"\b(" + "|".join(ACCESSORS) + r")\s*<[^>]*>\s*\(\s*([^),]+?)\s*\)"
    )
    resolved, unresolved = [], []
    for accessor, arg in pat.findall(src):
        arg = arg.strip()
        m = re.fullmatch(r'"([A-Za-z_][A-Za-z0-9_]*)"', arg)
        if m:
            resolved.append((accessor, m.group(1), "literal"))
        elif arg in members:
            for lit in sorted(members[arg]):
                resolved.append((accessor, lit, f"{arg} default"))
        else:
            unresolved.append((accessor, arg))
    return resolved, unresolved


def _problems_for(src: str) -> list[str]:
    produced = set(re.findall(r'IOobject\s*\(\s*"([A-Za-z0-9_]+)"', code_only(src)))
    resolved, _u = _scan_text(src)
    return [
        f"asks for `{n}`"
        for _a, n, _h in resolved
        if n not in produced and n not in EXTERNALLY_PRODUCED
    ]


def main() -> int:
    fails = selftests()
    if fails:
        for f in fails:
            print("FAIL " + f)
        print("=== the check itself is broken; its verdict on the library means "
              "nothing ===")
        return 1
    problems = run_check()
    for p in problems:
        print("FAIL " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
