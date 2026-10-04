#!/usr/bin/env python3
"""Every field in 0/ must give every mesh patch a boundary condition that exists.

WHY THIS EXISTS. A new geometry brings new patches, and a new boundary
condition brings a new type name. Neither can be checked by running
OpenFOAM here: there is no blockMesh and no solver on the build machine. Both
fail only at construction, in a paid Colab session:

  * a patch the mesh has but a field does not mention  -> FatalIOError
  * a type OpenFOAM 13 does not have (the mural gate records `parabolicVelocity`
    as exactly this trap: "would fatal at run time, not at build time")
  * a constraint patch (empty / wedge / symmetryPlane / cyclic) whose field
    entry is not the matching constraint type
  * an entry for a patch the mesh does not have -- the template's patch names
    instead of the mesh's (see check_stage2_case_generation.py,
    "the patch names match the MESH, not the shared template")

WHAT IT CHECKS, per generated child (a directory holding 0/ plus
system/blockMeshDict or constant/polyMesh/boundary):
  1. the patch list and patch types, from the mesh description;
  2. every file in 0/ that has a boundaryField: each patch is matched by an
     exact name, a quoted regex, or its patch-type group ("wall");
  3. each matched entry's `type` exists: a TypeName() in an OpenFOAM 13
     patch-field source file, or in user-applications/;
  4. constraint patches carry their constraint type;
  5. no entry names a patch that does not exist.

WHAT IT DOES NOT CHECK: which fields the solver requires in 0/ (the solver
reads U and p through the OpenFOAM base solver, so its own MUST_READ sites do
not enumerate them -- check_refused_dictionary_values.py and
check_registry_lookups_resolve.py cover construction), nor whether a
condition is PHYSICALLY right. A dictionary that uses #include, an entry-level
$macro or a $type inside the parsed sections is reported UNVERIFIABLE, never
passed; `value $internalField;` is allowed.

Usage:
  python3 -B scripts/check_case_boundaries.py                 # every cases/*/runs/*/
  python3 -B scripts/check_case_boundaries.py <dir> [<dir>...] # a child, a runs/ dir, or a case
Exit 0 only if at least one child was checked and none failed.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OF_SRC = Path("/Volumes/OpenFOAM-13/source/src")
USER_SRC = ROOT / "user-applications"
CONSTRAINT = {"empty", "wedge", "symmetryPlane", "symmetry", "cyclic", "cyclicAMI", "processor"}


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def block_after(text: str, start: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    """Return the text inside the bracket that opens at or after `start`."""
    i = text.index(open_ch, start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == open_ch:
            depth += 1
        elif text[j] == close_ch:
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
    raise ValueError(f"unbalanced {open_ch}{close_ch}")


def named_blocks(body: str) -> list[tuple[str, str]]:
    """Top-level `name { ... }` entries of a dictionary body."""
    out, i = [], 0
    pat = re.compile(r'\s*("(?:[^"\\]|\\.)*"|[A-Za-z0-9_.:|()*+?\[\]-]+)\s*\{')
    while True:
        m = pat.search(body, i)
        if not m:
            break
        inner, end = block_after(body, m.end() - 1, "{", "}")
        out.append((m.group(1), inner))
        i = end
    return out


def entry_type(inner: str) -> str | None:
    m = re.search(r"(?m)(?:^|;|\s)type\s+([A-Za-z0-9_:<>]+)\s*;", inner)
    return m.group(1) if m else None


def mesh_patches(child: Path) -> tuple[dict[str, str], str]:
    poly = child / "constant" / "polyMesh" / "boundary"
    bmd = child / "system" / "blockMeshDict"
    if poly.is_file():
        text = strip_comments(poly.read_text())
        header = re.search(r"\bFoamFile\s*\{", text)
        if header:  # drop the header dictionary, then the patch list is the first ( ... )
            _, end = block_after(text, header.end() - 1, "{", "}")
            text = text[end:]
        body, _ = block_after(text, 0, "(", ")")
        src = "constant/polyMesh/boundary"
    elif bmd.is_file():
        text = strip_comments(bmd.read_text())
        m = re.search(r"(?m)^\s*boundary\s*\(", text)
        if not m:
            raise ValueError("blockMeshDict has no `boundary (` section")
        body, _ = block_after(text, m.end() - 1, "(", ")")
        src = "system/blockMeshDict"
    else:
        raise FileNotFoundError("no system/blockMeshDict or constant/polyMesh/boundary")
    if "#include" in body or "$" in body:
        raise ValueError(f"{src}: #include or $macro in the boundary section -- UNVERIFIABLE")
    patches = {}
    for name, inner in named_blocks(body):
        t = entry_type(inner)
        if t is None:
            raise ValueError(f"{src}: patch {name} has no type")
        patches[name.strip('"')] = t
    if not patches:
        raise ValueError(f"{src}: no patches parsed")
    return patches, src


def known_types() -> set[str]:
    if not OF_SRC.is_dir():
        raise SystemExit(
            f"check_case_boundaries: {OF_SRC} is not mounted, so no type can be "
            "verified. Mount it (scripts/syntax_check_solver.sh does) and rerun. "
            "Refusing to pass unverified types."
        )
    names = set()
    tn = re.compile(r'TypeName\("([A-Za-z0-9_:]+)"\)')
    for base in (OF_SRC, USER_SRC):
        for dirpath, _, files in os.walk(base):
            for f in files:
                if not f.endswith(".H"):
                    continue
                if base == OF_SRC and "PatchField" not in f and "FvPatch" not in f:
                    continue
                try:
                    names.update(tn.findall(Path(dirpath, f).read_text(errors="ignore")))
                except OSError:
                    pass
    if len(names) < 50:
        raise SystemExit(f"check_case_boundaries: only {len(names)} patch types found -- source tree incomplete")
    return names


def check_child(child: Path, types: set[str]) -> list[str]:
    errs = []
    try:
        patches, src = mesh_patches(child)
    except (ValueError, FileNotFoundError) as e:
        return [f"mesh: {e}"]
    fields = sorted(p for p in (child / "0").iterdir() if p.is_file() and not p.name.startswith("."))
    checked = 0
    for fp in fields:
        text = strip_comments(fp.read_text(errors="ignore"))
        m = re.search(r"\bboundaryField\s*\{", text)
        if not m:
            continue
        checked += 1
        body, _ = block_after(text, m.end() - 1, "{", "}")
        # `value $internalField;` is ordinary and harmless. What cannot be
        # verified is a macro that could INJECT an entry or a type.
        if "#include" in body or re.search(r"(?m)^\s*\$[\w:.]+\s*;|\btype\s+\$", body):
            errs.append(f"0/{fp.name}: #include, an entry-level $macro or a $type in boundaryField -- UNVERIFIABLE")
            continue
        entries = named_blocks(body)
        exact = {k: v for k, v in entries if not k.startswith('"')}
        regex = [(k.strip('"'), v) for k, v in entries if k.startswith('"')]
        groups = set(patches.values())
        for key in exact:
            if key not in patches and key not in groups:
                errs.append(f"0/{fp.name}: entry `{key}` names no patch in {src} (patches: {', '.join(patches)})")
        for patch, ptype in patches.items():
            inner = exact.get(patch)
            if inner is None:
                for pat, v in reversed(regex):
                    if re.fullmatch(pat, patch):
                        inner = v
                        break
            if inner is None:
                inner = exact.get(ptype)  # patch-type group, e.g. `wall`
            if inner is None:
                errs.append(f"0/{fp.name}: no entry for patch `{patch}` ({ptype})")
                continue
            t = entry_type(inner)
            if t is None:
                errs.append(f"0/{fp.name}: patch `{patch}` entry has no type")
            elif t not in types:
                errs.append(f"0/{fp.name}: patch `{patch}` type `{t}` exists in neither OpenFOAM 13 nor user-applications/")
            elif ptype in CONSTRAINT and t != ptype:
                errs.append(f"0/{fp.name}: constraint patch `{patch}` is `{ptype}` but the field says `{t}`")
    if checked == 0:
        errs.append("0/ holds no field with a boundaryField")
    return errs


def children_of(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        p = p.resolve()
        if (p / "0").is_dir():
            out.append(p)
            continue
        runs = p / "runs" if (p / "runs").is_dir() else p
        out.extend(sorted(c for c in runs.iterdir() if c.is_dir() and (c / "0").is_dir()))
    return out


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv] or sorted(ROOT.glob("cases/*/runs"))
    kids = children_of(targets)
    if not kids:
        print("check_case_boundaries: FAIL -- no generated child found; nothing was checked")
        return 1
    types = known_types()
    failed = 0
    for child in kids:
        errs = check_child(child, types)
        if errs:
            failed += 1
            print(f"FAIL {child.relative_to(ROOT) if child.is_relative_to(ROOT) else child}")
            for e in errs:
                print(f"     {e}")
    print(f"check_case_boundaries: {len(kids)} children checked, {failed} failed, "
          f"{len(types)} known patch types")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
