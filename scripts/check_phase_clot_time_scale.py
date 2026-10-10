#!/usr/bin/env python3
"""S25 section 2F / section 12 item 7: every committed phaseClotProperties states timeScale 1.0.

THE SOLVER REFUSES A MISSING timeScale ON EVERY DICTIONARY. networkRupture
fibreRelease has been built in since S21, so the solver's s25TimeScale()
FatalErrors when `timeScale` is absent, non-finite or <= 0 (no default). A
refusal is only as good as the dictionaries that no longer rely on the old
default, so this guard DERIVES the population from the repository (every file
named phaseClotProperties, wherever it is, outside preflight/ and packages/ at
the top level and outside any directory called runs/) and requires of each:

  1. exactly one top-level `timeScale <number>;` entry (comments stripped, so a
     commented-out key does not count; a repeated key is refused);
  2. the number is finite and > 0 (the solver's own predicate);
  3. the number is 1.0 (a committed dictionary is the UNSCALED series: a
     scaled series states S_t in mainCaseControls.json `time_scale`, and the
     generator writes it into the child; the generator's other writers of
     physical times (networkReleaseTime, networkViscosityTau, gammaLow,
     gammaHigh) are physical-seconds values times S_t, so a committed S_t
     other than 1.0 would silently rescale every template it feeds).

It also refuses an EMPTY population (an empty list proves nothing) and a file
whose name merely starts with phaseClotProperties (a backup or variant that a
generator could copy: it is checked as if it were the real thing).

``--self-test`` mutates a scratch copy of the real population, one mutation per
statement, and requires every one to be caught (exit 1 if any is missed).
``--root DIR`` checks another tree (used to show the failure on the
pre-edit dictionaries).
"""

import argparse
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP_LEVEL_EXCLUDED = ("preflight", "packages", ".git")
EXCLUDED_ANYWHERE = ("runs",)
NAME = "phaseClotProperties"
COMMITTED_VALUE = 1.0


def population(root: Path) -> list:
    """Every file whose name starts with phaseClotProperties, outside the excluded trees."""
    found = []
    for here, dirs, files in os.walk(root):
        rel = Path(here).relative_to(root)
        dirs[:] = sorted(
            d for d in dirs
            if d not in EXCLUDED_ANYWHERE
            and not (rel == Path(".") and d in TOP_LEVEL_EXCLUDED)
        )
        for name in sorted(files):
            if name.startswith(NAME):
                found.append(Path(here) / name)
    return found


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def verdict(path_name: str, text: str):
    """None when the dictionary is fine, else the reason it is not."""
    if path_name != NAME:
        return f"file name {path_name!r} is a variant of {NAME}: remove it or state timeScale and rename"
    body = strip_comments(text)
    values = re.findall(r"(?m)^\s*timeScale\s+([^;\s]+)\s*;", body)
    if not values:
        return "lacks `timeScale` (the solver FatalErrors on a missing key)"
    if len(values) > 1:
        return f"names timeScale {len(values)} times: {values}"
    try:
        value = float(values[0])
    except ValueError:
        return f"timeScale {values[0]!r} is not a number"
    if not (math.isfinite(value) and value > 0):
        return f"timeScale {values[0]} is not finite and > 0 (the solver's predicate)"
    if value != COMMITTED_VALUE:
        return (f"timeScale {values[0]} is not {COMMITTED_VALUE}: a committed dictionary is the unscaled series; "
                f"S_t is stated by mainCaseControls.json time_scale")
    return None


def check(root: Path) -> list:
    """The list of (relative path, reason) failures; [(None, reason)] for an empty population."""
    files = population(root)
    if not files:
        return [(None, f"no {NAME} found under {root}: an empty population proves nothing")]
    out = []
    for f in files:
        why = verdict(f.name, f.read_text())
        if why:
            out.append((str(f.relative_to(root)), why))
    return out


def self_test() -> int:
    real = population(ROOT)
    assert real, "no real population to mutate"
    missed = []

    def scratch():
        tmp = Path(tempfile.mkdtemp(prefix="timescale-selftest-"))
        for f in real:
            dst = tmp / f.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        return tmp

    def edit(tmp, rel, fn):
        p = tmp / rel
        p.write_text(fn(p.read_text()))

    victim = real[0].relative_to(ROOT)
    other = real[-1].relative_to(ROOT)

    def expect(label, mutate, want_fail, want_in=None):
        tmp = scratch()
        try:
            mutate(tmp)
            res = check(tmp)
            ok = bool(res) == want_fail
            if ok and want_fail and want_in is not None:
                ok = any(want_in in (r[0] or "") for r in res)
            print(f"  {'ok  ' if ok else 'MISS'} {label:<72} -> {len(res)} failure(s)"
                  + (f": {res[0][1][:70]}" if res else ""))
            if not ok:
                missed.append(label)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    key = re.compile(r"(?m)^timeScale[^\n]*\n")
    expect("pristine copy of the real population passes", lambda t: None, False)
    expect("key removed from one dictionary", lambda t: edit(t, victim, lambda s: key.sub("", s)), True, str(victim))
    for bad in ("0", "-1", "nan", "inf", "2.0", "abc", "1e-400"):
        expect(f"timeScale {bad}", lambda t, b=bad: edit(t, other, lambda s: key.sub(f"timeScale {b};\n", s)), True, str(other))
    expect("key only in a // comment", lambda t: edit(t, victim, lambda s: key.sub("// timeScale 1.0;\n", s)), True, str(victim))
    expect("key only in a /* */ comment", lambda t: edit(t, victim, lambda s: key.sub("/* timeScale 1.0; */\n", s)), True, str(victim))
    expect("key written twice", lambda t: edit(t, victim, lambda s: s + "timeScale 1.0;\n"), True, str(victim))
    expect("key misspelt (timescale)", lambda t: edit(t, victim, lambda s: key.sub("timescale 1.0;\n", s)), True, str(victim))
    expect("key misspelt (timeScales)", lambda t: edit(t, victim, lambda s: key.sub("timeScales 1.0;\n", s)), True, str(victim))

    def new_template(t):
        d = t / "cases" / "AM2020_a_brand_new_case" / "template" / "constant"
        d.mkdir(parents=True)
        (d / NAME).write_text("FoamFile\n{\n}\nrho 1050;\n")
    expect("a NEW case's dictionary without the key (the population is derived)", new_template, True, "AM2020_a_brand_new_case")

    def variant(t):
        p = t / real[1].relative_to(ROOT).parent / (NAME + ".orig")
        p.write_text("rho 1;\n")
    expect("a variant file phaseClotProperties.orig without the key", variant, True, NAME + ".orig")

    def excluded(t):
        for sub in ("preflight/run1/constant", "packages/p/constant", "cases/AM2020_x/runs/c1/constant",
                    "cases/AM2020_x/template/runs/c1/constant"):
            d = t / sub
            d.mkdir(parents=True)
            (d / NAME).write_text("rho 1;\n")
    expect("keyless dictionaries under preflight/, packages/ and runs/ are NOT part of the population", excluded, False)

    def not_excluded_nested(t):
        d = t / "cases" / "AM2020_x" / "preflight" / "constant"   # preflight is excluded at the TOP level only
        d.mkdir(parents=True)
        (d / NAME).write_text("rho 1;\n")
    expect("a nested directory called preflight is NOT excluded (only the top-level one is)", not_excluded_nested, True, "preflight")

    def empty(t):
        for f in population(t):
            f.unlink()
    expect("an empty population is refused", empty, True)
    if missed:
        print(f"SELF-TEST FAILED: not caught: {missed}")
        return 1
    print(f"self-test: every mutation of the {len(real)}-file population was caught")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    files = population(args.root)
    failures = check(args.root)
    for rel, why in failures:
        print(f"FAIL {rel}: {why}" if rel else f"FAIL {why}")
    if failures:
        print(f"check_phase_clot_time_scale: {len(failures)} of {len(files)} dictionaries fail")
        return 1
    print(f"check_phase_clot_time_scale: all {len(files)} committed phaseClotProperties state timeScale {COMMITTED_VALUE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
