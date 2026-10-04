#!/usr/bin/env python3
"""Fail when a case cannot CONSTRUCT on the current solver.

Three things can stop a case in the constructor, and a check that models
only the first reports the other two as clean:

  1. the case names a value the solver REFUSES outright
     (``refuseDeadValue(...)`` -- the implementation was deleted);
  2. the case names a value whose SWITCH PRECONDITION it does not meet
     (``if (key_ == "v") { if (!contentModel_) FatalError ... }``);
  3. **every legal value of the key is fatal for this case** -- the dead
     one is refused and the survivor's precondition is unmet. Such a case
     has no legal dictionary at all. It is not misconfigured; it cannot
     run on this solver, and saying "set the key to the content form" is
     impossible advice.

Every table below is DERIVED from the solver source and from the cases,
never listed here. Both sides, failing on the intersection, is the shape
``check_dictionary_read_forms.py`` took after the ``mobility`` fatal of
run 20260920-190025.

``--expect-fatal N`` pins the measured count so the gate is wired in while
the library is still broken: a rise is new breakage, a fall is the repair
landing and must be lowered deliberately.
"""

import argparse
import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOLVER = (ROOT / "user-applications" / "phaseClotAxisymmetricFluid"
          / "phaseClotAxisymmetricFluid.C")
GENERATOR = (ROOT / "cases" / "AM2020_planar_mural_detachment" / "scripts"
             / "create_cases.py")
CONTROLS = (ROOT / "cases" / "AM2020_planar_mural_detachment" / "constant"
            / "mainCaseControls.json")

# A dropped refusal site would shrink the derived set and report a clean
# library. Pinned so the set cannot quietly narrow.
EXPECTED_REFUSAL_SITES = 5


# ---------------------------------------------------------------- solver

def strip_cxx(text):
    """Refusal MESSAGES quote the dead values verbatim, so comments must
    go before anything is matched."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def solver_refusals(src):
    body = strip_cxx(src)
    sites = len(re.findall(r'refuseDeadValue\s*\(\s*"', body))
    parsed = dict(re.findall(
        r'refuseDeadValue\s*\(\s*"([A-Za-z0-9_]+)"\s*,'
        r'\s*[A-Za-z0-9_]+\s*,\s*"([A-Za-z0-9_]+)"', body, re.S))
    return parsed, sites


def solver_defaults(src):
    """Both spellings: "legacy" and word("legacy")."""
    return dict(re.findall(
        r'lookupOrDefault<word>\s*\(\s*"([A-Za-z0-9_]+)"\s*,'
        r'\s*(?:word\s*\(\s*)?"([A-Za-z0-9_]*)"', strip_cxx(src), re.S))


def solver_switch_defaults(src):
    return {k: v.strip().lower() == "true" for k, v in re.findall(
        r'lookupOrDefault<Switch>\s*\(\s*"([A-Za-z0-9_]+)"\s*,\s*(\w+)\s*\)',
        strip_cxx(src), re.S)}


def solver_legal_values(src):
    """From the unconditional validators:
    ``if (k_ != "a" && k_ != "b") FatalError``."""
    legal = {}
    for m in re.finditer(
        r'if\s*\(\s*([A-Za-z0-9_]+)_\s*!=\s*"([A-Za-z0-9_]+)"\s*'
        r'&&\s*\1_\s*!=\s*"([A-Za-z0-9_]+)"\s*\)', strip_cxx(src), re.S
    ):
        legal[m.group(1)] = {m.group(2), m.group(3)}
    return legal


def solver_preconditions(src):
    """``if (k_ == "v") { ... if (!switch_) FatalError ... }`` -> the
    switches that value requires. Brace-walked, not regexed across the
    block, so a nested if cannot be missed or over-claimed."""
    lines = strip_cxx(src).split("\n")
    out = {}
    for i, line in enumerate(lines):
        m = re.match(r'\s*if\s*\(\s*([A-Za-z0-9_]+)_\s*==\s*'
                     r'"([A-Za-z0-9_]+)"\s*\)\s*$', line)
        if not m:
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines) or lines[j].strip() != "{":
            continue
        depth, switches = 0, []
        for k in range(j, len(lines)):
            depth += lines[k].count("{") - lines[k].count("}")
            hit = re.match(r'\s*if\s*\(\s*!\s*([A-Za-z0-9_]+)_\s*\)', lines[k])
            if hit:
                switches.append(hit.group(1))
            if depth == 0 and k > j:
                break
        if switches:
            out.setdefault((m.group(1), m.group(2)), set()).update(switches)
    return out


# ------------------------------------------------------------- generator

_GEN_WRITE = re.compile(
    r'add_or_replace_entry\s*\(\s*props\s*,\s*"([A-Za-z0-9_]+)"\s*,\s*'
    r'(.*?)\s*,?\s*\)\s*\n', re.S)
_CHILD_GET = re.compile(
    r'child\.get\s*\(\s*"([A-Za-z0-9_]+)"\s*,\s*'
    r'(?:"([A-Za-z0-9_]*)"|(True|False))\s*\)')


def _resolve(expr, src):
    """(json key, default) for a written value, following one level of
    local variable. Returns None when it cannot be resolved -- callers
    must treat that as a failure, never a skip: the first version of this
    matcher read only the inline form and silently missed
    phaseGrowthDriver, fatal in 154 children."""
    hit = _CHILD_GET.search(expr)
    if hit:
        return hit.group(1), (hit.group(2) if hit.group(2) is not None
                              else hit.group(3) == "True")
    name = expr.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        assign = re.search(
            rf'\b{name}\s*=\s*(?:str|bool)?\s*\(?\s*child\.get\s*\('
            rf'\s*"([A-Za-z0-9_]+)"\s*,\s*'
            rf'(?:"([A-Za-z0-9_]*)"|(True|False))\s*\)', src, re.S)
        if assign:
            return assign.group(1), (assign.group(2)
                                     if assign.group(2) is not None
                                     else assign.group(3) == "True")
    return None


def generator_map(src, required):
    """dict key -> (json key, default). Raises for any REQUIRED key whose
    value expression cannot be resolved."""
    out, unresolved = {}, {}
    for key, expr in _GEN_WRITE.findall(src):
        got = _resolve(expr, src)
        if got:
            out[key] = got
        else:
            unresolved.setdefault(key, " ".join(expr.split())[:70])
    blocked = sorted(k for k in required if k not in out and k in unresolved)
    if blocked:
        raise SystemExit(
            "FAIL: the generator writes these keys with a value this check "
            "cannot resolve, so their children cannot be priced: "
            + "; ".join(f"{k} <- {unresolved[k]}" for k in blocked))
    return out


# ------------------------------------------------------------------ scan

def dict_entries(text):
    words = dict(re.findall(r'^\s*([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*;',
                            text, re.M))
    switches = {k: v.lower() in ("yes", "true", "on", "1")
                for k, v in words.items()}
    return words, switches


def children(controls):
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "child_id" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(controls)
    return found


def classify(key, value, legal, refused, precond, switch_of):
    """'' | 'refused' | 'precondition' -- why this value aborts, if it
    does."""
    if refused.get(key) == value:
        return "refused"
    for switch in precond.get((key, value), ()):
        if not switch_of(switch):
            return f"precondition:{switch}"
    return ""


def scan(verbose=False):
    src = SOLVER.read_text()
    refused, sites = solver_refusals(src)
    if sites != EXPECTED_REFUSAL_SITES or len(refused) != sites:
        raise SystemExit(
            f"FAIL: {sites} refuseDeadValue call sites ({len(refused)} "
            f"parsed), expected {EXPECTED_REFUSAL_SITES}. The refused set "
            f"this check derives is no longer the solver's.")
    defaults = solver_defaults(src)
    sw_default = solver_switch_defaults(src)
    legal = solver_legal_values(src)
    precond = solver_preconditions(src)
    keys = sorted(refused)
    missing = [k for k in keys if k not in defaults or k not in legal]
    if missing:
        raise SystemExit(f"FAIL: refused keys with no default or no legal-"
                         f"value validator: {missing}")

    fatal, unrunnable, scanned = [], [], []

    def price(case, value_of, switch_of):
        scanned.append(case)
        rows, dead_keys = [], []
        for key in keys:
            value = value_of(key)
            why = classify(key, value, legal, refused, precond, switch_of)
            if why:
                rows.append((case, key, value, why))
            if all(classify(key, v, legal, refused, precond, switch_of)
                   for v in legal[key]):
                dead_keys.append(key)
        fatal.extend(rows)
        if dead_keys:
            unrunnable.append((case, dead_keys))

    # --- committed templates (runs/ excluded) --------------------------
    for path in sorted((ROOT / "cases").rglob("phaseClotProperties")):
        if "/runs/" in str(path):
            continue
        words, switches = dict_entries(path.read_text())
        price(str(path.relative_to(ROOT)),
              lambda k, w=words: w.get(k, defaults[k]),
              lambda s, sw=switches: sw.get(s, sw_default.get(s, False)))

    # --- main-case children --------------------------------------------
    gen = generator_map(GENERATOR.read_text(), required=keys)
    for child in children(json.loads(CONTROLS.read_text())):
        def value_of(k, c=child):
            if k in gen:
                jk, dv = gen[k]
                return str(c.get(jk, dv))
            return defaults[k]

        def switch_of(s, c=child):
            if s in gen:
                jk, dv = gen[s]
                return bool(c.get(jk, dv))
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
            return bool(c.get(snake, sw_default.get(s, False)))

        price("child:" + child["child_id"], value_of, switch_of)

    if verbose:
        for row in fatal:
            print("  FATAL", *row)
    dupes = len(scanned) - len(set(scanned))
    if dupes:
        raise SystemExit(f"FAIL: {dupes} case name(s) scanned twice; the "
                         f"live/archived set arithmetic below would be "
                         f"wrong and cannot be trusted.")
    return fatal, unrunnable, set(scanned), generator_map(
        GENERATOR.read_text(), required=keys), set(keys)


def refusal_sites(src):
    """Each refuseDeadValue call as (key, dead, switch, needs)."""
    return [(m.group(1), m.group(2), m.group(3), m.group(4))
            for m in re.finditer(
                r'refuseDeadValue\s*\(\s*'
                r'"([A-Za-z0-9_]+)"\s*,\s*[A-Za-z0-9_]+\s*,\s*'
                r'"([A-Za-z0-9_]+)"\s*,\s*'
                r'([A-Za-z0-9_]+)_\s*,\s*'
                r'"([^"]*)"', strip_cxx(src), re.S)]


def refusal_message_failures(sites, legal, precond):
    """The message tells the reader which switch to turn on. Derive that
    from the SAME precondition table the gate prices cases with, so the
    advice cannot drift from the guard.

    This is the only part of the S7 change no run can exercise: the
    refusal fires solely on an archived case, and archived cases are
    never run. If it is wrong, it is wrong silently and forever.
    """
    out = []
    for key, dead, switch, needs in sites:
        survivors = legal.get(key, set()) - {dead}
        if len(survivors) != 1:
            out.append(f"{key}: {len(survivors)} surviving value(s); the "
                       f"message assumes exactly one")
            continue
        survivor = survivors.pop()
        required = precond.get((key, survivor), set())
        if required != {switch}:
            out.append(
                f"{key} '{dead}' is refused with switch {switch}_, but "
                f"'{survivor}' actually requires {sorted(required) or 'no '
                'switch'}. The message would send the reader to the wrong "
                f"switch.")
        elif switch not in needs:
            out.append(
                f"{key}: the message says \"{needs}\" but the survivor "
                f"requires {switch}. The prose and the guard disagree.")
    return out

# ----------------------------------------------------------- the archive
#
# 2026-09-20, Alireza: "Ignore and archive the unrunnable legacy cases --
# we do not need them."
#
# Archiving is a REGISTRY, not a directory move. Three of the 39 template
# paths are live dependencies of the very pipeline that archives them --
# AM2020_straight_tube_growth_mechanics_smoke/template IS the main case's
# template -- and moving them would break the build in the same turn that
# claims to have tidied it. So nothing relocates; what changes is which
# cases the gate HOLDS TO A STANDARD.
#
# The invariant is stronger than the pinned counts it replaces. Those were
# two magic numbers that drift with the library. This says: every case NOT
# in the registry must construct, every case IN it must still be there and
# must still be unrunnable, and the batch the notebook has SELECTED must
# not name one. A pinned count cannot express any of that, and a count
# that "rises" tells you nothing about which case it was.

ARCHIVE = ROOT / "cases" / "ARCHIVED_CASES.json"
NOTEBOOK = ROOT / "notebooks" / "AM2020_main_case_colab.ipynb"


def archive_entries(registry):
    """The archived case names, in the SAME namespace scan() reports:
    template paths as written, children as ``child:<id>``."""
    return (set(registry["templates"])
            | {"child:" + c for c in registry["children"]})


def live_set_failures(scanned, aborting, archived):
    """The three ways the registry and the library can disagree.

    ``scanned``  every case the scan visited
    ``aborting`` those that cannot construct
    ``archived`` the registry's names

    Taken as sets rather than counts on purpose: a count says a number
    moved, and the question is always WHICH case.
    """
    out = []
    live_broken = sorted(aborting - archived)
    if live_broken:
        out.append(
            f"{len(live_broken)} LIVE case(s) cannot construct and are not "
            f"in {ARCHIVE.name}. Repair them, or archive them "
            f"deliberately: " + ", ".join(live_broken[:8]))
    stale = sorted(archived - scanned)
    if stale:
        out.append(
            f"{len(stale)} archived name(s) no longer exist in the library, "
            f"so the registry is describing cases that are gone: "
            + ", ".join(stale[:8]))
    revived = sorted((archived & scanned) - aborting)
    if revived:
        out.append(
            f"{len(revived)} archived case(s) CAN now construct. An archive "
            f"that silently absorbs working cases hides them: delete the "
            f"entry to bring them back: " + ", ".join(revived[:8]))
    return out


def selected_batch(text):
    """(n, children) for the notebook's selected BATCH -- the same
    brace-walk check_notebook_batch_selector.py uses, because a regex over
    a dict whose values carry braces in their prose reads the wrong end."""
    sel = re.search(r"^BATCH = (\d+)\s*$", text, re.M)
    start = text.find("BATCHES = {")
    if not sel or start < 0:
        raise SystemExit("FAIL: the notebook has no BATCH selector or no "
                         "BATCHES mapping; this gate cannot see what the "
                         "package would run.")
    depth, end = 0, None
    for i in range(text.index("{", start), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise SystemExit("FAIL: the BATCHES mapping is not brace-balanced.")
    mapping = ast.literal_eval(text[text.index("{", start):end])
    n = int(sel.group(1))
    if n not in mapping:
        raise SystemExit(f"FAIL: BATCH = {n} is not a key of BATCHES.")
    return n, mapping[n][0].split()


def selected_batch_failures(n, named, archived):
    """A package whose selected batch names an archived child would boot a
    paid Colab session and abort in the constructor. This is the specific
    shape of the two sessions lost to batches 43 and 44 -- except those
    aborted on a value, and this one would abort on a case we have already
    written off."""
    dead = [c for c in named if "child:" + c in archived]
    if not dead:
        return []
    return [f"BATCH = {n} names {len(dead)} ARCHIVED child(ren) -- they "
            f"cannot construct on this solver and the session would abort: "
            + ", ".join(dead)]


def generator_covers_refused_keys(gen, refused):
    """The premise the template exemption rests on, asserted rather than
    asserted-in-prose: a main-case child never inherits the template's
    value for a refused key, because create_cases.py writes all five."""
    missing = sorted(k for k in refused if k not in gen)
    if missing:
        return [f"the generator does NOT write {missing}, so main-case "
                f"children inherit the template's value for them -- and "
                f"every template is archived as unrunnable. The exemption "
                f"in {ARCHIVE.name} is void."]
    return []


# ------------------------------------------------------------- self-test

def self_test():
    src = SOLVER.read_text()
    refused, sites = solver_refusals(src)
    assert sites == EXPECTED_REFUSAL_SITES == 5, sites
    assert refused.get("phaseGrowthDriver") == "front", refused

    cut = re.sub(r'refuseDeadValue\s*\(\s*"phaseNormalBasis".*?\);', "",
                 strip_cxx(src), count=1, flags=re.S)
    assert cut != strip_cxx(src), "mutation target absent"
    assert solver_refusals(cut)[1] == 4, "a dropped refusal site was absorbed"

    n = solver_refusals(src + '\n// refuseDeadValue("f", x_, "legacy");\n')[1]
    assert n == 5, f"a commented refusal was counted as a site ({n})"

    # BOTH default spellings, on a FIXTURE. Pinning a live default here
    # would make the self-test fail the moment the repair flips it, which
    # is a test of the setting rather than of the matcher.
    fixture = ('a(phaseClotProperties_.lookupOrDefault<word>("kOne", "legacy"));'
               'b(phaseClotProperties_.lookupOrDefault<word>\n('
               '\n"kTwo", word("content")\n));')
    fd = solver_defaults(fixture)
    assert fd == {"kOne": "legacy", "kTwo": "content"}, fd
    # and every refused key must have SOME default in the live source,
    # because omission is what the default decides.
    d = solver_defaults(src)
    assert all(k in d for k in refused), [k for k in refused if k not in d]

    legal = solver_legal_values(src)
    assert legal.get("phaseGrowthDriver") == {"front", "transfer"}, legal
    assert legal.get("phaseFluidFractionBasis") == {"legacy", "content"}

    pre = solver_preconditions(src)
    assert pre[("phaseFluidFractionBasis", "content")] == {"contentModel"}
    assert pre[("phaseGrowthDriver", "transfer")] == {"plateletContentModel"}

    sd = solver_switch_defaults(src)
    assert sd.get("contentModel") is False, sd.get("contentModel")

    gsrc = GENERATOR.read_text()
    gen = generator_map(gsrc, required=list(refused))
    assert gen["phaseNormalBasis"][0] == "phase_normal_basis", gen
    assert gen["phaseGrowthDriver"] == ("phase_growth_driver", "transfer")
    assert gen["contentModel"] == ("content_model", False), gen["contentModel"]

    broken = gsrc.replace('props, "phaseGrowthDriver", growth_driver',
                          'props, "phaseGrowthDriver", opaque(child)')
    assert broken != gsrc, "mutation target absent"
    try:
        generator_map(broken, required=list(refused))
    except SystemExit as exc:
        assert "phaseGrowthDriver" in str(exc), exc
    else:
        raise AssertionError("an unresolvable write of a refused key was "
                             "dropped, not raised")

    # A case meeting every precondition and naming every surviving value
    # must be clean; the same case with the dead value must not be.
    on = lambda s: True
    assert not classify("phaseGrowthDriver", "transfer", legal, refused,
                        pre, on)
    assert classify("phaseGrowthDriver", "front", legal, refused, pre, on) \
        == "refused"
    assert classify("phaseGrowthDriver", "transfer", legal, refused, pre,
                    lambda s: False).startswith("precondition:")
    # THE INVARIANT MUST BE ABLE TO FAIL. A case naming the dead value
    # while the survivor's precondition IS met is fixable, and must not be
    # absorbed into "cannot run at all" -- that is the distinction the
    # whole gate rests on, and before the S7 migration it separated 16
    # fixable children from 185 unrunnable ones.
    synth_legal = {"kA": {"dead", "live"}}
    synth_refused = {"kA": "dead"}
    synth_pre = {("kA", "live"): {"someSwitch"}}
    both_fatal = lambda on: all(
        classify("kA", v, synth_legal, synth_refused, synth_pre, on)
        for v in synth_legal["kA"])
    assert classify("kA", "dead", synth_legal, synth_refused, synth_pre,
                    lambda s: True) == "refused"
    assert not both_fatal(lambda s: True), \
        "a fixable case was classified as having no legal value"
    assert both_fatal(lambda s: False), \
        "a case with every value fatal was classified as fixable"

    # --- the archive invariant, on FIXTURES ------------------------------
    # Each of the three disagreements must be caught SEPARATELY. A single
    # "the sets match" assertion passes on two errors that cancel: an
    # archived case that vanished and a live case that broke move the
    # counts by the same amount in opposite directions.
    scanned = {"a", "b", "c"}
    assert live_set_failures(scanned, {"a"}, {"a"}) == [], \
        "a correctly archived broken case was reported as a problem"
    assert any("LIVE" in m for m in
               live_set_failures(scanned, {"a", "b"}, {"a"})), \
        "a live case that cannot construct was absorbed"
    assert any("no longer exist" in m for m in
               live_set_failures(scanned, {"a"}, {"a", "gone"})), \
        "a stale archive entry was absorbed"
    assert any("CAN now construct" in m for m in
               live_set_failures(scanned, set(), {"a"})), \
        "an archived case that works again was absorbed"
    # and the cancelling pair must still fail, twice.
    both = live_set_failures(scanned, {"b"}, {"a"})
    assert len(both) == 2, both

    # THE BRACE WALK. The BATCHES values are prose containing braces and
    # apostrophes; a regex for the closing brace stops at the first one
    # inside a description and reads a truncated mapping.
    fixture = (
        "BATCHES = {\n"
        "    7: ('emA emB', 0.3, 'a note with a { brace } in it'),\n"
        "    8: ('emC', 0.1, 'another'),\n"
        "}\n"
        "BATCH = 8\n"
    )
    n, named = selected_batch(fixture)
    assert (n, named) == (8, ["emC"]), (n, named)
    assert selected_batch_failures(8, ["emC"], {"child:emZ"}) == []
    assert any("ARCHIVED" in m for m in
               selected_batch_failures(8, ["emC"], {"child:emC"})), \
        "a selected batch naming an archived child was allowed to ship"

    # THE REFUSAL MESSAGE, and it needs a mutation because nothing else
    # can reach it: the refusal fires only on an archived case.
    sites = refusal_sites(src)
    assert len(sites) == EXPECTED_REFUSAL_SITES, sites
    assert ("phaseGrowthDriver", "front", "plateletContentModel",
            "plateletContentModel true") in sites, sites
    assert refusal_message_failures(sites, legal, pre) == [], \
        refusal_message_failures(sites, legal, pre)
    wrong = [(k, d, "contentModel" if s_ != "contentModel" else "x", n)
             for k, d, s_, n in sites]
    assert any("wrong switch" in m for m in
               refusal_message_failures(wrong, legal, pre)), \
        "a refusal naming the wrong switch was not noticed"
    drifted = [(k, d, s_, "turn on somethingElse") for k, d, s_, n in sites]
    assert any("prose and the guard disagree" in m for m in
               refusal_message_failures(drifted, legal, pre)), \
        "a message whose prose names no switch was not noticed"

    assert generator_covers_refused_keys({"k1": 1, "k2": 2},
                                         {"k1", "k2"}) == []
    assert generator_covers_refused_keys({"k1": 1}, {"k1", "k2"}), \
        "a refused key the generator does not write was not noticed"

    # The live registry must PARSE and be in the scan's namespace. Not a
    # pin on its contents: the set arithmetic above is what judges those,
    # and pinning them here would fail the next deliberate archiving.
    reg = json.loads(ARCHIVE.read_text())
    assert reg["templates"] and reg["children"], "the registry is empty"
    assert all(t.startswith("cases/") for t in reg["templates"]), \
        "a template entry is not a repo-relative path as scan() reports it"
    assert all(":" not in c for c in reg["children"]), \
        "a child entry carries the child: prefix; scan() adds it"

    print("self-test OK: 5 refusals; dropped site seen; comment ignored; "
          "both default spellings; legal sets, preconditions and switch "
          "defaults derived; both generator write forms; unresolvable "
          "write raises; fixable vs unrunnable separates; live/stale/"
          "revived caught separately and the cancelling pair twice; brace "
          "walk survives a brace in prose; an archived child in the "
          "selected batch refused; an unwritten refused key noticed; a "
          "refusal naming the wrong switch and one whose prose names "
          "no switch both caught")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--expect-fatal", type=int, default=None)
    ap.add_argument("--expect-unrunnable", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    fatal, unrunnable, scanned, gen, refused_keys = scan(verbose=args.verbose)
    by_reason = {}
    for _, key, _, why in fatal:
        tag = f"{key} [{why.split(':')[0]}]"
        by_reason[tag] = by_reason.get(tag, 0) + 1
    for tag in sorted(by_reason, key=lambda t: -by_reason[t]):
        print(f"  fatal {tag:52s} {by_reason[tag]}")
    aborting = {row[0] for row in fatal}
    kids = [c for c in aborting if c.startswith("child:")]
    print(f"TOTAL fatal (case, key) pairs: {len(fatal)}")
    print(f"  cases scanned: {len(scanned)}")
    print(f"  aborting at construction: {len(aborting)} "
          f"({len(aborting) - len(kids)} templates, {len(kids)} children)")
    print(f"  of which have NO LEGAL VALUE for some key "
          f"(cannot run at all): {len(unrunnable)}")

    problems = []

    # 1. Nothing may abort for a reason a dictionary edit could fix. This
    #    is the invariant S7 Route A established and it is independent of
    #    the archive: a fixable case is a bug whether or not it is retired.
    fixable = sorted(aborting - {c for c, _ in unrunnable})
    if fixable:
        problems.append(
            f"{len(fixable)} case(s) abort for a FIXABLE reason -- a legal "
            f"value exists and the case does not name it: "
            + ", ".join(fixable[:8]))

    # 2. The live set. Every case not in the registry must construct.
    registry = json.loads(ARCHIVE.read_text())
    archived = archive_entries(registry)
    problems += live_set_failures(scanned, aborting, archived)

    # 3. The premise under the registry's template exemption.
    problems += generator_covers_refused_keys(gen, refused_keys)

    #    and the refusal MESSAGE, which no run can reach: it fires
    #    only on an archived case, and archived cases are not run.
    src = SOLVER.read_text()
    problems += refusal_message_failures(
        refusal_sites(src), solver_legal_values(src),
        solver_preconditions(src))

    # 4. The package must not ship pointed at an archived child.
    nb = "\n".join("".join(c["source"])
                    for c in json.loads(NOTEBOOK.read_text())["cells"])
    n, named = selected_batch(nb)
    problems += selected_batch_failures(n, named, archived)

    live = len(scanned) - len(archived & scanned)
    print(f"  archived (registry): {len(archived)} "
          f"({len(registry['templates'])} templates, "
          f"{len(registry['children'])} children)")
    print(f"  LIVE cases required to construct: {live}")
    print(f"  BATCH = {n}: {len(named)} child(ren), none archived"
          if not any("BATCH" in m for m in problems)
          else f"  BATCH = {n}: {len(named)} child(ren)")

    for name, got, want in (("fatal", len(fatal), args.expect_fatal),
                            ("unrunnable", len(unrunnable),
                             args.expect_unrunnable)):
        if want is not None and got != want:
            problems.append(f"expected {want} {name}, found {got}")

    if problems:
        for m in problems:
            print("FAIL: " + m)
        return 1
    print("OK: every live case constructs; the archive is exact "
          "(nothing stale, nothing revived); the generator writes all "
          f"{len(refused_keys)} refused keys so no child inherits an "
          "archived template's value; and the selected batch names only "
          "live children.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
