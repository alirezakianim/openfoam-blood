#!/usr/bin/env python3
"""S21 V21-3: the solver refuses every key S21 took out of the dictionary,
and no generated dictionary names one.

Design: research/designs/2026-10-05-S21-clean-solver.md, section 1.2 and
section 4 V21-3 as amended (docket 20261005-1621, reports 06, 09, 10).

THE PREMISE IS INVERTED. Until S21 this guard derived the solver's
``refuseDeadValue(...)`` sites (five dead values of live keys) and held the
generator to writing ALL of those keys into every child, so that no child
inherited an archived template's value. S21 deleted refuseDeadValue and every
one of those keys' other values: a key is now either an option (eight are
left) or built in. A built-in key in a dictionary is fatal, so the generator
must now write NONE of them -- the opposite of what this guard used to hold.

What it holds now, each statement both ways:

  1. The solver's ``s21BuiltInKeys`` table equals section 1.2's 78 built-in
     keys at the raw token every kept child uses (the key audit rev. 2,
     probes/ce-s21-key-audit.txt), including the three phase-basis keys
     resolved from the multi-line initialiser form (red team 05 S6).
  2. The solver's ``s21DeletedKeys`` table equals the union of the engineers'
     deleted-entry lists (06: none; 09: the four released* parameters; 10:
     eighteen), 22 entries (probes/se10-s21c1-deleted-keys.txt).
  3. Each table is REFUSED: a loop over it calls found() on the dictionary
     and ends in FatalError; the built-in message names the key and its
     built-in value, the deleted one says "S21: deleted, no effect".
  4. The solver reads none of the 100 keys anywhere else (no lookup or
     found of the literal), and still reads each of the 8 kept options,
     none of which is in either table.
  5. No generated dictionary names one: the main case's generator is run on
     its whole matrix in a scratch copy, and no child's
     constant/phaseClotProperties contains any of the 100 keys (a token grep
     of the WHOLE file), no other constant/ or system/ dictionary has one as
     an entry (comments stripped), and no child.json names one; nor does the
     main case's own template dictionary.

The two tables are TYPED here, not read from the solver or the generator,
so the check shares no table with the code it judges.

``--self-test`` applies one mutation per statement and requires each to be
caught; it exits 1 if any is missed.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = (ROOT / "user-applications" / "phaseClotAxisymmetricFluid"
          / "phaseClotAxisymmetricFluid.C")
CASE = ROOT / "cases" / "AM2020_planar_mural_detachment"
GATE = ROOT / "cases" / "AM2020_mural_geometry_gate"
CONTROLS = CASE / "constant" / "mainCaseControls.json"
TEMPLATE_DICT = CASE / "template" / "constant" / "phaseClotProperties"

KEPT8 = ("coreShell", "materialLabelBasis", "momentumPressureCoupling",
         "networkPressureWeight", "networkViscosity", "plateletTransportOnly",
         "projSiteLog", "releaseCoreExempt")

# Section 1.2's 78 built-in keys, at the raw token the solver table must
# carry: Switch keys as "true"/"false", word keys raw (explicitStabilityBounds
# is a word compared to "on"/"off": red team 05 S2).
BUILT_IN = {
    # group A (docket report 06)
    "contentModel": "true", "contentOperator": "content",
    "contentTransportScheme": "fct", "cplusOrder": "shared",
    "materialCarrierBlend": "false", "phaseDeltaBasis": "content",
    "phaseFieldMaterialVelocity": "true", "phaseFluidFractionBasis": "content",
    "phaseForceBasis": "content", "phaseGrowthDriver": "transfer",
    "phaseIndicatorBasis": "content", "phaseLimiterBasis": "content",
    "phaseMomentumFluxBasis": "shared", "phaseNormalBasis": "content",
    "phasePlateletFluxBasis": "content", "phaseWeightBasis": "content",
    "plateletBindLocalisation": "interface", "plateletCarrierArchive": "false",
    "plateletContentFormulation": "true", "plateletContentModel": "true",
    "plateletFluxConsistent": "false", "plateletFluxConvention": "intrinsic",
    "plateletOuterForceReject": "false", "plateletOuterTransactions": "false",
    # group B (docket report 09)
    "advectReleased": "true", "damageTransport": "contentPartition",
    "elasticCoupling": "implicitTangent", "elasticStabilityStiffness": "tangent",
    "fibreArithmetic": "fieldUnits", "fibreSlack": "v1",
    "frozenPendingPressure": "none", "networkDamageProgressive": "true",
    "networkPressureDamage": "tensile", "networkPressureForce": "faceGradient",
    "networkPressureModel": "convexExcess",
    "networkPressureReference": "deposited", "networkRupture": "fibreRelease",
    "releasedCarrier": "material", "stressFaceTransmission": "series",
    # group C (docket report 10)
    "beDetRenormalise": "true", "beFluxConsistentGradU": "true",
    "beRelaxBasis": "networkAbsent", "beStretchGradient": "fluxConsistent",
    "beTermDiagnostic": "true", "beTraceFreeSource": "false",
    "blockCoupledLumen": "eliminate", "blockFaceInverse": "faceAssembled",
    "bondHoldCarrier": "true", "bondHoldTaper": "false",
    "clotMechanics": "true", "constitutiveModel": "neoHookean",
    "dragDriver": "legacy", "dragOccupancyBasis": "intrinsic",
    "eulerianBrinkman": "true", "eulerianMaterialFluxIsPhi": "true",
    "eulerianSolid": "true", "explicitStabilityBounds": "on",
    "faceConsistentElasticForce": "false", "finiteStrain": "true",
    "fsiAitken": "false", "fsiSweepTrace": "true",
    "growthClosure": "reaction", "interfaceDepletionModel": "none",
    "operatorAudit": "false", "operatorLocalisationAudit": "false",
    "permeabilityModel": "kozenyCarmanSolid",
    "shearEnhancementModel": "thresholdSaturating",
    "skeletonDiffuseDomain": "true", "skeletonFloorBasis": "occupancy",
    "skeletonMomentumForm": "phaseContinuity", "skeletonPressureForce": "false",
    "skeletonResidualDiagnostic": "false", "skeletonWallFace": "zeroed",
    "stronglyCoupledFSI": "false", "twoVelocitySkeleton": "true",
    "unlocalisedPhaseWeightsAccepted": "false", "vnCapEnabled": "false",
    "wallPatchSplit": "split",
}
PHASE_BASIS_FROM_MULTILINE = ("phaseFluidFractionBasis", "phaseLimiterBasis",
                              "phaseNormalBasis")

# The union of the deleted-entry lists of reports 06 (none), 09 (four) and
# 10 (eighteen).
DELETED = (
    "releasedAmbientQ", "releasedMaskDilation", "releasedMaskLimit",
    "releasedMaskSweeps",
    "fsiNormProbeFraction", "fsiOmegaMax", "fsiOmegaMin", "fsiTolerance", "k1",
    "materialCourant", "materialDeltaTMin", "nMaterialFloorSteps",
    "nFSICorrectors", "nSkeletonCorrectors", "nStaggeredCorrectors",
    "nSkeletonDivergedSteps", "skeletonDivergenceGrowth", "operatorAuditEvery",
    "phi0", "skeletonFixedPointTolerance", "skeletonImplicitFactor",
    "skeletonUsBound",
)

assert len(BUILT_IN) == 78, len(BUILT_IN)
assert len(DELETED) == 22 and len(set(DELETED)) == 22
assert not set(BUILT_IN) & set(DELETED)
assert not (set(BUILT_IN) | set(DELETED)) & set(KEPT8)
assert set(PHASE_BASIS_FROM_MULTILINE) <= set(BUILT_IN)
GONE = set(BUILT_IN) | set(DELETED)


# ---------------------------------------------------------------- solver

def strip_cxx(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def flat(text):
    return re.sub(r"\s+", "", strip_cxx(text))


def table_block(body, decl):
    """The brace body of ``<decl> = { ... };``, or None."""
    m = re.search(re.escape(decl) + r"\s*=\s*\{", body)
    if not m:
        return None
    depth, i = 1, m.end()
    while depth and i < len(body):
        depth += {"{": 1, "}": -1}.get(body[i], 0)
        i += 1
    return body[m.end():i - 1] if depth == 0 else None


def solver_tables(src):
    body = strip_cxx(src)
    b = table_block(body, "static const char* const s21BuiltInKeys[][2]")
    d = table_block(body, "static const char* const s21DeletedKeys[]")
    built = None if b is None else re.findall(
        r'\{\s*"([A-Za-z0-9_]+)"\s*,\s*"([A-Za-z0-9_]*)"\s*\}', b)
    deleted = None if d is None else re.findall(r'"([A-Za-z0-9_]+)"', d)
    return built, deleted


def loop_body(fl, head):
    """The flattened body of the for-loop whose header is ``head``."""
    i = fl.find(head)
    if i < 0 or fl.find(head, i + 1) >= 0:
        return None
    j = i + len(head)
    if fl[j:j + 1] != "{":
        return None
    depth, k = 1, j + 1
    while depth and k < len(fl):
        depth += {"{": 1, "}": -1}.get(fl[k], 0)
        k += 1
    return fl[j + 1:k - 1]


def solver_failures(src):
    bad = []
    built, deleted = solver_tables(src)
    if built is None:
        return ["solver: no `static const char* const s21BuiltInKeys[][2] = {...}` table"]
    if deleted is None:
        return ["solver: no `static const char* const s21DeletedKeys[] = {...}` table"]

    keys = [k for k, _ in built]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        bad.append(f"s21BuiltInKeys names a key twice: {dup}")
    have = dict(built)
    miss = sorted(set(BUILT_IN) - set(have))
    extra = sorted(set(have) - set(BUILT_IN))
    if miss:
        bad.append(f"s21BuiltInKeys lacks {len(miss)} of section 1.2's built-in keys: {miss}")
    if extra:
        bad.append(f"s21BuiltInKeys has {len(extra)} key(s) section 1.2 does not build in: {extra}")
    wrong = sorted(k for k in set(have) & set(BUILT_IN) if have[k] != BUILT_IN[k])
    if wrong:
        bad.append("s21BuiltInKeys carries the wrong built-in token for "
                   + ", ".join(f"{k} ({have[k]!r}, section 1.2 {BUILT_IN[k]!r})" for k in wrong))
    if len(built) != 78:
        bad.append(f"s21BuiltInKeys has {len(built)} rows, section 1.2 has 78")

    dup = sorted({k for k in deleted if deleted.count(k) > 1})
    if dup:
        bad.append(f"s21DeletedKeys names a key twice: {dup}")
    miss = sorted(set(DELETED) - set(deleted))
    extra = sorted(set(deleted) - set(DELETED))
    if miss:
        bad.append(f"s21DeletedKeys lacks {len(miss)} deleted entr(ies) of reports 06/09/10: {miss}")
    if extra:
        bad.append(f"s21DeletedKeys has {len(extra)} entr(ies) no report deleted: {extra}")
    if len(deleted) != 22:
        bad.append(f"s21DeletedKeys has {len(deleted)} entries, reports 06/09/10 list 22")

    both = sorted(set(have) & set(deleted))
    if both:
        bad.append(f"key(s) in both tables: {both}")
    kept_in = sorted((set(have) | set(deleted)) & set(KEPT8))
    if kept_in:
        bad.append(f"kept option(s) refused by an S21 table: {kept_in}")

    # 3. each table is refused, and says why
    fl = flat(src)
    lb = loop_body(fl, "for(constauto&builtIn:s21BuiltInKeys)")
    if lb is None:
        bad.append("the built-in refusal loop `for (const auto& builtIn : s21BuiltInKeys)` "
                   "is missing or not unique")
    elif not ("if(phaseClotProperties_.found(builtIn[0]))" in lb
              and "FatalErrorInFunction" in lb and "exit(FatalError)" in lb
              and "<<builtIn[0]<<" in lb and "<<builtIn[1]<<" in lb):
        bad.append("the built-in refusal loop no longer refuses a found key with a "
                   "FatalError naming the key and its built-in value")
    ld = loop_body(fl, "for(constchar*constdeleted:s21DeletedKeys)")
    if ld is None:
        bad.append("the deleted-key refusal loop `for (const char* const deleted : "
                   "s21DeletedKeys)` is missing or not unique")
    elif not ("if(phaseClotProperties_.found(deleted))" in ld
              and "FatalErrorInFunction" in ld and "exit(FatalError)" in ld
              and "<<deleted<<" in ld and 'S21:"<<"deleted,noeffect' in ld):
        bad.append("the deleted-key refusal loop no longer refuses a found key with a "
                   "FatalError naming it and saying \"S21: deleted, no effect\"")

    # 4. no other read of a gone key; every kept option still read
    reads = re.findall(
        r'phaseClotProperties_\s*\.\s*(?:lookup\w*|found|readIfPresent|subDict\w*|isDict)'
        r'\s*(?:<[^>]*>)?\s*\(\s*"([A-Za-z0-9_]+)"', strip_cxx(src))
    gone_read = sorted(set(reads) & GONE)
    if gone_read:
        bad.append(f"the solver still reads S21-removed key(s) from the dictionary: {gone_read}")
    unread = sorted(set(KEPT8) - set(reads))
    if unread:
        bad.append(f"kept option(s) the solver no longer reads: {unread}")
    return bad


# ---------------------------------------------------------------- generated

def children(controls):
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "child_id" in node:
                found.append(node["child_id"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(controls)
    return found


def snake(key):
    return re.sub(r"(?<=[a-z0-9])([A-Z])", lambda m: "_" + m.group(1).lower(), key).lower()


SNAKE = {snake(k): k for k in GONE}


def json_names(obj):
    hit = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in GONE or k in SNAKE:
                hit.add(k)
            hit |= json_names(v)
    elif isinstance(obj, list):
        for v in obj:
            hit |= json_names(v)
    return hit


def whole_file_tokens(text):
    return sorted(k for k in GONE if re.search(rf"\b{k}\b", text))


def entry_tokens(text):
    t = re.sub(r"(?s)/\*.*?\*/", "", re.sub(r"//[^\n]*", "", text))
    return sorted(set(re.findall(r"(?m)^\s*([A-Za-z]\w*)\b", t)) & GONE)


def generate(work):
    """The main case's generator on its whole matrix, in a scratch copy
    beside the mural gate (create_cases.py imports it). Returns runs/."""
    root = work / "cases"
    case = root / CASE.name
    shutil.copytree(CASE, case, ignore=shutil.ignore_patterns("runs", "__pycache__", ".claude"))
    shutil.copytree(GATE, root / GATE.name, ignore=shutil.ignore_patterns("__pycache__"))
    r = subprocess.run([sys.executable, "-B", "scripts/create_cases.py"], cwd=case,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"generator exit {r.returncode}: {(r.stdout + r.stderr)[-800:]}")
    return case / "runs"


def generated_failures(runs, want_ids, template_text):
    bad = []
    have = sorted(p.name for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []
    if not have:
        return ["the generator wrote no child: nothing was checked"]
    if sorted(want_ids) != have:
        bad.append(f"generated children {have} differ from the controls' {sorted(want_ids)}")
    for d in (runs / c for c in have):
        pcp = d / "constant" / "phaseClotProperties"
        if not pcp.is_file():
            bad.append(f"{d.name}: no constant/phaseClotProperties")
            continue
        hit = whole_file_tokens(pcp.read_text())
        if hit:
            bad.append(f"{d.name}: constant/phaseClotProperties names {hit}")
        for f in sorted(p for sub in ("constant", "system") for p in (d / sub).rglob("*")
                        if p.is_file() and p.name != "phaseClotProperties"):
            try:
                t = f.read_text()
            except UnicodeDecodeError:
                continue
            hit = entry_tokens(t)
            if hit:
                bad.append(f"{d.name}: {f.relative_to(d)} has entr(ies) {hit}")
        cj = d / "child.json"
        if cj.is_file():
            hit = sorted(json_names(json.loads(cj.read_text())))
            if hit:
                bad.append(f"{d.name}: child.json names {hit}")
    hit = whole_file_tokens(template_text)
    if hit:
        bad.append(f"the main case's template phaseClotProperties names {hit}")
    return bad


# ---------------------------------------------------------------- self-test

def self_test():
    src = SOLVER.read_text()
    base = solver_failures(src)
    if base:
        print("self-test: the live solver already fails, so no mutation can be judged:")
        for b in base:
            print("  " + b)
        return 1

    def sub(old, new):
        if src.count(old) != 1:
            raise SystemExit(f"self-test anchor not unique ({src.count(old)}): {old!r}")
        return src.replace(old, new)

    solver_mutations = (
        ("one built-in key dropped (phaseNormalBasis, a multi-line-initialiser key)",
         sub('            {"phaseNormalBasis", "content"},\n', ""), "lacks"),
        ("one built-in key added that section 1.2 keeps as an option (coreShell)",
         sub('            {"wallPatchSplit", "split"}\n',
             '            {"wallPatchSplit", "split"},\n            {"coreShell", "false"}\n'),
         "has 1 key(s)"),
        ("one built-in token changed (explicitStabilityBounds on -> true)",
         sub('{"explicitStabilityBounds", "on"}', '{"explicitStabilityBounds", "true"}'),
         "wrong built-in token"),
        ("one deleted key dropped (k1)", sub('            "k1",\n', ""), "s21DeletedKeys lacks"),
        ("one deleted key added that no report deleted (permeabilityFloor)",
         sub('            "skeletonUsBound"\n',
             '            "skeletonUsBound",\n            "permeabilityFloor"\n'),
         "s21DeletedKeys has 1"),
        ("the deleted-key refusal's FatalError removed",
         sub("                    << \" older run from release-4198bb17.\"\n"
             "                    << exit(FatalError);\n            }\n        }\n    }\n"
             "    // THE BOND HOLD",
             "                    << \" older run from release-4198bb17.\";\n"
             "            }\n        }\n    }\n    // THE BOND HOLD"),
         "deleted-key refusal loop no longer refuses"),
        ("the built-in refusal's found() test removed",
         sub("if (phaseClotProperties_.found(builtIn[0]))", "if (false)"),
         "built-in refusal loop no longer refuses"),
        ("a deleted key read again (phi0)",
         sub("    // ---- S21: THE BUILT-IN KEYS ---",
             '    const scalar phi0Read = phaseClotProperties_.lookupOrDefault<scalar>("phi0", 0);\n'
             "    // ---- S21: THE BUILT-IN KEYS ---"),
         "still reads S21-removed"),
        ("a kept option no longer read (projSiteLog)",
         sub('lookupOrDefault<Switch>("projSiteLog"', 'lookupOrDefault<Switch>("projSiteLogX"'),
         "kept option(s) the solver no longer reads"),
    )
    missed = []
    for label, mutated, tag in solver_mutations:
        got = solver_failures(mutated)
        ok = any(tag in g for g in got)
        print(f"  mutation {'caught' if ok else 'NOT CAUGHT'}: {label}"
              f"{'' if ok else ' -- got ' + repr(got)}")
        if not ok:
            missed.append(label)

    work = Path(tempfile.mkdtemp(prefix="s21_v213_"))
    try:
        runs = generate(work)
        ids = children(json.loads(CONTROLS.read_text()))
        tmpl = TEMPLATE_DICT.read_text()
        base = generated_failures(runs, ids, tmpl)
        if base:
            print("self-test: the generated tree already fails:")
            for b in base:
                print("  " + b)
            return 1
        first = sorted(p for p in runs.iterdir() if p.is_dir())[0]
        last = sorted(p for p in runs.iterdir() if p.is_dir())[-1]

        def with_edit(path, text, label, tag, tmpl_text=tmpl):
            saved = path.read_text() if path is not None else None
            if path is not None:
                path.write_text(text(saved))
            try:
                got = generated_failures(runs, ids, tmpl_text)
            finally:
                if path is not None:
                    path.write_text(saved)
            ok = any(tag in g for g in got)
            print(f"  mutation {'caught' if ok else 'NOT CAUGHT'}: {label}")
            if not ok:
                missed.append(label)

        pcp = first / "constant" / "phaseClotProperties"
        with_edit(pcp, lambda t: t + "\ncplusOrder      shared;\n",
                  f"a built-in key re-added to {first.name}'s dictionary",
                  "constant/phaseClotProperties names ['cplusOrder']")
        with_edit(last / "constant" / "phaseClotProperties", lambda t: t + "\nk1 0;\n",
                  f"a deleted key (k1) re-added to {last.name}'s dictionary",
                  "constant/phaseClotProperties names ['k1']")
        with_edit(first / "system" / "fvSolution", lambda t: t + "\nfsiAitken off;\n",
                  "a built-in key as an entry of a child's fvSolution", "system/fvSolution has entr")
        with_edit(first / "child.json",
                  lambda t: json.dumps(dict(json.loads(t), content_transport_scheme="fct")),
                  "a built-in member re-added to a child.json", "child.json names")
        with_edit(None, None, "a built-in key re-added to the main case's template dictionary",
                  "template phaseClotProperties names", tmpl + "\nclotMechanics true;\n")
        shutil.rmtree(last)
        with_edit(None, None, "a child the generator did not write", "differ from the controls'")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if missed:
        print(f"self-test: FAILED ({len(missed)} mutation(s) not caught)")
        return 1
    print("self-test OK: every mutation of the two solver tables, their refusals, the "
          "solver's reads and the generated tree is caught")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    src = SOLVER.read_text()
    problems = solver_failures(src)
    built, deleted = solver_tables(src)
    ids = children(json.loads(CONTROLS.read_text()))
    work = Path(tempfile.mkdtemp(prefix="s21_v213_"))
    try:
        runs = generate(work)
        problems += generated_failures(runs, ids, TEMPLATE_DICT.read_text())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if problems:
        for m in problems:
            print("FAIL: " + m)
        return 1
    print(f"OK: s21BuiltInKeys = section 1.2's {len(built)} built-in keys at their tokens and "
          f"s21DeletedKeys = reports 06/09/10's {len(deleted)} deleted entries, both ways; each "
          f"table refused by a FatalError; none of the {len(GONE)} read elsewhere, all 8 kept "
          f"options read; the generator's {len(ids)} children and the template name none of them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
