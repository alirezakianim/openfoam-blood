#!/usr/bin/env python3
"""V21-4: configuration equivalence of the S21 pipeline cut, on raw dictionary tokens.

Design: research/designs/2026-10-05-S21-clean-solver.md section 4, V21-4 (red team 05 S2, S3).

The OLD generator is the shipped one (the backup's case/ directory plus the unedited
smoke template). The NEW generator is the main case as it stands. Both build the 12
pre-existing kept children into scratch trees; nothing in the repository is written.

For each of the 12 children:
  (1) the old dictionary at the 78 built-in keys equals the built-in token (a key the old
      generator left out is the solver's default, which the audit records as the built-in);
  (2) the old dictionary's 8 kept keys equal the new dictionary's (raw token, or absent in both);
  (3) the new constant/phaseClotProperties contains none of the 78 keys and none of the 22
      deleted parameters (s21DeletedKeys, docket report 10): a token grep of the WHOLE file,
      comments included;
  (4) the new dictionary's statements, in order, are the old dictionary's with the 78
      built-in keys and the 22 deleted parameters removed: every other entry is byte-equal;
  (5) every other file of the child (0/, system/, constant/ other than the dictionary) is
      byte-equal, by sha256; child.json equals the old one with the built-in members
      removed and has none of them itself; matrix.json lists the 12 in the old order.
Then, on the whole new matrix (all children), the twins differ from their parents only in
the window; no child.json names a built-in or deleted key; no phaseClotProperties contains
one (whole-file token grep), and no other dictionary under constant/ or system/ has one as an
entry (comments stripped: the shipped fvSolution/fvSchemes comments name four of them).

Mutations, each of which must make its check FAIL (exit 1 if one is not caught):
  M1 a built-in key re-added to a new dictionary          -> (3)
  M2 one parameter value edited in a new dictionary       -> (4)
  M3 one kept key's token edited in a new dictionary      -> (2)
  M4 one old token changed away from its built-in value   -> (1)
  M5 one byte flipped in a new 0/ field                   -> (5)
  M6 a built-in member re-added to a new child.json       -> (5)
  M7 a deleted parameter (k1) re-added to a new dictionary -> (3)
  M8 a deleted parameter re-added to a new child.json      -> (5)
  M9 an fvSchemes edit beyond the div(qPhi,w) removal     -> (5)
  N1 a deleted parameter in a twin's dictionary            -> the all-children check
  N2 a built-in key as an entry of a child's fvSolution    -> the all-children check

Exit 0 only if every check passes and every mutation is caught.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE_CASE = (REPO.parent / "OpenFoam_blood_archive" / "solver_backups"
                / "20261005-pre-clean-release-4198bb17" / "case")
NEW_CASE = REPO / "cases" / "AM2020_planar_mural_detachment"

PRE_EXISTING = ["em76_G4_s18", "em78_G4_s18p_ctl", "em79_G4_s18p_Cnp", "fs_G40_klo_rest_seg",
                "fsb_G40_klo_rest", "fsb_ctl_r3", "fsf_G40_klo_v250", "fsf_G4_kxu_v250",
                "xu5_g1000", "xu5_g2000", "xu5_g250", "xu5_g500"]
TWINS = {"em78_id": "em78_G4_s18p_ctl", "xu5_id": "xu5_g250", "fsf_id": "fsf_G4_kxu_v250"}
KEPT8 = ["coreShell", "materialLabelBasis", "momentumPressureCoupling", "networkPressureWeight",
         "networkViscosity", "plateletTransportOnly", "projSiteLog", "releaseCoreExempt"]

# The 78 built-in keys and the token the clean solver runs: S21 section 1.2 and the key audit
# rev. 2 (docket 20261005-1621, probes/ce-s21-key-audit.txt, column "kept values"). Switch keys
# are True/False; every word key is its raw token. Typed here, not read from the generator, so
# the check does not share a table with the code it judges.
T, F = True, False
BUILT_IN = {
    "advectReleased": T, "beDetRenormalise": T, "beFluxConsistentGradU": T,
    "beRelaxBasis": "networkAbsent", "beStretchGradient": "fluxConsistent",
    "beTermDiagnostic": T, "beTraceFreeSource": F, "blockCoupledLumen": "eliminate",
    "blockFaceInverse": "faceAssembled", "bondHoldCarrier": T, "bondHoldTaper": F,
    "clotMechanics": T, "constitutiveModel": "neoHookean", "contentModel": T,
    "contentOperator": "content", "contentTransportScheme": "fct", "cplusOrder": "shared",
    "damageTransport": "contentPartition", "dragDriver": "legacy",
    "dragOccupancyBasis": "intrinsic", "elasticCoupling": "implicitTangent",
    "elasticStabilityStiffness": "tangent", "eulerianBrinkman": T,
    "eulerianMaterialFluxIsPhi": T, "eulerianSolid": T, "explicitStabilityBounds": "on",
    "faceConsistentElasticForce": F, "fibreArithmetic": "fieldUnits", "fibreSlack": "v1",
    "finiteStrain": T, "frozenPendingPressure": "none", "fsiAitken": F, "fsiSweepTrace": T,
    "growthClosure": "reaction", "interfaceDepletionModel": "none", "materialCarrierBlend": F,
    "networkDamageProgressive": T, "networkPressureDamage": "tensile",
    "networkPressureForce": "faceGradient", "networkPressureModel": "convexExcess",
    "networkPressureReference": "deposited", "networkRupture": "fibreRelease",
    "operatorAudit": F, "operatorLocalisationAudit": F, "permeabilityModel": "kozenyCarmanSolid",
    "phaseDeltaBasis": "content", "phaseFieldMaterialVelocity": T,
    "phaseFluidFractionBasis": "content", "phaseForceBasis": "content",
    "phaseGrowthDriver": "transfer", "phaseIndicatorBasis": "content",
    "phaseLimiterBasis": "content", "phaseMomentumFluxBasis": "shared",
    "phaseNormalBasis": "content", "phasePlateletFluxBasis": "content",
    "phaseWeightBasis": "content", "plateletBindLocalisation": "interface",
    "plateletCarrierArchive": F, "plateletContentFormulation": T, "plateletContentModel": T,
    "plateletFluxConsistent": F, "plateletFluxConvention": "intrinsic",
    "plateletOuterForceReject": F, "plateletOuterTransactions": F, "releasedCarrier": "material",
    "shearEnhancementModel": "thresholdSaturating", "skeletonDiffuseDomain": T,
    "skeletonFloorBasis": "occupancy", "skeletonMomentumForm": "phaseContinuity",
    "skeletonPressureForce": F, "skeletonResidualDiagnostic": F, "skeletonWallFace": "zeroed",
    "stressFaceTransmission": "series", "stronglyCoupledFSI": F, "twoVelocitySkeleton": T,
    "unlocalisedPhaseWeightsAccepted": F, "vnCapEnabled": F, "wallPatchSplit": "split",
}
assert len(BUILT_IN) == 78 and not set(BUILT_IN) & set(KEPT8)
YES, NO = {"yes", "on", "true", "y", "1"}, {"no", "off", "false", "n", "0"}


def snake(key: str) -> str:
    return re.sub(r"(?<=[a-z0-9])([A-Z])", lambda m: "_" + m.group(1).lower(), key).lower()


# The 22 parameters S21 deleted (solver table s21DeletedKeys; docket 20261005-1621,
# probes/se10-s21c1-deleted-keys.txt), typed here for the same reason as BUILT_IN.
DELETED = {"releasedAmbientQ", "releasedMaskDilation", "releasedMaskLimit", "releasedMaskSweeps",
           "fsiNormProbeFraction", "fsiOmegaMax", "fsiOmegaMin", "fsiTolerance", "k1",
           "materialCourant", "materialDeltaTMin", "nMaterialFloorSteps", "nFSICorrectors",
           "nSkeletonCorrectors", "nStaggeredCorrectors", "nSkeletonDivergedSteps",
           "skeletonDivergenceGrowth", "operatorAuditEvery", "phi0", "skeletonFixedPointTolerance",
           "skeletonImplicitFactor", "skeletonUsBound"}
assert len(DELETED) == 22 and not DELETED & set(BUILT_IN) and not DELETED & set(KEPT8)
GONE = set(BUILT_IN) | DELETED
SNAKE = {snake(k) for k in GONE}


def norm(expected, token):
    """token compared with the built-in: switches by meaning, words raw."""
    if isinstance(expected, bool):
        t = token.lower()
        return t in YES if expected else t in NO
    return token == expected


def statements(path: Path):
    """The dictionary as an ordered list of (key, value text): comments and blank lines dropped,
    and the FoamFile header block skipped."""
    text = re.sub(r"//[^\n]*", "", path.read_text())
    text = re.sub(r"(?s)FoamFile\s*\{.*?\}", "", text, count=1)
    out = []
    for m in re.finditer(r"(?m)^([A-Za-z]\w*)\s+([^;]*);", text):
        out.append((m.group(1), " ".join(m.group(2).split())))
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path, skip=()):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and str(p.relative_to(root)) not in skip}


def strip_builtin(obj):
    """A child.json / matrix entry without the built-in members; dicts left empty are dropped."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in GONE or k in SNAKE:
                continue
            v = strip_builtin(v)
            if isinstance(v, dict) and not v:
                continue
            out[k] = v
        return out
    if isinstance(obj, list):
        return [strip_builtin(x) for x in obj]
    return obj


def names_builtin(obj):
    hit = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in GONE or k in SNAKE:
                hit.add(k)
            hit |= names_builtin(v)
    elif isinstance(obj, list):
        for x in obj:
            hit |= names_builtin(x)
    return hit


def build(case_src: Path, label: str, work: Path, ids, template_src: Path | None, gate_src: Path):
    """Copy a case (without runs/) next to the mural gate (and the smoke template if the case has
    none of its own), run its generator, return the runs directory."""
    root = work / label / "cases"
    case = root / "AM2020_planar_mural_detachment"
    shutil.copytree(case_src, case, ignore=shutil.ignore_patterns("runs", "__pycache__", ".claude"))
    shutil.copytree(gate_src, root / "AM2020_mural_geometry_gate",
                    ignore=shutil.ignore_patterns("__pycache__"))
    if template_src is not None:
        dst = root / "AM2020_straight_tube_growth_mechanics_smoke" / "template"
        dst.parent.mkdir(parents=True)
        shutil.copytree(template_src, dst)
    r = subprocess.run([sys.executable, "-B", "scripts/create_cases.py", *ids], cwd=case,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{label} generator exit {r.returncode}: {(r.stdout + r.stderr)[-600:]}")
    return case / "runs"


# ---- the checks: each returns a list of failure strings (empty = pass) ---------------------
def check_old_tokens(old_runs, ids):
    bad = []
    for cid in ids:
        toks = dict(statements(old_runs / cid / "constant" / "phaseClotProperties"))
        for key, want in BUILT_IN.items():
            if key in toks and not norm(want, toks[key]):
                bad.append(f"(1) {cid}: {key} is {toks[key]!r} in the old dictionary, built-in {want!r}")
    return bad


def check_kept(old_runs, new_runs, ids):
    bad = []
    for cid in ids:
        o = dict(statements(old_runs / cid / "constant" / "phaseClotProperties"))
        n = dict(statements(new_runs / cid / "constant" / "phaseClotProperties"))
        for key in KEPT8:
            if o.get(key) != n.get(key):
                bad.append(f"(2) {cid}: {key} old {o.get(key)!r} new {n.get(key)!r}")
    return bad


def check_no_builtin_in_new(new_runs, ids):
    bad = []
    for cid in ids:
        text = (new_runs / cid / "constant" / "phaseClotProperties").read_text()
        for key in sorted(GONE):
            if re.search(rf"\b{key}\b", text):
                bad.append(f"(3) {cid}: the new phaseClotProperties contains {key}")
    return bad


def check_statements(old_runs, new_runs, ids):
    bad = []
    for cid in ids:
        old = [(k, v) for k, v in statements(old_runs / cid / "constant" / "phaseClotProperties")
               if k not in GONE]
        new = statements(new_runs / cid / "constant" / "phaseClotProperties")
        if old != new:
            first = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
            bad.append(f"(4) {cid}: statements differ at #{first}: old "
                       f"{old[first] if first < len(old) else None} new {new[first] if first < len(new) else None}"
                       f" ({len(old)} old, {len(new)} new)")
    return bad


def pinned_to_s22(new_runs, cid) -> bool:
    """S23 rev. 4.1 section 13.2 item 5 (docket 20261007-0840, case-engineer report 25): a child pinned to
    solver_library s22 gains exactly two things over the pre-S21 generator's output, a child.env and the
    child.json member `solver_library`. Nothing else may differ (the checks below still compare it all)."""
    return json.loads((new_runs / cid / "child.json").read_text()).get("solver_library") == "s22"


def check_files(old_runs, new_runs, ids):
    bad = []
    skip = ("constant/phaseClotProperties", "child.json")
    for cid in ids:
        pin = pinned_to_s22(new_runs, cid)
        if pin and not (new_runs / cid / "child.env").is_file():
            bad.append(f"(5) {cid}: pinned to s22 but has no child.env")
        sk = skip + (("child.env",) if pin else ())
        o, n = tree_hashes(old_runs / cid, sk), tree_hashes(new_runs / cid, sk)
        # S21 pass 2b: the one fvSchemes entry no solver line reads any more, div(qPhi,w)
        # (the fvm::div(qPhi, w_) call left with the w branch), is removed; the old file
        # minus exactly that line must equal the new one byte for byte.
        fv = "system/fvSchemes"
        if fv in o and fv in n:
            old_t = (old_runs / cid / fv).read_text()
            cut = old_t.replace("    div(qPhi,w) bounded Gauss upwind;\n", "", 1)
            if cut != old_t and cut == (new_runs / cid / fv).read_text():
                o[fv] = n[fv]
        if o != n:
            diff = sorted(set(o) ^ set(k for k in n if o.get(k) != n[k]) | {k for k in o if n.get(k) != o[k]})
            bad.append(f"(5) {cid}: files differ: {diff[:4]}")
        oj = json.loads((old_runs / cid / "child.json").read_text())
        nj = json.loads((new_runs / cid / "child.json").read_text())
        sj = strip_builtin(nj)
        if pin:
            sj.pop("solver_library", None)          # the one member the pin adds (an s22 child.json names it)
        if strip_builtin(oj) != sj:
            keys = sorted(k for k in set(strip_builtin(oj)) | set(sj)
                          if strip_builtin(oj).get(k) != sj.get(k))
            bad.append(f"(5) {cid}: child.json differs beyond the built-in members: {keys[:5]}")
        if names_builtin(nj):
            bad.append(f"(5) {cid}: the new child.json names {sorted(names_builtin(nj))[:4]}")
    return bad


def check_matrix(old_runs, new_runs, ids):
    om = [c["child_id"] for c in json.loads((old_runs / "matrix.json").read_text())]
    nm = [c["child_id"] for c in json.loads((new_runs / "matrix.json").read_text()) if c["child_id"] in om]
    return [] if om == nm else [f"(5) matrix.json order differs: old {om} new {nm}"]


def run_all(old_runs, new_runs, ids):
    return (check_old_tokens(old_runs, ids) + check_kept(old_runs, new_runs, ids)
            + check_no_builtin_in_new(new_runs, ids) + check_statements(old_runs, new_runs, ids)
            + check_files(old_runs, new_runs, ids) + check_matrix(old_runs, new_runs, ids))


def check_twins(new_runs):
    bad = []
    for twin, parent in TWINS.items():
        o, n = tree_hashes(new_runs / parent, ("child.json",)), tree_hashes(new_runs / twin, ("child.json",))
        differ = {k for k in set(o) | set(n) if o.get(k) != n.get(k)}
        if differ - {"system/controlDict"}:
            bad.append(f"twin {twin}: files other than controlDict differ from {parent}: {sorted(differ)[:4]}")
        cd = [l for l in difflines(new_runs / parent / "system/controlDict", new_runs / twin / "system/controlDict")]
        if any(not re.match(r"\s*(endTime|writeInterval|stopTime|maxDeltaT)\b", l[1:]) for l in cd):
            bad.append(f"twin {twin}: controlDict differs from {parent} beyond the window: {cd[:4]}")
    return bad


def difflines(a: Path, b: Path):
    import difflib
    return [l for l in difflib.unified_diff(a.read_text().splitlines(), b.read_text().splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def check_new_all(new_runs):
    bad = []
    for d in sorted(p for p in new_runs.iterdir() if p.is_dir()):
        text = (d / "constant" / "phaseClotProperties").read_text()
        for key in sorted(GONE):
            if re.search(rf"\b{key}\b", text):
                bad.append(f"{d.name}: dictionary contains {key}")
        for f in sorted(p for sub in ("constant", "system") for p in (d / sub).rglob("*")
                        if p.is_file() and p.name != "phaseClotProperties"):
            try:
                t = f.read_text()
            except UnicodeDecodeError:
                continue
            t = re.sub(r"(?s)/\*.*?\*/", "", re.sub(r"//[^\n]*", "", t))
            hit = sorted(set(re.findall(r"(?m)^\s*([A-Za-z]\w*)\b", t)) & GONE)
            if hit:
                bad.append(f"{d.name}: {f.relative_to(d)} has entries {hit[:4]}")
        hit = names_builtin(json.loads((d / "child.json").read_text()))
        if hit:
            bad.append(f"{d.name}: child.json names {sorted(hit)[:4]}")
    return bad


def mutate(new_runs, work, kind):
    """a copy of the new runs with one defect; returns its path and the check that must see it"""
    m = work / ("mut_" + kind)
    shutil.copytree(new_runs, m)
    cid = PRE_EXISTING[7]
    props = m / cid / "constant" / "phaseClotProperties"
    if kind == "M1":
        props.write_text(props.read_text().rstrip("\n") + "\n\nexplicitStabilityBounds on;\n")
    elif kind == "M2":
        t = props.read_text(); t2 = re.sub(r"(?m)^(G\s+)4\.0;", r"\g<1>4.1;", t, count=1); assert t != t2
        props.write_text(t2)
    elif kind == "M3":
        t = props.read_text(); t2 = re.sub(r"(?m)^(coreShell\s+)no;", r"\g<1>yes;", t, count=1); assert t != t2
        props.write_text(t2)
    elif kind == "M5":
        f = next(p for p in sorted((m / cid / "0").glob("*")) if p.is_file() and p.stat().st_size > 200)
        b = bytearray(f.read_bytes()); b[-5] ^= 0x01; f.write_bytes(bytes(b))
    elif kind == "M6":
        p = m / cid / "child.json"; j = json.loads(p.read_text()); j["fibreSlack"] = "v1"
        p.write_text(json.dumps(j, indent=2) + "\n")
    elif kind == "M7":
        props.write_text(props.read_text().rstrip("\n") + "\n\nk1 [-1 2 1 0 0 0 0] 0;\n")
    elif kind == "M8":
        p = m / cid / "child.json"; j = json.loads(p.read_text()); j["skeleton_implicit_factor"] = 1.0
        p.write_text(json.dumps(j, indent=2) + "\n")
    elif kind == "M9":
        f = m / cid / "system" / "fvSchemes"
        t = f.read_text(); t2 = t.replace("bounded Gauss", "Gauss", 1); assert t != t2
        f.write_text(t2)
    elif kind == "N1":
        p = m / "fsf_id" / "constant" / "phaseClotProperties"
        p.write_text(p.read_text().rstrip("\n") + "\n\nnFSICorrectors 60;\n")
    elif kind == "N2":
        p = m / cid / "system" / "fvSolution"
        p.write_text(p.read_text().rstrip("\n") + "\n\noperatorAudit no;\n")
    return m


def main() -> int:
    old_case = Path(sys.argv[1]) if len(sys.argv) > 1 else ARCHIVE_CASE
    new_case = Path(sys.argv[2]) if len(sys.argv) > 2 else NEW_CASE
    gate = REPO / "cases" / "AM2020_mural_geometry_gate"
    smoke_template = REPO / "cases" / "AM2020_straight_tube_growth_mechanics_smoke" / "template"
    for p in (old_case / "scripts" / "create_cases.py", new_case / "scripts" / "create_cases.py",
              gate, smoke_template, new_case / "template"):
        if not p.exists():
            print(f"MISSING {p}"); return 2
    work = Path(tempfile.mkdtemp(prefix="s21_v214_"))
    try:
        old_runs = build(old_case, "old", work, PRE_EXISTING, smoke_template, gate)
        new_runs = build(new_case, "new", work, [], None, gate)       # the whole kept matrix
        have = sorted(p.name for p in new_runs.iterdir() if p.is_dir())
        print(f"old generator: {len(PRE_EXISTING)} children; new generator: {len(have)} children: {have}")
        print("\nRAW TOKENS of the 78 built-in keys in the OLD dictionaries (12 children), against the built-in:")
        toks = {c: dict(statements(old_runs / c / "constant" / "phaseClotProperties")) for c in PRE_EXISTING}
        for key, want in BUILT_IN.items():
            cnt = Counter(toks[c].get(key, "<absent>") for c in PRE_EXISTING)
            print(f"  {key:33s} built-in {str(want):20s} old tokens {dict(cnt)}")
        print("\nRAW TOKENS of the 8 kept keys, old | new, per child:")
        for c in PRE_EXISTING:
            n = dict(statements(new_runs / c / "constant" / "phaseClotProperties"))
            print(f"  {c:21s} " + " ".join(f"{k}={toks[c].get(k, '-')}|{n.get(k, '-')}" for k in KEPT8))
        print("\nINVENTORY: statements per dictionary (old -> new, minus the 78 built-in and 22 deleted): " + ", ".join(
            f"{c}:{len(toks[c])}->{len(statements(new_runs / c / 'constant' / 'phaseClotProperties'))}"
            for c in PRE_EXISTING))
        fails = run_all(old_runs, new_runs, PRE_EXISTING) + check_twins(new_runs) + check_new_all(new_runs)
        for f in fails:
            print("FAIL", f)
        print(f"\nV21-4 checks: {'PASS' if not fails else 'FAIL'} ({len(fails)} failures)")
        missed = []
        want = {"M1": "(3)", "M2": "(4)", "M3": "(2)", "M5": "(5)", "M6": "(5)", "M7": "(3)",
                "M8": "(5)", "M9": "(5)"}
        for kind, tag in want.items():
            m = mutate(new_runs, work, kind)
            got = run_all(old_runs, m, PRE_EXISTING)
            ok = any(g.startswith(tag) for g in got)
            print(f"mutation {kind}: {'caught' if ok else 'NOT CAUGHT'} by check {tag} ({len(got)} failure lines)")
            if not ok:
                missed.append(kind)
            shutil.rmtree(m)
        for kind in ("N1", "N2"):
            m = mutate(new_runs, work, kind)
            got = check_new_all(m)
            ok = bool(got)
            print(f"mutation {kind}: {'caught' if ok else 'NOT CAUGHT'} by the all-children check "
                  f"({got[:1]})")
            if not ok:
                missed.append(kind)
            shutil.rmtree(m)
        # M4: an old token away from its built-in value
        p = old_runs / PRE_EXISTING[7] / "constant" / "phaseClotProperties"
        saved = p.read_text(); p.write_text(re.sub(r"(?m)^(contentTransportScheme\s+)fct;", r"\g<1>upwind;", saved, count=1))
        ok = any(g.startswith("(1)") for g in check_old_tokens(old_runs, PRE_EXISTING))
        p.write_text(saved)
        print(f"mutation M4: {'caught' if ok else 'NOT CAUGHT'} by check (1)")
        if not ok:
            missed.append("M4")
        rc = 0 if not fails and not missed else 1
        print(f"\nRESULT: {'V21-4 OK' if rc == 0 else 'V21-4 FAILED'} (mutations missed: {missed or 'none'})")
        return rc
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
