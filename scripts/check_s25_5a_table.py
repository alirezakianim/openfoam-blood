#!/usr/bin/env python3
"""S25 B76-5a: the generated 5a children carry physics 09's table, key for key, in the WRITTEN files.

INDEPENDENT of the generator: this reads the files a generation wrote (runs/<child>/constant/phaseClotProperties,
constant/physicalProperties, 0/p, 0/U, 0/nP, 0/Us, system/controlDict, system/blockMeshDict) and compares them with the
table of research/investigations/20261009-2313-physiological-scaling/09-physics-analyst.md (sim units S_L 5/3, S_t 25/3,
S_P 0.04, S_rho 1), typed in below with the table's own digits. A generator that reads the same constants it writes would
agree with itself; this table is typed once, here, from the report.

  usage:  check_s25_5a_table.py <runs_dir>      (a FRESH generation's runs/, not the stale in-tree one)
          check_s25_5a_table.py --self-test <runs_dir>   (one mutation per statement, each must be caught)

Exit 0: every row holds in both children, the pair differs ONLY in elasticCourant (0.5 / 0.25), and every documented
deviation is as documented. Exit 1: any row differs. Exit 2: usage or an absent artefact (an empty population proves nothing).

DOCUMENTED DEVIATIONS from the table (they are asserted too, so a silent change is a FAIL):
  D 1e-10          the table lists 3.3333e-11 (INERT at k0 0); the generator has no per-child D.
  endTime 0.002    the table lists the production window 0.04; the smoke is 2 ms simulated (design 7 B76-5a).
  writeInterval 5e-4   the table lists 5e-3, longer than the run (no full write would exist).
  the numerical keys of the table are D75's and are compared with D75's values.
"""

import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

CHILDREN = ("S25_5a_tan", "S25_5a_tan_half")
# (key, table value, relative tolerance). The tolerance is the table's printed precision, never wider than 1e-6, so a
# 1e-3 slip anywhere is caught. Words are compared exactly.
NUMERIC = {
    "rho": (1050.0, 1e-12), "nu": (6.7636023629e-07, 1e-12),
    "G": (4.68, 1e-12), "poissonRatio": (0.12, 1e-12), "k0p": (9.7222222e-14, 1e-9), "Gc": (0.3698, 1e-9),
    "ell": (1.5e-06, 1e-12), "epsilonC": (3e-06, 1e-12), "networkViscosityTau": (1.2283333e-2, 1e-6),
    "networkReleaseTime": (2.75e-3, 1e-9), "timeScale": (8.333333333333334, 1e-12), "gammaLow": (360.0, 1e-9),
    "gammaHigh": (960.0, 1e-9), "k0": (0.0, 0), "alpha": (0.0, 0), "mobility": (0.0, 0), "D": (1e-10, 1e-12),
    "clotErosionTauPa": (0.12, 1e-9), "clotErosionRate": (0.0, 0), "plateletBindRate": (5.04, 1e-9),
    "plateletAdhesionSpeed": (1.209e-3, 1e-9), "plateletYield": (4.6296e-17, 1e-9), "vP": (4.6296e-17, 1e-9),
    "plateletDiffusivity": (0.0, 0), "skeletonDiluteDragRate": (0.0, 0), "etaP": (1e-05, 1e-12),
    "gradFloor": (666.6666666666666, 1e-12), "bondHoldPenalty": (1e7, 1e-12), "bondHoldDilation": (4.0, 0),
    "skeletonVolumeFraction": (0.45, 1e-12), "residualChi": (2.2222e-4, 1e-4), "residualPhiS": (1e-4, 1e-12),
    "networkPressureCoeff": (0.98, 1e-12), "phiSMax": (0.6, 1e-12), "networkCriticalStretch": (2.8, 1e-12),
    "networkDamageOnsetStretch": (2.105, 1e-12), "networkDamageStretchScale": (1.0, 1e-12), "DgDetach": (0.95, 1e-12),
    "etaRes": (1e-06, 1e-12), "qFloor": (1e-06, 1e-12), "penaltyExponent": (1.0, 0), "enhancementMax": (20.0, 1e-12),
    "fsiRelaxation": (0.5, 1e-12), "releaseSupportLimit": (0.5, 1e-12), "beCorrectors": (3.0, 0),
    "nPhaseCorrectors": (2.0, 0),
}
WORDS = {
    "skeletonLaw": "valanisLandelTrilinear", "elasticCourantModulus": "tangent", "coreShell": "no", "clipPhaseField": "no",
    "plateletTransportOnly": "yes", "momentumPressureCoupling": "blockCoupled", "networkViscosity": "kelvinVoigt",
    "networkPressureWeight": "networkPresence", "materialLabelBasis": "transportedLabel",
}
LT_BLOCK = {"E1": 9.36, "E2": 232.8, "D1": 0.107, "D2": 0.574, "kappa": 0.98}
COURANT = {"S25_5a_tan": 0.5, "S25_5a_tan_half": 0.25}
P_IN = 0.02770371528


def strip(text: str) -> str:
    text = re.sub(r"(?s)/\*.*?\*/", "", text)
    return re.sub(r"//[^\n]*", "", text)


def entries(text: str, key: str) -> list:
    return re.findall(rf"(?m)^\s*{re.escape(key)}\s+(?:\[[^\]]+\]\s+)?([^;{{}}\s]+)\s*;", strip(text))


def block(text: str, key: str):
    m = re.search(rf"(?m)^\s*{key}\s*\{{([^}}]*)\}}", strip(text))
    return None if m is None else {k: v for k, v in re.findall(r"(\w+)\s+([^;\s]+)\s*;", m.group(1))}


def close(a: float, b: float, rel: float) -> bool:
    return a == b if rel == 0 else math.isclose(a, b, rel_tol=rel, abs_tol=0.0)


def check_child(runs: Path, cid: str) -> list:
    bad = []
    case = runs / cid
    if not case.is_dir():
        return [f"{cid}: absent from {runs}"]

    def read(rel: str) -> str:
        p = case / rel
        if not p.is_file():
            bad.append(f"{cid}: {rel} is absent")
            return ""
        return p.read_text()
    props, phys = read("constant/phaseClotProperties"), read("constant/physicalProperties")
    for key, (want, rel) in NUMERIC.items():
        got = entries(props, key)
        if len(got) != 1:
            bad.append(f"{cid}: {key} written {len(got)} times")
            continue
        try:
            x = float(got[0])
        except ValueError:
            bad.append(f"{cid}: {key} = {got[0]!r} is not a number")
            continue
        if not close(x, want, rel):
            bad.append(f"{cid}: {key} = {x!r}, the table has {want!r}")
    for key, want in WORDS.items():
        got = entries(props, key)
        if got != [want]:
            bad.append(f"{cid}: {key} = {got!r}, expected {want!r}")
    got = entries(props, "elasticCourant")
    if got != [repr(COURANT[cid])] and [float(x) for x in got] != [COURANT[cid]]:
        bad.append(f"{cid}: elasticCourant = {got!r}, expected {COURANT[cid]!r}")
    lt = block(props, "skeletonLawCoeffs")
    if lt is None or {k: float(v) for k, v in lt.items()} != LT_BLOCK:
        bad.append(f"{cid}: skeletonLawCoeffs = {lt!r}, the table has {LT_BLOCK!r}")
    g, e1 = entries(props, "G"), (lt or {}).get("E1")
    if g and e1 and float(g[0]) != 0.5 * float(e1):
        bad.append(f"{cid}: G {g[0]} is not exactly E1/2 ({e1})")
    nu_p = entries(phys, "nu")
    if len(nu_p) != 1 or float(nu_p[0]) != float(entries(props, "nu")[0] if entries(props, "nu") else "nan"):
        bad.append(f"{cid}: physicalProperties nu {nu_p!r} differs from phaseClotProperties nu {entries(props, 'nu')!r}")
    # the boundary conditions: every patch of p, U, nP; the bond of Us
    p, u, nP, us = read("0/p"), read("0/U"), read("0/nP"), read("0/Us")

    def patch(text: str, name: str) -> str:
        m = re.search(rf"(?m)^\s*{name}\s*\{{([^}}]*)\}}", text) or re.search(rf"(?m)^\s*{name}\s+\{{([^}}]*)\}}", text)
        return "" if m is None else " ".join(m.group(1).split())
    for fld, text, name, want in (
        ("p", p, "inlet", f"type fixedValue; value uniform {P_IN}"), ("p", p, "outlet", "type fixedValue; value uniform 0"),
        ("p", p, "bottom", "type zeroGradient"), ("p", p, "top", "type zeroGradient"), ("p", p, "front", "type empty"),
        ("U", u, "inlet", "type zeroGradient"), ("U", u, "outlet", "type pressureInletOutletVelocity; value uniform (0 0 0)"),
        ("U", u, "bottom", "type noSlip"), ("U", u, "top", "type noSlip"), ("U", u, "front", "type empty"),
        ("nP", nP, "inlet", "type fixedValue; value uniform 0.0"), ("nP", nP, "outlet", "type zeroGradient"),
    ):
        got = patch(text, name).rstrip(";").strip()
        if got != want:
            bad.append(f"{cid}: 0/{fld} {name} = {got!r}, expected {want!r}")
    for wall in ("bottom", "top"):
        body = patch(us, wall)
        if not re.search(r"adhesiveStrength\s+0\.468;", body) or "sigma sigmaElastic;" not in body:
            bad.append(f"{cid}: 0/Us {wall} lacks adhesiveStrength 0.468 / sigma sigmaElastic: {body[:120]!r}")
    cd = read("system/controlDict")
    for key, want in (("endTime", 0.002), ("writeInterval", 5e-4)):
        got = entries(cd, key)
        if not got or not math.isclose(float(got[0]), want, rel_tol=1e-12):
            bad.append(f"{cid}: controlDict {key} = {got!r}, expected {want!r}")
    bm = read("system/blockMeshDict")
    if not re.search(r"\(448 70 1\)", bm):
        bad.append(f"{cid}: blockMeshDict is not 448 x 70 x 1")
    if (case / "constant" / "throatProfile.json").exists():
        bad.append(f"{cid}: constant/throatProfile.json exists: the 5a channel has NO throat")
    return bad


def check(runs: Path) -> list:
    bad = []
    for cid in CHILDREN:
        bad += check_child(runs, cid)
    a, b = (runs / c / "constant" / "phaseClotProperties" for c in CHILDREN)
    if a.is_file() and b.is_file():
        la, lb = a.read_text().splitlines(), b.read_text().splitlines()
        diff = [(x, y) for x, y in zip(la, lb) if x != y]
        if len(la) != len(lb) or len(diff) != 1 or not diff[0][0].startswith("elasticCourant 0.5") \
                or not diff[0][1].startswith("elasticCourant 0.25"):
            bad.append(f"the pair differs in more than elasticCourant: {len(diff)} lines, {diff[:2]!r}")
    for rel in ("0/p", "0/U", "0/nP", "0/Us", "system/controlDict", "system/blockMeshDict", "constant/physicalProperties"):
        pa, pb = runs / CHILDREN[0] / rel, runs / CHILDREN[1] / rel
        if pa.is_file() and pb.is_file() and pa.read_bytes() != pb.read_bytes():
            bad.append(f"the pair differs in {rel}")
    return bad


def self_test(runs: Path) -> int:
    """One mutation per statement on a scratch copy; every one must turn the check red."""
    muts = (
        ("constant/phaseClotProperties", r"(?m)^G 4\.68;", "G 4.69;", "G one percent off"),
        ("constant/phaseClotProperties", r"(?m)^k0p 9\.7222222e-14;", "k0p 3.5e-14;", "k0p left physical"),
        ("constant/phaseClotProperties", r"(?m)^Gc 0\.3698;", "Gc 5.547;", "Gc left physical"),
        ("constant/phaseClotProperties", r"(?m)^timeScale \S+;", "timeScale 1.0;", "timeScale 1"),
        ("constant/phaseClotProperties", r"(?m)^networkViscosityTau \S+;", "networkViscosityTau 0.0015;", "tau_v unscaled"),
        ("constant/phaseClotProperties", r"(?m)^gammaHigh \S+;", "gammaHigh 8000.0;", "gammaHigh unscaled"),
        ("constant/phaseClotProperties", r"(?m)^nu \[0 2 -1 0 0 0 0\] \S+;", "nu [0 2 -1 0 0 0 0] 3.3333333333e-06;", "nu D75's"),
        ("constant/phaseClotProperties", r"(?m)^    E2 \S+;", "    E2 5820.0;", "E2 physical"),
        ("constant/phaseClotProperties", r"(?m)^networkPressureCoeff \S+;", "networkPressureCoeff 1.0;", "networkPressureCoeff 1"),
        ("constant/phaseClotProperties", r"(?m)^elasticCourantModulus \S+;", "elasticCourantModulus constant;", "CFL on G0"),
        ("constant/phaseClotProperties", r"(?m)^skeletonVolumeFraction \S+;", "skeletonVolumeFraction 0.68;", "phi0 0.68"),
        ("constant/physicalProperties", r"(?m)^nu (\[[^\]]+\] )?\S+;", "nu [0 2 -1 0 0 0 0] 3.3333333333e-6;",
         "physicalProperties nu left at the template's"),
        ("0/p", r"uniform 0\.02770371528", "uniform 0.0737066", "inlet pressure D75's"),
        ("0/U", r"inlet \{ type zeroGradient; \}", "inlet { type fixedValue; value uniform (0.01 0 0); }", "velocity inlet"),
        ("0/Us", r"adhesiveStrength 0\.468;", "adhesiveStrength 0.4;", "adhesiveStrength D75's"),
        ("system/controlDict", r"(?m)^endTime\s+\S+;", "endTime 0.04;", "endTime 0.04"),
        ("system/blockMeshDict", r"\(448 70 1\)", "(448 71 1)", "mesh"),
    )
    missed = 0
    with tempfile.TemporaryDirectory(prefix="s25-5a-table-") as tmp:
        for cid in CHILDREN:
            shutil.copytree(runs / cid, Path(tmp) / cid)
        base = check(Path(tmp))
        if base:
            print("self-test: the unmutated copy already fails:", base[:2])
            return 1
        for rel, pat, new, label in muts:
            f = Path(tmp) / CHILDREN[0] / rel
            orig = f.read_text()
            mutated, n = re.subn(pat, new, orig, count=1)
            if n != 1 or mutated == orig:
                print(f"self-test: MUTATION DID NOT APPLY: {label}")
                missed += 1
                continue
            f.write_text(mutated)
            got = check(Path(tmp))
            f.write_text(orig)
            if not got:
                print(f"self-test: MISSED: {label}")
                missed += 1
        # a throat profile present, and the pair differing in a second key
        tp = Path(tmp) / CHILDREN[0] / "constant" / "throatProfile.json"
        tp.write_text("{}")
        if not check(Path(tmp)):
            print("self-test: MISSED: a throatProfile.json")
            missed += 1
        tp.unlink()
        f = Path(tmp) / CHILDREN[1] / "constant" / "phaseClotProperties"
        orig = f.read_text()
        f.write_text(orig.replace("Gc 0.3698;", "Gc 0.3699;"))
        if not check(Path(tmp)):
            print("self-test: MISSED: the pair differing in Gc")
            missed += 1
        f.write_text(orig)
        # an absent child proves nothing
        shutil.rmtree(Path(tmp) / CHILDREN[1])
        if not check(Path(tmp)):
            print("self-test: MISSED: an absent child")
            missed += 1
    n = len(muts) + 3
    print(f"self-test: {n - missed} of {n} mutations caught")
    return 1 if missed else 0


def main(argv: list) -> int:
    st = "--self-test" in argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1 or not Path(args[0]).is_dir():
        print(__doc__)
        return 2
    runs = Path(args[0])
    if not all((runs / c).is_dir() for c in CHILDREN):
        print(f"check_s25_5a_table: {runs} does not hold {', '.join(CHILDREN)} (an empty population proves nothing)")
        return 2
    if st:
        return self_test(runs)
    bad = check(runs)
    for line in bad:
        print("FAIL", line)
    if bad:
        return 1
    print(f"check_s25_5a_table: OK, {len(NUMERIC)} numeric keys, {len(WORDS)} words, 12 patch rows, the LT block, "
          f"the pair differs in elasticCourant alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
