#!/usr/bin/env python3
"""D74 guard: the passive intact reference ref_I (thetaNetRefIntact_) has NO READER but its own.

research/designs/2026-10-09-D74-passive-intact-reference.md rev. 2, section 2 (red team 36
changes 2 and 3; docket research/investigations/20261007-1807-s24-consistent-network-stress/,
reports 32, 33, 35, 36). ref_I is a MEASUREMENT: no law may read it, so theta, ref, the partition
and every momentum face value stay those of the build without it. Its only sites are:

  seed/write  the constructor: READ_IF_PRESENT + AUTO_WRITE, all-or-none with the partition,
              thetaNetRef's patch types, seeded ref_I = ref after ref is seeded;
  transport   ref's OWN label operator (ddt + convection - Sp(surfaceIntegrate(carrier))), solved
              with ref's solver dictionary built in code at tolerance 0, minIter = maxIter = ref's
              executed iterations; the FCT part r_d*lambda*AR, r_d = ref_I^low/ref^low of the donor,
              the denominator guarded, lambda and AR READ (never written), before fctApply;
  transfers   rupture ref_I*pI_new/pI_old where pI_old > 0; the volumetric tear (A1's event) 0;
              growth ref's own increment thetaNew - thetaStar; deletion 0;
  checkpoint  beginPlateletOuterTransaction, the corrector roll-back, the negAfter count;
  log         PHASE_CLOT_REFI from reportIntactReference() in postSolve(), theta_I = (P1/S) theta.

Each predicate pins source text (comments and string literals stripped); the reader census pins,
per solver function, how often ref_I and each of its local aliases is named, so a law that reads
it -- in a new function or in an allowed one -- changes a count. Every predicate has at least one
planted defect that must be CAUGHT; design section 2's six (a law reads it, a transfer omitted, the
write missing, lambda written by the kernel, the Sp term dropped, the restore omitted) are labelled
(a)-(f). A no-op control edit must change nothing (MISSED).

usage: python3 -B scripts/check_d74_refI.py      (exit 0 = pass, 1 = fail)
Written by solver-engineer 37 (docket 20261007-1807-s24-consistent-network-stress); the min and p01 of
the signed proxy deviation (the pole side; CE ruling 2026-10-09) pinned by solver-engineer 39.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "user-applications"
SOLVER = APP / "phaseClotAxisymmetricFluid" / "phaseClotAxisymmetricFluid.C"
HEADER = APP / "phaseClotAxisymmetricFluid" / "phaseClotAxisymmetricFluid.H"

FIELD = "thetaNetRefIntact_"


def code_only(t: str) -> str:
    """Comments removed, string literals emptied, newlines kept (positions stay per line)."""
    def sub(m: re.Match) -> str:
        g = m.group(0)
        if g.startswith("/*"):
            return re.sub(r"[^\n]", " ", g)
        return '""' if g.startswith('"') else ""
    return re.sub(r'"(?:\\.|[^"\\\n])*"|//[^\n]*|/\*.*?\*/', sub, t, flags=re.S)


def ns(t: str) -> str:
    return re.sub(r"\s+", "", code_only(t))


DEF = re.compile(r"^(?:[A-Za-z_][\w:<>, ]*\s)?Foam::solvers::phaseClotAxisymmetricFluid::(~?\w+)\s*(?:\(|$)",
                 re.M)


def census(c: str, name: str) -> dict:
    """Mentions of `name` (whole word, code only) attributed to the solver function they sit in."""
    code = code_only(c)
    defs = [(m.start(), m.group(1)) for m in DEF.finditer(code)]
    out = Counter()
    for m in re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", code):
        owner = [n for p, n in defs if p <= m.start()]
        out[owner[-1] if owner else "<file scope>"] += 1
    return dict(out)


def fn_body(c: str, name: str) -> str:
    """Flattened code of the solver function `name` (definition to the next definition)."""
    code = code_only(c)
    defs = [(m.start(), m.group(1)) for m in DEF.finditer(code)]
    for k, (p, n) in enumerate(defs):
        if n == name:
            end = defs[k + 1][0] if k + 1 < len(defs) else len(code)
            return re.sub(r"\s+", "", code[p:end])
    return ""


# ---- the reader census: every function that names ref_I, and how often ------------------------
EXPECT_FIELD = {
    "phaseClotAxisymmetricFluid": 6,      # construction, all-or-none, patch types, seed
    "ruptureOverstretchedNetwork": 2,     # refIW alias, patches
    "releaseDetachedMaterial": 3,         # booked, zeroed, patches
    "beginPlateletOuterTransaction": 2,   # checkpoint, inventory
    "solvePhaseField": 2,                 # roll-back
    "solveContentComponents": 10,         # predictor, balance, FCT part, negAfter
    "solvePlateletContent": 2,            # growth
    "reportIntactReference": 1,           # the log
}
EXPECT_ALIASES = {
    "refIW": {"ruptureOverstretchedNetwork": 6},
    "refIOld": {"ruptureOverstretchedNetwork": 3},
    "pIOldD74": {"ruptureOverstretchedNetwork": 3},
    "refITransfer": {"ruptureOverstretchedNetwork": 4},
    "refIR": {"reportIntactReference": 11},
    "refILowI": {"solveContentComponents": 2},
    "refIC": {"solveContentComponents": 3},
    "refISolved": {"solveContentComponents": 2},
    "thetaNetRefIntactStepAccepted_": {"beginPlateletOuterTransaction": 1, "solvePhaseField": 2},
}


def p_no_reader(c: str, h: str = "") -> bool:
    h = h or HEADER.read_text(encoding="utf-8")
    if census(c, FIELD) != EXPECT_FIELD:
        return False
    for alias, want in EXPECT_ALIASES.items():
        if census(c, alias) != want:
            return False
    hc = code_only(h)
    # the header only declares it: no inline reader
    if len(re.findall(r"(?<![A-Za-z0-9_])thetaNetRefIntact_(?![A-Za-z0-9_])", hc)) != 1:
        return False
    # no registry lookup by name anywhere in the solver sources
    for f in APP.rglob("*"):
        if f.suffix in (".C", ".H") and f.is_file() and f.name not in (SOLVER.name, HEADER.name):
            if "thetaNetRefIntact" in f.read_text(encoding="utf-8", errors="replace"):
                return False
    return re.search(r"lookupObject[^;]*thetaNetRefIntact", c) is None


# ---- pinned flattened text ---------------------------------------------------------------------
CONSTRUCT = ('thetaNetRefIntact_(IOobject("",runTime.name(),mesh,IOobject::READ_IF_PRESENT,'
             'IOobject::AUTO_WRITE),mesh,dimensionedScalar("",dimless,0),'
             'zeroGradientFvPatchField<scalar>::typeName),refISumStepStart_(0),')
SEED = ('{constboolhasRefI=thetaNetRefIntact_.headerOk();constboolhasPartition=thetaIntact_.headerOk()'
        '||thetaPending_.headerOk()||thetaBroken_.headerOk();if(hasRefI!=hasPartition){FatalErrorInFunction')
SEED_TYPES = ('if(thetaNetRefIntact_.boundaryField().types()!=thetaNetRef_.boundaryField().types())'
              '{FatalErrorInFunction')
SEED_SET = 'if(!hasRefI){thetaNetRefIntact_=thetaNetRef_;thetaNetRefIntact_.correctBoundaryConditions();}'
REF_SEED = 'thetaNetRef_=thetaC_+thetaH_;thetaNetRef_.correctBoundaryConditions();}}'

REF_SOLVE = ('-fvm::Sp(fvc::surfaceIntegrate(carrier),thetaNetRef_));'
             'constlabelrefIters=thetaNetRefEqn.solve().nIterations();'
             'thetaNetRef_.correctBoundaryConditions();')
CONTROLS = ('{dictionaryrefIControls(mesh_.solution().solverDict(thetaNetRef_.select('
            '!mesh_.schemes().steady()&&solutionControl::finalIteration(mesh_))));'
            'refIControls.set("",scalar(0));refIControls.set("",scalar(0));'
            'refIControls.set("",refIters);refIControls.set("",refIters);')
PREDICTOR = ('fvScalarMatrixthetaNetRefIntactEqn(fvm::ddt(thetaNetRefIntact_)'
             '+phaseConservativeConvectionWith(carrier,thetaNetRefIntact_,tContentWeights())'
             '-fvm::Sp(fvc::surfaceIntegrate(carrier),thetaNetRefIntact_));'
             'thetaNetRefIntactEqn.solve(refIControls);thetaNetRefIntact_.correctBoundaryConditions();')
CONTROL_KEYS = re.compile(r'refIControls\.set\(\s*"tolerance",\s*scalar\(0\)\);\s*'
                          r'refIControls\.set\(\s*"relTol",\s*scalar\(0\)\);\s*'
                          r'refIControls\.set\(\s*"minIter",\s*refIters\);\s*'
                          r'refIControls\.set\(\s*"maxIter",\s*refIters\);')

FCT_AFTER = "fctLimiter(thetaNetRef_,thetaNetRef_.oldTime(),tAR()));}"
FCT_PART = ('if(tAR.valid()){constlabelUList&ownI=mesh_.owner();constlabelUList&neiI=mesh_.neighbour();'
            'constscalarField&lamI=lambda.primitiveField();constscalarField&aRI=tAR().primitiveField();'
            'constscalarField&refLowI=thetaNetRef_.primitiveField();'
            'constscalarField&refILowI=thetaNetRefIntact_.primitiveField();'
            'constscalarField&VI=mesh_.V();constscalardtI=runTime.deltaTValue();'
            'scalarFielddivI(mesh_.nCells(),0.0);forAll(ownI,facei){'
            'constscalarlaI=lamI[facei]*aRI[facei];'
            'constlabeldonorI=laI>0?ownI[facei]:neiI[facei];'
            'constscalarrD=refLowI[donorI]!=0?refILowI[donorI]/refLowI[donorI]:0.0;'
            'constscalarfluxI=rD*laI;divI[ownI[facei]]+=fluxI;divI[neiI[facei]]-=fluxI;}'
            'scalarField&refIC=thetaNetRefIntact_.primitiveFieldRef();'
            'forAll(refIC,celli){refIC[celli]-=dtI/VI[celli]*divI[celli];}'
            'thetaNetRefIntact_.correctBoundaryConditions();}')
FCT_BEFORE = "fctApply(thetaC_,lambda,AC);"
NEG_REF = "if(thetaNetRef_[celli]<0){++negAfter;}"
NEG_REFI = "if(thetaNetRefIntact_[celli]<0){++negAfter;}"

CHECKPOINT = ('thetaNetRefIntactStepAccepted_=tmp<volScalarField>(newvolScalarField("",thetaNetRefIntact_));')
ROLLBACK = ('thetaNetRefIntact_=thetaNetRefIntactStepAccepted_();'
            'thetaNetRefIntact_.correctBoundaryConditions();refITransferStep_=0;')
REF_ROLLBACK = ('thetaNetRef_=thetaNetRefStepAccepted_();thetaNetRef_.correctBoundaryConditions();}')

RUP_OLD = "constscalarpIOldD74=pI[celli];scalardAfter=d;"
RUP_LAW = ('if(pIOldD74>0){constscalarrefIOld=refIW[celli];refIW[celli]=refIOld*(pI[celli]/pIOldD74);'
           'refITransfer+=(refIW[celli]-refIOld)*vol[celli];}')
RUP_TEAR = ('if(sC>0&&volTear&&pI[celli]==0){refITransfer-=refIW[celli]*vol[celli];refIW[celli]=0.0;}'
            'deltaD=max(deltaD,mag(dAfter-dOld));')
RUP_ALIAS = "scalarField&refIW=thetaNetRefIntact_.primitiveFieldRef();scalarrefITransfer=0;"
RUP_TAIL = "thetaNetRefIntact_.correctBoundaryConditions();refITransferStep_+=refITransfer;"
A1 = "if(volTear&&pI[celli]==0){if(!refW){refW=&thetaNetRef_.primitiveFieldRef();}(*refW)[celli]=theta;}"
GROWTH = ('thetaNetRef_+=thetaNew-thetaStar;thetaNetRef_.correctBoundaryConditions();'
          '{thetaNetRefIntact_+=thetaNew-thetaStar;thetaNetRefIntact_.correctBoundaryConditions();')
DELETION = ('thetaNetRef_[celli]=0.0;{refITransferStep_-=thetaNetRefIntact_[celli]*mesh.V()[celli];'
            'thetaNetRefIntact_[celli]=0.0;}')
DELETION_BC = "thetaNetRef_.correctBoundaryConditions();thetaNetRefIntact_.correctBoundaryConditions();"

LOG_SHARES = ('fibreReleaseKernel::shares(P1R[celli],P2R[celli],P3R[celli],aIs,aPs,omdS);'
              'constscalardStar=(P2R[celli]+P3R[celli])/S;'
              'constscalarthetaI=aIs*(tCR[celli]+tHR[celli]);'
              'constscalaraI=phi0R+thetaI-refIR[celli];')
LOG_PROXY = ('if(dStar>0&&dStar<1){constscalaraIProxy=phi0R+thetaI-aIs*refR[celli];'
             'proxyDev.push_back(aIProxy-aI);}')
# the pole side of the SIGNED deviation (CE ruling 2026-10-09): p01 and min over the SAME vector
LOG_QUANT = ('conststd::size_tk01=std::min(nD-1,std::size_t(0.01*scalar(nD-1)+0.5));'
             'std::nth_element(proxyDev.begin(),proxyDev.begin()+k01,proxyDev.end());'
             'proxyDevP01=proxyDev[k01];'
             'proxyDevMin=*std::min_element(proxyDev.begin(),proxyDev.end());')
LOG_PRINT = re.compile(r'<<\s*" proxyDevMax="\s*<<\s*proxyDevMax\s*'
                       r'<<\s*" proxyDevP01="\s*<<\s*proxyDevP01\s*'
                       r'<<\s*" proxyDevMin="\s*<<\s*proxyDevMin\s*<<')
LOG_KEYS = ("time", "step", "aICells", "aIMax", "aIMaxCell", "aIMaxThetaI", "aIMaxRefI", "aIMaxDStar",
            "nearPole", "poleBand", "proxyDevCells", "proxyDevP99", "proxyDevMax", "proxyDevP01",
            "proxyDevMin", "refMax", "refIMin",
            "excessMax", "p1ZeroCells", "p1ZeroRefIMax", "s0RefCells", "dStar0Cells", "dStar0Unequal",
            "dStar0RelDevMax", "refISum", "dSum", "transfer", "outflux", "spSource", "balanceResidual")
LOG_RAW = re.compile(r'Info<<\s*"PHASE_CLOT_REFI time="' + "".join(
    r'[^;]*?<<\s*" ' + k + r'="' for k in LOG_KEYS[1:]) + r'[^;]*?<<\s*endl;')


def p_seed_and_write(c: str) -> bool:
    b = fn_body(c, "phaseClotAxisymmetricFluid")
    i_ref, i_seed = b.find(REF_SEED), b.find(SEED)
    return (b.count(CONSTRUCT) == 1 and 0 <= i_ref < i_seed
            and b.find(SEED) < b.find(SEED_TYPES) < b.find(SEED_SET)
            and b.count(SEED_SET) == 1
            and '"thetaNetRefIntact", runTime.name(), mesh,\n            IOobject::READ_IF_PRESENT, '
                'IOobject::AUTO_WRITE' in c)


def p_predictor(c: str) -> bool:
    b = fn_body(c, "solveContentComponents")
    k = b.find(REF_SOLVE)
    return (k >= 0 and b.count(REF_SOLVE) == 1
            and b.find(REF_SOLVE) < b.find(CONTROLS) < b.find(PREDICTOR)
            and b.count(PREDICTOR) == 1
            and CONTROL_KEYS.search(c) is not None
            and "thetaNetRefIntactEqn.solve()" not in b
            and "fvSolution" not in b[b.find(CONTROLS):b.find(PREDICTOR)])


def p_fct_part(c: str) -> bool:
    b = fn_body(c, "solveContentComponents")
    i0, i1, i2 = b.find(FCT_AFTER), b.find(FCT_PART), b.find(FCT_BEFORE)
    if not (0 <= i0 < i1 < i2 and b.count(FCT_PART) == 1):
        return False
    seg = b[i1:i1 + len(FCT_PART)]
    # lambda and AR are read: no write to either, no limiter, no fctApply in ref_I's part
    return ("lambda=" not in seg and "lambda.primitiveFieldRef" not in seg and "lamI[" + "facei]=" not in seg
            and "aRI[facei]=" not in seg and "fctLimiter(" not in seg and "fctApply(" not in seg
            and seg.count("primitiveFieldRef()") == 1
            and b.count("contentFctLambda_=") == 1)


def p_checkpoint(c: str) -> bool:
    bt = fn_body(c, "beginPlateletOuterTransaction")
    bp = fn_body(c, "solvePhaseField")
    bc = fn_body(c, "solveContentComponents")
    i_ref = bp.find(REF_ROLLBACK)
    return (bt.count(CHECKPOINT) == 1
            and bt.find("thetaNetRefStepAccepted_=") < bt.find(CHECKPOINT)
            and 0 <= i_ref < bp.find(ROLLBACK) < bp.find('restoreDamagePartition("")')
            and bp.count(ROLLBACK) == 1
            and bc.find(NEG_REF) >= 0 and bc.find(NEG_REF + "" + NEG_REFI) >= 0)


def p_transfers(c: str) -> bool:
    br = fn_body(c, "ruptureOverstretchedNetwork")
    bg = fn_body(c, "solvePlateletContent")
    bd = fn_body(c, "releaseDetachedMaterial")
    return (br.count(RUP_ALIAS) == 1 and br.count(RUP_OLD) == 1
            and br.count(RUP_LAW + RUP_TEAR) == 1
            and br.find(A1) < br.find(RUP_LAW)
            and br.find(RUP_OLD) < br.find(A1)
            and br.count(RUP_TAIL) == 1 and br.find(RUP_LAW) < br.find(RUP_TAIL)
            and bg.count(GROWTH) == 1 and "hNew-hStar;thetaNetRefIntact" not in bg
            and bd.count(DELETION) == 1 and bd.count(DELETION_BC) == 1)


def p_log(c: str) -> bool:
    b = fn_body(c, "reportIntactReference")
    post = fn_body(c, "postSolve")
    return (b.count(LOG_SHARES) == 1 and b.find(LOG_SHARES) < b.find(LOG_PROXY)
            and "constscalarthetaI=P1R[celli]" not in b
            and b.count("proxyDev.push_back(") == 1
            and b.count(LOG_QUANT) == 1 and b.find(LOG_PROXY) < b.find(LOG_QUANT)
            and LOG_RAW.search(c) is not None
            and LOG_PRINT.search(c) is not None
            and post.count("reportIntactReference();") == 1
            and post.find("writeBalanceDiagnostic();") < post.find("reportIntactReference();"))


PREDICATES = (
    ("no reader: ref_I and its aliases are named only by their own sites, the header only declares it, "
     "no lookup by name (section 2 'No reader')", p_no_reader),
    ("seed and write: READ_IF_PRESENT + AUTO_WRITE, all-or-none with the partition, ref's patch types, "
     "ref_I = ref after ref's seed (section 2)", p_seed_and_write),
    ("predictor: ref's own label operator with the Sp term, ref's solver dictionary built in code, "
     "tolerance 0, minIter = maxIter = ref's executed iterations (section 2, RT36 change 2)", p_predictor),
    ("FCT part: r_d lambda AR on the donor, denominator guarded, lambda/AR read only, between ref's "
     "limiter and fctApply (section 2)", p_fct_part),
    ("checkpoint: beginPlateletOuterTransaction, the corrector roll-back after ref's, negAfter beside "
     "ref's (section 2, RT36 change 3)", p_checkpoint),
    ("transfers: rupture ratio where pI_old > 0, volumetric tear 0 on A1's event, growth ref's increment, "
     "deletion 0 (section 2 table)", p_transfers),
    ("log: PHASE_CLOT_REFI with design section 2's keys, theta_I = (P1/S) theta, called in postSolve "
     "after the balance (section 2, RT36 change 7)", p_log),
)

# Planted defects: (label, unique anchor in the .C, replacement, predicate index)
MUTATIONS = (
    ("a law reads it (a)", "          + (thetaC_ + thetaH_)\n          - thetaNetRef_\n        );",
     "          + (thetaC_ + thetaH_)\n          - thetaNetRefIntact_\n        );", 0),
    ("a law reads it via an alias (a)",
     "                refITransfer += (refIW[celli] - refIOld)*vol[celli];\n",
     "                refITransfer += (refIW[celli] - refIOld)*vol[celli];\n"
     "                bI[celli] *= (refIW[celli] > 0 ? 1.0 : 1.0);\n", 0),
    ("a law reads it in the growth's stress-free mixing (a)",
     "                const scalar r = r0[celli];\n",
     "                const scalar r = thetaNetRefIntact_[celli];\n", 0),
    ("a transfer omitted: rupture (b)",
     "                refIW[celli] = refIOld*(pI[celli]/pIOldD74);\n", "", 5),
    ("a transfer omitted: volumetric tear (b)",
     "                refITransfer -= refIW[celli]*vol[celli];\n                refIW[celli] = 0.0;\n",
     "                refITransfer -= refIW[celli]*vol[celli];\n", 5),
    ("a transfer omitted: growth (b)", "            thetaNetRefIntact_ += thetaNew - thetaStar;\n", "", 5),
    ("a transfer omitted: deletion (b)", "                thetaNetRefIntact_[celli] = 0.0;\n", "", 5),
    ("rupture without the pI_old guard (b)", "            if (pIOldD74 > 0)\n", "            if (true)\n", 5),
    ("growth on P1's increment (b)",
     "            thetaNetRefIntact_ += thetaNew - thetaStar;\n",
     "            thetaNetRefIntact_ += hNew - hStar;\n", 5),
    ("the write missing (c)",
     '            "thetaNetRefIntact", runTime.name(), mesh,\n'
     "            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE\n",
     '            "thetaNetRefIntact", runTime.name(), mesh,\n'
     "            IOobject::READ_IF_PRESENT, IOobject::NO_WRITE\n", 1),
    ("the seed omitted (c)",
     "            thetaNetRefIntact_ = thetaNetRef_;\n            thetaNetRefIntact_.correctBoundaryConditions();\n",
     "            thetaNetRefIntact_.correctBoundaryConditions();\n", 1),
    ("the all-or-none refusal removed (c)", "        if (hasRefI != hasPartition)\n",
     "        if (false)\n", 1),
    ("lambda written by the kernel (d)",
     "                const scalar fluxI = rD*laI;\n",
     "                const scalar fluxI = rD*laI;\n                lambda[facei] = rD*lamI[facei];\n", 3),
    ("lambda tightened by ref_I (d)",
     "            const scalarField& lamI = lambda.primitiveField();\n",
     "            lambda = min(lambda, lambda);\n"
     "            const scalarField& lamI = lambda.primitiveField();\n", 3),
    ("the donor ratio's product guarded, not its denominator (d)",
     "                    refLowI[donorI] != 0\n", "                    laI != 0\n", 3),
    ("the ratio's denominator not ref^low (d)",
     "            const scalarField& refLowI = thetaNetRef_.primitiveField();\n",
     "            const scalarField& refLowI = thetaNetRefIntact_.primitiveField();\n", 3),
    ("the Sp term dropped (e)",
     "              - fvm::Sp(fvc::surfaceIntegrate(carrier), thetaNetRefIntact_)\n", "", 2),
    ("a plain solve to tolerance (e)", "            thetaNetRefIntactEqn.solve(refIControls);\n",
     "            thetaNetRefIntactEqn.solve();\n", 2),
    ("maxIter not matched (e)", '            refIControls.set("maxIter", refIters);\n', "", 2),
    ("tolerance not 0 (e)", '            refIControls.set("tolerance", scalar(0));\n',
     '            refIControls.set("tolerance", scalar(1e-12));\n', 2),
    ("the restore omitted (f)",
     "                thetaNetRefIntact_ = thetaNetRefIntactStepAccepted_();\n", "", 4),
    ("the checkpoint omitted (f)",
     '                "thetaNetRefIntactStepAccepted", thetaNetRefIntact_\n',
     '                "thetaNetRefIntactStepAccepted", thetaNetRef_\n', 4),
    ("negAfter omitted (f)", "            if (thetaNetRefIntact_[celli] < 0)\n",
     "            if (false)\n", 4),
    ("log key dropped", '        << " proxyDevP99=" << proxyDevP99\n', "", 6),
    ("log key dropped: proxyDevMin", '        << " proxyDevMin=" << proxyDevMin\n', "", 6),
    ("log key dropped: proxyDevP01", '        << " proxyDevP01=" << proxyDevP01\n', "", 6),
    ("proxyDevMin computed as the max",
     "        proxyDevMin = *std::min_element(proxyDev.begin(), proxyDev.end());\n",
     "        proxyDevMin = *std::max_element(proxyDev.begin(), proxyDev.end());\n", 6),
    ("proxyDevMin printed from the max", '        << " proxyDevMin=" << proxyDevMin\n',
     '        << " proxyDevMin=" << proxyDevMax\n', 6),
    ("proxyDevP01 computed as the p99 (index)",
     "            std::min(nD - 1, std::size_t(0.01*scalar(nD - 1) + 0.5));\n",
     "            std::min(nD - 1, std::size_t(0.99*scalar(nD - 1) + 0.5));\n", 6),
    ("proxyDevP01 computed as the p99 (read at k99)", "        proxyDevP01 = proxyDev[k01];\n",
     "        proxyDevP01 = proxyDev[k99];\n", 6),
    ("theta_I = P1 (numerics 35)", "        const scalar thetaI = aIs*(tCR[celli] + tHR[celli]);\n",
     "        const scalar thetaI = P1R[celli];\n", 6),
    ("the log never called", "    reportIntactReference();\n", "", 6),
)


def run(c: str) -> list[bool]:
    return [bool(p(c)) for _, p in PREDICATES]


def main() -> int:
    c = SOLVER.read_text(encoding="utf-8")
    ok = True
    live = run(c)
    for (name, _), r in zip(PREDICATES, live):
        print(("PASS " if r else "FAIL ") + name)
        ok &= r
    ctl = run(c + "\n// control edit\n")
    if ctl != live:
        print("FAIL the no-op control edit changed a predicate")
        ok = False
    else:
        print("MISSED control: a no-op edit (appended comment) changes no predicate")
    covered = set()
    for label, old, new, idx in MUTATIONS:
        n = c.count(old)
        if n != 1:
            print(f"FAIL ({label}) mutation anchor occurs {n} times (must be 1): {old.strip()[:60]!r}")
            ok = False
            continue
        r = run(c.replace(old, new, 1))
        caught = not r[idx]
        print(("CAUGHT " if caught else "MISSED ") + f"({label}) -> {PREDICATES[idx][0][:60]}")
        ok &= caught
        covered.add(idx)
    if covered != set(range(len(PREDICATES))):
        print(f"FAIL predicates without a planted defect: {sorted(set(range(len(PREDICATES))) - covered)}")
        ok = False
    letters = {re.search(r"\(([a-f])\)$", m[0]).group(1) for m in MUTATIONS if re.search(r"\([a-f]\)$", m[0])}
    if letters != set("abcdef"):
        print(f"FAIL section 2 defects without a mutation: {sorted(set('abcdef') - letters)}")
        ok = False
    print("D74 GUARD PASSED" if ok else "D74 GUARD FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
