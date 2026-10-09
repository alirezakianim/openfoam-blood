#!/usr/bin/env python3
"""S23 source guard (design research/designs/2026-10-07-S23-torn-network-debris.md rev. 3, and
rev. 4.1 section 13.2: majority T, the w form, the R3 refusals, the gate-population and F1' logging).

Pins each design property of S23 in the solver source, and proves each pin has teeth:
every predicate is re-run on an in-memory copy of the source with ONE planted defect
(a unique anchor, replaced once) and must then FAIL, while a no-op control edit must
leave every predicate passing. Pure Python; reads the live solver files only.

Rev. 4.1 adds one BEHAVIOURAL predicate: scripts/s23_torn_kernel_test.cpp (report 19 Q3's 2-step
test, the w form at d* = 1, F1''s one-site defect forms, Pi's rings) compiled against the in-memory
damagePartitionKernel.H with `c++ -std=c++14 -O2 -ffp-contract=off` and run; a planted defect in the
kernel must make it fail. The tau_v floor is also evaluated, from the solver's own operator and
literal, at the boundary 0.0005 / 5e-4 (accepted) and 4.99e-4 (refused).

OI-59 (solver-engineer 03, docket 20261008-1507-run081155-b72-s23-validation) adds one predicate:
G23-6's outflux is the transport's own inlet+outlet flux, read from the thetaC/thetaH matrices after
their solves and BEFORE the FCT, and the balance books it rather than re-reading the post-FCT patch value.

usage: python3 -B scripts/check_s23_torn_network.py      (exit 0 = pass, 1 = fail)
Written by verification-engineer 14 (docket 20261007-0840-oi55-frame-analysis); rev. 4.1 predicates
by solver-engineer 24.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "user-applications" / "phaseClotAxisymmetricFluid"
FILES = {
    "C": APP / "phaseClotAxisymmetricFluid.C",
    "H": APP / "phaseClotAxisymmetricFluid.H",
    "K": APP / "damagePartitionKernel.H",
    "F": APP / "fibreReleaseKernel.H",
    "T": ROOT / "scripts" / "s23_torn_kernel_test.cpp",
}


def strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def ns(text: str) -> str:
    return re.sub(r"\s+", "", strip_comments(text))


def fn_body(src: str, sig: str) -> str:
    """The brace-matched body after the first occurrence of `sig` (comments stripped)."""
    s = strip_comments(src)
    i = s.index(sig)
    j = s.index("{", i)
    depth = 0
    for k in range(j, len(s)):
        depth += s[k] == "{"
        depth -= s[k] == "}"
        if depth == 0:
            return s[j:k + 1]
    raise ValueError(sig)


def body(t, sig):
    try:
        return ns(fn_body(t["C"], sig))
    except ValueError:
        return ""


RUP = "::ruptureOverstretchedNetwork()"
TORN = "::tornNetworkCells("
CORR = "::correctTornFaces\n"
APPLY = "::applyTornStressWeight\n"
FLUX = "::skeletonElasticForceFlux() const"
CELL = "::skeletonElasticForceCell() const"
FCT = "::fctLimiter\n"
REC = "::recordStressWeight"
SCC = "::solveContentComponents\n"
BAL = "::writeBalanceDiagnostic()"
PHF = "::solvePhaseField()"


def p_refusal(t):
    c = ns(t["C"])
    return ('if(!(std::isfinite(volumetricTearRatio_)&&volumetricTearRatio_>1)){FatalErrorInFunction' in c
            and '"volumetricTearRatio",21.95' in c and "volumetricTearRatio_;" in ns(t["H"]))


def p_event_product_isskel(t):
    b = body(t, RUP)
    return ("dsLaw=1.0-exp(-(l1-lf)/(2.0*dlf));}}volTear=isSkelC&&refI[celli]>=Rc*theta;"
            "if(volTear){dsLaw=1.0;}}constscalarsC=" in b
            and "constboolpresent=refI[celli]>residualContent;if(present){" in b
            and b.count("volTear=isSkelC") == 1 and b.count("volTear=") == 2)


def p_isskel_rule(t):
    b = body(t, RUP)
    tb = body(t, TORN)
    c = ns(t["C"])
    return ("constboolisSkelC=residualChi_>0?indI[celli]-0.5*residualChi_>=0"
            ":phiSkelI[celli]-0.5*residualPhiS_>=0;" in b
            and "constvolScalarFieldindR(phaseIndicator());" in b
            and "constboolisSkelC=residualChi_>0?ind[celli]-0.5*residualChi_>=0"
                ":phiSI[celli]-0.5*residualPhiS_>=0;" in tb
            and 'pos0(indB-dimensionedScalar("halfChi",dimless,0.5*residualChi_))()' in c
            and 'pos0(phiSB-dimensionedScalar("half",dimless,0.5*residualPhiS_))()' in c)


def p_irreversible(t):
    return "constscalardsNew=max(ds,dsLaw);" in body(t, RUP)


def p_r_logged_before_a1(t):
    b = body(t, RUP)
    i_ev = b.find('"PHASE_CLOT_S23_EVENTtime="')
    i_a1 = b.find("(*refW)[celli]=theta;")
    return (0 <= i_ev < i_a1
            and '<<"R="<<(theta>0?refI[celli]/theta:-1.0)' in b
            and "if(isSkelC&&theta>0){Rskel.push_back(refI[celli]/theta);}" in b)


def p_a1_same_event(t):
    b = body(t, RUP)
    return ("pI[celli]=sC*(1.0-dsNew);pP[celli]=sC*pendNew;pB[celli]=sC*dNew;dAfter=dNew;}"
            "if(pI[celli]==0){bI[celli]=symmTensor::I;++nReset;}"
            "if(volTear&&pI[celli]==0){if(!refW){refW=&thetaNetRef_.primitiveFieldRef();}"
            "(*refW)[celli]=theta;}elseif(volTear){++nA1Deferred;}" in b)


def p_ref_untouched_without_firing(t):
    b = body(t, RUP)
    return (b.count("thetaNetRef_.primitiveFieldRef()") == 1
            and b.count("(*refW)") == 1
            and "scalarField*refW=nullptr;" in b
            and "if(returnReduce(refW!=nullptr,orOp<bool>())){thetaNetRef_.correctBoundaryConditions();}" in b
            and b.count("thetaNetRef_.correctBoundaryConditions()") == 1)


def p_lambda_f(t):
    return ("fibreReleaseKernel::eventStretch(ev[0],lc);" in body(t, RUP)
            and "inlinedoubleeventStretch(constdoubleb1,constdoublelc){returnstd::min(std::sqrt(b1),lc);}"
            in ns(t["F"]))


def kbody(t, sig):
    try:
        return ns(fn_body(t["K"], sig))
    except ValueError:
        return ""


def p_torn_set(t):
    # rev. 4.1 section 13.2 item 1: majority membership through the kernel's tornMember,
    # the threshold from progressiveDamageCeiling(); no presence test, no d* == 1
    tb = body(t, TORN)
    return ("constscalarS=pI[celli]+(pP[celli]+pB[celli]);" in tb
            and "constscalards=S>0?(pP[celli]+pB[celli])/S:0.0;" in tb
            and "constscalarthrT=damagePartitionKernel::tornThreshold(progressiveDamageCeiling());" in tb
            and "torn[celli]=damagePartitionKernel::tornMember(isSkelC,ds,thrT);" in tb
            and "if(dStar){(*dStar)[celli]=ds;}" in tb
            and "returnisSkel&&dStar>=thr;" in kbody(t, "inline bool tornMember")
            and "==1" not in tb and "present" not in tb and "residualContent" not in tb)


def p_threshold_from_law(t):
    # d_p is the rupture law's own expression at l1 = lc (never a literal), and the
    # threshold (1 + d_p)/2: the kernel's text equals the law's, and the member passes
    # the law's three parameters
    law = re.search(r"dsLaw=(1\.0-exp\(-\(l1-lf\)/\(2\.0\*dlf\)\));", body(t, RUP))
    pc = kbody(t, "inline Scalar progressiveCeiling")
    m = re.fullmatch(r"\{constScalarl1=lc;return(.*);\}", pc)
    dp = body(t, "::progressiveDamageCeiling() const")
    return (law is not None and m is not None and m.group(1).replace("std::", "") == law.group(1)
            and kbody(t, "inline Scalar tornThreshold") == "{return0.5*(1.0+dp);}"
            and dp == "{returndamagePartitionKernel::progressiveCeiling(networkCriticalStretch_,"
                      "networkDamageOnsetStretch_,networkDamageStretchScale_);}")


def p_packing_scale(t):
    return "sTorn[celli]=(1.0-ds)+ds*ind[celli];" in body(t, TORN)


def p_face_list(t):
    cb = body(t, CORR)
    return ("if(!(torn[o]||torn[n])){continue;}" in cb and "if(!torn[o]){continue;}" in cb
            and cb.count("f[facei]=fNew;") == 1 and cb.count("fp[facei]=fNew;") == 1)


def p_face_formula(t):
    cb = body(t, CORR)
    return ("constscalarsf=min(sX[o],sX[n]);constscalarfNew=-sf*(gPn[facei]-dc[facei]*(PtI[n]-PtI[o]))"
            "*magS[facei]/rho;" in cb
            and "constscalarfNew=-sX[o]*(gp[facei]-dcp[facei]*(Ptp[facei]-PtI[o]))*msp[facei]/rho;" in cb
            and "constvolScalarFieldPn(networkPartialPressure());constsurfaceScalarFieldgPn(fvc::snGrad(Pn));" in cb
            and 'intactShare()*min(pnU,dimensionedScalar(pnU.dimensions(),0))' in cb
            and "constvolScalarFieldpnU(networkPartialPressureUndamaged());" in cb
            and "constsurfaceScalarField&dcF=mesh.nonOrthDeltaCoeffs();" in cb
            and "constscalarField&magS=mesh.magSf().primitiveField();" in cb)


def p_empty_t_returns_first(t):
    cb = body(t, CORR)
    i = cb.find("if(nT==0){return0;}")
    return 0 <= i < cb.find("constvolScalarFieldPn(")


def p_flux_after_s22(t):
    fb = body(t, FLUX)
    return ("tf.ref()-=fvc::snGrad(networkPartialPressure())*mesh.magSf()/rho_;}"
            "correctTornFaces(tf.ref(),nullptr);returntf;}" in fb)


def p_cell_parity_guarded(t):
    cb = body(t, CELL)
    return ("if(tornNetworkCells(tornC,sTornC)>0){" in cb
            and "if(correctTornFaces(fS22,&dF)>0){tc.ref()+=fvc::reconstruct(dF);}" in cb
            and cb.count("tc.ref()+=") == 1
            and 'fvc::interpolate(divSigmaE)&mesh.Sf()' in cb
            and "fS22-=fvc::snGrad(networkPartialPressure())*mesh.magSf()/rho_;" in cb
            and "dimensionedScalar(fS22.dimensions(),0)" in cb)


def p_builders_dstar(t):
    c = ns(t["C"])
    return ("{w=skeletonStressWeight();}applyTornStressWeight(w);recordStressWeight(w.primitiveField(),0);" in c
            and "{phiS=skeletonStressWeight();}applyTornStressWeight(phiS);"
                "recordStressWeight(phiS.primitiveField(),1);" in c
            and c.count("applyTornStressWeight(") == 3)


def p_apply_only_torn(t):
    # rev. 4.1 section 13.2 item 2: w = max(ind, floor)(1 - d*) on T (the kernel's form),
    # T's own patch faces carry the cell's value; nothing when T is empty
    ab = body(t, APPLY)
    return ("if(tornNetworkCells(torn,sTorn,&dsT,&indT)==0){return;}" in ab
            and "constscalarfloorT=skeletonSolidFloor_;" in ab
            and "if(torn[celli]){wI[celli]=damagePartitionKernel::tornStressWeight(indT[celli],floorT,"
                "dsT[celli]);}" in ab
            and "if(torn[fc[facei]]){wp[facei]=wI[fc[facei]];}" in ab
            and ab.count("wI[celli]=") == 1 and ab.count("wp[facei]=") == 1
            and kbody(t, "inline Scalar tornStressWeight") == "{returnstd::max(ind,floorW)*(Scalar(1)-dStar);}")


def p_others_keep_one_minus_d(t):
    c = ns(t["C"])
    sw = body(t, "::skeletonStressWeight() const")
    return ("torn" not in sw and sw != ""
            and "constbooltornRef=site<2&&tornNetworkCells(tornW,sTornW,&dsW)>0;" in body(t, REC)
            and "constscalardRef=tornRef&&tornW[celli]?dsW[celli]:dI[celli];" in body(t, REC)
            and c.count("tornNetworkCells(") == 7)


def _refused(t, tau):
    """Evaluate the solver's own tau_v floor test (operator and literal from the source)."""
    c = ns(t["C"])
    lit = re.search(r"constscalars23TauVFloor=([0-9.eE+-]+);", c)
    op = re.search(r"if\(!\(networkViscosityTau_(>=|>)s23TauVFloor\)\)", c)
    if not (lit and op):
        return None
    floor = float(lit.group(1))
    return not (tau >= floor if op.group(1) == ">=" else tau > floor)


def p_refusal_marker(t):
    # rev. 4.1 section 13.2 item 4: none without the marker, a marker other than tanDelta0,
    # both FatalError in the constructor; the marker echoed on PHASE_CLOT_S23_CONFIG
    c = ns(t["C"])
    return ('sensitivityLimit_(phaseClotProperties_.lookupOrDefault<word>("sensitivityLimit","none")),' in c
            and 'if(phaseClotProperties_.found("sensitivityLimit")&&sensitivityLimit_!="tanDelta0")'
                '{FatalErrorInFunction' in c
            and 'constbools23Marked=sensitivityLimit_=="tanDelta0";' in c
            and 'if(s23Marked&&networkViscosity_!="none"){FatalErrorInFunction' in c
            and 'if(networkViscosity_=="none"&&!s23Marked){FatalErrorInFunction' in c
            and '<<"sensitivityLimit="<<sensitivityLimit_' in c
            and '<<"tornSet="<<"majorityIsSkel"' in c
            and "wordsensitivityLimit_;" in ns(t["H"])
            and c.index('constbools23Marked=') < c.index('"PHASE_CLOT_S23_CONFIG"'))


def p_refusal_tau(t):
    # kelvinVoigt without an explicit networkViscosityTau, or tau_v < 5e-4 s, unless marked;
    # the boundary: 0.0005 and 5e-4 (S23v_G40) accepted, 4.99e-4 refused
    c = ns(t["C"])
    return ('if(networkViscosity_=="kelvinVoigt"&&!s23Marked){if(!phaseClotProperties_.found('
            '"networkViscosityTau")){FatalErrorInFunction' in c
            and _refused(t, float("0.0005")) is False and _refused(t, float("5e-4")) is False
            and _refused(t, float("4.99e-4")) is True and _refused(t, float("nan")) is True)


def p_oi53(t):
    fb = body(t, FCT)
    k = ns(t["K"])
    return (":Pp[celli]>Qp?Qp/Pp[celli]:1.0;" in fb and ":Pm[celli]>Qm?Qm/Pm[celli]:1.0;" in fb
            and "Rp[celli]=Qp<tiny?0.0" in fb and "Rm[celli]=Qm<tiny?0.0" in fb
            and "Rp[c]=Qp<tiny?Scalar(0):Pp[c]>Qp?Qp/Pp[c]:Scalar(1);" in k
            and "Rm[c]=Qm<tiny?Scalar(0):Pm[c]>Qm?Qm/Pm[c]:Scalar(1);" in k)


S23_KEYS = ("volEvents", "stretchEvents", "volEventsTotal", "tornCells", "tornFaces", "RMax", "tornMaxSlip",
            "tornMaxVsOverU", "tornMaxKresOverKd", "tornZeroKdCells", "tornMaxAI", "tornMaxAbsPt",
            "contentResidualIntOverV0", "outfluxIntOverV0", "deltaT")
# rev. 4.1 section 13.2 item 6: printed on the per-step PHASE_CLOT_S23 line itself
S23R4_KEYS = ("piCells", "piMaxVsOverMaxU", "piMaxVsCell", "piMaxVsSlip", "outPiMaxVsOverMaxU",
              "outPiMaxVsCell", "outPiMaxVsSlip", "minorityMaxVsOverMaxU", "minorityMaxVsCell",
              "minorityMaxVsSlip", "f1WSite0Dev", "f1WSite0Cells", "f1WSite1Dev", "f1WSite1Cells",
              "f1WZeroOccCells", "f1PtDev", "f1PtCells", "tornMaxKresOverKdCell")
CONFIG_KEYS = ("tornSet", "progressiveCeiling", "tornThreshold", "sensitivityLimit")


def s23_line(c):
    """The flattened per-step PHASE_CLOT_S23 statement (from its Info to its endl)."""
    f = ns(c)
    i = f.find('Info<<"PHASE_CLOT_S23time="')
    return f[i:f.index("<<endl;", i)] if i >= 0 else ""


def p_pi_logging(t):
    # the gate populations every step on the solver's own T and end-of-step partition:
    # Pi = isSkel within two point-neighbour rings of T (the kernel's pointRings), outside
    # Pi, and M = isSkel AND d_p < d* < (1 + d_p)/2; each with its argmax cell and slip
    c = ns(t["C"])
    line = s23_line(t["C"])
    k = kbody(t, "inline long pointRings")
    return (line != ""
            and "scalarFieldsTornL,dsL;constlabelnTornL=tornNetworkCells(tornL,sTornL,&dsL);" in c
            and "constscalardpL=progressiveDamageCeiling();"
                "constscalarthrL=damagePartitionKernel::tornThreshold(dpL);constintpiRings=2;" in c
            and "seedL[i]=tornL[i]?1:0;" in c
            and "damagePartitionKernel::pointRings(mesh.cellPoints(),mesh.pointCells(),seedL,piRings,ringL);" in c
            and "if(!(isSkelB[i]>0.5)){continue;}constscalarmagVs=mag(Vs_[i]);if(ringL[i]<=piRings)"
                "{++nPiL;if(magVs>piVsL){piVsL=magVs;piCellL=i;}}elseif(magVs>outPiVsL)"
                "{outPiVsL=magVs;outPiCellL=i;}if(dsL[i]>dpL&&dsL[i]<thrL&&magVs>mVsL){mVsL=magVs;mCellL=i;}" in c
            and '<<"piCells="<<nPiL' in line and '<<"piMaxVsCell="<<piCellL' in line
            and "(piCellL>=0?mag(Vs_[piCellL]-U[piCellL]):-1.0)" in line
            and "for(intk=1;k<=nRings;++k)" in k and "if(ring[m]>k){ring[m]=k;nextFront.push_back(m);}" in k
            and "constauto&pc=ptCells[cp[q]];" in k)


def p_f1_logging(t):
    # F1' from the weights AS CONSUMED at BOTH builders: recorded per site right after
    # applyTornStressWeight and the S16w record, with no write to the weight after it; Pt
    # from the field the face correction consumes, against aI min(Wp, 0) recomputed in the log
    fs = body(t, "::skeletonFaceStress() const")
    es = body(t, "::elasticStress() const")
    cb = body(t, CORR)
    c = ns(t["C"])
    seq0 = "applyTornStressWeight(w);recordStressWeight(w.primitiveField(),0);recordTornWeight(w.primitiveField(),0);"
    seq1 = ("applyTornStressWeight(phiS);recordStressWeight(phiS.primitiveField(),1);"
            "recordTornWeight(phiS.primitiveField(),1);")
    wr = r"(?<![\w.])%s(\s*[-+*/]?=(?!=)|\.primitiveFieldRef|\.ref\(|\.boundaryFieldRef)"
    ok_after = (seq0 in fs and seq1 in es
                and not re.search(wr % "w", fs[fs.find(seq0) + len(seq0):])
                and not re.search(wr % "phiS", es[es.find(seq1) + len(seq1):]))
    rb = body(t, "::recordTornWeight(")
    return (ok_after and c.count("recordTornWeight(") == 3
            and "constscalaroccF=max(indF[celli],floorF);if(occF>0){devF=max(devF,"
                "damagePartitionKernel::f1WeightDeviation(wUsed[celli],occF,aIW[celli]));}else{++nZeroF;}" in rb
            and "constvolScalarFieldaIF(intactShare());" in rb
            and kbody(t, "inline Scalar f1WeightDeviation") == "{returnstd::abs(w/occ-aI);}"
            and "constscalarField&PtI=Pt.primitiveField();s23PtUsed_=PtI;"
                "s23PtUsedIndex_=mesh.time().timeIndex();" in cb
            and "if(s23PtUsedIndex_==runTime.timeIndex()&&s23PtUsed_.size()==nC)" in c
            and "mag(s23PtUsed_[i]-aIS[i]*min(pnUS[i],scalar(0)))" in c
            and "s23F1WDev_[0]=0;s23F1WDev_[1]=0;s23F1WCells_[0]=-1;s23F1WCells_[1]=-1;s23F1WZeroOcc_=0;}}" in c)


_KT_CACHE = {}


def p_kernel_test(t):
    """BEHAVIOUR: the 2-step host test etc., compiled against the in-memory kernel text."""
    key = hashlib.sha256((t["K"] + "\0" + t["T"]).encode()).hexdigest()
    if key not in _KT_CACHE:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "damagePartitionKernel.H").write_text(t["K"])
            (d / "kt.cpp").write_text(t["T"])
            b = subprocess.run(["c++", "-std=c++14", "-O2", "-ffp-contract=off", "-I", str(d),
                                str(d / "kt.cpp"), "-o", str(d / "kt")], capture_output=True, text=True)
            if b.returncode != 0:
                _KT_CACHE[key] = (False, b.stdout + b.stderr)
            else:
                r = subprocess.run([str(d / "kt")], capture_output=True, text=True)
                _KT_CACHE[key] = (r.returncode == 0 and "S23 KERNEL TEST PASSED" in r.stdout, r.stdout)
    return _KT_CACHE[key][0]

EVENT_KEYS = ("x", "y", "branch", "R", "theta", "ref", "lambda1", "lambdaF")


def p_log_keys(t):
    c = t["C"]
    line = s23_line(c)
    cfg = ns(c)[ns(c).find('"PHASE_CLOT_S23_CONFIG"'):]
    return (all(f'" {k}="' in c for k in S23_KEYS + EVENT_KEYS)
            and all(f'<<"{k}="' in line for k in S23R4_KEYS)
            and all(f'<<"{k}="' in cfg[:cfg.find("<<endl;")] for k in CONFIG_KEYS)
            and '"PHASE_CLOT_S23 time="' in c and '"PHASE_CLOT_S23_EVENT time="' in c) and (
        "if(m>maxVsAny){maxVsAny=m;maxVsPos=cc[celli];maxVsIsClot=inClot?1:0;"
        "maxVsSlip=mag(vsIn[celli]-UIn[celli]);}" in ns(c)
        and '<<"maxVsSlip="<<maxVsSlip' in ns(c)
        # re-check 23 R3: G23-7's maximum carries its cell (the argmax, not a separate scan)
        and 'if(kresOverKd>tornKresOverKd){tornKresOverKd=kresOverKd;tornKresCell=i;}' in ns(c))


def p_log_denominators(t):
    c = ns(t["C"])
    return ("if(magU>0){tornVsOverU=max(tornVsOverU,magVs/magU);}" in c
            and "if(KdS[i]>0){constscalarkresOverKd=KresS[i]/KdS[i];if(kresOverKd>tornKresOverKd){tornKresOverKd=kresOverKd;tornKresCell=i;}}" in c   # the same guard on Kd; rev. 4.1 R3 adds the argmax cell
            and "(s23ContentV0_>0?s23ContentResInt_/s23ContentV0_:-1.0)" in c
            and "(piCellL>=0&&maxUS>0?piVsL/maxUS:-1.0)" in c
            and "(outPiCellL>=0&&maxUS>0?outPiVsL/maxUS:-1.0)" in c
            and "(mCellL>=0&&maxUS>0?mVsL/maxUS:-1.0)" in c)


def p_content_outflux_booked(t):
    """OI-59: contentOuterFlux (G23-6's outflux and its residual) is the transport's own boundary flux."""
    s, b, f = body(t, SCC), body(t, BAL), body(t, PHF)
    i0, i1 = b.find("constscalarcontentVolume="), b.find("constscalarcontentResidual=")
    seg = b[i0:i1] if 0 <= i0 < i1 else ""
    k = s.find("contentBoundaryFlux_=")
    return (s.count("contentBoundaryFlux_=") == 1
            # stored after both solves and before the FCT moves the outlet cells
            and 0 <= s.find("thetaHEqn.solve();") < k < s.find("fctApply(thetaC_,lambda,AC);")
            # each component's own matrix, in fvMatrix::flux()'s boundary form
            and "eqn.internalCoeffs()[patchi]*pf.patchInternalField()" in s
            and "faceFlux-=eqn.boundaryCoeffs()[patchi];" in s
            and all(f"solvedPatchFlux(thetaCEqn,thetaC_,{p})+solvedPatchFlux(thetaHEqn,thetaH_,{p});" in s
                    for p in ("contentInlet", "contentOutlet"))
            # OI-62 build (verification 08's holes D4, D5, D8, D9): the coupled branch reads the
            # neighbour value, the lambda returns the unscaled sum, both patches are booked when
            # they exist, and the store is the booked sum itself
            and ("if(pf.coupled()){faceFlux-=eqn.boundaryCoeffs()[patchi]*pf.patchNeighbourField();}"
                 "else{faceFlux-=eqn.boundaryCoeffs()[patchi];}returngSum(faceFlux);};") in s
            and s.count("returngSum(faceFlux);") == 1
            and "if(contentInlet>=0){bookedFlux+=solvedPatchFlux(thetaCEqn,thetaC_,contentInlet)" in s
            and "if(contentOutlet>=0){bookedFlux+=solvedPatchFlux(thetaCEqn,thetaC_,contentOutlet)" in s
            and "contentBoundaryFlux_=bookedFlux;" in s
            # the balance books it, and re-reads no patch value of theta
            and "constscalarcontentOuterFlux=contentBoundaryFlux_;" in seg
            and "boundaryField" not in seg and "interpolate" not in seg
            and seg.count("thetaC_") == 1 and seg.count("thetaH_") == 1   # contentVolume only
            # zeroed with the carrier, so a step without transport books 0
            and "phaseCarrier_.clear();" in f and "contentBoundaryFlux_=0;" in f
            # the S23 keys keep their meaning
            and "s23ContentResInt_+=mag(contentResidual)*deltaT;" in b
            and "s23OutfluxInt_+=contentOuterFlux*deltaT;" in b)


PREDICATES = (
    ("refusal R_c finite and > 1, default 21.95 (s4, s2.6)", p_refusal),
    ("volumetric branch: isSkel AND ref >= R_c theta, inside present, d*_law = 1 (s2.1)", p_event_product_isskel),
    ("isSkel is the block's rule at the event and in T (s2.1, s2.2)", p_isskel_rule),
    ("d* irreversible: max(ds, dsLaw) (s2.1)", p_irreversible),
    ("R logged before the A1 reset; R quotients guarded (s2.1, s2.5)", p_r_logged_before_a1),
    ("A1 ref <- theta in the statement after thetaIntact = 0 and be = I (s2.1)", p_a1_same_event),
    ("thetaNetRef_ written only by a volumetric firing (V23-0)", p_ref_untouched_without_firing),
    ("lambda_f = min(lambda1, lc), R not in it (s2.1)", p_lambda_f),
    ("T = isSkel AND d* >= (1 + d_p)/2, majority, without present_P (s13.2 item 1)", p_torn_set),
    ("s_X = (1 - d*) + d* ind (s2.2 rule 3)", p_packing_scale),
    ("face list: only faces with a T side are overwritten (s2.2, V23-0)", p_face_list),
    ("torn-face formula, |S_f| on both terms, s_f = min(s_P, s_N) (s2.2)", p_face_formula),
    ("empty T returns before any field is built (V23-0)", p_empty_t_returns_first),
    ("Flux: S22's fSigE first, then the overwrite (s2.2)", p_flux_after_s22),
    ("Cell parity guarded by 'any torn face' (s2.2)", p_cell_parity_guarded),
    ("d* weight at the two builders only (s2.2, rev. 3)", p_builders_dstar),
    ("w = max(ind, floor)(1 - d*) on T only, patch = cell value, nothing when T is empty (s13.2 item 2)",
     p_apply_only_torn),
    ("other readers keep (1 - d); S16w reference sites 0 and 1 only", p_others_keep_one_minus_d),
    ("OI-53 at fctLimiter and limiterR1 (s2.3)", p_oi53),
    ("log keys read by the comparator, maxVsSlip at the max-|Vs| cell (s2.5)", p_log_keys),
    ("log quotients guarded on their own denominators", p_log_denominators),
    # rev. 4.1 (indices 21-27)
    ("d_p from the rupture law's own expression at lc, threshold (1 + d_p)/2 (s13.2 item 1)",
     p_threshold_from_law),
    ("refusal: networkViscosity none unless sensitivityLimit tanDelta0; marker whitelist; CONFIG echo "
     "(s13.2 item 4)", p_refusal_marker),
    ("refusal: kelvinVoigt needs an explicit tau_v >= 5e-4 s unless marked; 5e-4 accepted, 4.99e-4 "
     "refused (s13.2 item 4)", p_refusal_tau),
    ("step log: Pi (two point rings of T), outside Pi and M, each max |Vs|/maxU, cell, slip "
     "(s13.2 item 6)", p_pi_logging),
    ("step log: F1' from the weights as consumed at both builders, Pt as the face correction used it "
     "(s13.2 item 6)", p_f1_logging),
    ("kernel test (behaviour): 2-step N -> P -> D c 0.1 rev. 3 T 0 / rev. 4 T 1, c 0.6 T 0; w at "
     "d* = 1 is +0.0; F1' one-site forms fail; Pi = Chebyshev 2 (s13.2 item 7)", p_kernel_test),
    # OI-59 (index 27)
    ("G23-6 outflux: the transport's own inlet+outlet flux, stored after the solves and before the FCT, "
     "booked by the balance (OI-59)", p_content_outflux_booked),
)

# ONE planted defect per predicate: (file, unique anchor, replacement, predicate index)
MUTATIONS = (
    ("C", "if (!(std::isfinite(volumetricTearRatio_) && volumetricTearRatio_ > 1))",
     "if (!(std::isfinite(volumetricTearRatio_) && volumetricTearRatio_ > 0))", 0),
    ("C", "volTear = isSkelC && refI[celli] >= Rc*theta;", "volTear = refI[celli] >= Rc*theta;", 1),
    ("C", "              ? indI[celli] - 0.5*residualChi_ >= 0\n              : phiSkelI[celli]",
     "              ? indI[celli] - residualChi_ >= 0\n              : phiSkelI[celli]", 2),
    ("C", "const scalar dsNew = max(ds, dsLaw);", "const scalar dsNew = dsLaw;", 3),
    ("C", '<< " R=" << (theta > 0 ? refI[celli]/theta : -1.0)', '<< " R=" << refI[celli]/theta', 4),
    ("C", "                    if (volTear && pI[celli] == 0)\n", "                    if (volTear)\n", 5),
    ("C", "        if (returnReduce(refW != nullptr, orOp<bool>()))\n",
     "        if (returnReduce(true, orOp<bool>()))\n", 6),
    ("F", "    return std::min(std::sqrt(b1), lc);", "    return std::sqrt(b1);", 7),
    # rev. 4.1: the majority threshold planted back to rev. 3's d* == 1
    ("C", "        torn[celli] = damagePartitionKernel::tornMember(isSkelC, ds, thrT);",
     "        torn[celli] = isSkelC && ds == 1;", 8),
    ("C", "        sTorn[celli] = (1.0 - ds) + ds*ind[celli];", "        sTorn[celli] = ind[celli];", 9),
    ("C", "        if (!(torn[o] || torn[n]))\n", "        if (!(torn[o] && torn[n]))\n", 10),
    ("C", "           -sf*(gPn[facei] - dc[facei]*(PtI[n] - PtI[o]))*magS[facei]/rho;",
     "           -sf*gPn[facei]*magS[facei]/rho + sf*dc[facei]*(PtI[n] - PtI[o])/rho;", 11),
    ("C", "    if (nT == 0)\n    {\n        return 0;\n    }\n\n    const volScalarField Pn(",
     "\n    const volScalarField Pn(", 12),
    ("C", "    correctTornFaces(tf.ref(), nullptr);\n\n    return tf;",
     "    return tf;\n    correctTornFaces(tf.ref(), nullptr);", 13),
    ("C", "            if (correctTornFaces(fS22, &dF) > 0)\n            {\n                tc.ref() += fvc::reconstruct(dF);\n            }",
     "            correctTornFaces(fS22, &dF);\n            tc.ref() += fvc::reconstruct(dF);", 14),
    ("C", "    applyTornStressWeight(phiS);\n", "\n", 15),
    # rev. 4.1: the w form planted back to rev. 3's w = 0 on T
    ("C", "            wI[celli] = damagePartitionKernel::tornStressWeight\n            (\n"
          "                indT[celli], floorT, dsT[celli]\n            );",
     "            wI[celli] = 0.0;", 16),
    ("C", "        site < 2 && tornNetworkCells(tornW, sTornW, &dsW) > 0;",
     "        tornNetworkCells(tornW, sTornW, &dsW) > 0;", 17),
    ("K", "          : Pp[c] > Qp ? Qp/Pp[c] : Scalar(1);", "          : Pp[c] > 0 ? std::min(Scalar(1), Qp/Pp[c]) : Scalar(1);", 18),
    ("C", "                maxVsSlip = mag(vsIn[celli] - UIn[celli]);\n", "", 19),
    # G23-7's cell: the argmax assignment dropped (the cell would print -1 or a stale index)
    ("C", "                        tornKresCell = i;\n", "", 19),
    ("C", "                if (KdS[i] > 0)\n", "                if (true)\n", 20),
    # ---- rev. 4.1 planted defects (solver-engineer 24) ----
    # the majority threshold: the kernel's membership at d* == 1 (behaviour: T_P 0 at c 0.1)
    ("K", "    return isSkel && dStar >= thr;", "    return isSkel && dStar == Scalar(1);", 26),
    ("K", "    return isSkel && dStar >= thr;", "    return isSkel && dStar > thr;", 8),
    # the threshold: a literal d_p in place of the law, and a literal threshold
    ("C", "    return damagePartitionKernel::progressiveCeiling\n    (\n        networkCriticalStretch_,",
     "    return 0.2936;\n    return damagePartitionKernel::progressiveCeiling\n    (\n        "
     "networkCriticalStretch_,", 21),
    ("K", "    return 0.5*(1.0 + dp);", "    return Scalar(0.6468);", 26),
    ("K", "    return 1.0 - std::exp(-(l1 - lf)/(2.0*dlf));", "    return 1.0 - std::exp(-(l1 - lf)/dlf);", 21),
    # the w form: the kernel's weight at 0 (behaviour: w_P and F1' fail), the patch at 0.0
    ("K", "    return std::max(ind, floorW)*(Scalar(1) - dStar);", "    return Scalar(0)*ind*floorW*dStar;", 26),
    ("C", "                wp[facei] = wI[fc[facei]];", "                wp[facei] = 0.0;", 16),
    # the refusals: none admitted without the marker; any marker value admitted; no echo
    ("C", '    if (networkViscosity_ == "none" && !s23Marked)',
     '    if (false && networkViscosity_ == "none" && !s23Marked)', 22),
    ("C", '     && sensitivityLimit_ != "tanDelta0"', '     && sensitivityLimit_ == "tanDelta0x"', 22),
    # the marker admitted with kelvinVoigt (it would waive the tau_v floor; the CE, report 24)
    ("C", '    if (s23Marked && networkViscosity_ != "none")',
     '    if (false && s23Marked && networkViscosity_ != "none")', 22),
    ("C", '        << " sensitivityLimit=" << sensitivityLimit_\n', "", 22),
    # the tau_v floor at the boundary: 4.99e-4 admitted; 5e-4 exactly refused; no found()
    ("C", "    const scalar s23TauVFloor = 5e-4;", "    const scalar s23TauVFloor = 4.99e-4;", 23),
    ("C", "        if (!(networkViscosityTau_ >= s23TauVFloor))",
     "        if (!(networkViscosityTau_ > s23TauVFloor))", 23),
    ("C", '        if (!phaseClotProperties_.found("networkViscosityTau"))', "        if (false)", 23),
    # the step-level Pi logging: one ring; Pi without isSkel; M without its lower bound
    ("C", "        const int piRings = 2;", "        const int piRings = 1;", 24),
    ("C", "            if (!(isSkelB[i] > 0.5))\n            {\n                continue;\n            }\n"
          "            const scalar magVs = mag(Vs_[i]);\n            if (ringL[i] <= piRings)",
     "            const scalar magVs = mag(Vs_[i]);\n            if (ringL[i] <= piRings)", 24),
    ("C", "            if (dsL[i] > dpL && dsL[i] < thrL && magVs > mVsL)",
     "            if (dsL[i] < thrL && magVs > mVsL)", 24),
    ("K", "    for (int k = 1; k <= nRings; ++k)", "    for (int k = 1; k < nRings; ++k)", 26),
    # F1' as ONE-site mutations: w = 0 at builder 0 (before the record, and after it), the
    # (1 - d) weight at builder 1 (applyTornStressWeight dropped there), Pt not captured
    ("C", "    applyTornStressWeight(w);\n    recordStressWeight(w.primitiveField(), 0);\n",
     "    applyTornStressWeight(w);\n    w *= 0;\n    recordStressWeight(w.primitiveField(), 0);\n", 25),
    ("C", "    recordTornWeight(w.primitiveField(), 0);\n",
     "    recordTornWeight(w.primitiveField(), 0);\n    w *= 0;\n", 25),
    ("C", "    applyTornStressWeight(phiS);\n    recordStressWeight(phiS.primitiveField(), 1);\n",
     "    recordStressWeight(phiS.primitiveField(), 1);\n", 25),
    ("C", "    s23PtUsed_ = PtI;\n", "", 25),
    ("K", "    return std::abs(w/occ - aI);", "    return Scalar(0)*std::abs(w/occ - aI);", 26),
    # ---- OI-59 planted defects (solver-engineer 03) ----
    # the balance re-reads the post-FCT patch value (the shipped defect, two spellings)
    ("C", "    const scalar contentOuterFlux = contentBoundaryFlux_;\n",
     "    scalar contentOuterFlux = 0;\n    {\n        const surfaceScalarField contentFlux\n        (\n"
     "            phaseCarrier_()*upwind<scalar>(mesh_, phaseCarrier_()).interpolate(thetaC_ + thetaH_)\n"
     "        );\n        contentOuterFlux += gSum(contentFlux.boundaryField()[inletPatch])\n"
     "          + gSum(contentFlux.boundaryField()[outletPatch]);\n    }\n", 27),
    ("C", "    const scalar contentOuterFlux = contentBoundaryFlux_;\n",
     "    const scalar contentOuterFlux = contentBoundaryFlux_\n"
     "      + gSum(phaseCarrier_().boundaryField()[outletPatch]*thetaC_.boundaryField()[outletPatch]);\n", 27),
    # booked after the FCT has moved the outlet cells
    ("C", "        fctApply(thetaH_, lambda, AH);\n",
     "        fctApply(thetaH_, lambda, AH);\n        contentBoundaryFlux_ =\n"
     "            gSum(carrier.boundaryField()[outletI]*(thetaC_ + thetaH_)().boundaryField()[outletI]);\n", 27),
    # one component dropped at the outlet
    ("C", "              + solvedPatchFlux(thetaHEqn, thetaH_, contentOutlet);\n", ";\n", 27),
    # the patch value in place of the matrix's internal coefficient form; the uncoupled boundaryCoeffs dropped
    ("C", "                eqn.internalCoeffs()[patchi]*pf.patchInternalField()\n",
     "                eqn.internalCoeffs()[patchi]*pf\n", 27),
    ("C", "                faceFlux -= eqn.boundaryCoeffs()[patchi];\n", "", 27),
    # not zeroed with the carrier: a step without transport would book the last step's outflux
    ("C", "    contentBoundaryFlux_ = 0;\n", "", 27),
    # ---- OI-62 build (solver-engineer 12): verification 08's D4, D5, D8, D9, plus the inlet guard ----
    # D4: the coupled branch reads the patch-internal value instead of the neighbour's
    ("C", "eqn.boundaryCoeffs()[patchi]*pf.patchNeighbourField();",
     "eqn.boundaryCoeffs()[patchi]*pf.patchInternalField();", 27),
    # D5: the stored value is not the booked sum
    ("C", "        contentBoundaryFlux_ = bookedFlux;\n", "        contentBoundaryFlux_ = 2*bookedFlux;\n", 27),
    # D8: the lambda returns a scaled sum
    ("C", "            return gSum(faceFlux);\n", "            return 0.5*gSum(faceFlux);\n", 27),
    # D9: the outlet guard inverted (the outlet is never booked), and the same at the inlet
    ("C", "        if (contentOutlet >= 0)\n", "        if (contentOutlet < 0)\n", 27),
    ("C", "        if (contentInlet >= 0)\n", "        if (contentInlet < 0)\n", 27),
)


def run(texts) -> list[bool]:
    return [bool(p(texts)) for _, p in PREDICATES]


def main() -> int:
    texts = {k: p.read_text() for k, p in FILES.items()}
    ok = True
    live = run(texts)
    for (name, _), r in zip(PREDICATES, live):
        print(("PASS " if r else "FAIL ") + name)
        ok &= r
    # the control: a no-op edit leaves every predicate passing
    ctl = dict(texts)
    ctl["C"] = ctl["C"] + "\n// control edit\n"
    if run(ctl) != live:
        print("FAIL the no-op control edit changed a predicate")
        ok = False
    covered = set()
    for f, old, new, idx in MUTATIONS:
        n = texts[f].count(old)
        if n != 1:
            print(f"FAIL mutation anchor for predicate {idx} occurs {n} times (must be 1): {old[:60]!r}")
            ok = False
            continue
        mut = dict(texts)
        mut[f] = texts[f].replace(old, new, 1)
        r = run(mut)
        caught = not r[idx]
        print(("CAUGHT " if caught else "MISSED ") + f"planted defect -> {PREDICATES[idx][0]}")
        ok &= caught
        covered.add(idx)
    if covered != set(range(len(PREDICATES))):
        print(f"FAIL predicates without a planted defect: {sorted(set(range(len(PREDICATES))) - covered)}")
        ok = False
    print("S23 GUARD PASSED" if ok else "S23 GUARD FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
