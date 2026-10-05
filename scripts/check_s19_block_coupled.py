#!/usr/bin/env python3
"""S19 guard: the block-coupled (U, Vs, p) step and the R3 option.

research/designs/2026-10-03-S19-block-coupled-step.md, rev. 4, APPROVED by
Alireza 2026-10-04 (docket research/investigations/20261003-0205-np-block-
coupled-solve/), sections 2, 5 and 12.

1. The kernel test scripts/s19_block_kernel_test.cpp, compiled against the
   solver's OWN blockCoupledKernel.H and SuiteSparse (UMFPACK), passes.
2. Each kernel mutation FAILS its named check (design section 5: Rhie-Chow
   removed, drag x0.9 in both rows, the non-orthogonal correction lagged, the
   symbolic factorisation reused after a pattern change; section 12: the
   transpose dropped, a non-zero wall weight).
3. Source pins: the switch, its default and the refusals in the solver; the
   segregated path's dispatch untouched; the s19BlockCoupled() predicate
   (verification 10 D1); R3's applied face coefficient built from the
   elastic stress's own weight and gD with no 1/phi0 (fix round 12, physics
   11 row 5b); r3Power/r3Max over skeleton rows only.
4. Each solver-text mutation (SOURCE_MUTATIONS) FAILS its named pin.
5. S19a (research/designs/2026-10-04-S19a-umfpack-row-equilibration.md,
   rev. 2, APPROVED by Alireza 2026-10-04, section 3): K12 and K13 (with
   its R3 variant and G6's perturbations) pass; the old orientation (the
   CSR handed over as A^T, UMFPACK_At) FAILS K12's spread AND NOFF_DIAG
   checks; backSolve in the wrong orientation FAILS K12's linRes. K13 runs
   only when every earlier check passed, so no mutated kernel is factored at
   full size.
6. The assessor (assess_case.py s19_report, S19a section 4 pre-registration),
   on synthetic log text: a healthy 3-step blockCoupled log passes G0, G6, G7
   and G9; each ASSESSOR_MUTATIONS entry flips ONLY its own gate to False.
   G4 needs case files (0/p, the inletFlux/outletFlux function-object
   output) and is out of scope here: it must read None on log text alone.

SuiteSparse is found from $SUITESPARSE_ROOT (include/suitesparse, lib), then
/usr, /usr/local, /opt/homebrew. If none has umfpack.h this guard FAILS (exit
2): the design forbids skipping the check (section 2, Build).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER_DIR = ROOT / "user-applications/phaseClotAxisymmetricFluid"
KERNEL = SOLVER_DIR / "blockCoupledKernel.H"
SOLVER = SOLVER_DIR / "phaseClotAxisymmetricFluid.C"
HEADER = SOLVER_DIR / "phaseClotAxisymmetricFluid.H"
OPTIONS = SOLVER_DIR / "Make/options"
KERNEL_TEST = ROOT / "scripts/s19_block_kernel_test.cpp"

# (name, [(anchor, replacement), ...], the check tag that must FAIL; a tuple:
# every one of them must FAIL)
MUTATIONS = [
    ("Rhie-Chow term removed",
     [("addRhieChow(T, row, sg, c.rAf[f], m.Sx[f], m.Sy[f], G[o], wf, &G[n], 1 - wf, cRC);",
       "cRC += 0;")],
     "FAIL K1 known solution recovered"),
    ("drag x0.9 in both rows",
     [("const double KV = c.K[P]*m.V[P];", "const double KV = 0.9*c.K[P]*m.V[P];"),
      ("const double KsV = (c.K[P] + c.Kres[P])*m.V[P];",
       "const double KsV = 0.9*(c.K[P] + c.Kres[P])*m.V[P];")],
     "FAIL K1 known solution recovered"),
    ("non-orthogonal correction lagged from p^n (k = 1)",
     [("const int nBack = nNonOrth;", "const int nBack = nNonOrth > 0 ? 1 : 0;")],
     "FAIL K3 k=2 back-substitutions equal the reference chain"),
    ("symbolic factorisation reused after a pattern change",
     [("const bool samePattern = (symbolic_ != nullptr && A.n == n_ && A.p == Ap_ && A.i == Ai_);",
       "const bool samePattern = (symbolic_ != nullptr);")],
     "FAIL K4 symbolic redone on a pattern change"),
    ("R3 transpose cross terms dropped",
     [("const double transposeOn = 1.0;", "const double transposeOn = 0.0;")],
     "FAIL K5 rigid rotation gives zero R3 force"),
    ("R3 diagonal omitted from the face inverse's A_s (CE ruling F16)",
     [("        dAs[P] = 0.5*sum[P]/m.V[P];\n", "        dAs[P] = 0.0;\n")],
     "FAIL K10 R3 diagonal in A_s equals the dense reference"),
    ("R3 non-zero wall weight",
     [("        bW[f] = 0.0;\n", "        bW[f] = bW[f];\n")],
     "FAIL K8 every boundary R3 weight is 0"),
    ("R3 face coefficient divided by phi0 again (fix round 12)",
     [("    return tauV*elasticFaceModulus(sW, gDGf)/rho;\n",
       "    return tauV*elasticFaceModulus(sW, gDGf)/(rho*0.45);\n")],
     "FAIL K11 applied R3 face coefficient = tau_v x the elastic face coefficient"),
    ("S19a: the old orientation (the CSR handed over as A^T, UMFPACK_At)",
     [("        cscOfCsr(A, Cp_, Ci_, Cx_);\n", "        Cp_ = A.p; Ci_ = A.i; Cx_ = A.x;\n"),
      ("            UMFPACK_A, Cp_.data(), Ci_.data(), Cx_.data(),\n",
       "            UMFPACK_At, Cp_.data(), Ci_.data(), Cx_.data(),\n")],
     ("FAIL K12 row abs-sums of R.A on A's CSR rows within a factor 10",
      "FAIL K12 NOFF_DIAG == 0")),
    ("S19a: backSolve solves the wrong orientation (UMFPACK_At on A's factor)",
     [("            UMFPACK_A, Cp_.data(), Ci_.data(), Cx_.data(),\n",
       "            UMFPACK_At, Cp_.data(), Ci_.data(), Cx_.data(),\n")],
     "FAIL K12 linRes <= 1e-12"),
    # Verification 05 (S19a docket): on a kept symbolic the CSC VALUES must
    # be rebuilt from the new A on every factor() call; K4's value-only change
    # (K x 1.7, same pattern) is the case that sees stale values.
    ("S19a: CSC values rebuilt only on a pattern change (stale values on a kept symbolic)",
     [("        cscOfCsr(A, Cp_, Ci_, Cx_);\n",
       "        if (!samePattern) { cscOfCsr(A, Cp_, Ci_, Cx_); }\n")],
     "FAIL K4 solution correct on the kept symbolic"),
]

# Mutations of the SOLVER text: (name, [(anchor, replacement)], the pin that
# must then be reported missing).
SOURCE_MUTATIONS = [
    ("s19BlockCoupled() returns true (verification 10 D1)",
     [('    return momentumPressureCoupling_ == "blockCoupled";\n',
       "    return true;\n")],
     "the s19BlockCoupled() predicate"),
    ("R3 weight divided by phi0 again (physics 11 row 5b)",
     [("                sWR3[f], gDGR3[f], networkViscosityTau_, rho_.value()\n",
       "                sWR3[f], gDGR3[f], networkViscosityTau_/skeletonVolumeFraction_, rho_.value()\n")],
     "R3's applied face coefficient: tau_v/rho x the elastic face modulus"),
    ("gD dropped from R3's weight (CE ruling, fix round 12)",
     [("        const surfaceScalarField gDGR3(fvc::interpolate(gDR3*GField_));\n",
       "        const surfaceScalarField gDGR3(fvc::interpolate(GField_));\n")],
     "R3 carries the elastic stress's gD"),
    ("lumen identity rows counted in r3Power/r3Max",
     [("                if (!coef.isSkel[i])\n                {\n                    continue;\n                }\n                r3Power",
       "                r3Power")],
     "r3Power and r3Max exclude the lumen identity rows"),
    # S19a (verification 05 hand-off): batch 65 died with nothing about the
    # matrix in its log; the diagnostics that would have named it are pinned.
    ("S19a FatalError loses the row sums (CE, verification 05)",
     [('            << " rowSumMin=" << st.rowSumMin\n            << " rowSumMax=" << st.rowSumMax\n            << "; the symbolic',
       '            << "; the symbolic')],
     "S19a FatalError carries the matrix and the symbolic estimates"),
    ("S19a log loses numericPeakGB (CE, verification 05)",
     [('            << " numericPeakGB=" << st.numericPeakGB\n', "")],
     "S19a log carries G6's and G9's keys"),
]


ASSESS = ROOT / "cases/AM2020_planar_mural_detachment/scripts/assess_case.py"

# A healthy PHASE_CLOT_S19 step (the solver's field order, values of the
# K13 replay, docket 20261004-1608 report 03): every gate passes.
S19_HEALTHY = [
    ("time", None), ("step", None), ("mode", "blockCoupled"), ("tSolve", "0.05"),
    ("tNumeric", "0.61"), ("tSymbolic", "0.02"), ("symbolicCount", "1"),
    ("backSubstitutions", "2"), ("linRes", "2.7e-15"), ("linResScaled", "3.9e-16"),
    ("rcond", "1e-12"), ("n", "156800"), ("nnz", "1947768"), ("strategyUsed", "1"),
    ("orderingUsed", "1"), ("lunz", "30000000"), ("numericPeakGB", "0.2226"),
    ("noffdiag", "0"), ("rowSumMin", "1.3e-17"), ("rowSumMax", "2"),
    ("contResAbs", "1e-20"), ("kMatch", "0"), ("kresMatch", "0"),
    ("bodySlipRms", "1e-6"), ("qInlet", "1e-7"), ("qOutlet", "-1e-7"),
    ("lumenIdentityRows", "25088"),
]
S19_SPEC = {"s19": {"momentumPressureCoupling": "blockCoupled", "networkViscosity": "none"}}

# (name, step index (0-2), {key: value, or None to drop the field}; or
# step "drop" to remove that step's record, the gate that must flip alone)
ASSESSOR_MUTATIONS = [
    ("linResScaled 2e-10 on one step", 1, {"linResScaled": "2e-10"}, "g6"),
    ("linRes 2e-10 on one step", 2, {"linRes": "2e-10"}, "g6"),
    ("numericPeakGB 1.9 on one step", 0, {"numericPeakGB": "1.9"}, "g9"),
    ("tNumeric 300 on one step", 1, {"tNumeric": "300"}, "g9"),
    ("linResScaled field missing on one step", 2, {"linResScaled": None}, "g6"),
    ("kMatch 1e-30 on one step", 0, {"kMatch": "1e-30"}, "g7"),
    ("one step without a record", 1, "drop", "g0"),
]
S19_GATES = ("g0", "g6", "g7", "g9")


def s19_log(mutate: tuple[int, object] | None = None) -> str:
    """Three 'Time =' steps, each followed by its PHASE_CLOT_S19 record."""
    lines = ["PHASE_CLOT_S19_CONFIG momentumPressureCoupling=blockCoupled networkViscosity=none"]
    for k, t in enumerate(("0.001", "0.002", "0.003")):
        lines += [f"Time = {t}", ""]
        rec = dict(S19_HEALTHY, time=t, step=str(k + 1))
        if mutate is not None and mutate[0] == k:
            if mutate[1] == "drop":
                continue
            for key, v in mutate[1].items():
                if v is None:
                    del rec[key]
                else:
                    rec[key] = v
        lines.append("PHASE_CLOT_S19 " + " ".join(f"{a}={b}" for a, b in rec.items()))
    return "\n".join(lines) + "\nEnd\n"


def assessor_gates() -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location("assess_case_s19", ASSESS)
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    status = 0
    r = a.s19_report(s19_log(), S19_SPEC, None)
    healthy = {g: r[g] for g in S19_GATES}
    want = {"records": 3, "steps": 3, "linres_scaled_max": 3.9e-16,
            "numeric_peak_gb_max": 0.2226, "t_numeric_max": 0.61}
    wwant = {"noffdiag_max": 0.0, "row_sum_min_min": 1.3e-17, "row_sum_max_max": 2.0,
             "n_last": 156800.0, "nnz_last": 1947768.0}
    ok = (all(v is True for v in healthy.values()) and r["g4"] is None
          and all(r[k] == v for k, v in want.items())
          and all(r["watch"][k] == v for k, v in wwant.items()))
    print(f"{'OK  ' if ok else 'FAIL'} assessor healthy 3-step blockCoupled log -> "
          + ", ".join(f"{g} {v}" for g, v in healthy.items())
          + f", g4 {r['g4']} (out of scope: needs case files), "
          + ", ".join(f"{k} {r[k]}" for k in want) + ", watch "
          + ", ".join(f"{k} {r['watch'][k]}" for k in wwant))
    if not ok:
        status = 1
    for name, step, edit, gate in ASSESSOR_MUTATIONS:
        m = a.s19_report(s19_log((step, edit)), S19_SPEC, None)
        got = {g: m[g] for g in S19_GATES}
        ok = got[gate] is False and all(got[g] is True for g in S19_GATES if g != gate)
        print(f"{'OK  ' if ok else 'FAIL'} assessor mutation '{name}' -> "
              + ", ".join(f"{g} {v}" for g, v in got.items()) + f" (must flip only {gate})")
        if not ok:
            status = 1
    return status


def suitesparse() -> tuple[Path, Path] | None:
    roots = []
    if os.environ.get("SUITESPARSE_ROOT"):
        roots.append(Path(os.environ["SUITESPARSE_ROOT"]))
    roots += [Path("/usr"), Path("/usr/local"), Path("/opt/homebrew")]
    for r in roots:
        inc = r / "include/suitesparse"
        if (inc / "umfpack.h").is_file():
            for lib in (r / "lib", r / "lib/x86_64-linux-gnu", r / "lib64"):
                if any(lib.glob("libumfpack.*")):
                    return inc, lib
    return None


def compile_and_run(kernel_dir: Path, inc: Path, lib: Path) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "s19k"
        c = subprocess.run(
            ["c++", "-std=c++14", "-O2", "-ffp-contract=off",
             "-I", str(kernel_dir), "-I", str(inc), str(KERNEL_TEST),
             "-L", str(lib), "-lumfpack", f"-Wl,-rpath,{lib}", "-o", str(exe)],
            capture_output=True, text=True)
        if c.returncode != 0:
            return 99, c.stdout + c.stderr
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


def flat(s: str) -> str:
    return re.sub(r"\s+", "", s)


def source_pins(src: str | None = None) -> list[str]:
    """What must hold in the solver text (empty list: all hold)."""
    src = SOLVER.read_text() if src is None else src
    hdr = HEADER.read_text()
    f = flat(src)
    bad = []
    pins = [
        ("the switch is read with the shipped default",
         'lookupOrDefault<word>("momentumPressureCoupling","segregated")'),
        ("R3 key, default none", 'lookupOrDefault<word>("networkViscosity","none")'),
        ("R3 tau key, default 2.5e-4", 'lookupOrDefault<scalar>("networkViscosityTau",2.5e-4)'),
        ("pressureCorrector dispatches to the block only under blockCoupled",
         "if(s19BlockCoupled()){solveBlockCoupledStep();tUEqn.clear();return;}incompressibleFluid::pressureCorrector();"),
        ("the segregated corrector is still the shipped dispatch",
         "if(tVsEqn.valid()){correctPressureTwoVelocity();return;}"),
        ("the predictors are skipped only under blockCoupled",
         "if(s19BlockCoupled()){return;}if(pimple.momentumPredictor())"),
        ("the shipped U predictor is unchanged", "solve(UEqn==-phiFluid_*fvc::grad(p_));"),
        ("the config line", 'Info<<"PHASE_CLOT_S19_CONFIG"'),
        ("the per-step record", 'Info<<"PHASE_CLOT_S19time="'),
        ("K read from the solver's drag field", "coef.K[i]=KdIn[i];"),
        ("Kres read from skeletonResidualDrag()", 'constvolScalarFieldKresB("Kres",skeletonResidualDrag());'),
        ("lagged non-orthogonal correction refused", "blockCoupledlagsthenon-orthogonal"),
        ("run-time bond refusal", "blockCoupledisforfreeclots"),
        ("UMFPACK status refusal", "UMFPACKstatus"),
        ("R3 refused under segregated", "momentumPressureCouplingblockCoupled:R3isunstable"),
        ("R3 boundary weights zeroed", "blockCoupled::r3ZeroBoundaryWeights(coef.bNu3w);"),
        ("F16: R3's diagonal in A_s of the face inverse, R3 only",
         'constvolScalarFieldDblkB("DblkB",VsEqn.A()+KdB+KresB+r3AsB);'),
        ("the shipped lumen overwrite is untouched", "Vs_=isSkel*Vs_+(1.0-isSkel)*U;"),
        ("the shipped isSkel expression is reused", "isSkelB"),
        ("the s19BlockCoupled() predicate",
         'boolFoam::solvers::phaseClotAxisymmetricFluid::s19BlockCoupled()const'
         '{returnmomentumPressureCoupling_=="blockCoupled";}'),
        ("R3's weight is the elastic stress's own weight",
         'constvolScalarFieldwR3("wR3",skeletonStressWeight());'
         'constvolScalarFieldgDR3(sqr(1.0-Dg_)+etaRes_);'
         'constsurfaceScalarFieldsWR3(seriesFaceWeights(wR3));'),
        ("R3 carries the elastic stress's gD",
         "constsurfaceScalarFieldgDGR3(fvc::interpolate(gDR3*GField_));"),
        ("R3's applied face coefficient: tau_v/rho x the elastic face modulus",
         "coef.nu3w[f]=blockCoupled::r3FaceCoefficient"
         "(sWR3[f],gDGR3[f],networkViscosityTau_,rho_.value());"),
        ("the elastic site's gD and modulus are the ones R3 copies",
         'constvolScalarFieldgD(sqr(1.0-Dg_)+etaRes_);'
         'volSymmTensorFieldS("sigmaDevIntrinsic",gD*GField_*dev(be_));'),
        ("the elastic site's occupancy weight is skeletonStressWeight()",
         'if(skeletonFloorBasis_=="occupancy"){//S16B:max(ind,floor)*(1-d)--'
         'thefloorguardsoccupancyonly.w=skeletonStressWeight();}'),
        ("the elastic face stress is series(w) x interp(gD G M)",
         "seriesFaceWeights(w)*fvc::interpolate(S)"),
        ("r3Power and r3Max exclude the lumen identity rows",
         "if(!coef.isSkel[i]){continue;}r3Power+=vx[i]*fx[i]+vy[i]*fy[i];"),
        ("S19a FatalError carries the matrix and the symbolic estimates",
         '<<"Thematrix(S19asection2.4):n="<<st.n<<"nnz="<<st.nnz'
         '<<"rowSumMin="<<st.rowSumMin<<"rowSumMax="<<st.rowSumMax'
         '<<";thesymbolicestimates:lunzEstimate="<<st.lunzEstimate'
         '<<"numericSizeEstimateGB="<<st.numericSizeEstimateGB'
         '<<"peakMemoryEstimateGB="<<st.peakMemoryEstimateGB'),
        ("S19a log carries G6's and G9's keys",
         '<<"tNumeric="<<st.tNum'),
        ("S19a log carries G6's and G9's keys",
         '<<"linResScaled="<<linResScaled'),
        ("S19a log carries G6's and G9's keys",
         '<<"numericPeakGB="<<st.numericPeakGB<<"noffdiag="<<st.noffdiag'),
    ]
    for name, pin in pins:
        if flat(pin) not in f:
            bad.append(f"solver pin missing: {name}")
    if '#include "blockCoupledKernel.H"' not in hdr:
        bad.append("header does not include blockCoupledKernel.H")
    opts = OPTIONS.read_text()
    if "-I/usr/include/suitesparse" not in opts or "-lumfpack" not in opts:
        bad.append("Make/options lacks -I/usr/include/suitesparse or -lumfpack")
    return bad


def main() -> int:
    ss = suitesparse()
    if ss is None:
        print("check_s19_block_coupled: no SuiteSparse (umfpack.h) found; set "
              "SUITESPARSE_ROOT. The check is not skipped (design section 2).")
        return 2
    inc, lib = ss
    status = 0
    code, out = compile_and_run(SOLVER_DIR, inc, lib)
    print(out.rstrip())
    if code != 0:
        print("FAIL: the kernel test does not pass on the shipped kernel")
        status = 1
    for name, edits, must in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            text = KERNEL.read_text()
            for a, b in edits:
                if text.count(a) != 1:
                    print(f"FAIL mutation '{name}': anchor count {text.count(a)} != 1: {a[:60]}")
                    status = 1
                    break
                text = text.replace(a, b)
            else:
                (d / "blockCoupledKernel.H").write_text(text)
                mcode, mout = compile_and_run(d, inc, lib)
                musts = must if isinstance(must, tuple) else (must,)
                ok = mcode != 0 and all(m in mout for m in musts)
                print(f"{'OK  ' if ok else 'FAIL'} mutation '{name}' -> exit {mcode}, " + ", ".join(
                      f"'{m}' {'seen' if m in mout else 'NOT seen'}" for m in musts))
                if not ok:
                    status = 1
    for b in source_pins():
        print("FAIL", b)
        status = 1
    src = SOLVER.read_text()
    for name, edits, must in SOURCE_MUTATIONS:
        text = src
        for a, b in edits:
            if text.count(a) != 1:
                print(f"FAIL source mutation '{name}': anchor count {text.count(a)} != 1: {a[:60]}")
                status = 1
                break
            text = text.replace(a, b)
        else:
            seen = f"solver pin missing: {must}" in source_pins(text)
            print(f"{'OK  ' if seen else 'FAIL'} source mutation '{name}' -> "
                  f"pin '{must}' {'FAILS' if seen else 'still holds'}")
            if not seen:
                status = 1
    if assessor_gates() != 0:
        status = 1
    print("S19 GUARD", "PASSED" if status == 0 else "FAILED")
    return status


if __name__ == "__main__":
    sys.exit(main())
