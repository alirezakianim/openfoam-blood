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

# (name, [(anchor, replacement), ...], the check tag that must FAIL)
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
]


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
                ok = mcode != 0 and must in mout
                print(f"{'OK  ' if ok else 'FAIL'} mutation '{name}' -> exit {mcode}, "
                      f"'{must}' {'seen' if must in mout else 'NOT seen'}")
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
    print("S19 GUARD", "PASSED" if status == 0 else "FAILED")
    return status


if __name__ == "__main__":
    sys.exit(main())
