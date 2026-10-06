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
7. S19b (research/designs/2026-10-04-S19b-lumen-elimination.md rev. 2,
   APPROVED by Alireza 2026-10-04, section 3): the kernel test's K14
   (identityRows is S19a byte for byte) and K15 (eliminate = identityRows to
   1e-12 on a system where every remapped class is asserted nonzero; the
   eliminated matrix factors in both orientations) pass; each of the seven
   K15 mutations FAILS K15; K12 and K13 are pinned to identityRows (F6) and
   G6's perturbations are rewritten for eliminate (K13-G6e). Source pins on
   the switch (default eliminate), its refusals and the new log keys, each
   with a mutation. The assessor: the healthy log is the DEFAULT (eliminate;
   D1), a healthy identityRows log passes C1, and each C1/D1 mutation flips
   only its gate (nCells from a temporary case's polyMesh/owner note). The
   E1 scorer's self-test (scripts/s19b_e1_score.py --self-test) passes.

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
    # S19b section 3: the seven K15 mutations. Each acts under eliminate only,
    # so K1-K14 (identityRows) still pass and K15 runs; each must FAIL K15.
    ("S19b: drag lumen-Vs references dropped (the lumen U row keeps dU + KV)",
     [("            T.add(rU, idx(P, 2 + k), -KV);\n",
       "            if (c.isSkel[P] || !lay.eliminate) { T.add(rU, idx(P, 2 + k), -KV); }\n")],
     "FAIL K15 eliminate = identityRows"),
    ("S19b: upVs/loVs interface references dropped",
     [("                T.add(idx(o, 2 + k), idx(n, 2 + k), c.upVs[f]);\n",
       "                if (c.isSkel[n] || !lay.eliminate) { T.add(idx(o, 2 + k), idx(n, 2 + k), c.upVs[f]); }\n"),
      ("                T.add(idx(n, 2 + k), idx(o, 2 + k), c.loVs[f]);\n",
       "                if (c.isSkel[o] || !lay.eliminate) { T.add(idx(n, 2 + k), idx(o, 2 + k), c.loVs[f]); }\n")],
     "FAIL K15 eliminate = identityRows"),
    ("S19b: R3 interface references dropped",
     [("                const int col = idx(F.cell[t], 2 + F.comp[t]);\n",
       "                const int col = idx(F.cell[t], 2 + F.comp[t]);\n"
       "                if (lay.eliminate && !c.isSkel[F.cell[t]]) { continue; }\n")],
     "FAIL K15-R3 eliminate = identityRows"),
    ("S19b: p-row lumen-Vs references dropped (red team 02 F4)",
     [("                T.add(row, idx(o, 2 + k), sg*c.phiSf[f]*Sk*wf);\n",
       "                if (c.isSkel[o] || !lay.eliminate) { T.add(row, idx(o, 2 + k), sg*c.phiSf[f]*Sk*wf); }\n"),
      ("                T.add(row, idx(n, 2 + k), sg*c.phiSf[f]*Sk*(1 - wf));\n",
       "                if (c.isSkel[n] || !lay.eliminate) { T.add(row, idx(n, 2 + k), sg*c.phiSf[f]*Sk*(1 - wf)); }\n"),
      ("            T.add(row, idx(P, 2 + k), c.bPhiS[bf]*Sk);\n",
       "            if (c.isSkel[P] || !lay.eliminate) { T.add(row, idx(P, 2 + k), c.bPhiS[bf]*Sk); }\n")],
     "FAIL K15 eliminate = identityRows"),
    ("S19b: a lumen-Vs reference mapped to the wrong component",
     [("                lay.col[i] = lay.col[idx(P, k - 2)];\n",
       "                lay.col[i] = lay.col[idx(P, 3 - k)];\n")],
     "FAIL K15 eliminate = identityRows"),
    # The full (cell, var) number used where the Layout's is meant; wrapped
    # into range so the defect is defined behaviour.
    ("S19b: the non-orthogonal rhs not remapped through the Layout",
     [("                rhs[lay.row[idx(m.own[f], 4)]] -= e;\n"
       "                rhs[lay.row[idx(m.nei[f], 4)]] += e;\n",
       "                rhs[idx(m.own[f], 4) % lay.n] -= e;\n"
       "                rhs[idx(m.nei[f], 4) % lay.n] += e;\n")],
     "FAIL K15 eliminate = identityRows"),
    ("S19b: lumen Vs expanded from the old Vs instead of U",
     [("        s.x[i] = xc[lay.col[i]];\n",
       "        s.x[i] = lay.row[i] < 0 ? x0[i] : xc[lay.col[i]];\n")],
     "FAIL K15 eliminate = identityRows, max|d|/max|x| <= 1e-12: lumen Vs (= U)"),
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
    # S19b section 3: the switch, its refusals and the new log keys.
    # S21: the default, value-refusal, outside-key and config-echo mutations
    # retired with their pins (hunks b15, c38, c39, c43 above).
    ("S19b kernel switch hard-wired off",
     [('    coef.eliminateLumen = true;\n', "    coef.eliminateLumen = false;\n")],
     "S19b: the kernel eliminates the lumen (blockCoupledLumen built in at eliminate)"),
    ("the elastic site's weight left at the occupancy-blind form",
     [('        w = skeletonStressWeight();\n', "")],
     "the elastic site's occupancy weight is skeletonStressWeight()"),
    ("S19b record loses lumenEliminated",
     [('            << " lumenEliminated=" << st.nEliminated\n', "")],
     "S19b: PHASE_CLOT_S19 logs lumenEliminated and the 17-digit maxima"),
    ("S19b maxima at the stream's default precision",
     [("        maxesB.precision(17);\n", "")],
     "S19b: uMax, vsMax, pMax at 17 significant digits over every cell"),
]


ASSESS = ROOT / "cases/AM2020_planar_mural_detachment/scripts/assess_case.py"

# A healthy PHASE_CLOT_S19 step (the solver's field order, values of the
# K13 replay, docket 20261004-1608 report 03): every gate passes. S19b: the
# DEFAULT is eliminate, so the healthy record carries lumenEliminated (the
# replay's 25088 lumen cells), no identity rows, n = 5 x 31360 - 2 x 25088,
# and the 17-digit maxima (frozen site, S19b section 3).
S19_HEALTHY = [
    ("time", None), ("step", None), ("mode", "blockCoupled"), ("tSolve", "0.05"),
    ("tNumeric", "0.61"), ("tSymbolic", "0.02"), ("symbolicCount", "1"),
    ("backSubstitutions", "2"), ("linRes", "2.7e-15"), ("linResScaled", "3.9e-16"),
    ("rcond", "1e-12"), ("n", "106624"), ("nnz", "1947768"), ("strategyUsed", "1"),
    ("orderingUsed", "1"), ("lunz", "30000000"), ("numericPeakGB", "0.2226"),
    ("noffdiag", "0"), ("rowSumMin", "1.3e-17"), ("rowSumMax", "2"),
    ("contResAbs", "1e-20"), ("kMatch", "0"), ("kresMatch", "0"),
    ("bodySlipRms", "1e-6"), ("qInlet", "1e-7"), ("qOutlet", "-1e-7"),
    ("lumenIdentityRows", "0"), ("lumenEliminated", "25088"),
    ("uMax", "0.0045678901234567891"), ("vsMax", "0.0012345678901234567"),
    ("pMax", "0.073706600000000001"),
]
S19_SPEC = {"s19": {"momentumPressureCoupling": "blockCoupled", "networkViscosity": "none"}}
# S19b C1: the identityRows control (S19a's system: n = 5 x nCells).
S19_HEALTHY_ID = [(k, {"n": "156800", "lumenIdentityRows": "25088", "lumenEliminated": "0"}.get(k, v))
                  for k, v in S19_HEALTHY]
S19_SPEC_ID = {"s19": {"momentumPressureCoupling": "blockCoupled", "networkViscosity": "none",
                       "blockCoupledLumen": "identityRows"}}
S19_NCELLS = 31360

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
    # S19b D1 (the default is engaged).
    ("S19b: identity rows under the default on one step", 1,
     {"lumenIdentityRows": "25088", "lumenEliminated": "0", "n": "156800"}, "d1"),
    ("S19b: n = 5 x nCells under the default on one step", 2, {"n": "156800"}, "d1"),
    ("S19b: lumenEliminated missing on one step", 0, {"lumenEliminated": None}, "d1"),
    ("S19b: the default's config echoes identityRows", "config", "identityRows", "d1"),
]
# S19b C1 (the control is the control), on the identityRows log.
ASSESSOR_MUTATIONS_ID = [
    ("S19b: no identity rows on one step of the control", 1, {"lumenIdentityRows": "0"}, "c1"),
    ("S19b: n < 5 x nCells on one step of the control", 0, {"n": "106624"}, "c1"),
    ("S19b: lumenIdentityRows missing on one step of the control", 2, {"lumenIdentityRows": None}, "c1"),
    ("S19b: the control's config echoes eliminate", "config", "eliminate", "c1"),
]
S19_GATES = ("g0", "g6", "g7", "g9")


def s19_log(mutate: tuple[object, object] | None = None, healthy=None,
            lumen: str = "eliminate") -> str:
    """Three 'Time =' steps, each followed by its PHASE_CLOT_S19 record;
    mutate (step, edits) or ("config", lumen value)."""
    healthy = S19_HEALTHY if healthy is None else healthy
    if mutate is not None and mutate[0] == "config":
        lumen = mutate[1]
    lines = ["PHASE_CLOT_S19_CONFIG momentumPressureCoupling=blockCoupled networkViscosity=none "
             f"blockCoupledLumen={lumen}"]
    for k, t in enumerate(("0.001", "0.002", "0.003")):
        lines += [f"Time = {t}", ""]
        rec = dict(healthy, time=t, step=str(k + 1))
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
    with tempfile.TemporaryDirectory() as tmp:
        # S19b: C1/D1 read nCells from the case mesh, never from the record.
        case = Path(tmp)
        (case / "constant/polyMesh").mkdir(parents=True)
        (case / "constant/polyMesh/owner").write_text(
            'FoamFile\n{\n    note        "nPoints: 63758 nCells: %d nFaces: 125958 '
            'nInternalFaces: 62202";\n}\n' % S19_NCELLS)
        gates = S19_GATES + ("d1",)
        r = a.s19_report(s19_log(), S19_SPEC, case)
        healthy = {g: r[g] for g in gates}
        want = {"records": 3, "steps": 3, "linres_scaled_max": 3.9e-16,
                "numeric_peak_gb_max": 0.2226, "t_numeric_max": 0.61, "n_cells": S19_NCELLS,
                "c1": None, "lumen_expected": "eliminate"}
        wwant = {"noffdiag_max": 0.0, "row_sum_min_min": 1.3e-17, "row_sum_max_max": 2.0,
                 "n_last": 106624.0, "nnz_last": 1947768.0, "lumen_eliminated_last": 25088.0,
                 "lumen_identity_rows_last": 0.0}
        ok = (all(v is True for v in healthy.values()) and r["g4"] is None
              and all(r[k] == v for k, v in want.items())
              and all(r["watch"][k] == v for k, v in wwant.items()))
        print(f"{'OK  ' if ok else 'FAIL'} assessor healthy 3-step blockCoupled log (default eliminate) -> "
              + ", ".join(f"{g} {v}" for g, v in healthy.items())
              + f", g4 {r['g4']} (out of scope: needs case files), "
              + ", ".join(f"{k} {r[k]}" for k in want) + ", watch "
              + ", ".join(f"{k} {r['watch'][k]}" for k in wwant))
        if not ok:
            status = 1
        # Without the case mesh C1/D1 cannot be scored: None, never True.
        r0 = a.s19_report(s19_log(), S19_SPEC, None)
        ok = r0["d1"] is None and r0["c1"] is None
        print(f"{'OK  ' if ok else 'FAIL'} assessor without the case mesh -> d1 {r0['d1']}, c1 {r0['c1']} (must be None)")
        if not ok:
            status = 1
        for name, step, edit, gate in ASSESSOR_MUTATIONS:
            m = a.s19_report(s19_log((step, edit)), S19_SPEC, case)
            got = {g: m[g] for g in gates}
            ok = got[gate] is False and all(got[g] is True for g in gates if g != gate)
            print(f"{'OK  ' if ok else 'FAIL'} assessor mutation '{name}' -> "
                  + ", ".join(f"{g} {v}" for g, v in got.items()) + f" (must flip only {gate})")
            if not ok:
                status = 1
        gates_id = S19_GATES + ("c1",)
        r = a.s19_report(s19_log(healthy=S19_HEALTHY_ID, lumen="identityRows"), S19_SPEC_ID, case)
        got = {g: r[g] for g in gates_id}
        ok = all(v is True for v in got.values()) and r["d1"] is None and r["lumen_expected"] == "identityRows"
        print(f"{'OK  ' if ok else 'FAIL'} assessor healthy identityRows control log -> "
              + ", ".join(f"{g} {v}" for g, v in got.items()) + f", d1 {r['d1']} (must be None)")
        if not ok:
            status = 1
        for name, step, edit, gate in ASSESSOR_MUTATIONS_ID:
            m = a.s19_report(s19_log((step, edit), healthy=S19_HEALTHY_ID, lumen="identityRows"),
                             S19_SPEC_ID, case)
            got = {g: m[g] for g in gates_id}
            ok = got[gate] is False and all(got[g] is True for g in gates_id if g != gate)
            print(f"{'OK  ' if ok else 'FAIL'} assessor mutation '{name}' -> "
                  + ", ".join(f"{g} {v}" for g, v in got.items()) + f" (must flip only {gate})")
            if not ok:
                status = 1
        # BATCH 68 (CE ruling on docket 20261005-0010 report 03): under R3 the
        # config echo's networkViscosityTau must match the child's tau_v_s
        # (relative 1e-9). A tau off by 1e-6 relative flips ONLY the config
        # check; every gate (and G8r, D1) stays True.
        tau = 5.0e-4
        spec_r3 = {"s19": {"momentumPressureCoupling": "blockCoupled",
                           "networkViscosity": "kelvinVoigt", "networkViscosityTau": tau}}
        def r3_log(echo_tau: float) -> str:
            text = s19_log().replace(
                "networkViscosity=none ",
                f"networkViscosity=kelvinVoigt networkViscosityTau={echo_tau!r} ")
            return re.sub(r"^(PHASE_CLOT_S19 .*)$", r"\1 r3Power=-3.2e-16 r3Max=1.1e-3",
                          text, flags=re.M)
        gates_r3 = S19_GATES + ("d1", "g8r")
        for name, echo, want_cfg in (("R3 healthy, tau echoed exactly", tau, True),
                                     ("R3 tau off by 1e-6 relative", tau * (1 + 1e-6), False)):
            m = a.s19_report(r3_log(echo), spec_r3, case)
            got = {g: m[g] for g in gates_r3}
            ok = (m["config_matches_child"] is want_cfg and m["config_tau_matches"] is want_cfg
                  and all(v is True for v in got.values()))
            print(f"{'OK  ' if ok else 'FAIL'} assessor '{name}' -> config_matches_child "
                  f"{m['config_matches_child']}, config_tau_matches {m['config_tau_matches']}, "
                  + ", ".join(f"{g} {v}" for g, v in got.items())
                  + f" (config must be {want_cfg}; gates True)")
            if not ok:
                status = 1
    return status


def e1_self_test() -> int:
    """S19b E1's offline scorer passes its own synthetic self-test."""
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/s19b_e1_score.py"), "--self-test"],
                       capture_output=True, text=True)
    print(r.stdout.rstrip() + r.stderr.rstrip())
    return r.returncode


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
        # S21 (docket 20261005-1621 report 10, hunk d10.0; release-4198bb17
        # .C at `if (skeletonFloorBasis_ == "occupancy")` in the elastic site):
        # skeletonFloorBasis is built in at occupancy, so its guard line went
        # and the block it guarded stays. The pin keeps the block, now anchored
        # to the elastic site's own weight declaration.
        ("the elastic site's occupancy weight is skeletonStressWeight()",
         'volScalarFieldw("elasticSolidWeight",phaseIndicator()*intactFraction());'
         '{//S16B:max(ind,floor)*(1-d)--'
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
        # S19b section 2.6-2.7. S21 built blockCoupledLumen in at eliminate
        # (docket 20261005-1621 report 10). Retired with the code they pinned:
        # the read with default eliminate (hunk b15, release-4198bb17
        # .C:1096-1100), the value refusal (c38, .C:3583-3590), the refusal of
        # the key outside blockCoupled (c39, .C:3592-3597) and the
        # PHASE_CLOT_S19_CONFIG echo (c43, .C:3662). A dictionary naming the
        # key is now refused by s21BuiltInKeys (check_refused_dictionary_
        # values.py). The kernel flag is kept, at its built-in value (d56).
        ("S19b: the kernel eliminates the lumen (blockCoupledLumen built in at eliminate)",
         'coef.eliminateLumen=true;'),
        ("S19b: PHASE_CLOT_S19 logs lumenEliminated and the 17-digit maxima",
         '<<"lumenEliminated="<<st.nEliminated<<maxesB.str().c_str();'),
        ("S19b: uMax, vsMax, pMax at 17 significant digits over every cell",
         'for(labeli=0;i<nC;++i){uMaxB=max(uMaxB,mag(U[i]));vsMaxB=max(vsMaxB,mag(Vs_[i]));'
         'pMaxB=max(pMaxB,mag(p[i]));}std::ostringstreammaxesB;maxesB.precision(17);'
         'maxesB<<"uMax="<<uMaxB<<"vsMax="<<vsMaxB<<"pMax="<<pMaxB;'),
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
    if e1_self_test() != 0:
        status = 1
    print("S19 GUARD", "PASSED" if status == 0 else "FAILED")
    return status


if __name__ == "__main__":
    sys.exit(main())
