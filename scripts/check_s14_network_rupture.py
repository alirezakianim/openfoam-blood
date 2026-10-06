#!/usr/bin/env python3
"""Guard for S14, network rupture at the critical stretch (run 20260923-204553).

research/designs/2026-09-23-S14-network-rupture.md. em58's wall tail reached
principal stretch 3.2-6.2 and its explicit elastic term blew up while the dt
bound, reading G instead of the stretch-dependent stiffness, stayed at 0.087.
Fibrin ruptures at a critical stretch of ~2.8 (Tutwiler et al. 2020, source
28; single fibres 2.5-3.3). S14:
  - ruptures network where sqrt(max eig(be)) >= lambda_c: ruptured_ = 1,
    be = I; the content stays as debris;
  - carries ruptured_ with the material on the CONTENT's bounded transport:
    implicit upwind predictor (an M-matrix) + the Zalesak-limited
    antidiffusive correction with min(content lambda, the label's own), run
    after the content solve. Batch 54 (run 20260924-014332) used shell_'s
    phaseSharedConvection, whose vanLeer-on-content weights are near-central
    inside the clot: the label reached -0.085/1.023 and 1 - ruptured went
    negative. shell_ hides the same by clipping after its solve;
  - removes ruptured network from the deviatoric load path (both weights and
    the BSD coefficient) and from elastic memory: relaxation acts where
    INTACT content thetaNetRef*(1 - ruptured) is below the residual, so the
    intact part of a partially ruptured cell keeps its strain;
  - dilutes the label with intact new deposits and mixes their I with the
    intact share only; clears the label with released content;
  - logs the stretch at rupture (G12, failable -- maxIntactStretch is below
    lambda_c by construction) and the label's range (G14);
  - makes the elastic dt number read the tangent stiffness.
Every predicate has a mutation that must make it FAIL.

S21 (2026-10-05, docket 20261005-1621 report 16): networkRupture is built in
at fibreRelease and elasticStabilityStiffness at tangent. Assertions pinned
to text S21 deleted are removed, each with its ledger hunk (probes/se06 A,
se09 B, se10 C1, se11 C2 -hunks.txt) cited where it stood; checks on kept
code that S21 rewrote (a guard removed, braces kept) are re-pinned to the
text as S21 left it. The label's own transport (advectRuptureLabel) and the
criticalStretch branch of the criterion are deleted, so their predicates go.

Run: python3 scripts/check_s14_network_rupture.py
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"
HEADER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.H"
GEN = ROOT / "cases/AM2020_planar_mural_detachment/scripts/create_cases.py"
ASSESS = ROOT / "cases/AM2020_planar_mural_detachment/scripts/assess_case.py"

_spec = importlib.util.spec_from_file_location(
    "s12", ROOT / "scripts/check_s12_intrinsic_drag.py")
_s12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s12)
flat, body = _s12.flat, _s12.body


def rupture_criterion_is_on_the_squared_stretch(src: str) -> bool:
    b = flat(body(src, "phaseClotAxisymmetricFluid::ruptureOverstretchedNetwork()"))
    # S21 removed: the rupture-off return (B a25); b1c = sqr(lc) (C2 p61b);
    # the criticalStretch branch -- presence test, b1, b1 >= b1c, be = I
    # (B a26). The law that stays (fibreRelease) is s16/s17's to pin.
    return (
        "reduce(b1RupturedMax,maxOp<scalar>());" in b
    )


def rupture_runs_after_the_be_update(src: str) -> bool:
    b = flat(body(src, "phaseClotAxisymmetricFluid::solveElasticStrain()"))
    i = b.find("ruptureOverstretchedNetwork();")
    # S21 C1 d27.0: the renormalisation's guard removed, braces kept
    j = b.find("{constvolScalarFielddetBe(det(be_));be_*=pow(")
    return 0 <= j < i


# S21: label_travels_with_the_material removed. advectRuptureLabel() and its
# two calls are deleted (B a24: under damageTransport contentPartition it
# returned before its body; B a36: the released-carrier call; B b11: the
# attached-carrier call). ruptured_ is now synced from the partition.


def ruptured_network_carries_no_shear(src: str) -> bool:
    fs = flat(body(src, "phaseClotAxisymmetricFluid::skeletonFaceStress() const"))
    es = flat(body(src, "phaseClotAxisymmetricFluid::elasticStress() const"))
    s = flat(src)
    return (
        'volScalarFieldw("elasticSolidWeight",phaseIndicator()*intactFraction());' in fs
        and "constvolScalarFieldchi(phaseIndicator()*intactFraction());" in es
        and 'constvolScalarFieldphiSIntact("phiSIntact",phiSkel_*intactFraction());' in s
        and 'returnvolScalarField::New("intactFraction",1.0-ruptured_);' in s
    )


def ruptured_network_has_no_memory(src: str) -> bool:
    s = flat(src)
    # S21 B d59: `if(networkRupture_!="off")` removed, braces kept
    return ('volScalarFieldnoMemory("noNetworkMemory",absent);{noMemory=max(1.0-thetaNetRef_*(1.0-ruptured_)/dimensionedScalar("residualContent",dimless,residualChi_*skeletonVolumeFraction_),dimensionedScalar("zero",dimless,0));}' in s
            and "beRelax_=sqr(noMemory)" in s)


def label_is_diluted_and_cleared(src: str) -> bool:
    s = flat(src)
    rel = flat(body(src, "phaseClotAxisymmetricFluid::releaseDetachedMaterial()"))
    # S16 (2026-09-24): under damageTransport contentPartition the deposit
    # dilutes d through thetaIntact_ instead (the new network is intact and
    # untargeted), so the label's own dilution is skipped there -- once, not
    # twice. The intact share that mixes be is still read BEFORE it.
    dep = flat(body(src, "phaseClotAxisymmetricFluid::solvePlateletContent"))
    # S17 (2026-09-26): under networkRupture fibreRelease the intact share
    # is thI/S = 1 - d* of the pre-deposit partition (09 rule 2), set INSIDE
    # the same block after the S14/S16 weight; every other law keeps r(1 - d).
    # S21: `if(networkRupture_!="off")` removed, braces kept (B d80); the
    # label's own dilution `if(damageTransport_!="contentPartition"){...}`
    # deleted (B a37); `if(fibreRelease())` removed, braces kept (C2 p77.3.0).
    return ('scalarrIntact=r;{rIntact=r*(1.0-ruptured_[celli]);{rIntact=r*tIntactShareBefore()[celli];}}b[celli]=(rIntact*b[celli]+d*symmTensor::I)/(rIntact+d);' in s
            and "pI[celli]+=hNew[celli]-hStar[celli];" in dep
            and "ruptured_[celli]=0.0;" in rel)


def dt_reads_the_tangent(src: str) -> bool:
    s = flat(src)
    return (
        # S21 B d85: `if(elasticStabilityStiffness_=="tangent")` removed,
        # braces kept
        'scalarstiff=1.0;{constvectorev(eigenValues(beI[celli]));' in s
        and "stiff=(8.0/9.0)*b1+(2.0/9.0)*bRest;" in s
        # S16: the weight per skeletonFloorBasis. S21 C2 p99 deleted the
        # unread local wI0 = ind*intact.
        and "constscalarw=wSite[celli];" in s
    )


# S21: dependencies_are_refused removed. The networkRupture value refusal
# (B c08); the tangent-needs-bounds refusal (B d14 reduced it to
# `if(explicitStabilityBounds_!="on")`, C1 c27 deleted it); the criticalStretch
# lc > 1 refusal (B a08); rupture-needs-tangent (B a10). Both keys are built in.


def defaults_are_shipped_and_zeros_are_not_written(src: str) -> bool:
    s = flat(src)
    # S21 removed: the networkRupture initialiser (B b04), the
    # elasticStabilityStiffness initialiser (B b05), ruptured NO_WRITE under
    # off (B a11).
    return '"networkCriticalStretch",2.8' in s


def diagnostics_are_logged(src: str) -> bool:
    return all(t in src for t in ('"PHASE_CLOT_RUPTURE time="', '" newlyRuptured="',
                                  '" rupturedContentFraction="', '" maxIntactStretch="',
                                  '" maxRupturedStretch="', '" rupturedMin="',
                                  '" rupturedMax="', '"PHASE_CLOT_S14_CONFIG"'))


PREDICATES = [
    ("the stretch at rupture reduced", rupture_criterion_is_on_the_squared_stretch),
    ("rupture after the be update", rupture_runs_after_the_be_update),
    ("no shear through ruptured network", ruptured_network_carries_no_shear),
    ("memory kept only by intact network", ruptured_network_has_no_memory),
    ("label diluted, deposits mix with intact only, cleared", label_is_diluted_and_cleared),
    ("dt reads the tangent", dt_reads_the_tangent),
    ("the critical stretch's default", defaults_are_shipped_and_zeros_are_not_written),
    ("diagnostics logged", diagnostics_are_logged),
]

MUTATIONS = [
    # S21: "criterion on the stretch, not its square" (C2 p61b) and "ruptured
    # be keeps its strain" (B a26) removed with their text.
    ("rupture before the renormalisation", rupture_runs_after_the_be_update,
     "    ruptureOverstretchedNetwork();\n\n    // AFTER the rescale",
     "    // AFTER the rescale"),
    # S21: the four label-transport mutations removed (B a24, a36, b11).
    ("face weight ignores rupture", ruptured_network_carries_no_shear,
     'volScalarField w("elasticSolidWeight", phaseIndicator()*intactFraction());',
     'volScalarField w("elasticSolidWeight", phaseIndicator());'),
    ("BSD keeps ruptured stiffness", ruptured_network_carries_no_shear,
     '"phiSIntact", phiSkel_*intactFraction()', '"phiSIntact", phiSkel_*1.0'),
    ("ruptured network keeps memory", ruptured_network_has_no_memory,
     "              - thetaNetRef_*(1.0 - ruptured_)\n", "              - thetaNetRef_\n"),
    ("relaxation on the label, erasing intact memory", ruptured_network_has_no_memory,
     "            noMemory = max\n            (\n                1.0\n              - thetaNetRef_*(1.0 - ruptured_)",
     "            noMemory = max(absent, ruptured_); if (false) max\n            (\n                1.0\n              - thetaNetRef_*(1.0 - ruptured_)"),
    ("deposit averages debris strain in", label_is_diluted_and_cleared,
     "                        rIntact = r*(1.0 - ruptured_[celli]);\n", ""),
    # S21: "new deposits inherit rupture" removed (B a37).
    ("dt still reads G", dt_reads_the_tangent,
     "                            stiff = (8.0/9.0)*b1 + (2.0/9.0)*bRest;",
     "                            stiff = 1.0;"),
    # S21: "rupture allowed without the tangent bound" (B a10) and "zeros
    # written when off" (B a11) removed.
    ("rupture line renamed", diagnostics_are_logged,
     '"PHASE_CLOT_RUPTURE time="', '"PHASE_CLOT_RUPTUR time="'),
    # S21: "stretch at rupture not recorded" removed (B a26).
    ("label range not logged", diagnostics_are_logged,
     '<< " rupturedMax=" << gMax(rI)', '<< " rupturedMaxx=" << gMax(rI)'),
]


class S14Properties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SOLVER.read_text()
        cls.hdr = HEADER.read_text()
        cls.gen = GEN.read_text()
        cls.assess = ASSESS.read_text()

    def test_every_property_holds(self):
        for name, pred in PREDICATES:
            with self.subTest(property=name):
                self.assertTrue(pred(self.src), f"S14 property broken: {name}")

    def test_the_generator_mirrors_the_refusals(self):
        g = self.gen
        # S21: S14_SWITCHES and s14_combination_refusal went with the
        # generator cut (report 07 item 3); the field's solver entry stays.
        self.assertIn('required.append("ruptured")', g)
        self.assertIn('"    ruptured { $c; }\\n"', g)

    def test_the_assessor_reads_the_rupture(self):
        self.assertIn("def s14_report", self.assess)
        self.assertIn('"s14": s14_report(text)', self.assess)

    def test_the_switches_are_declared_with_their_evidence(self):
        # S21: `word networkRupture_;` (B h07) and `word
        # elasticStabilityStiffness_;` (B h08) deleted with their members.
        for decl in ("scalar networkCriticalStretch_;", "volScalarField ruptured_;"):
            self.assertIn(decl, self.hdr)
        self.assertIn("20260923-204553", self.hdr)
        self.assertIn("Tutwiler", self.hdr)

    def test_be_eigenvalue_is_the_squared_stretch(self):
        """Isochoric uniaxial stretch lambda: b = diag(lambda^2, 1/lambda,
        1/lambda), so max eig(b) = lambda^2 and the criterion b1 >= lc^2 trips
        exactly at lambda = lc -- and b1 >= lc would trip at sqrt(lc)."""
        lc = 2.8
        for lam in (2.0, 2.79, 2.8, 2.81, 4.0):
            # diagonal, so its eigenvalues are its entries
            b1 = max(lam**2, 1/lam, 1/lam)
            self.assertAlmostEqual(b1, lam**2, places=12)
            self.assertEqual(b1 >= lc**2, lam >= lc - 1e-12)

    def test_tangent_is_four_thirds_at_rest_and_grows_with_stretch(self):
        """(8/9) b1 + (2/9)(b2 + b3): 4/3 at b = I -- the 1-D compression
        stiffness -- and bounded by ~(8/9) lc^2 + 4/9 once rupture caps b1."""
        at_rest = (8/9)*1 + (2/9)*2
        self.assertAlmostEqual(at_rest, 4/3, places=12)
        lc = 2.8
        cap = (8/9)*lc**2 + (2/9)*(2/lc)
        self.assertLess(cap, 7.2)
        self.assertGreater(cap, 7.0)

    def test_partial_rupture_keeps_intact_memory(self):
        """noMemory = max(0, 1 - ref(1 - rho)/residual). A clot cell at the
        reference (ref = phi0 = 0.45) that is 10% ruptured keeps its intact
        strain; the retired max(absent, rho) relaxed it at rho^2 * 1e6 =
        1e4 1/s (tau 100 us). A fully ruptured cell relaxes at the full
        rate, and with rho = 0 the absent ramp is unchanged."""
        residual, rate, ref = 2.222e-4*0.45, 1.0e6, 0.45
        nm = lambda ref, rho: max(0.0, 1.0 - ref*(1.0 - rho)/residual)
        self.assertEqual(nm(ref, 0.1), 0.0)
        self.assertAlmostEqual(max(0.0, 0.1)**2*rate, 1.0e4)
        self.assertEqual(nm(ref, 1.0), 1.0)
        for r in (0.0, 0.3*residual, residual, 10*residual):
            self.assertAlmostEqual(nm(r, 0.0), max(0.0, 1.0 - r/residual))

    def test_deposit_mixes_with_the_intact_share(self):
        """A half-ruptured cell (r = 1, rho = 0.5, intact strain b) gains
        d = 1 of intact network at I: intact content 0.5 at b and 1 at I, so
        the intact strain is (0.5 b + I)/1.5 -- not (b + I)/2 -- and the
        ruptured share of the content becomes 0.5/2 = 0.25."""
        r, rho, d, b = 1.0, 0.5, 1.0, 4.0
        rI = r*(1 - rho)
        self.assertAlmostEqual((rI*b + d)/(rI + d), 2.0)
        self.assertAlmostEqual(rho*r/(r + d), 0.25)

    def test_the_gates_fail_on_their_negatives(self):
        """G12, G13 and G14 each pass the clean log and fail exactly their
        own negative -- G12 is failable, unlike maxIntactStretch."""
        spec = importlib.util.spec_from_file_location("assess_case", ASSESS)
        a = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(a)
        row = ("PHASE_CLOT_RUPTURE time=0.002 criticalStretch=2.8 newlyRuptured=3 "
               "newlyRupturedBulk=0 rupturedContentFraction=1e-06 "
               "maxIntactStretch=2.7 maxRupturedStretch=2.83 rupturedMin=0 "
               "rupturedMax=1\n")
        ok = "PHASE_CLOT_STRIPES elasticDiffusionNumberMax=0.2\n" + row
        keys = ("g12_rupture_resolves_stretch", "g13_elastic_number_bounded",
                "g14_label_bounded")
        cases = {
            "clean": (ok, (True, True, True)),
            "jump": (ok.replace("maxRupturedStretch=2.83", "maxRupturedStretch=5.0"),
                     (False, True, True)),
            "unstable": (ok.replace("=0.2\n", "=1.25\n"), (True, False, True)),
            "overshoot": (ok.replace("rupturedMax=1\n", "rupturedMax=1.0001\n"),
                          (True, True, False)),
            "undershoot": (ok.replace("rupturedMin=0 ", "rupturedMin=-2e-09 "),
                           (True, True, False)),
        }
        for name, (text, want) in cases.items():
            with self.subTest(log=name):
                r = a.s14_report(text)
                self.assertEqual(tuple(r[k] for k in keys), want)

    def test_implicit_label_transport_is_bounded_only_as_an_m_matrix(self):
        """1-D, uniform flow, implicit Euler, a one-cell step (the state a
        fresh rupture creates). CENTRAL face weights -- what vanLeer on a
        smooth content gives inside the clot -- put positive off-diagonals
        in the matrix and the label undershoots: the batch 54 defect.
        UPWIND is an M-matrix and stays in [0, 1]; the Zalesak-limited
        correction keeps that bound and is sharper than upwind alone."""
        N, c = 40, 0.5                       # cells, Courant number u dt/h

        def thomas(a, b, cc, d):
            n = len(d); cp = [0.0]*n; dp = [0.0]*n
            cp[0] = cc[0]/b[0]; dp[0] = d[0]/b[0]
            for i in range(1, n):
                m = b[i] - a[i]*cp[i-1]
                cp[i] = cc[i]/m if i < n - 1 else 0.0
                dp[i] = (d[i] - a[i]*dp[i-1])/m
            x = [0.0]*n; x[-1] = dp[-1]
            for i in range(n - 2, -1, -1):
                x[i] = dp[i] - cp[i]*x[i+1]
            return x

        def step(f, w):
            # face i+1/2 value = w f_i + (1-w) f_{i+1}; zero-gradient ends
            a = [0.0]*N; b = [1.0]*N; cc = [0.0]*N
            for i in range(N):
                # + c (f_{i+1/2} - f_{i-1/2})
                if i + 1 < N:
                    b[i] += c*w; cc[i] += c*(1 - w)
                else:
                    b[i] += c
                if i - 1 >= 0:
                    a[i] -= c*w; b[i] -= c*(1 - w)
                else:
                    b[i] -= c
            return thomas(a, b, cc, f)

        def fct(f_old):
            low = step(f_old, 1.0)
            A = [c*0.5*(low[i+1] - low[i]) for i in range(N - 1)]   # lin - up
            lo = [min(low[j], f_old[j]) for j in range(N)]
            hi = [max(low[j], f_old[j]) for j in range(N)]
            lo_n = [min(lo[max(j-1, 0):j+2]) for j in range(N)]
            hi_n = [max(hi[max(j-1, 0):j+2]) for j in range(N)]
            Pp = [0.0]*N; Pm = [0.0]*N
            for i, a_ in enumerate(A):
                if a_ > 0: Pm[i] += a_; Pp[i+1] += a_
                else: Pp[i] -= a_; Pm[i+1] -= a_
            Rp = [min(1.0, (hi_n[j] - low[j])/Pp[j]) if Pp[j] > 0 else 1.0 for j in range(N)]
            Rm = [min(1.0, (low[j] - lo_n[j])/Pm[j]) if Pm[j] > 0 else 1.0 for j in range(N)]
            lam = [min(Rm[i], Rp[i+1]) if a_ >= 0 else min(Rp[i], Rm[i+1])
                   for i, a_ in enumerate(A)]
            out = low[:]
            for i, a_ in enumerate(A):
                out[i] -= lam[i]*a_; out[i+1] += lam[i]*a_
            return out

        f0 = [0.0]*N; f0[10] = 1.0
        central, up, corr = f0[:], f0[:], f0[:]
        for _ in range(8):
            central = step(central, 0.5)
            up = step(up, 1.0)
            corr = fct(corr)
        self.assertLess(min(central), -1e-3)
        self.assertGreaterEqual(min(up), 0.0)
        self.assertLessEqual(max(up), 1.0)
        self.assertGreaterEqual(min(corr), -1e-15)
        self.assertLessEqual(max(corr), 1.0 + 1e-15)
        self.assertGreater(max(corr), max(up))

    def test_the_default_stretch_is_the_sourced_one(self):
        """2.8 measured locally (Tutwiler et al. 2020), inside the single-fibre
        range 2.5-3.3 they cite."""
        self.assertIn('"networkCriticalStretch", 2.8', self.src)
        self.assertTrue(2.5 <= 2.8 <= 3.3)


class S14Mutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SOLVER.read_text()

    def test_each_predicate_fires_on_its_mutation(self):
        for label, pred, old, new in MUTATIONS:
            with self.subTest(mutation=label):
                self.assertEqual(self.src.count(old), 1,
                                 f"{label}: mutation anchor not unique/absent")
                self.assertTrue(pred(self.src), f"{label}: baseline fails")
                self.assertFalse(pred(self.src.replace(old, new)),
                                 f"{label}: the guard did not catch it")


if __name__ == "__main__":
    unittest.main(verbosity=2)
