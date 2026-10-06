#!/usr/bin/env python3
"""Guard for `networkPressureModel poleCubic` -- the compaction law.

WHY THIS FILE EXISTS. Run 20260921-072935 (`em47_horizon_probe`) reached
`phiSOverNominal` 1.1913 in 4e-03 s, still accelerating, and was graded
INADMISSIBLE on `phase out of bounds`. The compaction was measured -- not
argued -- to be mechanical rather than depositional:

  * the over-dense cells are ANTI-correlated with the interface where binding
    happens (excess-weighted mean interfaceDelta / clot mean = 0.781; the
    three worst cells read interfaceDelta/max = 0.00, 0.027, 0.31),
  * `capClamped` was 0 on all 5440 transfer records, so the deposition cap
    never ran,
  * the compacted cells sit at chi = 1.19, far outside the Kres/isSkel
    regularisation band (chi < 5e-03), so that is not the mechanism either.

The same `cubic` law reached endTime 0.03 s at `phiSOverNominal` 1.0524 in run
20260906-210336. What changed between them is `k0p`: 1e-11 -> 1e-13 (shell),
1e-14 -> 1e-15 (core), Xu's values. The Darcy load a mural clot must carry
scales as 1/k, and the pressure difference across the clot is MEASURED at
12.1 Pa in em47's written p field. The cubic's largest reply anywhere in that
run was 2.59 Pa.

WHAT THE LAW HAS TO BE. Neither shipped law is sufficient and each failure is
a run on disk:

  poleAtMax  SIGFPE at t = 9.71e-05 s (20260906-210336) -- p_n(0) != 0 puts a
             finite suction on the diffuse fringe where the momentum diagonal
             is ~0.
  cubic      no pole anywhere, so nothing bounds phi_s by force balance.

`poleCubic` is the cubic multiplied by (phiSMax - phiS0)/(phiSMax - phi_s).
This file asserts the six properties that makes it have, and -- because a
guard that cannot fail is worthless -- that each MUTATION of the expression
loses at least one of them.

S21: the solver keeps only convexExcess with the deposited reference, both
built in. The pins on the deleted laws, the switch tests, their refusals and
the generator's copies are removed; the source checks now pin the convexExcess
code as S21 left it (report 18 of docket
20261005-1621-run110939-b68-free-clots lists each with its hunk).
"""

import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "user-applications" / "phaseClotAxisymmetricFluid" \
              / "phaseClotAxisymmetricFluid.C"

# The main case's anchors. Not free parameters here: they are what
# PHASE_CLOT_CONFIG printed in run 20260921-072935.
PHI0 = 0.45
PHIMAX = 0.6
CAP = 0.99 * PHIMAX

# Measured in that run, for the pricing test below.
LOAD_PA = 12.1          # dp across the clot, 0.01151 m2/s2 * rho 1050
G_SHELL = 4.0
G_CORE = 40.0
CUBIC_REPLY_PA = 2.59   # pnMax at t = 4e-03 s


def cubic(phi, phi0=PHI0, phimax=PHIMAX, cap=None):
    """p_n / K for networkPressureModel `cubic`."""
    phic = min(phi, 0.99 * phimax if cap is None else cap)
    return phic * phic * (phic - phi0) / (4.0 * phi0 ** 3)


def pole(phi, phi0=PHI0, phimax=PHIMAX, cap=None):
    """p_n / K for networkPressureModel `poleAtMax`."""
    phic = min(phi, 0.99 * phimax if cap is None else cap)
    return (phic - phi0) / (phimax - phic)


def pole_cubic(phi, phi0=PHI0, phimax=PHIMAX, cap=None):
    """p_n / K for networkPressureModel `poleCubic` -- the shipped form.

    `cap` mirrors the solver's 0.99*phiSMax clamp. It is a parameter so the
    divergence can be probed BEYOND the clamp: the clamp is a numerical
    backstop, and the property being asserted is a property of the law.
    """
    phic = min(phi, 0.99 * phimax if cap is None else cap)
    return (phic * phic * (phic - phi0) * (phimax - phi0)
            / (4.0 * phi0 ** 3 * (phimax - phic)))


def slope(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2.0 * h)


class LawProperties(unittest.TestCase):
    """The six properties, each one a measured requirement."""

    def test_1_double_root_at_zero(self):
        """p_n(0) = 0 AND p_n'(0) = 0 -- the property poleAtMax lacks."""
        self.assertEqual(pole_cubic(0.0), 0.0)
        self.assertAlmostEqual(slope(pole_cubic, 0.0), 0.0, places=10)
        # and poleAtMax genuinely fails it, which is why it died
        self.assertNotEqual(pole(0.0), 0.0)
        self.assertAlmostEqual(pole(0.0), -PHI0 / PHIMAX, places=12)

    def test_2_fringe_is_weaker_than_the_cubic(self):
        """At the fringe poleCubic is (phimax-phi0)/phimax of the cubic.

        The cubic was adopted to kill a fringe force. A replacement that is
        stronger there would reopen run 20260905-223740's defect.
        """
        for phi in (1e-6, 1e-4, 1e-3, 1e-2):
            # the ratio is exactly (phimax - phi0)/(phimax - phi), which
            # TENDS to 0.25 as phi -> 0 but is not 0.25 at finite phi
            self.assertAlmostEqual(
                pole_cubic(phi) / cubic(phi),
                (PHIMAX - PHI0) / (PHIMAX - phi), places=9,
                msg=f"fringe ratio wrong at phi_s={phi}")
            self.assertLess(
                pole_cubic(phi) / cubic(phi), 0.2543,
                "the fringe must get WEAKER than the cubic, not stronger")
        self.assertAlmostEqual(
            pole_cubic(1e-9) / cubic(1e-9), (PHIMAX - PHI0) / PHIMAX, places=6)

    def test_3_equilibrium_at_nominal(self):
        """Zero at phi_s0, negative below it (resists dilation too)."""
        self.assertAlmostEqual(pole_cubic(PHI0), 0.0, places=12)
        for phi in (0.05, 0.2, 0.35, 0.449):
            self.assertLess(pole_cubic(phi), 0.0)
        for phi in (0.451, 0.5, 0.55):
            self.assertGreater(pole_cubic(phi), 0.0)

    def test_4_diverges_at_close_packing(self):
        """The property the cubic gave up, and the reason for this change."""
        prev = pole_cubic(0.55)
        for phi in (0.57, 0.58, 0.59, 0.5939):
            cur = pole_cubic(phi)
            self.assertGreater(cur, prev)
            prev = cur
        # UNCLAMPED, the law genuinely diverges: each halving of the gap to
        # phimax roughly doubles p_n, without bound.
        far = pole_cubic(0.57, cap=0.5999999)
        for gap in (1e-3, 1e-4, 1e-5, 1e-6):
            self.assertGreater(
                pole_cubic(PHIMAX - gap, cap=0.5999999) / far, 0.0)
        self.assertGreater(
            pole_cubic(PHIMAX - 1e-6, cap=0.5999999) / far, 1.0e4,
            "the law must diverge once the clamp is lifted")
        # CLAMPED, which is what the solver runs: 6.4x over 0.57 -> 0.5939.
        # Stated as a measured number rather than a round one, so a change
        # to the clamp moves this test instead of passing silently.
        self.assertAlmostEqual(
            pole_cubic(0.5939) / pole_cubic(0.57), 6.402, delta=0.01)
        # the cubic does not do this
        self.assertLess(cubic(0.5939) / cubic(0.57), 2.0)

    def test_5_slope_at_nominal_matches_the_cubic(self):
        """The normalisation is DERIVED, not fitted.

        (phimax - phi0) is exactly the factor that makes dp_n/dphi_s at
        nominal identical to the cubic's, K/(4*phi0). That is what preserves
        run 20260906-210336's 30 ms near-equilibrium behaviour.
        """
        self.assertAlmostEqual(
            slope(pole_cubic, PHI0), slope(cubic, PHI0), places=6)
        self.assertAlmostEqual(slope(cubic, PHI0), 1.0 / (4.0 * PHI0), places=6)

    def test_6_priced_against_the_measured_load(self):
        """It must reply to 12.1 Pa inside the admissible range, and the
        pricing must be HONEST about the shell.

        This is the test that refuses to let the law be shipped on the
        strength of its shape alone.
        """
        # the cubic's measured reply is reproduced by the model of it here
        self.assertAlmostEqual(
            cubic(0.5361) * G_CORE, CUBIC_REPLY_PA, delta=0.2)

        def settle(G):
            lo, hi = PHI0, CAP
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if pole_cubic(mid) * G < LOAD_PA:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        core = settle(G_CORE)
        shell = settle(G_SHELL)
        # the core settles comfortably inside
        self.assertLess(core, 0.57, f"core settles at {core}")
        self.assertGreater(core, PHI0)
        # THE SHELL IS MARGINAL AND THE GUARD SAYS SO RATHER THAN HIDING IT.
        # A 4 Pa network under a 12 Pa load nearly close-packs; that is the
        # material, not the law. pnCapCells is what measures whether the
        # clamp actually engages.
        self.assertGreater(shell, 0.58, f"shell settles at {shell}")
        self.assertLess(shell, CAP)


class SourceMatchesTheLaw(unittest.TestCase):
    """The C++ actually implements the form tested above."""

    # S21: poleAtMax, cubic and poleCubic are deleted from the solver
    # (S21B hunks a19, a20, a22, d02), so the branch, its shape and the
    # unknown-model refusal have no source left to pin. The law's properties
    # above stay: they are the record of why convexExcess replaced them.

    def setUp(self):
        self.src = SOLVER.read_text(encoding="utf-8")

    def test_construction_refuses_a_pole_with_no_room(self):
        """phiSMax <= phi_s0 inverts the law's sign; it must be fatal.
        S21 (S21B hunk d03): the cubic exemption went with the cubic, so the
        refusal is unconditional."""
        self.assertIn("needs phiSMax > skeletonVolumeFraction", self.src)
        flat = "".join(self.src.split())
        self.assertIn(
            'if(phiSMax_<=skeletonVolumeFraction_){FatalErrorInFunction'
            '<<"networkPressureModel="<<"convexExcess"'
            '<<"needsphiSMax>skeletonVolumeFraction,butphiSMax="', flat,
            "phiSMax <= phi_s0 must be refused, with no exemption")

    def test_the_cap_is_instrumented(self):
        """A bound no law can report is unfailable (this project's retired
        G3). What CAN fail is the clamp engaging, so it must be counted."""
        self.assertIn("pnCapCells", self.src)
        self.assertIn('" pnCapCells="', self.src)
        self.assertIn('" pnCapFrac="', self.src)


# S21: GeneratorAndSolverAgree is retired. networkPressureModel is built in
# (convexExcess): the solver has no value test left to derive a set from, and
# the generator writes none of the built-in keys (case-engineer report 07,
# item 3), so neither side of the comparison exists.


class MutationsMustFail(unittest.TestCase):
    """A guard that cannot fail is not a guard.

    Each mutation is a plausible mis-implementation of the law; every one
    must lose at least one property above.
    """

    def test_dropping_the_pole_loses_the_divergence(self):
        self.assertLess(cubic(0.5939) / cubic(0.57), 2.0)

    def test_dropping_the_double_root_reopens_the_fringe(self):
        bad = lambda phi: (min(phi, CAP) - PHI0) / (PHIMAX - min(phi, CAP))
        self.assertNotAlmostEqual(bad(0.0), 0.0, places=6)

    def test_multiplying_by_the_pole_instead_of_dividing_inverts_it(self):
        bad = lambda phi: (min(phi, CAP) ** 2 * (min(phi, CAP) - PHI0)
                           * (PHIMAX - min(phi, CAP)) * (PHIMAX - PHI0)
                           / (4.0 * PHI0 ** 3))
        self.assertLess(bad(0.5939), bad(0.55),
                        "a multiplied pole DECREASES near packing")

    def test_omitting_the_derived_norm_breaks_the_slope_match(self):
        bad = lambda phi: (min(phi, CAP) ** 2 * (min(phi, CAP) - PHI0)
                           / (4.0 * PHI0 ** 3 * (PHIMAX - min(phi, CAP))))
        self.assertNotAlmostEqual(slope(bad, PHI0), slope(cubic, PHI0), places=3)

    def test_a_pole_at_nominal_would_be_refused(self):
        """phimax = phi0 is the configuration the constructor now refuses."""
        with self.assertRaises(ZeroDivisionError):
            pole_cubic(PHI0, phi0=PHI0, phimax=PHI0, cap=PHI0)
        # WITH phimax < phi0 THE LAW INVERTS BELOW NOMINAL. Both (phi - phi0)
        # and (phimax - phi0) go negative, their product goes positive, and a
        # network that is DILATING gets a positive (outward) pressure instead
        # of the negative one that should pull it back. Measured here rather
        # than asserted in prose, because I got the direction of this
        # inversion wrong the first time and the test caught it.
        self.assertGreater(
            pole_cubic(0.40, phi0=PHI0, phimax=0.44, cap=0.4356), 0.0,
            "phimax < phi0 must invert the sign below nominal")
        # the correctly-configured law has the right sign at the same point
        self.assertLess(
            pole_cubic(0.40, phi0=PHI0, phimax=PHIMAX), 0.0)


class DepositedReference(unittest.TestCase):
    """S10 fix 1: networkPressureReference `deposited`.

    Run 20260922-104613 (em49, G = 400) broke 24 of 56 wall bonds ON STEP 1.
    The shipped law reads theta < phi_s0 on the seeded diffuse interface as
    DILATED network, so the interface is in tension at t = 0 and its wall
    faces carry theta*|p_n| of normal traction before anything has moved.
    `deposited` evaluates the law at phi_s0 + (theta - thetaNetRef), with
    thetaNetRef = theta at the start of the run.
    """

    COEFF = 1.0

    def traction(self, chi, arg, G):
        """|t| on a wall face = theta * |p_n(arg)| with K_n = coeff*G."""
        theta = chi * PHI0
        return theta * abs(self.COEFF * G * pole_cubic(arg))

    def test_nominal_reproduces_the_em49_step_one_number(self):
        """0.00725*G at chi ~ 0.82: 2.8996 Pa logged at G = 400, step 1."""
        worst = max(self.traction(c / 10000.0, c / 10000.0 * PHI0, 400.0)
                    for c in range(1, 10000))
        self.assertAlmostEqual(worst / 2.8996, 1.0, delta=1e-3)

    def test_deposited_is_stress_free_at_the_start_for_every_chi(self):
        for c in (1e-9, 1e-4, 0.01, 0.3, 0.5, 0.818, 0.95, 1.0):
            theta = c * PHI0
            arg = PHI0 + (theta - theta)          # thetaNetRef = theta
            self.assertEqual(self.traction(c, arg, 4000.0), 0.0)

    def test_deposited_bulk_is_the_shipped_law(self):
        """thetaNetRef = phi_s0 in the undeformed bulk: argument = theta."""
        for theta in (0.40, 0.45, 0.50, 0.57):
            self.assertEqual(pole_cubic(PHI0 + (theta - PHI0)),
                             pole_cubic(theta))

    def test_compaction_and_dilation_are_both_resisted(self):
        ref = 0.2                                  # a part-filled edge cell
        self.assertGreater(pole_cubic(PHI0 + (ref * 1.1 - ref)), 0.0)
        self.assertLess(pole_cubic(PHI0 + (ref * 0.9 - ref)), 0.0)

    def test_growth_is_not_compaction(self):
        theta, ref, d = 0.23, 0.21, 0.017
        self.assertEqual(PHI0 + ((theta + d) - (ref + d)),
                         PHI0 + (theta - ref))

    def test_why_an_excess_and_not_a_ratio(self):
        """The lumen: content is a numerical tail and the skeleton flux is not
        solenoidal, so theta/thetaNetRef can drift by orders of magnitude.
        The ratio form then sits on the pole; the excess form does not."""
        theta, ref = 1e-4, 1e-6                   # ratio 100
        ratio_arg = PHI0 * theta / ref
        excess_arg = PHI0 + (theta - ref)
        ratio_term = theta * abs(pole_cubic(ratio_arg))
        excess_term = theta * abs(pole_cubic(excess_arg))
        self.assertGreater(ratio_term, 1e3 * excess_term)
        self.assertLess(excess_term, 1e-7)

    def test_the_source_implements_the_excess(self):
        src = SOLVER.read_text()
        i = src.find("phaseClotAxisymmetricFluid::networkPressureArgument()")
        self.assertGreater(i, 0, "networkPressureArgument() is gone")
        body = "".join(src[i:src.find("\n}\n", i)].split())
        # S21: deposited is built in -- the guard (S21B hunk d34) and the
        # nominal forceFraction() fall-back (a18) are deleted, so the excess
        # is the function's last and only return.
        self.assertIn("skeletonVolumeFraction_)+(thetaC_+thetaH_)-thetaNetRef_",
                      body)
        self.assertTrue(body.endswith("+(thetaC_+thetaH_)-thetaNetRef_);}"),
                        "the excess must be the argument the function returns")

    def test_a_ratio_form_in_the_source_would_fail(self):
        src = SOLVER.read_text()
        good = "          + (thetaC_ + thetaH_)\n          - thetaNetRef_"
        self.assertEqual(src.count(good), 1, "mutation target absent")
        bad = "          *(thetaC_ + thetaH_)\n          /thetaNetRef_"
        mutated = src.replace(good, bad, 1)
        self.assertNotEqual(mutated, src)
        i = mutated.find("phaseClotAxisymmetricFluid::networkPressureArgument()")
        body = "".join(mutated[i:mutated.find("\n}\n", i)].split())
        self.assertNotIn(
            "skeletonVolumeFraction_)+(thetaC_+thetaH_)-thetaNetRef_", body)


# ---------------------------------------------------------------------------
# S11 (2026-09-23): networkPressureModel convexExcess.
#
# research/designs/2026-09-23-S11-root-causes-em52.md. The dense network's
# volumetric Cauchy stress from a CONVEX energy, s(J) = K(1-J_c)J(J-1)/(J-J_c)
# with J = phi0/a and J_c = phi0/phimax; in the excess argument a,
#     p(a)/K = (phimax - phi0)(a - phi0)/(a (phimax - a)),
# weighted by the DEPOSITED occupancy thetaNetRef/phi0. em52 (run
# 20260923-031942) died of a void that poleCubic's softening let open: its
# partial tension peaks at 2.9 Pa (0.7% of G = 400) and falls to zero.
# ---------------------------------------------------------------------------

def convex_excess(a, phi0=PHI0, phimax=PHIMAX):
    """p/K for networkPressureModel convexExcess, argument a (the excess)."""
    return (phimax - phi0)*(a - phi0)/(a*(phimax - a))


def partial(law, theta, theta_ref, phi0=PHI0, weight="deposited"):
    """Partial (per mixture area) pressure Pi at content theta.

    convexExcess weights by the deposited occupancy theta_ref/phi0; the
    legacy laws by the current content theta (forceFraction).
    """
    a = phi0 + theta - theta_ref
    w = theta_ref/phi0 if weight == "deposited" else theta
    return w*law(a)


def softens_somewhere(law, theta_ref, weight):
    """True if Pi DEcreases with theta anywhere a is in (0, phimax): a
    spinodal, where a dilated cell pulls back less the more it dilates."""
    lo = max(theta_ref - PHI0, 0.0) + 1e-4
    hi = PHIMAX - PHI0 + theta_ref - 1e-4
    n = 20000
    prev = None
    for i in range(n + 1):
        th = lo + (hi - lo)*i/n
        if th <= 0:
            continue
        v = partial(law, th, theta_ref, weight=weight)
        if prev is not None and v < prev - 1e-12:
            return True
        prev = v
    return False


class ConvexExcess(unittest.TestCase):
    def test_stress_free_as_deposited(self):
        self.assertEqual(convex_excess(PHI0), 0.0)

    def test_bulk_modulus_is_K_exactly(self):
        """dp/dJ = -K at the reference, J = phi0/a: poleCubic realised K/4."""
        dpda = slope(convex_excess, PHI0)
        self.assertAlmostEqual(dpda*PHI0, 1.0, places=6)
        self.assertAlmostEqual(slope(pole_cubic, PHI0)*PHI0, 0.25, places=6)

    def test_strictly_monotone_on_the_whole_domain(self):
        prev = convex_excess(1e-5)
        for i in range(1, 60000):
            a = 1e-5 + (PHIMAX - 2e-5)*i/60000
            v = convex_excess(a)
            self.assertGreater(v, prev, f"not increasing at a = {a}")
            prev = v

    def test_closed_form_derivative_numerator_has_no_root(self):
        """d/da [(a-phi0)/(a(phimax-a))] has numerator (a-phi0)^2 +
        phi0(phimax-phi0) > 0 -- the reason the solver's comment gives."""
        for a in (0.01, 0.2, PHI0, 0.55, 0.59):
            num = (a - PHI0)**2 + PHI0*(PHIMAX - PHI0)
            analytic = (PHIMAX - PHI0)*num/(a*(PHIMAX - a))**2
            self.assertAlmostEqual(slope(convex_excess, a), analytic,
                                   delta=1e-4*abs(analytic))
            self.assertGreater(num, 0)

    def test_both_ends_are_infinite_barriers(self):
        self.assertLess(convex_excess(1e-9), -1e6)
        self.assertGreater(convex_excess(PHIMAX - 1e-9), 1e6)

    def test_the_partial_pressure_never_softens(self):
        """No spinodal for ANY deposited occupancy, the property poleCubic
        lacks -- checked across the fringe and the bulk."""
        for ref in (0.01, 0.05, 0.2, 0.45):
            self.assertFalse(softens_somewhere(convex_excess, ref, "deposited"),
                             f"convexExcess softens at thetaNetRef {ref}")

    def test_poleCubic_and_Cogan_Guy_DO_soften(self):
        """The contrast that makes the previous test mean something: the
        shipped law softens below its tension peak (em52's voids), and so
        does Cogan-Guy's pole form in partial form, below theta = 0.30."""
        self.assertTrue(softens_somewhere(pole_cubic, PHI0, "content"))
        self.assertTrue(softens_somewhere(pole, PHI0, "content"))

    def test_tension_capacity_versus_poleCubic(self):
        """poleCubic's partial tension peaks at 0.00725*G (2.9 Pa at 400);
        convexExcess at a = 0.1 already holds 1.05*K."""
        pc_peak = -min(partial(pole_cubic, th/1000, PHI0, weight="content")
                       for th in range(1, 600))
        self.assertAlmostEqual(pc_peak, 0.00725, delta=2e-4)
        self.assertAlmostEqual(-convex_excess(0.1), 1.05, places=6)

    def test_the_source_implements_it(self):
        src = SOLVER.read_text(encoding="utf-8")
        # S21: convexExcess is built in. Its `if` (S21B hunk d35) is gone, so
        # the body checked is all of networkPressure(); the FatalError the
        # old branch held was the deposited refusal (a21), deleted, and the
        # out-of-domain report is the shared check it calls.
        i = src.index("phaseClotAxisymmetricFluid::networkPressure() const")
        body = src[i:src.index("\n}\n", i)]
        flat = "".join(body.split())
        self.assertIn(
            "(phiSMax-phiS0)*(phiS-phiS0)/(phiS*(phiSMax-phiS))", flat)
        self.assertIn('checkNetworkArgument(phiS,"convexExcessnetworkpressure");',
                      flat, "out-of-domain must be reported, not bounded")
        self.assertNotIn("min(phiS", flat)
        self.assertNotIn("max(phiS", flat)
        self.assertNotIn("phiSc", flat, "the legacy clamp must not be used")
        # S16 (2026-09-24): the S11 body moved, unchanged, into
        # networkPartialPressureUndamaged(); networkPartialPressure() is now
        # networkPressureDamaged() of it (the identity unless
        # networkPressureDamage tensile). Both halves are checked.
        j = src.index("phaseClotAxisymmetricFluid::networkPartialPressureUndamaged()")
        pp = "".join(src[j:src.index("\n}\n", j)].split())
        self.assertIn('thetaNetRef_/dimensionedScalar("phiS0"', pp)
        # S21: the legacy-law forceFraction() weight was unreachable once
        # convexExcess is built in and is deleted (S21B hunk a30).
        k = src.index("phaseClotAxisymmetricFluid::networkPartialPressure() const")
        wrap = "".join(src[k:src.index("\n}\n", k)].split())
        self.assertIn(
            "returnnetworkPressureDamaged(networkPartialPressureUndamaged());",
            wrap)

    # S21: test_generator_and_solver_refuse_it_without_deposited is retired:
    # the solver refusal is deleted (S21B hunk a21) and the generator writes
    # neither key (case-engineer report 07, item 3).


class ConvexExcessMutations(unittest.TestCase):
    """Each property above must reject a plausible wrong law."""

    def test_dropping_the_one_over_a_is_Cogan_Guy_and_softens(self):
        cg = lambda a: (PHIMAX - PHI0)*(a - PHI0)/(PHI0*(PHIMAX - a))
        self.assertTrue(softens_somewhere(cg, PHI0, "content"))

    # NOT a mutation, and recorded because I first wrote it as one: for
    # fully deposited material, theta*p(theta) = K(phimax-phi0)(theta-phi0)/
    # (phimax-theta) is Cogan-Guy's INTRINSIC form, which is monotone -- the
    # 1/a cancels the content weight exactly. The deposited weight is chosen
    # because it is exact for the dense network and zero where nothing was
    # deposited, not because the other weight would soften.
    def test_content_weight_is_Cogan_Guy_intrinsic_when_fully_deposited(self):
        for th in (0.05, 0.2, 0.44, 0.55):
            cg = (PHIMAX - PHI0)*(th - PHI0)/(PHIMAX - th)
            self.assertAlmostEqual(
                partial(convex_excess, th, PHI0, weight="content"), cg,
                places=12)

    def test_a_missing_normalisation_breaks_the_bulk_modulus(self):
        bad = lambda a: (a - PHI0)/(a*(PHIMAX - a))
        self.assertNotAlmostEqual(slope(bad, PHI0)*PHI0, 1.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
