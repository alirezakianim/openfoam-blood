#!/usr/bin/env python3
"""Guard for S12, the intrinsic occupancy drag (run 20260923-085741).

research/designs/2026-09-23-S12-intrinsic-occupancy-drag.md. Under
dragOccupancyBasis intrinsic the interphase drag of a partial cell is its
occupancy times the network's OWN Kozeny-Carman drag at its intrinsic density,
ind*(1-a)^2*nu/k_KC(a) -- the diffuse-domain weighting the S11 stress already
has. This file asserts that the solver source does exactly that and that the
law's numbers are the ones the design priced. (S21: intrinsic is built in --
the switch, its default, its refusals and the generator's mirror of them are
deleted, so their pins are too; report 18 of docket
20261005-1621-run110939-b68-free-clots lists each with its hunk.) Every predicate has a mutation beside it that must
make it FAIL (write-the-negative-test-before-trusting-a-guard).

What it does NOT check: that the fix cures the run. That is em55/em56's job,
pre-registered in mainCaseControls.json (G10, G11).

Run: python3 scripts/check_s12_intrinsic_drag.py
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"
HEADER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.H"
GEN = ROOT / "cases/AM2020_planar_mural_detachment/scripts/create_cases.py"
ASSESS = ROOT / "cases/AM2020_planar_mural_detachment/scripts/assess_case.py"

LEXEME = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', re.S)


def code(text: str) -> str:
    """Comments removed, string literals kept: a property of the CODE must
    not be satisfied by a comment that describes it."""
    return LEXEME.sub(
        lambda m: m.group() if m.group().startswith('"') else "", text)


def flat(text: str) -> str:
    return re.sub(r"\s+", "", code(text))


def body(src: str, signature: str) -> str:
    """Brace-matched body of the member function whose definition line
    contains `signature`."""
    i = src.find(signature)
    if i < 0:
        return ""
    j = src.find("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j:k + 1]
    return ""


# ---- predicates -----------------------------------------------------------

def permeability_is_the_case_law_at_the_intrinsic_density(src: str) -> bool:
    """k_KC(a), anchored at k0 at a = phi0, a from networkPressureArgument(),
    checked in its domain, with NO floor and NO cap."""
    b = flat(body(src, "phaseClotAxisymmetricFluid::intrinsicPermeability() const"))
    return (
        "constvolScalarFielda(networkPressureArgument());" in b
        and 'checkNetworkArgument(a,"intrinsicpermeability");' in b
        and "constscalarref=pow3(1.0-phi0)/sqr(phi0);" in b
        and 'k0pField_*pow3(1.0-a)/sqr(a)/dimensionedScalar("ref",dimless,ref)' in b
        and "max(" not in b and "min(" not in b
    )


def drag_is_occupancy_times_intrinsic(src: str) -> bool:
    """ind*(1-a)^2*nu/k_KC(a), the indicator UNFLOORED (a bulk coefficient,
    not a support) and the same a as the permeability."""
    b = flat(body(src, "phaseClotAxisymmetricFluid::intrinsicDragRate() const"))
    return (
        "constvolScalarFielda(networkPressureArgument());" in b
        and "phaseIndicator()*sqr(1.0-a)*nu_/intrinsicPermeability()" in b
        and "max(" not in b
    )


def both_drag_sites_and_the_written_field_use_it(src: str) -> bool:
    """S21: intrinsic is built in, so both overwrites are unconditional -- the
    momentum drag straight after its legacy construction, and the written
    poroPerm in updatePermeability(). (The dynamic rho_* site went with
    solveSkeleton(), S21C1 hunk b23.)"""
    s = flat(src)
    perm = flat(body(src, "phaseClotAxisymmetricFluid::updatePermeability()"))
    return (
        'permeabilityFloor_)))());{dragCoeff=intrinsicDragRate();poroPerm_=intrinsicPermeability();}' in s
        and '(tPermeability_);{poroPerm_=intrinsicPermeability();}poroPerm_.correctBoundaryConditions();' in perm
    )


def dependencies_are_refused(src: str) -> bool:
    """S21: the switch, its value refusal and the requirements that only
    other switches could break are gone with those switches; the refusal of
    the calibrated floor on top (a real parameter) is kept."""
    s = flat(src)
    return (
        'if(skeletonDiluteDragRate_>0){FatalErrorInFunction<<"dragOccupancyBasisintrinsicgives' in s
    )


def domain_check_reports_cells_first(src: str) -> bool:
    """The cell loop and its FatalError precede the boundary loop, and the
    convexExcess law goes through the shared check."""
    b = flat(body(src, "phaseClotAxisymmetricFluid::checkNetworkArgument"))
    cell = b.find("if(!(aIn[celli]>0&&aIn[celli]<phiSMax_)){worst=celli;break;}")
    fatal = b.find("if(returnReduce(worst>=0,orOp<bool>()))")
    bnd = b.find("forAll(a.boundaryField(),patchi)")
    return (
        0 <= cell < fatal < bnd
        and 'checkNetworkArgument(phiS,"convexExcessnetworkpressure");' in flat(src)
    )


def diagnostic_is_logged(src: str) -> bool:
    s = flat(src)
    return (
        all(t in src for t in (
            '"PHASE_CLOT_FRINGE time="', '" fringeOverBodySlipRms="',
            '" fringeDragOverIntrinsic="', '"PHASE_CLOT_S12_CONFIG"',
            '" fringeSlipMax="'))
        and "reportSkeletonVelocity();reportFringe();" in s
    )


def assessor_reads_the_gates(assess: str) -> bool:
    return ("def s12_report" in assess and '"s12": s12_report(text)' in assess
            and "<= 8.0" in assess and "1e-9" in assess)


PREDICATES = [
    ("intrinsic permeability", permeability_is_the_case_law_at_the_intrinsic_density),
    ("occupancy x intrinsic drag", drag_is_occupancy_times_intrinsic),
    ("both sites + written field", both_drag_sites_and_the_written_field_use_it),
    ("dependencies refused", dependencies_are_refused),
    ("domain check cells first", domain_check_reports_cells_first),
    ("diagnostic logged", diagnostic_is_logged),
]

MUTATIONS = [
    ("permeability floored", permeability_is_the_case_law_at_the_intrinsic_density,
     'k0pField_*pow3(1.0 - a)/sqr(a)/dimensionedScalar("ref", dimless, ref)',
     'max(k0pField_*pow3(1.0 - a)/sqr(a)/dimensionedScalar("ref", dimless, ref), dimensionedScalar("kf", dimArea, permeabilityFloor_))'),
    ("permeability at the cell average", permeability_is_the_case_law_at_the_intrinsic_density,
     '    const volScalarField a(networkPressureArgument());\n    checkNetworkArgument(a, "intrinsic permeability");',
     '    const volScalarField a(solidFraction());\n    checkNetworkArgument(a, "intrinsic permeability");'),
    ("occupancy floored like a support", drag_is_occupancy_times_intrinsic,
     "phaseIndicator()*sqr(1.0 - a)*nu_/intrinsicPermeability()",
     'max(phaseIndicator(), dimensionedScalar("f", dimless, 1e-4))*sqr(1.0 - a)*nu_/intrinsicPermeability()'),
    ("poroFactor at the cell average", drag_is_occupancy_times_intrinsic,
     "phaseIndicator()*sqr(1.0 - a)*nu_/intrinsicPermeability()",
     "phaseIndicator()*sqr(phiFluid_)*nu_/intrinsicPermeability()"),
    ("kinematic site not overwritten", both_drag_sites_and_the_written_field_use_it,
     "            dragCoeff = intrinsicDragRate();\n            poroPerm_ = intrinsicPermeability();",
     "            poroPerm_ = intrinsicPermeability();"),
    ("written poroPerm left legacy", both_drag_sites_and_the_written_field_use_it,
     "    {\n        poroPerm_ = intrinsicPermeability();\n    }\n\n    poroPerm_.correctBoundaryConditions();",
     "    if (false)\n    {\n        poroPerm_ = intrinsicPermeability();\n    }\n\n    poroPerm_.correctBoundaryConditions();"),
    ("floor on top allowed", dependencies_are_refused,
     "        if (skeletonDiluteDragRate_ > 0)\n        {\n            FatalErrorInFunction\n                << \"dragOccupancyBasis intrinsic gives",
     "        if (skeletonDiluteDragRate_ > 1e30)\n        {\n            FatalErrorInFunction\n                << \"dragOccupancyBasis intrinsic gives"),
    ("cell check disabled", domain_check_reports_cells_first,
     "        if (!(aIn[celli] > 0 && aIn[celli] < phiSMax_))\n        {\n            worst = celli;",
     "        if (false)\n        {\n            worst = celli;"),
    ("fringe line never printed", diagnostic_is_logged,
     "        reportSkeletonVelocity();\n        reportFringe();",
     "        reportSkeletonVelocity();"),
]


def k_int(a: float, k0: float, nu: float = 3.3333e-6, phi0: float = 0.45) -> float:
    """The C++ expression, transcribed: (1-a)^2 nu / (k0 (1-a)^3/a^2 / ref)."""
    ref = (1 - phi0)**3/phi0**2
    k = k0*(1 - a)**3/a**2/ref
    return (1 - a)**2*nu/k


def k_legacy(chi: float, k0: float, k1: float, nu: float = 3.3333e-6,
             phi0: float = 0.45) -> float:
    """The legacy kinematic site: max(chi*(1-phiS)^2*nu/k_KC(phiS), k1*chi),
    phiS = chi*phi0, with the legacy cap k <= 1e4*k0."""
    phis = chi*phi0
    ref = (1 - phi0)**3/phi0**2
    k = min(k0*(1 - phis)**3/phis**2/ref, 1e4*k0)
    return max(chi*(1 - phis)**2*nu/k, k1*chi)


class S12Properties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SOLVER.read_text()
        cls.hdr = HEADER.read_text()
        cls.gen = GEN.read_text()
        cls.assess = ASSESS.read_text()

    def test_every_property_holds(self):
        for name, pred in PREDICATES:
            with self.subTest(property=name):
                self.assertTrue(pred(self.src), f"S12 property broken: {name}")

    def test_the_assessor_reads_the_gates(self):
        self.assertTrue(assessor_reads_the_gates(self.assess))

    def test_the_switch_is_declared_with_its_evidence(self):
        # S21: `word dragOccupancyBasis_;` is gone (S21C1 hunk h26); the
        # evidence and the four functions stay.
        self.assertIn("20260923-085741", self.hdr)
        for f in ("checkNetworkArgument", "intrinsicPermeability",
                  "intrinsicDragRate", "reportFringe"):
            self.assertIn(f, self.hdr)

    def test_reference_values_are_the_priced_ones(self):
        """1.008e7 (shell) and 1.008e9 (core) at the reference -- the plateaus
        CLAUDE.md quotes for clotDrag -- and exactly (1-phi0)^2 nu/k0."""
        self.assertAlmostEqual(k_int(0.45, 1e-13)/1.00833e7, 1.0, places=4)
        self.assertAlmostEqual(k_int(0.45, 1e-15)/1.00833e9, 1.0, places=4)

    def test_it_is_the_legacy_law_in_the_body(self):
        """A FULLY DEPOSITED cell (thetaNetRef = phi0, so a = theta and
        ind = theta/phi0 = chi), compacted or dilated: the intrinsic basis IS
        the legacy expression wherever the legacy floor does not bind -- the
        body is unchanged. (Written first without the ind factor, this test
        read 1/chi = 1.5 at theta 0.30: the identity needs ind*K_int(a).)"""
        for theta in (0.30, 0.40, 0.45, 0.50, 0.58):
            ind = theta/0.45
            for k0 in (1e-13, 1e-15):
                self.assertAlmostEqual(
                    ind*k_int(theta, k0)/k_legacy(ind, k0, 1e6), 1.0,
                    places=12)

    def test_the_fringe_gains_the_priced_factor(self):
        """A fringe cell at the reference density, where the legacy FLOOR
        binds: intrinsic/legacy is the network's own drag over k1 -- 10.08 in
        the shell (the 10.1 priced cell by cell on run 20260923-085741's
        fields) and 1008 next to the core. Next to the core at ind 0.1 the
        legacy Kozeny-Carman (5.8e5) already exceeds the floor (1e5), so the
        factor there is the smaller K_int/K_KC, ~174 -- asserted too, so the
        claim is not quietly stretched to where it does not hold."""
        for ind in (1e-3, 1e-2, 0.1):
            self.assertAlmostEqual(
                ind*k_int(0.45, 1e-13)/k_legacy(ind, 1e-13, 1e6), 10.0833,
                places=3)
        for ind in (1e-3, 1e-2):
            self.assertAlmostEqual(
                ind*k_int(0.45, 1e-15)/k_legacy(ind, 1e-15, 1e6)/1008.33, 1.0,
                places=3)
        f = 0.1*k_int(0.45, 1e-15)/k_legacy(0.1, 1e-15, 1e6)
        self.assertTrue(150 < f < 200, f)

    def test_the_intrinsic_drag_rises_with_compaction(self):
        """d/da [a^2/(1-a)] > 0 on (0, phiSMax): a compacted network resists
        flow more, a dilated one less -- no inversion anywhere in the domain
        convexExcess allows."""
        prev = 0.0
        for i in range(1, 600):
            a = 0.6*i/600
            v = k_int(a, 1e-13)
            self.assertGreater(v, prev)
            prev = v


class S12Mutations(unittest.TestCase):
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
