#!/usr/bin/env python3
"""The interface localiser must be wired the way the specification says.

Run 20260915-073208 measured what the uniform rate does: 86.5% of all
deposition landed in cells that are >99% blood, at a per-volume rate 1.82x the
rate in the clot core. The repair is not a new model -- AM2020-phase-clot-v1
eq. (P) already puts uptake on a surface delta measure, and the LEGACY platelet
sink already implements it. This guard checks that the Stage 2 path now does
the same thing, and that the two traps around it cannot be fallen into
silently.

THE TWO TRAPS, both of which break no dimension check and do not crash:

  1. interfaceDelta_ vs phaseGradient_. They differ by a deliberate factor of
     two (0.5*|grad c| against |grad c|). The platelet sink is a flux per unit
     area and carries the UNIT delta. Swapping them silently halves or doubles
     the deposition.

  2. A mode word that falls back. RETIRED by S21: plateletBindLocalisation is
     built in at interface (S21A b-hunks, c14), so there is no word to fall back.

Every assertion below is negative-tested against a mutation of the source it
reads: see MUTATIONS.

Run: python3 scripts/check_platelet_bind_localisation.py
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"
HEADER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.H"
CONTENT = ROOT / "user-applications/plateletContent/plateletContent.H"
GEN = ROOT / "cases/AM2020_planar_mural_detachment/scripts/create_cases.py"


def rate_block(text: str) -> str:
    """The construction of the rate field, and nothing else."""
    start = text.index("volScalarField bindRate")
    end = text.index("bindRate.correctBoundaryConditions();", start)
    return text[start:end]


# (name, predicate over the relevant source, mutation that must break it)
def transfer_takes_a_field(text: str) -> bool:
    sig = text[text.index("inline void acceptedTransfer"):]
    sig = sig[:sig.index(")")]
    return "const volScalarField& rate" in sig


def rate_uses_unit_delta(text: str) -> bool:
    block = rate_block(text)
    return "interfaceDelta_" in block and "phaseGradient_" not in block


def rate_is_dimensionally_checked(text: str) -> bool:
    return bool(re.search(
        r"rate\.dimensions\(\)\s*!=\s*dimless/dimTime", text))


# S21 (2026-10-05, docket 20261005-1621 report 15): mode_is_validated() pinned the plateletBindLocalisation value
# refusal, deleted by S21A c14 (release .C:14042-14054).


def reaction_gated_on_the_field(text: str) -> bool:
    return "bindRateMax > 0" in text and "plateletBindRate_ > 0" not in text[
        text.index("volScalarField bindRate"):]


class LocaliserTests(unittest.TestCase):
    def setUp(self):
        self.solver = SOLVER.read_text()
        self.content = CONTENT.read_text()
        self.gen = GEN.read_text()
        self.header = HEADER.read_text()

    # ---- the wiring ------------------------------------------------

    def test_accepted_transfer_takes_a_rate_field(self):
        self.assertTrue(transfer_takes_a_field(self.content))

    def test_the_portable_kernel_was_not_changed(self):
        """acceptedIncrement already took a per-call scalar, so every
        executable T4 test stays valid. If this ever becomes a field the
        kernel's own tests must be revisited."""
        kern = (ROOT / "user-applications/plateletContent/"
                "plateletTransferKernel.H").read_text()
        self.assertIn("Scalar N, Scalar theta, Scalar dt, Scalar lambda", kern)

    def test_the_rate_uses_the_unit_delta_not_the_phase_gradient(self):
        self.assertTrue(rate_uses_unit_delta(self.solver))

    def test_the_rate_is_dimensionally_checked_as_per_second(self):
        self.assertTrue(rate_is_dimensionally_checked(self.content))

    def test_the_reaction_is_gated_on_the_field_not_the_scalar(self):
        """Gating on plateletBindRate_ would switch the whole interface mode
        off whenever lambda was left at its default -- precisely the
        configuration an interface child wants."""
        self.assertTrue(reaction_gated_on_the_field(self.solver))

    def test_the_localisation_is_reported_every_step(self):
        for key in ("bindLocalisation=", "bindRateMax=", "bindRateClotFrac="):
            self.assertIn(key, self.solver, key)

    # S21 (2026-10-05, docket 20261005-1621 report 15): test_the_mode_word_is_validated_against_an_explicit_list
    # (S21A c14) and test_the_unlocalised_mode_announces_itself (the
    # uniform-mode banner, S21A a34, release .C:14090-14099) retired with the
    # uniform mode.

    # ---- the generator ---------------------------------------------

    def test_the_generator_validates_the_same_two_words(self):
        self.assertIn('platelet_bind_localisation must be', self.gen)
        self.assertIn('("uniform", "interface")', self.gen)

    def test_the_generator_writes_the_adhesion_speed_key(self):
        # S21: the generator no longer writes plateletBindLocalisation (built
        # in; release-4198bb17 create_cases.py L4052).
        self.assertIn('"plateletAdhesionSpeed"', self.gen)

    # S21: test_the_ceiling_applies_only_to_the_uniform_mode retired with the
    # ceiling (release-4198bb17 create_cases.py L2247, L311).

    def test_the_header_records_the_factor_of_two_contract(self):
        self.assertIn("UNIT surface delta", self.header)


# Every predicate above is only worth having if a mutation breaks it.
MUTATIONS = [
    ("transfer_takes_a_field", CONTENT,
     "const volScalarField& rate", "const scalar rate", transfer_takes_a_field),
    ("rate_uses_unit_delta", SOLVER,
     "interfaceDelta_\n           *(", "phaseGradient_\n           *(",
     rate_uses_unit_delta),
    ("rate_is_dimensionally_checked", CONTENT,
     "rate.dimensions() != dimless/dimTime",
     "rate.dimensions() != dimless", rate_is_dimensionally_checked),
    # S21: the mode_is_validated mutation went with its predicate (S21A c14).
]


class MutationTests(unittest.TestCase):
    def test_every_predicate_fails_on_its_mutation(self):
        for name, path, old, new, predicate in MUTATIONS:
            with self.subTest(predicate=name):
                text = path.read_text()
                self.assertIn(old, text,
                              f"{name}: the anchor it mutates is gone")
                self.assertTrue(predicate(text), f"{name}: passes clean")
                broken = text.replace(old, new, 1)
                self.assertNotEqual(broken, text, f"{name}: no-op mutation")
                self.assertFalse(
                    predicate(broken),
                    f"{name} survived its own mutation -- the check is vacuous")


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=1)
             .result.wasSuccessful() else 1)
