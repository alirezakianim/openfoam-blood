#!/usr/bin/env python3
"""Static wiring and exact-rational checks of the free-platelet carrier.

The coupled pressure path makes phi_ a MIXTURE flux, not either phase's
intrinsic flux.  The platelet operator must receive the fluid's own flux in
its declared convention.  These tests protect the bounded dispatch repair in
research/changes/2026-09-14-content-indicator-root-repair-prerequisites.md.

This is deliberately NOT a CFD test or a substitute for a Linux compile.  The
source check verifies the small dispatch block inside the actual function;
the independent arithmetic demonstrates the convention error under slip.
Neither establishes momentum/content porosity consistency, positivity, a
transient solution, or the broader content/indicator model.

Run: python3 scripts/check_platelet_fluid_carrier.py
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"
QUALIFIED = "Foam::solvers::phaseClotAxisymmetricFluid::solvePlateletContent"
# Match quoted strings before comments: a slash inside an error string is not
# the beginning of a comment. Keep newlines for readable failure locations.
LEXEME = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/', re.S)
QUOTED = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def strip_comments(source: str) -> str:
    return LEXEME.sub(
        lambda match: "\n" * match.group().count("\n")
        if match.group().startswith(("//", "/*")) else match.group(),
        source,
    )


def enclosed(source: str, opening: int, left: str, right: str) -> tuple[str, int]:
    if source[opening] != left:
        raise ValueError(f"expected {left!r}")
    depth, position = 1, opening + 1
    while position < len(source):
        quoted = QUOTED.match(source, position)
        if quoted:
            position = quoted.end()
            continue
        char = source[position]
        depth += (char == left) - (char == right)
        if depth == 0:
            return source[opening + 1:position], position + 1
        position += 1
    raise ValueError(f"unbalanced {left!r} block")


def function_source(source: str) -> tuple[str, str]:
    clean = strip_comments(source)
    matches = list(re.finditer(r"\bvoid\s+" + re.escape(QUALIFIED) + r"\s*\(", clean))
    if len(matches) != 1:
        raise ValueError(f"expected one target definition, found {len(matches)}")
    arguments, end = enclosed(clean, matches[0].end() - 1, "(", ")")
    opening = end + len(clean[end:]) - len(clean[end:].lstrip())
    body, _ = enclosed(clean, opening, "{", "}")
    return arguments, body


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


DISPATCH = compact('''
surfaceScalarField carrier("plateletCarrier", suppliedCarrier);
if (twoVelocitySkeleton_)
{
    carrier = phiFluidOwn_;
    if (plateletFluxConvention_ == "superficial")
    {
        carrier *= fvc::interpolate(epsP_);
    }
}
''')


# THE ONE SANCTIONED SECOND USE OF suppliedCarrier.
#
# The test-only evidence dump has to be handed the supplied carrier: the
# whole point of the archive is that a reader can see BOTH candidate
# fluxes and tell which one the operator consumed. Writing only the
# selection would make the substitution unverifiable.
#
# It is excised before the counts below rather than loosening them, so the
# remainder of the function is held to exactly the contract it had before:
# one suppliedCarrier, initialising the selected carrier and nothing else.
# Anything that is not this exact call still trips the count.
ARCHIVE_CALL = compact('''
if (plateletCarrierArchive_)
{
    archivePlateletCarrier(suppliedCarrier, carrier, Feps, phaseCorrector);
}
''')


def wiring_errors(source: str) -> list[str]:
    """Fail closed on changes to this intentionally small dispatch contract.

    This is a structural guard, not a general C++ semantic analyser. Rewriting
    the dispatch to a different equivalent implementation requires updating
    this test deliberately, together with its rejecting mutations.
    """
    try:
        arguments, body = function_source(source)
    except ValueError as exc:
        return [str(exc)]
    arguments, body = compact(arguments), compact(body)
    errors = []
    archives = body.count(ARCHIVE_CALL)
    if archives > 1:
        errors.append("the evidence archive must be called at most once")
    if "archivePlateletCarrier" in body and archives != 1:
        errors.append("archivePlateletCarrier appears in some other form;"
                      " the evidence dump is exempt from the counts below"
                      " only in its exact sanctioned shape")
    counted = body.replace(ARCHIVE_CALL, "", 1)
    if arguments != "constsurfaceScalarField&suppliedCarrier,constlabelphaseCorrector":
        errors.append("input must be the distinctly named suppliedCarrier")
    if body.count(DISPATCH) != 1:
        errors.append("missing or changed phase/convention dispatch")
    else:
        start = body.index(DISPATCH)
        if not body.startswith("if(!plateletContentModel_){return;}"):
            errors.append("disabled model must return before carrier selection")
        porosity = "epsP_.correctBoundaryConditions();"
        if porosity not in body[:start]:
            errors.append("dispatch must follow the current porosity update")
        tail = body[start + len(DISPATCH):]
        flux = "plateletContent::flux(plateletFluxConvention_,epsP_,carrier)"
        if flux not in tail or body.count("plateletContent::flux(") != 1:
            errors.append("shared helper must consume the selected carrier once")
    # There must be no second path quietly changing the one-velocity carrier,
    # or a second porosity multiplication elsewhere in this function.
    if len(re.findall(r"\bcarrier(?:\*=|/=|\+=|-=|=(?!=))", counted)) != 2:
        errors.append("unexpected carrier assignment or porosity multiplication")
    if len(re.findall(r"\bsuppliedCarrier\b", counted)) != 1:
        errors.append("supplied carrier must only initialize the selected carrier")
    if "phiSkelOwn_" in counted:
        errors.append("free platelets must not use the skeleton carrier")
    for diagnostic in (
        "forAll(carrier.boundaryField(),patchi)",
        "carrierBoundarySum+=gSum(carrier.boundaryField()[patchi]);",
        "carrierMin=gMin(carrier.primitiveField());",
        "carrierMax=gMax(carrier.primitiveField());",
        "carrierAbsSum=gSum(mag(carrier.primitiveField()));",
    ):
        if diagnostic not in body:
            errors.append("missing selected-carrier diagnostic: " + diagnostic)
    return errors


def selected_carrier(two_velocity: bool, convention: str, eps: Fraction,
                     supplied: Fraction, fluid: Fraction) -> Fraction:
    """Manufactured arithmetic counterpart; this does not execute C++."""
    if convention not in ("intrinsic", "superficial"):
        raise ValueError("unknown convention")
    if not two_velocity:
        return supplied
    return eps * fluid if convention == "superficial" else fluid


def volume_flux(convention: str, eps: Fraction, carrier: Fraction) -> Fraction:
    if convention not in ("intrinsic", "superficial"):
        raise ValueError("unknown convention")
    return eps * carrier if convention == "intrinsic" else carrier


class CarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text()

    def test_current_function_wiring(self) -> None:
        self.assertEqual(wiring_errors(self.source), [])

    def test_mutations_rejected_inside_target(self) -> None:
        arguments, body = function_source(self.source)
        target = "void " + QUALIFIED + "(" + arguments + "){" + body + "}"
        self.assertEqual(wiring_errors(target), [])
        mutations = (
            ("wrong phase", "carrier = phiFluidOwn_;", "carrier = phiSkelOwn_;"),
            ("mixture phase", "carrier = phiFluidOwn_;", "carrier = suppliedCarrier;"),
            ("missing preweight", "carrier *= fvc::interpolate(epsP_);", ""),
            ("double preweight", "carrier *= fvc::interpolate(epsP_);",
             "carrier *= fvc::interpolate(epsP_); carrier *= fvc::interpolate(epsP_);"),
            ("wrong convention", 'plateletFluxConvention_ == "superficial"',
             'plateletFluxConvention_ == "intrinsic"'),
            ("wrong phase condition", "if (twoVelocitySkeleton_)",
             "if (!twoVelocitySkeleton_)"),
            ("one velocity changed", '"plateletCarrier", suppliedCarrier',
             '"plateletCarrier", phiFluidOwn_'),
            ("helper bypasses dispatch", "epsP_, carrier)", "epsP_, suppliedCarrier)"),
            ("diagnostics use stale carrier", "gMin(carrier.primitiveField())",
             "gMin(suppliedCarrier.primitiveField())"),
            # The evidence archive is exempt from the suppliedCarrier and
            # phiSkelOwn_ counts. These five confirm the exemption is a hole
            # exactly the size of that one call and no larger: a reordered or
            # renamed archive call stops being exempt, and transport that
            # reaches for either field is still caught while the archive is
            # present and well-formed.
            ("archive arguments reordered",
             "archivePlateletCarrier(suppliedCarrier, carrier, Feps,"
             " phaseCorrector);",
             "archivePlateletCarrier(carrier, suppliedCarrier, Feps,"
             " phaseCorrector);"),
            ("archive call renamed",
             "archivePlateletCarrier(suppliedCarrier",
             "archivePlateletCarrierX(suppliedCarrier"),
            ("archive ungated",
             "if (plateletCarrierArchive_)\n    {\n"
             "        archivePlateletCarrier(",
             "if (true)\n    {\n        archivePlateletCarrier("),
            ("transport reaches for the skeleton flux despite the archive",
             "const surfaceScalarField& Feps = tFeps();",
             "const surfaceScalarField& Feps = tFeps();"
             " const surfaceScalarField sneak(phiSkelOwn_);"),
            ("transport reuses the supplied carrier despite the archive",
             "const surfaceScalarField& Feps = tFeps();",
             "const surfaceScalarField& Feps = tFeps();"
             " const surfaceScalarField sneak(suppliedCarrier);"),
        )
        for name, before, after in mutations:
            with self.subTest(mutation=name):
                self.assertEqual(target.count(before), 1, "mutation must actually land")
                bad = target.replace(before, after, 1)
                self.assertTrue(wiring_errors(bad), "mutation passed: " + name)

    def test_comments_and_other_functions_cannot_satisfy_contract(self) -> None:
        arguments, body = function_source(self.source)
        missing = "void " + QUALIFIED + "(" + arguments + "){return;}"
        decoys = "\n// " + DISPATCH + "\nvoid decoy(){" + body + "}\n"
        self.assertTrue(wiring_errors(missing + decoys))

    def test_missing_duplicate_and_malformed_definition_rejected(self) -> None:
        self.assertTrue(wiring_errors("// " + QUALIFIED))
        self.assertTrue(wiring_errors(self.source + self.source))
        self.assertTrue(wiring_errors("void " + QUALIFIED + "(int x){"))

    def test_two_conventions_same_correct_flux(self) -> None:
        eps, fluid, solid = Fraction(11, 20), Fraction(1), Fraction(1, 5)
        mixture = eps * fluid + (1 - eps) * solid
        self.assertEqual(mixture, Fraction(16, 25))
        for convention in ("intrinsic", "superficial"):
            carrier = selected_carrier(True, convention, eps, mixture, fluid)
            self.assertEqual(volume_flux(convention, eps, carrier), Fraction(11, 20))

    def test_old_mixture_input_is_discriminated(self) -> None:
        eps, mixture, correct = Fraction(11, 20), Fraction(16, 25), Fraction(11, 20)
        wrong_intrinsic = volume_flux("intrinsic", eps, mixture)
        wrong_superficial = volume_flux("superficial", eps, mixture)
        self.assertEqual(wrong_intrinsic, Fraction(44, 125))  # 0.352
        self.assertEqual(wrong_superficial, Fraction(16, 25))  # 0.64
        self.assertEqual((wrong_intrinsic - correct) / correct, Fraction(-9, 25))
        self.assertEqual((wrong_superficial - correct) / correct, Fraction(9, 55))

    def test_single_velocity_supplied_input_unchanged(self) -> None:
        eps, supplied, unrelated_fluid = Fraction(11, 20), Fraction(7, 13), Fraction(17)
        for convention in ("intrinsic", "superficial"):
            carrier = selected_carrier(False, convention, eps, supplied, unrelated_fluid)
            self.assertEqual(carrier, supplied)
            self.assertEqual(volume_flux(convention, eps, carrier),
                             volume_flux(convention, eps, supplied))

    def test_reverse_and_zero_flow(self) -> None:
        for eps in (Fraction(1), Fraction(11, 20), Fraction(1, 100)):
            for fluid in (Fraction(-3, 2), Fraction(0), Fraction(7, 3)):
                for convention in ("intrinsic", "superficial"):
                    carrier = selected_carrier(True, convention, eps, Fraction(99), fluid)
                    self.assertEqual(volume_flux(convention, eps, carrier), eps * fluid)


if __name__ == "__main__":
    print("STATIC WIRING + EXACT-RATIONAL ARITHMETIC; no CFD or compile claim.", flush=True)
    unittest.main(verbosity=2)
