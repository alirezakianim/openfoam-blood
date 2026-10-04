#!/usr/bin/env python3
"""Guard for S6c-2 -- `c` is DELETED, and the deletion is complete.

WHAT THIS FILE USED TO BE. Until 2026-09-20 it guarded S6c-1, the
`phaseNormalBasis` key: that both normals were built every step, that the
legacy one was textually verbatim, and that `deltaC_`/`phaseGradient_` had
NOT followed the key. Batch 42 ran that experiment and returned
`dAdhesionSpeedMax = 0` at all 272 steps, which authorised the deletion --
so every one of those assertions is now a statement about code that no
longer exists. A guard that pins a retired implementation does not protect
it; it blocks the change it was written to enable.

The subject is therefore inverted. This file now asserts THE REMOVAL IS
REAL, which is the property S6c-2 can actually get wrong.

WHY A PRESENCE CHECK IS NOT ENOUGH, AND WHY THE FACTOR OF TWO IS STILL HERE.
`c = 1 - 2*ind` with `ind = (thetaC + thetaH)/theta_ref`, so

    grad(c) = -2*grad(ind)      EXACTLY

and the 2 is load-bearing rather than cosmetic. `normal_` is a unit vector
only where the gradient clears `gradFloor_`; below the floor it is
`grad/gradFloor_` and its MAGNITUDE carries the factor. `gradFloor_` is
666.667 1/m on the main case, which is not small. A dropped 2 halves
`normal_` in every sub-floor cell while leaving its direction exactly right,
so no direction-only check can see it. Deleting c did not retire that trap:
the surviving expression still has to carry both the 2 and the floor.

AND `phaseGradient_` SURVIVED, WHICH THE SCOPING DOC DID NOT EXPECT. Its
site table gave it two consumers, `frontSource` and `chiSource`, both on
key values S6c-2 refuses, and concluded the field was dead. There is a
third: `clotGrowthFront = domainIntegrate(Vn_*phaseGradient_)`, read by
PHASE_CLOT_GROWTH_STOICH every step -- the 1.6277-against-1.6304 identity
every archived run is compared on. It is rebuilt on theta and it must stay
a factor of two apart from `interfaceDelta_`.

THE OTHER HALF OF THE JOB IS THE REFUSALS. Deleting c left five dictionary
values with no implementation. Silently treating a dead value as its
surviving sibling would run something other than what the case file says,
which is the failure class this project keeps being caught by. Each must
FAIL CLOSED.

Every check below has a negative test in
test_the_checks_fire_on_their_mutations: a guard that cannot fail is not a
guard, and this project has shipped four of those.

Full rationale: research/designs/2026-09-20-S6c-2-delete-c.md
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"
HEADER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.H"

# The five members deleted in S6c-2, plus c itself.
DELETED = ("c_", "muC_", "mobility_", "epsilonC_", "deltaC_")

# The dictionary values that lost their implementation with c.
DEAD_VALUES = (
    ("phaseNormalBasis", "legacy"),
    ("phaseDeltaBasis", "legacy"),
    ("phaseLimiterBasis", "legacy"),
    ("phaseFluidFractionBasis", "legacy"),
    ("phaseGrowthDriver", "front"),
)


def strip_comments(text: str) -> str:
    """C++ source with // and /* */ removed.

    The distinction is the whole point of this file: `c_` in a comment is
    history and `c_` in an expression is a compile error waiting to happen,
    and a bare grep cannot tell them apart. String literals are left alone
    -- the emitted config lines are checked separately, on the raw text.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def strip_literals(text: str) -> str:
    """Blank out string literals, preserving line structure.

    Needed because the REFUSAL MESSAGES name the deleted fields in prose:
    `"interfaceDelta_ was deltaC_ = 0.5*|grad c|."` is a diagnostic string,
    not a reference to a field. Without this the deletion check reports the
    very messages that explain the deletion. Found by this file failing on
    its own first run, which is what it is for.

    Only live_uses strips literals; every other helper here matches code
    that legitimately CONTAINS literals, such as the refusal table.
    """
    return re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', text)


def live_uses(src: str, name: str) -> list[str]:
    """Every line using `name` as a whole identifier in real code.

    Comments AND string literals are removed first: `c_` in prose is
    history and `c_` in an expression is a field that no longer exists,
    and a bare grep cannot tell the three apart.
    """
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
    body = strip_literals(strip_comments(src))
    return [ln for ln in body.splitlines() if pattern.search(ln)]


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def content_form_carries_the_factor_of_two(src: str) -> bool:
    body = compact(strip_comments(src))
    return "const volVectorField gradNContent ( -2.0 *fvc::grad" in body


def content_form_divides_by_the_floor(src: str) -> bool:
    body = compact(strip_comments(src))
    return "normal_ = gradNContent/max(magGradN, gradFloor_);" in body


def phase_gradient_is_rebuilt_on_theta(src: str) -> bool:
    body = compact(strip_comments(src))
    return (
        "phaseGradient_ = sqrt(sqr(magGradN) + sqr(gradFloor_)) - gradFloor_;"
        in body
    )


def the_two_localisers_stay_a_factor_of_two_apart(src: str) -> bool:
    """interfaceDelta_ is delta_Gamma; phaseGradient_ is 2*delta_Gamma.

    Assigning one from the other, or building both from the same
    expression, is the substitution the header has warned about since S3.
    """
    body = compact(strip_comments(src))
    if "interfaceDelta_ = deltaTheta_;" not in body:
        return False
    return "interfaceDelta_ = phaseGradient_" not in body


def dead_value_is_refused(src: str, key: str, value: str) -> bool:
    body = compact(strip_comments(src))
    return f'refuseDeadValue ( "{key}", {key}_, "{value}",' in body


def nonzero_mobility_is_refused(src: str) -> bool:
    """...and refused by a read that can actually reach the value.

    THE FIRST VERSION OF THIS PREDICATE PINNED A FATAL. It required
    `readScalar(phaseClotProperties_.lookup("mobility"))`, which is how the
    guard was written, and the guard was written wrong: every generator
    emits `mobility [0 4 -1 0 0 0 0] 0;`, and readScalar() aborts on the '['
    before deciding anything. Run 20260920-190025 died there, at
    construction, having reached no timestep. The predicate passed on a
    refusal that could never refuse, because a text match cannot see that
    the text is unreachable.

    So it now pins the DIMENSIONED read -- dimensioned<T>::initialise()
    takes an optional name, optional units and then the value, so it parses
    both spellings -- and asserts the fatal form is absent.
    scripts/check_dictionary_read_forms.py enforces the same rule against
    every key, derived from the templates rather than named here.
    """
    body = compact(strip_comments(src))
    return (
        'phaseClotProperties_.found("mobility")' in body
        and 'const dimensionedScalar mobilitySet ( '
            'phaseClotProperties_.lookup("mobility") );' in body
        and "mag(mobilitySet.value()) > vSmall" in body
        and 'readScalar(phaseClotProperties_.lookup("mobility"))' not in body
    )


def detachment_is_theta_native_and_booked(src: str) -> bool:
    body = compact(strip_comments(src))
    return (
        "thetaC_[celli] = 0.0; thetaH_[celli] = 0.0;" in body
        and "releasedContent += (thetaC_[celli] + thetaH_[celli])*mesh.V()[celli];"
        in body
        and "detachedContentStep_ += totalContent;" in body
    )


def content_balance_present(src: str) -> bool:
    required = (
        "contentVolume=", "contentDVdt=", "contentOuterFlux=",
        "contentTransferRate=", "contentRemoved=", "detachedContentRate=",
        "contentResidual=",
    )
    if 'Info<< "PHASE_CLOT_CONTENT_BALANCE"' not in src:
        return False
    return all(f'" {k}"' in src for k in required)


def balance_uses_the_operators_own_scheme(src: str) -> bool:
    """The content equation assembles with upwind; the flux must match.

    Reusing phaseWeights_ here would measure the mismatch between two
    schemes rather than the conservation error -- the warning inherited
    verbatim from the c balance this replaced.
    """
    body = compact(strip_comments(src))
    return (
        "upwind<scalar>(mesh_, contentCarrier).interpolate(thetaTotal)" in body
    )


def bounds_reported_in_q(src: str) -> bool:
    return (
        '<< " qMin=" << qMin' in src
        and '<< " qUnder=" << max(scalar(0), -qMin)' in src
        and '<< " cMin="' not in src
    )


STALE_DEFERRALS = ("stays-on-c", "stay-on-c", "until-S4", "until-S5",
                   "until-S6c-2")


class DeleteC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SOLVER.read_text()
        cls.hdr = HEADER.read_text()

    # ---- the deletion itself -------------------------------------------

    def test_the_deleted_members_appear_in_no_live_code(self):
        for name in DELETED:
            for label, text in (("C", self.src), ("H", self.hdr)):
                uses = live_uses(text, name)
                self.assertEqual(
                    uses, [],
                    f"{name} still appears in live {label} code, so the "
                    f"deletion is incomplete and the field it names does "
                    f"not exist: {uses[:3]}",
                )

    def test_the_comments_that_remain_are_only_comments(self):
        """c_ in prose is history and is allowed; this pins that reading.

        Without it, a later reader could 'clean up' the explanatory
        comments and destroy the record of why the field went.
        """
        self.assertIn("c_", self.src,
                      "every trace of c is gone including the explanation of "
                      "why, which makes the deletion unreviewable")

    # ---- what the surviving expression must still carry -----------------

    def test_the_factor_of_two_is_present(self):
        self.assertTrue(
            content_form_carries_the_factor_of_two(self.src),
            "the content normal lost its -2.0. grad(c) = -2*grad(ind) "
            "exactly, and below gradFloor_ the magnitude carries the "
            "factor, so a dropped 2 halves normal_ in every sub-floor cell "
            "with the direction still exactly right",
        )

    def test_the_normal_still_divides_by_the_floor(self):
        self.assertTrue(
            content_form_divides_by_the_floor(self.src),
            "the content normal no longer carries gradFloor_ in its "
            "denominator, so it is a bare unit vector and the factor of "
            "two above has nothing to act on",
        )

    def test_phase_gradient_survived_and_is_built_on_theta(self):
        self.assertTrue(
            phase_gradient_is_rebuilt_on_theta(self.src),
            "phaseGradient_ is gone or is not built from magGradN. The "
            "scoping doc said it was dead; clotGrowthFront reads it every "
            "step for PHASE_CLOT_GROWTH_STOICH, the 2*alpha/D identity "
            "every archived run is compared on",
        )

    def test_the_two_localisers_stay_a_factor_of_two_apart(self):
        self.assertTrue(
            the_two_localisers_stay_a_factor_of_two_apart(self.src),
            "interfaceDelta_ and phaseGradient_ have been conflated. They "
            "are delta_Gamma and 2*delta_Gamma; substituting one silently "
            "halves or doubles the growth rate and breaks no dimension "
            "check",
        )

    def test_clot_growth_front_still_reads_it(self):
        self.assertIn(
            "fvc::domainIntegrate(Vn_*phaseGradient_).value()",
            compact(strip_comments(self.src)),
            "clotGrowthFront is gone, so PHASE_CLOT_GROWTH_STOICH has no "
            "front integral and the stoichiometry identity is retired in "
            "silence",
        )

    # ---- fail closed on every value that lost its implementation --------

    def test_every_dead_key_value_is_refused(self):
        for key, value in DEAD_VALUES:
            self.assertTrue(
                dead_value_is_refused(self.src, key, value),
                f"{key} '{value}' is not refused. Its implementation read "
                f"c and is deleted, so a case naming it would silently run "
                f"the surviving value instead -- a different model under "
                f"the name of the one that was asked for",
            )

    def test_a_nonzero_mobility_is_refused(self):
        self.assertTrue(
            nonzero_mobility_is_refused(self.src),
            "a nonzero mobility is accepted. The Cahn-Hilliard term it "
            "switches on was deleted with muC_, so it can no longer do "
            "anything, and ignoring it would be a model change reported "
            "as a no-op",
        )

    # ---- the ported detachment, and the balance that must see it --------

    def test_the_detachment_is_theta_native_and_booked(self):
        self.assertTrue(
            detachment_is_theta_native_and_booked(self.src),
            "releaseDetachedMaterial() no longer zeroes theta, or no "
            "longer books what it removed. It is a CONTENT DELETION: "
            "unbooked, contentResidual reports it as a conservation "
            "failure rather than as the modelled release it is",
        )

    def test_the_content_balance_replaced_the_c_one(self):
        self.assertTrue(
            content_balance_present(self.src),
            "PHASE_CLOT_CONTENT_BALANCE is missing a term. It replaced "
            "PHASE_CLOT_MASS, which was this solver's ONLY closing mass "
            "balance -- without it clotVolume cannot distinguish material "
            "leaving the domain from material being destroyed",
        )

    def test_the_balance_uses_the_operators_own_face_scheme(self):
        self.assertTrue(
            balance_uses_the_operators_own_scheme(self.src),
            "the content flux is not interpolated with upwind against the "
            "stored carrier. The theta equations assemble with "
            "upwindConv, so any other scheme makes the residual measure a "
            "scheme mismatch rather than a conservation error",
        )

    def test_the_bounds_moved_to_q(self):
        self.assertTrue(
            bounds_reported_in_q(self.src),
            "PHASE_CLOT_PHASE_BOUNDS still reports c. q in [0,1] and "
            "c in [-1,1] are the same bound under c = 2q - 1, and the "
            "assessor converts on read -- but only if the solver emits q",
        )

    # ---- no deferral may outlive the stage it names ---------------------

    def test_no_stale_deferral_string_survives(self):
        """OI-17, applied to this stage rather than deferred again.

        `phaseNormal(stays-on-c-until-S5)` was printed at every start-up
        for a stage that had already run without taking it. It was at the
        head of every log read for three days and nothing read it back.
        """
        for token in STALE_DEFERRALS:
            self.assertNotIn(
                token, self.src,
                f"the solver still prints '{token}' at start-up. S6c-2 is "
                f"the stage those deferrals named, and it has run",
            )

    # ---- the negative tests --------------------------------------------

    def test_the_checks_fire_on_their_mutations(self):
        src = self.src

        broken = src.replace("-2.0\n       *fvc::grad", "-1.0\n       *fvc::grad", 1)
        self.assertNotEqual(broken, src, "the factor-of-two mutation did not apply")
        self.assertFalse(
            content_form_carries_the_factor_of_two(broken),
            "dropping the 2 from the content normal is NOT caught",
        )

        broken = src.replace(
            "normal_ = gradNContent/max(magGradN, gradFloor_);",
            "normal_ = gradNContent/magGradN;", 1)
        self.assertNotEqual(broken, src, "the floor mutation did not apply")
        self.assertFalse(
            content_form_divides_by_the_floor(broken),
            "removing gradFloor_ from the normal's denominator is NOT caught",
        )

        broken = src.replace(
            "phaseGradient_ = sqrt(sqr(magGradN) + sqr(gradFloor_)) - gradFloor_;",
            "phaseGradient_ = magGradN;", 1)
        self.assertNotEqual(broken, src, "the phaseGradient mutation did not apply")
        self.assertFalse(
            phase_gradient_is_rebuilt_on_theta(broken),
            "rebuilding phaseGradient_ as a bare gradient is NOT caught",
        )

        broken = src.replace("interfaceDelta_ = deltaTheta_;",
                             "interfaceDelta_ = phaseGradient_;", 1)
        self.assertNotEqual(broken, src, "the localiser mutation did not apply")
        self.assertFalse(
            the_two_localisers_stay_a_factor_of_two_apart(broken),
            "conflating the two localisers is NOT caught -- this is the "
            "silent factor of two",
        )

        for key, value in DEAD_VALUES:
            broken = src.replace(f'"{key}", {key}_, "{value}",', "", 1)
            self.assertNotEqual(broken, src,
                                f"the {key} refusal mutation did not apply")
            self.assertFalse(
                dead_value_is_refused(broken, key, value),
                f"removing the {key} '{value}' refusal is NOT caught",
            )

        broken = src.replace('phaseClotProperties_.found("mobility")', "false", 1)
        self.assertNotEqual(broken, src, "the mobility mutation did not apply")
        self.assertFalse(
            nonzero_mobility_is_refused(broken),
            "removing the nonzero-mobility refusal is NOT caught",
        )

        # REGRESSING TO THE FATAL FORM MUST ALSO BE CAUGHT. This is the
        # mutation the old predicate would have PASSED -- it was the old
        # predicate's required text.
        broken = src.replace(
            "const dimensionedScalar mobilitySet\n            (\n"
            '                phaseClotProperties_.lookup("mobility")\n'
            "            );",
            'const scalar mobilitySet =\n'
            '                readScalar(phaseClotProperties_.lookup("mobility"));',
            1,
        )
        self.assertNotEqual(
            broken, src, "the readScalar-regression mutation did not apply")
        self.assertFalse(
            nonzero_mobility_is_refused(broken),
            "reading `mobility` with readScalar() is NOT caught. Every "
            "generator writes it dimensioned; readScalar dies on the '[' "
            "at construction, which is how run 20260920-190025 was lost",
        )

        broken = src.replace("detachedContentStep_ += totalContent;", "", 1)
        self.assertNotEqual(broken, src, "the detachment mutation did not apply")
        self.assertFalse(
            detachment_is_theta_native_and_booked(broken),
            "dropping the content booking from the detachment is NOT "
            "caught, and an unbooked deletion reads as a conservation "
            "failure",
        )

        broken = src.replace('<< " contentResidual=" << contentResidual', "", 1)
        self.assertNotEqual(broken, src, "the balance mutation did not apply")
        self.assertFalse(
            content_balance_present(broken),
            "dropping contentResidual from the balance is NOT caught",
        )

        broken = src.replace(
            "upwind<scalar>(mesh_, contentCarrier).interpolate(thetaTotal)",
            "fvc::interpolate(thetaTotal)", 1)
        self.assertNotEqual(broken, src, "the scheme mutation did not apply")
        self.assertFalse(
            balance_uses_the_operators_own_scheme(broken),
            "swapping the balance's face scheme for a linear interpolate "
            "is NOT caught",
        )

        # And the deletion check itself must see a reintroduction.
        broken = src.replace("    q_.correctBoundaryConditions();",
                             "    c_ = q_;\n    q_.correctBoundaryConditions();", 1)
        self.assertNotEqual(broken, src, "the reintroduction did not apply")
        self.assertNotEqual(
            live_uses(broken, "c_"), [],
            "reintroducing c_ as live code is NOT caught by live_uses",
        )

        # ... and must NOT fire on a comment mentioning it.
        commented = src.replace("    q_.correctBoundaryConditions();",
                                "    // c_ = q_;\n    q_.correctBoundaryConditions();", 1)
        self.assertNotEqual(commented, src, "the comment mutation did not apply")
        self.assertEqual(
            live_uses(commented, "c_"), [],
            "live_uses reports a COMMENT as live code, so it would block "
            "the explanatory comments this deletion depends on",
        )

        # ... nor a string literal. The refusal messages name the deleted
        # fields in prose, and this file failed on its own first run for
        # exactly that reason.
        literal = src.replace(
            "    q_.correctBoundaryConditions();",
            '    Info<< "c_ is gone" << endl;\n    q_.correctBoundaryConditions();',
            1)
        self.assertNotEqual(literal, src, "the literal mutation did not apply")
        self.assertEqual(
            live_uses(literal, "c_"), [],
            "live_uses reports a STRING LITERAL as live code, so the "
            "refusal messages that explain the deletion would fail it",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
