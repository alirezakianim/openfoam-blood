#!/usr/bin/env python3
"""Guard for S6b -- the two keys that take the phase FACE WEIGHTS theta-native.

WHY THERE ARE TWO KEYS AND NOT ONE. The S6 scoping document specified a single
step, "phaseWeights_ off c -- vanLeer on ind", and gave it Gate A-exact on
|Wp - q|. Reading run 20260919-213132 showed that gate cannot be met by that
change, for a reason the design did not have in front of it:

  * Wp = w*q, so at saturation the content must EQUAL the fluid fraction. That
    survives discretisation only when Wp's convection operator is the operator
    that DEFINES q.
  * Before S6a, q = 0.5*(1 + c) and Wp rode c's weights, so it did.
  * After S6a, q = 1 - theta/theta_ref, and theta rides FIRST-ORDER UPWIND --
    fixed as a contract inside solveContentComponents(), field-independent on
    purpose -- while Wp still rides a limited scheme.

So the mismatch after S6a is in the OPERATOR, and changing which field the
limiter reads does not touch it. Measured, from that run's own logs
(PHASE_CLOT_PLATELET WpRawQDevMax at t = 4e-04 s):

    em39 / em40   q from c, same operator as Wp      1.643e-03
    em41          q from theta, Wp still limited     1.584e-02   (9.6x)

and 1.584e-02 is the c/theta divergence itself (dqMax 1.5479e-02): Wp is
tracking the field q no longer comes from.

  phaseLimiterBasis      -- which FIELD phaseFieldFaceWeights() limits on.
                            Frees the weights from c, so c can be deleted.
                            Consumers: the c equation, shell_, Wp.
  phasePlateletFluxBasis -- which FACE VALUES Wp is advected with. The OI-6
                            identity applied one equation further on.
                            Consumer: Wp alone.

AND GATE A-EXACT DOES NOT APPLY TO THIS CASE AT ALL. em39's two operators are
identical by construction, and it still reports 1.643e-03 -- that residual is
the adhesion sink and the explicit laplacian, not transport. A 1e-11 gate here
would be six decades under the instrument, which is the failure this project
has now made five times.

Run: python3 scripts/check_phase_weight_basis_split.py
"""

import importlib.util
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOLVER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"
HEADER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.H"


def compact(text: str) -> str:
    """Whitespace-insensitive, so reformatting the source is not a failure."""
    return re.sub(r"\s+", "", text)


def validation_block(src: str, key: str) -> str:
    """The body of `if (<key>_ == "content")`.

    Scoped per key on purpose: five basis keys carry textually identical
    `if (!contentModel_)` refusals, and a predicate that searched the whole
    file would pass on ANOTHER key's guard after this one was deleted. A
    guard that reads a different block's condition cannot fail.
    """
    m = re.search(r'if \(%s_ == "content"\)\s*\{(.*?)\n    \}' % key, src, re.S)
    return m.group(1) if m else ""


def limiter_weights_body(src: str) -> str:
    """S6c-2: the limiter field is no longer a conditional tmp.

    It was `tLimiterField = volScalarField::New(...)` built only on the
    content branch, because the legacy branch bound `c_` itself. With c
    deleted there is one branch and one plain field, so the pattern the
    old regex looked for is gone. Same construction, different shape.
    """
    m = re.search(
        r"const volScalarField limiterField\s*\((.*?)\n    \);",
        src,
        re.S,
    )
    return m.group(1) if m else ""


def legacy_limiter_binds_c_itself(src: str) -> bool:
    """S6c-2: there is no legacy branch, and c_ must not come back.

    The rule it enforced still matters and is now enforced the other way
    round. vanLeer forms r from fvc::grad(vf), and fvc::grad looks its
    scheme up by the field's NAME -- which is why the surviving field is
    still constructed as "phaseLimiterField" rather than left anonymous.
    What must not happen is c_ reappearing as the limiter argument, so
    that is what is asserted.
    """
    c = compact(src)
    return (
        "constvolScalarFieldlimiterField(" in c
        and '"phaseLimiterField",' in c
        and "tLimiterField()" not in c
        and "limiterField=contentLimiter?" not in c
        and "weights(c_)" not in c
    )


def platelet_convection_body(src: str) -> str:
    m = re.search(
        r"phaseClotAxisymmetricFluid::plateletContentConvection\s*\(.*?\)\s*"
        r"const\s*\{(.*?)\n\}",
        src,
        re.S,
    )
    return m.group(1) if m else ""


# ---- the keys themselves --------------------------------------------------

def key_default(src: str, key: str):
    """The key's lookupOrDefault fallback, or None if it has none.

    Anchored on lookupOrDefault so it cannot match the refuseDeadValue
    call site, which also spells the key beside a quoted word.
    """
    m = re.search(r'lookupOrDefault<word>\("%s","(\w*)"\)' % key,
                  compact(src))
    return m.group(1) if m else None


def key_is_declared(src: str, hdr: str, key: str) -> bool:
    return ("word %s_;" % key in hdr) and bool(key_default(src, key))


def default_is_not_a_refused_value(src: str, key: str, refused) -> bool:
    """THE PROPERTY THIS TEST USED TO PIN, AND WHY IT MOVED.

    Until 2026-09-20 this asserted the default was literally "legacy",
    on the reasoning that a flip "would silently change the physics of
    every archived child". S7 flipped it deliberately, and the test
    failed the repair instead of the repair failing the test -- the
    third time in this project a guard has pinned a live setting.

    The durable property is the one S6c-2 broke: a key must not DEFAULT
    to a value the solver REFUSES. That made omission fatal in 201 of
    203 cases, and every local gate passed. It can still fail, which
    "the default is content" cannot.
    """
    value = key_default(src, key)
    return value is not None and refused.get(key) != value


def basis_validated_against_explicit_list(src: str, key: str) -> bool:
    c = compact(src)
    return ('%s_!="legacy"' % key in c and '%s_!="content"' % key in c)


def content_requires(src: str, key: str, needle: str) -> bool:
    return needle in compact(validation_block(src, key))


# ---- S6b-1: the limiter field ---------------------------------------------

def limiter_reads_the_selected_field(src: str) -> bool:
    """weights() must be evaluated on limiterField, never on c_ directly."""
    c = compact(src)
    return ("tinterp().weights(limiterField)" in c
            and "tinterp().weights(c_)" not in c)


def limiter_field_is_the_unclamped_ratio(src: str) -> bool:
    body = compact(limiter_weights_body(src))
    if not body:
        return False
    return ('(thetaC_+thetaH_)/dimensionedScalar("thetaRef",dimless,'
            "skeletonVolumeFraction_)" in body
            and "max(" not in body
            and "min(" not in body
            and "phaseIndicator()" not in body)


def interface_bin_is_basis_aware_and_legacy_is_unchanged(src: str) -> bool:
    """|c| < 0.9 is -0.9 < c < 0.9; c = 2q - 1 and q = 1 - ind make the same
    bin 0.05 < ind < 0.95. The legacy numbers must be exactly the old ones."""
    c = compact(src)
    return ("constscalarinterfaceLo=contentLimiter?0.05:-0.9;" in c
            and "constscalarinterfaceHi=contentLimiter?0.95:0.9;" in c
            and "mag(cCellIn[limOwn[facei]])<0.9" not in c)


# ---- S6b-2: the platelet flux ---------------------------------------------

def legacy_path_is_called_through_not_reimplemented(src: str) -> bool:
    body = compact(platelet_convection_body(src))
    return ('if(phasePlateletFluxBasis_!="content"){return'
            "phaseConservativeConvection(carrier,f);}" in body)


def content_path_uses_the_content_weights(src: str) -> bool:
    body = compact(platelet_convection_body(src))
    return ("returnphaseConservativeConvectionWith(carrier,f,"
            "contentFaceWeights_());" in body)


def content_path_refuses_absent_weights(src: str) -> bool:
    body = compact(platelet_convection_body(src))
    return ("if(!contentFaceWeights_.valid()){FatalErrorInFunction" in body)


def only_Wp_takes_the_selected_weights(src: str) -> bool:
    """The key must move ONE equation. shell_ is a label and c and thetaC/H
    are transported by their own operators; a second call site would make this
    key two changes and its gate unattributable."""
    calls = re.findall(r"plateletContentConvection\(([^)]*)\)", src)
    calls = [a for a in calls if "const surfaceScalarField&" not in a]
    return len(calls) == 1 and "Wp_" in calls[0]


def one_implementation_of_the_operator(src: str) -> bool:
    """phaseConservativeConvection must delegate, not carry a second copy of
    the gaussConvectionScheme construction."""
    c = compact(src)
    # TWO, and the second is phaseSharedConvection's -- a DIFFERENT operator
    # (the bounded/material form for labels), legitimately its own. A third
    # is phaseConservativeConvection carrying a copy again.
    return (c.count("newsuppliedWeightsScheme(mesh_,weights)") == 1
            and c.count("newsuppliedWeightsScheme(mesh_,phaseWeights_())") == 1
            and c.count("newsuppliedWeightsScheme(") == 2)


# ---- the capture the whole thing rests on ---------------------------------

def content_face_weights_follow_the_matrix(src: str) -> bool:
    """contentFaceWeights_ must be the weights the theta matrix ACTUALLY used.

    S6c-2 INVERTED THIS CHECK, and the inversion is the point. The capture
    used to be a two-branch selection because contentOperator 'matched'
    assembled with phaseWeights_ while 'content' assembled with upwind; a
    capture that ignored the distinction handed the momentum flux and the
    Wp equation the face values of an operator neither equation used, and
    PHASE_CLOT_MMC could not see it because both of its terms read this
    same member.

    'matched' is refused since S6c-2, so there is exactly one assembling
    operator and the capture must be unconditionally upwind -- the very
    thing this check was written to forbid. Reintroducing the phaseWeights_
    branch now would capture weights no equation uses, so BOTH the
    conditional and the phaseWeights_ form are refused here.

    S10 CHANGED THE SPELLING, NOT THE PROPERTY. The operator became
    selectable (contentTransportScheme upwind | limitedLinear), so requiring
    the literal upwind capture would pin a setting and fail the repair. The
    property is that the capture and the theta matrix share ONE weights
    object: thetaC_ is assembled from tContentWeights() and the capture
    copies tContentWeights(). A capture that rebuilds its weights -- upwind
    again, or a second limiter evaluation -- fails, as does either refused
    form above.
    """
    c = compact(src)
    return (
        "phaseConservativeConvectionWith(carrier,thetaC_,tContentWeights())"
        in c
        and 'newsurfaceScalarField("contentFaceWeights",tContentWeights())'
        in c
        and "if(matched" not in c
        and 'newsurfaceScalarField("contentFaceWeights",phaseWeights_())'
        not in c
    )


def both_configs_are_emitted(src: str) -> bool:
    c = compact(src)
    return ('"PHASE_CLOT_PHASE_LIMITER_CONFIG"' in c
            and '"PHASE_CLOT_PLATELET_FLUX_CONFIG"' in c
            and '"limiterBasis="' in c)


class PhaseWeightBasisSplit(unittest.TestCase):
    """Every predicate paired with the mutation that must break it."""

    @classmethod
    def setUpClass(cls):
        cls.src = SOLVER.read_text()
        cls.hdr = HEADER.read_text()
        # ONE OWNER for "which values does the solver refuse".
        # check_refused_dictionary_values.py derives it from the
        # refuseDeadValue call sites with comments stripped first --
        # the messages quote the dead values verbatim -- and pins the
        # site count so the set cannot quietly narrow. Re-deriving it
        # here with a second regex is the two-place rule broken again.
        spec = importlib.util.spec_from_file_location(
            "_refused", pathlib.Path(__file__).resolve().parent
            / "check_refused_dictionary_values.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.refused = mod.solver_refusals(cls.src)[0]

    # ---- the keys ---------------------------------------------------------

    def test_both_keys_are_declared_with_a_default_the_solver_accepts(self):
        for key in ("phaseLimiterBasis", "phasePlateletFluxBasis"):
            self.assertTrue(
                key_is_declared(self.src, self.hdr, key),
                "%s is not declared, or has no default -- a key with no "
                "fallback is fatal on every case that omits it" % key)
            self.assertTrue(
                default_is_not_a_refused_value(self.src, key, self.refused),
                "%s DEFAULTS to the value the solver refuses (%r). Omitting "
                "the key is then fatal, and most cases omit it: that is the "
                "S6c-2 defect, 201 of 203 cases aborting at construction "
                "with every local gate green"
                % (key, self.refused.get(key)))
            self.assertTrue(
                basis_validated_against_explicit_list(self.src, key),
                "a misspelled %s would fall through silently" % key)

    def test_each_content_basis_is_refused_without_its_prerequisites(self):
        self.assertTrue(
            content_requires(self.src, "phaseLimiterBasis",
                             "if(!contentModel_)"),
            "phaseLimiterBasis 'content' no longer requires contentModel: "
            "the limiter would read a field frozen at initialisation, which "
            "is central differencing on every interior face")
        self.assertTrue(
            content_requires(self.src, "phaseLimiterBasis",
                             "if(skeletonVolumeFraction_<=small)"),
            "the theta_ref divisor is unguarded in the limiter block")
        self.assertTrue(
            content_requires(self.src, "phaseLimiterBasis",
                             'phaseFluidFractionBasis_!="content"'),
            "phaseLimiterBasis 'content' no longer requires q to come from "
            "theta -- theta's weights against a c-defined q is a NEW "
            "mismatch, not a repair")

        self.assertTrue(
            content_requires(self.src, "phasePlateletFluxBasis",
                             "if(!contentModel_)"),
            "phasePlateletFluxBasis 'content' no longer requires "
            "contentModel: contentFaceWeights_ would never be set")
        self.assertTrue(
            content_requires(self.src, "phasePlateletFluxBasis",
                             "if(!plateletContentFormulation_)"),
            "phasePlateletFluxBasis 'content' no longer requires the content "
            "formulation: Wp is not solved without it and the key is inert")
        self.assertTrue(
            content_requires(self.src, "phasePlateletFluxBasis",
                             'phaseFluidFractionBasis_!="content"'),
            "phasePlateletFluxBasis 'content' no longer requires q to come "
            "from theta -- the identity being repaired is Wp == q")

    # ---- S6b-1 ------------------------------------------------------------

    def test_the_limiter_reads_the_selected_unclamped_field(self):
        self.assertTrue(
            limiter_reads_the_selected_field(self.src),
            "the weights are still built from c_ directly; the key is inert")
        self.assertTrue(
            limiter_field_is_the_unclamped_ratio(self.src),
            "the content limiter field is not the unclamped theta ratio. "
            "phaseIndicator()'s max(.., 0) protects an fvm::Sp on the "
            "momentum diagonal; a limiter argument is not a diagonal, and "
            "the clamp costs the exactness of the affine-invariance argument")
        self.assertTrue(
            interface_bin_is_basis_aware_and_legacy_is_unchanged(self.src),
            "the interface counter is not the same bin in both spellings")
        self.assertTrue(
            legacy_limiter_binds_c_itself(self.src),
            "the legacy branch no longer binds c_ itself. A renamed copy "
            "changes the fvc::grad scheme-lookup key inside the limiter, so "
            "the control is no longer guaranteed bit-identical")

    # ---- S6b-2 ------------------------------------------------------------

    def test_the_platelet_flux_selects_and_only_for_Wp(self):
        self.assertTrue(
            legacy_path_is_called_through_not_reimplemented(self.src),
            "the legacy platelet path is no longer the previous call reached "
            "through the same function; it can now drift from it")
        self.assertTrue(
            content_path_uses_the_content_weights(self.src),
            "the content path does not use contentFaceWeights_ -- the whole "
            "point is that Wp inherits the face values theta transported with")
        self.assertTrue(
            content_path_refuses_absent_weights(self.src),
            "the content path no longer refuses absent weights; Wp would be "
            "advected with last step's, silently")
        self.assertTrue(
            only_Wp_takes_the_selected_weights(self.src),
            "a second call site appeared. shell_ is a LABEL carried on "
            "pathlines and has no Wp = q identity to preserve; moving it in "
            "the same key would smear the core-shell boundary, which carries "
            "two decades of permeability contrast, with nothing to attribute "
            "the change to")
        self.assertTrue(
            one_implementation_of_the_operator(self.src),
            "the conservative operator has been copied rather than "
            "delegated; two copies differing only in which member they read "
            "is the desynchronisation phaseWeights_ exists to prevent")

    # ---- the capture ------------------------------------------------------

    def test_the_captured_weights_are_the_ones_the_matrix_used(self):
        self.assertTrue(
            content_face_weights_follow_the_matrix(self.src),
            "contentFaceWeights_ is unconditionally upwind again. Under "
            "contentOperator 'matched' the theta matrix assembles with "
            "phaseWeights_, so the capture would hand the momentum flux AND "
            "Wp the face values of an operator neither equation used -- and "
            "PHASE_CLOT_MMC reads zero regardless, because both of its terms "
            "come from this member")

    def test_both_keys_announce_themselves_in_the_log(self):
        self.assertTrue(
            both_configs_are_emitted(self.src),
            "a config emit is missing. The em38 packaging failure was only "
            "caught because the new diagnostics were countable in the log")

    # ---- the mutations ----------------------------------------------------

    def test_the_checks_fire_on_their_mutations(self):
        for key in ("phaseLimiterBasis", "phasePlateletFluxBasis"):
            current = key_default(self.src, key)
            self.assertTrue(current, "%s has no default to mutate" % key)

            # 1. The default removed entirely.
            stripped = self.src.replace('"%s", "%s"' % (key, current),
                                        '"%s", ""' % key, 1)
            self.assertNotEqual(stripped, self.src, "nothing was mutated")
            self.assertFalse(key_is_declared(stripped, self.hdr, key),
                             "emptying %s's default was not detected" % key)

            # 2. The default moved ONTO a refused value -- the S6c-2
            #    defect itself. Only reachable for a key that HAS a
            #    refused value; phasePlateletFluxBasis has none, and
            #    skipping silently is how a mutation test goes vacuous,
            #    so assert which case we are in.
            dead = self.refused.get(key)
            if dead is None:
                self.assertNotIn(
                    key, self.refused,
                    "%s gained a refused value; the mutation below must "
                    "now run for it too" % key)
            else:
                onto = self.src.replace('"%s", "%s"' % (key, current),
                                        '"%s", "%s"' % (key, dead), 1)
                self.assertNotEqual(onto, self.src, "nothing was mutated")
                self.assertFalse(
                    default_is_not_a_refused_value(onto, key, self.refused),
                    "%s defaulting to the refused %r was not detected"
                    % (key, dead))

            undeclared = self.hdr.replace("word %s_;" % key, "", 1)
            self.assertNotEqual(undeclared, self.hdr, "nothing was mutated")
            self.assertFalse(key_is_declared(self.src, undeclared, key),
                             "removing %s's declaration was not detected" % key)

            block = validation_block(self.src, key)
            self.assertTrue(block, "%s's content block was not found" % key)
            unguarded = self.src.replace(
                block, block.replace("if (!contentModel_)", "if (false)", 1), 1)
            self.assertNotEqual(unguarded, self.src, "nothing was mutated")
            self.assertFalse(
                content_requires(unguarded, key, "if(!contentModel_)"),
                "dropping %s's contentModel refusal was not detected -- and "
                "the identical refusal in the other four basis blocks is "
                "still present, which is what the scoping is for" % key)

        # S6b-1
        inert = self.src.replace("tinterp().weights(limiterField)",
                                 "tinterp().weights(c_)", 1)
        self.assertNotEqual(inert, self.src, "nothing was mutated")
        self.assertFalse(limiter_reads_the_selected_field(inert),
                         "reverting the weights to c_ was not detected")

        body = limiter_weights_body(self.src)
        self.assertTrue(body, "the limiter field construction was not found")
        clamped = self.src.replace(
            body, body.replace("(thetaC_ + thetaH_)",
                               "max(thetaC_ + thetaH_, dimensionedScalar"
                               '("z", dimless, 0))', 1), 1)  # noqa: E501
        self.assertNotEqual(clamped, self.src, "nothing was mutated")
        self.assertFalse(limiter_field_is_the_unclamped_ratio(clamped),
                         "a clamp on the limiter field was not detected")

        # S6c-2: the legacy branch is gone, so the mutation that must be
        # caught is c_ coming BACK as the limiter argument -- the one field
        # whose name fvc::grad would resolve differently.
        copied_c = self.src.replace(
            "            tinterp().weights(limiterField)",
            "            tinterp().weights(c_)", 1)
        self.assertNotEqual(copied_c, self.src, "nothing was mutated")
        self.assertFalse(legacy_limiter_binds_c_itself(copied_c),
                         "rebinding the limiter to c_ was not detected")

        reconditioned = self.src.replace(
            "    const volScalarField limiterField",
            "    const volScalarField& limiterField = contentLimiter ?"
            " tLimiterField() : q_;\n    const volScalarField limiterField",
            1)
        self.assertNotEqual(reconditioned, self.src, "nothing was mutated")
        self.assertFalse(
            legacy_limiter_binds_c_itself(reconditioned),
            "reintroducing a conditional limiter selection was not "
            "detected, and the second branch can only name a dead basis")

        rebinned = self.src.replace(
            "const scalar interfaceLo = contentLimiter ? 0.05 : -0.9;",
            "const scalar interfaceLo = contentLimiter ? 0.05 : -0.5;", 1)
        self.assertNotEqual(rebinned, self.src, "nothing was mutated")
        self.assertFalse(
            interface_bin_is_basis_aware_and_legacy_is_unchanged(rebinned),
            "moving the LEGACY half of the interface bin was not detected")

        # S6b-2
        pbody = platelet_convection_body(self.src)
        self.assertTrue(pbody, "plateletContentConvection was not found")

        collapsed = self.src.replace(
            pbody,
            pbody.replace('if (phasePlateletFluxBasis_ != "content")',
                          "if (false)", 1),
            1)
        self.assertNotEqual(collapsed, self.src, "nothing was mutated")
        self.assertFalse(
            legacy_path_is_called_through_not_reimplemented(collapsed),
            "collapsing the legacy branch was not detected")

        unguarded_w = self.src.replace(
            pbody, pbody.replace("if (!contentFaceWeights_.valid())",
                                 "if (false)", 1), 1)
        self.assertNotEqual(unguarded_w, self.src, "nothing was mutated")
        self.assertFalse(content_path_refuses_absent_weights(unguarded_w),
                         "dropping the absent-weights refusal was not "
                         "detected")

        wrong_weights = self.src.replace(
            pbody,
            pbody.replace("contentFaceWeights_());", "phaseWeights_());", 1),
            1)
        self.assertNotEqual(wrong_weights, self.src, "nothing was mutated")
        self.assertFalse(content_path_uses_the_content_weights(wrong_weights),
                         "swapping the content weights back to the phase "
                         "field's was not detected")

        spread = self.src.replace(
            "+ phaseSharedConvection(carrier, shell_)",
            "+ plateletContentConvection(carrier, shell_)", 1)
        self.assertNotEqual(spread, self.src, "nothing was mutated")
        self.assertFalse(
            only_Wp_takes_the_selected_weights(spread),
            "extending the key to shell_ was not detected -- that is the "
            "bundled second change this split exists to prevent")

        copied = self.src.replace(
            "return phaseConservativeConvectionWith(carrier, f, "
            "phaseWeights_());",
            "return\n        fv::gaussConvectionScheme<scalar>\n        (\n"
            "            mesh_,\n            carrier,\n"
            "            tmp<surfaceInterpolationScheme<scalar>>\n"
            "            (\n                new suppliedWeightsScheme"
            "(mesh_, phaseWeights_())\n            )\n"
            "        ).fvmDiv(carrier, f);", 1)
        self.assertNotEqual(copied, self.src, "nothing was mutated")
        self.assertFalse(one_implementation_of_the_operator(copied),
                         "re-copying the operator body was not detected")

        # the capture
        # S6c-2: the mutation that must now be caught is the REVERSE one --
        # reintroducing a conditional capture on an operator that can no
        # longer be selected, which would hand downstream the weights of an
        # equation nobody assembled.
        uncond = self.src.replace(
            "    contentFaceWeights_ = tmp<surfaceScalarField>",
            "    if (matched && phaseWeights_.valid())\n"
            "    {\n"
            "        contentFaceWeights_ = tmp<surfaceScalarField>\n"
            "        (\n"
            '            new surfaceScalarField("contentFaceWeights",'
            " phaseWeights_())\n"
            "        );\n"
            "    }\n"
            "    contentFaceWeights_ = tmp<surfaceScalarField>",
            1)
        self.assertNotEqual(uncond, self.src, "nothing was mutated")
        self.assertFalse(content_face_weights_follow_the_matrix(uncond),
                         "reintroducing the matched capture branch was not "
                         "detected")

        # S10: a capture that REBUILDS its weights instead of copying the
        # matrix's must fail -- under limitedLinear that is a second limiter.
        rebuilt = self.src.replace(
            'new surfaceScalarField("contentFaceWeights", tContentWeights())',
            'new surfaceScalarField("contentFaceWeights", '
            'upwind<scalar>(mesh, carrier).weights())', 1)
        self.assertNotEqual(rebuilt, self.src, "nothing was mutated")
        self.assertFalse(content_face_weights_follow_the_matrix(rebuilt),
                         "a rebuilt capture was not detected")

        silent = self.src.replace('Info<< "PHASE_CLOT_PLATELET_FLUX_CONFIG"',
                                  'Info<< "PHASE_CLOT_PLATELET_FLUX_OFF"', 1)
        self.assertNotEqual(silent, self.src, "nothing was mutated")
        self.assertFalse(both_configs_are_emitted(silent),
                         "renaming a config emit was not detected")


if __name__ == "__main__":
    print("SOURCE-DERIVED guard for S6b. It says what the code DOES. It does "
          "NOT say which theta profile is right -- that is still open.",
          flush=True)
    unittest.main(verbosity=2)
