#!/usr/bin/env python3
"""Guard for the phase INDICATOR basis (phaseIndicatorBasis).

WHAT THIS KEY IS, AND WHY IT IS NOT THE FRACTION KEY. Eleven sites computed
`chi = max(1 - q, 0)` inline and NOT ONE of them scaled it by theta_ref. They
are consumers of the raw indicator -- a marker in [0, ~1] -- not inlined
copies of the solid volume fraction, which is theta_ref = 0.45 times smaller.
A migration that wrote `theta` at those sites would silently rescale the
deviatoric elastic stress, the damage driver, the Kres ramp, the isSkel
threshold and the bond hold by 0.45 in a single edit, and nothing downstream
would notice. The content indicator is `theta/theta_ref`, and

    THE DIVISION BY theta_ref IS THE WHOLE SUBSTITUTION.

which is why it has its own predicate and its own mutation test below.

THE UNCLAMPED DECISION. Both indicators already exceed 1 -- chi at 1.0021786
from the c undershoot, theta/theta_ref at 1.0021686 from compression past the
anchor. The decision taken 2026-09-15 is NOT to clamp above, and to measure
the overshoot instead (PHASE_CLOT_PHASE_INDICATOR: indMax, indAboveOne). Two
consequences are guarded here because both are easy to erode:

  * the accessor must stay unclamped above -- a later "tidy-up" min(.., 1)
    would silently retire the decision AND blind the instrument;
  * the ONE site where ind > 1 is genuinely unsafe -- the material-carrier
    blend, where chif > 1 makes (1 - chif) negative and SUBTRACTS the blood
    flux -- must keep its own min(.., 1). The key sets the basis of the ratio,
    never a consumer's own bounds.

The lower guard at zero is a different decision and is guarded separately: a
negative indicator reaches fvm::Sp on the momentum and skeleton diagonals
through holdChi, and the "matched" content operator drives theta to -7.5e-18
in 2354 cells.

Run: python3 scripts/check_phase_indicator_basis.py
"""

import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"
HDR = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.H"
GEN = ROOT / "cases/AM2020_planar_mural_detachment/scripts/create_cases.py"

sys.path.insert(0, str(ROOT / "scripts"))
from check_content_consumer_map import (  # noqa: E402
    INDICATOR, INDICATOR_ACCESSORS, call_sites, code_lines,
    enclosing_functions)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def live(text: str) -> str:
    """Whitespace-free source with comments and string bodies removed.

    S6c-2 turned many of this file's predicates into ABSENCE assertions, and
    absence cannot be tested against `compact(src)`: the solver's comments
    explain deltaC_, frontSource and the c increment at length -- the
    refusal messages quote them by name -- so a plain substring test reads
    prose as live code. That exact mistake cost a debugging round on
    check_phase_normal_basis.py, where the string literal
    "interfaceDelta_ was deltaC_ = 0.5*|grad c|." inside a FatalError was
    read as a surviving use of the deleted field.
    """
    return compact("\n".join(code_lines(text)))


def _function_body(src: str, signature: str) -> str:
    """The CODE of one function, comments stripped, by brace matching.

    Comments must go. This accessor's own comment explains at length why it
    divides by theta_ref and why it is not clamped above, and a substring
    search over the raw text would match that prose rather than any code --
    a predicate that reads its own documentation proves nothing. That is not
    hypothetical: it happened to the forceFraction guard on 2026-09-15.
    """
    i = src.find(signature)
    if i < 0:
        return ""
    j = src.find("{", i)
    if j < 0:
        return ""
    depth, k = 0, j
    while k < len(src):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    body = src[j:k + 1]
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return body


def _indicator_body(src: str) -> str:
    return _function_body(
        src, "phaseClotAxisymmetricFluid::phaseIndicator() const")


def _blend_body(src: str) -> str:
    """The materialCarrierBlend branch of solvePhaseField()."""
    i = src.find("if (materialCarrierBlend_)")
    return src[i:i + 1600] if i >= 0 else ""


# ---- predicates ---------------------------------------------------------

def accessor_exists(src: str) -> bool:
    return bool(_indicator_body(src))


def content_branch_divides_by_theta_ref(src: str) -> bool:
    """THE substitution. theta is NOT the indicator; theta/theta_ref is.

    Without this division every migrated site is rescaled by 0.45 -- a
    constitutive recalibration disguised as a representation change.
    """
    body = compact(_indicator_body(src))
    return ('/dimensionedScalar("thetaRef",dimless,skeletonVolumeFraction_)'
            in body)


def content_branch_reads_the_content_fields(src: str) -> bool:
    # S21A d59 removed the `phaseIndicatorBasis_ == "content"` guard.
    body = compact(_indicator_body(src))
    return "thetaC_+thetaH_" in body


def indicator_is_guarded_below_at_zero(src: str) -> bool:
    body = compact(_indicator_body(src))
    return ('max(thetaC_+thetaH_,dimensionedScalar("zero",dimless,0))'
            in body)


def indicator_is_not_clamped_above(src: str) -> bool:
    """The decision of 2026-09-15, written as a test so it cannot drift back.

    A min() anywhere in this accessor is a ceiling on the ratio. If one is
    ever wanted the decision has to be retaken in the consumer map and this
    predicate rewritten -- which is the point of asserting an ABSENCE here.
    """
    return "min(" not in compact(_indicator_body(src))


def accessor_is_not_the_fraction(src: str) -> bool:
    """It must not quietly become solidFraction()/forceFraction()/phiSkel_.

    Those are volume fractions; this is an indicator theta_ref times larger.
    phiSkel_ additionally is constructed as ZERO and refreshed only inside
    momentumPredictor(), while elasticStress() reads the indicator before the
    predictor runs on the first step.
    """
    body = compact(_indicator_body(src))
    return not any(name in body for name in
                   ("phiSkel_", "solidFraction()", "forceFraction()"))


# S21 (2026-10-05, docket 20261005-1621 report 15): legacy_branch_is_the_old_expression() retired; its text was deleted by S21A a43 (legacy branch of phaseIndicator()).



# S21 (2026-10-05, docket 20261005-1621 report 15): basis_validated_against_explicit_list() retired; its text was deleted by S21A c07.



# S21 (2026-10-05, docket 20261005-1621 report 15): validation_is_outside_the_content_model_gate() retired; its text was deleted by S21A c07.



# S21 (2026-10-05, docket 20261005-1621 report 15): content_requires_the_content_model() retired; its text was deleted by S21A a16.



def content_guards_the_divisor(src: str) -> bool:
    """theta_ref is a case parameter. At zero the ratio divides by zero in
    every migrated site at once rather than in one obvious place.

    SCOPED to this refusal's own message, for the reason
    content_refuses_unlocalised_deposition already carries: since S3 the
    phaseDeltaBasis validation holds an identical `if
    (skeletonVolumeFraction_ <= 0)` a few dozen lines down, and a whole-file
    search found it the moment this one was deleted -- the mutation stopped
    firing and the guard silently stopped guarding.
    """
    msg = "phaseIndicatorBasis 'content' divides by"
    i = src.find(msg)
    if i < 0:
        return False
    return "if (skeletonVolumeFraction_ <= 0)" in src[max(0, i - 700):i]


# S21 (2026-10-05, docket 20261005-1621 report 15): content_refuses_unlocalised_deposition() retired; its text was deleted by S21A a17.



# S21 (2026-10-05, docket 20261005-1621 report 15): carrier_blend_keeps_its_own_clamp() retired; its text was deleted by S21A a29 (materialCarrierBlend branch).



def deviatoric_weight_keeps_its_floor(src: str) -> bool:
    # S21C1 d28.1 deleted the skeletonDiffuseDomain_ `max(chi, solidFloor)`
    # branch; the kept elasticStress weight is skeletonStressWeight()
    # (occupancy-floored), assigned unconditionally.
    i = src.find("sigmaElasticInstant")
    body = compact(src[max(0, i - 4000): i + 400] if i >= 0 else "")
    return "phiS=skeletonStressWeight();" in body \
        and "phiS*gD*GField_*dev(be_)" in body


def damage_driver_keeps_its_square(src: str) -> bool:
    return "sqr(chiDamage)*psiPlusRaw" in compact(src)


def isSkel_keeps_its_threshold(src: str) -> bool:
    """pos0(ind) is NOT the same boolean: the test is against 0.5*residualChi_
    and would otherwise fire in exactly the thin-skeleton cells it excludes."""
    c = compact(src)
    return ('pos0(phaseIndicator()-dimensionedScalar("halfChi",dimless,'
            '0.5*residualChi_))' in c)


def kres_ramp_reads_the_indicator(src: str) -> bool:
    return 'volScalarFieldchiReg("chiReg",phaseIndicator())' in compact(src)


def bond_hold_uses_one_basis_at_both_sites(src: str) -> bool:
    """Two independent fvm::Sp damping terms that share one coefficient. They
    are NOT an equal-and-opposite exchange pair, but they must not diverge."""
    return compact(src).count("bondHold_*phaseIndicator()") == 2


def instrument_reports_the_unclamped_overshoot(src: str) -> bool:
    c = compact(src)
    return ("PHASE_CLOT_PHASE_INDICATOR" in c
            and '"indAboveOne="' in c
            and '"indMax="' in c
            and '"chiLegacyMax="' in c)


def instrument_reports_the_two_sets_that_move(src: str) -> bool:
    """kresCells and isSkelCells are the quantities the migration CHANGES.

    indMax moves by 6.9 ppm; the sets defined by the residualChi_ thresholds
    move by 34 and 27 cells. Pre-registering a prediction on an unlogged
    quantity is a prediction that cannot be graded -- three runs have already
    been spent that way -- so the sets are logged.
    """
    c = compact(src)
    return ('"kresCells="' in c and '"isSkelCells="' in c
            and "if(ind[celli]<residualChi_)" in c
            and "if(ind[celli]>0.5*residualChi_)" in c)


def na_is_reported_rather_than_fabricated(src: str) -> bool:
    """Below residualChi_ <= 0 the solver keys both sets off phiS instead.

    Printing a chi-threshold count there would be a made-up number, not a
    missing one.
    """
    c = compact(src)
    return "if(residualChi_<=0){kresCells=-1;isSkelCells=-1;}" in c


def instrument_keeps_a_legacy_reference(src: str) -> bool:
    """chiLegacyMax must be the RAW legacy form, not phaseIndicator().

    Migrating it would make chiLegacyMax equal indMax by construction and the
    comparison would destroy itself -- the same defect the C1 diagnostic in
    momentumPredictor was deliberately protected from.
    """
    return ('constvolScalarFieldchiLegacy(max(1.0-q_,dimensionedScalar('
            '"zero",dimless,0)))' in compact(src))


def surface_delta_instrument_exists(src: str) -> bool:
    """The tag survives S6c-2, reporting only its theta side.

    deltaCInt / deltaRatio / deltaTV / deltaShift / rmsC / supportRatio went
    with `c`. deltaThetaInt, deltaThetaIntPost and deltaThetaIntAtTransfer
    are theta-only and together they ARE OI-11 -- which time level the
    reaction reads the geometry at, a question deleting c does not touch.
    """
    c = compact(src)
    return ('Info<<"PHASE_CLOT_SURFACE_DELTA"' in c
            and '<<"deltaThetaInt="<<deltaThetaInt' in c
            and '<<"deltaThetaIntPost="<<deltaThetaIntPost' in c
            and '<<"deltaThetaIntAtTransfer="<<deltaThetaIntAtTransfer_' in c)

def surface_delta_is_normalised_by_theta_ref(src: str) -> bool:
    """INT |grad theta|/theta_ref dn = theta_bulk/theta_ref = indMax.

    Without the division the integral is theta_ref times too small and the
    ratio no longer tracks indMax -- which is the entire reason this form was
    chosen, because it makes the delta's calibration error a number already in
    the log.
    """
    #
    # BOUND TO THE ASSIGNMENT IN updateDerivedFields(), not to "anywhere in
    # the file". Since S3 the expression appears twice -- once as the field
    # the consumers read, once as the instrument's postSolve recomputation
    # (deltaThetaIntPost) -- so a whole-file search stays true after the
    # consumed one is broken, and the mutation below stopped firing.
    c = compact(src)
    return ('deltaTheta_=mag(fvc::grad(thetaC_+thetaH_))/dimensionedScalar'
            '("thetaRef",dimless,skeletonVolumeFraction_);' in c)


def surface_delta_uses_fvc_grad(src: str) -> bool:
    """Not a hand reconstruction. One was wrong by 2x in this project.

    Bound to the assignment for the same reason as the normalisation above:
    the expression occurs twice since S3 and the second occurrence kept this
    true after the first was broken.
    """
    c = compact(src)
    return "deltaTheta_=mag(fvc::grad(thetaC_+thetaH_))" in c


def instrument_compares_deltaC_against_deltaTheta(src: str) -> bool:
    """RETIRED. The comparison had two operands and one of them is deleted.

    Keeping it would mean rebuilding deltaC_ from theta and comparing theta
    with theta. This asserts the comparison is gone rather than silently
    self-satisfying, which is the whole reason it could not be migrated.
    """
    c = live(src)
    return ("deltaC_" not in c
            and "if(deltaCInt>0&&deltaThetaInt>0)" not in c)

def instrument_reports_the_postsolve_recomputation(src: str) -> bool:
    """deltaThetaIntPost separates "the localiser moved" from "the instrument
    moved".

    deltaTheta_ is assigned in updateDerivedFields(), which is where the
    consumers read it; the instrument runs in postSolve(), after
    solvePlatelets() has committed the transfer. Recomputing the integral
    there and printing both makes that time-level difference a measured
    number every step instead of an assumption that it is negligible.
    """
    c = compact(src)
    return ('"deltaThetaIntPost="' in c and "deltaThetaIntPost=fvc" in c)


def _surface_delta_block(src: str) -> str:
    """The instrument's own braces, by brace matching -- not a char window.

    The block grew when the statistic was replaced, and a fixed-size window
    would have silently started reading the neighbouring diagnostic instead of
    failing. Locate the opening brace that precedes the block's first
    statement and match it.
    """
    i = src.find("scalar deltaThetaInt = -1;")
    if i < 0:
        return ""
    start = src.rfind("{", 0, i)
    if start < 0:
        return ""
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
    return ""


def surface_delta_reports_a_distribution_statistic(src: str) -> bool:
    """RETIRED IN S6c-2, and the retirement is declared in the line itself.

    deltaTV was the half-L1 distance between deltaC_ and deltaTheta_ once
    each was normalised to unit mass. deltaC_ is deleted. Rebuilding it from
    theta would make the instrument compare a field with itself -- deltaTV
    identically zero, a triumphant pass measuring nothing, which is exactly
    the G3 defect this instrument's own comment warned about.

    So the guard inverts: the retired names must be GONE, and the emitted
    line must say they were retired, so a reader of an old log against a new
    one is told rather than left to infer it from an absence.
    """
    c = compact(src)
    retired = ("deltaTV", "deltaRatio", "deltaCInt", "deltaShiftMag", "rmsC")
    if any('<<"%s="' % n in c for n in retired):
        return False
    return ('<<"retiredInS6c2="' in c
            and "deltaCInt+deltaRatio+deltaTV+deltaShift+rmsC+supportRatio"
            in c)

def surface_delta_names_the_ratio_as_reported_not_gated(src: str) -> bool:
    """The gate label must name a gate that still exists.

    It used to read `gateIs=deltaTV`, with deltaRatio reported beside it and
    explicitly not gated. deltaTV is retired, so a label naming it would
    point at nothing -- the failure mode this project keeps hitting, an
    emitted string describing a stage that has already gone.
    """
    c = compact(src)
    return ('<<"gateIs=OI-11-deltaThetaIntAtTransfer"' in c
            and '<<"gateIs=deltaTV"' not in c)

def surface_delta_tv_differences_normalised_fields(src: str) -> bool:
    """Retired with deltaTV: nothing is differenced any more."""
    c = live(src)
    # `deltaTV`, and not the `deltaTValue()` that appears fifteen times: a
    # bare substring test here passed on runTime.deltaTValue and reported a
    # retired instrument as still present.
    return (not re.search(r"deltaTV(?![A-Za-z])", c)
            and "constvolScalarFielddCn(deltaC_/deltaCInt);" not in c)

def surface_delta_tv_is_halved(src: str) -> bool:
    """Retired with deltaTV."""
    c = live(src)
    return "0.5*fvc::domainIntegrate(mag(dCn-dTn)).value();" not in c

def delta_theta_is_a_written_field(src: str) -> bool:
    """The per-cell comparison must not depend on this instrument's choice.

    The first version reported one number and the question turned out to be a
    distribution; there was then no way to ask the run anything else without
    running it again. AUTO_WRITE puts deltaTheta beside interfaceDelta in
    every time directory, so the next question can be asked of the same run.
    """
    c = compact(src)
    return ('"deltaTheta",runTime.name(),mesh,IOobject::NO_READ,'
            "IOobject::AUTO_WRITE" in c)


def surface_delta_guards_every_divisor(src: str) -> bool:
    """ONE divisor survives, and it is still guarded.

    Both normalising integrals went with deltaC_. What is left divides by
    skeletonVolumeFraction_ to form deltaThetaIntPost, under the same guard
    that gates the whole block, and the sentinel stays -1 -- outside the
    range of a non-negative integral, so `not computable` can never be read
    as a measurement.
    """
    # S21A d53 reduced `plateletContentModel_ && skeletonVolumeFraction_ > 0`
    # to its second clause; pinned contiguous with the two sentinels.
    c = compact(src)
    return ("scalardeltaThetaInt=-1;scalardeltaThetaIntPost=-1;"
            "if(skeletonVolumeFraction_>0)" in c)

def surface_delta_decomposes_moved_from_spread(src: str) -> bool:
    """Retired with deltaTV: deltaShift separated a translated interface from
    a smeared one, and both operands are gone."""
    c = compact(src)
    return not any('<<"%s="' % n in c
                   for n in ("deltaShiftMag", "deltaShiftX", "deltaShiftY"))

def delta_theta_reaches_equations_only_through_interfaceDelta(src: str) -> bool:
    """deltaTheta_ is consumed now, but only by ONE assignment.

    Under phaseDeltaBasis 'content' this field localises the accepted transfer
    and the bound-platelet uptake -- but it does so by BEING interfaceDelta_,
    not by being read directly. Any other function touching deltaTheta_ is a
    second, unkeyed consumer: it would be live whatever the key says, and the
    legacy setting would stop being a control.

    Permitted owners: the constructor's initialiser, updateDerivedFields()
    (the assignment and the selection) and writeBalanceDiagnostic() (the
    instrument). Line-wise over code_lines, so the paragraphs that NAME the
    field while explaining it are not counted as reads.

    S4 ADDS EXACTLY ONE MORE, AND IT IS WHITELISTED BY ITS TEXT, NOT BY ITS
    FUNCTION. solvePlateletContent() snapshots the field's integral where the
    reaction reads it (OI-11). Allowing the whole function would have opened
    the door to a second unkeyed consumer in the one place it would do the
    most damage -- the transfer itself -- so only that single statement is
    permitted, and any other mention of deltaTheta_ there still fails.
    """
    lines = code_lines(src)
    owners = enclosing_functions(lines)
    allowed = {"updateDerivedFields", "writeBalanceDiagnostic"}
    snapshot = ("deltaThetaIntAtTransfer_ = "
                "fvc::domainIntegrate(deltaTheta_).value();")
    for n, ln in enumerate(lines):
        if "deltaTheta_" not in ln:
            continue
        if owners[n] in allowed:
            continue
        # WHITELISTED BY TEXT, NOT BY OWNER, AND THE REASON IS THE OWNER
        # HEURISTIC. enclosing_functions() attributes this line to
        # checkPlateletConfiguration rather than solvePlateletContent -- the
        # same signature-spanning-lines weakness that put two sites in
        # "line 7423" when cxx_query.py was first written. The statement text
        # is unique in the file, and its LOCATION is asserted separately and
        # reliably by oi11_snapshot_is_taken_where_the_reaction_reads_it(),
        # which slices solvePlateletContent()'s body by braces.
        if ln.strip() == snapshot:
            continue
        # The initialiser-list entry: the bare member name on its own line.
        if re.fullmatch(r"\s*deltaTheta_\s*", ln):
            continue
        return False
    return True


def interface_delta_is_selected_by_the_key(src: str) -> bool:
    """ONE ARM LEFT, AND THAT IS NOW CORRECT -- but it must be unconditional.

    Until S6c-2 this had to be a two-armed selection, because a one-armed
    `selection` is a rename wearing a key's clothes. deltaC_ is deleted, so
    there is no second arm to offer: the assignment is unconditional, there
    is exactly one of it, and the key that used to choose is refused on its
    dead value elsewhere (see check_phase_normal_basis.py). A surviving
    branch here would mean the key still selects something.
    """
    c = compact(src)
    if "interfaceDelta_=deltaTheta_;" not in c:
        return False
    if 'if(phaseDeltaBasis_=="content"){interfaceDelta_=deltaTheta_;}' in c:
        return False
    assigns = [ln for ln in code_lines(src)
               if re.search(r"^\s*interfaceDelta_\s*=", ln)]
    return len(assigns) == 1

def delta_c_is_the_legacy_form(src: str) -> bool:
    """deltaC_ = 0.5*|grad c| is DELETED. Reintroducing it puts `c` back."""
    return "deltaC_" not in live(src)

def delta_c_is_a_written_field(src: str) -> bool:
    """And it is no longer written, so no time directory carries it.

    The assessor was migrated with it: `assess_case.py` reconstructs the
    phase field as c = 2q - 1 rather than reading a written one.
    """
    c = compact(src)
    return ('"deltaC",runTime.name(),mesh,IOobject::NO_READ,'
            "IOobject::AUTO_WRITE" not in c)

def phase_gradient_stays_on_c(src: str) -> bool:
    """IT MOVED, DELIBERATELY, AND IT IS NOT DEAD.

    The S3-era guard held phaseGradient_ on |grad c| because its only live
    consumer was the Stefan front in the c equation and moving it would have
    changed c's own trajectory. S4 removed that consumer and the S6c scoping
    doc then recorded the field as dead. It is not: clotGrowthFront reads it
    every step and feeds PHASE_CLOT_GROWTH_STOICH, unconditionally, on every
    driver -- the 1.6277-against-1.6304 identity every archived run is
    compared on.

    So it is rebuilt on theta, from the same magGradN the normal uses, and
    the guard now pins THAT: one expression, sharing the normal's gradient,
    with the floor subtracted so it is zero where the field is flat.
    """
    c = compact(src)
    return ("phaseGradient_=sqrt(sqr(magGradN)+sqr(gradFloor_))-gradFloor_;"
            in c
            and "sqr(magGradC)" not in c)

# S21 (2026-10-05, docket 20261005-1621 report 15): delta_basis_validated_against_explicit_list() retired; its text was deleted by S21A c08.



# S21 (2026-10-05, docket 20261005-1621 report 15): delta_basis_requires_the_content_model() retired; its text was deleted by S21A a18.



def delta_basis_guards_its_divisor(src: str) -> bool:
    """SCOPED to its own message, for the reason the indicator's twin is."""
    msg = "phaseDeltaBasis 'content' divides by"
    i = src.find(msg)
    if i < 0:
        return False
    return "if (skeletonVolumeFraction_ <= 0)" in src[max(0, i - 700):i]


def delta_basis_defaults_to_content(src: str) -> bool:
    """Alireza, 2026-09-16: the new method is the default, not the exception.

    Deliberately the opposite of default_is_legacy() above, which guards the
    three keys that have NOT been flipped. If those three ever move, this is
    the pattern to follow -- and until they do, the asymmetry is the record
    that only one was decided.
    """
    return '"phaseDeltaBasis", "content"' in src


def delta_basis_declared_in_header(hdr: str) -> bool:
    """The key survives with one admissible value; deltaC_ does not."""
    # S21A h20 deleted `word phaseDeltaBasis_;`.
    c = compact(hdr)
    return ("volScalarFielddeltaC_;" not in c
            and "volScalarFielddeltaTheta_;" in c)

# S21 (2026-10-05, docket 20261005-1621 report 15): generator_writes_the_delta_key() retired; its text was deleted by generator: release-4198bb17 create_cases.py L4149-4153.



# S21 (2026-10-05, docket 20261005-1621 report 15): default_is_legacy() retired; its text was deleted by S21A b10.



def declared_in_header(hdr: str) -> bool:
    # S21A h19 deleted `word phaseIndicatorBasis_;`.
    c = compact(hdr)
    return "tmp<volScalarField>phaseIndicator()const;" in c


# S21 (2026-10-05, docket 20261005-1621 report 15): generator_validates_the_key() retired; its text was deleted by generator: release create_cases.py L2057.



def not_migrated_sites_still_read_the_raw_form(src: str) -> bool:
    """i2, i7 and i9 were deliberately NOT migrated, and i10 keeps a raw
    reference. If one of them quietly moves, the consumer map's reasons --
    unreachable, superseded, diagnostic-last -- stop describing the code.
    """
    raw = {f: v for f, v in call_sites(r"max\s*\(\s*1\.0\s*-\s*q_").items()
           if f not in INDICATOR_ACCESSORS}
    # S21C1 b23 deleted solveSkeleton() and b27 writeSkeletonResidual().
    return set(raw) == {"momentumPredictor", "writeBalanceDiagnostic"}


def every_indicator_site_is_one_of_two_forms(src: str) -> bool:
    """A CENSUS, so that a site appearing or vanishing has to be declared.

    13 sites across 8 functions until S5. S5 added two, one each in
    updateReleasedIndicator() and updateBondHold(): those two mask builders
    were the last places reading c_.primitiveField() directly, which is why
    run 20260919-000047 carried phaseIndicatorBasis=content everywhere except
    in the masks that decide what detaches. 15 across 10.

    S6a to S6c-2 took it to 31 across 17, and the jump is the point of the
    census: every read of `c` that survived into a live path had to become a
    read of the accessor before `c` could be deleted. The seven new functions
    are the ones the deletion swept -- among them releaseDetachedMaterial(),
    whose reset now zeroes thetaC and thetaH rather than setting c, and
    writeBalanceDiagnostic(), which carries the replacement for the retired
    PHASE_CLOT_MASS.

    S11 (2026-09-23) took it to 33 across 18: skeletonFaceStress() builds the
    FACE stress's deviatoric weight exactly as elasticStress() builds the
    cell one (the floored indicator -- a support, never a fraction), and
    correctPressureTwoVelocity() gained one read for the S11 diagnostics.
    Both are declared in the consumer map as i25 and i26.

    S12 (2026-09-23) took it to 35 across 20: intrinsicDragRate() weights the
    network's own drag by the indicator -- UNFLOORED, because a drag is a bulk
    coefficient and not a support (i27) -- and reportFringe() weights its slip
    rms by it (i28). intrinsicDragRate is in the required set: it is the one
    new site that drives an equation.

    S14 (2026-09-23) took it to 36 across 21: ruptureOverstretchedNetwork()
    reads the indicator only to classify a rupture as bulk or fringe in its
    report (i29).

    S16 (2026-09-24) took it to 38 across 23: skeletonStressWeight() builds
    the occupancy-floored deviatoric weight max(ind, floor)*(1 - d) that
    skeletonFaceStress() and elasticStress() use under skeletonFloorBasis
    occupancy -- a floored SUPPORT, like i1 and i25 (i30) -- and
    recordStressWeight() reads it to build S16w's reference independently of
    the sites it checks; reported, drives nothing (i31).

    S17 (2026-09-26) took it to 40 across 25: the Kres regularisation's read
    (i5) MOVED, unchanged, from correctPressureTwoVelocity() into
    skeletonResidualDrag(), so the 2x2 block and S17 C's Vs predictor put
    one Kres on the skeleton diagonal; correctPressureTwoVelocity() gained
    the PHASE_CLOT_S17 diagnostic's read (P2's population, i32); and
    writeS17SecondDifferences() reads the face-interpolated indicator to
    select the clot-surface faces of the CB pair's instrument (i33). Both new
    reads are diagnostics and drive nothing.

    S19 (2026-10-04) took it to 41 across 26: solveBlockCoupledStep() (the
    block-coupled step, blockCoupled only) hoists ONE indicator for i6's
    isSkel threshold (unchanged; it selects the lumen identity rows), the
    run-time bond-seed test, VS_SPLIT and bodySlipRms (i34).

    S21 (2026-10-05) took it to 34 across 22 by deletion only: S21C1 b23
    solveSkeleton() (1), b26 solveMechanicsExchange() (1), b27
    writeSkeletonResidual() (2), and solvePhaseField()'s three reads (S21A
    a29 carrier blend, S21B a36 rigid-body released carrier, S21C1 d36.2
    material-flux chain element). No site was added; the required set is
    unchanged.

    S23 (2026-10-07) took it to 35 across 23: tornNetworkCells() reads the
    indicator for the torn set's isSkel (the block's rule) and for the
    packing scale s = (1 - d*) + d* ind of the torn-face correction, which
    drives the skeleton force (required).
    """
    sites = {f: v for f, v in call_sites(INDICATOR).items()
             if f not in INDICATOR_ACCESSORS}
    return (sum(len(v) for v in sites.values()) == 35 and len(sites) == 23
            and {"updateReleasedIndicator", "updateBondHold", "tornNetworkCells",
                 "releaseDetachedMaterial", "skeletonFaceStress",
                 "writeBalanceDiagnostic", "intrinsicDragRate",
                 "skeletonStressWeight", "recordStressWeight",
                 "skeletonResidualDrag", "writeS17SecondDifferences"}
            <= set(sites))

# ---------------------------------------------------------------------------
# S4 -- THE GROWTH DRIVER. phaseGrowthDriver_ front | transfer.
#
# These guard a MODEL change, not a discretisation one: the Stefan front
# leaves the c equation and the accepted platelet transfer becomes the only
# growth. A half-applied S4 is silent -- the run completes, every dimension
# check passes, every balance closes -- which is why each predicate below has
# a mutation beside it.
# ---------------------------------------------------------------------------


def front_is_built_zero_and_assigned_only_under_front(src: str) -> bool:
    """frontSource IS GONE, because the equation it sourced is gone.

    S4's rule was that the product Vn_*phaseGradient_ must never be FORMED
    under 'transfer' -- a zero multiplier is not equivalent, one non-finite
    Vn becomes a nan over the whole equation. S6c-2 deletes the c equation
    outright and refuses `phaseGrowthDriver front` at construction, so the
    stronger statement holds: the field does not exist.
    """
    c = live(src)
    return "frontSource" not in c

def all_three_c_branches_read_the_gate(src: str) -> bool:
    """Zero branches, because there is no c equation to branch on.

    Three `- frontSource` terms became none. The guard is kept rather than
    deleted so that reintroducing any of them -- gated or not -- fails.
    """
    lines = code_lines(src)
    return not [ln for ln in lines
                if re.match(r"^\s*-\s*frontSource\s*$", ln)
                or re.match(r"^\s*-\s*Vn_\*phaseGradient_\s*$", ln)]

def front_source_is_built_outside_the_corrector_loop(src: str) -> bool:
    """Vacuously true once frontSource is gone, so it asserts the premise.

    An ordering guard on a field that does not exist would pass forever
    without measuring anything -- the unfailable-gate defect. It checks the
    absence instead, and the ordering guard that still matters (the transfer
    accumulator's reset) is tested separately below.
    """
    body_text = _function_body(
        src,
        "void Foam::solvers::phaseClotAxisymmetricFluid::solvePhaseField")
    return "volScalarField frontSource" not in body_text

def increment_uses_the_applied_theta_change(src: str) -> bool:
    """dTheta STAYS; the write to c it fed does not.

    It was two statements: `c -= (2/theta_ref)*dTheta`, keeping the legacy
    field in step with the transfer, and the integral that defines
    transferClotGrowth_. The first is deleted. The second must still take the
    APPLIED change -- plateletTransferKernel::applyIncrement clamps against
    phiSMax_, so a step taken from Y*dB would leave the currencies
    disagreeing exactly when the cap bites.
    """
    c = compact(src)
    return ('constvolScalarFielddTheta("dTheta",thetaNew-thetaStar);' in c
            and "fvc::domainIntegrate(dTheta).value()" in c
            and "c_-=" not in c)

def increment_converts_by_two_over_theta_ref(src: str) -> bool:
    """AND THE 2/theta_ref SURVIVES THE DELETION, on purpose.

    It looks like a reference to c -- chi = 0.5*(1-c) and theta =
    theta_ref*chi give dc = -(2/theta_ref)*dtheta -- but it is the
    normalisation transferClotGrowth_ is DEFINED in, and
    PHASE_CLOT_GROWTH_STOICH divides it by plateletUptakeCommitted to get the
    exchange rate (1.6277 measured against a 1.6304 target). Rescaling it to
    "theta currency" for tidiness would move that ratio by a constant and
    silently invalidate every stoichiometry record on file.
    """
    c = compact(src)
    return ("transferClotGrowth_=(scalar(2)/skeletonVolumeFraction_)"
            "*fvc::domainIntegrate(dTheta).value()" in c
            and "c_-=(scalar(2)/skeletonVolumeFraction_)*dTheta;" not in c)

def increment_corrects_boundary_conditions(src: str) -> bool:
    """The field that needed the correction is deleted.

    A zeroGradient patch evaluated off an un-incremented neighbour used to
    carry the old c back in through the face flux. There is no c. The guard
    asserts the pair is gone together: an increment reinstated without its
    boundary correction is the half-migration this predicate was written for.
    """
    c = compact(src)
    return ("c_-=" not in c and "c_.correctBoundaryConditions();" not in c)

def transfer_growth_is_overwritten_not_accumulated(src: str) -> bool:
    """fvm::ddt(c_) rebuilds c from oldTime at the top of every corrector, so
    corrector 1 DISCARDS corrector 0's increment. Summing here would report a
    growth the field never received, and PHASE_CLOT_MASS would close on it."""
    c = compact(src)
    return ("transferClotGrowth_+=" not in c
            and "transferClotGrowth_=(scalar(2)/skeletonVolumeFraction_)" in c
            and "transferClotGrowth_=0;" in c)


def transfer_growth_is_reset_per_step(src: str) -> bool:
    """Reset in solvePhaseField(), OUTSIDE the corrector loop: a step whose
    transfer never ran must report zero rather than the last step's value."""
    body = _function_body(
        src,
        "void Foam::solvers::phaseClotAxisymmetricFluid::solvePhaseField")
    reset = body.find("transferClotGrowth_ = 0;")
    loop = body.find("for (label corr = 0; corr < nPhaseCorrectors_; ++corr)")
    return 0 <= reset < loop


def mass_budget_closes_on_the_active_source(src: str) -> bool:
    """ONE DRIVER LEFT, SO NO SELECTION -- and the front integral must not
    creep back into the budget.

    Leaving INT Vn*phaseGradient in it gave a residual of ~1e-15 against a
    control whose worst was 5.93e-19, four orders out and otherwise silent.
    `phaseGrowthDriver front` is refused at construction since S6c-2, so the
    ternary that chose between them is gone and clotGrowth IS the transfer.
    clotGrowthFront is still computed and still reported -- under its own
    name, for the stoichiometry -- which is precisely why the budget has to
    be pinned to the other one.
    """
    c = compact(src)
    return ("constscalarclotGrowth=transferClotGrowth_;" in c
            and 'constscalarclotGrowth=(phaseGrowthDriver_=="transfer")'
                "?transferClotGrowth_:clotGrowthFront;" not in c)

def front_integral_kept_under_its_own_name(src: str) -> bool:
    """A number that no longer drives anything must not keep the name of the
    thing that drives. clotGrowthFront is still computed and still reported,
    so the 1.61x exchange rate stays visible after it stops acting."""
    c = compact(src)
    return ("constscalarclotGrowthFront=fvc::domainIntegrate"
            "(Vn_*phaseGradient_).value();" in c
            and '<<"clotGrowthFront="<<clotGrowthFront' in c)


def stoichiometry_stays_on_the_front(src: str) -> bool:
    """stoichRatio tests the FRONT closure and is the number every archived
    run reports. Feeding it the S4 growth would silently redefine a logged
    name and break that comparison while measuring something else."""
    c = compact(src)
    return ("?clotGrowthFront/plateletUptake" in c
            and "?clotGrowth/plateletUptake:" not in c)


# S21 (2026-10-05, docket 20261005-1621 report 15): growth_driver_validated_against_explicit_list() retired; its text was deleted by S21A c10.



# S21 (2026-10-05, docket 20261005-1621 report 15): growth_driver_requires_the_content_model() retired; its text was deleted by S21A a20.



def growth_driver_guards_its_divisor(src: str) -> bool:
    """SCOPED to its own message, for the reason the indicator's twin is:
    three sibling guards carry an identical `if (skeletonVolumeFraction_ <= 0)`
    and an unscoped search stays true when this one is deleted."""
    msg = "phaseGrowthDriver 'transfer' divides by"
    i = src.find(msg)
    if i < 0:
        return False
    return "if (skeletonVolumeFraction_ <= 0)" in src[max(0, i - 700):i]


# S21 (2026-10-05, docket 20261005-1621 report 15): growth_driver_requires_the_content_delta() retired; its text was deleted by S21A a21.



# S21 (2026-10-05, docket 20261005-1621 report 15): growth_driver_refuses_the_matched_operator() retired; its text was deleted by S21A a01.


def growth_driver_defaults_to_transfer(src: str) -> bool:
    """Alireza's standing direction: the new method is the default. Same
    two-place rule as phaseDeltaBasis -- see generator_writes_the_growth_key.
    """
    return '"phaseGrowthDriver", "transfer"' in src


def growth_driver_declared_in_header(hdr: str) -> bool:
    # S21A h22 deleted `word phaseGrowthDriver_;`.
    c = compact(hdr)
    return ("scalartransferClotGrowth_;" in c
            and "scalardBSumCommitted_;" in c
            and "scalardeltaThetaIntAtTransfer_;" in c)


def growth_driver_is_reported_at_startup(src: str) -> bool:
    """A run is read from its log, never from its JSON. Absent, or
    driver=front, makes an S4 run VOID rather than null."""
    c = compact(src)
    # S21A d16 substituted the built-in value into the startup line.
    return 'Info<<"PHASE_CLOT_GROWTH_DRIVER"<<"driver=transfer"' in c


# S21 (2026-10-05, docket 20261005-1621 report 15): generator_writes_the_growth_key() retired; its text was deleted by generator: release create_cases.py L4187-4244.



# S21 (2026-10-05, docket 20261005-1621 report 15): generator_refuses_the_same_combinations() retired; its text was deleted by generator: release create_cases.py L4229-4237.



def vn_stays_live_in_the_surface_loss(src: str) -> bool:
    """S4 DOES NOT DEMOTE Vn TO A DIAGNOSTIC, whatever the design text said.

    surfaceLossCoeff_ is built from Vn_ and enters the bound-platelet equation
    as fvm::Sp(surfaceLossCoeff_, w_) on every step. Deleting Vn_ along with
    the front would delete live physics.
    """
    c = compact(src)
    # S21A a39 deleted the legacy w_ equation's fvm::Sp(surfaceLossCoeff_, w_)
    # and S21C1 b28 the operator audit's two copies; the Vn_ term itself is kept.
    return ("surfaceLossCoeff_=interfaceDelta_*(adhesionSpeedEff_+Vn_"
            "-(U_&normal_));" in c)


def oi11_snapshot_is_taken_where_the_reaction_reads_it(src: str) -> bool:
    """deltaThetaIntPost was bit-identical to deltaThetaInt in all 272 records
    because both are postSolve values. The snapshot must sit in
    solvePlateletContent(), beside the bindRate it describes."""
    body = _function_body(
        src,
        "void Foam::solvers::phaseClotAxisymmetricFluid::solvePlateletContent")
    return ("deltaThetaIntAtTransfer_ = fvc::domainIntegrate(deltaTheta_)"
            ".value();" in body
            and "deltaThetaIntAtTransfer=" in src)


def oi12_committed_uptake_is_reported_beside_the_proxy(src: str) -> bool:
    """ADDED, NOT SUBSTITUTED. plateletUptake has no N in it and is blind to
    depletion; the gap between the two ratios IS the depletion. Replacing it
    would break comparison with every archived run."""
    c = compact(src)
    return ("constscalarplateletUptakeCommitted=plateletYield_"
            "*dBSumCommitted_/deltaT;" in c
            and '<<"plateletUptake="<<plateletUptake' in c
            and '<<"plateletUptakeCommitted="<<plateletUptakeCommitted' in c)


# ---- mutations ----------------------------------------------------------
#
# Each pair is (name, predicate, old, new). The mutation must be applied to
# the SOURCE the predicate reads, and the predicate must be true before and
# false after. A predicate that survives its own mutation is not a guard.

# The deltaTheta_ assignment in updateDerivedFields(), verbatim. Two
# mutations target it and both need the SAME text, so it is named once here
# rather than pasted twice and allowed to drift.
DELTA_THETA_ASSIGNMENT = (
    "        deltaTheta_ =\n"
    "            mag(fvc::grad(thetaC_ + thetaH_))\n"
    '           /dimensionedScalar("thetaRef", dimless,'
    " skeletonVolumeFraction_);"
)

MUTATIONS = [
    ("take the increment from Y*dB instead of the applied change",
     increment_uses_the_applied_theta_change,
     '        const volScalarField dTheta("dTheta", thetaNew - thetaStar);',
     '        const volScalarField dTheta("dTheta", depositedYield*dB);'),
    ("accumulate the growth across correctors instead of overwriting",
     transfer_growth_is_overwritten_not_accumulated,
     "        transferClotGrowth_ =\n            (scalar(2)/skeletonVolumeFraction_)",
     "        transferClotGrowth_ +=\n            (scalar(2)/skeletonVolumeFraction_)"),
    ("stop resetting the accumulator at the top of the step",
     transfer_growth_is_reset_per_step,
     "    transferClotGrowth_ = 0;\n\n    for (label corr = 0;",
     "    for (label corr = 0;"),
    ("stop reporting the front integral under its own name",
     front_integral_kept_under_its_own_name,
     "    const scalar clotGrowthFront =\n"
     "        fvc::domainIntegrate(Vn_*phaseGradient_).value();",
     "    const scalar clotGrowthFront = 0;"),
    ("feed the stoichiometry the S4 growth, silently redefining it",
     stoichiometry_stays_on_the_front,
     "          ? clotGrowthFront/plateletUptake",
     "          ? clotGrowth/plateletUptake"),
    ("stop guarding the growth driver's divisor",
     growth_driver_guards_its_divisor,
     '        if (skeletonVolumeFraction_ <= 0)\n        {\n            FatalErrorInFunction\n                << "phaseGrowthDriver \'transfer\' divides by"',
     '        if (false)\n        {\n            FatalErrorInFunction\n                << "phaseGrowthDriver \'transfer\' divides by"'),
    ("default the growth driver back to the front",
     growth_driver_defaults_to_transfer,
     '{"phaseGrowthDriver", "transfer"}',
     '{"phaseGrowthDriver", "front"}'),
    ("stop announcing the driver at startup",
     growth_driver_is_reported_at_startup,
     '    Info<< "PHASE_CLOT_GROWTH_DRIVER"\n        << " driver=transfer"',
     '    Info<< "PHASE_CLOT_GROWTH_DRIVER_X"\n        << " d=transfer"'),
    ("delete Vn from the surface loss along with the front",
     vn_stays_live_in_the_surface_loss,
     "    surfaceLossCoeff_ =\n        interfaceDelta_*(adhesionSpeedEff_ + Vn_ - (U_ & normal_));",
     "    surfaceLossCoeff_ =\n        interfaceDelta_*(adhesionSpeedEff_ - (U_ & normal_));"),
    ("take the OI-11 snapshot at postSolve, where it measures nothing",
     oi11_snapshot_is_taken_where_the_reaction_reads_it,
     "    deltaThetaIntAtTransfer_ = fvc::domainIntegrate(deltaTheta_).value();",
     "    // snapshot moved to writeBalanceDiagnostic()"),
    ("substitute the committed uptake for the proxy instead of adding it",
     oi12_committed_uptake_is_reported_beside_the_proxy,
     '            << " plateletUptake=" << plateletUptake\n            << " plateletUptakeCommitted=" << plateletUptakeCommitted',
     '            << " plateletUptake=" << plateletUptakeCommitted'),
    ("drop the theta_ref division -- the 0.45 rescaling",
     content_branch_divides_by_theta_ref,
     '           /dimensionedScalar\n            (\n                "thetaRef", dimless, skeletonVolumeFraction_\n            )',
     ""),
    ("clamp the ratio above, retiring the decision",
     indicator_is_not_clamped_above,
     '        return volScalarField::New\n        (\n            "phaseIndicator",\n            max\n            (\n                thetaC_ + thetaH_,',
     '        return volScalarField::New\n        (\n            "phaseIndicator",\n            min\n            (\n                thetaC_ + thetaH_,'),
    ("drop the lower guard at zero",
     indicator_is_guarded_below_at_zero,
     "            max\n            (\n                thetaC_ + thetaH_,\n                dimensionedScalar(\"zero\", dimless, 0)\n            )",
     "            (thetaC_ + thetaH_)"),
    ("let the accessor return the FRACTION instead",
     accessor_is_not_the_fraction,
     '            "phaseIndicator",\n            max\n            (\n                thetaC_ + thetaH_',
     '            "phaseIndicator",\n            max\n            (\n                phiSkel_'),
    ("stop guarding the divisor",
     content_guards_the_divisor,
     "        if (skeletonVolumeFraction_ <= 0)",
     "        if (false)"),
    ("move the deviatoric weight onto the unfloored indicator",
     deviatoric_weight_keeps_its_floor,
     "        phiS = skeletonStressWeight();\n    }\n",
     "        phiS = chi;\n    }\n"),
    ("threshold isSkel at zero instead of 0.5*residualChi_",
     isSkel_keeps_its_threshold,
     "          phaseIndicator()\n          - dimensionedScalar(\"halfChi\", dimless, 0.5*residualChi_)",
     "          phaseIndicator()"),
    ("leave the Kres ramp on the raw form",
     kres_ramp_reads_the_indicator,
     '        "chiReg",\n        phaseIndicator()',
     '        "chiReg",\n        max(1.0 - q_, dimensionedScalar("zero", dimless, 0))'),
    ("migrate only one of the two bond-hold sites",
     bond_hold_uses_one_basis_at_both_sites,
     "                    bondHold_*phaseIndicator()",
     '                    bondHold_*max(1.0 - q_, dimensionedScalar("zero", dimless, 0))'),
    ("let the instrument compare the indicator with itself",
     instrument_keeps_a_legacy_reference,
     "            const volScalarField chiLegacy\n            (\n                max(1.0 - q_, dimensionedScalar(\"zero\", dimless, 0))\n            );",
     "            const volScalarField chiLegacy(phaseIndicator());"),
    ("report the Kres set on the wrong threshold",
     instrument_reports_the_two_sets_that_move,
     "                if (ind[celli] < residualChi_)",
     "                if (ind[celli] < 0.0)"),
    ("fabricate a count where the branch does not apply",
     na_is_reported_rather_than_fabricated,
     "            if (residualChi_ <= 0)\n            {\n                kresCells = -1;\n                isSkelCells = -1;\n            }",
     ""),
    ("stop reporting the overshoot at all",
     instrument_reports_the_unclamped_overshoot,
     '                << " indAboveOne=" << indAboveOne',
     ""),
    ("drop the theta_ref normalisation from the surface delta",
     surface_delta_is_normalised_by_theta_ref,
     DELTA_THETA_ASSIGNMENT,
     "        deltaTheta_ = mag(fvc::grad(thetaC_ + thetaH_));"),
    ("hand-roll the gradient instead of using fvc::grad",
     surface_delta_uses_fvc_grad,
     DELTA_THETA_ASSIGNMENT,
     DELTA_THETA_ASSIGNMENT.replace("fvc::grad", "fvc::snGrad")),
    ("stop writing deltaTheta as a field",
     delta_theta_is_a_written_field,
     '            "deltaTheta", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE',
     '            "deltaTheta", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE'),
    ("let a momentum path read deltaTheta directly",
     delta_theta_reaches_equations_only_through_interfaceDelta,
     # S21: the old anchor `clotDrag_ = dragCoeff/rho_;` was deleted; the
     # second consumer is planted in elasticStress() instead.
     "    recordStressWeight(phiS.primitiveField(), 1);",
     "    recordStressWeight(phiS.primitiveField(), 1); phiS += 0.0*deltaTheta_;"),
    ("stop reporting the postSolve recomputation",
     instrument_reports_the_postsolve_recomputation,
     '                << " deltaThetaIntPost=" << deltaThetaIntPost',
     ""),
    ("stop guarding the delta basis divisor",
     delta_basis_guards_its_divisor,
     '        if (skeletonVolumeFraction_ <= 0)\n        {\n            FatalErrorInFunction\n                << "phaseDeltaBasis \'content\' divides by"',
     '        if (false)\n        {\n            FatalErrorInFunction\n                << "phaseDeltaBasis \'content\' divides by"'),
    ("default the delta basis back to legacy",
     delta_basis_defaults_to_content,
     '{"phaseDeltaBasis", "content"}',
     '{"phaseDeltaBasis", "legacy"}'),
    # ---- S6c-2: NINETEEN MUTATIONS REBUILT ----------------------------
    #
    # Every one of these used to mutate code the deletion removed, so the
    # subTest failed with `mutation target absent` -- not with a false
    # predicate, which is a different and much quieter kind of broken: the
    # guard was neither passing nor measuring. The predicates above are
    # ABSENCE assertions now, so their mutations REINTRODUCE the deleted
    # form, which is what a guard on a deletion has to catch -- the
    # deletion being quietly undone. Every anchor below is a line that is
    # live in the solver today, and the test asserts that before mutating.
    (
        'reintroduce frontSource, the S4 product whose equation is gone',
        front_is_built_zero_and_assigned_only_under_front,
        '    interfaceDelta_ = deltaTheta_;',
        '    interfaceDelta_ = deltaTheta_;\n    const volScalarField frontSource(Vn_*phaseGradient_);',
    ),
    (
        'put an ungated front term back into an equation',
        all_three_c_branches_read_the_gate,
        '    interfaceDelta_ = deltaTheta_;',
        '    interfaceDelta_ = deltaTheta_;\n          - Vn_*phaseGradient_',
    ),
    (
        'rebuild the front inside solvePhaseField',
        front_source_is_built_outside_the_corrector_loop,
        '    transferClotGrowth_ = 0;',
        '    transferClotGrowth_ = 0;\n    volScalarField frontSource(Vn_*phaseGradient_);',
    ),
    (
        'write to c again beside the transfer integral',
        increment_converts_by_two_over_theta_ref,
        '        const volScalarField dTheta("dTheta", thetaNew - thetaStar);',
        '        const volScalarField dTheta("dTheta", thetaNew - thetaStar);\n        c_ -= (scalar(2)/skeletonVolumeFraction_)*dTheta;',
    ),
    (
        'bring back the boundary correction for a field that is gone',
        increment_corrects_boundary_conditions,
        '        const volScalarField dTheta("dTheta", thetaNew - thetaStar);',
        '        const volScalarField dTheta("dTheta", thetaNew - thetaStar);\n        c_.correctBoundaryConditions();',
    ),
    (
        "select the budget's source again, on a driver that is refused",
        mass_budget_closes_on_the_active_source,
        '    const scalar clotGrowth = transferClotGrowth_;',
        '    const scalar clotGrowth =\n        (phaseGrowthDriver_ == "transfer")\n      ? transferClotGrowth_\n      : clotGrowthFront;',
    ),
    (
        'stop reporting the OI-11 snapshot',
        surface_delta_instrument_exists,
        '                << " deltaThetaIntAtTransfer=" << deltaThetaIntAtTransfer_',
        '                << " retiredInS6c2Too=1"',
    ),
    (
        'report deltaTV again, comparing theta with itself',
        surface_delta_reports_a_distribution_statistic,
        '                << " deltaThetaInt=" << deltaThetaInt',
        '                << " deltaThetaInt=" << deltaThetaInt\n                << " deltaTV=" << deltaTV',
    ),
    (
        'point the gate label at the retired statistic',
        surface_delta_names_the_ratio_as_reported_not_gated,
        '                << " gateIs=OI-11-deltaThetaIntAtTransfer"',
        '                << " gateIs=deltaTV"',
    ),
    (
        'declare deltaTV again',
        surface_delta_tv_differences_normalised_fields,
        '            scalar deltaThetaInt = -1;',
        '            scalar deltaThetaInt = -1;\n            scalar deltaTV = -1;',
    ),
    (
        'rebuild the half-L1 distance between the two deltas',
        surface_delta_tv_is_halved,
        '            scalar deltaThetaInt = -1;',
        '            scalar deltaThetaInt = -1;\n            deltaTV =\n                0.5*fvc::domainIntegrate(mag(dCn - dTn)).value();',
    ),
    (
        'turn the not-computable sentinel into a plausible number',
        surface_delta_guards_every_divisor,
        '            scalar deltaThetaIntPost = -1;',
        '            scalar deltaThetaIntPost = 0;',
    ),
    (
        'report the shift decomposition again',
        surface_delta_decomposes_moved_from_spread,
        '                << " deltaThetaInt=" << deltaThetaInt',
        '                << " deltaThetaInt=" << deltaThetaInt\n                << " deltaShiftMag=" << deltaShiftMag',
    ),
    (
        'rebuild deltaC_ so the instrument compares theta with itself',
        instrument_compares_deltaC_against_deltaTheta,
        '    interfaceDelta_ = deltaTheta_;',
        '    interfaceDelta_ = deltaTheta_;\n    deltaC_ = 0.5*magGradN;',
    ),
    (
        'make the delta assignment two-armed again',
        interface_delta_is_selected_by_the_key,
        '    interfaceDelta_ = deltaTheta_;',
        '    if (phaseDeltaBasis_ == "content")\n    {\n        interfaceDelta_ = deltaTheta_;\n    }\n    else\n    {\n        interfaceDelta_ = deltaC_;\n    }',
    ),
    (
        'put the legacy delta back into updateDerivedFields',
        delta_c_is_the_legacy_form,
        '    phaseGradient_ = sqrt(sqr(magGradN) + sqr(gradFloor_)) - gradFloor_;',
        '    phaseGradient_ = sqrt(sqr(magGradN) + sqr(gradFloor_)) - gradFloor_;\n    deltaC_ = 0.5*magGradN;',
    ),
    (
        'write deltaC as a field again',
        delta_c_is_a_written_field,
        '    interfaceDelta_ = deltaTheta_;',
        '    interfaceDelta_ = deltaTheta_;\n    IOobject\n    (\n        "deltaC", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE\n    );',
    ),
    (
        'move phaseGradient back onto |grad c|',
        phase_gradient_stays_on_c,
        '    phaseGradient_ = sqrt(sqr(magGradN) + sqr(gradFloor_)) - gradFloor_;',
        '    phaseGradient_ = sqrt(sqr(magGradC) + sqr(gradFloor_)) - gradFloor_;',
    ),
]



# S21 (2026-10-05, docket 20261005-1621 report 15): mutations whose anchor
# and predicate were deleted with the S21 built-in keys. Each name -> hunk.
S21_RETIRED_MUTATIONS = {
    "accept any spelling of the growth driver": "S21A c10",
    "stop refusing transfer on a dead content model": "S21A a20",
    "let S4 run on a legacy delta, i.e. skip S3": "S21A a21",
    "accept any spelling of the basis": "S21A c07",
    "stop refusing content on a dead content model": "S21A a16",
    "stop refusing unlocalised deposition": "S21A a17",
    "drop the carrier blend's own clamp": "S21A a29",
    "default the key to content": "S21A b10",
    "accept any spelling of the delta basis": "S21A c08",
    "stop refusing a content delta on a dead content model": "S21A a18",
    "refuse only `matched` instead of everything but `content`": "S21A a01",
}
assert not S21_RETIRED_MUTATIONS.keys() & {m[0] for m in MUTATIONS}

class IndicatorBasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SRC.read_text()
        cls.hdr = HDR.read_text()
        cls.gen = GEN.read_text() if GEN.exists() else ""

    def test_accessor_exists(self):
        self.assertTrue(accessor_exists(self.src))

    def test_the_substitution_is_theta_over_theta_ref(self):
        self.assertTrue(content_branch_divides_by_theta_ref(self.src),
                        "the content indicator must be theta/theta_ref; "
                        "returning theta rescales seven sites by 0.45")

    def test_content_branch_reads_the_content_fields(self):
        self.assertTrue(content_branch_reads_the_content_fields(self.src))

    def test_unclamped_above(self):
        self.assertTrue(indicator_is_not_clamped_above(self.src))

    def test_guarded_below(self):
        self.assertTrue(indicator_is_guarded_below_at_zero(self.src))

    def test_not_a_fraction(self):
        self.assertTrue(accessor_is_not_the_fraction(self.src))

    # S21: test_legacy_branch_unchanged retired (S21A a43).

    # S21: test_basis_validated retired (S21A c07).

    # S21: test_validation_runs_on_every_path retired (S21A c07).

    # S21: test_the_every_path_guard_fires_when_the_check_moves_into_the_gate retired (S21A c07).

    # S21: test_content_requires_the_model retired (S21A a16).

    def test_divisor_guarded(self):
        self.assertTrue(content_guards_the_divisor(self.src))

    # S21: test_unlocalised_refused retired (S21A a17).

    # S21: test_carrier_blend_clamp retired (S21A a29).

    def test_deviatoric_floor(self):
        self.assertTrue(deviatoric_weight_keeps_its_floor(self.src))

    def test_damage_square(self):
        self.assertTrue(damage_driver_keeps_its_square(self.src))

    def test_isSkel_threshold(self):
        self.assertTrue(isSkel_keeps_its_threshold(self.src))

    def test_kres_ramp(self):
        self.assertTrue(kres_ramp_reads_the_indicator(self.src))

    def test_bond_hold_both_sites(self):
        self.assertTrue(bond_hold_uses_one_basis_at_both_sites(self.src))

    def test_instrument(self):
        self.assertTrue(instrument_reports_the_unclamped_overshoot(self.src))

    def test_instrument_reports_the_moving_sets(self):
        self.assertTrue(instrument_reports_the_two_sets_that_move(self.src))

    def test_instrument_reports_na_not_a_fabricated_count(self):
        self.assertTrue(na_is_reported_rather_than_fabricated(self.src))

    def test_instrument_reference(self):
        self.assertTrue(instrument_keeps_a_legacy_reference(self.src))

    def test_surface_delta_instrument(self):
        self.assertTrue(surface_delta_instrument_exists(self.src))

    def test_surface_delta_normalisation(self):
        self.assertTrue(surface_delta_is_normalised_by_theta_ref(self.src))

    def test_surface_delta_uses_the_solver_operator(self):
        self.assertTrue(surface_delta_uses_fvc_grad(self.src))

    def test_surface_delta_reports_a_distribution(self):
        self.assertTrue(surface_delta_reports_a_distribution_statistic(self.src))

    def test_surface_delta_labels_the_gate(self):
        self.assertTrue(
            surface_delta_names_the_ratio_as_reported_not_gated(self.src))

    def test_surface_delta_tv_normalises_first(self):
        self.assertTrue(surface_delta_tv_differences_normalised_fields(self.src))

    def test_surface_delta_tv_is_a_fraction(self):
        self.assertTrue(surface_delta_tv_is_halved(self.src))

    def test_delta_theta_is_written(self):
        self.assertTrue(delta_theta_is_a_written_field(self.src))

    def test_surface_delta_divisors_are_guarded(self):
        self.assertTrue(surface_delta_guards_every_divisor(self.src))

    def test_surface_delta_decomposition(self):
        self.assertTrue(surface_delta_decomposes_moved_from_spread(self.src))

    def test_delta_theta_has_one_consumer_and_it_is_the_key(self):
        self.assertTrue(
            delta_theta_reaches_equations_only_through_interfaceDelta(self.src))

    def test_instrument_compares_the_two_candidates(self):
        self.assertTrue(instrument_compares_deltaC_against_deltaTheta(self.src))

    def test_instrument_reports_the_postsolve_recomputation(self):
        self.assertTrue(instrument_reports_the_postsolve_recomputation(self.src))

    def test_interface_delta_is_selected_by_the_key(self):
        self.assertTrue(interface_delta_is_selected_by_the_key(self.src))

    def test_delta_c_is_the_legacy_form(self):
        self.assertTrue(delta_c_is_the_legacy_form(self.src))

    def test_delta_c_is_written(self):
        self.assertTrue(delta_c_is_a_written_field(self.src))

    def test_phase_gradient_did_not_move(self):
        self.assertTrue(phase_gradient_stays_on_c(self.src))

    # S21: test_delta_basis_validated retired (S21A c08).

    # S21: test_delta_basis_requires_the_model retired (S21A a18).

    def test_delta_basis_divisor_guarded(self):
        self.assertTrue(delta_basis_guards_its_divisor(self.src))

    def test_delta_basis_defaults_to_content(self):
        self.assertTrue(delta_basis_defaults_to_content(self.src))

    def test_delta_basis_declared_in_header(self):
        self.assertTrue(delta_basis_declared_in_header(self.hdr))

    # S21: test_generator_writes_the_delta_key retired (generator cut).

    # S21: test_default_is_legacy retired (S21A b10).

    def test_header_declares_both(self):
        self.assertTrue(declared_in_header(self.hdr))

    # S21: test_generator_validates retired (generator cut).

    def test_not_migrated_sites_are_still_raw(self):
        self.assertTrue(not_migrated_sites_still_read_the_raw_form(self.src))

    def test_site_inventory_totals(self):
        self.assertTrue(every_indicator_site_is_one_of_two_forms(self.src))


    # ---- S4, the growth driver -------------------------------------------

    def test_front_is_never_formed_under_transfer(self):
        self.assertTrue(
            front_is_built_zero_and_assigned_only_under_front(self.src),
            "under 'transfer' the product Vn_*phaseGradient_ must not be "
            "formed at all; a zero multiplier turns one non-finite Vn into a "
            "nan applied to the whole equation")

    def test_all_three_c_branches_are_gated(self):
        self.assertTrue(
            all_three_c_branches_read_the_gate(self.src),
            "exactly three `- frontSource` terms and no surviving "
            "`- Vn_*phaseGradient_`; this case never takes the no-carrier "
            "branch, so a branch left ungated would be invisible in its log")

    def test_front_source_precedes_the_corrector_loop(self):
        self.assertTrue(front_source_is_built_outside_the_corrector_loop(
            self.src))

    def test_increment_is_the_applied_change(self):
        self.assertTrue(increment_uses_the_applied_theta_change(self.src),
                        "Y*dB is not the applied increment wherever "
                        "applyIncrement clamps against phiSMax_")

    def test_increment_conversion_factor(self):
        self.assertTrue(increment_converts_by_two_over_theta_ref(self.src),
                        "dc = -(2/theta_ref)*dtheta; 1/theta_ref misses the "
                        "step-1 gate by ~50%")

    def test_increment_corrects_boundaries(self):
        self.assertTrue(increment_corrects_boundary_conditions(self.src))

    def test_growth_is_overwritten_per_corrector(self):
        self.assertTrue(
            transfer_growth_is_overwritten_not_accumulated(self.src),
            "fvm::ddt(c_) discards the previous corrector's increment, so "
            "summing would report growth the field never received")

    def test_growth_is_reset_per_step(self):
        self.assertTrue(transfer_growth_is_reset_per_step(self.src))

    def test_mass_budget_follows_the_source(self):
        self.assertTrue(mass_budget_closes_on_the_active_source(self.src),
                        "a budget left on the front integral reports a "
                        "residual of ~1e-15 against a control worst of "
                        "5.93e-19, and nothing else notices")

    def test_front_integral_survives_under_its_own_name(self):
        self.assertTrue(front_integral_kept_under_its_own_name(self.src))

    def test_stoichiometry_is_not_silently_redefined(self):
        self.assertTrue(stoichiometry_stays_on_the_front(self.src))

    # S21: test_growth_driver_validated retired (S21A c10).

    # S21: test_growth_driver_requires_content_model retired (S21A a20).

    def test_growth_driver_guards_divisor(self):
        self.assertTrue(growth_driver_guards_its_divisor(self.src))

    # S21: test_growth_driver_requires_content_delta retired (S21A a21).

    # S21: test_growth_driver_refuses_matched retired (S21A a01).

    def test_growth_driver_defaults_to_transfer(self):
        self.assertTrue(growth_driver_defaults_to_transfer(self.src))

    def test_growth_driver_declared_in_header(self):
        self.assertTrue(growth_driver_declared_in_header(self.hdr))

    def test_growth_driver_announced(self):
        self.assertTrue(growth_driver_is_reported_at_startup(self.src))

    # S21: test_generator_writes_the_growth_key retired (generator cut).

    # S21: test_generator_refuses_the_same_combinations retired (generator cut).

    def test_vn_stays_live(self):
        self.assertTrue(vn_stays_live_in_the_surface_loss(self.src),
                        "S4 does not demote Vn: surfaceLossCoeff_ enters the "
                        "bound-platelet equation every step")

    def test_oi11_snapshot_location(self):
        self.assertTrue(oi11_snapshot_is_taken_where_the_reaction_reads_it(
            self.src))

    def test_oi11_predicate_is_sensitive_to_LOCATION_not_only_presence(self):
        """THE NEGATIVE TEST THE MUTATION LIST CANNOT EXPRESS.

        The listed mutation deletes the snapshot, which fails both halves of
        the predicate and therefore proves only that it notices an ABSENCE.
        OI-11 was never an absence -- deltaThetaIntPost existed, was emitted
        every step, and measured nothing because of WHERE it was computed. So
        this moves the statement, intact, into writeBalanceDiagnostic(), which
        is exactly the defect being guarded against, and requires the
        predicate to fail there.
        """
        snapshot = ("    deltaThetaIntAtTransfer_ = "
                    "fvc::domainIntegrate(deltaTheta_).value();")
        self.assertIn(snapshot, self.src)
        host = "    const scalar clotGrowthFront ="
        self.assertIn(host, self.src)
        moved = self.src.replace(snapshot + "\n", "", 1)
        moved = moved.replace(host, snapshot + "\n" + host, 1)
        self.assertNotEqual(moved, self.src)
        self.assertIn(snapshot, moved)          # still present, just wrong
        self.assertTrue(
            oi11_snapshot_is_taken_where_the_reaction_reads_it(self.src))
        self.assertFalse(
            oi11_snapshot_is_taken_where_the_reaction_reads_it(moved),
            "the guard must fail on a snapshot that is present but taken in "
            "the diagnostic -- that IS OI-11")

    def test_oi12_reports_both(self):
        self.assertTrue(oi12_committed_uptake_is_reported_beside_the_proxy(
            self.src))


class MutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SRC.read_text()

    def test_each_predicate_fires_on_its_mutation(self):
        for name, pred, old, new in MUTATIONS:
            with self.subTest(mutation=name):
                self.assertIn(old, self.src, f"{name}: mutation target absent")
                mutated = self.src.replace(old, new, 1)
                self.assertNotEqual(mutated, self.src,
                                    f"{name}: mutation did not apply")
                self.assertTrue(pred(self.src),
                                f"{name}: predicate already false")
                self.assertFalse(pred(mutated),
                                 f"{name}: predicate survived its own mutation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
