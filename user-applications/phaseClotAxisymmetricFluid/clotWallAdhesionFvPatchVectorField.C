/*---------------------------------------------------------------------------*\
    Traction-separation wall condition for the clot skeleton. See the header
    for the derivation and for why the law is rigid-brittle rather than
    cohesive.
\*---------------------------------------------------------------------------*/

#include "clotWallAdhesionFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "unitConversion.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::clotWallAdhesionFvPatchVectorField::clotWallAdhesionFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    mixedFvPatchVectorField(p, iF),
    adhesiveStrength_(great),
    clotThresholdQ_(0.5),
    qName_("q"),
    sigmaName_("sigmaD"),
    bonded_(p.size(), 1.0),
    bondedStep_(p.size(), 1.0),
    peakTraction_(0),
    peakTractionAny_(0),
    bondedClotFaces_(0),
    peakTractionNormal_(0),
    peakTractionShear_(0),
    sigmaFound_(false),
    injury_(p.size(), 1.0),
    armed_(p.size(), 0.0),
    armedStep_(p.size(), 0.0),
    injuryLatched_(false),
    reformedFaces_(0)
{
    refValue() = Zero;
    refGrad() = Zero;
    valueFraction() = 1.0;
    fvPatchVectorField::operator=(Zero);
}


Foam::clotWallAdhesionFvPatchVectorField::clotWallAdhesionFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    mixedFvPatchVectorField(p, iF, dict, false),
    // lookupOrDefault rather than lookup<scalar> purely because the former is
    // the idiom already proven to compile against this OpenFOAM build in
    // phaseClotAxisymmetricFluid.C. The entry is still effectively required:
    // the sentinel is rejected below.
    adhesiveStrength_(dict.lookupOrDefault<scalar>("adhesiveStrength", -1)),
    clotThresholdQ_(dict.lookupOrDefault<scalar>("clotThresholdQ", 0.5)),
    qName_(dict.lookupOrDefault<word>("q", "q")),
    sigmaName_(dict.lookupOrDefault<word>("sigma", "sigmaD")),
    bonded_
    (
        dict.found("bonded")
      ? scalarField("bonded", unitFraction, dict, p.size())
      : scalarField(p.size(), 1.0)
    ),
    bondedStep_(bonded_),
    peakTraction_(0),
    peakTractionAny_(0),
    bondedClotFaces_(0),
    peakTractionNormal_(0),
    peakTractionShear_(0),
    sigmaFound_(false),
    // A restart supplies the footprint it latched; a fresh case does not have
    // one yet and latches it at the first evaluation, once q is readable.
    injury_
    (
        dict.found("injury")
      ? scalarField("injury", unitFraction, dict, p.size())
      : scalarField(p.size(), 1.0)
    ),
    armed_
    (
        dict.found("armed")
      ? scalarField("armed", unitFraction, dict, p.size())
      : scalarField(p.size(), 0.0)
    ),
    armedStep_(armed_),
    injuryLatched_(dict.found("injury")),
    reformedFaces_(0)
{
    // THE RETIRED INDICATOR KEYS ARE REFUSED, NOT IGNORED. Before S6c-2
    // this condition read c and compared against clotThreshold, with c = 0
    // the blood/clot midpoint. c no longer exists. A dictionary that still
    // names either key was written for that indicator, and silently reading
    // it as though it named q would move the eligibility boundary without
    // saying so -- 0 is the midpoint in c and a nearly-packed skeleton in q.
    // No case on record writes either key, so this refuses nothing that
    // exists today; it refuses a stale one being revived.
    if (dict.found("c") || dict.found("clotThreshold"))
    {
        FatalIOErrorInFunction(dict)
            << "clotWallAdhesion no longer reads the phase field c: S6c-2"
            << " deleted it from the solver, and this condition now reads the"
            << " liquid fraction q (q = 1 - (thetaC + thetaH)/thetaRef)." << nl
            << "Replace `c <name>` with `q <name>` and `clotThreshold <T>`"
            << " with `clotThresholdQ <(T + 1)/2>`; the old default 0 becomes"
            << " 0.5."
            << exit(FatalIOError);
    }

    if (adhesiveStrength_ <= 0)
    {
        FatalIOErrorInFunction(dict)
            << "clotWallAdhesion requires a positive adhesiveStrength [Pa]; "
            << "got " << adhesiveStrength_
            << (dict.found("adhesiveStrength") ? "" : " (entry missing)") << nl
            << "A non-positive strength debonds the whole wall on the first "
            << "assembly, which is not a model of anything."
            << exit(FatalIOError);
    }

    refValue() = Zero;
    refGrad() = Zero;
    valueFraction() = bonded_;

    if (dict.found("value"))
    {
        fvPatchVectorField::operator=
        (
            vectorField("value", iF.dimensions(), dict, p.size())
        );
    }
    else
    {
        fvPatchVectorField::operator=(Zero);
    }
}


Foam::clotWallAdhesionFvPatchVectorField::clotWallAdhesionFvPatchVectorField
(
    const clotWallAdhesionFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fieldMapper& mapper
)
:
    mixedFvPatchVectorField(ptf, p, iF, mapper),
    adhesiveStrength_(ptf.adhesiveStrength_),
    clotThresholdQ_(ptf.clotThresholdQ_),
    qName_(ptf.qName_),
    sigmaName_(ptf.sigmaName_),
    bonded_(mapper(ptf.bonded_)),
    bondedStep_(mapper(ptf.bondedStep_)),
    peakTraction_(ptf.peakTraction_),
    peakTractionAny_(ptf.peakTractionAny_),
    bondedClotFaces_(ptf.bondedClotFaces_),
    peakTractionNormal_(ptf.peakTractionNormal_),
    peakTractionShear_(ptf.peakTractionShear_),
    sigmaFound_(ptf.sigmaFound_),
    injury_(mapper(ptf.injury_)),
    armed_(mapper(ptf.armed_)),
    armedStep_(mapper(ptf.armedStep_)),
    injuryLatched_(ptf.injuryLatched_),
    reformedFaces_(ptf.reformedFaces_)
{}


Foam::clotWallAdhesionFvPatchVectorField::clotWallAdhesionFvPatchVectorField
(
    const clotWallAdhesionFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    mixedFvPatchVectorField(ptf, iF),
    adhesiveStrength_(ptf.adhesiveStrength_),
    clotThresholdQ_(ptf.clotThresholdQ_),
    qName_(ptf.qName_),
    sigmaName_(ptf.sigmaName_),
    bonded_(ptf.bonded_),
    bondedStep_(ptf.bondedStep_),
    peakTraction_(ptf.peakTraction_),
    peakTractionAny_(ptf.peakTractionAny_),
    bondedClotFaces_(ptf.bondedClotFaces_),
    peakTractionNormal_(ptf.peakTractionNormal_),
    peakTractionShear_(ptf.peakTractionShear_),
    sigmaFound_(ptf.sigmaFound_),
    injury_(ptf.injury_),
    armed_(ptf.armed_),
    armedStep_(ptf.armedStep_),
    injuryLatched_(ptf.injuryLatched_),
    reformedFaces_(ptf.reformedFaces_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::clotWallAdhesionFvPatchVectorField::map
(
    const fvPatchVectorField& ptf,
    const fieldMapper& mapper
)
{
    mixedFvPatchVectorField::map(ptf, mapper);

    const clotWallAdhesionFvPatchVectorField& awptf =
        refCast<const clotWallAdhesionFvPatchVectorField>(ptf);

    mapper(bonded_, awptf.bonded_);
    mapper(bondedStep_, awptf.bondedStep_);
    mapper(injury_, awptf.injury_);
    mapper(armed_, awptf.armed_);
    mapper(armedStep_, awptf.armedStep_);
    injuryLatched_ = awptf.injuryLatched_;
}


void Foam::clotWallAdhesionFvPatchVectorField::reset
(
    const fvPatchVectorField& ptf
)
{
    mixedFvPatchVectorField::reset(ptf);

    const clotWallAdhesionFvPatchVectorField& awptf =
        refCast<const clotWallAdhesionFvPatchVectorField>(ptf);

    bonded_.reset(awptf.bonded_);
    bondedStep_.reset(awptf.bondedStep_);
    injury_.reset(awptf.injury_);
    armed_.reset(awptf.armed_);
    armedStep_.reset(awptf.armedStep_);
    injuryLatched_ = awptf.injuryLatched_;
}


void Foam::clotWallAdhesionFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // The skeleton stress is built inside solveSkeleton() and is not in the
    // registry before the first mechanics solve. Until it is, every face is
    // simply bonded -- which is the state the model had before this condition
    // existed, so nothing silently changes underneath a case that has not
    // reached its first mechanics step.
    sigmaFound_ = db().foundObject<volSymmTensorField>(sigmaName_);

    if (sigmaFound_)
    {
        const symmTensorField& sigmap =
            patch().lookupPatchField<volSymmTensorField, symmTensor>
            (
                sigmaName_
            );

        const scalarField qi
        (
            patch().lookupPatchField<volScalarField, scalar>
            (
                qName_
            ).patchInternalField()
        );

        // Effective (skeleton) stress traction. It is the skeleton that the
        // bond holds, so it is the effective stress and not the total stress
        // that the adhesive carries. Recorded as a declared modelling choice.
        // HOISTED. patch().nf() returns a tmp<vectorField>; calling it per
        // face inside the loop below would allocate the whole field once per
        // face and index a temporary. One named copy, used twice.
        const vectorField nHat(patch().nf());
        const vectorField t(nHat & sigmap);
        const scalarField magT(mag(t));

        // peakTraction_ is the margin: the largest traction on a face that is
        // STILL BONDED, i.e. that could still fail. Run 20260810-010103
        // measured it over all clot-covered faces including debonded ones and
        // reported utilisation climbing to 18.6 while newFaces stayed 0 -
        // every clot face had already released, so nothing could fail, yet the
        // headline number said the bond was eighteen times overloaded. A
        // debonded face is traction free in the BC sense but sigmaD at the
        // patch keeps its tangential part, so it never stops contributing.
        //
        // peakTractionAny_ retains the old all-clot-faces value for reference.
        // bondedClotFaces_ makes "nothing left to fail" explicit rather than
        // indistinguishable from "no load".
        // Latch the adhesive footprint on the first evaluation, which is the
        // earliest moment q is readable. Faces outside it are set permanently
        // debonded as well as non-adhesive: the solver's support propagation
        // seeds on bonded(), so leaving them at 1 is precisely what let growth
        // re-anchor the body on intact endothelium in run 20260812-062701.
        if (!injuryLatched_)
        {
            forAll(injury_, facei)
            {
                injury_[facei] = (qi[facei] < clotThresholdQ_) ? 1.0 : 0.0;
                if (injury_[facei] < 0.5)
                {
                    bonded_[facei] = 0.0;
                }
            }
            bondedStep_ = bonded_;
            armed_ = 0.0;
            armedStep_ = armed_;
            injuryLatched_ = true;
        }

        peakTraction_ = 0;
        peakTractionAny_ = 0;
        bondedClotFaces_ = 0;
        peakTractionNormal_ = 0;
        peakTractionShear_ = 0;

        forAll(bonded_, facei)
        {
            // Intact endothelium. Nothing adheres here, whatever grows over
            // it, so it can neither fail nor anchor.
            if (injury_[facei] < 0.5)
            {
                continue;
            }

            // Only clot-covered faces are eligible. Elsewhere the solver pins
            // the skeleton to zero and a debond would be meaningless.
            if (qi[facei] >= clotThresholdQ_)
            {
                // The clot has lifted clear of this denuded face. The exposed
                // matrix is still there, so arm it: if clot comes back down
                // on it, it adheres again.
                if (bonded_[facei] < 0.5)
                {
                    armed_[facei] = 1.0;
                }
                continue;
            }

            // Clot has settled back onto a face it had previously vacated.
            if (armed_[facei] > 0.5)
            {
                bonded_[facei] = 1.0;
                armed_[facei] = 0.0;
            }

            peakTractionAny_ = max(peakTractionAny_, magT[facei]);

            // Evaluated BEFORE the failure test, so a face that fails on this
            // assembly still contributes the traction that broke it.
            if (bonded_[facei] > 0.5)
            {
                ++bondedClotFaces_;
                peakTraction_ = max(peakTraction_, magT[facei]);

                // THE SPLIT. n.t is the peeling component and t - n(n.t) the
                // shearing one. Tracked separately because an adhesive bond
                // does not fail the same way under the two, and because the
                // claim that this condition sees no normal traction at all --
                // made when run 20260824-195730 was first read -- was wrong
                // and needs a measurement rather than another argument.
                const scalar tn = (nHat[facei] & t[facei]);
                peakTractionNormal_ = max(peakTractionNormal_, mag(tn));
                peakTractionShear_ = max
                (
                    peakTractionShear_,
                    mag(t[facei] - tn*nHat[facei])
                );

                if (magT[facei] >= adhesiveStrength_)
                {
                    bonded_[facei] = 0.0;
                }
            }
        }
    }

    // bonded -> fixedValue (0 0 0); debonded -> traction free.
    valueFraction() = bonded_;

    mixedFvPatchVectorField::updateCoeffs();
}


void Foam::clotWallAdhesionFvPatchVectorField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);

    writeEntry(os, "adhesiveStrength", adhesiveStrength_);

    if (clotThresholdQ_ != 0.5)
    {
        writeEntry(os, "clotThresholdQ", clotThresholdQ_);
    }
    if (qName_ != "q")
    {
        writeEntry(os, "q", qName_);
    }
    if (sigmaName_ != "sigmaD")
    {
        writeEntry(os, "sigma", sigmaName_);
    }

    // The bond state is the run's memory. Without it a restart silently
    // re-bonds a peeled clot.
    writeEntry(os, "bonded", bonded_);
    writeEntry(os, "injury", injury_);
    writeEntry(os, "armed", armed_);
    writeEntry(os, "value", *this);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        clotWallAdhesionFvPatchVectorField
    );
}


// ************************************************************************* //
