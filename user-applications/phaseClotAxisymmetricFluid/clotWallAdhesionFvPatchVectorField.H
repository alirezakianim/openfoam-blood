/*---------------------------------------------------------------------------*\
Class
    Foam::clotWallAdhesionFvPatchVectorField

Description
    Traction-separation wall condition for the poroelastic clot skeleton
    displacement Us. Models ADHESIVE failure of the clot/vessel-wall bond,
    which is how venous thrombi actually embolise: they peel off the wall,
    they do not crack through their own bulk.

    Why this exists. Until now the wall carried

        wall { type fixedValue; value uniform (0 0 0); }

    i.e. the skeleton was WELDED to the vessel. With that condition the only
    detachment path in the model is cohesive fracture through the clot
    interior, and run 20260809-113416 showed that path is unreachable: at the
    measured whole-clot modulus G = 58.6 Pa and toughness Gc = 5.55 J/m2 the
    AT2 localisation criterion H >= Gc/ell needs an elastic energy density of
    55474 J/m3, which for the oedometric modulus M = 135.7 Pa corresponds to a
    strain of about 29. A small-strain linear-elastic skeleton cannot deliver
    2900% strain, so damage saturated at Dg = 0.072 at dP = 4000 Pa and no
    material ever released. Ramanujam et al. (2024), the source of that Gc,
    say the same thing independently: predicting embolisation "requires a
    large-deformation, three-dimensional FE calculation that utilizes the
    nonlinear poroelastic constitutive model", and they too assume "the clot
    is perfectly bonded to the vessel wall", noting the answer "likely depends
    on the extent to which the clot is adhered to the vessel wall".

    That is the gap this condition fills, and the reason it is reachable where
    bulk fracture is not: the wall traction is an EQUILIBRIUM quantity. It does
    not have to be earned through strain. Statics on a fully occlusive plug
    gives a mean bond traction dP*r/(2L), and the wall-adjacent skeleton stress
    measured in run 20260809-113416 at dP = 4000 Pa already peaked at 778 Pa
    with consolidation only 2% complete.

    Constitutive law. RIGID-BRITTLE, deliberately:

        bonded    ->  fixedValue (0 0 0)          (valueFraction 1)
        |t| >= sigmaA, irreversibly ->
        debonded  ->  traction free               (valueFraction 0)

    with t = n & sigma the effective (skeleton) stress traction. A cohesive
    traction-separation spring would be smoother, but it needs an adhesive
    stiffness that NO source in research/literature/AM.2020.0056-19/sources/
    provides, and this model already carries more unsourced parameters than it
    should. The rigid-brittle law adds exactly one: sigmaA. If the debond front
    proves mesh-sensitive, that is a measurable finding and the spring is the
    documented repair -- not a knob to be turned in advance.

    sigmaA is UNSOURCED. Nothing in the local literature set gives a
    thrombus/vessel-wall adhesive strength. It is therefore a SWEPT parameter,
    and the deliverable is the strength REQUIRED for detachment, not a claim
    that any particular value is physiological.

    Only faces whose adjacent cell is inside the clot (q < clotThresholdQ) are
    eligible to fail. Elsewhere the skeleton is pinned to zero by the solver's
    outsidePin term anyway and a "debond" would be meaningless.

    THE INDICATOR IS q, NOT c. Until S6c-2 this condition read the phase
    field c and tested c < 0. S6c-2 deleted c from the solver, and this
    translation unit -- the library's ONLY user of the object registry -- was
    not migrated with it, which cost run 20260920-201014 at its first
    mechanics step. q = 1 - (thetaC + thetaH)/thetaRef is the theta-native
    liquid fraction; c was 2q - 1, so the map is exact and monotone,
    c < T  <->  q < (T + 1)/2, and the default threshold 0 becomes 0.5.
    The retired keys c and clotThreshold are REFUSED rather than ignored: a
    dictionary that still names them was written for the old indicator, and
    reading it as though it meant the new one would be a model change
    reported as a no-op.

    THE INJURY FOOTPRINT. Adhesion is confined to the wall faces that are
    clot-covered at the FIRST evaluation, latched once into injury_. Every
    other wall face is permanently non-adhesive.

    This is not a numerical convenience, it is the biology. A thrombus adheres
    where the endothelium is denuded and the subendothelial matrix -- collagen,
    tissue factor, von Willebrand factor -- is exposed to the blood; that
    exposure is what recruits and holds platelets. Intact endothelium is
    actively anti-thrombotic and offers nothing for the clot to hold onto.
    Modelling all 160 wall faces as adhesive therefore models an injury that
    covers the entire vessel, which is not the case being simulated.

    It also fixes an observed failure. Run 20260812-062701 debonded all 60
    faces, the clot began to travel, and then the total bonded+debonded count
    rose to 61 and 62: growth was advancing the clot footprint onto virgin wall
    faces, each of which arrived with bonded_ = 1 and full strength. Because
    the solver's support propagation treats ONE bonded face as anchoring every
    contiguous clot cell, each new bond arrested the whole body. The clot
    ratcheted downstream in stop-start fashion, welding itself to healthy
    endothelium as it slid.

    RE-ADHESION, and why it is allowed but narrow. Within the injury footprint
    the exposed matrix does not disappear when the bond breaks, so a clot that
    lifts off and later settles back onto the same denuded patch can genuinely
    re-adhere. That path is kept, gated on a face having actually lost its clot
    cover first (armed_):

        bonded, covered            -> may fail on traction
        debonded, still covered    -> stays debonded (no instant re-weld)
        debonded, cover lost       -> armed
        armed, cover returns       -> bonded again

    The intermediate "lost cover" state is what stops the law from collapsing
    into reversibility: without it a face would re-bond on the sweep after it
    broke, since the clot is still sitting on it, and no peel could ever
    propagate. Re-adhesion is reported separately from failure so it can never
    be mistaken for the peel running backwards.

    Sweep safety. The bond state is IRREVERSIBLE, and an irreversible ratchet
    evaluated inside a fixed-point iteration destroys the fixed point: a face
    that fails on sweep 3 of a not-yet-converged iterate stays failed even if
    the converged iterate would never have broken it. This is exactly the
    defect the solver's HStep_ field exists to prevent for the damage history.
    The same remedy is used here. bondedStep_ holds the state at the start of
    the time step; the solver calls restore() at the top of every exchange
    sweep and commit() once, after the exchange has converged.

Usage
    \table
        Property        | Description                      | Reqd | Default
        adhesiveStrength| bond failure traction [Pa]       | yes  |
        clotThresholdQ  | q below which a face is clot-covered | no | 0.5
        q               | name of the liquid-fraction field| no   | q
        sigma           | name of the skeleton stress field| no   | sigmaD
        bonded          | initial per-face bond state      | no   | 1
        injury          | adhesive footprint, 1 adhesive   | no   | latched
        armed           | per-face re-adhesion state       | no   | 0
    \endtable

    injury and armed are written to the time directory so a restart resumes
    with the footprint and re-adhesion state it had. Omitted, the footprint is
    latched from the clot cover at the first evaluation, which is what a fresh
    case wants.

    \verbatim
    wall
    {
        type             clotWallAdhesion;
        adhesiveStrength 500;
        value            uniform (0 0 0);
    }
    \endverbatim

SourceFiles
    clotWallAdhesionFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef clotWallAdhesionFvPatchVectorField_H
#define clotWallAdhesionFvPatchVectorField_H

#include "mixedFvPatchFields.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
             Class clotWallAdhesionFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class clotWallAdhesionFvPatchVectorField
:
    public mixedFvPatchVectorField
{
    // Private Data

        //- Bond failure traction [Pa]. Unsourced; swept.
        scalar adhesiveStrength_;

        //- A face is clot-covered when its adjacent cell has q below this.
        //  q is the liquid fraction, 1 in blood and 0 in a fully packed
        //  skeleton, so the test keeps the sense the old c < 0 test had.
        scalar clotThresholdQ_;

        //- Field names, so the condition is not hard-wired to the solver.
        word qName_;
        word sigmaName_;

        //- Per-face bond state, 1 bonded and 0 debonded. Written to the time
        //  directory so a restart resumes with the bond it had, and so
        //  ParaView can colour the peel front.
        scalarField bonded_;

        //- Bond state at the START of the time step. See the sweep-safety
        //  note in the class description: without this the irreversible
        //  ratchet is evaluated on unconverged iterates.
        scalarField bondedStep_;

        //- Largest |t| on a face that is STILL BONDED, i.e. that could still
        //  fail. This is the margin, and it is reported whether or not
        //  anything failed: a run that never debonds still has to say how
        //  close it came.
        scalar peakTraction_;

        //- Largest |t| over ALL clot-covered faces, debonded included. Kept
        //  for reference only. Run 20260810-010103 used this as the margin and
        //  reported utilisation 18.6 with nothing left that could fail.
        scalar peakTractionAny_;

        //- How many clot-covered faces are still bonded. Makes "nothing left
        //  to fail" distinguishable from "no load".
        label bondedClotFaces_;

        //- Largest NORMAL component |n.t| over bonded clot faces [Pa].
        //
        //  ADDED AFTER A WRONG CLAIM OF MINE, recorded so it is not repeated.
        //  Reading run 20260824-195730 I wrote that the Eulerian bond sees
        //  "no peeling traction at all" because sigmaElastic is deviatoric.
        //  THAT IS WRONG. The traction is t = n & sigma and its normal part is
        //  n.t = sigma_yy = phiS*gD*G*dev(be)_yy, which is nonzero whenever
        //  be_yy differs from tr(be)/3 -- deviatoric means TRACE-free, not
        //  normal-component-free.
        //
        //  What IS genuinely absent is the pressure, and that is the
        //  condition's own declared modelling choice: the bond carries the
        //  EFFECTIVE stress, not the total. And unlike the partitioned path's
        //  sigmaD, there is no lambda*tr(E)*I term -- not an omission but what
        //  incompressibility means, since the Eulerian skeleton holds
        //  det(be) = 1 and an incompressible skeleton has no volumetric
        //  elastic stress to contribute.
        //
        //  So the open question was never "is there a normal traction" but
        //  "how big is it", and that is a measurement. This is it.
        scalar peakTractionNormal_;

        //- Largest TANGENTIAL component |t - n(n.t)| over bonded clot faces.
        scalar peakTractionShear_;

        //- Was the named stress field actually in the registry on the last
        //  evaluation?
        //
        //  THIS EXISTS BECAUSE ITS ABSENCE HID A DEFECT FOR THE WHOLE
        //  PROJECT. updateCoeffs() legitimately does nothing before the first
        //  mechanics solve, when the stress does not exist yet. But a case
        //  that names a field which is NEVER created -- or, as on the
        //  Eulerian path, one that exists and is never written -- produces
        //  exactly the same log line as a case under no load at all:
        //  peakTractionAny 0, utilisation 0, nothing debonded. Run
        //  20260824-033102 read that for 20 ms and it meant the failure
        //  criterion was inert, not that the bond was unloaded.
        //
        //  Reported every step so "the bond never saw any load" and "the bond
        //  was never shown any stress" can never again be the same line.
        bool sigmaFound_;

        //- Adhesive footprint: 1 where the endothelium is denuded, 0 on
        //  intact wall. Latched from the clot cover at the first evaluation
        //  and never changed afterwards. See the injury-footprint note above.
        scalarField injury_;

        //- Re-adhesion state: 1 where a debonded injury face has since lost
        //  its clot cover and would re-adhere if the clot returned.
        scalarField armed_;

        //- armed_ at the START of the time step, for the same sweep-safety
        //  reason as bondedStep_. The state machine is a ratchet too.
        scalarField armedStep_;

        //- Has the footprint been latched yet?
        bool injuryLatched_;

        //- Faces that re-adhered during the step. Reported separately from
        //  failures so a re-bond can never read as the peel reversing.
        label reformedFaces_;


public:

    //- Runtime type information
    TypeName("clotWallAdhesion");


    // Constructors

        //- Construct from patch and internal field
        clotWallAdhesionFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        clotWallAdhesionFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping onto a new patch
        clotWallAdhesionFvPatchVectorField
        (
            const clotWallAdhesionFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fieldMapper&
        );

        //- Disallow copy without setting internal field reference
        clotWallAdhesionFvPatchVectorField
        (
            const clotWallAdhesionFvPatchVectorField&
        ) = delete;

        //- Copy constructor setting internal field reference
        clotWallAdhesionFvPatchVectorField
        (
            const clotWallAdhesionFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchVectorField> clone
        (
            const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new clotWallAdhesionFvPatchVectorField(*this, iF)
            );
        }


    // Member Functions

        // Access

            //- Per-face bond state, 1 bonded and 0 debonded
            const scalarField& bonded() const
            {
                return bonded_;
            }

            //- Largest traction on a still-BONDED clot face [Pa]: the margin
            scalar peakTraction() const
            {
                return peakTraction_;
            }

            //- Largest traction over all clot faces, debonded included [Pa]
            scalar peakTractionAny() const
            {
                return peakTractionAny_;
            }

            //- Clot-covered faces still bonded
            label bondedClotFaces() const
            {
                return bondedClotFaces_;
            }

            //- Peak normal (peeling) traction on a bonded clot face [Pa]
            scalar peakTractionNormal() const
            {
                return peakTractionNormal_;
            }

            //- Peak tangential (shearing) traction on a bonded clot face [Pa]
            scalar peakTractionShear() const
            {
                return peakTractionShear_;
            }

            //- Was the named stress field found in the registry?
            bool sigmaFound() const
            {
                return sigmaFound_;
            }

            //- The stress field this condition reads the traction from
            const word& sigmaName() const
            {
                return sigmaName_;
            }

            //- Adhesive footprint, 1 on denuded wall and 0 on intact wall
            const scalarField& injury() const
            {
                return injury_;
            }

            //- Faces that re-adhered during the last committed step
            label reformedFaces() const
            {
                return reformedFaces_;
            }

            //- Bond failure traction [Pa]
            scalar adhesiveStrength() const
            {
                return adhesiveStrength_;
            }


        // Bond-state control, called by the solver

            //- Discard failures decided on this exchange sweep. Called at the
            //  top of every FSI sweep so the fixed-point map is memoryless.
            void restore()
            {
                bonded_ = bondedStep_;
                armed_ = armedStep_;
            }

            //- Adopt the converged sweep's failures as this step's state.
            //  Returns the number of faces that failed during the step.
            //  Re-adhesions are counted separately into reformedFaces_ rather
            //  than netted off, so a step in which one face broke and another
            //  re-bonded reports 1 and 1, not 0.
            label commit()
            {
                label failed = 0;
                reformedFaces_ = 0;
                forAll(bonded_, facei)
                {
                    if (bondedStep_[facei] > 0.5 && bonded_[facei] < 0.5)
                    {
                        ++failed;
                    }
                    else if (bondedStep_[facei] < 0.5 && bonded_[facei] > 0.5)
                    {
                        ++reformedFaces_;
                    }
                }
                bondedStep_ = bonded_;
                armedStep_ = armed_;
                return failed;
            }


        // Mapping functions

            //- Map the given fvPatchField onto this fvPatchField
            virtual void map(const fvPatchVectorField&, const fieldMapper&);

            //- Reset the fvPatchField to the given fvPatchField
            virtual void reset(const fvPatchVectorField&);


        // Evaluation functions

            //- Update the coefficients
            virtual void updateCoeffs();


        //- Write
        virtual void write(Ostream&) const;
};


} // End namespace Foam

#endif

// ************************************************************************* //
