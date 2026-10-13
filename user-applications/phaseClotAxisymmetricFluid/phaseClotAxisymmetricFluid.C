/*---------------------------------------------------------------------------*\
  Runtime-selectable canonical solver for AM2020-phase-clot-v1 verification.

  The implementation follows the controlled specification, not the uncorrected
  Čapek (2020) equations. It is deliberately a serial-first research prototype
  and must pass the case verification ladder before scientific use.
\*---------------------------------------------------------------------------*/

#include "phaseClotAxisymmetricFluid.H"
#include "damagePartitionKernel.H"
#include "networkPressureKernel.H"
#include "fibreReleaseKernel.H"
#include "clotWallAdhesionFvPatchVectorField.H"
#include <cmath>   // std::isfinite, for the skeleton divergence guard
#include <algorithm>  // S23: nth_element, the R quantiles
#include <chrono>  // per-phase wall clock; see tSkeleton_ et al. in the header
#include <sstream> // S18p: the %.17g cell dump
#include <sys/resource.h>  // S22: getrusage, the record's rssGB

// The Eulerian elastic strain transport, SHARED UNCHANGED with the stage-1
// gate application that qualified it (run 20260816-221135: spatial order
// 1.960, temporal 1.995, SPD at principal stretch 3.271). Included rather
// than copied so the gate stays evidence about this solver.
#include "beTransport.H"
#include "plateletContent.H"  // Stage 2 operator, SHARED with plateletTransportGate
#include "contentDrag.H"     // Stage 3 drag driver, SHARED kernel + wrapper
#include "addToRunTimeSelectionTable.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "gaussGrad.H"
#include "surfaceInterpolate.H"
#include "harmonic.H"   // harmonic-mean face diffusivity; see solvePlatelets()
#include "fvcLaplacian.H"
#include "fvcSnGrad.H"
#include "fvcVolumeIntegrate.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "gaussConvectionScheme.H"  // Stage 1 content transport
#include "upwind.H"                 // ... with explicit upwind weights
// The seven headers correctPressure() needs. Taken from the include list of
// OpenFOAM-13's own applications/modules/incompressibleFluid/correctPressure.cpp
// rather than from recall: this file cannot be compiled on the machine that
// writes it, so a missing header is not caught until the Colab VM has already
// booted and installed its toolchain. fvcGrad/fvcSnGrad/fvmLaplacian above are
// already present and are not repeated.
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "adjustPhi.H"
#include "fvcMeshPhi.H"
#include "fvcFlux.H"
#include "fvcDdt.H"
#include "fvcReconstruct.H"
#include "fvmSup.H"
#include "PstreamReduceOps.H"
#include "Switch.H"
#include "zeroGradientFvPatchFields.H"
#include "slipFvPatchFields.H"      // S10: skeleton wall condition
#include "fixedValueFvPatchFields.H"
#include "gaussConvectionScheme.H"
#include "surfaceInterpolationScheme.H"

// quiet_NaN() for the diagnostics that must report an undefined ratio
// rather than fabricate one or trap under FOAM_SIGFPE. See the D = 0
// handling in updateDerivedFields().
#include <limits>

namespace
{

//- A surfaceInterpolationScheme whose weights are SUPPLIED rather than
//  computed from the field being interpolated.
//
//  THE WHOLE POINT OF THIS CLASS. vanLeer weights are a function of the
//  TRANSPORTED FIELD. c and Wp are different fields, so `div(phiReleased,c)`
//  and `div(phiReleased,Wp)` assembled DIFFERENT matrices even when the
//  scheme strings were token-identical and the flux was one shared object.
//  Run 20260822-103655 measured the consequence directly: |Wp - q| = 2.27e-04
//  with vanLeer against 5.85e-12 with `linear`, whose weights are
//  field-independent. Eight orders, on one changed line.
//
//  W = q*w and q = 0.5*(c+1), so Wp = q is preserved by exact arithmetic ONLY
//  if the two equations share face weights. Handing c's weights to Wp is what
//  this class is for. It ignores the field argument by design -- that argument
//  is precisely the thing that must not influence the weights.
//
//  corrected() is left at the base default of false, matching vanLeer: a
//  limiter acts through its weights and contributes no explicit correction.
class suppliedWeightsScheme
:
    public Foam::surfaceInterpolationScheme<Foam::scalar>
{
    //- Renamed copy, not a reference. The source is a tmp owned by the solver
    //  and this object is owned by a tmp inside gaussConvectionScheme; a
    //  reference would couple two lifetimes that are not managed together.
    //  The rename keeps it from colliding with its source in the registry.
    const Foam::surfaceScalarField weights_;

public:

    suppliedWeightsScheme
    (
        const Foam::fvMesh& mesh,
        const Foam::surfaceScalarField& w
    )
    :
        Foam::surfaceInterpolationScheme<Foam::scalar>(mesh),
        weights_("suppliedWeights", w)
    {}

    virtual Foam::tmp<Foam::surfaceScalarField> weights
    (
        const Foam::volScalarField&
    ) const
    {
        return weights_;
    }
};

} // End anonymous namespace
#include "wallFvPatch.H"

namespace
{
    // S25 F (design 2026-10-10-S25 section 2F): timeScale = S_t, read with
    // NO default. Its consumers are networkRupture fibreRelease (t_rel;
    // built in since S21, so every dictionary) and networkViscosity
    // kelvinVoigt (tau_v and its floor), so it is required on every
    // dictionary. Refused if missing, non-finite or <= 0. A free function so
    // the member-initialiser list holds no braces.
    Foam::scalar s25TimeScale(const Foam::dictionary& props)
    {
        using namespace Foam;
        if (!props.found("timeScale"))
        {
            FatalErrorInFunction
                << "phaseClotProperties key `timeScale` is missing. It is "
                << "S_t = t_sim/t_phys and has no default: networkRupture "
                << "fibreRelease (built in) consumes t_rel, and "
                << "networkViscosity kelvinVoigt consumes tau_v and its "
                << "floor, all physical times in seconds. State timeScale "
                << "1.0 for an unscaled (legacy) series (design "
                << "2026-10-10-S25 section 2F)." << exit(FatalError);
        }
        const scalar ts = props.lookup<scalar>("timeScale");
        if (!(std::isfinite(ts) && ts > 0))
        {
            FatalErrorInFunction
                << "phaseClotProperties key `timeScale` = " << ts
                << " must be finite and > 0: it is the ratio S_t = "
                << "t_sim/t_phys that scales t_rel, tau_v and the tau_v "
                << "floor (design 2026-10-10-S25 section 2F)."
                << exit(FatalError);
        }
        return ts;
    }

    // S25 A (design 2026-10-10-S25 sections 2A, 3A, 5): skeletonLaw,
    // default "shipped". Free functions, so the initialiser list holds no
    // braces; each refusal names its key and its reason.
    Foam::word s25SkeletonLaw(const Foam::dictionary& props)
    {
        using namespace Foam;
        const word law =
            props.lookupOrDefault<word>("skeletonLaw", "shipped");
        if (law != "shipped" && law != "valanisLandelTrilinear")
        {
            FatalErrorInFunction
                << "phaseClotProperties key `skeletonLaw` = " << law
                << " is not one of shipped (G dev(be), the default) or "
                << "valanisLandelTrilinear (design 2026-10-10-S25 section "
                << "2A)." << exit(FatalError);
        }
        return law;
    }

    // skeletonLawCoeffs {E1 E2 D1 D2 kappa}: read ONLY under LT (under the
    // shipped law nothing new is evaluated, F1). Each value present and
    // finite; E1, E2 > 0; 0 <= D1 < D2 (the segment table's breaks 1 -+ D2,
    // 1 -+ D1 must be ordered); kappa > 0 (K0 = kappa E1/2).
    ::skeletonLawKernel::Coeffs s25SkeletonLawCoeffs
    (
        const Foam::dictionary& props,
        const bool lt
    )
    {
        using namespace Foam;
        skeletonLawKernel::Coeffs c = {0, 0, 0, 0, 0};
        if (!lt)
        {
            return c;
        }
        if (!props.isDict("skeletonLawCoeffs"))
        {
            FatalErrorInFunction
                << "skeletonLaw valanisLandelTrilinear needs the subdictionary "
                << "`skeletonLawCoeffs {E1 E2 D1 D2 kappa;}` (design "
                << "2026-10-10-S25 section 2A); it is missing."
                << exit(FatalError);
        }
        const dictionary& d = props.subDict("skeletonLawCoeffs");
        const char* const names[5] = {"E1", "E2", "D1", "D2", "kappa"};
        scalar v[5] = {0, 0, 0, 0, 0};
        for (label k = 0; k < 5; ++k)
        {
            if (!d.found(names[k]))
            {
                FatalErrorInFunction
                    << "skeletonLawCoeffs key `" << names[k] << "` is "
                    << "missing; skeletonLaw valanisLandelTrilinear has no "
                    << "default for any of E1 E2 D1 D2 kappa (design "
                    << "2026-10-10-S25 section 5)." << exit(FatalError);
            }
            v[k] = d.lookup<scalar>(names[k]);
            if (!std::isfinite(v[k]))
            {
                FatalErrorInFunction
                    << "skeletonLawCoeffs key `" << names[k] << "` = "
                    << v[k] << " is not finite (design 2026-10-10-S25 "
                    << "section 5)." << exit(FatalError);
            }
        }
        if (!(v[0] > 0) || !(v[1] > 0))
        {
            FatalErrorInFunction
                << "skeletonLawCoeffs E1 = " << v[0] << ", E2 = " << v[1]
                << ": both moduli must be > 0 (design 2026-10-10-S25 "
                << "section 5)." << exit(FatalError);
        }
        if (!(v[2] < v[3]))
        {
            FatalErrorInFunction
                << "skeletonLawCoeffs D1 = " << v[2] << ", D2 = " << v[3]
                << ": D1 >= D2 is refused; the blend between them needs "
                << "D1 < D2 (design 2026-10-10-S25 section 5)."
                << exit(FatalError);
        }
        if (!(v[2] >= 0))
        {
            FatalErrorInFunction
                << "skeletonLawCoeffs D1 = " << v[2] << " < 0: the toe "
                << "is |lambda - 1| < D1, so the breaks 1 - D1 and 1 + D1 "
                << "would be out of order and the tau segment table "
                << "undefined." << exit(FatalError);
        }
        if (!(v[4] > 0))
        {
            FatalErrorInFunction
                << "skeletonLawCoeffs kappa = " << v[4] << " <= 0 is "
                << "refused: K0 = kappa E1/2 is the bulk modulus of the "
                << "pole law at s'(1), and kappa <= 0 removes or inverts it "
                << "(design 2026-10-10-S25 section 5, rev. 2.1; the "
                << "generator refuses it too)." << exit(FatalError);
        }
        c.E1 = v[0];
        c.E2 = v[1];
        c.D1 = v[2];
        c.D2 = v[3];
        c.kappa = v[4];
        return c;
    }

    // The tauHat table, built only under LT (makeLaw divides by E1 and by
    // D2 - D1, both refused above when they are not > 0).
    ::skeletonLawKernel::Law s25SkeletonLawTable
    (
        const ::skeletonLawKernel::Coeffs& c,
        const bool lt
    )
    {
        ::skeletonLawKernel::Law L =
            {{0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
        if (lt)
        {
            L = ::skeletonLawKernel::makeLaw(c);
        }
        return L;
    }

    // elasticCourantModulus: "tangent" by default under LT, "constant"
    // under shipped; tangent under shipped is refused (it would move the
    // wave CFL off G0 and break F1).
    Foam::word s25ElasticCourantModulus
    (
        const Foam::dictionary& props,
        const bool lt
    )
    {
        using namespace Foam;
        const word m =
            props.lookupOrDefault<word>
            (
                "elasticCourantModulus", lt ? "tangent" : "constant"
            );
        if (m != "constant" && m != "tangent")
        {
            FatalErrorInFunction
                << "phaseClotProperties key `elasticCourantModulus` = " << m
                << " is not one of constant (G0) or tangent (max over cells "
                << "of G (3/4) stiff) (design 2026-10-10-S25 section 3A)."
                << exit(FatalError);
        }
        if (m == "tangent" && !lt)
        {
            FatalErrorInFunction
                << "elasticCourantModulus tangent is refused under "
                << "skeletonLaw shipped: the shipped law's wave CFL reads G0, "
                << "and moving it would break F1 (design 2026-10-10-S25 "
                << "section 5). Use constant, or skeletonLaw "
                << "valanisLandelTrilinear." << exit(FatalError);
        }
        return m;
    }
}

namespace
{
    // Monotonic wall clock in seconds. std::chrono rather than Foam::clockTime
    // because clockTime measures from ITS OWN construction, which makes it
    // awkward to accumulate across separate call sites in one time step.
    inline Foam::scalar phaseClotWallNow()
    {
        return std::chrono::duration<Foam::scalar>
        (
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    // Scope guard, not a matched pair of calls. solveSkeleton() and
    // solveDamage() both have early returns; hand-placed stop calls would
    // silently under-count exactly the paths that matter most.
    class phaseClotScopeTimer
    {
        Foam::scalar& acc_;
        const Foam::scalar t0_;

    public:

        explicit phaseClotScopeTimer(Foam::scalar& acc)
        :
            acc_(acc),
            t0_(phaseClotWallNow())
        {}

        ~phaseClotScopeTimer()
        {
            acc_ += phaseClotWallNow() - t0_;
        }

        phaseClotScopeTimer(const phaseClotScopeTimer&) = delete;
        void operator=(const phaseClotScopeTimer&) = delete;
    };
}


namespace Foam
{
namespace solvers
{
    defineTypeNameAndDebug(phaseClotAxisymmetricFluid, 0);
    addToRunTimeSelectionTable
    (
        solver,
        phaseClotAxisymmetricFluid,
        fvMesh
    );
}
}


Foam::solvers::phaseClotAxisymmetricFluid::phaseClotAxisymmetricFluid
(
    fvMesh& mesh
)
:
    incompressibleFluid(mesh),

    phaseClotProperties_
    (
        IOobject
        (
            "phaseClotProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    ),

    w_
    (
        IOobject
        (
            "w", runTime.name(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE
        ),
        mesh
    ),

    // Copy-constructed from w_ so it inherits w's boundary TYPES (inlet
    // fixedValue 1, zeroGradient elsewhere). The VALUES are overwritten with
    // q*w in the constructor body once q exists. Section 3.5 of the derivation.
    Wp_
    (
        IOobject("Wp", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        w_
    ),
    // S25 B (R1): e = Wp - q, on w's boundary TYPES with value 0 (the
    // inlet's fixedValue becomes 0: q = 1 and Wp = 1 there). Read on a
    // restart; otherwise set from today's seed in the constructor body.
    e_
    (
        IOobject
        (
            "e", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", w_.dimensions(), 0),
        w_.boundaryField().types()
    ),

    q_
    (
        IOobject("q", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("one", dimless, 1)
    ),
    normal_
    (
        IOobject
        (
            "phaseNormal", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector("zero", dimless, Zero)
    ),
    phaseGradient_
    (
        IOobject
        (
            "phaseGradient", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless/dimLength, 0)
    ),
    interfaceDelta_
    (
        IOobject
        (
            "interfaceDelta", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless/dimLength, 0)
    ),
    deltaTheta_
    (
        IOobject
        (
            "deltaTheta", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless/dimLength, 0)
    ),
    shearPa_
    (
        IOobject
        (
            "shearPa", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimPressure, 0)
    ),
    shearRate_
    (
        IOobject
        (
            "shearRate", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless/dimTime, 0)
    ),
    adhesionEnhancement_
    (
        IOobject
        (
            "adhesionEnhancement", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("one", dimless, 1)
    ),
    adhesionSpeed_
    (
        IOobject
        (
            "adhesionSpeed", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimVelocity, 0)
    ),
    adhesionSpeedEff_
    (
        IOobject
        (
            "adhesionSpeedEff", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimVelocity, 0)
    ),
    Vn_
    (
        IOobject
        (
            "Vn", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimVelocity, 0)
    ),
    lambda_
    (
        IOobject
        (
            "clotPenalty", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless/dimTime, 0)
    ),
    surfaceLossCoeff_
    (
        IOobject
        (
            "plateletLossCoeff", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless/dimTime, 0)
    ),

    Us_
    (
        IOobject("Us", runTime.name(), mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
        mesh,
        dimensionedVector("zero", dimLength, Zero),
        zeroGradientFvPatchField<vector>::typeName
    ),
    UsPrev_
    (
        IOobject("UsPrev", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedVector("zero", dimLength, Zero),
        zeroGradientFvPatchField<vector>::typeName
    ),
    Dg_
    (
        IOobject("Dg", runTime.name(), mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    H_
    (
        IOobject("H", runTime.name(), mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("zero", dimPressure, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    poroPerm_
    (
        IOobject("poroPerm", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("zero", dimArea, 1e-14)
    ),
    // THE FIELD TO LOOK AT IN PARAVIEW, and the reason it is written.
    //
    // poroPerm is Kozeny-Carman, k ~ q^3/(1-q)^2, so it DIVERGES as q -> 1.
    // In run 20260812-100157 it spanned 2.2e-16 m2 in the clot core to 315 m2
    // in the blood: nineteen decades. On a linear colour map that is
    // unreadable, and it reads BACKWARDS -- the blood is the red end, because
    // free fluid is the infinitely permeable limit, not the impermeable one.
    // It was misread as "the whole domain is porous and the inlet has no
    // permeability", which is the exact opposite of what it says.
    //
    // clotDrag is the quantity that actually appears in the momentum
    // equation: chi*nu/k, with chi = 1 - q. It is zero in blood by BOTH
    // factors, so it shows the resisting region and nothing else.
    //
    // KINEMATIC (1/s), matching lambda_ and matching what fvm::Sp(coeff, U)
    // means in the kinematic momentum equation incompressibleFluid writes.
    // See the header for the dynamic-vs-kinematic defect this replaced.
    clotDrag_
    (
        IOobject("clotDrag", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("zero", dimless/dimTime, 0)
    ),
    released_
    (
        IOobject("released", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("zero", dimless, 0.0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    // Zero until the first prePredictor() sets it. A run with nothing
    // released never leaves this value, and the relative-velocity drag then
    // reduces exactly to the lab-frame form the previous builds assembled.
    UbodyCached_(Zero),
    bondHold_
    (
        IOobject("bondHold", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar("zero", dimless, 0.0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    sigmaD_
    (
        IOobject("sigmaD", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedSymmTensor("zero", dimPressure, Zero)
    ),
    // AUTO_WRITE: the whole point of this field is that the bond's view of the
    // stress can be inspected, and a log number that cannot be looked at in
    // ParaView is how the zero traction went unexamined in the first place.
    sigmaElastic_
    (
        IOobject
        (
            "sigmaElastic", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedSymmTensor("zero", dimPressure, Zero)
    ),

    be_
    (
        // READ_IF_PRESENT with AUTO_WRITE: be is STATE. A restart that lost it
        // would silently resume from an unstrained clot, which is the kind of
        // failure that looks like a physical relaxation rather than a bug.
        // Written for the same reason, and so ParaView can show the strain.
        IOobject
        (
            "be", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedSymmTensor("I", dimless, symmTensor::I),
        zeroGradientFvPatchField<symmTensor>::typeName
    ),
    beRelax_
    (
        IOobject("beRelax", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("zero", dimless/dimTime, 0.0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    beRelaxRate_
    (
        // 1/tau in blood. Large enough that be returns to I within a timestep
        // there -- the fluid must not remember strain -- and applied through a
        // q-weighting so the clot side sees essentially zero. Default sized
        // against the main case's dt of 7.4e-6 s: 1e6 1/s gives tau = 1 us,
        // seven times shorter, so blood relaxes fully between steps.
        phaseClotProperties_.lookupOrDefault<scalar>("beRelaxRate", 1.0e6)
    ),
    beCorrectors_
    (
        phaseClotProperties_.lookupOrDefault<label>("beCorrectors", 3)
    ),
    elasticCourant_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("elasticCourant", 0.5)
    ),
    // 1/16 is the standard 2-D coefficient for an explicit fourth-order
    // operator. Exposed rather than hard-coded so a case that wants the
    // stricter 1/32 can say so, but NOT defaulted loose: the failure this
    // guards against is silent.
    biharmonicCourant_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "biharmonicCourant", 1.0/16.0
        )
    ),
    // 0.2 of tau_kv. Chosen against the measurement rather than picked: the
    // slab gate's worst-behaved child ran at dt/tau_kv = 13.5 and its best at
    // 1.35, so the acceptable region is well below 1, and production already
    // sits at 0.0085-0.085 so this default costs it nothing.
    elasticRelaxCourant_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("elasticRelaxCourant", 0.2)
    ),
    beSweepChange_(0.0),
    beMinMinor_(1.0),
    beMaxStretch_(1.0),
    beDetError_(0.0),
    beDetErrorPost_(0.0),

    rho_(phaseClotProperties_.lookup("rho")),
    nu_(phaseClotProperties_.lookup("nu")),
    D_(phaseClotProperties_.lookup("D")),
    alpha_(phaseClotProperties_.lookup("alpha")),
    k0_(phaseClotProperties_.lookup("k0")),
    etaP_(phaseClotProperties_.lookup("etaP")),
    gradFloor_(phaseClotProperties_.lookup("gradFloor")),
    qFloor_(phaseClotProperties_.lookup("qFloor")),
    penaltyExponent_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("penaltyExponent", 1)
    ),
    // Read as plain scalars in 1/s and dimension them here: v13 has no
    // dictionary lookupOrDefault overload for dimensionedScalar.
    gammaLow_
    (
        "gammaLow",
        dimless/dimTime,
        phaseClotProperties_.lookupOrDefault<scalar>("gammaLow", 3000)
    ),
    gammaHigh_
    (
        "gammaHigh",
        dimless/dimTime,
        phaseClotProperties_.lookupOrDefault<scalar>("gammaHigh", 8000)
    ),
    enhancementMax_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("enhancementMax", 20)
    ),
    G_("G", dimPressure, phaseClotProperties_.lookupOrDefault<scalar>("G", 58.6)),
    poissonRatio_(phaseClotProperties_.lookupOrDefault<scalar>("poissonRatio", 0.12)),
    lambdaLame_
    (
        "lambdaLame",
        dimPressure,
        2.0*phaseClotProperties_.lookupOrDefault<scalar>("G", 58.6)
       *phaseClotProperties_.lookupOrDefault<scalar>("poissonRatio", 0.12)
       /(1.0 - 2.0*phaseClotProperties_.lookupOrDefault<scalar>("poissonRatio", 0.12))
    ),
    k0p_("k0p", dimArea, phaseClotProperties_.lookupOrDefault<scalar>("k0p", 3.5e-14)),
    Gc_("Gc", dimPressure*dimLength, phaseClotProperties_.lookupOrDefault<scalar>("Gc", 1.0)),
    ell_("ell", dimLength, phaseClotProperties_.lookupOrDefault<scalar>("ell", 2.6e-4)),
    etaRes_(phaseClotProperties_.lookupOrDefault<scalar>("etaRes", 1e-6)),
    DgDetach_(phaseClotProperties_.lookupOrDefault<scalar>("DgDetach", 0.95)),
    fsiRelaxation_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("fsiRelaxation", 1.0)
    ),
    releaseSupportSweeps_
    (
        phaseClotProperties_.lookupOrDefault<label>("releaseSupportSweeps", 400)
    ),
    releaseSupportLimit_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("releaseSupportLimit", 0.0)
    ),
    bondHoldDilation_
    (
        phaseClotProperties_.lookupOrDefault<label>("bondHoldDilation", -1)
    ),
    bondHoldPenalty_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("bondHoldPenalty", 0.0)
    ),
    releaseCoreExempt_
    (
        phaseClotProperties_.lookupOrDefault<Switch>("releaseCoreExempt", true)
    ),
    skelElasticForce_
    (
        IOobject
        (
            "skelElasticForce", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector("zero", dimAcceleration, Zero)
    ),
    skelPressForce_
    (
        IOobject
        (
            "skelPressForce", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector("zero", dimAcceleration, Zero)
    ),
    Vs_
    (
        IOobject
        (
            "Vs", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector("zero", dimVelocity, Zero),
        // ZEROGRADIENT IS LOAD-BEARING, NOT COSMETIC. Vs_ used to be built
        // without an explicit patch type, which gives CALCULATED patch fields.
        // That was harmless while Vs_ was only ever assigned from an algebraic
        // expression. It is fatal now that Vs_ is the unknown of an fvm matrix:
        // calculatedFvPatchField::valueInternalCoeffs() raises a FatalError, so
        // fvm::div(phiSf, Vs_) would abort on the first step.
        //
        // zeroGradient rather than fixedValue at walls is deliberate. A clot
        // bonded to the wall is held by the bond-hold term in the skeleton
        // momentum equation, which exists only where a bond exists and
        // vanishes the instant it fails. A hard no-slip BC would instead pin
        // the material to the wall permanently and a released clot could never
        // slide away -- the same objection that removed the lab-frame Darcy
        // term from this path. No-penetration is enforced separately, and
        // where it belongs, by skeletonFlux().
        zeroGradientFvPatchField<vector>::typeName
    ),
    skeletonVolumeFraction_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "skeletonVolumeFraction", 0.01
        )
    ),
    // Declared, not fitted. 1e-17 m3 is an order-of-magnitude platelet volume
    // and is a PLACEHOLDER for Stage 1, where kinetics are off and vP
    // therefore multiplies nothing. It must be sourced before Stage 2 gives
    // it any authority over a transfer.
    vP_(phaseClotProperties_.lookupOrDefault<scalar>("vP", 1e-17)),
    residualPhiS_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("residualPhiS", 1e-4)
    ),
    residualChi_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("residualChi", 0.0)
    ),
    networkPressureCoeff_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "networkPressureCoeff", 1.0
        )
    ),
    phiSMax_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("phiSMax", 0.6)
    ),
    depositionContentCap_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "depositionContentCap", phiSMax_
        )
    ),
    skeletonLaw_(s25SkeletonLaw(phaseClotProperties_)),
    skeletonLawLT_(skeletonLaw_ == "valanisLandelTrilinear"),
    skeletonLawCoeffs_
    (
        s25SkeletonLawCoeffs(phaseClotProperties_, skeletonLawLT_)
    ),
    skeletonLawTable_(s25SkeletonLawTable(skeletonLawCoeffs_, skeletonLawLT_)),
    elasticCourantModulus_
    (
        s25ElasticCourantModulus(phaseClotProperties_, skeletonLawLT_)
    ),
    explicitDiffusionRate_(0),
    networkPressureWeight_
    (
        phaseClotProperties_.lookupOrDefault<word>
        (
            "networkPressureWeight", "depositedOccupancy"
        )
    ),
    networkCriticalStretch_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "networkCriticalStretch", 2.8
        )
    ),
    networkDamageOnsetStretch_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "networkDamageOnsetStretch", 2.105
        )
    ),
    networkDamageStretchScale_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "networkDamageStretchScale", 1.0
        )
    ),
    materialLabelBasis_
    (
        phaseClotProperties_.lookupOrDefault<word>
        (
            "materialLabelBasis", "transportedLabel"
        )
    ),
    // S25 F (design 2026-10-10-S25 section 2F): timeScale = S_t, NO
    // default. Its consumers are networkRupture fibreRelease (t_rel; built
    // in since S21, so every dictionary) and networkViscosity kelvinVoigt
    // (tau_v and its floor), so it is required on every dictionary. Legacy
    // series state 1.0 explicitly (the generator writes it).
    timeScale_(s25TimeScale(phaseClotProperties_)),
    // S16: every default is the shipped behaviour. networkReleaseTime and
    // networkDamageProgressive are read only under networkRupture
    // terminalRelease; 3.3e-4 s is waiver D4's value (source 63). S25 F:
    // the default is timeScale*3.3e-4 (a physical time in seconds, scaled).
    networkReleaseTime_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "networkReleaseTime", timeScale_*3.3e-4
        )
    ),
    // S23 section 2.6: R_c, a physical constant (21.95 = lc^3, waived).
    volumetricTearRatio_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "volumetricTearRatio", 21.95
        )
    ),
    // S18p: the default is S18 as built (no counters, no dump).
    projSiteLog_
    (
        phaseClotProperties_.lookupOrDefault<Switch>("projSiteLog", false)
    ),
    // S19: each default is the shipped behaviour (segregated, no R3).
    momentumPressureCoupling_
    (
        phaseClotProperties_.lookupOrDefault<word>
        (
            "momentumPressureCoupling", "segregated"
        )
    ),
    networkViscosity_
    (
        phaseClotProperties_.lookupOrDefault<word>("networkViscosity", "none")
    ),
    // S25 F: the default tau_v is timeScale*2.5e-4 (source 97, scaled).
    networkViscosityTau_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "networkViscosityTau", timeScale_*2.5e-4
        )
    ),
    sensitivityLimit_
    (
        phaseClotProperties_.lookupOrDefault<word>("sensitivityLimit", "none")
    ),
    // S22: the default is PARDISO (section 2.4). blockSolverReproducible is
    // read strictly in the constructor body (section 4: a bool or refused).
    blockLinearSolver_
    (
        phaseClotProperties_.lookupOrDefault<word>
        (
            "blockLinearSolver", "pardiso"
        )
    ),
    blockSolverThreads_
    (
        phaseClotProperties_.lookupOrDefault<label>("blockSolverThreads", 0)
    ),
    blockSolverReproducible_(blockCoupled::blockSolverReproducibleDefault),
    // S25 C (OI-65; S25 01 finding 3b): read on a restart, so the first
    // step's volumetric-tear population is not empty. No file: 0, as before.
    phiSkel_
    (
        IOobject
        (
            "phiSkel", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0)
    ),
    phiFluid_
    (
        IOobject
        (
            "phiFluid", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("one", dimless, 1)
    ),
    // Seeded from phi_, which at construction is the single-velocity flux and
    // is therefore exactly the fluid's own flux -- the correct cold start.
    // S25 C (OI-65; S25 01 findings 1-3): both carriers are WRITTEN, so a
    // restart reads the fluxes the last step used rather than the mixture
    // flux (fluid) and zero (skeleton). No file: today's seeds.
    phiFluidOwn_
    (
        IOobject
        (
            "phiFluidOwn", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        phi_
    ),
    phiSkelOwn_
    (
        IOobject
        (
            "phiSkelOwn", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        phi_*dimensionedScalar("zero", dimless, 0)
    ),
    skeletonDivergenceRun_
    (
        phaseClotProperties_.lookupOrDefault<label>
        (
            "skeletonDivergenceRun", 5
        )
    ),
    deltaTIncrement_(runTime.deltaTValue()),
    fsiResidual_
    (
        IOobject("fsiResidual", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedVector("zero", dimLength, Zero)
    ),
    HStep_
    (
        IOobject("HStep", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedScalar("zero", dimPressure, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    fsiOmega_(1.0),
    couplingResidual_(0.0),
    couplingResidual0_(0.0),
    couplingIters_(0),
    couplingConverged_(false),
    mechanicsFinalised_(false),

    tPhaseField_(0.0),
    tPlatelets_(0.0),
    tSkeleton_(0.0),
    tDamage_(0.0),
    tPermeability_(0.0),
    tDerived_(0.0),
    tRelease_(0.0),
    tStepStart_(0.0),


    nPhaseCorrectors_
    (
        phaseClotProperties_.lookupOrDefault<label>("nPhaseCorrectors", 1)
    ),
    previousInventory_(0),
    detachedContentStep_(0),
    previousContentVolume_(0),
    contentBoundaryFlux_(0),
    transferClotGrowth_(0),
    dBSumCommitted_(0),
    deltaThetaIntAtTransfer_(0),

    // --- core/shell material regions; default false = single material ---
    coreShell_
    (
        phaseClotProperties_.lookupOrDefault<Switch>("coreShell", false)
    ),
    shell_
    (
        IOobject
        (
            "shell", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("one", dimless, 1),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    ruptured_
    (
        IOobject
        (
            "ruptured", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    // Zero here; the body below seeds them from the legacy c and shell when
    // contentModel is on. READ_IF_PRESENT so a restart uses the written
    // contents rather than re-deriving them from a q that may by then be
    // outside its domain -- the failure mode that makes Wp's NO_READ
    // reconstruction unsafe (Codex, 2026-09-09).
    thetaCHeaderPresent_
    (
        IOobject
        (
            "thetaC", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::NO_WRITE
        ).headerOk()
    ),
    thetaHHeaderPresent_
    (
        IOobject
        (
            "thetaH", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::NO_WRITE
        ).headerOk()
    ),
    nPHeaderPresent_
    (
        IOobject
        (
            "nP", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::NO_WRITE
        ).headerOk()
    ),
    thetaC_
    (
        IOobject
        (
            "thetaC", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    thetaH_
    (
        IOobject
        (
            "thetaH", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    thetaNetRefHeaderPresent_
    (
        IOobject
        (
            "thetaNetRef", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::NO_WRITE
        ).headerOk()
    ),
    thetaNetRef_
    (
        IOobject
        (
            "thetaNetRef", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    // D74: ref_I, thetaNetRef_'s construction (patch types checked equal in
    // the constructor body, where it is also seeded or refused).
    thetaNetRefIntact_
    (
        IOobject
        (
            "thetaNetRefIntact", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    refISumStepStart_(0),
    refITransferStep_(0),
    refIBoundaryFluxStep_(0),
    refISpSourceStep_(0),
    // S16 C: the damage partition. Zero here; seeded from the contents (and
    // ruptured_) in the constructor body when not read, and switched to
    // NO_WRITE there unless damageTransport is contentPartition.
    thetaIntact_
    (
        IOobject
        (
            "thetaIntact", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    thetaPending_
    (
        IOobject
        (
            "thetaPending", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    thetaBroken_
    (
        IOobject
        (
            "thetaBroken", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    // S17 A: the pending fibre. The four intensive fields are read
    // all-or-none and switched to NO_WRITE in the constructor body unless
    // networkRupture is fibreRelease. S25 C (red team 04 change 8): the two
    // contents are now WRITTEN and read back (all-or-none), so a restart
    // restores the written values instead of re-deriving them from the
    // rounded intensive fields. No files: re-derived, as before.
    thetaPendingB_
    (
        IOobject
        (
            "thetaPendingB", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedSymmTensor("zero", dimless, Zero),
        zeroGradientFvPatchField<symmTensor>::typeName
    ),
    thetaPendingP_
    (
        IOobject
        (
            "thetaPendingP", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimPressure, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    pendingAxis_
    (
        IOobject
        (
            "pendingAxis", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedVector("ex", dimless, vector(1, 0, 0)),
        zeroGradientFvPatchField<vector>::typeName
    ),
    pendingStretch_
    (
        IOobject
        (
            "pendingStretch", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("one", dimless, 1),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    pendingB_
    (
        IOobject
        (
            "pendingB", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedSymmTensor("I", dimless, symmTensor::I),
        zeroGradientFvPatchField<symmTensor>::typeName
    ),
    pendingPressure_
    (
        IOobject
        (
            "pendingPressure", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimPressure, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    // S17b A1: the taut share of the pending content; NO_WRITE unless
    // fibreSlack v1 (switched in the constructor body).
    thetaTaut_
    (
        IOobject
        (
            "thetaTaut", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    plateletAttempt_
    (
        phaseClotProperties_.lookupOrDefault<label>("plateletAttempt", 0)
    ),
    plateletDiffusivity_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "plateletDiffusivity", 0.0
        )
    ),
    // DEFAULTS TO vP_, which is the DECLARED platelet-equivalent
    // specialisation Y = vP and not an identity. It is logged as such.
    plateletYield_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "plateletYield", vP_
        )
    ),
    plateletBindRate_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "plateletBindRate", 0.0
        )
    ),
    plateletAdhesionSpeed_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "plateletAdhesionSpeed", -1.0
        )
    ),
    plateletTransportOnly_
    (
        phaseClotProperties_.lookupOrDefault<Switch>
        (
            "plateletTransportOnly", false
        )
    ),
    nP_
    (
        IOobject
        (
            "nP", runTime.name(), mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless/dimVolume, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    NP_
    (
        IOobject
        (
            "NP", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("zero", dimless/dimVolume, 0),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    epsP_
    (
        IOobject
        (
            "epsP", runTime.name(), mesh,
            IOobject::NO_READ, IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("one", dimless, 1),
        zeroGradientFvPatchField<scalar>::typeName
    ),
    // Initialised to the SHELL (single-material) values. updateMaterialFields()
    // in the body below blends in the core where one is declared; with
    // coreShell false it re-assigns exactly these same values, so the fields
    // are uniform and every product below is a multiply by the same number in
    // every cell.
    GField_
    (
        IOobject("Gfield", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        G_,
        zeroGradientFvPatchField<scalar>::typeName
    ),
    lambdaField_
    (
        IOobject("lambdaField", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        lambdaLame_,
        zeroGradientFvPatchField<scalar>::typeName
    ),
    GcField_
    (
        IOobject("Gcfield", runTime.name(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        Gc_,
        zeroGradientFvPatchField<scalar>::typeName
    ),
    k0pField_
    (
        IOobject("k0pField", runTime.name(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        k0p_,
        zeroGradientFvPatchField<scalar>::typeName
    ),
    // Core values default to the shell values, so a dictionary that names no
    // core describes one material and the blend is the identity.
    Gcore_
    (
        "Gcore", dimPressure,
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "Gcore", phaseClotProperties_.lookupOrDefault<scalar>("G", 58.6)
        )
    ),
    Gccore_
    (
        "Gccore", dimPressure*dimLength,
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "Gccore", phaseClotProperties_.lookupOrDefault<scalar>("Gc", 1.0)
        )
    ),
    k0pcore_
    (
        "k0pcore", dimArea,
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "k0pcore", phaseClotProperties_.lookupOrDefault<scalar>("k0p", 3.5e-14)
        )
    ),

    skeletonSolidFloor_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("skeletonSolidFloor", 1e-4)
    ),
    clotErosionRate_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("clotErosionRate", 0.0)
    ),
    clotErosionTauPa_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("clotErosionTauPa", 3.0)
    ),
    skeletonDiluteDragRate_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "skeletonDiluteDragRate", 0.0
        )
    ),
    skeletonPinExponent_
    (
        phaseClotProperties_.lookupOrDefault<scalar>
        (
            "skeletonPinExponent", 8.0
        )
    ),
    permeabilityFloor_
    (
        phaseClotProperties_.lookupOrDefault<scalar>("permeabilityFloor", small)
    ),
    plateletDepositStep_(mesh.nCells(), 0.0),
    // S25 C (OI-65; S25 01 Q1(c)): the scalar state a step reads before it
    // writes it, persisted at <time>/uniform/phaseClotState. Filled every
    // step at the end of postSolve() (so a writeNow from a function object
    // writes the current state too); restored at the end of the constructor
    // body. No file: today's seeds.
    phaseClotState_
    (
        IOobject
        (
            "phaseClotState", runTime.name(), "uniform", mesh,
            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE
        )
    )
{
    // ---- S21: THE BUILT-IN KEYS ------------------------------------------
    //
    // research/designs/2026-10-05-S21-clean-solver.md section 1.2. Each key
    // in this table is no longer an option: the solver runs only the listed
    // value, and the code of every other value was deleted with the key's
    // member. A dictionary that still names one is REFUSED, whatever value it
    // gives, so no case can believe it selects something the solver no longer
    // implements. Old runs are reproduced from release-4198bb17 or its
    // archive backup. Groups B and C append their keys to this one table.
    {
        static const char* const s21BuiltInKeys[][2] =
        {
            // S21 group A
            {"contentModel", "true"},
            {"contentOperator", "content"},
            {"contentTransportScheme", "fct"},
            {"cplusOrder", "shared"},
            {"materialCarrierBlend", "false"},
            {"phaseDeltaBasis", "content"},
            {"phaseFieldMaterialVelocity", "true"},
            {"phaseFluidFractionBasis", "content"},
            {"phaseForceBasis", "content"},
            {"phaseGrowthDriver", "transfer"},
            {"phaseIndicatorBasis", "content"},
            {"phaseLimiterBasis", "content"},
            {"phaseMomentumFluxBasis", "shared"},
            {"phaseNormalBasis", "content"},
            {"phasePlateletFluxBasis", "content"},
            {"phaseWeightBasis", "content"},
            {"plateletBindLocalisation", "interface"},
            {"plateletCarrierArchive", "false"},
            {"plateletContentFormulation", "true"},
            {"plateletContentModel", "true"},
            {"plateletFluxConsistent", "false"},
            {"plateletFluxConvention", "intrinsic"},
            {"plateletOuterForceReject", "false"},
            {"plateletOuterTransactions", "false"},
            // S21 group B
            {"advectReleased", "true"},
            {"damageTransport", "contentPartition"},
            {"elasticCoupling", "implicitTangent"},
            {"elasticStabilityStiffness", "tangent"},
            {"fibreArithmetic", "fieldUnits"},
            {"fibreSlack", "v1"},
            {"frozenPendingPressure", "none"},
            {"networkDamageProgressive", "true"},
            {"networkPressureDamage", "tensile"},
            {"networkPressureForce", "faceGradient"},
            {"networkPressureModel", "convexExcess"},
            {"networkPressureReference", "deposited"},
            {"networkRupture", "fibreRelease"},
            {"releasedCarrier", "material"},
            {"stressFaceTransmission", "series"},
            // S21 group C
            {"beDetRenormalise", "true"},
            {"beFluxConsistentGradU", "true"},
            {"beRelaxBasis", "networkAbsent"},
            {"beStretchGradient", "fluxConsistent"},
            {"beTermDiagnostic", "true"},
            {"beTraceFreeSource", "false"},
            {"blockCoupledLumen", "eliminate"},
            {"blockFaceInverse", "faceAssembled"},
            {"bondHoldCarrier", "true"},
            {"bondHoldTaper", "false"},
            {"clotMechanics", "true"},
            {"constitutiveModel", "neoHookean"},
            {"dragDriver", "legacy"},
            {"dragOccupancyBasis", "intrinsic"},
            {"eulerianBrinkman", "true"},
            {"eulerianMaterialFluxIsPhi", "true"},
            {"eulerianSolid", "true"},
            {"explicitStabilityBounds", "on"},
            {"faceConsistentElasticForce", "false"},
            {"finiteStrain", "true"},
            {"fsiAitken", "false"},
            {"fsiSweepTrace", "true"},
            {"growthClosure", "reaction"},
            {"interfaceDepletionModel", "none"},
            {"operatorAudit", "false"},
            {"operatorLocalisationAudit", "false"},
            {"permeabilityModel", "kozenyCarmanSolid"},
            {"shearEnhancementModel", "thresholdSaturating"},
            {"skeletonDiffuseDomain", "true"},
            {"skeletonFloorBasis", "occupancy"},
            {"skeletonMomentumForm", "phaseContinuity"},
            {"skeletonPressureForce", "false"},
            {"skeletonResidualDiagnostic", "false"},
            {"skeletonWallFace", "zeroed"},
            {"stronglyCoupledFSI", "false"},
            {"twoVelocitySkeleton", "true"},
            {"unlocalisedPhaseWeightsAccepted", "false"},
            {"vnCapEnabled", "false"},
            {"wallPatchSplit", "split"}
        };

        for (const auto& builtIn : s21BuiltInKeys)
        {
            if (phaseClotProperties_.found(builtIn[0]))
            {
                FatalErrorInFunction
                    << "phaseClotProperties key `" << builtIn[0] << "` is"
                    << " built in since S21, at `" << builtIn[1] << "`, and"
                    << " must not be set. Remove it from the dictionary: its"
                    << " other values were deleted with their code. Reproduce"
                    << " an older run from release-4198bb17."
                    << exit(FatalError);
            }
        }
    }
    // ---- S21: THE DELETED KEYS ----------------------------------------
    //
    // Each entry in this table was read only by code S21 deleted (groups A,
    // B and C): no value of it has any effect any more. A dictionary that
    // still names one is REFUSED, so a case cannot believe it sets
    // something the solver no longer reads.
    {
        static const char* const s21DeletedKeys[] =
        {
            "fsiNormProbeFraction",
            "fsiOmegaMax",
            "fsiOmegaMin",
            "fsiTolerance",
            "k1",
            "materialCourant",
            "materialDeltaTMin",
            "nFSICorrectors",
            "nMaterialFloorSteps",
            "nSkeletonCorrectors",
            "nSkeletonDivergedSteps",
            "nStaggeredCorrectors",
            "operatorAuditEvery",
            "phi0",
            "releasedAmbientQ",
            "releasedMaskDilation",
            "releasedMaskLimit",
            "releasedMaskSweeps",
            "skeletonDivergenceGrowth",
            "skeletonFixedPointTolerance",
            "skeletonImplicitFactor",
            "skeletonUsBound"
        };

        for (const char* const deleted : s21DeletedKeys)
        {
            if (phaseClotProperties_.found(deleted))
            {
                FatalErrorInFunction
                    << "phaseClotProperties key `" << deleted << "`: S21:"
                    << " deleted, no effect. Remove it from the dictionary:"
                    << " the code that read it was deleted. Reproduce an"
                    << " older run from release-4198bb17."
                    << exit(FatalError);
            }
        }
    }
    // THE BOND HOLD IS APPLIED ONCE, AND ON THE TWO-VELOCITY PATH IT IS
    // APPLIED TO Vs.
    //
    // bondHoldCarrier multiplies the phase-field carrier by
    // (1 - interpolate(bondHold)) to pin material a bond is holding. That was
    // the only way to do it while U was the sole velocity. Since 2026-09-04
    // the anchor is an fvm::Sp on the SKELETON momentum equation, which holds
    // the material directly, so the carrier factor is a second copy of the
    // same constraint.
    //
    // It is not merely redundant, it is destructive: multiplying a flux by a
    // spatially varying mask destroys whatever solenoidality it had, and
    // clotResidual == -0.5*int c*div(carrier) dV is exactly sensitive to that.
    // MEASURED -- cumulative clot-volume drift, one-velocity children only,
    // so the two-velocity path is not what separates them:
    //
    //     g04..g40   bondHoldCarrier off   +0.00031   (the discretisation floor)
    //     em_free    bondHoldCarrier on    +0.00478   (15x the floor)
    //
    // Forced rather than rejected, because every existing two-velocity child
    // sets it and failing them would only invite someone to turn the anchor
    // off instead. Reported so the override cannot be silent.

    // WHICH VOLUMETRIC LAW IS ACTUALLY RUNNING. A model selected by a word in
    // a dictionary is exactly the kind of thing that gets assumed rather than
    // read, and the two forms differ by six orders of magnitude at the
    // diffuse fringe. Stated once, at startup, next to phi_s0 -- which is the
    // other half of the answer, since the fringe force scales as phi_s0^2.
    {
        Info<< "PHASE_CLOT_CONFIG networkPressureModel="
            << "convexExcess"
            << " networkPressureCoeff=" << networkPressureCoeff_
            << " skeletonVolumeFraction=" << skeletonVolumeFraction_
            << " phiSMax=" << phiSMax_
            << " skeletonLaw=" << skeletonLaw_
            << " networkPole=" << networkPole()
            << " residualPhiS=" << residualPhiS_
            << " residualChi=" << residualChi_
            << (residualChi_ > 0
                    ? " KresSupport=chi-anchor-invariant"
                    : " KresSupport=phi_s-ABSOLUTE-shrinks-as-the-anchor-rises")
            << " clotErosionRate=" << clotErosionRate_
            << " clotErosionTauPa=" << clotErosionTauPa_
            << (clotErosionRate_ > 0
                    ? " erosion=ON"
                    : " erosion=OFF-the-clot-CANNOT-shorten")
            << " skeletonDiluteDragRate=" << skeletonDiluteDragRate_
            << (skeletonDiluteDragRate_ > 0
                    ? " dragFloor=ON-K-at-least-k1-times-chi"
                    : " dragFloor=OFF-KozenyCarman-alone-below-the-packed-limit")
            << " note=poleAtMax-applies-a-fringe-force-see-networkPressure"
            << nl;

        // S25 A: LT HOLDS ONE MODULUS AND ONE BULK RATIO (design
        // 2026-10-10-S25 section 2A). G0 = E1/2 exactly (halving is exact
        // in binary, so a dictionary that writes both from one value
        // passes); K0 = kappa G0 is the convexExcess law's K_n =
        // networkPressureCoeff gD G at s'(1), so networkPressureCoeff must
        // equal kappa exactly. A disagreeing (or absent, i.e. defaulted) G
        // or networkPressureCoeff is refused, never overridden. coreShell
        // is refused: its G is a geometric blend of Gcore and G that one E1
        // cannot match (red team 04 change 10).
        if (skeletonLawLT_)
        {
            if (coreShell_)
            {
                FatalErrorInFunction
                    << "skeletonLaw valanisLandelTrilinear with coreShell "
                    << "true is refused: GField is a geometric blend of "
                    << "Gcore and G there, which one E1 cannot match (design "
                    << "2026-10-10-S25 sections 2A, 5)." << exit(FatalError);
            }
            if
            (
                !phaseClotProperties_.found("G")
             || G_.value() != 0.5*skeletonLawCoeffs_.E1
            )
            {
                FatalErrorInFunction
                    << "skeletonLaw valanisLandelTrilinear: G must be "
                    << "stated and equal E1/2 = " << 0.5*skeletonLawCoeffs_.E1
                    << " exactly (the solver holds ONE modulus, G0 = E1/2), "
                    << "but G = "
                    << (
                           phaseClotProperties_.found("G")
                         ? G_.value() : -1.0
                       )
                    << (phaseClotProperties_.found("G") ? "" : " (absent)")
                    << " (design 2026-10-10-S25 sections 2A, 5)."
                    << exit(FatalError);
            }
            if
            (
                !phaseClotProperties_.found("networkPressureCoeff")
             || networkPressureCoeff_ != skeletonLawCoeffs_.kappa
            )
            {
                FatalErrorInFunction
                    << "skeletonLaw valanisLandelTrilinear: "
                    << "networkPressureCoeff must be stated and equal "
                    << "skeletonLawCoeffs kappa = " << skeletonLawCoeffs_.kappa
                    << " exactly (K0 = kappa E1/2 is the convexExcess law's "
                    << "K_n = networkPressureCoeff gD G at s'(1)), but "
                    << "networkPressureCoeff = " << networkPressureCoeff_
                    << (
                           phaseClotProperties_.found("networkPressureCoeff")
                         ? "" : " (absent, defaulted)"
                       )
                    << " (design 2026-10-10-S25 section 2A)."
                    << exit(FatalError);
            }
        }

        // THE POLE NEEDS ROOM TO BE A POLE.
        //
        // poleAtMax and poleCubic both divide by (phi_s_max - phi_s), and
        // poleCubic's derived normalisation carries (phi_s_max - phi_s0) in
        // the numerator. If phi_s_max <= phi_s0 the nominal density is at or
        // beyond close packing: the equilibrium root and the pole coincide or
        // invert, the law changes SIGN, and a compacting network would be
        // PUSHED further in. That is not a tolerance -- it is the law
        // ceasing to mean what its derivation says. Refused rather than
        // clamped, because there is no value of phi_s for which the answer
        // would be right.
        //
        // The cubic is exempt and the exemption is checked, not assumed: it
        // never reads phiSMax_, which is why the banner above prints
        // note=phiSMax-is-UNUSED-by-the-cubic.
        //
        // S25 A (design 2026-10-10-S25 sections 2A, 3A, 5): under LT the
        // pole is networkPole() = 1, so the check is phi0 < 1; phiSMax_
        // is not a pole there (it only defaults depositionContentCap).
        // Under the shipped law the check below is today's, unchanged.
        if (skeletonLawLT_)
        {
            if (!(skeletonVolumeFraction_ < networkPole()))
            {
                FatalErrorInFunction
                    << "skeletonLaw valanisLandelTrilinear needs "
                    << "skeletonVolumeFraction < the network pole "
                    << networkPole() << ", but skeletonVolumeFraction = "
                    << skeletonVolumeFraction_ << " (design 2026-10-10-S25 "
                    << "section 5: phi0 >= 1 is refused)."
                    << exit(FatalError);
            }
        }
        else if
        (
            phiSMax_ <= skeletonVolumeFraction_
        )
        {
            FatalErrorInFunction
                << "networkPressureModel = " << "convexExcess"
                << " needs phiSMax > skeletonVolumeFraction, but phiSMax = "
                << phiSMax_ << " and skeletonVolumeFraction = "
                << skeletonVolumeFraction_ << "." << nl
                << "The nominal density would sit at or past close packing, "
                << "which inverts the sign of the network pressure." << nl
                << "Either raise phiSMax above the nominal solid fraction or "
                << "select networkPressureModel cubic, which does not use it."
                << exit(FatalError);
        }
    }

    // SILENT-WRONG-CONFIGURATION GUARDS FOR eulerianBrinkman.
    //
    // A switch that is set in a dictionary and does nothing is the failure
    // this project keeps paying for: the run completes, the assessor reports
    // passed, and the answer is about a model that was never assembled. Both
    // of these would do exactly that.

    updateDerivedFields();

    // W = q*w, once q exists. Done BEFORE the second updateDerivedFields()
    // because on the content branch Vn is built from W, and the first call
    // above ran while W still held w's initial values.
    //
    // S25 B (R1): Wp := q + e. With e read (a directory written by S25),
    // that is the restored state, exact at e == 0 at any precision. Without
    // it (a cold start, or a directory written before S25), today's seed
    // Wp = q*w, carried as e = q*(w - 1): exactly 0 where w == 1.
    {
        if (!e_.headerOk())
        {
            e_.primitiveFieldRef() =
                q_.primitiveField()*(w_.primitiveField() - 1.0);
            e_.correctBoundaryConditions();
        }
        Wp_.primitiveFieldRef() = q_.primitiveField() + e_.primitiveField();
        Wp_.correctBoundaryConditions();
        Info<< "PHASE_CLOT_S25_R1 seed="
            << (e_.headerOk() ? "eRead" : "eFromW")
            << " eAbsMax=" << gMax(mag(e_.primitiveField())())
            << endl;

        // The inlet W BC is fixedValue 1, inherited from w. That is only the
        // right content if the inlet is pure blood. Checked rather than
        // assumed: a clot touching the inlet would silently inject platelets.
        forAll(q_.boundaryField(), patchi)
        {
            if (!isA<fixedValueFvPatchScalarField>(Wp_.boundaryField()[patchi]))
            {
                continue;
            }
            const scalarField& qp = q_.boundaryField()[patchi];
            if (qp.size() && gMin(qp) < 1.0 - 1.0e-6)
            {
                FatalErrorInFunction
                    << "plateletContentFormulation: patch "
                    << q_.boundaryField()[patchi].patch().name()
                    << " has a fixedValue W but q = " << gMin(qp)
                    << " < 1 there, so W = q*w is not the inlet content."
                    << nl << "See research/designs/"
                    << "AM2020_platelet_content_formulation_derivation.md"
                    << " section 3.5" << exit(FatalError);
            }
        }

        updateDerivedFields();
    }

    updateMaterialFields();
    previousInventory_ = fvc::domainIntegrate(q_*w_).value();

    // SEED THE CONTENTS FROM THE LEGACY STATE, ONCE.
    //
    // theta = theta_ref*max(1-q,0) is the SAME algebraic map solidFraction()
    // already uses, minus its 0.9 cap -- the cap is a regularisation of the
    // legacy indicator and must not silently enter a conserved content. The
    // split follows the existing material label, so at t=0 the content model
    // and the legacy model describe the same clot; that is what makes the
    // equivalence test in solveContentComponents() meaningful rather than a
    // comparison of two different initial conditions.
    //
    // Only when contentModel is on, and only if a restart has not already
    // supplied the fields (gMax > 0 means they were read).
    // STAGE 3 DRAG DRIVER -- THREE REFUSALS, ALL AT CONSTRUCTION.
    //
    // Spec: research/model-specs/AM2020-content-drag-v1.md.

    if
    (
        thetaCHeaderPresent_ != thetaHHeaderPresent_
    )
    {
        FatalErrorInFunction
            << "Incomplete content restart at time " << runTime.name()
            << ": thetaC header present=" << thetaCHeaderPresent_
            << ", thetaH header present=" << thetaHHeaderPresent_ << nl
            << "Provide both thetaC and thetaH, or neither for legacy seeding."
            << exit(FatalError);
    }

    if
    (
        nPHeaderPresent_ != thetaCHeaderPresent_
    )
    {
        FatalErrorInFunction
            << "Incomplete persistent content restart at time "
            << runTime.name()
            << ": thetaC header present=" << thetaCHeaderPresent_
            << ", thetaH header present=" << thetaHHeaderPresent_
            << ", nP header present=" << nPHeaderPresent_ << nl
            << "Provide thetaC, thetaH, and nP together, or omit all three "
            << "for legacy theta seeding with zero free platelets."
            << exit(FatalError);
    }

    if (plateletAttempt_ < 0 || plateletAttempt_ != 0)
    {
        FatalErrorInFunction
            << "plateletAttempt must be non-negative and a nonzero value requires "
            << "plateletOuterTransactions true."
            << exit(FatalError);
    }

    if (!thetaCHeaderPresent_ && !thetaHHeaderPresent_)
    {
        const volScalarField theta0
        (
            skeletonVolumeFraction_
           *max(scalar(1) - q_, dimensionedScalar("zero", dimless, 0))
        );
        const volScalarField sClamp
        (
            min
            (
                max(shell_, dimensionedScalar("zero", dimless, 0)),
                dimensionedScalar("one", dimless, 1)
            )
        );
        thetaH_ = theta0*sClamp;
        thetaC_ = theta0*(scalar(1) - sClamp);
        thetaC_.correctBoundaryConditions();
        thetaH_.correctBoundaryConditions();

        Info<< "PHASE_CLOT_CONTENT_INIT"
            << " thetaRef=" << skeletonVolumeFraction_
            << " vP=" << vP_
            << " thetaMin=" << gMin(volScalarField(thetaC_ + thetaH_)().primitiveField())
            << " thetaMax=" << gMax(volScalarField(thetaC_ + thetaH_)().primitiveField())
            << " contentVolume="
            << fvc::domainIntegrate(thetaC_ + thetaH_).value()
            << endl;
    }

    // ---- S10: THE THREE ROOT-CAUSE SWITCHES, VALIDATED AND ANNOUNCED ----
    //
    // Each defaults to the shipped behaviour, so every earlier child is
    // unchanged. Each refuses a value it does not implement and a combination
    // it cannot honour, because a switch that silently does nothing reads
    // exactly like a fix that did not work.

    // THE REFERENCE IS THE CONTENT AS DEPOSITED, SO AT t = 0 IT IS theta.
    // That is the whole of the step-1 fix: with thetaNetRef = theta, the
    // network pressure is exactly zero in every cell, the interface included,
    // before anything has moved. Read from disk on a restart, so a restart
    // does not silently re-reference a compacted clot as stress-free.
    {
        if (!thetaNetRefHeaderPresent_)
        {
            thetaNetRef_ = thetaC_ + thetaH_;
            thetaNetRef_.correctBoundaryConditions();
        }
    }

    // D74 (design 2026-10-09-D74 rev. 2 section 2): THE INTACT REFERENCE
    // ref_I STARTS AS THE REFERENCE, ref_I = ref, here, after ref is read or
    // seeded. It is read all-or-none with the damage partition: a restart
    // that carries the partition but not ref_I (any BATCH 72/73 write) holds
    // the proxy (P1/S) ref only, exact only where d* is 0 or 1 (numerics 33
    // L48), so it is refused rather than seeded with a guess; and ref_I
    // without the partition has nothing to be the intact share of.
    {
        const bool hasRefI = thetaNetRefIntact_.headerOk();
        const bool hasPartition =
            thetaIntact_.headerOk()
         || thetaPending_.headerOk()
         || thetaBroken_.headerOk();
        if (hasRefI != hasPartition)
        {
            FatalErrorInFunction
                << "thetaNetRefIntact (D74) is read all-or-none with the "
                << "damage partition: found thetaNetRefIntact " << hasRefI
                << ", partition (thetaIntact/thetaPending/thetaBroken) "
                << hasPartition << " at time " << runTime.name() << ". A "
                << "partition without thetaNetRefIntact has only the proxy "
                << "(P1/S)*thetaNetRef, which is not the intact reference "
                << "where 0 < d* < 1; start from t = 0 or from a D74 write."
                << exit(FatalError);
        }
        if
        (
            thetaNetRefIntact_.boundaryField().types()
         != thetaNetRef_.boundaryField().types()
        )
        {
            FatalErrorInFunction
                << "thetaNetRefIntact (D74) must carry thetaNetRef's patch "
                << "types: " << thetaNetRefIntact_.boundaryField().types()
                << " against " << thetaNetRef_.boundaryField().types()
                << exit(FatalError);
        }
        if (!hasRefI)
        {
            thetaNetRefIntact_ = thetaNetRef_;
            thetaNetRefIntact_.correctBoundaryConditions();
        }
        Info<< "PHASE_CLOT_D74_CONFIG thetaNetRefIntact="
            << (hasRefI ? "read" : "seeded-from-thetaNetRef")
            << " transport=thetaNetRefLabelOperator"
            << " solve=iterationMatched fct=donorRatio"
            << " transfers=rupture,volumetricTear,growth,deletion"
            << " readers=none" << endl;
    }

    // S10 FIX 2: THE SKELETON'S WALL CONDITION BECOMES SLIP.
    //
    // zeroGradient (see Vs_'s construction) already gives the zero tangential
    // traction a debonded clot needs to move along the wall. What it does not
    // give is NO PENETRATION: it copies the wall cell's normal velocity onto
    // the wall face, so every gradient of Vs taken at a wall cell sees a
    // skeleton passing through the wall. skeletonFlux() zeroes the FLUX there,
    // and S10 fix 4 now zeroes the mixture flux too, so no volume crosses --
    // but the velocity field itself still says it does. slip is exactly
    // "normal component zero, tangential component zero-gradient".
    //
    // Only under releasedCarrier "material", where the solved Vs carries
    // released clot and its wall behaviour therefore shapes what the clot
    // does. Bonded material is still held by the bond-hold term, which is
    // where the bond lives; a hard no-slip here would weld a released clot
    // to the wall for ever -- the objection recorded at Vs_'s construction.
    {
        label nSlip = 0;
        forAll(mesh.boundary(), patchi)
        {
            if (isA<wallFvPatch>(mesh.boundary()[patchi]))
            {
                Vs_.boundaryFieldRef().set
                (
                    patchi,
                    fvPatchField<vector>::New
                    (
                        slipFvPatchVectorField::typeName,
                        mesh.boundary()[patchi],
                        Vs_
                    )
                );
                ++nSlip;
            }
        }
        Vs_.correctBoundaryConditions();

        if (nSlip == 0)
        {
            FatalErrorInFunction
                << "releasedCarrier material found no wall patch to give the "
                << "skeleton a slip condition. The release model needs a wall "
                << "to release from." << exit(FatalError);
        }
    }

    Info<< "PHASE_CLOT_S10_CONFIG"
        << " thetaNetRefSource="
        << (
               thetaNetRefHeaderPresent_ ? "read" : "content-at-start"
           )
        << " wallFluxConstraint=on"
        << " skeletonWall="
        << "slip"
        << " zeroFluxRule=exactZero"
        << endl;

    // ---- S11: THE SIX ROOT-CAUSE SWITCHES OF RUN 20260923-031942 ----
    //
    // Same contract as S10: each defaults to the shipped behaviour, refuses a
    // value it does not implement, and refuses a combination it cannot
    // honour. Three of the six are REQUIRED by another, and the refusals say
    // why, because each dependency is a failure that would otherwise be
    // shipped silently.
    {



        // phaseContinuity removes the ddt(phi) half of the continuity error
        // and relies on the BOUNDED div scheme to remove the div(F) half.
        // Read the scheme rather than assume it: an unbounded scheme would
        // leave the div half in, a bounded one plus a full correction would
        // remove it twice. primitiveEntry::stream() rewinds, so reading the
        // first word here does not disturb the later fvm::div.
        {
            for
            (
                const word& name
              : {word("div(phiSkelFlux,Vs)"), word("div(phiFluidFlux,U)")}
            )
            {
                const word first(mesh.schemes().div(name));
                if (first != "bounded")
                {
                    FatalErrorInFunction
                        << "skeletonMomentumForm phaseContinuity adds "
                        << "-SuSp(fvc::ddt(phi), U) and relies on the BOUNDED "
                        << "convection scheme for the div(F) half of the "
                        << "continuity error. " << name << " is '" << first
                        << " ...', not 'bounded ...'." << exit(FatalError);
                }
            }
        }

    }

    // ---- S12: THE DRAG, WEIGHTED AS THE STRESS ALREADY IS ----------------
    //
    // Every requirement below is a dependency, refused with its reason, so
    // the basis cannot ship in a configuration where it means something
    // else. research/designs/2026-09-23-S12-intrinsic-occupancy-drag.md.
    {
        if (skeletonDiluteDragRate_ > 0)
        {
            FatalErrorInFunction
                << "dragOccupancyBasis intrinsic gives every partial cell the "
                << "network's own drag, 1.008e7*ind 1/s in the shell at the "
                << "reference. skeletonDiluteDragRate " << skeletonDiluteDragRate_
                << " would add a second, calibrated drag floor on top -- the "
                << "unanchored constant this basis retires (OI-31). Set it "
                << "to 0." << exit(FatalError);
        }
    }
    // ---- S13: THE TWO GRID-SCALE DEFECTS OF RUN 20260923-135545 ----------
    //
    // research/designs/2026-09-23-S13-grid-scale-restoring.md.
    if
    (
        networkPressureWeight_ != "depositedOccupancy"
     && networkPressureWeight_ != "networkPresence"
     && networkPressureWeight_ != "contentRatio"
    )
    {
        FatalErrorInFunction
            << "networkPressureWeight '" << networkPressureWeight_
            << "' is not recognised: use depositedOccupancy, "
            << "networkPresence or contentRatio." << exit(FatalError);
    }
    // S18 C (R2b, DECISIONS 2026-09-29): contentRatio is networkPresence
    // times theta/thetaNetRef, so it needs everything networkPresence does,
    // and the deposited reference its ratio divides by.
    if (networkPressureWeight_ == "contentRatio")
    {
    }
    if
    (
        networkPressureWeight_ == "networkPresence"
     || networkPressureWeight_ == "contentRatio"
    )
    {
        if (!(residualChi_ > 0))
        {
            FatalErrorInFunction
                << "networkPressureWeight " << networkPressureWeight_
                << " ramps to zero below "
                << "residualChi of deposited network, which is what keeps the "
                << "lumen free of network pressure; residualChi is "
                << residualChi_ << ". Set it positive." << exit(FatalError);
        }
    }
    // ---- S14: RUPTURE AT THE CRITICAL STRETCH, AND THE ELASTIC BOUND -----
    //
    // research/designs/2026-09-23-S14-network-rupture.md.
    // S15: every consequence of a damaged network -- the (1 - d) stress and
    // BSD weights, the intact-content memory, the deposit dilution, the
    // label transport -- is shared by both rupture laws; only how d is set
    // differs. So the requirements below are the same for both.
    Info<< "PHASE_CLOT_S14_CONFIG"
        << " networkCriticalStretch=" << networkCriticalStretch_
        << endl;

    // ---- S15: THE MATERIAL LABEL FROM THE CONTENTS; PROGRESSIVE DAMAGE ---
    //
    // research/designs/2026-09-24-S15-composition-label-and-progressive-damage.md.
    if
    (
        materialLabelBasis_ != "transportedLabel"
     && materialLabelBasis_ != "contentComposition"
    )
    {
        FatalErrorInFunction
            << "materialLabelBasis '" << materialLabelBasis_
            << "' is not recognised: use transportedLabel or "
            << "contentComposition." << exit(FatalError);
    }
    if
    (
        materialLabelBasis_ == "contentComposition"
     && !coreShell_
    )
    {
        FatalErrorInFunction
            << "materialLabelBasis contentComposition derives the core/shell "
            << "label from the transported core and shell contents, "
            << "thetaH/(thetaC + thetaH); it needs contentModel true (the "
            << "contents) and coreShell true (a label to derive)."
            << exit(FatalError);
    }
    Info<< "PHASE_CLOT_S15_CONFIG"
        << " materialLabelBasis=" << materialLabelBasis_
        << " networkDamageOnsetStretch=" << networkDamageOnsetStretch_
        << " networkDamageStretchScale=" << networkDamageStretchScale_
        << endl;
    // The contents were seeded above (or read); the label they imply is the
    // material from the first step on.
    syncMaterialLabelFromContent();

    // ---- S16: TERMINAL RUPTURE WITH A TIMED RELEASE, DAMAGE AS CONTENT ---
    //
    // research/designs/2026-09-24-S16-timed-terminal-rupture.md (APPROVED
    // 2026-09-24; decisions and waivers in the docket's DECISIONS.md). Each
    // switch defaults to the shipped behaviour. The VALUE of every switch is
    // checked before any combination, so a misspelt value is reported as
    // what it is and not as a missing dependency.
    // S17 A: fibreRelease IS terminalRelease's event and release (design
    // 2A "The event, as S16" and "The release, as S16"); only what the
    // pending share carries differs. Every S16 requirement applies to it.
    {
        if (!(networkCriticalStretch_ > 1))
        {
            FatalErrorInFunction
                << "networkRupture " << "fibreRelease"
                << ": networkCriticalStretch "
                << networkCriticalStretch_ << " is the rupture stretch lc, "
                << "a principal stretch, and must exceed 1 (2.8, source 28)."
                << exit(FatalError);
        }
        if (!(networkReleaseTime_ > 0))
        {
            FatalErrorInFunction
                << "networkRupture " << "fibreRelease"
                << ": networkReleaseTime "
                << networkReleaseTime_ << " s is the release time tau of "
                << "dd/dt = (d* - d)/tau and must be positive (3.3e-4 s, "
                << "source 63, waiver D4)." << exit(FatalError);
        }
        {
            if
            (
                !(networkDamageOnsetStretch_ > 1
               && networkDamageStretchScale_ > 0)
            )
            {
                FatalErrorInFunction
                    << "networkRupture " << "fibreRelease" << " with "
                    << "networkDamageProgressive on: networkDamageOnsetStretch "
                    << networkDamageOnsetStretch_ << " must exceed 1 and "
                    << "networkDamageStretchScale "
                    << networkDamageStretchScale_ << " must be positive "
                    << "(source 32 Eq. 6: 2.105 and 1.0)." << exit(FatalError);
            }
            if (!(networkCriticalStretch_ > networkDamageOnsetStretch_))
            {
                FatalErrorInFunction
                    << "networkRupture " << "fibreRelease" << " with "
                    << "networkDamageProgressive on: networkCriticalStretch "
                    << networkCriticalStretch_ << " must exceed "
                    << "networkDamageOnsetStretch "
                    << networkDamageOnsetStretch_ << " -- the progressive "
                    << "part (Eq. 6) lives between them. A rupture stretch at "
                    << "or below the onset is buildable only terminal-only: "
                    << "set networkDamageProgressive off." << exit(FatalError);
            }
        }
    }
    // S16 C, THE INITIAL CONDITION: the partition is read all-or-none, or
    // seeded from the contents and the damage the case starts with
    // (ruptured_, 0 on a fresh start): (theta(1 - d), 0, theta d), d* = d.
    // theta*(1 - 0) is theta exactly, so a fresh start is (theta, 0, 0).
    {
        const bool hasI = thetaIntact_.headerOk();
        const bool hasP = thetaPending_.headerOk();
        const bool hasB = thetaBroken_.headerOk();
        if ((hasI || hasP || hasB) && !(hasI && hasP && hasB))
        {
            FatalErrorInFunction
                << "damageTransport contentPartition: the partition is read "
                << "all-or-none; found thetaIntact " << hasI
                << ", thetaPending " << hasP << ", thetaBroken " << hasB
                << "." << exit(FatalError);
        }
        if (!hasI)
        {
            const scalarField& tC = thetaC_.primitiveField();
            const scalarField& tH = thetaH_.primitiveField();
            const scalarField& d0 = ruptured_.primitiveField();
            scalarField& pI = thetaIntact_.primitiveFieldRef();
            scalarField& pP = thetaPending_.primitiveFieldRef();
            scalarField& pB = thetaBroken_.primitiveFieldRef();
            forAll(pI, celli)
            {
                const scalar theta = tC[celli] + tH[celli];
                const scalar d = d0[celli];
                if (!(d >= 0 && d <= 1))
                {
                    FatalErrorInFunction
                        << "damageTransport contentPartition: the initial "
                        << "ruptured " << d << " in cell " << celli
                        << " is outside [0, 1]." << exit(FatalError);
                }
                pI[celli] = theta*(1.0 - d);
                pP[celli] = 0.0;
                pB[celli] = theta*d;
            }
            thetaIntact_.correctBoundaryConditions();
            thetaPending_.correctBoundaryConditions();
            thetaBroken_.correctBoundaryConditions();
        }
        // Old-time storage from the start, so fvm::ddt and the per-step
        // transport-inclusive |dd| read the start-of-step state.
        thetaIntact_.oldTime();
        thetaPending_.oldTime();
        thetaBroken_.oldTime();
        syncDamageFromPartition("initial condition");
    }
    // ---- S17: FIBRE-FORM RELEASE AND IMPLICIT ELASTIC COUPLING ----------
    //
    // research/designs/2026-09-26-S17-frozen-release-implicit-elastic-
    // startup.md (revision 4, APPROVED 2026-09-26), sections 2A, 2C, 4, 5.
    // The VALUE of the new switch first, then every combination.
    {
        const dictionary& solvers = mesh.solution().solversDict();
        for (const word& name : {word("Vs"), word("VsFinal")})
        {
            if (!solvers.found(name))
            {
                FatalErrorInFunction
                    << "elasticCoupling implicitTangent solves a Vs "
                    << "predictor; fvSolution has no '" << name << "' solver "
                    << "entry (design section 4)." << exit(FatalError);
            }
            const word solverType
            (
                solvers.subDict(name).lookup<word>("solver")
            );
            if
            (
                solverType == "PCG" || solverType == "PPCG"
             || solverType == "PPCR"
            )
            {
                FatalErrorInFunction
                    << "elasticCoupling implicitTangent: the Vs matrix is an "
                    << "M-matrix but ASYMMETRIC (the bounded upwind "
                    << "convection), so '" << name << "' cannot use "
                    << solverType << ", a symmetric-only solver: use "
                    << "PBiCGStab or smoothSolver (design 2C, section 4)."
                    << exit(FatalError);
            }
        }
    }

    // ---- S17b: SLACK FIBRES (V1), FIELD-UNIT FIBRE ARITHMETIC, THE WALL
    //      FACE IN BOTH RECONSTRUCTS ------------------------------------
    //
    // research/designs/2026-09-27-S17b-slack-fibre-field-units-wall-face.md
    // (revision 2, APPROVED 2026-09-27), section 2. Values first, then every
    // combination; each default is S17's behaviour.
    {
        // theta_T <= theta_P EXACTLY rests on the predictors running the
        // same matrix through the same operation sequence (06: "same
        // M-matrix, b_T <= b_P and monotone Gauss-Seidel"; red team 16:
        // 0 of 2384 cells on symGaussSeidel), and C_B/theta_T and
        // q_P/theta_T stay exact convex combinations only with the SAME
        // sweep count as well (16). Design section 8 a and g: every one of
        // the four predictors has an Euler ddt (a non-negative source) and
        // ONE identical fixed-sweep Gauss-Seidel entry, minIter = maxIter =
        // 3, Final included. Anything else is refused.
        const wordList fixedFields
        ({
            word("thetaPending"), word("thetaTaut"),
            word("thetaPendingB"), word("thetaPendingP")
        });
        forAll(fixedFields, fi)
        {
            const word ddtKey("ddt(" + fixedFields[fi] + ")");
            ITstream& ddtIs = mesh.schemes().ddt(ddtKey);
            ddtIs.rewind();
            const word ddtName(ddtIs);
            ddtIs.rewind();
            if (ddtName != "Euler")
            {
                FatalErrorInFunction
                    << "fibreSlack v1: theta_T <= theta_P exactly needs the "
                    << "predictors' ddt to be Euler (a non-negative source); "
                    << ddtKey << " is " << ddtName << " (design A1 and "
                    << "section 8 g; docket 06)." << exit(FatalError);
            }
        }
        const dictionary& solvers = mesh.solution().solversDict();
        if (!solvers.found("thetaPending"))
        {
            FatalErrorInFunction
                << "fibreSlack v1: fvSolution needs 'thetaPending' (design "
                << "section 8 a)." << exit(FatalError);
        }
        const dictionary& dRef = solvers.subDict("thetaPending");
        const label minR = dRef.lookupOrDefault<label>("minIter", 0);
        const label maxR = dRef.lookupOrDefault<label>("maxIter", -1);
        const word solverR(dRef.lookupOrDefault<word>("solver", word::null));
        const word smootherR
        (
            dRef.lookupOrDefault<word>("smoother", word::null)
        );
        if
        (
            minR != 3 || maxR != 3 || solverR != "smoothSolver"
         || (smootherR != "symGaussSeidel" && smootherR != "GaussSeidel")
        )
        {
            FatalErrorInFunction
                << "fibreSlack v1: 'thetaPending' must be the fixed-sweep "
                << "Gauss-Seidel entry (solver smoothSolver, smoother "
                << "symGaussSeidel or GaussSeidel, minIter = maxIter = 3); "
                << "found solver " << solverR << ", smoother " << smootherR
                << ", minIter " << minR << ", maxIter " << maxR
                << " (design section 8 a)." << exit(FatalError);
        }
        // every entry compared as written (macros expanded) with P2's
        OStringStream osRef;
        dRef.write(osRef, false);
        for (const word& sfx : {word(""), word("Final")})
        {
            forAll(fixedFields, fi)
            {
                const word name(fixedFields[fi] + sfx);
                if (!solvers.found(name))
                {
                    FatalErrorInFunction
                        << "fibreSlack v1: fvSolution needs '" << name
                        << "', the same fixed-sweep entry as 'thetaPending' "
                        << "(design section 8 a)." << exit(FatalError);
                }
                OStringStream os;
                solvers.subDict(name).write(os, false);
                if (os.str() != osRef.str())
                {
                    FatalErrorInFunction
                        << "fibreSlack v1: '" << name << "' differs from "
                        << "'thetaPending'; the four fibre predictors and "
                        << "each Final must run ONE identical fixed-sweep "
                        << "entry (theta_T <= theta_P exactly, and exact "
                        << "passenger ratios; design section 8 a)."
                        << exit(FatalError);
                }
            }
        }
    }
    // S17b A1, THE INITIAL CONDITION of the taut share: read all-or-none
    // with the pending fibre; a fresh start has no pending content, so 0.
    // Pending content with no taut share read cannot be split by guessing.
    {
        const bool hasT = thetaTaut_.headerOk();
        if (!hasT && gMax(thetaPending_.primitiveField()) > 0)
        {
            FatalErrorInFunction
                << "fibreSlack v1: the damage partition holds pending content "
                << "but no thetaTaut was read -- its taut share cannot be "
                << "invented." << exit(FatalError);
        }
        const scalarField& tT = thetaTaut_.primitiveField();
        const scalarField& pP0 = thetaPending_.primitiveField();
        forAll(tT, celli)
        {
            if (!(std::isfinite(tT[celli]) && tT[celli] >= 0 && tT[celli] <= pP0[celli]))
            {
                FatalErrorInFunction
                    << "fibreSlack v1: the read thetaTaut in cell " << celli
                    << " is " << tT[celli] << " against thetaPending "
                    << pP0[celli] << "; 0 <= theta_T <= theta_P is required."
                    << exit(FatalError);
            }
        }
        // the same matrix needs the same boundary coefficients: a read
        // thetaTaut must carry thetaPending's patch types (06's premise)
        forAll(thetaTaut_.boundaryField(), patchi)
        {
            const word tType(thetaTaut_.boundaryField()[patchi].type());
            const word pType(thetaPending_.boundaryField()[patchi].type());
            if (tType != pType)
            {
                FatalErrorInFunction
                    << "fibreSlack v1: thetaTaut's patch "
                    << mesh.boundary()[patchi].name() << " is " << tType
                    << " but thetaPending's is " << pType << "; theta_T <= "
                    << "theta_P exactly needs the same matrix, boundary "
                    << "coefficients included (design A1; docket 06)."
                    << exit(FatalError);
            }
        }
        thetaTaut_.oldTime();
    }
    s17bSlackEvents_ = 0;
    s17bSlackContent_ = 0;
    s17bSlackDdMax_ = 0;
    s17bSlackAtRest_ = 0;
    s17bSlackPreTMax_ = 0;
    s17bSlackPreTRatioMax_ = 0;
    s17bSlackPreQpMax_ = 0;
    s17bSlackPreQpRatioMax_ = 0;
    s17bSlackMuMax_ = 0;
    s17bSlackQpMax_ = 0;
    s17bMaxLfTransport_ = 0;
    s17bMaxLfRotation_ = 0;
    s17bMaxLfLaw_ = 0;
    s17bProjRaisedPresent_ = 0;
    s17bProjRaisedAbsent_ = 0;
    s17bSumLnMu_.setSize(mesh.nCells(), 0.0);
    s17bSumMinLnMu_.setSize(mesh.nCells(), 0.0);
    s17bLastSlackTime_.setSize(mesh.nCells(), -1.0);

    // ---- S18, THE SOFT STAGE: NO FROZEN PRESSURE, THE WALL-PATCH SPLIT,
    //      THE CONTENT-WEIGHTED INTACT PRESSURE, ONE C+ ORDER ------------
    //
    // research/designs/2026-09-29-S18-no-frozen-pressure-content-weight-
    // network-viscosity.md, rev. 10 (APPROVED 2026-09-29), section 2 A, B,
    // C, E, F. D (R3), I and H are deferred to OI-44 and are not here.
    // Values first, then each switch's dependency; each default is S17b's.
    s18CarrierDtLimit_ = great;
    s25DtFluidCo_ = great;
    s25DtWave_ = great;
    s25DtKV_ = great;
    s25DtKVTangent_ = great;
    s25DtNetDiff_ = great;
    s25DtMax_ = great;
    s25DtGrowth_ = great;
    s25NetDiffCell_ = -1;
    s25NetDiffIAS_ = 0;
    s25NetDiffR3As_ = 0;
    s25NetDiffW_ = 0;
    s25NetDiffA_ = 0;
    s18WallSplitDevMax_ = 0;
    Info<< "PHASE_CLOT_S18_CONFIG"
        << " networkPressureWeight=" << networkPressureWeight_
        << endl;

    // ---- S18p, THE em76 ATTRIBUTION PROBE: LOG-ONLY ----------------------
    // research/designs/2026-10-01-S18p-em76-attribution-probe.md, rev. 2,
    // section 2. projSiteLog yes counts P-LS by site and dumps raised
    // cells; nothing it computes is read by a field, flux, solve or dt.
    if (projSiteLog_ && Pstream::parRun())
    {
        FatalErrorInFunction
            << "projSiteLog yes is serial only: the cell dump's quotas (5 "
            << "raised cells per site per 0.25 ms window, 200 network-present "
            << "cells per run) and its cell indices are per process, and only "
            << "the master prints (S18p design section 2)." << exit(FatalError);
    }
    for (label k = 0; k < 2; ++k)
    {
        s18pCount_[k] = 0;
        s18pRaised_[k] = 0;
        s18pRaisedPresent_[k] = 0;
        s18pRaisedAbsent_[k] = 0;
        s18pExcessMax_[k] = 0;
        s18pWindow_[k] = -1;
        s18pWindowDumped_[k] = 0;
    }
    s18pPresentDumped_ = 0;
    Info<< "PHASE_CLOT_S18P_CONFIG"
        << " projSiteLog=" << (projSiteLog_ ? "yes" : "no")
        << endl;

    // ---- S19, THE BLOCK-COUPLED STEP AND R3 ----------------------------
    // research/designs/2026-10-03-S19-block-coupled-step.md, rev. 4
    // (APPROVED 2026-10-04), sections 2 and 12. Values first, then each
    // switch's requirements; each default is the shipped behaviour.
    if
    (
        momentumPressureCoupling_ != "segregated"
     && momentumPressureCoupling_ != "blockCoupled"
    )
    {
        FatalErrorInFunction
            << "momentumPressureCoupling '" << momentumPressureCoupling_
            << "' is not recognised: use segregated or blockCoupled."
            << exit(FatalError);
    }
    if (networkViscosity_ != "none" && networkViscosity_ != "kelvinVoigt")
    {
        FatalErrorInFunction
            << "networkViscosity '" << networkViscosity_ << "' is not "
            << "recognised: use none or kelvinVoigt." << exit(FatalError);
    }
    if (networkViscosity_ != "none" && !s19BlockCoupled())
    {
        FatalErrorInFunction
            << "networkViscosity " << networkViscosity_ << " needs "
            << "momentumPressureCoupling blockCoupled: R3 is unstable at the "
            << "design dt in every segregated placement (OI-44; S19 section "
            << "12)." << exit(FatalError);
    }
    if
    (
        networkViscosity_ != "none"
     && networkPressureWeight_ == "contentRatio"
    )
    {
        FatalErrorInFunction
            << "networkViscosity " << networkViscosity_ << " with "
            << "networkPressureWeight contentRatio: C stays off in S19 "
            << "(section 12)." << exit(FatalError);
    }
    if (networkViscosity_ != "none" && !(networkViscosityTau_ > 0))
    {
        FatalErrorInFunction
            << "networkViscosityTau " << networkViscosityTau_ << " must be "
            << "positive: eta_n = G tau_v/phi0 (S19 section 12)."
            << exit(FatalError);
    }
    // S19b (research/designs/2026-10-04-S19b-lumen-elimination.md rev. 2,
    // section 2.6): the value first, then the key outside blockCoupled.
    if (s19BlockCoupled())
    {
        if (networkPressureWeight_ == "contentRatio")
        {
            FatalErrorInFunction
                << "momentumPressureCoupling blockCoupled with "
                << "networkPressureWeight contentRatio: refused (S19 section "
                << "2; C off)." << exit(FatalError);
        }
        if (Pstream::parRun())
        {
            FatalErrorInFunction
                << "momentumPressureCoupling blockCoupled is serial: one "
                << "UMFPACK factorisation of the whole block (S19 sections 2, "
                << "4)." << exit(FatalError);
        }
        if (mesh.nSolutionD() != 2 || mesh.solutionD()[vector::Z] != -1)
        {
            FatalErrorInFunction
                << "momentumPressureCoupling blockCoupled solves the 2-D (x, "
                << "y) block, z empty; this mesh has " << mesh.nSolutionD()
                << " solution directions (S19 section 2)." << exit(FatalError);
        }
    }
    // ---- S22, THE BLOCK'S LINEAR SOLVER ----------------------------------
    // research/designs/2026-10-07-S22-pardiso-block-solver.md rev. 2
    // (APPROVED 2026-10-07), sections 2.3-2.6 and 4: the values first, then
    // the keys outside blockCoupled; MKL is loaded here, before step 1, so
    // the config line below can echo its version and library.
    if (blockLinearSolver_ != "pardiso" && blockLinearSolver_ != "umfpack")
    {
        FatalErrorInFunction
            << "blockLinearSolver '" << blockLinearSolver_ << "' is not "
            << "recognised: use pardiso or umfpack (S22 section 4)."
            << exit(FatalError);
    }
    if (blockSolverThreads_ < 0)
    {
        FatalErrorInFunction
            << "blockSolverThreads " << blockSolverThreads_ << " is negative: "
            << "use 0 (MKL_NUM_THREADS, else the physical cores; OpenBLAS "
            << "untouched under umfpack) or a positive count (S22 section 4)."
            << exit(FatalError);
    }
    if (phaseClotProperties_.found("blockSolverReproducible"))
    {
        const ITstream& reproIs =
            phaseClotProperties_.lookup("blockSolverReproducible");
        if
        (
            reproIs.size() != 1
         || !reproIs[0].isWord()
         || !Switch(reproIs[0].wordToken(), true).valid()
        )
        {
            FatalErrorInFunction
                << "blockSolverReproducible "
                << static_cast<const tokenList&>(reproIs) << " is not a bool: "
                << "use true or false (S22 section 4)." << exit(FatalError);
        }
        blockSolverReproducible_ = Switch(reproIs[0].wordToken());
    }
    if (!s19BlockCoupled())
    {
        static const char* const s22Keys[] =
        {
            "blockLinearSolver",
            "blockSolverThreads",
            "blockSolverReproducible"
        };
        for (const char* const key : s22Keys)
        {
            if (phaseClotProperties_.found(key))
            {
                FatalErrorInFunction
                    << "phaseClotProperties key `" << key << "` is read only "
                    << "under momentumPressureCoupling blockCoupled; under "
                    << momentumPressureCoupling_ << " it would be silently "
                    << "inert (S22 section 4)." << exit(FatalError);
            }
        }
    }
    label s22SolverThreads = 0;
    std::string s22ThreadsSource("none");
    label s22MklMaxThreads = 0;
    std::string s22MklVersion("none");
    std::string s22MklLibrary("none");
    // S25 section 12.2: mkl_cbwr_get(MKL_CBWR_BRANCH) after the solver's own
    // set, AUTO's branch on this CPU, and the CPU's model and ISA flags, once.
    std::string s25CbwrBranch("none");
    std::string s25CbwrAutoBranch("none");
    std::string s25CpuModel, s25CpuFlags;
    blockCoupled::cpuIdentity(s25CpuModel, s25CpuFlags);
    if (s19BlockCoupled() && blockLinearSolver_ == "pardiso")
    {
        int nThreads = 0;
        std::string err;
        if
        (
            !blockCoupled::resolveThreads
            (
                blockSolverThreads_, nThreads, s22ThreadsSource, err
            )
        )
        {
            FatalErrorInFunction
                << err.c_str() << exit(FatalError);
        }
        const char* mklRt = std::getenv("PHASE_CLOT_MKL_RT");
        const std::string mklPath =
            mklRt ? std::string(mklRt)
          : std::string(blockCoupled::mklRuntimeSoname);
        blockPardiso_.reset(new blockCoupled::PardisoLU());
        if
        (
            !blockPardiso_->load
            (
                mklPath, nThreads, bool(blockSolverReproducible_), err
            )
        )
        {
            FatalErrorInFunction
                << "momentumPressureCoupling blockCoupled, blockLinearSolver "
                << "pardiso: " << err.c_str() << exit(FatalError);
        }
        s22SolverThreads = blockPardiso_->threads();
        s22MklMaxThreads = blockPardiso_->mklMaxThreads();
        s22MklVersion = blockPardiso_->version();
        std::replace(s22MklVersion.begin(), s22MklVersion.end(), ' ', '_');
        s22MklLibrary = blockPardiso_->library();
        s25CbwrBranch = std::to_string(blockPardiso_->cbwrBranch());
        s25CbwrAutoBranch = std::to_string(blockPardiso_->cbwrAutoBranch());
    }
    else if (s19BlockCoupled() && blockSolverThreads_ > 0)
    {
        std::string err;
        if (!blockCoupled::setOpenblasThreads(blockSolverThreads_, err))
        {
            FatalErrorInFunction
                << err.c_str() << exit(FatalError);
        }
        s22SolverThreads = blockSolverThreads_;
        s22ThreadsSource = "blockSolverThreads";
    }
    Info<< "PHASE_CLOT_S19_CONFIG"
        << " momentumPressureCoupling=" << momentumPressureCoupling_
        << " networkViscosity=" << networkViscosity_
        << " networkViscosityTau=" << networkViscosityTau_
        << " etaNMax="
        << (
               networkViscosity_ == "none"
             ? 0.0
             : gMax(GField_.primitiveField())*networkViscosityTau_
              /skeletonVolumeFraction_
           )
        << " nCorrectorsUsed=" << (s19BlockCoupled() ? "no" : "yes")
        // S22 section 2.7 (none outside blockCoupled; the MKL keys none
        // under umfpack; mklVersion with its spaces as underscores).
        << " blockLinearSolver="
        << (s19BlockCoupled() ? blockLinearSolver_ : word("none"))
        << " solverThreads=" << s22SolverThreads
        << " solverThreadsSource=" << s22ThreadsSource.c_str()
        << " mklMaxThreads=" << s22MklMaxThreads
        << " blockSolverReproducible="
        << (
               s19BlockCoupled()
             ? (blockSolverReproducible_ ? "true" : "false")
             : "none"
           )
        << " mklVersion=" << s22MklVersion.c_str()
        << " mklLibrary=" << s22MklLibrary.c_str()
        << " mklCbwrBranch=" << s25CbwrBranch.c_str()
        << " mklCbwrAutoBranch=" << s25CbwrAutoBranch.c_str()
        << " cpuModel=" << s25CpuModel.c_str()
        << " cpuFlags=" << s25CpuFlags.c_str()
        << endl;

    // S17 A, THE INITIAL CONDITION: the fibre is read all-or-none; on a
    // fresh start there is no pending content, so the contents are zero and
    // the fibre is the isotropic one (n = e_x, lf = 1, p_P = 0). A partition
    // that holds pending content with no fibre to carry it is refused.
    {
        const bool hasN = pendingAxis_.headerOk();
        const bool hasL = pendingStretch_.headerOk();
        const bool hasPp = pendingPressure_.headerOk();
        if ((hasN || hasL || hasPp) && !(hasN && hasL && hasPp))
        {
            FatalErrorInFunction
                << "networkRupture fibreRelease: the pending fibre is read "
                << "all-or-none; found pendingAxis " << hasN
                << ", pendingStretch " << hasL << ", pendingPressure "
                << hasPp << "." << exit(FatalError);
        }
        if (!hasN && gMax(thetaPending_.primitiveField()) > 0)
        {
            FatalErrorInFunction
                << "networkRupture fibreRelease: the damage partition holds "
                << "pending content but no pendingAxis/pendingStretch/"
                << "pendingPressure were read -- the fibre it carries cannot "
                << "be invented." << exit(FatalError);
        }
        // S25 C: the contents, when written, are read back as written (all
        // or none); otherwise they are re-derived below, as before S25.
        const bool hasPB = thetaPendingB_.headerOk();
        const bool hasPP = thetaPendingP_.headerOk();
        if (hasPB != hasPP)
        {
            FatalErrorInFunction
                << "networkRupture fibreRelease: the pending contents are "
                << "read all-or-none; found thetaPendingB " << hasPB
                << ", thetaPendingP " << hasPP << " (design 2026-10-10-S25 "
                << "section 2C)." << exit(FatalError);
        }
        const bool pendingContentsRead = hasPB && hasPP;
        // S17b A1: the contents are the CARRIER's (theta_T under v1).
        const scalarField& pP = fibreCarrier().primitiveField();
        vectorField& nI = pendingAxis_.primitiveFieldRef();
        const scalarField& lI = pendingStretch_.primitiveField();
        const scalarField& ppI = pendingPressure_.primitiveField();
        symmTensorField& cB = thetaPendingB_.primitiveFieldRef();
        scalarField& qP = thetaPendingP_.primitiveFieldRef();
        forAll(pP, celli)
        {
            if
            (
                !(std::isfinite(lI[celli]) && lI[celli] >= 1
               && std::isfinite(ppI[celli]) && mag(nI[celli]) > 0)
            )
            {
                FatalErrorInFunction
                    << "networkRupture fibreRelease: the read fibre in cell "
                    << celli << " is not a fibre (pendingStretch "
                    << lI[celli] << ", |pendingAxis| " << mag(nI[celli])
                    << ", pendingPressure " << ppI[celli] << ")."
                    << exit(FatalError);
            }
            // unit again after the written precision rounded it
            const vector n(nI[celli]/mag(nI[celli]));
            nI[celli] = n;
            const double nA[3] = {n.x(), n.y(), n.z()};
            double B[6];
            if (!pendingContentsRead)
            {
                fibreReleaseKernel::fibre(nA, lI[celli], B);
                cB[celli] =
                    pP[celli]*symmTensor(B[0], B[1], B[2], B[3], B[4], B[5]);
                qP[celli] = pP[celli]*ppI[celli];
            }
        }
        pendingAxis_.correctBoundaryConditions();
        thetaPendingB_.correctBoundaryConditions();
        thetaPendingP_.correctBoundaryConditions();
        thetaPendingB_.oldTime();
        thetaPendingP_.oldTime();
        syncPendingFibres("initial condition");
    }
    s17TransferB1Max_ = 0;
    s17Transfers_ = 0;
    s17RotRatioMedian_ = 1;
    s17RotCells_ = 0;
    s17RotLdtMedian_ = 0;
    s17RotCandidates_ = 0;
    s17EigRelMax_ = 0;
    s17UniUp_ = 0;
    s17ProjCount_ = 0;
    s17ProjRaised_ = 0;
    s17ProjExcessMax_ = 0;
    s17ProjTies_ = 0;
    s17SplitDevMax_[0] = 0;
    s17SplitDevMax_[1] = 0;
    s17SplitDevMax_[2] = 0;
    s17SplitSites_ = 0;
    s17P2Max_ = 0;
    s17W2Max_ = 0;
    s17W1MaxDeg_ = 0;
    s17W1Over5_ = 0;
    s17W1Over10_ = 0;
    s17ResetCells_ = 0;
    s17PendingPressureMax_ = 0;
    s17SecondDiffSteps_ = 0;

    // ONE LINE, compared with child.json by assess_case.py (the S17 config
    // gate).
    Info<< "PHASE_CLOT_S17_CONFIG"
        << " pendingForm=" << "fibre"
        << endl;

    s16LawReleaseMax_ = 0;
    s16LawRate_ = 0;
    s16PendingMax_ = 0;
    s16DeltaDMax_ = 0;
    s16DampedStiffMax_ = 0;
    s16WeightAbsDevMax_ = 0;
    s16WeightSites_ = 0;
    s16FoutMax_ = 0;
    s16MuMin_ = 1;
    s16MuMean_ = 1;
    s16MirrorDevMax_ = 0;

    // ONE LINE, compared with child.json by assess_case.py (the config gate).
    Info<< "PHASE_CLOT_S16_CONFIG"
        << " networkCriticalStretch=" << networkCriticalStretch_
        << " networkReleaseTime=" << networkReleaseTime_
        << " networkDamageOnsetStretch=" << networkDamageOnsetStretch_
        << " networkDamageStretchScale=" << networkDamageStretchScale_
        << " releaseLaw="
        << (
             "progressive+terminal"
           )
        << endl;

    Info<< "PHASE_CLOT_S13_CONFIG"
        << " networkPressureWeight=" << networkPressureWeight_
        << endl;

    // ---- S23: THE VOLUMETRIC TEAR (design 2026-10-07-S23 rev. 3) ----------
    // Section 4: R_c at or below 1 would tear every cell whose content has
    // not grown past its deposit (R = 1 is the fresh, stress-free network),
    // and a non-finite R_c is no criterion. Refused before step 1. (The
    // second refusal of section 4, the key outside a fibreRelease build,
    // has no reachable case: S21 built networkRupture fibreRelease in, so
    // every build is one.)
    if (!(std::isfinite(volumetricTearRatio_) && volumetricTearRatio_ > 1))
    {
        FatalErrorInFunction
            << "volumetricTearRatio " << volumetricTearRatio_ << " is the "
            << "volumetric tear ratio R_c of S23 (a skeleton cell tears when "
            << "thetaNetRef >= R_c*theta) and must be finite and exceed 1 "
            << "(21.95 = lc^3, lc 2.8, source 28; waiver DECISIONS "
            << "2026-10-07)." << exit(FatalError);
    }
    s23VolEvents_ = 0;
    s23StretchEvents_ = 0;
    s23VolEventsTotal_ = 0;
    s23StretchEventsTotal_ = 0;
    s23A1Deferred_ = 0;
    s23RCells_ = 0;
    s23RMax_ = 0;
    s23RP50_ = 0;
    s23RP90_ = 0;
    s23RP99_ = 0;
    s23TornCells_ = 0;
    s23TornFaces_ = 0;
    s23ContentResInt_ = 0;
    s23OutfluxInt_ = 0;
    s23ContentV0_ = -1;
    s23F1WDev_[0] = 0;
    s23F1WDev_[1] = 0;
    s23F1WCells_[0] = -1;
    s23F1WCells_[1] = -1;
    s23F1WZeroOcc_ = 0;
    s23PtUsedIndex_ = -1;

    // ---- S23 rev. 4.1 section 13.2 item 4: R3 IS REQUIRED ON THIS BUILD --
    // R3 (the Kelvin-Voigt network viscosity) is physically required: fibrin
    // tan delta is 0.67-4.7 (sources 93 and 94; S24 physics 02), and R3 is
    // what holds the fringe on the real row (report 18, R3/Kd 28-307). So
    // networkViscosity none -- the default when the key is absent -- is the
    // tan delta -> 0 sensitivity limit, not physics, and runs only when
    // labelled with sensitivityLimit tanDelta0. kelvinVoigt needs an
    // explicit tau_v >= 5e-4 s (the physics-assured floor; the default
    // 2.5e-4 s is below it, and the check above is only tau_v > 0). The
    // marker labels the R3-OFF limit only, so it is refused with
    // kelvinVoigt (the CE, ratifying report 24: otherwise a marked
    // kelvinVoigt with a tiny tau_v would waive the floor). 5e-4 exactly is
    // accepted. Mirrors the generator's refusals (defence in depth against
    // a hand-edited dictionary). S25 F: the floor is a physical time in
    // seconds, so it scales: timeScale*5e-4 (timeScale 1.0 is the S23 floor
    // exactly).
    const scalar s23TauVFloor = timeScale_*5e-4;
    if
    (
        phaseClotProperties_.found("sensitivityLimit")
     && sensitivityLimit_ != "tanDelta0"
    )
    {
        FatalErrorInFunction
            << "sensitivityLimit '" << sensitivityLimit_ << "' is not "
            << "recognised: its only value is tanDelta0, the labelled tan "
            << "delta -> 0 sensitivity limit (R3 off). Remove the key on "
            << "every physics run (design 2026-10-07-S23 section 13.2 item "
            << "4)." << exit(FatalError);
    }
    const bool s23Marked = sensitivityLimit_ == "tanDelta0";
    if (s23Marked && networkViscosity_ != "none")
    {
        FatalErrorInFunction
            << "sensitivityLimit tanDelta0 labels the R3-off (tan delta -> 0) "
            << "sensitivity limit and is refused with networkViscosity "
            << networkViscosity_ << ": the labelled arm runs networkViscosity "
            << "none, and an R3-on child carries no marker (design "
            << "2026-10-07-S23 section 13.2 item 4)." << exit(FatalError);
    }
    if (networkViscosity_ == "none" && !s23Marked)
    {
        FatalErrorInFunction
            << "networkViscosity none (the default when the key is absent) "
            << "is refused on the S23 build: R3, the Kelvin-Voigt network "
            << "viscosity, is physically required (fibrin tan delta "
            << "0.67-4.7, sources 93 and 94). Set networkViscosity "
            << "kelvinVoigt with networkViscosityTau >= " << s23TauVFloor
            << " s, or label the tan delta -> 0 sensitivity arm with "
            << "sensitivityLimit tanDelta0 (design 2026-10-07-S23 section "
            << "13.2 item 4)." << exit(FatalError);
    }
    if (networkViscosity_ == "kelvinVoigt" && !s23Marked)
    {
        if (!phaseClotProperties_.found("networkViscosityTau"))
        {
            FatalErrorInFunction
                << "networkViscosity kelvinVoigt without an explicit "
                << "networkViscosityTau is refused on the S23 build: the "
                << "default " << networkViscosityTau_ << " s is below the "
                << "physics-assured floor " << s23TauVFloor << " s (design "
                << "2026-10-07-S23 section 13.2 item 4)." << exit(FatalError);
        }
        if (!(networkViscosityTau_ >= s23TauVFloor))
        {
            FatalErrorInFunction
                << "networkViscosityTau " << networkViscosityTau_ << " s is "
                << "below the S23 floor " << s23TauVFloor << " s (the "
                << "physics-assured 5e-4 and 1.5e-3 s; design 2026-10-07-S23 "
                << "section 13.2 item 4)." << exit(FatalError);
        }
    }

    // ONE LINE, compared with child.json by assess_case.py. rev. 4.1: T is
    // majority membership, with its threshold and d_p as the solver
    // evaluates them, and the sensitivity marker (none when absent).
    Info<< "PHASE_CLOT_S23_CONFIG"
        << " volumetricTearRatio=" << volumetricTearRatio_
        << " tornSet=" << "majorityIsSkel"
        << " tornFace=" << "packingTimesMaterial"
        << " progressiveCeiling=" << progressiveDamageCeiling()
        << " tornThreshold="
        << damagePartitionKernel::tornThreshold(progressiveDamageCeiling())
        << " sensitivityLimit=" << sensitivityLimit_
        << endl;

    // ---- S25 (design 2026-10-10-S25, build 1: parts B-G) -------------------
    // G: the deposition acceptance cap is a packing fraction. The transfer
    // kernel refuses one outside (0, 1) on its first call; refused here, at
    // construction, by name.
    if
    (
        !(
            std::isfinite(depositionContentCap_)
         && depositionContentCap_ > 0
         && depositionContentCap_ < 1
        )
    )
    {
        FatalErrorInFunction
            << "phaseClotProperties key `depositionContentCap` = "
            << depositionContentCap_ << " must be finite and in (0, 1): it "
            << "is the as-deposited packing fraction up to which a cell "
            << "accepts new deposit (default phiSMax; design 2026-10-10-S25 "
            << "section 2G)." << exit(FatalError);
    }
    // ONE LINE: the S25 keys as the solver evaluates them.
    Info<< "PHASE_CLOT_S25_CONFIG"
        << " timeScale=" << timeScale_
        << " networkReleaseTime=" << networkReleaseTime_
        << " networkViscosityTau=" << networkViscosityTau_
        << " tauVFloor=" << s23TauVFloor
        << " depositionContentCap=" << depositionContentCap_
        << " plateletTransport=" << "deviationR1"
        << " restartState=" << "persisted"
        << " skeletonLaw=" << skeletonLaw_
        << " elasticCourantModulus=" << elasticCourantModulus_
        << " networkPole=" << networkPole()
        << " ltE1=" << skeletonLawCoeffs_.E1
        << " ltE2=" << skeletonLawCoeffs_.E2
        << " ltD1=" << skeletonLawCoeffs_.D1
        << " ltD2=" << skeletonLawCoeffs_.D2
        << " ltKappa=" << skeletonLawCoeffs_.kappa
        << endl;

    {
        const scalar phi0 = skeletonVolumeFraction_;
        const scalar Kshell =
            sqr(1.0 - phi0)*nu_.value()/max(k0p_.value(), vSmall);
        const scalar Kcore =
            sqr(1.0 - phi0)*nu_.value()/max(k0pcore_.value(), vSmall);
        Info<< "PHASE_CLOT_S12_CONFIG"
            << " intrinsicDragAtReferenceShell=" << Kshell
            << " intrinsicDragAtReferenceCore=" << Kcore
            << " diluteFloorK1=" << skeletonDiluteDragRate_
            << endl;
    }

    // PRIME THE CONTENT BALANCE, OR THE FIRST STEP BOOKS THE WHOLE CLOT AS
    // CREATION.
    //
    // previousContentVolume_ is constructed at 0 and written only at the end
    // of the emitter, so without this line the first PHASE_CLOT_CONTENT_BALANCE
    // reports contentDVdt = contentVolume/deltaT -- the entire seeded clot
    // appearing out of nothing in one step -- and the assessor's drift
    // integral, which weights each residual by its interval, collects exactly
    // V0 from that one step. Run 20260920-220246 was graded INADMISSIBLE at
    // "+100.0% cumulative drift" on a run whose other 271 steps sum to
    // -1.2e-11 of V0 and whose worst single residual is 4.3e-22.
    //
    // This is the sibling of the previousInventory_ prime above; the two sat
    // on ADJACENT LINES before S6c-2, as
    //
    //     previousClotVolume_ = fvc::domainIntegrate(scalar(1) - q_).value();
    //
    // and that line was deleted with the legacy indicator rather than
    // migrated, because it read 1 - q in the retired form. The member's
    // rename to previousContentVolume_ kept the (0) initialiser, so nothing
    // was left to complain.
    //
    // It must sit AFTER the content-seed block: thetaC_/thetaH_ are assigned
    // there, and priming ahead of it would store zero and reproduce the bug.
    // theta_ref*(1 - q) is the same integral the old line took, in content
    // currency -- measured 0.45 * 1.3482e-15 = 6.0669e-16, which is the
    // contentVolume the run reported at its first step.
    previousContentVolume_ =
        fvc::domainIntegrate(thetaC_ + thetaH_).value();

    // q is a derived but persistent physical indicator.  Allocate its old
    // time before the first phase update so fvm::ddt(q,w) represents the
    // conservative product history from the first physical step onward.
    q_.oldTime();

    // ---- STAGE 2 INITIALISATION ----
    //
    // epsP_ must hold the CORRECT value before its oldTime() is instantiated,
    // because instantiation copies the current field. Assign, then instantiate,
    // then let the registry rotate it from the next step onward. Getting this
    // order wrong makes ddt(eps, nP) lose the porosity change silently; the
    // per-step budget in solvePlateletContent() is what would catch it.
    epsP_ = scalar(1) - (thetaC_ + thetaH_);
    epsP_.correctBoundaryConditions();
    epsP_.oldTime();
    nP_.oldTime();
    NP_ = epsP_*nP_;
    NP_.correctBoundaryConditions();

    checkPlateletConfiguration();


    // ---- S6c-2: THE KEY VALUES THAT DIED WITH c --------------------------
    //
    // Deleting c left five dictionary values with no implementation. Each is
    // REFUSED here rather than quietly treated as its surviving sibling: a
    // case that names a dead value would otherwise run something other than
    // what it says, which is the silent class this project keeps being
    // caught by. The list is derived from the site table in
    // research/designs/2026-09-20-S6c-2-delete-c.md, not from memory.
    //
    // The keys themselves survive one release so the refusal has something
    // to fire on; S7 retires them.
    {
        // mobility and epsilonC are no longer read at all. A case still
        // setting a NONZERO mobility is asking for the Cahn-Hilliard term,
        // and getting silence instead would be a model change disguised as a
        // no-op -- so it is refused. A mobility of exactly 0, which is every
        // main-case child on record, is accepted and ignored.
        // READ IT THE WAY THE DELETED MEMBER DID. Every case on record
        // writes `mobility [0 4 -1 0 0 0 0] 0;` -- a DIMENSIONED entry --
        // and readScalar() on that token stream dies on the '[' before the
        // guard can decide anything. Run 20260920-190025 was lost to exactly
        // that: "wrong token type - expected Scalar, found the punctuation
        // token '['", at construction, with the solver otherwise healthy.
        // dimensioned<Type>::initialise() reads an optional name, optional
        // units and then the value, so this form accepts both the
        // dimensioned spelling every generator emits and a bare `mobility 0;`.
        if (phaseClotProperties_.found("mobility"))
        {
            const dimensionedScalar mobilitySet
            (
                phaseClotProperties_.lookup("mobility")
            );

            if (mag(mobilitySet.value()) > vSmall)
            {
                FatalErrorInFunction
                    << "mobility is nonzero, but the Cahn-Hilliard term"
                    << " laplacian(mobility, muC) was deleted in S6c-2"
                    << " together with c and muC. A nonzero mobility can no"
                    << " longer do anything, and silently ignoring it would"
                    << " be a model change reported as a no-op. Set mobility"
                    << " to 0."
                    << exit(FatalError);
            }
        }
    }

    // ---- S6a: WHAT DEFINES q ----------------------------------------------
    {
        // The ratio divides by theta_ref. A solver that reached here with the
        // default 0.01 would rescale q by 45x and every consumer with it.
        if (skeletonVolumeFraction_ <= small)
        {
            FatalErrorInFunction
                << "phaseFluidFractionBasis 'content' divides by"
                << " skeletonVolumeFraction (found "
                << skeletonVolumeFraction_ << ")."
                << exit(FatalError);
        }
    }

    // ---- S6b-1: WHICH FIELD THE PHASE LIMITER READS -----------------------
    {
        if (skeletonVolumeFraction_ <= small)
        {
            FatalErrorInFunction
                << "phaseLimiterBasis 'content' divides by"
                << " skeletonVolumeFraction (found "
                << skeletonVolumeFraction_ << ")."
                << exit(FatalError);
        }
    }

    // ---- THE INDICATOR BASIS ---------------------------------------------
    {
        // THE DIVISOR. theta/theta_ref is the whole substitution and
        // theta_ref is a case parameter, so a case that sets it to zero would
        // divide by zero in seven places rather than in one obvious one.
        if (skeletonVolumeFraction_ <= 0)
        {
            FatalErrorInFunction
                << "phaseIndicatorBasis 'content' divides by"
                << " skeletonVolumeFraction, which is "
                << skeletonVolumeFraction_ << ". The content indicator is"
                << " theta/theta_ref and has no meaning at theta_ref <= 0."
                << exit(FatalError);
        }
    }

    {
        // The same divisor guard the indicator carries, for the same reason:
        // theta_ref is a case parameter and dividing by zero in the delta
        // would surface as an infinite bindRate inside the transfer kernel
        // rather than here.
        if (skeletonVolumeFraction_ <= 0)
        {
            FatalErrorInFunction
                << "phaseDeltaBasis 'content' divides by"
                << " skeletonVolumeFraction, which is "
                << skeletonVolumeFraction_ << ". The content surface delta is"
                << " |grad theta|/theta_ref and has no meaning at"
                << " theta_ref <= 0."
                << exit(FatalError);
        }
    }

    {
        if (skeletonVolumeFraction_ <= 0)
        {
            FatalErrorInFunction
                << "phaseNormalBasis 'content' divides by"
                << " skeletonVolumeFraction, which is "
                << skeletonVolumeFraction_ << ". The content normal is"
                << " -2*grad((thetaC + thetaH)/theta_ref) and has no meaning"
                << " at theta_ref <= 0."
                << exit(FatalError);
        }
    }

    {
        // The conversion is dc = -(2/theta_ref)*dtheta and theta_ref is the
        // divisor. Caught here rather than as an infinite increment applied
        // to c inside solvePlateletContent().
        if (skeletonVolumeFraction_ <= 0)
        {
            FatalErrorInFunction
                << "phaseGrowthDriver 'transfer' divides by"
                << " skeletonVolumeFraction, which is "
                << skeletonVolumeFraction_ << ". The conversion from a"
                << " content increment to a phase increment is"
                << " dc = -(2/theta_ref)*dtheta and has no meaning at"
                << " theta_ref <= 0."
                << exit(FatalError);
        }
    }

    Info<< "PHASE_CLOT_GROWTH_DRIVER"
        << " driver=transfer"
        << " frontTerm=removed"
        << " cIncrement=-(2/theta_ref)*(thetaNew-thetaStar)"
        << " thetaRef=" << skeletonVolumeFraction_
        << " VnStaysLiveIn=surfaceLossCoeff"
        << " phaseGradientMigrated=yes(rebuilt-on-theta-in-S6c-2)"
        << endl;



    // W IS CARRIED BY THE PHASE FIELD'S FLUX AND MUST USE THE PHASE FIELD'S
    // OPERATOR. Run 20260819-231939 established the failure this prevents: c
    // rode phiReleased (the rigid-body flux inside the clot) while W rode phi_
    // (the fluid flux, which the Brinkman penalty drives to zero exactly
    // there), so w = W/q was a ratio of two differently-carried fields and
    // left 1 on the FIRST step, before anything could grow. Full argument in
    // the header note on plateletContentFormulation_.
    //
    // The check is token-identity against div(phiReleased,c) rather than a
    // list of approved spellings. The requirement is not "vanLeer" or
    // "bounded" in the abstract -- it is THE SAME OPERATOR, and only the c
    // entry knows what that is. Checking against a hardcoded string would go
    // stale the first time the c scheme is retuned, silently, which is the
    // class of defect this project keeps being caught by.
    {
        // THE Wp EQUATION NO LONGER HAS A SCHEME OF ITS OWN, and that is the
        // fix rather than an omission. It reuses the face weights captured
        // from div(phiReleased,c) (see phaseFieldFaceWeights), so there is
        // nothing left for a div(phiReleased,Wp) entry to control.
        //
        // The guard this replaces required the two entries to be
        // TOKEN-IDENTICAL. That was necessary and it was not sufficient, and
        // believing it was sufficient cost three runs. vanLeer weights are a
        // function of the transported field, so identical strings on one
        // shared flux still assembled different matrices: run 20260822-103655
        // measured |Wp - q| = 2.27e-04 under vanLeer against 5.85e-12 under
        // field-independent linear weights, with every source term zero and
        // Wp initialised equal to q.
        //
        // A leftover div(phiReleased,Wp) entry is now INERT, and inert config
        // that looks live is a trap -- someone will tune it and measure
        // nothing. It is refused rather than ignored.
        if (mesh_.schemes().dict().subDict("divSchemes").found("div(phiReleased,Wp)"))
        {
            FatalErrorInFunction
                << "div(phiReleased,Wp) is present in divSchemes but is no "
                << "longer used." << nl
                << "The platelet content equation reuses the face weights of "
                << "div(phiReleased,c) directly, because matching the scheme "
                << "NAME is not enough -- vanLeer weights depend on the field "
                << "being transported, so c and Wp assembled different "
                << "matrices from identical strings (run 20260822-103655: "
                << "|Wp - q| 2.27e-04 vanLeer vs 5.85e-12 linear)." << nl
                << "Remove the entry. Leaving it implies a control that does "
                << "not exist."
                << exit(FatalError);
        }

        // The capture in phaseFieldFaceWeights() consumes two leading tokens
        // and needs the interpolation scheme to follow them. Checked here so
        // the failure is at construction rather than at the first solve.
        ITstream& cStream = mesh_.schemes().div("div(phiReleased,c)");
        cStream.rewind();
        const word wrapper(cStream);
        const word gauss(cStream);
        cStream.rewind();

        if (wrapper != "bounded" || gauss != "Gauss")
        {
            FatalErrorInFunction
                << "plateletContentFormulation requires div(phiReleased,c) to "
                << "be spelled `bounded Gauss <limiter>`; its weights are "
                << "captured and reused by the Wp equation." << nl
                << "  found: " << wrapper << ' ' << gauss << " ..." << nl
                << "`bounded` is not decorative here: phiReleased jumps from "
                << "zero to Ubody across the mask boundary and is deliberately "
                << "not solenoidal."
                << exit(FatalError);
        }
    }

    // A core/shell case whose shell indicator never leaves 1 has no core, and
    // would run as a single soft material that tears everywhere -- looking
    // exactly like the run it was built to replace. Checked on the VALUE
    // rather than on whether a file exists, because a present-but-uniform
    // field is the failure that would otherwise be silent.
    if (coreShell_ && gMin(shell_.primitiveField()) > 0.5)
    {
        FatalErrorInFunction
            << "coreShell is on, but the `shell` indicator is above 0.5 in "
            << "every cell: min = " << gMin(shell_.primitiveField()) << nl
            << "There is no core. The initialiser must write 0/shell with 0 "
            << "inside the core region." << nl
            << "See research/designs/AM2020_core_shell_design.md"
            << exit(FatalError);
    }

    // skeletonPinStart is GONE, and a case that still sets it is refused
    // rather than silently ignored. Its switch-on threshold left cells with
    // neither stiffness nor pin -- see the note in solveSkeleton() -- and a
    // stale entry would now mean nothing while looking like it still tuned
    // the pin.
    if (phaseClotProperties_.found("skeletonPinStart"))
    {
        FatalErrorInFunction
            << "skeletonPinStart is no longer used and has been replaced by "
            << "skeletonPinExponent." << nl
            << "The old threshold form switched the pin on only above a value "
            << "of q, which left a band of cells with 0.2-5% of the material "
            << "stiffness and no pin at all; run 20260815-201949 measured "
            << "420x the pinned displacement there and the skeleton tipped "
            << "over at t = 6.2e-4 s." << nl
            << "The pin is now q^skeletonPinExponent, which has no such gap. "
            << "Remove skeletonPinStart from phaseClotProperties."
            << exit(FatalError);
    }

    if (skeletonPinExponent_ < 1)
    {
        FatalErrorInFunction
            << "skeletonPinExponent must be at least 1; got "
            << skeletonPinExponent_ << nl
            << "Below 1 the pin does not fall away inside the clot and it "
            << "clamps the clot's own surface, which is the artefact the "
            << "diffuse-domain option exists to remove."
            << exit(FatalError);
    }

    // WHAT MATERIAL ACTUALLY GOT BUILT, once, at construction. Every number
    // that decides where this clot fails is on one line, so a mis-wired
    // dictionary is visible in the first screen of the log instead of being
    // inferred from the fields afterwards.
    label coreCells = 0;
    forAll(shell_.primitiveField(), celli)
    {
        if (shell_.primitiveField()[celli] < 0.5)
        {
            ++coreCells;
        }
    }

    // A MISSPELLED MODEL NAME MUST NOT FALL BACK SILENTLY. lookupOrDefault
    // would turn "neohookean" or "neoHookian" into stVenantKirchhoff and the
    // run would look normal while answering a different question -- the same
    // silent-wrong-answer class as the permeability floor and the missing
    // linear solver. Only the two known names are accepted.

    // Neo-Hookean IS a finite-strain law: its stress is built from C, J and
    // C^-1, which are identities at zero displacement and meaningless as a
    // small-strain approximation. Silently ignoring it under finiteStrain
    // false would run St Venant-Kirchhoff while the dictionary said otherwise.

    Info<< "PHASE_CLOT_CONSTITUTIVE model=" << "neoHookean"
        << " finiteStrain=" << "true" << endl;

    Info<< "PHASE_CLOT_MATERIAL coreShell=" << Switch(coreShell_)
        << " Gshell=" << G_.value() << " Gcore=" << Gcore_.value()
        << " Gcshell=" << Gc_.value() << " Gccore=" << Gccore_.value()
        << " k0pshell=" << k0p_.value() << " k0pcore=" << k0pcore_.value()
        << " GfieldMin=" << gMin(GField_.primitiveField())
        << " GfieldMax=" << gMax(GField_.primitiveField())
        << " GcfieldMin=" << gMin(GcField_.primitiveField())
        << " GcfieldMax=" << gMax(GcField_.primitiveField())
        << " coreCells=" << returnReduce(coreCells, sumOp<label>())
        << " diffuseDomain=" << "true"
        << " solidFloor=" << skeletonSolidFloor_
        << " pinExponent=" << skeletonPinExponent_
        << " permeabilityFloor=" << permeabilityFloor_
        << endl;

    // S25.1 A (OI-77; design 2026-10-13-S25.1 rev. 2 section A; docket
    // 20261013-1405, 01 Q1 and 03 Q1-Q2): THE PHASE WEIGHTS' OLD TIME ON A
    // RESTART. phiSkel_ (read) and phiFluid_ (NO_READ, so 1 here) had no
    // stored old time, so the first oldTime() -- inside fvm::ddt(phiSkel_,
    // Vs_) and fvm::ddt(phiFluid_, U_), AFTER momentumPredictor() has updated
    // both -- copied level n+1 (OF-13 OldTimeField.C:322-349), and the first
    // restarted step solved phi^{n+1}(X - X^n)/dt in draining cells. Here
    // phiFluid_ takes the written level's value (the operation the step
    // itself applies, on the same double) and BOTH old times are stored. The
    // next step's first write rotates the CURRENT value -- level n -- into
    // them (storeOldTimesInner: field0 = field), as on the continuous run, so
    // what is load-bearing is that the old times EXIST and that phiFluid_
    // holds 1 - phiSkel_ when the step starts (docket 20261013-1405, 04,
    // host test se04_restart_oldtime_host.py). A cold start reads no
    // phiSkel and takes neither branch. Wp_'s old time (read only by the LED
    // audit in solvePlatelets()) is stored on a restart only, after Wp_'s
    // last constructor write (updateDerivedFields() above). No field read
    // here is restored by restoreRestartState().
    //
    // A restart with no phiSkel file (a directory written before S25) is
    // REFUSED: there is no written level to restore, and today's lazy copy
    // would silently keep the defect.
    const bool restartS251 = runTime.value() > 0;
    if (restartS251 && !phiSkel_.headerOk())
    {
        FatalErrorInFunction
            << "restart at time " << runTime.name() << " has no phiSkel "
            << "file (" << phiSkel_.objectPath() << "): the phase weights' "
            << "old time is the written level n and cannot be restored "
            << "without it (design 2026-10-13-S25.1 section A, OI-77). "
            << "Restart from a directory written by S25 or later, or start "
            << "from t = 0." << exit(FatalError);
    }
    if (phiSkel_.headerOk())
    {
        phiFluid_ = 1.0 - phiSkel_;
        phiSkel_.oldTime();
        phiFluid_.oldTime();
    }
    if (restartS251)
    {
        Wp_.oldTime();
    }

    // S25 C: LAST, after every seed above, so a restored value is the one
    // the first step reads.
    restoreRestartState();
}


void Foam::solvers::phaseClotAxisymmetricFluid::updateMaterialFields()
{
    if (!coreShell_)
    {
        // Uniform, and equal to the scalars the dictionary already declared.
        // Assigning the dimensionedScalar sets every cell to the same double,
        // so every product downstream is bit-for-bit what it was when these
        // were dimensionedScalars.
        GField_ = G_;
        lambdaField_ = lambdaLame_;
        GcField_ = Gc_;
        k0pField_ = k0p_;
        return;
    }

    // GEOMETRIC blend, not linear. G spans 10x, k0p 500x and Gc nine decades,
    // and a linear blend of quantities that far apart is not a blend at all:
    // at s = 0.5 a linear Gc between 5.547 and 1e-9 is 2.77, i.e. the core
    // value, so the transition would collapse onto the last cell before the
    // shell and the material interface would be a step. The geometric mean
    // puts s = 0.5 at sqrt(Gc_core*Gc_shell) and gives a monotone transition
    // that is symmetric in the logarithm -- the natural interpolation for
    // positive quantities spanning decades. Both endpoints are reproduced
    // exactly.
    //
    // Written as exp(s*log(ratio)) rather than pow(field, field) so the only
    // library calls are exp() on a dimensionless field and log() on a plain
    // double.
    // S15: under contentComposition shell_ = thetaH/theta is in [0, 1] by
    // construction (checked where it is set), so it is used as it is. The
    // transported label keeps its clip, exactly as shipped.
    volScalarField s("materialComposition", shell_);
    if (materialLabelBasis_ != "contentComposition")
    {
        s = min
        (
            max(shell_, dimensionedScalar("zero", dimless, 0)),
            dimensionedScalar("one", dimless, 1)
        );
    }

    GField_ = Gcore_*exp(s*std::log(G_.value()/Gcore_.value()));
    GcField_ = Gccore_*exp(s*std::log(Gc_.value()/Gccore_.value()));
    k0pField_ = k0pcore_*exp(s*std::log(k0p_.value()/k0pcore_.value()));

    // lambda FOLLOWS G through the same Poisson ratio, rather than being
    // blended independently. Blending the two Lame parameters separately
    // would give a nu that is neither material's anywhere in the transition.
    lambdaField_ =
        2.0*GField_*poissonRatio_/(1.0 - 2.0*poissonRatio_);

    GField_.correctBoundaryConditions();
    lambdaField_.correctBoundaryConditions();
    GcField_.correctBoundaryConditions();
    k0pField_.correctBoundaryConditions();
}


Foam::tmp<Foam::fvScalarMatrix>
Foam::solvers::phaseClotAxisymmetricFluid::phaseSharedConvection
(
    const surfaceScalarField& carrier,
    const volScalarField& f
) const
{
    // THE GUARD, and it is not defensive noise. Every caller must have set
    // phaseWeights_ first. A future edit that advects the phase field before
    // the weights exist would otherwise dereference an invalid tmp, and the
    // failure this replaces was silent -- so this one is not allowed to be.
    if (!phaseWeights_.valid())
    {
        FatalErrorInFunction
            << "phaseSharedConvection called before phaseWeights_ was set."
            << nl << "The phase field and the platelet content must be"
            << " advected by the SAME weights; building them here from the"
            << " field's own limiter is exactly the defect this function"
            << " exists to prevent."
            << exit(FatalError);
    }

    // `bounded`, spelled out. fvSchemes' `bounded Gauss vanLeer` expands to
    // gaussConvectionScheme::fvmDiv MINUS Sp(surfaceIntegrate(flux)); the
    // carrier is deliberately not solenoidal because it may carry
    // bondHoldFactor, so dropping that term would reinstate the double count
    // the content derivation was written to remove.
    return
        fv::gaussConvectionScheme<scalar>
        (
            mesh_,
            carrier,
            tmp<surfaceInterpolationScheme<scalar>>
            (
                new suppliedWeightsScheme(mesh_, phaseWeights_())
            )
        ).fvmDiv(carrier, f)
      - fvm::Sp(fvc::surfaceIntegrate(carrier), f);
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::networkPressureArgument() const
{
    // WHY AN EXCESS AND NOT A RATIO.
    //
    // The exact volumetric stretch of the network is thetaNetRef/theta. It
    // is singular where there is no network: the lumen skeleton flux is not
    // solenoidal (Cogan & Guy 2010 after Eq. 7 -- a per-phase velocity never
    // is), so theta/thetaNetRef drifts by orders of magnitude in cells whose
    // content is a numerical tail, and the ratio would switch the pole on
    // there. That is the failure class of run 20260905-191135, a network
    // pressure acting where there is no network, which killed that run from
    // the lumen while every clot metric read healthy.
    //
    // The excess theta - thetaNetRef_ has none of that. It is the content
    // gained by compaction alone, bounded by the content present, and it is
    // exactly the legacy argument wherever the reference is nominal:
    //   bulk, undeformed:      thetaNetRef = phi_s0 -> argument = theta
    //                          (the shipped law, unchanged)
    //   interface, undeformed: theta = thetaNetRef  -> argument = phi_s0
    //                          -> p_n = 0 (the step-1 defect, removed)
    //   compacted by d:        argument = phi_s0 + d, whatever the cell's
    //                          deposited content was
    // Linearised, p_n ~ K_n*p_n'(phi_s0)*thetaNetRef*(1/J - 1): the bulk
    // stiffness of a partly filled cell scales with the network it holds,
    // the same diffuse-domain weighting the deviatoric stress already has.
    {
        return volScalarField::New
        (
            "networkPressureArgument",
            dimensionedScalar("phiS0", dimless, skeletonVolumeFraction_)
          + (thetaC_ + thetaH_)
          - thetaNetRef_
        );
    }
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::networkPressure() const
{
    // p_n = K_n * (phi_s - phi_s0) / (phi_s_max - phi_s)
    //
    // THE THREE PROPERTIES THAT MATTER, and each is why a clip on c is not an
    // acceptable substitute:
    //
    //  1. ZERO at phi_s = phi_s0. The nominal density is an EQUILIBRIUM, not a
    //     starting value that drifts. Run 20260905-113709 had no equilibrium
    //     at all, which is why chi walked from 1.00 to 1.87 without anything
    //     opposing it.
    //  2. DIVERGES as phi_s -> phi_s_max. Close packing becomes unreachable by
    //     a force balance rather than by clamping a field afterwards.
    //  3. NEGATIVE below phi_s0, so it resists DILATION too. Batch 15 dilated
    //     and Batch 16 compacted; a one-sided law would have fixed one run and
    //     left the other. Both are the same missing term.
    //
    // K_n = networkPressureCoeff * gD * G. Same gD as the deviatoric stress,
    // so broken material loses its bulk stiffness exactly as it loses its
    // shear stiffness -- a fully damaged network must not resist compaction
    // like an intact one. Reusing GField_ means the core/shell contrast is
    // inherited rather than re-declared.
    //
    // phi_s IS CLAMPED STRICTLY BELOW phi_s_max before the divide. Not
    // cosmetic: the denominator is the whole point of the term, and a cell
    // that reaches phi_s_max in one step would produce an infinity in a
    // quantity the pressure equation reads. The clamp is at 0.99*phi_s_max,
    // where p_n is already 99x K_n -- large enough that nothing physical gets
    // there, so the clamp is a numerical backstop and not the mechanism.
    // S10: the argument, not always forceFraction() -- see
    // networkPressureArgument(). Under "nominal" this is the same field.
    const volScalarField phiS(networkPressureArgument());
    const volScalarField gD(sqr(1.0 - Dg_) + etaRes_);

    const dimensionedScalar phiS0("phiS0", dimless, skeletonVolumeFraction_);
    // S25 A: the pole (phiSMax_ under the shipped law, 1 under LT).
    const dimensionedScalar phiSMax("phiSMax", dimless, networkPole());

    // S11 FIX 5: A LAW DERIVED FROM A CONVEX VOLUMETRIC ENERGY.
    //
    // The dense network's volumetric Cauchy stress s(J) = dU/dJ, with
    // J = phi_s0/a the volume ratio the excess argument a = phi_s0 + theta -
    // thetaNetRef measures, and J_c = phi_s0/phiSMax the close-packing ratio:
    //
    //     s(J) = kappa*(1 - J_c)*J*(J - 1)/(J - J_c)
    //
    // s(1) = 0 (stress-free as deposited), s'(1) = kappa (the bulk modulus is
    // EXACTLY K_n = coeff*gD*G, the design's "bulk of the order of shear"
    // anchor -- poleCubic realised K_n/4, Poisson -0.36), s' > 0 everywhere
    // on (J_c, inf) because J^2 - 2 J_c J + J_c has no real root for
    // 0 < J_c < 1 (U convex, so no loss of ellipticity and no cavitation),
    // s -> -inf at close packing, and tension grows without bound as the
    // network dilates -- tearing is the damage model's job, not the
    // volumetric law's. In a, with p = -s:
    //
    //     p(a) = K_n*(phiSMax - phi_s0)*(a - phi_s0)/(a*(phiSMax - a))
    //
    // i.e. Cogan & Guy's pole form times (phiSMax - phi_s0)/a. This is the
    // INTRINSIC stress of the dense network; networkPartialPressure()
    // weights it by the deposited occupancy.
    //
    // NO CLAMP. The law is defined on 0 < a < phiSMax and the barrier at
    // each end is infinite, so a cell outside that interval has already
    // crossed an infinite energy -- a defect upstream, reported with its
    // location, not a value to bound.
    {
        checkNetworkArgument(phiS, "convexExcess network pressure");

        return volScalarField::New
        (
            "networkPressure",
            networkPressureCoeff_*gD*GField_
           *(phiSMax - phiS0)*(phiS - phiS0)/(phiS*(phiSMax - phiS))
        );
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::checkNetworkArgument
(
    const volScalarField& a,
    const char* who
) const
{
    // CELLS FIRST. Run 20260923-085741 (em53) reported "patch bottom face
    // 77" -- a zeroGradient copy of the cell beside it -- because the
    // boundary loop fired before the cell was reported, and the cell line
    // is the one that carries theta and thetaNetRef. The cells are checked
    // and reported first; a boundary face is reported only when every cell
    // is inside, with the owner cell's state beside it.
    // S25 A: ONE pole for the pressure's and the permeability's calls
    // (phiSMax under the shipped law, 1 under LT).
    const scalar phiSMax = networkPole();
    const scalarField& aIn = a.primitiveField();
    label worst = -1;
    forAll(aIn, celli)
    {
        if (!(aIn[celli] > 0 && aIn[celli] < phiSMax))
        {
            worst = celli;
            break;
        }
    }
    if (returnReduce(worst >= 0, orOp<bool>()))
    {
        if (worst >= 0)
        {
            FatalErrorInFunction
                << who << ": network argument a = " << aIn[worst]
                << " at cell " << worst << " (" << mesh.C()[worst]
                << ") is outside (0, phiSMax = " << phiSMax
                << "): theta = " << thetaC_[worst] + thetaH_[worst]
                << ", thetaNetRef = " << thetaNetRef_[worst]
                << ". Both ends are infinite barriers, so this is a transport "
                << "or growth defect, not a state the law may be evaluated at."
                << exit(FatalError);
        }
        FatalErrorInFunction
            << who << ": network argument out of range on another processor."
            << exit(FatalError);
    }

    // The boundary values go through the same expressions, and a zero there
    // is a trapped floating-point division, not a reported defect.
    forAll(a.boundaryField(), patchi)
    {
        const scalarField& aB = a.boundaryField()[patchi];
        const labelUList& fc = mesh.boundary()[patchi].faceCells();
        forAll(aB, facei)
        {
            if (!(aB[facei] > 0 && aB[facei] < phiSMax))
            {
                const label c = fc[facei];
                FatalErrorInFunction
                    << who << ": network argument a = " << aB[facei]
                    << " on patch " << mesh.boundary()[patchi].name()
                    << " face " << facei << " is outside (0, phiSMax = "
                    << phiSMax << ") while every cell is inside; owner cell "
                    << c << " (" << mesh.C()[c] << ") has a = " << aIn[c]
                    << ", theta = " << thetaC_[c] + thetaH_[c]
                    << ", thetaNetRef = " << thetaNetRef_[c]
                    << ". The patch condition, not the transport, produced it."
                    << exit(FatalError);
            }
        }
    }
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::intrinsicPermeability() const
{
    // k_KC(a) = k0 * [(1-a)^3/a^2] / [(1-phi0)^3/phi0^2]: the case's own
    // kozenyCarmanSolid law, with the same k0 map (shell/core), evaluated at
    // the network's INTRINSIC density a instead of the cell average. At the
    // reference a = phi0 it is exactly k0. NO FLOOR AND NO CAP: a is inside
    // (0, phiSMax) or checkNetworkArgument() stops the run with the cell.
    // Inside that interval k/k0 runs from 0.12 (a = 0.6) upward, finite.
    const volScalarField a(networkPressureArgument());
    checkNetworkArgument(a, "intrinsic permeability");

    const scalar phi0 = skeletonVolumeFraction_;
    const scalar ref = pow3(1.0 - phi0)/sqr(phi0);

    return volScalarField::New
    (
        "intrinsicPermeability",
        k0pField_*pow3(1.0 - a)/sqr(a)/dimensionedScalar("ref", dimless, ref)
    );
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::intrinsicDragRate() const
{
    // K = ind * (1-a)^2 * nu / k_KC(a), 1/s.
    //
    // WHICH WEIGHT, NAMED (AGENTS.md: a diffuse-domain weight is not a volume
    // fraction). ind = phaseIndicator(), the diffuse-domain indicator, the
    // SAME field and time level skeletonFaceStress() weights the deviatoric
    // stress with. Drag is a bulk term: Li, Lowengrub, Raetz & Voigt (2009)
    // Eq. 2.27 multiply every bulk term by the indicator; multiphaseEuler's
    // dispersedDragModel::K is alpha_d*Ki with Ki per unit dispersed volume.
    // The SOLID VOLUME FRACTION enters only through the intrinsic law, at a,
    // which is where a packed-bed law belongs (a ~ 0.43-0.49 in em53/em54;
    // Wufsus et al. 2013 use Kozeny-Carman above 0.2). The legacy basis
    // evaluated it at chi*phi0 -> 0 in the fringe, two decades outside that,
    // and a calibrated floor k1*chi then set the fringe's drag in 100% of
    // its cells, 10x (shell) to 1000x (core) below the network's own.
    //
    // (1-a)^2: the interstitial fluid fraction INSIDE the network, the Darcy
    // superficial-to-interstitial factor the legacy poroFactor applied at
    // the cell average.
    const volScalarField a(networkPressureArgument());
    return volScalarField::New
    (
        "intrinsicDragRate",
        phaseIndicator()*sqr(1.0 - a)*nu_/intrinsicPermeability()
    );
}


void Foam::solvers::phaseClotAxisymmetricFluid::reportFringe() const
{
    // PHASE_CLOT_FRINGE, read by assess_case.py (s12 block, G10/G11).
    //
    // slipRms(band) = sqrt( sum ind V |Vs-U|^2 / sum ind V ), occupancy
    // weighted so a cell counts by the network it holds. Priced on run
    // 20260923-085741's fields before this line existed: em53 12.5-25.9 and
    // em54 13.5-38.8 under legacy; 1.4-4.2 priced under intrinsic; em52
    // (pre-S11, no fringe memory) 1.4-4.2 to 1 ms.
    //
    // fringeDragOverIntrinsic = sum V Kd / sum V K_int over the fringe: the
    // drag the block used against the network's own at the same a. 1 by
    // construction under intrinsic, ~0.099 under legacy (the k1 floor). It
    // is the configuration asserting itself, not a physics result.
    const volScalarField ind(phaseIndicator());
    const scalarField& indI = ind.primitiveField();
    const scalarField& vol = mesh.V();
    const vectorField& vs = Vs_.primitiveField();
    const vectorField& u = U_.primitiveField();
    const scalarField& kd = clotDrag_.primitiveField();

    // K_int needs a inside (0, phiSMax); outside the deposited/convexExcess
    // pair it is not defined, and the ratio is reported as -1.
    tmp<volScalarField> tKint;
    {
        tKint = intrinsicDragRate();
    }

    scalar fW = 0, fS = 0, bW = 0, bS = 0, fMax = 0;
    scalar fKd = 0, fKint = 0;
    label fN = 0, bN = 0;
    forAll(indI, celli)
    {
        const scalar w = indI[celli]*vol[celli];
        const scalar s2 = magSqr(vs[celli] - u[celli]);
        if (indI[celli] >= 1e-3 && indI[celli] < 0.2)
        {
            fW += w;
            fS += w*s2;
            fMax = max(fMax, s2);
            ++fN;
            {
                fKd += vol[celli]*kd[celli];
                fKint += vol[celli]*tKint()[celli];
            }
        }
        else if (indI[celli] >= 0.9)
        {
            bW += w;
            bS += w*s2;
            ++bN;
        }
    }
    reduce(fW, sumOp<scalar>());
    reduce(fS, sumOp<scalar>());
    reduce(bW, sumOp<scalar>());
    reduce(bS, sumOp<scalar>());
    reduce(fMax, maxOp<scalar>());
    reduce(fKd, sumOp<scalar>());
    reduce(fKint, sumOp<scalar>());
    reduce(fN, sumOp<label>());
    reduce(bN, sumOp<label>());

    const scalar fRms = fW > 0 ? sqrt(fS/fW) : 0;
    const scalar bRms = bW > 0 ? sqrt(bS/bW) : 0;

    Info<< "PHASE_CLOT_FRINGE time=" << runTime.userTimeValue()
        << " basis=" << "intrinsic"
        << " fringeCells=" << fN
        << " bodyCells=" << bN
        << " fringeSlipRms=" << fRms
        << " bodySlipRms=" << bRms
        << " fringeOverBodySlipRms=" << (bRms > 0 ? fRms/bRms : -1.0)
        << " fringeSlipMax=" << sqrt(fMax)
        << " fringeDragOverIntrinsic="
        << (fKint > 0 ? fKd/fKint : -1.0)
        << endl;
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::networkPresence() const
{
    // min(1, thetaNetRef/(residualChi*phi0)): 1 wherever network was
    // deposited, a linear ramp to 0 below residualChi. The complement of the
    // networkAbsent field solveElasticStrain() relaxes be on -- the same
    // quotient, the same residual, so "network present" is one statement.
    return volScalarField::New
    (
        "networkPresence",
        min
        (
            thetaNetRef_
           /dimensionedScalar
            (
                "residualContent", dimless,
                residualChi_*skeletonVolumeFraction_
            ),
            dimensionedScalar("one", dimless, 1)
        )
    );
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::intactFraction() const
{
    // 1 - ruptured_: the share of the cell's network that still carries
    // shear. Ruptured network is debris -- it keeps its content, its packing
    // resistance (the network pressure) and its drag, and loses only the
    // deviatoric load path and the elastic memory. Identically 1 when the
    // rupture model is off, so every earlier child is unchanged.
    return volScalarField::New("intactFraction", 1.0 - ruptured_);
}


void Foam::solvers::phaseClotAxisymmetricFluid::syncMaterialLabelFromContent()
{
    // S15. THE CORE/SHELL LABEL IS THE SHELL SHARE OF THE SOLID.
    //
    // thetaC_ and thetaH_ are the core and shell solid contents, both
    // non-negative and carried by the bounded FCT transport; growth deposits
    // shell. So the composition s = thetaH/(thetaC + thetaH) is the material
    // identity, already transported, conserved and bounded: with thetaC >= 0
    // the rounded denominator is never below thetaH, so s <= 1 in floating
    // point too. Where there is no solid (theta = 0) the label is the shell
    // convention, 1 -- there is nothing there for a material property to
    // describe, and every bulk term carries the indicator ~ 0.
    //
    // The separately transported, clipped shell_ this replaces had drifted
    // from s by up to 0.18-0.25 in ~220 core/shell-boundary cells by batch 54
    // (dispersive over/undershoot at a step the content-limited weights see
    // as smooth), moving k0p -- x100 core to shell -- by up to 3x there.
    if (materialLabelBasis_ != "contentComposition")
    {
        return;
    }
    const scalarField& tC = thetaC_.primitiveField();
    const scalarField& tH = thetaH_.primitiveField();
    scalarField& sI = shell_.primitiveFieldRef();
    forAll(sI, celli)
    {
        const scalar theta = tC[celli] + tH[celli];
        sI[celli] = theta > 0 ? tH[celli]/theta : 1.0;
    }
    shell_.correctBoundaryConditions();

    // Bounded by construction; a value outside [0, 1] can only come from a
    // negative content, which is a transport defect to report, not a label
    // to clip.
    const scalar sMin = gMin(sI);
    const scalar sMax = gMax(sI);
    if (!(sMin >= 0 && sMax <= 1))
    {
        FatalErrorInFunction
            << "material composition thetaH/theta left [0, 1]: min " << sMin
            << ", max " << sMax << ". thetaC and thetaH must be "
            << "non-negative (the FCT reports negAfter)."
            << exit(FatalError);
    }

    updateMaterialFields();
}


void Foam::solvers::phaseClotAxisymmetricFluid::ruptureOverstretchedNetwork()
{

    // S15 progressiveDamage: Kliuchnikov et al. 2025 (source 32) Eq. 6 --
    // "we assume that the stress decreases linearly as damage increases,
    // S = S0 (1 - d)" with d(l1) = 1 - exp(-H(l1 - lf)(l1 - lf)/(2 dlf)),
    // l1 the largest principal stretch. d is the running maximum (damage is
    // irreversible), stored in ruptured_, so intactFraction() = 1 - d is
    // exactly their (1 - d). No reset of be: the damaged network keeps the
    // strain of what is still intact, and the intact-content relaxation
    // forgets it only once ref*(1 - d) falls below the residual.
    //
    // Why this and not the S14 step: the step released a cell's whole stress
    // in one step, and the drag-limited recoil of its neighbour, ~sigma/(h K),
    // grows as the mesh is refined (run 20260924-014332: 0.12-0.21 m/s). Here
    // the load leaves the network as its fibres break, and the damaged
    // material still HARDENS up to l1 ~ 4.1 (d ~ 0.63; the (1 - d)-weighted
    // neo-Hookean stress peaks there), so the stretch range where S14 fired
    // stays well-posed. Its tangent (1 - d)(8/9)b1 is at most 5.51 for every
    // stretch -- the tail that killed em58 is bounded without a cap.
    const scalar lf = networkDamageOnsetStretch_;
    const scalar dlf = networkDamageStretchScale_;
    scalar damageStepMax = 0;

    // Tutwiler et al. 2020 (source 28): "Rupture occurs when a critical
    // stretch is attained over a representative volume element" -- 2.8
    // locally ahead of a crack, single fibres 2.5-3.3. be is the elastic
    // LEFT Cauchy-Green tensor, so its largest eigenvalue is the SQUARE of
    // the largest principal stretch; the criterion is on b1 >= lambda_c^2.
    //
    // Where network exists (the same presence test as the relaxation basis)
    // and has reached it: the network there ruptures. ruptured_ = 1 and be_
    // = I -- broken network carries no elastic state -- and from then on
    // intactFraction() removes it from the deviatoric load path and the
    // relaxation keeps its be at I. The content is NOT removed: debris stays,
    // with its packing resistance and its drag.
    const scalar residualContent = residualChi_*skeletonVolumeFraction_;
    const scalarField& refI = thetaNetRef_.primitiveField();
    const volScalarField indR(phaseIndicator());
    const scalarField& indI = indR.primitiveField();
    scalarField& rI = ruptured_.primitiveFieldRef();
    symmTensorField& bI = be_.primitiveFieldRef();
    const scalarField& vol = mesh.V();

    label nNew = 0, nNewBulk = 0;
    scalar b1IntactMax = 1;
    // The largest b1 among cells rupturing NOW: how far past lambda_c the
    // stretch got before the criterion acted. maxIntactStretch is below
    // lambda_c by construction and cannot fail; this can. One step at the
    // wall shear (~1e4 1/s) adds 0.5-1.5% of stretch, so a value well past
    // lambda_c is a stretch rate the step did not resolve -- em58's tail went
    // 2.6 -> 5 -> 164 in consecutive steps.
    scalar b1RupturedMax = 1;
    scalar rupContent = 0, allContent = 0;
    {
        // S16 C: THE LAW ACTS ON THE PARTITION, and d (ruptured_) is derived
        // from it afterwards. Per cell, with s = P1 + (P2 + P3) the content,
        // d* = (P2 + P3)/s and d = P3/s (both 0 where s = 0):
        //
        //   terminalRelease (S16 A):
        //     d*_law = 1                               l1 >= lc
        //            = 1 - exp(-(l1 - lf)/(2 dlf))     lf < l1 < lc, progressive on
        //            = 0                               otherwise
        //     d*    <- max(d*, d*_law)                 irreversible (the event)
        //     d_{n+1} = (d_n + r d*)/(1 + r), r = dt/tau (the release),
        //   written as the pending share p = d* - d decaying to p/(1 + r),
        //   d = d* - p/(1 + r): exactly (d_n + r d*)/(1 + r), and with
        //   0 <= p/(1 + r) <= p <= d* <= 1 every component written below,
        //   s(1 - d*), s p/(1 + r), s d, is >= 0 in floating point.
        //   The release acts in every cell with a pending share; the target
        //   rises only where network exists (the S14 presence test), because
        //   l1 means nothing where there is none.
        //
        //   progressiveDamage / criticalStretch under the partition: d* = d,
        //   set as S15 / S14 set d (P2 stays 0).
        //
        // ONCE PER STEP (solveElasticStrain(), first PIMPLE iteration only),
        // so the release is applied once per dt, never once per corrector.
        // S17 A: fibreRelease is this event and this release; what it adds
        // is the pending share's fibre (below, "S17 A").
        const scalar lc = networkCriticalStretch_;
        const scalar r =
            runTime.deltaTValue()/networkReleaseTime_;
        const scalarField& oI = thetaIntact_.oldTime().primitiveField();
        const scalarField& oP = thetaPending_.oldTime().primitiveField();
        const scalarField& oB = thetaBroken_.oldTime().primitiveField();
        scalarField& pI = thetaIntact_.primitiveFieldRef();
        scalarField& pP = thetaPending_.primitiveFieldRef();
        scalarField& pB = thetaBroken_.primitiveFieldRef();
        // S17 A: the pending fibre's state, and the network pressure the
        // frozen tension is taken from (q_P += dthP min(p_n, 0), 07/06).
        symmTensorField& cB = thetaPendingB_.primitiveFieldRef();
        scalarField& qP = thetaPendingP_.primitiveFieldRef();
        vectorField& nP = pendingAxis_.primitiveFieldRef();
        scalarField& lP = pendingStretch_.primitiveFieldRef();
        // S17b: the taut share (V1) and the fibre read minimum (A2)
        scalarField& tTL = thetaTaut_.primitiveFieldRef();
        const scalar readMinL = fibreReadMin();
        label nRaisedPresentL = 0, nRaisedAbsentL = 0;
        tmp<volScalarField> tPn;
        {
            tPn = networkPartialPressureUndamaged();
        }
        scalar transferB1 = 0;
        label nTransfer = 0, nReset = 0, nProj = 0, nRaised = 0, nTie = 0;
        scalar projExcess = 0;
        scalar lawRelease = 0;
        scalar deltaD = 0;
        // The largest share still pending after this step, d* - d: G12r's
        // "missing release" leg reads it (a pending share on one record and
        // a zero release on the next is a release that did not happen).
        scalar pendingAfter = 0;
        // S23 section 2.1: the volumetric branch's population is the shipped
        // isSkel (the block's isSkelB rule), and R_c is tested as a product.
        // A1 writes thetaNetRef only where a volumetric firing resets it, so
        // the reference field is not touched while nothing fires.
        const scalar Rc = volumetricTearRatio_;
        const scalarField& phiSkelI = phiSkel_.primitiveField();
        scalarField* refW = nullptr;
        label nVol = 0, nStretch = 0, nA1Deferred = 0;
        std::vector<scalar> Rskel;
        Rskel.reserve(bI.size());
        // D74 (design 2026-10-09-D74 rev. 3.1 section 2 table): ref_I's
        // event transfers in this loop -- the rupture (inside dsNew > ds),
        // the emptying and the volumetric tear -- and the content they move
        // (booked for PHASE_CLOT_REFI's label balance).
        scalarField& refIW = thetaNetRefIntact_.primitiveFieldRef();
        scalar refITransfer = 0;
        forAll(bI, celli)
        {
            const scalar theta = thetaC_[celli] + thetaH_[celli];
            const bool isSkelC =
                residualChi_ > 0
              ? indI[celli] - 0.5*residualChi_ >= 0
              : phiSkelI[celli] - 0.5*residualPhiS_ >= 0;
            // R over isSkel, BEFORE any reset (section 2.5); theta > 0 is
            // the quotient's own guard (isSkel implies it while residualChi
            // > 0, but not on the phi_s rule).
            if (isSkelC && theta > 0)
            {
                Rskel.push_back(refI[celli]/theta);
            }
            bool volTear = false;
            allContent += theta*vol[celli];

            const scalar sOld = oI[celli] + (oP[celli] + oB[celli]);
            const scalar dOld = sOld > 0 ? oB[celli]/sOld : 0.0;

            // The law's target, where network exists (l1 means nothing
            // elsewhere) -- evaluated whatever the partition holds, so the
            // stretch statistics are those of the S14/S15 loop.
            scalar dsLaw = 0;
            scalar b1 = 1;
            const bool present = refI[celli] > residualContent;
            if (present)
            {
                b1 = cmptMax(eigenValues(bI[celli]));
                const scalar l1 = sqrt(b1);
                {
                    if (l1 >= lc)
                    {
                        dsLaw = 1.0;
                    }
                    else if (l1 > lf)
                    {
                        dsLaw = 1.0 - exp(-(l1 - lf)/(2.0*dlf));
                    }
                }
                // S23 section 2.1, THE VOLUMETRIC BRANCH: d*_law = 1 also
                // where (isSkel AND ref >= R_c theta), the product form (no
                // division by theta); Eq. 6 of l1 otherwise, as above.
                volTear = isSkelC && refI[celli] >= Rc*theta;
                if (volTear)
                {
                    dsLaw = 1.0;
                }
            }

            const scalar sC = pI[celli] + (pP[celli] + pB[celli]);
            const scalar ds = sC > 0 ? (pP[celli] + pB[celli])/sC : 0.0;
            const scalar d = sC > 0 ? pB[celli]/sC : 0.0;
            const scalar dsNew = max(ds, dsLaw);
            // S23 section 2.5: ONE LINE PER EVENT, logged BEFORE the A1 reset
            // (the C3 test and OI-55's metric read R here). An event is a
            // volumetric firing (each one resets ref) or a NEW stretch tear.
            const bool stretchTear =
                present && sqrt(b1) >= lc && dsNew > ds;
            if (sC > 0 && (volTear || stretchTear))
            {
                if (volTear) { ++nVol; }
                if (stretchTear) { ++nStretch; }
                const scalar phi0 = skeletonVolumeFraction_;
                Info<< "PHASE_CLOT_S23_EVENT time="
                    << runTime.userTimeValue()
                    << " step=" << runTime.timeIndex()
                    << " cell=" << celli
                    << " x=" << mesh_.C()[celli].x()
                    << " y=" << mesh_.C()[celli].y()
                    << " branch="
                    << (
                           volTear && stretchTear ? "both"
                         : volTear ? "volumetric" : "stretch"
                       )
                    << " R=" << (theta > 0 ? refI[celli]/theta : -1.0)
                    << " theta=" << theta
                    << " ref=" << refI[celli]
                    << " ind=" << theta/phi0
                    << " dStarBefore=" << ds
                    << " d=" << d
                    << " lambda1=" << sqrt(b1)
                    << " lambdaF="
                    << fibreReleaseKernel::eventStretch(b1, lc)
                    << endl;
            }
            // maxIntactStretch as S15 (every cell with network) and S14
            // (cells not ruptured) define it.
            if (present)
            {
                b1IntactMax = max(b1IntactMax, b1);
            }
            // D74: this cell's intact content before its event.
            const scalar pIOldD74 = pI[celli];
            scalar dAfter = d;
            if (sC > 0)
            {
                if (dsNew > ds)
                {
                    b1RupturedMax = max(b1RupturedMax, b1);
                    ++nNew;
                    if (indI[celli] >= 0.5) { ++nNewBulk; }
                    // S18 F (16 #11): each DAMAGE EVENT -- a cell's first
                    // damage, d* from 0 -- with the cell's content ind =
                    // theta/phi0, its occupancy thetaNetRef/phi0, lambda1 =
                    // sqrt(b1) (isochoric, the law's), and source 32's TOTAL
                    // stretch (ref/theta)^(1/3) lambda1 (04's convention; -1
                    // where theta <= 0). For the REV first-damage reading.
                    if (!(ds > 0))
                    {
                        const scalar phi0 = skeletonVolumeFraction_;
                        Info<< "PHASE_CLOT_S18_DAMAGE time="
                            << runTime.userTimeValue()
                            << " cell=" << celli
                            << " x=" << mesh_.C()[celli].x()
                            << " y=" << mesh_.C()[celli].y()
                            << " ind=" << theta/phi0
                            << " occupancy=" << refI[celli]/phi0
                            << " lambda1=" << sqrt(b1)
                            << " totalStretch="
                            << networkPressureKernel::totalStretch
                               (
                                   refI[celli], theta, b1
                               )
                            << " dStar=" << dsNew
                            << endl;
                    }
                    // D74 rev. 3.1, RUPTURE (intact -> pending; physics 32
                    // section 3, 83 Eq 2.4): J_I = ref_I/theta_I is kept, so
                    // ref_I scales with the intact share, S being unchanged
                    // by the event: ref_I *= (1 - d*_new)/(1 - d*). ONLY on a
                    // damage increase: rev. 2's pI_new/pI_old fired on every
                    // round-off rewrite of pI below and ratcheted ref_I
                    // (BATCH 74). dsNew > ds gives ds < dsNew <= 1, so
                    // 1 - ds > 0 guards the quotient, and IEEE subtraction is
                    // monotone, so the factor lies in [0, 1]; dsNew = 1 gives 0.
                    const scalar refIOld = refIW[celli];
                    refIW[celli] = refIOld*((1.0 - dsNew)/(1.0 - ds));
                    refITransfer += (refIW[celli] - refIOld)*vol[celli];
                }

                {
                    const scalar pend = dsNew - d;
                    if (pend > 0)
                    {
                        const scalar pendNew = pend/(1.0 + r);
                        const scalar dNew = dsNew - pendNew;
                        lawRelease = max(lawRelease, dNew - d);
                        // S17 A: THE PENDING SHARE IS A FIBRE (design 2A).
                        // The content moving intact -> pending this step,
                        // dthP = s (d* - d*_old), freezes as a fibre along
                        // be's major axis at lf = min(l1(be), lc): fibres
                        // break at lc, so this is the event value, not a
                        // clamp (09 rule 4), and G-tr reads b1/lc^2 BEFORE
                        // the min. The blend with the cell's pending fibre
                        // is projected back to one fibre (P-LS, 12), the
                        // tension p_P is frozen at the event's min(p_n, 0),
                        // and the release scales C_B and q_P with thP, so
                        // B_P and p_P are unchanged by it.
                        {
                            const scalar thOld = pP[celli];
                            const scalar thPost = sC*pendNew;
                            const scalar dTh =
                                dsNew > ds ? sC*(dsNew - ds) : 0.0;
                            const scalar thMix = thOld + dTh;
                            // S17b A1 (V1): the fibre's carrier is the TAUT
                            // share; a transfer joins it (07: "transfers
                            // join theta_T only"), slack content keeps none.
                            // Its post-release value is thPost times the
                            // carrier's fraction of P2, so theta_T <= P2
                            // exactly (a quotient <= 1 times thPost).
                            // Without V1 the carrier is P2 itself (S17).
                            const scalar cOld = tTL[celli];
                            const scalar cMix = cOld + dTh;
                            scalar cPost = thPost;
                            if (dTh > 0 && cMix > 0)
                            {
                                const symmTensor& be = bI[celli];
                                double Bb[3][3] =
                                {
                                    {be.xx(), be.xy(), be.xz()},
                                    {be.xy(), be.yy(), be.yz()},
                                    {be.xz(), be.yz(), be.zz()}
                                };
                                double ev[3], V[3][3];
                                fibreReleaseKernel::eigSym(Bb, ev, V);
                                transferB1 = max(transferB1, ev[0]/sqr(lc));
                                ++nTransfer;
                                const double nE[3] = {V[0][0], V[1][0], V[2][0]};
                                // be is SPD (a loss is fatal under C and
                                // logged BE_SPD_LOST otherwise); a negative
                                // b1 would give a NaN that
                                // syncPendingFibres() refuses.
                                const scalar lE =
                                    fibreReleaseKernel::eventStretch(ev[0], lc);
                                double Bf[6];
                                fibreReleaseKernel::fibre(nE, lE, Bf);
                                const symmTensor Bmix
                                (
                                    (cB[celli]
                                   + dTh*symmTensor
                                     (Bf[0], Bf[1], Bf[2], Bf[3], Bf[4], Bf[5])
                                    )/cMix
                                );
                                const double BmixA[6] =
                                {
                                    Bmix.xx(), Bmix.xy(), Bmix.xz(),
                                    Bmix.yy(), Bmix.yz(), Bmix.zz()
                                };
                                // the previous axis, or the event's where
                                // the cell held no pending content
                                const double nPrev[3] =
                                {
                                    cOld > 0 ? nP[celli].x() : nE[0],
                                    cOld > 0 ? nP[celli].y() : nE[1],
                                    cOld > 0 ? nP[celli].z() : nE[2]
                                };
                                // S17b A2: the blend is read only where the
                                // carrier is >= DBL_MIN; below it the cell
                                // keeps its last valid fibre (or takes the
                                // event's, if it held none).
                                fibreReleaseKernel::Projection Pj;
                                if (cMix >= readMinL)
                                {
                                    Pj = fibreReleaseKernel::projectLS(BmixA, nPrev);
                                }
                                else
                                {
                                    Pj.ok = true;
                                    Pj.tie = false;
                                    Pj.psiMix = Pj.psiOut = 0;
                                    for (int k = 0; k < 3; ++k)
                                    {
                                        Pj.n[k] = cOld > 0 ? nPrev[k] : nE[k];
                                    }
                                    Pj.lf = cOld > 0 ? lP[celli] : lE;
                                    Pj.Tls = fibreReleaseKernel::fibreTension(Pj.lf);
                                }
                                if (!Pj.ok || !std::isfinite(Pj.lf))
                                {
                                    FatalErrorInFunction
                                        << "P-LS at the transfer did not "
                                        << "converge in cell " << celli
                                        << "." << exit(FatalError);
                                }
                                ++nProj;
                                if (Pj.tie) { ++nTie; }
                                if (!fibreReleaseKernel::dissipative(Pj))
                                {
                                    ++nRaised;
                                    if (refI[celli] > residualContent) { ++nRaisedPresentL; }
                                    else { ++nRaisedAbsentL; }
                                    projExcess =
                                        max(projExcess, Pj.psiOut - Pj.psiMix);
                                }
                                // S18p: the transfer site's raised cell, with
                                // the blend's input and its donors (log only).
                                if
                                (
                                    projSiteLog_
                                 && !fibreReleaseKernel::dissipative(Pj)
                                 && s18pTakeCell
                                    (
                                        0, refI[celli] > residualContent
                                    )
                                )
                                {
                                    std::ostringstream os;
                                    os.precision(17);
                                    os << "PHASE_CLOT_S18P_CELL time="
                                       << runTime.value()
                                       << " step=" << runTime.timeIndex()
                                       << " site=T cell=" << celli
                                       << " present="
                                       << (refI[celli] > residualContent)
                                       << " th=" << cMix;
                                    for (int k = 0; k < 6; ++k)
                                    {
                                        os << " Bmix" << k << "=" << BmixA[k];
                                    }
                                    os << " psiMix=" << Pj.psiMix
                                       << " psiOut=" << Pj.psiOut
                                       << " lfOut=" << Pj.lf
                                       << " cOld=" << cOld
                                       << " dTh=" << dTh
                                       << " lE=" << lE
                                       << " lPold=" << lP[celli];
                                    const symmTensor& cBd = cB[celli];
                                    const double cBdA[6] =
                                    {
                                        cBd.xx(), cBd.xy(), cBd.xz(),
                                        cBd.yy(), cBd.yz(), cBd.zz()
                                    };
                                    for (int k = 0; k < 6; ++k)
                                    {
                                        os << " cBold" << k << "=" << cBdA[k]
                                           << " Bf" << k << "=" << Bf[k];
                                    }
                                    for (int k = 0; k < 3; ++k)
                                    {
                                        os << " nE" << k << "=" << nE[k]
                                           << " nPrev" << k << "=" << nPrev[k];
                                    }
                                    Info<< os.str().c_str() << endl;
                                }
                                const scalar pMix =
                                    (qP[celli]
                                   + dTh*min(tPn()[celli], scalar(0)))/cMix;
                                {
                                    cPost = thPost*(cMix/thMix);
                                }
                                nP[celli] = vector(Pj.n[0], Pj.n[1], Pj.n[2]);
                                lP[celli] = Pj.lf;
                                qP[celli] = cPost*pMix;
                            }
                            else if (cOld > 0)
                            {
                                // the release alone: p_P = q_P/carrier kept
                                {
                                    cPost = thPost*(cOld/thOld);
                                }
                                qP[celli] = cPost*(qP[celli]/cOld);
                            }
                            else
                            {
                                qP[celli] = 0.0;
                                {
                                    cPost = 0.0;
                                }
                            }
                            {
                                tTL[celli] = cPost;
                            }
                            if (cPost > 0)
                            {
                                const double nA[3] =
                                    {nP[celli].x(), nP[celli].y(), nP[celli].z()};
                                double Bo[6];
                                fibreReleaseKernel::fibre(nA, lP[celli], Bo);
                                cB[celli] = cPost*symmTensor
                                (
                                    Bo[0], Bo[1], Bo[2], Bo[3], Bo[4], Bo[5]
                                );
                            }
                            else
                            {
                                cB[celli] = Zero;
                                qP[celli] = 0.0;
                                lP[celli] = 1.0;
                            }
                        }
                        pI[celli] = sC*(1.0 - dsNew);
                        pP[celli] = sC*pendNew;
                        pB[celli] = sC*dNew;
                        dAfter = dNew;
                    }
                    // S17 A, 09 rule 1: be := I WHERE NO INTACT NETWORK
                    // REMAINS -- the value of an empty population, as d := 0
                    // at theta = 0. After the fibre above has been frozen
                    // from be (B_P first, then the reset). Only where the
                    // cell holds network content: an empty cell (S = 0)
                    // keeps the shipped relaxation, so em68 stays identical
                    // to em63 until the first damage (G-id).
                    if (pI[celli] == 0)
                    {
                        bI[celli] = symmTensor::I;
                        ++nReset;
                    }
                    // S23 A1 (report 10), THE SAME EVENT: a volumetric firing
                    // has just set thetaIntact = sC (1 - 1) = 0 and be = I
                    // (above, the S17 block kept verbatim); the reference now
                    // takes the content present, ref <- theta, with nothing
                    // read in between. The debris keeps no label-content
                    // inconsistency (R = 1), so a later deposit cannot tear on
                    // arrival. Force-neutral in its step (Pn = 0 before and
                    // after). R was logged above, before this reset.
                    if (volTear && pI[celli] == 0)
                    {
                        if (!refW)
                        {
                            refW = &thetaNetRef_.primitiveFieldRef();
                        }
                        (*refW)[celli] = theta;
                    }
                    else if (volTear)
                    {
                        // d* = 1 with a thetaIntact below half an ulp of the
                        // partition sum (no pending share to release): no A1
                        // this step; counted, not hidden.
                        ++nA1Deferred;
                    }
                }
            }
            // D74 rev. 3.1, EMPTYING: the partition rewrite above,
            // pI = sC (1 - dsNew), runs every step wherever a pending share
            // exists, with or without a damage increase; it RECOMPUTES pI
            // (it does not leave it unchanged), so a ratio pI_new/pI_old is
            // not 1 there, which was rev. 2's defect. Where P1/S is below
            // about 0.5-0.75 ulp, ds rounds to 1 and dsNew == ds: no rupture
            // transfer fired, yet the rewrite gives pI = 0, and the shipped
            // code ends the intact network there (be := I where pI == 0,
            // above). ref_I <- 0, booked. pI is written nowhere else in this
            // loop, so this fires once per emptying and never where the
            // rewrite leaves P1 > 0.
            if (pIOldD74 > 0 && pI[celli] == 0)
            {
                refITransfer -= refIW[celli]*vol[celli];
                refIW[celli] = 0.0;
            }
            // D74: a VOLUMETRIC TEAR is A1's event above (sC > 0, volTear,
            // the intact content now 0): no intact network is left,
            // ref_I <- 0.
            if (sC > 0 && volTear && pI[celli] == 0)
            {
                refITransfer -= refIW[celli]*vol[celli];
                refIW[celli] = 0.0;
            }
            deltaD = max(deltaD, mag(dAfter - dOld));
            if (sC > 0)
            {
                pendingAfter = max(pendingAfter, max(ds, dsLaw) - dAfter);
            }
            rupContent += dAfter*theta*vol[celli];
        }
        thetaIntact_.correctBoundaryConditions();
        thetaPending_.correctBoundaryConditions();
        thetaBroken_.correctBoundaryConditions();
        // S23 A1: the reset reference's patches follow its cells.
        if (returnReduce(refW != nullptr, orOp<bool>()))
        {
            thetaNetRef_.correctBoundaryConditions();
        }
        // D74: ref_I's patches follow its cells; its transfers are booked.
        thetaNetRefIntact_.correctBoundaryConditions();
        refITransferStep_ += refITransfer;
        // S23 section 2.5: the step's events, and R's max and quantiles over
        // isSkel BEFORE the reset (serial: the design's S23 is serial).
        reduce(nVol, sumOp<label>());
        reduce(nStretch, sumOp<label>());
        reduce(nA1Deferred, sumOp<label>());
        s23VolEvents_ = nVol;
        s23StretchEvents_ = nStretch;
        s23VolEventsTotal_ += nVol;
        s23StretchEventsTotal_ += nStretch;
        s23A1Deferred_ += nA1Deferred;
        s23RCells_ = label(Rskel.size());
        s23RMax_ = 0;
        s23RP50_ = 0;
        s23RP90_ = 0;
        s23RP99_ = 0;
        if (!Rskel.empty())
        {
            const std::size_t nR = Rskel.size();
            auto quantile = [&](const scalar q)
            {
                const std::size_t k =
                    std::min(nR - 1, std::size_t(q*scalar(nR - 1) + 0.5));
                std::nth_element(Rskel.begin(), Rskel.begin() + k, Rskel.end());
                return Rskel[k];
            };
            s23RP50_ = quantile(0.5);
            s23RP90_ = quantile(0.9);
            s23RP99_ = quantile(0.99);
            s23RMax_ = *std::max_element(Rskel.begin(), Rskel.end());
        }
        syncDamageFromPartition("rupture law");
        {
            thetaPendingB_.correctBoundaryConditions();
            thetaPendingP_.correctBoundaryConditions();
            pendingAxis_.correctBoundaryConditions();
            pendingStretch_.correctBoundaryConditions();
            {
                thetaTaut_.correctBoundaryConditions();
            }
            syncPendingFibres("rupture law");
            reduce(nRaisedPresentL, sumOp<label>());
            reduce(nRaisedAbsentL, sumOp<label>());
            s17bProjRaisedPresent_ += nRaisedPresentL;
            s17bProjRaisedAbsent_ += nRaisedAbsentL;
            {
                const scalarField& carL = fibreCarrier().primitiveField();
                scalar lfMax = 0;
                forAll(carL, celli)
                {
                    if (carL[celli] >= std::numeric_limits<scalar>::min())
                    {
                        lfMax = max(lfMax, lP[celli]);
                    }
                }
                reduce(lfMax, maxOp<scalar>());
                s17bMaxLfLaw_ = max(s17bMaxLfLaw_, lfMax);
            }
            reduce(transferB1, maxOp<scalar>());
            reduce(nTransfer, sumOp<label>());
            reduce(nReset, sumOp<label>());
            reduce(nProj, sumOp<label>());
            reduce(nRaised, sumOp<label>());
            reduce(nTie, sumOp<label>());
            reduce(projExcess, maxOp<scalar>());
            s17TransferB1Max_ = max(s17TransferB1Max_, transferB1);
            s17Transfers_ += nTransfer;
            s17ResetCells_ += nReset;
            s17ProjCount_ += nProj;
            s17ProjRaised_ += nRaised;
            s17ProjTies_ += nTie;
            s17ProjExcessMax_ = max(s17ProjExcessMax_, projExcess);
            if (projSiteLog_)
            {
                s18pCount_[0] += nProj;
                s18pRaised_[0] += nRaised;
                s18pRaisedPresent_[0] += nRaisedPresentL;
                s18pRaisedAbsent_[0] += nRaisedAbsentL;
                s18pExcessMax_[0] = max(s18pExcessMax_[0], projExcess);
            }
        }

        reduce(lawRelease, maxOp<scalar>());
        reduce(deltaD, maxOp<scalar>());
        reduce(pendingAfter, maxOp<scalar>());
        s16PendingMax_ = pendingAfter;
        damageStepMax = lawRelease;
        s16LawReleaseMax_ = max(s16LawReleaseMax_, lawRelease);
        s16LawRate_ = r;
        s16DeltaDMax_ = max(s16DeltaDMax_, deltaD);
    }
    ruptured_.correctBoundaryConditions();
    be_.correctBoundaryConditions();

    reduce(nNew, sumOp<label>());
    reduce(nNewBulk, sumOp<label>());
    reduce(b1IntactMax, maxOp<scalar>());
    reduce(b1RupturedMax, maxOp<scalar>());
    reduce(damageStepMax, maxOp<scalar>());
    reduce(rupContent, sumOp<scalar>());
    reduce(allContent, sumOp<scalar>());

    Info<< "PHASE_CLOT_RUPTURE time=" << runTime.userTimeValue()
        << " criticalStretch=" << networkCriticalStretch_
        << " newlyRuptured=" << nNew
        << " newlyRupturedBulk=" << nNewBulk
        << " rupturedContentFraction="
        << (allContent > 0 ? rupContent/allContent : 0.0)
        << " maxIntactStretch=" << sqrt(b1IntactMax)
        << " maxRupturedStretch=" << sqrt(b1RupturedMax)
        // intactFraction() = 1 - ruptured_ weights the stress, the BSD and
        // the dt bound, so a label outside [0, 1] would flip or inflate them
        // silently. The transport is bounded by construction (S14b: upwind
        // M-matrix predictor + Zalesak); logged so a violation is seen, not
        // assumed away.
        << " rupturedMin=" << gMin(rI)
        << " rupturedMax=" << gMax(rI)
        // S15: the largest one-step rise of d in any cell. Loading resolved
        // at the wall shear raises the stretch by 0.5-4% per step, so d by
        // at most ~0.02; a stretch jump like em58's 2.6 -> 5 would be ~0.5.
        << " damageStepMax=" << damageStepMax
        << endl;
}


void Foam::solvers::phaseClotAxisymmetricFluid::syncDamageFromPartition
(
    const char* site
)
{
    // d = D/theta, formed from the partition's OWN sum so that it lies in
    // [0, 1] in floating point: with every component >= 0, P3 <= P2 + P3 <=
    // P1 + (P2 + P3), and division by one positive denominator is monotone.
    // Where the partition is empty the cell holds no network, and d is 0 by
    // definition (red-team Change 4): no 0/0 reaches intactFraction(), the
    // noMemory relaxation, the BSD or the tangent bound.
    const scalarField& pI = thetaIntact_.primitiveField();
    const scalarField& pP = thetaPending_.primitiveField();
    const scalarField& pB = thetaBroken_.primitiveField();
    scalarField& dI = ruptured_.primitiveFieldRef();
    label nNeg = 0;
    label firstNeg = -1;
    forAll(dI, celli)
    {
        // std::isfinite first (13c; 14's non-blocking gap): +Inf passes a
        // ">= 0" test and would give d = NaN.
        if
        (
            !(
                std::isfinite(pI[celli]) && std::isfinite(pP[celli])
             && std::isfinite(pB[celli])
             && pI[celli] >= 0 && pP[celli] >= 0 && pB[celli] >= 0
            )
        )
        {
            if (firstNeg < 0) { firstNeg = celli; }
            ++nNeg;
            continue;
        }
        const scalar s = pI[celli] + (pP[celli] + pB[celli]);
        dI[celli] = s > 0 ? pB[celli]/s : 0.0;
    }
    // NO CLAMP. The partition is non-negative by construction; a negative
    // (or non-finite) component is a defect at the named site, not a value
    // to bound.
    if (firstNeg >= 0)
    {
        Info<< "PHASE_CLOT_S16_PARTITION_NEGATIVE site=" << site
            << " cell=" << firstNeg
            << " thetaIntact=" << pI[firstNeg]
            << " thetaPending=" << pP[firstNeg]
            << " thetaBroken=" << pB[firstNeg] << endl;
    }
    reduce(nNeg, sumOp<label>());
    if (nNeg > 0)
    {
        FatalErrorInFunction
            << "damage partition has " << nNeg << " cell(s) with a negative "
            << "or non-finite component after " << site << " (the first is "
            << "in PHASE_CLOT_S16_PARTITION_NEGATIVE). It is non-negative by "
            << "construction; this is a transport or bookkeeping defect, not "
            << "something to bound." << exit(FatalError);
    }
    ruptured_.correctBoundaryConditions();
}


void Foam::solvers::phaseClotAxisymmetricFluid::restoreDamagePartition
(
    const char* site
)
{
    if
    (
        !thetaIntactStepAccepted_.valid()
     || !thetaPendingStepAccepted_.valid()
     || !thetaBrokenStepAccepted_.valid()
    )
    {
        FatalErrorInFunction
            << "damage partition has no accepted timestep checkpoint at "
            << site << ": theta would roll back and the partition would not."
            << exit(FatalError);
    }
    thetaIntact_ = thetaIntactStepAccepted_();
    thetaPending_ = thetaPendingStepAccepted_();
    thetaBroken_ = thetaBrokenStepAccepted_();
    thetaIntact_.correctBoundaryConditions();
    thetaPending_.correctBoundaryConditions();
    thetaBroken_.correctBoundaryConditions();
    syncDamageFromPartition(site);
    // S17 A: the pending fibre rolls back with the partition that weights
    // it (design section 5), at both roll-back sites that call this.
    {
        if
        (
            !thetaPendingBStepAccepted_.valid()
         || !thetaPendingPStepAccepted_.valid()
         || !pendingAxisStepAccepted_.valid()
         || !pendingStretchStepAccepted_.valid()
        )
        {
            FatalErrorInFunction
                << "the pending fibre has no accepted timestep checkpoint at "
                << site << ": the partition would roll back and its fibre "
                << "would not." << exit(FatalError);
        }
        thetaPendingB_ = thetaPendingBStepAccepted_();
        thetaPendingP_ = thetaPendingPStepAccepted_();
        pendingAxis_ = pendingAxisStepAccepted_();
        pendingStretch_ = pendingStretchStepAccepted_();
        // S17b A1: the taut share with its fibre (V1).
        {
            if (!thetaTautStepAccepted_.valid())
            {
                FatalErrorInFunction
                    << "the taut share has no accepted timestep checkpoint at "
                    << site << "." << exit(FatalError);
            }
            thetaTaut_ = thetaTautStepAccepted_();
            thetaTaut_.correctBoundaryConditions();
        }
        thetaPendingB_.correctBoundaryConditions();
        thetaPendingP_.correctBoundaryConditions();
        pendingAxis_.correctBoundaryConditions();
        pendingStretch_.correctBoundaryConditions();
        syncPendingFibres(site);
    }
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::skeletonStressWeight() const
{
    // S16 B, skeletonFloorBasis occupancy: the floor guards the OCCUPANCY
    // only, w = max(ind, floor)*(1 - d). Refused without the diffuse domain,
    // so the floor always exists here. Under intactWeight each site keeps
    // its own shipped expression, untouched.
    return volScalarField::New
    (
        "elasticSolidWeight",
        max
        (
            phaseIndicator(),
            dimensionedScalar("solidFloor", dimless, skeletonSolidFloor_)
        )
       *intactFraction()
    );
}


// The signature opens on the definition line so that source scans that
// attribute a line to its function by the definition line
// (check_content_consumer_map.py) attribute this one's indicator read here.
void Foam::solvers::phaseClotAxisymmetricFluid::recordStressWeight(
    const scalarField& wUsed,
    const label site
) const
{
    // S16w, TWO-SIDED (13c; 14 item 2, 15 point a): max over cells of
    // |w_used - max(ind, floor)*(1 - d)|, the reference built HERE,
    // independently of the site, from phaseIndicator() and ruptured_.
    // Exactly 0 when the site uses the occupancy weight; 1e-4*d in a
    // floor-bound damaged cell when it over-weights (the intactWeight form);
    // 1e-4*(1 - d) in a lumen cell when it drops the floor. site is 0, 1, 2
    // for skeletonFaceStress, elasticStress, the tangent bound.
    const volScalarField indW(phaseIndicator());
    const scalarField& indI = indW.primitiveField();
    const scalarField& dI = ruptured_.primitiveField();
    const scalar floorW = skeletonSolidFloor_;
    // S23 section 2.2: the two builders (sites 0 and 1) weight the torn set
    // T with (1 - d*) (rev. 4.1: majority T, d* <= 1); the tangent bound
    // (site 2) keeps (1 - d).
    boolList tornW;
    scalarField sTornW, dsW;
    const bool tornRef =
        site < 2 && tornNetworkCells(tornW, sTornW, &dsW) > 0;
    scalar absDev = 0;
    forAll(wUsed, celli)
    {
        const scalar occ = max(indI[celli], floorW);
        const scalar dRef =
            tornRef && tornW[celli] ? dsW[celli] : dI[celli];
        absDev = max(absDev, mag(wUsed[celli] - occ*(1.0 - dRef)));
    }
    reduce(absDev, maxOp<scalar>());
    s16WeightAbsDevMax_ = max(s16WeightAbsDevMax_, absDev);
    s16WeightSites_ |= (1 << site);
}


// S23 rev. 4.1 F1' (design section 13.4 G23-5; red team 21 change 7).
void Foam::solvers::phaseClotAxisymmetricFluid::recordTornWeight(
    const scalarField& wUsed,
    const label site
) const
{
    // On T: w/max(ind, floor) = aI, to round-off, for the weight AS CONSUMED
    // -- called at each builder after applyTornStressWeight() and
    // recordStressWeight(), with no write to the weight after it. aI is
    // intactShare(), the share the face correction weights Pt with. The
    // maximum since the last PHASE_CLOT_S23 line, per site; the largest |T|
    // a call examined; a zero occupancy is counted, never divided.
    boolList tornF;
    scalarField sTornF, dsF, indF;
    const label nTF = tornNetworkCells(tornF, sTornF, &dsF, &indF);
    s23F1WCells_[site] = max(s23F1WCells_[site], nTF);
    if (nTF == 0)
    {
        return;
    }
    const volScalarField aIF(intactShare());
    const scalarField& aIW = aIF.primitiveField();
    const scalar floorF = skeletonSolidFloor_;
    scalar devF = 0;
    label nZeroF = 0;
    forAll(wUsed, celli)
    {
        if (!tornF[celli])
        {
            continue;
        }
        const scalar occF = max(indF[celli], floorF);
        if (occF > 0)
        {
            devF = max
            (
                devF,
                damagePartitionKernel::f1WeightDeviation
                (
                    wUsed[celli], occF, aIW[celli]
                )
            );
        }
        else
        {
            ++nZeroF;
        }
    }
    reduce(devF, maxOp<scalar>());
    reduce(nZeroF, sumOp<label>());
    s23F1WDev_[site] = max(s23F1WDev_[site], devF);
    s23F1WZeroOcc_ = max(s23F1WZeroOcc_, nZeroF);
}


Foam::scalar
Foam::solvers::phaseClotAxisymmetricFluid::progressiveDamageCeiling() const
{
    // S23 rev. 4.1 section 13.2 item 1: d_p is the progressive branch of the
    // rupture law (ruptureOverstretchedNetwork(): 1 - exp(-(l1 - lf)/(2
    // dlf)) for lf < l1 < lc) at its upper end l1 = lc, evaluated from the
    // law's parameters, never as a literal (0.2936 at 2.8/2.105/1.0).
    return damagePartitionKernel::progressiveCeiling
    (
        networkCriticalStretch_,
        networkDamageOnsetStretch_,
        networkDamageStretchScale_
    );
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::networkPressureDamaged
(
    const tmp<volScalarField>& tp
) const
{
    // S16 D (F3): p_n -> max(p_n, 0) + (1 - d) min(p_n, 0). p_n > 0 is
    // packing (a > phi_s0), p_n < 0 tension (dilation): a broken network
    // cannot resist dilation, its debris still resists packing. With d = 0
    // this is p_n exactly (p + 1*0 and 0 + 1*p), so an undamaged cell is
    // unchanged bit for bit.
    const volScalarField& p = tp();
    const dimensionedScalar zero("zero", p.dimensions(), 0);
    {
        // S17 A (design 2A "Stress"): p_used = max(p_n, 0) + (thI/S)
        // min(p_n, 0) + (thP/S) p_P. The intact share resists dilation at
        // the current tension; the pending share keeps the tension frozen at
        // its event, q_P/S = (thP/S) p_P, and releases it with its content
        // (the release scales q_P with thP). Undamaged: thI/S = 1 and
        // q_P = 0, so this is S16's max + 1*min bit for bit.
        volScalarField qByS
        (
            IOobject
            (
                "pendingPressureShare", mesh_.time().name(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE, false
            ),
            mesh_,
            dimensionedScalar(p.dimensions(), 0),
            zeroGradientFvPatchField<scalar>::typeName
        );
        {
            const scalarField& pI = thetaIntact_.primitiveField();
            const scalarField& pP = thetaPending_.primitiveField();
            const scalarField& pB = thetaBroken_.primitiveField();
            const scalarField& qP = thetaPendingP_.primitiveField();
            scalarField& qS = qByS.primitiveFieldRef();
            // S18 A (R1): under frozenPendingPressure none the frozen
            // tension enters no force (fibreReleaseKernel::
            // frozenPressureShare, the expression frozenPower reads too).
            forAll(qS, celli)
            {
                const scalar S = pI[celli] + (pP[celli] + pB[celli]);
                qS[celli] = fibreReleaseKernel::frozenPressureShare
                (
                    qP[celli], S, false
                );
            }
            qByS.correctBoundaryConditions();
        }
        return volScalarField::New
        (
            p.name(),
            max(p, zero) + intactShare()*min(p, zero) + qByS
        );
    }
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::networkPartialPressure() const
{
    // S16 D: the one place every force consumer of the partial pressure
    // (skeletonFaceStress, the face-gradient force and its cell twin,
    // elasticStress, the balance diagnostic) reads it from.
    return networkPressureDamaged(networkPartialPressureUndamaged());
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::networkPartialPressureUndamaged()
const
{
    // convexExcess: the dense network's stress times the occupancy that was
    // deposited. Exact for fully deposited material (thetaNetRef = phi_s0,
    // weight 1); zero where nothing was deposited, however the lumen's
    // numerical tails move -- which is why the argument stays the S10 excess
    // and not the ratio thetaNetRef/theta (AMENDMENT S10: the ratio is
    // singular where there is no network).
    {
        // S13 FIX A: THE EXCESS IS ALREADY PARTIAL. A partial cell of
        // occupancy occ whose network is compacted by intrinsic strain eps
        // holds content excess theta - thetaNetRef = occ*phi0*eps, so
        // p(phi0 + excess) ~ (Kn/phi0)*occ*phi0*eps = occ*Kn*eps -- the
        // diffuse-domain partial pressure, occupancy times the network's own.
        // The S11 weight thetaNetRef/phi0 multiplied that by occ AGAIN
        // (occ^2*Kn*eps), so a fringe cell at occ 1e-3 had a thousandth of
        // the volumetric resistance its own network has. Run 20260923-135545
        // (em55): 80% of the far fringe under tension had no grid-scale
        // restoring against the deviatoric stress's pull, and it ran away.
        // Precedent: multiphaseEuler's phase pressure is p(alpha), applied
        // as pPrime*snGrad(alpha) (momentumTransferSystem::Fs) -- a function
        // of the fraction, never weighted by the fraction again.
        // networkPresence() keeps the lumen at zero exactly as the S11
        // weight did, below residualChi of deposited network.
        if (networkPressureWeight_ == "networkPresence")
        {
            return volScalarField::New
            (
                "networkPartialPressure",
                networkPresence()*networkPressure()
            );
        }
        // S18 C (R2b; DECISIONS 2026-09-29, approved as a first-principles
        // law): source 48 Eq. 25 applies the network's own pressure as
        // -grad(phi_n psi), the intrinsic pressure weighted ONCE, inside the
        // gradient, by the network fraction present. The presence-weighted
        // excess law is, to first order, the deposited network's partial
        // pressure, so its Eq.-25 weight is theta/ref = 1/J. It multiplies
        // the intact pressure on BOTH branches (it replaces batch 58's
        // min(1, theta/ref)), before the damage mapping of
        // networkPressureDamaged(): theta/ref >= 0, so max(w p, 0) +
        // aI min(w p, 0) = w [max(p, 0) + aI min(p, 0)] -- the weight
        // multiplies p_used's intact part. Every force reader of
        // networkPartialPressure() takes it through -grad(w p).
        if (networkPressureWeight_ == "contentRatio")
        {
            return volScalarField::New
            (
                "networkPartialPressure",
                networkPresence()*networkContentRatio()*networkPressure()
            );
        }
        return volScalarField::New
        (
            "networkPartialPressure",
            thetaNetRef_
           /dimensionedScalar("phiS0", dimless, skeletonVolumeFraction_)
           *networkPressure()
        );
    }
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::networkContentRatio() const
{
    // S18 C (R2b): theta/thetaNetRef with theta = thetaC + thetaH, the
    // content networkPressureArgument() reads. 0 where no network was
    // deposited (thetaNetRef = 0), where networkPresence() is 0 as well,
    // so the product does not depend on it. Its patches follow the cell
    // (zeroGradient), as theta's and thetaNetRef's do.
    tmp<volScalarField> tw
    (
        new volScalarField
        (
            IOobject
            (
                "networkContentRatio", mesh_.time().name(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE, false
            ),
            mesh_,
            dimensionedScalar(dimless, 0),
            zeroGradientFvPatchField<scalar>::typeName
        )
    );
    scalarField& wI = tw.ref().primitiveFieldRef();
    const scalarField& refI = thetaNetRef_.primitiveField();
    forAll(wI, celli)
    {
        wI[celli] = networkPressureKernel::contentRatio
        (
            thetaC_[celli] + thetaH_[celli], refI[celli]
        );
    }
    tw.ref().correctBoundaryConditions();
    return tw;
}


// ---- S17 A: THE PENDING SHARE AS A FIBRE ----------------------------------
//
// research/designs/2026-09-26-S17-frozen-release-implicit-elastic-startup.md
// section 2A; the arithmetic is fibreReleaseKernel.H, tested by
// scripts/s17_fibre_kernel_test.cpp.

const Foam::volScalarField&
Foam::solvers::phaseClotAxisymmetricFluid::fibreCarrier() const
{
    // S17b A1 (V1): the fibre's tensor, tension and stress weight belong to
    // the TAUT content; slack content (theta_P - theta_T) carries none.
    return thetaTaut_;
}


Foam::scalar Foam::solvers::phaseClotAxisymmetricFluid::fibreReadMin() const
{
    // S17b A2: below DBL_MIN a content has no significand to divide by, so
    // C_B/theta is not a blend of fibres (05 D2). A representability limit,
    // not a presence rule or a tolerance: the stress it skips is <= ~6e-252 G
    // (12). Under "shipped", S17's "> 0" (the smallest positive double).
    return std::numeric_limits<scalar>::min();
}


template<class Type>
void Foam::solvers::phaseClotAxisymmetricFluid::scaleToFieldUnits
(
    fvMatrix<Type>& m
) const
{
    // S17b A2 (05 D2): every row of an assembled predictor times dt/V, so
    // the diagonal is O(1) and a Gauss-Seidel division no longer turns one
    // subnormal ulp of the content into 4.94e-324 V^-1 dt of noise. The
    // solution is the same system's; only the arithmetic changes.
    // upper[f] is row lowerAddr[f]'s coefficient, lower[f] row upperAddr[f]'s.
    const scalar dt = runTime.deltaTValue();
    const scalarField& V = mesh_.V();
    const scalarField sc(dt/V);
    const labelUList& l = m.lduAddr().lowerAddr();
    const labelUList& u = m.lduAddr().upperAddr();
    scalarField& lo = m.lower();
    scalarField& up = m.upper();
    forAll(up, f)
    {
        up[f] *= sc[l[f]];
        lo[f] *= sc[u[f]];
    }
    m.diag() *= sc;
    m.source() *= sc;
    forAll(m.internalCoeffs(), patchi)
    {
        const labelUList& fc = m.lduAddr().patchAddr(patchi);
        Field<Type>& ic = m.internalCoeffs()[patchi];
        Field<Type>& bc = m.boundaryCoeffs()[patchi];
        forAll(fc, i)
        {
            ic[i] *= sc[fc[i]];
            bc[i] *= sc[fc[i]];
        }
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::s17bForceParts
(
    const surfaceScalarField& iAD11f,
    const label w,
    const vector& dir,
    scalar parts[4]
) const
{
    // The force paths of skeletonElasticForceFlux()/skeletonFaceStress(),
    // one part at a time (the force is linear in them): the deviatoric
    // split M = M_I + M_P, and p_used = [max(p,0) + aI min(p,0)] + q_P/S.
    // S18 F (H11; docket 03 D3): each part carries the APPLIED force's wall
    // patches -- p_used's patches follow the cell (zeroGradient partition,
    // theta and thetaNetRef), and splitDev's wall patch is dev(be)
    // (wallPatchSplit shipped) or the cell's split (split) -- where the
    // S17b diagnostic kept the undamaged pressure and a zero frozen part.
    // The frozen part is fibreReleaseKernel::frozenPressureShare, the
    // applied expression (0 under frozenPendingPressure none).
    for (int k = 0; k < 4; ++k) { parts[k] = 0; }
    if (w < 0)
    {
        return;
    }
    const scalarField& pI = thetaIntact_.primitiveField();
    const scalarField& pP = thetaPending_.primitiveField();
    const scalarField& pB = thetaBroken_.primitiveField();
    const scalarField& pC = fibreCarrier().primitiveField();
    const scalarField& qP = thetaPendingP_.primitiveField();
    const vectorField& nI = pendingAxis_.primitiveField();
    const scalarField& lI = pendingStretch_.primitiveField();
    volSymmTensorField MI("MI", dev(be_));
    volSymmTensorField MP("MP", 0.0*dev(be_));
    volScalarField pInt("pIntactPart", networkPartialPressureUndamaged());
    volScalarField pFro("pFrozenPart", 0.0*pInt);
    if (skeletonLawLT_)
    {
        // S25 A: the intact part's deviator is LT's (as splitDeviatoricStress).
        MI = skeletonLawDeviatorField();
    }
    {
        symmTensorField& mi = MI.primitiveFieldRef();
        symmTensorField& mp = MP.primitiveFieldRef();
        scalarField& pi = pInt.primitiveFieldRef();
        scalarField& pf = pFro.primitiveFieldRef();
        forAll(mi, celli)
        {
            scalar aI, aP, omd;
            fibreReleaseKernel::sharesCarrier
            (
                pI[celli], pP[celli], pB[celli], pC[celli], aI, aP, omd
            );
            const symmTensor d(mi[celli]);
            const double devBe[6] = {d.xx(), d.xy(), d.xz(), d.yy(), d.yz(), d.zz()};
            const double n[3] = {nI[celli].x(), nI[celli].y(), nI[celli].z()};
            double m1[6], m2[6];
            fibreReleaseKernel::splitDev(aI, devBe, 0.0, lI[celli], n, omd, m1);
            fibreReleaseKernel::splitDev(0.0, devBe, aP, lI[celli], n, omd, m2);
            mi[celli] = symmTensor(m1[0], m1[1], m1[2], m1[3], m1[4], m1[5]);
            mp[celli] = symmTensor(m2[0], m2[1], m2[2], m2[3], m2[4], m2[5]);
            const scalar pn = pi[celli];
            const scalar S = pI[celli] + (pP[celli] + pB[celli]);
            pi[celli] = max(pn, scalar(0)) + aI*min(pn, scalar(0));
            pf[celli] = fibreReleaseKernel::frozenPressureShare
            (
                qP[celli], S, false
            );
        }
        MI.correctBoundaryConditions();
        MP.correctBoundaryConditions();
        pInt.correctBoundaryConditions();
        pFro.correctBoundaryConditions();
        // the applied force's wall patches (S18 F, H11)
        forAll(mesh.boundary(), patchi)
        {
            if (!isA<wallFvPatch>(mesh.boundary()[patchi]))
            {
                continue;
            }
            pInt.boundaryFieldRef()[patchi] =
                pInt.boundaryField()[patchi].patchInternalField();
            pFro.boundaryFieldRef()[patchi] =
                pFro.boundaryField()[patchi].patchInternalField();
            {
                MI.boundaryFieldRef()[patchi] =
                    MI.boundaryField()[patchi].patchInternalField();
                MP.boundaryFieldRef()[patchi] =
                    MP.boundaryField()[patchi].patchInternalField();
            }
        }
    }
    const volScalarField wS(skeletonStressWeight());
    const volScalarField gD(sqr(1.0 - Dg_) + etaRes_);
    auto devFlux = [&](const volSymmTensorField& M)
    {
        const volSymmTensorField Sd("Sd", gD*GField_*M);
        const surfaceSymmTensorField sf
        (
          surfaceSymmTensorField::New
            (
                "sfPart", seriesFaceWeights(wS)*fvc::interpolate(Sd)
            )
        );
        return surfaceScalarField::New
        (
            "fPart",
            fvc::interpolate
            (
                fvc::surfaceIntegrateExtrapolate(mesh.Sf() & sf)/rho_
            ) & mesh.Sf()
        );
    };
    auto presFlux = [&](const volScalarField& pp)
    {
        {
            return surfaceScalarField::New
            (
                "fPart", -fvc::snGrad(pp)*mesh.magSf()/rho_
            );
        }
    };
    const vector dHat(dir/max(mag(dir), vSmall));
    parts[0] = fvc::reconstruct(iAD11f*devFlux(MI)())()[w] & dHat;
    parts[1] = fvc::reconstruct(iAD11f*devFlux(MP)())()[w] & dHat;
    parts[2] = fvc::reconstruct(iAD11f*presFlux(pInt)())()[w] & dHat;
    parts[3] = fvc::reconstruct(iAD11f*presFlux(pFro)())()[w] & dHat;
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::intactShare() const
{
    // thI/S = P1/S, 1 where S = 0 (fibreReleaseKernel::shares). The share of
    // the cell's network that is intact: its stress weight on dev(be), the
    // noMemory relaxation's intact content and the deposit's mixing weight
    // (docket 09 rules 2, 3).
    tmp<volScalarField> tA
    (
        new volScalarField
        (
            IOobject
            (
                "intactShare", mesh_.time().name(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE, false
            ),
            mesh_,
            dimensionedScalar(dimless, 1),
            zeroGradientFvPatchField<scalar>::typeName
        )
    );
    const scalarField& pI = thetaIntact_.primitiveField();
    const scalarField& pP = thetaPending_.primitiveField();
    const scalarField& pB = thetaBroken_.primitiveField();
    scalarField& a = tA.ref().primitiveFieldRef();
    forAll(a, celli)
    {
        scalar aI, aP, omd;
        fibreReleaseKernel::shares(pI[celli], pP[celli], pB[celli], aI, aP, omd);
        a[celli] = aI;
    }
    tA.ref().correctBoundaryConditions();
    return tA;
}


void Foam::solvers::phaseClotAxisymmetricFluid::syncPendingFibres
(
    const char* site
)
{
    // The derived, written fields from the carried state: B_P = fibre(n, lf)
    // and p_P = q_P/thP (0 where thP = 0). Every value finite, lf >= 1, n
    // unit: a violation is a defect at the named site, not a value to bound.
    const scalarField& pP = thetaPending_.primitiveField();
    // S17b A1: the fibre's carrier (theta_T under V1, P2 otherwise)
    const scalarField& pC = fibreCarrier().primitiveField();
    const scalarField& qP = thetaPendingP_.primitiveField();
    const vectorField& nI = pendingAxis_.primitiveField();
    const scalarField& lI = pendingStretch_.primitiveField();
    symmTensorField& bP = pendingB_.primitiveFieldRef();
    scalarField& ppI = pendingPressure_.primitiveFieldRef();
    label nBad = 0, firstBad = -1;
    scalar ppMax = -great;
    forAll(pP, celli)
    {
        const scalar lf = lI[celli];
        const scalar nMag = mag(nI[celli]);
        if
        (
            !(std::isfinite(lf) && lf >= 1 && std::isfinite(qP[celli])
           && mag(nMag - 1.0) <= 1e-12)
        )
        {
            if (firstBad < 0) { firstBad = celli; }
            ++nBad;
            continue;
        }
        const double nA[3] = {nI[celli].x(), nI[celli].y(), nI[celli].z()};
        double B[6];
        fibreReleaseKernel::fibre(nA, lf, B);
        bP[celli] = symmTensor(B[0], B[1], B[2], B[3], B[4], B[5]);
        ppI[celli] = pC[celli] > 0 ? qP[celli]/pC[celli] : 0.0;
        if (pC[celli] > 0) { ppMax = max(ppMax, ppI[celli]); }
    }
    if (firstBad >= 0)
    {
        Info<< "PHASE_CLOT_S17_FIBRE_INVALID site=" << site
            << " cell=" << firstBad
            << " pendingStretch=" << lI[firstBad]
            << " axisMag=" << mag(nI[firstBad])
            << " thetaPendingP=" << qP[firstBad] << endl;
    }
    reduce(nBad, sumOp<label>());
    if (nBad > 0)
    {
        FatalErrorInFunction
            << "the pending fibre is invalid in " << nBad << " cell(s) after "
            << site << " (lf < 1, a non-unit axis or a non-finite value; the "
            << "first is in PHASE_CLOT_S17_FIBRE_INVALID). P-LS, the rotation "
            << "and the event write only fibres; this is a defect, not "
            << "something to bound." << exit(FatalError);
    }
    reduce(ppMax, maxOp<scalar>());
    s17PendingPressureMax_ = max(s17PendingPressureMax_, ppMax);
    pendingB_.correctBoundaryConditions();
    pendingPressure_.correctBoundaryConditions();
}


bool Foam::solvers::phaseClotAxisymmetricFluid::s18pTakeCell
(
    const label site,
    const bool present
)
{
    // S18p (design section 2): every raised network-present cell, up to 200
    // per run; otherwise 5 raised cells per site in each 0.25 ms window.
    if (present && s18pPresentDumped_ < 200)
    {
        ++s18pPresentDumped_;
        return true;
    }
    const label w = label(std::floor(runTime.value()/2.5e-4));
    if (w != s18pWindow_[site])
    {
        s18pWindow_[site] = w;
        s18pWindowDumped_[site] = 0;
    }
    if (s18pWindowDumped_[site] < 5)
    {
        ++s18pWindowDumped_[site];
        return true;
    }
    return false;
}


void Foam::solvers::phaseClotAxisymmetricFluid::projectPendingFibres
(
    const char* site
)
{
    // P-LS (docket 12 section 1) in every cell holding pending content:
    // B_mix = C_B/thP, n the major eigenvector of dev(B_mix), T the
    // least-squares tension, lf from lf^2 - 1/lf = T, ties to the previous
    // axis; C_B <- thP fibre(n, lf). G-proj: the energy never rises against
    // the blend (fibreReleaseKernel::dissipative), counted every call.
    // S17b: the CARRIER (theta_T under V1: slack content is never blended
    // into a taut fibre, 07), read only where it is >= fibreReadMin() (A2).
    const scalarField& pP = fibreCarrier().primitiveField();
    symmTensorField& cB = thetaPendingB_.primitiveFieldRef();
    vectorField& nI = pendingAxis_.primitiveFieldRef();
    scalarField& lI = pendingStretch_.primitiveFieldRef();
    const scalar readMin = fibreReadMin();
    const scalarField& refP = thetaNetRef_.primitiveField();
    const scalar residualP = residualChi_*skeletonVolumeFraction_;
    label nProj = 0, nRaised = 0, nTie = 0, nFail = 0, firstFail = -1;
    label nRaisedPresent = 0, nRaisedAbsent = 0;
    scalar excess = 0;
    forAll(pP, celli)
    {
        const scalar th = pP[celli];
        if (!(th > 0))
        {
            // no pending content: the empty population's fibre
            cB[celli] = Zero;
            lI[celli] = 1.0;
            continue;
        }
        if (!(th >= readMin))
        {
            // S17b A2: below DBL_MIN the blend cannot be formed; the cell
            // keeps its last valid fibre (n, lf >= 1), and its content is
            // re-expressed from it.
            const double nA[3] = {nI[celli].x(), nI[celli].y(), nI[celli].z()};
            double Bk[6];
            fibreReleaseKernel::fibre(nA, lI[celli], Bk);
            cB[celli] = th*symmTensor(Bk[0], Bk[1], Bk[2], Bk[3], Bk[4], Bk[5]);
            continue;
        }
        const symmTensor Bm(cB[celli]/th);
        const double Bmix[6] =
            {Bm.xx(), Bm.xy(), Bm.xz(), Bm.yy(), Bm.yz(), Bm.zz()};
        const double nPrev[3] = {nI[celli].x(), nI[celli].y(), nI[celli].z()};
        const fibreReleaseKernel::Projection P =
            fibreReleaseKernel::projectLS(Bmix, nPrev);
        if (!P.ok || !std::isfinite(P.lf))
        {
            if (firstFail < 0) { firstFail = celli; }
            ++nFail;
            continue;
        }
        ++nProj;
        if (P.tie) { ++nTie; }
        if (!fibreReleaseKernel::dissipative(P))
        {
            ++nRaised;
            if (refP[celli] > residualP) { ++nRaisedPresent; }
            else { ++nRaisedAbsent; }
            excess = max(excess, P.psiOut - P.psiMix);
        }
        // S18p: the transport site's raised cell, with the blend's input and
        // the passengers kernel's locals of this step (log only).
        if
        (
            projSiteLog_ && !fibreReleaseKernel::dissipative(P)
         && s18pTakeCell(1, refP[celli] > residualP)
        )
        {
            std::ostringstream os;
            os.precision(17);
            os << "PHASE_CLOT_S18P_CELL time=" << runTime.value()
               << " step=" << runTime.timeIndex()
               << " site=P cell=" << celli
               << " present=" << (refP[celli] > residualP)
               << " th=" << th;
            for (int k = 0; k < 6; ++k)
            {
                os << " Bmix" << k << "=" << Bmix[k];
            }
            os << " psiMix=" << P.psiMix << " psiOut=" << P.psiOut
               << " lfOut=" << P.lf;
            const damagePartitionKernel::PassengersDebug<scalar>& kd =
                s18pKernelDbg_;
            if (kd.nCells == long(pP.size()))
            {
                const std::size_t c = std::size_t(celli);
                const std::size_t n = std::size_t(kd.nCells);
                os << " kernel=passengersShared";
                for (int x = 0; x < kd.nX; ++x)
                {
                    os << " A" << x << "=" << kd.A[std::size_t(x)*n + c]
                       << " spec" << x << "=" << kd.spec[std::size_t(x)*n + c]
                       << " inX" << x << "=" << kd.inX[std::size_t(x)*n + c];
                }
                os << " kept=" << kd.kept[c] << " inB=" << kd.inB[c]
                   << " Bdiv=" << kd.Bdiv[c]
                   << " PkAfterC=" << kd.PkAfterC[c]
                   << " PkNew=" << kd.PkNew[c];
            }
            else
            {
                // cplusOrder shipped: passengers() has no A/B divisor.
                os << " kernel=none";
            }
            Info<< os.str().c_str() << endl;
        }
        nI[celli] = vector(P.n[0], P.n[1], P.n[2]);
        lI[celli] = P.lf;
        double B[6];
        fibreReleaseKernel::fibre(P.n, P.lf, B);
        cB[celli] = th*symmTensor(B[0], B[1], B[2], B[3], B[4], B[5]);
    }
    reduce(nFail, sumOp<label>());
    if (nFail > 0)
    {
        FatalErrorInFunction
            << "P-LS did not converge in " << nFail << " cell(s) after "
            << site << " (the first is cell " << firstFail << " on its "
            << "processor): the blend is not a finite symmetric tensor."
            << exit(FatalError);
    }
    reduce(nProj, sumOp<label>());
    reduce(nRaised, sumOp<label>());
    reduce(nTie, sumOp<label>());
    reduce(excess, maxOp<scalar>());
    reduce(nRaisedPresent, sumOp<label>());
    reduce(nRaisedAbsent, sumOp<label>());
    s17ProjCount_ += nProj;
    s17ProjRaised_ += nRaised;
    s17ProjTies_ += nTie;
    s17ProjExcessMax_ = max(s17ProjExcessMax_, excess);
    s17bProjRaisedPresent_ += nRaisedPresent;
    s17bProjRaisedAbsent_ += nRaisedAbsent;
    if (projSiteLog_)
    {
        s18pCount_[1] += nProj;
        s18pRaised_[1] += nRaised;
        s18pRaisedPresent_[1] += nRaisedPresent;
        s18pRaisedAbsent_[1] += nRaisedAbsent;
        s18pExcessMax_[1] = max(s18pExcessMax_[1], excess);
    }
    thetaPendingB_.correctBoundaryConditions();
    pendingAxis_.correctBoundaryConditions();
    pendingStretch_.correctBoundaryConditions();
    syncPendingFibres(site);
    // the site-tagged max lf over read cells (12 change 6)
    scalar lfMax = 0;
    forAll(pP, celli)
    {
        if (pP[celli] >= std::numeric_limits<scalar>::min())
        {
            lfMax = max(lfMax, lI[celli]);
        }
    }
    reduce(lfMax, maxOp<scalar>());
    s17bMaxLfTransport_ = max(s17bMaxLfTransport_, lfMax);
}


void Foam::solvers::phaseClotAxisymmetricFluid::rotatePendingFibres
(
    const volTensorField& gradU
)
{
    // ONCE PER STEP, after the be transport and before the law (docket 09
    // section 2; 12's order: transport, P-LS, rotation, stretch update,
    // release and stress). L = grad(U)^T: OpenFOAM's gradient is
    // (grad U)_ij = d_i U_j. G-rot, G-eig and G-uni are measured HERE,
    // across this kernel only (12).
    const scalar dt = runTime.deltaTValue();
    // S17b A1: the taut content carries the fibre under V1.
    const scalarField& pP = fibreCarrier().primitiveField();
    const tensorField& gI = gradU.primitiveField();
    symmTensorField& cB = thetaPendingB_.primitiveFieldRef();
    vectorField& nI = pendingAxis_.primitiveFieldRef();
    scalarField& lI = pendingStretch_.primitiveFieldRef();
    scalarField& tT = thetaTaut_.primitiveFieldRef();
    scalarField& qP = thetaPendingP_.primitiveFieldRef();
    const scalarField& p1S = thetaIntact_.primitiveField();
    const scalarField& p2S = thetaPending_.primitiveField();
    const scalarField& p3S = thetaBroken_.primitiveField();
    const scalarField& volS = mesh_.V();
    const scalarField& GI = GField_.primitiveField();
    DynamicList<scalar> ratios;
    DynamicList<scalar> ldts;
    scalar eigMax = 0;
    label nUp = 0, nFail = 0, firstFail = -1;
    label nSlack = 0, nAtRest = 0;
    scalar slackContent = 0, slackDd = 0, slackMu = 0, slackQp = 0;
    scalar slackPreT = 0, slackPreTRatio = 0, slackPreQp = 0, slackPreQpRatio = 0;
    forAll(pP, celli)
    {
        const scalar th = pP[celli];
        if (!(th > 0))
        {
            // no taut population: H-PA2's accumulators restart with the next
            s17bSumLnMu_[celli] = 0;
            s17bSumMinLnMu_[celli] = 0;
            continue;
        }
        const tensor& g = gI[celli];
        const double L[3][3] =
        {
            {g.xx(), g.yx(), g.zx()},
            {g.xy(), g.yy(), g.zy()},
            {g.xz(), g.yz(), g.zz()}
        };
        double n[3] = {nI[celli].x(), nI[celli].y(), nI[celli].z()};
        double lf = lI[celli];
        const double lfIn = lf;
        double B[6];
        const fibreReleaseKernel::Rotation r =
            fibreReleaseKernel::rotateStretch(L, dt, n, lf, B);
        if (!r.ok)
        {
            if (firstFail < 0) { firstFail = celli; }
            ++nFail;
            continue;
        }
        eigMax = max(eigMax, r.eigRel);
        if (lf > lfIn) { ++nUp; }
        // G-rot, redefined after verification 17 (design section 6): the
        // kernel's one exact step differs from the rate by a known
        // O(|L|dt) term (1 - 0.316 |L|dt in simple shear), so the ratio is
        // evaluated only where |L|dt <= 1e-3 (the term <= 3.2e-4 against
        // the +-0.001 band); the median |L|dt of every candidate cell is
        // logged beside it, so the qualifying set's representativeness is
        // visible. No qualifying cell: NOT-EVALUABLE (rotCells = 0).
        if (r.omega > 1e-2*r.Lnorm && r.omega*dt > 0)
        {
            ldts.append(r.Lnorm*dt);
            if (r.Lnorm*dt <= 1e-3)
            {
                ratios.append(r.angle/(dt*r.omega));
            }
        }
        nI[celli] = vector(n[0], n[1], n[2]);
        s17bSumLnMu_[celli] += std::log(r.mu);
        s17bSumMinLnMu_[celli] += min(scalar(0), std::log(r.mu));
        {
            // S17b A1, V1 (07 section 2): the slack rule and the q_P clause,
            // fibreReleaseKernel::slackStep. lfOut equals the kernel's UNI
            // value bit for bit on a taut step.
            double lfV1 = lfIn, qRatio = 1;
            if (fibreReleaseKernel::slackStep(lfIn, r.mu, lfV1, qRatio))
            {
                // THE SLACK EVENT: the taut population, its tensor and its
                // frozen tension go to 0 at once; the content stays in P2
                // (the partition is not written here), so d and the load-
                // path weight do not step. Design section 8 d:
                // - THE GATE: |dd|, from the partition before and after
                //   (it fails if the event writes P2 or P3);
                // - CONFORMANCE (read back from the stores): q_P = 0 after,
                //   and A1's condition "mu < 1 or lf_in = 1" (max mu over
                //   the events with lf_in != 1, and the lf_in = 1 count);
                // - A RUNTIME WATCH on the residual the event drops, read
                //   BEFORE the stores: the tension T(lf_in) theta_T/S and
                //   |q_P|/S, each against the one-step contraction T(1/mu)
                //   (as T(lf_in)/T(1/mu), and |q_P|/(S G T(1/mu))).
                const scalar S0 = p1S[celli] + (p2S[celli] + p3S[celli]);
                const scalar d0 = S0 > 0 ? p3S[celli]/S0 : 0.0;
                const scalar qIn = qP[celli];
                const scalar thOld = tT[celli];
                tT[celli] = 0.0;
                cB[celli] = Zero;
                qP[celli] = 0.0;
                lI[celli] = 1.0;
                const scalar S1 = p1S[celli] + (p2S[celli] + p3S[celli]);
                const scalar d1 = S1 > 0 ? p3S[celli]/S1 : 0.0;
                ++nSlack;
                slackContent += thOld*volS[celli];
                slackDd = max(slackDd, mag(d1 - d0));
                if (lfIn == 1.0)
                {
                    ++nAtRest;
                }
                else
                {
                    slackMu = max(slackMu, r.mu);
                }
                slackQp = max(slackQp, mag(qP[celli]));
                if (S0 > 0)
                {
                    const scalar tIn =
                        fibreReleaseKernel::fibreTension(lfIn)*thOld/S0;
                    const scalar qInS = mag(qIn)/S0;
                    slackPreT = max(slackPreT, tIn);
                    slackPreQp = max(slackPreQp, qInS);
                    const scalar Tb =
                        r.mu < 1.0
                      ? fibreReleaseKernel::fibreTension(1.0/r.mu)
                      : 0.0;
                    if (Tb > 0)
                    {
                        slackPreTRatio = max
                        (
                            slackPreTRatio,
                            fibreReleaseKernel::fibreTension(lfIn)/Tb
                        );
                        if (GI[celli] > 0)
                        {
                            slackPreQpRatio =
                                max(slackPreQpRatio, qInS/(GI[celli]*Tb));
                        }
                    }
                }
                s17bLastSlackTime_[celli] = runTime.value();
                s17bSumLnMu_[celli] = 0;
                s17bSumMinLnMu_[celli] = 0;
                continue;
            }
            // taut: the kernel's UNI value, and q_P unloads with T
            lI[celli] = lfV1;
            qP[celli] = qP[celli]*qRatio;
            cB[celli] = th*symmTensor(B[0], B[1], B[2], B[3], B[4], B[5]);
            continue;
        }
        lI[celli] = lf;
        cB[celli] = th*symmTensor(B[0], B[1], B[2], B[3], B[4], B[5]);
    }
    reduce(nSlack, sumOp<label>());
    reduce(nAtRest, sumOp<label>());
    reduce(slackContent, sumOp<scalar>());
    reduce(slackDd, maxOp<scalar>());
    reduce(slackMu, maxOp<scalar>());
    reduce(slackQp, maxOp<scalar>());
    reduce(slackPreT, maxOp<scalar>());
    reduce(slackPreTRatio, maxOp<scalar>());
    reduce(slackPreQp, maxOp<scalar>());
    reduce(slackPreQpRatio, maxOp<scalar>());
    s17bSlackEvents_ += nSlack;
    s17bSlackAtRest_ += nAtRest;
    s17bSlackContent_ += slackContent;
    s17bSlackDdMax_ = max(s17bSlackDdMax_, slackDd);
    s17bSlackMuMax_ = max(s17bSlackMuMax_, slackMu);
    s17bSlackQpMax_ = max(s17bSlackQpMax_, slackQp);
    s17bSlackPreTMax_ = max(s17bSlackPreTMax_, slackPreT);
    s17bSlackPreTRatioMax_ = max(s17bSlackPreTRatioMax_, slackPreTRatio);
    s17bSlackPreQpMax_ = max(s17bSlackPreQpMax_, slackPreQp);
    s17bSlackPreQpRatioMax_ = max(s17bSlackPreQpRatioMax_, slackPreQpRatio);
    {
        thetaTaut_.correctBoundaryConditions();
        thetaPendingP_.correctBoundaryConditions();
    }
    reduce(nFail, sumOp<label>());
    if (nFail > 0)
    {
        FatalErrorInFunction
            << "the pending fibre's step deformation F = exp(L dt) collapsed "
            << "the line or inverted the element in " << nFail << " cell(s) "
            << "(the first is cell " << firstFail << " on its processor): "
            << "|F n| = 0 or det F <= 0." << exit(FatalError);
    }
    // The median over this processor's rotated cells (the case runs
    // serial; in parallel this is the median of the master's cells).
    scalar median = 1;
    label nRot = ratios.size();
    if (nRot > 0)
    {
        List<scalar> r(ratios);
        std::nth_element(r.begin(), r.begin() + r.size()/2, r.end());
        median = r[r.size()/2];
    }
    scalar ldtMedian = 0;
    label nCand = ldts.size();
    if (nCand > 0)
    {
        List<scalar> q(ldts);
        std::nth_element(q.begin(), q.begin() + q.size()/2, q.end());
        ldtMedian = q[q.size()/2];
    }
    reduce(nCand, sumOp<label>());
    s17RotLdtMedian_ = ldtMedian;
    s17RotCandidates_ = nCand;
    reduce(nUp, sumOp<label>());
    reduce(eigMax, maxOp<scalar>());
    reduce(nRot, sumOp<label>());
    s17RotRatioMedian_ = median;
    s17RotCells_ = nRot;
    s17EigRelMax_ = max(s17EigRelMax_, eigMax);
    s17UniUp_ += nUp;
    thetaPendingB_.correctBoundaryConditions();
    pendingAxis_.correctBoundaryConditions();
    pendingStretch_.correctBoundaryConditions();
    syncPendingFibres("rotation");
    scalar lfMax = 0;
    forAll(pP, celli)
    {
        if (pP[celli] >= std::numeric_limits<scalar>::min())
        {
            lfMax = max(lfMax, lI[celli]);
        }
    }
    reduce(lfMax, maxOp<scalar>());
    s17bMaxLfRotation_ = max(s17bMaxLfRotation_, lfMax);
}


Foam::tmp<Foam::volSymmTensorField>
Foam::solvers::phaseClotAxisymmetricFluid::splitDeviatoricStress() const
{
    // M = [(thI/S) dev(be) + (thP/S) T dev(n n)]/(1 - d), per cell
    // (fibreReleaseKernel::splitDev); the stress sites multiply it by the
    // occupancy weight max(ind, floor)(1 - d) and gD G, so the stress is
    // gD G max(ind, floor) [(thI/S) dev(be) + (thP/S) T dev(n n)] (design
    // 2A). Undamaged cells: dev(be) bit for bit.
    tmp<volSymmTensorField> tM
    (
        volSymmTensorField::New("splitDev", dev(be_))
    );
    if (skeletonLawLT_)
    {
        // S25 A: M_LT on the cells AND on every patch face (the non-wall
        // patches keep it; the wall patches are re-set below exactly as
        // under the shipped law).
        tM.ref() = skeletonLawDeviatorField();
    }
    const scalarField& pI = thetaIntact_.primitiveField();
    const scalarField& pP = thetaPending_.primitiveField();
    // S17b A1: the fibre's carrier (theta_T under V1, P2 otherwise)
    const scalarField& pC = fibreCarrier().primitiveField();
    const scalarField& pB = thetaBroken_.primitiveField();
    const vectorField& nI = pendingAxis_.primitiveField();
    const scalarField& lI = pendingStretch_.primitiveField();
    symmTensorField& M = tM.ref().primitiveFieldRef();
    forAll(M, celli)
    {
        scalar aI, aP, omd;
        fibreReleaseKernel::sharesCarrier(pI[celli], pP[celli], pB[celli], pC[celli], aI, aP, omd);
        const symmTensor& d = M[celli];
        const double devBe[6] = {d.xx(), d.xy(), d.xz(), d.yy(), d.yz(), d.zz()};
        const double n[3] = {nI[celli].x(), nI[celli].y(), nI[celli].z()};
        double m[6];
        fibreReleaseKernel::splitDev(aI, devBe, aP, lI[celli], n, omd, m);
        M[celli] = symmTensor(m[0], m[1], m[2], m[3], m[4], m[5]);
    }
    // Boundary values as dev(be)'s (the split needs the partition, whose
    // patches are zeroGradient; S17split reads the internal field).
    //
    // S18 B (H12; docket 03 D4): that wall patch is the wall-face stress of
    // the skeleton's face force (surfaceIntegrateExtrapolate), and it is
    // what clotWallAdhesion reads through sigmaElastic_ (elasticStress(),
    // prePredictor). Left as dev(be) it applies the UNDAMAGED intact
    // deviatoric at full weight at every damaged wall-row cell. Under
    // wallPatchSplit split the wall patch takes the split M of its cell --
    // be's wall patch is zeroGradient, so this is the same rule applied to
    // the same state. Only wall patches: no partition reaches the others
    // (S = 0 gives M = dev(be) bit for bit there). The departure of each
    // wall face from its cell is recorded (0 under split, by construction;
    // the stale deviatoric under shipped).
    scalar wallDev = 0;
    forAll(tM().boundaryField(), patchi)
    {
        if (!isA<wallFvPatch>(mesh_.boundary()[patchi]))
        {
            continue;
        }
        const symmTensorField inner
        (
            tM().boundaryField()[patchi].patchInternalField()
        );
        {
            tM.ref().boundaryFieldRef()[patchi] = inner;
        }
        const symmTensorField& pf = tM().boundaryField()[patchi];
        forAll(pf, facei)
        {
            wallDev = max(wallDev, mag(pf[facei] - inner[facei]));
        }
    }
    reduce(wallDev, maxOp<scalar>());
    s18WallSplitDevMax_ = max(s18WallSplitDevMax_, wallDev);
    return tM;
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::mixedTangentStiffness() const
{
    // k_eff = [(thI/S) stiff + (thP/S)(lf^2 - 1/lf)]/(1 - d), stiff the
    // STRIPES form: 1 under elasticStabilityStiffness modulus, (8/9) b1 +
    // (2/9)(b2 + b3) under tangent (docket 10, fibreReleaseKernel::
    // mixedStiffness). Undamaged: stiff bit for bit.
    tmp<volScalarField> tK
    (
        new volScalarField
        (
            IOobject
            (
                "mixedTangentStiffness", mesh_.time().name(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE, false
            ),
            mesh_,
            dimensionedScalar(dimless, 0),
            zeroGradientFvPatchField<scalar>::typeName
        )
    );
    const scalarField& pI = thetaIntact_.primitiveField();
    const scalarField& pP = thetaPending_.primitiveField();
    // S17b A1: the fibre's carrier (theta_T under V1, P2 otherwise)
    const scalarField& pC = fibreCarrier().primitiveField();
    const scalarField& pB = thetaBroken_.primitiveField();
    const scalarField& lI = pendingStretch_.primitiveField();
    const symmTensorField& beI = be_.primitiveField();
    scalarField& k = tK.ref().primitiveFieldRef();
    forAll(k, celli)
    {
        scalar stiff = 1.0;
        {
            const vector ev(eigenValues(beI[celli]));
            const scalar b1 = cmptMax(ev);
            stiff = fibreReleaseKernel::stiffIntact
            (
                b1, ev.x() + ev.y() + ev.z() - b1
            );
        }
        if (skeletonLawLT_)
        {
            // S25 A: LT's tangent (design section 3A).
            stiff = intactTangentStiffness(beI[celli]);
        }
        scalar aI, aP, omd;
        fibreReleaseKernel::sharesCarrier(pI[celli], pP[celli], pB[celli], pC[celli], aI, aP, omd);
        k[celli] = fibreReleaseKernel::mixedStiffness(aI, stiff, aP, lI[celli], omd);
    }
    tK.ref().correctBoundaryConditions();
    return tK;
}


Foam::scalar
Foam::solvers::phaseClotAxisymmetricFluid::intactTangentStiffness
(
    const symmTensor& be
) const
{
    if (skeletonLawLT_)
    {
        // S25 A: stiff_LT = [(4/9) t1 + (1/9)(t2 + t3)]/J (numerics 07
        // finding 4), refused unless be is SPD.
        const vector evL(eigenValues(be));
        const double eL[3] = {evL.x(), evL.y(), evL.z()};
        double stiffL = 0;
        if
        (
            !skeletonLawKernel::tangentStiffness
            (
                skeletonLawTable_, eL, det(be), stiffL
            )
        )
        {
            FatalErrorInFunction
                << "skeletonLaw valanisLandelTrilinear: be = " << be
                << " (eigenvalues " << evL << ", det " << det(be) << ") is "
                << "not SPD, so LT's tangent has no real stretches. be is "
                << "checked SPD after each update; this is a defect."
                << exit(FatalError);
        }
        return stiffL;
    }
    // The shipped STRIPES stiff, (8/9) b1 + (2/9)(b2 + b3), as
    // mixedTangentStiffness forms it (same kernel, same argument order).
    const vector evT(eigenValues(be));
    const scalar bMax = cmptMax(evT);
    return fibreReleaseKernel::stiffIntact
    (
        bMax, evT.x() + evT.y() + evT.z() - bMax
    );
}


Foam::scalar
Foam::solvers::phaseClotAxisymmetricFluid::intactWaveTangent
(
    const symmTensor& be
) const
{
    // Shipped law (G dev(be)): rho c_ij^2/G = (b_i - b_j)/(b_i - b_j) b_i
    // = b_i, so k_c = the largest eigenvalue of be (numerics 07 finding 4
    // with tau = lambda^2).
    if (skeletonLawLT_)
    {
        // S25 A: max_{i != j} tauHat[l_i,l_j] l_i^2/(l_i + l_j), over J.
        const vector evW(eigenValues(be));
        const double eW[3] = {evW.x(), evW.y(), evW.z()};
        double kcL = 0;
        if
        (
            !skeletonLawKernel::waveTangent
            (
                skeletonLawTable_, eW, det(be), kcL
            )
        )
        {
            FatalErrorInFunction
                << "skeletonLaw valanisLandelTrilinear: be = " << be
                << " (eigenvalues " << evW << ", det " << det(be) << ") is "
                << "not SPD, so LT's wave tangent has no real stretches."
                << exit(FatalError);
        }
        return kcL;
    }
    return cmptMax(eigenValues(be));
}


Foam::scalar Foam::solvers::phaseClotAxisymmetricFluid::networkPole() const
{
    // S25 A (design 2026-10-10-S25 section 3A): ONE pole. Under LT the
    // constituents' incompressibility gives a <= 1 <=> J >= phi0, so the
    // pole is 1 and no fitted number ships with it.
    return skeletonLawLT_ ? 1.0 : phiSMax_;
}


Foam::symmTensor
Foam::solvers::phaseClotAxisymmetricFluid::skeletonLawDeviator
(
    const symmTensor& be
) const
{
    const vector evD(eigenValues(be));
    const double bD[6] = {be.xx(), be.xy(), be.xz(), be.yy(), be.yz(), be.zz()};
    const double eD[3] = {evD.x(), evD.y(), evD.z()};
    double mD[6];
    if
    (
        !skeletonLawKernel::deviator
        (
            skeletonLawTable_, bD, eD, det(be), mD
        )
    )
    {
        FatalErrorInFunction
            << "skeletonLaw valanisLandelTrilinear: be = " << be
            << " (eigenvalues " << evD << ", det " << det(be) << ") is not "
            << "SPD, so LT's deviator has no real stretches. be is checked "
            << "SPD after each update; this is a transport defect, not a "
            << "state the law may be evaluated at." << exit(FatalError);
    }
    return symmTensor(mD[0], mD[1], mD[2], mD[3], mD[4], mD[5]);
}


Foam::tmp<Foam::volSymmTensorField>
Foam::solvers::phaseClotAxisymmetricFluid::skeletonLawDeviatorField() const
{
    // dev(be)'s shape and patch types (calculated), then M_LT on every
    // cell and every patch face of be_.
    tmp<volSymmTensorField> tL
    (
        volSymmTensorField::New("skeletonLawDev", dev(be_))
    );
    symmTensorField& LI = tL.ref().primitiveFieldRef();
    const symmTensorField& beL = be_.primitiveField();
    forAll(LI, celli)
    {
        LI[celli] = skeletonLawDeviator(beL[celli]);
    }
    volSymmTensorField::Boundary& LB = tL.ref().boundaryFieldRef();
    forAll(LB, patchi)
    {
        const symmTensorField& beP = be_.boundaryField()[patchi];
        forAll(LB[patchi], facei)
        {
            LB[patchi][facei] = skeletonLawDeviator(beP[facei]);
        }
    }
    return tL;
}


void Foam::solvers::phaseClotAxisymmetricFluid::recordStressSplit(
    const symmTensorField& Mused,
    const label site
) const
{
    // S17split (design section 6; 11 objection 2): the reference is rebuilt
    // HERE from the raw fields -- the partition, be_, the fibre -- with the
    // kernel's splitDev, independently of the site. Exactly 0 when the site
    // used the split; a site that keeps S16's dev(be) reads (thP/S)|T
    // dev(n n) - dev(be)|/(1 - d) in every pending cell. S16w reads the
    // weight; this reads the tensor it multiplies.
    const scalarField& pI = thetaIntact_.primitiveField();
    const scalarField& pP = thetaPending_.primitiveField();
    // S17b A1: the fibre's carrier (theta_T under V1, P2 otherwise)
    const scalarField& pC = fibreCarrier().primitiveField();
    const scalarField& pB = thetaBroken_.primitiveField();
    const vectorField& nI = pendingAxis_.primitiveField();
    const scalarField& lI = pendingStretch_.primitiveField();
    const symmTensorField& beI = be_.primitiveField();
    scalar devMax = 0;
    forAll(Mused, celli)
    {
        scalar aI, aP, omd;
        fibreReleaseKernel::sharesCarrier(pI[celli], pP[celli], pB[celli], pC[celli], aI, aP, omd);
        const symmTensor d(dev(beI[celli]));
        // S25 A: under LT the reference is rebuilt on LT's deviator.
        const symmTensor dLaw
        (
            skeletonLawLT_ ? skeletonLawDeviator(beI[celli]) : d
        );
        const double devBe[6] =
            {dLaw.xx(), dLaw.xy(), dLaw.xz(), dLaw.yy(), dLaw.yz(), dLaw.zz()};
        const double n[3] = {nI[celli].x(), nI[celli].y(), nI[celli].z()};
        double m[6];
        fibreReleaseKernel::splitDev(aI, devBe, aP, lI[celli], n, omd, m);
        const symmTensor ref(m[0], m[1], m[2], m[3], m[4], m[5]);
        devMax = max(devMax, mag(Mused[celli] - ref));
    }
    reduce(devMax, maxOp<scalar>());
    s17SplitDevMax_[site] = max(s17SplitDevMax_[site], devMax);
    s17SplitSites_ |= (1 << site);
}


void Foam::solvers::phaseClotAxisymmetricFluid::recordStiffnessSplit(
    const scalarField& kUsed
) const
{
    // S17split, site 2 (the tangent bound): the stiffness per unit weight,
    // rebuilt HERE with the kernel's mixedStiffness.
    const scalarField& pI = thetaIntact_.primitiveField();
    const scalarField& pP = thetaPending_.primitiveField();
    // S17b A1: the fibre's carrier (theta_T under V1, P2 otherwise)
    const scalarField& pC = fibreCarrier().primitiveField();
    const scalarField& pB = thetaBroken_.primitiveField();
    const scalarField& lI = pendingStretch_.primitiveField();
    const symmTensorField& beI = be_.primitiveField();
    scalar devMax = 0;
    forAll(kUsed, celli)
    {
        scalar stiff = 1.0;
        {
            const vector ev(eigenValues(beI[celli]));
            const scalar b1 = cmptMax(ev);
            stiff = fibreReleaseKernel::stiffIntact
            (
                b1, ev.x() + ev.y() + ev.z() - b1
            );
        }
        if (skeletonLawLT_)
        {
            // S25 A: LT's tangent (design section 3A).
            stiff = intactTangentStiffness(beI[celli]);
        }
        scalar aI, aP, omd;
        fibreReleaseKernel::sharesCarrier(pI[celli], pP[celli], pB[celli], pC[celli], aI, aP, omd);
        const scalar ref =
            fibreReleaseKernel::mixedStiffness(aI, stiff, aP, lI[celli], omd);
        devMax = max(devMax, mag(kUsed[celli] - ref));
    }
    reduce(devMax, maxOp<scalar>());
    s17SplitDevMax_[2] = max(s17SplitDevMax_[2], devMax);
    s17SplitSites_ |= (1 << 2);
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::skeletonResidualDrag() const
{
    // THE SUPPORT OF Kres IS ANCHOR-DEPENDENT UNLESS residualChi IS SET.
    // (The full rationale is at its use in correctPressureTwoVelocity().)
    // ONE expression, so the block and S17 C's Vs predictor put the same
    // regularisation on the skeleton row's diagonal.
    const volScalarField chiReg
    (
        "chiReg",
        phaseIndicator()
    );
    return volScalarField::New
    (
        "Kres",
        residualChi_ > 0
      ? (
            dimensionedScalar
            (
                "rscale", dimless, residualPhiS_/residualChi_
            )
           *max
            (
                dimensionedScalar("rchi", dimless, residualChi_) - chiReg,
                dimensionedScalar("zero", dimless, 0)
            )/runTime.deltaT()
        )()
      : (
            max
            (
                dimensionedScalar("rphis", dimless, residualPhiS_) - phiSkel_,
                dimensionedScalar("zero", dimless, 0)
            )/runTime.deltaT()
        )()
    );
}


void Foam::solvers::phaseClotAxisymmetricFluid::writeS17SecondDifferences()
{
    // S17 (design section 6, change 5; docket 10): THE DISCRIMINATORS READ
    // FACE BY FACE -- the net outlet flux is blind to the 70-row mode, which
    // carries no net flux. Per step, A_n = ||phi_n - 2 phi_{n-1} +
    // phi_{n-2}||_2 over the outlet faces (the mixture flux), and the same on
    // the skeleton flux over the clot-surface faces (0.05 < ind_f < 0.95,
    // the CB pair's instrument). The two-step ratio A_n/A_{n-2} is formed by
    // the assessor, read only while A_n/||phi_n|| > 1e-10.
    const label outletPatch = mesh.boundaryMesh().findIndex("outlet");
    scalarField phiOut;
    if (outletPatch >= 0)
    {
        phiOut = phi_.boundaryField()[outletPatch];
    }
    const scalarField& phiS = phiSkelOwn_.primitiveField();
    const surfaceScalarField indF(fvc::interpolate(phaseIndicator()));
    const scalarField& indFI = indF.primitiveField();

    const bool valid = s17SecondDiffSteps_ >= 2;
    scalar aOut = 0, nOut = 0, aSurf = 0, nSurf = 0, sumOut = 0;
    // the net outlet flux, the start-up's face-flux reading (section 6)
    forAll(phiOut, facei)
    {
        sumOut += phiOut[facei];
    }
    label surfFaces = 0;
    if (valid)
    {
        forAll(phiOut, facei)
        {
            aOut += sqr
            (
                phiOut[facei] - 2.0*s17OutletPhi1_[facei]
              + s17OutletPhi2_[facei]
            );
            nOut += sqr(phiOut[facei]);
        }
        forAll(phiS, facei)
        {
            if (indFI[facei] > 0.05 && indFI[facei] < 0.95)
            {
                aSurf += sqr
                (
                    phiS[facei] - 2.0*s17SurfPhi1_[facei] + s17SurfPhi2_[facei]
                );
                nSurf += sqr(phiS[facei]);
                ++surfFaces;
            }
        }
    }
    // S17b (12 change 2): THE MODE'S SIGNATURE IS SIGN ALTERNATION, which the
    // norm cannot see. c_n = <a_n, a_{n-1}>/(|a_n| |a_{n-1}|), a_n the
    // face-wise second difference, over the outlet faces and over this
    // step's clot-surface faces; valid from the fourth step.
    const bool validC = s17SecondDiffSteps_ >= 3;
    scalar cOutDot = 0, cOutA = 0, cOutB = 0;
    scalar cSurfDot = 0, cSurfA = 0, cSurfB = 0;
    if (validC)
    {
        forAll(phiOut, facei)
        {
            const scalar an =
                phiOut[facei] - 2.0*s17OutletPhi1_[facei] + s17OutletPhi2_[facei];
            const scalar am =
                s17OutletPhi1_[facei] - 2.0*s17OutletPhi2_[facei]
              + s17OutletPhi3_[facei];
            cOutDot += an*am; cOutA += an*an; cOutB += am*am;
        }
        forAll(phiS, facei)
        {
            if (indFI[facei] > 0.05 && indFI[facei] < 0.95)
            {
                const scalar an =
                    phiS[facei] - 2.0*s17SurfPhi1_[facei] + s17SurfPhi2_[facei];
                const scalar am =
                    s17SurfPhi1_[facei] - 2.0*s17SurfPhi2_[facei]
                  + s17SurfPhi3_[facei];
                cSurfDot += an*am; cSurfA += an*an; cSurfB += am*am;
            }
        }
    }
    reduce(cOutDot, sumOp<scalar>());
    reduce(cOutA, sumOp<scalar>());
    reduce(cOutB, sumOp<scalar>());
    reduce(cSurfDot, sumOp<scalar>());
    reduce(cSurfA, sumOp<scalar>());
    reduce(cSurfB, sumOp<scalar>());
    const scalar cOut =
        cOutA > 0 && cOutB > 0 ? cOutDot/(sqrt(cOutA)*sqrt(cOutB)) : 0.0;
    const scalar cSurf =
        cSurfA > 0 && cSurfB > 0 ? cSurfDot/(sqrt(cSurfA)*sqrt(cSurfB)) : 0.0;
    reduce(aOut, sumOp<scalar>());
    reduce(nOut, sumOp<scalar>());
    reduce(sumOut, sumOp<scalar>());
    reduce(aSurf, sumOp<scalar>());
    reduce(nSurf, sumOp<scalar>());
    reduce(surfFaces, sumOp<label>());
    Info<< "PHASE_CLOT_S17_SECONDDIFF time=" << runTime.userTimeValue()
        << " step=" << runTime.timeIndex()
        << " valid=" << (valid ? 1 : 0)
        << " outletSecondDiff=" << sqrt(aOut)
        << " outletPhiNorm=" << sqrt(nOut)
        << " outletPhiSum=" << sumOut
        << " surfaceSecondDiff=" << sqrt(aSurf)
        << " surfacePhiNorm=" << sqrt(nSurf)
        << " surfaceFaces=" << surfFaces
        << " validCorr=" << (validC ? 1 : 0)
        << " outletSecondDiffCorr=" << cOut
        << " surfaceSecondDiffCorr=" << cSurf
        << endl;

    s17OutletPhi3_ = s17OutletPhi2_;
    s17SurfPhi3_ = s17SurfPhi2_;
    s17OutletPhi2_ = s17OutletPhi1_;
    s17OutletPhi1_ = phiOut;
    s17SurfPhi2_ = s17SurfPhi1_;
    s17SurfPhi1_ = phiS;
    ++s17SecondDiffSteps_;
}


Foam::tmp<Foam::surfaceScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::seriesFaceWeights
(
    const volScalarField& w
) const
{
    // Boundary faces keep the patch value, exactly as linear interpolation
    // gives them: a boundary face has one cell, so there is nothing to
    // combine in series.
    tmp<surfaceScalarField> tWf(fvc::interpolate(w));
    tWf.ref().rename("series(" + w.name() + ')');

    const scalarField& wIn = w.primitiveField();
    const scalar wMin = gMin(wIn);
    if (wMin < 0)
    {
        FatalErrorInFunction
            << "series face weight of " << w.name() << " needs a non-negative "
            << "field; min = " << wMin << ". A negative occupancy is a "
            << "transport defect, not something to bound here."
            << exit(FatalError);
    }

    scalarField& wf = tWf.ref().primitiveFieldRef();
    const scalarField& lambda = mesh.weights().primitiveField();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();

    // 1/w_f = l/w_P + (1 - l)/w_N, written without the reciprocals:
    // w_f = w_P*w_N/(l*w_N + (1 - l)*w_P). An empty side makes the product
    // zero; two empty sides make the denominator zero too, and the series
    // value of two empty half-cells is exactly zero.
    forAll(wf, facei)
    {
        const scalar a = wIn[own[facei]];
        const scalar b = wIn[nei[facei]];
        const scalar l = lambda[facei];
        const scalar den = l*b + (1.0 - l)*a;
        wf[facei] = den > 0 ? a*b/den : 0.0;
    }

    return tWf;
}


Foam::tmp<Foam::surfaceSymmTensorField>
Foam::solvers::phaseClotAxisymmetricFluid::skeletonFaceStress() const
{
    // The deviatoric weight exactly as elasticStress() builds it: the phase
    // indicator, floored at skeletonSolidFloor under the diffuse domain.
    // S14: the load-bearing occupancy is the INTACT network's -- ruptured
    // network is debris with no shear path, so its faces carry no deviatoric
    // traction (the series weight of a zero side is zero). intactFraction()
    // is 1 with the rupture model off, so the shipped weight is unchanged.
    volScalarField w("elasticSolidWeight", phaseIndicator()*intactFraction());
    {
        // S16 B: max(ind, floor)*(1 - d) -- the floor guards occupancy only.
        w = skeletonStressWeight();
    }
    // S23: T's weight uses d* (one weight with elasticStress()'s).
    applyTornStressWeight(w);
    recordStressWeight(w.primitiveField(), 0);
    // S23 rev. 4.1 F1': the weight as consumed (nothing writes w after).
    recordTornWeight(w.primitiveField(), 0);
    const volScalarField gD(sqr(1.0 - Dg_) + etaRes_);
    volSymmTensorField S("sigmaDevIntrinsic", gD*GField_*dev(be_));
    {
        // S17 A: the split per unit weight (S17split, site 0).
        const volSymmTensorField M(splitDeviatoricStress());
        recordStressSplit(M.primitiveField(), 0);
        S = gD*GField_*M;
    }

    tmp<surfaceSymmTensorField> tSigf
    (
        surfaceSymmTensorField::New
        (
            "sigmaElasticFace",
            seriesFaceWeights(w)*fvc::interpolate(S)
        )
    );

    return tSigf;
}


Foam::tmp<Foam::surfaceScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::skeletonElasticForceFlux() const
{
    // The face-stress divergence goes to faces the way the shipped force
    // does (cell divergence, then interpolated), so the Rhie-Chow pairing
    // with the pressure is unchanged; only the stress that crosses each face
    // is.
    tmp<surfaceScalarField> tf
    (
        surfaceScalarField::New
        (
            "fSigmaE",
            fvc::interpolate
            (
                fvc::surfaceIntegrateExtrapolate
                (
                    mesh.Sf() & skeletonFaceStress()
                )/rho_
            ) & mesh.Sf()
        )
    );

    // The isotropic network stress as a PHASE PRESSURE: -grad(Pi) on the
    // face with the compact two-point stencil, as multiphaseEuler applies
    // its phase pressure (momentumTransferSystem::Fs:
    // pPrimef*snGrad(alpha)*magSf). The interpolated cell divergence it
    // replaces is exactly blind to a 2-cell mode and 4x weak at 3 cells;
    // run 20260923-031942's compaction stripes sat at 2.2-3.9 cells.
    {
        tf.ref() -= fvc::snGrad(networkPartialPressure())*mesh.magSf()/rho_;
    }

    // S23 section 2.2: S22's fSigE above is computed first and left as it
    // is; then ONLY the faces with a torn side (c_f = 0) are overwritten.
    // While T is empty no face is visited: bitwise S22.
    correctTornFaces(tf.ref(), nullptr);

    return tf;
}


Foam::tmp<Foam::volVectorField>
Foam::solvers::phaseClotAxisymmetricFluid::skeletonElasticForceCell() const
{
    tmp<volVectorField> tc
    (
        volVectorField::New
        (
            "skeletonElasticForce",
            fvc::surfaceIntegrateExtrapolate
                (
                    mesh.Sf() & skeletonFaceStress()
                )/rho_
        )
    );

    // The compact face force's cell counterpart, reconstructed from the SAME
    // face values, as Vs_ is reconstructed from its face fluxes.
    {
        tc.ref() -=
            fvc::reconstruct
            (
                fvc::snGrad(networkPartialPressure())*mesh.magSf()
            )/rho_;
    }

    // S23 section 2.2, PARITY: the same torn-face correction, added as
    // reconstruct(delta f), delta f = (overwritten - S22) on the torn faces
    // and exactly 0 on every other face. Guarded by "any torn face", so
    // that with T empty nothing is added (no -0 + 0 can flip a bit). The
    // S22 face values are rebuilt from this function's own divergence (the
    // field skeletonElasticForceFlux() interpolates), before the pressure
    // term above was subtracted from tc.
    {
        boolList tornC;
        scalarField sTornC;
        if (tornNetworkCells(tornC, sTornC) > 0)
        {
            const volVectorField divSigmaE
            (
                "divSigmaE",
                fvc::surfaceIntegrateExtrapolate
                (
                    mesh.Sf() & skeletonFaceStress()
                )/rho_
            );
            surfaceScalarField fS22
            (
                "fSigmaES22",
                fvc::interpolate(divSigmaE) & mesh.Sf()
            );
            fS22 -= fvc::snGrad(networkPartialPressure())*mesh.magSf()/rho_;
            surfaceScalarField dF
            (
                IOobject
                (
                    "tornFaceDelta", mesh.time().name(), mesh,
                    IOobject::NO_READ, IOobject::NO_WRITE, false
                ),
                mesh,
                dimensionedScalar(fS22.dimensions(), 0)
            );
            if (correctTornFaces(fS22, &dF) > 0)
            {
                tc.ref() += fvc::reconstruct(dF);
            }
        }
    }

    return tc;
}


Foam::label Foam::solvers::phaseClotAxisymmetricFluid::tornNetworkCells(
    boolList& torn,
    scalarField& sTorn,
    scalarField* dStar,
    scalarField* indOut
) const
{
    // S23 rev. 4.1 section 13.2 item 1 (report 15: T as (d* = 1) emptied one
    // step after each tear, because the upwind partition refills pI from an
    // intact donor): T_P = isSkel_P AND d*_P >= (1 + d_p)/2, MAJORITY
    // membership, d_p the progressive law's ceiling from the law's own
    // expression (progressiveDamageCeiling()). Identical to rev. 3 at d* =
    // 1; never true on progressive-only damage (d* <= d_p). WITHOUT the
    // presence test (re-check 11 change 1: after A1 a torn cell with theta
    // <= 1e-4 reads present = false, and T must not hand it S22's push
    // back). d* is the rupture law's own expression, (thP + thB)/S with S =
    // thI + (thP + thB), 0 where S = 0; isSkel is the block's (isSkelB)
    // rule. s_X = (1 - d*_X) + d*_X ind_X is the packing scale (rule 3).
    const volScalarField indTorn(phaseIndicator());
    const scalarField& ind = indTorn.primitiveField();
    const scalarField& phiSI = phiSkel_.primitiveField();
    const scalarField& pI = thetaIntact_.primitiveField();
    const scalarField& pP = thetaPending_.primitiveField();
    const scalarField& pB = thetaBroken_.primitiveField();
    const scalar thrT =
        damagePartitionKernel::tornThreshold(progressiveDamageCeiling());
    torn.setSize(ind.size());
    sTorn.setSize(ind.size());
    if (dStar)
    {
        dStar->setSize(ind.size());
    }
    if (indOut)
    {
        *indOut = ind;
    }
    label nT = 0;
    forAll(ind, celli)
    {
        const scalar S = pI[celli] + (pP[celli] + pB[celli]);
        const scalar ds = S > 0 ? (pP[celli] + pB[celli])/S : 0.0;
        const bool isSkelC =
            residualChi_ > 0
          ? ind[celli] - 0.5*residualChi_ >= 0
          : phiSI[celli] - 0.5*residualPhiS_ >= 0;
        torn[celli] = damagePartitionKernel::tornMember(isSkelC, ds, thrT);
        sTorn[celli] = (1.0 - ds) + ds*ind[celli];
        if (dStar)
        {
            (*dStar)[celli] = ds;
        }
        if (torn[celli])
        {
            ++nT;
        }
    }
    return returnReduce(nT, sumOp<label>());
}


void Foam::solvers::phaseClotAxisymmetricFluid::applyTornStressWeight
(
    volScalarField& w
) const
{
    // S23 rev. 4.1 section 13.2 item 2 (re-check 11 change 3; report 16): at
    // the two builders of the one deviatoric weight (skeletonFaceStress(),
    // elasticStress()), T's weight is max(ind, floor)*(1 - d*), not the
    // relaxing (1 - d): otherwise the neighbour's divergence keeps the torn
    // face's stress for ~3 tau. Under majority T this is not 0 (w = 0 would
    // give a member's dense neighbours a free face, report 16); at d* = 1 it
    // is +0.0, rev. 3's value bit for bit. A T cell's own patch faces carry
    // the cell's value (d* follows the cell as d does). Every other reader
    // of intactFraction() keeps (1 - d).
    boolList torn;
    scalarField sTorn, dsT, indT;
    if (tornNetworkCells(torn, sTorn, &dsT, &indT) == 0)
    {
        return;
    }
    const scalar floorT = skeletonSolidFloor_;
    scalarField& wI = w.primitiveFieldRef();
    forAll(torn, celli)
    {
        if (torn[celli])
        {
            wI[celli] = damagePartitionKernel::tornStressWeight
            (
                indT[celli], floorT, dsT[celli]
            );
        }
    }
    forAll(w.boundaryField(), patchi)
    {
        fvPatchScalarField& wp = w.boundaryFieldRef()[patchi];
        if (wp.coupled())
        {
            continue;
        }
        const labelUList& fc = wp.patch().faceCells();
        forAll(wp, facei)
        {
            if (torn[fc[facei]])
            {
                wp[facei] = wI[fc[facei]];
            }
        }
    }
}


Foam::label Foam::solvers::phaseClotAxisymmetricFluid::correctTornFaces
(
    surfaceScalarField& fSigE,
    surfaceScalarField* delta
) const
{
    // S23 section 2.2 (report 09 section 1; re-check 11 change 6). On every
    // face with a T side, c_f = 0 (internal faces, and the active boundary
    // faces T owns), and only there:
    //   fSigE_f = c_f [interp(D).S]_f
    //           - s_f [snGrad(Pn)_f - (1 - c_f) dc_f (Pt_N - Pt_P)] |S_f|/rho
    // with c_f = 0 (T is binary), so the deviatoric part is gone (rule 1),
    // the two-point tension is removed from the corrected snGrad (rule 2)
    // and the packing push is scaled by s_f = min(s_P, s_N) (rule 3; s_f =
    // ind of the torn side on a T|intact face). Pt = aI min(Wp, 0), the
    // tension branch; dc_f the nonOrthDeltaCoeffs. On a boundary face the
    // N side is the patch value, as snGrad's own boundary form takes it.
    // Products only: no new denominator but rho.
    boolList torn;
    scalarField sX;
    const label nT = tornNetworkCells(torn, sX);
    s23TornCells_ = nT;
    s23TornFaces_ = 0;
    if (nT == 0)
    {
        return 0;
    }

    const volScalarField Pn(networkPartialPressure());
    const surfaceScalarField gPn(fvc::snGrad(Pn));
    const volScalarField pnU(networkPartialPressureUndamaged());
    const volScalarField Pt
    (
        "tensionBranch",
        intactShare()*min(pnU, dimensionedScalar(pnU.dimensions(), 0))
    );
    const scalarField& PtI = Pt.primitiveField();
    // S23 rev. 4.1 F1': the Pt this correction consumes, as PHASE_CLOT_S23
    // reads it (f1PtDev against aI min(Wp, 0), recomputed there).
    s23PtUsed_ = PtI;
    s23PtUsedIndex_ = mesh.time().timeIndex();
    const surfaceScalarField& dcF = mesh.nonOrthDeltaCoeffs();
    const scalarField& dc = dcF.primitiveField();
    const scalarField& magS = mesh.magSf().primitiveField();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const scalar rho = rho_.value();

    scalarField& f = fSigE.primitiveFieldRef();
    label nF = 0;
    forAll(own, facei)
    {
        const label o = own[facei];
        const label n = nei[facei];
        if (!(torn[o] || torn[n]))
        {
            continue;
        }
        const scalar sf = min(sX[o], sX[n]);
        const scalar fNew =
           -sf*(gPn[facei] - dc[facei]*(PtI[n] - PtI[o]))*magS[facei]/rho;
        if (delta)
        {
            delta->primitiveFieldRef()[facei] = fNew - f[facei];
        }
        f[facei] = fNew;
        ++nF;
    }
    forAll(fSigE.boundaryField(), patchi)
    {
        fvsPatchScalarField& fp = fSigE.boundaryFieldRef()[patchi];
        const labelUList& fc = mesh.boundary()[patchi].faceCells();
        const scalarField& dcp = dcF.boundaryField()[patchi];
        const scalarField& msp = mesh.magSf().boundaryField()[patchi];
        const scalarField& gp = gPn.boundaryField()[patchi];
        const scalarField& Ptp = Pt.boundaryField()[patchi];
        forAll(fp, facei)
        {
            const label o = fc[facei];
            if (!torn[o])
            {
                continue;
            }
            if (fp.coupled())
            {
                FatalErrorInFunction
                    << "S23: torn cell " << o << " owns a face of coupled "
                    << "patch " << mesh.boundary()[patchi].name() << ". The "
                    << "torn-face correction is serial and uncoupled (design "
                    << "2026-10-07-S23 section 5)." << exit(FatalError);
            }
            const scalar fNew =
               -sX[o]*(gp[facei] - dcp[facei]*(Ptp[facei] - PtI[o]))
               *msp[facei]/rho;
            if (delta)
            {
                delta->boundaryFieldRef()[patchi][facei] = fNew - fp[facei];
            }
            fp[facei] = fNew;
            ++nF;
        }
    }
    s23TornFaces_ = returnReduce(nF, sumOp<label>());
    return s23TornFaces_;
}


Foam::tmp<Foam::fvScalarMatrix>
Foam::solvers::phaseClotAxisymmetricFluid::phaseConservativeConvectionWith
(
    const surfaceScalarField& carrier,
    const volScalarField& f,
    const surfaceScalarField& weights
) const
{
    // ONE implementation of the operator, with the weights named by the
    // caller. Two copies differing only in which member they read is the
    // desynchronisation phaseWeights_ itself exists to prevent.
    return
        fv::gaussConvectionScheme<scalar>
        (
            mesh_,
            carrier,
            tmp<surfaceInterpolationScheme<scalar>>
            (
                new suppliedWeightsScheme(mesh_, weights)
            )
        ).fvmDiv(carrier, f);
}


// ---- S10 FIX 3: FLUX-CORRECTED TRANSPORT (Boris & Book 1973; Zalesak 1979)
//
// WHY THIS AND NOT A LIMITED SCHEME. A content is conserved and cannot be
// negative. Godunov: no LINEAR scheme is both second order and monotone.
// Implicit upwind is monotone (an M-matrix) and smears a translated profile
// at dx*(1-Co) of variance per unit distance -- 39.7 um^2 over 57 um,
// measured on run 20260922-104613. An implicit limited scheme is not an
// M-matrix and made content negative -- run 20260923-012704 died on step 2.
// FCT keeps the upwind predictor and adds only as much of the
// (linear - upwind) flux difference as keeps every cell within the range of
// its neighbours. It adds no flux of its own: it chooses, face by face,
// between the low- and high-order values of the SAME physical flux. This is
// the treatment OpenFOAM's multiphaseEuler gives every phase fraction
// (MULES, phaseSystemSolve.C:559-726), without compression.
//
// Priced before it was built, on the solver's own discretisation --
// scripts/price_content_transport.py, the real 31360-cell mesh, the seeded
// content, a returned flux, IMPLICIT assembly: 200 steps, zero negative cells,
// content conserved to 1.7e-14.

Foam::tmp<Foam::surfaceScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::fctAntidiffusiveFlux
(
    const surfaceScalarField& carrier,
    const volScalarField& f
) const
{
    tmp<surfaceScalarField> tA
    (
        surfaceScalarField::New
        (
            "fctAntidiffusiveFlux",
            mesh_,
            dimensionedScalar(carrier.dimensions()*f.dimensions(), 0)
        )
    );
    scalarField& Ai = tA.ref().primitiveFieldRef();

    const scalarField& wLin = mesh_.weights().primitiveField();
    const scalarField& Fi = carrier.primitiveField();
    const scalarField& fi = f.primitiveField();
    const labelUList& own = mesh_.owner();
    const labelUList& nei = mesh_.neighbour();

    // linear - upwind = F*(wLin - wUp)*(f_owner - f_nei). Boundary faces keep
    // zero: the correction never moves content through a boundary, so the
    // boundary flux is exactly the upwind flux the predictor already carried.
    forAll(own, facei)
    {
        const scalar wUp = pos0(Fi[facei]);
        Ai[facei] =
            Fi[facei]*(wLin[facei] - wUp)*(fi[own[facei]] - fi[nei[facei]]);
    }

    return tA;
}


Foam::tmp<Foam::surfaceScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::fctLimiter
(
    const volScalarField& fLow,
    const volScalarField& fOld,
    const surfaceScalarField& A
) const
{
    const labelUList& own = mesh_.owner();
    const labelUList& nei = mesh_.neighbour();
    const scalarField& fL = fLow.primitiveField();
    const scalarField& fO = fOld.primitiveField();
    const scalarField& Ai = A.primitiveField();
    const scalarField& V = mesh_.V();
    const scalar dt = runTime.deltaTValue();

    // Local bounds: min/max of the predictor AND the old value, over the cell
    // and its face neighbours (Zalesak 1979, eq. 17).
    scalarField hi(max(fL, fO));
    scalarField lo(min(fL, fO));
    forAll(own, facei)
    {
        const label o = own[facei];
        const label n = nei[facei];
        hi[o] = max(hi[o], max(fL[n], fO[n]));
        lo[o] = min(lo[o], min(fL[n], fO[n]));
        hi[n] = max(hi[n], max(fL[o], fO[o]));
        lo[n] = min(lo[n], min(fL[o], fO[o]));
    }

    // Antidiffusive flux INTO (Pp) and OUT OF (Pm) each cell. A > 0 leaves
    // the owner.
    scalarField Pp(fL.size(), 0.0);
    scalarField Pm(fL.size(), 0.0);
    forAll(own, facei)
    {
        const scalar a = Ai[facei];
        if (a > 0)
        {
            Pm[own[facei]] += a;
            Pp[nei[facei]] += a;
        }
        else if (a < 0)
        {
            Pp[own[facei]] -= a;
            Pm[nei[facei]] -= a;
        }
    }

    // FLOATING-POINT HONESTY -- three measured requirements, not tuning.
    // Each was found by scripts/price_content_transport.py failing without it
    // (worst negative cells over 200 steps, real mesh):
    //  1. V/dt FIRST, as one ratio (~1e-12, normal). (f - lo)*V underflows
    //     into denormals below content ~1e-290 and the budget loses ~12
    //     digits: cell 681 overspent by 7.4e-12 and went to -4.7e-306.
    //  2. a 1e-12 relative margin on the budget, for ordinary rounding in the
    //     sums: 1200 negatives at -8.8e-44 without it.
    //  3. a budget or a correction below the smallest NORMAL double is not
    //     represented to the precision the bound assumes, so it is not
    //     spent: 150 negatives at -3.9e-310 without it.
    // With all three: zero. None of them changes a representable result by
    // more than 1e-12 relative, and none of them touches the physics.
    const scalar tiny = std::numeric_limits<scalar>::min();
    const scalar keep = 1.0 - 1.0e-12;

    scalarField Rp(fL.size(), 1.0);
    scalarField Rm(fL.size(), 1.0);
    forAll(fL, celli)
    {
        const scalar VbyDt = V[celli]/dt;
        const scalar Qp = (hi[celli] - fL[celli])*VbyDt*keep;
        const scalar Qm = (fL[celli] - lo[celli])*VbyDt*keep;
        // S23 section 2.3 (OI-53): the quotient is formed only where it is
        // below 1 (P > Q >= tiny), so no Q/P can overflow (Q = 4, P =
        // DBL_MIN trapped S22 here); elsewhere 1 or 0 as S22 gave them, bit
        // for bit (report 09 section 4: 0 differences in 15.48M cases).
        Rp[celli] =
            Qp < tiny ? 0.0
          : Pp[celli] > Qp ? Qp/Pp[celli] : 1.0;
        Rm[celli] =
            Qm < tiny ? 0.0
          : Pm[celli] > Qm ? Qm/Pm[celli] : 1.0;
    }

    tmp<surfaceScalarField> tlambda
    (
        surfaceScalarField::New
        (
            "fctLambda", mesh_, dimensionedScalar(dimless, 0)
        )
    );
    scalarField& lam = tlambda.ref().primitiveFieldRef();
    // S20 (design 2026-10-05-S20 rev. 4 section 2.1): R1's exact-zero rule
    // (damagePartitionKernel.H limiterR1). A face where this field's flux is
    // EXACTLY 0 carries none of it, so it places no constraint on the shared
    // minimum: lambda = 1. A subnormal flux is still not spent (requirement
    // 3 above): lambda = 0. A field uniform 0 (thetaH) no longer vetoes every
    // face.
    forAll(own, facei)
    {
        const scalar a = Ai[facei];
        if (a == 0)
        {
            lam[facei] = 1.0;
        }
        else if (mag(a) < tiny)
        {
            lam[facei] = 0.0;
        }
        else if (a > 0)
        {
            lam[facei] = min(Rm[own[facei]], Rp[nei[facei]]);
        }
        else
        {
            lam[facei] = min(Rp[own[facei]], Rm[nei[facei]]);
        }
    }

    return tlambda;
}


Foam::tmp<Foam::surfaceScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::contentTransportedWeights() const
{
    tmp<surfaceScalarField> tw
    (
        new surfaceScalarField
        (
            "contentTransportedWeights", contentFaceWeights_()
        )
    );
    if (contentFctLambda_.valid())
    {
        // Effective weight of upwind + lambda*(linear - upwind). Convex in
        // [w_lin, w_up], and exactly contentFaceWeights_ where lambda = 0,
        // which includes every boundary face.
        tw.ref() +=
            contentFctLambda_()*(mesh_.weights() - contentFaceWeights_());
    }
    return tw;
}


void Foam::solvers::phaseClotAxisymmetricFluid::fctApply
(
    volScalarField& f,
    const surfaceScalarField& lambda,
    const surfaceScalarField& A
) const
{
    const labelUList& own = mesh_.owner();
    const labelUList& nei = mesh_.neighbour();
    const scalarField& lam = lambda.primitiveField();
    const scalarField& Ai = A.primitiveField();
    const scalarField& V = mesh_.V();
    const scalar dt = runTime.deltaTValue();

    // Conservative: one limited flux per face, added to one cell and taken
    // from the other.
    scalarField div(mesh_.nCells(), 0.0);
    forAll(own, facei)
    {
        const scalar la = lam[facei]*Ai[facei];
        div[own[facei]] += la;
        div[nei[facei]] -= la;
    }

    scalarField& fi = f.primitiveFieldRef();
    forAll(fi, celli)
    {
        fi[celli] -= dt/V[celli]*div[celli];
    }
    f.correctBoundaryConditions();
}


Foam::tmp<Foam::fvScalarMatrix>
Foam::solvers::phaseClotAxisymmetricFluid::plateletContentConvection
(
    const surfaceScalarField& carrier,
    const volScalarField& f
) const
{
    if (!contentFaceWeights_.valid())
    {
        FatalErrorInFunction
            << "phasePlateletFluxBasis 'content' but contentFaceWeights_ is"
            << " absent: the platelet content must inherit the face values"
            << " the theta equation transported with, and this step did not"
            << " assemble one. Advecting Wp with last step's weights, or"
            << " with the phase field's, is the mismatch this key removes."
            << exit(FatalError);
    }

    return phaseConservativeConvectionWith(carrier, f, contentFaceWeights_());
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::carrierDivergence
(
    const surfaceScalarField& carrier
) const
{
    // THE CLOT IS A VOLUME, NOT A LABEL, AND A NON-SOLENOIDAL CARRIER MOVES
    // THE TWO DIFFERENTLY.
    //
    // MEASURED, run 20260905-054441 child em_2vel: the clot went from 2620 to
    // 18797 cells and int chi dV from 1.35e-15 to 5.36e-15 -- 4.0x in 14 ms --
    // with c never leaving [-1,1]. PHASE_CLOT_MASS put 98% of dV/dt in
    // clotResidual, which working the budget against the solved equation
    // shows is identically
    //
    //     clotResidual == -0.5 * int c * div(carrier) dV
    //
    // Zero for a solenoidal carrier and only for one. Cumulative drift as a
    // fraction of the initial clot volume, every run on disk:
    //
    //     phi_                       (g04..g40, five children)   +0.00031
    //     phi_*(1-bondHold)          (em_free, em_freeG40)        +0.0048
    //     phiSkelOwn_*(1-bondHold)   (em_2vel, 09-04 and 09-05)   +1.12, +3.21
    //
    // THE CARRIER IS NOT AT FAULT AND CANNOT BE FIXED. Cogan & Guy 2010
    // (source 48), immediately after their Eq. 7: "the two fluids are
    // effectively compressible because their individual velocities are not
    // divergence-free." A per-phase flux is never solenoidal; only the mixture
    // is, and run 20260905-054441 confirms ours still is -- sum local
    // continuity error flat at 1.3e-14 from the first step to the last. So the
    // TRANSPORT must be corrected, not the velocity.
    //
    // THE EQUATION. Their Eq. 21 carries the volume fraction in conservative
    // divergence form. With phi_s proportional to chi = 0.5(1-c),
    //
    //     d(chi)/dt + div(chi v_s) = G
    //  => ddt(c) + div(c v_s)      = div(v_s) - 2G
    //
    // so the c equation gains this term on its right-hand side and drops the
    // bounded scheme's Sp -- see phaseConservativeConvection() above. The two
    // changes are one change and neither is correct alone.
    //
    // fvc::surfaceIntegrate, NOT fvc::div. fvc::div(surfaceScalarField) is
    // surfaceIntegrateExtrapolate, which differs at boundary faces.
    // gaussConvectionScheme and boundedConvectionScheme both build their Sp
    // from plain surfaceIntegrate, so the correction has to be the same
    // operator or it leaves a boundary residual that looks exactly like the
    // defect it is removing.
    //
    // COPIED INTO A FULL FIELD ON PURPOSE. fvc::surfaceIntegrate returns
    // tmp<VolInternalField>, a DimensionedField with no boundary. Multiplying
    // that by a volScalarField resolves to bare Field arithmetic and returns
    // tmp<Field<scalar>> -- DIMENSIONLESS, silently. The local syntax check
    // caught it as a New() overload failure on 2026-09-05; had the types
    // happened to line up it would have shipped a dimensionally meaningless
    // source term. zeroGradient because the term is a cell source and only its
    // internal field is ever read by the matrix.
    tmp<volScalarField> tDiv
    (
        new volScalarField
        (
            IOobject
            (
                "carrierDivergence", mesh_.time().name(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar("zero", dimless/dimTime, 0),
            zeroGradientFvPatchScalarField::typeName
        )
    );

    tDiv.ref().primitiveFieldRef() = fvc::surfaceIntegrate(carrier)();
    tDiv.ref().correctBoundaryConditions();
    return tDiv;
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::normalCellSize() const
{
    // h_n = 2V / sum_f |Sf . n|.  For a cube of side h with n along an axis
    // the sum is 2h^2 and this returns h exactly. Uses the CURRENT normal_,
    // so it is the extent in the direction the interface is actually facing.
    tmp<volScalarField> tH
    (
        new volScalarField
        (
            IOobject
            (
                "hNormal", mesh_.time().name(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar("zero", dimLength, 0)
        )
    );

    scalarField projected(mesh_.nCells(), 0.0);
    const surfaceVectorField& Sf = mesh_.Sf();
    const labelUList& own = mesh_.owner();
    const labelUList& nei = mesh_.neighbour();
    const vectorField& nIn = normal_.primitiveField();

    forAll(own, facei)
    {
        const scalar a = mag(Sf[facei] & nIn[own[facei]]);
        const scalar b = mag(Sf[facei] & nIn[nei[facei]]);
        projected[own[facei]] += a;
        projected[nei[facei]] += b;
    }

    forAll(Sf.boundaryField(), patchi)
    {
        const fvsPatchVectorField& SfP = Sf.boundaryField()[patchi];
        const labelUList& fc = mesh_.boundary()[patchi].faceCells();
        forAll(SfP, i)
        {
            projected[fc[i]] += mag(SfP[i] & nIn[fc[i]]);
        }
    }

    scalarField& hIn = tH.ref().primitiveFieldRef();
    const scalarField& VIn = mesh_.V();
    forAll(hIn, celli)
    {
        // normal_ is grad(c)/max(|grad c|, gradFloor), so it is a UNIT vector
        // only where |grad c| clears the floor -- i.e. at the interface, where
        // |grad c| ~ 1/epsilonC ~ 3e5 against a floor of 667. Away from the
        // interface it shrinks towards zero, and 2V/sum|Sf.n| would then
        // return a divergent "cell size" for a direction that does not exist.
        //
        // Multiplying by |n| makes the quotient 2V/sum|Sf.nHat| identically,
        // which is the true cell extent wherever a normal exists and is
        // insensitive to how short the stored vector is. Where no normal
        // exists at all the size is reported as zero, which propagates to
        // Da = 0 and kEff = k: no interface, no depletion layer, nothing to
        // correct. That is the physically right fallback as well as the
        // numerically safe one.
        const scalar magN = mag(nIn[celli]);
        hIn[celli] =
            (projected[celli] > vSmall)
          ? 2.0*VIn[celli]*magN/projected[celli]
          : 0.0;
    }
    tH.ref().correctBoundaryConditions();

    return tH;
}


void Foam::solvers::phaseClotAxisymmetricFluid::updateDerivedFields()
{
    const phaseClotScopeTimer costGuard_(tDerived_);

    // S6c-2 -- q IS THE CONTENT FORM, AND THERE IS NO SECOND ONE LEFT.
    //
    // Until S6c-2 both forms were built every step and PHASE_CLOT_QBASIS
    // reported dqMax between them. That instrument -- like the Stage-1
    // equivalence test, the S1' delta comparison and the S6c-1 normal
    // comparison -- was a c-against-theta measurement. With c deleted it
    // would compare a field against itself and read identically zero, which
    // is the unfailable gate retired as G3 of the phase-volume spec. All
    // five are retired rather than migrated.
    //
    // dqMax, equivMax and 0.5*|c - (1 - 2*ind)| were THE SAME NUMBER, and it
    // ran 3.3e-16 -> 0.0158 over 4e-04 s in run 20260920-060033. c and theta
    // were the same equation and had measurably separated; that drift is why
    // the deletion was worth doing rather than merely permitted. See
    // research/designs/2026-09-20-S6c-2-delete-c.md.
    //
    // NOT CLAMPED, exactly as q = 0.5*(1 + c) was not. A deletion must not
    // quietly add a bound.
    q_ =
        scalar(1)
      - (thetaC_ + thetaH_)
       /dimensionedScalar("thetaRef", dimless, skeletonVolumeFraction_);
    q_.correctBoundaryConditions();

    // S25 B (R1): Wp := q + e WHEREVER q IS FORMED. A change of q that is
    // not an event -- the content solve, a corrector roll-back, a partition
    // re-expression, the detachment release -- moves Wp with it and leaves e
    // unchanged; so after the release Wp is q + e before wGrowth below or a
    // write reads it (physics 02 Q3). The inlet keeps its fixedValue
    // (q = 1, Wp = 1, asserted at construction).
    Wp_.primitiveFieldRef() = q_.primitiveField() + e_.primitiveField();
    Wp_.correctBoundaryConditions();

    // THE INTERFACE NORMAL. S6c-1 built both forms and batch 42 returned
    // dAdhesionSpeedMax = 0 at every one of 272 steps, which is what
    // authorised this deletion. phaseNormalBasis 'legacy' is REFUSED at
    // construction now that grad(c) does not exist.
    //
    // THE FACTOR OF TWO IS NOT COSMETIC, and it is kept explicit. The legacy
    // form was grad(c)/max(|grad c|, gradFloor) with c = 1 - 2*ind, so
    // grad(c) = -2*grad(ind) exactly. normal_ is a unit vector only where the
    // gradient clears gradFloor_; below the floor it is grad/gradFloor_ and
    // the magnitude carries the factor. gradFloor_ is 666.667 1/m on the main
    // case, which is not small.
    //
    // Written in the same association as the S6c-1 content branch -- one
    // fvc::grad, then the multiply -- because em47 must reproduce em45's
    // normal bit-for-bit and floating-point multiplication does not commute
    // with a re-association of the gradient.
    const volVectorField gradNContent
    (
       -2.0
       *fvc::grad
        (
            (thetaC_ + thetaH_)
           /dimensionedScalar("thetaRef", dimless, skeletonVolumeFraction_)
        )
    );
    const volScalarField magGradN(mag(gradNContent));

    normal_ = gradNContent/max(magGradN, gradFloor_);

    // phaseGradient_, REBUILT ON theta AND NOT DELETED. |grad c| is
    // |grad(-2*ind)| = magGradN exactly, because c = 1 - 2*ind, so this is
    // the same expression the field always carried -- only its input field
    // changed. It is 2*delta_Gamma and interfaceDelta_ is delta_Gamma;
    // neither may be substituted for the other.
    //
    // KEPT BECAUSE OF A CONSUMER THE SCOPING TABLE MISSED. That table listed
    // frontSource and chiSource, both now on refused key values, and
    // concluded the field was dead. clotGrowthFront reads it every step for
    // PHASE_CLOT_GROWTH_STOICH, which is the growth/uptake identity and the
    // number every archived run is compared on.
    phaseGradient_ = sqrt(sqr(magGradN) + sqr(gradFloor_)) - gradFloor_;

    // fvc::grad, not a finite difference over cell centres -- a hand
    // reconstruction of a gradient in this project was wrong by 2x once
    // already, and the solver's own operator is the authority.
    //
    // ASSIGNED HERE, NOT IN writeBalanceDiagnostic(). It used to live in the
    // diagnostic, which runs in postSolve() AFTER solvePlatelets() has
    // committed the accepted transfer. A consumer reading it there would have
    // been localised on the PREVIOUS step's theta. The diagnostic now reads
    // this field and additionally reports deltaThetaIntPost, its own
    // recomputation at postSolve.
    //
    // deltaThetaIntPost MEASURES NOTHING, and this is OI-11. Both it and
    // deltaThetaInt are postSolve values -- the last updateDerivedFields()
    // call of the step writes deltaTheta_, and writeBalanceDiagnostic() runs
    // after it -- so they were bit-identical in all 272 records of run
    // 20260917-003709. The number that does the job is
    // deltaThetaIntAtTransfer, snapshotted in solvePlateletContent() where
    // bindRate is built, i.e. where the reaction actually reads the geometry.
    // All three are reported; only the third can differ from the other two.
    //
    // Left at zero when there is no content model or theta_ref <= 0. Under
    // phaseDeltaBasis 'content' both of those configurations are refused at
    // construction, so a zero delta can never reach a consumer.
    //
    // ONE CALL IS EARLIER THAN THE SEED. The constructor calls this before it
    // seeds thetaC_/thetaH_ from the legacy state, so on that call
    // deltaTheta_ is zero. Nothing between there and the first
    // prePredictor() -- which calls this again, after the seed -- reads
    // interfaceDelta_.
    if (skeletonVolumeFraction_ > 0)
    {
        deltaTheta_ =
            mag(fvc::grad(thetaC_ + thetaH_))
           /dimensionedScalar("thetaRef", dimless, skeletonVolumeFraction_);
    }

    // THE SELECTION. interfaceDelta_ is one physical object -- the unit
    // surface delta delta_Gamma -- so every consumer of it moves together:
    // the free-platelet transfer's bindRate, the bound-platelet uptake, and
    // surfaceLossCoeff_ below.
    //
    // ONE ADMISSIBLE VALUE SINCE S6c-2. deltaC_ is deleted, so
    // phaseDeltaBasis 'legacy' is refused at construction. The key survives
    // for one release so that a case file naming the dead value is REFUSED
    // rather than silently given the surviving one; S7 retires it.
    interfaceDelta_ = deltaTheta_;

    phaseGradient_.correctBoundaryConditions();

    const volSymmTensorField strain(symm(fvc::grad(U_)));
    const volVectorField viscousTraction(2*rho_*nu_*(strain & normal_));
    const volVectorField tangentialTraction
    (
        viscousTraction - normal_*(normal_ & viscousTraction)
    );
    shearPa_ = mag(tangentialTraction);

    // Local shear rate from the tangential viscous traction magnitude.
    // mu = rho*nu; the floor only guards a pathological zero-viscosity
    // dictionary, it never activates for a physical fluid.
    const dimensionedScalar mu
    (
        max(rho_*nu_, dimensionedScalar("muFloor", dimPressure*dimTime, small))
    );
    shearRate_ = shearPa_/mu;

    const dimensionedScalar zeroDimless("zero", dimless, 0);
    const dimensionedScalar oneDimless("one", dimless, 1);

    {
        // Shankar, Zhang, Sinno & Diamond (2022), PLoS Comput Biol 18,
        // e1009850, eq. (9). The enhancement is identically one below
        // gammaLow, ramps linearly to enhancementMax at gammaHigh, and
        // saturates above it. k1 is unused in this mode.
        const dimensionedScalar gammaSpan
        (
            max
            (
                gammaHigh_ - gammaLow_,
                dimensionedScalar("span", dimless/dimTime, small)
            )
        );

        adhesionEnhancement_ =
            oneDimless
          + (enhancementMax_ - 1.0)
           *min(max((shearRate_ - gammaLow_)/gammaSpan, zeroDimless), oneDimless);

        adhesionSpeed_ = max
        (
            k0_*adhesionEnhancement_,
            dimensionedScalar("zero", dimVelocity, 0)
        );
    }

    // ---- THE S6c-1 INSTRUMENT WAS RETIRED WITH c IN S6c-2 ---------------
    //
    // PHASE_CLOT_NORMAL_BASIS compared the legacy normal grad(c)/|grad c|
    // against the content normal -2*grad(ind)/|.| and reported dNormalMax,
    // dShearPaMax and dAdhesionSpeedMax. Both operands were needed; one of
    // them is deleted, and rebuilding it from theta would compare a field
    // against itself -- dNormalMax identically zero, the G3 defect.
    //
    // WHAT IT ANSWERED, run 20260920-060033, before it was retired:
    //   dAdhesionSpeedMax  = 0 exactly, all 272 steps, both children
    //   cellsAboveGammaLow = 0, peak shear 795.2 1/s against gammaLow 3000
    //   dNormalMax         = 0.241 in 26516 of 31360 cells (~14 degrees)
    //   selfCheck          = 0 on all 1090 lines
    //
    // The last two are the finding and the first is the permission. The two
    // normals are ALGEBRAICALLY IDENTICAL -- grad(c) = -2*grad(ind) exactly
    // under c = 1 - 2*ind -- and they disagreed by 14 degrees in 85% of
    // cells, because the premise had stopped holding: c and 1 - 2*ind had
    // drifted apart by 0.0317. Only the enhancement clamp's 3.8x margin
    // kept that out of the growth rate.
    //
    // Full record: research/runs/20260920-060033-s6c1-normal-basis.md.

    // ---- kEff: THE ONE ADHESION SPEED BOTH SIDES USE --------------------
    //
    // The platelet equation removes  delta_Gamma * kEff * w  and the phase
    // equation deposits  (alpha/D) * kEff * w * delta_Gamma.  Because both
    // read the SAME field, the clot volume created per platelet consumed is
    // exactly alpha/D -- a constant with a physical meaning, not a number
    // that drifts with the mesh. Any depletion model has to enter here, once,
    // or it would silently break that identity.
    {
        adhesionSpeedEff_ = adhesionSpeed_;
    }
    adhesionSpeedEff_.correctBoundaryConditions();

    // ---- THE GROWTH CLOSURE ---------------------------------------------
    // research/designs/AM2020_growth_closure_reaction_form.md
    //
    // Capek (2020) states the growth law twice, and the two statements are
    // for DIFFERENT formulations:
    //
    //   (3.5) sharp interface:    u_i = alpha*grad(w)
    //   (4.6) diffuse interface:  partial_t c - div(M grad mu)
    //                                = alpha D^-1 k w |grad phi_eps(c)|
    //
    // They agree only where the Robin condition (3.3), D n.grad(w) = k w,
    // holds pointwise. The paper made that substitution deliberately on the
    // way from sharp to diffuse, because in a diffuse-interface method there
    // is no surface at which to evaluate n.grad(w).
    //
    // This solver had been running (3.5) inside a diffuse-interface method.
    // MEASURED on run 20260901-200203: the Robin residual D dn(w)/(k w),
    // which is 1 by construction of the model and in which alpha cancels,
    // came out at a median of 0.092 (g04) and 0.072 (g40), ranging over three
    // decades across the interface. The reason is resolution, and it was
    // already recorded in AM2020_mural_detachment_parameters.md Sec.5: the
    // depletion layer is D/k, which is 3.04 cells at k0 and 0.94 cells at
    // g40's measured shear enhancement. A gradient across less than one cell
    // is set by the discretisation, so alpha absorbed the error -- and the
    // calibration confirms it exactly: the FIRST-TRIAL alpha in run
    // 20260810-035337 was 8.125e-14, and D*Vn_target/k0 (this closure, w=1)
    // is 8.152e-14. They agree to 0.33%. The 6.81x "correction" applied
    // afterwards is the gradient closure's under-read at that one operating
    // point, and it does not transfer to any other.
    //
    // WHY THE REACTION FORM IS THE RIGHT ONE, not merely the cheaper one:
    //   - Platelets consumed per unit time is  INT delta_Gamma kEff w dV,
    //     which is  INT kEff w dA  because delta_Gamma = |grad c|/2 integrates
    //     to one across a tanh interface. Clot volume created is INT Vn dA.
    //     Setting Vn = (alpha/D) kEff w makes the ratio exactly alpha/D
    //     POINTWISE: the clot can no longer grow at a rate unrelated to the
    //     platelets it eats. Under the gradient closure that ratio was
    //     measured at 0.034-0.10 of alpha/D and DRIFTED BY 3x over 30 ms.
    //   - It contains no gradient, so no boundary layer has to be resolved
    //     and nothing scales as 1/epsilonC under refinement -- which is the
    //     objection that killed Vn = alpha*(n & grad W) in Sec.4.2 of the
    //     derivation, and this form is immune to it for the same reason.
    //   - It removes the last of the three sites where 1/q was reconstructed
    //     inside a differential operator. Every defect found in this model has
    //     been at one of those three.
    //
    // THE DECLARED LIMIT. Vn = (alpha/D) k w is the REACTION-LIMITED branch.
    // It over-reads when Da = k h / D > 1. That is not hidden: Da is measured
    // and printed every step (PHASE_CLOT_DAMKOHLER), and
    // interfaceDepletionModel seriesResistance restores the other branch
    // through kEff above, for both the sink and the growth at once.
    {
        // The same operand the adhesion sink uses, TODAY'S
        // Wp/max(|q|, qFloor) with Wp = q + e, in every cell (S25 R1;
        // design rev. 2.1 amendment 10: the band law is WITHDRAWN, Alireza
        // 2026-10-10). NOT a gradient of it. It is NOT
        // the diffusion operand or the written w_ (OI-62): those are C6,
        // w = W/q signed, so at Wp = q <= -qFloor they read w = +1 where
        // this and the sink read -1. The law at q < 0 is OI-63, owed; no
        // record may take the written w as the w growth used at q < 0.
        volScalarField wGrowth
        (
            IOobject
            (
                "wGrowth", mesh_.time().name(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            w_
        );
        {
            // mag(q_), NOT q_. THE FLOOR WAS AMPLIFYING BY EXACTLY 1e+06.
            //
            // qFloor guards a division against a SMALL q. It does nothing
            // about a NEGATIVE one: max(-0.056, 1e-06) returns the floor, so
            // a Wp that went negative together with q is divided by +1e-06
            // instead of by -0.056 -- the sign flips and the magnitude is
            // multiplied by 5.6e+04.
            //
            // MEASURED, run 20260908-010836 at 7.9 ms. c undershoots below
            // -1 in 230 cells, so q = (c+1)/2 goes negative there, to
            // -0.0948. In every one of those 230 cells w/Wp is EXACTLY
            // 1.000000e+06, and wMin = -30626.8 is precisely Wp = -0.030627
            // divided by the floor. In the healthy cells w/Wp never exceeds
            // 8.5e+04. wMin = -3.1e+04 was never an instability; it was this
            // division. VnMagMax = 1.15 m/s is the same artefact one step
            // downstream -- all 30 cells with |Vn| > 0.01 have q < 0.
            //
            // RETRACTED 2026-09-08: the feedback SIGN below is wrong. 29 of the 30
    // cells with |Vn| > 0.01 have Vn < 0, and -Vn*phaseGradient > 0 RAISES c.
    // The amplified cells were pushing c back toward the bound, not away.
    // The 1e+06 amplification arithmetic is correct; its causal role is not.
    //
    // AND IT CLOSES A FEEDBACK LOOP. Vn built from that w drives the
            // -Vn*phaseGradient source in the c equation, and the growth
            // source in the undershoot cells measures 18x the clot median
            // (13.1 against 0.71). Deeper undershoot -> more negative q ->
            // larger amplification -> more growth. mag() breaks it: the
            // divisor tracks |q| instead of collapsing to the floor, so the
            // amplification is bounded by 1/|q| and the sign relationship
            // between Wp and q survives. At the worst cell w becomes -0.547
            // rather than -30626.8.
            //
            // WHERE q >= 0 THIS IS BIT-IDENTICAL. mag() is a no-op on a
            // non-negative field, so no earlier result moves.
            //
            // This is the amplifier, not the root. The root is that c has no
            // mechanism holding it in [-1, 1] under a growth source with
            // mobility = 0; see PHASE_CLOT_PHASE_BOUNDS qNegCells.
            wGrowth = (q_ + e_)/max(mag(q_), qFloor_);
        }

        // THE STOICHIOMETRIC RATIO, and why it is not simply alpha_/D_.
        //
        // Run 20260902-141852 killed FOUR children with SIGFPE at
        // construction -- gate_a_tol, gate_a_floor, gate_a_nodiff,
        // gate_a_linear -- all of which set D = 0. The gradient closure this
        // replaced was `alpha_*(n & grad w)` and never divided by D, so those
        // children had always run. The reaction form introduced the division
        // and 0.0/0.0 traps under FOAM_SIGFPE before the first timestep.
        //
        // The two cases are NOT the same and must not be collapsed into a
        // max(D_, vSmall) guard, which would silently return a huge finite
        // ratio for a case that is actually ill-posed:
        //
        //   alpha == 0  -> no growth was requested. Vn is identically zero
        //                  whatever D is, so the ratio is 0 and nothing is
        //                  divided. Exact, and it is what these four children
        //                  mean: alpha = k = mobility = 0, w == 1 exact.
        //   alpha != 0
        //   and D == 0  -> genuinely ill-posed. Capek's Robin condition
        //                  D dn(w) = k w degenerates and the interface speed
        //                  has no finite value. That deserves a readable
        //                  refusal, not a stack trace and not a fabricated
        //                  number.
        dimensionedScalar stoichRatio("stoichRatio", dimless, 0.0);
        if (alpha_.value() != 0.0)
        {
            if (D_.value() == 0.0)
            {
                FatalErrorInFunction
                    << "growthClosure 'reaction' needs D > 0: it computes"
                    << " Vn = (alpha/D)*kEff*w, and D = 0 with alpha = "
                    << alpha_.value() << " makes that ratio infinite." << nl
                    << "Capek's Robin condition D n.grad(w) = k w degenerates"
                    << " at D = 0, so the interface speed is not defined."
                    << nl
                    << "Set D > 0, or set alpha = 0 if this case is meant to"
                    << " have no growth, or use growthClosure 'gradient'."
                    << exit(FatalError);
            }
            stoichRatio = alpha_/D_;
        }

        Vn_ = stoichRatio*adhesionSpeedEff_*wGrowth;
    }

    // ---- SURFACE EROSION: THE ONLY ROUTE BY WHICH THIS CLOT CAN LET GO ----
    //
    // Vn_ is the interface normal speed and enters the phase equation as
    // -Vn_*phaseGradient_, so a POSITIVE Vn_ advances clot. Erosion is a
    // NEGATIVE contribution to the same term: it needs no new operator, no new
    // equation and no new carrier, and it retracts the interface exactly where
    // growth advances it. Applied HERE, before surfaceLossCoeff_ consumes Vn_
    // below, so the surface loss sees the NET interface speed.
    //
    // The full argument -- that bulk fracture is physically unreachable at
    // this stiffness and size, that wall peel arrests, and that the
    // connectivity release criterion cannot sever a connected tail -- is on
    // clotErosionRate_ in the header, with the measurements and both source
    // anchors for the threshold. Default 0 = OFF, so every result obtained
    // before 2026-09-08 reproduces byte for byte.
    // AN INDICATOR-SATURATION CAP ON GROWTH. NOT A ROOT FIX -- SEE BELOW.
    //
    // RETRACTED 2026-09-08, after independent review. This block was shipped
    // as "a cell with no blood left cannot grow more clot". That is FALSE:
    // at c = -1, phi_s = 0.45*chi = 0.45 and phi_f = 0.55, so the cell is 55%
    // fluid. chi is an INDICATOR, not a volume fraction, and this cap is a
    // numerical bound on indicator saturation, not missing physics.
    //
    // AND IT PROVABLY CANNOT BOUND c. With chi = 1, div(v_s) = -100 1/s and
    // G = 0 EXACTLY, implicit Euler gives chi = 1/(1 + dt*div v) =
    // 1.000149276 and c = -1.000298552. Compression alone leaves the bound
    // with zero growth, so a growth-only cap cannot be the root fix.
    // Forward invariance of chi <= 1 requires G - div(v_s) <= 0 at chi = 1,
    // which is a constraint on the CARRIER, not on the source.
    //
    // Kept as an instrumented, measurable cap. This is the GROWTH half,
    // applied before erosion because it must scale adhesionSpeedEff_ too.
    //
    // WHY BOTH FIELDS. Under growthClosure 'reaction',
    //     Vn = (alpha/D)*adhesionSpeedEff*w
    // and the platelet uptake that pays for that growth is
    //     plateletUptake = int interfaceDelta*adhesionSpeedEff*w dV
    // which does NOT read Vn_. Clamping Vn_ alone would keep consuming
    // platelets while producing no clot, and PHASE_CLOT_GROWTH_STOICH
    // stoichRatio -- currently 1.6277 against a target of 1.6304 -- would
    // drift for a reason nothing in the log could explain. Scaling both by
    // the same per-cell factor leaves clotGrowth/plateletUptake EXACTLY
    // where it was, because Vn is linear in adhesionSpeedEff.
    //
    // The scaling is gated on the reaction closure: under 'gradient', Vn_ is
    // not proportional to adhesionSpeedEff_ and scaling it would be wrong.
    {
        const dimensionedScalar dtVnG("dtVnG", dimTime, runTime.deltaTValue());
        const volScalarField pgSafeG
        (
            max
            (
                phaseGradient_,
                dimensionedScalar
                (
                    "pgSmall", phaseGradient_.dimensions(), small
                )
            )
        );
        // 1 + c = 2*(1 - ind) = 2*q, exactly, under c = 1 - 2*ind. The
        // headroom is unchanged; only the representation is. This counter is
        // MEASURED and not applied while vnCapEnabled is off, so its value
        // moves with the c/theta drift and is on S6c-2's may-move list.
        const volScalarField VnGrowthMax
        (
            max(scalar(2)*q_, dimensionedScalar("zero", dimless, 0))
           /(dtVnG*pgSafeG)
        );


        // MEASUREMENT IS NOT GATED ON APPLICATION. Until 2026-09-08 both the
        // counter and the scaling sat inside `if (capOn && ...)`, so with the
        // switch OFF the counters read zero no matter how many cells would
        // have bound -- and the shipped comment claimed they were "the only
        // measure of how hard the source would have to be withheld", which
        // was then false for the only configuration anyone would run. Codex
        // caught it. The would-limit numbers are now computed on every cell
        // that exceeds the bound, and only the MUTATION is switched.
        //
        // wouldLimit / wouldWorstFactor / wouldMaxExcessMps describe the
        // state; growthLimited / worstFactor describe what was done to it.
        // With capOn they agree; with capOn off the second pair is zero and
        // the first still measures. No volume-rate estimate is emitted: Vn is
        // a normal speed and converting it to a withheld volume needs an
        // interface area this loop does not have.
        label growthLimited = 0;
        scalar worstFactor = 1;
        label growthWouldLimit = 0;
        scalar wouldWorstFactor = 1;
        scalar wouldMaxExcess = 0;

        scalarField& VnIn = Vn_.primitiveFieldRef();
        scalarField& aseIn = adhesionSpeedEff_.primitiveFieldRef();
        const scalarField& gMaxIn = VnGrowthMax.primitiveField();

        forAll(VnIn, celli)
        {
            if (VnIn[celli] <= gMaxIn[celli])
            {
                continue;
            }

            const scalar f =
                (VnIn[celli] > vSmall)
              ? max(gMaxIn[celli], 0.0)/VnIn[celli]
              : 0.0;

            ++growthWouldLimit;
            wouldWorstFactor = min(wouldWorstFactor, f);
            wouldMaxExcess = max(wouldMaxExcess, VnIn[celli] - gMaxIn[celli]);

            {
                continue;
            }

            ++growthLimited;
            worstFactor = min(worstFactor, f);

            VnIn[celli] *= f;
            {
                aseIn[celli] *= f;
            }
        }

        Vn_.correctBoundaryConditions();
        adhesionSpeedEff_.correctBoundaryConditions();

        reduce(growthLimited, sumOp<label>());
        reduce(worstFactor, minOp<scalar>());
        reduce(growthWouldLimit, sumOp<label>());
        reduce(wouldWorstFactor, minOp<scalar>());
        reduce(wouldMaxExcess, maxOp<scalar>());

        // The count is per CELL-EVALUATION, not per cell-timestep: this block
        // runs four times per physical timestep plus twice at startup, so a
        // run of N steps emits 4N+2 records. Batch 25's 2,505,301 is 80,398
        // records, not 20,099. Do not divide by four either -- different
        // evaluations see different states and different cells.
        //
        // perStepEvaluationsAssumed IS A HARD-CODED DECLARATION, NOT A
        // MEASUREMENT. The 4 was measured for Batch 25's control path by
        // counting records against steps (80,398 = 4*20,099 + 2); it is
        // printed here so a reader knows the intended stride, but nothing
        // verifies it at runtime and a different corrector count, a different
        // solve path or an added updateDerivedFields() call would silently
        // falsify it. Codex, 2026-09-08. If a future analysis depends on the
        // stride, emit an actual step/corrector/stage identifier per record
        // rather than trusting this number.
        Info<< "PHASE_CLOT_VN_BOUND_GROWTH time=" << runTime.userTimeValue()
            << " capOn=" << Switch(false)
            << " growthLimited=" << growthLimited
            << " worstFactor=" << worstFactor
            << " growthWouldLimit=" << growthWouldLimit
            << " wouldWorstFactor=" << wouldWorstFactor
            << " wouldMaxExcessMps=" << wouldMaxExcess
            << " uptakeScaled=" << Switch(true)
            << " perStepEvaluationsAssumed=4"
            << endl;
    }

    if (clotErosionRate_ > 0)
    {
        volScalarField Ve
        (
            "clotErosionSpeed",
            dimensionedScalar
            (
                "kEro", dimVelocity/dimPressure, clotErosionRate_
            )
           *max
            (
                shearPa_
              - dimensionedScalar("tauEro", dimPressure, clotErosionTauPa_),
                dimensionedScalar("zeroPa", dimPressure, 0)
            )
        );

        // THE CORE MAY NOT ERODE. Dr He Li, 2026-08-25: the core "does not
        // slide or detach or rupture", while the shell "can deform, rupture,
        // and detach".
        //
        // CORRECTION, Codex 2026-09-08, VERIFIED IN SOURCE. This comment used
        // to claim the gate below was "on the SAME switch
        // releaseDetachedMaterial() uses, so the two cannot disagree about
        // which material is eligible to leave". That was false, and it is
        // exactly the failure mode of asserting a guard in a comment that no
        // code implements. There are THREE separate mechanisms by which
        // material leaves, and they do NOT share an eligibility rule:
        //
        //   1. updateReleasedIndicator() -- skips core cells when
        //      releaseCoreExempt_ is on. Sharp test, shell < 0.5.
        //   2. this erosion gate -- Ve *= shell_ is a SOFT WEIGHT, not an
        //      exclusion. A cell at shell = 0.3 still erodes at 30% rate. At
        //      Batch 25's final write the minimum label was 0.221929581462,
        //      so NO cell had a zero multiplier.
        //   3. releaseDetachedMaterial() -- converts any cell with
        //      Dg > DgDetach and c < 0 to c = 1, with NO core check at all.
        //      Nothing exempts the core from damage-driven conversion.
        //
        // Mechanism 3 is an open gap against the "does not detach" clause and
        // is NOT fixed here: adding an exemption changes physics and is
        // pending Dr He Li's ruling on what attachment actually means (Q14).
        // Do not close it by editing this comment.
        if (coreShell_ && releaseCoreExempt_)
        {
            Ve *= shell_;
        }

        Vn_ -= Ve;

        label eroding = 0;
        const scalarField& VeIn = Ve.primitiveField();
        // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
        // so every threshold below is the same threshold restated:
        // c < L becomes ind > (1 - L)/2, and the inequality turns
        // over with it. Same cells, named in the surviving field.
        const volScalarField indHoldEro(phaseIndicator());
        const scalarField& indEro = indHoldEro.primitiveField();
        forAll(VeIn, celli)
        {
            if (VeIn[celli] > 0 && indEro[celli] > 0.25)
            {
                ++eroding;
            }
        }
        reduce(eroding, sumOp<label>());

        Info<< "PHASE_CLOT_EROSION time=" << runTime.userTimeValue()
            << " tauEroPa=" << clotErosionTauPa_
            << " rate=" << clotErosionRate_
            << " maxShearPa=" << gMax(shearPa_.primitiveField())
            << " maxErosionSpeed=" << gMax(VeIn)
            << " maxGrowthSpeed=" << gMax(adhesionSpeedEff_.primitiveField())
            << " maxVnNet=" << gMax(Vn_.primitiveField())
            << " minVnNet=" << gMin(Vn_.primitiveField())
            << " erodingClotCells=" << eroding
            << " coreExempt="
            << Switch(coreShell_ && releaseCoreExempt_)
            << endl;
    }

    // A CELL WITH NO BLOOD LEFT CANNOT GROW MORE CLOT.
    //
    // This is the constraint the model never had, and its absence is the root
    // of the c undershoot -- not the platelet divisor, which only amplified
    // what this let through.
    //
    // THE EQUATION HAS NO FIXED POINT AT PURE CLOT. carrierDivergence()
    // derives, correctly, that chi = 0.5(1-c) transported conservatively
    // gives  ddt(c) + div(c v) = div(v) - 2G.  Evaluate it on the two pure
    // states:
    //
    //     c = +1 (pure blood):  dc/dt = -2G          -- preserved at G = 0
    //     c = -1 (pure clot) :  dc/dt = 2 div(v) - 2G -- NOT preserved
    //
    // Blood is a fixed point and clot is not. So growth keeps running in
    // cells that are already 100% clot, consuming blood that is not there,
    // and c walks past -1.
    //
    // MEASURED, run 20260908-010836 at 7.9 ms:
    //   - 213 of the 239 cells with c < -0.999 still carry a live growth
    //     source, and |grad c| there has median 2.78e+04. A tanh profile
    //     would give ~0 at the plateau; the two-velocity carrier distorts it,
    //     which is why the source does not switch itself off.
    //   - RETRACTED: the "55 cells overshooting in one step" figure was a
    //     TAUTOLOGY. The test used max(1+c, 0), which is zero once c < -1,
    //     so any positive growth passed it. True new source-only crossings
    //     from c >= -1 at that snapshot: ZERO. Those 55 cells were already
    //     outside the bound.
    //   - and the split is total across every run on record: cUnder is
    //     EXACTLY 0 in all 9 one-velocity children (cMin = -0.99998), and
    //     nonzero in 14 of 16 two-velocity ones. The one-velocity carrier is
    //     the pressure-corrected phi_, which is solenoidal, so div(v) = 0 and
    //     c = -1 IS a fixed point there. Same growth source, same schemes.
    //
    // THE BOUND. The growth source contributes dc = -dt*Vn*phaseGradient, so
    // requiring c + dc to stay inside [-1, 1] is a limit on Vn alone:
    //
    //     c + dc >= -1   =>   Vn <=  (1 + c)/(dt*|grad c|)
    //     c + dc <= +1   =>   Vn >= -(1 - c)/(dt*|grad c|)
    //
    // At c = -1 the upper bound is 0: growth stops exactly when the blood is
    // gone. At c = +1 the lower bound is 0: erosion stops exactly when the
    // clot is gone. The two are the same statement and it is symmetric.
    //
    // IT IS APPLIED TO Vn, NOT TO THE c EQUATION, deliberately. lambda_,
    // surfaceLossCoeff_ and the stoichiometric platelet uptake are all built
    // from Vn_ below, so bounding the speed keeps the growth closure
    // consistent. NOTE, measured after this was written: the pairing is NOT
    // exact. prePredictor() runs updateDerivedFields() AGAIN between
    // solvePhaseField() and solvePlatelets(), so the factor is recomputed on
    // a changed c and the two solves do not share one accepted transfer.
    // Frozen-field integration of run 20260908-010836 puts the induced shift
    // in stoichNorm at +0.00126 (0.9984 -> 0.9996), small but nonzero,
    // because P/|grad c| varies cell to cell.
    // Limiting the c source alone would have made the clot grow without
    // consuming the platelets that pay for it.
    //
    // THIS IS NOT THE CLIP. clipPhaseField_ removes material AFTER the field
    // has already left its bounds, which is why its own comment calls it a
    // measurement and not a repair. This never creates the violation: it
    // withholds a source that has no material to act on.
    //
    // AND IT IS INERT IN EVERY HEALTHY CELL. At the interface (c in
    // [-0.9, -0.1]) the measured Vn/limit is at most 3.35e-04 -- three
    // thousand times below binding -- and it binds in 55 of 31360 cells, all
    // of them already pathological. The nine one-velocity runs never reach it
    // at all, so their results are unchanged.
    {
        const dimensionedScalar dtVn("dtVn", dimTime, runTime.deltaTValue());
        const dimensionedScalar zeroDl("zero", dimless, 0);

        // phaseGradient_ already carries gradFloor_, but it reaches 0 in a
        // uniform field and this divides by it.
        const volScalarField pgSafe
        (
            max
            (
                phaseGradient_,
                dimensionedScalar
                (
                    "pgSmall", phaseGradient_.dimensions(), small
                )
            )
        );

        const volScalarField VnErodeMax
        (
            // 1 - c = 2*ind = 2*(1 - q), exactly.
            max(scalar(2)*(scalar(1) - q_), zeroDl)/(dtVn*pgSafe)
        );

        // Measured always, applied only when switched on -- see the note on
        // the growth cap above. Note this is the SAME switch: vnCapEnabled_
        // governs the erosion clamp as well as the growth cap, which is why
        // it is no longer called growthCapEnabled_ -- that name said "growth"
        // while silently controlling erosion too (Codex, 2026-09-08).
        label erodeLimited = 0;
        scalar worstExcess = 0;
        label erodeWouldLimit = 0;
        scalar wouldWorstExcess = 0;

        scalarField& VnIn = Vn_.primitiveFieldRef();
        const scalarField& eMaxIn = VnErodeMax.primitiveField();

        forAll(VnIn, celli)
        {
            if (VnIn[celli] >= -eMaxIn[celli])
            {
                continue;
            }

            ++erodeWouldLimit;
            wouldWorstExcess =
                max(wouldWorstExcess, -eMaxIn[celli] - VnIn[celli]);

            {
                continue;
            }

        }

        Vn_.correctBoundaryConditions();

        reduce(erodeLimited, sumOp<label>());
        reduce(worstExcess, maxOp<scalar>());
        reduce(erodeWouldLimit, sumOp<label>());
        reduce(wouldWorstExcess, maxOp<scalar>());

        Info<< "PHASE_CLOT_VN_BOUND time=" << runTime.userTimeValue()
            << " capOn=" << "no"
            << " erodeLimited=" << erodeLimited
            << " worstExcessMps=" << worstExcess
            << " erodeWouldLimit=" << erodeWouldLimit
            << " wouldWorstExcessMps=" << wouldWorstExcess
            << " maxVnAfter=" << gMax(Vn_.primitiveField())
            << " minVnAfter=" << gMin(Vn_.primitiveField())
            << endl;
    }

    lambda_ =
        pow(max(scalar(1) - q_, scalar(0)), penaltyExponent_)/etaP_;

    // Controlled surface loss: delta_Gamma*(k + Vn - U.n)*w.
    // The coefficient may be signed if a no-slip canonical assumption fails;
    // that is a diagnostic failure, not silently clipped physics.
    surfaceLossCoeff_ =
        interfaceDelta_*(adhesionSpeedEff_ + Vn_ - (U_ & normal_));

    normal_.correctBoundaryConditions();
    deltaTheta_.correctBoundaryConditions();
    interfaceDelta_.correctBoundaryConditions();
    shearPa_.correctBoundaryConditions();
    shearRate_.correctBoundaryConditions();
    adhesionEnhancement_.correctBoundaryConditions();
    adhesionSpeed_.correctBoundaryConditions();
    adhesionSpeedEff_.correctBoundaryConditions();
    Vn_.correctBoundaryConditions();
    lambda_.correctBoundaryConditions();
    surfaceLossCoeff_.correctBoundaryConditions();
}


void Foam::solvers::phaseClotAxisymmetricFluid::updatePermeability()
{
    const phaseClotScopeTimer costGuard_(tPermeability_);

    // NOT `k0p_` ANY MORE, `k0pField_` -- identical when no core is declared,
    // and the core/shell contrast when one is.
    //
    // S12: under dragOccupancyBasis intrinsic the permeability is the
    // NETWORK's, at its intrinsic density -- the constructor has already
    // required kozenyCarmanSolid, so this is that law at a rather than at the
    // cell average, with no floor and no cap. momentumPredictor() assigns
    // the same field again at the drag site, from the state the drag is
    // built from, so the written poroPerm is the permeability that was used.
    {
        poroPerm_ = intrinsicPermeability();
    }

    poroPerm_.correctBoundaryConditions();

    // The Stage 3 coefficient, when it is selected. poroPerm_ above is left
    // exactly as it was: it keeps its name, its units and its consumers, and
    // the content branch does not read it.

    // WHAT PERMEABILITY THE CLOT ACTUALLY GOT, and how much of it is the
    // floor rather than the model. One line, so the defect above can never
    // again be something that has to be excavated from a written field.
    {
        const scalar floorValue = permeabilityFloor_;
        scalar kMin = great;
        scalar kMax = 0;
        label clotCells = 0;
        label flooredCells = 0;
        // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
        // so every threshold below is the same threshold restated:
        // c < L becomes ind > (1 - L)/2, and the inequality turns
        // over with it. Same cells, named in the surviving field.
        const volScalarField indHold(phaseIndicator());
        const scalarField& indIn = indHold.primitiveField();
        const scalarField& kIn = poroPerm_.primitiveField();
        forAll(indIn, celli)
        {
            if (indIn[celli] > 0.95)
            {
                ++clotCells;
                kMin = min(kMin, kIn[celli]);
                kMax = max(kMax, kIn[celli]);
                if (kIn[celli] <= floorValue*(1.0 + 1e-12))
                {
                    ++flooredCells;
                }
            }
        }
        // THE RATIO, NOT THE ABSOLUTE VALUE. kClotMin and kClotMax alone are
        // blind to the thing this model exists to produce. Run
        // 20260905-223740 read kClotMin 1.00097e-14 -> 1.00091e-14 and
        // kClotMax 1.10822e-11 -> 1.10922e-11 across its whole life, which
        // looks like "nothing happened" -- but the field snapshot at
        // t = 5e-04 had 1552 of 2619 clot cells away from BOTH nominals and
        // a cell at 4.035e-11, four times k0p. The extremes are pinned by the
        // core/shell CONTRAST (1e-14 against 1e-11), so a shell cell that
        // halves its permeability lands between them and is invisible.
        //
        // k/k0pField is the quantity the model actually changes: it is 1
        // everywhere at nominal density by construction, so any departure is
        // the compaction response and nothing else. Reported as a
        // distribution, because a single extreme cannot say whether the clot
        // moved or one cell did.
        scalar rMin = great;
        scalar rMax = 0;
        scalar rSum = 0;
        label rCells = 0;
        label nDown2 = 0;      // k/k0p < 0.5   : materially less permeable
        label nUp2   = 0;      // k/k0p > 2     : materially more permeable
        label nOff10 = 0;      // outside [0.9, 1.1]
        const scalarField& k0In = k0pField_.primitiveField();
        forAll(indIn, celli)
        {
            if (indIn[celli] > 0.95 && k0In[celli] > 0)
            {
                const scalar r = kIn[celli]/k0In[celli];
                ++rCells;
                rSum += r;
                rMin = min(rMin, r);
                rMax = max(rMax, r);
                if (r < 0.5)                  ++nDown2;
                if (r > 2.0)                  ++nUp2;
                if (r < 0.9 || r > 1.1)       ++nOff10;
            }
        }
        const label rCellsAll = returnReduce(rCells, sumOp<label>());

        Info<< "PHASE_CLOT_PERMEABILITY time=" << runTime.userTimeValue()
            << " model=" << "kozenyCarmanSolid"
            << " clotCells=" << returnReduce(clotCells, sumOp<label>())
            << " kClotMin="
            << (clotCells > 0 ? returnReduce(kMin, minOp<scalar>()) : 0.0)
            << " kClotMax=" << returnReduce(kMax, maxOp<scalar>())
            << " flooredCells=" << returnReduce(flooredCells, sumOp<label>())
            << " floor=" << floorValue
            << " kRatioMin="
            << (rCellsAll > 0 ? returnReduce(rMin, minOp<scalar>()) : 1.0)
            << " kRatioMax=" << returnReduce(rMax, maxOp<scalar>())
            << " kRatioMean="
            << (rCellsAll > 0
                    ? returnReduce(rSum, sumOp<scalar>())/rCellsAll : 1.0)
            << " kCellsBelowHalf=" << returnReduce(nDown2, sumOp<label>())
            << " kCellsAboveTwice=" << returnReduce(nUp2, sumOp<label>())
            << " kCellsOff10pc=" << returnReduce(nOff10, sumOp<label>())
            << " kRatioCells=" << rCellsAll
            << endl;
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::commitWallBonds()
{
    label newlyFailed = 0;
    label reformed = 0;
    label debondedFaces = 0;
    label clotFaces = 0;
    label injuryFaces = 0;
    label bondedClot = 0;
    scalar debondedArea = 0;
    scalar peak = 0;
    scalar peakAny = 0;
    scalar peakNormal = 0;
    scalar peakShear = 0;
    scalar strength = 0;
    bool present = false;
    label sigmaSeen = 1;
    word sigmaField;

    volVectorField::Boundary& Usbf = Us_.boundaryFieldRef();

    forAll(Usbf, patchi)
    {
        if (!isA<clotWallAdhesionFvPatchVectorField>(Usbf[patchi]))
        {
            continue;
        }

        present = true;

        clotWallAdhesionFvPatchVectorField& bc =
            refCast<clotWallAdhesionFvPatchVectorField>(Usbf[patchi]);

        newlyFailed += bc.commit();
        reformed += bc.reformedFaces();
        peak = max(peak, bc.peakTraction());
        peakAny = max(peakAny, bc.peakTractionAny());
        peakNormal = max(peakNormal, bc.peakTractionNormal());
        peakShear = max(peakShear, bc.peakTractionShear());
        bondedClot += bc.bondedClotFaces();
        strength = bc.adhesiveStrength();
        if (!bc.sigmaFound()) { sigmaSeen = 0; }
        sigmaField = bc.sigmaName();

        const scalarField& bonded = bc.bonded();
        const scalarField& injury = bc.injury();
        const scalarField& magSf = bc.patch().magSf();

        forAll(bonded, facei)
        {
            ++clotFaces;

            // Only the injury footprint can debond. Counting the intact-wall
            // faces here would report 100 debonded faces at t = 0, since they
            // are held permanently unbonded by design.
            if (injury[facei] < 0.5)
            {
                continue;
            }

            ++injuryFaces;
            if (bonded[facei] < 0.5)
            {
                ++debondedFaces;
                debondedArea += magSf[facei];
            }
        }
    }

    if (!returnReduce(present, orOp<bool>()))
    {
        return;
    }

    reduce(newlyFailed, sumOp<label>());
    reduce(reformed, sumOp<label>());
    reduce(debondedFaces, sumOp<label>());
    reduce(clotFaces, sumOp<label>());
    reduce(injuryFaces, sumOp<label>());
    reduce(bondedClot, sumOp<label>());
    reduce(debondedArea, sumOp<scalar>());
    reduce(peak, maxOp<scalar>());
    reduce(peakAny, maxOp<scalar>());
    reduce(peakNormal, maxOp<scalar>());
    reduce(peakShear, maxOp<scalar>());
    reduce(strength, maxOp<scalar>());
    reduce(sigmaSeen, minOp<label>());

    // Printed EVERY step, not only when a face fails. Run 20260809-113416
    // reported "no rupture" without ever saying how close it came, and the
    // margin turned out to be the whole finding. peakTraction/strength is
    // that margin, and it is now always on the record.
    //
    // peakTraction is the maximum over faces that are STILL BONDED, so
    // utilisation is the fraction of the way the surviving bond has left to
    // go. Run 20260810-010103 took the maximum over ALL clot faces and
    // reported utilisation 18.6 once every clot face had already released:
    // an eighteen-fold overload on a bond that no longer existed.
    // peakTractionAny keeps that quantity, clearly named, and
    // bondedClotFaces=0 says plainly that nothing is left to fail.
    // sigmaFound / sigmaField ARE NOT DECORATION. Until run 20260824-033102
    // this line could report peakTractionAny 0 and utilisation 0 for an entire
    // run and mean either "the bond carried no load" or "the bond was never
    // shown a stress at all". On the Eulerian path it meant the second: the
    // condition reads sigmaD, and only solveSkeleton() writes sigmaD, and the
    // Eulerian path never calls solveSkeleton(). Those two readings must never
    // again produce the same line.
    Info<< "PHASE_CLOT_DEBOND time=" << runTime.userTimeValue()
        << " sigmaFound=" << sigmaSeen
        << " sigmaField=" << sigmaField
        << " newFaces=" << newlyFailed
        << " reformedFaces=" << reformed
        << " debondedFaces=" << debondedFaces
        << " bondedClotFaces=" << bondedClot
        << " injuryFaces=" << injuryFaces
        << " wallFaces=" << clotFaces
        << " debondedArea=" << debondedArea
        << " peakTraction=" << peak
        << " peakTractionAny=" << peakAny
        // THE SPLIT: peeling against shearing. An adhesive bond does not fail
        // the same way under the two, and the total |t| alone cannot say which
        // mode is loading it.
        << " peakTractionNormal=" << peakNormal
        << " peakTractionShear=" << peakShear
        << " strength=" << strength
        << " utilisation=" << (strength > 0 ? peak/strength : 0.0)
        << endl;
}


void Foam::solvers::phaseClotAxisymmetricFluid::solveElasticStrain()
{
    const phaseClotScopeTimer costGuard_(tSkeleton_);


    // Relaxation rate: ~0 in clot, beRelaxRate_ in blood.
    //
    // q is the blood indicator (q = 0.5(c+1)), so weighting by q^2 gives a
    // rate and a first derivative that are both continuous at the interface --
    // the same reasoning as the pin weight in solveSkeleton(), for the same
    // reason: a kink in a coefficient is a needless obstacle for an iteration.
    //
    // THIS IS WHAT REPLACES THE PIN, and it is why the pin's damage artefact
    // cannot recur. The pin forced Us to zero through the whole four-cell
    // interface and measured an interface-to-bulk driving energy of 13.5 on
    // run 20260814-055141 -- damage driven by a numerical constraint rather
    // than by the material. Relaxing be to I is not a constraint on a
    // displacement, it is the constitutive statement that blood does not store
    // elastic strain, and it puts no gradient into the interface at all.
    const volScalarField qClamped
    (
        min(max(q_, dimensionedScalar("z", dimless, 0)),
            dimensionedScalar("o", dimless, 1))
    );
    beRelax_ = sqr(qClamped)
       *dimensionedScalar("rate", dimless/dimTime, beRelaxRate_);

    // S11 FIX 6: RELAX ONLY WHERE THERE IS NO NETWORK.
    //
    // be is the elastic state of the network in the cell. Relaxing it at
    // rate*q^2 made the clot's own diffuse edge forget its strain in ~1 us
    // (q^2 ~ 0.9 there): run 20260923-031942's fringe carried be = I to
    // 1.001 while moving at 13.9 mm/s against the body's 7.9, and was combed
    // off by the flow. Fibrin relaxes over seconds, not microseconds, and it
    // does not relax because blood shares its cell. The relaxation exists
    // to keep be at I where it describes nothing, so it is confined there:
    // where the DEPOSITED occupancy thetaNetRef/phi_s0 is below the solver's
    // own residual (residualChi), ramping to zero at it. A dilated network
    // keeps its memory -- dilation does not remove network. Newly deposited
    // content is mixed in stress-free where it is added (see the growth
    // increment of thetaNetRef_), which is what relaxation did for it.
    {
        const volScalarField absent
        (
            "networkAbsent",
            max
            (
                1.0
              - thetaNetRef_
               /dimensionedScalar
                (
                    "residualContent", dimless,
                    residualChi_*skeletonVolumeFraction_
                ),
                dimensionedScalar("zero", dimless, 0)
            )
        );
        // S14: RUPTURED NETWORK HAS NO ELASTIC MEMORY EITHER, and the test is
        // the same one applied to the INTACT network. be is the strain of the
        // cell's intact network; debris carries none. The intact content is
        // thetaNetRef_*(1 - ruptured_), so a fully ruptured cell relaxes at
        // the full rate (its be was set to I and stays there), while the
        // intact part of a partially ruptured cell keeps its strain.
        // max(absent, ruptured_) relaxed that intact part at
        // ruptured^2*beRelaxRate -- tau 100 us at ruptured 0.1, against the
        // 280-330 fractional cells the same transport leaves at the shell
        // label's front (batch 53) -- so every smeared rupture front would
        // have erased the memory of the network behind it.
        volScalarField noMemory("noNetworkMemory", absent);
        {
            noMemory = max
            (
                1.0
              - thetaNetRef_*(1.0 - ruptured_)
               /dimensionedScalar
                (
                    "residualContent", dimless,
                    residualChi_*skeletonVolumeFraction_
                ),
                dimensionedScalar("zero", dimless, 0)
            );
        }
        beRelax_ = sqr(noMemory)
           *dimensionedScalar("rate", dimless/dimTime, beRelaxRate_);

        // S17 A, 09 rule 3: under fibreRelease the intact content is
        // thetaNetRef (1 - d*) = thetaNetRef thI/S, not thetaNetRef (1 - d):
        // pending content is broken network, whose strain is its own fibre,
        // not be's. Applied after the shipped block, which it replaces here.
        {
            noMemory = max
            (
                1.0
              - thetaNetRef_*intactShare()
               /dimensionedScalar
                (
                    "residualContent", dimless,
                    residualChi_*skeletonVolumeFraction_
                ),
                dimensionedScalar("zero", dimless, 0)
            );
            beRelax_ = sqr(noMemory)
               *dimensionedScalar("rate", dimless/dimTime, beRelaxRate_);
        }
    }

    // FIRST STEP WITH EULER. See the declaration in the header: `backward` has
    // only one old-time level at step 1 and its degenerate stencil injects an
    // O(dt) error that persists for the whole run. One Euler step costs
    // O(dt^2) once and nothing after.
    const bool firstStep = runTime.timeIndex() <= 1;
    const word ddtName(firstStep ? word("Euler") : word::null);

    // THE VELOCITY GRADIENT THE STRETCH TERM USES.
    //
    // fvc::grad(U_) is the Gauss operator on the INTERPOLATED face velocity:
    //     grad(U)_ij = (1/V) sum_f Sf_i (interpolate(U))_f,j
    // so its trace is (1/V) sum_f (interpolate(U) & Sf) -- the divergence of a
    // flux NOBODY constrained. The pressure solve makes phi_ solenoidal, and
    // phi_ differs from (interpolate(U) & Sf) by the Rhie-Chow correction.
    //
    // MEASURED, run 20260818-125909: |div phi| <= 8e-6 1/s while tr(fvc::grad U)
    // reached 527 1/s in the clot bulk, 53% of the shear rate, and at G = 400 Pa
    // the ratio |tr L|/|L| hit 1.3825 against the 2-D ceiling of sqrt(2) -- the
    // cell-centred gradient was 98% pure dilatation. d(ln det be)/dt = 2 tr(L),
    // so det(be) drifted to 4.24.
    //
    // Building the SAME Gauss operator on a face velocity whose normal
    // component is exactly phi_/|Sf| makes the trace
    //     (1/V) sum_f (Uf & Sf) = (1/V) sum_f phi_f = div(phi)
    // which is machine zero. Not approximately -- identically, by construction.
    //
    // WHY THIS IS NOT THE dev(gradU) MISTAKE AGAIN. That projected the trace out
    // of the source, and with |tr L|/|L| near 1 the isotropic part removed was
    // ~58% of the tensor norm, so it destroyed the deviatoric response too and
    // made det(be) 16x WORSE (run 20260818-100608, and it lost
    // positive-definiteness). This does not project anything. It builds a
    // different and better discrete L: the correction to each face velocity is
    // purely normal and of order the Rhie-Chow term, about 1% of |U| here, and
    // it is the SAME flux that advects be, so the stretch term and the advection
    // are finally consistent with each other. The trace vanishing is a
    // consequence of that consistency, not the goal imposed on it.
    //
    // fv::gaussGrad::gradf uses Sf[facei]*issf[facei], i.e. Sf_i U_j, the same
    // index convention as fvc::grad, so twoSymm(be & gradU) stays correct --
    // see the transpose note at the top of beTransport.H.
    tmp<volTensorField> tGradU;
    {
        const surfaceVectorField Uf(fvc::interpolate(U_));
        const surfaceVectorField UfFlux
        (
            "UfFlux",
            Uf + mesh.Sf()*(phi_ - (Uf & mesh.Sf()))/magSqr(mesh.Sf())
        );
        tGradU = fv::gaussGrad<vector>::gradf(UfFlux, "gradUflux");
    }

    {
        // be is the SKELETON's elastic strain. It is carried by the skeleton
        // and stretched by the skeleton's velocity gradient, not the fluid's;
        // carrying it on phi_ while calling it a solid strain is the same
        // conflation as advecting the phase field on phi_.
        //
        // A non-solenoidal carrier is safe here: beTransport::eqn is
        // fvm::div(phi,be) - fvm::Sp(fvc::div(phi), be), and that second term
        // is exactly the correction a flux with divergence needs. The skeleton
        // flux legitimately HAS divergence -- a skeleton compacts and dilates,
        // and only the MIXTURE flux is solenoidal.
        const surfaceScalarField phiS(skeletonFlux());

        // S13 FIX B: THE STRETCH SEES THE COMPACTION THE CONTENT SEES.
        //
        // fvc::grad(Vs_) is the Gauss gradient of the LINEARLY INTERPOLATED
        // cell velocity: in 1-D (v[i+1] - v[i-1])/2h, blind to a grid-scale
        // mode. The content, and be itself, are carried by phiS, whose
        // divergence is compact. So a grid-scale compaction of the network
        // changed its content and left its strain untouched, and nothing
        // elastic resisted it: under tension the content-weighted deviatoric
        // stress then pulls network INTO the denser cells, an anti-diffusion
        // the cell gradient cannot answer. Run 20260923-135545: em55's fringe
        // and em56's stretched body grew exactly that mode.
        //
        // The fix is the construction the one-velocity path already uses
        // (beFluxConsistentGradU above): the Gauss gradient of face velocities
        // whose NORMAL component is phiS/|Sf|, the flux that transports be and
        // the content. Tangential components stay interpolated. Its trace is
        // (1/V) sum phiS = div(phiS), so compaction of the content and strain
        // of the network are one statement, face by face. Discrete dispersion
        // of the linearised skeleton row, priced on the returned fields: no
        // unstable cell in either child at any written time under this form.
        tmp<volTensorField> tGradVs;
        {
            const surfaceVectorField Vsf(fvc::interpolate(Vs_));
            const surfaceVectorField VsfFlux
            (
                "VsfFlux",
                Vsf + mesh.Sf()*(phiS - (Vsf & mesh.Sf()))/magSqr(mesh.Sf())
            );
            tGradVs = fv::gaussGrad<vector>::gradf(VsfFlux, "gradVsflux");
        }
        const volTensorField& gradVs = tGradVs();
        beSweepChange_ = beTransport::solve
        (
            be_, phiS, gradVs, beRelax_, beCorrectors_, ddtName,
            false
        );
        // S17 A: the pending fibre's rotation and UNI, ONCE PER STEP, from
        // the SAME gradient that just stretched be (09 section 2).
        rotatePendingFibres(gradVs);
    }

    be_.correctBoundaryConditions();

    // MEASURED BEFORE ANY RENORMALISATION, deliberately. This is the drift the
    // advection actually created, and it stays the headline number whether or
    // not it is then papered over -- a band-aid that also hides the wound is
    // worse than no band-aid.
    beDetError_ = beTransport::maxDetError(be_);

    // THE be EQUATION TERM BY TERM AT THE CELL THAT IS ACTUALLY FAILING.
    //
    // WHY PER-CELL AND NOT A FIELD AVERAGE. Run 20260823-204058 lost positive-
    // definiteness at t = 3.044e-3 s, and the offline analysis localised the
    // damage to the FIRST THREE CELL ROWS against the wall: det(be) by row is
    // 1.13 mean with a 0.2655 minimum at y = 1.5 um, and from row 3 outward it
    // sits within 2% of 1. It is one small cluster, advecting downstream at the
    // clot's own speed, decaying MONOTONICALLY over ~1500 timesteps while its
    // immediate neighbours hold det = 1. Eleven hypotheses were tested against
    // bulk means and max-over-all-cells numbers and every one of them died,
    // because no global quantity can see this.
    //
    // WHAT IS EXCLUDED ALREADY, so this does not re-ask a settled question.
    // The wall BC: beTransport::eqn has no diffusion term and phi is zero at a
    // wall, so the BC cannot reach these cells. Flux inconsistency in the
    // stretch: beFluxConsistentGradU builds gradU from face velocities whose
    // normal components equal phi exactly, and divPhiMax measures 4.9e-08.
    // Conditioning: at the worst cell lmax is 5.499 so 1/lmax is 0.1818 while
    // lmin is 0.0483 -- 3.8x below what the strain allows and ten orders above
    // the double-precision floor. The strain is modest; something is removing
    // volume.
    //
    // HOW IT IS WEIGHED. d(det)/dt = det * tr(be^-1 . d(be)/dt), and for
    // symmetric arguments tr(A.B) is the double inner product A && B. So each
    // right-hand-side term of beTransport::eqn is reconstructed explicitly and
    // converted to the rate of det it is responsible for. detRateActual is the
    // rate the solve actually produced. IF THE TERMS SUM TO IT, the culprit is
    // whichever term is large and negative. IF THEY DO NOT, the discrepancy is
    // the time discretisation or the linear solve, and that gap is itself the
    // answer -- the same reading that settled the momentum question.
    //
    // The per-cell inverse is taken inside the loop rather than as a field, so
    // a cell whose det has already collapsed cannot put an inf into a
    // whole-field operation.
    {
        const volSymmTensorField tAdv
        (
            -fvc::div(phi_, be_, "div(phi,be)")
        );
        const volSymmTensorField tNonCons(be_*fvc::div(phi_));
        const volSymmTensorField tStretch
        (
            beTransport::stretch(be_, tGradU(), false)
        );
        const volSymmTensorField tRelax
        (
            -beRelax_*(be_ - beTransport::identity())
        );

        const symmTensorField& bp = be_.primitiveField();
        const symmTensorField& bo = be_.oldTime().primitiveField();
        const symmTensorField& ad = tAdv.primitiveField();
        const symmTensorField& nc = tNonCons.primitiveField();
        const symmTensorField& st = tStretch.primitiveField();
        const symmTensorField& rx = tRelax.primitiveField();
        // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
        // so every threshold below is the same threshold restated:
        // c < L becomes ind > (1 - L)/2, and the inequality turns
        // over with it. Same cells, named in the surviving field.
        const volScalarField indHold(phaseIndicator());
        const scalarField& indIn = indHold.primitiveField();
        const scalarField& qIn = q_.primitiveField();
        const vectorField& cc = mesh.C().primitiveField();
        const scalarField& vol = mesh.V();
        const scalar dtB = max(runTime.deltaTValue(), vSmall);

        // The worst cell is re-selected every step BECAUSE IT MOVES: the
        // failing cluster advects at ~4.7e-3 m/s, so a fixed index would
        // follow the mesh and lose the material.
        label worst = -1;
        scalar lminWorst = great;
        forAll(indIn, celli)
        {
            if (indIn[celli] <= 0.95) { continue; }
            scalar lmin = 0, lmax = 0;
            beTransport::extremeEigenvalues(bp[celli], lmin, lmax);
            if (lmin < lminWorst) { lminWorst = lmin; worst = celli; }
        }

        scalar wAdv = 0, wNon = 0, wStr = 0, wRel = 0, wAct = 0;
        scalar wDet = 0, wLmax = 0, wX = 0, wY = 0, wInd = 0, wQ = 0, wVol = 0;
        if (worst >= 0)
        {
            const symmTensor B = bp[worst];
            const scalar dB = det(B);
            wDet = dB;
            scalar lminW = 0, lmaxW = 0;
            beTransport::extremeEigenvalues(B, lminW, lmaxW);
            wLmax = lmaxW;
            wX = cc[worst].x(); wY = cc[worst].y();
            wInd = indIn[worst]; wQ = qIn[worst]; wVol = vol[worst];
            if (mag(dB) > vSmall)
            {
                const symmTensor Bi = inv(B);
                wAdv = dB*(Bi && ad[worst]);
                wNon = dB*(Bi && nc[worst]);
                wStr = dB*(Bi && st[worst]);
                wRel = dB*(Bi && rx[worst]);
            }
            wAct = (dB - det(bo[worst]))/dtB;
        }

        Info<< "PHASE_CLOT_BETERM"
            << " time=" << runTime.name()
            << " lminWorst=" << lminWorst
            << " detWorst=" << wDet
            << " lmaxWorst=" << wLmax
            << " xWorst=" << wX
            << " yWorst=" << wY
            // LABELLED FOR WHAT IT HOLDS. S6c-2 repointed this from c_
            // to ind and left the key saying cWorst, so em46 printed
            // 0.9555 where em45 printed -0.9110 for the same cell --
            // same cell, opposite sign, and read literally it calls a
            // 95%-clot cell blood. ind = 1 - q; the c reconstruction,
            // where one is wanted, is 2*q - 1.
            << " indWorst=" << wInd
            << " qWorst=" << wQ
            << " volWorst=" << wVol
            << " detRateAdvection=" << wAdv
            << " detRateNonCons=" << wNon
            << " detRateStretch=" << wStr
            << " detRateRelax=" << wRel
            << " detRateTermSum=" << (wAdv + wNon + wStr + wRel)
            << " detRateActual=" << wAct
            << " detRateGap=" << (wAct - (wAdv + wNon + wStr + wRel))
            << nl;
    }

    // THE BAND-AID, AND IT IS NAMED ONE.
    //
    // det(be) = 1 is exact for any isochoric deformation, and run
    // 20260818-223637 established WHY it drifts anyway: not the source, whose
    // trace was driven to 3.9e-07 with no effect, but NUMERICAL MIXING IN THE
    // ADVECTION. Minkowski's determinant inequality says that for SPD A, B and
    // lambda in [0,1],
    //     det(lambda A + (1-lambda) B)^(1/n)
    //         >= lambda det(A)^(1/n) + (1-lambda) det(B)^(1/n)
    // so a convex combination of unit-determinant SPD tensors has det >= 1,
    // with equality only when A = B. A limited/upwinded scheme forms exactly
    // such combinations, so it can only push det UP. Measured at t=3e-3:
    // 12473 of 12500 cells above 1 and 27 below, worst at the clot/interface
    // boundary where cell-to-cell contrast in be is largest, and exactly 1.0000
    // in blood where every neighbour is identical.
    //
    // Rescaling by det^(-1/3) restores det = 1 exactly (det(s*be) = s^3 det be
    // in 3-D) and preserves the eigenVECTORS and the RATIOS of the eigenvalues.
    // What it does NOT do is undo the smearing: the deviatoric part is still
    // mixed, so the shape of the strain is still wrong, only its volume is
    // right. It is here to answer one question cheaply -- with det forced to 1,
    // is the rest of the physics sane? -- before committing to the
    // log-conformation formulation, where tr(log be) is LINEAR and advective
    // mixing preserves det identically rather than by correction.
    //
    // NOT A DEFAULT. If it were on by default, the det gate would pass on every
    // future run while the underlying drift continued unmeasured, which is the
    // exact failure this project has been caught by twice.
    {
        const volScalarField detBe(det(be_));
        be_ *= pow
        (
            max(detBe, dimensionedScalar("detFloor", dimless, small)),
            -1.0/3.0
        );
        be_.correctBoundaryConditions();
        beDetErrorPost_ = beTransport::maxDetError(be_);
    }

    // S14: AFTER the rescale, so the criterion reads the isochoric be the
    // momentum equation will use, and BEFORE the diagnostics below, so they
    // describe the network that survived.
    ruptureOverstretchedNetwork();

    // AFTER the rescale, so they describe the be the momentum equation will
    // actually use.
    beMinMinor_ = beTransport::minLeadingMinor(be_);
    beMaxStretch_ = beTransport::maxStretch(be_);

    // S17 C (design 2C; change 10): Gamma_t >= 0 -- the M-matrix property of
    // the Vs predictor -- holds by construction only while be is SPD, so
    // under implicitTangent a loss of SPD is fatal here, before the stress
    // and the matrix read it, and not only the BE_SPD_LOST log line.
    if (!(beMinMinor_ > 0))
    {
        FatalErrorInFunction
            << "elasticCoupling implicitTangent: be lost positive-"
            << "definiteness (min leading minor " << beMinMinor_ << " at t = "
            << runTime.name() << "). Gamma_t >= 0 and the Vs matrix's "
            << "M-matrix property rest on SPD be (design 2C)."
            << exit(FatalError);
    }
}


Foam::tmp<Foam::volSymmTensorField>
Foam::solvers::phaseClotAxisymmetricFluid::elasticStress() const
{
    // sigma_e = phiS * gD * G(chi) * dev(be)
    //
    // Deviatoric only: the volumetric part is carried by the pressure, which
    // is what makes this compatible with the incompressible momentum equation
    // it is being added to. dev(be) is exactly the deviatoric Cauchy stress of
    // an incompressible neo-Hookean divided by G, verified offline against the
    // strain-energy route 2 dev((dW/db).b) in
    // scripts/check_b_transport_algebra.py.
    //
    // The two weights are the SAME ones the skeleton stress carries, and they
    // are kept for the same reasons. gD = (1-Dg)^2 + eta degrades broken
    // material. phiS fades the elastic operator out with the material so the
    // clot's outer surface is traction-free rather than clamped -- the
    // diffuse-domain fix from run 20260815-201949, which is not something the
    // change of kinematics makes unnecessary.
    // S14: the intact network's occupancy, exactly as skeletonFaceStress()
    // builds it -- two builders of one weight must stay one weight.
    const volScalarField chi
    (
        phaseIndicator()*intactFraction()
    );
    const volScalarField gD(sqr(1.0 - Dg_) + etaRes_);

    volScalarField phiS("elasticSolidWeight", chi);
    {
        // S16 B: max(ind, floor)*(1 - d), as skeletonFaceStress() builds it.
        phiS = skeletonStressWeight();
    }
    // S23: T's weight uses d*, exactly as skeletonFaceStress() applies it.
    applyTornStressWeight(phiS);
    recordStressWeight(phiS.primitiveField(), 1);
    // S23 rev. 4.1 F1': the weight as consumed (nothing writes phiS after).
    recordTornWeight(phiS.primitiveField(), 1);

    // THE VOLUMETRIC PART, ON THE TWO-VELOCITY PATH ONLY.
    //
    // The comment above says the volumetric part "is carried by the pressure".
    // That was TRUE for one velocity and one pressure and is FALSE once p
    // enforces only the MIXTURE constraint div(phi_f*U + phi_s*Vs) = 0: the
    // skeleton's own volumetric response then has no carrier at all, and run
    // 20260905-113709 measured the consequence -- phi_s nearly doubled, c
    // reached -2.74, SIGFPE at t = 0.0028.
    //
    // sigma_n = -phi_s * p_n * I, so div(sigma_n) = -grad(phi_s p_n), which is
    // Cogan & Guy Eq. 19's network-pressure term exactly.
    //
    // ADDED HERE RATHER THAN IN THE MOMENTUM EQUATION ON PURPOSE. sigma_e
    // already reaches the skeleton through correctPressureTwoVelocity() in
    // FACE form, so the pressure Laplacian can cancel it face by face -- the
    // Rhie-Chow-like cure Ghione prescribes for forces that scale with a
    // phase-fraction gradient. Folding p_n into the same tensor means it
    // inherits that treatment with no new plumbing, and it cannot get out of
    // step with the deviatoric part.
    //
    // GATED on twoVelocitySkeleton_ so no existing one-velocity result moves:
    // on that path the original comment still holds and the term would be
    // double-counted against the mixture pressure.
    {
        // THE TWO WEIGHTS ARE DIFFERENT FIELDS, AND MIXING THEM COST A RUN.
        //
        // `phiS` above is the ELASTIC DIFFUSE-DOMAIN WEIGHT,
        // max(chi, skeletonSolidFloor) -- chi, not a volume fraction, and
        // floored at 1e-04 so the elastic operator does not vanish outside the
        // clot. The network pressure needs the SOLID VOLUME FRACTION,
        // phi_s = chi*skeletonVolumeFraction, because Cogan & Guy Eq. 19 is
        // -grad(phi_n p_n) with phi_n the genuine fraction.
        //
        // Run 20260905-191135 used `phiS` for both. Two consequences:
        //   - inside the clot chi ~ 1 against phi_s = 0.01, so the term was
        //     100x too strong;
        //   - IN THE LUMEN phi_s = 0 but the floor makes phiS = 1e-04, and
        //     p_n there is a constant -0.0667 Pa (measured: pnMin was exactly
        //     -0.0666667 every step, all run). A network pressure acted where
        //     there is no network.
        //
        // The lumen is the worst place for a spurious force on the skeleton:
        // every term of VsEqn carries phiSkel_, which is 0 there, so
        // VsEqn.A() = 0 and the only diagonal is Kres = residualPhiS/deltaT =
        // 66.7 1/s -- against 3.3e+05 of drag inside the clot. MEASURED:
        // whole-field maxVs was 48x run 20260905-113709's from the FIRST
        // samples (0.0294 against 0.00061) and ran to 0.62 m/s, while every
        // CLOT-CELL metric -- meanVsOverU 0.9386, maxVsOverU 1.49,
        // exceedFractionMargin 0.0133 -- was indistinguishable between the two
        // runs. The clot was never the problem.
        //
        // solidFraction() is EXACTLY 0 in the lumen, so the term vanishes
        // identically there, which is what Eq. 19 requires.
        tmp<volSymmTensorField> tSigma
        (
            volSymmTensorField::New
            (
                "sigmaElasticInstant",
                phiS*gD*GField_*dev(be_)
              - networkPartialPressure()*symmTensor::I
            )
        );
        {
            // S17 A (design 2A "Stress"): the tensor the weight multiplies is
            // the split M = [(thI/S) dev(be) + (thP/S) T dev(n n)]/(1 - d),
            // not S16's dev(be) -- recorded (S17split, site 1) and applied as
            // the difference, so the pressure term above stays the one
            // expression every weight guard reads. Undamaged, M = dev(be)
            // bit for bit and the difference is exactly zero.
            const volSymmTensorField M(splitDeviatoricStress());
            recordStressSplit(M.primitiveField(), 1);
            tSigma.ref() += phiS*gD*GField_*(M - dev(be_));
        }
        return tSigma;
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::solvePassiveDisplacement()
{
    const phaseClotScopeTimer costGuard_(tSkeleton_);


    // dUs/dt + (v.grad)Us = v, written conservatively.
    //
    // div(phi,Us) - Sp(div(phi),Us) is the material derivative for a mesh
    // whose velocity field is not discretely divergence-free. The second term
    // is not optional bookkeeping: dropping it makes a uniform Us grow wherever
    // the discrete continuity residual is nonzero, which is everywhere before
    // the pressure solve has fully converged. Same construction as the be
    // equation, for the same reason.
    //
    // FIRST STEP WITH EULER, matching solveElasticStrain(). If the diagnostic
    // took the degenerate `backward` first step and be did not, the two would
    // disagree at O(dt) for a reason that has nothing to do with either one's
    // physics -- and this field exists precisely to be compared against.
    const bool firstStep = runTime.timeIndex() <= 1;

    fvVectorMatrix UsEqn
    (
        (
            firstStep
          ? fv::EulerDdtScheme<vector>(Us_.mesh()).fvmDdt(Us_)
          : fvm::ddt(Us_)
        )
      + fvm::div(phi_, Us_, "div(phi,Us)")
      - fvm::Sp(fvc::div(phi_), Us_)
     ==
        U_
    );

    UsEqn.solve();
    Us_.correctBoundaryConditions();
}


void Foam::solvers::phaseClotAxisymmetricFluid::solveDamage()
{
    const phaseClotScopeTimer costGuard_(tDamage_);

    // Ambrosio-Tortorelli / Miehe phase-field fracture.
    //
    // Tension-compression split: the volumetric-deviatoric decomposition of
    // Amor et al. (2009) is used rather than a full spectral split. It needs
    // no eigen-decomposition, is standard, and is adequate here because the
    // drained Poisson ratio is 0.12, far from incompressible. The choice is
    // recorded in AM2020-clot-mechanics-v3.md and is a declared approximation.
    const volTensorField gradUs(fvc::grad(Us_));

    // The phase INDICATOR, the same quantity solveSkeleton() calls chi --
    // not the solid volume fraction, which is theta_ref times smaller.
    // Declared here because the damage driver is weighted by it, and SQUARED;
    // see the long note at psiPlus below. Follows phaseIndicatorBasis_, so an
    // 0.22% overshoot above 1 reaches psiPlus as 0.43%.
    const volScalarField chiDamage
    (
        phaseIndicator()
    );

    // Same strain measure as solveSkeleton(), or psi+ would be computed from a
    // different kinematics than the stress that produced it.
    const volSymmTensorField eps
    (
      symm(gradUs) + 0.5*symm(gradUs & gradUs.T())
    );
    const volScalarField trEps(tr(eps));
    const volScalarField bulkK(lambdaField_ + (2.0/3.0)*GField_);

    const volScalarField trPos(max(trEps, dimensionedScalar("z", dimless, 0)));
    const volSymmTensorField devEps(dev(eps));

    // psi+ = 0.5 K <tr eps>+^2 + G |dev eps|^2, with K and G as FIELDS so a
    // core/shell clot stores the energy each region's own modulus implies.
    // This is where the whole localisation comes from: psi+ ~ tau^2/G at a
    // given load, so Xu's 4 Pa shell over a 40 Pa core stores TEN TIMES the
    // driving energy in the shell before any toughness contrast is applied.
    // NEO-HOOKEAN: the Amor split taken on the SAME energy the stress comes
    // from, not on the small-strain proxy above.
    //
    //     psi+ = K/2 <ln J>+^2  +  G/2 (I1bar - 3)
    //
    // Volumetric part positive only in dilatation, exactly as <tr eps>+ is
    // below; deviatoric part is I1bar - 3 >= 0 by AM-GM for every deformation,
    // so it is always a valid tensile driver and never has to be clipped.
    //
    // It reduces to 0.5 K <tr eps>+^2 + G |dev eps|^2 at small strain, the
    // same limit the stress reduces to, so a case that never leaves small
    // strain gets the same damage field from either model. That agreement is
    // the property the verification gate measures.
    volScalarField psiPlusRaw
    (
        "psiPlusRaw",
        0.5*bulkK*sqr(trPos) + GField_*magSqr(devEps)
    );

    {
        // THE SAME ENERGY, REACHED WITHOUT A DISPLACEMENT FIELD.
        //
        // The Eulerian formulation carries be = F F^T on the current
        // configuration and never forms F, so the two invariants the neo-
        // Hookean psi+ needs have to come from be directly:
        //
        //     J     = sqrt(det be)            since det be = det(F F^T) = J^2
        //     I1bar = J^(-2/3) tr(be)         since tr(F F^T) = tr(F^T F) = I1
        //
        // Both are frame-indifferent in the strong sense that matters here: a
        // superposed rigid rotation Q sends be -> Q be Q^T, which changes every
        // component and leaves det and tr alone. That is checked offline in
        // scripts/check_b_transport_algebra.py (case_damage_energy), which
        // evaluates psi+ by this route and by the grad(Us) route on five
        // deformations including a rotated one and agrees to 2.2e-16, with a
        // planted J = det(be) bug caught by that case and no other.
        //
        // Same floor, same reason as the branch below: the floor has to bound
        // J^(-2/3), not merely keep the logarithm finite.
        const volScalarField J
        (
            max
            (
                sqrt(max(det(be_), dimensionedScalar("zero", dimless, 0))),
                dimensionedScalar("Jfloor", dimless, 1e-3)
            )
        );
        const volScalarField lnJPos
        (
            max(log(J), dimensionedScalar("zero", dimless, 0))
        );
        const volScalarField I1bar(pow(J, -2.0/3.0)*tr(be_));

        psiPlusRaw =
            0.5*bulkK*sqr(lnJPos)
          + 0.5*GField_*(I1bar - dimensionedScalar("three", dimless, 3));

        if (skeletonLawLT_)
        {
            // S25 A (design section 2A, physics 08 Q1): the damage energy
            // under LT is LT's own isochoric energy, G0 sum_i w(lbar_i)
            // (w' = tauHat/l; 0.5 G (I1bar - 3) is the same sum for
            // tauHat = l^2), with today's vol/dev split: the same J and
            // the same volumetric half.
            volScalarField WLT("WLT", 0.0*I1bar);
            const auto lawEnergy = [this]
            (
                const symmTensor& beV, const scalar JV
            )
            {
                const vector evE(eigenValues(beV));
                const double eE[3] = {evE.x(), evE.y(), evE.z()};
                double WV = 0;
                if
                (
                    !skeletonLawKernel::energy
                    (
                        skeletonLawTable_, eE, JV, WV
                    )
                )
                {
                    FatalErrorInFunction
                        << "skeletonLaw valanisLandelTrilinear: be = " << beV
                        << " (eigenvalues " << evE << ", J " << JV << ") is "
                        << "not SPD, so LT's energy has no real stretches."
                        << exit(FatalError);
                }
                return WV;
            };
            {
                scalarField& WI = WLT.primitiveFieldRef();
                const symmTensorField& beW = be_.primitiveField();
                const scalarField& JW = J.primitiveField();
                forAll(WI, celli)
                {
                    WI[celli] = lawEnergy(beW[celli], JW[celli]);
                }
                volScalarField::Boundary& WB = WLT.boundaryFieldRef();
                forAll(WB, patchi)
                {
                    const symmTensorField& beP = be_.boundaryField()[patchi];
                    const scalarField& JP = J.boundaryField()[patchi];
                    forAll(WB[patchi], facei)
                    {
                        WB[patchi][facei] = lawEnergy(beP[facei], JP[facei]);
                    }
                }
            }
            psiPlusRaw = 0.5*bulkK*sqr(lnJPos) + GField_*WLT;
        }
    }

    // WEIGHT THE DRIVING ENERGY BY THE SOLID FRACTION.
    //
    // Run 20260814-055141 tore every child from the outside in, and the cause
    // is here rather than in the fracture model. solveSkeleton() pins Us to
    // zero in the fluid with `outsidePin = q*(2G+lambda)/ell^2`, because
    // outside the clot there is nothing to anchor a displacement and the
    // operator would be singular. That pin is a NUMERICAL device, but it
    // drags Us from its clot value down to zero across the four cells of the
    // interface, and grad(Us) there is correspondingly enormous. Measured on
    // that run, H by phase:
    //
    //     bulk clot  c < -0.9      maxH = 2.64e-02
    //     interface  -0.5..0.5     maxH = 3.58e-01     <- 13.5x the bulk
    //     blood      c > 0.99      maxH = 2.51e-03
    //
    // So the damage criterion was met first in a numerical boundary layer,
    // not in the material. Every "tear" was the interface failing, which is
    // exactly why the damage always reached the outer surface and never left
    // intact clot beyond it.
    //
    // The energy that can drive fracture is the energy stored in SOLID, so it
    // scales with how much solid is present. chi^2 rather than chi: chi alone
    // only halves the interface (chi = 0.5 at c = 0) against a 13x error, and
    // the square is the standard interpolation for a solid-phase energy in a
    // diffuse-interface formulation.
    //
    // chi^2 SUPPRESSES the artefact; it does not remove its cause. That cause
    // is now addressed directly by skeletonDiffuseDomain (see the header):
    // with the elastic operator weighted consistently with the body force, the
    // clot's outer surface is traction-free rather than clamped, and grad(Us)
    // across the interface is a physical gradient rather than a numerical
    // boundary layer.
    //
    // SO THE WEIGHT CHANGES WITH THE FORMULATION.
    //
    //   diffuse domain OFF: chi^2. The strain in the interface is still
    //       spurious, and a square is what it takes to hold a 13.5x error
    //       down -- chi alone only halves it, since chi = 0.5 at c = 0.
    //
    //   diffuse domain ON: chi. The energy that can drive fracture is the
    //       energy stored in the solid that is actually there, which is what
    //       the chi-weighted operator stores. Keeping the square here would
    //       suppress fracture initiating AT the clot surface, and a real crack
    //       very often starts there -- so the square would trade one wrong
    //       answer for another.
    //
    // The exact diffuse-domain statement is that the weight cancels: both
    // sides of (Gc/ell)(Dg - ell^2 lap Dg) = 2(1-Dg)H carry the indicator, so
    // Dg is invariant under it. Keeping chi on the driver alone is therefore
    // deliberately CONSERVATIVE -- it makes surface damage harder, never
    // easier -- and the phase-binned diagnostic below reports whether it was
    // needed at all.
    volScalarField psiPlus("psiPlus", sqr(chiDamage)*psiPlusRaw);
    {
        psiPlus = chiDamage*psiPlusRaw;
    }

    // Irreversibility: H is the running maximum of psi+.
    H_ = max(H_, psiPlus);
    H_.correctBoundaryConditions();

    // WHERE the driving energy lives, by phase. Three bins, so a run says for
    // itself whether the damage is being driven by the material or by the
    // interface. If interface >> bulk the result is an artefact, whatever the
    // tear numbers say.
    {
        scalar hBulk = 0, hInterface = 0, hBlood = 0;
        // And, for a core/shell clot, the question the whole design turns on:
        // is the driving energy in the SHELL, where it is supposed to be, or
        // in the core, which is meant to hold its shape? Both are restricted
        // to bulk clot so the interface cannot contaminate either.
        scalar hCore = 0, hShell = 0;
        scalar dgCore = 0, dgShell = 0;
        // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
        // so every threshold below is the same threshold restated:
        // c < L becomes ind > (1 - L)/2, and the inequality turns
        // over with it. Same cells, named in the surviving field.
        const volScalarField indHold(phaseIndicator());
        const scalarField& indIn = indHold.primitiveField();
        const scalarField& hIn = H_.primitiveField();
        const scalarField& dgIn = Dg_.primitiveField();
        const scalarField& sIn = shell_.primitiveField();
        forAll(indIn, celli)
        {
            if (indIn[celli] > 0.95)
            {
                hBulk = max(hBulk, hIn[celli]);
                if (sIn[celli] < 0.5)
                {
                    hCore = max(hCore, hIn[celli]);
                    dgCore = max(dgCore, dgIn[celli]);
                }
                else
                {
                    hShell = max(hShell, hIn[celli]);
                    dgShell = max(dgShell, dgIn[celli]);
                }
            }
            else if (indIn[celli] > 0.25)
            {
                hInterface = max(hInterface, hIn[celli]);
            }
            else
            {
                hBlood = max(hBlood, hIn[celli]);
            }
        }
        Info<< "PHASE_CLOT_DAMAGE_DRIVER time=" << runTime.userTimeValue()
            << " Hbulk=" << returnReduce(hBulk, maxOp<scalar>())
            << " Hinterface=" << returnReduce(hInterface, maxOp<scalar>())
            << " Hblood=" << returnReduce(hBlood, maxOp<scalar>())
            << " Hcore=" << returnReduce(hCore, maxOp<scalar>())
            << " Hshell=" << returnReduce(hShell, maxOp<scalar>())
            << " DgCore=" << returnReduce(dgCore, maxOp<scalar>())
            << " DgShell=" << returnReduce(dgShell, maxOp<scalar>())
            << " maxDg=" << gMax(Dg_) << endl;
    }

    // (Gc/ell)(Dg - ell^2 lap Dg) = 2 (1 - Dg) H
    //  => [Gc/ell + 2H] Dg - (Gc ell) lap Dg = 2H
    //
    // Gc IS A FIELD. A core at the physiological 5.547 J/m2 under a shell at a
    // calibrated value cannot be damaged by any load this model reaches --
    // Dg > 0.95 needs H > 9.5*Gc/ell = 3.5e7 J/m3 there -- so "the core keeps
    // its shape" is a property of the material assignment rather than an
    // outcome to be hoped for. That is the guarantee the core/shell case rests
    // on.
    const volScalarField coeff(GcField_/ell_ + 2.0*H_);

    fvScalarMatrix DgEqn
    (
        fvm::Sp(coeff, Dg_)
      - fvm::laplacian(GcField_*ell_, Dg_)
     ==
        2.0*H_
    );

    fvConstraints().constrain(DgEqn);
    DgEqn.solve();

    // Damage is bounded and monotone by construction; clip against round-off.
    Dg_ = min
    (
        max(Dg_, dimensionedScalar("z", dimless, 0)),
        dimensionedScalar("o", dimless, 1)
    );
    Dg_.correctBoundaryConditions();
}


void Foam::solvers::phaseClotAxisymmetricFluid::releaseDetachedMaterial()
{
    const phaseClotScopeTimer costGuard_(tRelease_);

    // Fully damaged clot material is returned to the fluid phase.
    //
    // S6c-2 PORTED THIS, IT DID NOT TRANSLITERATE IT, AND THE DIFFERENCE IS
    // THE WHOLE POINT. The old form was
    //
    //     if (Dg > DgDetach && c < 0) c = 1.0;
    //
    // which returned the LEGACY field to fluid and left thetaC/thetaH
    // untouched. It was half a model: on the one configuration where it
    // fired, the two representations of the same clot would have separated
    // by construction. With c deleted there is only one representation left,
    // so the release has to act on the content itself.
    //
    // THAT MAKES IT A CONTENT DELETION, NOT A RELABEL. Zeroing thetaC/thetaH
    // removes material from a conserved field, so it MUST appear in the
    // balance or contentResidual will report it as a conservation failure --
    // which is exactly the class of defect that balance was added to catch.
    // detachedContentStep_ carries it to writeBalanceDiagnostic() as a named
    // sink; it is reset there, after the report.
    //
    // THE THRESHOLD IS THE SAME THRESHOLD. c < 0 <-> ind > 0.5 exactly under
    // c = 1 - 2*ind, the mapping already documented at the release support
    // limit. Written as a literal 0.5 rather than derived, because unlike
    // supportLimit and releasedMaskLimit this bound was never a dictionary
    // key -- it is the midpoint of the phase field, and the midpoint of the
    // indicator is 0.5 whatever the case says.
    //
    // IT HAS NEVER FIRED. Run 20260920-060033 reached DgCore 8.93503e-08 and
    // DgShell 5.01935e-08 against DgDetach 0.95 -- seven decades of margin,
    // and zero PHASE_CLOT_DETACHMENT lines in either child. So this port
    // cannot move S6c-2's bit-identity gate, and equally: the gate cannot
    // validate it. Pre-registration row 6 asserts the margin holds in both
    // children rather than assuming it.
    const volScalarField indDetach(phaseIndicator());
    const scalarField& indDetIn = indDetach.primitiveField();

    label released = 0;
    scalar releasedVolume = 0;
    scalar releasedContent = 0;

    forAll(Dg_, celli)
    {
        if (Dg_[celli] > DgDetach_ && indDetIn[celli] > 0.5)
        {
            // Booked BEFORE the zeroing, or the sink is identically zero and
            // the balance closes against a deletion it cannot see.
            releasedContent +=
                (thetaC_[celli] + thetaH_[celli])*mesh.V()[celli];

            thetaC_[celli] = 0.0;
            thetaH_[celli] = 0.0;
            // S10: the deleted network takes its reference with it, or the
            // emptied cell would read as network dilated to nothing.
            thetaNetRef_[celli] = 0.0;
            // D74: and the intact reference, booked for PHASE_CLOT_REFI.
            {
                refITransferStep_ -= thetaNetRefIntact_[celli]*mesh.V()[celli];
                thetaNetRefIntact_[celli] = 0.0;
            }
            // S14: and its rupture label -- the cell holds no network.
            ruptured_[celli] = 0.0;
            // S16 C: and its damage partition, in the same statement as
            // theta (Change 4): an empty cell has theta = 0 and d = d* = 0.
            {
                thetaIntact_[celli] = 0.0;
                thetaPending_[celli] = 0.0;
                thetaBroken_[celli] = 0.0;
            }
            // S17 A: and its pending fibre -- no pending content, no fibre.
            {
                thetaPendingB_[celli] = Zero;
                thetaPendingP_[celli] = 0.0;
                pendingStretch_[celli] = 1.0;
                // S17b A1: no pending content, no taut share (V1).
                thetaTaut_[celli] = 0.0;
            }

            releasedVolume += mesh.V()[celli];
            ++released;
        }
    }

    const label totalReleased = returnReduce(released, sumOp<label>());

    if (totalReleased > 0)
    {
        thetaC_.correctBoundaryConditions();
        thetaH_.correctBoundaryConditions();
        thetaNetRef_.correctBoundaryConditions();
        thetaNetRefIntact_.correctBoundaryConditions();
        ruptured_.correctBoundaryConditions();
        {
            thetaIntact_.correctBoundaryConditions();
            thetaPending_.correctBoundaryConditions();
            thetaBroken_.correctBoundaryConditions();
            syncDamageFromPartition("detachment");
        }
        {
            thetaPendingB_.correctBoundaryConditions();
            thetaPendingP_.correctBoundaryConditions();
            pendingStretch_.correctBoundaryConditions();
            thetaTaut_.correctBoundaryConditions();
            syncPendingFibres("detachment");
        }

        const scalar totalVolume =
            returnReduce(releasedVolume, sumOp<scalar>());
        const scalar totalContent =
            returnReduce(releasedContent, sumOp<scalar>());

        detachedContentStep_ += totalContent;

        Info<< "PHASE_CLOT_DETACHMENT time=" << runTime.userTimeValue()
            << " cells=" << totalReleased
            << " volume=" << totalVolume
            << " contentRemoved=" << totalContent
            << " maxDamage=" << gMax(Dg_)
            << endl;
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::updateReleasedIndicator()
{
    // WHICH CLOT MATERIAL IS FREE TO LEAVE?
    //
    // Run 20260810-212211 released 59 of 60 bonded faces and the clot did not
    // move: debonding changes a boundary condition on Us, and Us is a
    // displacement of order 1e-7 m on a fixed mesh. The phase field, which is
    // what anyone actually looks at, has no advective term at all. So the
    // model could show a bond failing but never an embolus leaving.
    //
    // A cell is SUPPORTED if a path of clot cells connects it to a wall face
    // whose bond is intact. Everything else is released. That is a connected-
    // component question, computed here by max-propagation: seed 1 on clot
    // cells touching an intact bond, then repeatedly take the maximum over
    // clot neighbours. After enough sweeps the support has reached every cell
    // it can reach.
    //
    // Connectivity rather than a geometric rule -- "the column above a
    // debonded face is free" -- because a geometric rule encodes this one
    // geometry, and the next case would silently get a wrong answer instead
    // of an obviously wrong one.
    released_ = dimensionedScalar("zero", dimless, 0.0);

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    // S5: the mask reads the PHASE INDICATOR, not c. phaseIndicator()
    // already carries the basis switch -- legacy max(1-q,0), content
    // (thetaC+thetaH)/theta_ref -- and every other chi consumer in this
    // solver goes through it. These two mask builders were the sites still
    // reading c_ directly, which is why the run on record has
    // phaseIndicatorBasis=content everywhere EXCEPT here.
    //
    // The thresholds transform exactly, because ind = 1 - q = (1 - c)/2
    // under the legacy basis:
    //     c <  0            <->  ind > 0.5
    //     c <  supportLimit <->  ind > (1 - supportLimit)/2
    // Verified on run 20260919-000047's written fields: 0 differing cells
    // at BOTH thresholds across all 8 write times. Under the content basis
    // the clot mask moves 0-2 cells of ~2620 and the support mask 1-15 of
    // ~2942, growing -- that difference is the point of the stage, not a
    // defect.
    const volScalarField ind(phaseIndicator());
    const scalarField& indIn = ind.primitiveField();

    scalarField support(mesh.nCells(), 0.0);

    // THE MATERIAL THRESHOLD FOR CONNECTIVITY, separate from the threshold
    // that defines the released set below. A cell may CONDUCT support while
    // carrying only a little clot; only a cell that is genuinely clot (c < 0)
    // can itself be released. Keeping the two apart is what lets the
    // connectivity be made robust without redefining what "the clot" is.
    const scalar supportLimit = releaseSupportLimit_;
    const scalar indSupportLimit = 0.5*(1.0 - supportLimit);
    const scalar indClotLimit = 0.5;

    // Seed. A patch WITHOUT the adhesion condition is a weld: clot touching it
    // is supported unconditionally. A patch with it supports only where the
    // bond has not yet failed.
    const volVectorField::Boundary& Usbf = Us_.boundaryField();
    forAll(Usbf, patchi)
    {
        const fvPatch& patch = Usbf[patchi].patch();
        if (!isA<wallFvPatch>(patch))
        {
            continue;
        }
        const labelUList& faceCells = patch.faceCells();

        if (isA<clotWallAdhesionFvPatchVectorField>(Usbf[patchi]))
        {
            const clotWallAdhesionFvPatchVectorField& bond =
                refCast<const clotWallAdhesionFvPatchVectorField>(Usbf[patchi]);
            const scalarField& bonded = bond.bonded();
            forAll(faceCells, facei)
            {
                const label celli = faceCells[facei];
                if (bonded[facei] > 0.5 && indIn[celli] > indSupportLimit)
                {
                    support[celli] = 1.0;
                }
            }
        }
        else
        {
            forAll(faceCells, facei)
            {
                const label celli = faceCells[facei];
                if (indIn[celli] > indSupportLimit)
                {
                    support[celli] = 1.0;
                }
            }
        }
    }

    // Propagate through clot material only, so support cannot leak across the
    // lumen and re-attach an embolus that has floated next to the wall.
    for (label sweep = 0; sweep < releaseSupportSweeps_; ++sweep)
    {
        scalar changed = 0;
        forAll(owner, facei)
        {
            const label own = owner[facei];
            const label nei = neighbour[facei];
            if (indIn[own] <= indSupportLimit || indIn[nei] <= indSupportLimit)
            {
                continue;
            }
            const scalar high = max(support[own], support[nei]);
            if (support[own] < high) { support[own] = high; ++changed; }
            if (support[nei] < high) { support[nei] = high; ++changed; }
        }
        if (returnReduce(changed, sumOp<scalar>()) < 0.5)
        {
            break;
        }
    }

    // THE CORE MAY NOT EMBOLISE. Dr He Li, 2026-08-25: "the core region is
    // stiff and not permittable to the flow, so it only deforms under the flow
    // stress, but does not slide or detach or rupture. While the shell region
    // is soft and permeable ... It can deform, rupture, and detach from the
    // shell region itself." Before this, the support test ran on the c level
    // set with no material distinction at all, so a connectivity pinch could
    // and did hand core cells to Ubody. releaseCoreExempt is on by default
    // ONLY when coreShell is on -- a single-material case has no core to
    // exempt and reproduces exactly as before.
    const scalarField& shellIn = shell_.primitiveField();
    const bool exemptCore = coreShell_ && releaseCoreExempt_;
    scalarField& rel = released_.primitiveFieldRef();
    label releasedCells = 0;
    label coreExempted = 0;
    forAll(rel, celli)
    {
        if (indIn[celli] > indClotLimit && support[celli] < 0.5)
        {
            if (exemptCore && shellIn[celli] < 0.5)
            {
                ++coreExempted;
                continue;
            }
            rel[celli] = 1.0;
            ++releasedCells;
        }
    }
    reduce(coreExempted, sumOp<label>());
    released_.correctBoundaryConditions();

    // THE NUMBER THAT SHOULD ALWAYS BE ZERO, AND WAS TWENTY.
    //
    // A cell sitting directly on an INTACT bond cannot be free. Run
    // 20260824-220555 released 1297 of 1317 clot cells while every one of the
    // forty bonds was still holding, and twenty of the released cells were the
    // wall cells of those bonds. Nothing printed it, so nothing noticed.
    // Counted here so the contradiction is an observable rather than something
    // that has to be reconstructed from written fields afterwards.
    label releasedOnBond = 0;
    label bondSeeds = 0;
    forAll(Usbf, patchi)
    {
        if (!isA<clotWallAdhesionFvPatchVectorField>(Usbf[patchi]))
        {
            continue;
        }
        const clotWallAdhesionFvPatchVectorField& bond =
            refCast<const clotWallAdhesionFvPatchVectorField>(Usbf[patchi]);
        const scalarField& bonded = bond.bonded();
        const labelUList& faceCells = Usbf[patchi].patch().faceCells();
        forAll(faceCells, facei)
        {
            if (bonded[facei] < 0.5)
            {
                continue;
            }
            const label celli = faceCells[facei];
            if (indIn[celli] > indSupportLimit)
            {
                ++bondSeeds;
            }
            if (rel[celli] > 0.5)
            {
                ++releasedOnBond;
            }
        }
    }

    const label total = returnReduce(releasedCells, sumOp<label>());
    label supported = 0;
    forAll(support, celli)
    {
        if (support[celli] > 0.5)
        {
            ++supported;
        }
    }
    label clotCells = 0;
    forAll(indIn, celli)
    {
        if (indIn[celli] > indClotLimit)
        {
            ++clotCells;
        }
    }
    reduce(supported, sumOp<label>());
    reduce(clotCells, sumOp<label>());
    reduce(bondSeeds, sumOp<label>());
    reduce(releasedOnBond, sumOp<label>());

    // UNCONDITIONAL. The previous version printed only when something was
    // already released, so the approach to the failure -- the seed count
    // falling 26, 25, 23, 21, 20 while supported stayed at the full body --
    // was invisible, and the step that snapped could not be distinguished
    // from a gradual one.
    Info<< "PHASE_CLOT_RELEASED time=" << runTime.userTimeValue()
        << " cells=" << total
        << " volume=" << fvc::domainIntegrate(released_).value()
        << " clotCells=" << clotCells
        << " supported=" << supported
        << " bondSeeds=" << bondSeeds
        << " releasedOnBond=" << releasedOnBond
        << " coreExempted=" << coreExempted
        << " supportLimit=" << supportLimit
        << " indSupportLimit=" << indSupportLimit
        << " indBasis=content"
        << endl;
}


void Foam::solvers::phaseClotAxisymmetricFluid::updateBondHold()
{
    // THE ANCHOR THE EULERIAN PATH NEVER HAD.
    //
    // Cleared first and unconditionally, so a case that switches the hold off
    // cannot inherit a stale field, and so the written field is honest about
    // there being no anchor when there is none.
    bondHold_ = dimensionedScalar("zero", dimless, 0.0);

    if (bondHoldDilation_ < 0)
    {
        return;
    }

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    // S5: same substitution as updateReleasedIndicator(); see the note there.
    const volScalarField ind(phaseIndicator());
    const scalarField& indIn = ind.primitiveField();
    const scalar indClotLimit = 0.5;
    scalarField& h = bondHold_.primitiveFieldRef();

    // Seed on the wall cells of INTACT bonds that actually carry material.
    // An intact bond under bare blood anchors nothing.
    label seeds = 0;
    const volVectorField::Boundary& Usbf = Us_.boundaryField();
    forAll(Usbf, patchi)
    {
        if (!isA<clotWallAdhesionFvPatchVectorField>(Usbf[patchi]))
        {
            continue;
        }
        const clotWallAdhesionFvPatchVectorField& bond =
            refCast<const clotWallAdhesionFvPatchVectorField>(Usbf[patchi]);
        const scalarField& bonded = bond.bonded();
        const labelUList& faceCells = Usbf[patchi].patch().faceCells();
        forAll(faceCells, facei)
        {
            const label celli = faceCells[facei];
            if (bonded[facei] > 0.5 && indIn[celli] > indClotLimit)
            {
                h[celli] = 1.0;
                ++seeds;
            }
        }
    }

    // Grow a FIXED number of cells through material.
    //
    // A fixed count, not a grow-through-everything like the release mask,
    // and the difference matters. The mask's boundary must land where grad(c)
    // is negligible because the advecting velocity JUMPS across it. This
    // boundary is the top of an anchored layer inside the clot bulk, where c
    // is close to -1 and grad(c) is negligible for exactly the opposite
    // reason -- being deep inside the material rather than far outside it.
    // Growing it through the whole body instead would pin the entire clot in
    // the lab frame, which is the objection that removed the Darcy drag from
    // this path in the first place, and would make a deforming clot
    // impossible to distinguish from a rigid one.
    // THE WEIGHT CARRIED OUTWARD IS THE TAPER, so the ramp is built by the
    // same sweep that builds the layer and costs nothing extra. With the taper
    // off, every sweep carries 1.0 and this reduces EXACTLY to the binary
    // dilation that ran before 2026-08-25.
    //
    // With it on, sweep k writes (D - k - 1)/D, so the seed row keeps 1.0 and
    // the outermost sweep reaches 0. There is no cell boundary across which
    // the hold jumps, which is the whole point: the severed core Li rejected
    // was material shearing along exactly such a jump.
    for (label sweep = 0; sweep < bondHoldDilation_; ++sweep)
    {
        const scalar w =
            1.0;
        const scalarField prev(h);
        forAll(owner, facei)
        {
            const label own = owner[facei];
            const label nei = neighbour[facei];
            if (indIn[own] > indClotLimit && prev[nei] > 0 && h[own] <= 0)
            {
                h[own] = w;
            }
            if (indIn[nei] > indClotLimit && prev[own] > 0 && h[nei] <= 0)
            {
                h[nei] = w;
            }
        }
    }
    bondHold_.correctBoundaryConditions();

    // Both are reported because with a taper they differ: heldCells counts
    // every cell carrying ANY hold, heldWeight sums the weight and is what the
    // penalty actually applies. A taper that has collapsed to nothing shows up
    // as heldWeight falling while heldCells does not move.
    label held = 0;
    scalar heldWeight = 0;
    forAll(h, celli)
    {
        if (h[celli] > 0)
        {
            ++held;
            heldWeight += h[celli];
        }
    }
    reduce(held, sumOp<label>());
    reduce(heldWeight, sumOp<scalar>());
    reduce(seeds, sumOp<label>());

    Info<< "PHASE_CLOT_BONDHOLD time=" << runTime.userTimeValue()
        << " seeds=" << seeds
        << " heldCells=" << held
        << " heldWeight=" << heldWeight
        << " taper=" << 0
        << " dilation=" << bondHoldDilation_
        << " penalty=" << bondHoldPenalty_
        << " carrier=" << 0
        << endl;
}


void Foam::solvers::phaseClotAxisymmetricFluid::reportWoundState()
{
    // WHAT IS STILL STANDING ON THE WOUND.
    //
    // The question this whole stage is about -- does the clot stay on the
    // injury it grew on -- had NO instrument. It was answered after the fact
    // by reading written fields offline, which is how it went unnoticed for
    // the whole Eulerian effort. Measured on run 20260824-220555, the clot
    // over the wound fell from 1093 cells at 1 ms to 6 at 12 ms and its height
    // collapsed from 34 cells to 2 between 6 and 7 ms, with every one of the
    // forty bonds intact and adhesiveStrength at 1e9 Pa. None of that appeared
    // in any log line.
    //
    // THE COLUMN WALK IS MESH-AGNOSTIC. Height is counted by stepping from the
    // wall cell to the face-neighbour whose centre displacement is most
    // aligned with the inward normal, so it does not assume a structured grid
    // and does not break if the mesh is ever graded or refined.
    const volVectorField::Boundary& Usbf = Us_.boundaryField();
    // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
    // so every threshold below is the same threshold restated:
    // c < L becomes ind > (1 - L)/2, and the inequality turns
    // over with it. Same cells, named in the surviving field.
    const volScalarField indHold(phaseIndicator());
    const scalarField& indIn = indHold.primitiveField();
    const vectorField& cc = mesh.C();
    const cellList& meshCells = mesh.cells();
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    const label nInt = mesh.nInternalFaces();

    label injuryFaces = 0;
    label clotOnInjury = 0;
    label maxHeight = 0;
    scalar maxHeightX = 0;
    label totalHeight = 0;
    scalar xMin = great;
    scalar xMax = -great;
    scalar xMinAny = great;
    scalar xMaxAny = -great;
    bool sawBond = false;

    forAll(Usbf, patchi)
    {
        if (!isA<clotWallAdhesionFvPatchVectorField>(Usbf[patchi]))
        {
            continue;
        }
        sawBond = true;
        const clotWallAdhesionFvPatchVectorField& bond =
            refCast<const clotWallAdhesionFvPatchVectorField>(Usbf[patchi]);
        const scalarField& injury = bond.injury();
        const fvPatch& patch = Usbf[patchi].patch();
        const labelUList& faceCells = patch.faceCells();
        const vectorField nHat(patch.nf());

        forAll(faceCells, facei)
        {
            const label wallCell = faceCells[facei];

            // Any wall cell carrying clot, injured or not -- this is how
            // material walking onto intact endothelium becomes visible.
            if (indIn[wallCell] > 0.5)
            {
                xMinAny = min(xMinAny, cc[wallCell].x());
                xMaxAny = max(xMaxAny, cc[wallCell].x());
            }

            if (injury[facei] < 0.5)
            {
                continue;
            }
            ++injuryFaces;

            if (indIn[wallCell] <= 0.5)
            {
                continue;
            }
            ++clotOnInjury;
            xMin = min(xMin, cc[wallCell].x());
            xMax = max(xMax, cc[wallCell].x());

            // Walk inward while the material lasts.
            const vector inward(-nHat[facei]);
            label celli = wallCell;
            label height = 0;
            label prev = -1;
            while (celli >= 0 && indIn[celli] > 0.5 && height < 10000)
            {
                ++height;
                const labelList& cFaces = meshCells[celli];
                label best = -1;
                scalar bestAlign = 0.3;   // must be genuinely inward
                forAll(cFaces, k)
                {
                    const label facej = cFaces[k];
                    if (facej >= nInt)
                    {
                        continue;
                    }
                    const label other =
                        (owner[facej] == celli) ? neighbour[facej]
                                                : owner[facej];
                    if (other == prev)
                    {
                        continue;
                    }
                    vector d(cc[other] - cc[celli]);
                    const scalar dm = mag(d);
                    if (dm < vSmall)
                    {
                        continue;
                    }
                    const scalar align = (d/dm) & inward;
                    if (align > bestAlign)
                    {
                        bestAlign = align;
                        best = other;
                    }
                }
                prev = celli;
                celli = best;
            }
            // maxHeightX is recorded with maxHeight because the pair is what
            // makes the number readable. maxHeight is a MAX OVER A WINDOW
            // WITH A HARD EDGE, so it steps discontinuously when the tallest
            // column advects past the last injury face while the clot itself
            // does nothing. Measured, run 20260901-110348 child g40: at
            // 18.6339 ms maxHeight fell 31 -> 14 in one step while clotCells
            // went 2585 -> 2584, clotVolume did not change, and every other
            // diagnostic was smooth. The 31 was real -- a wall-connected
            // column at x = 93.9 um, on the downstream ramp of the tail --
            // but it is not the mural clot, which was 12-13 cells across
            // x = 59-94 um. Reporting where the max sits makes that legible
            // instead of looking like a collapse.
            //
            // meanHeight below (= cellsOverInjury/clotOnInjury) is the
            // CONTINUOUS instrument and is what a height trend should be read
            // from; it moves smoothly through the same event.
            if (height > maxHeight)
            {
                maxHeight = height;
                maxHeightX = cc[wallCell].x();
            }
            totalHeight += height;
        }
    }

    if (!sawBond)
    {
        return;
    }

    reduce(injuryFaces, sumOp<label>());
    reduce(clotOnInjury, sumOp<label>());
    reduce(totalHeight, sumOp<label>());

    // Two-step so the x travels with the winning height across ranks.
    const label maxHeightAll = returnReduce(maxHeight, maxOp<label>());
    scalar maxHeightXAll =
        (maxHeight == maxHeightAll && maxHeight > 0) ? maxHeightX : great;
    reduce(maxHeightXAll, minOp<scalar>());
    if (maxHeightAll == 0)
    {
        maxHeightXAll = 0;
    }
    maxHeight = maxHeightAll;
    maxHeightX = maxHeightXAll;
    reduce(xMin, minOp<scalar>());
    reduce(xMax, maxOp<scalar>());
    reduce(xMinAny, minOp<scalar>());
    reduce(xMaxAny, maxOp<scalar>());

    // Emitted with a sentinel rather than omitted when the wound is bare, so
    // a parser cannot confuse "no clot left on the injury" with "the run
    // stopped printing".
    if (clotOnInjury == 0)
    {
        xMin = 0;
        xMax = 0;
    }
    if (xMinAny > great/2)
    {
        xMinAny = 0;
        xMaxAny = 0;
    }

    Info<< "PHASE_CLOT_WOUND time=" << runTime.userTimeValue()
        << " injuryFaces=" << injuryFaces
        << " clotOnInjury=" << clotOnInjury
        << " coverage="
        << (injuryFaces > 0 ? scalar(clotOnInjury)/injuryFaces : 0.0)
        << " maxHeight=" << maxHeight
        << " maxHeightX=" << maxHeightX
        << " meanHeight="
        << (clotOnInjury > 0 ? scalar(totalHeight)/clotOnInjury : 0.0)
        << " cellsOverInjury=" << totalHeight
        << " xMin=" << xMin
        << " xMax=" << xMax
        << " wallClotXMin=" << xMinAny
        << " wallClotXMax=" << xMaxAny
        << endl;
}


// Capture the face weights the c equation is about to use, so the Wp equation
// can assemble the SAME matrix. Derived from the div(phiReleased,c) entry
// itself rather than from a second scheme string: a duplicate string is the
// desynchronisation this exists to prevent.
//
// The entry is "bounded Gauss <limiter>". `bounded` is a convectionScheme
// wrapper and `Gauss` is the convection scheme; the INTERPOLATION scheme is
// what follows, and that is what owns the weights. The two leading tokens are
// consumed and checked rather than assumed.
Foam::tmp<Foam::surfaceScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::phaseFieldFaceWeights
(
    const surfaceScalarField& carrier
) const
{
    ITstream& cStream = mesh_.schemes().div("div(phiReleased,c)");
    cStream.rewind();

    const word wrapper(cStream);
    const word gauss(cStream);

    if (wrapper != "bounded" || gauss != "Gauss")
    {
        FatalErrorInFunction
            << "plateletContentFormulation captures the phase field's face "
            << "weights from div(phiReleased,c), which must be spelled "
            << "`bounded Gauss <limiter>`." << nl
            << "  found: " << wrapper << ' ' << gauss << " ..." << nl
            << "The Wp equation reuses these weights so that Wp = q is "
            << "preserved by exact arithmetic. If the c entry changes shape "
            << "this capture must be updated with it, deliberately, rather "
            << "than silently reading the wrong tokens."
            << exit(FatalError);
    }

    tmp<surfaceInterpolationScheme<scalar>> tinterp
    (
        surfaceInterpolationScheme<scalar>::New(mesh_, carrier, cStream)
    );
    cStream.rewind();

    // S6b-1: THE FIELD THE LIMITER IS EVALUATED ON, per phaseLimiterBasis_.
    // Named once here and used for the weights, for the more-upwind override
    // and for the interface counter, so the three cannot describe different
    // fields. The ratio is UNCLAMPED on purpose -- see the header.
    // S6c-2: phaseLimiterBasis 'legacy' is refused at construction, so this
    // is true on every admissible dictionary. Kept as a named constant
    // rather than folded away so the interface band below still reads as
    // the indicator's band and not as a pair of magic numbers.

    // S6c-2: ONE BRANCH LEFT. The legacy branch bound c_ ITSELF, not a
    // renamed copy, and that was deliberate: the vanLeer limiter computes
    // fvc::grad(vf) to form r, and fvc::grad looks its scheme up BY THE
    // FIELD'S NAME, so a copy called "phaseLimiterField" would resolve
    // grad(phaseLimiterField) rather than grad(c). With c deleted there is
    // no field to bind and phaseLimiterBasis 'legacy' is REFUSED at
    // construction. The name below still matters for the same reason: it is
    // what gradSchemes is looked up under.
    const volScalarField limiterField
    (
        "phaseLimiterField",
        (thetaC_ + thetaH_)
       /dimensionedScalar("thetaRef", dimless, skeletonVolumeFraction_)
    );

    tmp<surfaceScalarField> tw
    (
        new surfaceScalarField
        (
            "phaseWeights",
            tinterp().weights(limiterField)
        )
    );

    // THE MORE UPWIND OF THE TWO LIMITERS.
    //
    // WHAT WENT WRONG. Wp rides c's vanLeer weights so that Wp = q survives
    // to machine precision, and a limiter is TVD only for the field it was
    // BUILT FROM. Run 20260901-200203 measured the consequence: in the
    // advected tail of g30 at 27 ms, along y = 20.36 um,
    //
    //   w x1000   610   581   443   252    68   -30    -4    97   248
    //   q x1000  0.147 0.193 0.269 0.398 0.623 1.035 1.836 3.443 6.662
    //
    // c changes by ~1e-03 per cell across that row while w changes by ~0.2
    // per cell. c is essentially UNIFORM exactly where w has a near
    // discontinuity, so vanLeer evaluated on c returns psi -> 2 -- pure
    // DOWNWIND -- and that face value is then applied to a steep Wp front.
    // The result is the classic dispersive undershoot of a downwind-biased
    // scheme: a smooth, coherent dip that goes NEGATIVE at the trough.
    // Measured min(Wp) = -5.32e-05 in g30 from 25 ms and -9.99e-07 in g40
    // from 29 ms. It is not a checkerboard and not a blow-up -- max(Wp) was
    // clean to 1.00000001 -- but a negative platelet content is not physical
    // and the same mechanism gets worse as fronts steepen.
    //
    // THE FIX. Take, face by face, whichever of the two limiters is more
    // upwind. Both weight fields come from the same blend,
    // w = L*CD + (1 - L)*pos0(flux), so "more upwind" is "smaller L", which
    // on a face with flux >= 0 is the LARGER weight and on a face with
    // flux < 0 the SMALLER one. No limiter needs to be recovered.
    //
    // WHY THIS IS TVD FOR Wp. The Sweby region is 0 <= psi <= min(2r, 2).
    // vanLeer evaluated on Wp satisfies it for Wp by construction, and the
    // choice above is never LESS upwind than that, so it stays inside the
    // region. psi_c >= 0 keeps the lower bound.
    //
    // WHY Wp = q STILL SURVIVES. The vanLeer limiter is invariant under an
    // affine change of the transported field: for phi = a*c + b both the
    // numerator and the denominator of r scale by a, so r_Wp == r_c exactly
    // when Wp == q. The two weight fields are then the same field and the
    // choice is a no-op. In floating point they differ by rounding, which is
    // what the tolerance below exists for.
    //
    // THE TOLERANCE, DERIVED. Gate A-exact holds |Wp - q| <= 1.0e-11. A
    // perturbation eps of the transported field moves r by ~2*eps/|phi_N -
    // phi_P| and moves L by at most about twice that. With the interface
    // differences this equation actually carries, |phi_N - phi_P| ~ 1e-01 to
    // 1e-03, so the two limiters can disagree by as much as ~4e-08 while
    // still describing the SAME field. 1e-06 sits two decades above that and
    // six decades below the O(1) disagreement measured in the tail, so the
    // override cannot fire on rounding and cannot fail to fire on the defect.
    // Below the tolerance c's weight is kept UNCHANGED, so on a gate-A case
    // the assembled matrix is bit-identical to today's and the 5.85e-12
    // result is preserved by construction rather than by hope.
    const scalar limiterAgreementTol = 1.0e-06;

    const surfaceScalarField wWp("plateletWeights", tinterp().weights(Wp_));
    surfaceScalarField& wc = tw.ref();

    label overridden = 0;
    // WHERE the override fires, not just how often. Run 20260902-104340 found
    // it engaging on 1600 faces at step 1 with Wp == q to the last bit, which
    // can only be the floating-point conditioning of q = 0.5*(c+1) deep in the
    // clot -- and there both fields are flat, so an upwind-biased weight
    // changes the face value by nothing. THAT ARGUMENT IS NOW MEASURED: this
    // counts how many of the overridden faces touch the diffuse interface,
    // where a weight change does move material.
    label overriddenAtInterface = 0;

    scalarField& wcIn = wc.primitiveFieldRef();
    const scalarField& wWpIn = wWp.primitiveField();
    const scalarField& carIn = carrier.primitiveField();
    const scalarField& limCellIn = limiterField.primitiveField();
    // THE SAME BIN IN BOTH SPELLINGS. The legacy test is |c| < 0.9; with
    // c = 2q - 1 that is 0.05 < q < 0.95, and after S6a q = 1 - ind, so the
    // content spelling of the identical bin is 0.05 < ind < 0.95.
    const scalar interfaceLo = 0.05;
    const scalar interfaceHi = 0.95;
    const labelUList& limOwn = mesh_.owner();
    const labelUList& limNei = mesh_.neighbour();
    forAll(wcIn, facei)
    {
        if (mag(wWpIn[facei] - wcIn[facei]) <= limiterAgreementTol)
        {
            continue;
        }
        const scalar chosen =
            (carIn[facei] >= 0)
          ? max(wcIn[facei], wWpIn[facei])
          : min(wcIn[facei], wWpIn[facei]);
        if (chosen != wcIn[facei])
        {
            ++overridden;
            const scalar limOwnV = limCellIn[limOwn[facei]];
            const scalar limNeiV = limCellIn[limNei[facei]];
            if (
                (limOwnV > interfaceLo && limOwnV < interfaceHi)
             || (limNeiV > interfaceLo && limNeiV < interfaceHi)
            )
            {
                ++overriddenAtInterface;
            }
            wcIn[facei] = chosen;
        }
    }

    surfaceScalarField::Boundary& wcBf = wc.boundaryFieldRef();
    const surfaceScalarField::Boundary& wWpBf = wWp.boundaryField();
    const surfaceScalarField::Boundary& carBf = carrier.boundaryField();
    forAll(wcBf, patchi)
    {
        scalarField& pwc = wcBf[patchi];
        const scalarField& pwWp = wWpBf[patchi];
        const scalarField& pcar = carBf[patchi];
        forAll(pwc, i)
        {
            if (mag(pwWp[i] - pwc[i]) <= limiterAgreementTol)
            {
                continue;
            }
            const scalar chosen =
                (pcar[i] >= 0) ? max(pwc[i], pwWp[i]) : min(pwc[i], pwWp[i]);
            if (chosen != pwc[i])
            {
                ++overridden;
                pwc[i] = chosen;
            }
        }
    }

    reduce(overridden, sumOp<label>());
    reduce(overriddenAtInterface, sumOp<label>());

    Info<< "PHASE_CLOT_PLATELET_LIM time=" << mesh_.time().userTimeValue()
        << " facesUpwinded=" << overridden
        << " facesUpwindedAtInterface=" << overriddenAtInterface
        << " tol=" << limiterAgreementTol
        << " limiterBasis=content"
        << endl;

    return tw;
}


void Foam::solvers::phaseClotAxisymmetricFluid::beginPlateletOuterTransaction()
{
    // S16 C: the partition is checkpointed beside the content it partitions
    // (red-team Change 4), and restored at both roll-backs below.
    {
        thetaIntactStepAccepted_ = tmp<volScalarField>
        (
            new volScalarField("thetaIntactStepAccepted", thetaIntact_)
        );
        thetaPendingStepAccepted_ = tmp<volScalarField>
        (
            new volScalarField("thetaPendingStepAccepted", thetaPending_)
        );
        thetaBrokenStepAccepted_ = tmp<volScalarField>
        (
            new volScalarField("thetaBrokenStepAccepted", thetaBroken_)
        );
    }
    // S17 A: and the pending fibre beside it (design section 5).
    {
        thetaPendingBStepAccepted_ = tmp<volSymmTensorField>
        (
            new volSymmTensorField("thetaPendingBStepAccepted", thetaPendingB_)
        );
        thetaPendingPStepAccepted_ = tmp<volScalarField>
        (
            new volScalarField("thetaPendingPStepAccepted", thetaPendingP_)
        );
        pendingAxisStepAccepted_ = tmp<volVectorField>
        (
            new volVectorField("pendingAxisStepAccepted", pendingAxis_)
        );
        pendingStretchStepAccepted_ = tmp<volScalarField>
        (
            new volScalarField("pendingStretchStepAccepted", pendingStretch_)
        );
    }
    // S17b A1: and the taut share (V1).
    {
        thetaTautStepAccepted_ = tmp<volScalarField>
        (
            new volScalarField("thetaTautStepAccepted", thetaTaut_)
        );
    }

    thetaCStepAccepted_ = tmp<volScalarField>
    (
        new volScalarField("thetaCStepAccepted", thetaC_)
    );
    thetaHStepAccepted_ = tmp<volScalarField>
    (
        new volScalarField("thetaHStepAccepted", thetaH_)
    );
    thetaNetRefStepAccepted_ = tmp<volScalarField>
    (
        new volScalarField("thetaNetRefStepAccepted", thetaNetRef_)
    );
    // D74 (design section 2, red team 36 change 3): ref_I is checkpointed
    // beside ref, and its label balance starts here.
    {
        thetaNetRefIntactStepAccepted_ = tmp<volScalarField>
        (
            new volScalarField
            (
                "thetaNetRefIntactStepAccepted", thetaNetRefIntact_
            )
        );
        const scalarField& VI = mesh_.V();
        refISumStepStart_ =
            gSum(thetaNetRefIntact_.primitiveField()*VI);
        refITransferStep_ = 0;
        refIBoundaryFluxStep_ = 0;
        refISpSourceStep_ = 0;
    }
    nPStepAccepted_ = tmp<volScalarField>
    (
        new volScalarField("nPStepAccepted", nP_)
    );
}


void Foam::solvers::phaseClotAxisymmetricFluid::solvePhaseField()
{
    const phaseClotScopeTimer costGuard_(tPhaseField_);

    // CLEARED FIRST, ALWAYS. If the branch taken below does not advect c, the
    // carrier must be absent rather than left over from the previous step --
    // solvePlatelets() reads validity to decide whether to advect W at all,
    // and a stale carrier would silently advect W by a flux c never saw. That
    // is the same class of defect this whole change is repairing.
    phaseCarrier_.clear();
    phaseWeights_.clear();
    // Same contract, one equation further on: if solveContentComponents()
    // does not run this step, the momentum flux must find this ABSENT and
    // abort rather than advect skeleton momentum with last step's weights.
    contentFaceWeights_.clear();
    contentFctLambda_.clear();
    // OI-59: and the content flux those weights carried through the inlet
    // and outlet. A step that does not run solveContentComponents() books
    // no outflux, as the balance did when it found phaseCarrier_ absent.
    contentBoundaryFlux_ = 0;

    // q AS IT STANDS NOW IS q^n: this runs before c is advanced, and the
    // updateDerivedFields() immediately preceding solvePhaseField() built q_
    // from c^n. Captured here so the Wp equation's explicit diffusion source
    // can be evaluated at one consistent time level.
    qOldPlatelet_ = tmp<volScalarField>
    (
        new volScalarField("qOldPlatelet", q_)
    );

    // THE MATERIAL FLUX OF THE MIXTURE, when the case asks for it.
    //
    //     v = chi*v_s + q*u,   v_s = (Us - UsPrev)/dt
    //
    // the volume-fraction weighted velocity of the two phases. Inside the clot
    // chi -> 1 and this is the skeleton velocity; in blood q -> 1 and it is
    // the fluid velocity; across the interface it interpolates continuously.
    // No threshold, no branch, and no new parameter.
    //
    // Without it the clot's phase field cannot move at all, because the fluid
    // flux it was advected by is driven to zero inside the clot by the
    // Brinkman penalty (measured: 2.18e-7 against 2.28e-3 m/s). A deforming
    // clot stayed exactly where it was, and the only way material could ever
    // leave was the damage variable erasing it -- which is why showing a tear
    // had always required a Gc far below the measured one. Source 33 has no
    // toughness at all; their shell is dragged off by the flow.
    //
    // v_s is the same skeleton velocity the momentum equation uses for the
    // Darcy exchange, so the phase field and the flow cannot disagree about
    // how fast the solid is moving.
    tmp<surfaceScalarField> tphiMaterial;
    {
        // THE CLOT RIDES ITS OWN SKELETON. This is the change the whole
        // two-velocity path exists for. Previously the material flux WAS the
        // fluid flux, so the clot was carried at the seepage velocity -- fluid
        // percolating through a porous solid, mistaken for the solid moving.
        //
        // AND IT RIDES IT ONLY WHERE IT HAS ONE. phiSkelOwn_ is defined on
        // every face in the domain, but in the lumen it is the skeleton row
        // of a block that is nearly singular there -- run 20260908-010836
        // measured |Vs| = 0.0386 m/s, 110% of max|U|, in a cell with
        // chi = 4.03e-11. Carrying c and shell on that is what killed all 14
        // two-velocity runs on record. See materialCarrierBlend_ in the
        // header for the full 23-run table and the two failure modes.
        const surfaceScalarField phiS(skeletonFlux());

        {
            tphiMaterial = tmp<surfaceScalarField>
            (
                new surfaceScalarField("phiMaterial", phiS)
            );
        }
    }

    // IMMUTABLE ACCEPTED START-OF-STEP STATE FOR EVERY PHASE TRIAL.
    // Run 20260910-222019 proved why relying on fvm::ddt alone is wrong: on
    // the no-carrier branch solveContentComponents() is skipped, so thetaH
    // retained the previous trial's deposit while nP was reconstructed. The
    // result gained exactly one extra Y*dB per extra phase corrector.
    // beginPlateletOuterTransaction() owns allocation. Keeping that capture
    // outside this function is load-bearing: this function may later execute
    // once per outer, but the accepted state must still be captured once per
    // physical timestep.

    // ---- THE STEFAN FRONT IS GONE -------------------------------------
    //
    // S4 moved growth onto phaseGrowthDriver 'transfer' and S6c-2 deleted the
    // 'front' implementation with c: the source was Vn_*phaseGradient_ and
    // both the c equation it sourced and phaseGradient_ itself no longer
    // exist. phaseGrowthDriver 'front' is REFUSED at construction rather
    // than silently treated as 'transfer'.

    // S25 B: the deposit booked into e is reset per step too, for the reason
    // given for transferClotGrowth_ below.
    plateletDepositStep_ = 0;

    // Reset per step, not per corrector: solvePlateletContent() overwrites it
    // on every corrector it runs, and a step whose transfer never ran must
    // report zero growth rather than the previous step's.
    transferClotGrowth_ = 0;

    for (label corr = 0; corr < nPhaseCorrectors_; ++corr)
    {
        {
            if
            (
                !thetaCStepAccepted_.valid()
             || !thetaHStepAccepted_.valid()
             || !nPStepAccepted_.valid()
            )
            {
                FatalErrorInFunction
                    << "platelet-content trial has no accepted timestep "
                    << "checkpoint" << exit(FatalError);
            }

            thetaC_ = thetaCStepAccepted_();
            thetaH_ = thetaHStepAccepted_();
            nP_ = nPStepAccepted_();
            thetaC_.correctBoundaryConditions();
            thetaH_.correctBoundaryConditions();
            nP_.correctBoundaryConditions();
            // S10: the reference rolls back with the content it references.
            if (thetaNetRefStepAccepted_.valid())
            {
                thetaNetRef_ = thetaNetRefStepAccepted_();
                thetaNetRef_.correctBoundaryConditions();
            }
            // D74: and the intact reference with it, or it is transported
            // once per corrector. The roll-back discards every transfer
            // since the checkpoint, so the balance's terms restart too.
            {
                if (!thetaNetRefIntactStepAccepted_.valid())
                {
                    FatalErrorInFunction
                        << "thetaNetRefIntact (D74) has no accepted timestep "
                        << "checkpoint at the corrector roll-back."
                        << exit(FatalError);
                }
                thetaNetRefIntact_ = thetaNetRefIntactStepAccepted_();
                thetaNetRefIntact_.correctBoundaryConditions();
                refITransferStep_ = 0;
                refIBoundaryFluxStep_ = 0;
                refISpSourceStep_ = 0;
            }
            // S16 C: and so does the damage partition (Change 4).
            restoreDamagePartition("corrector roll-back");
        }

        // partial_t(c) - div(M grad(mu_c)) = -Vn*G_c.
        // An fvMatrix is associated with exactly one solved field. muC_ is
        // updated from the current Picard iterate above, so its diffusion is
        // an explicit source in the c_ equation rather than an incompatible
        // fvMatrix for a second unknown.
        // S10 FIX 2: under releasedCarrier "material" the rigid-body branch is
        // not taken, and released material falls through to the attached
        // branch below: the SOLVED skeleton flux carries it, exactly as it
        // carries attached material, and the bond-hold factor still pins
        // whatever is bonded. The two branches perform the same five
        // operations and differ only in the carrier. Release then changes
        // what it physically should -- the wall bond -- and nothing about
        // how the material moves. The comments below describe the
        // one-velocity history that made a kinematic carrier necessary: there
        // was no skeleton velocity then, and on the two-velocity path there
        // is. Run 20260922-104613 measured Ubody at 13.7-14.8 mm/s (a band
        // average 58-71% throat by volume) while the solved Vs said the freed
        // clot was ROLLING at ~6 mm/s.
        if (tphiMaterial.valid())
        {
            // Attached, deforming, and now able to move: the phase field is
            // carried by the material velocity even though nothing has been
            // released. This is the case source 33 runs -- their clot never
            // detaches at the wall, and everything interesting happens in the
            // body being deformed and dragged.
            surfaceScalarField phiAttached
            (
                "phiAttached",
                tphiMaterial()
            );

            phaseCarrier_ = tmp<surfaceScalarField>
            (
                new surfaceScalarField("phaseCarrier", phiAttached)
            );
            phaseWeights_ = phaseFieldFaceWeights(phiAttached);

            advectMaterialLabel(phiAttached);
            // Same carrier as the label and the phase field, so the
            // Stage 1 equivalence test compares like with like.
            solveContentComponents(phiAttached);
        }
        else
        {
            // NO CARRIER ON THIS PATH, SO THE CONTENT IS NOT ADVECTED -- AND
            // A FROZEN theta MUST NOT BE READ AS AN EQUIVALENCE RESULT.
            //
            // c relaxes here with no transport at all, so theta would sit at
            // its seeded value while chi moved, and equivMax would grow for a
            // reason that has nothing to do with the operator under test.
            // Nothing downstream would notice: the record stream would simply
            // be shorter, and no gate counts records against timesteps.
            //
            // This branch was NOT taken in run 20260909-115458 -- 544
            // PHASE_CLOT_CONTENT records for 272 steps at two correctors
            // each, so every corrector took one of the two carrier branches
            // -- which is why this is a report rather than a FatalError. It
            // marks the history UNASSESSABLE instead of killing a paid
            // session over a path these children do not use.
            {
                Info<< "PHASE_CLOT_CONTENT_NOT_ADVECTED time="
                    << runTime.userTimeValue()
                    << " operator=content"
                    << " reason=noCarrier" << endl;
            }
        }
        // STAGE 2. Once per phase corrector, AFTER the contents are solved,
        // so it sees this corrector's theta_star.
        //
        // The supplied carrier remains the legacy one-velocity input. On the
        // two-velocity path phi_ is the MIXTURE flux, not either declared
        // fluid convention. solvePlateletContent selects phiFluidOwn_ there
        // and converts it to the declared convention using its trial epsP_.
        // The contents ride phiReleased/phiAttached; free platelets ride the
        // fluid. The consistency residual
        // ddt(eps) + div(F_eps) is measured every step -- because the
        // continuity weighting in this solver comes from q_, not from these
        // contents, and the two stop agreeing the moment growth is on.
        //
        // Being inside the corrector loop is what makes the transfer contract
        // hold: fvm::ddt rebuilds every trial from oldTime, so corrector 2
        // REPLACES corrector 1 rather than accumulating a second reaction.
        solvePlateletContent(phi_, corr);

        // THE CLIP, AND ITS COST. Default OFF; see clipPhaseField_ in the
        // header for why this is a MEASUREMENT and not a repair.
        //
        // It sits inside the corrector loop so the next iteration -- and
        // every field derived from c -- reads a bounded value. The pre-clip
        // extremes are captured BEFORE it acts, because otherwise
    }

    // A COMMIT IS EXPLICIT, NEVER INFERRED FROM THE LAST FLOATING-POINT TIME.
    // The default path still commits on its sole first-outer visit. The
    // experimental outer path commits only its final PIMPLE trial.
    {
        {
            Info<< "PHASE_CLOT_FREEPLATELET_COMMIT"
            << " time=" << runTime.userTimeValue()
            << " step=" << runTime.timeIndex()
            << " attempt=" << plateletAttempt_
            << " outer=" << 0
            << " phase=" << (nPhaseCorrectors_ - 1)
            << " trials=" << nPhaseCorrectors_
            << " committed=1"
            << " convergence=fixedPhaseCount"
            << endl;
        }
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::solveContentComponents
(
    const surfaceScalarField& carrier
)
{
    // CONSERVATIVE, NOT BOUNDED, AND NOT THE MATERIAL FORM.
    //
    // A volume fraction is content PER TOTAL VOLUME, so the balance is
    // div(carrier*theta) and the divergence form is the correct one even
    // though the solid carrier is compressible. OpenFOAM's `bounded` wrapper
    // subtracts Sp(surfaceIntegrate(carrier), f), which converts this into
    // the material derivative -- right for a label riding pathlines (shell_,
    // w), wrong for a content. The two spellings can be made algebraically
    // equivalent by carrying -theta*div(u_s) explicitly, but the discrete
    // flux, divergence and time levels all have to match, and there is no
    // reason to start with the error-prone spelling when direct conservative
    // assembly is available.
    //
    // FIRST-ORDER UPWIND, CONSTRUCTED EXPLICITLY. The weights are built here
    // rather than looked up in fvSchemes for two reasons: the scheme is part
    // of the Stage 1 contract, not a per-case choice, and a limited scheme
    // recomputes its limiter from the field being transported -- which is
    // exactly the defect that made c and shell_ assemble different matrices
    // from one flux. Both components see IDENTICAL, field-independent weights
    // here, so their fluxes sum exactly to the total solid flux and the sum
    // theta = thetaC + thetaH obeys the same equation its parts do.
    // S6c-2: contentOperator 'matched' IS REFUSED AT CONSTRUCTION, and the
    // branch is deleted rather than left unreachable. Its source was
    // 0.5*Vn*phaseGradient - 0.5*laplacian(mobility, muC) mapped into
    // content currency, and the mobility/muC half of that no longer exists.
    // Unreachable code that cannot compile is not dead code.

    // S10 FIX 3: ONE SET OF FACE WEIGHTS, FOR EVERY CONSUMER.
    //
    // "upwind" (the shipped default) is pos0(faceFlux), unchanged. It smears
    // a translated profile at dSigma^2 = dx*(1 - Co) per unit distance: run
    // 20260922-104613 measured 39.7 and 39.4 um^2 on the rear and front edges
    // over 57 um, against 40.7 predicted, and the top edge -- which the
    // carrier runs along, not across -- gained only 6.0.
    //
    // "limitedLinear" is OpenFOAM's limitedLinear 1: limiter min(max(2r,0),1)
    // in [0,1], so every face weight is a CONVEX combination of the linear
    // and upwind weights and every component's face value lies between its
    // two cell values.
    //
    // RETRACTED 2026-09-23 -- THAT IS NOT ENOUGH, AND A RUN SAID SO. Bounded
    // FACE values do not make the CELL update non-negative. Assembled
    // implicitly, any weight below the upwind one puts a positive
    // off-diagonal in the matrix; the matrix is then not an M-matrix and its
    // solve can return negatives. Run 20260923-012704 (em51) died on its
    // SECOND step, the first with a nonzero carrier: thetaCNeg 2172,
    // thetaHNeg 24, theta min -4.47e-36, and the platelet kernel refused
    // cell 258 at -3.7e-251. That cell was seeded with EXACTLY zero content,
    // 86 um downstream of the clot. Upwind cannot do this: em49 logged 66400
    // content records with zero negative cells. The offline pricing below
    // was explicit, 1-D and single-field, and so structurally blind to it.
    // limitedLinear stays selectable only so the failure can be reproduced.
    // The positivity-preserving replacement is contentTransportScheme fct
    // (upwind predictor + Zalesak-limited correction, fctLimiter()) -- see
    // research/runs/20260923-012704-em51-*.md.
    //
    // Priced offline on the measured
    // profile over the same 57 um at Co 0.00295: upwind 40.10 um^2,
    // limitedLinear 0.79, bounded, mass error 4e-16.
    //
    // THE LIMITER IS COMPUTED ONCE, FROM THE TOTAL theta, AND NEVER FROM A
    // COMPONENT. A limited scheme builds its limiter from the field it
    // transports, and letting thetaC and thetaH (and thetaNetRef_) each do
    // that would give three matrices from one flux -- the defect that cost run
    // 20260822-103655 (|Wp - q| 2.27e-04 under vanLeer against 5.85e-12 with
    // shared weights). Shared weights keep thetaC + thetaH obeying the
    // equation its parts obey, and keep the momentum face value identical to
    // the transported one (OI-6, Rudman consistency).
    // Under "fct" the implicit matrices are UPWIND -- the M-matrix predictor --
    // and the second-order part is the limited correction applied after the
    // solves below. So "upwind" and "fct" assemble identically here.
    tmp<surfaceScalarField> tContentWeights;
    {
        tContentWeights = upwind<scalar>(mesh, carrier).weights();
    }

    // ASSEMBLED THROUGH phaseConservativeConvectionWith(), NOT A LOCAL COPY.
    // It is the conservative supplied-weights operator Wp already uses under
    // phasePlateletFluxBasis 'content', so theta and Wp are now assembled by
    // literally the same code from the same weights. Under "upwind" the
    // weights are pos0(faceFlux) and the matrix is term for term the one the
    // shipped upwind scheme assembled.

    // CAPTURED FOR THE SKELETON MOMENTUM EQUATION -- OI-6, and the reason is
    // in the header next to contentFaceWeights_. These are the face weights
    // this equation transports theta with, and phiSkelFlux must multiply
    // phiSkelOwn_ by THIS face value rather than re-interpolating the same
    // field linearly.
    //
    // A SECOND CONSTRUCTION OF upwind<scalar> IS SAFE HERE AND NOWHERE ELSE.
    // upwind::weights() is pos0(faceFlux) -- it does not read the transported
    // field at all, so this returns exactly the weights the matrix above
    // used. The identical trick on a LIMITED scheme is the vanLeer defect of
    // run 20260822-103655 (|Wp - q| 2.27e-04 against 5.85e-12), which is why
    // phaseWeights_ exists and why the matrices are left untouched here: the
    // capture writes a member and changes no assembled equation, so the
    // 'legacy' basis is byte-for-byte what it was.
    //
    // PHASE_CLOT_MMC measures the claim every step instead of asserting it.
    // WHICHEVER OPERATOR THE MATRIX ABOVE WILL USE, not always upwind. The
    // capture was unconditionally upwind while the `matched` branch assembles
    // with phaseWeights_, so under contentOperator 'matched' it would have
    // handed the momentum flux -- and now the Wp equation -- the face values
    // of an operator neither equation used, with PHASE_CLOT_MMC reading zero
    // throughout because both of its terms come from this same member. Inert
    // in every case generated today (contentOperator 'content' everywhere),
    // written correctly rather than left as a trap.
    //
    // S10: CAPTURED FROM THE SAME OBJECT THE MATRICES ARE BUILT FROM, rather
    // than rebuilt. Under "upwind" this is the pos0(faceFlux) it always was;
    // under "limitedLinear" rebuilding would recompute a limiter, which is
    // the one thing that must not happen twice.
    contentFaceWeights_ = tmp<surfaceScalarField>
    (
        new surfaceScalarField("contentFaceWeights", tContentWeights())
    );

    // THE MAPPED SOURCE, AND WHY IT IS THIS AND NOT SOMETHING ELSE.
    //
    // The legacy phase equation solved immediately above this call is
    //
    //     ddt(c) + div_w(carrier, c)
    //         = laplacian(mobility, muC) - Vn*phaseGradient
    //           + carrierDivergence(carrier)
    //
    // and chi = 0.5*(1 - c), so c = 1 - 2*chi. Substituting, and using
    // div_w(carrier, uniform 1) == surfaceIntegrate(carrier) == the
    // carrierDivergence term -- which is exactly why carrierDivergence() is
    // built from surfaceIntegrate rather than fvc::div, see its comment --
    // the two constants cancel and
    //
    //     ddt(chi) + div_w(carrier, chi)
    //         = -0.5*laplacian(mobility, muC) + 0.5*Vn*phaseGradient
    //
    // theta = theta_ref*chi, so the source scales by theta_ref and nothing
    // else. EVERY INPUT IS UNCHANGED BETWEEN THE TWO SOLVES, checked rather
    // than assumed: q_, Vn_ and phaseGradient_ are all assigned in
    // updateDerivedFields(), which runs BEFORE solvePhaseField(); muC_ is
    // assigned once inside solvePhaseField() before the c solve; and nothing
    // reassigns any of them between that solve and this call. That is the
    // same class of defect as the stale q_ this function's chiLegacy had.
    //
    // THE LAPLACIAN IS WRITTEN EVEN THOUGH IT IS ZERO HERE. mobility is 0.0
    // in every main-case child (see the header's phase-field clip note, and
    // clotDiffusion = 0 in all 272 PHASE_CLOT_MASS records of run
    // 20260909-115458), so this term contributes nothing to the case that
    // motivated the test. Omitting it would make the mapping silently
    // case-specific and would break the moment anyone gave mobility a value.
    //
    // THE SPLIT BETWEEN THE COMPONENTS FOLLOWS shell_. Only the SUM enters
    // the identity being tested, so any split summing to the whole is
    // admissible; shell_ is the choice that matches how the two contents were
    // seeded, and it is defined everywhere including where theta is zero,
    // which theta-proportional weights are not.

    // Inventory before the solve, for the conservation check below. Taken
    // from the OLD time so it is the quantity the step actually starts from.
    const scalar contentBefore =
        fvc::domainIntegrate(thetaC_.oldTime() + thetaH_.oldTime()).value();

    // The expected inventory change from the source alone, so the balance can
    // be closed rather than eyeballed. Zero in `content` mode by construction.
    // ZERO SINCE S6c-2, AND STILL REPORTED. contentOperator 'matched' was
    // the only operator that sourced theta inside this function, and it is
    // refused. The transfer that does grow theta is applied in
    // solvePlateletContent(), AFTER this call, so it is outside this solve's
    // window by construction -- which is why PHASE_CLOT_CONTENT_BALANCE
    // closes the timestep in writeBalanceDiagnostic() rather than here.
    // Kept as a named zero so a reader does not have to prove its absence.
    const scalar contentSourceDt = 0.0;

    fvScalarMatrix thetaCEqn
    (
        fvm::ddt(thetaC_)
      + phaseConservativeConvectionWith(carrier, thetaC_, tContentWeights())
    );
    thetaCEqn.solve();

    fvScalarMatrix thetaHEqn
    (
        fvm::ddt(thetaH_)
      + phaseConservativeConvectionWith(carrier, thetaH_, tContentWeights())
    );
    thetaHEqn.solve();

    thetaC_.correctBoundaryConditions();
    thetaH_.correctBoundaryConditions();

    // OI-59: THE CONTENT THIS TRANSPORT TOOK THROUGH THE INLET AND OUTLET,
    // AS THE TWO SOLVES TOOK IT. Booked here, after the solves and BEFORE
    // fctApply() below, because this is the only time level at which it
    // exists. The solves removed (internalCoeffs*theta_P - boundaryCoeffs)
    // per boundary face -- phi_b*theta_low,P^(n+1) on the zeroGradient
    // inlet and outlet -- and the FCT then corrects the outlet cells through
    // interior faces only. writeBalanceDiagnostic() used to re-interpolate
    // theta onto the patches at the end of the step and so booked
    // phi_b*theta_FCT,P instead: in BATCH 72 S23v_G4_r3 that difference,
    // dt*sum phi_b*(theta_FCT - theta_low), was 100% of the 6.6e-5 content
    // residual (docket 20261008-1507-run081155-b72-s23-validation, 02).
    //
    // Read from each component's OWN matrix in fvMatrix::flux()'s boundary
    // form, so it is the solve's flux whatever the patch type. flux() is not
    // called: it aborts unless the field is listed in fluxRequired. The
    // coefficient accessors are non-const in OpenFOAM 13 and allocate
    // nothing. ASSIGNED, NOT ACCUMULATED: each phase corrector re-solves
    // from the old time, and the last solve is the one the balance closes.
    {
        const auto solvedPatchFlux =
        [](fvScalarMatrix& eqn, const volScalarField& f, const label patchi)
        {
            const fvPatchScalarField& pf = f.boundaryField()[patchi];
            scalarField faceFlux
            (
                eqn.internalCoeffs()[patchi]*pf.patchInternalField()
            );
            if (pf.coupled())
            {
                faceFlux -=
                    eqn.boundaryCoeffs()[patchi]*pf.patchNeighbourField();
            }
            else
            {
                faceFlux -= eqn.boundaryCoeffs()[patchi];
            }
            return gSum(faceFlux);
        };

        const label contentInlet = mesh_.boundaryMesh().findIndex("inlet");
        const label contentOutlet = mesh_.boundaryMesh().findIndex("outlet");
        scalar bookedFlux = 0;
        if (contentInlet >= 0)
        {
            bookedFlux +=
                solvedPatchFlux(thetaCEqn, thetaC_, contentInlet)
              + solvedPatchFlux(thetaHEqn, thetaH_, contentInlet);
        }
        if (contentOutlet >= 0)
        {
            bookedFlux +=
                solvedPatchFlux(thetaCEqn, thetaC_, contentOutlet)
              + solvedPatchFlux(thetaHEqn, thetaH_, contentOutlet);
        }
        contentBoundaryFlux_ = bookedFlux;
    }

    // S10 FIX 1: THE DEPOSITED REFERENCE, ON THE SAME CARRIER AND WEIGHTS.
    //
    // NON-conservative: the Sp term removes thetaNetRef_*div(carrier), so
    // this field follows the material without taking up its compression.
    // Where the discrete carrier is divergence-free the two matrices are
    // identical term for term and theta - thetaNetRef_ stays exactly where
    // it started; where the network compacts theta rises and this does not.
    // Same operator spelling as phaseSharedConvection() uses for a label.
    {
        fvScalarMatrix thetaNetRefEqn
        (
            fvm::ddt(thetaNetRef_)
          + phaseConservativeConvectionWith
            (
                carrier, thetaNetRef_, tContentWeights()
            )
          - fvm::Sp(fvc::surfaceIntegrate(carrier), thetaNetRef_)
        );
        const label refIters = thetaNetRefEqn.solve().nIterations();
        thetaNetRef_.correctBoundaryConditions();

        // D74 (design 2026-10-09-D74 rev. 2 section 2, red team 36 change
        // 2): THE INTACT REFERENCE ref_I, BY ref's OWN LABEL OPERATOR -- the
        // same ddt, convection (carrier, weights) and Sp term, so where ref_I
        // = ref (d* = 0) the two systems are the same system. Solved with
        // ref's solver dictionary, built here in code (no fvSolution key, the
        // case dictionary byte for byte), at tolerance 0 with minIter =
        // maxIter = the iterations ref's solve EXECUTED this corrector, so
        // both receive the same sweeps (maxIter 0: no solve, as ref's 0).
        {
            dictionary refIControls
            (
                mesh_.solution().solverDict
                (
                    thetaNetRef_.select
                    (
                        !mesh_.schemes().steady()
                     && solutionControl::finalIteration(mesh_)
                    )
                )
            );
            refIControls.set("tolerance", scalar(0));
            refIControls.set("relTol", scalar(0));
            refIControls.set("minIter", refIters);
            refIControls.set("maxIter", refIters);
            fvScalarMatrix thetaNetRefIntactEqn
            (
                fvm::ddt(thetaNetRefIntact_)
              + phaseConservativeConvectionWith
                (
                    carrier, thetaNetRefIntact_, tContentWeights()
                )
              - fvm::Sp(fvc::surfaceIntegrate(carrier), thetaNetRefIntact_)
            );
            thetaNetRefIntactEqn.solve(refIControls);
            thetaNetRefIntact_.correctBoundaryConditions();

            // PHASE_CLOT_REFI's label balance (REPORTED, design section 4):
            // the boundary outflux this solve took through the inlet and
            // outlet, in the solve's own boundary form (as OI-59 books
            // theta's), and the Sp source dt sum s ref_I V at the solved
            // ref_I, before the FCT (conservative across internal faces).
            // ASSIGNED: the last corrector's solve is the one that stands.
            {
                scalar refIOut = 0;
                for
                (
                    const label patchi
                  : {
                        mesh_.boundaryMesh().findIndex("inlet"),
                        mesh_.boundaryMesh().findIndex("outlet")
                    }
                )
                {
                    if (patchi < 0) { continue; }
                    const fvPatchScalarField& pfI =
                        thetaNetRefIntact_.boundaryField()[patchi];
                    scalarField faceFluxI
                    (
                        thetaNetRefIntactEqn.internalCoeffs()[patchi]
                       *pfI.patchInternalField()
                    );
                    if (pfI.coupled())
                    {
                        faceFluxI -=
                            thetaNetRefIntactEqn.boundaryCoeffs()[patchi]
                           *pfI.patchNeighbourField();
                    }
                    else
                    {
                        faceFluxI -=
                            thetaNetRefIntactEqn.boundaryCoeffs()[patchi];
                    }
                    refIOut += gSum(faceFluxI);
                }
                refIBoundaryFluxStep_ = refIOut;
                const tmp<volScalarField::Internal> tDivCI
                (
                    fvc::surfaceIntegrate(carrier)
                );
                const scalarField& divCI = tDivCI();
                const scalarField& refISolved =
                    thetaNetRefIntact_.primitiveField();
                const scalarField& VI = mesh_.V();
                refISpSourceStep_ =
                    runTime.deltaTValue()*gSum(divCI*refISolved*VI);
            }
        }
    }

    // S16 C+: THE DAMAGE PARTITION, LIVE FROM STEP 1 (design 2C, 17).
    //
    // Its upwind predictors are the contents' matrix on the contents'
    // carrier and weights: each is an M-matrix, so each is non-negative.
    // The correction below READS theta's limited flux and never writes into
    // theta's limiter: the contents' transport, the block after this one,
    // is package 2dd6246c's text, unchanged, so theta is bit-identical to
    // S15 at every step, before and after the first damage.
    tmp<volScalarField> tThetaLowP;
    tmp<surfaceScalarField> tATP;
    {
        fvScalarMatrix thetaIntactEqn
        (
            fvm::ddt(thetaIntact_)
          + phaseConservativeConvectionWith
            (
                carrier, thetaIntact_, tContentWeights()
            )
        );
        {
            scaleToFieldUnits(thetaIntactEqn);
        }
        thetaIntactEqn.solve();
        thetaIntact_.correctBoundaryConditions();

        fvScalarMatrix thetaPendingEqn
        (
            fvm::ddt(thetaPending_)
          + phaseConservativeConvectionWith
            (
                carrier, thetaPending_, tContentWeights()
            )
        );
        {
            scaleToFieldUnits(thetaPendingEqn);
        }
        thetaPendingEqn.solve();
        thetaPending_.correctBoundaryConditions();

        fvScalarMatrix thetaBrokenEqn
        (
            fvm::ddt(thetaBroken_)
          + phaseConservativeConvectionWith
            (
                carrier, thetaBroken_, tContentWeights()
            )
        );
        {
            scaleToFieldUnits(thetaBrokenEqn);
        }
        thetaBrokenEqn.solve();
        thetaBroken_.correctBoundaryConditions();

        // S17 A: the pending fibre's contents ride P2's predictor -- the
        // same conservative upwind matrix on the same carrier (the content
        // weights are upwind wherever the partition is allowed: limitedLinear
        // is refused), so C_B/thP and q_P/thP are convex combinations of the
        // donors' up to the linear solver's tolerance (08: <= 7.8e-6).
        {
            fvSymmTensorMatrix thetaPendingBEqn
            (
                fvm::ddt(thetaPendingB_)
              + fv::gaussConvectionScheme<symmTensor>
                (
                    mesh_,
                    carrier,
                    tmp<surfaceInterpolationScheme<symmTensor>>
                    (
                        new upwind<symmTensor>(mesh_, carrier)
                    )
                ).fvmDiv(carrier, thetaPendingB_)
            );
            {
                scaleToFieldUnits(thetaPendingBEqn);
            }
            thetaPendingBEqn.solve();
            thetaPendingB_.correctBoundaryConditions();

            fvScalarMatrix thetaPendingPEqn
            (
                fvm::ddt(thetaPendingP_)
              + phaseConservativeConvectionWith
                (
                    carrier, thetaPendingP_, tContentWeights()
                )
            );
            {
                scaleToFieldUnits(thetaPendingPEqn);
            }
            thetaPendingPEqn.solve();
            thetaPendingP_.correctBoundaryConditions();

            // S17b A1 (V1): the taut share's predictor -- P2's matrix, P2's
            // arithmetic (field units under A2) and, refused otherwise, P2's
            // fixed-sweep solver entry, so theta_T <= P2 holds exactly after
            // it (06: same M-matrix, b_T <= b_P, monotone Gauss-Seidel).
            {
                fvScalarMatrix thetaTautEqn
                (
                    fvm::ddt(thetaTaut_)
                  + phaseConservativeConvectionWith
                    (
                        carrier, thetaTaut_, tContentWeights()
                    )
                );
                {
                    scaleToFieldUnits(thetaTautEqn);
                }
                thetaTautEqn.solve();
                thetaTaut_.correctBoundaryConditions();
            }
        }

        {
            // theta's low-order total and total antidiffusive flux, formed
            // by the SAME expressions the block below forms thetaLow and AT
            // with (thetaC_ + thetaH_; AC + AH), before fctApply moves them.
            tThetaLowP = volScalarField::New
            (
                "thetaLowPartition", thetaC_ + thetaH_
            );
            tATP = surfaceScalarField::New
            (
                "fctFluxTotalPartition",
                fctAntidiffusiveFlux(carrier, thetaC_)
              + fctAntidiffusiveFlux(carrier, thetaH_)
            );
        }
    }

    // S20 (rev. 4 section 2.2): the FCT block's numbers, carried to the
    // PHASE_CLOT_CONTENT_FCT line printed after C+ below, whose partition
    // sum that line also reports. -1: not yet set (the block always sets
    // them).
    label fctActive = -1;
    label fctLimited = -1;
    scalar fctLamMin = -1;
    scalar fctLamFluxMean = -1;
    label fctNegAfter = -1;

    // S10 FIX 3: THE FLUX CORRECTION. Everything above is the upwind
    // predictor; this adds, face by face, as much of the (linear - upwind)
    // flux as keeps EVERY transported field within its local bounds. ONE
    // limiter for all of them -- the minimum -- so thetaC + thetaH still obeys
    // the equation its parts obey, and thetaNetRef moves with them exactly
    // where the carrier is divergence-free. The total is limited in its own
    // right as well, so theta itself creates no new extremum.
    {
        const volScalarField thetaLow("thetaLow", thetaC_ + thetaH_);
        const volScalarField thetaOld
        (
            "thetaOld", thetaC_.oldTime() + thetaH_.oldTime()
        );

        const surfaceScalarField AC(fctAntidiffusiveFlux(carrier, thetaC_));
        const surfaceScalarField AH(fctAntidiffusiveFlux(carrier, thetaH_));
        const surfaceScalarField AT("fctFluxTotal", AC + AH);

        surfaceScalarField lambda
        (
            "contentFctLambda", fctLimiter(thetaC_, thetaC_.oldTime(), AC)
        );
        lambda = min(lambda, fctLimiter(thetaH_, thetaH_.oldTime(), AH));
        lambda = min(lambda, fctLimiter(thetaLow, thetaOld, AT));

        tmp<surfaceScalarField> tAR;
        {
            tAR = fctAntidiffusiveFlux(carrier, thetaNetRef_);
            lambda = min
            (
                lambda,
                fctLimiter(thetaNetRef_, thetaNetRef_.oldTime(), tAR())
            );
        }

        // D74 (design section 2): ref_I's FCT PART, ref_I -= (dt/V) sum
        // +-(r_d lambda AR), with r_d = ref_I^low/ref^low of the cell the
        // face's lambda AR LEAVES (owner where lambda AR > 0, as fctApply
        // takes it from the owner), read here, after both predictors and
        // before fctApply moves ref. r_d = 0 where ref^low = 0: the
        // DENOMINATOR is guarded (its outflow is then zero anyway, red team
        // 36 F9). lambda and AR are READ, never written: theta's, ref's and
        // the partition's limited fluxes are the shipped ones.
        if (tAR.valid())
        {
            const labelUList& ownI = mesh_.owner();
            const labelUList& neiI = mesh_.neighbour();
            const scalarField& lamI = lambda.primitiveField();
            const scalarField& aRI = tAR().primitiveField();
            const scalarField& refLowI = thetaNetRef_.primitiveField();
            const scalarField& refILowI = thetaNetRefIntact_.primitiveField();
            const scalarField& VI = mesh_.V();
            const scalar dtI = runTime.deltaTValue();
            scalarField divI(mesh_.nCells(), 0.0);
            forAll(ownI, facei)
            {
                const scalar laI = lamI[facei]*aRI[facei];
                const label donorI = laI > 0 ? ownI[facei] : neiI[facei];
                const scalar rD =
                    refLowI[donorI] != 0
                  ? refILowI[donorI]/refLowI[donorI]
                  : 0.0;
                const scalar fluxI = rD*laI;
                divI[ownI[facei]] += fluxI;
                divI[neiI[facei]] -= fluxI;
            }
            scalarField& refIC = thetaNetRefIntact_.primitiveFieldRef();
            forAll(refIC, celli)
            {
                refIC[celli] -= dtI/VI[celli]*divI[celli];
            }
            thetaNetRefIntact_.correctBoundaryConditions();
        }

        fctApply(thetaC_, lambda, AC);
        fctApply(thetaH_, lambda, AH);
        if (tAR.valid())
        {
            fctApply(thetaNetRef_, lambda, tAR());
        }

        contentFctLambda_ = tmp<surfaceScalarField>
        (
            new surfaceScalarField("contentFctLambda", lambda)
        );

        // REPORTED, AND READ BY assess_case.py (s10 block). negAfter is the
        // quantity the whole construction exists to hold at zero; the
        // platelet kernel refuses the step if it is not. lambdaFluxMean is
        // how much of the second-order flux was actually admitted, weighted
        // by the antidiffusive flux it limits.
        const scalar tiny = std::numeric_limits<scalar>::min();
        const scalarField& lam = lambda.primitiveField();
        const scalarField& aT = AT.primitiveField();
        label active = 0;
        label limited = 0;
        scalar lamMin = 1.0;
        scalar num = 0;
        scalar den = 0;
        forAll(lam, facei)
        {
            if (mag(aT[facei]) >= tiny)
            {
                ++active;
                if (lam[facei] < 1.0) { ++limited; }
                lamMin = min(lamMin, lam[facei]);
                num += lam[facei]*mag(aT[facei]);
                den += mag(aT[facei]);
            }
        }
        label negAfter = 0;
        forAll(thetaC_, celli)
        {
            if (thetaC_[celli] < 0 || thetaH_[celli] < 0) { ++negAfter; }
            if
            (
                thetaNetRef_[celli] < 0
            )
            {
                ++negAfter;
            }
            // D74: and the intact reference (design section 2).
            if (thetaNetRefIntact_[celli] < 0)
            {
                ++negAfter;
            }
        }
        reduce(active, sumOp<label>());
        reduce(limited, sumOp<label>());
        reduce(lamMin, minOp<scalar>());
        reduce(num, sumOp<scalar>());
        reduce(den, sumOp<scalar>());
        reduce(negAfter, sumOp<label>());

        // S20: printed after C+, below.
        fctActive = active;
        fctLimited = limited;
        fctLamMin = lamMin;
        fctLamFluxMean = (den > 0 ? num/den : 1.0);
        fctNegAfter = negAfter;
    }


    // S16 C+: THE PARTITION FOLLOWS THETA'S LIMITED FLUX (design 2C; the
    // replica is 17's probes/numerics-analyst-p17-r3.py, C_plus L183-212).
    //   C  -- each component takes its donor cell's share of
    //         g = lambda_theta*A_theta (donor_apply, L86-120);
    //   C+ -- plus the zero-sum composition antidiffusion
    //         E_k = lambda_theta (A_Pk - s_k A_theta), limited by one mu in
    //         field units with the exact-zero convention (R1; lam_R1 L69-72,
    //         apply_R1 L75-77).
    // Non-negative by construction; a non-finite input is refused.
    {
        const labelUList& own = mesh_.owner();
        const labelUList& nei = mesh_.neighbour();
        const label nC = mesh_.nCells();
        const label nF = own.size();
        const scalarField& V = mesh_.V();
        const scalar Vmin = gMin(V);
        const scalarField Vrel(V/Vmin);
        const scalar dt = runTime.deltaTValue();
        const scalar sFU = dt/Vmin;
        const scalarField& lamTheta = contentFctLambda_().primitiveField();
        const scalarField& ATheta = tATP().primitiveField();
        const scalarField g(lamTheta*ATheta);
        const scalarField& thL = tThetaLowP().primitiveField();

        volScalarField* parts[3] =
            {&thetaIntact_, &thetaPending_, &thetaBroken_};
        scalarField Plow[3];
        scalarField Pold[3];
        scalarField APk[3];
        scalarField Pnew[3];
        for (label k = 0; k < 3; ++k)
        {
            Plow[k] = parts[k]->primitiveField();
            Pold[k] = parts[k]->oldTime().primitiveField();
            APk[k] =
                fctAntidiffusiveFlux(carrier, *parts[k])().primitiveField();
            Pnew[k].setSize(nC);
        }
        const scalar* PlowP[3] = {Plow[0].cdata(), Plow[1].cdata(), Plow[2].cdata()};
        const scalar* PoldP[3] = {Pold[0].cdata(), Pold[1].cdata(), Pold[2].cdata()};
        const scalar* APkP[3] = {APk[0].cdata(), APk[1].cdata(), APk[2].cdata()};
        scalar* PnewP[3] = {Pnew[0].data(), Pnew[1].data(), Pnew[2].data()};

        // S17 A: P2 after C, for the passengers' C+ share (spec = X/P2 of
        // the cell losing P2 across the face).
        scalarField P2afterC;
        std::vector<scalar> fFace;
        const damagePartitionKernel::Stats<scalar> stC =
            damagePartitionKernel::donorShares<scalar, label>
            (
                nC, nF, own.cdata(), nei.cdata(), V.cdata(), dt,
                thL.cdata(), g.cdata(), 3, PlowP, PnewP, fFace
            );
        if (stC.error)
        {
            FatalErrorInFunction
                << "S16 C+ donor shares refused: " << stC.error
                << " (index " << label(stC.where) << ")."
                << exit(FatalError);
        }
        P2afterC = Pnew[1];
        std::vector<scalar> muF;
        std::vector<std::vector<scalar>> EF;
        const damagePartitionKernel::Stats<scalar> stE =
            damagePartitionKernel::composition<scalar, label>
            (
                nC, nF, own.cdata(), nei.cdata(), sFU, Vrel.cdata(),
                lamTheta.cdata(), ATheta.cdata(), g.cdata(), 3,
                PlowP, PoldP, APkP, PnewP,
                damagePartitionKernel::MuMode::limited, &muF, &EF
            );
        if (stE.error)
        {
            FatalErrorInFunction
                << "S16 C+ composition correction refused: " << stE.error
                << " (index " << label(stE.where) << ")."
                << exit(FatalError);
        }
        for (label k = 0; k < 3; ++k)
        {
            parts[k]->primitiveFieldRef() = Pnew[k];
            parts[k]->correctBoundaryConditions();
        }
        // S17 A: THE FIBRE RIDES P2's C AND C+ FLUXES (12 section 1; 08/10:
        // q_P's flux is p_P of the donor times thP's limited flux). Seven
        // passengers: C_B's six components and q_P.
        {
            symmTensorField& cB = thetaPendingB_.primitiveFieldRef();
            scalarField& qP = thetaPendingP_.primitiveFieldRef();
            scalarField Xlow[7], Xnew[7];
            for (label x = 0; x < 6; ++x)
            {
                Xlow[x] = cB.component(x);
                Xnew[x].setSize(nC);
            }
            Xlow[6] = qP;
            Xnew[6].setSize(nC);
            const scalar* XlowP[7];
            scalar* XnewP[7];
            for (label x = 0; x < 7; ++x)
            {
                XlowP[x] = Xlow[x].cdata();
                XnewP[x] = Xnew[x].data();
            }
            // S18 E (H9): under cplusOrder shared the passengers take
            // tautShare()'s A/B operation order (07: 0 raises in 352 491
            // kernel projections against 2401 shipped).
            const char* err =
                damagePartitionKernel::passengersShared<scalar, label>
                (
                    nC, nF, own.cdata(), nei.cdata(), V.cdata(),
                    g.cdata(), fFace, sFU, Vrel.cdata(),
                    P2afterC.cdata(), Pnew[1].cdata(), muF, EF[1], 7,
                    XlowP, XnewP, projSiteLog_ ? &s18pKernelDbg_ : nullptr
                );
            if (err)
            {
                FatalErrorInFunction
                    << "S17 pending-fibre transport refused: " << err << "."
                    << exit(FatalError);
            }
            for (label x = 0; x < 6; ++x)
            {
                cB.replace(x, Xnew[x]);
            }
            qP = Xnew[6];
            thetaPendingB_.correctBoundaryConditions();
            thetaPendingP_.correctBoundaryConditions();

            // S17b A1 (V1): the taut share rides P2's C and C+ fluxes with
            // the donor's taut fraction, theta_T <= P2 EXACT by the kernel's
            // monotone construction (damagePartitionKernel::tautShare).
            {
                scalarField& tT = thetaTaut_.primitiveFieldRef();
                const scalarField tLow(tT);
                scalarField tNew(nC);
                const char* errT =
                    damagePartitionKernel::tautShare<scalar, label>
                    (
                        nC, nF, own.cdata(), nei.cdata(), V.cdata(),
                        g.cdata(), fFace, sFU, Vrel.cdata(),
                        Plow[1].cdata(), P2afterC.cdata(), Pnew[1].cdata(),
                        muF, EF[1], tLow.cdata(), tNew.data()
                    );
                if (errT)
                {
                    FatalErrorInFunction
                        << "S17b taut-share transport refused: " << errT
                        << "." << exit(FatalError);
                }
                tT = tNew;
                thetaTaut_.correctBoundaryConditions();
            }
        }
        s16FoutMax_ = max(s16FoutMax_, returnReduce(stC.FoutMax, maxOp<scalar>()));
        s16MuMin_ = min(s16MuMin_, returnReduce(stE.muMin, minOp<scalar>()));
        s16MuMean_ = stE.muMean;
    }
    // S20 (rev. 4 sections 2.1-2.2): THE FCT LINE, once the partition has
    // taken theta's limited flux. Read by assess_case.py (s10 block: G3',
    // G3'lambda) and run_main_case_colab.py's 5 ms watcher (V6), which
    // treats a missing key as an error. zeroFluxRule names fctLimiter's rule
    // (an exactly-zero flux places no constraint). partitionSumDevMax is
    // PHASE_CLOT_S16's quantity, max|P1 + (P2 + P3) - theta| over max theta,
    // here after the transport and on every step: C+ above is unconditional
    // in this function, so the number is always defined where this prints.
    {
        const scalarField& p1 = thetaIntact_.primitiveField();
        const scalarField& p2 = thetaPending_.primitiveField();
        const scalarField& p3 = thetaBroken_.primitiveField();
        const scalarField& c1 = thetaC_.primitiveField();
        const scalarField& h1 = thetaH_.primitiveField();
        scalar devT = 0;
        scalar thetaTop = 0;
        forAll(p1, celli)
        {
            const scalar thc = c1[celli] + h1[celli];
            thetaTop = max(thetaTop, thc);
            devT = max(devT, mag(p1[celli] + (p2[celli] + p3[celli]) - thc));
        }
        reduce(devT, maxOp<scalar>());
        reduce(thetaTop, maxOp<scalar>());

        Info<< "PHASE_CLOT_CONTENT_FCT time=" << runTime.userTimeValue()
            << " activeFaces=" << fctActive
            << " limitedFaces=" << fctLimited
            << " lambdaMin=" << fctLamMin
            << " lambdaFluxMean=" << fctLamFluxMean
            << " negAfter=" << fctNegAfter
            << " zeroFluxRule=exactZero"
            << " partitionSumDevMax="
            << (thetaTop > 0 ? devT/thetaTop : 0)
            << endl;
    }
    // S16 C: ruptured_ = D/theta for every consumer of d; the partition is
    // checked (negative or non-finite is fatal), never bounded.
    {
        syncDamageFromPartition("content transport");
    }
    // S17 A: P-LS after the transport (12: transport, then P-LS), so every
    // later step reads one fibre per cell.
    projectPendingFibres("content transport");

    // NOTHING IS CLIPPED HERE, DELIBERATELY. Clipping a conserved content
    // destroys the balance this stage exists to establish, and would hide the
    // very undershoot being investigated. Negative values are REPORTED; the
    // step is not silently repaired.
    const volScalarField theta(thetaC_ + thetaH_);
    const scalarField& thC = thetaC_.primitiveField();
    const scalarField& thH = thetaH_.primitiveField();
    const scalarField& th = theta.primitiveField();

    label negC = 0, negH = 0;
    forAll(thC, celli)
    {
        if (thC[celli] < 0) { ++negC; }
        if (thH[celli] < 0) { ++negH; }
    }
    reduce(negC, sumOp<label>());
    reduce(negH, sumOp<label>());

    const scalar contentAfter = fvc::domainIntegrate(theta).value();

    // ---- THE STAGE-1 EQUIVALENCE TEST WAS RETIRED IN S6c-2 --------------
    //
    // It compared theta/theta_ref against chiLegacy = max(0.5*(1 - c), 0)
    // and reported equivMax, equivL1 and chiVolume. With c deleted the only
    // way to keep it is to rebuild chiLegacy from theta, at which point it
    // compares a field against itself and reads identically zero -- a
    // triumphant pass measuring nothing, which is the failure mode retired
    // as G3 of the phase-volume spec. Deleted rather than migrated.
    //
    // WHAT IT WAS ACTUALLY MEASURING, and nothing read it back for eleven
    // days: equivMax = 0.5*|c - (1 - 2*ind)|, the drift between the two
    // representations. It ran 3.33066907388e-16 at t = 0 to
    // 0.0158337123018 at t = 4e-04 s in run 20260920-060033 -- monotone,
    // ~t^1.9, no plateau in the window. PHASE_CLOT_QBASIS's dqMax was the
    // same number by a different route. Two instruments printed this drift
    // on every step since Stage 1 and it was found by diffing written
    // fields, not by reading either of them.
    //
    // The lesson is recorded, not just the deletion: c and theta were the
    // same equation under an exact algebraic identity and had separated by
    // 3% of the interface. Which of the two transports was the accurate one
    // is NOT settled by deleting one of them; it is queued for Codex in
    // research/designs/2026-09-20-S6c-2-delete-c.md section 7, question 3.

    Info<< "PHASE_CLOT_CONTENT time=" << runTime.userTimeValue()
        << " thetaMin=" << gMin(th)
        << " thetaMax=" << gMax(th)
        << " thetaCNeg=" << negC
        << " thetaHNeg=" << negH
        << " contentVolume=" << contentAfter
        << " contentChange=" << (contentAfter - contentBefore)
        << " operator=content"
        << " contentSourceDt=" << contentSourceDt
        << " kinetics=off"
        << endl;
}


void Foam::solvers::phaseClotAxisymmetricFluid::checkPlateletConfiguration()
{
    // FAIL CLOSED. Every combination that could produce a plausible-looking
    // wrong answer is refused here rather than discovered in the fields.
    // THE DOUBLE-GROWTH COMBINATION. `matched` reproduces the legacy phase
    // equation INCLUDING its growth source; the accepted transfer below also
    // adds solid. Running both means two independent things deposit content
    // into thetaH and the budget still closes, because each is internally
    // paired. Refuse it.
    // The double-growth refusal that stood here is subsumed: `matched` was
    // the only operator that carried the legacy growth source, and S6c-2
    // refuses it at construction.

    if (plateletDiffusivity_ < 0)
    {
        FatalErrorInFunction
            << "plateletDiffusivity must be >= 0 (0 is the supported"
            << " transport-isolation case); found " << plateletDiffusivity_
            << exit(FatalError);
    }

    if (plateletYield_ <= 0)
    {
        FatalErrorInFunction
            << "plateletYield Y must be > 0; found " << plateletYield_
            << exit(FatalError);
    }

    if (plateletBindRate_ < 0)
    {
        FatalErrorInFunction
            << "plateletBindRate lambda must be >= 0; found "
            << plateletBindRate_
            << exit(FatalError);
    }

    if (plateletAdhesionSpeed_ == 0)
    {
        FatalErrorInFunction
            << "plateletBindLocalisation is 'interface' but"
            << " plateletAdhesionSpeed is exactly 0, which deposits nothing"
            << " anywhere. Use plateletTransportOnly to disable binding, or a"
            << " negative value to reuse adhesionSpeedEff."
            << exit(FatalError);
    }

    // Y = vP IS A DECLARED SPECIALISATION, NOT AN IDENTITY. Log which one
    // this case is running so no reader has to infer it from two numbers
    // happening to match.
    const bool specialised = (mag(plateletYield_ - vP_) <= 0);

    Info<< "PHASE_CLOT_FREEPLATELET_CONFIG"
        << " Deff=" << plateletDiffusivity_
        << " Y=" << plateletYield_
        << " vP=" << vP_
        << " yieldModel="
        << (specialised ? "platelet-equivalent(Y=vP)" : "independent-Y")
        << " lambda=" << plateletBindRate_
        << " adhesionSpeedSource="
        << (plateletAdhesionSpeed_ < 0
                ? "reuse-adhesionSpeedEff"
                : "dictionary")
        << " adhesionSpeed=" << plateletAdhesionSpeed_
        << " transportOnly=" << plateletTransportOnly_
        << " thetaMax=" << phiSMax_
        << " depositionContentCap=" << depositionContentCap_
        << " networkPole=" << networkPole()
        << endl;


    if (specialised)
    {
        Info<< "PHASE_CLOT_FREEPLATELET_NOTE Y = vP is the declared"
            << " platelet-equivalent verification specialisation: one"
            << " consumed free platelet-equivalent deposits vP of bound"
            << " platelet-equivalent solid at constant material volume."
            << " It is NOT a platelet-count to wet-clot-volume conversion,"
            << " and Y*N + theta is total platelet-equivalent material,"
            << " not physical platelet number, whenever theta also holds"
            << " independently deposited material." << endl;
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::solvePlateletContent
(
    const surfaceScalarField& suppliedCarrier,
    const label phaseCorrector
)
{
    const scalar dt = runTime.deltaTValue();
    const dimensionedScalar deltaT("dt", dimTime, dt);

    // ---- 1. THE POROSITY, FROM THE CONTENT FIELDS ----
    //
    // theta here is theta_star: solveContentComponents() has already run for
    // this corrector, so this is the transported trial, which is the state
    // the platelet transport must see.
    const volScalarField thetaStar(thetaC_ + thetaH_);
    epsP_ = scalar(1) - thetaStar;
    epsP_.correctBoundaryConditions();

    // MATERIAL VALIDITY BEFORE ANY SENSITIVE EVALUATION. eps <= 0 is not a
    // small number to be floored, it is a cell with no pore space, and
    // dividing by a floored version of it would manufacture platelets.
    const scalar epsMin = gMin(epsP_.primitiveField());
    if (epsMin <= 0)
    {
        FatalErrorInFunction
            << "eps = 1 - theta reached " << epsMin << " (<= 0) at time "
            << runTime.userTimeValue() << ": the mixture has no pore volume"
            << " and the platelet transport is undefined there. This is a"
            << " material violation, not a numerical one -- it is refused"
            << " rather than clamped, because a floored eps would"
            << " manufacture pore concentration out of nothing."
            << exit(FatalError);
    }

    // ---- 2. THE FLUX, IN THE DECLARED CONVENTION ----
    //
    // F_eps is the POROSITY-WEIGHTED fluid-volume flux and advects nP.
    // Selecting a convention does not turn a MIXTURE flux into a fluid flux.
    // correctPressureTwoVelocity writes phi_ as the mixture and maintains
    // phiFluidOwn_ separately. Use that phase's intrinsic flux under slip.
    //
    // Preserve the supplied carrier on all one-velocity gate paths. For the
    // two-velocity superficial convention, form the superficial INPUT here;
    // the shared helper then uses it unchanged. For intrinsic, the helper
    // supplies the single epsilon factor. Both yield eps_f*phiFluidOwn_.
    // No claim follows that this content porosity already agrees with the
    // legacy momentum fractions: the volume residual below still measures it.
    surfaceScalarField carrier("plateletCarrier", suppliedCarrier);
    {
        carrier = phiFluidOwn_;
    }

    const tmp<surfaceScalarField> tFeps
    (
        plateletContent::flux(word("intrinsic"), epsP_, carrier)
    );
    const surfaceScalarField& Feps = tFeps();

    // Direct carrier evidence for outer-transaction gates. A momentum-solver
    // residual is not a measurement of the face flux consumed above, and the
    // platelet solve occurs before that outer's momentum correction. Report
    // both the supplied carrier and the porosity-weighted operator flux so a
    // gate can distinguish changing transport from changing state history.
    scalar carrierBoundarySum = 0;
    scalar FepsBoundarySum = 0;
    forAll(carrier.boundaryField(), patchi)
    {
        carrierBoundarySum += gSum(carrier.boundaryField()[patchi]);
        FepsBoundarySum += gSum(Feps.boundaryField()[patchi]);
    }
    const scalar carrierMin = gMin(carrier.primitiveField());
    const scalar carrierMax = gMax(carrier.primitiveField());
    const scalar carrierAbsSum = gSum(mag(carrier.primitiveField()));
    const scalar FepsMin = gMin(Feps.primitiveField());
    const scalar FepsMax = gMax(Feps.primitiveField());
    const scalar FepsAbsSum = gSum(mag(Feps.primitiveField()));

    // ---- 3. THE FLUID-VOLUME CONSISTENCY RESIDUAL ----
    //
    // A uniform pore concentration is preserved by transport only when
    // ddt(eps) + div(F_eps) = 0 holds DISCRETELY, at the same time level.
    // In this solver the continuity flux is weighted by phiFluid_, which is
    // built from q_ and NOT from these contents (see the header note on
    // epsP_), so this residual is MEASURED and reported rather than assumed
    // to vanish. A T5 claim on a case where this is not small is void.
    //
    // fvc::surfaceIntegrate, NOT fvc::div, for the same reason the rest of
    // this solver uses it: fvc::div(surfaceScalarField) is
    // surfaceIntegrateExtrapolate, which differs at boundary faces. It also
    // keeps this out of the fvSchemes scheme-name scan, which cannot tell a
    // single-argument div (no lookup) from a two-argument one (lookup) and
    // would otherwise demand a div(Feps) entry no case needs.
    scalar volResMax = 0, volResL1 = 0;
    plateletContent::volumeConsistency(epsP_, Feps, dt, volResMax, volResL1);

    // ---- 4. TRANSPORT AND DIFFUSION: THE TRIAL, WITH NO REACTION ----
    //
    // Reaction is deliberately NOT in this matrix. The plan's contract is one
    // accepted transfer computed from the transport trial and applied once to
    // both consumers; folding an implicit Sp in here would make the accepted
    // increment a by-product of the linear solve and there would be no single
    // array to apply to thetaH.
    //
    // Upwind weights built explicitly, for the same reason the contents use
    // them: the scheme is part of the contract, and a limited scheme
    // recomputes its limiter from the transported field, so nP and the
    // contents would ride different matrices from one flux.
    const scalar NBefore =
        fvc::domainIntegrate(epsP_.oldTime()*nP_.oldTime()).value();

    tmp<fvScalarMatrix> tnEqn
    (
        plateletContent::transportEqn
        (
            epsP_,
            nP_,
            Feps,
            dimensionedScalar("Deff", dimArea/dimTime, plateletDiffusivity_)
        )
    );
    fvScalarMatrix& nEqn = tnEqn.ref();
    nEqn.solve();
    nP_.correctBoundaryConditions();

    // The boundary outflow the assembled operator actually applied, so the
    // budget below closes against a measured number rather than an assumed
    // zero. Internal faces cancel in the sum by construction.
    const surfaceScalarField nFaceFlux(nEqn.flux());
    scalar boundaryOut = 0;
    forAll(nFaceFlux.boundaryField(), patchi)
    {
        boundaryOut += gSum(nFaceFlux.boundaryField()[patchi]);
    }

    const volScalarField NStar(epsP_*nP_);
    const scalar NStarVol = fvc::domainIntegrate(NStar).value();

    // Transport budget: (N* - N_old) + dt*(net outflow) == 0. THIS is the
    // check that catches a wrong old-time porosity in ddt(eps, nP); the
    // header's reasoning about registered fields is an explanation, not
    // evidence.
    const scalar transportResidual =
        (NStarVol - NBefore) + dt*boundaryOut;

    // ---- 5. THE ONE ACCEPTED TRANSFER ----
    //
    // dB is computed ONCE, constrained ONCE, and applied to BOTH consumers.
    // Note it is built from the trial NStar, not from the old-time state:
    // with transport on, Y*N + theta is not conserved against the same
    // cell's old-time value, because platelets and solid cross different
    // faces at different velocities.
    volScalarField dB
    (
        IOobject("dB", runTime.name(), mesh),
        mesh,
        dimensionedScalar("zero", dimless/dimVolume, 0),
        zeroGradientFvPatchField<scalar>::typeName
    );

    label nAvail = 0, nCap = 0;
    scalar dBSum = 0;

    // THE RATE FIELD. Built here, once, so the two modes differ in exactly
    // one object and the log can report what was actually used.
    //
    // "interface" is the specified law: AM2020-phase-clot-v1 eq. (P) puts
    // uptake on delta_Gamma, and interfaceDelta_ IS delta_Gamma -- either
    // 0.5*|grad c|, which integrates to 1 across a field running -1 to +1, or
    // |grad theta|/theta_ref, which integrates to theta_bulk/theta_ref and is
    // therefore calibrated to whatever indMax reports. phaseDeltaBasis_ says
    // which, and THIS SITE IS THE ONE STAGE S3 EXISTS FOR. phaseGradient_ is
    // twice the c form and must NOT be used here -- the platelet sink is a flux per
    // unit area, exactly as the legacy sink at :8309 already has it.
    volScalarField bindRate
    (
        IOobject("bindRate", runTime.name(), mesh),
        mesh,
        dimensionedScalar("zero", dimless/dimTime, 0),
        zeroGradientFvPatchField<scalar>::typeName
    );

    {
        const dimensionedScalar kFixed
        (
            "plateletAdhesionSpeed", dimLength/dimTime, plateletAdhesionSpeed_
        );
        bindRate =
            interfaceDelta_
           *(
                plateletAdhesionSpeed_ < 0
              ? adhesionSpeedEff_
              : volScalarField
                (
                    IOobject("kFixed", runTime.name(), mesh),
                    mesh,
                    kFixed,
                    zeroGradientFvPatchField<scalar>::typeName
                )
            );
    }
    bindRate.correctBoundaryConditions();

    // OI-11, FIXED HERE. int deltaTheta_ dV taken at the point the reaction
    // reads it -- bindRate above is built from interfaceDelta_, which under
    // phaseDeltaBasis 'content' IS deltaTheta_. The instrument's old
    // deltaThetaIntPost recomputed the same integral in
    // writeBalanceDiagnostic(), which runs in postSolve() after the last
    // updateDerivedFields(), so it was bit-identical to deltaThetaInt in all
    // 272 records of run 20260917-003709 and measured nothing. This one can
    // differ from both, and the difference is the time-level error the
    // instrument was supposed to bound.
    deltaThetaIntAtTransfer_ = fvc::domainIntegrate(deltaTheta_).value();

    // The reaction runs if the rate is nonzero ANYWHERE. Gating on the
    // scalar plateletBindRate_ would have switched the whole interface mode
    // off whenever lambda was left at its default, which is precisely the
    // configuration an interface child wants.
    const scalar bindRateMax = gMax(bindRate.primitiveField());

    // WHERE the rate actually lives, reported every step so the localisation
    // is a measured number in the log and not a claim about a dictionary key.
    // chi = max(1-q,0) is the solid indicator, so this is the share of the
    // rate-weighted volume that sits on clot rather than in the lumen. Under
    // "uniform" it collapses to the clot volume fraction (~0.03 here); under
    // "interface" it should be large.
    scalar bindRateClotFrac = 0;
    {
        const volScalarField chiLoc(max(1 - q_, dimensionedScalar(dimless, 0)));
        const scalar weighted =
            fvc::domainIntegrate(bindRate*chiLoc).value();
        const scalar total = fvc::domainIntegrate(bindRate).value();
        bindRateClotFrac = (total > 0) ? weighted/total : 0;
    }

    if (!plateletTransportOnly_ && bindRateMax > 0)
    {
        plateletContent::acceptedTransfer
        (
            NStar, thetaStar, dt, bindRate, plateletYield_,
            depositionContentCap_, dB, nAvail, nCap
        );
        dBSum = fvc::domainIntegrate(dB).value();
    }

    // S25 G (design section 2G, red team 04 change 12): the acceptance law
    // removed a tripwire (theta > cap was fatal), so its activity is LOGGED
    // every step on the line below: the count of cells whose transport-trial
    // content exceeds depositionContentCap_ (they accept no deposit), and
    // max(theta - cap) over all cells (negative when none does).
    label contentOverCapCells = 0;
    scalar contentOverCapMax = -great;
    {
        const scalarField& thSOC = thetaStar.primitiveField();
        forAll(thSOC, celli)
        {
            const scalar excess = thSOC[celli] - depositionContentCap_;
            if (excess > 0) { ++contentOverCapCells; }
            contentOverCapMax = max(contentOverCapMax, excess);
        }
        reduce(contentOverCapCells, sumOp<label>());
        reduce(contentOverCapMax, maxOp<scalar>());
    }

    // S17 A (09 rule 2): the stress-free mixing below weights the old strain
    // by the PRE-deposit intact share thI/S = 1 - d*, taken here, before the
    // transfer is applied and joins thetaIntact_. thetaPending_ is not written by the
    // deposit (new network is intact), so the pending fibre -- its content,
    // axis, stretch and tension -- is unchanged by it: nothing to mirror.
    tmp<volScalarField> tIntactShareBefore;
    {
        tIntactShareBefore = intactShare();
    }

    // Shared paired application: one array, immutable transport-trial inputs,
    // and no recomputation of uptake between the two consumers.
    const volScalarField thetaHStar("thetaHStar", thetaH_);
    volScalarField NNew("NNew", NStar);
    plateletContent::applyAcceptedTransfer
    (
        NStar, thetaC_, thetaHStar, dB, plateletYield_, depositionContentCap_,
        NNew, thetaH_
    );
    // S16 C: thetaH_ WAS JUST WRITTEN, through the output parameter above
    // (red-team Change 4; lesson a-writer-can-be-an-output-parameter). The
    // deposit is new network, intact and untargeted, so it joins thetaIntact_
    // and nothing else: d and d* are diluted by exactly the share it adds.
    // d itself (ruptured_) is re-derived after the stress-free mixing below,
    // which must read the PRE-deposit intact share.
    {
        const scalarField pIBefore(thetaIntact_.primitiveField());
        const scalarField& hNew = thetaH_.primitiveField();
        const scalarField& hStar = thetaHStar.primitiveField();
        scalarField& pI = thetaIntact_.primitiveFieldRef();
        forAll(pI, celli)
        {
            pI[celli] += hNew[celli] - hStar[celli];
        }
        thetaIntact_.correctBoundaryConditions();

        // S16m (16 R5): THE MIRROR, MEASURED AT ITS SITE, independently of
        // the statement above: max over cells of |dP1 - dtheta| against
        // 1e-12*max theta. No solve intervenes, so it is round-off (~1e-16)
        // when the mirror is there and the whole deposit (>= 5.29e-9 in
        // em62, 13b) when it is not.
        const scalarField& tC = thetaC_.primitiveField();
        const scalarField& pIA = thetaIntact_.primitiveField();
        scalar dev = 0, thMax = 0;
        forAll(pIA, celli)
        {
            const scalar dP = pIA[celli] - pIBefore[celli];
            const scalar dTheta =
                (tC[celli] + hNew[celli]) - (tC[celli] + hStar[celli]);
            dev = max(dev, mag(dP - dTheta));
            thMax = max(thMax, tC[celli] + hNew[celli]);
        }
        reduce(dev, maxOp<scalar>());
        reduce(thMax, maxOp<scalar>());
        s16MirrorDevMax_ =
            max(s16MirrorDevMax_, thMax > 0 ? dev/thMax : dev);
    }

    // S25 B (R1, physics 02 Q1): the deposit is an EVENT for e. It lowers q
    // by Y*dB/theta_ref and leaves Wp unchanged (the ledger removes only the
    // bound platelets, N - dB), so e rises by the same amount. Booked from
    // the ACCEPTED array; this corrector's value replaces the last one's.
    plateletDepositStep_ = plateletYield_*dB.primitiveField();

    // The pairing residual, per cell. Zero by construction here, which is
    // exactly why it is reported: if it is ever not zero, the two consumers
    // have stopped sharing one array.
    const volScalarField thetaNew(thetaC_ + thetaH_);
    const dimensionedScalar depositedYield("plateletYield", dimVolume, plateletYield_);
    const volScalarField pairErr
    (
        depositedYield*(NNew - NStar) + (thetaNew - thetaStar)
    );
    const scalar pairMax = gMax(mag(pairErr.primitiveField()));

    // S10 FIX 1: NEW NETWORK IS LAID DOWN AT ITS OWN REFERENCE DENSITY.
    //
    // The APPLIED increment, thetaNew - thetaStar, for the reason 5b below
    // gives for c: applyIncrement clamps against phiSMax_, and an increment
    // taken from Y*dB would disagree with theta exactly when the cap bites.
    // Added to the reference as well as to theta, so deposition leaves
    // theta - thetaNetRef_ unchanged: growth is not compaction.
    {
        // S11 FIX 6: NEW NETWORK IS ALSO LAID DOWN STRESS-FREE.
        //
        // Under beRelaxBasis networkAbsent be no longer relaxes wherever
        // network exists, so content deposited into a strained cell would
        // inherit that cell's strain. It polymerised in the current
        // configuration: its own b is I. The cell's elastic state is the
        // occupancy-weighted mixture, and because dev(b) is linear in b the
        // mixed stress is exactly the sum of the two parts' stresses.
        // Removal (a negative increment) takes material of the cell's own
        // state and leaves be unchanged. Renormalised to det = 1 as
        // solveElasticStrain() does, so be stays isochoric between them.
        {
            const scalarField& r0 = thetaNetRef_.primitiveField();
            const scalarField& tN = thetaNew.primitiveField();
            const scalarField& tS = thetaStar.primitiveField();
            symmTensorField& b = be_.primitiveFieldRef();
            label mixed = 0;
            forAll(b, celli)
            {
                const scalar d = tN[celli] - tS[celli];
                const scalar r = r0[celli];
                if (d > 0 && r + d > 0)
                {
                    // S14: be is the strain of the INTACT network and the
                    // new network is intact and stress-free, so the old
                    // strain is weighted by the intact share r*(1 - ruptured)
                    // -- debris has no strain to average in -- and the
                    // ruptured fraction is diluted by exactly the share the
                    // deposit adds. With rupture off rIntact is r.
                    scalar rIntact = r;
                    {
                        rIntact = r*(1.0 - ruptured_[celli]);
                        // S16 C: under contentPartition the deposit has
                        // already diluted d through thetaIntact_; diluting
                        // the label again here would count it twice.
                        // S17 A, 09 rule 2: under fibreRelease the old
                        // strain is weighted by r (1 - d*) = r thI/S of the
                        // PRE-deposit partition, not r (1 - d). A deposit
                        // into a cell with no intact network leaves be = I
                        // exactly (G-dep).
                        {
                            rIntact = r*tIntactShareBefore()[celli];
                        }
                    }
                    b[celli] =
                        (rIntact*b[celli] + d*symmTensor::I)/(rIntact + d);
                    {
                        const scalar dB = det(b[celli]);
                        if (!(dB > 0))
                        {
                            FatalErrorInFunction
                                << "stress-free deposition produced det(be) = "
                                << dB << " at cell " << celli << ": a mixture "
                                << "of two positive-definite tensors cannot."
                                << exit(FatalError);
                        }
                        b[celli] *= pow(dB, -1.0/3.0);
                    }
                    ++mixed;
                }
            }
            be_.correctBoundaryConditions();
            ruptured_.correctBoundaryConditions();
            reduce(mixed, sumOp<label>());
            Info<< "PHASE_CLOT_DEPOSIT_STRESSFREE time="
                << runTime.userTimeValue() << " cellsMixed=" << mixed << endl;
        }

        thetaNetRef_ += thetaNew - thetaStar;
        thetaNetRef_.correctBoundaryConditions();
        // D74: ref_I takes ref's OWN increment, the same expression (not
        // P1's hNew - hStar, which differs by round-off), so it stays equal
        // to ref where d* = 0; booked for PHASE_CLOT_REFI.
        {
            thetaNetRefIntact_ += thetaNew - thetaStar;
            thetaNetRefIntact_.correctBoundaryConditions();
            const scalarField& tNI = thetaNew.primitiveField();
            const scalarField& tSI = thetaStar.primitiveField();
            const scalarField& VI = mesh_.V();
            forAll(tNI, celli)
            {
                refITransferStep_ += (tNI[celli] - tSI[celli])*VI[celli];
            }
        }
    }
    // S16 C: d from the partition, now that the deposit is in it and the
    // stress-free mixing has read the pre-deposit intact share.
    syncDamageFromPartition("deposit");

    // ---- 5b. THE SAME INCREMENT, IN THE OTHER CURRENCY (S4) -------------
    //
    // phaseGrowthDriver 'transfer': the accepted transfer is the only growth,
    // so c receives exactly what theta received, converted once.
    //
    //     chi   = 0.5*(1 - c)      =>  dc = -2*dchi
    //     theta = theta_ref*chi    =>  dchi = dtheta/theta_ref
    //     ==> dc = -(2/theta_ref)*dtheta
    //
    // dtheta is (thetaNew - thetaStar), the APPLIED increment, not Y*dB:
    // plateletTransferKernel::applyIncrement clamps against phiSMax_, and a
    // capped step taken from Y*dB would leave the two currencies disagreeing
    // exactly when the cap bites.
    //
    // APPLIED HERE, AFTER THE TRANSPORT, BECAUSE THAT IS WHAT theta DOES.
    // solveContentComponents() transports theta and this function adds the
    // transfer to the transported field; c is transported by the solve at the
    // top of this corrector and receives its increment here. One array, one
    // time level, one commit point.
    //
    // NOT ACCUMULATED ACROSS CORRECTORS. fvm::ddt(c_) rebuilds c from oldTime
    // at the top of every corrector, discarding the previous corrector's
    // increment before this one is applied -- which is the same contract
    // fvm::ddt(thetaC_) gives theta, and why dBSum is bit-identical between
    // correctors on all 272 steps of run 20260917-003709. transferClotGrowth_
    // is therefore OVERWRITTEN, never summed.
    //
    // The phase-field clip runs after this call and so sees the incremented
    // field, which is what it must.
    {
        const volScalarField dTheta("dTheta", thetaNew - thetaStar);

        // S6c-2: THE WRITE TO c IS GONE, THE NORMALISATION IS NOT.
        //
        // This used to be two statements: c -= (2/theta_ref)*dTheta, which
        // kept the legacy field in step with the transfer, and the integral
        // below. c is deleted; the integral stays EXACTLY as it was.
        //
        // The (2/theta_ref) factor is not a reference to c -- it is the
        // normalisation clotGrowth is defined in, and PHASE_CLOT_GROWTH_STOICH
        // divides it by plateletUptakeCommitted to get the 2*alpha/D exchange
        // rate (1.6277 measured against a 1.6304 target). Rescaling it here
        // to "theta currency" would move that ratio by a constant and
        // silently invalidate every stoichiometry record on file for a
        // cosmetic reason. Left alone deliberately.
        transferClotGrowth_ =
            (scalar(2)/skeletonVolumeFraction_)
           *fvc::domainIntegrate(dTheta).value()
           /dt;
    }

    // Carried out for the diagnostics that must report the transfer that
    // happened rather than a proxy for it. OI-12: plateletUptake has no N in
    // it. OI-11: deltaTheta_'s integral is snapshotted where the reaction
    // read it, above, not where the report is written.
    dBSumCommitted_ = dBSum;

    // eps follows the new theta, and nP follows from the CONSERVED N. N is
    // the inventory; the pore concentration is what that inventory becomes
    // in the pore space that is left.
    epsP_ = scalar(1) - thetaNew;
    epsP_.correctBoundaryConditions();
    nP_ = NNew/epsP_;
    nP_.correctBoundaryConditions();
    NP_ = epsP_*nP_;
    NP_.correctBoundaryConditions();

    // ---- 6. REPORT. Nothing is clipped; negatives are measured. ----
    const scalarField& Ni = NP_.primitiveField();
    label NNeg = 0;
    forAll(Ni, celli)
    {
        if (Ni[celli] < 0) { ++NNeg; }
    }
    reduce(NNeg, sumOp<label>());

    const scalar NAfter = fvc::domainIntegrate(NP_).value();

    Info<< "PHASE_CLOT_FREEPLATELET time=" << runTime.userTimeValue()
        << " step=" << runTime.timeIndex()
        << " attempt=" << plateletAttempt_
        << " outer=" << 0
        << " phase=" << phaseCorrector
        << " nMin=" << gMin(nP_.primitiveField())
        << " nMax=" << gMax(nP_.primitiveField())
        << " NMin=" << gMin(Ni)
        << " NMax=" << gMax(Ni)
        << " NNeg=" << NNeg
        << " NVolume=" << NAfter
        << " NChange=" << (NAfter - NBefore)
        << " NStarVolume=" << NStarVol
        << " transportResidual=" << transportResidual
        << " boundaryOut=" << boundaryOut
        << " dBSum=" << dBSum
        << " pairMax=" << pairMax
        << " bindLocalisation=interface"
        << " bindRateMax=" << bindRateMax
        << " bindRateClotFrac=" << bindRateClotFrac
        << " volResMax=" << volResMax
        << " volResL1=" << volResL1
        << " epsMin=" << epsMin
        << " availClamped=" << nAvail
        << " capClamped=" << nCap
        << " contentOverCapCells=" << contentOverCapCells
        << " contentOverCapMax=" << contentOverCapMax
        << " Y=" << plateletYield_
        << " Deff=" << plateletDiffusivity_
        << " convention=intrinsic"
        << " kinetics=" << (plateletTransportOnly_ ? "off" : "on")
        << " carrierMin=" << carrierMin
        << " carrierMax=" << carrierMax
        << " carrierAbsSum=" << carrierAbsSum
        << " carrierBoundarySum=" << carrierBoundarySum
        << " FepsMin=" << FepsMin
        << " FepsMax=" << FepsMax
        << " FepsAbsSum=" << FepsAbsSum
        << " FepsBoundarySum=" << FepsBoundarySum
        << endl;
}


void Foam::solvers::phaseClotAxisymmetricFluid::advectMaterialLabel
(
    const surfaceScalarField& carrier
)
{
    // THE CORE/SHELL LABEL IS A MATERIAL PROPERTY AND MUST TRAVEL WITH THE
    // MATERIAL.
    //
    // shell_ is a label attached to the stuff, not to the mesh. The moment the
    // phase field can move, a fixed label is wrong in a way that is worse than
    // having no core at all: shell material dragged downstream would arrive in
    // cells still labelled core and silently acquire the core's modulus,
    // toughness and permeability. The clot would appear to stiffen as it
    // deformed, for no reason anyone could see.
    //
    // S15: under materialLabelBasis contentComposition the label is not
    // transported at all -- it IS thetaH/theta, set by
    // syncMaterialLabelFromContent() after the contents move.
    //
    // Same carrier flux and same scheme as c, so the label and the field it
    // labels cannot separate. Clipped to [0, 1] against the small overshoots
    // any bounded scheme leaves; the material blend is only meaningful inside
    // that range.
    if (!coreShell_ || materialLabelBasis_ == "contentComposition")
    {
        return;
    }

    // THE LABEL AND THE FIELD IT LABELS ASSEMBLE THE SAME MATRIX.
    //
    // This used to be fvm::div(carrier, shell_, "div(phiReleased,c)"), which
    // shares the flux and the scheme NAME with c but not the weights: a
    // limited scheme recomputes its limiter from the field being transported,
    // so c (smooth, in [-1,1]) and shell_ (a 0/1 material label, 27293 of
    // 31360 cells at exactly 1.0 in run 20260908-010836) produced different
    // matrices from one flux. That is the same defect fvSchemes already
    // records for the platelet content -- "vanLeer weights depend on the
    // field being transported ... run 20260822-103655, |Wp - q| 2.27e-04
    // vanLeer vs 5.85e-12 with field-independent weights" -- and Wp was fixed
    // by capturing phaseWeights_. advectMaterialLabel never inherited it.
    //
    // phaseSharedConvection() is the bounded form spelled out with those
    // shared weights, and its own comment names shell_ as what it is for. It
    // keeps the -Sp(surfaceIntegrate(carrier)) term, which is correct here:
    // the label rides pathlines, so the material derivative is what it wants,
    // and that Sp is what preserves the uniform plateau under a carrier that
    // is deliberately not solenoidal.
    fvScalarMatrix shellEqn
    (
        fvm::ddt(shell_)
      + phaseSharedConvection(carrier, shell_)
    );

    // THE DIAGONAL, AND THE CARRIER'S OWN COURANT NUMBER.
    //
    // 13 of 14 two-velocity runs died of SIGFPE inside the linear solve of
    // THIS equation, and not one of them left a number behind that moved
    // beforehand. symGaussSeidelSmoother divides by the diagonal, so a
    // diagonal that reaches zero -- or a matrix that carries a non-finite
    // coefficient -- is the whole failure, and neither was instrumented.
    //
    // The carrier's Courant number is the other half. maxCo in controlDict
    // bounds phi_, the BLOOD flux; nothing has ever bounded or reported the
    // Courant number of the flux that actually transports c and shell. The
    // solver already says so in the one-velocity branch of solvePhaseField:
    // "the Courant number of THIS flux -- not the fluid one that maxCo bounds
    // -- is what limits the accuracy of the clot's motion."
    //
    // Reported, not enforced. A non-positive diagonal does not always fault
    // -- only an exact zero or an overflow does -- so killing the run here
    // would throw away the correlation this line exists to collect. The
    // diagonal is read BEFORE solve(), so it excludes the boundary
    // internalCoeffs that addBoundaryDiag folds in; it is normalised by
    // V/deltaT, which is what fvm::ddt alone would put there.
    {
        const scalarField sumMagPhi
        (
            fvc::surfaceSum(mag(carrier))().primitiveField()
        );
        const scalarField& diagS = shellEqn.diag();
        const scalarField& Vc = mesh_.V();
        const scalar dt = mesh_.time().deltaTValue();

        scalar maxCo = 0;
        scalar minDiagRatio = great;
        label coWorst = -1;
        label diagWorst = -1;
        label nonPos = 0;
        label nonFinite = 0;

        forAll(diagS, celli)
        {
            const scalar co = 0.5*dt*sumMagPhi[celli]/Vc[celli];
            if (co > maxCo)
            {
                maxCo = co;
                coWorst = celli;
            }

            const scalar dv = diagS[celli];
            if (dv != dv || mag(dv) > vGreat)
            {
                ++nonFinite;
            }
            else
            {
                if (dv <= 0)
                {
                    ++nonPos;
                }
                const scalar ratio = dv*dt/Vc[celli];
                if (ratio < minDiagRatio)
                {
                    minDiagRatio = ratio;
                    diagWorst = celli;
                }
            }
        }

        // Scalar reductions: gMax/gMin are field functions, so a bare
        // scalar goes through returnReduce with an explicit op.
        const scalar maxCoAll = returnReduce(maxCo, maxOp<scalar>());
        const scalar minDiagAll = returnReduce(minDiagRatio, minOp<scalar>());
        const label nonPosAll = returnReduce(nonPos, sumOp<label>());
        const label nonFiniteAll = returnReduce(nonFinite, sumOp<label>());

        const vector coX =
            coWorst >= 0 ? mesh_.C()[coWorst] : vector::zero;
        const vector diagX =
            diagWorst >= 0 ? mesh_.C()[diagWorst] : vector::zero;

        Info<< "PHASE_CLOT_CARRIER time=" << mesh_.time().value()
            << " maxCarrierCo=" << maxCoAll
            << " maxCarrierCoX=" << coX.x()
            << " maxCarrierCoY=" << coX.y()
            << " maxMagCarrier=" << gMax(mag(carrier.primitiveField()))
            << " minShellDiagOverVbyDt=" << minDiagAll
            << " minShellDiagX=" << diagX.x()
            << " minShellDiagY=" << diagX.y()
            << " shellDiagNonPositive=" << nonPosAll
            << " shellDiagNonFinite=" << nonFiniteAll
            << " carrierBlend=OFF"
            << nl;

        if (nonPosAll + nonFiniteAll > 0)
        {
            WarningInFunction
                << "shell matrix diagonal is not strictly positive: "
                << nonPosAll << " non-positive, "
                << nonFiniteAll << " non-finite. "
                << "The Gauss-Seidel smoother divides by this. Worst cell at ("
                << diagX.x() << " " << diagX.y() << "), carrier Courant "
                << maxCoAll << "." << endl;
        }
    }

    // THE FINAL MATRIX -- THE ONE symGaussSeidelSmoother ACTUALLY DIVIDES BY.
    //
    // The PHASE_CLOT_CARRIER block above reads shellEqn.diag() before
    // solve(), so it CANNOT see the boundary internalCoeffs that
    // addBoundaryDiag() folds in, and it never looked at the off-diagonals,
    // the source, or psi at all. 13 of 14 two-velocity runs died inside this
    // solve and every previous instrument measured a matrix that is not the
    // one that faulted. Codex named this the largest outstanding item and it
    // is fixed here.
    //
    // addBoundaryDiag/addBoundarySource are PROTECTED in fvMatrix, so the
    // assembly is reproduced from the public coefficient accessors exactly as
    // fvMatrix::solveSegregated does it:
    //
    //   diag   += internalCoeffs[patch]           at lduAddr().patchAddr()
    //   source += boundaryCoeffs[patch]           for UNCOUPLED patches only
    //
    // Off-diagonal row sums follow lduMatrix::Amul's convention --
    // Apsi[upperAddr[f]] += lower[f]*psi[lowerAddr[f]] and
    // Apsi[lowerAddr[f]] += upper[f]*psi[upperAddr[f]] -- so the owner's row
    // carries upper[f] and the neighbour's row carries lower[f]. Getting this
    // backwards would silently mis-attribute every dominance figure.
    //
    // Reported, never enforced: this must not change what the run computes.
    {
        // BIND THROUGH A CONST REFERENCE. lduMatrix::upper() and ::lower()
        // have non-const overloads that ALLOCATE the missing triangle -- a
        // matrix holding only upper_ gets a mirrored lower_ on first call,
        // which turns a symmetric system into an asymmetric one and changes
        // which solver fvMatrix::solveSegregated selects. A diagnostic that
        // silently alters the solve is worse than no diagnostic. The const
        // overloads return the existing triangle and allocate nothing.
        const fvScalarMatrix& eqn = shellEqn;
        const lduAddressing& addr = eqn.lduAddr();
        const scalarField& Vc = mesh_.V();

        scalarField finalDiag(eqn.diag());
        forAll(shellEqn.internalCoeffs(), patchi)
        {
            const labelUList& pa = addr.patchAddr(patchi);
            const scalarField& ic = shellEqn.internalCoeffs()[patchi];
            forAll(pa, facei)
            {
                finalDiag[pa[facei]] += ic[facei];
            }
        }

        scalarField finalSource(eqn.source());
        forAll(shell_.boundaryField(), patchi)
        {
            if (shell_.boundaryField()[patchi].coupled())
            {
                continue;
            }
            const labelUList& pa = addr.patchAddr(patchi);
            const scalarField& bc = shellEqn.boundaryCoeffs()[patchi];
            forAll(pa, facei)
            {
                finalSource[pa[facei]] += bc[facei];
            }
        }

        scalarField offSum(finalDiag.size(), 0.0);
        label nonFiniteOff = 0;
        if (eqn.hasUpper() || eqn.hasLower())
        {
            const labelUList& lo = addr.lowerAddr();
            const labelUList& up = addr.upperAddr();
            const scalarField& upC = eqn.upper();
            const scalarField& loC = eqn.lower();
            forAll(lo, facei)
            {
                const scalar u = upC[facei];
                const scalar l = loC[facei];
                if (!std::isfinite(u) || !std::isfinite(l))
                {
                    ++nonFiniteOff;
                    continue;
                }
                offSum[lo[facei]] += mag(u);
                offSum[up[facei]] += mag(l);
            }
        }

        scalar minMagDiag = great;
        scalar minDominance = great;
        scalar maxOffSum = 0;
        label zeroDiag = 0;
        label nonFiniteDiag = 0;
        label nonFiniteSrc = 0;
        label nonFinitePsi = 0;
        label notDominant = 0;
        label domWorst = -1;

        const scalarField& psiIn = shell_.primitiveField();

        forAll(finalDiag, celli)
        {
            const scalar d = finalDiag[celli];

            if (!std::isfinite(d)) { ++nonFiniteDiag; continue; }
            if (!std::isfinite(finalSource[celli])) { ++nonFiniteSrc; }
            if (!std::isfinite(psiIn[celli])) { ++nonFinitePsi; }

            // This is the divisor. An exact zero is the SIGFPE.
            if (d == 0) { ++zeroDiag; }

            minMagDiag = min(minMagDiag, mag(d));
            maxOffSum = max(maxOffSum, offSum[celli]);

            // Row dominance normalised by V/dt so it is comparable across
            // cells of different size; the raw margin is what bounds
            // ||A^-1||_inf, so both are wanted and the normalised one is what
            // ranks the worst row.
            const scalar delta = (mag(d) - offSum[celli])*mesh_.time().deltaTValue()/Vc[celli];
            if (delta < minDominance)
            {
                minDominance = delta;
                domWorst = celli;
            }
            if (mag(d) < offSum[celli]) { ++notDominant; }
        }

        const scalar minMagDiagAll = returnReduce(minMagDiag, minOp<scalar>());
        const scalar minDomAll = returnReduce(minDominance, minOp<scalar>());
        const scalar maxOffAll = returnReduce(maxOffSum, maxOp<scalar>());
        const label zeroAll = returnReduce(zeroDiag, sumOp<label>());
        const label nfDiagAll = returnReduce(nonFiniteDiag, sumOp<label>());
        const label nfOffAll = returnReduce(nonFiniteOff, sumOp<label>());
        const label nfSrcAll = returnReduce(nonFiniteSrc, sumOp<label>());
        const label nfPsiAll = returnReduce(nonFinitePsi, sumOp<label>());
        const label notDomAll = returnReduce(notDominant, sumOp<label>());

        const vector domX =
            domWorst >= 0 ? mesh_.C()[domWorst] : vector::zero;

        Info<< "PHASE_CLOT_SHELL_MATRIX time=" << mesh_.time().value()
            << " minMagDiagFinal=" << minMagDiagAll
            << " minRowDominance=" << minDomAll
            << " minRowDominanceX=" << domX.x()
            << " minRowDominanceY=" << domX.y()
            << " maxOffDiagRowSum=" << maxOffAll
            << " rowsNotDominant=" << notDomAll
            << " zeroDiagCells=" << zeroAll
            << " nonFiniteDiag=" << nfDiagAll
            << " nonFiniteOffDiag=" << nfOffAll
            << " nonFiniteSource=" << nfSrcAll
            << " nonFinitePsi=" << nfPsiAll
            << nl;

        if (zeroAll + nfDiagAll + nfOffAll + nfSrcAll + nfPsiAll > 0)
        {
            WarningInFunction
                << "the shell system is not solvable as assembled: "
                << zeroAll << " zero diagonals, "
                << nfDiagAll << "/" << nfOffAll << "/" << nfSrcAll << "/"
                << nfPsiAll << " non-finite diag/offdiag/source/psi. "
                << "symGaussSeidelSmoother divides by that diagonal."
                << endl;
        }
    }

    shellEqn.solve();

    shell_ = min
    (
        max(shell_, dimensionedScalar("zero", dimless, 0)),
        dimensionedScalar("one", dimless, 1)
    );
    shell_.correctBoundaryConditions();

    // The material fields follow the label. Cheap -- four algebraic
    // expressions on a 12000-cell mesh -- and skipping it would leave the
    // moduli behind while the material they belong to moved.
    updateMaterialFields();
}


void Foam::solvers::phaseClotAxisymmetricFluid::solvePlatelets()
{
    const phaseClotScopeTimer costGuard_(tPlatelets_);

    // qEff is an explicitly recorded auxiliary extension. The physical
    // transient storage, convective flux, and diagnostics use q, so the
    // conservative physical quantity remains q*w. qEff is retained only for
    // the clot-side diffusion extension that keeps the w system well posed.
    // mag(q_): see wGrowth in updateDerivedFields() for the measurement.
    // Bit-identical wherever q >= 0.
    const volScalarField qEff(max(mag(q_), qFloor_));
    // Give the flux the stable field name used by system/fvSchemes. Without
    // this, OpenFOAM derives an expression name from qEff and cannot select
    // the declared div(qPhi,w) discretisation.
    const surfaceScalarField qPhi
    (
        "qPhi",
        fvc::interpolate(q_)*phi_
    );

    {
        // THE CONSERVED FORM. Section 3.4 of
        // research/designs/AM2020_platelet_content_formulation_derivation.md
        //
        //   partial_t W + div(u W) = div(q D grad w) - delta*k*w,   W = q*w
        //
        // There is NO division by q anywhere in the assembled system, which is
        // the whole point: the w branch had to divide by a q that reaches 5e-5
        // in the clot, amplifying every discretisation error by ~2e4 per step.
        //
        // THE CARRIER IS THE PHASE FIELD'S, NOT phi. c is advected by
        // phiReleased, which inside the clot is the rigid-body flux; advecting
        // W by phi_ made w = W/q the ratio of two differently-carried fields
        // and broke it on step one. The scheme is bounded to match, because
        // phiReleased is deliberately not solenoidal. Constructor guard enforces
        // token-identity with div(phiReleased,c); header note has the argument
        // and the measurements.
        //
        // The diffusion is EXPLICIT and that is safe, not lazy: the physical
        // flux is exactly qEff*D*grad(w) with no approximation, and the
        // explicit Fourier number is D*dt/h^2 = 1e-10 * 2.5e-6 / (0.6e-6)^2
        // = 6.9e-4, three orders below the stability limit. Making it implicit
        // would require either a W-diagonal approximation of a q-weighted
        // operator or a 1/q, and both would give back what this change removes.
        //
        // The adhesion sink stays NON-NEGATIVE on the diagonal, the property
        // the predecessor derivation established: delta >= 0, k >= 0 and
        // qEff >= qFloor > 0, so delta*k/qEff >= 0.
        //
        // ONE TIME LEVEL FOR THE EXPLICIT SOURCE. This reconstruction feeds
        // fvc::laplacian(qEff*D, w) and nothing else -- Vn is built from Wp
        // directly, and the diagnostic w is rebuilt after the solve.
        //
        // It used to divide Wp at n by qEff at n+1, because
        // updateDerivedFields() runs between solvePhaseField() and this
        // function. laplacian(gamma, w) is discretely EXACTLY zero for uniform
        // w, so that quotient was the ONLY way the diffusion term could inject
        // anything -- and it injected exactly one step's change in q, every
        // step. Run 20260823-002216, with the limiter fix in place and the two
        // cases differing in nothing but D: |Wp - q| = 2.589e-11 at D = 0
        // against 4.3755e-05 at D = 1e-10.
        //
        // Using q at n for both the coefficient and the quotient restores
        // Wp^n = q^n => w == 1 exactly, and is the consistent evaluation for an
        // explicit source under Euler in any case.
        // Written as an if, not a ?:, because the two branches have different
        // C++ types -- tmp<volScalarField> from max() against volScalarField --
        // and gcc refuses the conditional even though each converts to the
        // other. On the very first platelet solve, before solvePhaseField() has
        // ever run, there is no stored old level and qEff (== q at n+1) is the
        // only thing available; that step is exact anyway, Wp0 == q0.
        volScalarField qEffOld("qEffOld", qEff);

        if (qOldPlatelet_.valid())
        {
            qEffOld = max(mag(qOldPlatelet_()), qFloor_);
        }

        // OI-62: THE RATIO w = W/q, SIGNED, FOR EVERY q.
        // research/designs/2026-10-08-OI62-platelet-ratio-at-negative-q.md s3
        //
        // W/max(|q|, qFloor) broke the invariant this whole term relies on.
        // At Wp = q it read w = -1 wherever q < 0 and +1 one row away, and
        // w = q/qFloor < 1 inside the band 0 < |q| < qFloor, so the explicit
        // diffusion carried Wp across the sign change of q. Run 20261008-081155
        // (BATCH 72) S23v_G4_r3: one step from Wp = q seeds 2.48e-4 / 2.59e-3
        // / 1.65e-3 at 25 / 30 / 35 ms, against <= 7.8e-16 with D = 0
        // (docket 20261008-1507-run081155-b72-s23-validation, 07 and 10).
        //
        // C6 (numerics 10):
        //   |q| >= qFloor:  w = W/q -- bitwise today's W/max(|q|, qFloor) for
        //                   q >= qFloor, and the signed ratio for q <= -qFloor;
        //   |q| <  qFloor:  w = 1 + (W - q)/q_s, q_s = sign(q)*max(|q|, qFloor)
        //                   with sign(0) = +1 -- the deviation form, so that
        //                   Wp = q gives exactly 1 in the band too.
        // Both denominators are >= qFloor in magnitude. The same rule builds
        // the written w after the solve, at n+1, so Wp = q*w on restart
        // (.C constructor, W = q*w) is exact for every q.
        //
        // Built on a calculated-patch temporary and ASSIGNED to w_, as
        // w_ = Wp_/qEffOld was: fixedValue patches ignore operator= (OF-13
        // fixedValueFvPatchField.H:179-193), so the inlet keeps its fixed
        // value exactly as before. The adhesion sink, wGrowth and wStoich
        // read today's operand (q + e)/max(|q|, qFloor) in every cell (S25
        // R1, rev. 2.1 amendment 10), so OI-63 stays undecided.
        const volScalarField qOldS
        (
            "qOldS",
            qOldPlatelet_.valid() ? qOldPlatelet_() : q_
        );

        // S25 B (R1): the content W is an ARGUMENT, because Wp_ is now
        // formed as q + e wherever q is formed; the operand at time level n
        // is C6(q^n + e^n, q^n) (S25 01 Q2(1)), not C6(Wp_, q^n).
        const auto plateletRatioC6 = [this]
        (
            const volScalarField& WS,
            const volScalarField& qS,
            const word& ratioName
        )
        {
            tmp<volScalarField> tRatio
            (
                volScalarField::New
                (
                    ratioName,
                    mesh_,
                    dimensionedScalar(WS.dimensions()/qS.dimensions(), 0)
                )
            );
            const scalar qFloorC6 = qFloor_.value();
            const auto c6 = [qFloorC6](const scalar W, const scalar q)
            {
                return mag(q) >= qFloorC6
                    ? W/q
                    : 1.0 + (W - q)/(sign(q)*max(mag(q), qFloorC6));
            };

            volScalarField& ratio = tRatio.ref();
            scalarField& ratioIn = ratio.primitiveFieldRef();
            const scalarField& WpIn = WS.primitiveField();
            const scalarField& qSIn = qS.primitiveField();
            forAll(ratioIn, celli)
            {
                ratioIn[celli] = c6(WpIn[celli], qSIn[celli]);
            }
            volScalarField::Boundary& ratioB = ratio.boundaryFieldRef();
            forAll(ratioB, patchi)
            {
                scalarField& ratioP = ratioB[patchi];
                const scalarField& WpP = WS.boundaryField()[patchi];
                const scalarField& qSP = qS.boundaryField()[patchi];
                forAll(ratioP, facei)
                {
                    ratioP[facei] = c6(WpP[facei], qSP[facei]);
                }
            }
            return tRatio;
        };

        const volScalarField WOldS("WOldS", qOldS + e_);
        w_ = plateletRatioC6(WOldS, qOldS, "wOperandC6");
        w_.correctBoundaryConditions();

        // THE FACE DIFFUSIVITY IS A HARMONIC MEAN, NOT AN ARITHMETIC ONE.
        //
        // Run 20260901-110348 child g20 died at 16.4274 ms and the platelet
        // field went first. The signature is not noise, it is a period-2
        // instability: wMin ran 0.3748, -1.121, 0.3747, -6.534, 0.3746,
        // -28.97 -- alternating sign, growing by 4.5x every two steps, while
        // WpDevMax, VnMagMax and beMinMinor stayed smooth to six digits.
        // That is a linear amplification factor of about -2.15 per step.
        //
        // The cause is this term. `fvc::laplacian(gamma, w)` with a VOLUME
        // gamma interpolates it to faces ARITHMETICALLY, and with
        // `w = Wp/qEff` the coefficient multiplying the cell's own Wp is
        //
        //     A_P = dt/V_P * sum_f gamma_f |Sf| deltaCoeff_f / qEff_P
        //
        // which is integrated EXPLICITLY, so the step is stable only while
        // A_P < 2. gamma = qEff*D spans four decades across the interface, and
        // an arithmetic face mean is dominated by the WET side: measured in
        // g20 at 16 ms, the worst cell (x=76.1, y=6.8 um) had q = 6.83e-06
        // with a neighbour at 1.49e-02, a ratio of 2182, giving
        // gamma_f/qEff_P ~ 1.1e3 and A_P = 0.449 and climbing 5x per ms. The
        // observed growth rate puts A_P at ~3.2 when it let go.
        //
        // The harmonic mean is the correct face conductance for a series
        // arrangement (Patankar 1980 §4.2.3) and it is what this term
        // physically is: platelets crossing from a wet cell to a dry one are
        // limited by the DRY cell, which holds no blood to carry them. The
        // arithmetic mean instead lets a wet neighbour set the flux out of a
        // cell that has nothing to give, which is both wrong and stiff.
        //
        // It is also unconditionally stable here. harmonic <= 2*min(gamma_P,
        // gamma_N) <= 2*gamma_P, so gamma_f/qEff_P <= 2D and
        //
        //     A_P <= 8*D*dt/h^2 = 8 * 4.008e-04 = 3.2e-03
        //
        // for this mesh and timestep -- a factor of 600 below the limit, and
        // independent of how sharp q becomes. The arithmetic form has no such
        // bound: it grows without limit as the interface steepens.
        //
        // Gate A is untouched. Under uniform blood w == 1, grad(w) == 0 and
        // this whole term vanishes whatever the face interpolation is.
        // D = 0 CANNOT GO THROUGH THE HARMONIC MEAN.
        //
        // Run 20260902-150654: with D = 0 the field qEffOld*D_ is identically
        // zero, and harmonic::interpolate divides BY the field -- SIGFPE at
        // step 1, inside solvePlatelets(). This is the same D = 0 family as
        // the closure trap fixed after run 20260902-141852; that fix let the
        // four children construct, and they died one step later here.
        //
        // Skipping the term is EXACT, not a guard. The harmonic mean of two
        // zeros is zero in the limit, and D = 0 means there is no diffusive
        // flux to compute. Flooring the argument instead would fabricate a
        // diffusivity the case explicitly set to nothing.
        //
        // qEffOld itself can never be the zero: it is max(..., qFloor_) with
        // qFloor_ = 1e-06 on both branches, so this triggers on D and only D.
        // Verified across all 119 children -- exactly four set D = 0 and all
        // four are gate children; no sweep child does.
        // Built through a tmp so that the D != 0 path uses EXACTLY the
        // original (word, tmp) constructor on the harmonic result. An earlier
        // version of this fix constructed a zero field and assigned into it,
        // which would have given gammaPlateletf `calculated` boundary patch
        // types rather than the ones the interpolation produces. It is
        // consumed by fvc::laplacian(gammaPlateletf, w_) below, where the
        // BOUNDARY field carries real flux -- so that difference could have
        // changed every D > 0 child silently while looking like a pure
        // D = 0 guard. Not worth the risk for zero benefit.
        tmp<surfaceScalarField> tGammaPlateletf;
        if (D_.value() != 0.0)
        {
            tGammaPlateletf = harmonic(mesh_).interpolate(qEffOld*D_);

            // OI-62: NO FREE BLOOD, NO PLATELET DIFFUSION. q = 1 - theta/
            // theta_ref is the free-blood fraction, so a face with a
            // q_old <= 0 side has no blood to carry platelets across it and
            // its conductance is exactly 0 (physics 09; Capek eq. 4.20). The
            // harmonic mean above still sees |q|*D there, as if the cell held
            // |q| of blood (25% of bulk at q = -0.246).
            //
            // MASKED IN PLACE on OpenFOAM's own result, so every face with no
            // dry side keeps today's value BITWISE, band faces included, and
            // the patch types stay the interpolation's (see the tmp note
            // above). Not a hand-written mean: physics 09's gP*gN/(...) form
            // disagrees with OF-13's harmonic on 55% of random face pairs
            // (red team 11). Its argument is >= qFloor*D > 0, so nothing
            // divides by zero; the zero is written afterwards. A boundary
            // face is dry when its patch value or its owner is.
            surfaceScalarField& gammaMasked = tGammaPlateletf.ref();
            const scalarField& qOldIn = qOldS.primitiveField();
            const labelUList& own = mesh_.owner();
            const labelUList& nei = mesh_.neighbour();
            scalarField& gammaIn = gammaMasked.primitiveFieldRef();
            forAll(gammaIn, facei)
            {
                if (qOldIn[own[facei]] <= 0 || qOldIn[nei[facei]] <= 0)
                {
                    gammaIn[facei] = 0;
                }
            }
            surfaceScalarField::Boundary& gammaB =
                gammaMasked.boundaryFieldRef();
            forAll(gammaB, patchi)
            {
                const scalarField& qOldP = qOldS.boundaryField()[patchi];
                const labelUList& fc = mesh_.boundary()[patchi].faceCells();
                scalarField& gammaP = gammaB[patchi];
                forAll(gammaP, facei)
                {
                    if (qOldP[facei] <= 0 || qOldIn[fc[facei]] <= 0)
                    {
                        gammaP[facei] = 0;
                    }
                }
            }
        }
        else
        {
            tGammaPlateletf = tmp<surfaceScalarField>
            (
                new surfaceScalarField
                (
                    IOobject
                    (
                        "gammaPlateletfZero", mesh_.time().name(), mesh_,
                        IOobject::NO_READ, IOobject::NO_WRITE
                    ),
                    mesh_,
                    dimensionedScalar("zero", D_.dimensions(), 0.0)
                )
            );
        }
        const surfaceScalarField gammaPlateletf
        (
            "gammaPlateletf",
            tGammaPlateletf
        );

        // Measured every step, because the failure above was silent in every
        // existing diagnostic. plateletFo is A_P above; the explicit scheme
        // is unstable at 2.
        scalar plateletFo = 0;
        {
            const scalarField sumGamma
            (
                fvc::surfaceSum
                (
                    gammaPlateletf*mesh_.magSf()*mesh_.deltaCoeffs()
                )().primitiveField()
            );
            const scalarField& qEffOldIn = qEffOld.primitiveField();
            const scalarField& VIn = mesh_.V();
            const scalar dtValue = mesh_.time().deltaTValue();
            forAll(sumGamma, celli)
            {
                plateletFo = max
                (
                    plateletFo,
                    dtValue*sumGamma[celli]/(VIn[celli]*qEffOldIn[celli])
                );
            }
            reduce(plateletFo, maxOp<scalar>());
        }

        // OI-62: faces with a q_old <= 0 side that still conduct, counted on
        // the gammaPlateletf the laplacian reads. 0 by construction of the
        // mask above; the BATCH 73 gate fails any nonzero value. Serial
        // production (Allrun nProcs 1): on a decomposed mesh a coupled face
        // would count once per side.
        label dryFacesConducting = 0;
        {
            const scalarField& qDryIn = qOldS.primitiveField();
            const labelUList& ownDry = mesh_.owner();
            const labelUList& neiDry = mesh_.neighbour();
            const scalarField& gammaDryIn = gammaPlateletf.primitiveField();
            forAll(gammaDryIn, facei)
            {
                const bool drySide =
                    qDryIn[ownDry[facei]] <= 0 || qDryIn[neiDry[facei]] <= 0;
                if (drySide && gammaDryIn[facei] != 0)
                {
                    ++dryFacesConducting;
                }
            }
            forAll(gammaPlateletf.boundaryField(), patchi)
            {
                const scalarField& qDryP = qOldS.boundaryField()[patchi];
                const labelUList& fcDry = mesh_.boundary()[patchi].faceCells();
                const scalarField& gammaDryP =
                    gammaPlateletf.boundaryField()[patchi];
                forAll(gammaDryP, facei)
                {
                    const bool drySide =
                        qDryP[facei] <= 0 || qDryIn[fcDry[facei]] <= 0;
                    if (drySide && gammaDryP[facei] != 0)
                    {
                        ++dryFacesConducting;
                    }
                }
            }
            reduce(dryFacesConducting, sumOp<label>());
        }

        Info<< "PHASE_CLOT_PLATELET_FO time=" << mesh_.time().userTimeValue()
            << " plateletFo=" << plateletFo
            << " plateletFoLimit=2"
            << " dryFacesConducting=" << dryFacesConducting
            << endl;

        if (plateletFo > 1.0)
        {
            Info<< "PHASE_CLOT_PLATELET_FO_WARNING time="
                << mesh_.time().userTimeValue()
                << " plateletFo=" << plateletFo
                << " -- explicit platelet diffusion is within a factor of two"
                   " of its stability limit; the content field is no longer"
                   " trustworthy" << endl;
        }

        // S25 B: the accepted deposit as e's source, +Y dB/(theta_ref dt).
        // theta_ref = skeletonVolumeFraction_ > 0 is q's own divisor
        // (updateDerivedFields) and dt > 0.
        volScalarField::Internal eDepositS25
        (
            IOobject
            (
                "eDeposit", mesh_.time().name(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar(e_.dimensions()/dimTime, 0)
        );
        eDepositS25.primitiveFieldRef() =
            plateletDepositStep_
           /(skeletonVolumeFraction_*mesh_.time().deltaTValue());

        // S25.1 B (OI-78; design 2026-10-13-S25.1 rev. 2 section B; docket
        // 20261013-1405, 01 Q2 and 03 Q3): w's diffusion acts on the
        // DEVIATION w - 1. At e == 0, w == 1 exactly (C6), so the operand is
        // exactly 0 in every cell and on every patch, its gradient is exactly
        // 0, and so is the non-orthogonal correction that carried grad(1)
        // round-off (sum S_f/V) into e on the throat (e <= 1.3e-16). With w's
        // BCs it is the same operator: on zeroGradient patches
        // (w_b - 1) - (w_P - 1) = 0, w's own snGrad; on the inlet (fixedValue
        // 1, here calculated 0) deltaCoeffs*(0 - (w_P - 1)). It is formed
        // AFTER w_.correctBoundaryConditions() above, so its patch values are
        // this step's. The scheme name is passed EXPLICITLY and is today's:
        // fvcLaplacian.C names it "laplacian(" + gamma.name() + ',' +
        // vf.name() + ')', i.e. laplacian(gammaPlateletf,w).
        const volScalarField wDevS251
        (
            "wDev",
            w_ - dimensionedScalar("one", w_.dimensions(), 1)
        );
        const word wLapSchemeS251("laplacian(gammaPlateletf,w)");

        if (phaseCarrier_.valid())
        {
            // THE SAME OPERATOR AS c, NOT MERELY THE SAME SCHEME NAME.
            //
            // fvm::div(carrier, Wp_, "div(phiReleased,Wp)") was the previous
            // form and it is what run 20260822-103655 falsified: the scheme
            // strings were token-identical, the flux was one shared object,
            // every source term was zero and Wp started equal to q -- and
            // |Wp - q| still reached 2.27e-04, because vanLeer weights are a
            // function of the TRANSPORTED FIELD and c and Wp are different
            // fields. The same run measured 5.85e-12, flat, once the weights
            // were made field-independent.
            //
            // gaussConvectionScheme::fvmDiv calls interpScheme().weights(vf).
            // suppliedWeightsScheme ignores vf and returns the captured
            // weights, so the two equations assemble the same matrix and
            // Wp = q survives.
            //
            // AMENDED 2026-09-02, AND THE FIRST AMENDMENT WAS WRONG.
            //
            // The captured weights are no longer c's alone:
            // phaseFieldFaceWeights() returns the MORE UPWIND of c's vanLeer
            // limiter and Wp's own, because c's limiter is TVD for c and for
            // nothing else.
            //
            // I then wrote here that "the two matrices are still identical
            // wherever the limiters agree to 1e-06, which includes every
            // gate-A case by the affine invariance of the limiter". RUN
            // 20260902-104340 REFUTED THAT DIRECTLY. On step 1 of gate_a_exact,
            // with WpDevMax = 0 -- Wp equal to q to the last bit -- the
            // override still fired on 1600 faces. Affine invariance is exact
            // in real arithmetic and NOT in floating point: q = 0.5*(c+1)
            // cancels catastrophically where c -> -1, so deep in the clot
            // q carries only a few significant digits, the vanLeer r-ratio
            // built from neighbouring q differences is far worse, and the two
            // limiters separate by more than 1e-06.
            //
            // That alone was harmless. What was NOT harmless is that the c
            // equation went on calling fvm::div(phi, c_, "div(phiReleased,c)"),
            // which resolves `bounded Gauss vanLeer` from fvSchemes and
            // rebuilds C'S OWN weights, while Wp used the captured ones. The
            // two fields silently assembled DIFFERENT matrices from that step
            // onward and gate A-exact went 6.686e-12 -> 1.784e-05, growing as
            // t^2 to 3.11e-03 by 15 ms.
            //
            // Both equations now go through phaseSharedConvection(), which is
            // the only place the operator is built. Identical weights are then
            // a structural property rather than a claim about limiters.
            //
            // AMENDED 2026-09-05: CONSERVATIVE, BECAUSE c IS.
            //
            // This equation used phaseSharedConvection() -- the bounded form,
            // carrying -Sp(surfaceIntegrate(carrier), Wp) -- and the comment
            // here said it was not optional "for the same reason the c
            // equation carries it". The c equation no longer carries it: c is
            // a VOLUME FRACTION and is now transported conservatively, which
            // is what stopped the clot growing 4x. That rationale is gone, and
            // leaving this equation on the old operator would break the one
            // structural property this whole shared-operator apparatus exists
            // to protect.
            //
            // WHY Wp MUST FOLLOW c EXACTLY. Write R[f] = ddt(f) + div(carrier
            // f). Because the carrier is not solenoidal, R[1] = SI(carrier),
            // NOT zero -- that single fact is what breaks the old pairing.
            // With c conservative, R[c] = SI + S_c, so for q = 0.5(c+1),
            //
            //     R[q] = 0.5*R[c] + 0.5*R[1] = SI + 0.5*S_c
            //
            // Keeping this equation bounded would demand chi*SI = 0 for
            // Wp = q to survive -- false wherever the skeleton dilates, which
            // is the whole clot, at ~300 1/s. Making it conservative but
            // WITHOUT the SI source would demand SI = 0, equally false. Only
            // the same pair c uses works: conservative convection plus
            // +carrierDivergence. Then at Wp = q,
            //
            //     R[Wp] = SI - (delta*k/qEff)*q     must equal    SI + 0.5*S_c
            //  => 0.5*S_c + (delta*k/qEff)*q = 0
            //
            // which is EXACTLY the stoichiometric closure the solver already
            // enforces and reports as VnClosureResidual = 0. The invariant is
            // preserved for the same reason it was before, not a new one.
            //
            // Physically the SI term is not bookkeeping: where the network
            // dilates the cell holds more fluid, and at saturation it holds
            // proportionally more bound platelets. Wp is a CONTENT, so it is
            // created by dilation exactly as q is.
            //
            // Same weights as before (phaseWeights_), same carrier, and now
            // the same operator as c again -- so the two matrices are
            // identical, which is the property run 20260902-104340 showed is
            // not safe to assume from limiter agreement alone.
            const surfaceScalarField& carrier = phaseCarrier_();

            // S25 B, R1 (design 2026-10-10-S25 section 2B and 3 B-G; S25 01
            // Q2; physics 02 Q1-Q3): THE DEVIATION e = Wp - q IS TRANSPORTED,
            // NOT Wp. Its matrix is theta's (contentFaceWeights_, the same
            // ddt and conservative convection); the carrier divergence and
            // q's own inflow cancel against q's equation and are absent. The
            // sources are events and the laws only:
            //   +lap(gamma, w^n - 1), w^n = C6(q^n + e^n, q^n): Wp's
            //     diffusion, kept (red team 04 change 6); exactly 0 at e == 0
            //     (w == 1; S25.1 B: the operand is the deviation);
            //   the adhesion sink, TODAY'S: -(delta k) Wp/qEff with
            //     Wp = q + e and qEff = max(|q|, qFloor), so implicit
            //     Sp(delta k/qEff, e) and explicit -delta k q/qEff (design
            //     rev. 2.1 amendment 10: the band law is WITHDRAWN, Alireza
            //     2026-10-10, "Revert to today's sink"; its fixed point was
            //     Wp = q - qFloor < 0 in band cells, V-A 13 finding 5). For
            //     |q| >= qFloor, q/qEff = q/|q| = +-1 exactly (IEEE), so the
            //     explicit part is -delta k sign(q) bit for bit; in the band
            //     it is -delta k q/qFloor, today's. OI-63 stays undecided;
            //   +Y dB/(theta_ref dt): the accepted deposit T (Wp does not
            //     follow q at deposition).
            // The release books nothing (e unchanged; Wp is re-formed in
            // updateDerivedFields after it). At e^n == 0 with no adhesion and
            // no deposit the right-hand side is exactly 0 and so is e.
            fvScalarMatrix eEqn
            (
                fvm::ddt(e_)
              + plateletContentConvection(carrier, e_)
              + fvm::Sp(interfaceDelta_*adhesionSpeedEff_/qEff, e_)
              - eDepositS25
             ==
                fvc::laplacian(gammaPlateletf, wDevS251, wLapSchemeS251)
              - interfaceDelta_*adhesionSpeedEff_*(q_/qEff)
            );
            // Wp's solver controls (no new fvSolution key): e replaces Wp's
            // transport.
            eEqn.solve
            (
                mesh_.solution().solverDict
                (
                    Wp_.select
                    (
                        !mesh_.schemes().steady()
                     && solutionControl::finalIteration(mesh_)
                    )
                )
            );

            // S10 FIX 3 carried to e: the content's limited (linear - upwind)
            // correction with the content's own limiter lambda_c, on e's
            // antidiffusive flux (design section 3 B-G). Wp's own limiter is
            // gone with Wp's transport: at e == 0, A(e) = 0 and nothing moves.
            if
            (
                contentFctLambda_.valid()
            )
            {
                const surfaceScalarField AE(fctAntidiffusiveFlux(carrier, e_));
                fctApply(e_, contentFctLambda_(), AE);
            }
        }
        else
        {
            // c was not advected on this step, so neither is e: the same
            // sources, no convection.
            fvScalarMatrix eEqn
            (
                fvm::ddt(e_)
              + fvm::Sp(interfaceDelta_*adhesionSpeedEff_/qEff, e_)
              - eDepositS25
             ==
                fvc::laplacian(gammaPlateletf, wDevS251, wLapSchemeS251)
              - interfaceDelta_*adhesionSpeedEff_*(q_/qEff)
            );
            eEqn.solve
            (
                mesh_.solution().solverDict
                (
                    Wp_.select
                    (
                        !mesh_.schemes().steady()
                     && solutionControl::finalIteration(mesh_)
                    )
                )
            );
        }
        e_.correctBoundaryConditions();
        // Wp := q + e^{n+1} (q is this step's: no content solve since).
        Wp_.primitiveFieldRef() = q_.primitiveField() + e_.primitiveField();
        Wp_.correctBoundaryConditions();
        {
            const scalar eAbsMaxS25 = gMax(mag(e_.primitiveField())());
            Info<< "PHASE_CLOT_S25_R1 time=" << mesh_.time().userTimeValue()
                << " eAbsMax=" << eAbsMaxS25
                << " depositBooked="
                << gSum(plateletDepositStep_*mesh_.V().primitiveField())
                << endl;
        }

        // THE LED AUDIT -- does the Wp convection CREATE a local extremum?
        //
        // Wp is advected with the phase field's vanLeer weights
        // (suppliedWeightsScheme) under phasePlateletFluxBasis 'legacy',
        // deliberately, so that Wp = q survives to machine precision under
        // uniform blood -- an argument that holds only while q is defined by
        // the field those weights were built from. Under 'content' Wp takes
        // contentFaceWeights_ instead and this paragraph's hazard does not
        // arise, because upwind weights are field-independent. The price, never measured until now, is that a
        // limiter is TVD only for the field it was BUILT FROM: c's psi can
        // select a near-downwind face value at a face where Wp -- a different
        // field -- has a local extremum, and downwind differencing amplifies
        // extrema instead of damping them.
        //
        // Run 20260901-110348 gave NO evidence of this happening: wMax never
        // exceeded 1.00000000001 and wOverBound stayed 0 across five children
        // and 14670 steps each. So nothing was changed. This is the instrument
        // that settles it either way instead of leaving it a standing worry.
        //
        // WHAT IT MEASURES. For an M-matrix system the discrete maximum
        // principle gives Wp^{n+1}_P <= max(Wp^n_P, Wp^n over face-neighbours)
        // and >= the corresponding min. The OVERSHOOT is the sharp signal:
        // the adhesion term is a pure sink, the harmonic explicit diffusion at
        // Fo ~ 2.4e-03 is a convex combination, and this equation carries no
        // growth source -- so nothing but the convection weights can lift a
        // cell above every one of its neighbours. An undershoot is reported
        // too but is NOT diagnostic on its own, because the sink can produce
        // one legitimately.
        //
        // RESOLUTION. Wp solves to tolerance 1e-12 absolute with Wp = O(1) in
        // blood, so the 1e-09 counting threshold is a thousand times the noise
        // floor -- a limit this instrument can actually resolve.
        {
            const scalarField& WpNew = Wp_.primitiveField();
            const scalarField& WpOld = Wp_.oldTime().primitiveField();
            scalarField loLED(WpOld);
            scalarField hiLED(WpOld);

            const labelUList& ownLED = mesh_.owner();
            const labelUList& neiLED = mesh_.neighbour();
            forAll(ownLED, facei)
            {
                const label cP = ownLED[facei];
                const label cN = neiLED[facei];
                loLED[cP] = min(loLED[cP], WpOld[cN]);
                hiLED[cP] = max(hiLED[cP], WpOld[cN]);
                loLED[cN] = min(loLED[cN], WpOld[cP]);
                hiLED[cN] = max(hiLED[cN], WpOld[cP]);
            }

            // Boundary values are legitimate bounds: the inlet holds Wp = 1
            // and a cell beside it may rise to meet it without that being a
            // new extremum.
            const volScalarField::Boundary& WpOldBf =
                Wp_.oldTime().boundaryField();
            forAll(WpOldBf, patchi)
            {
                const fvPatchScalarField& pf = WpOldBf[patchi];
                const labelUList& bfc = pf.patch().faceCells();
                forAll(bfc, i)
                {
                    loLED[bfc[i]] = min(loLED[bfc[i]], pf[i]);
                    hiLED[bfc[i]] = max(hiLED[bfc[i]], pf[i]);
                }
            }

            scalar over = 0;
            scalar under = 0;
            label violCells = 0;
            scalar overX = 0;
            scalar overY = 0;
            const vectorField& ccLED = mesh_.C();
            forAll(WpNew, celli)
            {
                const scalar o = WpNew[celli] - hiLED[celli];
                const scalar u = loLED[celli] - WpNew[celli];
                if (o > over)
                {
                    over = o;
                    overX = ccLED[celli].x();
                    overY = ccLED[celli].y();
                }
                if (u > under)
                {
                    under = u;
                }
                if (o > 1.0e-9 || u > 1.0e-9)
                {
                    ++violCells;
                }
            }

            const scalar overAll = returnReduce(over, maxOp<scalar>());
            scalar overXAll = (over == overAll && over > 0) ? overX : great;
            scalar overYAll = (over == overAll && over > 0) ? overY : great;
            reduce(overXAll, minOp<scalar>());
            reduce(overYAll, minOp<scalar>());
            if (overAll <= 0)
            {
                overXAll = 0;
                overYAll = 0;
            }
            reduce(under, maxOp<scalar>());
            reduce(violCells, sumOp<label>());

            // THE UNAMBIGUOUS TEST, added 2026-09-02.
            //
            // The overshoot/undershoot pair above compares Wp^{n+1} against
            // the OLD time level's face-neighbour range. For an IMPLICIT
            // scheme that is not a boundedness criterion: information travels
            // further than one cell per step through the matrix, so a bounded
            // scheme can legitimately exceed its old neighbours. Run
            // 20260901-200203 proved this the expensive way -- WpOvershoot
            // reached 7.3e-06 in every child and was reported as the vanLeer
            // hole firing, while max(Wp) never left 1.00000001. The local
            // test was a FALSE POSITIVE and the conclusion drawn from it was
            // wrong.
            //
            // WpMin and WpMax are the criterion that cannot be argued with.
            // There is no source in this equation and the inlet holds Wp = 1,
            // so Wp must stay in [0, 1] for all time. Wp < 0 is a negative
            // platelet content and is the defect the more-upwind limiter
            // exists to remove; it was REAL in run 20260901-200203
            // (min(Wp) = -5.32e-05 in g30 from 25 ms).
            scalar WpMinAll = great;
            scalar WpMaxAll = -great;
            forAll(WpNew, celli)
            {
                WpMinAll = min(WpMinAll, WpNew[celli]);
                WpMaxAll = max(WpMaxAll, WpNew[celli]);
            }
            reduce(WpMinAll, minOp<scalar>());
            reduce(WpMaxAll, maxOp<scalar>());

            Info<< "PHASE_CLOT_PLATELET_LED time="
                << mesh_.time().userTimeValue()
                << " WpMin=" << WpMinAll
                << " WpMax=" << WpMaxAll
                << " WpOvershoot=" << overAll
                << " WpUndershoot=" << under
                << " violCells=" << violCells
                << " overX=" << overXAll
                << " overY=" << overYAll
                << endl;
        }

        // The MECHANISM half of the same question. The phase field's vanLeer
        // weights (built from c or from theta/theta_ref, per
        // phaseLimiterBasis) blend
        // upwind and linear; "upwindness" is 1 for pure upwind, 0.5 for
        // central and 0 for pure DOWNWIND, which vanLeer reaches at psi = 2.
        // Only a downwind-biased face can create an extremum, so if the LED
        // audit above ever fires, the answer to "was it the weights?" is
        // whether these two fire in the same place at the same step.
        if (phaseCarrier_.valid() && phaseWeights_.valid())
        {
            const scalarField& lamIn = phaseWeights_().primitiveField();
            const scalarField& carIn = phaseCarrier_().primitiveField();
            scalar minUpwindness = 1.0;
            label downwindFaces = 0;
            forAll(lamIn, facei)
            {
                const scalar up =
                    (carIn[facei] > 0) ? lamIn[facei] : (1.0 - lamIn[facei]);
                if (up < minUpwindness)
                {
                    minUpwindness = up;
                }
                if (up < 0.5)
                {
                    ++downwindFaces;
                }
            }
            reduce(minUpwindness, minOp<scalar>());
            reduce(downwindFaces, sumOp<label>());

            Info<< "PHASE_CLOT_PLATELET_WGT time="
                << mesh_.time().userTimeValue()
                << " minUpwindness=" << minUpwindness
                << " downwindFaces=" << downwindFaces
                << endl;
        }

        // w is a DERIVED diagnostic from here on: written for ParaView and read
        // by the PHASE_CLOT_PLATELET gate, never solved and never differentiated
        // to build Vn. OI-62: C6 at n+1, the same rule as the operand above,
        // so w = 1 wherever Wp = q for either sign of q. It feeds wMin, wMax,
        // wOverBound and wDevBlood, and the restart's Wp = q*w.
        w_ = plateletRatioC6(Wp_, q_, "wC6");
        w_.correctBoundaryConditions();
    }
    w_.correctBoundaryConditions();

}



Foam::scalar Foam::solvers::phaseClotAxisymmetricFluid::maxDeltaT() const
{
    scalar dt = incompressibleFluid::maxDeltaT();

    // S25 D (design 2026-10-10-S25 section 2D; OI-69): each limit, stored
    // for PHASE_CLOT_DT. The base solver's two, as fluidSolver::maxDeltaT()
    // forms them (OF-13 fluidSolver.C L229-239, read L54-60): the fluid
    // Courant limit and the controlDict/fvModels maxDeltaT; and foamRun's
    // growth limit, deltaTFactor*dt_prev (setDeltaT.C adjustDeltaT).
    {
        const dictionary& cdS25 = runTime.controlDict();
        const scalar maxCoS25 = cdS25.lookupOrDefault<scalar>("maxCo", vGreat);
        s25DtFluidCo_ =
            (maxCoS25 < vGreat && CoNum > small)
          ? maxCoS25/CoNum*runTime.deltaTValue()
          : great;
        s25DtMax_ =
            min
            (
                fvModels().maxDeltaT(),
                cdS25.found("maxDeltaT")
              ? cdS25.lookup<scalar>("maxDeltaT", runTime.userUnits())
              : vGreat
            );
        s25DtGrowth_ = solver::deltaTFactor*runTime.deltaTValue();
        s25DtWave_ = great;
        s25DtKV_ = great;
        s25DtKVTangent_ = great;
    }

    // The fluid Courant number is the wrong one once c rides the mixture
    // material velocity. In run 20260815-182036 the fluid stayed inside its
    // maxCo of 0.5 while vsCourant climbed 1.16 -> 1.31 -> 1.45, because the
    // skeleton moves at up to 0.060 m/s against a mean flow of 0.0083 -- seven
    // times faster than the fluid whose Courant number was setting dt.
    //
    // Advection above Courant one does not transport, it smears: run
    // 20260811-195034 lost c_min from -0.997 to -0.145 at Co 16 while the
    // clot appeared to slide downstream. The phase equation is implicit, so
    // this bounds ACCURACY, not stability, which is exactly why nothing
    // crashed and nothing reported it.
    // THE ELASTIC COURANT LIMIT, AND WHY IT REPLACES THE MATERIAL ONE.
    //
    // In the Eulerian formulation the elastic stress is added explicitly to
    // the momentum equation, so dt is bounded by the shear wave it supports,
    // c_s = sqrt(G/rho). At the main case's h = 1 um that is 1.6e-5 s in the
    // 4 Pa shell and 5.1e-6 s in the 40 Pa core at Courant one -- the core
    // limit is BELOW the 7.4e-6 s the main case was taking, so this is a real
    // constraint and not a formality.
    //
    // The material-Courant limiter is skipped here on purpose. It exists
    // because c rode a skeleton velocity computed from the Us increment, and
    // it carried a self-feeding failure mode (run 20260815-191231: v_s 0.19 ->
    // 386 m/s, dt collapsed to 1.3e-9 while every step reported converged).
    // With one velocity, c is advected by U, whose Courant number the base
    // solver already limits. Us is now a passive diagnostic that follows U, so
    // its increment carries no information the fluid limit does not already
    // have, and reading dt off it would resurrect the feedback loop for
    // nothing.
    {
        if (elasticCourant_ > 0)
        {
            // gMax over the whole field: G is the clot modulus inside the clot
            // and the elastic stress is faded out by phiS in the blood, so the
            // maximum is attained where the wave actually propagates. Taking
            // the maximum is the conservative direction.
            const scalar gMaxValue = gMax(GField_.primitiveField());
            const scalar rhoValue = rho_.value();

            if (gMaxValue > 0 && rhoValue > 0)
            {
                // S25 (design section 3A; red team 04 change 5): the
                // tangent modulus, max over cells of G (3/4) stiff (= G at
                // be = I). Every cell, as gMax(G). Read by the wave CFL under
                // elasticCourantModulus tangent (LT only) and logged as
                // dtKVTangent always.
                scalar gTangentMax = 0;
                {
                    const scalarField& GTan = GField_.primitiveField();
                    const symmTensorField& beTan = be_.primitiveField();
                    forAll(GTan, celli)
                    {
                        gTangentMax =
                            max
                            (
                                gTangentMax,
                                GTan[celli]*0.75
                               *intactTangentStiffness(beTan[celli])
                            );
                    }
                    reduce(gTangentMax, maxOp<scalar>());
                }
                const bool waveOnTangent = elasticCourantModulus_ == "tangent";
                if (waveOnTangent && !(gTangentMax > 0))
                {
                    FatalErrorInFunction
                        << "elasticCourantModulus tangent: max over cells "
                        << "of G (3/4) stiff = " << gTangentMax << " is not "
                        << "> 0 while max G = " << gMaxValue << " > 0; stiff "
                        << "> 0 for SPD be, so this is a defect."
                        << exit(FatalError);
                }
                const scalar cShear =
                    Foam::sqrt
                    (
                        (waveOnTangent ? gTangentMax : gMaxValue)/rhoValue
                    );

                // Same length scale as the material limiter used, so the two
                // cannot disagree about what h means.
                const scalar hMinElastic =
                    max(Foam::pow(gMin(mesh.V()), 1.0/3.0), vSmall);

                dt = min(dt, elasticCourant_*hMinElastic/cShear);
                s25DtWave_ = elasticCourant_*hMinElastic/cShear;

                // THE SECOND ELASTIC LIMIT, AND RUN 20260817-052515 SHOWED IT
                // IS THE ONE THAT ACTUALLY BINDS IN A STIFF CASE.
                //
                // The wave CFL above bounds how far a shear wave travels in a
                // step. It says nothing about how fast the material RELAXES
                // onto its own equilibrium, which is the Kelvin-Voigt time
                // tau_kv = mu/G: the clot shears until the elastic stress
                // balances the viscous traction, and it does so on tau_kv, not
                // on h/c_s. With the stress lagged one step, a dt above tau_kv
                // overshoots the balance and the answer becomes dt-dependent.
                //
                // Measured, on the slab gate at nu = 1e-6: dt/tau_kv was 1.35,
                // 4.3 and 13.5 at G = 100, 1000 and 10000 Pa, and the L2 error
                // in the clot strain tracked it -- 0.175, 0.462, 0.934. The
                // halved-time-step control isolated it cleanly, since it
                // changes dt and nothing else: L2 0.462 -> 0.329, the
                // simple-shear pattern error 2.58e-6 -> 6.18e-7, det error
                // 8.8e-6 -> 1.8e-6.
                //
                // NOT BINDING IN PRODUCTION, which is why this is a guard
                // rather than a new cost. With mu = rho*nu = 3.5e-3 Pa s and
                // Xu's G = 4 Pa (shell) / 40 Pa (core), tau_kv is 8.8e-4 s and
                // 8.8e-5 s, so the limit below is 1.8e-4 s and 1.8e-5 s
                // against a step of 7.4e-6 s. The wave CFL and the fluid
                // Courant number both bind first there.
                const scalar tauKelvinVoigt =
                    rhoValue*nu_.value()/gMaxValue;

                dt = min(dt, elasticRelaxCourant_*tauKelvinVoigt);
                s25DtKV_ = elasticRelaxCourant_*tauKelvinVoigt;

                // S25 (design section 3A; red team 04 change 5): the same
                // limit on the TANGENT modulus gTangentMax (above), LOGGED
                // and not applied. The coded KV reads G0 (above).
                s25DtKVTangent_ =
                    gTangentMax > 0
                  ? elasticRelaxCourant_*rhoValue*nu_.value()/gTangentMax
                  : great;
            }
        }

        // S11: THE TWO EXPLICIT TERMS THE EULERIAN PATH NEVER BOUNDED.
        //
        // The skeleton carrier's Courant number, computed exactly as
        // OpenFOAM's CourantNo computes the fluid's, held to the fluid's own
        // 0.5; and the compact network pressure's diffusion number, held
        // under the 2-D explicit limit. Both are numerical-stability bounds
        // on dt for explicit terms; neither touches a field.
        {
            const scalarField sumPhi
            (
                fvc::surfaceSum(mag(phiSkelOwn_))().primitiveField()
            );
            const scalarField& vol = mesh.V();
            const scalar coRate = 0.5*gMax(sumPhi/vol);
            // S18 F: the limit as computed, for the per-step log
            s18CarrierDtLimit_ = coRate > vSmall ? 0.5/coRate : great;
            if (coRate > vSmall)
            {
                dt = min(dt, 0.5/coRate);
            }
            if (explicitDiffusionRate_ > vSmall)
            {
                dt = min(dt, 0.2/explicitDiffusionRate_);
            }
            // S25 D: the network-pressure diffusion limit, from the block
            // form (solveBlockCoupledStep) or the segregated form
            // (correctPressureTwoVelocity), whichever path ran.
            s25DtNetDiff_ =
                explicitDiffusionRate_ > vSmall
              ? 0.2/explicitDiffusionRate_
              : great;
        }

        return dt;
    }

}


void Foam::solvers::phaseClotAxisymmetricFluid::prePredictor()
{
    // foamRun calls this inside the outer PIMPLE loop. Solve the coupled
    // phase/platelet split only once per physical time step.
    if (pimple.firstIter())
    {
        // S25 D (design 2026-10-10-S25 section 2D; OI-69): every dt limit
        // that set THIS step's dt (maxDeltaT() ran in adjustDeltaT, before
        // runTime++), the binder -- the argmin, or writeAlign when deltaT
        // sits below every limit (Time::adjustDeltaT only lowers it) -- and
        // the block-path network bound's argmax cell as the block priced it
        // last step (the step whose rate set this limit).
        {
            const scalar dtS25 = runTime.deltaTValue();
            const scalar dtCarrierS25 = s18CarrierDtLimit_;
            const scalar limS25[7] =
            {
                s25DtFluidCo_, s25DtWave_, s25DtKV_, dtCarrierS25,
                s25DtNetDiff_, s25DtMax_, s25DtGrowth_
            };
            const char* const limNameS25[7] =
            {
                "fluidCo", "wave", "KV", "carrier", "netDiff", "max", "growth"
            };
            label argS25 = 0;
            for (label k = 1; k < 7; ++k)
            {
                if (limS25[k] < limS25[argS25]) { argS25 = k; }
            }
            const bool writeAlignS25 =
                dtS25 < limS25[argS25]*(1.0 - 1e-12);
            Info<< "PHASE_CLOT_DT step=" << runTime.timeIndex()
                << " time=" << runTime.userTimeValue()
                << " deltaT=" << dtS25
                << " dtFluidCo=" << s25DtFluidCo_
                << " dtWave=" << s25DtWave_
                << " dtKV=" << s25DtKV_
                << " dtKVTangent=" << s25DtKVTangent_
                << " dtCarrier=" << dtCarrierS25
                << " dtNetDiff=" << s25DtNetDiff_
                << " netDiffForm="
                << (s19BlockCoupled() ? "block" : "segregated")
                << " dtMax=" << s25DtMax_
                << " dtGrowth=" << s25DtGrowth_
                << " binder="
                << (writeAlignS25 ? "writeAlign" : limNameS25[argS25])
                << " threads="
                << (
                       blockPardiso_.valid()
                     ? label(blockPardiso_->threads())
                     : blockSolverThreads_
                   )
                << " netDiffCell=" << s25NetDiffCell_
                << " netDiffIAS=" << s25NetDiffIAS_
                << " netDiffR3As=" << s25NetDiffR3As_
                << " netDiffW=" << s25NetDiffW_
                << " netDiffA=" << s25NetDiffA_
                << endl;
        }

        // S16: the per-step maxima start again at every step.
        s16LawReleaseMax_ = 0;
        s16LawRate_ = 0;
        s16PendingMax_ = 0;
        s16DeltaDMax_ = 0;
        s16DampedStiffMax_ = 0;
        s16WeightAbsDevMax_ = 0;
        s16WeightSites_ = 0;
        s16FoutMax_ = 0;
        s16MuMin_ = 1;
        s16MuMean_ = 1;
        s16MirrorDevMax_ = 0;
        s17TransferB1Max_ = 0;
        s17Transfers_ = 0;
        s17RotRatioMedian_ = 1;
        s17RotCells_ = 0;
        s17RotLdtMedian_ = 0;
        s17RotCandidates_ = 0;
        s17EigRelMax_ = 0;
        s17UniUp_ = 0;
        s17ProjCount_ = 0;
        s17ProjRaised_ = 0;
        s17ProjExcessMax_ = 0;
        s17ProjTies_ = 0;
        s17SplitDevMax_[0] = 0;
        s17SplitDevMax_[1] = 0;
        s17SplitDevMax_[2] = 0;
        s17SplitSites_ = 0;
        s17ResetCells_ = 0;
        s17PendingPressureMax_ = 0;
        s17bSlackEvents_ = 0;
        s17bSlackContent_ = 0;
        s17bSlackDdMax_ = 0;
        s17bSlackAtRest_ = 0;
        s17bSlackPreTMax_ = 0;
        s17bSlackPreTRatioMax_ = 0;
        s17bSlackPreQpMax_ = 0;
        s17bSlackPreQpRatioMax_ = 0;
        s17bSlackMuMax_ = 0;
        s17bSlackQpMax_ = 0;
        s17bMaxLfTransport_ = 0;
        s17bMaxLfRotation_ = 0;
        s17bMaxLfLaw_ = 0;
        s17bProjRaisedPresent_ = 0;
        s17bProjRaisedAbsent_ = 0;
        s18WallSplitDevMax_ = 0;
        if (projSiteLog_)
        {
            for (label k = 0; k < 2; ++k)
            {
                s18pCount_[k] = 0;
                s18pRaised_[k] = 0;
                s18pRaisedPresent_[k] = 0;
                s18pRaisedAbsent_[k] = 0;
                s18pExcessMax_[k] = 0;
            }
            s18pKernelDbg_.nCells = 0;
        }

        // Zero the per-phase clocks for this step. Done here and not at the
        // end of the previous step so that a run which terminates mid-step
        // still reports what it had spent.
        tPhaseField_ = 0.0;
        tPlatelets_ = 0.0;
        tSkeleton_ = 0.0;
        tDamage_ = 0.0;
        tPermeability_ = 0.0;
        tDerived_ = 0.0;
        tRelease_ = 0.0;
        tStepStart_ = phaseClotWallNow();

        updateDerivedFields();
        // Decide what is free BEFORE the phase field moves, so a cell
        // that lost its last bond this step is advected this step.
        updateReleasedIndicator();

        // THE EMBOLUS VELOCITY, FIXED FOR THE STEP, BEFORE ANYTHING USES IT.
        //
        // It has to be computed HERE rather than where it is first needed,
        // because it now has two consumers that must agree: solvePhaseField()
        // carries the released material at this vector, and the Brinkman drag
        // in momentumPredictor() resists flow RELATIVE to it. Computing it
        // twice would let them drift apart within a step -- the carrier flux
        // is built before the momentum solve and the momentum solve changes
        // U, so a second call would return a different vector and the body
        // would be dragged towards a velocity it is not travelling at.
        //
        // Evaluated off the START-OF-STEP U, which is exactly what the single
        // call inside solvePhaseField() used, so a run with the drag change
        // switched off reproduces the old behaviour bit for bit.
        //
        // GUARDED THE SAME WAY THE OLD CALL SITE WAS. solvePhaseField()
        // computed this inside `if (advectReleased_)`, so calling it
        // unconditionally here would evaluate an O(nCells) parallel
        // reduction, and print PHASE_CLOT_BODY_VELOCITY, on every step of
        // every series that has the release machinery switched off -- new
        // cost and new log lines in runs that are supposed to be unchanged.
        // With the switch off nothing is ever released, so the cached vector
        // stays Zero and the drag reduces to the lab-frame form regardless.
        //
        // releasedBodyVelocity() already returns Zero, and says so in the
        // log, when nothing is released or the ambient band is empty.
        // S10 FIX 2: not estimated at all under releasedCarrier "material".
        // Nothing reads it there -- the two-velocity drag is in the block
        // against the solved Vs -- and printing a rigid-body speed for a clot
        // that is not carried at it would be read as though it were.
        // The anchored layer is rebuilt from the SAME bond state the release
        // test just used, before anything is advected by it.
        updateBondHold();
        reportWoundState();

        // Vs_ IS NO LONGER COMPUTED HERE. It used to be evaluated
        // algebraically at this point, before solvePhaseField(), so that the
        // phase field, be and the drag all saw the same vector within a step.
        // It is now SOLVED in correctPressure(), which runs later in the same
        // step -- so within one step the carrier sees the PREVIOUS step's Vs_.
        //
        // That lag is deliberate and is the smaller of two errors. The
        // alternative is to advect the phase field with a velocity that has
        // not yet been made consistent with the pressure, which is the defect
        // this whole change exists to remove. One step of lag in the carrier
        // is a first-order transport error that the material Courant limit
        // already bounds; an inconsistent Vs_ is a violated momentum balance
        // and is bounded by nothing.

        beginPlateletOuterTransaction();
        solvePhaseField();
        syncMaterialLabelFromContent();
        updateDerivedFields();
        solvePlatelets();
        updateDerivedFields();


        {
            // Snapshot the displacement at the start of the step; the
            // skeleton velocity is measured against this. It must be taken
            // ONCE PER TIME STEP - refreshing it inside the exchange would
            // drive the skeleton velocity to zero and delete the very
            // coupling being iterated.
            UsPrev_ = Us_;
            // From here the pending increment accumulates over THIS step, so
            // this is the dt it will span. Recorded next to the assignment it
            // belongs to, because the two must never drift apart.
            deltaTIncrement_ = runTime.deltaTValue();
            HStep_ = H_;
            updatePermeability();

            // Reset the exchange state for this step.
            couplingConverged_ = false;
            mechanicsFinalised_ = false;
            couplingIters_ = 0;
            couplingResidual_ = 0.0;
            couplingResidual0_ = 0.0;
            fsiOmega_ = fsiRelaxation_;
            fsiResidual_ = dimensionedVector("zero", dimLength, Zero);

            {
                // ---- ONE VELOCITY: NEITHER COUPLING PATH APPLIES ----
                //
                // CHECKED FIRST, ABOVE THE stronglyCoupledFSI_ SPLIT, AND THAT
                // ORDERING IS THE WHOLE POINT. Both branches below exist to
                // reconcile two fields -- a displacement skeleton and a flow --
                // solved in turn. The Eulerian formulation has one velocity and
                // therefore no partition to reconcile, so it belongs to
                // neither, and making it a sub-case of either would tie it to a
                // switch that means something else.
                //
                // The first build of the stage-2 gate wired this into
                // solveMechanicsExchange() ONLY, which is reached from line
                // ~4109 and only when stronglyCoupledFSI_ is true. That switch
                // defaults to FALSE, so solveElasticStrain() was never called:
                // be would have stayed at I for the whole run, dev(be) = 0, the
                // elastic stress identically zero, and the clot would never
                // lock. The gate would have run to completion and failed on
                // "the clot did not lock" -- a true statement about a plumbing
                // defect, reported as though it were about the physics.
                //
                // Setting stronglyCoupledFSI yes instead would NOT have worked
                // either: the guard at line ~4025 aborts that combination with
                // nOuterCorrectors 1, which is what this gate runs.
                solveElasticStrain();

                // PUBLISH THE STRESS THE BOND MUST SEE, immediately after the
                // field it is built from is advanced and before anything can
                // evaluate a boundary condition against a stale one.
                //
                // Without this the Eulerian path leaves sigmaD_ at zero --
                // solveSkeleton(), its only writer, is never called here --
                // and clotWallAdhesion computes a traction of exactly zero on
                // every face for the whole run. See the declaration of
                // sigmaElastic_ for the run that measured it.
                sigmaElastic_ = elasticStress();

                // Diagnostic only; see the declaration. Before solveDamage() so
                // the two report the same time level.
                solvePassiveDisplacement();

                solveDamage();

                releaseDetachedMaterial();
                commitWallBonds();
                updateDerivedFields();
                updatePermeability();
                mechanicsFinalised_ = true;

                // No staggered loop and no Aitken sweep: be does not depend on
                // Dg within a step, so there is no fixed point here to iterate
                // toward. Recorded as one iteration that converged rather than
                // left at zero, so the coupling diagnostics do not read as if
                // the mechanics never ran.
                couplingIters_ = 1;
                couplingConverged_ = true;
            }
        }
    }

    // `&& !eulerianSolid_`: the Eulerian path has already run, above, outside
    // the coupling split. Without this exclusion a case that set BOTH switches
    // would advance be twice in one timestep -- silently, since two transport
    // solves in a row still produce a smooth, plausible field.
}


void Foam::solvers::phaseClotAxisymmetricFluid::pressureCorrector()
{
    // S19: one block solve replaces the PIMPLE correctors (nCorrectors is
    // logged and unused under blockCoupled).
    if (s19BlockCoupled())
    {
        solveBlockCoupledStep();
        tUEqn.clear();
        return;
    }
    incompressibleFluid::pressureCorrector();
}


void Foam::solvers::phaseClotAxisymmetricFluid::correctPressureTwoVelocity()
{
    // ================================================================
    //  THE COUPLED TWO-VELOCITY POROELASTIC PRESSURE CORRECTOR
    // ================================================================
    //
    // This is the whole point of the two-velocity path, and it is what the
    // algebraic closure it replaces did not have. Two properties, both
    // required, neither optional:
    //
    //  1. THE DRAG IS INVERTED, NOT EVALUATED. A per-cell 2x2 block couples
    //     U and Vs, so both momentum equations are satisfied SIMULTANEOUSLY
    //     by the inversion. There is no residual left over to dump into Vs.
    //
    //  2. THE PRESSURE ENFORCES MIXTURE CONTINUITY, div(phi_f*U + phi_s*Vs),
    //     and its equation is assembled from BOTH momentum equations. This
    //     is the defect that produced 88.3% of clot cells with |Vs| > |U| in
    //     run 20260904-214345: p was solved from the FLUID continuity alone,
    //     so nothing could force grad(p) consistent with div(sigma_e), and
    //     the mixture balance had no mechanism by which to hold.
    //
    // Du & Fogelson (source 47) Eq. 2.9, on exactly this point: "Anywhere
    // that theta_f and theta_b are both nonzero, neither individual velocity
    // field u_f and u_b need be incompressible, but this average velocity
    // field must be." Their p is "a Lagrange multiplier ... whose role is to
    // enforce the co-incompressibility condition". Cogan & Guy (48) and
    // Cogan & Keener (49) state the same independently, and OpenFOAM-13's
    // multiphaseEuler implements it by summing the pressure equation over
    // all phases (cellPressureCorrector.C:284,308).
    //
    // WHY NOT REUSE OpenFOAM'S momentumTransferSystem: it holds a
    // const phaseSystem& and is built on phaseInterfaceKey, blendedDragModel
    // and phaseModel. Adopting it means rewriting this solver as a
    // multiphaseEuler derivative. The ALGORITHM is ported here; the class is
    // not. For n = 2 the LU decomposition OpenFOAM needs for general n is
    // unnecessary -- the 2x2 inverse is closed form and exact.

    volScalarField& p(p_);
    volVectorField& U(U_);
    surfaceScalarField& phi(phi_);

    fvVectorMatrix& UEqn = tUEqn.ref();
    fvVectorMatrix& VsEqn = tVsEqn.ref();

    // SIMPLEC is refused rather than approximated, for the same reason the
    // single-velocity face-consistent path refuses it: pimple.consistent()
    // rebuilds rAtU from UEqn.H1() and adjusts both the flux and the cell
    // update, and carrying that through a 2x2 block correctly is separate
    // work. Guessing at it is the failure mode this project keeps paying for.
    if (pimple.consistent())
    {
        FatalErrorInFunction
            << "twoVelocitySkeleton is not implemented for pimple consistent "
            << "(SIMPLEC). Set consistent no, or extend this function "
            << "deliberately."
            << exit(FatalError);
    }

    // The SAME registered fields momentumPredictor() assembled the matrices
    // with. Recomputing them here would let the pressure equation and the
    // momentum equations disagree about the phase split by a step of q, and
    // the mixture balance would then hold for neither.
    const volScalarField& phiS = phiSkel_;
    const volScalarField& phiF = phiFluid_;
    const surfaceScalarField phiSf(fvc::interpolate(phiS));
    const surfaceScalarField phiFf(fvc::interpolate(phiF));

    // ---- THE 2x2 BLOCK ----
    //
    //   [ A_f + K        -K          ] [U ]   [ H_f - phi_f*grad(p)          ]
    //   [ -(K + Kres)   A_s + K+Kres ] [Vs] = [ H_s - phi_s*grad(p) + F_s    ]
    //
    // A_f and A_s carry NO drag -- momentumPredictor() deliberately leaves it
    // out of both matrices so it can be added here once.
    //
    // Kres is the vanishing-phase regularisation, on the SKELETON ROW ONLY so
    // the fluid is not perturbed. Where phi_s falls below residualPhiS the
    // skeleton momentum equation degenerates (there is no skeleton in the
    // lumen), and Kres drives Vs -> U smoothly instead. This is OpenFOAM's
    // residualAlpha (momentumTransferSystem.C:611) and Du & Fogelson's
    // "temporarily increasing theta_b by a small positive constant" -- two
    // independent codebases arriving at the same fix. Inside the clot it is
    // ~100 1/s against a drag of 3.3e+05 minimum, i.e. negligible.
    const volScalarField Kd("Kdrag", clotDrag_);

    // THE SUPPORT OF Kres IS ANCHOR-DEPENDENT UNLESS residualChi IS SET.
    //
    // The absolute form is a triangular ramp in phi_s: residualPhiS/deltaT at
    // phi_s = 0, falling linearly to ZERO at phi_s = residualPhiS, and zero
    // above. Because phi_s = chi*skeletonVolumeFraction, raising the anchor
    // slides the whole diffuse tail of c across that fixed threshold and
    // switches the regularisation OFF in the lumen. Run 20260906-101812:
    // every one of the top 100 cells by |Vs| had c > 0.99 and Kres = 0.
    //
    // With residualChi > 0 the ramp is measured on chi, which does not move
    // with the anchor, and is RESCALED so its peak is still
    // residualPhiS/deltaT. The region changes; the strength does not. That
    // keeps this a one-variable change against runs 20260906-052320 and
    // -101812.
    // The expression lives in skeletonResidualDrag() (S17 C: the Vs
    // predictor puts the same Kres on its diagonal), unchanged.
    const volScalarField Kres("Kres", skeletonResidualDrag());

    const volScalarField Ablk("Ablk", UEqn.A() + Kd);
    const volScalarField Bblk("Bblk", -Kd);
    const volScalarField Cblk("Cblk", -(Kd + Kres));
    const volScalarField Dblk("Dblk", VsEqn.A() + Kd + Kres);

    // det = A_f*A_s + A_f*(K+Kres) + K*A_s + K*Kres, a sum of positive terms.
    // It cannot vanish while A_f > 0, which fvm::ddt guarantees on a transient
    // run. The floor is a tripwire, not an expected path -- if it ever binds,
    // the momentum matrix has a non-positive diagonal and that is the bug.
    const volScalarField det("detBlk", Ablk*Dblk - Bblk*Cblk);
    const scalar detMin = gMin(det.primitiveField());
    if (detMin <= 0)
    {
        FatalErrorInFunction
            << "the 2x2 momentum block is singular or sign-indefinite: "
            << "min(det) = " << detMin << ". A_f and A_s must both be "
            << "positive. This is a defect in the assembled matrices, not "
            << "something to floor away."
            << exit(FatalError);
    }

    // Closed-form inverse of [[a,b],[c,d]] = (1/det)[[d,-b],[-c,a]].
    const volScalarField iAD00("iAD00", Dblk/det);
    const volScalarField iAD01("iAD01", -Bblk/det);
    const volScalarField iAD10("iAD10", -Cblk/det);
    const volScalarField iAD11("iAD11", Ablk/det);

    // Face counterparts. The block is applied at faces for the flux and at
    // cells for the cell velocity, and interpolate(iAD*H) != interpolate(iAD)*
    // flux(H). That asymmetry IS the Rhie-Chow construction and suppresses
    // checker-boarding; OpenFOAM keeps the same pair, invADVs alongside
    // invADVfs.
    surfaceScalarField iAD00f(fvc::interpolate(iAD00));
    surfaceScalarField iAD01f(fvc::interpolate(iAD01));
    surfaceScalarField iAD10f(fvc::interpolate(iAD10));
    surfaceScalarField iAD11f(fvc::interpolate(iAD11));

    // S11 FIX 2: ASSEMBLE THE BLOCK ON THE FACE, THEN INVERT IT THERE.
    //
    // The face control volume holds half of each cell's inertia and drag, so
    // its diagonal is the interpolated DIAGONAL, not the interpolated
    // inverse. multiphaseEuler's facePressureCorrector does exactly this
    // (Afs = fvc::interpolate(A), then invADVfs on faces). The difference is
    // negligible where both cells are clot and decisive at the clot surface:
    // a lumen cell's skeleton diagonal is only Kres ~ residualPhiS/dt, so the
    // interpolated INVERSE hands a clot-surface face ~half the lumen's
    // mobility, ~1e4x the clot's. The face inverse keeps the Rhie-Chow pair:
    // the cell velocity still uses the cell inverse against H, the face flux
    // the face inverse against flux(H).
    {
        const surfaceScalarField Af(fvc::interpolate(Ablk));
        const surfaceScalarField Bf(fvc::interpolate(Bblk));
        const surfaceScalarField Cf(fvc::interpolate(Cblk));
        const surfaceScalarField Df(fvc::interpolate(Dblk));
        const surfaceScalarField detf("detBlkf", Af*Df - Bf*Cf);

        // Positive by the same argument as the cell det, face by face:
        // interpolation with weights in [0, 1] keeps A_f, D_f > 0 and
        // |B_f| = K_f <= both. A non-positive value is a defect in the
        // assembled matrices, reported, not floored.
        const scalar detfMin = gMin(detf.primitiveField());
        if (detfMin <= 0)
        {
            FatalErrorInFunction
                << "the face-assembled 2x2 block is singular or "
                << "sign-indefinite: min(det_f) = " << detfMin
                << exit(FatalError);
        }

        iAD00f = Df/detf;
        iAD01f = -Bf/detf;
        iAD10f = -Cf/detf;
        iAD11f = Af/detf;
    }

    // ---- PREDICTED VELOCITIES AND PRESSURE COEFFICIENTS ----
    // H() is expensive and is needed four times; taken once so the cell and
    // face forms cannot be built from different evaluations.
    const volVectorField Hf("Hf", UEqn.H());
    const volVectorField Hs("Hs", VsEqn.H());

    volVectorField HbyADf(constrainHbyA(iAD00*Hf + iAD01*Hs, U, p));
    volVectorField HbyADs("HbyADs", iAD10*Hf + iAD11*Hs);

    // alphaByAD_i = sum_j invAD_ij * phi_j : the coefficient multiplying
    // -grad(p) in each phase velocity. Both rows mix both fractions, which is
    // how the skeleton's share of the pressure reaches the fluid and back.
    const volScalarField aByADf("aByADf", iAD00*phiF + iAD01*phiS);
    const volScalarField aByADs("aByADs", iAD10*phiF + iAD11*phiS);
    surfaceScalarField aByADff(fvc::interpolate(aByADf));
    surfaceScalarField aByADsf(fvc::interpolate(aByADs));

    // Under faceAssembled the pressure coefficients come from the SAME face
    // inverse as the fluxes. rAf below is built from them and the flux
    // correction subtracts aByAD*f*snGp: one inverse on both sides, or the
    // mixture flux would not be the one the pressure equation made
    // divergence-free.
    {
        aByADff = iAD00f*phiFf + iAD01f*phiSf;
        aByADsf = iAD10f*phiFf + iAD11f*phiSf;
    }

    // ---- THE ELASTIC FORCE, IN FACE FORM, THROUGH THE BLOCK ----
    //
    // Face form for the same reason the single-velocity path uses it: so the
    // pressure Laplacian can cancel it face by face the way it cancels its own
    // snGrad. It is also the Rhie-Chow-like cure Ghione (source 51) records
    // for forces proportional to a phase-fraction gradient, which left
    // explicit in the momentum equation cause momentum/continuity decoupling
    // and checker-boarding.
    //
    // THE CROSS TERM iAD01 IS THE PHYSICS THE OLD CLOSURE OMITTED. The
    // skeleton's elastic force pushes the FLUID too, through the drag that
    // couples them. That channel exists in the block and existed nowhere in
    // the algebraic version.
    //
    // S11: built by skeletonElasticForceFlux(), which returns exactly the
    // expression that stood here when stressFaceTransmission is linear and
    // networkPressureForce is stressDivergence, and otherwise the series face
    // stress plus the compact phase-pressure force. One construction, shared
    // with every diagnostic through skeletonElasticForceCell().
    const surfaceScalarField fSigmaE(skeletonElasticForceFlux());
    const surfaceScalarField FByADf("FByADf", iAD01f*fSigmaE);
    const surfaceScalarField FByADs("FByADs", iAD11f*fSigmaE);

    // WHERE THE ELASTIC FORCE ACTUALLY LANDS, AND AGAINST WHAT DIAGONAL.
    //
    // Run 20260906-052320 killed both children with maxVsInClot = 0 on EVERY
    // sample: the skeleton velocity ran away OUTSIDE the clot while every
    // clot-cell metric stayed ordinary. A mechanism was inferred for that and
    // never measured, so this block measures it rather than assuming it.
    //
    // In the lumen phiSkel_ = 0, so VsEqn.A() = 0 and Kd = 0, and the skeleton
    // row of the 2x2 block reduces to Kres*(Vs - U) = Hs + F_s with
    // Kres = residualPhiS/deltaT = 68 1/s -- against 3.3e+05 of drag inside
    // the clot. Any force that survives out there is divided by a diagonal
    // some 5000x too small.
    //
    // elasticStress() carries TWO DIFFERENT WEIGHTS and only one of them
    // vanishes in the lumen: the DEVIATORIC part is weighted by
    // max(chi, skeletonSolidFloor), which FLOORS AT 1e-4, while the isotropic
    // network-pressure part is weighted by solidFraction(), which is exactly
    // zero there. So the lumen still carries 1e-4*gD*G*dev(be).
    //
    // vsElas* is the quantity that settles it: the elastic force's OWN
    // contribution to Vs, in m/s, reconstructed exactly the way Vs_ is built
    // below. If vsElasBlood dominates vsElasBulk the inferred loop is real. If
    // it does not, the inference is WRONG and the cause is elsewhere -- which
    // is the point of measuring instead of patching.
    //
    // Bins are c < -0.9 clot bulk, c < 0.5 interface, else blood: THE SAME
    // three PHASE_CLOT_BE_PHASE uses, so stretchBlood and vsElasBlood can be
    // read on one line. NOTE det is shadowed by the local detBlk field above,
    // so nothing here may call Foam::det.
    {
        const volVectorField vsElas("vsElas", fvc::reconstruct(FByADs));

        // As is recovered from Dblk rather than re-evaluating VsEqn.A(), so
        // the reported split cannot disagree with the matrix actually solved.
        const volScalarField As("As", Dblk - Kd - Kres);

        // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
        // so every threshold below is the same threshold restated:
        // c < L becomes ind > (1 - L)/2, and the inequality turns
        // over with it. Same cells, named in the surviving field.
        const volScalarField indHold(phaseIndicator());
        const scalarField& indIn = indHold.primitiveField();
        const scalarField magVsElas(mag(vsElas.primitiveField()));
        const scalarField& DblkIn = Dblk.primitiveField();
        const scalarField& AsIn = As.primitiveField();
        const scalarField& KdIn = Kd.primitiveField();
        const scalarField& KresIn = Kres.primitiveField();
        const scalarField& phiSIn = phiS.primitiveField();
        const symmTensorField& beIn = be_.primitiveField();

        scalar vBulk = 0, vInt = 0, vBlood = 0;
        scalar dBulk = great, dInt = great, dBlood = great;
        scalar devBulk = 0, devBlood = 0;
        scalar AsBlood = 0, KdBlood = 0, KresBlood = 0;
        scalar phiSBlood = 0, wElasBlood = 0;

        forAll(indIn, celli)
        {
            const scalar v = magVsElas[celli];
            const scalar D = DblkIn[celli];
            const scalar magDevBe = mag(dev(beIn[celli]));

            if (indIn[celli] > 0.95)
            {
                vBulk = max(vBulk, v);
                dBulk = min(dBulk, D);
                devBulk = max(devBulk, magDevBe);
            }
            else if (indIn[celli] > 0.25)
            {
                vInt = max(vInt, v);
                dInt = min(dInt, D);
            }
            else
            {
                vBlood = max(vBlood, v);
                dBlood = min(dBlood, D);
                devBlood = max(devBlood, magDevBe);
                AsBlood = max(AsBlood, AsIn[celli]);
                KdBlood = max(KdBlood, KdIn[celli]);
                KresBlood = max(KresBlood, KresIn[celli]);
                phiSBlood = max(phiSBlood, phiSIn[celli]);

                // The elastic weight as elasticStress() applies it:
                // max(chi, skeletonSolidFloor) with chi = max(0.5*(1-c), 0).
                wElasBlood = max
                (
                    wElasBlood,
                    max(max(0.5*(1.0 - indIn[celli]), 0.0), skeletonSolidFloor_)
                );
            }
        }

        reduce(vBulk, maxOp<scalar>());
        reduce(vInt, maxOp<scalar>());
        reduce(vBlood, maxOp<scalar>());
        reduce(dBulk, minOp<scalar>());
        reduce(dInt, minOp<scalar>());
        reduce(dBlood, minOp<scalar>());
        reduce(devBulk, maxOp<scalar>());
        reduce(devBlood, maxOp<scalar>());
        reduce(AsBlood, maxOp<scalar>());
        reduce(KdBlood, maxOp<scalar>());
        reduce(KresBlood, maxOp<scalar>());
        reduce(phiSBlood, maxOp<scalar>());
        reduce(wElasBlood, maxOp<scalar>());

        // THE ARGMAX CELL, WITH ALL OF ITS PROPERTIES ON ONE LINE.
        //
        // The per-bin numbers above reduce Dblk with min and vsElas with max,
        // so THEY LAND IN DIFFERENT CELLS and cannot be read as a description
        // of one place. Run 20260906-101812 was misread exactly that way:
        // DblkBlood = 68 was reported as the diagonal of the runaway cell,
        // when the runaway cells in fact had Kres = 0 and a diagonal of order
        // 0.01-0.1 1/s. Recovering that took a hand parse of the written
        // fields. This line is that parse, done in the solver.
        //
        // SERIAL-CORRECT ONLY: the argmax is rank-local, so on more than one
        // process this reports the worst cell of rank 0's subdomain, not the
        // domain. Every run of this case is CONCURRENCY=1; if that changes,
        // this needs a gathered argmax before it can be trusted.
        {
            label wi = -1;
            scalar wv = -1;
            forAll(indIn, celli)
            {
                if (magVsElas[celli] > wv)
                {
                    wv = magVsElas[celli];
                    wi = celli;
                }
            }

            if (wi >= 0)
            {
                const scalar chiW = max(0.5*(1.0 - indIn[wi]), 0.0);

                Info<< "PHASE_CLOT_LUMEN_WORST"
                    << " time=" << runTime.name()
                    << " vsElas=" << wv
                    // ind, not c -- see the note on indWorst above.
                    << " ind=" << indIn[wi]
                    << " chi=" << chiW
                    << " phiS=" << phiSIn[wi]
                    << " Dblk=" << DblkIn[wi]
                    << " As=" << AsIn[wi]
                    << " Kd=" << KdIn[wi]
                    << " Kres=" << KresIn[wi]
                    << " devBe=" << mag(dev(beIn[wi]))
                    << " residualChi=" << residualChi_
                    << " residualPhiS=" << residualPhiS_
                    << nl;
            }
        }

        Info<< "PHASE_CLOT_LUMEN_FORCE"
            << " time=" << runTime.name()
            << " vsElasBulk=" << vBulk
            << " vsElasInterface=" << vInt
            << " vsElasBlood=" << vBlood
            << " DblkBulk=" << dBulk
            << " DblkInterface=" << dInt
            << " DblkBlood=" << dBlood
            << " AsBlood=" << AsBlood
            << " KdBlood=" << KdBlood
            << " KresBlood=" << KresBlood
            << " phiSBlood=" << phiSBlood
            << " wElasBlood=" << wElasBlood
            << " devBeBulk=" << devBulk
            << " devBeBlood=" << devBlood
            << " solidFloor=" << skeletonSolidFloor_
            << nl;
    }

    // ---- THE MIXTURE FLUX AND THE SHARED PRESSURE ----
    //
    // ddtCorr enters HERE, inside the block, on the fluid row only -- it is
    // the fluid's transient flux correction and the skeleton has no Uf. This
    // is where multiphaseEuler puts its own ddtCorrs too: added into phiHs
    // BEFORE the invADVfs multiply, not bolted on after it.
    // ddtCorr takes the FLUID'S OWN flux, not phi. phi_ is the mixture flux
    // now, and ddtCorr(U, phi, Uf) is a correction built from the mismatch
    // between the cell velocity and its own face flux -- handing it a flux
    // that is part skeleton would make it correct for a velocity that does
    // not exist.
    const surfaceScalarField phiHf
    (
        "phiHf", fvc::flux(Hf) + fvc::ddtCorr(U, phiFluidOwn_, Uf)
    );
    const surfaceScalarField phiHs
    (
        "phiHs", fvc::flux(Hs) + fSigmaE
    );

    // Both sums run over BOTH phases. These two lines are what make the
    // pressure equation contain the solid momentum equation.
    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        phiFf*(iAD00f*phiHf + iAD01f*phiHs)
      + phiSf*(iAD10f*phiHf + iAD11f*phiHs)
    );

    // S10 FIX 4: THE TWO-PHASE constrainHbyA. On every patch where U is
    // prescribed, the predicted mixture flux is the wall's own flux.
    //
    // phiHf and phiHs above are fvc::flux of the RAW H fields, whose wall
    // values are extrapolated from the wall cells. Single-phase OpenFOAM never
    // sees this, because it fluxes a HbyA that constrainHbyA has already
    // overwritten with U_b wherever U is not assignable. Here constrainHbyA
    // reached HbyADf (the cell field U is rebuilt from) and never the flux.
    // With zeroGradient p on the walls the pressure equation adds nothing at a
    // wall face, so whatever the prediction carried through the wall stayed in
    // phi.
    //
    // MEASURED on run 20260922-104613 (em49): after release the no-slip
    // bottom wall carried -1.42e-14 m^3/s, 3.4% of the throughput. It was
    // identical to PHASE_CLOT_DIAGNOSTIC outerFlux, and face by face it
    // tracked the wall cells' own normal velocities (r = 0.98 against
    // theta*Vs_y*|Sf|). The p maximum, 868 Pa against a 351 Pa inlet, sat on
    // the leaking cell. em48, never released, held it at ~1e-16 because its
    // wall cells had no normal velocity to leak.
    //
    // BOTH PHASES TAKE THE WALL'S NORMAL VELOCITY. A no-slip fluid and a
    // skeleton that cannot penetrate the wall share U_b.n, and phiFf + phiSf
    // is 1, so the mixture flux is exactly U_b & Sf -- zero on these static
    // walls. The same criterion as constrainHbyA (assignable()), so an
    // outlet or a pressure-driven inlet is left to the pressure equation.
    {
        surfaceScalarField::Boundary& phiHbyAbf = phiHbyA.boundaryFieldRef();
        forAll(U.boundaryField(), patchi)
        {
            if (!U.boundaryField()[patchi].assignable())
            {
                phiHbyAbf[patchi] =
                    U.boundaryField()[patchi]
                  & mesh.Sf().boundaryField()[patchi];
            }
        }
    }

    MRF.makeRelative(phiHbyA);

    if (p.needReference())
    {
        fvc::makeRelative(phiHbyA, U);
        adjustPhi(phiHbyA, U, p);
        fvc::makeAbsolute(phiHbyA, U);
    }

    // Pressure "diffusivity": sum_i phi_i * alphaByAD_i, again over BOTH.
    const surfaceScalarField rAf("rAf", phiFf*aByADff + phiSf*aByADsf);

    // The cell-form counterpart of rAf: fixedFluxPressure turns a flux
    // mismatch into a pressure gradient, and the coefficient it must use is
    // the MIXTURE one, not the fluid's share of it.
    const volScalarField rAmix("rAmix", phiF*aByADf + phiS*aByADs);
    constrainPressure(p, U, phiHbyA, rAmix, MRF);

    fvScalarMatrix pEqnSource(fvModels().sourceProxy(p));

    tmp<surfaceScalarField> tpEqnFlux;

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvm::laplacian(rAf, p)
         ==
            fvc::div(phiHbyA)
          - pEqnSource
        );

        pEqn.setReference
        (
            pressureReference.refCell(),
            pressureReference.refValue()
        );

        pEqn.solve();

        if (pimple.finalNonOrthogonalIter())
        {
            tpEqnFlux = pEqn.flux();

            // THE MIXTURE FLUX. phi_ carries phi_f*U + phi_s*Vs from here on,
            // and it is this quantity -- not the fluid flux -- that is
            // divergence-free. Everything downstream that advects with phi_ is
            // advecting with the mixture, which is correct: the phase field
            // rides the material, and the material is the mixture.
            phi = phiHbyA - tpEqnFlux();

            // EACH PHASE'S OWN FLUX, from the same block and the same pressure
            // gradient. These are what the next momentum assembly convects
            // with. Their phi-weighted sum reproduces phi above identically,
            // because rAf is by construction phiFf*aByADff + phiSf*aByADsf --
            // so the two cannot drift apart, which is the property that makes
            // storing them safe rather than a second source of truth.
            // THE PRESSURE FACE TERM COMES FROM THE MATRIX FLUX, NOT FROM
            // snGrad(p).
            //
            // We need X such that phiFf*phiFluidOwn + phiSf*phiSkelOwn
            // reproduces phi EXACTLY. Working it through:
            //
            //   phiFf*phiFluidOwn + phiSf*phiSkelOwn
            //     = phiHbyA - (phiFf*aByADff + phiSf*aByADsf)*X
            //     = phiHbyA - rAf*X
            //
            // and phi = phiHbyA - pEqnFlux, so X must be pEqnFlux/rAf.
            //
            // fvc::snGrad(p)*magSf is NOT that. pEqn.flux() carries the
            // non-orthogonal correction and a bare snGrad does not, so on a
            // skewed mesh the two differ and the phase fluxes would no longer
            // sum to the flux the pressure equation actually imposed.
            //
            // This is not hypothetical here. em_2vel is a THROAT child, and
            // apply_throat bends the mesh to 33.9 degrees of cell skew at the
            // taper midpoint -- which is why that geometry ships
            // nNonOrthogonalCorrectors 2 rather than 0. The error would have
            // sat exactly where the clot lodges.
            //
            // rAf is strictly positive: phiFf > 0 and aByADff > 0, both by
            // construction from a positive-definite block.
            const surfaceScalarField snGp("snGp", tpEqnFlux()/rAf);
            phiFluidOwn_ =
                iAD00f*phiHf + iAD01f*phiHs - aByADff*snGp;
            phiSkelOwn_ =
                iAD10f*phiHf + iAD11f*phiHs - aByADsf*snGp;

            // S10 FIX 4, CONTINUED: each phase's own flux takes the wall flux
            // on the same patches as phiHbyA above. Without it phi would be
            // U_b & Sf there while the phase fluxes still carried the
            // extrapolated prediction, and the decomposition identity checked
            // just below would break on every wall face. With it the identity
            // holds on walls as well: (phiFf + phiSf)*(U_b & Sf) = phi_b.
            forAll(U.boundaryField(), patchi)
            {
                if (!U.boundaryField()[patchi].assignable())
                {
                    const scalarField wallFlux
                    (
                        U.boundaryField()[patchi]
                      & mesh.Sf().boundaryField()[patchi]
                    );
                    phiFluidOwn_.boundaryFieldRef()[patchi] = wallFlux;
                    phiSkelOwn_.boundaryFieldRef()[patchi] = wallFlux;
                }
            }

            // ---- G4: THE DECOMPOSITION IDENTITY, MEASURED NOT ASSUMED ----
            //
            // phi == phiFf*phiFluidOwn_ + phiSf*phiSkelOwn_ is exact by the
            // construction of rAf above, for the reason the comment there
            // works through. But "exact by construction" is a property of the
            // code as written, not a measurement, and it is the FIRST thing a
            // change to the phase weights breaks -- silently, because every
            // consumer downstream reads the phase fluxes individually and
            // nothing ever adds them back up.
            //
            // MEASURED HERE rather than archived for an offline check. The
            // weights are rebuilt in momentumPredictor() on every outer
            // iteration, so a later reader cannot know it holds the same
            // phiFf/phiSf these fluxes were built from -- that is the stale-q_
            // class of defect this project has paid for twice. At this point
            // all four arrays are in scope at one unambiguous time level.
            //
            // BEFORE the wall zeroing below, which deliberately breaks the
            // identity on wall faces. The boundary figure is therefore
            // reported SEPARATELY, so that intended break stays visible
            // instead of being averaged into the internal number.
            {
                const surfaceScalarField decomposed
                (
                    phiFf*phiFluidOwn_ + phiSf*phiSkelOwn_
                );
                const surfaceScalarField residual(mag(phi - decomposed));

                const scalar scale =
                    max(gMax(mag(phi.primitiveField())), vSmall);
                const scalar rInternal = gMax(residual.primitiveField());

                scalar rBoundary = 0;
                forAll(residual.boundaryField(), patchi)
                {
                    const fvsPatchScalarField& rP =
                        residual.boundaryField()[patchi];

                    // Coupled patches are interior to the decomposed domain,
                    // not boundary of the physical one; counting them would
                    // make this number depend on the decomposition.
                    if (rP.coupled() || !rP.size())
                    {
                        continue;
                    }

                    rBoundary = max(rBoundary, Foam::max(rP));
                }
                reduce(rBoundary, maxOp<scalar>());

                // S10 FIX 4 GATE: the largest flux left on a face whose U is
                // prescribed, and the net volume through those patches. Both
                // are exactly 0 on static walls once the constraint holds.
                // Run 20260922-104613 would have printed ~1.4e-14 for the
                // net; the absolute flux on one face there reached 2.3e-15.
                scalar prescribedFluxMax = 0;
                scalar prescribedFluxNet = 0;
                forAll(phi.boundaryField(), patchi)
                {
                    if
                    (
                        U.boundaryField()[patchi].assignable()
                     || !phi.boundaryField()[patchi].size()
                    )
                    {
                        continue;
                    }
                    const scalarField dev
                    (
                        phi.boundaryField()[patchi]
                      - (
                            U.boundaryField()[patchi]
                          & mesh.Sf().boundaryField()[patchi]
                        )
                    );
                    prescribedFluxMax =
                        max(prescribedFluxMax, Foam::max(mag(dev)));
                    prescribedFluxNet += sum(phi.boundaryField()[patchi]);
                }
                reduce(prescribedFluxMax, maxOp<scalar>());
                reduce(prescribedFluxNet, sumOp<scalar>());

                Info<< "PHASE_CLOT_FLUX_DECOMPOSITION"
                    << " time=" << runTime.name()
                    << " maxInternal=" << rInternal
                    << " maxBoundary=" << rBoundary
                    << " scale=" << scale
                    << " relInternal=" << rInternal/scale
                    << " relBoundary=" << rBoundary/scale
                    << " prescribedFluxMax=" << prescribedFluxMax
                    << " prescribedFluxNet=" << prescribedFluxNet
                    << endl;
            }

            // NO SKELETON FLUX THROUGH A WALL. Same reason skeletonFlux()
            // zeroes it: the throat's top wall slopes by up to 25.2 degrees,
            // and a skeleton flux with a normal component there would push
            // material into a boundary it cannot enter.
            forAll(phiSkelOwn_.boundaryFieldRef(), patchi)
            {
                if (isA<wallFvPatch>(mesh.boundary()[patchi]))
                {
                    phiSkelOwn_.boundaryFieldRef()[patchi] = 0.0;
                }
            }
        }
    }

    if (!tpEqnFlux.valid())
    {
        FatalErrorInFunction
            << "the non-orthogonal corrector loop produced no final flux"
            << exit(FatalError);
    }

    continuityErrors();

    p.relax();

    // ---- BOTH VELOCITIES, FROM THE SAME FACE FIELDS AS THE FLUX ----
    //
    // reconstruct() of the identical face fields that produced phi above, so
    // the cell velocities and the flux cannot disagree. The base's
    // U = HbyA - rAtU*grad(p) uses a CELL gradient against a face snGrad,
    // which for pressure alone is the Rhie-Chow damping and is wanted -- but
    // the elastic force here is many times the applied load, and the same
    // discrepancy on it is a real unbalanced force.
    // THE SAME X the phase fluxes used, for the same reason: the cell
    // velocities and the flux must be reconstructed from identical face
    // fields or they describe different flows. This mirrors what the
    // single-velocity path already does with tpEqnFlux()/rAUf.
    //
    // Formed BEFORE p.relax() above? No -- tpEqnFlux was, and that is the
    // point: the cell update follows the flux rather than the relaxed field,
    // which is the same precedence interFoam uses and the same one the
    // single-velocity branch documents.
    const surfaceScalarField snGradpSf("snGradpSf", tpEqnFlux()/rAf);

    // S17b A3 (design 2A3; H-NA14; 12 change 1): the WALL-PATCH face term of
    // both reconstructs, kept separately. Its difference (Vs's minus U's) is
    // the slip the wall face adds; logged at the G8 cell as a projection on
    // the slip (em74's sign rule), whether or not A3 removes it.
    surfaceScalarField reconU("reconU", FByADf - aByADff*snGradpSf);
    surfaceScalarField reconS("reconS", FByADs - aByADsf*snGradpSf);
    tmp<volVectorField> tWallSlipTerm;
    {
        surfaceScalarField wallU("wallU", 0.0*reconU);
        surfaceScalarField wallS("wallS", 0.0*reconS);
        forAll(wallU.boundaryField(), patchi)
        {
            if (isA<wallFvPatch>(mesh.boundary()[patchi]))
            {
                wallU.boundaryFieldRef()[patchi] = reconU.boundaryField()[patchi];
                wallS.boundaryFieldRef()[patchi] = reconS.boundaryField()[patchi];
            }
        }
        tWallSlipTerm = fvc::reconstruct(wallS) - fvc::reconstruct(wallU);
    }
    {
        // THE SAME LOOP AND VALUE as phiSkelOwn_'s wall zeroing above: the
        // reconstructs then read the face fields the fluxes took, on both
        // phases, so the cell velocities and the fluxes agree on walls too.
        forAll(reconU.boundaryField(), patchi)
        {
            if (isA<wallFvPatch>(mesh.boundary()[patchi]))
            {
                reconU.boundaryFieldRef()[patchi] = 0.0;
                reconS.boundaryFieldRef()[patchi] = 0.0;
            }
        }
        U = HbyADf + fvc::reconstruct(reconU);
    }
    U.correctBoundaryConditions();
    fvConstraints().constrain(U);

    {
        Vs_ = HbyADs + fvc::reconstruct(reconS);
    }
    Vs_.correctBoundaryConditions();
    fvConstraints().constrain(Vs_);

    // WHERE THERE IS NO SKELETON THERE IS NO SKELETON VELOCITY. Kres already
    // drives Vs -> U in the lumen through the block, so this is belt and
    // braces against a cell where phi_s underflowed before Kres could act. It
    // is a clamp on a diagnostic-visible field, not a physics term: inside the
    // clot isSkel is 1 and this line is the identity.
    // ON THE SAME BASIS AS Kres, or the two safety nets miss the same cells.
    // Run 20260906-101812's runaway cells sat at phi_s = 1.7e-04, above BOTH
    // the Kres threshold (1e-4) and this clamp's 0.5e-4, so neither fired.
    const volScalarField isSkel
    (
        residualChi_ > 0
      ? pos0
        (
            phaseIndicator()
          - dimensionedScalar("halfChi", dimless, 0.5*residualChi_)
        )()
      : pos0
        (
            phiS - dimensionedScalar("half", dimless, 0.5*residualPhiS_)
        )()
    );
    Vs_ = isSkel*Vs_ + (1.0 - isSkel)*U;

    // ONE LINE PER STEP, NOT ONE PER PISO CORRECTOR. correctPressure() is
    // called nCorrectors times (2 on this case), and the old algebraic
    // closure printed this once per step from prePredictor(). Doubling the
    // line would double the log for no information: the assess script reads
    // the LAST record of a step, which is this one. pimpleNoLoopControl
    // inherits pisoControl, so finalPisoIter() is available here.
    if (pimple.finalPisoIter())
    {
        reportSkeletonVelocity();
        reportFringe();

        // ---- S11 DIAGNOSTICS: THE MEASUREMENTS RUN 20260923-031942 LACKED --
        //
        // Every number here was either reconstructed offline from fields
        // written 53 steps apart or could not be reconstructed at all. None
        // feeds back.
        const volScalarField indS(phaseIndicator());
        const scalarField& indI = indS.primitiveField();
        const scalar residualContent =
            residualChi_*skeletonVolumeFraction_;
        const scalarField& refI = thetaNetRef_.primitiveField();

        // (1) THE WORST DILUTE CELL'S VELOCITY, SPLIT BY TERM. Among cells
        // holding network but less than half of it (voids, fringe), the one
        // whose skeleton outruns its fluid the most, and what put it there:
        // the momentum/drag part HbyADs, the elastic face force, the
        // pressure face force -- against the diagonal each is divided by.
        {
            const volVectorField vsF("vsF", fvc::reconstruct(FByADs));
            const volVectorField vsP
            (
                "vsP", fvc::reconstruct(-aByADsf*snGradpSf)
            );
            const volScalarField ddtPhiS("ddtPhiS", fvc::ddt(phiSkel_));
            const vectorField& vsI = Vs_.primitiveField();
            const vectorField& uI = U.primitiveField();
            label w = -1;
            scalar worstSlip = -1;
            forAll(indI, celli)
            {
                const bool network =
                  refI[celli] > residualContent;
                if (network && indI[celli] < 0.5)
                {
                    const scalar s = mag(vsI[celli] - uI[celli]);
                    if (s > worstSlip) { worstSlip = s; w = celli; }
                }
            }
            if (w >= 0)
            {
                Info<< "PHASE_CLOT_VS_SPLIT time=" << runTime.userTimeValue()
                    << " x=" << mesh.C()[w].x() << " y=" << mesh.C()[w].y()
                    << " ind=" << indI[w]
                    << " thetaNetRef=" << refI[w]
                    << " slip=" << worstSlip
                    << " Vs=" << mag(vsI[w]) << " U=" << mag(uI[w])
                    << " HbyADs=" << mag(HbyADs[w])
                    << " elasticFace=" << mag(vsF[w])
                    << " pressureFace=" << mag(vsP[w])
                    << " Dblk=" << Dblk[w]
                    << " As=" << Dblk[w] - Kd[w] - Kres[w]
                    << " Kd=" << Kd[w] << " Kres=" << Kres[w]
                    << " ddtPhiS=" << ddtPhiS[w]
                    << " momentumForm=" << "phaseContinuity"
                    << endl;

                // S16 (red-team Change 3): the state of THIS cell -- G8's
                // slip argmax -- so an exceedance can be classified: d* = 1
                // is A1's release; d* < 1 with lambda1 < lc is the inherited
                // intact tail; a large transport |dd| (PHASE_CLOT_S16) is a
                // transported step. w is the deviatoric weight as the stress
                // uses it under skeletonFloorBasis.
                {
                    const scalar dW = ruptured_[w];
                    scalar dStarW = dW;
                    {
                        const scalar sW =
                            thetaIntact_[w]
                          + (thetaPending_[w] + thetaBroken_[w]);
                        dStarW =
                            sW > 0
                          ? (thetaPending_[w] + thetaBroken_[w])/sW
                          : 0.0;
                    }
                    const scalar l1W =
                        sqrt(cmptMax(eigenValues(be_[w])));
                    const scalar indW = indI[w];
                    const scalar wUsed =
                      max(indW, skeletonSolidFloor_)*(1.0 - dW);
                    const bool floorBinds =
                          indW < skeletonSolidFloor_;
                    Info<< "PHASE_CLOT_S16_G8CELL time="
                        << runTime.userTimeValue()
                        << " x=" << mesh.C()[w].x()
                        << " y=" << mesh.C()[w].y()
                        << " slip=" << worstSlip
                        << " d=" << dW
                        << " dStar=" << dStarW
                        << " lambda1=" << l1W
                        << " ind=" << indW
                        << " w=" << wUsed
                        << " floorBinds=" << (floorBinds ? 1 : 0)
                        << endl;
                }
                // S17b (06; 12 changes 5 and 7): the G8 cell's content,
                // its four force parts along the slip, the wall-face term's
                // projection on the slip (positive: the wall face AIDED the
                // slip, so A3 lowers it), the age of its last slack event,
                // and the slack CONTENT-SHARE PROXY (theta_P - theta_T)/
                // (theta_I + theta_P): the slack fraction of the load-path
                // content, a proxy for V1's tau lag, not the imported face
                // traction (design section 8 g).
                {
                    const vector slipDir(vsI[w] - uI[w]);
                    scalar parts[4];
                    s17bForceParts(iAD11f, w, slipDir, parts);
                    const scalar refW = refI[w];
                    const scalar thW = thetaC_[w] + thetaH_[w];
                    const scalar wallProj =
                        tWallSlipTerm.valid()
                      ? (tWallSlipTerm()[w] & (slipDir/max(mag(slipDir), vSmall)))
                      : 0.0;
                    const scalar p12 = thetaIntact_[w] + thetaPending_[w];
                    const scalar slackShare =
                        p12 > 0
                      ? (thetaPending_[w] - thetaTaut_[w])/p12
                      : 0.0;
                    const scalar tSlack = s17bLastSlackTime_[w];
                    const bool wallRow = [&]()
                    {
                        const labelUList& fc = mesh.cells()[w];
                        forAll(fc, i)
                        {
                            const label f = fc[i];
                            if (!mesh.isInternalFace(f))
                            {
                                const label pi = mesh.boundaryMesh().whichPatch(f);
                                if (isA<wallFvPatch>(mesh.boundary()[pi]))
                                {
                                    return true;
                                }
                            }
                        }
                        return false;
                    }();
                    Info<< "PHASE_CLOT_S17B_G8CELL time="
                        << runTime.userTimeValue()
                        << " x=" << mesh.C()[w].x()
                        << " y=" << mesh.C()[w].y()
                        << " slip=" << worstSlip
                        << " thetaByRef=" << (refW > 0 ? thW/refW : -1.0)
                        // S18 G: the deposited occupancy thetaNetRef/phi0
                        // (amendment c's theta/ref test is replaced by it)
                        << " occupancy=" << refW/skeletonVolumeFraction_
                        << " VsInd=" << mag(vsI[w])*indI[w]
                        << " devIntact=" << parts[0]
                        << " devPending=" << parts[1]
                        << " pressIntact=" << parts[2]
                        << " pressFrozen=" << parts[3]
                        << " wallRow=" << (wallRow ? 1 : 0)
                        << " wallFaceProj=" << wallProj
                        << " slackAge=" << (tSlack >= 0 ? runTime.value() - tSlack : -1.0)
                        << " slackContentShare=" << slackShare
                        << endl;
                }
            }
            else
            {
                Info<< "PHASE_CLOT_VS_SPLIT time=" << runTime.userTimeValue()
                    << " noDiluteNetworkCell" << endl;
                {
                    Info<< "PHASE_CLOT_S16_G8CELL time="
                        << runTime.userTimeValue()
                        << " noDiluteNetworkCell" << endl;
                }
            }
        }

        // (2) THE FASTEST SKELETON FACE. The clot-surface failure of the
        // cell-interpolated inverse is a FACE speed, invisible in any cell
        // field; this is where it would show.
        {
            const scalarField& fS = phiSkelOwn_.primitiveField();
            const scalarField& mSf = mesh.magSf().primitiveField();
            label fw = -1;
            scalar fMax = 0;
            forAll(fS, facei)
            {
                const scalar v = mag(fS[facei])/mSf[facei];
                if (v > fMax) { fMax = v; fw = facei; }
            }
            Info<< "PHASE_CLOT_FACE_SKELETON time="
                << runTime.userTimeValue()
                << " maxFaceVs=" << fMax
                << " x=" << (fw >= 0 ? mesh.Cf()[fw].x() : 0)
                << " y=" << (fw >= 0 ? mesh.Cf()[fw].y() : 0)
                << " maxU=" << gMax(mag(U.primitiveField()))
                << " blockFaceInverse=" << "faceAssembled" << endl;
        }

        // (3) COMPACTION STRIPES, VOIDS, THE FRINGE, AND THE NETWORK
        // PRESSURE'S OWN STABILITY NUMBER. Only where the deposited reference
        // defines them.
        {
            const volScalarField argF(networkPressureArgument());
            const volScalarField hp
            (
                "argHighPass", argF - fvc::average(fvc::interpolate(argF))
            );
            const scalarField& aI = argF.primitiveField();
            const scalarField& hI = hp.primitiveField();
            const vectorField& vsI = Vs_.primitiveField();
            const scalar phi0 = skeletonVolumeFraction_;

            // The fringe: does the clot's diffuse edge keep elastic memory
            // (be max eigenvalue above 1), and does it stay smooth (the
            // content's own high-pass)? em52's fringe had be = I to 1.001
            // and broke into grid-scale rows.
            const volScalarField thetaAll("thetaAll", thetaC_ + thetaH_);
            const volScalarField thp
            (
                "thetaHighPass",
                thetaAll - fvc::average(fvc::interpolate(thetaAll))
            );
            const scalarField& tI = thp.primitiveField();
            const symmTensorField& bI = be_.primitiveField();
            scalar fLam = 0, fT2 = 0;

            scalar s2 = 0, hMax = 0, fSum = 0, bSum = 0;
            label nIn = 0, nVoid = 0, nF = 0, nB = 0;
            forAll(aI, celli)
            {
                if (refI[celli] > 0.9*phi0)
                {
                    s2 += sqr(hI[celli]);
                    hMax = max(hMax, mag(hI[celli]));
                    ++nIn;
                }
                if (refI[celli] > 0.4*phi0 && aI[celli] < 0.1) { ++nVoid; }
                if (indI[celli] > 0.02 && indI[celli] < 0.3
                 && refI[celli] > residualContent)
                {
                    fSum += mag(vsI[celli]); ++nF;
                    fLam += cmptMax(eigenValues(bI[celli]));
                    fT2 += sqr(tI[celli]);
                }
                if (indI[celli] > 0.9) { bSum += mag(vsI[celli]); ++nB; }
            }
            reduce(s2, sumOp<scalar>()); reduce(hMax, maxOp<scalar>());
            reduce(nIn, sumOp<label>()); reduce(nVoid, sumOp<label>());
            reduce(fSum, sumOp<scalar>()); reduce(nF, sumOp<label>());
            reduce(bSum, sumOp<scalar>()); reduce(nB, sumOp<label>());
            reduce(fLam, sumOp<scalar>()); reduce(fT2, sumOp<scalar>());
            const scalar fMean = nF ? fSum/nF : 0;
            const scalar bMean = nB ? bSum/nB : 0;

            // THE EXPLICIT STABILITY NUMBERS, D*dt/h^2, of the two elastic
            // terms that act explicitly on the skeleton, each against the
            // diagonal the block actually inverted (iAD11: inertia, drag and
            // the BSD stabiliser), h from the cell's closest neighbour.
            //   network: D = phi_s*iAD11*dPi/da/rho, the compact phase
            //     pressure (convexExcess, where dPi/da has a closed form);
            //   elastic: D = w*gD*G*iAD11/rho, the deviatoric partial
            //     stiffness -- measured only where network was deposited,
            //     because elsewhere be relaxes to I in ~1 us and there is no
            //     explicit elastic mode to bound. Under beRelaxBasis
            //     networkAbsent the clot's diffuse edge keeps its strain for
            //     the first time, and this is the number that says whether
            //     that edge is explicitly stable at this dt.
            // The larger of the two, per unit dt, is what maxDeltaT() bounds.
            scalar nuMax = 0, nuElMax = 0;
            {
                scalarField maxDC(mesh.nCells(), 0.0);
                const scalarField& dc = mesh.deltaCoeffs().primitiveField();
                const labelUList& own = mesh.owner();
                const labelUList& nei = mesh.neighbour();
                forAll(dc, facei)
                {
                    maxDC[own[facei]] = max(maxDC[own[facei]], dc[facei]);
                    maxDC[nei[facei]] = max(maxDC[nei[facei]], dc[facei]);
                }
                const scalarField& GI = GField_.primitiveField();
                const scalarField& DgI = Dg_.primitiveField();
                const scalarField& phiSI = phiSkel_.primitiveField();
                const scalarField& iA = iAD11.primitiveField();
                const scalar pm = networkPole();
                const scalar dt = runTime.deltaTValue();
                const scalar rhoV = rho_.value();
                const volScalarField intactF(intactFraction());
                const scalarField& intactI = intactF.primitiveField();
                const symmTensorField& beI = be_.primitiveField();
                // S16w: the weight THIS site used, per cell; recorded after
                // the loop. Cells outside the deposited network do not enter
                // the bound, and they carry the same formula, so they are
                // filled with it and cannot hide a defect in the ones that do.
                scalarField wSite(aI.size(), 0.0);
                scalar dampedStiffMax = 0;
                // S17 A (design 2A, docket 10): the stiffness per unit weight
                // is [(thI/S) stiff + (thP/S)(lf^2 - 1/lf)]/(1 - d), so the
                // number is max(ind, floor)[(thI/S) stiff + (thP/S)(lf^2 -
                // 1/lf)]. kSite is what THIS site used (S17split, site 2);
                // cells outside the bound carry the same formula.
                tmp<volScalarField> tKeff;
                scalarField kSite;
                {
                    tKeff = mixedTangentStiffness();
                    kSite = tKeff().primitiveField();
                }
                forAll(aI, celli)
                {
                    const scalar gDc = sqr(1.0 - DgI[celli]) + etaRes_;
                    const scalar h2 = sqr(maxDC[celli]);
                    {
                        wSite[celli] =
                          max(indI[celli], skeletonSolidFloor_)
                           *intactI[celli];
                    }
                    if (refI[celli] > residualContent)
                    {
                        // S14: the INTACT network's occupancy, as the stress
                        // uses it; and under elasticStabilityStiffness
                        // tangent, the incremental stiffness of G*dev(be)
                        // along its principal stretch rather than G. At
                        // be = I that is 4/3 G; stretched it grows ~b1. em58
                        // (run 20260923-204553) blew up with this number at
                        // 0.087 in the G form and 1.25 in the tangent form.
                        // S16 B: the weight is wSite, built above per
                        // skeletonFloorBasis (intactWeight: the shipped
                        // max(ind*(1 - d), floor), unchanged).
                        const scalar w = wSite[celli];
                        scalar stiff = 1.0;
                        {
                            const vector ev(eigenValues(beI[celli]));
                            const scalar b1 = cmptMax(ev);
                            const scalar bRest = ev.x() + ev.y() + ev.z() - b1;
                            stiff = (8.0/9.0)*b1 + (2.0/9.0)*bRest;
                        }
                        // H-A1n's falsifier (design section 6): the damaged
                        // tangent (1 - d)*stiff, whatever the weight.
                        dampedStiffMax =
                            max(dampedStiffMax, intactI[celli]*stiff);
                        // S17 A: the stiffness per unit weight this site
                        // uses, recorded (after H-A1n above read the raw
                        // intact tangent).
                        {
                            stiff = tKeff()[celli];
                            kSite[celli] = stiff;
                        }
                        const scalar Del =
                            w*gDc*stiff*GI[celli]*iA[celli]/rhoV;
                        nuElMax = max(nuElMax, Del*dt*h2);
                    }
                    {
                        const scalar a = aI[celli];
                        const scalar Kn = networkPressureCoeff_*gDc*GI[celli];
                        const scalar dpda =
                            Kn*(pm - phi0)*(sqr(a - phi0) + phi0*(pm - phi0))
                           /sqr(a*(pm - a));
                        // The SAME weight networkPartialPressure() applies,
                        // or the bound would read the force it no longer is.
                        const scalar wPi =
                            networkPressureWeight_ == "depositedOccupancy"
                          ? refI[celli]/phi0
                          : min(1.0, refI[celli]/residualContent);
                        // S18 C (R2b): d(w p)/da with w = presence
                        // theta/ref and da = dtheta: presence [(theta/ref)
                        // dp/da + p/ref] (numerics 10's slope w dp + p dw).
                        // Fix 25d: this loop runs over EVERY cell; in the
                        // lumen thetaNetRef = 0 and the unguarded form
                        // divided by it (batch 60, SIGFPE on step 1).
                        // networkPressureKernel::contentRatioSlope is 0
                        // where ref = 0 BY CONSTRUCTION: no network was
                        // deposited, presence w p is identically 0, so its
                        // slope is exactly 0 -- not a clamp.
                        scalar slopeA = wPi*dpda;
                        if (networkPressureWeight_ == "contentRatio")
                        {
                            const scalar thc = thetaC_[celli] + thetaH_[celli];
                            const scalar pa =
                                networkPressureKernel::convexExcessPressure
                                (
                                    Kn, pm, phi0, a
                                );
                            slopeA =
                                wPi
                               *networkPressureKernel::contentRatioSlope
                                (
                                    thc, refI[celli], dpda, pa
                                );
                        }
                        const scalar D =
                            phiSI[celli]*iA[celli]*slopeA/rhoV;
                        nuMax = max(nuMax, D*dt*h2);
                    }
                }
                reduce(nuMax, maxOp<scalar>());
                reduce(nuElMax, maxOp<scalar>());

                // For maxDeltaT(): the binding number per unit dt. S17 C
                // (design 2C): under implicitTangent the elastic number
                // leaves the rate -- the pair carries the tangent implicitly
                // -- and nuNet <= 0.2 and the wave CFL stay. The elastic
                // number is still measured and printed.
                explicitDiffusionRate_ = max(nuMax, nuElMax)/max(dt, vSmall);
                {
                    explicitDiffusionRate_ = nuMax/max(dt, vSmall);
                }

                // S16: the third weight site, and H-A1n's number.
                recordStressWeight(wSite, 2);
                {
                    recordStiffnessSplit(kSite);
                }
                reduce(dampedStiffMax, maxOp<scalar>());
                s16DampedStiffMax_ = max(s16DampedStiffMax_, dampedStiffMax);
            }

            Info<< "PHASE_CLOT_STRIPES time=" << runTime.userTimeValue()
                << " argHighPassRms=" << (nIn ? sqrt(s2/nIn) : 0)
                << " argHighPassMax=" << hMax
                << " interiorCells=" << nIn
                << " voidCells=" << nVoid
                << " fringeCells=" << nF
                << " fringeOverBodySpeed=" << (bMean > 0 ? fMean/bMean : 0)
                << " fringeBeLambdaMaxMean=" << (nF ? fLam/nF : 0)
                << " fringeThetaHighPassRms=" << (nF ? sqrt(fT2/nF) : 0)
                << " networkDiffusionNumberMax=" << nuMax
                << " elasticDiffusionNumberMax=" << nuElMax
                << endl;

            // S16 (design section 6), once all three weight sites and the
            // rupture law have run this step. Read by assess_case.py:
            // G12r (lawReleaseMax <= 1.001 r/(1 + r), r on this line), S16w
            // (weightAbsDevMax = 0 with weightSites = 7, two-sided), S16m
            // (mirrorDevMax <= 1e-12), W-dd (deltaDMax), H-A1n
            // (dampedStiffMax), and the R4 partition diagnostics: its
            // minimum component, the cells holding a subnormal component,
            // C+'s mu (min, mean) and donor fraction, and max|sum P - theta|
            // (absolute; partitionSumDevMax is the same relative to max
            // theta, the WATCH).
            {
                scalar sumDev = -1, sumDevAbs = -1, pMin = -1;
                label nSub = -1;
                {
                    const scalarField& pI = thetaIntact_.primitiveField();
                    const scalarField& pP = thetaPending_.primitiveField();
                    const scalarField& pB = thetaBroken_.primitiveField();
                    const scalarField& tC = thetaC_.primitiveField();
                    const scalarField& tH = thetaH_.primitiveField();
                    const scalar tinyP = std::numeric_limits<scalar>::min();
                    scalar dev = 0, thMax = 0;
                    pMin = great;
                    nSub = 0;
                    forAll(pI, celli)
                    {
                        const scalar th = tC[celli] + tH[celli];
                        thMax = max(thMax, th);
                        dev = max
                        (
                            dev,
                            mag(pI[celli] + (pP[celli] + pB[celli]) - th)
                        );
                        pMin = min(pMin, min(pI[celli], min(pP[celli], pB[celli])));
                        const scalar m3[3] = {pI[celli], pP[celli], pB[celli]};
                        bool sub = false;
                        for (label k = 0; k < 3; ++k)
                        {
                            sub = sub || (m3[k] != 0 && mag(m3[k]) < tinyP);
                        }
                        if (sub) { ++nSub; }
                    }
                    reduce(dev, maxOp<scalar>());
                    reduce(thMax, maxOp<scalar>());
                    reduce(pMin, minOp<scalar>());
                    reduce(nSub, sumOp<label>());
                    sumDevAbs = dev;
                    sumDev = thMax > 0 ? dev/thMax : 0;
                }
                Info<< "PHASE_CLOT_S16 time=" << runTime.userTimeValue()
                    << " r=" << s16LawRate_
                    << " lawReleaseMax=" << s16LawReleaseMax_
                    << " lawReleaseBound="
                    << s16LawRate_/(1.0 + s16LawRate_)
                    << " pendingMax=" << s16PendingMax_
                    << " deltaDMax=" << s16DeltaDMax_
                    << " dampedStiffMax=" << s16DampedStiffMax_
                    << " weightAbsDevMax=" << s16WeightAbsDevMax_
                    << " weightSites=" << s16WeightSites_
                    << " mirrorDevMax=" << s16MirrorDevMax_
                    << " partitionMin=" << pMin
                    << " partitionSubnormalCells=" << nSub
                    << " muMin=" << s16MuMin_
                    << " muMean=" << s16MuMean_
                    << " foutMax=" << s16FoutMax_
                    << " partitionSumDevAbs=" << sumDevAbs
                    << " partitionSumDevMax=" << sumDev
                    << endl;
            }

            // S17 (design section 6), once per step, after the law, the
            // rotation, P-LS and all three stress sites have run. Read by
            // assess_case.py: G-tr (transferB1Max <= 1.05), G-rot
            // (rotRatioMedian 1 +- 0.001 over cells with |L|dt <= 1e-3,
            // rotCells of them; NOT-EVALUABLE where rotCells = 0; the
            // median |L|dt of all candidates in rotLdtMedian), G-eig
            // (eigRelMax <= 1e-12), G-uni (uniUp = 0), G-proj (projRaised
            // = 0), S17split (splitDevMax0/1/2 = 0 with splitSites = 7), P2
            // (p2Max <= 1.05), W2 (w2Max), W1 (w1MaxDeg, w1Over5, w1Over10).
            {
                scalar p2Max = 0, w2Max = 0, w1Max = 0;
                label w1Over5 = 0, w1Over10 = 0, nPend = 0, nMixed = 0;
                {
                    const scalar lc = networkCriticalStretch_;
                    const scalar sc = sqr(lc) - 1.0/sqr(lc);
                    const scalarField& pI = thetaIntact_.primitiveField();
                    const scalarField& pP = thetaPending_.primitiveField();
                    const scalarField& pB = thetaBroken_.primitiveField();
                    const scalarField& lI = pendingStretch_.primitiveField();
                    const vectorField& nI = pendingAxis_.primitiveField();
                    const symmTensorField& beI = be_.primitiveField();
                    const volScalarField indS(phaseIndicator());
                    const scalarField& indSI = indS.primitiveField();
                    // S17b: the fibre's carrier (theta_T under V1), W2 read
                    // only where it is >= fibreReadMin() (A2).
                    const scalarField& pCS = fibreCarrier().primitiveField();
                    const scalar readMinS = fibreReadMin();
                    forAll(pP, celli)
                    {
                        if (!(pCS[celli] > 0))
                        {
                            continue;
                        }
                        scalar aI, aP, omd;
                        fibreReleaseKernel::sharesCarrier
                        (
                            pI[celli], pP[celli], pB[celli], pCS[celli],
                            aI, aP, omd
                        );
                        const symmTensor& b = beI[celli];
                        double Bb[3][3] =
                        {
                            {b.xx(), b.xy(), b.xz()},
                            {b.xy(), b.yy(), b.yz()},
                            {b.xz(), b.yz(), b.zz()}
                        };
                        double ev[3], V[3][3];
                        fibreReleaseKernel::eigSym(Bb, ev, V);
                        const scalar T =
                            fibreReleaseKernel::fibreTension(lI[celli]);
                        if (pCS[celli] >= readMinS)
                        {
                            w2Max = max(w2Max, T/sc);
                        }
                        if (indSI[celli] >= 1e-4)
                        {
                            ++nPend;
                            // 03's measure, split: the intact share at
                            // its own stretch, the pending share its fibre.
                            const scalar sCell =
                                aI*(ev[0] - 1.0/ev[0]) + aP*T;
                            p2Max = max(p2Max, sCell/sc);
                        }
                        if (aP >= 0.5 && ev[0] > 1.1*ev[1])
                        {
                            ++nMixed;
                            // the axis angle in [0, 90] deg, as atan2 of
                            // |n x v| and |n . v| (no acos domain issue)
                            const vector v(V[0][0], V[1][0], V[2][0]);
                            const scalar ang = radToDeg
                            (
                                atan2
                                (
                                    mag(nI[celli] ^ v),
                                    mag(nI[celli] & v)
                                )
                            );
                            w1Max = max(w1Max, ang);
                            if (ang > 5) { ++w1Over5; }
                            if (ang > 10) { ++w1Over10; }
                        }
                    }
                    reduce(p2Max, maxOp<scalar>());
                    reduce(w2Max, maxOp<scalar>());
                    reduce(w1Max, maxOp<scalar>());
                    reduce(w1Over5, sumOp<label>());
                    reduce(w1Over10, sumOp<label>());
                    reduce(nPend, sumOp<label>());
                    reduce(nMixed, sumOp<label>());
                }
                Info<< "PHASE_CLOT_S17 time=" << runTime.userTimeValue()
                    << " step=" << runTime.timeIndex()
                    << " deltaT=" << runTime.deltaTValue()
                    << " transferB1Max=" << s17TransferB1Max_
                    << " transfers=" << s17Transfers_
                    << " resetCells=" << s17ResetCells_
                    << " rotRatioMedian=" << s17RotRatioMedian_
                    << " rotCells=" << s17RotCells_
                    << " rotLdtMedian=" << s17RotLdtMedian_
                    << " rotCandidates=" << s17RotCandidates_
                    << " eigRelMax=" << s17EigRelMax_
                    << " uniUp=" << s17UniUp_
                    << " projCount=" << s17ProjCount_
                    << " projRaised=" << s17ProjRaised_
                    << " projExcessMax=" << s17ProjExcessMax_
                    << " projTies=" << s17ProjTies_
                    << " splitDevMax0=" << s17SplitDevMax_[0]
                    << " splitDevMax1=" << s17SplitDevMax_[1]
                    << " splitDevMax2=" << s17SplitDevMax_[2]
                    << " splitSites=" << s17SplitSites_
                    << " pendingCells=" << nPend
                    << " p2Max=" << p2Max
                    << " w2Max=" << w2Max
                    << " mixedCells=" << nMixed
                    << " w1MaxDeg=" << w1Max
                    << " w1Over5=" << w1Over5
                    << " w1Over10=" << w1Over10
                    << " pendingPressureMax=" << s17PendingPressureMax_
                    << " elasticCoupling=implicitTangent"
                    << " nuElMax=" << nuElMax
                    << " nuNetMax=" << nuMax
                    << endl;

                // S17b (design section 2 "New logs"; 02, 05, 06, 08, 12).
                {
                    const scalarField& pIb = thetaIntact_.primitiveField();
                    const scalarField& pPb = thetaPending_.primitiveField();
                    const scalarField& pBb = thetaBroken_.primitiveField();
                    const scalarField& pCb = fibreCarrier().primitiveField();
                    const scalarField& tTb = thetaTaut_.primitiveField();
                    const scalarField& lIb = pendingStretch_.primitiveField();
                    const scalarField& qPb = thetaPendingP_.primitiveField();
                    const vectorField& ccb = mesh.C().primitiveField();
                    const scalarField& volb = mesh.V();
                    // theta_T <= theta_P, exactly (V1)
                    label tautAbove = 0;
                    scalar tautAboveMax = 0;
                    // min lf over the carrier, with its content, presence,
                    // position and H-PA2's accumulators
                    scalar minLf = great, minLfTh = 0, minLfPres = 0;
                    vector minLfX(Zero);
                    scalar minLfSum = 0, minLfSumMin = 0;
                    // the slack content-share proxy of V1's tau lag
                    // (section 8 g: a content share, not a face traction)
                    scalar slackShareMax = 0;
                    // |p_P|/|min(p_n,U, 0)| and the frozen tension's power
                    const volScalarField pnU(networkPartialPressureUndamaged());
                    const scalarField& pnUI = pnU.primitiveField();
                    // div(Vs) of the skeleton's own flux: surfaceIntegrate,
                    // which is what fvc::div of a surface field returns, with
                    // no scheme lookup (default none)
                    const tmp<volScalarField::Internal> tDivVs
                    (
                        fvc::surfaceIntegrate(phiSkelOwn_)
                    );
                    const scalarField& divVsI = tDivVs();
                    scalar pPRatioMax = 0, frozenPower = 0;
                    forAll(pPb, celli)
                    {
                        if (tTb[celli] > pPb[celli])
                        {
                            ++tautAbove;
                            tautAboveMax =
                                max(tautAboveMax, tTb[celli] - pPb[celli]);
                        }
                        const scalar S =
                            pIb[celli] + (pPb[celli] + pBb[celli]);
                        // S18 A: the power of the APPLIED frozen force --
                        // the same frozenPressureShare the force takes, so
                        // it is identically 0 under frozenPendingPressure
                        // none (the conformance clause, design section 5).
                        frozenPower -=
                            fibreReleaseKernel::frozenPressureShare
                            (
                                qPb[celli], S, false
                            )*divVsI[celli]*volb[celli];
                        if (pIb[celli] + pPb[celli] > 0)
                        {
                            slackShareMax = max
                            (
                                slackShareMax,
                                (pPb[celli] - tTb[celli])
                               /(pIb[celli] + pPb[celli])
                            );
                        }
                        if (!(pCb[celli] > 0))
                        {
                            continue;
                        }
                        if (lIb[celli] < minLf)
                        {
                            minLf = lIb[celli];
                            minLfTh = pCb[celli];
                            minLfPres =
                                thetaNetRef_[celli]
                               /(residualChi_*skeletonVolumeFraction_);
                            minLfX = ccb[celli];
                            minLfSum = s17bSumLnMu_[celli];
                            minLfSumMin = s17bSumMinLnMu_[celli];
                        }
                        if
                        (
                            pCb[celli] >= std::numeric_limits<scalar>::min()
                         && pnUI[celli] < 0
                        )
                        {
                            pPRatioMax = max
                            (
                                pPRatioMax,
                                mag(qPb[celli]/pCb[celli])/mag(pnUI[celli])
                            );
                        }
                    }
                    reduce(tautAbove, sumOp<label>());
                    reduce(tautAboveMax, maxOp<scalar>());
                    reduce(slackShareMax, maxOp<scalar>());
                    reduce(pPRatioMax, maxOp<scalar>());
                    reduce(frozenPower, sumOp<scalar>());
                    // 08's counts: network cells below theta/ref 1/2, and
                    // with an anti-restoring partial pressure, S = dPi/dtheta
                    // < 0 at fixed shares (the shipped, presence-weighted
                    // form: W p'(a) (1 on packing, aI on tension)).
                    label halfCells = 0, sNeg = 0, netCells = 0;
                    {
                        const scalar phi0 = skeletonVolumeFraction_;
                        const scalar pm = networkPole();
                        const scalar resid = residualChi_*phi0;
                        const scalarField& refb = thetaNetRef_.primitiveField();
                        const scalarField& GIb = GField_.primitiveField();
                        const scalarField& DgIb = Dg_.primitiveField();
                        forAll(refb, celli)
                        {
                            if (!(refb[celli] > resid))
                            {
                                continue;
                            }
                            const scalar th = thetaC_[celli] + thetaH_[celli];
                            if (th < 0.5*refb[celli]) { ++halfCells; }
                            const scalar a = phi0 + th - refb[celli];
                            const scalar gDc = sqr(1.0 - DgIb[celli]) + etaRes_;
                            const scalar Kn = networkPressureCoeff_*gDc*GIb[celli];
                            const scalar dpda =
                                Kn*(pm - phi0)*(sqr(a - phi0) + phi0*(pm - phi0))
                               /sqr(a*(pm - a));
                            const scalar Wp =
                                networkPressureWeight_ == "depositedOccupancy"
                              ? refb[celli]/phi0
                              : min(1.0, refb[celli]/resid);
                            scalar aI = 1, aP = 0, omd = 1;
                            fibreReleaseKernel::shares
                            (
                                pIb[celli], pPb[celli], pBb[celli], aI, aP, omd
                            );
                            // S18 C (R2b): the content weight's own slope
                            // p/ref joins dp/da (08's S < 0, R2b's anti-
                            // restoring share, section 5's watch)
                            scalar dwp = dpda;
                            if (networkPressureWeight_ == "contentRatio")
                            {
                                // the kernel the dt bound takes (fix 25d);
                                // here refb > resid > 0 already
                                const scalar pa =
                                    networkPressureKernel::convexExcessPressure
                                    (
                                        Kn, pm, phi0, a
                                    );
                                dwp = networkPressureKernel::contentRatioSlope
                                (
                                    th, refb[celli], dpda, pa
                                );
                            }
                            ++netCells;
                            const scalar slope =
                                Wp*dwp*(pnUI[celli] >= 0 ? 1.0 : aI);
                            if (slope < 0) { ++sNeg; }
                        }
                    }
                    reduce(halfCells, sumOp<label>());
                    reduce(sNeg, sumOp<label>());
                    reduce(netCells, sumOp<label>());
                    Info<< "PHASE_CLOT_S17B time=" << runTime.userTimeValue()
                        << " step=" << runTime.timeIndex()
                        << " slackEvents=" << s17bSlackEvents_
                        << " slackContent=" << s17bSlackContent_
                        << " slackDdMax=" << s17bSlackDdMax_
                        << " slackMuMax=" << s17bSlackMuMax_
                        << " slackAtRest=" << s17bSlackAtRest_
                        << " slackQpMax=" << s17bSlackQpMax_
                        << " slackPreTMax=" << s17bSlackPreTMax_
                        << " slackPreTRatioMax=" << s17bSlackPreTRatioMax_
                        << " slackPreQpMax=" << s17bSlackPreQpMax_
                        << " slackPreQpRatioMax=" << s17bSlackPreQpRatioMax_
                        << " tautAboveCells=" << tautAbove
                        << " tautAboveMax=" << tautAboveMax
                        << " minLf=" << (minLf < great ? minLf : -1.0)
                        << " minLfTheta=" << minLfTh
                        << " minLfPresence=" << minLfPres
                        << " minLfX=" << minLfX.x()
                        << " minLfY=" << minLfX.y()
                        << " minLfSumLnMu=" << minLfSum
                        << " minLfSumMinLnMu=" << minLfSumMin
                        << " maxLfTransport=" << s17bMaxLfTransport_
                        << " maxLfRotation=" << s17bMaxLfRotation_
                        << " maxLfLaw=" << s17bMaxLfLaw_
                        << " projRaisedPresent=" << s17bProjRaisedPresent_
                        << " projRaisedAbsent=" << s17bProjRaisedAbsent_
                        << " thetaRefHalfCells=" << halfCells
                        << " sNegCells=" << sNeg
                        << " pPRatioMax=" << pPRatioMax
                        << " frozenPower=" << frozenPower
                        << " slackContentShareMax=" << slackShareMax
                        << " fibreSlack=v1"
                        << endl;

                    // S18 F (design section 2 F; docket 03 D6): the
                    // skeleton-carrier Courant number of this step, dt 0.5
                    // max(sum|phi_s|/V), with its cell, beside the limit
                    // maxDeltaT() computed (0.5/coRate; the step is carrier-
                    // bound when deltaT reaches it); B's wall-patch split
                    // departure (0 under split); the network-cell count
                    // behind S17B's sNegCells (R2b's anti-restoring share).
                    const scalarField sumPhiS
                    (
                        fvc::surfaceSum(mag(phiSkelOwn_))().primitiveField()
                    );
                    scalar coMax = -1;
                    label coCell = -1;
                    forAll(sumPhiS, celli)
                    {
                        const scalar co =
                            0.5*runTime.deltaTValue()*sumPhiS[celli]
                           /mesh.V()[celli];
                        if (co > coMax)
                        {
                            coMax = co;
                            coCell = celli;
                        }
                    }
                    const vector coX
                    (
                        coCell >= 0 ? mesh.C()[coCell] : vector::zero
                    );
                    Info<< "PHASE_CLOT_S18 time=" << runTime.userTimeValue()
                        << " step=" << runTime.timeIndex()
                        << " deltaT=" << runTime.deltaTValue()
                        << " carrierDtLimit=" << s18CarrierDtLimit_
                        << " coSkel=" << coMax
                        << " coSkelCell=" << coCell
                        << " coSkelX=" << coX.x()
                        << " coSkelY=" << coX.y()
                        << " wallSplitDevMax=" << s18WallSplitDevMax_
                        << " netCells=" << netCells
                        << " sNegCells=" << sNeg
                        << " frozenPower=" << frozenPower
                        << " frozenPendingPressure=none"
                        << " wallPatchSplit=" << "split"
                        << " networkPressureWeight=" << networkPressureWeight_
                        << " cplusOrder=shared"
                        << endl;

                    // S18p (design section 2): once per step, beside S18's
                    // line; the network cells (thetaNetRef > 1e-4) and the
                    // drained ones (theta/ref < 0.1, theta = thetaC +
                    // thetaH) as probes/ce-drained-em76-vs-em72.py counts
                    // them. ref > 1e-4 guards the quotient.
                    if (projSiteLog_)
                    {
                        const scalarField& refN = thetaNetRef_.primitiveField();
                        const scalarField& thCN = thetaC_.primitiveField();
                        const scalarField& thHN = thetaH_.primitiveField();
                        label s18pNet = 0, s18pDrained = 0;
                        forAll(refN, celli)
                        {
                            if (refN[celli] > 1e-4)
                            {
                                ++s18pNet;
                                if ((thCN[celli] + thHN[celli])/refN[celli] < 0.1)
                                {
                                    ++s18pDrained;
                                }
                            }
                        }
                        reduce(s18pNet, sumOp<label>());
                        reduce(s18pDrained, sumOp<label>());
                        const char* tag[2] = {"T", "P"};
                        Info<< "PHASE_CLOT_S18P time=" << runTime.userTimeValue()
                            << " step=" << runTime.timeIndex();
                        for (label k = 0; k < 2; ++k)
                        {
                            Info<< " projCount" << tag[k] << "=" << s18pCount_[k]
                                << " projRaised" << tag[k] << "=" << s18pRaised_[k]
                                << " projRaisedPresent" << tag[k] << "="
                                << s18pRaisedPresent_[k]
                                << " projRaisedAbsent" << tag[k] << "="
                                << s18pRaisedAbsent_[k]
                                << " projExcessMax" << tag[k] << "="
                                << s18pExcessMax_[k];
                        }
                        Info<< " networkCells=" << s18pNet
                            << " drainedCells=" << s18pDrained
                            << endl;
                    }
                }
            }
            {
                writeS17SecondDifferences();
            }
        }
    }

    fvc::correctUf(Uf, U, phi, MRF);
    fvc::makeRelative(phi, U);
}


bool Foam::solvers::phaseClotAxisymmetricFluid::s19BlockCoupled() const
{
    return momentumPressureCoupling_ == "blockCoupled";
}


void Foam::solvers::phaseClotAxisymmetricFluid::solveBlockCoupledStep()
{
    // ================================================================
    //  S19: THE STEP'S LINEAR SYSTEM, SOLVED EXACTLY
    // ================================================================
    //
    // research/designs/2026-10-03-S19-block-coupled-step.md, rev. 4,
    // APPROVED by Alireza 2026-10-04, section 2. One direct solve per step
    // for (Ux, Uy, Vsx, Vsy, p) replaces the U predictor, the Vs predictor and
    // the nCorrectors segregated correctors of correctPressureTwoVelocity(),
    // whose per-cell drag inversion left the OI-47 leftover (-2.11 per step
    // at k 4.07e-13, dt 4.76e-6). The rows and the UMFPACK interface are in
    // blockCoupledKernel.H (OpenFOAM-free, kernel-tested); everything below
    // reads the solver's EXISTING matrices and fields into it and writes the
    // solution back. Nothing here is computed twice: K and Kres are the drag
    // fields themselves (section 5 (a); G7 kMatch = 0).
    volScalarField& p(p_);
    volVectorField& U(U_);
    surfaceScalarField& phi(phi_);

    if (!tUEqn.valid() || !tVsEqn.valid())
    {
        FatalErrorInFunction
            << "momentumPressureCoupling blockCoupled: the two-velocity "
            << "matrices (UEqn, VsEqn) were not built this step, so there is "
            << "no shipped row to assemble (S19 section 2)." << exit(FatalError);
    }
    const label nNonOrth = pimple.nCorrNonOrth();
    if (nNonOrth == 1)
    {
        FatalErrorInFunction
            << "momentumPressureCoupling blockCoupled lags the non-orthogonal "
            << "correction from p^n with nNonOrthogonalCorrectors 1: lambda_min "
            << "-0.1845 on the throat mesh, the OI-47 defect class (S19 "
            << "section 2). Use 0 (orthogonal) or >= 2 back-substitutions."
            << exit(FatalError);
    }

    // ONE indicator for the whole step: the shipped isSkel, the bond-seed
    // test, VS_SPLIT and bodySlipRms all read it.
    const volScalarField indB(phaseIndicator());
    const scalarField& indBI = indB.primitiveField();

    // ---- RUN-TIME REFUSAL: FREE CLOTS ONLY (section 1, section 2) ----
    // The counts of commitWallBonds() (bondedClotFaces) and updateBondHold()
    // (an intact bond under material, ind > 0.5), from the same patch fields.
    {
        label bondedFaces = 0;
        label seeds = 0;
        const volVectorField::Boundary& Usbf = Us_.boundaryField();
        forAll(Usbf, patchi)
        {
            if (!isA<clotWallAdhesionFvPatchVectorField>(Usbf[patchi]))
            {
                continue;
            }
            const clotWallAdhesionFvPatchVectorField& bond =
                refCast<const clotWallAdhesionFvPatchVectorField>
                (
                    Usbf[patchi]
                );
            bondedFaces += bond.bondedClotFaces();
            const scalarField& bonded = bond.bonded();
            const labelUList& fc = Usbf[patchi].patch().faceCells();
            forAll(fc, facei)
            {
                if (bonded[facei] > 0.5 && indBI[fc[facei]] > 0.5)
                {
                    ++seeds;
                }
            }
        }
        reduce(bondedFaces, sumOp<label>());
        reduce(seeds, sumOp<label>());
        if (bondedFaces > 0 || seeds > 0)
        {
            FatalErrorInFunction
                << "momentumPressureCoupling blockCoupled is for free clots "
                << "(S19 section 1): this step has bondedClotFaces "
                << bondedFaces << " and bond-hold seeds " << seeds
                << "." << exit(FatalError);
        }
    }

    fvVectorMatrix& UEqn = tUEqn.ref();
    fvVectorMatrix& VsEqn = tVsEqn.ref();
    // Const views for diag/upper/lower/source: the non-const lower() of a
    // symmetric lduMatrix would allocate a copy and change the matrix.
    const fvVectorMatrix& UEqnC = UEqn;
    const fvVectorMatrix& VsEqnC = VsEqn;

    const volScalarField& phiSB = phiSkel_;
    const volScalarField& phiFB = phiFluid_;
    const surfaceScalarField phiSfB(fvc::interpolate(phiSB));
    const surfaceScalarField phiFfB(fvc::interpolate(phiFB));

    // THE SOLVER'S DRAG FIELDS, read, not recomputed.
    const volScalarField KdB("KdB", clotDrag_);
    const volScalarField KresB("Kres", skeletonResidualDrag());

    // The explicit elastic force at time level n: face form, and its cell
    // form as the Vs predictor reconstructs it.
    const surfaceScalarField fSigmaEB(skeletonElasticForceFlux());
    const volVectorField FelB("FelB", fvc::reconstruct(fSigmaEB));

    // THE SHIPPED isSkel (correctPressureTwoVelocity, residualChi > 0:
    // ind >= residualChi/2, else phi_s >= residualPhiS/2).
    const volScalarField isSkelB
    (
        residualChi_ > 0
      ? pos0
        (
            indB - dimensionedScalar("halfChi", dimless, 0.5*residualChi_)
        )()
      : pos0
        (
            phiSB - dimensionedScalar("half", dimless, 0.5*residualPhiS_)
        )()
    );

    // ---- THE MESH AND THE COEFFICIENTS, INTO PLAIN ARRAYS ----
    const label nC = mesh.nCells();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const surfaceVectorField& Sf = mesh.Sf();
    const surfaceScalarField& magSf = mesh.magSf();
    const surfaceScalarField& wts = mesh.weights();
    const surfaceScalarField& ndcf = mesh.nonOrthDeltaCoeffs();
    const surfaceVectorField& kvf = mesh.nonOrthCorrectionVectors();

    blockCoupled::Mesh bm;
    blockCoupled::Coeffs coef;
    bm.nCells = nC;
    bm.V.resize(nC);
    bm.Cx.resize(nC);
    bm.Cy.resize(nC);
    forAll(mesh.C(), i)
    {
        bm.V[i] = mesh.V()[i];
        bm.Cx[i] = mesh.C()[i].x();
        bm.Cy[i] = mesh.C()[i].y();
    }
    const label nF = own.size();
    bm.own.resize(nF); bm.nei.resize(nF);
    bm.Sx.resize(nF); bm.Sy.resize(nF); bm.magS.resize(nF); bm.w.resize(nF);
    bm.ndc.resize(nF); bm.kx.resize(nF); bm.ky.resize(nF);
    forAll(own, f)
    {
        bm.own[f] = own[f];
        bm.nei[f] = nei[f];
        bm.Sx[f] = Sf[f].x();
        bm.Sy[f] = Sf[f].y();
        bm.magS[f] = magSf[f];
        bm.w[f] = wts[f];
        bm.ndc[f] = ndcf[f];
        bm.kx[f] = kvf[f].x();
        bm.ky[f] = kvf[f].y();
    }

    // The momentum rows exactly as fvMatrix::solveSegregated forms them:
    // diag + the component of internalCoeffs, source + boundaryCoeffs.
    const scalarField& KdIn = KdB.primitiveField();
    const scalarField& KresIn = KresB.primitiveField();
    for (direction k = 0; k < 2; ++k)
    {
        coef.dU[k].resize(nC); coef.sU[k].resize(nC);
        coef.dVs[k].resize(nC); coef.sVs[k].resize(nC);
        for (label i = 0; i < nC; ++i)
        {
            coef.dU[k][i] = UEqnC.diag()[i];
            coef.sU[k][i] = UEqnC.source()[i].component(k);
            coef.dVs[k][i] = VsEqnC.diag()[i];
            coef.sVs[k][i] = VsEqnC.source()[i].component(k);
        }
    }
    coef.upU.resize(nF); coef.loU.resize(nF);
    coef.upVs.resize(nF); coef.loVs.resize(nF);
    forAll(own, f)
    {
        coef.upU[f] = UEqnC.upper()[f];
        coef.loU[f] = UEqnC.lower()[f];
        coef.upVs[f] = VsEqnC.upper()[f];
        coef.loVs[f] = VsEqnC.lower()[f];
    }
    coef.K.resize(nC); coef.Kres.resize(nC);
    coef.phiF.resize(nC); coef.phiS.resize(nC);
    coef.FelX.resize(nC); coef.FelY.resize(nC); coef.isSkel.resize(nC);
    for (label i = 0; i < nC; ++i)
    {
        coef.K[i] = KdIn[i];
        coef.Kres[i] = KresIn[i];
        coef.phiF[i] = phiFB[i];
        coef.phiS[i] = phiSB[i];
        coef.FelX[i] = FelB[i].x();
        coef.FelY[i] = FelB[i].y();
        coef.isSkel[i] = isSkelB[i] > 0.5 ? 1 : 0;
    }
    // R3 (section 12): -div(nu3w_f (grad Vs + grad Vs^T)). Fix round 12
    // (physics 11 row 5b; CE ruling 2026-10-04): nu3w_f is tau_v/rho times
    // the elastic stress's own face modulus series(w)_f interp(gD G)_f, on
    // its own weight w = skeletonStressWeight() = max(ind, 1e-4)(1 - d) and
    // its own gD, so tau_KV = tau_v in every cell. With ind = theta/phi0
    // this is series(theta(1 - d)) eta_n gD, eta_n = G tau_v/phi0.
    coef.r3 = networkViscosity_ == "kelvinVoigt";
    // S19b section 2.2: eliminate removes the lumen Vs unknowns; identityRows
    // keeps S19a's identity rows.
    coef.eliminateLumen = true;
    coef.nu3w.assign(nF, 0.0);
    tmp<surfaceScalarField> tNu3w;
    if (coef.r3)
    {
        const volScalarField wR3("wR3", skeletonStressWeight());
        const volScalarField gDR3(sqr(1.0 - Dg_) + etaRes_);
        const surfaceScalarField sWR3(seriesFaceWeights(wR3));
        const surfaceScalarField gDGR3(fvc::interpolate(gDR3*GField_));
        // Boundary values only: every boundary weight is zeroed below.
        tNu3w = sWR3*gDGR3;
        forAll(own, f)
        {
            coef.nu3w[f] = blockCoupled::r3FaceCoefficient
            (
                sWR3[f], gDGR3[f], networkViscosityTau_, rho_.value()
            );
        }
    }


    // CE ruling F16 (2026-10-04): under R3 the face inverse's A_s carries
    // R3's diagonal (numerics 02 section 1), consistent with the Vs rows
    // solved; zero, and the shipped inverse exactly, with R3 off.
    volScalarField r3AsB
    (
        IOobject("r3AsB", runTime.name(), mesh),
        mesh,
        dimensionedScalar("zero", dimless/dimTime, 0),
        extrapolatedCalculatedFvPatchScalarField::typeName
    );
    if (coef.r3)
    {
        std::vector<blockCoupled::LsqRow> lsqA;
        if (!blockCoupled::lsqRows(bm, lsqA))
        {
            FatalErrorInFunction
                << "networkViscosity kelvinVoigt: a cell lacks two independent "
                << "neighbours for R3's least-squares gradient (S19 section "
                << "12)." << exit(FatalError);
        }
        std::vector<double> dAs;
        blockCoupled::r3AsDiagonal(bm, coef, lsqA, dAs);
        forAll(r3AsB, i)
        {
            r3AsB[i] = dAs[i];
        }
        r3AsB.correctBoundaryConditions();
    }

    // ---- THE SHIPPED FACE-ASSEMBLED 2x2 INVERSE (.C correctPressure-
    // TwoVelocity, blockFaceInverse faceAssembled): rA_f and the elastic
    // coefficient cE_f of the Rhie-Chow face come from it (section 2).
    const volScalarField AblkB("AblkB", UEqn.A() + KdB);
    const volScalarField BblkB("BblkB", -KdB);
    const volScalarField CblkB("CblkB", -(KdB + KresB));
    const volScalarField DblkB("DblkB", VsEqn.A() + KdB + KresB + r3AsB);

    // ---- S25 D: THE NETWORK-PRESSURE DIFFUSION BOUND ON THE BLOCK PATH -----
    // (design 2026-10-10-S25 section 2D; OI-74; S25 03 findings 1-2.) The
    // block applies the network pressure EXPLICITLY (time level n, in the
    // elastic force fSigmaEB above), and its diffusion bound was computed
    // only in correctPressureTwoVelocity(), which this path never calls --
    // so it was dead here since S19. It is a stability bound (S25 03 Q3).
    // The mobility is the solve's PRESSURE-PROJECTED (Schur) one, from the
    // cell 2x2 assembled just above -- with R3's diagonal r3AsB in Dblk --
    // and NOT the local inverse Ablk/det, which overstates the clot-cell
    // mobility by ~Kd*dt/phiF^2:
    //   iA_S = i11 - aS*cE/rA,  aS = i10*phiF + i11*phiS,
    //   cE = phiF*i01 + phiS*i11,  rA = phiF*(i00*phiF + i01*phiS) + phiS*aS,
    // the cell form of aByADsfB, cEfB and rAfB below. The number per unit
    // dt is phiS*iA_S*slope*maxDC^2/rho, with the slope wPi*dPi/da exactly
    // as the segregated site forms it (convexExcess, the same weight
    // networkPartialPressure() applies), and maxDeltaT() holds it to 0.2 at
    // the next step, the same one-step lag as the segregated path. No extra
    // solve; no field is written.
    //
    // The population is the cells that carry the term: phiS > 0 (the lumen
    // has no skeleton, so phiS*slope = 0 there, skipped BEFORE any division)
    // and a positive slope (wPi > 0: deposited network). In that population
    // the cell det and rA are refused if not positive, as the face tripwire
    // below refuses det_f.
    {
        const volScalarField argS25(networkPressureArgument());
        const volScalarField wS25(skeletonStressWeight());
        const scalarField& aS25 = argS25.primitiveField();
        const scalarField& wS25I = wS25.primitiveField();
        const scalarField& refS25 = thetaNetRef_.primitiveField();
        const scalarField& GS25 = GField_.primitiveField();
        const scalarField& DgS25 = Dg_.primitiveField();
        const scalar phi0S25 = skeletonVolumeFraction_;
        const scalar pmS25 = networkPole();
        const scalar residualContentS25 = residualChi_*skeletonVolumeFraction_;
        const scalar rhoS25 = rho_.value();

        scalarField maxDCS25(mesh.nCells(), 0.0);
        {
            const scalarField& dcS25 = mesh.deltaCoeffs().primitiveField();
            const labelUList& ownS25 = mesh.owner();
            const labelUList& neiS25 = mesh.neighbour();
            forAll(dcS25, facei)
            {
                maxDCS25[ownS25[facei]] =
                    max(maxDCS25[ownS25[facei]], dcS25[facei]);
                maxDCS25[neiS25[facei]] =
                    max(maxDCS25[neiS25[facei]], dcS25[facei]);
            }
        }

        scalar netRateS25 = 0;
        label netCellS25 = -1;
        scalar netIASS25 = 0;
        forAll(aS25, celli)
        {
            const scalar phiSc = phiSB[celli];
            if (!(phiSc > 0))
            {
                continue;
            }
            const scalar wPiS25 =
                networkPressureWeight_ == "depositedOccupancy"
              ? refS25[celli]/phi0S25
              : min(1.0, refS25[celli]/residualContentS25);
            if (!(wPiS25 > 0))
            {
                continue;
            }
            const scalar aC = aS25[celli];
            const scalar gDS25 = sqr(1.0 - DgS25[celli]) + etaRes_;
            const scalar KnS25 = networkPressureCoeff_*gDS25*GS25[celli];
            const scalar dpdaS25 =
                KnS25*(pmS25 - phi0S25)
               *(sqr(aC - phi0S25) + phi0S25*(pmS25 - phi0S25))
               /sqr(aC*(pmS25 - aC));
            scalar slopeS25 = wPiS25*dpdaS25;
            if (networkPressureWeight_ == "contentRatio")
            {
                const scalar thS25 = thetaC_[celli] + thetaH_[celli];
                const scalar paS25 =
                    networkPressureKernel::convexExcessPressure
                    (
                        KnS25, pmS25, phi0S25, aC
                    );
                slopeS25 =
                    wPiS25
                   *networkPressureKernel::contentRatioSlope
                    (
                        thS25, refS25[celli], dpdaS25, paS25
                    );
            }
            if (!(slopeS25 > 0))
            {
                continue;
            }

            const scalar AbS25 = AblkB[celli];
            const scalar BbS25 = BblkB[celli];
            const scalar CbS25 = CblkB[celli];
            const scalar DbS25 = DblkB[celli];
            const scalar detS25 = AbS25*DbS25 - BbS25*CbS25;
            if (!(detS25 > 0))
            {
                FatalErrorInFunction
                    << "S25 network diffusion bound: the cell 2x2 block is "
                    << "singular or sign-indefinite in cell " << celli
                    << " (det " << detS25 << ", phiS " << phiSc
                    << ") where the explicit network term acts."
                    << exit(FatalError);
            }
            const scalar i00S25 = DbS25/detS25;
            const scalar i01S25 = -BbS25/detS25;
            const scalar i10S25 = -CbS25/detS25;
            const scalar i11S25 = AbS25/detS25;
            const scalar phiFc = phiFB[celli];
            const scalar aSS25 = i10S25*phiFc + i11S25*phiSc;
            const scalar cES25 = phiFc*i01S25 + phiSc*i11S25;
            const scalar rAS25 = phiFc*(i00S25*phiFc + i01S25*phiSc) + phiSc*aSS25;
            if (!(rAS25 > 0))
            {
                FatalErrorInFunction
                    << "S25 network diffusion bound: the projected mixture "
                    << "mobility rA is not positive in cell " << celli
                    << " (rA " << rAS25 << ", phiS " << phiSc << ", phiF "
                    << phiFc << ")." << exit(FatalError);
            }
            const scalar iASS25 = i11S25 - aSS25*cES25/rAS25;
            const scalar rateS25 =
                phiSc*iASS25*slopeS25*sqr(maxDCS25[celli])/rhoS25;
            if (rateS25 > netRateS25)
            {
                netRateS25 = rateS25;
                netCellS25 = celli;
                netIASS25 = iASS25;
            }
        }
        // Serial (S19: the block path refuses coupled patches).
        explicitDiffusionRate_ = netRateS25;
        s25NetDiffCell_ = netCellS25;
        s25NetDiffIAS_ = netIASS25;
        s25NetDiffR3As_ = netCellS25 >= 0 ? r3AsB[netCellS25] : 0;
        s25NetDiffW_ = netCellS25 >= 0 ? wS25I[netCellS25] : 0;
        s25NetDiffA_ = netCellS25 >= 0 ? aS25[netCellS25] : 0;
    }
    const surfaceScalarField AfB(fvc::interpolate(AblkB));
    const surfaceScalarField BfB(fvc::interpolate(BblkB));
    const surfaceScalarField CfB(fvc::interpolate(CblkB));
    const surfaceScalarField DfB(fvc::interpolate(DblkB));
    const surfaceScalarField detfB("detfB", AfB*DfB - BfB*CfB);
    {
        // Every face, internal and boundary: the same tripwire as the
        // shipped face inverse, before any division by det_f.
        scalar detfMin = gMin(detfB.primitiveField());
        forAll(detfB.boundaryField(), patchi)
        {
            if (detfB.boundaryField()[patchi].size())
            {
                detfMin = min(detfMin, Foam::min(detfB.boundaryField()[patchi]));
            }
        }
        reduce(detfMin, minOp<scalar>());
        if (detfMin <= 0)
        {
            FatalErrorInFunction
                << "the face-assembled 2x2 block is singular or "
                << "sign-indefinite: min(det_f) = " << detfMin
                << exit(FatalError);
        }
    }
    const surfaceScalarField iAD00fB(DfB/detfB);
    const surfaceScalarField iAD01fB(-BfB/detfB);
    const surfaceScalarField iAD10fB(-CfB/detfB);
    const surfaceScalarField iAD11fB(AfB/detfB);
    const surfaceScalarField aByADffB(iAD00fB*phiFfB + iAD01fB*phiSfB);
    const surfaceScalarField aByADsfB(iAD10fB*phiFfB + iAD11fB*phiSfB);
    const surfaceScalarField rAfB("rAfB", phiFfB*aByADffB + phiSfB*aByADsfB);
    const surfaceScalarField cEfB("cEfB", phiFfB*iAD01fB + phiSfB*iAD11fB);

    coef.phiFf.resize(nF); coef.phiSf.resize(nF); coef.rAf.resize(nF);
    coef.cEf.resize(nF); coef.fSigE.resize(nF);
    forAll(own, f)
    {
        coef.phiFf[f] = phiFfB[f];
        coef.phiSf[f] = phiSfB[f];
        coef.rAf[f] = rAfB[f];
        coef.cEf[f] = cEfB[f];
        coef.fSigE[f] = fSigmaEB[f];
    }

    // The boundary faces. Empty patches have no faces; a coupled patch is
    // refused (serial, this series' patches only), as is a p condition the
    // block's continuity face cannot represent.
    std::vector<label> bPatch, bFace;
    forAll(mesh.boundary(), patchi)
    {
        const fvPatch& fp = mesh.boundary()[patchi];
        if (!fp.size())
        {
            continue;
        }
        const fvPatchScalarField& pp = p.boundaryField()[patchi];
        const fvPatchVectorField& Up = U.boundaryField()[patchi];
        const fvPatchVectorField& Vsp = Vs_.boundaryField()[patchi];
        if (fp.coupled())
        {
            FatalErrorInFunction
                << "momentumPressureCoupling blockCoupled: patch "
                << fp.name() << " is coupled; the block is serial over this "
                << "series' patches (S19 section 2)." << exit(FatalError);
        }
        if (pp.type() != "fixedValue" && pp.type() != "zeroGradient")
        {
            FatalErrorInFunction
                << "momentumPressureCoupling blockCoupled: p on patch "
                << fp.name() << " is " << pp.type() << "; the block's "
                << "continuity face takes fixedValue or zeroGradient p only "
                << "(S19 section 2 boundary table)." << exit(FatalError);
        }
        if
        (
            Up.assignable()
         && (
                (
                    Up.type() != "zeroGradient"
                 && Up.type() != "pressureInletOutletVelocity"
                )
             || Vsp.type() != "zeroGradient"
            )
        )
        {
            FatalErrorInFunction
                << "momentumPressureCoupling blockCoupled: open patch "
                << fp.name() << " has U " << Up.type() << " and Vs "
                << Vsp.type() << "; its continuity face takes U and Vs from "
                << "the interior (zeroGradient or pressureInletOutletVelocity "
                << "U, zeroGradient Vs; S19 section 2 boundary table)."
                << exit(FatalError);
        }
        forAll(fp, facei)
        {
            bPatch.push_back(patchi);
            bFace.push_back(facei);
        }
    }
    const label nB = bPatch.size();
    bm.bCell.resize(nB); bm.bSx.resize(nB); bm.bSy.resize(nB);
    bm.bMagS.resize(nB); bm.bOpen.resize(nB);
    coef.bPhiF.resize(nB); coef.bPhiS.resize(nB); coef.bRAf.resize(nB);
    coef.bCE.resize(nB); coef.bFSigE.resize(nB); coef.bWallFlux.assign(nB, 0.0);
    coef.pvIC.resize(nB); coef.pvBC.resize(nB);
    coef.pgIC.resize(nB); coef.pgBC.resize(nB);
    coef.bNu3w.assign(nB, 0.0);
    {
        label b = 0;
        forAll(mesh.boundary(), patchi)
        {
            const fvPatch& fp = mesh.boundary()[patchi];
            if (!fp.size())
            {
                continue;
            }
            const labelUList& fc = fp.faceCells();
            const fvPatchScalarField& pp = p.boundaryField()[patchi];
            const fvPatchVectorField& Up = U.boundaryField()[patchi];
            const tmp<scalarField> tvIC
            (
                pp.valueInternalCoeffs(tmp<scalarField>(wts.boundaryField()[patchi]))
            );
            const tmp<scalarField> tvBC
            (
                pp.valueBoundaryCoeffs(tmp<scalarField>(wts.boundaryField()[patchi]))
            );
            const tmp<scalarField> tgIC(pp.gradientInternalCoeffs());
            const tmp<scalarField> tgBC(pp.gradientBoundaryCoeffs());
            const vectorField& Sb = Sf.boundaryField()[patchi];
            const bool open = Up.assignable();
            forAll(fp, facei)
            {
                bm.bCell[b] = fc[facei];
                bm.bSx[b] = Sb[facei].x();
                bm.bSy[b] = Sb[facei].y();
                bm.bMagS[b] = magSf.boundaryField()[patchi][facei];
                bm.bOpen[b] = open ? 1 : 0;
                coef.bPhiF[b] = phiFfB.boundaryField()[patchi][facei];
                coef.bPhiS[b] = phiSfB.boundaryField()[patchi][facei];
                coef.bRAf[b] = rAfB.boundaryField()[patchi][facei];
                coef.bCE[b] = cEfB.boundaryField()[patchi][facei];
                coef.bFSigE[b] = fSigmaEB.boundaryField()[patchi][facei];
                if (!open)
                {
                    coef.bWallFlux[b] = Up[facei] & Sb[facei];
                }
                coef.pvIC[b] = tvIC()[facei];
                coef.pvBC[b] = tvBC()[facei];
                coef.pgIC[b] = tgIC()[facei];
                coef.pgBC[b] = tgBC()[facei];
                if (coef.r3)
                {
                    coef.bNu3w[b] = tNu3w().boundaryField()[patchi][facei];
                }
                ++b;
            }
        }
        // R3 puts no traction on any boundary face (section 12).
        blockCoupled::r3ZeroBoundaryWeights(coef.bNu3w);
    }

    // The momentum rows' boundary terms, per patch face cell.
    forAll(mesh.boundary(), patchi)
    {
        const labelUList& fc = mesh.boundary()[patchi].faceCells();
        const vectorField& iU = UEqn.internalCoeffs()[patchi];
        const vectorField& bU = UEqn.boundaryCoeffs()[patchi];
        const vectorField& iV = VsEqn.internalCoeffs()[patchi];
        const vectorField& bV = VsEqn.boundaryCoeffs()[patchi];
        forAll(fc, facei)
        {
            for (direction k = 0; k < 2; ++k)
            {
                coef.dU[k][fc[facei]] += iU[facei].component(k);
                coef.sU[k][fc[facei]] += bU[facei].component(k);
                coef.dVs[k][fc[facei]] += iV[facei].component(k);
                coef.sVs[k][fc[facei]] += bV[facei].component(k);
            }
        }
    }

    // ---- THE SOLVE ----
    std::vector<double> x0(blockCoupled::nVar*nC);
    for (label i = 0; i < nC; ++i)
    {
        x0[blockCoupled::idx(i, 0)] = U[i].x();
        x0[blockCoupled::idx(i, 1)] = U[i].y();
        x0[blockCoupled::idx(i, 2)] = Vs_[i].x();
        x0[blockCoupled::idx(i, 3)] = Vs_[i].y();
        x0[blockCoupled::idx(i, 4)] = p[i];
    }
    // S22 section 2.4: PARDISO (loaded in the constructor) or UMFPACK.
    const bool pardisoB = (blockLinearSolver_ == "pardiso");
    blockCoupled::Step st;
    if (pardisoB)
    {
        blockCoupled::solveStep(bm, coef, nNonOrth, x0, blockPardiso_(), st);
    }
    else
    {
        if (!blockLU_.valid())
        {
            blockLU_.reset(new blockCoupled::UmfpackLU());
        }
        blockCoupled::solveStep(bm, coef, nNonOrth, x0, blockLU_(), st);
    }
    // S22 sections 2.8 and 4: any PARDISO error, and any perturbed pivot,
    // is refused; iparm 15-17 and the matrix in place of UMFPACK's
    // estimates.
    if (pardisoB && st.status == blockCoupled::PardisoLU::statusPerturbed)
    {
        FatalErrorInFunction
            << "momentumPressureCoupling blockCoupled, blockLinearSolver "
            << "pardiso: phase 22 perturbed " << st.perturbedPivots
            << " pivot(s) (perturbedPivots, iparm 14). A perturbed pivot can "
            << "silence a singular matrix, so it is refused as UMFPACK's "
            << "singular-matrix warning is (S22 sections 2.8, 4)."
            << " The matrix: n=" << st.n << " nnz=" << st.nnz
            << " rowSumMin=" << st.rowSumMin << " rowSumMax=" << st.rowSumMax
            << "; PARDISO's memory (KB): iparm15=" << st.iparm15KB
            << " iparm16=" << st.iparm16KB << " iparm17=" << st.iparm17KB
            << exit(FatalError);
    }
    if (pardisoB && st.status != blockCoupled::PardisoLU::statusOk)
    {
        FatalErrorInFunction
            << "momentumPressureCoupling blockCoupled, blockLinearSolver "
            << "pardiso: PARDISO error " << st.status << " in phase "
            << blockPardiso_->failedPhase() << " at stage " << st.stage
            << " (phases: 11 analysis, 22 numeric factorisation, 33 "
            << "back-substitution, -1 release; oneMKL's pardiso errors are "
            << "-1 to -15; the matrix checker, iparm 27, runs on the first "
            << "analysis and reports an unsorted or invalid CSR there. Stage "
            << "4 is the setup: -1000 a cell without two independent "
            << "neighbours for R3, -1001 a non-zero R3 boundary weight). "
            << "Any error other than 0 is refused (S22 sections 2.8, 4)."
            << " The matrix: n=" << st.n << " nnz=" << st.nnz
            << " rowSumMin=" << st.rowSumMin << " rowSumMax=" << st.rowSumMax
            << "; PARDISO's memory (KB): iparm15=" << st.iparm15KB
            << " iparm16=" << st.iparm16KB << " iparm17=" << st.iparm17KB
            << " perturbedPivots=" << st.perturbedPivots
            << exit(FatalError);
    }
    if (!pardisoB && st.status != UMFPACK_OK)
    {
        FatalErrorInFunction
            << "momentumPressureCoupling blockCoupled: UMFPACK status "
            << st.status << " at stage " << st.stage << " (1 symbolic, 2 "
            << "numeric, 3 solve, 4 setup: -1000 a cell without two "
            << "independent neighbours for R3, -1001 a non-zero R3 boundary "
            << "weight). Any status other than UMFPACK_OK, the singular-"
            << "matrix warning included, is refused (S19 section 2)."
            << " The matrix (S19a section 2.4): n=" << st.n
            << " nnz=" << st.nnz
            << " rowSumMin=" << st.rowSumMin
            << " rowSumMax=" << st.rowSumMax
            << "; the symbolic estimates: lunzEstimate=" << st.lunzEstimate
            << " numericSizeEstimateGB=" << st.numericSizeEstimateGB
            << " peakMemoryEstimateGB=" << st.peakMemoryEstimateGB
            << " (GB = Info Units x UMFPACK_SIZE_OF_UNIT / 1e9; -1 where "
            << "UMFPACK reported none)."
            << exit(FatalError);
    }

    // ---- WRITE BACK: the velocities, p and the fluxes from the ONE solve ----
    for (label i = 0; i < nC; ++i)
    {
        U[i] = vector
        (
            st.x[blockCoupled::idx(i, 0)], st.x[blockCoupled::idx(i, 1)],
            U[i].z()
        );
        Vs_[i] = vector
        (
            st.x[blockCoupled::idx(i, 2)], st.x[blockCoupled::idx(i, 3)],
            Vs_[i].z()
        );
        p[i] = st.x[blockCoupled::idx(i, 4)];
    }
    p.correctBoundaryConditions();
    U.correctBoundaryConditions();
    fvConstraints().constrain(U);
    Vs_.correctBoundaryConditions();
    fvConstraints().constrain(Vs_);

    // phi = F_f. Each phase's own flux by the same Rhie-Chow construction,
    // with the phase's share of the face inverse:
    //   fluid:    S.interp(U)  - aByADff Xp + iAD01f El
    //   skeleton: S.interp(Vs) - aByADsf Xp + iAD11f El
    // so phiFf*phiFluidOwn + phiSf*phiSkelOwn = F_f exactly (rA_f and cE_f
    // are these coefficients' phi-weighted sums).
    forAll(own, f)
    {
        phi[f] = st.F[f];
        phiFluidOwn_[f] =
            st.Fu[f] - aByADffB[f]*st.Xp[f] + iAD01fB[f]*st.El[f];
        phiSkelOwn_[f] =
            st.Fv[f] - aByADsfB[f]*st.Xp[f] + iAD11fB[f]*st.El[f];
    }
    scalar qInlet = 0, qOutlet = 0;
    const label inletI = mesh.boundaryMesh().findIndex("inlet");
    const label outletI = mesh.boundaryMesh().findIndex("outlet");
    if (inletI < 0 || outletI < 0)
    {
        FatalErrorInFunction
            << "momentumPressureCoupling blockCoupled logs qInlet and qOutlet "
            << "on the patches inlet and outlet; this mesh lacks one (S19 "
            << "section 2 diagnostics)." << exit(FatalError);
    }
    for (label b = 0; b < nB; ++b)
    {
        const label patchi = bPatch[b], facei = bFace[b];
        phi.boundaryFieldRef()[patchi][facei] = st.bF[b];
        if (bm.bOpen[b])
        {
            phiFluidOwn_.boundaryFieldRef()[patchi][facei] =
                st.bFu[b]
              - aByADffB.boundaryField()[patchi][facei]*st.bXp[b]
              + iAD01fB.boundaryField()[patchi][facei]*st.bEl[b];
            phiSkelOwn_.boundaryFieldRef()[patchi][facei] =
                st.bFv[b]
              - aByADsfB.boundaryField()[patchi][facei]*st.bXp[b]
              + iAD11fB.boundaryField()[patchi][facei]*st.bEl[b];
        }
        else
        {
            phiFluidOwn_.boundaryFieldRef()[patchi][facei] = st.bF[b];
            phiSkelOwn_.boundaryFieldRef()[patchi][facei] = st.bF[b];
        }
        if (patchi == inletI) { qInlet += st.bF[b]; }
        if (patchi == outletI) { qOutlet += st.bF[b]; }
    }
    // NO SKELETON FLUX THROUGH A WALL, as the shipped corrector.
    forAll(phiSkelOwn_.boundaryFieldRef(), patchi)
    {
        if (isA<wallFvPatch>(mesh.boundary()[patchi]))
        {
            phiSkelOwn_.boundaryFieldRef()[patchi] = 0.0;
        }
    }

    continuityErrors();

    p.relax();

    // ---- PHASE_CLOT_S19: one record per step (section 2 diagnostics) ----
    {
        scalar contResAbs = 0;
        for (label i = 0; i < nC; ++i)
        {
            contResAbs = max(contResAbs, mag(st.cellSum[i]));
        }
        // kMatch: the drag the block used against the solver's drag fields,
        // re-read here; exactly 0 by construction (G7).
        const volScalarField KresNow("KresNow", skeletonResidualDrag());
        scalar dK = 0, dKres = 0, kMax = 0, kresMax = 0;
        for (label i = 0; i < nC; ++i)
        {
            dK = max(dK, mag(coef.K[i] - clotDrag_[i]));
            dKres = max(dKres, mag(coef.Kres[i] - KresNow[i]));
            kMax = max(kMax, mag(clotDrag_[i]));
            kresMax = max(kresMax, mag(KresNow[i]));
        }
        const scalar kMatch = kMax > 0 ? dK/kMax : dK;
        const scalar kresMatch = kresMax > 0 ? dKres/kresMax : dKres;
        const scalar linRes = st.bNorm > 0 ? st.rNorm/st.bNorm : st.rNorm;
        // S19a section 2.5: the row-sum-scaled residual, guarded as linRes.
        const scalar linResScaled =
            st.bsNorm > 0 ? st.rsNorm/st.bsNorm : st.rsNorm;

        // bodySlipRms: reportFringe()'s body population (ind >= 0.9), the
        // same weight ind*V.
        scalar bW = 0, bS = 0;
        for (label i = 0; i < nC; ++i)
        {
            if (indBI[i] >= 0.9)
            {
                const scalar w = indBI[i]*mesh.V()[i];
                bW += w;
                bS += w*magSqr(Vs_[i] - U[i]);
            }
        }
        const scalar bodySlipRms = bW > 0 ? sqrt(bS/bW) : 0;

        // S19b section 2.7 (F5, for E1): max |.| over the cells of U, Vs and
        // p as this step leaves them, at 17 significant digits.
        scalar uMaxB = 0, vsMaxB = 0, pMaxB = 0;
        for (label i = 0; i < nC; ++i)
        {
            uMaxB = max(uMaxB, mag(U[i]));
            vsMaxB = max(vsMaxB, mag(Vs_[i]));
            pMaxB = max(pMaxB, mag(p[i]));
        }
        std::ostringstream maxesB;
        maxesB.precision(17);
        maxesB << " uMax=" << uMaxB << " vsMax=" << vsMaxB
               << " pMax=" << pMaxB;

        // S22 section 2.7 (red team 03 F6): the process's peak resident set,
        // getrusage ru_maxrss (KB on Linux) / 1e6.
        struct rusage usageB;
        if (getrusage(RUSAGE_SELF, &usageB) != 0)
        {
            FatalErrorInFunction
                << "S22: getrusage(RUSAGE_SELF) failed; the record's rssGB "
                << "cannot be read (S22 section 2.7)." << exit(FatalError);
        }
        const scalar rssGB = scalar(usageB.ru_maxrss)/1e6;

        Info<< "PHASE_CLOT_S19 time=" << runTime.userTimeValue()
            << " step=" << runTime.timeIndex()
            << " mode=blockCoupled"
            << " linearSolver=" << blockLinearSolver_
            << " tSolve=" << st.tSolve
            << " tNumeric=" << st.tNum
            << " tSymbolic=" << st.tSym
            << " symbolicCount="
            << label
               (
                   pardisoB
                 ? blockPardiso_->symbolicCount()
                 : blockLU_->symbolicCount()
               )
            << " backSubstitutions=" << st.nBack
            << " linRes=" << linRes
            << " linResScaled=" << linResScaled
            << " rcond=" << st.rcond
            // S19a section 2.4: the matrix and its factor.
            << " n=" << st.n
            << " nnz=" << st.nnz
            << " strategyUsed=" << st.strategyUsed
            << " orderingUsed=" << st.orderingUsed
            << " lunz=" << st.lunz
            << " numericPeakGB=" << st.numericPeakGB
            << " noffdiag=" << st.noffdiag
            // S22 section 2.7: -1 where the backend has no such figure.
            << " perturbedPivots=" << st.perturbedPivots
            << " irSteps=" << st.irSteps
            << " solverThreads="
            << (
                   pardisoB
                 ? label(blockPardiso_->threads())
                 : blockSolverThreads_
               )
            << " mklPeakGB=" << st.mklPeakGB
            << " rssGB=" << rssGB
            << " rowSumMin=" << st.rowSumMin
            << " rowSumMax=" << st.rowSumMax
            << " contResAbs=" << contResAbs
            << " kMatch=" << kMatch
            << " kresMatch=" << kresMatch
            << " bodySlipRms=" << bodySlipRms
            << " qInlet=" << qInlet
            << " qOutlet=" << qOutlet
            << " lumenIdentityRows=" << st.nIdentity
            // S19b section 2.7: one of the two counts is 0 (lumen cells).
            << " lumenEliminated=" << st.nEliminated
            << maxesB.str().c_str();
        if (coef.r3)
        {
            std::vector<blockCoupled::LsqRow> lsq;
            blockCoupled::lsqRows(bm, lsq);
            std::vector<double> vx(nC), vy(nC), fx, fy;
            for (label i = 0; i < nC; ++i)
            {
                vx[i] = Vs_[i].x();
                vy[i] = Vs_[i].y();
            }
            blockCoupled::r3Force(bm, coef, lsq, vx, vy, fx, fy);
            scalar r3Power = 0, r3ForceMax = 0, dragMax = 0;
            for (label i = 0; i < nC; ++i)
            {
                // Fix round 12 (physics 11 row 5h): the lumen identity rows
                // never assemble R3, so they enter neither r3Power nor r3Max.
                if (!coef.isSkel[i])
                {
                    continue;
                }
                r3Power += vx[i]*fx[i] + vy[i]*fy[i];
                r3ForceMax =
                    max(r3ForceMax, sqrt(fx[i]*fx[i] + fy[i]*fy[i])/mesh.V()[i]);
                dragMax = max(dragMax, KdIn[i]*mag(U[i] - Vs_[i]));
            }
            Info<< " r3Power=" << r3Power
                << " r3Max=" << (dragMax > 0 ? r3ForceMax/dragMax : r3ForceMax)
                << " etaN="
                << gMax(GField_.primitiveField())*networkViscosityTau_
                  /skeletonVolumeFraction_;
        }
        Info<< endl;
    }

    // ---- RE-EMITTED (red team 05 #6): the G8 watch survives ----
    // The shipped pair, re-emitted once per step (this-> keeps the S12
    // guard's pin on the corrector's own pair meaningful).
    reportSkeletonVelocity();
    this->reportFringe();
    {
        // PHASE_CLOT_VS_SPLIT's population and argmax as the shipped
        // corrector; the H/A split (HbyADs, elasticFace, pressureFace) does
        // not exist in one block solve and is not printed.
        const scalar residualContent = residualChi_*skeletonVolumeFraction_;
        const scalarField& refI = thetaNetRef_.primitiveField();
        const volScalarField ddtPhiSB("ddtPhiSB", fvc::ddt(phiSkel_));
        label w = -1;
        scalar worstSlip = -1;
        forAll(indBI, celli)
        {
            const bool network =
              refI[celli] > residualContent;
            if (network && indBI[celli] < 0.5)
            {
                const scalar s = mag(Vs_[celli] - U[celli]);
                if (s > worstSlip) { worstSlip = s; w = celli; }
            }
        }
        if (w >= 0)
        {
            Info<< "PHASE_CLOT_VS_SPLIT time=" << runTime.userTimeValue()
                << " x=" << mesh.C()[w].x() << " y=" << mesh.C()[w].y()
                << " ind=" << indBI[w]
                << " thetaNetRef=" << refI[w]
                << " slip=" << worstSlip
                << " Vs=" << mag(Vs_[w]) << " U=" << mag(U[w])
                << " Dblk=" << DblkB[w]
                << " As=" << DblkB[w] - KdB[w] - KresB[w]
                << " Kd=" << KdB[w] << " Kres=" << KresB[w]
                << " ddtPhiS=" << ddtPhiSB[w]
                << " momentumForm=" << "phaseContinuity"
                << " mode=blockCoupled"
                << endl;
        }
        else
        {
            Info<< "PHASE_CLOT_VS_SPLIT time=" << runTime.userTimeValue()
                << " noDiluteNetworkCell" << endl;
        }
    }

    // ---- S23 section 2.5: PHASE_CLOT_S23, ONE LINE PER STEP ----
    // The event side (counts, R before the A1 reset) comes from this step's
    // ruptureOverstretchedNetwork(); the rest is measured here on the Vs and
    // U this block solve leaves. Every quotient is guarded on its own
    // denominator (a zero is counted, not divided).
    {
        boolList tornL;
        scalarField sTornL, dsL;
        const label nTornL = tornNetworkCells(tornL, sTornL, &dsL);
        const scalar dtS = runTime.deltaTValue();
        const scalarField& KdS = KdB.primitiveField();
        const scalarField& KresS = KresB.primitiveField();
        const scalarField& refS = thetaNetRef_.primitiveField();
        const scalarField& pIS = thetaIntact_.primitiveField();
        const scalarField& pPS = thetaPending_.primitiveField();
        const scalarField& pBS = thetaBroken_.primitiveField();
        const scalarField& dS = ruptured_.primitiveField();
        const volScalarField aIS(intactShare());
        const volScalarField pnUS(networkPartialPressureUndamaged());
        const scalar phi0S = skeletonVolumeFraction_;
        const scalar diluteHi = 0.1;

        scalar maxUS = 0;
        scalar tornSlip = 0, tornVsOverU = 0, tornKresOverKd = 0;
        scalar tornAI = 0, tornPt = 0;
        label tornZeroU = 0, tornZeroKd = 0;
        // S23 rev. 4.1 (re-check 23 R3): G23-7's maximum is printed with its
        // cell, so a failing step names the cell; -1 when no T cell has Kd > 0.
        label tornKresCell = -1;
        label wS = -1;
        scalar wVs = -1;
        label diluteN = 0;
        scalar diluteSlip = 0, diluteVs = 0;
        for (label i = 0; i < nC; ++i)
        {
            const scalar magU = mag(U[i]);
            const scalar magVs = mag(Vs_[i]);
            const scalar slip = mag(Vs_[i] - U[i]);
            maxUS = max(maxUS, magU);
            if (tornL[i])
            {
                tornSlip = max(tornSlip, slip);
                if (magU > 0)
                {
                    tornVsOverU = max(tornVsOverU, magVs/magU);
                }
                else
                {
                    ++tornZeroU;
                }
                if (KdS[i] > 0)
                {
                    const scalar kresOverKd = KresS[i]/KdS[i];
                    if (kresOverKd > tornKresOverKd)
                    {
                        tornKresOverKd = kresOverKd;
                        tornKresCell = i;
                    }
                }
                else
                {
                    ++tornZeroKd;
                }
                // rev. 3's G23-5 F1 (aI = 0 and Pt = 0 in T). Under rev.
                // 4.1's majority T these are the largest intact share and
                // tension on T, not 0; F1' (f1W*, f1Pt*) is the identity.
                tornAI = max(tornAI, mag(aIS[i]));
                tornPt = max(tornPt, mag(aIS[i]*min(pnUS[i], scalar(0))));
            }
            if (isSkelB[i] > 0.5)
            {
                if (magVs > wVs)
                {
                    wVs = magVs;
                    wS = i;
                }
                // the dilute band, down to the isSkel edge (ind 1.111e-4)
                if (indBI[i] <= diluteHi)
                {
                    ++diluteN;
                    diluteSlip = max(diluteSlip, slip);
                    diluteVs = max(diluteVs, magVs);
                }
            }
        }

        // S23 rev. 4.1 section 13.2 item 6 (red team 21 change 3): the gate
        // populations on this step's END-OF-STEP partition (transport in
        // solvePhaseField(), the event in solveElasticStrain(), then this
        // block solve; with nOuterCorrectors 1 nothing writes the partition
        // after it in the step). Pi: isSkel cells within two point-neighbour
        // rings of a T cell (Chebyshev 2 on the quad mesh), T included;
        // outside Pi: every other isSkel cell; M, the minority-torn class:
        // isSkel AND d_p < d* < (1 + d_p)/2. Each: max |Vs|/maxU with its
        // cell and its slip |Vs - U|; -1 for all three when the population
        // is empty, and the ratio -1 when maxU is 0 (not divided).
        const scalar dpL = progressiveDamageCeiling();
        const scalar thrL = damagePartitionKernel::tornThreshold(dpL);
        const int piRings = 2;
        std::vector<char> seedL(nC, 0);
        for (label i = 0; i < nC; ++i)
        {
            seedL[i] = tornL[i] ? 1 : 0;
        }
        std::vector<int> ringL;
        damagePartitionKernel::pointRings
        (
            mesh.cellPoints(), mesh.pointCells(), seedL, piRings, ringL
        );
        label nPiL = 0;
        label piCellL = -1, outPiCellL = -1, mCellL = -1;
        scalar piVsL = -1, outPiVsL = -1, mVsL = -1;
        for (label i = 0; i < nC; ++i)
        {
            if (!(isSkelB[i] > 0.5))
            {
                continue;
            }
            const scalar magVs = mag(Vs_[i]);
            if (ringL[i] <= piRings)
            {
                ++nPiL;
                if (magVs > piVsL)
                {
                    piVsL = magVs;
                    piCellL = i;
                }
            }
            else if (magVs > outPiVsL)
            {
                outPiVsL = magVs;
                outPiCellL = i;
            }
            if (dsL[i] > dpL && dsL[i] < thrL && magVs > mVsL)
            {
                mVsL = magVs;
                mCellL = i;
            }
        }
        // F1' Pt: the field the face correction consumed this step, against
        // aI min(Wp, 0) recomputed here. Cells -1: T is not empty but the
        // correction did not run on this step's partition.
        scalar f1PtDevL = 0;
        label f1PtCellsL = 0;
        if (nTornL > 0)
        {
            if
            (
                s23PtUsedIndex_ == runTime.timeIndex()
             && s23PtUsed_.size() == nC
            )
            {
                for (label i = 0; i < nC; ++i)
                {
                    if (tornL[i])
                    {
                        ++f1PtCellsL;
                        f1PtDevL = max
                        (
                            f1PtDevL,
                            mag
                            (
                                s23PtUsed_[i]
                              - aIS[i]*min(pnUS[i], scalar(0))
                            )
                        );
                    }
                }
            }
            else
            {
                f1PtCellsL = -1;
            }
        }

        // THE WORST CELL (argmax |Vs| over isSkel): its per-term force, per
        // unit mass like FelB, each as a magnitude. packing/tension are the
        // S22 split of the pressure force, -reconstruct(snGrad(.)|S|)/rho;
        // devInterp/devOwn the interpolated and own divergence of the face
        // stress; fel the APPLIED force (after the torn-face correction);
        // drag/kres K|Vs - U|; r3 the R3 force/V.
        scalar wPack = 0, wTens = 0, wDevI = 0, wDevO = 0, wFel = 0;
        scalar wDrag = 0, wKres = 0, wR3 = 0, wWall = -1;
        scalar wR = -1, wDs = 0, wInd = 0, wD = 0;
        if (wS >= 0)
        {
            const volScalarField PpackS
            (
                "PpackS",
                max(pnUS, dimensionedScalar(pnUS.dimensions(), 0))
            );
            const volScalarField PtS
            (
                "PtS",
                aIS*min(pnUS, dimensionedScalar(pnUS.dimensions(), 0))
            );
            const volVectorField divSigmaS
            (
                "divSigmaS",
                fvc::surfaceIntegrateExtrapolate
                (
                    mesh.Sf() & skeletonFaceStress()
                )/rho_
            );
            wPack = mag
            (
                fvc::reconstruct(fvc::snGrad(PpackS)*mesh.magSf())()[wS]
            )/rho_.value();
            wTens = mag
            (
                fvc::reconstruct(fvc::snGrad(PtS)*mesh.magSf())()[wS]
            )/rho_.value();
            wDevI = mag
            (
                fvc::reconstruct(fvc::interpolate(divSigmaS) & mesh.Sf())()[wS]
            );
            wDevO = mag(divSigmaS[wS]);
            wFel = mag(FelB[wS]);
            const scalar slipW = mag(Vs_[wS] - U[wS]);
            wDrag = KdS[wS]*slipW;
            wKres = KresS[wS]*slipW;
            if (coef.r3)
            {
                std::vector<blockCoupled::LsqRow> lsqS;
                blockCoupled::lsqRows(bm, lsqS);
                std::vector<double> vxS(nC), vyS(nC), fxS, fyS;
                for (label i = 0; i < nC; ++i)
                {
                    vxS[i] = Vs_[i].x();
                    vyS[i] = Vs_[i].y();
                }
                blockCoupled::r3Force(bm, coef, lsqS, vxS, vyS, fxS, fyS);
                wR3 = sqrt(sqr(fxS[wS]) + sqr(fyS[wS]))/mesh.V()[wS];
            }
            forAll(mesh.boundary(), patchi)
            {
                if (!isA<wallFvPatch>(mesh.boundary()[patchi]))
                {
                    continue;
                }
                const vectorField& Cfw = mesh.boundary()[patchi].Cf();
                forAll(Cfw, facei)
                {
                    const scalar dist = mag(Cfw[facei] - mesh.C()[wS]);
                    wWall = wWall < 0 ? dist : min(wWall, dist);
                }
            }
            const scalar thetaW = thetaC_[wS] + thetaH_[wS];
            wR = thetaW > 0 ? refS[wS]/thetaW : -1.0;
            const scalar SW = pIS[wS] + (pPS[wS] + pBS[wS]);
            wDs = SW > 0 ? (pPS[wS] + pBS[wS])/SW : 0.0;
            wInd = thetaW/phi0S;
            wD = dS[wS];
        }

        Info<< "PHASE_CLOT_S23 time=" << runTime.userTimeValue()
            << " step=" << runTime.timeIndex()
            << " deltaT=" << dtS
            << " volumetricTearRatio=" << volumetricTearRatio_
            << " volEvents=" << s23VolEvents_
            << " stretchEvents=" << s23StretchEvents_
            << " volEventsTotal=" << s23VolEventsTotal_
            << " stretchEventsTotal=" << s23StretchEventsTotal_
            << " a1Deferred=" << s23A1Deferred_
            << " tornCells=" << nTornL
            << " tornFaces=" << s23TornFaces_
            << " RCells=" << s23RCells_
            << " RMax=" << s23RMax_
            << " RP50=" << s23RP50_
            << " RP90=" << s23RP90_
            << " RP99=" << s23RP99_
            << " tornMaxSlip=" << tornSlip
            << " tornMaxVsOverU=" << tornVsOverU
            << " tornZeroUCells=" << tornZeroU
            << " tornMaxKresOverKd=" << tornKresOverKd
            << " tornMaxKresOverKdCell=" << tornKresCell
            << " tornZeroKdCells=" << tornZeroKd
            << " tornMaxAI=" << tornAI
            << " tornMaxAbsPt=" << tornPt
            << " maxU=" << maxUS
            << " worstCell=" << wS
            << " worstX=" << (wS >= 0 ? mesh.C()[wS].x() : 0.0)
            << " worstY=" << (wS >= 0 ? mesh.C()[wS].y() : 0.0)
            << " worstVs=" << (wS >= 0 ? wVs : 0.0)
            << " worstVsOverMaxU="
            << (wS >= 0 && maxUS > 0 ? wVs/maxUS : -1.0)
            << " worstTorn=" << (wS >= 0 && tornL[wS] ? 1 : 0)
            << " worstR=" << wR
            << " worstInd=" << wInd
            << " worstDStar=" << wDs
            << " worstD=" << wD
            << " worstWallDist=" << wWall
            << " worstPacking=" << wPack
            << " worstTension=" << wTens
            << " worstDevInterp=" << wDevI
            << " worstDevOwn=" << wDevO
            << " worstFel=" << wFel
            << " worstDrag=" << wDrag
            << " worstKres=" << wKres
            << " worstR3=" << wR3
            << " diluteLow=" << 0.5*residualChi_
            << " diluteHigh=" << diluteHi
            << " diluteCells=" << diluteN
            << " diluteMaxSlip=" << diluteSlip
            << " diluteMaxVsOverMaxU="
            << (maxUS > 0 ? diluteVs/maxUS : -1.0)
            // through the previous step's PHASE_CLOT_CONTENT_BALANCE
            << " contentResidualIntOverV0="
            << (s23ContentV0_ > 0 ? s23ContentResInt_/s23ContentV0_ : -1.0)
            << " outfluxIntOverV0="
            << (s23ContentV0_ > 0 ? s23OutfluxInt_/s23ContentV0_ : -1.0)
            << " contentV0=" << s23ContentV0_
            // S23 rev. 4.1 section 13.2 item 6: the gate populations
            << " piCells=" << nPiL
            << " piMaxVsOverMaxU="
            << (piCellL >= 0 && maxUS > 0 ? piVsL/maxUS : -1.0)
            << " piMaxVsCell=" << piCellL
            << " piMaxVsSlip="
            << (piCellL >= 0 ? mag(Vs_[piCellL] - U[piCellL]) : -1.0)
            << " outPiMaxVsOverMaxU="
            << (outPiCellL >= 0 && maxUS > 0 ? outPiVsL/maxUS : -1.0)
            << " outPiMaxVsCell=" << outPiCellL
            << " outPiMaxVsSlip="
            << (outPiCellL >= 0 ? mag(Vs_[outPiCellL] - U[outPiCellL]) : -1.0)
            << " minorityMaxVsOverMaxU="
            << (mCellL >= 0 && maxUS > 0 ? mVsL/maxUS : -1.0)
            << " minorityMaxVsCell=" << mCellL
            << " minorityMaxVsSlip="
            << (mCellL >= 0 ? mag(Vs_[mCellL] - U[mCellL]) : -1.0)
            // F1' at both builders, since the previous line, and Pt
            << " f1WSite0Dev=" << s23F1WDev_[0]
            << " f1WSite0Cells=" << s23F1WCells_[0]
            << " f1WSite1Dev=" << s23F1WDev_[1]
            << " f1WSite1Cells=" << s23F1WCells_[1]
            << " f1WZeroOccCells=" << s23F1WZeroOcc_
            << " f1PtDev=" << f1PtDevL
            << " f1PtCells=" << f1PtCellsL
            << endl;
        // F1' accumulates per line: the next line reports the next step's
        // builder calls (and any made after this line in this step).
        s23F1WDev_[0] = 0;
        s23F1WDev_[1] = 0;
        s23F1WCells_[0] = -1;
        s23F1WCells_[1] = -1;
        s23F1WZeroOcc_ = 0;
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::correctPressure()
{
    // THE CONTROL ARM IS THE BASE CODE ITSELF, NOT A COPY OF IT.
    //
    // elasticForce_ is only valid when faceConsistentElasticForce_ is on AND
    // the Eulerian branch of momentumPredictor() ran, so every other child --
    // and the rigid v2 path, and the partitioned path -- reaches exactly the
    // implementation it reached before this function existed. Re-typing the
    // base as the control is how an A/B silently acquires a second variable.
    // THE COUPLED TWO-VELOCITY SOLVE TAKES PRECEDENCE, and it is selected by
    // the validity of tVsEqn -- set on exactly one branch of
    // momentumPredictor(), cleared everywhere else -- for the same reason
    // elasticForce_ selects the face-consistent path below: a switch read from
    // the dictionary would not know whether the matrix was actually built.
    if (tVsEqn.valid())
    {
        correctPressureTwoVelocity();
        return;
    }

    {
        incompressibleFluid::correctPressure();
        return;
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::momentumPredictor()
{
    // SAME CONTRACT AS elasticForce_. correctPressure() reads the validity of
    // tVsEqn as "this step built a skeleton momentum equation, use the coupled
    // solve". Only the two-velocity branch below sets it, so every other path
    // must leave it invalid rather than inherit the previous call's matrix.
    tVsEqn.clear();

    // ONCE PER STEP, BEFORE EITHER MATRIX IS ASSEMBLED. These carry oldTime
    // for the two ddt terms; see the note on their declaration for why they
    // cannot be temporaries. Refreshed unconditionally so the written field is
    // meaningful on every path, not only the two-velocity one.
    {
        // C1/C2. phiSf and phiFf at :9706-9707 are the interpolants of these
        // two fields and of nothing else, so assigning theta here migrates
        // the flux-decomposition weight AND the weight the platelet equation
        // stores against, in one place. Both are read at the same time level:
        // prePredictor() runs solvePhaseField() then solvePlatelets() -- which
        // is what writes thetaC_/thetaH_ -- before this function is called.
        //
        // THE CLAMP IS NOT A MODEL CHOICE. It is the invariant solidFraction()
        // supplies and theta does not. Nothing bounds thetaC_/thetaH_: they
        // are the solution of an upwind transport equation with a source, and
        // there is no bounding step anywhere. phiFluid_ = 1 - phiSkel_ is a
        // momentum-equation weight whose strict positivity the 2x2 block
        // relies on at :10128, and solidFraction()'s own note says a zero
        // there "would make the FLUID phase degenerate, which is the failure
        // this whole construction exists to avoid".
        //
        // The bounds are solidFraction()'s OWN 0 and 0.9, deliberately, so
        // that the basis is the only thing that changes between the two
        // settings. 0.9 is a legacy regularisation and not the content
        // model's limit (phiSMax_), but adopting phiSMax_ here would be a
        // second change in the same run. Neither bound binds in the cases on
        // record (theta in [1.0e-19, 4.510e-01]) -- and the slack is
        // MEASURED below rather than assumed, because a clamp that starts
        // binding is a C1 violation that would otherwise be silent.
        const volScalarField theta("theta", thetaC_ + thetaH_);

        phiSkel_ = min
        (
            max(theta, dimensionedScalar("zero", dimless, 0)),
            dimensionedScalar("cap", dimless, 0.9)
        );
        phiFluid_ = 1.0 - phiSkel_;

        // WHAT THE CLAMP ACTUALLY DID, and how far the two bases differ.
        // c1Max is the quantity the migration exists to remove: under
        // "content" it must fall to the clamp slack, and it is reported
        // rather than gated because it is a measurement, not a threshold.
        const scalar thetaLo = gMin(theta.primitiveField());
        const scalar thetaHi = gMax(theta.primitiveField());
        const volScalarField legacy(solidFraction());
        const scalar c1Max =
            gMax(mag(legacy.primitiveField() - phiSkel_.primitiveField()));
        label nLo = 0;
        label nHi = 0;
        forAll(theta, celli)
        {
            if (theta[celli] < 0)
            {
                ++nLo;
            }
            if (theta[celli] > 0.9)
            {
                ++nHi;
            }
        }
        reduce(nLo, sumOp<label>());
        reduce(nHi, sumOp<label>());

        Info<< "PHASE_CLOT_PHASE_WEIGHT_BASIS"
            << " t=" << runTime.userTimeValue()
            << " basis=content"
            << " thetaMin=" << thetaLo
            << " thetaMax=" << thetaHi
            << " clampedLow=" << nLo
            << " clampedHigh=" << nHi
            << " c1Max=" << c1Max
            << endl;
    }

    {
        // Poroelastic drag relative to the MOVING skeleton. This is the term
        // that couples deformation back into the flow, and it replaces the
        // numerical penalty with a measured permeability.
        //
        // THE COEFFICIENT HERE IS nu/k, NOT mu/k, AND THAT ASYMMETRY WITH
        // solveSkeleton() IS DELIBERATE. incompressibleFluid solves the
        // KINEMATIC momentum equation - every term already divided by rho, with
        // p_ the kinematic pressure and divDevSigma built on nu - so the Darcy
        // force per unit volume (mu/k)(U - Us) enters divided by rho as
        // (nu/k)(U - Us). Dimensionally: (m^2/s)/(m^2) = 1/s, which times U
        // gives m/s^2 and matches fvm::ddt(U_). solveSkeleton() by contrast
        // assembles a DIMENSIONAL force balance in Pa/m, so there the same
        // physical drag correctly carries mu = rho*nu.
        //
        // A `const dimensionedScalar mu(rho_*nu_);` used to sit on the next line
        // here, unused, directly above the line that uses nu_. It was removed
        // because it reads as an oversight rather than as the deliberate choice
        // it sits next to: "fixing" it would multiply this coefficient by
        // rho = 1050 and silently break the coupling in the direction no gate
        // currently checks. The rho/no-rho asymmetry between the two sides is
        // recorded in cases/AM2020_staggered_coupling_probe/README.md, which is
        // why density was rejected as a probe axis.
        // Clamped, like its twin in solveSkeleton(): an unclamped chi went
        // slightly negative wherever q overshot 1, which put a negative
        // contribution on the momentum diagonal.
        const volScalarField chi
        (
            max(1.0 - q_, dimensionedScalar("zero", dimless, 0))
        );
        // THE POROSITY FACTOR phi_f^2, AND WHY IT IS ONLY ON THE
        // TWO-VELOCITY PATH.
        //
        // Darcy relates the SUPERFICIAL flux to the pressure gradient:
        // q = -(k/nu)*grad(p). In a phase-weighted two-fluid system the fluid
        // unknown U is the INTERSTITIAL velocity and the superficial flux is
        // the phase flux phi_f*U. Quasi-static with the skeleton held gives
        // U = -phi_f*grad(p)/K, so q = phi_f*U = -phi_f^2*grad(p)/K, and
        // matching Darcy requires
        //
        //     K = nu * phi_f^2 / k
        //
        // At the default skeletonVolumeFraction = 0.01 this factor is 0.98 --
        // 2%, inside the noise on k. IT IS NOT SMALL IF THE FRACTION IS SWEPT,
        // which the plan explicitly asks for: at phi_s = 0.1 the uncorrected
        // coefficient is 1.24x too large, and at Du & Fogelson's platelet
        // maximum of 0.6 it is 6.25x too large. Without this factor, sweeping
        // the one parameter the formulation is most sensitive to would
        // silently break Darcy's law.
        //
        // The one-velocity paths keep the bare chi*nu/k they were verified
        // with, so every existing child still reproduces bit-for-bit. There is
        // no phi_f there to correct with -- a single-velocity Brinkman model
        // has no phase split.
        // NOT A ?: BETWEEN A tmp<> AND A FIELD. That construct has already
        // failed to compile in this file twice; it is built, then overwritten.
        volScalarField poroFactor
        (
            "poroFactor",
            0.0*phiFluid_ + dimensionedScalar("one", dimless, 1)
        );
        {
            poroFactor = sqr(phiFluid_);
        }
        // Floored to match the dynamic site above -- see the note there. The
        // floor is NOT multiplied by poroFactor: poroFactor is 0.98 or more
        // everywhere the floor binds (chi < 0.35), so weighting it would
        // change nothing measurable while making the two sites textually
        // different, which is how they drift.
        volScalarField dragCoeff
        (
            skeletonDiluteDragRate_ > 0
          ? max
            (
                chi*poroFactor*nu_
               /max
                (
                    poroPerm_,
                    dimensionedScalar("kf", dimArea, permeabilityFloor_)
                ),
                dimensionedScalar
                (
                    "k1", dimless/dimTime, skeletonDiluteDragRate_
                )*chi
            )()
          : (
                chi*poroFactor*nu_
               /max
                (
                    poroPerm_,
                    dimensionedScalar("kf", dimArea, permeabilityFloor_)
                )
            )()
        );
        // S12: THE NETWORK'S OWN DRAG, WEIGHTED BY ITS OCCUPANCY. Built, then
        // overwritten -- the pattern this site already uses. The legacy
        // expression above is Kozeny-Carman at the cell AVERAGE, so a fringe
        // cell reads as uniformly dilute network and K collapses as ~chi^3
        // under the k1 floor; the constructor has refused k1 > 0 here. See
        // intrinsicDragRate() for the weighting and its precedent. poroPerm_
        // is assigned from the same state so the written field is the
        // permeability this drag was built from.
        {
            dragCoeff = intrinsicDragRate();
            poroPerm_ = intrinsicPermeability();
        }
        {
            // ONE VELOCITY. The elastic stress enters the momentum equation
            // directly, and THE DARCY DRAG IS GONE -- not omitted by oversight
            // but because it has nothing left to act on. That term is
            // (mu/k)(U - Us_velocity), a force proportional to the RELATIVE
            // motion of fluid and skeleton, and a single-velocity formulation
            // has no relative motion by construction.
            //
            // THIS IS THE REAL PHYSICS COST OF THE REFORMULATION AND IT IS
            // NOT SMALL: Biot consolidation is gone. The two-field skeleton
            // could drain, this cannot, and it is a genuine loss of capability
            // traded for kinematics that survive large strain. It must be
            // reported as such and not quietly dropped -- see section 5 of
            // research/literature/AM.2020.0056-19/KINEMATICS_DECISION_2026-08-15.md.
            //
            // Keeping the drag with Us set to zero would be worse than
            // dropping it: it would pin the clot in the LAB FRAME, so a clot
            // being carried downstream -- the thing Dr Li asked to see --
            // would be held in place by a term that is meant to describe
            // seepage.
            //
            // Sign: incompressibleFluid writes the KINEMATIC equation, with
            // divDevSigma on the left representing -div(tau)/rho, so an added
            // stress divergence enters the left with a minus. sigma_e is in
            // Pa, its divergence Pa/m, and dividing by rho gives m/s^2, which
            // matches fvm::ddt(U_).
            // THE ELASTIC FORCE GOES IN ONE OF TWO PLACES, NEVER BOTH.
            //
            // Cell-centred inside UEqn.H() (the original, still the default),
            // or held aside in elasticForce_ for correctPressure() to apply in
            // face form. The reasoning, the eight-order measurement and the two
            // dead rival explanations are on faceConsistentElasticForce_ in the
            // header; the mechanics are at correctPressure().
            //
            // SIGN. incompressibleFluid writes the KINEMATIC equation with
            // divDevSigma on the left representing -div(tau)/rho, so the stress
            // divergence enters the left with a minus and the equivalent body
            // force on the RIGHT is +fvc::div(sigma_e)/rho. That is what is
            // stored: an acceleration in m/s^2, the same kind of quantity as
            // -fvc::grad(p), which is how correctPressure() treats it.
            {
                // ---- TWO MOMENTUM EQUATIONS, BOTH SOLVED ----
                //
                // THE ELASTIC FORCE IS NOT THE FLUID'S. div(sigma_e) belongs
                // to the SKELETON balance. Leaving it in the fluid equation as
                // well would count it twice: applying the old two-velocity
                // closure to one-velocity fields -- where the elastic force IS
                // in the fluid equation -- returned |Vs|/|U| ~ 1 + chi ~ 1.87,
                // measured offline at 1.68. That factor IS the double count.
                //
                // It reaches the fluid anyway, and correctly, through the
                // off-diagonal of the 2x2 block in correctPressure(): the
                // skeleton's stress pushes the fluid VIA THE DRAG, which is
                // the only channel physics gives it.


                // PHASE-WEIGHTED, AND ONLY ON THIS BRANCH. phi_f and phi_s sum
                // to one exactly, which is what makes grad(p) cancel when the
                // two momentum equations are added -- i.e. what makes the
                // mixture balance div(sigma_e) = grad(p) hold by construction
                // instead of by luck. Every other branch keeps the unweighted
                // form, so all existing children reproduce exactly.
                //
                // The weighting is the CONSERVATIVE (phase-weighted) form,
                // ddt(phi_f*U), not the phase-intensive phi_f*ddt(U). Ghione
                // (source 51) records that the intensive form divides by a
                // vanishing phase fraction and is being abandoned in OpenFOAM
                // for exactly that reason.
                // THE VARIABLE NAME AND THE FIELD NAME MUST MATCH.
                // check_scheme_coverage.py reads the C++ ARGUMENT name for the
                // two-argument fvm::div form, while OpenFOAM looks up
                // "div(" + flux.name() + "," + field.name() + ")" at runtime.
                // Naming the variable phiFf and the field phiFluidFlux would
                // make the checker demand a scheme that the solver never asks
                // for, and stay silent about the one it does -- a guard
                // reporting on a name nothing uses.
                const surfaceScalarField phiFluidFlux
                (
                    "phiFluidFlux", fvc::interpolate(phiFluid_)*phiFluidOwn_
                );

                tUEqn =
                (
                    fvm::ddt(phiFluid_, U_)
                  + fvm::div(phiFluidFlux, U_, "div(phiFluidFlux,U)")
                  + MRF.DDt(U_)
                  // phi_f*div(tau), NOT div(phi_f*tau), and the difference
                  // is tau*grad(phi_f). Measured against this mesh: grad(phi_f)
                  // across the interface is at most 0.01/(4*1e-6) = 2.5e+03
                  // 1/m while div(tau)/tau is ~1e+06 1/m, so the omitted term
                  // is 0.25% of the one kept, and only in the interface.
                  //
                  // Taken deliberately, because scaling the WHOLE matrix keeps
                  // the diagonal consistent with the ddt term -- and that
                  // diagonal is A_f in the 2x2 block, where an inconsistency
                  // would not stay 0.25%.
                  + phiFluid_*momentumTransport->divDevSigma(U_)
                 ==
                    fvModels().source(U_)
                );

                // S11 FIX 1, fluid row: the same phase momentum form, for
                // the same reason multiphaseEuler applies it to EVERY phase.
                // phiFluid >= 0.1 here, so this row never degenerates; it
                // is applied so that both rows of the block are one form.
                {
                    tUEqn.ref() += fvm::SuSp(-fvc::ddt(phiFluid_), U_);
                }

                // ---- THE SKELETON MOMENTUM EQUATION ----
                //
                // ddt(phi_s*Vs) + div(phi_s_flux, Vs)
                //     = -phi_s*grad(p) + div(sigma_e)/rho + K*(U - Vs)
                //
                // grad(p) and the drag are BOTH absent here. The pressure is
                // applied in correctPressure() from the shared mixture
                // equation, and the drag is the off-diagonal of the 2x2 block
                // -- added there, not here, exactly as OpenFOAM's
                // multiphaseEuler does it (momentumTransferSystem.C:618-622
                // adds Kd to the diagonal and -Kd off it, on top of
                // UEqns[i].A()). Putting the drag in this matrix as well would
                // count it twice and destroy the block.
                //
                // div(sigma_e) is likewise NOT here: it is carried in face
                // form through correctPressure() so the pressure Laplacian can
                // cancel it face by face, which is the treatment the
                // single-velocity path already uses (phiSigmaE) and the
                // Rhie-Chow-like cure Ghione prescribes for forces that scale
                // with a phase-fraction gradient.
                //
                // NO INVENTED VISCOSITY. The skeleton is elastic, not
                // viscous; its deviatoric response is sigma_e. A viscosity
                // here would be a constitutive parameter nobody measured.
                //
                // What IS added below is both-sides diffusion, which is not
                // one: the same operator is subtracted explicitly, so the
                // converged solution is unchanged to machine precision and
                // there is no new parameter. See the block after the matrix.
                // phiSkelOwn_ is the SOLVED skeleton flux from the pressure
                // corrector, not fvc::flux(Vs_). The reconstructed flux does
                // not satisfy the discrete continuity the pressure equation
                // just imposed, so convecting skeleton momentum with it would
                // put a spurious source in the clot.
                //
                // WHICH FACE VALUE OF phiSkel_ -- OI-6, and it is a matching
                // condition, not a scheme preference. The theta equation
                // transports with contentFaceWeights_; re-interpolating the
                // same field linearly here advects skeleton momentum with a
                // flux the solid volume never moved at. Measured on run
                // 20260919-083453 at t = 4e-04: max|linear - upwind| =
                // 2.8071e-02 in volume fraction, 1.61% flux-weighted, 26.5%
                // of solid-carrying faces past 20% relative -- 4.0x the C1
                // violation the weight migration exists to remove.
                // research/designs/2026-09-19-OI6-correctness-criterion.md.
                if
                (
                    !contentFaceWeights_.valid()
                )
                {
                    FatalErrorInFunction
                        << "phaseMomentumFluxBasis 'shared' but"
                        << " contentFaceWeights_ is absent: the skeleton"
                        << " momentum flux was built before"
                        << " solveContentComponents() ran this step. Advecting"
                        << " with the previous step's weights is the defect"
                        << " this key repairs, so it is refused rather than"
                        << " silently substituted."
                        << exit(FatalError);
                }

                const surfaceScalarField thetaFaceMom
                (
                    "thetaFaceMom",
                    suppliedWeightsScheme(mesh_, contentTransportedWeights()())
                        .interpolate(phiSkel_)
                );

                const surfaceScalarField phiSkelFlux
                (
                    "phiSkelFlux", thetaFaceMom*phiSkelOwn_
                );

                // PHASE_CLOT_MMC -- the mass-momentum consistency residual,
                // measured, not asserted.
                //
                // theta_f^mass is the UNCLAMPED theta interpolated with the
                // weights its own equation used. Under 'shared' the residual
                // is therefore exactly the clamp slack in phiSkel_, which is
                // zero in every case on record (clampedLow/clampedHigh = 0 in
                // 272/272 records of run 20260919-083453) -- so this reads
                // 0.0 when the repair holds and becomes nonzero the moment
                // the clamp starts binding, which is a real event worth
                // seeing rather than a tolerance to widen.
                //
                // WHAT A FAILING RUN LOOKS LIKE: mmcMax > 0 under 'shared'
                // with clampedLow = clampedHigh = 0. That means the two face
                // values stopped being the same field -- someone
                // re-interpolated.
                //
                // THE NEGATIVE TEST IS THE 'legacy' BRANCH, AND IT IS ALREADY
                // MEASURED: under 'legacy' this must report mmcMax ~ 2.8e-02
                // on the main case. A diagnostic that reads zero there is
                // broken, not clean.
                if (contentFaceWeights_.valid())
                {
                    const volScalarField thetaUnclamped
                    (
                        "thetaUnclamped", thetaC_ + thetaH_
                    );
                    const surfaceScalarField thetaFaceMass
                    (
                        suppliedWeightsScheme
                        (
                            mesh_, contentTransportedWeights()()
                        ).interpolate(thetaUnclamped)
                    );
                    const surfaceScalarField mmc
                    (
                        mag(thetaFaceMom - thetaFaceMass)
                    );

                    const scalar mmcMax = gMax(mmc.primitiveField());
                    const scalar massSum =
                        gSum(mag(thetaFaceMass.primitiveField()));
                    const scalar mmcMean =
                        massSum > vSmall
                      ? gSum(mmc.primitiveField())/massSum
                      : 0.0;

                    Info<< "PHASE_CLOT_MMC"
                        << " t=" << runTime.userTimeValue()
                        << " basis=shared"
                        << " mmcMax=" << mmcMax
                        << " mmcMeanRel=" << mmcMean
                        << endl;
                }

                tVsEqn =
                (
                    fvm::ddt(phiSkel_, Vs_)
                  + fvm::div(phiSkelFlux, Vs_, "div(phiSkelFlux,Vs)")
                );

                // S11 FIX 1: THE PHASE MOMENTUM FORM OF multiphaseEuler.
                //
                // MovingPhaseModel::UEqn is ddt(alpha,rho,U) + div(aRhoPhi,U)
                // + SuSp(-continuityError, U), continuityError = ddt(alpha
                // rho) + div(aRhoPhi). The BOUNDED div scheme (checked at
                // construction) already removes the div half; this removes
                // the ddt half. Together:
                //
                //   phi_old*(Vs - Vs_old)/dt + sum_in F*(Vs - Vs_N) = forces
                //
                // -- a CONVEX update: with the forces off, Vs is a weighted
                // mean of its own old value and its upwind neighbours, and
                // cannot exceed them however small phi becomes. Without it
                // a draining cell keeps the momentum of the mass that left,
                // Vs_new ~ (phi_old/phi_new)*Vs_old per step; run
                // 20260923-031942's void cell followed phi0/phi to 3-17%
                // for 12 steps before it ran away. Growth enters theta
                // without a flux, so it is part of the error removed here:
                // deposited content takes the local skeleton velocity.
                {
                    tVsEqn.ref() += fvm::SuSp(-fvc::ddt(phiSkel_), Vs_);
                }

                // BOTH-SIDES DIFFUSION, BECAUSE THE ELASTIC STRESS IS
                // EXPLICIT AND ITS WAVE SPEED IS NOT WHAT ANYONE EXPECTS.
                //
                // sigma_e reaches this equation only through correctPressure()
                // in face form, evaluated at the old time. An explicit elastic
                // stress carries a shear-wave CFL,
                //
                //     dt < dx / sqrt(G / (rho * phi_s))
                //
                // and phi_s = 0.01 makes the skeleton's effective inertia 100x
                // smaller than the fluid's, so the wave runs 10x faster than
                // the naive estimate. MEASURED on run 20260905-054441:
                //
                //     shell  G = 4 Pa   c = 0.62 m/s   dt* = 1.16e-06 s
                //     core   G = 39 Pa  c = 1.93 m/s   dt* = 3.71e-07 s
                //
                // against a realised dt of 1.52e-06 -- 1.31x over in the shell
                // and 4.10x over in the core. Predicted growth 1.31 per step,
                // observed 1.19. The Courant limiter cannot see this: deltaT
                // took six distinct values all run, set by adjustableRunTime
                // subdividing the write interval, and the fluid Courant was
                // 0.045 against maxCo 0.5. It never bound.
                //
                // Respecting the limit means dt ~ 1.9e-07 s: 8x the steps,
                // about 45 h instead of 5.6 h. Both-sides diffusion removes it
                // instead. Adding Gamma*laplacian(Vs) implicitly and
                // subtracting the identical operator explicitly leaves the
                // converged equation untouched -- the two cancel term by term
                // once the outer iteration converges -- while the implicit
                // half puts the elastic response on the diagonal, which is
                // what the stability limit was asking for. Standard
                // Guenette-Fortin/EVSS practice for exactly this problem.
                //
                // Gamma = phi_s * gD * G * dt / rho is DERIVED, not chosen: it
                // is the elastic stress linearised in the strain rate over one
                // step, which is the coefficient the implicit term must carry
                // to represent the response it is stabilising. Same phi_s and
                // same gD as elasticStress(), so degraded material is
                // stabilised as weakly as it is stiff. Units:
                // [-][-][Pa][s]/[kg/m^3] = m^2/s, and laplacian(m^2/s, m/s)
                // is m/s^2, matching div(sigma_e)/rho.
                //
                // IF THE TWO HALVES EVER DISAGREE the run is not converged,
                // not stabilised -- which is why the residual of the pair is
                // reported rather than assumed. See PHASE_CLOT_SKEL_BSD.
                {
                    const volScalarField gD(sqr(1.0 - Dg_) + etaRes_);
                    // S14: the stiffness being stabilised is the INTACT
                    // network's; ruptured debris has none. intactFraction()
                    // is exactly 1 with the rupture model off.
                    const volScalarField phiSIntact
                    (
                        "phiSIntact", phiSkel_*intactFraction()
                    );
                    volScalarField GammaBSD
                    (
                        "GammaBSD",
                        phiSIntact*gD*GField_*runTime.deltaT()/rho_
                    );

                    // SIGNS. fvMatrix::operator-=(fvm) puts -laplacian on
                    // the LHS, i.e. +Gamma*lap(Vs) on the RHS -- the implicit
                    // diffusion. operator+=(fvc field) puts +laplacian on the
                    // LHS, i.e. -Gamma*lap(Vs) on the RHS -- the explicit
                    // subtraction that cancels it. -= on BOTH would leave
                    // +2*Gamma*lap(Vs) standing at convergence: a permanent
                    // doubled viscosity, which is precisely the invented
                    // constitutive parameter the comment above refuses.
                    //
                    // S11 FIX 3: under stressFaceTransmission series the face
                    // coefficient is the SERIES value of phi_s times the
                    // interpolated gD*G -- the stiffness the face actually
                    // transmits. Interpolated linearly it coupled a void to
                    // its dense neighbour at half the neighbour's stiffness,
                    // the same defect as the linear face stress.
                    surfaceScalarField GammaBSDf
                    (
                        "GammaBSD", fvc::interpolate(GammaBSD)
                    );
                    {
                        GammaBSDf =
                            seriesFaceWeights(phiSIntact)
                           *fvc::interpolate(gD*GField_)
                           *runTime.deltaT()/rho_;
                    }
                    // S17 C (design 2C; docket 10): THE TANGENT, NOT THE
                    // MODULUS, and the stress's own weight. Gamma_t,f =
                    // series(w)_f interp(gD G k_eff)_f dt/rho, w =
                    // max(ind, floor)(1 - d) and k_eff =
                    // [(thI/S) stiff + (thP/S)(lf^2 - 1/lf)]/(1 - d): the
                    // linearisation of the shipped face stress
                    // series(w) interp(gD G M). The explicit half below uses
                    // the same Gamma_t. >= 0 by construction with be SPD
                    // (refused fatally otherwise). Series is required.
                    {
                        const volScalarField wT(skeletonStressWeight());
                        const volScalarField kEff(mixedTangentStiffness());
                        GammaBSDf =
                            seriesFaceWeights(wT)
                           *fvc::interpolate(gD*GField_*kEff)
                           *runTime.deltaT()/rho_;
                        GammaBSD = wT*gD*GField_*kEff*runTime.deltaT()/rho_;

                        // S25 F3 (design section 3A; rev. 2.1 amendment 11,
                        // V-A 13 finding 4): r = k_true/k_used over the cells
                        // where Gamma_t acts (> 0). k_used is kEff above, the
                        // very field Gamma_t was built from
                        // (mixedTangentStiffness: the pending share and the
                        // damage in). k_true mixes the law's true tangent k_c
                        // (intactWaveTangent) with the SAME shares, by the
                        // same kernels on the same arguments as
                        // mixedTangentStiffness (sharesCarrier,
                        // mixedStiffness). A wrong stiff in Gamma_t moves
                        // k_used and not k_true, so r sees it; build 2's
                        // k_c/intactTangentStiffness could not. The pair is
                        // neutrally stable while r s/(1 + s) <= 4 (numerics
                        // 07 finding 6); F3 FAILS on max r > 4. A cell whose
                        // k_used is not > 0 is counted (stiffNonPositive),
                        // not divided by.
                        {
                            const scalarField& gamF3 =
                                GammaBSD.primitiveField();
                            const scalarField& kUsedF3 = kEff.primitiveField();
                            const symmTensorField& beF3 = be_.primitiveField();
                            const scalarField& pIF3 =
                                thetaIntact_.primitiveField();
                            const scalarField& pPF3 =
                                thetaPending_.primitiveField();
                            const scalarField& pCF3 =
                                fibreCarrier().primitiveField();
                            const scalarField& pBF3 =
                                thetaBroken_.primitiveField();
                            const scalarField& lIF3 =
                                pendingStretch_.primitiveField();
                            label cellsF3 = 0, stiffBadF3 = 0, cellF3 = -1;
                            scalar rMaxF3 = -1, kTrueF3 = 0, kUsedCellF3 = 0;
                            forAll(gamF3, celli)
                            {
                                if (!(gamF3[celli] > 0))
                                {
                                    continue;
                                }
                                ++cellsF3;
                                const scalar kU = kUsedF3[celli];
                                if (!(kU > 0))
                                {
                                    ++stiffBadF3;
                                    continue;
                                }
                                scalar aIF3, aPF3, omdF3;
                                fibreReleaseKernel::sharesCarrier
                                (
                                    pIF3[celli], pPF3[celli], pBF3[celli],
                                    pCF3[celli], aIF3, aPF3, omdF3
                                );
                                const scalar kT =
                                    fibreReleaseKernel::mixedStiffness
                                    (
                                        aIF3, intactWaveTangent(beF3[celli]),
                                        aPF3, lIF3[celli], omdF3
                                    );
                                if (kT/kU > rMaxF3)
                                {
                                    rMaxF3 = kT/kU;
                                    kTrueF3 = kT;
                                    kUsedCellF3 = kU;
                                    cellF3 = celli;
                                }
                            }
                            reduce(cellsF3, sumOp<label>());
                            reduce(stiffBadF3, sumOp<label>());
                            reduce(rMaxF3, maxOp<scalar>());
                            Info<< "PHASE_CLOT_S25_F3"
                                << " time=" << runTime.userTimeValue()
                                << " cells=" << cellsF3
                                << " stiffNonPositive=" << stiffBadF3;
                            if (rMaxF3 >= 0)
                            {
                                Info<< " rMax=" << rMaxF3
                                    << " rCell=" << cellF3
                                    << " kc=" << kTrueF3
                                    << " stiffGamma=" << kUsedCellF3;
                            }
                            else
                            {
                                Info<< " rMax=none";
                            }
                            Info<< endl;
                        }
                    }
                    {
                        tVsEqn.ref() -= fvm::laplacian(GammaBSDf, Vs_);
                        tVsEqn.ref() += fvc::laplacian(GammaBSDf, Vs_)();
                    }

                    // The pair's imbalance, as a fraction of the implicit
                    // half. Explicit at the current Vs_, so at convergence
                    // this is round-off; a value that does NOT fall is the
                    // outer iteration failing, and it must be visible rather
                    // than silently absorbed into the stabilisation.
                    const volVectorField bsdExp
                    (
                        fvc::laplacian(GammaBSDf, Vs_)
                    );
                    const scalar bsdMag = gMax(mag(bsdExp.primitiveField()));
                    const scalar gammaMax = gMax(GammaBSD.primitiveField());
                    Info<< "PHASE_CLOT_SKEL_BSD"
                        << " time=" << runTime.name()
                        << " GammaMax=" << gammaMax
                        << " explicitMag=" << bsdMag
                        << nl;
                }

                // THE ANCHOR ACTS ON THE SKELETON, NOT THE FLUID.
                //
                // On the one-velocity path the bond hold had to be applied to
                // U because U was the only velocity there was -- it resisted
                // FLOW through a layer a bond was holding, which is an
                // approximation to holding the MATERIAL. Here the material has
                // its own velocity and the bond can hold the thing it actually
                // holds. This is a capability the two-velocity path adds, not
                // a relocation of an equivalent term.
                if (bondHoldPenalty_ > 0 && bondHoldDilation_ >= 0)
                {
                    const volScalarField holdChi
                    (
                        "holdChi",
                        bondHold_
                       *phaseIndicator()
                    );
                    tVsEqn.ref() +=
                        fvm::Sp
                        (
                            dimensionedScalar
                            (
                                "bondHoldPenalty", dimless/dimTime,
                                bondHoldPenalty_
                            )*holdChi,
                            Vs_
                        );
                }

                tVsEqn.ref().relax();
                fvConstraints().constrain(tVsEqn.ref());
            }

            // THE FLUID SIDE OF THE ANCHOR.
            //
            // fvm::Sp(bondHoldPenalty*bondHold*chi, U) resists flow through
            // the layer the bond is holding. This is the term whose ABSENCE
            // the momentum-balance diagnostic below reports as
            // anchorsInUEqn=0, and it is added here rather than as a general
            // lambda(q) penalty so that it exists ONLY where a bond exists and
            // vanishes the instant that bond fails -- the objection that
            // removed the Darcy drag from this path was that it pinned the
            // clot in the lab frame everywhere, and a bond-gated term does
            // not.
            //
            // IT IS NOT SUFFICIENT ON ITS OWN and must not be read as the
            // repair. Once the release mask covers the body the phase field is
            // carried at Ubody, independent of U, so this term can slow the
            // fluid to a standstill and the clot will still translate.
            // bondHoldCarrier is the term that reaches c. Both are separate
            // switches so the A/B says which does what instead of one change
            // being credited with the other's effect.
            //
            // The penalty is proportional to the local phase INDICATOR --
            // NOT the solid fraction, which is theta_ref times smaller -- so
            // it dies smoothly across the interface rather than at a
            // threshold. Follows phaseIndicatorBasis_: max(1-q,0) on
            // "legacy", theta/theta_ref on "content".
            if (bondHoldPenalty_ > 0 && bondHoldDilation_ >= 0)
            {
                const volScalarField holdChi
                (
                    "holdChi",
                    bondHold_*phaseIndicator()
                );
                tUEqn.ref() +=
                    fvm::Sp
                    (
                        dimensionedScalar
                        (
                            "bondHoldPenalty", dimless/dimTime,
                            bondHoldPenalty_
                        )*holdChi,
                        U_
                    );
            }

            // ---- THE BRINKMAN DRAG, IN RELATIVE-VELOCITY FORM ----
            //
            // WHAT THIS REPLACED, AND WHY. Until 2026-09-04 this term was
            // fvm::Sp(dragCoeff*(1 - released_), U): the LAB-FRAME form,
            // switched OFF wherever material was released. Xu (2008) eq.
            // (3.2) does write -(eta*phi3/k)u with no skeleton velocity in
            // it, and that is correct in their setting because their network
            // does not move -- the drag is what makes it not move. The gate
            // on released_ was added so that the term could not also pin an
            // embolus that ought to be leaving.
            //
            // THE COST OF THAT GATE WAS THE PHENOMENON. brinkmanHold = 0 on
            // released material means a detached embolus feels zero drag,
            // and a body that feels zero drag exerts none: it is
            // hydraulically invisible and cannot obstruct anything. Run
            // 20260903-114913 is the record -- the whole point of the
            // narrowing-channel study is a detached clot blocking a vessel,
            // and the formulation made "moves" and "blocks" mutually
            // exclusive. Dr Li's question could not be answered by a model
            // built this way, whatever the parameters.
            //
            // THE FIX IS THE TERM THE SKELETON PATH ALREADY USES. Look at
            // the else branch below: it assembles
            //
            //     fvm::Sp(dragCoeff, U_) - dragCoeff*skeletonVelocity
            //
            // with skeletonVelocity = (Us_ - UsPrev_)/dt. That is the proper
            // two-phase Darcy exchange, -(mu/k)(u_f - u_s): blood still has
            // to seep through the pores of a MOVING skeleton, so a moving
            // embolus still resists flow, and the term vanishes only when
            // the fluid travels WITH the material rather than whenever the
            // material is free. This branch now assembles the same thing.
            //
            // The skeleton velocity on this path is released_*UbodyCached_:
            // zero where the material is still attached, which is the
            // model's standing assumption for a mural clot and reproduces
            // the old lab-frame term exactly; UbodyCached_ where it is free,
            // which is the same vector solvePhaseField() carries it at.
            // Those two MUST be the same number, which is why it is cached
            // in prePredictor() rather than computed at each use.
            //
            // WHAT THIS DOES NOT MODEL, and it must not be claimed. Ubody is
            // a KINEMATIC estimator -- the ambient flow averaged around the
            // body -- not a force balance. The momentum this term takes out
            // of the fluid is therefore not handed to the body, so the pair
            // is not momentum-conserving; an external agent holds the body
            // at Ubody. accDragReaction below reports exactly that
            // imbalance so it is judged rather than assumed small. There is
            // also still no CONTACT mechanics: a body wedged in the throat
            // is held there because the ambient flow it is estimated from
            // stalls, not because it touches the wall.
            //
            // STABILITY. The implicit part is unchanged in structure and
            // sign: fvm::Sp with chi >= 0 by the clamp and poroPerm_ > 0 by
            // the floor, so it only ever adds to the diagonal. The new
            // explicit part is a source of size dragCoeff*Ubody, and the
            // pair is a relaxation of U towards Ubody at rate dragCoeff --
            // stable for any dt because the sink is implicit, and its fixed
            // point U = Ubody is the physically right one: fluid inside a
            // released body travels with it, which is what makes the body
            // displace fluid instead of leaking it.
            {
                // Clamped for the same reason chi is: released_ is built as
                // a 0/1 indicator, but it is a written field with
                // zeroGradient boundaries and nothing downstream should
                // depend on it never being nudged outside [0, 1].
                const volScalarField releasedHold
                (
                    "releasedHold",
                    min
                    (
                        max
                        (
                            released_,
                            dimensionedScalar("zero", dimless, 0)
                        ),
                        dimensionedScalar("one", dimless, 1)
                    )
                );

                // NOT A ?: BETWEEN A tmp<> AND A FIELD -- that construct has
                // already failed to compile in this file once and cost a VM
                // boot. Built, then overwritten.
                volVectorField skeletonVelocity
                (
                    "skeletonVelocity",
                    releasedHold
                   *dimensionedVector("Ubody", dimVelocity, UbodyCached_)
                );

                // THE DRAG IS NOT ASSEMBLED INTO EITHER MATRIX ON THE
                // TWO-VELOCITY PATH.
                //
                // It is the off-diagonal coupling of the 2x2 block that
                // correctPressure() inverts per cell, added there on top of
                // UEqn.A() and VsEqn.A() -- the same construction OpenFOAM
                // uses (momentumTransferSystem.C:618-622). Adding it here as
                // well would count it twice and leave the block singular in
                // the wrong direction.
                //
                // Why it MUST be the block and not fvm::Sp here: Rusche (2002)
                // section 3.2 pp. 107-108 reports that of the four standard
                // drag treatments, "only option 3 ensures convergence when
                // [the drag multiplier] is high or very non-uniform." Option 3
                // is partial elimination -- the block. Our clotDrag spans
                // 3.3e+05 in the shell to 3.3e+08 in the core across a diffuse
                // interface: high AND very non-uniform, both conditions at
                // once. Rusche himself uses the weaker option 2 only because
                // his own drag is mild.

                // PUBLISHED FROM THE ASSEMBLED EXPRESSION, NOT RECOMPUTED.
                // clotDrag_ is what writeBalanceDiagnostic() turns into
                // accDrag, and accDrag now enters accSum on this path. A
                // recomputation here could drift from the matrix and the
                // budget would close on a term the solver never saw --
                // the sigmaD_ defect class exactly.
                //
                // NOW THE BARE COEFFICIENT, not coefficient-times-gate. It
                // used to be dragCoeff*brinkmanHold, which was the whole
                // assembled coefficient because the term was
                // fvm::Sp(that, U). The term is now a PAIR -- an implicit
                // fvm::Sp(dragCoeff, U) and an explicit
                // -dragCoeff*skeletonVelocity -- so no single coefficient
                // field describes it any more. writeBalanceDiagnostic()
                // reconstructs the force as
                // -clotDrag_*(U - released_*UbodyCached_), which is exactly
                // the pair above, and reports the explicit half separately
                // as accDragReaction.
                clotDrag_ = dragCoeff;
            }
        }
    }
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();
    fvConstraints().constrain(UEqn);

    // S19: under blockCoupled the U and Vs predictors are part of the one
    // block solve in pressureCorrector() (design section 2).
    if (s19BlockCoupled())
    {
        return;
    }

    if (pimple.momentumPredictor())
    {
        if (tVsEqn.valid())
        {
            // THE FLUID CARRIES ONLY ITS SHARE OF THE PRESSURE on the
            // two-velocity path, because its momentum equation is weighted by
            // phi_f. Predicting it against the full grad(p) would over-push
            // the fluid by a factor 1/phi_f and hand correctPressure() a worse
            // starting point than no prediction at all.
            //
            // The skeleton is deliberately NOT predicted here. Its velocity
            // only becomes meaningful once the block couples it to U, and a
            // standalone predict would be the evaluated-not-solved mistake all
            // over again, one stage earlier.
            solve(UEqn == -phiFluid_*fvc::grad(p_));

            // S17 C (design 2C; docket 07 Q2, change 10): UNDER
            // implicitTangent THE SKELETON IS PREDICTED, because the pair now
            // carries the tangent and a pair that is only swept twice through
            // H/A is a pseudo-inertia (07: x2-47 slower modes), not an
            // implicit stiffness. The skeleton row of the block, solved with
            // the predicted U: VsEqn (with Gamma_t) + (K + Kres) Vs ==
            // (K + Kres) U + reconstruct(F_s) - phi_s reconstruct(snGrad(p)
            // |Sf|) (07's form; p is kinematic here, as in UEqn), Kres ON
            // THE DIAGONAL (red team 08: without it the lumen rows are a pure
            // Laplacian). An M-matrix, asymmetric (bounded upwind
            // convection): PBiCGStab or smoothSolver, refused PCG.
            {
                const volScalarField KdKres
                (
                    "KdKres", clotDrag_ + skeletonResidualDrag()
                );
                const surfaceScalarField fSigmaEVs
                (
                    skeletonElasticForceFlux()
                );
                // A named copy of the skeleton row: tVsEqn itself is
                // untouched, so the block reads the same A_s and H_s it did.
                fvVectorMatrix VsEqn(tVsEqn() + fvm::Sp(KdKres, Vs_));
                solve
                (
                    VsEqn
                 ==
                    KdKres*U_
                  + fvc::reconstruct(fSigmaEVs)
                  - phiSkel_*fvc::reconstruct(fvc::snGrad(p_)*mesh.magSf())
                );
                fvConstraints().constrain(Vs_);
            }
        }
        else
        {
            solve(UEqn == -fvc::grad(p_));
        }
        fvConstraints().constrain(U_);
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::writeBalanceDiagnostic()
{
    const scalar inventory = fvc::domainIntegrate(q_*w_).value();
    const scalar clotVolume =
        fvc::domainIntegrate(scalar(1) - q_).value();
    const scalar surfaceLoss =
        fvc::domainIntegrate(surfaceLossCoeff_*w_).value();

    // Signed outward platelet flux at the named open boundaries. The wedge,
    // axis and wall do not contribute to this canonical balance.
    const surfaceScalarField plateletFlux
    (
        fvc::interpolate(q_)*phi_*fvc::interpolate(w_)
      - fvc::interpolate(q_)*D_*fvc::snGrad(w_)*mesh.magSf()
    );

    scalar outerFlux = 0;
    const label inletPatch = mesh.boundaryMesh().findIndex("inlet");
    const label outletPatch = mesh.boundaryMesh().findIndex("outlet");
    scalar outletMinUx = 0;
    scalar outletMeanUx = 0;
    label outletCells = 0;
    if (inletPatch >= 0)
    {
        outerFlux += gSum(plateletFlux.boundaryField()[inletPatch]);
    }
    if (outletPatch >= 0)
    {
        outerFlux += gSum(plateletFlux.boundaryField()[outletPatch]);

        const vectorField outletInternalU
        (
            U_.boundaryField()[outletPatch].patchInternalField()
        );
        const scalarField outletInternalUx(outletInternalU.component(0));
        outletCells =
            returnReduce(outletInternalUx.size(), sumOp<label>());
        if (outletCells > 0)
        {
            outletMinUx = gMin(outletInternalUx);
            outletMeanUx = gAverage(outletInternalUx);
        }
    }

    const scalar deltaT = max(runTime.deltaTValue(), SMALL);
    const scalar balanceResidual =
        (inventory - previousInventory_)/deltaT + surfaceLoss + outerFlux;

    Info<< "PHASE_CLOT_DIAGNOSTIC"
        << " time=" << runTime.name()
        << " inventory=" << inventory
        << " surfaceLoss=" << surfaceLoss
        << " outerFlux=" << outerFlux
        << " balanceResidual=" << balanceResidual
        << " clotVolume=" << clotVolume
        << " outletMinUx=" << outletMinUx
        << " outletMeanUx=" << outletMeanUx
        << " outletCells=" << outletCells
        << nl;

    // ---- THE c-CURRENCY MASS BALANCE WAS RETIRED IN S6c-2 ----------------
    //
    // PHASE_CLOT_MASS closed the equation d(c)/dt + div(F c) = L - G. That
    // equation no longer exists: c is deleted, L = laplacian(mobility, muC)
    // is deleted with it, and G = Vn*phaseGradient went with
    // phaseGrowthDriver 'front'. A balance rebuilt on theta and reported
    // under the old name would be a different identity wearing it.
    //
    // WHAT REPLACED IT is contentResidual on PHASE_CLOT_CONTENT, which
    // closes the equation the solver actually integrates. The replacement is
    // not optional: PHASE_CLOT_MASS was this solver's ONLY closing mass
    // balance, and the reason it was added -- that clotVolume alone "could
    // not distinguish MATERIAL LEFT THE DOMAIN from MATERIAL WAS NUMERICALLY
    // DESTROYED" -- is unchanged by the currency it is measured in.
    //
    // clotVolume itself survives and is reported on PHASE_CLOT_DIAGNOSTIC:
    // it is domainIntegrate(1 - q_) and q_ is now the content form, so it
    // never depended on c.



    // THE FRONT'S INTEGRAL, WHICH AFTER S4 IS NOT THE GROWTH.
    //
    // KEPT THROUGH S6c-2, AND THE SCOPING TABLE SAID IT COULD GO. That table
    // gave phaseGradient_ exactly two consumers -- frontSource, under
    // phaseGrowthDriver 'front', and chiSource, under contentOperator
    // 'matched' -- and concluded that refusing both key values retired the
    // field. This is a third, it is unconditional, and it is live:
    // PHASE_CLOT_GROWTH_STOICH reads clotGrowthFront every step to test the
    // front closure, that Vn*phaseGradient converts platelets to clot at
    // 2*alpha/D. It is the 1.6277-against-1.6304 number every archived run
    // is compared on, and deleting phaseGradient_ would have retired it in
    // silence.
    const scalar clotGrowthFront =
        fvc::domainIntegrate(Vn_*phaseGradient_).value();

    // WHAT SOURCED THE CLOT THIS STEP. One branch left: phaseGrowthDriver
    // 'front' is refused at construction since S6c-2, its source having gone
    // with the c equation it sourced. clotGrowth survives because
    // PHASE_CLOT_GROWTH_STOICH divides by it; the c-currency mass balance
    // that used to consume it does not.
    const scalar clotGrowth = transferClotGrowth_;

    // ---- THE CONTENT BALANCE: WHAT REPLACED PHASE_CLOT_MASS --------------
    //
    // PHASE_CLOT_MASS closed d(c)/dt + div(F c) = L - G and was this
    // solver's ONLY closing mass balance. The reason it was added is
    // unchanged by the currency it is measured in: clotVolume alone "could
    // not distinguish MATERIAL LEFT THE DOMAIN from MATERIAL WAS NUMERICALLY
    // DESTROYED". This closes the equation the solver actually integrates.
    //
    //     d(theta)/dt + div(F theta) = S
    //
    // theta carries NO CONSTANT OFFSET, which is the one simplification the
    // change buys: psi = (1 - c)/2 did, and that is where the c-balance's
    // -div(F)/2 term came from. There is no such term here.
    //
    // THE OUTFLUX IS THE OPERATOR'S OWN, READ FROM ITS OWN MATRICES.
    // contentBoundaryFlux_ is booked in solveContentComponents() from the
    // thetaC and thetaH matrices' inlet and outlet coefficients, after their
    // solves and before the FCT: the content the transport actually removed.
    // Reusing phaseWeights_ here -- which is what the c balance correctly did
    // for ITS operator -- would measure the mismatch between two schemes
    // rather than the conservation error. That warning is inherited verbatim
    // from the balance this replaces.
    //
    // OI-59 (2026-10-08): THE RE-INTERPOLATION THIS REPLACED WAS NOT EXACT.
    // It interpolated thetaC + thetaH onto the inlet and outlet here, at the
    // end of the step. A non-coupled boundary face does return the patch
    // value under any scheme, but under the S10 FCT the patch value is no
    // longer the one the solve used: fctApply() corrects the outlet cells
    // through interior faces and re-evaluates the zeroGradient patch, so the
    // balance booked phi_b*theta_FCT,P against a transport that removed
    // phi_b*theta_low,P. In BATCH 72 S23v_G4_r3 that was all of G23-6's
    // 6.6e-5 residual.
    //
    // IT IS MEASURED OVER THE TIMESTEP, NOT OVER THE SOLVE, and that is why
    // it lives here rather than beside contentBefore/contentAfter in
    // solveContentComponents(). Two of the three source terms happen outside
    // that function: the accepted transfer is applied in
    // solvePlateletContent() afterwards, and releaseDetachedMaterial() can
    // delete content later still. A residual closed across the solve alone
    // would be blind to both.
    const scalar contentVolume =
        fvc::domainIntegrate(thetaC_ + thetaH_).value();

    // 0 when no content transport ran this step (zeroed with the carrier).
    const scalar contentOuterFlux = contentBoundaryFlux_;

    // transferClotGrowth_ is int dTheta dV / dt scaled by 2/theta_ref -- the
    // c-currency normalisation PHASE_CLOT_GROWTH_STOICH is defined in, kept
    // deliberately so that ratio stays comparable with every archived run.
    // Undone here, and only here, to get the content the transfer actually
    // deposited.
    const scalar contentTransferRate =
        0.5*skeletonVolumeFraction_*transferClotGrowth_;

    // A CONTENT DELETION IS A TERM, NOT AN ERROR. releaseDetachedMaterial()
    // zeroes thetaC/thetaH in fully damaged cells; booked here as a named
    // sink so the balance closes THROUGH the release rather than reporting
    // it as a conservation failure. Zero in every run to date -- DgMax is
    // seven decades below DgDetach -- which is exactly why it must be
    // written now rather than when it first fires.
    const scalar detachedContentRate = detachedContentStep_/deltaT;

    const scalar contentResidual =
        (contentVolume - previousContentVolume_)/deltaT
      + contentOuterFlux
      - contentTransferRate
      + detachedContentRate;

    // S23 G23-6: int |residual| dt and int outflux dt, over V0 = the first
    // contentVolume; PHASE_CLOT_S23 prints them.
    if (s23ContentV0_ < 0)
    {
        s23ContentV0_ = contentVolume;
    }
    s23ContentResInt_ += mag(contentResidual)*deltaT;
    s23OutfluxInt_ += contentOuterFlux*deltaT;

    Info<< "PHASE_CLOT_CONTENT_BALANCE"
        << " time=" << runTime.name()
        << " contentVolume=" << contentVolume
        << " contentDVdt="
        << ((contentVolume - previousContentVolume_)/deltaT)
        << " contentOuterFlux=" << contentOuterFlux
        << " contentTransferRate=" << contentTransferRate
        << " contentRemoved=" << detachedContentStep_
        << " detachedContentRate=" << detachedContentRate
        << " contentResidual=" << contentResidual
        << " thetaRef=" << skeletonVolumeFraction_
        << nl;

    previousContentVolume_ = contentVolume;
    detachedContentStep_ = 0;

    // ---- WHAT THE CLIP COST, IF IT IS ON ---------------------------------
    //
    // Reported unconditionally so that a run WITHOUT the clip still says so
    // in one grep-able line, rather than the reader having to prove a
    // negative from an absent diagnostic.
    //
    // Read it against clotVolume on the same line. The pre-registered
    // discriminator for run 20260906-004750's interior overshoot is
    // clipVolumeCum against 7.10e-18 m^3, the INSTANTANEOUS excess measured
    // there at t = 1.225e-03:
    //
    //   cum stays within a few x 7.1e-18  -> a saturating scheme overshoot.
    //                                        A bounded scheme on c is enough
    //                                        and costs nothing real.
    //   cum runs away (>10x, growing)     -> the advecting flux is genuinely
    //                                        convergent inside the clot. ANY
    //                                        limiter would DISCARD that
    //                                        material silently; the pressure
    //                                        equation's volume source has to
    //                                        be fixed first.
    //
    // NOT MULES: STAGE 6.79 ruled the VoF/alpha route out on its premise --
    // the clot is interpenetrating (Cogan & Guy 2010) and 99.5% liquid
    // (Wufsus 2013), so c is an indicator in a mixture, not a volume-of-fluid
    // fraction between immiscible phases.
    //
    // clipDVdt is the clip's own contribution to clotDVdt, and therefore the
    // amount by which it moves clotResidual on this same line. It is the
    // independent cross-check on the two numbers above: if the clip removed
    // volume and the residual did NOT move by clipDVdt, the balance is not
    // measuring what it claims to.

    // ---- THE PHASE-FIELD BOUNDS ------------------------------------------
    //
    // c must lie in [-1, 1]: q = 0.5(c+1) is a VOLUME FRACTION. Until
    // 2026-09-02 NOTHING IN THIS SOLVER REPORTED IT. That is not a cosmetic
    // gap. If growth drove c below -1 then q would go negative,
    // max(q, qFloor) would silently clamp it, and chi = max(1-q, 0) would
    // exceed 1 -- so the Brinkman drag would be built from an impossible
    // solid fraction and the only symptom would be a clot that behaves
    // slightly wrong. Exactly the silent class this project keeps being
    // caught by.
    //
    // It is added HERE, in the same commit that makes the growth source
    // 1.5-4.3x stronger, because that is the change most able to violate it.
    // MEASURED on run 20260901-200203's written fields: min c = -0.999999,
    // max c = +1.000000, violation identically zero -- but those are 30
    // samples out of 14670 steps, which is why this runs every step.
    {
        // S6c-2: REPORTED IN q, WHICH IS THE SURVIVING CURRENCY. c = 2q - 1
        // exactly, so c in [-1, 1] is q in [0, 1] and the two statements are
        // the same bound. The thresholds move with the variable; the check
        // does not.
        const scalarField& qBoundIn = q_.primitiveField();
        scalar qMin = great;
        scalar qMax = -great;
        forAll(qBoundIn, celli)
        {
            qMin = min(qMin, qBoundIn[celli]);
            qMax = max(qMax, qBoundIn[celli]);
        }
        reduce(qMin, minOp<scalar>());
        reduce(qMax, maxOp<scalar>());

        // HOW FAR THE UNDERSHOOT ACTUALLY REACHES, and what it costs
        // downstream. cUnder alone reports the worst cell; these say how
        // many cells are unphysical and how hard the platelet divisor is
        // being pushed there.
        //
        // q = (c+1)/2, so q < 0 is exactly c < -1. Run 20260908-010836 had
        // 230 such cells at 7.9 ms, ALL of them at chi > 0.9 -- deep clot,
        // none at the interface, none in the lumen -- and the divisor
        // amplification 1/max(|q|, qFloor) sat at exactly 1e+06 in every one
        // of them, which is what made wMin -3.1e+04 and VnMagMax 1.15 m/s.
        // With mag() in the divisor the amplification is now bounded by
        // 1/|q|, so qNegAmpMax is the number that says whether the guard is
        // holding: it should read ~1/|qMin| (about 18 there), not 1e+06.
        label qNegCells = 0;
        scalar qNegMin = 0;
        scalar qNegAmpMax = 0;
        // A POINTWISE EXTREMUM AND AN INTEGRAL CANNOT REPLACE EACH OTHER.
        //
        // qNegMin is a max-norm: it catches one catastrophic cell and is
        // deliberately blind to a thousand shallow ones. qNegContent is the
        // complementary L1 measure and qNegVolume the affected extent; a
        // global integral alone would hide an arbitrarily deep violation in a
        // sufficiently small cell, so both are printed and neither is a
        // substitute for the other. Cell counts are not comparable across
        // meshes; the volume is.
        //
        // NAMING IS DELIBERATE. qNegContent is
        //
        //     sum_i V_i * max(-q_i, 0)
        //
        // which is a VIOLATION FUNCTIONAL OF THE LEGACY INDICATOR, NOT a
        // conserved quantity and NOT negative solid mass. Under
        // theta = theta0*(1 - q), theta0*max(-q,0) = max(theta - theta0, 0):
        // it measures content above the NOMINAL reference density, which can
        // be legitimate compaction while still invalidating the accessibility
        // equation that reads q. Codex, 2026-09-08. Do not rename it to
        // anything containing "mass" and do not build a conservation argument
        // on it without settling the content/indicator contract first.
        scalar qNegVolume = 0;
        scalar qNegContent = 0;
        {
            const scalarField& qIn = q_.primitiveField();
            const scalarField& vIn = mesh.V();
            const scalar qF = qFloor_.value();
            forAll(qIn, celli)
            {
                if (qIn[celli] < 0)
                {
                    ++qNegCells;
                    qNegMin = min(qNegMin, qIn[celli]);
                    qNegAmpMax =
                        max(qNegAmpMax, 1.0/max(mag(qIn[celli]), qF));
                    qNegVolume += vIn[celli];
                    qNegContent += vIn[celli]*(-qIn[celli]);
                }
            }
        }
        reduce(qNegCells, sumOp<label>());
        reduce(qNegMin, minOp<scalar>());
        reduce(qNegAmpMax, maxOp<scalar>());
        reduce(qNegVolume, sumOp<scalar>());
        reduce(qNegContent, sumOp<scalar>());

        Info<< "PHASE_CLOT_PHASE_BOUNDS"
            << " time=" << runTime.name()
            << " qMin=" << qMin
            << " qMax=" << qMax
            << " qUnder=" << max(scalar(0), -qMin)
            << " qOver=" << max(scalar(0), qMax - scalar(1))
            << " qNegCells=" << qNegCells
            << " qNegMin=" << qNegMin
            << " qNegAmpMax=" << qNegAmpMax
            << " qNegVolume=" << qNegVolume
            << " qNegContent=" << qNegContent
            << nl;

        // WHAT WE CHOSE NOT TO CLAMP, MEASURED WHERE IT IS PRODUCED.
        //
        // phaseIndicatorBasis_ leaves theta/theta_ref unclamped above 1. That
        // is defensible only while somebody is watching the overshoot, so it
        // is watched here rather than inferred later from a field dump: a
        // decision not to bound a quantity is a commitment to measure it.
        //
        // indAboveOne is the number that can move. At 0.4 ms it is a thin
        // rim at the compressed interface and indMax is 1.0022; a run in
        // which indMax walks up, or in which indAboveOne grows into the body,
        // is compacting past the anchor and the cubic backstop is not holding
        // it. chiLegacyMax is printed on both bases so the two indicators are
        // comparable within one run and across archived ones -- and note it
        // is a DIFFERENCE OF MAXIMA, which at 0.4 ms agreed to 8 ppm while
        // the pointwise difference was 1.6e-02, 2144x larger. It is a
        // trending instrument, not an agreement test.
        {
            const volScalarField ind(phaseIndicator());
            const volScalarField chiLegacy
            (
                max(1.0 - q_, dimensionedScalar("zero", dimless, 0))
            );

            // THE TWO SET SIZES THE MIGRATION ACTUALLY MOVES.
            //
            // Priced offline against em32's own written fields before this
            // key was ever run: swapping the basis changes indMax by 6.9 ppm
            // and the POINTWISE indicator by 1.556e-02, 2239x more -- and
            // 1.556e-02 is larger than residualChi_ itself. So the sets Kres
            // and isSkel are defined by move, by 34 and 27 cells of 31360 at
            // t = 4e-04. Neither number was logged anywhere, and a
            // pre-registration on an unlogged quantity is a prediction that
            // cannot be graded, which this project has already paid for three
            // times. They are logged here.
            //
            // -1 means "not applicable": below residualChi_ <= 0 the solver
            // keys both sets off phiS against residualPhiS_ instead, and
            // printing a chi-threshold count there would be a fabricated
            // number rather than a missing one.
            label indAboveOne = 0;
            label kresCells = 0;
            label isSkelCells = 0;
            forAll(ind, celli)
            {
                if (ind[celli] > 1.0)
                {
                    indAboveOne++;
                }
                if (ind[celli] < residualChi_)
                {
                    kresCells++;
                }
                if (ind[celli] > 0.5*residualChi_)
                {
                    isSkelCells++;
                }
            }
            reduce(indAboveOne, sumOp<label>());
            reduce(kresCells, sumOp<label>());
            reduce(isSkelCells, sumOp<label>());

            if (residualChi_ <= 0)
            {
                kresCells = -1;
                isSkelCells = -1;
            }

            const scalar indMax = gMax(ind.primitiveField());

            Info<< "PHASE_CLOT_PHASE_INDICATOR"
                << " time=" << runTime.name()
                << " basis=content"
                << " indMax=" << indMax
                << " indMin=" << gMin(ind.primitiveField())
                << " indAboveOne=" << indAboveOne
                << " indExcessMax=" << max(indMax - 1.0, scalar(0))
                << " chiLegacyMax=" << gMax(chiLegacy.primitiveField())
                << " residualChi=" << residualChi_
                << " kresCells=" << kresCells
                << " isSkelCells=" << isSkelCells
                << nl;
        }

        // ---- THE SURFACE DELTA, BOTH WAYS. INSTRUMENT ONLY. ----
        //
        // Nothing consumes deltaTheta. It exists because retiring `c` needs a
        // surface delta made from theta, and the substitution has to be
        // MEASURED on a real run before any deposition depends on it.
        //
        // WHY THIS IS THE RIGHT FORM. A surface delta must satisfy
        // INT delta dn = 1 across the interface -- that is what makes
        // kEff*delta a flux per unit area, and it is the contract
        // plateletContent.H enforces with a dimension check. For c running
        // -1 to +1, interfaceDelta_ = 0.5*|grad c| satisfies it exactly. For
        // theta running 0 to theta_bulk,
        //
        //     INT |grad theta|/theta_ref dn  =  theta_bulk/theta_ref
        //
        // which is exactly indMax above. So the calibration error of the
        // theta delta is a number this line already prints, and it needs no
        // new instrument to watch -- 1.00217 at 0.4 ms of run
        // 20260916-031327, i.e. 0.22%.
        //
        // It is NOT sensitive to interface width: INT |grad theta| dn is
        // theta_bulk however smeared the profile is, so theta having no
        // controlled epsilon does not break the identity. Width sets how
        // spread the deposition is, not how much of it there is.
        //
        // AND THAT SENTENCE IS THE DEFECT IN THE FIRST VERSION OF THIS
        // INSTRUMENT. Run 20260916-053445 gated on deltaRatio, the ratio of
        // the two domain integrals, and it PASSED -- 0.999457 against an
        // indMax of 1.00217, a 0.27% departure, held for all 272 steps. The
        // total binding flux is preserved to 0.05%. Meanwhile, measured from
        // the same run's written fields in the interface band and weighted by
        // interfaceDelta_, theta/theta_ref and chi = (1-c)/2 had already
        // separated by 2.2% in the mean and 15.6% at p99, growing as t^1.83
        // on a 0.4 ms window that is 1.3% of a main case. An integral cannot
        // distinguish a conserved TOTAL from a conserved DISTRIBUTION, and
        // the paragraph above says so one line before the gate was set on it.
        //
        // So deltaRatio stays -- it is the calibration check and it is the
        // right instrument for that -- but it is REPORTED, NEVER GATED. The
        // gate is deltaTV below, and deltaTheta_ is now a written field so
        // the per-cell comparison does not depend on this instrument having
        // guessed the right statistic.
        //
        // fvc::grad, NOT a finite difference over written cell centres. A
        // hand reconstruction of a gradient in this project was wrong by 2x
        // once already, and the solver's own operator is the authority.
        //
        // Both integrals are areas [m2] -- each integrates to the clot
        // surface -- so their RATIO is the dimensionless measurement wanted,
        // and it should track indMax if the argument above is right.
        //
        // S6c-2: THE c-SIDE OF THIS INSTRUMENT IS GONE, THE theta SIDE IS NOT.
        //
        // It compared deltaC_ = 0.5*|grad c| against deltaTheta_ and gated on
        // deltaTV, the half-L1 distance between them once each was normalised
        // to unit mass. deltaC_ is deleted, and rebuilding it from theta would
        // make the instrument compare a field against itself -- deltaTV
        // identically zero, a triumphant pass measuring nothing, the G3 defect
        // this instrument's own comment warned about.
        //
        // deltaCInt, deltaRatio, deltaTV, deltaShift{Mag,X,Y}, rmsC and
        // supportRatio are retired with it. deltaThetaInt,
        // deltaThetaIntPost and deltaThetaIntAtTransfer are NOT: they are
        // theta-only, and the three of them together are OI-11 -- the
        // question of which time level the reaction reads the geometry at.
        // That question is untouched by deleting c.
        {
            scalar deltaThetaInt = -1;
            scalar deltaThetaIntPost = -1;

            if (skeletonVolumeFraction_ > 0)
            {
                // The field as the CONSUMERS saw it -- assigned in
                // updateDerivedFields(), before solvePlatelets() committed
                // the accepted transfer.
                deltaThetaInt = fvc::domainIntegrate(deltaTheta_).value();

                // And the same integral recomputed HERE, at postSolve, from
                // the theta the step ended with. The difference between the
                // two is exactly the cost of evaluating the delta at the
                // start of the step rather than the end. It used to be the
                // only number reported, silently; it is now reported beside
                // the one that matters so the relocation can never be
                // confounded with the substitution. Expected to be small --
                // contentVolume moves ~2.4e-06 per step -- but measured
                // rather than assumed.
                deltaThetaIntPost =
                    fvc::domainIntegrate
                    (
                        mag(fvc::grad(thetaC_ + thetaH_))
                       /dimensionedScalar
                        (
                            "thetaRef", dimless, skeletonVolumeFraction_
                        )
                    ).value();

            }

            // -1 means "not computable here", never a plausible number: with
            // no content model theta is frozen at its initialisation and any
            // figure printed from it would describe nothing. deltaTV is in
            // [0,1] and deltaShiftMag/rmsC/supportRatio are non-negative, so
            // for those four the sentinel is outside the range and cannot be
            // read as a measurement.
            //
            // deltaShiftX and deltaShiftY are SIGNED and -1 is a legal value
            // for them in principle -- absurd on a 1e-04 m domain, but legal.
            // Read their availability off deltaShiftMag, which is the one
            // that cannot be negative, and never off the components.
            Info<< "PHASE_CLOT_SURFACE_DELTA"
                << " time=" << runTime.name()
                << " deltaThetaInt=" << deltaThetaInt
                << " deltaThetaIntPost=" << deltaThetaIntPost
                << " deltaThetaIntAtTransfer=" << deltaThetaIntAtTransfer_
                << " thetaRef=" << skeletonVolumeFraction_
                << " gateIs=OI-11-deltaThetaIntAtTransfer"
                << " retiredInS6c2="
                   "deltaCInt+deltaRatio+deltaTV+deltaShift+rmsC+supportRatio"
                << " consumedBy=deltaTheta-IS-interfaceDelta"
                << nl;
        }

        // THE SKELETON'S DENSITY AND WHAT IS HOLDING IT THERE.
        //
        // cMin above says the phase field left its bounds; this says WHY, in
        // the variable the constitutive law actually acts on. Run
        // 20260905-113709 reached phi_s = 0.0187 against a nominal 0.01 with
        // p_n absent entirely -- there was no restoring force and no way to
        // see that there wasn't. phiSOverNominal is the number to watch: 1.0
        // is the equilibrium, and a run walking away from it is compacting
        // (or dilating) whatever c happens to read.
        {
            // THE FRACTION THE FORCE ACTUALLY USES, not solidFraction().
            //
            // networkPressure() follows phaseForceBasis_, so pairing its
            // output with solidFraction() here would report
            // solidFraction()*p_n(theta) -- a MIXED quantity that is neither
            // the legacy term nor the migrated one, and that no reader could
            // interpret. The legacy fraction is still published below as
            // phiSLegacyMax so archived runs stay comparable.
            const volScalarField phiSdiag(forceFraction());
            const volScalarField pnDiag(networkPressure());
            const scalar phiSMaxSeen = gMax(phiSdiag.primitiveField());
            const scalar phiSMinSeen = gMin(phiSdiag.primitiveField());
            const scalar pnMax = gMax(pnDiag.primitiveField());
            const scalar pnMin = gMin(pnDiag.primitiveField());

            // THE FAILABLE GATE ON THE POLE.
            //
            // Every form of this law caps phi_s at 0.99*phi_s_max before the
            // divide, so no law can report a phi_s above it and NO BOUND ON
            // phi_s CAN EVER FIRE -- that is the unfailable-gate defect this
            // project retired as G3, and writing a bound on phiSOverNominal
            // here would repeat it.
            //
            // What CAN fail is the cap itself. If the force balance is doing
            // the bounding, no cell ever reaches the clamp and this is 0. A
            // nonzero count means the discretisation walked a cell THROUGH
            // the pole between two steps and the clamp -- not the physics --
            // is what held it, which is precisely the claim "close packing is
            // unreachable by force balance" failing. Reported every step so
            // the first occurrence has a time against it.
            const scalar capLevel = 0.99*networkPole();
            label pnCapCells = 0;
            // S10: the cap is on the law's ARGUMENT, which under "deposited"
            // is not the force fraction. Counting on phiSdiag would gate a
            // quantity the law no longer reads.
            const volScalarField pnArg(networkPressureArgument());
            const scalarField& phiSdIn = pnArg.primitiveField();
            forAll(phiSdIn, celli)
            {
                if (phiSdIn[celli] >= capLevel)
                {
                    ++pnCapCells;
                }
            }
            reduce(pnCapCells, sumOp<label>());
            const label pnTotalCells = returnReduce(phiSdIn.size(), sumOp<label>());

            // HOW MUCH AUTHORITY THE NETWORK PRESSURE ACTUALLY HAS.
            //
            // pnMax/pnMin above are the RAW law, evaluated everywhere. In the
            // lumen that is a constant -K_n*phi_s0/(phiSMax - phi_s0)
            // = -0.0667 Pa whether or not there is any network there, so
            // pnMin cannot distinguish "the law has a value" from "that value
            // reaches the momentum equation". Run 20260905-191135 printed
            // pnMin=-0.0666667 on every line while elasticStress() multiplied
            // it by max(chi, 1e-4) instead of phi_s, and that constant DID
            // reach VsEqn in the lumen -- where A() = 0 and only
            // Kres = 66.7 1/s opposes it. maxVs was 48x Batch 16's from the
            // first sample.
            //
            // pnTermMax/pnTermMin are phi_s*p_n, the isotropic term as it
            // enters the stress. They are NOT a bug detector: with the
            // solidFraction() weighting the term is identically zero wherever
            // chi is (chi = max(1-q,0) is clipped, so exactly zero in the
            // lumen), and pnTermMin is instead a real negative number from the
            // diffuse edge, where phi_s < phi_s0 and the law resists dilation.
            // Predicted -1.7e-04 Pa at G = 4 and -1.7e-03 Pa at G = 40, both
            // at phi_s ~ 0.005.
            //
            // WHAT THEY ARE FOR is the magnitude question, which is the one
            // open parameter (networkPressureCoeff = 1.0 is not fitted to
            // anything). Against the deviatoric term chi*gD*G*dev(be) the
            // ratio is 0.01*(phi_s - phi_s0)/((phiSMax - phi_s)*|dev be|) and
            // G CANCELS. Measured along run 20260905-113709: 4.1e-05 at
            // t = 2.2e-04 rising only to 2.6e-04 at its worst compaction
            // (phi_s = 1.87x nominal). Parity needs phi_s = 0.59, i.e. 59x
            // nominal. THE LAW AS ANCHORED IS A CLOSE-PACKING BACKSTOP AND
            // NOTHING ELSE -- it has no authority in the range these runs
            // occupy, which is what the matched-time comparison of 113709 and
            // 191135 independently showed (identical cUnder to four figures).
            // S11: the isotropic term AS IT ENTERS sigma, whichever law --
            // identical to phiSdiag*pnDiag for every legacy law.
            const volScalarField pnTerm(networkPartialPressure());
            const scalar pnTermMax = gMax(pnTerm.primitiveField());
            const scalar pnTermMin = gMin(pnTerm.primitiveField());

            Info<< "PHASE_CLOT_NETWORK"
                << " time=" << runTime.name()
                << " phiSMax=" << phiSMaxSeen
                << " phiSMin=" << phiSMinSeen
                << " phiSNominal=" << skeletonVolumeFraction_
                << " phiSOverNominal="
                << phiSMaxSeen/max(skeletonVolumeFraction_, small)
                << " phiSFracOfMax=" << phiSMaxSeen/max(networkPole(), small)
                << " forceBasis=content"
                << " phiSLegacyMax="
                << gMax(volScalarField(solidFraction())().primitiveField())
                << " pnMax=" << pnMax
                << " pnMin=" << pnMin
                << " pnTermMax=" << pnTermMax
                << " pnTermMin=" << pnTermMin
                << " pnModel=convexExcess"
                << " pnCapLevel=" << capLevel
                << " pnCapCells=" << pnCapCells
                << " pnCapFrac="
                << scalar(pnCapCells)/max(scalar(pnTotalCells), small)
                << " pnReference=deposited"
                << " pnArgMax=" << gMax(pnArg.primitiveField())
                << " pnArgMin=" << gMin(pnArg.primitiveField())
                << nl;
        }
    }

    // ---- THE STOICHIOMETRY INSTRUMENT -----------------------------------
    //
    // The one number that says whether growth and consumption are the same
    // physical process. Platelets leave the blood at
    //     uptake = INT delta_Gamma * kEff * w dV = INT kEff w dA,
    // because delta_Gamma = |grad c|/2 integrates to one across the
    // interface. Clot volume appears at
    //     growth/2 = INT Vn * |grad c| / 2 dV = INT Vn dA.
    // So growth/uptake must be 2*alpha/D, EXACTLY and pointwise, under the
    // reaction closure -- alpha/D is then literally "clot volume per unit of
    // platelet consumed" and has a physical meaning.
    //
    // Under the old gradient closure this ratio was MEASURED at 0.034 to 0.10
    // of its target on run 20260901-200203, and it drifted by 3x over 30 ms:
    // the clot was growing at a rate unrelated to the platelets it ate. That
    // is the defect this instrument exists to make impossible to miss again.
    //
    // Both integrals are formed from the fields as the LAST updateDerivedFields
    // left them, and w is reconstructed exactly as the closure reconstructs
    // it, so the only admissible discrepancy is the gradFloor softening in
    // phaseGradient_ against the bare |grad c| in interfaceDelta_. That is
    // one-signed (phaseGradient <= |grad c|), so the ratio can only come in
    // at or BELOW 1, never above.
    {
        volScalarField wStoich
        (
            IOobject
            (
                "wStoich", runTime.name(), mesh_,
                IOobject::NO_READ, IOobject::NO_WRITE
            ),
            w_
        );
        {
            // mag(q_): see wGrowth. A diagnostic that reports the
            // 1e+06-amplified quotient is worse than no diagnostic. The
            // sink's own operand (q + e)/max(|q|, qFloor), as wGrowth.
            wStoich = (q_ + e_)/max(mag(q_), qFloor_);
        }

        const scalar plateletUptake =
            fvc::domainIntegrate
            (
                interfaceDelta_*adhesionSpeedEff_*wStoich
            ).value();

        // THE CLOSURE IDENTITY, checked to machine precision.
        //
        // If growthClosure is "reaction" then Vn IS (alpha/D)*kEff*w by
        // construction, so this residual is ~1e-16 and any other number means
        // the field the phase equation is using is not the field the closure
        // claims to have written -- a stray term, a wrong alpha or D, or the
        // gradient branch running while the dictionary says otherwise. Under
        // the gradient closure it is O(1) by design, which is what makes this
        // an instrument with seven decades of separation rather than a
        // threshold somebody has to argue about.
        scalar vnClosureResid = 0;
        scalar vnClosureRef = 0;
        {
            const scalarField& vnIn = Vn_.primitiveField();
            const scalarField& kEffIn = adhesionSpeedEff_.primitiveField();
            const scalarField& wIn = wStoich.primitiveField();
            // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
            // so every threshold below is the same threshold restated:
            // c < L becomes ind > (1 - L)/2, and the inequality turns
            // over with it. Same cells, named in the surviving field.
            const volScalarField indHold(phaseIndicator());
            const scalarField& indIn = indHold.primitiveField();
            // Same D = 0 trap as the closure (run 20260902-141852), one
            // step later. A DIAGNOSTIC must not kill a run, and must not
            // invent a number either: 0 when no growth was requested, nan
            // when the ratio is genuinely undefined. stoichNorm already
            // reports nan as "undefined" rather than gating it.
            const scalar aOverD =
                (alpha_.value() == 0.0) ? 0.0
              : (D_.value() == 0.0)
              ? std::numeric_limits<scalar>::quiet_NaN()
              : (alpha_/D_).value();

            forAll(vnIn, celli)
            {
                // Only the diffuse interface can grow; outside it Vn is
                // multiplied by a vanishing phaseGradient and its value is
                // not a statement about anything.
                // |c| < 0.9 IS NOT mag(ind) < 0.9. S6c-2 carried the
                // comparison across and left the absolute value behind.
                // c = 1 - 2*ind puts the interface at ind = 0.5, not at
                // ind = 0, so |c| < 0.9 is the BAND 0.05 < ind < 0.95 --
                // while mag(ind) < 0.9 is everything below ind = 0.9,
                // which is the whole lumen. Measured on run
                // 20260920-220246's own written q: this selected 29345 of
                // 31360 cells, 93.6%, against em45's 1658.
                if (mag(scalar(1) - 2*indIn[celli]) < 0.9)
                {
                    vnClosureResid = max
                    (
                        vnClosureResid,
                        mag(vnIn[celli] - aOverD*kEffIn[celli]*wIn[celli])
                    );
                    vnClosureRef = max(vnClosureRef, mag(vnIn[celli]));
                }
            }
            reduce(vnClosureResid, maxOp<scalar>());
            reduce(vnClosureRef, maxOp<scalar>());
        }
        const scalar vnClosureRel =
            vnClosureResid/max(vnClosureRef, vSmall);

        const scalar stoichTarget =
            (alpha_.value() == 0.0) ? 0.0
          : (D_.value() == 0.0)
          ? std::numeric_limits<scalar>::quiet_NaN()
          : 2.0*(alpha_/D_).value();
        // STOICHIOMETRY STAYS ON THE FRONT, DELIBERATELY.
        //
        // This identity tests the FRONT closure -- that Vn*phaseGradient
        // converts platelets to clot at 2*alpha/D -- and it is the number
        // every archived run reports. Feeding it the S4 growth instead would
        // silently redefine a logged name and break that comparison, while
        // measuring something else entirely. It is read against
        // clotGrowthFront on every driver.
        const scalar stoichRatio =
            (mag(plateletUptake) > vSmall)
          ? clotGrowthFront/plateletUptake
          : 0.0;
        const scalar stoichNorm =
            (mag(stoichTarget) > vSmall) ? stoichRatio/stoichTarget : 0.0;

        // ---- OI-12: THE TRANSFER THAT HAPPENED, BESIDE THE PROXY --------
        //
        // plateletUptake = INT interfaceDelta*adhesionSpeedEff*w dV has NO N
        // in it, while the committed transfer dB = dt*lambda*N/(1 + dt*lambda)
        // does. So plateletUptake is a RATE-side quantity -- how fast the
        // surface would bind if the reactant were free -- and it is
        // structurally blind to depletion. Measured across run
        // 20260917-003709: dBSum fell 21.76% while plateletUptake moved
        // 0.025% and stoichNorm held to 0.054%. Both of stoichNorm's inputs
        // are front-side, which is why it could not see it.
        //
        // Added rather than substituted: both are wanted, and the GAP between
        // the two ratios IS the depletion.
        const scalar plateletUptakeCommitted =
            plateletYield_*dBSumCommitted_/deltaT;

        // EXACTLY 2/theta_ref UNLESS applyIncrement CLAMPED. Under
        // phaseGrowthDriver 'transfer' c receives (2/theta_ref) times the
        // APPLIED theta increment, and that increment is Y*dB except where
        // plateletTransferKernel::applyIncrement caps against phiSMax_. So
        // this is a CAP DETECTOR, not a stoichiometry: it reads 4.4444...
        // for theta_ref = 0.45 on a run where capClamped is 0 on every
        // record, and departs from it exactly where the cap bit. Read it
        // beside availClamped/capClamped in PHASE_CLOT_FREEPLATELET, never
        // as evidence about the model.
        const scalar transferStoich =
            (mag(plateletUptakeCommitted) > vSmall)
          ? clotGrowth/plateletUptakeCommitted
          : 0.0;

        Info<< "PHASE_CLOT_GROWTH_STOICH"
            << " time=" << runTime.name()
            << " closure=" << "reaction"
            << " depletion=" << "none"
            // WHICH PLATELET BRANCH RAN. The Fourier, LED and limiter
            // instruments live inside the CONTENT branch of solvePlatelets(),
            // so a plateletContentFormulation=no child can never emit them.
            // Without this token the assessor read that absence as "the binary
            // predates the harmonic fix" and failed gate_a_advect -- the
            // legacy control -- for doing exactly its job. Run 20260902-104340.
            << " content=yes"
            << " clotGrowth=" << clotGrowth
            << " clotGrowthFront=" << clotGrowthFront
            << " growthDriver=transfer"
            << " plateletUptake=" << plateletUptake
            << " plateletUptakeCommitted=" << plateletUptakeCommitted
            << " transferStoich=" << transferStoich
            << " transferStoichTarget=" << (2.0/skeletonVolumeFraction_)
            << " stoichRatio=" << stoichRatio
            << " stoichTarget=" << stoichTarget
            << " stoichNorm=" << stoichNorm
            << " VnClosureResidual=" << vnClosureRel
            << " VnClosureRef=" << vnClosureRef
            << nl;
    }

    // ---- IS THE MATERIAL LABEL STILL ON THE MATERIAL? --------------------
    //
    // shell_ marks core from shell and is advected by the SAME carrier as c,
    // but by `fvm::div(carrier, shell_, "div(phiReleased,c)")` -- its OWN
    // vanLeer limiter, because a limiter is a function of the transported
    // field. So c and shell_ have never shared weights, and after the
    // 2026-09-02 shared-operator fix c is advected with the more-upwind
    // combination while shell_ still is not. The gap is PRE-EXISTING and it
    // widened; neither state has ever been measured.
    //
    // It matters for the reason advectMaterialLabel() already states: shell
    // material arriving in a cell still labelled core silently acquires the
    // core's modulus, toughness and permeability. In the shell-modulus sweep
    // that is the difference between the two materials being swept.
    //
    // MEASURED, NOT ASSERTED, and reported rather than gated -- there is no
    // measurement yet from which to derive a threshold, and a guessed one is
    // how this project has produced four unpassable gates. `labelInBlood` is
    // the fraction of the label sitting where there is no clot to carry it.
    if (coreShell_)
    {
        const scalarField& shellIn = shell_.primitiveField();
        const scalarField& qIn2 = q_.primitiveField();
        const scalarField& VIn2 = mesh_.V();

        scalar labelTotal = 0;
        scalar labelInBlood = 0;
        forAll(shellIn, celli)
        {
            const scalar m = shellIn[celli]*VIn2[celli];
            labelTotal += m;
            // q is the BLOOD fraction, so q weights the part of the label
            // that is sitting in blood rather than on clot.
            labelInBlood += m*qIn2[celli];
        }
        reduce(labelTotal, sumOp<scalar>());
        reduce(labelInBlood, sumOp<scalar>());

        Info<< "PHASE_CLOT_LABEL"
            << " time=" << runTime.name()
            << " labelTotal=" << labelTotal
            << " labelInBlood="
            << (labelTotal > vSmall ? labelInBlood/labelTotal : 0.0)
            << nl;
    }

    // ---- WHERE THE GROWTH COMES FROM -------------------------------------
    //
    // ALIREZA'S QUESTION, 2026-09-02, made permanent: in ParaView `Wp` is zero
    // inside the clot with a sharp edge, but `w` is NOT -- it keeps an O(1)
    // value with a smooth gradient in there. Is it legitimate to drive growth
    // with `w` when `w` does not represent the clot geometry?
    //
    // THE ANSWER IS THAT `w` IS NOT SUPPOSED TO CARRY THE GEOMETRY. `c` does.
    // `w = Wp/q` is an INTENSIVE property -- the concentration in the blood
    // that is present -- so it is O(1) wherever any blood remains, and inside
    // the clot it is a mathematical extension of a field whose physical domain
    // is the blood. The source paper says so outright: platelet values in the
    // thrombus region "have no physical meaning". The growth term multiplies
    // it by |grad c|, so the SHAPE comes from c and only the RATE comes from w.
    //
    // But "the localiser handles it" is an argument, and arguments are what
    // this project keeps getting caught by. So it is measured instead. This
    // reports the share of the growth that comes from cells where w is a ratio
    // of two small numbers and means nothing.
    //
    // MEASURED on run 20260901-200203, 5 children x 7 times: 4.0% at 1 ms when
    // the interface is most diffuse, settling to ~2%; and the growth-weighted
    // mean q sits at 0.53-0.63, i.e. centred on the c = 0 surface where the
    // deposition physically happens. If the clot-interior extension ever
    // starts driving the growth, this is the number that says so.
    {
        const scalarField& qIn = q_.primitiveField();
        const scalarField& vnIn = Vn_.primitiveField();
        const scalarField& pgIn = phaseGradient_.primitiveField();
        const scalarField& VIn = mesh_.V();

        scalar gTot = 0;
        scalar gBelow005 = 0;
        scalar gBelow015 = 0;
        scalar gQ = 0;

        forAll(vnIn, celli)
        {
            const scalar g = vnIn[celli]*pgIn[celli]*VIn[celli];
            if (g <= 0)
            {
                continue;
            }
            gTot += g;
            gQ += qIn[celli]*g;
            if (qIn[celli] < 0.05) { gBelow005 += g; }
            if (qIn[celli] < 0.15) { gBelow015 += g; }
        }

        reduce(gTot, sumOp<scalar>());
        reduce(gBelow005, sumOp<scalar>());
        reduce(gBelow015, sumOp<scalar>());
        reduce(gQ, sumOp<scalar>());

        const scalar denom = max(gTot, vSmall);

        Info<< "PHASE_CLOT_GROWTH_SUPPORT"
            << " time=" << runTime.name()
            << " growthBelowQ005=" << (gTot > 0 ? gBelow005/denom : 0.0)
            << " growthBelowQ015=" << (gTot > 0 ? gBelow015/denom : 0.0)
            << " growthMeanQ=" << (gTot > 0 ? gQ/denom : 0.0)
            << nl;
    }

    // ---- THE REGIME INSTRUMENT ------------------------------------------
    //
    // Da = kEff * h_n / D compares the deposition rate against how fast
    // diffusion can resupply across one cell. Da < 1: reaction-limited, and
    // Vn = (alpha/D) k w is the right closure. Da > 1: the depletion layer
    // D/k is thinner than a cell, the cell-average w over-reads the surface
    // value, and the reaction closure over-reads growth -- which is when
    // interfaceDepletionModel seriesResistance earns its place.
    //
    // This is REPORTED, NOT ACTED ON. It is the number that tells Alireza
    // whether the declared limit of the closure has been reached, instead of
    // the limit being discovered later from a wrong answer. On this mesh
    // D/k0 = 3.04 cells (Da = 0.33) and the closure is comfortable; under the
    // full 20x shear enhancement D/k = 0.15 cells (Da = 6.6) and it is not.
    {
        const volScalarField hNormal(normalCellSize());

        const scalarField& kIn = adhesionSpeedEff_.primitiveField();
        const scalarField& hIn = hNormal.primitiveField();
        // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
        // so every threshold below is the same threshold restated:
        // c < L becomes ind > (1 - L)/2, and the inequality turns
        // over with it. Same cells, named in the surviving field.
        const volScalarField indHold(phaseIndicator());
        const scalarField& indIn = indHold.primitiveField();
        const scalar Dv = max(D_.value(), vSmall);

        scalar daMax = 0;
        scalar daInterfaceMax = 0;
        scalar layerCellsMin = great;
        label interfaceCells = 0;

        forAll(kIn, celli)
        {
            const scalar da = kIn[celli]*hIn[celli]/Dv;
            daMax = max(daMax, da);

            // Only the diffuse interface can grow, so only there does the
            // regime mean anything.
            // |c| < 0.9 IS NOT mag(ind) < 0.9. S6c-2 carried the
            // comparison across and left the absolute value behind.
            // c = 1 - 2*ind puts the interface at ind = 0.5, not at
            // ind = 0, so |c| < 0.9 is the BAND 0.05 < ind < 0.95 --
            // while mag(ind) < 0.9 is everything below ind = 0.9,
            // which is the whole lumen. Measured on run
            // 20260920-220246's own written q: this selected 29345 of
            // 31360 cells, 93.6%, against em45's 1658.
            if (mag(scalar(1) - 2*indIn[celli]) < 0.9)
            {
                ++interfaceCells;
                daInterfaceMax = max(daInterfaceMax, da);
                if (da > vSmall)
                {
                    layerCellsMin = min(layerCellsMin, 1.0/da);
                }
            }
        }

        reduce(daMax, maxOp<scalar>());
        reduce(daInterfaceMax, maxOp<scalar>());
        reduce(layerCellsMin, minOp<scalar>());
        reduce(interfaceCells, sumOp<label>());

        Info<< "PHASE_CLOT_DAMKOHLER"
            << " time=" << runTime.name()
            << " DaMax=" << daMax
            << " DaInterfaceMax=" << daInterfaceMax
            << " layerCellsMin="
            << (interfaceCells > 0 ? layerCellsMin : 0.0)
            << " interfaceCells=" << interfaceCells
            << nl;
    }

    // THE EULERIAN STATE, AND THE PROOF THAT THE PATH RAN AT ALL.
    //
    // solveElasticStrain() computed all four of these and, until now, nothing
    // printed them -- they were measured and thrown away. That is not merely a
    // lost diagnostic: with no per-step evidence, a wiring fault that stops the
    // Eulerian mechanics from running at all is INVISIBLE. Exactly that
    // happened. The first build reached solveElasticStrain() only from
    // solveMechanicsExchange(), which is called only when stronglyCoupledFSI_
    // is true; it defaults to false, so be sat at I for the whole run and the
    // only symptom would have been the gate reporting "the clot did not lock"
    // -- a physics sentence for a plumbing defect.
    //
    // The presence of this line is now itself the check: the gate's assessor
    // requires it, so a run in which the Eulerian path never executed fails on
    // that, naming the cause, instead of on a downstream consequence.
    {
        Info<< "PHASE_CLOT_EULERIAN"
            << " time=" << runTime.name()
            << " beMinMinor=" << beMinMinor_
            << " beMaxStretch=" << beMaxStretch_
            << " beDetError=" << beDetError_
            << " beDetErrorPost=" << beDetErrorPost_
            << " beSweepChange=" << beSweepChange_
            << nl;

        // Emitted on the same terms the stage-1 gate used, so the assessor's
        // SPD check reads one spelling across both stages.
        if (beMinMinor_ <= 0)
        {
            Info<< "BE_SPD_LOST at time " << runTime.name() << nl;
        }

        // WHERE THE STRAIN AND THE VOLUME ERROR LIVE, BY PHASE.
        //
        // Run 20260817-155652 measured det(be) drifting monotonically away
        // from 1 -- 2.2e-8 to 6.4e-7 across the second half of one child --
        // while the clot INTERIOR was converging beautifully (0.58% off the
        // closed form at the finest mesh, ~2nd order in mesh). A domain-wide
        // maximum cannot say whether those are the same region, and the two
        // have opposite implications: a growing error in the bulk means the
        // coupling itself needs stabilising, while one confined to the
        // interface is a diffuse-interface artefact with a much cheaper fix.
        //
        // The suspect for the interface case is written down in advance so
        // this is a test and not a fishing trip: beRelax = q^2 * beRelaxRate,
        // so at q = 0.1 the relaxation time is 100x the blood value and
        // interface cells can accumulate strain that neither the clot's
        // elasticity nor blood's relaxation removes.
        //
        // Same three bins as the damage driver, for the same reason -- if the
        // two diagnostics binned differently they could not be read together.
        {
            scalar sBulk = 0, sInterface = 0, sBlood = 0;
            scalar dBulk = 0, dInterface = 0, dBlood = 0;
            // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
            // so every threshold below is the same threshold restated:
            // c < L becomes ind > (1 - L)/2, and the inequality turns
            // over with it. Same cells, named in the surviving field.
            const volScalarField indHold(phaseIndicator());
            const scalarField& indIn = indHold.primitiveField();
            const symmTensorField& beIn = be_.primitiveField();
            forAll(indIn, celli)
            {
                const symmTensor& b = beIn[celli];
                const scalar stretch = Foam::sqrt(max(b.xx(), max(b.yy(), b.zz())));
                const scalar detErr = mag(det(b) - 1.0);
                if (indIn[celli] > 0.95)
                {
                    sBulk = max(sBulk, stretch);
                    dBulk = max(dBulk, detErr);
                }
                else if (indIn[celli] > 0.25)
                {
                    sInterface = max(sInterface, stretch);
                    dInterface = max(dInterface, detErr);
                }
                else
                {
                    sBlood = max(sBlood, stretch);
                    dBlood = max(dBlood, detErr);
                }
            }
            reduce(sBulk, maxOp<scalar>());
            reduce(sInterface, maxOp<scalar>());
            reduce(sBlood, maxOp<scalar>());
            reduce(dBulk, maxOp<scalar>());
            reduce(dInterface, maxOp<scalar>());
            reduce(dBlood, maxOp<scalar>());

            Info<< "PHASE_CLOT_BE_PHASE"
                << " time=" << runTime.name()
                << " stretchBulk=" << sBulk
                << " stretchInterface=" << sInterface
                << " stretchBlood=" << sBlood
                << " detBulk=" << dBulk
                << " detInterface=" << dInterface
                << " detBlood=" << dBlood
                << nl;
        }

        // WHY det(be) DRIFTS: the discrete trace of the velocity gradient.
        //
        // Run 20260817-234636 measured det(be) climbing from 6.7e-16 to 3.24
        // over 1174 steps in the Xu child, against the 1e-6 stage 2 held on
        // the 1-D slab. det(be) = 1 is exact for ANY isochoric deformation, so
        // this is not a large-strain excuse -- it is volume being created.
        //
        // The evolution obeys d(ln det be)/dt = 2 tr(L), so det is preserved
        // if and only if tr(L) = 0. The pressure solve enforces zero
        // divergence on the FACE FLUX -- continuity errors were 7.5e-15, i.e.
        // machine level -- but the be source uses twoSymm(be & fvc::grad(U)),
        // and tr(fvc::grad(U)) from the CELL-CENTRED velocity is a DIFFERENT
        // discrete operator. On a collocated mesh the two differ by the
        // Rhie-Chow correction, which is small for a smooth field and need not
        // be small across an interface carrying a jump in elastic stress.
        //
        // Inverting the measured growth gives a mean tr(L) of about 240 1/s
        // against a 1000 1/s shear rate -- a 24% trace error. THAT NUMBER IS
        // AN INFERENCE FROM det, NOT A MEASUREMENT, which is what this
        // diagnostic exists to fix. Reported as a fraction of the local
        // gradient magnitude so it is scale-free and comparable between the
        // two geometries.
        //
        // divPhiCell is fvc::div(phi) interpolated to cells: the same
        // quantity the continuity check reports, printed alongside so the two
        // operators can be compared on one line. If traceFrac is large while
        // divPhi stays at machine zero, the discretisation is the cause and
        // the remedy is a trace-free source, not a smaller time step.
        {
            const volTensorField gradU(fvc::grad(U_));
            // THE SAME TRACE, FROM THE FLUX-CONSISTENT OPERATOR. Without this
            // the fix is invisible: beFluxConsistentGradU changes what the be
            // equation uses, while this diagnostic reads fvc::grad(U_) and
            // would keep reporting the old 1.0 whatever the be source does.
            // Reporting both is what makes the change verifiable rather than
            // asserted -- and it keeps the OLD number on the record, so the
            // spurious cell-centred dilatation is not quietly hidden by a fix
            // that only routes around it.
            const surfaceVectorField UfDiag(fvc::interpolate(U_));
            const volTensorField gradUFlux
            (
                fv::gaussGrad<vector>::gradf
                (
                    surfaceVectorField
                    (
                        "UfFluxDiag",
                        UfDiag
                      + mesh.Sf()*(phi_ - (UfDiag & mesh.Sf()))
                       /magSqr(mesh.Sf())
                    ),
                    "gradUFluxDiag"
                )
            );
            const volScalarField trLFlux(tr(gradUFlux));
            scalar tFluxClot = 0;
            const volScalarField trL(tr(gradU));
            const volScalarField magL(mag(gradU));
            const volScalarField divPhi(fvc::div(phi_));

            scalar tClot = 0, tInt = 0, tBlood = 0;
            scalar fClot = 0, fInt = 0, fBlood = 0;
            scalar dPhiMax = 0;

            // MAXIMA CANNOT DISTINGUISH ONE BAD CELL FROM A BAD FIELD, and
            // every number on this line used to be a maximum.
            //
            // traceFracBlood read 1.41352 against the 2-D ceiling sqrt(2) =
            // 1.41421 in run 20260823-052707 -- 99.95% of the ceiling in the
            // BLOOD, where there is no elastic force at all -- and that single
            // number has been carried through several runs as evidence of
            // "98% spurious dilatation". It is the maximum over roughly 11000
            // cells. One cell sets it, and nothing printed here could tell us
            // whether it was one cell or all of them. That is the long-standing
            // debt entry "trace_frac* are maxima not magnitude-weighted".
            //
            // Added: a VOLUME-WEIGHTED mean of |tr L| and a volume-weighted RMS
            // of the fraction per bin, plus the fraction of each bin's volume
            // above f = 1. If the weighted numbers are small while the maxima
            // are at the ceiling, the maxima are a handful of cells and every
            // reading built on them needs revisiting. If both are large, the
            // field really is dilatational and the question moves to WHY.
            //
            // Also printed: the wall-normal position of the worst blood cell.
            // A maximum sitting on a boundary row is a boundary-condition
            // artefact and not a bulk property.
            scalar vClot = 0, vInt = 0, vBlood = 0;
            scalar twClot = 0, twInt = 0, twBlood = 0;
            scalar fwClot = 0, fwInt = 0, fwBlood = 0;
            scalar vHiClot = 0, vHiInt = 0, vHiBlood = 0;
            scalar yWorstBlood = 0;

            // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
            // so every threshold below is the same threshold restated:
            // c < L becomes ind > (1 - L)/2, and the inequality turns
            // over with it. Same cells, named in the surviving field.
            const volScalarField indHold(phaseIndicator());
            const scalarField& indIn = indHold.primitiveField();
            const scalarField& trIn = trL.primitiveField();
            const scalarField& mgIn = magL.primitiveField();
            const scalarField& dpIn = divPhi.primitiveField();
            const scalarField& vol = mesh.V();
            const vectorField& cc = mesh.C();
            forAll(indIn, celli)
            {
                const scalar a = mag(trIn[celli]);
                // Scale-free: |tr L| / |L|. Guarded so a quiescent cell, where
                // both are zero, cannot report a 0/0 as a large fraction.
                const scalar f =
                    mgIn[celli] > vSmall ? a/mgIn[celli] : 0.0;
                const scalar v = vol[celli];
                dPhiMax = max(dPhiMax, mag(dpIn[celli]));
                if (indIn[celli] > 0.95)
                {
                    tClot = max(tClot, a);
                    fClot = max(fClot, f);
                    vClot += v; twClot += v*a; fwClot += v*f*f;
                    if (f > 1.0) { vHiClot += v; }
                    tFluxClot =
                        max(tFluxClot, mag(trLFlux.primitiveField()[celli]));
                }
                else if (indIn[celli] > 0.25)
                {
                    tInt = max(tInt, a);
                    fInt = max(fInt, f);
                    vInt += v; twInt += v*a; fwInt += v*f*f;
                    if (f > 1.0) { vHiInt += v; }
                }
                else
                {
                    if (f > fBlood) { yWorstBlood = cc[celli].y(); }
                    tBlood = max(tBlood, a);
                    fBlood = max(fBlood, f);
                    vBlood += v; twBlood += v*a; fwBlood += v*f*f;
                    if (f > 1.0) { vHiBlood += v; }
                }
            }
            // yWorstBlood is picked on the LOCAL maximum and cannot be reduced
            // with the global one, so it is only meaningful in serial. These
            // runs are serial; if that changes, this must become a pair
            // reduction rather than a quietly wrong number -- the mistake
            // qAtWpDev already made once.
            reduce(tClot, maxOp<scalar>());
            reduce(tInt, maxOp<scalar>());
            reduce(tBlood, maxOp<scalar>());
            reduce(fClot, maxOp<scalar>());
            reduce(fInt, maxOp<scalar>());
            reduce(fBlood, maxOp<scalar>());
            reduce(dPhiMax, maxOp<scalar>());
            reduce(tFluxClot, maxOp<scalar>());
            reduce(vClot, sumOp<scalar>());
            reduce(vInt, sumOp<scalar>());
            reduce(vBlood, sumOp<scalar>());
            reduce(twClot, sumOp<scalar>());
            reduce(twInt, sumOp<scalar>());
            reduce(twBlood, sumOp<scalar>());
            reduce(fwClot, sumOp<scalar>());
            reduce(fwInt, sumOp<scalar>());
            reduce(fwBlood, sumOp<scalar>());
            reduce(vHiClot, sumOp<scalar>());
            reduce(vHiInt, sumOp<scalar>());
            reduce(vHiBlood, sumOp<scalar>());

            const scalar wClot = vClot > vSmall ? twClot/vClot : 0.0;
            const scalar wInt = vInt > vSmall ? twInt/vInt : 0.0;
            const scalar wBlood = vBlood > vSmall ? twBlood/vBlood : 0.0;
            const scalar rClot = vClot > vSmall ? Foam::sqrt(fwClot/vClot) : 0.0;
            const scalar rInt = vInt > vSmall ? Foam::sqrt(fwInt/vInt) : 0.0;
            const scalar rBlood =
                vBlood > vSmall ? Foam::sqrt(fwBlood/vBlood) : 0.0;
            const scalar hClot = vClot > vSmall ? vHiClot/vClot : 0.0;
            const scalar hInt = vInt > vSmall ? vHiInt/vInt : 0.0;
            const scalar hBlood = vBlood > vSmall ? vHiBlood/vBlood : 0.0;

            Info<< "PHASE_CLOT_TRACE"
                << " time=" << runTime.name()
                << " trBulk=" << tClot
                << " trInterface=" << tInt
                << " trBlood=" << tBlood
                << " traceFracBulk=" << fClot
                << " traceFracInterface=" << fInt
                << " traceFracBlood=" << fBlood
                << " divPhiMax=" << dPhiMax
                << " trBulkFluxConsistent=" << tFluxClot
                << " trBulkVw=" << wClot
                << " trInterfaceVw=" << wInt
                << " trBloodVw=" << wBlood
                << " fracBulkRms=" << rClot
                << " fracInterfaceRms=" << rInt
                << " fracBloodRms=" << rBlood
                << " volFracBulkOver1=" << hClot
                << " volFracInterfaceOver1=" << hInt
                << " volFracBloodOver1=" << hBlood
                << " yWorstBlood=" << yWorstBlood
                << nl;

            // THE MOMENTUM BALANCE OVER THE CLOT, TERM BY TERM.
            //
            // ELEVEN HYPOTHESES ARE DEAD. The 1/q amplifier, the carrier
            // mismatch, the limiter weights, the diffusion time level, the
            // trace-free source, dev(gradU), the flux-consistent gradient,
            // det renormalisation, elapsed time, the face-consistent elastic
            // force, and the be advection scheme -- each named, each tested,
            // and the clot still moves at 72% of the blood. Guessing a twelfth
            // is not a method. This asks the equation directly: which force is
            // missing, or wrong-signed?
            //
            // SIGNS. incompressibleFluid writes the KINEMATIC equation and
            // solves UEqn == -grad(p), with divDevSigma on the left standing
            // for -div(tau)/rho. Rearranged, the accelerations acting on the
            // fluid are
            //
            //     ddt(U) = -div(phi,U) + lap(nuEff,U) + div(sigma_e)/rho
            //              - grad(p)
            //
            // so the four are reported with the signs that make them SUM TO
            // THE INERTIA, and the residual is that sum minus ddt(U). A
            // residual far from zero means the reported set is incomplete --
            // which is itself the answer.
            //
            // THESE ARE fvc RECONSTRUCTIONS, NOT THE ASSEMBLED MATRIX, and the
            // viscous term uses lap(nuEff,U) where the solver uses
            // divDevSigma(U). For this constant-viscosity Newtonian case those
            // agree to the boundary treatment; they are not identical, and a
            // small residual must NOT be read as a discovered imbalance.
            //
            // THE ANCHOR TERMS ARE PRINTED BECAUSE THEY SHOULD BE ZERO. On the
            // Eulerian path lambda_ (the rigid penalty) and clotDrag_ (the
            // partitioned Darcy drag) are BOTH absent from UEqn -- the only
            // thing holding the clot is div(sigma_e). Printing them makes that
            // structural fact measurable instead of a claim about the source,
            // and on the other two paths it shows what an anchored clot's
            // balance looks like.
            {
                const dimensionedScalar dtM
                (
                    "dtM", dimTime, max(runTime.deltaTValue(), vSmall)
                );
                const volVectorField aInert((U_ - U_.oldTime())/dtM);
                // NAMED EXPLICITLY. OpenFOAM would derive "div(phi,U)" from
                // the arguments anyway and that entry does exist, but
                // check_fvschemes_coverage.py refuses an unnamed div on
                // principle and it is right to: the derived name is a property
                // of the field names at the call site, and a rename elsewhere
                // would silently change which scheme this reads. Caught at
                // build time on this machine rather than on the VM.
                const volVectorField aConv
                (
                    -fvc::div(phi_, U_, "div(phi,U)")
                );
                const volVectorField aVisc
                (
                    fvc::laplacian(momentumTransport->nuEff(), U_)
                );
                // S11: the force the block applies, not a recomputation of
                // it -- identical to div(sigma)/rho under the shipped keys.
                const volVectorField aElas(skeletonElasticForceCell());
                const volVectorField aPres(-fvc::grad(p_));

                const vectorField& fI = aInert.primitiveField();
                const vectorField& fC = aConv.primitiveField();
                const vectorField& fV = aVisc.primitiveField();
                const vectorField& fE = aElas.primitiveField();
                const vectorField& fP = aPres.primitiveField();
                const scalarField& lam = lambda_.primitiveField();
                const scalarField& drg = clotDrag_.primitiveField();
                const vectorField& uu = U_.primitiveField();
                const vectorField& vsx = Vs_.primitiveField();

                // THE THIRD ANCHOR, WHICH WAS IN THE EQUATION AND NOT IN THE
                // BUDGET. momentumPredictor() adds
                // fvm::Sp(bondHoldPenalty*bondHold*chi, U) on the Eulerian
                // path, and nothing below ever accounted for it: accPenalty
                // reports lambda_, which that term is not. So on every
                // eulerian_* run to date accResidual carried an in-equation
                // force as if it were an imbalance. It is broken out
                // separately rather than folded into accPenalty because the
                // eulerian_brinkman series turns bondHold OFF and Brinkman
                // ON in the same child, and two holds summed into one number
                // cannot be attributed.
                const scalarField& bh = bondHold_.primitiveField();
                const scalarField& qq = q_.primitiveField();
                const bool bondHoldInUEqn =
                 bondHoldPenalty_ > 0
                 && bondHoldDilation_ >= 0;

                // THE SATURATION OBSERVABLE. The offline analysis of run
                // 20260823-102340 found the clot is anchored and SHEARING --
                // Ux spans a factor of 19 from wall row to crest -- so
                // lockRatio, a volume-mean over a body reaching from the wall
                // into the fast core, cannot tell "held and deforming" from
                // "carried along". What is actually being driven is the shear
                // strain, and the question is whether it PLATEAUS. be_xy and
                // the shear stress it carries are therefore reported directly,
                // so the pass criterion can be a plateau in the driven
                // quantity rather than a ratio read at an arbitrary stopping
                // time.
                const volSymmTensorField sigmaE(elasticStress());
                const symmTensorField& bef = be_.primitiveField();
                const symmTensorField& sef = sigmaE.primitiveField();

                scalar vM = 0, mI = 0, mC = 0, mV = 0, mE = 0, mP = 0;
                scalar mPen = 0, mDrg = 0, xBar = 0, uxBar = 0;
                scalar mDrgR = 0;
                scalar beXY = 0, sigXY = 0, beXYmax = 0;
                scalar mBH = 0;
                // The realised drag coefficient over clot cells, so the
                // 1000x core/shell contrast is a measurement per step and
                // not an inference from k0pcore and k0pshell.
                //
                // dragFreeCells NO LONGER MEANS "the released set", and the
                // comment that said so was retired with the lab-frame term
                // on 2026-09-04. clotDrag_ is now the bare dragCoeff, which
                // is positive in every clot cell whether released or not,
                // so this count should be ZERO for the whole run. A nonzero
                // value now means something else entirely -- chi or
                // poroPerm_ collapsing inside the clot -- and is still worth
                // seeing, but it is no longer the release gate's assertion.
                //
                // The release gate is asserted where it belongs instead:
                // PHASE_CLOT_RELEASED prints cells/supported/bondSeeds every
                // step, and accDragReaction is nonzero if and only if
                // something is released AND moving.
                scalar dragMin = great, dragMax = 0;
                label dragFree = 0, clotCellsM = 0;
                forAll(indIn, celli)
                {
                    if (indIn[celli] <= 0.95) { continue; }
                    const scalar v = vol[celli];
                    vM += v;
                    ++clotCellsM;
                    dragMin = min(dragMin, drg[celli]);
                    dragMax = max(dragMax, drg[celli]);
                    if (drg[celli] <= 0) { ++dragFree; }
                    mI += v*fI[celli].x();
                    mC += v*fC[celli].x();
                    mV += v*fV[celli].x();
                    mE += v*fE[celli].x();
                    mP += v*fP[celli].x();
                    // Both enter UEqn as fvm::Sp(coeff, U), i.e. an
                    // acceleration of -coeff*U.
                    mPen += v*(-lam[celli]*uu[celli].x());
                    mDrg += v*(-drg[celli]*uu[celli].x());
                    // THE OTHER HALF OF THE RELATIVE-VELOCITY DRAG.
                    // momentumPredictor() assembles the pair
                    //   +fvm::Sp(dragCoeff, U) - dragCoeff*skeletonVelocity
                    // and mDrg above is only the implicit half. Summing the
                    // budget without this term would report a force the
                    // solver did not apply, on released material, and
                    // accResidual would open up by exactly this amount and
                    // be blamed on the discretisation.
                    //
                    // It is ALSO the momentum the body takes from the fluid
                    // and never gives back, because Ubody is a kinematic
                    // estimator rather than a force balance. Reported on its
                    // own so that non-conservation is a measured number
                    // instead of a caveat in a comment.
                    //
                    // ON THE TWO-VELOCITY PATH THE SKELETON VELOCITY IS Vs,
                    // NOT released*Ubody, and the drag is not in UEqn at all
                    // -- it is the off-diagonal of the 2x2 block, which
                    // applies exactly -clotDrag*(U - Vs). mDrg above is still
                    // the -clotDrag*U half; this is the +clotDrag*Vs half.
                    //
                    // Left as released*Ubody it would read ZERO on every run
                    // where nothing has released, so the budget would be short
                    // by the whole +K*Vs term and accResidual would open up
                    // and be blamed on the discretisation. That is the exact
                    // failure this comment block was written to prevent, one
                    // formulation later.
                    mDrgR += v*
                    (
                      drg[celli]*vsx[celli].x()
                    );
                    // Same expression as the fvm::Sp in momentumPredictor():
                    // bondHoldPenalty_*bondHold_*max(1 - q, 0).
                    mBH += v*
                    (
                       -bondHoldPenalty_*bh[celli]
                       *max(1.0 - qq[celli], 0.0)*uu[celli].x()
                    );
                    xBar += v*cc[celli].x();
                    uxBar += v*uu[celli].x();
                    beXY += v*bef[celli].xy();
                    sigXY += v*sef[celli].xy();
                    beXYmax = max(beXYmax, mag(bef[celli].xy()));
                }
                reduce(vM, sumOp<scalar>());
                reduce(mI, sumOp<scalar>());
                reduce(mC, sumOp<scalar>());
                reduce(mV, sumOp<scalar>());
                reduce(mE, sumOp<scalar>());
                reduce(mP, sumOp<scalar>());
                reduce(mPen, sumOp<scalar>());
                reduce(mDrg, sumOp<scalar>());
                reduce(mBH, sumOp<scalar>());
                reduce(dragMin, minOp<scalar>());
                reduce(dragMax, maxOp<scalar>());
                reduce(dragFree, sumOp<label>());
                reduce(clotCellsM, sumOp<label>());
                reduce(xBar, sumOp<scalar>());
                reduce(uxBar, sumOp<scalar>());
                reduce(beXY, sumOp<scalar>());
                reduce(sigXY, sumOp<scalar>());
                reduce(beXYmax, maxOp<scalar>());

                if (vM > vSmall)
                {
                    mI /= vM; mC /= vM; mV /= vM; mE /= vM; mP /= vM;
                    mPen /= vM; mDrg /= vM; xBar /= vM; uxBar /= vM;
                    mDrgR /= vM;
                    beXY /= vM; sigXY /= vM; mBH /= vM;
                }
                if (clotCellsM == 0) { dragMin = 0; }

                // RUN 20260823-102340 PAID FOR THIS LINE. mPen and mDrg were
                // summed unconditionally, but lambda_ and clotDrag_ are
                // nonzero FIELDS even on the Eulerian path, where neither is
                // in UEqn (fvm::Sp(lambda_, U_) is in the else branch only).
                // accPenalty read -590.02 for mom_eul and dragged accSum and
                // accResidual with it, so both totals had to be recomputed by
                // hand from the printed terms. The anchors are still PRINTED
                // -- that they are nonzero as fields while absent from the
                // equation is exactly the structural fact worth seeing -- but
                // they enter the sum only on the path that assembles them.
                //
                // THREE FLAGS, NOT ONE, AND THE PLAN SAID SO BEFORE THE CODE
                // WAS WRITTEN. anchorsInUEqn used to gate mPen and mDrg
                // together, which was correct only while the two were always
                // in or out together -- true on the skeleton path, where both
                // are assembled, and true on the Eulerian path, where neither
                // was. eulerianBrinkman breaks that: mDrg is in the equation
                // while lambda_ is not, and bondHold is in the equation on a
                // switch of its own. Reusing one flag for terms that can now
                // move independently is how a budget silently starts summing
                // a force the solver never applied.
                // mDrgR RIDES ON brinkmanInUEqn AND NOTHING ELSE, for the
                // reason the block above gives about three flags rather than
                // one: the explicit reaction exists only in the branch that
                // assembles the relative-velocity pair. On the skeleton path
                // the equivalent term is -dragCoeff*skeletonVelocity built
                // from Us_, which is inside that branch's own assembly and
                // is not this field.
                const scalar mAnchor =
                    0
                  + (mDrg + mDrgR)
                  + (bondHoldInUEqn ? mBH : 0);
                const scalar mSum = mC + mV + mE + mP + mAnchor;

                Info<< "PHASE_CLOT_MOMENTUM"
                    << " time=" << runTime.name()
                    << " accInertia=" << mI
                    << " accConvection=" << mC
                    << " accViscous=" << mV
                    << " accElastic=" << mE
                    << " accPressure=" << mP
                    << " accPenalty=" << mPen
                    << " accDrag=" << mDrg
                    << " accDragReaction=" << mDrgR
                    << " UbodyX=" << UbodyCached_.x()
                    << " accBondHold=" << mBH
                    << " anchorsInUEqn=" << 0
                    << " brinkmanInUEqn=" << 1
                    << " bondHoldInUEqn=" << (bondHoldInUEqn ? 1 : 0)
                    << " dragMin=" << dragMin
                    << " dragMax=" << dragMax
                    << " dragFreeCells=" << dragFree
                    << " clotCellsMom=" << clotCellsM
                    << " accSum=" << mSum
                    << " accResidual=" << (mSum - mI)
                    << " clotXbar=" << xBar
                    << " clotUxbar=" << uxBar
                    << " clotBeXY=" << beXY
                    << " clotSigmaXY=" << sigXY
                    << " clotBeXYmax=" << beXYmax
                    << " clotVolume=" << vM
                    << nl;
            }
        }

        // IS THE CLOT ACTUALLY HELD? The stage-2 slab gate's headline number
        // was clotU/U = 8.4e-4 -- the clot LOCKED -- and nothing in the main
        // case ever measured the equivalent. Run 20260818-100608 was read as a
        // det(be) failure for three sessions before anyone checked the
        // velocity, and Ux in the deep clot bulk turned out to be 7.3e-3 m/s
        // against a mean flow of 8.33e-3: the clot was flowing at 88% of the
        // free stream while the assessor reported "peak stretch 4.63" and
        // called it deformation.
        //
        // An elastic solid bonded to a no-slip wall reaches a FINITE strain and
        // then stops. A lockRatio that stays near 1 means it never stops, and
        // no amount of work on the be equation can fix a clot that is simply
        // being carried away. This is the number that says which of those is
        // happening, and it costs one reduction per step.
        //
        // Bulk clot is c < -0.9 and blood is c > 0.5, the same bins the trace
        // diagnostic uses, so the two can be read against each other.
        {
            const volScalarField magU(mag(U_));
            // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
            // so every threshold below is the same threshold restated:
            // c < L becomes ind > (1 - L)/2, and the inequality turns
            // over with it. Same cells, named in the surviving field.
            const volScalarField indHold(phaseIndicator());
            const scalarField& indIn = indHold.primitiveField();
            const scalarField& uIn = magU.primitiveField();
            const scalarField& vIn = mesh.V();

            scalar clotSum = 0, clotVol = 0, clotMax = 0;
            scalar bloodSum = 0, bloodVol = 0;
            label clotCells = 0;
            forAll(indIn, celli)
            {
                if (indIn[celli] > 0.95)
                {
                    clotSum += uIn[celli]*vIn[celli];
                    clotVol += vIn[celli];
                    clotMax = max(clotMax, uIn[celli]);
                    ++clotCells;
                }
                else if (indIn[celli] < 0.25)
                {
                    bloodSum += uIn[celli]*vIn[celli];
                    bloodVol += vIn[celli];
                }
            }
            reduce(clotSum, sumOp<scalar>());
            reduce(clotVol, sumOp<scalar>());
            reduce(bloodSum, sumOp<scalar>());
            reduce(bloodVol, sumOp<scalar>());
            reduce(clotMax, maxOp<scalar>());
            reduce(clotCells, sumOp<label>());

            // VOLUME-WEIGHTED, not a cell average: the bins have very different
            // cell counts and an unweighted mean would drift as the clot loses
            // cells, which is itself one of the things being watched here.
            const scalar clotU = clotVol > vSmall ? clotSum/clotVol : 0.0;
            const scalar bloodU = bloodVol > vSmall ? bloodSum/bloodVol : 0.0;

            Info<< "PHASE_CLOT_LOCK"
                << " time=" << runTime.name()
                << " clotUMean=" << clotU
                << " clotUMax=" << clotMax
                << " bloodUMean=" << bloodU
                << " lockRatio=" << (bloodU > vSmall ? clotU/bloodU : 0.0)
                << " clotCells=" << clotCells
                << nl;
        }
    }

    // Platelet runaway watch. UNCONDITIONAL, unlike the lock diagnostic above,
    // because the failure it catches is silent by construction.
    //
    // w is a concentration whose physical bound is ~1.1 (RESEARCH_STATE
    // 2026-07-30 records w_max=1.64931 against it as a preserved failure). In
    // run 20260818-232055 it reached 6372 and no gate noticed, because nothing
    // was watching it. That matters beyond w itself: Vn = alpha*(n & grad(w))
    // feeds the phase-field source as -Vn*phaseGradient, so an unbounded w
    // becomes an unbounded interface velocity (0.61 m/s against a 0.01 m/s
    // flow) and then cell-scale noise in c. The visible artefact is three
    // steps downstream of the quantity that is actually diverging.
    //
    // The bound is a property of the model, not of a case, so it is a local
    // constant rather than a dictionary entry; it is a diagnostic threshold
    // and changes no equation.
    {
        const scalar plateletBound = 1.1;

        const volScalarField magVn(mag(Vn_));
        const scalarField& wIn = w_.primitiveField();
        const scalarField& vnIn = magVn.primitiveField();

        scalar wMax = -great, wMin = great, vnMax = 0;
        label wOverBound = 0;

        // THE ABSOLUTE DEVIATION, AND WHY IT IS REPORTED ALONGSIDE THE RATIO.
        //
        // For positive q above qFloor, |w - 1| is |Wp - q| DIVIDED BY q.
        // Outside that domain the implemented denominator is max(|q|,qFloor),
        // and the historical WpDevMax uses that EFFECTIVE reference, not raw
        // q. Schema 2 below reports the two separately without redefining old
        // keys. In the deep clot q reaches 2e-5 (run 20260822-085149), so the
        // ratio carries a 5e4 amplification that has nothing to do with the
        // transport being tested: max|w - 1| = 2.8e-3 there corresponds to an
        // absolute discrepancy of 5.6e-8 in the conserved quantity.
        //
        // gate_a_exact was specified at 1e-10 on the RATIO. That demanded
        // |Wp - q| ~ 1e-15 -- below the linear solver's own `tolerance 1e-12`,
        // so it was a threshold beneath the resolution of the instrument and
        // could not have passed however correct the discretisation was. That
        // failure mode has now happened four times in this project and this
        // diagnostic exists so the gate can be written on the quantity that
        // is actually conserved.
        //
        // Reported, NOT substituted: the ratio is still what feeds Vn through
        // grad(w), so a large |w - 1| in a low-q cell remains a real hazard
        // even when |Wp - q| is tiny. Both numbers are needed and neither
        // alone is the answer.
        // mag(q_): see wGrowth.
        const volScalarField qEffDiag(max(mag(q_), qFloor_));
        const scalarField& qIn = qEffDiag.primitiveField();
        const scalarField& rawQIn = q_.primitiveField();
        const scalarField& WpIn = Wp_.primitiveField();
        scalar wpDevMax = 0, wDevRes = 0, qMinDiag = great;
        scalar wpRawQDevMax = 0, wDevRawBlood = 0, rawQMin = great;
        label rawBloodCells = 0;

        forAll(wIn, celli)
        {
            wMax = max(wMax, wIn[celli]);
            wMin = min(wMin, wIn[celli]);
            vnMax = max(vnMax, vnIn[celli]);
            if (wIn[celli] > plateletBound)
            {
                ++wOverBound;
            }

            wpDevMax = max(wpDevMax, mag(WpIn[celli] - qIn[celli]));
            // Keep the historical effective-denominator metric unchanged.
            // Raw q is a distinct reference: q=Wp=-0.7 gives zero here but
            // 1.4 above. Neither value alone certifies physical admissibility.
            wpRawQDevMax = max
            (
                wpRawQDevMax,
                mag(WpIn[celli] - rawQIn[celli])
            );
            rawQMin = min(rawQMin, rawQIn[celli]);
            if (rawQIn[celli] >= 0.5)
            {
                ++rawBloodCells;
                wDevRawBlood = max(wDevRawBlood, mag(wIn[celli] - 1.0));
            }

            // NOT "q at the cell where the deviation peaks". That cannot be
            // reduced across processors -- the master would print ITS local
            // peak's q while wpDevMax came from another rank, and the two
            // numbers would silently describe different cells. The domain
            // minimum reduces correctly and is what the ratio needs to be
            // read against anyway.
            qMinDiag = min(qMinDiag, qIn[celli]);

            // The ratio restricted to cells where q is RESOLVED. If the
            // deviation survives here it is not a conditioning artefact,
            // because no amplification is available above q = 0.5.
            if (qIn[celli] >= 0.5)
            {
                wDevRes = max(wDevRes, mag(wIn[celli] - 1.0));
            }
        }
        reduce(wMax, maxOp<scalar>());
        reduce(wMin, minOp<scalar>());
        reduce(vnMax, maxOp<scalar>());
        reduce(wOverBound, sumOp<label>());
        reduce(wpDevMax, maxOp<scalar>());
        reduce(wDevRes, maxOp<scalar>());
        reduce(qMinDiag, minOp<scalar>());
        reduce(wpRawQDevMax, maxOp<scalar>());
        reduce(wDevRawBlood, maxOp<scalar>());
        reduce(rawQMin, minOp<scalar>());
        reduce(rawBloodCells, sumOp<label>());

        Info<< "PHASE_CLOT_PLATELET"
            << " time=" << runTime.name()
            << " wMax=" << wMax
            << " wMin=" << wMin
            << " VnMagMax=" << vnMax
            << " wOverBound=" << wOverBound
            << " WpDevMax=" << wpDevMax
            << " qMin=" << qMinDiag
            << " wDevBlood=" << wDevRes
            << " metricSchema=2"
            << " WpRawQDevMax=" << wpRawQDevMax
            << " rawQMin=" << rawQMin
            << " wDevRawBlood=" << wDevRawBlood
            << " rawBloodCells=" << rawBloodCells
            << nl;
    }

    // Per-phase wall clock. `step` is the whole step including everything not
    // itemised (the PIMPLE momentum/pressure solves, mesh work, I/O), so
    // `step` minus the sum of the parts is the fluid side plus overhead --
    // reported as `other` rather than left for the reader to subtract.
    const scalar tStep = phaseClotWallNow() - tStepStart_;
    const scalar tParts =
        tPhaseField_ + tPlatelets_ + tSkeleton_ + tDamage_
      + tPermeability_ + tDerived_ + tRelease_;

    Info<< "PHASE_CLOT_COST"
        << " time=" << runTime.name()
        << " step=" << tStep
        << " skeleton=" << tSkeleton_
        << " damage=" << tDamage_
        << " phaseField=" << tPhaseField_
        << " platelets=" << tPlatelets_
        << " permeability=" << tPermeability_
        << " derived=" << tDerived_
        << " release=" << tRelease_
        << " other=" << (tStep - tParts)
        << " skeletonFraction=" << (tStep > SMALL ? tSkeleton_/tStep : 0.0)
        << nl;

    previousInventory_ = inventory;
}


Foam::tmp<Foam::surfaceScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::skeletonFlux() const
{
    // THE SOLVED SKELETON FLUX, not fvc::flux(Vs_).
    //
    // It used to be the reconstruction, because Vs_ was not solved and there
    // was no conservative flux to return. There is now: correctPressure()
    // produces phiSkelOwn_ from the same 2x2 block and the same pressure
    // gradient that produced the mixture flux, and their phi-weighted sum
    // reproduces phi_ identically.
    //
    // This matters for every caller. The phase-field carrier and the be
    // transport both advect with this flux, and a RECONSTRUCTED flux does not
    // satisfy the discrete continuity the pressure equation just imposed --
    // so material would be created and destroyed inside the clot at the level
    // of the reconstruction error, silently.
    //
    // No-penetration through walls is already enforced on phiSkelOwn_ where it
    // is built, for the sloping throat wall, so it is not re-applied here.
    return surfaceScalarField::New("phiSkeleton", phiSkelOwn_);
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::forceFraction() const
{
    // THE FRACTION THE NETWORK-PRESSURE FORCE IS EVALUATED WITH.
    //
    // Under "content" this is the SAME clamp(theta) the momentum weights use,
    // with the same [0, 0.9] bounds, so the force and the weight cannot
    // disagree about what the solid fraction is. That disagreement is exactly
    // what migrating the weights alone created: the skeleton row of the 2x2
    // block carried -phi_s*grad(p) with phi_s = theta while F_s carried
    // -grad(phi_s p_n) with phi_s = solidFraction().
    //
    // Deliberately NOT phiSkel_, even though that field holds the same value.
    // phiSkel_ is refreshed once per step inside momentumPredictor() and is
    // constructed as ZERO, so a caller that runs earlier -- networkPressure()
    // reaches elasticStress() well before the momentum predictor on the first
    // step -- would evaluate the law at phi_s = 0 and report a network
    // pressure of zero for a step. Recomputed from thetaC_/thetaH_ here so it
    // is correct whenever it is called.
    {
        return volScalarField::New
        (
            "phiSkelForce",
            min
            (
                max
                (
                    thetaC_ + thetaH_,
                    dimensionedScalar("zero", dimless, 0)
                ),
                dimensionedScalar("cap", dimless, 0.9)
            )
        );
    }
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::phaseIndicator() const
{
    // THE INDICATOR, NOT THE FRACTION. These are two different quantities
    // that differ by theta_ref = skeletonVolumeFraction_, and the eleven
    // inline chi sites of the consumer map consume THIS one. Returning
    // theta here instead of theta/theta_ref would rescale the deviatoric
    // elastic stress, the damage driver, the Kres ramp, the isSkel threshold
    // and the bond hold by 0.45 in one edit, with nothing downstream to
    // notice.
    //
    // NOT CLAMPED ABOVE. Both bases already exceed 1 -- chi at 1.0021786 from
    // the c undershoot, theta/theta_ref at 1.0021686 from compression past
    // the anchor -- and the decision is to publish that rather than hide it.
    // PHASE_CLOT_PHASE_INDICATOR reports indMax and indAboveOne every step so
    // the overshoot is measured at the one place it is produced. The single
    // site where ind > 1 is genuinely unsafe is the material-carrier blend,
    // chif*phiS + (1 - chif)*phi_, and that site keeps its own min(.., 1):
    // this accessor sets the BASIS of the ratio, never a consumer's bounds.
    //
    // CLAMPED BELOW, and that is not the same decision. A negative indicator
    // reaches fvm::Sp on the momentum and skeleton diagonals through
    // holdChi, and the "matched" content operator does drive theta to
    // -7.5e-18 in 2354 cells. max(.., 0) costs nothing where theta is already
    // non-negative and is a sign guard where it is not.
    //
    // Recomputed rather than cached for the reason forceFraction() gives:
    // elasticStress() runs before momentumPredictor() on the first step, so
    // anything refreshed inside the predictor is zero when the first caller
    // reads it.
    {
        return volScalarField::New
        (
            "phaseIndicator",
            max
            (
                thetaC_ + thetaH_,
                dimensionedScalar("zero", dimless, 0)
            )
           /dimensionedScalar
            (
                "thetaRef", dimless, skeletonVolumeFraction_
            )
        );
    }
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::phaseClotAxisymmetricFluid::solidFraction() const
{
    // phi_s = chi * skeletonVolumeFraction_, and phi_f = 1 - phi_s.
    //
    // chi = max(1-q, 0) is an INDICATOR. It reaches 0.993 in the median clot
    // cell, and 71.7% of clot cells have q < 0.05 (run 20260904-214345). Using
    // it directly as the solid volume fraction would leave phi_f ~ 0.007 --
    // blood essentially absent from the clot -- which contradicts both the
    // permeability model that lets blood seep through and He, Kim & Ku's
    // "water content more than 95% in the fibrin matrix".
    //
    // The factor is what turns an indicator into a fraction. Clamped to
    // [0, 1) so phi_f stays strictly positive: phi_f is a momentum-equation
    // weight and a zero there would make the FLUID phase degenerate, which is
    // the failure this whole construction exists to avoid.
    const volScalarField chi
    (
        max(1.0 - q_, dimensionedScalar("zero", dimless, 0))
    );

    return volScalarField::New
    (
        "phiSkel",
        min
        (
            chi*skeletonVolumeFraction_,
            dimensionedScalar("cap", dimless, 0.9)
        )
    );
}


void Foam::solvers::phaseClotAxisymmetricFluid::reportSkeletonVelocity()
{
    // DIAGNOSTIC ONLY. This function used to COMPUTE Vs_, algebraically:
    //
    //     Vs = U + isSkeleton*(fElastic - fPressure)/max(clotDrag, dragFloor)
    //
    // That closure is deleted, not disabled, and it must not come back. It
    // measured |Vs| > |U| in 88.3% of clot cells in run 20260904-214345 -- a
    // skeleton outrunning the fluid that drags it, which the fluid momentum
    // equation alone forbids, since it gives u - v = -phi_f*grad(p)/K and that
    // is downstream unconditionally.
    //
    // The reason was structural, not numerical. Solving the fluid equation and
    // EVALUATING the solid one makes their difference -- the mixture-balance
    // residual div(sigma_e) - grad(p) -- land undamped in Vs. The measurement
    // that proves it: where Vs > U the median |fElastic|/|fPress| is 0.894,
    // where Vs <= U it is 2.054. The ratio straddles 1 and the sign of Vs - U
    // follows it, which is that residual changing sign.
    //
    // Vs_ is now the unknown of tVsEqn, coupled to U_ through the 2x2 block in
    // correctPressure(). This function only reports what that solve produced.
    //
    // Published so the fields exist for post-mortem. In run 20260904-214345
    // these were AUTO_WRITE members populated only under a switch that was
    // off, so they shipped in all twenty time directories as EXACT ZEROS --
    // data-shaped and empty, and they cost the field-level diagnosis. A field
    // written whether or not it was computed is worse than one that is absent:
    // absence is visible, zeros are not.
    skelElasticForce_ = skeletonElasticForceCell();
    skelPressForce_ = phiSkel_*fvc::grad(p_);

    // TWO NUMBERS, AND ONLY ONE OF THEM IS THE GATE.
    //
    // exceedFraction is the BARE count of |Vs| > |U|, kept because 0.883 is
    // the measured baseline from run 20260904-214345 and a replacement metric
    // with no comparable history is not evidence of anything.
    //
    // exceedFractionMargin IS THE GATE, and the margin is not padding.
    // Solving the 2x2 for a FREE element with no elastic stress gives
    //
    //     u - v  ~  A_s*U/K      inertial LAG   (skeleton slower)
    //            +  phi_s*grad(p)/K   pressure PUSH  (skeleton faster)
    //
    // and A_s = phi_s/deltaT, so the two BALANCE at deltaT* = U/|grad(p)| =
    // 1e-02/1.7e+03 = 5.9e-06 s. Our timestep is adaptive around 1e-06, i.e.
    // just below that crossover.
    //
    // So the bare count does not merely run strict -- it sits ON A SIGN
    // CHANGE. It reads ~0% while deltaT stays small and climbs once deltaT
    // grows past ~6e-06 s, on a run that is entirely correct either way. A
    // gate keyed to that is flaky, which is worse than one that is tight.
    //
    // Across deltaT from 1e-07 to 3e-05 the free-element ratio stays inside
    // 0.70 .. 1.005, so a 10% margin never trips on it, while the pathology it
    // must catch ran a mean ratio of 1.4 and a max of 3.2.
    //
    // The |U| floor removes stagnation cells, where |Vs|/|U| is arbitrarily
    // large for no physical reason and would otherwise dominate the count.
    const scalar vsMarginGate = 1.10;
    const vectorField& vsIn = Vs_.primitiveField();
    const vectorField& UIn = U_.primitiveField();
    const scalar uRef = gMax(mag(UIn));
    const scalar uFloor = 1e-3*uRef;

    scalar sumRatio = 0;
    scalar maxRatio = 0;
    label clotCells = 0;
    label exceedCells = 0;
    label exceedMarginCells = 0;
    // S6c-2: THE PHASE INDICATOR, NOT c. c = 1 - 2*ind exactly,
    // so every threshold below is the same threshold restated:
    // c < L becomes ind > (1 - L)/2, and the inequality turns
    // over with it. Same cells, named in the surviving field.
    const volScalarField indHold(phaseIndicator());
    const scalarField& indIn = indHold.primitiveField();
    const scalarField& dragIn = clotDrag_.primitiveField();
    forAll(indIn, celli)
    {
        if (indIn[celli] > 0.5 && dragIn[celli] > 1.0)
        {
            const scalar magU = mag(UIn[celli]);
            if (magU > max(uFloor, vSmall))
            {
                const scalar ratio = mag(vsIn[celli])/magU;
                sumRatio += ratio;
                maxRatio = max(maxRatio, ratio);
                ++clotCells;
                if (ratio > 1.0)
                {
                    ++exceedCells;
                }
                if (ratio > vsMarginGate)
                {
                    ++exceedMarginCells;
                }
            }
        }
    }
    reduce(sumRatio, sumOp<scalar>());
    reduce(maxRatio, maxOp<scalar>());
    reduce(clotCells, sumOp<label>());
    reduce(exceedCells, sumOp<label>());
    reduce(exceedMarginCells, sumOp<label>());

    // WHERE THE WHOLE-FIELD MAXIMUM IS, AND WHETHER IT IS EVEN IN THE CLOT.
    //
    // maxVs below is gMax over the WHOLE field, while meanVsOverU,
    // maxVsOverU, exceedFraction and exceedFractionMargin are computed over
    // CLOT CELLS ONLY (c < 0 and drag > 1). Run 20260905-191135 diverged with
    // maxVs at 0.62 m/s while every clot-cell metric read normal, and nothing
    // in the log could say the runaway was in the LUMEN -- which it was.
    // These four fields make that unambiguous: if maxVsInClot reads 0, the
    // two-velocity gate is not measuring the thing that is failing.
    scalar maxVsClot = 0;
    scalar maxVsAny = -1;
    vector maxVsPos = Zero;
    label maxVsIsClot = 0;
    // S23 (G23-1, recoil-aware): |Vs - U| at the same whole-field max-|Vs|
    // cell, so a step whose maximum sits in a recoil window is judged exactly.
    scalar maxVsSlip = 0;
    {
        const vectorField& cc = mesh_.C().primitiveField();
        forAll(vsIn, celli)
        {
            const scalar m = mag(vsIn[celli]);
            const bool inClot = (indIn[celli] > 0.5 && dragIn[celli] > 1.0);
            if (inClot && m > maxVsClot)
            {
                maxVsClot = m;
            }
            if (m > maxVsAny)
            {
                maxVsAny = m;
                maxVsPos = cc[celli];
                maxVsIsClot = inClot ? 1 : 0;
                maxVsSlip = mag(vsIn[celli] - UIn[celli]);
            }
        }
        reduce(maxVsClot, maxOp<scalar>());
    }

    Info<< "PHASE_CLOT_SKELETON_VELOCITY time=" << runTime.userTimeValue()
        << " clotCells=" << clotCells
        << " meanVsOverU=" << (clotCells ? sumRatio/scalar(clotCells) : 0.0)
        << " maxVsOverU=" << maxRatio
        << " exceedFraction="
        << (clotCells ? scalar(exceedCells)/scalar(clotCells) : 0.0)
        << " exceedFractionMargin="
        << (clotCells ? scalar(exceedMarginCells)/scalar(clotCells) : 0.0)
        << " vsMargin=" << vsMarginGate
        << " maxVs=" << gMax(mag(vsIn))
        << " maxU=" << gMax(mag(UIn))
        << " maxPhiS=" << gMax(phiSkel_.primitiveField())
        << " maxVsClot=" << maxVsClot
        << " maxVsInClot=" << maxVsIsClot
        << " maxVsX=" << maxVsPos.x()
        << " maxVsY=" << maxVsPos.y()
        << " maxVsSlip=" << maxVsSlip
        << endl;

    // THE DILUTE BAND, REPORTED SO IT DOES NOT HAVE TO BE RECOVERED FROM THE
    // WRITTEN FIELDS AGAIN.
    //
    // Run 20260907-111028 died of a skeleton velocity at 3.90x max|U| in cells
    // that were 98% blood, and NOTHING in the diagnostics above showed it:
    // maxVsInClot read 0 at every sample of that run and of its predecessor,
    // because the clot mask does not contain the cells that run away. Reading
    // it cost a full field-extraction pass over the returned tarball.
    //
    // The band is chi in [0.005, 0.1] -- measured, not chosen: |Vs|/max|U| is
    // 1.00 below it, 3.4-3.9 inside it, and back under 1 above chi ~ 0.3.
    // compliance = max(chi, skeletonSolidFloor)/(Kd + Kres) is the gain from
    // elastic stress into Vs and is the quantity the drag floor bounds; with
    // skeletonDiluteDragRate = k1 it cannot exceed 1/k1 where the floor binds.
    //
    // Parallel-correct: every number below is a gMax or a gSum, unlike
    // PHASE_CLOT_LUMEN_WORST which is a rank-local argmax and says so.
    {
        const scalarField& qIn = q_.primitiveField();
        const scalarField& KdIn = clotDrag_.primitiveField();
        const scalarField magVs(mag(vsIn));
        const scalar dtVal = runTime.deltaTValue();
        const scalar maxUAll = gMax(mag(UIn));

        label bandCells = 0;
        scalar bandMaxVs = 0.0;
        // THE SLIP, |Vs - U|, IS THE QUANTITY THE DRAG FLOOR ACTUALLY SETS.
        // Measured on run 20260907-204203 across K spanning 685 to 8.3e+06 --
        // a factor of 12000 in one run at one instant -- the balance closes as
        //     |Vs - U| * K = 0.95 * chi * |grad p|
        // with the coefficient between 0.92 and 1.24 throughout. Where the
        // floor binds, K = k1*chi, the chi CANCELS, and the slip is
        // 0.95*|grad p|/k1 regardless of how little skeleton is present. That
        // is the number to read, and reconstructing it from the written fields
        // cost a full extraction pass once already.
        scalar bandMaxSlip = 0.0;
        scalar bandMaxCompliance = 0.0;
        scalar bandMinDrag = great;
        label flooredCells = 0;

        forAll(qIn, celli)
        {
            const scalar chiC = max(1.0 - qIn[celli], 0.0);
            if (chiC < 0.005 || chiC > 0.1)
            {
                continue;
            }
            ++bandCells;
            bandMaxVs = max(bandMaxVs, magVs[celli]);
            bandMaxSlip = max(bandMaxSlip, mag(vsIn[celli] - UIn[celli]));

            const scalar KresC =
                residualChi_ > 0
              ? (residualPhiS_/residualChi_)*max(residualChi_ - chiC, 0.0)/dtVal
              : max(residualPhiS_ - chiC*skeletonVolumeFraction_, 0.0)/dtVal;

            const scalar Ktot = KdIn[celli] + KresC;
            bandMinDrag = min(bandMinDrag, Ktot);
            if (Ktot > 0)
            {
                bandMaxCompliance =
                    max(bandMaxCompliance, max(chiC, skeletonSolidFloor_)/Ktot);
            }
            if
            (
                skeletonDiluteDragRate_ > 0
             && KdIn[celli] >= skeletonDiluteDragRate_*chiC*(1.0 - 1e-6)
            )
            {
                ++flooredCells;
            }
        }

        reduce(bandCells, sumOp<label>());
        reduce(flooredCells, sumOp<label>());
        reduce(bandMaxVs, maxOp<scalar>());
        reduce(bandMaxSlip, maxOp<scalar>());
        reduce(bandMaxCompliance, maxOp<scalar>());
        reduce(bandMinDrag, minOp<scalar>());

        Info<< "PHASE_CLOT_DILUTE time=" << runTime.userTimeValue()
            << " band=chi0.005-0.1"
            << " cells=" << bandCells
            << " maxVsBand=" << bandMaxVs
            << " maxVsBandOverMaxU="
            << (maxUAll > 0 ? bandMaxVs/maxUAll : 0.0)
            << " maxSlipBand=" << bandMaxSlip
            << " maxSlipBandOverMaxU="
            << (maxUAll > 0 ? bandMaxSlip/maxUAll : 0.0)
            << " maxCompliance=" << bandMaxCompliance
            << " minDragRate=" << (bandCells ? bandMinDrag : 0.0)
            << " flooredCells=" << flooredCells
            << " k1=" << skeletonDiluteDragRate_
            << " complianceBound="
            << (skeletonDiluteDragRate_ > 0
                    ? 1.0/skeletonDiluteDragRate_
                    : 0.0)
            << endl;
    }
}


void Foam::solvers::phaseClotAxisymmetricFluid::reportIntactReference()
{
    // D74 (research/designs/2026-10-09-D74-passive-intact-reference.md rev.
    // 2 sections 2 and 4): ONE LINE PER STEP, at the end of the step (after
    // the transport, growth, rupture and deletion), on ref_I. theta_I is
    // (P1/S) theta, not P1 (numerics 35: P1 differs from theta on d* = 0
    // cells at theta ~ 1e-290), with the shares of fibreReleaseKernel::shares
    // on S > 0. a_I = phi0 + theta_I - ref_I; its proxy reads (P1/S) ref.
    //   aIMax*       max a_I over S > 0, theta_I > 0, with its cell and that
    //                cell's theta_I, ref_I and d*; nearPole counts a_I >
    //                phiSMax - 0.05 there;
    //   proxyDev*    (a_I^proxy - a_I) over 0 < d* < 1, S > 0: p99 and max,
    //                and p01 and min (the SIGNED deviation: drift TOWARD the
    //                pole, true a_I above the proxy, is the min side; CE
    //                ruling 2026-10-09, solver-engineer 39);
    //   the INVARIANTS (section 4): refIMin and excessMax (= max ref_I - ref)
    //   against refMax (GATED at 1e-12 refMax, absolute); p1ZeroRefIMax over
    //   P1 = 0, S > 0 (GATED); s0RefCells (S = 0, ref > 0, counted apart);
    //   d* = 0 cells, those with ref_I != ref and their max relative
    //   deviation over ref > 0 (REPORTED); and the label balance
    //   dSum - (transfer - dt outflux + spSource) (REPORTED).
    // Serial, as S23's quantiles: the p99 and the cell are this process's.
    const scalarField& tCR = thetaC_.primitiveField();
    const scalarField& tHR = thetaH_.primitiveField();
    const scalarField& P1R = thetaIntact_.primitiveField();
    const scalarField& P2R = thetaPending_.primitiveField();
    const scalarField& P3R = thetaBroken_.primitiveField();
    const scalarField& refR = thetaNetRef_.primitiveField();
    const scalarField& refIR = thetaNetRefIntact_.primitiveField();
    const scalarField& VR = mesh_.V();
    const scalar phi0R = skeletonVolumeFraction_;
    const scalar poleBand = networkPole() - 0.05;

    label aICells = 0;
    label aIMaxCell = -1;
    scalar aIMax = 0;
    scalar aIMaxThetaI = 0;
    scalar aIMaxRefI = 0;
    scalar aIMaxDStar = 0;
    label nearPole = 0;
    std::vector<scalar> proxyDev;
    proxyDev.reserve(refIR.size());
    label p1ZeroCells = 0;
    scalar p1ZeroRefIMax = 0;
    label s0RefCells = 0;
    label dStar0Cells = 0;
    label dStar0Unequal = 0;
    scalar dStar0RelDevMax = 0;
    forAll(refIR, celli)
    {
        const scalar S = P1R[celli] + (P2R[celli] + P3R[celli]);
        if (!(S > 0))
        {
            if (refR[celli] > 0) { ++s0RefCells; }
            continue;
        }
        double aIs = 1, aPs = 0, omdS = 1;
        fibreReleaseKernel::shares
        (
            P1R[celli], P2R[celli], P3R[celli], aIs, aPs, omdS
        );
        const scalar dStar = (P2R[celli] + P3R[celli])/S;
        const scalar thetaI = aIs*(tCR[celli] + tHR[celli]);
        const scalar aI = phi0R + thetaI - refIR[celli];
        if (thetaI > 0)
        {
            if (aIMaxCell < 0 || aI > aIMax)
            {
                aIMax = aI;
                aIMaxCell = celli;
                aIMaxThetaI = thetaI;
                aIMaxRefI = refIR[celli];
                aIMaxDStar = dStar;
            }
            ++aICells;
            if (aI > poleBand) { ++nearPole; }
        }
        if (dStar > 0 && dStar < 1)
        {
            const scalar aIProxy = phi0R + thetaI - aIs*refR[celli];
            proxyDev.push_back(aIProxy - aI);
        }
        if (P1R[celli] == 0)
        {
            ++p1ZeroCells;
            p1ZeroRefIMax = max(p1ZeroRefIMax, mag(refIR[celli]));
        }
        if (P2R[celli] + P3R[celli] == 0)
        {
            ++dStar0Cells;
            if (refIR[celli] != refR[celli])
            {
                ++dStar0Unequal;
                if (refR[celli] > 0)
                {
                    dStar0RelDevMax = max
                    (
                        dStar0RelDevMax,
                        mag(refIR[celli] - refR[celli])/refR[celli]
                    );
                }
            }
        }
    }
    scalar proxyDevP99 = 0;
    scalar proxyDevMax = 0;
    scalar proxyDevP01 = 0;
    scalar proxyDevMin = 0;
    const label proxyDevCells = label(proxyDev.size());
    if (!proxyDev.empty())
    {
        const std::size_t nD = proxyDev.size();
        const std::size_t k99 =
            std::min(nD - 1, std::size_t(0.99*scalar(nD - 1) + 0.5));
        std::nth_element(proxyDev.begin(), proxyDev.begin() + k99, proxyDev.end());
        proxyDevP99 = proxyDev[k99];
        proxyDevMax = *std::max_element(proxyDev.begin(), proxyDev.end());
        const std::size_t k01 =
            std::min(nD - 1, std::size_t(0.01*scalar(nD - 1) + 0.5));
        std::nth_element(proxyDev.begin(), proxyDev.begin() + k01, proxyDev.end());
        proxyDevP01 = proxyDev[k01];
        proxyDevMin = *std::min_element(proxyDev.begin(), proxyDev.end());
    }
    reduce(aICells, sumOp<label>());
    reduce(nearPole, sumOp<label>());
    reduce(p1ZeroCells, sumOp<label>());
    reduce(p1ZeroRefIMax, maxOp<scalar>());
    reduce(s0RefCells, sumOp<label>());
    reduce(dStar0Cells, sumOp<label>());
    reduce(dStar0Unequal, sumOp<label>());
    reduce(dStar0RelDevMax, maxOp<scalar>());

    const scalar refMax = gMax(refR);
    const scalar refIMin = gMin(refIR);
    const scalar excessMax = gMax(refIR - refR);

    const scalar dtR = runTime.deltaTValue();
    const scalar sumNow = gSum(refIR*VR);
    const scalar dSum = sumNow - refISumStepStart_;
    const scalar transfer = returnReduce(refITransferStep_, sumOp<scalar>());
    const scalar outflux = dtR*refIBoundaryFluxStep_;
    const scalar balanceResidual =
        dSum - (transfer - outflux + refISpSourceStep_);

    Info<< "PHASE_CLOT_REFI time=" << runTime.userTimeValue()
        << " step=" << runTime.timeIndex()
        << " aICells=" << aICells
        << " aIMax=" << aIMax
        << " aIMaxCell=" << aIMaxCell
        << " aIMaxThetaI=" << aIMaxThetaI
        << " aIMaxRefI=" << aIMaxRefI
        << " aIMaxDStar=" << aIMaxDStar
        << " nearPole=" << nearPole
        << " poleBand=" << poleBand
        << " proxyDevCells=" << proxyDevCells
        << " proxyDevP99=" << proxyDevP99
        << " proxyDevMax=" << proxyDevMax
        << " proxyDevP01=" << proxyDevP01
        << " proxyDevMin=" << proxyDevMin
        << " refMax=" << refMax
        << " refIMin=" << refIMin
        << " excessMax=" << excessMax
        << " p1ZeroCells=" << p1ZeroCells
        << " p1ZeroRefIMax=" << p1ZeroRefIMax
        << " s0RefCells=" << s0RefCells
        << " dStar0Cells=" << dStar0Cells
        << " dStar0Unequal=" << dStar0Unequal
        << " dStar0RelDevMax=" << dStar0RelDevMax
        << " refISum=" << sumNow
        << " dSum=" << dSum
        << " transfer=" << transfer
        << " outflux=" << outflux
        << " spSource=" << refISpSourceStep_
        << " balanceResidual=" << balanceResidual
        << endl;
}


namespace
{
    // S25 C: the keys of uniform/phaseClotState, written and read together.
    const char* const s25StateScalars[] =
    {
        "explicitDiffusionRate",
        "s23ContentV0",
        "s23ContentResInt",
        "s23OutfluxInt"
    };
    const char* const s25StateFields[] =
    {
        "s17OutletPhi1", "s17OutletPhi2", "s17OutletPhi3",
        "s17SurfPhi1", "s17SurfPhi2", "s17SurfPhi3"
    };
    const char* const s25StateLabels[] =
    {
        "s17SecondDiffSteps",
        "s18pPresentDumped",
        "s18pWindow0", "s18pWindow1",
        "s18pWindowDumped0", "s18pWindowDumped1"
    };
}


void Foam::solvers::phaseClotAxisymmetricFluid::storeRestartState()
{
    // S25 C (design 2026-10-10-S25 section 2C): every step, so whatever
    // writes the time directory -- the write interval or a function object's
    // writeNow at the start of the next step -- writes this step's state.
    // Written at the run's writePrecision (17 on S25 children).
    dictionary& d = phaseClotState_;
    d.set("explicitDiffusionRate", explicitDiffusionRate_);
    d.set("s23ContentV0", s23ContentV0_);
    d.set("s23ContentResInt", s23ContentResInt_);
    d.set("s23OutfluxInt", s23OutfluxInt_);
    d.set("s17OutletPhi1", s17OutletPhi1_);
    d.set("s17OutletPhi2", s17OutletPhi2_);
    d.set("s17OutletPhi3", s17OutletPhi3_);
    d.set("s17SurfPhi1", s17SurfPhi1_);
    d.set("s17SurfPhi2", s17SurfPhi2_);
    d.set("s17SurfPhi3", s17SurfPhi3_);
    d.set("s17SecondDiffSteps", s17SecondDiffSteps_);
    d.set("s18pPresentDumped", s18pPresentDumped_);
    d.set("s18pWindow0", s18pWindow_[0]);
    d.set("s18pWindow1", s18pWindow_[1]);
    d.set("s18pWindowDumped0", s18pWindowDumped_[0]);
    d.set("s18pWindowDumped1", s18pWindowDumped_[1]);
}


void Foam::solvers::phaseClotAxisymmetricFluid::restoreRestartState()
{
    // S25 C: no file (a cold start, or a directory written before S25) keeps
    // every constructor seed, exactly as before. A file is read all-or-none:
    // a partial one is a defect, not a state to complete with seeds.
    const dictionary& d = phaseClotState_;
    if (d.empty())
    {
        Info<< "PHASE_CLOT_S25_RESTART state=seeded" << endl;
        return;
    }
    wordList missing;
    for (const char* k : s25StateScalars)
    {
        if (!d.found(k)) { missing.append(word(k)); }
    }
    for (const char* k : s25StateFields)
    {
        if (!d.found(k)) { missing.append(word(k)); }
    }
    for (const char* k : s25StateLabels)
    {
        if (!d.found(k)) { missing.append(word(k)); }
    }
    if (missing.size())
    {
        FatalErrorInFunction
            << "uniform/phaseClotState is read all-or-none, and it lacks "
            << missing << " (design 2026-10-10-S25 section 2C)."
            << exit(FatalError);
    }
    explicitDiffusionRate_ = d.lookup<scalar>("explicitDiffusionRate");
    s23ContentV0_ = d.lookup<scalar>("s23ContentV0");
    s23ContentResInt_ = d.lookup<scalar>("s23ContentResInt");
    s23OutfluxInt_ = d.lookup<scalar>("s23OutfluxInt");
    s17OutletPhi1_ = d.lookup<scalarField>("s17OutletPhi1");
    s17OutletPhi2_ = d.lookup<scalarField>("s17OutletPhi2");
    s17OutletPhi3_ = d.lookup<scalarField>("s17OutletPhi3");
    s17SurfPhi1_ = d.lookup<scalarField>("s17SurfPhi1");
    s17SurfPhi2_ = d.lookup<scalarField>("s17SurfPhi2");
    s17SurfPhi3_ = d.lookup<scalarField>("s17SurfPhi3");
    s17SecondDiffSteps_ = d.lookup<label>("s17SecondDiffSteps");
    s18pPresentDumped_ = d.lookup<label>("s18pPresentDumped");
    s18pWindow_[0] = d.lookup<label>("s18pWindow0");
    s18pWindow_[1] = d.lookup<label>("s18pWindow1");
    s18pWindowDumped_[0] = d.lookup<label>("s18pWindowDumped0");
    s18pWindowDumped_[1] = d.lookup<label>("s18pWindowDumped1");
    Info<< "PHASE_CLOT_S25_RESTART state=restored"
        << " explicitDiffusionRate=" << explicitDiffusionRate_
        << " s23ContentV0=" << s23ContentV0_
        << " s17SecondDiffSteps=" << s17SecondDiffSteps_
        << endl;
}


void Foam::solvers::phaseClotAxisymmetricFluid::reportLoadRatio() const
{
    // S25 E (design 2026-10-10-S25 section 2E; S25 01 Q3): O(N) reductions,
    // after the pressure solve. A ratio whose denominator or population is
    // empty is printed as `none`, never as a number.
    const volScalarField indLR(phaseIndicator());
    const scalarField& indLRI = indLR.primitiveField();
    const scalarField& pLRI = p.primitiveField();
    label clotCellsLR = 0;
    scalar pClotMax = -great, pClotMin = great;
    forAll(indLRI, celli)
    {
        if (indLRI[celli] > 0.5)
        {
            ++clotCellsLR;
            pClotMax = max(pClotMax, pLRI[celli]);
            pClotMin = min(pClotMin, pLRI[celli]);
        }
    }
    reduce(clotCellsLR, sumOp<label>());
    reduce(pClotMax, maxOp<scalar>());
    reduce(pClotMin, minOp<scalar>());
    const scalar GLR = gMax(GField_.primitiveField());
    const scalar rhoLR = rho_.value();

    // Area-weighted mean of the kinematic p on a named patch; false when
    // the patch is absent or has no area.
    const auto patchMeanP = [this](const word& name, scalar& mean)
    {
        const label patchi = mesh.boundaryMesh().findIndex(name);
        if (patchi < 0) { return false; }
        const scalar area = gSum(mesh.magSf().boundaryField()[patchi]);
        if (!(area > 0)) { return false; }
        mean =
            gSum
            (
                p.boundaryField()[patchi]*mesh.magSf().boundaryField()[patchi]
            )/area;
        return true;
    };
    scalar pInLR = 0, pOutLR = 0;
    const bool hasIn = patchMeanP("inlet", pInLR);
    const bool hasOut = patchMeanP("outlet", pOutLR);

    Info<< "PHASE_CLOT_LAMBDA time=" << runTime.userTimeValue()
        << " step=" << runTime.timeIndex()
        << " G=" << GLR
        << " clotCells=" << clotCellsLR;
    if (GLR > 0 && clotCellsLR > 0)
    {
        Info<< " lambdaClot=" << rhoLR*(pClotMax - pClotMin)/GLR;
    }
    else
    {
        Info<< " lambdaClot=none";
    }
    if (GLR > 0 && hasIn && hasOut)
    {
        Info<< " lambdaDom=" << rhoLR*(pInLR - pOutLR)/GLR;
    }
    else
    {
        Info<< " lambdaDom=none";
    }
    Info<< endl;
}


void Foam::solvers::phaseClotAxisymmetricFluid::postSolve()
{

    writeBalanceDiagnostic();

    // D74: the intact reference's per-step line, after every writer of it.
    reportIntactReference();

    // S25 E: the load ratio, after the pressure solve.
    reportLoadRatio();

    // S25 C: LAST, after every writer of the persisted scalars this step.
    storeRestartState();

    // Diagnostic only, and only when asked for. Evaluated at write times so
    // the cost is bounded by the number of write steps rather than by the
    // number of solver steps -- 60 evaluations against 19 920 on the run this
    // was built for.
}
