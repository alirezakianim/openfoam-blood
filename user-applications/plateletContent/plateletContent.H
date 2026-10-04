/*---------------------------------------------------------------------------*\
Namespace
    Foam::plateletContent

Description
    Stage 2 of the content model: the free-platelet transport operator and the
    one accepted paired transfer, in ONE place.

    THIS HEADER EXISTS SO THE GATE AND THE SOLVER CANNOT DIVERGE. It is
    included unchanged by phaseClotAxisymmetricFluid (which runs it on the real
    case) and by plateletTransportGate (which runs it on the D0-D3/T7 unit
    tests against closed-form answers). A gate that assembled its own copy of
    the operator would verify the copy, which is worth nothing. Same pattern,
    and the same reason, as ../beTransport/beTransport.H.

    THE UNKNOWN IS THE PORE CONCENTRATION nP, NOT THE BULK CONTENT NP.

    The two are related by NP = eps*nP, and the choice is not cosmetic. The
    required diffusive face flux is the neighbour form

        T_f*(N_j/eps_j - N_i/eps_i),   T_f = (eps*D_eff)_f*|S_f|/d_ij

    which, with nP as the unknown and conservative storage ddt(eps, nP), IS
    exactly fvm::laplacian(eps*D_eff, nP) -- a standard operator -- because nP
    is N/eps by storage. Solving for NP instead would need a
    variable-coefficient Laplacian on NP that OpenFOAM has no primitive for,
    and every hand-rolled version of it is a place to put the epsilon in the
    wrong spot.

    THE FLUX CONVENTION IS DECLARED, NEVER INFERRED. F_eps is the
    POROSITY-WEIGHTED fluid-volume flux and it advects the PORE concentration.
    Advecting the bulk content with the same flux is an EXACT factor of eps too
    small -- 0.55x at eps = 0.55, a 45% deficit -- and it is invisible in
    uniform fields, because both divergences are zero either way. That is why
    the convention is a validated dictionary word and why T7 uses a
    non-uniform pulse.

    REACTION IS NOT IN THE MATRIX. The contract is one accepted increment dB,
    computed from the transport trial, constrained once, and applied to both
    consumers. Folding an implicit Sp into the transport solve would make the
    accepted increment a by-product of the linear solver, and there would be no
    single array to hand to thetaH. It also makes the corrector-count
    independence test meaningful: from the same old state, 1, 2 or 4 correctors
    must all land on N0/(1 + dt*lambda), never on N0/(1 + dt*lambda)^k.

    PAIRING ALONE DOES NOT VERIFY THE TRANSFER. With Y = 0.1, N0 = 2,
    theta0 = 0.2 and dt*lambda = 0.2, the correct single uptake and the
    twice-applied one BOTH preserve Y*N + theta = 0.4 exactly. Conservation is
    necessary and nowhere near sufficient; the kinetics must be asserted
    separately.

\*---------------------------------------------------------------------------*/

#ifndef plateletContent_H
#define plateletContent_H

#include "volFields.H"
#include "surfaceFields.H"
#include "fvmDdt.H"
#include "fvmLaplacian.H"
#include "fvcSurfaceIntegrate.H"
#include "fvcAverage.H"
#include "gaussConvectionScheme.H"
#include "upwind.H"
#include "fvMatrices.H"
#include "plateletTransferKernel.H"

namespace Foam
{
namespace plateletContent
{

//- The two admissible flux conventions. There is deliberately no default.
inline bool knownConvention(const word& c)
{
    return c == "intrinsic" || c == "superficial";
}


//- F_eps from the supplied carrier, in the declared convention.
//
//  intrinsic   : carrier is u_f & Sf, so F_eps = eps_f*carrier.
//  superficial : carrier is ALREADY eps*u_f & Sf, so F_eps = carrier.
//
//  The result is NAMED, so the diagnostics and any derived field carry
//  "Feps" rather than an auto-generated "interpolate(epsP)*phi".
inline tmp<surfaceScalarField> flux
(
    const word& convention,
    const volScalarField& eps,
    const surfaceScalarField& carrier
)
{
    if (convention == "superficial")
    {
        return tmp<surfaceScalarField>
        (
            new surfaceScalarField("Feps", carrier)
        );
    }

    return tmp<surfaceScalarField>
    (
        new surfaceScalarField("Feps", fvc::interpolate(eps)*carrier)
    );
}


//- ddt(eps) + div(F_eps), the fluid-volume consistency residual.
//
//  A uniform pore concentration survives transport ONLY when this vanishes
//  discretely at the same time level. It is measured rather than assumed
//  because in the coupled solver the continuity flux is weighted by a
//  porosity built from q, not from the contents -- so a T5 claim on a case
//  where this is not small is void.
//
//  fvc::surfaceIntegrate, NOT fvc::div: fvc::div on a surfaceScalarField is
//  surfaceIntegrateExtrapolate, which differs at boundary faces, and it also
//  drags a spurious div(Feps) requirement into the fvSchemes coverage scan.
inline void volumeConsistency
(
    const volScalarField& eps,
    const surfaceScalarField& Feps,
    const scalar dt,
    scalar& resMax,
    scalar& resL1
)
{
    const fvMesh& mesh = eps.mesh();
    const scalarField dEpsDt
    (
        (eps.primitiveField() - eps.oldTime().primitiveField())/dt
    );
    const scalarField res(dEpsDt + fvc::surfaceIntegrate(Feps)().primitiveField());
    resMax = gMax(mag(res));
    resL1 = gSum(mag(res)*mesh.cellVolumes());
}


//- The transport trial: conservative storage, upwind advection, pore
//  diffusion. NO reaction, by contract.
//
//  Upwind weights are constructed here rather than looked up in fvSchemes for
//  the same reason the Stage 1 contents build theirs explicitly: the scheme is
//  part of the contract, not a per-case choice, and a limited scheme
//  recomputes its limiter from the field being transported, so nP and the
//  contents would ride different matrices built from one flux.
inline tmp<fvScalarMatrix> transportEqn
(
    const volScalarField& eps,
    volScalarField& nP,
    const surfaceScalarField& Feps,
    const dimensionedScalar& Deff
)
{
    const fvMesh& mesh = nP.mesh();

    fv::gaussConvectionScheme<scalar> upwindConv
    (
        mesh,
        Feps,
        tmp<surfaceInterpolationScheme<scalar>>
        (
            new upwind<scalar>(mesh, Feps)
        )
    );

    const surfaceScalarField epsDeff("epsDeff", fvc::interpolate(eps)*Deff);

    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            fvm::ddt(eps, nP)
          + upwindConv.fvmDiv(Feps, nP)
          - fvm::laplacian(epsDeff, nP)
        )
    );
}


//- The one accepted transfer. Computed once, constrained once, and returned
//  as a single array for BOTH consumers to use unchanged.
//
//  Built from the TRIAL state, not from oldTime: with transport on,
//  Y*N + theta is not conserved against the same cell's old-time value,
//  because platelets and solid cross different faces at different velocities
//  and diffusion moves inventory too. Only a source-only test has star == old.
//  THE RATE IS A FIELD, NOT A SCALAR, and that is the whole point.
//
//  Until 2026-09-15 this took `const scalar lambda` and applied it in EVERY
//  cell. Run 20260915-073208 measured what that does: 86.5% of all deposition
//  landed in cells that are >99% blood, at a per-volume rate 1.82x the rate in
//  the clot core, because dB ~ dt*lambda*N and N = epsP*nP is LARGEST where
//  there is no solid. The deposition was not merely unlocalised, it was
//  anti-localised.
//
//  The controlled specification AM2020-phase-clot-v1 eq. (P) puts platelet
//  uptake on a surface delta measure -- `delta_Gamma[(k(s) + V_n - u.n) w]` --
//  and the legacy path already implements exactly that
//  (`fvm::Sp(interfaceDelta_*adhesionSpeedEff_, w_)`). The localiser was never
//  missing from the model or from the code; it was missing from THIS path.
//
//  The dimensional signature is the tell: the specified rate k is m/s and
//  pairs with a delta of 1/m; the old uniform `lambda` was 1/s and paired with
//  nothing. Passing the rate as a field lets the caller supply
//  kEff [m/s] * interfaceDelta_ [1/m], which is 1/s and localised, while the
//  portable kernel below is UNCHANGED -- acceptedIncrement already took a
//  per-call scalar, so every executable T4 test remains valid.
//
//  The caller must pass interfaceDelta_ (the UNIT delta), never
//  phaseGradient_ (2x it). Swapping them breaks no dimension check and
//  silently halves or doubles the deposition; see the contract at the fields'
//  declaration in phaseClotAxisymmetricFluid.H.
//
//  SINCE S3 THE UNIT DELTA HAS TWO FORMS, and interfaceDelta_ is whichever
//  phaseDeltaBasis_ selects: deltaC_ = 0.5*|grad c|, which integrates to 1
//  exactly and by construction, or deltaTheta_ = |grad theta|/theta_ref,
//  which integrates to theta_bulk/theta_ref -- the quantity the log already
//  reports every step as indMax, measured 1.00217, so its calibration error
//  is monitored rather than assumed. This kernel is unchanged by that: it
//  takes a rate field in 1/s and does not know or care which delta built it.
inline void acceptedTransfer
(
    const volScalarField& NStar,
    const volScalarField& thetaStar,
    const scalar dt,
    const volScalarField& rate,
    const scalar Y,
    const scalar thetaMax,
    volScalarField& dB,
    label& nAvail,
    label& nCap
)
{
    nAvail = 0;
    nCap = 0;

    if (NStar.dimensions() != dimless/dimVolume
        || thetaStar.dimensions() != dimless
        || dB.dimensions() != NStar.dimensions()
        || &NStar.mesh() != &thetaStar.mesh() || &NStar.mesh() != &dB.mesh())
    {
        FatalErrorInFunction << "inconsistent transfer input dimensions or mesh"
            << exit(FatalError);
    }

    // The rate carries its own dimension check because it is the one input
    // whose meaning changed. A field in m/s or 1/m reaching here is a caller
    // that forgot to pair kEff with the delta, and it must not be silently
    // multiplied by dt.
    if (rate.dimensions() != dimless/dimTime)
    {
        FatalErrorInFunction
            << "transfer rate must have dimensions 1/s; found "
            << rate.dimensions()
            << ". Pass kEff [m/s] * interfaceDelta [1/m], not either alone."
            << exit(FatalError);
    }
    if (&rate.mesh() != &NStar.mesh())
    {
        FatalErrorInFunction << "transfer rate is on a different mesh"
            << exit(FatalError);
    }

    scalarField& dBi = dB.primitiveFieldRef();
    const scalarField& NSi = NStar.primitiveField();
    const scalarField& thSi = thetaStar.primitiveField();
    const scalarField& rateI = rate.primitiveField();
    forAll(dBi, celli)
    {
        const auto increment = plateletTransferKernel::acceptedIncrement
        (
            NSi[celli], thSi[celli], dt, rateI[celli], Y, thetaMax
        );
        if (increment.error)
        {
            FatalErrorInFunction << increment.error << " in cell " << celli
                << ": NStar=" << NSi[celli] << " thetaStar=" << thSi[celli]
                << exit(FatalError);
        }
        nCap += increment.capacityLimited;
        nAvail += increment.availabilityLimited;
        dBi[celli] = increment.amount;
    }

    reduce(nAvail, sumOp<label>());
    reduce(nCap, sumOp<label>());
    dB.correctBoundaryConditions();
}


//- Apply one accepted array to both consumers. Yield is volume-valued, not a
//  dimensionless scalar multiplying a number-density field. Both output
//  fields are committed only after every cell passes the input checks.
inline void applyAcceptedTransfer
(
    const volScalarField& NStar,
    const volScalarField& thetaCStar,
    const volScalarField& thetaHStar,
    const volScalarField& dB,
    const scalar Y,
    const scalar thetaMax,
    volScalarField& NNew,
    volScalarField& thetaHNew
)
{
    if (NStar.dimensions() != dimless/dimVolume
        || dB.dimensions() != NStar.dimensions()
        || NNew.dimensions() != NStar.dimensions()
        || thetaCStar.dimensions() != dimless
        || thetaHStar.dimensions() != dimless
        || thetaHNew.dimensions() != dimless
        || &NStar.mesh() != &thetaCStar.mesh()
        || &NStar.mesh() != &thetaHStar.mesh()
        || &NStar.mesh() != &dB.mesh()
        || &NStar.mesh() != &NNew.mesh()
        || &NStar.mesh() != &thetaHNew.mesh())
    {
        FatalErrorInFunction << "inconsistent platelet transfer dimensions"
            << exit(FatalError);
    }
    scalarField newN(NStar.primitiveField());
    scalarField newH(thetaHStar.primitiveField());
    forAll(newN, celli)
    {
        const auto state = plateletTransferKernel::applyIncrement
        (
            NStar[celli], thetaCStar[celli], thetaHStar[celli], dB[celli],
            Y, thetaMax
        );
        if (state.error)
        {
            FatalErrorInFunction << state.error << " in cell " << celli
                << exit(FatalError);
        }
        newN[celli] = state.N;
        newH[celli] = state.H;
    }
    NNew.primitiveFieldRef() = newN;
    thetaHNew.primitiveFieldRef() = newH;
    NNew.correctBoundaryConditions();
    thetaHNew.correctBoundaryConditions();
}

} // End namespace plateletContent
} // End namespace Foam

#endif

// ************************************************************************* //
