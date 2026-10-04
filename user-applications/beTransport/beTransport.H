/*---------------------------------------------------------------------------*\
    Eulerian elastic strain transport.

    The left Cauchy-Green tensor be, carried on the fixed Eulerian mesh:

        d(be)/dt + div(v be) - be div(v)  =  L.be + be.L^T - (1/tau)(be - I)

    with L the velocity gradient and 1/tau nonzero only where the material is
    fluid, relaxing be back to I. This replaces the total-Lagrangian
    displacement skeleton, which inverted at 31% shell elongation in run
    20260815-231941 because a fixed reference cannot follow material advected
    into cells whose displacement is still zero. The argument for the change is
    in research/literature/AM.2020.0056-19/KINEMATICS_DECISION_2026-08-15.md;
    the plan is research/designs/AM2020_eulerian_solid_plan.md.

    THIS HEADER IS SHARED ON PURPOSE. It is included by the stage-1 gate
    application beTransportGate, which exercises it against closed-form answers
    in a prescribed velocity field, and it will be included unchanged by
    phaseClotAxisymmetricFluid at stage 2. If the two ever hold separate copies
    of this algebra, the gate stops being evidence about the solver.

    The offline algebra is verified independently, without OpenFOAM, by
    scripts/check_b_transport_algebra.py, which is run by the package build.

  ------------------------------------------------------------------------
    THE TRANSPOSE, WHICH IS THE WHOLE TRAP
  ------------------------------------------------------------------------
    Continuum mechanics writes the velocity gradient as

        L_ij = d(v_i)/d(x_j)

    OpenFOAM's fvc::grad(U) returns the OTHER one:

        (fvc::grad(U))_ij = d(U_j)/d(x_i)  =  L_ji

    Both are smooth, bounded and plausible. Substituting one for the other
    produces a stress field that is wrong in direction while every scalar
    diagnostic -- trace, determinant, positive-definiteness -- stays perfect.

    So the stretch term is written in the form OpenFOAM itself uses for the
    upper-convected derivative in its Maxwell, Giesekus and PTT viscoelastic
    models (src/MomentumTransportModels/.../laminar/Maxwell/Maxwell.C, the line
    `twoSymm(sigma & gradU)`), with no explicit transpose anywhere:

        twoSymm(be & gradU)

    Written out, with gradU_kj = d(U_j)/d(x_k):

        (be & gradU)_ij = be_ik gradU_kj = be_ik d(v_j)/d(x_k) = be_ik L_jk
                        = (be . L^T)_ij

    and twoSymm(A) = A + A^T, so

        twoSymm(be & gradU) = be.L^T + L.be

    which is the term required. Exactly one arrangement of the two operands is
    correct and it is not the one that reads left-to-right like the formula.

    Do NOT "simplify" this to twoSymm(gradU & be). That is L^T.be + be.L, the
    planted `transposed_L` bug in the offline check, and it is invisible to a
    rotation test started from an unstrained state -- see the docstring of
    scripts/check_b_transport_algebra.py, where believing otherwise cost a day.
\*---------------------------------------------------------------------------*/

#ifndef beTransport_H
#define beTransport_H

#include "volFields.H"
#include "surfaceFields.H"
#include "fvMatrices.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmSup.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "EulerDdtScheme.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace beTransport
{

//- The unstrained state, as a field-compatible dimensioned quantity.
inline dimensionedSymmTensor identity()
{
    return dimensionedSymmTensor("I", dimless, symmTensor::I);
}


//- The upper-convected stretch term  L.be + be.L^T, explicit.
//
//  Explicit because it is linear in be but not diagonal, so it cannot go into
//  the matrix without coupling the six components. Its growth rate is |L|,
//  which is the local shear rate: 7425 1/s at the peak of run
//  20260816-031412, so |L| dt = 0.074 at the dt that run was already taking.
//
//  EXPLICIT IS STABLE HERE, BUT IT IS ALSO FIRST ORDER, AND THAT IS NOT FREE.
//
//  Evaluated once per timestep at the old be, this term caps the GLOBAL
//  temporal order at 1 no matter what ddtScheme the case names. Run
//  20260816-060212 measured exactly that: the `backward` (BDF2) children came
//  out at order 0.99, and were 2.0x WORSE than the first-order Euler control
//  at every dt on the ladder. BDF2's left side approximates db/dt at t^{n+1}
//  while this source is f(t^n), so the two are inconsistent at O(dt) and the
//  scheme silently degrades to first order with a larger error constant than
//  the first-order scheme it replaced.
//
//  The `extension_eu` children exist only to make that visible, and on their
//  first run they did: a ladder reporting 2 for BDF2 and 2 for Euler would
//  have meant the ladder was measuring nothing, and a ladder reporting 1 for
//  both means what it says.
//
//  The remedy is to iterate the source rather than lag it -- see solve()
//  below. This function stays explicit; making it converge is the caller's
//  job, because in the coupled solver the same iteration is the PIMPLE outer
//  loop and it must not be nested inside a second one.
//
//  TRACE-FREE VARIANT (traceFree = true). Exactly:
//
//      d(ln det be)/dt = tr(be^-1 (L.be + be.L^T)) = 2 tr(L)
//
//  so det(be) is conserved if and only if tr(L) = 0, at ANY strain magnitude.
//  The material here is incompressible, so tr(L) is zero in the continuum and
//  every nonzero value is discretisation error -- but on a collocated mesh the
//  pressure solve makes the FACE FLUX solenoidal, and tr(fvc::grad(U)) from the
//  cell-centred velocity is a DIFFERENT discrete operator that is not.
//
//  Run 20260818-081527 measured the gap directly: |div phi| = 3.3e-8 (machine
//  zero against a 1000 1/s shear rate) while |tr L|/|L| in the clot bulk peaked
//  at 1.0. det(be) reached 4.24. Ten Picard sweeps (sweepChange down 2900x) and
//  a quarter time step (4x the steps) each left the det error unchanged in the
//  fifth digit, which is what rules out iteration and temporal error and leaves
//  only the operator inconsistency.
//
//  Substituting dev(gradU) makes tr = 0 by construction, so det is preserved
//  identically. This is a projection of error, not a change of physics: the
//  elastic stress in the coupled solver already uses only dev(be), so the
//  volumetric part of be was never feeding the momentum equation -- it was only
//  corrupting the SPD state that the deviatoric part is extracted from.
//
//  MEASURED HARMFUL. RUN 20260818-100608 RAN THE A/B AND THIS LOST.
//  det(be) error went from 3.24 (control) to 52.3 with dev(gradU) alone and
//  41.4 with dev(gradU) + 5 outer correctors, and BOTH trace-free children LOST
//  POSITIVE-DEFINITENESS. Do not enable this without new evidence.
//
//  Why the derivation above is right and the fix is still wrong: it is correct
//  that tr(dev L) = 0 makes the source conserve det(be). But with |tr L|/|L|
//  near 1, the isotropic part removed has norm |tr L|/sqrt(3), about 58% of the
//  whole tensor, so the projection mangles the deviatoric response too. More
//  fundamentally the det drift is a SYMPTOM: run 20260818-100608 measured the
//  clot moving at 88% of the free stream with material velocity twice the mean
//  flow, creeping 4 um downstream in 3 ms, while the elastic stress reached 5x
//  the applied load. The clot never reaches elastic equilibrium, so be grows
//  without bound and no change to the be source can fix that.
//
//  It is a switch and not a default because it is a modelling choice, and
//  because the A/B is the evidence -- which it duly provided, against itself. Note also what it does NOT fix: if tr(L) is
//  genuinely that large, the velocity field itself is suspect, and dev() hides
//  that from be without curing it. Read it beside the nOuterCorrectors arm.
inline tmp<volSymmTensorField> stretch
(
    const volSymmTensorField& be,
    const volTensorField& gradU,
    const bool traceFree = false
)
{
    return traceFree ? twoSymm(be & dev(gradU)) : twoSymm(be & gradU);
}


//- Assemble  d(be)/dt + div(phi be) - be div(phi) = stretch - rTau (be - I).
//
//  The div(phi) term makes the advection NON-CONSERVATIVE, which is what be
//  requires: be is a state carried by material points, not a density. On an
//  exactly divergence-free discrete flux the term vanishes and costs nothing;
//  where the discrete flux is not exactly divergence-free it stops the
//  continuity error from being read as strain. It is here because leaving it
//  out is invisible in every test with a solenoidal analytic velocity, which
//  is every test in the stage-1 gate.
//
//  rTau is 1/tau: zero in the solid, large in the fluid. Implicit, following
//  Maxwell::sigmaSource, because tau in blood is short against dt.
inline tmp<fvSymmTensorMatrix> eqn
(
    volSymmTensorField& be,
    const surfaceScalarField& phi,
    const volTensorField& gradU,
    const volScalarField& rTau,
    const word& ddtScheme = word::null,
    const bool traceFree = false
)
{
    // THERE IS NO fvm::ddt(word, field) IN v13 -- the overload set is
    // (field), (dimensionedScalar, field), (volScalarField, field) and the
    // one/two-coefficient variants, and nothing takes a scheme name. So the
    // Euler override instantiates the scheme class directly rather than going
    // through the dictionary, which is also less indirection: the case's own
    // ddtSchemes entry stays in force for every other equation and for this
    // one on every step but the first.
    return tmp<fvSymmTensorMatrix>
    (
        new fvSymmTensorMatrix
        (
            (
                ddtScheme == "Euler"
              ? fv::EulerDdtScheme<symmTensor>(be.mesh()).fvmDdt(be)
              : fvm::ddt(be)
            )
          + fvm::div(phi, be, "div(phi,be)")
          - fvm::Sp(fvc::div(phi), be)
         ==
            stretch(be, gradU, traceFree)
          - fvm::Sp(rTau, be)
          + rTau*identity()
        )
    );
}


//- Solve the transport over one timestep, iterating the explicit source.
//
//  Picard iteration. Each sweep re-assembles the stretch term from the LATEST
//  be rather than the old-time one; the fixed point of that iteration is the
//  fully implicit scheme, at which point the ddtScheme's own order is
//  recovered. `fvm::ddt(be)` reads be.oldTime(), which only advances when the
//  Time object does, so sweeping inside a step is a correct Picard iteration
//  and not an accidental double advance.
//
//  It converges at rate ~|L| dt per sweep, which is 0.074 at the main case's
//  operating point, so two sweeps are already deep in the asymptotic regime
//  and three are comfortable. It is not free -- nCorr sweeps cost nCorr
//  linear solves -- but the be solve is a small fraction of a step (stage 0
//  measured the floor it is replacing at 6.79 s/step), so trebling it does not
//  move the cost case.
//
//  Returns the max-norm change over the FINAL sweep, so a caller can report
//  whether the iteration actually converged rather than assuming nCorr was
//  enough.
//  `ddtScheme`, when not null, overrides the case's ddt scheme for this solve.
//  Its only intended use is passing "Euler" on the FIRST timestep: OpenFOAM's
//  `backward` has one old-time level there, its stencil degenerates to
//  1.5 (f^1 - f^0)/dt, and the first increment comes out at exactly 2/3 of the
//  correct one -- measured to five significant figures in stage 1, where it
//  held the temporal order at 1 until it was fixed. A single Euler step is a
//  one-off O(dt^2) local error and does not degrade second order; the
//  degenerate BDF2 step is O(dt) and does.
inline scalar solve
(
    volSymmTensorField& be,
    const surfaceScalarField& phi,
    const volTensorField& gradU,
    const volScalarField& rTau,
    const label nCorr,
    const word& ddtScheme = word::null,
    const bool traceFree = false
)
{
    scalar change = 0;

    for (label corr = 0; corr < max(nCorr, label(1)); ++corr)
    {
        const volSymmTensorField bePrevSweep(be);

        fvSymmTensorMatrix beEqn(eqn(be, phi, gradU, rTau, ddtScheme, traceFree));
        beEqn.solve();

        change = gMax(mag((be - bePrevSweep)().primitiveField()));
    }

    return change;
}


// * * * * * * * * * * * * * * Health diagnostics  * * * * * * * * * * * * * //
//
// be must stay symmetric positive-definite. Symmetry is structural -- it is a
// symmTensor and every operator above preserves it -- so the only thing that
// can actually be lost is positive-definiteness, and losing it is the failure
// mode the stage-1 gate exists to detect. It is checked by Sylvester's
// criterion, which for a symmetric 3x3 is exact and needs no eigensolver.

//- Smallest leading principal minor over all cells, normalised per order so
//  the three are comparable. Positive everywhere <=> SPD everywhere.
inline scalar minLeadingMinor(const volSymmTensorField& be)
{
    scalar worst = great;

    forAll(be, celli)
    {
        const symmTensor& b = be[celli];

        const scalar m1 = b.xx();
        const scalar m2 = b.xx()*b.yy() - b.xy()*b.xy();
        const scalar m3 = det(b);

        // cbrt/sqrt so a minor of a stretched state is compared on the same
        // scale as an unstretched one; without this the third minor dominates
        // the minimum purely by being a triple product.
        worst = min(worst, m1);
        worst = min(worst, m2 > 0 ? sqrt(m2) : m2);
        worst = min(worst, m3 > 0 ? cbrt(m3) : m3);
    }

    reduce(worst, minOp<scalar>());
    return worst;
}


//- max |det(be) - 1| over all cells.
//
//  For any isochoric motion det(be) = (det F)^2 = 1 exactly, and all four
//  motions in the stage-1 gate are isochoric, so this is a free conservation
//  diagnostic with a known answer. It is independent of the closed-form
//  comparison and does not need one, which makes it the diagnostic that
//  survives into the coupled runs where no closed form exists.
inline scalar maxDetError(const volSymmTensorField& be)
{
    scalar worst = 0;

    forAll(be, celli)
    {
        worst = max(worst, mag(det(be[celli]) - 1.0));
    }

    reduce(worst, maxOp<scalar>());
    return worst;
}


//- Extreme eigenvalues of one symmetric tensor, closed form.
//
//  The SAME Smith (1961) route maxStretch() uses below, factored out and
//  extended to the smallest root, so a per-cell diagnostic does not have to
//  call OpenFOAM's eigenValues(). That function is correct, but it is not used
//  anywhere else in this codebase and an unproven call in a block that only
//  runs on Colab is a compile failure that costs a VM boot -- which has
//  already happened once in this project, on a `?:` between tmp<> and a field.
//
//  Roots of a symmetric 3x3 in ascending order are
//      q + 2p cos(phi + 2k pi/3),  k = 2, 1, 0
//  so k=0 is the largest and k=2 the smallest.
inline void extremeEigenvalues
(
    const symmTensor& b,
    scalar& eigMin,
    scalar& eigMax
)
{
    const scalar p1 = sqr(b.xy()) + sqr(b.xz()) + sqr(b.yz());

    if (p1 < small)
    {
        eigMin = min(b.xx(), min(b.yy(), b.zz()));
        eigMax = max(b.xx(), max(b.yy(), b.zz()));
        return;
    }

    const scalar q = tr(b)/3.0;
    const scalar p2 =
        sqr(b.xx() - q) + sqr(b.yy() - q) + sqr(b.zz() - q) + 2.0*p1;
    const scalar p = sqrt(p2/6.0);
    const symmTensor B((b - q*symmTensor::I)/p);
    const scalar r = min(max(det(B)/2.0, -1.0), 1.0);
    const scalar phi = acos(r)/3.0;

    eigMax = q + 2.0*p*cos(phi);
    eigMin = q + 2.0*p*cos(phi + 2.0*constant::mathematical::pi/3.0);
}


//- Largest principal stretch anywhere, = sqrt(max eigenvalue of be).
//
//  Reported so the gate can state the stretch it actually reached rather than
//  the stretch it was configured for. The gate criterion is SPD held past a
//  stretch of 1.5, which is past where the displacement skeleton failed, so
//  the number that criterion is read against must be measured.
//
//  Largest eigenvalue by the closed-form symmetric 3x3 route (Smith 1961), the
//  same one OpenFOAM's eigenValues() uses, but reduced to the maximum only.
inline scalar maxStretch(const volSymmTensorField& be)
{
    scalar worst = 0;

    forAll(be, celli)
    {
        const symmTensor& b = be[celli];

        const scalar p1 = sqr(b.xy()) + sqr(b.xz()) + sqr(b.yz());
        scalar eigMax;

        if (p1 < small)
        {
            eigMax = max(b.xx(), max(b.yy(), b.zz()));
        }
        else
        {
            const scalar q = tr(b)/3.0;
            const scalar p2 =
                sqr(b.xx() - q) + sqr(b.yy() - q) + sqr(b.zz() - q) + 2.0*p1;
            const scalar p = sqrt(p2/6.0);
            const symmTensor B((b - q*symmTensor::I)/p);
            const scalar r = min(max(det(B)/2.0, -1.0), 1.0);
            const scalar phi = acos(r)/3.0;
            eigMax = q + 2.0*p*cos(phi);
        }

        worst = max(worst, eigMax);
    }

    reduce(worst, maxOp<scalar>());
    return worst > 0 ? sqrt(worst) : 0;
}


} // End namespace beTransport
} // End namespace Foam

#endif

// ************************************************************************* //
