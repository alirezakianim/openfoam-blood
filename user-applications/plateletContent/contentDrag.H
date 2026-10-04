#ifndef contentDrag_H
#define contentDrag_H

// OpenFOAM field wrapper for the Stage 3 content-driven drag coefficient.
// The arithmetic lives in contentDragKernel.H, which is portable and is
// checked against scripts/check_content_drag_driver.py at 1e-14 relative by
// scripts/check_content_drag_kernel.py. NOTHING is recomputed here.
//
// Implementation authority: research/model-specs/AM2020-content-drag-v1.md.

#include "contentDragKernel.H"
#include "volFields.H"

namespace Foam
{
namespace contentDrag
{

//- One cell's refusal, carried out of the loop so the report can name it.
struct Refusal
{
    label celli;
    const char* reason;
};

//- Fill K and the contentInvPermKC diagnostic from the conserved content.
//
//  RETURNS a refusal rather than clipping. A cell outside the declared domain
//  is a REJECT with a machine-readable reason (spec section 4); it is never
//  clamped and promoted, and no stored field is touched here. `K` and
//  `invPerm` are left in whatever state the loop reached, because the caller
//  must not use them after a refusal -- it must report and stop.
//
//  `nKC` counts cells on the Kozeny-Carman branch and `nDilute` those on the
//  named dilute continuation. They are reported because contentInvPermKC
//  describes the KC branch ONLY, which is not necessarily the branch the
//  momentum equation used (spec section 7), and the difference must never be
//  inferred from the diagnostic field.
inline bool evaluate
(
    const volScalarField& thetaC,
    const volScalarField& thetaH,
    const contentDragKernel::Parameters<scalar>& params,
    volScalarField& K,
    volScalarField& invPerm,
    label& nKC,
    label& nDilute,
    scalar& discrepancySum,
    Refusal& refusal
)
{
    nKC = 0;
    nDilute = 0;
    discrepancySum = 0;
    refusal = {-1, nullptr};

    if (thetaC.dimensions() != dimless || thetaH.dimensions() != dimless
     || K.dimensions() != dimless/dimTime
     || invPerm.dimensions() != dimless/dimArea
     || &thetaC.mesh() != &thetaH.mesh() || &thetaC.mesh() != &K.mesh()
     || &thetaC.mesh() != &invPerm.mesh())
    {
        FatalErrorInFunction
            << "inconsistent content-drag dimensions or mesh"
            << exit(FatalError);
    }

    scalarField& Ki = K.primitiveFieldRef();
    scalarField& Ii = invPerm.primitiveFieldRef();
    const scalarField& Ci = thetaC.primitiveField();
    const scalarField& Hi = thetaH.primitiveField();

    forAll(Ki, celli)
    {
        const contentDragKernel::Coefficient<scalar> r =
            contentDragKernel::evaluate(Ci[celli], Hi[celli], params);

        if (r.error)
        {
            refusal = {celli, r.error};
            return false;
        }

        Ki[celli] = r.K;
        Ii[celli] = r.invPerm;
        discrepancySum += r.discrepancy;
        if (r.K > 0)
        {
            if (r.kcBranch) { ++nKC; } else { ++nDilute; }
        }
    }

    reduce(nKC, sumOp<label>());
    reduce(nDilute, sumOp<label>());
    reduce(discrepancySum, sumOp<scalar>());
    K.correctBoundaryConditions();
    invPerm.correctBoundaryConditions();
    return true;
}

} // End namespace contentDrag
} // End namespace Foam

#endif
