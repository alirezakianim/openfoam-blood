#ifndef contentDragKernel_H
#define contentDragKernel_H

// Portable arithmetic for the Stage 3 content-driven interphase drag.
// Implementation authority: research/model-specs/AM2020-content-drag-v1.md.
// Independent oracle: scripts/check_content_drag_driver.py -- this file is
// required to reproduce it, and the oracle was written first.
//
// theta = thetaC + thetaH : solid volume fraction (dimensionless)
// s     = thetaH/theta    : composition, shell fraction of the content
// K     : 1/s. Dynamic force uses rho*K, kinematic force uses K.
//
// TWO THINGS THIS FILE WILL NOT DO.
//
// 1. It never evaluates the permeability k in isolation. The drag is formed
//    directly as A(s)*theta^3/(1-theta); evaluating k alone puts a 1/theta^2
//    singularity in the middle of an expression that has none.
// 2. It never clips a stored field. An out-of-domain input is REFUSED with a
//    machine-readable reason, so the runner can tell a source-side refusal
//    from an infrastructure failure (spec section 4).
//
// CONVENTION WARNING (spec section 2.1). The cancellation to
// theta^3/(1-theta) assumes the two-velocity fluid-fraction factor
// (1-theta)^2. Without it the same substitution gives theta^3/(1-theta)^3 --
// a different rational function, differing by 1/(1-theta)^2, which is
// 3.30578512397 at theta_ref = 0.45 and 1.5625 at theta = 0.2. It is not a
// prefactor. This kernel implements the TWO-VELOCITY convention only and must
// not be applied to a one-velocity path.
#include <algorithm>
#include <cmath>

namespace contentDragKernel
{

template<class Scalar> struct Parameters
{
    Scalar thetaRef;      // calibration anchor AND force-convention parameter
    Scalar thetaMax;      // declared content-validity limit (phiSMax)
    Scalar kShell;        // permeability at the anchor, shell material
    Scalar kCore;         // permeability at the anchor, core material
    Scalar nu;            // kinematic viscosity
    Scalar k1;            // dilute drag continuation rate; 0 disables it
    Scalar trace;         // declared trace-negative allowance; 0 = strict
    Scalar kFloor;        // permeability floor RETAINED from the legacy clip
                          // (spec section 6). MUST be > 0: a zero here is not
                          // "no floor", it is an uninitialised aggregate, and
                          // parametersValid() refuses it so that a
                          // construction site which forgets this field fails
                          // loudly instead of silently dropping the bound.
};

template<class Scalar> struct Coefficient
{
    Scalar K;             // the coefficient the momentum equations use
    Scalar invPerm;       // contentInvPermKC, m^-2, DIAGNOSTIC ONLY
    Scalar discrepancy;   // theta_eval - theta; an evaluation discrepancy,
                          // never "volume removed by clipping"
    bool kcBranch;        // which branch produced K
    const char* error;    // nullptr, or a machine-readable refusal reason
};

template<class Scalar>
inline bool parametersValid(const Parameters<Scalar>& p)
{
    return std::isfinite(p.thetaRef) && p.thetaRef > 0 && p.thetaRef < 1
        && std::isfinite(p.thetaMax) && p.thetaMax > 0 && p.thetaMax < 1
        && std::isfinite(p.kShell) && p.kShell > 0
        && std::isfinite(p.kCore) && p.kCore > 0
        && std::isfinite(p.nu) && p.nu > 0
        && std::isfinite(p.k1) && p.k1 >= 0
        && std::isfinite(p.trace) && p.trace >= 0
        && std::isfinite(p.kFloor) && p.kFloor > 0;
}

//- Log interpolation between core and shell reference permeability:
//  log k_ref(s) = (1-s)*log(k_core) + s*log(k_shell).
//  The clamp is on the ARGUMENT of an interpolation -- a composition outside
//  [0,1] is not a mixing fraction -- and never on a stored field.
template<class Scalar>
inline Scalar referencePermeability(Scalar s, const Parameters<Scalar>& p)
{
    const Scalar sc = std::min(std::max(s, Scalar(0)), Scalar(1));
    return p.kCore*std::exp(sc*std::log(p.kShell/p.kCore));
}

//- The coefficient, behind the full guard ladder of spec section 5.
//  THE ORDER IS LOAD-BEARING and check_content_drag_wiring.py asserts it:
//  config -> components -> upper domain -> non-negative -> zero solid.
template<class Scalar>
inline Coefficient<Scalar> evaluate
(
    Scalar thetaC, Scalar thetaH, const Parameters<Scalar>& p
)
{
    const Coefficient<Scalar> zero = {Scalar(0), Scalar(0), Scalar(0), false, nullptr};

    // Rung 1: configuration and finiteness, before any log or division.
    if (!parametersValid(p))
    {
        return {Scalar(0), Scalar(0), Scalar(0), false,
                "invalid_configuration"};
    }
    if (!std::isfinite(thetaC) || !std::isfinite(thetaH))
    {
        return {Scalar(0), Scalar(0), Scalar(0), false, "nonfinite_content"};
    }

    // Rung 3: components screened SEPARATELY and BEFORE the zero-total
    // shortcut. thetaC = -a with thetaH = +a sums to zero, and a zero-total
    // shortcut placed first would return 0 for it and never look.
    if (thetaC < -p.trace || thetaH < -p.trace)
    {
        return {Scalar(0), Scalar(0), Scalar(0), false, "negative_component"};
    }

    const Scalar theta = thetaC + thetaH;

    // Rung 2: upper domain, INCLUSIVE. theta == thetaMax is accepted: at
    // thetaMax = 0.6 the coefficient is finite, and there is no reason for
    // drag evaluation to refuse exactly the state the transfer is permitted
    // to reach. This matches plateletTransferKernel.H, which rejects on
    // theta > thetaMax only.
    if (theta > p.thetaMax)
    {
        return {Scalar(0), Scalar(0), Scalar(0), false,
                "content_exceeds_theta_max"};
    }

    // Rung 4: a screened trace negative is EVALUATED at zero. The caller's
    // stored field is not touched; the difference is reported.
    const Scalar thetaEval = std::max(theta, Scalar(0));
    const Scalar discrepancy = thetaEval - theta;

    // Rung 5: zero solid gives exactly zero drag WITHOUT evaluating s or
    // k_ref(s). s = thetaH/theta is 0/0 here and a finite K does not rescue it.
    if (thetaEval == Scalar(0))
    {
        Coefficient<Scalar> out = zero;
        out.discrepancy = discrepancy;
        return out;
    }

    const Scalar s = thetaH/thetaEval;
    const Scalar kRef = referencePermeability(s, p);
    const Scalar oneMinus = Scalar(1) - thetaEval;
    if (!(oneMinus > Scalar(0)))
    {
        return {Scalar(0), Scalar(0), discrepancy, false,
                "content_exceeds_theta_max"};
    }

    const Scalar a =
        p.nu*(Scalar(1) - p.thetaRef)*(Scalar(1) - p.thetaRef)
       *(Scalar(1) - p.thetaRef)
       /(p.thetaRef*p.thetaRef*p.thetaRef*kRef);
    const Scalar kc = a*thetaEval*thetaEval*thetaEval/oneMinus;
    const Scalar dilute = p.k1*thetaEval/p.thetaRef;

    // contentInvPermKC, m^-2. DIAGNOSTIC ONLY: it describes the KC branch,
    // which is not necessarily the branch the momentum equation used, so the
    // selected branch is reported alongside it and never inferred from it.
    const Scalar invPerm =
        ((Scalar(1) - p.thetaRef)*(Scalar(1) - p.thetaRef)
        *(Scalar(1) - p.thetaRef)/(p.thetaRef*p.thetaRef))
       *thetaEval*thetaEval/(kRef*oneMinus*oneMinus*oneMinus);

    if (!std::isfinite(kc) || !std::isfinite(dilute) || !std::isfinite(invPerm))
    {
        return {Scalar(0), Scalar(0), discrepancy, false,
                "nonfinite_coefficient"};
    }

    // SPEC SECTION 6, the legacy permeability clip, retained.
    //
    // The shipped legacy path does not use an unlimited Kozeny-Carman
    // permeability: it clips to [kFloor, 1e4*k_endpoint] BEFORE forming the
    // drag, and only then takes the dilute maximum. The cancelled form above
    // is exact for the UNCAPPED expression, so dropping those branches would
    // have been a second model change riding along with the driver change.
    //
    // Established mechanically in scripts/check_content_drag_cap_equivalence.py
    // over the declared domain at the shipped parameters:
    //
    //   * the UPPER cap (k <= 1e4*k_endpoint) fires only for
    //     theta < 0.0108532442273, and the retained dilute floor already
    //     meets or exceeds the capped value at every such point. MASKED --
    //     nothing to retain.
    //   * the LOWER clip (k >= kFloor) never fires for shell material, but
    //     for core material it fires at theta > 0.597610545563, which is
    //     INSIDE the declared domain theta <= thetaMax = 0.6. At theta = 0.6
    //     the uncapped form is 2.6% above the capped one. NOT masked, so it
    //     is retained here.
    //
    // Retained as the equivalent BOUND ON K rather than by clipping k,
    // because forming k in isolation is exactly what the cancellation exists
    // to avoid:
    //
    //     k >= kFloor   <=>   K <= (theta/thetaRef)*(1-theta)^2*nu/kFloor
    //
    // Applied to the KC term BEFORE the dilute maximum, matching the legacy
    // order: the clip is on poroPerm_ (inside), the dilute max is outside.
    const Scalar kcBound =
        (thetaEval/p.thetaRef)*oneMinus*oneMinus*p.nu/p.kFloor;
    const Scalar kcUsed = kc < kcBound ? kc : kcBound;

    const bool kcBranch = kcUsed >= dilute;
    return {std::max(kcUsed, dilute), invPerm, discrepancy, kcBranch, nullptr};
}

}
#endif
