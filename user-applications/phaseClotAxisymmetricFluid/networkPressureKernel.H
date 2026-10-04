/*---------------------------------------------------------------------------*\
    networkPressureKernel.H -- the convexExcess network pressure's per-cell
    arithmetic that S18 C (R2b) added, as OpenFOAM-free inline functions, so
    the solver and an executable FP-trapping test run the SAME expressions.

    research/designs/2026-09-29-S18-no-frozen-pressure-content-weight-network-
    viscosity.md, rev. 10, section 2 C; fix round 25d (docket
    research/investigations/20261001-1800-run023222-s18-step1-sigfpe/):
    batch 60's em76 died of SIGFPE on step 1 because the dt bound divided by
    thetaNetRef in lumen cells (thetaNetRef = 0). Every division here is
    guarded BY CONSTRUCTION, by the zero set of the weight it serves:

    - the content weight w = theta/ref is defined where ref > 0; where
      ref = 0 no network was deposited, networkPresence() = 0 and the force
      presence w p is IDENTICALLY 0 for every theta, so w and every
      derivative of presence w p are exactly 0 there. Returning 0 is the
      value of the force's derivative, not a clamp of a large one.
    - the convexExcess pressure divides by a (pm - a), the denominator of
      the dt bound's dp/da since S10; the solver's checkNetworkArgument()
      refuses any a outside (0, pm).

    THE GUARD MUST NOT BE A TERNARY AROUND THE DIVISION ALONE. Without
    trapping-math semantics an optimiser may evaluate both arms of
    `ref > 0 ? theta/ref : 0` and select (clang -O2 does by default; the FP-
    trap test caught it), so the division runs at ref = 0 even though its
    result is discarded. (gcc's default -ftrapping-math, the VM's compiler,
    forbids that; the test is compiled with clang's equivalent,
    -ffp-exception-behavior=maytrap.) Each guarded division here divides,
    UNCONDITIONALLY, by a denominator that is never 0 -- the guarded value,
    or 1 where the guard fails -- and only then does the guard select.

    Tested by scripts/s18_fpe_kernel_test.cpp with FP traps enabled.
\*---------------------------------------------------------------------------*/

#ifndef networkPressureKernel_H
#define networkPressureKernel_H

#include <cmath>

namespace networkPressureKernel
{

//- p(a) of the convexExcess law, per unit (coefficient gD G): Kn (pm - phi0)
//  (a - phi0)/(a (pm - a)) -- networkPressure()'s expression, cellwise.
inline double convexExcessPressure
(
    const double Kn, const double pm, const double phi0, const double a
)
{
    return Kn*(pm - phi0)*(a - phi0)/(a*(pm - a));
}


//- S18 C (R2b): the content weight theta/ref, 0 where ref = 0 (no network
//  deposited: the force presence w p is identically 0 there).
inline double contentRatio(const double theta, const double ref)
{
    const bool has = ref > 0;
    const double r = has ? ref : 1.0;
    const double q = theta/r;
    return has ? q : 0.0;
}


//- S18 C (R2b): d(w p)/da with w = theta/ref and da = dtheta, i.e.
//  (theta/ref) dp/da + p/ref (numerics 10's "w dp + p dw"). Exactly 0 where
//  ref = 0: there w p = 0 identically (see the header). The dt bound and the
//  anti-restoring (S < 0) watch both take it, each times its presence.
inline double contentRatioSlope
(
    const double theta, const double ref, const double dpda, const double pa
)
{
    const bool has = ref > 0;
    const double r = has ? ref : 1.0;
    const double v = (theta/r)*dpda + pa/r;
    return has ? v : 0.0;
}


//- S18 F: source 32's TOTAL stretch at a damage event, (ref/theta)^(1/3)
//  lambda1 with lambda1 = sqrt(b1) (04's convention); -1 where theta <= 0
//  (no content to stretch: the log's "not defined" value).
inline double totalStretch(const double ref, const double theta, const double b1)
{
    const bool has = theta > 0;
    const double t = has ? theta : 1.0;
    const double v = std::cbrt(ref/t)*std::sqrt(b1);
    return has ? v : -1.0;
}

} // End namespace networkPressureKernel

#endif
