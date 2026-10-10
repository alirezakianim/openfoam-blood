#ifndef plateletTransferKernel_H
#define plateletTransferKernel_H

// Portable arithmetic shared by the OpenFOAM wrapper and executable T4 tests.
// N,dB: platelet-equivalents / mixture volume; C,H: solid volume fraction;
// Y: deposited volume / platelet-equivalent. Binding only, constant Y.
#include <algorithm>
#include <cmath>

namespace plateletTransferKernel
{
template<class Scalar> struct Increment
{
    Scalar amount;
    bool capacityLimited;
    bool availabilityLimited;
    const char* error;
};

template<class Scalar> struct Applied
{
    Scalar N, C, H;
    const char* error;
};

template<class Scalar>
inline bool parametersValid(Scalar Y, Scalar thetaMax)
{
    return std::isfinite(Y) && Y > 0 && std::isfinite(thetaMax)
        && thetaMax > 0 && thetaMax < 1;
}

template<class Scalar>
inline Increment<Scalar> acceptedIncrement
(
    Scalar N, Scalar theta, Scalar dt, Scalar lambda, Scalar Y, Scalar thetaMax
)
{
    // S25 G (design 2026-10-10-S25 section 2G; stiffening 08 Q3): theta
    // above thetaMax is a LEGAL state (compaction past the as-deposited
    // packing) and is no longer an input error. Such a cell accepts no new
    // deposit: its capacity below is zero. Bit-identical wherever theta <=
    // thetaMax.
    if (!parametersValid(Y, thetaMax) || !std::isfinite(N) || N < 0
        || !std::isfinite(theta) || theta < 0
        || !std::isfinite(dt) || dt < 0 || !std::isfinite(lambda) || lambda < 0)
    {
        return {Scalar(0), false, false, "invalid transfer input (no state was clipped)"};
    }
    const Scalar dtl = dt*lambda;
    if (!std::isfinite(dtl))
    {
        return {Scalar(0), false, false, "nonfinite dt*lambda"};
    }
    // Same implicit rate as the original wrapper. Refuse overflow rather than
    // turning an invalid proposal into a seemingly legitimate capacity limit.
    const Scalar proposal = dtl*N/(Scalar(1) + dtl);
    if (!std::isfinite(proposal))
    {
        return {Scalar(0), false, false, "nonfinite reaction proposal"};
    }
    // S25 G: the acceptance law, capacity = (cap - theta)_+/Y. It bounds
    // the SOURCE (new deposit needs open pore space) and never writes theta.
    const Scalar capacity = std::max(thetaMax - theta, Scalar(0))/Y;
    const bool capped = proposal > capacity;
    return {std::min(proposal, std::min(N, capacity)), capped, proposal > N, nullptr};
}

template<class Scalar>
inline Applied<Scalar> applyIncrement
(
    Scalar N, Scalar C, Scalar H, Scalar dB, Scalar Y, Scalar thetaMax
)
{
    // Invalid input returns the original values; this function never commits
    // a partial state. Callers must reject error != nullptr. S25 G: C + H
    // above thetaMax is legal; dB <= capacity = (thetaMax - (C + H))_+/Y
    // stays, so such a cell accepts exactly dB = 0.
    if (!parametersValid(Y, thetaMax) || !std::isfinite(N) || N < 0
        || !std::isfinite(C) || C < 0 || !std::isfinite(H) || H < 0
        || !std::isfinite(dB) || dB < 0 || dB > N
        || dB > std::max(thetaMax - (C + H), Scalar(0))/Y)
    {
        return {N, C, H, "invalid accepted transfer or component state"};
    }
    return {N - dB, C, H + Y*dB, nullptr};
}
}
#endif
