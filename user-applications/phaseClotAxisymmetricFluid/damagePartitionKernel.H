/*---------------------------------------------------------------------------*\
    S16 C+ (research/designs/2026-09-24-S16-timed-terminal-rupture.md, 2C):
    the damage partition follows theta's LIMITED antidiffusive flux.

    Header-only, no OpenFOAM dependency, so that the arithmetic the solver
    runs is the arithmetic scripts/check_s16_timed_terminal_rupture.py
    compiles and tests (as plateletTransferKernel.H is for the transfer).

    The formulas are the reference replica's, docket
    research/investigations/20260924-1551-run074145-s15-first-return,
    probes/numerics-analyst-p17-r3.py (17):
      limiterR1      <- lam_R1       L69-72, Zalesak of p4-dtransport L94-133
      applyR1        <- apply_R1     L75-77, apply of p4-dtransport  L136-143
      donorShares    <- donor_apply  L86-120
      composition    <- step, C_plus L183-212

    theta (thetaC, thetaH, thetaNetRef) is NOT touched here: it is limited and
    applied by the shipped fctLimiter/fctApply, exactly as in package 2dd6246c.
    The partition only READS theta's shared lambda and total antidiffusive
    flux.

    FLOATING POINT (red team 16, R1). The partition's own limiter and its
    correction are evaluated in FIELD units -- the flux multiplied by
    s = dt/Vmin first, cell volumes as Vrel = V/Vmin, dt = 1 -- and only an
    EXACT zero flux is "no flux" (contributes 1). A subnormal flux is a flux:
    x dt/V it is a normal number (~1e-304), and the limiter bounds it.
\*---------------------------------------------------------------------------*/

#ifndef damagePartitionKernel_H
#define damagePartitionKernel_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace damagePartitionKernel
{

//- mu as computed (the solver), or forced -- test hooks for the guard's
//  negative controls only: forceOne must produce negatives, forceZero must
//  reproduce C bit for bit.
enum class MuMode { limited, forceOne, forceZero };

template<class Scalar>
struct Stats
{
    Scalar FoutMax;     // largest fraction of a donor's content leaving it
    Scalar muMin;       // over faces carrying a nonzero composition flux
    Scalar muMean;      // weighted by sum_k |E_k|
    long nActive;       // faces carrying a nonzero composition flux
    const char* error;  // nullptr, or why the step is refused
    long where;         // the cell or face the error names
};


template<class Scalar>
inline bool finite(Scalar x)
{
    return std::isfinite(x);
}


//- Zalesak's face limiter in field units (R1). A is the antidiffusive flux
//  in content-volume per second; it is scaled by s = dt/Vmin before any
//  comparison with tiny. lam = 1 on a face whose A is EXACTLY zero.
template<class Scalar, class Label>
inline void limiterR1
(
    const Label nCells, const Label nFaces,
    const Label* own, const Label* nei,
    const Scalar* xL, const Scalar* xO, const Scalar* A,
    const Scalar s, const Scalar* Vrel,
    Scalar* lam
)
{
    const Scalar tiny = std::numeric_limits<Scalar>::min();
    const Scalar keep = Scalar(1) - Scalar(1.0e-12);

    std::vector<Scalar> hi(nCells), lo(nCells);
    for (Label c = 0; c < nCells; ++c)
    {
        hi[c] = std::max(xL[c], xO[c]);
        lo[c] = std::min(xL[c], xO[c]);
    }
    for (Label f = 0; f < nFaces; ++f)
    {
        const Label o = own[f], n = nei[f];
        hi[o] = std::max(hi[o], std::max(xL[n], xO[n]));
        lo[o] = std::min(lo[o], std::min(xL[n], xO[n]));
        hi[n] = std::max(hi[n], std::max(xL[o], xO[o]));
        lo[n] = std::min(lo[n], std::min(xL[o], xO[o]));
    }

    std::vector<Scalar> Pp(nCells, Scalar(0)), Pm(nCells, Scalar(0));
    for (Label f = 0; f < nFaces; ++f)
    {
        const Scalar a = A[f]*s;
        if (a > 0)
        {
            Pm[own[f]] += a;
            Pp[nei[f]] += a;
        }
        else if (a < 0)
        {
            Pp[own[f]] -= a;
            Pm[nei[f]] -= a;
        }
    }

    std::vector<Scalar> Rp(nCells), Rm(nCells);
    for (Label c = 0; c < nCells; ++c)
    {
        // p4-dtransport L118-119 with V = Vrel and dt = 1.
        const Scalar Qp = (hi[c] - xL[c])*Vrel[c]*keep;
        const Scalar Qm = (xL[c] - lo[c])*Vrel[c]*keep;
        Rp[c] =
            Qp < tiny ? Scalar(0)
          : Pp[c] > 0 ? std::min(Scalar(1), Qp/Pp[c]) : Scalar(1);
        Rm[c] =
            Qm < tiny ? Scalar(0)
          : Pm[c] > 0 ? std::min(Scalar(1), Qm/Pm[c]) : Scalar(1);
    }

    for (Label f = 0; f < nFaces; ++f)
    {
        const Scalar a = A[f]*s;
        if (A[f] == 0)
        {
            lam[f] = Scalar(1);                 // exact zero: no flux (R1)
        }
        else if (std::abs(a) < tiny)
        {
            lam[f] = Scalar(0);
        }
        else if (a > 0)
        {
            lam[f] = std::min(Rm[own[f]], Rp[nei[f]]);
        }
        else
        {
            lam[f] = std::min(Rp[own[f]], Rm[nei[f]]);
        }
    }
}


//- x -= div(lam*A) in field units: fx = lam*(A*s), x - (1/Vrel)*div.
template<class Scalar, class Label>
inline void applyR1
(
    const Label nCells, const Label nFaces,
    const Label* own, const Label* nei,
    const Scalar* lam, const Scalar* A,
    const Scalar s, const Scalar* Vrel,
    Scalar* x
)
{
    std::vector<Scalar> div(nCells, Scalar(0));
    for (Label f = 0; f < nFaces; ++f)
    {
        const Scalar la = lam[f]*(A[f]*s);
        div[own[f]] += la;
        div[nei[f]] -= la;
    }
    for (Label c = 0; c < nCells; ++c)
    {
        x[c] = x[c] - (Scalar(1)/Vrel[c])*div[c];
    }
}


//- C: every component takes its donor cell's share of theta's limited
//  antidiffusive flux g = lambda_theta*A_theta (g > 0 moves content from the
//  owner to the neighbour). f = |g| dt/(V_donor theta_low(donor)); theta's
//  Zalesak bound gives sum_out f <= keep < 1, so
//    P_new = P_low (1 - F_out) + sum_in P_low(donor) f V_donor / V
//  is a sum of non-negative terms, evaluated in field units (dt/V and
//  V_donor/V_receiver formed first; R7). Refuses (error) a non-finite
//  input, an active face whose donor holds no theta, and F_out >= 1.
template<class Scalar, class Label>
inline Stats<Scalar> donorShares
(
    const Label nCells, const Label nFaces,
    const Label* own, const Label* nei,
    const Scalar* V, const Scalar dt,
    const Scalar* thL, const Scalar* g,
    const int nComp, const Scalar* const* Plow, Scalar* const* Pout,
    std::vector<Scalar>& fFace
)
{
    Stats<Scalar> st{Scalar(0), Scalar(1), Scalar(1), 0, nullptr, -1};
    for (Label c = 0; c < nCells; ++c)
    {
        if (!finite(thL[c]) || !finite(V[c]) || !(V[c] > 0))
        {
            st.error = "non-finite theta_low or cell volume";
            st.where = c;
            return st;
        }
        for (int k = 0; k < nComp; ++k)
        {
            if (!finite(Plow[k][c]) || Plow[k][c] < 0)
            {
                st.error = "non-finite or negative partition predictor";
                st.where = c;
                return st;
            }
        }
    }
    fFace.assign(nFaces, Scalar(0));
    std::vector<Scalar> Fout(nCells, Scalar(0));
    for (Label f = 0; f < nFaces; ++f)
    {
        if (!finite(g[f]))
        {
            st.error = "non-finite limited theta flux";
            st.where = f;
            return st;
        }
        if (g[f] == 0)
        {
            continue;
        }
        const Label d = g[f] > 0 ? own[f] : nei[f];
        if (!(thL[d] > 0))
        {
            // 17, Not checked: Zalesak gives lambda = 0 there, so this
            // cannot occur; asserted, not assumed.
            st.error = "active theta flux out of a donor holding no theta";
            st.where = f;
            return st;
        }
        // R7 (red team 19): FIELD-UNITS ORDER, dt/V formed first. The
        // raw-units form |g|*dt/(V*thL) builds two SUBNORMAL products at a
        // leading-edge donor with thL below ~1e-292 (V 2e-19 m^3 in the
        // throat), and their rounding exceeds theta's 1e-12 keep margin: a
        // correct bound then reads F_out >= 1 (19: 7501 of 12921 binding
        // donors in [1e-296, 1e-294)). Algebraically identical (17, D1).
        // |g| >= ~tiny wherever theta's limiter admits a flux (Qm >= tiny),
        // so every factor here is a normal number.
        fFace[f] = (std::abs(g[f])*(dt/V[d]))/thL[d];
        Fout[d] += fFace[f];
    }
    for (Label c = 0; c < nCells; ++c)
    {
        st.FoutMax = std::max(st.FoutMax, Fout[c]);
        if (!(Fout[c] < 1))
        {
            st.error = "donor outflow fraction >= 1 (theta's bound broken)";
            st.where = c;
            return st;
        }
    }
    for (int k = 0; k < nComp; ++k)
    {
        std::vector<Scalar> inc(nCells, Scalar(0));
        for (Label f = 0; f < nFaces; ++f)
        {
            if (g[f] == 0)
            {
                continue;
            }
            const bool fromOwner = g[f] > 0;
            const Label d = fromOwner ? own[f] : nei[f];
            const Label r = fromOwner ? nei[f] : own[f];
            // Field units too: the donor's share per RECEIVER volume,
            // P_d f (V_d/V_r), never the subnormal intermediate P_d f V_d.
            inc[r] += Plow[k][d]*(fFace[f]*(V[d]/V[r]));
        }
        for (Label c = 0; c < nCells; ++c)
        {
            Pout[k][c] = Plow[k][c]*(Scalar(1) - Fout[c]) + inc[c];
        }
    }
    return st;
}


//- C+: after C, the composition antidiffusion
//    E_k = lambda_theta (A_Pk - s_k A_theta),  s_k = P_k,low/sum P_low (donor),
//  whose sum over k is lambda_theta (A_sumP - A_theta) -- zero when the
//  partition sums to theta, solver tolerance otherwise -- limited by ONE mu,
//  the minimum over the components of limiterR1(P_k after C, P_k old, E_k),
//  and applied to every component. mu never reaches theta.
template<class Scalar, class Label>
inline Stats<Scalar> composition
(
    const Label nCells, const Label nFaces,
    const Label* own, const Label* nei,
    const Scalar s, const Scalar* Vrel,
    const Scalar* lamTheta, const Scalar* ATheta, const Scalar* g,
    const int nComp,
    const Scalar* const* Plow, const Scalar* const* Pold,
    const Scalar* const* APk,
    Scalar* const* P,              // in: after C; out: after C+
    const MuMode mode = MuMode::limited,
    std::vector<Scalar>* muOut = nullptr,                 // S17 A: mu, E_k
    std::vector<std::vector<Scalar>>* EOut = nullptr      // for passengers()
)
{
    Stats<Scalar> st{Scalar(0), Scalar(1), Scalar(1), 0, nullptr, -1};
    for (Label f = 0; f < nFaces; ++f)
    {
        bool ok = finite(lamTheta[f]) && finite(ATheta[f]) && finite(g[f]);
        for (int k = 0; k < nComp && ok; ++k)
        {
            ok = finite(APk[k][f]);
        }
        if (!ok)
        {
            st.error = "non-finite flux in the composition correction";
            st.where = f;
            return st;
        }
    }
    for (Label c = 0; c < nCells; ++c)
    {
        for (int k = 0; k < nComp; ++k)
        {
            if (!finite(Pold[k][c]) || !finite(P[k][c]) || P[k][c] < 0)
            {
                st.error = "non-finite or negative partition before C+";
                st.where = c;
                return st;
            }
        }
    }

    std::vector<Scalar> sP(nCells);
    for (Label c = 0; c < nCells; ++c)
    {
        Scalar acc = Plow[0][c];
        for (int k = 1; k < nComp; ++k)
        {
            acc = acc + Plow[k][c];
        }
        sP[c] = acc;
    }

    std::vector<std::vector<Scalar>> E(nComp, std::vector<Scalar>(nFaces));
    std::vector<Scalar> mu(nFaces, Scalar(1)), l(nFaces);
    for (int k = 0; k < nComp; ++k)
    {
        for (Label f = 0; f < nFaces; ++f)
        {
            // donor side as the replica takes it: owner where g > 0, else
            // the neighbour (L189-199)
            const Label d = g[f] > 0 ? own[f] : nei[f];
            const Scalar den = sP[d] > 0 ? sP[d] : Scalar(1);
            const Scalar sk = Plow[k][d]/den;
            E[k][f] = lamTheta[f]*(APk[k][f] - sk*ATheta[f]);
        }
        limiterR1<Scalar, Label>
        (
            nCells, nFaces, own, nei, P[k], Pold[k], E[k].data(), s, Vrel,
            l.data()
        );
        for (Label f = 0; f < nFaces; ++f)
        {
            mu[f] = std::min(mu[f], l[f]);
        }
    }
    if (mode == MuMode::forceOne)
    {
        std::fill(mu.begin(), mu.end(), Scalar(1));
    }
    else if (mode == MuMode::forceZero)
    {
        std::fill(mu.begin(), mu.end(), Scalar(0));
    }

    Scalar num = 0, den = 0;
    for (Label f = 0; f < nFaces; ++f)
    {
        Scalar w = 0;
        for (int k = 0; k < nComp; ++k)
        {
            w += std::abs(E[k][f]);
        }
        if (w > 0)
        {
            ++st.nActive;
            st.muMin = std::min(st.muMin, mu[f]);
            num += mu[f]*w;
            den += w;
        }
    }
    st.muMean = den > 0 ? num/den : Scalar(1);

    for (int k = 0; k < nComp; ++k)
    {
        applyR1<Scalar, Label>
        (
            nCells, nFaces, own, nei, mu.data(), E[k].data(), s, Vrel, P[k]
        );
    }
    if (muOut) { *muOut = mu; }
    if (EOut) { *EOut = E; }
    return st;
}


//- S17 A (research/designs/2026-09-26-S17-frozen-release-implicit-elastic-
//  startup.md, 2A; docket 12 section 1, 08/10): species carried BY one
//  component P_k -- thetaP's fibre content C_B = thetaP B_P and frozen
//  tension q_P = thetaP p_P -- moved by exactly P_k's C and C+ fluxes:
//    C   X_low (1 - F_out) + sum_in X_low(donor) f (V_donor/V_receiver),
//        the expression donorShares() applies to P_k, applied to X (so the
//        passenger per unit P_k of a donor leaves at the donor's ratio);
//    C+  spec(donor) mu E_k, spec = X/P_k after C of the cell LOSING P_k
//        across the face (0 where it holds none), in field units as
//        applyR1 applies mu E_k.
//  X may be signed (off-diagonal components, q_P <= 0). No limiter of X's
//  own: a separate one breaks X/P_k's convex-combination bound (07, 17's
//  lesson). Returns nullptr, or why the step is refused.
template<class Scalar, class Label>
inline const char* passengers
(
    const Label nCells, const Label nFaces,
    const Label* own, const Label* nei,
    const Scalar* V,
    const Scalar* g, const std::vector<Scalar>& fFace,
    const Scalar s, const Scalar* Vrel,
    const Scalar* PkAfterC,
    const std::vector<Scalar>& mu, const std::vector<Scalar>& Ek,
    const int nX, const Scalar* const* Xlow, Scalar* const* Xout
)
{
    if
    (
        Label(fFace.size()) != nFaces || Label(mu.size()) != nFaces
     || Label(Ek.size()) != nFaces
    )
    {
        return "passenger flux arrays do not match the faces";
    }
    std::vector<Scalar> FoutX(nCells, Scalar(0));
    for (Label f = 0; f < nFaces; ++f)
    {
        if (g[f] == 0) { continue; }
        FoutX[g[f] > 0 ? own[f] : nei[f]] += fFace[f];
    }
    for (int x = 0; x < nX; ++x)
    {
        for (Label c = 0; c < nCells; ++c)
        {
            if (!finite(Xlow[x][c]))
            {
                return "non-finite passenger predictor";
            }
        }
        std::vector<Scalar> gain(nCells, Scalar(0));
        for (Label f = 0; f < nFaces; ++f)
        {
            if (g[f] == 0) { continue; }
            const bool ownerGives = g[f] > 0;
            const Label giver = ownerGives ? own[f] : nei[f];
            const Label taker = ownerGives ? nei[f] : own[f];
            gain[taker] += Xlow[x][giver]*(fFace[f]*(V[giver]/V[taker]));
        }
        for (Label c = 0; c < nCells; ++c)
        {
            Xout[x][c] = Xlow[x][c]*(Scalar(1) - FoutX[c]) + gain[c];
        }
        std::vector<Scalar> divX(nCells, Scalar(0));
        for (Label f = 0; f < nFaces; ++f)
        {
            const Scalar la = mu[f]*(Ek[f]*s);
            if (la == 0) { continue; }
            const Label loser = la > 0 ? own[f] : nei[f];
            const Scalar spec =
                PkAfterC[loser] > 0 ? Xout[x][loser]/PkAfterC[loser] : Scalar(0);
            const Scalar laX = spec*la;
            divX[own[f]] += laX;
            divX[nei[f]] -= laX;
        }
        for (Label c = 0; c < nCells; ++c)
        {
            Xout[x][c] = Xout[x][c] - (Scalar(1)/Vrel[c])*divX[c];
            if (!finite(Xout[x][c]))
            {
                return "non-finite passenger after C+";
            }
        }
    }
    return nullptr;
}

//- S17b A1, V1 (docket 07 section 2, 06; design section 2): THE TAUT SHARE
//  theta_T OF THE PENDING CONTENT P_k, carried by exactly P_k's C and C+
//  fluxes (the donor's taut fraction), with theta_T <= P_k EXACT in floating
//  point -- structural, not a tolerance:
//    C   X_afterC = X_low (1 - F_out) + sum_in X_low(donor) f V_d/V_r, the
//        donorShares expression, monotone in X_low: X_low <= P_low (the
//        predictors run the same matrix and the same fixed sweeps; checked
//        here, refused otherwise) gives X_afterC <= P_afterC exactly.
//    C+  the taut fraction f = X_afterC/P_afterC <= 1 (monotone division);
//        per cell, with out/in the |mu E_k s|/Vrel P_k leaves/receives,
//          A = f_c (P_c - out_c) + sum_in f_donor in,   (taut)
//          B =     (P_c - out_c) + sum_in       in,   (all)
//        in the SAME operation order, so A <= B exactly, and
//          theta_T = P_new (A/B) <= P_new exactly.
//  A negative (P_c - out_c) would break the monotonicity; the limiter makes
//  it >= 0, and a violation is refused, never bounded. Returns nullptr, or
//  why the step is refused.
template<class Scalar, class Label>
inline const char* tautShare
(
    const Label nCells, const Label nFaces,
    const Label* own, const Label* nei,
    const Scalar* V,
    const Scalar* g, const std::vector<Scalar>& fFace,
    const Scalar s, const Scalar* Vrel,
    const Scalar* PkLow, const Scalar* PkAfterC, const Scalar* PkNew,
    const std::vector<Scalar>& mu, const std::vector<Scalar>& Ek,
    const Scalar* XLow, Scalar* XNew
)
{
    for (Label c = 0; c < nCells; ++c)
    {
        if (!finite(XLow[c]) || XLow[c] < 0)
        {
            return "non-finite or negative taut predictor";
        }
        if (XLow[c] > PkLow[c])
        {
            return "taut content above the pending content after the predictor";
        }
    }
    // C: the donorShares expression, applied to X
    std::vector<Scalar> FoutT(nCells, Scalar(0)), gainT(nCells, Scalar(0));
    for (Label f = 0; f < nFaces; ++f)
    {
        if (g[f] == 0) { continue; }
        FoutT[g[f] > 0 ? own[f] : nei[f]] += fFace[f];
    }
    for (Label f = 0; f < nFaces; ++f)
    {
        if (g[f] == 0) { continue; }
        const bool ownerGives = g[f] > 0;
        const Label giver = ownerGives ? own[f] : nei[f];
        const Label taker = ownerGives ? nei[f] : own[f];
        gainT[taker] += XLow[giver]*(fFace[f]*(V[giver]/V[taker]));
    }
    std::vector<Scalar> fr(nCells);
    for (Label c = 0; c < nCells; ++c)
    {
        const Scalar xc = XLow[c]*(Scalar(1) - FoutT[c]) + gainT[c];
        if (xc > PkAfterC[c])
        {
            return "taut content above the pending content after C";
        }
        fr[c] = PkAfterC[c] > 0 ? xc/PkAfterC[c] : Scalar(0);
    }
    // C+: the A/B form
    std::vector<Scalar> out(nCells, Scalar(0)), inA(nCells, Scalar(0)),
        inB(nCells, Scalar(0));
    for (Label f = 0; f < nFaces; ++f)
    {
        const Scalar la = mu[f]*(Ek[f]*s);
        if (la == 0) { continue; }
        const Label loser = la > 0 ? own[f] : nei[f];
        const Label gainer = la > 0 ? nei[f] : own[f];
        const Scalar a = std::abs(la);
        out[loser] += a;
        inA[gainer] += fr[loser]*(a/Vrel[gainer]);
        inB[gainer] += a/Vrel[gainer];
    }
    for (Label c = 0; c < nCells; ++c)
    {
        const Scalar kept = PkAfterC[c] - out[c]/Vrel[c];
        if (kept < 0)
        {
            return "the pending content's C+ outflow exceeds the cell's content";
        }
        const Scalar A = fr[c]*kept + inA[c];
        const Scalar B = kept + inB[c];
        XNew[c] = B > 0 ? PkNew[c]*(A/B) : Scalar(0);
        if (!finite(XNew[c]))
        {
            return "non-finite taut content after C+";
        }
    }
    return nullptr;
}

//- S18p (research/designs/2026-10-01-S18p-em76-attribution-probe.md rev. 2,
//  section 2): passengersShared()'s per-cell locals, COPIED out when the
//  caller passes a non-null pointer (the solver does so only under
//  projSiteLog yes). No arithmetic is added to the kernel. Per passenger x
//  and cell c at [x*nCells + c]: A, spec, inX; per cell: kept, inB, Bdiv
//  (the A/B divisor B, before its B > 0 selection), PkAfterC, PkNew.
//  nCells stays 0 until every passenger has been written.
template<class Scalar>
struct PassengersDebug
{
    long nCells = 0;
    int nX = 0;
    std::vector<Scalar> A, spec, inX;
    std::vector<Scalar> kept, inB, Bdiv, PkAfterC, PkNew;
};

//- S18 E (H9; design section 2 E; docket 07 item 3): the passengers of P_k
//  -- C_B's six components and q_P -- in tautShare()'s operation order, so
//  the fibre's contents and its taut share are carried by ONE arithmetic:
//    C   X_afterC = X_low (1 - F_out) + sum_in X_low(donor) f V_d/V_r
//        (the donorShares expression, as passengers() and tautShare());
//    C+  spec = X_afterC/P_afterC (0 where P_afterC holds none); per cell,
//          A = spec_c kept_c + sum_in spec_donor |mu E s|/Vrel,
//          B = kept_c + sum_in |mu E s|/Vrel,  kept = P_afterC - out/Vrel,
//        X_new = P_new (A/B) (0 where B <= 0).
//  With the same P_new as the partition's, X_new/P_new = A/B is a convex
//  combination of the stencil's ratios (A and B carry the same weights), so
//  C_B/theta and q_P/theta cannot leave their donors' hull at the Zalesak
//  keep margin, where the shipped X - divX/Vrel raised 2401 of 352 491
//  P-LS projections (07). X may be signed. Each face and cell sum runs in
//  tautShare()'s order, so a passenger X <= P_k gives tautShare()'s result
//  bit for bit (the kernel test holds that). Refuses a C+ outflow above
//  the cell's content, as tautShare() does.
template<class Scalar, class Label>
inline const char* passengersShared
(
    const Label nCells, const Label nFaces,
    const Label* own, const Label* nei,
    const Scalar* V,
    const Scalar* g, const std::vector<Scalar>& fFace,
    const Scalar s, const Scalar* Vrel,
    const Scalar* PkAfterC, const Scalar* PkNew,
    const std::vector<Scalar>& mu, const std::vector<Scalar>& Ek,
    const int nX, const Scalar* const* Xlow, Scalar* const* Xout,
    PassengersDebug<Scalar>* dbg = nullptr
)
{
    if
    (
        Label(fFace.size()) != nFaces || Label(mu.size()) != nFaces
     || Label(Ek.size()) != nFaces
    )
    {
        return "passenger flux arrays do not match the faces";
    }
    const std::size_t nDbg = dbg ? std::size_t(nCells) : 0;
    if (dbg)
    {
        dbg->nCells = 0;
        dbg->nX = nX;
        dbg->A.assign(std::size_t(nX)*nDbg, Scalar(0));
        dbg->spec.assign(std::size_t(nX)*nDbg, Scalar(0));
        dbg->inX.assign(std::size_t(nX)*nDbg, Scalar(0));
        dbg->kept.assign(nDbg, Scalar(0));
        dbg->inB.assign(nDbg, Scalar(0));
        dbg->Bdiv.assign(nDbg, Scalar(0));
        dbg->PkAfterC.assign(PkAfterC, PkAfterC + nDbg);
        dbg->PkNew.assign(PkNew, PkNew + nDbg);
    }
    std::vector<Scalar> FoutX(nCells, Scalar(0));
    for (Label f = 0; f < nFaces; ++f)
    {
        if (g[f] == 0) { continue; }
        FoutX[g[f] > 0 ? own[f] : nei[f]] += fFace[f];
    }
    std::vector<Scalar> out(nCells, Scalar(0)), inB(nCells, Scalar(0));
    for (Label f = 0; f < nFaces; ++f)
    {
        const Scalar la = mu[f]*(Ek[f]*s);
        if (la == 0) { continue; }
        const Label donor = la > 0 ? own[f] : nei[f];
        const Label receiver = la > 0 ? nei[f] : own[f];
        const Scalar a = std::abs(la);
        out[donor] += a;
        inB[receiver] += a/Vrel[receiver];
    }
    for (Label c = 0; c < nCells; ++c)
    {
        if (PkAfterC[c] - out[c]/Vrel[c] < 0)
        {
            return "the pending content's C+ outflow exceeds the cell's content";
        }
    }
    for (int x = 0; x < nX; ++x)
    {
        for (Label c = 0; c < nCells; ++c)
        {
            if (!finite(Xlow[x][c]))
            {
                return "non-finite passenger predictor";
            }
        }
        std::vector<Scalar> gainX(nCells, Scalar(0));
        for (Label f = 0; f < nFaces; ++f)
        {
            if (g[f] == 0) { continue; }
            const bool ownerGives = g[f] > 0;
            const Label from = ownerGives ? own[f] : nei[f];
            const Label to = ownerGives ? nei[f] : own[f];
            gainX[to] += Xlow[x][from]*(fFace[f]*(V[from]/V[to]));
        }
        std::vector<Scalar> spec(nCells);
        for (Label c = 0; c < nCells; ++c)
        {
            const Scalar xc = Xlow[x][c]*(Scalar(1) - FoutX[c]) + gainX[c];
            // the division never sees 0, then the guard selects (an
            // optimiser may evaluate both arms of a ternary: fix round 25d)
            const bool has = PkAfterC[c] > 0;
            const Scalar pd = has ? PkAfterC[c] : Scalar(1);
            const Scalar q = xc/pd;
            spec[c] = has ? q : Scalar(0);
        }
        std::vector<Scalar> inX(nCells, Scalar(0));
        for (Label f = 0; f < nFaces; ++f)
        {
            const Scalar la = mu[f]*(Ek[f]*s);
            if (la == 0) { continue; }
            const Label donor = la > 0 ? own[f] : nei[f];
            const Label receiver = la > 0 ? nei[f] : own[f];
            const Scalar a = std::abs(la);
            inX[receiver] += spec[donor]*(a/Vrel[receiver]);
        }
        for (Label c = 0; c < nCells; ++c)
        {
            const Scalar kept = PkAfterC[c] - out[c]/Vrel[c];
            const Scalar A = spec[c]*kept + inX[c];
            const Scalar B = kept + inB[c];
            const bool hasB = B > 0;
            const Scalar Bd = hasB ? B : Scalar(1);
            const Scalar v = PkNew[c]*(A/Bd);
            Xout[x][c] = hasB ? v : Scalar(0);
            if (!finite(Xout[x][c]))
            {
                return "non-finite passenger after C+";
            }
            if (dbg)
            {
                const std::size_t at = std::size_t(x)*nDbg + std::size_t(c);
                dbg->A[at] = A;
                dbg->spec[at] = spec[c];
                dbg->inX[at] = inX[c];
                dbg->kept[c] = kept;
                dbg->inB[c] = inB[c];
                dbg->Bdiv[c] = B;
            }
        }
    }
    if (dbg)
    {
        dbg->nCells = long(nCells);
    }
    return nullptr;
}

} // End namespace damagePartitionKernel

#endif
