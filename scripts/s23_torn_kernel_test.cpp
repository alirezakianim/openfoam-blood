// S23 rev. 4.1 kernel test (design research/designs/2026-10-07-S23-torn-network-debris.md
// section 13.2 items 1, 2, 6 and 7; docket 20261007-0840-oi55-frame-analysis report 19 Q3).
//
// Compiled against the solver's own header damagePartitionKernel.H (the torn set's
// arithmetic: progressiveCeiling, tornThreshold, tornMember, tornStressWeight,
// f1WeightDeviation, pointRings; d* is the .C's expression (pP + pB)/S, 0 at S = 0), so a change to what the solver
// runs is a change to what this test runs. Run by scripts/check_s23_torn_network.py:
//   c++ -std=c++14 -O2 -ffp-contract=off -I <solver dir> s23_torn_kernel_test.cpp
// Exit 0 = every case passes; 1 = a case fails (each prints PASS/FAIL).
//
// THE 2-STEP TEST (report 19 Q3). A 1-D chain N -> P -> D of equal volume, carrier
// Courant c = F dt/V, the inflow face carrying intact content theta. theta = 0.02 in
// every cell (ind = theta/phi0 = 0.0444, isSkel), thI = theta, thP = thB = 0, ref_N =
// ref_D = theta, ref_P = 30 theta. The partition's low-order step is implicit upwind
// (the solver's thetaIntactEqn, an fvMatrix solve, replicated here for a chain:
// P_c = (P_c,old + c P_c-1)/(1 + c)); the antidiffusive part is zero, so the
// solver's donorShares (theta's limited flux g = 0) returns the low-order state.
// The event is the solver's rupture statements for d*_law = 1 (volumetric branch,
// isSkel AND ref >= R_c theta): thI = S(1 - d*_new), thP = S pend/(1 + r), thB =
// S (d*_new - pend/(1 + r)), then A1 ref <- theta.
// Step 1: transport, event at P (R_P = 27.36 >= 21.95): d*_P = 1, T_P = 1 in rev. 3
// and rev. 4. Step 2: transport only: d*_P = 1/(1 + c) = 0.909091 at c = 0.1. Rev. 3's
// predicate (d* == 1, the .C before S23 rev. 4) gives T_P = 0 (the defect); rev. 4's
// tornMember gives 1. The negative c = 0.6 gives d*_P = 0.625 < 0.646774: T_P = 0
// (with ref_P = 40 theta: at report 19's 30 theta, R_P = 19.1 and nothing fires).

#include "damagePartitionKernel.H"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace K = damagePartitionKernel;

static int nFail = 0;

// d* exactly as tornNetworkCells() and the rupture law evaluate it
static double dStarOf(const double thI, const double thP, const double thB)
{
    const double S = thI + (thP + thB);
    return S > 0 ? (thP + thB)/S : 0.0;
}

static void check(const bool ok, const char* name, const double got, const double want)
{
    std::printf("%s %s: got %.17g, want %.17g\n", ok ? "PASS" : "FAIL", name, got, want);
    if (!ok)
    {
        ++nFail;
    }
}

struct Chain
{
    // cells 0 = N, 1 = P, 2 = D; components of the partition, and ref
    double thI[3], thP[3], thB[3], ref[3], theta[3];
};

// implicit upwind for one component on the chain, inflow value pIn on N's face
static void upwind(double* P, const double c, const double pIn)
{
    double up = pIn;
    for (int k = 0; k < 3; ++k)
    {
        P[k] = (P[k] + c*up)/(1.0 + c);
        up = P[k];
    }
}

// the solver's step composition with theta's limited antidiffusive flux g = 0:
// donorShares returns the low-order state (outflow fraction 0, no inflow)
static bool donorIdentity(Chain& s)
{
    const long own[2] = {0, 1}, nei[2] = {1, 2};
    const double V[3] = {1, 1, 1}, g[2] = {0, 0};
    double* lo[4] = {s.thI, s.thP, s.thB, s.ref};
    double out[4][3];
    double* po[4] = {out[0], out[1], out[2], out[3]};
    std::vector<double> fFace;
    const K::Stats<double> st = K::donorShares<double, long>
    (
        3L, 2L, own, nei, V, 1.0, s.theta, g, 4, lo, po, fFace
    );
    if (st.error)
    {
        return false;
    }
    for (int k = 0; k < 4; ++k)
    {
        for (int c = 0; c < 3; ++c)
        {
            lo[k][c] = out[k][c];
        }
    }
    return true;
}

static void transport(Chain& s, const double c, const double thetaIn)
{
    upwind(s.thI, c, thetaIn);  // the inflow carries intact content
    upwind(s.thP, c, 0.0);
    upwind(s.thB, c, 0.0);
    upwind(s.ref, c, thetaIn);  // inflow network at its deposit (R = 1)
    upwind(s.theta, c, thetaIn);
}

// the volumetric branch's event (d*_law = 1) and A1, as the solver applies it
static void event(Chain& s, const int k, const double r)
{
    const double sC = s.thI[k] + (s.thP[k] + s.thB[k]);
    const double d = sC > 0 ? s.thB[k]/sC : 0.0;
    const double dsNew = 1.0;
    const double pend = dsNew - d;
    const double pendNew = pend/(1.0 + r);
    const double dNew = dsNew - pendNew;
    s.thI[k] = sC*(1.0 - dsNew);
    s.thP[k] = sC*pendNew;
    s.thB[k] = sC*dNew;
    if (s.thI[k] == 0)
    {
        s.ref[k] = s.theta[k];
    }
}

struct TwoStep
{
    double dStarP, dStarD, wP, aIP, dP;
    bool rev3P, rev4P, rev4D, rev4N, fired, donorOk;
};

static TwoStep twoStep(const double c, const double thr, const double refP)
{
    const double theta = 0.02, phi0 = 0.45, floorW = 1e-4, Rc = 21.95, r = 0.5;
    Chain s;
    for (int k = 0; k < 3; ++k)
    {
        s.thI[k] = theta;
        s.thP[k] = 0;
        s.thB[k] = 0;
        s.ref[k] = theta;
        s.theta[k] = theta;
    }
    s.ref[1] = refP*theta;
    TwoStep o{};
    // step 1: transport -> event
    transport(s, c, theta);
    o.donorOk = donorIdentity(s);
    o.fired = s.ref[1] >= Rc*s.theta[1];
    if (o.fired)
    {
        event(s, 1, r);
    }
    // step 2: transport (no re-event: R_P = 1)
    transport(s, c, theta);
    o.donorOk = o.donorOk && donorIdentity(s);
    o.fired = o.fired && !(s.ref[1] >= Rc*s.theta[1]);
    o.dStarP = dStarOf(s.thI[1], s.thP[1], s.thB[1]);
    o.dStarD = dStarOf(s.thI[2], s.thP[2], s.thB[2]);
    const double SP = s.thI[1] + (s.thP[1] + s.thB[1]);
    o.aIP = s.thI[1]/SP;
    o.dP = s.thB[1]/SP;
    o.rev3P = true && o.dStarP == 1;  // rev. 3's T: isSkel AND d* == 1
    o.rev4P = K::tornMember(true, o.dStarP, thr);
    o.rev4D = K::tornMember(true, o.dStarD, thr);
    o.rev4N = K::tornMember(true, dStarOf(s.thI[0], s.thP[0], s.thB[0]), thr);
    o.wP = K::tornStressWeight(s.theta[1]/phi0, floorW, o.dStarP);
    return o;
}

int main()
{
    // 1. the threshold, from the law's own expression (never a literal): the
    //    shipped parameters (lc 2.8, lf 2.105, dlf 1.0) and a second set
    const double dp = K::progressiveCeiling(2.8, 2.105, 1.0);
    const double thr = K::tornThreshold(dp);
    check(std::fabs(dp - (1.0 - std::exp(-(2.8 - 2.105)/2.0))) == 0, "d_p = law at lc (2.8, 2.105, 1)",
          dp, 1.0 - std::exp(-(2.8 - 2.105)/2.0));
    check(std::fabs(thr - 0.646774) < 1e-6, "threshold (1 + d_p)/2 = 0.646774 (report 19)", thr, 0.646774);
    const double dp2 = K::progressiveCeiling(3.0, 2.0, 0.5);
    check(std::fabs(dp2 - (1.0 - std::exp(-1.0))) < 1e-15, "d_p = 1 - e^-1 at (3, 2, 0.5)", dp2,
          1.0 - std::exp(-1.0));
    check(std::fabs(K::tornThreshold(dp2) - 0.5*(2.0 - std::exp(-1.0))) < 1e-15,
          "threshold at (3, 2, 0.5)", K::tornThreshold(dp2), 0.5*(2.0 - std::exp(-1.0)));

    // 2. the 2-step test, c = 0.1
    const TwoStep a = twoStep(0.1, thr, 30.0);
    check(a.donorOk && a.fired, "c 0.1: step 1 fires at P, step 2 does not; donorShares ran", a.fired, 1);
    check(std::fabs(a.dStarP - 1.0/1.1) < 1e-12, "c 0.1: d*_P after step 2 = 1/(1 + c)", a.dStarP, 1.0/1.1);
    check(!a.rev3P, "c 0.1: rev. 3 (d* == 1) gives T_P 0, the defect", a.rev3P, 0);
    check(a.rev4P, "c 0.1: rev. 4 (majority) gives T_P 1", a.rev4P, 1);
    check(std::fabs(a.dStarD - 0.1/(1.1*1.1)) < 1e-12 && !a.rev4D && !a.rev4N,
          "c 0.1: d*_D = c/(1 + c)^2, T_D 0 and T_N 0 (majority does not spread)", a.dStarD, 0.1/(1.21));
    check(std::fabs(a.wP - (0.02/0.45)*(1.0 - 1.0/1.1)) < 1e-15, "c 0.1: w_P = ind (1 - d*) = 4.04e-3", a.wP,
          (0.02/0.45)*(1.0 - 1.0/1.1));

    // 3. the planted negative: c = 0.6 gives d*_P 0.625 < 0.646774, T_P 0.
    //    At report 19's ref_P = 30 theta the step-1 transport gives R_P =
    //    30.6/1.6 = 19.1 < 21.95: NO event, d* = 0, and the negative would
    //    pass for the wrong reason. ref_P = 40 theta (R_P 25.4) fires it.
    const TwoStep b0 = twoStep(0.6, thr, 30.0);
    check(!b0.fired && b0.dStarP == 0, "c 0.6, ref_P 30 theta: no event (R_P 19.1 < R_c)", b0.dStarP, 0);
    const TwoStep b = twoStep(0.6, thr, 40.0);
    check(b.donorOk && b.fired, "c 0.6, ref_P 40 theta: step 1 fires at P, step 2 does not", b.fired, 1);
    check(std::fabs(b.dStarP - 0.625) < 1e-12, "c 0.6: d*_P = 0.625", b.dStarP, 0.625);
    check(!b.rev4P && !b.rev3P, "c 0.6: T_P 0 in rev. 4 too (the threshold is live)", b.rev4P, 0);

    // 4. bitwise rev. 3 at d* = 1: the weight is +0.0 (rev. 3 wrote 0.0)
    {
        bool same = true;
        const double zero = 0.0;
        for (double ind = 1e-6; ind < 1.5; ind *= 1.7)
        {
            const double w = K::tornStressWeight(ind, 1e-4, 1.0);
            same = same && std::memcmp(&w, &zero, sizeof w) == 0;
        }
        check(same, "w at d* = 1 is +0.0 bit for bit over ind 1e-6..1.5", same, 1);
    }

    // 5. F1' (G23-5): w/max(ind, floor) = aI on T, to round-off, for the
    //    weight as the builders consume it; each one-site defect form fails
    {
        const double occ = std::max(0.02/0.45, 1e-4);
        const double devOk = K::f1WeightDeviation(a.wP, occ, a.aIP);
        const double devZero = K::f1WeightDeviation(0.0, occ, a.aIP);
        const double devD = K::f1WeightDeviation(occ*(1.0 - a.dP), occ, a.aIP);
        check(devOk < 1e-15, "F1': the rev. 4 weight passes (|w/occ - aI| at round-off)", devOk, 0);
        check(devZero > 1e-3, "F1': w = 0 at a builder FAILS (deviation = aI)", devZero, a.aIP);
        check(devD > 1e-3, "F1': w = occ (1 - d) at a builder FAILS (deviation = pending share)", devD,
              a.dStarP - a.dP);
    }

    // 6. Pi: two point-neighbour rings of T = the Chebyshev-2 box on a
    //    structured 9 x 9 quad grid (cell (i, j) has points (i..i+1, j..j+1))
    {
        const int n = 9;
        std::vector<std::vector<long>> cp(n*n), pc((n + 1)*(n + 1));
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                const long c = j*n + i;
                const long p[4] = {j*(n + 1) + i, j*(n + 1) + i + 1,
                                   (j + 1)*(n + 1) + i, (j + 1)*(n + 1) + i + 1};
                for (const long q : p)
                {
                    cp[c].push_back(q);
                    pc[q].push_back(c);
                }
            }
        }
        std::vector<char> seed(n*n, 0);
        seed[4*n + 4] = 1;      // the centre
        seed[0] = 1;            // a corner
        std::vector<int> ring;
        const long nPi = K::pointRings(cp, pc, seed, 2, ring);
        long nCheb = 0;
        bool same = true;
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                const int dc = std::max(std::abs(i - 4), std::abs(j - 4));
                const int d0 = std::max(i, j);
                const int cheb = std::min(dc, d0);
                const bool in = cheb <= 2;
                nCheb += in;
                same = same && (in == (ring[j*n + i] <= 2)) && (!in || ring[j*n + i] == cheb);
            }
        }
        check(same && nPi == nCheb, "Pi = Chebyshev <= 2 of T (rings = distance), 33 cells", double(nPi),
              double(nCheb));
    }

    std::printf("%s (%d failed)\n", nFail ? "S23 KERNEL TEST FAILED" : "S23 KERNEL TEST PASSED", nFail);
    return nFail ? 1 : 0;
}
