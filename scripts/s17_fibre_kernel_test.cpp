// S17 A/C kernel arithmetic test (dispatch 16). Compiled and run by
// scripts/check_s17_fibre_release.py and scripts/build_main_case_colab_package.sh:
//   c++ -std=c++14 -O2 -ffp-contract=off -I<solver dir> this.cpp
// -ffp-contract=off: the Colab VM is x86-64 without FMA, so no contraction.
//
// It tests the SOLVER'S OWN arithmetic (fibreReleaseKernel.H, included
// unchanged). Design: research/designs/2026-09-26-S17-frozen-release-implicit-
// elastic-startup.md, revision 4. Tests and their sources (docket
// research/investigations/20260926-0900-run112658-s16-first-return/):
//   K1-K5  P-LS (12 section 1, unit test (a)-(d); P-b1 the mutation negative)
//   K6-K8  ML rotation and UNI (09 section 2 S3; G-rot, G-eig, G-uni)
//   K9     G-dep (09 rule 2: a unit deposit into a fully broken cell -> I)
//   K10    bit-identity of mixedStiffness/splitDev where nothing is damaged
//   K11    fibreStretch solves lf^3 - T lf - 1 = 0
//   KP     the passengers (damagePartitionKernel::passengers): C_B and q_P
//          ride P2's C and C+ fluxes, so X/P2 stays a convex combination
//          of the stencil's ratios, and a uniform ratio is kept exactly
//   K12-K16 S17b (design 2026-09-27, revision 2): the event value, V1's
//          slack step, the carrier shares, the taut share <= P2 exactly
//          through C and C+ and its donor-following, and its refusals
//          (scripts/check_s17b_slack_fibre.py)
//   K17-K18 S18 (design 2026-09-29, rev. 10): A's applied frozen share with
//          the frozenPower unit negative, and E's shared C+ order
//          (scripts/check_s18_soft.py)
//   UA     A: the pending stiffness term in the explicit bound (10, the
//          patch configurations; term deleted -> radius > 2)
//   UC     C: the consistent Gamma_t (10, the single cell; the design and
//          08 forms -> 1.094)
// The operator in UA/UC is a transcription of 10's probe
// numerics-analyst-unit-test.py (the linear replica of the solver's explicit
// elastic step, 11 x 11 periodic); the STIFFNESS it is fed and the face
// Gamma are the kernel's mixedStiffness -- the solver's formula.
//
// Prints one "PASS <name>" or "FAIL <name>: ..." per test; exit 1 on any FAIL.

#include "fibreReleaseKernel.H"
#include "damagePartitionKernel.H"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace fibreReleaseKernel;

static int failures = 0;
static void report(const char* name, bool ok, const char* detail = "")
{
    std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, ok ? "" : ": ",
                ok ? "" : detail);
    if (!ok) { ++failures; }
}

static const double LC = 2.8;

// A rounding floor for the energy comparisons, which are differences of
// numbers of order 3 (tr B - 3): 16*3*eps = 24 ulp(3). ARITHMETIC, not a tolerance
// on the physics -- K2b measures that without it the exact inequality
// false-fires on near-isotropic blends, and that P-b1 still fails with it.
static const double PSI_FLOOR = fibreReleaseKernel::psiFloor;

static void randomAxis(std::mt19937_64& g, double n[3], bool planar)
{
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (;;)
    {
        n[0] = u(g); n[1] = u(g); n[2] = planar ? 0.0 : u(g);
        const double m = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (m > 0.1 && m <= 1.0)
        {
            n[0] /= m; n[1] /= m; n[2] /= m;
            return;
        }
    }
}

// ---------------------------------------------------------------- P-LS
static void testProjection()
{
    std::mt19937_64 g(12345);
    std::uniform_real_distribution<double> uL(1.0, LC), u01(0.0, 1.0);

    // K1 idempotence on random fibres (12 (a)): <= 1e-12
    {
        double worst = 0;
        for (int t = 0; t < 2000; ++t)
        {
            double n[3];
            randomAxis(g, n, t % 2 == 0);
            const double lf = uL(g);
            double B[6];
            fibre(n, lf, B);
            const Projection P = projectLS(B, n);
            double Bo[6];
            fibre(P.n, P.lf, Bo);
            double e = std::abs(P.lf - lf)/lf;
            for (int k = 0; k < 6; ++k)
            {
                e = std::max(e, std::abs(Bo[k] - B[k])/(lf*lf));
            }
            worst = std::max(worst, e);
        }
        char buf[128];
        std::snprintf(buf, sizeof buf, "max relative error %.3e > 1e-12", worst);
        report("K1 P-LS idempotent on 2000 fibres", worst <= 1e-12, buf);
        std::printf("   K1 max relative error %.3e\n", worst);
    }

    // K2 random blends of 2-4 families (12 (b)): lf_out <= max lf_in,
    // T_out <= sum w T_i, psi_out <= sum w psi_i, each to x(1 + 1e-12)
    // (+ PSI_FLOOR on the energy, an arithmetic floor measured below).
    // K4 the same blends through P-b1 (lf^2 = lambda_max(B)): must violate
    // the stress bound (12 (d): 20875 of 200000, worst +31.1%).
    {
        long nBad = 0, nBadNoFloor = 0, nB1 = 0;
        double worstB1 = 0;
        const long N = 200000;
        for (long t = 0; t < N; ++t)
        {
            const int m = 2 + int(t % 3);
            double w[4], sw = 0;
            double Bmix[6] = {0, 0, 0, 0, 0, 0};
            double lfMax = 0, Tsum = 0, psiSum = 0;
            double nFirst[3];
            for (int i = 0; i < m; ++i)
            {
                w[i] = u01(g) + 1e-3;
                sw += w[i];
            }
            for (int i = 0; i < m; ++i)
            {
                double n[3];
                randomAxis(g, n, true);
                if (i == 0) { nFirst[0] = n[0]; nFirst[1] = n[1]; nFirst[2] = n[2]; }
                const double lf = uL(g);
                double B[6];
                fibre(n, lf, B);
                const double wi = w[i]/sw;
                for (int k = 0; k < 6; ++k) { Bmix[k] += wi*B[k]; }
                lfMax = std::max(lfMax, lf);
                Tsum += wi*fibreTension(lf);
                psiSum += wi*energy(B);
            }
            const Projection P = projectLS(Bmix, nFirst);
            const double tol = 1.0 + 1e-12;
            const double Tout = fibreTension(P.lf);
            const bool bad =
                !(P.ok)
             || P.lf > lfMax*tol
             || Tout > Tsum*tol + PSI_FLOOR
             || P.psiOut > psiSum*tol + PSI_FLOOR
             || P.psiOut > P.psiMix*tol + PSI_FLOOR
             || Tout < 0;
            const bool badNoFloor =
                P.psiOut > psiSum*tol || P.psiOut > P.psiMix*tol;
            if (bad) { ++nBad; }
            if (badNoFloor) { ++nBadNoFloor; }
            // P-b1: lf^2 = lambda_max(B_mix)
            double A[3][3], ev[3], V[3][3];
            toMatrix(Bmix, A);
            eigSym(A, ev, V);
            const double lb1 = std::sqrt(ev[0]);
            const double Tb1 = fibreTension(lb1);
            if (Tb1 > Tsum*tol + PSI_FLOOR)
            {
                ++nB1;
                worstB1 = std::max(worstB1, Tb1/Tsum - 1.0);
            }
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "%ld of %ld blends violate a bound", nBad, N);
        report("K2 P-LS never raises lf, tension or energy (200000 blends)", nBad == 0, buf);
        std::printf("   K2 violations %ld; without the %.2e arithmetic floor %ld\n",
                    nBad, PSI_FLOOR, nBadNoFloor);
        std::snprintf(buf, sizeof buf, "P-b1 raised tension in %ld blends (must be > 0)", nB1);
        report("K4 mutation negative: P-b1 fails the stress bound", nB1 > 0, buf);
        std::printf("   K4 P-b1 raises tension in %ld of %ld blends, worst +%.1f%%\n",
                    nB1, N, 100.0*worstB1);
    }

    // K2b THE FLOOR, MEASURED: near-isotropic blends (lf within 1e-9 of 1,
    // the tails a transport step leaves) false-fire the exact (1 + 1e-12)
    // form by up to ~1.3e-15 absolute -- rounding of tr(B) - 3 -- and pass
    // with PSI_FLOOR (24 ulp(3), 12x the worst measured). The solver's
    // G-proj carries the same floor.
    {
        std::uniform_real_distribution<double> uT(1.0, 1.0 + 1e-9);
        long noFloor = 0, withFloor = 0;
        double worst = 0;
        for (long t = 0; t < 200000; ++t)
        {
            const int m = 2 + int(t % 3);
            double w[4], sw = 0, Bmix[6] = {0, 0, 0, 0, 0, 0}, nFirst[3];
            for (int i = 0; i < m; ++i) { w[i] = u01(g) + 1e-3; sw += w[i]; }
            for (int i = 0; i < m; ++i)
            {
                double n[3];
                randomAxis(g, n, true);
                if (i == 0) { nFirst[0] = n[0]; nFirst[1] = n[1]; nFirst[2] = n[2]; }
                double B[6];
                fibre(n, uT(g), B);
                for (int k = 0; k < 6; ++k) { Bmix[k] += w[i]/sw*B[k]; }
            }
            const Projection P = projectLS(Bmix, nFirst);
            if (P.psiOut > P.psiMix*(1.0 + 1e-12))
            {
                ++noFloor;
                worst = std::max(worst, P.psiOut - P.psiMix);
            }
            if (!dissipative(P)) { ++withFloor; }
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "without the floor %ld (worst %.3e), with it %ld",
                      noFloor, worst, withFloor);
        report("K2b the G-proj floor is arithmetic: needed without, 0 with", noFloor > 0 && withFloor == 0, buf);
        std::printf("   K2b %s\n", buf);
    }

    // K3 the 0/90 degree tie (12 (c)): T_out/T_in = 0.25 +- 1e-12, n the
    // unit projection of the previous axis onto the tied eigenspace.
    {
        const double ex[3] = {1, 0, 0}, ey[3] = {0, 1, 0};
        double Bx[6], By[6], Bm[6];
        fibre(ex, LC, Bx);
        fibre(ey, LC, By);
        for (int k = 0; k < 6; ++k) { Bm[k] = 0.5*Bx[k] + 0.5*By[k]; }
        const double a = 30.0*M_PI/180.0;
        const double nPrev[3] = {std::cos(a), std::sin(a), 0};
        const Projection P = projectLS(Bm, nPrev);
        const double ratio = P.Tls/fibreTension(LC);
        const double align = std::abs(P.n[0]*nPrev[0] + P.n[1]*nPrev[1] + P.n[2]*nPrev[2]);
        char buf[160];
        std::snprintf(buf, sizeof buf, "tie %d, T_out/T_in %.15f, |n.nPrev| %.15f",
                      int(P.tie), ratio, align);
        report("K3 0/90 tie: T_out/T_in = 0.25, n = the tie-break vector",
               P.tie && std::abs(ratio - 0.25) <= 1e-12 && std::abs(align - 1.0) <= 1e-12,
               buf);
        std::printf("   K3 %s\n", buf);
    }

    // K5 aligned families at lf 2.8 and 1, equal weights: T_out = sum w T_i
    // = 3.7414 exactly (12 S1).
    {
        const double ex[3] = {1, 0, 0};
        double B1[6], B2[6], Bm[6];
        fibre(ex, LC, B1);
        fibre(ex, 1.0, B2);
        for (int k = 0; k < 6; ++k) { Bm[k] = 0.5*B1[k] + 0.5*B2[k]; }
        const Projection P = projectLS(Bm, ex);
        const double Tsum = 0.5*fibreTension(LC) + 0.5*fibreTension(1.0);
        const double Tout = fibreTension(P.lf);
        char buf[128];
        std::snprintf(buf, sizeof buf, "T_out %.12f, sum w T %.12f", Tout, Tsum);
        report("K5 aligned 2.8 and 1: T_out = sum w T_i",
               std::abs(Tout - Tsum) <= 1e-12*Tsum && std::abs(Tsum - 3.7414) < 5e-5, buf);
    }
}

// ------------------------------------------------------ rotation and UNI
static void testRotation()
{
    std::mt19937_64 g(777);
    std::uniform_real_distribution<double> u(-1.0, 1.0), uL(1.0, LC), uE(-6.7, -0.7);

    // K6 G-eig on 2000 random steps, |L| dt from ~2e-7 to 0.2
    // K8 G-uni: lf never increases; and it decreases by exactly mu < 1
    double eigWorst = 0;
    long nUp = 0, nMuWrong = 0, nNotOk = 0;
    for (int t = 0; t < 2000; ++t)
    {
        double L[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        L[0][0] = u(g); L[0][1] = u(g); L[1][0] = u(g); L[1][1] = u(g);
        double nrm = 0;
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) nrm += L[i][j]*L[i][j];
        nrm = std::sqrt(nrm);
        const double dt = std::pow(10.0, uE(g))/nrm;
        double n[3];
        randomAxis(g, n, true);
        double lf = uL(g);
        const double lf0 = lf;
        double Bout[6];
        const Rotation r = rotateStretch(L, dt, n, lf, Bout);
        if (!r.ok) { ++nNotOk; continue; }
        eigWorst = std::max(eigWorst, r.eigRel);
        if (lf > lf0) { ++nUp; }
        const double expect = lf0*std::min(1.0, r.mu);
        if (lf != expect) { ++nMuWrong; }
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "max eigenvalue change %.3e (> 1e-12), not ok %ld",
                  eigWorst, nNotOk);
    report("K6 G-eig: B_P eigenvalues preserved across the rotation (2000 steps)",
           eigWorst <= 1e-12 && nNotOk == 0, buf);
    std::printf("   K6 max relative eigenvalue change %.3e\n", eigWorst);
    std::snprintf(buf, sizeof buf, "%ld increases, %ld not lf*min(1, mu)", nUp, nMuWrong);
    report("K8 G-uni: lf never increases; lf' = lf min(1, mu)", nUp == 0 && nMuWrong == 0, buf);

    // K7 G-rot at |L| dt = 1e-3: median angle/(dt omega) = 1.000 +- 0.001
    // (09 S3: 1.000010). Mutation: applied twice reads ~2.
    std::vector<double> once, twice;
    for (int t = 0; t < 2000; ++t)
    {
        double L[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        L[0][0] = u(g); L[0][1] = u(g); L[1][0] = u(g); L[1][1] = -L[0][0];
        double nrm = 0;
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) nrm += L[i][j]*L[i][j];
        nrm = std::sqrt(nrm);
        const double dt = 1e-3/nrm;
        double n[3];
        randomAxis(g, n, true);
        double lf = uL(g);
        double n2[3] = {n[0], n[1], n[2]};
        double lf2 = lf;
        double Bout[6];
        const Rotation r = rotateStretch(L, dt, n, lf, Bout);
        if (!(r.omega > 1e-2*r.Lnorm)) { continue; }
        once.push_back(r.angle/(dt*r.omega));
        // the mutation: the same step applied twice in one time step
        const Rotation r1 = rotateStretch(L, dt, n2, lf2, Bout);
        const Rotation r2 = rotateStretch(L, dt, n2, lf2, Bout);
        twice.push_back((r1.angle + r2.angle)/(dt*r1.omega));
    }
    auto median = [](std::vector<double> v)
    {
        std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
        return v[v.size()/2];
    };
    // K7c the new axis IS the material line: n' = F n/|F n| (to 1e-12)
    {
        double worst = 0;
        for (int t = 0; t < 2000; ++t)
        {
            double L[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            L[0][0] = u(g); L[0][1] = u(g); L[1][0] = u(g); L[1][1] = u(g);
            double nrm = 0;
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) nrm += L[i][j]*L[i][j];
            const double dt = std::pow(10.0, uE(g))/std::sqrt(nrm);
            double n[3];
            randomAxis(g, n, true);
            double F[3][3];
            stepGradient(L, dt, F);
            double m[3], mm = 0;
            for (int i = 0; i < 3; ++i)
            {
                m[i] = F[i][0]*n[0] + F[i][1]*n[1] + F[i][2]*n[2];
                mm += m[i]*m[i];
            }
            mm = std::sqrt(mm);
            double lf = uL(g), Bout[6];
            rotateStretch(L, dt, n, lf, Bout);
            for (int i = 0; i < 3; ++i)
            {
                worst = std::max(worst, std::abs(n[i] - m[i]/mm));
            }
        }
        char b2[96];
        std::snprintf(b2, sizeof b2, "max |n' - F n/|F n|| = %.3e", worst);
        report("K7c the rotated axis is the material line F n/|F n|", worst <= 1e-12, b2);
    }
    const double m1 = median(once), m2 = median(twice);
    std::snprintf(buf, sizeof buf, "median %.6f over %zu steps", m1, once.size());
    report("K7 G-rot: median angle/(dt omega) = 1.000 +- 0.001", std::abs(m1 - 1.0) <= 1e-3, buf);
    std::snprintf(buf, sizeof buf, "double application median %.6f (must read ~2)", m2);
    report("K7m G-rot mutation: a double application reads 2", std::abs(m2 - 2.0) <= 0.01, buf);
    std::printf("   K7 median once %.6f, twice %.6f (%zu steps)\n", m1, m2, once.size());
}

// ------------------------------------------------------------- G-dep
static void testDeposit()
{
    // Solver, deposit mixing (.C, S11 FIX 6 block), under fibreRelease:
    //   rIntact = r*aI, aI = P1/S of the PRE-deposit partition (09 rule 2),
    //   b = (rIntact*b + d*I)/(rIntact + d).
    // A fully broken cell: P1 = 0, so aI = 0 and b must come out I exactly.
    double aI, aP, omd;
    shares(0.0, 0.3, 0.7, aI, aP, omd);
    const double r = 0.45, d = 1e-6;
    const double rIntact = r*aI;
    const double b[6] = {5.1, 1.7, 0.0, 0.4, 0.0, 0.48};
    const double I[6] = {1, 0, 0, 1, 0, 1};
    double bn[6];
    for (int k = 0; k < 6; ++k)
    {
        bn[k] = (rIntact*b[k] + d*I[k])/(rIntact + d);
    }
    bool isI = true;
    for (int k = 0; k < 6; ++k) { isI = isI && bn[k] == I[k]; }
    report("K9 G-dep: a unit deposit into a fully broken cell leaves be = I", aI == 0.0 && isI);
    // negative: the S16 weight r*(1 - d) (d = 0.7) keeps the old strain
    const double rS16 = r*omd;
    for (int k = 0; k < 6; ++k)
    {
        bn[k] = (rS16*b[k] + d*I[k])/(rS16 + d);
    }
    report("K9m G-dep negative: S16's r*(1 - d) does not leave I", bn[XX] != 1.0);
}

// ------------------------------------------------ bit identity, cubic
static void testIdentity()
{
    std::mt19937_64 g(99);
    std::uniform_real_distribution<double> u(-3.0, 3.0), uS(0.5, 20.0);
    bool okK = true, okM = true;
    for (int t = 0; t < 10000; ++t)
    {
        const double stiff = uS(g);
        if (mixedStiffness(1.0, stiff, 0.0, LC, 1.0) != stiff) { okK = false; }
        double dv[6], M[6];
        for (int k = 0; k < 6; ++k) { dv[k] = u(g); }
        if (t % 7 == 0) { dv[XZ] = -0.0; dv[YZ] = -0.0; }
        const double n[3] = {1, 0, 0};
        splitDev(1.0, dv, 0.0, LC, n, 1.0, M);
        for (int k = 0; k < 6; ++k)
        {
            if (M[k] != dv[k] || std::signbit(M[k]) != std::signbit(dv[k])) { okM = false; }
        }
    }
    report("K10a mixedStiffness is stiff bit for bit with no damage", okK);
    // K10e the split with a pending share: [(aI) dev(be) + (aP) T dev(n n)]
    // /(1 - d), against an independent evaluation; and the stiffness.
    {
        const double dv[6] = {0.7, -0.3, 0.0, -0.2, 0.0, -0.5};
        const double a = 25.0*M_PI/180.0;
        const double n[3] = {std::cos(a), std::sin(a), 0.0};
        const double aI = 0.3, aP = 0.5, omd = 0.8, lf = 2.3;
        const double T = lf*lf - 1.0/lf;
        const double nn[6] = {n[0]*n[0] - 1.0/3.0, n[0]*n[1], 0.0,
                              n[1]*n[1] - 1.0/3.0, 0.0, -1.0/3.0};
        double M[6];
        splitDev(aI, dv, aP, lf, n, omd, M);
        double e = 0;
        for (int k = 0; k < 6; ++k)
        {
            e = std::max(e, std::abs(M[k] - (aI*dv[k] + aP*T*nn[k])/omd));
        }
        const double k1 = mixedStiffness(aI, 1.6504, aP, lf, omd);
        const double k0 = (aI*1.6504 + aP*T)/omd;
        char b3[96];
        std::snprintf(b3, sizeof b3, "split error %.3e, stiffness error %.3e", e, std::abs(k1 - k0));
        report("K10e splitDev and mixedStiffness carry the pending fibre",
               e <= 1e-14 && std::abs(k1 - k0) <= 1e-14*k0, b3);
    }
    report("K10b splitDev is dev(be) bit for bit (signed zeros too) with no damage", okM);
    double aI, aP, omd;
    shares(0.3, 0.0, 0.0, aI, aP, omd);
    report("K10c shares of an undamaged cell are exactly (1, 0, 1)", aI == 1.0 && aP == 0.0 && omd == 1.0);
    shares(0.0, 0.0, 0.0, aI, aP, omd);
    report("K10d an empty cell reads (1, 0, 1)", aI == 1.0 && aP == 0.0 && omd == 1.0);

    double worst = 0;
    for (int t = 0; t <= 1000; ++t)
    {
        const double T = 8.0*t/1000.0;
        const double l = fibreStretch(T);
        const double res = std::abs((l*l - T)*l - 1.0)/std::max(1.0, T*l);
        worst = std::max(worst, res);
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, "worst scaled residual %.3e", worst);
    report("K11 fibreStretch solves lf^3 - T lf - 1 = 0 (T in [0, 8])",
           worst <= 4e-16*8 && fibreStretch(0.0) == 1.0, buf);
}


// ------------------------------------------------- passengers (KP)
static void testPassengers()
{
    std::mt19937_64 g(2026);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const int N = 400, F = N - 1;
    std::vector<int> own(F), nei(F);
    for (int f = 0; f < F; ++f) { own[f] = f; nei[f] = f + 1; }
    long nBound = 0, nUniform = 0;
    double worstU = 0;
    for (int trial = 0; trial < 200; ++trial)
    {
        std::vector<double> V(N), Vrel(N), P(N), r(N), thL(N, 1.0), gF(F), fF(F);
        double Vmin = 1e300;
        for (int c = 0; c < N; ++c)
        {
            V[c] = 1.0 + u01(g);
            Vmin = std::min(Vmin, V[c]);
            P[c] = 1e-3 + u01(g);
            r[c] = -1.0 + u01(g);              // a q_P-like signed ratio
        }
        for (int c = 0; c < N; ++c) { Vrel[c] = V[c]/Vmin; }
        // donor fractions f with F_out < 1 per donor
        std::vector<double> Fout(N, 0.0);
        for (int f = 0; f < F; ++f)
        {
            gF[f] = (u01(g) < 0.5 ? -1.0 : 1.0)*(0.1 + u01(g));
            if (u01(g) < 0.1) { gF[f] = 0.0; }
            fF[f] = gF[f] == 0 ? 0.0 : 0.45*u01(g);
        }
        // P after C, the donorShares expression
        std::vector<double> PC(N);
        {
            std::vector<double> inc(N, 0.0);
            for (int f = 0; f < F; ++f)
            {
                if (gF[f] == 0) continue;
                const int d = gF[f] > 0 ? own[f] : nei[f];
                const int q = gF[f] > 0 ? nei[f] : own[f];
                Fout[d] += fF[f];
                inc[q] += P[d]*(fF[f]*(V[d]/V[q]));
            }
            for (int c = 0; c < N; ++c) { PC[c] = P[c]*(1.0 - Fout[c]) + inc[c]; }
        }
        // a composition flux mu E s that removes at most a fifth of either
        // side's P after C (the limiter's job; here by construction)
        const double s = 1.0;
        std::vector<double> mu(F), E(F);
        for (int f = 0; f < F; ++f)
        {
            mu[f] = u01(g);
            const double cap = 0.2*std::min(PC[own[f]]*Vrel[own[f]], PC[nei[f]]*Vrel[nei[f]]);
            E[f] = (2.0*u01(g) - 1.0)*cap;
        }
        std::vector<double> PF(PC);
        {
            std::vector<double> div(N, 0.0);
            for (int f = 0; f < F; ++f)
            {
                const double la = mu[f]*(E[f]*s);
                div[own[f]] += la; div[nei[f]] -= la;
            }
            for (int c = 0; c < N; ++c) { PF[c] = PF[c] - (1.0/Vrel[c])*div[c]; }
        }
        std::vector<double> X(N), Xu(N), Xo(N), Xuo(N);
        const double rU = -0.37;
        for (int c = 0; c < N; ++c) { X[c] = r[c]*P[c]; Xu[c] = rU*P[c]; }
        const double* XlP[2] = {X.data(), Xu.data()};
        double* XoP[2] = {Xo.data(), Xuo.data()};
        const char* err = damagePartitionKernel::passengers<double, int>
        (
            N, F, own.data(), nei.data(), V.data(), gF.data(), fF, s,
            Vrel.data(), PC.data(), mu, E, 2, XlP, XoP
        );
        if (err) { report("KP passengers run", false, err); return; }
        for (int c = 0; c < N; ++c)
        {
            // the stencil, radius 2: C mixes c with its face neighbours,
            // and C+ then moves a neighbour's POST-C ratio (itself a mix of
            // that neighbour's neighbours) across the face
            double lo = r[c], hi = r[c];
            for (int k = std::max(0, c - 2); k <= std::min(N - 1, c + 2); ++k)
            {
                lo = std::min(lo, r[k]);
                hi = std::max(hi, r[k]);
            }
            const double ratio = Xo[c]/PF[c];
            const double tol = 1e-12*(1.0 + std::abs(lo) + std::abs(hi));
            if (!(ratio >= lo - tol && ratio <= hi + tol)) { ++nBound; }
            const double eu = std::abs(Xuo[c]/PF[c] - rU)/std::abs(rU);
            worstU = std::max(worstU, eu);
            if (eu > 1e-12) { ++nUniform; }
        }
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "%ld cells outside their stencil's ratio range", nBound);
    report("KP passengers: X/P2 a convex combination of the stencil's ratios", nBound == 0, buf);
    std::snprintf(buf, sizeof buf, "%ld cells lost a uniform ratio (worst %.3e)", nUniform, worstU);
    report("KP passengers: a uniform ratio is kept", nUniform == 0, buf);
    std::printf("   KP uniform-ratio worst relative error %.3e\n", worstU);
}

// ----------------------------------------------- S17b V1 (K12-K16)
static void testS17b()
{
    // K12 the event unit test (design section 4; 12 change 3): l1 3.0
    // freezes lf = 2.8; below lc the event takes l1 itself.
    {
        const double e1 = eventStretch(9.0, LC);
        const double e2 = eventStretch(4.0, LC);
        char b[96];
        std::snprintf(b, sizeof b, "l1 3.0 -> %.15g, l1 2.0 -> %.15g", e1, e2);
        report("K12 the event value: l1 3.0 freezes lf = 2.8", e1 == LC && e2 == 2.0, b);
    }
    // K13 the slack step (07 section 2; 05 D1's sequence; design section
    // 8 b): lf 1.02 at mu 0.99 -> 1.0098, taut, q_P scaled by T ratio;
    // again -> slack (lf 1, q ratio 0); a fibre at lf = 1 is slack whatever
    // mu is (A1 as written); a taut step equals S17's UNI value bit for bit.
    {
        bool ok = true;
        double lf = 1.02, lo, q;
        const bool s1 = slackStep(lf, 0.99, lo, q);
        ok = ok && !s1 && lo == 1.02*std::min(1.0, 0.99)
             && q == fibreTension(lo)/fibreTension(1.02);
        const bool s2 = slackStep(lo, 0.99, lf, q);
        ok = ok && s2 && lf == 1.0 && q == 0.0;
        ok = ok && slackStep(1.0, 1.0, lo, q) && lo == 1.0 && q == 0.0;
        ok = ok && slackStep(1.0, 1.5, lo, q) && lo == 1.0 && q == 0.0;
        ok = ok && slackStep(1.0, 0.999999, lo, q) && q == 0.0;
        // a taut fibre that does not contract keeps q_P (ratio 1)
        ok = ok && !slackStep(1.5, 1.2, lo, q) && lo == 1.5 && q == 1.0;
        // every random step: slack only with mu < 1, taut value = UNI's
        std::mt19937_64 g(31);
        std::uniform_real_distribution<double> uL(1.0, LC), uM(0.9, 1.1);
        long bad = 0;
        for (int t = 0; t < 100000; ++t)
        {
            const double l = uL(g), m = uM(g);
            double lo2, q2;
            const bool sl = slackStep(l, m, lo2, q2);
            if (sl && !(m < 1 || l == 1)) { ++bad; }
            if (!sl && lo2 != l*std::min(1.0, m)) { ++bad; }
            if (!sl && !(lo2 > 1.0)) { ++bad; }
            if (!sl && (q2 < 0 || q2 > 1)) { ++bad; }
        }
        char b[96];
        std::snprintf(b, sizeof b, "%ld bad random steps", bad);
        report("K13 V1 slack step: mu < 1 or lf = 1, lf > 1 if taut, q_P ratio in [0, 1]", ok && bad == 0, b);
    }
    // K13b THE BOUNDARY (section 8 b): lf min(1, mu) = 1 EXACTLY is slack
    // (2 x 0.5 = 1 in binary); one ulp above it is taut; lf = 1 with
    // mu >= 1 is slack. A test "lfNew < 1" or "mu < 1 &&" fails here.
    {
        double lo = 0, q = -1;
        const bool atOne = slackStep(2.0, 0.5, lo, q);
        const bool okAt = atOne && lo == 1.0 && q == 0.0;
        const double muUp = std::nextafter(0.5, 1.0);
        const bool above = slackStep(2.0, muUp, lo, q);
        const bool okAbove = !above && lo == 2.0*muUp && lo > 1.0 && q > 0.0 && q < 1.0;
        const bool rest = slackStep(1.0, 1.05, lo, q);
        const bool okRest = rest && lo == 1.0 && q == 0.0;
        char b[128];
        std::snprintf(b, sizeof b, "at 1: %d, one ulp above: %d, lf 1 at mu 1.05: %d",
                      int(atOne), int(above), int(rest));
        report("K13b the slack boundary lf min(1, mu) = 1 is slack, above it taut", okAt && okAbove && okRest, b);
    }
    // K14 sharesCarrier with C = P2 is shares() bit for bit; with C = theta_T
    // the pending share is theta_T/S.
    {
        std::mt19937_64 g(5);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        bool ok = true;
        for (int t = 0; t < 10000; ++t)
        {
            const double P1 = u(g), P2 = u(g), P3 = u(g), T = u(g)*P2;
            double a1, b1, c1, a2, b2, c2;
            shares(P1, P2, P3, a1, b1, c1);
            sharesCarrier(P1, P2, P3, P2, a2, b2, c2);
            ok = ok && a1 == a2 && b1 == b2 && c1 == c2;
            sharesCarrier(P1, P2, P3, T, a2, b2, c2);
            ok = ok && b2 == T/(P1 + (P2 + P3));
        }
        report("K14 sharesCarrier(C = P2) is shares() bit for bit", ok);
    }
    // K15 THE TAUT SHARE IS <= P2 EXACTLY through C and C+ (tautShare), on
    // random 1-D chains with theta_T = P2 in a third of the cells (the
    // 1-ulp-tight case), and a uniform taut fraction is kept.
    {
        std::mt19937_64 g(77);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        const int N = 300, F = N - 1;
        std::vector<int> own(F), nei(F);
        for (int f = 0; f < F; ++f) { own[f] = f; nei[f] = f + 1; }
        long above = 0, errs = 0;
        double worstU = 0;
        for (int trial = 0; trial < 300; ++trial)
        {
            std::vector<double> V(N), Vrel(N), P(N), X(N), g0(F), fF(F);
            double Vmin = 1e300;
            for (int c = 0; c < N; ++c)
            {
                V[c] = 1.0 + u01(g); Vmin = std::min(Vmin, V[c]);
                P[c] = std::pow(10.0, -300.0*u01(g)*(trial % 2)) * (1e-3 + u01(g));
                const double r = u01(g);
                X[c] = r < 0.33 ? P[c] : (r < 0.5 ? 0.0 : P[c]*u01(g));
            }
            for (int c = 0; c < N; ++c) { Vrel[c] = V[c]/Vmin; }
            std::vector<double> Fo(N, 0.0), inc(N, 0.0), PC(N);
            for (int f = 0; f < F; ++f)
            {
                g0[f] = u01(g) < 0.1 ? 0.0 : (u01(g) < 0.5 ? -1.0 : 1.0);
                fF[f] = g0[f] == 0 ? 0.0 : 0.45*u01(g);
            }
            for (int f = 0; f < F; ++f)
            {
                if (g0[f] == 0) continue;
                const int d = g0[f] > 0 ? own[f] : nei[f];
                const int q = g0[f] > 0 ? nei[f] : own[f];
                Fo[d] += fF[f];
                inc[q] += P[d]*(fF[f]*(V[d]/V[q]));
            }
            for (int c = 0; c < N; ++c) { PC[c] = P[c]*(1.0 - Fo[c]) + inc[c]; }
            std::vector<double> mu(F), E(F), PN(PC);
            for (int f = 0; f < F; ++f)
            {
                mu[f] = u01(g);
                const double cap = 0.2*std::min(PC[own[f]]*Vrel[own[f]], PC[nei[f]]*Vrel[nei[f]]);
                E[f] = (2.0*u01(g) - 1.0)*cap;
            }
            {
                std::vector<double> div(N, 0.0);
                for (int f = 0; f < F; ++f)
                {
                    const double la = mu[f]*(E[f]*1.0);
                    div[own[f]] += la; div[nei[f]] -= la;
                }
                for (int c = 0; c < N; ++c) { PN[c] = PN[c] - (1.0/Vrel[c])*div[c]; }
            }
            std::vector<double> XN(N), XU(N), XUN(N);
            const char* e = damagePartitionKernel::tautShare<double, int>
            (
                N, F, own.data(), nei.data(), V.data(), g0.data(), fF, 1.0,
                Vrel.data(), P.data(), PC.data(), PN.data(), mu, E, X.data(), XN.data()
            );
            if (e) { ++errs; continue; }
            for (int c = 0; c < N; ++c) { if (XN[c] > PN[c]) { ++above; } }
            for (int c = 0; c < N; ++c) { XU[c] = 0.4*P[c]; }
            e = damagePartitionKernel::tautShare<double, int>
            (
                N, F, own.data(), nei.data(), V.data(), g0.data(), fF, 1.0,
                Vrel.data(), P.data(), PC.data(), PN.data(), mu, E, XU.data(), XUN.data()
            );
            if (e) { ++errs; continue; }
            for (int c = 0; c < N; ++c)
            {
                if (PN[c] > 1e-280) { worstU = std::max(worstU, std::abs(XUN[c]/PN[c] - 0.4)); }
            }
        }
        char b[128];
        std::snprintf(b, sizeof b, "%ld cells with theta_T > P2, %ld refusals, uniform-fraction error %.3e", above, errs, worstU);
        report("K15 the taut share <= P2 exactly through C and C+", above == 0 && errs == 0, b);
        report("K15b a uniform taut fraction is kept (to 1e-12)", worstU <= 1e-12, b);
        std::printf("   K15 %s\n", b);
    }
    // K15c/K15d THE TAUT SHARE FOLLOWS ITS DONOR: two cells of equal P2,
    // all-taut (0) beside all-slack (1). Through C+ alone (K15c) and through
    // C alone (K15d) half of cell 0's content moves to cell 1; the taut
    // content moved is the donor's, so cell 1 gains 0.5 of taut content and
    // the total is conserved (the fraction taken at the gainer moves none).
    {
        int own[1] = {0}, nei[1] = {1};
        double V[2] = {1, 1}, Vrel[2] = {1, 1}, P[2] = {1, 1}, X[2] = {1, 0}, XN[2];
        // K15c: C+ only (la = mu E s = 0.5 > 0: the owner loses)
        {
            double g0[1] = {0}, PC[2] = {1, 1}, PN[2] = {0.5, 1.5};
            std::vector<double> fF(1, 0.0), mu(1, 1.0), E(1, 0.5);
            const char* e = damagePartitionKernel::tautShare<double, int>
            (2, 1, own, nei, V, g0, fF, 1.0, Vrel, P, PC, PN, mu, E, X, XN);
            char b[96];
            std::snprintf(b, sizeof b, "X = (%.17g, %.17g)", XN[0], XN[1]);
            report("K15c C+ moves the donor's taut fraction", !e
                   && std::abs(XN[0] - 0.5) <= 1e-15 && std::abs(XN[1] - 0.5) <= 1e-15, b);
        }
        // K15d: C only (g > 0: the owner gives fFace 0.5)
        {
            double g0[1] = {1}, PC[2] = {0.5, 1.5};
            std::vector<double> fF(1, 0.5), mu(1, 0.0), E(1, 0.0);
            const char* e = damagePartitionKernel::tautShare<double, int>
            (2, 1, own, nei, V, g0, fF, 1.0, Vrel, P, PC, PC, mu, E, X, XN);
            char b[96];
            std::snprintf(b, sizeof b, "X = (%.17g, %.17g)", XN[0], XN[1]);
            report("K15d C moves the donor's taut content", !e
                   && std::abs(XN[0] - 0.5) <= 1e-15 && std::abs(XN[1] - 0.5) <= 1e-15, b);
        }
    }
    // K16 the refusals: a predictor with theta_T above P2 is refused, never
    // bounded (K16); so is a taut content above P2 after C (K16b). Each input
    // reaches only its own check: in K16 P2 after C is large enough that the
    // C check cannot fire, in K16b the predictor is within P2.
    {
        const int N = 2, F = 1;
        int own[1] = {0}, nei[1] = {1};
        double V[2] = {1, 1}, Vrel[2] = {1, 1}, g0[1] = {0}, XN[2];
        std::vector<double> fF(1, 0.0), mu(1, 0.0), E(1, 0.0);
        {
            double P[2] = {1, 1}, PC[2] = {2, 2}, X[2] = {1, 1.5};
            const char* e = damagePartitionKernel::tautShare<double, int>
            (N, F, own, nei, V, g0, fF, 1.0, Vrel, P, PC, PC, mu, E, X, XN);
            report("K16 a taut predictor above P2 is refused", e != nullptr);
        }
        {
            double P[2] = {1, 1}, PC[2] = {1, 0.5}, X[2] = {1, 1};
            const char* e = damagePartitionKernel::tautShare<double, int>
            (N, F, own, nei, V, g0, fF, 1.0, Vrel, P, PC, PC, mu, E, X, XN);
            report("K16b a taut content above P2 after C is refused", e != nullptr);
        }
        // K16c (design section 8 i): a C+ outflow larger than the cell's
        // pending content (kept < 0) is refused: la = mu E s = 2 takes 2
        // from cell 0, which holds 1 after C.
        {
            double P[2] = {1, 1}, PC[2] = {1, 1}, PN[2] = {1, 1}, X[2] = {0, 0};
            std::vector<double> mu1(1, 1.0), E2(1, 2.0);
            const char* e = damagePartitionKernel::tautShare<double, int>
            (N, F, own, nei, V, g0, fF, 1.0, Vrel, P, PC, PN, mu1, E2, X, XN);
            report("K16c a C+ outflow above the cell's content is refused", e != nullptr);
        }
    }
}


// ------------------------------------------- UA / UC: 10's unit test
namespace op
{
const int NX = 11, NY = 11, NC = NX*NY;
double TI[3][4], TP[3][4];
inline int id(int j, int i) { return ((j + NY) % NY)*NX + ((i + NX) % NX); }

void tangents()
{
    // T_int(B12): d(dev b)/dL, the solver's isochoric projection included
    const double B[3] = {1.44, 1/1.2, 1/1.2};
    const int idx[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    for (int col = 0; col < 4; ++col)
    {
        double Lm[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        Lm[idx[col][0]][idx[col][1]] = 1.0;
        double db[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                db[i][j] = Lm[i][j]*B[j] + B[i]*Lm[j][i];
        double trBi = 0;
        for (int i = 0; i < 3; ++i) { trBi += db[i][i]/B[i]; }
        for (int i = 0; i < 3; ++i) { db[i][i] -= trBi/3.0*B[i]; }
        const double tr = db[0][0] + db[1][1] + db[2][2];
        TI[0][col] = db[0][0] - tr/3.0;
        TI[1][col] = db[0][1];
        TI[2][col] = db[1][1] - tr/3.0;
        for (int r = 0; r < 3; ++r) { TP[r][col] = 0.0; }
    }
    TP[1][2] = fibreTension(LC);     // dS_xy = T L_yx, n = e_x
}

double series(double a, double b) { return (a + b > 0) ? 2*a*b/(a + b) : 0.0; }

struct Config { std::vector<bool> pend; std::vector<double> w; };

Config config(const char* name)
{
    Config c;
    c.pend.assign(NC, false);
    c.w.assign(NC, 1.0);
    if (std::string(name) == "single")
    {
        c.pend[id(5, 5)] = true;
    }
    else
    {
        const double sur = std::string(name) == "patch/0.1" ? 0.1 : 1e-3;
        for (int k = 0; k < NC; ++k) { c.w[k] = sur; }
        for (int j = 4; j < 7; ++j)
            for (int i = 4; i < 7; ++i)
            {
                c.pend[id(j, i)] = true;
                c.w[id(j, i)] = 1.0;
            }
    }
    return c;
}

// the cell's stiffness per unit weight, from the SOLVER'S formula: intact
// cells uniaxial at 1.2 (aI 1), pending cells a fibre at lc (aP 1).
double K12() { return stiffIntact(1.44, 2.0/1.2); }
double kcell(bool pend, bool termDeleted)
{
    const double aI = pend ? 0.0 : 1.0;
    const double aP = pend ? (termDeleted ? 0.0 : 1.0) : 0.0;
    return mixedStiffness(aI, K12(), aP, LC, 1.0);
}

struct Step
{
    const Config& c;
    double dt;
    bool implicit;
    std::vector<double> gE, gN;        // face Gamma (implicit only)
    std::vector<double> Minv;          // (I - Lm)^-1, NC x NC
    std::vector<double> Lm;
    Step(const Config& cc, double d) : c(cc), dt(d), implicit(false) {}

    void setGamma(const std::vector<double>& e, const std::vector<double>& n)
    {
        implicit = true;
        gE = e; gN = n;
        Lm.assign(NC*NC, 0.0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
            {
                const int a = id(j, i);
                const int bE = id(j, i + 1), bN = id(j + 1, i);
                const double ge = gE[a], gn = gN[a];
                Lm[a*NC + a] -= ge; Lm[bE*NC + bE] -= ge; Lm[a*NC + bE] += ge; Lm[bE*NC + a] += ge;
                Lm[a*NC + a] -= gn; Lm[bN*NC + bN] -= gn; Lm[a*NC + bN] += gn; Lm[bN*NC + a] += gn;
            }
        // Gauss-Jordan inverse of I - Lm (SPD, diagonally dominant)
        std::vector<double> A(NC*NC), X(NC*NC, 0.0);
        for (int r = 0; r < NC; ++r)
        {
            for (int s = 0; s < NC; ++s) { A[r*NC + s] = (r == s ? 1.0 : 0.0) - Lm[r*NC + s]; }
            X[r*NC + r] = 1.0;
        }
        for (int p = 0; p < NC; ++p)
        {
            int piv = p;
            for (int r = p + 1; r < NC; ++r)
                if (std::abs(A[r*NC + p]) > std::abs(A[piv*NC + p])) piv = r;
            if (piv != p)
                for (int s = 0; s < NC; ++s) { std::swap(A[p*NC + s], A[piv*NC + s]); std::swap(X[p*NC + s], X[piv*NC + s]); }
            const double inv = 1.0/A[p*NC + p];
            for (int s = 0; s < NC; ++s) { A[p*NC + s] *= inv; X[p*NC + s] *= inv; }
            for (int r = 0; r < NC; ++r)
            {
                if (r == p) continue;
                const double f = A[r*NC + p];
                if (f == 0) continue;
                for (int s = 0; s < NC; ++s) { A[r*NC + s] -= f*A[p*NC + s]; X[r*NC + s] -= f*X[p*NC + s]; }
            }
        }
        Minv = X;
    }

    // x = (S[NC][3], V[NC][2]) -> next
    void apply(const std::vector<double>& S, const std::vector<double>& V,
               std::vector<double>& Sn, std::vector<double>& Vn) const
    {
        Sn.assign(3*NC, 0.0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
            {
                const int a = id(j, i);
                const int e = id(j, i + 1), w = id(j, i - 1), n = id(j + 1, i), s = id(j - 1, i);
                const double Lv[4] =
                {
                    (V[2*e] - V[2*w])/2, (V[2*n] - V[2*s])/2,
                    (V[2*e + 1] - V[2*w + 1])/2, (V[2*n + 1] - V[2*s + 1])/2
                };
                const double (*T)[4] = c.pend[a] ? TP : TI;
                for (int r = 0; r < 3; ++r)
                {
                    double acc = 0;
                    for (int q = 0; q < 4; ++q) acc += T[r][q]*Lv[q];
                    Sn[3*a + r] = S[3*a + r] + dt*acc;
                }
            }
        std::vector<double> sE(3*NC), sN(3*NC), fx(NC), fy(NC), gxE(NC), gyN(NC);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
            {
                const int a = id(j, i), e = id(j, i + 1), n = id(j + 1, i);
                const double wE = series(c.w[a], c.w[e]), wN = series(c.w[a], c.w[n]);
                for (int r = 0; r < 3; ++r)
                {
                    sE[3*a + r] = wE*0.5*(Sn[3*a + r] + Sn[3*e + r]);
                    sN[3*a + r] = wN*0.5*(Sn[3*a + r] + Sn[3*n + r]);
                }
            }
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
            {
                const int a = id(j, i), w = id(j, i - 1), s = id(j - 1, i);
                fx[a] = (sE[3*a + 0] - sE[3*w + 0]) + (sN[3*a + 1] - sN[3*s + 1]);
                fy[a] = (sE[3*a + 1] - sE[3*w + 1]) + (sN[3*a + 2] - sN[3*s + 2]);
            }
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
            {
                const int a = id(j, i);
                gxE[a] = 0.5*(fx[a] + fx[id(j, i + 1)]);
                gyN[a] = 0.5*(fy[a] + fy[id(j + 1, i)]);
            }
        Vn.assign(2*NC, 0.0);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
            {
                const int a = id(j, i);
                Vn[2*a] = 0.5*(gxE[a] + gxE[id(j, i - 1)]);
                Vn[2*a + 1] = 0.5*(gyN[a] + gyN[id(j - 1, i)]);
            }
        if (!implicit) return;
        std::vector<double> rhs(2*NC);
        for (int comp = 0; comp < 2; ++comp)
        {
            for (int r = 0; r < NC; ++r)
            {
                double acc = 0;
                for (int s = 0; s < NC; ++s) acc += Lm[r*NC + s]*V[2*s + comp];
                rhs[2*r + comp] = Vn[2*r + comp] - acc;
            }
        }
        for (int comp = 0; comp < 2; ++comp)
        {
            for (int r = 0; r < NC; ++r)
            {
                double acc = 0;
                for (int s = 0; s < NC; ++s) acc += Minv[r*NC + s]*rhs[2*s + comp];
                Vn[2*r + comp] = acc;
            }
        }
    }
};

// spectral radius of the one-step map by power iteration: the mean log
// growth over the last `window` of `steps` normalised iterations.
double radius(const Step& st, long steps, long window)
{
    std::mt19937_64 g(4242);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> S(3*NC), V(2*NC), Sn, Vn;
    for (auto& x : S) x = u(g);
    for (auto& x : V) x = u(g);
    double logSum = 0;
    for (long k = 0; k < steps; ++k)
    {
        st.apply(S, V, Sn, Vn);
        double nrm = 0;
        for (double x : Sn) nrm += x*x;
        for (double x : Vn) nrm += x*x;
        nrm = std::sqrt(nrm);
        if (k >= steps - window) logSum += std::log(nrm);
        for (int q = 0; q < 3*NC; ++q) S[q] = Sn[q]/nrm;
        for (int q = 0; q < 2*NC; ++q) V[q] = Vn[q]/nrm;
    }
    return std::exp(logSum/window);
}
} // namespace op

static void testUnitA()
{
    using namespace op;
    for (const char* name : {"patch/0.1", "patch/1e-3"})
    {
        const Config c = config(name);
        double Nt = 0, Nd = 0;     // max w k per unit dt: with the term / deleted
        for (int a = 0; a < NC; ++a)
        {
            Nt = std::max(Nt, c.w[a]*kcell(c.pend[a], false));
            Nd = std::max(Nd, c.w[a]*kcell(c.pend[a], true));
        }
        const double dtT = 0.2/Nt, dtD = 0.2/Nd;
        const double rT = radius(Step(c, dtT), 300000, 100000);
        const double rD = radius(Step(c, dtD), 2000, 500);
        const double growth = std::log(std::max(rT, 1.0))/dtT;
        char buf[200];
        std::snprintf(buf, sizeof buf, "%s with the term: dt %.5g radius %.6f growth %.4g/model time (pass <= 0.01)",
                      name, dtT, rT, growth);
        report((std::string("UA ") + name + " with the pending term: growth <= 0.01/model time").c_str(),
               growth <= 0.01, buf);
        std::printf("   UA %s\n", buf);
        std::snprintf(buf, sizeof buf, "%s term deleted: dt %.5g radius %.6f (must be > 2)", name, dtD, rD);
        report((std::string("UAm ") + name + " term deleted: radius > 2").c_str(), rD > 2.0, buf);
        std::printf("   UAm %s\n", buf);
    }
}

static void testUnitC()
{
    using namespace op;
    for (const char* name : {"single", "patch/0.1", "patch/1e-3"})
    {
        const Config c = config(name);
        double Np = 0;
        for (int a = 0; a < NC; ++a)
        {
            if (c.pend[a]) Np = std::max(Np, c.w[a]*kcell(true, false));
        }
        const double dtC = 22.6/Np;
        std::vector<double> gE(NC), gN(NC), dE(NC), dN(NC), eE(NC), eN(NC);
        const double TF = fibreTension(LC);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
            {
                const int a = id(j, i), e = id(j, i + 1), n = id(j + 1, i);
                const double ka = kcell(c.pend[a], false);
                const double ke = kcell(c.pend[e], false), kn = kcell(c.pend[n], false);
                // consistent: series(w) interp(k_eff) -- the solver's Gamma_t
                gE[a] = dtC*series(c.w[a], c.w[e])*0.5*(ka + ke);
                gN[a] = dtC*series(c.w[a], c.w[n])*0.5*(ka + kn);
                // design: series(w_I) interp(stiff); 08: + series(w_P) T
                auto wI = [&](int q) { return c.pend[q] ? 0.0 : c.w[q]; };
                auto wP = [&](int q) { return c.pend[q] ? c.w[q] : 0.0; };
                dE[a] = dtC*series(wI(a), wI(e))*K12();
                dN[a] = dtC*series(wI(a), wI(n))*K12();
                eE[a] = dE[a] + dtC*series(wP(a), wP(e))*TF;
                eN[a] = dN[a] + dtC*series(wP(a), wP(n))*TF;
            }
        Step sc(c, dtC), sd(c, dtC), s8(c, dtC);
        sc.setGamma(gE, gN);
        sd.setGamma(dE, dN);
        s8.setGamma(eE, eN);
        const double rc = radius(sc, 6000, 2000);
        const double rd = radius(sd, 3000, 1000);
        const double r8 = radius(s8, 3000, 1000);
        const double gc = std::log(std::max(rc, 1.0))/dtC;
        const double gd = std::log(std::max(rd, 1.0))/dtC;
        const double g8 = std::log(std::max(r8, 1.0))/dtC;
        char buf[240];
        std::snprintf(buf, sizeof buf, "%s dt %.4g: consistent radius %.6f (%.4g/model time), design %.6f, 08 %.6f",
                      name, dtC, rc, gc, rd, r8);
        std::printf("   UC %s\n", buf);
        report((std::string("UC ") + name + " consistent Gamma_t: growth <= 0.01/model time").c_str(),
               gc <= 0.01, buf);
        if (std::string(name) == "single")
        {
            report("UCm single: the design form fails (radius ~1.094)", gd > 0.01 && rd > 1.09, buf);
            report("UCm single: 08's separate-share form fails (radius ~1.094)", g8 > 0.01 && r8 > 1.09, buf);
        }
        else
        {
            report((std::string("UCm ") + name + " the design form fails").c_str(), gd > 0.01, buf);
        }
    }
}

// ----------------------------------------------- S18 (K17-K18)
// Design research/designs/2026-09-29-S18-no-frozen-pressure-content-weight-
// network-viscosity.md, rev. 10: A's applied frozen share (K17, with the
// section 5 unit negative) and E's shared C+ order (K18).
static void testS18()
{
    // K17 A (R1): the frozen share the force applies, and the power it does.
    // THE UNIT NEGATIVE (section 5): A off, one synthetic transfer -- theta_T
    // 0.2 joins at the event with p_n = -100 (q_P = thPost pMix = -20),
    // S = 1, div Vs = 0.5, V = 1 -- frozenPower = -share div V must be
    // NONZERO (+10); under A (applied = false) it is exactly 0.
    {
        const double qP = 0.2*(-100.0), S = 1.0, divVs = 0.5, V = 1.0;
        const double powOff = -frozenPressureShare(qP, S, true)*divVs*V;
        const double powOn = -frozenPressureShare(qP, S, false)*divVs*V;
        bool ok = powOff == 10.0 && powOn == 0.0
               && frozenPressureShare(qP, 0.0, true) == 0.0
               && frozenPressureShare(-3.0, 1.5, true) == -2.0;
        char b[96];
        std::snprintf(b, sizeof b, "A off %.17g, A on %.17g", powOff, powOn);
        report("K17 frozenPower: nonzero with A off after a transfer, 0 under A", ok, b);
    }
    // K18 E (H9): passengersShared is tautShare's operation order. (a) For a
    // passenger X in [0, P2] it returns tautShare's result BIT FOR BIT, on
    // K15's random chains (subnormal P included, a third at X = P2); (b) a
    // uniform ratio, signed or not, is kept to 1e-12 (the keep-margin cells
    // included).
    {
        std::mt19937_64 g(77);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        const int N = 300, F = N - 1;
        std::vector<int> own(F), nei(F);
        for (int f = 0; f < F; ++f) { own[f] = f; nei[f] = f + 1; }
        long differ = 0, errs = 0, ran = 0, nearEmpty = 0;
        double worstU = 0;
        for (int trial = 0; trial < 600; ++trial)
        {
            std::vector<double> V(N), Vrel(N), P(N), X(N), g0(F), fF(F);
            double Vmin = 1e300;
            for (int c = 0; c < N; ++c)
            {
                V[c] = 1.0 + u01(g); Vmin = std::min(Vmin, V[c]);
                P[c] = std::pow(10.0, -300.0*u01(g)*(trial % 2)) * (1e-3 + u01(g));
                const double r = u01(g);
                X[c] = r < 0.33 ? P[c] : (r < 0.5 ? 0.0 : P[c]*u01(g));
            }
            for (int c = 0; c < N; ++c) { Vrel[c] = V[c]/Vmin; }
            std::vector<double> Fo(N, 0.0), inc(N, 0.0), PC(N);
            for (int f = 0; f < F; ++f)
            {
                g0[f] = u01(g) < 0.1 ? 0.0 : (u01(g) < 0.5 ? -1.0 : 1.0);
                fF[f] = g0[f] == 0 ? 0.0 : 0.45*u01(g);
            }
            for (int f = 0; f < F; ++f)
            {
                if (g0[f] == 0) continue;
                const int d = g0[f] > 0 ? own[f] : nei[f];
                const int q = g0[f] > 0 ? nei[f] : own[f];
                Fo[d] += fF[f];
                inc[q] += P[d]*(fF[f]*(V[d]/V[q]));
            }
            for (int c = 0; c < N; ++c) { PC[c] = P[c]*(1.0 - Fo[c]) + inc[c]; }
            std::vector<double> mu(F), E(F), PN(PC);
            for (int f = 0; f < F; ++f)
            {
                mu[f] = u01(g);
                const double cap = 0.2
                   *std::min(PC[own[f]]*Vrel[own[f]], PC[nei[f]]*Vrel[nei[f]]);
                E[f] = (2.0*u01(g) - 1.0)*cap;
                if (trial % 3 == 0)
                {
                    // THE KEEP MARGIN (07: where the shipped order raised):
                    // every even cell loses on both its faces, each taking
                    // just under half its content, so it keeps ~1e-9 of it
                    const int loser = (own[f] % 2 == 0) ? own[f] : nei[f];
                    const double la =
                        0.5*(1.0 - 1e-9*(1.0 + u01(g)))*PC[loser]*Vrel[loser];
                    mu[f] = 1.0;
                    E[f] = loser == own[f] ? la : -la;
                }
            }
            {
                std::vector<double> div(N, 0.0), outA(N, 0.0);
                for (int f = 0; f < F; ++f)
                {
                    const double la = mu[f]*(E[f]*1.0);
                    div[own[f]] += la; div[nei[f]] -= la;
                    outA[la > 0 ? own[f] : nei[f]] += std::abs(la);
                }
                bool over = false;
                for (int c = 0; c < N; ++c) { if (PC[c] - outA[c]/Vrel[c] < 0) over = true; }
                if (over) { continue; }
                ++ran;
                for (int c = 0; c < N; ++c)
                {
                    if (PC[c] > 0 && (PC[c] - outA[c]/Vrel[c])/PC[c] < 1e-3) { ++nearEmpty; }
                }
                for (int c = 0; c < N; ++c) { PN[c] = PN[c] - (1.0/Vrel[c])*div[c]; }
            }
            std::vector<double> XT(N), XS(N);
            const char* e1 = damagePartitionKernel::tautShare<double, int>
            (
                N, F, own.data(), nei.data(), V.data(), g0.data(), fF, 1.0,
                Vrel.data(), P.data(), PC.data(), PN.data(), mu, E, X.data(), XT.data()
            );
            const double* xl[1] = {X.data()};
            double* xo[1] = {XS.data()};
            const char* e2 = damagePartitionKernel::passengersShared<double, int>
            (
                N, F, own.data(), nei.data(), V.data(), g0.data(), fF, 1.0,
                Vrel.data(), PC.data(), PN.data(), mu, E, 1, xl, xo
            );
            if (e1 || e2) { ++errs; continue; }
            for (int c = 0; c < N; ++c) { if (XT[c] != XS[c]) { ++differ; } }
            // (b) uniform ratios 0.4 and -0.7 (a signed passenger, q_P <= 0)
            for (const double r : {0.4, -0.7})
            {
                std::vector<double> XU(N), XUN(N);
                for (int c = 0; c < N; ++c) { XU[c] = r*P[c]; }
                const double* ul[1] = {XU.data()};
                double* uo[1] = {XUN.data()};
                if (damagePartitionKernel::passengersShared<double, int>
                    (N, F, own.data(), nei.data(), V.data(), g0.data(), fF, 1.0,
                     Vrel.data(), PC.data(), PN.data(), mu, E, 1, ul, uo))
                {
                    ++errs; continue;
                }
                for (int c = 0; c < N; ++c)
                {
                    if (PN[c] > 1e-280) { worstU = std::max(worstU, std::abs(XUN[c]/PN[c] - r)); }
                }
            }
        }
        char b[200];
        std::snprintf(b, sizeof b, "%ld chains, %ld keep-margin cells (< 1e-3); %ld cells differ from "
                      "tautShare, %ld refusals, uniform-ratio error %.3e",
                      ran, nearEmpty, differ, errs, worstU);
        report("K18 passengersShared is tautShare's order bit for bit",
               differ == 0 && errs == 0 && ran >= 300 && nearEmpty > 0, b);
        report("K18b passengersShared keeps a uniform ratio (to 1e-12)", worstU <= 1e-12 && errs == 0, b);
        std::printf("   K18 %s\n", b);
    }
}


int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    testProjection();
    testRotation();
    testDeposit();
    testIdentity();
    testPassengers();
    testS17b();
    testS18();
    op::tangents();
    testUnitA();
    testUnitC();
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
