// S18 FP-trap test (fix round 25d; docket research/investigations/
// 20261001-1800-run023222-s18-step1-sigfpe/). Batch 60's em76 died of SIGFPE
// on step 1: the dt bound's R2b slope divided by thetaNetRef in lumen cells.
// No offline check had EXECUTED the expression with traps on. This one does.
//
//   c++ -std=c++14 -O2 -ffp-contract=off -I<solver dir> this.cpp -o t
//   ./t            every S18 per-cell expression, the SOLVER'S OWN headers
//                  (networkPressureKernel.H, fibreReleaseKernel.H,
//                  damagePartitionKernel.H), at a lumen cell (thetaNetRef 0,
//                  theta 0), a fringe cell and a bulk cell, with FE_DIVBYZERO,
//                  FE_INVALID and FE_OVERFLOW TRAPPED. Must exit 0.
//   ./t unfixed    the expression as batch 60 shipped it (.C 2025dcda
//                  L17890-17891, transcribed), at the lumen cell. Must
//                  CRASH (a trap): that proves the traps are live here, so
//                  a 0 from the first mode means something.
//
// A trap is caught (SIGFPE, and SIGILL for arm64 trap delivery) and the
// process _exits with 136 (128 + SIGFPE), the code a signal death gives a
// shell. It does not die by the signal, so no OS crash report is filed:
// on 2026-10-04 macOS ReportCrash wedged on queued reports from these
// deliberate crashes and hung a release in uninterruptible exit. The test's
// verdicts are unchanged: "fixed" must still exit 0, "unfixed" must not.
//
// Traps: glibc feenableexcept on Linux (the VM); the FPCR trap bits on
// Darwin arm64 (fenv's __fpcr); MXCSR on Darwin x86-64.

#include "networkPressureKernel.H"
#include "fibreReleaseKernel.H"
#include "damagePartitionKernel.H"

#include <cfenv>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <vector>
#if defined(__APPLE__) && (defined(__x86_64__) || defined(__i386__))
#include <xmmintrin.h>
#endif

static void enableTraps()
{
#if defined(__linux__)
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#elif defined(__APPLE__) && defined(__aarch64__)
    fenv_t env;
    fegetenv(&env);
    env.__fpcr |= __fpcr_trap_divbyzero | __fpcr_trap_invalid | __fpcr_trap_overflow;
    fesetenv(&env);
#elif defined(__APPLE__)
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK()
        & ~(_MM_MASK_DIV_ZERO | _MM_MASK_INVALID | _MM_MASK_OVERFLOW));
#else
#error "no FP-trap mechanism for this platform"
#endif
}

static int failures = 0;
static void report(const char* name, bool ok, double v)
{
    std::printf("%s %s (%.17g)\n", ok ? "PASS" : "FAIL", name, v);
    if (!ok) { ++failures; }
}

// Read through a volatile so the compiler cannot fold the expressions at
// compile time (a folded division never traps at run time).
static double vol(double x) { volatile double v = x; return v; }

struct Cell { const char* name; double theta, ref, Gn; };

static void onTrap(int)
{
    static const char msg[] = "FP TRAP caught: exiting 136\n";
    const ssize_t ignored = write(STDOUT_FILENO, msg, sizeof(msg) - 1);
    (void)ignored;
    _exit(136);
}

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::signal(SIGFPE, onTrap);
    std::signal(SIGILL, onTrap);
    enableTraps();
    const double phi0 = vol(0.45), pm = vol(0.6), resid = vol(0.0002222*0.45);
    const double rho = vol(1060.0), coeff = vol(1.0), G = vol(4.0);

    if (argc > 1 && std::strcmp(argv[1], "unfixed") == 0)
    {
        // batch 60's expression at the lumen cell: refI = 0, theta = 0
        const double thc = vol(0.0), ref = vol(0.0);
        const double a = phi0 + thc - ref;
        const double Kn = coeff*G;
        const double dpda = Kn*(pm - phi0)*((a - phi0)*(a - phi0) + phi0*(pm - phi0))
                          /((a*(pm - a))*(a*(pm - a)));
        const double pa = Kn*(pm - phi0)*(a - phi0)/(a*(pm - a));
        const double wPi = std::min(1.0, ref/resid);
        const double slopeA = wPi*((thc/ref)*dpda + pa/ref);   // 0/0 and 0/0
        std::printf("UNFIXED did not trap: slopeA = %g\n", slopeA);
        return 0;
    }

    const Cell cells[] =
    {
        {"lumen (thetaNetRef 0, theta 0)", 0.0, 0.0, 4.0},
        {"lumen with content (thetaNetRef 0, theta 1e-6)", 1e-6, 0.0, 4.0},
        {"fringe (thetaNetRef 5e-5, theta 2e-5)", 2e-5, 5e-5, 4.0},
        {"bulk (thetaNetRef 0.45, theta 0.46)", 0.46, 0.45, 4.0},
    };
    for (const Cell& c : cells)
    {
        const double thc = vol(c.theta), ref = vol(c.ref);
        const double a = phi0 + thc - ref;
        const double Kn = coeff*vol(c.Gn);
        const double dpda = Kn*(pm - phi0)*((a - phi0)*(a - phi0) + phi0*(pm - phi0))
                          /((a*(pm - a))*(a*(pm - a)));
        // the dt bound, as the solver now computes it
        const double wPi = std::min(1.0, ref/resid);
        const double pa = networkPressureKernel::convexExcessPressure(Kn, pm, phi0, a);
        const double slopeA = wPi*networkPressureKernel::contentRatioSlope(thc, ref, dpda, pa);
        const double w = networkPressureKernel::contentRatio(thc, ref);
        const double ts = networkPressureKernel::totalStretch(ref, thc, vol(1.21));
        const double qS = fibreReleaseKernel::frozenPressureShare(vol(-2.0), thc, true);
        const double qS0 = fibreReleaseKernel::frozenPressureShare(vol(-2.0), thc, false);
        const double D = vol(0.45)*vol(1e-3)*slopeA/rho;
        char n[160];
        std::snprintf(n, sizeof n, "%s: dt-bound slope finite", c.name);
        report(n, std::isfinite(slopeA) && std::isfinite(D), slopeA);
        if (ref == 0)
        {
            std::snprintf(n, sizeof n, "%s: slope and weight exactly 0", c.name);
            report(n, slopeA == 0.0 && w == 0.0, slopeA + w);
        }
        else
        {
            // (the reference formula with the same never-0 denominator: an
            // optimiser may hoist this branch's division to the lumen cell)
            const double rr = ref > 0 ? ref : 1.0;
            const double expect = wPi*((thc/rr)*dpda + pa/rr);
            std::snprintf(n, sizeof n, "%s: slope = the R2b formula", c.name);
            report(n, slopeA == expect, slopeA - expect);
        }
        std::snprintf(n, sizeof n, "%s: total stretch, frozen share finite", c.name);
        report(n, std::isfinite(ts) && std::isfinite(qS) && qS0 == 0.0, ts);
    }

    // E: passengersShared on a chain with EMPTY cells (P_afterC = 0, B = 0),
    // the lumen's state: every division is guarded (spec, A/B) or by V > 0.
    {
        const int N = 6, F = N - 1;
        std::vector<int> own(F), nei(F);
        for (int f = 0; f < F; ++f) { own[f] = f; nei[f] = f + 1; }
        std::vector<double> V(N, 1.0), Vrel(N, 1.0), g(F, 0.0), fF(F, 0.0),
            mu(F, 0.0), E(F, 0.0);
        std::vector<double> PC = {0, 0, 0.3, 0.2, 0, 0}, PN = PC, X = {0, 0, 0.1, -0.05, 0, 0}, XN(N);
        g[1] = 1; fF[1] = 0.0;                   // a C face from an empty cell
        mu[2] = 1; E[2] = 0.1;                   // C+ out of cell 2 into 3
        PN[2] = 0.2; PN[3] = 0.3;
        const double* xl[1] = {X.data()};
        double* xo[1] = {XN.data()};
        const char* err = damagePartitionKernel::passengersShared<double, int>
        (
            N, F, own.data(), nei.data(), V.data(), g.data(), fF, 1.0,
            Vrel.data(), PC.data(), PN.data(), mu, E, 1, xl, xo
        );
        bool fin = err == nullptr;
        for (int c = 0; c < N; ++c) { fin = fin && std::isfinite(XN[c]); }
        report("E: passengersShared with empty cells, no trap, finite", fin && XN[0] == 0 && XN[5] == 0, XN[3]);
    }

    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASS (traps live)", failures);
    return failures ? 1 : 0;
}
