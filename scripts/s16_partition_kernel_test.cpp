// S16 C+ kernel arithmetic test (13c). Compiled and run by
// scripts/check_s16_timed_terminal_rupture.py:
//   c++ -std=c++14 -O2 -ffp-contract=off -I<solver dir> this.cpp
// -ffp-contract=off: the Colab VM is x86-64 without FMA, so no contraction.
//
// It tests the SOLVER'S OWN arithmetic (damagePartitionKernel.H, included
// unchanged). The only other code here is control: a transcription of the
// shipped fctLimiter/fctApply (.C, package 2dd6246c) for theta and for the
// as-built Change 1 negative, and an exact 1-D implicit upwind predictor.
//
// Prints one "PASS <name>" or "FAIL <name>: ..." per test; exit 1 on any FAIL.

#include "damagePartitionKernel.H"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace damagePartitionKernel;
typedef double S;
typedef int L;

static int failures = 0;
static void report(const char* name, bool ok, const char* detail = "")
{
    std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, ok ? "" : ": ", ok ? "" : detail);
    if (!ok) { ++failures; }
}

static const S TINY = std::numeric_limits<S>::min();
static const S KEEP = 1.0 - 1.0e-12;

// ---- control: the shipped fctAntidiffusiveFlux / fctLimiter / fctApply ----
struct Mesh1D
{
    L nC, nF;
    std::vector<L> own, nei;
    std::vector<S> V, F, wLin;
    Mesh1D(L n, S v, S f) : nC(n), nF(n - 1), own(n - 1), nei(n - 1), V(n, v), F(n - 1, f), wLin(n - 1, 0.5)
    {
        for (L i = 0; i < nF; ++i) { own[i] = i; nei[i] = i + 1; }
    }
};

static std::vector<S> antidiff(const Mesh1D& m, const std::vector<S>& x)
{
    std::vector<S> A(m.nF);
    for (L f = 0; f < m.nF; ++f)
    {
        const S wUp = m.F[f] >= 0 ? 1.0 : 0.0;
        A[f] = m.F[f]*(m.wLin[f] - wUp)*(x[m.own[f]] - x[m.nei[f]]);
    }
    return A;
}

// noflux: value a face with |A| < tiny gets (shipped 0; as-built Change 1: 1)
static std::vector<S> shippedLimiter(const Mesh1D& m, const std::vector<S>& fL, const std::vector<S>& fO,
                                     const std::vector<S>& A, S dt, S noflux)
{
    std::vector<S> hi(m.nC), lo(m.nC);
    for (L c = 0; c < m.nC; ++c) { hi[c] = std::max(fL[c], fO[c]); lo[c] = std::min(fL[c], fO[c]); }
    for (L f = 0; f < m.nF; ++f)
    {
        const L o = m.own[f], n = m.nei[f];
        hi[o] = std::max(hi[o], std::max(fL[n], fO[n])); lo[o] = std::min(lo[o], std::min(fL[n], fO[n]));
        hi[n] = std::max(hi[n], std::max(fL[o], fO[o])); lo[n] = std::min(lo[n], std::min(fL[o], fO[o]));
    }
    std::vector<S> Pp(m.nC, 0), Pm(m.nC, 0);
    for (L f = 0; f < m.nF; ++f)
    {
        const S a = A[f];
        if (a > 0) { Pm[m.own[f]] += a; Pp[m.nei[f]] += a; }
        else if (a < 0) { Pp[m.own[f]] -= a; Pm[m.nei[f]] -= a; }
    }
    std::vector<S> Rp(m.nC), Rm(m.nC);
    for (L c = 0; c < m.nC; ++c)
    {
        const S VbyDt = m.V[c]/dt;
        const S Qp = (hi[c] - fL[c])*VbyDt*KEEP;
        const S Qm = (fL[c] - lo[c])*VbyDt*KEEP;
        Rp[c] = Qp < TINY ? 0.0 : Pp[c] > 0 ? std::min(S(1), Qp/Pp[c]) : 1.0;
        Rm[c] = Qm < TINY ? 0.0 : Pm[c] > 0 ? std::min(S(1), Qm/Pm[c]) : 1.0;
    }
    std::vector<S> lam(m.nF);
    for (L f = 0; f < m.nF; ++f)
    {
        const S a = A[f];
        if (std::abs(a) < TINY) lam[f] = noflux;
        else if (a > 0) lam[f] = std::min(Rm[m.own[f]], Rp[m.nei[f]]);
        else lam[f] = std::min(Rp[m.own[f]], Rm[m.nei[f]]);
    }
    return lam;
}

static void shippedApply(const Mesh1D& m, std::vector<S>& x, const std::vector<S>& lam, const std::vector<S>& A, S dt)
{
    std::vector<S> div(m.nC, 0);
    for (L f = 0; f < m.nF; ++f) { const S la = lam[f]*A[f]; div[m.own[f]] += la; div[m.nei[f]] -= la; }
    for (L c = 0; c < m.nC; ++c) x[c] -= dt/m.V[c]*div[c];
}

// exact implicit upwind predictor, conservative form, F >= 0 on every face,
// no boundary flux: x_i (V/dt + F_out) = x_old V/dt + F_in x_{i-1}
static std::vector<S> upwindSolve(const Mesh1D& m, const std::vector<S>& xo, S dt)
{
    std::vector<S> x(m.nC);
    for (L c = 0; c < m.nC; ++c)
    {
        const S Fout = c < m.nC - 1 ? m.F[c] : 0.0;
        const S Fin = c > 0 ? m.F[c - 1] : 0.0;
        const S rhs = xo[c]*m.V[c]/dt + (c > 0 ? Fin*x[c - 1] : 0.0);
        x[c] = rhs/(m.V[c]/dt + Fout);
    }
    return x;
}

static std::vector<S> minv(const std::vector<S>& a, const std::vector<S>& b)
{
    std::vector<S> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = std::min(a[i], b[i]);
    return r;
}


// ---- oracle: 17's C_plus composition, transcribed from
//      probes/numerics-analyst-p17-r3.py L188-208 (with lam_R1 L69-72,
//      apply_R1 L75-77, p4-dtransport limiter L94-133, apply L136-143) in the
//      replica's own operation order, 1-D. Written independently of the
//      kernel, so a kernel that takes s_k at the receiver or limits mu
//      against the predictor (18's K6, K7) cannot agree with it.
static std::vector<S> replicaLimiter(const Mesh1D& m, const std::vector<S>& xL, const std::vector<S>& xO,
                                     const std::vector<S>& Ax, const std::vector<S>& V, S dt)
{
    const L n = m.nC;
    std::vector<S> hi(n), lo(n);
    for (L c = 0; c < n; ++c) { hi[c] = std::max(xL[c], xO[c]); lo[c] = std::min(xL[c], xO[c]); }
    std::vector<S> hi2 = hi, lo2 = lo;
    for (L f = 0; f < m.nF; ++f)      // hi2[:, :-1] = max(hi2[:, :-1], m[:, 1:]) etc.
    {
        hi2[f] = std::max(hi2[f], hi[f + 1]);
        lo2[f] = std::min(lo2[f], lo[f + 1]);
    }
    for (L f = 0; f < m.nF; ++f)
    {
        hi2[f + 1] = std::max(hi2[f + 1], hi[f]);
        lo2[f + 1] = std::min(lo2[f + 1], lo[f]);
    }
    std::vector<S> Pp(n, 0), Pm(n, 0);
    for (L f = 0; f < m.nF; ++f) { Pm[f] += std::max(Ax[f], S(0)); Pp[f + 1] += std::max(Ax[f], S(0)); }
    for (L f = 0; f < m.nF; ++f) { Pp[f] += std::max(-Ax[f], S(0)); Pm[f + 1] += std::max(-Ax[f], S(0)); }
    std::vector<S> Rp(n), Rm(n);
    for (L c = 0; c < n; ++c)
    {
        const S Qp = (hi2[c] - xL[c])*V[c]/dt*KEEP;
        const S Qm = (xL[c] - lo2[c])*V[c]/dt*KEEP;
        Rp[c] = Qp < TINY ? 0.0 : (Pp[c] > 0 ? std::min(S(1), Qp/Pp[c]) : 1.0);
        Rm[c] = Qm < TINY ? 0.0 : (Pm[c] > 0 ? std::min(S(1), Qm/Pm[c]) : 1.0);
    }
    std::vector<S> lx(m.nF);
    for (L f = 0; f < m.nF; ++f)
        lx[f] = std::abs(Ax[f]) < TINY ? 0.0 : (Ax[f] > 0 ? std::min(Rm[f], Rp[f + 1]) : std::min(Rp[f], Rm[f + 1]));
    return lx;
}

static void replicaCPlus(const Mesh1D& m, const std::vector<S> low[3], const std::vector<S> old[3],
                         const std::vector<S> A[3], const std::vector<S>& lamT, const std::vector<S>& AT,
                         const std::vector<S>& Vrel, S s, std::vector<S> base[3])
{
    const L n = m.nC;
    std::vector<S> g(m.nF), sP(n);
    for (L f = 0; f < m.nF; ++f) g[f] = lamT[f]*AT[f];
    for (L c = 0; c < n; ++c) sP[c] = low[0][c] + low[1][c] + low[2][c];
    std::vector<S> mu(m.nF, 1.0);
    std::vector<S> E[3];
    for (int k = 0; k < 3; ++k)
    {
        E[k].resize(m.nF);
        for (L f = 0; f < m.nF; ++f)
        {
            const bool donx = g[f] > 0;
            const S sPd = donx ? sP[f] : sP[f + 1];
            const S sx = (donx ? low[k][f] : low[k][f + 1])/(sPd > 0 ? sPd : 1.0);
            E[k][f] = lamT[f]*(A[k][f] - sx*AT[f]);
        }
        std::vector<S> As(m.nF);
        for (L f = 0; f < m.nF; ++f) As[f] = E[k][f]*s;
        std::vector<S> l = replicaLimiter(m, base[k], old[k], As, Vrel, 1.0);
        for (L f = 0; f < m.nF; ++f) { if (E[k][f] == 0) l[f] = 1.0; mu[f] = std::min(mu[f], l[f]); }
    }
    for (int k = 0; k < 3; ++k)
    {
        std::vector<S> div(n, 0);
        for (L f = 0; f < m.nF; ++f) { const S fx = mu[f]*(E[k][f]*s); div[f] += fx; div[f + 1] -= fx; }
        for (L c = 0; c < n; ++c) base[k][c] = base[k][c] - 1.0/Vrel[c]*div[c];
    }
}

// ------------------------------------------------------------------------
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // nothing lost if a mutation crashes
    // ---- T1. R1 on the verification-engineer's 5-cell case (14 section 5;
    //      probes/verification-engineer-numerics.py L167-262): a far-tail P3
    //      beside exact zeros, subnormal antidiffusive fluxes.
    {
        const S Vc = 2.041e-19, dt = 4.77042903072e-7, F = 2.9e-16;
        Mesh1D m(5, Vc, F);
        std::vector<S> theta = {0.44, 0.445, 0.45, 0.455, 0.46};
        std::vector<S> P3 = {0.0, 0.0, 0.0, 1e-300, 1e-300};
        std::vector<S> P1(5);
        for (int i = 0; i < 5; ++i) P1[i] = theta[i] - P3[i];
        std::vector<S> AT = antidiff(m, theta), A1 = antidiff(m, P1), A3 = antidiff(m, P3);
        bool subn = false;
        for (S a : A3) subn = subn || (a != 0 && std::abs(a) < TINY);
        report("T1a the case has subnormal partition fluxes", subn);

        // control: the as-built Change 1 (|A| < tiny -> 1, V/dt units) in the
        // shared minimum -- must produce a negative (red team 16, R1 negative)
        std::vector<S> lam = shippedLimiter(m, theta, theta, AT, dt, 0.0);
        lam = minv(lam, shippedLimiter(m, P1, P1, A1, dt, 1.0));
        lam = minv(lam, shippedLimiter(m, P3, P3, A3, dt, 1.0));
        std::vector<S> P3b = P3;
        shippedApply(m, P3b, lam, A3, dt);
        bool neg = false;
        for (S v : P3b) neg = neg || v < 0;
        char buf[200];
        std::snprintf(buf, sizeof buf, "P3[2] = %.4e", P3b[2]);
        report("T1b as-built Change 1 goes negative (negative control)", neg, buf);

        // the kernel: limiterR1 + applyR1 in field units, exact-zero test
        S Vmin = Vc;
        std::vector<S> Vrel(5, Vc/Vmin), l3(4);
        const S s = dt/Vmin;
        limiterR1<S, L>(5, 4, m.own.data(), m.nei.data(), P3.data(), P3.data(), A3.data(), s, Vrel.data(), l3.data());
        std::vector<S> P3k = P3;
        applyR1<S, L>(5, 4, m.own.data(), m.nei.data(), l3.data(), A3.data(), s, Vrel.data(), P3k.data());
        bool negk = false;
        for (S v : P3k) negk = negk || v < 0;
        std::snprintf(buf, sizeof buf, "min P3 = %.4e", *std::min_element(P3k.begin(), P3k.end()));
        report("T1c R1 kernel keeps the subnormal case non-negative", !negk, buf);

        // exact zero -> 1; a nonzero subnormal is NOT "no flux"
        std::vector<S> Az = {0.0, 0.0, 0.0, 0.0}, lz(4);
        std::vector<S> Z(5, 0.0);
        limiterR1<S, L>(5, 4, m.own.data(), m.nei.data(), Z.data(), Z.data(), Az.data(), s, Vrel.data(), lz.data());
        report("T1d an exactly-zero flux contributes 1", lz[0] == 1 && lz[1] == 1 && lz[2] == 1 && lz[3] == 1);
        std::vector<S> As = {4.9e-324, 0.0, 0.0, 0.0}, ls(4);
        limiterR1<S, L>(5, 4, m.own.data(), m.nei.data(), Z.data(), Z.data(), As.data(), s, Vrel.data(), ls.data());
        report("T1e a nonzero subnormal flux is limited, not waved through", ls[0] == 0 && ls[1] == 1);
    }

    // ---- T2. C+ against C and the mu = 1 negative control, 1-D, 300 steps.
    //      theta: a clot body with tails; damage: a sharp D spike and a
    //      pending band, so the composition antidiffusion has work to do.
    {
        const L N = 80;
        const S Vc = 2.041e-19, dt = 4.77042903072e-7, F = 1.2e-16;
        Mesh1D m(N, Vc, F);
        // NON-UNIFORM volumes (x1 .. x1.6, as the throat's Vmin 2.04e-19 to
        // Vmax 5.1e-19): the donor share per receiver volume, f V_d/V_r, is
        // only tested where V_d != V_r.
        for (L i = 0; i < N; ++i) m.V[i] = Vc*(1.0 + 0.6*S(i % 7)/6.0);
        std::vector<S> C(N), H(N), P1(N), P2(N), P3(N);
        for (L i = 0; i < N; ++i)
        {
            const S x = S(i);
            // a compacting body (theta ramps 0.30 -> 0.42): theta carries
            // antidiffusive flux there, so lambda_theta > 0 and the
            // composition correction E = lambda_theta (A_P - s A_theta) acts.
            // (Where theta is flat, A_theta = 0 and the shipped limiter gives
            // lambda_theta = 0, so E = 0 whatever mu is.)
            const S th = (i >= 10 && i < 40) ? 0.30 + 0.004*(x - 10.0)
                                             : 0.30*std::exp(-std::abs(x - 25.0)/2.0)*1e-6;
            C[i] = 0.6*th; H[i] = 0.4*th;
            const S dstar = (i >= 18 && i < 24) ? 0.8 : 0.0;
            const S d = (i == 20 || i == 21) ? 0.7 : 0.0;
            P3[i] = th*d; P2[i] = th*(dstar - d > 0 ? dstar - d : 0.0); P1[i] = th - P2[i] - P3[i];
        }
        const S Vmin = *std::min_element(m.V.begin(), m.V.end());
        std::vector<S> Vrel(N);
        for (L i = 0; i < N; ++i) Vrel[i] = m.V[i]/Vmin;
        const S s = dt/Vmin;
        auto content = [&](const std::vector<S>& x) { S t = 0; for (L i = 0; i < N; ++i) t += x[i]*m.V[i]; return t; };

        struct Run { std::vector<S> P[3]; bool neg = false; S minv = 0; S devMax = 0; bool bounded = true; bool err = false; S cons[3] = {0, 0, 0}; };
        auto run = [&](int mode, Run& R, std::vector<std::vector<S>>* trace)
        {
            std::vector<S> c = C, h = H, p[3] = {P1, P2, P3};
            for (int step = 0; step < 300; ++step)
            {
                std::vector<S> cL = upwindSolve(m, c, dt), hL = upwindSolve(m, h, dt);
                std::vector<S> pL[3], pO[3];
                for (int k = 0; k < 3; ++k) { pO[k] = p[k]; pL[k] = upwindSolve(m, p[k], dt); }
                std::vector<S> thL(N), thO(N);
                for (L i = 0; i < N; ++i) { thL[i] = cL[i] + hL[i]; thO[i] = c[i] + h[i]; }
                std::vector<S> AC = antidiff(m, cL), AH = antidiff(m, hL), AT(m.nF);
                for (L f = 0; f < m.nF; ++f) AT[f] = AC[f] + AH[f];
                std::vector<S> lam = shippedLimiter(m, cL, c, AC, dt, 0.0);
                lam = minv(lam, shippedLimiter(m, hL, h, AH, dt, 0.0));
                lam = minv(lam, shippedLimiter(m, thL, thO, AT, dt, 0.0));
                c = cL; h = hL;
                shippedApply(m, c, lam, AC, dt);
                shippedApply(m, h, lam, AH, dt);
                std::vector<S> g(m.nF);
                for (L f = 0; f < m.nF; ++f) g[f] = lam[f]*AT[f];
                std::vector<S> out[3] = {std::vector<S>(N), std::vector<S>(N), std::vector<S>(N)};
                const S* PlowP[3] = {pL[0].data(), pL[1].data(), pL[2].data()};
                S* outP[3] = {out[0].data(), out[1].data(), out[2].data()};
                std::vector<S> fFace;
                Stats<S> st = donorShares<S, L>(N, m.nF, m.own.data(), m.nei.data(), m.V.data(), dt,
                                                thL.data(), g.data(), 3, PlowP, outP, fFace);
                if (st.error) { R.err = true; std::printf("   (mode %d) kernel refused the next step: %s\n", mode, st.error); return; }
                if (mode >= 0)
                {
                    std::vector<S> AP[3];
                    for (int k = 0; k < 3; ++k) AP[k] = antidiff(m, pL[k]);
                    const S* PoldP[3] = {pO[0].data(), pO[1].data(), pO[2].data()};
                    const S* APP[3] = {AP[0].data(), AP[1].data(), AP[2].data()};
                    MuMode mm = mode == 0 ? MuMode::limited : mode == 1 ? MuMode::forceOne : MuMode::forceZero;
                    Stats<S> se = composition<S, L>(N, m.nF, m.own.data(), m.nei.data(), s, Vrel.data(),
                                                    lam.data(), AT.data(), g.data(), 3, PlowP, PoldP, APP, outP, mm);
                    if (se.error)
                    {
                        // a negative produced by forceOne is refused on the
                        // NEXT step's input check: that is the fatal path.
                        R.neg = true; return;
                    }
                }
                for (int k = 0; k < 3; ++k) p[k] = out[k];
                for (L i = 0; i < N; ++i)
                {
                    for (int k = 0; k < 3; ++k) { if (p[k][i] < 0) { R.neg = true; R.minv = std::min(R.minv, p[k][i]); } }
                    const S sP = p[0][i] + (p[1][i] + p[2][i]);
                    const S d = sP > 0 ? p[2][i]/sP : 0.0, ds = sP > 0 ? (p[1][i] + p[2][i])/sP : 0.0;
                    if (!(0 <= d && d <= ds && ds <= 1)) R.bounded = false;
                    R.devMax = std::max(R.devMax, std::abs(sP - (c[i] + h[i])));
                }
                if (trace) { for (int k = 0; k < 3; ++k) trace->push_back(p[k]); }
            }
            for (int k = 0; k < 3; ++k)
            {
                R.P[k] = p[k];
                const S c0 = content(k == 0 ? P1 : k == 1 ? P2 : P3);
                R.cons[k] = std::abs(content(p[k]) - c0)/c0;
            }
        };
        Run lim, one, zero, conly;
        std::vector<std::vector<S>> tz, tc;
        run(0, lim, nullptr);
        run(1, one, nullptr);
        run(2, zero, &tz);
        run(-1, conly, &tc);
        char buf[200];
        std::snprintf(buf, sizeof buf, "negatives %d, bounded %d, max|sumP - theta| %.3e, error %d",
                      int(lim.neg), int(lim.bounded), lim.devMax, int(lim.err));
        report("T2a C+ (mu limited): no negatives, 0 <= d <= d* <= 1, 300 steps", !lim.neg && lim.bounded && !lim.err, buf);
        std::printf("   C+ max|sumP - theta| = %.3e\n", lim.devMax);
        std::snprintf(buf, sizeof buf, "min component %.3e", one.minv);
        report("T2b C+ with mu forced to 1 goes negative (negative control, 17)", one.neg, buf);
        bool same = tz.size() == tc.size() && !tz.empty();
        for (size_t i = 0; same && i < tz.size(); ++i)
            same = std::memcmp(tz[i].data(), tc[i].data(), sizeof(S)*tz[i].size()) == 0;
        report("T2c C+ with mu = 0 equals C bit for bit, every step", same);
        bool differs = false;
        for (int k = 0; k < 3; ++k)
            differs = differs || lim.P[k].size() != size_t(N) || conly.P[k].size() != size_t(N)
                   || std::memcmp(lim.P[k].data(), conly.P[k].data(), sizeof(S)*N) != 0;
        report("T2d C+ (mu limited) differs from C (mu is not inert)", differs);
        // 18's gap (a): conservation of EACH component over the closed 1-D
        // domain (the predictor, the donor shares and E are all
        // conservative). Dropping the donor inflow or the outflow is an
        // error of 7e-4 to 1e-3 (18, K3/K4).
        std::snprintf(buf, sizeof buf, "relative drift P1 %.3e, P2 %.3e, P3 %.3e (C+); P3 %.3e (C)",
                      lim.cons[0], lim.cons[1], lim.cons[2], conly.cons[2]);
        bool cons = !lim.err && !conly.err;
        for (int k = 0; k < 3; ++k) cons = cons && lim.cons[k] <= 1e-12 && conly.cons[k] <= 1e-12;
        report("T2e each component conserved to 1e-12 over 300 steps (C and C+)", cons, buf);
        std::printf("   %s\n", buf);
        std::snprintf(buf, sizeof buf, "max|sumP - theta| %.3e", lim.devMax);
        report("T2f the partition sums to theta to 1e-13 (C+)", lim.devMax <= 1e-13, buf);
    }

    // ---- T3. Non-finite inputs are refused, not propagated.
    {
        Mesh1D m(3, 1.0, 1.0);
        std::vector<S> th = {0.4, 0.4, 0.4}, g = {1e-3, 0.0};
        std::vector<S> a = {0.1, 0.1, 0.1}, b = {0.1, std::numeric_limits<S>::infinity(), 0.1}, c = {0.2, 0.2, 0.2};
        const S* Pl[3] = {a.data(), b.data(), c.data()};
        std::vector<S> o0(3), o1(3), o2(3);
        S* Po[3] = {o0.data(), o1.data(), o2.data()};
        std::vector<S> fF;
        Stats<S> st = donorShares<S, L>(3, 2, m.own.data(), m.nei.data(), m.V.data(), 1.0, th.data(), g.data(), 3, Pl, Po, fF);
        report("T3a +Inf in a partition predictor is refused, BY ITS OWN CHECK",
               st.error && std::strcmp(st.error, "non-finite or negative partition predictor") == 0,
               st.error ? st.error : "(accepted)");
        b[1] = 0.1;
        std::vector<S> gn = {std::nan(""), 0.0};
        st = donorShares<S, L>(3, 2, m.own.data(), m.nei.data(), m.V.data(), 1.0, th.data(), gn.data(), 3, Pl, Po, fF);
        report("T3b NaN in theta's limited flux is refused, BY ITS OWN CHECK",
               st.error && std::strcmp(st.error, "non-finite limited theta flux") == 0,
               st.error ? st.error : "(accepted)");
        std::vector<S> th0 = {0.0, 0.4, 0.4};
        st = donorShares<S, L>(3, 2, m.own.data(), m.nei.data(), m.V.data(), 1.0, th0.data(), g.data(), 3, Pl, Po, fF);
        report("T3c an active face out of an empty donor is refused, BY ITS OWN CHECK",
               st.error && std::strcmp(st.error, "active theta flux out of a donor holding no theta") == 0,
               st.error ? st.error : "(accepted)");
    }

    // ---- T4. R7 (red team 19, probes/red-team-fout-arithmetic.py): a
    //      leading-edge donor where theta's Zalesak bound BINDS (lo = 0, one
    //      outflow face, A > Qm), in the throat: V 2.041e-19 m^3, dt
    //      4.717e-7 s. Exact arithmetic gives F_out = keep = 1 - 1e-12.
    //      The raw-units order |g|*dt/(V*thL) forms subnormal products below
    //      thL ~ 1e-292 and reads F_out >= 1 (a false "bound broken"); the
    //      kernel's field-units order must not.
    {
        const S Vt = 2.041e-19, dtt = 4.717e-7;
        Mesh1D m(2, Vt, 1.0);
        auto bindingG = [&](S thL, S A)
        {
            std::vector<S> fL = {thL, 0.0}, AA = {A};
            return shippedLimiter(m, fL, fL, AA, dtt, 0.0)[0]*A;
        };
        auto oldOrder = [&](S g, S thL) { return (std::abs(g)*dtt)/(Vt*thL); };
        auto kernelFout = [&](S g, S thL, const char** err)
        {
            std::vector<S> th = {thL, 0.0}, gg = {g}, a = {thL, 0.0}, z = {0.0, 0.0};
            const S* Pl[3] = {a.data(), z.data(), z.data()};
            std::vector<S> o0(2), o1(2), o2(2);
            S* Po[3] = {o0.data(), o1.data(), o2.data()};
            std::vector<S> fF;
            Stats<S> st = donorShares<S, L>(2, 1, m.own.data(), m.nei.data(), m.V.data(), dtt,
                                            th.data(), gg.data(), 3, Pl, Po, fF);
            *err = st.error;
            return st.FoutMax;
        };
        // the single donor the dispatch names: thL = 1e-295
        const S thL0 = 1e-295;
        S Afound = 0, oldF = 0;
        for (int k = 0; k < 40000 && Afound == 0; ++k)
        {
            const S A = thL0*std::pow(10.0, -12.3 + 1e-4*k);
            const S g = bindingG(thL0, A);
            if (g != 0 && oldOrder(g, thL0) >= 1) { Afound = A; oldF = oldOrder(g, thL0); }
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, "thL 1e-295: A %.4e, old-order F_out %.17g", Afound, oldF);
        report("T4a the old order reads F_out >= 1 at a binding donor, thL = 1e-295 (control)", Afound != 0, buf);
        std::printf("   %s\n", buf);
        if (Afound != 0)
        {
            const char* err = nullptr;
            const S fo = kernelFout(bindingG(thL0, Afound), thL0, &err);
            std::snprintf(buf, sizeof buf, "kernel F_out %.17g, error %s", fo, err ? err : "none");
            report("T4b the kernel accepts the same donor, F_out < 1", !err && fo < 1, buf);
            std::printf("   %s\n", buf);
        }
        // and 19's sweep, deterministic: 20000 donors per band
        unsigned long long x = 19;
        auto u01 = [&]() { x = x*6364136223846793005ULL + 1442695040888963407ULL; return S((x >> 11) & ((1ULL << 53) - 1))/S(1ULL << 53); };
        const S bands[3][2] = {{-296, -294}, {-294, -292}, {-292, -290}};
        long oldBad = 0, kerBad = 0, used = 0;
        S kerMax = 0;
        for (auto& b : bands)
        {
            for (int j = 0; j < 20000; ++j)
            {
                const S thL = std::pow(10.0, b[0] + (b[1] - b[0])*u01());
                const S A = thL*std::pow(10.0, -14.0 + 4.0*u01());
                const S g = bindingG(thL, A);
                if (g == 0) continue;
                ++used;
                oldBad += oldOrder(g, thL) >= 1;
                const char* err = nullptr;
                const S fo = kernelFout(g, thL, &err);
                kerBad += err != nullptr;
                kerMax = std::max(kerMax, fo);
            }
        }
        std::snprintf(buf, sizeof buf, "%ld binding donors: old order F_out >= 1 at %ld; kernel refusals %ld, max F_out %.17g",
                      used, oldBad, kerBad, kerMax);
        report("T4c 19's sweep: the old order fails, the kernel never refuses", oldBad > 0 && kerBad == 0 && kerMax < 1, buf);
        std::printf("   %s\n", buf);
    }


    // ---- T5. The kernel's composition equals 17's C_plus, transcribed
    //      (oracle above), BIT FOR BIT, over 50 steps of T2's fixture on
    //      non-uniform volumes. Closes 18's K6 (share at the receiver) and
    //      K7 (mu limited against the predictor) numerically.
    {
        const L N = 40;
        const S Vc = 2.041e-19, dt = 4.77042903072e-7, F = 1.2e-16;
        Mesh1D m(N, Vc, F);
        for (L i = 0; i < N; ++i) m.V[i] = Vc*(1.0 + 0.6*S(i % 7)/6.0);
        const S Vmin = *std::min_element(m.V.begin(), m.V.end());
        std::vector<S> Vrel(N);
        for (L i = 0; i < N; ++i) Vrel[i] = m.V[i]/Vmin;
        const S s = dt/Vmin;
        std::vector<S> c(N), h(N), p[3] = {std::vector<S>(N), std::vector<S>(N), std::vector<S>(N)};
        for (L i = 0; i < N; ++i)
        {
            const S th = (i >= 5 && i < 25) ? 0.30 + 0.004*S(i - 5) : 1e-7;
            c[i] = 0.6*th; h[i] = 0.4*th;
            const S dstar = (i >= 10 && i < 16) ? 0.8 : 0.0, d = (i == 12 || i == 13) ? 0.7 : 0.0;
            p[2][i] = th*d; p[1][i] = th*(dstar - d); p[0][i] = th - p[1][i] - p[2][i];
        }
        bool same = true;
        int steps = 0;
        for (int step = 0; step < 50 && same; ++step, ++steps)
        {
            std::vector<S> cL = upwindSolve(m, c, dt), hL = upwindSolve(m, h, dt);
            std::vector<S> pL[3], pO[3], AP[3];
            for (int k = 0; k < 3; ++k) { pO[k] = p[k]; pL[k] = upwindSolve(m, p[k], dt); AP[k] = antidiff(m, pL[k]); }
            std::vector<S> thL(N), thO(N);
            for (L i = 0; i < N; ++i) { thL[i] = cL[i] + hL[i]; thO[i] = c[i] + h[i]; }
            std::vector<S> AC = antidiff(m, cL), AH = antidiff(m, hL), AT(m.nF);
            for (L f = 0; f < m.nF; ++f) AT[f] = AC[f] + AH[f];
            std::vector<S> lam = shippedLimiter(m, cL, c, AC, dt, 0.0);
            lam = minv(lam, shippedLimiter(m, hL, h, AH, dt, 0.0));
            lam = minv(lam, shippedLimiter(m, thL, thO, AT, dt, 0.0));
            c = cL; h = hL;
            shippedApply(m, c, lam, AC, dt);
            shippedApply(m, h, lam, AH, dt);
            std::vector<S> g(m.nF);
            for (L f = 0; f < m.nF; ++f) g[f] = lam[f]*AT[f];
            std::vector<S> out[3] = {std::vector<S>(N), std::vector<S>(N), std::vector<S>(N)};
            const S* PlowP[3] = {pL[0].data(), pL[1].data(), pL[2].data()};
            S* outP[3] = {out[0].data(), out[1].data(), out[2].data()};
            std::vector<S> fFace;
            Stats<S> st = donorShares<S, L>(N, m.nF, m.own.data(), m.nei.data(), m.V.data(), dt,
                                            thL.data(), g.data(), 3, PlowP, outP, fFace);
            if (st.error) { same = false; break; }
            std::vector<S> ref[3] = {out[0], out[1], out[2]};
            const S* PoldP[3] = {pO[0].data(), pO[1].data(), pO[2].data()};
            const S* APP[3] = {AP[0].data(), AP[1].data(), AP[2].data()};
            Stats<S> se = composition<S, L>(N, m.nF, m.own.data(), m.nei.data(), s, Vrel.data(),
                                            lam.data(), AT.data(), g.data(), 3, PlowP, PoldP, APP, outP);
            if (se.error) { same = false; break; }
            replicaCPlus(m, pL, pO, AP, lam, AT, Vrel, s, ref);
            for (int k = 0; k < 3; ++k)
                same = same && std::memcmp(out[k].data(), ref[k].data(), sizeof(S)*N) == 0;
            for (int k = 0; k < 3; ++k) p[k] = out[k];
        }
        char buf[120];
        std::snprintf(buf, sizeof buf, "agreed for %d of 50 steps", same ? 50 : steps);
        report("T5 the kernel's composition equals 17's C_plus (transcribed) bit for bit, 50 steps", same, buf);
    }

    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
