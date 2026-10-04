// S19 kernel test (research/designs/2026-10-03-S19-block-coupled-step.md
// rev. 4, sections 5 and 12). Compiled against the solver's OWN
// blockCoupledKernel.H and SuiteSparse by scripts/check_s19_block_coupled.py,
// which also mutates a temporary copy of the header and requires each
// mutation to FAIL.
//
// The reference is an INDEPENDENT dense assembly written here with operator
// products (FI, Dv, G), in the block layout [Ux | Uy | Vsx | Vsy | p], solved
// by Gaussian elimination -- the structure of the replay probe
// probes/numerics-analyst-06-g1.py, not the kernel's face loops.
//
//   K1  known solution: x* random, b* = A_ref x*; the kernel's CSR + UMFPACK
//       return x* (rel. 1e-10) with ||A x - b*||/||b*|| <= 1e-12
//   K2  the kernel's right-hand side equals the reference's (rel. 1e-12)
//   K3  k = 2 back-substitutions equal the reference chain x2 (rel. 1e-9),
//       the final cell sums of F_f vanish (<= 1e-12 of the flux scale), and
//       the lagged chain x1 differs from x2 by more than 1e-6
//   K4  the symbolic factorisation is redone on a pattern change (count +1,
//       solution correct) and kept on a value-only change (count +0)
//   K5  R3: rigid translation and rigid rotation on a theta ramp give zero
//       force (<= 1e-12 relative), on a skewed mesh
//   K6  R3: -sum Vs.R3(Vs) >= 0 on 100 random fields over a random
//       theta(1 - d) map
//   K7  R3: eta_n spot values 2.22e-3, 2.22e-2, 0.222 (G 4, 40, 400)
//   K8  R3: every boundary weight is 0 after r3ZeroBoundaryWeights, and the
//       step runs with R3 on
//   K10 R3: the diagonal added to A_s for the face inverse (CE F16) equals
//       the probed dense reference (rel. 1e-12)
//   K9  R3: the assembled block applies exactly -R3 force to the skeleton
//       rows; on a uniform orthogonal grid v = (x y, 0) gives the exact
//       div(nu (grad v + grad v^T)) = nu (0, 1) in interior cells
//   K11 R3 (fix round 12): the applied face coefficient r3FaceCoefficient
//       equals tau_v/rho times the elastic face coefficient series(w)_f
//       interp(gD G)_f built here from cell values (rel. 1e-12, Dg > 0
//       included), and on uniform cells it is theta(1 - d) eta_n gD
//       (G 4, 40, 400; Dg 0 and 0.5)
#include "blockCoupledKernel.H"

#include <cstdio>
#include <cstdlib>
#include <random>

using namespace blockCoupled;

static int failures = 0;
static void check(const bool ok, const char* tag, const char* what, const double v)
{
    std::printf("%s %s %s (%.3e)\n", ok ? "PASS" : "FAIL", tag, what, v);
    if (!ok) ++failures;
}

// ---------------------------------------------------------------------------
// A skewed, stretched nx x ny mesh: vertices (x_i + s y_j, y_j), y_j graded.
// Left patch open (inlet), right open (outlet), bottom/top walls.
// ---------------------------------------------------------------------------
struct Geo
{
    Mesh m;
    int nx, ny;
    std::vector<int> bPatch;  // 0 inlet 1 outlet 2 wall
};

static Geo makeMesh(const int nx, const int ny, const double s, const double grade)
{
    Geo g;
    g.nx = nx; g.ny = ny;
    Mesh& m = g.m;
    const double Lx = 1.0, Ly = 0.6;
    std::vector<double> xs(nx + 1), ys(ny + 1);
    for (int i = 0; i <= nx; ++i) xs[i] = Lx*i/nx;
    for (int j = 0; j <= ny; ++j) ys[j] = Ly*std::pow(double(j)/ny, grade);
    auto vx = [&](int i, int j) { return xs[i] + s*ys[j]; };
    auto vy = [&](int, int j) { return ys[j]; };
    const int N = nx*ny;
    m.nCells = N;
    m.V.resize(N); m.Cx.resize(N); m.Cy.resize(N);
    auto cid = [&](int i, int j) { return j*nx + i; };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
            const int c = cid(i, j);
            const double dx = xs[i + 1] - xs[i], dy = ys[j + 1] - ys[j];
            m.V[c] = dx*dy;
            m.Cx[c] = 0.25*(vx(i, j) + vx(i + 1, j) + vx(i, j + 1) + vx(i + 1, j + 1));
            m.Cy[c] = 0.5*(ys[j] + ys[j + 1]);
        }
    auto addFace = [&](int o, int n, double ax, double ay, double bx, double by)
    {
        // face from vertex a to b; S = (by - ay, -(bx - ax)) points to the
        // right of a->b; orient it from o to n.
        double Sx = by - ay, Sy = -(bx - ax);
        const double dxo = m.Cx[n] - m.Cx[o], dyo = m.Cy[n] - m.Cy[o];
        if (Sx*dxo + Sy*dyo < 0) { Sx = -Sx; Sy = -Sy; }
        const double fx = 0.5*(ax + bx), fy = 0.5*(ay + by);
        const double magS = std::sqrt(Sx*Sx + Sy*Sy);
        const double so = std::fabs(Sx*(fx - m.Cx[o]) + Sy*(fy - m.Cy[o]));
        const double sn = std::fabs(Sx*(m.Cx[n] - fx) + Sy*(m.Cy[n] - fy));
        const double nxv = Sx/magS, nyv = Sy/magS;
        const double ndot = nxv*dxo + nyv*dyo, dmag = std::sqrt(dxo*dxo + dyo*dyo);
        const double ndc = 1.0/std::max(ndot, 0.05*dmag);
        m.own.push_back(o); m.nei.push_back(n);
        m.Sx.push_back(Sx); m.Sy.push_back(Sy); m.magS.push_back(magS);
        m.w.push_back(sn/(so + sn)); m.ndc.push_back(ndc);
        m.kx.push_back(nxv - dxo*ndc); m.ky.push_back(nyv - dyo*ndc);
    };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
            if (i + 1 < nx) addFace(cid(i, j), cid(i + 1, j), vx(i + 1, j), vy(i + 1, j), vx(i + 1, j + 1), vy(i + 1, j + 1));
            if (j + 1 < ny) addFace(cid(i, j), cid(i, j + 1), vx(i, j + 1), vy(i, j + 1), vx(i + 1, j + 1), vy(i + 1, j + 1));
        }
    auto addB = [&](int c, double Sx, double Sy, int patch)
    {
        m.bCell.push_back(c); m.bSx.push_back(Sx); m.bSy.push_back(Sy);
        m.bMagS.push_back(std::sqrt(Sx*Sx + Sy*Sy)); m.bOpen.push_back(patch < 2 ? 1 : 0);
        g.bPatch.push_back(patch);
    };
    for (int j = 0; j < ny; ++j)
    {
        const double dy = ys[j + 1] - ys[j];
        addB(cid(0, j), -dy, s*dy, 0);            // inlet, outward -x
        addB(cid(nx - 1, j), dy, -s*dy, 1);       // outlet, outward +x
    }
    for (int i = 0; i < nx; ++i)
    {
        const double dx = xs[i + 1] - xs[i];
        addB(cid(i, 0), 0.0, -dx, 2);
        addB(cid(i, ny - 1), 0.0, dx, 2);
    }
    return g;
}

static Coeffs makeCoeffs(const Geo& g, std::mt19937& rng, const double lumenShare)
{
    const Mesh& m = g.m;
    std::uniform_real_distribution<double> U01(0.0, 1.0), Upm(-1.0, 1.0);
    const int N = m.nCells; const std::size_t nF = m.own.size(), nB = m.bCell.size();
    Coeffs c;
    for (int k = 0; k < 2; ++k)
    {
        c.dU[k].resize(N); c.sU[k].resize(N); c.dVs[k].resize(N); c.sVs[k].resize(N);
    }
    c.upU.resize(nF); c.loU.resize(nF); c.upVs.resize(nF); c.loVs.resize(nF);
    std::vector<double> offU(N, 0.0), offV(N, 0.0);
    for (std::size_t f = 0; f < nF; ++f)
    {
        c.upU[f] = -0.5*(1 + U01(rng)); c.loU[f] = -0.5*(1 + U01(rng));
        c.upVs[f] = -0.3*(1 + U01(rng)); c.loVs[f] = -0.3*(1 + U01(rng));
        offU[m.own[f]] += std::fabs(c.upU[f]); offU[m.nei[f]] += std::fabs(c.loU[f]);
        offV[m.own[f]] += std::fabs(c.upVs[f]); offV[m.nei[f]] += std::fabs(c.loVs[f]);
    }
    c.K.resize(N); c.Kres.resize(N); c.phiF.resize(N); c.phiS.resize(N);
    c.FelX.resize(N); c.FelY.resize(N); c.isSkel.resize(N);
    for (int P = 0; P < N; ++P)
    {
        for (int k = 0; k < 2; ++k)
        {
            c.dU[k][P] = offU[P] + m.V[P]*(5 + U01(rng));
            c.dVs[k][P] = offV[P] + m.V[P]*(3 + U01(rng));
            c.sU[k][P] = m.V[P]*Upm(rng);
            c.sVs[k][P] = m.V[P]*Upm(rng);
        }
        c.K[P] = U01(rng) < 0.2 ? 0.0 : 40*U01(rng);
        c.Kres[P] = 3*U01(rng);
        c.phiS[P] = 0.45*U01(rng);
        c.phiF[P] = 1 - c.phiS[P];
        c.FelX[P] = Upm(rng); c.FelY[P] = Upm(rng);
        c.isSkel[P] = U01(rng) < lumenShare ? 0 : 1;
    }
    c.phiFf.resize(nF); c.phiSf.resize(nF); c.rAf.resize(nF); c.cEf.resize(nF); c.fSigE.resize(nF);
    for (std::size_t f = 0; f < nF; ++f)
    {
        const int o = m.own[f], n = m.nei[f];
        c.phiFf[f] = m.w[f]*c.phiF[o] + (1 - m.w[f])*c.phiF[n];
        c.phiSf[f] = 1 - c.phiFf[f];
        c.rAf[f] = 0.05*(1 + U01(rng));
        c.cEf[f] = 0.02*Upm(rng);
        c.fSigE[f] = m.magS[f]*Upm(rng);
    }
    c.bPhiF.resize(nB); c.bPhiS.resize(nB); c.bRAf.resize(nB); c.bCE.resize(nB);
    c.bFSigE.resize(nB); c.bWallFlux.assign(nB, 0.0);
    c.pvIC.resize(nB); c.pvBC.resize(nB); c.pgIC.resize(nB); c.pgBC.resize(nB);
    for (std::size_t b = 0; b < nB; ++b)
    {
        const int P = m.bCell[b];
        c.bPhiF[b] = c.phiF[P]; c.bPhiS[b] = c.phiS[P];
        c.bRAf[b] = 0.05*(1 + U01(rng)); c.bCE[b] = 0.02*Upm(rng);
        c.bFSigE[b] = m.bMagS[b]*Upm(rng);
        if (g.bPatch[b] == 2)
        {
            c.pvIC[b] = 1; c.pvBC[b] = 0; c.pgIC[b] = 0; c.pgBC[b] = 0;
            c.bWallFlux[b] = 0.0;
        }
        else
        {
            const double pb = g.bPatch[b] == 0 ? 0.0737066 : 0.0;
            // boundary deltaCoeffs: 1/(n.(Cf - C_P)), here from the cell size
            const double dcb = 2.0*m.bMagS[b]/m.V[P];
            c.pvIC[b] = 0; c.pvBC[b] = pb; c.pgIC[b] = -dcb; c.pgBC[b] = dcb*pb;
        }
    }
    c.r3 = false;
    c.nu3w.assign(nF, 0.0);
    c.bNu3w.assign(nB, 0.0);
    return c;
}

// ---------------------------------------------------------------------------
// The independent dense reference, block layout [Ux|Uy|Vsx|Vsy|p].
// ---------------------------------------------------------------------------
typedef std::vector<double> Vec;
struct Dense { int r, c; Vec a; Dense(int r_, int c_) : r(r_), c(c_), a(std::size_t(r_)*c_, 0.0) {}
    double& operator()(int i, int j) { return a[std::size_t(i)*c + j]; }
    double operator()(int i, int j) const { return a[std::size_t(i)*c + j]; } };

static Dense mul(const Dense& A, const Dense& B)
{
    Dense C(A.r, B.c);
    for (int i = 0; i < A.r; ++i)
        for (int k = 0; k < A.c; ++k)
        {
            const double v = A(i, k);
            if (v == 0) continue;
            for (int j = 0; j < B.c; ++j) C(i, j) += v*B(k, j);
        }
    return C;
}
static Dense diag(const Vec& d) { Dense D(int(d.size()), int(d.size())); for (std::size_t i = 0; i < d.size(); ++i) D(int(i), int(i)) = d[i]; return D; }
static Vec mv(const Dense& A, const Vec& x) { Vec y(A.r, 0.0); for (int i = 0; i < A.r; ++i) { double s = 0; for (int j = 0; j < A.c; ++j) s += A(i, j)*x[j]; y[i] = s; } return y; }

static Vec gaussSolve(Dense A, Vec b)
{
    const int n = A.r;
    for (int k = 0; k < n; ++k)
    {
        int piv = k; double best = std::fabs(A(k, k));
        for (int i = k + 1; i < n; ++i) if (std::fabs(A(i, k)) > best) { best = std::fabs(A(i, k)); piv = i; }
        if (piv != k) { for (int j = 0; j < n; ++j) std::swap(A(k, j), A(piv, j)); std::swap(b[k], b[piv]); }
        for (int i = k + 1; i < n; ++i)
        {
            const double f = A(i, k)/A(k, k);
            if (f == 0) continue;
            for (int j = k; j < n; ++j) A(i, j) -= f*A(k, j);
            b[i] -= f*b[k];
        }
    }
    Vec x(n);
    for (int i = n - 1; i >= 0; --i) { double s = b[i]; for (int j = i + 1; j < n; ++j) s -= A(i, j)*x[j]; x[i] = s/A(i, i); }
    return x;
}

struct Ref
{
    Dense A; Vec b0; Dense Ek; Vec ekc;   // b = b0 - (Ek p + ekc) in p rows
    Ref(int n, int N) : A(n, n), b0(n, 0.0), Ek(N, N) {}
};

static Ref reference(const Mesh& m, const Coeffs& c)
{
    const int N = m.nCells, nF = int(m.own.size()), nB = int(m.bCell.size());
    Ref R(5*N, N);
    Dense FI(nF, N), Dv(N, nF), Ed(nF, N);
    for (int f = 0; f < nF; ++f)
    {
        FI(f, m.own[f]) = m.w[f]; FI(f, m.nei[f]) = 1 - m.w[f];
        Dv(m.own[f], f) = 1; Dv(m.nei[f], f) = -1;
        Ed(f, m.own[f]) = -1; Ed(f, m.nei[f]) = 1;
    }
    Vec iV(N); for (int P = 0; P < N; ++P) iV[P] = 1.0/m.V[P];
    Dense Gk[2] = {Dense(N, N), Dense(N, N)}; Vec gc[2] = {Vec(N, 0.0), Vec(N, 0.0)};
    for (int k = 0; k < 2; ++k)
    {
        Vec Sk(nF); for (int f = 0; f < nF; ++f) Sk[f] = k == 0 ? m.Sx[f] : m.Sy[f];
        Gk[k] = mul(Dv, mul(diag(Sk), FI));
        for (int b = 0; b < nB; ++b)
        {
            const int P = m.bCell[b]; const double bs = k == 0 ? m.bSx[b] : m.bSy[b];
            Gk[k](P, P) += bs*c.pvIC[b]; gc[k][P] += bs*c.pvBC[b];
        }
        Gk[k] = mul(diag(iV), Gk[k]);
        for (int P = 0; P < N; ++P) gc[k][P] *= iV[P];
    }
    auto put = [&](int bi, int bj, const Dense& D) { for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) R.A(bi*N + i, bj*N + j) += D(i, j); };
    Dense OffU(N, N), OffV(N, N);
    for (int f = 0; f < nF; ++f)
    {
        OffU(m.own[f], m.nei[f]) += c.upU[f]; OffU(m.nei[f], m.own[f]) += c.loU[f];
        OffV(m.own[f], m.nei[f]) += c.upVs[f]; OffV(m.nei[f], m.own[f]) += c.loVs[f];
    }
    Vec KV(N), KsV(N), sk(N), lu(N), pFV(N), pSV(N);
    for (int P = 0; P < N; ++P)
    {
        KV[P] = c.K[P]*m.V[P]; KsV[P] = (c.K[P] + c.Kres[P])*m.V[P];
        sk[P] = c.isSkel[P] ? 1 : 0; lu[P] = 1 - sk[P];
        pFV[P] = c.phiF[P]*m.V[P]; pSV[P] = c.phiS[P]*m.V[P];
    }
    for (int k = 0; k < 2; ++k)
    {
        Vec dU(N), dV(N), mKV(N), mKsV(N); for (int P = 0; P < N; ++P) { dU[P] = c.dU[k][P] + KV[P]; dV[P] = c.dVs[k][P] + KsV[P]; mKV[P] = -KV[P]; mKsV[P] = -KsV[P]; }
        put(k, k, diag(dU)); put(k, k, OffU); put(k, 2 + k, diag(mKV)); put(k, 4, mul(diag(pFV), Gk[k]));
        put(2 + k, 2 + k, mul(diag(sk), mul(Dense(diag(dV)), Dense(diag(Vec(N, 1.0)))))); put(2 + k, 2 + k, mul(diag(sk), OffV)); put(2 + k, 2 + k, diag(lu));
        Vec mlu(N); for (int P = 0; P < N; ++P) mlu[P] = -lu[P];
        put(2 + k, k, mul(diag(sk), diag(mKsV))); put(2 + k, k, diag(mlu));
        put(2 + k, 4, mul(diag(sk), mul(diag(pSV), Gk[k])));
        for (int P = 0; P < N; ++P)
        {
            R.b0[k*N + P] = c.sU[k][P] - pFV[P]*gc[k][P];
            R.b0[(2 + k)*N + P] = sk[P]*(c.sVs[k][P] + m.V[P]*(k == 0 ? c.FelX[P] : c.FelY[P]) - pSV[P]*gc[k][P]);
        }
    }
    // p rows: Dv F + boundary.
    for (int k = 0; k < 2; ++k)
    {
        Vec aF(nF), aS(nF); for (int f = 0; f < nF; ++f) { const double Sk = k == 0 ? m.Sx[f] : m.Sy[f]; aF[f] = c.phiFf[f]*Sk; aS[f] = c.phiSf[f]*Sk; }
        put(4, k, mul(Dv, mul(diag(aF), FI))); put(4, 2 + k, mul(Dv, mul(diag(aS), FI)));
    }
    Vec cp(nF), rS[2] = {Vec(nF), Vec(nF)};
    for (int f = 0; f < nF; ++f) { cp[f] = -c.rAf[f]*m.magS[f]*m.ndc[f]; rS[0][f] = c.rAf[f]*m.Sx[f]; rS[1][f] = c.rAf[f]*m.Sy[f]; }
    put(4, 4, mul(Dv, mul(diag(cp), Ed)));
    for (int k = 0; k < 2; ++k) put(4, 4, mul(Dv, mul(diag(rS[k]), mul(FI, Gk[k]))));
    Vec cF(nF, 0.0);
    for (int f = 0; f < nF; ++f)
    {
        const int o = m.own[f], n = m.nei[f]; const double w = m.w[f];
        cF[f] = c.rAf[f]*(m.Sx[f]*(w*gc[0][o] + (1 - w)*gc[0][n]) + m.Sy[f]*(w*gc[1][o] + (1 - w)*gc[1][n]))
              + c.cEf[f]*(c.fSigE[f] - m.Sx[f]*(w*c.FelX[o] + (1 - w)*c.FelX[n]) - m.Sy[f]*(w*c.FelY[o] + (1 - w)*c.FelY[n]));
    }
    Vec pc = mv(Dv, cF);
    for (int b = 0; b < nB; ++b)
    {
        const int P = m.bCell[b];
        if (!m.bOpen[b]) { pc[P] += c.bWallFlux[b]; continue; }
        R.A(4*N + P, 0*N + P) += c.bPhiF[b]*m.bSx[b]; R.A(4*N + P, 1*N + P) += c.bPhiF[b]*m.bSy[b];
        R.A(4*N + P, 2*N + P) += c.bPhiS[b]*m.bSx[b]; R.A(4*N + P, 3*N + P) += c.bPhiS[b]*m.bSy[b];
        R.A(4*N + P, 4*N + P) += -c.bRAf[b]*m.bMagS[b]*c.pgIC[b];
        for (int j = 0; j < N; ++j) R.A(4*N + P, 4*N + j) += c.bRAf[b]*(m.bSx[b]*Gk[0](P, j) + m.bSy[b]*Gk[1](P, j));
        pc[P] += -c.bRAf[b]*m.bMagS[b]*c.pgBC[b] + c.bRAf[b]*(m.bSx[b]*gc[0][P] + m.bSy[b]*gc[1][P])
               + c.bCE[b]*(c.bFSigE[b] - m.bSx[b]*c.FelX[P] - m.bSy[b]*c.FelY[P]);
    }
    for (int P = 0; P < N; ++P) R.b0[4*N + P] = -pc[P];
    // explicit non-orthogonal part: Ek p + ekc, per cell (p rows)
    Vec kk[2] = {Vec(nF), Vec(nF)};
    for (int f = 0; f < nF; ++f) { kk[0][f] = -c.rAf[f]*m.magS[f]*m.kx[f]; kk[1][f] = -c.rAf[f]*m.magS[f]*m.ky[f]; }
    R.Ek = Dense(N, N); R.ekc = Vec(N, 0.0);
    for (int k = 0; k < 2; ++k)
    {
        Dense T = mul(Dv, mul(diag(kk[k]), mul(FI, Gk[k])));
        for (std::size_t q = 0; q < T.a.size(); ++q) R.Ek.a[q] += T.a[q];
        Vec cf(nF); for (int f = 0; f < nF; ++f) cf[f] = kk[k][f]*(m.w[f]*gc[k][m.own[f]] + (1 - m.w[f])*gc[k][m.nei[f]]);
        Vec t = mv(Dv, cf); for (int P = 0; P < N; ++P) R.ekc[P] += t[P];
    }
    return R;
}

static Vec toBlock(const Vec& xi, int N) { Vec y(5*N); for (int P = 0; P < N; ++P) for (int k = 0; k < 5; ++k) y[k*N + P] = xi[idx(P, k)]; return y; }
static Vec toInter(const Vec& yb, int N) { Vec x(5*N); for (int P = 0; P < N; ++P) for (int k = 0; k < 5; ++k) x[idx(P, k)] = yb[k*N + P]; return x; }
static double relmax(const Vec& a, const Vec& b) { double d = 0, s = 0; for (std::size_t i = 0; i < a.size(); ++i) { d = std::max(d, std::fabs(a[i] - b[i])); s = std::max(s, std::fabs(b[i])); } return s > 0 ? d/s : d; }

static Vec refStep(const Ref& R, const Vec& xb0, int N, int k)
{
    Vec p(xb0.begin() + 4*N, xb0.end()), x;
    const int ns = k > 0 ? k : 1;
    for (int it = 0; it < ns; ++it)
    {
        Vec b = R.b0;
        if (k > 0) { Vec e = mv(R.Ek, p); for (int P = 0; P < N; ++P) b[4*N + P] -= e[P] + R.ekc[P]; }
        x = gaussSolve(R.A, b);
        p.assign(x.begin() + 4*N, x.end());
    }
    return x;
}

int main()
{
    setvbuf(stdout, 0, _IONBF, 0);
    std::mt19937 rng(19);
    std::uniform_real_distribution<double> Upm(-1.0, 1.0), U01(0.0, 1.0);

    // ---------------- K1-K3 on the skewed mesh ----------------
    Geo g = makeMesh(8, 6, 0.3, 1.3);
    const Mesh& m = g.m; const int N = m.nCells;
    Coeffs c = makeCoeffs(g, rng, 0.3);
    Ref R = reference(m, c);
    std::vector<GradRow> G = gaussGradRows(m, c); std::vector<LsqRow> L;
    Csr A; Vec b0;
    assemble(m, c, G, L, A, b0);
    {
        Vec xs(5*N); for (auto& v : xs) v = Upm(rng);
        Vec bs = mv(R.A, xs);
        UmfpackLU lu; double ts, tn; int stage;
        const int st = lu.factor(A, ts, tn, stage);
        Vec bi = toInter(bs, N), x;
        const int st2 = lu.backSolve(A, bi, x);
        double rn, bn; residualNorms(A, x, bi, rn, bn);
        check(st == UMFPACK_OK && st2 == UMFPACK_OK, "K1", "UMFPACK status OK", double(st));
        check(relmax(toBlock(x, N), xs) <= 1e-10, "K1", "known solution recovered", relmax(toBlock(x, N), xs));
        check(rn/bn <= 1e-12, "K1", "||Ax-b||/||b|| <= 1e-12", rn/bn);
    }
    check(relmax(toBlock(b0, N), R.b0) <= 1e-12, "K2", "right-hand side equals the reference", relmax(toBlock(b0, N), R.b0));
    {
        Vec x0(5*N); for (auto& v : x0) v = 0.05*Upm(rng);
        UmfpackLU lu; Step s;
        solveStep(m, c, 2, x0, lu, s);
        const Vec x2 = refStep(R, toBlock(x0, N), N, 2), x1 = refStep(R, toBlock(x0, N), N, 1);
        double fs = 0, cs = 0; for (std::size_t f = 0; f < s.F.size(); ++f) fs = std::max(fs, std::fabs(s.F[f]));
        for (double v : s.cellSum) cs = std::max(cs, std::fabs(v));
        check(s.status == UMFPACK_OK, "K3", "step status OK", double(s.status));
        check(relmax(toBlock(s.x, N), x2) <= 1e-9, "K3", "k=2 back-substitutions equal the reference chain", relmax(toBlock(s.x, N), x2));
        check(cs <= 1e-12*fs, "K3", "final cell sums of F_f vanish", fs > 0 ? cs/fs : cs);
        check(s.rNorm <= 1e-12*s.bNorm, "K3", "linRes of the final system", s.bNorm > 0 ? s.rNorm/s.bNorm : s.rNorm);
        check(relmax(x1, x2) > 1e-6, "K3", "the lagged chain (k=1) differs from k=2", relmax(x1, x2));
    }

    // ---------------- K4: symbolic keyed to the pattern ----------------
    {
        UmfpackLU lu; Step s; Vec x0(5*N, 0.0);
        solveStep(m, c, 0, x0, lu, s);
        const long n1 = lu.symbolicCount();
        Coeffs c2 = c; int flips = 0;
        for (int P = 0; P < N && flips < 4; ++P) if (P % 7 == 3) { c2.isSkel[P] = !c2.isSkel[P]; ++flips; }
        Ref R2 = reference(m, c2);
        solveStep(m, c2, 0, x0, lu, s);
        const long n2 = lu.symbolicCount();
        const Vec xr = refStep(R2, toBlock(x0, N), N, 0);
        check(s.status == UMFPACK_OK, "K4", "status OK after the pattern change", double(s.status));
        check(n2 == n1 + 1, "K4", "symbolic redone on a pattern change", double(n2 - n1));
        check(s.status == UMFPACK_OK && relmax(toBlock(s.x, N), xr) <= 1e-9, "K4", "solution correct after the pattern change", s.x.empty() ? 1.0 : relmax(toBlock(s.x, N), xr));
        Coeffs c3 = c2; for (int P = 0; P < N; ++P) c3.K[P] *= 1.7;
        Ref R3 = reference(m, c3);
        solveStep(m, c3, 0, x0, lu, s);
        const Vec xr3 = refStep(R3, toBlock(x0, N), N, 0);
        check(lu.symbolicCount() == n2, "K4", "symbolic kept on a value-only change", double(lu.symbolicCount() - n2));
        check(s.status == UMFPACK_OK && relmax(toBlock(s.x, N), xr3) <= 1e-9, "K4", "solution correct on the kept symbolic", s.x.empty() ? 1.0 : relmax(toBlock(s.x, N), xr3));
    }

    // ---------------- K5-K9: R3 ----------------
    std::vector<LsqRow> Lq;
    check(lsqRows(m, Lq), "K5", "least-squares rows built", 0.0);
    const double nu3 = 2.2222e-3/1025.0;
    auto seriesW = [&](const Vec& wc, Vec& wf) { wf.assign(m.own.size(), 0.0); for (std::size_t f = 0; f < m.own.size(); ++f) { const double a = wc[m.own[f]], b = wc[m.nei[f]]; wf[f] = (a > 0 && b > 0) ? 2*a*b/(a + b) : 0.0; } };
    Coeffs cr = c; cr.r3 = true;
    {
        Vec wc(N); for (int P = 0; P < N; ++P) wc[P] = 0.2 + 0.6*m.Cx[P];
        Vec wf; seriesW(wc, wf); for (std::size_t f = 0; f < wf.size(); ++f) cr.nu3w[f] = nu3*wf[f];
        double scale = 0; for (std::size_t f = 0; f < wf.size(); ++f) scale = std::max(scale, cr.nu3w[f]*m.magS[f]);
        Vec vx(N), vy(N), fx, fy; double mx = 0;
        for (int P = 0; P < N; ++P) { vx[P] = 1.3; vy[P] = -0.4; }
        r3Force(m, cr, Lq, vx, vy, fx, fy); mx = 0; for (int P = 0; P < N; ++P) mx = std::max(mx, std::hypot(fx[P], fy[P]));
        check(mx <= 1e-12*scale*1.3, "K5", "rigid translation gives zero R3 force", mx/(scale*1.3));
        const double om = 2.7;
        for (int P = 0; P < N; ++P) { vx[P] = -om*(m.Cy[P] - 0.3); vy[P] = om*(m.Cx[P] - 0.5); }
        r3Force(m, cr, Lq, vx, vy, fx, fy); mx = 0; for (int P = 0; P < N; ++P) mx = std::max(mx, std::hypot(fx[P], fy[P]));
        check(mx <= 1e-12*scale*om, "K5", "rigid rotation gives zero R3 force", mx/(scale*om));
    }
    {
        int bad = 0; double worst = -1e300;
        for (int t = 0; t < 100; ++t)
        {
            Vec wc(N); for (int P = 0; P < N; ++P) wc[P] = U01(rng) < 0.15 ? 0.0 : U01(rng);
            Vec wf; seriesW(wc, wf); Coeffs cd = cr; for (std::size_t f = 0; f < wf.size(); ++f) cd.nu3w[f] = nu3*wf[f];
            Vec vx(N), vy(N), fx, fy; for (int P = 0; P < N; ++P) { vx[P] = Upm(rng); vy[P] = Upm(rng); }
            r3Force(m, cd, Lq, vx, vy, fx, fy);
            double pw = 0, sc = 0; for (int P = 0; P < N; ++P) { pw += vx[P]*fx[P] + vy[P]*fy[P]; sc += std::fabs(vx[P]*fx[P]) + std::fabs(vy[P]*fy[P]); }
            if (pw > 1e-14*sc) ++bad;
            worst = std::max(worst, sc > 0 ? pw/sc : 0.0);
        }
        check(bad == 0, "K6", "-sum Vs.R3(Vs) >= 0 on 100 random fields (worst power/scale)", worst);
    }
    {
        const double e4 = etaN(4, 2.5e-4, 0.45), e40 = etaN(40, 2.5e-4, 0.45), e400 = etaN(400, 2.5e-4, 0.45);
        // the design quotes three significant figures: half a unit in the last place
        check(std::fabs(e4 - 2.22e-3) <= 0.5e-5 && std::fabs(e40 - 2.22e-2) <= 0.5e-4 && std::fabs(e400 - 0.222) <= 0.5e-3,
              "K7", "eta_n spot values 2.22e-3 / 2.22e-2 / 0.222", e4);
    }
    {
        Vec bw(m.bCell.size(), 0.7); r3ZeroBoundaryWeights(bw);
        double mxb = 0; for (double v : bw) mxb = std::max(mxb, std::fabs(v));
        Coeffs cw = cr; cw.bNu3w = bw;
        UmfpackLU lu; Step s; Vec x0(5*N, 0.0); solveStep(m, cw, 2, x0, lu, s);
        check(mxb == 0.0, "K8", "every boundary R3 weight is 0", mxb);
        check(s.status == UMFPACK_OK, "K8", "the step runs with R3 on", double(s.status));
    }
    {
        // R3 at O(1) weight here so it dominates the row and the comparison is
        // not limited by the round-off of the other terms.
        Coeffs ca = cr; for (int P = 0; P < N; ++P) ca.isSkel[P] = 1;
        for (std::size_t f = 0; f < ca.nu3w.size(); ++f) ca.nu3w[f] = 0.2 + 0.6*U01(rng);
        Coeffs cb = ca; cb.r3 = false;
        Csr Aa, Ab; Vec ba, bb; std::vector<GradRow> Ga = gaussGradRows(m, ca);
        assemble(m, ca, Ga, Lq, Aa, ba); assemble(m, cb, Ga, Lq, Ab, bb);
        Vec x(5*N); for (auto& v : x) v = Upm(rng);
        auto apply = [&](const Csr& M, const Vec& v) { Vec y(M.n, 0.0); for (int r = 0; r < M.n; ++r) for (int j = M.p[r]; j < M.p[r + 1]; ++j) y[r] += M.x[j]*v[M.i[j]]; return y; };
        Vec ya = apply(Aa, x), yb = apply(Ab, x), vx(N), vy(N), fx, fy;
        for (int P = 0; P < N; ++P) { vx[P] = x[idx(P, 2)]; vy[P] = x[idx(P, 3)]; }
        r3Force(m, ca, Lq, vx, vy, fx, fy);
        double d = 0, s = 0; for (int P = 0; P < N; ++P) { d = std::max(d, std::fabs(ya[idx(P, 2)] - yb[idx(P, 2)] + fx[P])); d = std::max(d, std::fabs(ya[idx(P, 3)] - yb[idx(P, 3)] + fy[P])); s = std::max(s, std::max(std::fabs(fx[P]), std::fabs(fy[P]))); }
        check(d <= 1e-12*s, "K9", "assembled R3 block equals -R3 force on skeleton rows", s > 0 ? d/s : d);

        Geo go = makeMesh(10, 8, 0.0, 1.0); const Mesh& mo = go.m; const int No = mo.nCells;
        std::vector<LsqRow> Lo; lsqRows(mo, Lo);
        Coeffs co = makeCoeffs(go, rng, 0.0); co.r3 = true; for (std::size_t f = 0; f < mo.own.size(); ++f) co.nu3w[f] = nu3;
        Vec ux(No), uy(No, 0.0), gx, gy; for (int P = 0; P < No; ++P) ux[P] = mo.Cx[P]*mo.Cy[P];
        r3Force(mo, co, Lo, ux, uy, gx, gy);
        double e = 0; int cnt = 0;
        for (int j = 2; j < 8 - 2; ++j) for (int i = 2; i < 10 - 2; ++i) { const int P = j*10 + i; e = std::max(e, std::fabs(gx[P]/(nu3*mo.V[P]))); e = std::max(e, std::fabs(gy[P]/(nu3*mo.V[P]) - 1.0)); ++cnt; }
        check(cnt > 0 && e <= 1e-9, "K9", "v=(xy,0): div(nu(grad v+grad v^T)) = nu(0,1) in interior cells", e);
    }

    // ---------------- K10: R3's diagonal for the face inverse (F16) ----------------
    {
        // Reference: probe r3Force with unit vectors e_(P,k); minus the force
        // on (P,k) is the row's own coefficient. Mean over k, per volume.
        Coeffs cg = cr;
        for (std::size_t f = 0; f < cg.nu3w.size(); ++f) cg.nu3w[f] = 0.2 + 0.6*U01(rng);
        std::vector<double> dAs; r3AsDiagonal(m, cg, Lq, dAs);
        double d = 0, s = 0;
        for (int P = 0; P < N; ++P)
        {
            double ref = 0;
            for (int k = 0; k < 2; ++k)
            {
                Vec vx(N, 0.0), vy(N, 0.0), fx, fy;
                (k == 0 ? vx : vy)[P] = 1.0;
                r3Force(m, cg, Lq, vx, vy, fx, fy);
                ref += -(k == 0 ? fx[P] : fy[P]);
            }
            ref *= 0.5/m.V[P];
            d = std::max(d, std::fabs(dAs[P] - ref)); s = std::max(s, std::fabs(ref));
        }
        check(s > 0 && d <= 1e-12*s, "K10", "R3 diagonal in A_s equals the dense reference", s > 0 ? d/s : 1.0);
    }

    {
        // K11: random faces, the elastic coefficient built from cell values
        // exactly as the solver's elastic site does (series weight, linear
        // interpolation of gD*G), independently of the kernel's helpers.
        const double tauV = 2.5e-4, rho = 1060.0, etaR = 1e-6, fl = 1e-4;
        std::mt19937 rk(1105);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double d = 0, s = 0;
        for (int f = 0; f < 4000; ++f)
        {
            double wC[2], gG[2];
            for (int k = 0; k < 2; ++k)
            {
                const double ind = 1.2*u(rk), dmg = u(rk), Dgk = u(rk), Gk = 4 + 396*u(rk);
                wC[k] = std::max(ind, fl)*(1 - dmg);
                gG[k] = ((1 - Dgk)*(1 - Dgk) + etaR)*Gk;
            }
            const double l = 0.2 + 0.6*u(rk);
            const double den = l*wC[1] + (1 - l)*wC[0];
            const double sWf = den > 0 ? wC[0]*wC[1]/den : 0.0;
            const double eF = sWf*(l*gG[0] + (1 - l)*gG[1]);
            const double nu = r3FaceCoefficient(sWf, l*gG[0] + (1 - l)*gG[1], tauV, rho);
            d = std::max(d, std::fabs(nu*rho - tauV*eF)); s = std::max(s, tauV*eF);
        }
        check(s > 0 && d <= 1e-12*s, "K11", "applied R3 face coefficient = tau_v x the elastic face coefficient", s > 0 ? d/s : 1.0);
        double worst = 0;
        for (const double Dg : {0.0, 0.5})
        {
            for (const double G : {4.0, 40.0, 400.0})
            {
                // uniform cells: theta = 0.45 (ind 1), d = 0, so series(w) = 1
                const double theta = 0.45, gD = (1 - Dg)*(1 - Dg) + etaR;
                const double code = r3FaceCoefficient(1.0, gD*G, tauV, rho)*rho;
                const double design = theta*etaN(G, tauV, 0.45)*gD;
                std::printf("K11 spot G=%g Dg=%g: design series(theta(1-d)) eta_n gD = %.6e Pa s, code nu3w*rho = %.6e Pa s, tau_KV = %.6e s\n",
                            G, Dg, design, code, code/(gD*G));
                worst = std::max(worst, std::fabs(code - design)/design);
            }
        }
        check(worst <= 1e-12, "K11", "uniform-cell face coefficient = theta(1 - d) eta_n gD at G 4/40/400", worst);
    }

    std::printf("%s: %d failure(s)\n", failures ? "KERNEL TEST FAILED" : "KERNEL TEST PASSED", failures);
    return failures ? 1 : 0;
}
