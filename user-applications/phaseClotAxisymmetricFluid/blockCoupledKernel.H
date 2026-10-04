/*---------------------------------------------------------------------------*\
    blockCoupledKernel.H -- S19, the block-coupled (Ux, Uy, Vsx, Vsy, p) step

    research/designs/2026-10-03-S19-block-coupled-step.md, rev. 4, APPROVED
    by Alireza 2026-10-04 (docket research/investigations/20261003-0205-np-
    block-coupled-solve/). Header-only and free of OpenFOAM, so the kernel
    test (scripts/s19_block_kernel_test.cpp, run by scripts/
    check_s19_block_coupled.py) compiles and MUTATES the very text the solver
    includes.

    What is here (design section 2; numerics 02 section 1 rows; the replay
    probe probes/numerics-analyst-06-g1.py is the reference assembly):

      - the rows: U (the shipped UEqn coefficients + K U - K Vs + phi_f grad p),
        Vs (the shipped VsEqn coefficients + (K+Kres)(Vs - U) + phi_s grad p
        [+ R3]), and the p row, mixture continuity on implicit Rhie-Chow
        faces; lumen cells (isSkel 0) carry the identity rows Vs - U = 0;
      - UMFPACK direct LU, the symbolic factorisation kept while the CSR
        pattern (Ap, Ai) is unchanged and redone whenever it changes (red
        team 07 F5), the numeric factorisation every step;
      - k = nNonOrthogonalCorrectors back-substitutions with the same LU, the
        explicit non-orthogonal flux correction updated from the new p between
        them; the final F_f carries the correction the LAST back-substitution
        was solved with (red team 07 F4), so linRes and the cell sums of F_f
        are taken on that final system;
      - R3 (section 12), -div(w_f nu3 (grad Vs + grad Vs^T)), fully implicit:
        the compact parts (2 nu3 normal, nu3 tangential) and the transpose
        cross terms Vsx <-> Vsy from face-interpolated central (least-squares,
        internal neighbours only) tangential derivatives, in the same block.

    Units: the momentum rows are volume-integrated and kinematic (fvMatrix
    units: the shipped UEqn/VsEqn diag and source); K, Kres are 1/s; the p row
    is the face-flux sum in m^3/s.
\*---------------------------------------------------------------------------*/

#ifndef blockCoupledKernel_H
#define blockCoupledKernel_H

#include <umfpack.h>

#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <utility>

namespace blockCoupled
{

//- Unknowns per cell, interleaved: Ux, Uy, Vsx, Vsy, p.
static const int nVar = 5;

inline int idx(const int c, const int k)
{
    return nVar*c + k;
}


//- The mesh, 2-D (x, y). Internal faces own -> nei; boundary faces carry the
//  cell they belong to and whether U is assignable there (open: inlet,
//  outlet) or prescribed (wall: F = U_b.S, no p, no R3).
struct Mesh
{
    int nCells = 0;
    std::vector<double> V, Cx, Cy;

    std::vector<int> own, nei;
    std::vector<double> Sx, Sy, magS, w, ndc, kx, ky;

    std::vector<int> bCell;
    std::vector<double> bSx, bSy, bMagS;
    std::vector<int> bOpen;
};


//- One step's coefficients. Every array is read from the solver's existing
//  fields and matrices; the kernel computes none of them (design section 5
//  (a): K and Kres are the solver's drag fields, never recomputed).
struct Coeffs
{
    // Momentum rows, per cell and component (0 x, 1 y): the diagonal with
    // the boundary internalCoeffs added, and the source with the boundary
    // boundaryCoeffs added, exactly as fvMatrix::solveSegregated forms them.
    std::vector<double> dU[2], sU[2], dVs[2], sVs[2];
    // Off-diagonals per internal face: upper = row own, column nei; lower =
    // row nei, column own (lduMatrix addressing).
    std::vector<double> upU, loU, upVs, loVs;

    std::vector<double> K, Kres, phiF, phiS;
    // The explicit elastic force in cell form, reconstruct(fSigmaE) (per
    // unit volume, the units of H).
    std::vector<double> FelX, FelY;
    std::vector<char> isSkel;

    // Internal faces: phase fractions, the face-assembled 2x2 inverse's
    // pressure coefficient rA_f and its elastic coefficient
    // cE_f = phiFf*iAD01f + phiSf*iAD11f, and the face elastic force flux.
    std::vector<double> phiFf, phiSf, rAf, cEf, fSigE;
    // Boundary faces: the same, plus the prescribed flux on walls.
    std::vector<double> bPhiF, bPhiS, bRAf, bCE, bFSigE, bWallFlux;
    // p on boundary faces: value p_b = pvIC p_P + pvBC; snGrad p_b =
    // pgIC p_P + pgBC (the p patch field's own coefficients).
    std::vector<double> pvIC, pvBC, pgIC, pgBC;

    // R3: per internal face nu3_f*w_f (0: no viscosity across the face);
    // per boundary face the weight, which must be 0 (r3ZeroBoundaryWeights).
    bool r3 = false;
    std::vector<double> nu3w, bNu3w;
};


//- Compressed sparse rows.
struct Csr
{
    int n = 0;
    std::vector<int> p, i;
    std::vector<double> x;
};


// ---------------------------------------------------------------------------
//  R3 parameters (section 12)
// ---------------------------------------------------------------------------

//- eta_n = G tau_v / phi0 (Pa s). Spot values: 2.22e-3 at G 4, 2.22e-2 at
//  G 40, 0.222 at G 400 (tau_v 2.5e-4 s, phi0 0.45).
inline double etaN(const double G, const double tauV, const double phi0)
{
    return G*tauV/phi0;
}

//- The elastic stress's face modulus per unit stress shape on the face:
//  series(w)_f times interp(gD G)_f (the solver's elastic site builds
//  seriesFaceWeights(w)*fvc::interpolate(gD*G*M)).
inline double elasticFaceModulus(const double sW, const double gDGf)
{
    return sW*gDGf;
}

//- R3's applied face coefficient nu3w_f (m^2/s): tau_v times the elastic
//  face modulus on the same weight and the same gD, over rho, so R3 over
//  the elastic coefficient is tau_v on every face (fix round 12, physics 11
//  row 5b). With w = theta(1 - d)/phi0 it is series(theta(1 - d)) eta_n gD.
inline double r3FaceCoefficient
(
    const double sW, const double gDGf, const double tauV, const double rho
)
{
    return tauV*elasticFaceModulus(sW, gDGf)/rho;
}

//- R3 puts no viscous traction on any boundary face: seriesFaceWeights
//  keeps the patch value (.C 7653-7655), so every boundary weight is set
//  to 0 here, explicitly.
inline void r3ZeroBoundaryWeights(std::vector<double>& bW)
{
    for (std::size_t f = 0; f < bW.size(); ++f)
    {
        bW[f] = 0.0;
    }
}


// ---------------------------------------------------------------------------
//  Operators shared by the assembly, the fluxes and the diagnostics
// ---------------------------------------------------------------------------

//- One cell's Gauss-linear gradient of p as a linear form:
//  grad p_P = sum_j (gx_j, gy_j) p_{cell_j} + (cx, cy).
struct GradRow
{
    std::vector<int> cell;
    std::vector<double> gx, gy;
    double cx = 0, cy = 0;

    void add(const int c, const double ax, const double ay)
    {
        for (std::size_t j = 0; j < cell.size(); ++j)
        {
            if (cell[j] == c)
            {
                gx[j] += ax;
                gy[j] += ay;
                return;
            }
        }
        cell.push_back(c);
        gx.push_back(ax);
        gy.push_back(ay);
    }
};

//- Gauss linear, as fvc::grad(p) under gradSchemes Gauss linear: internal
//  face values w p_own + (1 - w) p_nei, boundary values from p's patch
//  coefficients.
inline std::vector<GradRow> gaussGradRows(const Mesh& m, const Coeffs& c)
{
    std::vector<GradRow> G(m.nCells);
    for (std::size_t f = 0; f < m.own.size(); ++f)
    {
        const int o = m.own[f], n = m.nei[f];
        const double wf = m.w[f];
        const double io = 1.0/m.V[o], in = 1.0/m.V[n];
        G[o].add(o, m.Sx[f]*wf*io, m.Sy[f]*wf*io);
        G[o].add(n, m.Sx[f]*(1 - wf)*io, m.Sy[f]*(1 - wf)*io);
        G[n].add(o, -m.Sx[f]*wf*in, -m.Sy[f]*wf*in);
        G[n].add(n, -m.Sx[f]*(1 - wf)*in, -m.Sy[f]*(1 - wf)*in);
    }
    for (std::size_t b = 0; b < m.bCell.size(); ++b)
    {
        const int P = m.bCell[b];
        const double iv = 1.0/m.V[P];
        G[P].add(P, m.bSx[b]*c.pvIC[b]*iv, m.bSy[b]*c.pvIC[b]*iv);
        G[P].cx += m.bSx[b]*c.pvBC[b]*iv;
        G[P].cy += m.bSy[b]*c.pvBC[b]*iv;
    }
    return G;
}

//- Evaluate the gradient rows on a p field (one entry per cell).
inline void evalGrad
(
    const std::vector<GradRow>& G, const std::vector<double>& p,
    std::vector<double>& gx, std::vector<double>& gy
)
{
    const std::size_t n = G.size();
    gx.assign(n, 0.0);
    gy.assign(n, 0.0);
    for (std::size_t P = 0; P < n; ++P)
    {
        double ax = G[P].cx, ay = G[P].cy;
        for (std::size_t j = 0; j < G[P].cell.size(); ++j)
        {
            ax += G[P].gx[j]*p[G[P].cell[j]];
            ay += G[P].gy[j]*p[G[P].cell[j]];
        }
        gx[P] = ax;
        gy[P] = ay;
    }
}


//- The least-squares gradient weights over INTERNAL neighbours only, weight
//  1/|d|^2: grad v_P = sum_m g_{P,m} (v_m - v_P). Exact for linear fields on
//  any mesh (boundary cells included), and the central difference in the
//  interior of a uniform grid -- the "central tangential derivatives" of
//  section 12. A cell without two independent neighbours cannot carry it:
//  returns false.
struct LsqRow
{
    std::vector<int> cell;
    std::vector<double> gx, gy;
};

inline bool lsqRows(const Mesh& m, std::vector<LsqRow>& L)
{
    L.assign(m.nCells, LsqRow());
    std::vector<double> mxx(m.nCells, 0.0), mxy(m.nCells, 0.0), myy(m.nCells, 0.0);
    for (std::size_t f = 0; f < m.own.size(); ++f)
    {
        const int o = m.own[f], n = m.nei[f];
        const double dx = m.Cx[n] - m.Cx[o], dy = m.Cy[n] - m.Cy[o];
        const double wt = 1.0/(dx*dx + dy*dy);
        mxx[o] += wt*dx*dx; mxy[o] += wt*dx*dy; myy[o] += wt*dy*dy;
        mxx[n] += wt*dx*dx; mxy[n] += wt*dx*dy; myy[n] += wt*dy*dy;
    }
    for (int P = 0; P < m.nCells; ++P)
    {
        const double det = mxx[P]*myy[P] - mxy[P]*mxy[P];
        if (!(det > 1e-12*(mxx[P]*myy[P])))
        {
            return false;
        }
    }
    for (std::size_t f = 0; f < m.own.size(); ++f)
    {
        const int o = m.own[f], n = m.nei[f];
        const double dx = m.Cx[n] - m.Cx[o], dy = m.Cy[n] - m.Cy[o];
        const double wt = 1.0/(dx*dx + dy*dy);
        for (int side = 0; side < 2; ++side)
        {
            const int P = side == 0 ? o : n;
            const int Q = side == 0 ? n : o;
            const double sx = side == 0 ? dx : -dx;
            const double sy = side == 0 ? dy : -dy;
            const double det = mxx[P]*myy[P] - mxy[P]*mxy[P];
            const double ixx = myy[P]/det, ixy = -mxy[P]/det, iyy = mxx[P]/det;
            L[P].cell.push_back(Q);
            L[P].gx.push_back(wt*(ixx*sx + ixy*sy));
            L[P].gy.push_back(wt*(ixy*sx + iyy*sy));
        }
    }
    return true;
}


//- R3's face traction T_f = nu3w_f (Jf + Jf^T).S_f as a linear form in Vs:
//  T_a = sum_t coef[t][a] Vs_{cell[t], comp[t]}. Jf = Ji + (delta - Ji e) e^T
//  with Ji = w J_own + (1 - w) J_nei (least-squares cell gradients),
//  e = d/|d|, delta = (Vs_nei - Vs_own)/|d|. Expanded:
//    T = nu3w [ (e.S) delta + e (delta.S)       -- compact: 2nu3 normal,
//                                                   nu3 tangential
//             + Ji s_perp                         -- non-orthogonal part
//             + (I - e e^T) Ji^T S ]             -- the transpose cross
//                                                   terms Vsx <-> Vsy
//  with s_perp = S - (e.S) e.
struct FaceForm
{
    std::vector<int> cell, comp;
    std::vector<double> c0, c1;

    void add(const int ce, const int k, const double a0, const double a1)
    {
        for (std::size_t t = 0; t < cell.size(); ++t)
        {
            if (cell[t] == ce && comp[t] == k)
            {
                c0[t] += a0;
                c1[t] += a1;
                return;
            }
        }
        cell.push_back(ce);
        comp.push_back(k);
        c0.push_back(a0);
        c1.push_back(a1);
    }
};

inline void r3FaceForm
(
    const Mesh& m, const std::vector<LsqRow>& L, const std::size_t f,
    const double nw, FaceForm& T
)
{
    T = FaceForm();
    const int o = m.own[f], n = m.nei[f];
    const double dx = m.Cx[n] - m.Cx[o], dy = m.Cy[n] - m.Cy[o];
    const double dmag = std::sqrt(dx*dx + dy*dy);
    const double e[2] = {dx/dmag, dy/dmag};
    const double S[2] = {m.Sx[f], m.Sy[f]};
    const double eS = e[0]*S[0] + e[1]*S[1];
    const double sp[2] = {S[0] - eS*e[0], S[1] - eS*e[1]};

    // Compact: delta_i = (v_n,i - v_o,i)/|d|.
    for (int i = 0; i < 2; ++i)
    {
        const double a0 = nw/dmag*(eS*(i == 0 ? 1.0 : 0.0) + e[0]*S[i]);
        const double a1 = nw/dmag*(eS*(i == 1 ? 1.0 : 0.0) + e[1]*S[i]);
        T.add(n, i, a0, a1);
        T.add(o, i, -a0, -a1);
    }

    // The interpolated cell gradients: Ji_ij = sum_P wP sum_q g_{P,q,j}
    // (v_q,i - v_P,i).
    const double transposeOn = 1.0;
    for (int side = 0; side < 2; ++side)
    {
        const int P = side == 0 ? o : n;
        const double wP = side == 0 ? m.w[f] : 1.0 - m.w[f];
        for (std::size_t q = 0; q < L[P].cell.size(); ++q)
        {
            const int Q = L[P].cell[q];
            const double g[2] = {L[P].gx[q], L[P].gy[q]};
            // Ji s_perp: component a of T reads v component a.
            const double gs = nw*wP*(g[0]*sp[0] + g[1]*sp[1]);
            T.add(Q, 0, gs, 0.0);
            T.add(P, 0, -gs, 0.0);
            T.add(Q, 1, 0.0, gs);
            T.add(P, 1, 0.0, -gs);
            // (I - e e^T) Ji^T S: component a reads v component j through
            // h_a S_j, h = g - e (e.g).
            const double eg = e[0]*g[0] + e[1]*g[1];
            const double h[2] = {g[0] - e[0]*eg, g[1] - e[1]*eg};
            for (int j = 0; j < 2; ++j)
            {
                const double b0 = transposeOn*nw*wP*h[0]*S[j];
                const double b1 = transposeOn*nw*wP*h[1]*S[j];
                T.add(Q, j, b0, b1);
                T.add(P, j, -b0, -b1);
            }
        }
    }
}

//- The R3 force on every cell (volume-integrated, kinematic): +T on the
//  owner, -T on the neighbour. vs is (Vsx, Vsy) per cell.
inline void r3Force
(
    const Mesh& m, const Coeffs& c, const std::vector<LsqRow>& L,
    const std::vector<double>& vsx, const std::vector<double>& vsy,
    std::vector<double>& fx, std::vector<double>& fy
)
{
    fx.assign(m.nCells, 0.0);
    fy.assign(m.nCells, 0.0);
    FaceForm T;
    for (std::size_t f = 0; f < m.own.size(); ++f)
    {
        if (c.nu3w[f] == 0)
        {
            continue;
        }
        r3FaceForm(m, L, f, c.nu3w[f], T);
        double t0 = 0, t1 = 0;
        for (std::size_t t = 0; t < T.cell.size(); ++t)
        {
            const double v = T.comp[t] == 0 ? vsx[T.cell[t]] : vsy[T.cell[t]];
            t0 += T.c0[t]*v;
            t1 += T.c1[t]*v;
        }
        fx[m.own[f]] += t0; fy[m.own[f]] += t1;
        fx[m.nei[f]] -= t0; fy[m.nei[f]] -= t1;
    }
}


//- R3's diagonal for the face-assembled 2x2 inverse (CE ruling F16,
//  2026-10-04; numerics 02 section 1: "A_s includes R3's diagonal"): per
//  cell and PER UNIT VOLUME, the mean over x and y of the coefficient each
//  skeleton row puts on its own cell's own component (-T on the owner row,
//  +T on the neighbour row, as assemble() does) -- the component average,
//  as fvMatrix::A() averages the boundary diagonal.
inline void r3AsDiagonal
(
    const Mesh& m, const Coeffs& c, const std::vector<LsqRow>& L,
    std::vector<double>& dAs
)
{
    std::vector<double> sum(m.nCells, 0.0);
    FaceForm F;
    for (std::size_t f = 0; f < m.own.size(); ++f)
    {
        if (c.nu3w[f] == 0)
        {
            continue;
        }
        r3FaceForm(m, L, f, c.nu3w[f], F);
        const int o = m.own[f], n = m.nei[f];
        for (std::size_t t = 0; t < F.cell.size(); ++t)
        {
            const double own = F.comp[t] == 0 ? F.c0[t] : F.c1[t];
            if (F.cell[t] == o) { sum[o] -= own; }
            if (F.cell[t] == n) { sum[n] += own; }
        }
    }
    dAs.assign(m.nCells, 0.0);
    for (int P = 0; P < m.nCells; ++P)
    {
        dAs[P] = 0.5*sum[P]/m.V[P];
    }
}


// ---------------------------------------------------------------------------
//  Assembly
// ---------------------------------------------------------------------------

struct Triplets
{
    std::vector<std::vector<std::pair<int, double>>> rows;

    explicit Triplets(const int n) : rows(n) {}

    void add(const int r, const int col, const double v)
    {
        rows[r].push_back(std::make_pair(col, v));
    }

    //- Sorted columns, duplicates summed; every structural entry is kept
    //  (an exact zero included), so the pattern depends on the mesh, the
    //  lumen mask and the R3 switch only.
    void toCsr(Csr& A) const
    {
        const int n = int(rows.size());
        A.n = n;
        A.p.assign(n + 1, 0);
        A.i.clear();
        A.x.clear();
        std::vector<std::pair<int, double>> r;
        for (int k = 0; k < n; ++k)
        {
            r = rows[k];
            std::sort
            (
                r.begin(), r.end(),
                [](const std::pair<int, double>& a, const std::pair<int, double>& b)
                { return a.first < b.first; }
            );
            for (std::size_t j = 0; j < r.size(); ++j)
            {
                if (!A.i.empty() && int(A.i.size()) > A.p[k] && A.i.back() == r[j].first)
                {
                    A.x.back() += r[j].second;
                }
                else
                {
                    A.i.push_back(r[j].first);
                    A.x.push_back(r[j].second);
                }
            }
            A.p[k + 1] = int(A.i.size());
        }
    }
};

//- Add rA*S.interp(G) of face (o, n) -- the implicit n.interp(grad p) part
//  of the Rhie-Chow face -- to p row `row` with sign sg; its constant part
//  goes to `cst`.
inline void addRhieChow
(
    Triplets& T, const int row, const double sg, const double coef,
    const double Sx, const double Sy, const GradRow& Go, const double wo,
    const GradRow* Gn, const double wn, double& cst
)
{
    for (std::size_t j = 0; j < Go.cell.size(); ++j)
    {
        T.add(row, idx(Go.cell[j], 4), sg*coef*wo*(Sx*Go.gx[j] + Sy*Go.gy[j]));
    }
    cst += coef*wo*(Sx*Go.cx + Sy*Go.cy);
    if (Gn)
    {
        for (std::size_t j = 0; j < Gn->cell.size(); ++j)
        {
            T.add(row, idx(Gn->cell[j], 4), sg*coef*wn*(Sx*Gn->gx[j] + Sy*Gn->gy[j]));
        }
        cst += coef*wn*(Sx*Gn->cx + Sy*Gn->cy);
    }
}

//- The block matrix and its right-hand side WITHOUT the explicit
//  non-orthogonal correction (added per back-substitution). Returns the
//  number of lumen identity rows.
inline int assemble
(
    const Mesh& m, const Coeffs& c, const std::vector<GradRow>& G,
    const std::vector<LsqRow>& L, Csr& A, std::vector<double>& b
)
{
    const int N = m.nCells;
    Triplets T(nVar*N);
    b.assign(nVar*N, 0.0);
    int nIdentity = 0;

    // ---- the momentum rows ----
    for (int P = 0; P < N; ++P)
    {
        // THE DRAG IS THE SOLVER'S OWN FIELD (design section 5 (a)): K and
        // Kres read as given, the same values in both rows.
        const double KV = c.K[P]*m.V[P];
        const double KsV = (c.K[P] + c.Kres[P])*m.V[P];
        for (int k = 0; k < 2; ++k)
        {
            const int rU = idx(P, k);
            T.add(rU, rU, c.dU[k][P] + KV);
            T.add(rU, idx(P, 2 + k), -KV);
            b[rU] += c.sU[k][P];

            const int rS = idx(P, 2 + k);
            if (c.isSkel[P])
            {
                T.add(rS, rS, c.dVs[k][P] + KsV);
                T.add(rS, idx(P, k), -KsV);
                b[rS] += c.sVs[k][P] + m.V[P]*(k == 0 ? c.FelX[P] : c.FelY[P]);
            }
            else
            {
                // THE LUMEN IDENTITY ROW, Vs - U = 0, replacing the shipped
                // post-solve overwrite Vs = isSkel Vs + (1 - isSkel) U.
                T.add(rS, rS, 1.0);
                T.add(rS, idx(P, k), -1.0);
            }
        }
        if (!c.isSkel[P])
        {
            ++nIdentity;
        }

        // phi grad p, Gauss linear, on both phases' rows.
        for (int k = 0; k < 2; ++k)
        {
            const double cU = c.phiF[P]*m.V[P];
            const double cS = c.phiS[P]*m.V[P];
            for (std::size_t j = 0; j < G[P].cell.size(); ++j)
            {
                const double g = k == 0 ? G[P].gx[j] : G[P].gy[j];
                T.add(idx(P, k), idx(G[P].cell[j], 4), cU*g);
                if (c.isSkel[P])
                {
                    T.add(idx(P, 2 + k), idx(G[P].cell[j], 4), cS*g);
                }
            }
            const double gc = k == 0 ? G[P].cx : G[P].cy;
            b[idx(P, k)] -= cU*gc;
            if (c.isSkel[P])
            {
                b[idx(P, 2 + k)] -= cS*gc;
            }
        }
    }

    // The shipped matrices' off-diagonals.
    for (std::size_t f = 0; f < m.own.size(); ++f)
    {
        const int o = m.own[f], n = m.nei[f];
        for (int k = 0; k < 2; ++k)
        {
            T.add(idx(o, k), idx(n, k), c.upU[f]);
            T.add(idx(n, k), idx(o, k), c.loU[f]);
            if (c.isSkel[o])
            {
                T.add(idx(o, 2 + k), idx(n, 2 + k), c.upVs[f]);
            }
            if (c.isSkel[n])
            {
                T.add(idx(n, 2 + k), idx(o, 2 + k), c.loVs[f]);
            }
        }
    }

    // R3 on the skeleton rows: -div(tau) integrated is -T on the owner and
    // +T on the neighbour.
    if (c.r3)
    {
        FaceForm F;
        for (std::size_t f = 0; f < m.own.size(); ++f)
        {
            if (c.nu3w[f] == 0)
            {
                continue;
            }
            r3FaceForm(m, L, f, c.nu3w[f], F);
            const int o = m.own[f], n = m.nei[f];
            for (std::size_t t = 0; t < F.cell.size(); ++t)
            {
                const int col = idx(F.cell[t], 2 + F.comp[t]);
                if (c.isSkel[o])
                {
                    T.add(idx(o, 2), col, -F.c0[t]);
                    T.add(idx(o, 3), col, -F.c1[t]);
                }
                if (c.isSkel[n])
                {
                    T.add(idx(n, 2), col, F.c0[t]);
                    T.add(idx(n, 3), col, F.c1[t]);
                }
            }
        }
    }

    // ---- the p row: sum over the cell's faces of the outward F_f = 0 ----
    for (std::size_t f = 0; f < m.own.size(); ++f)
    {
        const int o = m.own[f], n = m.nei[f];
        const double wf = m.w[f];
        double cst = 0;
        for (int sgi = 0; sgi < 2; ++sgi)
        {
            const int row = idx(sgi == 0 ? o : n, 4);
            const double sg = sgi == 0 ? 1.0 : -1.0;
            for (int k = 0; k < 2; ++k)
            {
                const double Sk = k == 0 ? m.Sx[f] : m.Sy[f];
                T.add(row, idx(o, k), sg*c.phiFf[f]*Sk*wf);
                T.add(row, idx(n, k), sg*c.phiFf[f]*Sk*(1 - wf));
                T.add(row, idx(o, 2 + k), sg*c.phiSf[f]*Sk*wf);
                T.add(row, idx(n, 2 + k), sg*c.phiSf[f]*Sk*(1 - wf));
            }
            const double cp = c.rAf[f]*m.magS[f]*m.ndc[f];
            T.add(row, idx(n, 4), -sg*cp);
            T.add(row, idx(o, 4), sg*cp);
            // THE IMPLICIT RHIE-CHOW TERM, + rA_f S.interp(grad p).
            double cRC = 0;
            addRhieChow(T, row, sg, c.rAf[f], m.Sx[f], m.Sy[f], G[o], wf, &G[n], 1 - wf, cRC);
            if (sgi == 0)
            {
                cst += cRC;
            }
        }
        // The explicit elastic face term, cE_f (fSigmaE - S.interp(Fel)).
        cst += c.cEf[f]*
        (
            c.fSigE[f]
          - m.Sx[f]*(wf*c.FelX[o] + (1 - wf)*c.FelX[n])
          - m.Sy[f]*(wf*c.FelY[o] + (1 - wf)*c.FelY[n])
        );
        b[idx(o, 4)] -= cst;
        b[idx(n, 4)] += cst;
    }
    for (std::size_t bf = 0; bf < m.bCell.size(); ++bf)
    {
        const int P = m.bCell[bf];
        const int row = idx(P, 4);
        if (!m.bOpen[bf])
        {
            b[row] -= c.bWallFlux[bf];
            continue;
        }
        double cst = 0;
        for (int k = 0; k < 2; ++k)
        {
            const double Sk = k == 0 ? m.bSx[bf] : m.bSy[bf];
            T.add(row, idx(P, k), c.bPhiF[bf]*Sk);
            T.add(row, idx(P, 2 + k), c.bPhiS[bf]*Sk);
        }
        T.add(row, idx(P, 4), -c.bRAf[bf]*m.bMagS[bf]*c.pgIC[bf]);
        cst -= c.bRAf[bf]*m.bMagS[bf]*c.pgBC[bf];
        addRhieChow(T, row, 1.0, c.bRAf[bf], m.bSx[bf], m.bSy[bf], G[P], 1.0, nullptr, 0.0, cst);
        cst += c.bCE[bf]*(c.bFSigE[bf] - m.bSx[bf]*c.FelX[P] - m.bSy[bf]*c.FelY[P]);
        b[row] -= cst;
    }

    T.toCsr(A);
    return nIdentity;
}


//- The explicit non-orthogonal part of |S| snGrad p from a p field:
//  kc_f = |S_f| k_f.interp(grad p*) (zero on boundary faces, as OpenFOAM's
//  non-orthogonal correction vectors are). The face flux carries -rA_f kc_f.
inline void nonOrthFaceCorr
(
    const Mesh& m, const std::vector<GradRow>& G,
    const std::vector<double>& pStar, std::vector<double>& kc
)
{
    std::vector<double> gx, gy;
    evalGrad(G, pStar, gx, gy);
    kc.assign(m.own.size(), 0.0);
    for (std::size_t f = 0; f < m.own.size(); ++f)
    {
        const int o = m.own[f], n = m.nei[f];
        const double wf = m.w[f];
        kc[f] = m.magS[f]*
        (
            m.kx[f]*(wf*gx[o] + (1 - wf)*gx[n])
          + m.ky[f]*(wf*gy[o] + (1 - wf)*gy[n])
        );
    }
}


// ---------------------------------------------------------------------------
//  UMFPACK, the symbolic factorisation keyed to the CSR pattern
// ---------------------------------------------------------------------------

class UmfpackLU
{
    void* symbolic_;
    void* numeric_;
    int n_;
    std::vector<int> Ap_, Ai_;
    long symbolicCount_;
    double rcond_;
    double control_[UMFPACK_CONTROL];
    double info_[UMFPACK_INFO];

    UmfpackLU(const UmfpackLU&);
    void operator=(const UmfpackLU&);

    static double now()
    {
        return std::chrono::duration<double>
        (
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

public:

    UmfpackLU()
    :
        symbolic_(nullptr), numeric_(nullptr), n_(-1),
        symbolicCount_(0), rcond_(0)
    {
        umfpack_di_defaults(control_);
        control_[UMFPACK_PRL] = 0;
    }

    ~UmfpackLU()
    {
        clear();
    }

    void clear()
    {
        if (numeric_) { umfpack_di_free_numeric(&numeric_); numeric_ = nullptr; }
        if (symbolic_) { umfpack_di_free_symbolic(&symbolic_); symbolic_ = nullptr; }
        n_ = -1;
        Ap_.clear();
        Ai_.clear();
    }

    long symbolicCount() const { return symbolicCount_; }
    double rcond() const { return rcond_; }

    //- The CSR rows of A are handed to UMFPACK as the columns of A^T; the
    //  solve below therefore asks for the transposed system. Returns the
    //  first status that is not UMFPACK_OK (the singular-matrix warning
    //  included), with `stage` 1 symbolic, 2 numeric.
    int factor(const Csr& A, double& tSym, double& tNum, int& stage)
    {
        tSym = 0;
        tNum = 0;
        stage = 0;
        const bool samePattern = (symbolic_ != nullptr && A.n == n_ && A.p == Ap_ && A.i == Ai_);
        if (!samePattern)
        {
            const double t0 = now();
            if (symbolic_) { umfpack_di_free_symbolic(&symbolic_); symbolic_ = nullptr; }
            const int st = umfpack_di_symbolic
            (
                A.n, A.n, A.p.data(), A.i.data(), A.x.data(),
                &symbolic_, control_, info_
            );
            tSym = now() - t0;
            if (st != UMFPACK_OK)
            {
                stage = 1;
                return st;
            }
            n_ = A.n;
            Ap_ = A.p;
            Ai_ = A.i;
            ++symbolicCount_;
        }
        const double t1 = now();
        if (numeric_) { umfpack_di_free_numeric(&numeric_); numeric_ = nullptr; }
        const int st = umfpack_di_numeric
        (
            A.p.data(), A.i.data(), A.x.data(), symbolic_, &numeric_,
            control_, info_
        );
        tNum = now() - t1;
        rcond_ = info_[UMFPACK_RCOND];
        if (st != UMFPACK_OK)
        {
            stage = 2;
        }
        return st;
    }

    int backSolve(const Csr& A, const std::vector<double>& b, std::vector<double>& x)
    {
        x.assign(A.n, 0.0);
        return umfpack_di_solve
        (
            UMFPACK_At, A.p.data(), A.i.data(), A.x.data(),
            x.data(), b.data(), numeric_, control_, info_
        );
    }
};


//- ||A x - b||_2 and ||b||_2.
inline void residualNorms
(
    const Csr& A, const std::vector<double>& x, const std::vector<double>& b,
    double& rNorm, double& bNorm
)
{
    double r2 = 0, b2 = 0;
    for (int k = 0; k < A.n; ++k)
    {
        double s = -b[k];
        for (int j = A.p[k]; j < A.p[k + 1]; ++j)
        {
            s += A.x[j]*x[A.i[j]];
        }
        r2 += s*s;
        b2 += b[k]*b[k];
    }
    rNorm = std::sqrt(r2);
    bNorm = std::sqrt(b2);
}


// ---------------------------------------------------------------------------
//  The step
// ---------------------------------------------------------------------------

struct Step
{
    int status = 0;        // UMFPACK status (UMFPACK_OK = 0), or -1000 setup
    int stage = 0;         // 1 symbolic, 2 numeric, 3 solve, 4 setup
    int nIdentity = 0;
    int nBack = 0;
    double tSym = 0, tNum = 0, tSolve = 0;
    double rNorm = 0, bNorm = 0, rcond = 0;
    std::vector<double> x;
    // Face fluxes and their parts: F = phiFf Fu + phiSf Fv - rA Xp + cE El.
    std::vector<double> F, Fu, Fv, Xp, El;
    std::vector<double> bF, bFu, bFv, bXp, bEl;
    std::vector<double> cellSum;   // sum over the cell's faces of outward F
};

//- One step: assemble, factor, k back-substitutions, fluxes. x0 holds the
//  old state (its p is p^n, the start of the explicit correction).
inline void solveStep
(
    const Mesh& m, const Coeffs& c, const int nNonOrth,
    const std::vector<double>& x0, UmfpackLU& lu, Step& s
)
{
    const int N = m.nCells;
    s = Step();
    std::vector<LsqRow> L;
    if (c.r3 && !lsqRows(m, L))
    {
        s.status = -1000;
        s.stage = 4;
        return;
    }
    for (std::size_t bf = 0; bf < c.bNu3w.size(); ++bf)
    {
        if (c.bNu3w[bf] != 0)
        {
            s.status = -1001;
            s.stage = 4;
            return;
        }
    }
    const std::vector<GradRow> G(gaussGradRows(m, c));
    Csr A;
    std::vector<double> b0;
    s.nIdentity = assemble(m, c, G, L, A, b0);
    s.status = lu.factor(A, s.tSym, s.tNum, s.stage);
    s.rcond = lu.rcond();
    if (s.status != UMFPACK_OK)
    {
        return;
    }

    std::vector<double> pStar(N), kc(m.own.size(), 0.0), rhs;
    for (int P = 0; P < N; ++P)
    {
        pStar[P] = x0[idx(P, 4)];
    }
    // k back-substitutions with the same LU; k = 0 is the orthogonal solve.
    const int nBack = nNonOrth;
    s.nBack = nBack;
    const int nSolve = nBack > 0 ? nBack : 1;
    for (int it = 0; it < nSolve; ++it)
    {
        rhs = b0;
        if (nBack > 0)
        {
            nonOrthFaceCorr(m, G, pStar, kc);
            for (std::size_t f = 0; f < m.own.size(); ++f)
            {
                const double e = -c.rAf[f]*kc[f];
                rhs[idx(m.own[f], 4)] -= e;
                rhs[idx(m.nei[f], 4)] += e;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int st = lu.backSolve(A, rhs, s.x);
        s.tSolve += std::chrono::duration<double>
        (
            std::chrono::steady_clock::now() - t0
        ).count();
        if (st != UMFPACK_OK)
        {
            s.status = st;
            s.stage = 3;
            return;
        }
        for (int P = 0; P < N; ++P)
        {
            pStar[P] = s.x[idx(P, 4)];
        }
    }
    residualNorms(A, s.x, rhs, s.rNorm, s.bNorm);

    // ---- the final face fluxes, with the correction the last solve used ----
    std::vector<double> gx, gy;
    std::vector<double> p(N);
    for (int P = 0; P < N; ++P)
    {
        p[P] = s.x[idx(P, 4)];
    }
    evalGrad(G, p, gx, gy);
    const std::size_t nF = m.own.size(), nB = m.bCell.size();
    s.F.assign(nF, 0.0); s.Fu.assign(nF, 0.0); s.Fv.assign(nF, 0.0);
    s.Xp.assign(nF, 0.0); s.El.assign(nF, 0.0);
    s.bF.assign(nB, 0.0); s.bFu.assign(nB, 0.0); s.bFv.assign(nB, 0.0);
    s.bXp.assign(nB, 0.0); s.bEl.assign(nB, 0.0);
    s.cellSum.assign(N, 0.0);
    for (std::size_t f = 0; f < nF; ++f)
    {
        const int o = m.own[f], n = m.nei[f];
        const double wf = m.w[f];
        const double* x = s.x.data();
        s.Fu[f] =
            m.Sx[f]*(wf*x[idx(o, 0)] + (1 - wf)*x[idx(n, 0)])
          + m.Sy[f]*(wf*x[idx(o, 1)] + (1 - wf)*x[idx(n, 1)]);
        s.Fv[f] =
            m.Sx[f]*(wf*x[idx(o, 2)] + (1 - wf)*x[idx(n, 2)])
          + m.Sy[f]*(wf*x[idx(o, 3)] + (1 - wf)*x[idx(n, 3)]);
        // Xp: |S|(snGrad p - n.interp grad p), the non-orthogonal part of
        // snGrad taken from the p the last solve was built with.
        double xp =
            m.magS[f]*m.ndc[f]*(p[n] - p[o])
          - m.Sx[f]*(wf*gx[o] + (1 - wf)*gx[n])
          - m.Sy[f]*(wf*gy[o] + (1 - wf)*gy[n]);
        if (nBack > 0)
        {
            xp += kc[f];
        }
        s.Xp[f] = xp;
        s.El[f] =
            c.fSigE[f]
          - m.Sx[f]*(wf*c.FelX[o] + (1 - wf)*c.FelX[n])
          - m.Sy[f]*(wf*c.FelY[o] + (1 - wf)*c.FelY[n]);
        s.F[f] =
            c.phiFf[f]*s.Fu[f] + c.phiSf[f]*s.Fv[f]
          - c.rAf[f]*s.Xp[f] + c.cEf[f]*s.El[f];
        s.cellSum[o] += s.F[f];
        s.cellSum[n] -= s.F[f];
    }
    for (std::size_t bf = 0; bf < nB; ++bf)
    {
        const int P = m.bCell[bf];
        if (!m.bOpen[bf])
        {
            s.bF[bf] = c.bWallFlux[bf];
            s.cellSum[P] += s.bF[bf];
            continue;
        }
        const double* x = s.x.data();
        s.bFu[bf] = m.bSx[bf]*x[idx(P, 0)] + m.bSy[bf]*x[idx(P, 1)];
        s.bFv[bf] = m.bSx[bf]*x[idx(P, 2)] + m.bSy[bf]*x[idx(P, 3)];
        s.bXp[bf] =
            m.bMagS[bf]*(c.pgIC[bf]*p[P] + c.pgBC[bf])
          - m.bSx[bf]*gx[P] - m.bSy[bf]*gy[P];
        s.bEl[bf] = c.bFSigE[bf] - m.bSx[bf]*c.FelX[P] - m.bSy[bf]*c.FelY[P];
        s.bF[bf] =
            c.bPhiF[bf]*s.bFu[bf] + c.bPhiS[bf]*s.bFv[bf]
          - c.bRAf[bf]*s.bXp[bf] + c.bCE[bf]*s.bEl[bf];
        s.cellSum[P] += s.bF[bf];
    }
}

} // End namespace blockCoupled

#endif
