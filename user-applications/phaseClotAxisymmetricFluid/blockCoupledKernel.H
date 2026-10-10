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
        team 07 F5), the numeric factorisation every step; S19a (design
        2026-10-04-S19a-umfpack-row-equilibration.md): UMFPACK factors and
        solves A itself (its CSC, UMFPACK_A), so it equilibrates A's rows;
      - k = nNonOrthogonalCorrectors back-substitutions with the same LU, the
        explicit non-orthogonal flux correction updated from the new p between
        them; the final F_f carries the correction the LAST back-substitution
        was solved with (red team 07 F4), so linRes and the cell sums of F_f
        are taken on that final system;
      - R3 (section 12), -div(w_f nu3 (grad Vs + grad Vs^T)), fully implicit:
        the compact parts (2 nu3 normal, nu3 tangential) and the transpose
        cross terms Vsx <-> Vsy from face-interpolated central (least-squares,
        internal neighbours only) tangential derivatives, in the same block;
      - S19b (research/designs/2026-10-04-S19b-lumen-elimination.md rev. 2,
        APPROVED by Alireza 2026-10-04, section 2): under eliminateLumen a
        lumen cell's Vs is not an unknown (Vs = U there). A Layout numbers
        the unknowns, skipping Vs in lumen cells; every coefficient on a
        lumen Vs lands on that cell's U, same component; the identity rows
        are not emitted; b, the non-orthogonal rhs and p* go through the
        Layout, and x is expanded back to 5 per cell (lumen Vs := U) before
        the fluxes. Without it (identityRows) the system is S19a's, byte for
        byte (K14).

      - S22 (research/designs/2026-10-07-S22-pardiso-block-solver.md rev. 2,
        APPROVED by Alireza 2026-10-07, section 2): a second LU, PardisoLU,
        MKL PARDISO on the CSR of A itself (analysis on a pattern change by
        UmfpackLU's test, numeric factorisation every step, one PARDISO
        back-substitution each). MKL is never linked: PardisoLU dlopen's
        libmkl_rt RTLD_LOCAL and takes its entry points through dlsym, typed
        by the prototypes declared below (LP64, MKL_INT = int), so a UMFPACK
        run never has MKL in its process. solveStep is one template over the
        LU; each LU fills its own Step statistics (stats, solveStats), and
        the UMFPACK instantiation makes P2's calls in P2's order.

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
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <limits>
#include <thread>
#include <dlfcn.h>

// S22 section 2.3: the MKL entry points PardisoLU takes through dlsym, as
// declared by mkl_pardiso.h and mkl_service.h for LP64 (MKL_INT = int;
// _MKL_DSS_HANDLE_t = void*). Declared, never defined or called by name: no
// object here references an MKL symbol, so nothing links MKL. The lower-case
// names of the design (mkl_set_num_threads, ...) are mkl_service.h macros for
// these C symbols. A compile that includes the two MKL headers beside this
// file (V22-2) turns any mismatch into a conflicting-declaration error.
extern "C"
{
    void pardiso
    (
        void* pt, const int* maxfct, const int* mnum, const int* mtype,
        const int* phase, const int* n, const void* a, const int* ia,
        const int* ja, int* perm, const int* nrhs, int* iparm,
        const int* msglvl, void* b, void* x, int* error
    );
    void pardisoinit(void* pt, const int* mtype, int* iparm);
    void MKL_Set_Num_Threads(int nth);
    void MKL_Set_Dynamic(int bool_MKL_DYNAMIC);
    int MKL_Get_Max_Threads(void);
    void MKL_Get_Version_String(char* buffer, int len);
    long long int MKL_Peak_Mem_Usage(int reset);
    int MKL_CBWR_Set(int);
    int MKL_CBWR_Get(int);
    int MKL_CBWR_Get_Auto_Branch(void);
}

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

    // S19b (section 2.6): true = blockCoupledLumen eliminate (lumen Vs is
    // not an unknown), false = identityRows (S19a's system exactly). The
    // solver sets it from its switch; here false keeps S19a's behaviour.
    bool eliminateLumen = false;
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

//- S19b (section 2.1): the unknown numbering. The assembly writes every
//  coefficient at its (cell, var) number idx(P, k); toCsr carries row i to
//  row[i] (-1: not emitted) and column i to col[i]. Under eliminateLumen a
//  lumen cell's Vs rows are not emitted and its Vs columns are its U's, the
//  same component; otherwise both maps are the identity and n = nVar*N.
struct Layout
{
    bool eliminate = false;
    int nFull = 0;          // nVar*nCells
    int n = 0;              // the unknowns
    int nEliminated = 0;    // lumen cells whose Vs is not an unknown
    std::vector<int> row, col;
};

inline void makeLayout(const Coeffs& c, const int N, Layout& lay)
{
    lay.eliminate = c.eliminateLumen;
    lay.nFull = nVar*N;
    lay.n = 0;
    lay.nEliminated = 0;
    lay.row.assign(lay.nFull, -1);
    lay.col.assign(lay.nFull, -1);
    for (int P = 0; P < N; ++P)
    {
        const bool lumenVsEliminated = lay.eliminate && !c.isSkel[P];
        for (int k = 0; k < nVar; ++k)
        {
            const int i = idx(P, k);
            if (lumenVsEliminated && (k == 2 || k == 3))
            {
                // Vs = U in a lumen cell: a reference to Vs_k is one to U_k.
                lay.col[i] = lay.col[idx(P, k - 2)];
            }
            else
            {
                lay.row[i] = lay.n;
                lay.col[i] = lay.n;
                ++lay.n;
            }
        }
        if (lumenVsEliminated)
        {
            ++lay.nEliminated;
        }
    }
}

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
    //  lumen mask and the R3 switch only. S19b: rows and columns through the
    //  Layout (identity maps: S19a's arrays exactly); a coefficient on a
    //  lumen Vs merges with the one on that cell's U.
    void toCsr(Csr& A, const Layout& lay) const
    {
        const int nRows = int(rows.size());
        const int n = lay.n;
        A.n = n;
        A.p.assign(n + 1, 0);
        A.i.clear();
        A.x.clear();
        std::vector<std::pair<int, double>> r;
        for (int kFull = 0; kFull < nRows; ++kFull)
        {
            const int k = lay.row[kFull];
            if (k < 0)
            {
                continue;
            }
            r = rows[kFull];
            for (std::size_t j = 0; j < r.size(); ++j)
            {
                r[j].first = lay.col[r[j].first];
            }
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
//  number of lumen cells carrying identity rows (0 under eliminateLumen).
//  S19b: `lay` is built here from c (makeLayout); A and b are in its
//  numbering.
inline int assemble
(
    const Mesh& m, const Coeffs& c, const std::vector<GradRow>& G,
    const std::vector<LsqRow>& L, Csr& A, std::vector<double>& b,
    Layout& lay
)
{
    const int N = m.nCells;
    makeLayout(c, N, lay);
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
            else if (!lay.eliminate)
            {
                // THE LUMEN IDENTITY ROW, Vs - U = 0, replacing the shipped
                // post-solve overwrite Vs = isSkel Vs + (1 - isSkel) U.
                // S19b: not emitted under eliminateLumen, where the U row's
                // -KV above lands on its own diagonal (dU + KV - KV).
                T.add(rS, rS, 1.0);
                T.add(rS, idx(P, k), -1.0);
            }
        }
        if (!c.isSkel[P] && !lay.eliminate)
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

    T.toCsr(A, lay);
    // S19b section 2.3: b through the Layout (a lumen Vs row's b is 0 and
    // is not emitted).
    std::vector<double> bFull;
    bFull.swap(b);
    b.assign(lay.n, 0.0);
    for (int i = 0; i < lay.nFull; ++i)
    {
        if (lay.row[i] >= 0)
        {
            b[lay.row[i]] = bFull[i];
        }
    }
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

//- S19a (research/designs/2026-10-04-S19a-umfpack-row-equilibration.md
//  section 2.1): the compressed sparse COLUMNS of the CSR matrix A -- A
//  itself, not A^T -- in O(nnz). Column j lists its rows in ascending order.
inline void cscOfCsr
(
    const Csr& A, std::vector<int>& Cp, std::vector<int>& Ci,
    std::vector<double>& Cx
)
{
    const int nz = A.p[A.n];
    Cp.assign(A.n + 1, 0);
    Ci.resize(nz);
    Cx.resize(nz);
    for (int q = 0; q < nz; ++q)
    {
        ++Cp[A.i[q] + 1];
    }
    for (int j = 0; j < A.n; ++j)
    {
        Cp[j + 1] += Cp[j];
    }
    std::vector<int> next(Cp.begin(), Cp.end() - 1);
    for (int r = 0; r < A.n; ++r)
    {
        for (int q = A.p[r]; q < A.p[r + 1]; ++q)
        {
            const int at = next[A.i[q]]++;
            Ci[at] = r;
            Cx[at] = A.x[q];
        }
    }
}

struct Step;

class UmfpackLU
{
    void* symbolic_;
    void* numeric_;
    int n_;
    std::vector<int> Ap_, Ai_;
    // S19a: A in CSC, rebuilt from the CSR once per factor(); its pattern
    // is the one the kept symbolic factorisation was built on.
    std::vector<int> Cp_, Ci_;
    std::vector<double> Cx_;
    long symbolicCount_;
    double rcond_;
    double control_[UMFPACK_CONTROL];
    double info_[UMFPACK_INFO];
    // S19a: Info as the last symbolic factorisation left it (its estimates
    // price a failure even when the symbolic is kept).
    double symInfo_[UMFPACK_INFO];

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
        for (int k = 0; k < UMFPACK_INFO; ++k)
        {
            info_[k] = -1;
            symInfo_[k] = -1;
        }
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
        Cp_.clear();
        Ci_.clear();
        Cx_.clear();
    }

    long symbolicCount() const { return symbolicCount_; }
    double rcond() const { return rcond_; }

    //- S22 section 2.2: the status of a good factor() and backSolve().
    static const int statusOk = UMFPACK_OK;

    //- S22 section 2.2: UMFPACK's statistics into the step after factor()
    //  (P2's Info reads, verbatim; defined after Step).
    void stats(Step& s) const;

    //- S22 section 2.2: UMFPACK reports no refinement steps and no MKL
    //  memory; the step keeps -1 for both.
    void solveStats(Step&) const
    {}

    //- UMFPACK's Info after the last call (after factor(): the numeric
    //  factorisation's statistics; -1 where UMFPACK reported none).
    const double* info() const { return info_; }

    //- Info as the last symbolic factorisation left it.
    const double* symbolicInfo() const { return symInfo_; }

    //- The numeric factorisation's row scale factors Rs and do_recip
    //  (umfpack_di_get_numeric): row r of the factored matrix was multiplied
    //  by Rs[r] when doRecip is 1, divided by it when 0 (S19a K12).
    int rowScale(std::vector<double>& Rs, int& doRecip) const
    {
        doRecip = -1;
        Rs.assign(n_ > 0 ? n_ : 0, 0.0);
        if (!numeric_)
        {
            return UMFPACK_ERROR_invalid_Numeric_object;
        }
        return umfpack_di_get_numeric
        (
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, &doRecip, Rs.data(), numeric_
        );
    }

    //- S19a: the CSR rows of A are transposed ONCE per call into the CSC of
    //  A itself, and UMFPACK factors A, so its default SCALE_SUM divides each
    //  ROW of A by its abs-sum (design section 2.1; S19 handed the CSR over
    //  as A^T and so equilibrated A's columns only). The pattern test still
    //  compares the CSR pattern; the CSC pattern is kept with it. Returns the
    //  first status that is not UMFPACK_OK (the singular-matrix warning
    //  included), with `stage` 1 symbolic, 2 numeric.
    int factor(const Csr& A, double& tSym, double& tNum, int& stage)
    {
        tSym = 0;
        tNum = 0;
        stage = 0;
        const bool samePattern = (symbolic_ != nullptr && A.n == n_ && A.p == Ap_ && A.i == Ai_);
        cscOfCsr(A, Cp_, Ci_, Cx_);
        if (!samePattern)
        {
            const double t0 = now();
            if (symbolic_) { umfpack_di_free_symbolic(&symbolic_); symbolic_ = nullptr; }
            const int st = umfpack_di_symbolic
            (
                A.n, A.n, Cp_.data(), Ci_.data(), Cx_.data(),
                &symbolic_, control_, info_
            );
            tSym = now() - t0;
            for (int k = 0; k < UMFPACK_INFO; ++k)
            {
                symInfo_[k] = info_[k];
            }
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
            Cp_.data(), Ci_.data(), Cx_.data(), symbolic_, &numeric_,
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

    //- S19a: solves A x = b (UMFPACK_A) on the CSC factor()'s last call built.
    int backSolve(const Csr& A, const std::vector<double>& b, std::vector<double>& x)
    {
        x.assign(A.n, 0.0);
        return umfpack_di_solve
        (
            UMFPACK_A, Cp_.data(), Ci_.data(), Cx_.data(),
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


//- S19a: the abs-sum of row k of the CSR, sum_j |A_kj|.
inline double rowAbsSum(const Csr& A, const int k)
{
    double a = 0;
    for (int j = A.p[k]; j < A.p[k + 1]; ++j)
    {
        a += std::fabs(A.x[j]);
    }
    return a;
}


//- S19a (design section 2.4): the smallest and largest row abs-sum of A.
inline void rowAbsSumRange(const Csr& A, double& lo, double& hi)
{
    lo = 0;
    hi = 0;
    for (int k = 0; k < A.n; ++k)
    {
        const double a = rowAbsSum(A, k);
        lo = k == 0 ? a : std::min(lo, a);
        hi = k == 0 ? a : std::max(hi, a);
    }
}


//- S19a (design section 2.5): ||R^-1 (A x - b)||_2 and ||R^-1 b||_2 with
//  R = diag(sum_j |A_kj|), on the same CSR as residualNorms. A zero row is
//  singular and refused by UMFPACK before this is reached; its weight is
//  guarded as linRes guards ||b|| = 0.
inline void scaledResidualNorms
(
    const Csr& A, const std::vector<double>& x, const std::vector<double>& b,
    double& rNorm, double& bNorm
)
{
    double r2 = 0, b2 = 0;
    for (int k = 0; k < A.n; ++k)
    {
        const double a = rowAbsSum(A, k);
        double s = -b[k];
        for (int j = A.p[k]; j < A.p[k + 1]; ++j)
        {
            s += A.x[j]*x[A.i[j]];
        }
        const double rs = a > 0 ? s/a : s;
        const double bs = a > 0 ? b[k]/a : b[k];
        r2 += rs*rs;
        b2 += bs*bs;
    }
    rNorm = std::sqrt(r2);
    bNorm = std::sqrt(b2);
}


// ---------------------------------------------------------------------------
//  The step
// ---------------------------------------------------------------------------

struct Step
{
    int status = 0;        // the LU's status (0 = OK, both), or -1000 setup
    int stage = 0;         // 1 symbolic, 2 numeric, 3 solve, 4 setup
    int nIdentity = 0;
    int nEliminated = 0;   // S19b: lumen cells whose Vs is not an unknown
    int nBack = 0;
    double tSym = 0, tNum = 0, tSolve = 0;
    double rNorm = 0, bNorm = 0, rcond = 0;
    // S19a (design sections 2.4, 2.5): the matrix and its factor. The
    // numeric statistics come from UMFPACK's Info after an OK factor (-1
    // otherwise); the estimates are the last symbolic factorisation's; GB
    // are Info's Units times Info[UMFPACK_SIZE_OF_UNIT] over 1e9.
    int n = 0, nnz = 0;
    double strategyUsed = -1, orderingUsed = -1;
    double lunz = -1, numericPeakGB = -1, noffdiag = -1;
    double lunzEstimate = -1, numericSizeEstimateGB = -1, peakMemoryEstimateGB = -1;
    double rowSumMin = 0, rowSumMax = 0;
    double rsNorm = 0, bsNorm = 0;  // ||R^-1(Ax - b)||_2, ||R^-1 b||_2
    // S22 section 2.7 (-1 under UMFPACK): pivots perturbed by phase 22
    // (iparm 14), the most refinement steps over the step's
    // back-substitutions (iparm 7), MKL's peak (mkl_peak_mem_usage, GB), and
    // PARDISO's memory iparm 15-17 (KB) for the failure message (section 2.8).
    int perturbedPivots = -1, irSteps = -1;
    double mklPeakGB = -1;
    double iparm15KB = -1, iparm16KB = -1, iparm17KB = -1;
    std::vector<double> x;
    // Face fluxes and their parts: F = phiFf Fu + phiSf Fv - rA Xp + cE El.
    std::vector<double> F, Fu, Fv, Xp, El;
    std::vector<double> bF, bFu, bFv, bXp, bEl;
    std::vector<double> cellSum;   // sum over the cell's faces of outward F
};


//- S22 section 2.2: P2's UMFPACK Info reads (solveStep, P2 L1176-1191),
//  moved here verbatim; called right after factor(), as they were.
inline void UmfpackLU::stats(Step& s) const
{
    s.rcond = rcond();
    {
        const double* si = symbolicInfo();
        const double unit = si[UMFPACK_SIZE_OF_UNIT];
        s.strategyUsed = si[UMFPACK_STRATEGY_USED];
        s.orderingUsed = si[UMFPACK_ORDERING_USED];
        s.lunzEstimate = si[UMFPACK_LNZ_ESTIMATE] + si[UMFPACK_UNZ_ESTIMATE];
        s.numericSizeEstimateGB = si[UMFPACK_NUMERIC_SIZE_ESTIMATE]*unit/1e9;
        s.peakMemoryEstimateGB = si[UMFPACK_PEAK_MEMORY_ESTIMATE]*unit/1e9;
        if (s.status == UMFPACK_OK)
        {
            const double* ni = info();
            s.lunz = ni[UMFPACK_LNZ] + ni[UMFPACK_UNZ];
            s.numericPeakGB = ni[UMFPACK_PEAK_MEMORY]*unit/1e9;
            s.noffdiag = ni[UMFPACK_NOFF_DIAG];
        }
    }
}


// ---------------------------------------------------------------------------
//  S22: MKL PARDISO, loaded at run time (design sections 2.1, 2.3, 2.5, 2.6)
// ---------------------------------------------------------------------------

//- S22 section 3.2: THE DEFAULT OF blockSolverReproducible (the key absent).
//  ONE constant. Rev. 2.1 (Alireza 2026-10-07, "Physical cores,
//  reproducible"; docket 20261007-0134 report 05: CNR at the 6 physical
//  cores bitwise identical in 7/7 repeats at 0.93x the default mode's
//  refactorisation): true.
static const bool blockSolverReproducibleDefault = true;

//- S22 section 2.3: dlopen'ed when PHASE_CLOT_MKL_RT is unset (MKL 2026.1's
//  soname, red team 03 F3), and the version MKL must report (section 4).
static const char* const mklRuntimeSoname = "libmkl_rt.so.3";
static const char* const mklPinnedVersion = "2026.1";

//- mkl_types.h and mkl_service.h values (the V22-2 compile asserts them).
static const int mklCbwrAuto = 2;          // MKL_CBWR_AUTO
static const int mklCbwrSuccess = 0;       // MKL_CBWR_SUCCESS
static const int mklCbwrBranch = 1;        // MKL_CBWR_BRANCH (mkl_cbwr_get)
static const int mklPeakMemEnable = 1;     // MKL_PEAK_MEM_ENABLE
static const int mklPeakMem = 2;           // MKL_PEAK_MEM

typedef decltype(&::pardiso) MklPardisoFn;
typedef decltype(&::pardisoinit) MklPardisoInitFn;
typedef decltype(&::MKL_Set_Num_Threads) MklSetNumThreadsFn;
typedef decltype(&::MKL_Set_Dynamic) MklSetDynamicFn;
typedef decltype(&::MKL_Get_Max_Threads) MklGetMaxThreadsFn;
typedef decltype(&::MKL_Get_Version_String) MklGetVersionStringFn;
typedef decltype(&::MKL_Peak_Mem_Usage) MklPeakMemUsageFn;
typedef decltype(&::MKL_CBWR_Set) MklCbwrSetFn;
typedef decltype(&::MKL_CBWR_Get) MklCbwrGetFn;
typedef decltype(&::MKL_CBWR_Get_Auto_Branch) MklCbwrGetAutoBranchFn;


//- S22 section 4: true when MKL's version string names the pinned version,
//  "Version 2026.1" followed by a non-digit or the end (2026.10 is not it).
inline bool mklVersionPinned(const std::string& v)
{
    const std::string key = std::string("Version ") + mklPinnedVersion;
    const std::size_t at = v.find(key);
    if (at == std::string::npos)
    {
        return false;
    }
    const std::size_t end = at + key.size();
    return end == v.size() || !std::isdigit(static_cast<unsigned char>(v[end]));
}


//- S22 section 2.5 (rev. 2.1): the physical cores, the distinct
//  (physical_package_id, core_id) pairs of the online CPUs under
//  <root>/devices/system/cpu (root "/sys"). False when any of it cannot be
//  read or parsed ("unreadable": the caller falls back).
inline bool readSmallFile(const std::string& path, std::string& text)
{
    text.clear();
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f)
    {
        return false;
    }
    char buf[4096];
    const std::size_t got = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[got] = '\0';
    text = buf;
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    {
        text.erase(text.size() - 1);
    }
    return !text.empty();
}

//- S25 section 12.2: the CPU's model and instruction-set flags, from the
//  first "model name" and "flags" (x86) or "Features" (arm) lines of path
//  (/proc/cpuinfo), each as ONE key=value token for PHASE_CLOT_S19_CONFIG:
//  whitespace runs become '_' in the model and ',' in the flags. "none"
//  where the file or the line is absent (not Linux). MKL's AUTO branch
//  follows these flags, so B76-OFF compares two sessions only when they
//  agree (design section 12.1).
inline std::string cpuInfoToken(const std::string& v, const char sep)
{
    std::string out;
    bool gap = false;
    for (const char ch : v)
    {
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            gap = true;
            continue;
        }
        if (gap && !out.empty())
        {
            out += sep;
        }
        gap = false;
        out += ch;
    }
    return out.empty() ? std::string("none") : out;
}

inline void cpuIdentity
(
    std::string& model, std::string& flags,
    const std::string& path = "/proc/cpuinfo"
)
{
    model = "none";
    flags = "none";
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f)
    {
        return;
    }
    bool haveModel = false, haveFlags = false;
    std::string line;
    int ch = 0;
    while (!(haveModel && haveFlags))
    {
        line.clear();
        while ((ch = std::fgetc(f)) != EOF && ch != '\n')
        {
            line += char(ch);
        }
        const std::size_t colon = line.find(':');
        if (colon != std::string::npos)
        {
            const std::string key = cpuInfoToken(line.substr(0, colon), ' ');
            const std::string value = line.substr(colon + 1);
            if (!haveModel && key == "model name")
            {
                model = cpuInfoToken(value, '_');
                haveModel = true;
            }
            else if (!haveFlags && (key == "flags" || key == "Features"))
            {
                flags = cpuInfoToken(value, ',');
                haveFlags = true;
            }
        }
        if (ch == EOF)
        {
            break;
        }
    }
    std::fclose(f);
}

inline bool parseNonNegative(const std::string& s, long& v)
{
    if (s.empty() || !std::isdigit(static_cast<unsigned char>(s[0])))
    {
        return false;
    }
    char* end = nullptr;
    v = std::strtol(s.c_str(), &end, 10);
    return *end == '\0';
}

inline bool physicalCores(const std::string& root, int& n)
{
    n = 0;
    const std::string cpu = root + "/devices/system/cpu/";
    std::string online;
    if (!readSmallFile(cpu + "online", online))
    {
        return false;
    }
    std::vector<std::pair<long, long>> pairs;
    std::size_t at = 0;
    while (at <= online.size())
    {
        const std::size_t comma = std::min(online.find(',', at), online.size());
        const std::string range = online.substr(at, comma - at);
        const std::size_t dash = range.find('-');
        long lo = 0, hi = 0;
        if
        (
            !parseNonNegative(range.substr(0, dash), lo)
         || !parseNonNegative(dash == std::string::npos ? range.substr(0, dash) : range.substr(dash + 1), hi)
         || hi < lo
        )
        {
            return false;
        }
        for (long c = lo; c <= hi; ++c)
        {
            const std::string topo = cpu + "cpu" + std::to_string(c) + "/topology/";
            std::string pkgS, coreS;
            long pkg = 0, core = 0;
            if
            (
                !readSmallFile(topo + "physical_package_id", pkgS)
             || !readSmallFile(topo + "core_id", coreS)
             || !parseNonNegative(pkgS, pkg)
             || !parseNonNegative(coreS, core)
            )
            {
                return false;
            }
            pairs.push_back(std::make_pair(pkg, core));
        }
        at = comma + 1;
    }
    std::sort(pairs.begin(), pairs.end());
    n = int(std::unique(pairs.begin(), pairs.end()) - pairs.begin());
    return n > 0;
}


//- S22 section 2.5 (rev. 2.1): MKL's thread count under pardiso.
//  blockSolverThreads > 0 is used as given; 0 is MKL_NUM_THREADS when set (a
//  positive integer, or refused), else the physical cores of sysfsRoot
//  (physicalCores), else, only when sysfs is unreadable,
//  std::thread::hardware_concurrency() (0, unknown: refused). source names
//  the one used (blockSolverThreads, MKL_NUM_THREADS, physicalCores,
//  hardwareConcurrency).
inline bool resolveThreads
(
    const int requested, int& n, std::string& source, std::string& err,
    const std::string& sysfsRoot = "/sys"
)
{
    n = 0;
    source = "none";
    if (requested < 0)
    {
        err = "blockSolverThreads " + std::to_string(requested)
            + " is negative: use 0 (MKL_NUM_THREADS, else the physical"
              " cores) or a positive count (S22 section 4).";
        return false;
    }
    if (requested > 0)
    {
        n = requested;
        source = "blockSolverThreads";
        return true;
    }
    const char* env = std::getenv("MKL_NUM_THREADS");
    if (env)
    {
        char* end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end == env || *end != '\0' || v < 1 || v > std::numeric_limits<int>::max())
        {
            err = std::string("blockSolverThreads 0 reads MKL_NUM_THREADS, which is \"")
                + env + "\", not a positive integer (S22 section 2.5).";
            return false;
        }
        n = int(v);
        source = "MKL_NUM_THREADS";
        return true;
    }
    if (physicalCores(sysfsRoot, n))
    {
        source = "physicalCores";
        return true;
    }
    n = int(std::thread::hardware_concurrency());
    source = "hardwareConcurrency";
    if (n < 1)
    {
        err = "blockSolverThreads 0 with MKL_NUM_THREADS unset and "
            + sysfsRoot + "/devices/system/cpu unreadable: "
              "std::thread::hardware_concurrency() reports no count; set "
              "blockSolverThreads (S22 section 2.5).";
        return false;
    }
    return true;
}


//- S22 section 2.5: blockSolverThreads n > 0 under umfpack sets OpenBLAS's
//  thread count through dlsym(RTLD_DEFAULT); refused if the process has no
//  openblas_set_num_threads (section 4).
inline bool setOpenblasThreads(const int n, std::string& err)
{
    dlerror();
    void* f = dlsym(RTLD_DEFAULT, "openblas_set_num_threads");
    if (!f)
    {
        const char* e = dlerror();
        err = "blockSolverThreads " + std::to_string(n)
            + " under blockLinearSolver umfpack: openblas_set_num_threads is"
              " not in the process (dlsym(RTLD_DEFAULT): "
            + std::string(e ? e : "no message")
            + "), so UMFPACK's BLAS thread count cannot be set (S22 section 4).";
        return false;
    }
    reinterpret_cast<void (*)(int)>(f)(n);
    return true;
}


//- S22 section 2.1: MKL PARDISO (mtype 11, real unsymmetric) on the CSR of
//  A. Phase 11 (analysis) when the CSR pattern differs from the kept one,
//  by UmfpackLU's test; 22 (numeric) every factor(); 33 per backSolve(); -1
//  (release) before a new analysis and in the destructor. iparm as
//  benchmarked (docket 20261007-0134 benchkit/bench_pardiso.cpp): pardisoinit
//  for mtype 11, then (1-based numbers, as the design) 8 = 2, 10 = 13,
//  11 = 1, 13 = 1, 18 = -1, 35 = 1, 27 = 1 on the first analysis only, and
//  34 = the thread count under CNR. The arrays are 0-based (iparm 8 is
//  iparm_[7]).
class PardisoLU
{
public:

    //- factor() and backSolve(): PARDISO's error (0 = OK), or
    //  statusPerturbed when phase 22 perturbed a pivot (section 2.8; positive,
    //  as UMFPACK's singular-matrix warning).
    static const int statusOk = 0;
    static const int statusPerturbed = 1;

private:

    void* lib_;
    MklPardisoFn pardiso_;
    MklPardisoInitFn pardisoInit_;
    MklSetNumThreadsFn setNumThreads_;
    MklSetDynamicFn setDynamic_;
    MklGetMaxThreadsFn getMaxThreads_;
    MklGetVersionStringFn getVersionString_;
    MklPeakMemUsageFn peakMemUsage_;
    MklCbwrSetFn cbwrSet_;
    MklCbwrGetFn cbwrGet_;
    MklCbwrGetAutoBranchFn cbwrGetAutoBranch_;
    std::string path_, library_, version_;
    int threads_;
    int mklMaxThreads_;
    int cbwrBranch_;
    int cbwrAutoBranch_;
    void* pt_[64];
    int iparm_[64];
    int iparmIn_[64];
    bool live_;       // pt_ holds an analysis; phase -1 releases it
    bool checked_;    // the matrix checker (iparm 27) has run
    int n_;
    std::vector<int> Ap_, Ai_;
    std::vector<double> b_;
    long symbolicCount_;
    int failedPhase_;
    int irMax_;

    PardisoLU(const PardisoLU&);
    void operator=(const PardisoLU&);

    static double now()
    {
        return std::chrono::duration<double>
        (
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    template<class Fn>
    bool symbol(const char* name, Fn& fn, std::string& err)
    {
        dlerror();
        void* p = dlsym(lib_, name);
        if (!p)
        {
            const char* e = dlerror();
            err = "dlsym(\"" + std::string(name) + "\") in " + path_
                + " failed: " + std::string(e ? e : "no message")
                + ". Set blockLinearSolver umfpack or install mkl==2026.1.0"
                  " (S22 sections 2.3, 4).";
            return false;
        }
        fn = reinterpret_cast<Fn>(p);
        return true;
    }

    //- One PARDISO call on the kept pattern; dummies where the phase reads
    //  no values (the oneMKL examples' form).
    int phase(const int ph, const double* a, double* b, double* x)
    {
        const int mtype = 11, maxfct = 1, mnum = 1, nrhs = 1, msglvl = 0;
        int error = 0;
        double ddum = 0;
        pardiso_
        (
            pt_, &maxfct, &mnum, &mtype, &ph, &n_, a ? a : &ddum,
            Ap_.data(), Ai_.data(), nullptr, &nrhs, iparm_, &msglvl,
            b ? b : &ddum, x ? x : &ddum, &error
        );
        return error;
    }

public:

    PardisoLU()
    :
        lib_(nullptr), pardiso_(nullptr), pardisoInit_(nullptr),
        setNumThreads_(nullptr), setDynamic_(nullptr),
        getMaxThreads_(nullptr), getVersionString_(nullptr),
        peakMemUsage_(nullptr), cbwrSet_(nullptr), cbwrGet_(nullptr),
        cbwrGetAutoBranch_(nullptr), threads_(0),
        mklMaxThreads_(0), cbwrBranch_(-1), cbwrAutoBranch_(-1), live_(false), checked_(false), n_(-1), symbolicCount_(0),
        failedPhase_(0), irMax_(-1)
    {
        std::memset(pt_, 0, sizeof pt_);
        for (int k = 0; k < 64; ++k)
        {
            iparm_[k] = 0;
            iparmIn_[k] = 0;
        }
    }

    ~PardisoLU()
    {
        if (live_)
        {
            phase(-1, nullptr, nullptr, nullptr);
        }
    }

    //- S22 sections 2.3, 2.5, 2.6: dlopen(path, RTLD_NOW | RTLD_LOCAL), the
    //  ten entry points, CNR mode (mkl_cbwr_set(MKL_CBWR_AUTO), refused
    //  unless MKL_CBWR_SUCCESS) before any other MKL call when reproducible, the pinned
    //  version, mkl_set_dynamic(0) and mkl_set_num_threads(threads) once, the
    //  peak-memory counter, then pardisoinit and the benchmarked iparm.
    //  False with err set on any refusal of section 4.
    bool load
    (
        const std::string& path, const int threads, const bool reproducible,
        std::string& err
    )
    {
        path_ = path;
        if (path.empty())
        {
            err = "PHASE_CLOT_MKL_RT is set but empty: it must name libmkl_rt"
                  " (S22 section 2.3).";
            return false;
        }
        dlerror();
        lib_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib_)
        {
            const char* e = dlerror();
            err = "dlopen(\"" + path + "\", RTLD_NOW | RTLD_LOCAL) failed: "
                + std::string(e ? e : "no message")
                + ". Set blockLinearSolver umfpack or install mkl==2026.1.0"
                  " (S22 sections 2.3, 4).";
            return false;
        }
        if
        (
            !symbol("pardiso", pardiso_, err)
         || !symbol("pardisoinit", pardisoInit_, err)
         || !symbol("MKL_Set_Num_Threads", setNumThreads_, err)
         || !symbol("MKL_Set_Dynamic", setDynamic_, err)
         || !symbol("MKL_Get_Max_Threads", getMaxThreads_, err)
         || !symbol("MKL_Get_Version_String", getVersionString_, err)
         || !symbol("MKL_Peak_Mem_Usage", peakMemUsage_, err)
         || !symbol("MKL_CBWR_Set", cbwrSet_, err)
         || !symbol("MKL_CBWR_Get", cbwrGet_, err)
         || !symbol("MKL_CBWR_Get_Auto_Branch", cbwrGetAutoBranch_, err)
        )
        {
            return false;
        }
        if (reproducible)
        {
            const int r = cbwrSet_(mklCbwrAuto);
            if (r != mklCbwrSuccess)
            {
                err = "blockSolverReproducible true: mkl_cbwr_set(MKL_CBWR_AUTO)"
                      " returned " + std::to_string(r)
                    + ", not MKL_CBWR_SUCCESS (S22 section 2.6).";
                return false;
            }
        }
        char buf[256];
        std::memset(buf, 0, sizeof buf);
        getVersionString_(buf, int(sizeof buf) - 1);
        version_ = buf;
        if (!mklVersionPinned(version_))
        {
            err = "MKL at " + path + " reports \"" + version_ + "\", not the"
                  " pinned " + std::string(mklPinnedVersion)
                + ": PARDISO runs only on the MKL the benchmark measured (S22"
                  " section 4). Install mkl==2026.1.0 or set"
                  " blockLinearSolver umfpack.";
            return false;
        }
        setDynamic_(0);
        setNumThreads_(threads);
        threads_ = threads;
        mklMaxThreads_ = getMaxThreads_();
        // S25 section 12.2: the CNR setting in force after the set above
        // (mkl_cbwr_get(MKL_CBWR_BRANCH)) and the code path AUTO selects on
        // this CPU (mkl_cbwr_get_auto_branch), read once for
        // PHASE_CLOT_S19_CONFIG: B76-OFF is comparable only between
        // sessions on the same branch.
        cbwrBranch_ = cbwrGet_(mklCbwrBranch);
        cbwrAutoBranch_ = cbwrGetAutoBranch_();
        peakMemUsage_(mklPeakMemEnable);
        Dl_info di;
        library_ =
            (dladdr(reinterpret_cast<void*>(pardiso_), &di) && di.dli_fname)
          ? std::string(di.dli_fname) : path;

        const int mtype = 11;
        pardisoInit_(pt_, &mtype, iparmIn_);
        iparmIn_[7] = 2;       // iparm 8: at most 2 refinement steps
        iparmIn_[9] = 13;      // iparm 10: pivot perturbation 1e-13
        iparmIn_[10] = 1;      // iparm 11: scaling
        iparmIn_[12] = 1;      // iparm 13: matching
        iparmIn_[17] = -1;     // iparm 18: report the factor nnz
        iparmIn_[34] = 1;      // iparm 35: zero-based ia, ja
        if (reproducible)
        {
            iparmIn_[33] = threads;    // iparm 34: CNR at the thread count n
        }
        std::memcpy(iparm_, iparmIn_, sizeof iparm_);
        return true;
    }

    long symbolicCount() const { return symbolicCount_; }
    int threads() const { return threads_; }
    int mklMaxThreads() const { return mklMaxThreads_; }
    int cbwrBranch() const { return cbwrBranch_; }
    int cbwrAutoBranch() const { return cbwrAutoBranch_; }
    const std::string& version() const { return version_; }
    const std::string& library() const { return library_; }

    //- The PARDISO phase whose error the last failing call returned (11,
    //  22, 33 or -1); 0 when none failed.
    int failedPhase() const { return failedPhase_; }

    //- Phase -1 when a new pattern needs a new analysis, then 11, on the
    //  CSR of A; 22 every call. Returns statusOk, a PARDISO error (stage 1
    //  analysis or its release, 2 numeric) or statusPerturbed (stage 2).
    int factor(const Csr& A, double& tSym, double& tNum, int& stage)
    {
        tSym = 0;
        tNum = 0;
        stage = 0;
        irMax_ = -1;
        const bool keptPattern = (live_ && A.n == n_ && A.p == Ap_ && A.i == Ai_);
        if (!keptPattern)
        {
            const double t0 = now();
            if (live_)
            {
                const int er = phase(-1, nullptr, nullptr, nullptr);
                live_ = false;
                std::memset(pt_, 0, sizeof pt_);
                if (er != 0)
                {
                    stage = 1;
                    failedPhase_ = -1;
                    return er;
                }
            }
            std::memcpy(iparm_, iparmIn_, sizeof iparm_);
            if (!checked_)
            {
                iparm_[26] = 1;   // iparm 27: the matrix checker, first analysis
            }
            n_ = A.n;
            Ap_ = A.p;
            Ai_ = A.i;
            live_ = true;
            const int ea = phase(11, A.x.data(), nullptr, nullptr);
            checked_ = true;
            tSym = now() - t0;
            if (ea != 0)
            {
                n_ = -1;
                stage = 1;
                failedPhase_ = 11;
                return ea;
            }
            ++symbolicCount_;
        }
        const double t1 = now();
        const int en = phase(22, A.x.data(), nullptr, nullptr);
        tNum = now() - t1;
        if (en != 0)
        {
            stage = 2;
            failedPhase_ = 22;
            return en;
        }
        if (iparm_[13] > 0)
        {
            stage = 2;
            return statusPerturbed;
        }
        return statusOk;
    }

    //- Phase 33: A x = b on factor()'s factorisation of the same A.
    int backSolve(const Csr& A, const std::vector<double>& b, std::vector<double>& x)
    {
        x.assign(A.n, 0.0);
        b_ = b;
        const int es = phase(33, A.x.data(), b_.data(), x.data());
        if (es != 0)
        {
            failedPhase_ = 33;
        }
        irMax_ = std::max(irMax_, iparm_[6]);
        return es;
    }

    //- S22 section 2.7, after factor(): lunz = iparm 18, numericPeakGB =
    //  max(iparm 15, iparm 16 + iparm 17) KB / 1e6 and orderingUsed = iparm
    //  2 after an OK factor; rcond, strategyUsed and noffdiag -1 (UMFPACK
    //  only); perturbedPivots = iparm 14; iparm 15-17 for the message.
    void stats(Step& s) const
    {
        s.rcond = -1;
        s.strategyUsed = -1;
        s.noffdiag = -1;
        s.orderingUsed = iparm_[1];
        s.perturbedPivots = iparm_[13];
        s.iparm15KB = iparm_[14];
        s.iparm16KB = iparm_[15];
        s.iparm17KB = iparm_[16];
        if (s.status == statusOk)
        {
            s.lunz = iparm_[17];
            s.numericPeakGB =
                std::max(double(iparm_[14]), double(iparm_[15]) + double(iparm_[16]))/1e6;
        }
    }

    //- S22 section 2.7, after the back-substitutions: irSteps (the most
    //  over this step's phase-33 calls) and MKL's peak, mkl_peak_mem_usage
    //  (MKL_PEAK_MEM) bytes / 1e9.
    void solveStats(Step& s) const
    {
        s.irSteps = irMax_;
        s.mklPeakGB = double(peakMemUsage_(mklPeakMem))/1e9;
    }
};

//- One step: assemble, factor, k back-substitutions, fluxes. x0 holds the
//  old state (its p is p^n, the start of the explicit correction).
//  S22 section 2.2: one template over the LU (UmfpackLU, PardisoLU).
template<class LU>
inline void solveStep
(
    const Mesh& m, const Coeffs& c, const int nNonOrth,
    const std::vector<double>& x0, LU& lu, Step& s
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
    Layout lay;
    s.nIdentity = assemble(m, c, G, L, A, b0, lay);
    s.nEliminated = lay.nEliminated;
    s.n = A.n;
    s.nnz = A.p[A.n];
    rowAbsSumRange(A, s.rowSumMin, s.rowSumMax);
    s.status = lu.factor(A, s.tSym, s.tNum, s.stage);
    lu.stats(s);
    if (s.status != LU::statusOk)
    {
        return;
    }

    // S19b: xc is the solution in the Layout's numbering; s.x is expanded
    // from it below.
    std::vector<double> pStar(N), kc(m.own.size(), 0.0), rhs, xc;
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
                rhs[lay.row[idx(m.own[f], 4)]] -= e;
                rhs[lay.row[idx(m.nei[f], 4)]] += e;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int st = lu.backSolve(A, rhs, xc);
        s.tSolve += std::chrono::duration<double>
        (
            std::chrono::steady_clock::now() - t0
        ).count();
        if (st != LU::statusOk)
        {
            s.status = st;
            s.stage = 3;
            return;
        }
        for (int P = 0; P < N; ++P)
        {
            pStar[P] = xc[lay.col[idx(P, 4)]];
        }
    }
    lu.solveStats(s);
    residualNorms(A, xc, rhs, s.rNorm, s.bNorm);
    scaledResidualNorms(A, xc, rhs, s.rsNorm, s.bsNorm);

    // ---- S19b section 2.4: x back to 5 per cell, lumen Vs := U ----
    s.x.assign(lay.nFull, 0.0);
    for (int i = 0; i < lay.nFull; ++i)
    {
        s.x[i] = xc[lay.col[i]];
    }

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
