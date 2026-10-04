/*---------------------------------------------------------------------------*\
    S17 A (research/designs/2026-09-26-S17-frozen-release-implicit-elastic-
    startup.md, revision 4, section 2A): THE PENDING SHARE AS A FIBRE.

    Header-only, no OpenFOAM dependency, so the arithmetic the solver runs is
    the arithmetic scripts/s17_fibre_kernel_test.cpp compiles and tests (as
    damagePartitionKernel.H is for S16 C+).

    The pending (broken, not yet released) share of a cell's network carries
    tension only along its own axis n (docket 09 section 1):
        B_P   = (1/lf) I + (lf^2 - 1/lf) n n        uniaxial, det 1
        sigma = G dev(B_P) = G T dev(n n),  T = lf^2 - 1/lf   (per unit G)
    Every function below works per unit G.

      projectLS      P-LS (docket 12 section 1): a content-weighted blend of
                     fibres back to ONE fibre -- n the major eigenvector of
                     dev(B_mix), T = (3/2) lambda_max(dev B_mix), lf from
                     lf^2 - 1/lf = T; ties go to the cell's previous axis.
      stepGradient   F = exp(L dt), the step's deformation increment from the
                     same velocity gradient the be transport uses (09 s. 2).
      rotateStretch  the material-line rotation R n (R the exact rotation
                     taking n to F n/|F n|), then the non-increase of the
                     fibre stretch lf <- lf min(1, |F n| det(F)^(-1/3))
                     (source 59's mu^n = min(mu^(n-1), mu); 09 section 2).
      mixedStiffness the tangent number's stiffness per unit weight,
                     [(thI/S) stiff + (thP/S)(lf^2 - 1/lf)]/(1 - d)
                     (docket 10; the STRIPES bound, G13 and C's Gamma_t).
      splitDev       the deviatoric stress per unit weight,
                     [(thI/S) dev(be) + (thP/S) T dev(n n)]/(1 - d)
                     (design 2A "Stress"; the weight is max(ind, floor)(1 - d)).

    NOTHING HERE BOUNDS A VALUE. lf = min(l1(be), lc) at the event is the
    event value (fibres break at lc; 09 rule 4) and is formed by the caller;
    the non-increase of lf is source 59's law, not a clip.
\*---------------------------------------------------------------------------*/

#ifndef fibreReleaseKernel_H
#define fibreReleaseKernel_H

#include <algorithm>
#include <cmath>
#include <limits>

namespace fibreReleaseKernel
{

//- Symmetric 3x3 as components xx, xy, xz, yy, yz, zz -- OpenFOAM's
//  symmTensor order, so the solver copies components straight across.
enum { XX = 0, XY = 1, XZ = 2, YY = 3, YZ = 4, ZZ = 5 };


inline void toMatrix(const double s[6], double A[3][3])
{
    A[0][0] = s[XX]; A[0][1] = s[XY]; A[0][2] = s[XZ];
    A[1][0] = s[XY]; A[1][1] = s[YY]; A[1][2] = s[YZ];
    A[2][0] = s[XZ]; A[2][1] = s[YZ]; A[2][2] = s[ZZ];
}


inline void fromMatrix(const double A[3][3], double s[6])
{
    // symmetric part, so a rounding asymmetry in R B R^T is not kept
    s[XX] = A[0][0];
    s[XY] = 0.5*(A[0][1] + A[1][0]);
    s[XZ] = 0.5*(A[0][2] + A[2][0]);
    s[YY] = A[1][1];
    s[YZ] = 0.5*(A[1][2] + A[2][1]);
    s[ZZ] = A[2][2];
}


//- Cyclic Jacobi (Numerical Recipes' jacobi, with its underflow test after
//  four sweeps), to machine precision. w descending; V's COLUMNS are the
//  eigenvectors. Returns false only if 64 sweeps did not annihilate the
//  off-diagonal (never observed; the caller refuses rather than guesses).
inline bool eigSym(const double Ain[3][3], double w[3], double V[3][3])
{
    double a[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            a[i][j] = Ain[i][j];
            V[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    double d[3] = {a[0][0], a[1][1], a[2][2]};
    double b[3] = {d[0], d[1], d[2]};
    double z[3] = {0, 0, 0};
    bool converged = false;
    for (int sweep = 1; sweep <= 64; ++sweep)
    {
        const double sm =
            std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
        if (sm == 0.0)
        {
            converged = true;
            break;
        }
        const double tresh = sweep < 4 ? 0.2*sm/9.0 : 0.0;
        for (int p = 0; p < 2; ++p)
        {
            for (int q = p + 1; q < 3; ++q)
            {
                const double g = 100.0*std::abs(a[p][q]);
                if
                (
                    sweep > 4
                 && std::abs(d[p]) + g == std::abs(d[p])
                 && std::abs(d[q]) + g == std::abs(d[q])
                )
                {
                    a[p][q] = 0.0;
                }
                else if (std::abs(a[p][q]) > tresh)
                {
                    const double h = d[q] - d[p];
                    double t;
                    if (std::abs(h) + g == std::abs(h))
                    {
                        t = a[p][q]/h;
                    }
                    else
                    {
                        const double theta = 0.5*h/a[p][q];
                        t = 1.0/(std::abs(theta) + std::sqrt(1.0 + theta*theta));
                        if (theta < 0.0) { t = -t; }
                    }
                    const double c = 1.0/std::sqrt(1.0 + t*t);
                    const double s = t*c;
                    const double tau = s/(1.0 + c);
                    const double hh = t*a[p][q];
                    z[p] -= hh; z[q] += hh;
                    d[p] -= hh; d[q] += hh;
                    a[p][q] = 0.0;
                    for (int j = 0; j < 3; ++j)
                    {
                        if (j == p || j == q) { continue; }
                        // rotate the (p, j) and (q, j) off-diagonals
                        double& apj = (j < p) ? a[j][p] : a[p][j];
                        double& aqj = (j < q) ? a[j][q] : a[q][j];
                        const double g1 = apj, h1 = aqj;
                        apj = g1 - s*(h1 + g1*tau);
                        aqj = h1 + s*(g1 - h1*tau);
                    }
                    for (int j = 0; j < 3; ++j)
                    {
                        const double g1 = V[j][p], h1 = V[j][q];
                        V[j][p] = g1 - s*(h1 + g1*tau);
                        V[j][q] = h1 + s*(g1 - h1*tau);
                    }
                }
            }
        }
        for (int p = 0; p < 3; ++p)
        {
            b[p] += z[p];
            d[p] = b[p];
            z[p] = 0.0;
        }
    }
    // descending, vectors with them
    int idx[3] = {0, 1, 2};
    std::sort
    (
        idx, idx + 3,
        [&d](int i, int j) { return d[i] > d[j]; }
    );
    double Vs[3][3];
    for (int k = 0; k < 3; ++k)
    {
        w[k] = d[idx[k]];
        for (int i = 0; i < 3; ++i)
        {
            Vs[i][k] = V[i][idx[k]];
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            V[i][k] = Vs[i][k];
        }
    }
    return converged;
}


//- T/G of a fibre at stretch lf: lf^2 - 1/lf.
inline double fibreTension(const double lf)
{
    return lf*lf - 1.0/lf;
}


//- The fibre stretch lf >= 1 of tension T >= 0 (per unit G): the unique
//  root of lf^3 - T lf - 1 = 0 in [1, inf). Newton from ABOVE, at
//  sqrt(T) + 1 (where the cubic is 2T + 3 sqrt(T) >= 0): the cubic is
//  convex there, so the iterates decrease monotonically onto the root and
//  the loop stops when rounding stops the decrease. T <= 0 is the isotropic
//  fibre, lf = 1 (lambda_max of a traceless tensor is >= 0; a -1e-17 from
//  rounding is that value).
inline double fibreStretch(const double T)
{
    if (!(T > 0))
    {
        return 1.0;
    }
    double l = std::sqrt(T) + 1.0;
    for (int it = 0; it < 200; ++it)
    {
        const double f = (l*l - T)*l - 1.0;
        const double fp = 3.0*l*l - T;
        if (!(fp > 0))
        {
            break;
        }
        const double ln = l - f/fp;
        if (!(ln < l))
        {
            break;
        }
        l = ln;
    }
    return l;
}


//- B = (1/lf) I + (lf^2 - 1/lf) n n, n unit.
inline void fibre(const double n[3], const double lf, double s[6])
{
    const double iso = 1.0/lf;
    const double T = fibreTension(lf);
    s[XX] = iso + T*n[0]*n[0];
    s[XY] = T*n[0]*n[1];
    s[XZ] = T*n[0]*n[2];
    s[YY] = iso + T*n[1]*n[1];
    s[YZ] = T*n[1]*n[2];
    s[ZZ] = iso + T*n[2]*n[2];
}


//- dev(n n) = n n - I/3.
inline void devAxis(const double n[3], double s[6])
{
    const double third = 1.0/3.0;
    s[XX] = n[0]*n[0] - third;
    s[XY] = n[0]*n[1];
    s[XZ] = n[0]*n[2];
    s[YY] = n[1]*n[1] - third;
    s[YZ] = n[1]*n[2];
    s[ZZ] = n[2]*n[2] - third;
}


//- (G/2)(tr B - 3), per unit G/2: the stored energy of a blend of fibres is
//  the content-weighted sum of theirs (docket 12, exact identity).
inline double energy(const double s[6])
{
    return (s[XX] + s[YY] + s[ZZ]) - 3.0;
}


//- G-proj's rounding floor: tr(B) - 3 is a difference of numbers of
//  order 3, so the exact inequality psi_out <= psi_mix (1 + 1e-12) is read
//  to 16*3*eps = 48 eps = 24 ulp(3) (1.07e-14; ulp(3) = 2 eps). MEASURED (scripts/s17_fibre_kernel_test.cpp
//  K2b): near-isotropic blends exceed the exact form by up to 1.3e-15 and
//  never exceed it with this floor; P-b1 still fails (K4).
const double psiFloor = 16.0*3.0*std::numeric_limits<double>::epsilon();


struct Projection
{
    double n[3];      // the fibre axis (unit)
    double lf;        // its stretch
    double Tls;       // (3/2) lambda_max(dev B_mix), the least-squares T
    double psiMix;    // tr(B_mix) - 3
    double psiOut;    // tr(B_out) - 3
    bool tie;         // the top two eigenvalues of dev(B_mix) were equal
    bool ok;          // false: the eigen-solve did not converge (refuse)
};


//- P-LS. tieTol is relative to lambda_max(dev B_mix): 1e-12 (docket 12).
inline Projection projectLS
(
    const double Bmix[6],
    const double nPrev[3],
    const double tieTol = 1e-12
)
{
    Projection P;
    const double tr = Bmix[XX] + Bmix[YY] + Bmix[ZZ];
    const double third = tr/3.0;
    double D[3][3];
    toMatrix(Bmix, D);
    D[0][0] -= third; D[1][1] -= third; D[2][2] -= third;
    double w[3], V[3][3];
    P.ok = eigSym(D, w, V);
    const double e1 = w[0], e2 = w[1];
    P.tie = !(e1 - e2 > tieTol*std::abs(e1));
    double n[3] = {V[0][0], V[1][0], V[2][0]};
    if (P.tie)
    {
        // the unit projection of the previous axis onto the tied eigenspace
        const double a = nPrev[0]*V[0][0] + nPrev[1]*V[1][0] + nPrev[2]*V[2][0];
        const double b = nPrev[0]*V[0][1] + nPrev[1]*V[1][1] + nPrev[2]*V[2][1];
        const double m = std::sqrt(a*a + b*b);
        if (m > 0)
        {
            for (int i = 0; i < 3; ++i)
            {
                n[i] = (a*V[i][0] + b*V[i][1])/m;
            }
        }
    }
    const double nn = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    for (int i = 0; i < 3; ++i)
    {
        P.n[i] = n[i]/nn;
    }
    P.Tls = 1.5*e1;
    P.lf = fibreStretch(P.Tls);
    double Bout[6];
    fibre(P.n, P.lf, Bout);
    P.psiMix = energy(Bmix);
    P.psiOut = energy(Bout);
    return P;
}


//- G-proj: P-LS never raised the stored energy against the blend (12).
inline bool dissipative(const Projection& P)
{
    return !(P.psiOut > P.psiMix*(1.0 + 1e-12) + psiFloor);
}


//- F = exp(L dt), L_ij = d u_i/d x_j. Scaling and squaring with a Taylor
//  series to machine precision (09 used a 30-term series).
inline void stepGradient(const double L[3][3], const double dt, double F[3][3])
{
    double A[3][3];
    double nrm = 0;
    for (int i = 0; i < 3; ++i)
    {
        double row = 0;
        for (int j = 0; j < 3; ++j)
        {
            A[i][j] = L[i][j]*dt;
            row += std::abs(A[i][j]);
        }
        nrm = std::max(nrm, row);
    }
    int sq = 0;
    while (nrm > 0.5 && sq < 60)
    {
        nrm *= 0.5;
        ++sq;
    }
    const double scale = std::ldexp(1.0, -sq);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            A[i][j] *= scale;
        }
    }
    double term[3][3], sum[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            term[i][j] = (i == j) ? 1.0 : 0.0;
            sum[i][j] = term[i][j];
        }
    }
    for (int k = 1; k <= 30; ++k)
    {
        double nt[3][3];
        double tmax = 0;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                double acc = 0;
                for (int m = 0; m < 3; ++m)
                {
                    acc += term[i][m]*A[m][j];
                }
                nt[i][j] = acc/k;
                tmax = std::max(tmax, std::abs(nt[i][j]));
            }
        }
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                term[i][j] = nt[i][j];
                sum[i][j] += nt[i][j];
            }
        }
        if (tmax == 0)
        {
            break;
        }
    }
    for (int s = 0; s < sq; ++s)
    {
        double P2[3][3];
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                double acc = 0;
                for (int m = 0; m < 3; ++m)
                {
                    acc += sum[i][m]*sum[m][j];
                }
                P2[i][j] = acc;
            }
        }
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                sum[i][j] = P2[i][j];
            }
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            F[i][j] = sum[i][j];
        }
    }
}


inline double det3(const double F[3][3])
{
    return
        F[0][0]*(F[1][1]*F[2][2] - F[1][2]*F[2][1])
      - F[0][1]*(F[1][0]*F[2][2] - F[1][2]*F[2][0])
      + F[0][2]*(F[1][0]*F[2][1] - F[1][1]*F[2][0]);
}


struct Rotation
{
    double angle;     // the angle applied, rad
    double omega;     // the line's rotation rate |(I - n n) L n|, 1/s
    double Lnorm;     // |L| (Frobenius), 1/s
    double eigRel;    // max |eig(R B R^T) - eig(B)| / max |eig(B)|
    double mu;        // |F n| det(F)^(-1/3), the isochoric line stretch
    double lfIn;      // lf before
    double lfOut;     // lf after the non-increase
    bool ok;          // false: F n vanished or F is not orientation-keeping
};


//- The ML rotation and UNI, once per step (09 section 2; step order in 12:
//  after P-LS, before the release and the stress). n and lf are updated in
//  place; Bout is the new fibre tensor fibre(n', lf').
inline Rotation rotateStretch
(
    const double L[3][3],
    const double dt,
    double n[3],
    double& lf,
    double Bout[6]
)
{
    Rotation r;
    r.ok = true;
    r.lfIn = lf;
    double F[3][3];
    stepGradient(L, dt, F);
    double m[3];
    for (int i = 0; i < 3; ++i)
    {
        m[i] = F[i][0]*n[0] + F[i][1]*n[1] + F[i][2]*n[2];
    }
    const double mMag = std::sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
    const double J = det3(F);
    if (!(mMag > 0) || !(J > 0))
    {
        r.ok = false;
        r.angle = r.omega = r.Lnorm = r.eigRel = r.mu = 0;
        r.lfOut = lf;
        fibre(n, lf, Bout);
        return r;
    }
    const double np[3] = {m[0]/mMag, m[1]/mMag, m[2]/mMag};
    // R = I + K + K^2/(1 + c), k = n x n', c = n.n': the exact rotation
    // taking n to n' about their common normal (Rodrigues).
    const double k[3] =
    {
        n[1]*np[2] - n[2]*np[1],
        n[2]*np[0] - n[0]*np[2],
        n[0]*np[1] - n[1]*np[0]
    };
    const double c = n[0]*np[0] + n[1]*np[1] + n[2]*np[2];
    const double sMag = std::sqrt(k[0]*k[0] + k[1]*k[1] + k[2]*k[2]);
    r.angle = std::atan2(sMag, c);
    double K[3][3] =
    {
        {0, -k[2], k[1]},
        {k[2], 0, -k[0]},
        {-k[1], k[0], 0}
    };
    double R[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double K2 = 0;
            for (int q = 0; q < 3; ++q)
            {
                K2 += K[i][q]*K[q][j];
            }
            R[i][j] = ((i == j) ? 1.0 : 0.0) + K[i][j] + K2/(1.0 + c);
        }
    }
    // The rotation of B_P, measured (G-eig: across the rotation only).
    double Bs[6], B[3][3];
    fibre(n, lf, Bs);
    toMatrix(Bs, B);
    double RB[3][3], RBRt[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double acc = 0;
            for (int q = 0; q < 3; ++q)
            {
                acc += R[i][q]*B[q][j];
            }
            RB[i][j] = acc;
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double acc = 0;
            for (int q = 0; q < 3; ++q)
            {
                acc += RB[i][q]*R[j][q];
            }
            RBRt[i][j] = acc;
        }
    }
    double w0[3], w1[3], V0[3][3], V1[3][3];
    const bool ok0 = eigSym(B, w0, V0);
    double Bsym[6], Bm[3][3];
    fromMatrix(RBRt, Bsym);
    toMatrix(Bsym, Bm);
    const bool ok1 = eigSym(Bm, w1, V1);
    r.ok = ok0 && ok1;
    double scale = 0, dev = 0;
    for (int i = 0; i < 3; ++i)
    {
        scale = std::max(scale, std::abs(w0[i]));
        dev = std::max(dev, std::abs(w1[i] - w0[i]));
    }
    r.eigRel = scale > 0 ? dev/scale : dev;

    // The line's rotation rate on the INCOMING axis, |(I - n n) L n|, for
    // G-rot's ratio angle/(dt omega): 1 when the step applied the line's
    // own rotation once, 2 if applied twice, 0 if not at all.
    double Ln[3], Ln2 = 0, nLn = 0;
    r.Lnorm = 0;
    for (int i = 0; i < 3; ++i)
    {
        Ln[i] = L[i][0]*n[0] + L[i][1]*n[1] + L[i][2]*n[2];
        nLn += n[i]*Ln[i];
        for (int j = 0; j < 3; ++j)
        {
            r.Lnorm += L[i][j]*L[i][j];
        }
    }
    r.Lnorm = std::sqrt(r.Lnorm);
    for (int i = 0; i < 3; ++i)
    {
        const double t = Ln[i] - nLn*n[i];
        Ln2 += t*t;
    }
    r.omega = std::sqrt(Ln2);

    // The new axis is the rotated axis R n.
    double nr[3];
    for (int i = 0; i < 3; ++i)
    {
        nr[i] = R[i][0]*n[0] + R[i][1]*n[1] + R[i][2]*n[2];
    }
    const double nrMag = std::sqrt(nr[0]*nr[0] + nr[1]*nr[1] + nr[2]*nr[2]);
    for (int i = 0; i < 3; ++i)
    {
        n[i] = nr[i]/nrMag;
    }

    // UNI: the isochoric stretch of the material line; lf never increases.
    r.mu = mMag*std::pow(J, -1.0/3.0);
    lf = lf*std::min(1.0, r.mu);
    r.lfOut = lf;
    fibre(n, lf, Bout);
    return r;
}


//- S17b (research/designs/2026-09-27-S17b-slack-fibre-field-units-wall-
//  face.md, A1 and section 4): THE EVENT VALUE. A fibre breaks at lc, so
//  content moved to the pending share at a stretch l1 = sqrt(b1) freezes at
//  min(l1, lc) (docket 09 rule 4). One expression for the law and its unit
//  test (l1 3.0 -> 2.8).
inline double eventStretch(const double b1, const double lc)
{
    return std::min(std::sqrt(b1), lc);
}


//- S17b A1, V1 (docket 07 section 2; Alireza's choice; design section 8 b):
//  ONE UNI STEP OF A TAUT FIBRE. mu is the step's isochoric line stretch
//  |F n| det(F)^(-1/3). If lf min(1, mu) <= 1 -- A1 as written -- the fibre
//  is at (or has reached within the step) its rest length: it is SLACK. It
//  buckles and carries no force from then on (sources 56, 75, 76), and the
//  caller zeroes the taut content, C_B and q_P at once (lfOut = 1,
//  qRatio = 0). This includes a fibre already at lf = 1 whatever mu is:
//  T(1) = 0 and UNI can never re-tension it (15). The event conformance is
//  therefore "mu < 1 or lf = 1". Otherwise lfOut = lf min(1, mu) > 1, the
//  S17 UNI value bit for bit; on a contraction q_P scales with the tension,
//  qRatio = T(lfOut)/T(lf) (03's clause), qRatio = 1 when the line does not
//  contract, and q_P := 0 wherever T(lfOut) = 0 (section 8 b; on this
//  branch T(lfOut) > 0, so the clause states the invariant).
inline bool slackStep
(
    const double lf, const double mu, double& lfOut, double& qRatio
)
{
    const double lfNew = lf*std::min(1.0, mu);
    if (!(lfNew > 1.0))
    {
        lfOut = 1.0;
        qRatio = 0.0;
        return true;
    }
    lfOut = lfNew;
    const double Tnew = fibreTension(lfNew);
    qRatio = !(Tnew > 0.0) ? 0.0 : (mu < 1.0 ? Tnew/fibreTension(lf) : 1.0);
    return false;
}


//- The intact tangent stiffness the STRIPES bound uses (8/9) b1 +
//  (2/9)(b2 + b3), from be's eigenvalues -- the solver keeps its own
//  expression; this is for the tests.
inline double stiffIntact(const double b1, const double bRest)
{
    return (8.0/9.0)*b1 + (2.0/9.0)*bRest;
}


//- k_eff = [(thI/S) stiff + (thP/S)(lf^2 - 1/lf)]/(1 - d), 0 where
//  1 - d = 0 (the weight max(ind, floor)(1 - d) is 0 there too). With
//  aP = 0 and aI = omd = 1 this is stiff exactly.
inline double mixedStiffness
(
    const double aI, const double stiff,
    const double aP, const double lf,
    const double omd
)
{
    if (!(omd > 0))
    {
        return 0.0;
    }
    return (aI*stiff + aP*fibreTension(lf))/omd;
}


//- M = [(thI/S) dev(be) + (thP/S) T dev(n n)]/(1 - d), 0 where 1 - d = 0.
//  Where there is no pending share, M = dev(be)*(aI/omd): with aI = omd = 1
//  that is dev(be) bit for bit (x*1 is x, including the sign of a zero).
inline void splitDev
(
    const double aI, const double devBe[6],
    const double aP, const double lf, const double n[3],
    const double omd,
    double M[6]
)
{
    if (!(omd > 0))
    {
        for (int k = 0; k < 6; ++k) { M[k] = 0.0; }
        return;
    }
    if (!(aP > 0))
    {
        const double f = aI/omd;
        for (int k = 0; k < 6; ++k) { M[k] = devBe[k]*f; }
        return;
    }
    double dn[6];
    devAxis(n, dn);
    const double T = fibreTension(lf);
    for (int k = 0; k < 6; ++k)
    {
        M[k] = (aI*devBe[k] + aP*(T*dn[k]))/omd;
    }
}


//- The shares a cell's partition implies: aI = P1/S, aP = P2/S and
//  1 - d = 1 - P3/S (d exactly as ruptured_ is formed); aI = 1, aP = 0,
//  1 - d = 1 where S = 0 (no network: no broken fibres, S16's convention).
inline void shares
(
    const double P1, const double P2, const double P3,
    double& aI, double& aP, double& omd
)
{
    const double S = P1 + (P2 + P3);
    if (S > 0)
    {
        aI = P1/S;
        aP = P2/S;
        omd = 1.0 - P3/S;
    }
    else
    {
        aI = 1.0;
        aP = 0.0;
        omd = 1.0;
    }
}

//- S17b A1 (V1): the shares with the fibre's CARRIER -- aP = C/S, the taut
//  content under fibreSlack v1 (slack content carries no tension, 07), the
//  whole pending content otherwise (C = P2: exactly shares()).
inline void sharesCarrier
(
    const double P1, const double P2, const double P3, const double C,
    double& aI, double& aP, double& omd
)
{
    shares(P1, P2, P3, aI, aP, omd);
    const double S = P1 + (P2 + P3);
    aP = S > 0 ? C/S : 0.0;
}

//- S18 A (R1; design section 2 A): the frozen pending pressure the force
//  applies, q_P/S, or none. ONE expression for the applied force
//  (networkPressureDamaged) and for the power it does (the frozenPower
//  conformance log), so the log cannot read a force the solver no longer
//  applies. applied = false under frozenPendingPressure none: q_P stays
//  stored (S17's bookkeeping) and enters no force.
inline double frozenPressureShare
(
    const double qP, const double S, const bool applied
)
{
    // the division never sees 0, then the guard selects (an optimiser may
    // evaluate both arms of a ternary: fix round 25d, networkPressureKernel.H)
    const bool has = applied && S > 0;
    const double d = has ? S : 1.0;
    const double v = qP/d;
    return has ? v : 0.0;
}


} // End namespace fibreReleaseKernel

#endif
