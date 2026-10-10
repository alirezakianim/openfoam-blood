/*---------------------------------------------------------------------------*\
    skeletonLawKernel.H -- S25 A: the Valanis-Landel trilinear (LT) skeleton
    law, header-only, no OpenFOAM dependencies (unit-testable on the host).

    Design: research/designs/2026-10-10-S25-arteriole-bundle.md, sections 2A
    and 3A. The discrete form is numerics 07's (docket 20261009-2327, report
    07 findings 1-4, probes/na07_lt_form.py), translated one to one:

    - tauHat(l) = tau(l)/G0, G0 = E1/2, is quadratic in l on 5 segments
      (E2 linear / blend / E1 toe / blend / E2 linear), the compressive
      branch the odd extension (C1 at the 4 breaks 1 -+ D2, 1 -+ D1).
    - The stress per unit G0 is M = [c1 dev Vbar + c2 dev bbar]/J, bbar =
      det(be)^(-1/3) be, J = sqrt(det be), with c2 = tauHat[l1,l2,l3] and
      c1 = tauHat[l1,l2] - c2 (l1 + l2) (Newton divided differences, closed
      form per segment, convex combinations across breaks) and Vbar from
      bbar and the invariants of l by Hoger-Carlson. NOTHING DIVIDES BY AN
      EIGENVALUE GAP and NO EIGENVECTOR IS TAKEN (OF-13's eigenVectors loses
      orthogonality near a double root: rel. err 0.75 at gap 1e-15).
    - The tangent per unit G0: stiff = [(4/9) t1 + (1/9)(t2 + t3)]/J,
      t_i = l_i tauHat'(l_i), axis 1 the largest (4/3 at I, as NH).
    - The largest principal shear-wave c^2 rho/G0:
      max_{i != j} tauHat[l_i,l_j] l_i^2/(l_i + l_j), over J.
    - The isochoric energy per unit G0: W = sum_i w(l_i), w(l) = int_1^l
      tauHat(x)/x dx = A ln x + B x + C x^2/2 per segment.

    Every function returns false (and writes nothing meaningful) when det or
    an eigenvalue is not finite and > 0: the caller refuses, never clamps.
\*---------------------------------------------------------------------------*/

#ifndef skeletonLawKernel_H
#define skeletonLawKernel_H

#include <cmath>

namespace skeletonLawKernel
{

//- The dictionary's skeletonLawCoeffs: E1, E2 (Pa), D1, D2 (strain), kappa
//  (K0/G0; the solver holds it as networkPressureCoeff).
struct Coeffs
{
    double E1, E2, D1, D2, kappa;
};

//- tauHat = A[k] + B[k] l + C[k] l^2 on segment k; segment k is
//  l < hi[k] (k = 0..3), and segment 4 is l >= hi[3].
struct Law
{
    double hi[4];
    double A[5], B[5], C[5];
};

inline Law makeLaw(const Coeffs& c)
{
    const double e1 = 2.0;
    const double e2 = 2.0*c.E2/c.E1;
    const double pq = (e2 - e1)/(2.0*(c.D2 - c.D1));
    const double qq = e1 - 2.0*c.D1*pq;
    const double rq = e1*c.D1 - qq*c.D1 - pq*c.D1*c.D1;
    const double gD2 = pq*c.D2*c.D2 + qq*c.D2 + rq;

    Law L;
    L.hi[0] = 1.0 - c.D2;
    L.hi[1] = 1.0 - c.D1;
    L.hi[2] = 1.0 + c.D1;
    L.hi[3] = 1.0 + c.D2;

    L.A[0] = -e2*(1.0 - c.D2) - gD2;  L.B[0] = e2;            L.C[0] = 0.0;
    L.A[1] = -(pq + qq + rq);         L.B[1] = 2.0*pq + qq;   L.C[1] = -pq;
    L.A[2] = -e1;                     L.B[2] = e1;            L.C[2] = 0.0;
    L.A[3] = pq - qq + rq;            L.B[3] = qq - 2.0*pq;   L.C[3] = pq;
    L.A[4] = gD2 - e2*(1.0 + c.D2);   L.B[4] = e2;            L.C[4] = 0.0;
    return L;
}

inline int segOf(const Law& L, const double l)
{
    for (int k = 0; k < 4; ++k)
    {
        if (l < L.hi[k])
        {
            return k;
        }
    }
    return 4;
}

inline double tauHat(const Law& L, const double l)
{
    const int k = segOf(L, l);
    return L.A[k] + L.B[k]*l + L.C[k]*l*l;
}

inline double tauHatPrime(const Law& L, const double l)
{
    const int k = segOf(L, l);
    return L.B[k] + 2.0*L.C[k]*l;
}

//- tauHat[a, b]: within a segment B + C (a + b) (the confluent limit at
//  a = b); across breaks the length-weighted mean over the pieces.
inline double divDiff1(const Law& L, double a, double b)
{
    if (a > b)
    {
        const double t = a; a = b; b = t;
    }
    const int ka = segOf(L, a);
    const int kb = segOf(L, b);
    if (ka == kb)
    {
        return L.B[ka] + L.C[ka]*(a + b);
    }
    double tot = 0.0;
    double x0 = a;
    for (int k = ka; k < kb; ++k)
    {
        const double x1 = L.hi[k];
        tot += (x1 - x0)*(L.B[k] + L.C[k]*(x0 + x1));
        x0 = x1;
    }
    tot += (b - x0)*(L.B[kb] + L.C[kb]*(x0 + b));
    return tot/(b - a);
}

//- tauHat[a, b, c]: C of the segment when all three share it; otherwise
//  (tauHat[b,c] - tauHat[a,b])/(c - a) on the sorted triple (c > a there).
inline double divDiff2(const Law& L, double a, double b, double c)
{
    double t;
    if (a > b) { t = a; a = b; b = t; }
    if (b > c) { t = b; b = c; c = t; }
    if (a > b) { t = a; a = b; b = t; }
    const int ka = segOf(L, a);
    if (ka == segOf(L, b) && ka == segOf(L, c))
    {
        return L.C[ka];
    }
    return (divDiff1(L, b, c) - divDiff1(L, a, b))/(c - a);
}

//- l_i = sqrt(ev_i*scale), ascending; false unless every value is finite
//  and > 0.
inline bool stretches(const double ev[3], const double scale, double l[3])
{
    double v[3] = {ev[0], ev[1], ev[2]};
    double t;
    if (v[0] > v[1]) { t = v[0]; v[0] = v[1]; v[1] = t; }
    if (v[1] > v[2]) { t = v[1]; v[1] = v[2]; v[2] = t; }
    if (v[0] > v[1]) { t = v[0]; v[0] = v[1]; v[1] = t; }
    for (int i = 0; i < 3; ++i)
    {
        const double s = v[i]*scale;
        if (!(std::isfinite(s) && s > 0))
        {
            return false;
        }
        l[i] = std::sqrt(s);
    }
    return true;
}

//- The stress per unit G0 at be (xx xy xz yy yz zz), its eigenvalues ev and
//  det(be): M = [c1 dev Vbar + c2 dev bbar]/J.
inline bool deviator
(
    const Law& L, const double be[6], const double ev[3], const double detBe,
    double M[6]
)
{
    if (!(std::isfinite(detBe) && detBe > 0))
    {
        return false;
    }
    const double s = std::pow(detBe, -1.0/3.0);
    const double J = std::sqrt(detBe);
    double l[3];
    if (!stretches(ev, s, l))
    {
        return false;
    }
    const double c2 = divDiff2(L, l[0], l[1], l[2]);
    const double c1 = divDiff1(L, l[0], l[1]) - c2*(l[0] + l[1]);

    // bbar and bbar^2 (symmetric: xx xy xz yy yz zz)
    const double b[6] =
        {be[0]*s, be[1]*s, be[2]*s, be[3]*s, be[4]*s, be[5]*s};
    const double b2[6] =
    {
        b[0]*b[0] + b[1]*b[1] + b[2]*b[2],
        b[0]*b[1] + b[1]*b[3] + b[2]*b[4],
        b[0]*b[2] + b[1]*b[4] + b[2]*b[5],
        b[1]*b[1] + b[3]*b[3] + b[4]*b[4],
        b[1]*b[2] + b[3]*b[4] + b[4]*b[5],
        b[2]*b[2] + b[4]*b[4] + b[5]*b[5]
    };
    // Hoger-Carlson: Vbar = [-bbar^2 + (I^2 - II) bbar + I III 1]/(I II -
    // III), I, II, III the invariants of l; the denominator is
    // (l1 + l2)(l2 + l3)(l3 + l1) > 0.
    const double I1 = l[0] + l[1] + l[2];
    const double I2 = l[0]*l[1] + l[1]*l[2] + l[2]*l[0];
    const double I3 = l[0]*l[1]*l[2];
    const double den = I1*I2 - I3;
    const double cb = I1*I1 - I2;
    const double ci = I1*I3;
    double V[6];
    for (int k = 0; k < 6; ++k)
    {
        V[k] = -b2[k] + cb*b[k];
    }
    V[0] += ci;
    V[3] += ci;
    V[5] += ci;
    for (int k = 0; k < 6; ++k)
    {
        V[k] /= den;
    }
    const double trV = (V[0] + V[3] + V[5])/3.0;
    const double trB = (b[0] + b[3] + b[5])/3.0;
    for (int k = 0; k < 6; ++k)
    {
        const bool diag = (k == 0 || k == 3 || k == 5);
        M[k] =
            (
                c1*(V[k] - (diag ? trV : 0.0))
              + c2*(b[k] - (diag ? trB : 0.0))
            )/J;
    }
    return true;
}

//- The tangent per unit G0 (the STRIPES measure along the largest stretch):
//  [(4/9) t1 + (1/9)(t2 + t3)]/J, t_i = l_i tauHat'(l_i).
inline bool tangentStiffness
(
    const Law& L, const double ev[3], const double detBe, double& stiff
)
{
    if (!(std::isfinite(detBe) && detBe > 0))
    {
        return false;
    }
    double l[3];
    if (!stretches(ev, std::pow(detBe, -1.0/3.0), l))
    {
        return false;
    }
    const double t1 = l[2]*tauHatPrime(L, l[2]);
    const double t2 = l[1]*tauHatPrime(L, l[1]);
    const double t3 = l[0]*tauHatPrime(L, l[0]);
    stiff = ((4.0/9.0)*t1 + (1.0/9.0)*(t2 + t3))/std::sqrt(detBe);
    return true;
}

//- k_c: the largest principal shear-wave c^2 rho/G0,
//  max_{i != j} tauHat[l_i,l_j] l_i^2/(l_i + l_j), over J.
inline bool waveTangent
(
    const Law& L, const double ev[3], const double detBe, double& kc
)
{
    if (!(std::isfinite(detBe) && detBe > 0))
    {
        return false;
    }
    double l[3];
    if (!stretches(ev, std::pow(detBe, -1.0/3.0), l))
    {
        return false;
    }
    double best = 0.0;
    bool first = true;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (i == j)
            {
                continue;
            }
            const double c = divDiff1(L, l[i], l[j])*l[i]*l[i]/(l[i] + l[j]);
            if (first || c > best)
            {
                best = c;
                first = false;
            }
        }
    }
    kc = best/std::sqrt(detBe);
    return true;
}

//- w(l) = int_1^l tauHat(x)/x dx, piecewise closed form.
inline double energy1(const Law& L, const double l)
{
    const double a0 = l >= 1.0 ? 1.0 : l;
    const double a1 = l >= 1.0 ? l : 1.0;
    double acc = 0.0;
    double x0 = a0;
    for (int k = 0; k <= 4; ++k)
    {
        const double lo = (k == 0) ? -HUGE_VAL : L.hi[k - 1];
        const double hi = (k == 4) ? HUGE_VAL : L.hi[k];
        if (!(hi > x0) || !(lo < a1))
        {
            continue;
        }
        const double x1 = hi < a1 ? hi : a1;
        acc +=
            L.A[k]*std::log(x1/x0)
          + L.B[k]*(x1 - x0)
          + L.C[k]*(x1*x1 - x0*x0)/2.0;
        x0 = x1;
        if (!(x0 < a1))
        {
            break;
        }
    }
    return l >= 1.0 ? acc : -acc;
}

//- The isochoric energy per unit G0, sum_i w(l_i), l_i = sqrt(ev_i J^(-2/3))
//  with the J the caller's volumetric split uses.
inline bool energy
(
    const Law& L, const double ev[3], const double J, double& W
)
{
    if (!(std::isfinite(J) && J > 0))
    {
        return false;
    }
    double l[3];
    if (!stretches(ev, std::pow(J, -2.0/3.0), l))
    {
        return false;
    }
    W = energy1(L, l[0]) + energy1(L, l[1]) + energy1(L, l[2]);
    return true;
}

} // End namespace skeletonLawKernel

#endif
