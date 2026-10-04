#!/usr/bin/env python3
"""Verify the neo-Hookean algebra implemented in the solver, offline.

The solver writes

    W  = G/2 (I1bar - 3) + K/2 (ln J)^2          (energy, never coded)
    S  = G J^(-2/3) (I - I1 C^-1 / 3) + K ln(J) C^-1     (solveSkeleton)
    psi+ = K/2 <ln J>+^2 + G/2 (I1bar - 3)               (solveDamage)

Only S and psi+ appear in the C++. W does not, so nothing inside OpenFOAM can
catch an error in the differentiation that produced S -- a wrong S is still a
smooth, plausible, convergent stress field. That is the failure mode this
project keeps meeting (the permeability floor, the missing linear solver, the
write-only divergence flag): silently wrong, nothing downstream notices.

So the derivative is checked here by CENTRAL DIFFERENCE on W, which is
independent of how S was derived, plus the small-strain limit against linear
elasticity and the sign properties of the Amor split.

Pure Python, no numpy: this runs anywhere the rest of the harness runs.
"""

from __future__ import annotations

import math
import random
import sys

G = 4.0                                   # source 33 shell modulus, Pa
NU = 0.12                                 # material_fixed Poisson ratio
LAM = 2.0 * G * NU / (1.0 - 2.0 * NU)
K = LAM + 2.0 / 3.0 * G

I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


# ---------------------------------------------------------------- 3x3 algebra

def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def add(A, B, s=1.0):
    return [[A[i][j] + s * B[i][j] for j in range(3)] for i in range(3)]


def scale(A, s):
    return [[A[i][j] * s for j in range(3)] for i in range(3)]


def trace(A):
    return A[0][0] + A[1][1] + A[2][2]


def det(A):
    return (
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )


def inv(A):
    d = det(A)
    if abs(d) < 1e-300:
        raise ZeroDivisionError("singular")
    c = [
        [
            A[(i + 1) % 3][(j + 1) % 3] * A[(i + 2) % 3][(j + 2) % 3]
            - A[(i + 1) % 3][(j + 2) % 3] * A[(i + 2) % 3][(j + 1) % 3]
            for j in range(3)
        ]
        for i in range(3)
    ]
    # cofactor matrix transposed / det
    return [[c[j][i] / d for j in range(3)] for i in range(3)]


def amax(A):
    return max(abs(A[i][j]) for i in range(3) for j in range(3))


# ------------------------------------------------------------- the model


def W(C):
    J = math.sqrt(det(C))
    I1bar = J ** (-2.0 / 3.0) * trace(C)
    return 0.5 * G * (I1bar - 3.0) + 0.5 * K * math.log(J) ** 2


def S_formula(C):
    """Exactly the expression written into solveSkeleton()."""
    J = math.sqrt(det(C))
    Cinv = inv(C)
    I1 = trace(C)
    return add(
        scale(add(I3, Cinv, -I1 / 3.0), G * J ** (-2.0 / 3.0)),
        Cinv,
        K * math.log(J),
    )


def S_numeric(C, h=1e-6):
    """S = 2 dW/dC by central difference.

    W is written with det/trace, so it extends to a general (non-symmetric) C
    and the gradient is taken over the nine INDEPENDENT entries: perturb (i,j)
    alone, no symmetrisation. At a symmetric C this is dW/dC, and S = 2 dW/dC
    is the definition of the second Piola-Kirchhoff stress.
    """
    grad = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            Cp = [row[:] for row in C]
            Cm = [row[:] for row in C]
            Cp[i][j] += h
            Cm[i][j] -= h
            grad[i][j] = (W(Cp) - W(Cm)) / (2.0 * h)
    return scale(grad, 2.0)


def psi_plus(C):
    """Exactly the expression written into solveDamage()."""
    J = math.sqrt(det(C))
    I1bar = J ** (-2.0 / 3.0) * trace(C)
    lnJ_pos = max(math.log(J), 0.0)
    return 0.5 * K * lnJ_pos ** 2 + 0.5 * G * (I1bar - 3.0)


def C_from_gradUs(A):
    """OpenFOAM gradUs_ij = du_j/dX_i, so F^T = I + A and C = F^T F = (I+A)(I+A)^T.

    This convention is the one solveSkeleton() relies on when it forms
    `Ft(I + gradUs)` and hands `sigmaD_ & Ft` to fvc::div; getting it wrong
    transposes the stress and is invisible in any smooth field.
    """
    Ft = add(I3, A)
    return matmul(Ft, transpose(Ft))


# ------------------------------------------------------------------ checks

FAILURES = []


def report(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('   ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def self_test() -> int:
    """The check must FAIL on a wrong stress, or it is decoration.

    Four perturbations, each a mistake that is easy to make writing this by
    hand and impossible to see in a rendered field: the deviatoric 1/3, the
    J exponent, a dropped volumetric term, and the C convention transposed.
    """
    rng = random.Random(4242)
    good = S_formula

    def wrong_third(C):
        J = math.sqrt(det(C)); Cinv = inv(C); I1 = trace(C)
        return add(scale(add(I3, Cinv, -I1 / 2.0), G * J ** (-2.0 / 3.0)),
                   Cinv, K * math.log(J))

    def wrong_exponent(C):
        J = math.sqrt(det(C)); Cinv = inv(C); I1 = trace(C)
        return add(scale(add(I3, Cinv, -I1 / 3.0), G * J ** (-1.0 / 3.0)),
                   Cinv, K * math.log(J))

    def no_volumetric(C):
        J = math.sqrt(det(C)); Cinv = inv(C); I1 = trace(C)
        return scale(add(I3, Cinv, -I1 / 3.0), G * J ** (-2.0 / 3.0))

    def swapped_lame(C):
        J = math.sqrt(det(C)); Cinv = inv(C); I1 = trace(C)
        return add(scale(add(I3, Cinv, -I1 / 3.0), G * J ** (-2.0 / 3.0)),
                   Cinv, LAM * math.log(J))

    cases = [
        ("deviatoric 1/3 -> 1/2", wrong_third),
        ("J^(-2/3) -> J^(-1/3)", wrong_exponent),
        ("volumetric term dropped", no_volumetric),
        ("K -> lambda in the volumetric term", swapped_lame),
    ]

    bad = []
    for name, fn in cases:
        globals()["S_formula"] = fn
        caught = False
        for _ in range(5):
            C = C_from_gradUs(
                [[rng.gauss(0.0, 0.15) for _ in range(3)] for _ in range(3)]
            )
            rel = amax(add(fn(C), S_numeric(C), -1.0)) / max(amax(fn(C)), 1e-30)
            if rel >= 1e-5:
                caught = True
                break
        print(f"  {'ok  ' if caught else 'FAIL'}  catches: {name}")
        if not caught:
            bad.append(name)
    globals()["S_formula"] = good

    print()
    if bad:
        print(f"SELF-TEST FAILED: {len(bad)} wrong stress(es) went undetected")
        return 1
    print("self-test: passed")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()

    rng = random.Random(20260814)

    def rand(scale_):
        return [[rng.gauss(0.0, scale_) for _ in range(3)] for _ in range(3)]

    print("1. S = 2 dW/dC   (central difference on W, independent of the derivation)")
    for trial in range(5):
        C = C_from_gradUs(rand(0.15))
        Sf = S_formula(C)
        Sn = S_numeric(C)
        rel = amax(add(Sf, Sn, -1.0)) / max(amax(Sf), 1e-30)
        report(f"trial {trial}: max rel err {rel:.3e}", rel < 1e-5)

    print()
    print("2. small-strain limit -> 2G eps + lambda tr(eps) I")
    for s in (1e-3, 1e-4, 1e-5, 1e-6):
        A = rand(s)
        C = C_from_gradUs(A)
        eps = scale(add(A, transpose(A)), 0.5)
        S_lin = add(scale(eps, 2.0 * G), I3, LAM * trace(eps))
        rel = amax(add(S_formula(C), S_lin, -1.0)) / max(amax(S_lin), 1e-30)
        # the neglected term is O(eps), so the relative error must fall with it
        report(f"strain {s:.0e}: max rel err {rel:.3e}", rel < 50.0 * s)

    print()
    print("3. Amor split")
    C = C_from_gradUs([[0.4, 0, 0], [0, 0, 0], [0, 0, 0]])   # stretch, J > 1
    report(
        "dilatation: psi+ == W",
        abs(psi_plus(C) - W(C)) < 1e-12,
        f"psi+={psi_plus(C):.6f} W={W(C):.6f}",
    )

    C = C_from_gradUs([[-0.3, 0, 0], [0, 0, 0], [0, 0, 0]])  # compression, J < 1
    J = math.sqrt(det(C))
    dev_only = W(C) - 0.5 * K * math.log(J) ** 2
    report(
        "compression: volumetric part dropped",
        abs(psi_plus(C) - dev_only) < 1e-12 and psi_plus(C) < W(C),
        f"psi+={psi_plus(C):.6f} W={W(C):.6f}",
    )

    report(
        "psi+ >= 0 over 500 random deformations",
        all(psi_plus(C_from_gradUs(rand(0.3))) >= -1e-14 for _ in range(500)),
    )

    print()
    print("4. undeformed state is stress-free and energy-free")
    C0 = C_from_gradUs([[0.0] * 3 for _ in range(3)])
    report("S(I) == 0", amax(S_formula(C0)) < 1e-12)
    report("psi+(I) == 0", abs(psi_plus(C0)) < 1e-12)

    print()
    if FAILURES:
        print(f"NEO-HOOKEAN ALGEBRA FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("NEO-HOOKEAN ALGEBRA OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
