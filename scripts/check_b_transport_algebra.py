#!/usr/bin/env python3
"""Verify the Eulerian elastic strain transport before any of it is written in C++.

WHY THIS EXISTS
---------------
The Eulerian solid plan replaces the displacement skeleton with a transported
left Cauchy-Green tensor b. The whole formulation rests on one evolution law:

    db/dt + (v.grad)b  =  L.b + b.L^T  -  (1/tau)(b - I)

with L the velocity gradient, and 1/tau nonzero only in blood, where it relaxes
b back to I. Get any part of that wrong and the solver produces a plausible,
smooth, completely incorrect stress field -- the exact failure class this
project keeps getting caught by (min_advance_cells in cells not length; the
permeability/k0p mixup; the cost model's missing staggered loop).

THE TRAP THIS IS MOSTLY HERE FOR
--------------------------------
OpenFOAM's fvc::grad(U) returns (grad U)_ij = d(U_j)/d(x_i), which is the
TRANSPOSE of the velocity gradient L used in continuum mechanics, where
L_ij = d(v_i)/d(x_j). Use the wrong one and the stress field stays smooth,
bounded and plausible while being wrong.

CORRECTED 2026-08-16. This docstring used to claim the rotation case was the
test that caught it. That was FALSE, and the self-test did not notice because it
only asks whether SOME case catches each bug. At b = I the two forms are
algebraically identical --

    L.I + I.L^T  =  L + L^T  =  L^T + L  =  L^T.I + I.(L^T)^T

-- so a rotation started from an unstrained state stays at I under both, and
`case_rotation` returned err = 0.000e+00 for the correct law AND for the planted
`transposed_L`. It was catching nothing. The bug was in fact being caught by
simple shear and by extension alone.

The rotation case now starts from a PRE-STRAINED b0 and compares against
R b0 R^T. Under the transpose bug the state rotates the wrong way, to
R^T b0 R, which differs whenever b0 is anisotropic and the angle is not a
multiple of pi. That makes rotation load-bearing for real, and it is the case
that matters most for the clot: material orbiting the dome is rotated far more
than it is stretched.

Pure Python on purpose: no numpy, so it runs anywhere the repo does, matching
scripts/check_neohookean_algebra.py.

    python3 scripts/check_b_transport_algebra.py
    python3 scripts/check_b_transport_algebra.py --self-test
"""

import argparse
import math
import sys

N = 3


def eye():
    return [[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)]


def zeros():
    return [[0.0] * N for _ in range(N)]


def matmul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(N)) for j in range(N)]
        for i in range(N)
    ]


def transpose(a):
    return [[a[j][i] for j in range(N)] for i in range(N)]


def add(a, b, sb=1.0):
    return [[a[i][j] + sb * b[i][j] for j in range(N)] for i in range(N)]


def scale(a, s):
    return [[a[i][j] * s for j in range(N)] for i in range(N)]


def trace(a):
    return sum(a[i][i] for i in range(N))


def maxabs(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(N) for j in range(N))


def expm(a, terms=60):
    """Matrix exponential by scaling and squaring with a Taylor series."""
    norm = max(sum(abs(x) for x in row) for row in a)
    squarings = 0
    while norm > 0.5:
        a = scale(a, 0.5)
        norm *= 0.5
        squarings += 1
    result = eye()
    term = eye()
    for k in range(1, terms):
        term = scale(matmul(term, a), 1.0 / k)
        result = add(result, term)
    for _ in range(squarings):
        result = matmul(result, result)
    return result


def is_spd(a, tol=0.0):
    """Sylvester's criterion. Exact for symmetric matrices, no eigensolver."""
    def det(m, k):
        if k == 1:
            return m[0][0]
        if k == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )
    return all(det(a, k) > tol for k in (1, 2, 3))


# ---------------------------------------------------------------------------
# The law under test. `bug` injects the mistakes the self-test must catch.
# ---------------------------------------------------------------------------

def db_dt(b, L, inv_tau, bug=None):
    if bug == "transposed_L":
        # fvc::grad(U) used directly, without transposing.
        L = transpose(L)

    if bug == "dropped_second_term":
        # Only L.b, missing b.L^T. Leaves b non-symmetric and halves the rate.
        stretch = matmul(L, b)
    else:
        stretch = add(matmul(L, b), matmul(b, transpose(L)))

    if bug == "relax_sign":
        relax = scale(add(b, eye(), -1.0), +inv_tau)
    else:
        relax = scale(add(b, eye(), -1.0), -inv_tau)

    return add(stretch, relax)


def integrate(L, inv_tau, t_end, steps, bug=None, b0=None):
    b = b0 if b0 is not None else eye()
    dt = t_end / steps
    spd_held = True
    for _ in range(steps):
        k1 = db_dt(b, L, inv_tau, bug)
        k2 = db_dt(add(b, k1, 0.5 * dt), L, inv_tau, bug)
        k3 = db_dt(add(b, k2, 0.5 * dt), L, inv_tau, bug)
        k4 = db_dt(add(b, k3, dt), L, inv_tau, bug)
        # k1 + 2 k2 + 2 k3 + k4. Written with explicit halving on k4 because
        # the obvious grouping add(add(k1,k2,2), add(k3,k4,1), 2) silently
        # weights k4 by 2 -- which is how this integrator was first written,
        # and the shear and extension cases caught it.
        incr = add(add(k1, k2, 2.0), add(k3, k4, 0.5), 2.0)
        b = add(b, incr, dt / 6.0)
        if not is_spd(b):
            spd_held = False
    return b, spd_held


def analytic_b(L, t):
    """b = F F^T with F = exp(L t), valid for spatially uniform L and no relaxation."""
    F = expm(scale(L, t))
    return matmul(F, transpose(F))


def deviatoric(a):
    return add(a, eye(), -trace(a) / 3.0)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def case_simple_shear(bug=None):
    """v_x = gdot*y. L_01 = gdot. b should reach [[1+g^2, g],[g,1]]."""
    gdot, t = 1000.0, 3.0e-4          # g = 0.3, a realistic shell strain
    L = zeros()
    L[0][1] = gdot
    b, spd = integrate(L, 0.0, t, 4000, bug)
    g = gdot * t
    exact = eye()
    exact[0][0] = 1.0 + g * g
    exact[0][1] = exact[1][0] = g
    return maxabs(b, exact), spd, "simple shear, gamma=%.2f" % g


def case_uniaxial(bug=None):
    """Planar extension. L = diag(edot, -edot, 0), b = diag(e^2et, e^-2et, 1)."""
    edot, t = 500.0, 4.0e-4
    L = zeros()
    L[0][0], L[1][1] = edot, -edot
    b, spd = integrate(L, 0.0, t, 4000, bug)
    exact = eye()
    exact[0][0] = math.exp(2.0 * edot * t)
    exact[1][1] = math.exp(-2.0 * edot * t)
    return maxabs(b, exact), spd, "planar extension, stretch=%.3f" % math.exp(edot * t)


def case_rotation_unstrained(bug=None):
    """Rigid rotation of an UNSTRAINED body. b must stay exactly I, forever.

    Kept because it is the cheapest possible guard against a stretch term that
    manufactures strain out of pure spin, which is what a sign error in the
    antisymmetric part looks like.

    It is NOT a test of the fvc::grad transpose -- see the docstring at the top
    of this file. At b = I both forms give L + L^T, which is zero for
    antisymmetric L, so this case is blind to that bug by construction. Do not
    re-label it as the transpose test again.
    """
    omega, t = 2000.0, 2.0e-3         # 4 radians, well past a quarter turn
    L = zeros()
    L[0][1], L[1][0] = -omega, omega
    b, spd = integrate(L, 0.0, t, 8000, bug)
    return maxabs(b, eye()), spd, "rigid rotation of I, %.1f rad" % (omega * t)


def case_rotation_strained(bug=None):
    """Rigid rotation of a PRE-STRAINED body: b(t) must equal R b0 R^T.

    The load-bearing transpose test. Rotation neither creates nor destroys
    strain, it carries it, so the invariants are conserved and the whole
    content of the answer is WHICH WAY the state turns. The transpose bug turns
    it the other way, to R^T b0 R, and the two differ by a finite amount for
    any anisotropic b0 at any angle that is not a multiple of pi.

    Rotation matters more than stretch for this application: material orbiting
    the clot dome accumulates far more rotation than elongation, so a solver
    that got this backwards would produce a stress field pointing the wrong way
    while every scalar diagnostic -- trace, det, SPD -- stayed perfect.
    """
    omega, t = 2000.0, 5.0e-4         # 1 rad: not a multiple of pi/2 either
    L = zeros()
    L[0][1], L[1][0] = -omega, omega

    # det EXACTLY 1, which requires the off-diagonal in the arithmetic:
    # det = a*d - c^2, so d = (1 + c^2)/a, not 1/a.
    #
    # Written as 1.44 and 1/1.44 with c = 0.4 first, which gives det = 0.84.
    # Harmless here -- nothing in this file checks det -- but the SAME
    # expression was in beTransportGate.C with a comment claiming "det = 1 so
    # the det(be) diagnostic starts at its exact target", and run
    # 20260816-060212 duly reported |det - 1| = 0.1600 at its FIRST timestep
    # and failed two children on it. The solver had added 0.0048 over a full
    # radian of rotation; the other 0.1600 was this line.
    a, off = 1.44, 0.4
    b0 = eye()
    b0[0][0], b0[1][1] = a, (1.0 + off*off)/a
    b0[0][1] = b0[1][0] = off

    b, spd = integrate(L, 0.0, t, 8000, bug, b0=b0)

    theta = omega * t
    c, s = math.cos(theta), math.sin(theta)
    R = eye()
    R[0][0], R[0][1] = c, -s
    R[1][0], R[1][1] = s, c
    exact = matmul(matmul(R, b0), transpose(R))

    return maxabs(b, exact), spd, "rotation of a strained b, %.1f rad" % theta


def case_fluid_relaxation(bug=None):
    """In blood, a strained b must decay to I as exp(-t/tau)."""
    tau = 1.0e-4
    t = 3.0 * tau
    b0 = eye()
    b0[0][0], b0[0][1], b0[1][0] = 1.44, 0.4, 0.4
    b, spd = integrate(zeros(), 1.0 / tau, t, 6000, bug, b0=b0)
    decay = math.exp(-t / tau)
    exact = add(eye(), add(b0, eye(), -1.0), decay)
    return maxabs(b, exact), spd, "fluid relaxation, 3 tau"


def case_stress(bug=None):
    """Deviatoric Cauchy stress of an incompressible neo-Hookean is G*dev(b).

    Cross-checked against the strain energy W = G/2 (tr(b) - 3): the deviatoric
    stress must equal 2 * dev(dW/db . b) = G dev(b).
    """
    G = 4.0
    gdot, t = 1000.0, 3.0e-4
    L = zeros()
    L[0][1] = gdot
    b, spd = integrate(L, 0.0, t, 4000, bug)

    if bug == "stress_uses_b_not_dev":
        sigma = scale(b, G)
    else:
        sigma = scale(deviatoric(b), G)

    # dW/db = (G/2) I, so 2 dev((dW/db).b) = G dev(b).
    dWdb = scale(eye(), 0.5 * G)
    energy_route = scale(deviatoric(matmul(dWdb, b)), 2.0)
    return maxabs(sigma, energy_route), spd, "neo-Hookean deviatoric stress"


def case_damage_energy(bug=None):
    """psi+ from the displacement gradient must equal psi+ from be = F.F^T.

    STAGE 2 RESTS ON THIS ONE LINE OF ALGEBRA. solveDamage() currently builds
    the AT2 driving energy from F = I + grad(Us):

        J     = det F
        I1bar = J^(-2/3) tr(F.F^T)
        psi+  = K/2 <ln J>+^2  +  G/2 (I1bar - 3)

    The Eulerian formulation has no Us and no F, only be. The claim is that the
    identical number comes out of

        J     = sqrt(det be)
        I1bar = J^(-2/3) tr(be)

    because be IS F.F^T, so det be = (det F)^2 and tr be = tr(F.F^T). If that
    mapping is wrong the damage field is wrong everywhere while remaining
    smooth, bounded and plausible -- and the elastic-slab gate, which measures
    displacement, would not notice. Checked here against several deformations
    including a rotated one, since a rotation must change F and leave psi+
    alone.
    """
    K, G = 30.0, 4.0

    def psi_from_F(F):
        J = det3(F)
        I1bar = J ** (-2.0 / 3.0) * trace(matmul(F, transpose(F)))
        lnJpos = max(math.log(J), 0.0)
        return 0.5 * K * lnJpos ** 2 + 0.5 * G * (I1bar - 3.0)

    def psi_from_be(be):
        detbe = det3(be)
        if bug == "J_not_sqrt":
            J = detbe                      # forgetting det be = J^2
        else:
            J = math.sqrt(detbe)
        I1bar = J ** (-2.0 / 3.0) * trace(be)
        lnJpos = max(math.log(J), 0.0)
        return 0.5 * K * lnJpos ** 2 + 0.5 * G * (I1bar - 3.0)

    worst = 0.0
    for F in deformation_samples():
        be = matmul(F, transpose(F))
        worst = max(worst, abs(psi_from_F(F) - psi_from_be(be)))
    return worst, True, "damage energy, grad(Us) route vs be route"


def det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def deformation_samples():
    """A spread of F: shear, dilatation, compression, and a ROTATED shear.

    The rotated one is the point. A rotation changes every component of F and
    none of psi+, so it separates a correct invariant-based mapping from one
    that accidentally agrees on axis-aligned deformations.
    """
    out = []

    F = eye(); F[0][1] = 0.35                       # simple shear
    out.append(F)

    F = eye(); F[0][0], F[1][1], F[2][2] = 1.2, 1.1, 1.05   # dilatation
    out.append(F)

    F = eye(); F[0][0], F[1][1] = 0.85, 0.9         # compression, J < 1
    out.append(F)

    F = eye(); F[0][0], F[1][1] = 1.4, 1.0 / 1.4    # isochoric extension
    out.append(F)

    th = 0.7                                        # rotated simple shear
    c, s = math.cos(th), math.sin(th)
    R = eye(); R[0][0], R[0][1], R[1][0], R[1][1] = c, -s, s, c
    Fs = eye(); Fs[0][1] = 0.35
    out.append(matmul(R, Fs))

    return out


def case_couette_locked_slab(bug=None):
    """The steady state the stage-2 Eulerian slab gate compares against.

    That gate asserts the clot LOCKS: it shears until the elastic stress
    balances the traction from the flow above, then stops, holding be at a
    simple shear of gamma = tau/G with det(be) = 1. The assertion has two
    halves and only one of them is algebra.

    The algebra half is that be = F F^T for F = I + gamma e_x (x) e_y really
    does have be_xy = gamma and det = 1. That is checked directly below.

    The half worth integrating is that the transport equation LANDS there. The
    clot is not sheared at a constant rate -- it decelerates as its own stress
    builds -- so this integrates be under a shear rate that decays to zero on
    the Kelvin-Voigt timescale mu/G, which is the real history, and confirms
    the endpoint is the closed form the gate uses. If the transport had a
    spurious source, be would keep drifting after the shearing stopped and the
    endpoint would not be gamma.
    """
    # Slab settings from cases/AM2020_eulerian_slab_gate.
    mu, shear_modulus = 1.0e-3, 1000.0        # Pa s, Pa
    top_speed, clear_height = 1.0e-4, 5.0e-5  # m/s, m
    tau = mu * top_speed / clear_height
    gamma = tau / shear_modulus

    # gdot(t) = (tau/mu) exp(-t G/mu): the Kelvin-Voigt relaxation of the
    # shear rate, whose time integral is exactly tau/G = gamma.
    relax_time = mu / shear_modulus
    t_end = 40.0 * relax_time                  # e^-40; the gate runs far longer
    steps = 20000
    dt = t_end / steps

    b = eye()
    spd_held = True
    for step in range(steps):
        def rate_at(offset):
            L = zeros()
            L[0][1] = (tau / mu) * math.exp(-(step * dt + offset) / relax_time)
            return L
        k1 = db_dt(b, rate_at(0.0), 0.0, bug)
        k2 = db_dt(add(b, k1, 0.5 * dt), rate_at(0.5 * dt), 0.0, bug)
        k3 = db_dt(add(b, k2, 0.5 * dt), rate_at(0.5 * dt), 0.0, bug)
        k4 = db_dt(add(b, k3, dt), rate_at(dt), 0.0, bug)
        incr = add(add(k1, k2, 2.0), add(k3, k4, 0.5), 2.0)
        b = add(b, incr, dt / 6.0)
        if not is_spd(b):
            spd_held = False

    exact = eye()
    exact[0][0] = 1.0 + gamma * gamma
    exact[0][1] = exact[1][0] = gamma

    # det = 1 is gated here as well as the components: the slab assessor gates
    # it on the run, and a reference whose own determinant drifted would make
    # that check meaningless.
    error = max(maxabs(b, exact), abs(det3(b) - 1.0))
    return error, spd_held, "Couette-locked slab, gamma=%.3e" % gamma


CASES = [
    case_rotation_unstrained,
    case_couette_locked_slab,
    case_rotation_strained,
    case_simple_shear,
    case_uniaxial,
    case_fluid_relaxation,
    case_stress,
    case_damage_energy,
]

TOL = 1.0e-8

# Each planted error must be caught, and BY THE CASES NAMED HERE.
#
# This used to be a list of bugs, checked only for "caught by something". That
# is too weak, and it hid a real defect for a day: `case_rotation` was believed
# to be the transpose test, was blind to it, and the self-test still said
# `transposed_L -> caught` because shear and extension were quietly covering
# for it. A check whose coverage nobody can see is a check nobody can trust.
#
# So the expectation is written down per case. A bug that starts being caught
# by MORE cases is fine and is reported; a bug that stops being caught by a
# case named here fails the self-test, even if some other case still catches
# it. That is the difference between "the suite works" and "this test tests
# what its name says".
# WHICH CASES ARE BLIND TO WHICH BUG, AND WHY. Every exclusion below is
# structural -- the case cannot detect that bug, no matter how it is tuned --
# and each one was found by writing an over-broad expectation here and letting
# the self-test refute it.
#
#   planar extension vs transposed_L: L = diag(edot, -edot, 0) is SYMMETRIC, so
#       L^T = L and the bug is invisible. Any pure-stretch test is blind to the
#       transpose; only shear and rotation carry an antisymmetric part.
#   rotation of I vs transposed_L: at b = I both forms reduce to L + L^T. See
#       the file docstring. This is the one that was wrong before.
#   Couette-locked slab vs BOTH transport bugs: it catches neither, and the
#       reason is worth writing down because it was not what I expected when
#       the case was added. Its strain is gamma = 2e-6, so every term the two
#       bugs actually move is O(gamma^2) = 4e-12, below TOL = 1e-8.
#       Specifically: with L_01 the only nonzero entry, (b.L^T)_01 = 0 at every
#       b of this form, so dropping the second term leaves the SHEAR component
#       untouched and only halves the O(gamma^2) growth of b_xx; and a
#       transposed L moves that same gamma^2 from b_xx to b_yy. Both are real
#       errors and both are invisible at this strain. The case is here to
#       verify the ENDPOINT the slab gate compares against, not to catch
#       transport bugs -- simple shear at gamma = 0.30 does that, three orders
#       of strain higher, and it is what the expectations name.
#   neo-Hookean stress vs any transport bug: case_stress compares G dev(b)
#       against 2 dev((dW/db).b) using the SAME b on both sides, so it is an
#       identity in b. It tests the stress algebra and nothing else, and a
#       transport bug cancels out of it exactly.
EXPECTED_CATCHES = {
    "transposed_L": {
        "rotation of a strained b",
        "simple shear",
    },
    "dropped_second_term": {
        "rotation of I",
        "rotation of a strained b",
        "simple shear",
        "planar extension",
    },
    "relax_sign": {"fluid relaxation"},
    "stress_uses_b_not_dev": {"neo-Hookean deviatoric stress"},
    # J = sqrt(det be), not det be. Only the damage-energy case can see this:
    # it is a mapping between two routes to the same quantity and no transport
    # case touches it.
    "J_not_sqrt": {"damage energy"},
}


def run(bug=None, verbose=True):
    worst = 0.0
    failures = []
    for case in CASES:
        err, spd, label = case(bug)
        worst = max(worst, err)
        ok = err < TOL and spd
        if not ok:
            failures.append(label)
        if verbose:
            print(
                "  %-42s err=%.3e  spd=%-5s %s"
                % (label, err, spd, "ok" if ok else "FAIL")
            )
    return worst, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    print("b-transport algebra: db/dt = L.b + b.L^T - (1/tau)(b - I)")
    worst, failures = run(None)
    if failures:
        print("\nB-TRANSPORT ALGEBRA FAILED: %s" % ", ".join(failures))
        return 1

    if args.self_test:
        print("\nself-test: each planted error must be caught, by the cases named")
        problems = []
        for bug, expected in EXPECTED_CATCHES.items():
            _, caught_labels = run(bug, verbose=False)
            caught = {
                key
                for key in expected | set().union(*EXPECTED_CATCHES.values())
                if any(key in label for label in caught_labels)
            }
            missing = expected - caught
            print(
                "  %-22s caught by: %s"
                % (bug, ", ".join(sorted(caught)) or "NOTHING")
            )
            if missing:
                print("      MISSING: %s" % ", ".join(sorted(missing)))
                problems.append(bug)
        if problems:
            print(
                "\nSELF-TEST FAILED. A case named in EXPECTED_CATCHES no longer\n"
                "catches the bug it is supposed to catch: %s\n"
                "Do not relax the expectation to make this pass -- that is how\n"
                "case_rotation spent a day claiming to test the transpose."
                % ", ".join(problems)
            )
            return 1
        print("self-test: passed")

    print("\nB-TRANSPORT ALGEBRA OK (worst error %.2e)" % worst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
