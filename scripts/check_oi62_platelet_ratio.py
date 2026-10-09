#!/usr/bin/env python3
"""OI-62 guard: the platelet ratio w and its diffusive conductance where q < qFloor.

research/designs/2026-10-08-OI62-platelet-ratio-at-negative-q.md rev. 2, s3 and s6 V62-8
(docket research/investigations/20261008-1507-run081155-b72-s23-validation/, 10 and 11).

The solver's platelet equation advances W = q*w with the explicit source
fvc::laplacian(gammaPlateletf, w_). Before OI-62 the operand was W/max(|q|, qFloor) and the
conductance harmonic(max(|q|, qFloor)*D): at Wp = q that read w = -1 at q < 0 and q/qFloor in the
band, and a dry face (no free blood) conducted |q|*D, so the term seeded |Wp - q| up to 2.6e-3 per
step (BATCH 72). OI-62 (no switch) makes:
  (a) the operand C6 at n:   W/q where |q_old| >= qFloor; 1 + (W - q_old)/q_s,old in the band;
  (b) the conductance today's harmonic(qEffOld*D_), masked IN PLACE to 0 on every face with a
      q_old <= 0 side (boundary: patch value or owner); D = 0 path unchanged;
  (c) the written w_ C6 at n+1;
  (d) PHASE_CLOT_PLATELET_FO ... plateletFoLimit=2 dryFacesConducting=<n>;
  and leaves the adhesion sink, wGrowth and wStoich on |q| (OI-63).

Each predicate pins source text in solvePlatelets(); the C6 rule is also EVALUATED (its return
expression, translated to Python doubles) at Wp = q for every sign of q and against today's
quotient for q >= qFloor. Every predicate has at least one planted defect that must be CAUGHT,
labelled with V62-8's letter (a)-(h); a no-op control edit must change nothing (MISSED).

usage: python3 -B scripts/check_oi62_platelet_ratio.py      (exit 0 = pass, 1 = fail)
Written by solver-engineer 12 (docket 20261008-1507-run081155-b72-s23-validation).
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "user-applications" / "phaseClotAxisymmetricFluid" / "phaseClotAxisymmetricFluid.C"
SIG = "::solvePlatelets()"


def strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def ns(text: str) -> str:
    return re.sub(r"\s+", "", strip_comments(text))


def fn_body(src: str, sig: str) -> str:
    """The brace-matched body after the first occurrence of `sig` (comments stripped)."""
    s = strip_comments(src)
    i = s.index(sig)
    j = s.index("{", i)
    depth = 0
    for k in range(j, len(s)):
        depth += s[k] == "{"
        depth -= s[k] == "}"
        if depth == 0:
            return s[j:k + 1]
    raise ValueError(sig)


def body(c: str) -> str:
    try:
        return ns(fn_body(c, SIG))
    except ValueError:
        return ""


# ---- pinned flattened text ----------------------------------------------------------------
Q_OLD_S = 'constvolScalarFieldqOldS("qOldS",qOldPlatelet_.valid()?qOldPlatelet_():q_);'
OPERAND = 'w_=plateletRatioC6(qOldS,"wOperandC6");w_.correctBoundaryConditions();'
WRITTEN = 'w_=plateletRatioC6(q_,"wC6");w_.correctBoundaryConditions();}w_.correctBoundaryConditions();}'
LAPLACIAN = "fvc::laplacian(gammaPlateletf,w_)"
C6_HEAD = ('constautoplateletRatioC6=[this](constvolScalarField&qS,constword&ratioName){'
           'tmp<volScalarField>tRatio(volScalarField::New(ratioName,mesh_,'
           'dimensionedScalar(Wp_.dimensions()/qS.dimensions(),0)));'
           'constscalarqFloorC6=qFloor_.value();'
           'constautoc6=[qFloorC6](constscalarW,constscalarq){return')
C6_LOOPS = ('volScalarField&ratio=tRatio.ref();scalarField&ratioIn=ratio.primitiveFieldRef();'
            'constscalarField&WpIn=Wp_.primitiveField();constscalarField&qSIn=qS.primitiveField();'
            'forAll(ratioIn,celli){ratioIn[celli]=c6(WpIn[celli],qSIn[celli]);}'
            'volScalarField::Boundary&ratioB=ratio.boundaryFieldRef();forAll(ratioB,patchi){'
            'scalarField&ratioP=ratioB[patchi];constscalarField&WpP=Wp_.boundaryField()[patchi];'
            'constscalarField&qSP=qS.boundaryField()[patchi];'
            'forAll(ratioP,facei){ratioP[facei]=c6(WpP[facei],qSP[facei]);}}returntRatio;};')
C6_RETURN = "mag(q)>=qFloorC6?W/q:1.0+(W-q)/(sign(q)*max(mag(q),qFloorC6))"
Q_EFF = "constvolScalarFieldqEff(max(mag(q_),qFloor_));"
Q_EFF_OLD = ('volScalarFieldqEffOld("qEffOld",qEff);if(qOldPlatelet_.valid())'
             '{qEffOld=max(mag(qOldPlatelet_()),qFloor_);}')
HARMONIC = "if(D_.value()!=0.0){tGammaPlateletf=harmonic(mesh_).interpolate(qEffOld*D_);"
MASK = ('surfaceScalarField&gammaMasked=tGammaPlateletf.ref();'
        'constscalarField&qOldIn=qOldS.primitiveField();'
        'constlabelUList&own=mesh_.owner();constlabelUList&nei=mesh_.neighbour();'
        'scalarField&gammaIn=gammaMasked.primitiveFieldRef();'
        'forAll(gammaIn,facei){if(qOldIn[own[facei]]<=0||qOldIn[nei[facei]]<=0){gammaIn[facei]=0;}}'
        'surfaceScalarField::Boundary&gammaB=gammaMasked.boundaryFieldRef();'
        'forAll(gammaB,patchi){constscalarField&qOldP=qOldS.boundaryField()[patchi];'
        'constlabelUList&fc=mesh_.boundary()[patchi].faceCells();scalarField&gammaP=gammaB[patchi];'
        'forAll(gammaP,facei){if(qOldP[facei]<=0||qOldIn[fc[facei]]<=0){gammaP[facei]=0;}}}}')
D_ZERO = ('else{tGammaPlateletf=tmp<surfaceScalarField>(newsurfaceScalarField(IOobject('
          '"gammaPlateletfZero",mesh_.time().name(),mesh_,IOobject::NO_READ,IOobject::NO_WRITE),'
          'mesh_,dimensionedScalar("zero",D_.dimensions(),0.0)));}'
          'constsurfaceScalarFieldgammaPlateletf("gammaPlateletf",tGammaPlateletf);')
GAMMA_START = "tmp<surfaceScalarField>tGammaPlateletf;"
GAMMA_END = 'constsurfaceScalarFieldgammaPlateletf("gammaPlateletf",tGammaPlateletf);'
COUNT = ('labeldryFacesConducting=0;{constscalarField&qDryIn=qOldS.primitiveField();'
         'constlabelUList&ownDry=mesh_.owner();constlabelUList&neiDry=mesh_.neighbour();'
         'constscalarField&gammaDryIn=gammaPlateletf.primitiveField();'
         'forAll(gammaDryIn,facei){constbooldrySide=qDryIn[ownDry[facei]]<=0||qDryIn[neiDry[facei]]<=0;'
         'if(drySide&&gammaDryIn[facei]!=0){++dryFacesConducting;}}'
         'forAll(gammaPlateletf.boundaryField(),patchi){constscalarField&qDryP=qOldS.boundaryField()[patchi];'
         'constlabelUList&fcDry=mesh_.boundary()[patchi].faceCells();'
         'constscalarField&gammaDryP=gammaPlateletf.boundaryField()[patchi];'
         'forAll(gammaDryP,facei){constbooldrySide=qDryP[facei]<=0||qDryIn[fcDry[facei]]<=0;'
         'if(drySide&&gammaDryP[facei]!=0){++dryFacesConducting;}}}'
         'reduce(dryFacesConducting,sumOp<label>());}')
LOG = ('Info<<"PHASE_CLOT_PLATELET_FOtime="<<mesh_.time().userTimeValue()<<"plateletFo="<<plateletFo'
       '<<"plateletFoLimit=2"<<"dryFacesConducting="<<dryFacesConducting<<endl;')
LOG_RAW = re.compile(r'Info<<\s*"PHASE_CLOT_PLATELET_FO time="\s*<<\s*mesh_\.time\(\)\.userTimeValue\(\)'
                     r'\s*<<\s*" plateletFo="\s*<<\s*plateletFo\s*<<\s*" plateletFoLimit=2"'
                     r'\s*<<\s*" dryFacesConducting="\s*<<\s*dryFacesConducting\s*<<\s*endl;')
SINK = "fvm::Sp(interfaceDelta_*adhesionSpeedEff_/qEff,Wp_)"
W_GROWTH = "wGrowth=Wp_/max(mag(q_),qFloor_);"
W_STOICH = "wStoich=Wp_/max(mag(q_),qFloor_);"


def count_assign(text: str, name: str) -> int:
    """Assignments `name=` (not ==, not a longer identifier ending in name)."""
    return len(re.findall(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"=(?!=)", text))


# ---- the C6 rule, evaluated --------------------------------------------------------------
def _split_ternary(e: str):
    depth, q = 0, -1
    for i, ch in enumerate(e):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "?" and depth == 0 and q < 0:
            q = i
        elif ch == ":" and depth == 0 and q >= 0:
            return e[:q], e[q + 1:i], e[i + 1:]
    return None


def c6_from_source(b: str):
    """The c6 lambda's return expression from the flattened body, as a Python callable."""
    i = b.find(C6_HEAD)
    if i < 0:
        return None
    j = b.find(";};", i + len(C6_HEAD))
    if j < 0:
        return None
    expr = b[i + len(C6_HEAD):j]
    parts = _split_ternary(expr)
    py = (f"(({parts[1]}) if ({parts[0]}) else ({parts[2]}))" if parts else f"({expr})")
    if re.search(r"[^A-Za-z0-9_.()+\-*/<>=!,  ]", py.replace(" if ", "").replace(" else ", "")):
        return None
    env = {"mag": abs, "max": max, "min": min, "sign": lambda s: 1 if s >= 0 else -1,
           "__builtins__": {}}
    try:
        code = compile(py, "<c6>", "eval")
    except SyntaxError:
        return None

    def f(W: float, q: float, qF: float) -> float:
        return eval(code, dict(env, qFloorC6=qF), {"W": W, "q": q})
    return f


def c6_behaves(b: str) -> bool:
    f = c6_from_source(b)
    if f is None:
        return False
    rng = random.Random(62)
    try:
        for qF in (1e-6, 1e-3):
            qs = [-1.0, -0.5, -0.246, -1e-3, -2 * qF, -qF, -0.99 * qF, -0.5 * qF, -1e-12, -0.0, 0.0,
                  1e-12, 0.5 * qF, 0.99 * qF, qF, 1.1 * qF, 2 * qF, 0.3, 1.0]
            # the invariant: Wp = q gives exactly w = 1, for every sign of q
            if any(f(q, q, qF) != 1.0 for q in qs):
                return False
            # q = +-0 with W != q: q_s = +qFloor (OpenFOAM's sign(0) = +1, and -0.0 >= 0)
            if any(f(W, z, qF) != 1.0 + W / qF for W in (0.3, -0.2, 1e-7) for z in (0.0, -0.0)):
                return False
            for _ in range(400):
                W = rng.uniform(-1.5, 1.5)
                qp = qF + rng.random() * (1.0 - qF)
                # q >= qFloor: bitwise today's W/max(|q|, qFloor)
                if f(W, qp, qF) != W / max(abs(qp), qF) or f(W, qF, qF) != W / qF:
                    return False
                # q <= -qFloor: the signed quotient
                if f(W, -qp, qF) != W / -qp:
                    return False
                # the band: 1 + (W - q)/q_s, q_s = sign(q)*qFloor, sign(0) = +1
                qb = (rng.random() * 2.0 - 1.0) * 0.999 * qF
                s = 1 if qb >= 0 else -1
                if f(W, qb, qF) != 1.0 + (W - qb) / (s * max(abs(qb), qF)):
                    return False
    except (ArithmeticError, TypeError, NameError):
        return False
    return True


# ---- predicates ----------------------------------------------------------------------------
def p_operand(c):
    b = body(c)
    k, lap = b.find(OPERAND), b.find(LAPLACIAN)
    return (Q_OLD_S in b and 0 <= b.find(Q_OLD_S) < k < lap
            and b.count(OPERAND) == 1
            and count_assign(b, "w_") == 2              # the operand and the written w, nothing else
            and "Wp_/qEffOld" not in b and "Wp_/qEff;" not in b)


def p_c6_rule(c):
    b = body(c)
    return (b.count(C6_HEAD) == 1 and C6_HEAD + C6_RETURN + ";};" + C6_LOOPS in b
            and b.find(C6_HEAD) < b.find(OPERAND)
            and c6_behaves(b))


def p_harmonic(c):
    b, nc = body(c), ns(c)
    i0, i1 = b.find(GAMMA_START), b.find(GAMMA_END)
    seg = b[i0:i1] if 0 <= i0 < i1 else ""
    return (Q_EFF_OLD in b and b.find(Q_EFF_OLD) < b.find(Q_OLD_S)
            and nc.count("harmonic(mesh_).interpolate(") == 1
            and HARMONIC in seg and seg.count("tGammaPlateletf=") == 2
            and D_ZERO in b
            # no hand-written face mean anywhere in the conductance: no division, no weights
            and "/" not in seg and "weights" not in seg and "reverseLinear" not in seg
            and "interpolate(" not in seg.replace("harmonic(mesh_).interpolate(qEffOld*D_)", ""))


def p_mask(c):
    b = body(c)
    i1 = b.find(GAMMA_END)
    return (HARMONIC + MASK + D_ZERO in b and 0 <= b.find(MASK) < i1
            and b.count("gammaMasked") == 3)


def p_written(c):
    b = body(c)
    k = b.find(WRITTEN)
    return (b.endswith(WRITTEN) and b.count(WRITTEN) == 1
            and b.rfind("Wp_.correctBoundaryConditions();") < k
            and "w_=Wp_/" not in b)


def p_oi63_untouched(c):
    b, nc = body(c), ns(c)
    return (Q_EFF in b and b.count(SINK) == 2
            and nc.count("fvm::Sp(interfaceDelta_*adhesionSpeedEff_/") == 2
            and nc.count(W_GROWTH) == 1 and count_assign(nc, "wGrowth") == 1
            and nc.count(W_STOICH) == 1 and count_assign(nc, "wStoich") == 1
            # C6 and the signed q_old are used in solvePlatelets() only
            and nc.count("plateletRatioC6") == b.count("plateletRatioC6") == 3
            and nc.count("qOldS") == b.count("qOldS"))


def p_log(c):
    b = body(c)
    return (b.count(COUNT) == 1 and b.count(LOG) == 1
            and b.find(GAMMA_END) < b.find(COUNT) < b.find(LOG)
            and LOG_RAW.search(strip_comments(c)) is not None
            and b.count("dryFacesConducting") == 6)


def p_comments(c):
    return ("a conserved, non-negative platelet content" not in c
            and "cannot disagree about what w is" not in c
            and "it is NOT non-negative" in c
            and "It is NOT the diffusion operand" in c)


PREDICATES = (
    ("operand at n: w_ = C6 on the signed q_old, before the laplacian; no W/max(|q|, qFloor) (s3 a)",
     p_operand),
    ("C6 rule: |q| >= qFloor -> W/q, band -> 1 + (W - q)/q_s, on every cell and patch value; "
     "evaluated: Wp = q gives 1 for every q, bitwise today for q >= qFloor (s3 a, c)", p_c6_rule),
    ("conductance: today's harmonic(qEffOld*D_) once, D != 0 only, qEffOld unchanged, D = 0 path "
     "unchanged, no hand-written mean (s3 b)", p_harmonic),
    ("dry-face mask in place on the harmonic result: internal faces with a q_old <= 0 side, boundary "
     "faces whose patch value or owner is <= 0, before gammaPlateletf (s3 b)", p_mask),
    ("written w_ at n+1 = C6 on q_, after the solve (s3 c)", p_written),
    ("OI-63 sites unchanged: sink on qEff = max(|q|, qFloor), wGrowth and wStoich on W/max(|q|, qFloor); "
     "C6 used only in solvePlatelets (s3 Not changed)", p_oi63_untouched),
    ("log: PHASE_CLOT_PLATELET_FO ... plateletFoLimit=2 dryFacesConducting=<n>, counted on the "
     "gammaPlateletf the laplacian reads, reduce/sumOp (s3 d)", p_log),
    ("comments corrected: Wp not non-negative; wGrowth is not the diffusion operand (s3 e)", p_comments),
)

# Planted defects: (V62-8 letter, unique anchor in the .C, replacement, predicate index)
MUTATIONS = (
    ("a", '        w_ = plateletRatioC6(qOldS, "wOperandC6");\n', "        w_ = Wp_/qEffOld;\n", 0),
    ("a", '        w_ = plateletRatioC6(qOldS, "wOperandC6");\n',
     '        w_ = plateletRatioC6(q_, "wOperandC6");\n', 0),
    ("a", "                    ? W/q\n", "                    ? W/mag(q)\n", 1),
    ("b", "                if (qOldIn[own[facei]] <= 0 || qOldIn[nei[facei]] <= 0)\n"
          "                {\n                    gammaIn[facei] = 0;\n                }\n", "", 3),
    ("b", "                    if (qOldP[facei] <= 0 || qOldIn[fc[facei]] <= 0)\n"
          "                    {\n                        gammaP[facei] = 0;\n                    }\n", "", 3),
    ("b", "                if (qOldIn[own[facei]] <= 0 || qOldIn[nei[facei]] <= 0)\n",
     "                if (qOldIn[own[facei]] < 0 || qOldIn[nei[facei]] < 0)\n", 3),
    ("b", "            const scalarField& qOldIn = qOldS.primitiveField();\n",
     "            const scalarField& qOldIn = qEffOld.primitiveField();\n", 3),
    ("c", "                return mag(q) >= qFloorC6\n                    ? W/q\n"
          "                    : 1.0 + (W - q)/(sign(q)*max(mag(q), qFloorC6));\n",
     "                return W/(sign(q)*max(mag(q), qFloorC6));\n", 1),
    ("c", "                    ratioP[facei] = c6(WpP[facei], qSP[facei]);\n", "", 1),
    ("d", "            tGammaPlateletf = harmonic(mesh_).interpolate(qEffOld*D_);\n",
     "            tGammaPlateletf = harmonic(mesh_).interpolate\n"
     "            (\n                max(qOldS, dimensionedScalar(dimless, 0))*D_\n            );\n", 2),
    ("e", "            tGammaPlateletf = harmonic(mesh_).interpolate(qEffOld*D_);\n",
     "            tGammaPlateletf = fvc::interpolate(qEffOld*D_);\n            {\n"
     "                surfaceScalarField& g09 = tGammaPlateletf.ref();\n"
     "                const scalarField gC((qEffOld*D_)().primitiveField());\n"
     "                const scalarField& wts = mesh_.weights();\n"
     "                forAll(g09, facei)\n                {\n"
     "                    const scalar gP = gC[mesh_.owner()[facei]];\n"
     "                    const scalar gN = gC[mesh_.neighbour()[facei]];\n"
     "                    g09[facei] = gP*gN/(wts[facei]*gN + (1 - wts[facei])*gP);\n"
     "                }\n            }\n", 2),
    ("f", '        w_ = plateletRatioC6(q_, "wC6");\n', "        w_ = Wp_/qEff;\n", 4),
    ("f", '        w_ = plateletRatioC6(q_, "wC6");\n', '        w_ = plateletRatioC6(qOldS, "wC6");\n', 4),
    ("g", "              + fvm::Sp(interfaceDelta_*adhesionSpeedEff_/qEff, Wp_)\n             ==\n"
          "                fvc::laplacian(gammaPlateletf, w_)\n              + carrierDivergence",
     "              + fvm::Sp(interfaceDelta_*adhesionSpeedEff_/(sign(qOldS)*qEff), Wp_)\n             ==\n"
     "                fvc::laplacian(gammaPlateletf, w_)\n              + carrierDivergence", 5),
    ("g", "            wGrowth = Wp_/max(mag(q_), qFloor_);\n",
     "            wGrowth = Wp_/(sign(q_)*max(mag(q_), qFloor_));\n", 5),
    ("g", "            wStoich = Wp_/max(mag(q_), qFloor_);\n",
     "            wStoich = Wp_/(sign(q_)*max(mag(q_), qFloor_));\n", 5),
    ("h", '            << " dryFacesConducting=" << dryFacesConducting\n', "", 6),
    ("h", "            reduce(dryFacesConducting, sumOp<label>());\n", "", 6),
    ("h", "                if (drySide && gammaDryIn[facei] != 0)\n",
     "                if (drySide && gammaDryIn[facei] == 0)\n", 6),
    ("s3e-comment", "            // a conserved platelet content; it is NOT non-negative: where\n",
     "            // a conserved, non-negative platelet content, and handing it the\n", 7),
    ("s3e-comment", "        // the same way. NOT a gradient of it. It is NOT the diffusion operand\n",
     "        // the same way, so the two cannot disagree about what w is.\n", 7),
)


def run(c: str) -> list[bool]:
    return [bool(p(c)) for _, p in PREDICATES]


def main() -> int:
    c = SOLVER.read_text(encoding="utf-8")
    ok = True
    live = run(c)
    for (name, _), r in zip(PREDICATES, live):
        print(("PASS " if r else "FAIL ") + name)
        ok &= r
    ctl = run(c + "\n// control edit\n")
    if ctl != live:
        print("FAIL the no-op control edit changed a predicate")
        ok = False
    else:
        print("MISSED control: a no-op edit (appended comment) changes no predicate")
    covered = set()
    for letter, old, new, idx in MUTATIONS:
        n = c.count(old)
        if n != 1:
            print(f"FAIL ({letter}) mutation anchor for predicate {idx} occurs {n} times (must be 1): "
                  f"{old.strip()[:60]!r}")
            ok = False
            continue
        r = run(c.replace(old, new, 1))
        caught = not r[idx]
        print(("CAUGHT " if caught else "MISSED ") + f"({letter}) {old.strip()[:50]!r} -> "
              f"{PREDICATES[idx][0][:60]}")
        ok &= caught
        covered.add(idx)
    if covered != set(range(len(PREDICATES))):
        print(f"FAIL predicates without a planted defect: {sorted(set(range(len(PREDICATES))) - covered)}")
        ok = False
    letters = {m[0] for m in MUTATIONS}
    if not set("abcdefgh") <= letters:
        print(f"FAIL V62-8 defects without a mutation: {sorted(set('abcdefgh') - letters)}")
        ok = False
    print("OI-62 GUARD PASSED" if ok else "OI-62 GUARD FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
