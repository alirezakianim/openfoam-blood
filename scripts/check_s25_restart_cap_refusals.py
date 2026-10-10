#!/usr/bin/env python3
"""S25 guard (verification V-C): the restart fallback, the deposition acceptance law and the
refusal guards, pinned in the SOURCE text.  Pure Python; it never builds anything.

  python3 -B scripts/check_s25_restart_cap_refusals.py            every property, on the live tree
  python3 -B scripts/check_s25_restart_cap_refusals.py --self-test  each property seen to FAIL on a mutated
                                                                    in-memory copy (one mutation per property,
                                                                    every anchor unique, plus a no-op control)

What it pins (design 2026-10-10-S25 sections 2C, 2F, 2G, 5; V25-7, V25-8, V25-9):
  kernel   acceptedIncrement / applyIncrement: no rejection above the cap, capacity floored at 0.
  .C       the cap reaches the kernel as depositionContentCap_, not phiSMax_.
  .C       the restart members: IOobject options, and the cold-start SEED expressions equal to the pre-S25 text
           (compared with the backup when ../OpenFoam_blood_archive is present; otherwise pinned literally).
  .C       restoreRestartState: first statement returns on an empty file, last constructor statement; store and
           restore name the same keys; the pending contents are re-derived only when not read.
  .C       timeScale: required, finite, > 0; the three consumers multiply by it.
  .C       every LT refusal's condition (the host harness in the docket probes/ve25c_cfg runs them).
  cases    every child states a time_scale > 0.
What it does NOT show: the dynamics (the host harnesses in research/investigations/20261010-0056-*/probes/ve25c_*).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLV = ROOT / "user-applications/phaseClotAxisymmetricFluid"
CFILE = SOLV / "phaseClotAxisymmetricFluid.C"
KFILE = ROOT / "user-applications/plateletContent/plateletTransferKernel.H"
CTRL = ROOT / "cases/AM2020_planar_mural_detachment/constant/mainCaseControls.json"
PRE = ROOT.parent / "OpenFoam_blood_archive/solver_backups/2026-10-10_pre_S25b1/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"


def squash(s):
    return re.sub(r"\s+", "", s)


def once(text, anchor):
    n = text.count(anchor)
    if n != 1:
        raise AssertionError(f"anchor occurs {n} times, not 1: {anchor[:70]!r}")


def init_entry(text, member):
    """The member-initialiser entry `member\\n    (` ... balanced `)`, comments removed, IO options normalised."""
    m = re.search(r"\n    " + re.escape(member) + r"\n    \(\n", text)
    if not m:
        raise AssertionError(f"initialiser entry {member} not found")
    i = m.end() - 2
    depth, j = 0, i
    while True:
        c = text[j]
        depth += (c == "(") - (c == ")")
        j += 1
        if depth == 0:
            break
    body = re.sub(r"//[^\n]*", "", text[i:j])
    body = re.sub(r"IOobject::(NO_READ|READ_IF_PRESENT)", "IOobject::R", body)
    body = re.sub(r"IOobject::(NO_WRITE|AUTO_WRITE)", "IOobject::W", body)
    return squash(body)


def io_opts(text, member):
    m = re.search(r"\n    " + re.escape(member) + r"\n    \(\n(.*?)\n    \)", text, re.S)
    return squash(re.findall(r"IOobject::\w+,\s*IOobject::\w+", m.group(1))[0]) if m else None


def keys_of(block, how):
    return sorted(set(re.findall(how, block)))


# Each property: (name, predicate(files)->None or raises AssertionError, mutation(files)->files)
def P_k_acc_cap(f):
    k = f["K"]
    once(k, "const Scalar capacity = std::max(thetaMax - theta, Scalar(0))/Y;")
    assert "theta > thetaMax" not in k, "acceptedIncrement rejects theta above the cap again"


def P_k_app_cap(f):
    k = f["K"]
    once(k, "|| dB > std::max(thetaMax - (C + H), Scalar(0))/Y)")
    assert "C + H > thetaMax" not in k, "applyIncrement rejects C + H above the cap again"


def P_c_cap_arg(f):
    c = f["C"]
    once(c, "NStar, thetaStar, dt, bindRate, plateletYield_,\n            depositionContentCap_, dB, nAvail, nCap")
    once(c, "NStar, thetaC_, thetaHStar, dB, plateletYield_, depositionContentCap_,")


def P_c_cap_default(f):
    c = f["C"]
    once(c, '"depositionContentCap", phiSMax_')
    assert re.search(r"std::isfinite\(depositionContentCap_\)\s*&&\s*depositionContentCap_ > 0\s*&&\s*depositionContentCap_ < 1", c)


RESTART = {  # member -> (read, write) tokens as shipped
    "phiSkel_": "IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE",
    "phiFluidOwn_": "IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE",
    "phiSkelOwn_": "IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE",
    "thetaPendingB_": "IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE",
    "thetaPendingP_": "IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE",
    "e_": "IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE",
}


def P_c_restart_io(f):
    for m, want in RESTART.items():
        got = io_opts(f["C"], m)
        assert got == want, f"{m}: IOobject options {got} != {want}"
    once(f["C"], '"phaseClotState", runTime.name(), "uniform", mesh,\n            IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE')


def P_c_seeds(f):
    c = f["C"]
    once(c, 'phi_*dimensionedScalar("zero", dimless, 0)\n    ),\n    skeletonDivergenceRun_')
    if PRE.is_file():
        pre = PRE.read_text()
        for m in ("phiSkel_", "phiFluidOwn_", "phiSkelOwn_", "thetaPendingB_", "thetaPendingP_"):
            a, b = init_entry(pre, m), init_entry(c, m)
            assert a == b, f"cold-start seed of {m} differs from the pre-S25 text (after normalising the IO options)"


def P_c_restore(f):
    c = f["C"]
    s = c.split("void Foam::solvers::phaseClotAxisymmetricFluid::storeRestartState()", 1)[1]
    store, rest = s.split("void Foam::solvers::phaseClotAxisymmetricFluid::restoreRestartState()", 1)
    restore = rest.split("void Foam::solvers::phaseClotAxisymmetricFluid::reportLoadRatio()", 1)[0]
    restore_nc = re.sub(r"//[^\n]*", "", restore)
    assert squash(restore_nc).startswith("{" + squash(
        "const dictionary& d = phaseClotState_;if (d.empty()){Info<<\"PHASE_CLOT_S25_RESTART state=seeded\"<<endl;return;}")), \
        "restoreRestartState does not return first on an empty file"
    ks = keys_of(store, r'd\.set\("(\w+)"')
    kr = keys_of(restore, r'd\.lookup<\w+>\("(\w+)"')
    assert ks == kr and len(ks) == 16, f"store keys {len(ks)} != restore keys {len(kr)} or != 16"
    arr = c.split("const char* const s25StateScalars[]", 1)[1].split("}\n\n\nvoid Foam::solvers", 1)[0]
    assert keys_of(arr, r'"(\w+)"') == ks, "the all-or-none key tables differ from the stored keys"
    assert squash(c).count(squash("restoreRestartState();\n}")) == 1, "restoreRestartState is not the last constructor statement"
    post = c.split("void Foam::solvers::phaseClotAxisymmetricFluid::postSolve()", 1)[1]
    assert squash(post).index("reportLoadRatio();") < squash(post).index("storeRestartState();"), "store is not last in postSolve"


def P_c_pending(f):
    once(f["C"], "if (!pendingContentsRead)\n            {\n                fibreReleaseKernel::fibre(nA, lI[celli], B);")
    once(f["C"], "const bool pendingContentsRead = hasPB && hasPP;")


def P_c_e_seed(f):
    c = f["C"]
    once(c, "if (!e_.headerOk())\n        {\n            e_.primitiveFieldRef() =\n                q_.primitiveField()*(w_.primitiveField() - 1.0);")
    once(c, "e_.correctBoundaryConditions();\n        }\n        Wp_.primitiveFieldRef() = q_.primitiveField() + e_.primitiveField();")


def P_c_timescale(f):
    c = f["C"]
    once(c, 'if (!props.found("timeScale"))')
    once(c, "if (!(std::isfinite(ts) && ts > 0))")
    assert len(re.findall(r"timeScale_\*3\.3e-4", c)) == 1 and len(re.findall(r"timeScale_\*2\.5e-4", c)) == 1
    assert len(re.findall(r"timeScale_\*5e-4", c)) == 1, "the tau_v floor must be timeScale_*5e-4 exactly once"


LT_ANCHORS = [  # the refusal conditions, one unique statement each
    '        if (law != "shipped" && law != "valanisLandelTrilinear")\n',
    '        if (!props.isDict("skeletonLawCoeffs"))\n',
    '            if (!d.found(names[k]))\n',
    '            if (!std::isfinite(v[k]))\n',
    '        if (!(v[0] > 0) || !(v[1] > 0))\n',
    '        if (!(v[2] < v[3]))\n',
    '        if (!(v[2] >= 0))\n',
    '        if (m != "constant" && m != "tangent")\n',
    '        if (m == "tangent" && !lt)\n',
    '            if (coreShell_)\n            {\n                FatalErrorInFunction\n                    << "skeletonLaw valanisLandelTrilinear with coreShell "',
    '                !phaseClotProperties_.found("G")\n             || G_.value() != 0.5*skeletonLawCoeffs_.E1\n',
    '                !phaseClotProperties_.found("networkPressureCoeff")\n             || networkPressureCoeff_ != skeletonLawCoeffs_.kappa\n',
    '            if (!(skeletonVolumeFraction_ < networkPole()))\n',
]


def P_c_lt_refusals(f):
    for a in LT_ANCHORS:
        once(f["C"], a)


def P_cases_time_scale(f):
    bad = []

    def walk(o):
        if isinstance(o, dict):
            if "child_id" in o or ("id" in o and "series" not in o):
                ts = o.get("time_scale")
                if not (isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > 0):
                    bad.append(o.get("child_id") or o.get("id"))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(json.loads(f["J"]))
    assert not bad, f"children without a positive time_scale: {bad[:5]}"


def rep(key, old, new):
    def m(f):
        g = dict(f)
        once(g[key], old)
        g[key] = g[key].replace(old, new)
        return g
    return m


PROPS = [
    ("kernel: acceptedIncrement floors capacity at 0 and never rejects theta > cap", P_k_acc_cap,
     rep("K", "const Scalar capacity = std::max(thetaMax - theta, Scalar(0))/Y;", "const Scalar capacity = (thetaMax - theta)/Y;")),
    ("kernel: acceptedIncrement rejection restored (pre-S25)", P_k_acc_cap,
     rep("K", "|| !std::isfinite(theta) || theta < 0\n", "|| !std::isfinite(theta) || theta < 0 || theta > thetaMax\n")),
    ("kernel: applyIncrement keeps dB <= floored capacity", P_k_app_cap,
     rep("K", "|| dB > std::max(thetaMax - (C + H), Scalar(0))/Y)", "|| dB > (thetaMax - (C + H))/Y)")),
    ("kernel: applyIncrement rejection restored (pre-S25)", P_k_app_cap,
     rep("K", "|| !std::isfinite(dB) || dB < 0 || dB > N\n", "|| C + H > thetaMax || !std::isfinite(dB) || dB < 0 || dB > N\n")),
    ("cap reaches acceptedTransfer as depositionContentCap_", P_c_cap_arg,
     rep("C", "plateletYield_,\n            depositionContentCap_, dB, nAvail, nCap", "plateletYield_,\n            phiSMax_, dB, nAvail, nCap")),
    ("cap reaches applyAcceptedTransfer as depositionContentCap_", P_c_cap_arg,
     rep("C", "dB, plateletYield_, depositionContentCap_,", "dB, plateletYield_, phiSMax_,")),
    ("depositionContentCap defaults to phiSMax_ and is range-checked", P_c_cap_default,
     rep("C", '"depositionContentCap", phiSMax_', '"depositionContentCap", 0.9')),
    ("restart members: READ_IF_PRESENT + AUTO_WRITE", P_c_restart_io,
     rep("C", '"phiSkelOwn", runTime.name(), mesh,\n            IOobject::READ_IF_PRESENT', '"phiSkelOwn", runTime.name(), mesh,\n            IOobject::NO_READ')),
    ("cold-start seed of phiSkelOwn_ is phi_*0 (as pre-S25)", P_c_seeds,
     rep("C", 'phi_*dimensionedScalar("zero", dimless, 0)\n    ),\n    skeletonDivergenceRun_', 'phi_\n    ),\n    skeletonDivergenceRun_')),
    ("restoreRestartState returns first on an empty file", P_c_restore,
     rep("C", '        Info<< "PHASE_CLOT_S25_RESTART state=seeded" << endl;\n        return;\n', '        Info<< "PHASE_CLOT_S25_RESTART state=seeded" << endl;\n')),
    ("store and restore name the same 16 keys", P_c_restore,
     rep("C", '    d.set("s23OutfluxInt", s23OutfluxInt_);\n', '')),
    ("pending contents re-derived only when not read", P_c_pending,
     rep("C", "if (!pendingContentsRead)\n            {\n                fibreReleaseKernel::fibre(nA, lI[celli], B);", "if (true)\n            {\n                fibreReleaseKernel::fibre(nA, lI[celli], B);")),
    ("e seed: q*(w-1) when no e file, Wp := q + e", P_c_e_seed,
     rep("C", "q_.primitiveField()*(w_.primitiveField() - 1.0);", "q_.primitiveField()*(w_.primitiveField() - 0.0);")),
    ("timeScale required, finite, > 0; consumers multiply by it", P_c_timescale,
     rep("C", "timeScale_*5e-4", "5e-4")),
    ("LT refusals: skeletonLaw name", P_c_lt_refusals, rep("C", LT_ANCHORS[0], "        if (false)\n")),
    ("LT refusals: coeffs subdictionary", P_c_lt_refusals, rep("C", LT_ANCHORS[1], "        if (false)\n")),
    ("LT refusals: E1, E2 > 0", P_c_lt_refusals, rep("C", LT_ANCHORS[4], "        if (false)\n")),
    ("LT refusals: D1 < D2", P_c_lt_refusals, rep("C", LT_ANCHORS[5], "        if (false)\n")),
    ("LT refusals: tangent under shipped", P_c_lt_refusals, rep("C", LT_ANCHORS[8], "        if (false)\n")),
    ("LT refusals: LT + coreShell", P_c_lt_refusals, rep("C", LT_ANCHORS[9], LT_ANCHORS[9].replace("if (coreShell_)", "if (false)"))),
    ("LT refusals: G = E1/2", P_c_lt_refusals, rep("C", LT_ANCHORS[10], "                false\n")),
    ("LT refusals: phi0 < pole", P_c_lt_refusals, rep("C", LT_ANCHORS[12], "            if (false)\n")),
]


def files():
    return {"C": CFILE.read_text(), "K": KFILE.read_text(), "J": CTRL.read_text()}


def cases_mut(f):
    j = json.loads(f["J"])
    for ch in j["children"]:
        if (ch.get("child_id") or ch.get("id")) == "D75_refI":
            ch.pop("time_scale")
    return {**f, "J": json.dumps(j)}


def run(f, label):
    bad = 0
    for name, pred, _ in PROPS + [("every child states a positive time_scale", P_cases_time_scale, cases_mut)]:
        try:
            pred(f)
            print(f"PASS {label}{name}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {label}{name}: {e}")
    return bad


def main():
    f = files()
    bad = run(f, "")
    if "--self-test" not in sys.argv:
        print(f"check_s25_restart_cap_refusals: {'FAIL' if bad else 'OK'} ({len(PROPS) + 1} properties, {bad} failed)")
        return 1 if bad else 0
    if bad:
        print("self-test needs a passing live tree")
        return 1
    miss = 0
    # no-op control: whitespace appended to a statement must change nothing
    ctrl = {**f, "C": f["C"].replace("    s17TransferB1Max_ = 0;\n", "    s17TransferB1Max_ = 0;  \n", 1)}
    for name, pred, _ in PROPS:
        try:
            pred(ctrl)
        except AssertionError as e:
            miss += 1
            print(f"SELF-TEST FAIL: no-op control broke {name}: {e}")
    for name, pred, mut in PROPS + [("every child states a positive time_scale", P_cases_time_scale, cases_mut)]:
        try:
            g = mut(f)
        except AssertionError as e:
            miss += 1
            print(f"SELF-TEST FAIL: mutation anchor for {name!r} is not unique: {e}")
            continue
        try:
            pred(g)
        except AssertionError as e:
            print(f"SELF-TEST ok: mutation of {name!r} fails with: {str(e)[:90]}")
        else:
            miss += 1
            print(f"SELF-TEST FAIL: mutation of {name!r} was NOT caught")
    print(f"check_s25_restart_cap_refusals --self-test: {'FAIL' if miss else 'OK'} ({miss} problems)")
    return 1 if miss else 0


if __name__ == "__main__":
    sys.exit(main())
