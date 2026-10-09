#!/usr/bin/env python3
"""Generate the main-case children (S21: the kept set only).


Three series on one geometry: growth against flow rate, required bond strength
against the PI's three competing factors, and three demonstration children that
actually come off the wall.

The geometry, mesh, field boundary conditions and initialiser are IMPORTED from
AM2020_mural_geometry_gate rather than transcribed. That gate returned
MURAL_READY twice with identical numbers, so it is the source of truth for this
configuration, and a copy would drift the first time either side changed.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path


# Measured on run 20260906-004750: one time directory of this case is 19.5 MB
# of ASCII fields (37 fields, 31360 cells, writePrecision 12). Used only by the
# output budget guard below.
PER_TIME_DIR_GB = 19.5 / 1024.0
OUTPUT_BUDGET_GB = 8.0


# S21 section 1.2: the 78 options the solver now runs built in, at the raw
# dictionary token every kept child used (audit rev. 2). The generator
# writes none of them and refuses a control key that names one.
BUILT_IN = {
    'advectReleased': 'yes',
    'beDetRenormalise': 'yes',
    'beFluxConsistentGradU': 'yes',
    'beRelaxBasis': 'networkAbsent',
    'beStretchGradient': 'fluxConsistent',
    'beTermDiagnostic': 'yes',
    'beTraceFreeSource': 'no',
    'blockCoupledLumen': 'eliminate',
    'blockFaceInverse': 'faceAssembled',
    'bondHoldCarrier': 'yes',
    'bondHoldTaper': 'no',
    'clotMechanics': 'yes',
    'constitutiveModel': 'neoHookean',
    'contentModel': 'yes',
    'contentOperator': 'content',
    'contentTransportScheme': 'fct',
    'cplusOrder': 'shared',
    'damageTransport': 'contentPartition',
    'dragDriver': 'legacy',
    'dragOccupancyBasis': 'intrinsic',
    'elasticCoupling': 'implicitTangent',
    'elasticStabilityStiffness': 'tangent',
    'eulerianBrinkman': 'yes',
    'eulerianMaterialFluxIsPhi': 'yes',
    'eulerianSolid': 'yes',
    'explicitStabilityBounds': 'on',
    'faceConsistentElasticForce': 'no',
    'fibreArithmetic': 'fieldUnits',
    'fibreSlack': 'v1',
    'finiteStrain': 'yes',
    'frozenPendingPressure': 'none',
    'fsiAitken': 'no',
    'fsiSweepTrace': 'yes',
    'growthClosure': 'reaction',
    'interfaceDepletionModel': 'none',
    'materialCarrierBlend': 'no',
    'networkDamageProgressive': 'on',
    'networkPressureDamage': 'tensile',
    'networkPressureForce': 'faceGradient',
    'networkPressureModel': 'convexExcess',
    'networkPressureReference': 'deposited',
    'networkRupture': 'fibreRelease',
    'operatorAudit': 'no',
    'operatorLocalisationAudit': 'no',
    'permeabilityModel': 'kozenyCarmanSolid',
    'phaseDeltaBasis': 'content',
    'phaseFieldMaterialVelocity': 'yes',
    'phaseFluidFractionBasis': 'content',
    'phaseForceBasis': 'content',
    'phaseGrowthDriver': 'transfer',
    'phaseIndicatorBasis': 'content',
    'phaseLimiterBasis': 'content',
    'phaseMomentumFluxBasis': 'shared',
    'phaseNormalBasis': 'content',
    'phasePlateletFluxBasis': 'content',
    'phaseWeightBasis': 'content',
    'plateletBindLocalisation': 'interface',
    'plateletCarrierArchive': 'no',
    'plateletContentFormulation': 'yes',
    'plateletContentModel': 'yes',
    'plateletFluxConsistent': 'no',
    'plateletFluxConvention': 'intrinsic',
    'plateletOuterForceReject': 'no',
    'plateletOuterTransactions': 'no',
    'releasedCarrier': 'material',
    'shearEnhancementModel': 'thresholdSaturating',
    'skeletonDiffuseDomain': 'yes',
    'skeletonFloorBasis': 'occupancy',
    'skeletonMomentumForm': 'phaseContinuity',
    'skeletonPressureForce': 'no',
    'skeletonResidualDiagnostic': 'no',
    'skeletonWallFace': 'zeroed',
    'stressFaceTransmission': 'series',
    'stronglyCoupledFSI': 'no',
    'twoVelocitySkeleton': 'yes',
    'unlocalisedPhaseWeightsAccepted': 'no',
    'vnCapEnabled': 'no',
    'wallPatchSplit': 'split',
}


# S21 group C1 (docket 20261005-1621, report 10, probes/se10-s21c1-deleted-keys.txt): the
# 22 phaseClotProperties parameters whose last read the S21 passes deleted. The solver's
# s21DeletedKeys table refuses each by name ("S21: deleted, no effect"), so the generator
# writes none of them and refuses a control key that names one (camel or snake case),
# together with the control keys that only fed them.
DELETED_KEYS = frozenset({
    "releasedAmbientQ", "releasedMaskDilation", "releasedMaskLimit", "releasedMaskSweeps",
    "fsiNormProbeFraction", "fsiOmegaMax", "fsiOmegaMin", "fsiTolerance", "k1",
    "materialCourant", "materialDeltaTMin", "nMaterialFloorSteps", "nFSICorrectors",
    "nSkeletonCorrectors", "nStaggeredCorrectors", "nSkeletonDivergedSteps",
    "skeletonDivergenceGrowth", "operatorAuditEvery", "phi0", "skeletonFixedPointTolerance",
    "skeletonImplicitFactor", "skeletonUsBound",
})
assert len(DELETED_KEYS) == 22 and not DELETED_KEYS & set(BUILT_IN)
# control keys whose only use was to write one of the 22 (pre-S21 create_cases.py)
DELETED_CONTROL_KEYS = {
    "material_courant": "materialCourant",
    "operator_audit_every": "operatorAuditEvery",
    "skeleton_implicit_factor": "skeletonImplicitFactor",
}


CASE = Path(__file__).resolve().parents[1]
# The main case owns its template (S21 section 3): the smoke template with its six
# built-in entries (clotMechanics, shearEnhancementModel, permeabilityModel,
# stronglyCoupledFSI, fsiAitken, fsiSweepTrace) deleted.
TEMPLATE = CASE / "template"
RUNS = CASE / "runs"

MURAL = CASE.parent / "AM2020_mural_geometry_gate"
if not (MURAL / "scripts" / "create_cases.py").is_file():
    raise SystemExit(
        f"create_cases.py: the mural gate is required but missing: {MURAL}"
    )
sys.path.insert(0, str(MURAL / "scripts"))
import create_cases as mural_gen  # noqa: E402
# The Xu dome's signed distance, IMPORTED from the initialiser that writes it,
# so the resolution floors refused here are computed on the very shape Colab
# draws rather than on a transcription of it.
import initialise_cases as mural_init  # noqa: E402
# channel_height(): the EXACT cosine-taper formula apply_throat.py bakes into
# the mesh points after blockMesh. Imported so the free-shell disc's gap to
# the curved top wall can be checked analytically, before blockMesh ever
# runs, against the SAME formula rather than a transcription of it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_throat  # noqa: E402




def verify_linear_solvers_exist(
    solution: Path, core_shell: bool, eulerian_solid: bool = False,
    rupture: bool = False, partition: bool = False,
    fibre: bool = False, implicit_tangent: bool = False,
    slack_v1: bool = False,
) -> None:
    """Every field the solver solves must have a linear solver in fvSolution.

    Run 20260814-221530 died after ONE timestep because `shell` -- the
    core/shell material label, a field added in this round -- had no entry in
    `solvers`. Nothing before the run could see it: it type-checks, it
    compiles, the case generates, blockMesh and the initialiser pass, and it
    is a FatalIOError at the first advection solve.

    The cost of missing it is a full Colab session, so it is checked here
    rather than discovered there again. `<name>Final` is checked as well as
    `<name>`, because fvMatrix::solve() appends "Final" on the last PIMPLE
    outer iteration -- an entry for only one of the two fails on the step it
    matters and passes on every other.
    """
    # W is solved whenever plateletContentFormulation is on. Added to this
    # list at the same time as the field itself -- the two previous fields to
    # miss it, `shell` and `be`, each cost a full Colab session.
    required = ["p", "U", "c", "w", "Wp", "Dg", "Us"]
    if core_shell:
        required.append("shell")
    # `be` is the field the Eulerian formulation solves, and it went missing
    # the same way `shell` did. Run 20260817-210800: stage3_xu_eul died after
    # one timestep on "keyword be is undefined in dictionary
    # .../fvSolution/solvers". This function existed precisely to stop that and
    # did not, because the new field was added to the solver and not to this
    # list. Derived from the child's own flag now, so it cannot be forgotten
    # again for the next field the same way.
    required.append("be")
    # The Stage 1 content model solves two more fields. Added here at the same
    # time as the fields themselves, which is the step that was missed for
    # both `shell` and `be`. They are declared unconditionally in the template,
    # so this is a belt-and-braces check rather than a conditional one.
    required += ["thetaC", "thetaH"]
    # S10: the deposited reference content. Added in the same edit as the
    # field, which is the step `shell` and `be` each missed.
    required += ["thetaNetRef"]
    # S14: the rupture label, solved only under network_rupture
    # criticalStretch -- added in the same edit as the field, the step
    # `shell` and `be` each missed.
    required.append("ruptured")
    # S16: the damage partition, solved (once damage exists) under
    # damage_transport contentPartition -- added in the same edit as the
    # fields.
    required += list(S16_PARTITION_FIELDS)
    # S17: the pending fibre's contents under fibreRelease, and the Vs
    # predictor under implicitTangent -- added in the same edit as the
    # solves.
    required += list(S17_FIBRE_FIELDS)
    required.append("Vs")
    # S17b A1: V1's taut share
    required += list(S17B_FIBRE_FIELDS)

    text = solution.read_text()
    missing = [
        name
        for base in required
        for name in (base, base + "Final")
        if not re.search(rf"(?m)^\s*{re.escape(name)}\s*\{{", text)
    ]
    if missing:
        raise ValueError(
            f"{solution}: no linear solver declared for {', '.join(missing)}. "
            "The solver will abort with a FatalIOError at the first solve of "
            "that field. Add it to the `solvers` block of the template."
        )


def verify_platelet_carrier_scheme(schemes: Path) -> None:
    """The Wp equation must have NO div entry, and c's must be capturable.

    This check previously required div(phiReleased,Wp) to be token-identical to
    div(phiReleased,c). That was necessary and NOT sufficient, and believing it
    was sufficient cost three Colab runs. vanLeer weights are a function of the
    transported field, so identical scheme strings on one shared flux still
    assembled different matrices -- run 20260822-103655 measured |Wp - q| =
    2.27e-04 under vanLeer against 5.85e-12 under field-independent linear
    weights, with every source term zero and Wp initialised equal to q.

    The Wp equation now reuses c's captured face weights, so a
    div(phiReleased,Wp) entry controls nothing. It is refused rather than
    ignored: inert config that looks live gets tuned by someone expecting an
    effect. The solver refuses it too; this check is here because the solver's
    refusal costs a Colab round trip and this one costs nothing.
    """
    text = schemes.read_text()

    if re.search(r"(?m)^\s*div\(phiReleased,Wp\)\s", text):
        raise ValueError(
            f"{schemes}: div(phiReleased,Wp) is present but is no longer used. "
            f"The platelet content equation reuses the face weights of "
            f"div(phiReleased,c). Remove the entry -- leaving it implies a "
            f"control that does not exist."
        )

    m = re.search(r"(?m)^\s*div\(phiReleased,c\)\s+([^;]+);", text)
    if not m:
        raise ValueError(
            f"{schemes}: no divScheme for div(phiReleased,c). divSchemes "
            f"declares `default none`, so this is a FatalIOError at the first "
            f"solve -- and the Wp equation captures its weights from this "
            f"entry, so without it there is nothing to transport Wp with."
        )

    tokens = m.group(1).split()
    if tokens[:2] != ["bounded", "Gauss"]:
        raise ValueError(
            f"{schemes}: div(phiReleased,c) must be spelled "
            f"`bounded Gauss <limiter>`; the solver consumes those two tokens "
            f"to reach the interpolation scheme whose weights the Wp equation "
            f"reuses.\n  found: {' '.join(tokens)}\n"
            f"`bounded` is not decorative -- phiReleased is deliberately not "
            f"solenoidal."
        )


def replace_entry(path: Path, key: str, value: object) -> None:
    text = path.read_text()
    updated, count = re.subn(
        rf"(?m)^(\s*{re.escape(key)}\s+(?:\[[^\]]+\]\s+)?)[^;]+;",
        lambda m: f"{m.group(1)}{value};", text, count=1,
    )
    if count != 1:
        raise ValueError(f"{path}: expected exactly one {key}")
    path.write_text(updated)


def add_or_replace_entry(path: Path, key: str, value: object) -> None:
    text = path.read_text()
    if re.search(rf"(?m)^\s*{re.escape(key)}\s+", text):
        replace_entry(path, key, value)
        return
    path.write_text(text.rstrip("\n") + f"\n\n{key} {value};\n")















S13_SWITCHES = (
    ("network_pressure_weight", "networkPressureWeight",
     ("networkPresence", "contentRatio")),
)








S15_SWITCHES = (
    ("material_label_basis", "materialLabelBasis",
     ("transportedLabel", "contentComposition")),
)





# The damage partition's three fields (solver: thetaIntact_, thetaPending_,
# thetaBroken_). Each needs a linear solver under contentPartition.
S16_PARTITION_FIELDS = ("thetaIntact", "thetaPending", "thetaBroken")






# The Vs predictor's linear solver under implicitTangent: the matrix is an
# M-matrix but asymmetric (bounded upwind convection), so a symmetric-only
# solver is refused (design section 4). The first is the default.
S17_VS_SOLVERS = ("smoothSolver", "PBiCGStab")
# The pending fibre's carried contents (solver thetaPendingB_, thetaPendingP_),
# each needing a linear solver under fibreRelease.
S17_FIBRE_FIELDS = ("thetaPendingB", "thetaPendingP")




# Scheduling metadata, read only by scripts/check_notebook_batch_selector.py:
# it says why a defined child is not run by any BATCHES entry. It is never
# configuration, so no parent check compares it (S21 P2, 2026-10-06).
_SCHEDULING_KEYS = frozenset({"not_scheduled_because"})


def _flat_diff(a: dict, b: dict, prefix: str = "") -> set:
    out = set()
    for k in set(a) | set(b):
        if not prefix and k in _SCHEDULING_KEYS:
            continue
        path = prefix + k
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out |= _flat_diff(va, vb, path + ".")
        elif va != vb:
            out.add(path)
    return out




# V1's taut share, a new predictor field under fibre_slack v1.
S17B_FIBRE_FIELDS = ("thetaTaut",)
# Design section 8 amendment a: under fibre_slack v1 the pending content
# (thetaPending), its taut share (thetaTaut) and the fibre's two passengers
# (thetaPendingB = C_B, thetaPendingP = q_P), and each Final, run ONE
# identical fixed-sweep entry (the solver refuses anything else). The same
# matrix and the same sweep count make theta_T <= theta_P exact and keep
# C_B/theta_T and q_P/theta_T exact convex combinations (red team 16).
# Three sweeps: >= batch 58's maximum at every G under `$c` (red-team-16-
# p2-iterations.out.txt: em67 672 of 1062 non-trivial P2 solves at 3).
S17B_FIXED_SWEEPS = 3
S17B_FIXED_SWEEP_ENTRY = (
    "{ solver smoothSolver; smoother symGaussSeidel; tolerance 0; relTol 0; "
    f"minIter {S17B_FIXED_SWEEPS}; maxIter {S17B_FIXED_SWEEPS}; }}")










# S18p, the em76 attribution probe (2026-10-01): research/designs/2026-10-01-
# S18p-em76-attribution-probe.md, rev. 2 (concept APPROVED by Alireza
# 2026-10-01, docket research/investigations/20261001-2159-run223505-s18-
# em76-tear/). LOG-ONLY. (child key, dictionary key, accepted values); the
# first is S18 as built, and is never written.
S18P_SWITCHES = (
    ("proj_site_log", "projSiteLog", ("no", "yes")),
)
# Design section 4: each child is em76 with proj_site_log yes, plus at most
# one S18 switch reverted (em72's value, or the key removed so the solver
# default applies). Checked on the CONTROLS entries, for every build.
S18P_PARENT_PAIRS = (
    ("em78_G4_s18p_ctl", "em76_G4_s18", {"proj_site_log"}, set()),
    ("em79_G4_s18p_Cnp", "em76_G4_s18",
     {"proj_site_log", "network_pressure_weight"}, set()),
)




def s18p_parent_check(children: dict) -> None:
    """S18p section 4 on the CONTROLS entries: each child differs from em76
    only in its named keys, and in each of them."""
    for ctl, ref, switches, run in S18P_PARENT_PAIRS:
        if ctl not in children:
            continue
        if ref not in children:
            raise ValueError(
                f"S18p parent check: {ctl}'s parent {ref} is missing from "
                f"the controls.")
        got = _flat_diff(children[ctl], children[ref])
        extra = sorted(got - (switches | run | {"child_id", "expectation"}))
        if extra:
            raise ValueError(
                f"S18p parent check: {ctl} differs from {ref} in "
                f"{', '.join(extra)} -- beyond its named keys "
                f"{', '.join(sorted(switches))} (S18p design section 4).")
        missing = sorted(switches - got)
        if missing:
            raise ValueError(
                f"S18p parent check: {ctl} does not differ from {ref} in its "
                f"named key {', '.join(missing)}.")






# S19, the block-coupled step (2026-10-04): research/designs/2026-10-03-S19-
# block-coupled-step.md, rev. 4, APPROVED by Alireza 2026-10-04 (docket
# research/investigations/20261003-0205-np-block-coupled-solve/). (child key,
# dictionary key, accepted values); the first is the shipped behaviour and an
# absent key means it, so every earlier dictionary is identical. tau_v_s is
# networkViscosityTau (s), read only under network_viscosity kelvinVoigt.
S19_SWITCHES = (
    ("momentum_pressure_coupling", "momentumPressureCoupling",
     ("segregated", "blockCoupled")),
    ("network_viscosity", "networkViscosity", ("none", "kelvinVoigt")),
)
S19_KEYS = ("momentum_pressure_coupling", "network_viscosity", "tau_v_s")

# S22 (research/designs/2026-10-07-S22-pardiso-block-solver.md rev. 2.1,
# APPROVED by Alireza 2026-10-07), sections 2.4-2.6 and 4: the block step's
# linear solver. (child key, dictionary key); written ONLY when the child
# names them, so every earlier dictionary is byte-identical and an absent key
# is the solver's default (blockLinearSolver pardiso, blockSolverThreads 0,
# blockSolverReproducible true). The solver refuses all three outside
# blockCoupled (S22 section 4); this generator refuses the same, and the one
# combination the solver leaves silently inert (blockSolverReproducible under
# umfpack, solver-engineer report 04 hand-off).
S22_BLOCK_SOLVERS = ("pardiso", "umfpack")
S22_KEYS = (
    ("block_linear_solver", "blockLinearSolver"),
    ("block_solver_threads", "blockSolverThreads"),
    ("block_solver_reproducible", "blockSolverReproducible"),
)
# A child that runs on another solver LIBRARY than the package's own (S22 section 7
# arm A0, BATCH 71; S23 section 7 arm (1'), BATCH 72). The runner compiles the named
# tag into a SEPARATE FOAM_USER_LIBBIN (run_main_case_colab.py --p2prime-solver /
# --s22-solver); the generator writes runs/<child>/child.env, which Allrun sources
# inside the child's subshell before foamRun and which prepends that directory to
# LD_LIBRARY_PATH. It is never a dictionary key. name -> (the environment variable the
# runner exports, what the library is, the design section that asks for it).
SOLVER_LIBRARIES = {
    "p2prime": ("PHASE_CLOT_P2PRIME_LIBBIN", "P2-prime", "S22 section 7, arm A0"),
    "s22": ("PHASE_CLOT_S22_LIBBIN", "S22 reference", "S23 section 7, arm (1')"),
}
SOLVER_LIBRARY_FILE = "libphaseClotAxisymmetricFluid.so"


def solver_library_env(name: str) -> str:
    """The text of runs/<child>/child.env for solver_library `name`. For p2prime it is
    byte-for-byte what BATCH 71 wrote (checked against the pre-B72 generator). For every
    library: an UNSET or EMPTY path is refused with `return 97`. For s22 a path that does not
    hold the library file is refused too, with `return 98` (the p2prime text is frozen)."""
    var, label, where = SOLVER_LIBRARIES[name]
    text = (
        f"# Written by create_cases.py for solver_library {name} ({where}).\n"
        "# Sourced by Allrun inside this child's subshell, before foamRun. `return` (not\n"
        "# `exit`) so Allrun can record the refusal and skip the solve.\n"
        f'if [ -z "${{{var}:-}}" ]; then\n'
        f'    echo "child.env: {var} is empty: the {label} library was not built;'
        f' refusing to run this child on the {"S22" if name == "p2prime" else label} library" >&2\n'
        "    return 97\n"
        "fi\n"
    )
    if name != "p2prime":
        text += (
            f'if [ ! -f "${{{var}}}/{SOLVER_LIBRARY_FILE}" ]; then\n'
            f'    echo "child.env: ${{{var}}} holds no {SOLVER_LIBRARY_FILE}: the {label} library was not built;'
            f' refusing to run this child on another library" >&2\n'
            "    return 98\n"
            "fi\n"
        )
    text += f'export LD_LIBRARY_PATH="${{{var}}}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"\n'
    return text


def s22_key_refusal(child: dict) -> None:
    """S22 section 4 on the CONTROLS entry, for EVERY child (the solver
    refuses each of these at construction; a child carrying one would abort
    on Colab, or run with the key silently inert)."""
    cid = child["child_id"]
    named = [k for k, _d in S22_KEYS if k in child]
    block = str(child.get("momentum_pressure_coupling", "segregated")) == "blockCoupled"
    if named and not block:
        raise ValueError(
            f"{cid}: {', '.join(named)} is read only under "
            f"momentum_pressure_coupling blockCoupled; under segregated the key "
            f"would be silently inert (S22 section 4).")
    if "block_linear_solver" in child:
        _s = child["block_linear_solver"]
        if not isinstance(_s, str) or _s not in S22_BLOCK_SOLVERS:
            raise ValueError(
                f"{cid}: block_linear_solver {_s!r} is not one of "
                f"{', '.join(S22_BLOCK_SOLVERS)} (S22 section 4).")
    if "block_solver_threads" in child:
        _n = child["block_solver_threads"]
        if isinstance(_n, bool) or not isinstance(_n, int) or _n < 0:
            raise ValueError(
                f"{cid}: block_solver_threads {_n!r} must be an integer >= 0 "
                f"(0 = MKL_NUM_THREADS, else the physical cores, under pardiso; "
                f"OpenBLAS untouched under umfpack; S22 section 4).")
    if "block_solver_reproducible" in child:
        _r = child["block_solver_reproducible"]
        if not isinstance(_r, bool):
            raise ValueError(
                f"{cid}: block_solver_reproducible {_r!r} must be true or false "
                f"(S22 section 4).")
        if child.get("block_linear_solver", "pardiso") == "umfpack":
            raise ValueError(
                f"{cid}: block_solver_reproducible is MKL's conditional-numerical-"
                f"reproducibility switch; under block_linear_solver umfpack it would "
                f"be silently inert (S22 section 2.6).")
    if "solver_library" in child:
        _l = child["solver_library"]
        if not isinstance(_l, str) or _l not in SOLVER_LIBRARIES:
            raise ValueError(
                f"{cid}: solver_library {_l!r} is not one of "
                f"{', '.join(SOLVER_LIBRARIES)} (S22 section 7).")
        if _l == "p2prime" and (child.get("block_linear_solver") != "umfpack" or any(
                k in child for k in ("block_solver_threads", "block_solver_reproducible"))):
            raise ValueError(
                f"{cid}: solver_library p2prime runs the P2-prime solver, which has "
                f"only UMFPACK and ignores the S22 keys: the child must name "
                f"block_linear_solver umfpack and no other S22 key, or the key "
                f"would be silently inert (S22 section 7, arm A0).")


# S23 (design 2026-10-07-S23-torn-network-debris rev. 3, section 2.6 and 4): the one
# parameter, R_c, the volumetric tear ratio of the terminal event (default 21.95 in the
# solver). (child key, dictionary key); written ONLY when the child names it, so every
# earlier dictionary is byte-identical and an absent key is the solver's default.
S23_KEYS = (("volumetric_tear_ratio", "volumetricTearRatio"),)


def s23_key_refusal(child: dict) -> None:
    """S23 section 4 on the CONTROLS entry, for EVERY child. The solver refuses a
    volumetricTearRatio that is not finite or is <= 1 (FatalError before step 1); a child
    carrying one would abort on Colab, so the generator refuses the same. The solver's
    second refusal (the key outside a fibreRelease build) cannot be reached: S21 built
    fibreRelease in (solver-engineer report 12, finding 6), so there is nothing to refuse.
    The key is also refused on a child that runs the S22 reference library, which does not
    read it (it would be silently inert)."""
    cid = child["child_id"]
    if "volumetric_tear_ratio" not in child:
        return
    _r = child["volumetric_tear_ratio"]
    if isinstance(_r, bool) or not isinstance(_r, (int, float)):
        raise ValueError(
            f"{cid}: volumetric_tear_ratio {_r!r} must be a number "
            f"(R_c, the volumetric tear ratio; S23 section 4).")
    if not math.isfinite(_r):
        raise ValueError(
            f"{cid}: volumetric_tear_ratio {_r!r} must be finite and exceed 1 "
            f"(S23 section 4: the solver refuses a non-finite R_c).")
    if not _r > 1:
        raise ValueError(
            f"{cid}: volumetric_tear_ratio {_r!r} must be finite and exceed 1 "
            f"(S23 section 4: the solver refuses R_c <= 1).")
    if child.get("solver_library") == "s22":
        raise ValueError(
            f"{cid}: volumetric_tear_ratio is read only by the S23 solver; "
            f"solver_library s22 runs the S22 reference library, so the key would be "
            f"silently inert (S23 section 7, arm (1')).")


# S23 rev. 4.1 (design 2026-10-07-S23-torn-network-debris section 13.2 items 4 and 5, APPROVED by
# Alireza 2026-10-07): R3 (network viscosity) is REQUIRED on the S23 build, because it is the term
# that holds the fringe (H-Keff, report 18) and fibrin's tan delta is 0.67-4.7 (S24 physics 02), so
# R3-off is the tan delta -> 0 sensitivity limit and not physics. The solver mirrors these with
# FatalError before step 1 (solver-engineer report 24, R1-R4 and the marker-with-kelvinVoigt
# refusal); this generator is the only place that bounds WHICH child may carry the marker.
#   * "the S23 build" is a child with NO solver_library key (the package solver). A child pinned to
#     solver_library s22 or p2prime runs another library and keeps the physics of its earlier runs;
#   * network_viscosity none (or absent: the dictionary default is none) is refused, except on the
#     one labelled sensitivity child, which must carry sensitivity_limit "tanDelta0" (the solver
#     writes it as `sensitivityLimit tanDelta0;`);
#   * the marker is refused on any other child, with kelvinVoigt, with any other value, and on a
#     child that runs another library (the key would be silently inert there);
#   * kelvinVoigt needs an explicit tau_v_s, and tau_v_s >= 5e-4 s. The comparison is the solver's
#     own `!(tau >= 5e-4)`: 5e-4 and 0.0005 are the same double and are accepted; 4.99e-4 is not.
#     The solver's default 2.5e-4 s is below the physics-assured 5e-4 and 1.5e-3 s.
S23_SENSITIVITY_CHILD = "S23v_G4_nor3"
S23_SENSITIVITY_VALUES = ("tanDelta0",)
S23_TAU_V_FLOOR_S = 5e-4


def s23_build_refusal(child: dict) -> None:
    """Design section 13.2 item 4, on the CONTROLS entry, for EVERY child."""
    cid = child["child_id"]
    lib = child.get("solver_library")
    s23_build = lib is None
    nv = str(child.get("network_viscosity", "none"))
    has_marker = "sensitivity_limit" in child
    if has_marker:
        _m = child["sensitivity_limit"]
        if not isinstance(_m, str) or _m not in S23_SENSITIVITY_VALUES:
            raise ValueError(
                f"{cid}: sensitivity_limit {_m!r} is not one of "
                f"{', '.join(S23_SENSITIVITY_VALUES)} (S23 section 13.2 item 4: the solver "
                f"refuses any other value).")
        if cid != S23_SENSITIVITY_CHILD:
            raise ValueError(
                f"{cid}: sensitivity_limit is allowed only on {S23_SENSITIVITY_CHILD}, the one "
                f"labelled tan delta -> 0 sensitivity arm (S23 section 13.2 item 4); on any other "
                f"child it would label an R3-off run as a sensitivity arm.")
        if nv == "kelvinVoigt":
            raise ValueError(
                f"{cid}: sensitivity_limit together with network_viscosity kelvinVoigt is refused: "
                f"the marker labels the R3-off limit only, so it cannot waive the tau_v floor "
                f"(S23 section 13.2 item 4, CE ratification of solver-engineer report 24).")
        if not s23_build:
            raise ValueError(
                f"{cid}: sensitivity_limit is read only by the S23 solver; solver_library "
                f"{lib} runs another library, so the key would be silently inert.")
    if not s23_build:
        return
    # The PHASE_CLOT_S23 populations (T, Pi, M) are the END-OF-STEP partition only while nothing
    # rewrites the partition after the step's block solve, which holds for nOuterCorrectors 1
    # (solver-engineer report 24, site list; design section 13.6 hand-off to the case engineer).
    # The default is 1 (fvSolution is rewritten per child); a child that names another is refused.
    if "n_outer_correctors" in child and child["n_outer_correctors"] != 1:
        raise ValueError(
            f"{cid}: n_outer_correctors {child['n_outer_correctors']!r} is refused on the S23 build: "
            f"the logged T, Pi and M are the end-of-step partition only with nOuterCorrectors 1 "
            f"(S23 section 13.6).")
    if nv == "none":
        if cid != S23_SENSITIVITY_CHILD:
            raise ValueError(
                f"{cid}: network_viscosity none (or absent: the default is none) is refused on the "
                f"S23 build: R3 is required (S23 section 13.2 item 4). Name network_viscosity "
                f"kelvinVoigt with tau_v_s >= {S23_TAU_V_FLOOR_S:g}, or pin solver_library s22 to "
                f"keep an earlier run's physics.")
        if not has_marker:
            raise ValueError(
                f"{cid}: network_viscosity none on the S23 build needs the explicit marker "
                f"sensitivity_limit \"tanDelta0\" (S23 section 13.2 item 4: the labelled tan delta "
                f"-> 0 sensitivity arm).")
        return
    if nv == "kelvinVoigt":
        if "tau_v_s" not in child:
            raise ValueError(
                f"{cid}: network_viscosity kelvinVoigt on the S23 build needs an explicit tau_v_s "
                f"(the solver default 2.5e-4 s is below the {S23_TAU_V_FLOOR_S:g} s floor; "
                f"S23 section 13.2 item 4).")
        _t = child["tau_v_s"]
        if not (_t >= S23_TAU_V_FLOOR_S):
            raise ValueError(
                f"{cid}: tau_v_s {_t!r} is below the {S23_TAU_V_FLOOR_S:g} s floor of the S23 build "
                f"(physics-assured 5e-4 and 1.5e-3 s; S23 section 13.2 item 4).")


def kept_option_refusal(child: dict, core_shell: bool) -> None:
    """Refuse, for EVERY child, a value or combination of the eight options that
    stay options (S21 section 1.2) which the solver refuses. The 78 built-in
    options have no refusal here: builtin_control_refusal() refuses the key.
    Each guard below is the shipped one with the built-in values substituted
    (contentModel true, networkPressureModel convexExcess, networkPressureReference
    deposited, eulerianSolid true, releasedCarrier material, elasticCoupling
    implicitTangent, faceConsistentElasticForce false, networkRupture fibreRelease)."""
    cid = child["child_id"]
    for _key, _dict_key, _allowed in (*S13_SWITCHES, *S15_SWITCHES, *S18P_SWITCHES,
                                      *S19_SWITCHES):
        if _key in child and str(child[_key]) not in _allowed:
            raise ValueError(
                f"{cid}: {_key} {str(child[_key])!r} is not one of "
                f"{', '.join(_allowed)}."
            )
    _npw = child.get("network_pressure_weight")
    if _npw in ("networkPresence", "contentRatio") and not float(
            child.get("residual_chi", 0.0)) > 0:
        raise ValueError(
            f"{cid}: network_pressure_weight {_npw} ramps to zero below "
            f"residual_chi of deposited network, which keeps the lumen free of "
            f"network pressure; residual_chi must be positive."
        )
    if child.get("material_label_basis") == "contentComposition" and not core_shell:
        raise ValueError(
            f"{cid}: material_label_basis contentComposition derives the "
            f"core/shell label from the core and shell contents; it needs a "
            f"core_shell series."
        )
    if "tau_v_s" in child:
        _t = child["tau_v_s"]
        if isinstance(_t, bool) or not isinstance(_t, (int, float)) or not _t > 0:
            raise ValueError(
                f"{cid}: tau_v_s {_t!r} must be a positive number of seconds "
                f"(eta_n = G tau_v/phi0; S19 section 12)."
            )
    block = str(child.get("momentum_pressure_coupling", "segregated")) == "blockCoupled"
    r3 = str(child.get("network_viscosity", "none")) == "kelvinVoigt"
    if r3 and not block:
        raise ValueError(
            f"{cid}: network_viscosity kelvinVoigt needs "
            f"momentum_pressure_coupling blockCoupled: R3 is unstable at the "
            f"design dt in every segregated placement (OI-44; S19 section 12)."
        )
    if r3 and _npw == "contentRatio":
        raise ValueError(
            f"{cid}: network_viscosity kelvinVoigt with "
            f"network_pressure_weight contentRatio is refused: C stays off "
            f"(S19 section 12)."
        )
    if block and _npw == "contentRatio":
        raise ValueError(
            f"{cid}: momentum_pressure_coupling blockCoupled with "
            f"network_pressure_weight contentRatio is refused (S19 section 2; "
            f"C off)."
        )
    if "tau_v_s" in child and not r3:
        raise ValueError(
            f"{cid}: tau_v_s is read only under network_viscosity "
            f"kelvinVoigt; without it the key would be silently inert (S19 "
            f"section 12)."
        )


# Keys the retired children used and no kept child sets; their read sites are deleted.
RETIRED_CONTROL_KEYS = frozenset({
    "linear_schemes", "platelet_solver_tolerance", "platelet_solver_maxiter",
    "s17_discriminator", "unlocalised_deposition_accepted", "clot_in_throat",
})


def dictionary_key_refusal(props: Path) -> None:
    """The written phaseClotProperties may carry no built-in option and no deleted
    parameter, whatever put it there (template, a switch table, a child override):
    the solver refuses each by name at construction (s21BuiltInKeys, s21DeletedKeys),
    so a child carrying one would abort on Colab. Statements only; comments are not
    entries."""
    text = re.sub(r"//[^\n]*", "", props.read_text())
    text = re.sub(r"(?s)/\*.*?\*/", "", text)
    text = re.sub(r"(?s)FoamFile\s*\{.*?\}", "", text, count=1)
    keys = re.findall(r"(?m)^\s*([A-Za-z]\w*)\s", text)
    built = sorted(set(keys) & set(BUILT_IN))
    dead = sorted(set(keys) & DELETED_KEYS)
    if built or dead:
        raise ValueError(
            f"{props}: the dictionary carries "
            + (f"built-in option(s) {built} (S21 section 1.2) " if built else "")
            + (f"deleted parameter(s) {dead} (s21DeletedKeys) " if dead else "")
            + "which the solver refuses by name at construction; remove them from the "
            "template or the writer.")


def builtin_control_refusal(controls: dict) -> None:
    """A control key naming a built-in option is refused: it would be silently
    inert (S21 section 1.2 and 3; pattern 7). The solver refuses the same keys
    in a dictionary."""
    camel = set(BUILT_IN)
    snake = {re.sub(r"(?<=[a-z0-9])([A-Z])", lambda m: "_" + m.group(1).lower(), k).lower()
             for k in camel}
    deleted_snake = {re.sub(r"(?<=[a-z0-9])([A-Z])", lambda m: "_" + m.group(1).lower(), k).lower(): k
                     for k in DELETED_KEYS}
    where = [("child " + c["child_id"], c) for c in controls["children"]]
    where += [("series " + n, s) for n, s in controls["series"].items()]
    where += [(n, controls[n]) for n in ("coupling", "material_fixed", "growth_parameters")]
    for label, block in where:
        for key in block:
            if key in RETIRED_CONTROL_KEYS:
                raise ValueError(
                    f"{label}: {key!r} was read only by a branch S21 deleted with the "
                    f"retired children; it would be silently inert (S21 section 3).")
            if key in DELETED_KEYS or key in deleted_snake or key in DELETED_CONTROL_KEYS:
                name = (key if key in DELETED_KEYS else DELETED_CONTROL_KEYS.get(key)
                        or deleted_snake[key])
                raise ValueError(
                    f"{label}: {key!r} names the deleted parameter {name}: S21 deleted its "
                    f"last read and the solver refuses it by name (s21DeletedKeys); "
                    f"remove the key (docket 20261005-1621 report 10).")
            if key in camel or key in snake:
                name = key if key in camel else next(
                    k for k in camel
                    if re.sub(r"(?<=[a-z0-9])([A-Z])", lambda m: "_" + m.group(1).lower(), k).lower() == key)
                raise ValueError(
                    f"{label}: {key!r} names the built-in option {name}, which the "
                    f"solver runs at {BUILT_IN[name]!r} and no longer reads; "
                    f"remove the key (S21 section 1.2)."
                )




# XU ET AL. 2017, FIGURE 5 (2026-09-30): research/designs/2026-09-29-case-
# xu2017-fig5.md, rev. 4, APPROVED by Alireza 2026-09-30 (docket research/
# investigations/20260929-1430-case-xu2017-partially-obstructive-clot/).
# Route A: a new series of this case on the soft S18. NOT a source-33 family
# member: that family refuses growth off, and Xu's clot is predefined and does
# not grow. It gets its own refusal instead, carrying every source-33 check
# that applies plus the ones this case adds.
XU_SERIES = ("xu2017_fig5",)
# Each Xu child is em76_G4_s18 (batch 60) except in these keys, checked on the
# CONTROLS entries at generation (as S18's parent rule), so a copy that drifts
# in a solver switch is refused rather than launched. Design section 1.
XU_PARENT = "em76_G4_s18"
XU_PARENT_DIFFERS_IN = {
    "series", "child_id", "expectation", "_every_number_and_its_source",
    "_what_a_failing_run_looks_like", "_why_this_child",
    # the material (DECISIONS 2026-09-29: the Xu-equivalent moduli; physics
    # 03 K3 permeability)
    "G_pa", "G_core_pa", "k0p_m2", "homogeneous_stiffness_endpoint",
    # the flow: one variable, the shear rate, with its declared velocity
    "wall_shear_rate_1_s", "mean_velocity_m_s",
    # the unbreakable bond and no core exemption (DECISIONS 2026-09-29)
    "adhesive_strength_pa", "release_core_exempt",
    # growth off and no platelet inflow (design section 1; red team 05 #3)
    "growth_rate_multiplier", "platelet_transport_only",
    "platelet_inlet_density_m3",
    # the 10 ms first leg (DECISIONS 2026-09-29)
    "end_time_s", "write_interval_s", "wall_budget_h",
}
# The four children differ from each other ONLY in the shear rate (and in the
# prose that names it: the expectation's header and the per-rate failing
# values are the same design section 4, addressed to the child).
XU_SIBLINGS_DIFFER_IN = {"child_id", "wall_shear_rate_1_s",
                         "mean_velocity_m_s", "_every_number_and_its_source",
                         "_why_this_child", "expectation",
                         "_what_a_failing_run_looks_like"}


def release_core_exempt_refusal(child: dict, series: dict) -> None:
    """releaseCoreExempt (solver default true, .C L724-726) is read in two
    places: updateReleasedIndicator() (L11252, returns at L11142 when
    advectReleased is off) and the erosion gate (L8772); both AND it with
    coreShell. Named on a child that has no core, or whose series does not
    advect released material, it would be silently inert -- refused."""
    if "release_core_exempt" not in child:
        return
    cid = child["child_id"]
    if not isinstance(child["release_core_exempt"], bool):
        raise ValueError(
            f"{cid}: release_core_exempt {child['release_core_exempt']!r} must "
            f"be true or false.")
    if not series.get("core_shell"):
        raise ValueError(
            f"{cid}: release_core_exempt is ANDed with coreShell in both of its "
            f"readers (.C L8772, L11252); this series has no core, so it would "
            f"be silently inert.")


def series_plasma_refusal(name: str, series: dict) -> None:
    """Series-level plasma (red team 05 #1): rho_kg_m3 and nu_m2_s together,
    positive and finite, or neither (then material_fixed, byte for byte)."""
    keys = [k for k in ("rho_kg_m3", "nu_m2_s") if k in series]
    if not keys:
        return
    if len(keys) != 2:
        raise ValueError(
            f"series {name}: declares {keys[0]} without its pair; a plasma is "
            f"a density AND a kinematic viscosity, and half of one would run "
            f"the other at the main case's value.")
    for k in keys:
        v = series[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (
                v > 0 and math.isfinite(v)):
            raise ValueError(f"series {name}: {k} {v!r} must be a positive number.")


def xu_series_refusal(child: dict, series: dict, cohesive_bound: float) -> None:
    """Every way the Xu Figure 5 case can be mis-wired and still run. Design
    sections 1-3; DECISIONS 2026-09-29 and 2026-09-30; red team 05."""
    cid = child["child_id"]
    # THE SILENT TRAP (numerics 02 Q4): a pressure-driven series runs every
    # child at one inlet pressure whatever its shear rate. Declared FALSE,
    # never inherited or omitted.
    if series.get("pressure_driven") is not False:
        raise ValueError(
            f"{cid}: the Xu series must declare pressure_driven false. Xu's "
            f"inlet is a velocity (ESM A.19), and a pressure-driven series "
            f"would run all four children at one inlet pressure whatever their "
            f"shear rate (numerics 02 Q4).")
    if "inlet_pressure_m2_s2" in series or "inlet_pressure_m2_s2" in child:
        raise ValueError(
            f"{cid}: inlet_pressure_m2_s2 is read only by a pressure-driven "
            f"inlet; under the velocity inlet it would be silently inert.")
    if not series.get("flow_axis"):
        raise ValueError(
            f"{cid}: the Xu series must declare its own flow_axis (the global "
            f"one tabulates U for a 60 um channel).")
    if series.get("wall_condition") != "clotWallAdhesion":
        raise ValueError(
            f"{cid}: must use clotWallAdhesion; a fixedValue patch emits no "
            f"PHASE_CLOT_DEBOND, so A7 (debondedFaces = 0) could not be read.")
    if child["adhesive_strength_pa"] != series["unbreakable_strength_pa"]:
        raise ValueError(
            f"{cid}: the bond must be unbreakable at "
            f"{series['unbreakable_strength_pa']} Pa (DECISIONS 2026-09-29).")
    if child["adhesive_strength_pa"] <= cohesive_bound:
        raise ValueError(
            f"{cid}: 'unbreakable' must exceed the cohesive bound "
            f"{cohesive_bound} Pa.")
    if not series.get("core_shell"):
        raise ValueError(f"{cid}: the Xu clot has a core; core_shell must be true.")
    for key in ("G_core_pa", "k0p_core_m2"):
        if child.get(key) is None:
            raise ValueError(f"{cid}: must declare {key}.")
    if not child["G_core_pa"] > child["G_pa"]:
        raise ValueError(f"{cid}: the core must be stiffer than the shell.")
    if not child["k0p_m2"] > child["k0p_core_m2"]:
        raise ValueError(f"{cid}: the shell must be more permeable than the core.")
    if child.get("homogeneous_stiffness_endpoint"):
        raise ValueError(
            f"{cid}: homogeneous_stiffness_endpoint is meaningful only with "
            f"G_core_pa == G_pa; here it would be silently inert.")
    if child.get("Gc_j_m2") is not None or child.get("Gc_core_j_m2"):
        raise ValueError(f"{cid}: must not declare a toughness (Xu has none).")
    if series.get("growth") is not False:
        raise ValueError(
            f"{cid}: growth must be declared false; Xu's clot is predefined "
            f"and does not grow.")
    if child.get("platelet_transport_only") is not True:
        raise ValueError(
            f"{cid}: platelet_transport_only must be true: with growth off the "
            f"transfer driver still binds platelets unless the kinetics are off "
            f"(.C L14018), and A4 requires contentTransferRate = 0 exactly.")
    if child.get("platelet_inlet_density_m3") != 0.0:
        raise ValueError(
            f"{cid}: platelet_inlet_density_m3 must be declared 0.0: no "
            f"platelet inflow (design section 2).")
    if child.get("release_core_exempt") is not False:
        raise ValueError(
            f"{cid}: release_core_exempt must be declared false: nothing about "
            f"the core is enforced (Alireza, 2026-09-29).")
    if not (True
            and child.get("bond_hold_dilation") == 4
            and True):
        raise ValueError(
            f"{cid}: the held layer must be the solver default as em72 "
            f"(bond_hold_carrier true, bond_hold_dilation 4, bond_hold_taper "
            f"false; Alireza, 2026-09-30).")
    if "wall_budget_h" not in child:
        raise ValueError(f"{cid}: must declare wall_budget_h (the stopAtClockTime).")
    clot = series.get("clot_override", {})
    if clot.get("shape") != "halfEllipseDome":
        raise ValueError(f"{cid}: the Xu clot is the halfEllipseDome shape.")
    if clot.get("material_width_m") is None:
        raise ValueError(
            f"{cid}: material_width_m must be declared (3e-6, the model's; the "
            f"2h default is x2.09 steeper at the corner, red team 05 #1).")
    if "rho_kg_m3" not in series or "nu_m2_s" not in series:
        raise ValueError(
            f"{cid}: the Xu series must declare its plasma (rho_kg_m3, "
            f"nu_m2_s): 4 cP, not the main case's 3.5 (red team 05 #1).")


# BATCH 68 (design 2026-10-05-case-free-shell-centred-s19b, rev. 3, APPROVED
# by Alireza 2026-10-05; docket research/investigations/20261005-0010-
# run042536-b67-s19b-validation/). Each child is its batch-63 parent plus
# EXACTLY these keys (switches, then run controls), checked on the CONTROLS
# entries for every build, filtered or not. "_"-prefixed keys are commentary.
#   fsf_*: S19b blockCoupled (default eliminate) + R3 kelvinVoigt at a
#     case-specific tau_v (section 2.4) + the centred placement (section
#     2.1) + 25 ms, the 22/16 h budget and the frame output (section 3);
#   the OI-48 control pair (section 2.3): the dense parent exactly (resting),
#     to 2 ms, block and segregated; wall_budget_h 3.0 each so slot 2
#     (loose <= 16 h, then the pair) stays <= 22 h inside the dense child's;
#     frames every 0.2 ms and a full write at 2 ms (O1 fixed at the root).
FSF_RUN_KEYS = {"end_time_s", "wall_budget_h", "write_interval_s",
                "frame_interval_s", "frame_fields"}
FSF_PARENT_PAIRS = (
    # The two retired batch-63 parents (fs_G4_kxu_v250, fs_G40_klo_v250) are gone; the
    # kept free-shell children are checked against each other and against BATCH 69.
    ("fsb_G40_klo_rest", "fs_G40_klo_rest_seg", {"momentum_pressure_coupling"}, set()),
    # BATCH 69 (design 2026-10-05-case-loose-40ms-r3-pair rev. 4 section 1): the
    # batch-68 loose child at 40 ms, a 22.0 h clock budget and nothing else.
    ("fsf_G4_kxu_r3_40", "fsf_G4_kxu_v250", set(), {"end_time_s", "wall_budget_h"}),
    # solver_library: the 12 R3-off children are pinned to the S22 library (S23 section 13.2 item 5).
    ("fsf_G4_kxu_nor3_40", "fsf_G4_kxu_r3_40", {"network_viscosity", "tau_v_s", "solver_library"}, set()),
    # V21-8: the seeded q < 0 child (S20 V5), 20 steps, no frames.
    ("fsf_G4_kxu_seedq", "fsf_G4_kxu_v250", {"seed_theta_peak"},
     FSF_RUN_KEYS),
    # V21-7: the identity twins, the parent with a 0.5 ms window.
    ("em78_id", "em78_G4_s18p_ctl", set(), {"end_time_s", "write_interval_s", "wall_budget_h"}),
    ("xu5_id", "xu5_g250", set(), {"end_time_s", "wall_budget_h"}),
    # V21-6 (S21 section 1.1, CE amendment of docket 20261005-1621 report 07): the 0.2 ms window
    # as a twin, because the runner has no end-time cap. Frames are the parent's.
    ("fsf_id", "fsf_G4_kxu_v250", set(), {"end_time_s", "write_interval_s", "wall_budget_h"}),
    # BATCH 71 (S22 section 7, the validation session): controls-only twins. Case 1
    # is fsf_G40_klo_v250 at 0.6 ms with fields every 0.2 ms; case 2 is
    # fsf_G4_kxu_r3_40 at 1.0 ms. Arms: A umfpack; E = A + OpenBLAS at 1 thread;
    # B = pardiso at 1 thread; C = pardiso (every default); D = C again; A0 = A on the
    # P2' library to 0.2 ms.
    ("S22v_G40_A", "fsf_G40_klo_v250", {"block_linear_solver"},
     {"end_time_s", "write_interval_s", "wall_budget_h"}),
    ("S22v_G40_E", "S22v_G40_A", {"block_solver_threads"}, set()),
    ("S22v_G40_B", "S22v_G40_A", {"block_linear_solver", "block_solver_threads"}, set()),
    ("S22v_G40_C", "S22v_G40_A", {"block_linear_solver"}, set()),
    ("S22v_G40_D", "S22v_G40_C", set(), set()),
    ("S22v_G40_A0", "S22v_G40_A", {"solver_library"}, {"end_time_s", "wall_budget_h"}),
    ("S22v_G4_A", "fsf_G4_kxu_r3_40", {"block_linear_solver"},
     {"end_time_s", "write_interval_s", "wall_budget_h"}),
    ("S22v_G4_C", "S22v_G4_A", {"block_linear_solver"}, set()),
    # BATCH 72 (S23 section 7, the validation): controls-only twins of the BATCH 69 loose
    # pair and the stiff control, each differing from its parent only in the clock budget
    # (the parents carry the 22 h budget of a serial UMFPACK run). (1') is (1) on the S22
    # library to 30 ms. No child names volumetric_tear_ratio: R_c is the solver's default.
    ("S23v_G4_r3", "fsf_G4_kxu_r3_40", set(), {"wall_budget_h"}),
    ("S23v_G4_r3_S22", "S23v_G4_r3", {"solver_library"}, {"end_time_s", "wall_budget_h"}),
    # Rev. 4.1: the parent is pinned to the S22 library, the twin runs the S23 build and carries the
    # sensitivity marker (the only child that may).
    ("S23v_G4_nor3", "fsf_G4_kxu_nor3_40", {"solver_library", "sensitivity_limit"}, {"wall_budget_h"}),
    ("S23v_G40", "fsf_G40_klo_v250", set(), {"wall_budget_h"}),
    # BATCH 73 (production, Alireza 2026-10-08: "both 40 ms. both in one session with 2
    # concurrency"; docket 20261008-1507-run081155-b72-s23-validation, case-engineer 05):
    # the loose clot repeats S23v_G4_r3 (already 40 ms, 11 h cap: no key differs, so the pair
    # names none); the stiff clot is S23v_G40 extended from 25 to 40 ms with an 18 h cap.
    ("P73_G4_r3_40", "S23v_G4_r3", set(), set()),
    ("P73_G40_40", "S23v_G40", set(), {"end_time_s", "wall_budget_h"}),
    # BATCH 74 (D74, the passive intact-network reference; design 2026-10-09-D74-passive-intact-reference.md
    # rev. 2 section 3, Alireza 2026-10-09: "Approve; run it alone"): the loose clot again, S23v_G4_r3 byte for
    # byte under a diagnostic id (already 40 ms, 11 h cap: no key differs, so the pair names none). The
    # passive field belongs to the BUILD; no dictionary key is added, read or refused for it.
    ("D74_refI", "S23v_G4_r3", set(), set()),
    # BATCH 75 (D75, the re-run of the D74 measurement on rev. 3.1; design 2026-10-09-D74-passive-intact-reference.md
    # rev. 3.1 sections 3-4, APPROVED by Alireza 2026-10-09: "Approve; rebuild and re-run (Recommended)"; docket
    # 20261009-0955-run100437-b74-d74-refI, case-engineer 04): D74_refI's dictionary again, S23v_G4_r3 byte for byte
    # under a NEW id (D74_refI stays as BATCH 74's record). Only the BUILD differs from BATCH 74's (the rupture
    # transfer and the emptying row); no dictionary key is added, read or refused for it.
    ("D75_refI", "S23v_G4_r3", set(), set()),
)


def fsf_parent_check(children: dict) -> None:
    """BATCH 68 design section 6 item 1 on the CONTROLS entries: each child
    differs from its batch-63 parent only in its named keys, and in each."""
    for ctl, ref, switches, run in FSF_PARENT_PAIRS:
        if ctl not in children:
            continue
        if ref not in children:
            raise ValueError(
                f"FSF parent check: {ctl}'s parent {ref} is missing from "
                f"the controls.")
        got = {k for k in _flat_diff(children[ctl], children[ref])
               if not k.startswith("_")}
        extra = sorted(got - (switches | run | {"child_id", "expectation"}))
        if extra:
            raise ValueError(
                f"FSF parent check: {ctl} differs from {ref} in "
                f"{', '.join(extra)} -- beyond its named keys "
                f"{', '.join(sorted(switches | run))} (BATCH 68 design "
                f"section 6).")
        missing = sorted((switches | run) - got)
        if missing:
            raise ValueError(
                f"FSF parent check: {ctl} does not differ from {ref} in its "
                f"named key {', '.join(missing)}.")


# THE FRAME FIELDS (BATCH 68 design section 3; red team 01 changes 3 and 8):
# a writeObjects function object writes ONLY these between the full writes.
# Derived from what each reader opens at EVERY written time, not from a list:
#   q        free_shell_timeseries.py field_states (L433, L438: the clot is
#            q < 0.5); clog_animation.py L57, L82; assess_case.py
#            transport_measurement's per-write loop (L4382 clot_geometry ->
#            phase_c L3868-3874: q, since this solver writes no c after 0/);
#            render_sweep_animation.py L454 (--g-shade mask) and the q views
#   p        clog_animation.py L81 (the pressure panel)
#   U        render_returned_run.sh L166-167 (flowLines, flowVectors)
#   Vs       render_returned_run.sh L60, L168-169 (Vs, clotLines, clotVectors)
#   clotDrag render_returned_run.sh L60 (clotDrag view)
#   poroPerm render_returned_run.sh L60 (poroPerm view)
#   Gfield   render_returned_run.sh L161, L211 (--g-shade on every view
#            but clotDrag) -> render_sweep_animation.py L453-454
#   Us       archive_openfoam_evidence.py L78 (a written time counts only if
#            it has Us) and L95-97 (the per-write max |Us| tail series)
# Not per-write: the archive's requested_fields are taken per field where
# present (archive L138-140); assess_child's final-state reads (L4813-4830)
# take dirs[-1], which is a FULL write on every stop (endTime is a multiple
# of write_interval_s, refused below; stopAtClockTime and stopAtFile act
# writeNow); xu2017_report, occlusion_measurement and tear_measurement are
# gated to other series (assess_case.py L4796, L4785-4790, L3995).
FRAME_FIELDS_REQUIRED = {
    "q": "free_shell_timeseries.py L433/L438; clog_animation.py L57/L82; "
         "assess_case.py L4382->L3868; render_sweep_animation.py L454",
    "p": "clog_animation.py L81",
    "U": "render_returned_run.sh L166-167",
    "Vs": "render_returned_run.sh L60, L168-169",
    "clotDrag": "render_returned_run.sh L60",
    "poroPerm": "render_returned_run.sh L60",
    "Gfield": "render_returned_run.sh L161, L211",
    "Us": "archive_openfoam_evidence.py L78, L95-97",
}
FRAME_KEYS = ("frame_interval_s", "frame_fields")


def _is_multiple(a: float, b: float) -> bool:
    n = a / b
    return round(n) >= 1 and abs(n - round(n)) <= 1e-9 * max(1.0, n)


def frame_output_refusal(child: dict, series: dict) -> None:
    """Every way the BATCH 68 frame output could silently lose what a reader
    needs, or lose the full final write (design section 3, section 6 item 2).
    Applied to EVERY child: the keys are refused where they would be inert."""
    cid = child["child_id"]
    named = [k for k in FRAME_KEYS if k in child]
    if not named:
        return
    if len(named) != len(FRAME_KEYS):
        raise ValueError(
            f"{cid}: frame_interval_s and frame_fields go together; "
            f"{named[0]} alone would be silently inert.")
    if "wall_budget_h" not in child or not series.get("step1_watcher"):
        raise ValueError(
            f"{cid}: the frame function object is written into the "
            f"wall-budget functions block of a step1_watcher series; here "
            f"frame_interval_s would be silently inert.")
    fields = child["frame_fields"]
    if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields) \
            or len(set(fields)) != len(fields):
        raise ValueError(f"{cid}: frame_fields must be a list of distinct field names.")
    missing = [f for f in FRAME_FIELDS_REQUIRED if f not in fields]
    if missing:
        raise ValueError(
            f"{cid}: frame_fields lacks {', '.join(missing)}, which "
            + "; ".join(f"{f} is read at every write by {FRAME_FIELDS_REQUIRED[f]}"
                        for f in missing)
            + " (BATCH 68 design section 3).")
    fi = child["frame_interval_s"]
    wi = child.get("write_interval_s", series["write_interval_s"])
    end = child.get("end_time_s", series["end_time_s"])
    if isinstance(fi, bool) or not isinstance(fi, (int, float)) or not fi > 0:
        raise ValueError(f"{cid}: frame_interval_s must be a positive number of seconds.")
    if not _is_multiple(wi, fi):
        raise ValueError(
            f"{cid}: write_interval_s {wi} is not a whole number of frames "
            f"({fi}); every full write must also be a frame time.")
    # THE FULL FINAL WRITE AT endTime. OF-13 writes at a step only when
    # writeTime_ is set (Time.C L1199-1210, adjustableRunTime: the write
    # index advances) or on a writeNow stop (L1252-1255); reaching endTime
    # sets neither (L1245-1260), so an endTime that is not a whole number of
    # write intervals ends on a FRAME-ONLY directory and every final-state
    # read comes back empty.
    if not _is_multiple(end, wi):
        raise ValueError(
            f"{cid}: end_time_s {end} is not a whole number of write "
            f"intervals ({wi}); the run would end without a full write.")


FREE_SHELL_SERIES = ("throat_free_shell",)


def free_shell_series_refusal(child: dict, series: dict) -> None:
    """The free, shell-only disc at the throat opening (design
    2026-10-02-case-free-shell-throat, rev. 3; DECISIONS 2026-10-02;
    red team 03 STOP, red team 04 PROCEED WITH CHANGES). Every way this
    series' central claim -- a clot held by nothing but drag, from step 1 --
    can be silently undone by a dictionary edit."""
    cid = child["child_id"]
    if series.get("wall_condition") != "clotWallAdhesion":
        raise ValueError(
            f"{cid}: must use clotWallAdhesion; a fixedValue (non-adhesive) "
            f"wall emits no PHASE_CLOT_DEBOND, so V1/V6 could not be read, "
            f"and a fixedValue Us pin on an UNBONDED clot is read by "
            f"solveSkeleton (design S5)."
        )
    if not series.get("wall_bond_free"):
        raise ValueError(
            f"{cid}: wall_bond_free must be true; with no injury entry the "
            f"condition latches every face under the clot as bonded at its "
            f"first evaluation (red team 03 F1)."
        )
    if not series.get("top_wall_adhesion"):
        raise ValueError(
            f"{cid}: top_wall_adhesion must be true; any wall patch that is "
            f"not clotWallAdhesion supports every clot cell touching it "
            f"(the weld branch, red team 03 F2)."
        )
    if series.get("core_shell"):
        raise ValueError(
            f"{cid}: core_shell must be false; a free shell fragment has no "
            f"core (design S3: 'core_shell false')."
        )
    # The clot the case is BUILT with: the series' override, then the
    # child's (the same merge, in the same order, as the geometry stage).
    clot = {k: v for k, v in series.get("clot_override", {}).items()
            if not k.startswith("_")}
    clot.update({k: v for k, v in child.get("clot_override", {}).items()
                 if not k.startswith("_")})
    if clot.get("shape") != "disc":
        raise ValueError(f"{cid}: the free-shell clot is the disc shape.")
    if clot.get("core_radius_m") is not None:
        raise ValueError(f"{cid}: a disc must not declare core_radius_m.")
    # THE TWO-WALL GAP RULE (design 2026-10-05-case-free-shell-centred-s19b
    # rev. 3, section 2.1; red team 01 change 9) replaces the refusal of
    # rests_on_wall false. The placement must be DECLARED either way: true
    # is P1 (a zero bottom gap, and only that); false is a free disc held to
    # the 3-epsilonC floor at BOTH walls -- the true distance to the curved
    # top wall and the bottom gap -- by the disc branch of the geometry
    # stage, which refuses either gap below 3.0 interface widths.
    if not isinstance(clot.get("rests_on_wall"), bool):
        raise ValueError(
            f"{cid}: rests_on_wall must be declared true (P1, resting on the "
            f"bottom wall) or false (free, >= 3 epsilonC from both walls).")
    if series.get("growth") is not False:
        raise ValueError(
            f"{cid}: growth must be false; a detached embolus fragment does "
            f"not grow (design S3, 'growth and platelet kinetics off by "
            f"their named keys')."
        )
    if child.get("platelet_transport_only") is not True:
        raise ValueError(
            f"{cid}: platelet_transport_only must be true, or the transfer "
            f"driver still binds platelets with growth off (.C L14018)."
        )
    if child.get("platelet_inlet_density_m3") != 0.0:
        raise ValueError(
            f"{cid}: platelet_inlet_density_m3 must be 0.0: no platelet "
            f"inflow (an embolus fragment's kinetics are off)."
        )
    if series.get("pressure_driven") is not True:
        raise ValueError(
            f"{cid}: this series is pressure-driven (Alireza, DECISIONS "
            f"2026-10-02): a velocity inlet cannot show occlusion."
        )
    if "inlet_pressure_m2_s2" not in series:
        raise ValueError(f"{cid}: must declare the series' inlet_pressure_m2_s2.")
    if not series.get("flow_axis"):
        raise ValueError(
            f"{cid}: must declare its own flow_axis (the global one tabulates "
            f"U for a 60 um channel and has no 250 1/s entry)."
        )
    if child.get("wall_shear_rate_1_s") != 250.0:
        raise ValueError(
            f"{cid}: Alireza asked for venous, 250 1/s (DECISIONS "
            f"2026-10-02)."
        )
    if not series.get("step1_watcher"):
        raise ValueError(
            f"{cid}: must declare step1_watcher true; editing controlDict "
            f"mid-run does nothing (runTimeModifiable false), so the stop "
            f"goes through stopAtFile and a background watcher, or a latch "
            f"failure costs the full wall budget instead of minutes (red "
            f"team 03 #3)."
        )
    if "wall_budget_h" not in child:
        raise ValueError(f"{cid}: must declare wall_budget_h.")
    if child.get("homogeneous_stiffness_endpoint"):
        raise ValueError(
            f"{cid}: homogeneous_stiffness_endpoint is meaningful only with "
            f"a core; this clot has none."
        )
    if child.get("release_core_exempt") is not None:
        raise ValueError(
            f"{cid}: release_core_exempt is ANDed with coreShell in both of "
            f"its readers; this series has no core, so it must be omitted, "
            f"not declared."
        )


def xu_parent_check(children: dict) -> None:
    """Each Xu child is em76 except in XU_PARENT_DIFFERS_IN, and the Xu
    siblings differ only in the shear rate. On the CONTROLS entries, for every
    build, filtered or not."""
    xu = sorted(c for c, v in children.items() if v["series"] in XU_SERIES)
    if not xu:
        return
    if XU_PARENT not in children:
        raise ValueError(f"Xu parent check: {XU_PARENT} is missing from the controls.")
    for cid in xu:
        extra = sorted(_flat_diff(children[cid], children[XU_PARENT])
                       - XU_PARENT_DIFFERS_IN)
        if extra:
            raise ValueError(
                f"Xu parent check: {cid} differs from {XU_PARENT} in "
                f"{', '.join(extra)} -- beyond the keys the design changes "
                f"(design section 1).")
    # xu5_id is the V21-7 identity twin of xu5_g250 (S21 section 1.1): it differs from the
    # parent only in its window, which the parent comparison above already bounds.
    sib = [c for c in xu if c != "xu5_id"]
    for cid in sib[1:]:
        extra = sorted(_flat_diff(children[cid], children[sib[0]])
                       - XU_SIBLINGS_DIFFER_IN)
        if extra:
            raise ValueError(
                f"Xu parent check: {cid} differs from {sib[0]} in "
                f"{', '.join(extra)}; the four children differ only in the "
                f"shear rate (BRIEF: one variable).")


def dome_floors(child_id: str, g: dict, clot: dict) -> dict:
    """The resolution floors of the halfEllipseDome, refused at generation on
    the shape the initialiser draws (mural_init, the same function):
      - the lumen gap above the dome is >= 3 epsilonC;
      - the core is buried: the clot's signed distance at every sample of the
        core's arc is <= -3 epsilonC. The clot's signed distance is convex
        (a convex region) and so is the core, so its maximum over the core is
        on the core's boundary; the base segment lies between the two arc
        end points. Hence every core cell has c <= tanh(-3) = -0.995 -- the
        initialiser's own 'exposed' guard (c > -0.9) cannot fire;
      - the attachment guard: the wall-row cell centre under the dome reaches
        c <= -0.9 (the initialiser refuses otherwise), and the top-row cells
        stay c >= 0.9 (a one-sided clot, not a bridge).
    Returns the margins for the record."""
    eps = g["interface_width_m"]
    samples = int(clot["signed_distance_samples"])
    x0, a, b = g["injury_centre_x_m"], clot["half_length_m"], clot["height_m"]
    if not math.isclose(2.0 * a, g["injury_length_m"], rel_tol=1e-9):
        raise ValueError(
            f"{child_id}: 2 * half_length_m {2.0 * a} must equal "
            f"injury_length_m {g['injury_length_m']} (the footprint is the injury).")
    for key in ("core_centre_x_m", "core_half_length_m", "core_height_m"):
        if clot.get(key) is None:
            raise ValueError(f"{child_id}: halfEllipseDome needs {key}.")
    gap = g["channel_height_m"] - b
    if gap < 3.0 * eps:
        raise ValueError(
            f"{child_id}: the lumen above the dome is {gap:.3g} m = "
            f"{gap / eps:.2f} interface widths, below the 3.0 floor.")
    cpx, cpy = mural_init.half_ellipse_arc(
        clot["core_centre_x_m"], clot["core_half_length_m"],
        clot["core_height_m"], samples)
    d_core = mural_init.half_ellipse_signed_distance(
        list(zip(cpx, cpy)), x0, a, b, samples)
    clearance = -max(d_core)
    if clearance < 3.0 * eps:
        raise ValueError(
            f"{child_id}: the core's nearest point is {clearance:.3g} m inside "
            f"the clot surface = {clearance / eps:.2f} interface widths, below "
            f"the 3.0 floor (a core must be buried in bulk clot).")
    h, nx = g["h_m"], int(g["nx"])
    wall = [((i + 0.5) * h, 0.5 * h) for i in range(nx)]
    top = [((i + 0.5) * h, g["channel_height_m"] - 0.5 * h) for i in range(nx)]
    c_wall = min(math.tanh(d / eps) for d in
                 mural_init.half_ellipse_signed_distance(wall, x0, a, b, samples))
    c_top = min(math.tanh(d / eps) for d in
                mural_init.half_ellipse_signed_distance(top, x0, a, b, samples))
    if c_wall > -0.9:
        raise ValueError(
            f"{child_id}: the dome does not attach: min c on the wall row is "
            f"{c_wall:.4f}, above the initialiser's -0.9 guard.")
    if c_top < 0.9:
        raise ValueError(
            f"{child_id}: the dome reaches the top wall (min c there "
            f"{c_top:.4f}); it must be one-sided.")
    return {"lumen_gap_m": gap, "lumen_gap_interface_widths": gap / eps,
            "core_clearance_m": clearance,
            "core_clearance_interface_widths": clearance / eps,
            "c_min_wall_row": c_wall, "c_min_top_row": c_top}














def validate_stage2_platelet(child: dict, content_operator: str) -> None:
    """Every Stage 2 misconfiguration that would RUN rather than fail.

    Each of these is refused at generation, on the machine that can fix it,
    rather than in Colab. The expensive ones are the last two: they do not
    crash, they produce a clean-looking null.
    """
    cid = child.get("child_id")



    convention = str('intrinsic')
    if convention not in ("intrinsic", "superficial"):
        raise SystemExit(
            "child %s: platelet_flux_convention must be 'intrinsic' or "
            "'superficial'; found %r. There is no default -- the wrong one "
            "is an EXACT factor of eps and is invisible in uniform fields."
            % (cid, convention)
        )

    if "platelet_number_density_m3" not in child:
        raise SystemExit(
            "child %s: platelet_content_model requires "
            "platelet_number_density_m3. nP is READ_IF_PRESENT and defaults "
            "to ZERO, and unlike thetaC/thetaH the solver does not seed it -- "
            "so omitting it runs a case with no platelets and reports a "
            "clean null instead of failing." % (cid,)
        )
    if float(child["platelet_number_density_m3"]) <= 0:
        raise SystemExit(
            "child %s: platelet_number_density_m3 must be > 0; found %r. "
            "Zero is the failure this check exists to prevent."
            % (cid, child["platelet_number_density_m3"])
        )
    if float(child.get("platelet_bind_rate", 0.0)) <= 0 and not child.get(
            "platelet_transport_only", False):
        raise SystemExit(
            "child %s: platelet_bind_rate must be > 0 for the accepted "
            "transfer to deposit anything. With lambda = 0 the run is a "
            "transport-only case and should say so via "
            "platelet_transport_only." % (cid,)
        )

    # ---- THE LOCALISATION MODE ----
    #
    # Validated against the same explicit list the solver validates, because a
    # child whose manifest claims "interface" but whose spelling slipped would
    # run the unlocalised path and be read as a localiser that did not work.
    localisation = str('interface')
    if localisation not in ("uniform", "interface"):
        raise SystemExit(
            "child %s: platelet_bind_localisation must be 'uniform' or "
            "'interface'; found %r" % (cid, localisation)
        )
    if localisation == "interface":
        if "platelet_adhesion_speed" not in child:
            raise SystemExit(
                "child %s: platelet_bind_localisation 'interface' requires "
                "platelet_adhesion_speed [m/s] (negative reuses the v2 "
                "closure adhesionSpeedEff)." % (cid,)
            )
        if float(child["platelet_adhesion_speed"]) == 0:
            raise SystemExit(
                "child %s: platelet_adhesion_speed is exactly 0, which "
                "deposits nothing anywhere." % (cid,)
            )

    # ---- THE UNLOCALISED-DEPOSITION CEILING ----
    #
    # The Stage 2 accepted transfer has NO spatial localiser: lambda is a
    # uniform scalar, dB applies in every cell, and the capacity term
    # (thetaMax - theta)/Y is LARGEST where there is no clot -- so deposition
    # is fastest in the lumen. The controlled specification requires the
    # opposite: AM2020-phase-clot-v1 (P) localises uptake by a surface delta,
    # delta_Gamma[(k(s) + V_n - u.n) w], and v2 carries "the interface-delta
    # construction" forward unchanged. The solver already implements that for
    # the LEGACY path -- "the platelet equation removes delta_Gamma*kEff*w"
    # -- using interfaceDelta_ = 0.5*|grad c|. Stage 2 kept the conservation
    # pairing and dropped the localisation.
    #
    # Until that is repaired, a Stage 2 child is only honest while its
    # deposition is SMALL: small enough that the unphysical lumen growth stays
    # far below the quantity being measured. The C1 weight discrepancy is
    # 7.41e-03 in volume fraction (run 20260909-182243, em27), so the ceiling
    # below sits about seven times under it.
    #
    # This is a KNIFE EDGE, which is why it is a refusal and not a comment:
    # em29 is safe at lambda = 42 because Y*lambda*T*nP = 4.2e-05. Raising
    # lambda to get "realistic" growth is the obvious next edit and is exactly
    # what must not happen silently.
    # REPAIRED 2026-09-15: the ceiling exists because the transfer had no
    # localiser. Under "interface" it now has the specified one, so the
    # ceiling does not apply -- the uniform-gain figure it computes is not
    # even the right quantity there (the rate is kEff*interfaceDelta_, which
    # is zero away from the interface). It still applies, unchanged, to every
    # "uniform" child, and that is now the only mode it can reach.
    end_time = float(child.get("end_time_s", 0.0))


def write_platelet_number_density(case: Path, value: float, inlet_value: float) -> None:
    """Write 0/nP with the PLANAR patch names this case's mesh actually has.

    WHY THIS EXISTS AND WHY IT REFUSES A MISSING VALUE. nP is READ_IF_PRESENT
    in the solver and defaults to ZERO, and nothing seeds it -- unlike thetaC
    and thetaH, which the solver seeds from theta_ref*chi when contentModel is
    on. So a Stage 2 case that forgets this field runs with no platelets:
    dB = 0, no deposition, no growth, and a log that reads as a clean result
    rather than a misconfiguration. That is the "multiply by zero" trap the
    carrier gate already hit once, where every arm returned identical fields
    because the quantity under test was zero everywhere.

    THE PATCH NAMES ARE THE PLANAR ONES, NOT THE TEMPLATE'S. The shared
    template is the axisymmetric straight tube -- inlet/outlet/wall/axis with
    wedge front and back -- and this case's mesh is the planar slab
    mural_gen.planar_mesh() writes: inlet/outlet/bottom/top with EMPTY front
    and back. Copying the template's boundary block would name patches that do
    not exist and fatal at run time. The entries below mirror 0/w, which is
    the same kind of field: platelets enter at the inlet at the reference
    concentration, everything else zeroGradient.
    """
    source = TEMPLATE / "0" / "w"
    if not source.is_file():
        raise SystemExit(f"missing template field for nP structure: {source}")
    text = source.read_text()
    text = re.sub(r"(?m)^(\s*)object\s+w;", r"\1object nP;", text)
    text = re.sub(r"(?m)^dimensions\s+\[[^\]]*\];",
                  "dimensions [0 -3 0 0 0 0 0];", text)
    text = re.sub(r"(?m)^internalField\s+[^;]+;",
                  f"internalField uniform {value!r};", text)
    target = case / "0" / "nP"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    mural_gen.field_boundary(target, f"""    inlet {{ type fixedValue; value uniform {inlet_value!r}; }}
    outlet {{ type zeroGradient; }}
    bottom {{ type zeroGradient; }}
    top {{ type zeroGradient; }}
    front {{ type empty; }}
    back {{ type empty; }}""")


def main() -> int:
    controls = json.loads(
        (CASE / "constant" / "mainCaseControls.json").read_text()
    )
    if RUNS.exists():
        raise ValueError("runs/ already exists; use a fresh immutable case copy")
    if not TEMPLATE.is_dir():
        raise ValueError(f"missing template: {TEMPLATE}")

    mural_controls = json.loads(
        (MURAL / "constant" / "muralGeometryControls.json").read_text()
    )
    gate_geometry = mural_controls["geometry"]
    gate_clot = mural_controls["clot"]
    coupling = controls["coupling"]
    fixed = controls["material_fixed"]
    growth = controls["growth_parameters"]
    series_spec = controls["series"]
    builtin_control_refusal(controls)

    # The Allrun must derive its child list from runs/matrix.json. A hard-coded
    # list is what silently skipped two of the finite-strain gate's children in
    # run 20260810-075435.
    allrun = (CASE / "Allrun").read_text()
    if "matrix.json" not in allrun:
        raise ValueError("Allrun must read its child list from runs/matrix.json")

    spec_by_id = {c["child_id"]: c for c in controls["children"]}
    if len(spec_by_id) != len(controls["children"]):
        raise ValueError("duplicate child_id")
    for child_id in spec_by_id:
        if child_id in allrun:
            raise ValueError(
                f"Allrun names the child '{child_id}' literally; it must "
                f"iterate matrix.json instead"
            )

    # Each series must contain what it claims to, or the report describes a
    # different experiment from the one that ran.
    for name, spec in series_spec.items():
        found = sum(1 for c in controls["children"] if c["series"] == name)
        if found != spec["children"]:
            raise ValueError(
                f"series {name} declares {spec['children']} children, found {found}"
            )

    # Wall condition and strength must agree, and -- since run 20260810-212211
    # -- any child that must MEASURE a traction has to carry the adhesion
    # condition, because a fixedValue patch emits no PHASE_CLOT_DEBOND at all.
    cohesive_bound = controls["adhesive_strength"]["upper_bound_pa"]
    for _name, _spec in series_spec.items():
        series_plasma_refusal(_name, _spec)
    for child in controls["children"]:
        spec = series_spec[child["series"]]
        welded = spec["wall_condition"] == "fixedValue"
        # A PER-CHILD INLET VELOCITY is read only against a series' own
        # flow_axis (the Xu series); anywhere else it would be silently inert
        # while reading as though it set the inlet.
        if "mean_velocity_m_s" in child and not spec.get("flow_axis"):
            raise ValueError(
                f"{child['child_id']}: mean_velocity_m_s is read only when the "
                f"series declares its own flow_axis; series "
                f"'{child['series']}' does not, so it would be silently inert.")
        release_core_exempt_refusal(child, spec)
        if child["series"] in XU_SERIES:
            # Before the S-stage refusals' generic tail and the source-33
            # branch: the Xu series is refused on its own terms, then falls
            # through to the S-stage refusals below like every other child.
            xu_series_refusal(child, spec, cohesive_bound)
        if child["series"] in FREE_SHELL_SERIES:
            free_shell_series_refusal(child, spec)
        frame_output_refusal(child, spec)
        kept_option_refusal(child, bool(spec.get("core_shell")))
        s22_key_refusal(child)
        s23_key_refusal(child)
        s23_build_refusal(child)

        # Only the cohesive-tear series may set its own toughness, and it is
        # explicitly a SWEEP that reports how far below physiology it had to
        # go. Anywhere else a per-child Gc is a silent change to the material
        # under a published result -- and in the source-33 child it would be
        # the difference between reproducing a mechanism and manufacturing
        # one.
        if child.get("Gc_j_m2") is not None and child["series"] != "cohesive_tear":
            raise ValueError(
                f"{child['child_id']}: only the cohesive_tear series may "
                f"override Gc; this child is in '{child['series']}'. If the "
                f"intent is to make something fail, that is tuning the answer, "
                f"not modelling it."
            )

        # THE CORE/SHELL SERIES. Every check here is a way its central claim --
        # "the core cannot fail, so what fails is the shell" -- can be
        # silently undone by a dictionary edit.
        # The stage-3 series ARE source-33 children -- same material, same
        # operating point, same core/shell claim -- so they must clear the same
        # eight checks. Listing them here rather than giving them a branch of
        # their own is deliberate: a separate branch is how a series quietly
        # acquires weaker validation than the one it was copied from.
        # THE XU SERIES was refused on its own terms above (xu_series_refusal,
        # which carries every source-33 check that applies to a clot that does
        # not grow) and has now cleared the S-stage refusals; the generic
        # tail below is for breakable demonstration bonds and would refuse
        # its unbreakable one for the wrong reason.
        if child["series"] in XU_SERIES:
            continue
        if child["series"] == "throat_embolus":
            if welded:
                raise ValueError(
                    f"{child['child_id']}: must use clotWallAdhesion. A "
                    f"fixedValue patch emits no PHASE_CLOT_DEBOND, so the run "
                    f"could not PROVE the wall held -- and that is the control "
                    f"on the whole experiment."
                )
            # THE PREMISE OF THIS GUARD WAS FALSIFIED ON 2026-08-24, AND IT
            # IS NARROWED RATHER THAN REMOVED.
            #
            # It was written because a breakable bond was believed to CAUSE
            # the whole-body sliding Dr Li rejected. Offline analysis of run
            # 20260824-220555 measured that sliding happening with an
            # UNBREAKABLE bond: adhesiveStrength 1e9 Pa, debondedFaces = 0 at
            # every one of ~15000 steps, bonded faces 55-94 all forty still
            # intact at 12 ms -- and the clot standing over the wound still
            # fell from 1093 cells to 6, its height over the footprint
            # collapsing 34 -> 2 cells between 6 and 7 ms. Whatever moved the
            # clot, the bond strength was not it, so refusing a breakable bond
            # does not protect against the thing this guard names.
            #
            # WHAT STILL NEEDS PROTECTING is a child that claims a detachment
            # while having no anchor at all: there a low strength really would
            # produce departure for the wrong reason, and it would be
            # indistinguishable from the real one. So the exemption is gated on
            # the child actually CARRYING an anchor -- the carrier hold
            # specifically, since a momentum penalty alone cannot reach the
            # flux that moves c -- and on nothing else. A child that lowers the
            # strength without one is refused exactly as before.
            anchored = (
                bool(True)
                and int(child.get("bond_hold_dilation", -1)) >= 0
            )
            if (
                child["adhesive_strength_pa"]
                != spec["unbreakable_strength_pa"]
                and not anchored
            ):
                raise ValueError(
                    f"{child['child_id']}: must be unbreakable at "
                    f"{spec['unbreakable_strength_pa']} Pa unless the child "
                    f"carries the bond-gated carrier hold (bond_hold_carrier "
                    f"with bond_hold_dilation >= 0). Without an anchor a "
                    f"breakable bond reproduces the whole-body sliding Dr Li "
                    f"rejected -- though run 20260824-220555 showed the "
                    f"ANCHOR, not the strength, was the real cause."
                )
            if not spec.get("core_shell") and not spec.get("core_shell_waiver"):
                raise ValueError(
                    f"{child['child_id']}: the series must set core_shell "
                    f"true, or the solver runs one material and the core that "
                    f"is supposed to survive does not exist. A series that "
                    f"genuinely cannot carry a core must say so in "
                    f"core_shell_waiver, with the arithmetic -- silence here "
                    f"reads as an oversight and is treated as one."
                )
            if not spec["growth"]:
                raise ValueError(
                    f"{child['child_id']}: growth must be ON in this series. "
                    f"It exists because Dr Li asked to see deformation and "
                    f"detachment AS THE CLOT GROWS."
                )
            if spec.get("core_shell"):
                for key in ("G_core_pa", "k0p_core_m2"):
                    if child.get(key) is None:
                        raise ValueError(
                            f"{child['child_id']}: must declare {key}. A missing "
                            f"core property silently defaults to the shell value, "
                            f"which makes the clot one soft material with no "
                            f"contrast and no mechanism."
                        )
            else:
                # A WAIVED CORE MUST BE DECLARED AND ARGUED, NOT INFERRED FROM
                # AN ABSENT KEY. Run 20260817-210800 refused the Li-dome
                # children at the initialiser: a core/shell needs the SHELL
                # ANNULUS to be 3-4 interface widths thick, and on an 8.5 um
                # dome at epsilonC = 3 um, 3*epsilonC = 9 um exceeds the whole
                # radius -- so no core radius works, not even zero. That is a
                # real geometric limit and the honest response is to run the
                # dome homogeneous and say so, not to shrink the core until
                # something passes.
                if not spec.get("core_shell_waiver"):
                    raise ValueError(
                        f"{child['child_id']}: this series has core_shell off "
                        f"and no core_shell_waiver. A source-33 child without a "
                        f"core has given up the series' central claim -- 'the "
                        f"core cannot fail, so what fails is the shell' -- and "
                        f"must say in the controls file WHY, with the arithmetic."
                    )
                for key in ("G_core_pa", "k0p_core_m2"):
                    if child.get(key) is not None:
                        raise ValueError(
                            f"{child['child_id']}: core_shell is waived but "
                            f"{key} is set. The solver would ignore it and the "
                            f"controls file would read as though a core existed."
                        )
            # NO TOUGHNESS CONTRAST, AND THAT IS THE POINT. Source 33 has no
            # fracture toughness anywhere in its model; their shell is dragged
            # off by the flow. A Gc set low enough for AT2 to fire would make
            # the result an artefact of the number chosen, so this series is
            # refused one.
            if child.get("Gc_j_m2") is not None or child.get("Gc_core_j_m2"):
                raise ValueError(
                    f"{child['child_id']}: must NOT declare a toughness. It "
                    f"inherits the physiological {fixed['Gc_j_m2']} J/m2 in "
                    f"both regions. Source 33 has no toughness at all, and a "
                    f"Gc chosen so that the damage criterion fires is the "
                    f"manufactured tear this series was rebuilt to remove."
                )
            # The two contrasts source 33 actually supplies, each checked in
            # the direction their mechanism needs. Skipped -- not weakened --
            # on a child whose core is waived: there is no contrast to check
            # because there is no core, and the waiver above already refused
            # to let that pass silently.
            # INVERSION IS ALWAYS REFUSED. EQUALITY MUST BE DECLARED.
            #
            # The check below used to be `<=`, which refused both. Inversion
            # (core softer than shell) is a genuine error: strain goes as 1/G,
            # so the core would deform more than the shell and source 33's
            # result turns over. Equality is not that -- it is the HOMOGENEOUS
            # ENDPOINT of a shell-stiffness sweep, where the shell has been
            # raised to meet the core and the clot becomes one material. That
            # is the far end of the sweep Dr Li asked for on 2026-09-01, and
            # it is the child that says how much of the behaviour came from
            # the core/shell CONTRAST rather than from the absolute modulus.
            #
            # It still may not happen by accident: a child that lands on
            # equality because someone edited one number and not the other
            # looks exactly like a child that meant it. So equality is allowed
            # only with an explicit declaration, and the controls file has to
            # carry the word.
            if spec.get("core_shell") and child["G_core_pa"] < child["G_pa"]:
                raise ValueError(
                    f"{child['child_id']}: the core must be STIFFER than the "
                    f"shell ({child['G_core_pa']} vs {child['G_pa']} Pa), as "
                    f"in source 33 (40 against 4). Strain goes as 1/G, so a "
                    f"core softer than its shell deforms more and the paper's "
                    f"result inverts."
                )
            if (
                spec.get("core_shell")
                and child["G_core_pa"] == child["G_pa"]
                and not child.get("homogeneous_stiffness_endpoint")
            ):
                raise ValueError(
                    f"{child['child_id']}: core and shell moduli are EQUAL "
                    f"({child['G_pa']} Pa), so this clot has no stiffness "
                    f"contrast and source 33's core/shell mechanism is gone. "
                    f"That is legitimate as the endpoint of a shell-stiffness "
                    f"sweep, but it must be declared: set "
                    f"homogeneous_stiffness_endpoint true on the child and say "
                    f"in _every_number_and_its_source why a homogeneous clot "
                    f"is the intended configuration."
                )
            if spec.get("core_shell") and child["k0p_m2"] <= child["k0p_core_m2"]:
                raise ValueError(
                    f"{child['child_id']}: the shell must be MORE permeable "
                    f"than the core ({child['k0p_m2']} vs "
                    f"{child['k0p_core_m2']} m2), as in source 33 (1e-11 "
                    f"against 1e-14). Their whole permeability result is that "
                    f"the permeable shell is what goes."
                )
            if (
                child.get("growth_rate_multiplier", 1.0) == 1.0
                and not child.get("growth_multiplier_waiver")
            ):
                raise ValueError(
                    f"{child['child_id']}: declare growth_rate_multiplier. At "
                    f"the physiological alpha this clot grows 1.9e-9 m over "
                    f"the run -- a four-hundredth of a cell -- so 'growth is "
                    f"on' would be true and invisible at the same time."
                )
            continue

        if welded and child["adhesive_strength_pa"] is not None:
            raise ValueError(f"{child['child_id']}: welded, but a strength is set")
        if not welded and child["adhesive_strength_pa"] is None:
            raise ValueError(f"{child['child_id']}: adhesive, but no strength")
        if not welded:
            # The demonstration values must stay BELOW the cohesive strength,
            # or the clot tears instead of peeling and the picture shows the
            # wrong failure mode.
            if child["adhesive_strength_pa"] >= cohesive_bound:
                raise ValueError(
                    f"{child['child_id']}: strength at or above the cohesive "
                    f"bound; the clot would tear rather than peel"
                )

    if fixed["Gc_j_m2"] > 100.0:
        raise ValueError(
            "the main case restores the PHYSIOLOGICAL Gc; an unreachable value "
            "would exclude cohesive failure by construction"
        )

    # SERIES FILTER, for debugging turnaround.
    #
    # The full matrix is 19 children and about three and a half hours. The
    # transport behaviour is being iterated on right now and lives entirely in
    # the three demonstration children, which cost about twenty minutes. The
    # filter is a COMMAND-LINE argument rather than a controls edit, so a
    # partial run can never be mistaken for the full one: the controls file on
    # disk is unchanged, and the matrix records what was actually built.
    #
    # Validation above runs against the WHOLE controls file regardless, so
    # filtering can never smuggle an invalid child past a refusal.
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    selected = [
        c for c in controls["children"]
        if not only or c["series"] in only or c["child_id"] in only
    ]
    if only and not selected:
        raise ValueError(
            f"filter {only} matched no child; series are "
            f"{sorted(series_spec)}"
        )
    if only:
        print(f"FILTERED RUN: {len(selected)} of {len(controls['children'])} "
              f"children, matching {only}")

    RUNS.mkdir()
    matrix = []
    for child in selected:
        child_id = child["child_id"]
        series = series_spec[child["series"]]

        # PER-SERIES GEOMETRY. Everything without an override gets the mural
        # gate's geometry byte for byte, so no existing child moves. The
        # cohesive-tear series needs its own because a smaller clot needs a
        # smaller epsilonC needs a finer mesh -- see the override's own _why.
        g = dict(gate_geometry)
        clot = dict(gate_clot)
        g.update({k: v for k, v in series.get("geometry_override", {}).items()
                  if not k.startswith("_")})
        clot.update({k: v for k, v in series.get("clot_override", {}).items()
                     if not k.startswith("_")})
        # CHILD-LEVEL OVERRIDES, added for the throat-occlusion study, where
        # the whole point is that children differ in clot radius and clot
        # POSITION while sharing one geometry. Applied after the series ones
        # so a child wins, and "_"-prefixed keys stay commentary as everywhere
        # else. Every existing child omits both keys and is untouched.
        g.update({k: v for k, v in child.get("geometry_override", {}).items()
                  if not k.startswith("_")})
        clot.update({k: v for k, v in child.get("clot_override", {}).items()
                     if not k.startswith("_")})
        cells = g["nx"] * g["ny"] * g["nz"]
        if cells != g["expected_cell_count"]:
            raise ValueError(
                f"{child_id}: geometry declares {g['expected_cell_count']} "
                f"cells but the mesh has {cells}"
            )
        # The interface must stay resolved. Four cells is what every case in
        # this project has used; three is where the tanh starts to wobble and
        # |grad c| stops being a reliable interface locator.
        # THE SHELL ANNULUS MUST CONTAIN BULK MATERIAL, checked HERE rather
        # than discovered in Colab.
        #
        # Run 20260817-210800 lost both Li-dome children to the initialiser's
        # own guard -- "max chi over shell cells is 0.902, below 0.95" -- after
        # blockMesh had already run, in a session that had to be paid for. The
        # condition is arithmetic on numbers this file already holds, so there
        # is no reason it could not have been refused before the package was
        # ever built.
        #
        # A core/shell clot has a shell of thickness R - core_radius. If that
        # is thinner than about 3 interface widths the annulus lies inside the
        # clot/blood interface, never reaches chi = 1, and the core's modulus
        # and permeability get applied to material that is not solid.
        #
        # On Dr Li's 8.5 um dome at epsilonC = 3 um, 3*epsilonC = 9 um EXCEEDS
        # THE WHOLE RADIUS, so no core radius satisfies it -- not even zero.
        # Resolving a core there needs epsilonC <= 2 um and h <= 0.5 um, i.e.
        # 300x90 cells and a smaller dt: about 3.4x the work. That is why the
        # stage-3 Li children waive the core instead.
        if series.get("core_shell") and clot.get("shape") == "semisphere":
            # A SEATED core is 0.5R off the clot's centre, so its clearance is
            # R - sqrt(r^2 + 0.25 R^2), NOT R - r. Using the concentric formula
            # on a seated core overstates the shell by several microns and
            # would wave through exactly the configuration the initialiser then
            # refuses in a paid Colab session. See initialise_cases.py.
            if clot.get("core_seated"):
                shell_thickness = clot["radius_m"] - math.hypot(
                    clot.get("core_radius_m", 0.0), 0.5 * clot["radius_m"]
                )
            else:
                shell_thickness = (
                    clot["radius_m"] - clot.get("core_radius_m", 0.0)
                )
            widths = shell_thickness / g["interface_width_m"]
            if widths < 3.0:
                raise ValueError(
                    f"{child_id}: the shell annulus is {shell_thickness:.3g} m "
                    f"= {widths:.2f} interface widths, below the 3.0 the "
                    f"initialiser requires (it checks max chi over shell cells "
                    f">= 0.95). radius {clot['radius_m']:.3g} m, core "
                    f"{clot.get('core_radius_m', 0.0):.3g} m, epsilonC "
                    f"{g['interface_width_m']:.3g} m. Note 3*epsilonC = "
                    f"{3 * g['interface_width_m']:.3g} m: if that exceeds the "
                    f"radius, NO core works and the core must be waived or "
                    f"epsilonC reduced with h refined to match."
                )

        # THE XU DOME (design 2026-09-29-case-xu2017-fig5, section 1): its
        # floors -- lumen gap, buried core, attachment -- refused here on the
        # shape the initialiser draws. Recorded in child.json below.
        dome_margins = None
        if clot.get("shape") == "halfEllipseDome":
            dome_margins = dome_floors(child_id, g, clot)

        cells_across = g["interface_width_m"] / g["h_m"]
        if cells_across < 4.0 - 1e-9:
            raise ValueError(
                f"{child_id}: epsilonC = {g['interface_width_m']} on h = "
                f"{g['h_m']} is {cells_across:.2f} cells across the interface; "
                f"four is the floor this project has held to"
            )
        if not math.isclose(g["h_m"], g["channel_height_m"] / g["ny"],
                            rel_tol=1e-9):
            raise ValueError(f"{child_id}: h_m disagrees with channel_height/ny")
        if not math.isclose(g["h_m"], g["channel_length_m"] / g["nx"],
                            rel_tol=1e-9):
            raise ValueError(f"{child_id}: h_m disagrees with channel_length/nx")

        # A THROAT CHILD KEEPS BOTH INVARIANTS ABOVE. The mesh blockMesh
        # builds is the ordinary uniform block; apply_throat.py bends it
        # afterwards by scaling y, which changes no cell count and no
        # streamwise spacing. What it does change is the throat's dy, and
        # that is what sets dt -- maxDeltaT() uses the CUBE ROOT OF THE
        # MINIMUM CELL VOLUME, so a 20 um throat at ny = 70 takes dt from
        # 2.045e-06 to 1.508e-06 s. Checked, not assumed: it is the whole
        # cost argument for this study.
        if g.get("throat_height_m") is not None:
            for key in ("taper_start_x_m", "taper_length_m", "throat_height_m"):
                if g.get(key) is None:
                    raise ValueError(
                        f"{child_id}: a throat child needs {key}"
                    )
            hth = g["throat_height_m"]
            if not 0.0 < hth < g["channel_height_m"]:
                raise ValueError(
                    f"{child_id}: throat_height_m {hth} must lie strictly "
                    f"between 0 and the channel height {g['channel_height_m']}"
                )
            x1 = g["taper_start_x_m"]
            x2 = x1 + g["taper_length_m"]
            if x2 > g["channel_length_m"]:
                raise ValueError(
                    f"{child_id}: the taper ends at {x2} m, past the end of "
                    f"the {g['channel_length_m']} m channel"
                )
            # THE CLOT'S STREAMWISE EXTENT, for either shape. A semisphere is
            # bounded by its radius; a cap by half its injury length. Reading
            # radius_m unconditionally would raise KeyError on a cap child
            # rather than checking it, which is the wrong kind of failure for
            # a geometry guard.
            if clot.get("shape") in ("semisphere", "disc"):
                half_extent = clot["radius_m"]
            else:
                half_extent = 0.5 * g["injury_length_m"]
            # A DISC'S CENTRE IS ITS OWN KEY (centre_x_m), not
            # injury_centre_x_m -- the free-shell series places it by the
            # case design (P1), not by the injury band every other throat
            # shape reuses. Falling back to injury_centre_x_m here would
            # silently check the wrong window against the taper.
            clot_centre_x = (
                clot["centre_x_m"] if clot.get("shape") == "disc"
                else g["injury_centre_x_m"]
            )
            clot_x_min = clot_centre_x - half_extent
            clot_x_max = clot_centre_x + half_extent

            if clot.get("shape") == "disc":
                # A FREE DISC PLACED IN THE TAPER (design
                # 2026-10-02-case-free-shell-throat, rev. 3, S2-S3): neither
                # of the two placements below applies. The lodged placement
                # (clot_in_throat) demands the clot sit entirely past x2; the
                # "clear of the narrowing" placement below demands it sit
                # entirely before x1. This design deliberately places the
                # disc AT x = 154.9 um, inside the 120-220 um taper, close to
                # the opening (Alireza, DECISIONS 2026-10-02). The floor that
                # applies is the TRUE distance to the CURVED top wall at
                # every x the disc spans (apply_throat.channel_height, the
                # same formula the mesh is bent with), not the flat
                # channel_height_m the other two placements assume.
                if clot_x_min < 0.0 or clot_x_max > g["channel_length_m"]:
                    raise ValueError(
                        f"{child_id}: the disc spans x = {clot_x_min:.3g}-"
                        f"{clot_x_max:.3g} m, outside the "
                        f"{g['channel_length_m']:.3g} m channel"
                    )
                disc_cx = clot["centre_x_m"]
                disc_cy = clot["centre_y_m"]
                disc_r = clot["radius_m"]
                epsilon = g["interface_width_m"]
                # THE TRUE (EUCLIDEAN) MINIMUM DISTANCE from the disc's
                # centre to the SAMPLED top-wall curve, over the WHOLE curve
                # -- not the vertical distance directly above the centre,
                # which is wrong wherever the wall slopes (`ce-placement-
                # 250.py`'s top_gap, reproduced term for term so the floor
                # checked here is the floor the placement was chosen
                # against). h_m-scale sampling (0.1 um step here, matching
                # the probe) resolves the 0.286 um throat cell comfortably.
                n_samples = 3201
                step = g["channel_length_m"] / (n_samples - 1)
                nearest = min(
                    math.hypot(i * step - disc_cx,
                               apply_throat.channel_height(
                                   i * step, g["channel_height_m"], hth, x1, x2
                               ) - disc_cy)
                    for i in range(n_samples)
                )
                top_gap = nearest - disc_r
                widths = top_gap / epsilon
                if widths < 3.0:
                    raise ValueError(
                        f"{child_id}: the disc's narrowest true gap to the "
                        f"top wall is {top_gap:.3g} m = {widths:.2f} "
                        f"interface widths, below the 3.0 floor (epsilonC "
                        f"{epsilon:.3g} m)"
                    )
                # THE BOTTOM GAP. Zero is allowed ONLY under a declared
                # rests_on_wall ("rests_on_wall true allows a zero bottom gap
                # and only that", design S2); any other bottom clearance
                # holds the same 3*epsilon floor as the top.
                bottom_gap = disc_cy - disc_r
                if clot.get("rests_on_wall"):
                    if not math.isclose(bottom_gap, 0.0, abs_tol=1e-9):
                        raise ValueError(
                            f"{child_id}: rests_on_wall is declared but the "
                            f"disc's bottom gap is {bottom_gap:.3g} m, not "
                            f"zero; centre_y_m must equal radius_m exactly "
                            f"for a wall-resting disc"
                        )
                elif bottom_gap / epsilon < 3.0:
                    raise ValueError(
                        f"{child_id}: the disc's bottom gap is "
                        f"{bottom_gap:.3g} m = {bottom_gap / epsilon:.2f} "
                        f"interface widths, below the 3.0 floor, and "
                        f"rests_on_wall is not declared"
                    )
            else:
                # The clot must start in the FULL-HEIGHT section. If the taper
                # began under the clot, the initialised clot would be cut by
                # the sloping wall and its blockage at t=0 would not be the
                # value every calibrated number was derived for.
                if clot_x_max >= x1:
                    raise ValueError(
                        f"{child_id}: the clot reaches x = {clot_x_max:.3g} m "
                        f"but the taper starts at {x1:.3g} m; the clot must "
                        f"be clear of the narrowing at t = 0"
                    )
            # THE SKEW THIS COSTS. Scaling y tilts the top of each cell by
            # dH/dx*h against a cell height of H/ny; the cosine puts its
            # steepest slope at the taper midpoint. The case ships
            # nNonOrthogonalCorrectors 0, which is right for the orthogonal
            # Cartesian mesh every other child runs on and NOT right here, so
            # throat children raise it -- see the fvSolution edit below.
            slope_max = (
                (g["channel_height_m"] - hth) * math.pi
                / (2.0 * g["taper_length_m"])
            )
            dy_mid = 0.5 * (g["channel_height_m"] + hth) / g["ny"]
            skew_deg = math.degrees(math.atan(slope_max * g["h_m"] / dy_mid))
            if skew_deg > 45.0:
                raise ValueError(
                    f"{child_id}: the taper skews cells by {skew_deg:.1f} deg "
                    f"(max wall slope {math.degrees(math.atan(slope_max)):.1f} "
                    f"deg). Lengthen taper_length_m: skew falls as 1/L."
                )

        gamma = child["wall_shear_rate_1_s"]
        u_mean = gamma * g["channel_height_m"] / 6.0
        override = series.get("geometry_override", {})
        # A SERIES WITH ITS OWN FLOW AXIS (the Xu series, 2026-09-30): each
        # child declares its OWN inlet velocity, and both the rate and the
        # velocity must sit on the series' axis. Numerics 02 Q4's trap: one
        # series-level velocity would run all four Xu children at the 1000
        # 1/s inlet. The check moves to the series' axis; it does not weaken
        # -- the geometry check below still binds U = gammaDot H / 6.
        series_axis = series.get("flow_axis")
        if series_axis:
            table = dict(zip(series_axis["wall_shear_rate_1_s"],
                             series_axis["mean_velocity_m_s"]))
            if len(table) != len(series_axis["wall_shear_rate_1_s"]) or len(
                    series_axis["wall_shear_rate_1_s"]) != len(
                    series_axis["mean_velocity_m_s"]):
                raise ValueError(
                    f"{child_id}: series '{child['series']}' flow_axis must "
                    f"pair each shear rate with one velocity")
            if gamma not in table:
                raise ValueError(
                    f"{child_id}: wall shear rate {gamma} 1/s is not on series "
                    f"'{child['series']}''s flow_axis "
                    f"{series_axis['wall_shear_rate_1_s']}; declare it or fix "
                    f"the typo")
            if "mean_velocity_m_s" in override:
                raise ValueError(
                    f"{child_id}: series '{child['series']}' declares a "
                    f"flow_axis AND a series-level mean_velocity_m_s; one "
                    f"inlet velocity for every child is the trap the axis "
                    f"exists to close")
            if "mean_velocity_m_s" not in child:
                raise ValueError(
                    f"{child_id}: series '{child['series']}' declares its own "
                    f"flow_axis, so each child must declare its "
                    f"mean_velocity_m_s; a missing one must not fall back to "
                    f"any other child's inlet")
            if not math.isclose(child["mean_velocity_m_s"], table[gamma],
                                rel_tol=1e-9):
                raise ValueError(
                    f"{child_id}: mean_velocity_m_s "
                    f"{child['mean_velocity_m_s']} is not the flow_axis value "
                    f"{table[gamma]} for gammaDot {gamma}")
            expected = child["mean_velocity_m_s"]
        else:
            # Both axes are consulted: the venous flow_axis and the arterial
            # deformation_axis. A shear rate on NEITHER is a typo, not a new
            # operating point, and must not be silently accepted.
            declared = {}
            for axis in ("flow_axis", "deformation_axis"):
                a = controls[axis]
                declared.update(
                    dict(zip(a["wall_shear_rate_1_s"], a["mean_velocity_m_s"]))
                )
            if gamma not in declared:
                raise ValueError(
                    f"{child_id}: wall shear rate {gamma} 1/s is on neither "
                    f"flow_axis nor deformation_axis; declare it or fix the typo"
                )
            expected = declared[gamma]
        # A SERIES WITH ITS OWN CHANNEL HEIGHT HAS ITS OWN U_mean. flow_axis
        # tabulates U against a 60 um channel; the cohesive-tear series runs a
        # 45 um one, so comparing against the tabulated value would refuse a
        # correct case. It must declare the velocity its own geometry implies,
        # and that declaration is then checked against the geometry exactly as
        # before -- the check moves, it does not weaken.
        if "channel_height_m" in override and not series_axis:
            if "mean_velocity_m_s" not in override:
                raise ValueError(
                    f"{child_id}: series '{child['series']}' overrides "
                    f"channel_height_m but does not declare mean_velocity_m_s. "
                    f"Falling back to the flow_axis value would silently score "
                    f"this child against a channel height it does not have."
                )
            expected = override["mean_velocity_m_s"]
        if not math.isclose(u_mean, expected, rel_tol=1e-9):
            raise ValueError(
                f"{child_id}: U_mean {u_mean} does not match the declared "
                f"{expected} for gammaDot {gamma}"
            )

        case = RUNS / child_id
        shutil.copytree(TEMPLATE, case)
        # A THROAT CHILD USES THE SAME SINGLE-BLOCK MESH AS EVERY OTHER ONE.
        # The narrowing is applied AFTER blockMesh by apply_throat.py, which
        # scales each point in y by H(x)/H. That keeps the topology identical
        # -- no extra blocks, no shared faces, no divisions to disagree -- and
        # it is the only way to get a wall with ZERO slope at both joins.
        # A multi-block taper was built first and discarded: it put a slope
        # discontinuity at each join, which is a salient corner in the one
        # place the measurement is taken.
        (case / "system" / "blockMeshDict").write_text(mural_gen.planar_mesh(g))
        mural_gen.configure_fields(case, u_mean, {
            # Dr Li, 2026-09-01. A fixed inlet velocity pins the flow rate, so
            # an occluding throat cannot show a velocity drop -- the solver
            # raises pressure instead. Only children that ask for it switch;
            # every existing child leaves these keys unset and keeps the
            # velocity inlet it was validated with.
            "pressure_driven": bool(series.get("pressure_driven")),
            "inlet_pressure_m2_s2": child.get(
                "inlet_pressure_m2_s2",
                series.get("inlet_pressure_m2_s2", 0.0),
            ),
            "wall_condition": series["wall_condition"],
            "adhesive_strength_pa": child["adhesive_strength_pa"],
            # Selects which stress the bond reads. See us_bottom_entry():
            # sigmaD is only written by solveSkeleton(), which the Eulerian
            # path never calls, so an Eulerian child pointed at it has an
            # inert failure criterion.
            "eulerian_solid": bool(True),
            # Whether the clot was carried by its own skeleton velocity or by
            # the fluid. It changes what the phase field MEANS, so it is
            # recorded next to the result rather than left in the case files.
            "two_velocity_skeleton": True,
            # FREE WALLS (design 2026-10-02-case-free-shell-throat, rev. 3,
            # S5/S7.3). Series-level only, so every existing child -- which
            # omits both -- keeps the implicit-latch bottom entry and the
            # fixedValue top entry it was validated with.
            "wall_bond_free": bool(series.get("wall_bond_free")),
            "top_wall_adhesion": bool(series.get("top_wall_adhesion")),
        })

        # THE GENERATED 0/Us IS RE-READ AND CHECKED, not trusted, because
        # this is the one dictionary red team 03/04 found silently wrong
        # twice (F1, F2, 04 #1). Refuses a free-shell child whose 0/Us lacks
        # ANY of adhesiveStrength/sigma/q/injury/bonded on EITHER wall
        # (design S3b/S7.3). Gated on wall_bond_free so no existing child's
        # build is touched.
        if series.get("wall_bond_free"):
            us_text = (case / "0" / "Us").read_text()
            m_bottom = re.search(r"(?s)bottom\s*\{(.*?)\}", us_text)
            m_top = re.search(r"(?s)top\s*\{(.*?)\}", us_text)
            if not m_bottom or not m_top:
                raise ValueError(
                    f"{child_id}: 0/Us has no readable bottom/top patch block"
                )
            for label, body in (("bottom", m_bottom.group(1)),
                                 ("top", m_top.group(1))):
                for key in ("adhesiveStrength", "sigma", "q", "injury",
                            "bonded"):
                    if not re.search(rf"\b{key}\b", body):
                        raise ValueError(
                            f"{child_id}: 0/Us {label} patch is missing "
                            f"'{key}'; a free-shell wall must declare "
                            f"adhesiveStrength, sigma, q, injury and bonded "
                            f"explicitly or the bond is not provably off "
                            f"(design S3/S3b)"
                        )
                if not re.search(r"injury\s+uniform\s+0\b", body):
                    raise ValueError(
                        f"{child_id}: 0/Us {label} patch's injury is not "
                        f"'uniform 0'; a non-empty injury band would latch a "
                        f"bond this series exists to avoid"
                    )
                if not re.search(r"bonded\s+uniform\s+0\b", body):
                    raise ValueError(
                        f"{child_id}: 0/Us {label} patch's bonded is not "
                        f"'uniform 0'"
                    )

        # Read BEFORE the properties block, because the FSI entries written
        # there depend on it. It used to be defined a hundred lines below its
        # first use, which is a NameError waiting for the first child that
        # sets it.
        eulerian_solid = bool(True)

        # READ AND CHECKED HERE, OUTSIDE the `if eulerian_solid:` block below.
        # This guard first went inside it, where it was unreachable in exactly
        # the case it exists to reject: with eulerian_solid False the whole
        # block is skipped and the guard could never fire. dry_run.py caught
        # it -- "the generator ACCEPTED two_velocity_skeleton without
        # eulerian_solid" -- which is the entire reason the negative test was
        # written before the guard was trusted. A check that cannot fail has
        # not passed.
        two_velocity = bool(True)

        props = case / "constant" / "phaseClotProperties"
        growing = bool(series["growth"])

        # GROWTH RATE MULTIPLIER. alpha ONLY, and it defaults to 1 so every
        # existing child keeps the calibrated value byte for byte.
        #
        # Growth is a 100 s process and the tear a 0.01 s one. There is no
        # window in which both are visible at their true rates, and running
        # the tear at the growth timestep is 800000 steps. Scaling alpha is a
        # declared time rescaling of the deposition-to-interface-speed
        # conversion; the platelet kinetics (k0, D) are untouched, so growth
        # stays transport-limited and simply runs faster. The realised Vn and
        # clotVolume are logged every step, so the factor is auditable.
        growth_multiplier = float(child.get("growth_rate_multiplier", 1.0))
        if growth_multiplier != 1.0 and not growing:
            raise ValueError(
                f"{child_id}: growth_rate_multiplier "
                f"{growth_multiplier} is set on a child whose series has "
                f"growth off; it would do nothing and read as though it had"
            )
        alpha_used = growth["alpha_m2_s"] * growth_multiplier if growing else 0.0

        # MECHANICS OFF IS A REAL CONFIGURATION, NOT A DEGRADED ONE, and the
        # throat-occlusion study needs it. With clotMechanics off the solver
        # takes the v2 rigid path, whose momentum equation carries
        # fvm::Sp(lambda_, U_) -- the Angot/Bruneau/Fabrie penalisation that
        # makes the clot a STATIONARY OBSTACLE. That is exactly what a static
        # occlusion curve wants, and it is also what makes it cheap: no Us, no
        # be, and dt bounded by the fluid Courant alone rather than by the
        # elastic one.
        #
        # It matters that this is the `else` branch. Run 20260823-102340
        # established that lambda_ is a nonzero FIELD on the Eulerian path
        # while being absent from the equation there -- so a child that wanted
        # a rigid obstruction and left mechanics ON would get no obstruction
        # and a plausible-looking clotPenalty field to go with it.
        mechanics_on = bool(True)
        for key, value in {
            # DIFFUSION OFF, and it exists because gate_a_exact failed at
            # 3.04e-3 (run 20260822-085149) after the carrier fix took the
            # deviation from 0.0211 to 0.00304 -- a real 7x, and not the
            # machine precision that was predicted.
            #
            # The c equation carries NO diffusion when mobility = 0. The Wp
            # equation always carries fvc::laplacian(qEff*D, w), and at
            # D = 1e-10, h = 0.5 um, dt = 2.5e-6 s the Fourier number is 1e-3
            # PER STEP -- about 0.4 over the gate window, which is not a
            # rounding. Diffusion of a uniform field is exactly zero so it
            # cannot seed the deviation, but it spreads whatever seed exists
            # across the whole domain, which is why the residual shows up at
            # q >= 0.5 (pure blood, |w-1| = 3.6e-4) where no 1/q amplification
            # is available to explain it.
            #
            # This knob is a DIAGNOSTIC, not a physics choice: D = 0 removes
            # platelet diffusion entirely and is not a configuration any
            # production run may use. It exists so the question "is the
            # diffusion term carrying the residual?" is answered by measurement
            # instead of by another confident prediction.
            "D": 0.0 if child.get("diffusion_off") else growth["D_m2_s"],
            "alpha": alpha_used,
            # CLOT-SIDE DIFFUSION EXTENSION. Written per child so the platelet
            # runaway of run 20260818-232055 (w: 1 -> 6372 against a bound of
            # 1.1) can be probed without a recompile. The template value 1e-6
            # is far below the q ~ 0.02 band where that runaway actually lives,
            # so the floor never activated there; raising it is the direct test
            # of whether the concentration is what needs arresting.
            "qFloor": child.get("q_floor", 1e-6),
            # ADHESION OFF is a gate condition, not a physics choice. Gate A of
            # AM2020_platelet_content_formulation_derivation.md needs k = 0 so
            # that adhesion cannot act on w.
            #
            # THIS COMMENT USED TO CLAIM that k = 0 plus mobility = 0 makes
            # "w == 1 an exact solution and the gate a machine-precision test".
            # Run 20260822-085149 disproved it: with both off, and with the
            # carrier fix in, gate_a_exact still measured 3.04e-3. The claim
            # forgot the diffusion term, which the Wp equation carries and the
            # c equation does not. k = 0 and mobility = 0 are NECESSARY for
            # exactness; whether they are SUFFICIENT is what diffusion_off is
            # being used to find out.
            "k0": (
                0.0 if child.get("adhesion_off")
                else (growth["k0_m_s"] if growing else 0.0)
            ),
            # MOBILITY OFF makes c purely advected, so q satisfies its own
            # conservation law and w == 1 is EXACT rather than approximate.
            # With mobility on it is only approximate -- gate B, reported not
            # gated. Same derivation, section 5.
            "mobility": (
                0.0 if child.get("mobility_off")
                else growth["mobility_m4_s"]
            ),
            "epsilonC": g["interface_width_m"],
            # gradFloor MUST SCALE WITH THE INTERFACE. It regularises
            # normal_ = gradC/max(|gradC|, gradFloor), and at the shipped
            # 1e-12 it sat SEVENTEEN orders below the true interface gradient
            # 2/epsilonC = 5e5 1/m. Consequence measured in run
            # 20260819-204541: a bulk noise gradient of 4.4e-5 1/m -- in a cell
            # with c = 1.000000 and interfaceDelta ten orders below the real
            # interface -- was renormalised to a FULL UNIT NORMAL, and Vn
            # reported 1.05e-6 m/s in pure blood. The c source is masked by
            # phaseGradient so it never corrupted c, but the diagnostic and
            # gate E both read it.
            #
            # 1e-3 of the true interface gradient: the real interface
            # (|gradC| ~ 5e5) is unaffected and still yields a unit normal,
            # while bulk noise is suppressed by ~1e7. It shifts phaseGradient
            # at the interface by 0.1%, which is a real if small change to the
            # growth weight and is reported rather than buried.
            "gradFloor": 1.0e-3*2.0/g["interface_width_m"],
            "ell": g["ell_m"],
            "G": child["G_pa"],
            "poissonRatio": fixed["poissonRatio"],
            "k0p": child["k0p_m2"],
            "etaRes": fixed["etaRes"],
            "DgDetach": fixed["DgDetach"],
            "Gc": child.get("Gc_j_m2") or fixed["Gc_j_m2"],
            # THE PLASMA, per series (red team 05 #1): the Xu series runs
            # Xu's 4 cP (rho 1025, nu 3.9024e-6); every other series omits
            # the keys and keeps material_fixed byte for byte.
            "rho": series.get("rho_kg_m3", fixed["rho_kg_m3"]),
            "nu": series.get("nu_m2_s", fixed["nu_m2_s"]),
            "fsiRelaxation": coupling["fsiRelaxation"],
        }.items():
            replace_entry(props, key, value)


        # CORE/SHELL MATERIAL. Off by default; when off the solver's blend is
        # the identity and every material field is uniform and equal to the
        # scalar it was built from.
        core_shell = bool(series.get("core_shell", False))
        # THE PHASE-FIELD CLIP -- a MEASUREMENT, default off.
        #
        # Nothing else in this solver bounds c: mobility is 0.0 in every
        # main-case child, so the double well is inert and c is a pure
        # advected scalar on `div(phiReleased,c) bounded Gauss vanLeer`,
        # a scheme that is TVD only for a divergence-free flux --  which
        # phiReleased is not. Run 20260906-004750 measured cMin = -1.619
        # with 881 of 942 offending cells having ALL FOUR neighbours also
        # below -0.9, i.e. the overshoot is interior, not interfacial.
        #
        # Turning this on clips c into [-1, 1] and REPORTS what that cost
        # (PHASE_CLOT_CLIP: clipVolumeCum, clipDVdt, clipFracOfClot). It
        # exists to decide, before any interface rewrite, whether the
        # overshoot is a saturating scheme artefact -- in which case MULES
        # is the right and sufficient repair -- or a genuinely convergent
        # advecting flux, in which case MULES would silently discard real
        # material and the pressure equation's volume source comes first.
        #
        # Left False by default so no existing child moves by one bit.
        add_or_replace_entry(
            props,
            "clipPhaseField",
            "yes" if child.get("clip_phase_field", False) else "no",
        )

        add_or_replace_entry(props, "coreShell", "yes" if core_shell else "no")
        # S15: written here, beside coreShell, and NOT inside the two-velocity
        # block -- a child that sets it anywhere gets it, and a child that
        # does not is byte-identical. Refused beforehand unless the content
        # model and a core_shell series are both present.
        for _key, _dict_key, _allowed in S15_SWITCHES:
            if _key in child:
                add_or_replace_entry(props, _dict_key, str(child[_key]))
        if core_shell:
            # Gccore is written to the SAME value as Gc on purpose. Source 33
            # has no toughness contrast -- it has no toughness -- so the blend
            # is the identity for Gc and the material difference is entirely
            # in the modulus and the permeability, which is what they measured.
            for key, value in {
                "Gcore": child["G_core_pa"],
                "Gccore": child.get("Gc_core_j_m2")
                or (child.get("Gc_j_m2") or fixed["Gc_j_m2"]),
                "k0pcore": child["k0p_core_m2"],
            }.items():
                add_or_replace_entry(props, key, value)


        # THE TWO PLATELET FORMULATIONS ARE ALTERNATIVES, never both. The
        # solver refuses the combination; this keeps the dictionaries from ever
        # expressing it. Derivation: research/designs/
        # AM2020_platelet_content_formulation_derivation.md
        #
        # DEFAULT FLIPPED TO ON, 2026-08-23, to match the solver. It was OFF,
        # and run 20260823-023233 paid for that: force_cell and force_face were
        # built by copying renorm_off, which predates this work, and both
        # silently inherited the w branch -- Wp never solved and sitting at
        # uniform 1, wMax 7312, 1446 cells out of bounds, and Vn at 0.594 and
        # 1.368 m/s contaminating the clot geometry the momentum A/B was
        # measuring.
        #
        # Wp = q*w is the conserved content; w = Wp/q divides by a quantity
        # that reaches 1.3e-5 in the clot core, a 77000x amplifier. The six
        # children that WANT the w branch pin it to false explicitly, so the
        # default no longer decides anything by accident.
        content_form = bool(True)

        # CONSTITUTIVE LAW, written for every child so it is recorded rather
        # than inherited from whatever the solver default happens to be. The
        # source-33 child asks for neoHookean because a 4 Pa shell at 1000 1/s
        # is taken well past where St Venant-Kirchhoff is valid; run
        # 20260814-224138 diverged at 72% strain doing exactly that.
        # Verified equivalent to St Venant-Kirchhoff at small strain by the
        # constitutive twins of the elastic-slab gate, run 20260815-072849:
        # both landed on max|Us| = 5.096e-12 m against a closed form.
        constitutive_model = 'neoHookean'

        # THE EULERIAN SOLID SWITCH -- stage 3 of
        # research/designs/AM2020_eulerian_solid_plan.md.
        #
        # Defaults to false, so every existing child of this case generates
        # byte for byte as before. Only the stage-3 children set it.
        #
        # Under `eulerianSolid` the skeleton displacement equation is replaced
        # by transport of the left Cauchy-Green tensor `be` on the single
        # mixture velocity. There is then no relative fluid/skeleton motion, so
        # the Brinkman drag that couples them has nothing to act on and the
        # partitioned FSI exchange has nothing to exchange. Combining it with
        # stronglyCoupledFSI would be asking for a coupling between a field and
        # itself; the solver has its own FatalError for that, and refusing here
        # names the child.
        # PICARD SWEEPS OF THE EXPLICIT STRETCH TERM, written explicitly so
        # the value a run used is in its own dictionary rather than being
        # the solver's default. Run 20260817-234636 showed three is not
        # always enough here: beSweepChange grew from 3.8e-16 to 7.9e-4
        # over the run, i.e. the iteration stopped converging as the clot
        # reached stretch 4.6. The diagnostic sweeps this.
        add_or_replace_entry(
            props, "beCorrectors", int(child.get("be_correctors", 3))
        )

        # THE ANCHOR AND THE RELEASE TEST. All four default to the
        # historical behaviour, so a child that names none of them is
        # bit-for-bit what it was before 2026-08-24. Written
        # UNCONDITIONALLY once any of them is on any child of the series,
        # so a run's own dictionary always states which anchor it had --
        # the sigmaD failure was invisible precisely because the case file
        # said nothing about it.
        add_or_replace_entry(
            props,
            "releaseSupportLimit",
            f"{float(child.get('release_support_limit', 0.0)):.12g}",
        )
        add_or_replace_entry(
            props,
            "bondHoldDilation",
            str(int(child.get("bond_hold_dilation", -1))),
        )
        add_or_replace_entry(
            props,
            "bondHoldPenalty",
            f"{float(child.get('bond_hold_penalty', 0.0)):.12g}",
        )
        # THE CORE EXEMPTION, written ONLY when the child names it, so
        # every other dictionary is byte-identical and keeps the solver
        # default (true, .C L724-726). The Xu series turns it off
        # (Alireza, 2026-09-29: nothing about the core is enforced).
        # Refused beforehand where it would be inert
        # (release_core_exempt_refusal).
        if "release_core_exempt" in child:
            add_or_replace_entry(
                props,
                "releaseCoreExempt",
                "yes" if child["release_core_exempt"] else "no",
            )

        # THE BRINKMAN DRAG, GATED ON RELEASE. Off by default, so every
        # child that predates it assembles the same matrix and no passed
        # gate moves. See research/designs/
        # AM2020_stage1_brinkman_implementation_plan.md.
        #
        # REFUSED HERE AS WELL AS IN THE SOLVER, and deliberately twice.
        # The solver's FatalError costs a VM boot and an OpenFOAM restore
        # to discover; this one costs nothing and names the child. Without
        # advect_released the release mask is identically zero, the drag
        # acts on detached material too, and the run would answer a
        # question about a model that pins emboli in the lab frame.
        brinkman = bool(True)


        # THE SKELETON VOLUME FRACTION, and it is the largest
        # uncertainty in the two-velocity formulation.
        #
        # chi = 1 - q is an INDICATOR, not a volume fraction: measured
        # on run 20260904-214345, the median clot cell has q = 0.0066
        # and 71.7% of clot cells have q < 0.05. Using chi as phi_s
        # would leave phi_f ~ 0.007 -- blood essentially absent from
        # the clot -- which contradicts the permeability model that
        # lets blood seep through it.
        #
        # 0.01 is anchored, not guessed: He, Kim & Ku (source 23)
        # report Wufsus et al. (2013) measuring a 1% platelet volume
        # fraction at k = 1.1e-2 um2 = 1.1e-14 m2, and k0pcore here is
        # 1e-14 m2. Written per child so a sweep over it is a
        # dictionary change, not a code change.
        add_or_replace_entry(
            props,
            "skeletonVolumeFraction",
            repr(float(child.get("skeleton_volume_fraction", 0.01))),
        )
        # Vanishing-phase regularisation for the skeleton row of the
        # 2x2 block. OpenFOAM calls it residualAlpha; Du & Fogelson
        # describe the same fix as "temporarily increasing theta_b by a
        # small positive constant".
        add_or_replace_entry(
            props,
            "residualPhiS",
            repr(float(child.get("residual_phi_s", 1e-4))),
        )
        # ANCHOR-INVARIANT SUPPORT for that regularisation. 0 = off,
        # and off is byte-for-byte the previous behaviour.
        #
        # residualPhiS is an ABSOLUTE threshold on phi_s, but
        # phi_s = chi*skeletonVolumeFraction, so the band it covers
        # shrinks as the anchor rises. Run 20260906-101812 measured
        # every one of the top 100 cells by |Vs| sitting at c > 0.99
        # with Kres = 0 -- the regularisation had switched itself off
        # in exactly the cells that were running away, because at
        # phi_s0 = 0.45 the diffuse tail of c puts phi_s at 1.7e-04,
        # above the 1e-4 threshold. At phi_s0 = 0.10 the same tail
        # gives 3.8e-05 and Kres still fires.
        #
        # residual_chi measures the same ramp on chi instead, rescaled
        # so the PEAK is still residualPhiS/deltaT. Region changes,
        # strength does not.
        add_or_replace_entry(
            props,
            "residualChi",
            repr(float(child.get("residual_chi", 0.0))),
        )
        # THE DILUTE DRAG FLOOR, K >= k1*chi, in 1/s. Zero = off, and
        # off reproduces every earlier child bit-for-bit.
        #
        # Kozeny-Carman is a PACKED-BED correlation and this solver
        # evaluates it two decades below its range. Measured on run
        # 20260907-111028, binning by chi: |Vs|/max|U| is 1.00 where
        # compliance (elastic weight / total drag rate) is at or below
        # ~1e-05 and 3.4-3.9 in chi = 0.005-0.1 where it reaches
        # 3.0e-04. The floor pins compliance at exactly 1/k1 wherever
        # it binds, so the peak cannot re-form somewhere else -- which
        # widening residualChi (factor 2.2, measured) cannot do.
        # Full table and the rejected alternatives are on
        # skeletonDiluteDragRate_ in the solver header.
        # SURFACE EROSION -- the only route by which this clot can
        # let go. Bulk fracture is physically unreachable here
        # (Gc/G = 1.39 m against a 40 um clot; Dg reached 6.4e-06 of
        # 1.0 after 7.9 ms on run 20260908-010836), wall peel arrests
        # once the remaining bonds unload (utilisation 0.985 -> 0.389),
        # and the release criterion is a connectivity flood fill that
        # cannot sever a connected tail. So the clot stretches without
        # bound. Threshold sourced to 1.75-3.0 Pa: Xu (33) balance the
        # accumulation and detachment coefficients at 500 1/s = 1.75 Pa
        # and Wu (15) give tau_emb = 30 dyn/cm2 = 3.0 Pa = 865 1/s.
        # THIS CASE RUNS AT 1000 1/s. The RATE is unsourced and swept,
        # exactly as adhesiveStrength is. Zero = off and reproduces
        # every earlier child byte for byte.
        add_or_replace_entry(
            props,
            "clotErosionRate",
            repr(float(child.get("clot_erosion_rate", 0.0))),
        )
        add_or_replace_entry(
            props,
            "clotErosionTauPa",
            repr(float(child.get("clot_erosion_tau_pa", 3.0))),
        )
        add_or_replace_entry(
            props,
            "skeletonDiluteDragRate",
            repr(float(child.get("skeleton_dilute_drag_rate", 0.0))),
        )
        # WHICH OPERATOR solveContentComponents() ASSEMBLES.
        #
        # `content` is the shipped Stage 1 model: upwind, no sources.
        # `matched` is the T2a identity test: the captured
        # phaseWeights_ plus the mapped source
        # theta_ref*(-0.5*laplacian(mobility,muC) + 0.5*Vn*gradPhase),
        # which makes it the exact conservative image of the legacy
        # chi equation.
        #
        # VALIDATED HERE AS WELL AS IN THE SOLVER. The solver's
        # lookupOrDefault returns "content" for a misspelling just as
        # it does for a missing key, so the solver fatals on an
        # unknown word -- but that costs a Colab boot to discover.
        # Refusing it at generation costs nothing and fails on the
        # machine that can fix it.
        content_operator = str('content')
        # ---- STAGE 2: THE ACCEPTED-TRANSFER GROWTH PATH ----
        #
        # This is the SECOND way solid enters the content, and the one
        # a search for assignments to thetaH_ cannot see: it is written
        # through an output parameter of
        # plateletContent::applyAcceptedTransfer(). The first is the
        # `matched` source above.
        #
        # THE TWO ARE MUTUALLY EXCLUSIVE and the solver refuses the
        # combination outright, because both deposit solid and the
        # inventory still balances -- each is internally paired, so
        # nothing downstream catches it. Refused here as well, for the
        # reason the contentOperator validation above gives: a Colab
        # boot is an expensive place to learn this.
        platelet_content = bool(
            True
        )
        platelet_transport_only = bool(
            child.get("platelet_transport_only", False)
        )
        # ---- STAGE 2: THE ACCEPTED-TRANSFER GROWTH PATH ----
        #
        # The SECOND way solid enters the content, and the one a
        # search for assignments to thetaH_ cannot see: it is written
        # through an output parameter of
        # plateletContent::applyAcceptedTransfer(). The first is the
        # `matched` source above, and the two are mutually exclusive.
        validate_stage2_platelet(child, content_operator)
        add_or_replace_entry(
            props,
            "plateletBindRate",
            repr(float(child.get("platelet_bind_rate", 0.0))),
        )
        add_or_replace_entry(
            props,
            "plateletDiffusivity",
            repr(float(child.get("platelet_diffusivity", 0.0))),
        )
        if "platelet_adhesion_speed" in child:
            add_or_replace_entry(
                props,
                "plateletAdhesionSpeed",
                repr(float(child["platelet_adhesion_speed"])),
            )
        # S4. DEFAULTS TO "transfer", for the same reason and by
        # the same two-place rule as phaseDeltaBasis above: the
        # solver's own fallback is "transfer" too, and this
        # generator writes the key unconditionally, so flipping
        # only the solver would have been cosmetic on this case.
        # A child that wants the Stefan front must say
        # phase_growth_driver "front" explicitly; em36 is pinned
        # to it because its JSON entry must keep describing the
        # run that exists.
        cid = child.get("child_id")
        growth_driver = str(
            'transfer'
        )
        if float(
            child.get("skeleton_volume_fraction", 0.0)
        ) <= 0:
            raise SystemExit(
                "child %s: phase_growth_driver 'transfer' "
                "divides by skeleton_volume_fraction, which "
                "is %r. The conversion is "
                "dc = -(2/theta_ref)*dtheta."
                % (cid, child.get("skeleton_volume_fraction"))
            )
        if "platelet_yield_m3" in child:
            add_or_replace_entry(
                props,
                "plateletYield",
                repr(float(child["platelet_yield_m3"])),
            )
        if platelet_transport_only:
            add_or_replace_entry(
                props, "plateletTransportOnly", "yes"
            )
        write_platelet_number_density(
            case,
            float(child["platelet_number_density_m3"]),
            float(child.get(
                "platelet_inlet_density_m3",
                child["platelet_number_density_m3"])),
        )
        # nP IS SOLVED, so it needs a linear solver, and
        # solvePlateletContent calls nEqn.flux() for the boundary
        # inventory, which needs fluxRequired. Neither is in the
        # shared template -- no previous main-case child ran the
        # platelet equation. Both were missing from run
        # 20260915-062201; the restart contract fatalled first, so
        # they would have been the next two failures, one run each.
        # NO div SCHEME IS NEEDED: plateletContent::transportEqn
        # builds explicit upwind weights rather than looking one
        # up (.C:7098-7101).
        solution = case / "system" / "fvSolution"
        solution_text = solution.read_text()
        if not re.search(r"(?m)^\s*nP\s*$", solution_text):
            entry = (
                "    nP\n"
                "    {\n"
                "        solver smoothSolver;\n"
                "        smoother symGaussSeidel;\n"
                "        tolerance 1e-12;\n"
                "        relTol 0;\n"
                "    }\n"
                "    nPFinal\n"
                "    {\n"
                "        $nP;\n"
                "        relTol 0;\n"
                "    }\n"
            )
            patched, count = re.subn(
                r"(?m)^\}\n\nPIMPLE", entry + "}\n\nPIMPLE",
                solution_text, count=1)
            if count != 1:
                raise SystemExit(
                    "child %s: could not place the nP solver entry "
                    "in fvSolution -- the `}\\n\\nPIMPLE` anchor "
                    "the template has always had is gone, and a "
                    "silent skip would fatal in Colab"
                    % (child.get("child_id"),)
                )
            solution.write_text(patched)
        schemes = case / "system" / "fvSchemes"
        schemes_text = schemes.read_text()
        if "fluxRequired" not in schemes_text:
            schemes.write_text(
                schemes_text.rstrip("\n")
                + "\n\n// nEqn.flux() supplies the measured "
                  "boundary inventory flux.\nfluxRequired\n{\n"
                  "    default no;\n    nP;\n}\n"
            )
        elif not re.search(r"fluxRequired\s*\{[^}]*\bnP\s*;",
                           schemes_text, re.S):
            raise SystemExit(
                "child %s: fvSchemes already has a fluxRequired "
                "block that does not name nP; merging it blindly "
                "could drop an entry another field needs."
                % (child.get("child_id"),)
            )
        # Declared volume per bound platelet-equivalent. Inert while
        # kinetics are off -- Stage 1 multiplies nothing by it -- so
        # the placeholder cannot silently acquire authority before it
        # is sourced.
        add_or_replace_entry(
            props, "vP", repr(float(child.get("v_p_m3", 1e-17)))
        )
        # THE NETWORK PRESSURE. Run 20260905-113709 conserved the
        # clot to 2.4e-05 and then drove c to -2.74 at STEP 4: phi_s
        # nearly doubled and nothing in the model could object,
        # because elasticStress() is dev(be) only and `be` is an
        # INCOMPRESSIBLE neo-Hookean with det(be) = 1 enforced -- the
        # constitutive model had no bulk modulus anywhere.
        #
        # p_n = K_n*(phi_s - phi_s0)/(phi_s_max - phi_s), Cogan & Guy
        # Eq. 19. K_n = networkPressureCoeff * gD * G: the default of
        # 1.0 introduces NO new number, since a cross-linked gel's
        # bulk modulus is of the same order as its shear modulus and
        # reusing GField_ inherits the core/shell contrast. Anchored,
        # not fitted -- written per child so it can be swept.
        add_or_replace_entry(
            props,
            "networkPressureCoeff",
            repr(float(child.get("network_pressure_coeff", 1.0))),
        )
        # Close packing, where p_n diverges. 0.6 is Du & Fogelson
        # Eq. 2.25's theta_b_max, after Wufsus et al. 2013 -- the same
        # paper skeletonVolumeFraction is anchored to.
        #
        # READ THIS BEFORE REUSING 0.6: it is the anchor of their DRAG
        # coefficient xi(theta_b), not of a pressure, and Cogan & Guy
        # -- the source the pressure law is taken from -- prescribe a
        # CUBIC with a double root at phi_n = 0 rather than a pole
        # anywhere. See research/literature/AM.2020.0056-19/
        # NETWORK_COMPACTION_AUDIT.md. phiSMax is UNUSED when
        # networkPressureModel is cubic; it is still written so the
        # dictionary records what the run was configured with.
        add_or_replace_entry(
            props,
            "phiSMax",
            repr(float(child.get("phi_s_max", 0.6))),
        )
        # poleAtMax (default, the shipped law) or cubic. The pole form
        # applies a finite suction wherever there is a trace of
        # network, because p_n(0) = -K_n*phi_s0/phi_s_max is not zero;
        # run 20260905-223740 measured 88 diffuse-edge cells above
        # max|U| at phi_s0 = 0.10, one at |Vs|/|U| = 646. The cubic has
        # a double root at phi_s = 0 and the fringe term falls from
        # O(phi_s) to O(phi_s^3). Default left at poleAtMax so no
        # existing child moves.
        #
        # poleCubic, ADDED 2026-09-21, is the cubic multiplied by
        # (phiSMax - phi_s0)/(phiSMax - phi_s): the double root at
        # zero AND a pole at close packing. Both shipped laws have
        # exactly one of those and each lost a run for want of the
        # other -- poleAtMax SIGFPE at t = 9.71e-05 s
        # (20260906-210336), cubic phiSOverNominal 1.1913 and still
        # accelerating at 4e-03 s (20260921-072935) once k0p moved to
        # Xu's values and the Darcy load rose two decades. Properties
        # and pricing in scripts/check_network_pressure_law.py.
        #
        # REFUSED HERE AS WELL AS IN THE SOLVER. The solver
        # FatalErrors on an unknown name, which costs a VM boot; a
        # typo should not. The set is asserted equal to the solver's
        # by check_network_pressure_law.py so the two cannot drift.
        _npm = str('convexExcess')




        # S13 (2026-09-23): the grid-scale restoring of run
        # 20260923-135545 (research/designs/2026-09-23-S13-grid-scale-
        # restoring.md). Same contract as S10-S12.
        for _key, _dict_key, _allowed in S13_SWITCHES:
            if _key not in child:
                continue
            _val = str(child[_key])
            if _val not in _allowed:
                raise SystemExit(
                    f"create_cases.py: {child['child_id']}: {_key} "
                    f"{_val!r} is not one of {', '.join(_allowed)}."
                )
            add_or_replace_entry(props, _dict_key, _val)


        # S16 (2026-09-24): terminal rupture with a timed release,
        # the damage carried as content, the occupancy floor and the
        # tensile pressure damage (research/designs/2026-09-24-S16-
        # timed-terminal-rupture.md). Same contract as S10-S15: every
        # key is written only when the child names it (or, for the
        # terminalRelease parameters, only with the switch that reads
        # them), so every earlier child's dictionary is byte-identical.
        # Refused beforehand by s16_value_refusal and
        # s16_combination_refusal, for every child.
        _prog = bool(True)
        add_or_replace_entry(
            props, "networkCriticalStretch",
            repr(float(child.get("network_critical_stretch", 2.8))),
        )
        add_or_replace_entry(
            props, "networkReleaseTime",
            repr(float(child.get("network_release_time_s", 3.3e-4))),
        )
        add_or_replace_entry(
            props, "networkDamageOnsetStretch",
            repr(float(child.get(
                "network_damage_onset_stretch", 2.105))),
        )
        add_or_replace_entry(
            props, "networkDamageStretchScale",
            repr(float(child.get(
                "network_damage_stretch_scale", 1.0))),
        )
        # S18p (2026-10-01): the log-only probe. Same contract:
        # written only when named, so every earlier dictionary is
        # identical.
        for _key, _dict_key, _allowed in S18P_SWITCHES:
            if _key not in child:
                continue
            _val = str(child[_key])
            if _val not in _allowed:
                raise SystemExit(
                    f"create_cases.py: {child['child_id']}: {_key} "
                    f"{_val!r} is not one of {', '.join(_allowed)}."
                )
            add_or_replace_entry(props, _dict_key, _val)
        # S19 (2026-10-04): the block-coupled step and R3. Same
        # contract: written only when named, so every earlier
        # dictionary is identical (an absent key is the shipped step).
        for _key, _dict_key, _allowed in S19_SWITCHES:
            if _key not in child:
                continue
            _val = str(child[_key])
            if _val not in _allowed:
                raise SystemExit(
                    f"create_cases.py: {child['child_id']}: {_key} "
                    f"{_val!r} is not one of {', '.join(_allowed)}."
                )
            add_or_replace_entry(props, _dict_key, _val)
        if "tau_v_s" in child:
            add_or_replace_entry(props, "networkViscosityTau",
                                 float(child["tau_v_s"]))
        # S22 (2026-10-07): the block step's linear solver. Same contract:
        # written only when named, so every earlier dictionary is identical.
        # Validated by s22_key_refusal() (called with the series refusals).
        for _key, _dict_key in S22_KEYS:
            if _key not in child:
                continue
            _val = child[_key]
            add_or_replace_entry(
                props, _dict_key,
                ("true" if _val else "false") if isinstance(_val, bool) else _val)
        # S23 (2026-10-07): R_c, the volumetric tear ratio. Same contract: written only
        # when named (s23_key_refusal), so every earlier dictionary is identical.
        for _key, _dict_key in S23_KEYS:
            if _key in child:
                add_or_replace_entry(props, _dict_key, float(child[_key]))
        # S23 rev. 4.1: the sensitivity marker, a word. Written ONLY on the labelled child
        # (s23_build_refusal), so every other dictionary is identical.
        if "sensitivity_limit" in child:
            add_or_replace_entry(props, "sensitivityLimit", str(child["sensitivity_limit"]))


        # ON only for the demonstration series. Everywhere else the bond never
        # fails, so `released` would be identically zero -- but an fvm::div
        # with a zero flux still enters the matrix, and the growth and
        # required-strength series stay provably identical to run
        # 20260810-212211 by not having the term at all.
        # ADVECTION IS OFF FOR THE PEEL, ALWAYS. Stage 2 of Allrun turns it on.
        #
        # Run 20260811-195034 generated the demonstration children with
        # advectReleased already ON, so the moment the first faces debonded at
        # t = 0.08 the clot began advecting at the PEEL timestep: 1.216e-3 m/s
        # with dt = 0.02 and h = 1.5e-6 is an interface Courant number of 16.2,
        # 24 um of travel per step. Upwind at Co = 16 does not transport, it
        # smears: c_min went -0.997 -> -0.937 -> -0.652 -> -0.445 -> -0.145
        # between t = 0.04 and t = 0.40 while the centroid slid 120 -> 137 um.
        # The clot dissolved in place, which is precisely the outcome the
        # two-stage design was written to avoid -- and the design never
        # enforced its own precondition.
        #
        # `advecting` still records the SERIES intent, because Allrun reads it
        # from matrix.json to decide whether a child gets a transport stage at
        # all. What changes is that the dictionary the PEEL runs with says no.
        advecting = bool(True)
        for key, value in {
            "gammaLow": growth["gammaLow_1_s"],
            "gammaHigh": growth["gammaHigh_1_s"],
            "enhancementMax": growth["enhancementMax"],
        }.items():
            add_or_replace_entry(props, key, value)

        # PER-CHILD DURATION. A ladder whose rungs finish at wildly different
        # times should not all pay for the slowest: run 20260814-055141 spent
        # 8.15 hours and most of it bought nothing, because gc1em9 had
        # destroyed the clot by t = 0.02 and gc1em7 was never going to tear.
        # Falls back to the series value, so every other series is untouched.
        end_time = child.get("end_time_s", series["end_time_s"])
        write_interval = child.get(
            "write_interval_s", series["write_interval_s"]
        )

        # OUTPUT BUDGET GUARD -- refuse a child that would return a download
        # nobody can take delivery of.
        #
        # Run 20260906-210336 returned 38.89 GB because em_2vel_ps45_cub_rchi
        # finally SURVIVED: endTime 0.03 at writeInterval 2.5e-05 is 1200 time
        # directories, and a time directory of this case measures 19.5 MB
        # across its 37 fields (measured on run 20260906-004750, ASCII,
        # writePrecision 12, 31360 cells). Every child that died early hid this
        # -- Batch 20's pair wrote 2 and 5 directories -- so the cost only
        # appeared once the physics started working, which is the worst moment
        # to discover it.
        #
        # The estimate is deliberately crude: it is a per-time-directory
        # constant for THIS case's mesh and field count, and it is here to
        # catch order-of-magnitude mistakes, not to predict bytes. A child that
        # genuinely wants the full series sets accept_large_output and says why
        # in its own record.
        projected_writes = end_time / write_interval
        projected_gb = projected_writes * PER_TIME_DIR_GB
        if (
            projected_gb > OUTPUT_BUDGET_GB
            and not child.get("accept_large_output", False)
        ):
            raise ValueError(
                f"{child_id}: end_time_s {end_time} / write_interval_s "
                f"{write_interval} = {projected_writes:.0f} write times, "
                f"~{projected_gb:.1f} GB returned if this child SURVIVES, "
                f"against a {OUTPUT_BUDGET_GB} GB budget. Raise "
                f"write_interval_s, shorten end_time_s, or set "
                f'"accept_large_output": true on the child and record why. '
                f"See run 20260906-210336, which returned 38.89 GB."
            )
        if end_time > series["end_time_s"]:
            raise ValueError(
                f"{child_id}: end_time_s {end_time} exceeds the series "
                f"budget {series['end_time_s']}; a child may shorten its run, "
                f"not extend it past what the series costed for"
            )
        control = case / "system" / "controlDict"
        for key, value in {
            # PER CHILD, defaulting to the series value so nothing that does
            # not ask moves. The stage-3 diagnostic needs a dt control: if
            # det(be) is a temporal-resolution problem, a quarter step must
            # show it, and if it is a discretisation trace error it will not.
            "deltaT": float(child.get("delta_t_s", series["delta_t_s"])),
            "endTime": end_time,
            "writeControl": "runTime",
            "writeInterval": write_interval,
        }.items():
            replace_entry(control, key, value)

        # S16 (design section 6, em63): A WALL BUDGET. OpenFOAM 13's
        # stopAtClockTime function object (libutilityFunctionObjects) sets
        # stopAt writeNow once the solver's elapsed clock time exceeds
        # stopTime, and prints "stopAtClockTime(timeIndex=N): stopping now
        # after writing", which assess_case.py reads. runTimeModifiable is
        # false in these cases, so editing controlDict mid-run would not be
        # seen; a function object needs no re-read. Written ONLY for a child
        # that names the key, so every other controlDict is byte-identical.
        if "wall_budget_h" in child:
            ctl_text = control.read_text()
            if re.search(r"(?m)^\s*functions\b", ctl_text):
                raise SystemExit(
                    f"create_cases.py: {child_id}: controlDict already has a "
                    f"functions block; the wall budget would be ambiguous")
            budget_s = float(child["wall_budget_h"])*3600.0
            functions_body = (
                "    wallBudget\n    {\n"
                "        type            stopAtClockTime;\n"
                "        libs            (\"libutilityFunctionObjects.so\");\n"
                f"        stopTime        {budget_s:.16g};\n"
                "        action          writeNow;\n"
                "    }\n"
            )
            # THE STEP-1 WATCHER (design 2026-10-02-case-free-shell-throat,
            # rev. 3, S7.3b; red team 03 #3, red team 04 #2). Editing
            # controlDict mid-run does nothing under runTimeModifiable false,
            # so the stop goes through OF-13's stopAtFile function object
            # (src/functionObjects/utilities/stopAt/stopAtFile.C): its default
            # watch file is "$FOAM_CASE/<this object's own name>", i.e.
            # <case>/stopWatcher here, and a background process (launched by
            # Allrun) touches that file once the log shows a non-zero V1 key.
            # ONE block, with wallBudget and the function objects below, or
            # the refusal three lines above would fire on the second write.
            if series.get("step1_watcher"):
                functions_body += (
                    "    stopWatcher\n    {\n"
                    "        type            stopAtFile;\n"
                    "        libs            (\"libutilityFunctionObjects.so\");\n"
                    "        action          writeNow;\n"
                    "    }\n"
                )
                # THE FUNCTION OBJECTS (design S7.5): every time step, so the
                # time series the event trigger reads cannot miss a write.
                for name, patch in (("inletFlux", "inlet"), ("outletFlux", "outlet")):
                    functions_body += (
                        f"    {name}\n    {{\n"
                        "        type            surfaceFieldValue;\n"
                        "        libs            (\"libfieldFunctionObjects.so\");\n"
                        "        writeControl    timeStep;\n"
                        "        writeInterval   1;\n"
                        "        writeFields     no;\n"
                        # OF-13 selects the region by WHICH KEY is present
                        # (NamedEnum::select, NamedEnum.C:86-113: one of
                        # faceZone/patch/patches/sampledSurface, else
                        # FatalIOError) and reads the patch from that key
                        # (surfaceFieldValue.C:617-619). The pre-OF-9
                        # "regionType patch; name <p>;" form selects nothing
                        # and kills foamRun at function-object construction
                        # (functionObjectList.C:159, no try).
                        f"        patch           {patch};\n"
                        "        operation       orientedSum;\n"
                        "        fields          (phi);\n"
                        "    }\n"
                    )
                for name, patch in (("inletP", "inlet"), ("outletP", "outlet")):
                    functions_body += (
                        f"    {name}\n    {{\n"
                        "        type            surfaceFieldValue;\n"
                        "        libs            (\"libfieldFunctionObjects.so\");\n"
                        "        writeControl    timeStep;\n"
                        "        writeInterval   1;\n"
                        "        writeFields     no;\n"
                        # OF-13 selects the region by WHICH KEY is present
                        # (NamedEnum::select, NamedEnum.C:86-113: one of
                        # faceZone/patch/patches/sampledSurface, else
                        # FatalIOError) and reads the patch from that key
                        # (surfaceFieldValue.C:617-619). The pre-OF-9
                        # "regionType patch; name <p>;" form selects nothing
                        # and kills foamRun at function-object construction
                        # (functionObjectList.C:159, no try).
                        f"        patch           {patch};\n"
                        "        operation       areaAverage;\n"
                        "        fields          (p);\n"
                        "    }\n"
                    )
                # PROBE LOCATIONS (design S7.5): (100,25), (120,25), (225,10),
                # (270,10) um -- clear of the disc at t = 0 (P1 spans
                # 139.9-169.9 um), upstream of the taper, at the throat
                # entrance and mid-throat/downstream.
                probe_points = (
                    (100e-6, 25e-6), (120e-6, 25e-6),
                    (225e-6, 10e-6), (270e-6, 10e-6),
                )
                probe_lines = "\n".join(
                    f"            ({px:.6g} {py:.6g} 0)" for px, py in probe_points
                )
                functions_body += (
                    "    probesPU\n    {\n"
                    "        type            probes;\n"
                    "        libs            (\"libsampling.so\");\n"
                    "        writeControl    timeStep;\n"
                    "        writeInterval   1;\n"
                    "        fields          (p U);\n"
                    "        probeLocations\n        (\n"
                    f"{probe_lines}\n        );\n"
                    "    }\n"
                )
            # THE FRAMES (BATCH 68 design section 3): between the full
            # writes, OF-13's writeObjects (src/functionObjects/utilities/
            # writeObjects, in libutilityFunctionObjects: Make/files L15,
            # L19) writes ONLY the reader-derived fields (FRAME_FIELDS_
            # REQUIRED, refused offline if any is missing). writeOption
            # autoWrite: an object is written only if the solver itself
            # writes it (writeObjects.C L69-76), and at a full write time it
            # is left to the full write (L93-98), so no field is written
            # twice. A non-write time gets its uniform/time from
            # writeObjectsBase::write. adjustableRunTime: the frame times
            # enter Time::adjustDeltaT through functionObjects_.
            # timeToNextAction() (Time.C L72-80), as batch 63's 0.2 ms
            # writeInterval did, so the dt sequence is that of a 0.2 ms
            # writer. Only for a child that declares the keys, so every
            # earlier controlDict is byte-identical.
            if "frame_interval_s" in child:
                functions_body += (
                    "    frames\n    {\n"
                    "        type            writeObjects;\n"
                    "        libs            (\"libutilityFunctionObjects.so\");\n"
                    "        objects         ("
                    + " ".join(child["frame_fields"]) + ");\n"
                    "        writeOption     autoWrite;\n"
                    "        writeControl    adjustableRunTime;\n"
                    f"        writeInterval   {float(child['frame_interval_s']):.16g};\n"
                    "    }\n"
                )
            control.write_text(
                ctl_text.rstrip("\n")
                + "\n\nfunctions\n{\n" + functions_body + "}\n")

        # ADAPTIVE TIMESTEP, where the series asks for it.
        #
        # This replaces the two-stage restart. That existed because the peel
        # was happy at dt = 0.02 while advection needs Courant below one, and
        # it bought the separation with three things that are not physics:
        # advection disabled while the clot was ALREADY FREE (run
        # 20260811-195034 debonded all 60 faces at t = 0.08 s and then held the
        # clot in place for another 1.1 s), growth switched off mid-flight, and
        # an unconditional release that asserted detachment instead of
        # resolving it.
        #
        # With adjustTimeStep the solver sizes dt from the flow itself, so one
        # continuous run peels, deforms and transports. maxCo bounds the FLUID
        # Courant; the released clot travels at about half the free stream, so
        # bounding the fluid at 0.5 bounds the phase-field advection below that
        # -- the phase equation is implicit (fvm::div) but Courant still
        # governs its ACCURACY, which is what smeared c_min from -0.997 to
        # -0.145 at Co = 16.2.
        if series.get("adjust_time_step"):
            for key, value in {
                "adjustTimeStep": "yes",
                "maxCo": series["max_co"],
                # PER CHILD. Setting deltaT alone is not a time-step control:
                # adjustTimeStep is on, so the solver climbs straight back to
                # the Courant limit within a few steps and the "quarter step"
                # child would run at the same step as its reference. The cap
                # has to move with it.
                "maxDeltaT": float(
                    child.get("max_delta_t_s", series["max_delta_t_s"])
                ),
                # adjustableRunTime, or the writer would miss its interval
                # whenever dt does not divide it and the frames would be
                # unevenly spaced -- which matters when the whole point is
                # watching something move.
                "writeControl": "adjustableRunTime",
            }.items():
                add_or_replace_entry(control, key, value)

        solution = case / "system" / "fvSolution"

        # A BENT MESH NEEDS NON-ORTHOGONAL CORRECTION, AND THIS CASE SHIPS
        # ZERO. That is correct for the orthogonal Cartesian block every other
        # child runs on, and it is a divergence risk on a throat child:
        # apply_throat.py scales y, which tilts the top of each cell against
        # its own height and skews cells by ~34 deg at a 100 um taper. With no
        # corrector the pressure equation carries that error uncorrected.
        # The skew is bounded above at build time (see the taper block); this
        # supplies the correction for what remains.
        if g.get("throat_height_m") is not None:
            add_or_replace_entry(solution, "nNonOrthogonalCorrectors", 2)
            # apply_throat.py reads this, not the controls file: the profile
            # has to travel WITH the case into Colab, and a case that carries
            # its own geometry cannot be bent with somebody else's numbers.
            (case / "constant" / "throatProfile.json").write_text(
                json.dumps(
                    {
                        "channel_height_m": g["channel_height_m"],
                        "throat_height_m": g["throat_height_m"],
                        "taper_start_x_m": g["taper_start_x_m"],
                        "taper_length_m": g["taper_length_m"],
                    },
                    indent=1,
                )
                + "\n"
            )

        # THE EULERIAN PATH NEEDS A DIFFERENT LINEAR-ALGEBRA CONFIGURATION,
        # and run 20260817-223437 proved it three ways in one log.
        #
        # (1) `Us` was declared PCG/DIC, which is a SYMMETRIC solver. Under
        #     eulerianSolid, solvePassiveDisplacement() builds
        #     ddt(Us) + div(phi,Us) - Sp(div(phi),Us) == U, and the div term
        #     makes that matrix ASYMMETRIC:
        #         "Unknown asymmetric matrix solver PCG"
        #     Fatal, after the be solves had already run. The non-Eulerian Us
        #     equation is symmetric, so PCG is correct there and is left alone.
        #
        # (2) `be` stalled: twelve solves in the only timestep that ran, each
        #     hitting maxIter 1000 with the residual pinned at 2.625e-07 and
        #     not moving at all. symGaussSeidel smoothing makes no progress on
        #     this matrix at this size. PBiCGStab with DILU is the standard
        #     choice for an advection-dominated asymmetric system and changes
        #     only HOW the linear system is solved, not the discretisation.
        #     The tolerance drops from the slab gate's 1e-14 -- which was set
        #     for a 800-cell periodic slab and is not reachable here -- to the
        #     1e-12 every other field in this case already uses.
        #
        # (3) `nOuterCorrectors` was 60, and 60 is the FSI number. The Eulerian
        #     path has ONE velocity, sets couplingIters_ = 1 and marks itself
        #     converged, so those correctors iterate a coupling that no longer
        #     exists. Leaving them would have measured the Eulerian solve
        #     wrapped in the very loop the reformulation removes -- and the
        #     speedup is the number this stage exists to produce. The slab
        #     gate that qualified the formulation ran nOuterCorrectors 1.
        for key, block in {
            "Us": "solver smoothSolver; smoother symGaussSeidel; "
                  "tolerance 1e-12; relTol 0;",
            "be": "solver PBiCGStab; preconditioner DILU; "
                  "tolerance 1e-12; relTol 0;",
        }.items():
            for name in (key, key + "Final"):
                text = solution.read_text()
                pattern = re.compile(
                    rf"(?m)^(\s*){re.escape(name)}\s*\{{[^}}]*\}}"
                )
                if not pattern.search(text):
                    raise ValueError(
                        f"{child_id}: no {name} block to replace in "
                        f"{solution}; the template changed shape"
                    )
                solution.write_text(
                    pattern.sub(rf"\g<1>{name} {{ {block} }}", text, count=1)
                )

        # 1 is the default on the Eulerian path for the reason in (3), but it
        # is now overridable per child. Run 20260818-081527 left a live
        # suspicion against it: with one outer corrector, c is advected on a
        # velocity lagged by a step and the c-U-be coupling is never iterated
        # within the step. diag_xu_base drifted to c = -1.029 -- the visible
        # noise in the q animation -- while nogrowth, sweeps10 and dt4 all
        # stayed bounded, their minimum over the whole run being the initial
        # condition. That is a DIFFERENT defect from the det(be) drift, which
        # those same three children did not move at all, and it needs its own
        # arm rather than being assumed away.
        outer = child.get("n_outer_correctors")
        if outer is None:
            outer = 1
        replace_entry(solution, "nOuterCorrectors", int(outer))
        # S17 B (design 2B): the PISO corrector count per child -- 6 with
        # final_relaxation_unity, 2-4 in the named discriminators. Absent,
        # the shared coupling number, so every earlier child is unchanged.
        replace_entry(solution, "nCorrectors",
                      int(child.get("n_correctors", coupling["nCorrectors"])))
        mural_gen.replace_relaxation(solution, "p", coupling["pRelaxation"])
        mural_gen.replace_relaxation(solution, "U", coupling["URelaxation"])

        # THE be ADVECTION SCHEME. Rewritten PER CHILD rather than in the
        # template, because the template is shared with
        # AM2020_straight_tube_growth_mechanics_smoke and three other cases,
        # and this is a probe rather than a decision.
        #
        # WHY. The clot creeps instead of coming to rest. Measured on run
        # 20260823-061943: at G = 400 Pa, 68.6 Kelvin-Voigt relaxation times
        # elapsed (tau_kv = mu/G = 8.75e-6 s), and lockRatio is still 0.207
        # where exponential decay predicts ~0; at G = 4 Pa and 3.43 tau it is
        # 0.722 against a predicted 0.032, off by 22x. An elastic solid at
        # constant load must not creep, so stored elastic strain is being
        # destroyed as fast as it accumulates.
        #
        # be is advected with `Gauss limitedLinear 1`, and k = 1 is the MOST
        # limiting setting -- the limiter is max(min(2r/k, 1), 0), verified in
        # OpenFOAM-13's limitedLinear.hpp, so k = 1 blends hardest toward
        # upwind and is the most diffusive of the family. Run 20260818-223637
        # established by Minkowski's determinant inequality that diffusive
        # advection of be can only inflate det(be), and measured 12473 of 12500
        # cells with det > 1.
        #
        # WHY THE det RENORMALISATION TEST DID NOT ALREADY KILL THIS. Run
        # 20260818-232055 pinned det(be) to 1e-14 by rescaling and lockRatio
        # did not move, which reads like a refutation. It is not: rescaling
        # fixes only the TRACE. That run's own record says it "does not stop
        # the deviatoric part being smeared" -- and dev(be) is what carries the
        # stress. The hypothesis that smearing destroys the restoring force has
        # never actually been tested.
        if child.get("be_div_scheme"):
            schemes = case / "system" / "fvSchemes"
            text = schemes.read_text()
            text, n = re.subn(
                r"(?m)^(\s*div\(phi,be\)\s+)[^;]+;",
                lambda m: f"{m.group(1)}{child['be_div_scheme']};",
                text, count=1,
            )
            if n != 1:
                raise ValueError(
                    f"{schemes}: could not rewrite div(phi,be); the template's "
                    f"fvSchemes layout changed"
                )
            schemes.write_text(text)
            got = re.search(r"(?m)^\s*div\(phi,be\)\s+([^;]+);",
                            schemes.read_text())
            if not got or got.group(1).strip() != child["be_div_scheme"]:
                raise ValueError(
                    f"{schemes}: div(phi,be) readback mismatch -- wanted "
                    f"{child['be_div_scheme']!r}, found "
                    f"{got.group(1).strip() if got else None!r}"
                )



        # S14: the rupture label's linear solver, ONLY for a child that
        # solves it, so every other child's fvSolution is byte-identical. The
        # same insertion the nP entry uses, refused rather than skipped if the
        # template's anchor is gone. It is shell_'s class of equation (the
        # same shared operator on the same carrier), so it takes shell's
        # solver, $c.
        rupture = True
        solution_text = solution.read_text()
        if not re.search(r"(?m)^\s*ruptured\s*\{", solution_text):
            entry = (
                "    // S14: the rupture label, carried with the material "
                "like shell.\n"
                "    ruptured { $c; }\n"
                "    rupturedFinal { $c; relTol 0; }\n"
            )
            patched, count = re.subn(
                r"(?m)^\}\n\nPIMPLE", entry + "}\n\nPIMPLE",
                solution_text, count=1)
            if count != 1:
                raise SystemExit(
                    "child %s: could not place the ruptured solver entry "
                    "in fvSolution -- the `}\\n\\nPIMPLE` anchor is gone, "
                    "and a silent skip would fatal in Colab"
                    % (child.get("child_id"),)
                )
            solution.write_text(patched)
        # S16 C: the partition's three linear solvers, ONLY for a child that
        # carries it, placed exactly as the ruptured entry is. They are
        # thetaC's class of equation (the same conservative upwind matrix on
        # the same carrier), so they take thetaC's solver, $c.
        partition = True
        solution_text = solution.read_text()
        if not any(re.search(rf"(?m)^\s*{f}\s*\{{", solution_text)
                   for f in S16_PARTITION_FIELDS):
            entry = (
                "    // S16: the damage partition of the network content, "
                "carried like thetaC.\n"
                + "".join(
                    f"    {f} {{ $c; }}\n    {f}Final {{ $c; relTol 0; }}\n"
                    for f in S16_PARTITION_FIELDS)
            )
            patched, count = re.subn(
                r"(?m)^\}\n\nPIMPLE", entry + "}\n\nPIMPLE",
                solution_text, count=1)
            if count != 1:
                raise SystemExit(
                    "child %s: could not place the damage-partition solver "
                    "entries in fvSolution -- the `}\\n\\nPIMPLE` anchor "
                    "is gone, and a silent skip would fatal in Colab"
                    % (child.get("child_id"),)
                )
            solution.write_text(patched)

        # S16 E (H-P4, em65 only): the final PIMPLE iteration relaxes neither
        # U nor p. OpenFOAM 13 looks up "<name>Final" on the final iteration
        # (GeometricField::relax -> relaxField("pFinal"), fvMatrix::relax ->
        # relaxEquation("UFinal")); with no entry it falls back to the
        # non-final factors, 0.7 and 0.3, on EVERY step. Dictionary-only.
        if child.get("final_relaxation_unity"):
            solution_text = solution.read_text()
            if re.search(r"(?m)^\s*(UFinal|pFinal)\s+[0-9]", solution_text):
                raise SystemExit(
                    "child %s: fvSolution already declares a UFinal/pFinal "
                    "relaxation factor; final_relaxation_unity would be "
                    "ambiguous" % (child.get("child_id"),))
            patched, n_p = re.subn(
                r"(?m)^(\s*fields\s*\{\s*\n\s*p\s+[^;]+;\n)",
                lambda m: m.group(1) + "        pFinal 1;\n",
                solution_text, count=1)
            patched, n_u = re.subn(
                r"(?m)^(\s*equations\s*\{\s*\n\s*U\s+[^;]+;\n)",
                lambda m: m.group(1) + "        UFinal 1;\n",
                patched, count=1)
            if n_p != 1 or n_u != 1:
                raise SystemExit(
                    "child %s: could not place pFinal/UFinal in fvSolution's "
                    "relaxationFactors (fields p %d, equations U %d) -- the "
                    "template layout changed, and a silent skip would ship "
                    "em64 under em65's name"
                    % (child.get("child_id"), n_p, n_u))
            solution.write_text(patched)
        # S17 A: the pending fibre's two carried contents, placed as the
        # partition's are, with thetaC's solver (the same conservative upwind
        # matrix on the same carrier). S17 C: the Vs predictor's solver --
        # asymmetric M-matrix, so smoothSolver (U's) or PBiCGStab.
        fibre = True
        implicit_tangent = True
        solution_text = solution.read_text()
        entry = ""
        if any(re.search(rf"(?m)^\s*{f}\s*\{{", solution_text)
               for f in S17_FIBRE_FIELDS):
            raise SystemExit(
                "child %s: fvSolution already declares a pending-fibre "
                "solver; the template changed" % (child.get("child_id"),))
        # S17b section 8 a: the passengers run P2's fixed entry
        entry += (
            "    // S17 A: the pending fibre's contents, carried with "
            "thetaPending (S17b section 8 a: its fixed entry).\n"
            + "".join(
                f"    {f} {S17B_FIXED_SWEEP_ENTRY}\n"
                f"    {f}Final {S17B_FIXED_SWEEP_ENTRY}\n"
                for f in S17_FIBRE_FIELDS)
        )
        if re.search(r"(?m)^\s*Vs\s*\{", solution_text):
            raise SystemExit(
                "child %s: fvSolution already declares Vs; the "
                "template changed" % (child.get("child_id"),))
        vs_solver = str(child.get("vs_linear_solver", S17_VS_SOLVERS[0]))
        # Refuse rather than substitute (report 14): the branch below writes
        # PBiCGStab for any value but smoothSolver, so PCG -- invalid on this
        # asymmetric matrix -- would otherwise run silently as PBiCGStab.
        if vs_solver not in S17_VS_SOLVERS:
            raise SystemExit(
                "child %s: vs_linear_solver %r is not one of %s; the Vs "
                "matrix is asymmetric, so PCG is never valid"
                % (child.get("child_id"), vs_solver, S17_VS_SOLVERS))
        vs_body = (
            "solver smoothSolver; smoother symGaussSeidel;"
            if vs_solver == "smoothSolver"
            else "solver PBiCGStab; preconditioner DILU;"
        )
        entry += (
            "    // S17 C: the Vs predictor (asymmetric M-matrix).\n"
            f"    Vs {{ {vs_body} tolerance 1e-12; relTol 0; }}\n"
            "    VsFinal { $Vs; relTol 0; }\n"
        )
        patched, count = re.subn(
            r"(?m)^\}\n\nPIMPLE", entry + "}\n\nPIMPLE",
            solution_text, count=1)
        if count != 1:
            raise SystemExit(
                "child %s: could not place the S17 solver entries in "
                "fvSolution -- the `}\\n\\nPIMPLE` anchor is gone"
                % (child.get("child_id"),))
        solution.write_text(patched)
        # S17b A1 (V1): the taut share's solver, and the SAME fixed-sweep
        # entry for the pending content and the taut share (the solver
        # refuses anything else: theta_T <= theta_P exactly, 06).
        slack_v1 = str('v1') == "v1"
        solution_text = solution.read_text()
        old_pend = ("    thetaPending { $c; }\n"
                    "    thetaPendingFinal { $c; relTol 0; }\n")
        if solution_text.count(old_pend) != 1:
            raise SystemExit(
                "child %s: the S16 thetaPending entries are not where the "
                "S17b fixed-sweep rewrite expects them" % (child.get("child_id"),))
        if re.search(r"(?m)^\s*thetaTaut\s*\{", solution_text):
            raise SystemExit(
                "child %s: fvSolution already declares thetaTaut; the "
                "template changed" % (child.get("child_id"),))
        new_pend = (
            "    // S17b A1 and section 8 a: the pending content, its taut "
            "share and the two passengers\n"
            "    // run ONE fixed-sweep entry (theta_T <= theta_P exactly; "
            "the solver refuses anything else).\n"
            + "".join(f"    {f} {S17B_FIXED_SWEEP_ENTRY}\n"
                      for f in ("thetaPending", "thetaPendingFinal",
                                "thetaTaut", "thetaTautFinal")))
        solution.write_text(solution_text.replace(old_pend, new_pend))
        verify_linear_solvers_exist(
            solution, core_shell, eulerian_solid, rupture=rupture,
            partition=partition, fibre=fibre,
            implicit_tangent=implicit_tangent, slack_v1=slack_v1)
        verify_platelet_carrier_scheme(case / "system" / "fvSchemes")

        (case / "case.foam").write_text("")

        spec = {
            "child_id": child_id,
            # The mural initialiser accepts this experiment name; it draws the
            # same one-sided cap for all three of them.
            "experiment": "planar_mural_detachment",
            "series": child["series"],
            "role": series["question"],
            "wall_shear_rate_1_s": gamma,
            "mean_velocity_m_s": u_mean,
            "G_pa": child["G_pa"],
            "k0p_m2": child["k0p_m2"],
            "wall_condition": series["wall_condition"],
            "adhesive_strength_pa": child["adhesive_strength_pa"],
            "growth": growing,
            "alpha_m2_s": alpha_used,
            "alpha_physiological_m2_s": growth["alpha_m2_s"],
            "growth_rate_multiplier": growth_multiplier,
            "delta_t_s": float(child.get("delta_t_s", series["delta_t_s"])),
            "end_time_s": end_time,
            "expected_steps": series["steps"],
            "write_interval_s": write_interval,
            "geometry": g,
            "clot": clot,
            "Gc_j_m2": child.get("Gc_j_m2") or fixed["Gc_j_m2"],
            "core_shell": core_shell,
            "G_core_pa": child.get("G_core_pa"),
            # The REALISED core toughness, not the declared one. The
            # source-33 child declares neither, and the report has to state
            # the 5.547 J/m2 it actually ran with rather than a null that
            # reads as "no toughness was set".
            "Gc_core_j_m2": (
                child.get("Gc_core_j_m2")
                or (child.get("Gc_j_m2") or fixed["Gc_j_m2"])
            ),
            "k0p_core_m2": child.get("k0p_core_m2"),
            # tear_measurement locates the tear by Dg > DgDetach, so the
            # threshold has to travel with the child rather than being
            # re-guessed downstream.
            "DgDetach": fixed["DgDetach"],
            # THE ANCHOR, IN THE MANIFEST. Recorded so the assessor can check
            # that the case a child ran actually carried the anchor its
            # child_id claims, instead of trusting the id.
            "release_support_limit": float(
                child.get("release_support_limit", 0.0)
            ),
            "bond_hold_dilation": int(child.get("bond_hold_dilation", -1)),
            "bond_hold_penalty": float(child.get("bond_hold_penalty", 0.0)),
            # No transport_stage key: the restart is gone and nothing reads it.
            # Leaving a dead field in matrix.json invites the next reader to
            # believe a two-stage run still happens.
            "adjust_time_step": bool(series.get("adjust_time_step")),
            "max_co": series.get("max_co"),
        }
        # S16: what the solver's PHASE_CLOT_S16_CONFIG line must echo (the
        # config gate), and the two run-control keys -- ONLY for a child that
        # names an S16 key, so every earlier child.json is byte-identical.
        _s16_keys = ("damage_transport", "skeleton_floor_basis",
                     "network_pressure_damage", "network_release_time_s",
                     "network_damage_progressive")
        _term = True
        _prog = bool(True)
        spec["s16"] = {
            "networkCriticalStretch": float(
                child.get("network_critical_stretch", 2.8)),
            "networkReleaseTime": float(
                child.get("network_release_time_s", 3.3e-4)),
            "networkDamageOnsetStretch": float(
                child.get("network_damage_onset_stretch", 2.105)),
            "networkDamageStretchScale": float(
                child.get("network_damage_stretch_scale", 1.0)),
            "releaseLaw": ('progressive+terminal'),
        }
        if child.get("final_relaxation_unity"):
            spec["final_relaxation_unity"] = True
        if "wall_budget_h" in child:
            spec["wall_budget_h"] = float(child["wall_budget_h"])
        # S17: what PHASE_CLOT_S17_CONFIG must echo, the corrector count and
        # the discriminator role -- ONLY for a child that names an S17 key,
        # so every earlier child.json is byte-identical.
        _s17_keys = ("elastic_coupling", "n_correctors", "s17_discriminator",
                     "vs_linear_solver")
        spec["s17"] = {
            "pendingForm": ('fibre'),
            "nCorrectors": int(child.get("n_correctors",
                                         coupling["nCorrectors"])),
        }
        # S17b: what PHASE_CLOT_S17B_CONFIG must echo -- ONLY for a child
        # that names an S17b key, so every earlier child.json is identical.
        if "network_pressure_weight" in child:
            spec["s18"] = {
                "networkPressureWeight": str(child["network_pressure_weight"]),
            }
        if any(k in child for k, _d, _a in S18P_SWITCHES):
            spec["s18p"] = {"projSiteLog": str(child.get("proj_site_log", "no"))}
        if any(k in child for k in S19_KEYS):
            spec["s19"] = {
                "momentumPressureCoupling": str(
                    child.get("momentum_pressure_coupling", "segregated")),
                "networkViscosity": str(child.get("network_viscosity", "none")),
                "networkViscosityTau": float(child.get("tau_v_s", 2.5e-4)),
            }
        # S22: what the PHASE_CLOT_S19_CONFIG echo must show, and (A0) which
        # library the child runs on -- ONLY for a child that names an S22 key,
        # so every earlier child.json is byte-identical. Read by
        # scripts/s22_validation_compare.py.
        if any(_k in child for _k, _d in S22_KEYS):
            spec["s22"] = {_k: child[_k] for _k, _d in S22_KEYS if _k in child}
        if "solver_library" in child:
            spec["solver_library"] = str(child["solver_library"])
        if any(_k in child for _k, _d in S23_KEYS):
            spec["s23"] = {_k: float(child[_k]) for _k, _d in S23_KEYS if _k in child}
        if "sensitivity_limit" in child:
            spec.setdefault("s23", {})["sensitivity_limit"] = str(child["sensitivity_limit"])
        if "rho_kg_m3" in series:
            spec["plasma"] = {"rho_kg_m3": series["rho_kg_m3"],
                              "nu_m2_s": series["nu_m2_s"]}
        if series.get("flow_axis"):
            spec["pressure_driven"] = bool(series.get("pressure_driven"))
            spec["declared_mean_velocity_m_s"] = child["mean_velocity_m_s"]
        if "release_core_exempt" in child:
            spec["release_core_exempt"] = bool(child["release_core_exempt"])
        if child.get("platelet_transport_only"):
            spec["platelet_transport_only"] = True
            spec["platelet_inlet_density_m3"] = float(
                child.get("platelet_inlet_density_m3",
                          child.get("platelet_number_density_m3", 0.0)))
        if dome_margins is not None:
            spec["dome_floors"] = dome_margins
        if "seed_theta_peak" in child:
            # V21-8 (S20 V5): the seeded q < 0 test. 0/thetaC and 0/thetaH are
            # patched after the initialiser and seed_stage2_content.py have run;
            # Allrun calls seed_theta_peak.py when this file exists (the pattern
            # of constant/throatProfile.json).
            _sd = child["seed_theta_peak"]
            _tref = float(child["skeleton_volume_fraction"])
            if set(_sd) != {"theta", "cells"} or not float(_sd["theta"]) > _tref \
                    or int(_sd["cells"]) < 1:
                raise ValueError(
                    f"{child_id}: seed_theta_peak needs exactly theta and cells, with "
                    f"theta above skeleton_volume_fraction {_tref} (so q = 1 - theta/"
                    f"theta_ref < 0) and cells >= 1; got {_sd!r}")
            _sd = {"theta": float(_sd["theta"]), "cells": int(_sd["cells"]),
                   "theta_ref": _tref}
            (case / "constant" / "seedTheta.json").write_text(
                json.dumps(_sd, indent=1) + "\n")
            spec["seed_theta_peak"] = _sd
        dictionary_key_refusal(case / "constant" / "phaseClotProperties")
        if child.get("solver_library") in SOLVER_LIBRARIES:
            # S22 section 7 arm A0 / S23 section 7 arm (1'): Allrun sources this before foamRun.
            (case / "child.env").write_text(solver_library_env(child["solver_library"]))
        (case / "child.json").write_text(json.dumps(spec, indent=2) + "\n")
        matrix.append(spec)

    order = {'throat_embolus': 21,
    'xu2017_fig5': 22,
    'throat_free_shell': 23}
    missing = {c["series"] for c in controls["children"]} - set(order)
    if missing:
        raise ValueError(f"no run order declared for series {sorted(missing)}")
    within = {'em76_G4_s18': 16,
    'em78_G4_s18p_ctl': 21,
    'em79_G4_s18p_Cnp': 22,
    'xu5_g250': 17,
    'xu5_g500': 18,
    'xu5_g1000': 19,
    'xu5_g2000': 20,
    'fsf_G40_klo_v250': 1,
    'fsf_G4_kxu_v250': 2,
    'fsb_G40_klo_rest': 3,
    'fs_G40_klo_rest_seg': 4,
    'em78_id': 25,
    'xu5_id': 21,
    'fsf_G4_kxu_r3_40': 5,
    'fsf_G4_kxu_nor3_40': 6,
    'fsf_G4_kxu_seedq': 0,
    # BATCH 71: A0 (the cheap control) first, then the case 1 arms, then case 2.
    'S22v_G40_A0': 30,
    'S22v_G40_A': 31,
    'S22v_G40_E': 32,
    'S22v_G40_B': 33,
    'S22v_G40_C': 34,
    'S22v_G40_D': 35,
    'S22v_G4_A': 36,
    'S22v_G4_C': 37,
    # BATCH 72: the matrix order is (1), (1'), (2), (3); the notebook's pairs follow it.
    'S23v_G4_r3': 40,
    'S23v_G4_r3_S22': 41,
    'S23v_G4_nor3': 42,
    'S23v_G40': 43,
    # BATCH 73: the production pair, loose first (it is the shorter run), after the BATCH 72 arms.
    'P73_G4_r3_40': 50,
    'P73_G40_40': 51,
    # BATCH 74: the one diagnostic child, after the BATCH 73 pair.
    'D74_refI': 60,
    # BATCH 75: the re-run of the diagnostic child on rev. 3.1, after D74_refI.
    'D75_refI': 61}
    # S17 (design section 6): the mechanical control check, before anything
    # is ordered for launch. The controls-entry half runs for every pair; the
    # child.json half for the pairs whose control is built in this run
    # (s17_control_check; fix round 13c).
    s18p_parent_check({c["child_id"]: c for c in controls["children"]})
    fsf_parent_check({c["child_id"]: c for c in controls["children"]})
    # Xu (design section 1): each child is em76 but for the design's keys.
    xu_parent_check({c["child_id"]: c for c in controls["children"]})
    matrix.sort(key=lambda s: (order[s["series"]],
                               within.get(s["child_id"], 0),
                               s["child_id"]))
    (RUNS / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"generated {len(matrix)} main-case children")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"create_cases.py: {error}")
