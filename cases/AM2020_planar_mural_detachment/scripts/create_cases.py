#!/usr/bin/env python3
"""Generate the fifteen main-case children.


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


CASE = Path(__file__).resolve().parents[1]
TEMPLATE = CASE.parent / "AM2020_straight_tube_growth_mechanics_smoke" / "template"
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


# The source-33 family: one replication child and the two stage-3 cost
# children. They share a material, an operating point and a set of validation
# checks; they differ only in geometry, window and the eulerianSolid switch.
SOURCE33_SERIES = (
    "source33_replication",
    "eulerian_stage3_xu",
    "eulerian_stage3_li",
    "eulerian_diag_xu",
    "eulerian_diag_li",
    # Cloned from eulerian_diag_xu -- same material, same operating point, same
    # unbreakable wall -- and differing only in the window. Declared a member
    # of the family whose validation rules it follows, rather than exempted
    # from the cohesive-bound check it correctly tripped.
    "eulerian_drift_probe",
    # Cloned from eulerian_diag_xu for the same reason and on the same terms:
    # same material, same operating point, same unbreakable wall, differing in
    # the window and (for two of the three children) in growth and G. A family
    # member so it inherits the family's validation rather than being exempted
    # from it.
    "eulerian_saturation",
    # Cloned from eulerian_saturation, which was cloned from eulerian_diag_xu:
    # same material, same operating point, same unbreakable wall. A family
    # member so it inherits the family's validation.
    "eulerian_beterm",
    # Cloned from eulerian_beterm on the same terms: same material, same
    # operating point, same unbreakable wall, differing only in the be
    # advection scheme, the timestep ceiling and the window.
    "eulerian_spdfix",
    # Cloned from eulerian_spdfix's bond_sat on the same terms: same material,
    # same operating point, same be scheme, same window. What differs is the
    # ANCHOR -- the release-support threshold and the bond-gated hold -- and
    # for one child the adhesive strength. A family member so it inherits the
    # family's validation rather than being exempted from it.
    "eulerian_anchor",
    "eulerian_seated",
    # Cloned from eulerian_seated's sd_deep on the same terms: same seated
    # core, same mesh, same tapered hold, same unbreakable wall. What differs
    # is G -- restored to source 33's own 4/40 in st_xu, held at the inherited
    # 40/400 in st_ctl as the control -- and the window. A family member so it
    # inherits the family's validation rather than being exempted from it.
    "eulerian_stiffness",
    # Cloned from eulerian_stiffness's st_xu, byte for byte -- same sourced
    # 4/40 material, same seated geometry, same window -- and differing in
    # ONE switch, beDetRenormalise. A family member so it inherits the
    # family's validation rather than being exempted from it.
    "eulerian_detrenorm",
    # Cloned from eulerian_detrenorm's dr_on, byte for byte -- same sourced
    # 4/40 material, same seated geometry, same 24 ms window, same unbreakable
    # bond, det renormalisation ON -- and differing in ONE integer,
    # bondHoldDilation. Built on dr_on rather than st_xu because run
    # 20260827-005457 withdrew st_xu/st_ctl as physics. A family member so it
    # inherits the family's validation rather than being exempted from it.
    "eulerian_anchor_depth",
    # Cloned from eulerian_anchor_depth's ad_d20 (= dr_on), differing in
    # growth_rate_multiplier and, for two of three children, in
    # adhesive_strength_pa. THE FIRST SERIES TO RESTORE GROWTH: it had been 0.0
    # in every seated child, so Dr Li's actual question had never been run.
    # A family member so it inherits the family's validation.
    "eulerian_growth_peel",
    "eulerian_peel_propagation",
    # Cloned from eulerian_growth_peel's gr_hold, byte for byte -- same 4/40
    # material, same seated core, same 24 ms window, same unbreakable bond,
    # same growth multiplier -- and differing in the MOMENTUM ANCHOR alone:
    # eulerianBrinkman, and for two children the bond-hold keys it replaces.
    # A family member so it inherits the family's validation rather than being
    # exempted from it.
    "eulerian_brinkman",
    # Cloned from eulerian_brinkman's bk_soft and bk_on, byte for byte --
    # Brinkman on, bond hold off, same 4/40 material, same seated core, same
    # unbreakable bond -- and differing in ONE key, end_time_s, 24 -> 50 ms.
    # A family member so it inherits the family's validation rather than being
    # exempted from it.
    "eulerian_core_fate",
    # Cloned from eulerian_core_fate's cf_xu, byte for byte -- Brinkman on,
    # bond hold off, seated core, unbreakable bond, source 33's 1e-14/1e-11
    # permeabilities -- and differing in TWO keys: G_pa, the SHELL modulus,
    # and the window, 50 -> 30 ms. Dr Li's 2026-09-01 request. A family member
    # so it inherits the family's validation rather than being exempted from
    # it, including the stiffness-contrast check its g40 child must declare an
    # exception to.
    "shell_modulus_sweep",
    # Dr Li's 2026-09-01 occlusion request. A source-33 child in every respect
    # that these checks cover -- same 4/40 Pa material, same 1e-11/1e-14
    # permeabilities, seated core, growth multiplier 1000 -- and differing in
    # the GEOMETRY (a narrowing channel) and the BOUNDARY CONDITION (pressure
    # driven, because a fixed inlet velocity cannot show an occlusion). Listed
    # here rather than branched, for the reason the family comment gives: a
    # separate branch is how a series quietly acquires weaker validation than
    # the one it was copied from.
    "throat_occlusion",
    # The lodged-embolus arm of the same study. Same material family, same
    # operating point, same pressure-driven boundary condition, differing in
    # WHERE the clot starts (inside the throat) and in waiving the core -- a
    # detached fragment has no core by construction, because in source 33 the
    # core is precisely the part that stays on the wall. Listed here so it
    # inherits the family's validation rather than being exempted from it.
    "throat_lodged",
    # The EMBOLUS arm. Same material family and operating point again,
    # differing in the bond strength (weak enough that every face fails) and
    # in the solver term it exercises: the relative-velocity Brinkman drag
    # added 2026-09-04, without which a released body carries no drag and
    # cannot obstruct anything. Listed here so it inherits the family's
    # validation rather than being exempted from it.
    "throat_embolus",
)


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
    if eulerian_solid:
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
    if rupture:
        required.append("ruptured")
    # S16: the damage partition, solved (once damage exists) under
    # damage_transport contentPartition -- added in the same edit as the
    # fields.
    if partition:
        required += list(S16_PARTITION_FIELDS)
    # S17: the pending fibre's contents under fibreRelease, and the Vs
    # predictor under implicitTangent -- added in the same edit as the
    # solves.
    if fibre:
        required += list(S17_FIBRE_FIELDS)
    if implicit_tangent:
        required.append("Vs")
    # S17b A1: V1's taut share
    if slack_v1:
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



# See validate_stage2_platelet: the largest uniform theta gain a Stage 2
# child may deposit while the accepted transfer is unlocalised.
UNLOCALISED_DEPOSITION_CEILING = 1e-3


# S10 (2026-09-22): (child key, dictionary key, accepted values). The first
# value of each is the solver's shipped default. Kept equal to the solver's
# own refusals in its constructor (PHASE_CLOT_S10_CONFIG).
S10_SWITCHES = (
    ("network_pressure_reference", "networkPressureReference",
     ("nominal", "deposited")),
    ("content_transport_scheme", "contentTransportScheme",
     ("upwind", "fct", "limitedLinear")),
    ("released_carrier", "releasedCarrier",
     ("rigidBody", "material")),
)


S11_SWITCHES = (
    ("skeleton_momentum_form", "skeletonMomentumForm",
     ("conservative", "phaseContinuity")),
    ("block_face_inverse", "blockFaceInverse",
     ("cellInverse", "faceAssembled")),
    ("stress_face_transmission", "stressFaceTransmission",
     ("linear", "series")),
    ("network_pressure_force", "networkPressureForce",
     ("stressDivergence", "faceGradient")),
    ("be_relax_basis", "beRelaxBasis",
     ("blood", "networkAbsent")),
    ("explicit_stability_bounds", "explicitStabilityBounds",
     ("off", "on")),
)


def s11_combination_refusal(child: dict) -> None:
    """Refuse, for EVERY child, an S11 combination the solver refuses.

    The same dependencies the solver's constructor enforces, checked here so
    a wrong combination costs nothing instead of a VM boot. Each is a failure
    that would otherwise ship silently, and the reason is stated with it
    (research/designs/2026-09-23-S11-root-causes-em52.md).
    """
    cid = child["child_id"]
    present = [k for k, _, _ in S11_SWITCHES if k in child]
    convex = child.get("network_pressure_model") == "convexExcess"
    if (present or convex) and not (
        child.get("two_velocity_skeleton") and child.get("eulerian_solid")
    ):
        raise ValueError(
            f"{cid}: the S11 switches act on the two-velocity Eulerian "
            f"skeleton; they need two_velocity_skeleton and eulerian_solid."
        )
    if convex and child.get("network_pressure_reference") != "deposited":
        raise ValueError(
            f"{cid}: network_pressure_model convexExcess weights by the "
            f"DEPOSITED occupancy thetaNetRef/phi_s0; it needs "
            f"network_pressure_reference deposited."
        )
    if child.get("network_pressure_force") == "faceGradient":
        if not convex:
            raise ValueError(
                f"{cid}: network_pressure_force faceGradient is a compact "
                f"force and is well-posed only for a monotone partial "
                f"pressure; it needs network_pressure_model convexExcess."
            )
        if child.get("explicit_stability_bounds") != "on":
            raise ValueError(
                f"{cid}: network_pressure_force faceGradient is explicit in "
                f"the content; it needs explicit_stability_bounds on."
            )
    if child.get("be_relax_basis") == "networkAbsent" and not (
        child.get("network_pressure_reference") == "deposited"
        and child.get("block_face_inverse") == "faceAssembled"
    ):
        raise ValueError(
            f"{cid}: be_relax_basis networkAbsent keeps the clot edge's "
            f"strain, so it needs network_pressure_reference deposited and "
            f"block_face_inverse faceAssembled (under cellInverse a "
            f"clot-surface face carries the lumen's skeleton mobility)."
        )


S12_SWITCHES = (
    ("drag_occupancy_basis", "dragOccupancyBasis",
     ("legacy", "intrinsic")),
)


def s12_combination_refusal(child: dict, permeability_model: str) -> None:
    """Refuse, for EVERY child, an S12 combination the solver refuses.

    dragOccupancyBasis intrinsic gives a partial cell its network's OWN drag,
    ind*(1-a)^2*nu/k_KC(a), weighted as the stress already is
    (research/designs/2026-09-23-S12-intrinsic-occupancy-drag.md). Each
    dependency is the solver constructor's, checked here so a wrong
    combination costs nothing instead of a VM boot. `permeability_model` is
    the child's RESOLVED value, the one written to the dictionary.
    """
    if child.get("drag_occupancy_basis", "legacy") != "intrinsic":
        return
    cid = child["child_id"]
    if not (child.get("two_velocity_skeleton") and child.get("eulerian_solid")):
        raise ValueError(
            f"{cid}: drag_occupancy_basis intrinsic weights the drag of the "
            f"two-velocity 2x2 block; it needs two_velocity_skeleton and "
            f"eulerian_solid."
        )
    if not (child.get("network_pressure_reference") == "deposited"
            and child.get("network_pressure_model") == "convexExcess"):
        raise ValueError(
            f"{cid}: drag_occupancy_basis intrinsic evaluates the "
            f"permeability at the intrinsic density a = phi_s0 + theta - "
            f"thetaNetRef, which exists only under network_pressure_reference "
            f"deposited and is held inside (0, phiSMax) only by "
            f"network_pressure_model convexExcess."
        )
    if permeability_model != "kozenyCarmanSolid":
        raise ValueError(
            f"{cid}: drag_occupancy_basis intrinsic re-evaluates the "
            f"kozenyCarmanSolid law at a; permeability_model is "
            f"{permeability_model!r}."
        )
    if float(child.get("skeleton_dilute_drag_rate", 0.0)) > 0:
        raise ValueError(
            f"{cid}: drag_occupancy_basis intrinsic gives every partial cell "
            f"the network's own drag (1.008e7*ind 1/s in the shell); "
            f"skeleton_dilute_drag_rate "
            f"{child['skeleton_dilute_drag_rate']} would add the calibrated "
            f"floor this basis retires (OI-31). Set it to 0."
        )


S13_SWITCHES = (
    # "contentRatio": S18 C (R2b), networkPresence x theta/thetaNetRef
    # (research/designs/2026-09-29-S18-...md section 2 C; DECISIONS).
    ("network_pressure_weight", "networkPressureWeight",
     ("depositedOccupancy", "networkPresence", "contentRatio")),
    ("be_stretch_gradient", "beStretchGradient",
     ("cellGradient", "fluxConsistent")),
)


def s13_combination_refusal(child: dict) -> None:
    """Refuse, for EVERY child, an S13 combination the solver refuses.

    research/designs/2026-09-23-S13-grid-scale-restoring.md. The two fixes of
    run 20260923-135545's grid-scale mode: the convexExcess partial pressure
    weighted by network PRESENCE (the excess is already partial; S11 counted
    the occupancy twice), and be's stretch built from the flux that carries
    be and the content.
    """
    cid = child["child_id"]
    _npw = child.get("network_pressure_weight")
    if _npw is not None and str(_npw) not in S13_SWITCHES[0][2]:
        raise ValueError(
            f"{cid}: network_pressure_weight {str(_npw)!r} is not one of "
            f"{', '.join(S13_SWITCHES[0][2])}."
        )
    if _npw in ("networkPresence", "contentRatio"):
        if child.get("network_pressure_model") != "convexExcess":
            raise ValueError(
                f"{cid}: network_pressure_weight {_npw} re-weights "
                f"the convexExcess partial pressure; it needs "
                f"network_pressure_model convexExcess."
            )
        if not float(child.get("residual_chi", 0.0)) > 0:
            raise ValueError(
                f"{cid}: network_pressure_weight {_npw} ramps to zero "
                f"below residual_chi of deposited network, which keeps the "
                f"lumen free of network pressure; residual_chi must be "
                f"positive."
            )
    if (_npw == "contentRatio"
            and child.get("network_pressure_reference") != "deposited"):
        raise ValueError(
            f"{cid}: network_pressure_weight contentRatio weights the intact "
            f"network pressure by theta/thetaNetRef; it needs "
            f"network_pressure_reference deposited (S18 section 2 C)."
        )
    if child.get("be_stretch_gradient") == "fluxConsistent" and not (
        child.get("two_velocity_skeleton") and child.get("eulerian_solid")
    ):
        raise ValueError(
            f"{cid}: be_stretch_gradient fluxConsistent builds be's stretch "
            f"from the two-velocity skeleton flux; it needs "
            f"two_velocity_skeleton and eulerian_solid."
        )


S14_SWITCHES = (
    ("network_rupture", "networkRupture",
     ("off", "criticalStretch", "progressiveDamage", "terminalRelease",
      "fibreRelease")),
    ("elastic_stability_stiffness", "elasticStabilityStiffness",
     ("modulus", "tangent")),
)


def s14_combination_refusal(child: dict) -> None:
    """Refuse, for EVERY child, an S14 combination the solver refuses.

    research/designs/2026-09-23-S14-network-rupture.md: network ruptures at
    the critical principal stretch (Tutwiler et al. 2020, source 28: 2.8),
    and the elastic dt bound reads the stretch-dependent tangent stiffness.
    """
    cid = child["child_id"]
    # A MISSPELT VALUE IS REPORTED AS A MISSPELT VALUE, first. Otherwise a
    # typo in network_rupture would surface as "network_critical_stretch is
    # silently inert" -- true, but the wrong reason, and it sends the reader
    # to the wrong key.
    for _key, _dict_key, _allowed in S14_SWITCHES:
        if _key in child and str(child[_key]) not in _allowed:
            raise ValueError(
                f"{cid}: {_key} {str(child[_key])!r} is not one of "
                f"{', '.join(_allowed)}."
            )
    tangent = child.get("elastic_stability_stiffness") == "tangent"
    if tangent and child.get("explicit_stability_bounds") != "on":
        raise ValueError(
            f"{cid}: elastic_stability_stiffness tangent changes the number "
            f"explicit_stability_bounds on bounds dt by; it needs "
            f"explicit_stability_bounds on."
        )
    # S16: terminalRelease reads the rupture stretch too (it is lc); S17's
    # fibreRelease is terminalRelease's event and release.
    if "network_critical_stretch" in child and child.get(
            "network_rupture") not in ("criticalStretch", "terminalRelease",
                                       "fibreRelease"):
        raise ValueError(
            f"{cid}: network_critical_stretch is read only under "
            f"network_rupture criticalStretch, terminalRelease or "
            f"fibreRelease; set without them, it would be silently inert."
        )
    # S15: the two progressive-damage parameters (Kliuchnikov et al. 2025,
    # source 32, Eq. 6) are read only under progressiveDamage -- and, S16,
    # under terminalRelease with its progressive part on (refused as inert
    # with it off in s16_combination_refusal).
    for _k in ("network_damage_onset_stretch", "network_damage_stretch_scale"):
        if _k in child and child.get("network_rupture") not in (
                "progressiveDamage", "terminalRelease", "fibreRelease"):
            raise ValueError(
                f"{cid}: {_k} is read only under network_rupture "
                f"progressiveDamage, terminalRelease or fibreRelease; set "
                f"without them, it would be silently inert."
            )
    if child.get("network_rupture") == "progressiveDamage":
        lf = float(child.get("network_damage_onset_stretch", 2.105))
        dlf = float(child.get("network_damage_stretch_scale", 1.0))
        if not (lf > 1 and dlf > 0):
            raise ValueError(
                f"{cid}: network_damage_onset_stretch {lf} must exceed 1 and "
                f"network_damage_stretch_scale {dlf} must be positive "
                f"(source 32 Eq. 6: 2.105 and 1.0)."
            )
    if child.get("network_rupture") == "criticalStretch":
        lam = float(child.get("network_critical_stretch", 2.8))
        if not lam > 1:
            raise ValueError(
                f"{cid}: network_critical_stretch {lam} is a principal "
                f"stretch and must exceed 1 (source 28: 2.8)."
            )
    if child.get("network_rupture", "off") != "off":
        mode = child.get("network_rupture")
        if not (child.get("two_velocity_skeleton") and child.get("eulerian_solid")
                and child.get("network_pressure_reference") == "deposited"
                and child.get("be_relax_basis") == "networkAbsent"):
            raise ValueError(
                f"{cid}: network_rupture {mode} needs the two-velocity "
                f"Eulerian skeleton, network_pressure_reference deposited and "
                f"be_relax_basis networkAbsent."
            )
        if not tangent:
            raise ValueError(
                f"{cid}: network_rupture {mode} bounds the tangent stiffness "
                f"of the network it acts on so that the dt bound can resolve "
                f"it; it needs elastic_stability_stiffness tangent."
            )


S15_SWITCHES = (
    ("material_label_basis", "materialLabelBasis",
     ("transportedLabel", "contentComposition")),
)


def s15_combination_refusal(child: dict, core_shell: bool) -> None:
    """Refuse, for EVERY child, an S15 material-label combination the solver
    refuses. research/designs/2026-09-24-S15-composition-label-and-progressive-damage.md:
    under contentComposition the core/shell label is thetaH/(thetaC + thetaH),
    derived from the two transported contents, so it needs the content model
    (the contents) and core_shell (a label to derive)."""
    cid = child["child_id"]
    for _key, _dict_key, _allowed in S15_SWITCHES:
        if _key in child and str(child[_key]) not in _allowed:
            raise ValueError(
                f"{cid}: {_key} {str(child[_key])!r} is not one of "
                f"{', '.join(_allowed)}."
            )
    if child.get("material_label_basis") == "contentComposition" and not (
            child.get("content_model") and core_shell):
        raise ValueError(
            f"{cid}: material_label_basis contentComposition derives the "
            f"core/shell label from the core and shell contents; it needs "
            f"content_model true and a core_shell series."
        )


S16_SWITCHES = (
    ("skeleton_floor_basis", "skeletonFloorBasis",
     ("intactWeight", "occupancy")),
    ("damage_transport", "damageTransport",
     ("volumeFlux", "contentPartition")),
    ("network_pressure_damage", "networkPressureDamage",
     ("off", "tensile")),
)

# The damage partition's three fields (solver: thetaIntact_, thetaPending_,
# thetaBroken_). Each needs a linear solver under contentPartition.
S16_PARTITION_FIELDS = ("thetaIntact", "thetaPending", "thetaBroken")


def s16_value_refusal(child: dict) -> None:
    """The VALUE of every S16 key, for EVERY child, before any dependency of
    any stage is checked -- so a misspelt value is reported as a misspelt
    value and never as a missing prerequisite or an inert parameter."""
    cid = child["child_id"]
    _rupture_allowed = dict((k, a) for k, _d, a in S14_SWITCHES)["network_rupture"]
    if "network_rupture" in child and str(child["network_rupture"]) not in _rupture_allowed:
        raise ValueError(
            f"{cid}: network_rupture {str(child['network_rupture'])!r} is not "
            f"one of {', '.join(_rupture_allowed)}."
        )
    for _key, _dict_key, _allowed in S16_SWITCHES:
        if _key in child and str(child[_key]) not in _allowed:
            raise ValueError(
                f"{cid}: {_key} {str(child[_key])!r} is not one of "
                f"{', '.join(_allowed)}."
            )
    for _key in ("network_damage_progressive", "final_relaxation_unity"):
        if _key in child and not isinstance(child[_key], bool):
            raise ValueError(
                f"{cid}: {_key} {child[_key]!r} must be true or false."
            )
    for _key in ("network_release_time_s", "wall_budget_h"):
        if _key not in child:
            continue
        try:
            _v = float(child[_key])
        except (TypeError, ValueError):
            raise ValueError(f"{cid}: {_key} {child[_key]!r} is not a number.")
        if not (_v > 0 and math.isfinite(_v)):
            raise ValueError(
                f"{cid}: {_key} {_v} must be a positive, finite number "
                + ("(the release time tau, seconds)." if _key.startswith("network")
                   else "of hours (the stopAtClockTime writeNow budget).")
            )


def s16_combination_refusal(child: dict, series: dict) -> None:
    """Refuse, for EVERY child, an S16 combination the solver refuses.

    research/designs/2026-09-24-S16-timed-terminal-rupture.md section 4.
    network_rupture terminalRelease: the target damage d* rises to 1 at the
    rupture stretch lc (source 32 p. 355) and d relaxes to it over the release
    time tau (source 63, waiver D4); it needs the damage carried as content,
    the occupancy floor and the tensile pressure damage (F3). The VALUE of
    each switch is checked before any dependency, and a parameter nothing
    reads under the chosen switch is refused as silently inert.
    """
    cid = child["child_id"]
    # 1. Values first -- repeated from s16_value_refusal(), which main()
    # calls before every other stage's refusal, so this function is also
    # safe to call on its own.
    s16_value_refusal(child)
    rupture = str(child.get("network_rupture", "off"))
    # S17 A: fibreRelease is terminalRelease's event and release; every S16
    # requirement applies to it.
    terminal = rupture in ("terminalRelease", "fibreRelease")
    progressive_on = bool(child.get("network_damage_progressive", True))
    # 2. Inert parameters.
    for _key in ("network_release_time_s", "network_damage_progressive"):
        if _key in child and not terminal:
            raise ValueError(
                f"{cid}: {_key} is read only under network_rupture "
                f"terminalRelease or fibreRelease; set without it, it would "
                f"be silently inert."
            )
    if terminal and not progressive_on:
        for _k in ("network_damage_onset_stretch",
                   "network_damage_stretch_scale"):
            if _k in child:
                raise ValueError(
                    f"{cid}: {_k} is read only by the progressive part; "
                    f"with network_damage_progressive false it would be "
                    f"silently inert."
                )
    floor_basis = str(child.get("skeleton_floor_basis", "intactWeight"))
    transport = str(child.get("damage_transport", "volumeFlux"))
    pressure = str(child.get("network_pressure_damage", "off"))
    if rupture == "off" and (floor_basis != "intactWeight"
                             or transport != "volumeFlux"
                             or pressure != "off"):
        raise ValueError(
            f"{cid}: skeleton_floor_basis {floor_basis}, damage_transport "
            f"{transport} and network_pressure_damage {pressure} act on the "
            f"network damage d; network_rupture off has none, so each would "
            f"be silently inert."
        )
    if floor_basis == "occupancy" and not series.get("skeleton_diffuse_domain"):
        raise ValueError(
            f"{cid}: skeleton_floor_basis occupancy moves the diffuse-domain "
            f"floor off the damage; its series has no skeleton_diffuse_domain, "
            f"so there is no floor and the switch would be silently inert."
        )
    if transport == "contentPartition":
        if not child.get("content_model"):
            raise ValueError(
                f"{cid}: damage_transport contentPartition carries the damage "
                f"as a partition of the network content on the content's own "
                f"transport; it needs content_model true."
            )
        if str(child.get("content_transport_scheme", "upwind")) == "limitedLinear":
            raise ValueError(
                f"{cid}: damage_transport contentPartition is non-negative "
                f"only on an M-matrix predictor; content_transport_scheme "
                f"limitedLinear is not one (run 20260923-012704)."
            )
    # 3. terminalRelease: its parameters, then its three companions.
    if terminal:
        lam = float(child.get("network_critical_stretch", 2.8))
        if not lam > 1:
            raise ValueError(
                f"{cid}: network_rupture {rupture}: "
                f"network_critical_stretch {lam} is the rupture stretch lc and "
                f"must exceed 1 (source 28: 2.8)."
            )
        tau = float(child.get("network_release_time_s", 3.3e-4))
        if not (tau > 0 and math.isfinite(tau)):
            raise ValueError(
                f"{cid}: network_rupture {rupture}: "
                f"network_release_time_s {tau} must be positive (3.3e-4 s, "
                f"source 63, waiver D4)."
            )
        if progressive_on:
            lf = float(child.get("network_damage_onset_stretch", 2.105))
            dlf = float(child.get("network_damage_stretch_scale", 1.0))
            if not (lf > 1 and dlf > 0):
                raise ValueError(
                    f"{cid}: network_damage_onset_stretch {lf} must exceed 1 "
                    f"and network_damage_stretch_scale {dlf} must be positive "
                    f"(source 32 Eq. 6: 2.105 and 1.0)."
                )
            if not lam > lf:
                raise ValueError(
                    f"{cid}: network_rupture {rupture} with the "
                    f"progressive part on: network_critical_stretch {lam} must "
                    f"exceed network_damage_onset_stretch {lf}. A rupture "
                    f"stretch at or below the onset is buildable only "
                    f"terminal-only: set network_damage_progressive false."
                )
        missing = [
            f"{k} {v}" for k, v, ok in (
                ("damage_transport", "contentPartition",
                 transport == "contentPartition"),
                ("skeleton_floor_basis", "occupancy",
                 floor_basis == "occupancy"),
                ("network_pressure_damage", "tensile", pressure == "tensile"),
            ) if not ok
        ]
        if missing:
            raise ValueError(
                f"{cid}: network_rupture {rupture} releases a ruptured "
                f"cell to zero stress, which holds only with "
                f"{', '.join(missing)}."
            )


# S17 (2026-09-26): research/designs/2026-09-26-S17-frozen-release-implicit-
# elastic-startup.md, revision 4, APPROVED. A is a network_rupture value
# (fibreRelease); C is elastic_coupling; B is final_relaxation_unity with
# n_correctors 6. (child key, dictionary key, accepted values); the first
# value is the solver's shipped default.
S17_SWITCHES = (
    ("elastic_coupling", "elasticCoupling", ("explicit", "implicitTangent")),
)
# The Vs predictor's linear solver under implicitTangent: the matrix is an
# M-matrix but asymmetric (bounded upwind convection), so a symmetric-only
# solver is refused (design section 4). The first is the default.
S17_VS_SOLVERS = ("smoothSolver", "PBiCGStab")
S17_VS_SOLVERS_REFUSED = ("PCG", "PPCG", "PPCR")
# The pending fibre's carried contents (solver thetaPendingB_, thetaPendingP_),
# each needing a linear solver under fibreRelease.
S17_FIBRE_FIELDS = ("thetaPendingB", "thetaPendingP")
# The named discriminators (design section 6): the ONLY children allowed
# final_relaxation_unity with n_correctors < 6.
S17_DISCRIMINATORS = ("D1", "D2", "D2b", "D3", "D3b", "CB6", "CB4")
# em65_G4_s16_final shipped in batch 57 with final_relaxation_unity and
# nCorrectors 2, before this rule existed; it is the as-run datum the
# discriminators are read against (H8), and it must regenerate byte-identical
# (design section 5). It is the only child the rule does not cover.
S17_FINAL_RELAXATION_GRANDFATHERED = ("em65_G4_s16_final",)
S17_B_CORRECTORS = 6


# The control check (design section 6, change 12): each CONTROL differs from
# the production child it controls ONLY in its named switches, plus the run
# controls it declares (the G 400 controls end at 2.5 ms and carry no wall
# budget). Checked mechanically on BOTH the controls entry (every key) and the
# generated child.json, at generation, so a control that drifts is never
# launched. (control, production child, switches, run controls)
S17_CONTROL_PAIRS = (
    ("em69_G400_s17_noimpl", "em66_G400_s17", {"elastic_coupling"},
     {"end_time_s", "wall_budget_h"}),
    ("em68_G400_s17_law", "em66_G400_s17",
     {"elastic_coupling", "final_relaxation_unity", "n_correctors"},
     {"end_time_s", "wall_budget_h"}),
    ("em70_G4_s17_noimpl", "em67_G4_s17", {"elastic_coupling"}, set()),
)
# The house rule that every NEW child equals its parent except in its new
# keys, for the production children and the discriminators. Checked by
# scripts/check_s17_fibre_release.py, NOT at generation: a legitimate edit of
# a parent (dry_run.py's terminal-only acceptance test edits em63) must not
# make the generator refuse the batch.
S17_PARENT_PAIRS = (
    ("em66_G400_s17", "em63_G400_s16",
     {"network_rupture", "elastic_coupling", "final_relaxation_unity",
      "n_correctors"}, set()),
    ("em67_G4_s17", "em64_G4_s16",
     {"network_rupture", "elastic_coupling", "final_relaxation_unity",
      "n_correctors"}, {"wall_budget_h"}),
    ("em68_G400_s17_law", "em63_G400_s16", {"network_rupture"},
     {"end_time_s", "wall_budget_h"}),
    ("s17_D1_nc3", "em65_G4_s16_final", {"n_correctors", "s17_discriminator"},
     {"end_time_s"}),
    ("s17_D2_nc4", "em65_G4_s16_final", {"n_correctors", "s17_discriminator"},
     {"end_time_s"}),
    ("s17_D2b_nc6", "em65_G4_s16_final", {"n_correctors", "s17_discriminator"},
     {"end_time_s"}),
    ("s17_D3_dt18", "em65_G4_s16_final", {"max_delta_t_s", "s17_discriminator"},
     {"end_time_s"}),
    ("s17_D3b_dt25", "em65_G4_s16_final",
     {"max_delta_t_s", "s17_discriminator"}, {"end_time_s"}),
    ("s17_CB6", "em66_G400_s17", {"s17_discriminator"},
     {"end_time_s", "wall_budget_h"}),
    ("s17_CB4", "s17_CB6", {"n_correctors", "s17_discriminator"}, set()),
)
# child.json keys each controls key surfaces as (flattened "a.b" paths).
# ("s17" whole: the parent of a new child has no s17 block at all.)
S17_SPEC_PATHS = {
    "elastic_coupling": {"s17.elasticCoupling", "s17"},
    "n_correctors": {"s17.nCorrectors", "s17"},
    "final_relaxation_unity": {"final_relaxation_unity"},
    "network_rupture": {"s16.networkRupture", "s16.releaseLaw",
                        "s17.networkRupture", "s17.pendingForm", "s17"},
    "s17_discriminator": {"s17.discriminator", "s17"},
    "end_time_s": {"end_time_s"},
    "wall_budget_h": {"wall_budget_h"},
    "max_delta_t_s": {"s17.maxDeltaT"},
    "fibre_slack": {"s17b.fibreSlack", "s17b"},
    "fibre_arithmetic": {"s17b.fibreArithmetic", "s17b"},
    "skeleton_wall_face": {"s17b.skeletonWallFace", "s17b"},
    "frozen_pending_pressure": {"s18.frozenPendingPressure", "s18"},
    "wall_patch_split": {"s18.wallPatchSplit", "s18"},
    "cplus_order": {"s18.cplusOrder", "s18"},
    "network_pressure_weight": {"s18.networkPressureWeight", "s18"},
}


def _flat_diff(a: dict, b: dict, prefix: str = "") -> set:
    out = set()
    for k in set(a) | set(b):
        path = prefix + k
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out |= _flat_diff(va, vb, path + ".")
        elif va != vb:
            out.add(path)
    return out


def s17_control_check(children: dict, specs: dict, pairs=None) -> None:
    """Refuse a batch-58 (or, with pairs, batch-59) control that differs
    from its reference outside its named switches and declared run controls
    (S17 design section 6; S17b design section 3).

    `children` is the WHOLE controls file; `specs` holds the child.json of
    the children BUILT in this run (a series filter builds a subset). Two
    halves, by construction (fix round 13c, docket 19):
    - the controls-entry diff runs for EVERY pair, filtered or not;
    - the child.json diff guards "a control that drifts is never launched",
      so it applies when the CONTROL is built. A built control whose
      reference is not built is refused: the pair cannot be checked, and
      the filter must name the reference too. A control that is not built
      is not launched; its child.json was checked when its own batch was
      built, and only this half is skipped for it."""
    where = ("S17 design section 6" if pairs is None
             else "S17b design section 3")
    for ctl, ref, switches, run in (S17_CONTROL_PAIRS if pairs is None else pairs):
        if ctl not in children or ref not in children:
            raise ValueError(
                f"S17 control check: {ctl} or {ref} is missing from the "
                f"controls; the pair cannot be checked.")
        allowed = switches | run | {"child_id", "expectation"}
        got = _flat_diff(children[ctl], children[ref])
        extra = sorted(got - allowed)
        if extra:
            raise ValueError(
                f"S17 control check: {ctl} differs from {ref} in "
                f"{', '.join(extra)} -- beyond its named switches "
                f"{', '.join(sorted(switches))} ({where}).")
        missing = sorted(switches - got)
        if missing:
            raise ValueError(
                f"S17 control check: {ctl} does not differ from {ref} in "
                f"its named switch {', '.join(missing)}.")
        if ctl not in specs:
            continue
        if ref not in specs:
            raise ValueError(
                f"S17 control check: the control {ctl} is being built but its "
                f"reference {ref} is not, so {ctl}'s child.json cannot be "
                f"checked against {ref}'s ({where}). Add {ref} to the filter.")
        spec_allowed = {"child_id"}
        for key in switches | run:
            spec_allowed |= S17_SPEC_PATHS[key]
        got_spec = _flat_diff(specs[ctl], specs[ref])
        extra_spec = sorted(p for p in got_spec if p not in spec_allowed)
        if extra_spec:
            raise ValueError(
                f"S17 control check: {ctl}'s child.json differs from {ref}'s "
                f"in {', '.join(extra_spec)} ({where}).")


# S17b (2026-09-27): research/designs/2026-09-27-S17b-slack-fibre-field-
# units-wall-face.md, revision 2, APPROVED. (child key, dictionary key,
# accepted values); the first is S17's behaviour.
S17B_SWITCHES = (
    ("fibre_slack", "fibreSlack", ("off", "v1")),
    ("fibre_arithmetic", "fibreArithmetic", ("shipped", "fieldUnits")),
    ("skeleton_wall_face", "skeletonWallFace", ("shipped", "zeroed")),
)
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
S17B_FIXED_SWEEP_FIELDS = ("thetaPending", "thetaTaut", "thetaPendingB",
                           "thetaPendingP")
# The control check (design section 3): em73 controls C and em74 controls
# A3 against em71; the parent rule as S17's (guard only).
S17B_CONTROL_PAIRS = (
    ("em73_G400_s17b_noimpl", "em71_G400_s17b", {"elastic_coupling"},
     {"end_time_s", "wall_budget_h"}),
    ("em74_G400_s17b_noA3", "em71_G400_s17b", {"skeleton_wall_face"},
     {"end_time_s", "wall_budget_h"}),
)
S17B_PARENT_PAIRS = (
    ("em71_G400_s17b", "em66_G400_s17",
     {"fibre_slack", "fibre_arithmetic", "skeleton_wall_face"}, set()),
    ("em72_G4_s17b", "em67_G4_s17",
     {"fibre_slack", "fibre_arithmetic", "skeleton_wall_face"}, set()),
    ("em73_G400_s17b_noimpl", "em69_G400_s17_noimpl",
     {"fibre_slack", "fibre_arithmetic", "skeleton_wall_face"},
     {"wall_budget_h"}),   # design section 8 e: 6 h stopAtClockTime
)


# S18, the soft stage (2026-09-29): research/designs/2026-09-29-S18-no-
# frozen-pressure-content-weight-network-viscosity.md, rev. 10, APPROVED.
# (child key, dictionary key, accepted values); the first is S17b's
# behaviour. C (R2b) is network_pressure_weight contentRatio (S13's key).
# D (R3), I and H are deferred to OI-44 and have no key here.
S18_SWITCHES = (
    ("frozen_pending_pressure", "frozenPendingPressure", ("shipped", "none")),
    ("wall_patch_split", "wallPatchSplit", ("shipped", "split")),
    ("cplus_order", "cplusOrder", ("shipped", "shared")),
)
# The parent rule (design section 4): em76 differs from em72 only in the
# four S18 switches. Checked on the CONTROLS entries at generation, for
# every build, filtered or not -- em72 need not be built (13c; the launch
# filter of batch 60 is em76 alone).
S18_PARENT_PAIRS = (
    ("em76_G4_s18", "em72_G4_s17b",
     {"frozen_pending_pressure", "wall_patch_split", "cplus_order",
      "network_pressure_weight"}, set()),
)


def s18_value_refusal(child: dict) -> None:
    cid = child["child_id"]
    for _key, _dict_key, _allowed in S18_SWITCHES:
        if _key in child and str(child[_key]) not in _allowed:
            raise ValueError(
                f"{cid}: {_key} {str(child[_key])!r} is not one of "
                f"{', '.join(_allowed)}."
            )


def s18_combination_refusal(child: dict) -> None:
    """Refuse, for EVERY child, an S18 combination the solver refuses
    (design section 2): values first, then each switch's dependency."""
    cid = child["child_id"]
    s18_value_refusal(child)
    named = [k for k, _d, a in S18_SWITCHES
             if str(child.get(k, a[0])) != a[0]]
    if named and child.get("network_rupture") != "fibreRelease":
        raise ValueError(
            f"{cid}: {', '.join(named)} act on the pending fibre (its frozen "
            f"tension, its wall-patch split, its passengers); without "
            f"network_rupture fibreRelease they would be silently inert "
            f"(S18 section 2 A, B, E)."
        )


def s18_parent_check(children: dict) -> None:
    """The S18 parent rule on the CONTROLS entries (never on child.json, so
    it holds under any series filter): each child differs from its parent
    only in its named switches, and in each of them."""
    for ctl, ref, switches, run in S18_PARENT_PAIRS:
        if ctl not in children:
            continue
        if ref not in children:
            raise ValueError(
                f"S18 parent check: {ctl}'s parent {ref} is missing from the "
                f"controls.")
        got = _flat_diff(children[ctl], children[ref])
        extra = sorted(got - (switches | run | {"child_id", "expectation"}))
        if extra:
            raise ValueError(
                f"S18 parent check: {ctl} differs from {ref} in "
                f"{', '.join(extra)} -- beyond its named switches "
                f"{', '.join(sorted(switches))} (S18 design section 4).")
        missing = sorted(switches - got)
        if missing:
            raise ValueError(
                f"S18 parent check: {ctl} does not differ from {ref} in its "
                f"named switch {', '.join(missing)}.")


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
    ("em80_G4_s18p_Afz", "em76_G4_s18",
     {"proj_site_log", "frozen_pending_pressure"}, set()),
    ("em81_G4_s18p_Esh", "em76_G4_s18",
     {"proj_site_log", "cplus_order"}, set()),
)


def s18p_combination_refusal(child: dict) -> None:
    """Refuse, for EVERY child, what the solver refuses (S18p section 2):
    an unknown value, and proj_site_log yes without the pending fibre,
    under which neither counted projection runs (silently inert)."""
    cid = child["child_id"]
    for _key, _dict_key, _allowed in S18P_SWITCHES:
        if _key in child and str(child[_key]) not in _allowed:
            raise ValueError(
                f"{cid}: {_key} {str(child[_key])!r} is not one of "
                f"{', '.join(_allowed)}."
            )
    if (str(child.get("proj_site_log", "no")) == "yes"
            and child.get("network_rupture") != "fibreRelease"):
        raise ValueError(
            f"{cid}: proj_site_log yes counts the P-LS projections of the "
            f"pending fibre; without network_rupture fibreRelease neither "
            f"site runs and it would be silently inert (S18p section 2)."
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



# FSP, the free-shell flux-oscillation probe (2026-10-03): research/designs/
# 2026-10-03-probe-fs-flux-oscillation.md, rev. 1 (Alireza: "Prepare it now",
# docket research/investigations/20261003-0841-run000038-free-shell-live/).
# Each child is fs_G4_kxu_v250 with end_time_s 4e-4 and wall_budget_h 1.0,
# plus at most one key. Checked on the CONTROLS entries, for every build.
FSP_PARENT_PAIRS = (
    ("fsp_ctl", "fs_G4_kxu_v250", set(), {"end_time_s", "wall_budget_h"}),
    ("fsp_k13", "fs_G4_kxu_v250", {"k0p_m2"}, {"end_time_s", "wall_budget_h"}),
    ("fsp_dt15", "fs_G4_kxu_v250", {"max_delta_t_s"},
     {"end_time_s", "wall_budget_h"}),
    ("fsp_G40", "fs_G4_kxu_v250", {"G_pa"}, {"end_time_s", "wall_budget_h"}),
)


def fsp_parent_check(children: dict) -> None:
    """FSP design section 2 on the CONTROLS entries: each child differs from
    fs_G4_kxu_v250 only in its named key and run controls, and in each of
    them. "_"-prefixed keys are commentary and are not compared."""
    for ctl, ref, switches, run in FSP_PARENT_PAIRS:
        if ctl not in children:
            continue
        if ref not in children:
            raise ValueError(
                f"FSP parent check: {ctl}'s parent {ref} is missing from "
                f"the controls.")
        got = {k for k in _flat_diff(children[ctl], children[ref])
               if not k.startswith("_")}
        extra = sorted(got - (switches | run | {"child_id", "expectation"}))
        if extra:
            raise ValueError(
                f"FSP parent check: {ctl} differs from {ref} in "
                f"{', '.join(extra)} -- beyond its named keys "
                f"{', '.join(sorted(switches | run))} (FSP design section 2).")
        missing = sorted((switches | run) - got)
        if missing:
            raise ValueError(
                f"FSP parent check: {ctl} does not differ from {ref} in its "
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
    # S19b (research/designs/2026-10-04-S19b-lumen-elimination.md rev. 2,
    # APPROVED by Alireza 2026-10-04, section 2.6): the lumen treatment under
    # blockCoupled. Absent = eliminate (the solver's default); identityRows
    # is S19a's system, the control. Refused outside blockCoupled.
    ("block_coupled_lumen", "blockCoupledLumen", ("eliminate", "identityRows")),
)
S19_KEYS = ("momentum_pressure_coupling", "network_viscosity", "tau_v_s",
            "block_coupled_lumen")
# Section 6: batch 65. Each child is its parent plus its named keys only;
# fsb_off is fsp_k13 with the key ABSENT (V1, bit-identical). Checked on the
# CONTROLS entries, for every build. "_"-prefixed keys are commentary.
S19_PARENT_PAIRS = (
    ("fsb_off", "fsp_k13", set(), set()),
    ("fsb_k13", "fsp_k13", {"momentum_pressure_coupling"}, set()),
    ("fsb_G40", "fsp_G40", {"momentum_pressure_coupling"}, set()),
    ("fsb_ctl", "fsp_ctl", {"momentum_pressure_coupling"}, set()),
    ("fsb_ctl_r3", "fsb_ctl", {"network_viscosity"}, set()),
    # S19b section 4 (batch 67): the control is fsb_k13 with identityRows.
    ("fsb_k13_id", "fsb_k13", {"block_coupled_lumen"}, set()),
)


def s19_combination_refusal(child: dict) -> None:
    """Refuse, for EVERY child, what the solver refuses (S19 sections 2 and
    12): values first, then each switch's requirements, then keys that would
    be silently inert."""
    cid = child["child_id"]
    for _key, _dict_key, _allowed in S19_SWITCHES:
        if _key in child and str(child[_key]) not in _allowed:
            raise ValueError(
                f"{cid}: {_key} {str(child[_key])!r} is not one of "
                f"{', '.join(_allowed)}."
            )
    if "tau_v_s" in child:
        _t = child["tau_v_s"]
        if isinstance(_t, bool) or not isinstance(_t, (int, float)) or not _t > 0:
            raise ValueError(
                f"{cid}: tau_v_s {_t!r} must be a positive number of seconds "
                f"(eta_n = G tau_v/phi0; S19 section 12)."
            )
    named = [k for k in S19_KEYS if k in child]
    if named and not child.get("two_velocity_skeleton"):
        raise ValueError(
            f"{cid}: {', '.join(named)} act on the two-velocity block; "
            f"without two_velocity_skeleton the key would be silently "
            f"dropped (S19 section 2)."
        )
    block = str(child.get("momentum_pressure_coupling", "segregated")) == "blockCoupled"
    r3 = str(child.get("network_viscosity", "none")) == "kelvinVoigt"
    # R3 first, as the solver orders them, so each refusal is reachable.
    if r3 and not block:
        raise ValueError(
            f"{cid}: network_viscosity kelvinVoigt needs "
            f"momentum_pressure_coupling blockCoupled: R3 is unstable at the "
            f"design dt in every segregated placement (OI-44; S19 section 12)."
        )
    if r3 and child.get("network_pressure_weight") == "contentRatio":
        raise ValueError(
            f"{cid}: network_viscosity kelvinVoigt with "
            f"network_pressure_weight contentRatio is refused: C stays off "
            f"(S19 section 12)."
        )
    if block:
        if not child.get("eulerian_solid") or \
                str(child.get("released_carrier", "rigidBody")) != "material":
            raise ValueError(
                f"{cid}: momentum_pressure_coupling blockCoupled needs "
                f"eulerian_solid true and released_carrier material (S19 "
                f"section 2)."
            )
        if child.get("network_pressure_weight") == "contentRatio":
            raise ValueError(
                f"{cid}: momentum_pressure_coupling blockCoupled with "
                f"network_pressure_weight contentRatio is refused (S19 "
                f"section 2; C off)."
            )
        if str(child.get("elastic_coupling", "explicit")) != "implicitTangent":
            raise ValueError(
                f"{cid}: momentum_pressure_coupling blockCoupled needs "
                f"elastic_coupling implicitTangent (S19 section 2)."
            )
        if child.get("face_consistent_elastic_force"):
            raise ValueError(
                f"{cid}: momentum_pressure_coupling blockCoupled with "
                f"face_consistent_elastic_force true is refused (S19 section 2)."
            )
        # block_face_inverse faceAssembled needs no refusal here: implicitTangent
        # (above) needs fibreRelease, which needs be_relax_basis networkAbsent,
        # which needs faceAssembled -- refused upstream, so an S19 refusal
        # would be unreachable (dry run, solver-engineer 09). The solver
        # refuses it at construction regardless.
    if "tau_v_s" in child and not r3:
        raise ValueError(
            f"{cid}: tau_v_s is read only under network_viscosity "
            f"kelvinVoigt; without it the key would be silently inert (S19 "
            f"section 12)."
        )
    if "block_coupled_lumen" in child and not block:
        raise ValueError(
            f"{cid}: block_coupled_lumen is read only under "
            f"momentum_pressure_coupling blockCoupled; without it the key "
            f"would be silently inert (S19b section 2.6)."
        )


def s19_parent_check(children: dict) -> None:
    """S19 section 6 on the CONTROLS entries: each child differs from its
    parent only in its named keys, and in each of them."""
    for ctl, ref, switches, run in S19_PARENT_PAIRS:
        if ctl not in children:
            continue
        if ref not in children:
            raise ValueError(
                f"S19 parent check: {ctl}'s parent {ref} is missing from "
                f"the controls.")
        got = {k for k in _flat_diff(children[ctl], children[ref])
               if not k.startswith("_")}
        extra = sorted(got - (switches | run | {"child_id", "expectation"}))
        if extra:
            raise ValueError(
                f"S19 parent check: {ctl} differs from {ref} in "
                f"{', '.join(extra)} -- beyond its named keys "
                f"{', '.join(sorted(switches | run)) or '(none)'} (S19 design "
                f"section 6).")
        missing = sorted((switches | run) - got)
        if missing:
            raise ValueError(
                f"S19 parent check: {ctl} does not differ from {ref} in its "
                f"named key {', '.join(missing)}.")


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
    if not series.get("advect_released"):
        raise ValueError(
            f"{cid}: release_core_exempt is read by updateReleasedIndicator(), "
            f"which returns before reading it when advect_released is off "
            f"(.C L11142-11145); set without it, it would be silently inert.")
    if not child.get("eulerian_solid"):
        raise ValueError(
            f"{cid}: release_core_exempt is written with the Eulerian anchor "
            f"keys; a partitioned child would not carry it.")


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
    if not series.get("advect_released"):
        raise ValueError(
            f"{cid}: advect_released must be true, or PHASE_CLOT_RELEASED is "
            f"never printed (.C L11142-11145) and A4/P8 would read an absent "
            f"line as zero (red team 05 #3).")
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
    if not (child.get("bond_hold_carrier") is True
            and child.get("bond_hold_dilation") == 4
            and child.get("bond_hold_taper") is False):
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
    ("fsf_G40_klo_v250", "fs_G40_klo_v250",
     {"momentum_pressure_coupling", "network_viscosity", "tau_v_s",
      "clot_override"}, FSF_RUN_KEYS),
    ("fsf_G4_kxu_v250", "fs_G4_kxu_v250",
     {"momentum_pressure_coupling", "network_viscosity", "tau_v_s",
      "clot_override"}, FSF_RUN_KEYS),
    # The pair takes the main children's output plan too (CE ruling on
    # report 03): frames every 0.2 ms, one full write at endTime 2 ms.
    ("fsb_G40_klo_rest", "fs_G40_klo_v250", {"momentum_pressure_coupling"},
     FSF_RUN_KEYS),
    ("fs_G40_klo_rest_seg", "fs_G40_klo_v250", set(), FSF_RUN_KEYS),
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
    if not series.get("advect_released"):
        raise ValueError(
            f"{cid}: advect_released must be true, or a released clot is "
            f"never advected and 'free from step 1' has no mechanism."
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
    # R0 (S21 V21-7 reference): xu5_id is xu5_g250's identity twin and differs
    # from its siblings only in end_time_s / wall_budget_h, as in the clean
    # generator. Refusal logic only; no generated file changes.
    for cid in [c for c in xu if c != "xu5_id"][1:]:
        extra = sorted(_flat_diff(children[cid], children[xu[0]])
                       - XU_SIBLINGS_DIFFER_IN)
        if extra:
            raise ValueError(
                f"Xu parent check: {cid} differs from {xu[0]} in "
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


def s17b_value_refusal(child: dict) -> None:
    cid = child["child_id"]
    for _key, _dict_key, _allowed in S17B_SWITCHES:
        if _key in child and str(child[_key]) not in _allowed:
            raise ValueError(
                f"{cid}: {_key} {str(child[_key])!r} is not one of "
                f"{', '.join(_allowed)}."
            )


def s17b_combination_refusal(child: dict) -> None:
    """Refuse, for EVERY child, an S17b combination the solver refuses
    (design section 2): values first, then each switch's dependency."""
    cid = child["child_id"]
    s17b_value_refusal(child)
    if (str(child.get("fibre_slack", "off")) == "v1"
            and child.get("network_rupture") != "fibreRelease"):
        raise ValueError(
            f"{cid}: fibre_slack v1 acts on the pending share's fibre; it "
            f"needs network_rupture fibreRelease (design A1)."
        )
    if (str(child.get("fibre_arithmetic", "shipped")) == "fieldUnits"
            and child.get("damage_transport") != "contentPartition"):
        raise ValueError(
            f"{cid}: fibre_arithmetic fieldUnits assembles the partition and "
            f"fibre predictors in field units; without damage_transport "
            f"contentPartition there are none, so it would be silently inert "
            f"(design A2)."
        )
    if (str(child.get("skeleton_wall_face", "shipped")) == "zeroed"
            and not child.get("two_velocity_skeleton")):
        raise ValueError(
            f"{cid}: skeleton_wall_face zeroed acts on the two-velocity "
            f"block's Vs and U reconstructs; without two_velocity_skeleton it "
            f"would be silently inert (design A3)."
        )


def s17_value_refusal(child: dict) -> None:
    """The VALUE of every S17 key, for EVERY child, before any dependency."""
    cid = child["child_id"]
    for _key, _dict_key, _allowed in S17_SWITCHES:
        if _key in child and str(child[_key]) not in _allowed:
            raise ValueError(
                f"{cid}: {_key} {str(child[_key])!r} is not one of "
                f"{', '.join(_allowed)}."
            )
    if "vs_linear_solver" in child:
        _v = str(child["vs_linear_solver"])
        if _v in S17_VS_SOLVERS_REFUSED:
            raise ValueError(
                f"{cid}: vs_linear_solver {_v} is symmetric-only; the Vs "
                f"matrix under elastic_coupling implicitTangent is an M-matrix "
                f"but asymmetric (bounded upwind convection): use "
                f"{' or '.join(S17_VS_SOLVERS)}."
            )
        if _v not in S17_VS_SOLVERS:
            raise ValueError(
                f"{cid}: vs_linear_solver {_v!r} is not one of "
                f"{', '.join(S17_VS_SOLVERS)}."
            )
    if "n_correctors" in child:
        _n = child["n_correctors"]
        if isinstance(_n, bool) or not isinstance(_n, int) or _n < 1:
            raise ValueError(
                f"{cid}: n_correctors {_n!r} must be a positive integer (the "
                f"PISO corrector count)."
            )
    if "s17_discriminator" in child and str(child["s17_discriminator"]) \
            not in S17_DISCRIMINATORS:
        raise ValueError(
            f"{cid}: s17_discriminator {child['s17_discriminator']!r} is not "
            f"one of {', '.join(S17_DISCRIMINATORS)}."
        )


def s17_combination_refusal(child: dict, series: dict, coupling: dict) -> None:
    """Refuse, for EVERY child, an S17 combination the solver refuses
    (design section 4), values first, then dependencies, then inert keys."""
    cid = child["child_id"]
    s17_value_refusal(child)
    rupture = str(child.get("network_rupture", "off"))
    fibre = rupture == "fibreRelease"
    coupling_mode = str(child.get("elastic_coupling", "explicit"))
    implicit = coupling_mode == "implicitTangent"
    if fibre and not child.get("be_det_renormalise"):
        raise ValueError(
            f"{cid}: network_rupture fibreRelease needs be_det_renormalise "
            f"true: guarantee (iii) is stated for the renormalised "
            f"(isochoric) model, and the solver calls it NOT A DEFAULT "
            f"(design section 4)."
        )
    if implicit and not fibre:
        raise ValueError(
            f"{cid}: elastic_coupling implicitTangent needs network_rupture "
            f"fibreRelease: Gamma_t >= 0 is the linearisation of the fibre "
            f"form's non-negative loading tangent (design section 4)."
        )
    if implicit and str(child.get("stress_face_transmission", "linear")) != "series":
        raise ValueError(
            f"{cid}: elastic_coupling implicitTangent linearises the SERIES "
            f"face stress (Gamma_t,f = series(w)_f interp(gD G k_eff)_f "
            f"dt/rho); stress_face_transmission series is required."
        )
    if "vs_linear_solver" in child and not implicit:
        raise ValueError(
            f"{cid}: vs_linear_solver is read only under elastic_coupling "
            f"implicitTangent (the Vs predictor); set without it, it would be "
            f"silently inert."
        )
    n_corr = int(child.get("n_correctors", coupling["nCorrectors"]))
    disc = child.get("s17_discriminator")
    if (child.get("final_relaxation_unity") and n_corr < S17_B_CORRECTORS
            and disc is None
            and cid not in S17_FINAL_RELAXATION_GRANDFATHERED):
        raise ValueError(
            f"{cid}: final_relaxation_unity with nCorrectors {n_corr}: the "
            f"70-row throat's spectral radius is 1.21-1.27 at 4 and 0.97-1.01 "
            f"at 5 (docket 10); B is nCorrectors {S17_B_CORRECTORS}. Fewer is "
            f"allowed only in a named discriminator (s17_discriminator)."
        )
    if disc is not None and not child.get("final_relaxation_unity"):
        raise ValueError(
            f"{cid}: s17_discriminator {disc} reads the Final-1 start-up; "
            f"without final_relaxation_unity it would be silently inert."
        )


def s10_combination_refusal(child: dict) -> None:
    """Refuse, for EVERY child, an S10 key the build could not honour.

    Called from the loop over all children, NOT from the two-velocity block
    that writes the keys. A refusal placed inside that block could never
    fire for a child that sets a key WITHOUT two_velocity_skeleton -- the key
    would simply not be written, and the child would run the shipped
    behaviour under a name that claims the fix (the guard-inside-the-block
    defect of 2026-09-04).
    """
    present = [k for k, _, _ in S10_SWITCHES if k in child]
    if present and not child.get("two_velocity_skeleton"):
        raise ValueError(
            f"{child['child_id']}: {', '.join(present)} "
            f"{'is' if len(present) == 1 else 'are'} only written on the "
            f"two-velocity path; without two_velocity_skeleton the key would "
            f"be silently dropped and the child would run the shipped "
            f"behaviour."
        )
    if child.get("network_pressure_reference") == "deposited" and not (
        child.get("content_model")
        and child.get("phase_force_basis") == "content"
    ):
        raise ValueError(
            f"{child['child_id']}: network_pressure_reference deposited "
            f"measures compaction as theta - thetaNetRef, so it needs "
            f"content_model true and phase_force_basis 'content'. The solver "
            f"refuses the same combination at construction."
        )
    if (child.get("content_transport_scheme", "upwind") != "upwind"
            and not child.get("content_model")):
        raise ValueError(
            f"{child['child_id']}: content_transport_scheme "
            f"{child['content_transport_scheme']} is the operator of the "
            f"content equation and needs content_model true."
        )


def validate_phase_weight_basis(child: dict) -> None:
    """The C1/C2 weight basis, validated on EVERY child.

    Deliberately NOT part of validate_stage2_platelet(), which returns early
    when platelet_content_model is false. This key changes the momentum and
    pressure weights, so a misspelling -- or a 'content' basis on a child with
    no content model -- has to fail here, on the machine that can fix it,
    rather than after a Colab boot. The solver validates the same two words
    for the same reason; this is the cheap copy of that refusal.
    """
    cid = child.get("child_id")
    localisation = str(child.get("platelet_bind_localisation", "uniform"))

    basis = str(child.get("phase_weight_basis", "legacy"))
    if basis not in ("legacy", "content"):
        raise SystemExit(
            "child %s: phase_weight_basis must be 'legacy' or 'content'; "
            "found %r" % (cid, basis)
        )

    # ---- THE FORCE BASIS ----
    #
    # Separate key from the weight basis, so each migration stays a one-key
    # diff. It moves networkPressure()'s argument and the ISOTROPIC elastic
    # term only; the deviatoric weight stays max(chi, floor) and must never
    # follow it -- chi reaches 1.0022 where theta reaches 0.450977, so
    # substituting there rescales the deviatoric stress by theta_ref and drops
    # the floor. That is what run 20260905-191135 died of.
    force = str(child.get("phase_force_basis", "legacy"))
    if force not in ("legacy", "content"):
        raise SystemExit(
            "child %s: phase_force_basis must be 'legacy' or 'content'; "
            "found %r" % (cid, force)
        )
    if force == "content":
        if not child.get("platelet_content_model", False):
            raise SystemExit(
                "child %s: phase_force_basis 'content' requires "
                "platelet_content_model true." % (cid,)
            )
        if basis != "content":
            raise SystemExit(
                "child %s: phase_force_basis 'content' requires "
                "phase_weight_basis 'content'. Content forces against legacy "
                "weights disagree about the solid fraction inside a single "
                "momentum equation, which is worse than either basis used "
                "consistently." % (cid,)
            )

    # ---- THE INDICATOR BASIS ----
    #
    # A THIRD key, and it is not a synonym for either of the two above. The
    # indicator is theta/theta_ref, not theta: the eleven inline chi sites are
    # consumers of a marker in [0, ~1], not of a volume fraction, so writing
    # theta at them would rescale the deviatoric elastic stress, the damage
    # driver, the Kres ramp, the isSkel threshold and the bond hold by 0.45.
    #
    # Checked BEFORE the `basis != "content"` early return below, because the
    # indicator has nothing to do with the weight basis and a child may move
    # one without the other. Putting it after that return is exactly the trap
    # this function was extracted from validate_stage2_platelet() to avoid.
    indicator = str(child.get("phase_indicator_basis", "legacy"))
    if indicator not in ("legacy", "content"):
        raise SystemExit(
            "child %s: phase_indicator_basis must be 'legacy' or 'content'; "
            "found %r" % (cid, indicator)
        )
    if indicator == "content":
        if not child.get("platelet_content_model", False):
            raise SystemExit(
                "child %s: phase_indicator_basis 'content' requires "
                "platelet_content_model true." % (cid,)
            )
        if float(child.get("skeleton_volume_fraction", 0.0)) <= 0:
            raise SystemExit(
                "child %s: phase_indicator_basis 'content' divides by "
                "skeleton_volume_fraction, which is %r."
                % (cid, child.get("skeleton_volume_fraction"))
            )
        # A DIFFERENT MECHANISM FROM THE WEIGHT REFUSAL, not a copy of it.
        # Kres and isSkel both switch off above residual_chi; unlocalised
        # deposition grows lumen content linearly toward it. Measured
        # 4.18e-05 (indicator 9.29e-05) at 0.4 ms against a threshold of
        # 1e-02 -- 108x of margin at the measurement, but the main case runs
        # 75x longer and linear extrapolation lands within a factor of 1.4.
        if (localisation != "interface"
                and not child.get("platelet_transport_only", False)
                and float(child.get("platelet_bind_rate", 0.0)) > 0
                and not child.get("unlocalised_phase_weights_accepted",
                                  False)):
            raise SystemExit(
                "child %s: phase_indicator_basis 'content' with an "
                "unlocalised accepted transfer grows lumen content toward "
                "the residual_chi threshold, above which Kres and the isSkel "
                "clamp both switch off in the lumen. Set "
                "platelet_bind_localisation 'interface', or set "
                "unlocalised_phase_weights_accepted true with a recorded "
                "reason." % (cid,)
            )

    if basis != "content":
        return

    if not child.get("platelet_content_model", False):
        raise SystemExit(
            "child %s: phase_weight_basis 'content' requires "
            "platelet_content_model true -- without it nothing solves "
            "thetaC/thetaH and the momentum weights would be frozen at "
            "their initialisation." % (cid,)
        )

    # THE ORDERING CONSTRAINT, MEASURED. A uniform bind rate deposits in every
    # cell, so theta is nonzero throughout the lumen: 4.18e-05 in all 19119
    # lumen cells of run 20260915-073208, where solidFraction() is exactly 0.
    # Using that as a momentum weight puts a force where the skeleton diagonal
    # is only Kres ~ 67 1/s against 3.3e+05 inside the clot. Run
    # 20260905-191135 died of a lumen weight of 1e-04. With the interface
    # localiser the same cells carry 2.66e-13.
    if (localisation != "interface"
            and not child.get("platelet_transport_only", False)
            and float(child.get("platelet_bind_rate", 0.0)) > 0
            and not child.get("unlocalised_phase_weights_accepted", False)):
        raise SystemExit(
            "child %s: phase_weight_basis 'content' with an unlocalised "
            "accepted transfer puts solid in every lumen cell and then uses "
            "it as a momentum weight. Set platelet_bind_localisation "
            "'interface', or set unlocalised_phase_weights_accepted true "
            "with a recorded reason." % (cid,)
        )


def validate_stage2_platelet(child: dict, content_operator: str) -> None:
    """Every Stage 2 misconfiguration that would RUN rather than fail.

    Each of these is refused at generation, on the machine that can fix it,
    rather than in Colab. The expensive ones are the last two: they do not
    crash, they produce a clean-looking null.
    """
    cid = child.get("child_id")
    if not child.get("platelet_content_model", False):
        if child.get("platelet_flux_convention") is not None:
            raise SystemExit(
                "child %s: platelet_flux_convention is set but "
                "platelet_content_model is not -- no platelet equation "
                "would be solved" % (cid,)
            )
        return

    if not child.get("content_model", False):
        raise SystemExit(
            "child %s: platelet_content_model requires content_model true -- "
            "the platelet porosity is eps = 1 - (thetaC + thetaH) and those "
            "fields are not solved otherwise" % (cid,)
        )

    if content_operator == "matched" and not child.get(
            "platelet_transport_only", False):
        raise SystemExit(
            "child %s: contentOperator `matched` carries the legacy growth "
            "source and platelet_content_model adds the accepted transfer. "
            "Both would deposit solid and the inventory would STILL balance, "
            "because each is internally paired. Use content_operator "
            "`content`, or set platelet_transport_only true." % (cid,)
        )

    convention = str(child.get("platelet_flux_convention", "undeclared"))
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
    localisation = str(child.get("platelet_bind_localisation", "uniform"))
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
    if (localisation == "uniform" and end_time > 0
            and not child.get("unlocalised_deposition_accepted", False)):
        uniform_gain = (
            float(child.get("platelet_yield_m3", 1e-17))
            * float(child.get("platelet_bind_rate", 0.0))
            * end_time
            * float(child["platelet_number_density_m3"])
        )
        if uniform_gain > UNLOCALISED_DEPOSITION_CEILING:
            raise SystemExit(
                "child %s: the accepted transfer has no spatial localiser, "
                "and this configuration would deposit theta ~ %.3e uniformly "
                "over the run -- above the %.0e ceiling, and comparable to "
                "the 7.41e-03 C1 discrepancy it is supposed to measure. The "
                "deposition is FASTEST IN THE LUMEN, so this would not be "
                "clot growth. Repair the localiser (interfaceDelta_ and kEff "
                "already exist; see the coupling spec section 6a), lower "
                "lambda, or set unlocalised_deposition_accepted true with a "
                "recorded reason." % (cid, uniform_gain,
                                      UNLOCALISED_DEPOSITION_CEILING)
            )


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

    # The three axes must actually be swept, or the "competition" has no data.
    req = [c for c in controls["children"] if c["series"] == "required_strength"]
    for axis, key in (("flow", "wall_shear_rate_1_s"), ("stiffness", "G_pa"),
                      ("porosity", "k0p_m2")):
        if len({c[key] for c in req}) < 2:
            raise ValueError(
                f"the required-strength series does not vary {axis} ({key}); "
                f"the PI's competition needs all three"
            )

    # Wall condition and strength must agree, and -- since run 20260810-212211
    # -- any child that must MEASURE a traction has to carry the adhesion
    # condition, because a fixedValue patch emits no PHASE_CLOT_DEBOND at all.
    unbreakable = controls["series"]["required_strength"]["unbreakable_strength_pa"]
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
        # S16 values first: a misspelt S16 value must not surface as some
        # other stage's missing prerequisite.
        s16_value_refusal(child)
        s17_value_refusal(child)
        s17b_value_refusal(child)
        s18_value_refusal(child)
        s10_combination_refusal(child)
        s11_combination_refusal(child)
        # The RESOLVED permeability model, child -> series -> axis, exactly
        # as the permeabilityModel entry below is written: throat_embolus
        # itself declares `constant` and the two-velocity children override it.
        s13_combination_refusal(child)
        s14_combination_refusal(child)
        s15_combination_refusal(child, bool(spec.get("core_shell")))
        s16_combination_refusal(child, spec)
        s17_combination_refusal(child, spec, coupling)
        s17b_combination_refusal(child)
        s18_combination_refusal(child)
        s18p_combination_refusal(child)
        s19_combination_refusal(child)
        s12_combination_refusal(
            child,
            child.get(
                "permeability_model",
                spec.get("permeability_model", controls["porosity_axis"]["model"]),
            ),
        )

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
        if child["series"] in SOURCE33_SERIES:
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
                bool(child.get("bond_hold_carrier", False))
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
            if not spec.get("advect_released"):
                raise ValueError(
                    f"{child['child_id']}: advect_released must be ON, or a "
                    f"piece that tears off has no advective term and sits in "
                    f"place instead of embolising"
                )
            if not spec["growth"]:
                raise ValueError(
                    f"{child['child_id']}: growth must be ON in this series. "
                    f"It exists because Dr Li asked to see deformation and "
                    f"detachment AS THE CLOT GROWS."
                )
            if spec.get("permeability_model") != "constant":
                raise ValueError(
                    f"{child['child_id']}: this series must use the `constant` "
                    f"permeability model. Under kozenyCarman the clot's "
                    f"realised permeability is the numerical floor "
                    f"(2.22e-16 m2, measured in every bulk-clot cell of run "
                    f"20260814-055141) and k0p has no effect at all, so the "
                    f"core/shell permeability contrast this child rests on "
                    f"would not exist."
                )
            if not spec.get("phase_field_material_velocity"):
                raise ValueError(
                    f"{child['child_id']}: phase_field_material_velocity must "
                    f"be ON. Without it the phase field is advected by the "
                    f"FLUID flux, which the Brinkman penalty drives to zero "
                    f"inside the clot, so the clot cannot move however far its "
                    f"skeleton deforms -- and the only way material could "
                    f"leave would be the damage variable erasing it, which is "
                    f"the manufactured tear this series exists to stop doing."
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

        # The series-specific diagnosis comes FIRST. The generic
        # welded-but-carries-a-strength check below is also true when a
        # required-strength child is mis-wired, and it would report the
        # symptom ("a strength is set") instead of the cause ("this child has
        # to measure a traction and a fixedValue patch reports none").
        # THE COHESIVE-TEAR SERIES inverts the demonstration series: the wall
        # must NOT fail, so that the only way out is through the body. Every
        # check below is a way that inversion can be silently undone.
        if child["series"] == "cohesive_tear":
            if welded:
                raise ValueError(
                    f"{child['child_id']}: a cohesive-tear child must use "
                    f"clotWallAdhesion. A fixedValue patch emits no "
                    f"PHASE_CLOT_DEBOND, so the run could not PROVE the wall "
                    f"held -- and 'the wall held' is the control on the whole "
                    f"experiment."
                )
            if child["adhesive_strength_pa"] != spec["unbreakable_strength_pa"]:
                raise ValueError(
                    f"{child['child_id']}: must be unbreakable at "
                    f"{spec['unbreakable_strength_pa']} Pa. Any breakable bond "
                    f"reproduces the whole-body sliding Dr Li rejected."
                )
            if child["adhesive_strength_pa"] <= cohesive_bound:
                raise ValueError(
                    f"{child['child_id']}: 'unbreakable' must exceed the "
                    f"cohesive bound {cohesive_bound} Pa"
                )
            if spec["growth"]:
                raise ValueError(
                    f"{child['child_id']}: growth must be OFF in this series. "
                    f"Growth moves the surface while the tear is being "
                    f"located, which makes 'where did it break' unanswerable."
                )
            if not spec.get("advect_released"):
                raise ValueError(
                    f"{child['child_id']}: advect_released must be ON, or the "
                    f"piece that tears off has no advective term and sits in "
                    f"place instead of embolising"
                )
            # The deliberately non-physiological toughness must be DECLARED on
            # the child and must be on the declared ladder. A child that
            # silently inherits the physiological Gc would never tear, and the
            # series would report "no cohesive failure" as though that were
            # the measurement rather than a mis-wiring.
            ladder = controls["cohesive_tear_axis"]["Gc_j_m2"]
            if child.get("Gc_j_m2") is None:
                raise ValueError(
                    f"{child['child_id']}: must declare its own Gc_j_m2. "
                    f"Inheriting the physiological "
                    f"{fixed['Gc_j_m2']} J/m2 gives a margin of ~5900 and the "
                    f"child would silently report no tear."
                )
            if child["Gc_j_m2"] not in ladder:
                raise ValueError(
                    f"{child['child_id']}: Gc {child['Gc_j_m2']} is not on the "
                    f"declared ladder {ladder}"
                )
            if child["Gc_j_m2"] >= fixed["Gc_j_m2"]:
                raise ValueError(
                    f"{child['child_id']}: Gc must be BELOW the physiological "
                    f"{fixed['Gc_j_m2']} J/m2; this series exists to measure "
                    f"how far below"
                )
            continue

        if child["series"] == "required_strength":
            if welded:
                raise ValueError(
                    f"{child['child_id']}: a required-strength child must use "
                    f"clotWallAdhesion. A fixedValue patch reports no traction, "
                    f"which returned null for all nine of them in run "
                    f"20260810-212211."
                )
            if child["adhesive_strength_pa"] != unbreakable:
                raise ValueError(
                    f"{child['child_id']}: must be unbreakable at "
                    f"{unbreakable} Pa. A breakable bond caps its own reported "
                    f"peak at its strength, because peakTraction is measured "
                    f"over still-bonded faces only."
                )
            if child["adhesive_strength_pa"] <= cohesive_bound:
                raise ValueError(
                    f"{child['child_id']}: 'unbreakable' must exceed the "
                    f"cohesive bound {cohesive_bound} Pa, or the bond could "
                    f"still fail during the measurement"
                )
            continue

        # A DEFORMATION child measures how far the clot bends while it is
        # still attached. Two ways that measurement can be silently ruined,
        # both refused here rather than discovered in the output.
        if child["series"] == "deformation":
            if welded:
                raise ValueError(
                    f"{child['child_id']}: a deformation child must use "
                    f"clotWallAdhesion, so the traction driving the "
                    f"deformation is reported alongside it"
                )
            if child["adhesive_strength_pa"] != spec["unbreakable_strength_pa"]:
                raise ValueError(
                    f"{child['child_id']}: must be unbreakable at "
                    f"{spec['unbreakable_strength_pa']} Pa. At arterial shear "
                    f"the traction is 13-106 Pa; any breakable bond debonds in "
                    f"the first steps and the clot detaches before it deforms, "
                    f"so the run would measure the wrong thing entirely."
                )
            arterial = controls["deformation_axis"]["wall_shear_rate_1_s"]
            if child["wall_shear_rate_1_s"] not in arterial:
                raise ValueError(
                    f"{child['child_id']}: shear rate "
                    f"{child['wall_shear_rate_1_s']} 1/s is not on the "
                    f"deformation axis {arterial}. This series exists to leave "
                    f"the venous range; a venous rate here would reproduce the "
                    f"0.53% strain it was built to escape."
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
            elif series.get("clot_in_throat"):
                # THE LODGED-EMBOLUS PLACEMENT, and a DIFFERENT guard rather
                # than a weakened one. Run 20260903-114913 measured why this
                # series exists: a clot that travels to the throat arrives as
                # a 2.0-2.6 um film, and sqrt(k) = 3.16 um at Xu's
                # permeability, so the deposit is thinner than the Brinkman
                # screening length and is hydraulically close to invisible.
                # Placing the fragment in the throat separates the HYDRAULIC
                # question (what does a lodged clot do to the flow) from the
                # TRANSPORT question (can one get there), which are coupled
                # through k in opposite directions and cannot both be
                # satisfied at one permeability.
                #
                # It must sit ENTIRELY inside the prismatic throat. A clot
                # overlapping the taper would be cut by the sloping wall and
                # its fill fraction would not be the number the prediction was
                # made for -- the same reason the journey children must be
                # clear of the narrowing.
                if clot_x_min < x2:
                    raise ValueError(
                        f"{child_id}: a lodged clot starts at x = "
                        f"{clot_x_min:.3g} m but the throat only begins at "
                        f"{x2:.3g} m; it would be cut by the taper wall"
                    )
                if clot_x_max > g["channel_length_m"]:
                    raise ValueError(
                        f"{child_id}: a lodged clot reaches x = "
                        f"{clot_x_max:.3g} m, past the outlet at "
                        f"{g['channel_length_m']:.3g} m"
                    )
                # THE LUMEN MUST BE RESOLVED. The clot top and the far wall
                # each carry a tanh interface of width epsilonC. If the gap
                # they leave is not several interface widths the two profiles
                # overlap, the "open" lumen never reaches c = +1, and the
                # measured fill fraction is an artefact of epsilonC rather
                # than of the clot. Three widths is the floor this project
                # holds elsewhere (the shell-annulus check above).
                gap = hth - clot["height_m"]
                widths = gap / g["interface_width_m"]
                if widths < 3.0:
                    raise ValueError(
                        f"{child_id}: a {clot['height_m']:.3g} m clot in a "
                        f"{hth:.3g} m throat leaves a {gap:.3g} m lumen = "
                        f"{widths:.2f} interface widths, below the 3.0 floor. "
                        f"At epsilonC = {g['interface_width_m']:.3g} m the "
                        f"tallest resolvable clot here is "
                        f"{hth - 3.0 * g['interface_width_m']:.3g} m."
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
            "eulerian_solid": bool(child.get("eulerian_solid", False)),
            # Whether the clot was carried by its own skeleton velocity or by
            # the fluid. It changes what the phase field MEANS, so it is
            # recorded next to the result rather than left in the case files.
            "two_velocity_skeleton": bool(
                child.get("two_velocity_skeleton", False)
            ),
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
        eulerian_solid = bool(child.get("eulerian_solid", False))

        # READ AND CHECKED HERE, OUTSIDE the `if eulerian_solid:` block below.
        # This guard first went inside it, where it was unreachable in exactly
        # the case it exists to reject: with eulerian_solid False the whole
        # block is skipped and the guard could never fire. dry_run.py caught
        # it -- "the generator ACCEPTED two_velocity_skeleton without
        # eulerian_solid" -- which is the entire reason the negative test was
        # written before the guard was trusted. A check that cannot fail has
        # not passed.
        two_velocity = bool(child.get("two_velocity_skeleton", False))
        if two_velocity and not eulerian_solid:
            raise ValueError(
                f"{child_id}: two_velocity_skeleton requires eulerian_solid. "
                f"Vs is built from be through elasticStress(), and the "
                f"partitioned path solves for a displacement skeleton instead."
            )

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
        mechanics_on = bool(series.get("clot_mechanics", True))
        if not mechanics_on and series.get("growth"):
            raise ValueError(
                f"{child_id}: clot_mechanics is off but growth is on. The "
                f"rigid path holds the clot stationary; growing it there "
                f"means the obstruction changes shape during the very "
                f"measurement it is the independent variable of."
            )
        for key, value in {
            "clotMechanics": "yes" if mechanics_on else "no",
            "stronglyCoupledFSI": "no" if eulerian_solid else "yes",
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
            "k1": 0.0 if child.get("adhesion_off") else growth["k1"],
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
            "phi0": fixed["phi0"],
            "etaRes": fixed["etaRes"],
            "DgDetach": fixed["DgDetach"],
            "Gc": child.get("Gc_j_m2") or fixed["Gc_j_m2"],
            # THE PLASMA, per series (red team 05 #1): the Xu series runs
            # Xu's 4 cP (rho 1025, nu 3.9024e-6); every other series omits
            # the keys and keeps material_fixed byte for byte.
            "rho": series.get("rho_kg_m3", fixed["rho_kg_m3"]),
            "nu": series.get("nu_m2_s", fixed["nu_m2_s"]),
            "nSkeletonCorrectors": coupling["nSkeletonCorrectors"],
            "skeletonFixedPointTolerance": coupling["skeletonFixedPointTolerance"],
            "nFSICorrectors": 1 if eulerian_solid else coupling["nFSICorrectors"],
            "fsiTolerance": coupling["fsiTolerance"],
            "fsiRelaxation": coupling["fsiRelaxation"],
        }.items():
            replace_entry(props, key, value)

        # PERMEABILITY MODEL: per CHILD, then per series, then the porosity
        # axis's, so nothing that does not ask moves. source33_replication asks
        # for `constant` because kozenyCarman's clot permeability is the
        # numerical floor rather than k0p -- see the series' own
        # _why_constant_permeability.
        #
        # THE CHILD LEVEL WAS ADDED 2026-09-05 for `kozenyCarmanSolid`, which
        # is a property of the two-velocity SOLVER path rather than of the
        # throat_embolus series -- em_2vel needs it and its four siblings
        # (th_hold, th_break, em_free, em_freeG40) must not move.
        replace_entry(
            props,
            "permeabilityModel",
            child.get(
                "permeability_model",
                series.get(
                    "permeability_model", controls["porosity_axis"]["model"]
                ),
            ),
        )

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

        # THE PHASE FIELD MOVES WITH ITS MATERIAL. Off by default, and off
        # means the previous advection exactly.
        add_or_replace_entry(
            props,
            "phaseFieldMaterialVelocity",
            "yes" if series.get("phase_field_material_velocity", False) else "no",
        )

        # DIFFUSE-DOMAIN SKELETON WEIGHTING. Off by default, and off means the
        # assembled matrix is bit-for-bit the one every completed run used.
        add_or_replace_entry(
            props,
            "skeletonDiffuseDomain",
            "yes" if series.get("skeleton_diffuse_domain", False) else "no",
        )
        replace_entry(props, "fsiAitken", "yes" if coupling["fsiAitken"] else "no")
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
        content_form = bool(child.get("platelet_content_formulation", True))
        add_or_replace_entry(
            props, "plateletContentFormulation", "yes" if content_form else "no"
        )
        add_or_replace_entry(
            props, "plateletFluxConsistent", "no" if content_form else "yes"
        )

        # TERM-BY-TERM BUDGET FOR THE w EQUATION. Off by default because it
        # costs eight extra matrix assemblies per audited step. Run
        # 20260819-030512 left exactly one question standing -- int(q*w) grows
        # 8.28x under an equation whose only source is a SINK, with growth
        # entirely off -- and three proposed mechanisms for the w runaway have
        # already been falsified by measurement. This machinery already existed
        # in the solver and had never been switched on.
        #
        # add_or_replace_entry, NOT the replace_entry dict above: these keys are
        # not in the template, and that dict's writer deliberately refuses keys
        # the template never declared. The refusal fired on the first attempt
        # here and was correct.
        add_or_replace_entry(
            props,
            "operatorAudit",
            "yes" if child.get("operator_audit", False) else "no",
        )
        add_or_replace_entry(
            props,
            "operatorAuditEvery",
            int(child.get("operator_audit_every", 20)),
        )
        add_or_replace_entry(
            props, "finiteStrain", "yes" if fixed["finiteStrain"] else "no"
        )
        # CONSTITUTIVE LAW, written for every child so it is recorded rather
        # than inherited from whatever the solver default happens to be. The
        # source-33 child asks for neoHookean because a 4 Pa shell at 1000 1/s
        # is taken well past where St Venant-Kirchhoff is valid; run
        # 20260814-224138 diverged at 72% strain doing exactly that.
        # Verified equivalent to St Venant-Kirchhoff at small strain by the
        # constitutive twins of the elastic-slab gate, run 20260815-072849:
        # both landed on max|Us| = 5.096e-12 m against a closed form.
        constitutive_model = series.get("constitutive_model", "stVenantKirchhoff")
        if constitutive_model not in ("stVenantKirchhoff", "neoHookean"):
            raise ValueError(
                f"{child_id}: constitutive_model {constitutive_model!r} is not "
                f"known; use stVenantKirchhoff or neoHookean"
            )
        if constitutive_model == "neoHookean" and not fixed["finiteStrain"]:
            raise ValueError(
                f"{child_id}: constitutive_model neoHookean requires "
                f"finiteStrain; the solver refuses this combination too, but "
                f"catching it here costs nothing"
            )
        add_or_replace_entry(props, "constitutiveModel", constitutive_model)

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
        if eulerian_solid:
            if not fixed["finiteStrain"]:
                raise ValueError(
                    f"{child_id}: eulerian_solid requires finiteStrain. The "
                    f"whole point of transporting be is finite kinematics, and "
                    f"the solver's constructor validates this before it ever "
                    f"consults eulerianSolid -- so the child would die at "
                    f"construction in under a second, which is exactly how the "
                    f"first six children of the stage-2 gate were lost."
                )
            # This looked in series["strongly_coupled_fsi"], which does not
            # exist -- the flag lives in controls["coupling"] -- so the guard
            # could never fire and run 20260817-223437 went out with
            # stronglyCoupledFSI on and 60 FSI correctors. Not fatal there, but
            # it would have priced the speedup against the wrong configuration.
            # It is now a WARNING plus an override rather than a refusal,
            # because the FSI settings are shared with every other child in
            # this case and refusing would make the two mutually exclusive.
            if coupling.get("stronglyCoupledFSI"):
                # State the number actually written. Saying "forcing to 1" on a
                # child that is deliberately running 5 would make the console
                # log disagree with the dictionary, and the console log is what
                # gets read when a run is being explained months later.
                outer_note = child.get("n_outer_correctors")
                # WHY the FSI iteration goes, and it is NOT the same reason on
                # the two paths. On the one-velocity Eulerian path there is no
                # relative motion to iterate on at all. On the two-velocity
                # path there certainly is -- that is the whole point of it --
                # but the exchange is solved MONOLITHICALLY, by the per-cell
                # 2x2 block in correctPressure(), not by a partitioned outer
                # sweep. Saying "one velocity" about a two-velocity child would
                # be false in the console log that gets read months later.
                reason = (
                    "The fluid/skeleton exchange is solved monolithically by "
                    "the 2x2 block, not by a partitioned outer sweep"
                    if two_velocity else
                    "One velocity means there is no fluid/skeleton relative "
                    "motion to iterate on, and iterating it anyway is what "
                    "the reformulation exists to remove"
                )
                print(
                    f"  {child_id}: eulerianSolid -- forcing stronglyCoupledFSI "
                    f"off and nFSICorrectors to 1. {reason}. "
                    f"nOuterCorrectors = "
                    + (f"{int(outer_note)} (PER-CHILD OVERRIDE, not the "
                       f"Eulerian default of 1)" if outer_note is not None
                       else "1")
                )
        add_or_replace_entry(
            props, "eulerianSolid", "yes" if eulerian_solid else "no"
        )
        if eulerian_solid:
            # PICARD SWEEPS OF THE EXPLICIT STRETCH TERM, written explicitly so
            # the value a run used is in its own dictionary rather than being
            # the solver's default. Run 20260817-234636 showed three is not
            # always enough here: beSweepChange grew from 3.8e-16 to 7.9e-4
            # over the run, i.e. the iteration stopped converging as the clot
            # reached stretch 4.6. The diagnostic sweeps this.
            add_or_replace_entry(
                props, "beCorrectors", int(child.get("be_correctors", 3))
            )
            # PROJECT THE TRACE OUT OF THE STRETCH SOURCE. det(be) must be
            # exactly 1 for any isochoric deformation, and d(ln det be)/dt =
            # 2 tr(L), so it is conserved iff tr(L) = 0. Run 20260818-081527
            # measured |tr L|/|L| up to 1.0 in the clot bulk while |div phi|
            # stayed at 3.3e-8: the face flux is solenoidal and the
            # cell-centred gradient is not. det(be) reached 4.24 and neither
            # ten Picard sweeps nor a quarter time step moved it.
            #
            # Off by default so the A/B is a real comparison and not an
            # assertion.
            add_or_replace_entry(
                props,
                "beTraceFreeSource",
                "yes" if child.get("be_trace_free_source", False) else "no",
            )
            # ONE VELOCITY MEANS ONE FLUX. Under eulerianSolid the phase field
            # was being advected by a chi-blend of phi_ and a flux rebuilt from
            # the cell-centred vs -- a construction that belongs to the
            # PARTITIONED formulation, where the skeleton really does move at a
            # different velocity. On this path it is two defects: the rebuilt
            # flux is not divergence-free (run 20260818-125909 measured its
            # divergence at 527 1/s in the clot against |div phi| <= 8e-6), and
            # vs is the Eulerian derivative of a passive Us rather than the
            # material velocity, which is why vsMax exceeded max|U| in three of
            # four children. Defaults ON: it is written explicitly here so the
            # value a run used is in its own dictionary, and so the A/B can
            # turn it off.
            # Build the stretch term's velocity gradient by Gauss on a face
            # velocity whose normal component is exactly phi_/|Sf|, so its
            # trace is div(phi) and machine zero by construction. Written
            # explicitly so the value a run used is in its own dictionary.
            # Rescale be by det^(-1/3) each step. A BAND-AID and named one:
            # it restores the volume without undoing the advective smearing
            # that destroyed it. Default OFF so the det gate cannot pass
            # forever on a masked drift.
            add_or_replace_entry(
                props,
                "beDetRenormalise",
                "yes" if child.get("be_det_renormalise", False) else "no",
            )
            # Apply the elastic force through the FACE flux instead of as a
            # cell-centred source inside UEqn.H(). Default OFF so every
            # existing child keeps the behaviour it was measured with, and
            # written explicitly so the value a run used is in its own
            # dictionary rather than inferred from a solver default.
            #
            # This is the A/B for the momentum-force imbalance: cell-centred
            # tr(grad U) is 240-527 1/s in the clot while div(phi) is 6e-6,
            # eight orders, scaling 1400x with G while divPhiMax stays flat.
            # Rationale and the two dead rival explanations are on
            # faceConsistentElasticForce_ in phaseClotAxisymmetricFluid.H.
            add_or_replace_entry(
                props,
                "faceConsistentElasticForce",
                "yes"
                if child.get("face_consistent_elastic_force", False)
                else "no",
            )
            add_or_replace_entry(
                props,
                "beFluxConsistentGradU",
                "yes" if child.get("be_flux_consistent_grad_u", True) else "no",
            )
            add_or_replace_entry(
                props,
                "beTermDiagnostic",
                "yes" if child.get("be_term_diagnostic", False) else "no",
            )
            add_or_replace_entry(
                props,
                "eulerianMaterialFluxIsPhi",
                "yes" if child.get("eulerian_material_flux_is_phi", True)
                else "no",
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
            add_or_replace_entry(
                props,
                "bondHoldCarrier",
                "yes" if child.get("bond_hold_carrier", False) else "no",
            )
            add_or_replace_entry(
                props,
                "bondHoldTaper",
                "yes" if child.get("bond_hold_taper", False) else "no",
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
            brinkman = bool(child.get("eulerian_brinkman", False))
            if brinkman and not series.get("advect_released", False):
                raise ValueError(
                    f"{child_id}: eulerian_brinkman requires the series' "
                    f"advect_released. updateReleasedIndicator() returns with "
                    f"released = 0 everywhere when advectReleased is off, so "
                    f"the drag would hold a fully detached embolus in place -- "
                    f"the exact objection the release gate exists to answer."
                )
            add_or_replace_entry(
                props, "eulerianBrinkman", "yes" if brinkman else "no"
            )

            # TWO-VELOCITY SKELETON. The clot gets its own velocity, so the
            # phase field and be ride the SKELETON rather than the fluid, and
            # div(sigma_e) moves out of the fluid momentum equation into the
            # skeleton balance.
            #
            # REQUIRES eulerianBrinkman, and the reason is structural rather
            # than stylistic: the drag pair that becomes dragCoeff*(U - Vs) is
            # assembled inside the eulerianBrinkman block. Without it the phase
            # field would ride the skeleton while the fluid drag still pinned U
            # toward the LABORATORY frame -- the two halves of the change
            # disagreeing with each other, which is worse than neither.
            # two_velocity was read and checked against eulerian_solid above.
            if two_velocity and not brinkman:
                raise ValueError(
                    f"{child_id}: two_velocity_skeleton requires "
                    f"eulerian_brinkman. The relative-velocity drag pair "
                    f"dragCoeff*(U - Vs) is assembled inside the "
                    f"eulerianBrinkman block, so without it the phase field "
                    f"would be carried by the skeleton while the fluid was "
                    f"still dragged toward the lab frame."
                )
            add_or_replace_entry(
                props,
                "twoVelocitySkeleton",
                "yes" if two_velocity else "no",
            )

            if two_velocity:
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
                # THE MATERIAL CARRIER, and it is the defect that killed
                # every two-velocity run on record.
                #
                # The two-velocity branch carried c and shell on the RAW
                # skeleton flux phiSkelOwn_ over the whole domain, including
                # pure blood, where that flux is the skeleton row of a 2x2
                # block held up only by the Kres regularisation. Run
                # 20260908-010836 measured |Vs| = 0.0386 m/s -- 110% of
                # max|U| -- in a lumen cell with chi = 4.03e-11.
                #
                # The record across preflight/ is unambiguous:
                #   carrier = phi_ (one-velocity):        9 children, 9 done.
                #   carrier = phiSkelOwn_ (two-velocity): 14 children,
                #     14 failed -- 13 SIGFPE inside advectMaterialLabel(),
                #     1 (cub_rchi) inert, its clot centroid flat to four
                #     digits from 14.7 to 30 ms while reporting 8.96 mm/s.
                #
                # Blending to phi_ as chi -> 0 is not a new construction: the
                # one-velocity branch has always built its carrier that way.
                # Inside the clot chi -> 1 and nothing changes, so the clot
                # still rides its own skeleton.
                #
                # DEFAULT OFF as of 2026-09-08, on Codex's recommendation,
                # made twice and resisted twice. It is an EXPERIMENTAL
                # intervention with no isolated evidence: Batch 25 ran it, but
                # bundled with three other changes, and the returned assessor
                # rejected both children outright (children_admissible = 0,
                # c_bound/wp/vs gates all false). "It completed" is not
                # validation when the completed state is inadmissible.
                #
                # Turning the default off discards nothing. Both Batch 25
                # children set material_carrier_blend explicitly in
                # mainCaseControls.json, so their configuration is unchanged
                # and reproducible; only children that never named the key are
                # affected, and none of those has run with it.
                add_or_replace_entry(
                    props,
                    "materialCarrierBlend",
                    "yes" if child.get("material_carrier_blend", False) else "no",
                )
                # STAGE 1 CONTENT MODEL. Default OFF for the same reason
                # materialCarrierBlend is: twelve returned two-velocity
                # children never name the key, and an ON default would hand
                # them a model they never ran.
                add_or_replace_entry(
                    props,
                    "contentModel",
                    "yes" if child.get("content_model", False) else "no",
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
                content_operator = str(child.get("content_operator", "content"))
                if content_operator not in ("content", "matched"):
                    raise SystemExit(
                        "child %s: content_operator must be 'content' or "
                        "'matched', found %r"
                        % (child.get("child_id"), content_operator)
                    )
                add_or_replace_entry(props, "contentOperator", content_operator)
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
                    child.get("platelet_content_model", False)
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
                validate_phase_weight_basis(child)
                validate_stage2_platelet(child, content_operator)
                if platelet_content:
                    add_or_replace_entry(
                        props,
                        "plateletFluxConvention",
                        str(child["platelet_flux_convention"]),
                    )
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
                    add_or_replace_entry(
                        props,
                        "plateletBindLocalisation",
                        str(child.get("platelet_bind_localisation",
                                      "uniform")),
                    )
                    if "platelet_adhesion_speed" in child:
                        add_or_replace_entry(
                            props,
                            "plateletAdhesionSpeed",
                            repr(float(child["platelet_adhesion_speed"])),
                        )
                    add_or_replace_entry(
                        props,
                        "phaseWeightBasis",
                        str(child.get("phase_weight_basis", "legacy")),
                    )
                    add_or_replace_entry(
                        props,
                        "phaseForceBasis",
                        str(child.get("phase_force_basis", "legacy")),
                    )
                    add_or_replace_entry(
                        props,
                        "phaseIndicatorBasis",
                        str(child.get("phase_indicator_basis", "legacy")),
                    )
                    # OI-6. DEFAULTS TO "legacy", UNLIKE phaseDeltaBasis
                    # below, and the default is FORCED rather than chosen:
                    # "shared" is refused by the solver without
                    # phaseWeightBasis content + contentModel +
                    # twoVelocitySkeleton, and most children of this case have
                    # none of the three. A "shared" default here would abort
                    # every legacy child at startup. The solver's own fallback
                    # is "legacy" too, by the same two-place rule.
                    add_or_replace_entry(
                        props,
                        "phaseMomentumFluxBasis",
                        str(child.get("phase_momentum_flux_basis", "legacy")),
                    )
                    # S6a. DEFAULTS TO "legacy" for the same reason as
                    # phaseMomentumFluxBasis: "content" is refused without
                    # contentModel, and it redefines q -- which 53 sites read
                    # -- so a default flip would silently change the physics
                    # of every archived child. The solver's fallback is
                    # "legacy" too, by the two-place rule.
                    add_or_replace_entry(
                        props,
                        "phaseFluidFractionBasis",
                        # S7. WAS "legacy". S6c-2 REFUSES "legacy" here, so
                        # the old fallback made OMISSION fatal -- 201 cases
                        # omit these keys and aborted at construction.
                        # Flipped with the solver's own fallback in the same
                        # change, per the two-place rule. A child with the
                        # content model off is NOT revived by this: "content"
                        # is refused there too, and OI-24 records that such a
                        # child has no legal value for this key at all.
                        str(child.get("phase_fluid_fraction_basis", "content")),
                    )
                    # S6b-1 and S6b-2. BOTH DEFAULT TO "legacy", and both
                    # are written unconditionally by the two-place rule.
                    #
                    # phaseLimiterBasis picks the field
                    # phaseFieldFaceWeights() evaluates the limiter on; its
                    # weights reach the c equation, shell_ and Wp, so a
                    # default flip would change three equations in every
                    # archived child at once.
                    #
                    # phasePlateletFluxBasis picks the face values Wp is
                    # advected with. Both are refused by the solver without
                    # phaseFluidFractionBasis content, which most children of
                    # this case do not set, so a "content" default here would
                    # abort them at startup. The solver's fallbacks are
                    # "legacy" too.
                    add_or_replace_entry(
                        props,
                        "phaseLimiterBasis",
                        # S7. WAS "legacy". S6c-2 REFUSES "legacy" here, so
                        # the old fallback made OMISSION fatal -- 201 cases
                        # omit these keys and aborted at construction.
                        # Flipped with the solver's own fallback in the same
                        # change, per the two-place rule. A child with the
                        # content model off is NOT revived by this: "content"
                        # is refused there too, and OI-24 records that such a
                        # child has no legal value for this key at all.
                        str(child.get("phase_limiter_basis", "content")),
                    )
                    add_or_replace_entry(
                        props,
                        "phasePlateletFluxBasis",
                        str(child.get("phase_platelet_flux_basis", "legacy")),
                    )
                    # DEFAULTS TO "content", UNLIKE THE THREE ABOVE.
                    # Alireza, 2026-09-16: the new method is the default.
                    # The solver's own fallback for phaseDeltaBasis is
                    # "content" too -- they had to move together, because this
                    # generator writes the key unconditionally and so the
                    # solver default is otherwise never reached on this case.
                    # A child that wants the c-derived surface delta must say
                    # phase_delta_basis "legacy" explicitly; the S3 control is
                    # the reason that value still exists.
                    add_or_replace_entry(
                        props,
                        "phaseDeltaBasis",
                        str(child.get("phase_delta_basis", "content")),
                    )
                    # S6c-1. DEFAULTS TO "legacy", and unlike phaseDeltaBasis
                    # it must stay that way until the instrument has spoken.
                    # normal_ is the LAST live consumer of c: it reaches the
                    # physics only through shearPa_ -> adhesionEnhancement_
                    # -> adhesionSpeed_ -> Vn_, and that path is clamped to
                    # exactly 1 while shearRate < gammaLow. Whether the clamp
                    # holds on the current parameters is NOT yet measured --
                    # the 782 1/s figure behind the 2026-09-16 deferral is
                    # from an older run and shearRate is neither written nor
                    # logged by em43. PHASE_CLOT_NORMAL_BASIS measures it.
                    # Written unconditionally by the two-place rule; the
                    # solver's own fallback is "legacy" too.
                    add_or_replace_entry(
                        props,
                        "phaseNormalBasis",
                        # S7. WAS "legacy". S6c-2 REFUSES "legacy" here, so
                        # the old fallback made OMISSION fatal -- 201 cases
                        # omit these keys and aborted at construction.
                        # Flipped with the solver's own fallback in the same
                        # change, per the two-place rule. A child with the
                        # content model off is NOT revived by this: "content"
                        # is refused there too, and OI-24 records that such a
                        # child has no legal value for this key at all.
                        str(child.get("phase_normal_basis", "content")),
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
                        child.get("phase_growth_driver", "transfer")
                    )
                    if growth_driver not in ("front", "transfer"):
                        raise SystemExit(
                            "child %s: phase_growth_driver must be 'front' or "
                            "'transfer'; found %r" % (cid, growth_driver)
                        )
                    if growth_driver == "transfer":
                        # Each of these is refused by the solver too. They are
                        # repeated here because the generator runs locally and
                        # the solver runs in Colab: catching it here costs
                        # seconds, catching it there costs a VM boot and a
                        # packaged binary.
                        if not child.get("platelet_content_model", False):
                            raise SystemExit(
                                "child %s: phase_growth_driver 'transfer' "
                                "requires platelet_content_model true -- it "
                                "removes the Stefan front and replaces it "
                                "with a transfer nothing would compute, so "
                                "the clot would never grow while every "
                                "balance still closed." % (cid,)
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
                        if str(
                            child.get("phase_delta_basis", "content")
                        ) != "content":
                            raise SystemExit(
                                "child %s: phase_growth_driver 'transfer' "
                                "requires phase_delta_basis 'content'. S3 "
                                "must precede S4, or the theta clot grows on "
                                "the geometry of the c clot it replaces."
                                % (cid,)
                            )
                        if content_operator == "matched":
                            raise SystemExit(
                                "child %s: phase_growth_driver 'transfer' is "
                                "incompatible with content_operator "
                                "'matched' -- the matched operator maps the c "
                                "equation's Stefan front into theta, and "
                                "'transfer' removes that front, so theta "
                                "would be sourced twice." % (cid,)
                            )
                    add_or_replace_entry(
                        props, "phaseGrowthDriver", growth_driver
                    )
                    if child.get("unlocalised_phase_weights_accepted", False):
                        add_or_replace_entry(
                            props,
                            "unlocalisedPhaseWeightsAccepted",
                            "yes",
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
                add_or_replace_entry(
                    props,
                    "plateletContentModel",
                    "yes" if platelet_content else "no",
                )
                # Declared volume per bound platelet-equivalent. Inert while
                # kinetics are off -- Stage 1 multiplies nothing by it -- so
                # the placeholder cannot silently acquire authority before it
                # is sourced.
                add_or_replace_entry(
                    props, "vP", repr(float(child.get("v_p_m3", 1e-17)))
                )
                # THE Vn CAP, DISPROVED AND DEFAULTED OFF.
                # Batch 25 (run 20260908-061511, em_2vel_carrier) fired it
                # 2,505,301 times and cUnder still peaked at 1.47179651851
                # (c = -2.47179651851) at t = 0.01826716418; the pre-registered
                # pass condition was
                # qNegCells = 0 and it failed by every sample. It also cannot
                # work in principle: with chi = 1, div(v_s) = -100 1/s and
                # G = 0, implicit Euler gives c = -1.000298552, so compression
                # alone leaves the bound with no growth at all.
                add_or_replace_entry(
                    props,
                    "vnCapEnabled",
                    "yes" if child.get("vn_cap_enabled", False) else "no",
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
                _npm = str(child.get("network_pressure_model", "poleAtMax"))
                if _npm not in ("poleAtMax", "cubic", "poleCubic",
                                "convexExcess"):
                    raise SystemExit(
                        f"create_cases.py: {child['child_id']}: "
                        f"network_pressure_model {_npm!r} is not one of "
                        f"poleAtMax, cubic, poleCubic, convexExcess."
                    )
                add_or_replace_entry(props, "networkPressureModel", _npm)

                # S10 (2026-09-22): the three root-cause switches, see
                # research/designs/2026-09-22-S10-root-causes-em49.md. Each is
                # written ONLY when the child names it, so every existing
                # child's dictionary stays byte-identical and the solver's
                # shipped defaults apply. Refused here as well as in the
                # solver, for the reason given above: a typo should cost
                # nothing, not a VM boot.
                for _key, _dict_key, _allowed in S10_SWITCHES:
                    if _key not in child:
                        continue
                    _val = str(child[_key])
                    if _val not in _allowed:
                        raise SystemExit(
                            f"create_cases.py: {child['child_id']}: {_key} "
                            f"{_val!r} is not one of {', '.join(_allowed)}."
                        )
                    add_or_replace_entry(props, _dict_key, _val)

                # S11 (2026-09-23): the six root-cause switches of run
                # 20260923-031942 (research/designs/2026-09-23-S11-root-
                # causes-em52.md). Same contract as S10: written only when
                # the child names them, refused here on a typo.
                for _key, _dict_key, _allowed in S11_SWITCHES:
                    if _key not in child:
                        continue
                    _val = str(child[_key])
                    if _val not in _allowed:
                        raise SystemExit(
                            f"create_cases.py: {child['child_id']}: {_key} "
                            f"{_val!r} is not one of {', '.join(_allowed)}."
                        )
                    add_or_replace_entry(props, _dict_key, _val)

                # S12 (2026-09-23): the drag weighted as the stress already
                # is, occupancy x intrinsic (research/designs/2026-09-23-
                # S12-intrinsic-occupancy-drag.md). Same contract: written
                # only when the child names it, so every earlier child's
                # dictionary is byte-identical; refused here on a typo.
                for _key, _dict_key, _allowed in S12_SWITCHES:
                    if _key not in child:
                        continue
                    _val = str(child[_key])
                    if _val not in _allowed:
                        raise SystemExit(
                            f"create_cases.py: {child['child_id']}: {_key} "
                            f"{_val!r} is not one of {', '.join(_allowed)}."
                        )
                    add_or_replace_entry(props, _dict_key, _val)

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

                # S14 (2026-09-23): rupture at the critical stretch, and the
                # tangent elastic bound (research/designs/2026-09-23-S14-
                # network-rupture.md). Same contract as S10-S13; the stretch
                # is written only with the switch that reads it.
                for _key, _dict_key, _allowed in S14_SWITCHES:
                    if _key not in child:
                        continue
                    _val = str(child[_key])
                    if _val not in _allowed:
                        raise SystemExit(
                            f"create_cases.py: {child['child_id']}: {_key} "
                            f"{_val!r} is not one of {', '.join(_allowed)}."
                        )
                    add_or_replace_entry(props, _dict_key, _val)
                if child.get("network_rupture") == "criticalStretch":
                    add_or_replace_entry(
                        props, "networkCriticalStretch",
                        repr(float(child.get("network_critical_stretch", 2.8))),
                    )
                if child.get("network_rupture") == "progressiveDamage":
                    add_or_replace_entry(
                        props, "networkDamageOnsetStretch",
                        repr(float(child.get("network_damage_onset_stretch", 2.105))),
                    )
                    add_or_replace_entry(
                        props, "networkDamageStretchScale",
                        repr(float(child.get("network_damage_stretch_scale", 1.0))),
                    )

                # S16 (2026-09-24): terminal rupture with a timed release,
                # the damage carried as content, the occupancy floor and the
                # tensile pressure damage (research/designs/2026-09-24-S16-
                # timed-terminal-rupture.md). Same contract as S10-S15: every
                # key is written only when the child names it (or, for the
                # terminalRelease parameters, only with the switch that reads
                # them), so every earlier child's dictionary is byte-identical.
                # Refused beforehand by s16_value_refusal and
                # s16_combination_refusal, for every child.
                if child.get("network_rupture") in ("terminalRelease",
                                                    "fibreRelease"):
                    _prog = bool(child.get("network_damage_progressive", True))
                    add_or_replace_entry(
                        props, "networkCriticalStretch",
                        repr(float(child.get("network_critical_stretch", 2.8))),
                    )
                    add_or_replace_entry(
                        props, "networkReleaseTime",
                        repr(float(child.get("network_release_time_s", 3.3e-4))),
                    )
                    add_or_replace_entry(
                        props, "networkDamageProgressive",
                        "on" if _prog else "off",
                    )
                    if _prog:
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
                for _key, _dict_key, _allowed in S16_SWITCHES:
                    if _key not in child:
                        continue
                    _val = str(child[_key])
                    if _val not in _allowed:
                        raise SystemExit(
                            f"create_cases.py: {child['child_id']}: {_key} "
                            f"{_val!r} is not one of {', '.join(_allowed)}."
                        )
                    add_or_replace_entry(props, _dict_key, _val)
                # S17 (2026-09-26): the implicit tangent coupling. Written only
                # when the child names it, so every earlier child is
                # byte-identical; refused beforehand by s17_value_refusal and
                # s17_combination_refusal, for every child.
                for _key, _dict_key, _allowed in S17_SWITCHES:
                    if _key not in child:
                        continue
                    _val = str(child[_key])
                    if _val not in _allowed:
                        raise SystemExit(
                            f"create_cases.py: {child['child_id']}: {_key} "
                            f"{_val!r} is not one of {', '.join(_allowed)}."
                        )
                    add_or_replace_entry(props, _dict_key, _val)
                # S17b (2026-09-27): slack fibres, field-unit arithmetic, the
                # wall face. Same contract: written only when named.
                for _key, _dict_key, _allowed in S17B_SWITCHES:
                    if _key not in child:
                        continue
                    _val = str(child[_key])
                    if _val not in _allowed:
                        raise SystemExit(
                            f"create_cases.py: {child['child_id']}: {_key} "
                            f"{_val!r} is not one of {', '.join(_allowed)}."
                        )
                    add_or_replace_entry(props, _dict_key, _val)
                # S18 (2026-09-29): A, B and E. Same contract: written only
                # when named (C is networkPressureWeight, written with S13).
                for _key, _dict_key, _allowed in S18_SWITCHES:
                    if _key not in child:
                        continue
                    _val = str(child[_key])
                    if _val not in _allowed:
                        raise SystemExit(
                            f"create_cases.py: {child['child_id']}: {_key} "
                            f"{_val!r} is not one of {', '.join(_allowed)}."
                        )
                    add_or_replace_entry(props, _dict_key, _val)
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

        # The Courant bound on the MIXTURE MATERIAL velocity that advects c.
        # maxCo bounds the fluid, which is the wrong velocity here: run
        # 20260815-182036 held the fluid inside 0.5 while vsCourant climbed to
        # 1.45, because the skeleton was moving seven times faster than the
        # flow. Written explicitly rather than left to the solver default, so
        # the value a run used is in its own dictionary.
        add_or_replace_entry(
            props,
            "materialCourant",
            f"{float(series.get('material_courant', 0.5)):.16g}",
        )
        # The implicit stabiliser is raised for the ARTERIAL children. Run
        # 20260811-051109 lost deform_s2000, _s4000 and _s8000 to SIGFPE inside
        # solveSkeleton(): at that load the finite-strain geometric term stops
        # being a contraction against an operator sized for the material
        # stiffness alone. Raising K does not change the converged answer --
        # the two laplacians it sits between cancel identically -- so this is a
        # stabiliser, not a fudge. Whether it is ENOUGH is unknown; the
        # divergence guard in the solver reports the answer either way.
        #
        # Keyed off the series declaring a map rather than off the series NAME:
        # the demonstration child moved to an arteriolar rate too, and hard-
        # wiring this to "deformation" would have silently left it at the
        # venous factor of 1 at 1500 1/s, which is the load that produced the
        # SIGFPEs in the first place.
        implicit_factor = coupling["skeletonImplicitFactor"]
        if "skeleton_implicit_factor" in series:
            key = str(int(child["wall_shear_rate_1_s"]))
            if key not in series["skeleton_implicit_factor"]:
                raise ValueError(
                    f"{child['child_id']}: series '{child['series']}' declares "
                    f"skeleton_implicit_factor but has no entry for {key} 1/s. "
                    f"Falling back to the default of "
                    f"{coupling['skeletonImplicitFactor']} at arterial load is "
                    f"how deform_s2000/_s4000/_s8000 were lost to SIGFPE in run "
                    f"20260811-051109; refused rather than defaulted."
                )
            implicit_factor = series["skeleton_implicit_factor"][key]
        add_or_replace_entry(props, "skeletonImplicitFactor", implicit_factor)

        # THE PRESSURE FORCE ON THE SKELETON, absent until run
        # 20260811-051109. On wherever deformation or detachment is the
        # measurement; off for the growth series, which is only fitting an
        # interface speed and whose published result must not move.
        add_or_replace_entry(
            props, "skeletonPressureForce",
            "yes" if series.get("skeleton_pressure_force", False) else "no",
        )
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
        advecting = bool(series.get("advect_released", False))
        add_or_replace_entry(
            props, "advectReleased", "yes" if advecting else "no"
        )
        for key, value in {
            "shearEnhancementModel": growth["shearEnhancementModel"],
            "gammaLow": growth["gammaLow_1_s"],
            "gammaHigh": growth["gammaHigh_1_s"],
            "enhancementMax": growth["enhancementMax"],
            # THE GROWTH CLOSURE, written explicitly on every child from
            # 2026-09-02. The solver DEFAULTS to "gradient" so that an old
            # dictionary reproduces an old run byte for byte; nothing here may
            # rely on that default, because a child that silently fell back to
            # it would be a different model reported under the same name.
            "growthClosure": growth.get("growthClosure", "reaction"),
            "interfaceDepletionModel": growth.get(
                "interfaceDepletionModel", "none"
            ),
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
        if eulerian_solid:
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
            outer = 1 if eulerian_solid else coupling["nOuterCorrectors"]
        elif not eulerian_solid:
            raise ValueError(
                f"{child_id}: n_outer_correctors is an Eulerian-path override; "
                f"on the partitioned path nOuterCorrectors is the shared FSI "
                f"number and setting it per child would silently desynchronise "
                f"this child from every other one it is compared against."
            )
        replace_entry(solution, "nOuterCorrectors", int(outer))
        # S17 B (design 2B): the PISO corrector count per child -- 6 with
        # final_relaxation_unity, 2-4 in the named discriminators. Absent,
        # the shared coupling number, so every earlier child is unchanged.
        replace_entry(solution, "nCorrectors",
                      int(child.get("n_correctors", coupling["nCorrectors"])))
        mural_gen.replace_relaxation(solution, "p", coupling["pRelaxation"])
        mural_gen.replace_relaxation(solution, "U", coupling["URelaxation"])
        # UNLIMITED LINEAR CONVECTION, BOTH EQUATIONS, AS A DIAGNOSTIC.
        #
        # gate_a_nodiff (run 20260822-100024) left |Wp - q| at 2.27e-4 with
        # D = k = alpha = mobility = 0, Euler ddt, the SAME stored flux and a
        # guard-verified token-identical `bounded Gauss vanLeer` on both
        # equations, starting from Wp = q exactly. Step 1 is exact; ~1.3e-6 is
        # injected per step from step 2 and accumulates linearly. 7528 of
        # 12500 cells stay bit-identical, so it is not a global mismatch.
        #
        # The remaining suspect is the LIMITER: vanLeer weights are computed
        # from the transported field -- from c in one equation and from Wp in
        # the other. The argument that this is safe rests on the limiter ratio
        # being invariant under the affine map q = 0.5*(c+1). The measurement
        # contradicts that argument, and OpenFOAM's scalar limiter internals
        # cannot be read on this machine, so it is TESTED rather than guessed.
        #
        # `linear` weights are field-independent, hence provably identical for
        # c and Wp. If the deviation collapses to machine precision the limiter
        # is the cause; if it survives, the cause is outside the convection
        # operator entirely. Both entries are rewritten TOGETHER from one flag
        # -- they cannot desynchronise, and verify_platelet_carrier_scheme()
        # below still enforces token-identity afterwards.
        #
        # Unlimited linear on a sharp interface WILL oscillate. That is
        # acceptable for a 1e-3 s diagnostic and c bounds are reported. This is
        # not a configuration any production run may use.
        if child.get("linear_schemes"):
            schemes = case / "system" / "fvSchemes"
            text = schemes.read_text()
            # Only the c entry now. The Wp equation has no scheme of its own
            # -- it reuses c's captured weights -- so changing c changes BOTH
            # sides by construction, which is what this probe wanted all along
            # and now gets for free.
            for name in ("div(phiReleased,c)",):
                text, n = re.subn(
                    rf"(?m)^(\s*{re.escape(name)}\s+)[^;]+;",
                    lambda m: f"{m.group(1)}bounded Gauss linear;", text, count=1,
                )
                if n != 1:
                    raise ValueError(
                        f"{schemes}: linear_schemes could not rewrite {name}"
                    )
            schemes.write_text(text)

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

        # LINEAR-SOLVER FLOOR PROBE. Tightens ONLY the c and w solvers.
        #
        # gate_a_nodiff landed at |Wp - q| = 2.589e-11 against a 1e-11 limit
        # (run 20260823-002216) with the limiter fix in and D = 0, i.e. with
        # nothing left that is known to be wrong. Accumulated linear-solver
        # error over 395 steps at `tolerance 1e-12` could plausibly account for
        # it, which would make 1e-11 too tight -- but relaxing a limit because
        # it has become inconvenient is the exact pattern that hid this defect
        # for four runs. So it is MEASURED instead.
        #
        # THE FIRST ATTEMPT AT THIS PROBE MEASURED NOTHING, run 20260823-010445.
        # `gate_a_tol` asked for 1e-14 and got 2.5894e-11 against the 1e-12
        # control's 2.5893e-11. The reason is that the REQUESTED TOLERANCE WAS
        # NEVER BINDING: smoothSolver goes from an initial residual of 3.2e-04
        # to a FINAL residual of 3.2e-15 in three iterations, overshooting 1e-12
        # by three orders, and 1e-14 lands inside the same jump -- identical
        # iteration counts, final residuals 3.19962e-15 against 3.19928e-15.
        # Fifth test in this project set up where the mechanism under test could
        # not act. Read the ACHIEVED residual before choosing a tolerance.
        #
        # The corrected probe asks for a tolerance BELOW the achieved 3e-15 and
        # caps the iteration count, so Gauss-Seidel sweeps to its own round-off
        # floor instead of exiting on a satisfied criterion. That needs
        # `platelet_solver_maxiter` alongside the tolerance -- without a cap an
        # unreachable tolerance runs to the 1000-iteration default on every
        # solve of every step.
        #
        # p, U, be, Dg and Us are deliberately untouched -- changing them would
        # alter the flow and the probe would no longer be one variable. c and w
        # are both tightened because BOTH solves contribute to |Wp - q|; Wp
        # inherits w's entry via `$w`.
        if child.get("platelet_solver_tolerance"):
            tol = child["platelet_solver_tolerance"]
            text = solution.read_text()
            for field in ("c", "w"):
                text, n = re.subn(
                    rf"(?m)^(\s*{field}\s*\{{[^}}]*?tolerance\s+)[^;]+;",
                    lambda m: f"{m.group(1)}{tol};", text, count=1,
                )
                if n != 1:
                    raise ValueError(
                        f"{solution}: could not set the {field} solver "
                        f"tolerance; the fvSolution layout changed"
                    )
            solution.write_text(text)

        # The iteration cap that makes an unreachable tolerance affordable.
        # Applied to the same two entries and only those; Wp inherits w's via
        # `$w`, and the `Final` variants inherit through `$c` / `$w` too.
        if child.get("platelet_solver_maxiter"):
            iters = int(child["platelet_solver_maxiter"])
            text = solution.read_text()
            for field in ("c", "w"):
                text, n = re.subn(
                    rf"(?m)^(\s*{field}\s*\{{[^}}]*?relTol\s+0;\s*)\}}",
                    lambda m: f"{m.group(1)}maxIter {iters}; }}",
                    text, count=1,
                )
                if n != 1:
                    raise ValueError(
                        f"{solution}: could not set the {field} solver "
                        f"maxIter; the fvSolution layout changed"
                    )
            solution.write_text(text)
            check = solution.read_text()
            if check.count(f"maxIter {iters};") != 2:
                raise ValueError(
                    f"{solution}: expected exactly 2 maxIter entries, found "
                    f"{check.count(f'maxIter {iters};')}"
                )

        # S14: the rupture label's linear solver, ONLY for a child that
        # solves it, so every other child's fvSolution is byte-identical. The
        # same insertion the nP entry uses, refused rather than skipped if the
        # template's anchor is gone. It is shell_'s class of equation (the
        # same shared operator on the same carrier), so it takes shell's
        # solver, $c.
        rupture = child.get("network_rupture", "off") != "off"
        if rupture:
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
        partition = child.get("damage_transport") == "contentPartition"
        if partition:
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
        fibre = child.get("network_rupture") == "fibreRelease"
        implicit_tangent = child.get("elastic_coupling") == "implicitTangent"
        if fibre or implicit_tangent:
            solution_text = solution.read_text()
            entry = ""
            if fibre:
                if any(re.search(rf"(?m)^\s*{f}\s*\{{", solution_text)
                       for f in S17_FIBRE_FIELDS):
                    raise SystemExit(
                        "child %s: fvSolution already declares a pending-fibre "
                        "solver; the template changed" % (child.get("child_id"),))
                if str(child.get("fibre_slack", "off")) == "v1":
                    # S17b section 8 a: the passengers run P2's fixed entry
                    entry += (
                        "    // S17 A: the pending fibre's contents, carried with "
                        "thetaPending (S17b section 8 a: its fixed entry).\n"
                        + "".join(
                            f"    {f} {S17B_FIXED_SWEEP_ENTRY}\n"
                            f"    {f}Final {S17B_FIXED_SWEEP_ENTRY}\n"
                            for f in S17_FIBRE_FIELDS)
                    )
                else:
                    entry += (
                        "    // S17 A: the pending fibre's contents, carried with "
                        "thetaPending.\n"
                        + "".join(
                            f"    {f} {{ $c; }}\n    {f}Final {{ $c; relTol 0; }}\n"
                            for f in S17_FIBRE_FIELDS)
                    )
            if implicit_tangent:
                if re.search(r"(?m)^\s*Vs\s*\{", solution_text):
                    raise SystemExit(
                        "child %s: fvSolution already declares Vs; the "
                        "template changed" % (child.get("child_id"),))
                vs_solver = str(child.get("vs_linear_solver", S17_VS_SOLVERS[0]))
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
        slack_v1 = str(child.get("fibre_slack", "off")) == "v1"
        if slack_v1:
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
            "permeability_model": child.get(
                "permeability_model",
                series.get(
                    "permeability_model", controls["porosity_axis"]["model"]
                ),
            ),
            "skeleton_diffuse_domain": bool(
                series.get("skeleton_diffuse_domain", False)
            ),
            "phase_field_material_velocity": bool(
                series.get("phase_field_material_velocity", False)
            ),
            # tear_measurement locates the tear by Dg > DgDetach, so the
            # threshold has to travel with the child rather than being
            # re-guessed downstream.
            "DgDetach": fixed["DgDetach"],
            "advect_released": advecting,
            # THE ANCHOR, IN THE MANIFEST. Recorded so the assessor can check
            # that the case a child ran actually carried the anchor its
            # child_id claims, instead of trusting the id.
            "release_support_limit": float(
                child.get("release_support_limit", 0.0)
            ),
            "bond_hold_dilation": int(child.get("bond_hold_dilation", -1)),
            "bond_hold_penalty": float(child.get("bond_hold_penalty", 0.0)),
            "bond_hold_carrier": bool(child.get("bond_hold_carrier", False)),
            "bond_hold_taper": bool(child.get("bond_hold_taper", False)),
            # THE SECOND ANCHOR, in the manifest for the same reason as the
            # first: bk_ctl and bk_on differ only in this key and in the
            # bond-hold keys above, and the assessor must be able to check
            # that from the manifest rather than from the child_id.
            "eulerian_brinkman": bool(child.get("eulerian_brinkman", False)),
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
        if (child.get("network_rupture") in ("terminalRelease", "fibreRelease")
                or any(k in child for k in _s16_keys)):
            _term = child.get("network_rupture") in ("terminalRelease",
                                                     "fibreRelease")
            _prog = bool(child.get("network_damage_progressive", True))
            spec["s16"] = {
                "networkRupture": str(child.get("network_rupture", "off")),
                "networkCriticalStretch": float(
                    child.get("network_critical_stretch", 2.8)),
                "networkReleaseTime": float(
                    child.get("network_release_time_s", 3.3e-4)),
                "networkDamageProgressive": "on" if _prog else "off",
                "networkDamageOnsetStretch": float(
                    child.get("network_damage_onset_stretch", 2.105)),
                "networkDamageStretchScale": float(
                    child.get("network_damage_stretch_scale", 1.0)),
                "skeletonFloorBasis": str(
                    child.get("skeleton_floor_basis", "intactWeight")),
                "damageTransport": str(
                    child.get("damage_transport", "volumeFlux")),
                "networkPressureDamage": str(
                    child.get("network_pressure_damage", "off")),
                "releaseLaw": ("none" if not _term
                               else "progressive+terminal" if _prog
                               else "terminal-only"),
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
        if (child.get("network_rupture") == "fibreRelease"
                or any(k in child for k in _s17_keys)):
            spec["s17"] = {
                "networkRupture": str(child.get("network_rupture", "off")),
                "elasticCoupling": str(child.get("elastic_coupling", "explicit")),
                "beDetRenormalise":
                    "yes" if child.get("be_det_renormalise") else "no",
                "pendingForm": ("fibre" if child.get("network_rupture")
                                == "fibreRelease" else "none"),
                "nCorrectors": int(child.get("n_correctors",
                                             coupling["nCorrectors"])),
            }
            if "s17_discriminator" in child:
                spec["s17"]["discriminator"] = str(child["s17_discriminator"])
            # verification 17: the swept maxDeltaT of D3/D3b in child.json,
            # where the renderer reads swept parameters (inside the s17
            # block, so no earlier child.json changes).
            if "max_delta_t_s" in child:
                spec["s17"]["maxDeltaT"] = float(child["max_delta_t_s"])
        # S17b: what PHASE_CLOT_S17B_CONFIG must echo -- ONLY for a child
        # that names an S17b key, so every earlier child.json is identical.
        if any(k in child for k, _d, _a in S17B_SWITCHES):
            spec["s17b"] = {
                "fibreSlack": str(child.get("fibre_slack", "off")),
                "fibreArithmetic": str(child.get("fibre_arithmetic", "shipped")),
                "skeletonWallFace": str(child.get("skeleton_wall_face", "shipped")),
            }
        # S18: what PHASE_CLOT_S18_CONFIG must echo -- ONLY for a child that
        # names an S18 key or the contentRatio weight, so every earlier
        # child.json is identical.
        if (any(k in child for k, _d, _a in S18_SWITCHES)
                or child.get("network_pressure_weight") == "contentRatio"):
            spec["s18"] = {
                "frozenPendingPressure": str(child.get("frozen_pending_pressure", "shipped")),
                "wallPatchSplit": str(child.get("wall_patch_split", "shipped")),
                "networkPressureWeight": str(child.get("network_pressure_weight",
                                                       "depositedOccupancy")),
                "cplusOrder": str(child.get("cplus_order", "shipped")),
            }
        # S18p: what PHASE_CLOT_S18P_CONFIG must echo -- ONLY for a child
        # that names proj_site_log, so every earlier child.json is identical.
        if any(k in child for k, _d, _a in S18P_SWITCHES):
            spec["s18p"] = {"projSiteLog": str(child.get("proj_site_log", "no"))}
        # S19: what PHASE_CLOT_S19_CONFIG must echo -- ONLY for a child that
        # names an S19 key, so every earlier child.json is identical.
        if any(k in child for k in S19_KEYS):
            spec["s19"] = {
                "momentumPressureCoupling": str(
                    child.get("momentum_pressure_coupling", "segregated")),
                "networkViscosity": str(child.get("network_viscosity", "none")),
                "networkViscosityTau": float(child.get("tau_v_s", 2.5e-4)),
            }
            # S19b: ONLY for a child that names it, so every earlier
            # child.json is identical; absent, the assessor expects
            # eliminate under blockCoupled (D1).
            if "block_coupled_lumen" in child:
                spec["s19"]["blockCoupledLumen"] = str(child["block_coupled_lumen"])
        # XU (2026-09-30): what the assessor's config echo (A0) and field
        # tests (A1) read -- ONLY for a series that declares a plasma, a
        # flow axis or a core exemption, so every earlier child.json is
        # identical.
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
        (case / "child.json").write_text(json.dumps(spec, indent=2) + "\n")
        matrix.append(spec)

    # Ordered so an interrupted session still returns whole series.
    # Deformation runs before the demonstration children: they are cheap,
    # they carry no transport stage, and an interrupted Colab session should
    # lose the long children rather than the short ones.
    # Stage 3 runs FIRST. It is the cheapest thing in the file and it is a
    # go/no-go for stage 4, so an interrupted session must not lose it behind
    # three hours of children whose results are already recorded.
    order = {"eulerian_drift_probe": -5,
             "eulerian_diag_xu": -4, "eulerian_diag_li": -3,
             "eulerian_stage3_xu": -2, "eulerian_stage3_li": -1,
             "growth": 0, "required_strength": 1,
             "deformation": 2, "demonstration": 3, "cohesive_tear": 4,
             "source33_replication": 5,
             # LAST, DELIBERATELY. These are the longest children in the file
             # (2e-2 s against everything else's 3e-3), and the rule at the top
             # of this block is to lose the long children rather than the short
             # ones. When the saturation question is the point of the session
             # they are selected by ONLY and the ordering does not bind.
             "eulerian_saturation": 6,
             # Same reasoning as eulerian_saturation: longer than the 3e-3
             # children and selected by ONLY when it is the point of the
             # session, so it sorts to the end.
             "eulerian_beterm": 7,
             # Two of its four children run to 2e-2 s, so it sorts behind
             # eulerian_beterm by the same rule. Selected by ONLY when it is
             # the point of the session.
             "eulerian_spdfix": 8,
             # Every child runs the full 1.2e-2 s window, so it is the most
             # expensive series in the file and sorts last.
             "eulerian_anchor": 9,
             # The seated core runs on a 1.96x mesh, so it is dearer still.
             "eulerian_seated": 10,
             # Last: a 24 ms window against eulerian_seated's 12 ms. Note the
             # cost does NOT simply double -- dt is elastic-wave-limited and
             # goes as 1/sqrt(G), so the SOFT child is the cheap one. Within
             # the series, child_id sorts st_ctl before st_xu, which is the
             # wrong way round for an interrupted session; see the ordering
             # override immediately below.
             "eulerian_stiffness": 11,
             # Both children are 4/40 and therefore CHEAP -- 0.84 h each at
             # the measured 0.2572 s/step -- but they sort last because they
             # are the newest question, not because they are the dearest.
             "eulerian_detrenorm": 12,
             # Last: the newest question. All three children are 4/40 with G
             # unchanged, so dt and the step count are dr_on's exactly --
             # 11736 steps at the MEASURED 0.3716 s/step = 1.21 h each, 3.6 h
             # for the series. The dilation does not touch dt.
             "eulerian_anchor_depth": 13,
             # Last: growth restored and the bond finally breakable. Same
             # 11736 steps (dt is elastic-wave-limited, G unchanged), so
             # ~1.21 h/child at the measured 0.3716 s/step PLUS growth
             # overhead the baseline did not carry -- budget ~4 h.
             "eulerian_growth_peel": 14,
             # Last: the 50 ms window. The 24 ms series could not answer its
             # own question because peakTraction had not saturated -- it was
             # at 86% of a 14.8-15.3 Pa asymptote and still climbing. ~24450
             # steps at the MEASURED 0.4008 s/step = ~2.7 h/child, ~8.2 h for
             # three. It sorts last because it is both the newest question and
             # by far the dearest series in the file.
             "eulerian_peel_propagation": 15,
             # Last: the newest question, and a 24 ms window rather than 50 --
             # "does the shell tear while the core holds" is a mechanism
             # question visible early, not a saturation question. Four
             # children at gr_hold's 11736 steps and its MEASURED 0.3716
             # s/step = 1.21 h each, ~4.8 h for the series; bk_soft is the
             # only one whose dt could differ and it moves k, not G, so it
             # does not. Cheaper per child than eulerian_peel_propagation and
             # it sorts behind it only because it is newer.
             "eulerian_brinkman": 16,
             # Last: the newest question and the dearest per child in the file
             # after eulerian_peel_propagation. 24450 steps at the MEASURED
             # 0.3859 / 0.3911 s/step = 2.62 h + 2.66 h, ~5.3 h for the pair.
             # Selected by ONLY when it is the point of the session, so the
             # ordering does not bind in practice.
             "eulerian_core_fate": 17,
             # Last: Dr Li's shell-stiffness sweep, requested 2026-09-01. Five
             # children at 30 ms. dt is elastic-wave-limited on gMax(GField_),
             # which is the CORE at 40 Pa in every child -- raising the SHELL
             # to 10/20/30/40 never exceeds it -- so every child takes cf_xu's
             # 2.045e-06 s step and none of them is dearer than the control.
             # 24450*(30/50) = 14670 steps at the MEASURED 0.3575 s/step =
             # 1.46 h each, ~7.3 h for the five.
             "shell_modulus_sweep": 18,
             "throat_occlusion": 19,
             # Last: the lodged-embolus arm, added after run 20260903-114913
             # measured that the journey arm cannot occlude at Xu's k. Its
             # children run a 10 ms window rather than 30 -- the hydraulic
             # question settles in about one viscous time (H^2/nu = 0.12 ms in
             # the throat) and the rest of the window is there to show whether
             # the fragment holds or washes out. ~6630 steps at the MEASURED
             # 1.008 s/step of 20260903-114913 = ~1.9 h/child.
             "throat_lodged": 20,
             # Last: the newest question and the only series that depends on
             # the relative-velocity drag. 30 ms, so ~19890 steps at the
             # MEASURED 1.008 s/step = ~5.6 h/child, ~5.6 h for two at
             # CONCURRENCY 2.
             "throat_embolus": 21,
             # Last: the Xu 2017 Figure 5 case (batch 61), four 10 ms
             # children at dt 2.604167e-6 s, 3840 steps, ~4.6 h each.
             "xu2017_fig5": 22,
             # Last: the free-shell throat series (design
             # 2026-10-02-case-free-shell-throat, rev. 3). 30 ms, wall_budget_h
             # 13 each; concurrency 2, one session.
             "throat_free_shell": 23}
    missing = {c["series"] for c in controls["children"]} - set(order)
    if missing:
        raise ValueError(f"no run order declared for series {sorted(missing)}")
    # WITHIN a series, child_id order is the default and is right almost
    # everywhere. eulerian_stiffness is the exception: its physics child
    # (st_xu, 1.4 h) must run before its control (st_ctl, 4.4 h) so that a
    # session that times out returns the child the series exists to produce.
    # Alphabetical order would run the dear control first and lose st_xu.
    # anchor_depth follows the same rule for a different reason: ad_d20 is the
    # reproducibility control and every other child is unreadable until it has
    # been checked against dr_on, so it must run FIRST even though all three
    # cost the same. Alphabetical order already gives that; it is pinned here
    # so a later rename cannot silently reorder it.
    # eulerian_peel_propagation INVERTS the usual control-first rule, and the
    # inversion is deliberate. At ~2.7 h/child an 8.2 h session is likely to be
    # cut short, so the order is chosen for what a TRUNCATED session returns.
    # pp_p11 is the only child guaranteed to produce the event the series
    # exists to observe, and pp_hold is what makes that event attributable
    # (without the erosion baseline, material lost by pp_p11 cannot be
    # separated from the ~1%/ms that goes anyway). Those two are the minimum
    # readable result, so they run first. pp_p15 is a coin-flip on a 5%
    # extrapolation and is therefore the cheapest thing to lose.
    # eulerian_growth_peel: gr_hold is BOTH the control and the measurement the
    # other two are bracketed against, so it runs first. Alphabetically gr_hold
    # already sorts before gr_p15/gr_p30, but the rungs must also run
    # high-strength first (p30 is the one expected to HOLD, so a session that
    # times out returns the informative pair).
    # eulerian_brinkman keeps the control FIRST, and here that is not a habit
    # but the precondition for reading anything else. bk_ctl has the new
    # switch OFF, so it must reproduce gr_hold's solution exactly; if it does
    # not, the edit touched something it should not have and the other three
    # children are measuring an unknown change as well as the intended one.
    # bk_on is next because it is the clean test, bk_both third because it
    # only refines an answer bk_on has already given, and bk_soft last because
    # it is contingent -- it is the response to an over-constrained core, and
    # if the core is not over-constrained it answers nothing.
    within = {"st_xu": 0, "st_ctl": 1, "dr_off": 0, "dr_on": 1,
              "ad_d20": 0, "ad_d16": 1, "ad_d12": 2,
              "gr_hold": 0, "gr_p30": 1, "gr_p15": 2,
              "pp_p11": 0, "pp_hold": 1, "pp_p15": 2,
              "bk_ctl": 0, "bk_on": 1, "bk_both": 2, "bk_soft": 3,
              # eulerian_core_fate runs the QUESTION first, not the control,
              # and that is safe here in a way it usually is not: BOTH children
              # already have a 24 ms trajectory on disk from run
              # 20260827-190800, so a truncated session still leaves cf_soft
              # comparable against something. cf_soft is the child the series
              # exists to answer, so it is the one that must survive.
              "cf_soft": 0, "cf_xu": 1,
              # shell_modulus_sweep runs the CONTROL first, for bk_ctl's
              # reason: g04 is cf_xu's material at a shortened window, so it
              # must reproduce cf_xu's trajectory step for step at 30 ms. If
              # it does not, the sweep is measuring an unknown change as well
              # as the intended one. After the control the order is chosen for
              # what a TRUNCATED session returns: g40 is the homogeneous
              # endpoint and the largest contrast, g20 is the midpoint, so
              # g04/g40/g20 is a readable three-point curve on its own. g10
              # and g30 only refine it.
              "g04": 0, "g40": 1, "g20": 2, "g10": 3, "g30": 4,
              # th_hold first: it is the child that can produce the
              # result, and th_break is the control that explains it. If a
              # session is cut after one child there is a result, not a
              # control with nothing to control.
              "th_hold": 0, "th_break": 1,
              # S16, batch 57 (design section 6, red-team Change 8): em64 (G 4,
              # ~40 min) and em63 (G 400, 3-12 h, wall budget 14 h) start
              # together at CONCURRENCY 2, and em65 takes em64's slot when
              # em64 finishes. At CONCURRENCY 1 the same order would make em65
              # wait for em63's whole budget, which is why batch 57 is
              # declared at CONCURRENCY 2.
              "em64_G4_s16": 0, "em63_G400_s16": 1, "em65_G4_s16_final": 2,
              # S17, batch 58 (design section 6, change 12): the
              # discriminators first, then em67 with em66, then em68 with
              # em69, then em70 -- in two slots (CONCURRENCY 2).
              "s17_D1_nc3": 0, "s17_D2_nc4": 1, "s17_D2b_nc6": 2,
              "s17_D3_dt18": 3, "s17_D3b_dt25": 4, "s17_CB6": 5, "s17_CB4": 6,
              "em67_G4_s17": 7, "em66_G400_s17": 8,
              "em68_G400_s17_law": 9, "em69_G400_s17_noimpl": 10,
              "em70_G4_s17_noimpl": 11,
              # S17b, batch 59 (design section 3): em72 (G 4, 3 h) with em71
              # (G 400, 14 h), then em73 with em74 (2.5 ms each), CONCURRENCY 2.
              "em72_G4_s17b": 12, "em71_G400_s17b": 13,
              "em73_G400_s17b_noimpl": 14, "em74_G400_s17b_noA3": 15,
              # S18, batch 60 (design section 4): em76 alone.
              "em76_G4_s18": 16,
              # Xu 2017 Figure 5, batch 61 (design section 3): 250 with 500,
              # then 1000 with 2000, at CONCURRENCY 2. Pinned, because
              # child_id order would run g1000 and g2000 first.
              "xu5_g250": 17, "xu5_g500": 18, "xu5_g1000": 19,
              "xu5_g2000": 20,
              # S18p, batch 62 (design section 4): em78 with em79, then em80
              # with em81, at CONCURRENCY 2.
              "em78_G4_s18p_ctl": 21, "em79_G4_s18p_Cnp": 22,
              "em80_G4_s18p_Afz": 23, "em81_G4_s18p_Esh": 24,
              # BATCH 68 (design 2026-10-05 rev. 3, section 2.1): dense
              # first, then loose, then the OI-48 control pair, so at
              # CONCURRENCY 2 slot 2 runs loose then the pair. After every
              # earlier throat_free_shell child (all 0), whose order is
              # unchanged.
              "fsf_G40_klo_v250": 1, "fsf_G4_kxu_v250": 2,
              "fsb_G40_klo_rest": 3, "fs_G40_klo_rest_seg": 4}
    # S17 (design section 6): the mechanical control check, before anything
    # is ordered for launch. The controls-entry half runs for every pair; the
    # child.json half for the pairs whose control is built in this run
    # (s17_control_check; fix round 13c).
    if any(c["child_id"] == "em66_G400_s17" for c in controls["children"]):
        s17_control_check(
            {c["child_id"]: c for c in controls["children"]},
            {s["child_id"]: s for s in matrix})
    if any(c["child_id"] == "em71_G400_s17b" for c in controls["children"]):
        s17_control_check(
            {c["child_id"]: c for c in controls["children"]},
            {s["child_id"]: s for s in matrix}, S17B_CONTROL_PAIRS)
    # S18 (design section 4): the parent rule on the controls entries.
    s18_parent_check({c["child_id"]: c for c in controls["children"]})
    s18p_parent_check({c["child_id"]: c for c in controls["children"]})
    fsp_parent_check({c["child_id"]: c for c in controls["children"]})
    s19_parent_check({c["child_id"]: c for c in controls["children"]})
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
