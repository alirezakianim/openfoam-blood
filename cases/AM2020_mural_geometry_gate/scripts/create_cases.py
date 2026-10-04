#!/usr/bin/env python3
"""Generate the four planar mural-geometry smoke children.

Planar, NOT axisymmetric: front and back are `empty`, the walls are `bottom`
(injured, carries the clot) and `top`. The 0/ fields and constant/ dictionaries
come from the axisymmetric template because they are geometry-agnostic - every
boundaryField is rewritten below.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path


GATE = Path(__file__).resolve().parents[1]
TEMPLATE = GATE.parent / "AM2020_straight_tube_growth_mechanics_smoke" / "template"
RUNS = GATE / "runs"


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


def replace_relaxation(path: Path, field: str, value: float) -> None:
    text = path.read_text()
    updated, count = re.subn(
        rf"(?m)^(\s*{re.escape(field)}\s+)[0-9.eE+-]+;",
        lambda m: f"{m.group(1)}{value};", text, count=1,
    )
    if count != 1:
        raise ValueError(f"{path}: expected one relaxation entry for {field}")
    path.write_text(updated)


def field_boundary(path: Path, body: str) -> None:
    text = path.read_text()
    start = text.find("boundaryField")
    if start < 0:
        raise ValueError(f"{path}: boundaryField not found")
    path.write_text(text[:start] + "boundaryField\n{\n" + body.rstrip() + "\n}\n")


def planar_mesh(g: dict) -> str:
    """A one-cell-thick planar slab. Patch names follow the existing
    AM2020_planar_* cases: inlet/outlet streamwise, bottom/top walls, empty
    front and back."""
    lx, ly = g["channel_length_m"], g["channel_height_m"]
    hz = 0.5 * g["depth_m"]
    return f"""FoamFile
{{
    format ascii;
    class dictionary;
    object blockMeshDict;
}}
convertToMeters 1;
vertices
(
    (0 0 {-hz:.16g})
    ({lx:.16g} 0 {-hz:.16g})
    ({lx:.16g} {ly:.16g} {-hz:.16g})
    (0 {ly:.16g} {-hz:.16g})
    (0 0 {hz:.16g})
    ({lx:.16g} 0 {hz:.16g})
    ({lx:.16g} {ly:.16g} {hz:.16g})
    (0 {ly:.16g} {hz:.16g})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({g['nx']} {g['ny']} {g['nz']}) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    bottom {{ type wall; faces ((0 1 5 4)); }}
    top {{ type wall; faces ((3 7 6 2)); }}
    front {{ type empty; faces ((4 5 6 7)); }}
    back {{ type empty; faces ((0 3 2 1)); }}
);
mergePatchPairs ();
"""


def us_bottom_entry(spec: dict) -> str:
    """The bond. This is the ONLY line that differs between the traction child
    and the peel children."""
    if spec["wall_condition"] == "fixedValue":
        return "    bottom { type fixedValue; value uniform (0 0 0); }"
    if spec["wall_condition"] == "clotWallAdhesion":
        # WHICH STRESS THE BOND READS. The condition defaults to "sigmaD",
        # which ONLY solveSkeleton() writes. The Eulerian path never calls
        # solveSkeleton() -- it runs solveElasticStrain() instead -- so on that
        # path the condition found sigmaD at its constructed value of ZERO and
        # computed a traction of exactly zero on every face, every step. Run
        # 20260824-033102 read peakTractionAny = 0 and utilisation = 0 for
        # 20 ms against a strength of 1e9 Pa and it meant the failure criterion
        # was INERT, not that the bond was unloaded: a strength of 1e-9 Pa
        # would have behaved identically.
        #
        # Eulerian children are therefore pointed at sigmaElastic, the
        # persistent member the solver now publishes after every
        # solveElasticStrain(). Partitioned children keep sigmaD, which their
        # own path does write, so nothing that already worked moves.
        sigma = "sigmaElastic" if spec.get("eulerian_solid") else "sigmaD"
        # FREE WALL (design 2026-10-02-case-free-shell-throat, rev. 3, S5;
        # red team 03 F1). With no `injury` entry the condition LATCHES every
        # face under the clot as an injury face with bonded 1 at its first
        # evaluation (clotWallAdhesionFvPatchVectorField.C ~296-310), which
        # seeds a bond nobody asked for. `wall_bond_free` writes `injury` and
        # `bonded` explicitly at 0, which sets injuryLatched_ true (the
        # constructor's `dict.found("injury")`) so the latch never runs, and
        # starts the wall with no bonded face. Opt-in only: every existing
        # child omits this key and keeps the implicit-latch entry it was
        # validated with.
        extra = ""
        if spec.get("wall_bond_free"):
            extra = (
                "\n        q                q;"
                "\n        injury           uniform 0;"
                "\n        bonded           uniform 0;"
            )
        return (
            "    bottom\n"
            "    {\n"
            "        type             clotWallAdhesion;\n"
            f"        adhesiveStrength {spec['adhesive_strength_pa']:.12g};\n"
            f"        sigma            {sigma};"
            f"{extra}\n"
            "        value            uniform (0 0 0);\n"
            "    }"
        )
    raise ValueError(f"unknown wall_condition {spec['wall_condition']}")


def us_top_entry(spec: dict) -> str:
    """The top wall's Us entry. Every existing child keeps the hard-wired
    `fixedValue (0 0 0)` (a pin the two-velocity skeleton has run on since
    this gate's first child). `top_wall_adhesion` (design S5, S7.3, red team
    03 #2 / 04 #1) makes the top wall an adhesion patch too, so the weld
    branch (every non-adhesion wall supports every clot cell touching it,
    clotWallAdhesionFvPatchVectorField.C ~11386-11395) cannot fire there
    either, and copies the bottom entry's adhesiveStrength/sigma/q keys so
    the two patches cannot silently disagree (red team 04 #1). It REQUIRES
    wall_bond_free: an adhesion top wall with an implicit bond would latch
    the top wall instead of leaving it traction-free, which is the opposite
    of what asking for this key means."""
    if not spec.get("top_wall_adhesion"):
        return "    top { type fixedValue; value uniform (0 0 0); }"
    if spec["wall_condition"] != "clotWallAdhesion":
        raise ValueError(
            "top_wall_adhesion requires wall_condition clotWallAdhesion; a "
            "fixedValue bottom wall has no adhesiveStrength/sigma to copy."
        )
    if not spec.get("wall_bond_free"):
        raise ValueError(
            "top_wall_adhesion requires wall_bond_free: an adhesion top wall "
            "with an implicit (latched) bond would weld on contact, which is "
            "the one configuration this key exists to avoid."
        )
    sigma = "sigmaElastic" if spec.get("eulerian_solid") else "sigmaD"
    return (
        "    top\n"
        "    {\n"
        "        type             clotWallAdhesion;\n"
        f"        adhesiveStrength {spec['adhesive_strength_pa']:.12g};\n"
        f"        sigma            {sigma};\n"
        "        q                q;\n"
        "        injury           uniform 0;\n"
        "        bonded           uniform 0;\n"
        "        value            uniform (0 0 0);\n"
        "    }"
    )


def configure_fields(case: Path, u_mean: float, spec: dict) -> None:
    # Flow is VELOCITY-driven, not pressure-driven: the wall shear rate is the
    # quantity the PI asked about and the one source 22 imposes.
    #
    # A UNIFORM inlet is correct here and no custom profile condition is
    # needed. Re = U*H/nu = 0.036, so this is Stokes flow and the entrance
    # length 0.05*Re*H = 1.1e-7 m is 0.07 of one cell: the parabolic profile
    # establishes within a fraction of the first cell. Only standard OpenFOAM
    # conditions are used, which matters because a non-existent type such as
    # `parabolicVelocity` would fatal at run time, not at build time.
    # PRESSURE-DRIVEN, when the child asks for it. Dr Li, 2026-09-01: "you
    # need to use pressure to drive the flow not enforcing a constant flow
    # inlet BCs". He is right, and it is not a preference: with a fixed inlet
    # velocity the flow rate CANNOT fall, so occluding the throat is
    # unobservable by construction -- the solver would raise pressure instead.
    #
    # The pair below is the one AM2020_planar_pressure_outlet established on
    # returned run 20260728-075920, which PASSED all 14 criteria on both
    # children. pressureInletOutletVelocity is taken rather than the
    # zeroGradient twin because it is the one that stays defined if a face
    # ever reverses; the two agreed on that gate, so nothing is traded.
    #
    # THE GATE'S OWN CAVEAT TRAVELS WITH IT: it is "a valid narrow E-08 result
    # for a forward-outflow, all-fluid planar" case. A static occlusion study
    # is inside that scope -- the recirculation behind a lodged clot is
    # INTERNAL and the boundaries stay forward -- but a child that drives the
    # blockage to full closure would leave it, and must not be read as covered.
    if spec.get("pressure_driven"):
        dp = spec["inlet_pressure_m2_s2"]
        field_boundary(case / "0" / "U", """    inlet { type zeroGradient; }
    outlet { type pressureInletOutletVelocity; value uniform (0 0 0); }
    bottom { type noSlip; }
    top { type noSlip; }
    front { type empty; }
    back { type empty; }""")
        field_boundary(case / "0" / "p", f"""    inlet {{ type fixedValue; value uniform {dp:.12g}; }}
    outlet {{ type fixedValue; value uniform 0; }}
    bottom {{ type zeroGradient; }}
    top {{ type zeroGradient; }}
    front {{ type empty; }}
    back {{ type empty; }}""")
    else:
        field_boundary(case / "0" / "U", f"""    inlet {{ type fixedValue; value uniform ({u_mean:.12g} 0 0); }}
    outlet {{ type zeroGradient; }}
    bottom {{ type noSlip; }}
    top {{ type noSlip; }}
    front {{ type empty; }}
    back {{ type empty; }}""")

        field_boundary(case / "0" / "p", """    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    bottom { type zeroGradient; }
    top { type zeroGradient; }
    front { type empty; }
    back { type empty; }""")

    field_boundary(case / "0" / "Us", f"""    inlet {{ type zeroGradient; }}
    outlet {{ type zeroGradient; }}
{us_bottom_entry(spec)}
{us_top_entry(spec)}
    front {{ type empty; }}
    back {{ type empty; }}""")

    # Platelets enter at the inlet at the reference concentration; everything
    # else is zeroGradient.
    field_boundary(case / "0" / "w", """    inlet { type fixedValue; value uniform 1; }
    outlet { type zeroGradient; }
    bottom { type zeroGradient; }
    top { type zeroGradient; }
    front { type empty; }
    back { type empty; }""")

    for name in ("c", "Dg", "H", "muC"):
        field_boundary(case / "0" / name, """    inlet { type zeroGradient; }
    outlet { type zeroGradient; }
    bottom { type zeroGradient; }
    top { type zeroGradient; }
    front { type empty; }
    back { type empty; }""")


def main() -> int:
    controls = json.loads(
        (GATE / "constant" / "muralGeometryControls.json").read_text()
    )
    if RUNS.exists():
        raise ValueError("runs/ already exists; use a fresh immutable gate copy")
    if not TEMPLATE.is_dir():
        raise ValueError(f"missing template: {TEMPLATE}")

    g = controls["geometry"]
    clot = controls["clot"]
    material = controls["material"]
    coupling = controls["coupling"]
    growth = controls["growth"]
    flow = controls["flow"]

    if g["nx"] * g["ny"] * g["nz"] != g["expected_cell_count"]:
        raise ValueError("cell count mismatch")
    if not math.isclose(g["channel_length_m"] / g["nx"], g["h_m"], rel_tol=1e-12):
        raise ValueError("streamwise spacing must equal h")
    if not math.isclose(g["channel_height_m"] / g["ny"], g["h_m"], rel_tol=1e-12):
        raise ValueError("wall-normal spacing must equal h")
    if not math.isclose(g["interface_width_m"] / g["h_m"], 4.0):
        raise ValueError("epsilonC/h must be 4")
    if not math.isclose(g["ell_m"] / g["h_m"], 2.0):
        raise ValueError("ell/h must be 2")

    # The trap that aborted smoke attempt 1: a clot thinner than a few
    # interface widths has no bulk and the initialiser refuses it. Check here,
    # before any mesh is written.
    c_base = math.tanh(-clot["height_m"] / g["interface_width_m"])
    if c_base > -0.9:
        raise ValueError(
            f"clot has no bulk: c at the base would be {c_base:.4f}. "
            f"Need height >= about 2.5*epsilonC = "
            f"{2.5 * g['interface_width_m']:.3g} m."
        )

    # The blockage fraction must match the declared one, or the predicted
    # traction that sizes the peel children is describing a different clot.
    blockage = clot["height_m"] / g["channel_height_m"]
    if not math.isclose(blockage, clot["blockage_fraction"], rel_tol=1e-9):
        raise ValueError(
            f"declared blockage {clot['blockage_fraction']} != {blockage}"
        )

    if not math.isclose(
        flow["mean_velocity_m_s"],
        flow["wall_shear_rate_1_s"] * g["channel_height_m"] / 6.0,
        rel_tol=1e-9,
    ):
        raise ValueError("U_mean must equal gammaDot*H/6 for plane Poiseuille")

    if material["Gc_j_m2"] < 1e5:
        raise ValueError("Gc must be unreachable so any release is adhesive")

    spec_by_id = {c["child_id"]: c for c in controls["children"]}
    if sorted(spec_by_id) != [
        "mural_growth", "mural_peel_hi", "mural_peel_lo", "mural_traction"
    ]:
        raise ValueError(f"unexpected child set: {sorted(spec_by_id)}")
    # The Allrun must derive its child list from runs/matrix.json. A
    # hard-coded list is what silently skipped two of Gate A's children in run
    # 20260810-075435; this list was correct at the time, which is exactly what
    # that one was too.
    allrun = (GATE / "Allrun").read_text()
    if "matrix.json" not in allrun:
        raise ValueError("Allrun must read its child list from runs/matrix.json")
    for child_id in spec_by_id:
        if child_id in allrun:
            raise ValueError(
                f"Allrun names the child '{child_id}' literally; it must "
                f"iterate matrix.json instead"
            )

    if spec_by_id["mural_traction"]["wall_condition"] != "fixedValue":
        raise ValueError("mural_traction must keep a welded bond: it MEASURES")
    if (
        spec_by_id["mural_peel_lo"]["adhesive_strength_pa"]
        >= spec_by_id["mural_peel_hi"]["adhesive_strength_pa"]
    ):
        raise ValueError("peel_lo must be weaker than peel_hi")

    RUNS.mkdir()
    matrix = []
    for spec_in in controls["children"]:
        child_id = spec_in["child_id"]
        phase = spec_in["phase"]
        t = controls["time"][phase]
        growing = bool(spec_in["growth"])

        case = RUNS / child_id
        shutil.copytree(TEMPLATE, case)
        (case / "system" / "blockMeshDict").write_text(planar_mesh(g))
        configure_fields(case, flow["mean_velocity_m_s"], spec_in)

        props = case / "constant" / "phaseClotProperties"
        for key, value in {
            "clotMechanics": "yes",
            "stronglyCoupledFSI": "yes",
            "D": growth["D_m2_s"],
            "alpha": growth["alpha_trial_m2_s"] if growing else 0.0,
            "k0": growth["k0_m_s"] if growing else 0.0,
            "k1": growth["k1"],
            "mobility": growth["mobility_m4_s"],
            "epsilonC": g["interface_width_m"],
            "ell": g["ell_m"],
            "G": material["G_pa"],
            "poissonRatio": material["poissonRatio"],
            "k0p": material["k0p_m2"],
            "phi0": material["phi0"],
            "etaRes": material["etaRes"],
            "DgDetach": material["DgDetach"],
            "Gc": material["Gc_j_m2"],
            "rho": material["rho_kg_m3"],
            "nu": material["nu_m2_s"],
            "nSkeletonCorrectors": coupling["nSkeletonCorrectors"],
            "skeletonFixedPointTolerance": coupling["skeletonFixedPointTolerance"],
            "nFSICorrectors": coupling["nFSICorrectors"],
            "fsiTolerance": coupling["fsiTolerance"],
            "fsiRelaxation": coupling["fsiRelaxation"],
            "shearEnhancementModel": growth["shearEnhancementModel"],
            "gammaLow": growth["gammaLow_1_s"],
            "gammaHigh": growth["gammaHigh_1_s"],
            "enhancementMax": growth["enhancementMax"],
        }.items():
            replace_entry(props, key, value)
        replace_entry(props, "permeabilityModel", material["permeabilityModel"])
        replace_entry(props, "fsiAitken", "yes" if coupling["fsiAitken"] else "no")
        add_or_replace_entry(props, "plateletFluxConsistent", "yes")
        # Both gates of this smoke phase exercise the new constitutive law.
        add_or_replace_entry(props, "finiteStrain", "yes")

        control = case / "system" / "controlDict"
        for key, value in {
            "deltaT": t["delta_t_s"],
            "endTime": t["end_time_s"],
            "writeControl": "runTime",
            "writeInterval": t["write_interval_s"],
        }.items():
            replace_entry(control, key, value)

        solution = case / "system" / "fvSolution"
        replace_entry(solution, "nOuterCorrectors", coupling["nOuterCorrectors"])
        replace_entry(solution, "nCorrectors", coupling["nCorrectors"])
        replace_relaxation(solution, "p", coupling["pRelaxation"])
        replace_relaxation(solution, "U", coupling["URelaxation"])

        (case / "case.foam").write_text("")

        spec = {
            "child_id": child_id,
            "experiment": "mural_geometry_gate",
            "role": spec_in["role"],
            "expected": spec_in["expected"],
            "wall_condition": spec_in["wall_condition"],
            "adhesive_strength_pa": spec_in["adhesive_strength_pa"],
            "growth": growing,
            "phase": phase,
            "alpha_m2_s": growth["alpha_trial_m2_s"] if growing else 0.0,
            "delta_t_s": t["delta_t_s"],
            "end_time_s": t["end_time_s"],
            "expected_steps": t["expected_steps"],
            "write_interval_s": t["write_interval_s"],
            "geometry": g,
            "clot": clot,
            "material": material,
            "flow": flow,
            "coupling": coupling,
            "predicted_traction": controls["predicted_traction"],
        }
        (case / "child.json").write_text(json.dumps(spec, indent=2) + "\n")
        matrix.append(spec)

    (RUNS / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"generated {len(matrix)} planar mural-geometry children")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"create_cases.py: {error}")
