#!/usr/bin/env python3
"""Assess the planar mural-geometry smoke gate.

Three questions, one per child group:

  mural_traction   does a one-sided clot load its bond with a GRADIENT?
  mural_peel_*     does it then peel PROGRESSIVELY?
  mural_growth     does the recalibrated alpha give Storti's rate?

The traction profile is computed from the written fields, not from a log line:
mural_traction keeps a welded bond and therefore prints no PHASE_CLOT_DEBOND at
all. The measure is the wall-tangential skeleton traction
`G * |dUs_x/dn|` on the wall-adjacent cell row, which is the same quantity that
was measured by hand on run 20260809-113416 (778 Pa) and run 20260810-015336
(the 141.3-152.8 Pa plateau). It is a small-strain estimator, so the RATIO is
what it gates on; the peak LOCATION is reported only, and the magnitude is
never quoted as an absolute stress. Run 20260810-035337 put a number on that
last caveat: the estimator gave 1.76 Pa where the wall BC measured 2.65.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path


TIME_LINE = re.compile(r"(?m)^Time = ([0-9.eE+-]+)\s*s?\s*$")
FSI_LINE = re.compile(
    r"PHASE_CLOT_FSI time=(\S+) iterations=(\d+) residual0=(\S+) "
    r"residual=(\S+) contraction=(\S+) omega=(\S+) converged=(\S+)"
)
DEBOND_LINE = re.compile(
    r"PHASE_CLOT_DEBOND time=(\S+) newFaces=(\d+) debondedFaces=(\d+) "
    r"bondedClotFaces=(\d+) wallFaces=(\d+) debondedArea=(\S+) "
    r"peakTraction=(\S+) peakTractionAny=(\S+) strength=(\S+) utilisation=(\S+)"
)
DIAGNOSTIC_LINE = re.compile(
    r"PHASE_CLOT_DIAGNOSTIC time=(\S+) .*?clotVolume=(\S+)"
)
DETACHMENT_LINE = re.compile(r"PHASE_CLOT_DETACHMENT .*?cells=(\d+)")
COMPILE_FAILURE = re.compile(
    r"(?m)\berror:|^make(?:\[\d+\])?: \*\*\* |^\s*Error [0-9]"
)
COMPLETED = ("End\n", "ExecutionTime =")


def find_field(d: Path, name: str) -> Path | None:
    exact = d / name
    if exact.is_file():
        return exact
    cands = [
        p for p in d.iterdir()
        if p.is_file() and p.name.startswith(name + ".") and p.name.count(".") == 1
    ]
    return cands[0] if len(cands) == 1 else None


def read_scalars(path: Path) -> list[float]:
    t = path.read_text(errors="replace")
    m = re.search(
        r"(?ms)internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;", t
    )
    if m:
        return [float(v) for v in m.group(2).split()]
    u = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", t)
    if u:
        return [float(u.group(1))]
    raise ValueError(f"{path}: cannot parse scalars")


def read_vectors(path: Path) -> list[tuple[float, float, float]]:
    t = path.read_text(errors="replace")
    m = re.search(
        r"(?ms)internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;", t
    )
    if m:
        return [tuple(float(x) for x in r.split())
                for r in re.findall(r"\(([^()]+)\)", m.group(2))]
    u = re.search(r"internalField\s+uniform\s+\(([^)]+)\)\s*;", t)
    if u:
        return [tuple(float(x) for x in u.group(1).split())]
    raise ValueError(f"{path}: cannot parse vectors")


def time_dirs(case: Path) -> list[tuple[float, Path]]:
    out = []
    for p in case.iterdir():
        if p.is_dir():
            try:
                out.append((float(p.name), p))
            except ValueError:
                pass
    return sorted(out)


def wall_traction_profile(case: Path, spec: dict) -> dict:
    """Tangential skeleton traction on each bottom-wall cell, clot-covered only."""
    out = {"available": False, "reason": None}
    dirs = time_dirs(case)
    if not dirs:
        out["reason"] = "no time directories"
        return out
    zero = case / "0"
    final = dirs[-1][1]
    need = {"Ccx": zero, "Ccy": zero, "Us": final, "c": final}
    paths = {}
    for name, d in need.items():
        p = find_field(d, name)
        if p is None:
            out["reason"] = f"missing {name} in {d.name}/"
            return out
        paths[name] = p

    x = read_scalars(paths["Ccx"])
    y = read_scalars(paths["Ccy"])
    us = read_vectors(paths["Us"])
    c = read_scalars(paths["c"])
    if not (len(x) == len(y) == len(us) == len(c)):
        out["reason"] = "field lengths differ"
        return out

    g = spec["geometry"]
    G = spec["material"]["G_pa"]
    y_wall = min(y)                      # wall-adjacent cell-centre height
    dn = y_wall                          # distance from the wall to that centre
    row = [i for i in range(len(y)) if math.isclose(y[i], y_wall, rel_tol=1e-9)]
    row.sort(key=lambda i: x[i])
    clot = [i for i in row if c[i] < 0.0]
    if len(clot) < 3:
        out["reason"] = f"only {len(clot)} clot-covered wall cells"
        return out

    # BOTH components, as of v2. v1 computed only the shear one and then
    # applied a location requirement -- "the peak must sit upstream" -- whose
    # justification was about the NORMAL peel stress. Two different quantities;
    # the check could not have tested what it claimed to. The normal component
    # is reported alongside so the peel mode can be read off the profile rather
    # than assumed.
    shear = [G * abs(us[i][0]) / dn for i in clot]
    normal = [G * abs(us[i][1]) / dn for i in clot]
    traction = shear
    xs = [x[i] for i in clot]
    peak_index = max(range(len(traction)), key=lambda k: traction[k])
    span = xs[-1] - xs[0]
    peak_fraction = ((xs[peak_index] - xs[0]) / span) if span > 0 else 0.0
    lo = min(traction)

    out.update({
        "available": True,
        "time_s": dirs[-1][0],
        "clot_covered_wall_cells": len(clot),
        "traction_min_pa": lo,
        "traction_max_pa": max(traction),
        "max_over_min": (max(traction) / lo) if lo > 0 else None,
        "peak_x_m": xs[peak_index],
        "peak_position_fraction": peak_fraction,
        "normal_min_pa": min(normal),
        "normal_max_pa": max(normal),
        "normal_peak_x_m": xs[max(range(len(normal)), key=lambda k: normal[k])],
        "profile": [
            {"x_m": xi, "traction_pa": ti, "normal_pa": ni}
            for xi, ti, ni in zip(xs, traction, normal)
        ],
        "estimator": (
            "shear G*|dUs_x/dn| and normal G*|dUs_y/dn| on the wall-adjacent "
            "row. Used for the RATIO, which is gated, and for the peak "
            "LOCATION, which as of v2 is reported only. The magnitude is an "
            "estimate: the authoritative traction is peakTraction from the "
            "wall BC, which measured 2.65 Pa where this estimator gives 1.76."
        ),
    })
    return out


def assess_child(case: Path, spec: dict, controls: dict) -> dict:
    log = case / "log.phaseClotAxisymmetricFluid"
    text = log.read_text(errors="replace") if log.is_file() else ""
    exit_path = case / "run-exit-code.txt"
    exit_code = int(exit_path.read_text().strip()) if exit_path.is_file() else None

    times = [float(v) for v in TIME_LINE.findall(text)]
    sweeps = [
        {"iterations": int(m[1]), "residual": float(m[3]),
         "converged": m[6].strip().lower() == "true"}
        for m in FSI_LINE.findall(text)
    ]
    debonds = [
        {"time": float(m[0]), "new_faces": int(m[1]), "debonded_faces": int(m[2]),
         "bonded_clot_faces": int(m[3]), "peak_traction_pa": float(m[6]),
         "strength_pa": float(m[8]), "utilisation": float(m[9])}
        for m in DEBOND_LINE.findall(text)
    ]
    volumes = [
        (float(m[0]), float(m[1])) for m in DIAGNOSTIC_LINE.findall(text)
    ]

    completed = bool(
        exit_code == 0 and any(k in text for k in COMPLETED) and times
        and math.isclose(max(times), spec["end_time_s"], rel_tol=1e-6)
    )

    health = controls["exchange_health"]
    settled = sweeps[int(len(sweeps) * health["settled_fraction_discarded"]):]
    unconverged = sum(1 for s in settled if not s["converged"])

    r = {
        "child_id": spec["child_id"], "role": spec["role"],
        "expected": spec["expected"],
        "wall_condition": spec["wall_condition"],
        "adhesive_strength_pa": spec["adhesive_strength_pa"],
        "growth": spec["growth"], "alpha_m2_s": spec["alpha_m2_s"],
        "exit_code": exit_code, "step_count": len(times),
        "final_log_time_s": max(times) if times else None,
        "completed": completed, "ran": len(times) > 0,
        "debond_lines": len(debonds),
        "debond_history": debonds,
        "final_debonded_faces": debonds[-1]["debonded_faces"] if debonds else 0,
        "total_new_faces": sum(d["new_faces"] for d in debonds),
        "steps_with_failures": sum(1 for d in debonds if d["new_faces"] > 0),
        "largest_single_step_loss": max((d["new_faces"] for d in debonds), default=0),
        "cohesive_release_cells": sum(int(n) for n in DETACHMENT_LINE.findall(text)),
        "exchange": {
            "steps": len(sweeps), "settled_steps": len(settled),
            "settled_unconverged": unconverged,
            "settled_unconverged_fraction": (
                unconverged / len(settled) if settled else None),
        },
    }
    r["exchange_healthy"] = bool(
        settled and unconverged / len(settled)
        <= health["maximum_unconverged_settled_fraction"]
    )

    dirs = time_dirs(case)
    c_min = c_max = None
    if dirs:
        p = find_field(dirs[-1][1], "c")
        if p is not None:
            v = read_scalars(p)
            c_min, c_max = min(v), max(v)
    lo, hi = controls["bounds"]["phase"]
    r["c_min"], r["c_max"] = c_min, c_max
    r["bounded"] = bool(c_min is not None and lo <= c_min and c_max <= hi)

    # Growth rate, from the clot volume the solver prints every step.
    r["growth_measurement"] = {"available": False}
    if spec["growth"] and len(volumes) >= 4:
        t0, v0 = volumes[0]
        t1, v1 = volumes[-1]
        if t1 > t0 and v0 > 0:
            # A one-sided cap grows mainly in height over a patch of fixed
            # length, so dV/dt ~ Vn * (patch length * depth).
            g = spec["geometry"]
            face_area = g["injury_length_m"] * g["depth_m"]
            vn = (v1 - v0) / (t1 - t0) / face_area
            target = controls["growth_target"]["Vn_target_m_s"]
            r["growth_measurement"] = {
                "available": True,
                "volume_initial_m3": v0, "volume_final_m3": v1,
                "window_s": t1 - t0,
                "dVdt_m3_s": (v1 - v0) / (t1 - t0),
                "growing_face_area_m2": face_area,
                "Vn_measured_m_s": vn,
                "Vn_target_m_s": target,
                "ratio_measured_over_target": (vn / target) if target else None,
                "alpha_trial_m2_s": spec["alpha_m2_s"],
                "alpha_corrected_m2_s": (
                    spec["alpha_m2_s"] * target / vn if vn > 0 else None),
                "note": "alpha_corrected is what the MAIN case should use.",
            }
    r["_traction"] = wall_traction_profile(case, spec)
    return r


def main() -> int:
    gate = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    controls = json.loads(
        (gate / "constant" / "muralGeometryControls.json").read_text())
    runs = gate / "runs"
    matrix = json.loads((runs / "matrix.json").read_text())
    children = {
        s["child_id"]: assess_child(runs / s["child_id"], s, controls)
        for s in matrix
    }

    build_log = gate / "log.Allwmake"
    build_text = build_log.read_text(errors="replace") if build_log.is_file() else ""
    library_compiled = bool(build_text) and not COMPILE_FAILURE.search(build_text)

    ran = all(c["ran"] for c in children.values())

    # --- the decisive measurement -------------------------------------------
    gc = controls["gradient_check"]
    tr = children["mural_traction"]["_traction"]
    gradient = dict(tr)
    gradient["preconditions_met"] = bool(
        children["mural_traction"]["ran"]
        and children["mural_traction"]["completed"]
        and tr.get("available")
    )
    gradient["required_max_over_min"] = gc["minimum_max_over_min"]
    gradient["reference_occlusive_plateau_ratio"] = gc[
        "reference_occlusive_plateau_ratio"]
    gradient["gradient_present"] = bool(
        gradient["preconditions_met"]
        and tr.get("max_over_min") is not None
        and tr["max_over_min"] >= gc["minimum_max_over_min"]
    )
    gradient["peak_upstream"] = bool(
        gradient["preconditions_met"]
        and tr.get("peak_position_fraction") is not None
        and tr["peak_position_fraction"] <= gc["peak_must_lie_in_upstream_fraction"]
    )
    gradient.pop("profile", None)
    gradient["profile"] = tr.get("profile", [])

    # --- progressiveness -----------------------------------------------------
    pc = controls["peel_check"]
    peel = {}
    for cid in ("mural_peel_lo", "mural_peel_hi"):
        child = children[cid]
        order_ok = True
        seen = [d for d in child["debond_history"] if d["new_faces"] > 0]
        peel[cid] = {
            "strength_pa": child["adhesive_strength_pa"],
            "steps_with_failures": child["steps_with_failures"],
            "total_faces_lost": child["total_new_faces"],
            "largest_single_step_loss": child["largest_single_step_loss"],
            "progressive": bool(
                child["total_new_faces"] > 0
                and child["steps_with_failures"] >= pc["minimum_steps_with_failures"]
            ),
            "all_at_once": bool(
                child["total_new_faces"] > 0 and child["steps_with_failures"] == 1
            ),
            "never_fired": child["total_new_faces"] == 0,
        }
    any_progressive = any(v["progressive"] for v in peel.values())
    # v1 collapsed "nothing debonded at all" into PEEL_NOT_PROGRESSIVE, which
    # reads as a statement about HOW the bond fails when in fact nothing had
    # failed. Both peel children sat above the 2.65 Pa the bond actually
    # carried. The two diagnoses send you to different places -- one to the
    # strengths, one to the timestep -- so they get different verdicts.
    nothing_fired = all(v["never_fired"] for v in peel.values())

    # --- growth calibration --------------------------------------------------
    gm = children["mural_growth"]["growth_measurement"]
    factor = controls["growth_target"]["acceptance_factor"]
    growth_ok = bool(
        gm.get("available")
        and gm.get("ratio_measured_over_target") is not None
        and (1.0 / factor) <= gm["ratio_measured_over_target"] <= factor
    )

    checks = {
        "library_compiled": library_compiled,
        "children_ran": ran,
        "all_completed": all(c["completed"] for c in children.values()),
        "all_bounded": all(c["bounded"] for c in children.values()),
        "traction_gradient_present": gradient["gradient_present"],
        "peel_fired": not nothing_fired,
        "peel_progressive": any_progressive,
        "growth_rate_on_target": growth_ok,
        "no_cohesive_release": all(
            c["cohesive_release_cells"] == 0 for c in children.values()),
        "traction_child_silent": children["mural_traction"]["debond_lines"] == 0,
    }

    if not library_compiled:
        status, verdict = "INVALID", "INVALID"
    elif not ran:
        status, verdict = "VALID", "SETUP_FAILED"
    elif not (checks["all_completed"] and checks["all_bounded"]):
        status, verdict = "VALID", "INADMISSIBLE"
    elif not checks["traction_gradient_present"]:
        # The real negative. Ranked above everything else that follows,
        # because without a gradient the peel and growth results describe a
        # geometry that does not do what it was chosen to do.
        status, verdict = "VALID", "NO_TRACTION_GRADIENT"
    elif not checks["peel_fired"]:
        status, verdict = "VALID", "PEEL_NEVER_FIRED"
    elif not checks["peel_progressive"]:
        status, verdict = "VALID", "PEEL_NOT_PROGRESSIVE"
    elif not checks["growth_rate_on_target"]:
        status, verdict = "VALID", "GROWTH_RATE_OFF"
    elif all(checks.values()):
        status, verdict = "VALID", "MURAL_READY"
    else:
        status, verdict = "VALID", "INADMISSIBLE"

    for c in children.values():
        c.pop("_traction", None)

    result = {
        "probe": "AM2020_mural_geometry_gate",
        "question": controls["_question"],
        "status": status, "verdict": verdict,
        "checks": checks,
        "traction_gradient": gradient,
        "peel": peel,
        "growth_calibration": gm,
        "children": children,
        "not_a_physics_result": controls["authorization_boundary"],
    }
    out = gate / "postProcessing" / "muralGeometryGate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if verdict == "MURAL_READY" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"assess_gate.py: {error}")
