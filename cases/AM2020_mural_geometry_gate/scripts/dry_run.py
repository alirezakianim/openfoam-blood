#!/usr/bin/env python3
"""Drive assess_gate.py end to end on synthesised OpenFOAM-shaped output.

All log formats are the REAL ones, copied from the solver's Info<< statements:

  Time = 0.02s                                       v13 carries the unit
  PHASE_CLOT_FSI time=... iterations=... residual0=... residual=...
                 contraction=... omega=... converged=...
  PHASE_CLOT_DEBOND time=... newFaces=... debondedFaces=... bondedClotFaces=...
                 wallFaces=... debondedArea=... peakTraction=...
                 peakTractionAny=... strength=... utilisation=...
  PHASE_CLOT_DIAGNOSTIC time=... inventory=... surfaceLoss=... outerFlux=...
                 balanceResidual=... clotVolume=... outletMinUx=...
                 outletMeanUx=... outletCells=...

The synthesised Us field is built so the traction profile it implies is a
CHOSEN shape: `gradient_ratio` and `peak_fraction` are dialled directly, which
is what lets the decisive check be exercised in both directions offline.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


GATE = Path(__file__).resolve().parents[1]
BOUNDARY = (
    "    inlet { type zeroGradient; }\n"
    "    outlet { type zeroGradient; }\n"
    "    bottom { type zeroGradient; }\n"
    "    top { type zeroGradient; }\n"
    "    front { type empty; }\n"
    "    back { type empty; }\n"
)


def write_scalar(path: Path, values: list[float], cls: str = "volScalarField") -> None:
    body = "\n".join(f"{v:.10g}" for v in values)
    path.write_text(
        f"FoamFile\n{{\n    format ascii;\n    class {cls};\n"
        f"    object {path.name};\n}}\n\ndimensions [0 0 0 0 0 0 0];\n"
        f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;\n"
        f"boundaryField\n{{\n{BOUNDARY}}}\n"
    )


def write_vector(path: Path, values: list[tuple]) -> None:
    body = "\n".join(f"({x:.10g} {y:.10g} {z:.10g})" for x, y, z in values)
    path.write_text(
        "FoamFile\n{\n    format ascii;\n    class volVectorField;\n"
        f"    object {path.name};\n}}\n\ndimensions [0 1 0 0 0 0 0];\n"
        f"internalField nonuniform List<vector>\n{len(values)}\n(\n{body}\n)\n;\n"
        f"boundaryField\n{{\n{BOUNDARY}}}\n"
    )


def geometry(controls: dict):
    g = controls["geometry"]
    nx, ny = g["nx"], g["ny"]
    L, H = g["channel_length_m"], g["channel_height_m"]
    eps = g["interface_width_m"]
    x0, half = g["injury_centre_x_m"], 0.5 * g["injury_length_m"]
    hc = controls["clot"]["height_m"]
    xs, ys, cs = [], [], []
    for j in range(ny):
        y = (j + 0.5) * H / ny
        for i in range(nx):
            x = (i + 0.5) * L / nx
            xs.append(x)
            ys.append(y)
            cs.append(math.tanh(max(abs(x - x0) - half, y - hc) / eps))
    return g, xs, ys, cs


def us_field(controls, xs, ys, cs, gradient_ratio: float, peak_fraction: float,
             peak_traction: float):
    """Displacement whose implied wall traction G*|dUs_x/dn| has the requested
    peak value, max/min ratio and peak position."""
    G = controls["material"]["G_pa"]
    y_wall = min(ys)
    clot_x = sorted(x for x, y, c in zip(xs, ys, cs) if y == y_wall and c < 0)
    x_lo, x_hi = clot_x[0], clot_x[-1]
    span = x_hi - x_lo
    x_peak = x_lo + peak_fraction * span

    def shape(x):
        if span <= 0:
            return 1.0
        # Triangular ramp peaking at x_peak, falling to 1/gradient_ratio at the
        # far end, so max/min is exactly gradient_ratio.
        far = max(abs(x_lo - x_peak), abs(x_hi - x_peak))
        if far <= 0:
            return 1.0
        d = abs(x - x_peak) / far
        return 1.0 - d * (1.0 - 1.0 / gradient_ratio)

    out = []
    for x, y, c in zip(xs, ys, cs):
        if c < 0 and y == y_wall:
            ux = peak_traction * shape(x) * y_wall / G
        else:
            ux = 1e-9
        out.append((ux, 0.0, 0.0))
    return out


def synthesise_child(case: Path, controls: dict, spec: dict,
                     gradient_ratio=6.0, peak_fraction=0.15,
                     peak_traction=7.9, debond_schedule=None,
                     growth_ratio=1.0, exit_code=0,
                     unconverged_fraction=0.0, cohesive_cells=0,
                     omit_fields=False, spurious_debond=False):
    t = controls["time"][spec["phase"]]
    dt, end, wi = t["delta_t_s"], t["end_time_s"], t["write_interval_s"]
    n_steps = int(round(end / dt))
    g, xs, ys, cs = geometry(controls)
    n = len(xs)

    face_area = g["injury_length_m"] * g["depth_m"]
    v0 = 1.2e-14
    vn = controls["growth_target"]["Vn_target_m_s"] * growth_ratio

    lines, debonded = [], 0
    for k in range(1, n_steps + 1):
        tt = k * dt
        conv = (k / n_steps) > unconverged_fraction
        lines.append(f"Time = {tt:.10g}s\n")
        lines.append(
            f"PHASE_CLOT_FSI time={tt:.10g} iterations={3 if conv else 60} "
            f"residual0=9.8e-04 residual={1e-6 if conv else 3.2e-3:.6g} "
            f"contraction=0.37 omega=0.5 converged={'true' if conv else 'false'}\n"
        )
        if spec["growth"]:
            vol = v0 + vn * face_area * tt
            lines.append(
                f"PHASE_CLOT_DIAGNOSTIC time={tt:.10g} inventory=1.0 "
                f"surfaceLoss=0 outerFlux=0 balanceResidual=1e-12 "
                f"clotVolume={vol:.10g} outletMinUx=0.001 outletMeanUx=0.002 "
                f"outletCells=40\n"
            )
        emits = spec["wall_condition"] == "clotWallAdhesion" or spurious_debond
        if emits:
            # Padded rather than indexed blind: v2 doubled the run to 40 steps
            # and every hand-written schedule was 20 long.
            new = (
                debond_schedule[k - 1]
                if debond_schedule and k - 1 < len(debond_schedule)
                else 0
            )
            debonded += new
            bonded = max(0, 60 - debonded)
            strength = spec["adhesive_strength_pa"] or 1.0
            peak = peak_traction if bonded else 0.0
            lines.append(
                f"PHASE_CLOT_DEBOND time={tt:.10g} newFaces={new} "
                f"debondedFaces={debonded} bondedClotFaces={bonded} "
                f"wallFaces={g['nx']} debondedArea={debonded*2.25e-12:.6g} "
                f"peakTraction={peak:.6g} peakTractionAny={peak_traction:.6g} "
                f"strength={strength:.6g} "
                f"utilisation={(peak/strength if strength else 0):.6g}\n"
            )
        if cohesive_cells and k == n_steps:
            lines.append(
                f"PHASE_CLOT_DETACHMENT time={tt:.10g} cells={cohesive_cells} "
                "volume=3.9e-18 maxDamage=0.97\n"
            )
    lines.append("ExecutionTime = 100 s  ClockTime = 102 s\n\nEnd\n"
                 if exit_code == 0 else "Foam::sigFpe::sigHandler(int)\n")
    (case / "log.phaseClotAxisymmetricFluid").write_text("".join(lines))
    (case / "run-exit-code.txt").write_text(f"{exit_code}\n")
    (case / "log.wallTime").write_text("WALL_SECONDS=102\n")

    zero = case / "0"
    zero.mkdir(parents=True, exist_ok=True)
    if not omit_fields:
        write_scalar(zero / "Ccx", xs)
        write_scalar(zero / "Ccy", ys)
    us = us_field(controls, xs, ys, cs, gradient_ratio, peak_fraction, peak_traction)
    value = wi
    while value <= end + 1e-12:
        d = case / f"{value:.10g}"
        d.mkdir(parents=True, exist_ok=True)
        write_scalar(d / "c", cs)
        write_vector(d / "Us", us)
        value = round(value + wi, 10)


def build_gate(tmp: Path) -> Path:
    cases = tmp / "cases"
    cases.mkdir()
    gate = cases / GATE.name
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "runs", "postProcessing", "log.*")
    shutil.copytree(GATE, gate, ignore=ignore)
    template = GATE.parent / "AM2020_straight_tube_growth_mechanics_smoke"
    shutil.copytree(template, cases / template.name, ignore=ignore)
    subprocess.run([sys.executable, str(gate / "scripts" / "create_cases.py")],
                   check=True, stdout=subprocess.DEVNULL)
    return gate


def run_assessor(gate: Path):
    done = subprocess.run(
        [sys.executable, str(gate / "scripts" / "assess_gate.py"), str(gate)],
        text=True, capture_output=True)
    p = gate / "postProcessing" / "muralGeometryGate" / "result.json"
    if not p.is_file():
        raise AssertionError(f"no result.json\n{done.stdout}\n{done.stderr}")
    return done.returncode, json.loads(p.read_text())


GOOD_BUILD = "wmake libso phaseClotAxisymmetricFluid\n'lib.so' is up to date.\n"
BAD_BUILD = ("mural.C:1:1: error: bad\nmake: *** [x.o] Error 1\n")

HEALTHY = {
    "mural_traction": dict(),
    "mural_peel_lo": dict(debond_schedule=[0, 0, 3, 6, 8, 7, 5, 4, 2, 1]
                          + [0] * 10),
    "mural_peel_hi": dict(debond_schedule=[0] * 6 + [2, 4, 5, 3] + [0] * 10),
    "mural_growth": dict(growth_ratio=1.0),
}


def variant(**over):
    m = {k: dict(v) for k, v in HEALTHY.items()}
    for cid, ch in over.items():
        m[cid].update(ch)
    return m


def exercise(name, per_child, expected_status, expected_verdict,
             expected_failed=None, build_log=GOOD_BUILD, setup_failed=False):
    with tempfile.TemporaryDirectory(prefix="am2020-mural-") as tmp:
        gate = build_gate(Path(tmp))
        (gate / "log.Allwmake").write_text(build_log)
        controls = json.loads(
            (gate / "constant" / "muralGeometryControls.json").read_text())
        specs = {c["child_id"]: c for c in controls["children"]}
        for cid, kwargs in per_child.items():
            case = gate / "runs" / cid
            if setup_failed:
                (case / "run-exit-code.txt").write_text("1\n")
                continue
            synthesise_child(case, controls, specs[cid], **kwargs)
        code, result = run_assessor(gate)
        failed = [k for k, v in result["checks"].items() if not v]
        if (result["status"] != expected_status
                or result["verdict"] != expected_verdict):
            raise AssertionError(
                f"{name}: expected {expected_status}/{expected_verdict}, got "
                f"{result['status']}/{result['verdict']} failed={failed}\n"
                f"  gradient={ {k: v for k, v in result['traction_gradient'].items() if k != 'profile'} }\n"
                f"  peel={result['peel']}\n"
                f"  growth={result['growth_calibration']}")
        if expected_failed is not None and expected_failed not in failed:
            raise AssertionError(f"{name}: expected {expected_failed}, got {failed}")
        if expected_verdict == "MURAL_READY" and code != 0:
            raise AssertionError(f"{name}: MURAL_READY must exit 0")
        print(f"{name:<30} -> {result['status']}/{result['verdict']}")


def main() -> int:
    exercise("ready", variant(), "VALID", "MURAL_READY")

    exercise("compile_failed", variant(), "INVALID", "INVALID",
             "library_compiled", build_log=BAD_BUILD)
    exercise("setup_failed", variant(), "VALID", "SETUP_FAILED",
             "children_ran", setup_failed=True)

    # THE real negative: the mural clot loads its bond as flatly as the
    # occlusive plug did (that plateau measured 1.08).
    exercise("no_gradient",
             variant(mural_traction=dict(gradient_ratio=1.08),
                     mural_peel_lo=dict(gradient_ratio=1.08,
                                        debond_schedule=[0, 60] + [0] * 18),
                     mural_peel_hi=dict(gradient_ratio=1.08),
                     mural_growth=dict(gradient_ratio=1.08)),
             "VALID", "NO_TRACTION_GRADIENT", "traction_gradient_present")
    # A DOWNSTREAM PEAK IS NOT A FAILURE, as of v2. Run 20260810-035337
    # measured it at 0.93 and v1 called that INADMISSIBLE on the strength of my
    # "levering off starts upstream" reasoning -- which was about the NORMAL
    # peel stress, while the estimator computes the SHEAR component. The
    # location is reported now, not gated. This scenario pins that: re-promote
    # the location check and it fails.
    exercise("peak_downstream_is_reported_not_failed",
             variant(mural_traction=dict(peak_fraction=0.9)),
             "VALID", "MURAL_READY")
    # Missing cell centres must NOT pass vacuously.
    exercise("traction_unmeasurable",
             variant(mural_traction=dict(omit_fields=True)),
             "VALID", "NO_TRACTION_GRADIENT", "traction_gradient_present")

    # THE ACTUAL OUTCOME OF RUN 20260810-035337: neither peel child debonded a
    # single face, because both strengths sat above the 2.65 Pa the bond
    # carried. v1 reported that as PEEL_NOT_PROGRESSIVE, which describes HOW a
    # bond failed and sends the reader to the timestep. Nothing had failed, and
    # the fix was the strengths. Different diagnosis, different verdict.
    exercise("peel_never_fired",
             variant(mural_peel_lo=dict(debond_schedule=None),
                     mural_peel_hi=dict(debond_schedule=None)),
             "VALID", "PEEL_NEVER_FIRED", "peel_fired")

    # Gradient fine, but both peel children still go all at once.
    exercise("peel_all_at_once",
             variant(mural_peel_lo=dict(debond_schedule=[0, 60] + [0] * 18),
                     mural_peel_hi=dict(debond_schedule=[0, 0, 60] + [0] * 17)),
             "VALID", "PEEL_NOT_PROGRESSIVE", "peel_progressive")
    # One progressive child is enough; the other may legitimately never fire
    # if the strength estimate was high.
    exercise("hi_never_fires",
             variant(mural_peel_hi=dict(debond_schedule=[0] * 20)),
             "VALID", "MURAL_READY")

    exercise("growth_too_fast", variant(mural_growth=dict(growth_ratio=40.0)),
             "VALID", "GROWTH_RATE_OFF", "growth_rate_on_target")
    exercise("growth_too_slow", variant(mural_growth=dict(growth_ratio=0.01)),
             "VALID", "GROWTH_RATE_OFF", "growth_rate_on_target")

    # THE FSI EXCHANGE IS REPORTED, NOT GATED, as of v2. It is a standing
    # project-wide limitation -- the coupling floor already carried as a stated
    # caveat -- not something this gate is testing, and gating on it would let
    # a known condition suppress the geometry answer the gate exists to give.
    # The numbers stay in the report. This scenario pins the decision.
    exercise("unconverged_exchange_is_reported_not_gated",
             variant(mural_peel_lo=dict(
                 debond_schedule=[0, 0, 3, 6, 8, 7, 5, 4, 2, 1] + [0] * 10,
                 unconverged_fraction=1.0),
                 mural_traction=dict(unconverged_fraction=1.0),
                 mural_peel_hi=dict(unconverged_fraction=1.0),
                 mural_growth=dict(unconverged_fraction=1.0)),
             "VALID", "MURAL_READY")
    exercise("child_aborted", variant(mural_growth=dict(exit_code=136)),
             "VALID", "INADMISSIBLE", "all_completed")
    # Gc is unreachable here, so any release means the isolation broke.
    exercise("cohesive_leak", variant(mural_peel_lo=dict(
                 debond_schedule=[0, 0, 3, 6, 8, 7, 5, 4, 2, 1] + [0] * 10,
                 cohesive_cells=3)),
             "VALID", "INADMISSIBLE", "no_cohesive_release")
    # The welded child must print nothing: commitWallBonds must stay a no-op.
    exercise("traction_child_noisy",
             variant(mural_traction=dict(spurious_debond=True)),
             "VALID", "INADMISSIBLE", "traction_child_silent")

    print("All verdicts and isolated invalid paths passed offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
