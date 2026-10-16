#!/usr/bin/env python3
"""Drive assess_case.py end to end on synthesised OpenFOAM-shaped output.

The log formats are the REAL ones, taken from the returned logs of run
20260810-192247, including the PHASE_CLOT_DIAGNOSTIC line's full field list --
an earlier draft of the assessor matched a `maxShearRate=` line that the solver
does not print, and would have reported None forever while the control file
claimed the enhancement threshold was checked.

A results script has a failure mode a gate does not: it can quietly average an
inadmissible child into a table, or report a number computed from a field that
was never written. Most of the scenarios below are aimed at that.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CASE = Path(__file__).resolve().parents[1]
N_CELLS = 6400
BOUNDARY = (
    "    inlet { type zeroGradient; }\n"
    "    outlet { type zeroGradient; }\n"
    "    bottom { type zeroGradient; }\n"
    "    top { type zeroGradient; }\n"
    "    front { type empty; }\n"
    "    back { type empty; }\n"
)
# Measured on this configuration, run 20260810-192247.
PEAK_TRACTION = 2.6456
V0 = 2.78885234631e-15
VN_AT_200 = 3.7577e-08
# The MEASURED skeleton displacement at 200/s, run 20260811-004647. The
# fixture scales it linearly with shear so the deformation table has a real
# trend to report rather than a constant.
US_MAX_AT_200 = 1.06e-07
# Must match the `cap` the assessor tests skelIters against.
SKEL_CAP = 20
BONDED_FACES = 60
# The shear rate the three demonstration regimes (full release / arrest /
# stall) were measured at, in runs 20260811-004647 and 20260812-062701.
REFERENCE_SHEAR_1_S = 200.0


# WHY THESE TWO FUNCTIONS ARE CACHED.
#
# The harness runs 46 scenarios over 136 synthesised children, and each child
# writes several fields of 31360 cells. That is tens of millions of float
# repr() calls, and a `sample` of a running build showed the process sitting in
# float_repr under PyUnicode_Join for most of its life -- the dry run had grown
# to ~70 minutes of a ~2 hour package build, almost all of it re-formatting the
# SAME numbers.
#
# The bodies are memoised on the exact value sequence, so the output is
# byte-identical by construction: a cache hit returns the string the join would
# have produced. The uniform fast path is the common case (a synthesised field
# is usually one value everywhere) and turns n repr() calls into one.
_BODY_CACHE: dict = {}
_BODY_CACHE_MAX = 64


def _cached_body(key, build):
    hit = _BODY_CACHE.get(key)
    if hit is None:
        if len(_BODY_CACHE) >= _BODY_CACHE_MAX:
            _BODY_CACHE.clear()
        hit = _BODY_CACHE[key] = build()
    return hit


def write_scalar(path: Path, values: list[float]) -> None:
    if values and min(values) == max(values):
        body = _cached_body(
            ("s", len(values), values[0]),
            lambda: "\n".join([repr(values[0])] * len(values)),
        )
    else:
        body = _cached_body(("S", tuple(values)),
                            lambda: "\n".join(repr(v) for v in values))
    path.write_text(
        "FoamFile\n{\n    format ascii;\n    class volScalarField;\n"
        f"    object {path.name};\n}}\n\n"
        "dimensions [0 0 0 0 0 0 0];\n"
        f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;\n"
        f"boundaryField\n{{\n{BOUNDARY}}}\n"
    )


def write_vector(path: Path, values: list[tuple]) -> None:
    if values and len(set(values)) == 1:
        x, y, z = values[0]
        body = _cached_body(
            ("v", len(values), values[0]),
            lambda: "\n".join([f"({x!r} {y!r} {z!r})"] * len(values)),
        )
    else:
        body = _cached_body(
            ("V", tuple(values)),
            lambda: "\n".join(f"({a!r} {b!r} {c!r})" for a, b, c in values),
        )
    path.write_text(
        "FoamFile\n{\n    format ascii;\n    class volVectorField;\n"
        f"    object {path.name};\n}}\n\n"
        "dimensions [0 1 0 0 0 0 0];\n"
        f"internalField nonuniform List<vector>\n{len(values)}\n(\n{body}\n)\n;\n"
        f"boundaryField\n{{\n{BOUNDARY}}}\n"
    )


def synthesise_y_child(case: Path, spec: dict, exit_code: int = 0, short: bool = False, **options) -> None:
    """The Y-bifurcation gate's returned child (BATCH 79): the log, the exit code and the wall time, then the fields scripts/y_gate/y_synth.py lays on the replica's
    real cell and face positions (phi, p, U, be, q, thetaNetRef at the 18 runTime writes k tau1; the logged inletFlux / outletFlux; gate1.json). The same fixture whatever
    the phase-field convention (`theta_native` is the channel clot's): the Y report reads q, be and thetaNetRef only. The log runs to the step that passes endTime - dt/2
    (Time::run), as the solver's does without an adjustable write time. `short` stops it at a third of the steps (the INCOMPLETE negative)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "y_gate"))
    import y_synth
    g6 = spec["y_gate"]["gate6"]
    dt, end = g6["dt_est_s"], spec["end_time_s"]
    n_steps = int(math.ceil((end - 0.5 * dt) / dt - 1e-9))              # Time::run() is true while t < endTime - dt/2: the step that first reaches it is the last
    if short:
        n_steps = max(1, n_steps // 3)
    lines = [f"Time = {k * dt:.10g}s\n" for k in range(1, n_steps + 1)]
    lines.append("ExecutionTime = 500 s  ClockTime = 510 s\n\nEnd\n" if exit_code == 0 else "Foam::sigFpe::sigHandler(int) at ??:?\n")
    (case / "log.phaseClotAxisymmetricFluid").write_text("".join(lines))
    (case / "run-exit-code.txt").write_text(f"{exit_code}\n")
    (case / "log.wallTime").write_text("WALL_SECONDS=510\n")
    y_synth.build_case(case, spec["y_gate"], spec["plasma"]["nu_m2_s"], **options)


def synthesise_y_plug_child(case: Path, spec: dict, exit_code: int = 0, short: bool = False, **options) -> None:
    """A child of the symmetric Y's PLUG series (case-engineer 27b): the restart-test arm Y80_RT_A / Y80_RT_B (spec has stop_after_step) or the clot child Y81_*_PLUG. The plug
    spec's y_gate block has no gate6 (kind 'plug': mesh, geometry_sim, drive, plug), so it must NOT go to synthesise_y_child (KeyError). scripts/y_gate/y_rt_synth.py lays out
    the files: for the pair, an EXACT restart (A: full writes at steps leg1 - 4 .. leg2 and a log from step 1; B: the leg-2 case, restart.json, the step-leg2 directory and a log
    from step leg1 + 1), so assess_case.y_rt_pair PASSES it; for the clot child, a log to endTime - dt/2 and the initial c drawn by y_plug.py at the replica's cell centres.
    `options` plant ONE defect (y_rt_synth.build_rt_case: ulp=, dt_ulp=, late_leg1=, cold_b=, ...); `short` stops the log early (INCOMPLETE)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "y_gate"))
    import y_rt_synth
    if spec.get("stop_after_step") is None:
        y_rt_synth.build_plug_case(case, spec, exit_code=exit_code, short=short, **options)
        return
    controls = json.loads((Path(__file__).resolve().parents[1] / "constant" / "mainCaseControls.json").read_text())
    entry = next(e for e in controls["y_rt_pairs"] if spec["child_id"] in (e["a"], e["b"]))
    which = "A" if spec["child_id"] == entry["a"] else "B"
    y_rt_synth.build_rt_case(case, which, entry, **options)
    log = case / "log.phaseClotAxisymmetricFluid"
    if short:
        lines = log.read_text().split("\n")
        log.write_text("\n".join(lines[: len(lines) // 2]) + "\n")                          # no End, and the last step is below the stop
    elif exit_code != 0:
        log.write_text(log.read_text().replace("End\n", "Foam::sigFpe::sigHandler(int) at ??:?\n"))
    (case / "run-exit-code.txt").write_text(f"{exit_code}\n")


def synthesise_child(
    case: Path,
    spec: dict,
    peak_traction: float | None = None,
    vn: float | None = None,
    debond_schedule: list[int] | None = None,
    exit_code: int = 0,
    c_extreme: float = 0.999,
    dg_peak: float = 1e-9,
    cohesive_cells: int = 0,
    omit_fields: bool = False,
    nan_field: bool = False,
    # THETA-NATIVE OUTPUT, as the solver writes it after S6c-2 deleted `c`.
    # The written fields become q, thetaC and thetaH; `c` survives only in 0/,
    # because create_cases.py still writes an initial condition for it. That
    # asymmetry is the real post-deletion shape of a returned run and it is
    # what the assessor's per-case convention choice has to cope with:
    # reading 0/c against a reconstructed final would compare two conventions
    # whose difference is precisely the drift S6c-2 removes.
    theta_native: bool = False,
    short: bool = False,
    transport_shift_cells: int | None = 20,
    transport_exit_code: int = 0,
    transport_dissolves: bool = False,
    transport_smears_to: float | None = None,
    transport_erodes_cells: int = 0,
    skeleton_at_cap: bool = False,
    us_scale: float = 1.0,
    skeleton_diverges: bool = False,
    # The MEASURED ambient-band velocity from run 20260811-100050's field,
    # not the 4.06e-6 the broken adjacency shell reported.
    body_velocity: float = 1.21e-3,
    # Per-step growth of the ADHESIVE FOOTPRINT onto virgin wall. Zero is the
    # post-fix behaviour; a non-zero schedule reproduces run 20260812-062701's
    # ratchet so the assessor is tested against it.
    footprint_growth: list[int] | None = None,
    # How the run ENDS once the core has gone. None keeps the body in the
    # domain. "clean_exit" removes it and its material. "stranded" removes the
    # core but leaves the material behind -- run 20260812-200349, where the
    # clot reached the outlet and left 83% of itself as a frozen obstruction.
    final_state: str | None = None,
    # Extra ROWS added to the clot over the run, keeping its length. Run
    # 20260812-200349: a spurious Uy reaching 17% of Ux advected the body into
    # the no-slip wall it sits on, so it stretched instead of translating --
    # length pinned at 90 um, height 20 -> 34 um, core 336 -> 557 cells.
    transport_stretches_rows: int = 0,
    # THE COHESIVE TEAR. A band of cells whose damage has reached
    # DgDetach, at a given height above the wall in CELL ROWS. Row 0 is
    # the wall itself, which is the case tear_measurement must call out
    # as a bond failure wearing a tear's clothes rather than as the
    # through-body break Dr Li asked for.
    tear_rows: tuple[int, int] | None = None,
    tear_from_step: float = 0.4,
    # Peak fracture driving energy in the phase interface, as a MULTIPLE
    # of the bulk value. Run 20260814-055141 measured 13.5 -- the damage
    # was driven by the numerical Us pin, not by the material, and every
    # child eroded from the outside in as a result. Default is below 1,
    # i.e. the post-fix behaviour.
    interface_driver_ratio: float = 0.3,
    # STREAMWISE extent of the tear band, in cells. The default spans the
    # whole clot AND one column past it, which is right for the homogeneous
    # cohesive-tear fixtures but makes every tear touch fluid and therefore
    # read as depth 1. A band inside the body is what a core/shell tear looks
    # like, so its extent has to be settable.
    tear_cols: tuple[int, int] = (50, 111),
    # CORE/SHELL. Rows and columns of the buried core, or None for a single
    # material. The core is the region whose Gc is physiological, so nothing
    # in it may ever be damaged; a fixture that puts the tear inside it is how
    # the assessor's core_survived check gets tested.
    core_rows: tuple[int, int] | None = None,
    core_cols: tuple[int, int] = (60, 100),
    # Extra clot rows added over the run by GROWTH, as distinct from
    # transport_stretches_rows, which is the advection defect of run
    # 20260812-200349 and has its own scenario.
    growth_rows: int = 0,
    # Realised clot permeability, as the solver now reports it. `floored`
    # reproduces every run before 2026-08-14: kozenyCarman on the phase
    # fraction returns ~1e-22 inside the clot, the floor takes over, and k0p
    # has no effect at all.
    permeability_floored: bool = False,
    # Cells the SHELL is carried downstream over the run while the core stays
    # put. This is source 33's signature -- their core retains its shape at
    # every shear rate while the shell is pushed into an elongated filament --
    # and it is the thing the report now measures instead of asking whether a
    # damage variable crossed a threshold.
    shell_shift_cells: int = 0,
    # ... and the failure that looks like it from a distance: the WHOLE body
    # translating, core and all, which is exactly what Dr Li rejected.
    core_shift_cells: int = 0,
    # THE CONTENT-FCT LINE (S20 G3'lambda), one per step, in the solver's own format
    # (phaseClotAxisymmetricFluid.C, the PHASE_CLOT_CONTENT_FCT Info). None emits no
    # line (every other scenario); (lambdaFluxMean, zeroFluxRule) emits one, with
    # zeroFluxRule None giving the pre-S20 line (run 20261001-102832 xu5_g500 read
    # "activeFaces=56369 limitedFaces=10111 lambdaMin=0 lambdaFluxMean=0.99982924806
    # negAfter=0"). Step 1 is idle (activeFaces=0, lambdaFluxMean=1), as that log's was.
    fct: tuple | None = None,
    # BATCH 79 (the Y gate's child only): options for scripts/y_gate/y_synth.py build_case, each planting ONE defect (split=, q0_scale=, last_k=, content=, ...)
    y_options: dict | None = None,
) -> None:
    if spec.get("y_gate") and spec["y_gate"].get("kind") == "plug":
        return synthesise_y_plug_child(case, spec, exit_code=exit_code, short=short, **(y_options or {}))
    if spec.get("y_gate"):
        return synthesise_y_child(case, spec, exit_code=exit_code, short=short, **(y_options or {}))
    assert not y_options, f"{spec['child_id']}: y_options is read only by the Y child"
    dt = spec["delta_t_s"]

    nx_cells = spec["geometry"]["nx"]

    def in_core(idx: int) -> bool:
        if core_rows is None:
            return False
        row, col = idx // nx_cells, idx % nx_cells
        return (core_rows[0] <= row <= core_rows[1]
                and core_cols[0] <= col < core_cols[1])

    def damage_field(frac: float) -> list[float]:
        """Dg at a fraction `frac` through the run."""
        dg = [0.0, dg_peak * 0.5, dg_peak] + [0.0] * (N_CELLS - 3)
        if tear_rows is not None and frac >= tear_from_step:
            lo, hi = tear_rows
            for row in range(lo, hi + 1):
                for i in range(*tear_cols):
                    idx = row * nx_cells + i
                    if idx < N_CELLS:
                        dg[idx] = 0.99
        return dg

    # The last step lands ON end_time, as foamRun's adjustable step does (the seeded
    # child's 9.5238e-5 s is 19.05 steps of 5e-6 s: its 20th step is short).
    n_steps = int(math.ceil(spec["end_time_s"] / dt - 1e-9))
    # S25 B76-RT: a child with stop_after_step ends by the step-count stop (stopAfterStep), not at end_time_s (only its backstop). Its
    # log therefore runs exactly stop_after_step steps, each with the solver's PHASE_CLOT_DT record (what assess_case reads the stop
    # from), and `short` still cuts it below that stop, which is the negative: a log that stops early must not come out COMPLETE.
    stop_step = spec.get("stop_after_step")
    if stop_step is not None:
        n_steps = int(stop_step)
    if short:
        n_steps = max(1, n_steps // 3)
    area = spec["geometry"]["injury_length_m"] * spec["geometry"]["depth_m"]
    if peak_traction is None:
        peak_traction = PEAK_TRACTION * spec["wall_shear_rate_1_s"] / 200.0
    if vn is None:
        vn = VN_AT_200 * (spec["wall_shear_rate_1_s"] / 200.0) ** (1.0 / 3.0)

    lines, debonded = [], 0
    footprint, reformed = BONDED_FACES, 0
    for k in range(1, n_steps + 1):
        t = min(k * dt, spec["end_time_s"])
        lines.append(f"Time = {t:.10g}s\n")
        if stop_step is not None:
            lines.append(f"PHASE_CLOT_DT step={k} time={t:.10g} deltaT={dt:.17g} dtFluidCo=1 dtWave=1\n")
        # A skeleton at its iteration cap with a residual four orders worse:
        # the fixed point ran out of budget rather than converging. This is
        # what the finite-strain gate saw under 3x load, and the deformation
        # series runs at up to 40x, so the report has to survive it.
        lines.append(
            f"PHASE_CLOT_FSI_SWEEP time={t:.10g} sweep=1 "
            f"skelIters={SKEL_CAP if skeleton_at_cap else 9} "
            f"skelResidual={'1.7e-04' if skeleton_at_cap else '8.7e-08'}\n"
        )
        lines.append(
            f"PHASE_CLOT_FSI time={t:.10g} iterations=8 residual0=9.8e-04 "
            f"residual=4.1e-05 contraction=0.37 omega=0.5 converged=true\n"
        )
        # A diverging child stops EARLY. It writes the guard line, keeps the
        # fields it had reached, and does not run to the end -- which is the
        # whole point: run 20260811-051109 lost three children to SIGFPE and
        # returned nothing at all about the arterial regime.
        if skeleton_diverges and k == 3:
            lines.append(
                f"PHASE_CLOT_SKELETON_DIVERGED time={t:.10g} corrector=7 "
                f"reason=residualGrowing residual=3.4e-02 "
                f"consecutiveIncreases=5 UsPeak=8.1e-06\n"
            )
        if spec.get("advect_released"):
            # This line MUST track the solver's real format. When it did not,
            # the positional regex in assess_case silently matched nothing and
            # run 20260813-084207 reported a null body velocity on a run that
            # moved 49 cells. The parser is key-based now, and this fixture
            # carries the fields the solver actually prints.
            fields = [
                f"time={t:.10g}",
                f"Ux={body_velocity:.6g}",
                "Uy=0",
                "UyDiscarded=-9.1e-05",
                "crossFraction=0.054",
                "axisTilt=0.031",
                "bandCells=1144",
                "freeStreamUx=2.456e-03",
                f"ratioToFreeStream={abs(body_velocity)/2.456e-3:.6g}",
            ]
            if spec.get("body_velocity_field_dropped"):
                fields = [
                    f for f in fields
                    if not f.startswith(spec["body_velocity_field_dropped"] + "=")
                ]
            lines.append("PHASE_CLOT_BODY_VELOCITY " + " ".join(fields) + "\n")
        if spec["growth"]:
            lines.append(
                f"PHASE_CLOT_DIAGNOSTIC time={t:.10g} inventory=1.63e-14 "
                f"surfaceLoss=2.48e-15 outerFlux=-1.4e-16 "
                f"balanceResidual=-1.4e-16 "
                f"clotVolume={V0 + vn * area * t:.12g} "
                f"outletMinUx=0.000166 outletMeanUx=0.002 outletCells=40\n"
            )
        if spec["wall_condition"] == "clotWallAdhesion":
            new = (
                debond_schedule[k - 1]
                if debond_schedule and k - 1 < len(debond_schedule) else 0
            )
            debonded += new
            # THE RATCHET, reproducible on demand. footprint_growth models the
            # pre-fix solver: growth extends the adhesive footprint onto virgin
            # wall faces, each arriving fully bonded, so the total climbs past
            # BONDED_FACES and one new bond re-anchors the whole body. The
            # injury latch is supposed to make this impossible; the harness can
            # still generate it so the assessor is tested against a log that
            # shows it rather than only against logs that do not.
            grown = (
                footprint_growth[k - 1]
                if footprint_growth and k - 1 < len(footprint_growth) else 0
            )
            footprint += grown
            reformed += grown
            bonded = max(0, footprint - debonded)
            strength = spec["adhesive_strength_pa"]
            lines.append(
                f"PHASE_CLOT_DEBOND time={t:.10g} newFaces={new} "
                f"reformedFaces={grown} "
                f"debondedFaces={debonded} bondedClotFaces={bonded} "
                f"injuryFaces={footprint} "
                f"wallFaces=160 debondedArea={debonded * 2.25e-12:.6g} "
                f"peakTraction={peak_traction if bonded else 0.0:.6g} "
                f"peakTractionAny={peak_traction:.6g} strength={strength:.6g} "
                f"utilisation={peak_traction / strength:.6g}\n"
            )
        # A fixedValue patch emits NOTHING. PHASE_CLOT_DEBOND belongs to the
        # clotWallAdhesion condition, and the solver skips every other patch
        # type outright. An earlier version of this harness synthesised the
        # line for welded children anyway, with a comment asserting that they
        # report -- my assumption about the solver, written into the test, so
        # the test confirmed the assumption rather than the behaviour. All nine
        # required-strength children of run 20260810-212211 came back null.
        if cohesive_cells and k == n_steps:
            lines.append(
                f"PHASE_CLOT_DETACHMENT time={t:.10g} cells={cohesive_cells} "
                "volume=3.9e-18 maxDamage=0.97\n"
            )
        # THE TEAR, as the solver would log it: damaged cells are erased
        # (PHASE_CLOT_DETACHMENT), and the piece they cut off loses its path
        # to the wall and is flagged released (PHASE_CLOT_RELEASED) with no
        # face having debonded.
        h_bulk = 2.6e-02
        driver = (
            f"PHASE_CLOT_DAMAGE_DRIVER time={t:.10g} Hbulk={h_bulk:.6g} "
            f"Hinterface={h_bulk * interface_driver_ratio:.6g} "
            f"Hblood=2.5e-06"
        )
        if core_rows is not None:
            # The core is ten times stiffer, and psi+ goes as tau^2/G, so it
            # stores about a tenth of the shell's driving energy. That ratio
            # is the design; a fixture that inverted it would let the report
            # pass on a clot whose core was the weaker material.
            driver += f" Hcore={h_bulk * 0.1:.6g} Hshell={h_bulk:.6g}"
            driver += " DgCore=0 DgShell=0.99"
        lines.append(driver + f" maxDg={dg_peak:.6g}\n")
        # The realised clot permeability, from the solver rather than from
        # k0p. Two decades apart under kozenyCarman, and the difference was
        # invisible until run 20260814-055141 was measured cell by cell.
        k_floor = 2.22e-16
        if permeability_floored:
            k_min = k_max = k_floor
            floored = 324
        else:
            k_min = spec.get("k0p_core_m2") or spec["k0p_m2"]
            k_max = spec["k0p_m2"]
            floored = 0
        lines.append(
            f"PHASE_CLOT_PERMEABILITY time={t:.10g} "
            f"model={spec.get('permeability_model', 'kozenyCarman')} "
            f"clotCells=324 kClotMin={k_min:.6g} kClotMax={k_max:.6g} "
            f"flooredCells={floored} floor={k_floor:.6g}\n"
        )

        # A HEALTHY SYNTHETIC RUN MUST CARRY THE INSTRUMENTS A HEALTHY REAL
        # RUN CARRIES. The solver emits PHASE_CLOT_PHASE_BOUNDS and
        # PHASE_CLOT_PLATELET on EVERY timestep -- Batch 25 logged 20,099 of
        # each for 20,099 steps -- and this fixture emitted neither. That was
        # invisible while both gates were fail-open, and it broke the build
        # the moment they were made fail-closed on 2026-09-08: the "healthy"
        # scenario was asserting exit 0 from a log that a correct assessor
        # must reject as unassessable. The fixture was wrong, not the gate.
        #
        # Derived from c_extreme so it stays consistent with the c field
        # written below, which means the out-of-bounds scenarios keep
        # discriminating: c_extreme 0.999 gives cUnder = cOver = 0, while a
        # scenario driving c_extreme past 1 produces a real overshoot here.
        c_lo, c_hi = -abs(c_extreme), abs(c_extreme)
        q_min_syn = 0.5*(c_lo + 1.0)
        # S6c-2: EMITTED IN q, AS THE SOLVER NOW DOES. Derived from the same
        # c_lo/c_hi so the scenarios keep discriminating exactly as before:
        # q = 0.5*(c + 1), and assess_case converts back on read.
        q_hi_syn = 0.5*(c_hi + 1.0)
        lines.append(
            f"PHASE_CLOT_PHASE_BOUNDS time={t:.10g} "
            f"qMin={q_min_syn:.12g} qMax={q_hi_syn:.12g} "
            f"qUnder={max(0.0, -q_min_syn):.12g} "
            f"qOver={max(0.0, q_hi_syn - 1.0):.12g} "
            f"qNegCells={0 if q_min_syn >= 0 else 1} "
            f"qNegMin={min(0.0, q_min_syn):.12g} "
            f"qNegAmpMax={0.0 if q_min_syn >= 0 else 1.0/max(abs(q_min_syn), 1e-6):.12g} "
            f"qNegVolume={0.0 if q_min_syn >= 0 else 5e-19:.12g} "
            f"qNegContent={0.0 if q_min_syn >= 0 else 5e-19*abs(q_min_syn):.12g}\n"
        )
        lines.append(
            f"PHASE_CLOT_PLATELET time={t:.10g} wMax=1 wMin=0.3 "
            f"VnMagMax=3.75e-05 wOverBound=0 WpDevMax=0.05 qMin=1e-06 "
            f"wDevBlood=0.1\n"
        )
        if fct is not None:
            lam, rule = fct
            act, lim, lam_k = (0, 0, 1) if k == 1 else (56369, 10111, lam)
            lines.append(
                f"PHASE_CLOT_CONTENT_FCT time={t:.10g} activeFaces={act} "
                f"limitedFaces={lim} lambdaMin={0 if act else 1} lambdaFluxMean={lam_k} "
                f"negAfter=0"
                + ("" if rule is None else f" zeroFluxRule={rule} partitionSumDevMax=0")
                + "\n")
        if tear_rows is not None and k >= max(1, int(tear_from_step * n_steps)):
            if k == max(1, int(tear_from_step * n_steps)):
                lines.append(
                    f"PHASE_CLOT_DETACHMENT time={t:.10g} cells=61 "
                    "volume=1.0e-16 maxDamage=0.99\n"
                )
            lines.append(
                f"PHASE_CLOT_RELEASED time={t:.10g} cells=180 volume=3.0e-16\n"
            )
    lines.append(
        "ExecutionTime = 500 s  ClockTime = 510 s\n\nEnd\n"
        if exit_code == 0 else "Foam::sigFpe::sigHandler(int) at ??:?\n"
    )
    (case / "log.phaseClotAxisymmetricFluid").write_text("".join(lines))
    (case / "run-exit-code.txt").write_text(f"{exit_code}\n")
    (case / "log.wallTime").write_text("WALL_SECONDS=510\n")

    # theta_ref for the reconstruction. Fixed rather than read: the gate
    # children carry no skeletonVolumeFraction, so the fixture has to supply
    # one, and it must be the same number the assessor will read back.
    theta_ref = 0.45

    def write_phase(d: Path, c_field: list[float]) -> None:
        """The phase field, in whichever convention this child writes."""
        if not theta_native:
            write_scalar(d / "c", c_field)
            return
        q = [(v + 1.0) / 2.0 for v in c_field]
        write_scalar(d / "q", q)
        write_scalar(d / "thetaC", [theta_ref * (1.0 - v) for v in q])
        write_scalar(d / "thetaH", [0.0] * len(q))

    def make_theta_native() -> None:
        """Convert 0/ and the dictionary, once the time dirs are written.

        0/c stays, exactly as create_cases.py leaves it, and 0/thetaC and
        0/thetaH are derived FROM it so the two conventions agree at t = 0 --
        which is what run 20260920-060033 measured on the real solver
        (6.7e-16). A fixture that seeded them independently would hide a
        convention error behind a plausible initial condition.
        """
        if not theta_native:
            return
        props = case / "constant" / "phaseClotProperties"
        text = props.read_text() if props.exists() else ""
        if "skeletonVolumeFraction" not in text:
            props.parent.mkdir(parents=True, exist_ok=True)
            props.write_text(text + f"\nskeletonVolumeFraction {theta_ref};\n")
        zero_c = case / "0" / "c"
        if not zero_c.is_file():
            raise AssertionError(
                "theta_native fixture: 0/c is absent, so 0/thetaC cannot be "
                "derived from it and the t=0 measurements would test nothing"
            )
        import re as _re
        m = _re.search(
            r"(?ms)internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*"
            r"\(\s*(.*?)\s*\)\s*;",
            zero_c.read_text(),
        )
        if m:
            c0 = [float(v) for v in m.group(2).split()]
        else:
            u = _re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;",
                           zero_c.read_text())
            if not u:
                raise AssertionError("theta_native fixture: cannot parse 0/c")
            c0 = [float(u.group(1))] * N_CELLS
        q0 = [(v + 1.0) / 2.0 for v in c0]
        write_scalar(case / "0" / "thetaC", [theta_ref * (1.0 - v) for v in q0])
        write_scalar(case / "0" / "thetaH", [0.0] * len(q0))

    if omit_fields:
        make_theta_native()
        return
    value = spec["write_interval_s"]
    write_times = []
    if stop_step is not None:
        # write_every_step: a full time directory at EVERY step, of which controlDict's purgeWrite keeps the last K (create_cases.py
        # S25_RT1_PURGE_WRITE = 10, so steps stop-9..stop: N0 = stop-5 survives). K is read from the case's own controlDict, not restated.
        pw = re.search(r"^\s*purgeWrite\s+(\d+)\s*;", (case / "system" / "controlDict").read_text(), re.M)
        assert pw and spec.get("write_every_step"), "a stop_after_step child must carry write_every_step and a purgeWrite"
        first = max(1, n_steps - int(pw.group(1)) + 1)
        write_times = [round(min(k * dt, spec["end_time_s"]), 10) for k in range(first, n_steps + 1)]
    else:
        while value <= spec["end_time_s"] + 1e-12:
            write_times.append(value)
            value = round(value + spec["write_interval_s"], 10)
    for value in write_times:
        d = case / f"{value:.10g}"
        d.mkdir(parents=True, exist_ok=True)
        c = [c_extreme, -c_extreme] + [0.5] * (N_CELLS - 2)
        write_phase(d, c)
        write_scalar(d / "Dg", damage_field(value / spec["end_time_s"]))
        write_scalar(d / "poroPerm", [spec["k0p_m2"]] * N_CELLS)
        write_scalar(d / "shearRate", [spec["wall_shear_rate_1_s"]] * N_CELLS)
        # Us SCALES WITH THE SHEAR RATE. A constant here would let the
        # deformation table report the same strain at 1000/s and 8000/s and
        # still pass, which is precisely the kind of fixture that let the
        # transport test pass on a clot that never moved.
        load = spec["wall_shear_rate_1_s"] / 200.0
        us = [(US_MAX_AT_200 * load * us_scale, 1e-8, 0.0)] * N_CELLS
        if nan_field:
            us[0] = (float("nan"), 0.0, 0.0)
        write_vector(d / "Us", us)

    # The clot, as a block of cells, so a centroid can be computed and a shift
    # can be seen. NX matches the mural mesh the case actually uses.
    nx, h = spec["geometry"]["nx"], spec["geometry"]["h_m"]
    write_scalar(case / "0" / "Ccx",
                 [((i % nx) + 0.5) * h for i in range(N_CELLS)])
    # Ccy too, or the HEIGHT metrics silently return None and `translating`
    # degrades to a length-only check -- which is exactly the blind spot that
    # let run 20260812-200349's 70% vertical stretch read as a clean
    # translation.
    write_scalar(case / "0" / "Ccy",
                 [((i // nx) + 0.5) * h for i in range(N_CELLS)])

    # THE MATERIAL INDICATOR, written only for a core/shell child, exactly as
    # the initialiser does. Absent otherwise, which is what the solver's
    # single-material path expects and what every existing scenario gets.
    #
    # It is written at EVERY time, not only at 0, because the solver now
    # advects it: the label is a property of the material, so when the shell
    # is carried downstream the label goes with it. A fixture that kept it
    # fixed would let a report pass that had silently swapped the two
    # materials' moduli as the clot moved.
    def core_shell_fields(shell_dx: int, core_dx: int, extra_rows: int):
        """c and shell at one time, with the two regions moved separately."""
        c_field = [c_extreme] * N_CELLS
        shell_field = [1.0] * N_CELLS
        rows_total = 13 + extra_rows
        # Shell first, then core over the top of it, so the core wins where
        # they overlap -- which is what a buried core means.
        for row in range(rows_total):
            for col in range(50 + shell_dx, min(110 + shell_dx, nx)):
                c_field[row * nx + col] = -c_extreme
        for row in range(core_rows[0], core_rows[1] + 1):
            for col in range(core_cols[0] + core_dx,
                             min(core_cols[1] + core_dx, nx)):
                idx = row * nx + col
                c_field[idx] = -c_extreme
                shell_field[idx] = 0.0
        return c_field, shell_field

    if core_rows is not None:
        c0f, s0f = core_shell_fields(0, 0, 0)
        write_scalar(case / "0" / "shell", s0f)
        write_scalar(case / "0" / "c", c0f)
        if theta_native:
            q0 = [(v + 1.0) / 2.0 for v in c0f]
            write_scalar(case / "0" / "thetaC",
                         [theta_ref * (1.0 - v) for v in q0])
            write_scalar(case / "0" / "thetaH", [0.0] * len(q0))

    def clot_at(offset_cells: int, dissolve: bool = False,
                depth: float | None = None,
                trailing_extra: int = 0, extra_rows: int = 0) -> list[float]:
        # trailing_extra advances the UPSTREAM edge further than the
        # downstream one, which is erosion from behind rather than
        # translation -- the failure mode of run 20260812-041322, where the
        # trailing edge ran 17 cells and the leading edge only 9.
        lo, hi = 50 + offset_cells + trailing_extra, 110 + offset_cells
        rows = 1 if dissolve else 13 + extra_rows
        # `depth` weakens c towards zero while KEEPING the footprint, which is
        # what advecting at Courant 16.2 actually did in run 20260811-195034:
        # c_min -0.997 -> -0.145 with the cell count unchanged at ~721. The
        # `dissolve` flag above shrinks the body instead, so the cell count
        # moves and volume_change_fraction sees it. Only this mode reproduces
        # the failure that got past the report.
        # c_extreme must survive into these fields: the out-of-bounds
        # scenario injects it, and an earlier version of this block overwrote
        # it with a hard-coded 0.9, silently disarming that test.
        inside = -(depth if depth is not None else c_extreme)
        field = [c_extreme] * N_CELLS
        for row in range(rows):
            for i in range(lo, min(hi, nx)):
                field[row * nx + i] = inside
        return field

    # ONE CONTINUOUS TIME SERIES. There is no transport stage any more: the
    # timestep is adaptive and the clot peels, deforms and travels in the same
    # run, so the fixture writes one series in which the body moves.
    #
    # THE SYNTHETIC DATA MUST OBEY THE SOLVER'S COUPLING, not the scenario's
    # wishes. Advection is gated on `released`, which the connectivity sweep
    # sets to zero for the WHOLE clot while any face remains bonded. A body
    # that never fully debonds therefore cannot move, and synthesising motion
    # for one would let a scenario "prove" transport the solver could never
    # produce -- the same defect as synthesising PHASE_CLOT_DEBOND lines for
    # welded children, which is why run 20260810-212211 shipped nine nulls
    # against a passing test.
    fully_released = sum(debond_schedule or []) >= BONDED_FACES
    # A COHESIVE TEAR RELEASES A FRAGMENT WITH NO DEBONDING AT ALL. That is
    # the whole mechanism of the tear series: updateReleasedIndicator() asks
    # whether a clot cell still has a path through clot material to an intact
    # bond, and the tear cuts that path. So `moves` must not require the bond
    # to have failed, or the fixture would assert that a welded clot can never
    # shed anything -- the opposite of what this series is testing.
    torn_free = tear_rows is not None and spec.get("advect_released")
    moves = (
        spec.get("advect_released")
        and (fully_released or torn_free)
        and transport_shift_cells is not None
    )
    total_shift = transport_shift_cells if moves else 0

    # Motion starts only once the bond has actually gone, which for the real
    # schedules is partway through the series.
    release_step = 0
    if torn_free:
        # Motion starts when the tear opens, not when the bond goes.
        release_step = int(tear_from_step * len(
            [1 for _ in range(int(round(spec["end_time_s"]
                                       / spec["write_interval_s"])))]
        ))
    if debond_schedule:
        run = 0
        for i, n in enumerate(debond_schedule):
            run += n
            if run >= BONDED_FACES:
                release_step = i + 1
                break

    # The transport fixtures re-write the material fields at the same directories the fields above were written at; for a
    # stop_after_step child those are its every-step directories (write_times), not the write_interval_s ones.
    times = []
    value = spec["write_interval_s"]
    if stop_step is not None:
        times = list(write_times)
    else:
        while value <= spec["end_time_s"] + 1e-12:
            times.append(round(value, 10))
            value = round(value + spec["write_interval_s"], 10)


    for k, t_now in enumerate(times, start=1):
        d = case / f"{t_now:.10g}"
        d.mkdir(parents=True, exist_ok=True)
        frac = 0.0
        if moves and len(times) > release_step:
            frac = max(0.0, (k - release_step) / (len(times) - release_step))
        depth = None
        if transport_smears_to is not None:
            depth = c_extreme + (transport_smears_to - c_extreme)*min(1.0, k/len(times))
        # Dissolution has to be PROGRESSIVE, or the first frame is already
        # dissolved and there is nothing to compare the last one against --
        # the metric would report a conserved body that was never there.
        shrinking = transport_dissolves and k > len(times) // 2
        # Growth adds rows through the run, on the same mechanism as the
        # stretch defect but for the opposite reason: one is the clot gaining
        # material, the other is the phase field being pulled into a wall.
        # Separate knobs so the scenario that pins the stretch cannot be
        # satisfied by a fixture that grew instead.
        c_field = clot_at(int(round(total_shift*frac)), shrinking, depth,
                          int(round(transport_erodes_cells*frac)),
                          int(round(transport_stretches_rows*frac))
                          + int(round(growth_rows*k/len(times))))
        if final_state is not None and k == len(times):
            # The core is gone either way; what differs is the MATERIAL.
            c_field = [c_extreme] * N_CELLS
            if final_state == "stranded":
                # The trail the body left along its whole path: same 13 rows
                # and same upstream edge as clot_at uses, running all the way
                # to the outlet the core departed through.
                for row in range(13):
                    for i in range(50, nx):
                        c_field[row * nx + i] = 0.4
            elif final_state != "clean_exit":
                raise ValueError(f"unknown final_state {final_state!r}")
        load = spec["wall_shear_rate_1_s"] / 200.0
        us = [(US_MAX_AT_200*load*us_scale, 1e-8, 0.0)] * N_CELLS
        if core_rows is not None:
            # The two regions move independently, and Us follows the material
            # contrast: strain goes as 1/G, so the 4 Pa shell reaches ten
            # times the 40 Pa core. A fixture with one uniform Us would let a
            # report pass on a solve in which the moduli never differed.
            progress = k / len(times)
            c_field, shell_field = core_shell_fields(
                int(round(shell_shift_cells * progress)),
                int(round(core_shift_cells * progress)),
                int(round(growth_rows * progress)),
            )
            write_scalar(d / "shell", shell_field)
            us = [
                (US_MAX_AT_200 * load * us_scale * (1.0 if s < 0.5 else 10.0),
                 1e-8, 0.0)
                for s in shell_field
            ]
        write_phase(d, c_field)
        write_scalar(d / "Dg", damage_field(k / len(times)))
        write_scalar(d / "poroPerm", [spec["k0p_m2"]] * N_CELLS)
        write_scalar(d / "shearRate", [spec["wall_shear_rate_1_s"]] * N_CELLS)
        if nan_field:
            us[0] = (float("nan"), 0.0, 0.0)
        write_vector(d / "Us", us)

    make_theta_native()


def build_case(tmp: Path) -> Path:
    cases = tmp / "cases"
    cases.mkdir()
    root = cases / CASE.name
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "runs", "postProcessing", "log.*"
    )
    shutil.copytree(CASE, root, ignore=ignore)
    for name in ("AM2020_straight_tube_growth_mechanics_smoke",
                 "AM2020_mural_geometry_gate"):
        shutil.copytree(CASE.parent / name, cases / name, ignore=ignore)
    subprocess.run(
        [sys.executable, str(root / "scripts" / "create_cases.py")],
        check=True, stdout=subprocess.DEVNULL,
    )
    return root


def run_assessor(root: Path) -> tuple[int, dict]:
    done = subprocess.run(
        [sys.executable, str(root / "scripts" / "assess_case.py"), str(root)],
        text=True, capture_output=True,
    )
    path = root / "postProcessing" / "mainCase" / "result.json"
    if not path.is_file():
        raise AssertionError(
            f"no result.json\nstdout:{done.stdout}\nstderr:{done.stderr}"
        )
    return done.returncode, json.loads(path.read_text())


GOOD_BUILD = "'libphaseClotAxisymmetricFluid.so' is up to date.\n"
BAD_BUILD = (
    "phaseClotAxisymmetricFluid.C:718:31: error: no match for 'operator&'\n"
    "make: *** [phaseClotAxisymmetricFluid.o] Error 1\n"
)

# A peel that advances: the shape the demonstration children must produce.
PEEL_FULL = [0, 0, 2, 5, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1]

# 59 of 60, one face surviving.  This is what run 20260811-004647 actually did
# at 1.2 Pa: the peel arrests because the traction relaxes below the strength
# once the clot hangs by a single face.  It is a distinct outcome from a full
# release and the report must not blur the two, so the schedules differ by one
# face and the expectations below check the consequence -- released_cells and
# transport are zero here and non-zero for the 0.8 Pa child.
PEEL_ARREST = PEEL_FULL[:-1] + [0]
PEEL_PARTIAL = [0] * 8 + [1, 1, 2, 2, 1, 1, 1]

# Above this the bond is 'unbreakable' by the controls file's own
# definition; mainCaseControls sets adhesive_strength.upper_bound_pa.
COHESIVE_BOUND_PA = 15600.0


def peel_schedule(strength_pa: float, shear_rate_1_s: float) -> list:
    """Route a demonstration child to the outcome its strength produces.

    Routed on strength RELATIVE TO THE FLOW, not on absolute Pa. The bond
    traction is set by the flow, so 6.0 Pa at 1500 1/s is the same operating
    point as 0.8 Pa at 200 1/s -- the venous rate the three regimes were
    measured at. Keying on absolute Pa sent the arteriolar demonstration child
    to PEEL_PARTIAL, i.e. the fixture quietly asserted it would stall, which is
    the opposite of what it is for.
    """
    # AN UNBREAKABLE BOND NEVER DEBONDS. The required-strength and
    # cohesive-tear children run at 1e9 Pa precisely so the wall cannot fail,
    # and routing them to PEEL_PARTIAL made the fixture assert nine faces came
    # off a bond that by construction cannot break. For the tear series that
    # is not cosmetic: `wall_held` is the control on the whole experiment, and
    # a fixture that breaks the wall would fail it for a reason the solver
    # will never produce.
    if strength_pa >= COHESIVE_BOUND_PA:
        return []
    equivalent = strength_pa * REFERENCE_SHEAR_1_S / shear_rate_1_s
    if equivalent < 1.0:
        return PEEL_FULL
    if equivalent < 1.5:
        return PEEL_ARREST
    return PEEL_PARTIAL


def variant(**overrides) -> dict:
    merged = {}
    for child_id, changes in overrides.items():
        merged[child_id] = dict(changes)
    return merged



def report_differences(a, b, path: str = "", out: list | None = None) -> list:
    """Every place two assessor reports disagree, as readable paths.

    Timings and absolute paths are excluded by name: they differ between two
    temporary directories for reasons that are not the report.
    """
    if out is None:
        out = []
    volatile = ("wall_seconds", "execution_time_s", "clock_time_s",
                "generated_at", "case_dir", "root", "path")
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in volatile:
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'missing in base' if k not in a else 'missing in new'}")
                continue
            report_differences(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                report_differences(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        # The reconstruction is exact in exact arithmetic and 2q-1 costs one
        # rounding, so a last-digit difference is the float, not the report.
        if not (a == b or abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out

def exercise(name: str, per_child: dict, expectations, build_log=GOOD_BUILD,
             every_child: dict | None = None, after=None):
    with tempfile.TemporaryDirectory(prefix="am2020-main-") as tmp:
        root = build_case(Path(tmp))
        (root / "log.Allwmake").write_text(build_log)
        matrix = json.loads((root / "runs" / "matrix.json").read_text())
        for spec in matrix:
            kwargs = per_child.get(spec["child_id"], {})
            if kwargs.get("_skip"):
                continue
            if every_child:
                kwargs = {**every_child, **kwargs}
            if spec["wall_condition"] == "clotWallAdhesion" and "debond_schedule" not in kwargs:
                kwargs = dict(kwargs)
                kwargs["debond_schedule"] = peel_schedule(
                    spec["adhesive_strength_pa"], spec["wall_shear_rate_1_s"]
                )
            synthesise_child(root / "runs" / spec["child_id"], spec, **kwargs)
        if after is not None:            # files the runner, not the solver, leaves behind
            after(root)
        code, result = run_assessor(root)
        try:
            expectations(result, code)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}\n{json.dumps(result, indent=1)[:3000]}")
        print(f"{name:<38} -> {result['children_admissible']}/"
              f"{result['children_total']} admissible")
        return result


def _s16(controls: dict, child_id: str, **changes) -> None:
    """S16 negatives: set (or, with None, delete) keys on ONE named child.
    Refuses a child id that is not in the controls, so a renamed child
    cannot turn a negative into a test of nothing."""
    hits = [c for c in controls["children"] if c["child_id"] == child_id]
    if len(hits) != 1:
        raise AssertionError(f"S16 negative names {child_id!r}: {len(hits)} match")
    for key, value in changes.items():
        if value is None:
            hits[0].pop(key, None)
        else:
            hits[0][key] = value


# ---- S19, THE BLOCK-COUPLED STEP (batch 65; design 2026-10-03-S19-block-
# coupled-step rev. 4, APPROVED 2026-10-04) ----
#
# One negative per generator refusal, module-level so each can be run alone
# (each was run once, directly, by the solver-engineer, docket 20261003-0205
# report 09). (label, mutate(controls), the refusal's own words).
S19_REFUSALS = (
    ("a misspelt network_viscosity",
     lambda c: _s16(c, "fsb_ctl_r3", network_viscosity="kv"),
     "network_viscosity 'kv' is not one of none, kelvinVoigt."),
    ("a non-positive tau_v_s",
     lambda c: _s16(c, "fsb_ctl_r3", tau_v_s=0.0),
     "tau_v_s 0.0 must be a positive number of seconds"),
    ("R3 under segregated",
     lambda c: _s16(c, "fsb_ctl_r3", momentum_pressure_coupling=None),
     "network_viscosity kelvinVoigt needs momentum_pressure_coupling blockCoupled"),
    ("R3 with the content weight",
     lambda c: _s16(c, "fsb_ctl_r3", network_pressure_weight="contentRatio"),
     "network_viscosity kelvinVoigt with network_pressure_weight contentRatio"),
)


# ---- S22, THE BLOCK STEP'S LINEAR SOLVER (batch 71; design 2026-10-07-S22-pardiso-
# block-solver rev. 2.1 sections 4 and 7) ----
#
# One negative per generator refusal (create_cases.py s22_key_refusal), module-level so
# each runs alone: `python3 -B -c "import dry_run; dry_run.s22_generator_negatives('<label>')"`
# from the scripts directory. (label, mutate(controls), the refusal's own words).
S22_REFUSALS = (
    ("a misspelt block_linear_solver",
     lambda c: _s16(c, "S22v_G40_C", block_linear_solver="mkl"),
     "block_linear_solver 'mkl' is not one of pardiso, umfpack (S22 section 4)."),
    ("a non-string block_linear_solver",
     lambda c: _s16(c, "S22v_G40_C", block_linear_solver=True),
     "block_linear_solver True is not one of pardiso, umfpack (S22 section 4)."),
    ("a negative block_solver_threads",
     lambda c: _s16(c, "S22v_G40_E", block_solver_threads=-1),
     "block_solver_threads -1 must be an integer >= 0"),
    ("a fractional block_solver_threads",
     lambda c: _s16(c, "S22v_G40_E", block_solver_threads=1.5),
     "block_solver_threads 1.5 must be an integer >= 0"),
    ("a boolean block_solver_threads",
     lambda c: _s16(c, "S22v_G40_E", block_solver_threads=True),
     "block_solver_threads True must be an integer >= 0"),
    ("a non-bool block_solver_reproducible",
     lambda c: _s16(c, "S22v_G40_C", block_solver_reproducible="yes"),
     "block_solver_reproducible 'yes' must be true or false (S22 section 4)."),
    ("a label block_solver_reproducible",
     lambda c: _s16(c, "S22v_G40_C", block_solver_reproducible=1),
     "block_solver_reproducible 1 must be true or false (S22 section 4)."),
    ("block_solver_reproducible under umfpack",
     lambda c: _s16(c, "S22v_G40_A", block_solver_reproducible=True),
     "block_solver_reproducible is MKL's conditional-numerical-reproducibility switch; under "
     "block_linear_solver umfpack it would be silently inert"),
    ("block_linear_solver without blockCoupled",
     lambda c: _s16(c, "S22v_G40_C", momentum_pressure_coupling=None, network_viscosity=None, tau_v_s=None,
                    _tau_v_s_status=None),
     "block_linear_solver is read only under momentum_pressure_coupling blockCoupled"),
    ("block_solver_threads without blockCoupled",
     lambda c: _s16(c, "em78_G4_s18p_ctl", block_solver_threads=2),
     "block_solver_threads is read only under momentum_pressure_coupling blockCoupled"),
    ("block_solver_reproducible without blockCoupled",
     lambda c: _s16(c, "em78_G4_s18p_ctl", block_solver_reproducible=False),
     "block_solver_reproducible is read only under momentum_pressure_coupling blockCoupled"),
    ("an unknown solver_library",
     lambda c: _s16(c, "S22v_G40_A0", solver_library="p1"),
     "solver_library 'p1' is not one of p2prime, s22, baseline (S22 section 7)."),
    ("solver_library p2prime without block_linear_solver umfpack",
     lambda c: _s16(c, "S22v_G40_A0", block_linear_solver="pardiso"),
     "solver_library p2prime runs the P2-prime solver, which has only UMFPACK"),
    ("solver_library p2prime with a thread count P2' ignores",
     lambda c: _s16(c, "S22v_G40_A0", block_solver_threads=1),
     "solver_library p2prime runs the P2-prime solver, which has only UMFPACK"),
    ("S22v_G40_E drifting from its parent",
     lambda c: _s16(c, "S22v_G40_E", k0p_m2=2e-15),
     "FSF parent check: S22v_G40_E differs from S22v_G40_A in k0p_m2"),
    ("S22v_G40_C without its named key",
     lambda c: _s16(c, "S22v_G40_C", block_linear_solver="umfpack"),
     "FSF parent check: S22v_G40_C does not differ from S22v_G40_A in its named key block_linear_solver"),
)


def s22_generator_negatives(only: str | None = None) -> None:
    """Each S22 generator refusal, seen to fire for its stated reason on a fresh copy
    (all of them, or the one whose label contains `only`)."""
    import copy
    ran = 0
    with tempfile.TemporaryDirectory(prefix="am2020-s22-neg-") as tmp:
        root = build_case(Path(tmp))
        shutil.rmtree(root / "runs")
        controls_path = root / "constant" / "mainCaseControls.json"
        pristine = controls_path.read_text()
        for label, mutate, expected in S22_REFUSALS:
            if only is not None and only not in label:
                continue
            broken = copy.deepcopy(json.loads(pristine))
            mutate(broken)
            controls_path.write_text(json.dumps(broken, indent=1))
            done = subprocess.run(
                [sys.executable, "-B", str(root / "scripts" / "create_cases.py")],
                capture_output=True, text=True)
            said = (done.stdout + done.stderr).strip().splitlines()
            last = said[-1] if said else ""
            if done.returncode == 0:
                raise AssertionError(f"the generator ACCEPTED {label}; it must refuse")
            if expected not in last:
                raise AssertionError(f"the generator refused {label} for the WRONG reason: {last}")
            print(f"generator refuses: {label:<58} -> {last[:96]}")
            shutil.rmtree(root / "runs", ignore_errors=True)
            ran += 1
        controls_path.write_text(pristine)
        # the pristine controls still generate (a negative that always fails proves nothing)
        done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")],
                              capture_output=True, text=True)
        assert done.returncode == 0, f"the pristine controls were refused: {(done.stdout + done.stderr)[-300:]}"
        # S22 keys are written to the dictionary only for a child that names them, with the exact words
        # the solver reads (S22 sections 2.4-2.6), and never to any of the 18 earlier children.
        props = lambda cid: (root / "runs" / cid / "constant" / "phaseClotProperties").read_text()
        assert "blockLinearSolver umfpack;" in props("S22v_G40_A")
        assert "blockLinearSolver pardiso;" in props("S22v_G40_C") and "blockSolverThreads" not in props("S22v_G40_C")
        assert "blockSolverThreads 1;" in props("S22v_G40_E") and "blockSolverThreads 1;" in props("S22v_G40_B")
        assert not any(k in props(c) for c in ("fsf_G40_klo_v250", "fsf_G4_kxu_r3_40", "em76_G4_s18")
                       for k in ("blockLinearSolver", "blockSolverThreads", "blockSolverReproducible"))
        assert (root / "runs" / "S22v_G40_A0" / "child.env").is_file()
        assert not any((root / "runs" / c / "child.env").exists() for c in ("S22v_G40_A", "fsf_G40_klo_v250"))
        shutil.rmtree(root / "runs", ignore_errors=True)
    assert ran > 0, f"no S22 negative matches {only!r}"
    print(f"   {ran} S22 generator negatives refused; the written keys are exactly the named ones")



# ---- S23, THE TORN-NETWORK FIX (batch 72; design 2026-10-07-S23-torn-network-debris
# rev. 3 sections 4 and 7) ----
#
# One negative per generator refusal (create_cases.py s23_key_refusal, and the BATCH 72
# parent pairs), module-level so each runs alone:
#   python3 -B -c "import dry_run; dry_run.s23_generator_negatives('<label>')"
# from the scripts directory. (label, mutate(controls), the refusal's own words).
# The children pinned to the S22 library by rev. 4.1 (design section 13.2 item 5: the 12 R3-off children),
# plus fsb_ctl_r3 (kelvinVoigt at the solver default tau_v 2.5e-4 s, below the 5e-4 s floor; CE report 25).
S23_PINNED_S22 = (
    "em76_G4_s18", "em78_G4_s18p_ctl", "em79_G4_s18p_Cnp", "em78_id", "xu5_g250", "xu5_g500",
    "xu5_g1000", "xu5_g2000", "xu5_id", "fsb_G40_klo_rest", "fs_G40_klo_rest_seg",
    "fsf_G4_kxu_nor3_40", "fsb_ctl_r3")
S23_REFUSALS = (
    ("volumetric_tear_ratio = 1 (the boundary: R_c <= 1 is refused)",
     lambda c: _s16(c, "S23v_G40", volumetric_tear_ratio=1.0),
     "volumetric_tear_ratio 1.0 must be finite and exceed 1"),
    ("volumetric_tear_ratio below 1",
     lambda c: _s16(c, "S23v_G40", volumetric_tear_ratio=0.5),
     "volumetric_tear_ratio 0.5 must be finite and exceed 1"),
    ("a negative volumetric_tear_ratio",
     lambda c: _s16(c, "S23v_G40", volumetric_tear_ratio=-21.95),
     "volumetric_tear_ratio -21.95 must be finite and exceed 1"),
    ("a NaN volumetric_tear_ratio",
     lambda c: _s16(c, "S23v_G40", volumetric_tear_ratio=float("nan")),
     "volumetric_tear_ratio nan must be finite and exceed 1"),
    ("an infinite volumetric_tear_ratio",
     lambda c: _s16(c, "S23v_G40", volumetric_tear_ratio=float("inf")),
     "volumetric_tear_ratio inf must be finite and exceed 1"),
    ("a boolean volumetric_tear_ratio",
     lambda c: _s16(c, "S23v_G40", volumetric_tear_ratio=True),
     "volumetric_tear_ratio True must be a number"),
    ("a string volumetric_tear_ratio",
     lambda c: _s16(c, "S23v_G40", volumetric_tear_ratio="21.95"),
     "volumetric_tear_ratio '21.95' must be a number"),
    ("volumetric_tear_ratio on a child that runs the S22 library",
     lambda c: _s16(c, "S23v_G4_r3_S22", volumetric_tear_ratio=21.95),
     "volumetric_tear_ratio is read only by the S23 solver; solver_library s22 runs the S22 reference library"),
    ("volumetric_tear_ratio on an old child too (every child is checked)",
     lambda c: _s16(c, "fsf_G40_klo_v250", volumetric_tear_ratio=0.0),
     "volumetric_tear_ratio 0.0 must be finite and exceed 1"),
    ("S23v_G40 drifting from its parent",
     lambda c: _s16(c, "S23v_G40", k0p_m2=2e-15),
     "FSF parent check: S23v_G40 differs from fsf_G40_klo_v250 in k0p_m2"),
    ("S23v_G4_r3 drifting from its parent (a physics switch)",
     lambda c: _s16(c, "S23v_G4_r3", tau_v_s=0.001),
     "FSF parent check: S23v_G4_r3 differs from fsf_G4_kxu_r3_40 in tau_v_s"),
    # BATCH 73 (production): P73_G4_r3_40 is S23v_G4_r3 with NO named key; P73_G40_40 is S23v_G40 with end_time_s and wall_budget_h
    ("P73_G4_r3_40 drifting from S23v_G4_r3 in a run control",
     lambda c: _s16(c, "P73_G4_r3_40", wall_budget_h=12.0),
     "FSF parent check: P73_G4_r3_40 differs from S23v_G4_r3 in wall_budget_h"),
    ("P73_G4_r3_40 drifting from S23v_G4_r3 in a physics key",
     lambda c: _s16(c, "P73_G4_r3_40", k0p_m2=5e-13),
     "FSF parent check: P73_G4_r3_40 differs from S23v_G4_r3 in k0p_m2"),
    ("P73_G40_40 drifting from S23v_G40 in a physics key",
     lambda c: _s16(c, "P73_G40_40", tau_v_s=0.001),
     "FSF parent check: P73_G40_40 differs from S23v_G40 in tau_v_s"),
    ("P73_G40_40 not extended (no end_time_s difference)",
     lambda c: _s16(c, "P73_G40_40", end_time_s=0.025),
     "FSF parent check: P73_G40_40 does not differ from S23v_G40 in its named key end_time_s"),
    ("P73_G40_40 keeping S23v_G40's cap (no wall_budget_h difference)",
     lambda c: _s16(c, "P73_G40_40", wall_budget_h=11.0),
     "FSF parent check: P73_G40_40 does not differ from S23v_G40 in its named key wall_budget_h"),
    ("P73_G40_40 with a frame key dropped (beyond its named keys)",
     lambda c: _s16(c, "P73_G40_40", frame_interval_s=None),
     "frame_interval_s"),
    # BATCH 74 (D74, the passive intact-network reference): D74_refI is S23v_G4_r3 with NO named key. The passive field is a
    # property of the BUILD, so a dictionary difference of any kind (a run control, a physics key, a library pin) is refused.
    ("D74_refI drifting from S23v_G4_r3 in a run control",
     lambda c: _s16(c, "D74_refI", wall_budget_h=12.0),
     "FSF parent check: D74_refI differs from S23v_G4_r3 in wall_budget_h"),
    ("D74_refI drifting from S23v_G4_r3 in the end time",
     lambda c: _s16(c, "D74_refI", end_time_s=0.03),
     "FSF parent check: D74_refI differs from S23v_G4_r3 in end_time_s"),
    ("D74_refI drifting from S23v_G4_r3 in a physics key",
     lambda c: _s16(c, "D74_refI", k0p_m2=5e-13),
     "FSF parent check: D74_refI differs from S23v_G4_r3 in k0p_m2"),
    ("D74_refI pinned to a reference library (it would run without the passive field)",
     lambda c: _s16(c, "D74_refI", solver_library="s22"),
     "FSF parent check: D74_refI differs from S23v_G4_r3 in solver_library"),
    ("D74_refI with a frame key dropped (beyond its named keys)",
     lambda c: _s16(c, "D74_refI", frame_interval_s=None),
     "frame_interval_s"),
    # BATCH 75 (D75, the re-run of the D74 measurement on rev. 3.1): D75_refI is S23v_G4_r3 with NO named key, exactly as D74_refI.
    # The rev. 3.1 changes are in the BUILD, so a dictionary difference of any kind is refused.
    ("D75_refI drifting from S23v_G4_r3 in a run control",
     lambda c: _s16(c, "D75_refI", wall_budget_h=12.0),
     "FSF parent check: D75_refI differs from S23v_G4_r3 in wall_budget_h"),
    ("D75_refI drifting from S23v_G4_r3 in the end time",
     lambda c: _s16(c, "D75_refI", end_time_s=0.03),
     "FSF parent check: D75_refI differs from S23v_G4_r3 in end_time_s"),
    ("D75_refI drifting from S23v_G4_r3 in a physics key",
     lambda c: _s16(c, "D75_refI", k0p_m2=5e-13),
     "FSF parent check: D75_refI differs from S23v_G4_r3 in k0p_m2"),
    ("D75_refI pinned to a reference library (it would run without the passive field)",
     lambda c: _s16(c, "D75_refI", solver_library="s22"),
     "FSF parent check: D75_refI differs from S23v_G4_r3 in solver_library"),
    ("D75_refI with a frame key dropped (beyond its named keys)",
     lambda c: _s16(c, "D75_refI", frame_interval_s=None),
     "frame_interval_s"),
    ("S23v_G4_r3_S22 without its named key solver_library",
     lambda c: _s16(c, "S23v_G4_r3_S22", solver_library=None),
     "FSF parent check: S23v_G4_r3_S22 does not differ from S23v_G4_r3 in its named key solver_library"),
    ("S23v_G4_r3_S22 not ending at 30 ms (no end_time_s difference)",
     lambda c: _s16(c, "S23v_G4_r3_S22", end_time_s=0.04),
     "FSF parent check: S23v_G4_r3_S22 does not differ from S23v_G4_r3 in its named key end_time_s"),
    ("an unknown solver_library on the S23 reference child",
     lambda c: _s16(c, "S23v_G4_r3_S22", solver_library="s21"),
     "solver_library 's21' is not one of p2prime, s22, baseline (S22 section 7)."),
    # ---- rev. 4.1 (design section 13.2 items 4 and 5): R3 required on the S23 build ----
    # R1 (solver report 24): the marker's only value is tanDelta0.
    ("R1: a marker value other than tanDelta0 (sensitivityLimit tanDelta1)",
     lambda c: _s16(c, "S23v_G4_nor3", sensitivity_limit="tanDelta1"),
     "sensitivity_limit 'tanDelta1' is not one of tanDelta0"),
    ("R1: the marker value none (the solver refuses a PRESENT key that is not tanDelta0)",
     lambda c: _s16(c, "S23v_G4_nor3", sensitivity_limit="none"),
     "sensitivity_limit 'none' is not one of tanDelta0"),
    ("R1: a non-string marker",
     lambda c: _s16(c, "S23v_G4_nor3", sensitivity_limit=True),
     "sensitivity_limit True is not one of tanDelta0"),
    # R2: networkViscosity none (or absent) and not marked.
    ("R2: network_viscosity absent on the S23 build (the dictionary default is none)",
     lambda c: _s16(c, "S23v_G40", network_viscosity=None, tau_v_s=None, _tau_v_s_status=None),
     "network_viscosity none (or absent: the default is none) is refused on the S23 build"),
    ("R2: network_viscosity none, explicit, on a child other than S23v_G4_nor3",
     lambda c: _s16(c, "S23v_G40", network_viscosity="none", tau_v_s=None, _tau_v_s_status=None),
     "network_viscosity none (or absent: the default is none) is refused on the S23 build"),
    ("WHITELIST: the labelled sensitivity child without its marker",
     lambda c: _s16(c, "S23v_G4_nor3", sensitivity_limit=None),
     "S23v_G4_nor3: network_viscosity none on the S23 build needs the explicit marker"),
    ("WHITELIST: the marker on an R3-on child of the S23 build",
     lambda c: _s16(c, "S23v_G40", sensitivity_limit="tanDelta0"),
     "S23v_G40: sensitivity_limit is allowed only on S23v_G4_nor3"),
    ("WHITELIST: the marker on an R3-off child pinned to the S22 library",
     lambda c: _s16(c, "fsf_G4_kxu_nor3_40", sensitivity_limit="tanDelta0"),
     "fsf_G4_kxu_nor3_40: sensitivity_limit is allowed only on S23v_G4_nor3"),
    ("WHITELIST: the marker on the labelled child once it is pinned to another library (inert there)",
     lambda c: _s16(c, "S23v_G4_nor3", solver_library="s22"),
     "S23v_G4_nor3: sensitivity_limit is read only by the S23 solver"),
    # the CE's tightening: the marker labels the R3-off limit only, so it cannot waive the tau_v floor.
    ("MARKER + kelvinVoigt (tau_v 1.5 ms)",
     lambda c: _s16(c, "S23v_G4_nor3", network_viscosity="kelvinVoigt", tau_v_s=0.0015),
     "sensitivity_limit together with network_viscosity kelvinVoigt is refused"),
    ("MARKER + kelvinVoigt with a tiny tau_v (the floor must not be waived by the marker)",
     lambda c: _s16(c, "S23v_G4_nor3", network_viscosity="kelvinVoigt", tau_v_s=1e-9),
     "sensitivity_limit together with network_viscosity kelvinVoigt is refused"),
    # R3: kelvinVoigt without an explicit tau_v (the solver default 2.5e-4 s is below the floor).
    ("R3: kelvinVoigt without an explicit tau_v_s",
     lambda c: _s16(c, "S23v_G40", tau_v_s=None),
     "network_viscosity kelvinVoigt on the S23 build needs an explicit tau_v_s"),
    # R4: tau_v >= 5e-4 s. fsb_ctl_r3 has no BATCH parent pair, so it carries every tau_v here; it is
    # unpinned (solver_library None) because the floor applies to the S23 build only.
    ("R4: tau_v 4.99e-4 s (just below the floor)",
     lambda c: _s16(c, "fsb_ctl_r3", solver_library=None, tau_v_s=4.99e-4),
     "tau_v_s 0.000499 is below the 0.0005 s floor"),
    ("R4: the double just below 5e-4 (the comparison is >=, to the last bit)",
     lambda c: _s16(c, "fsb_ctl_r3", solver_library=None, tau_v_s=math.nextafter(5e-4, 0.0)),
     "is below the 0.0005 s floor"),
    ("R4: the solver default 2.5e-4 s written explicitly",
     lambda c: _s16(c, "fsb_ctl_r3", solver_library=None, tau_v_s=2.5e-4),
     "tau_v_s 0.00025 is below the 0.0005 s floor"),
    ("R4: tau_v 1e-6 s",
     lambda c: _s16(c, "fsb_ctl_r3", solver_library=None, tau_v_s=1e-6),
     "tau_v_s 1e-06 is below the 0.0005 s floor"),
    # S25 section 2F (time-unit scaling): time_scale is REQUIRED on every child, and the floor is the
    # solver's own timeScale x 5e-4 on the WRITTEN value. One negative per refusal.
    ("S25: a child without time_scale",
     lambda c: _s16(c, "P73_G4_r3_40", time_scale=None),
     "P73_G4_r3_40: time_scale is missing"),
    ("S25: time_scale 0",
     lambda c: _s16(c, "P73_G4_r3_40", time_scale=0),
     "time_scale 0 must be a finite number > 0"),
    ("S25: time_scale negative",
     lambda c: _s16(c, "P73_G4_r3_40", time_scale=-8.33),
     "time_scale -8.33 must be a finite number > 0"),
    ("S25: time_scale NaN",
     lambda c: _s16(c, "P73_G4_r3_40", time_scale=float("nan")),
     "time_scale nan must be a finite number > 0"),
    ("S25: time_scale a string",
     lambda c: _s16(c, "P73_G4_r3_40", time_scale="8.33"),
     "time_scale '8.33' must be a finite number > 0"),
    ("S25: time_scale true (a bool is not a number)",
     lambda c: _s16(c, "P73_G4_r3_40", time_scale=True),
     "time_scale True must be a finite number > 0"),
    ("S25: a non-unit time_scale on a child pinned to the s22 library (silently inert)",
     lambda c: _s16(c, "fsb_ctl_r3", time_scale=8.33),
     "that library predates S25 and never reads timeScale"),
    ("S25: tau_v 4.99e-4 s physical at time_scale 8.33 (the floor scales)",
     lambda c: _s16(c, "fsb_ctl_r3", solver_library=None, tau_v_s=4.99e-4, time_scale=8.33),
     "At time_scale 8.33 the written networkViscosityTau"),
    # nOuterCorrectors 1 (the end-of-step partition rule, section 13.6 hand-off)
    ("nOuterCorrectors 2 on an S23-build arm",
     lambda c: _s16(c, "S23v_G4_r3", n_outer_correctors=2),
     "n_outer_correctors 2 is refused on the S23 build"),
) + tuple(
    # item 5: each of the 13 pinned children, unpinned ALONE, is refused on the S23 build (the pin is
    # what lets it generate; fsb_ctl_r3 is the 13th, see its _why_solver_library)
    (f"PIN: {cid} without solver_library s22 is refused on the S23 build",
     (lambda cid_: (lambda c: _s16(c, cid_, solver_library=None)))(cid),
     ("network_viscosity kelvinVoigt on the S23 build needs an explicit tau_v_s" if cid == "fsb_ctl_r3"
      else "is refused on the S23 build: R3 is required"))
    for cid in S23_PINNED_S22
)


S23_SOLVER_C = Path(__file__).resolve().parents[3] / "user-applications" / "phaseClotAxisymmetricFluid" / "phaseClotAxisymmetricFluid.C"


# ---- THE Y-BIFURCATION GATE (BATCH 79; design 2026-10-14-case-y-bifurcation-gate.md rev. 2.2; case-engineer 12 and 13) ----
#
# One negative per generator refusal of the Y series' keys, module-level so each can be run alone
# (y_gate_generator_negatives(only=<label part>)): `y_gate` and `non_orthogonal_correctors` are read ONLY for the Y series, the mesh files are
# pinned by sha256, the keys that mean nothing on an unstructured mesh must be null, the write schedule is runTime tau1 with endTime 18 tau1 + [3 dt_est, 0.1 tau1]
# (rev. 2.2), `write_every_n_steps` (rev. 2.1) is read by nothing, and a Y child carries no run-control key of its own.
def _y_series(c: dict) -> dict:
    return c["series"]["s25_ygate"]


def _y_edit(c: dict, path: tuple, value) -> None:
    d = _y_series(c)
    for k in path[:-1]:
        d = d[k]
    if value is None:
        d.pop(path[-1], None)
    else:
        d[path[-1]] = value


Y_GATE_REFUSALS = (
    ("y_gate block on another series", lambda c: c["series"]["s25_5a"].update({"y_gate": _y_series(c)["y_gate"]}),
     "y_gate is read only for the Y series s25_ygate, s25_ys794, s25_ys850, s25_ys794_plug, s25_ys850_plug; here it would be silently inert"),
    ("non_orthogonal_correctors on another series", lambda c: c["series"]["s25_5a"].update({"non_orthogonal_correctors": 2}),
     "non_orthogonal_correctors is read only for the Y series s25_ygate, s25_ys794, s25_ys850, s25_ys794_plug, s25_ys850_plug; here it would be silently inert"),
    ("the Y series without non_orthogonal_correctors", lambda c: _y_edit(c, ("non_orthogonal_correctors",), None),
     "non_orthogonal_correctors is required by the Y series"),
    ("write_every_n_steps (the rev. 2.1 step schedule) on the Y child", lambda c: _s16(c, "B79_Y_GATE", write_every_n_steps=163),
     "write_every_n_steps is read by nothing"),
    ("write_every_n_steps on a non-Y child", lambda c: _s16(c, "S25_5a_tan", write_every_n_steps=163),
     "write_every_n_steps is read by nothing"),
    ("write_every_n_steps left in the gate-6 block", lambda c: _y_edit(c, ("y_gate", "gate6", "write_every_n_steps"), 163),
     "gate6.write_every_n_steps was the rev. 2.1 step schedule"),
    ("write_interval_s the rev. 2.1 time equivalent (163 dt_est) instead of tau1", lambda c: _y_edit(c, ("write_interval_s",), 0.0005083644),
     "write_interval_s 0.0005083644 != gate6.tau1_s 0.00050976"),
    ("end_time_s exactly 18 tau1 (no margin for the 18th write)", lambda c: _y_edit(c, ("end_time_s",), 0.00917568),
     "must be in [3 dt_est"),
    ("end_time_s the rev. 2.1 value 9.176e-3 (0.5 dt_est past 18 tau1)", lambda c: _y_edit(c, ("end_time_s",), 0.009176),
     "must be in [3 dt_est"),
    ("end_time_s 19 tau1 (a 19th write, a tail that buys nothing)", lambda c: _y_edit(c, ("end_time_s",), 0.00968544 + 2e-5),
     "must be in [3 dt_est"),
    ("a Y child with its own end_time_s", lambda c: _s16(c, "B79_Y_GATE", end_time_s=0.0092),
     "end_time_s is not a Y-child key"),
    ("a Y child with its own write_interval_s", lambda c: _s16(c, "B79_Y_GATE", write_interval_s=0.0005),
     "write_interval_s is not a Y-child key"),
    ("a Y child with a wall budget above one session", lambda c: _s16(c, "B79_Y_GATE", wall_budget_h=12.0),
     "needs wall_budget_h in (0, 11] h"),
    ("the dictionary is not the pinned file", lambda c: _y_edit(c, ("y_gate", "mesh", "blockMeshDict_sha256"), "0" * 64),
     "is not the pinned file"),
    ("a stale geometry key (nx) on an unstructured mesh", lambda c: _y_edit(c, ("geometry_override", "nx"), 448),
     "have no meaning on an unstructured mesh"),
    ("the expected cell count is not the dictionary's", lambda c: _y_edit(c, ("geometry_override", "expected_cell_count"), 78913),
     "expected_cell_count 78913 != the mesh's 78912"),
    ("the step-1 watcher on a clot-free run", lambda c: _y_edit(c, ("step1_watcher",), True),
     "step1_watcher must be off"),
    ("a velocity drive instead of the pressure drive", lambda c: _y_edit(c, ("pressure_driven",), False),
     "pressure_driven must be True"),
    ("a clot on the Y mesh", lambda c: _y_edit(c, ("clot_override", "shape"), "semisphere"),
     "clot_override.shape must be 'none'"),
    ("the outlet split does not add up", lambda c: _y_edit(c, ("y_gate", "mesh", "outlet_faces", "d1_y_positive"), 89),
     "does not add up to the outlet patch"),
    ("the cells across the daughters are not 100, 88, 60", lambda c: _y_edit(c, ("y_gate", "geometry_sim", "N"), [100, 88, 59]),
     "geometry_sim.N"),
    ("d2 at sqrt(0.3815) D0 instead of the pinned 0.618 D0", lambda c: _y_edit(c, ("y_gate", "geometry_sim", "widths_m"), [5.833333333333333e-05, 5.833333333333333e-05, 0.6177 * 5.833333333333333e-05]),
     "are not D0, d1 x D0, d2 x D0"),
    ("the drive's mean velocity is not centreline / 1.5 / (S_t / S_L)", lambda c: _y_edit(c, ("y_gate", "gate5", "u_mean_sim_m_s"), 0.00125),
     "the drive's u_mean_sim_m_s 0.00125 != centreline/1.5/(S_t/S_L)"),
    ("tau1 is not D0^2 / (pi^2 nu)", lambda c: _y_edit(c, ("y_gate", "gate6", "tau1_s"), 0.0005),
     "gate6.tau1_s 0.0005 != D0^2/(pi^2 nu)"),
    # case-engineer 41b: the output guard prices a Y child from its MEASURED write (22.694452 MB per 19,728 cells); 400 writes = 9.1 GB > 8.0. The channel constant (19.5 MB / 31,360 cells x 1.274)
    # priced the same run at 6.1 GB and ACCEPTED it, so this negative can only fire on the repriced guard.
    ("the Y plug series run to 400 ms (400 writes x 22.7 MB = 9.1 GB against the 8 GB budget)", lambda c: c["series"]["s25_ys794_plug"].update({"end_time_s": 0.40005}),
     "GB returned if this child SURVIVES"),
    ("a Y81 child with a blank role text", lambda c: _s16(c, "Y81_YS794_PLUG", role="  "),
     "must be a non-empty string"),
)


def y_gate_generator_negatives(only: str | None = None) -> None:
    """Each Y-series generator refusal, seen to fire for its stated reason on a fresh copy (all of them, or the one whose label contains `only`); then the positive
    side: the pristine controls generate, and the Y child's controlDict, fvSolution and child.json carry exactly the schedule the refusals protect."""
    import copy
    ran = 0
    with tempfile.TemporaryDirectory(prefix="am2020-y-neg-") as tmp:
        root = build_case(Path(tmp))
        shutil.rmtree(root / "runs")
        controls_path = root / "constant" / "mainCaseControls.json"
        pristine = controls_path.read_text()
        for label, mutate, expected in Y_GATE_REFUSALS:
            if only is not None and only not in label:
                continue
            broken = copy.deepcopy(json.loads(pristine))
            mutate(broken)
            assert broken != json.loads(pristine), f"{label}: the mutation changed nothing"
            controls_path.write_text(json.dumps(broken, indent=1))
            done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")], capture_output=True, text=True)
            said = (done.stdout + done.stderr).strip().splitlines()
            last = said[-1] if said else ""
            if done.returncode == 0:
                raise AssertionError(f"the generator ACCEPTED {label}; it must refuse")
            if expected not in last:
                raise AssertionError(f"the generator refused {label} for the WRONG reason: {last}")
            print(f"generator refuses: {label:<66} -> {last[:96]}")
            shutil.rmtree(root / "runs", ignore_errors=True)
            ran += 1
        controls_path.write_text(pristine)
        done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")], capture_output=True, text=True)
        assert done.returncode == 0, f"the pristine controls were refused: {(done.stdout + done.stderr)[-300:]}"
        y = root / "runs" / "B79_Y_GATE"
        cd = (y / "system" / "controlDict").read_text()
        top = cd.split("functions", 1)[0]
        assert re.findall(r"(?m)^writeControl\s+(\S+);", top) == ["runTime"] and re.findall(r"(?m)^writeInterval\s+(\S+);", top) == ["0.00050976"], top
        assert re.findall(r"(?m)^endTime\s+(\S+);", top) == ["0.009196"] and "adjustableRunTime" not in cd
        j = json.loads((y / "child.json").read_text())
        assert j["write_interval_s"] == 0.00050976 and j["end_time_s"] == 0.009196 and "write_every_n_steps" not in j and j["non_orthogonal_correctors"] == 2
        shutil.rmtree(root / "runs", ignore_errors=True)
    assert ran > 0, f"no Y-gate negative matches {only!r}"
    print(f"   {ran} Y-gate generator negatives refused; the pristine controls write runTime / tau1 / endTime 9.196e-3")


def s23_solver_would_refuse(props_text: str):
    """The constructor's R1-R4 and marker-with-kelvinVoigt FatalErrors, read off phaseClotProperties text the way
    lookupOrDefault reads it (solver-engineer report 24; phaseClotAxisymmetricFluid.C, 'S23 rev. 4.1 section 13.2
    item 4'). Returns None (accepted) or 'R1' (marker not tanDelta0), 'MK' (marker with a non-none networkViscosity),
    'R2' (none, unmarked), 'R3' (kelvinVoigt without networkViscosityTau), 'R4' (tau_v < timeScale*5e-4 or not a number),
    'TS' (S25: timeScale absent, not a number or not > 0), 'CAP' (S25 G: depositionContentCap, or phiSMax when it is
    not named, not finite or not in (0, 1))."""
    text = re.sub(r"//[^\n]*", "", props_text)

    def entry(key):
        m = re.search(rf"(?m)^\s*{key}\s+([^;\s]+)\s*;", text)
        return None if m is None else m.group(1)
    marker, nv, tau = entry("sensitivityLimit"), entry("networkViscosity") or "none", entry("networkViscosityTau")
    # S25 section 2F: timeScale has NO default and the solver FatalErrors without it wherever a physical-time
    # consumer is active; networkRupture fibreRelease is built in, so on every dictionary. The floor is
    # timeScale x 5e-4 (the solver's own comparison, formed the way the generator forms the written value).
    ts_text = entry("timeScale")
    try:
        ts = float(ts_text) if ts_text is not None else None
    except ValueError:
        ts = None
    if ts is None or not (ts > 0 and ts < float("inf")):
        return "TS"
    if marker is not None and marker != "tanDelta0":
        return "R1"
    marked = marker == "tanDelta0"
    if marked and nv != "none":
        return "MK"
    if nv == "none" and not marked:
        return "R2"
    if nv == "kelvinVoigt" and not marked:
        if tau is None:
            return "R3"
        if not (float(tau) >= ts * 5e-4):
            return "R4"
    # S25 build 1 part G (solver .C, after the S23 refusals): the acceptance cap is a packing fraction. It follows
    # phiSMax (default 0.6) unless named, and must be finite and in (0, 1). A value the solver cannot even parse is
    # refused too (by the dictionary reader, earlier): the mirror reports it as CAP.
    cap_text = entry("depositionContentCap")
    try:
        cap = float(cap_text if cap_text is not None else (entry("phiSMax") or "0.6"))
    except ValueError:
        return "CAP"
    if not (math.isfinite(cap) and 0 < cap < 1):
        return "CAP"
    return None


def s23_solver_mirror_checks() -> None:
    """(a) The mirror above is pinned to the solver's SOURCE (the floor literal, the comparison, the marker word and key,
    the default); (b) report 24's negative dictionaries, one each, and the accepted ones, run through the mirror."""
    assert S23_SOLVER_C.is_file(), f"the solver source is absent: {S23_SOLVER_C}"
    src = S23_SOLVER_C.read_text()
    # S25 section 2F: the floor is timeScale_*5e-4, and timeScale has no default (required on every dictionary:
    # networkRupture fibreRelease is built in), finite and > 0.
    pins = ('const scalar s23TauVFloor = timeScale_*5e-4;', '!(networkViscosityTau_ >= s23TauVFloor)',
            '!props.found("timeScale")', '!(std::isfinite(ts) && ts > 0)',
            'sensitivityLimit_ != "tanDelta0"', 'lookupOrDefault<word>("sensitivityLimit", "none")',
            'lookupOrDefault<word>("networkViscosity", "none")', '!phaseClotProperties_.found("networkViscosityTau")',
            'if (s23Marked && networkViscosity_ != "none")', 'if (networkViscosity_ == "none" && !s23Marked)')
    stale = lambda s: [pin for pin in pins if s.count(pin) != 1]
    assert not stale(src), f"the solver source no longer has exactly one of {stale(src)}: the dry-run mirror is stale"
    # the pin check can fail: each drift of the solver the mirror would not follow is seen (a scratch copy of the text)
    for what, drifted in (("the floor 4.99e-4", src.replace('s23TauVFloor = timeScale_*5e-4;', 's23TauVFloor = timeScale_*4.99e-4;')),
                          ("the floor no longer scaled", src.replace('s23TauVFloor = timeScale_*5e-4;', 's23TauVFloor = 5e-4;')),
                          ("timeScale given a default", src.replace('!props.found("timeScale")', 'false')),
                          ("timeScale allowed to be <= 0", src.replace('!(std::isfinite(ts) && ts > 0)', 'false')),
                          ("> instead of >=", src.replace('networkViscosityTau_ >= s23TauVFloor', 'networkViscosityTau_ > s23TauVFloor')),
                          ("a second marker value", src.replace('sensitivityLimit_ != "tanDelta0"', 'sensitivityLimit_ != "tanDelta1"')),
                          ("the marker allowed with kelvinVoigt", src.replace('if (s23Marked && networkViscosity_ != "none")', 'if (false)')),
                          ("the R3-required refusal removed", src.replace('if (networkViscosity_ == "none" && !s23Marked)', 'if (false)'))):
        assert drifted != src and stale(drifted), f"the pin check did not see {what}"
    gen = (CASE / "scripts" / "create_cases.py").read_text()
    assert "S23_TAU_V_FLOOR_S = 5e-4" in gen and 'S23_SENSITIVITY_VALUES = ("tanDelta0",)' in gen
    base = ("networkViscosity kelvinVoigt;\nnetworkViscosityTau 0.0015;\n")
    cases_ts1 = (
        ("accepted: kelvinVoigt 1.5e-3", base, None),
        ("accepted: kelvinVoigt exactly 5e-4", "networkViscosity kelvinVoigt;\nnetworkViscosityTau 5e-4;\n", None),
        ("accepted: kelvinVoigt exactly 0.0005", "networkViscosity kelvinVoigt;\nnetworkViscosityTau 0.0005;\n", None),
        ("accepted: none with the marker", "networkViscosity none;\nsensitivityLimit tanDelta0;\n", None),
        ("accepted: the marker with networkViscosity ABSENT (the default is none)", "sensitivityLimit tanDelta0;\n", None),
        ("R1: sensitivityLimit tanDelta1", "networkViscosity none;\nsensitivityLimit tanDelta1;\n", "R1"),
        ("R1: sensitivityLimit none (present)", "networkViscosity none;\nsensitivityLimit none;\n", "R1"),
        ("R2: the key absent", "", "R2"),
        ("R2: networkViscosity none, unmarked", "networkViscosity none;\n", "R2"),
        ("R3: kelvinVoigt only", "networkViscosity kelvinVoigt;\n", "R3"),
        ("R4: tau 4.99e-4", "networkViscosity kelvinVoigt;\nnetworkViscosityTau 4.99e-4;\n", "R4"),
        ("R4: tau 2.5e-4 (the default, written)", "networkViscosity kelvinVoigt;\nnetworkViscosityTau 2.5e-4;\n", "R4"),
        ("R4: tau NaN", "networkViscosity kelvinVoigt;\nnetworkViscosityTau nan;\n", "R4"),
        ("MK: the marker with kelvinVoigt 1.5e-3", base + "sensitivityLimit tanDelta0;\n", "MK"),
        ("MK: the marker with kelvinVoigt and a tiny tau (must not waive the floor)",
         "networkViscosity kelvinVoigt;\nnetworkViscosityTau 1e-9;\nsensitivityLimit tanDelta0;\n", "MK"),
    )
    # report 24's dictionaries all state timeScale 1.0 (the legacy value); the S25 cases follow
    cases = tuple((label, "timeScale 1.0;\n" + text, want) for label, text, want in cases_ts1) + (
        ("accepted: timeScale 8.33, tau = 8.33 x 5e-4 exactly (the floor, to the last bit)",
         f"timeScale 8.33;\nnetworkViscosity kelvinVoigt;\nnetworkViscosityTau {8.33 * 5e-4!r};\n", None),
        ("accepted: timeScale 8.33, tau = 8.33 x 1.5e-3",
         f"timeScale 8.33;\nnetworkViscosity kelvinVoigt;\nnetworkViscosityTau {8.33 * 1.5e-3!r};\n", None),
        ("R4: timeScale 8.33, tau 5e-4 (the PHYSICAL floor value, unscaled)",
         "timeScale 8.33;\nnetworkViscosity kelvinVoigt;\nnetworkViscosityTau 5e-4;\n", "R4"),
        ("R4: timeScale 8.33, tau 8.33 x 4.99e-4",
         f"timeScale 8.33;\nnetworkViscosity kelvinVoigt;\nnetworkViscosityTau {8.33 * 4.99e-4!r};\n", "R4"),
        ("TS: timeScale absent", base, "TS"),
        ("TS: timeScale 0", "timeScale 0;\n" + base, "TS"),
        ("TS: timeScale -1", "timeScale -1;\n" + base, "TS"),
        ("TS: timeScale nan", "timeScale nan;\n" + base, "TS"),
    )
    for label, text, want in cases:
        got = s23_solver_would_refuse(text)
        assert got == want, f"{label}: the mirror says {got!r}, report 24 / S25 2F says {want!r}"
    print(f"   the solver's R1-R4, TS and marker+kelvinVoigt mirror is pinned to its source ({len(cases)} dictionaries, both sides, "
          f"5e-4 and 0.0005 accepted, 4.99e-4 refused, the floor scaled by timeScale)")


# ---------------------------------------------------------------------------------------------------------------------
# S25 BUILD 1 (solver-engineer report 05; case-engineer report 08): the dry-run side of the new solver refusals, the
# gammaLow/gammaHigh scaling of design section 12 item 3, and the assessor's report-only S25 readers.
#
# There is no OpenFOAM here, so a solver refusal is shown the way s23_solver_mirror_checks shows the S23 ones: a Python
# MIRROR of the predicate, PINNED to the solver's source text (each pin occurs exactly once in the whitespace-squashed
# source), the pin check seen to FAIL when the guard is deleted from a scratch copy of the text, and the mirror run
# over dictionaries / directories on both sides of every boundary. Each section runs alone:
#   python3 -B -c "import dry_run; dry_run.s25_build1_checks('<section>')"   (sections: cap, state, dbound, gamma, assess)
# ---------------------------------------------------------------------------------------------------------------------
def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _s25_drift(src: str, pins, drifts, what: str) -> None:
    """The pins hold on `src`, and each drift (a scratch copy with ONE guard changed or deleted) breaks at least the
    pin it changed: a pin check that cannot fail would be a mirror pinned to nothing."""
    sq = _squash(src)
    stale = lambda t: [pin for pin in pins if t.count(pin) != 1]
    assert not stale(sq), f"{what}: the solver source no longer has exactly one of {stale(sq)}: the mirror is stale"
    for label, old, new in drifts:
        assert sq.count(old) == 1, f"{what}: the drift {label!r} anchors on text that is not unique: {old[:60]!r}"
        drifted = sq.replace(old, new)
        assert drifted != sq and stale(drifted), f"{what}: the pin check did not see: {label}"


def s25_state_keys(src: str) -> dict:
    """The keys of uniform/phaseClotState DERIVED from the solver (the three tables restoreRestartState checks, the
    d.set() keys storeRestartState writes and the d.lookup<>() keys it reads): nothing restated here."""
    tables = []
    for name in ("s25StateScalars", "s25StateFields", "s25StateLabels"):
        m = re.search(rf"{name}\[\]\s*=\s*\{{(.*?)\}};", src, re.S)
        assert m, f"the solver has no {name} table"
        tables.append(re.findall(r'"([^"]+)"', m.group(1)))

    def body(fn: str) -> str:
        m = re.search(rf"void Foam::solvers::phaseClotAxisymmetricFluid::{fn}\(\)\n\{{(.*?)\n\}}\n", src, re.S)
        assert m, f"no {fn} body"
        return m.group(1)
    return {"tables": [k for t in tables for k in t],
            "written": re.findall(r'\bd\.set\("([^"]+)"', body("storeRestartState")),
            "read": re.findall(r'\bd\.lookup<[^>]+>\("([^"]+)"\)', body("restoreRestartState"))}


def s25_state_would_refuse(state_text, keys):
    """restoreRestartState on the text of `<time>/uniform/phaseClotState` (None = no file). None = a cold seed or a full
    restore; else the sorted keys the all-or-none refusal names. An IOdictionary with only its FoamFile header is empty()."""
    if state_text is None:
        return None
    body = re.sub(r"FoamFile\s*\{.*?\}", "", re.sub(r"//[^\n]*", "", state_text), flags=re.S)
    present = set(re.findall(r"(?m)^([A-Za-z_]\w*)\s", body)) - {"nan", "inf"}   # list continuation lines are numbers
    if not present:
        return None
    missing = sorted(k for k in keys if k not in present)
    return missing or None


def s25_restart_dir_would_refuse(files, state_text, keys):
    """The constructor's pending-contents all-or-none (thetaPendingB xor thetaPendingP), then restoreRestartState."""
    if ("thetaPendingB" in files) != ("thetaPendingP" in files):
        return "PENDING"
    missing = s25_state_would_refuse(state_text, keys)
    return None if missing is None else "STATE:" + ",".join(missing)


def _s25_state_text(keys, drop=()):
    lines = ["FoamFile", "{", "    format ascii;", "    class dictionary;", f'    location "0.04/uniform";',
             "    object phaseClotState;", "}", ""]
    for i, k in enumerate(keys):
        if k in drop:
            continue
        lines.append(f"{k}  List<scalar> \n3\n(\n1.5\n-nan\n{i}\n)\n;" if k.startswith("s17") and "Steps" not in k
                     else f"{k}  {i + 1};")
    return "\n".join(lines) + "\n"


def _s25_cap_section(src: str) -> None:
    pins = ('phaseClotProperties_.lookupOrDefault<scalar>("phiSMax", 0.6)',
            'depositionContentCap_ ( phaseClotProperties_.lookupOrDefault<scalar> ( "depositionContentCap", phiSMax_ ) ),',
            'if ( !( std::isfinite(depositionContentCap_) && depositionContentCap_ > 0 && depositionContentCap_ < 1 ) )')
    guard = pins[2]
    drifts = (
        ("the guard deleted", guard, "if ( false )"),
        ("the upper bound allowed to reach 1", "depositionContentCap_ < 1 )", "depositionContentCap_ <= 1 )"),
        ("the lower bound allowed to reach 0", "depositionContentCap_ > 0 &&", "depositionContentCap_ >= 0 &&"),
        ("the finiteness test dropped", "std::isfinite(depositionContentCap_) &&", "true &&"),
        ("the default no longer follows phiSMax", '"depositionContentCap", phiSMax_', '"depositionContentCap", 0.5'),
        ("phiSMax's own default moved", '("phiSMax", 0.6)', '("phiSMax", 0.65)'))
    _s25_drift(src, pins, drifts, "part G")
    base = "timeScale 1.0;\nnetworkViscosity kelvinVoigt;\nnetworkViscosityTau 0.0015;\n"
    cases = (
        ("accepted: the key absent (follows phiSMax's default 0.6)", "", None),
        ("accepted: 0.6 (the shipped value)", "depositionContentCap 0.6;\n", None),
        ("accepted: 0.999999", "depositionContentCap 0.999999;\n", None),
        ("accepted: 1e-6", "depositionContentCap 1e-6;\n", None),
        ("accepted: phiSMax 0.7 alone (the cap follows it)", "phiSMax 0.7;\n", None),
        ("accepted: phiSMax 1.2 with a cap of 0.5 (this predicate judges the cap)", "phiSMax 1.2;\ndepositionContentCap 0.5;\n", None),
        ("CAP: 0", "depositionContentCap 0;\n", "CAP"),
        ("CAP: 1 (the pole of a packing fraction)", "depositionContentCap 1;\n", "CAP"),
        ("CAP: 1.0000001", "depositionContentCap 1.0000001;\n", "CAP"),
        ("CAP: -0.1", "depositionContentCap -0.1;\n", "CAP"),
        ("CAP: nan", "depositionContentCap nan;\n", "CAP"),
        ("CAP: inf", "depositionContentCap inf;\n", "CAP"),
        ("CAP: not a number", "depositionContentCap abc;\n", "CAP"),
        ("CAP: phiSMax 1.2 and the key absent (the default follows phiSMax out of range)", "phiSMax 1.2;\n", "CAP"),
        ("CAP: phiSMax 0 and the key absent", "phiSMax 0;\n", "CAP"),
    )
    for label, text, want in cases:
        got = s23_solver_would_refuse(base + text)
        assert got == want, f"{label}: the mirror says {got!r}, S25 G says {want!r}"
    # TS is the first refusal in the solver (member initialiser), so a missing timeScale wins over a bad cap
    assert s23_solver_would_refuse(base.replace("timeScale 1.0;\n", "") + "depositionContentCap 2;\n") == "TS"
    print(f"   cap: the mirror is pinned to the solver ({len(drifts)} drifts seen, {len(cases)} dictionaries, both sides of 0 and 1)")


def _s25_state_section(src: str) -> None:
    k = s25_state_keys(src)
    keys = k["tables"]
    assert len(keys) == 16 and len(set(keys)) == 16, keys
    # the WRITER and the READER agree with the checked tables: a drift between them would make every restart fatal
    assert sorted(k["written"]) == sorted(keys), (sorted(set(k["written"]) ^ set(keys)))
    assert sorted(k["read"]) == sorted(keys), (sorted(set(k["read"]) ^ set(keys)))
    pins = ("if (d.empty())", "if (missing.size())", "if (hasPB != hasPP)",
            "for (const char* k : s25StateScalars)", "for (const char* k : s25StateFields)",
            "for (const char* k : s25StateLabels)")
    drifts = (
        ("the all-or-none refusal deleted", "if (missing.size())", "if (false)"),
        ("a seed taken from an EMPTY file removed", "if (d.empty())", "if (false)"),
        ("the scalars not checked", "for (const char* k : s25StateScalars)", "for (const char* k : s25StateLabels)"),
        ("the fields not checked", "for (const char* k : s25StateFields)", "for (const char* k : s25StateScalars)"),
        ("the labels not checked", "for (const char* k : s25StateLabels)", "for (const char* k : s25StateFields)"),
        ("the pending-contents refusal deleted", "if (hasPB != hasPP)", "if (false)"))
    _s25_drift(src, pins, drifts, "part C")
    # writer/reader drift is seen too: rename one written key in a scratch copy of the text
    bad = src.replace('d.set("s23ContentV0", s23ContentV0_);', 'd.set("s23ContentV1", s23ContentV0_);')
    assert bad != src and sorted(s25_state_keys(bad)["written"]) != sorted(s25_state_keys(bad)["tables"]), \
        "a renamed writer key was not seen"
    full = _s25_state_text(keys)
    assert s25_state_would_refuse(full, keys) is None
    assert s25_state_would_refuse(None, keys) is None
    assert s25_state_would_refuse(_s25_state_text(keys, drop=keys), keys) is None, "a header-only file is a cold seed"
    for key in keys:
        got = s25_state_would_refuse(_s25_state_text(keys, drop=(key,)), keys)
        assert got == [key], (key, got)
    half = _s25_state_text(keys, drop=keys[8:])
    assert s25_state_would_refuse(half, keys) == sorted(keys[8:]), "a half-written file"
    assert s25_state_would_refuse(_s25_state_text(["someOtherKey"]), keys) == sorted(keys), "an unknown schema is not empty"
    both, one_b, one_p, none_ = {"thetaPendingB", "thetaPendingP"}, {"thetaPendingB"}, {"thetaPendingP"}, set()
    assert s25_restart_dir_would_refuse(both, full, keys) is None
    assert s25_restart_dir_would_refuse(none_, None, keys) is None
    assert s25_restart_dir_would_refuse(one_b, full, keys) == "PENDING"
    assert s25_restart_dir_would_refuse(one_p, None, keys) == "PENDING"
    assert s25_restart_dir_would_refuse(both, _s25_state_text(keys, drop=("s18pWindow1",)), keys) == "STATE:s18pWindow1"
    print(f"   state: {len(keys)} keys derived from the solver (tables == written == read); {len(keys)} single-key drops, a "
          f"half file, an unknown schema, thetaPendingB xor P (both ways) refused; {len(drifts)} drifts seen; empty/absent seed")


def s25_cell_det_ra(AU, AV, Kd, Kres, r3, phiF, phiS):
    """The block bound's cell algebra (solver .C, 'S25 D'): the 2x2 [[AU+Kd, -Kd], [-(Kd+Kres), AV+Kd+Kres+r3]], its
    determinant and the projected mixture mobility rA. (det, rA) with rA None when det <= 0 (the first refusal)."""
    A, B, C, D = AU + Kd, -Kd, -(Kd + Kres), AV + Kd + Kres + r3
    det = A * D - B * C
    if not det > 0:
        return det, None
    i00, i01, i10, i11 = D / det, -B / det, -C / det, A / det
    aS = i10 * phiF + i11 * phiS
    return det, phiF * (i00 * phiF + i01 * phiS) + phiS * aS


def _s25_dbound_section(src: str, root: Path = None) -> None:
    import random
    _s25_drift(src, ("if (!(detS25 > 0))", "if (!(rAS25 > 0))"),
               (("the det guard deleted", "if (!(detS25 > 0))", "if (false)"),
                ("the rA guard deleted", "if (!(rAS25 > 0))", "if (false)"),
                ("the det guard let a zero through", "if (!(detS25 > 0))", "if (!(detS25 >= 0))"),
                ("the rA guard let a zero through", "if (!(rAS25 > 0))", "if (!(rAS25 >= 0))")), "part D")
    rnd = random.Random(2510)
    pos = neg = fired = 0
    for _ in range(20000):
        AU, AV = 10 ** rnd.uniform(-3, 6), 10 ** rnd.uniform(-3, 6)
        Kd, Kres, r3 = (rnd.choice((0.0, 10 ** rnd.uniform(-6, 8))) for _ in range(3))
        phiS = rnd.uniform(1e-9, 1.0)
        phiF = rnd.choice((0.0, 1.0 - phiS, rnd.uniform(0, 1)))
        det, rA = s25_cell_det_ra(AU, AV, Kd, Kres, r3, phiF, phiS)
        # the closed forms: det = AU*(AV+Kd+Kres+r3) + Kd*(AV+r3);  rA*det = phiF^2 D + phiF phiS (2Kd+Kres) + phiS^2 A
        D, A = AV + Kd + Kres + r3, AU + Kd
        # the direct A*D - B*C cancels: its absolute error is a few ulp of A*D, not a fraction of det
        assert det > 0 and abs(det - (AU * D + Kd * (AV + r3))) <= 8 * 2.2e-16 * A * D, (det, AU, AV, Kd, Kres, r3)
        want = (phiF ** 2 * D + phiF * phiS * (2 * Kd + Kres) + phiS ** 2 * A) / det
        assert rA is not None and rA > 0 and abs(rA - want) <= 1e-6 * want, (rA, want)
        pos += 1
    # the guard can fire: a NEGATIVE drag (what a sign-flipped k0p, nu or etaRes would make) or a non-positive diagonal
    for _ in range(20000):
        AU, AV = 10 ** rnd.uniform(-3, 6), 10 ** rnd.uniform(-3, 6)
        Kd = -10 ** rnd.uniform(-3, 8)
        det, rA = s25_cell_det_ra(AU, AV, Kd, 0.0, 0.0, 0.5, 0.5)
        neg += 1
        fired += (rA is None)
    assert fired > 0, "no negative-drag cell reached det <= 0: the guard would be unreachable and the negative vacuous"
    assert s25_cell_det_ra(-1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5)[1] is None, "a non-positive U diagonal must trip it"
    # A GENERATED dictionary cannot take the drag negative: nu, k0p and etaRes (the three the drag is built from) are positive
    own = root is None
    tmp = tempfile.TemporaryDirectory(prefix="am2020-s25-dbound-") if own else None
    try:
        root = Path(tmp.name) / "r" if own else root
        if own:
            root = build_case(Path(tmp.name))
        n = 0
        for props in sorted((root / "runs").glob("*/constant/phaseClotProperties")):
            text = re.sub(r"//[^\n]*", "", props.read_text())
            for key in ("nu", "k0p", "etaRes", "k0pcore"):
                m = re.search(rf"(?m)^\s*{key}\s+(?:\[[^\]]*\]\s*)?([^;\s]+)\s*;", text)
                if m is not None:
                    assert float(m.group(1)) > 0, (props.parent.parent.name, key, m.group(1))
            n += 1
        assert n == KEPT_CHILDREN, n
    finally:
        if tmp is not None:
            tmp.cleanup()
    print(f"   dbound: {pos} positive-drag cells all have det > 0 and rA > 0 (closed forms agree to the cancellation of A*D - B*C); {fired} of {neg} "
          f"negative-drag cells reach det <= 0 (the guard fires); nu/k0p/etaRes > 0 in all {n} generated dictionaries, so a "
          f"generated dictionary cannot reach it")


def _s25_generate(patches=(), time_scale=None, child="fsb_ctl_r3"):
    """Generate ONE child in a scratch copy of the case, optionally with its source patched (old -> new, each anchor
    unique) and at time_scale; returns (returncode, last line, dictionary text or None)."""
    with tempfile.TemporaryDirectory(prefix="am2020-s25-gamma-") as tmp:
        root = build_case(Path(tmp))
        shutil.rmtree(root / "runs")
        if time_scale is not None:
            cp = root / "constant" / "mainCaseControls.json"
            c = json.loads(cp.read_text())
            _s16(c, child, solver_library=None, tau_v_s=5e-4, time_scale=time_scale)
            cp.write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n")
        gp = root / "scripts" / "create_cases.py"
        g = gp.read_text()
        for old, new in patches:
            assert g.count(old) == 1, f"generator patch anchor is not unique: {old[:70]!r}"
            g = g.replace(old, new)
        gp.write_text(g)
        done = subprocess.run([sys.executable, "-B", str(gp), child], capture_output=True, text=True)
        lines = (done.stdout + done.stderr).strip().splitlines()
        pf = root / "runs" / child / "constant" / "phaseClotProperties"
        return done.returncode, (lines[-1] if lines else ""), (pf.read_text() if pf.is_file() else None)


def _s25_gamma_section() -> None:
    low_scaled = '"gammaLow": s25_gamma_written(growth["gammaLow_1_s"], _time_scale),'
    high_scaled = '"gammaHigh": s25_gamma_written(growth["gammaHigh_1_s"], _time_scale),'
    guard_call = 's25_gamma_refusal(props, child["child_id"], growth, _time_scale)'
    ST = 8.33
    # the positive side: written as phys/S_t, to the last bit; S_t = 1 leaves the controls value untouched
    rc, last, props = _s25_generate(time_scale=ST)
    assert rc == 0, last
    ctl = json.loads((CASE / "constant" / "mainCaseControls.json").read_text())["growth_parameters"]
    lo, hi = ctl["gammaLow_1_s"], ctl["gammaHigh_1_s"]
    for line in (f"gammaLow {lo / ST!r};", f"gammaHigh {hi / ST!r};"):
        assert re.search(rf"(?m)^{re.escape(line)}$", props), f"{line!r} not written at time_scale {ST}"
    rc, last, props = _s25_generate()
    assert rc == 0, last
    for line in (f"gammaLow {lo!r};", f"gammaHigh {hi!r};"):
        assert re.search(rf"(?m)^{re.escape(line)}$", props), f"{line!r} not written at time_scale 1.0"
    # the refusal: a writer that leaves either unscaled (or scales only one) is refused, for ITS reason
    for label, patches, want in (
            ("gammaLow written unscaled at S_t 8.33", ((low_scaled, '"gammaLow": growth["gammaLow_1_s"],'),), "gammaLow"),
            ("gammaHigh written unscaled at S_t 8.33", ((high_scaled, '"gammaHigh": growth["gammaHigh_1_s"],'),), "gammaHigh"),
            ("gammaLow multiplied by S_t instead of divided",
             ((low_scaled, '"gammaLow": growth["gammaLow_1_s"] * _time_scale,'),), "gammaLow")):
        rc, last, _ = _s25_generate(patches=patches, time_scale=ST)
        assert rc != 0, f"the generator ACCEPTED {label}"
        assert want in last and "divided by time_scale 8.33" in last, f"{label}: refused for the wrong reason: {last}"
        print(f"generator refuses: {label:<52} -> {last[:100]}")
    # the same defect with the GUARD REMOVED is accepted: the refusal above is the guard's, not a side effect
    rc, last, props = _s25_generate(patches=((low_scaled, '"gammaLow": growth["gammaLow_1_s"],'), (guard_call, "pass")),
                                    time_scale=ST)
    assert rc == 0 and re.search(rf"(?m)^gammaLow {lo!r};$", props), (rc, last)
    # and at S_t = 1 the unscaled writer is the right one (x/1.0 == x), so it is not a refusal
    rc, last, _ = _s25_generate(patches=((low_scaled, '"gammaLow": growth["gammaLow_1_s"],'),))
    assert rc == 0, last
    print(f"   gamma: phys/S_t written exactly at S_t {ST} and unchanged at 1.0; 3 unscaled writers refused; the same writer "
          f"accepted once the guard call is removed")


S25_ASSESS_SAMPLE = (
    "PHASE_CLOT_S25_CONFIG timeScale=1 networkReleaseTime=0.00033 networkViscosityTau=0.00025 tauVFloor=0.0005 "
    "depositionContentCap=0.6 plateletTransport=deviationR1 restartState=persisted\n"
    "PHASE_CLOT_FREEPLATELET_CONFIG Deff=1e-10 Y=0.5 vP=0.5 yieldModel=independent-Y depositionContentCap=0.6\n"
    "PHASE_CLOT_S25_RESTART state=seeded\n"
    "PHASE_CLOT_S25_R1 seed=eFromW eAbsMax=0\n"
    "Time = 1e-06s\n"
    "PHASE_CLOT_DT step=1 time=1e-06 deltaT=1e-06 dtFluidCo=2e-06 dtWave=1e+300 dtKV=1e+300 dtCarrier=1.5e-06 "
    "dtNetDiff=1e-06 netDiffForm=block dtMax=1e-05 dtGrowth=1e+300 binder=netDiff threads=1 netDiffCell=17 "
    "netDiffIAS=0.25 netDiffR3As=3 netDiffW=0.5 netDiffA=0.62\n"
    "PHASE_CLOT_LAMBDA time=1e-06 step=1 G=1000 clotCells=40 lambdaClot=0.5 lambdaDom=1.5\n"
    "PHASE_CLOT_S25_R1 time=1e-06 eAbsMax=0.01 depositBooked=1e-12\n"
    "PHASE_CLOT_FREEPLATELET time=1e-06 step=1 contentOverCapCells=0 contentOverCapMax=-0.1 Y=0.5\n"
    "Time = 2e-06s\n"
    "PHASE_CLOT_DT step=2 time=2e-06 deltaT=5e-07 dtFluidCo=2e-06 dtWave=1e+300 dtKV=1e+300 dtCarrier=1.5e-06 "
    "dtNetDiff=1e+300 netDiffForm=block dtMax=1e-05 dtGrowth=1e+300 binder=writeAlign threads=1 netDiffCell=-1 "
    "netDiffIAS=0 netDiffR3As=0 netDiffW=0 netDiffA=0\n"
    "PHASE_CLOT_LAMBDA time=2e-06 step=2 G=1000 clotCells=0 lambdaClot=none lambdaDom=none\n"
    "PHASE_CLOT_S25_R1 time=2e-06 eAbsMax=0.03 depositBooked=4e-12\n"
    "PHASE_CLOT_FREEPLATELET time=2e-06 step=2 contentOverCapCells=7 contentOverCapMax=0.02 Y=0.5\n"
)


def _s25_assess_section(src: str) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("assess_case_s25", Path(__file__).resolve().parent / "assess_case.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    # the sample's keys are the solver's: each key the reader needs is printed by the solver's Info<< (a renamed key
    # would turn every field None, which the report would show as "no data" rather than as a failure)
    sq = _squash(src)
    for needle in ('"PHASE_CLOT_DT step="', '" dtFluidCo="', '" dtWave="', '" dtKV="', '" dtCarrier="', '" dtNetDiff="',
                   '" dtMax="', '" dtGrowth="', '" netDiffForm="', '" binder="', '" threads="',
                   '"PHASE_CLOT_LAMBDA time="', '" lambdaClot="', '" lambdaDom="', '" clotCells="', '" G="',
                   '"PHASE_CLOT_S25_R1 seed="', '"PHASE_CLOT_S25_R1 time="', '" eAbsMax="', '" depositBooked="',
                   '" contentOverCapCells="', '" contentOverCapMax="', '" depositionContentCap="',
                   '"PHASE_CLOT_S25_CONFIG"', '"PHASE_CLOT_S25_RESTART state=seeded"'):
        assert sq.count(needle) >= 1, f"the solver no longer prints {needle}: the assessor's S25 reader is stale"
    r = ac.s25_report(S25_ASSESS_SAMPLE)
    assert r["config"]["timeScale"] == "1" and r["config"]["depositionContentCap"] == "0.6", r["config"]
    assert r["restart_state"] == "seeded"
    d = r["dt"]
    assert d["records"] == 2 and d["binder_counts"] == {"netDiff": 1, "writeAlign": 1}, d
    assert d["net_diff_forms"] == ["block"] and d["threads"] == [1] and d["net_diff_binds_steps"] == 1
    assert d["min"]["deltaT"] == 5e-07 and d["min"]["dtFluidCo"] == 2e-06 and d["min"]["dtNetDiff"] == 1e-06, d["min"]
    assert d["min"]["dtWave"] is None and d["min"]["dtKV"] is None and d["min"]["dtGrowth"] is None, "great is no limit"
    lam = r["lambda"]
    assert lam["records"] == 2 and lam["clot_max"] == 0.5 and lam["dom_max"] == 1.5 and lam["G_last"] == 1000.0, lam
    assert lam["clot_none_records"] == 1 and lam["dom_none_records"] == 1 and lam["clot_last"] == 0.5, lam
    r1 = r["r1"]
    assert r1["seed"] == "eFromW" and r1["seed_e_abs_max"] == 0.0 and r1["records"] == 2, r1
    assert r1["e_abs_max_max"] == 0.03 and r1["e_abs_max_last"] == 0.03 and r1["deposit_booked_last"] == 4e-12, r1
    cc = r["content_over_cap"]
    assert cc == {"records": 2, "deposition_content_cap": 0.6, "cells_max": 7.0, "records_with_cells": 1,
                  "excess_max": 0.02}, cc
    # an older binary: the same shape, with nothing in it
    old = ac.s25_report("Time = 1s\nPHASE_CLOT_FREEPLATELET time=1 step=1 Y=0.5\n")
    assert old["config"] is None and old["dt"]["records"] == 0 and old["lambda"]["records"] == 0
    assert old["r1"]["seed"] is None and old["content_over_cap"]["records"] == 0 and old["content_over_cap"]["cells_max"] is None
    # a truncated token is not a number (the _numeric_kv rule): the field is skipped, not read as a prefix
    trunc = ac.s25_report("PHASE_CLOT_S25_R1 time=1 eAbsMax=0oops depositBooked=1e-12\n")
    assert trunc["r1"]["records"] == 1 and trunc["r1"]["e_abs_max_max"] is None and trunc["r1"]["deposit_booked_last"] == 1e-12
    # REPORT-ONLY: the reader is called once and nothing else in the assessor reads its result
    asr = Path(ac.__file__).read_text()
    # S25 build 2 added the F3 gate and the paired-CPU status, which read the report: exactly three lines carry the key
    # (the report's own insertion into the child, the pair report, the gate summary) and none touches `admissible`.
    assert asr.count("s25_report(") == 2 and not any('"s25"' in ln and "admissible" in ln for ln in asr.splitlines()) \
        and len([ln for ln in asr.splitlines() if '"s25"' in ln and "def " not in ln]) == 3 and "['s25']" not in asr, \
        "something else reads the S25 report, or a gate reads it into `admissible`"
    # the old FREEPLATELET reader still accepts the line with the new keys
    recs, _conv = ac.parse_freeplatelet(
        "PHASE_CLOT_FREEPLATELET time=1 step=1 attempt=1 outer=1 phase=1 nMin=0 nMax=1 NMin=0 NMax=1 NNeg=0 NVolume=1 "
        "NChange=0 NStarVolume=1 transportResidual=0 boundaryOut=0 dBSum=0 pairMax=0 volResMax=0 volResL1=0 epsMin=1 "
        "availClamped=0 capClamped=0 contentOverCapCells=3 contentOverCapMax=0.01 Y=0.5 Deff=1e-10 convention=intrinsic\n")
    assert len(recs) == 1 and recs[0]["contentOverCapCells"] == 3.0, recs
    print("   assess: the four lines + CONFIG/RESTART parse to report-only fields (great skipped, none counted, a truncated "
          "token skipped); the solver prints every key read; an old log has the same shape; nothing else reads s25")


def s25_build1_checks(only: str | None = None) -> None:
    assert S23_SOLVER_C.is_file(), f"the solver source is absent: {S23_SOLVER_C}"
    src = S23_SOLVER_C.read_text()
    sections = (("cap", lambda: _s25_cap_section(src)), ("state", lambda: _s25_state_section(src)),
                ("dbound", lambda: _s25_dbound_section(src)), ("gamma", _s25_gamma_section),
                ("assess", lambda: _s25_assess_section(src)))
    ran = 0
    for name, fn in sections:
        if only is not None and only != name:
            continue
        fn()
        ran += 1
    assert ran, f"no S25 section named {only!r}"


# ---------------------------------------------------------------------------------------------------------------------
# S25.1 A (solver-engineer 04, docket 20261013-1405-run023502-b76-s25-verification; design 2026-10-13-S25.1 rev. 2
# section A, OI-77): the constructor stores the phase weights' old time on a restart and REFUSES a restart (start time
# > 0) whose directory holds no phiSkel. Same mechanism as S25 build 1: a Python MIRROR of the refusal, PINNED to the
# whitespace-squashed solver text, each pin seen to break when its guard is deleted or loosened in a scratch copy, and
# the mirror run on both sides of the boundary. Runs alone:
#   python3 -B -c "import dry_run; dry_run.s251_checks()"
# ---------------------------------------------------------------------------------------------------------------------
def s251_restart_would_refuse(start_time: float, files) -> str | None:
    """The constructor's S25.1 A refusal: a restart (runTime.value() > 0 at construction) with no <time>/phiSkel.
    None = accepted (a cold start, with or without phiSkel; a restart that has it)."""
    return "PHISKEL" if (start_time > 0 and "phiSkel" not in files) else None


def s251_checks() -> None:
    assert S23_SOLVER_C.is_file(), f"the solver source is absent: {S23_SOLVER_C}"
    src = S23_SOLVER_C.read_text()
    pins = ("const bool restartS251 = runTime.value() > 0;",
            "if (restartS251 && !phiSkel_.headerOk())",
            "<< \"file (\" << phiSkel_.objectPath() << \"): the phase weights' \"",
            "if (phiSkel_.headerOk()) { phiFluid_ = 1.0 - phiSkel_; phiSkel_.oldTime(); phiFluid_.oldTime(); }",
            "if (restartS251) { Wp_.oldTime(); } // S25 C: LAST")
    drifts = (
        ("the refusal deleted", "if (restartS251 && !phiSkel_.headerOk())", "if (false)"),
        ("a cold start counted as a restart", "runTime.value() > 0;", "runTime.value() >= 0;"),
        ("the file not named", "phiSkel_.objectPath()", "phiSkel_.name()"),
        ("phiFluid_ left at 1 (its assignment dropped)", "{ phiFluid_ = 1.0 - phiSkel_; phiSkel_.oldTime();",
         "{ phiSkel_.oldTime();"),
        ("the phiFluid_ old time not stored", "phiSkel_.oldTime(); phiFluid_.oldTime(); }", "phiSkel_.oldTime(); }"),
        ("the Wp_ prime taken on a cold start too", "if (restartS251) { Wp_.oldTime(); }", "{ Wp_.oldTime(); }"))
    _s25_drift(src, pins, drifts, "S25.1 A")
    t42 = 0.0002003173603    # RT_P73's restart level (docket 20261013-1405, 01)
    cases = (
        ("accepted: a cold start, no phiSkel (every generated child's 0/)", 0.0, {"w", "U", "p"}, None),
        ("accepted: a cold start with a phiSkel file", 0.0, {"w", "U", "p", "phiSkel"}, None),
        ("accepted: a restart from an S25 write", t42, {"w", "U", "p", "phiSkel", "uniform/time"}, None),
        ("PHISKEL: a restart from a pre-S25 write (no phiSkel)", t42, {"w", "U", "p", "uniform/time"}, "PHISKEL"),
        ("PHISKEL: the smallest positive start time", 5e-324, {"w"}, "PHISKEL"),
    )
    for label, t0, files, want in cases:
        got = s251_restart_would_refuse(t0, files)
        assert got == want, f"{label}: the mirror says {got!r}, S25.1 A says {want!r}"
    print(f"   s251: the restart refusal and the old-time primes are pinned to the solver ({len(drifts)} drifts seen); "
          f"{len(cases)} directories judged: a pre-S25 restart refused (PHISKEL), cold starts and S25 restarts accepted")


# ---------------------------------------------------------------------------------------------------------------------
# S25 BUILD 2 (solver-engineer report 07; design 2026-10-10-S25 sections 2A, 3A, 5, 12): the LT keys' config refusals, the
# generator that writes them, and the assessor's F3 / dtKVTangent / paired-CPU readers. Same mechanism as build 1: a Python
# MIRROR of each solver predicate, PINNED to the whitespace-squashed solver text (each pin occurs exactly once), the pin check
# seen to FAIL when the guard is deleted or loosened in a scratch copy of the text, and the mirror run on both sides of every
# boundary. NEW here: the mirror takes `off`, a set of disabled guards, and every refusing dictionary is ALSO run with exactly
# its own guard off -- the refusal must disappear (or become a later one only if the dictionary breaks two), so each
# negative is the guard's and not a side effect of another. Each section runs alone:
#   python3 -B -c "import dry_run; dry_run.s25_build2_checks('<section>')"   (sections: ltsolver, ltgen, ltpairs, ltassess, 5agen, oi76, b76off)
# ---------------------------------------------------------------------------------------------------------------------
S25_LT_BASE = (
    "skeletonLaw valanisLandelTrilinear;\n"
    "skeletonLawCoeffs\n{\n    E1 234.0;\n    E2 5820.0;\n    D1 0.107;\n    D2 0.574;\n    kappa 0.98;\n}\n"
    "coreShell no;\nG 117.0;\nnetworkPressureCoeff 0.98;\nskeletonVolumeFraction 0.45;\n")
# the code each guard returns, in the solver's own order: law -> coeffs -> elasticCourantModulus (member initialisers, .H
# order) -> constructor body (coreShell, G, networkPressureCoeff, phi0)
S25_LT_GUARDS = ("SL", "CM", "CK", "CF", "CE", "CD", "CN", "KP", "EM", "ET", "CS", "G", "NP", "P0")


def s25_lt_solver_would_refuse(props_text: str, off=frozenset()):
    """The build-2 config FatalErrors, read off phaseClotProperties text the way lookupOrDefault / lookup read it
    (phaseClotAxisymmetricFluid.C s25SkeletonLaw / s25SkeletonLawCoeffs / s25ElasticCourantModulus and the constructor body
    after `G`/`networkPressureCoeff`). Returns None (accepted) or the first guard that fires: SL law not shipped/LT; CM
    skeletonLawCoeffs missing; CK a coefficient missing; CF a coefficient not finite; CE E1 or E2 not > 0; CD D1 >= D2; CN D1
    < 0; KP kappa not > 0 (S25b3); EM elasticCourantModulus not constant/tangent; ET tangent under shipped; CS LT + coreShell; G absent or != E1/2;
    NP networkPressureCoeff absent or != kappa; P0 phi0 >= 1. `off` disables guards by code (used to show each negative is its
    own guard's)."""
    text = re.sub(r"//[^\n]*", "", props_text)
    sub = re.search(r"(?m)^\s*skeletonLawCoeffs\s*\{([^}]*)\}", text)
    top = re.sub(r"(?m)^\s*skeletonLawCoeffs\s*\{[^}]*\}", "", text)

    def entry(key, src=None):
        m = re.search(rf"(?m)^\s*{key}\s+([^;\s]+)\s*;", top if src is None else src)
        return None if m is None else m.group(1)

    def num(tok):
        try:
            return float(tok)
        except (TypeError, ValueError):
            return float("nan")
    law = entry("skeletonLaw") or "shipped"
    if "SL" not in off and law not in ("shipped", "valanisLandelTrilinear"):
        return "SL"
    lt = law == "valanisLandelTrilinear"
    v = {}
    if lt:
        if sub is None:
            if "CM" not in off:
                return "CM"
        else:
            for k in ("E1", "E2", "D1", "D2", "kappa"):
                tok = entry(k, sub.group(1))
                if tok is None and "CK" not in off:
                    return "CK"
                v[k] = num(tok)
                if not math.isfinite(v[k]) and "CF" not in off:
                    return "CF"
            if "CE" not in off and not (v["E1"] > 0 and v["E2"] > 0):
                return "CE"
            if "CD" not in off and not (v["D1"] < v["D2"]):
                return "CD"
            if "CN" not in off and not (v["D1"] >= 0):
                return "CN"
            if "KP" not in off and not (v["kappa"] > 0):
                return "KP"
    m = entry("elasticCourantModulus") or ("tangent" if lt else "constant")
    if "EM" not in off and m not in ("constant", "tangent"):
        return "EM"
    if "ET" not in off and m == "tangent" and not lt:
        return "ET"
    if lt:
        if "CS" not in off and (entry("coreShell") or "no") in ("yes", "true", "on", "1", "y", "t"):
            return "CS"
        if "G" not in off and (entry("G") is None or not num(entry("G")) == 0.5 * v.get("E1", float("nan"))):
            return "G"
        if "NP" not in off and (entry("networkPressureCoeff") is None
                                or not num(entry("networkPressureCoeff")) == v.get("kappa", float("nan"))):
            return "NP"
        phi0 = num(entry("skeletonVolumeFraction") or "0.01")
        if "P0" not in off and not (phi0 < 1):
            return "P0"
    return None


def _s25_lt_dict(**changes) -> str:
    """S25_LT_BASE with edits: a key -> new value (None deletes it); `coeff_<K>` edits inside skeletonLawCoeffs; `raw` appends."""
    text = S25_LT_BASE
    for key, val in changes.items():
        if key == "raw":
            text += val
        elif key.startswith("coeff_"):
            k = key[6:]
            text = re.sub(rf"(?m)^    {k} [^;]*;\n", "" if val is None else f"    {k} {val};\n", text)
        elif key == "nocoeffs":
            text = re.sub(r"skeletonLawCoeffs\n\{[^}]*\}\n", "", text)
        else:
            text = re.sub(rf"(?m)^{key} [^;]*;\n", "" if val is None else f"{key} {val};\n", text)
    return text


def _s25_lt_solver_section(src: str) -> None:
    pins = (
        'if (law != "shipped" && law != "valanisLandelTrilinear")',
        'if (!props.isDict("skeletonLawCoeffs"))',
        'if (!d.found(names[k]))',
        'if (!std::isfinite(v[k]))',
        'if (!(v[0] > 0) || !(v[1] > 0))',
        'if (!(v[2] < v[3]))',
        'if (!(v[2] >= 0))',
        'if (m != "constant" && m != "tangent")',
        'if (m == "tangent" && !lt)',
        '"elasticCourantModulus", lt ? "tangent" : "constant"',
        'if (skeletonLawLT_) { if (coreShell_) {',
        '!phaseClotProperties_.found("G") || G_.value() != 0.5*skeletonLawCoeffs_.E1',
        '!phaseClotProperties_.found("networkPressureCoeff") || networkPressureCoeff_ != skeletonLawCoeffs_.kappa',
        'if (!(skeletonVolumeFraction_ < networkPole()))',
        'lookupOrDefault<scalar> ( "skeletonVolumeFraction", 0.01 )',
        'lookupOrDefault<Switch>("coreShell", false)',
        'v[k] = d.lookup<scalar>(names[k]);',
        'const char* const names[5] = {"E1", "E2", "D1", "D2", "kappa"};',
        'if (!(v[4] > 0))',
    )
    sq = _squash(src)
    # (label, anchor, replacement): one guard deleted or loosened per drift; the pin that anchors it must stop matching
    drifts = (
        ("SL: the guard deleted", pins[0], "if (false)"),
        ("SL: `shipped` no longer accepted", 'law != "shipped" && law', "law"),
        ("CM: the guard deleted", pins[1], "if (false)"),
        ("CK: the guard deleted", pins[2], "if (false)"),
        ("CF: the guard deleted", pins[3], "if (false)"),
        ("CE: the guard deleted", pins[4], "if (false)"),
        ("CE: > 0 loosened to >= 0 (E1 = 0 would pass)", "!(v[0] > 0) ||", "!(v[0] >= 0) ||"),
        ("CE: E2 not tested", "|| !(v[1] > 0)", "|| false"),
        ("CD: the guard deleted", pins[5], "if (false)"),
        ("CD: < loosened to <= (D1 = D2 would pass)", "!(v[2] < v[3])", "!(v[2] <= v[3])"),
        ("CN: the guard deleted", pins[6], "if (false)"),
        ("CN: >= tightened to > (D1 = 0 refused)", "!(v[2] >= 0)", "!(v[2] > 0)"),
        ("KP: the guard deleted", pins[18], "if (false)"),
        ("KP: > 0 loosened to >= 0 (kappa = 0 would pass)", "!(v[4] > 0)", "!(v[4] >= 0)"),
        ("EM: the guard deleted", pins[7], "if (false)"),
        ("ET: the guard deleted", pins[8], "if (false)"),
        ("ET: the default under LT no longer tangent", '"elasticCourantModulus", lt ? "tangent" : "constant"',
         '"elasticCourantModulus", "constant"'),
        ("CS: the guard deleted", pins[10], "if (skeletonLawLT_) { if (false) {"),
        ("G: the comparison deleted", "|| G_.value() != 0.5*skeletonLawCoeffs_.E1", "|| false"),
        ("G: 0.5*E1 changed", "G_.value() != 0.5*skeletonLawCoeffs_.E1", "G_.value() != 0.4*skeletonLawCoeffs_.E1"),
        ("G: absent no longer refused", '!phaseClotProperties_.found("G") ||', "false ||"),
        ("NP: the comparison deleted", "|| networkPressureCoeff_ != skeletonLawCoeffs_.kappa", "|| false"),
        ("NP: absent no longer refused", '!phaseClotProperties_.found("networkPressureCoeff") ||', "false ||"),
        ("P0: the guard deleted", pins[13], "if (false)"),
        ("P0: < loosened to <= (phi0 = 1 would pass)", "skeletonVolumeFraction_ < networkPole()", "skeletonVolumeFraction_ <= networkPole()"),
    )
    _s25_drift(src, pins, drifts, "LT config")
    # the mirror's boundaries, both sides: (label, dictionary text, the guard that must fire or None)
    cases = [
        ("accepted: the base LT dictionary (WB-human numbers, as test input)", _s25_lt_dict(), None),
        ("accepted: shipped, nothing named", "G 4.0;\nnetworkPressureCoeff 1.0;\n", None),
        ("accepted: shipped named explicitly", "skeletonLaw shipped;\n", None),
        ("accepted: shipped + elasticCourantModulus constant", "elasticCourantModulus constant;\n", None),
        ("accepted: LT + constant (the paired control)", _s25_lt_dict(raw="elasticCourantModulus constant;\n"), None),
        ("accepted: LT + tangent stated", _s25_lt_dict(raw="elasticCourantModulus tangent;\n"), None),
        ("accepted: coreShell yes under the SHIPPED law", "coreShell yes;\nG 4.0;\n", None),
        ("accepted: D1 = 0 (0 >= 0)", _s25_lt_dict(coeff_D1="0"), None),
        ("accepted: skeletonVolumeFraction 0.999999", _s25_lt_dict(skeletonVolumeFraction="0.999999"), None),
        ("accepted: G 1.17e2 (the same double as 117.0)", _s25_lt_dict(G="1.17e2"), None),
        ("accepted: E1 0.1 with G 0.05 (halving is exact)", _s25_lt_dict(coeff_E1="0.1", G="0.05"), None),
        ("accepted: skeletonVolumeFraction absent (default 0.01)", _s25_lt_dict(skeletonVolumeFraction=None), None),
        ("SL: skeletonLaw foo", "skeletonLaw foo;\n", "SL"),
        ("SL: skeletonLaw Shipped (case matters)", "skeletonLaw Shipped;\n", "SL"),
        ("SL wins over a missing coeffs block", "skeletonLaw foo;\n", "SL"),
        ("CM: LT with no skeletonLawCoeffs", _s25_lt_dict(nocoeffs=True), "CM"),
        ("CM: LT with a skeletonLawCoeffs that is an entry, not a dictionary",
         _s25_lt_dict(nocoeffs=True, raw="skeletonLawCoeffs 1.0;\n"), "CM"),
        ("CK: E1 missing", _s25_lt_dict(coeff_E1=None), "CK"),
        ("CK: E2 missing", _s25_lt_dict(coeff_E2=None), "CK"),
        ("CK: D1 missing", _s25_lt_dict(coeff_D1=None), "CK"),
        ("CK: D2 missing", _s25_lt_dict(coeff_D2=None), "CK"),
        ("CK: kappa missing", _s25_lt_dict(coeff_kappa=None), "CK"),
        ("CF: E1 nan", _s25_lt_dict(coeff_E1="nan"), "CF"),
        ("CF: E2 inf", _s25_lt_dict(coeff_E2="inf"), "CF"),
        ("CF: D2 -inf", _s25_lt_dict(coeff_D2="-inf"), "CF"),
        ("CF: kappa nan", _s25_lt_dict(coeff_kappa="nan"), "CF"),
        ("CE: E1 = 0", _s25_lt_dict(coeff_E1="0", G="0.0"), "CE"),
        ("CE: E1 < 0", _s25_lt_dict(coeff_E1="-1.0", G="-0.5"), "CE"),
        ("CE: E2 = 0", _s25_lt_dict(coeff_E2="0"), "CE"),
        ("CE: E2 < 0", _s25_lt_dict(coeff_E2="-5.0"), "CE"),
        ("CD: D1 = D2", _s25_lt_dict(coeff_D1="0.574"), "CD"),
        ("CD: D1 > D2", _s25_lt_dict(coeff_D1="0.6"), "CD"),
        ("CN: D1 < 0", _s25_lt_dict(coeff_D1="-0.1"), "CN"),
        ("KP: kappa = 0 (networkPressureCoeff equal, so NP is silent)", _s25_lt_dict(coeff_kappa="0", networkPressureCoeff="0"), "KP"),
        ("KP: kappa < 0", _s25_lt_dict(coeff_kappa="-0.98", networkPressureCoeff="-0.98"), "KP"),
        ("accepted: kappa the smallest positive double", _s25_lt_dict(coeff_kappa="5e-324", networkPressureCoeff="5e-324"), None),
        ("EM: elasticCourantModulus foo", "elasticCourantModulus foo;\n", "EM"),
        ("EM: elasticCourantModulus Tangent (case matters)", _s25_lt_dict(raw="elasticCourantModulus Tangent;\n"), "EM"),
        ("ET: tangent under the shipped law (nothing else named)", "elasticCourantModulus tangent;\n", "ET"),
        ("ET: tangent under skeletonLaw shipped named", "skeletonLaw shipped;\nelasticCourantModulus tangent;\n", "ET"),
        ("CS: LT + coreShell yes", _s25_lt_dict(coreShell="yes"), "CS"),
        ("CS: LT + coreShell true", _s25_lt_dict(coreShell="true"), "CS"),
        ("G: absent (G would default 58.6)", _s25_lt_dict(G=None), "G"),
        ("G: 118 against E1/2 = 117", _s25_lt_dict(G="118.0"), "G"),
        ("G: one ulp below E1/2", _s25_lt_dict(G=repr(math.nextafter(117.0, 0.0))), "G"),
        ("NP: absent (networkPressureCoeff would default 1.0)", _s25_lt_dict(networkPressureCoeff=None), "NP"),
        ("NP: 1.0 against kappa 0.98", _s25_lt_dict(networkPressureCoeff="1.0"), "NP"),
        ("NP: one ulp above kappa", _s25_lt_dict(networkPressureCoeff=repr(math.nextafter(0.98, 1.0))), "NP"),
        ("P0: skeletonVolumeFraction 1.0", _s25_lt_dict(skeletonVolumeFraction="1.0"), "P0"),
        ("P0: skeletonVolumeFraction 1.5", _s25_lt_dict(skeletonVolumeFraction="1.5"), "P0"),
    ]
    seen_guards = set()
    for label, text, want in cases:
        got = s25_lt_solver_would_refuse(text)
        assert got == want, f"{label}: the mirror says {got!r}, the solver text says {want!r}"
        if want is not None:
            seen_guards.add(want)
            # the refusal is THIS guard's: with it off the (single-defect) dictionary is accepted
            off = s25_lt_solver_would_refuse(text, off=frozenset({want}))
            if want in ("CM", "CK", "CF"):
                # a missing dictionary / key / non-finite value, with its guard off, is a lookup error or a NaN that the
                # NEXT guard catches (CE), never an accepted dictionary: assert only that this guard is not what fires
                assert off != want, f"{label}: with guard {want} off the mirror still says {off!r}"
            else:
                assert off is None, f"{label}: with guard {want} off the mirror still says {off!r}"
    assert seen_guards == set(S25_LT_GUARDS), sorted(set(S25_LT_GUARDS) - seen_guards)
    # order: the first refusal wins (law before coeffs; CK before CF; CE before CD; CS before G before NP before P0)
    assert s25_lt_solver_would_refuse("skeletonLaw foo;\n" + S25_LT_BASE.replace("skeletonLaw valanisLandelTrilinear;\n", "")) == "SL"
    assert s25_lt_solver_would_refuse(_s25_lt_dict(coeff_E1=None, coeff_E2="nan")) == "CK"
    assert s25_lt_solver_would_refuse(_s25_lt_dict(coeff_E1="0", coeff_D1="0.6", G="0.0")) == "CE"
    assert s25_lt_solver_would_refuse(_s25_lt_dict(coeff_D1="-0.1", coeff_kappa="0", networkPressureCoeff="0")) == "CN"
    assert s25_lt_solver_would_refuse(_s25_lt_dict(coeff_kappa="0")) == "KP"   # KP (coeffs) before NP (constructor body)
    assert s25_lt_solver_would_refuse(_s25_lt_dict(coreShell="yes", G=None, networkPressureCoeff=None)) == "CS"
    assert s25_lt_solver_would_refuse(_s25_lt_dict(G=None, networkPressureCoeff=None, skeletonVolumeFraction="2")) == "G"
    assert s25_lt_solver_would_refuse(_s25_lt_dict(networkPressureCoeff=None, skeletonVolumeFraction="2")) == "NP"
    print(f"   ltsolver: {len(cases)} dictionaries, all {len(S25_LT_GUARDS)} guards fire; with its own guard off a dictionary is accepted "
          f"(11 guards) or caught only by a later one (CM, CK, CF); {len(drifts)} drifts of the solver text seen by the pins; "
          f"first-refusal order checked")


S25_LT_OK = {"E1_pa": 234.0, "E2_pa": 5820.0, "D1": 0.107, "D2": 0.574, "kappa": 0.98}
# A throat_free_shell child that no parent pair (FSF_PARENT_PAIRS, S18P_PARENT_PAIRS, XU_PARENT) names as a child to compare, so
# a new key on it is not a "beyond its named keys" refusal. It is pinned to the S22 library in the controls and states no
# tau_v_s: the bench moves it onto the package solver (solver_library removed, tau_v_s 5e-4, the S23 floor) as _s25_generate does.
S25_LT_CHILD = "fsb_ctl_r3"
S25_LT_BASE_EDITS = {"solver_library": None, "tau_v_s": 5e-4}


class _S25Bench:
    """ONE scratch copy of the case; each trial rewrites one child's controls entry, optionally patches the generator source
    (each anchor unique), generates that child alone and returns (rc, last line, dictionary text or None)."""
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="am2020-s25-lt-")
        self.root = build_case(Path(self._tmp.name))
        shutil.rmtree(self.root / "runs")
        self.cp = self.root / "constant" / "mainCaseControls.json"
        self.gp = self.root / "scripts" / "create_cases.py"
        self.controls0 = self.cp.read_text()
        self.gen0 = self.gp.read_text()

    def close(self):
        self._tmp.cleanup()

    def run(self, edits=None, child=S25_LT_CHILD, patches=(), top=None, series=None):
        c = json.loads(self.controls0)
        _s16(c, child, **{**S25_LT_BASE_EDITS, **(edits or {})})
        if top:
            c.update(top)
        for name, changes in (series or {}).items():
            c["series"][name].update(changes)
        self.cp.write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n")
        g = self.gen0
        for old, new in patches:
            assert g.count(old) == 1, f"generator patch anchor is not unique: {old[:70]!r}"
            g = g.replace(old, new)
        self.gp.write_text(g)
        if (self.root / "runs").exists():
            shutil.rmtree(self.root / "runs")
        done = subprocess.run([sys.executable, "-B", str(self.gp), child], capture_output=True, text=True)
        lines = (done.stdout + done.stderr).strip().splitlines()
        pf = self.root / "runs" / child / "constant" / "phaseClotProperties"
        return done.returncode, (lines[-1] if lines else ""), (pf.read_text() if pf.is_file() else None)


def _s25_lt_edits(**changes):
    """LT on the child: the base keys (G_pa dropped: derived), then edits; `coeff_<K>` edits a coefficient (None deletes)."""
    e = {"skeleton_law": "valanisLandelTrilinear", "skeleton_law_coeffs": dict(S25_LT_OK), "G_pa": None}
    for k, v in changes.items():
        if k.startswith("coeff_"):
            if v is None:
                e["skeleton_law_coeffs"].pop(k[6:])
            else:
                e["skeleton_law_coeffs"][k[6:]] = v
        else:
            e[k] = v
    return e


def _s25_lt_gen_section() -> None:
    bench = _S25Bench()
    try:
        call = "        s25_skeleton_law_refusal(child, spec)\n"
        # --- positive: what is written, to the last bit
        rc, last, props = bench.run(_s25_lt_edits())
        assert rc == 0, last
        mir = s25_lt_solver_would_refuse(props)
        assert mir is None, f"the generated LT dictionary is refused by the solver mirror: {mir}"
        for line in ("skeletonLaw valanisLandelTrilinear;", "elasticCourantModulus tangent;", "G 117.0;",
                     "networkPressureCoeff 0.98;", "    E1 234.0;", "    E2 5820.0;", "    D1 0.107;", "    D2 0.574;", "    kappa 0.98;"):
            assert re.search(rf"(?m)^{re.escape(line)}$", props), f"{line!r} not written"
        assert len(re.findall(r"(?m)^G ", props)) == 1 and len(re.findall(r"(?m)^networkPressureCoeff ", props)) == 1
        rc, last, props = bench.run(_s25_lt_edits(elastic_courant_modulus="constant"))
        assert rc == 0 and re.search(r"(?m)^elasticCourantModulus constant;$", props), (rc, last)
        rc, last, props = bench.run(_s25_lt_edits(G_pa=117.0, network_pressure_coeff=0.98))
        assert rc == 0 and s25_lt_solver_would_refuse(props) is None, (rc, last)   # a second copy of the same float is accepted
        # a non-trivial E1: G is exactly half, and the solver's `!=` sees the same double
        for e1 in (234.0, 0.1, 3.0e-3 + 1e-17, 7.3, 1e9 / 3):
            rc, last, props = bench.run(_s25_lt_edits(coeff_E1_pa=e1))
            assert rc == 0 and s25_lt_solver_would_refuse(props) is None, (e1, rc, last)
            assert float(re.search(r"(?m)^G (\S+);$", props).group(1)) == 0.5 * e1
        rc, last, props = bench.run({"skeleton_law": "shipped", "elastic_courant_modulus": "constant"})
        assert rc == 0 and re.search(r"(?m)^skeletonLaw shipped;$", props) and "skeletonLawCoeffs" not in props, (rc, last)
        assert s25_lt_solver_would_refuse(props) is None
        # --- negatives: the generator refuses each for ITS reason; the SAME defect with the guard call removed does not
        # produce that refusal (it is accepted, or fails some other way)
        C = lambda **k: _s25_lt_edits(**k)
        negatives = (
            ("skeleton_law misspelt", {"skeleton_law": "valanisLandel"}, "is not one of shipped, valanisLandelTrilinear"),
            ("skeleton_law_coeffs under the shipped law (key absent)", {"skeleton_law_coeffs": dict(S25_LT_OK)}, "silently inert"),
            ("skeleton_law_coeffs under skeleton_law shipped", {"skeleton_law": "shipped", "skeleton_law_coeffs": dict(S25_LT_OK)},
             "silently inert"),
            ("LT without skeleton_law_coeffs", {"skeleton_law": "valanisLandelTrilinear"}, "skeleton_law_coeffs {E1_pa"),
            ("LT with E1_pa missing", C(coeff_E1_pa=None), "must hold exactly"),
            ("LT with D2 missing", C(coeff_D2=None), "must hold exactly"),
            ("LT with an extra coefficient", C(coeff_E3_pa=1.0), "must hold exactly"),
            ("LT with E1_pa nan", C(coeff_E1_pa=float("nan")), "is not a finite number"),
            ("LT with kappa inf", C(coeff_kappa=float("inf")), "is not a finite number"),
            ("LT with D2 a string", C(coeff_D2="0.574"), "is not a finite number"),
            ("LT with E1_pa 0", C(coeff_E1_pa=0.0), "both moduli must be > 0"),
            ("LT with E2_pa negative", C(coeff_E2_pa=-1.0), "both moduli must be > 0"),
            ("LT with D1 = D2", C(coeff_D1=0.574), "D1 < D2 is required"),
            ("LT with D1 > D2", C(coeff_D1=0.6), "D1 < D2 is required"),
            ("LT with D1 < 0", C(coeff_D1=-0.1), "breaks 1 -+ D1 would invert"),
            ("LT with kappa 0", C(coeff_kappa=0.0), "kappa 0.0 must be > 0"),
            ("LT with skeleton_volume_fraction 1.0", C(skeleton_volume_fraction=1.0), "must be < 1 under"),
            ("LT with elastic_courant_modulus misspelt", C(elastic_courant_modulus="tangential"), "is not one of constant, tangent"),
            ("tangent under the shipped law", {"elastic_courant_modulus": "tangent"}, "refused under the shipped law"),
            ("tangent under skeleton_law shipped", {"skeleton_law": "shipped", "elastic_courant_modulus": "tangent"},
             "refused under the shipped law"),
            ("LT with G_pa disagreeing with E1/2", C(G_pa=118.0), "disagrees with 0.5*E1_pa"),
            ("LT with network_pressure_coeff disagreeing with kappa", C(network_pressure_coeff=1.0), "disagrees with kappa"),
            ("LT on solver_library s22", C(solver_library="s22"), "predates S25 build 2"),
            ("elastic_courant_modulus on solver_library p2prime", {"elastic_courant_modulus": "constant", "solver_library": "p2prime",
                                                                  "block_linear_solver": "umfpack"}, "predates S25 build 2"),
        )
        removed = 0
        for label, edits, want in negatives:
            rc, last, _ = bench.run(edits)
            assert rc != 0, f"the generator ACCEPTED {label}"
            assert want in last, f"{label}: refused for the wrong reason: {last}"
            rc2, last2, _ = bench.run(edits, patches=((call, "        pass\n"),))
            assert rc2 == 0 or want not in last2, f"{label}: the refusal survives the removal of the guard call: {last2}"
            removed += rc2 == 0
            print(f"generator refuses: {label:<58} -> {want!r}; guard call removed: {'ACCEPTED' if rc2 == 0 else 'fails otherwise'}")
        # LT + coreShell. The only core_shell series are throat_embolus and xu2017_fig5, whose children every parent check
        # pins to em76_G4_s18; so the SERIES is flipped instead (throat_free_shell core_shell true) and free_shell_series_refusal,
        # which would refuse that first for its own reason, is patched out in BOTH runs below.
        free_call = '        if child["series"] in FREE_SHELL_SERIES:\n            free_shell_series_refusal(child, spec)\n'
        flip = {"throat_free_shell": {"core_shell": True}}
        rc, last, _ = bench.run(_s25_lt_edits(), patches=((free_call, ""),), series=flip)
        assert rc != 0 and "with coreShell is refused" in last and "throat_free_shell" in last, (rc, last)
        rc2, last2, _ = bench.run(_s25_lt_edits(), patches=((free_call, ""), (call, "        pass\n")), series=flip)
        assert rc2 == 0 or "with coreShell is refused" not in last2, last2   # no longer this guard's refusal
        # and without the flip the same child is not refused for coreShell: the guard reads the series, not the key
        rc3, last3, _ = bench.run(_s25_lt_edits())
        assert rc3 == 0, last3
        print(f"generator refuses: LT on a core_shell series{'':<37} -> 'with coreShell is refused'; guard call removed: "
              f"{'ACCEPTED' if rc2 == 0 else 'fails otherwise: ' + last2[:70]}")
        # --- the read-back guard: a writer that disagrees with the child is refused for ITS reason; with the read-back removed it is not
        read_call = "        s25_skeleton_law_readback(props, child)\n"
        writers = (
            ("G written from the child's G_pa instead of 0.5*E1", (('"G": s25_g_pa(child),', '"G": child.get("G_pa", 4.0),'),),
             "G written as"),
            ("networkPressureCoeff left at the default 1.0", (('repr(s25_network_pressure_coeff(child)),', 'repr(1.0),'),),
             "networkPressureCoeff written as"),
            ("a coefficient written wrong (D1 the D2 value)", (('body = "".join(f"    {dk} {k[dk]!r};\\n" for _ck, dk in S25_LT_COEFFS)',
                                                                 'body = "".join(f"    {dk} {k[dk]!r};\\n" for _ck, dk in S25_LT_COEFFS).replace("D1 0.107", "D1 0.574")'),),
             "skeletonLawCoeffs D1 written as"),
            ("elasticCourantModulus written constant when the default is tangent",
             (('child.get("elastic_courant_modulus", "tangent"))\n\n\ndef s25_skeleton_law_readback',
               'child.get("elastic_courant_modulus", "constant"))\n\n\ndef s25_skeleton_law_readback'),),
             "elasticCourantModulus written as"),
        )
        for label, patches, want in writers:
            rc, last, _ = bench.run(_s25_lt_edits(), patches=patches)
            assert rc != 0 and want in last, f"{label}: {rc} {last}"
            rc2, last2, props2 = bench.run(_s25_lt_edits(), patches=patches + ((read_call, "        pass\n"),))
            assert rc2 == 0, (label, rc2, last2)
            print(f"read-back refuses: {label:<66} -> {want!r}; read-back removed: ACCEPTED (the solver mirror says "
                  f"{s25_lt_solver_would_refuse(props2)!r})")
        print(f"   ltgen: LT written to the last bit (5 E1 values), {len(negatives) + 1} config refusals each for their own reason "
              f"and {removed} accepted with the guard call removed, 4 writer defects seen by the read-back")
    finally:
        bench.close()


def _s25_lt_pairs_section() -> None:
    bench = _S25Bench()
    try:
        ok = [{"pair_id": "B76-OFF", "arms": ["D74_refI", "D75_refI"]}]
        rc, last, _ = bench.run(top={"s25_paired_comparisons": ok})
        assert rc == 0, last
        call = "    s25_pairs_refusal(controls)\n"
        for label, val, want in (
                ("not a list", {"pair_id": "x"}, "must be a list"),
                ("an entry with an extra key", [{"pair_id": "p", "arms": ["D74_refI", "D75_refI"], "bitwise": True}], "exactly pair_id and arms"),
                ("an entry missing arms", [{"pair_id": "p"}], "exactly pair_id and arms"),
                ("a repeated pair_id", ok + ok, "non-empty string, unique"),
                ("one arm", [{"pair_id": "p", "arms": ["D75_refI"]}], "at least two distinct"),
                ("a repeated arm", [{"pair_id": "p", "arms": ["D75_refI", "D75_refI"]}], "at least two distinct"),
                ("an arm that is not a child", [{"pair_id": "p", "arms": ["D75_refI", "D75_refX"]}], "not child_ids")):
            rc, last, _ = bench.run(top={"s25_paired_comparisons": val})
            assert rc != 0 and want in last, f"{label}: {rc} {last}"
            rc2, last2, _ = bench.run(top={"s25_paired_comparisons": val}, patches=((call, ""),))
            assert rc2 == 0 or want not in last2, f"{label}: the refusal survives the removal of the guard: {last2}"
            print(f"generator refuses s25_paired_comparisons: {label:<28} -> {want!r}; guard removed: {'ACCEPTED' if rc2 == 0 else 'fails otherwise'}")
        print("   ltpairs: 7 malformed pair lists refused, each by the guard (gone with it); the well-formed list accepted")
    finally:
        bench.close()


S25_B2_SAMPLE = (
    "PHASE_CLOT_S19_CONFIG momentumPressureCoupling=blockCoupled networkViscosity=kelvinVoigt blockLinearSolver=pardiso "
    "mklVersion=2024.0 mklLibrary=libmkl_rt.so mklCbwrBranch=4 mklCbwrAutoBranch=4 cpuModel=Intel(R)_Xeon(R)_CPU_@_2.20GHz "
    "cpuFlags=fpu,vme,avx2,avx512f\n"
    "PHASE_CLOT_DT step=1 time=1e-06 deltaT=5e-06 dtFluidCo=2e-06 dtWave=4.35e-06 dtKV=1e+300 dtKVTangent=3.95e-06 dtCarrier=1.5e-06 "
    "dtNetDiff=1e-06 netDiffForm=block dtMax=1e-05 dtGrowth=1e+300 binder=netDiff threads=1\n"
    "PHASE_CLOT_S25_F3 time=1e-06 cells=40 stiffNonPositive=0 rMax=1.71 rCell=17 kc=2.3 stiffGamma=1.34\n"
    "PHASE_CLOT_DT step=2 time=2e-06 deltaT=5e-06 dtFluidCo=2e-06 dtWave=4.35e-06 dtKV=1e+300 dtKVTangent=1e+300 dtCarrier=1.5e-06 "
    "dtNetDiff=1e+300 netDiffForm=block dtMax=1e-05 dtGrowth=1e+300 binder=writeAlign threads=1\n"
    "PHASE_CLOT_S25_F3 time=2e-06 cells=0 stiffNonPositive=0 rMax=none\n"
    "PHASE_CLOT_S25_F3 time=3e-06 cells=41 stiffNonPositive=0 rMax=3.5 rCell=3 kc=4.7 stiffGamma=1.34\n"
)


def _s25_lt_assess_section(src: str) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("assess_case_s25b2", Path(__file__).resolve().parent / "assess_case.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    sq = _squash(src)
    for needle in ('" dtKVTangent="', '"PHASE_CLOT_S25_F3"', '" cells="', '" stiffNonPositive="', '" rMax="', '" rCell="',
                   '" mklCbwrBranch="', '" mklCbwrAutoBranch="', '" cpuModel="', '" cpuFlags="', 'Info<< "PHASE_CLOT_S19_CONFIG"'):
        assert sq.count(needle) >= 1, f"the solver no longer prints {needle}: the assessor's build-2 reader is stale"
    r = ac.s25_report(S25_B2_SAMPLE)
    # --- dtKVTangent: reported, great skipped, NOT one of the 7 limits
    kt = r["dt"]["dt_kv_tangent"]
    assert kt == {"records": 2, "min": 3.95e-06, "steps_delta_t_above": 1}, kt   # step 1: deltaT 5e-6 > 3.95e-6; step 2 is great
    assert "dtKVTangent" not in r["dt"]["min"] and r["dt"]["min"]["dtWave"] == 4.35e-06 and r["dt"]["binder_counts"] == {"netDiff": 1, "writeAlign": 1}
    # --- F3: the boundary of the gate, both sides
    f3 = r["f3"]
    assert f3["records"] == 3 and f3["r_max_max"] == 3.5 and f3["records_r_none"] == 1 and f3["verdict"] == "PASS", f3
    assert f3["cells_max"] == 41.0 and f3["stiff_non_positive_max"] == 0.0 and f3["population_complete"] is True and r["gates"] == {"F3": "PASS"}
    sample = lambda *rows: "".join(f"PHASE_CLOT_S25_F3 time=1e-06 cells=5 stiffNonPositive={b} rMax={m}\n" for m, b in rows)
    for rmax, bad, want in (("4", 0, "PASS"), ("4.0", 0, "PASS"), ("3.99999", 0, "PASS"), ("4.00001", 0, "FAIL"), ("14.1", 0, "FAIL"),
                            ("0", 0, "PASS"), ("nan", 0, "FAIL"), ("inf", 0, "FAIL"), ("-nan", 0, "FAIL"), ("4oops", 0, "UNASSESSABLE"),
                            ("none", 0, None)):
        got = ac.s25_report(sample((rmax, bad)))["f3"]["verdict"]
        assert got == want, f"rMax={rmax}: verdict {got!r}, expected {want!r}"
    # a FAIL anywhere fails the run (the last record may be fine), and a FAIL is not hidden by an unparseable record
    assert ac.s25_report(sample(("5.0", 0), ("1.0", 0)))["f3"]["verdict"] == "FAIL"
    assert ac.s25_report(sample(("5.0", 0), ("4oops", 0)))["f3"]["verdict"] == "FAIL"
    assert ac.s25_report(sample(("1.0", 0), ("4oops", 0)))["f3"]["verdict"] == "UNASSESSABLE"
    # the population: a cell whose stiff was not > 0 is in `cells` but cannot be in rMax, and the report says so
    assert ac.s25_report(sample(("1.0", 2)))["f3"]["population_complete"] is False
    # CE RULING (2026-10-10, 09a's hand-off, applied in 09b): stiffNonPositive > 0 is itself a FAIL, whatever rMax over the
    # survivors says, and also when no cell has a positive stiff (rMax=none)
    for rmax, bad, want in (("1.0", 2, "FAIL"), ("1.0", 1, "FAIL"), ("none", 5, "FAIL"), ("1.0", 0, "PASS"), ("none", 0, None),
                            ("4.00001", 3, "FAIL"), ("4oops", 2, "FAIL"), ("4oops", 0, "UNASSESSABLE")):
        got = ac.s25_report(sample((rmax, bad)))["f3"]["verdict"]
        assert got == want, f"rMax={rmax} stiffNonPositive={bad}: verdict {got!r}, expected {want!r}"
    # a FAIL anywhere fails the run: a clean last record does not hide an earlier cell with stiff <= 0
    assert ac.s25_report(sample(("1.0", 1), ("1.0", 0)))["f3"]["verdict"] == "FAIL"
    # a record with no stiffNonPositive token (or a truncated one) is not read as 0
    assert ac.s25_report("PHASE_CLOT_S25_F3 time=1e-06 cells=5 rMax=1.0\n")["f3"]["verdict"] == "UNASSESSABLE"
    assert ac.s25_report("PHASE_CLOT_S25_F3 time=1e-06 cells=5 stiffNonPositive=2oops rMax=1.0\n")["f3"]["verdict"] == "UNASSESSABLE"
    # the gated clause is exactly the new one: the same input with the clause removed would PASS (mutation, seen)
    _src = Path(ac.__file__).read_text()
    _clause = " or (bad and max(bad) > 0)"
    assert _src.count(_clause) == 1
    _ns = {}
    exec(compile(_src.replace(_clause, ""), "assess_case_nostiffgate", "exec"), _ns)
    assert _ns["s25_report"](sample(("1.0", 2)))["f3"]["verdict"] == "PASS", "the mutation is not seen"
    assert ac.s25_report("Time = 1s\n")["f3"]["verdict"] is None and ac.s25_report("Time = 1s\n")["f3"]["records"] == 0
    # a mutation of the gate is seen: the threshold moved to 5 would pass rMax=4.5 (the test above catches it at 4.00001)
    saved = ac.S25_F3_R_MAX
    ac.S25_F3_R_MAX = 5.0
    try:
        assert ac.s25_report(sample(("4.5", 0)))["f3"]["verdict"] == "PASS"
    finally:
        ac.S25_F3_R_MAX = saved
    assert ac.s25_report(sample(("4.5", 0)))["f3"]["verdict"] == "FAIL"
    # --- the CPU identity, parsed from PHASE_CLOT_S19_CONFIG
    assert r["cpu"] == {"mklCbwrBranch": "4", "mklCbwrAutoBranch": "4", "cpuModel": "Intel(R)_Xeon(R)_CPU_@_2.20GHz",
                        "cpuFlags": "fpu,vme,avx2,avx512f"}, r["cpu"]
    assert ac.s25_report("Time = 1s\n")["cpu"] == {k: None for k in ac.S25_CPU_KEYS}   # an old binary: the same shape, nothing in it
    # --- the paired comparison: VOID when the arms' cpuModel or flags (or the MKL branch) differ
    A = dict(r["cpu"])
    base = lambda **ch: {**A, **ch}
    cases = (
        ("same CPU, same flags, same branch", {"a": A, "b": base()}, "SAME_CPU", []),
        ("a different cpuModel", {"a": A, "b": base(cpuModel="AMD_EPYC_7B13")}, "VOID", ["cpuModel"]),
        ("the same model, one flag fewer", {"a": A, "b": base(cpuFlags="fpu,vme,avx2")}, "VOID", ["cpuFlags"]),
        ("the same model and flags, a different CBWR branch", {"a": A, "b": base(mklCbwrBranch="3")}, "VOID", ["mklCbwrBranch"]),
        ("the auto branch differs", {"a": A, "b": base(mklCbwrAutoBranch="3")}, "VOID", ["mklCbwrAutoBranch"]),
        ("a differing model AND flags", {"a": A, "b": base(cpuModel="x", cpuFlags="y")}, "VOID", ["cpuModel", "cpuFlags"]),
        ("three arms, one differs", {"a": A, "b": base(), "c": base(cpuModel="x")}, "VOID", ["cpuModel"]),
        ("an arm with no identity (an old binary)", {"a": A, "b": {k: None for k in ("mklCbwrBranch", "mklCbwrAutoBranch", "cpuModel", "cpuFlags")}},
         "UNVERIFIED", []),
        ("an arm with cpuModel none (not Linux)", {"a": A, "b": base(cpuModel="none", cpuFlags="none")}, "UNVERIFIED", []),
        ("an arm with no report at all", {"a": A, "b": None}, "UNVERIFIED", []),
        ("both arms unidentified", {"a": None, "b": None}, "UNVERIFIED", []),
        ("one identified arm differs while a third has none", {"a": A, "b": base(cpuModel="x"), "c": None}, "VOID", ["cpuModel"]),
        ("umfpack arms (the MKL keys are none, equal)", {"a": base(mklCbwrBranch="none", mklCbwrAutoBranch="none"),
                                                         "b": base(mklCbwrBranch="none", mklCbwrAutoBranch="none")}, "SAME_CPU", []),
    )
    for label, ids, want, differ in cases:
        got = ac.s25_pair_status(ids)
        assert got["status"] == want and got["differing_keys"] == differ, f"{label}: {got}"
    # mutation: a status function that ignores the flags is seen
    assert ac.s25_pair_status({"a": A, "b": base(cpuFlags="fpu")})["status"] == "VOID"
    # --- the report over children: two children, from their logs; an unknown arm is an error
    kids = {"a": {"s25": ac.s25_report(S25_B2_SAMPLE)}, "b": {"s25": ac.s25_report(S25_B2_SAMPLE.replace("avx512f", "avx512f,amx"))}}
    pr = ac.s25_pair_report(kids, {"s25_paired_comparisons": [{"pair_id": "P", "arms": ["a", "b"]}]})
    assert [p["status"] for p in pr] == ["VOID"] and pr[0]["pair_id"] == "P" and pr[0]["differing_keys"] == ["cpuFlags"], pr
    assert ac.s25_pair_report(kids, {}) == []
    # an arm that is not a child of THIS assessment (arms run one per session; an unrelated batch holds none) is reported, not an error:
    # a registered pair must not stop the assessment of a session that does not hold all of its arms (the assessor would write no result.json)
    ab = ac.s25_pair_report(kids, {"s25_paired_comparisons": [{"pair_id": "P", "arms": ["a", "zz"]}, {"pair_id": "Q", "arms": ["zz", "yy"]}]})
    assert [(p["pair_id"], p["status"], p["arms_absent"]) for p in ab] == [("P", "ARMS_NOT_IN_THIS_RUN", ["zz"]), ("Q", "ARMS_NOT_IN_THIS_RUN", ["yy", "zz"])], ab
    # ... and that is NOT a verdict on the CPU: with every arm present the status is still computed
    assert ac.s25_pair_report({"a": kids["a"], "zz": kids["b"]}, {"s25_paired_comparisons": [{"pair_id": "P", "arms": ["a", "zz"]}]})[0]["status"] == "VOID"
    # --- wiring: report-only for `admissible`; the readers of the "s25" key are the pair report and the gate summary only
    asr = Path(ac.__file__).read_text()
    readers = [ln for ln in asr.splitlines() if '"s25"' in ln and "def " not in ln]
    assert len(readers) == 3 and not any("admissible" in ln for ln in readers), readers
    assert "s25_f3_failures" in asr and "s25_paired" in asr
    print(f"   ltassess: dtKVTangent reported (great skipped, outside the 7 limits); F3 boundary checked both sides "
          f"(4.0 PASS, 4.00001 FAIL, nan/inf FAIL, truncated UNASSESSABLE, none N/A), stiffNonPositive > 0 FAILS even at rMax 1.0 or none "
          f"(clause removed: PASS, seen) and a moved threshold is seen; "
          f"CPU identity parsed; {len(cases)} pair cases (VOID on model, flags, branch)")


# B76-5a (case-engineer 09b; design rev. 2.2 section 7). The generator's 5a-only refusals and writers, seen on the bench: each negative is
# refused for ITS reason, and with its guard removed the same defect is accepted or fails otherwise; the positives are what is written.
S25_5A_CHILD = "S25_5a_tan"
S25_5A_HALF = "S25_5a_tan_half"


def _s25_5a_gen_section() -> None:
    import importlib.util
    bench = _S25Bench()
    try:
        def gen(mutate=None, child=S25_5A_CHILD, patches=()):
            c = json.loads(bench.controls0)
            if mutate:
                mutate(c)
            bench.cp.write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n")
            g = bench.gen0
            for old, new in patches:
                assert g.count(old) == 1, f"generator patch anchor is not unique: {old[:70]!r}"
                g = g.replace(old, new)
            bench.gp.write_text(g)
            if (bench.root / "runs").exists():
                shutil.rmtree(bench.root / "runs")
            done = subprocess.run([sys.executable, "-B", str(bench.gp), child], capture_output=True, text=True)
            lines = (done.stdout + done.stderr).strip().splitlines()
            return done.returncode, (lines[-1] if lines else ""), bench.root / "runs" / child

        def kid(c, cid):
            return next(x for x in c["children"] if x["child_id"] == cid)

        def edit(cid=S25_5A_CHILD, **ch):
            def m(c):
                k = kid(c, cid)
                for key, v in ch.items():
                    if v is None:
                        k.pop(key, None)
                    else:
                        k[key] = v
            return m

        def clot(**ch):
            def m(c):
                c["series"]["s25_5a"]["clot_override"].update(ch)
            return m

        def drop_series(key):
            def m(c):
                c["series"]["s25_5a"].pop(key)
            return m

        def both(*ms):
            def m(c):
                for f in ms:
                    f(c)
            return m
        rd = lambda d, rel: (d / rel).read_text()
        # --- positives: what is written, in both children
        rc, last, d = gen()
        assert rc == 0, last
        props, phys = rd(d, "constant/phaseClotProperties"), rd(d, "constant/physicalProperties")
        assert re.search(r"(?m)^elasticCourant 0\.5;$", props) and re.search(r"(?m)^nu \[0 2 -1 0 0 0 0\] 6\.7636023629e-07;$", phys), (props[-300:], phys)
        assert re.search(r"(?m)^nu \[0 2 -1 0 0 0 0\] 6\.7636023629e-07;$", props) and "timeScale 8.333333333333334;" in props
        assert s25_lt_solver_would_refuse(props) is None, s25_lt_solver_would_refuse(props)   # the solver mirror accepts the 5a dictionary
        rc, last, d2 = gen(child=S25_5A_HALF)
        assert rc == 0 and re.search(r"(?m)^elasticCourant 0\.25;$", rd(d2, "constant/phaseClotProperties")), last
        # the floor's boundary, both sides: 9.2 um below the disc is accepted here, 8.8 um is refused below (3 x 3 um = 9 um)
        rc, last, _ = gen(clot(centre_y_m=2.42e-05))
        assert rc == 0, last
        # --- negatives
        refusal_call = "        s25_5a_refusal(child, spec)\n"
        floor_bot, floor_top = "elif _bot / _eps < 3.0:", "if _top / _eps < 3.0:"
        gc_guard = 'if child.get("Gc_j_m2") is not None and child["series"] not in ("cohesive_tear",) + S25_5A_SERIES + Y_GATE_SERIES:'
        gamma_guard = 'if child.get("wall_shear_rate_1_s") != _want_gamma:'
        pair_row = '    ("S25_5a_tan_half", "S25_5a_tan", {"elastic_courant"}, set()),\n'
        negatives = (
            # (label, mutate, child, want, patches that remove ONLY this guard)
            ("elastic_courant on a child outside the 5a series", edit("fsb_ctl_r3", elastic_courant=0.5, solver_library=None, tau_v_s=5e-4),
             "fsb_ctl_r3", "is read only for the series s25_5a", ((refusal_call, "        pass\n"),)),
            ("5a child without elastic_courant", edit(elastic_courant=None), S25_5A_CHILD, "must state elastic_courant",
             ((refusal_call, "        pass\n"),)),
            ("elastic_courant 0 (the wave CFL off)", edit(elastic_courant=0.0), S25_5A_CHILD, "must be a number in (0, 1]",
             ((refusal_call, "        pass\n"),)),
            ("elastic_courant 1.5", edit(elastic_courant=1.5), S25_5A_CHILD, "must be a number in (0, 1]", ((refusal_call, "        pass\n"),)),
            ("elastic_courant true", edit(elastic_courant=True), S25_5A_CHILD, "must be a number in (0, 1]", ((refusal_call, "        pass\n"),)),
            ("elastic_courant a string", edit(elastic_courant="0.5"), S25_5A_CHILD, "must be a number in (0, 1]", ((refusal_call, "        pass\n"),)),
            ("a 5a child on the shipped law", edit(skeleton_law=None, skeleton_law_coeffs=None, elastic_courant_modulus=None, G_pa=4.68),
             S25_5A_CHILD, "is the LT series", ((refusal_call, "        pass\n"),)),
            ("the 5a series without its plasma", both(drop_series("rho_kg_m3"), drop_series("nu_m2_s")), S25_5A_CHILD,
             "must declare rho_kg_m3 and nu_m2_s", ((refusal_call, "        pass\n"),)),
            ("end_time_s not a whole number of write intervals (no frames)", edit(end_time_s=0.0021, frame_interval_s=None, frame_fields=None),
             S25_5A_CHILD, "OpenFOAM-13 would end the run without a full write", ((refusal_call, "        pass\n"),)),
            ("end_time_s not a whole number of write intervals (frames: the older guard)", edit(end_time_s=0.0021), S25_5A_CHILD,
             "the run would end without a full write", ()),
            ("Gc override outside the cohesive and 5a series", edit("fsb_ctl_r3", Gc_j_m2=0.3698, solver_library=None, tau_v_s=5e-4), "fsb_ctl_r3",
             "only the cohesive_tear series and the S25 5a series", ((gc_guard, "if False:"),)),
            ("disc bottom gap 8.8 um (< 3 epsilonC = 9 um)", clot(centre_y_m=2.38e-05), S25_5A_CHILD, "below the 3.0 floor, and rests_on_wall is not declared",
             ((floor_bot, "elif False:"),)),
            ("lumen above the disc 7.5 um", clot(centre_y_m=2.75e-05), S25_5A_CHILD, "the lumen above the disc is", ((floor_top, "if False:"),)),
            ("a wall-resting disc with a gap", clot(rests_on_wall=True), S25_5A_CHILD, "rests_on_wall is declared but the disc's bottom gap", ()),
            ("disc outside the channel", clot(centre_x_m=0.00315), S25_5A_CHILD, "outside the", ()),
            ("5a child at the venous 250 1/s", edit(wall_shear_rate_1_s=250.0, mean_velocity_m_s=250.0 * 5e-05 / 6.0), S25_5A_CHILD,
             "wall_shear_rate_1_s must be 3200.0", ((gamma_guard, "if False:"),)),
            ("the pair differing in a second key (Gc)", edit(S25_5A_HALF, Gc_j_m2=0.3699), S25_5A_HALF,
             "FSF parent check: S25_5a_tan_half differs from S25_5a_tan in Gc_j_m2", ((pair_row, ""),)),
            ("the pair not differing in elastic_courant", edit(S25_5A_HALF, elastic_courant=0.5), S25_5A_HALF,
             "does not differ from S25_5a_tan in its named key elastic_courant", ((pair_row, ""),)),
        )
        removed = 0
        for label, mutate, child, want, patches in negatives:
            rc, last, _ = gen(mutate, child)
            assert rc != 0, f"the generator ACCEPTED {label}"
            assert want in last, f"{label}: refused for the wrong reason: {last}"
            how = "no removable guard line (an inline raise)"
            if patches:
                rc2, last2, _ = gen(mutate, child, patches)
                assert rc2 == 0 or want not in last2, f"{label}: the refusal survives the removal of its guard: {last2}"
                removed += rc2 == 0
                how = "ACCEPTED" if rc2 == 0 else "fails otherwise"
            print(f"generator refuses: {label:<54} -> {want[:58]!r}; guard removed: {how}")
        # --- the writers' read-back: a defect is refused for ITS reason; with the read-back removed it is ACCEPTED and the file shows it
        rb_call = "        s25_5a_readback(case, props, child, series)\n"
        # (the physicalProperties nu defect moved with its writer to OI-76's general writer: the 'oi76' section refuses it for the 5a child as well)
        defects = (
            ("elasticCourant written 0.5 for the dt/2 child", S25_5A_HALF,
             (('add_or_replace_entry(props, "elasticCourant", repr(float(child["elastic_courant"])))',
               'add_or_replace_entry(props, "elasticCourant", repr(0.5))'),), "elasticCourant written as",
             ("constant/phaseClotProperties", r"(?m)^elasticCourant 0\.5;$")),
        )
        for label, child, patches, want, (rel, shown) in defects:
            rc, last, _ = gen(None, child, patches)
            assert rc != 0 and want in last, f"{label}: {rc} {last}"
            rc2, last2, d3 = gen(None, child, patches + ((rb_call, "        pass\n"),))
            assert rc2 == 0, (label, last2)
            assert re.search(shown, rd(d3, rel)), f"{label}: the defective file does not show it"
            print(f"read-back refuses: {label:<54} -> {want!r}; read-back removed: ACCEPTED (the file holds the defect)")
        # --- the table checker on a full generation, and its own mutations
        spec = importlib.util.spec_from_file_location("check_s25_5a_table", CASE.parents[1] / "scripts" / "check_s25_5a_table.py")
        chk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chk)
        with tempfile.TemporaryDirectory(prefix="am2020-s25-5a-") as tmp:
            root = build_case(Path(tmp))
            assert chk.check(root / "runs") == [], chk.check(root / "runs")
            assert chk.self_test(root / "runs") == 0
        print(f"   5agen: both 5a children written to the table (46 numeric keys, 9 words, 12 patch rows); {len(negatives)} generator refusals each for "
              f"their own reason, {removed} accepted with the guard removed; 1 writer defect seen by the read-back and accepted without it (the nu defect: section oi76); "
              f"the independent table checker passes and catches its 20 mutations")
    finally:
        bench.close()


# ---- case-engineer 09c: OI-76 (S25 section 12.8) and the B76-OFF baseline arm -------------------------------------------------------------
# Run alone: python3 -B -c "import dry_run; dry_run.s25_build2_checks('oi76')"  /  ('b76off')
def _gen_bench_section(label: str):
    """A scratch copy of the case and a `gen(mutate, child, patches)` that rewrites the controls entry, patches the generator source (each anchor
    unique), generates that child alone and returns (rc, last line, the child's run dir)."""
    bench = _S25Bench()

    def gen(mutate=None, child="xu5_g250", patches=()):
        c = json.loads(bench.controls0)
        if mutate:
            mutate(c)
        bench.cp.write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n")
        g = bench.gen0
        for old, new in patches:
            assert g.count(old) == 1, f"{label}: generator patch anchor is not unique: {old[:70]!r}"
            g = g.replace(old, new)
        bench.gp.write_text(g)
        if (bench.root / "runs").exists():
            shutil.rmtree(bench.root / "runs")
        done = subprocess.run([sys.executable, "-B", str(bench.gp), child], capture_output=True, text=True)
        lines = (done.stdout + done.stderr).strip().splitlines()
        return done.returncode, (lines[-1] if lines else ""), bench.root / "runs" / child
    return bench, gen


def _s25_oi76_section() -> None:
    """OI-76: the generator writes the series' declared nu into constant/physicalProperties (the momentum equation's) as well as
    phaseClotProperties (the drag's), reads both back, and touches no series that declares none. Each defect is refused for its own reason by the
    read-back and, with the read-back removed, ACCEPTED with the defective file on disk (so the read-back is what refuses it)."""
    bench, gen = _gen_bench_section("oi76")
    try:
        rd = lambda d, rel: (d / rel).read_text()
        xu_nu, xu_rho = "3.902439024390244e-06", "1025.0"
        # --- positives: Xu carries its nu in BOTH files; a series that declares no plasma keeps the template's physicalProperties bytes
        rc, last, d = gen()
        assert rc == 0, last
        assert re.search(rf"(?m)^nu \[0 2 -1 0 0 0 0\] {re.escape(xu_nu)};$", rd(d, "constant/physicalProperties")), rd(d, "constant/physicalProperties")
        assert re.search(rf"(?m)^nu \[0 2 -1 0 0 0 0\] {re.escape(xu_nu)};$", rd(d, "constant/phaseClotProperties"))
        assert re.search(rf"(?m)^rho \[1 -3 0 0 0 0 0\] {re.escape(xu_rho)};$", rd(d, "constant/phaseClotProperties"))
        tmpl = (bench.root.parent / "AM2020_straight_tube_growth_mechanics_smoke" / "template" / "constant" / "physicalProperties").read_text()
        rc, last, d2 = gen(child="P73_G4_r3_40")
        assert rc == 0, last
        assert rd(d2, "constant/physicalProperties") == tmpl, "a series that declares no nu_m2_s had its physicalProperties changed"
        # --- the read-back, with each writer defect (patches replace ONE expression)
        write = '    replace_entry(case / "constant" / "physicalProperties", "nu", repr(float(series["nu_m2_s"])))\n'
        props_nu = '"nu": series.get("nu_m2_s", fixed["nu_m2_s"]),\n'
        props_rho = '"rho": series.get("rho_kg_m3", fixed["rho_kg_m3"]),\n'
        rb = "        series_plasma_readback(case, props, child, series, fixed)\n"
        defects = (
            # (label, child, patches, the refusal's words, (file, regex of the defect the file then shows))
            ("a writer that skips physicalProperties (the OI-76 defect)", "xu5_g250", ((write, "    pass\n"),), "in physicalProperties and",
             ("constant/physicalProperties", r"(?m)^nu \[0 2 -1 0 0 0 0\] 3\.3333333333e-6;$")),
            ("a writer that rounds the physicalProperties nu", "xu5_g250",
             ((write, '    replace_entry(case / "constant" / "physicalProperties", "nu", repr(round(float(series["nu_m2_s"]), 7)))\n'),),
             "in physicalProperties and", ("constant/physicalProperties", r"(?m)^nu \[0 2 -1 0 0 0 0\] 3\.9e-06;$")),
            ("a writer that writes phaseClotProperties' nu at the main case's", "xu5_g250",
             ((props_nu, '"nu": fixed["nu_m2_s"],\n'),), "in physicalProperties and",
             ("constant/phaseClotProperties", r"(?m)^nu \[0 2 -1 0 0 0 0\] 3\.3333333333e-06;$")),
            ("a writer that writes phaseClotProperties' rho at the main case's", "xu5_g250",
             ((props_rho, '"rho": fixed["rho_kg_m3"],\n'),), "in phaseClotProperties; the series says",
             ("constant/phaseClotProperties", r"(?m)^rho \[1 -3 0 0 0 0 0\] 1050\.0;$")),
            ("the 5a series' physicalProperties nu skipped", "S25_5a_tan", ((write, "    pass\n"),), "in physicalProperties and",
             ("constant/physicalProperties", r"(?m)^nu \[0 2 -1 0 0 0 0\] 3\.3333333333e-6;$")),
        )
        for label, child, patches, want, (rel, shown) in defects:
            rc, last, _ = gen(None, child, patches)
            assert rc != 0 and want in last, f"{label}: {rc} {last}"
            rc2, last2, d3 = gen(None, child, patches + ((rb, "        pass\n"),))
            assert rc2 == 0, (label, last2)
            assert re.search(shown, rd(d3, rel)), f"{label}: the defective file does not show it"
            print(f"read-back refuses: {label:<62} -> {want!r}; read-back removed: ACCEPTED (the file holds the defect)")
        # a series that declares a nu but whose child is generated from a series with NO declaration is out of the fix's reach by design: the
        # main case's own children are read back against material_fixed, so a template drift in physicalProperties is refused too
        rc, last, _ = gen(None, "P73_G4_r3_40", ((write, "    pass\n"),))
        assert rc == 0, last   # nothing to write for this series: the skip is not a defect there
        print(f"   oi76: Xu nu {xu_nu} in both files, rho {xu_rho}; P73_G4_r3_40's physicalProperties is the template's bytes; "
              f"{len(defects)} writer defects each refused by the read-back for its own reason and accepted without it")
    finally:
        bench.close()


def _s25_b76off_section() -> None:
    """B76-OFF-A on the baseline library: the generator knows `baseline`, treats it as the S23 build that predates S25, and the pair differs in the
    binary alone."""
    bench, gen = _gen_bench_section("b76off")
    try:
        A, B = "B76_OFF_A", "B76_OFF_B"
        kid = lambda c, cid: next(x for x in c["children"] if x["child_id"] == cid)

        def edit(cid=A, **ch):
            def m(c):
                k = kid(c, cid)
                for key, v in ch.items():
                    if v is None:
                        k.pop(key, None)
                    else:
                        k[key] = v
            return m
        units = 'S25_TIME_SCALE_UNIT_LIBRARIES = ("s22", "p2prime", "baseline")'
        unread = 'S25_LT_UNREAD_LIBRARIES = ("s22", "p2prime", "baseline")'
        build = "S23_BUILD_LIBRARIES = (None, \"baseline\")"
        pair_row = '    ("B76_OFF_A", "B76_OFF_B", {"solver_library"}, set()),\n'
        pair_row_b = '    ("B76_OFF_B", "P73_G4_r3_40", set(), {"end_time_s"}),\n'
        lt = {"skeleton_law": "valanisLandelTrilinear", "skeleton_law_coeffs": {"E1_pa": 234.0, "E2_pa": 5820.0, "D1": 0.107, "D2": 0.574, "kappa": 0.98}}
        rc, last, d = gen(None, A)
        assert rc == 0, last
        env = (d / "child.env").read_text()
        assert "PHASE_CLOT_BASELINE_LIBBIN" in env and "return 97" in env and "return 98" in env and 'export LD_LIBRARY_PATH="${PHASE_CLOT_BASELINE_LIBBIN}' in env
        assert json.loads((d / "child.json").read_text())["solver_library"] == "baseline"
        negatives = (
            # (label, mutate, child, want, patches that remove ONLY this guard; the first three also drop the pair row, because the pair check
            # (B76_OFF_A differs from B76_OFF_B in solver_library ALONE) is a second guard that would refuse the same child)
            ("baseline with time_scale 8.33 (the library never reads timeScale)", edit(time_scale=8.33), A, "that library predates S25", ((units, 'S25_TIME_SCALE_UNIT_LIBRARIES = ("s22", "p2prime")'), (pair_row, ""))),
            ("baseline with the LT law (the library never reads it)", edit(G_pa=None, **lt), A, "that library predates S25 build 2", ((unread, 'S25_LT_UNREAD_LIBRARIES = ("s22", "p2prime")'), (pair_row, ""))),
            ("baseline without R3 (it IS the S23 build: R3 is required)", edit(network_viscosity=None, tau_v_s=None), A, "network_viscosity none (or absent",
             ((build, "S23_BUILD_LIBRARIES = (None,)"), (pair_row, ""))),
            ("an unknown library name", edit(solver_library="baseline2"), A, "is not one of p2prime, s22, baseline", ()),
            ("the pair differing in a second key (k0p_m2)", edit(A, k0p_m2=5e-13), A, "FSF parent check: B76_OFF_A differs from B76_OFF_B in k0p_m2", ((pair_row, ""),)),
            ("the pair not differing in solver_library", edit(A, solver_library=None), A, "does not differ from B76_OFF_B in its named key solver_library", ((pair_row, ""),)),
            ("B76_OFF_B drifting from P73_G4_r3_40 in a run control", edit(B, wall_budget_h=12.0), B, "FSF parent check: B76_OFF_B differs from P73_G4_r3_40 in wall_budget_h",
             ((pair_row_b, ""), (pair_row, ""))),
            ("B76_OFF_B with the end time of its parent (no difference)", edit(B, end_time_s=0.04), B, "does not differ from P73_G4_r3_40 in its named key end_time_s", ((pair_row_b, ""), (pair_row, ""))),
        )
        removed = 0
        for label, mutate, child, want, patches in negatives:
            rc, last, _ = gen(mutate, child)
            assert rc != 0, f"the generator ACCEPTED {label}"
            assert want in last, f"{label}: refused for the wrong reason: {last}"
            how = "no removable guard line (an inline raise)"
            if patches:
                rc2, last2, _ = gen(mutate, child, patches)
                assert rc2 == 0 or want not in last2, f"{label}: the refusal survives the removal of its guard: {last2}"
                removed += rc2 == 0
                how = "ACCEPTED" if rc2 == 0 else f"fails otherwise ({last2[:90]})"
            print(f"generator refuses: {label:<66} -> {want[:52]!r}; guard removed: {how}")
        # --- the runner side, offline: the library table, the flag and the start-of-session check know `baseline`
        runner_src = (CASE / "scripts" / "run_main_case_colab.py").read_text()
        assert '"baseline": ("PHASE_CLOT_BASELINE_LIBBIN", "baseline_libbin"' in runner_src and '"--baseline-solver"' in runner_src
        print(f"   b76off: child.env for baseline (97/98/prefix); {len(negatives)} generator refusals each for its own reason, {removed} accepted with the guard removed")
    finally:
        bench.close()


def _s25_b78_section() -> None:
    """BATCH 78 (S25.1 re-check; case-engineer 06): the six children generate alone, the parent-pair rows are the guard that refuses a drift (each negative seen to be refused
    for its stated reason and, with ONLY that guard removed from a scratch generator, ACCEPTED), the RT pair is scored by an s25_rt1_pairs entry, and an 18 ms end on a 5 ms write
    interval is refused. Run alone: python3 -B -c "import dry_run; dry_run.s25_build2_checks('b78')"."""
    bench, gen = _gen_bench_section("b78")
    try:
        A, B, C1, C2, RA, RB = "B78_OFF_A", "B78_OFF_B", "B78_CTL_1", "B78_CTL_2", "B78_RT_5a_A", "B78_RT_5a_B"
        kid = lambda c, cid: next(x for x in c["children"] if x["child_id"] == cid)

        def edit(cid, **ch):
            def m(c):
                k = kid(c, cid)
                for key, v in ch.items():
                    if v is None:
                        k.pop(key, None)
                    else:
                        k[key] = v
            return m
        row_b = '    ("B78_OFF_B", "P73_G4_r3_40", set(), {"end_time_s", "write_interval_s"}),\n'
        row_a = '    ("B78_OFF_A", "B78_OFF_B", {"solver_library"}, set()),\n'
        row_c1 = '    ("B78_CTL_1", "B78_OFF_B", set(), {"end_time_s", "write_interval_s"}),\n'
        row_c2 = '    ("B78_CTL_2", "B78_CTL_1", set(), set()),\n'
        row_rb = '    ("B78_RT_5a_B", "B78_RT_5a_A", set(), {"restart_from_child", "restart_index"}),\n'
        call_rt1 = "    s25_rt1_pairs_refusal(controls)\n"
        # --- positives: every B78 child generates alone, A on the baseline pin, the others on the package build
        for cid in (A, B, C1, C2, RA, RB):
            rc, last, d = gen(None, cid)
            assert rc == 0, (cid, last)
            j = json.loads((d / "child.json").read_text())
            assert j["child_id"] == cid and (d / "child.env").exists() == (cid == A), cid
        negatives = (
            ("B78_OFF_B drifting from P73_G4_r3_40 in a run control", edit(B, wall_budget_h=12.0), B, "FSF parent check: B78_OFF_B differs from P73_G4_r3_40 in wall_budget_h", ((row_b, ""), (row_a, ""), (row_c1, ""))),
            ("B78_OFF_B with its parent's end time and write interval (no difference)", edit(B, end_time_s=0.04, write_interval_s=0.005), B,
             "does not differ from P73_G4_r3_40 in its named key end_time_s, write_interval_s", ((row_b, ""), (row_a, ""), (row_c1, ""))),
            ("B78_OFF_A differing from B78_OFF_B in a second key (k0p_m2)", edit(A, k0p_m2=5e-13), A, "FSF parent check: B78_OFF_A differs from B78_OFF_B in k0p_m2", ((row_a, ""),)),
            ("B78_CTL_2 drifting from B78_CTL_1 (k0p_m2)", edit(C2, k0p_m2=5e-13), C2, "FSF parent check: B78_CTL_2 differs from B78_CTL_1 in k0p_m2", ((row_c2, ""),)),
            ("B78_RT_5a_B restarting from BATCH 76's RT_5a_A", edit(RB, restart_from_child="RT_5a_A"), RB, "must restart from B78_RT_5a_A (restart_from_child)", ((call_rt1, ""), (row_rb, ""))),
            ("an end time that is not a whole number of write intervals (18 ms on 5 ms)", edit(B, write_interval_s=0.005), B, "is not a whole number of write intervals", ()),
        )
        removed = 0
        for label, mutate, child, want, patches in negatives:
            rc, last, _ = gen(mutate, child)
            assert rc != 0, f"the generator ACCEPTED {label}"
            assert want in last, f"{label}: refused for the wrong reason: {last}"
            how = "no removable guard line (an inline raise)"
            if patches:
                rc2, last2, _ = gen(mutate, child, patches)
                assert rc2 == 0 or want not in last2, f"{label}: the refusal survives the removal of its guard: {last2}"
                removed += rc2 == 0
                how = "ACCEPTED" if rc2 == 0 else f"fails otherwise ({last2[:90]})"
            print(f"generator refuses: {label:<72} -> {want[:52]!r}; guard removed: {how}")

        def drop_pair(c):
            c["s25_rt1_pairs"] = [e for e in c["s25_rt1_pairs"] if e["pair_id"] != "B78_RT_5a"]
        label, want = "the B78_RT_5a pair without its s25_rt1_pairs entry (nothing would score it)", "every RT-1 child is in exactly one s25_rt1_pairs entry; unscored ['B78_RT_5a_A', 'B78_RT_5a_B']"
        rc, last, _ = gen(drop_pair, RA)
        assert rc != 0 and want in last, (label, rc, last)
        rc2, last2, _ = gen(drop_pair, RA, ((call_rt1, ""),))
        assert rc2 == 0, (label, last2)
        removed += 1
        print(f"generator refuses: {label:<72} -> {want[:52]!r}; guard removed: ACCEPTED")
        print(f"   b78: the six children generate alone; {len(negatives) + 1} generator refusals each for its own reason, {removed} accepted with only their guard removed")
    finally:
        bench.close()


# ---- case-engineer 09d: B76-RT (RT-1, the restart pairs) ---------------------------------------------------------------------------------
# Run alone: python3 -B -c "import dry_run; dry_run.s25_build2_checks('rt1')"   (about a minute; the sections below are rt1gen, rt1watch, rt1allrun, rt1gates; "stab", the S25.1 odd-even gate, runs alone as s25_build2_checks("stab"))
_RT1_FIELDS = (("thetaC", "scalar"), ("thetaH", "scalar"), ("w", "scalar"), ("q", "scalar"), ("Vs", "vector"),
               ("thetaPendingB", "symmTensor"), ("thetaPendingP", "scalar"))
_RT1_COMP = {"scalar": 1, "vector": 3, "symmTensor": 6}
_RT1_CLASS = {"scalar": "volScalarField", "vector": "volVectorField", "symmTensor": "volSymmTensorField"}


def _rt1_write_field(path: Path, kind: str, rows) -> None:
    body = "\n".join(("(" + " ".join(f"{x:.17g}" for x in r) + ")") if isinstance(r, (list, tuple)) else f"{r:.17g}" for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {_RT1_CLASS[kind]};\n    object {path.name};\n}}\n\n"
                    f"dimensions [0 0 0 0 0 0 0];\n\ninternalField nonuniform List<{kind}> \n{len(rows)}\n(\n{body}\n)\n;\n\nboundaryField\n{{\n}}\n\n"
                    "// ************************************************************************* //\n")


def _rt1_state(step: int, n: int = 30, seed: int = 7) -> dict:
    """Deterministic per-step fields: theta = thetaC + thetaH moves by 1e-4..1e-3 per step, Vs by a percent, the rest is smooth."""
    import random
    rng = random.Random(seed)
    base = {name: [[rng.uniform(0.1, 1.0) for _ in range(_RT1_COMP[kind])] for _ in range(n)] for name, kind in _RT1_FIELDS}
    rate = [rng.uniform(1e-4, 1e-3) for _ in range(n)]
    out = {}
    for name, kind in _RT1_FIELDS:
        rows = []
        for i in range(n):
            v = list(base[name][i])
            if name == "thetaC":
                v = [v[0] * 0.4 + step * rate[i]]
            elif name == "Vs":
                v = [x * (1.0 + 0.01 * step) for x in v]
            elif name in ("thetaPendingB", "thetaPendingP"):
                v = [x * 1e-3 * (1 + 0.001 * step) for x in v]
            rows.append(v if kind != "scalar" else v[0])
        out[name] = (kind, rows)
    return out


def _rt1_write_uniform(path: Path, value: float) -> None:
    """A scalar field the way the solver writes one whose cells are all equal: `internalField uniform v;` (thetaH before any deposit)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class volScalarField;\n    object {path.name};\n}}\n\n"
                    f"dimensions [0 0 0 0 0 0 0];\n\ninternalField uniform {value:.17g};\n\nboundaryField\n{{\n}}\n\n"
                    "// ************************************************************************* //\n")


def _rt1_dirs(root: Path, *, n0: int = 42, mutate=None, dt_text: str = "4.7619047619047618e-06", n: int = 30,
              fields=tuple(f for f, _ in _RT1_FIELDS), state_a: str = "seeded", state_b: str = "restored", vs_zero: bool = False,
              void_den: bool = False, write_precision: int | None = 17, theta_h_uniform: bool = False) -> tuple:
    """runs/A: full writes at steps n0-4 .. n0+5 and a log; runs/B: full writes at n0 .. n0+5 and a log. A perfect restart: B's values ARE A's."""
    a, b = root / "A", root / "B"
    s_end = n0 + 5
    for case, steps in ((a, range(n0 - 4, s_end + 1)), (b, range(n0, s_end + 1))):
        for k in steps:
            st = _rt1_state(k, n)
            if vs_zero:
                st["Vs"] = ("vector", [[0.0, 0.0, 0.0] for _ in range(n)])
            if void_den and k >= n0:
                st["thetaC"] = ("scalar", [r + 1e-12 * (k - n0) for r in _rt1_state(n0, n)["thetaC"][1]])    # theta moves by 1e-12 per step
            d = case / f"0.{k:04d}"
            for f in fields:
                if f == "thetaH" and theta_h_uniform:
                    _rt1_write_uniform(d / f, 0.0)      # as the solver writes thetaH before a platelet has deposited
                else:
                    _rt1_write_field(d / f, *st[f])
            (d / "uniform").mkdir(exist_ok=True)
            (d / "uniform" / "time").write_text(f"FoamFile\n{{\n    format ascii;\n}}\nvalue 0.{k:04d};\nname \"0.{k:04d}\";\nindex {k};\ndeltaT 4.7619047619047618e-06;\n")
            (d / "uniform" / "phaseClotState").write_text("FoamFile\n{\n    format ascii;\n}\nexplicitDiffusionRate 0;\n")
    if write_precision is not None:
        for case in (a, b):
            (case / "system").mkdir(exist_ok=True)
            (case / "system" / "controlDict").write_text(f"FoamFile\n{{\n    format ascii;\n}}\nwriteFormat ascii;\nwritePrecision {write_precision};\ntimePrecision 10;\n")
    for case, first, state in ((a, 1, state_a), (b, n0 + 1, state_b)):
        last = s_end
        lines = [f"PHASE_CLOT_S25_RESTART state={state}"]
        for k in range(first, last + 1):
            lines.append(f"PHASE_CLOT_DT step={k} time=0.{k:04d} deltaT={dt_text} dtFluidCo=1 dtWave=1")
        (case / "log.phaseClotAxisymmetricFluid").write_text("\n".join(lines) + "\n")
    if mutate:
        mutate(a, b)
    return a, b


def _rt1_rewrite(d: Path, field: str, fn) -> None:
    kind = dict(_RT1_FIELDS)[field]
    n = int(re.search(r"List<\w+>\s*\n(\d+)\n", (d / field).read_text()).group(1))
    rows = []
    for ln in (d / field).read_text().split("(\n", 1)[1].split("\n)\n;", 1)[0].splitlines():
        nums = [float(x) for x in re.findall(r"[-+0-9.eE]+", ln)]
        rows.append(nums if kind != "scalar" else nums[0])
    assert len(rows) == n
    _rt1_write_field(d / field, kind, fn(rows))


def _rt1_entry(**kw) -> dict:
    e = {"pair_id": "RT_test", "a": "A", "b": "B", "n0": 42, "required_fields": [], "needs_vs_nonzero": False}
    e.update(kw)
    return e


def _s25_rt1_gates_section() -> None:
    """RT-1a/b/c, each seen to FAIL on a synthetic restart (A and B written to disk the way the solver writes them), then each guard removed from a
    scratch copy of the assessor and its negative seen to PASS."""
    import importlib.util
    path = Path(__file__).resolve().parent / "assess_case.py"
    src = path.read_text()

    def load(text: str, tag: str):
        with tempfile.TemporaryDirectory(prefix="am2020-rt1-mod-") as t:
            p = Path(t) / f"assess_case_{tag}.py"
            p.write_text(text)
            spec = importlib.util.spec_from_file_location(f"assess_case_rt1_{tag}", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    ac = load(src, "real")

    def verdict(mod, entry=None, **kw):
        with tempfile.TemporaryDirectory(prefix="am2020-rt1-") as t:
            a, b = _rt1_dirs(Path(t), **{k: v for k, v in kw.items() if k != "entry"})
            return mod.rt1_pair(a, b, entry or kw.get("entry") or _rt1_entry())

    def gates(r):
        return {k: v["status"] for k, v in r["gates"].items()}
    # ---- the positive: a perfect restart passes all three gates, m is exactly 0, 5 steps compared
    r = verdict(ac)
    assert r["status"] == "PASS" and gates(r) == {"RT-1a": "PASS", "RT-1b": "PASS", "RT-1c": "PASS"}, r
    assert r["gates"]["RT-1a"]["m"] == 0.0 and r["gates"]["RT-1a"]["denominator"] > 1e-4, r["gates"]["RT-1a"]
    assert r["gates"]["RT-1c"]["fields"] == 7 and r["gates"]["RT-1b"]["digits_printed"] == 17, r["gates"]
    entry5a = _rt1_entry(required_fields=["thetaPendingB", "thetaPendingP"], needs_vs_nonzero=True)
    assert verdict(ac, entry=entry5a)["status"] == "PASS"
    r5 = verdict(ac, entry=entry5a)
    assert r5["vs_max_abs_at_n0"] > 0 and r5["pending_persistence_exercised"] is True and r["pending_max_abs_at_n0"]["thetaPendingP"] > 0
    shown = ["a perfect restart (5a entry, required thetaPendingB/P, Vs != 0): PASS 1a 1b 1c, m = 0; pending contents nonzero at N0 -> persistence exercised"]
    def zero_pending(a, b):
        for d in (a, b):
            for k in range(30, 50):
                for f in ("thetaPendingB", "thetaPendingP"):
                    if (d / f"0.{k:04d}" / f).is_file():
                        _rt1_rewrite(d / f"0.{k:04d}", f, lambda rows: [([0.0] * len(x) if isinstance(x, list) else 0.0) for x in rows])
    rz = verdict(ac, entry=entry5a, mutate=zero_pending)
    assert rz["status"] == "PASS" and rz["pending_persistence_exercised"] is False, rz
    shown.append("pending contents all 0 at N0: PASS but pending_persistence_exercised false (reported, not gated)")

    def neg(label, want_status, want_gate=None, want_gate_status=None, text=None, entry=None, **kw):
        r = verdict(ac, entry=entry, **kw)
        assert r["status"] == want_status, f"{label}: status {r['status']} (wanted {want_status}): {json.dumps(r)[:600]}"
        if want_gate:
            assert r["gates"][want_gate]["status"] == want_gate_status, f"{label}: {want_gate} is {r['gates'][want_gate]}"
        if text:
            blob = json.dumps(r)
            assert text in blob, f"{label}: {text!r} not in the report: {blob[:600]}"
        shown.append(f"{label}: {want_status}" + (f" ({want_gate} {want_gate_status})" if want_gate else ""))
        return r
    # ---- RT-1a
    def zero_carrier(a, b):          # B's step N0+1 theta did not move from N0 (the carrier-0 failure, m = 1)
        for f in ("thetaC",):
            src_rows = _rt1_state(42)[f][1]
            _rt1_rewrite(b / "0.0043", f, lambda rows: src_rows)
    r = neg("a restart whose first step does not move theta (m = 1)", "FAIL", "RT-1a", "FAIL", text="RT-1a", mutate=zero_carrier)
    assert 0.99 < r["gates"]["RT-1a"]["m"] < 1.01, r["gates"]["RT-1a"]

    def off_by(frac):                # B's theta at N0+1 off by frac x the step change, in one cell
        def m(a, b):
            t1 = _rt1_state(43)["thetaC"][1]
            t0 = _rt1_state(42)["thetaC"][1]
            den = max(abs(y - x) for x, y in zip(t0, t1))
            _rt1_rewrite(b / "0.0043", "thetaC", lambda rows: [rows[0] + frac * den] + rows[1:])
        return m
    r = neg("m = 0.99e-4 (B76's gate passed it; S25.1 passes only exactly 0)", "FAIL", "RT-1a", "FAIL", mutate=off_by(0.99e-4))
    assert 0.98e-4 < r["gates"]["RT-1a"]["m"] < 1.0e-4
    import math as _m

    def one_ulp(a, b):               # B's theta at N0+1 is ONE ulp off in one cell
        _rt1_rewrite(b / "0.0043", "thetaC", lambda rows: [_m.nextafter(rows[0], 2.0)] + rows[1:])
    r = neg("RT-1a: B's theta one ulp off in one cell", "FAIL", "RT-1a", "FAIL", mutate=one_ulp)
    assert r["gates"]["RT-1a"]["numerator_thetaC"] > 0.0 and r["gates"]["RT-1a"]["m"] < 1e-12, r["gates"]["RT-1a"]
    r = neg("m = 1.01e-4", "FAIL", "RT-1a", "FAIL", mutate=off_by(1.01e-4))
    assert 1.0e-4 < r["gates"]["RT-1a"]["m"] < 1.02e-4
    r = neg("the denominator is 1e-12 (< 1e-8): the test cannot discriminate", "VOID", "RT-1a", "VOID", text="cannot discriminate", void_den=True)
    assert r["gates"]["RT-1a"]["denominator"] < 1e-8
    # ---- RT-1b
    def dt_off(rel):
        def m(a, b):
            lg = b / "log.phaseClotAxisymmetricFluid"
            bad = f"{4.7619047619047618e-06 * (1 + rel):.17g}"
            lg.write_text(lg.read_text().replace("step=45 time=0.0045 deltaT=4.7619047619047618e-06", f"step=45 time=0.0045 deltaT={bad}"))
        return m
    r = neg("dt of step 45 off by 1e-10 relative", "FAIL", "RT-1b", "FAIL", mutate=dt_off(1e-10))
    assert r["gates"]["RT-1b"]["worst_step"] == 45 and 0.9e-10 < r["gates"]["RT-1b"]["worst_rel"] < 1.1e-10
    r = neg("dt of step 45 off by 1e-13 relative (B76's gate passed it)", "FAIL", "RT-1b", "FAIL", mutate=dt_off(1e-13))
    assert r["gates"]["RT-1b"]["worst_step"] == 45 and r["gates"]["RT-1b"]["worst_rel"] > 0.0
    neg("the logs print 12 digits: an exact equality is not resolvable", "VOID", "RT-1b", "VOID", text="below its resolution", dt_text="4.76190476190e-06")

    def drop_dt(a, b):
        lg = b / "log.phaseClotAxisymmetricFluid"
        lg.write_text("\n".join(x for x in lg.read_text().splitlines() if "step=44 " not in x) + "\n")
    neg("B's log lacks the step-44 record", "VOID", "RT-1b", "VOID", text="no PHASE_CLOT_DT record", mutate=drop_dt)
    # ---- RT-1c
    neg("B drops thetaPendingP at N0+5", "FAIL", "RT-1c", "FAIL", text="thetaPendingP: dropped", mutate=lambda a, b: (b / "0.0047" / "thetaPendingP").unlink())
    neg("B drops a field A has (q)", "FAIL", "RT-1c", "FAIL", text="q: dropped", mutate=lambda a, b: (b / "0.0047" / "q").unlink())

    def field_off(field, frac, cell=3):
        def m(a, b):
            _rt1_rewrite(b / "0.0047", field, lambda rows: [(r * (1 + frac) if i == cell else r) for i, r in enumerate(rows)])
        return m
    r = neg("thetaH off by 1e-6 relative at N0+5", "FAIL", "RT-1c", "FAIL", text="thetaH", mutate=field_off("thetaH", 1e-6))
    neg("thetaH off by 1e-9 relative (B76's gate passed it)", "FAIL", "RT-1c", "FAIL", text="thetaH", mutate=field_off("thetaH", 1e-9))

    def w_one_ulp(a, b):
        _rt1_rewrite(b / "0.0047", "w", lambda rows: [rows[0]] + [_m.nextafter(rows[1], 2.0)] + rows[2:])
    r = neg("RT-1c: w one ulp off in one cell at N0+5", "FAIL", "RT-1c", "FAIL", text="w: max|B-A|", mutate=w_one_ulp)
    neg("w off by 1e-7 relative", "FAIL", "RT-1c", "FAIL", text="w:", mutate=field_off("w", 1e-7))
    neg("a required field is absent from BOTH (thetaPendingB)", "FAIL", "RT-1c", "FAIL", text="required field thetaPendingB is absent",
        entry=entry5a, fields=("thetaC", "thetaH", "w", "q", "Vs", "thetaPendingP"))
    neg("the same fields absent from both, field not required (the P73 pair)", "PASS", "RT-1c", "PASS",
        fields=("thetaC", "thetaH", "w", "q", "Vs", "thetaPendingP"))
    neg("a NaN in B's field", "FAIL", "RT-1c", "FAIL", text="not finite",
        mutate=lambda a, b: _rt1_rewrite(b / "0.0047", "q", lambda rows: [float("nan")] + rows[1:]))
    neg("B's field has another size", "FAIL", "RT-1c", "FAIL", text="not the same kind and size",
        mutate=lambda a, b: _rt1_rewrite(b / "0.0047", "q", lambda rows: rows[:-1]))

    def zero_vs_b(a, b):             # A's field is exactly 0 and B's is 1e-20: rel = inf, not 0/0
        _rt1_rewrite(a / "0.0047", "thetaH", lambda rows: [0.0] * len(rows))
        _rt1_rewrite(b / "0.0047", "thetaH", lambda rows: [0.0] * (len(rows) - 1) + [1e-20])
    neg("a field that is 0 in A and 1e-20 in B", "FAIL", "RT-1c", "FAIL", text="thetaH", mutate=zero_vs_b)
    # ---- preconditions
    neg("the 5a pair with Vs = 0 at N0 (the skeleton has not moved)", "VOID", None, None, text="skeleton has not moved", entry=entry5a, vs_zero=True)
    neg("a P73-type pair with Vs = 0 is not void for it", "PASS", entry=_rt1_entry(needs_vs_nonzero=False), vs_zero=True)
    neg("A says state=restored (not a cold start)", "VOID", text="not a cold start", state_a="restored")
    neg("B says state=seeded (the restart did not restore)", "FAIL", text="did not read uniform/phaseClotState", state_b="seeded")
    neg("A's step-N0 directory purged (a late stop)", "VOID", text="purged", mutate=lambda a, b: shutil.rmtree(a / "0.0042"))
    neg("B's log missing", "VOID", mutate=lambda a, b: (b / "log.phaseClotAxisymmetricFluid").unlink())
    # ---- S25.1 C(a): a uniform-written field is broadcast, never a parse defect
    r = neg("thetaH written `uniform` in every directory (the solver's early form): RT-1a is judged, not VOID", "PASS", "RT-1a", "PASS", theta_h_uniform=True)
    assert r["gates"]["RT-1a"]["m"] == 0.0 and r["gates"]["RT-1a"]["denominator"] > 1e-4 and r["gates"]["RT-1c"]["status"] == "PASS", r["gates"]

    def uniform_differs(a, b):       # B's thetaH at N0+1 is uniform 1e-3, A's uniform 0: the broadcast must SEE it
        _rt1_write_uniform(b / "0.0043" / "thetaH", 1e-3)
    r = neg("thetaH uniform 0 in A and uniform 1e-3 in B at N0+1: broadcast sees it", "FAIL", "RT-1a", "FAIL", theta_h_uniform=True, mutate=uniform_differs)
    assert r["gates"]["RT-1a"]["numerator"] > 0.0, r["gates"]["RT-1a"]
    neg("thetaH uniform in A and nonuniform in B at N0+5: RT-1c names the kind", "FAIL", "RT-1c", "FAIL", text="not the same kind and size",
        theta_h_uniform=True, mutate=lambda a, b: _rt1_write_field(b / "0.0047" / "thetaH", "scalar", [0.0] * 30))
    # ---- S25.1 D3: the bitwise claim needs a file written at 17 digits
    neg("controlDict writePrecision 6: a bitwise comparison is below the resolution of the files", "VOID", text="writePrecision", write_precision=6)
    neg("no controlDict: the precision cannot be read", "VOID", text="writePrecision", write_precision=None)
    # ---- the registry: a pair whose arms are not in the assessed run is not an error
    rep = ac.s25_rt1_report(Path("/nonexistent"), {"x": {}}, {"s25_rt1_pairs": [_rt1_entry(pair_id="p")]})
    assert rep == [{"pair_id": "p", "status": "ARMS_NOT_IN_THIS_RUN", "arms_absent": ["A", "B"]}], rep
    shown.append("arms not in this run: ARMS_NOT_IN_THIS_RUN")
    # ---- each guard, deleted from a scratch copy of the assessor, lets its negative PASS (the gate can fail BECAUSE of it)
    def mutated(old: str, new: str, tag: str):
        # y_rt_pair (the symmetric Y's restart pair, case-engineer 27) repeats several of rt1_pair's guard lines VERBATIM, after it: a count of 2 is accepted when the FIRST copy lies
        # before `def y_rt_pair(` and the second after it, and then only rt1_pair's copy (the first) is mutated, so this section still judges rt1_pair alone. check_y_symmetric.py
        # deletes y_rt_pair's copies one at a time.
        n = src.count(old)
        assert n in (1, 2), f"the assessor text for {tag} moved: {old[:80]!r} ({n} copies)"
        if n == 2:
            cut = src.index("def y_rt_pair(")
            assert src.index(old) < cut < src.rindex(old), f"the two copies of the text for {tag} are not rt1_pair's then y_rt_pair's"
        return load(src.replace(old, new, 1), tag)
    removed = []
    m_gate = mutated('g["status"] = "PASS" if (num == 0.0 and num_c == 0.0 and num_h == 0.0) else "FAIL"', 'g["status"] = "PASS"', "mgate")
    assert verdict(m_gate, mutate=zero_carrier)["gates"]["RT-1a"]["status"] == "PASS"
    removed.append("RT-1a exact-zero test -> m = 1 PASSES")
    m_gate2 = mutated('g["status"] = "PASS" if (num == 0.0 and num_c == 0.0 and num_h == 0.0) else "FAIL"', 'g["status"] = "PASS" if num / den < 1e-4 else "FAIL"', "mgate2")
    assert verdict(m_gate2, mutate=one_ulp)["gates"]["RT-1a"]["status"] == "PASS" and verdict(ac, mutate=one_ulp)["gates"]["RT-1a"]["status"] == "FAIL"
    removed.append("RT-1a back to B76's m < 1e-4 -> the one-ulp difference PASSES")
    m_bc = mutated("""    return rt1_broadcast(c, n, f"{d.name}/thetaC"), rt1_broadcast(h, n, f"{d.name}/thetaH")""",
                   """    if len(c[2]) != len(h[2]):
        raise ValueError(f"{d.name}: thetaC and thetaH differ in size ({len(c[2])} against {len(h[2])})")
    return c[2], h[2]""", "bcast")
    r_old = verdict(m_bc, theta_h_uniform=True)
    assert r_old["gates"]["RT-1a"]["status"] == "VOID" and "differ in size" in r_old["gates"]["RT-1a"]["reason"], r_old["gates"]["RT-1a"]
    removed.append("RT-1a broadcast -> thetaH uniform (1 number) against thetaC (30) is VOID again, B76's parse defect")
    m_den = mutated('if not (math.isfinite(den) and math.isfinite(num)) or not den >= RT1_DENOMINATOR_MIN:', 'if False:', "den")
    assert verdict(m_den, void_den=True)["gates"]["RT-1a"]["status"] == "PASS"
    removed.append("RT-1a denominator floor -> the 1e-12 denominator PASSES")
    m_dt = mutated('g["status"] = "PASS" if g["worst_rel"] == 0.0 else "FAIL"', 'g["status"] = "PASS"', "dtgate")
    assert verdict(m_dt, mutate=dt_off(1e-10))["gates"]["RT-1b"]["status"] == "PASS"
    removed.append("RT-1b exact-equality test -> dt off by 1e-10 PASSES")
    m_dt2 = mutated('g["status"] = "PASS" if g["worst_rel"] == 0.0 else "FAIL"', 'g["status"] = "PASS" if g["worst_rel"] <= 1e-12 else "FAIL"', "dtgate2")
    assert verdict(m_dt2, mutate=dt_off(1e-13))["gates"]["RT-1b"]["status"] == "PASS"
    removed.append("RT-1b back to B76's 1e-12 -> dt off by 1e-13 PASSES")
    m_dig = mutated('if g["digits_printed"] < RT1_DT_MIN_DIGITS and g["status"] == "PASS":', 'if False:', "digits")
    assert verdict(m_dig, dt_text="4.76190476190e-06")["gates"]["RT-1b"]["status"] == "PASS"
    removed.append("RT-1b resolution rule -> the 12-digit log PASSES")
    m_drop = mutated('''                problems.append(f"{rel_name}: dropped (A has it at step {s_end}, B has not)")
                continue''', '                continue', "drop")
    assert verdict(m_drop, mutate=lambda a, b: (b / "0.0047" / "thetaPendingP").unlink())["gates"]["RT-1c"]["status"] == "PASS"
    removed.append("RT-1c dropped-field test -> the dropped thetaPendingP PASSES")
    m_req = mutated('''                if not (dd / rq).is_file():
                    problems.append''', '''                if False:
                    problems.append''', "req")
    assert verdict(m_req, entry=entry5a, fields=("thetaC", "thetaH", "w", "q", "Vs", "thetaPendingP"))["gates"]["RT-1c"]["status"] == "PASS"
    removed.append("RT-1c required-field test -> a field absent from both PASSES")
    m_rel = mutated("if diff != 0.0:", "if False:", "rel")
    assert verdict(m_rel, mutate=field_off("w", 1e-7))["gates"]["RT-1c"]["status"] == "PASS"
    removed.append("RT-1c exact-equality test -> w off by 1e-7 PASSES")
    m_rel2 = mutated("if diff != 0.0:", "if rel_err > 1e-8:", "rel2")
    assert verdict(m_rel2, mutate=field_off("thetaH", 1e-9))["gates"]["RT-1c"]["status"] == "PASS" and verdict(m_rel2, mutate=w_one_ulp)["gates"]["RT-1c"]["status"] == "PASS"
    removed.append("RT-1c back to B76's 1e-8 relative -> thetaH off by 1e-9 and w one ulp off PASS")
    m_wp = mutated("if v is None or v < RT1_WRITE_PRECISION_MIN:", "if False:", "wp")
    assert verdict(m_wp, write_precision=6)["status"] == "PASS"
    removed.append("RT-1 writePrecision rule -> files written at 6 digits PASS a bitwise claim")
    m_vs = mutated('if entry.get("needs_vs_nonzero") and vs is not None and not vs > 0.0:', 'if False:', "vs")
    assert verdict(m_vs, entry=entry5a, vs_zero=True)["status"] == "PASS"
    removed.append("Vs != 0 precondition -> Vs = 0 PASSES")
    for s in shown:
        print("   rt1 gate:", s)
    for s in removed:
        print("   rt1 guard removed:", s)
    print(f"   rt1gates: {len(shown)} synthetic restarts judged as expected; {len(removed)} guards deleted one at a time, each lets its negative pass")


def _s25_rt1_watch_section() -> None:
    """scripts/step_stop_watcher.py: touches stopAfterStep at PHASE_CLOT_DT step S-1 and not before; reads complete lines only; each guard deleted."""
    import importlib.util
    wp = CASE / "scripts" / "step_stop_watcher.py"
    wsrc = wp.read_text()

    def load(text: str, tag: str, dirp: Path):
        p = dirp / f"watch_{tag}.py"
        p.write_text(text)
        spec = importlib.util.spec_from_file_location(f"step_stop_watcher_{tag}", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    with tempfile.TemporaryDirectory(prefix="am2020-rt1-watch-") as t:
        tmp = Path(t)
        w = load(wsrc, "real", tmp)
        rec = lambda k: f"PHASE_CLOT_DT step={k} time=1e-05 deltaT=4.7619047619047618e-06 dtFluidCo=1\n"
        # ---- the line reader: complete lines only, the step number of PHASE_CLOT_DT records only
        tl = w.Tail()
        assert tl.feed("Time = 1e-05s\n" + rec(1) + rec(2)[:30]) == [1]
        assert tl.feed(rec(2)[30:] + "PHASE_CLOT_DTX step=9 \n" + "deltaT = 4e-06\n" + rec(3)) == [2, 3]
        assert tl.feed("PHASE_CLOT_DT step=46") == [] and tl.feed(" time=1e-05 deltaT=1\n") == [46]
        # ---- end to end: a case dir, a growing log, the watcher in a subprocess
        import subprocess, time as _t

        def run_case(name: str, lines_then_wait, stop: int = 47, script: Path = wp, child_json: str | None = None):
            case = tmp / name
            case.mkdir()
            (case / "child.json").write_text(child_json if child_json is not None else json.dumps({"child_id": "x", "stop_after_step": stop}))
            log = case / w.LOG_NAME
            proc = subprocess.Popen([sys.executable, "-B", str(script), str(case), "--poll-s", "0.02", "--max-wait-s", "20"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                with log.open("w") as fh:
                    for chunk, wait in lines_then_wait:
                        fh.write(chunk)
                        fh.flush()
                        _t.sleep(wait)
                    touched_mid = (case / "stopAfterStep").exists()
                    _t.sleep(0.4)
                return case, proc, touched_mid
            finally:
                pass
        # steps 1..45 logged: no touch; the record of step 46 (= S-1) touches; the touch names step 46
        case, proc, mid = run_case("ontime", [("".join(rec(k) for k in range(1, 46)), 0.4)], 47)
        assert not (case / "stopAfterStep").exists() and proc.poll() is None, "touched before step S-1"
        with (case / w.LOG_NAME).open("a") as fh:
            fh.write(rec(46))
        proc.wait(timeout=10)
        assert proc.returncode == 0 and (case / "stopAfterStep").exists()
        assert "step=46" in (case / "step-stop.txt").read_text() and "LATE" not in (case / "step-stop.txt").read_text()
        # a partial record of step 46 does not touch; its completion does
        case2, proc2, mid2 = run_case("partial", [("".join(rec(k) for k in range(1, 46)) + "PHASE_CLOT_DT step=46", 0.4)], 47)
        assert not (case2 / "stopAfterStep").exists(), "touched on a partial line"
        with (case2 / w.LOG_NAME).open("a") as fh:
            fh.write(" time=1e-05 deltaT=1\n")
        proc2.wait(timeout=10)
        assert (case2 / "stopAfterStep").exists()
        # started late: the first record seen is already past the trigger -> touched, noted LATE
        case3, proc3, _ = run_case("late", [(rec(49), 0.3)], 47)
        proc3.wait(timeout=10)
        assert (case3 / "stopAfterStep").exists() and "LATE" in (case3 / "step-stop.txt").read_text()
        # a restart: B's log starts at step 43 and stops at 46 for S = 47 (absolute index)
        case4, proc4, _ = run_case("restart", [(rec(43) + rec(44) + rec(45), 0.3)], 47)
        assert not (case4 / "stopAfterStep").exists()
        with (case4 / w.LOG_NAME).open("a") as fh:
            fh.write(rec(46))
        proc4.wait(timeout=10)
        assert (case4 / "stopAfterStep").exists()
        # malformed child.json: exit non-zero, no touch
        for label, cj in (("no stop_after_step", json.dumps({"child_id": "x"})), ("a boolean", json.dumps({"stop_after_step": True})),
                          ("1", json.dumps({"stop_after_step": 1})), ("a string", json.dumps({"stop_after_step": "47"})), ("not json", "{")):
            c = tmp / f"bad_{label.replace(' ', '_')}"
            c.mkdir()
            (c / "child.json").write_text(cj)
            done = subprocess.run([sys.executable, "-B", str(wp), str(c), "--max-wait-s", "1"], capture_output=True, text=True)
            assert done.returncode != 0 and not (c / "stopAfterStep").exists(), (label, done.returncode, done.stderr)
        print("   rt1watch: touches at step S-1 and not before; complete lines only; a late start is noted; a restart's absolute steps; 5 malformed child.json refused")
        # ---- each guard deleted from a scratch copy: the matching negative is then wrong
        def mutated(old, new, tag):
            assert wsrc.count(old) == 1, (tag, old)
            p = tmp / f"mut_{tag}.py"
            p.write_text(wsrc.replace(old, new))
            return p
        removed = []
        # (1) trigger one step early (>= stop - 2): touches at step 45 -> A would stop at 46
        p1 = mutated("if step >= stop - 1:", "if step >= stop - 2:", "early")
        c5, pr5, _ = run_case("m_early", [("".join(rec(k) for k in range(1, 46)), 0.5)], 47, script=p1)
        assert (c5 / "stopAfterStep").exists(), "the early-trigger mutation did not touch at step 45"
        removed.append("trigger at >= S-2 touches at step 45 (A would stop a step short)")
        # (2) partial lines accepted (no complete-line split): a partial 'step=46' record is read
        p2 = mutated('''        *lines, self.partial = self.partial.split("\\n")''', '''        lines, self.partial = self.partial.split("\\n"), ""''', "partial")
        c6, pr6, _ = run_case("m_partial", [("".join(rec(k) for k in range(1, 46)) + "PHASE_CLOT_DT step=46 ", 0.5)], 47, script=p2)
        assert (c6 / "stopAfterStep").exists(), "the partial-line mutation did not touch on a partial record"
        removed.append("accepting the unterminated last line touches on a partial record")
        # (3) the stop_after_step type check
        p3 = mutated("if isinstance(s, bool) or not isinstance(s, int) or s < 2:", "if False:", "type")
        c7 = tmp / "m_type"
        c7.mkdir()
        (c7 / "child.json").write_text(json.dumps({"stop_after_step": True}))
        (c7 / w.LOG_NAME).write_text(rec(1))
        done = subprocess.run([sys.executable, "-B", str(p3), str(c7), "--max-wait-s", "2", "--poll-s", "0.02"], capture_output=True, text=True)
        assert (c7 / "stopAfterStep").exists(), "the type-check mutation did not accept stop_after_step true"
        removed.append("no type check: stop_after_step true is accepted and touches at once")
        for s in removed:
            print("   rt1watch guard removed:", s)
        for pr in (proc, proc2, proc3, proc4, pr5, pr6):
            if pr.poll() is None:
                pr.kill()


def _s25_rt1_allrun_section() -> None:
    """Allrun's two RT-1 hooks, extracted from the file and run in bash against fixtures: B's sibling restart (and its refusals) and the watcher launch."""
    import subprocess
    text = (CASE / "Allrun").read_text()
    a0 = text.index('    if [ "$setup_code" -eq 0 ] && [ -f "$case/child.json" ] && grep -q \'"restart_from_child"\' "$case/child.json"; then')
    a1 = text.index("\n    fi\n", text.index("setup_code=$?\n        fi", a0)) + len("\n    fi\n")
    restart_snip = text[a0:a1]
    b0 = text.index('    step_watcher_pid=""')
    b1 = text.index('        step_watcher_pid=$!\n    fi\n', b0) + len('        step_watcher_pid=$!\n    fi\n')
    watch_snip = text[b0:b1]
    assert subprocess.run(["bash", "-n", str(CASE / "Allrun")]).returncode == 0
    drv = CASE / "scripts" / "restart_driver.py"
    wat = CASE / "scripts" / "step_stop_watcher.py"
    with tempfile.TemporaryDirectory(prefix="am2020-rt1-allrun-") as t:
        def fixture(name: str, *, finished: str | None = "0", with_key: bool = True):
            gate = Path(t) / name
            (gate / "scripts").mkdir(parents=True)
            shutil.copy(drv, gate / "scripts" / "restart_driver.py")
            shutil.copy(wat, gate / "scripts" / "step_stop_watcher.py")
            a = gate / "runs" / "A"
            b = gate / "runs" / "B"
            for cid in ("A", "B"):
                c = gate / "runs" / cid
                (c / "system").mkdir(parents=True)
                (c / "constant").mkdir()
                (c / "system" / "controlDict").write_text("FoamFile\n{\n    format ascii;\n}\nstartFrom startTime;\nstartTime 0;\nendTime 0.5;\nwritePrecision 17;\n")
                (c / "constant" / "phaseClotProperties").write_text("timeScale 1.0;\n")
                _rt1_write_field(c / "0" / "w", "scalar", [0.0])
            for k in range(38, 48):
                d = a / f"0.{k:04d}"
                for f in ("w", "U", "p", "phiSkel"):
                    _rt1_write_field(d / f, "scalar", [float(k)])
                (d / "uniform").mkdir()
                (d / "uniform" / "time").write_text(f"FoamFile\n{{\n    format ascii;\n}}\nvalue 0.{k:04d};\nname \"0.{k:04d}\";\nindex {k};\ndeltaT 1e-06;\n\n"
                                                    "// ************************************************************************* //\n")
                for f in sorted(d.rglob("*")):
                    if f.is_file() and "// ****" not in f.read_text():
                        f.write_text(f.read_text() + "\n// ************************************************************************* //\n")
            if finished is not None:
                (a / "run-exit-code.txt").write_text(finished + "\n")
            cj = {"child_id": "B", "stop_after_step": 47}
            if with_key:
                cj.update(restart_from_child="A", restart_index=42)
            (b / "child.json").write_text(json.dumps(cj))
            return gate

        def run(snippet: str, gate: Path, env_extra: str = "") -> tuple:
            script = f'set -u\ngate="{gate}"\ncase="$gate/runs/B"\nchild_id=B\nsetup_code=0\n{env_extra}\n{snippet}\necho "setup_code=$setup_code"\n'
            done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            code = int(re.search(r"setup_code=(\d+)", done.stdout).group(1))
            return code, (gate / "runs" / "B")
        g = fixture("ok")
        code, b = run(restart_snip, g)
        ctl = (b / "system" / "controlDict").read_text()
        assert code == 0 and "startTime 0.0042;" in ctl and (b / "0.0042" / "w").is_file() and "chosen_index" in (b / "restart.json").read_text(), (code, ctl)
        assert not (b / "0.0047").exists()
        g = fixture("unfinished", finished=None)
        code, b = run(restart_snip, g)
        assert code == 4 and "startTime 0;" in (b / "system" / "controlDict").read_text() and "has not finished normally" in (b / "log.restartDriver").read_text()
        g = fixture("noindex")
        shutil.rmtree(g / "runs" / "A" / "0.0042")
        code, b = run(restart_snip, g)
        assert code == 2, code
        g = fixture("twosources")
        code, b = run(restart_snip, g, 'export AM2020_RESTART_FROM=/somewhere')
        assert code == 4 and "two restart sources" in (b / "log.restartDriver").read_text()
        g = fixture("nokey", with_key=False)
        code, b = run(restart_snip, g)
        assert code == 0 and not (b / "log.restartDriver").exists() and "startTime 0;" in (b / "system" / "controlDict").read_text()
        # the exit status must survive: delete the `|| setup_code=$?` and the refused restart is read as success
        lost = restart_snip.replace("--apply > log.restartDriver 2>&1 ) || setup_code=$?", "--apply > log.restartDriver 2>&1 )")
        assert lost != restart_snip
        g = fixture("lost", finished=None)
        code, b = run(lost, g)
        assert code == 0, "without `|| setup_code=$?` the failure should be lost"
        # the watcher launch
        g = fixture("watch")
        b = g / "runs" / "B"
        (b / "system" / "controlDict").write_text("functions\n{\n    stopAfterStep\n    {\n        type stopAtFile;\n    }\n}\n")
        (b / "log.phaseClotAxisymmetricFluid").write_text("PHASE_CLOT_DT step=46 time=1 deltaT=1 dtFluidCo=1\n")
        script = f'set -u\ngate="{g}"\ncase="$gate/runs/B"\n{watch_snip}\nsleep 1.5\nkill "$step_watcher_pid" 2>/dev/null\nls "$case"\n'
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert "stopAfterStep" in done.stdout.split() and "log.stepWatcher" in done.stdout.split(), done.stdout + done.stderr
        g = fixture("nowatch")
        b = g / "runs" / "B"
        (b / "system" / "controlDict").write_text("functions\n{\n    stopWatcher\n    {\n        type stopAtFile;\n    }\n}\n")
        script = f'set -u\ngate="{g}"\ncase="$gate/runs/B"\n{watch_snip}\nprintf "pid=[%s]\\n" "$step_watcher_pid"\nls "$case"\n'
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert "pid=[]" in done.stdout and "log.stepWatcher" not in done.stdout.split(), done.stdout
        assert 'grep -q "stopWatcher"' in text and "stopWatcher" not in watch_snip, "the step watcher block must not name the step-1 watcher's file"
    print("   rt1allrun: B's sibling restart (ok; sibling unfinished -> 4; no step-42 write -> 2; two restart sources -> 4; no key -> no-op; "
          "`|| setup_code=$?` deleted -> the failure is lost); the step watcher launches for stopAfterStep only and not for stopWatcher")


def _s25_rt1_gen_section() -> None:
    """The generator side of RT-1: the four children generate; A and B are the same case but for child.json; the controlDict writes every step,
    purgeWrite and the stopAfterStep function object; every refusal fires for its own reason and (where a guard line exists) is ACCEPTED with the
    guard removed; the three controlDict writer defects are refused by the read-back and accepted without it."""
    bench, gen = _gen_bench_section("rt1gen")
    try:
        kid = lambda c, cid: next(x for x in c["children"] if x["child_id"] == cid)

        def edit(cid, **ch):
            def m(c):
                k = kid(c, cid)
                for key, v in ch.items():
                    if v is None:
                        k.pop(key, None)
                    else:
                        k[key] = v
            return m
        PA, PB, QA, QB = "RT_P73_A", "RT_P73_B", "RT_5a_A", "RT_5a_B"
        # ---- positives: all four generate; A and B differ in child.json alone
        dirs = {}
        stash = Path(tempfile.mkdtemp(prefix="am2020-rt1gen-stash-"))
        for cid in (PA, PB, QA, QB):
            rc, last, d = gen(None, cid)
            assert rc == 0, (cid, last)
            shutil.copytree(d, stash / cid)          # gen() clears runs/ on the next call
            d = dirs[cid] = stash / cid
            ctl = (d / "system" / "controlDict").read_text()
            top = ctl.split("functions", 1)[0]
            assert re.search(r"(?m)^writeControl timeStep;$", top) and re.search(r"(?m)^writeInterval 1;$", top) and re.search(r"(?m)^purgeWrite 10;$", top), top
            assert re.search(r"(?m)^writePrecision 17;$", top) and "startFrom startTime;" in top and "adjustableRunTime" not in ctl
            assert "stopAfterStep" in ctl and ctl.count("stopAtFile") == (2 if "stopWatcher" in ctl else 1), ctl
            spec = json.loads((d / "child.json").read_text())
            assert spec["write_every_step"] is True and spec["stop_after_step"] == 47, spec
        import filecmp

        def walk(c, rel=""):
            out = [rel + n for n in c.diff_files + c.left_only + c.right_only]
            for sub, sd in c.subdirs.items():
                out += walk(sd, rel + sub + "/")
            return out
        for x, y in ((PA, PB), (QA, QB)):
            assert walk(filecmp.dircmp(dirs[x], dirs[y])) == ["child.json"], walk(filecmp.dircmp(dirs[x], dirs[y]))
        sb = json.loads((dirs[PB] / "child.json").read_text())
        assert sb["restart_from_child"] == PA and sb["restart_index"] == 42 and "restart_from_child" not in json.loads((dirs[PA] / "child.json").read_text())
        props = lambda c: (dirs[c] / "constant" / "phaseClotProperties").read_text()
        for c in (PA, QA):                         # fibreRelease is built in: the pending-content solvers are in EVERY child's fvSolution
            assert "thetaPendingB" in (dirs[c] / "system" / "fvSolution").read_text() and "networkRupture" not in props(c), c
        # ---- negatives
        pair_p = '    ("RT_P73_B", "RT_P73_A", set(), {"restart_from_child", "restart_index"}),\n'
        pair_pa = ('    ("RT_P73_A", "P73_G4_r3_40", set(), {"end_time_s", "wall_budget_h", "frame_interval_s", "frame_fields",\n'
                   '                                         "write_every_step", "stop_after_step"}),\n')
        frames = {"frame_interval_s": 1e-4, "frame_fields": ["q", "p", "U", "Vs", "clotDrag", "poroPerm", "Gfield", "Us"]}

        def top(**kw):
            def m(c):
                for k, v in kw.items():
                    c[k] = v
            return m

        def drop_pair(c):
            c["s25_rt1_pairs"] = [p for p in c["s25_rt1_pairs"] if p["pair_id"] != "RT_5a"]

        def pairs_edit(**kw):
            def m(c):
                c["s25_rt1_pairs"][0].update(kw)
            return m
        no_pairs = lambda c: c.pop("s25_rt1_pairs")
        negatives = (
            # (label, mutate, child, want, patches that remove ONLY this guard)
            ("write_every_step false", edit(PA, write_every_step=False), PA, "write_every_step must be true", ()),
            ("stop_after_step without write_every_step", edit(PA, write_every_step=None), PA, "go together", ()),
            ("write_every_step without stop_after_step", edit(PA, stop_after_step=None), PA, "go together", ()),
            ("stop_after_step 5 (no room for N0 >= 1 and 5 steps)", edit(PA, stop_after_step=5), PA, "must be an integer of at least 6", ()),
            ("stop_after_step 47.0 (a float)", edit(PA, stop_after_step=47.0), PA, "must be an integer of at least 6", ()),
            ("stop_after_step true", edit(PA, stop_after_step=True), PA, "must be an integer of at least 6", ()),
            ("wall_budget_h 2.0 (the backstop may not be hours)", edit(PA, wall_budget_h=2.0), PA, "needs wall_budget_h in (0, 0.5]", ()),
            ("no end_time_s of its own", edit(PA, end_time_s=None), PA, "names its own end_time_s", ()),
            ("frames on an RT child", edit(QA, **frames), QA, "frames are written at adjustableRunTime", ()),
            ("restart_index without restart_from_child", edit(PB, restart_from_child=None), PB, "go together", ()),
            ("restart_from_child naming a child that is not there", edit(PB, restart_from_child="RT_nope"), PB, "must be another child_id", ()),
            ("restart_from_child naming itself", edit(PB, restart_from_child=PB), PB, "must be another child_id", ()),
            ("restart_index 41 (stop 47 - 41 = 6 steps)", edit(PB, restart_index=41), PB, "- restart_index == 5", ()),
            ("restart_index 0", edit(PB, restart_index=0), PB, "must be an integer >= 1", ()),
            ("B stopping at another step than A", edit(PB, stop_after_step=48, restart_index=43), PB, "its source RT_P73_A has stop_after_step 47", ()),
            ("A that is itself a restart", edit(PA, restart_from_child=PB, restart_index=42), PB, "is itself a restart", ()),
            ("B whose source is in another series", edit(PB, restart_from_child=QA), PB, "is in series", ()),
            ("B differing from A in a second key (k0p_m2)", edit(PB, k0p_m2=5e-13), PB, "FSF parent check: RT_P73_B differs from RT_P73_A in k0p_m2", ((pair_p, ""),)),
            ("A differing from its parent in a second key (G_pa)", edit(PA, G_pa=5.0), PA, "FSF parent check: RT_P73_A differs from P73_G4_r3_40 in G_pa", ((pair_pa, ""),)),
            ("no s25_rt1_pairs at all", no_pairs, PA, "hold no s25_rt1_pairs", ()),
            ("an RT child left out of the pairs", drop_pair, QA, "every RT-1 child is in exactly one", ()),
            ("a pair n0 that is not B's restart_index", pairs_edit(n0=43), PA, "must equal RT_P73_B's restart_index", ()),
            ("an unknown required field", pairs_edit(required_fields=["thetaX"]), PA, "must be a list of distinct names", ()),
            ("needs_vs_nonzero as a string", pairs_edit(needs_vs_nonzero="yes"), PA, "needs_vs_nonzero must be true or false", ()),
            ("an extra key in a pair", pairs_edit(extra=1), PA, "must hold exactly", ()),
        )
        for label, mutate, child, want, patches in negatives:
            rc, last, _ = gen(mutate, child)
            assert rc != 0, f"the generator ACCEPTED {label}"
            assert want in last, f"{label}: refused for the wrong reason: {last}"
            how = "no removable guard line (an inline raise)"
            if patches:
                rc2, last2, _ = gen(mutate, child, patches)
                assert rc2 == 0 or want not in last2, f"{label}: the refusal survives the removal of its guard: {last2}"
                how = "ACCEPTED" if rc2 == 0 else f"fails otherwise ({last2[:80]})"
            print(f"generator refuses: {label:<58} -> {want[:50]!r}; guard removed: {how}")
        # ---- the controlDict writer defects: refused by rt1_controldict_apply's read-back; accepted with the read-back cut off
        rb = '    purge = re.findall(r"(?m)^\\s*purgeWrite\\s+([^;\\s]+)\\s*;", top)\n'
        fo_old = '            if "stop_after_step" in child:\n                functions_body += (\n                    "    stopAfterStep'
        fo_new = '            if False:\n                functions_body += (\n                    "    stopAfterStep'
        defects = (
            ("purgeWrite left at 0", (('    replace_entry(control, "purgeWrite", S25_RT1_PURGE_WRITE)\n', ""),), "purgeWrite ['0']", r"(?m)^purgeWrite 0;$"),
            ("writeInterval left at the series' value", (('    replace_entry(control, "writeInterval", 1)\n', ""),), "not a full write at every step", r"(?m)^writeInterval 0\.\d+;$"),
            ("the stopAfterStep function object not written", ((fo_old, fo_new),), "exactly one stopAfterStep function object", r"\A(?:(?!stopAfterStep).)*\Z"),
        )
        for label, patches, want, shown in defects:
            rc, last, _ = gen(None, PA, patches)
            assert rc != 0 and want in last, f"{label}: {rc} {last}"
            rc2, last2, d2 = gen(None, PA, patches + ((rb, "    return\n" + rb),))
            assert rc2 == 0, (label, last2)
            ctl = (d2 / "system" / "controlDict").read_text()
            assert re.search(shown, ctl, re.S), f"{label}: the defective file does not show it"
            print(f"read-back refuses: {label:<54} -> {want!r}; read-back removed: ACCEPTED (the file holds the defect)")
        shutil.rmtree(stash)
        print(f"   rt1gen: 4 children generate, A and B differ in child.json alone, controlDict writes every step (purgeWrite 10, stopAfterStep); "
              f"{len(negatives)} generator refusals each for its own reason; 3 writer defects seen by the read-back and accepted without it")
    finally:
        bench.close()


# ---- case-engineer 05 (S25.1 C(b)): the odd-even stability gate on the clot phase ------------------------------------------------------
_STAB_NX, _STAB_NY = 40, 12


def _stab_child(root: Path, name: str, vs_fn, *, steps=tuple(range(70, 2100, 70)), clot=lambda i, j: 10 <= i < 30 and 3 <= j < 9,
                blocks: str = "hex (0 1 2 3 4 5 6 7) (40 12 1) simpleGrading (1 1 1)", drop_q: bool = False) -> Path:
    """A child directory with frames at `steps`: Vs = (vs_fn(step, i, j), 0, 0), q = 0.01 in the clot cells and 1 elsewhere."""
    case = root / name
    (case / "system").mkdir(parents=True)
    (case / "system" / "blockMeshDict").write_text(f"FoamFile\n{{\n    format ascii;\n}}\nblocks\n(\n    {blocks}\n);\n")
    for k in steps:
        d = case / f"{k * 1e-6:.6g}"
        _rt1_write_field(d / "Vs", "vector", [[vs_fn(k, i, j), 0.0, 0.0] for j in range(_STAB_NY) for i in range(_STAB_NX)])
        if not drop_q:
            _rt1_write_field(d / "q", "scalar", [0.01 if clot(i, j) else 1.0 for j in range(_STAB_NY) for i in range(_STAB_NX)])
        (d / "uniform").mkdir(exist_ok=True)
        (d / "uniform" / "time").write_text(f"FoamFile\n{{\n    format ascii;\n}}\nindex {k};\n")
    return case


def _s25_stability_section() -> None:
    """The S25.1 C(b) gate: the detector scores 0 on a tanh; the gate is on the clot phase only; it FAILS on a growing clot mode, at
    the registered bound; each guard deleted from a scratch assessor copy lets its negative pass."""
    import importlib.util, math as _m
    path = Path(__file__).resolve().parent / "assess_case.py"
    src = path.read_text()

    def load(text: str, tag: str):
        with tempfile.TemporaryDirectory(prefix="am2020-stab-mod-") as t:
            p = Path(t) / f"assess_case_{tag}.py"
            p.write_text(text)
            spec = importlib.util.spec_from_file_location(f"assess_case_stab_{tag}", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    ac = load(src, "real")
    nx, ny = _STAB_NX, _STAB_NY
    shown, removed = [], []
    # ---- the detector alone: zero on a tanh of any width, nonzero on a tanh plus an odd-even mode
    for w in (0.3, 0.5, 1.0, 3.0, 10.0):
        f = [_m.tanh((i - 20) / w) for _ in range(ny) for i in range(nx)]
        f2 = [v + 1e-6 * (-1.0) ** i for v, i in zip(f, [i for _ in range(ny) for i in range(nx)])]
        a0, a1 = ac.s25_zigzag_amplitude(f, nx, ny)[0], ac.s25_zigzag_amplitude(f2, nx, ny)[0]
        assert a0 == 0.0, (w, a0)
        # the mode is seen where the profile is flatter than the mode (3 alternating differences are needed): in the tails of a narrow tanh
        assert (a1 > 0.0) == (w <= 1.0), (w, a1)
    f = [_m.tanh((i - 20) / 1.0) + 1e-6 * (-1.0) ** i for _ in range(ny) for i in range(nx)]
    assert 1e-6 < ac.s25_zigzag_amplitude(f, nx, ny)[0] < 3e-6
    shown.append("detector: 0.0 on a tanh of width 0.3, 0.5, 1, 3, 10; a width-1 tanh plus a 1e-6 odd-even mode scores 1e-6 .. 3e-6 (seen in the flat tails)")

    def mode(step, i, j, e0=1e-3, g=0.0):         # odd-even amplitude e0 (1+g)^step
        return e0 * (1.0 + g) ** step * (-1.0) ** i

    def smooth(step, i, j, w=3.0):
        return 0.1 * _m.tanh((i - 20) / w)

    def judge(mod, fn, **kw):
        with tempfile.TemporaryDirectory(prefix="am2020-stab-") as t:
            return mod.s25_stability_report(_stab_child(Path(t), "c", fn, **kw))

    def case(label, want, fn, mod=None, **kw):
        r = judge(mod or ac, fn, **kw)
        assert r["verdict"] == want, f"{label}: {r['verdict']} (wanted {want}): {json.dumps(r)[:700]}"
        shown.append(f"{label}: {want}")
        return r
    inclot = lambda i, j: 10 <= i < 30 and 3 <= j < 9
    # ---- PASS
    r = case("a smooth tanh in the clot, nothing else", "PASS", smooth)
    assert max(r["amp_clot"]) == 0.0 and max(r["amp_whole_field"]) == 0.0 and r["frames"] == 29 and r["grid"] == [40, 12]
    r = case("a tanh that steepens from width 10 to 0.3 over the run (a front sharpening)", "PASS",
             lambda k, i, j: 0.1 * _m.tanh((i - 20) / (10.0 - 9.7 * k / 2030.0)))
    assert max(r["amp_clot"]) == 0.0, r["amp_clot"]
    r = case("a clot odd-even mode that is constant (1e-3, seen by the detector)", "PASS", lambda k, i, j: smooth(k, i, j) + (mode(k, i, j) if inclot(i, j) else 0.0))
    assert min(r["amp_clot"]) > 1e-4 and r["clot_worst_window"]["ratio"] == 1.0 or abs(r["clot_worst_window"]["ratio"] - 1.0) < 1e-9, r["clot_worst_window"]
    r = case("a clot mode growing at 0.5e-5 per step (inside the bound)", "PASS",
             lambda k, i, j: smooth(k, i, j) + (mode(k, i, j, g=0.5e-5) if inclot(i, j) else 0.0))
    assert min(r["amp_clot"]) > 1e-4 and 0.5 < r["clot_worst_window"]["over_bound"] < 1.0, r["clot_worst_window"]
    r = case("a clot mode of 1e-8 of max|Vs| that appears at two frames and goes (below the floor, as in the returned 5a children)", "PASS",
             lambda k, i, j: 0.1 + (1e-9 * (-1.0) ** i if inclot(i, j) and k in (700, 1400) else 0.0))
    assert 0.0 < max(r["amp_clot"]) < r["floor"], (r["amp_clot"], r["floor"])
    rb = case("odd-even growth x1.001 per step in the BLOOD phase only: clot gate PASSES, the whole field is REPORTED over the bound", "PASS",
              lambda k, i, j: smooth(k, i, j) + (0.0 if inclot(i, j) else mode(k, i, j, e0=1e-5, g=1e-3)))
    assert rb["clot_worst_window"]["over_bound"] < 1.0 and rb["whole_field_worst_window_reported_not_gated"]["over_bound"] > 1.0, rb
    assert all(a == 0.0 for a in rb["amp_clot"]), rb["amp_clot"]
    # ---- FAIL
    r = case("a clot mode growing at 1.5e-5 per step (outside the bound)", "FAIL",
             lambda k, i, j: smooth(k, i, j) + (mode(k, i, j, g=1.5e-5) if inclot(i, j) else 0.0))
    assert 1.0 < r["clot_worst_window"]["over_bound"] < 1.1, r["clot_worst_window"]
    r = case("a clot mode growing at 1e-3 per step", "FAIL", lambda k, i, j: smooth(k, i, j) + (mode(k, i, j, g=1e-3) if inclot(i, j) else 0.0))
    case("a clot mode absent for 1400 steps, then 1e-3 of max|Vs| (growth from zero)", "FAIL",
         lambda k, i, j: smooth(k, i, j) + (1e-3 * (-1.0) ** i if inclot(i, j) and k >= 1500 else 0.0))
    # ---- VOID, never PASS
    case("no frame holds a clot cell (q = 1 everywhere)", "VOID", smooth, clot=lambda i, j: False)
    r = case("frames span only 600 steps (no 1000-step window)", "VOID", smooth, steps=tuple(range(60, 660, 60)))
    assert "1000 steps apart" in r["reason"], r
    case("q missing from the frames", "VOID", smooth, drop_q=True)
    case("two blocks in the blockMeshDict", "VOID", smooth, blocks="hex (0 1 2 3 4 5 6 7) (40 12 1) simpleGrading (1 1 1)\n    hex (0 1 2 3 4 5 6 7) (4 4 1) simpleGrading (1 1 1)")
    r = case("Vs is 0 in every frame (the skeleton never moved)", "VOID", lambda k, i, j: 0.0)
    assert "never moved" in r["reason"], r
    # ---- each guard deleted from a scratch copy lets its negative pass
    def mutated(old: str, new: str, tag: str):
        assert src.count(old) == 1, f"the assessor text for {tag} moved: {old[:80]!r}"
        return load(src.replace(old, new), tag)
    m_mask = mutated("ac = max(s25_zigzag_amplitude(c, nx, ny, mask)[0] for c in comps)", "ac = max(s25_zigzag_amplitude(c, nx, ny)[0] for c in comps)", "mask")
    blood = lambda k, i, j: smooth(k, i, j) + (0.0 if inclot(i, j) else mode(k, i, j, e0=1e-5, g=1e-3))
    assert judge(m_mask, blood)["verdict"] == "FAIL" and judge(ac, blood)["verdict"] == "PASS"
    removed.append("clot mask -> the blood-phase growth FAILS the gate (the whole-field form B76 read)")
    m_alt = mutated("if d0 * d1 < 0.0 and d1 * d2 < 0.0:", "if True:", "alt")
    steep = lambda k, i, j: 0.1 * _m.tanh((i - 20) / (10.0 - 9.7 * k / 2030.0))
    assert judge(m_alt, steep)["verdict"] == "FAIL" and judge(ac, steep)["verdict"] == "PASS"
    removed.append("alternation test -> a steepening tanh scores > 0 and FAILS")
    m_gate = mutated('res["verdict"] = "FAIL" if wc["over_bound"] > 1.0 else "PASS"', 'res["verdict"] = "PASS"', "bound")
    grow = lambda k, i, j: smooth(k, i, j) + (mode(k, i, j, g=1e-3) if inclot(i, j) else 0.0)
    assert judge(m_gate, grow)["verdict"] == "PASS"
    removed.append("growth bound -> a clot mode growing 1e-3 per step PASSES")
    m_floor = mutated("floor = S25_STAB_FLOOR_REL * vmax", "floor = 1e-300 * vmax", "floor")
    tiny = lambda k, i, j: 0.1 + (1e-9 * (-1.0) ** i if inclot(i, j) and k in (700, 1400) else 0.0)
    assert judge(m_floor, tiny)["verdict"] == "FAIL" and judge(ac, tiny)["verdict"] == "PASS"
    removed.append("resolution floor -> a 1e-8 relative mode appearing from zero FAILS")
    m_win = mutated("if ds < S25_STAB_WINDOW_STEPS:", "if ds < 1:", "window")
    assert judge(m_win, grow, steps=tuple(range(60, 660, 60)))["verdict"] == "FAIL" and judge(ac, grow, steps=tuple(range(60, 660, 60)))["verdict"] == "VOID"
    removed.append("1000-step window -> a 600-step run is judged instead of VOID")
    m_clot = mutated("if not any(n_clot_cells):", "if False:", "noclot")
    assert judge(m_clot, smooth, clot=lambda i, j: False)["verdict"] == "PASS"
    removed.append("no-clot-cell test -> a run without a clot PASSES")
    for x in shown:
        print("   stability:", x)
    for x in removed:
        print("   stability guard removed:", x)
    print(f"   stab: {len(shown)} judgements as expected; {len(removed)} guards deleted one at a time, each lets its negative pass")


def s25_build2_checks(only: str | None = None) -> None:
    assert S23_SOLVER_C.is_file(), f"the solver source is absent: {S23_SOLVER_C}"
    src = S23_SOLVER_C.read_text()
    sections = (("ltsolver", lambda: _s25_lt_solver_section(src)), ("ltgen", _s25_lt_gen_section),
                ("ltpairs", _s25_lt_pairs_section), ("ltassess", lambda: _s25_lt_assess_section(src)),
                ("5agen", _s25_5a_gen_section), ("oi76", _s25_oi76_section), ("b76off", _s25_b76off_section), ("b78", _s25_b78_section),
                ("rt1gen", _s25_rt1_gen_section), ("rt1watch", _s25_rt1_watch_section), ("rt1allrun", _s25_rt1_allrun_section),
                ("rt1gates", _s25_rt1_gates_section), ("stab", _s25_stability_section))
    ran = 0
    for name, fn in sections:
        if only is not None and only != name and not (only == "rt1" and name.startswith("rt1")):
            continue
        fn()
        ran += 1
    assert ran, f"no S25 build-2 section named {only!r}"


def s23_generator_negatives(only: str | None = None) -> None:
    """Each S23 generator refusal, seen to fire for its stated reason on a fresh copy (all of
    them, or the one whose label contains `only`); then the positive side: the key is written
    only when named, the BATCH 72 children have the settings of their parents, and each
    child.env refuses an unset, an empty and a library-less path when SOURCED."""
    import copy
    ran = 0
    with tempfile.TemporaryDirectory(prefix="am2020-s23-neg-") as tmp:
        root = build_case(Path(tmp))
        shutil.rmtree(root / "runs")
        controls_path = root / "constant" / "mainCaseControls.json"
        pristine = controls_path.read_text()
        for label, mutate, expected in S23_REFUSALS:
            if only is not None and only not in label:
                continue
            broken = copy.deepcopy(json.loads(pristine))
            mutate(broken)
            controls_path.write_text(json.dumps(broken, indent=1))
            done = subprocess.run(
                [sys.executable, "-B", str(root / "scripts" / "create_cases.py")],
                capture_output=True, text=True)
            said = (done.stdout + done.stderr).strip().splitlines()
            last = said[-1] if said else ""
            if done.returncode == 0:
                raise AssertionError(f"the generator ACCEPTED {label}; it must refuse")
            if expected not in last:
                raise AssertionError(f"the generator refused {label} for the WRONG reason: {last}")
            print(f"generator refuses: {label:<64} -> {last[:96]}")
            shutil.rmtree(root / "runs", ignore_errors=True)
            ran += 1
        controls_path.write_text(pristine)
        if only is None:
            # the pristine controls still generate, and a NAMED R_c is written exactly (a refusal
            # that always fires proves nothing: 1.0000001 is accepted, 1.5 is written). fsb_ctl_r3 is
            # used because it has no parent pair: no BATCH 72 child names the key, and a twin that did
            # would be refused by the parent check
            named = json.loads(pristine)
            # fsb_ctl_r3 is pinned to the S22 library (which does not read R_c, so the key is refused there) and has no
            # tau_v_s (refused on the S23 build): the twin carries both here
            _s16(named, "fsb_ctl_r3", volumetric_tear_ratio=1.5, solver_library=None, tau_v_s=0.0015)
            controls_path.write_text(json.dumps(named, indent=1))
            done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")],
                                  capture_output=True, text=True)
            assert done.returncode == 0, f"a named R_c = 1.5 was refused: {(done.stdout + done.stderr)[-300:]}"
            props = lambda cid: (root / "runs" / cid / "constant" / "phaseClotProperties").read_text()
            assert "volumetricTearRatio 1.5;" in props("fsb_ctl_r3")
            assert json.loads((root / "runs" / "fsb_ctl_r3" / "child.json").read_text())["s23"] == {"volumetric_tear_ratio": 1.5}
            shutil.rmtree(root / "runs", ignore_errors=True)
            named = json.loads(pristine)
            _s16(named, "fsb_ctl_r3", volumetric_tear_ratio=1.0000001, solver_library=None, tau_v_s=0.0015)
            controls_path.write_text(json.dumps(named, indent=1))
            done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")],
                                  capture_output=True, text=True)
            assert done.returncode == 0, f"R_c = 1.0000001 (just above the boundary) was refused: {(done.stdout + done.stderr)[-300:]}"
            shutil.rmtree(root / "runs", ignore_errors=True)
            # S25 section 2F, the positive side: a NON-UNIT time_scale is accepted at the scaled floor
            # (tau_v_s 5e-4 physical, exactly on the floor) and every physical time is written at S_t x its
            # physical value: the refusals above could all be a generator that refuses everything.
            named = json.loads(pristine)
            _s16(named, "fsb_ctl_r3", solver_library=None, tau_v_s=5e-4, time_scale=8.33)
            controls_path.write_text(json.dumps(named, indent=1))
            done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")],
                                  capture_output=True, text=True)
            assert done.returncode == 0, f"time_scale 8.33 at the scaled floor was refused: {(done.stdout + done.stderr)[-300:]}"
            _p = (root / "runs" / "fsb_ctl_r3" / "constant" / "phaseClotProperties").read_text()
            for _line in ("timeScale 8.33;", f"networkViscosityTau {8.33 * 5e-4!r};",
                          f"networkReleaseTime {3.3e-4 * 8.33!r};"):
                assert _line in _p, f"{_line!r} not written for time_scale 8.33"
            _j = json.loads((root / "runs" / "fsb_ctl_r3" / "child.json").read_text())
            assert _j["time_scale"] == 8.33 and _j["s16"]["networkReleaseTime"] == 3.3e-4 * 8.33
            assert _j["s19"]["networkViscosityTau"] == 8.33 * 5e-4
            assert s23_solver_would_refuse(_p) != "TS" and s23_solver_would_refuse(_p) != "R4", _p
            shutil.rmtree(root / "runs", ignore_errors=True)
            controls_path.write_text(pristine)
        done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")],
                              capture_output=True, text=True)
        assert done.returncode == 0, f"the pristine controls were refused: {(done.stdout + done.stderr)[-300:]}"
        props = lambda cid: (root / "runs" / cid / "constant" / "phaseClotProperties").read_text()
        # the key is written for NO child that does not name it (all four BATCH 72 children and every earlier one)
        every = sorted(p.name for p in (root / "runs").iterdir() if (p / "child.json").is_file())
        assert len(every) == KEPT_CHILDREN, (len(every), KEPT_CHILDREN)
        assert not [c for c in every if "volumetricTearRatio" in props(c)], "volumetricTearRatio written for a child that does not name it"
        # S25: every legacy child states S_t = 1.0, writes it, and writes 17 digits (the grid is not a filter: all of them)
        # the legacy children (every child outside the S25 5a series, whose six children (two smoke, the RT-1 pair, BATCH 78's RT-1 pair) state S_t = 25/3 on purpose)
        # (B79_Y_GATE is the 49th child: series s25_ygate, the 5a dictionary at S_t = 25/3 on the Y mesh, so it is outside "legacy" too)
        # (the 6 symmetric-Y children, series s25_ys794 / s25_ys850 / s25_ys794_plug / s25_ys850_plug, are likewise the 5a dictionary at S_t = 25/3 on the f = 2 Y meshes)
        legacy = [c for c in every if json.loads((root / "runs" / c / "child.json").read_text())["series"] not in ("s25_5a", "s25_ygate", "s25_ys794", "s25_ys850", "s25_ys794_plug", "s25_ys850_plug")]
        assert len(legacy) == KEPT_CHILDREN - 14, (len(legacy), KEPT_CHILDREN)
        assert all("\ntimeScale 1.0;\n" in props(c) for c in legacy), "a child's dictionary lacks timeScale 1.0"
        assert all(json.loads((root / "runs" / c / "child.json").read_text())["time_scale"] == 1.0 for c in legacy)
        non_legacy = ["B78_RT_5a_A", "B78_RT_5a_B", "B79_Y_GATE", "RT_5a_A", "RT_5a_B", "S25_5a_tan", "S25_5a_tan_half",
                      "Y80_RT86_B", "Y80_RT_A", "Y80_RT_B", "Y80_YS794_FLOW", "Y80_YS850_FLOW", "Y81_YS794_PLUG", "Y81_YS850_PLUG"]
        assert [c for c in every if c not in legacy] == non_legacy
        assert all("\ntimeScale 8.333333333333334;\n" in props(c) for c in non_legacy)
        # THE SYMMETRIC Y (design 2026-10-14-case-symmetric-y-plug.md rev. 2.1; case-engineer 27 and 27b): the schedule and the shared dictionaries, read off the generated files
        _ys = lambda cid, f: (root / "runs" / cid / f).read_text()
        _top = lambda cid: re.sub(r"(?s)functions.*", "", _ys(cid, "system/controlDict"))
        for _g in ("Y80_YS794_FLOW", "Y80_YS850_FLOW"):                       # the flow gates: runTime tau1, 18 tau1 + 2e-5 s
            for _pat in (r"(?m)^writeControl\s+runTime;", r"(?m)^writeInterval\s+0\.00050976;", r"(?m)^endTime\s+0\.00919568;", r"(?m)^purgeWrite\s+0;"):
                assert re.search(_pat, _top(_g)), (_g, _pat)
        # the plug child: 100 ms + 5e-5 s (design rev. 2.2 A2; was 185 ms), NOT a multiple of the 1 ms write interval (CE ruling, DECISIONS 2026-10-14): the 100th write cannot tie with the stop
        for _g in ("Y81_YS794_PLUG", "Y81_YS850_PLUG"):
            for _pat in (r"(?m)^writeControl\s+runTime;", r"(?m)^writeInterval\s+0\.001;", r"(?m)^endTime\s+0\.10005;", r"(?m)^purgeWrite\s+0;"):
                assert re.search(_pat, _top(_g)), (_g, _pat)
        # the YS850 plug is the YS794 plug's dictionaries on the other mesh (BATCH 81, case-engineer 28): the same controlDict, fvSchemes, fvSolution, phaseClotProperties, physicalProperties and momentumTransport
        for _f in ("constant/phaseClotProperties", "constant/physicalProperties", "constant/momentumTransport", "system/controlDict", "system/fvSchemes", "system/fvSolution"):
            assert (root / "runs" / "Y81_YS794_PLUG" / _f).read_bytes() == (root / "runs" / "Y81_YS850_PLUG" / _f).read_bytes(), ("YS850 plug", _f)
        # RT_B and the plug child share every file byte for byte except the controlDict, and the controlDict differs in endTime, the wallBudget stopTime and the stopAfterStep object ONLY
        for _f in ("constant/phaseClotProperties", "constant/physicalProperties", "constant/momentumTransport", "system/fvSchemes", "system/fvSolution", "system/blockMeshDict"):
            assert (root / "runs" / "Y81_YS794_PLUG" / _f).read_bytes() == (root / "runs" / "Y80_RT_B" / _f).read_bytes(), _f
        _y81, _rtb = _ys("Y81_YS794_PLUG", "system/controlDict"), _ys("Y80_RT_B", "system/controlDict")
        _blk = re.findall(r"\n    stopAfterStep\n    \{[^}]*\}", _rtb)
        assert len(_blk) == 1 and "type            stopAtFile;" in _blk[0] and "action          writeNow;" in _blk[0], _blk
        _back = _rtb.replace(_blk[0], "")
        for _was, _is in (("endTime 0.003;", "endTime 0.10005;"), ("stopTime        1800;", "stopTime        36000;")):
            assert _back.count(_was) == 1, _was
            _back = _back.replace(_was, _is)
        assert _back == _y81, "RT_B's controlDict differs from the plug child's by more than endTime, the wallBudget stopTime and the stopAfterStep object"
        # RT_A writes every step (timeStep 1, purgeWrite 10) and stops after step 105 through the same object
        for _pat in (r"(?m)^writeControl\s+timeStep;", r"(?m)^writeInterval\s+1;", r"(?m)^purgeWrite\s+10;", r"stopAfterStep"):
            assert re.search(_pat, _ys("Y80_RT_A", "system/controlDict")), ("Y80_RT_A", _pat)
        # Y80_RT86_B (BATCH 82, case-engineer 33) is Y80_RT_B with ONE number moved, the leg-1 stop: EVERY file of the generated child is byte-identical to Y80_RT_B's except child.json, whose
        # only differences are child_id and stop_after_step.leg1 (86 for 100); the file lists are equal, so no file appears or disappears with the stop
        _rb, _r86 = root / "runs" / "Y80_RT_B", root / "runs" / "Y80_RT86_B"
        _fb = sorted(str(p_.relative_to(_rb)) for p_ in _rb.rglob("*") if p_.is_file())
        assert _fb == sorted(str(p_.relative_to(_r86)) for p_ in _r86.rglob("*") if p_.is_file()) and len(_fb) > 10, "Y80_RT86_B and Y80_RT_B do not hold the same files"
        assert [f for f in _fb if (_rb / f).read_bytes() != (_r86 / f).read_bytes()] == ["child.json"], "Y80_RT86_B differs from Y80_RT_B in more than child.json"
        _jb, _j86 = json.loads(_ys("Y80_RT_B", "child.json")), json.loads(_ys("Y80_RT86_B", "child.json"))
        assert sorted(k_ for k_ in set(_jb) | set(_j86) if _jb.get(k_) != _j86.get(k_)) == ["child_id", "stop_after_step"], "Y80_RT86_B's child.json differs from Y80_RT_B's in more than child_id and the stop"
        assert _jb["stop_after_step"] == {"leg1": 100, "leg2": 105} and _j86["stop_after_step"] == {"leg1": 86, "leg2": 105}, (_jb["stop_after_step"], _j86["stop_after_step"])
        for _c in ("Y80_YS794_FLOW", "Y80_YS850_FLOW", "Y80_RT_A", "Y80_RT_B", "Y80_RT86_B", "Y81_YS794_PLUG", "Y81_YS850_PLUG"):
            assert not (root / "runs" / _c / "child.env").exists() and "solver_library" not in json.loads(_ys(_c, "child.json")), _c
            assert re.findall(r"(?m)^\s*nNonOrthogonalCorrectors\s+(\S+);", _ys(_c, "system/fvSolution")) == ["2"], _c
        # BATCH 79 (design 2026-10-14-case-y-bifurcation-gate.md rev. 2.2): the Y child writes at runTime tau1 and stops at 18 tau1 + 2e-5 s, with 2 non-orthogonal correctors,
        # no child.env and no solver_library; the dictionary is S25_5a_tan's (the clot children inherit it)
        _y = root / "runs" / "B79_Y_GATE"
        _cd = (_y / "system" / "controlDict").read_text()
        for _pat in (r"(?m)^writeControl\s+runTime;", r"(?m)^writeInterval\s+0\.00050976;", r"(?m)^endTime\s+0\.009196;", r"(?m)^purgeWrite\s+0;"):
            assert re.search(_pat, _cd), ("B79_Y_GATE controlDict", _pat)
        assert "timeStep" not in re.sub(r"(?s)functions.*", "", _cd), "B79_Y_GATE writes by timeStep outside its function objects"
        assert re.findall(r"(?m)^\s*nNonOrthogonalCorrectors\s+(\S+);", (_y / "system" / "fvSolution").read_text()) == ["2"]
        assert not (_y / "child.env").exists() and "solver_library" not in json.loads((_y / "child.json").read_text())
        assert "write_every_n_steps" not in json.loads((_y / "child.json").read_text())
        # B76-RT (case-engineer 09d): the RT-1 children carry their parents' phaseClotProperties exactly (only the run controls differ), A = B, no library pin
        assert props("RT_P73_A") == props("RT_P73_B") == props("P73_G4_r3_40"), "an RT_P73 child's phaseClotProperties drifted from its parent"
        assert props("RT_5a_A") == props("RT_5a_B") == props("S25_5a_tan"), "an RT_5a child's phaseClotProperties drifted from its parent"
        # BATCH 78 (S25.1, case-engineer 06): the RT_5a pair is RT_5a_A / RT_5a_B's dictionary under new ids; the OFF/CTL children are P73_G4_r3_40's dictionary exactly (end time and write
        # interval are controlDict keys, not phaseClotProperties); only B78_OFF_A carries the baseline pin, so the library is the ONLY difference between the OFF arms' children
        assert props("B78_RT_5a_A") == props("B78_RT_5a_B") == props("RT_5a_A"), "a BATCH 78 RT_5a child's phaseClotProperties drifted from its parent"
        assert props("B78_OFF_A") == props("B78_OFF_B") == props("B78_CTL_1") == props("B78_CTL_2") == props("P73_G4_r3_40"), "a BATCH 78 OFF/CTL child's phaseClotProperties drifted from P73_G4_r3_40"
        assert json.loads((root / "runs" / "B78_OFF_A" / "child.json").read_text())["solver_library"] == "baseline"
        for _f in ("system/controlDict", "system/fvSolution", "system/fvSchemes", "constant/physicalProperties", "constant/momentumTransport"):
            assert (root / "runs" / "B78_OFF_A" / _f).read_bytes() == (root / "runs" / "B78_OFF_B" / _f).read_bytes(), _f
            assert (root / "runs" / "B78_CTL_1" / _f).read_bytes() == (root / "runs" / "B78_CTL_2" / _f).read_bytes(), _f
        assert (root / "runs" / "B78_CTL_1" / "child.json").read_text().replace("B78_CTL_1", "B78_CTL_2") == (root / "runs" / "B78_CTL_2" / "child.json").read_text()
        for c in ("RT_P73_A", "RT_P73_B", "RT_5a_A", "RT_5a_B", "B78_RT_5a_A", "B78_RT_5a_B", "B79_Y_GATE"):
            assert not (root / "runs" / c / "child.env").exists() and "solver_library" not in json.loads((root / "runs" / c / "child.json").read_text()), c
        assert all(re.search(r"(?m)^writePrecision\s+17;", (root / "runs" / c / "system" / "controlDict").read_text())
                   for c in every), "a controlDict is not at writePrecision 17"
        # rev. 4.1: the only child.json with an "s23" block is the labelled twin, and it carries the marker only
        assert [c for c in every if "s23" in json.loads((root / "runs" / c / "child.json").read_text())] == ["S23v_G4_nor3"]
        assert json.loads((root / "runs" / "S23v_G4_nor3" / "child.json").read_text())["s23"] == {"sensitivity_limit": "tanDelta0"}
        # (1) and (1') are the same dictionary
        assert props("S23v_G4_r3") == props("S23v_G4_r3_S22") == props("fsf_G4_kxu_r3_40")
        # the twin is its (pinned) parent's dictionary plus the marker, written exactly as the solver reads it
        assert props("S23v_G4_nor3") == props("fsf_G4_kxu_nor3_40").rstrip("\n") + "\n\nsensitivityLimit tanDelta0;\n"
        assert props("S23v_G40") == props("fsf_G40_klo_v250")
        # BATCH 73: the production pair has its parents' dictionaries, and no S23 block, no marker, no library pin
        assert props("P73_G4_r3_40") == props("S23v_G4_r3") and props("P73_G40_40") == props("S23v_G40")
        # BATCH 74: the diagnostic child has its parent's dictionary exactly (the passive field is the build's), no library pin, no S23 block
        assert props("D74_refI") == props("S23v_G4_r3")
        # BATCH 75: the re-run child has the same parent's dictionary, no library pin, no S23 block
        assert props("D75_refI") == props("S23v_G4_r3")
        for c in ("P73_G4_r3_40", "P73_G40_40", "D74_refI", "D75_refI", "B76_OFF_B", "B78_OFF_B", "B78_CTL_1", "B78_CTL_2"):
            assert not (root / "runs" / c / "child.env").exists(), c
            assert "solver_library" not in json.loads((root / "runs" / c / "child.json").read_text()), c
        # B76-OFF (S25 section 7): both arms have P73_G4_r3_40's dictionary exactly (timeScale 1.0 included: the baseline source ignores it) and
        # differ in the binary alone; A (and only A) carries the baseline pin, so the library is the ONLY difference between the arms' children
        assert props("B76_OFF_A") == props("B76_OFF_B") == props("P73_G4_r3_40")
        assert json.loads((root / "runs" / "B76_OFF_A" / "child.json").read_text())["solver_library"] == "baseline"
        for _f in ("system/controlDict", "system/fvSolution", "system/fvSchemes", "constant/physicalProperties", "constant/momentumTransport"):
            assert (root / "runs" / "B76_OFF_A" / _f).read_bytes() == (root / "runs" / "B76_OFF_B" / _f).read_bytes(), _f
        # the marker is written for ONE child of the 30 (the whitelist), as a word, with nothing else near it
        assert [c for c in every if "sensitivityLimit" in props(c)] == ["S23v_G4_nor3"]
        assert len(re.findall(r"(?m)^\s*sensitivityLimit\s+tanDelta0;\s*$", props("S23v_G4_nor3"))) == 1
        # child.env: the 2 earlier library children and the 13 pinned ones (item 5), nothing else
        pinned = sorted(S23_PINNED_S22)
        assert sorted(c for c in every if (root / "runs" / c / "child.env").exists()) == sorted(["S22v_G40_A0", "S23v_G4_r3_S22", "B76_OFF_A", "B78_OFF_A"] + pinned)
        for c in ["S23v_G4_r3_S22"] + pinned:
            assert json.loads((root / "runs" / c / "child.json").read_text())["solver_library"] == "s22", c
        assert [c for c in every if json.loads((root / "runs" / c / "child.json").read_text()).get("solver_library") == "s22"] == sorted(["S23v_G4_r3_S22"] + pinned)
        # nOuterCorrectors is 1 on the four arms (the end-of-step partition rule of the log; section 13.6 hand-off)
        for c in ("S23v_G4_r3", "S23v_G4_r3_S22", "S23v_G4_nor3", "S23v_G40", "P73_G4_r3_40", "P73_G40_40", "D74_refI", "D75_refI"):
            fvs = (root / "runs" / c / "system" / "fvSolution").read_text()
            assert re.findall(r"(?m)^\s*nOuterCorrectors\s+(\S+);", fvs) == ["1"], (c, re.findall(r"nOuterCorrectors\s+\S+;", fvs))
        # the solver's own R1-R4 + marker/kelvinVoigt predicate (mirrored from its source text, below), applied to
        # every generated dictionary: a child on the S23 build is accepted, a pinned child is exactly the kind the
        # build would refuse (so the pin is necessary), and no child is both on the build and refused
        for c in every:
            lib = json.loads((root / "runs" / c / "child.json").read_text()).get("solver_library")
            why = s23_solver_would_refuse(props(c))
            if lib in (None, "baseline"):   # the baseline IS the S23 build (release-e2a57db1): its dictionary is accepted by the S23 predicate
                assert why is None, (c, why)
            elif c in pinned:
                assert why in ("R2", "R3"), (c, why)
        assert {c: s23_solver_would_refuse(props(c)) for c in pinned}.get("fsb_ctl_r3") == "R3"
        assert sum(s23_solver_would_refuse(props(c)) == "R2" for c in pinned) == 12
        s23_solver_mirror_checks()
        # production write settings on all four (5 ms full writes, 0.2 ms frames)
        ends = {"S23v_G4_r3": 0.04, "S23v_G4_r3_S22": 0.03, "S23v_G4_nor3": 0.04, "S23v_G40": 0.025, "D74_refI": 0.04, "D75_refI": 0.04, "B76_OFF_A": 0.03, "B76_OFF_B": 0.03}
        for cid, end in ends.items():
            j = json.loads((root / "runs" / cid / "child.json").read_text())
            cd = (root / "runs" / cid / "system" / "controlDict").read_text()
            assert f"endTime         {end};" in cd or re.search(rf"endTime\s+{end};", cd), (cid, "endTime")
            assert re.search(r"writeInterval\s+0\.005;", cd), (cid, "writeInterval")
        # BATCH 78: 18 ms with full writes at 6 ms (18 is not a whole number of 5 ms), and the 84-step control with its one full write at 0.4 ms
        for cid, (end, wi) in {"B78_OFF_A": (0.018, 0.006), "B78_OFF_B": (0.018, 0.006), "B78_CTL_1": (0.0004, 0.0004), "B78_CTL_2": (0.0004, 0.0004)}.items():
            cd = (root / "runs" / cid / "system" / "controlDict").read_text()
            assert re.search(rf"(?m)^endTime\s+{end};", cd) and re.search(rf"(?m)^writeInterval\s+{wi};", cd), (cid, "B78 endTime / writeInterval")
            assert re.search(r"(?m)^writeControl\s+adjustableRunTime;", cd), (cid, "B78 writeControl")
        # child.env, SOURCED the way Allrun does (bash, `.`), each refusal seen to fire
        envs = {"s22": (root / "runs" / "S23v_G4_r3_S22" / "child.env", "PHASE_CLOT_S22_LIBBIN", "PHASE_CLOT_P2PRIME_LIBBIN"),
                "p2prime": (root / "runs" / "S22v_G40_A0" / "child.env", "PHASE_CLOT_P2PRIME_LIBBIN", "PHASE_CLOT_S22_LIBBIN"),
                "baseline": (root / "runs" / "B76_OFF_A" / "child.env", "PHASE_CLOT_BASELINE_LIBBIN", "PHASE_CLOT_S22_LIBBIN")}
        good = Path(tmp) / "libbin_good"
        good.mkdir()
        (good / "libphaseClotAxisymmetricFluid.so").write_bytes(b"x")
        bare = Path(tmp) / "libbin_empty"
        bare.mkdir()

        def source(envfile, setting, ld="/opt/foam/lib"):
            cmd = f'unset PHASE_CLOT_S22_LIBBIN PHASE_CLOT_P2PRIME_LIBBIN PHASE_CLOT_BASELINE_LIBBIN; {setting} LD_LIBRARY_PATH={ld}; ' \
                  f'. {envfile}; rc=$?; echo "rc=$rc"; echo "LD=$LD_LIBRARY_PATH"'
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            out = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
            return int(out["rc"]), out.get("LD", ""), r.stderr
        for name, (envfile, var, other) in envs.items():
            rc, ld, err = source(envfile, "")
            assert rc == 97 and ld == "/opt/foam/lib" and var in err, (name, "unset", rc, ld, err)
            rc, ld, err = source(envfile, f"export {var}='';")
            assert rc == 97 and ld == "/opt/foam/lib", (name, "empty", rc, ld)
            rc, ld, err = source(envfile, f"export {other}='{good}';")
            assert rc == 97 and ld == "/opt/foam/lib", (name, "the OTHER library's variable must not satisfy it", rc, ld)
            if name in ("s22", "baseline"):
                rc, ld, err = source(envfile, f"export {var}='{Path(tmp) / 'nonexistent'}';")
                assert rc == 98 and ld == "/opt/foam/lib", (name, "nonexistent dir", rc, ld)
                rc, ld, err = source(envfile, f"export {var}='{bare}';")
                assert rc == 98 and ld == "/opt/foam/lib", (name, "dir without the library", rc, ld)
            rc, ld, err = source(envfile, f"export {var}='{good}';")
            assert rc == 0 and ld == f"{good}:/opt/foam/lib", (name, "valid", rc, ld)
            print(f"child.env ({name}): unset -> 97, empty -> 97, the other library's variable -> 97"
                  + (", missing dir -> 98, library-less dir -> 98" if name in ("s22", "baseline") else "") + ", valid -> 0 with LD_LIBRARY_PATH prefixed")
        shutil.rmtree(root / "runs", ignore_errors=True)
    assert ran > 0, f"no S23 negative matches {only!r}"
    print(f"   {ran} S23 generator negatives refused; the key is written only when named; "
          f"the BATCH 72 children carry the production write settings; "
          f"the marker is written for S23v_G4_nor3 only; 13 children pinned to s22; nOuterCorrectors 1 on the four arms")



def filtered_generation(filt, mutate_controls=None, patch_generator=None):
    with tempfile.TemporaryDirectory(prefix="am2020-launch-") as tmp:
        root = build_case(Path(tmp))
        shutil.rmtree(root / "runs")
        if mutate_controls:
            cp = root / "constant" / "mainCaseControls.json"
            c = json.loads(cp.read_text())
            mutate_controls(c)
            cp.write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n")
        if patch_generator:
            gp = root / "scripts" / "create_cases.py"
            old, rep = patch_generator
            g = gp.read_text()
            assert g.count(old) == 1, "generator patch anchor"
            gp.write_text(g.replace(old, rep))
        done = subprocess.run(
            [sys.executable, "-B", str(root / "scripts" / "create_cases.py")] + list(filt),
            capture_output=True, text=True)
        built = ([e["child_id"] for e in json.loads((root / "runs" / "matrix.json").read_text())]
                 if done.returncode == 0 else None)
        lines = (done.stdout + done.stderr).strip().splitlines()
        return done.returncode, built, (lines[-1] if lines else "")


def launch_filters_generate(nbsel, batch: int, generate) -> None:
    """BATCH 72 is two Colab sessions: each launch runs create_cases.py with ITS OWN filter, so each
    filter must generate by itself (exactly its children) on a fresh copy. BATCH 73 is ONE session (no
    LAUNCHES entry): its single filter is the batch's children. `generate` is filtered_generation (or a
    stand-in); nbsel is check_notebook_batch_selector."""
    nb = Path(__file__).resolve().parents[3] / "notebooks" / "AM2020_main_case_colab.ipynb"
    names = nbsel.notebook_names(nbsel.launch_cell(nb))
    parts = names["LAUNCHES"].get(batch)
    if not parts:
        assert batch in names["BATCHES"], f"batch {batch} is not in BATCHES"
        parts = {1: (names["BATCHES"][batch][0], names["BATCHES"][batch][1])}
    for lnum, (only, _hours) in sorted(parts.items()):
        filt = only.split()
        rc, built, last = generate(filt)
        assert rc == 0, f"batch {batch} launch {lnum}'s filter was refused: {last}"
        assert sorted(built) == sorted(filt), (lnum, built, filt)
        if batch in names["ONE_CORE_BATCHES"]:
            # S25: Allrun runs a launch's children in MATRIX order (runs/matrix.json), never in the order BATCHES names them, and a restart pair must run A then B
            # (B restarts from its sibling and is refused unless A exited 0): the matrix order of each launch IS its run order, so it must equal the notebook's order
            assert built == filt, f"batch {batch} launch {lnum}: Allrun would run {built}, the notebook's session is {filt}"
        print(f"   batch {batch} launch {lnum}: {len(filt)} children built, exactly the launch's filter"
              f"{', in the notebook order' if batch in names['ONE_CORE_BATCHES'] else ''} ({last})")


def y_gate_only_matrix() -> None:
    """BATCH 79 AS THE NOTEBOOK LAUNCHES IT: create_cases.py with the batch's own filter builds ONE child, the clot-free Y gate, and assess_case.main runs
    on that one-child matrix (docket 20261013-2314 report 14: the Y child became the clot reference, deformation_table read its clot height, the
    assessor exited 1 after the 9 h run and wrote no result.json; every earlier Y scenario ran the full 49-child matrix, whose reference is a clot child).
    Clean y_synth data must give result.json, exit 0, READY; a planted wrong split (the rejected network's 0.809042) must give NOT_READY by gate 2 alone.
    The reference rule itself is exercised on literal matrices: a clot child is the reference even when the Y child sorts first, no clot child means
    no reference, and a growth or deformation child with no clot is refused by name."""
    import importlib.util
    root_repo = Path(__file__).resolve().parents[3]
    nb = root_repo / "notebooks" / "AM2020_main_case_colab.ipynb"
    nbs = importlib.util.spec_from_file_location("nbsel_yonly", root_repo / "scripts" / "check_notebook_batch_selector.py")
    nbsel = importlib.util.module_from_spec(nbs)
    nbs.loader.exec_module(nbsel)
    filt = nbsel.notebook_names(nbsel.launch_cell(nb))["BATCHES"][79][0].split()
    assert filt == ["B79_Y_GATE"], filt
    for label, y_options, verdict, failing in (
            ("clean", {}, "READY", []),
            ("wrong split 0.809042", dict(split=0.809042), "NOT_READY", ["g2"])):
        with tempfile.TemporaryDirectory(prefix="am2020-yonly-") as tmp:
            root = build_case(Path(tmp))
            shutil.rmtree(root / "runs")
            done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")] + filt, capture_output=True, text=True)
            assert done.returncode == 0, (done.stdout + done.stderr)[-600:]
            matrix = json.loads((root / "runs" / "matrix.json").read_text())
            assert [m["child_id"] for m in matrix] == filt, [m["child_id"] for m in matrix]
            assert matrix[0]["clot"].get("shape") == "none" and "height_m" not in matrix[0]["clot"], matrix[0]["clot"]
            (root / "log.Allwmake").write_text(GOOD_BUILD)
            synthesise_child(root / "runs" / filt[0], matrix[0], y_options=y_options)
            code, result = run_assessor(root)             # raises when no result.json: the pre-fix failure
            assert code == 0, f"{label}: assess_case exited {code}"
            assert result["children_total"] == 1 and result["children_admissible"] == 1 and not result["inadmissible"], (label, result["inadmissible"])
            y = result["children"][filt[0]]
            assert y["y_gate_verdict"] == verdict and y["y_gate"]["failing"] == failing, (label, y["y_gate_verdict"], y["y_gate"]["failing"])
            assert result["y_gate"] == {filt[0]: {"verdict": verdict, "failing": failing}}, result["y_gate"]
            d = result["deformation"]                     # the notebook's report cell reads these keys: the schema stays, with no clot height
            assert d["rows"] == [] and d["excluded"] == [] and d["clot_height_m"] is None and d["deforms_visibly"] is False, d
            assert result["growth_vs_flow"]["rows"] == [] and result["growth_vs_flow"]["flow_dependence"]["exponent"] is None
            print(f"   batch 79's one-child matrix, {label}: result.json written, exit 0, {filt[0]} {verdict} {failing}")
    sp = importlib.util.spec_from_file_location("assess_case_yonly", Path(__file__).resolve().parent / "assess_case.py")
    ac = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(ac)
    y = {"child_id": "Y", "series": "s25_ygate", "geometry": {"h_m": 1e-6}, "clot": {"shape": "none"}}
    d1 = {"child_id": "D1", "series": "deformation", "geometry": {"h_m": 2e-6}, "clot": {"shape": "disc", "height_m": 3e-5}}
    g1 = {"child_id": "G1", "series": "growth", "geometry": {"h_m": 2e-6}, "clot": {"shape": "disc", "height_m": 3e-5}}
    assert ac.clot_reference([y], set()) is None
    assert ac.clot_reference([y, d1], set()) is d1, "a clot child must be the reference even when the clot-free child sorts first"
    assert ac.clot_reference([d1, y], set()) is d1
    assert ac.clot_reference([y, d1, g1], {"deformation"}) is g1 and ac.clot_reference([y, g1, d1], {"growth"}) is d1, "own-geometry series are skipped"
    assert ac.clot_reference([y, d1], {"deformation"}) is d1, "only own-geometry clot children: the first clot child"
    for bad_matrix, why in (([y, dict(d1, clot={"shape": "none"})], "deformation child with no clot"),
                            ([g1, dict(d1, geometry={"h_m": 9e-6})], "geometry differs")):
        try:
            ac.clot_reference_refusal(bad_matrix, ac.clot_reference(bad_matrix, set()))
        except ValueError as error:
            print(f"   refused ({why}): {str(error)[:110]}")
        else:
            raise AssertionError(f"{why}: ACCEPTED")
    ac.clot_reference_refusal([y, d1, g1], ac.clot_reference([y, d1, g1], set()))
    # BATCH 80 (verification 29, F1): the free occlusive plug is a clot WITHOUT a height (y_plug.PLUG_KEYS), so it is never the reference, whatever sorts first
    plug = {"child_id": "P", "series": "s25_ys794_plug", "geometry": {"h_m": 1.5e-6}, "clot": {"shape": "occlusive plug", "width_parent_D0": 1.0}}
    assert not ac.carries_reference_clot(plug) and not ac.carries_reference_clot(y) and ac.carries_reference_clot(d1)
    assert ac.clot_reference([plug], set()) is None and ac.clot_reference([plug, y], set()) is None, "a plug-only matrix has no reference"
    assert ac.clot_reference([plug, d1], set()) is d1 and ac.clot_reference([plug, y, d1, g1], set()) is d1, "a plug that sorts first is never the reference"
    try:
        ac.clot_reference_refusal([plug, dict(d1, clot={"shape": "occlusive plug"})], ac.clot_reference([plug], set()))
    except ValueError as error:
        print(f"   refused (a deformation child whose clot is a plug): {str(error)[:110]}")
    else:
        raise AssertionError("a deformation child with a plug clot: ACCEPTED")
    ac.clot_reference_refusal([plug, y], None)


def s19_source_errors_are_case_relative() -> None:
    """THE REPORT NAMES THE CHILD, NOT WHERE IT WAS UNPACKED (case-engineer 31).

    s19_report writes the exception a reader raised (g4_source_error, n_cells_source_error). free_shell_timeseries.read_surface_field_value puts the case path it was
    given into the message, so, when numpy imports, the same returned run reported twice differed in the temporary directory's name (the release of BATCH 80 stopped
    there: theta_native_grades_identically, 38 children). Without numpy the text is `No module named 'numpy'` and nothing shows. The reader is stubbed here with the same
    message so the test does not depend on the interpreter; the same case is reported from two different directories and the two reports must be equal, with no
    directory of the run in either."""
    import importlib.util
    import types
    sp = importlib.util.spec_from_file_location("assess_case_relerr", Path(__file__).resolve().parent / "assess_case.py")
    ac = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(ac)
    stub = types.ModuleType("free_shell_timeseries")
    stub.Q0_REF_M3_S, stub.Q0_REF_DP = 1.0, 1.0

    def read_surface_field_value(case, name, column):
        raise FileNotFoundError(f"{case}: no postProcessing/{name}/<t>/surfaceFieldValue.dat")
    stub.read_surface_field_value = read_surface_field_value
    kept = sys.modules.get("free_shell_timeseries")
    sys.modules["free_shell_timeseries"] = stub
    try:
        seen = []
        for _ in range(2):
            with tempfile.TemporaryDirectory(prefix="am2020-main-") as t:
                case = Path(t) / "runs" / "C1"
                (case / "0").mkdir(parents=True)
                (case / "0" / "p").write_text("inlet\n{\n    type fixedValue;\n    value uniform 10;\n}\n")
                report = ac.s19_report("", {"s19": {}}, case)
                err = report["g4_source_error"]
                assert t not in err and str(case) not in err, f"the report carries the run's directory: {err}"
                seen.append(err)
                # any exception that names a file under the case (the n_cells reader's open), by the path as given and as resolved
                for named in (case / "constant" / "polyMesh" / "owner", case.resolve() / "constant" / "polyMesh" / "owner"):
                    text = ac.case_relative_error(OSError(f"cannot read {named}"), case)
                    assert text == "OSError('cannot read C1/constant/polyMesh/owner')", text
                assert ac.case_relative_error(ValueError("no path here"), case) == repr(ValueError("no path here"))
                assert ac.case_relative_error(ValueError(f"under {case.resolve()}/x"), None) == repr(ValueError(f"under {case.resolve()}/x")), "case None: the text is unchanged"
        assert seen[0] == seen[1] == "FileNotFoundError('C1: no postProcessing/inletFlux/<t>/surfaceFieldValue.dat')", seen
    finally:
        if kept is None:
            sys.modules.pop("free_shell_timeseries", None)
        else:
            sys.modules["free_shell_timeseries"] = kept


def y_launch_trees() -> None:
    """THE LAUNCH TREES AS THE NOTEBOOK BUILDS THEM (verification 29, F1): each Colab launch runs create_cases.py with ITS OWN filter and Allrun ends with assess_case.py on that
    tree, not on the full 55-child matrix the other scenarios score (whose clot reference is an earlier channel disc). BATCH 80 launch 1 and launch 2, BATCH 81 launch 1 and launch 2
    (the two plug chains, STAGED), BATCH 82 launch 1 and launch 2 (the confirming restart test, VOID) and BATCH 83 launch 1 and launch 2 (its corrected retest, the default) are read from the notebook with the selector's parser. Each tree gets every child synthesised as a clean returned run,
    and assess_case.main must exit 0, write result.json, and carry the y_gate and y_rt blocks; a plug child that wrote no full write adds NO y_plug summary (status NO_WRITES). Before the fix a plug became the clot reference and deformation_table read its
    missing height_m: KeyError, no result.json, INVALID / NO_ASSESSOR_RESULT."""
    import importlib.util
    root_repo = Path(__file__).resolve().parents[3]
    nbs = importlib.util.spec_from_file_location("nbsel_launch", root_repo / "scripts" / "check_notebook_batch_selector.py")
    nbsel = importlib.util.module_from_spec(nbs)
    nbs.loader.exec_module(nbsel)
    names = nbsel.notebook_names(nbsel.launch_cell(root_repo / "notebooks" / "AM2020_main_case_colab.ipynb"))
    trees = {"BATCH 80 launch 1": names["LAUNCHES"][80][1][0].split(), "BATCH 80 launch 2": names["LAUNCHES"][80][2][0].split(), "BATCH 81 launch 1": names["LAUNCHES"][81][1][0].split(), "BATCH 81 launch 2": names["LAUNCHES"][81][2][0].split(),
             "BATCH 82 launch 1": names["LAUNCHES"][82][1][0].split(), "BATCH 82 launch 2": names["LAUNCHES"][82][2][0].split(),
             "BATCH 83 launch 1": names["LAUNCHES"][83][1][0].split(), "BATCH 83 launch 2": names["LAUNCHES"][83][2][0].split()}
    assert trees["BATCH 80 launch 1"] == ["Y80_YS794_FLOW", "Y80_YS850_FLOW", "Y80_RT_A", "Y80_RT_B"] and trees["BATCH 80 launch 2"] == ["Y80_RT_B"], trees
    assert trees["BATCH 81 launch 1"] == ["Y81_YS794_PLUG"] and trees["BATCH 81 launch 2"] == ["Y81_YS850_PLUG"], trees
    assert trees["BATCH 82 launch 1"] == ["Y80_RT86_B"] and trees["BATCH 82 launch 2"] == ["Y80_RT86_B"], trees
    assert trees["BATCH 83 launch 1"] == ["Y80_RT86_B", "Y80_RT_A"] and trees["BATCH 83 launch 2"] == ["Y80_RT86_B"], trees       # the continuous arm and leg 1 in ONE launch, in the generator's matrix order
    for label, filt in trees.items():
        with tempfile.TemporaryDirectory(prefix="am2020-ylaunch-") as tmp:
            root = build_case(Path(tmp))
            shutil.rmtree(root / "runs")
            done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")] + filt, capture_output=True, text=True)
            assert done.returncode == 0, (done.stdout + done.stderr)[-600:]
            matrix = json.loads((root / "runs" / "matrix.json").read_text())
            assert sorted(m["child_id"] for m in matrix) == sorted(filt), (label, [m["child_id"] for m in matrix])
            (root / "log.Allwmake").write_text(GOOD_BUILD)
            for spec in matrix:
                synthesise_child(root / "runs" / spec["child_id"], spec)
            code, result = run_assessor(root)            # raises when no result.json: the pre-fix failure
            assert code == 0, f"{label}: assess_case exited {code}"
            assert "y_gate" in result and "y_rt" in result, (label, sorted(result)[:40])
            assert result["children_total"] == len(filt), (label, result["children_total"])
            gates = {c: v["verdict"] for c, v in result["y_gate"].items()}
            assert all(c.endswith("_FLOW") for c in gates) and sorted(gates) == sorted(f for f in filt if f.endswith("_FLOW")), (label, gates)
            assert all(v == "READY" for v in gates.values()), (label, result["y_gate"])
            assert "y_plug" not in result, (label, "a plug child with no full write must not add a y_plug summary", result.get("y_plug"))
            rt = [(e["pair_id"], e["status"]) for e in result["y_rt"]]
            assert result["deformation"]["rows"] == [] and result["deformation"]["clot_height_m"] is None, result["deformation"]
            print(f"   {label}: {filt} -> exit 0, result.json, y_gate {gates}, y_rt {rt}, {result['children_admissible']}/{result['children_total']} admissible")


def y_plug_chain_scenarios() -> None:
    """BATCH 81'S TWO CHAINS, each of two Colab sessions, END TO END ON RETURNED CASES (case-engineer 28b; the plug assessor, scripts/y_gate/y_plug_assess.py, and its CLI scripts/y_plug_chain.py).
    For each plug child (Y81_YS794_PLUG, then Y81_YS850_PLUG), generated by the notebook's own launch filter, a returned case of SESSION 1 (0/ and full writes 1..30 ms, then a
    writeNow at 30.41 ms: not write-aligned) and of SESSION 2 (restart.json, the restart point, then the writes to 4 ms past t_c + 2 tau_A) is laid out from physics 24's STUCK law on the
    real f = 2 mesh (y_plug_synth: a rigid shift of the C1 plug; the written fields as the solver writes them). Expected, each seen:
      session 1 alone                  y_plug_chain.py exit 3 (NOT_READABLE_YET: the 2 tau_A read lies past the session), and assess_case.y_plug_session says so
      session 2 alone                  assess_case.y_plug_session CHAIN_NEEDED (a fragment gives no verdict)
      the 2-session chain              exit 0, VERDICT STUCK (the model's R = exp(-tau_A/tau) = 0.37 <= 0.5), final, sessions 2, a merged series on the 1 ms grid with the restart point once
      a cold second session            exit 1, CHAIN_REFUSED (it repeats the chain from t = 0)
      the chain with a write removed   exit 1, CHAIN_REFUSED (a gap)
      the chain in swapped order       exit 1, CHAIN_REFUSED
    Pure python (no numpy): the same under both interpreters."""
    import importlib.util
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here / "y_gate"))
    sys.path.insert(0, str(here))
    import y_plug_assess as PA
    import y_plug_synth as PS
    controls_path = here.parent / "constant" / "mainCaseControls.json"
    params = json.loads(controls_path.read_text())["y_plug_verdict"]
    chain_cli = here / "y_plug_chain.py"

    def cli(*dirs):
        done = subprocess.run([sys.executable, "-B", str(chain_cli), *map(str, dirs), "--controls", str(controls_path)], capture_output=True, text=True)
        try:
            out = json.loads(done.stdout)
        except ValueError:
            raise AssertionError(f"y_plug_chain.py printed no JSON: {(done.stdout + done.stderr)[-400:]}")
        return done.returncode, out
    for tag, child in (("YS794", "Y81_YS794_PLUG"), ("YS850", "Y81_YS850_PLUG")):
        with tempfile.TemporaryDirectory(prefix="am2020-ychain-") as tmp:
            tmp = Path(tmp)
            root = build_case(tmp)
            shutil.rmtree(root / "runs")
            done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py"), child], capture_output=True, text=True)
            assert done.returncode == 0, (done.stdout + done.stderr)[-600:]
            tpl = root / "runs" / child
            spec = json.loads((tpl / "child.json").read_text())
            mesh = PS.Mesh(tag, spec)
            facts = PS.facts_for(tag)
            m = PS.model_numbers(tag, "mesh", "p35_75", 1.0)               # physics 35's lumped model of the 75% plug (design rev. 2.2 A1), mesh film
            tau_a = facts["lam"] * facts["D0"] / m["V0"] + facts["tau_v"]
            k_end = int(math.ceil((m["t_c"] + 2.0 * tau_a + 0.004) / 1e-3))
            disp_at = PS.model_displacement(m)
            rate_at = PS.model_rate(m)
            r_zero = PS.plug_r(mesh, 0.0)
            join_k, join_dt = 30, 4.1e-4
            # FAIL 6 (rev. 2.2.2): the section S read from the case's own polyMesh, and the fluxes PLANTED on it (a correct STUCK: the plug's relative flux is the Darcy seepage f = 3.4e-3 Q0,
            # the skeleton moves at the model rate; no gap). The gap variant drains the -y wall row and lets a 0.02 Q0 flux through it from the restart point on.
            _stub = tmp / "stub"
            PS.place_polymesh(_stub, mesh)
            sec = PA.read_poly_section(_stub, mesh.D0, mesh.depth, mesh.xt * mesh.D0, mesh.x, mesh.y)

            def write_for(t, idx, gap=False):
                r = PS.plug_r(mesh, disp_at(t) / mesh.D0)
                if gap:
                    for cc in (sec["own"][0], sec["nei"][0]):
                        r[cc] = 0.0
                phi, sown = PS.planted_section(sec, sec["n_internal"], 3.4e-3 * facts["q0"], vs_ms=rate_at(t), wall_rows=8, vs_wall_ms=rate_at(t), gap={0: 0.02 * facts["q0"]} if gap else None)
                return {"t": t, "r": r, "index": idx, "phi": phi, "phi_skel_own": sown}
            w1 = [write_for(k * 1e-3, k) for k in range(1, join_k + 1)]
            t_join = join_k * 1e-3 + join_dt
            w1.append(write_for(t_join, join_k + 1))
            s1 = PS.write_session(tmp / "s1", tpl, mesh, w1, zero_r=r_zero)
            jname = PS.time_name(t_join)
            w2 = [dict(w1[-1])] + [write_for(k * 1e-3, k + 2) for k in range(join_k + 1, k_end + 1)]
            s2 = PS.write_session(tmp / "s2", tpl, mesh, w2, zero_r=r_zero, restart=PS.restart_record(s1, jname, w1[-1]["index"]))
            w2g = [dict(w1[-1])] + [write_for(k * 1e-3, k + 2, gap=True) for k in range(join_k + 1, k_end + 1)]
            s2g = PS.write_session(tmp / "s2_gapflux", tpl, mesh, w2g, zero_r=r_zero, restart=PS.restart_record(s1, jname, w1[-1]["index"]))
            # session 1 alone: not yet readable
            rc, out = cli(s1)
            assert rc == 3 and out["status"] == "NOT_READABLE_YET", (tag, "session 1 alone", rc, out.get("status"), out.get("why"))
            sess = PA.assess_case_dirs([s1], spec, params, texts=[""], labels=["s1"])
            assert sess["status"] == "NOT_READABLE_YET", (tag, sess["status"])
            # session 2 alone: assess_case says it needs the chain
            spec_ac = importlib.util.spec_from_file_location("assess_case_ychain", here / "assess_case.py")
            ac = importlib.util.module_from_spec(spec_ac)
            spec_ac.loader.exec_module(ac)
            frag = ac.y_plug_session(s2, spec, json.loads(controls_path.read_text()), "")
            assert frag["status"] == "CHAIN_NEEDED" and frag["verdict"] is None, (tag, frag)
            # the chain
            rc, out = cli(s1, s2)
            assert rc == 0 and out["status"] == "VERDICT" and out["verdict"] == "STUCK" and out["final"] and out["sessions"] == 2, (tag, rc, out.get("status"), out.get("verdict"), out.get("why"))
            # the STUCK law is laid on BOTH meshes; the YS794 chain is predicted STUCK (agrees), the YS850 chain SQUEEZE (a STUCK there DISAGREES with its prediction and the record says so)
            assert out["predicted"] == {"YS794": "STUCK", "YS850": "SQUEEZE"}[tag] and out["agrees_with_prediction"] is (tag == "YS794"), (tag, out.get("agrees_with_prediction"), out.get("predicted"))
            # FAIL 6 on the planted section: silent at every valid post-contact write (Q_np/Q0 = the planted wall-row fluxes only), and the reading names it
            f6 = out["fail"]["F6_bypass"]
            assert f6["fired"] is False and f6["evaluated"] > 40 and f6["invalid_writes"] == 0 and f6["max_bypass"] < 5e-4 and "F6_bypass" not in out["fired"], (tag, out["fired"], f6)       # (F3 fires on this fixture: a rigid shift does not conserve the content; the assessor check conserves it)
            assert out["film_branch"] == "mesh film" and out["record_complete"] is False, (tag, out.get("film_branch"), out.get("record_complete"))
            if tag == "YS794":
                assert out["reading"].startswith("STUCK as predicted; FAIL 6 silent at"), out["reading"]
            else:
                assert out["reading"].startswith("STUCK against the prediction SQUEEZE: FALSIFIES physics 19") and "Q_np <= 0.005 Q0" in out["reading"], out["reading"]
            # the SAME chain with a 0.02 Q0 flux through a drained wall row from the restart point on: FAIL 6 FIRES, the verdict does not change, the reading says the run is not a clean test
            rc, outg = cli(s1, s2g)
            assert rc == 0 and outg["verdict"] == "STUCK" and outg["fail"]["F6_bypass"]["fired"] is True and outg["fail"]["F6_bypass"]["at_t"] > t_join, (tag, rc, outg["fail"]["F6_bypass"])
            assert "F6_bypass" in outg["fired"] and "BUT FAIL 6 FIRED" in outg["reading"] and "as predicted" not in outg["reading"] and "FALSIFIES" not in outg["reading"], outg["reading"]
            assert 0.02 < outg["fail"]["F6_bypass"]["max_bypass"] < 0.026, outg["fail"]["F6_bypass"]
            res = PA.assess_case_dirs([s1, s2], spec, params)
            ts = res["series_columns"]["t"]
            assert all(abs((b - a) - 1e-3) < 1e-9 for a, b in zip(ts, ts[1:])) and ts[0] == 0.0 and len(ts) == len(set(ts)), (tag, ts[28:34])
            R = out["reads"][0]["R"]
            assert abs(R - math.exp(-tau_a / m["tau"])) < 0.05, (tag, R, math.exp(-tau_a / m["tau"]))
            # refusals
            shutil.copytree(s2, tmp / "s2_cold")
            (tmp / "s2_cold" / "restart.json").unlink()
            rc, out = cli(s1, tmp / "s2_cold")
            assert rc == 1 and out["status"] == "CHAIN_REFUSED" and "cold start" in out["why"], (tag, rc, out.get("status"), out.get("why"))
            shutil.copytree(s2, tmp / "s2_gap")
            shutil.rmtree(tmp / "s2_gap" / PS.time_name(0.034))
            rc, out = cli(s1, tmp / "s2_gap")
            assert rc == 1 and out["status"] == "CHAIN_REFUSED" and "gap" in out["why"], (tag, rc, out.get("status"), out.get("why"))
            rc, out = cli(s2, s1)
            assert rc == 1 and out["status"] == "CHAIN_REFUSED", (tag, rc, out.get("status"))
            print(f"   {child}: 2-session chain of the 75% plug ({len(w1)} + {len(w2)} writes, join at {t_join:.5f} s) -> exit 0, VERDICT STUCK final, R {R:.3f} (model {math.exp(-tau_a / m['tau']):.3f}), FAIL 6 silent at {f6['evaluated']} writes "
                  f"(a 0.02 Q0 drained-row gap fires it: Q_np/Q0 {outg['fail']['F6_bypass']['max_bypass']:.4f}); session 1 alone exit 3, session 2 alone CHAIN_NEEDED, a cold second session / a gap / swapped sessions exit 1 CHAIN_REFUSED")


def reference_library_wiring() -> None:
    """The reference-library arm (S22 arm A0 and S23 arm (1')), offline: the notebook's pins and
    batch table agree with the generator's solver_library children, the build script's two release
    pins are still unique, and the runner's start-of-session check refuses what child.env would
    refuse after a whole session (each case seen both ways)."""
    import ast
    import importlib.util
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("nbsel", root / "scripts" / "check_notebook_batch_selector.py")
    nbsel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nbsel)
    text = nbsel.source_of(root / "notebooks" / "AM2020_main_case_colab.ipynb")
    batch, batches = nbsel.parse(text)
    # BATCH 83 (the corrected restart retest) is the default, the newest entry; BATCH 81 (the plug chains) stays STAGED: built and checked, but never selectable as the default (check_notebook_batch_selector.STAGED)
    assert batch == 83 and max(batches) == 83 and nbsel.STAGED == (81,), (batch, max(batches), nbsel.STAGED)
    assert nbsel.check_default_batch(text) == (batch, batches)
    # the build script re-pins by exactly these two patterns; a second match would break the release
    assert len(re.findall(r"EXPECTED_REPO_DIGEST = '[0-9a-f]{64}'", text)) == 1
    assert len(re.findall(r"EXPECTED_TAG = 'digest-[0-9a-f]{12}'", text)) == 1
    pins = dict(re.findall(r"^(P2PRIME_TAG|P2PRIME_REPO_DIGEST|S22REF_TAG|S22REF_REPO_DIGEST|BASELINE_TAG|BASELINE_REPO_DIGEST) = '([^']+)'", text, re.M))
    assert set(pins) == {"P2PRIME_TAG", "P2PRIME_REPO_DIGEST", "S22REF_TAG", "S22REF_REPO_DIGEST", "BASELINE_TAG", "BASELINE_REPO_DIGEST"}, pins
    for t, d in (("P2PRIME_TAG", "P2PRIME_REPO_DIGEST"), ("S22REF_TAG", "S22REF_REPO_DIGEST"), ("BASELINE_TAG", "BASELINE_REPO_DIGEST")):
        assert re.fullmatch(r"[0-9a-f]{64}", pins[d]) and pins[t] == "digest-" + pins[d][:12], (t, pins[t], pins[d])
    assert pins["S22REF_TAG"] == "digest-442b95afaec8", "BATCH 72's reference is the S22 release (design section 7)"
    # BATCH 76 launch 1's reference is the release BEFORE S25, release-e2a57db1 (RESEARCH_STATE: tags digest-6758ccf99eb4 and release-e2a57db1); the release build re-pins EXPECTED_* only
    assert pins["BASELINE_TAG"] == "digest-6758ccf99eb4", "BATCH 76's baseline is release-e2a57db1 (S25 design section 7)"
    table = re.search(r"(?s)REFERENCE_LIBRARY_BY_BATCH = (\{.*?\n\})", text).group(1)
    assert "P2PRIME_TAG" in table and "S22REF_TAG" in table and "BASELINE_TAG" in table
    flags = dict(re.findall(r"(\d+): \('(\w+)', \w+, \w+, '(--[\w-]+)'", table) and
                 [(b, f"{n}|{fl}") for b, n, fl in re.findall(r"(\d+): \('(\w+)', \w+, \w+, '(--[\w-]+)'", table)])
    assert flags == {"71": "p2prime|--p2prime-solver", "72": "s22|--s22-solver", "76": "baseline|--baseline-solver", "78": "baseline|--baseline-solver"}, flags
    # the runner's functions, executed from source
    runner_src = (CASE / "scripts" / "run_main_case_colab.py").read_text()
    ns = {"json": json, "Path": Path}
    want = {"REFERENCE_LIBRARIES", "reference_libraries_needed", "check_reference_libraries"}
    for node in ast.parse(runner_src).body:
        name = getattr(node, "name", None) or (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in want:
            exec(compile(ast.Module([node], []), "runner", "exec"), ns)
    assert set(ns) >= want | {"json", "Path"}
    controls = CASE / "constant" / "mainCaseControls.json"
    need, check = ns["reference_libraries_needed"], ns["check_reference_libraries"]
    for b, (name, flag) in {71: ("p2prime", "--p2prime-solver"), 72: ("s22", "--s22-solver")}.items():
        assert need(batches[b], controls) == {name}, (b, need(batches[b], controls))
        assert ns["REFERENCE_LIBRARIES"][name][4] == flag.lstrip("-").replace("-", "_")
    # rev. 4.1: BATCH 69 selects em78_id, xu5_id and fsf_G4_kxu_nor3_40, which are pinned to the S22 library, so a
    # session of it needs --s22-solver (the runner refuses it at the START otherwise); BATCH 72's selection is
    # unchanged. The notebook's table gives the library to BATCH 71 and 72 only.
    assert need(batches[69], controls) == {"s22"}, need(batches[69], controls)
    # the generator's child.env library and the runner's variable are one name
    var_of = {n: v[0] for n, v in ns["REFERENCE_LIBRARIES"].items()}
    gsrc = (CASE / "scripts" / "create_cases.py").read_text()
    for n, v in var_of.items():
        assert f'"{n}": ("{v}",' in gsrc, (n, v)

    def refused(needed, given, why):
        try:
            check(needed, given)
        except RuntimeError as error:
            assert why in str(error), (why, str(error))
            return
        raise AssertionError(f"accepted: {needed} {given}")
    check(need(batches[72], controls), {"p2prime": "", "s22": "/x"})
    check(need(batches[71], controls), {"p2prime": "/x", "s22": ""})
    refused(need(batches[72], controls), {"p2prime": "", "s22": ""}, "--s22-solver was not passed")
    refused(need(batches[71], controls), {"p2prime": "", "s22": ""}, "--p2prime-solver was not passed")
    refused(need(batches[72], controls), {"p2prime": "/x", "s22": "/y"}, "built for nothing")
    refused(need(batches[71], controls), {"p2prime": "/x", "s22": "/y"}, "built for nothing")
    refused({"p3"}, {"p3": "/x"}, "cannot build")
    # BATCH 72 is TWO sessions (2026-10-08): launch 1 = (1)+(1') names solver_library s22 and clones the S22
    # library; launch 2 = (2)+(3) names none and must pass nothing (the runner refuses a library built for
    # nothing). The rule is the notebook's REFERENCE_LIBRARY_BY_LAUNCH, keyed by (batch, launch); here each
    # launch's children are read against the controls, and EACH launch's exact runner argv (built by the
    # notebook's own launch_plan / library_args / runner_argv, executed from its source) is parsed by the
    # runner's own argparse and accepted by its own start-of-session check.
    cell = nbsel.launch_cell(root / "notebooks" / "AM2020_main_case_colab.ipynb")
    records = nbsel.check_launches(cell, controls, 72)
    by = {(r["batch"], r["launch"]): r for r in records if r.get("variant", "cold") == "cold"}
    by_restart = {(r["batch"], r["launch"]): r for r in records if r.get("variant") == "restart"}
    assert {(71, 1), (72, 1), (72, 2)} <= set(by), sorted(by)
    assert by[(72, 1)]["children"] == ["S23v_G4_r3", "S23v_G4_r3_S22"], by[(72, 1)]["children"]
    assert by[(72, 2)]["children"] == ["S23v_G4_nor3", "S23v_G40"], by[(72, 2)]["children"]
    assert need(" ".join(by[(72, 1)]["children"]), controls) == {"s22"}
    assert need(" ".join(by[(72, 2)]["children"]), controls) == set(), need(" ".join(by[(72, 2)]["children"]), controls)
    assert by[(72, 1)]["library_passed"] == ["s22"] and by[(72, 2)]["library_passed"] == [], (by[(72, 1)], by[(72, 2)])
    assert by[(71, 1)]["library_passed"] == ["p2prime"], by[(71, 1)]
    # BATCH 73 is ONE session (no launch 2): it names no solver_library, so it clones nothing and passes no flag
    assert [k for k in by if k[0] == 73] == [(73, 1)], sorted(by)
    assert by[(73, 1)]["children"] == ["P73_G4_r3_40", "P73_G40_40"], by[(73, 1)]["children"]
    assert need(" ".join(by[(73, 1)]["children"]), controls) == set(), need(" ".join(by[(73, 1)]["children"]), controls)
    assert by[(73, 1)]["library_passed"] == [] and "--s22-solver" not in by[(73, 1)]["argv"] and "--p2prime-solver" not in by[(73, 1)]["argv"]
    refused(need(" ".join(by[(73, 1)]["children"]), controls), {"p2prime": "", "s22": "/x"}, "built for nothing")
    check(need(" ".join(by[(73, 1)]["children"]), controls), {"p2prime": "", "s22": ""})
    # BATCH 74 (D74) is ONE session of ONE child (no launch 2): it names no solver_library, so it clones nothing and passes no flag
    assert [k for k in by if k[0] == 74] == [(74, 1)], sorted(by)
    assert by[(74, 1)]["children"] == ["D74_refI"], by[(74, 1)]["children"]
    assert need("D74_refI", controls) == set(), need("D74_refI", controls)
    assert by[(74, 1)]["library_passed"] == [] and "--s22-solver" not in by[(74, 1)]["argv"] and "--p2prime-solver" not in by[(74, 1)]["argv"]
    refused(need("D74_refI", controls), {"p2prime": "", "s22": "/x"}, "built for nothing")
    check(need("D74_refI", controls), {"p2prime": "", "s22": ""})
    # BATCH 75 (D75) is likewise ONE session of ONE child: no solver_library, nothing cloned, no flag passed
    assert [k for k in by if k[0] == 75] == [(75, 1)], sorted(by)
    assert by[(75, 1)]["children"] == ["D75_refI"], by[(75, 1)]["children"]
    assert need("D75_refI", controls) == set(), need("D75_refI", controls)
    assert by[(75, 1)]["library_passed"] == [] and "--s22-solver" not in by[(75, 1)]["argv"] and "--p2prime-solver" not in by[(75, 1)]["argv"]
    refused(need("D75_refI", controls), {"p2prime": "", "s22": "/x"}, "built for nothing")
    check(need("D75_refI", controls), {"p2prime": "", "s22": ""})
    # BATCH 76 (S25 verification) is THREE sessions on the 1-core runtime: only launch 1 (B76_OFF_A, solver_library baseline) clones the baseline release and passes --baseline-solver;
    # none passes --mkl-cbwr (design section 12.1: no CNR pin), each passes --mkl-threads 1
    assert [k for k in by if k[0] == 76] == [(76, 1), (76, 2), (76, 3)], sorted(by)
    assert by[(76, 1)]["children"] == ["B76_OFF_A"] and by[(76, 2)]["children"] == ["B76_OFF_B"], (by[(76, 1)]["children"], by[(76, 2)]["children"])
    assert by[(76, 3)]["children"] == ["RT_P73_A", "RT_P73_B", "RT_5a_A", "RT_5a_B", "S25_5a_tan", "S25_5a_tan_half"], by[(76, 3)]["children"]
    assert need("B76_OFF_A", controls) == {"baseline"} and need("B76_OFF_B", controls) == set() and need(" ".join(by[(76, 3)]["children"]), controls) == set()
    assert [by[(76, n)]["library_passed"] for n in (1, 2, 3)] == [["baseline"], [], []]
    for n in (1, 2, 3):
        a = by[(76, n)]["argv"]
        assert "--mkl-cbwr" not in a and a[a.index("--mkl-threads") + 1] == "1", (n, a)
        assert ("--baseline-solver" in a) == (n == 1) and "--s22-solver" not in a and "--p2prime-solver" not in a, (n, a)
    refused(need("B76_OFF_A", controls), {"p2prime": "", "s22": "", "baseline": ""}, "--baseline-solver was not passed")
    refused(need("B76_OFF_B", controls), {"p2prime": "", "s22": "", "baseline": "/x"}, "built for nothing")
    check(need("B76_OFF_A", controls), {"p2prime": "", "s22": "", "baseline": "/x"})
    # BATCH 78 (S25.1 re-check) is ONE session on the 1-core runtime (no launch 2): B78_OFF_A names solver_library baseline, so the one launch clones the baseline release and passes --baseline-solver;
    # it passes --mkl-threads 1 and no --mkl-cbwr; its children are the notebook's session in the generator's matrix order (both arms of every pair in it)
    assert [k for k in by if k[0] == 78] == [(78, 1)], sorted(by)
    assert by[(78, 1)]["children"] == ["B78_OFF_A", "B78_OFF_B", "B78_CTL_1", "B78_CTL_2", "B78_RT_5a_A", "B78_RT_5a_B"], by[(78, 1)]["children"]
    assert need(" ".join(by[(78, 1)]["children"]), controls) == {"baseline"} and need("B78_OFF_B B78_CTL_1 B78_CTL_2 B78_RT_5a_A B78_RT_5a_B", controls) == set()
    assert by[(78, 1)]["library_passed"] == ["baseline"]
    a78 = by[(78, 1)]["argv"]
    assert "--mkl-cbwr" not in a78 and a78[a78.index("--mkl-threads") + 1] == "1" and "--baseline-solver" in a78 and "--s22-solver" not in a78 and "--p2prime-solver" not in a78, a78
    refused(need(" ".join(by[(78, 1)]["children"]), controls), {"p2prime": "", "s22": "", "baseline": ""}, "--baseline-solver was not passed")
    check(need(" ".join(by[(78, 1)]["children"]), controls), {"p2prime": "", "s22": "", "baseline": "/x"})
    # BATCH 79 (the Y-bifurcation gate) is ONE session, ONE child, on the 1-core runtime, and clones NO reference library (it runs on the package's S25.1 build): --mkl-threads 1, no --mkl-cbwr, no library flag
    assert [k for k in by if k[0] == 79] == [(79, 1)], sorted(by)
    assert by[(79, 1)]["children"] == ["B79_Y_GATE"], by[(79, 1)]["children"]
    assert need("B79_Y_GATE", controls) == set() and by[(79, 1)]["library_passed"] == []
    a79 = by[(79, 1)]["argv"]
    assert "--mkl-cbwr" not in a79 and a79[a79.index("--mkl-threads") + 1] == "1" and not any(f in a79 for f in ("--baseline-solver", "--s22-solver", "--p2prime-solver")), a79
    refused(need("B79_Y_GATE", controls), {"p2prime": "", "s22": "", "baseline": "/x"}, "built for nothing")
    # BATCH 80 (the symmetric Y: two flow gates and the restart pair) is ONE session of 4 children (launch 1) and a SECOND runner invocation (launch 2: Y80_RT_B's leg 2 alone, restarted
    # from launch 1's tarball). Both are on the 1-core runtime and clone NO reference library: --mkl-threads 1, no --mkl-cbwr, no library flag. Y80_RT_B is the one child in two launches by
    # design (RESTART_LAUNCHES); the matrix order of launch 1 is the run order (A before B).
    assert [k for k in by if k[0] == 80] == [(80, 1), (80, 2)], sorted(by)
    assert by[(80, 1)]["children"] == ["Y80_YS794_FLOW", "Y80_YS850_FLOW", "Y80_RT_A", "Y80_RT_B"], by[(80, 1)]["children"]
    assert by[(80, 2)]["children"] == ["Y80_RT_B"], by[(80, 2)]["children"]
    for _n in (1, 2):
        assert need(" ".join(by[(80, _n)]["children"]), controls) == set() and by[(80, _n)]["library_passed"] == [], _n
        _a80 = by[(80, _n)]["argv"]
        assert "--mkl-cbwr" not in _a80 and _a80[_a80.index("--mkl-threads") + 1] == "1" and not any(f in _a80 for f in ("--baseline-solver", "--s22-solver", "--p2prime-solver")), (_n, _a80)
        assert ("Y80_RT_B" in _a80[_a80.index("--only") + 1].split()) and (_n == 1 or _a80[_a80.index("--only") + 1] == "Y80_RT_B"), (_n, _a80)
    assert any("restarted from" in _l and "--restart-from" in _l for _l in by[(80, 2)]["printed"]) and not any("restarted from" in _l for _l in by[(80, 1)]["printed"]), "launch 2 is the restart, launch 1 a cold start"
    refused(need("Y80_RT_B", controls), {"p2prime": "", "s22": "", "baseline": "/x"}, "built for nothing")
    check(need("Y80_RT_B", controls), {"p2prime": "", "s22": "", "baseline": ""})
    # BATCH 82 (the confirming restart test, case-engineer 33): the same two-launch shape as BATCH 80 for ONE child. Launch 1 is a cold start (leg 1 stops after step 86), launch 2 a SECOND runner invocation
    # restarted from launch 1's tarball (leg 2, stops after step 105); both on the 1-core runtime, no reference library, --mkl-threads 1, no --mkl-cbwr; Y80_RT86_B is the one child in two launches by design (RESTART_LAUNCHES)
    assert [k for k in by if k[0] == 82] == [(82, 1), (82, 2)], sorted(by)
    assert by[(82, 1)]["children"] == ["Y80_RT86_B"] and by[(82, 2)]["children"] == ["Y80_RT86_B"], (by[(82, 1)]["children"], by[(82, 2)]["children"])
    for _n in (1, 2):
        assert need("Y80_RT86_B", controls) == set() and by[(82, _n)]["library_passed"] == [], _n
        _a82 = by[(82, _n)]["argv"]
        assert "--mkl-cbwr" not in _a82 and _a82[_a82.index("--mkl-threads") + 1] == "1" and not any(f in _a82 for f in ("--baseline-solver", "--s22-solver", "--p2prime-solver")), (_n, _a82)
        assert _a82[_a82.index("--only") + 1] == "Y80_RT86_B", (_n, _a82)
    assert any("restarted from" in _l and "--restart-from" in _l for _l in by[(82, 2)]["printed"]) and not any("restarted from" in _l for _l in by[(82, 1)]["printed"]), "BATCH 82 launch 2 is the restart, launch 1 a cold start"
    assert any("stops after step 86" in _l for _l in by[(82, 1)]["printed"]) and any("stops after step 105" in _l for _l in by[(82, 2)]["printed"]), "BATCH 82 prints the two stops"
    refused(need("Y80_RT86_B", controls), {"p2prime": "", "s22": "", "baseline": "/x"}, "built for nothing")
    check(need("Y80_RT86_B", controls), {"p2prime": "", "s22": "", "baseline": ""})
    # BATCH 83 (the corrected restart retest, case-engineer 36): launch 1 = the continuous arm Y80_RT_A AND Y80_RT86_B's leg 1 in ONE cold session (the matrix order: Y80_RT86_B first), launch 2 = Y80_RT86_B's leg 2 (a second
    # runner invocation restarted from launch 1's tarball); 1-core runtime, no reference library, --mkl-threads 1, no --mkl-cbwr. Y80_RT86_B is the one child in two launches by design (RESTART_LAUNCHES).
    assert [k for k in by if k[0] == 83] == [(83, 1), (83, 2)], sorted(by)
    assert by[(83, 1)]["children"] == ["Y80_RT86_B", "Y80_RT_A"] and by[(83, 2)]["children"] == ["Y80_RT86_B"], (by[(83, 1)]["children"], by[(83, 2)]["children"])
    for _n in (1, 2):
        assert need(" ".join(by[(83, _n)]["children"]), controls) == set() and by[(83, _n)]["library_passed"] == [], _n
        _a83 = by[(83, _n)]["argv"]
        assert "--mkl-cbwr" not in _a83 and _a83[_a83.index("--mkl-threads") + 1] == "1" and not any(f in _a83 for f in ("--baseline-solver", "--s22-solver", "--p2prime-solver")), (_n, _a83)
        assert _a83[_a83.index("--only") + 1].split() == by[(83, _n)]["children"], (_n, _a83)
    assert any("restarted from" in _l and "--restart-from" in _l for _l in by[(83, 2)]["printed"]) and not any("restarted from" in _l for _l in by[(83, 1)]["printed"]), "BATCH 83 launch 2 is the restart, launch 1 a cold start"
    assert any("stops after step 86" in _l and "ONE session" in _l for _l in by[(83, 1)]["printed"]) and any("stops after step 105" in _l and "--run-b1" in _l for _l in by[(83, 2)]["printed"]), "BATCH 83 prints the two stops and the three-session scoring"
    refused(need("Y80_RT86_B Y80_RT_A", controls), {"p2prime": "", "s22": "", "baseline": "/x"}, "built for nothing")
    check(need("Y80_RT86_B Y80_RT_A", controls), {"p2prime": "", "s22": "", "baseline": ""})
    # BATCH 81 (the plug chains, STAGED): TWO launches, one chain each, on the 1-core runtime, no reference library. Each launch is built twice: cold (session 1, RESTART_FROM empty) and as a later session of its chain
    # (RESTART_FROM = the previous session's tarball): the restarted argv carries --restart-from and says RESTARTED, the cold one neither; both pass --mkl-threads 1 and no --mkl-cbwr.
    assert [k for k in by if k[0] == 81] == [(81, 1), (81, 2)] and sorted(k for k in by_restart if k[0] == 81) == [(81, 1), (81, 2)], (sorted(by), sorted(by_restart))
    assert by[(81, 1)]["children"] == ["Y81_YS794_PLUG"] and by[(81, 2)]["children"] == ["Y81_YS850_PLUG"], (by[(81, 1)]["children"], by[(81, 2)]["children"])
    for _n in (1, 2):
        assert need(" ".join(by[(81, _n)]["children"]), controls) == set() and by[(81, _n)]["library_passed"] == [] and by_restart[(81, _n)]["library_passed"] == [], _n
        for _rec, _restarted in ((by[(81, _n)], False), (by_restart[(81, _n)], True)):
            _a81 = _rec["argv"]
            assert "--mkl-cbwr" not in _a81 and _a81[_a81.index("--mkl-threads") + 1] == "1" and not any(f in _a81 for f in ("--baseline-solver", "--s22-solver", "--p2prime-solver")), (_n, _a81)
            assert _a81[_a81.index("--only") + 1] == by[(81, _n)]["children"][0], (_n, _a81)
            assert ("--restart-from" in _a81) == _restarted, (_n, _restarted, _a81)
            assert any(("RESTARTED" in _l) == _restarted and ("COLD" in _l) != _restarted for _l in _rec["printed"] if "session" in _l and "BATCH 81" in _l), (_n, _restarted, _rec["printed"])
    refused(need("Y81_YS794_PLUG", controls), {"p2prime": "", "s22": "", "baseline": "/x"}, "built for nothing")
    check(need("Y81_YS850_PLUG", controls), {"p2prime": "", "s22": "", "baseline": ""})
    assert "--s22-solver" in by[(72, 1)]["argv"] and "--s22-solver" not in by[(72, 2)]["argv"]
    # the same two negatives on this function's own exec'd runner functions
    check(need(" ".join(by[(72, 1)]["children"]), controls), {"p2prime": "", "s22": "/x"})
    check(need(" ".join(by[(72, 2)]["children"]), controls), {"p2prime": "", "s22": ""})
    refused(need(" ".join(by[(72, 2)]["children"]), controls), {"p2prime": "", "s22": "/x"}, "built for nothing")
    refused(need(" ".join(by[(72, 1)]["children"]), controls), {"p2prime": "", "s22": ""}, "--s22-solver was not passed")
    seen = nbsel.negatives(cell, controls, 72)
    print("   the reference-library wiring agrees: notebook pins, batch table, generator children, runner flags; "
          f"7 start-of-session cases seen both ways; {len(records)} launch argvs accepted by the runner's own parser "
          f"(BATCH 72 launch 1 passes --s22-solver, launch 2 does not; BATCH 73, BATCH 74 and BATCH 75, one session each, pass none; BATCH 76 launch 1 passes --baseline-solver, launches 2 and 3 do not, BATCH 78 (one session) passes --baseline-solver, and none passes --mkl-cbwr); {len(seen)} launch negatives each refused "
          "for its stated reason")


# ---- XU 2017 FIGURE 5 (batch 61; design 2026-09-29-case-xu2017-fig5 rev. 4) ----
#
# Module-level, so the docket probe (research/investigations/20260929-1430-
# case-xu2017-.../probes/case-engineer-07-xu-dry.py) runs EXACTLY these
# functions rather than a copy of them.
XU_REFUSALS = (
    # (label, mutate(controls), filter, the refusal's own words)
    ("pressure_driven true on the Xu series",
     lambda c: c["series"]["xu2017_fig5"].__setitem__("pressure_driven", True),
     ["xu2017_fig5"], "must declare pressure_driven false"),
    ("pressure_driven omitted (inheritance)",
     lambda c: c["series"]["xu2017_fig5"].pop("pressure_driven"),
     ["xu2017_fig5"], "must declare pressure_driven false"),
    ("a missing per-child velocity (xu5_g500)",
     lambda c: _xu_child(c, "xu5_g500").pop("mean_velocity_m_s"),
     ["xu2017_fig5"], "so each child must declare its mean_velocity_m_s"),
    ("250 1/s not on the series axis",
     lambda c: _xu_drop_rate(c, 250.0), ["xu2017_fig5"],
     "xu5_g250: wall shear rate 250.0 1/s is not on series 'xu2017_fig5''s flow_axis"),
    ("500 1/s not on the series axis",
     lambda c: _xu_drop_rate(c, 500.0), ["xu2017_fig5"],
     "xu5_g500: wall shear rate 500.0 1/s is not on series 'xu2017_fig5''s flow_axis"),
    ("2000 1/s not on the series axis",
     lambda c: _xu_drop_rate(c, 2000.0), ["xu2017_fig5"],
     "xu5_g2000: wall shear rate 2000.0 1/s is not on series 'xu2017_fig5''s flow_axis"),
    ("a child velocity off the axis value",
     lambda c: _xu_child(c, "xu5_g500").__setitem__("mean_velocity_m_s", 4.2e-3),
     ["xu2017_fig5"], "is not the flow_axis value"),
    ("an axis velocity that is not gammaDot H / 6",
     lambda c: (c["series"]["xu2017_fig5"]["flow_axis"]["mean_velocity_m_s"].__setitem__(1, 5e-3),
                _xu_child(c, "xu5_g500").__setitem__("mean_velocity_m_s", 5e-3)),
     ["xu2017_fig5"], "xu5_g500: U_mean 0.004166666666666667 does not match the declared 0.005"),
    ("a series-level velocity beside the axis",
     lambda c: c["series"]["xu2017_fig5"]["geometry_override"].__setitem__("mean_velocity_m_s", 8.333e-3),
     ["xu2017_fig5"], "declares a flow_axis AND a series-level mean_velocity_m_s"),
    ("releaseCoreExempt on",
     lambda c: _xu_child(c, "xu5_g250").__setitem__("release_core_exempt", True),
     ["xu2017_fig5"], "release_core_exempt must be declared false"),
    ("growth kinetics left on",
     lambda c: _xu_child(c, "xu5_g250").pop("platelet_transport_only"),
     ["xu2017_fig5"], "platelet_transport_only must be true"),
    ("platelet inflow left on",
     lambda c: _xu_child(c, "xu5_g250").__setitem__("platelet_inlet_density_m3", 2.5e14),
     ["xu2017_fig5"], "platelet_inlet_density_m3 must be declared 0.0"),
    ("growth on",
     lambda c: c["series"]["xu2017_fig5"].__setitem__("growth", True),
     ["xu2017_fig5"], "growth must be declared false"),
    ("the label width left to its default",
     lambda c: c["series"]["xu2017_fig5"]["clot_override"].pop("material_width_m"),
     ["xu2017_fig5"], "material_width_m must be declared"),
    ("half a plasma",
     lambda c: c["series"]["xu2017_fig5"].pop("nu_m2_s"),
     ["xu2017_fig5"], "series xu2017_fig5: declares rho_kg_m3 without its pair"),
    ("the held layer tapered",
     lambda c: _xu_child(c, "xu5_g250").__setitem__("bond_hold_taper", True),
     # S21: bondHoldTaper is built in (no); the key is refused by name.
     ["xu2017_fig5"], "names the built-in option bondHoldTaper"),
    ("an inlet pressure on the velocity series",
     lambda c: c["series"]["xu2017_fig5"].__setitem__("inlet_pressure_m2_s2", 0.3),
     ["xu2017_fig5"], "inlet_pressure_m2_s2 is read only by a pressure-driven inlet"),
    ("a solver switch drifting from em76",
     lambda c: _xu_child(c, "xu5_g1000").__setitem__("residual_chi", 1e-3),
     ["xu2017_fig5"], "Xu parent check: xu5_g1000 differs from em76_G4_s18 in residual_chi"),
    ("a sibling drifting in a non-rate key",
     lambda c: _xu_child(c, "xu5_g2000").__setitem__("end_time_s", 0.005),
     ["xu2017_fig5"], "Xu parent check: xu5_g2000 differs from xu5_g1000 in end_time_s"),
    ("a core too tall to be buried",
     lambda c: c["series"]["xu2017_fig5"]["clot_override"].__setitem__("core_height_m", 3.0e-5),
     ["xu2017_fig5"], "the core's nearest point is"),
    ("a dome that closes the lumen floor",
     lambda c: c["series"]["xu2017_fig5"]["clot_override"].__setitem__("height_m", 4.3e-5),
     ["xu2017_fig5"], "the lumen above the dome is"),
    ("a per-child velocity where the series has no axis",
     lambda c: _xu_child(c, "em76_G4_s18").__setitem__("mean_velocity_m_s", 8.333e-3),
     ["em76_G4_s18"], "mean_velocity_m_s is read only when the series declares its own flow_axis"),
)


def _xu_child(c: dict, cid: str) -> dict:
    hits = [x for x in c["children"] if x["child_id"] == cid]
    if len(hits) != 1:
        raise AssertionError(f"Xu negative names {cid!r}: {len(hits)} match")
    return hits[0]


def _xu_drop_rate(c: dict, rate: float) -> None:
    ax = c["series"]["xu2017_fig5"]["flow_axis"]
    k = ax["wall_shear_rate_1_s"].index(rate)
    ax["wall_shear_rate_1_s"].pop(k)
    ax["mean_velocity_m_s"].pop(k)


def xu_generator_negatives() -> None:
    """Each Xu refusal, seen to fire for its stated reason on a fresh copy."""
    for label, mutate, filt, expected in XU_REFUSALS:
        with tempfile.TemporaryDirectory(prefix="am2020-xu-neg-") as tmp:
            root = build_case(Path(tmp))
            shutil.rmtree(root / "runs")
            cp = root / "constant" / "mainCaseControls.json"
            c = json.loads(cp.read_text())
            mutate(c)
            cp.write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n")
            done = subprocess.run(
                [sys.executable, "-B", str(root / "scripts" / "create_cases.py")] + filt,
                capture_output=True, text=True)
            last = ((done.stdout + done.stderr).strip().splitlines() or [""])[-1]
            if done.returncode == 0:
                raise AssertionError(f"{label}: ACCEPTED; it must be refused")
            if expected not in last:
                raise AssertionError(f"{label}: refused for the WRONG reason: {last}")
            print(f"   refused: {label} -> {last[:160]}")


XU_SHIM = """
import sys
from pathlib import Path
src = Path(sys.argv[1])
ns = {"__name__": "initialise_cases_shim", "__file__": str(src)}
exec(compile(src.read_text(), str(src), "exec"), ns)
_read = ns["read_vectors"]
ns["read_vectors"] = lambda path, n: _read(path.with_name("Ccentres") if path.name == "C" else path, n)
sys.argv = [str(src), sys.argv[2]]
raise SystemExit(ns["main"]())
"""


def _xu_load_assessor():
    import importlib.util
    p = Path(__file__).resolve().parent / "assess_case.py"
    spec = importlib.util.spec_from_file_location("assess_case_xu_dry", p)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(p.read_text(), str(p), "exec"), mod.__dict__)
    return mod


def xu_fixture(tmp: Path) -> tuple:
    """A HEALTHY synthetic xu5_g250 on the real 29 400-cell mesh.

    0/ is the REAL initial state: create_cases.py, then the real initialiser
    and content seeding. (The cell centres are written as 0/Ccentres and read
    through a one-call shim, because this Mac's filesystem is case-insensitive
    and 0/C would overwrite 0/c; the seeding needs 0/c, after which 0/C is
    written for the assessor.) The time directories carry a static clot with
    the design's predicted behaviour: shell damage d* 0.05 from 6.0 ms, no
    bulk-core damage, bulk lambda1 1 + 0.044 t/ms, the core zone displaced
    0.372 um/ms, a 36-row wall tail past 131.9 um reaching +2.6 um at 2.5 ms
    and +20.3 um at 10 ms (the P6 marks), a
    parabolic flow with the clot's top rows at 1.15 x. The log carries every
    line the Xu rows read, healthy, for 3840 steps at the dt cap."""
    root = build_case(tmp)
    case = root / "runs" / "xu5_g250"
    spec = json.loads((case / "child.json").read_text())
    g = spec["geometry"]
    nx, ny, h, H = g["nx"], g["ny"], g["h_m"], g["channel_height_m"]
    n = nx*ny
    xs = [((k % nx) + 0.5)*h for k in range(n)]
    ys = [((k // nx) + 0.5)*h for k in range(n)]
    cent = "\n".join(f"({x!r} {y!r} 0)" for x, y in zip(xs, ys))
    head = "FoamFile\n{\n    format ascii;\n    class volVectorField;\n    object C;\n}\n" \
           "dimensions [0 1 0 0 0 0 0];\n"
    (case / "0" / "Ccentres").write_text(
        head + f"internalField nonuniform List<vector>\n{n}\n(\n{cent}\n)\n;\n")
    mural = root.parent / "AM2020_mural_geometry_gate" / "scripts" / "initialise_cases.py"
    for argv in ([sys.executable, "-B", "-c", XU_SHIM, str(mural), str(case)],
                 [sys.executable, "-B", str(root / "scripts" / "seed_stage2_content.py"), str(case)]):
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(f"Xu fixture initialisation failed: {r.stdout}{r.stderr}")
    (case / "0" / "c").unlink()
    (case / "0" / "Ccentres").rename(case / "0" / "C")

    def rd(name):
        t = (case / "0" / name).read_text()
        a = t.index("(", t.index("internalField"))
        return [float(v) for v in t[a + 1:t.index(")", a)].split()]
    tC0, tH0, sh0 = rd("thetaC"), rd("thetaH"), rd("shell")
    th0 = [a + b for a, b in zip(tC0, tH0)]
    G, Gc, U = spec["G_pa"], spec["G_core_pa"], spec["mean_velocity_m_s"]
    held_top = (spec["bond_hold_dilation"] + 1)*h
    for t_ms in (0.25, 0.5, 1.0, 2.5, 2.75, 3.5, 5.0, 6.0, 7.5, 10.0):
        d = case / f"{t_ms*1e-3:.10g}"
        d.mkdir()
        # the design's P6 marks: +2.6 um at 2.5 ms, +20.3 um at 10 ms (250 1/s)
        grow = 2.6*t_ms/2.5 if t_ms <= 2.5 else 2.6 + (20.3 - 2.6)*(t_ms - 2.5)/7.5
        tail = 131.9e-6 + grow*1e-6
        th, lab = list(th0), list(sh0)
        for k in range(n):
            if k // nx < 36 and xs[k] <= tail and th[k] < 0.405:
                th[k], lab[k] = 0.405, 1.0
        ind = [v/0.45 for v in th]
        ds = [0.05 if (t_ms >= 6.0 and ind[k] >= 1e-2 and lab[k] >= 0.5) else 0.0 for k in range(n)]
        gf = [Gc*math.exp(s*math.log(G/Gc)) for s in lab]
        write_scalar(d / "thetaC", [v*(1 - s) for v, s in zip(th, lab)])
        write_scalar(d / "thetaH", [v*s for v, s in zip(th, lab)])
        write_scalar(d / "thetaIntact", [v*(1 - q) for v, q in zip(th, ds)])
        write_scalar(d / "thetaPending", [v*q for v, q in zip(th, ds)])
        write_scalar(d / "thetaBroken", [0.0]*n)
        write_scalar(d / "thetaNetRef", th0)
        write_scalar(d / "shell", lab)
        write_scalar(d / "Gfield", gf)
        write_scalar(d / "released", [0.0]*n)
        write_vector(d / "Us", [(0.372e-6*t_ms, 0.0, 0.0)]*n)
        prof = [6*U*y*(H - y)/H**2 for y in ys]
        for k in range(n):
            if k // nx >= ny - 2 and 50e-6 < xs[k] < 135e-6:
                prof[k] *= 1.15
        write_vector(d / "U", [(u, 0.0, 0.0) for u in prof])
        if t_ms in (6.0, 10.0):
            lam = 1.0 + 0.044*t_ms
            bulk = [ind[k] >= 1e-2 and gf[k] >= 20.0 and ys[k] - held_top >= 2e-6 for k in range(n)]
            be = [(lam*lam if b else 1.0, 0.0, 0.0, 1.0/(lam*lam) if b else 1.0, 0.0, 1.0)
                  for b in bulk]
            (d / "be").write_text(
                "FoamFile\n{\n    format ascii;\n    class volSymmTensorField;\n    object be;\n}\n"
                "dimensions [0 0 0 0 0 0 0];\n"
                f"internalField nonuniform List<symmTensor>\n{n}\n(\n"
                + "\n".join("(" + " ".join(repr(v) for v in b) + ")" for b in be) + "\n)\n;\n")
    dt = 2.5e-4/96
    # S21: the solver's own CONFIG lines (fluxConvention, frozenPendingPressure,
    # wallPatchSplit and cplusOrder are built in and no longer printed).
    lines = ["PHASE_CLOT_FREEPLATELET_CONFIG transportOnly=1\n",
             "PHASE_CLOT_S18_CONFIG networkPressureWeight=contentRatio\n"]
    for k in range(1, 3841):
        t = k*dt
        lines.append(f"Time = {t:.12g}\n")
        lines.append(f"deltaT = {dt:.12g}\n")
        lines.append(f"PHASE_CLOT_RELEASED time={t:.12g} cells=0 volume=0\n")
        lines.append(f"PHASE_CLOT_CONTENT_BALANCE time={t:.12g} contentVolume=1e-15 contentDVdt=0 "
                     f"contentOuterFlux=0 contentTransferRate=0 contentRemoved=0 "
                     f"detachedContentRate=0 contentResidual=0 thetaRef=0.45\n")
        lines.append(f"PHASE_CLOT_DEBOND time={t:.12g} newFaces=0 debondedFaces=0 bondedClotFaces=113\n")
        lines.append(f"PHASE_CLOT_BONDHOLD time={t:.12g} seeds=113 heldCells=600 heldWeight=600 "
                     f"taper=0 dilation=4 penalty=10000000 carrier=1\n")
        lines.append(f"PHASE_CLOT_DIAGNOSTIC time={t:.12g} inventory=1 outletMeanUx={U!r} outletCells=70\n")
        lines.append(f"PHASE_CLOT_VS_SPLIT time={t:.12g} slip=0.001\n")
        lines.append(f"PHASE_CLOT_S17 time={t:.12g} nuNetMax=0.01\n")
    lines.append(f"ExecutionTime = 15000 s  ClockTime = {4.3*3840:.0f} s\n\nEnd\n")
    text = "".join(lines)
    healthy = {"s18": {"conformance_ok": True, "config_matches_child": True, "conformance": {}},
               "s17b": {"no_fibre_invalid_fatal": True, "slack_dd": True, "taut_le_pending": True,
                        "w2": True, "p2": True, "startup_alternation": {"passes": True}},
               "s17": {"g_proj": True, "g_rot": True, "g_eig": True, "g_uni": True, "s17split": True},
               "s16": {"g12r": True, "s16w": True, "s16m": True},
               "s14": {"g13_elastic_number_bounded": True}}
    return case, spec, text, healthy


def xu_gates() -> None:
    """The Xu report on a healthy synthetic run: every row PASS. Then one
    negative per row, each seen to FAIL that row (and the structural flag
    where the row is structural)."""
    ac = _xu_load_assessor()
    with tempfile.TemporaryDirectory(prefix="am2020-xu-gates-") as tmp:
        case, spec, text, res = xu_fixture(Path(tmp))
        r = ac.xu2017_report(case, text, spec, res)
        v = {k: w.get("verdict") for k, w in r["rows"].items() if isinstance(w, dict) and "verdict" in w}
        want_pass = ("A0", "A1", "A2", "A3", "A4", "A6", "A7", "P0", "P1", "N1", "N2", "N3",
                     "N5", "N6", "N7", "P3", "P4_N4", "P5", "P6", "P8", "P9")
        bad = {k: v.get(k) for k in want_pass if v.get(k) != "PASS"}
        assert not bad and r["structural_ok"] is True, (bad, {k: r["rows"][k] for k in bad})
        assert v["A5"] == "NOT-EVALUABLE", v["A5"]
        print(f"   healthy: {len(want_pass)} rows PASS; A5 NOT-EVALUABLE (offline); "
              f"P1 {r['rows']['P1']['ratio']:.4f}, D {r['rows']['P5']['D_um_at_mark']:.3f} um")

        def expect(label, row, t=text, results=res, structural=None):
            rr = ac.xu2017_report(case, t, spec, results)
            got = rr["rows"][row]["verdict"]
            assert str(got).startswith("FAIL"), f"{label}: {row} {got} -- the row cannot see it"
            if structural is not None:
                assert rr["structural_ok"] is structural, f"{label}: structural_ok {rr['structural_ok']}"
            print(f"   {label}: {row} {got}")

        def with_file(path, edit, label, row, structural=None):
            old = path.read_bytes()
            try:
                path.write_text(edit(old.decode()))
                expect(label, row, structural=structural)
            finally:
                path.write_bytes(old)

        first = sorted((p for p in case.iterdir() if p.is_dir() and p.name[0].isdigit()
                        and float(p.name) > 0), key=lambda p: float(p.name))
        w_first, w5, w10 = first[0], next(p for p in first if abs(float(p.name) - 5e-3) < 1e-9), first[-1]
        with_file(case / "0" / "U", lambda s: s.replace("(0.00208333333333 0 0)", "(0.00833333333333 0 0)"),
                  "A0: the inlet at the 1000 1/s velocity", "A0", structural=False)
        with_file(case / "constant" / "phaseClotProperties",
                  lambda s: s.replace("releaseCoreExempt no;", "releaseCoreExempt yes;"),
                  "A0: releaseCoreExempt on in the dictionary", "A0", structural=False)
        with_file(case / "constant" / "phaseClotProperties",
                  lambda s: s.replace("epsilonC [0 1 0 0 0 0 0] 3e-06;", "epsilonC [0 1 0 0 0 0 0] 2.9e-06;"),
                  "A0: epsilonC 2.9e-6 in the dictionary", "A0", structural=False)
        def nudge(s):
            a = s.index("(", s.index("internalField")) + 1
            b = s.index("\n", a + 1)
            v = float(s[a:b])
            return s[:a] + "\n" + repr(v + 2e-6) + s[b:]
        with_file(case / "0" / "shell", nudge, "A1: 0/shell off by 2e-6 in one cell", "A1", structural=False)
        with_file(w_first / "Gfield", nudge, "A1: Gfield off the blend by 2e-6 Pa", "A1", structural=False)
        expect("A2: a conformance breach", "A2", results={**res, "s18": {**res["s18"], "conformance_ok": False}},
               structural=False)
        expect("A3: W2 fails", "A3", results={**res, "s17b": {**res["s17b"], "w2": False}}, structural=False)
        expect("A4: one nonzero transfer", "A4",
               t=text.replace("contentTransferRate=0 ", "contentTransferRate=1e-30 ", 1), structural=False)
        expect("A4: one step without its release line", "A4",
               t=text.replace("PHASE_CLOT_RELEASED time=", "PHASE_CLOT_RELEASEX time=", 1), structural=False)
        # A6 RE-RULED (DECISIONS 2026-09-30): pass iff |res| <= 1e-6|F| + R_abs. The genuine
        # violation: a residual of 1e-15 on F 1e-14 (1e-1 relative, far above R_abs). The old
        # "F 1e-18 / residual 1e-20 fails" negative is WITHDRAWN: that residual is round-off.
        expect("A6: a residual of 1e-15 on F 1e-14", "A6",
               t=text.replace(XU_ZERO_BAL, _xu_bal(1e-14, 1e-15), 1), structural=False)
        # A6's HEALTHY cases (DECISIONS 2026-09-30, A6 RE-RULED), each must PASS:
        #  (i) a 1e-28 round-off trace (residual 1e-24) on 7 steps;
        #  (ii) a large crossing at round-off: the last 400 steps F 1e-14, residual 1.995e-22
        #       (em72's max |contentResidual|, run 20260928-210844);
        #  (iii) an ARRIVING crossing: the last 400 steps F rising log-uniformly 1e-20 -> 1e-13
        #       with the residual at em72's round-off (1.995e-22) -- verification 08's -a6-fixture C5.
        for label, tt in (
                ("(i) a 1e-28 trace on 7 steps", text.replace(XU_ZERO_BAL, _xu_bal(1e-28, 1e-24), 7)),
                ("(ii) a large crossing at round-off", _xu_last_n(text, 400, lambda k: _xu_bal(1e-14, 1.995e-22))),
                ("(iii) an arriving crossing 1e-20 -> 1e-13", _xu_last_n(
                    text, 400, lambda k: _xu_bal(10 ** (-20 + 7*k/399), 1.995e-22)))):
            a6 = ac.xu2017_report(case, tt, spec, res)["rows"]["A6"]
            assert a6["verdict"] == "PASS", (label, a6)
            print(f"   A6 healthy {label}: PASS (scored {a6['records_scored']}, crossings "
                  f"{a6['records_with_outer_flux']}, R_abs {a6['r_abs_m3_s']:.3e} m3/s)")
        expect("A7: a debonded face", "A7", t=text.replace("debondedFaces=0", "debondedFaces=1", 1),
               structural=False)
        expect("A7: a bond-hold seed lost", "A7", t=text.replace("seeds=113", "seeds=112", 1), structural=False)
        expect("P0: the flow 1% high", "P0",
               t=text.replace(f"outletMeanUx={spec['mean_velocity_m_s']!r}",
                              f"outletMeanUx={spec['mean_velocity_m_s']*1.01!r}", 1))
        expect("N1: 11 steps below half the cap", "N1",
               t=text.replace(f"deltaT = {2.5e-4/96:.12g}\n", "deltaT = 1e-06\n", 11))
        expect("N2: 8 s a step", "N2", t=text.replace(f"ClockTime = {4.3*3840:.0f} s",
                                                      f"ClockTime = {8.0*3840:.0f} s"))
        expect("N3: three slips above 4 u_max", "N3", t=text.replace("slip=0.001", "slip=0.02", 3))
        expect("N7: nuNet 0.2", "N7", t=text.replace("nuNetMax=0.01", "nuNetMax=0.2"))
        # field negatives on the written times
        def zero_pending(s):
            a = s.index("(", s.index("internalField")) + 1
            b = s.index(")", a)
            k = len(s[a:b].split())
            return s[:a] + "\n" + "\n".join(["0.0"]*k) + "\n" + s[b:]
        olds = {p: (p / "thetaPending").read_bytes() for p in first}
        try:
            for p in first:
                (p / "thetaPending").write_text(zero_pending(olds[p].decode()))
            expect("P3: no shell damage by 10 ms", "P3")
        finally:
            for p, o in olds.items():
                (p / "thetaPending").write_bytes(o)
        # bulk-core damage at 5 ms (early at 250): pending content in one bulk cell
        n = spec["geometry"]["nx"]*spec["geometry"]["ny"]
        gf = [float(x) for x in (w5 / "Gfield").read_text().split("(", 1)[1].split(")", 1)[0].split()]
        # a BULK core cell: Gfield >= 20 Pa and row 10 (y 7.5 um, >= 2 um above the held top)
        k0 = max((k for k in range(n) if (k // spec["geometry"]["nx"]) == 10), key=lambda k: gf[k])
        assert gf[k0] >= 20.0, gf[k0]
        def one_pending(s, k=k0):
            a = s.index("(", s.index("internalField")) + 1
            b = s.index(")", a)
            vals = s[a:b].split()
            vals[k] = repr(0.02)
            return s[:a] + "\n" + "\n".join(vals) + "\n" + s[b:]
        with_file(w5 / "thetaPending", one_pending, "P4: bulk-core damage at 5 ms", "P4_N4")
        with_file(w10 / "be", lambda s: s.replace(f"({(1.0 + 0.044*10.0)**2!r} ", "(2.9241 "),
                  "P4: bulk lambda1 1.71 at 10 ms", "P4_N4")
        with_file(w5 / "released",
                  lambda s: one_pending(s).replace(repr(0.02), "1.0"),
                  "P8: a released core cell at 5 ms", "P8")
        # P9: the clot's top rows at 10 ms narrow the gap like Xu's (x 5)
        U10 = (w10 / "U").read_text()
        a = U10.index("(", U10.index("internalField")) + 1
        b = U10.rindex(")\n;")
        rowsU = U10[a:b].strip().split("\n")
        nx_ = spec["geometry"]["nx"]; ny_ = spec["geometry"]["ny"]; h_ = spec["geometry"]["h_m"]
        for k in range((ny_ - 2)*nx_, ny_*nx_):
            if 50e-6 < ((k % nx_) + 0.5)*h_ < 135e-6:
                u = float(rowsU[k].strip("()").split()[0])
                rowsU[k] = f"({u*5.0!r} 0.0 0.0)"
        with_file(w10 / "U", lambda s: U10[:a] + "\n" + "\n".join(rowsU) + "\n" + U10[b:],
                  "P9: the gap narrowed like Xu's", "P9")
        # P1: the clot hydraulically absent at the first write (no bump)
        with_file(w_first / "U", lambda s: _xu_flat_top(s, spec), "P1: no gap hydraulics", "P1")
        # P5: the core zone three times further at 2.5 ms
        w25 = next(p for p in first if abs(float(p.name) - 2.5e-3) < 1e-9)
        with_file(w25 / "Us", lambda s: s.replace(f"({0.372e-6*2.5!r} ", f"({3*0.372e-6*2.5!r} "),
                  "P5: D x3 at 2.5 ms", "P5")
        # P6: no extrusion by 2.5 ms (the first write's contents there)
        oldC, oldH = (w25 / "thetaC").read_bytes(), (w25 / "thetaH").read_bytes()
        try:
            (w25 / "thetaC").write_bytes((w_first / "thetaC").read_bytes())
            (w25 / "thetaH").write_bytes((w_first / "thetaH").read_bytes())
            expect("P6: no extrusion by 2.5 ms", "P6")
        finally:
            (w25 / "thetaC").write_bytes(oldC)
            (w25 / "thetaH").write_bytes(oldH)
        # N5: six cells drained after 5 ms (theta/ref < 0.1 on network cells)
        w75 = next(p for p in first if abs(float(p.name) - 7.5e-3) < 1e-9)
        def drain(s, cells=6):
            a2 = s.index("(", s.index("internalField")) + 1
            b2 = s.index(")", a2)
            vals = s[a2:b2].split()
            hit = 0
            for k in range(len(vals)):
                if float(vals[k]) > 0.3 and hit < cells:
                    vals[k] = repr(float(vals[k])*20.0)
                    hit += 1
            return s[:a2] + "\n" + "\n".join(vals) + "\n" + s[b2:]
        with_file(w75 / "thetaNetRef", drain, "N5: six cells drained at 7.5 ms", "N5")
        # N6 (250 1/s): the extrusion stops -- the 10 ms contents equal 7.5 ms's
        oldC, oldH = (w10 / "thetaC").read_bytes(), (w10 / "thetaH").read_bytes()
        try:
            (w10 / "thetaC").write_bytes((w75 / "thetaC").read_bytes())
            (w10 / "thetaH").write_bytes((w75 / "thetaH").read_bytes())
            expect("N6: the extrusion stops after 7.5 ms", "N6")
        finally:
            (w10 / "thetaC").write_bytes(oldC)
            (w10 / "thetaH").write_bytes(oldH)


XU_ZERO_BAL = ("contentOuterFlux=0 contentTransferRate=0 contentRemoved=0 "
               "detachedContentRate=0 contentResidual=0")


def _xu_bal(f: float, r: float) -> str:
    return (f"contentOuterFlux={f!r} contentTransferRate=0 contentRemoved=0 "
            f"detachedContentRate=0 contentResidual={r!r}")


def _xu_last_n(text: str, n: int, fn) -> str:
    """The fixture's log with its LAST n content-balance records replaced by fn(k)."""
    idx = [m.start() for m in re.finditer(re.escape(XU_ZERO_BAL), text)]
    assert len(idx) >= n, len(idx)
    out, prev = [], 0
    for k, i in enumerate(idx[-n:]):
        out.append(text[prev:i])
        out.append(fn(k))
        prev = i + len(XU_ZERO_BAL)
    out.append(text[prev:])
    return "".join(out)


def _xu_flat_top(s: str, spec: dict) -> str:
    """The first write's U with the clot's 1.15 x top-row bump removed."""
    g = spec["geometry"]
    nx, ny, h = g["nx"], g["ny"], g["h_m"]
    a = s.index("(", s.index("internalField")) + 1
    b = s.rindex(")\n;")
    rows = s[a:b].strip().split("\n")
    for k in range((ny - 2)*nx, ny*nx):
        if 50e-6 < ((k % nx) + 0.5)*h < 135e-6:
            u = float(rows[k].strip("()").split()[0])
            rows[k] = f"({u/1.15!r} 0.0 0.0)"
    return s[:a] + "\n" + "\n".join(rows) + "\n" + s[b:]


def run_direct(name: str, check) -> None:
    """A scenario that tests a function directly, with no synthesised case.

    Some contracts are not expressible as a fixture: the debond parser has to
    cope with a log format that does not exist yet, and no generated log can
    contain a future field. Those are checked against the function itself.
    """
    try:
        check()
    except AssertionError as error:
        raise AssertionError(f"{name}: {error}")
    print(f"{name:<38} -> ok")


# THE STEP-1 WATCHER'S FIRING AND SILENT FIXTURES (design
# 2026-10-02-case-free-shell-throat, rev. 3, S7.3b, last bullet). The firing
# log carries em79's OWN first three V1 records, verbatim from
# preflight/20261002-090520-am2020-main-case/.../em79_G4_s18p_Cnp/
# log.phaseClotAxisymmetricFluid lines 81, 82, 148 (em79 is bonded, so they
# are non-zero); the silent log is the same three lines with every V1 key 0.
# The real clot_free_watcher.py runs as a subprocess, as Allrun launches it,
# and run_digest.py's child_facts reads what it left behind.
EM79_V1_LINES = (
    "PHASE_CLOT_RELEASED time=4.16666666667e-06 cells=0 volume=0 clotCells=2618 "
    "supported=2934 bondSeeds=62 releasedOnBond=0 coreExempted=0 supportLimit=0.5 "
    "indSupportLimit=0.25 indBasis=content",
    "PHASE_CLOT_BONDHOLD time=4.16666666667e-06 seeds=56 heldCells=286 heldWeight=286 "
    "taper=0 dilation=4 penalty=10000000 carrier=0",
    "PHASE_CLOT_DEBOND time=4.16666666667e-06 sigmaFound=1 sigmaField=sigmaElastic "
    "newFaces=0 reformedFaces=0 debondedFaces=0 bondedClotFaces=56 injuryFaces=56 "
    "wallFaces=448 debondedArea=0 peakTraction=4.93432948821e-16",
)


def free_shell_watcher_fixtures() -> None:
    import importlib.util
    watcher = Path(__file__).resolve().parent / "clot_free_watcher.py"
    digest_path = CASE.parents[1] / "scripts" / "run_digest.py"
    spec = importlib.util.spec_from_file_location("run_digest_fs_dry", digest_path)
    digest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(digest)
    silent_lines = tuple(
        re.sub(r"\b(bondedClotFaces|injuryFaces|supported|bondSeeds|seeds)=\S+",
               r"\1=0", ln) for ln in EM79_V1_LINES)
    assert silent_lines != EM79_V1_LINES and all("=56" not in ln for ln in silent_lines)
    with tempfile.TemporaryDirectory() as tmp:
        for label, lines in (("firing", EM79_V1_LINES), ("silent", silent_lines)):
            d = Path(tmp) / label
            (d / "system").mkdir(parents=True)
            (d / "system" / "controlDict").write_text(
                "functions\n{\n    stopWatcher\n    {\n        type stopAtFile;\n    }\n}\n")
            (d / "child.json").write_text(json.dumps(
                {"child_id": f"fs_{label}", "series": "throat_free_shell", "end_time_s": 0.03}))
            (d / "log.phaseClotAxisymmetricFluid").write_text(
                "Time = 4.166666667e-06s\n\n" + "\n".join(lines) + "\n\nEnd\n")
            # the solver's own exit after a stopAtFile writeNow is 0: the
            # watcher's verdict must not depend on it
            (d / "run-exit-code.txt").write_text("0\n")
            r = subprocess.run([sys.executable, "-B", str(watcher), str(d)],
                               capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, f"{label}: watcher exit {r.returncode}: {r.stderr[-400:]}"
            stop, why = d / "stopWatcher", d / "watcher-stop.txt"
            facts = digest.child_facts(str(d))
            if label == "firing":
                assert stop.is_file(), "firing: the stopAtFile watch file was not touched"
                assert why.is_file(), "firing: no watcher-stop.txt"
                first = why.read_text().splitlines()[0]
                assert first.startswith("WATCHER-STOPPED: PHASE_CLOT_RELEASED: "
                                        "supported=2934, bondSeeds=62"), first
                assert str(facts.get("stop", "")).startswith("WATCHER-STOPPED"), facts
            else:
                assert not stop.exists(), "silent: the watch file was touched on a clean log"
                assert not why.exists(), "silent: watcher-stop.txt written on a clean log"
                assert not str(facts.get("stop", "")).startswith("WATCHER"), facts
            print(f"   {label}: watch file {stop.exists()}, digest stop "
                  f"{str(facts.get('stop'))[:70]!r}")


# ---- S21 (docket 20261005-1621, pipeline pass 2b) ---------------------------------
# The generator must refuse a dictionary carrying a built-in option or a deleted
# parameter (the solver refuses both by name at construction: s21BuiltInKeys,
# s21DeletedKeys), and a control key naming either. Each negative is run on its own,
# on a pristine copy, and must fail with its own message; the pristine copy must
# generate.
S21_GENERATOR_NEGATIVES = (
    ("template phaseClotProperties carries a built-in option",
     "template", "fsiAitken no;", "built-in option(s) ['fsiAitken']"),
    ("template phaseClotProperties carries a deleted parameter",
     "template", "phi0 0.5;", "deleted parameter(s) ['phi0']"),
    ("growth_parameters carries the deleted k1",
     "controls", ("growth_parameters", None, "k1", 0.0), "names the deleted parameter k1"),
    ("a child carries nFSICorrectors (camel case)",
     "controls", ("child", "fsf_G4_kxu_v250", "nFSICorrectors", 60),
     "names the deleted parameter nFSICorrectors"),
    ("a child carries skeleton_us_bound (snake case)",
     "controls", ("child", "fsf_G4_kxu_v250", "skeleton_us_bound", 1.0),
     "names the deleted parameter skeletonUsBound"),
    ("a child carries operator_audit_every (a pre-S21 control key)",
     "controls", ("child", "xu5_g250", "operator_audit_every", 20),
     "names the deleted parameter operatorAuditEvery"),
    ("a series carries skeleton_implicit_factor",
     "controls", ("series", "throat_embolus", "skeleton_implicit_factor", {"1000": 8}),
     "names the deleted parameter skeletonImplicitFactor"),
    ("a child carries material_courant",
     "controls", ("child", "em76_G4_s18", "material_courant", 0.5),
     "names the deleted parameter materialCourant"),
    ("a child carries fibre_slack (a built-in option)",
     "controls", ("child", "em76_G4_s18", "fibre_slack", "v1"),
     "names the built-in option fibreSlack"),
)


# The number of children the generator emits: the 18 kept + the 8 S22 validation
# twins of BATCH 71 + the 4 S23 twins of BATCH 72 + the 2 production children of BATCH 73 + the 1 diagnostic child of BATCH 74
# (D74_refI, the passive intact-network reference) + the 1 re-run child of BATCH 75 (D75_refI, the same measurement on rev. 3.1) + the 2 B76-5a
# smoke children (S25_5a_tan, S25_5a_tan_half; case-engineer 09b) + the 2 B76-OFF children (B76_OFF_A on the baseline binary, B76_OFF_B on the S25 build; case-engineer 09c) + the 4 B76-RT children (RT_P73_A/B, RT_5a_A/B) + the 6 BATCH 78 children (B78_OFF_A/B, B78_CTL_1/2, B78_RT_5a_A/B; case-engineer 06) + the 1 BATCH 79 child (B79_Y_GATE, the Y-bifurcation geometry gate; case-engineer 12 and 13) + the 7 symmetric-Y children (Y80_YS794_FLOW, Y80_YS850_FLOW, Y80_RT_A, Y80_RT_B of BATCH 80, Y80_RT86_B of BATCH 82 and the plug children Y81_YS794_PLUG and Y81_YS850_PLUG of BATCH 81; case-engineer 27, 27b, 28 and 33). ONE pin, read by
# s21_generator_negatives, s23_generator_negatives and main(): BATCH 72 updated main's copy and left this
# function's at 26, and release 2026-10-07 died on it ten minutes into the build.
KEPT_CHILDREN = 56


def s21_generator_negatives() -> None:
    with tempfile.TemporaryDirectory(prefix="am2020-s21-gen-") as tmp:
        root = build_case(Path(tmp))
        n = len(json.loads((root / "runs" / "matrix.json").read_text()))
        assert n == KEPT_CHILDREN, f"the pristine copy generated {n} children, expected {KEPT_CHILDREN}"
        shutil.rmtree(root / "runs")
        controls_path = root / "constant" / "mainCaseControls.json"
        props_path = root / "template" / "constant" / "phaseClotProperties"
        pristine_c, pristine_p = controls_path.read_text(), props_path.read_text()
        for label, where, what, expected in S21_GENERATOR_NEGATIVES:
            controls_path.write_text(pristine_c)
            props_path.write_text(pristine_p)
            if where == "template":
                props_path.write_text(pristine_p.rstrip("\n") + f"\n{what}\n")
            else:
                c = json.loads(pristine_c)
                level, name, key, value = what
                if level == "child":
                    block = next(x for x in c["children"] if x["child_id"] == name)
                elif level == "series":
                    block = c["series"][name]
                else:
                    block = c[level]
                assert key not in block, f"{label}: {key} already present"
                block[key] = value
                controls_path.write_text(json.dumps(c, indent=1) + "\n")
            done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")],
                                  capture_output=True, text=True)
            said = done.stdout + done.stderr
            assert done.returncode != 0, f"{label}: the generator accepted it"
            assert expected in said, f"{label}: refused for another reason: {said[-400:]}"
            shutil.rmtree(root / "runs", ignore_errors=True)
            print(f"generator refuses: {label:<54} -> {expected}")


# G3'lambda (S20 section 2.3), on the assessor's own s10_report: N/A on a pre-S20 log,
# FAIL on batch 68's lambdaFluxMean 0, PASS on the loose replay's 0.99, and the 0.1
# threshold seen from both sides.
def _fct(lines):
    return "".join(
        f"PHASE_CLOT_CONTENT_FCT time={1e-6*(i+1):.6g} activeFaces={a} limitedFaces=0 "
        f"lambdaMin=0 lambdaFluxMean={lam}" + ("" if rule is None else f" zeroFluxRule={rule}")
        + " negAfter=0\n" for i, (a, lam, rule) in enumerate(lines))


def g3_lambda_contract() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "assess_case_g3l", Path(__file__).resolve().parent / "assess_case.py")
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    cases = (
        ("pre-S20 log, lambda 0", [(40, 0, None)] * 5, None),
        ("S20 log, lambda 0 on every active step (batch 68)", [(40, 0, "exactZero")] * 5, False),
        ("S20 log, loose replay 0.99", [(40, 0.99, "exactZero")] * 5, True),
        ("S20 log, median exactly 0.1", [(40, 0.05, "exactZero"), (40, 0.1, "exactZero"),
                                         (40, 0.2, "exactZero")], True),
        ("S20 log, median 0.0999", [(40, 0.05, "exactZero"), (40, 0.0999, "exactZero"),
                                    (40, 0.2, "exactZero")], False),
        ("S20 log, idle steps do not count", [(0, 0, "exactZero")] * 9 + [(40, 0.99, "exactZero")],
         True),
        ("S20 log, one line not exactZero", [(40, 0.99, "exactZero")] * 4 + [(40, 0.99, "legacy")],
         False),
        ("S20 log, no active face", [(0, 1.0, "exactZero")] * 3, None),
    )
    for label, lines, want in cases:
        got = ac.s10_report(_fct(lines), 4.0)
        assert got["g3_lambda"] is want, f"{label}: g3_lambda {got['g3_lambda']!r}, want {want!r} ({got})"
        if want is None and lines[0][2] is None:
            assert str(got["g3_lambda_note"]).startswith("N/A"), got["g3_lambda_note"]
        print(f"   G3'lambda {label:<52} -> {got['g3_lambda']} (median {got['g3_lambda_median']})")


# THE 5 ms WATCHER'S RECORD (pass 2a): runs/<child>/watcher-verdict.json. The child is
# synthesised COMPLETE (exit 0, "End", the last time at end_time) so the verdict file
# alone must decide: FAIL, WATCHER_ERROR and an unreadable file are STOPPED_BY_WATCHER
# and inadmissible; PASS and NOT_REACHED leave a complete child complete.
WATCHER_RECORDS = {
    "fsf_G4_kxu_r3_40": {"verdict": "FAIL", "clauses": [
        {"clause": "width", "status": "FAIL", "value": 4.54},
        {"clause": "lambda_median", "status": "FAIL", "value": 0.0}],
        "stop": {"touched": True, "acknowledged": "stopAtFile(timeIndex=1053): stopping now after writing"}},
    "fsf_G4_kxu_nor3_40": {"verdict": "PASS", "clauses": [
        {"clause": "width", "status": "REPORTED", "value": 4.54}]},
    "fsf_id": {"verdict": "WATCHER_ERROR", "reason": "no 0.005 frame", "clauses": []},
    "em78_id": "{ not json",
    "xu5_id": {"verdict": "NOT_REACHED", "reason": "the child ended first"},
}


def _write_watcher_records(root: Path) -> None:
    for cid, rec in WATCHER_RECORDS.items():
        (root / "runs" / cid / "watcher-verdict.json").write_text(
            rec if isinstance(rec, str) else json.dumps(rec))


def watcher_stopped(r, code):
    stopped = {"fsf_G4_kxu_r3_40", "fsf_id", "em78_id"}
    for cid in WATCHER_RECORDS:
        c = r["children"][cid]
        if cid in stopped:
            assert c["status"] == "STOPPED_BY_WATCHER", (cid, c["status"])
            assert not c["completed"] and not c["admissible"], (cid, c["completed"], c["admissible"])
        else:
            assert c["status"] == "COMPLETE" and c["admissible"], (cid, c["status"], c["admissible"])
    why = {e["child_id"]: e for e in r["inadmissible"]}
    assert set(why) == stopped, sorted(why)
    assert all(e["status"] == "STOPPED_BY_WATCHER" and e["why"].startswith("STOPPED_BY_WATCHER")
               for e in why.values()), why
    assert why["fsf_G4_kxu_r3_40"]["why"].endswith("['width', 'lambda_median'])"), why
    assert r["children"]["em78_id"]["watcher_verdict"]["verdict"] == "UNREADABLE"
    assert r["children"]["fsf_G4_kxu_r3_40"]["watcher_verdict"]["stop"]["touched"] is True



def main() -> int:
    # S21 (docket 20261005-1621, pipeline pass 2b): the kept set only. Every
    # scenario below runs on the 18 kept children (+ the 8 S22 twins) of throat_embolus, xu2017_fig5
    # and throat_free_shell; the scenarios of the retired series (growth,
    # required strength, demonstration, transport, tear, source 33, deformation)
    # and the generator negatives of the deleted refusals are cut (the pre-S21
    # file: release-4198bb17).
    KEPT = KEPT_CHILDREN   # the 18 kept children + the 8 S22 validation twins of BATCH 71 + the 4 S23 twins of BATCH 72 + the 2 production children of BATCH 73 + the 1 diagnostic child of BATCH 74 + the 1 re-run child of BATCH 75

    def healthy(r, code):
        assert code == 0, f"exit {code}"
        assert r["children_total"] == KEPT, r["children_total"]
        assert r["children_admissible"] == r["children_total"], r["inadmissible"]
        assert r["library_compiled"]
        assert all(c["status"] == "COMPLETE" for c in r["children"].values()), sorted(
            (k, c["status"]) for k, c in r["children"].items() if c["status"] != "COMPLETE")
        assert r["xu2017_fig5"], "no Xu report"
        assert [(e["pair_id"], e["status"]) for e in r["y_rt"]] == [("Y80_RT", "PASS"), ("Y80_RT86", "PASS")], ("the symmetric Y's two restart pairs (100 -> 105 and the confirming 86 -> 105, on the same A) on an exact synthetic restart", r.get("y_rt"))
    exercise("healthy", variant(), healthy)

    def compile_failed(r, code):
        assert code == 1 and not r["library_compiled"]
    exercise("compile_failed", variant(), compile_failed, build_log=BAD_BUILD)

    # AN INADMISSIBLE CHILD MUST BE NAMED AND EXCLUDED, never averaged in.
    def only(cid, reason=None):
        def check(r, code):
            bad = [e["child_id"] for e in r["inadmissible"]]
            assert bad == [cid], bad
            if reason:
                assert reason in r["children"][cid]["bounds_reason"], r["children"][cid]["bounds_reason"]
        return check
    exercise("crashed_child_is_excluded_and_named",
             variant(fsf_G4_kxu_v250=dict(exit_code=136)), only("fsf_G4_kxu_v250"))
    exercise("incomplete_child_is_excluded",
             variant(em76_G4_s18=dict(short=True)), only("em76_G4_s18"))
    exercise("nan_field_is_inadmissible",
             variant(xu5_g500=dict(nan_field=True)), only("xu5_g500", "non-finite"))

    # THE PHASE FIELD AFTER S6c-2 (q, thetaC, thetaH; no c): the reconstruction
    # c = 2q - 1 must give exactly the same report as the legacy fixture.
    def theta_native_matches(base):
        def check(r, code):
            assert code == 0, f"exit {code}"
            diffs = report_differences(base, r)
            assert not diffs, "theta-native report differs:\n  " + "\n  ".join(diffs[:12])
        return check
    baseline = exercise("healthy_baseline_for_theta_native", variant(), healthy)
    exercise("theta_native_grades_identically", variant(),
             theta_native_matches(baseline), every_child={"theta_native": True})
    exercise("out_of_bounds_phase_is_seen_in_theta",
             variant(fsb_ctl_r3=dict(c_extreme=1.5)), only("fsb_ctl_r3", "phase out of bounds"),
             every_child={"theta_native": True})
    exercise("out_of_bounds_phase_is_excluded",
             variant(fsb_ctl_r3=dict(c_extreme=1.5)), only("fsb_ctl_r3", "phase out of bounds"))
    # BATCH 79, the Y gate's child, end to end through assess_case.main on the synthetic returned case (y_synth on the replica's real cells): READY when clean (the healthy
    # scenario above), a run that stopped early is INCOMPLETE and named, a planted defect makes the child NOT_READY by the gate it names (never inadmissible: the measurement exists).
    exercise("y_gate_child_stopped_early_is_excluded_and_named",
             variant(B79_Y_GATE=dict(short=True)), only("B79_Y_GATE", "did not complete"))

    def y_not_ready(gate):
        def check(r, code):
            assert code == 0 and not r["inadmissible"], (code, r["inadmissible"])
            y = r["children"]["B79_Y_GATE"]
            assert y["y_gate_verdict"] == "NOT_READY" and y["y_gate"]["failing"] == [gate], (y["y_gate_verdict"], y["y_gate"]["failing"])
        return check
    exercise("y_gate_child_with_a_wrong_split_is_not_ready_by_gate_2",
             variant(B79_Y_GATE=dict(y_options=dict(split=0.809042))), y_not_ready("g2"))
    exercise("y_gate_child_with_a_misset_drive_is_not_ready_by_gate_5",
             variant(B79_Y_GATE=dict(y_options=dict(q0_scale=1.01))), y_not_ready("g5"))
    exercise("y_gate_child_with_a_step_3pct_high_is_still_ready",
             variant(B79_Y_GATE=dict(y_options=dict(dt=1.03 * 3.1188e-06))), healthy)
    run_direct("y_gate_one_child_matrix_is_assessed_ready_and_a_wrong_split_not_ready", y_gate_only_matrix)
    run_direct("y_launch_trees_are_assessed_exit_0_with_y_gate_and_y_rt", y_launch_trees)
    run_direct("y_plug_chains_of_two_sessions_merge_and_score_both_chains", y_plug_chain_scenarios)
    run_direct("s19_source_errors_are_case_relative_not_the_temporary_directory", s19_source_errors_are_case_relative)
    # THE SYMMETRIC Y (BATCH 80/81; case-engineer 27b), end to end through assess_case.main on y_synth / y_rt_synth cases. The symmetric gates read gate 2 as |q1/q0 - 0.5| <= 1e-7:
    # a 3e-5 split error (the size of a 1e-9 m mirror defect, design 6 A item 2) and one outlet face's flux on the wrong daughter (about 2e-2) are NOT_READY by gate 2.
    def ys_not_ready(cid, gate):
        def check(r, code):
            assert code == 0 and not r["inadmissible"], (code, r["inadmissible"])
            y = r["children"][cid]
            assert y["y_gate_verdict"] == "NOT_READY" and gate in y["y_gate"]["failing"], (cid, y["y_gate_verdict"], y["y_gate"]["failing"])
            assert y["y_gate"]["g2"]["verdict"] == "FAIL", y["y_gate"]["g2"]["verdict"]
        return check
    exercise("ys794_gate_child_with_a_3e-5_asymmetry_is_not_ready_by_gate_2",
             variant(Y80_YS794_FLOW=dict(y_options=dict(split=0.5 + 3e-5))), ys_not_ready("Y80_YS794_FLOW", "g2"))
    exercise("ys850_gate_child_with_a_swapped_outlet_face_flux_is_not_ready_by_gate_2",
             variant(Y80_YS850_FLOW=dict(y_options=dict(swap_faces=(54, 0)))), ys_not_ready("Y80_YS850_FLOW", "g2"))
    exercise("ys_gate_child_stopped_early_is_excluded_and_named", variant(Y80_YS850_FLOW=dict(short=True)), only("Y80_YS850_FLOW", "did not complete"))
    # the restart pair: an exact restart PASSES (the healthy scenario); one unit in the last place of one field of RT_B's step-105 directory, or of one logged deltaT, is FAIL
    # (the children are COMPLETE: the measurement exists), and a leg 2 that stopped early is INCOMPLETE and named.
    def y_rt_status(want, want86="PASS"):
        def check(r, code):
            assert code == 0 and not r["inadmissible"], (code, r["inadmissible"])
            assert [(e["pair_id"], e["status"]) for e in r["y_rt"]] == [("Y80_RT", want), ("Y80_RT86", want86)], r.get("y_rt")
        return check
    exercise("y_rt_one_ulp_in_a_field_is_fail", variant(Y80_RT_B=dict(y_options=dict(ulp="q"))), y_rt_status("FAIL"))
    exercise("y_rt_one_ulp_in_a_logged_dt_is_fail", variant(Y80_RT_B=dict(y_options=dict(dt_ulp=True))), y_rt_status("FAIL"))
    exercise("y_rt_a_cold_leg_2_is_void_not_pass", variant(Y80_RT_B=dict(y_options=dict(cold_b=True))), y_rt_status("VOID"))
    exercise("y_rt_leg_2_stopped_early_is_excluded_and_named", variant(Y80_RT_B=dict(short=True)), only("Y80_RT_B", "did not reach step"))
    # the confirming pair Y80_RT86 (BATCH 82; A = the same Y80_RT_A): the same planted defects on Y80_RT86_B move ONLY its own pair (the other stays PASS)
    exercise("y_rt86_one_ulp_in_a_field_is_fail", variant(Y80_RT86_B=dict(y_options=dict(ulp="q"))), y_rt_status("PASS", "FAIL"))
    exercise("y_rt86_one_ulp_in_a_logged_dt_is_fail", variant(Y80_RT86_B=dict(y_options=dict(dt_ulp=True))), y_rt_status("PASS", "FAIL"))
    exercise("y_rt86_a_cold_leg_2_is_void_not_pass", variant(Y80_RT86_B=dict(y_options=dict(cold_b=True))), y_rt_status("PASS", "VOID"))
    exercise("y_rt86_a_b_built_for_another_leg_1_stop_is_void", variant(Y80_RT86_B=dict(y_options=dict(child_json_leg1=90))), y_rt_status("PASS", "VOID"))
    exercise("y_rt86_leg_2_stopped_early_is_excluded_and_named", variant(Y80_RT86_B=dict(short=True)), only("Y80_RT86_B", "did not reach step"))
    exercise("y_plug_child_stopped_early_is_excluded_and_named", variant(Y81_YS794_PLUG=dict(short=True)), only("Y81_YS794_PLUG", "did not reach the end time"))
    exercise("y_plug_ys850_child_stopped_early_is_excluded_and_named", variant(Y81_YS850_PLUG=dict(short=True)), only("Y81_YS850_PLUG", "did not reach the end time"))
    # MISSING FIELDS MUST NOT PASS VACUOUSLY.
    exercise("missing_fields_are_inadmissible",
             variant(em79_G4_s18p_Cnp=dict(omit_fields=True)), only("em79_G4_s18p_Cnp"))

    # THE 5 ms WATCHER'S RECORD: a stopped child is STOPPED_BY_WATCHER, never complete.
    exercise("a_watcher_stopped_child_is_never_complete", variant(), watcher_stopped,
             after=_write_watcher_records)
    run_direct("g3_lambda_is_na_before_s20_and_fails_on_lambda_0", g3_lambda_contract)
    # G3'lambda IS AN ADMISSIBILITY GATE (S20 section 2.3; docket 20261005-1621 report 21):
    # an otherwise healthy child whose S20-format FCT lines read lambdaFluxMean 0 is
    # inadmissible and named for it; its S20 twin at 0.99 and a pre-S20 log at 0 (N/A)
    # stay admissible.
    exercise("g3_lambda_0_on_s20_log_is_inadmissible",
             variant(fsf_G4_kxu_v250=dict(fct=(0, "exactZero"))),
             only("fsf_G4_kxu_v250", "G3'lambda: the median lambdaFluxMean"))
    for g3l_label, g3l_fct in (("g3_lambda_0p99_on_s20_log_is_admissible", (0.99, "exactZero")),
                               ("g3_lambda_0_on_pre_s20_log_is_na", (0, None))):
        def g3l_kept(r, code, _fct=g3l_fct):
            healthy(r, code)
            s10 = r["children"]["fsf_G4_kxu_v250"]["s10"]
            want = None if _fct[1] is None else True
            assert s10["g3_lambda"] is want, (s10["g3_lambda"], s10["g3_lambda_note"])
            assert s10["fct_records"] > 1, s10["fct_records"]
        exercise(g3l_label, variant(fsf_G4_kxu_v250=dict(fct=g3l_fct)), g3l_kept)
    run_direct("s21_generator_refuses_built_in_and_deleted_keys", s21_generator_negatives)
    run_direct("s22_generator_refuses_bad_solver_keys", s22_generator_negatives)
    run_direct("y_gate_generator_refuses_bad_y_keys_and_schedule", y_gate_generator_negatives)
    run_direct("s23_generator_refuses_bad_tear_ratio", s23_generator_negatives)
    run_direct("s25_build1_refusals_gamma_scaling_and_assessor_fields", s25_build1_checks)
    run_direct("s25_build2_lt_refusals_generator_and_assessor_gates", s25_build2_checks)
    run_direct("s251_restart_phiskel_refusal_and_old_time_primes", s251_checks)


    # --- THE PARSER ITSELF --------------------------------------------------
    #
    # The debond line was matched by one long positional regex. Adding
    # reformedFaces and injuryFaces to the solver stopped it matching at all,
    # and the symptom is indistinguishable from a run that never debonded:
    # silence. Key-based parsing tolerates new fields; a REMOVED field must
    # raise. Checked directly, because no fixture can cover a format that does
    # not exist yet.
    def parser_contract():
        from assess_case import parse_debond
        base = (
            "PHASE_CLOT_DEBOND time=0.5 newFaces=1 debondedFaces=3 "
            "bondedClotFaces=57 wallFaces=160 debondedArea=6.75e-12 "
            "peakTraction=0.79 peakTractionAny=0.79 strength=0.8 "
            "utilisation=0.99"
        )
        # An OLD log, from before the change, must still parse -- and must
        # read as "no re-adhesion", which is the truth for those runs.
        old = parse_debond(base)
        assert len(old) == 1 and old[0]["reformed_faces"] == 0, old
        assert old[0]["injury_faces"] == 160, old

        # A NEW log, and a FUTURE one with a field nobody has written yet.
        new = parse_debond(
            base.replace("newFaces=1", "newFaces=1 reformedFaces=2")
                .replace("wallFaces=160", "injuryFaces=60 wallFaces=160")
            + " someFieldFromNextYear=7"
        )
        assert new[0]["reformed_faces"] == 2, new
        assert new[0]["injury_faces"] == 60, new

        # A field REMOVED must be loud, not silent.
        try:
            parse_debond(base.replace("peakTraction=0.79 ", ""))
        except ValueError as e:
            assert "peakTraction" in str(e), e
        else:
            raise AssertionError("a missing required field parsed silently")
    run_direct("debond_parser_tolerates_new_fields_and_refuses_missing_ones",
               parser_contract)

    # The SAME defect, in the body-velocity line, found the hard way. It went
    # unnoticed for three runs because this dry run's fixture emitted the old
    # format, so the fixture and the regex agreed with each other and both
    # disagreed with the solver. Run 20260813-084207 came back with
    # body_velocity_m_s = null on a run that translated 49 cells.
    def body_velocity_contract():
        from assess_case import parse_body_velocity
        base = (
            "PHASE_CLOT_BODY_VELOCITY time=0.05 Ux=1.67e-03 Uy=0 "
            "UyDiscarded=-9.1e-05 crossFraction=0.054 axisTilt=0.031 "
            "bandCells=1016 freeStreamUx=3.21e-03 ratioToFreeStream=0.521"
        )
        rows = parse_body_velocity(base + " someFieldFromNextYear=7\n")
        assert len(rows) == 1, rows
        assert rows[0]["Ux"] == 1.67e-03, rows
        assert rows[0]["ratioToFreeStream"] == 0.521, rows
        assert rows[0]["axisTilt"] == 0.031, rows

        # The zero-body branch prints a shorter line; it must still parse.
        assert parse_body_velocity(
            "PHASE_CLOT_BODY_VELOCITY time=0.01 Ux=0 Uy=0 bandCells=0 "
            "freeStreamUx=3.2e-03 ratioToFreeStream=0\n"
        )[0]["Ux"] == 0.0

        for gone in ("Ux", "ratioToFreeStream", "freeStreamUx"):
            try:
                parse_body_velocity(re.sub(rf"\b{gone}=\S+ ?", "", base))
            except ValueError as e:
                assert gone in str(e), e
            else:
                raise AssertionError(f"{gone} vanished silently")
    run_direct("body_velocity_parser_refuses_a_changed_log_format",
               body_velocity_contract)

    # --- COHESIVE TEAR: Dr Li's redirect -------------------------------------
    #
    # "We want to see the detachment of the blood clot FROM THE CLOT BODY.
    # Right now I see the entire clot is sliding away, which is not realistic
    # and it is not observed from any experiments." The series answers that by
    # welding the wall and weakening the body. Three things can quietly undo
    # it, and each has a scenario.

    # 5. A MISSING SHELL FIELD MUST BE NAMED. The solver refuses it outright,
    #    but a log from before that refusal, or a child whose initialiser was
    #    skipped, would otherwise report a core of None and read as a clean
    #    single-material run.
    def missing_shell_field():
        import assess_case as A
        with tempfile.TemporaryDirectory(prefix="am2020-cs-") as tmp:
            case = Path(tmp) / "c"
            (case / "0").mkdir(parents=True)
            (case / "0.01").mkdir()
            write_scalar(case / "0" / "Ccy", [0.5e-6] * 10)
            write_scalar(case / "0.01" / "Dg", [0.99] * 10)
            spec = {"series": "source33_replication", "DgDetach": 0.95,
                    "core_shell": True,
                    "geometry": {"h_m": 7.5e-07},
                    "clot": {"height_m": 1.275e-05}}
            out = A.tear_measurement(case, spec)
            assert out["core_shell"] is True, out
            assert "0/shell is missing" in out["reason_core_shell"], out
            assert out["core_survived"] is None, out
    run_direct("a_core_shell_child_without_a_shell_field_says_so",
               missing_shell_field)

    # THE GENERATOR MUST REFUSE the configuration that produced nine nulls.
    # A reporting test cannot catch this: by the time the assessor runs, the
    # measurement is already missing and "no PHASE_CLOT_DEBOND lines" is the
    # honest thing for it to say. The refusal has to happen at generation.
    import copy
    with tempfile.TemporaryDirectory(prefix="am2020-main-gen-") as tmp:
        root = build_case(Path(tmp))
        shutil.rmtree(root / "runs")
        controls_path = root / "constant" / "mainCaseControls.json"
        # Snapshot ONCE. Re-reading the file each time would mutate an
        # already-mutated control set, and the second case would fail for the
        # first one's reason -- a test that passes while checking nothing.
        pristine = controls_path.read_text()
        for label, mutate, expected in (
            ("a misspelt network_pressure_weight",
             lambda c: _s16(c, "em76_G4_s18", network_pressure_weight="contentratio"),
             "network_pressure_weight 'contentratio' is not one of"),
            ("the content weight with no residual ramp",
             lambda c: _s16(c, "em76_G4_s18", residual_chi=0.0),
             "network_pressure_weight contentRatio ramps to zero below residual_chi"),
            ("a misspelt proj_site_log",
             lambda c: _s16(c, "em78_G4_s18p_ctl", proj_site_log="on"),
             "proj_site_log 'on' is not one of no, yes."),
            ("em78 drifting from em76",
             lambda c: _s16(c, "em78_G4_s18p_ctl", k0p_m2=2e-13),
             "S18p parent check: em78_G4_s18p_ctl differs from em76_G4_s18 in k0p_m2"),
            ("em79 without its reverted switch",
             lambda c: _s16(c, "em79_G4_s18p_Cnp", network_pressure_weight="contentRatio"),
             "S18p parent check: em79_G4_s18p_Cnp does not differ from em76_G4_s18 "
             "in its named key network_pressure_weight"),
            *S19_REFUSALS,
        ):
            broken = copy.deepcopy(json.loads(pristine))
            mutate(broken)
            controls_path.write_text(json.dumps(broken, indent=2))
            done = subprocess.run(
                [sys.executable, str(root / "scripts" / "create_cases.py")],
                capture_output=True, text=True,
            )
            if done.returncode == 0:
                raise AssertionError(
                    f"the generator ACCEPTED {label}; it must refuse"
                )
            message = (done.stdout + done.stderr).strip().splitlines()[-1]
            if expected not in message:
                raise AssertionError(
                    f"the generator refused {label} for the WRONG reason: "
                    f"{message}"
                )
            print(f"generator refuses: {label:<42} -> {message[:78]}")
            if (root / "runs").exists():
                shutil.rmtree(root / "runs")

        controls_path.write_text(pristine)

    # ---- THE TWO-VELOCITY GATE: FIRES, DOES NOT FIRE SPURIOUSLY, AND IS
    #      NOT SET BELOW THE PHYSICS ----
    #
    # Written at the same time as the guard, not after. The last two guards
    # added to this case read correctly, reviewed correctly, and did nothing:
    # one sat inside the block it was guarding, and one was piped through
    # `tail` so the shell reported the pipe's exit code. A guard that cannot
    # fail has not passed -- and a guard that cannot PASS is just as useless,
    # which this project has now produced four times.
    import assess_case as ac

    line = (
        "PHASE_CLOT_SKELETON_VELOCITY time=0.0095 clotCells=9577 "
        "meanVsOverU={mean} maxVsOverU={mx} exceedFraction={bare} "
        "exceedFractionMargin={marg} vsMargin=1.1 "
        "maxVs=1.2e-02 maxU=1.1e-02 maxPhiS=0.0099"
    )

    def kv(**kw):
        got = ac.parse_skeleton_velocity(line.format(**kw))
        if not got:
            raise AssertionError(
                "parse_skeleton_velocity found NOTHING in a well-formed line; "
                "the gate would silently pass on every run"
            )
        return got[-1]

    # 1. It fires on the measured pathology of run 20260904-214345.
    bad = kv(mean=1.4, mx=3.2, bare=0.883, marg=0.62)
    if abs(bad["exceedFraction"] - 0.883) > 1e-12:
        raise AssertionError(f"misread exceedFraction: {bad}")
    if not (bad["exceedFractionMargin"] > ac.VS_EXCEED_LIMIT):
        raise AssertionError(
            f"VS_EXCEED_LIMIT is {ac.VS_EXCEED_LIMIT}, which PASSES the "
            f"pathology measured in run 20260904-214345. The gate is inert."
        )
    print(f"gate fires:   the pathology (mean 1.4, max 3.2) exceeds the "
          f"{100 * ac.VS_EXCEED_LIMIT:.0f}% margin limit")

    # 2. IT DOES NOT FIRE ON THE GENUINE FREE-ELEMENT SLIP. This is the test
    #    that would have caught the gate as first written. Solving the 2x2 for
    #    an unsupported element gives u - v -> phi_s*grad(p)/K = 5.2e-05 m/s,
    #    about 0.5% of |U| -- so a correct run can have most clot cells barely
    #    above 1.0 while none is above 1.1. A gate on the BARE count fails it.
    fine = kv(mean=1.004, mx=1.02, bare=0.71, marg=0.0)
    if fine["exceedFractionMargin"] > ac.VS_EXCEED_LIMIT:
        raise AssertionError(
            "the gate rejects a run whose only excess is the genuine "
            "phi_s*grad(p)/K free-element slip (bare count 71%, margin count "
            "0%). It is set below the physics and no correct run could pass."
        )
    print("gate passes:  a 71% BARE count that is entirely the genuine 0.5% "
          "free-element slip")

    # 2b. THE WHOLE-FIELD BOUND, which is a DIFFERENT quantity from the three
    #     above. Those are clot cells only; maxVs is gMax over everything.
    #     Run 20260905-191135 diverged with maxVs/maxU = 237.7 while its
    #     clot-cell metrics were indistinguishable from the run before it --
    #     the runaway was in the LUMEN and no clot-cell gate could see it.
    wf = kv(mean=0.9387, mx=1.489, bare=0.20, marg=0.0133)
    if wf["maxVs"] / wf["maxU"] > ac.VS_WHOLE_FIELD_LIMIT:
        raise AssertionError(
            "the synthetic line's own maxVs/maxU already trips the bound; "
            "this test cannot demonstrate anything"
        )
    ratio_bad = 237.7
    if not (ratio_bad > ac.VS_WHOLE_FIELD_LIMIT):
        raise AssertionError(
            f"VS_WHOLE_FIELD_LIMIT is {ac.VS_WHOLE_FIELD_LIMIT}, which PASSES "
            f"the 237.7 measured on run 20260905-191135. The gate is inert."
        )
    ratio_survivor = 6.357
    if ratio_survivor > ac.VS_WHOLE_FIELD_LIMIT:
        raise AssertionError(
            f"the bound rejects run 20260905-113709 at {ratio_survivor}, the "
            f"longest-surviving two-velocity run. No clean run exists to "
            f"calibrate against, so the bound must not be set below the best "
            f"behaviour yet observed."
        )
    print(f"gate fires:   whole-field maxVs/maxU {ratio_bad} "
          f"(20260905-191135, runaway in the LUMEN)")
    print(f"gate passes:  {ratio_survivor} (20260905-113709, best so far); "
          f"clot-cell metrics alone would have passed BOTH")

    # 3. A missing key must RAISE, not quietly yield a passing dict.
    try:
        ac.parse_skeleton_velocity(
            "PHASE_CLOT_SKELETON_VELOCITY time=0.0095 clotCells=9577 "
            "meanVsOverU=1.4 maxVsOverU=3.2 exceedFraction=0.883 "
            "maxVs=1.2e-02 maxU=1.1e-02"
        )
    except ValueError as error:
        if "exceedFractionMargin" not in str(error):
            raise AssertionError(
                f"a line missing exceedFractionMargin raised for the wrong "
                f"reason: {error}"
            ) from error
        print("gate refuses: a line with no exceedFractionMargin is an error, "
              "not a pass")
    else:
        raise AssertionError(
            "parse_skeleton_velocity ACCEPTED a line with no "
            "exceedFractionMargin; the gate would report nothing exceeded on "
            "a run where the solver never printed the number"
        )

    # ---- THE CONSERVATION GATE: FIRES ON THE 4x, PASSES THE CLEAN RUNS,
    #      AND IS NOT GATED ON A QUANTITY THAT CHANGES SIGN ----
    #
    # Every number below is measured from logs on disk, not invented. The
    # gate this replaces did not exist: PHASE_CLOT_MASS had been printing a
    # closed budget every step for weeks and assess_case.py never parsed it,
    # so run 20260905-054441 grew the clot 4.0x with 98% of the growth in the
    # residual and was reported as a CFL crash.

    # S6c-2: THE TAG AND THE CURRENCY CHANGED, THE MEASUREMENT DID NOT.
    #
    # The numbers below are the measured em_2vel volumes and residuals in c
    # currency. The drift gate is a RATIO -- drift/V0 and V_final/V0 -- so it
    # is invariant to the constant theta_ref between the two currencies, and
    # the fixture keeps reproducing the measured +3.21 and 4.0x without
    # rescaling. That invariance is the reason this fixture is still valid
    # evidence and not a number carried across a definition change.
    def mass_line(t, vol, res, dvdt):
        return (
            f"PHASE_CLOT_CONTENT_BALANCE time={t} contentVolume={vol} "
            f"contentOuterFlux=0 contentTransferRate=0 "
            f"detachedContentRate=0 contentRemoved=0 thetaRef=0.45 "
            f"contentResidual={res} contentDVdt={dvdt}\n"
        )

    # 1. IT FIRES on run 20260905-054441 child em_2vel. Two samples whose
    #    trapezoid reproduces the measured driftFraction of +3.21.
    #    residual 3.05e-13 held over 0.01418 s against V0 = 1.348e-15.
    bad_text = (
        mass_line("1.506024096e-06", "1.348e-15", "8.38e-21", "3.58e-15")
        + mass_line("0.01417629179", "5.386e-15", "3.0567e-13", "1.49e-13")
    )
    bad_rows = ac.parse_clot_mass(bad_text)
    if len(bad_rows) != 2:
        raise AssertionError(
            f"parse_clot_mass found {len(bad_rows)} of 2 well-formed lines; "
            f"the conservation gate would silently pass on every run"
        )
    bad_drift = ac.clot_volume_drift(bad_rows)
    if not (abs(bad_drift["drift_fraction"]) > ac.CLOT_DRIFT_LIMIT):
        raise AssertionError(
            f"CLOT_DRIFT_LIMIT is {ac.CLOT_DRIFT_LIMIT}, which PASSES the "
            f"+3.21 drift measured on em_2vel, a run that grew the clot 4x. "
            f"The gate is inert. Got {bad_drift['drift_fraction']:+.4f}"
        )
    if not (3.0 < bad_drift["drift_fraction"] < 3.4):
        raise AssertionError(
            f"the drift integral does not reproduce the measured +3.21: "
            f"got {bad_drift['drift_fraction']:+.4f}"
        )
    if not (3.9 < bad_drift["volume_ratio"] < 4.1):
        raise AssertionError(
            f"volume_ratio should be the measured 4.0x, got "
            f"{bad_drift['volume_ratio']:.3f}"
        )
    print(f"gate fires:   em_2vel drift {bad_drift['drift_fraction']:+.2f} of "
          f"V0, clot at {bad_drift['volume_ratio']:.2f}x initial volume")

    # 2. IT DOES NOT FIRE on the five g-children, which conserve to the
    #    discretisation floor of -1e-05 under the corrected budget. Their
    #    carrier is solenoidal, and for a solenoidal carrier the bounded and
    #    conservative operators are identical, so this is the floor of the
    #    closure actually in use. A limit set anywhere near it would be
    #    unpassable, which this project has now produced four times.
    ok_text = (
        mass_line("1.5e-06", "1.348e-15", "-1e-21", "3.5e-15")
        + mass_line("0.03", "1.483e-15", "-4.4933e-19", "2.8e-15")
    )
    ok_drift = ac.clot_volume_drift(ac.parse_clot_mass(ok_text))
    if abs(ok_drift["drift_fraction"]) > ac.CLOT_DRIFT_LIMIT:
        raise AssertionError(
            f"the gate rejects the g-sweep, which conserves clot volume to "
            f"the 3e-4 discretisation floor. It is set below the numerics "
            f"and no correct run could pass. Got "
            f"{ok_drift['drift_fraction']:+.6f}"
        )
    if not (-1.5e-5 < ok_drift["drift_fraction"] < -0.7e-5):
        raise AssertionError(
            f"the synthetic does not reproduce the MEASURED -0.00001 floor, "
            f"so 'passes' here would be demonstrating nothing: got "
            f"{ok_drift['drift_fraction']:+.6f}"
        )
    print(f"gate passes:  the g-sweep floor "
          f"{ok_drift['drift_fraction']:+.5f} of V0 "
          f"({abs(ac.CLOT_DRIFT_LIMIT / ok_drift['drift_fraction']):.0f}x "
          f"margin)")

    # 3. IT DOES NOT FIRE on em_free's bondHold-carrier error (+0.00478).
    #    That is 480x the floor and a real defect, but its final volume ratio
    #    is 1.10 -- indistinguishable from the clean runs -- so failing it
    #    would discard a scientifically usable run. The fix for em_free is the
    #    carrier, not the threshold.
    ef_text = (
        mass_line("1.5e-06", "1.348e-15", "1e-19", "3.5e-15")
        + mass_line("0.03", "1.480e-15", "2.1477e-16", "-2.6e-15")
    )
    ef_drift = ac.clot_volume_drift(ac.parse_clot_mass(ef_text))
    if abs(ef_drift["drift_fraction"]) > ac.CLOT_DRIFT_LIMIT:
        raise AssertionError(
            f"the gate rejects em_free, whose 0.5% drift leaves the final "
            f"volume ratio at 1.10, the same as the clean runs. Got "
            f"{ef_drift['drift_fraction']:+.5f}"
        )
    print(f"gate passes:  em_free's bondHold error "
          f"{ef_drift['drift_fraction']:+.5f} of V0 (real, but harmless)")

    # 4. THE INSTANTANEOUS RATIO IS NOT THE GATE, and this is why. On the
    #    g-children |residual|/|dVdt| reaches 58.2 and on em_free 974, both
    #    while conserving volume -- dV/dt passes through zero and the ratio is
    #    evaluated on a sign change. That is the identical defect that made
    #    the first exceedFraction gate flaky.
    sign_change = ac.clot_volume_drift(ac.parse_clot_mass(
        mass_line("1e-06", "1.348e-15", "5.82e-18", "1e-19")
    ))
    if abs(sign_change["drift_fraction"]) > ac.CLOT_DRIFT_LIMIT:
        raise AssertionError(
            "a single step with |residual|/|dVdt| = 58 -- a sign change in "
            "dV/dt, not a conservation failure -- trips the gate. The gate "
            "has been written on the instantaneous ratio after all."
        )
    print("gate passes:  a step at |res|/|dVdt| = 58 (a dV/dt sign change)")

    # 5. A missing key must RAISE, not quietly yield a passing dict.
    try:
        ac.parse_clot_mass(
            "PHASE_CLOT_CONTENT_BALANCE time=0.03 contentVolume=1.48e-15 "
            "contentOuterFlux=0 contentTransferRate=0 "
            "detachedContentRate=0 thetaRef=0.45 contentDVdt=2.8e-15"
        )
    except ValueError as error:
        if "contentResidual" not in str(error):
            raise AssertionError(
                f"a line missing contentResidual raised for the wrong "
                f"reason: {error}"
            ) from error
        print("gate refuses: a line with no contentResidual is an error, "
              "not a pass")
    else:
        raise AssertionError(
            "parse_clot_mass ACCEPTED a line with no contentResidual; the "
            "would report perfect conservation on a run where the solver "
            "never printed the number"
        )

    # ---- THE Wp-q ENVELOPE GUARD ----
    #
    # An envelope, not a conservation law: WpDevMax = max|Wp - q| is exactly
    # zero only in the zero-diffusivity gate_a_exact test. Every number below
    # is measured from logs on disk.

    def plate_line(t, dev):
        return (
            f"PHASE_CLOT_PLATELET time={t} wMax=1 wMin=0.3 VnMagMax=3.75e-05 "
            f"wOverBound=0 WpDevMax={dev} qMin=1e-06 wDevBlood=0.1\n"
        )

    # 1. IT FIRES on run 20260904-214345 em_2vel, which reached 0.9926 -- w
    #    driven to ~0, total local platelet depletion.
    pk_bad = max(r["WpDevMax"] for r in ac.parse_platelet(
        plate_line("0.001", "0.0021") + plate_line("0.02", "0.9926")))
    if not (pk_bad > ac.WP_DEV_LIMIT):
        raise AssertionError(
            f"WP_DEV_LIMIT is {ac.WP_DEV_LIMIT}, which PASSES the 0.9926 "
            f"measured on run 20260904-214345. The guard is inert."
        )
    print(f"gate fires:   Wp-q peak {pk_bad:.4f} (20260904-214345 em_2vel)")

    # 2. IT DOES NOT FIRE on the widest one-velocity run measured, th_break /
    #    em_free at 0.1114. A limit below that would reject every main-case
    #    child ever run, since D > 0 and growth are both on by design.
    pk_ok = max(r["WpDevMax"] for r in ac.parse_platelet(
        plate_line("0.001", "0.0021") + plate_line("0.03", "0.1114")))
    if pk_ok > ac.WP_DEV_LIMIT:
        raise AssertionError(
            f"the guard rejects th_break/em_free at {pk_ok}, the widest "
            f"one-velocity value ever measured. It is set inside the normal "
            f"operating envelope and no main-case child could pass."
        )
    print(f"gate passes:  Wp-q peak {pk_ok:.4f} (widest one-velocity run, "
          f"{ac.WP_DEV_LIMIT / pk_ok:.1f}x margin)")

    # 3. A missing key must RAISE.
    try:
        ac.parse_platelet(
            "PHASE_CLOT_PLATELET time=0.03 wMax=1 wMin=0.3 VnMagMax=3.75e-05 "
            "wOverBound=0 qMin=1e-06 wDevBlood=0.1"
        )
    except ValueError as error:
        if "WpDevMax" not in str(error):
            raise AssertionError(
                f"a line missing WpDevMax raised for the wrong reason: {error}"
            ) from error
        print("gate refuses: a line with no WpDevMax is an error, not a pass")
    else:
        raise AssertionError(
            "parse_platelet ACCEPTED a line with no WpDevMax; the guard would "
            "report perfect agreement on a run that never printed it"
        )

    # ---- THE PHASE-BOUND GATE ----
    #
    # Every number measured from logs on disk. This gate is the direct guard
    # on the failure of run 20260905-113709, whose cUnder the solver printed
    # 1855 times while nothing read it.

    # S6c-2: the fixture still speaks in c, because that is the scale the
    # 1.7788 measurement and C_BOUND_LIMIT live on. It converts to q on the
    # way out, exactly as the solver now emits, and assess_case converts
    # back on the way in. If either conversion were dropped the gate would
    # silently change by a factor of two, which is what this round trip
    # exists to prevent.
    def pb_line(t, cmin, cmax=1.0):
        qmin = 0.5*(cmin + 1.0)
        qmax = 0.5*(cmax + 1.0)
        return (
            f"PHASE_CLOT_PHASE_BOUNDS time={t} qMin={qmin} qMax={qmax} "
            f"qUnder={max(0.0, -qmin)} qOver={max(0.0, qmax - 1.0)}\n"
        )

    # 1. IT FIRES on run 20260905-113709: cMin -2.7788, cUnder 1.7788.
    bad_pb = ac.parse_phase_bounds(
        pb_line("1.5e-06", -0.9999994) + pb_line("0.00279", -2.7788))
    worst_bad = max(max(r["cUnder"], r["cOver"]) for r in bad_pb)
    if not (worst_bad > ac.C_BOUND_LIMIT):
        raise AssertionError(
            f"C_BOUND_LIMIT is {ac.C_BOUND_LIMIT}, which PASSES the 1.7788 "
            f"cUnder measured on run 20260905-113709. The gate is inert."
        )
    print(f"gate fires:   cUnder {worst_bad:.4f} (20260905-113709 em_2vel)")

    # 2. IT DOES NOT FIRE on the round-off every healthy run carries. The
    #    worst cOver ever measured is 1.09e-13 (g30); cUnder is EXACTLY 0 on
    #    all eleven other children.
    ok_pb = ac.parse_phase_bounds(
        pb_line("1.5e-06", -0.9999994) + pb_line("0.03", -1.0, 1.000000000000109))
    worst_ok = max(max(r["cUnder"], r["cOver"]) for r in ok_pb)
    if worst_ok > ac.C_BOUND_LIMIT:
        raise AssertionError(
            f"the gate rejects the 1.09e-13 round-off every healthy run "
            f"carries. Got {worst_ok:.3e}"
        )
    print(f"gate passes:  round-off {worst_ok:.3e} "
          f"({ac.C_BOUND_LIMIT / max(worst_ok, 1e-300):.1e}x margin)")

    # 3. IT DOES NOT FIRE at a physically negligible overshoot. c = -1.02 is
    #    chi = 1.01, i.e. phi_s 1% above nominal. c is now transported by a
    #    CONSERVATIVE scheme with no boundedness guarantee, so a gate at
    #    machine zero would be unpassable by construction.
    small_pb = ac.parse_phase_bounds(pb_line("0.03", -1.02))
    if max(r["cUnder"] for r in small_pb) > ac.C_BOUND_LIMIT:
        raise AssertionError(
            "the gate rejects c = -1.02, a 1% density error, on a scheme that "
            "carries no boundedness guarantee. It is set below the numerics."
        )
    print("gate passes:  c = -1.02 (chi 1.01, a 1% density error)")

    # 4. A missing key must RAISE.
    try:
        ac.parse_phase_bounds(
            "PHASE_CLOT_PHASE_BOUNDS time=0.03 qMin=0.0 qMax=1.0 qOver=0"
        )
    except ValueError as error:
        if "qUnder" not in str(error):
            raise AssertionError(
                f"a line missing qUnder raised for the wrong reason: {error}"
            ) from error
        print("gate refuses: a line with no qUnder is an error, not a pass")
    else:
        raise AssertionError(
            "parse_phase_bounds ACCEPTED a line with no qUnder; the gate would "
            "report c in bounds on a run that never printed it"
        )

    # ---- S17: THE Vs SOLVER REFUSAL AND THE S17 GATES, DIRECTLY ----
    #
    # implicitTangent without a Vs entry cannot be produced from the controls
    # (the generator writes Vs whenever it writes implicitTangent), so the
    # check that stands between a template change and a fatal first solve is
    # exercised on the function itself. The S17 assessor gates each FAIL on
    # their negative line and PASS on the control (the full set, with a
    # mutation per gate, is scripts/check_s17_fibre_release.py).
    # Loaded under its own name: create_cases imports the mural gate's
    # create_cases as mural_gen, and a plain `import create_cases` here would
    # hand it itself.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "create_cases_s17_dry", Path(__file__).resolve().parent / "create_cases.py")
    cc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(cc)

    def s17_vs_entry_refused():
        with tempfile.TemporaryDirectory(prefix="am2020-s17-vs-") as tmp:
            sol = Path(tmp) / "fvSolution"
            names = ["p", "U", "c", "w", "Wp", "Dg", "Us", "be", "thetaC",
                     "thetaH", "thetaNetRef", "ruptured", "thetaIntact",
                     "thetaPending", "thetaBroken", "thetaPendingB",
                     "thetaPendingP"]
            sol.write_text("solvers\n{\n" + "".join(
                f"    {n} {{ solver PCG; }}\n    {n}Final {{ solver PCG; }}\n"
                for n in names) + "}\n")
            try:
                cc.verify_linear_solvers_exist(
                    sol, False, True, rupture=True, partition=True,
                    fibre=True, implicit_tangent=True)
            except (SystemExit, ValueError) as error:
                # verify_linear_solvers_exist refuses with ValueError, which
                # main() turns into the generator's SystemExit.
                assert "no linear solver declared for Vs, VsFinal" in str(error), str(error)
                return
            raise AssertionError("an fvSolution without Vs was ACCEPTED "
                                 "under implicitTangent")
    run_direct("s17_implicit_tangent_without_a_Vs_entry_is_refused",
               s17_vs_entry_refused)

    def s17_gates():
        good = "PHASE_CLOT_S17_CONFIG pendingForm=fibre\n"   # S21: .C:2341
        row = ("PHASE_CLOT_S17 time=1.5e-03 step=10 deltaT=4e-07 "
               "transferB1Max={tr} transfers=3 resetCells=1 "
               "rotRatioMedian={rot} rotCells=5 rotLdtMedian=4e-4 "
               "rotCandidates=6 eigRelMax={eig} uniUp={uni} "
               "projCount=8 projRaised={proj} projExcessMax=0 projTies=0 "
               "splitDevMax0={sp} splitDevMax1=0 splitDevMax2=0 splitSites=7 "
               "pendingCells=4 p2Max={p2} w2Max=0.9 mixedCells=1 "
               "w1MaxDeg=0.3 w1Over5=0 w1Over10=0 pendingPressureMax=0 "
               "elasticCoupling=implicitTangent nuElMax=3 nuNetMax=0.1\n")
        spec = {"child_id": "em66_G400_s17",
                "s17": {"pendingForm": "fibre", "nCorrectors": 6}}
        base = dict(tr=1.004, rot=1.00003, eig=4e-16, uni=0, proj=0, sp=0,
                    p2=0.97)
        r = ac.s17_report(good + row.format(**base), spec)
        for g in ("g_tr", "g_rot", "g_eig", "g_uni", "g_proj", "s17split",
                  "p2", "config_matches_child"):
            assert r[g] is True, (g, r[g])
        for key, bad, gate in (("tr", 1.2, "g_tr"), ("rot", 2.0, "g_rot"),
                               ("eig", 1e-9, "g_eig"), ("uni", 1, "g_uni"),
                               ("proj", 2, "g_proj"), ("sp", 0.3, "s17split"),
                               ("p2", 1.3, "p2")):
            r = ac.s17_report(good + row.format(**{**base, key: bad}), spec)
            assert r[gate] is False, (gate, r[gate])
        r = ac.s17_report(good.replace("pendingForm=fibre", "pendingForm=none")
                          + row.format(**base), spec)
        assert r["config_matches_child"] is False, r["config_mismatched_keys"]
    run_direct("s17_gates_fail_on_their_negatives", s17_gates)

    # ---- S17b: THE TAUT SHARE'S SOLVER ENTRY AND THE S17b GATES ----
    # fibre_slack v1 without a thetaTaut entry cannot be produced from the
    # controls (the generator writes it with v1), so the check is exercised
    # on the function. The S17b gates each FAIL on their negative and PASS
    # on the control (the full set: scripts/check_s17b_slack_fibre.py).
    def s17b_taut_entry_refused():
        with tempfile.TemporaryDirectory(prefix="am2020-s17b-tt-") as tmp:
            sol = Path(tmp) / "fvSolution"
            names = ["p", "U", "c", "w", "Wp", "Dg", "Us", "be", "thetaC",
                     "thetaH", "thetaNetRef", "ruptured", "thetaIntact",
                     "thetaPending", "thetaBroken", "thetaPendingB",
                     "thetaPendingP", "Vs"]
            sol.write_text("solvers\n{\n" + "".join(
                f"    {n} {{ solver PBiCGStab; }}\n    {n}Final {{ solver PBiCGStab; }}\n"
                for n in names) + "}\n")
            try:
                cc.verify_linear_solvers_exist(
                    sol, False, True, rupture=True, partition=True,
                    fibre=True, implicit_tangent=True, slack_v1=True)
            except (SystemExit, ValueError) as error:
                assert "no linear solver declared for thetaTaut, thetaTautFinal" in str(error), str(error)
                return
            raise AssertionError("an fvSolution without thetaTaut was ACCEPTED "
                                 "under fibre_slack v1")
    run_direct("s17b_v1_without_a_thetaTaut_entry_is_refused",
               s17b_taut_entry_refused)

    def s17b_gates():
        cfg = ""        # S21: no S17B_CONFIG line; fibreSlack v1 is built in
        row = ("PHASE_CLOT_S17B time=1.5e-03 step=10 slackEvents=2 "
               "slackDdMax={dd} slackMuMax={mu} slackAtRest=0 "
               "slackQpMax={qp} slackPreTRatioMax=0.5 slackPreQpRatioMax=0.2 "
               "tautAboveCells={above}\n"
               "PHASE_CLOT_S17 time=1.5e-03 step=10 transferB1Max=1.3 "
               "transfers=3 p2Max={p2} w2Max={w2}\n"
               "PHASE_CLOT_VS_SPLIT time=1.5e-03 slip={slip}\n"
               "PHASE_CLOT_FACE_SKELETON time=1.5e-03 maxFaceVs=0.01\n")
        spec = {"child_id": "em76_G4_s18"}       # S21: child.json has no s17b block
        base = dict(dd=0, mu=0.99, qp=0, above=0, p2=0.97, w2=0.97, slip=0.05)
        r = ac.s17b_report(cfg + row.format(**base), spec)
        assert r["config_matches_child"] is None, r["config_matches_child"]
        for g in ("slack_dd", "taut_le_pending", "w2", "p2", "g8"):
            assert r[g] is True, (g, r[g])
        # design section 8 d: |dd| is the gate; the rest are conformance
        for key, bad, gate in (("dd", 1e-300, "slack_dd"),
                               ("above", 1, "taut_le_pending"),
                               ("w2", 0.98, "w2"), ("p2", 0.98, "p2"),
                               ("slip", 0.11, "g8")):
            r = ac.s17b_report(cfg + row.format(**{**base, key: bad}), spec)
            assert r[gate] is False, (gate, r[gate])
        r = ac.s17b_report(cfg + row.format(**{**base, "qp": 5e-324, "mu": 1.0}), spec)
        assert r["slack_dd"] is True and r["conformance_slack"] == {
            "qP_zero_after": False, "mu_below_1_or_at_rest": False}, r["conformance_slack"]
        # G-tr is a watch under S17b: a transfer b1/lc^2 of 1.3 is reported
        assert r["watch_g_tr_max"] == 1.3, r["watch_g_tr_max"]
    run_direct("s17b_gates_fail_on_their_negatives", s17b_gates)

    # ---- S18: THE em76 CLAUSES, DIRECTLY (design section 5) ----
    # The scored clauses FAIL on their negatives and PASS on the control;
    # the em72-referenced ones are NOT-EVALUABLE, never PASS; frozenPower
    # nonzero under A is a conformance breach (the full set, with a mutation
    # per switch: scripts/check_s18_soft.py).
    def s18_gates():
        spec = {"child_id": "em76_G4_s18", "s17": {"nCorrectors": 6},
                "s18": {"networkPressureWeight": "contentRatio"}}

        def log(slip=0.001, fp=0.0, ws=0.0, x2=16e-6, dtlim=1.0):
            out = ["PHASE_CLOT_S18_CONFIG networkPressureWeight=contentRatio\n",
                   "PHASE_CLOT_MOMENTUM time=0 clotXbar=0\n"]
            for k, x in ((1, 7e-6), (2, x2)):
                t = k*1e-3
                out.append(f"PHASE_CLOT_VS_SPLIT time={t} slip={slip}\n"
                           f"PHASE_CLOT_S17 time={t} projRaised=0\n"
                           f"PHASE_CLOT_S18 time={t} deltaT=4.17e-6 carrierDtLimit={dtlim} "
                           f"coSkel=0.01 wallSplitDevMax={ws} netCells=10 sNegCells=0 "
                           f"frozenPower={fp}\n"
                           f"PHASE_CLOT_MOMENTUM time={t} clotXbar={x}\n")
            return "".join(out)
        r = ac.s18_report(log(), spec)
        assert r["scored_ok"] is True and r["conformance_ok"] is True, r
        for k in ("onset_window", "tearing_vs_em72", "debond_watch_vs_em72"):
            assert r[k] == ac.NOT_EVALUABLE, (k, r[k])
        for kw, key in ((dict(slip=0.1000001), "g8"), (dict(x2=1e-6), "moved_gate"),
                        (dict(dtlim=4.17e-6), "dt_no_carrier_bound")):
            r = ac.s18_report(log(**kw), spec)
            assert r[key] is False and r["scored_ok"] is False, (key, r[key])
        r = ac.s18_report(log(fp=1e-15), spec)
        assert r["conformance"]["frozenPower_zero"] is False, r["conformance"]
        r = ac.s18_report(log(ws=1e-12), spec)
        assert r["conformance"]["wall_patch_split_holds"] is False, r["conformance"]
    run_direct("s18_gates_fail_on_their_negatives", s18_gates)

    # ---- S18p: V2 DIRECTLY (design section 5; report-only) ----
    # One record per step and T + P = S17's totals PASS on the control and
    # FAIL on a missing, a duplicated or a mis-summed record; em78's
    # non-empty clause FAILS on an instrument that never fires after 1.2 ms.
    def s18p_v2():
        spec = {"child_id": "em78_G4_s18p_ctl", "s18p": {"projSiteLog": "yes"}}

        def log(n=4, drop=None, dup=None, bump=None, raised=3):
            out = ["PHASE_CLOT_S18P_CONFIG projSiteLog=yes\n"]
            for k in range(1, n + 1):
                t = k*0.5e-3
                rT = raised if t > 1.2e-3 else 0
                out.append(f"PHASE_CLOT_S17 time={t} step={k} projCount={10 + rT} "
                           f"projRaised={rT + 1}\n")
                if k == drop:
                    continue
                cP = 4 + (1 if k == bump else 0)
                line = (f"PHASE_CLOT_S18P time={t} step={k} projCountT={6 + rT} "
                        f"projRaisedT={rT} projRaisedPresentT=0 projRaisedAbsentT={rT} "
                        f"projExcessMaxT=0 projCountP={cP} projRaisedP=1 "
                        f"projRaisedPresentP=0 projRaisedAbsentP=1 projExcessMaxP=0 "
                        f"networkCells=100 drainedCells=2\n")
                out.append(line + (line if k == dup else ""))
            out.append("PHASE_CLOT_S18P_CELL time=0.002 step=4 site=P cell=7 present=0 "
                       "th=1 kernel=passengersShared kept=1 inB=0 Bdiv=1e-301 "
                       "PkAfterC=1 PkNew=1\n")
            return "".join(out)
        r = ac.s18p_report(log(), spec)
        assert r["v2"] is True and r["config_matches_child"] is True, r
        assert r["ab_route"] == "supported" and r["dumped_cells"] == {"T": 0, "P": 1}, r
        for kw, key in ((dict(drop=2), "one_record_per_step"),
                        (dict(dup=3), "one_record_per_step"),
                        (dict(bump=4), "t_plus_p_equals_totals")):
            r = ac.s18p_report(log(**kw), spec)
            assert r[key] is False and r["v2"] is False, (kw, key, r[key], r["v2"])
        r = ac.s18p_report(log(raised=0).replace("projRaisedP=1", "projRaisedP=0")
                           .replace("projRaised=1", "projRaised=0"), spec)
        assert r["non_empty_after_1p2ms"] is False and r["v2"] is False, r
        print("   S18p V2 fails on a missing, a duplicated, a mis-summed and an empty record")
    run_direct("s18p_v2_fails_on_its_negatives", s18p_v2)

    # ---- XU 2017 FIGURE 5 (batch 61): the generator's refusals, each for
    # its stated reason, and the assessor's section 4 rows on a healthy
    # synthetic run with one negative per row (module-level functions above).
    run_direct("xu_generator_negatives_are_refused", xu_generator_negatives)
    run_direct("xu_report_rows_fail_on_their_negatives", xu_gates)

    # ---- THE LAUNCH COMMAND ITSELF (fix round 13c, docket 19) ----
    #
    # Batch 59 died in create_cases.py before any child ran (Colab run
    # 20260928-120134): the notebook's batch filter built 4 children, and
    # the S17 control check indexed the FILTERED child.json set by a batch-58
    # child id (KeyError 'em69_G400_s17_noimpl'). Every check here generated
    # the FULL matrix, so nothing ran the command the notebook runs. This
    # reads the CURRENT batch's filter out of the notebook (parsed with
    # check_notebook_batch_selector's own parser, never copied) and runs
    # create_cases.py with exactly that filter on a fresh copy; then batch
    # 58's filter; then three negatives, each refused for its stated reason.
    _nbs = importlib.util.spec_from_file_location(
        "nb_selector_dry",
        Path(__file__).resolve().parents[3] / "scripts" / "check_notebook_batch_selector.py")
    nbsel = importlib.util.module_from_spec(_nbs)
    _nbs.loader.exec_module(nbsel)
    _batch, _batches = nbsel.parse(nbsel.source_of(
        Path(__file__).resolve().parents[3] / "notebooks" / "AM2020_main_case_colab.ipynb"))

    _filtered_generation = filtered_generation   # module level: a docket probe calls it directly

    def launch_current_batch():
        filt = _batches[_batch].split()
        rc, built, last = _filtered_generation(filt)
        assert rc == 0, f"batch {_batch}'s launch filter was refused: {last}"
        assert sorted(built) == sorted(filt), (built, filt)
        if _batch in nbsel.notebook_names(nbsel.launch_cell(Path(__file__).resolve().parents[3] / "notebooks" / "AM2020_main_case_colab.ipynb"))["ONE_CORE_BATCHES"]:
            assert built == filt, f"batch {_batch}: Allrun would run {built} (matrix order), BATCHES names {filt}"
        print(f"   batch {_batch}: {len(filt)} children built, exactly the filter ({last})")
    run_direct("the_notebook_batch_filter_generates", launch_current_batch)

    run_direct("the_notebook_launch_filters_generate", lambda: launch_filters_generate(nbsel, _batch, _filtered_generation))
    if _batch != 72:
        # BATCH 72's two launches stay exercised while a later batch is the default
        run_direct("batch_72_launch_filters_generate", lambda: launch_filters_generate(nbsel, 72, _filtered_generation))
    if _batch != 73:
        # BATCH 73's one-session filter stays exercised while a later batch is the default
        run_direct("batch_73_launch_filters_generate", lambda: launch_filters_generate(nbsel, 73, _filtered_generation))
    if _batch != 74:
        # BATCH 74's one-session filter (BATCH 74's record) stays exercised while a later batch is the default
        run_direct("batch_74_launch_filters_generate", lambda: launch_filters_generate(nbsel, 74, _filtered_generation))
    if _batch != 75:
        # BATCH 75's one-session filter stays exercised while a later batch is the default
        run_direct("batch_75_launch_filters_generate", lambda: launch_filters_generate(nbsel, 75, _filtered_generation))
    if _batch != 76:
        # BATCH 76's three launches (1-core: each filter generates by itself and in the notebook's order) stay exercised while a later batch is the default
        run_direct("batch_76_launch_filters_generate", lambda: launch_filters_generate(nbsel, 76, _filtered_generation))

    if _batch != 78:
        # BATCH 78's one-session filter (1-core, the baseline library) stays exercised while a later batch is the default
        run_direct("batch_78_launch_filters_generate", lambda: launch_filters_generate(nbsel, 78, _filtered_generation))

    if _batch != 79:
        # BATCH 79's one-child filter (the Y gate, 1-core) stays exercised while a later batch is the default
        run_direct("batch_79_launch_filters_generate", lambda: launch_filters_generate(nbsel, 79, _filtered_generation))
    if _batch != 80:
        # BATCH 80's two launches (the flow gates and the restart pair) stay exercised while a later batch is the default
        run_direct("batch_80_launch_filters_generate", lambda: launch_filters_generate(nbsel, 80, _filtered_generation))
    if _batch != 82:
        # BATCH 82's two launches (the confirming restart test, VOID: leg 1, then leg 2, the same child) stay exercised while a later batch is the default
        run_direct("batch_82_launch_filters_generate", lambda: launch_filters_generate(nbsel, 82, _filtered_generation))
    if _batch != 83:
        # BATCH 83's two launches (the corrected retest: the continuous arm and leg 1 in one launch, then leg 2) stay exercised while a later batch is the default
        run_direct("batch_83_launch_filters_generate", lambda: launch_filters_generate(nbsel, 83, _filtered_generation))
    # BATCH 81 (the plug chains, STAGED: never the default): each chain's filter generates by itself and in the notebook's order, whichever batch BATCH selects
    run_direct("batch_81_launch_filters_generate", lambda: launch_filters_generate(nbsel, 81, _filtered_generation))

    def one_core_launch():
        # S25 (the 1-core runtime batches 76/77): BATCHES[77] has no entry until its children are written, so the
        # notebook's own launch_plan runs on a synthetic one-child entry for each (the runner's argparse parses the exact argv, which
        # carries --mkl-threads 1 and NO --mkl-cbwr, design section 12.1), and that filter generates by itself, as the cell hands it to Allrun.
        argvs = nbsel.check_one_core(nbsel.launch_cell(
            Path(__file__).resolve().parents[3] / "notebooks" / "AM2020_main_case_colab.ipynb"), nbsel.CONTROLS)
        assert sorted(argvs) == [76, 77, 78, 79, 80, 81, 82, 83], sorted(argvs)
        filt = ["P73_G4_r3_40"]
        rc, built, last = _filtered_generation(filt)
        assert rc == 0, f"the 1-core stand-in filter was refused: {last}"
        assert sorted(built) == filt, (built, filt)
        print(f"   1-core batches {sorted(argvs)}: argv carries --mkl-threads 1 and no MKL_CBWR pin; the filter builds exactly {filt} ({last})")
    run_direct("one_core_launch_filter_generates", one_core_launch)

    def launch_batch_60():
        filt = _batches[60].split()
        rc, built, last = _filtered_generation(filt)
        assert rc == 0, f"batch 60's filter was refused: {last}"
        assert filt == ["em76_G4_s18"] and built == filt, (built, filt)
        print(f"   batch 60: {len(filt)} child built ({last})")
    run_direct("batch_60_filter_generates", launch_batch_60)
    run_direct("reference_library_wiring", reference_library_wiring)

    # S21: batches 58 and 59 and their S17 control pairs are retired children, so
    # their filter tests are cut; a filter naming a retired child is refused.
    def launch_negatives():
        rc, _built, last = _filtered_generation(filt=["em72_G4_s17b"])
        assert rc != 0, "a filter naming a retired child was ACCEPTED"
        assert "matched no child" in last, f"refused for the WRONG reason: {last}"
        print(f"   refused: a retired child in the filter -> {last[:120]}")
    run_direct("launch_filter_negatives_are_refused", launch_negatives)
    run_direct("free_shell_watcher_fires_and_stays_silent",
               free_shell_watcher_fixtures)

    print("All main-case reporting paths passed offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
