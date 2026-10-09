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
) -> None:
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
    while value <= spec["end_time_s"] + 1e-12:
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
        value = round(value + spec["write_interval_s"], 10)

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

    times = []
    value = spec["write_interval_s"]
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
     "solver_library 'p1' is not one of p2prime, s22 (S22 section 7)."),
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
    ("S23v_G4_r3_S22 without its named key solver_library",
     lambda c: _s16(c, "S23v_G4_r3_S22", solver_library=None),
     "FSF parent check: S23v_G4_r3_S22 does not differ from S23v_G4_r3 in its named key solver_library"),
    ("S23v_G4_r3_S22 not ending at 30 ms (no end_time_s difference)",
     lambda c: _s16(c, "S23v_G4_r3_S22", end_time_s=0.04),
     "FSF parent check: S23v_G4_r3_S22 does not differ from S23v_G4_r3 in its named key end_time_s"),
    ("an unknown solver_library on the S23 reference child",
     lambda c: _s16(c, "S23v_G4_r3_S22", solver_library="s21"),
     "solver_library 's21' is not one of p2prime, s22 (S22 section 7)."),
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


def s23_solver_would_refuse(props_text: str):
    """The constructor's R1-R4 and marker-with-kelvinVoigt FatalErrors, read off phaseClotProperties text the way
    lookupOrDefault reads it (solver-engineer report 24; phaseClotAxisymmetricFluid.C, 'S23 rev. 4.1 section 13.2
    item 4'). Returns None (accepted) or 'R1' (marker not tanDelta0), 'MK' (marker with a non-none networkViscosity),
    'R2' (none, unmarked), 'R3' (kelvinVoigt without networkViscosityTau), 'R4' (tau_v < 5e-4 or not a number)."""
    text = re.sub(r"//[^\n]*", "", props_text)

    def entry(key):
        m = re.search(rf"(?m)^\s*{key}\s+([^;\s]+)\s*;", text)
        return None if m is None else m.group(1)
    marker, nv, tau = entry("sensitivityLimit"), entry("networkViscosity") or "none", entry("networkViscosityTau")
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
        if not (float(tau) >= 5e-4):
            return "R4"
    return None


def s23_solver_mirror_checks() -> None:
    """(a) The mirror above is pinned to the solver's SOURCE (the floor literal, the comparison, the marker word and key,
    the default); (b) report 24's negative dictionaries, one each, and the accepted ones, run through the mirror."""
    assert S23_SOLVER_C.is_file(), f"the solver source is absent: {S23_SOLVER_C}"
    src = S23_SOLVER_C.read_text()
    pins = ('const scalar s23TauVFloor = 5e-4;', '!(networkViscosityTau_ >= s23TauVFloor)',
            'sensitivityLimit_ != "tanDelta0"', 'lookupOrDefault<word>("sensitivityLimit", "none")',
            'lookupOrDefault<word>("networkViscosity", "none")', '!phaseClotProperties_.found("networkViscosityTau")',
            'if (s23Marked && networkViscosity_ != "none")', 'if (networkViscosity_ == "none" && !s23Marked)')
    stale = lambda s: [pin for pin in pins if s.count(pin) != 1]
    assert not stale(src), f"the solver source no longer has exactly one of {stale(src)}: the dry-run mirror is stale"
    # the pin check can fail: each drift of the solver the mirror would not follow is seen (a scratch copy of the text)
    for what, drifted in (("the floor 4.99e-4", src.replace('s23TauVFloor = 5e-4;', 's23TauVFloor = 4.99e-4;')),
                          ("> instead of >=", src.replace('networkViscosityTau_ >= s23TauVFloor', 'networkViscosityTau_ > s23TauVFloor')),
                          ("a second marker value", src.replace('sensitivityLimit_ != "tanDelta0"', 'sensitivityLimit_ != "tanDelta1"')),
                          ("the marker allowed with kelvinVoigt", src.replace('if (s23Marked && networkViscosity_ != "none")', 'if (false)')),
                          ("the R3-required refusal removed", src.replace('if (networkViscosity_ == "none" && !s23Marked)', 'if (false)'))):
        assert drifted != src and stale(drifted), f"the pin check did not see {what}"
    gen = (CASE / "scripts" / "create_cases.py").read_text()
    assert "S23_TAU_V_FLOOR_S = 5e-4" in gen and 'S23_SENSITIVITY_VALUES = ("tanDelta0",)' in gen
    base = ("networkViscosity kelvinVoigt;\nnetworkViscosityTau 0.0015;\n")
    cases = (
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
    for label, text, want in cases:
        got = s23_solver_would_refuse(text)
        assert got == want, f"{label}: the mirror says {got!r}, report 24 says {want!r}"
    print(f"   the solver's R1-R4 and marker+kelvinVoigt mirror is pinned to its source ({len(cases)} dictionaries, both sides, "
          f"5e-4 and 0.0005 accepted, 4.99e-4 refused)")


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
            controls_path.write_text(pristine)
        done = subprocess.run([sys.executable, "-B", str(root / "scripts" / "create_cases.py")],
                              capture_output=True, text=True)
        assert done.returncode == 0, f"the pristine controls were refused: {(done.stdout + done.stderr)[-300:]}"
        props = lambda cid: (root / "runs" / cid / "constant" / "phaseClotProperties").read_text()
        # the key is written for NO child that does not name it (all four BATCH 72 children and every earlier one)
        every = sorted(p.name for p in (root / "runs").iterdir() if (p / "child.json").is_file())
        assert len(every) == KEPT_CHILDREN, (len(every), KEPT_CHILDREN)
        assert not [c for c in every if "volumetricTearRatio" in props(c)], "volumetricTearRatio written for a child that does not name it"
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
        for c in ("P73_G4_r3_40", "P73_G40_40"):
            assert not (root / "runs" / c / "child.env").exists(), c
            assert "solver_library" not in json.loads((root / "runs" / c / "child.json").read_text()), c
        # the marker is written for ONE child of the 30 (the whitelist), as a word, with nothing else near it
        assert [c for c in every if "sensitivityLimit" in props(c)] == ["S23v_G4_nor3"]
        assert len(re.findall(r"(?m)^\s*sensitivityLimit\s+tanDelta0;\s*$", props("S23v_G4_nor3"))) == 1
        # child.env: the 2 earlier library children and the 13 pinned ones (item 5), nothing else
        pinned = sorted(S23_PINNED_S22)
        assert sorted(c for c in every if (root / "runs" / c / "child.env").exists()) == sorted(["S22v_G40_A0", "S23v_G4_r3_S22"] + pinned)
        for c in ["S23v_G4_r3_S22"] + pinned:
            assert json.loads((root / "runs" / c / "child.json").read_text())["solver_library"] == "s22", c
        assert [c for c in every if json.loads((root / "runs" / c / "child.json").read_text()).get("solver_library") == "s22"] == sorted(["S23v_G4_r3_S22"] + pinned)
        # nOuterCorrectors is 1 on the four arms (the end-of-step partition rule of the log; section 13.6 hand-off)
        for c in ("S23v_G4_r3", "S23v_G4_r3_S22", "S23v_G4_nor3", "S23v_G40", "P73_G4_r3_40", "P73_G40_40"):
            fvs = (root / "runs" / c / "system" / "fvSolution").read_text()
            assert re.findall(r"(?m)^\s*nOuterCorrectors\s+(\S+);", fvs) == ["1"], (c, re.findall(r"nOuterCorrectors\s+\S+;", fvs))
        # the solver's own R1-R4 + marker/kelvinVoigt predicate (mirrored from its source text, below), applied to
        # every generated dictionary: a child on the S23 build is accepted, a pinned child is exactly the kind the
        # build would refuse (so the pin is necessary), and no child is both on the build and refused
        for c in every:
            lib = json.loads((root / "runs" / c / "child.json").read_text()).get("solver_library")
            why = s23_solver_would_refuse(props(c))
            if lib is None:
                assert why is None, (c, why)
            elif c in pinned:
                assert why in ("R2", "R3"), (c, why)
        assert {c: s23_solver_would_refuse(props(c)) for c in pinned}.get("fsb_ctl_r3") == "R3"
        assert sum(s23_solver_would_refuse(props(c)) == "R2" for c in pinned) == 12
        s23_solver_mirror_checks()
        # production write settings on all four (5 ms full writes, 0.2 ms frames)
        ends = {"S23v_G4_r3": 0.04, "S23v_G4_r3_S22": 0.03, "S23v_G4_nor3": 0.04, "S23v_G40": 0.025}
        for cid, end in ends.items():
            j = json.loads((root / "runs" / cid / "child.json").read_text())
            cd = (root / "runs" / cid / "system" / "controlDict").read_text()
            assert f"endTime         {end};" in cd or re.search(rf"endTime\s+{end};", cd), (cid, "endTime")
            assert re.search(r"writeInterval\s+0\.005;", cd), (cid, "writeInterval")
        # child.env, SOURCED the way Allrun does (bash, `.`), each refusal seen to fire
        envs = {"s22": (root / "runs" / "S23v_G4_r3_S22" / "child.env", "PHASE_CLOT_S22_LIBBIN", "PHASE_CLOT_P2PRIME_LIBBIN"),
                "p2prime": (root / "runs" / "S22v_G40_A0" / "child.env", "PHASE_CLOT_P2PRIME_LIBBIN", "PHASE_CLOT_S22_LIBBIN")}
        good = Path(tmp) / "libbin_good"
        good.mkdir()
        (good / "libphaseClotAxisymmetricFluid.so").write_bytes(b"x")
        bare = Path(tmp) / "libbin_empty"
        bare.mkdir()

        def source(envfile, setting, ld="/opt/foam/lib"):
            cmd = f'unset PHASE_CLOT_S22_LIBBIN PHASE_CLOT_P2PRIME_LIBBIN; {setting} LD_LIBRARY_PATH={ld}; ' \
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
            if name == "s22":
                rc, ld, err = source(envfile, f"export {var}='{Path(tmp) / 'nonexistent'}';")
                assert rc == 98 and ld == "/opt/foam/lib", (name, "nonexistent dir", rc, ld)
                rc, ld, err = source(envfile, f"export {var}='{bare}';")
                assert rc == 98 and ld == "/opt/foam/lib", (name, "dir without the library", rc, ld)
            rc, ld, err = source(envfile, f"export {var}='{good}';")
            assert rc == 0 and ld == f"{good}:/opt/foam/lib", (name, "valid", rc, ld)
            print(f"child.env ({name}): unset -> 97, empty -> 97, the other library's variable -> 97"
                  + (", missing dir -> 98, library-less dir -> 98" if name == "s22" else "") + ", valid -> 0 with LD_LIBRARY_PATH prefixed")
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
        print(f"   batch {batch} launch {lnum}: {len(filt)} children built, exactly the launch's filter ({last})")


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
    assert batch == max(batches) == 73, (batch, max(batches))
    # the build script re-pins by exactly these two patterns; a second match would break the release
    assert len(re.findall(r"EXPECTED_REPO_DIGEST = '[0-9a-f]{64}'", text)) == 1
    assert len(re.findall(r"EXPECTED_TAG = 'digest-[0-9a-f]{12}'", text)) == 1
    pins = dict(re.findall(r"^(P2PRIME_TAG|P2PRIME_REPO_DIGEST|S22REF_TAG|S22REF_REPO_DIGEST) = '([^']+)'", text, re.M))
    assert set(pins) == {"P2PRIME_TAG", "P2PRIME_REPO_DIGEST", "S22REF_TAG", "S22REF_REPO_DIGEST"}, pins
    for t, d in (("P2PRIME_TAG", "P2PRIME_REPO_DIGEST"), ("S22REF_TAG", "S22REF_REPO_DIGEST")):
        assert re.fullmatch(r"[0-9a-f]{64}", pins[d]) and pins[t] == "digest-" + pins[d][:12], (t, pins[t], pins[d])
    assert pins["S22REF_TAG"] == "digest-442b95afaec8", "BATCH 72's reference is the S22 release (design section 7)"
    table = re.search(r"(?s)REFERENCE_LIBRARY_BY_BATCH = (\{.*?\n\})", text).group(1)
    assert "P2PRIME_TAG" in table and "S22REF_TAG" in table
    flags = dict(re.findall(r"(\d+): \('(\w+)', \w+, \w+, '(--[\w-]+)'", table) and
                 [(b, f"{n}|{fl}") for b, n, fl in re.findall(r"(\d+): \('(\w+)', \w+, \w+, '(--[\w-]+)'", table)])
    assert flags == {"71": "p2prime|--p2prime-solver", "72": "s22|--s22-solver"}, flags
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
    by = {(r["batch"], r["launch"]): r for r in records}
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
    assert "--s22-solver" in by[(72, 1)]["argv"] and "--s22-solver" not in by[(72, 2)]["argv"]
    # the same two negatives on this function's own exec'd runner functions
    check(need(" ".join(by[(72, 1)]["children"]), controls), {"p2prime": "", "s22": "/x"})
    check(need(" ".join(by[(72, 2)]["children"]), controls), {"p2prime": "", "s22": ""})
    refused(need(" ".join(by[(72, 2)]["children"]), controls), {"p2prime": "", "s22": "/x"}, "built for nothing")
    refused(need(" ".join(by[(72, 1)]["children"]), controls), {"p2prime": "", "s22": ""}, "--s22-solver was not passed")
    seen = nbsel.negatives(cell, controls, 72)
    print("   the reference-library wiring agrees: notebook pins, batch table, generator children, runner flags; "
          f"7 start-of-session cases seen both ways; {len(records)} launch argvs accepted by the runner's own parser "
          f"(BATCH 72 launch 1 passes --s22-solver, launch 2 does not; BATCH 73, one session, passes none); {len(seen)} launch negatives each refused "
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
# twins of BATCH 71 + the 4 S23 twins of BATCH 72 + the 2 production children of BATCH 73. ONE pin, read by
# s21_generator_negatives, s23_generator_negatives and main(): BATCH 72 updated main's copy and left this
# function's at 26, and release 2026-10-07 died on it ten minutes into the build.
KEPT_CHILDREN = 32


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
    KEPT = KEPT_CHILDREN   # the 18 kept children + the 8 S22 validation twins of BATCH 71 + the 4 S23 twins of BATCH 72 + the 2 production children of BATCH 73

    def healthy(r, code):
        assert code == 0, f"exit {code}"
        assert r["children_total"] == KEPT, r["children_total"]
        assert r["children_admissible"] == r["children_total"], r["inadmissible"]
        assert r["library_compiled"]
        assert all(c["status"] == "COMPLETE" for c in r["children"].values()), sorted(
            (k, c["status"]) for k, c in r["children"].items() if c["status"] != "COMPLETE")
        assert r["xu2017_fig5"], "no Xu report"
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
    run_direct("s23_generator_refuses_bad_tear_ratio", s23_generator_negatives)


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
        print(f"   batch {_batch}: {len(filt)} children built, exactly the filter ({last})")
    run_direct("the_notebook_batch_filter_generates", launch_current_batch)

    run_direct("the_notebook_launch_filters_generate", lambda: launch_filters_generate(nbsel, _batch, _filtered_generation))
    if _batch != 72:
        # BATCH 72's two launches stay exercised while a later batch is the default
        run_direct("batch_72_launch_filters_generate", lambda: launch_filters_generate(nbsel, 72, _filtered_generation))

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
