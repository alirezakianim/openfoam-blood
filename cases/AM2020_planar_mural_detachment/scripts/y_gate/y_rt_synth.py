#!/usr/bin/env python3
"""SYNTHETIC returned cases of the symmetric Y's PLUG series (offline; pure python, standard library only): the restart-test pair Y80_RT_A / Y80_RT_B and a clot child
(Y81_*_PLUG). Used by scripts/check_y_symmetric.py and by dry_run.py (`synthesise_y_plug_child`). NOT physics: the fields are deterministic numbers laid out the way the
solver writes them (ascii, 17 digits), so the exact-equality gates of assess_case.y_rt_pair can be shown to PASS on an exact restart and to FAIL, VOID, for the reason
they name, when one thing is planted.

    build_rt_case(case, which, entry, **o)      which = "A" (a cold start that wrote every step to leg2_stop) or "B" (the LEG-2 case: restart.json, the
                                                step-leg2_stop directory only, a log from step leg1_stop + 1)
    build_rt_pair(root, entry, **o) -> (a, b)   both, under root/A and root/B
    build_plug_case(case, spec, **o)            the clot child: a log to its end time, exit code, wall time, and the initial c drawn by y_plug.py at the fixture centres

Options of the RT case (each plants ONE defect on B, except where it says A; the default is an EXACT restart):
    ulp=<field>        B's step-leg2_stop file of this field differs from A's by ONE unit in the last place in one cell
    dt_ulp=True        B's logged deltaT of step leg1_stop + 2 differs from A's by one unit in the last place
    late_leg1=True     leg 1 stopped one step late: restart.json says chosen_index leg1_stop + 1, B's first logged step is leg1_stop + 2
    cold_b=True        B logs PHASE_CLOT_S25_RESTART state=seeded (the restart did not restore)
    warm_a=True        A logs state=restored (it was not a cold start)
    vs_zero=True       A's Vs is 0 at the restart step (the moving-skeleton state is not exercised)
    static=True        thetaC and thetaH do not move between the two steps (the denominator of RT-1a is 0)
    precision=<n>      both controlDicts say writePrecision n
    drop_field=<name>  B's step-leg2_stop directory lacks this field
    no_restart_json=True   B has no restart.json
    dt_digits=<n>      the logged deltaT carry only n significant digits
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIELDS = (("thetaC", "scalar"), ("thetaH", "scalar"), ("w", "scalar"), ("q", "scalar"), ("Vs", "vector"))
COMP = {"scalar": 1, "vector": 3}
CLASS = {"scalar": "volScalarField", "vector": "volVectorField"}
N_CELLS = 40


def _fmt(v: float) -> str:
    return f"{v:.17g}"


def state(step: int, n: int = N_CELLS, seed: int = 11, static: bool = False) -> dict:
    """{field: rows} at one step: a fixed random base, thetaC moving 1e-4..1e-3 per step (so the RT-1a denominator is well above 1e-8), Vs growing 1 % per step."""
    rng = random.Random(seed)
    base = {name: [[rng.uniform(0.1, 1.0) for _ in range(COMP[kind])] for _ in range(n)] for name, kind in FIELDS}
    rate = [rng.uniform(1e-4, 1e-3) for _ in range(n)]
    out = {}
    for name, kind in FIELDS:
        rows = []
        for i in range(n):
            v = list(base[name][i])
            moving = 0 if static else step
            if name == "thetaC":
                v = [v[0] * 0.4 + moving * rate[i]]
            elif name == "thetaH":
                v = [v[0] * 0.1 + moving * 0.5 * rate[i]]
            elif name == "Vs":
                v = [x * (1.0 + 0.01 * step) for x in v]
            rows.append(v)
        out[name] = rows
    return out


def write_field(path: Path, kind: str, rows) -> None:
    body = "\n".join("(" + " ".join(_fmt(x) for x in r) + ")" if kind == "vector" else _fmt(r[0]) for r in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {CLASS[kind]};\n    object {path.name};\n}}\n\n"
                    f"dimensions [0 0 0 0 0 0 0];\n\ninternalField nonuniform List<{kind}> \n{len(rows)}\n(\n{body}\n)\n;\n\nboundaryField\n{{\n}}\n\n"
                    "// ************************************************************************* //\n")


def dt_of(step: int) -> float:
    """A step size that differs from step to step in its last digits (so a one-ulp edit of one record is a different number)."""
    return 5e-6 * (1.0 + 0.0137 * step) / 3.0


def dt_text(step: int, digits: int = 17) -> str:
    return f"{dt_of(step):.{digits - 1}e}" if digits < 17 else repr(dt_of(step))


def _nudge(v: float) -> float:
    return math.nextafter(v, math.inf)


def build_rt_case(case: Path, which: str, entry: dict, **o) -> Path:
    case = Path(case)
    s1, s2 = int(entry["leg1_stop"]), int(entry["leg2_stop"])
    if which == "A":
        steps, first = range(s1 - 4, s2 + 1), 1
    else:
        steps, first = [s2], s1 + (2 if o.get("late_leg1") else 1)
    case.mkdir(parents=True, exist_ok=True)
    for k in steps:
        st = state(k, static=bool(o.get("static")))
        if which == "A" and o.get("vs_zero") and k == s1:
            st["Vs"] = [[0.0, 0.0, 0.0] for _ in st["Vs"]]
        d = case / f"{k * 1e-6:.6f}".rstrip("0")
        for name, kind in FIELDS:
            rows = st[name]
            if which == "B" and k == s2:
                if o.get("drop_field") == name:
                    continue
                if o.get("ulp") == name:
                    rows = [list(r) for r in rows]
                    rows[7][0] = _nudge(rows[7][0])
            write_field(d / name, kind, rows)
        (d / "uniform").mkdir(exist_ok=True)
        (d / "uniform" / "time").write_text(f"FoamFile\n{{\n    format ascii;\n}}\nvalue {k * 1e-6};\nname \"{k * 1e-6}\";\nindex {k};\ndeltaT {dt_text(k)};\n")
        (d / "uniform" / "phaseClotState").write_text("FoamFile\n{\n    format ascii;\n}\nexplicitDiffusionRate 0;\n")
    prec = o.get("precision", 17)
    (case / "system").mkdir(exist_ok=True)
    (case / "system" / "controlDict").write_text(f"FoamFile\n{{\n    format ascii;\n}}\nwriteFormat ascii;\nwritePrecision {prec};\ntimePrecision 10;\n")
    restored = (which == "B" and not o.get("cold_b")) or (which == "A" and o.get("warm_a"))
    lines = [f"PHASE_CLOT_S25_RESTART state={'restored' if restored else 'seeded'}"]
    for k in range(first, s2 + 1):
        digits = int(o.get("dt_digits", 17))
        t = dt_text(k, digits)
        if which == "B" and o.get("dt_ulp") and k == s1 + 2:
            t = repr(_nudge(dt_of(k)))
        lines.append(f"Time = {k * dt_of(k):.10g}s")
        lines.append(f"PHASE_CLOT_DT step={k} time={k * 1e-6:.6f} deltaT={t} dtFluidCo=1 dtWave=1")
    lines.append("ExecutionTime = 90 s  ClockTime = 91 s\n\nEnd\n")
    (case / "log.phaseClotAxisymmetricFluid").write_text("\n".join(lines))
    (case / "run-exit-code.txt").write_text("0\n")
    (case / "log.wallTime").write_text("WALL_SECONDS=91\n")
    if which == "B" and not o.get("no_restart_json"):
        (case / "restart.json").write_text(json.dumps({"chosen_index": s1 + (1 if o.get("late_leg1") else 0), "source": "synthetic"}) + "\n")
    return case


def build_rt_pair(root: Path, entry: dict, **o) -> tuple:
    root = Path(root)
    a_opts = {k: v for k, v in o.items() if k in ("vs_zero", "warm_a", "static", "precision", "dt_digits")}
    return build_rt_case(root / "A", "A", entry, **a_opts), build_rt_case(root / "B", "B", entry, **o)


def build_plug_case(case: Path, spec: dict, exit_code: int = 0, short: bool = False, dt: float = 5e-5, c_initial: bool = True) -> Path:
    """The clot child's returned case. The log runs to the step that passes endTime - dt/2 (Time::run), dt the series' max_delta_t_s; `short` stops it at a third
    of the steps. With c_initial the t = 0 field c is drawn by y_plug.py (the generator's own signed distance) at the fixture's cell centres into 0/c."""
    case = Path(case)
    case.mkdir(parents=True, exist_ok=True)
    end = spec["end_time_s"]
    n_steps = int(math.ceil((end - 0.5 * dt) / dt - 1e-9))
    if short:
        n_steps = max(1, n_steps // 3)
    lines = [f"Time = {k * dt:.10g}s\n" for k in range(1, n_steps + 1)]
    lines.append("ExecutionTime = 500 s  ClockTime = 510 s\n\nEnd\n" if exit_code == 0 else "Foam::sigFpe::sigHandler(int) at ??:?\n")
    (case / "log.phaseClotAxisymmetricFluid").write_text("".join(lines))
    (case / "run-exit-code.txt").write_text(f"{exit_code}\n")
    (case / "log.wallTime").write_text("WALL_SECONDS=510\n")
    if c_initial:
        import y_plug
        import y_synth
        ext = spec["y_gate"]
        plug = ext["plug"]
        p = {k: plug[k] for k in y_plug.PLUG_KEYS}
        main_case = HERE.parents[1]
        anchors = y_plug.dict_anchors((main_case / ext["mesh"]["blockMeshDict"]).read_text())
        cells, _faces = y_synth.fixtures(ext)
        c = y_plug.c_field(cells, p, plug["D0_m"], plug["interface_width_m"], anchors)
        write_field(case / "0" / "c", "scalar", [[v] for v in c])
    return case
