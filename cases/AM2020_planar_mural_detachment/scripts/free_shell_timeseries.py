#!/usr/bin/env python3
"""Post-process a free-shell throat child (design
2026-10-02-case-free-shell-throat, rev. 3, S7 item 7; S8 V3, V5).

    free_shell_timeseries.py <case_dir>

Reads
  * the function objects this series' controlDict writes. Paths and columns
    are OpenFOAM-13's, read from its source:
      surfaceFieldValue: postProcessing/<name>/<startTime>/surfaceFieldValue.dat
        (logFiles.C:42-44: baseFileDir()/prefix_/timeName/(name + ".dat");
        writeFile.C:33-35 "postProcessing"; logFiles.C:87 prefix_ = the
        object's name; fieldValue.C:93 resetName(valueType_) with valueType_ =
        typeName, surfaceFieldValue.C:542). Header lines start "#"
        (writeFile.C:115); the last is "# Time<TAB>orientedSum(phi)"
        (surfaceFieldValue.C:420-432). Rows: time then TAB result
        (surfaceFieldValue.C:748, surfaceFieldValueTemplates.C:366).
      probes: postProcessing/probesPU/<startTime>/<field> (probes.C:198-227);
        "# Probe i (x y z)" per location, then "# Time  0 1 ..."
        (probes.C:240-250); rows: time then one value per probe, a vector
        written as "(ux uy uz)" with spaces inside (probesTemplates.C:79-85).
      A restart opens a new <startTime> directory; later directories win.
  * the written fields: q at every write, the cell centres 0/Ccx, 0/Ccy
    (writeCellCentres), 0/Vc if present (the centroid's weight), and
    constant/throatProfile.json for the top wall, through apply_throat's own
    channel_height (imported, not transcribed).

BATCH 68 (design 2026-10-05-case-free-shell-centred-s19b, rev. 3, sections 4-5;
red team 01 changes 4-7): F0 on the 0/ fields with the q < 0.5 indicator (q
from thetaC + thetaH, as assess_case.phase_c does at 0/), both wall gaps per
frame, the trigger (either-wall contact or LE >= 175 um), the 2 ms
least-squares front speed, the five exclusive outcomes, and the DERIVED
dp_clot = dp (1 - Q/Q0). `--self-test [batch63_dense.csv]` runs the synthetic
outcome tests (and, given the CSV, batch 63's dense series, which must read
"entered").

Writes postProcessing/freeShellTimeseries.csv (one row per field write),
postProcessing/freeShellFlux.csv (one row per function-object step),
postProcessing/freeShellSummary.json and a PNG.

The clot is q < 0.5 (q is the fluid fraction: the content is the integral of
1 - q, design S8 V6). Edges are the q = 0.5 contour, linearly interpolated
between neighbouring cells along each mesh row (x) and column (y); leading =
the largest contour x, trailing = the smallest, over ALL clot material (no
fragment is dropped).
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_throat import channel_height  # noqa: E402  (pattern 1: import)

# --- every number and its source -------------------------------------------
Q0_REF_DP = 0.334326        # m2/s2, the series whose Q0 is recorded (design S2)
Q0_REF_M3_S = 4.7249e-13    # its Q0 (design S2; numerics 01 S4: reproduced to 1.0000)
# -> at 0.0737066 this gives 1.0417e-13 m3/s, design S2 / ce-placement-250.out.txt
Q_START_TIME_S = 1.0e-3     # "q_start (Q/Q0 at 1.0 ms)", design S8 V3
LUBRICATION_QSTART_BOUND = 0.574  # fixed impermeable P1 disc, design S3 / ce-qstart-250
# Open-channel share of the imposed dp between the probe pair (design S7.7,
# red team 03 F10). Pair 1 (120,25)->(225,10): 0.274, printed in the design.
# Pair 2 (120,25)->(270,10): DERIVED here from the same resistance shares
# (design S2: taper 23.9%, throat 70.7% over its 100 um uniform length):
# 0.239 + 0.707 * 50/100 = 0.5925. The same sum for pair 1, 0.239 + 0.707 *
# 5/100 = 0.2744, reproduces the printed 0.274.
R_SEG_SHARE = {"pair1": 0.274, "pair2": 0.239 + 0.707 * 0.50}
P_UP = (120e-6, 25e-6)
P_DOWN = {"pair1": (225e-6, 10e-6), "pair2": (270e-6, 10e-6)}
PROBE_SWITCH_X_M = 220e-6   # switch the pair once the leading edge passes 220 um (S7.7)
# BATCH 68 design rev. 3 section 5 (red team 01 changes 4, 6, 7):
ARREST_SPEED_M_S = 1.0e-4   # fitted front speed for "arrests" (resolution ~1e-5 m/s)
TRIGGER_LE_M = 175e-6       # the trigger: LE >= 175 um (H = 30 um at 180.8 um) ...
CONTACT_CELLS = 1.0         # ... or contact with EITHER wall: gap <= 1 local cell
THROAT_END_X_M = 220e-6     # "passes": TE >= 220 um; "entered": LE >= 220, TE < 220
CLOG_RATIO = 0.80           # "clogs": Q/q_start <= 0.80 ...
HOLD_S = 2.0e-3             # ... held >= 2 ms after the trigger; arrest window too
FIT_WINDOW_S = 2.0e-3       # least-squares LE fit over a sliding 2 ms window
RECEDE_SE = 2.0             # "receding": slope < -2 standard errors of the fit
FIT_MIN_POINTS = 6          # >= 6 frames in a window (11 at 0.2 ms frames)
F0_WIDTHS = 3.0             # F0: both initial gaps >= 3 epsilonC (9.0 um at 3 um)
V3_T0_S, V3_T1_S = 0.4e-3, 2.0e-3   # S8 V3
V3_ADVANCE_MIN_M = 1.87e-6          # 0.5 x 2.34e-3 m/s x 1.6 ms (S8 V3)
IDENTITY_TOL = 0.05                 # Q/Q0 = 1 - dp_clot/dp within 5% (S7.7)
WALL_SAMPLES = 3201                 # as ce-placement-250's top_gap
TIME_MATCH_REL = 1e-6


# --- OpenFOAM readers ------------------------------------------------------
def read_internal_scalar(path: Path, n: int | None = None) -> np.ndarray:
    """internalField of a volScalarField, nonuniform List<scalar> or uniform."""
    text = path.read_text(errors="replace")
    m = re.search(r"^internalField\s+(.*)$", text, re.M)
    if not m:
        raise ValueError(f"{path}: no internalField")
    head = m.group(1).strip()
    if head.startswith("uniform"):
        if n is None:
            raise ValueError(f"{path}: uniform internalField needs a cell count")
        return np.full(n, float(head.split()[1].rstrip(";")))
    rest = text[m.end():]
    cm = re.match(r"\s*(\d+)\s*\(", rest)
    if not cm:
        raise ValueError(f"{path}: cannot find the list count")
    count = int(cm.group(1))
    body = rest[cm.end():rest.index(")", cm.end())]
    vals = np.array(body.split(), dtype=float)
    if vals.size != count or (n is not None and count != n):
        raise ValueError(f"{path}: {vals.size} values, header {count}, mesh {n}")
    return vals


def _start_dirs(base: Path) -> list[Path]:
    dirs = [d for d in base.iterdir() if d.is_dir()] if base.is_dir() else []
    return sorted(dirs, key=lambda d: float(d.name))


def read_surface_field_value(case: Path, name: str, column: str) -> list[tuple[float, float]]:
    """One column of postProcessing/<name>/<t0>/surfaceFieldValue.dat, by its
    header label (e.g. "orientedSum(phi)"); restarts merged, later wins."""
    rows: dict[float, float] = {}
    files = [d / "surfaceFieldValue.dat" for d in _start_dirs(case / "postProcessing" / name)]
    files = [f for f in files if f.is_file()]
    if not files:
        raise FileNotFoundError(f"{case}: no postProcessing/{name}/<t>/surfaceFieldValue.dat")
    for f in files:
        col = None
        for line in f.read_text().splitlines():
            if line.startswith("#"):
                labels = [s.strip() for s in line.lstrip("#").split("\t")]
                if labels and labels[0] == "Time":
                    if column not in labels:
                        raise ValueError(f"{f}: no column {column!r} in {labels}")
                    col = labels.index(column)
                continue
            if not line.strip():
                continue
            if col is None:
                raise ValueError(f"{f}: data before the '# Time' header")
            cols = line.split()
            rows[float(cols[0])] = float(cols[col])
    return sorted(rows.items())


TOKEN = re.compile(r"\([^)]*\)|\S+")


def read_probes(case: Path, field: str, obj: str = "probesPU"):
    """-> (locations [(x,y,z)], rows [(t, [value per probe])]); a vector value
    is a tuple. Restarts merged, later wins."""
    locs: list[tuple] = []
    rows: dict[float, list] = {}
    files = [d / field for d in _start_dirs(case / "postProcessing" / obj)]
    files = [f for f in files if f.is_file()]
    if not files:
        raise FileNotFoundError(f"{case}: no postProcessing/{obj}/<t>/{field}")
    for f in files:
        flocs = []
        for line in f.read_text().splitlines():
            pm = re.match(r"#\s*Probe\s+(\d+)\s+\(([^)]*)\)", line)
            if pm:
                flocs.append(tuple(float(v) for v in pm.group(2).split()))
                continue
            if line.startswith("#") or not line.strip():
                continue
            toks = TOKEN.findall(line)
            vals = [tuple(float(v) for v in tk[1:-1].split()) if tk.startswith("(")
                    else float(tk) for tk in toks[1:]]
            if len(vals) != len(flocs):
                raise ValueError(f"{f}: {len(vals)} values for {len(flocs)} probes")
            rows[float(toks[0])] = vals
        if locs and flocs != locs:
            raise ValueError(f"{f}: probe locations differ from the first start directory")
        locs = flocs
    return locs, sorted(rows.items())


def probe_index(locs, point, tol=1e-9) -> int:
    for i, loc in enumerate(locs):
        if abs(loc[0] - point[0]) <= tol and abs(loc[1] - point[1]) <= tol:
            return i
    raise ValueError(f"no probe at {point}; probes are {locs}")


def at_time(series, t):
    """Value of a (time, value) series at time t: exact to TIME_MATCH_REL, else None."""
    for tt, v in series:
        if math.isclose(tt, t, rel_tol=TIME_MATCH_REL, abs_tol=1e-15):
            return v
    return None


# --- mesh and geometry -----------------------------------------------------
class Mesh:
    def __init__(self, ccx: np.ndarray, ccy: np.ndarray, profile: dict, vc=None):
        n = ccx.size
        drops = np.nonzero(np.diff(ccx) < 0)[0]
        if drops.size == 0:
            raise ValueError("cell centres are not row-ordered (no row break)")
        nx = int(drops[0]) + 1
        if n % nx:
            raise ValueError(f"{n} cells is not a multiple of the row length {nx}")
        self.nx, self.ny = nx, n // nx
        self.X = ccx.reshape(self.ny, nx)
        self.Y = ccy.reshape(self.ny, nx)
        if np.any(np.diff(self.X, axis=1) <= 0) or np.any(np.diff(self.Y, axis=0) <= 0):
            raise ValueError("cell centres are not a structured nx x ny block; refusing to guess")
        self.V = vc.reshape(self.ny, nx) if vc is not None else None
        self.profile = profile
        x1 = profile["taper_start_x_m"]
        args = (profile["channel_height_m"], profile["throat_height_m"], x1,
                x1 + profile["taper_length_m"])
        self.H = lambda x: channel_height(x, *args)
        dx0 = self.X[0, 1] - self.X[0, 0]
        xs = np.linspace(self.X[0, 0] - 0.5 * dx0, self.X[0, -1] + 0.5 * dx0, WALL_SAMPLES)
        self.wall = np.column_stack([xs, [self.H(x) for x in xs]])
        # the local cell height at the top wall, MEASURED per column
        self.dy_top = self.Y[-1, :] - self.Y[-2, :]
        self.dy_bot = self.Y[1, :] - self.Y[0, :]


def contour(mesh: Mesh, q: np.ndarray):
    """q = 0.5 crossings along rows (x) and columns (y) -> (row_pts, col_pts)."""
    Q = q.reshape(mesh.ny, mesh.nx) - 0.5
    X, Y = mesh.X, mesh.Y
    a, b = Q[:, :-1], Q[:, 1:]
    j, i = np.nonzero((a * b <= 0) & (a != b))
    s = a[j, i] / (a[j, i] - b[j, i])
    rx = X[j, i] + s * (X[j, i + 1] - X[j, i])
    ry = Y[j, i] + s * (Y[j, i + 1] - Y[j, i])
    a, b = Q[:-1, :], Q[1:, :]
    j, i = np.nonzero((a * b <= 0) & (a != b))
    s = a[j, i] / (a[j, i] - b[j, i])
    cx = X[j, i] + s * (X[j + 1, i] - X[j, i])
    cy = Y[j, i] + s * (Y[j + 1, i] - Y[j, i])
    return np.column_stack([rx, ry]), np.column_stack([cx, cy])


def write_state(mesh: Mesh, q: np.ndarray) -> dict:
    rows, cols = contour(mesh, q)
    clot = q.reshape(mesh.ny, mesh.nx) < 0.5
    out = {"leading_edge_x_m": None, "trailing_edge_x_m": None, "top_gap_m": None,
           "top_gap_cells": None, "bottom_gap_m": None, "bottom_gap_cells": None,
           "centroid_x_m": None, "centroid_y_m": None,
           "aspect_ratio": None, "clot_cells": int(clot.sum())}
    if rows.size:
        out["leading_edge_x_m"] = float(rows[:, 0].max())
        out["trailing_edge_x_m"] = float(rows[:, 0].min())
    pts = np.vstack([rows, cols])
    if pts.size:
        d = np.hypot(pts[:, None, 0] - mesh.wall[None, :, 0],
                     pts[:, None, 1] - mesh.wall[None, :, 1])
        k = np.unravel_index(np.argmin(d), d.shape)
        gap = float(d[k])
        col = int(np.argmin(np.abs(mesh.X[-1, :] - mesh.wall[k[1], 0])))
        out["top_gap_m"] = gap
        out["top_gap_cells"] = gap / float(mesh.dy_top[col])
        # THE BOTTOM WALL is y = 0 (apply_throat bends only the top). A clot
        # cell in the first row touches it: gap 0. Else the lowest contour
        # point, in units of the bottom cell height of its column.
        if clot[0, :].any():
            out["bottom_gap_m"], out["bottom_gap_cells"] = 0.0, 0.0
        else:
            kb = int(np.argmin(pts[:, 1]))
            colb = int(np.argmin(np.abs(mesh.X[0, :] - pts[kb, 0])))
            out["bottom_gap_m"] = float(pts[kb, 1])
            out["bottom_gap_cells"] = float(pts[kb, 1]) / float(mesh.dy_bot[colb])
        y_lo = 0.0 if clot[0, :].any() else float(pts[:, 1].min())
        y_hi = float(pts[:, 1].max())
        if rows.size and y_hi > y_lo:
            out["aspect_ratio"] = (out["leading_edge_x_m"] - out["trailing_edge_x_m"]) / (y_hi - y_lo)
    if mesh.V is not None:
        w = (1.0 - q.reshape(mesh.ny, mesh.nx)) * mesh.V
        if w.sum() > 0:
            out["centroid_x_m"] = float((w * mesh.X).sum() / w.sum())
            out["centroid_y_m"] = float((w * mesh.Y).sum() / w.sum())
    return out


# --- BATCH 68: the trigger, the fitted front speed, the outcome ------------
def trigger(states: list[dict]) -> dict:
    """The first frame at which the disc touches EITHER wall (a gap <= 1 local
    cell) or its leading edge reaches 175 um (design rev. 3 section 5)."""
    contact = next((s["time_s"] for s in states
                    if min(g for g in (s.get("top_gap_cells"), s.get("bottom_gap_cells"), math.inf)
                           if g is not None) <= CONTACT_CELLS), None)
    le = next((s["time_s"] for s in states
               if s["leading_edge_x_m"] is not None and s["leading_edge_x_m"] >= TRIGGER_LE_M), None)
    cands = [(t, k) for t, k in ((contact, "contact"), (le, "LE>=175um")) if t is not None]
    t0 = min((t for t, _ in cands), default=None)
    return {"trigger_time_s": t0,
            "trigger_kind": "+".join(k for t, k in cands if t == t0) if t0 is not None else None,
            "contact_time_s": contact, "le175_time_s": le}


def front_fit(states: list[dict], t_end: float) -> dict | None:
    """Least-squares slope of LE(t) over the frames in [t_end - 2 ms, t_end],
    with its standard error; None if the window is not covered."""
    tol = 1e-9
    pts = [(s["time_s"], s["leading_edge_x_m"]) for s in states
           if t_end - FIT_WINDOW_S - tol <= s["time_s"] <= t_end + tol
           and s["leading_edge_x_m"] is not None]
    if len(pts) < FIT_MIN_POINTS or pts[0][0] > t_end - FIT_WINDOW_S + tol:
        return None
    t = np.array([a for a, _ in pts]); x = np.array([b for _, b in pts])
    tm = t.mean(); sxx = float(((t - tm) ** 2).sum())
    slope = float(((t - tm) * (x - x.mean())).sum() / sxx)
    resid = x - (x.mean() + slope * (t - tm))
    se = float(math.sqrt(float((resid ** 2).sum()) / (len(t) - 2) / sxx))
    return {"t_end_s": t_end, "slope_m_s": slope, "slope_se_m_s": se, "n": len(t)}


def classify(states: list[dict], ratio: list[tuple[float, float]],
             end_time_s: float | None) -> dict:
    """The five exclusive outcomes, in the design's order (section 5).
    ratio: (t, Q/q_start) at every logged step. 'entered' is judged at the
    last frame (the end of the window the run reached)."""
    out = {"outcome": "undecided", "outcome_time_s": None, **trigger(states),
           "clog_from_s": None, "arrest_fit": None, "receding_fits": 0}
    last = states[-1] if states else None
    tp = next((s["time_s"] for s in states if s["trailing_edge_x_m"] is not None
               and s["trailing_edge_x_m"] >= THROAT_END_X_M), None)
    if tp is not None:
        out.update(outcome="passes", outcome_time_s=tp)
        return out
    if last is not None and last["leading_edge_x_m"] is not None \
            and last["leading_edge_x_m"] >= THROAT_END_X_M \
            and last["trailing_edge_x_m"] is not None \
            and last["trailing_edge_x_m"] < THROAT_END_X_M:
        out.update(outcome="entered", outcome_time_s=last["time_s"])
        return out
    t_tr = out["trigger_time_s"]
    if t_tr is None:
        return out
    run_start = None
    for t, r in ratio:
        if t < t_tr - 1e-12:
            continue
        if r <= CLOG_RATIO:
            run_start = t if run_start is None else run_start
            if t - run_start >= HOLD_S - 1e-12:
                out.update(outcome="clogs", outcome_time_s=t, clog_from_s=run_start)
                return out
        else:
            run_start = None
    # "the fitted front speed is <= 1e-4 m/s and not negative beyond its fit
    # error (not receding) for >= 2 ms": the fit (each over the 2 ms ending
    # at a frame, starting at or after the trigger) must satisfy both at
    # EVERY frame over a span >= 2 ms. A receding fit (slope below -2
    # standard errors: significantly negative; at 1 SE a stationary noisy
    # front reads receding in ~16% of windows and a 2 ms hold rarely
    # survives) or a moving one (> 1e-4 m/s) restarts the span.
    held_from = None
    for s in states:
        if s["time_s"] - FIT_WINDOW_S < t_tr - 1e-9:
            continue
        f = front_fit(states, s["time_s"])
        if f is None:
            continue
        receding = f["slope_m_s"] < -RECEDE_SE * f["slope_se_m_s"]
        out["receding_fits"] += receding
        if not receding and f["slope_m_s"] <= ARREST_SPEED_M_S:
            held_from = s["time_s"] if held_from is None else held_from
            if s["time_s"] - held_from >= HOLD_S - 1e-9:
                out.update(outcome="arrests", outcome_time_s=s["time_s"], arrest_fit=f,
                           arrest_from_s=held_from)
                return out
        else:
            held_from = None
    return out


def f0(mesh: "Mesh", q0: np.ndarray, epsilon_m: float | None) -> dict:
    """F0: on the 0/ fields, both wall gaps of the q < 0.5 region >= 3 eps."""
    st = write_state(mesh, q0)
    floor = None if epsilon_m is None else F0_WIDTHS * epsilon_m
    ok = None
    if floor is not None and st["top_gap_m"] is not None and st["bottom_gap_m"] is not None:
        ok = st["top_gap_m"] >= floor and st["bottom_gap_m"] >= floor
    return {"f0": None if ok is None else ("PASS" if ok else "FAIL"),
            "f0_top_gap_m": st["top_gap_m"], "f0_bottom_gap_m": st["bottom_gap_m"],
            "f0_floor_m": floor, "f0_clot_cells": st["clot_cells"]}


def q_at_zero(case: Path, n: int) -> np.ndarray | None:
    """q at 0/: 0/q if written, else 1 - (thetaC + thetaH)/skeletonVolumeFraction
    (assess_case.phase_c's 0/ branch, c = 2q - 1)."""
    if (case / "0" / "q").is_file():
        return read_internal_scalar(case / "0" / "q", n)
    tc, th = case / "0" / "thetaC", case / "0" / "thetaH"
    props = case / "constant" / "phaseClotProperties"
    if not (tc.is_file() and th.is_file() and props.is_file()):
        return None
    m = re.search(r"^\s*skeletonVolumeFraction\s+([-\d.eE+]+)\s*;", props.read_text(errors="replace"), re.M)
    if not m or not float(m.group(1)) > 0:
        return None
    return 1.0 - (read_internal_scalar(tc, n) + read_internal_scalar(th, n)) / float(m.group(1))


def v3(states: list[dict]) -> dict:
    by_t = {s["time_s"]: s for s in states}
    def trail(t):
        s = next((v for k, v in by_t.items() if math.isclose(k, t, rel_tol=TIME_MATCH_REL)), None)
        return None if s is None else s["trailing_edge_x_m"]
    a, b = trail(V3_T0_S), trail(V3_T1_S)
    if a is None or b is None:
        return {"v3": "NOT-EVALUABLE", "trailing_0p4ms_m": a, "trailing_2p0ms_m": b,
                "trailing_advance_m": None}
    adv = b - a
    return {"v3": "PASS" if adv >= V3_ADVANCE_MIN_M else "FAIL",
            "trailing_0p4ms_m": a, "trailing_2p0ms_m": b, "trailing_advance_m": adv,
            "v3_threshold_m": V3_ADVANCE_MIN_M}


def load_mesh(case: Path, ccx=None, ccy=None) -> Mesh:
    cx = read_internal_scalar(Path(ccx) if ccx else case / "0" / "Ccx")
    cy = read_internal_scalar(Path(ccy) if ccy else case / "0" / "Ccy", cx.size)
    vcp = case / "0" / "Vc"
    vc = read_internal_scalar(vcp, cx.size) if vcp.is_file() else None
    prof = json.loads((case / "constant" / "throatProfile.json").read_text())
    return Mesh(cx, cy, prof, vc)


def field_states(case: Path, mesh: Mesh) -> list[dict]:
    states = []
    for d in case.iterdir():
        if not d.is_dir() or not re.fullmatch(r"[0-9.eE+-]+", d.name) or not (d / "q").is_file():
            continue
        t = float(d.name)
        if t == 0.0:
            continue
        s = write_state(mesh, read_internal_scalar(d / "q", mesh.nx * mesh.ny))
        s["time_s"] = t
        states.append(s)
    return sorted(states, key=lambda s: s["time_s"])


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--self-test":
        return self_test(sys.argv[2] if len(sys.argv) > 2 else None)
    if len(sys.argv) != 2:
        raise SystemExit("usage: free_shell_timeseries.py <case_dir> | --self-test [batch63_dense.csv]")
    case = Path(sys.argv[1]).resolve()
    child = json.loads((case / "child.json").read_text())
    # The drive the solver RAN: 0/p inlet fixedValue (child.json does not carry it;
    # batch 63, 2026-10-03). outlet p = 0 (design S2).
    _m = re.search(r"inlet\s*\{[^}]*?value\s+uniform\s+([-+0-9.eE]+)", (case / "0" / "p").read_text())
    dp = float(child["inlet_pressure_m2_s2"]) if "inlet_pressure_m2_s2" in child else float(_m.group(1))
    q0 = Q0_REF_M3_S * dp / Q0_REF_DP

    flux = read_surface_field_value(case, "outletFlux", "orientedSum(phi)")
    locs, p_rows = read_probes(case, "p")
    i_up = probe_index(locs, P_UP)
    i_dn = {k: probe_index(locs, v) for k, v in P_DOWN.items()}
    q_start = at_time(flux, Q_START_TIME_S)
    q_start = None if q_start is None else q_start / q0

    mesh = load_mesh(case)
    states = field_states(case, mesh)
    ratio = [(t, qv / q0 / q_start) for t, qv in flux] if q_start else []
    ev = classify(states, ratio, child.get("end_time_s"))
    q0f = q_at_zero(case, mesh.nx * mesh.ny)
    eps = (child.get("geometry") or {}).get("interface_width_m")
    f0r = (f0(mesh, q0f, eps) if q0f is not None else
           {"f0": None, "f0_reason": "no 0/q and no 0/thetaC+thetaH with skeletonVolumeFraction"})
    rows = []
    for s in states:
        t = s["time_s"]
        q_out = at_time(flux, t)
        pv = at_time(p_rows, t)
        qq0 = None if q_out is None else q_out / q0
        pair = "pair2" if (s["leading_edge_x_m"] or 0.0) >= PROBE_SWITCH_X_M else "pair1"
        dp_clot = ident = None
        if pv is not None and qq0 is not None:
            dp_clot = (pv[i_up] - pv[i_dn[pair]]) - R_SEG_SHARE[pair] * dp * qq0
            ident = abs(qq0 - (1.0 - dp_clot / dp)) <= IDENTITY_TOL
        rows.append({**s, "q_out_m3_s": q_out, "q_over_q0": qq0,
                     "q_over_qstart": (qq0 / q_start) if (qq0 is not None and q_start) else None,
                     "probe_pair": pair, "dp_clot_m2_s2": dp_clot, "identity_within_5pct": ident,
                     # DERIVED from the flux by the series model, not measured
                     # (design rev. 3 section 5; the probe identity failed at
                     # 48 of 61 batch-63 writes).
                     "dp_clot_derived_m2_s2": None if qq0 is None else dp * (1.0 - qq0)})
    out = case / "postProcessing"
    out.mkdir(exist_ok=True)
    cols = ["time_s", "q_out_m3_s", "q_over_q0", "q_over_qstart", "probe_pair", "dp_clot_m2_s2",
            "identity_within_5pct", "leading_edge_x_m", "trailing_edge_x_m", "top_gap_m",
            "top_gap_cells", "centroid_x_m", "centroid_y_m", "aspect_ratio", "clot_cells",
            "bottom_gap_m", "bottom_gap_cells", "dp_clot_derived_m2_s2"]
    with (out / "freeShellTimeseries.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (out / "freeShellFlux.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "q_out_m3_s", "q_over_q0", "q_over_qstart"])
        for t, qv in flux:
            w.writerow([t, qv, qv / q0, (qv / q0 / q_start) if q_start else ""])
    last_t = states[-1]["time_s"] if states else None
    summary = {
        "child_id": child.get("child_id"), "q0_m3_s": q0, "q_start": q_start,
        "q_start_lubrication_bound": LUBRICATION_QSTART_BOUND, "n_writes": len(rows),
        "n_flux_steps": len(flux), **ev, **f0r, **v3(states), "last_write_s": last_t,
        # V5 (design rev. 3 section 4): endTime reached, or the clock budget
        # with an outcome already decided; else INCONCLUSIVE.
        "v5": (None if last_t is None else
               "PASS" if (child.get("end_time_s") is not None
                          and last_t >= float(child["end_time_s"]) - 1e-4 * float(child["end_time_s"]))
               or ev["outcome"] in ("passes", "clogs", "arrests") else "INCONCLUSIVE"),
        "min_top_gap_m": min((r["top_gap_m"] for r in rows if r["top_gap_m"] is not None), default=None),
        "min_bottom_gap_m": min((r["bottom_gap_m"] for r in rows if r["bottom_gap_m"] is not None), default=None),
        "identity_failures": sum(1 for r in rows if r["identity_within_5pct"] is False),
        "max_leading_edge_x_m": max((r["leading_edge_x_m"] for r in rows
                                     if r["leading_edge_x_m"] is not None), default=None),
    }
    (out / "freeShellSummary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
        ts = [r["time_s"] * 1e3 for r in rows]
        ax[0].plot([t * 1e3 for t, _ in flux], [v / q0 for _, v in flux], lw=0.8)
        ax[0].set_ylabel("Q/Q0")
        ax[1].plot(ts, [r["dp_clot_derived_m2_s2"] for r in rows])
        ax[1].set_ylabel("dp_clot DERIVED [m2/s2]")
        ax[2].plot(ts, [r["leading_edge_x_m"] for r in rows], label="leading")
        ax[2].plot(ts, [r["trailing_edge_x_m"] for r in rows], label="trailing")
        ax[2].set_ylabel("x [m]")
        ax[2].legend()
        if ev["trigger_time_s"] is not None:
            for a in ax:
                a.axvline(ev["trigger_time_s"] * 1e3, color="r", lw=0.6)
        ax[2].set_xlabel("t [ms]")
        fig.suptitle(str(child.get("child_id")))
        fig.savefig(out / "freeShellTimeseries.png", dpi=120)
    except ImportError:
        print("matplotlib not available; CSV/JSON written, no PNG", file=sys.stderr)
    return 0


# --- self-tests (design rev. 3 section 6 item 4) -------------------------
def _synthetic(le, te, ratio_fn, t_end=25e-3, top_cells=30.0, bot_cells=30.0, noise=0.0):
    """Frames every 0.2 ms with LE(t), TE(t) [m]; steps every 0.02 ms with
    Q/q_start = ratio_fn(t). Gaps in cells (both walls)."""
    rng = np.random.default_rng(7)
    n = int(round(t_end / 2e-4))
    states = []
    for k in range(1, n + 1):
        t = k * 2e-4
        states.append({"time_s": t, "leading_edge_x_m": le(t) + noise * rng.standard_normal(),
                       "trailing_edge_x_m": te(t),
                       "top_gap_cells": top_cells(t) if callable(top_cells) else top_cells,
                       "bottom_gap_cells": bot_cells(t) if callable(bot_cells) else bot_cells})
    ratio = [(k * 2e-5, ratio_fn(k * 2e-5)) for k in range(1, int(round(t_end / 2e-5)) + 1)]
    return states, ratio


def self_test(csv_path: str | None) -> int:
    um = 1e-6
    lin = lambda x0, v: (lambda t: x0 + v * t)
    stop_at = lambda x0, v, t1: (lambda t: x0 + v * min(t, t1))
    cases = [
        ("passes: TE reaches 220 um", "passes",
         _synthetic(lin(145 * um, 7e-3), lin(115 * um, 7e-3), lambda t: 0.95)),
        ("entered: LE 224 um, TE 160 um at the end", "entered",
         _synthetic(lin(145 * um, 3.16e-3), lin(115 * um, 1.8e-3), lambda t: 0.95)),
        ("clogs: held drop, no touch (trigger by LE >= 175 um)", "clogs",
         _synthetic(lin(145 * um, 2.0e-3), lin(115 * um, 2.0e-3),
                    lambda t: 0.95 if t < 18e-3 else 0.70)),
        ("clogs: trigger by bottom-wall contact only", "clogs",
         _synthetic(lin(145 * um, 0.5e-3), lin(115 * um, 0.5e-3),
                    lambda t: 0.95 if t < 10e-3 else 0.75,
                    bot_cells=lambda t: 30.0 if t < 8e-3 else 0.6)),
        ("arrests: front stops at 185 um (noise 53 nm), no 20% drop", "arrests",
         _synthetic(stop_at(145 * um, 2.5e-3, 16e-3), stop_at(115 * um, 2.5e-3, 16e-3),
                    lambda t: 0.90, noise=53e-9)),
        ("receding: front moves back at 5e-4 m/s after 175 um", "undecided",
         _synthetic(lambda t: 145 * um + 2.5e-3 * min(t, 14e-3) - 5e-4 * max(0.0, t - 14e-3),
                    lin(115 * um, 1e-3), lambda t: 0.90, noise=53e-9)),
        ("undecided: never reaches the trigger", "undecided",
         _synthetic(lin(145 * um, 1e-3), lin(115 * um, 1e-3), lambda t: 0.95)),
        ("drop BEFORE the trigger does not clog", "undecided",
         _synthetic(lin(145 * um, 1e-3), lin(115 * um, 1e-3), lambda t: 0.70)),
    ]
    bad = 0
    for name, want, (states, ratio) in cases:
        got = classify(states, ratio, 25e-3)
        ok = got["outcome"] == want
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {name}: {got['outcome']} (want {want}); trigger "
              f"{got['trigger_kind']} at {got['trigger_time_s']}; fit {got['arrest_fit']}")
    if csv_path:
        rows = list(csv.DictReader(open(csv_path)))
        f = lambda r, k: None if r.get(k) in (None, "", "None") else float(r[k])
        states = [{"time_s": f(r, "time_s"), "leading_edge_x_m": f(r, "leading_edge_x_m"),
                   "trailing_edge_x_m": f(r, "trailing_edge_x_m"),
                   "top_gap_cells": f(r, "top_gap_cells"), "bottom_gap_cells": None} for r in rows]
        ratio = [(s["time_s"], f(r, "q_over_qstart")) for s, r in zip(states, rows)
                 if f(r, "q_over_qstart") is not None]
        got = classify(states, ratio, None)
        ok = got["outcome"] == "entered"
        bad += not ok
        last = states[-1]
        print(f"{'OK ' if ok else 'BAD'} batch 63 dense CSV ({len(states)} writes, last {last['time_s']:.5g} s, "
              f"LE {last['leading_edge_x_m']*1e6:.1f} um, TE {last['trailing_edge_x_m']*1e6:.1f} um): "
              f"{got['outcome']} (want entered, not passes)")
    print(f"self-test: {len(cases) + bool(csv_path) - bad} of {len(cases) + bool(csv_path)} as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
