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
ARREST_SPEED_M_S = 2.34e-4  # 0.1 x 2.34e-3 m/s (S7.7)
ARREST_X_MIN_M = 175e-6     # "after it has passed 175 um" (S7.7)
CONTACT_CELLS = 1.0         # contact: min gap <= 1 local cell (S7.7)
V3_T0_S, V3_T1_S = 0.4e-3, 2.0e-3   # S8 V3
V3_ADVANCE_MIN_M = 1.87e-6          # 0.5 x 2.34e-3 m/s x 1.6 ms (S8 V3)
V5_AFTER_EVENT_S = 2.0e-3           # S8 V5
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
           "top_gap_cells": None, "centroid_x_m": None, "centroid_y_m": None,
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


# --- the event and V3 ------------------------------------------------------
def event(states: list[dict]) -> dict:
    """Earlier of contact (min gap <= 1 local cell) and arrest (leading-edge
    speed < 2.34e-4 m/s between two consecutive writes, both past 175 um).
    The arrest is dated at the second write of the pair (the first write at
    which it is observable). A tie records both kinds."""
    contact = next((s["time_s"] for s in states
                    if s["top_gap_cells"] is not None and s["top_gap_cells"] <= CONTACT_CELLS), None)
    arrest, arrest_speed = None, None
    for a, b in zip(states, states[1:]):
        xa, xb = a["leading_edge_x_m"], b["leading_edge_x_m"]
        if xa is None or xb is None or min(xa, xb) < ARREST_X_MIN_M or b["time_s"] <= a["time_s"]:
            continue
        v = (xb - xa) / (b["time_s"] - a["time_s"])
        if v < ARREST_SPEED_M_S:
            arrest, arrest_speed = b["time_s"], v
            break
    cands = [(t, k) for t, k in ((contact, "contact"), (arrest, "arrest")) if t is not None]
    if not cands:
        return {"event_time_s": None, "event_kind": None, "contact_time_s": None,
                "arrest_time_s": None, "arrest_speed_m_s": None}
    t0 = min(t for t, _ in cands)
    return {"event_time_s": t0, "event_kind": "+".join(k for t, k in cands if t == t0),
            "contact_time_s": contact, "arrest_time_s": arrest, "arrest_speed_m_s": arrest_speed}


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
    if len(sys.argv) != 2:
        raise SystemExit("usage: free_shell_timeseries.py <case_dir>")
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
    ev = event(states)
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
                     "probe_pair": pair, "dp_clot_m2_s2": dp_clot, "identity_within_5pct": ident})
    out = case / "postProcessing"
    out.mkdir(exist_ok=True)
    cols = ["time_s", "q_out_m3_s", "q_over_q0", "q_over_qstart", "probe_pair", "dp_clot_m2_s2",
            "identity_within_5pct", "leading_edge_x_m", "trailing_edge_x_m", "top_gap_m",
            "top_gap_cells", "centroid_x_m", "centroid_y_m", "aspect_ratio", "clot_cells"]
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
        "n_flux_steps": len(flux), **ev, **v3(states), "last_write_s": last_t,
        "v5_reached": (None if ev["event_time_s"] is None or last_t is None
                       else last_t >= ev["event_time_s"] + V5_AFTER_EVENT_S - 1e-12),
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
        ax[1].plot(ts, [r["dp_clot_m2_s2"] for r in rows])
        ax[1].set_ylabel("dp_clot [m2/s2]")
        ax[2].plot(ts, [r["leading_edge_x_m"] for r in rows], label="leading")
        ax[2].plot(ts, [r["trailing_edge_x_m"] for r in rows], label="trailing")
        ax[2].set_ylabel("x [m]")
        ax[2].legend()
        if ev["event_time_s"] is not None:
            for a in ax:
                a.axvline(ev["event_time_s"] * 1e3, color="r", lw=0.6)
        ax[2].set_xlabel("t [ms]")
        fig.suptitle(str(child.get("child_id")))
        fig.savefig(out / "freeShellTimeseries.png", dpi=120)
    except ImportError:
        print("matplotlib not available; CSV/JSON written, no PNG", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
