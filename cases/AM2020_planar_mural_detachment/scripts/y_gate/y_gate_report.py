#!/usr/bin/env python3
"""The Y-bifurcation geometry gate's assessor: gates 1-7 of research/designs/2026-10-14-case-y-bifurcation-gate.md
rev. 2.1 section 4, the per-daughter flux instrument of section 3, and nothing else.

PURE PYTHON ON PURPOSE (no numpy, no scipy): assess_case.py runs under plain `python3` in the dry run
(scripts/release.sh) and under Colab's python3 alike, and a numpy import here would make the two differ.

What each gate reads (all of it written by the child itself):
  gate 1  postProcessing/yGate/gate1.json, written by scripts/y_gate/y_mesh_gate.py BEFORE foamRun (the mesh-only
          pre-step). The verdict is RECOMPUTED here from the recorded numbers; the file's own `ok` is not trusted.
  gate 2  q1/q0 at the last written time T_d. q0 = -inletFlux (logged), q1 = the instrument.
  gate 3  |q1 + q2 - q0| / q0 at T_d.
  gate 4  the windowed pressure gradients at T_d: G_FV (1 + 2/N^2) = 12 nu Ubar / W^2, to 1e-3, Ubar = q / (W depth).
  gate 5  the parent mean q0 / (W0 depth) against the 0.94 cm/s centreline / the measured 1.5, x (1 + 2.5e-4).
  gate 6  the last four written times: q0, q1, q2 change < 1e-6 per interval; the full U field.
  gate 7  max(lambda1) - 1, max(thetaNetRef) and max content (1 - q) at T_d, from be, thetaNetRef and q.
THE INSTRUMENT (section 3): `phi` on the `outlet` patch, split by the y of the face centre (0/C boundaryField, written
by postProcess -func writeCellCentres; the same order as the patch's phi): y > 0 is d1, y < 0 is d2. At every time it
reads, its sum must reproduce the logged outletFlux (orientedSum(phi)) to 1e-10 relative, or the gate that reads it is an
INSTRUMENT_FAIL, labelled as such. A verdict is one of PASS, FAIL, INSTRUMENT_FAIL, NOT_RUN.
"""
from __future__ import annotations

import json
import math
import re
from array import array
from itertools import islice
from pathlib import Path

TIME_MATCH_REL = 1e-7          # a time directory name / a flux row time, 10 significant digits (timePrecision 10) either way;
                               # two steps are >= 6e-3 apart in relative terms, so 1e-7 cannot confuse them
INSTRUMENT_SUM_REL = 1e-10     # design section 3
GATES = ("g1", "g2", "g3", "g4", "g5", "g6", "g7")


# ----------------------------------------------------------------------------------------------- readers
def read_npy_f8(path: Path):
    """A 2-D little-endian float64 .npy -> (n, ncol, array('d')). Pure python (the numpy format is a header and raw bytes)."""
    raw = Path(path).read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{path}: not a .npy file")
    major = raw[6]
    hlen = int.from_bytes(raw[8:10], "little") if major == 1 else int.from_bytes(raw[8:12], "little")
    off = (10 if major == 1 else 12) + hlen
    header = raw[off - hlen:off].decode("latin1")
    if "'<f8'" not in header or "'fortran_order': False" not in header:
        raise ValueError(f"{path}: header {header!r} is not little-endian float64, C order")
    shape = re.search(r"'shape':\s*\((\d+),\s*(\d+)\)", header)
    if not shape:
        raise ValueError(f"{path}: header {header!r} has no 2-D shape")
    n, ncol = int(shape.group(1)), int(shape.group(2))
    data = array("d")
    data.frombytes(raw[off:])
    if len(data) != n * ncol:
        raise ValueError(f"{path}: {len(data)} values for shape ({n}, {ncol})")
    return n, ncol, data


def _parse_value(s: str, ncomp: int, where: str):
    """`s` begins right after the keyword (internalField / value). -> ('uniform', v) or ('nonuniform', [v...])."""
    s = s.lstrip()
    if s.startswith("uniform"):
        s = s[len("uniform"):]
        toks = s[:s.index(";")].replace("(", " ").replace(")", " ").split()
        v = [float(t) for t in toks]
        if len(v) != ncomp:
            raise ValueError(f"{where}: uniform value has {len(v)} components, expected {ncomp}")
        return "uniform", (v[0] if ncomp == 1 else tuple(v))
    if not s.startswith("nonuniform"):
        raise ValueError(f"{where}: neither uniform nor nonuniform")
    s = s[len("nonuniform"):].lstrip()
    if s.startswith("List<"):
        s = s[s.index(">") + 1:].lstrip()
    m = re.match(r"(\d+)\s*", s)
    if not m:
        raise ValueError(f"{where}: no list count")
    n = int(m.group(1))
    s = s[m.end():]
    if n == 0:
        return "nonuniform", []
    if not s.startswith("("):
        raise ValueError(f"{where}: list of {n} does not open with a parenthesis")
    s = s[1:]
    if ncomp == 1:
        body = s[:s.index(")")]
        vals = [float(t) for t in body.split()]
        if len(vals) != n:
            raise ValueError(f"{where}: {len(vals)} values, header {n}")
        return "nonuniform", vals
    entries = list(islice((tuple(float(t) for t in m_.group(1).split())
                           for m_ in re.finditer(r"\(([^()]*)\)", s)), n))
    if len(entries) != n or any(len(e) != ncomp for e in entries):
        raise ValueError(f"{where}: {len(entries)} entries of {ncomp} components, header {n}")
    return "nonuniform", entries


def read_internal(path: Path, ncomp: int, n: int):
    """internalField of a volField file as a list of n values (scalars, or ncomp-tuples). Uniform is broadcast."""
    text = Path(path).read_text()
    i = text.find("internalField")
    if i < 0:
        raise ValueError(f"{path}: no internalField")
    kind, v = _parse_value(text[i + len("internalField"):], ncomp, str(path))
    if kind == "uniform":
        return [v] * n
    if len(v) != n:
        raise ValueError(f"{path}: {len(v)} values, the mesh has {n} cells")
    return v


def read_internal_faces(path: Path, ncomp: int = 1):
    """internalField of a surfaceScalarField (phi) as it is, without a count check (the caller checks)."""
    text = Path(path).read_text()
    i = text.find("internalField")
    kind, v = _parse_value(text[i + len("internalField"):], ncomp, str(path))
    return kind, v


def _brace_block(text: str, start: int) -> str:
    """The text between the `{` at or after `start` and its matching `}`."""
    i = text.index("{", start)
    depth, j = 0, i
    while True:
        c = text[j]
        depth += (c == "{") - (c == "}")
        j += 1
        if depth == 0:
            return text[i + 1:j - 1]


def read_patch_value(path: Path, patch: str, ncomp: int):
    """boundaryField.<patch>.value as ('uniform', v) or ('nonuniform', [..]). Raises if the patch or its value is absent."""
    text = Path(path).read_text()
    b = text.find("boundaryField")
    if b < 0:
        raise ValueError(f"{path}: no boundaryField")
    m = re.search(rf"(?m)^\s*{re.escape(patch)}\s*\{{", text[b:])
    if not m:
        raise ValueError(f"{path}: no patch {patch!r} in boundaryField")
    block = _brace_block(text, b + m.start())
    k = re.search(r"\bvalue\s", block)
    if not k:
        raise ValueError(f"{path}: patch {patch!r} has no value entry")
    return _parse_value(block[k.end():], ncomp, f"{path}:{patch}")


def read_flux_series(case: Path, name: str, column: str = "orientedSum(phi)") -> dict:
    """{time: value} of one column of postProcessing/<name>/<start>/surfaceFieldValue.dat, restarts merged, later wins. The mirror of
    free_shell_timeseries.read_surface_field_value (L124-148; that module imports numpy at import time, so it cannot be used here)."""
    base = Path(case) / "postProcessing" / name
    starts = sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: float(d.name)) if base.is_dir() else []
    files = [d / "surfaceFieldValue.dat" for d in starts if (d / "surfaceFieldValue.dat").is_file()]
    if not files:
        raise FileNotFoundError(f"{case}: no postProcessing/{name}/<t>/surfaceFieldValue.dat")
    rows: dict = {}
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
    return rows


def _flux_at(series: dict, t: float):
    for tt, v in series.items():
        if math.isclose(tt, t, rel_tol=TIME_MATCH_REL, abs_tol=1e-15):
            return v
    return None


def written_times(case: Path):
    """[(t, dir)] of the FULL writes: numeric directories above 0 that hold phi, ascending."""
    out = []
    for d in Path(case).iterdir():
        if not d.is_dir():
            continue
        try:
            t = float(d.name)
        except ValueError:
            continue
        if t > 0.0 and (d / "phi").is_file():
            out.append((t, d))
    return sorted(out)


# ------------------------------------------------------------------------------------------- the instrument
class Instrument:
    """Per-daughter outlet flux from the written phi."""

    def __init__(self, case: Path, spec: dict):
        self.case = Path(case)
        self.ncells = spec["mesh"]["counts"]["nCells"]
        self.want = spec["mesh"]["outlet_faces"]
        c = self.case / "0" / "C"
        kind, faces = read_patch_value(c, "outlet", 3)
        if kind != "nonuniform":
            raise ValueError(f"{c}: the outlet face centres are not a nonuniform list")
        self.nface = len(faces)
        self.d1 = [i for i, f in enumerate(faces) if f[1] > 0.0]
        self.d2 = [i for i, f in enumerate(faces) if f[1] < 0.0]
        self.on_axis = self.nface - len(self.d1) - len(self.d2)
        self.problems = []
        if self.nface != spec["mesh"]["patches"]["outlet"]:
            self.problems.append(f"0/C has {self.nface} outlet face centres, the mesh has {spec['mesh']['patches']['outlet']}")
        if len(self.d1) != self.want["d1_y_positive"] or len(self.d2) != self.want["d2_y_negative"] or self.on_axis:
            self.problems.append(f"the y split of the outlet faces is {len(self.d1)} (y>0) / {len(self.d2)} (y<0) / {self.on_axis} on y=0; "
                                 f"the mesh is {self.want['d1_y_positive']} / {self.want['d2_y_negative']} / 0")
        self.outlet_log = read_flux_series(case, "outletFlux")
        self.inlet_log = read_flux_series(case, "inletFlux")

    def at(self, tdir: Path, t: float) -> dict:
        """-> {q1, q2, sum, logged_outlet, rel_err, q0_logged}. Raises ValueError with a reason when the write cannot be read."""
        kind, v = read_patch_value(tdir / "phi", "outlet", 1)
        if kind != "nonuniform" or len(v) != self.nface:
            raise ValueError(f"{tdir / 'phi'}: outlet phi is {kind} with {len(v) if kind == 'nonuniform' else '-'} faces, "
                             f"the patch has {self.nface}")
        if not all(math.isfinite(x) for x in v):
            raise ValueError(f"{tdir / 'phi'}: non-finite outlet phi")
        q1 = math.fsum(v[i] for i in self.d1)
        q2 = math.fsum(v[i] for i in self.d2)
        tot = math.fsum(v)
        logged = _flux_at(self.outlet_log, t)
        inlet = _flux_at(self.inlet_log, t)
        if logged is None or inlet is None:
            raise ValueError(f"no logged outletFlux/inletFlux row at t={t!r} (postProcessing/{{outletFlux,inletFlux}})")
        if logged == 0.0:
            raise ValueError(f"the logged outletFlux at t={t!r} is exactly 0")
        rel = abs(tot - logged) / abs(logged)
        return {"q1": q1, "q2": q2, "sum": tot, "logged_outlet": logged, "rel_err": rel, "q0_logged": -inlet,
                "inlet_logged": inlet}


# --------------------------------------------------------------------------------------------------- gates
def _verdict(v: str, **kw) -> dict:
    d = {"verdict": v}
    d.update(kw)
    return d


def gate1(case: Path, spec: dict) -> dict:
    """Recompute the verdict from postProcessing/yGate/gate1.json (written by y_mesh_gate.py before foamRun)."""
    p = Path(case) / "postProcessing" / "yGate" / "gate1.json"
    if not p.is_file():
        return _verdict("FAIL", why=f"{p.relative_to(case)} is absent: the mesh-only pre-step did not run or did not finish")
    try:
        rec = json.loads(p.read_text())
        m = spec["mesh"]
        why = []
        if rec.get("blockMesh_exit") != 0:
            why.append(f"blockMesh exit {rec.get('blockMesh_exit')!r}")
        if rec.get("checkMesh_ok") is not True:
            why.append("checkMesh did not report 'Mesh OK.' (or reported failed checks / a negative volume)")
        if rec.get("negative_volumes") != 0:
            why.append(f"negative volumes {rec.get('negative_volumes')!r}")
        for k, want in m["counts"].items():
            if rec.get("counts", {}).get(k) != want:
                why.append(f"{k} {rec.get('counts', {}).get(k)!r} != {want}")
        for k, want in m["patches"].items():
            if rec.get("patches", {}).get(k) != want:
                why.append(f"patch {k} {rec.get('patches', {}).get(k)!r} != {want}")
        no = rec.get("non_orth_max_deg")
        if not isinstance(no, (int, float)) or abs(no - m["non_orth_max_deg"]) > m["non_orth_tol_deg"]:
            why.append(f"non-orthogonality max {no!r} deg is not {m['non_orth_max_deg']} +- {m['non_orth_tol_deg']}")
        sk = rec.get("skewness_max")
        if not isinstance(sk, (int, float)) or abs(sk - m["skew_max"]) > m["skew_tol"]:
            why.append(f"skewness max {sk!r} is not {m['skew_max']} +- {m['skew_tol']}")
        pts = rec.get("points", {})
        if pts.get("format") != "ascii" or pts.get("gz_present") is not False:
            why.append(f"points header format {pts.get('format')!r}, points.gz present {pts.get('gz_present')!r}")
        if pts.get("n_plane") != m["points_ref_n"] or pts.get("n_ref") != m["points_ref_n"]:
            why.append(f"points on the z=0 plane {pts.get('n_plane')!r} (ref {pts.get('n_ref')!r}), want {m['points_ref_n']} in both")
        for k in ("max_dist_real_to_ref_m", "max_dist_ref_to_real_m"):
            d = pts.get(k)
            if not isinstance(d, (int, float)) or not math.isfinite(d) or d > m["points_tol_m"]:
                why.append(f"{k} {d!r} exceeds {m['points_tol_m']}")
        sp = rec.get("outlet_split", {})
        if sp.get("d1_y_positive") != m["outlet_faces"]["d1_y_positive"] or sp.get("d2_y_negative") != m["outlet_faces"]["d2_y_negative"]:
            why.append(f"outlet split {sp!r}, want {m['outlet_faces']!r}")
        if rec.get("error"):
            why.append(f"the pre-step recorded an error: {rec['error']}")
        why.extend(f"pre-step check: {c}" for c in rec.get("failing_checks") or [])
        return _verdict("FAIL" if why else "PASS", why=why, record=rec)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return _verdict("FAIL", why=[f"gate1.json unreadable: {type(error).__name__}: {error}"])


def _dirs_for_window(times, spec):
    g6 = spec["gate6"]
    tau1 = g6["tau1_s"]
    chosen = times[-4:]
    why = []
    if len(chosen) < 4:
        return chosen, [f"only {len(chosen)} written times with phi (4 needed)"]
    ts = [t for t, _ in chosen]
    if ts[-1] < g6["t_d_min_tau1"] * tau1:
        why.append(f"T_d {ts[-1]:.6e} s = {ts[-1] / tau1:.3f} tau1 < {g6['t_d_min_tau1']} tau1")
    if ts[0] < g6["t_a_min_tau1"] * tau1:
        why.append(f"T_a {ts[0]:.6e} s = {ts[0] / tau1:.3f} tau1 < {g6['t_a_min_tau1']} tau1")
    for a, b in zip(ts, ts[1:]):
        r = (b - a) / tau1
        if not (g6["interval_tau1"][0] <= r <= g6["interval_tau1"][1]):
            why.append(f"interval {a:.6e} -> {b:.6e} is {r:.3f} tau1, outside {g6['interval_tau1']}")
    return chosen, why


def _window_cells(centres, node, e, s0, s1, half):
    for k, (x, y, _z) in enumerate(centres):
        rx, ry = x - node[0], y - node[1]
        s = rx * e[0] + ry * e[1]
        if s0 <= s <= s1:
            n = -rx * e[1] + ry * e[0]
            if abs(n) <= half:
                yield k, s


def _fit_gradient(centres, p, node, e, s0, s1, half):
    ks, ss = [], []
    for k, s in _window_cells(centres, node, e, s0, s1, half):
        ks.append(k)
        ss.append(s)
    if len(ks) < 50:
        raise ValueError(f"only {len(ks)} cells in the window s in [{s0:.4e}, {s1:.4e}] m, band {half:.4e} m")
    sm = math.fsum(ss) / len(ss)
    pm = math.fsum(p[k] for k in ks) / len(ks)
    sxx = math.fsum((s - sm) ** 2 for s in ss)
    sxp = math.fsum((s - sm) * (p[k] - pm) for s, k in zip(ss, ks))
    return sxp / sxx, ks


def eig_symm3(a, b, c, d, e_, f):
    """Eigenvalues (ascending) of [[a b c][b d e_][c e_ f]] (Smith 1961, the closed form)."""
    p1 = b * b + c * c + e_ * e_
    q = (a + d + f) / 3.0
    if p1 == 0.0:
        return tuple(sorted((a, d, f)))
    p2 = (a - q) ** 2 + (d - q) ** 2 + (f - q) ** 2 + 2.0 * p1
    p = math.sqrt(p2 / 6.0)
    B = [[(a - q) / p, b / p, c / p], [b / p, (d - q) / p, e_ / p], [c / p, e_ / p, (f - q) / p]]
    detB = (B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1]) - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
            + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]))
    r = max(-1.0, min(1.0, detB / 2.0))
    phi = math.acos(r) / 3.0
    e3 = q + 2.0 * p * math.cos(phi)
    e1 = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    return e1, e2, e3


def lambda1_max(be):
    """max over cells of the largest principal stretch: sqrt(largest eigenvalue of be * det(be)^(-1/3)) (the replay of
    probes/na03_gtan.py: l = sqrt(ev det^(-1/3)))."""
    best = 0.0
    for (xx, xy, xz, yy, yz, zz) in be:
        det = (xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz))
        if not det > 0.0:
            return 1e300                                  # finite on purpose: result.json is written with allow_nan=False
        ev = eig_symm3(xx, xy, xz, yy, yz, zz)
        best = max(best, math.sqrt(ev[2] * det ** (-1.0 / 3.0)))
    return best


def assess(case: Path, spec: dict, nu: float, completed: bool) -> dict:
    """All seven gates -> {gate_id: verdict dict, 'verdict': READY | NOT_READY, 'instrument': {...}}. `spec` is child.json's y_gate."""
    case = Path(case)
    out = {g: _verdict("NOT_RUN", why="not reached") for g in GATES}
    out["g1"] = gate1(case, spec)
    notes = []
    if out["g1"]["verdict"] != "PASS":
        for g in GATES[1:]:
            out[g] = _verdict("NOT_RUN", why="gate 1 (the mesh) did not pass: the solver never ran, or the replica is wrong for the real mesh")
        return _finish(out, notes, None)
    if not completed:
        for g in GATES[1:]:
            out[g] = _verdict("NOT_RUN", why="the child did not reach its end time (see exit code, log.phaseClotAxisymmetricFluid)")
        return _finish(out, notes, None)

    geo = spec["geometry_sim"]
    D0, depth = geo["D0_m"], geo["depth_m"]
    W, N = geo["widths_m"], geo["N"]
    n = spec["mesh"]["counts"]["nCells"]
    times = written_times(case)
    # ---- the instrument ------------------------------------------------------------------------------------
    try:
        inst = Instrument(case, spec)
    except (OSError, ValueError, KeyError) as error:
        why = f"instrument could not be built: {type(error).__name__}: {error}"
        for g in GATES[1:]:
            out[g] = _verdict("INSTRUMENT_FAIL", why=why)
        return _finish(out, notes, {"error": why})
    inst_info = {"nface": inst.nface, "d1": len(inst.d1), "d2": len(inst.d2), "problems": list(inst.problems), "reads": []}
    if inst.problems:
        for g in GATES[1:]:
            out[g] = _verdict("INSTRUMENT_FAIL", why="; ".join(inst.problems))
        return _finish(out, notes, inst_info)
    if not times:
        for g in GATES[1:]:
            out[g] = _verdict("INSTRUMENT_FAIL", why="no written time directory holds phi")
        return _finish(out, notes, inst_info)

    def read_at(t, d):
        r = inst.at(d, t)
        inst_info["reads"].append({"t": t, "rel_err": r["rel_err"], "q1": r["q1"], "q2": r["q2"], "sum": r["sum"], "logged": r["logged_outlet"]})
        if not r["rel_err"] <= INSTRUMENT_SUM_REL:
            raise ValueError(f"at t={t:.9e} the instrument's outlet sum {r['sum']!r} differs from the logged outletFlux "
                             f"{r['logged_outlet']!r} by {r['rel_err']:.3e} relative (limit {INSTRUMENT_SUM_REL:g})")
        if not (r["logged_outlet"] > 0.0 and r["inlet_logged"] < 0.0):
            raise ValueError(f"at t={t:.9e} the logged fluxes have the wrong sign: outletFlux {r['logged_outlet']!r}, "
                             f"inletFlux {r['inlet_logged']!r} (outflow positive, inflow negative)")
        return r

    td_t, td_dir = times[-1]
    try:
        rd = read_at(td_t, td_dir)
    except (OSError, ValueError) as error:
        for g in ("g2", "g3", "g4", "g5"):
            out[g] = _verdict("INSTRUMENT_FAIL", why=str(error))
        rd = None
    q0, q1, q2 = (rd["q0_logged"], rd["q1"], rd["q2"]) if rd else (None, None, None)

    # ---- gate 2 ---------------------------------------------------------------------------------------------
    g2 = spec["gate2"]
    if rd:
        v = q1 / q0
        out["g2"] = _verdict("PASS" if abs(v - g2["expected"]) <= g2["tol"] else "FAIL", value=v, expected=g2["expected"],
                             tol=g2["tol"], at_t=td_t, q0=q0, q1=q1, q2=q2)
        # ---- gate 3 -----------------------------------------------------------------------------------------
        g3 = spec["gate3"]
        v3 = abs(q1 + q2 - q0) / q0
        out["g3"] = _verdict("PASS" if v3 <= g3["tol"] else "FAIL", value=v3, tol=g3["tol"], at_t=td_t,
                             instrument_floor=max(r_["rel_err"] for r_ in inst_info["reads"]))
        # ---- gate 5 -----------------------------------------------------------------------------------------
        g5 = spec["gate5"]
        u_sim = q0 / (W[0] * depth)
        expected = g5["u_mean_sim_m_s"] * (1.0 + g5["fv_bias"])
        v5 = u_sim / expected - 1.0
        out["g5"] = _verdict("PASS" if abs(v5) <= g5["tol"] else "FAIL", value=v5, tol=g5["tol"], u_mean_sim_m_s=u_sim,
                             expected_sim_m_s=expected, u_mean_phys_cm_s=u_sim * g5["sim_to_phys_velocity"] * 100.0, at_t=td_t)
    # ---- gate 4 (needs p and U at T_d, the cell centres, and the section fluxes) ---------------------------------
    g4 = spec["gate4"]
    if rd:
        try:
            centres = read_internal(case / "0" / "C", 3, n)
            p = read_internal(td_dir / "p", 1, n)
            if not all(math.isfinite(x) for x in p):
                raise ValueError(f"{td_dir / 'p'}: non-finite pressure")
            node = geo["node_m"]
            secs = {}
            for k, (name, qn) in enumerate((("parent", q0), ("d1", q1), ("d2", q2))):
                ang = math.radians(geo["axes_deg"][name])
                e = (math.cos(ang), math.sin(ang))
                w0, w1 = g4["windows_D0"][name]
                half = g4["band_fraction"] * W[k]
                G, ks = _fit_gradient(centres, p, node, e, w0 * D0, w1 * D0, half)
                ubar = qn / (W[k] * depth)
                g_pois = 12.0 * nu * ubar / W[k] ** 2
                ratio = abs(G) * (1.0 + 2.0 / N[k] ** 2) / g_pois - 1.0
                secs[name] = {"G": G, "ncells": len(ks), "Ubar_m_s": ubar, "G_pois": g_pois, "rel_err": ratio,
                              "sign_ok": G < 0.0, "pass": bool(G < 0.0 and abs(ratio) <= g4["tol"])}
            # REPORTED, NOT GATED (design section 4 gate 5): the parent window's centreline maximum, against 0.94 cm/s. The nearest
            # cell centre sits h/2 off the axis, so the maximum of the cell-centred U.e reads below the true centreline value.
            ang = math.radians(geo["axes_deg"]["parent"])
            e = (math.cos(ang), math.sin(ang))
            w0, w1 = g4["windows_D0"]["parent"]
            u = read_internal(td_dir / "U", 3, n)
            best = max((u[k][0] * e[0] + u[k][1] * e[1]) for k, _s in _window_cells(centres, node, e, w0 * D0, w1 * D0,
                                                                                  g4["band_fraction"] * W[0]))
            secs["parent"]["centreline_max_m_s"] = best
            secs["parent"]["centreline_max_phys_cm_s"] = best * spec["gate5"]["sim_to_phys_velocity"] * 100.0
            allp = all(s["pass"] for s in secs.values())
            out["g4"] = _verdict("PASS" if allp else "FAIL", sections=secs, tol=g4["tol"], at_t=td_t,
                                 failing=[k for k, s in secs.items() if not s["pass"]])
            if "centreline_max_phys_cm_s" in secs["parent"]:
                out["g5"]["centreline_max_phys_cm_s_REPORTED_NOT_GATED"] = secs["parent"]["centreline_max_phys_cm_s"]
        except (OSError, ValueError, KeyError) as error:
            out["g4"] = _verdict("INSTRUMENT_FAIL", why=f"{type(error).__name__}: {error}")

    # ---- gate 6 ---------------------------------------------------------------------------------------------
    g6 = spec["gate6"]
    chosen, why = _dirs_for_window(times, spec)
    if why:
        out["g6"] = _verdict("INSTRUMENT_FAIL", why="; ".join(why), times=[t for t, _ in chosen])
    else:
        try:
            series = {"q0": [], "q1": [], "q2": []}
            for t, d in chosen:
                r = read_at(t, d)
                series["q0"].append(r["q0_logged"])
                series["q1"].append(r["q1"])
                series["q2"].append(r["q2"])
            worst = {}
            for k, vals in series.items():
                worst[k] = max(abs(b - a) / abs(vals[-1]) for a, b in zip(vals, vals[1:]))
            u_d = read_internal(chosen[-1][1] / "U", 3, n)
            u_c = read_internal(chosen[-2][1] / "U", 3, n)
            if not all(math.isfinite(x) for v in u_d for x in v):
                raise ValueError(f"{chosen[-1][1] / 'U'}: non-finite velocity")
            umax = max(math.sqrt(sum(x * x for x in v)) for v in u_d)
            dU = max(math.sqrt(sum((a - b) ** 2 for a, b in zip(v, w))) for v, w in zip(u_d, u_c))
            if not umax > 0.0:
                raise ValueError(f"{chosen[-1][1] / 'U'}: U is identically zero")
            field = dU / umax
            ok = all(v < g6["tol"] for v in worst.values()) and field < g6["tol"]
            out["g6"] = _verdict("PASS" if ok else "FAIL", flux_max_interval_change=worst, field_change=field, tol=g6["tol"],
                                 times=[t for t, _ in chosen], times_tau1=[t / g6["tau1_s"] for t, _ in chosen])
        except (OSError, ValueError, KeyError) as error:
            out["g6"] = _verdict("INSTRUMENT_FAIL", why=f"{type(error).__name__}: {error}")

    # ---- gate 7 ---------------------------------------------------------------------------------------------
    g7 = spec["gate7"]
    try:
        be = read_internal(td_dir / "be", 6, n)
        lam = lambda1_max(be)
        tn = read_internal(td_dir / "thetaNetRef", 1, n)
        q = read_internal(td_dir / "q", 1, n)
        content = max(1.0 - x for x in q)
        tnmax = max(tn)
        ok = (lam - 1.0 <= g7["lambda1_minus_1_max"] and tnmax == 0.0 and content <= g7["content_max"])
        out["g7"] = _verdict("PASS" if ok else "FAIL", lambda1_minus_1=lam - 1.0, lambda1_limit=g7["lambda1_minus_1_max"],
                             thetaNetRef_max=tnmax, content_max=content, content_limit=g7["content_max"], at_t=td_t)
    except (OSError, ValueError, KeyError) as error:
        out["g7"] = _verdict("INSTRUMENT_FAIL", why=f"{type(error).__name__}: {error}")
    return _finish(out, notes, inst_info)


def _finish(out: dict, notes: list, inst) -> dict:
    ready = all(out[g]["verdict"] == "PASS" for g in GATES)
    out["verdict"] = "READY" if ready else "NOT_READY"
    out["failing"] = [g for g in GATES if out[g]["verdict"] != "PASS"]
    out["instrument"] = inst
    out["notes"] = notes
    return out
