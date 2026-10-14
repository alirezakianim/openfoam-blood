#!/usr/bin/env python3
"""SYNTHETIC returned cases for the Y gate (offline; pure python): the replica's cell centres and outlet face centres
(scripts/y_gate/fixtures/, built from the shipped dictionary by y_gate_fixture.py) carry fields that satisfy the gates' pre-registered
relations EXACTLY, so every gate can be shown to PASS on them and to FAIL, for its stated reason, when one thing is planted.

Used by scripts/check_y_gate.py and by dry_run.py (the Y child's synthesised output). NOT physics: the fields are the reference relations
(Poiseuille windows with the FV bias, the pre-registered split, a single decaying slow mode) laid on the real cell positions.

    case = build_case(root, ext, nu, **options)          # ext = the child's y_gate block, nu = the plasma's kinematic viscosity
Options (each plants ONE defect; the default is a clean, converged, READY run):
    split            q1/q0 (default: gate 2's expected value)       q0_scale   multiplies q0 (a mis-set drive)
    bias_sign        +1 (default) or -1: the sign of the 2/N^2 bias in the window gradients
    tau_slow_tau1    the slow mode's time constant in tau1 (default 1.0143, red team 11 N4); amp  its amplitude (default 1)
    dt               the (constant) time step in s (default gate 6's dt_est_s). The write times are the CHILD'S: writeControl runTime,
                     writeInterval tau1, so write k lands on the first step n with floor((n dt + dt/2) / tau1) >= k (OF-13 Time.C L1196-1208), for ANY dt
    every_n_steps    the REJECTED rev. 2.1 schedule (writeControl timeStep, writeInterval n): write k at step n k. Only for the negative that shows
                     why it was replaced (a step 5 % low puts T_d below 17.9 tau1)
    last_k           the last write is write number last_k (default 18 = 18 tau1 to within dt/2); skip_k  write numbers left out
    zero_face        (patch-local index) the outlet face whose written phi is 0 while the logged inletFlux keeps the true q0
    drop_entry       True: the written outlet phi has one entry fewer than the patch has faces
    flip_outlet_log  the logged outletFlux is perturbed by this relative amount (the instrument must catch it)
    content          q = 1 - content in one cell     theta_net   thetaNetRef in one cell     lam_amp  lambda1 - 1 (default 1.5e-3)
    gate1            False: no gate1.json is written
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import y_gate_report as R  # noqa: E402

FIX = HERE / "fixtures"
_CACHE: dict = {}


def fixtures():
    if "fix" not in _CACHE:
        n, nc, cells = R.read_npy_f8(FIX / "y_cells_2d.npy")
        m, mc, faces = R.read_npy_f8(FIX / "y_outlet_faces_2d.npy")
        _CACHE["fix"] = ([(cells[2 * i], cells[2 * i + 1]) for i in range(n)], [(faces[2 * i], faces[2 * i + 1]) for i in range(m)])
    return _CACHE["fix"]


def _fmt(v: float) -> str:
    return f"{v:.17g}"


def _header(cls: str, name: str) -> str:
    return f"FoamFile\n{{\n    format ascii;\n    class {cls};\n    object {name};\n}}\n"


def _vec_list(vals) -> str:
    return f"nonuniform List<vector>\n{len(vals)}\n(\n" + "\n".join(f"({_fmt(x)} {_fmt(y)} {_fmt(z)})" for x, y, z in vals) + "\n)\n"


def _scalar_list(vals) -> str:
    return f"nonuniform List<scalar>\n{len(vals)}\n(\n" + "\n".join(_fmt(v) for v in vals) + "\n)\n"


def _field(cls, name, internal, patches) -> str:
    body = "".join(f"    {p}\n    {{\n        {t}\n    }}\n" for p, t in patches.items())
    return _header(cls, name) + f"dimensions [0 0 0 0 0 0 0];\ninternalField {internal};\nboundaryField\n{{\n{body}}}\n"


def _calc(value: str) -> str:
    return f"type calculated;\n        value {value};"


def build_case(root: Path, ext: dict, nu: float, **o) -> Path:
    root = Path(root)
    geo, g4, g5, g6 = ext["geometry_sim"], ext["gate4"], ext["gate5"], ext["gate6"]
    D0, depth, W, N = geo["D0_m"], geo["depth_m"], geo["widths_m"], geo["N"]
    n = ext["mesh"]["counts"]["nCells"]
    cells2, faces2 = fixtures()
    z = 0.5 * depth
    # ---------------------------------------------------------------------------------------- flow numbers
    split = o.get("split", ext["gate2"]["expected"])
    q0_inf = g5["u_mean_sim_m_s"] * (1.0 + g5["fv_bias"]) * W[0] * depth * o.get("q0_scale", 1.0)
    q1_inf = split * q0_inf
    q2_inf = q0_inf - q1_inf
    q_inf = (q0_inf, q1_inf, q2_inf)
    sign = o.get("bias_sign", 1)
    tau_s = o.get("tau_slow_tau1", 1.0143) * g6["tau1_s"]
    amp = o.get("amp", 1.0)
    dt = float(o.get("dt", g6["dt_est_s"]))
    tau1 = g6["tau1_s"]

    def write_time(k: int) -> float:
        """The time of write k: the first step n with int((n dt + dt/2) / tau1) >= k (the child writes at runTime tau1; OF-13 Time.C L1196-1208)."""
        n = max(1, math.ceil(k * tau1 / dt - 0.5))
        while int((n * dt + 0.5 * dt) / tau1) < k:
            n += 1
        while n > 1 and int(((n - 1) * dt + 0.5 * dt) / tau1) >= k:
            n -= 1
        return n * dt
    last_k = o.get("last_k", 18)
    ks = [k for k in range(1, last_k + 1) if k not in set(o.get("skip_k", ()))]
    # ------------------------------------------------------------------------- spatial fields at unit amplitude
    node = geo["node_m"]
    axes = {k: (math.cos(math.radians(a)), math.sin(math.radians(a))) for k, a in geo["axes_deg"].items()}
    branch = []                                    # per cell: (section index, s, n) or None
    for (x, y) in cells2:
        rx, ry = x - node[0], y - node[1]
        pick = None
        if rx < 0.0:
            pick = (0, rx, ry)
        else:
            for sec, name in ((1, "d1"), (2, "d2")):
                e = axes[name]
                s = rx * e[0] + ry * e[1]
                nn = -rx * e[1] + ry * e[0]
                if s > 0.0 and abs(nn) <= 0.5 * W[sec]:
                    pick = (sec, s, nn) if pick is None else None      # a cell in both bands is a junction cell: left at 0
        branch.append(pick)
    gp = [12.0 * nu * (q_inf[k] / (W[k] * depth)) / W[k] ** 2 for k in range(3)]       # the Poiseuille pressure gradients (infinite time)
    gfv = [gp[k] / (1.0 + sign * 2.0 / N[k] ** 2) for k in range(3)]
    p_unit, u_unit = [], []
    for pick in branch:
        if pick is None:
            p_unit.append(0.0)
            u_unit.append((0.0, 0.0, 0.0))
            continue
        sec, s, nn = pick
        name = ("parent", "d1", "d2")[sec]
        e = axes[name]
        ubar = q_inf[sec] / (W[sec] * depth)
        prof = 1.5 * ubar * (1.0 - (2.0 * nn / W[sec]) ** 2)
        p_unit.append(-gfv[sec] * s + 1e-3)
        u_unit.append((prof * e[0], prof * e[1], 0.0))
    n_d1, n_d2 = sum(1 for f in faces2 if f[1] > 0), sum(1 for f in faces2 if f[1] < 0)
    lam = 1.0 + o.get("lam_amp", 1.5e-3)
    be_row = (lam ** 3, 0.0, 0.0, 1.0, 0.0, 1.0)
    # ------------------------------------------------------------------------------------------------ files
    (root / "0").mkdir(parents=True, exist_ok=True)
    (root / "constant").mkdir(exist_ok=True)
    cen = [(x, y, z) for (x, y) in cells2]
    fc = [(x, y, z) for (x, y) in faces2]
    pat = {"inlet": _calc("uniform (0 0 0)"), "wall": _calc("uniform (0 0 0)"), "front": "type empty;", "back": "type empty;",
           "outlet": _calc(_vec_list(fc).rstrip("\n"))}
    (root / "0" / "C").write_text(_field("volVectorField", "C", _vec_list(cen).rstrip("\n"), pat))
    flux_in, flux_out = [], []
    for k in ks:
        t = o["every_n_steps"] * k * dt if o.get("every_n_steps") else write_time(k)
        name = format(t, ".10g")
        t_num = float(name)                         # the directory name rounds to 10 digits, so does the logged time column
        f = 1.0 - amp * math.exp(-t / tau_s)
        d = root / name
        d.mkdir(exist_ok=True)
        q1, q2 = q1_inf * f, q2_inf * f
        phi_out = ([q1 / n_d1] * n_d1 + [q2 / n_d2] * n_d2)
        # outlet order = faces2 order (sorted by y then x in the fixture): d2 (y<0) faces first, d1 after
        order = [(q2 / n_d2) if fy < 0 else (q1 / n_d1) for (_fx, fy) in faces2]
        if o.get("zero_face") is not None:
            order[o["zero_face"]] = 0.0
        if o.get("drop_entry"):
            order = order[:-1]
        true_out = math.fsum(order) * (1.0 + o.get("flip_outlet_log", 0.0))
        phi_patches = {"inlet": _calc("uniform -1e-20"), "wall": _calc("uniform 0"), "front": "type empty;", "back": "type empty;",
                       "outlet": _calc(_scalar_list(order).rstrip("\n"))}
        (d / "phi").write_text(_header("surfaceScalarField", "phi") + "dimensions [0 3 -1 0 0 0 0];\ninternalField uniform 0;\n"
                               "boundaryField\n{\n" + "".join(f"    {p}\n    {{\n        {tt}\n    }}\n" for p, tt in phi_patches.items()) + "}\n")
        flux_in.append((t_num, -q0_inf * f))
        flux_out.append((t_num, true_out))
        if k >= ks[-1] - 3 or not o.get("light", True):
            zg = {"inlet": "type zeroGradient;", "outlet": "type zeroGradient;", "wall": "type zeroGradient;",
                  "front": "type empty;", "back": "type empty;"}
            (d / "p").write_text(_field("volScalarField", "p", _scalar_list([v * f for v in p_unit]).rstrip("\n"), zg))
            (d / "U").write_text(_field("volVectorField", "U", _vec_list([(a * f, b * f, c) for a, b, c in u_unit]).rstrip("\n"), zg))
            (d / "be").write_text(_header("volSymmTensorField", "be") + "dimensions [0 0 0 0 0 0 0];\ninternalField nonuniform List<symmTensor>\n"
                                  f"{n}\n(\n" + "\n".join(f"({' '.join(_fmt(v) for v in be_row)})" for _ in range(n)) + "\n)\n;\nboundaryField\n{\n}\n")
            q = [1.0] * n
            th = [0.0] * n
            if o.get("content") is not None:
                q[12345] = 1.0 - o["content"]
            if o.get("theta_net") is not None:
                th[23456] = o["theta_net"]
            (d / "q").write_text(_field("volScalarField", "q", _scalar_list(q).rstrip("\n"), zg))
            (d / "thetaNetRef").write_text(_field("volScalarField", "thetaNetRef", _scalar_list(th).rstrip("\n"), zg))
    for nm, rows in (("inletFlux", flux_in), ("outletFlux", flux_out)):
        dd = root / "postProcessing" / nm / "0"
        dd.mkdir(parents=True, exist_ok=True)
        extra = [(r[0] * 0.5, r[1] * 0.5) for r in rows]          # rows at other steps, so a lookup by time has company
        allrows = sorted(rows + extra)
        (dd / "surfaceFieldValue.dat").write_text(
            "# Region type : patch\n# Region name : " + nm[:-4] + "\n# Faces : 1\n# Area : 1\n# Time\torientedSum(phi)\n"
            + "".join(f"{t:.10g}\t{v:.17g}\n" for t, v in allrows))
    if o.get("gate1", True):
        write_gate1(root, ext)
    return root


def write_gate1(root: Path, ext: dict, **change) -> None:
    m = ext["mesh"]
    rec = {"blockMesh_exit": 0, "blockMesh_log_ends": True, "checkMesh_ok": True, "negative_volumes": 0,
           "non_orth_max_deg": 24.4074, "skewness_max": 0.6572, "counts": dict(m["counts"]), "counts_checkMesh": dict(m["counts"]),
           "patches": dict(m["patches"]),
           "points": {"format": "ascii", "gz_present": False, "n_ref": m["points_ref_n"], "n_plane": m["points_ref_n"],
                      "max_dist_real_to_ref_m": 1.3e-10, "max_dist_ref_to_real_m": 1.3e-10, "unmatched_real": 0, "unmatched_ref": 0},
           "outlet_split": {"d1_y_positive": 88, "d2_y_negative": 60, "on_axis": 0}, "roundtrip": {"ran": False, "why": "synthetic"},
           "error": None, "failing_checks": [], "ok": True, "why": []}
    rec.update(change)
    out = Path(root) / "postProcessing" / "yGate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate1.json").write_text(json.dumps(rec, indent=1) + "\n")


# ------------------------------------------------------------------------------------------ gate 1's mesh files
def build_mesh(root: Path, ext: dict, **o) -> Path:
    """Fake constant/polyMesh/{points,owner,boundary,faces}, log.checkMesh and log.blockMesh for y_mesh_gate.py. The z = 0 plane of `points` IS the
    reference (so the clean case passes with distance 0); the options plant one defect each:
      move=(i, dx, dy)   shifts reference point i       binary=True    header says binary    gz=True    a points.gz exists
      no_points=True     no points file                  extra_point=True   a 79,977th point on the plane
      cells_off=k        the owner header says nCells+k  non_orth=v     the checkMesh log says v   skew=v   likewise
      failed_checks=True 'Failed 1 mesh checks.'          neg_volume=k   the log reports k negative-volume cells
      flip_outlet=i      the i-th d1 outlet face has its y flipped (outlet split 87 / 61)"""
    root = Path(root)
    m = ext["mesh"]
    n, nc, ref = R.read_npy_f8(HERE.parents[1] / m["points_ref"])
    xy = [(ref[2 * i], ref[2 * i + 1]) for i in range(n)]
    pts = list(xy)
    if o.get("move"):
        i, dx, dy = o["move"]
        pts[i] = (xy[i][0] + dx, xy[i][1] + dy)
    if o.get("extra_point"):
        pts.append((xy[0][0] + 1e-3, xy[0][1] + 1e-3))
    poly = root / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    fmt = "binary" if o.get("binary") else "ascii"
    lines = [_header("vectorField", "points").replace("format ascii", f"format {fmt}") + "\n", f"{2 * len(pts)}\n(\n"]
    lines += [f"({_fmt(x)} {_fmt(y)} 0)\n" for x, y in pts] + [f"({_fmt(x)} {_fmt(y)} 1e-06)\n" for x, y in pts] + [")\n"]
    if not o.get("no_points"):
        (poly / "points").write_text("".join(lines))
    if o.get("gz"):
        (poly / "points.gz").write_bytes(b"\x1f\x8b")
    c = m["counts"]
    (poly / "owner").write_text(_header("labelList", "owner").replace("object owner;", f'note "nPoints:{c["nPoints"]}  nCells:{c["nCells"] + o.get("cells_off", 0)}  '
                                f'nFaces:{c["nFaces"]}  nInternalFaces:{c["nInternalFaces"]}";\n    object owner;') + "\n0\n(\n)\n")
    start, rows = c["nInternalFaces"], []
    for name in ("inlet", "outlet", "wall", "front", "back"):
        nf = m["patches"][name]
        rows.append(f"    {name}\n    {{\n        type {'empty' if name in ('front', 'back') else 'wall' if name == 'wall' else 'patch'};\n"
                    f"        nFaces {nf};\n        startFace {start};\n    }}\n")
        if name == "inlet":
            outlet_start = start + nf
        start += nf
    (poly / "boundary").write_text(_header("polyBoundaryMesh", "boundary") + f"\n{len(rows)}\n(\n" + "".join(rows) + ")\n")
    order = sorted(range(n), key=lambda i: xy[i][1])
    d2_pts, d1_pts = order[:m["outlet_faces"]["d2_y_negative"]], order[-m["outlet_faces"]["d1_y_positive"]:]
    face_lines = ["4(0 0 0 0)"] * outlet_start
    outl = [f"4({p} {p} {p} {p})" for p in d1_pts] + [f"4({p} {p} {p} {p})" for p in d2_pts]
    if o.get("flip_outlet") is not None:
        p = d2_pts[0]
        outl[o["flip_outlet"]] = f"4({p} {p} {p} {p})"
    face_lines += outl
    face_lines += ["4(0 0 0 0)"] * (c["nFaces"] - len(face_lines))
    (poly / "faces").write_text(_header("faceList", "faces") + f"\n{c['nFaces']}\n(\n" + "\n".join(face_lines) + "\n)\n")
    no = o.get("non_orth", 24.4074)
    sk = o.get("skew", 0.6572)
    vol = (f"Zero or negative cell volume detected.  Minimum negative volume: -1e-19, Number of negative volume cells: {o['neg_volume']}"
           if o.get("neg_volume") else "Min volume = 2.9e-19. Max volume = 5.7e-19.  Total volume = 1.1e-11.  Cell volumes OK.")
    (root / "log.checkMesh").write_text(
        "Mesh stats\n"
        f"    points:           {c['nPoints']}\n    internal points:  0\n    faces:            {c['nFaces']}\n"
        f"    internal faces:   {c['nInternalFaces']}\n    cells:            {c['nCells']}\n"
        "    faces per cell:   6\n\nChecking geometry...\n    Mesh has 2 geometric (non-empty/wedge) directions (1 1 0)\n"
        f"    {vol}\n    Mesh non-orthogonality Max: {no} average: 6.20747\n    Max skewness = {sk} OK.\n"
        + ("\nFailed 1 mesh checks.\n" if o.get("failed_checks") else "\nMesh OK.\n") + "\nEnd\n")
    (root / "log.blockMesh").write_text("Create time\n...\nEnd\n")
    (root / "constant").mkdir(exist_ok=True)
    (root / "constant" / "yGate.json").write_text(json.dumps(ext, indent=1) + "\n")
    return root
