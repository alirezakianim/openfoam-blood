#!/usr/bin/env python3
"""GATE 1 of the Y-bifurcation geometry gate (design 2026-10-14-case-y-bifurcation-gate.md rev. 2.1 section 4, gate 1; red team 11
change B). THE MESH-ONLY PRE-STEP: Allrun runs it after blockMesh and checkMesh and BEFORE foamRun; a non-zero exit makes the child's
setup fail, so the solver never starts and a mismatch costs minutes, not a session.

    y_mesh_gate.py <case dir> [--blockmesh-exit N]

Reads <case>/constant/yGate.json (the child's y_gate block, written by create_cases.py), log.checkMesh, log.blockMesh and
constant/polyMesh/{points,owner,boundary,faces}. Writes <case>/postProcessing/yGate/gate1.json. Exits NON-ZERO unless ALL of:
  * blockMesh exited 0 and checkMesh printed `Mesh OK.` with `Cell volumes OK.` (0 negative volumes);
  * the section 2 counts match EXACTLY: points, cells, faces, internal faces and every patch (from the polyMesh files, cross-checked
    against the checkMesh log's own `Mesh stats`);
  * max non-orthogonality is 24.41 +- 0.01 deg and max skewness 0.657 +- 0.01 (the checkMesh log);
  * THE POINTS: `points` says `format ascii` and no `points.gz` exists; the z = 0 plane holds 79,976 points in both the mesh and
    constant/yGate/na09_points_ref_2d.npy; the two-way max distance between them is <= 1e-9 m (the order of the points does not
    matter). The replica's own round trip is 1.3e-10 m (probes/na09_roundtrip_C.txt);
  * the outlet faces split 88 (y > 0) / 60 (y < 0), the split the per-daughter flux instrument reads.
An unreadable or missing file is a FAIL, never a crash that reads as a pass: every exception writes gate1.json with ok false and exits 1.

WRAPPER AROUND na09_roundtrip.py (shipped with na06_ymesh.py, which it imports): when numpy and scipy import (Colab: yes), the
wrapper also runs `na09_roundtrip.py <dict copy> --real <points>` and records its report line (replica -> real distance). That run is
EVIDENCE, not a gate: the gate is the real-versus-reference distance above, and the reference IS the replica's output. The point
comparison itself is pure python (a grid hash, no scipy), so this file runs under any python3.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import y_gate_report as R  # noqa: E402

MAIN_CASE = HERE.parents[1]
GRID_CELL_M = 2e-8          # the search cell of the point matcher: 20x the 1e-9 tolerance, 1/25 of the smallest point spacing
SENTINEL_M = 1.0            # a point with no neighbour within the search radius reads 1 m (finite: result.json forbids NaN/inf)
FLOAT = r"[-+0-9.eE]+"


def directed_max(a, b, cell=GRID_CELL_M):
    """max over points of `a` of the distance to the nearest point of `b`; points with no neighbour within one search cell
    read SENTINEL_M and are counted. Pure python grid hash. -> (max_dist, n_unmatched)."""
    grid: dict = {}
    for j, (x, y) in enumerate(b):
        grid.setdefault((math.floor(x / cell), math.floor(y / cell)), []).append(j)
    worst, unmatched = 0.0, 0
    for (x, y) in a:
        cx, cy = math.floor(x / cell), math.floor(y / cell)
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    d = math.hypot(x - b[j][0], y - b[j][1])
                    if best is None or d < best:
                        best = d
        if best is None:
            unmatched += 1
            worst = max(worst, SENTINEL_M)
        else:
            worst = max(worst, best)
    return worst, unmatched


def read_points(poly: Path):
    """-> (format, gz_present, [(x, y, z)]) from constant/polyMesh/points; ascii only (a binary file raises)."""
    gz = (poly / "points.gz").exists()
    p = poly / "points"
    if not p.is_file():
        raise FileNotFoundError(f"{p} is absent")
    raw = p.read_bytes()
    head = raw[:2000].decode("latin1")
    m = re.search(r"format\s+(\w+)\s*;", head)
    fmt = m.group(1) if m else None
    if fmt != "ascii":
        return fmt, gz, None
    text = raw.decode("latin1")
    body = text[text.index("(", text.index("\n", text.rindex("}", 0, 2000))):]
    pts = [(float(a), float(b), float(c)) for a, b, c in
           re.findall(rf"\(\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*\)", body)]
    return fmt, gz, pts


def owner_counts(poly: Path) -> dict:
    head = (poly / "owner").read_bytes()[:3000].decode("latin1")
    m = re.search(r'note\s+"nPoints:\s*(\d+)\s+nCells:\s*(\d+)\s+nFaces:\s*(\d+)\s+nInternalFaces:\s*(\d+)"', head)
    if not m:
        raise ValueError(f"{poly / 'owner'}: no `note \"nPoints:.. nCells:.. nFaces:.. nInternalFaces:..\"` in the header")
    return {"nPoints": int(m.group(1)), "nCells": int(m.group(2)), "nFaces": int(m.group(3)), "nInternalFaces": int(m.group(4))}


def boundary_patches(poly: Path) -> dict:
    text = (poly / "boundary").read_text()
    out = {}
    for m in re.finditer(r"(?m)^\s*(\w+)\s*\{([^{}]*)\}", text):
        body = m.group(2)
        nf, sf = re.search(r"nFaces\s+(\d+)\s*;", body), re.search(r"startFace\s+(\d+)\s*;", body)
        if nf and sf:
            out[m.group(1)] = {"nFaces": int(nf.group(1)), "startFace": int(sf.group(1))}
    if not out:
        raise ValueError(f"{poly / 'boundary'}: no patch with nFaces/startFace")
    return out


def outlet_split(poly: Path, patches: dict, pts) -> dict:
    """The y split of the outlet faces by the vertex-mean y of each face (the instrument splits by the face centre; the two agree
    in sign because the d1 outlet end is at y in [+1.37, +2.26] D0 and the d2 end at y in [-3.49, -3.14] D0)."""
    o = patches["outlet"]
    lines = (poly / "faces").read_text().splitlines()
    k = next(i for i, ln in enumerate(lines) if re.fullmatch(r"\s*\d+\s*", ln))
    n_faces = int(lines[k])
    first = k + 2                                   # count line, then `(`
    if lines[k + 1].strip() != "(":
        raise ValueError(f"{poly / 'faces'}: the face list does not open on the line after its count")
    pos = neg = zero = 0
    for i in range(o["startFace"], o["startFace"] + o["nFaces"]):
        ids = [int(t) for t in re.findall(r"\d+", lines[first + i].split("(", 1)[1])]
        y = math.fsum(pts[j][1] for j in ids) / len(ids)
        if y > 0:
            pos += 1
        elif y < 0:
            neg += 1
        else:
            zero += 1
    return {"d1_y_positive": pos, "d2_y_negative": neg, "on_axis": zero, "faces_in_file": n_faces}


def check_mesh_log(text: str) -> dict:
    def last(pat):
        m = re.findall(pat, text, re.M)
        return m[-1] if m else None
    no = last(r"Mesh non-orthogonality Max:\s*(" + FLOAT + r")\s+average:\s*(" + FLOAT + r")")
    sk = last(r"Max skewness\s*=\s*(" + FLOAT + ")")
    nneg = last(r"Number of negative volume cells:\s*(\d+)")
    return {
        "mesh_ok": bool(re.search(r"(?m)^\s*Mesh OK\.", text)) and not re.search(r"Failed \d+ mesh checks", text),
        "volumes_ok": "Cell volumes OK." in text and "Zero or negative cell volume" not in text,
        "negative_volumes": 0 if ("Cell volumes OK." in text and "Zero or negative cell volume" not in text)
        else (int(nneg) if nneg is not None else None),
        "non_orth_max_deg": float(no[0]) if no else None,
        "skewness_max": float(sk) if sk else None,
        "stats": {k: (int(v) if v else None) for k, v in (
            ("nPoints", last(r"(?m)^\s+points:\s+(\d+)\s*$")), ("nFaces", last(r"(?m)^\s+faces:\s+(\d+)\s*$")),
            ("nInternalFaces", last(r"(?m)^\s+internal faces:\s+(\d+)\s*$")), ("nCells", last(r"(?m)^\s+cells:\s+(\d+)\s*$")))},
    }


def roundtrip_evidence(case: Path, spec: dict) -> dict:
    """Run na09_roundtrip.py --real when numpy and scipy import. Evidence only; never gates."""
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as error:
        return {"ran": False, "why": f"{error}"}
    tmp = Path(tempfile.mkdtemp(prefix="y_roundtrip_"))
    try:
        d = tmp / "blockMeshDict"
        shutil.copyfile(case / "system" / "blockMeshDict", d)          # roundtrip writes <dict>.roundtrip_points.npy beside it
        r = subprocess.run([sys.executable, str(HERE / "na09_roundtrip.py"), str(d), "--real", str(case / "constant" / "polyMesh" / "points")],
                           capture_output=True, text=True, timeout=1200)
        line = next((ln.strip() for ln in r.stdout.splitlines() if "vs --real" in ln), None)
        return {"ran": True, "returncode": r.returncode, "report_line": line, "stderr_tail": r.stderr[-400:]}
    except (OSError, subprocess.SubprocessError) as error:
        return {"ran": True, "returncode": None, "error": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(case: Path, bm_exit: int) -> dict:
    spec = json.loads((case / "constant" / "yGate.json").read_text())
    m = spec["mesh"]
    poly = case / "constant" / "polyMesh"
    rec: dict = {"blockMesh_exit": bm_exit, "failing_checks": []}
    blog = case / "log.blockMesh"
    rec["blockMesh_log_ends"] = bool(blog.is_file() and "End" in blog.read_text(errors="replace")[-200:])
    cm = check_mesh_log((case / "log.checkMesh").read_text(errors="replace"))
    rec["checkMesh_ok"] = bool(cm["mesh_ok"] and cm["volumes_ok"])
    rec["negative_volumes"] = cm["negative_volumes"]
    rec["non_orth_max_deg"] = cm["non_orth_max_deg"]
    rec["skewness_max"] = cm["skewness_max"]
    rec["counts"] = owner_counts(poly)
    rec["counts_checkMesh"] = cm["stats"]
    if cm["stats"] != {k: rec["counts"][k] for k in cm["stats"]}:
        rec["failing_checks"].append(f"checkMesh's mesh stats {cm['stats']} differ from the polyMesh owner header {rec['counts']}")
    patches = boundary_patches(poly)
    rec["patches"] = {k: v["nFaces"] for k, v in patches.items()}
    fmt, gz, pts = read_points(poly)
    ref_n, ref_cols, ref = R.read_npy_f8(MAIN_CASE / m["points_ref"])
    ref_xy = [(ref[2 * i], ref[2 * i + 1]) for i in range(ref_n)]
    pr = {"format": fmt, "gz_present": gz, "n_ref": ref_n}
    if pts is not None:
        plane = [(x, y) for x, y, z in pts if abs(z - m["plane_z_m"]) < 1e-12]
        pr["n_plane"] = len(plane)
        d1, u1 = directed_max(plane, ref_xy)
        d2, u2 = directed_max(ref_xy, plane)
        pr.update({"max_dist_real_to_ref_m": d1, "max_dist_ref_to_real_m": d2, "unmatched_real": u1, "unmatched_ref": u2})
        rec["outlet_split"] = outlet_split(poly, patches, pts)
    rec["points"] = pr
    rec["roundtrip"] = roundtrip_evidence(case, spec)
    return rec


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    case = Path(argv[0]).resolve()
    bm_exit = int(argv[argv.index("--blockmesh-exit") + 1]) if "--blockmesh-exit" in argv else 0
    out = case / "postProcessing" / "yGate"
    out.mkdir(parents=True, exist_ok=True)
    try:
        rec = run(case, bm_exit)
        rec["error"] = None
    except Exception as error:      # noqa: BLE001 -- ANY failure to read is a FAIL (a crash must not read as a pass)
        rec = {"blockMesh_exit": bm_exit, "error": f"{type(error).__name__}: {error}"}
    (out / "gate1.json").write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
    try:
        spec = json.loads((case / "constant" / "yGate.json").read_text())
        verdict = R.gate1(case, spec)
        extra = rec.get("failing_checks") or []
        why = list(verdict.get("why", [])) + list(extra)
        ok = verdict["verdict"] == "PASS" and not extra
    except Exception as error:      # noqa: BLE001
        ok, why = False, [f"{type(error).__name__}: {error}"]
    rec["ok"] = bool(ok)
    rec["why"] = why
    (out / "gate1.json").write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
    print(f"GATE 1 {'PASS' if ok else 'FAIL'}: " + ("the mesh is the replica" if ok else "; ".join(why)))
    if rec.get("roundtrip", {}).get("report_line"):
        print("roundtrip evidence:", rec["roundtrip"]["report_line"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
