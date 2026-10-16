#!/usr/bin/env python3
"""Build the offline test fixture of the Y gate from the SHIPPED dictionary: the replica's cell centres (78,912) and outlet face centres
(148), in the z = 0 plane, as float64 .npy files in scripts/y_gate/fixtures/. Needs numpy and scipy (run it with a python that has both,
e.g. /usr/local/bin/python3); the checks that USE the fixture (check_y_gate.py, dry_run.py) are pure python.

The replica is na09_roundtrip.py's (rebuilt from the dictionary text only; it reproduces the points of a real blockMesh output to 6.4e-16 m
on arc edges, probes/na09_roundtrip_cyl.txt, and the optimiser's mesh to 1.3e-10 m). A cell centre here is the mean of the cell's four
corners, not OpenFOAM's volume-weighted centroid: the difference is a fraction of a cell and the fixture is for SYNTHETIC fields only.
The outlet faces are the 2-D boundary edges whose midpoint lies on one of the four straight outlet segments named by the dictionary's
`outlet` patch faces; there must be exactly 148, split 88 (y > 0) / 60 (y < 0).

    python3 y_gate_fixture.py            (writes fixtures/y_cells_2d.npy, fixtures/y_outlet_faces_2d.npy and prints a summary)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import na09_roundtrip as R  # noqa: E402

MAIN_CASE = HERE.parents[1]


def build(dict_path=None, with_info=False):
    V, B, E, P = R.parse(str(dict_path or (MAIN_CASE / "constant" / "yGate" / "na09_blockMeshDict")))
    Pu, Cu, tag, tol = R.build2d(V, B, E, 2)
    A = 0.5 * np.sum(Pu[Cu][:, :, 0] * np.roll(Pu[Cu][:, :, 1], -1, 1) - np.roll(Pu[Cu][:, :, 0], -1, 1) * Pu[Cu][:, :, 1], 1)
    if (A < 0).all():
        Cu = Cu[:, ::-1]
    assert (A != 0).all()
    centres = Pu[Cu].mean(axis=1)
    # boundary edges: edges used by exactly one cell
    e = np.sort(np.concatenate([Cu[:, [0, 1]], Cu[:, [1, 2]], Cu[:, [2, 3]], Cu[:, [3, 0]]]), axis=1)
    key = e[:, 0].astype(np.int64) * len(Pu) + e[:, 1]
    uq, cnt = np.unique(key, return_counts=True)
    bnd = uq[cnt == 1]
    ba, bb = bnd // len(Pu), bnd % len(Pu)
    mid = 0.5 * (Pu[ba] + Pu[bb])
    segs = []
    for f in P["outlet"][1]:                      # each outlet face lists (v_a, v_b, v_a', v_b'): the z = 0 pair is the first two
        segs.append((V[f[0]][:2], V[f[1]][:2]))
    def dist_to_seg(p, a, b):
        t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
        return np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)
    on = np.zeros(len(mid), bool)
    for a, b in segs:
        on |= dist_to_seg(mid, a, b) < 1e-9
    faces = mid[on]
    order = np.lexsort((faces[:, 0], faces[:, 1]))     # a deterministic order
    faces = faces[order]
    if with_info:
        # per cell: polygon area (m2) and whether it owns a WALL boundary edge (a boundary edge that is not on an inlet/outlet segment)
        inl = [(V[f[0]][:2], V[f[1]][:2]) for f in P["inlet"][1]]
        onio = on.copy()
        for a, b in inl:
            onio |= dist_to_seg(mid, a, b) < 1e-9
        wall_edges = bnd[~onio]
        wa, wb = wall_edges // len(Pu), wall_edges % len(Pu)
        wallflag = np.zeros(len(Cu))
        cell_of = {}
        for k in range(4):
            e2 = np.sort(Cu[:, [k, (k + 1) % 4]], axis=1)
            for ci, (u, v) in enumerate(e2):
                cell_of[(int(u), int(v))] = ci
        for u, v in zip(wa, wb):
            wallflag[cell_of[(int(u), int(v))]] = 1.0
        area = np.abs(0.5 * np.sum(Pu[Cu][:, :, 0] * np.roll(Pu[Cu][:, :, 1], -1, 1) - np.roll(Pu[Cu][:, :, 0], -1, 1) * Pu[Cu][:, :, 1], 1))
        return centres, faces, len(Pu), len(Cu), np.stack([area, wallflag], axis=1)
    return centres, faces, len(Pu), len(Cu)


def topology(dict_path):
    """(Pu, Cu): the replica's 2-D points and the cells' four point indices, counter-clockwise (the orientation build() uses). The cells' connectivity is what the
    plug assessor's FAIL 6 test needs to lay a polyMesh out offline (y_plug_synth.write_polymesh): the faces of the parent section, their owners and neighbours."""
    V, B, E, P = R.parse(str(dict_path))
    Pu, Cu, tag, tol = R.build2d(V, B, E, 2)
    A = 0.5 * np.sum(Pu[Cu][:, :, 0] * np.roll(Pu[Cu][:, :, 1], -1, 1) - np.roll(Pu[Cu][:, :, 0], -1, 1) * Pu[Cu][:, :, 1], 1)
    if (A < 0).all():
        Cu = Cu[:, ::-1]
    return Pu, Cu


def main_tag(tag):
    """The f = 2 symmetric meshes: fixtures/<tag>_cells_2d.npy, <tag>_outlet_faces_2d.npy, <tag>_cellinfo_2d.npy (area m2, wall-row flag), <tag>_cellpts_2d.npy (the four
    point indices of each cell as float64, into constant/yGate/na22_points_ref_2d_<tag>_h2.npy; exact below 2^53)."""
    want = {"YS794": (19728, 72), "YS850": (19416, 72)}[tag]
    centres, faces, npts, ncell, info = build(MAIN_CASE / "constant" / "yGate" / f"na22_blockMeshDict_{tag}_h2", with_info=True)
    out = HERE / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / f"{tag}_cells_2d.npy", centres)
    np.save(out / f"{tag}_outlet_faces_2d.npy", faces)
    np.save(out / f"{tag}_cellinfo_2d.npy", info)
    Pu, Cu = topology(MAIN_CASE / "constant" / "yGate" / f"na22_blockMeshDict_{tag}_h2")
    ref = np.load(MAIN_CASE / "constant" / "yGate" / f"na22_points_ref_2d_{tag}_h2.npy")
    assert Pu.shape == ref.shape and float(np.abs(Pu - ref).max()) == 0.0, "the replica's points are not the shipped points_ref"
    assert (Pu[Cu].mean(axis=1) == centres).all(), "the connectivity does not reproduce the fixture's cell centres"
    np.save(out / f"{tag}_cellpts_2d.npy", Cu.astype(np.float64))
    print(f"{tag}: points {npts} cells {ncell}; {len(faces)} outlet faces (y>0: {(faces[:, 1] > 0).sum()}, y<0: {(faces[:, 1] < 0).sum()}); "
          f"wall-row cells {int(info[:, 1].sum())}; area sum {info[:, 0].sum():.6e} m2")
    assert ncell == want[0] and len(faces) == want[1] and (faces[:, 1] > 0).sum() == want[1] // 2 and (faces[:, 1] < 0).sum() == want[1] // 2


def main():
    if len(sys.argv) > 1:
        for tag in sys.argv[1:]:
            main_tag(tag)
        return
    centres, faces, npts, ncell = build()
    out = HERE / "fixtures"
    out.mkdir(exist_ok=True)
    np.save(out / "y_cells_2d.npy", centres)
    np.save(out / "y_outlet_faces_2d.npy", faces)
    print(f"points {npts} cells {ncell}; fixtures: {centres.shape} cell centres, {faces.shape} outlet faces "
          f"(y>0: {(faces[:, 1] > 0).sum()}, y<0: {(faces[:, 1] < 0).sum()})")
    assert ncell == 78912 and len(faces) == 148 and (faces[:, 1] > 0).sum() == 88 and (faces[:, 1] < 0).sum() == 60


if __name__ == "__main__":
    main()
