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


def build():
    V, B, E, P = R.parse(str(MAIN_CASE / "constant" / "yGate" / "na09_blockMeshDict"))
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
    return centres, faces, len(Pu), len(Cu)


def main():
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
