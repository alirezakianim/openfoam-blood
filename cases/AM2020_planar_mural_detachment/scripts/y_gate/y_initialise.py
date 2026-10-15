#!/usr/bin/env python3
"""The Y gate's initialiser: NO CLOT (the flow gates) or the OCCLUSIVE PLUG (the plug series). Imports the mural gate's file helpers
(initialise_cases.read_vectors / replace_scalar / replace_vector) and writes the same fields its main() writes for a clot -- c, w, Dg, H, U, Us --
with c = +1 (blood) in every cell, or, when constant/yGate.json carries a `plug` block (the generator writes it for a plug child only), with
c = tanh(sd/eps) of the plug of scripts/y_gate/y_plug.py (design 2026-10-14-case-symmetric-y-plug.md section 4): the signed distance is drawn
at the cell centres of 0/C against the fillet tangent and apex tip READ FROM system/blockMeshDict.

WHY NOT initialise_cases.py ITSELF: its main() draws a one-sided clot on the bottom wall of a structured nx x ny channel and REFUSES
a mesh without one ("clot has no solid bulk", "no open lumen" -- fine -- but also the row indexing `centres[:nx]` that assumes a single
i-fastest block). The Y mesh is 15 blocks and carries no clot. The mural gate's script is left untouched (every other child still
runs it), and this one is chosen by Allrun only when the child carries constant/yGate.json.

    y_initialise.py <case dir>
Reads constant/yGate.json and 0/C (postProcess -func writeCellCentres); refuses a cell count that is not the spec's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MURAL_SCRIPTS = HERE.parents[2] / "AM2020_mural_geometry_gate" / "scripts"
sys.path.insert(0, str(MURAL_SCRIPTS))
import initialise_cases as mural_init  # noqa: E402
sys.path.insert(0, str(HERE))
import y_plug  # noqa: E402


def main(argv) -> int:
    case = Path(argv[0]).resolve() if argv else Path.cwd()
    spec = json.loads((case / "constant" / "yGate.json").read_text())
    n = int(spec["mesh"]["counts"]["nCells"])
    centres = mural_init.read_vectors(case / "0" / "C", n)     # refuses a count other than n
    plug = spec.get("plug")
    if plug is None:
        c_values = [1.0] * n
    else:
        anchors = y_plug.dict_anchors((case / "system" / "blockMeshDict").read_text())
        p = {k: plug[k] for k in y_plug.PLUG_KEYS}
        y_plug.plug_geometry(p, plug["D0_m"], plug["interface_width_m"], anchors)       # the keys, the consistency, the anchors against the controls
        c_values = y_plug.c_field(centres, p, plug["D0_m"], plug["interface_width_m"], anchors)
        if not (min(c_values) < -0.99 and max(c_values) > 0.99):
            raise ValueError(f"the plug's c ranges over [{min(c_values):.4f}, {max(c_values):.4f}]: no bulk clot or no open lumen")
    mural_init.replace_scalar(case / "0" / "c", c_values)
    mural_init.replace_scalar(case / "0" / "w", [1.0] * n)
    mural_init.replace_scalar(case / "0" / "Dg", [0.0] * n)
    mural_init.replace_scalar(case / "0" / "H", [0.0] * n)
    mural_init.replace_vector(case / "0" / "U", [(0.0, 0.0, 0.0)] * n)
    mural_init.replace_vector(case / "0" / "Us", [(0.0, 0.0, 0.0)] * n)
    out = case / "postProcessing" / "initialisation"
    out.mkdir(parents=True, exist_ok=True)
    solid = sum(1 for v in c_values if v < 0.0)
    summary = {"child_id": json.loads((case / "child.json").read_text())["child_id"], "cell_count": n, "solid_cells": solid,
               "solid_fraction": solid / n, "c_min": min(c_values), "c_max": max(c_values), "shape": "none" if plug is None else plug["shape"],
               "centre_x_range_m": [min(c[0] for c in centres), max(c[0] for c in centres)],
               "centre_y_range_m": [min(c[1] for c in centres), max(c[1] for c in centres)]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"y_initialise.py: {error}")
