#!/usr/bin/env python3
"""Bend a uniform planar block into a smoothly narrowing channel.

Dr Li, 2026-09-01: connect the channel to a narrower one so a clot can block
its entry. Alireza, 2026-09-03: "try not to make sharp edges in the entrance
of the throat because it could make divergence."

THE PROFILE HAS ZERO SLOPE AT BOTH JOINS, so there is no corner anywhere:

    H(x) = H                                        x <= x1
    H(x) = H - (H - Hth)*0.5*(1 - cos(pi*s))        x1 < x < x2,  s = (x-x1)/Lt
    H(x) = Hth                                      x >= x2

dH/dx = -(H - Hth)*(pi/(2*Lt))*sin(pi*s), which is 0 at s = 0 and s = 1. A
piecewise-LINEAR taper would instead put a slope discontinuity at each join --
two salient corners in the one place the measurement is taken -- and that is
what this exists to avoid. Same idea as apply_stenosis.py in
AM2020_stenosis_developed_v2, which displaces points after blockMesh rather
than asking blockMesh for a curved edge.

WHY DISPLACE POINTS RATHER THAN BUILD BLOCKS. A multi-block dict has block
connectivity, shared-face ordering and per-block divisions to get wrong, and
there is no blockMesh on the build machine to catch any of it -- run
20260730-225617 aborted at blockMesh in a paid session and returned no logs.
This changes NO topology: every column keeps its ny cells and every point is
scaled in y alone, so a mesh that was valid before is valid after. What it can
still do is invert a cell if the profile were non-monotonic or the throat were
taller than the channel, and both are checked below.
"""
import math
import re
import sys
from pathlib import Path

HEADER = re.compile(r"(?s)^(.*?\n\(\n)(.*?)(\n\)\n.*)$")


def channel_height(x, h_wide, h_throat, x1, x2):
    if x <= x1:
        return h_wide
    if x >= x2:
        return h_throat
    s = (x - x1) / (x2 - x1)
    return h_wide - (h_wide - h_throat) * 0.5 * (1.0 - math.cos(math.pi * s))


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: apply_throat.py <case> <controls.json>")
    case = Path(sys.argv[1]).resolve()
    import json
    g = json.loads(Path(sys.argv[2]).read_text())
    h_wide = float(g["channel_height_m"])
    h_throat = float(g["throat_height_m"])
    x1 = float(g["taper_start_x_m"])
    x2 = x1 + float(g["taper_length_m"])

    if not 0.0 < h_throat < h_wide:
        raise SystemExit(
            f"throat_height_m {h_throat} must lie strictly between 0 and the "
            f"channel height {h_wide}"
        )
    if x2 <= x1:
        raise SystemExit("taper_length_m must be positive")

    path = case / "constant" / "polyMesh" / "points"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run blockMesh first")
    m = HEADER.match(path.read_text())
    if not m:
        raise SystemExit("unrecognised points file layout")
    head, body, tail = m.groups()
    rows = body.split("\n")

    out = []
    ymax_seen = 0.0
    for row in rows:
        s = row.strip()
        if not (s.startswith("(") and s.endswith(")")):
            raise SystemExit(f"unrecognised point row: {row!r}")
        x, y, z = (float(v) for v in s[1:-1].split())
        ymax_seen = max(ymax_seen, y)
        scale = channel_height(x, h_wide, h_throat, x1, x2) / h_wide
        out.append(f"({x:.12g} {y * scale:.12g} {z:.12g})")

    # The scaling is only a height map if the block really started at height
    # h_wide. If it did not, every derived number -- dp, the resistance split,
    # the throat's dy and therefore dt -- is silently wrong.
    if abs(ymax_seen - h_wide) > 1e-12 * h_wide:
        raise SystemExit(
            f"the mesh's tallest point is {ymax_seen:.6g} m but "
            f"channel_height_m is {h_wide:.6g}; this is not the block this "
            f"profile was derived for"
        )

    path.write_text(head + "\n".join(out) + tail)
    n = 400
    slopes = [
        (channel_height(x1 + (i + 1) * (x2 - x1) / n, h_wide, h_throat, x1, x2)
         - channel_height(x1 + i * (x2 - x1) / n, h_wide, h_throat, x1, x2))
        / ((x2 - x1) / n)
        for i in range(n)
    ]
    print(
        f"apply_throat: {len(rows)} points; height {h_wide*1e6:.1f} -> "
        f"{h_throat*1e6:.1f} um over {(x2-x1)*1e6:.1f} um; "
        f"|dH/dx| max {max(abs(s) for s in slopes):.4f}, "
        f"at the joins {abs(slopes[0]):.2e} and {abs(slopes[-1]):.2e} "
        f"(zero by construction)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
