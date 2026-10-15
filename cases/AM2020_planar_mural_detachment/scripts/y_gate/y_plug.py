#!/usr/bin/env python3
"""The OCCLUSIVE PLUG of the symmetric Y: the signed distance that y_initialise.py draws and the generator checks
(design research/designs/2026-10-14-case-symmetric-y-plug.md rev. 2.1 section 4; the construction is numerics 25's, docket probe
probes/na25_plug.py `sd_st`/`sd_c1`, ported term for term).

PURE PYTHON, STANDARD LIBRARY ONLY: the generator, the initialiser on Colab and the dry run import it under plain python3.

THE SHAPE (every length in D0 = the parent width; x runs along the parent axis, the node at the origin, y across):
  * two capsules ("stadia") of axis y = shift (the lateral seed delta, toward +y), horizontal, front tip at xf = tip - gap, length L_c:
        W-stadium  half-width R_rear = Wp/2   (Wp 1.70: it OVERLAPS each parent wall, so the seal row is bulk clot)
        D-stadium  half-width R_front = 0.5   (physics 19's D0-wide stadium)
  * the signed distance is  sd = (1 - w) sd_W + w sd_D,  w = smoothstep(clip((x - xt)/(ell_b), 0, 1)),  xt = the parent fillet tangent
    and ell_b = blend_eps * eps: the Wp stadium in the straight parent, the D0 stadium past the tangent, blended over 2 eps (a hard
    switch leaves a notch: the wall row 0.998 -> 0.408 in one cell; 4 eps fails the width test; numerics 25 section 1).
  * the clot indicator is  c = tanh(sd / eps)  (the mural gate's convention: c = -1 in the clot, +1 in blood), so the content fraction
    theta/phi0 = (1 - c)/2 = (1 - tanh(sd/eps))/2.
THE ANCHORS xt (fillet tangent) and tip (the apex tip, the upstream-most point of the apex arc) are READ from the pinned blockMeshDict, not
typed: xt is the first interior point of the polyLine Pt-T1 (the parent wall, where the straight wall ends and the fillet begins) and tip is
the minimum x of the polyLine N2-N1 (the apex arc). The controls state the same two numbers to 5 digits (numerics 25's printed
`geometry (D0)` line); the generator refuses a dictionary that disagrees.
"""
from __future__ import annotations

import math
import re

FLOAT = r"[-+0-9.eE]+"
PLUG_KEYS = ("shape", "width_parent_D0", "rear_radius_D0", "front_radius_D0", "length_D0", "gap_D0", "shift_D0", "blend_eps",
             "anchors_check_D0")
ANCHOR_TOL_D0 = 1.5e-5        # the controls print the anchors to 5 decimals (half a unit in the last place = 5e-6 D0); 3x that


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", text), flags=re.S)


def dict_anchors(text: str) -> dict:
    """-> {'tangent_x_m', 'tip_x_m', 'parent_half_width_m'} from a blockMeshDict text (convertToMeters 1, vertex labels in the `// i Name` comments)."""
    ctm = float(re.search(r"convertToMeters\s+(" + FLOAT + ")", text).group(1))
    label = {}
    vert = {}
    for m in re.finditer(r"\(\s*(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*\)\s*//\s*(\d+)\s+(\w+)", text):
        if float(m.group(3)) != 0.0:
            continue                                   # the z = depth copies carry the same labels; the z = 0 plane is the mesh plane
        label[m.group(5)] = int(m.group(4))
        vert[int(m.group(4))] = (float(m.group(1)) * ctm, float(m.group(2)) * ctm)
    for need in ("Pt", "T1", "N1", "N2"):
        if need not in label:
            raise ValueError(f"the blockMeshDict has no vertex labelled {need!r} in its `// i Name` comments")
    body = _strip_comments(text)

    def polyline(a: int, b: int):
        m = re.search(rf"polyLine\s+{a}\s+{b}\s*\(", body)
        if not m:
            raise ValueError(f"the blockMeshDict has no `polyLine {a} {b}` edge")
        depth, j = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(body[j], 0)
            j += 1
        pts = [(float(x) * ctm, float(y) * ctm, float(z) * ctm) for x, y, z in
               re.findall(r"\(\s*(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*\)", body[m.end() - 1:j])]
        return [(x, y) for x, y, z in pts if z == 0.0]
    wall = polyline(label["Pt"], label["T1"])
    apex = polyline(label["N2"], label["N1"])
    if len(wall) < 3 or len(apex) < 3:
        raise ValueError("a polyLine of the Pt-T1 / N2-N1 edges has fewer than 3 points on z = 0")
    half = vert[label["Pt"]][1]
    if not (abs(wall[0][1] - half) <= 1e-12 and abs(wall[1][1] - half) > 1e-12):
        raise ValueError(f"the first interior point of Pt-T1 is not the end of the straight parent wall y = {half:.6e} "
                         f"(it must lie on the wall and the second must have left it)")
    return {"tangent_x_m": wall[0][0], "tip_x_m": min(x for x, _y in apex), "parent_half_width_m": half}


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def sd_capsule(x: float, y: float, radius: float, dy: float, xf: float, length: float) -> float:
    """numerics 25 `sd_st`: the signed distance to the capsule whose horizontal axis y = dy runs from x = xf - length + radius to xf - radius,
    so its front tip is at xf and its rear tip at xf - length. Lengths in any one unit."""
    ax, bx = xf - length + radius, xf - radius
    px = min(max(x, ax), bx)
    return math.hypot(x - px, y - dy) - radius


def signed_distance_D0(x: float, y: float, p: dict, xt: float, tip: float, eps: float) -> float:
    """sd (in D0) at (x, y) in D0: p = the plug numbers in D0, xt/tip/eps in D0."""
    xf = tip - p["gap_D0"]
    sw = sd_capsule(x, y, p["rear_radius_D0"], p["shift_D0"], xf, p["length_D0"])
    sdd = sd_capsule(x, y, p["front_radius_D0"], p["shift_D0"], xf, p["length_D0"])
    w = _smoothstep((x - xt) / (p["blend_eps"] * eps))
    return (1.0 - w) * sw + w * sdd


def plug_check(p: dict, name: str = "plug") -> None:
    """The keys and the internal consistency of the plug numbers (the generator and the initialiser both call it)."""
    missing = [k for k in PLUG_KEYS if k not in p]
    extra = [k for k in p if k not in PLUG_KEYS and not k.startswith("_")]
    if missing or extra:
        raise ValueError(f"{name}: keys missing {missing}, unexpected {extra} (the plug reads exactly {list(PLUG_KEYS)})")
    if p["shape"] != "occlusive plug":
        raise ValueError(f"{name}: shape {p['shape']!r} is not 'occlusive plug'")
    for k in PLUG_KEYS[1:-1]:
        v = p[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{name}: {k} {v!r} is not a finite number")
    if not math.isclose(p["rear_radius_D0"], 0.5 * p["width_parent_D0"], rel_tol=1e-12):
        raise ValueError(f"{name}: rear_radius_D0 {p['rear_radius_D0']} != width_parent_D0/2 = {0.5 * p['width_parent_D0']} "
                         f"(the rear cap has the width of the straight-parent plug)")
    if p["width_parent_D0"] < 1.35:
        raise ValueError(f"{name}: width_parent_D0 {p['width_parent_D0']} < 1.35 D0: the plug must overlap each wall "
                         f"(a D0-wide stadium's seal row is only half clot; physics 24 / red team 23 C1)")
    if not (p["front_radius_D0"] > 0 and p["length_D0"] > 2 * p["rear_radius_D0"] and p["gap_D0"] > 0 and p["shift_D0"] > 0
            and p["blend_eps"] > 0):
        raise ValueError(f"{name}: front radius, length (> 2 R_rear), gap, shift and blend_eps must be positive")


def plug_geometry(p: dict, D0: float, eps_m: float, anchors: dict) -> dict:
    """The resolved numbers (D0 units and metres) for the generator's checks and child.json."""
    plug_check(p)
    xt, tip = anchors["tangent_x_m"] / D0, anchors["tip_x_m"] / D0
    for k, got in (("fillet_tangent_x", xt), ("apex_tip_x", tip)):
        want = p["anchors_check_D0"][k]
        if abs(got - want) > ANCHOR_TOL_D0:
            raise ValueError(f"plug: the dictionary's {k} is {got:.6f} D0, the controls state {want}")
    eps = eps_m / D0
    xf = tip - p["gap_D0"]
    return {"xt_D0": xt, "tip_D0": tip, "eps_D0": eps, "front_x_D0": xf, "rear_x_D0": xf - p["length_D0"],
            "seal_straight_arithmetic_D0": xt - (xf - p["length_D0"] + p["rear_radius_D0"]), "blend_length_D0": p["blend_eps"] * eps,
            "blend_end_x_D0": xt + p["blend_eps"] * eps}


def c_field(centres_m, p: dict, D0: float, eps_m: float, anchors: dict):
    """-> [c per cell] = tanh(sd/eps) for cell centres (x, y[, z]) in metres."""
    plug_check(p)
    xt, tip, eps = anchors["tangent_x_m"] / D0, anchors["tip_x_m"] / D0, eps_m / D0
    out = []
    for ctr in centres_m:
        sd = signed_distance_D0(ctr[0] / D0, ctr[1] / D0, p, xt, tip, eps)
        out.append(math.tanh(sd / eps))
    return out
