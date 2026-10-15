#!/usr/bin/env python3
"""The three SHAPE TESTS of the occlusive plug (design research/designs/2026-10-14-case-symmetric-y-plug.md rev. 2.1 section 4, "Tests on the generated 0/ fields"),
run on the c field the REAL initialiser (y_initialise.py) wrote at the replica's cell centres. Pure python.

    width past the tangent   max over x-slices past the fillet tangent of the y extent of cells with theta/phi0 >= 0.5            limit <= 1.045 D0
    content at |y| > 0.5 D0  sum of theta/phi0 x cell area over cells past the tangent with |y| > 0.5 D0                       limit <= 0.045 D0^2
    seal row                 min over wall-row cells of the straight parent, back + R_rear < x < tangent, of 2 theta/phi0 - 1   limit >= 0.99

theta/phi0 = (1 - c)/2 (seed_stage2_content.py: theta0 = theta_ref max(0.5 (1 - c), 0)). A "wall-row" cell owns a wall boundary edge (the fixture's cellinfo flag); the slice height
is the median cell width h = sqrt(median area) (numerics 25's probes/na25_plug.py `width`, `report`). Lengths in D0.

    results = shape_tests(c, cells_xy_m, area_m2, wallflag, plug, anchors)      plug = the resolved plug block of constant/yGate.json
"""
from __future__ import annotations

import math

LIMITS = {"width_D0_max": 1.045, "content_far_D0sq_max": 0.045, "seal_min": 0.99}


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def shape_tests(c, cells_xy, area_m2, wallflag, plug: dict, anchors: dict) -> dict:
    D0 = plug["D0_m"]
    xt, tip = anchors["tangent_x_m"] / D0, anchors["tip_x_m"] / D0
    xs = [x / D0 for x, _y in cells_xy]
    ys = [y / D0 for _x, y in cells_xy]
    A = [a / D0 ** 2 for a in area_m2]
    th = [0.5 * (1.0 - v) for v in c]
    h = math.sqrt(_median(A))
    # width past the tangent
    widths = []
    xc = xt + 0.5 * h
    while xc < tip + 0.05:
        wet = [ys[i] for i in range(len(xs)) if abs(xs[i] - xc) < 0.5 * h and xs[i] > xt and th[i] >= 0.5]
        widths.append((xc, (max(wet) - min(wet)) if len(wet) > 1 else 0.0))
        xc += h
    wmax = max(widths, key=lambda t: t[1])
    # content at |y| > 0.5 past the tangent, and the seed
    beyond = [i for i in range(len(xs)) if xs[i] > xt]
    far = math.fsum(th[i] * A[i] for i in beyond if abs(ys[i]) > 0.5)
    content_past = math.fsum(th[i] * A[i] for i in beyond)
    f = [i for i in beyond if th[i] > 1e-6]
    a0 = math.fsum(th[i] * A[i] * ys[i] for i in f) / math.fsum(th[i] * A[i] for i in f) / 0.5
    # the seal row of the straight parent
    xf = tip - plug["gap_D0"]
    back_plus_r = xf - plug["length_D0"] + plug["rear_radius_D0"]
    seal = [i for i in range(len(xs)) if wallflag[i] > 0.5 and back_plus_r < xs[i] < xt]
    seal_min = min(2.0 * th[i] - 1.0 for i in seal) if seal else float("nan")
    top = sorted((xs[i], 2.0 * th[i] - 1.0) for i in seal if ys[i] > 0)
    run = 0.0
    k = len(top) - 1                                    # the contiguous clot >= 0.99 run ending at the tangent (numerics 25's `straight seal`)
    while k >= 0 and top[k][1] >= 0.99:
        k -= 1
    if top and k + 1 < len(top):
        run = top[-1][0] - top[k + 1][0]
    res = {
        "width_past_tangent_D0": wmax[1], "width_at_x_minus_tangent_D0": wmax[0] - xt, "h_median_D0": h,
        "content_far_D0sq": far, "seal_row_min": seal_min, "seal_row_cells": len(seal), "straight_seal_D0": run,
        "A0": a0, "content_past_tangent_D0sq": content_past, "total_content_D0sq": math.fsum(th[i] * A[i] for i in range(len(xs))),
        "limits": dict(LIMITS),
    }
    res["pass"] = {"width": wmax[1] <= LIMITS["width_D0_max"], "content_far": far <= LIMITS["content_far_D0sq_max"],
                   "seal_row": bool(seal) and seal_min >= LIMITS["seal_min"]}
    res["ok"] = all(res["pass"].values())
    return res
