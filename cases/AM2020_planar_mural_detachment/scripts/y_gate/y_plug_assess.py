#!/usr/bin/env python3
"""THE PLUG ASSESSOR of the symmetric Y (design research/designs/2026-10-14-case-symmetric-y-plug.md rev. 2.1 section 6 B; physics 24 C3/C4; numerics 25 section 2;
red team 26 D1-D8). PURE PYTHON, STANDARD LIBRARY ONLY (the Colab assessor and the offline checks run it under plain python3).

TWO LAYERS, so that the verdict logic can be shown on traces and the field readers on fields:
  reduce_case(case, spec, params)   the written fields of ONE session -> one record per full write (X, A, M+, M-, content, max theta/phi0, ...)
  merge_sessions(...)               the records of the restarted sessions of ONE chain, in time order; a gap or a duplicate is REFUSED
  analyse(records, facts, params)   the records -> t_c, V0, tau_A, R, R', the slopes, the verdict, the 6 FAIL items, the reported-not-gated values
  assess_case_dirs(cases, controls) the chain end to end (the CLI scripts/y_plug_chain.py and assess_case.py's y_plug_child call it)

EVERY THRESHOLD IS READ FROM the top-level controls block `y_plug_verdict` (mainCaseControls.json; validated by check_params, which the generator also calls, so a key
nobody reads cannot sit there). Nothing below types a threshold except the geometry of M+- (physics 24's definition) and the unit conversions.

THE OBSERVABLES (units: m and s of the SIMULATION, D0 = 58.333 um, ubar = the parent mean u_mean_sim_m_s):
  theta/phi0 = (thetaC + thetaH)/skeletonVolumeFraction (phi0 0.45); r below.
  X_b     the x-centroid of the SMOOTH-WEIGHT bulk content, w = theta*clip((r - 0.8)/0.2, 0, 1), sum over cells of w*A*x / sum w*A (the threshold-set form is
          refused: sigma 0.04 ubar from 2 writes, numerics 25 section 2). V_b = dX_b/dt / ubar.
  M+, M-  the bulk (r >= 0.9) content r*A/D0^2 past each daughter's NORMAL LINE at the apex tip (physics 24's definition, p24_plug.py L40-41): the half-plane
          (x - T).e1 > 0 for M+ (e1 = the +y daughter's axis), (x - T).e2 > 0 for M-, T = (tip, 0); the y = 0 row (|y| < half the median cell side) counts HALF in each.
  A       the y-centroid of the content past the fillet tangent, in units of 0.5 D0 (numerics 25's A0 = +4.371e-2 / +4.507e-2 at t = 0; reported, not gated).
  t_c     the time, interpolated between writes, at which X_b has moved the plug's front gap (0.10 D0) from its t = 0 value (design D3).
  V0      the least-squares slope of X_b over the writes of the approach window [approach_start, t_c - margin] (the least-squares mean of V_b over it, without the
          differencing noise: the same estimator for a linear approach); tau_A = Lambda*D0/V0 + tau_v, Lambda = p_in/G (physics 24).
  V_b entering R   the 5-write centred least-squares slope of X_b (sigma 0.0003 ubar), linearly interpolated to the requested time (D5).
  dM/dt   the least-squares slope of M+- over the writes in [t_c + tau_A, t_c + 2 tau_A] (>= 16 writes; D4), never a 2-write difference.
THE VERDICT at t_c + 2 tau_A (R = V_b(t_c + 2 tau_A)/V_b(t_c + tau_A)):
  STUCK    R <= stuck_r_max
  SQUEEZE  R >= squeeze_r_min AND V_b(2 tau_A) >= squeeze_vb_min AND for ONE daughter dM/dt >= squeeze_dm_min with the other's slope <= 1/3 of it (D8)
  else INDETERMINATE; re-read at t_c + 3 tau_A with R' = V_b(3 tau_A)/V_b(2 tau_A), the same rule on the slopes of [2 tau_A, 3 tau_A]; INDETERMINATE at both is final (D6).
THE 6 FAIL ITEMS (C5, D1, D2) are evaluated on every write and reported with the write that fired them; they never change the verdict, they say whether it can be trusted.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import y_gate_report as G  # noqa: E402
import y_plug  # noqa: E402

STATUSES = ("VERDICT", "NOT_READABLE_YET", "NO_CONTACT", "NOT_ASSESSABLE", "NO_WRITES")
FAIL_ITEMS = ("F1_weld", "F2_overspeed_or_flux", "F3_overfill_or_drift", "F4_damage_or_stretch", "F5_split", "F6_bypass")
REQUIRED_FIELDS = ("thetaC", "thetaH", "Dg", "Us", "be", "phi", "U", "p")      # a full write the assessor can reduce (restart_driver.classify holds the rest)
PLUG_SERIES = ("s25_ys794_plug", "s25_ys850_plug")

# the keys of the controls block `y_plug_verdict` and their types (number = int or float, not bool). The CHECKED RANGES live in check_params.
PARAM_NUMBERS = (
    "approach_start_s", "approach_end_margin_s", "approach_min_writes", "weight_lo", "weight_hi", "bulk_theta_over_phi0", "fit_writes", "slope_min_writes",
    "stuck_r_max", "squeeze_r_min", "squeeze_vb_min_ubar", "squeeze_dm_min_d0sq_per_ms", "squeeze_other_slope_fraction_max",
    "film_mesh_v0_min_ubar", "film_sqrt_k_v0_max_ubar",
    "fail_weld_vb_ubar", "fail_overspeed_vb_ubar", "fail_content_drift_rel", "fail_dg_max", "fail_bulk_lambda1_max", "fail_split_each_min_d0sq",
    "fail_bypass_max", "fail_bypass_vb_below_ubar", "bypass_slab_width_d0",
    "a_symmetry_flag", "tail_rt_min", "tail_q_outside_min", "tail_q_body_max", "spacing_min_fraction", "spacing_max_fraction")
SERIES_KEYS = ("predicted", "lambda_over_crit", "gate5_fv_bias", "gate5_tol", "flow_gate_series")


class ChainError(Exception):
    """A chain of restarted sessions that cannot be merged: a gap, a duplicate, a session that is not a restart of the previous one."""


# ------------------------------------------------------------------------------------------------------------- the parameters
def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_params(p: dict) -> None:
    """The controls block `y_plug_verdict`: exactly its keys, numbers where numbers are meant, the orderings the table needs. The generator calls this too."""
    if not isinstance(p, dict):
        raise ValueError("y_plug_verdict must be an object")
    allowed = set(PARAM_NUMBERS) | {"theta_over_phi0_band", "series"}
    extra = [k for k in p if k not in allowed and not k.startswith("_")]
    missing = [k for k in sorted(allowed) if k not in p]
    if extra or missing:
        raise ValueError(f"y_plug_verdict: keys {extra} are read by nobody (silently inert); keys {missing} are missing")
    for k in PARAM_NUMBERS:
        if not _num(p[k]):
            raise ValueError(f"y_plug_verdict.{k} {p[k]!r} is not a finite number")
    band = p["theta_over_phi0_band"]
    if not (isinstance(band, list) and len(band) == 2 and all(_num(v) for v in band) and 1.0 <= band[0] < band[1]):
        raise ValueError(f"y_plug_verdict.theta_over_phi0_band {band!r} must be [lo, hi] with 1 <= lo < hi")
    pos = [k for k in PARAM_NUMBERS if k not in ("weight_lo", "weight_hi")]
    for k in pos:
        if not p[k] > 0:
            raise ValueError(f"y_plug_verdict.{k} {p[k]!r} must be > 0")
    if not (0.0 <= p["weight_lo"] < p["weight_hi"] <= 1.0):
        raise ValueError("y_plug_verdict: 0 <= weight_lo < weight_hi <= 1 (the smooth weight clip((r - lo)/(hi - lo), 0, 1))")
    if not (p["weight_lo"] < p["bulk_theta_over_phi0"] <= 1.0):
        raise ValueError("y_plug_verdict.bulk_theta_over_phi0 must lie in (weight_lo, 1]")
    if not (p["fit_writes"] >= 3 and int(p["fit_writes"]) == p["fit_writes"] and int(p["fit_writes"]) % 2 == 1):
        raise ValueError("y_plug_verdict.fit_writes must be an odd integer >= 3 (a centred least-squares fit)")
    for k in ("approach_min_writes", "slope_min_writes"):
        if int(p[k]) != p[k] or p[k] < 3:
            raise ValueError(f"y_plug_verdict.{k} must be an integer >= 3")
    if not p["stuck_r_max"] < p["squeeze_r_min"]:
        raise ValueError("y_plug_verdict: stuck_r_max < squeeze_r_min (the INDETERMINATE band lies between them)")
    if not 0.0 < p["squeeze_other_slope_fraction_max"] < 1.0:
        raise ValueError("y_plug_verdict.squeeze_other_slope_fraction_max must lie in (0, 1)")
    if not (p["film_sqrt_k_v0_max_ubar"] < p["film_mesh_v0_min_ubar"]):
        raise ValueError("y_plug_verdict: film_sqrt_k_v0_max_ubar < film_mesh_v0_min_ubar")
    if not (0.0 < p["tail_q_body_max"] < p["tail_q_outside_min"] < 1.0):
        raise ValueError("y_plug_verdict: 0 < tail_q_body_max < tail_q_outside_min < 1 (the clot body, the middle band and the cells outside the body are disjoint populations of q)")
    if not (0.0 < p["spacing_min_fraction"] < 1.0 < p["spacing_max_fraction"]):
        raise ValueError("y_plug_verdict: 0 < spacing_min_fraction < 1 < spacing_max_fraction (a write interval is 1)")
    if not p["fail_weld_vb_ubar"] < p["fail_bypass_vb_below_ubar"] < p["fail_overspeed_vb_ubar"]:
        raise ValueError("y_plug_verdict: fail_weld_vb_ubar < fail_bypass_vb_below_ubar < fail_overspeed_vb_ubar")
    s = p["series"]
    if not (isinstance(s, dict) and sorted(s) == sorted(PLUG_SERIES)):
        raise ValueError(f"y_plug_verdict.series must hold exactly {list(PLUG_SERIES)}")
    for name, e in s.items():
        bad = [k for k in e if k not in SERIES_KEYS and not k.startswith("_")]
        miss = [k for k in SERIES_KEYS if k not in e]
        if bad or miss:
            raise ValueError(f"y_plug_verdict.series.{name}: keys {bad} are read by nobody; keys {miss} are missing")
        if e["predicted"] not in ("STUCK", "SQUEEZE"):
            raise ValueError(f"y_plug_verdict.series.{name}.predicted {e['predicted']!r} must be STUCK or SQUEEZE")
        for k in ("lambda_over_crit", "gate5_fv_bias", "gate5_tol"):
            if not _num(e[k]) or not e[k] > 0:
                raise ValueError(f"y_plug_verdict.series.{name}.{k} {e[k]!r} must be a positive number")
        if not isinstance(e["flow_gate_series"], str):
            raise ValueError(f"y_plug_verdict.series.{name}.flow_gate_series must be a series name")


def load_params(controls: dict) -> dict:
    p = controls.get("y_plug_verdict")
    if p is None:
        raise ValueError("the controls hold no y_plug_verdict block")
    check_params(p)
    return p


# ------------------------------------------------------------------------------------------------------------- small numerics
def ls_slope(ts, ys):
    """Least-squares slope of ys on ts (None when the abscissae do not spread)."""
    n = len(ts)
    if n < 2:
        return None
    tm, ym = math.fsum(ts) / n, math.fsum(ys) / n
    sxx = math.fsum((t - tm) ** 2 for t in ts)
    if not sxx > 0.0:
        return None
    return math.fsum((t - tm) * (y - ym) for t, y in zip(ts, ys)) / sxx


def interp(ts, vs, t):
    """Linear interpolation of (ts ascending, vs) at t; None outside [ts[0], ts[-1]]."""
    if not ts or t < ts[0] or t > ts[-1]:
        return None
    lo, hi = 0, len(ts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid
        else:
            hi = mid
    if hi == lo:
        return vs[lo]
    f = (t - ts[lo]) / (ts[hi] - ts[lo])
    return vs[lo] + f * (vs[hi] - vs[lo])


def centred_slopes(ts, xs, width):
    """[(t_i, LS slope of xs over the `width` writes centred on i)] for every i with width//2 writes on each side."""
    h = width // 2
    return [(ts[i], ls_slope(ts[i - h:i + h + 1], xs[i - h:i + h + 1])) for i in range(h, len(ts) - h)]


# ------------------------------------------------------------------------------------------------------------- the geometry of one case
class Geo:
    """Everything about the mesh and the child that the reduction needs, read ONCE from the case (0/C, 0/Vc, system/blockMeshDict, constant/phaseClotProperties, 0/p)
    and its child.json."""

    def __init__(self, case: Path, spec: dict, params: dict):
        case = Path(case)
        yg = spec["y_gate"]
        geo = yg["geometry_sim"]
        self.D0, self.depth = float(geo["D0_m"]), float(geo["depth_m"])
        self.ubar = float(yg["drive"]["u_mean_sim_m_s"])
        self.n = int(yg["mesh"]["counts"]["nCells"])
        centres = G.read_internal(case / "0" / "C", 3, self.n)
        vols = G.read_internal(case / "0" / "Vc", 1, self.n)
        self.x = [c[0] for c in centres]
        self.y = [c[1] for c in centres]
        self.area = [v / self.depth for v in vols]
        anchors = y_plug.dict_anchors((case / "system" / "blockMeshDict").read_text())
        self.xt, self.tip = anchors["tangent_x_m"], anchors["tip_x_m"]
        ax = geo["axes_deg"]
        a1, a2 = math.radians(float(ax["d1"])), math.radians(float(ax["d2"]))
        e1, e2 = (math.cos(a1), math.sin(a1)), (math.cos(a2), math.sin(a2))
        side = sorted(math.sqrt(a) for a in self.area)
        self.hmed = side[len(side) // 2] if len(side) % 2 else 0.5 * (side[len(side) // 2 - 1] + side[len(side) // 2])
        half = 0.5 * self.hmed
        self.plus, self.minus = [], []
        for i in range(self.n):
            dx, dy = self.x[i] - self.tip, self.y[i]
            wy = 0.5 if abs(dy) < half else 1.0
            if dx * e1[0] + dy * e1[1] > 0.0 and dy > -half:
                self.plus.append((i, wy))
            if dx * e2[0] + dy * e2[1] > 0.0 and dy < half:
                self.minus.append((i, wy))
        self.beyond = [i for i in range(self.n) if self.x[i] > self.xt]
        pcp = (case / "constant" / "phaseClotProperties").read_text()
        self.phi0 = _dict_number(pcp, "skeletonVolumeFraction")
        self.phi_s_max = _dict_number(pcp, "phiSMax")
        self.tail_law, self.tail_law_why = tail_law_of(pcp)
        self.G = float(spec["G_pa"])
        self.tau_v = float(spec["s19"]["networkViscosityTau"])
        rho = float(spec["plasma"]["rho_kg_m3"])
        kind, pin = G.read_patch_value(case / "0" / "p", "inlet", 1)
        if kind != "uniform":
            raise ValueError(f"{case / '0' / 'p'}: the inlet pressure is not uniform")
        self.p_in = pin * rho                                   # Pa in the simulation's units
        self.lam = self.p_in / self.G                            # Lambda (physics 24): has no plug term
        clot = spec["clot"]
        self.gap_m = float(clot["gap_D0"]) * self.D0
        self.seal_straight_m = float(clot["resolved_D0"]["seal_straight_arithmetic_D0"]) * self.D0
        self.slab_x0 = self.xt - 0.5 * self.seal_straight_m      # the centre of the straight seal at t = 0
        self.write_interval = float(spec["write_interval_s"])
        self.series = spec["series"]
        self.child_id = spec["child_id"]
        series_facts = params["series"][self.series]
        self.q0 = self.ubar * (1.0 + series_facts["gate5_fv_bias"]) * self.D0 * self.depth      # the clot-free flux at this drive (m3/s)
        self.q0_tol = series_facts["gate5_tol"]
        self.slab_w = params["bypass_slab_width_d0"] * self.D0


def _dict_number(text: str, key: str) -> float:
    m = re.search(rf"(?m)^\s*{key}\s+([-+0-9.eE]+)\s*;", re.sub(r"//[^\n]*", "", text))
    if not m:
        raise ValueError(f"constant/phaseClotProperties has no {key} entry")
    return float(m.group(1))


# ------------------------------------------------------------------------------------------------------------- the tangent ratio R_t of one write (numerics 18)
class TailLaw:
    """The Valanis-Landel trilinear skeleton law's tangent stiffness, a line-by-line port of user-applications/phaseClotAxisymmetricFluid/skeletonLawKernel.H (makeLaw L61-L83,
    segOf, tauHatPrime, stretches, tangentStiffness L234-L251): stiff = [(4/9) t1 + (1/9)(t2 + t3)]/J, t_i = l_i tauHat'(l_i), l_i = sqrt(ev_i det^(-1/3)) ascending."""

    def __init__(self, e1_pa: float, e2_pa: float, d1: float, d2: float):
        e1 = 2.0
        e2 = 2.0 * e2_pa / e1_pa
        pq = (e2 - e1) / (2.0 * (d2 - d1))
        qq = e1 - 2.0 * d1 * pq
        self.hi = (1.0 - d2, 1.0 - d1, 1.0 + d1, 1.0 + d2)
        self.b = (e2, 2.0 * pq + qq, e1, qq - 2.0 * pq, e2)
        self.c = (0.0, -pq, 0.0, pq, 0.0)

    def prime(self, l: float) -> float:
        k = 4
        for j in range(4):
            if l < self.hi[j]:
                k = j
                break
        return self.b[k] + 2.0 * self.c[k] * l

    def stiff(self, ev, det: float):
        """None when det or an eigenvalue is not finite and > 0 (the solver refuses there)."""
        if not (math.isfinite(det) and det > 0.0):
            return None
        scale = det ** (-1.0 / 3.0)
        v = sorted(ev)
        l = []
        for e in v:
            s_ = e * scale
            if not (math.isfinite(s_) and s_ > 0.0):
                return None
            l.append(math.sqrt(s_))
        t1, t2, t3 = l[2] * self.prime(l[2]), l[1] * self.prime(l[1]), l[0] * self.prime(l[0])
        return ((4.0 / 9.0) * t1 + (1.0 / 9.0) * (t2 + t3)) / math.sqrt(det)


def tail_law_of(pcp_text: str):
    """(TailLaw, None) from constant/phaseClotProperties when skeletonLaw is valanisLandelTrilinear, else (None, the reason)."""
    text = re.sub(r"//[^\n]*", "", pcp_text)
    m = re.search(r"(?m)^\s*skeletonLaw\s+(\w+)\s*;", text)
    if not m:
        return None, "constant/phaseClotProperties has no skeletonLaw entry"
    if m.group(1) != "valanisLandelTrilinear":
        return None, f"skeletonLaw {m.group(1)} is not valanisLandelTrilinear (the port covers the LT tangent only)"
    blk = re.search(r"skeletonLawCoeffs\s*\{([^}]*)\}", text)
    if not blk:
        return None, "constant/phaseClotProperties has no skeletonLawCoeffs block"
    vals = {}
    for key in ("E1", "E2", "D1", "D2"):
        mm = re.search(rf"(?m)^\s*{key}\s+([-+0-9.eE]+)\s*;", blk.group(1))
        if not mm:
            return None, f"skeletonLawCoeffs has no {key}"
        vals[key] = float(mm.group(1))
    return TailLaw(vals["E1"], vals["E2"], vals["D1"], vals["D2"]), None


def tail_record(law, be, gfield, q, x, y, params: dict) -> dict:
    """numerics 18's tangent ratio on the FIELDS of one full write: R_c = G_c (3/4) stiff(be_c) / (max G (3/4) stiff(I)), the solver's gTangentMax(step)/gTangentMax(step 1)
    (the solver's R = (dtWave(1)/dtWave)^2) cell by cell. Populations by the written q: OUTSIDE the clot body q > tail_q_outside_min, the BODY q < tail_q_body_max, the middle band
    between them. -> the maximum R of each, the argmax cell of all and of the outside population (q, x, y), and the class:
      BELOW         R_all < tail_rt_min
      TAIL          R_all >= tail_rt_min, the all-cell argmax lies OUTSIDE the body (q > tail_q_outside_min) and R_outside > R_body
      CLOT_BODY     R_all >= tail_rt_min and the argmax lies in the body (q < tail_q_body_max)
      INTERMEDIATE  R_all >= tail_rt_min and the argmax lies in the middle band (numerics 18's wall rows that have not yet drained), or lies outside the body with R_body >= R_outside
    Returns {'status': 'NOT_MEASURED', 'why': ...} when a cell's be is not SPD or max G is not positive (the solver refuses there; nothing is clamped)."""
    n = len(be)
    gmax = max(gfield)
    if not gmax > 0.0:
        return {"status": "NOT_MEASURED", "why": f"max Gfield {gmax!r} is not positive"}
    s_id = law.stiff((1.0, 1.0, 1.0), 1.0)
    denom = gmax * 0.75 * s_id
    q_out, q_body = params["tail_q_outside_min"], params["tail_q_body_max"]
    best = {"all": (-1.0, -1), "out": (-1.0, -1), "body": (-1.0, -1), "mid": (-1.0, -1)}
    counts = {"out": 0, "body": 0, "mid": 0}
    for i in range(n):
        xx, xy, xz, yy, yz, zz = be[i]
        if xx == 1.0 and yy == 1.0 and zz == 1.0 and xy == 0.0 and xz == 0.0 and yz == 0.0:
            st = s_id
        else:
            det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
            st = law.stiff(G.eig_symm3(xx, xy, xz, yy, yz, zz), det)
            if st is None:
                return {"status": "NOT_MEASURED", "why": f"be of cell {i} (det {det!r}) is not SPD, so the LT tangent has no real stretches (the solver refuses there)"}
        r = gfield[i] * 0.75 * st / denom
        pop = "out" if q[i] > q_out else ("body" if q[i] < q_body else "mid")
        counts[pop] += 1
        if r > best["all"][0]:
            best["all"] = (r, i)
        if r > best[pop][0]:
            best[pop] = (r, i)
    r_all, i_all = best["all"]
    thr = params["tail_rt_min"]
    q_arg = q[i_all]
    r_out, r_body = best["out"][0], best["body"][0]
    if r_all < thr:
        cls = "BELOW"
    elif q_arg > q_out and r_out > r_body:
        cls = "TAIL"
    elif q_arg < q_body:
        cls = "CLOT_BODY"
    else:
        cls = "INTERMEDIATE"
    i_out = best["out"][1]
    return {"status": "MEASURED", "class": cls, "R_all": r_all, "R_outside": r_out if counts["out"] else None, "R_body": r_body if counts["body"] else None,
            "R_mid": best["mid"][0] if counts["mid"] else None, "q_argmax": q_arg, "x_argmax": x[i_all], "y_argmax": y[i_all], "cell_argmax": i_all,
            "q_argmax_outside": q[i_out] if i_out >= 0 else None, "x_argmax_outside": x[i_out] if i_out >= 0 else None, "y_argmax_outside": y[i_out] if i_out >= 0 else None,
            "n_outside": counts["out"], "n_body": counts["body"], "n_mid": counts["mid"], "gmax": gmax}


# ------------------------------------------------------------------------------------------------------------- one write -> one record
def reduce_write(geo: Geo, d: Path, params: dict, x0=None, inlet=None, t=None) -> dict:
    """The record of one time directory (a full write, or 0/). x0 = X_b at t = 0 (for the tracked bypass slab); inlet = {time: inletFlux row} (or None)."""
    n = geo.n
    th_c = G.read_internal(d / "thetaC", 1, n)
    th_h = G.read_internal(d / "thetaH", 1, n)
    theta = [a + b for a, b in zip(th_c, th_h)]
    phi0, area = geo.phi0, geo.area
    lo, hi = params["weight_lo"], params["weight_hi"]
    bulk_min = params["bulk_theta_over_phi0"]
    sw = sx = sy = 0.0
    a_n = a_d = ab_n = ab_d = 0.0
    content = 0.0
    r_max = 0.0
    n_over = 0
    bulk = []
    r = [v / phi0 for v in theta]
    for i in range(n):
        ri = r[i]
        content += theta[i] * area[i]
        if ri > r_max:
            r_max = ri
        if theta[i] >= geo.phi_s_max:
            n_over += 1
        wgt = theta[i] * min(max((ri - lo) / (hi - lo), 0.0), 1.0)
        if wgt > 0.0:
            wa = wgt * area[i]
            sw += wa
            sx += wa * geo.x[i]
            sy += wa * geo.y[i]
        if ri >= bulk_min:
            bulk.append(i)
    for i in geo.beyond:
        if r[i] > 1e-6:
            wa = r[i] * area[i]
            a_d += wa
            a_n += wa * geo.y[i]
    bulk_set = set(bulk)
    for i in geo.beyond:
        if i in bulk_set:
            wa = r[i] * area[i]
            ab_d += wa
            ab_n += wa * geo.y[i]
    d0sq = geo.D0 * geo.D0
    mp = math.fsum(wy * r[i] * area[i] for i, wy in geo.plus if r[i] >= bulk_min) / d0sq
    mm = math.fsum(wy * r[i] * area[i] for i, wy in geo.minus if r[i] >= bulk_min) / d0sq
    dg = G.read_internal(d / "Dg", 1, n)
    rec = {
        "t": float(t if t is not None else d.name),
        "dir": d.name,
        "X": (sx / sw) if sw > 0.0 else None,
        "Yb": (sy / sw) if sw > 0.0 else None,
        "A": (a_n / a_d / (0.5 * geo.D0)) if a_d > 0.0 else None,
        "A_bulk": (ab_n / ab_d / (0.5 * geo.D0)) if ab_d > 0.0 else None,
        "Mp": mp, "Mm": mm,
        "content": content * geo.depth,
        "r_max": r_max, "n_over": n_over,
        "Dg_max": max(dg), "lam1_bulk_max": None, "Q": None, "Qs": None, "slab_ok": None,
    }
    be_path = d / "be"
    be = None
    if be_path.is_file():
        be = G.read_internal(be_path, 6, n)
        rec["lam1_bulk_max"] = G.lambda1_max([be[i] for i in bulk]) if bulk else 1.0
    rec["tail"] = _tail_of_write(geo, d, be, params, initial=(t == 0.0 and d.name == "0"))
    if inlet is not None:
        q = G._flux_at(inlet, rec["t"])
        rec["Q"] = (-q) if q is not None else None
    if x0 is not None and rec["X"] is not None and (d / "Us").is_file():
        xs = geo.slab_x0 + (rec["X"] - x0)
        rec["slab_ok"] = (xs + 0.5 * geo.slab_w) <= geo.xt                          # the tracked section lies inside the straight parent (the seal moves with the bulk)
        if rec["slab_ok"]:
            us = G.read_internal(d / "Us", 3, n)
            tot = math.fsum(us[i][0] * area[i] for i in range(n) if abs(geo.x[i] - xs) <= 0.5 * geo.slab_w)
            rec["Qs"] = tot / geo.slab_w * geo.depth
    return rec


def _tail_of_write(geo: Geo, d: Path, be, params: dict, initial: bool) -> dict:
    """The record's `tail` entry: tail_record on the fields of this write, or NOT_MEASURED with the reason (the initial state holds no be, q or Gfield)."""
    if initial:
        return {"status": "NOT_MEASURED", "why": "the initial state is not a solver write (no be, q, Gfield)"}
    if geo.tail_law is None:
        return {"status": "NOT_MEASURED", "why": geo.tail_law_why}
    missing = [f for f in ("be", "q", "Gfield") if not (d / f).is_file()]
    if missing:
        return {"status": "NOT_MEASURED", "why": f"the write holds no {', '.join(missing)}"}
    n = geo.n
    return tail_record(geo.tail_law, be, G.read_internal(d / "Gfield", 1, n), G.read_internal(d / "q", 1, n), geo.x, geo.y, params)


def read_restart_json(case: Path):
    p = Path(case) / "restart.json"
    return json.loads(p.read_text()) if p.is_file() else None


def reduce_case(case: Path, spec: dict, params: dict, x0_hint=None) -> dict:
    """-> {'records': [...ascending in t, t = 0 first when 0/ holds the fields], 'skipped': [{name, why}], 'restart': restart.json or None, 'geo': Geo-facts}.
    A time directory enters when restart_driver.classify says FULL and INTACT and it holds REQUIRED_FIELDS; any other numeric directory is listed under 'skipped'."""
    import restart_driver as RD
    case = Path(case)
    dirs = RD.time_dirs(case)
    full, skipped = [], []
    for d in dirs:
        kind, why = RD.classify(d)
        if kind != "full":
            skipped.append({"name": d.name, "why": f"{kind}: {why}"})
            continue
        missing = [f for f in REQUIRED_FIELDS if not (d / f).is_file()]
        if missing:
            skipped.append({"name": d.name, "why": f"full write without {', '.join(missing)}"})
            continue
        full.append(d)
    out = {"records": [], "skipped": skipped, "restart": read_restart_json(case), "geo": None}
    if not full:
        return out
    geo = Geo(case, spec, params)
    out["geo"] = {"D0": geo.D0, "ubar": geo.ubar, "lam": geo.lam, "tau_v": geo.tau_v, "gap_m": geo.gap_m, "q0": geo.q0, "q0_tol": geo.q0_tol,
                  "write_interval": geo.write_interval, "series": geo.series, "child_id": geo.child_id, "phi0": geo.phi0, "phi_s_max": geo.phi_s_max,
                  "xt": geo.xt, "tip": geo.tip, "slab_x0": geo.slab_x0, "slab_w": geo.slab_w, "p_in": geo.p_in, "G": geo.G, "depth": geo.depth}
    try:
        inlet = G.read_flux_series(case, "inletFlux")
    except (OSError, ValueError):
        inlet = None
    x0 = x0_hint
    zero = case / "0"
    if (zero / "thetaC").is_file() and (zero / "thetaH").is_file() and (zero / "Dg").is_file():
        rec0 = reduce_write(geo, zero, params, None, None, t=0.0)
        rec0["initial"] = True
        x0 = rec0["X"]
        out["records"].append(rec0)
    for d in full:
        out["records"].append(reduce_write(geo, d, params, x0, inlet))
    out["x0"] = x0
    return out


# ------------------------------------------------------------------------------------------------------------- merging the sessions of a chain
def merge_sessions(sessions: list, interval: float, params: dict) -> list:
    """sessions = [{'records': [...], 'restart': restart.json or None, 'label': str}, ...] in chain order -> ONE record list in time order.
    REFUSED (ChainError): a session without the initial state first / a later session that is not a restart; a restart point that is not the previous session's last write;
    a write that appears twice; a gap larger than spacing_max_fraction write intervals; two writes closer than spacing_min_fraction intervals, unless one of them is the LAST write
    of a session or a restart point (a session ends by a writeNow at an arbitrary time): that one is dropped and its neighbour kept."""
    lo_f, hi_f = params["spacing_min_fraction"], params["spacing_max_fraction"]
    merged: list = []
    for k, s in enumerate(sessions):
        recs = [dict(r) for r in s["records"]]
        label = s.get("label", f"session {k + 1}")
        if not recs:
            raise ChainError(f"{label}: holds no full write")
        recs[-1]["session_end"] = True                                       # a session ends by a writeNow at an arbitrary time: it may sit within a fraction of an interval of the previous write
        if k == 0:
            if not recs[0].get("initial"):
                raise ChainError(f"{label}: the first session of a chain must hold the initial state (0/ with thetaC, thetaH, Dg)")
            merged.extend(recs)
            continue
        rj = s.get("restart")
        if not rj or "chosen" not in rj:
            raise ChainError(f"{label}: holds no restart.json, so it is not a restart of the previous session (a cold start repeats the chain from t = 0)")
        t_r = float(rj["chosen"])
        recs = [r for r in recs if not r.get("initial")]                  # its own 0/ is the same initial state as session 1's
        prev_last = merged[-1]
        if not math.isclose(prev_last["t"], t_r, rel_tol=1e-9, abs_tol=1e-15):
            where = "later than" if prev_last["t"] > t_r else "earlier than"
            raise ChainError(f"{label}: restarts from {t_r!r} but the previous session's last write is {prev_last['t']!r} ({where} the restart point): "
                             f"{'writes appear twice' if prev_last['t'] > t_r else 'writes between them are missing'}")
        first = next((r for r in recs if math.isclose(r["t"], t_r, rel_tol=1e-9, abs_tol=1e-15)), None)
        if first is None:
            raise ChainError(f"{label}: does not hold its own restart point {t_r!r}")
        for key in ("X", "A", "Mp", "Mm", "content", "r_max"):
            if first[key] != prev_last[key]:
                raise ChainError(f"{label}: the restart point {t_r!r} differs from the previous session's write in {key} ({first[key]!r} vs {prev_last[key]!r}): "
                                 "not the same state (another package, or another case)")
        prev_last["restart_point"] = True
        merged.extend(r for r in recs if r is not first)
        if any(r["t"] < t_r for r in recs):
            raise ChainError(f"{label}: holds a write before its own restart point {t_r!r}")
    # spacing: every pair of neighbours
    out = [merged[0]]
    for r in merged[1:]:
        dt = r["t"] - out[-1]["t"]
        if dt <= 0.0 or dt <= 1e-12 * max(abs(r["t"]), 1e-300):
            raise ChainError(f"a write appears twice: t = {r['t']!r} (after {out[-1]['t']!r})")
        if dt > hi_f * interval:
            raise ChainError(f"a gap: no full write between t = {out[-1]['t']!r} and {r['t']!r} (more than {hi_f} write intervals of {interval!r})")
        if dt < lo_f * interval:
            if (out[-1].get("restart_point") or out[-1].get("session_end")) and not out[-1].get("initial"):
                out[-1] = r                                                     # a session end / restart point within a fraction of an interval of the next write: drop it
                continue
            if r.get("restart_point") or r.get("session_end"):
                continue
            raise ChainError(f"two writes {dt!r} s apart (less than {lo_f} of a write interval {interval!r}): t = {out[-1]['t']!r} and {r['t']!r} overlap")
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------------------- the analysis
def _rates(ts, xs, ubar, interval, params):
    """[(t_mid, V_b in ubar)] of the 2-write rates, over neighbours at least spacing_min_fraction apart."""
    out = []
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        if dt >= params["spacing_min_fraction"] * interval:
            out.append((0.5 * (ts[i] + ts[i - 1]), (xs[i] - xs[i - 1]) / dt / ubar))
    return out


def classify(r, v_ubar, sp, sm, n_window, params) -> tuple:
    """One read of the table. -> (verdict, why). sp, sm = dM+/dt, dM-/dt in D0^2/ms (None when the window is too short)."""
    if r is None:
        return "INDETERMINATE", "V_b at the earlier read is not positive, so R is undefined"
    if r <= params["stuck_r_max"]:
        return "STUCK", f"R {r:.4f} <= {params['stuck_r_max']}"
    reasons = []
    if not r >= params["squeeze_r_min"]:
        reasons.append(f"R {r:.4f} lies in the INDETERMINATE band ({params['stuck_r_max']}, {params['squeeze_r_min']})")
    if v_ubar is None or not v_ubar >= params["squeeze_vb_min_ubar"]:
        reasons.append(f"V_b {v_ubar!r} ubar < {params['squeeze_vb_min_ubar']}")
    if n_window < params["slope_min_writes"] or sp is None or sm is None:
        reasons.append(f"the slope window holds {n_window} writes (< {int(params['slope_min_writes'])})")
    else:
        dm, fr = params["squeeze_dm_min_d0sq_per_ms"], params["squeeze_other_slope_fraction_max"]
        one_sided = (sp >= dm and sm <= fr * sp) or (sm >= dm and sp <= fr * sm)
        if not one_sided:
            reasons.append(f"no ONE daughter with dM/dt >= {dm:g} D0^2/ms and the other <= {fr:.4g} of it (dM+/dt {sp:.3e}, dM-/dt {sm:.3e})")
    if not reasons:
        return "SQUEEZE", f"R {r:.4f} >= {params['squeeze_r_min']}, V_b {v_ubar:.4f} ubar, one-sided entry (dM+/dt {sp:.3e}, dM-/dt {sm:.3e} D0^2/ms)"
    return "INDETERMINATE", "; ".join(reasons)


def analyse(records: list, facts: dict, params: dict) -> dict:
    """records (ascending t; t = 0 first) -> the full result. facts: ubar, D0, lam, tau_v, gap_m, write_interval, series, q0, q0_tol."""
    out: dict = {"status": "NO_WRITES", "verdict": None, "final": False, "series": facts.get("series")}
    sf = params["series"].get(facts.get("series"), {})
    out["predicted"] = sf.get("predicted")
    recs = [r for r in records]
    if not recs:
        return out
    for r in recs:
        for k in ("X", "A", "Mp", "Mm", "content", "r_max"):
            v = r.get(k)
            if v is not None and not math.isfinite(v):
                out.update(status="NOT_ASSESSABLE", why=f"a non-finite {k} at t = {r['t']!r}")
                return out
    if recs[0]["t"] != 0.0 or recs[0].get("X") is None:
        out.update(status="NOT_ASSESSABLE", why="the record list does not start from the initial state (t = 0 with bulk content)")
        return out
    ubar, D0, interval = facts["ubar"], facts["D0"], facts["write_interval"]
    with_x = [r for r in recs if r["X"] is not None]
    if len(with_x) != len(recs):
        bad = next(r for r in recs if r["X"] is None)
        out.update(status="NOT_ASSESSABLE", why=f"no bulk content (theta/phi0 >= {params['weight_lo']}) at t = {bad['t']!r}: the plug is gone")
        return out
    ts = [r["t"] for r in recs]
    xs = [r["X"] for r in recs]
    x0 = xs[0]
    out["t_end"] = ts[-1]
    out["n_writes"] = len(recs)
    # ---- t_c (D3): the first interpolated time at which X_b has moved the front gap
    gap = facts["gap_m"]
    t_c = None
    for i in range(1, len(recs)):
        if xs[i] - x0 >= gap:
            a, b = xs[i - 1] - x0, xs[i] - x0
            t_c = ts[i - 1] + (gap - a) / (b - a) * (ts[i] - ts[i - 1]) if b > a else ts[i]
            break
    out["t_c"] = t_c
    rates = _rates(ts, xs, ubar, interval, params)
    t_a0 = params["approach_start_s"]
    t_a1 = (t_c - params["approach_end_margin_s"]) if t_c is not None else ts[-1]
    appr = [(t, v) for t, v in rates if t_a0 <= t <= t_a1]
    out["approach"] = {"from_s": t_a0, "to_s": t_a1, "n_rates": len(appr)}
    # ---- FAIL items
    out["fail"] = fail_items(recs, ts, xs, facts, params, appr, t_c, rates)
    out["fired"] = [k for k, v in out["fail"].items() if v["fired"]]
    # ---- reported, not gated
    out["reported"] = reported_values(recs, facts, params, t_c)
    out["tail"] = tail_summary(recs, params)
    if t_c is None:
        out.update(status="NO_CONTACT", why=f"the bulk centroid moved {max(x - x0 for x in xs) / D0:.4f} D0 of the {gap / D0:.4f} D0 front gap by t = {ts[-1]!r}: the plug never reached the apex")
        return out
    wr = [(t, x) for t, x in zip(ts, xs) if t_a0 <= t <= t_a1]
    if len(wr) < params["approach_min_writes"]:
        out.update(status="NOT_ASSESSABLE", why=f"the approach window [{t_a0}, {t_a1:.6g}] s holds {len(wr)} writes (< {int(params['approach_min_writes'])})")
        return out
    v0 = ls_slope([t for t, _ in wr], [x for _, x in wr])
    out["V0_ubar"] = v0 / ubar if v0 is not None else None
    if v0 is None or not v0 > 0.0:
        out.update(status="NOT_ASSESSABLE", why=f"V0 {v0!r} m/s is not positive: the approach has no speed")
        return out
    tau_a = facts["lam"] * D0 / v0 + facts["tau_v"]
    out["tau_A"] = tau_a
    out["film_branch"] = ("mesh film" if out["V0_ubar"] >= params["film_mesh_v0_min_ubar"] else
                          "sqrt(k) film" if out["V0_ubar"] <= params["film_sqrt_k_v0_max_ubar"] else "between the two branches")
    reads = [(2, "2tau_A"), (3, "3tau_A")]                       # structural (design 6 B): the verdict at t_c + 2 tau_A, the follow-up at t_c + 3 tau_A
    v5 = centred_slopes(ts, xs, int(params["fit_writes"]))
    v5t = [t for t, _ in v5]
    v5v = [v / ubar for _, v in v5]
    out["reads"] = []
    verdict = None
    pending = None
    for idx, (mult, label) in enumerate(reads):
        t_hi = t_c + mult * tau_a
        t_lo = t_c + (mult - 1) * tau_a
        va = interp(v5t, v5v, t_lo)
        vb = interp(v5t, v5v, t_hi)
        entry = {"label": label, "t_prev": t_lo, "t_read": t_hi, "V_prev_ubar": va, "V_ubar": vb}
        if vb is None or va is None:
            entry["why"] = f"the record ends at t = {ts[-1]!r}; the {label} read needs a centred fit at {t_hi!r}"
            out["reads"].append(entry)
            pending = label
            break
        r = (vb / va) if va > 0.0 else None
        win = [(t, r_) for t, r_ in zip(ts, recs) if t_lo <= t <= t_hi]
        nwin = len(win)
        sp = ls_slope([t for t, _ in win], [x["Mp"] for _, x in win]) if nwin >= 3 else None
        sm = ls_slope([t for t, _ in win], [x["Mm"] for _, x in win]) if nwin >= 3 else None
        sp = sp / 1000.0 if sp is not None else None                          # D0^2 per s -> per ms
        sm = sm / 1000.0 if sm is not None else None
        v, why = classify(r, vb, sp, sm, nwin, params)
        entry.update(R=r, n_window=nwin, dMp_dt=sp, dMm_dt=sm, verdict=v, why=why)
        out["reads"].append(entry)
        if v != "INDETERMINATE":
            verdict = v
            out["read"] = label
            break
        if idx == len(reads) - 1:
            out["read"] = label
    if verdict is not None:
        out.update(status="VERDICT", verdict=verdict, final=True)
    elif pending is not None:
        first = out["reads"][0]
        if pending == reads[0][1]:
            out.update(status="NOT_READABLE_YET", why=out["reads"][0]["why"], readable_at_s=t_c + 2 * tau_a + 0.5 * int(params["fit_writes"]) * interval)
        else:
            out.update(status="VERDICT", verdict="INDETERMINATE", final=False, read="2tau_A",
                       why=f"INDETERMINATE at 2 tau_A ({first['why']}); the {pending} follow-up is not yet readable")
    else:
        out.update(status="VERDICT", verdict="INDETERMINATE", final=True, read=reads[-1][1])
    # ---- the readings the table asks for
    rd = out["reads"][-1] if out["reads"] else {}
    if verdict is not None or out["status"] == "VERDICT":
        t_v = rd.get("t_read")
        out["S"] = split_values(recs, t_c, t_v)
        rec_v = _nearest(recs, t_v) if t_v is not None else None
        if rec_v is not None:
            out["A_at_verdict"] = rec_v["A"]
            out["A_symmetry_hold"] = (abs(rec_v["A"]) < params["a_symmetry_flag"]) if rec_v["A"] is not None else None
    out["agrees_with_prediction"] = (out.get("verdict") == out["predicted"]) if out.get("final") and out.get("verdict") != "INDETERMINATE" else None
    out["reading"] = reading(out, params)
    return out


def _nearest(recs, t):
    return min(recs, key=lambda r: abs(r["t"] - t)) if recs else None


def split_values(recs, t_c, t_v):
    """S = (M+ - M-)/(M+ + M-): from the raw M at the verdict write, and from the increments since contact. None when the denominator is not positive."""
    rv, rc = _nearest(recs, t_v), _nearest(recs, t_c)

    def s(a, b):
        return (a - b) / (a + b) if (a + b) > 0.0 else None
    return {"raw": s(rv["Mp"], rv["Mm"]), "since_contact": s(rv["Mp"] - rc["Mp"], rv["Mm"] - rc["Mm"]),
            "Mp": rv["Mp"], "Mm": rv["Mm"], "Mp_at_contact": rc["Mp"], "Mm_at_contact": rc["Mm"], "t_verdict": rv["t"]}


def reading(out: dict, params: dict) -> str:
    """How to read the record (C10): SQUEEZE is ENTRY ONSET; a YS850 STUCK or a YS794 SQUEEZE falsifies physics 19's STATIC energy criterion only if the verdict is
    decisive, the bypass is <= 0.01 and |A| > 1e-3."""
    v = out.get("verdict")
    if v is None or out.get("status") != "VERDICT":
        return "no verdict"
    if v == "INDETERMINATE":
        return "INDETERMINATE: no statement about physics 19's static criterion"
    pred = out.get("predicted")
    if v == pred:
        return f"{v} as predicted" + (" (SQUEEZE = ENTRY ONSET within the horizon; passage is out of scope)" if v == "SQUEEZE" else "")
    f6 = out["fail"]["F6_bypass"]
    bypass_ok = (f6["fired"] is False) and f6.get("evaluated", 0) > 0
    sym_ok = out.get("A_symmetry_hold") is False
    if bypass_ok and sym_ok:
        return f"{v} against the prediction {pred}: FALSIFIES physics 19's STATIC energy criterion (decisive, bypass <= {params['fail_bypass_max']}, |A| > {params['a_symmetry_flag']}); it says nothing about the timings"
    why = []
    if not bypass_ok:
        why.append("the bypass is not shown to be <= %g (%s)" % (params["fail_bypass_max"], "fired" if f6["fired"] else "not evaluable"))
    if not sym_ok:
        why.append(f"|A| {out.get('A_at_verdict')!r} <= {params['a_symmetry_flag']}: a symmetry hold, or not measured")
    return f"{v} against the prediction {pred}, but NOT a falsification of 19: " + "; ".join(why)


# ------------------------------------------------------------------------------------------------------------- the 6 FAIL items
def fail_items(recs, ts, xs, facts, params, appr, t_c, rates) -> dict:
    ubar = facts["ubar"]
    f: dict = {}
    # 1  a weld: V_b < weld during the approach
    vmin = min((v for _, v in appr), default=None)
    t1 = next((t for t, v in appr if v < params["fail_weld_vb_ubar"]), None)
    f["F1_weld"] = {"fired": (t1 is not None) if appr else None, "min_Vb_ubar": vmin, "at_t": t1, "limit": params["fail_weld_vb_ubar"],
                    "rule": "V_b < %g ubar anywhere in the approach window" % params["fail_weld_vb_ubar"]}
    # 2  V_b above the highest prediction in the approach, or Q > Q0
    vmax = max((v for _, v in appr), default=None)
    t2 = next((t for t, v in appr if v > params["fail_overspeed_vb_ubar"]), None)
    q_lim = facts["q0"] * (1.0 + facts["q0_tol"])
    qs = [(r["t"], r["Q"]) for r in recs if r.get("Q") is not None]
    qt = next((t for t, q in qs if q > q_lim), None)
    f["F2_overspeed_or_flux"] = {"fired": (t2 is not None or qt is not None) if (appr or qs) else None, "max_Vb_ubar": vmax, "at_t": t2,
                                 "max_Q_over_Q0": (max(q for _, q in qs) / facts["q0"]) if qs else None, "Q_at_t": qt,
                                 "limit_Vb": params["fail_overspeed_vb_ubar"], "limit_Q_over_Q0": 1.0 + facts["q0_tol"],
                                 "rule": "V_b > %g ubar in the approach, or Q > Q0 (1 + the flow gate's gate-5 tolerance)" % params["fail_overspeed_vb_ubar"]}
    # 3  D1: content drift > 1e-6 relative, or a cell at theta >= phiSMax (overfill against the code's cap); theta/phi0 > 1 is compaction
    c0 = recs[0]["content"]
    drift = max((abs(r["content"] - c0) / c0 for r in recs), default=None) if c0 > 0.0 else None
    t_drift = next((r["t"] for r in recs if c0 > 0.0 and abs(r["content"] - c0) / c0 > params["fail_content_drift_rel"]), None)
    t_over = next((r["t"] for r in recs if r["n_over"] > 0), None)
    f["F3_overfill_or_drift"] = {"fired": (t_drift is not None or t_over is not None) if c0 > 0.0 else None, "max_drift_rel": drift, "drift_at_t": t_drift,
                                 "overfill_at_t": t_over, "max_n_cells_at_cap": max(r["n_over"] for r in recs), "limit_drift": params["fail_content_drift_rel"],
                                 "rule": "total content drift > %g relative, or any cell at theta >= phiSMax (theta/phi0 >= phiSMax/phi0); theta/phi0 above 1 is COMPACTION" % params["fail_content_drift_rel"]}
    # 4  Dg > 1e-3 or the bulk lambda1 > 1.6
    t_dg = next((r["t"] for r in recs if r["Dg_max"] > params["fail_dg_max"]), None)
    l1 = [(r["t"], r["lam1_bulk_max"]) for r in recs if r.get("lam1_bulk_max") is not None]
    t_l1 = next((t for t, v in l1 if v > params["fail_bulk_lambda1_max"]), None)
    f["F4_damage_or_stretch"] = {"fired": (t_dg is not None or t_l1 is not None), "max_Dg": max(r["Dg_max"] for r in recs), "Dg_at_t": t_dg,
                                 "max_bulk_lambda1": max((v for _, v in l1), default=None), "lambda1_at_t": t_l1,
                                 "rule": "Dg > %g, or lambda1 of a BULK cell (theta/phi0 >= bulk) > %g" % (params["fail_dg_max"], params["fail_bulk_lambda1_max"])}
    # 5  the plug splitting at the apex: bulk content in BOTH daughters, M- and M+ each >= 0.05 D0^2
    lim = params["fail_split_each_min_d0sq"]
    t5 = next((r["t"] for r in recs if r["Mp"] >= lim and r["Mm"] >= lim), None)
    f["F5_split"] = {"fired": t5 is not None, "at_t": t5, "max_min_Mpm": max(min(r["Mp"], r["Mm"]) for r in recs), "limit": lim,
                     "rule": "M+ and M- both >= %g D0^2 at the same write" % lim}
    # 6  D2: a bypass: (Q - Q_s)/Q0 > 0.01 once V_b < 0.05 ubar, Q_s the skeleton flux across a section tracked inside the straight seal
    v5 = centred_slopes(ts, xs, int(params["fit_writes"]))
    v5d = {t: v / ubar for t, v in v5}
    n_eval, fired_at, worst = 0, None, None
    for r in recs:
        if r.get("Q") is None or r.get("Qs") is None or not r.get("slab_ok"):
            continue
        vb = v5d.get(r["t"])
        if vb is None:
            i = ts.index(r["t"])
            vb = (xs[i] - xs[i - 1]) / (ts[i] - ts[i - 1]) / ubar if i > 0 else None
        if vb is None or not vb < params["fail_bypass_vb_below_ubar"]:
            continue
        n_eval += 1
        by = (r["Q"] - r["Qs"]) / facts["q0"]
        if worst is None or by > worst:
            worst = by
        if by > params["fail_bypass_max"] and fired_at is None:
            fired_at = r["t"]
    f["F6_bypass"] = {"fired": (fired_at is not None) if n_eval else None, "evaluated": n_eval, "max_bypass": worst, "at_t": fired_at, "limit": params["fail_bypass_max"],
                      "rule": "(Q - Q_s)/Q0 > %g at a write with V_b < %g ubar (Q the inlet flux, Q_s the skeleton flux across a section tracked inside the straight seal; "
                              "the section must lie in the straight parent)" % (params["fail_bypass_max"], params["fail_bypass_vb_below_ubar"])}
    return f


def reported_values(recs, facts, params, t_c) -> dict:
    """The values that are reported, never gated."""
    rmax = max(r["r_max"] for r in recs)
    band = params["theta_over_phi0_band"]
    a = [(r["t"], r["A"]) for r in recs if r.get("A") is not None]
    return {"max_theta_over_phi0": rmax, "theta_over_phi0_band": band,
            "in_band": (band[0] <= rmax <= band[1]),
            "A_t0": recs[0]["A"], "A_end": recs[-1]["A"], "A_bulk_end": recs[-1].get("A_bulk"), "A_series_len": len(a),
            "M_end": {"Mp": recs[-1]["Mp"], "Mm": recs[-1]["Mm"]}}


def tail_summary(recs: list, params: dict) -> dict:
    """The localisation of the OI-81 tail over the full writes (numerics 18; reported, never gated, it decides S26): per write the class of tail_record; here the writes that
    were measured, the maximum R, the FIRST write at or above tail_rt_min with its class and its argmax (q, x, y), the last write, and the decision
    TAIL (some write is TAIL) / CLOT_BODY / INTERMEDIATE / NO_CROSSING (every measured write BELOW) / NOT MEASURED (no write measured, with the reason)."""
    rows = [(r["t"], r["tail"]) for r in recs if r.get("tail") and not r.get("initial")]
    ok = [(t, e) for t, e in rows if e.get("status") == "MEASURED"]
    if not ok:
        whys = sorted({e.get("why", "?") for _, e in rows}) or ["no full write"]
        return {"status": "NOT MEASURED", "why": "; ".join(whys), "writes": len(rows), "measured": 0}
    pick = lambda t, e: {"t": t, **{k: e[k] for k in ("class", "R_all", "R_outside", "R_body", "R_mid", "q_argmax", "x_argmax", "y_argmax", "q_argmax_outside")}}
    top = max(ok, key=lambda te: te[1]["R_all"])
    first = next(((t, e) for t, e in ok if e["R_all"] >= params["tail_rt_min"]), None)
    classes: dict = {}
    for _, e in ok:
        classes[e["class"]] = classes.get(e["class"], 0) + 1
    if "TAIL" in classes:
        decision = "TAIL"
    elif "CLOT_BODY" in classes:
        decision = "CLOT_BODY"
    elif "INTERMEDIATE" in classes:
        decision = "INTERMEDIATE"
    else:
        decision = "NO_CROSSING"
    return {"status": "MEASURED" if len(ok) == len(rows) else "PARTIAL", "decision": decision, "threshold": params["tail_rt_min"],
            "q_outside_min": params["tail_q_outside_min"], "q_body_max": params["tail_q_body_max"], "writes": len(rows), "measured": len(ok), "classes": classes,
            "max": pick(*top), "first_crossing": pick(*first) if first else None, "last": pick(*ok[-1]),
            "not_measured_why": sorted({e.get("why", "?") for _, e in rows if e.get("status") != "MEASURED"})}


# ------------------------------------------------------------------------------------------------------------- the log-side reported values
NOT_MEASURED_ROWS = {
    "odd_even": "the clot-phase odd-even masked amplitude needs a structured nx x ny grid (the existing instrument's masked stencil); the Y mesh is one unstructured block with no (i, j) "
                "indexing, so the instrument does not apply (CE ruling 2026-10-15, a stated limit)",
    "F2_a_below_1": "S25's F2 runtime clause 'a < 1 in every cell at every step' (research/designs/2026-10-10-S25-arteriole-bundle.md, F2 runtime part) has no logged key: the S25.1 Y log "
                    "(B79: 1843 PHASE_CLOT_S25_F3 lines) holds no aMax token, so it is not read here (CE ruling 2026-10-15, a stated limit)",
}


def log_reports(texts: list, t_c, params: dict, s25_report=None) -> dict:
    """R_t, the dt factor, F3 and the NOT MEASURED rows from the sessions' logs (reported, not gated). R_t = (dtWave(1)/dtWave)^2 at its maximum over every logged step; its
    localisation (the argmax q > 0.99 outside the body, numerics 18) is read from the FIELDS: analyse() `tail`. F3 = assess_case.s25_report's `f3` over the PHASE_CLOT_S25_F3 lines of every
    session (s25_report is that function, handed in so that this module stays standalone); NOT MEASURED, with the reason, when no reader is given or the log holds no such line."""
    out: dict = {"not_measured": dict(NOT_MEASURED_ROWS)}
    out["F3"] = f3_of_logs(texts, s25_report)
    recs = []
    for text in texts:
        for m in re.finditer(r"^PHASE_CLOT_DT step=(\d+) time=(\S+) deltaT=(\S+) .*?dtWave=(\S+) .*?binder=(\S+)", text, re.M):
            try:
                recs.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5)))
            except ValueError:
                continue
    if not recs:
        out["records"] = 0
        return out
    # A dtWave record is a RATE only when the wave clause was active: 0 < dtWave < 1e15. The solver writes 1e15
    # ("great", .C L2515/L13040/L13129) when the clause is inactive, and that sentinel must never become the step-1
    # REFERENCE either: (1e15 / 4e-6)^2 = 6e40 would flag a tail that is not there (case-engineer 32 hand-off;
    # the CE's root fix 2026-10-15). With no valid step-1 rate, R_t is not measured, and the reason is reported.
    def is_rate(v):
        return 0.0 < v < 1e15
    dtw1 = recs[0][3] if recs[0][0] == 1 and is_rate(recs[0][3]) else None
    rt = [((dtw1 / r[3]) ** 2, r[1]) for r in recs if dtw1 is not None and is_rate(r[3])]
    if dtw1 is None:
        out["R_t_not_measured"] = ("the log does not start at step 1" if recs[0][0] != 1 else
                                   "dtWave at step 1 is not a rate (%r: the wave clause was inactive)" % recs[0][3])
    out.update({"records": len(recs), "dtWave_step1": dtw1,
                "R_t_max": max(rt)[0] if rt else None, "R_t_max_at_t": max(rt)[1] if rt else None,
                "tail_flag": (max(rt)[0] >= params["tail_rt_min"]) if rt else None,
                "tail_localisation": "from the fields: see `tail` (the argmax of the tangent ratio over q > %g outside the clot body, at each full write)" % params["tail_q_outside_min"],
                "binder_counts": {}})
    for r in recs:
        out["binder_counts"][r[4]] = out["binder_counts"].get(r[4], 0) + 1
    if t_c is not None:
        pre = sorted(r[2] for r in recs if params["approach_start_s"] <= r[1] <= t_c)
        post = sorted(r[2] for r in recs if r[1] >= t_c)
        if pre and post:
            med = lambda v: v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
            out["dt_factor"] = {"pre_median": med(pre), "post_median": med(post), "post_min": post[0],
                                "median_ratio": med(post) / med(pre), "min_ratio": post[0] / med(pre)}
    return out


def f3_of_logs(texts: list, s25_report) -> dict:
    """F3 (design S25 3A: rMax = max k_true/k_used <= 4 and stiffNonPositive = 0, assess_case.s25_f3_report) from the PHASE_CLOT_S25_F3 lines of the sessions' logs."""
    if s25_report is None:
        return {"status": "NOT MEASURED", "why": "no s25_report reader was handed in (run through assess_case.py or scripts/y_plug_chain.py)"}
    joined = "\n".join(texts)
    if "PHASE_CLOT_S25_F3 " not in joined:
        return {"status": "NOT MEASURED", "why": "the log holds no PHASE_CLOT_S25_F3 line (a binary that predates S25 build 2, or an empty log)"}
    f3 = s25_report(joined).get("f3") or {}
    return {"status": "MEASURED", "verdict": f3.get("verdict"), "records": f3.get("records"), "r_max_max": f3.get("r_max_max"), "r_max_threshold": f3.get("r_max_threshold"),
            "stiff_non_positive_max": f3.get("stiff_non_positive_max"), "unparseable_records": f3.get("unparseable_records")}


# ------------------------------------------------------------------------------------------------------------- the chain end to end
def facts_of(info: dict) -> dict:
    g = info["geo"]
    return {"ubar": g["ubar"], "D0": g["D0"], "lam": g["lam"], "tau_v": g["tau_v"], "gap_m": g["gap_m"], "write_interval": g["write_interval"],
            "series": g["series"], "q0": g["q0"], "q0_tol": g["q0_tol"]}


def assess_case_dirs(cases: list, spec: dict, params: dict, texts=None, labels=None, s25_report=None) -> dict:
    """The chain of restarted sessions `cases` (case directories in order; the first holds the initial state) -> the analyse() result plus the chain's record.
    `spec` = child.json of the first session. A chain error is returned as status 'CHAIN_REFUSED' with the reason (never raised)."""
    sessions, x0 = [], None
    for k, c in enumerate(cases):
        try:
            red = reduce_case(c, spec, params, x0_hint=x0)
        except (OSError, ValueError, KeyError) as error:
            return {"status": "NOT_ASSESSABLE", "verdict": None, "final": False, "why": f"session {k + 1}: {type(error).__name__}: {error}", "sessions": k}
        if red["geo"] is None:
            if k == 0:
                return {"status": "NO_WRITES", "verdict": None, "final": False, "sessions": 1, "skipped": red["skipped"],
                        "why": "no full intact write in the session"}
            return {"status": "CHAIN_REFUSED", "verdict": None, "final": False, "why": f"session {k + 1} holds no full intact write", "sessions": k + 1}
        x0 = red.get("x0", x0)
        sessions.append({"records": red["records"], "restart": red["restart"], "label": (labels[k] if labels else f"session {k + 1}"), "info": red})
    info = sessions[0]["info"]
    facts = facts_of(info)
    if len(sessions) > 1:
        from restart_driver import compare_dictionaries
        for k in range(1, len(cases)):
            diff = compare_dictionaries(Path(cases[k]), Path(cases[k - 1]))
            if diff:
                return {"status": "CHAIN_REFUSED", "verdict": None, "final": False, "sessions": len(cases),
                        "why": f"session {k + 1}'s dictionaries differ from session {k}'s in {', '.join(diff)}: one pinned package per chain"}
    try:
        merged = merge_sessions(sessions, facts["write_interval"], params)
    except ChainError as error:
        return {"status": "CHAIN_REFUSED", "verdict": None, "final": False, "why": str(error), "sessions": len(cases)}
    out = analyse(merged, facts, params)
    out["sessions"] = len(cases)
    out["skipped"] = [dict(s, session=k + 1) for k, ss in enumerate(sessions) for s in ss["info"]["skipped"]]
    out["series_columns"] = columns(merged)
    if texts:
        out["log"] = log_reports(texts, out.get("t_c"), params, s25_report)
    return out


def columns(recs: list) -> dict:
    """The per-write series as columns (compact, for result.json and offline re-analysis)."""
    keys = ("t", "X", "A", "A_bulk", "Mp", "Mm", "content", "r_max", "n_over", "Dg_max", "lam1_bulk_max", "Q", "Qs")
    cols = {k: [r.get(k) for r in recs] for k in keys}
    for k in ("R_all", "R_outside", "R_body", "q_argmax"):
        cols["tail_" + k] = [(r.get("tail") or {}).get(k) for r in recs]
    return cols
