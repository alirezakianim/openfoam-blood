#!/usr/bin/env python3
"""S17 offline checks, run on a RETURNED batch-58 run (not assessor gates).

research/designs/2026-09-26-S17-frozen-release-implicit-elastic-startup.md,
revision 4, section 6.

  gid           G-id, em68 against em63: every PHASE_CLOT_* line common to
                both builds identical to the printed digits until the PARENT's
                first damage (em63: 1.352 ms). Delegated to
                s16_offline_checks.gid (the same comparator batch 57 used);
                tags only one build prints (PHASE_CLOT_S17*) are not compared,
                and said so.

  discriminator The two-step amplitude ratio A_n/A_{n-2} of the face-wise
                second difference (PHASE_CLOT_S17_SECONDDIFF), in 10's window
                for D1/D2/D2b (steps 20-40) and D3/D3b (steps 100-200), over
                every readable step for CB6/CB4 (the clot-surface line). Read
                only while A_n/||phi_n|| > 1e-10. D2b >= 1: B does not ship.
                CB6 >= 1: C destabilises the surface even with B. The other
                readings are reported against 10's 16/70-row predictions.

  fields        One streaming pass over a case-after-run archive (never
                extracted: the macOS c/C collision) or a case directory, for
                one child, at every written time:
                  P2    max s/s_c over pending cells with ind >= 1e-4, s =
                        (thI/S)(b1 - 1/b1) + (thP/S)(lf^2 - 1/lf), s_c =
                        lc^2 - lc^-2 (03's measure, split); FAIL > 1.05.
                  W2    max (lf^2 - 1/lf)/s_c over pending cells (WATCH,
                        0.9702).
                  W-tail the max lambda1 of INTACT wall-tail cells (x 45-60
                        um, the two wall rows, d* < 1) and their zigzag share
                        (WATCH; compared with --ref at matched time).
                  P1    every written d* = 1 cell (thI = 0) has be = I:
                        its own-stretch elasticity is 0 (rule 1); FAIL any.
                  P3    per the first 5 cells to reach d* = 1 (written): the
                        max slip |Vs - U| over [t_e, t_e + tau] at the event
                        cell and its face neighbours minus the same set's max
                        at the last pre-event write; median over the 5.
                        FAIL > 0.02 m/s; 0.010-0.02 recorded (suspected
                        under-release); < 0.004 recorded (the immediate part
                        can fall between writes).
                With --ref <child>: the strain comparison before the first
                rupture in either child (em69 vs em66, em70 vs em67): the
                bulk (ind >= 0.5) mean of lambda1 - 1 and of the weighted
                sigma_xy / G = max(ind, 1e-4)(1 - d) be_xy must agree within
                5%. FAIL > 5%.

  energy        16b, the pre-registered energy gate (design section 6, em66
                and em67: "dissipation <= the dp.Q ceiling at every write",
                03): at every written time, the discrete viscous dissipation
                of the written U,
                  Phi = mu sum_faces |U_N - U_P|^2 A_f/d_f
                        + mu sum_wall |U_P|^2 A_w/(d_w/2)   (no-slip walls),
                on the structured 448 x 70 mesh (0/Ccx, 0/Ccy, 0/Vc), against
                the physical power ceiling dp Q_open: dp = rho x
                throat_embolus.inlet_pressure_m2_s2, Q_open the open-channel
                Stokes flux in throat_embolus.measures (mainCaseControls.json;
                a clot only lowers it). PASS iff Phi <= dp Q_open at EVERY
                write. The measure is 03's, transcribed from
                probes/physics-em65-field-energy.py (numpy -> pure Python;
                its npz held U as float32, this reads the ascii doubles).
                Real-log negative em65 (batch 57): FAIL at x426-16520; control
                em64: PASS, <= 0.78 of the ceiling (energy-real-check).

  gid-physical  S17b (design section 4; 12 change 3): G-id em74 against
                em66 on the PHYSICAL lines, step 1 to the parent's first
                transfer (em66: 1.30519 ms). A2 changes the predictors'
                arithmetic from step 1, so the partition round-off keys, the
                projection counters, W2/P2 and mu are excluded, and so are
                keys and tags only one build prints; every exclusion is named.

  em74-sign     S17b (12 change 7): em74 (no A3) against em71 (A3), at each
                written time where em74's G8 cell is a wall-row cell: the
                sign of em71 - em74 in the G8 slip is predicted from em74's
                logged wall-face projection on the slip (positive: the wall
                face aided the slip, so A3 lowers G8). FAIL if it disagrees
                at more than 20% of those writes.

  energy-inlet  A5 of the Xu 2017 Figure 5 case (design 2026-09-29-case-
                xu2017-fig5 rev. 4, calibrated by numerics 02 addendum 3):
                the CUMULATIVE R = int Phi dt / int Q rho p_in dt at every
                write, Phi the phase-weighted viscous dissipation of the
                written U (fluid fraction 1 - clamp(thetaC + thetaH, 0, 0.9)),
                p_in the mean of column 0 of the written p, Q = U_mean H depth;
                rho, nu, depth and h READ FROM THE CASE. FAIL at R > 0.9915.

  self-test     The negative tests of every check.

Usage:
  python3 -B scripts/s17_offline_checks.py gid <em68 log> <em63 log> [--cutoff T]
  python3 -B scripts/s17_offline_checks.py discriminator <log> <D1|D2|D2b|D3|D3b|CB6|CB4>
  python3 -B scripts/s17_offline_checks.py fields <tar.gz | case dir> <child> [--ref <child>] [--tau 3.3e-4]
  python3 -B scripts/s17_offline_checks.py energy <tar.gz | case dir> <child> [--controls <mainCaseControls.json>]
  python3 -B scripts/s17_offline_checks.py energy-real-check <batch-57 case-after-run tar.gz>
  python3 -B scripts/s17_offline_checks.py energy-inlet <tar.gz | case dir> <xu child>
  python3 -B scripts/s17_offline_checks.py gid-physical <em74 log> <em66 log>
  python3 -B scripts/s17_offline_checks.py em74-sign <em74 log> <em71 log> [--write 2.5e-5]
  python3 -B scripts/s17_offline_checks.py self-test

Exit 0 = PASS, 1 = FAIL, 2 = usage / unreadable input.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import shutil
import sys
import tarfile
import tempfile
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSESS = ROOT / "cases/AM2020_planar_mural_detachment/scripts/assess_case.py"
S16 = ROOT / "scripts/s16_offline_checks.py"

LC = 2.8
S_C = LC**2 - LC**-2
PHI0 = 0.45
FLOOR = 1e-4
P2_LIMIT = 1.05
W2_LIMIT = 0.9702
P3_FAIL = 0.02
STRAIN_TOL = 0.05
FIELDS = ("thetaIntact", "thetaPending", "thetaBroken", "thetaC", "thetaH",
          "be", "pendingStretch", "U", "Vs", "thetaTaut")
DBL_MIN = sys.float_info.min   # S17b A2: the fibre is read where theta_T >= this
MESH = ("C", "owner", "neighbour")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(Path(path).read_text(), str(path), "exec"), mod.__dict__)
    return mod


# -------------------------------------------------------------- gid ------

def gid(child: Path, parent: Path, cutoff: float | None) -> tuple[bool, str]:
    return _load("s16_offline_for_s17", S16).gid(child, parent, cutoff)


# ---------------------------------------------------- discriminators -----

def discriminator(log_text: str, name: str) -> tuple[bool, str]:
    ac = _load("assess_case_s17_offline", ASSESS)
    if name in ac.S17_DISCRIMINATORS:
        window, p16, p70, gate = ac.S17_DISCRIMINATORS[name]
        r = ac.s17_second_difference(log_text, "outlet", window)
        head = (f"{name}: outlet two-step ratio median {r['median_ratio']} over "
                f"{r['readable']} readable steps in {window} (predicted {p16} / {p70} "
                f"at 16 / 70 rows)")
        if r["median_ratio"] is None:
            return False, head + "\nFAIL: nothing readable is not a reading"
        if gate:
            ok = r["median_ratio"] < 1
            return ok, head + ("\nPASS: B ships" if ok else "\nFAIL: >= 1 -- B does not ship; "
                               "the B children are read for A (and C) only")
        return True, head + "\nREPORTED (scores the model)"
    if name in ("CB6", "CB4"):
        r = ac.s17_second_difference(log_text, "surface", None)
        head = (f"{name}: clot-surface two-step ratio median {r['median_ratio']} over "
                f"{r['readable']} readable steps")
        if r["median_ratio"] is None:
            return False, head + "\nFAIL: nothing readable"
        if name == "CB6":
            ok = r["median_ratio"] < 1
            return ok, head + ("\nPASS" if ok else "\nFAIL: C destabilises the surface even with B")
        return True, head + "\nREPORTED (CB4 may exceed 1)"
    return False, f"unknown discriminator {name}"


# ------------------------------------------------------ field reading -----

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def parse_internal(text: str, n: int | None = None) -> list:
    """internalField of an ascii OpenFOAM field: a list of floats (scalar) or
    of tuples (vector/symmTensor); a uniform value is expanded to n."""
    m = re.search(r"internalField\s+uniform\s+(\([^)]*\)|" + _NUM + r")\s*;", text)
    if m:
        v = m.group(1)
        val = (tuple(float(x) for x in re.findall(_NUM, v)) if v.startswith("(")
               else float(v))
        if n is None:
            raise ValueError("a uniform field needs the cell count")
        return [val]*n
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\((.*?)\)\s*;",
                  text, re.S)
    if not m:
        raise ValueError("no internalField")
    kind, count, payload = m.group(1), int(m.group(2)), m.group(3)
    if kind == "scalar":
        vals = [float(x) for x in payload.split()]
    else:
        vals = [tuple(float(x) for x in t.split())
                for t in re.findall(r"\(([^()]*)\)", payload)]
    if len(vals) != count:
        raise ValueError(f"{kind} list: {len(vals)} values for {count}")
    return vals


def parse_labels(text: str) -> list[int]:
    m = re.search(r"(\d+)\s*\((.*?)\)", text[text.find("}") + 1:], re.S)
    if not m:
        raise ValueError("no label list")
    vals = [int(x) for x in m.group(2).split()]
    if len(vals) != int(m.group(1)):
        raise ValueError("label list length")
    return vals


def members(src: Path, child: str):
    """(time or 'mesh', name, text) for the child's fields and mesh, streamed,
    keyed by full path (no case-insensitive collision)."""
    wanted = set(FIELDS) | set(MESH)
    if src.is_dir():
        base = src / "runs" / child if (src / "runs" / child).is_dir() else src
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.name in wanted:
                rel = p.relative_to(base).parts
                yield _classify(rel, p.read_text(errors="replace"))
        return
    with tarfile.open(src, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = tuple(m.name.split("/"))
            if parts[-1] not in wanted or "runs" not in parts:
                continue
            i = parts.index("runs")
            if len(parts) <= i + 1 or parts[i + 1] != child:
                continue
            fh = tf.extractfile(m)
            yield _classify(parts[i + 2:], fh.read().decode("ascii", errors="replace"))


def _classify(rel: tuple, text: str):
    if len(rel) >= 3 and rel[-3] == "constant" and rel[-2] == "polyMesh":
        return ("mesh", rel[-1], text)
    if len(rel) == 2:
        try:
            float(rel[0])
        except ValueError:
            return (None, rel[-1], text)
        return (rel[0], rel[1], text)
    return (None, rel[-1], text)


def eig_sym3(b) -> tuple[float, float, float]:
    """Eigenvalues (descending) of a symmetric 3x3 (xx, xy, xz, yy, yz, zz),
    closed form (Smith 1961)."""
    a11, a12, a13, a22, a23, a33 = b
    p1 = a12*a12 + a13*a13 + a23*a23
    q = (a11 + a22 + a33)/3.0
    if p1 == 0:
        return tuple(sorted((a11, a22, a33), reverse=True))
    p2 = (a11 - q)**2 + (a22 - q)**2 + (a33 - q)**2 + 2*p1
    p = math.sqrt(p2/6.0)
    B = [(a11 - q)/p, a12/p, a13/p, (a22 - q)/p, a23/p, (a33 - q)/p]
    detB = (B[0]*(B[3]*B[5] - B[4]*B[4]) - B[1]*(B[1]*B[5] - B[4]*B[2])
            + B[2]*(B[1]*B[4] - B[3]*B[2]))
    r = max(-1.0, min(1.0, detB/2.0))
    phi = math.acos(r)/3.0
    e1 = q + 2*p*math.cos(phi)
    e3 = q + 2*p*math.cos(phi + 2*math.pi/3)
    return (e1, 3*q - e1 - e3, e3)


def zigzag_share(values: list[float]) -> float:
    """The share of samples inside runs of >= 3 alternating-sign successive
    differences (field-forensics' zigzag detector; 0 on a monotone or
    single-bump profile, lesson odd-even-detector)."""
    if len(values) < 4:
        return 0.0
    df = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    inside = [False]*len(values)
    i = 0
    while i < len(df) - 1:
        j = i
        while j + 1 < len(df) and df[j]*df[j + 1] < 0:
            j += 1
        if j - i + 1 >= 3:
            for k in range(i, j + 2):
                inside[k] = True
        i = j + 1 if j > i else i + 1
    return sum(inside)/len(values)


def per_write(f: dict, n: int, cc: list | None) -> dict:
    """The per-write quantities from one written time's fields."""
    tI, tP, tB = f["thetaIntact"], f["thetaPending"], f["thetaBroken"]
    # S17b A1/A2: under fibreSlack v1 (thetaTaut written) the fibre's carrier
    # is the taut share, read where it is >= DBL_MIN; otherwise S17's P2 > 0.
    carrier = f.get("thetaTaut", tP)
    read_min = DBL_MIN if "thetaTaut" in f else math.ulp(0.0)
    th = [a + b for a, b in zip(f["thetaC"], f["thetaH"])]
    be, lf = f["be"], f.get("pendingStretch", [1.0]*n)
    p2 = w2 = 0.0
    p1_bad = 0
    lam1, dstar, ind, slip, wbxy = array("d"), array("d"), array("d"), array("d"), array("d")
    for c in range(n):
        S = tI[c] + (tP[c] + tB[c])
        aI, aP = (tI[c]/S, carrier[c]/S) if S > 0 else (1.0, 0.0)
        d = tB[c]/S if S > 0 else 0.0
        ds = (tP[c] + tB[c])/S if S > 0 else 0.0
        b1 = eig_sym3(be[c])[0]
        ic = max(th[c], 0.0)/PHI0
        lam1.append(math.sqrt(b1))
        dstar.append(ds)
        ind.append(ic)
        if "U" in f and "Vs" in f:
            u, v = f["U"][c], f["Vs"][c]
            slip.append(math.sqrt(sum((v[k] - u[k])**2 for k in range(3))))
        wbxy.append(max(ic, FLOOR)*(1.0 - d)*be[c][1])
        # P1: a cell whose network is all broken or pending (d* = 1, so
        # thI = 0 exactly) has no own-stretch elasticity: its be is I
        # (rule 1) and the intact weight on it is 0.
        if S > 0 and tI[c] == 0 and tuple(be[c]) != (1.0, 0.0, 0.0, 1.0, 0.0, 1.0):
            p1_bad += 1
        if carrier[c] >= read_min:
            T = lf[c]**2 - 1.0/lf[c]
            w2 = max(w2, T/S_C)
            if ic >= 1e-4:
                p2 = max(p2, (aI*(b1 - 1.0/b1) + aP*T)/S_C)
    out = {"p2": p2, "w2": w2, "p1_bad": p1_bad, "lam1": lam1, "dstar": dstar,
           "ind": ind, "slip": slip, "wbxy": wbxy}
    bulk = [c for c in range(n) if ind[c] >= 0.5]
    out["bulk_strain"] = (sum(lam1[c] - 1.0 for c in bulk)/len(bulk)) if bulk else None
    out["bulk_sxy"] = (sum(wbxy[c] for c in bulk)/len(bulk)) if bulk else None
    out["any_rupture"] = any(x > 0 for x in dstar)
    if cc is not None:
        tail = [c for c in range(n) if 45e-6 <= cc[c][0] <= 60e-6]
        ys = sorted({round(cc[c][1], 12) for c in tail})[:2]
        rows = {y: sorted((c for c in tail if round(cc[c][1], 12) == y), key=lambda c: cc[c][0])
                for y in ys}
        intact = [c for y in ys for c in rows[y] if dstar[c] < 1]
        out["wtail_lam1_max"] = max((lam1[c] for c in intact), default=None)
        out["wtail_zigzag"] = (max(zigzag_share([lam1[c] for c in rows[y]]) for y in ys)
                               if ys else None)
    return out


def scan(src: Path, child: str) -> tuple[list, list, dict]:
    """[(t, per_write)], cell centres, face adjacency."""
    cur_t, cur, n = None, {}, None
    mesh: dict = {}
    writes = []
    cc = None

    def flush():
        nonlocal cur, cur_t
        if cur_t is not None and all(k in cur for k in ("thetaIntact", "thetaPending",
                                                         "thetaBroken", "thetaC", "thetaH", "be")):
            nn = len(cur["thetaC"])
            writes.append((float(cur_t), per_write(cur, nn, cc)))
        cur, cur_t = {}, None

    for t, name, text in members(src, child):
        if t == "mesh":
            mesh[name] = parse_labels(text)
            continue
        if t is None:
            continue
        if name == "C" and float(t) == 0:
            cc = parse_internal(text)
            continue
        if float(t) <= 0:
            continue
        if t != cur_t:
            flush()
            cur_t = t
        cur[name] = parse_internal(text, n)
        if n is None and isinstance(cur[name], list):
            n = len(cur[name])
    flush()
    writes.sort(key=lambda w: w[0])
    return writes, cc, mesh


def neighbours(mesh: dict, ncells: int) -> list[list[int]]:
    nb = [[] for _ in range(ncells)]
    own, nei = mesh.get("owner", []), mesh.get("neighbour", [])
    for f in range(len(nei)):
        a, b = own[f], nei[f]
        nb[a].append(b)
        nb[b].append(a)
    return nb


def p3(writes: list, nb: list, tau: float) -> tuple[bool | None, str]:
    events = []
    seen = set()
    for i, (t, w) in enumerate(writes):
        for c, ds in enumerate(w["dstar"]):
            if ds >= 1.0 and c not in seen:
                seen.add(c)
                events.append((t, c, i))
        if len(events) >= 5:
            break
    events = events[:5]
    if not events:
        return None, "P3: no cell reached d* = 1 in the written fields"
    rises = []
    for t_e, c, i in events:
        cells = [c] + nb[c]
        if i == 0 or not writes[i - 1][1]["slip"]:
            continue
        before = max(writes[i - 1][1]["slip"][k] for k in cells)
        after = max(w["slip"][k] for tt, w in writes[i:] if tt <= t_e + tau for k in cells)
        rises.append(after - before)
    if not rises:
        return None, "P3: no event with a pre-event write"
    rises.sort()
    med = rises[len(rises)//2]
    note = ("FAIL: > 0.02 m/s (IZ's scale; the timed release is not acting)" if med > P3_FAIL
            else "recorded: 0.010-0.02 (SUSPECTED under-release)" if med > 0.010
            else "recorded: < 0.004 (the immediate part can fall between writes)" if med < 0.004
            else "PASS: within 0.005-0.010 predicted")
    return med <= P3_FAIL, f"P3: median per-event slip rise {med:.6g} m/s over {len(rises)} event(s); {note}"


def strain(writes_a: list, writes_b: list) -> tuple[bool | None, str]:
    tb = {round(t, 12): w for t, w in writes_b}
    rows = []
    for t, w in writes_a:
        v = tb.get(round(t, 12))
        if v is None:
            continue
        if w["any_rupture"] or v["any_rupture"]:
            break
        rows.append((t, w, v))
    if not rows:
        return None, "strain: no matched write before the first rupture"
    worst = 0.0
    for t, w, v in rows:
        for k in ("bulk_strain", "bulk_sxy"):
            a, b = w[k], v[k]
            if a is None or b is None or b == 0:
                continue
            worst = max(worst, abs(a - b)/abs(b))
    ok = worst <= STRAIN_TOL
    return ok, (f"strain: lambda1 - 1 and sigma_xy agree within {100*worst:.3g}% over "
                f"{len(rows)} matched pre-rupture write(s) (limit 5%)"
                + ("" if ok else " -- FAIL: C changes the physics before damage"))


def fields(src: Path, child: str, ref: str | None, tau: float) -> tuple[bool, str]:
    writes, cc, mesh = scan(src, child)
    if not writes:
        return False, f"FAIL: no written time with the partition for {child}"
    lines, ok = [], True
    p2max = max(w["p2"] for _, w in writes)
    w2max = max(w["w2"] for _, w in writes)
    ok &= p2max <= P2_LIMIT
    lines.append(f"P2: max s/s_c {p2max:.6g} over {len(writes)} write(s) "
                 + ("PASS" if p2max <= P2_LIMIT else "FAIL (> 1.05)"))
    lines.append(f"W2 (watch): max T/s_c {w2max:.6g} (0.9702)")
    p1 = sum(w["p1_bad"] for _, w in writes)
    ok &= p1 == 0
    lines.append(f"P1: {p1} written d* = 1 cell(s) with be != I "
                 + ("PASS" if p1 == 0 else "FAIL (own-stretch elasticity not 0)"))
    wt = [(t, w.get("wtail_lam1_max"), w.get("wtail_zigzag")) for t, w in writes]
    lines.append("W-tail (watch): " + "; ".join(
        f"{t:.6g}: lam1 {a if a is None else round(a, 5)}, zigzag {z if z is None else round(z, 3)}"
        for t, a, z in wt[:40]))
    if mesh.get("owner"):
        r, msg = p3(writes, neighbours(mesh, len(writes[0][1]["dstar"])), tau)
        lines.append(msg)
        if r is False:
            ok = False
    else:
        lines.append("P3: no polyMesh owner/neighbour in the source -- NOT EVALUATED")
    if ref:
        wr, _, _ = scan(src, ref)
        r, msg = strain(writes, wr)
        lines.append(msg)
        if r is False:
            ok = False
        ta = {round(t, 12): w for t, w in wr}
        cmp_ = [(t, w.get("wtail_lam1_max"), ta.get(round(t, 12), {}).get("wtail_lam1_max"))
                for t, w in writes if round(t, 12) in ta]
        lines.append("W-tail vs ref at matched time (watch): " + "; ".join(
            f"{t:.6g}: {a} vs {b}" for t, a, b in cmp_[:40]))
    return ok, "\n".join([f"{child}:"] + lines)


# ------------------------------------------------------- 16b: energy ------

# rho, nu, the planar depth and the cell width are READ FROM THE CASE: the
# child's own constant/phaseClotProperties (rho, nu) and child.json
# (geometry.depth_m, geometry.h_m), and the cell width is MEASURED from 0/Ccx
# and required to equal h_m. Until 2026-09-30 they were hard-coded here to the
# throat case's 1050 / 3.3333e-6 / 5/7 um / 1 um, which would have scored the
# Xu case (rho 1025, 4 cP) x0.854 low -- in the masking direction (numerics 02
# addendum 3, docket 20260929-1430-case-xu2017-...).
CONTROLS = ROOT / "cases/AM2020_planar_mural_detachment/constant/mainCaseControls.json"
ENERGY_MESH = ("Ccx", "Ccy", "Vc")
CASE_FILES = ("child.json", "phaseClotProperties")


def _case_stream(src: Path, child: str, names: set):
    """(relative parts under the child's directory, text) for every file of
    the child whose name is in `names`: a case directory or a
    case-after-run tar.gz (streamed, never extracted -- the macOS c/C
    collision)."""
    if src.is_dir():
        base = src / "runs" / child if (src / "runs" / child).is_dir() else src
        for pth in sorted(base.rglob("*")):
            if pth.is_file() and pth.name in names:
                yield pth.relative_to(base).parts, pth.read_text(errors="replace")
        return
    with tarfile.open(src, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = tuple(m.name.split("/"))
            if parts[-1] not in names or "runs" not in parts:
                continue
            i = parts.index("runs")
            if len(parts) <= i + 2 or parts[i + 1] != child:
                continue
            yield parts[i + 2:], tf.extractfile(m).read().decode("ascii", errors="replace")


def case_constants(files: dict) -> dict:
    """rho, nu, mu, depth, h (and the Xu inlet's U_mean, H) from the child's
    OWN child.json and constant/phaseClotProperties. A missing value is an
    unreadable input, never a default."""
    import json
    props, cj = files.get(("constant", "phaseClotProperties")), files.get(("child.json",))
    if props is None or cj is None:
        raise ValueError("the child's constant/phaseClotProperties and child.json are both required "
                         "(rho, nu, depth and h are read from the case, never assumed)")
    got = {}
    for k in ("rho", "nu"):
        m = re.search(rf"(?m)^\s*{k}\s+(?:\[[^\]]*\]\s+)?([-\d.eE+]+)\s*;", props)
        if not m:
            raise ValueError(f"phaseClotProperties carries no {k}")
        got[k] = float(m.group(1))
    spec = json.loads(cj)
    g = spec["geometry"]
    got.update(mu=got["rho"]*got["nu"], depth=float(g["depth_m"]), h=float(g["h_m"]),
               H=float(g["channel_height_m"]), u_mean=float(spec["mean_velocity_m_s"]),
               pressure_driven=spec.get("pressure_driven"))
    return got


def energy_ceiling(controls: Path = CONTROLS, rho: float = 1050.0) -> tuple[float, float, float]:
    """(dp, Q_open, dp*Q_open), READ from the controls: dp = rho x the
    series' inlet kinematic pressure (outlet 0), Q_open the open-channel
    reference flux the series' `measures` text states. rho is the case's
    (energy() passes the child's own); 1050.0 is only the default a caller
    that has no case to read (the self-test's controls check) falls back to."""
    import json
    te = json.loads(Path(controls).read_text())["series"]["throat_embolus"]
    dp = rho*float(te["inlet_pressure_m2_s2"])
    m = re.search(r"open-channel reference \(([0-9.eE+-]+) m3/s\)", te["measures"])
    if not m:
        raise ValueError("throat_embolus.measures no longer states the open-channel flux")
    q = float(m.group(1))
    return dp, q, dp*q


class StructuredMesh:
    """A planar x-fastest mesh (cell index = j*NX + i). ASSERTED, not assumed:
    every column must share one x (to 1e-12 m), x must increase along a row at
    one spacing to 1e-4 relative, and that spacing (the mean over the row)
    must equal the case's h when given (same tolerance). The cell width is
    measured, the depth is the case's.
    WHY 1e-4 AND NOT TIGHTER, MEASURED: on batch 57's throat mesh
    (em64_G4_s16, 0/Ccx in the case-after-run archive) the centre spacing runs
    7.142767e-7 to 7.143082e-7 m from column 168, where the taper begins:
    apply_throat.py bends the cells and their centroids move by up to 2.2e-5
    of a spacing. 1e-9 and 1e-6 both refused that real mesh. The straight Xu
    channel is uniform to the 12 written digits."""

    def __init__(self, x: list, y: list, V: list, depth: float, h: float | None = None):
        n = len(x)
        row0 = 1
        while row0 < n and x[row0] > x[row0 - 1]:
            row0 += 1
        self.NX = row0
        if self.NX < 2 or n % self.NX:
            raise ValueError(f"not a structured x-fastest mesh ({n} cells, first row {self.NX})")
        self.NY = n // self.NX
        NX = self.NX
        for j in range(self.NY):
            for i in range(NX):
                if abs(x[j*NX + i] - x[i]) > 1e-12:
                    raise ValueError(f"column {i} is not one x at row {j}: the structured premise fails")
        self.dx0 = (x[NX - 1] - x[0])/(NX - 1)
        for i in range(NX - 1):
            if not math.isclose(x[i + 1] - x[i], self.dx0, rel_tol=1e-4):
                raise ValueError(f"the x spacing is not uniform at column {i}")
        if h is not None and not math.isclose(self.dx0, h, rel_tol=1e-4):
            raise ValueError(f"the measured cell width {self.dx0!r} is not the case's h {h!r}")
        self.depth = depth
        self.x, self.y = x, y
        self.dy = [v/(self.dx0*depth) for v in V]

    def dissipation(self, U: list, mu: float, phiF: list | None = None) -> float:
        """mu [sum_faces w_f |dU|^2 A_f/d_f + sum_walls w_P |U_P|^2 A_w/(d_w/2)],
        w = 1, or the fluid fraction phiF (faces: the arithmetic mean of the
        two cells, the linear interpolate on this uniform mesh; walls: the
        wall cell's) -- the dissipation of the equation the solver solves,
        whose fluid row is phase-weighted (numerics 02 addendum 3)."""
        NX, NY, x, y, dy, D = self.NX, self.NY, self.x, self.y, self.dy, self.depth
        wt = (lambda a, c: 1.0) if phiF is None else (lambda a, c: 0.5*(phiF[a] + phiF[c]))
        sx = sy = sw = 0.0
        for j in range(NY):
            b = j*NX
            for i in range(NX - 1):
                a, c = b + i, b + i + 1
                du = (U[c][0] - U[a][0])**2 + (U[c][1] - U[a][1])**2
                sx += wt(a, c)*du*(0.5*(dy[c] + dy[a])*D)/(x[c] - x[a])
        Afy = self.dx0*D
        for j in range(NY - 1):
            for i in range(NX):
                a, c = j*NX + i, (j + 1)*NX + i
                du = (U[c][0] - U[a][0])**2 + (U[c][1] - U[a][1])**2
                sy += wt(a, c)*du*Afy/(y[c] - y[a])
        for i in range(NX):
            for a in (i, (NY - 1)*NX + i):
                sw += wt(a, a)*(U[a][0]**2 + U[a][1]**2)*Afy/(dy[a]/2)
        return mu*(sx + sy + sw)


class _Ready:
    """Mesh and case constants, built as soon as the stream has delivered
    them; written fields that arrive earlier wait (parsed) in `pending`."""

    def __init__(self):
        self.files: dict = {}
        self.mesh_raw: dict = {}
        self.mesh = None
        self.k = None

    def take(self, parts, text) -> bool:
        """True if (parts, text) was a mesh or case file (consumed)."""
        if parts in (("child.json",), ("constant", "phaseClotProperties")):
            self.files[parts] = text
        elif len(parts) == 2 and parts[0] == "0" and parts[1] in ENERGY_MESH:
            self.mesh_raw[parts[1]] = parse_internal(text)
        else:
            return False
        if (self.mesh is None and len(self.files) == 2
                and all(m in self.mesh_raw for m in ENERGY_MESH)):
            self.k = case_constants(self.files)
            self.mesh = StructuredMesh(self.mesh_raw["Ccx"], self.mesh_raw["Ccy"],
                                       self.mesh_raw["Vc"], self.k["depth"], self.k["h"])
        return True


def energy(src: Path, child: str, controls: Path = CONTROLS) -> tuple[bool, str, dict]:
    r = _Ready()
    rows, pending = [], []
    try:
        for parts, text in _case_stream(src, child, set(ENERGY_MESH) | {"U"} | set(CASE_FILES)):
            if r.take(parts, text):
                if r.mesh is not None and pending:
                    rows += [(tv, r.mesh.dissipation(UU, r.k["mu"])) for tv, UU in pending]
                    pending = []
                continue
            if len(parts) != 2 or parts[1] != "U":
                continue
            try:
                tv = float(parts[0])
            except ValueError:
                continue
            if tv <= 0:
                continue
            UU = parse_internal(text, len(r.mesh_raw.get("Vc", [])) or None)
            if r.mesh is None:
                pending.append((tv, UU))
            else:
                rows.append((tv, r.mesh.dissipation(UU, r.k["mu"])))
    except (ValueError, KeyError) as error:
        return False, f"{child}: FAIL: {error}", {}
    if r.mesh is None:
        return False, (f"{child}: FAIL: the mesh (0/Ccx, 0/Ccy, 0/Vc) or the case files (child.json, "
                       f"constant/phaseClotProperties) are missing -- the energy gate cannot read them"), {}
    dp, q, pmax = energy_ceiling(controls, r.k["rho"])
    mesh = r.mesh
    if not rows:
        return False, f"{child}: FAIL: no written U after t = 0 -- nothing checked is not a pass", {}
    rows.sort()
    ratios = [phi/pmax for _, phi in rows]
    rmin, rmax = min(ratios), max(ratios)
    med = sorted(ratios)[len(ratios)//2]
    over = [(t, r) for (t, _), r in zip(rows, ratios) if r > 1.0]
    ok = not over
    msg = (f"{child}: energy gate, Phi/(dp Q_open) over {len(rows)} writes: min {rmin:.4g}, "
           f"median {med:.4g}, max {rmax:.4g} (dp {dp:.4g} Pa, Q_open {q:.5g} m3/s, "
           f"ceiling {pmax:.4g} W); mesh {mesh.NX} x {mesh.NY}\n"
           + ("PASS: dissipation <= dp Q_open at every write" if ok else
              f"FAIL: {len(over)} of {len(rows)} writes above the ceiling; first at "
              f"t = {over[0][0]:.6g} s (x{over[0][1]:.4g})"))
    return ok, msg, {"min": rmin, "median": med, "max": rmax, "writes": len(rows)}


def energy_real_check(archive: Path) -> tuple[bool, str]:
    """The Chief Engineer's two real-log tests on the batch-57 archive: em65
    must FAIL at x426-16520 (03: min 426.5, max 1.652e4), em64 must PASS at
    <= 0.78 of the ceiling (03: max 0.7799). Bands are 03's printed values
    to the precision printed (float32 there, doubles here)."""
    out, ok_all = [], True
    ok65, msg65, r65 = energy(archive, "em65_G4_s16_final")
    good65 = (not ok65) and r65 and 426.0 <= r65["min"] <= 427.0 and 1.64e4 <= r65["max"] <= 1.66e4
    ok_all &= bool(good65)
    out.append(msg65 + f"\n  negative: must FAIL in x426-16520 -> {'as expected' if good65 else 'NOT AS EXPECTED'}")
    ok64, msg64, r64 = energy(archive, "em64_G4_s16")
    good64 = ok64 and r64 and r64["max"] <= 0.78
    ok_all &= bool(good64)
    out.append(msg64 + f"\n  control: must PASS, <= 0.78 of the ceiling -> {'as expected' if good64 else 'NOT AS EXPECTED'}")
    return ok_all, "\n".join(out)


# ---------------------------------------- A5: energy, velocity inlet ------
#
# The Xu 2017 Figure 5 case (design 2026-09-29-case-xu2017-fig5 rev. 4,
# section 4 row A5 and its note, "as calibrated" by numerics 02 addendum 3):
#   R(t) = int_0^t Phi dt / int_0^t Q rho p_in dt   (cumulative trapezoid over
#          the written times, from t = 0 where U = 0 and p = 0);
#   Phi  = mu [sum_faces phiF_f |dU|^2 A/d + sum_walls phiF_P |U_P|^2 A/(d/2)],
#          phiF = 1 - clamp(thetaC + thetaH, 0, 0.9) (the solver's own fluid
#          fraction, .C L18605-18610), interior faces and the two walls, NOT
#          the inlet and outlet faces (the calibration's convention);
#   p_in = the mean of column 0 of the written (kinematic) p -- exact for the
#          plug inlet, whose p is zeroGradient; x rho from the case;
#   Q    = U_mean H depth, from the case (the plug).
#   FAIL at R > 0.9915 at any write: the empty-channel floor 0.9865 +- 0.0005
#   plus 0.005 (energy creation of ~0.5 % of the input).
A5_FAIL = 0.9915
A5_FLOOR = 0.9865
A5_FIELDS = ("U", "p", "thetaC", "thetaH")


def energy_inlet(src: Path, child: str) -> tuple[bool, str, dict]:
    r = _Ready()
    by_t: dict = {}
    try:
        for parts, text in _case_stream(src, child, set(ENERGY_MESH) | set(A5_FIELDS) | set(CASE_FILES)):
            if r.take(parts, text):
                continue
            if len(parts) != 2 or parts[1] not in A5_FIELDS:
                continue
            try:
                tv = float(parts[0])
            except ValueError:
                continue
            if tv > 0:
                by_t.setdefault(tv, {})[parts[1]] = text
    except (ValueError, KeyError) as error:
        return False, f"{child}: UNREADABLE: {error}", {"unreadable": True}
    if r.mesh is None:
        return False, (f"{child}: UNREADABLE: the mesh (0/Ccx, 0/Ccy, 0/Vc) or the case files "
                       f"(child.json, constant/phaseClotProperties) are missing"), {"unreadable": True}
    k, mesh = r.k, r.mesh
    if k["pressure_driven"] is not False:
        return False, (f"{child}: UNREADABLE: A5 is the velocity-inlet energy bound; child.json "
                       f"pressure_driven is {k['pressure_driven']!r}"), {"unreadable": True}
    n = mesh.NX*mesh.NY
    Q = k["u_mean"]*k["H"]*k["depth"]
    series = [(0.0, 0.0, 0.0)]
    for tv in sorted(by_t):
        f = by_t[tv]
        missing = [x for x in A5_FIELDS if x not in f]
        if missing:
            return False, f"{child}: UNREADABLE: {missing} not written at t = {tv}", {"unreadable": True}
        U = parse_internal(f["U"], n)
        p = parse_internal(f["p"], n)
        th = [a + b for a, b in zip(parse_internal(f["thetaC"], n), parse_internal(f["thetaH"], n))]
        phiF = [1.0 - min(max(v, 0.0), 0.9) for v in th]
        phi = mesh.dissipation(U, k["mu"], phiF)
        p_in = sum(p[j*mesh.NX] for j in range(mesh.NY))/mesh.NY
        series.append((tv, phi, Q*k["rho"]*p_in))
    if len(series) < 2:
        return False, f"{child}: FAIL: no written U/p after t = 0 -- nothing checked is not a pass", {}
    cum_phi = cum_in = 0.0
    rows = []
    for (t0, f0, p0), (t1, f1, p1) in zip(series, series[1:]):
        cum_phi += 0.5*(f0 + f1)*(t1 - t0)
        cum_in += 0.5*(p0 + p1)*(t1 - t0)
        rows.append((t1, cum_phi/cum_in if cum_in > 0 else float("inf"), f1/p1 if p1 > 0 else float("inf")))
    over = [(t, R) for t, R, _i in rows if R > A5_FAIL]
    ok = not over
    Rs = [R for _t, R, _i in rows]
    msg = (f"{child}: A5 cumulative R = int Phi / int Q rho p_in over {len(rows)} writes: first "
           f"{rows[0][1]:.5f} (empty-channel floor {A5_FLOOR}; the first write should read at or below it), "
           f"max {max(Rs):.5f}, last {Rs[-1]:.5f}; instantaneous last {rows[-1][2]:.5f} (reported); "
           f"rho {k['rho']}, nu {k['nu']:.6g}, depth {k['depth']:.3g}, h {mesh.dx0:.6g} (from the case); "
           f"mesh {mesh.NX} x {mesh.NY}\n"
           + (f"PASS: R <= {A5_FAIL} at every write" if ok else
              f"FAIL: {len(over)} of {len(rows)} writes above {A5_FAIL}; first at t = {over[0][0]:.6g} s "
              f"(R {over[0][1]:.5f})"))
    return ok, msg, {"first": rows[0][1], "max": max(Rs), "last": Rs[-1], "writes": len(rows),
                     "rows": rows}


# ------------------------------------------------ S17b: two-log checks --

GID_EXCLUDED_KEYS = (
    # the partition's round-off measures (PHASE_CLOT_S16; em66 prints
    # mirrorDevMax 1.2e-16 and partitionSumDevAbs 5.6e-17 at step 1)
    "partitionMin", "partitionSubnormalCells", "muMin", "muMean", "foutMax",
    "partitionSumDevAbs", "partitionSumDevMax", "mirrorDevMax",
    # the projection counters and W2/P2 (A2 changes what they read)
    "projRaised", "projCount",
    "projExcessMax", "projTies", "w2Max", "p2Max",
)
_KV = re.compile(r"(\w+)=(\S+)")
TIME = re.compile(r"\btime=(\S+)")


def _first_transfer(path: Path) -> float | None:
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("PHASE_CLOT_S17 "):
                d = dict(_KV.findall(line))
                try:
                    if float(d.get("transfers", 0)) > 0:
                        return float(d["time"])
                except ValueError:
                    continue
    return None


def _window(path: Path, cutoff: float) -> dict:
    out: dict = {}
    with open(path, errors="replace") as fh:
        for raw in fh:
            m = re.match(r"^(PHASE_CLOT_[A-Z0-9_]+)\s", raw)
            if not m:
                continue
            t = TIME.search(raw)
            if t and float(t.group(1)) >= cutoff:
                break
            out.setdefault(m.group(1), []).append(dict(_KV.findall(raw)))
    return out


def gid_physical(child: Path, parent: Path, cutoff: float | None) -> tuple[bool, str]:
    if cutoff is None:
        cutoff = _first_transfer(parent)
        if cutoff is None:
            return False, "FAIL: the parent log has no transfer; no window"
    a, b = _window(child, cutoff), _window(parent, cutoff)
    excluded_tags = sorted(t for t in set(a) | set(b)
                           if t.endswith("_CONFIG") or t == "PHASE_CLOT_COST"
                           or t not in a or t not in b)
    common = sorted(set(a) & set(b) - set(excluded_tags))
    rep_ = [f"window: time < {cutoff!r} (the parent's first transfer)",
            "excluded tags: " + ", ".join(excluded_tags),
            "excluded keys: " + ", ".join(GID_EXCLUDED_KEYS)
            + " and every key only one build prints"]
    compared, first_bad, dropped = 0, None, set()
    for tag in common:
        la, lb = a[tag], b[tag]
        if len(la) != len(lb) and first_bad is None:
            first_bad = (tag, min(len(la), len(lb)), f"{len(la)} records against {len(lb)}")
        for i, (x, y) in enumerate(zip(la, lb)):
            keys = (set(x) & set(y)) - set(GID_EXCLUDED_KEYS)
            dropped |= (set(x) ^ set(y))
            compared += 1
            if any(x[k] != y[k] for k in keys) and first_bad is None:
                k0 = next(k for k in sorted(keys) if x[k] != y[k])
                first_bad = (tag, i, f"{k0}: child {x[k0]} parent {y[k0]}")
    rep_.append(f"records compared: {compared} over {len(common)} tags; one-build keys: "
                + ", ".join(sorted(dropped))[:400])
    if compared == 0:
        return False, "\n".join(rep_ + ["FAIL: nothing compared"])
    if first_bad:
        tag, i, what = first_bad
        return False, "\n".join(rep_ + [f"FAIL: first difference at {tag} record {i}: {what}"])
    return True, "\n".join(rep_ + ["PASS: identical on the physical lines"])


def _series(path_or_text, tag: str) -> list[dict]:
    text = (path_or_text.read_text(errors="replace") if isinstance(path_or_text, Path)
            else path_or_text)
    out = []
    for m in re.finditer(rf"^{tag} (.*)$", text, re.M):
        d = {}
        for k, v in _KV.findall(m.group(1)):
            try:
                d[k] = float(v)
            except ValueError:
                pass
        out.append(d)
    return out


def em74_sign(em74, em71, write: float = 2.5e-5) -> tuple[bool | None, str]:
    """em74, em71: Path or log text."""
    def at_writes(rows):
        out = {}
        for r in rows:
            t = r.get("time")
            if t is None:
                continue
            k = round(t/write)
            if k > 0 and abs(t - k*write) <= 1e-9*write*max(k, 1):
                out[k] = r
        return out
    g74 = at_writes(_series(em74, "PHASE_CLOT_S17B_G8CELL"))
    g71 = at_writes(_series(em71, "PHASE_CLOT_S17B_G8CELL"))
    n = agree = 0
    rows = []
    for k in sorted(set(g74) & set(g71)):
        r74, r71 = g74[k], g71[k]
        if r74.get("wallRow") != 1:
            continue
        proj = r74.get("wallFaceProj", 0.0)
        if proj == 0:
            continue
        predicted = -1 if proj > 0 else 1        # sign of em71 - em74
        got = r71["slip"] - r74["slip"]
        if got == 0:
            continue
        n += 1
        ok = (got > 0) == (predicted > 0)
        agree += ok
        rows.append((k*write, proj, got, ok))
    if n == 0:
        return None, "em74-sign: no written time with a wall-row G8 cell -- NOT EVALUABLE"
    frac = 1 - agree/n
    verdict = frac <= 0.2
    return verdict, (f"em74-sign: {n} wall-row writes; the sign disagrees at {n - agree} "
                     f"({100*frac:.1f}%; limit 20%) -> {'PASS' if verdict else 'FAIL'}")


# ------------------------------------------------------------- self-test --

def self_test() -> bool:
    ok_all = True

    def expect(name, got, want):
        nonlocal ok_all
        good = got == want
        ok_all &= good
        print(f"{'PASS' if good else 'FAIL'} {name}: got {got}, want {want}")

    # discriminators, from synthetic second-difference lines
    def log(ratio, surface=False, steps=220):
        lines, a = [], 1.0
        for n in range(1, steps):
            a *= ratio ** 0.5
            lines.append(f"PHASE_CLOT_S17_SECONDDIFF time={n*4e-6} step={n} valid=1 "
                         f"outletSecondDiff={a} outletPhiNorm=1 outletPhiSum=1 "
                         f"surfaceSecondDiff={a} surfacePhiNorm=1 surfaceFaces=9\n")
        return "".join(lines)
    expect("D2b passes at 0.636", discriminator(log(0.636), "D2b")[0], True)
    expect("D2b fails at 1.2", discriminator(log(1.2), "D2b")[0], False)
    expect("D2b fails on an empty log", discriminator("", "D2b")[0], False)
    expect("CB6 fails at 1.1", discriminator(log(1.1), "CB6")[0], False)
    expect("CB6 passes at 0.8", discriminator(log(0.8), "CB6")[0], True)
    below = log(0.5).replace("outletPhiNorm=1", "outletPhiNorm=1e20")
    expect("a ratio below the 1e-10 floor is not read", discriminator(below, "D2b")[0], False)

    # the field checks on a synthetic 6-cell strip
    def fixture(p_frac, lf, slip_rise):
        n = 6
        be_iso = (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        base = {"thetaC": [0.4]*n, "thetaH": [0.05]*n, "be": [be_iso]*n,
                "U": [(0.01, 0, 0)]*n}
        w0 = dict(base, thetaIntact=[0.45]*n, thetaPending=[0.0]*n,
                  thetaBroken=[0.0]*n, pendingStretch=[1.0]*n, Vs=[(0.01, 0, 0)]*n)
        tP = [0.0]*n
        tP[2] = 0.45*p_frac
        w1 = dict(base, thetaIntact=[0.45 if c != 2 else 0.0 for c in range(n)],
                  thetaPending=tP,
                  thetaBroken=[0.0 if c != 2 else 0.45*(1 - p_frac) for c in range(n)],
                  pendingStretch=[lf if c == 2 else 1.0 for c in range(n)],
                  Vs=[(0.01 + (slip_rise if c == 2 else 0), 0, 0) for c in range(n)])
        return [(1e-3, per_write(w0, n, None)), (1.025e-3, per_write(w1, n, None))]
    ws = fixture(0.5, 2.8, 0.007)
    expect("P2 passes at the fibre's own 0.9702", max(w["p2"] for _, w in ws) <= P2_LIMIT, True)
    ws_bad = fixture(1.0, 4.0, 0.007)
    expect("P2 fails on a fibre stretched past lc", max(w["p2"] for _, w in ws_bad) <= P2_LIMIT, False)
    # S17b: under V1 the fibre is read on the taut share (>= DBL_MIN) only
    def v1_write(taut):
        n = 6
        w = dict(thetaC=[0.4]*n, thetaH=[0.05]*n, be=[(1.0, 0.0, 0.0, 1.0, 0.0, 1.0)]*n,
                 thetaIntact=[0.45 if c != 2 else 0.0 for c in range(n)],
                 thetaPending=[0.45 if c == 2 else 0.0 for c in range(n)],
                 thetaBroken=[0.0]*n, pendingStretch=[4.0 if c == 2 else 1.0 for c in range(n)],
                 thetaTaut=[taut if c == 2 else 0.0 for c in range(n)])
        return per_write(w, n, None)
    expect("S17b: a slack cell (theta_T = 0) is not read by W2", v1_write(0.0)["w2"], 0.0)
    expect("S17b: a theta_T below DBL_MIN is not read by W2", v1_write(DBL_MIN/2)["w2"], 0.0)
    expect("S17b: a taut cell is read by W2 and P2", v1_write(0.45)["p2"] > P2_LIMIT, True)
    nb = [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4]]
    expect("P1 passes a reset cell", sum(w["p1_bad"] for _, w in fixture(1.0, 2.8, 0.0)), 0)
    bad_fixture = fixture(1.0, 2.8, 0.0)
    n6 = 6
    wb = dict(thetaC=[0.4]*n6, thetaH=[0.05]*n6,
              be=[(1.3, 0.1, 0.0, 0.8, 0.0, 0.96)]*n6,
              thetaIntact=[0.0]*n6, thetaPending=[0.2]*n6, thetaBroken=[0.25]*n6,
              pendingStretch=[2.0]*n6)
    expect("P1 fails a d* = 1 cell that kept its strain", per_write(wb, n6, None)["p1_bad"] > 0, True)
    expect("P3 passes a 0.007 m/s rise", p3(fixture(1.0, 2.8, 0.007), nb, 3.3e-4)[0], True)
    expect("P3 fails a 0.05 m/s rise", p3(fixture(1.0, 2.8, 0.05), nb, 3.3e-4)[0], False)

    def strain_writes(scale):
        n = 4
        be = (1.1*scale, 0.2*scale, 0.0, 1/(1.1*scale), 0.0, 1.0)
        w = {"thetaC": [0.45]*n, "thetaH": [0.0]*n, "be": [be]*n,
             "thetaIntact": [0.45]*n, "thetaPending": [0.0]*n, "thetaBroken": [0.0]*n,
             "pendingStretch": [1.0]*n}
        return [(1e-3, per_write(w, n, None))]
    expect("strain agrees with itself", strain(strain_writes(1.0), strain_writes(1.0))[0], True)
    expect("strain fails a 10% change", strain(strain_writes(1.1), strain_writes(1.0))[0], False)

    expect("zigzag of a monotone profile is 0", zigzag_share([1, 2, 3, 4, 5, 6]), 0.0)
    expect("zigzag of an alternating profile is > 0", zigzag_share([1, 3, 1, 3, 1, 3]) > 0, True)
    e = eig_sym3((2.0, 0.5, 0.0, 1.0, 0.0, 0.5))
    expect("eigenvalues sum to the trace", abs(sum(e) - 3.5) < 1e-12, True)

    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.log", Path(tmp) / "b.log"
        common = ("PHASE_CLOT_RUPTURE time=0.001 newlyRuptured=0 x=1\n"
                  "PHASE_CLOT_FOO time=0.001 v=1\n"
                  "PHASE_CLOT_RUPTURE time=0.002 newlyRuptured=3 x=1\n")
        a.write_text(common + "PHASE_CLOT_S17 time=0.001 only=1\n")
        b.write_text(common)
        expect("G-id passes identical windows", gid(a, b, None)[0], True)
        a.write_text(common.replace("v=1", "v=2"))
        expect("G-id fails a changed line", gid(a, b, None)[0], False)
    # 16b: the energy gate on a synthetic structured channel. A plane
    # Poiseuille flow driven by dp through a H x L channel dissipates exactly
    # dp*Q (steady, no-slip): the discrete measure must read it near 1 of its
    # own ceiling, below 1 for the half-strength flow, and FAIL at 20x.
    # rho, nu, the depth and h are the case's since 2026-09-30; here, the
    # synthetic case's own (the throat case's values, so every number below
    # is the one this test has always checked).
    T_RHO, T_NU, T_DEPTH, T_DX = 1050.0, 3.3333333333e-06, 1e-6, 0.7142857142857e-6
    NXs, NYs, H, L = 60, 30, 20e-6, 60*T_DX
    xs = [T_DX*(i + 0.5) for i in range(NXs)]
    ys = [H*(j + 0.5)/NYs for j in range(NYs)]
    x = [xs[i] for j in range(NYs) for i in range(NXs)]
    y = [ys[j] for j in range(NYs) for i in range(NXs)]
    V = [T_DX*(H/NYs)*T_DEPTH]*(NXs*NYs)
    m = StructuredMesh(x, y, V, T_DEPTH, T_DX)
    mu = T_RHO*T_NU
    G = 1.0e6                       # pressure gradient, Pa/m
    def prof(scale):
        return [(scale*G/(2*mu)*yy*(H - yy), 0.0, 0.0) for yy in y]
    Q = G*H**3/(12*mu)*T_DEPTH
    ceil = G*L*Q
    r1 = m.dissipation(prof(1.0), mu)/ceil
    expect("energy: steady Poiseuille dissipates ~dp*Q (within 2%)", abs(r1 - 1.0) < 0.02, True)
    expect("energy: half-strength flow passes (< 1)", m.dissipation(prof(0.5), mu)/ceil < 1.0, True)
    expect("energy: 20x flow fails (> 1)", m.dissipation(prof(20**0.5), mu)/ceil > 1.0, True)
    bad = list(x); bad[NXs + 3] += 1e-7
    try:
        StructuredMesh(bad, y, V, T_DEPTH, T_DX)
        refused = False
    except ValueError:
        refused = True
    expect("energy: a non-structured mesh is refused, not guessed", refused, True)
    try:
        StructuredMesh(x, y, V, T_DEPTH, 2*T_DX)
        refused = False
    except ValueError:
        refused = True
    expect("energy: a cell width that is not the case's h is refused", refused, True)
    dpc, qc, pc = energy_ceiling()
    expect("energy: the ceiling is read from the controls (dp Q_open 1.659e-10 W)",
           abs(pc - 1.659e-10)/1.659e-10 < 1e-3, True)

    def sf(vals):
        return ("FoamFile{}\ninternalField nonuniform List<scalar> "
                f"{len(vals)}\n(\n" + "\n".join(repr(v) for v in vals) + "\n)\n;\n")

    def vf(vals):
        return ("FoamFile{}\ninternalField nonuniform List<vector> "
                f"{len(vals)}\n(\n" + "\n".join(f"({a!r} {b!r} {c!r})" for a, b, c in vals)
                + "\n)\n;\n")

    def case_files(base, rho, nu, depth, h, H_, u_mean, pressure_driven=None):
        import json
        (base / "constant").mkdir(parents=True, exist_ok=True)
        (base / "constant" / "phaseClotProperties").write_text(
            f"rho [1 -3 0 0 0 0 0] {rho!r};\nnu [0 2 -1 0 0 0 0] {nu!r};\n")
        spec = {"geometry": {"depth_m": depth, "h_m": h, "channel_height_m": H_},
                "mean_velocity_m_s": u_mean}
        if pressure_driven is not None:
            spec["pressure_driven"] = pressure_driven
        (base / "child.json").write_text(json.dumps(spec))

    with tempfile.TemporaryDirectory() as tmp:
        # an end-to-end pass over a synthetic case directory, and an empty one
        base = Path(tmp) / "runs" / "syn"
        (base / "0").mkdir(parents=True)
        (base / "0" / "Ccx").write_text(sf(x)); (base / "0" / "Ccy").write_text(sf(y))
        (base / "0" / "Vc").write_text(sf(V))
        expect("energy: no case files is a FAIL, not a pass (rho/nu are read, never assumed)",
               energy(Path(tmp), "syn")[0], False)
        case_files(base, T_RHO, T_NU, T_DEPTH, T_DX, H, Q/(H*T_DEPTH))
        expect("energy: no written U is a FAIL, not a pass", energy(Path(tmp), "syn")[0], False)
        (base / "1e-05").mkdir(); (base / "1e-05" / "U").write_text(vf(prof(1e-3)))
        (base / "2e-05").mkdir(); (base / "2e-05" / "U").write_text(vf(prof(1e-2)))
        expect("energy: a physical case passes end to end", energy(Path(tmp), "syn")[0], True)
        (base / "3e-05").mkdir(); (base / "3e-05" / "U").write_text(vf(prof(100.0)))
        expect("energy: one write above the ceiling fails the run", energy(Path(tmp), "syn")[0], False)

    # A5 (the Xu velocity inlet): a 3 x 2 mesh whose dissipation is computed
    # BY HAND here. h = depth = 1 um, rho 1000, nu 1e-6 (mu 1e-3). Rows of U:
    # (1, 0) and (3, 0). x faces: dU = 0. Three y faces: dU^2 = 4, A = 1e-12,
    # d = 1e-6 -> 4e-6 each, 1.2e-5. Walls: A/(d/2) = 2e-6 x (3 x 1 + 3 x 9) =
    # 6e-5. Phi = 1e-3 x 7.2e-5 = 7.2e-8 W. Q = U_mean H depth = 2 x 2e-6 x
    # 1e-6 = 4e-12 m3/s, so Q rho p_in = 4e-9 x p_in (kinematic, column 0).
    hx = 1e-6
    x5 = [hx*(i + 0.5) for j in range(2) for i in range(3)]
    y5 = [hx*(j + 0.5) for j in range(2) for i in range(3)]
    V5 = [hx*hx*hx]*6
    U5 = [(1.0, 0.0, 0.0)]*3 + [(3.0, 0.0, 0.0)]*3
    PHI5 = 7.2e-8

    def a5_case(tmp, writes, theta=0.0, nu=1e-6, pressure_driven=False, drop=None):
        base = Path(tmp) / "runs" / "a5"
        if base.exists():
            shutil.rmtree(base)
        (base / "0").mkdir(parents=True)
        (base / "0" / "Ccx").write_text(sf(x5)); (base / "0" / "Ccy").write_text(sf(y5))
        (base / "0" / "Vc").write_text(sf(V5))
        case_files(base, 1000.0, nu, hx, hx, 2*hx, 2.0, pressure_driven)
        for tv, p_in in writes:
            d = base / f"{tv:g}"
            d.mkdir()
            d_fields = {"U": vf(U5), "p": sf([p_in, 0.0, 0.0, p_in, 0.0, 0.0]),
                        "thetaC": sf([theta]*6), "thetaH": sf([0.0]*6)}
            for name, body in d_fields.items():
                if name != drop:
                    (d / name).write_text(body)
        return Path(tmp)

    def p_for(R):
        return PHI5/(R*4e-9)

    with tempfile.TemporaryDirectory() as tmp:
        ok, _m, r = energy_inlet(a5_case(tmp, [(1e-5, p_for(0.95))]), "a5")
        expect("A5: the hand-computed Phi/(Q rho p_in) = 0.95 reads 0.95 and PASSES",
               (ok, round(r["first"], 12)), (True, 0.95))
        ok, _m, r = energy_inlet(a5_case(tmp, [(1e-5, p_for(1.0))]), "a5")
        expect("A5: R = 1.0 FAILS (> 0.9915)", ok, False)
        ok, _m, r = energy_inlet(a5_case(tmp, [(1e-5, p_for(1.0))], theta=0.45), "a5")
        expect("A5: the phase weight acts: theta 0.45 everywhere gives R = 0.55 x 1.0",
               (ok, round(r["first"], 12)), (True, 0.55))
        ok, _m, r = energy_inlet(a5_case(tmp, [(1e-5, p_for(0.95))], nu=2e-6), "a5")
        expect("A5: nu is READ from the case: doubling it doubles R and FAILS",
               (ok, round(r["first"], 12)), (False, 1.9))
        ok, _m, r = energy_inlet(a5_case(tmp, [(1e-5, p_for(0.95)), (2e-5, p_for(0.95)/2)]), "a5")
        expect("A5: the integral is CUMULATIVE: 0.95 then an instantaneous 1.9 gives "
               "(1.5 Phi)/(1.25 P) x 0.95 = 1.14 and FAILS", (ok, round(r["max"], 12)), (False, 1.14))
        # PINNED (verification 08, M6 and M12; DECISIONS 2026-09-30): the verdict is on the
        # CUMULATIVE R at EVERY write. An instantaneous 1.2 after 0.5 is cumulative
        # (1.5 Phi)/(P1 + P2/2) = 1.5/(2 + 0.4167) = 0.6207 and PASSES (an instantaneous verdict
        # would fail it); an instantaneous 1.2 then 0.3 is cumulative 1.2 at the FIRST write and
        # FAILS (a last-write verdict, cumulative 0.6 there, would pass it).
        ok, _m, r = energy_inlet(a5_case(tmp, [(1e-5, p_for(0.5)), (2e-5, p_for(1.2))]), "a5")
        expect("A5: the verdict is cumulative, not instantaneous: 0.5 then an instantaneous 1.2 "
               "is cumulative 0.6207 and PASSES", (ok, round(r["max"], 4)), (True, 0.6207))
        ok, _m, r = energy_inlet(a5_case(tmp, [(1e-5, p_for(1.2)), (2e-5, p_for(0.3))]), "a5")
        expect("A5: every write is scored, not the last: 1.2 then 0.3 FAILS at the first write "
               "(cumulative 1.2; last 0.6)", (ok, round(r["max"], 4), round(r["last"], 4)),
               (False, 1.2, 0.6))
        ok, _m, r = energy_inlet(a5_case(tmp, [(1e-5, p_for(0.95))], drop="thetaH"), "a5")
        expect("A5: a missing field is UNREADABLE, not a pass", (ok, r.get("unreadable")), (False, True))
        ok, _m, r = energy_inlet(a5_case(tmp, [(1e-5, p_for(0.95))], pressure_driven=True), "a5")
        expect("A5: a pressure-driven child is refused (the bound is the velocity inlet's)",
               (ok, r.get("unreadable")), (False, True))

    # S17b: gid-physical and em74-sign on synthetic logs
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "em74.log", Path(tmp) / "em66.log"
        base = ("PHASE_CLOT_DIAGNOSTIC time=1e-06 inventory=1 outletMeanUx=0.1\n"
                "PHASE_CLOT_S16 time=1e-06 r=0.1 muMin=0.3 partitionSumDevAbs=1e-20\n"
                "PHASE_CLOT_S17 time=1e-06 transfers=0 w2Max=0\n"
                "PHASE_CLOT_S17 time=2e-06 transfers=2 w2Max=0\n"
                "PHASE_CLOT_DIAGNOSTIC time=2e-06 inventory=9 outletMeanUx=0.2\n")
        b.write_text(base)
        a.write_text(base.replace("muMin=0.3", "muMin=0.7").replace(
            "partitionSumDevAbs=1e-20", "partitionSumDevAbs=3e-19")
            + "PHASE_CLOT_S17B time=1e-06 slackEvents=0\n")
        expect("gid-physical passes when only round-off keys differ", gid_physical(a, b, None)[0], True)
        a.write_text(base.replace("inventory=1 ", "inventory=1.0000001 "))
        expect("gid-physical fails on a physical difference", gid_physical(a, b, None)[0], False)
        a.write_text(base.replace("inventory=9 ", "inventory=8 "))
        expect("gid-physical stops at the first transfer", gid_physical(a, b, None)[0], True)
    def g8(slip, proj, t, wall=1):
        return (f"PHASE_CLOT_S17B_G8CELL time={t} slip={slip} wallRow={wall} "
                f"wallFaceProj={proj}\n")
    l74 = "".join(g8(0.02, 1e-3, k*2.5e-5) for k in range(1, 11))
    l71 = "".join(g8(0.015, 0.0, k*2.5e-5) for k in range(1, 11))
    expect("em74-sign passes when A3 lowers G8 where the wall face aided the slip",
           em74_sign(l74, l71)[0], True)
    l71b = "".join(g8(0.025, 0.0, k*2.5e-5) for k in range(1, 11))
    expect("em74-sign fails when the sign disagrees everywhere", em74_sign(l74, l71b)[0], False)
    expect("em74-sign is not evaluable without a wall-row G8 cell",
           em74_sign(l74.replace("wallRow=1", "wallRow=0"), l71)[0], None)

    print("ALL PASS" if ok_all else "FAILED")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gid"); g.add_argument("child"); g.add_argument("parent")
    g.add_argument("--cutoff", type=float)
    d = sub.add_parser("discriminator"); d.add_argument("log"); d.add_argument("name")
    f = sub.add_parser("fields"); f.add_argument("src"); f.add_argument("child")
    f.add_argument("--ref"); f.add_argument("--tau", type=float, default=3.3e-4)
    e = sub.add_parser("energy"); e.add_argument("src"); e.add_argument("child")
    e.add_argument("--controls", default=str(CONTROLS))
    ei = sub.add_parser("energy-inlet"); ei.add_argument("src"); ei.add_argument("child")
    er = sub.add_parser("energy-real-check"); er.add_argument("archive")
    gp = sub.add_parser("gid-physical"); gp.add_argument("child"); gp.add_argument("parent")
    gp.add_argument("--cutoff", type=float)
    es = sub.add_parser("em74-sign"); es.add_argument("em74"); es.add_argument("em71")
    es.add_argument("--write", type=float, default=2.5e-5)
    sub.add_parser("self-test")
    a = ap.parse_args()
    try:
        if a.cmd == "gid":
            ok, msg = gid(Path(a.child), Path(a.parent), a.cutoff)
        elif a.cmd == "discriminator":
            ok, msg = discriminator(Path(a.log).read_text(errors="replace"), a.name)
        elif a.cmd == "fields":
            ok, msg = fields(Path(a.src), a.child, a.ref, a.tau)
        elif a.cmd == "energy":
            ok, msg, _ = energy(Path(a.src), a.child, Path(a.controls))
        elif a.cmd == "energy-inlet":
            ok, msg, r = energy_inlet(Path(a.src), a.child)
            if r.get("unreadable"):
                print(msg)
                return 2
        elif a.cmd == "gid-physical":
            ok, msg = gid_physical(Path(a.child), Path(a.parent), a.cutoff)
        elif a.cmd == "em74-sign":
            ok, msg = em74_sign(Path(a.em74), Path(a.em71), a.write)
            ok = bool(ok)
        elif a.cmd == "energy-real-check":
            ok, msg = energy_real_check(Path(a.archive))
        else:
            return 0 if self_test() else 1
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"unreadable input: {error}")
        return 2
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
