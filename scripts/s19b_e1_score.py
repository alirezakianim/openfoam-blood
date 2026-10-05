#!/usr/bin/env python3
"""S19b E1, scored OFFLINE: the elimination is exact in the full solver.

research/designs/2026-10-04-S19b-lumen-elimination.md rev. 2, APPROVED by
Alireza 2026-10-04 (docket research/investigations/20261004-1803-s19b-lumen-
elimination/), section 4. fsb_k13 (blockCoupledLumen eliminate, the default)
against fsb_k13_id (identityRows, the control), two returned child cases.

  VOID          C1 fails on the control (assess_case.s19_report): E1 is not
                scored, never passed.
  INCONCLUSIVE  (1) the precondition fails: the control is not SMOOTH (G2's
                flip fraction, passed in, > 0.10 or not given), or the lumen
                count differs at any step (lumenEliminated of fsb_k13 against
                lumenIdentityRows of fsb_k13_id, F5), or the two children do
                not log the same steps; or (4) step 1 passes, a write misses,
                and the per-step growth is smooth and monotone (reported, not
                failed).
  FAIL          (5) every step's delta is EXACTLY 0 for uMax, vsMax and pMax
                (both children ran the same path, F3); (2) step 1's
                |delta|/value > 1e-12 for any of uMax, vsMax, pMax (17-digit
                PHASE_CLOT_S19 fields); (3) a write miss without smooth growth.
  PASS          step 1 <= 1e-12 and, at the 0.2 and 0.4 ms writes, max_cells
                |delta| / max_cells |field| <= 1e-8 for U, Vs and p.

"Smooth, monotone growth" (design 4.4 names it, gives no number; THIS
SCORER'S CHOICE, stated here before any run): over the steps after the first
non-zero per-step delta d_k (the max over uMax, vsMax, pMax of |delta|/value),
no step exceeds 10 x the largest earlier d (no single-step jump of a decade),
and the last step's running maximum exceeds d at step 1 (it grew).

Usage:
  s19b_e1_score.py --eliminate <case fsb_k13> --identity <case fsb_k13_id>
                   --control-flip-fraction <G2 flip fraction of fsb_k13_id>
  s19b_e1_score.py --self-test
Exit 0 on PASS (and on a passing self-test), 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSESS = ROOT / "cases/AM2020_planar_mural_detachment/scripts/assess_case.py"

E1_STEP1_MAX = 1e-12      # design 4.2
E1_WRITE_MAX = 1e-8       # design 4.3
E1_WRITES_S = (2e-4, 4e-4)
E1_SMOOTH_MAX = 0.10      # design 4.1 (G2's SMOOTH)
E1_JUMP = 10.0            # this scorer's smoothness test (module docstring)
KEYS = ("uMax", "vsMax", "pMax")
NUM = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"


def _assess():
    spec = importlib.util.spec_from_file_location("assess_case_s19b_e1", ASSESS)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ASSESS.parent))
    spec.loader.exec_module(mod)
    return mod


def records(case: Path) -> dict[int, dict]:
    """PHASE_CLOT_S19 records by step (the last record of a step wins)."""
    text = (case / "log.phaseClotAxisymmetricFluid").read_text(errors="replace")
    out = {}
    for ln in re.findall(r"^PHASE_CLOT_S19 (.*)$", text, re.M):
        r = {k: float(v) for k, v in re.findall(r"(\w+)=" + NUM, ln)}
        if "step" in r:
            out[int(r["step"])] = r
    return out


def rel(a: float, b: float) -> float:
    """|a - b| / |b|; the denominator guarded: 0 when both are 0, inf when
    only b is."""
    if a == b:
        return 0.0
    return abs(a - b)/abs(b) if b != 0 else math.inf


def time_dir(case: Path, t: float) -> Path | None:
    for d in case.iterdir():
        try:
            v = float(d.name)
        except ValueError:
            continue
        if d.is_dir() and abs(v - t) <= 1e-9*t:
            return d
    return None


def field_delta(a_case: Path, b_case: Path, t: float, name: str, ac) -> float | None:
    """max_cells |a - b| / max_cells |b| (vector magnitude for U and Vs)."""
    da, db = time_dir(a_case, t), time_dir(b_case, t)
    if da is None or db is None or not (da / name).is_file() or not (db / name).is_file():
        return None
    if name == "p":
        fa, fb = ac.read_scalars(da / name), ac.read_scalars(db / name)
        fa, fb = [(x,) for x in fa], [(x,) for x in fb]
    else:
        fa, fb = ac._xu_read_tuples(da / name), ac._xu_read_tuples(db / name)
    if len(fa) != len(fb):
        return math.inf
    d = max(math.sqrt(sum((x - y)**2 for x, y in zip(u, v))) for u, v in zip(fa, fb))
    s = max(math.sqrt(sum(y*y for y in v)) for v in fb)
    return d/s if s > 0 else (0.0 if d == 0 else math.inf)


def score(elim: Path, ident: Path, flip: float | None) -> dict:
    ac = _assess()
    out: dict = {"eliminate": str(elim), "identity": str(ident), "control_flip_fraction": flip}
    spec = json.loads((ident / "child.json").read_text())
    text = (ident / "log.phaseClotAxisymmetricFluid").read_text(errors="replace")
    rep = ac.s19_report(text, spec, ident)
    out["c1"] = rep.get("c1")
    if rep.get("c1") is not True:
        out["verdict"] = "VOID"
        out["why"] = f"C1 on the control is {rep.get('c1')} (E1 is not scored)"
        return out
    ra, rb = records(elim), records(ident)
    steps = sorted(rb)
    why = []
    if sorted(ra) != steps or not steps:
        why.append(f"the children log different steps ({len(ra)} vs {len(rb)} records)")
    if flip is None or not flip <= E1_SMOOTH_MAX:
        why.append(f"the control is not established SMOOTH (flip fraction {flip}, needs <= {E1_SMOOTH_MAX})")
    flips = [k for k in steps if k in ra
             and ra[k].get("lumenEliminated", -1.0) != rb[k].get("lumenIdentityRows", -2.0)]
    out["lumen_count_mismatch_steps"] = flips[:5]
    if flips:
        why.append(f"lumen count differs at {len(flips)} step(s) (a mask flip, first step {flips[0]})")
    if why:
        out["verdict"] = "INCONCLUSIVE"
        out["why"] = "; ".join(why)
        return out
    per = []
    for k in steps:
        ds = [rel(ra[k].get(q, math.nan), rb[k].get(q, math.nan)) for q in KEYS]
        per.append(max(math.inf if math.isnan(x) else x for x in ds))
    out["d_step1"] = per[0]
    out["d_last"] = per[-1]
    out["d_max"] = max(per)
    out["steps"] = len(per)
    if all(d == 0.0 for d in per):
        out["verdict"] = "FAIL"
        out["why"] = "delta exactly 0 at every step: both children ran the same path (F3)"
        return out
    if not per[0] <= E1_STEP1_MAX:
        out["verdict"] = "FAIL"
        out["why"] = f"step 1 |delta|/value {per[0]:.3e} > {E1_STEP1_MAX:g}"
        return out
    writes = {}
    for t in E1_WRITES_S:
        for name in ("U", "Vs", "p"):
            writes[f"{name}@{t:g}"] = field_delta(elim, ident, t, name, ac)
    out["writes"] = writes
    missing = [k for k, v in writes.items() if v is None]
    if missing:
        out["verdict"] = "INCONCLUSIVE"
        out["why"] = f"write(s) absent: {', '.join(missing)}"
        return out
    miss = {k: v for k, v in writes.items() if not v <= E1_WRITE_MAX}
    if not miss:
        out["verdict"] = "PASS"
        out["why"] = f"step 1 {per[0]:.3e} <= {E1_STEP1_MAX:g}; writes max {max(writes.values()):.3e} <= {E1_WRITE_MAX:g}"
        return out
    first = next(i for i, d in enumerate(per) if d > 0)
    env, jump = per[first], False
    for d in per[first + 1:]:
        if d > E1_JUMP*env:
            jump = True
        env = max(env, d)
    grew = env > per[0]
    out["smooth_monotone_growth"] = bool(not jump and grew)
    if not jump and grew:
        out["verdict"] = "INCONCLUSIVE"
        out["why"] = (f"step 1 passes ({per[0]:.3e}); writes miss ({', '.join(miss)}) with smooth, "
                      f"monotone per-step growth to {per[-1]:.3e}: round-off growth, reported not failed")
    else:
        out["verdict"] = "FAIL"
        out["why"] = (f"writes miss ({', '.join(f'{k} {v:.3e}' for k, v in miss.items())}) "
                      f"{'with a single-step jump' if jump else 'without growth'}")
    return out


# ---------------------------------------------------------------- self-test
def _write_case(d: Path, lumen: str, rows: list[dict], fields: dict, n_cells: int = 3) -> None:
    (d / "constant/polyMesh").mkdir(parents=True)
    (d / "constant/polyMesh/owner").write_text(
        f'FoamFile\n{{\n    note        "nPoints: 8 nCells: {n_cells} nFaces: 10 nInternalFaces: 2";\n}}\n')
    s19 = {"momentumPressureCoupling": "blockCoupled", "networkViscosity": "none",
           "networkViscosityTau": 2.5e-4}
    if lumen == "identityRows":
        s19["blockCoupledLumen"] = "identityRows"
    (d / "child.json").write_text(json.dumps({"child_id": d.name, "s19": s19}))
    lines = ["PHASE_CLOT_S19_CONFIG momentumPressureCoupling=blockCoupled networkViscosity=none "
             f"blockCoupledLumen={lumen}"]
    for k, r in enumerate(rows):
        lines += [f"Time = {k + 1}e-06", ""]
        lines.append("PHASE_CLOT_S19 " + " ".join(f"{a}={b!r}" if isinstance(b, float) else f"{a}={b}"
                                                for a, b in dict(step=k + 1, mode="blockCoupled", **r).items()))
    (d / "log.phaseClotAxisymmetricFluid").write_text("\n".join(lines) + "\nEnd\n")
    for t, flds in fields.items():
        td = d / t
        td.mkdir()
        for name, vals in flds.items():
            if name == "p":
                body = "\n".join(repr(v) for v in vals)
                kind = "scalar"
            else:
                body = "\n".join(f"({v[0]!r} {v[1]!r} 0)" for v in vals)
                kind = "vector"
            (td / name).write_text(f"FoamFile\n{{\n}}\ninternalField   nonuniform List<{kind}> "
                                   f"{len(vals)}\n(\n{body}\n)\n;\n")


def _pair(tmp: Path, tag: str, step_d: list[float], write_d: float, ident_ok: bool = True) -> tuple[Path, Path]:
    base = {"uMax": 0.0123456789012345, "vsMax": 0.00987654321098765, "pMax": 0.0737066}
    rb, ra = [], []
    for d in step_d:
        rb.append({"n": 15, "lumenIdentityRows": 2 if ident_ok else 0, "lumenEliminated": 0, **base})
        ra.append({"n": 11, "lumenIdentityRows": 0, "lumenEliminated": 2,
                   **{k: v*(1 + d) for k, v in base.items()}})
    fb = {"U": [(1e-3, 2e-4), (2e-3, -1e-4), (5e-4, 0.0)], "Vs": [(1e-3, 2e-4), (1e-3, 0.0), (5e-4, 0.0)],
          "p": [0.07, 0.03, 0.0]}
    fa = {"U": [(x*(1 + write_d), y) for x, y in fb["U"]], "Vs": [(x*(1 + write_d), y) for x, y in fb["Vs"]],
          "p": [v*(1 + write_d) for v in fb["p"]]}
    a, b = tmp / f"{tag}_fsb_k13", tmp / f"{tag}_fsb_k13_id"
    _write_case(a, "eliminate", ra, {"0.0002": fa, "0.0004": fa})
    _write_case(b, "identityRows", rb, {"0.0002": fb, "0.0004": fb})
    return a, b


def self_test() -> int:
    status = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cases = [
            ("pass: round-off at every step, writes 1e-12", [3e-15, 5e-15, 4e-15, 8e-15], 1e-12, True, 0.02, "PASS"),
            ("delta exactly 0 at every step", [0.0, 0.0, 0.0, 0.0], 0.0, True, 0.02, "FAIL"),
            ("step 1 misses (a p-row remap defect, 4.9e-8)", [4.9e-8, 5e-8, 6e-8, 7e-8], 1e-7, True, 0.02, "FAIL"),
            ("smooth growth to a write miss", [1e-14 * 3.0**k for k in range(20)], 2e-7, True, 0.02, "INCONCLUSIVE"),
            ("a single-step jump to a write miss", [1e-14, 2e-14, 3e-14, 1e-6], 2e-6, True, 0.02, "FAIL"),
            ("the control not SMOOTH (flip fraction 0.5)", [3e-15, 5e-15], 1e-12, True, 0.5, "INCONCLUSIVE"),
            ("C1 fails on the control (no identity rows)", [3e-15, 5e-15], 1e-12, False, 0.02, "VOID"),
        ]
        for k, (name, step_d, write_d, ident_ok, flip, want) in enumerate(cases):
            a, b = _pair(tmp, f"c{k}", step_d, write_d, ident_ok)
            r = score(a, b, flip)
            ok = r["verdict"] == want
            print(f"{'OK  ' if ok else 'FAIL'} E1 self-test '{name}' -> {r['verdict']} (want {want}): {r.get('why')}")
            if not ok:
                status = 1
    print("E1 SELF-TEST", "PASSED" if status == 0 else "FAILED")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eliminate", type=Path)
    ap.add_argument("--identity", type=Path)
    ap.add_argument("--control-flip-fraction", type=float, default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.eliminate is None or a.identity is None:
        ap.error("--eliminate and --identity are required")
    r = score(a.eliminate, a.identity, a.control_flip_fraction)
    print(json.dumps(r, indent=1, default=str))
    print("E1", r["verdict"])
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
