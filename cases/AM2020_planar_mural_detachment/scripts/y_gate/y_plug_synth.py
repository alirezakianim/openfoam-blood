#!/usr/bin/env python3
"""SYNTHETIC inputs of the plug assessor (offline; pure python, standard library only). NOT physics: deterministic fields and traces laid out the way the solver writes them,
so that scripts/check_y_plug_assessor.py can show every verdict, every FAIL item and the chain merge FIRE and stay SILENT on data whose answer is known.

  traces      model_trace / rt23_trace / constant_trace -> per-write records (the analyse() input) from a V_b law, with the M+- the plug's kinematics gives
              (physics 24: M+ 0.0059 D0^2 at contact, +0.32 D0 per D0 of shift; M- +0.20 D0 per D0 after contact), quantised in cell quanta
  fields      plug_r(mesh, ...) the C1 plug's theta/phi0 on the REAL f = 2 mesh (the fixture's cell centres) under a rigid shift; write_session() lays a returned case out
              (0/C, 0/Vc, 0/p, the writes, uniform/time, restart.json, postProcessing/inletFlux, the dictionaries copied from a generated Y81 child)
The p24 constants below are physics 24's (probes/p24_model.py); they are used ONLY to build test traces and are re-derived against p24_model.txt by the check.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import y_gate_report as G  # noqa: E402
import y_plug  # noqa: E402
import y_synth  # noqa: E402

# ---- physics 24's lumped model (probes/p24_model.py); sim units, S_L 5/3, S_t 25/3
MU = 1050 * 6.7636e-7
D0 = 58.333e-6
UB = 1.2536e-3
GPA = 11.98
TV = 12.3e-3
P_IN = {"YS794": 1.341158e-3 * 1050, "YS850": 1.220658e-3 * 1050}
LW = {"YS794": {"lo": (1.167, 1.080), "cap": (1.405, 1.318), "flat": (1.544, 1.458)},
      "YS850": {"lo": (1.134, 1.044), "cap": (1.373, 1.283), "flat": (1.516, 1.426)}}
DELTA = {"sqrt k": math.sqrt(9.7222e-14), "mesh": 0.648 * 1.5e-6}
M_PLUS_CONTACT = 0.0059            # D0^2, physics 24 (p24_plug.txt, s = 0.10)
DMP_DS, DMM_DS = 0.32, 0.20         # D0 per D0 of shift after contact
QUANTUM = {"YS794": 4.95e-4, "YS850": 5.4e-4}   # D0^2, one cell of bulk content (numerics 25 section 2)


def z_film(lw: float, delta: float) -> float:
    return 2.0 * lw * D0 * MU / delta


def model_numbers(tag: str, film: str, lw: str, kappa: float) -> dict:
    """V0, t_c, tau (physics 24's STUCK law), x*, V_c for one geometry / film / L_w branch / kappa."""
    p, delta = P_IN[tag], DELTA[film]
    ls, lc = LW[tag][lw]
    zn = p * D0 / UB
    v0 = p * D0 / (z_film(0.5 * (ls + lc), delta) + zn)
    tau = (z_film(lc, delta) + zn) / (kappa * GPA) + TV
    xstar = p / GPA * D0 / kappa
    return {"V0": v0, "t_c": 0.10 * D0 / v0, "tau": tau, "x_star": xstar, "V_c": xstar / tau, "lam": p / GPA}


def rt23_numbers(p_in_label: str, film: str, lw: float) -> dict:
    """Red team 23's model traces: V0 = pD0/(zeta + zn) from the contact start, tau_rt = zeta/G + tau_v (the network term is missing from tau)."""
    p = {"p_in 1.2829": 1.2829, "x1.098": 1.2829 * 1.098}[p_in_label]
    delta = {"sqrt k": 0.312e-6, "mesh f2": 0.972e-6}[film]
    zz = z_film(lw, delta)
    v0 = p * D0 / (zz + p * D0 / UB)
    return {"V0": v0, "tau": zz / GPA + TV, "t_c": 0.10 * D0 / v0, "lam": p / GPA}


# ---------------------------------------------------------------------------------------------------- traces -> records
def _quant(v: float, q: float) -> float:
    return math.floor(v / q + 1e-9) * q


def records_from_displacement(times, disp_m, tag: str, x0_m=-0.7 * D0, m_plus=None, m_minus=None, quantum=None, noise_x=0.0, rng=None, extra=None):
    """Records (the analyse() input) from a displacement-in-m trace. M+- default to the plug's kinematics at the displacement (D0 units), quantised."""
    q = QUANTUM[tag] if quantum is None else quantum
    out = []
    for i, (t, s) in enumerate(zip(times, disp_m)):
        u = s / D0
        if m_plus is None:
            mp = M_PLUS_CONTACT * (u / 0.10) if u <= 0.10 else M_PLUS_CONTACT + DMP_DS * (u - 0.10)
        else:
            mp = m_plus[i]
        if m_minus is None:
            mm = 0.0 if u <= 0.10 else DMM_DS * (u - 0.10)
        else:
            mm = m_minus[i]
        x = x0_m + s + (rng.gauss(0.0, noise_x) if noise_x else 0.0)
        if i == 0:
            x = x0_m                                           # the initial state is exact
        r = {"t": t, "X": x, "A": 4.37e-2, "A_bulk": 4.0e-2, "Mp": _quant(mp, q) if q else mp, "Mm": _quant(mm, q) if q else mm, "content": 2.34e-9, "r_max": 1.0,
             "n_over": 0, "Dg_max": 0.0, "lam1_bulk_max": 1.2, "Q": None, "Qs": None, "slab_ok": None}
        if i == 0:
            r["initial"] = True
        if extra:
            r.update(extra(i, t, s))
        out.append(r)
    return out


def displacement_from_rate(times, rate_fn):
    """Integral of rate_fn(t) (m/s) by the exact trapezoid of a fine sub-grid (the analytic traces below are smooth, so 50 sub-steps are exact to 1e-9 relative)."""
    out, s, prev = [0.0], 0.0, times[0]
    for t in times[1:]:
        n = 50
        h = (t - prev) / n
        s += sum(0.5 * h * (rate_fn(prev + k * h) + rate_fn(prev + (k + 1) * h)) for k in range(n))
        out.append(s)
        prev = t
    return out


def model_displacement(m: dict):
    """The closed form of model_rate's integral: V0 t until contact, then V0 t_c + V_c tau (1 - exp(-(t - t_c)/tau))."""
    tc, tau, vc, v0 = m["t_c"], m["tau"], m["V_c"], m["V0"]
    return lambda t: v0 * t if t < tc else v0 * tc + vc * tau * (1.0 - math.exp(-(t - tc) / tau))


def times_grid(t_end=0.185, dt=1e-3):
    n = int(round(t_end / dt))
    return [k * dt for k in range(n + 1)]


def model_rate(m: dict):
    """Physics 24's STUCK law: V0 until contact, then V_c exp(-(t - t_c)/tau)."""
    tc, tau, vc, v0 = m["t_c"], m["tau"], m["V_c"], m["V0"]
    return lambda t: v0 if t < tc else vc * math.exp(-(t - tc) / tau)


def rt23_rate(m: dict):
    """Red team 23's trace: V0 until contact, then V0 exp(-(t - t_c)/tau_rt)."""
    tc, tau, v0 = m["t_c"], m["tau"], m["V0"]
    return lambda t: v0 if t < tc else v0 * math.exp(-(t - tc) / tau)


def facts_for(tag: str) -> dict:
    return {"ubar": 0.0012533333333333333, "D0": D0, "lam": P_IN[tag] / GPA, "tau_v": 0.012283333333333335, "gap_m": 0.10 * D0, "write_interval": 1e-3,
            "series": {"YS794": "s25_ys794_plug", "YS850": "s25_ys850_plug"}[tag],
            "q0": 0.0012533333333333333 * (1 + {"YS794": 0.00116, "YS850": 0.00115}[tag]) * D0 * 2e-6, "q0_tol": {"YS794": 0.00446, "YS850": 0.00474}[tag]}


# ---------------------------------------------------------------------------------------------------- fields
class Mesh:
    """The f = 2 replica mesh of a tag: cell centres (m), cell areas (m2), the depth."""

    def __init__(self, tag: str, spec: dict):
        ext = spec["y_gate"]
        cells, _faces = y_synth.fixtures(ext)
        self.tag = tag
        self.x = [c[0] for c in cells]
        self.y = [c[1] for c in cells]
        n, _nc, info = G.read_npy_f8(y_synth.FIX / f"{tag}_cellinfo_2d.npy")
        self.area = [info[2 * i] for i in range(n)]
        self.n = n
        self.depth = ext["geometry_sim"]["depth_m"]
        self.D0 = ext["geometry_sim"]["D0_m"]
        plug = spec["clot"]
        self.p = {k: plug[k] for k in y_plug.PLUG_KEYS}
        anchors = y_plug.dict_anchors((HERE.parents[1] / ext["mesh"]["blockMeshDict"]).read_text())
        self.xt, self.tip = anchors["tangent_x_m"] / self.D0, anchors["tip_x_m"] / self.D0
        self.eps = plug["interface_width_m"] / self.D0


def plug_r(mesh: Mesh, shift: float = 0.0, p_over=None):
    """theta/phi0 of the C1 plug under a RIGID shift (D0), as the initialiser draws it at shift 0: r = (1 - tanh(sd/eps))/2, the blend zone moving with the plug."""
    p = dict(mesh.p)
    p["gap_D0"] = p["gap_D0"] - shift
    if p_over:
        p.update(p_over)
    xt = mesh.xt + shift
    out = []
    inv = 1.0 / mesh.D0
    for x, y in zip(mesh.x, mesh.y):
        sd = y_plug.signed_distance_D0(x * inv, y * inv, p, xt, mesh.tip, mesh.eps)
        out.append(0.5 * (1.0 - math.tanh(sd / mesh.eps)))
    return out


FOOTER = "// ************************************************************************* //\n"


def _fmt(v: float) -> str:
    return f"{v:.17g}"


def field_text(cls: str, name: str, internal: str) -> str:
    return (f"FoamFile\n{{\n    format ascii;\n    class {cls};\n    location \"x\";\n    object {name};\n}}\n\ndimensions [0 0 0 0 0 0 0];\n\ninternalField {internal}\n\n"
            f"boundaryField\n{{\n    inlet {{ type fixedValue; value uniform 0; }}\n    outlet {{ type fixedValue; value uniform 0; }}\n    wall {{ type zeroGradient; }}\n}}\n\n{FOOTER}")


def scalar_internal(vals) -> str:
    return f"nonuniform List<scalar>\n{len(vals)}\n(\n" + "\n".join(_fmt(v) for v in vals) + "\n)\n;"


def vector_internal(vals) -> str:
    return f"nonuniform List<vector>\n{len(vals)}\n(\n" + "\n".join("(" + " ".join(_fmt(c) for c in v) + ")" for v in vals) + "\n)\n;"


def tensor_internal(vals) -> str:
    return f"nonuniform List<symmTensor>\n{len(vals)}\n(\n" + "\n".join("(" + " ".join(_fmt(c) for c in v) + ")" for v in vals) + "\n)\n;"


def time_name(t: float) -> str:
    return f"{t:.10g}"


def write_dir(d: Path, mesh: Mesh, t: float, index: int, r, phi0: float = 0.45, dg=None, us=None, be=None, theta_c_frac: float = 1.0, uniform_t: float = None, q=None, gfield=None) -> None:
    """One full write: thetaC/thetaH from r (theta = r*phi0), Dg, Us, be (nonuniform only when given, else uniform), and the members restart_driver.classify and the
    assessor require (w, phiSkel, U, p, phi, uniform/time) as one-line uniform fields."""
    d.mkdir(parents=True, exist_ok=True)
    theta = [v * phi0 for v in r]
    (d / "thetaC").write_text(field_text("volScalarField", "thetaC", scalar_internal([theta_c_frac * v for v in theta])))
    (d / "thetaH").write_text(field_text("volScalarField", "thetaH", scalar_internal([(1.0 - theta_c_frac) * v for v in theta])))
    (d / "Dg").write_text(field_text("volScalarField", "Dg", scalar_internal(dg) if dg is not None else "uniform 0;"))
    (d / "Us").write_text(field_text("volVectorField", "Us", vector_internal(us) if us is not None else "uniform (0 0 0);"))
    (d / "be").write_text(field_text("volSymmTensorField", "be", tensor_internal(be) if be is not None else "uniform (1 0 0 1 0 1);"))
    if q is not None:                                                            # the solver's written q and Gfield (the tail localisation reads them); absent unless given
        (d / "q").write_text(field_text("volScalarField", "q", scalar_internal(q)))
    if gfield is not None:
        (d / "Gfield").write_text(field_text("volScalarField", "Gfield", scalar_internal(gfield) if not isinstance(gfield, (int, float)) else f"uniform {_fmt(gfield)};"))
    for name, cls, val in (("w", "volScalarField", "uniform 1;"), ("phiSkel", "surfaceScalarField", "uniform 0;"), ("U", "volVectorField", "uniform (0 0 0);"),
                           ("p", "volScalarField", "uniform 0;"), ("phi", "surfaceScalarField", "uniform 0;")):
        (d / name).write_text(field_text(cls, name, val))
    (d / "uniform").mkdir(exist_ok=True)
    (d / "uniform" / "time").write_text(f"FoamFile\n{{\n    format ascii;\n    class dictionary;\n    object time;\n}}\n\nvalue {_fmt(t)};\nname \"{_fmt(t)}\";\nindex {index};\ndeltaT 5e-06;\n\n{FOOTER}")


COPY_FROM_CHILD = ("child.json", "system/blockMeshDict", "system/controlDict", "system/fvSchemes", "system/fvSolution", "constant/phaseClotProperties", "constant/physicalProperties",
                   "constant/momentumTransport", "constant/yGate.json", "0/p")


def write_session(case: Path, tpl: Path, mesh: Mesh, writes: list, with_zero: bool = True, restart: dict = None, flux: list = None, zero_r=None, phi0: float = 0.45,
                  flux_start: str = "0", log_text: str = "End\n") -> Path:
    """A returned case. tpl = a GENERATED Y81 child directory (its dictionaries, child.json and 0/p are copied). writes = [{'t', 'r', 'index', and optional 'dg', 'us', 'be',
    'theta_c_frac'}], ascending. zero_r = the initial theta/phi0 (default plug_r at shift 0). restart = the restart.json dict (sessions after the first). flux = [(t, inletFlux)]."""
    case = Path(case)
    case.mkdir(parents=True, exist_ok=True)
    for rel in COPY_FROM_CHILD:
        dst = case / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tpl / rel, dst)
    zero = case / "0"
    zero.mkdir(exist_ok=True)
    (zero / "C").write_text(field_text("volVectorField", "C", vector_internal([(x, y, 0.0) for x, y in zip(mesh.x, mesh.y)])))
    (zero / "Vc").write_text(field_text("volScalarField", "Vc", scalar_internal([a * mesh.depth for a in mesh.area])))
    shutil.copyfile(tpl / "0" / "p", zero / "p")
    if with_zero:
        r0 = zero_r if zero_r is not None else plug_r(mesh, 0.0)
        theta0 = [v * phi0 for v in r0]
        (zero / "thetaC").write_text(field_text("volScalarField", "thetaC", scalar_internal(theta0)))
        (zero / "thetaH").write_text(field_text("volScalarField", "thetaH", scalar_internal([0.0] * mesh.n)))
        (zero / "Dg").write_text(field_text("volScalarField", "Dg", "uniform 0;"))
        (zero / "Us").write_text(field_text("volVectorField", "Us", "uniform (0 0 0);"))
    for w in writes:
        write_dir(case / time_name(w["t"]), mesh, w["t"], w.get("index", 1), w["r"], phi0, w.get("dg"), w.get("us"), w.get("be"), w.get("theta_c_frac", 1.0), None, w.get("q"), w.get("gfield"))
    if restart is not None:
        (case / "restart.json").write_text(json.dumps(restart, indent=1) + "\n")
    if flux is not None:
        fd = case / "postProcessing" / "inletFlux" / flux_start
        fd.mkdir(parents=True, exist_ok=True)
        (fd / "surfaceFieldValue.dat").write_text("# Region : patch inlet\n# Time\torientedSum(phi)\n" + "".join(f"{_fmt(t)}\t{_fmt(q)}\n" for t, q in flux))
    (case / "log.phaseClotAxisymmetricFluid").write_text(log_text)
    (case / "run-exit-code.txt").write_text("0\n")
    return case


def restart_record(prev_case: Path, name: str, index: int) -> dict:
    """The restart.json restart_driver --apply writes for a restart from `name` of the previous session (its chosen directory and its sha256 tree)."""
    import restart_driver as RD
    return {"chosen": name, "chosen_index": index, "requested_index": None, "sibling": None, "source": "synthetic", "in_place": False, "end_time": "0.18505",
            "directories": [], "applied": True, "dictionaries_differ_and_were_accepted": [], "moved_aside": [],
            "sha256": RD.sha256_tree(Path(prev_case) / name)}


def rng(seed: int):
    return random.Random(seed)
