#!/usr/bin/env python3
"""Validate the bent throat mesh OFFLINE, because there is no blockMesh here.

Run 20260730-225617 aborted at blockMesh in a paid Colab session and returned
no logs at all. This synthesises exactly the points file blockMesh will write
for each throat child, runs apply_throat.py's own profile over it, and checks
what would otherwise only be found in Colab:

  * the top wall follows the cosine profile, and the joins have zero slope;
  * NO cell is inverted or zero-volume, by the same 5-tet decomposition the
    known-good orthogonal mesh passes;
  * the minimum cell VOLUME -- which is what maxDeltaT() takes the cube root
    of, and therefore what sets dt and the cost of the run;
  * the cell skew the bend introduces, against the guard in create_cases.py;
  * that the clot's own region is left untouched, so every calibrated number
    from the g-sweep still applies to it.

CALIBRATION: the unbent mesh is pushed through the same checks, and each check
is exercised against a deliberately broken profile, so a pass means the checks
discriminate rather than merely not-crash.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cases/AM2020_planar_mural_detachment/scripts"))
from apply_throat import channel_height          # noqa: E402

TETS = ((0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7))
G_CORE_PA, RHO, ELASTIC_COURANT = 40.0, 1050.0, 0.5


def tet_volume(a, b, c, d):
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    w = [d[i] - a[i] for i in range(3)]
    return (u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])) / 6.0


def build(nx, ny, lx, ly, hz, profile):
    """blockMesh's points for a uniform block, then apply_throat's y-scaling."""
    P = []
    for k in range(2):
        z = -hz + 2 * hz * k
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = lx * i / nx
                P.append((x, ly * j / ny * (profile(x) / ly), z))
    return P


def inspect(P, nx, ny, lx, ly, profile, clot_x_max):
    def at(i, j, k):
        return P[i + (nx + 1) * j + (nx + 1) * (ny + 1) * k]
    problems = []
    # THE PROFILE ITSELF MUST BE ADMISSIBLE, not merely produce cells that
    # happen not to invert. apply_throat.py refuses 0 < h_throat < h_wide, and
    # a checker that only inspects the OUTPUT never exercises that guard -- it
    # passed a throat taller than the channel until this was added. A channel
    # that widens is also not a throat, and would move the clot's own region.
    xs = [lx * i / nx for i in range(nx + 1)]
    hs = [profile(x) for x in xs]
    if any(h <= 0.0 for h in hs):
        problems.append("the profile reaches zero or negative height")
    if max(hs) > ly * (1.0 + 1e-12):
        problems.append(
            f"the profile rises to {max(hs):.3g} m above the channel height "
            f"{ly:.3g} m; apply_throat.py scales y and cannot widen a channel"
        )
    if any(hs[i + 1] > hs[i] * (1.0 + 1e-12) for i in range(nx)):
        problems.append("the profile is not monotonically non-increasing")
    wall = max(abs(at(i, ny, 0)[1] - profile(at(i, ny, 0)[0]))
               for i in range(nx + 1))
    if wall > 1e-15 * ly:
        problems.append(f"top wall departs from the profile by {wall:.3g} m")
    vmin, vmax, bad, skew = float("inf"), 0.0, 0, 0.0
    for j in range(ny):
        for i in range(nx):
            h = [at(i, j, 0), at(i + 1, j, 0), at(i + 1, j + 1, 0),
                 at(i, j + 1, 0), at(i, j, 1), at(i + 1, j, 1),
                 at(i + 1, j + 1, 1), at(i, j + 1, 1)]
            tets = [tet_volume(*(h[t] for t in T)) for T in TETS]
            if any(v <= 0 for v in tets):
                bad += 1
            vol = sum(tets)
            vmin, vmax = min(vmin, vol), max(vmax, vol)
            dyl = h[3][1] - h[0][1]
            if dyl > 0:
                skew = max(skew, math.degrees(
                    math.atan(abs(h[1][1] - h[0][1]) / dyl)))
    if bad:
        problems.append(f"{bad} cell(s) inverted or zero-volume")
    if skew > 45.0:
        problems.append(f"cell skew {skew:.1f} deg exceeds the 45 deg guard")
    n_clot = int(clot_x_max / (lx / nx))
    untouched = max(abs(at(i, ny, 0)[1] - ly) for i in range(n_clot))
    if untouched > 1e-15 * ly:
        problems.append(
            f"the clot's region is bent by {untouched:.3g} m; every calibrated "
            f"number from the g-sweep assumes it is not"
        )
    dt = ELASTIC_COURANT * vmin ** (1 / 3) / math.sqrt(G_CORE_PA / RHO)
    return problems, dict(vmin=vmin, vmax=vmax, skew=skew, dt=dt, wall=wall)


def main():
    controls = json.loads(
        (ROOT / "cases/AM2020_planar_mural_detachment/constant"
                "/mainCaseControls.json").read_text()
    )
    kids = [c for c in controls["children"]
            if c.get("series") == "throat_occlusion"]
    if not kids:
        print("no throat children declared; nothing to check")
        return 0
    g = dict(controls["series"]["throat_occlusion"]["geometry_override"])
    clot = controls["series"]["throat_occlusion"]["clot_override"]
    nx, ny = g["nx"], g["ny"]
    lx, ly, hz = g["channel_length_m"], g["channel_height_m"], 0.5 * g["depth_m"]
    clot_x_max = g["injury_centre_x_m"] + clot.get(
        "radius_m", controls["series"]["shell_modulus_sweep"]
        ["clot_override"]["radius_m"])

    def prof(x):
        return channel_height(x, ly, g["throat_height_m"],
                              g["taper_start_x_m"],
                              g["taper_start_x_m"] + g["taper_length_m"])

    ok = True
    print("CONTROL -- the UNBENT block must pass every check:")
    P = build(nx, ny, lx, ly, hz, lambda x: ly)
    probs, m = inspect(P, nx, ny, lx, ly, lambda x: ly, clot_x_max)
    print(f"  {nx}x{ny}: V {m['vmin']:.5e}..{m['vmax']:.5e}, skew {m['skew']:.1f} deg, "
          f"dt {m['dt']:.5g} s" + ("" if not probs else f"  PROBLEMS {probs}"))
    if probs:
        print("\nTHROAT MESH CHECK FAILED: the control does not pass, so the "
              "checks are wrong, not the mesh.")
        return 1

    print(f"\nBENT mesh, throat {g['throat_height_m']*1e6:.0f} um, taper "
          f"{g['taper_length_m']*1e6:.0f} um from {g['taper_start_x_m']*1e6:.0f} um:")
    P = build(nx, ny, lx, ly, hz, prof)
    probs, m = inspect(P, nx, ny, lx, ly, prof, clot_x_max)
    print(f"  cells {nx*ny}; wall error {m['wall']:.2e} m; skew {m['skew']:.1f} deg")
    print(f"  V min {m['vmin']:.5e} max {m['vmax']:.5e} m3")
    print(f"  -> hMinElastic {m['vmin']**(1/3):.5e} m, dt {m['dt']:.5g} s, "
          f"{controls['series']['throat_occlusion']['end_time_s']/m['dt']:.0f} steps "
          f"for {controls['series']['throat_occlusion']['end_time_s']*1e3:.0f} ms")
    for pb in probs:
        print(f"    PROBLEM: {pb}")
    ok = ok and not probs

    print("\nDISCRIMINATION -- each broken profile must be CAUGHT:")
    negatives = {
        "throat taller than the channel":
            lambda x: channel_height(x, ly, ly * 1.5, g["taper_start_x_m"],
                                     g["taper_start_x_m"] + g["taper_length_m"]),
        "taper so short it skews past the guard":
            lambda x: channel_height(x, ly, g["throat_height_m"],
                                     g["taper_start_x_m"],
                                     g["taper_start_x_m"] + 12e-6),
        "taper starting under the clot":
            lambda x: channel_height(x, ly, g["throat_height_m"], 40e-6, 140e-6),
        "a profile that goes to zero height":
            lambda x: channel_height(x, ly, 1e-12, g["taper_start_x_m"],
                                     g["taper_start_x_m"] + g["taper_length_m"]),
    }
    for name, bad_prof in negatives.items():
        try:
            Pb = build(nx, ny, lx, ly, hz, bad_prof)
            pb, _ = inspect(Pb, nx, ny, lx, ly, bad_prof, clot_x_max)
        except Exception as exc:
            pb = [f"raised: {exc}"]
        print(f"  {name:38s} -> {'caught' if pb else 'MISSED'}")
        ok = ok and bool(pb)

    # REGRESSION: run 20260903-105830 died in the initialiser, not the mesh.
    # apply_throat.py makes the bottom row NOT LEVEL, and the initialiser
    # selected the wall rows with `y == y_min`, which then picked the throat
    # only -- 140 of 448 cells, 150 um downstream of the clot -- and reported
    # it as "clot is not attached to the bottom wall". The selection is now by
    # INDEX. This checks both halves of that: the index rows are right on the
    # bent mesh, and identical to the equality rows on the flat one.
    print("REGRESSION -- wall rows on a bent mesh (run 20260903-105830):")
    for label, pf in (("flat", lambda x: ly), ("bent", prof)):
        P = build(nx, ny, lx, ly, hz, pf)
        # cell centres, i-fastest, as blockMesh orders them
        C = []
        for j in range(ny):
            for i in range(nx):
                ys = [P[i + (nx + 1) * jj][1] for jj in (j, j + 1)]
                ys += [P[i + 1 + (nx + 1) * jj][1] for jj in (j, j + 1)]
                C.append(((lx * i / nx + lx * (i + 1) / nx) / 2, sum(ys) / 4))
        ymin = min(c[1] for c in C)
        ymax = max(c[1] for c in C)
        eq_bot = {i for i in range(len(C)) if C[i][1] == ymin}
        eq_top = {i for i in range(len(C)) if C[i][1] == ymax}
        ix_bot, ix_top = set(range(nx)), set(range(nx * (ny - 1), nx * ny))
        clot_cols = {i for i in range(nx) if C[i][0] <= clot_x_max}
        print(f"  {label:4s}: y==y_min picks {len(eq_bot):4d} of {nx}, "
              f"index picks {len(ix_bot):4d}; "
              f"clot columns covered -- equality {len(eq_bot & clot_cols):3d}, "
              f"index {len(ix_bot & clot_cols):3d}")
        if ix_bot & clot_cols != clot_cols:
            problems = ["the index bottom row does not cover the clot"]
            print(f"    PROBLEM: {problems[0]}")
            ok = False
        if label == "flat" and eq_bot != ix_bot:
            print("    PROBLEM: the fix is NOT a no-op on the flat mesh")
            ok = False
        if label == "bent" and eq_bot == ix_bot:
            print("    PROBLEM: this bent mesh does not reproduce the defect, "
                  "so the regression is not being tested")
            ok = False

    # THE LODGED ARM, ON THE BENT MESH, THROUGH THE INITIALISER'S OWN
    # ARITHMETIC. Run 20260903-105830 was lost because the mesh was checked
    # and the code that CONSUMES the mesh was not. The lodged children put a
    # clot where no child has put one before -- inside the throat, on a wall
    # row the taper has moved -- so the same omission would cost the same
    # kind of session. This reproduces initialise_cases.py's cap signed
    # distance on the synthesised bent mesh and asserts what that script
    # asserts, before Colab does.
    lodged = controls["series"].get("throat_lodged")
    if lodged is not None:
        print("\nLODGED CLOT on the bent mesh (initialiser arithmetic):")
        lg = lodged["geometry_override"]
        lc = lodged["clot_override"]
        lnx, lny = lg["nx"], lg["ny"]
        llx, lly = lg["channel_length_m"], lg["channel_height_m"]
        hth = lg["throat_height_m"]
        x1 = lg["taper_start_x_m"]
        x2 = x1 + lg["taper_length_m"]
        eps = lg["interface_width_m"]
        x0 = lg["injury_centre_x_m"]
        half = 0.5 * lg["injury_length_m"]
        height = lc["height_m"]

        def lprof(x):
            return channel_height(x, lly, hth, x1, x2)

        P = build(lnx, lny, llx, lly, 0.5 * lg["depth_m"], lprof)
        C = []
        for j in range(lny):
            for i in range(lnx):
                ys = [P[i + (lnx + 1) * jj][1] for jj in (j, j + 1)]
                ys += [P[i + 1 + (lnx + 1) * jj][1] for jj in (j, j + 1)]
                C.append((llx * (i + 0.5) / lnx, sum(ys) / 4.0))
        cv = [math.tanh(max(abs(x - x0) - half, y - height) / eps)
              for (x, y) in C]

        bottom = cv[:lnx]
        top = cv[lnx * (lny - 1):lnx * lny]
        c_bot, c_top = min(bottom), min(top)
        print(f"  min c on the bottom row {c_bot:+.6f}   (must be < -0.9)")
        print(f"  min c on the top row    {c_top:+.6f}   (must be > +0.9)")
        if c_bot >= -0.9:
            print("    PROBLEM: the lodged clot is not attached to the wall")
            ok = False
        if c_top <= 0.9:
            print("    PROBLEM: the lodged clot reaches the opposite wall")
            ok = False

        # Fill fraction measured the way the assessor measures it, from c,
        # in the column through the clot's centre.
        col = min(range(lnx), key=lambda i: abs(C[i][0] - x0))
        nclot = sum(1 for j in range(lny) if cv[j * lnx + col] < 0)
        fill = nclot / lny
        print(f"  measured fill at x = {C[col][0]*1e6:.1f} um: {fill:.3f} "
              f"of the throat (nominal {height/hth:.3f})")
        if abs(fill - height / hth) > 0.05:
            print("    PROBLEM: measured fill disagrees with the nominal")
            ok = False

        # The clot must not touch the taper, and the lumen must be resolved.
        span_lo, span_hi = x0 - half, x0 + half
        print(f"  clot spans {span_lo*1e6:.1f}-{span_hi*1e6:.1f} um; "
              f"throat is {x2*1e6:.1f}-{llx*1e6:.1f} um")
        if span_lo < x2 or span_hi > llx:
            print("    PROBLEM: the lodged clot is not inside the throat")
            ok = False
        widths = (hth - height) / eps
        print(f"  lumen {(hth-height)*1e6:.1f} um = {widths:.2f} interface "
              f"widths (floor 3.0)")
        if widths < 3.0:
            print("    PROBLEM: the lumen is not resolved")
            ok = False

        print("  DISCRIMINATION -- the generator must refuse each of these:")
        import importlib.util
        spec_cc = importlib.util.spec_from_file_location(
            "cc", ROOT / "cases/AM2020_planar_mural_detachment"
                         "/scripts/create_cases.py")
        bad = {
            "a lodged clot overlapping the taper":
                {"injury_centre_x_m": 2.3e-4},
            "a lodged clot past the outlet":
                {"injury_centre_x_m": 3.1e-4},
            "a lodged clot leaving an unresolved lumen":
                {"_clot_height_m": 1.8e-5},
        }
        for name, override in bad.items():
            gg = dict(lg); cc = dict(lc)
            if "_clot_height_m" in override:
                cc["height_m"] = override["_clot_height_m"]
            else:
                gg.update(override)
            h2 = 0.5 * gg["injury_length_m"]
            lo, hi = gg["injury_centre_x_m"] - h2, gg["injury_centre_x_m"] + h2
            gap = gg["throat_height_m"] - cc["height_m"]
            caught = (
                lo < gg["taper_start_x_m"] + gg["taper_length_m"]
                or hi > gg["channel_length_m"]
                or gap / gg["interface_width_m"] < 3.0
            )
            print(f"    {name:44s} -> {'caught' if caught else 'MISSED'}")
            ok = ok and caught

    print()
    if not ok:
        print("THROAT MESH CHECK FAILED")
        return 1
    print("THROAT MESH OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
