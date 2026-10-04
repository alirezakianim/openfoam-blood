#!/usr/bin/env python3
"""Initialise the one-sided mural clot on the bottom wall.

The clot is the region {|x - x0| < Linj/2} AND {y < Hclot}, written as a signed
distance

    d = max(|x - x0| - Linj/2, y - Hclot)

which is negative inside the cap, then smoothed as c = tanh(d/epsilonC).
c < 0 is skeleton, c > 0 is fluid.

Note the difference from the axisymmetric plug: there the clot touched the wall
all the way round, so the constraint was radial. Here it touches ONE wall over
a finite patch, so the two constraints are streamwise extent and height. Run
20260810-015336 showed the axisymmetric plug loads its bond within 8% of a
single value across 70% of its area; this shape is the reason to expect
otherwise, and the gate measures whether that expectation holds.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


def read_vectors(path: Path, expected: int) -> list[tuple[float, float, float]]:
    text = path.read_text(errors="replace")
    match = re.search(
        r"(?ms)internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;",
        text,
    )
    if not match:
        raise ValueError(f"{path}: cannot parse vectors")
    values = [
        tuple(float(x) for x in row.split())
        for row in re.findall(r"\(([^()]+)\)", match.group(2))
    ]
    if int(match.group(1)) != expected or len(values) != expected:
        raise ValueError(f"{path}: expected {expected} vectors, got {len(values)}")
    return values


def replace_scalar(path: Path, values: list[float]) -> None:
    text = path.read_text()
    body = "\n".join(f"{v:.16g}" for v in values)
    replacement = (
        f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;"
    )
    updated, count = re.subn(
        r"internalField\s+(?:uniform\s+[^;]+|nonuniform\s+List<scalar>\s+\d+\s*\(.*?\))\s*;",
        lambda _: replacement, text, count=1, flags=re.S,
    )
    if count != 1:
        raise ValueError(f"{path}: cannot replace scalar field")
    path.write_text(updated)


def replace_vector(path: Path, values: list[tuple[float, float, float]]) -> None:
    text = path.read_text()
    body = "\n".join(f"({x:.16g} {y:.16g} {z:.16g})" for x, y, z in values)
    replacement = (
        f"internalField nonuniform List<vector>\n{len(values)}\n(\n{body}\n)\n;"
    )
    updated, count = re.subn(
        r"internalField\s+(?:uniform\s+\([^;]+\)|nonuniform\s+List<vector>\s+\d+\s*\(.*?\))\s*;",
        lambda _: replacement, text, count=1, flags=re.S,
    )
    if count != 1:
        raise ValueError(f"{path}: cannot replace vector field")
    path.write_text(updated)


def write_shell_field(path: Path, values: list[float]) -> None:
    """Write 0/shell from scratch.

    Not derived from a template field, because no template carries it: this is
    a new field and the axisymmetric template it would have been copied from
    has wedge patches this planar mesh does not have. The boundary conditions
    match the other scalars written by the mural gate's configure_fields --
    zeroGradient on the four real patches, empty front and back.
    """
    body = "\n".join(f"{v:.16g}" for v in values)
    path.write_text(
        "FoamFile\n"
        "{\n"
        "    format ascii;\n"
        "    class volScalarField;\n"
        '    location "0";\n'
        "    object shell;\n"
        "}\n"
        "dimensions [0 0 0 0 0 0 0];\n"
        f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;\n"
        "boundaryField\n"
        "{\n"
        "    inlet { type zeroGradient; }\n"
        "    outlet { type zeroGradient; }\n"
        "    bottom { type zeroGradient; }\n"
        "    top { type zeroGradient; }\n"
        "    front { type empty; }\n"
        "    back { type empty; }\n"
        "}\n"
    )


def half_ellipse_arc(
    cx: float, a: float, b: float, samples: int
) -> tuple[list[float], list[float]]:
    """The sampled upper arc of the half-ellipse x = cx + a cos t, y = b sin t,
    t in [0, pi], at `samples` points -- numpy's linspace(0, pi, samples)
    reproduced term for term (t_i = i * (pi / (samples - 1)), the last point
    set to pi exactly), because the Xu 2017 case must start from the SAME
    field numerics priced it on (docket 20260929-1430,
    probes/numerics-analyst-02-xu-piso.py, sdist_half_ellipse)."""
    if samples < 3:
        raise ValueError(f"half_ellipse_arc: {samples} samples is not an arc")
    step = math.pi / (samples - 1)
    t = [i * step for i in range(samples - 1)] + [math.pi]
    return [cx + a * math.cos(v) for v in t], [b * math.sin(v) for v in t]


def half_ellipse_signed_distance(
    points: list[tuple[float, float]], cx: float, a: float, b: float,
    samples: int,
) -> list[float]:
    """Signed distance to a half-ellipse seated on the wall (y = 0): the
    minimum Euclidean distance to the SAMPLED arc, negative inside
    ((x - cx)/a)^2 + (y/b)^2 < 1. Numerics' construction, cell for cell.

    THE SAMPLED ARC IS THE SPECIFICATION, NOT AN APPROXIMATION TO BE REFINED.
    The exact distance to the continuous arc is smaller by up to half a chord
    (0.015 um here, 0.02 h), and at the cell nearest the curve that moves the
    tanh profile by up to ~2.5e-3 -- far above the 1e-6 the pre-registration's
    first-write field test (A1) allows. The predictions were priced on the
    sampled field, so that is the field written.

    SPEED WITHOUT CHANGING THE ANSWER. The brute minimum over 4001 samples is
    235 million distance evaluations for the two Xu shapes; this returns the
    IDENTICAL minimum from ~1/15 of them. The distance to the samples is
    1-Lipschitz in the sample positions: |P - S_j| >= |P - S_k| - |k - j| *
    chord_max. So a block of samples around a coarse sample k can hold the
    minimum only if d_k - half_block * chord_max does not exceed the best
    found; every other block is skipped with a proof, not a guess."""
    px, py = half_ellipse_arc(cx, a, b, samples)
    n = len(px)
    chord = max(
        math.hypot(px[i + 1] - px[i], py[i + 1] - py[i]) for i in range(n - 1)
    )
    stride = 40
    half = stride // 2
    coarse = list(range(0, n, stride))
    if coarse[-1] != n - 1:
        coarse.append(n - 1)
    slack = half * chord * (1.0 + 1e-12) + 1e-18
    out = []
    for x, y in points:
        dk = sorted(
            (math.sqrt((x - px[k]) ** 2 + (y - py[k]) ** 2), k) for k in coarse
        )
        best = dk[0][0]
        for d0, k in dk:
            if d0 - slack > best:
                break
            for j in range(max(0, k - half), min(n, k + half + 1)):
                dj = math.sqrt((x - px[j]) ** 2 + (y - py[j]) ** 2)
                if dj < best:
                    best = dj
        inside = ((x - cx) / a) ** 2 + (y / b) ** 2 < 1.0
        out.append(-best if inside else best)
    return out


def main() -> int:
    case = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    spec = json.loads((case / "child.json").read_text())
    # The finite-strain gate borrows this initialiser for its five mural
    # children rather than keeping a second copy of the same signed distance.
    # One source of truth per configuration; the alternative drifts.
    if spec["experiment"] not in ("mural_geometry_gate", "finite_strain_gate",
                                  "planar_mural_detachment"):
        raise ValueError(f"unexpected experiment {spec['experiment']}")

    g = spec["geometry"]
    clot = spec["clot"]
    n = int(g["expected_cell_count"])
    centres = read_vectors(case / "0" / "C", n)

    x0 = g["injury_centre_x_m"]
    half = 0.5 * g["injury_length_m"]
    height = clot["height_m"]
    epsilon = g["interface_width_m"]

    # SHAPE. The rectangular cap is the original and stays the default, so
    # every existing child is bit-for-bit unchanged.
    #
    # `semisphere` is Dr He Li's 2026-08-13 request: a circle of radius R whose
    # centre sits R/2 ABOVE the wall, so the wall cuts off the bottom quarter
    # of its height and three quarters stand in the channel. Height above the
    # wall is 1.5R and the contact footprint is 2*sqrt(3)/2*R = 1.732R.
    #
    # The wall does the truncation on its own -- the mesh has no cells below
    # y = 0 -- so the signed distance is just the circle's. That matches the
    # rectangular cap, which likewise ignores the wall when computing d, and
    # keeps the interface profile a clean tanh right down to the contact.
    shape = clot.get("shape", "cap")
    if shape == "cap":
        def signed(x: float, y: float) -> float:
            return max(abs(x - x0) - half, y - height)
    elif shape == "semisphere":
        radius = clot["radius_m"]
        centre_y = 0.5 * radius
        expected_height = 1.5 * radius
        if not math.isclose(height, expected_height, rel_tol=1e-9):
            raise ValueError(
                f"semisphere: height_m {height} must be 1.5 * radius_m "
                f"({expected_height}); the two are one number, and letting "
                f"them disagree silently changes the blockage fraction that "
                f"the summary reports"
            )

        def signed(x: float, y: float) -> float:
            return math.hypot(x - x0, y - centre_y) - radius
    elif shape == "disc":
        # A FREE, SHELL-ONLY DISC (design 2026-10-02-case-free-shell-throat,
        # rev. 3, S2): a circle of radius radius_m centred at
        # (centre_x_m, centre_y_m), with NO core (core_shell false is
        # enforced in create_cases.py, which is where the series-level switch
        # lives). Unlike the semisphere, the centre is declared directly
        # rather than pinned at 0.5*radius_m above the wall, because this
        # shape is placed by the case design (P1), not fixed to the wall by
        # construction. `rests_on_wall` only relaxes the generator's 3*epsilon
        # gap floor (create_cases.py) for the bottom side; it changes nothing
        # here.
        disc_cx = clot["centre_x_m"]
        disc_cy = clot["centre_y_m"]
        disc_r = clot["radius_m"]

        def signed(x: float, y: float) -> float:
            return math.hypot(x - disc_cx, y - disc_cy) - disc_r
    elif shape == "halfEllipseDome":
        # XU ET AL. 2017, FIGURE 5 (design 2026-09-29-case-xu2017-fig5, rev. 4,
        # section 1): the clot is a half-ellipse seated on the wall, centred
        # on the injury, semi-axes half_length_m (streamwise) and height_m.
        # The injury IS its footprint, so the two are one number and are
        # refused if they disagree -- the same rule the semisphere applies to
        # its height and radius.
        dome_a = clot["half_length_m"]
        dome_samples = int(clot["signed_distance_samples"])
        if not math.isclose(2.0 * dome_a, g["injury_length_m"], rel_tol=1e-9):
            raise ValueError(
                f"halfEllipseDome: 2 * half_length_m = {2.0 * dome_a} must "
                f"equal injury_length_m {g['injury_length_m']}; the footprint "
                f"and the injury are one number"
            )
        dome_d = half_ellipse_signed_distance(
            [(x, y) for x, y, _z in centres], x0, dome_a, height, dome_samples
        )
    else:
        raise ValueError(
            f"unknown clot shape {shape!r}; use cap, semisphere, disc or "
            f"halfEllipseDome"
        )

    c_values = []
    if shape == "halfEllipseDome":
        c_values = [math.tanh(d / epsilon) for d in dome_d]
    else:
        for x, y, _z in centres:
            c_values.append(math.tanh(signed(x, y) / epsilon))

    if min(c_values) >= -0.9:
        raise ValueError(
            f"clot has no solid bulk: c_min={min(c_values)}. "
            "Raise the clot height or narrow epsilonC."
        )
    if max(c_values) <= 0.9:
        raise ValueError(f"no open lumen: c_max={max(c_values)}")

    # The clot must actually sit ON the bottom wall and NOT touch the top one,
    # or it is not the one-sided geometry this gate exists to test.
    # THE WALL ROWS ARE SELECTED BY INDEX, NOT BY COORDINATE EQUALITY.
    #
    # This used to be `y == y_min` and `y == y_max`, and BOTH were wrong.
    #
    # (1) It broke outright on a mesh whose bottom row is not level. Run
    #     20260903-105830 died here with "clot is not attached to the bottom
    #     wall: min c there is 1.0". The mesh was correct and the clot was
    #     correct: apply_throat.py scales y by H(x)/H, so the bottom-row cell
    #     centre is 0.357 um in the wide section and 0.143 um in the throat,
    #     and `y == y_min` selected 140 of 448 cells -- THE THROAT ONLY, 150 um
    #     downstream of the clot. It then reported a geometry-indexing failure
    #     as a physics failure.
    #
    # (2) It was ALREADY silently weak on the flat meshes. blockMesh averages
    #     vertices per cell, so the top row's y centres are not bit-identical
    #     across columns: on run 20260902-231316 g04, `y == y_max` selected
    #     284 of 350 cells. The "clot reaches the TOP wall" check has been
    #     testing 81% of that wall on every child ever run.
    #
    # blockMesh orders a single block i-fastest, then j, then k, so the bottom
    # row is [0, nx) and the top row is [nx*(ny-1), nx*ny). That is exact for
    # both meshes. VERIFIED to be a no-op on the flat mesh: on g04 the
    # index set and the `y == y_min` set are IDENTICAL, and the top-row check
    # goes from 284 cells to 350 with min c unchanged at 0.999918.
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    if nx * ny * nz != len(centres):
        raise ValueError(
            f"geometry says {nx}x{ny}x{nz} = {nx * ny * nz} cells but the mesh "
            f"has {len(centres)}; the row indexing below assumes they agree"
        )
    # The ordering is ASSUMED above and CHECKED here, because a silently
    # different ordering would pick a diagonal band and still return numbers.
    row = centres[:nx]
    if any(row[i][0] >= row[i + 1][0] for i in range(nx - 1)):
        raise ValueError(
            "the first nx cell centres are not increasing in x, so this mesh "
            "is not ordered i-fastest and the wall rows cannot be indexed"
        )
    if not centres[nx][1] > centres[0][1]:
        raise ValueError(
            "cell nx is not above cell 0, so this mesh is not ordered with j "
            "second and the wall rows cannot be indexed"
        )
    attached = c_values[:nx]
    opposite = c_values[nx * (ny - 1):nx * ny]
    if shape == "disc":
        # A CIRCLE RESTING ON A FLAT WALL TOUCHES AT ONE POINT, not along a
        # flat footprint, so the generic "a chunk of the bottom row sits deep
        # inside the clot (c <= -0.9)" test below is the wrong test for this
        # shape -- a tangent circle never satisfies it at any radius, and
        # would be refused for being exactly what it was asked to be (design
        # 2026-10-02-case-free-shell-throat, S2: "rests_on_wall true allows a
        # zero bottom gap and only that"). The actual contact floor (3*epsilon
        # against the wall, or a declared zero gap under rests_on_wall) is
        # checked analytically in create_cases.py, before blockMesh, because
        # it needs no mesh to evaluate. Here only the cheap, mesh-dependent
        # sanity check remains: the disc must not have been placed so that it
        # actually buries cells of the bottom row (min c there would then be
        # very negative) -- resting is a single tangent point, not a dig-in.
        if clot.get("rests_on_wall") and min(attached) < -0.9999:
            raise ValueError(
                f"disc: min c at the bottom wall is {min(attached)}, which "
                f"is a buried contact, not a tangent rest. Check "
                f"centre_y_m against radius_m."
            )
    else:
        if min(attached) >= -0.9:
            raise ValueError(
                f"clot is not attached to the bottom wall: min c there is "
                f"{min(attached)}"
            )
    if min(opposite) <= 0.9:
        raise ValueError(
            f"clot reaches the TOP wall (min c there is {min(opposite)}); "
            "this must be a one-sided mural clot, not a bridge"
        )

    solid_cells = sum(1 for v in c_values if v < 0.0)
    bonded_faces = sum(1 for c in attached if c < 0.0)

    # CORE/SHELL, source 33 (Xu et al. 2017). Absent unless the clot declares a
    # core radius, so every existing child writes no `shell` field at all and
    # the solver's material blend stays the identity.
    #
    # The core is the CONCENTRIC inward offset of the same dome: same centre,
    # radius R - t. It is not a separate shape, because the shell has to be a
    # layer of roughly even thickness over the core -- that is what makes the
    # stress in it accumulate from the free surface inward and peak AT the
    # core, which is where the tear is supposed to start.
    #
    # The indicator is smoothed over its own width, NOT over epsilonC. epsilonC
    # is 3 um and the core is 4.5 um in radius, so a tanh of width epsilonC
    # would leave the whole core inside its own transition and there would be
    # no core value anywhere. materialWidth is 2h, which spans the transition
    # over about three cells: sharp enough to be a material interface, smooth
    # enough that the modulus is not a step in the matrix.
    shell_values: list[float] | None = None
    core_cells = 0
    if clot.get("core_radius_m") is not None:
        if shape != "semisphere":
            raise ValueError(
                f"core_radius_m is only defined for the semisphere shape; "
                f"this clot is {shape!r}"
            )
        core_radius = clot["core_radius_m"]
        if not 0.0 < core_radius < clot["radius_m"]:
            raise ValueError(
                f"core_radius_m {core_radius} must lie strictly between 0 and "
                f"radius_m {clot['radius_m']}"
            )
        shell_thickness = clot["radius_m"] - core_radius
        cells_across_shell = shell_thickness / g["h_m"]
        if cells_across_shell < 4.0 - 1e-9:
            raise ValueError(
                f"the shell is {shell_thickness:.3g} m thick on h = "
                f"{g['h_m']:.3g}, which is {cells_across_shell:.2f} cells. "
                f"Four is the floor: a shell thinner than the tear it is "
                f"supposed to localise cannot show where the tear went."
            )
        material_width = clot.get("material_width_m", 2.0 * g["h_m"])

        # WHERE THE CORE SITS, AND WHY IT MATTERS MORE THAN HOW BIG IT IS.
        #
        # The clot is a circle of radius R centred at y = 0.5R, so a CONCENTRIC
        # core of radius r has its own centre 0.5R above the wall and its
        # lowest point at 0.5R - r. On the configuration run 20260825-013027
        # used -- R = 23 um, r = 9 um -- that is 2.5 um ABOVE THE WALL. The
        # core never touched the injury. Every cell the wall bond anchored was
        # SHELL, and the stiff core rode on soft shell, which is a far better
        # explanation of the severed shape Dr He Li rejected on 2026-08-25 than
        # the anchor's own edge was.
        #
        # core_seated puts the core's centre ON the wall, so it is a half-disc
        # with 2r of genuine contact with the injury. That is what Li asked
        # for: "please make the core region an half sphere or circle, so the
        # contact area to the injury site is larger."
        #
        # IT COSTS SHELL CLEARANCE AND THE ARITHMETIC MUST BE DONE HERE. A
        # seated core is 0.5R further from the clot's centre than a concentric
        # one, so the nearest clot surface is at
        #     R - sqrt(r^2 + 0.25 R^2)
        # and NOT the concentric R - r. At R = 23 um that ceiling is 11.5 um
        # even as r -> 0, so at epsilon = 4 um NO seated core clears 3 epsilon.
        # This is why the mesh was refined to h = 0.714 um / epsilon = 3 um.
        core_seated = bool(clot.get("core_seated", False))
        centre_y = 0.0 if core_seated else 0.5 * clot["radius_m"]

        if core_seated:
            clearance = clot["radius_m"] - math.hypot(
                core_radius, 0.5 * clot["radius_m"]
            )
            widths = clearance / g["interface_width_m"]
            if widths < 3.0:
                raise ValueError(
                    f"seated core: the nearest clot surface is "
                    f"{clearance:.3g} m away = {widths:.2f} interface widths, "
                    f"below the 3.0 floor. A seated core sits 0.5R = "
                    f"{0.5 * clot['radius_m']:.3g} m off the clot centre, so "
                    f"the clearance is R - sqrt(r^2 + 0.25 R^2) and NOT the "
                    f"concentric R - r. radius {clot['radius_m']:.3g} m, core "
                    f"{core_radius:.3g} m, epsilonC "
                    f"{g['interface_width_m']:.3g} m. The ceiling as r -> 0 is "
                    f"0.5R = {0.5 * clot['radius_m']:.3g} m, so if that is "
                    f"below 3 epsilon no seated core works at this epsilon."
                )

        shell_values = []
        for x, y, _z in centres:
            d_core = math.hypot(x - x0, y - centre_y) - core_radius
            shell_values.append(0.5 * (1.0 + math.tanh(d_core / material_width)))
    elif shape == "halfEllipseDome":
        # THE XU CORE: a flattened half-ellipse, seated on the wall, with its
        # OWN centre (Xu's core sits at x ~ 0.908 L, the clot's centre at
        # 0.914 L). The label width is DECLARED, never defaulted: the model
        # the predictions were priced on used 3 um, and the 2h default is
        # x2.09 steeper at the corner where every first core damage is
        # predicted (red team 05 #1). The core must be BURIED with 3 interface
        # widths of clot around it, measured as the clot's signed distance at
        # every sample of the core's own arc -- the same floor the seated
        # semisphere holds, on the shape actually drawn.
        for key in ("core_centre_x_m", "core_half_length_m", "core_height_m",
                    "material_width_m"):
            if clot.get(key) is None:
                raise ValueError(
                    f"halfEllipseDome: {key} is required; a missing core "
                    f"dimension or label width must not fall back to a default"
                )
        ccx = clot["core_centre_x_m"]
        cca = clot["core_half_length_m"]
        ccb = clot["core_height_m"]
        material_width = clot["material_width_m"]
        core_px, core_py = half_ellipse_arc(ccx, cca, ccb, dome_samples)
        d_at_core = half_ellipse_signed_distance(
            list(zip(core_px, core_py)), x0, dome_a, height, dome_samples
        )
        clearance = -max(d_at_core)
        widths = clearance / g["interface_width_m"]
        if widths < 3.0:
            raise ValueError(
                f"halfEllipseDome: the core's nearest point is {clearance:.3g} "
                f"m inside the clot surface = {widths:.2f} interface widths, "
                f"below the 3.0 floor (epsilonC {g['interface_width_m']:.3g} m)"
            )
        shell_thickness = clearance
        d_core_values = half_ellipse_signed_distance(
            [(x, y) for x, y, _z in centres], ccx, cca, ccb, dome_samples
        )
        shell_values = [
            0.5 * (1.0 + math.tanh(d / material_width)) for d in d_core_values
        ]

    if shell_values is not None:
        core_cells = sum(1 for s in shell_values if s < 0.5)

        if core_cells == 0:
            raise ValueError(
                "core_radius_m produced no core cells; the solver refuses this "
                "configuration too, but catching it here costs nothing"
            )
        # The core must be BURIED. A core cell that is not solid clot is a
        # core cell in the interface or in blood, and the whole design rests
        # on the core being the part that cannot fail.
        exposed = sum(
            1 for s, c in zip(shell_values, c_values) if s < 0.5 and c > -0.9
        )
        if exposed:
            raise ValueError(
                f"{exposed} core cells lie outside the bulk clot (c > -0.9). "
                f"The core must be buried inside the clot, or the material "
                f"that cannot fail is sitting in the fluid."
            )

        # THE SHELL MUST BE BULK MATERIAL, not a label painted on the
        # interface. This is measured on chi = 1 - q = -c/2 + 1/2 rather than
        # inferred from a thickness, because it is the quantity the solver
        # actually weights the stress, the drag and psi+ by.
        #
        # Run 20260814-224138 is why the check exists. Its shell was 4.5 um
        # thick on epsilon = 3 um, and the cells-across-shell test above
        # passed it at 6 cells. But chi in the 162 shell cells had mean 0.705
        # and max 0.869 -- not one cell was solid material -- against 0.99 in
        # the core. The shell's 4 Pa modulus and 1e-11 m2 permeability were
        # applied to material at most 87% solid, and the core/shell contrast
        # the run existed to show could not appear at any duration.
        #
        # Cells were never the constraint. The ratio of shell thickness to
        # INTERFACE WIDTH was, and no guard measured it. Exactly the class of
        # min_advance_cells: a threshold in cells where the physics needs a
        # length ratio.
        # s >= 0.9, not s >= 0.5. At 0.5 the sample includes the core/shell
        # TRANSITION cells, which sit deep inside the clot and carry a high
        # chi: on the failing geometry that reads max chi = 0.952 and the
        # check passes, missing the very defect it was written for. At 0.9 the
        # sample is cells that genuinely carry the shell's properties, and the
        # same geometry reads 0.869 over 162 cells -- reproducing the run's
        # measured value exactly.
        shell_chi = [
            0.5 * (1.0 - c) for s, c in zip(shell_values, c_values)
            if s >= 0.9 and c < 0.0
        ]
        shell_chi_max = max(shell_chi) if shell_chi else 0.0
        epsilon = g["interface_width_m"]
        if shell_chi_max < 0.95:
            raise ValueError(
                f"the shell never becomes bulk material: max chi over shell "
                f"cells is {shell_chi_max:.3f}, below 0.95. The shell is "
                f"{shell_thickness:.3g} m thick against an interface width of "
                f"{epsilon:.3g} m ({shell_thickness / epsilon:.2f} epsilon), "
                f"so it lies inside the clot/blood interface and its modulus "
                f"and permeability would be applied to material that is not "
                f"solid. Thicken the shell to at least 3-4 interface widths, "
                f"or narrow epsilon and refine the mesh to match."
            )

    replace_scalar(case / "0" / "c", c_values)
    replace_scalar(case / "0" / "w", [1.0] * n)
    replace_scalar(case / "0" / "Dg", [0.0] * n)
    replace_scalar(case / "0" / "H", [0.0] * n)
    replace_vector(case / "0" / "U", [(0.0, 0.0, 0.0)] * n)
    replace_vector(case / "0" / "Us", [(0.0, 0.0, 0.0)] * n)

    if shell_values is not None:
        write_shell_field(case / "0" / "shell", shell_values)

    out = case / "postProcessing" / "initialisation"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "child_id": spec["child_id"],
        "wall_condition": spec["wall_condition"],
        "adhesive_strength_pa": spec["adhesive_strength_pa"],
        "growth": spec["growth"],
        "alpha_m2_s": spec["alpha_m2_s"],
        "cell_count": n,
        "solid_cells": solid_cells,
        "solid_fraction": solid_cells / n,
        "clot_covered_bottom_cells": bonded_faces,
        "c_min": min(c_values),
        "c_max": max(c_values),
        "c_min_at_bottom_wall": min(attached),
        "c_min_at_top_wall": min(opposite),
        # AGAINST THE CHANNEL THE CLOT IS ACTUALLY IN. On a uniform channel
        # that is channel_height_m and this is what it has always been. On a
        # throat child with the clot lodged INSIDE the narrowing, dividing by
        # the 50 um inlet height reports 0.20 for a clot that fills half of a
        # 20 um throat -- a number that is wrong by 2.5x in the direction that
        # makes an occlusion look like nothing. throat_height_m is present
        # only on throat children, and only the lodged ones put the clot past
        # the taper, so the journey children keep the old value exactly.
        "blockage_fraction": clot["height_m"] / (
            g["throat_height_m"]
            if (
                g.get("throat_height_m") is not None
                and g["injury_centre_x_m"]
                >= g["taper_start_x_m"] + g["taper_length_m"]
            )
            else g["channel_height_m"]
        ),
        "shape": shape,
        # For a dome the peak height overstates the obstruction badly -- it is
        # the number Dr Li objected to at 0.333, and a dome of the same height
        # blocks far less. Report the AREA actually occupied as well, so the
        # comparison between the two shapes is like for like.
        "blocked_area_fraction": solid_cells / n,
        "one_sided": True,
        # Absent unless the clot declares a core, so no existing summary moves.
        "core_radius_m": clot.get("core_radius_m"),
        "core_cells": core_cells,
        "shell_cells": (solid_cells - core_cells) if shell_values else None,
        "shell_thickness_m": (
            clot["radius_m"] - clot["core_radius_m"]
            if clot.get("core_radius_m") is not None
            else None
        ),
    }
    if shape == "halfEllipseDome":
        # Added only for the dome, so no existing summary moves.
        summary["core_clearance_m"] = shell_thickness
        summary["core_clearance_interface_widths"] = (
            shell_thickness / g["interface_width_m"]
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"initialise_cases.py: {error}")
