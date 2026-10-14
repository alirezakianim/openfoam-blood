#!/usr/bin/env python3
"""Write 0/thetaC and 0/thetaH so the Stage 2 restart triple is complete.

WHY THIS EXISTS. The solver's persistent-content restart is ALL-OR-NONE
(`phaseClotAxisymmetricFluid.C:1711-1725`): with contentModel and
plateletContentModel on, `thetaC`, `thetaH` and `nP` must all be present or
all absent. Absent means "legacy theta seeding with ZERO free platelets" --
and zero platelets deposit nothing, which is the whole point of a Stage 2
run.

So a Stage 2 child must supply all three. `create_cases.py` writes `0/nP`,
but it runs BEFORE the initialiser and cannot write the contents: thetaC and
thetaH are functions of `c` and `shell`, which the initialiser produces. This
script runs immediately after it.

Run 20260915-062201 is what this is for: `0/nP` alone, and the solver
refused at the first construction with "Incomplete persistent content restart
at time 0". Nothing was solved.

WHAT IT REPRODUCES, EXACTLY. The seeding the solver would have done itself
(`.C:1753-1770`), because the point is a Stage 2 child that starts from the
SAME clot as em27/em28 rather than a different one:

    q      = 0.5*(1 + c)                 [initialise_cases.py:345, .C:6838-6845]
    theta0 = theta_ref * max(1 - q, 0) = theta_ref * max(0.5*(1 - c), 0)
    sClamp = min(max(shell, 0), 1)
    thetaH = theta0 * sClamp
    thetaC = theta0 * (1 - sClamp)

Every case seeds the triple since S21: contentModel and plateletContentModel are built in.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


PATCHES = """boundaryField
{
    inlet { type zeroGradient; }
    outlet { type zeroGradient; }
    bottom { type zeroGradient; }
    top { type zeroGradient; }
    front { type empty; }
    back { type empty; }
}
"""
# THE Y MESH'S PATCHES (inlet, outlet, wall, front, back): the Y gate's walls are ONE patch named `wall`, and thetaC/thetaH written with the
# channel's bottom/top would name patches that do not exist (a FatalIOError at the first read). Chosen by constant/yGate.json, the marker the
# generator writes for that child only; every other child keeps PATCHES above, byte for byte.
PATCHES_Y = """boundaryField
{
    inlet { type zeroGradient; }
    outlet { type zeroGradient; }
    wall { type zeroGradient; }
    front { type empty; }
    back { type empty; }
}
"""


def dictionary_value(text: str, key: str) -> str | None:
    m = re.search(rf"(?m)^\s*{re.escape(key)}\s+([^;]+);", text)
    return m.group(1).strip() if m else None


def read_internal(path: Path) -> list[float]:
    """Read internalField, uniform or nonuniform."""
    text = path.read_text()
    m = re.search(r"(?m)^internalField\s+uniform\s+([^;]+);", text)
    if m:
        raise ValueError(
            f"{path}: uniform internalField; this script needs the per-cell "
            "values the initialiser writes")
    m = re.search(r"(?m)^internalField\s+nonuniform\s+List<scalar>\s*\n?\s*"
                  r"(\d+)\s*\n?\s*\((.*?)\)\s*;", text, re.S)
    if not m:
        raise ValueError(f"{path}: no nonuniform scalar internalField found")
    values = [float(v) for v in m.group(2).split()]
    if len(values) != int(m.group(1)):
        raise ValueError(
            f"{path}: header says {m.group(1)} values, found {len(values)}")
    return values


def write_scalar(path: Path, name: str, values: list[float]) -> None:
    patches = PATCHES_Y if (path.parents[1] / "constant" / "yGate.json").is_file() else PATCHES
    body = "\n".join(f"{v:.16g}" for v in values)
    path.write_text(
        "FoamFile\n{\n    format ascii;\n    class volScalarField;\n"
        f'    location "0";\n    object {name};\n}}\n'
        "dimensions [0 0 0 0 0 0 0];\n"
        f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;\n"
        + patches
    )


def main() -> int:
    case = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    props = case / "constant" / "phaseClotProperties"
    if not props.is_file():
        raise SystemExit(f"{props} is missing")
    text = props.read_text()

    # S21: contentModel and plateletContentModel are built in (true), so they are no longer in
    # the dictionary and every case seeds the triple. Reading them here would find them absent
    # and skip the seeding, which the solver's all-or-none restart check then refuses.
    theta_ref = dictionary_value(text, "skeletonVolumeFraction")
    if theta_ref is None:
        raise SystemExit("skeletonVolumeFraction is not declared")
    theta_ref = float(theta_ref)

    c_values = read_internal(case / "0" / "c")
    n = len(c_values)

    shell_path = case / "0" / "shell"
    if shell_path.is_file():
        shell = read_internal(shell_path)
        if len(shell) != n:
            raise SystemExit(
                f"0/shell has {len(shell)} values, 0/c has {n}")
    else:
        # No core/shell child: shell_ is absent, the solver reads it as zero,
        # and every seeded cell is core content. Matching that is the point.
        shell = [0.0] * n

    nP = case / "0" / "nP"
    if not nP.is_file():
        raise SystemExit(
            "0/nP is missing. The restart triple is all-or-none, so writing "
            "thetaC and thetaH without it would fail the solver's check the "
            "other way round. create_cases.py writes it for a child that "
            "declares platelet_number_density_m3.")

    theta_c, theta_h = [], []
    for c, s in zip(c_values, shell):
        theta0 = theta_ref * max(0.5 * (1.0 - c), 0.0)
        s_clamp = min(max(s, 0.0), 1.0)
        theta_h.append(theta0 * s_clamp)
        theta_c.append(theta0 * (1.0 - s_clamp))

    write_scalar(case / "0" / "thetaC", "thetaC", theta_c)
    write_scalar(case / "0" / "thetaH", "thetaH", theta_h)

    total = sum(theta_c) + sum(theta_h)
    print(
        f"stage2 content seeded: {n} cells, theta_ref {theta_ref}, "
        f"sum(thetaC+thetaH) {total:.6e}, "
        f"max {max(max(theta_c), max(theta_h)):.6e}, "
        f"shell {'read' if shell_path.is_file() else 'absent (all core)'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
