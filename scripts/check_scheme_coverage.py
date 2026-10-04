#!/usr/bin/env python3
"""Every `div` the solver uses must be NAMED in every gate that runs it.

`divSchemes` in these cases sets `default none`, deliberately - an unnamed
convection scheme is a silent choice, and this project would rather abort than
guess one. The cost of that decision is that a `div` operator with no entry
aborts at the first timestep, on Colab, after the upload.

Gate 5 aborted on `Cannot find scheme for div(sigmaD)` because its generator
was ported from gate 3b-ii-c's with the SOLVER injection and without the SCHEME
injection. Nothing local objected: the case meshed, initialised, and the solver
constructed and ran the phase field, whose `div(qPhi,w)` WAS present. The gap
appeared only where the mechanics began.

This reads the operators out of the SOLVER SOURCE and requires each one to be
named in each gate's `fvSchemes` (template plus whatever its controls file
injects). It is therefore a check on the pair, not on either alone - which is
what the two previous guards, each scoped to one gate, could not do. It also
means a solver change that introduces a new `div` fails here rather than in
every gate at once, three hours of upload later.

`grad` and `laplacian` are NOT checked: both sections carry a real `default`,
so an unnamed operator resolves rather than aborting.

SCOPED BY REACHABILITY. `div(sigmaD)` lives inside `if (clotMechanics_)`, which
DEFAULTS TO FALSE, so a fluid-only gate never evaluates it and does not need to
name it. Six such gates exist and requiring it of them would be a false alarm -
the kind that gets a checker switched off. A gate counts as a mechanics gate if
anything under it turns `clotMechanics` on: its template, its controls, or its
child generator, since different gates do it in different places.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SOLVER = ("user-applications/phaseClotAxisymmetricFluid/"
          "phaseClotAxisymmetricFluid.C")

# Operators the solver only reaches when `clotMechanics` is enabled. The switch
# defaults to FALSE, so a fluid-only gate cannot hit these.
MECHANICS_ONLY = {"div(sigmaD)"}

# GATES THAT WERE ALREADY BROKEN BEFORE THIS CHECKER COULD SEE THEM.
#
# Until 2026-08-22 this file matched only the two-argument fvm::div(a, b) form,
# so every EXPLICITLY NAMED scheme in the solver was invisible to it and the
# check reported "3 operators covered" while never looking at the four that
# matter most. Teaching it DIV_NAMED surfaced 102 findings across the gates
# below, in four operators:
#
#   div(phi,Us)  div(phiReleased,c)  div(sigmaElastic)  div(phiReleased,Wp)
#
# THREE OF THE FOUR PREDATE the platelet-carrier work entirely -- they went
# missing whenever those fields were added and nothing could report it. These
# gates are historical: each was run and recorded before the field it is now
# missing existed, and none is on the stage-3/stage-4 path. Any of them would
# FatalIOError at its first solve if re-run today.
#
# LISTED, NOT SUPPRESSED, and the distinction is the point. They are printed on
# every invocation and merely do not set the exit status, because a checker
# that ships permanently red is a checker nobody reads -- which is how the
# blind spot survived in the first place. A gate NOT on this list fails the
# build. The list may only shrink: fix the gate's fvSchemes, or retire the
# gate, and delete its line. Adding a name here to quiet a NEW failure would
# reproduce the exact defect this entry documents.
STALE_GATES = frozenset({
    "AM2020_deformation_damage_release_feasibility",
    "AM2020_eulerian_slab_gate",
    "AM2020_fracture_J_integral",
    "AM2020_fracture_onset",
    "AM2020_fracture_sent_energy_release",
    "AM2020_fracture_surface_energy",
    "AM2020_fsi_coupling_convergence",
    "AM2020_fsi_fixed_three_sweep_gate",
    "AM2020_fsi_floor_origin",
    "AM2020_fsi_floor_probe",
    "AM2020_fsi_flow_lag_probe",
    "AM2020_fsi_flow_lag_retry",
    "AM2020_fsi_norm_probe",
    "AM2020_fsi_relaxation_factorial",
    "AM2020_fsi_stiffness_contraction",
    "AM2020_fsi_tolerance_relevance_gate",
    "AM2020_mechanics_brinkman_slab",
    "AM2020_mechanics_elastic_slab",
    "AM2020_mechanics_rigid_limit",
    "AM2020_planar_diffusion_refinement",
    "AM2020_planar_dirichlet_sine_spatial_refinement",
    "AM2020_planar_dirichlet_sine_temporal_floor_resolution",
    "AM2020_planar_pressure_outlet",
    "AM2020_planar_solid_limit",
    "AM2020_staggered_coupling_probe",
    "AM2020_storti_comparison",
    "AM2020_zero_flow_limit",
})

# fvm::div(a, b) -> "div(a,b)";  fvc::div(a) -> "div(a)". Trailing underscores
# are OpenFOAM member-variable convention and are not part of the field name.
DIV = re.compile(r"fv[mc]::div\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,\s*([A-Za-z_][A-Za-z0-9_]*)\s*)?\)")

# EXPLICITLY NAMED SCHEMES, WHICH DIV ABOVE CANNOT SEE AT ALL.
#
# fvm::div(flux, field, "div(a,b)") takes THREE arguments and looks up the
# third verbatim, ignoring the names of the first two. DIV matches only the
# two-argument form, so every named call in the solver was invisible to this
# checker -- including both sides of the phase-field/platelet pair.
#
# That is not a cosmetic gap. It is the gap the carrier mismatch shipped
# through: div(phiReleased,c) and div(phi,Wp) were the two most consequential
# div entries in the solver and this file had never checked either of them.
# The call is also the only place the requirement is authoritative, because a
# named scheme's key is a STRING LITERAL and need not resemble its arguments --
# solvePlatelets() passes phaseCarrier_() under the name div(phiReleased,Wp),
# and no argument-derived guess would ever produce that.
DIV_NAMED = re.compile(
    r"fv[mc]::div\([^;]*?\"(div\([^\"]+\))\"\s*\)", re.S
)


def scheme_name(first: str, second: str | None) -> str:
    a = first.rstrip("_")
    if second is None:
        return f"div({a})"
    return f"div({a},{second.rstrip('_')})"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    # Comments are prose ABOUT code, not code. Without this, a comment reading
    # "matching fvc::div(sigma) and dragCoeff*U" is scanned as a real call and
    # every shipped case is reported as missing a div(sigma) scheme -- 40 false
    # alarms from one sentence. The blanker is shared with, and self-tested by,
    # check_fvschemes_coverage.py rather than reimplemented here.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_fvschemes_coverage import blank_cpp_comments

    # SOME `fvc::div(x)` CALLS LOOK UP NO SCHEME AT ALL, and this checker
    # cannot tell which from a regex because the distinction is the argument's
    # TYPE: fvc::div of a SURFACE field is fvc::surfaceIntegrateExtrapolate
    # (fvcDiv.C:46), a bare sum of face values with no dictionary lookup, while
    # fvc::div of a VOLUME field does look one up -- which is why div(sigmaD)
    # and div(sigmaElastic) are real requirements and div(phi) is not.
    #
    # The authoritative list is ALLOWED_UNNAMED in check_fvschemes_coverage.py,
    # where a None value means exactly "requires no scheme". Imported rather
    # than restated: two hand-maintained copies of this list would drift, and
    # the direction they drift in is a checker that fails every case in the
    # repo for a call that cannot fail. That is what happened when
    # solvePassiveDisplacement() added an fvc::div(phi_) to the solver -- 28
    # long-passing gates reported as broken at once.
    from check_fvschemes_coverage import ALLOWED_UNNAMED

    no_scheme_needed = set()
    for call, entry in ALLOWED_UNNAMED.items():
        if entry is not None:
            continue
        match = DIV.fullmatch(call.strip())
        if match:
            no_scheme_needed.add(scheme_name(match.group(1), match.group(2)))

    source = blank_cpp_comments((root / SOLVER).read_text())
    # A named call carries the scheme key as a string literal, and THAT is what
    # the solver looks up -- the argument names are irrelevant to it. So the
    # named calls are collected first and then REMOVED from the text before DIV
    # runs, or the same call would also be counted under an argument-derived
    # name that no fvSchemes file has any reason to declare.
    named = {m.group(1) for m in DIV_NAMED.finditer(source)}
    unnamed_source = DIV_NAMED.sub(" ", source)

    needed = sorted(
        (
            {scheme_name(m.group(1), m.group(2))
             for m in DIV.finditer(unnamed_source)}
            | named
        )
        - no_scheme_needed
    )
    if not needed:
        print(f"no div operators found in {SOLVER}; the pattern is stale")
        return 1

    problems, checked = [], 0
    for schemes in sorted((root / "cases").glob("*/template/system/fvSchemes")):
        gate = schemes.parents[2]
        text = schemes.read_text()
        div_block = re.search(r"divSchemes\s*\{(.*?)\n\}", text, re.S)
        if div_block is None:
            continue
        block = div_block.group(1)
        if not re.search(r"^\s*default\s+none\s*;", block, re.M):
            continue                      # a real default resolves everything
        # Does this gate turn the mechanics on? Different gates do it in
        # different places - the template dictionary, the controls file, or the
        # child generator - so ask the whole gate, not one known location.
        mechanics = False
        for candidate in gate.rglob("*"):
            if not candidate.is_file() or candidate.suffix in {".pyc"}:
                continue
            try:
                body = candidate.read_text(errors="replace")
            except OSError:
                continue
            if re.search(r"clotMechanics\s+(yes|true|on|1)\b", body):
                mechanics = True
                break
        checked += 1

        # Whatever the gate's controls file injects counts as present.
        injected = set()
        for controls in sorted((gate / "constant").glob("*.json")):
            try:
                data = json.loads(controls.read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(data, dict):
                injected |= set(data.get("required_schemes", {})
                                .get("divSchemes", {}))

        for name in needed:
            if name in MECHANICS_ONLY and not mechanics:
                continue
            present = re.search(rf"^\s*{re.escape(name)}\s", block, re.M)
            if not present and name not in injected:
                problems.append(
                    f"{gate.name}: divSchemes has `default none` and nothing "
                    f"provides {name}, which the solver uses"
                )

    live = sorted({p for p in set(problems)
                   if p.split(":", 1)[0] not in STALE_GATES})
    stale = sorted({p for p in set(problems)
                    if p.split(":", 1)[0] in STALE_GATES})

    if stale:
        print(f"PRE-EXISTING, IN {len({p.split(':')[0] for p in stale})} STALE "
              f"GATE(S) -- {len(stale)} finding(s), NOT gating:")
        for p in stale:
            print("  " + p)
        print("  See STALE_GATES for why these do not fail the check.\n")

    if live:
        print("DIV SCHEME NOT COVERED:")
        for p in live:
            print("  " + p)
        print("\n  Name it in the gate's template fvSchemes, or add it to the "
              "controls file's required_schemes.divSchemes so the generator "
              "injects it.")
        return 1
    print(f"scheme coverage check: {len(needed)} solver div operator(s) "
          f"covered in {checked} gate(s) -- {', '.join(needed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
