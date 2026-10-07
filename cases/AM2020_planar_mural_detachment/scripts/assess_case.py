#!/usr/bin/env python3
"""Turn the main case's fifteen runs into the PI-facing results table.

This is NOT a gate. The smoke gates tested the code; this reports what the
model does. So there is no pass/fail verdict, only:

  * an ADMISSIBILITY check per child -- completed, bounded, no NaN. An
    inadmissible child is named and EXCLUDED from every table rather than
    silently averaged in, which is the failure mode a results script has that a
    gate does not.
  * three tables, one per series.

Every number carries its own preconditions. A table entry that could not be
computed says why, in the entry, instead of being absent.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


TIME_LINE = re.compile(r"(?m)^Time = ([0-9.eE+-]+)\s*s?\s*$")
FSI_LINE = re.compile(
    r"PHASE_CLOT_FSI time=(\S+) iterations=(\d+) residual0=(\S+) "
    r"residual=(\S+) contraction=(\S+) omega=(\S+) converged=(\S+)"
)
# Matched by KEY, not by position. The previous version was one long
# positional regex, so adding `reformedFaces` and `injuryFaces` to the solver
# in the arteriolar revision stopped it matching AT ALL -- and the failure mode
# is a silent "no PHASE_CLOT_DEBOND lines", i.e. the assessor reporting that a
# run never debonded when in fact it never parsed. Key-based matching tolerates
# new fields and reordering; only a REMOVED key can now break it, and that
# raises below instead of returning nothing.
DEBOND_LINE = re.compile(r"PHASE_CLOT_DEBOND (.+)")
DEBOND_REQUIRED = (
    "time", "newFaces", "debondedFaces", "bondedClotFaces", "wallFaces",
    "debondedArea", "peakTraction", "strength", "utilisation",
)


def parse_debond(text):
    """PHASE_CLOT_DEBOND lines as dicts, keyed by field name."""
    out = []
    for body in DEBOND_LINE.findall(text):
        fields = dict(
            tok.split("=", 1) for tok in body.split() if "=" in tok
        )
        missing = [k for k in DEBOND_REQUIRED if k not in fields]
        if missing:
            raise ValueError(
                f"PHASE_CLOT_DEBOND line is missing {missing}; the solver's "
                f"log format changed and this assessor would otherwise report "
                f"the run as never having debonded. Line: {body[:200]}"
            )
        out.append({
            "time": float(fields["time"]),
            "new_faces": int(fields["newFaces"]),
            # Absent before the injury-footprint change; 0 is the correct
            # reading for those older logs, re-adhesion did not exist.
            "reformed_faces": int(fields.get("reformedFaces", 0)),
            "debonded_faces": int(fields["debondedFaces"]),
            "bonded_clot_faces": int(fields["bondedClotFaces"]),
            # Before the change every wall face was adhesive, so wallFaces
            # WAS the footprint.
            "injury_faces": int(fields.get("injuryFaces", fields["wallFaces"])),
            "wall_faces": int(fields["wallFaces"]),
            "debonded_area_m2": float(fields["debondedArea"]),
            "peak_traction_pa": float(fields["peakTraction"]),
            "strength_pa": float(fields["strength"]),
            "utilisation": float(fields["utilisation"]),
        })
    return out
DIAGNOSTIC_LINE = re.compile(
    r"PHASE_CLOT_DIAGNOSTIC time=(\S+) .*?clotVolume=(\S+)"
)
DETACHMENT_LINE = re.compile(r"PHASE_CLOT_DETACHMENT .*?cells=(\d+)")
RELEASED_LINE = re.compile(r"PHASE_CLOT_RELEASED time=(\S+) cells=(\d+) volume=(\S+)")
SKELETON_LINE = re.compile(r"skelIters=(\d+) skelResidual=(\S+)")
# The guard added after run 20260811-051109 lost three arterial children to
# SIGFPE. A child that trips it now writes its fields and says why it stopped,
# so divergence becomes a reported outcome instead of an empty directory.
DIVERGED_LINE = re.compile(
    r"PHASE_CLOT_SKELETON_DIVERGED time=(\S+) corrector=(\d+) reason=(\S+)"
)
BODY_VELOCITY_LINE = re.compile(r"PHASE_CLOT_BODY_VELOCITY (?P<kv>.*)")

# WHERE the fracture driving energy lives. Run 20260814-055141 tore every child
# from the outside in because H peaked in the four-cell phase interface at 13.5x
# its bulk value -- an artefact of the numerical pin that holds Us at zero in
# the fluid, not a property of the material. A damage result is only meaningful
# when the bulk drives it, so the ratio is reported rather than assumed.
DAMAGE_DRIVER_LINE = re.compile(r"PHASE_CLOT_DAMAGE_DRIVER (?P<kv>.*)")


def parse_damage_driver(text: str) -> list[dict[str, float]]:
    out = []
    for m in DAMAGE_DRIVER_LINE.finditer(text):
        kv = dict(
            (k, float(v))
            for k, v in re.findall(r"(\w+)=(-?[\d.eE+-]+)", m.group("kv"))
        )
        for k in ("Hbulk", "Hinterface", "Hblood"):
            if k not in kv:
                raise ValueError(
                    f"PHASE_CLOT_DAMAGE_DRIVER is missing {k}; the interface "
                    f"artefact check would silently report nothing. "
                    f"Line: {m.group(0)[:200]}"
                )
        out.append(kv)
    return out


# THE TWO-VELOCITY GATE. A skeleton cannot outrun the fluid dragging it: the
# fluid momentum equation alone gives u_f - v_s = -phi_f*grad(p)/K, which is
# downstream unconditionally. Any cell with |Vs| > |U| is a direct measurement
# of a violated mixture balance div(sigma_e) = grad(p).
#
# Run 20260904-214345 measured exceedFraction = 0.883 with the algebraic
# closure. The coupled solve must drive it below 1%. That threshold is well
# above instrument resolution: the clot is ~6500 cells, so 1% is ~65 cells, not
# one. A gate set at "zero cells" would be unpassable for the ordinary reason
# that a diffuse interface has cells where |U| is near zero and the ratio is
# numerically meaningless.
SKELETON_VELOCITY_LINE = re.compile(r"PHASE_CLOT_SKELETON_VELOCITY (?P<kv>.*)")
SKELETON_VELOCITY_REQUIRED = ("clotCells", "meanVsOverU", "maxVsOverU",
                              "exceedFraction", "exceedFractionMargin")

# THE GATE IS THE MARGIN COUNT, NOT THE BARE ONE, AND THAT IS DELIBERATE.
#
# A bare "|Vs| > |U|" count sits on a SIGN CHANGE, which is worse than sitting
# too tight. Solving the 2x2 for a free element with no elastic stress:
#
#     u - v  ~  A_s*U/K  (inertial lag, skeleton slower)
#            +  phi_s*grad(p)/K  (pressure push, skeleton faster)
#
# with A_s = phi_s/deltaT, so they balance at deltaT* = U/|grad(p)| = 5.9e-06 s.
# Our timestep is adaptive around 1e-06, just below that. A bare count would
# read ~0% while deltaT stays small and climb once it grows past ~6e-06 s, on a
# run that is correct either way.
#
# The solver reports both. exceedFraction keeps the 0.883 baseline of run
# 20260904-214345 comparable; exceedFractionMargin counts |Vs| > 1.1|U| among
# cells with |U| above a floor. Across deltaT from 1e-07 to 3e-05 the genuine
# free-element ratio stays inside 0.70..1.005, so 10% never trips on it, while
# the pathology it must catch ran a mean ratio of 1.4 and a max of 3.2.
VS_EXCEED_LIMIT = 0.01

# THE WHOLE-FIELD VELOCITY BOUND, AND WHY IT IS DELIBERATELY LOOSE.
#
# Every metric above -- meanVsOverU, maxVsOverU, exceedFraction,
# exceedFractionMargin -- is computed over CLOT CELLS ONLY (c < 0 and
# drag > 1). `maxVs` is gMax over the WHOLE FIELD. Run 20260905-191135
# diverged with maxVs at 0.62 m/s, 75x the 0.00833 m/s mean flow, while its
# clot-cell metrics were indistinguishable from the run before it
# (meanVsOverU 0.9386 vs 0.9370, exceedFractionMargin 0.0133). The runaway was
# in the LUMEN and no gate here could see it.
#
# NO CLEAN TWO-VELOCITY RUN EXISTS TO CALIBRATE AGAINST. All four on disk
# failed, at peak maxVs/maxU of:
#
#     20260904-214345   2127      (pointwise closure, structurally wrong)
#     20260905-054441     27.7    (clot grew 4x)
#     20260905-113709      6.36   (compacted; survived longest)
#     20260905-191135    237.7    (network pressure on the wrong weight)
#
# So 20 is NOT a tight bound derived from healthy behaviour -- it is the value
# beyond which the field is pathological on any reading, since Vs is
# drag-locked to U inside the clot and driven to U by Kres outside it, so the
# ratio is O(1) in every regime the model admits. It fires on three of the
# four; the fourth (6.36) is caught by the conservation and c-bound gates
# anyway. TIGHTEN THIS once a run survives to t = 0.03, and record what a
# healthy peak actually is.
VS_WHOLE_FIELD_LIMIT = 20.0


def parse_skeleton_velocity(text: str) -> list[dict[str, float]]:
    """PHASE_CLOT_SKELETON_VELOCITY lines as dicts.

    Raises rather than returning empty when a field is missing, because a
    silently absent key here would make the gate report "nothing exceeded"
    on a run where the solver never printed the number at all.
    """
    out = []
    for m in SKELETON_VELOCITY_LINE.finditer(text):
        kv = dict(
            (k, float(v))
            for k, v in re.findall(r"(\w+)=(-?[\d.eE+-]+)", m.group("kv"))
        )
        missing = [k for k in SKELETON_VELOCITY_REQUIRED if k not in kv]
        if missing:
            raise ValueError(
                f"PHASE_CLOT_SKELETON_VELOCITY is missing {missing}; the "
                f"two-velocity gate would silently pass. "
                f"Line: {m.group(0)[:200]}"
            )
        out.append(kv)
    return out


# S6c-2: THE BALANCE MOVED CURRENCY, NOT PURPOSE.
#
# PHASE_CLOT_MASS closed d(c)/dt + div(F c) = L - G. c is deleted, so that
# equation no longer exists and the tag is gone with it. The replacement
# closes the equation the solver actually integrates,
# d(theta)/dt + div(F theta) = S, and it is read here under the same
# fail-loud contract: a missing key RAISES, because a conservation gate that
# silently finds no data reads exactly like one that passed.
MASS_LINE = re.compile(r"PHASE_CLOT_CONTENT_BALANCE (?P<kv>.*)")
MASS_REQUIRED = ("time", "contentVolume", "contentResidual", "contentDVdt",
                 "contentOuterFlux", "contentTransferRate",
                 "detachedContentRate")

# THE CONSERVATION GATE, AND WHY IT IS THE CUMULATIVE DRIFT.
#
# The solver has printed a closed clot-volume budget every step since long
# before 2026-09-05:
#
#     residual = dV/dt + outerFlux - 0.5*carrierNetFlux
#                      + 0.5*diffusion - 0.5*growth
#
# and NOTHING READ IT. Run 20260905-054441 child em_2vel grew the clot 4.0x in
# 14 ms with 98% of the growth sitting in that residual, for 9391 consecutive
# steps, and the assessment reported it as a two-velocity gate failure and a
# CFL crash. Both were true. Neither was the headline. This gate exists so a
# violated conservation law cannot again be the one thing not looked at.
#
# WHAT THE RESIDUAL IS. Working the budget against the equation actually
# solved -- `bounded` convection subtracts fvm::Sp(fvc::div(phi), c), so the
# solved form is ddt(c) + [div(phi c)]_scheme - c*div(phi) = S -- every term
# cancels except one:
#
#     clotResidual == -0.5 * int c * div(carrier) dV
#
# It is identically zero for a solenoidal carrier and only for a solenoidal
# carrier. `phi_` is solenoidal (|div phi| <= 8e-6 1/s, run 20260818-125909);
# `phiSkelOwn_` is not and CANNOT be -- Cogan & Guy 2010 state outright that
# the individual phase velocities of a two-phase mixture are not
# divergence-free. So this gate is not "is the carrier clean", it is "has the
# phase-field transport been corrected for the divergence its carrier
# legitimately has".
#
# WHY CUMULATIVE AND NOT INSTANTANEOUS. |residual|/|dV/dt| reaches 58.2 on the
# five g-children that conserve volume to 3e-4, and 974 on em_free -- because
# dV/dt passes through zero and the ratio is evaluated on a sign change. That
# is the identical defect that made the first exceedFraction gate flaky, and
# it is not repeated here. The conserved quantity is the VOLUME, so the gate
# is the drift in the volume:
#
#     driftFraction = sum(residual_i * dt_i) / clotVolume_0
#
# THE LIMIT IS DERIVED, NOT PICKED. Measured driftFraction, recomputed under
# the CORRECTED budget (the -0.5*clotCarrierNetFlux term was dropped on
# 2026-09-05 when c moved to conservative transport; see the note beside
# clotResidual in the solver):
#
#     g04 g10 g20 g30 g40   1-velocity, solenoidal carrier   -0.00001  (all 5)
#     th_hold               1-velocity, solenoidal carrier   -0.00002
#     em_freeG40            1-velocity + bondHold carrier    +0.00408
#     th_break              1-velocity + bondHold carrier    +0.00456
#     em_free               1-velocity + bondHold carrier    +0.00478
#     em_2vel  (09-04)      two-velocity                     +1.14811
#     em_2vel  (09-05)      two-velocity                     +3.20730
#
# 1e-5 is the discretisation floor. It is measured on the children whose
# carrier is solenoidal, and for a solenoidal carrier the bounded and
# conservative operators are IDENTICAL -- so that number is the floor of the
# closure now in use, not an extrapolation from a different one. (Under the
# old budget these same children read +0.00031; all of it was the
# carrierNetFlux term, not discretisation.)
#
# The +0.004..0.005 band is the bondHold carrier's interior divergence, under
# the OLD bounded operator. Those runs still finished at a volume ratio of
# 1.10, indistinguishable from the clean ones, so that error is real but
# scientifically harmless and must NOT be failed -- and conservative transport
# is expected to drop them to the floor anyway, since it corrects for exactly
# that divergence.
#
# 0.05 is 5000x the floor, 10x the largest error ever measured on a run whose
# conclusions were usable, and 23x below the smaller of the two defects it
# exists to catch. Well separated in both directions.
CLOT_DRIFT_LIMIT = 0.05


def parse_clot_mass(text: str) -> list[dict[str, float]]:
    """PHASE_CLOT_CONTENT_BALANCE lines as dicts.

    Raises rather than skipping when a key is missing, for the same reason
    parse_skeleton_velocity does: a conservation gate that silently finds no
    data reads exactly like a conservation gate that passed, and that is the
    failure this whole check is here to end.
    """
    out: list[dict[str, float]] = []
    for m in MASS_LINE.finditer(text):
        kv = dict(
            (k, float(v))
            for k, v in re.findall(r"(\w+)=(-?[\d.eE+-]+)", m.group("kv"))
        )
        missing = [k for k in MASS_REQUIRED if k not in kv]
        if missing:
            raise ValueError(
                f"PHASE_CLOT_CONTENT_BALANCE is missing {missing}; the "
                f"conservation gate would silently pass. "
                f"Line: {m.group(0)[:200]}"
            )
        # Read under the names the rest of this file uses, so the drift
        # arithmetic and the reports below are unchanged. theta and chi
        # differ by the constant theta_ref, and a RATIO of two volumes in
        # the same currency is invariant to it -- which is the only way
        # clotVolume enters the drift gate.
        kv["clotVolume"] = kv["contentVolume"]
        kv["clotResidual"] = kv["contentResidual"]
        kv["clotDVdt"] = kv["contentDVdt"]
        kv["clotOuterFlux"] = kv["contentOuterFlux"]
        out.append(kv)
    return out


def clot_volume_drift(rows: list[dict[str, float]]) -> dict[str, float | None]:
    """Cumulative conservation drift of the clot volume, as a fraction of V0.

    dt is differenced from the reported times rather than read from controls,
    because the run is adaptive and a nominal deltaT would misweight every
    step. The first interval is measured from t=0, because that is the span
    the first reported residual actually covers.

    THAT IS ONLY TRUE WHILE THE SOLVER PRIMES ITS PREVIOUS VOLUME, and this
    docstring previously justified the t=0 start by saying previousContentVolume_
    "starts" there -- which described the defect rather than the design. If
    the member is left at its 0 member-initialiser, the first reported
    contentDVdt is V0/dt1, this integral collects exactly V0 from that one
    step, and a perfectly conserved run grades at +100.0% drift. Run
    20260920-220246 did: 0.999997 of its total drift came from step 1, while
    its other 271 steps summed to -1.2e-11 of V0 and its worst single residual
    was 4.3e-22. The prime is asserted at build time by
    scripts/check_per_step_state_is_primed.py; do not re-derive the weighting
    from what the member happens to hold.
    """
    if not rows:
        return {"drift_fraction": None, "volume_ratio": None,
                "volume_initial_m3": None, "volume_final_m3": None}
    v0 = rows[0]["clotVolume"]
    drift = 0.0
    t_prev = 0.0
    for r in rows:
        drift += r["clotResidual"] * (r["time"] - t_prev)
        t_prev = r["time"]
    return {
        "drift_fraction": (drift / v0) if v0 else None,
        "volume_ratio": (rows[-1]["clotVolume"] / v0) if v0 else None,
        "volume_initial_m3": v0,
        "volume_final_m3": rows[-1]["clotVolume"],
    }


PHASE_BOUNDS_LINE = re.compile(r"PHASE_CLOT_PHASE_BOUNDS (?P<kv>.*)")
# S6c-2: THE BOUND IS REPORTED IN q. c in [-1, 1] and q in [0, 1] are the
# same statement under c = 2q - 1; the solver now prints the surviving
# variable. Same check, same cells, renamed thresholds.
PHASE_BOUNDS_REQUIRED = ("time", "qMin", "qMax", "qUnder", "qOver")

# c MUST LIE IN [-1, 1] AND THIS IS NOT A TOLERANCE.
#
# q = 0.5(c+1) is a volume fraction. Below c = -1 it goes NEGATIVE, qEff =
# max(q, qFloor) clamps to 1e-06, and w = Wp/qEff explodes -- run
# 20260905-113709 measured w = -2.8e+05 and died of SIGFPE in solvePhaseField.
# Everything downstream of q is then meaningless: drag, growth, platelets.
#
# The solver has printed cUnder and cOver every step for a long time and
# assess_case.py never read either. Measured peaks, every run on disk:
#
#     g04..g40, th_hold, th_break, em_free, em_freeG40,
#     em_2vel 09-04, em_2vel 09-05     cUnder = 0.0 EXACTLY, cOver <= 1.1e-13
#     em_2vel 09-05 (113709)           cUnder = 1.7788
#
# So the instrument floor is 1.1e-13 round-off and the failure is 1.78. 0.05 is
# 5e+11 times the floor and 36x below the failure, and it is also physically
# negligible in its own right: c = -1.05 is chi = 1.025, i.e. phi_s 2.5% above
# nominal. It fires EARLY in the failure mode, which walks 1.02 -> 2.49 within
# 1.6 ms, so a run that trips it has not yet wasted a session.
#
# Deliberately NOT zero: c is now transported by a CONSERVATIVE scheme, which
# unlike the bounded one it replaced carries no boundedness guarantee. A gate
# at machine zero would be unpassable by construction.
C_BOUND_LIMIT = 0.05


def parse_phase_bounds(text: str) -> list[dict[str, float]]:
    """PHASE_CLOT_PHASE_BOUNDS lines as dicts. Raises on a line missing a key."""
    out: list[dict[str, float]] = []
    for m in PHASE_BOUNDS_LINE.finditer(text):
        kv = dict(
            (k, float(v))
            for k, v in re.findall(r"(\w+)=(-?[\d.eE+-]+)", m.group("kv"))
        )
        missing = [k for k in PHASE_BOUNDS_REQUIRED if k not in kv]
        if missing:
            raise ValueError(
                f"PHASE_CLOT_PHASE_BOUNDS is missing {missing}; the phase "
                f"bound gate would silently pass. Line: {m.group(0)[:200]}"
            )
        # S6c-2: CONVERTED HERE SO THE LIMIT DOES NOT HAVE TO MOVE.
        #
        # The solver now reports the bound in q. c = 2q - 1, so an excursion
        # in c is exactly TWICE the excursion in q:
        #     cUnder = max(0, -1 - cMin) = max(0, -2*qMin) = 2*qUnder
        #     cOver  = max(0, cMax - 1)  = max(0, 2*qMax - 2) = 2*qOver
        # C_BOUND_LIMIT = 0.05 was derived from a measured cUnder of 1.7788
        # on run 20260905-113709. Rebasing the gate onto q would have meant
        # halving a threshold that came from a measurement, which is the
        # kind of quiet respecification this project keeps auditing for.
        # Converting instead leaves the limit, the fixtures and every
        # archived comparison on the scale they were established on.
        kv["cUnder"] = 2.0*kv["qUnder"]
        kv["cOver"] = 2.0*kv["qOver"]
        kv["cMin"] = 2.0*kv["qMin"] - 1.0
        kv["cMax"] = 2.0*kv["qMax"] - 1.0
        out.append(kv)
    return out


SHELL_MATRIX_LINE = re.compile(r"PHASE_CLOT_SHELL_MATRIX (?P<kv>.*)")

# EVERY PREDICATE FIELD IS REQUIRED ON EVERY RECORD.
#
# The gate below decides solvability from five counters. Reading them with
# .get(key, 0) means a record that LOST those fields -- truncated write, an
# older emitter, an interleaved parallel log -- scores as perfectly healthy,
# because "absent" and "zero" become the same thing. Codex reproduced exactly
# that on 2026-09-09: four good records plus one truncated record still
# returned gate_ok. It is the same fail-open shape as the c-bound and Wp gates
# a day earlier, reintroduced in the gate that was supposed to be the hard
# precondition. A record is either complete or the history is unassessable.
# A KEY WHOSE VALUE WILL NOT PARSE MUST GO MISSING, NOT CRASH THE ASSESSOR.
#
# The key=value sweep every parser below uses is
# re.findall(r"(\w+)=(-?[\d.eE+-]+)", ...), and `-` is itself a member of that
# character class. So `equivMax=-inf` captures the value `-` and float("-")
# raises ValueError -- the whole assessment dies on one corrupt record instead
# of marking that record unassessable.
#
# FOUND BY scripts/check_content_record_parser.py before either parser had
# ever read a real -inf, and it corrects STAGE 7.00's claim that "inf is
# caught as a MISSING KEY": bare `inf` is, and `-inf` was not -- I had only
# negative-tested the unsigned spelling.
#
# Applied to the two parsers that have an explicit required-key check and a
# status to downgrade to (content, shell matrix), where dropping the key
# yields UNASSESSABLE. The other six parsers keep the loud ValueError on
# purpose: they have no status field, so a crash is their fail-closed
# behaviour and turning it into a silently absent key would be the wrong
# direction.
def _numeric_kv(kv: str) -> dict[str, float]:
    # WHOLE TOKENS, NOT VALID PREFIXES. Codex, 2026-09-10: `thetaCNeg=0oops`
    # used to parse as 0.0, because the value pattern stops happily at the
    # first character it cannot use and float("0") succeeds. A truncated
    # token is corruption, and reading it as a number is exactly the silent
    # failure this parser exists to prevent. The lookahead forces the value to
    # run to whitespace or end of line; a partial token now simply does not
    # match, so the key goes MISSING and the required-key test catches it.
    out: dict[str, float] = {}
    for k, v in re.findall(r"(\w+)=(-?[\d.eE+-]+)(?=\s|$)", kv):
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


SHELL_MATRIX_REQUIRED = (
    "time", "minMagDiagFinal", "minRowDominance", "maxOffDiagRowSum",
    "rowsNotDominant", "zeroDiagCells", "nonFiniteDiag", "nonFiniteOffDiag",
    "nonFiniteSource", "nonFinitePsi",
)


def parse_shell_matrix(text: str) -> list[dict[str, float]]:
    """PHASE_CLOT_SHELL_MATRIX lines as dicts. Absent in pre-2026-09-08 logs."""
    out: list[dict[str, float]] = []
    for m in SHELL_MATRIX_LINE.finditer(text):
        out.append(_numeric_kv(m.group("kv")))
    return out


# THE CARRIER RECORD -- the cause side of the failure that ends two-velocity
# runs, and until 2026-09-14 read by NOTHING.
#
# PHASE_CLOT_SHELL_MATRIX above reports the matrix the smoother divides by.
# This record reports WHAT DROVE IT THERE: the Courant number of the flux that
# actually transports `c` and `shell` (maxCo in controlDict bounds phi_, not
# this one), the carrier magnitude, the pre-solve diagonal, and -- critically
# -- which carrier the run used.
#
# `carrierBlend` is a WORD, so _numeric_kv drops it. It is parsed separately
# below, because it is the single field that says which of the two
# configurations produced every other number on the line. Losing it silently
# would make blend-on and blend-off records indistinguishable, which is
# exactly the comparison em26_blend_on/em26_blend_off exists to make.
CARRIER_LINE = re.compile(r"PHASE_CLOT_CARRIER (?P<kv>.*)")

CARRIER_REQUIRED = (
    "time", "maxCarrierCo", "maxMagCarrier", "minShellDiagOverVbyDt",
    "shellDiagNonPositive", "shellDiagNonFinite",
)

CARRIER_COUNTERS = ("shellDiagNonPositive", "shellDiagNonFinite")

CARRIER_BLEND = re.compile(r"carrierBlend=(ON|OFF)(?=\s|$)")


def parse_carrier(text: str) -> list[dict]:
    """PHASE_CLOT_CARRIER lines as dicts, with the blend word preserved.

    Absent in pre-2026-09-08 logs, and absent whenever the shell equation is
    not solved -- an empty list is "not emitted", NOT "healthy", and the
    caller must distinguish those.
    """
    out: list[dict] = []
    for m in CARRIER_LINE.finditer(text):
        kv = m.group("kv")
        rec: dict = dict(_numeric_kv(kv))
        blend = CARRIER_BLEND.search(kv)
        # Missing rather than defaulted: a record whose blend word did not
        # parse must fail the required-key test, not quietly claim OFF.
        if blend:
            rec["carrierBlend"] = blend.group(1)
        out.append(rec)
    return out


# THE STAGE 1 CONTENT RECORDS. REPORTED, NEVER GATED.
#
# Run 20260909-115458 had to be read out of the raw logs by hand because
# nothing here parsed these lines, which is how a run stops being
# self-describing. They are reported and NOT gated, deliberately: equivMax is
# a MODEL DIFFERENCE under contentOperator `content` (the growth source the
# theta equation omits, plus upwind against vanLeer weights) and an
# IMPLEMENTATION error only under `matched`. One threshold cannot mean both
# things, and inventing one before the matched operator has ever run would be
# a threshold chosen from a single data point.
#
# `operator` and `contentSourceDt` are in REQUIRED because a log written
# before 2026-09-09 has neither, and silently reading such a log as the
# matched test is exactly the fail-open the shell-matrix gate was fixed for.
CONTENT_LINE = re.compile(r"PHASE_CLOT_CONTENT (?P<kv>.*)")
# Emitted by the carrier-less branch of solvePhaseField() when the content
# model is on. theta is frozen on that path while chi keeps moving, so any
# equivMax from such a run is meaningless. One occurrence is enough to
# make the whole history UNASSESSABLE.
CONTENT_NOT_ADVECTED = re.compile(r"PHASE_CLOT_CONTENT_NOT_ADVECTED\b")
# S6c-2: equivMax, equivL1 and chiVolume are GONE. They were the Stage-1
# equivalence test, theta/theta_ref against chiLegacy = max(0.5*(1-c), 0),
# and with c deleted the comparison has no second operand. Rebuilding
# chiLegacy from theta would have made it read identically zero forever --
# a required field that can never fail, which is worse than an absent one.
#
# They are not replaced. What they measured -- the drift between the two
# representations -- is not a property of the surviving one.
CONTENT_REQUIRED = (
    "time", "thetaMin", "thetaMax", "thetaCNeg", "thetaHNeg",
    "contentVolume", "contentChange",
    "contentSourceDt",
)


def parse_content(text: str) -> tuple[list[dict], list[str]]:
    """PHASE_CLOT_CONTENT lines. Returns (records, operator words seen).

    The operator word is captured separately because it is a WORD, not a
    number, so the numeric key/value sweep below drops it -- and a record
    whose operator cannot be read must not be assumed to be `content`.
    """
    out: list[dict] = []
    ops: list[str] = []
    for m in CONTENT_LINE.finditer(text):
        kv = m.group("kv")
        if kv.startswith("_INIT"):
            # PHASE_CLOT_CONTENT_INIT is the seeding line, not a step record.
            # This is the ONLY legitimate skip.
            continue
        # A CONTENT LINE WITH NO time= IS CORRUPTION, NOT AN INITIALISER.
        # Codex, 2026-09-10: it used to be dropped by the same `continue` as
        # the seeding line, so a truncated record vanished from the history
        # and the remaining good ones still scored REPORTED. It is kept here
        # WITHOUT a time key, so the required-key test marks it malformed and
        # the whole history becomes UNASSESSABLE.
        rec = _numeric_kv(kv)
        mo = re.search(r"\boperator=(\w+)(?=\s|$)", kv)
        rec["_operator"] = mo.group(1) if mo else ""
        ops.append(rec["_operator"])
        out.append(rec)
    return out, ops


# ---------------------------------------------------------------------------
# STAGE 2 -- FREE PLATELET NUMBER.
#
# DELIBERATELY NOT `PHASE_CLOT_PLATELET`. That prefix is already taken by the
# legacy Wp/w envelope diagnostic, whose required keys are wMax/wMin/WpDevMax/
# wDevBlood and whose gate fires on WpDevMax and cUnder. Emitting the Stage 2
# record under the same prefix would have fed records with none of those keys
# into that gate, and the two histories would have merged into one list with
# no error anywhere. Caught before the first run; the solver emits
# PHASE_CLOT_FREEPLATELET, which does not contain the literal
# "PHASE_CLOT_PLATELET" as a substring, so neither regex can see the other's
# records.
FREEPLATELET_LINE = re.compile(r"PHASE_CLOT_FREEPLATELET (?P<kv>.*)")
FREEPLATELET_COMMIT_LINE = re.compile(
    r"PHASE_CLOT_FREEPLATELET_COMMIT (?P<kv>.*)"
)
FREEPLATELET_REQUIRED = (
    "time", "step", "attempt", "outer", "phase",
    "nMin", "nMax", "NMin", "NMax", "NNeg", "NVolume", "NChange",
    "NStarVolume", "transportResidual", "boundaryOut", "dBSum", "pairMax",
    "volResMax", "volResL1", "epsMin", "availClamped", "capClamped",
    "Y", "Deff",
)
FREEPLATELET_COUNTERS = (
    "step", "attempt", "outer", "phase", "NNeg", "availClamped", "capClamped"
)
# pairMax and volResMax/volResL1 are built from mag(); dBSum is an integral of
# a quantity the solver clamps to be non-negative; Y and epsMin are refused at
# zero or below by checkPlateletConfiguration() and the eps validity check.
FREEPLATELET_NONNEGATIVE = ("pairMax", "volResMax", "volResL1", "dBSum", "Y")
FREEPLATELET_CONVENTIONS = ("intrinsic", "superficial")


def parse_freeplatelet(text: str) -> tuple[list[dict], list[str]]:
    """PHASE_CLOT_FREEPLATELET step records. Returns (records, conventions)."""
    out: list[dict] = []
    conv: list[str] = []
    for m in FREEPLATELET_LINE.finditer(text):
        kv = m.group("kv")
        if kv.startswith("_"):
            # _CONFIG / _NOTE / _OFF are announcements, not step records.
            continue
        rec = _numeric_kv(kv)
        mo = re.search(r"\bconvention=(\w+)(?=\s|$)", kv)
        rec["_convention"] = mo.group(1) if mo else ""
        mk = re.search(r"\bkinetics=(\w+)(?=\s|$)", kv)
        rec["_kinetics"] = mk.group(1) if mk else ""
        conv.append(rec["_convention"])
        out.append(rec)
    return out, conv


def freeplatelet_record_invalid(r: dict) -> str | None:
    """Schema check for one free-platelet record. None if admissible."""
    missing = [k for k in FREEPLATELET_REQUIRED if k not in r]
    if missing:
        return "missing " + ",".join(missing)
    for k in FREEPLATELET_REQUIRED:
        if not math.isfinite(r[k]):
            return "non-finite " + k
    if not r["_convention"]:
        return "no convention"
    if r["_convention"] not in FREEPLATELET_CONVENTIONS:
        return "unknown convention " + r["_convention"]
    if r["_kinetics"] not in ("on", "off"):
        return "kinetics must be on|off, found %r" % r["_kinetics"]
    for k in FREEPLATELET_COUNTERS:
        if r[k] < 0 or r[k] != int(r[k]):
            return "counter %s = %r is not a non-negative integer" % (k, r[k])
    for k in FREEPLATELET_NONNEGATIVE:
        if r[k] < 0:
            return "%s = %r is negative" % (k, r[k])
    # eps <= 0 is refused in the solver, so seeing it in a record means the
    # record did not come from the emitter this gate was written against.
    if r["epsMin"] <= 0:
        return "epsMin = %r is not positive" % r["epsMin"]
    # A transport-only run must not have moved any content between phases.
    if r["_kinetics"] == "off" and r["dBSum"] != 0:
        return "transport-only record carries dBSum = %r" % r["dBSum"]
    return None


def parse_freeplatelet_commits(text: str) -> list[dict]:
    """Explicit accepted-step markers; absent in pre-transaction logs."""
    out = []
    for match in FREEPLATELET_COMMIT_LINE.finditer(text):
        kv = match.group("kv")
        rec = _numeric_kv(kv)
        status = re.search(r"\bconvergence=(\w+)(?=\s|$)", kv)
        rec["_convergence"] = status.group(1) if status else ""
        out.append(rec)
    return out


def freeplatelet_commit_invalid(r: dict) -> str | None:
    required = (
        "time", "step", "attempt", "outer", "phase", "trials", "committed"
    )
    missing = [key for key in required if key not in r]
    if missing:
        return "missing " + ",".join(missing)
    if not all(math.isfinite(r[key]) for key in required):
        return "non-finite commit value"
    for key in ("step", "attempt", "outer", "phase", "trials", "committed"):
        if r[key] < 0 or r[key] != int(r[key]):
            return "commit %s is not a non-negative integer" % key
    if r["committed"] != 1 or r["trials"] < 1:
        return "commit marker/trial count is invalid"
    if r["phase"] != r["trials"] - 1:
        return "commit does not name the final phase"
    if r["_convergence"] != "fixedPhaseCount":
        return "unknown convergence status %r" % r["_convergence"]
    return None


def freeplatelet_transaction_invalid(text: str) -> str | None:
    """Validate trial/commit cardinality by integer identity, never by time."""
    trials, _ = parse_freeplatelet(text)
    commits = parse_freeplatelet_commits(text)
    if not trials and not commits:
        return "no transaction records"
    for row in trials:
        why = freeplatelet_record_invalid(row)
        if why:
            return "invalid trial: " + why
    for row in commits:
        why = freeplatelet_commit_invalid(row)
        if why:
            return "invalid commit: " + why
    trial_groups = {}
    for row in trials:
        key = tuple(int(row[k]) for k in ("step", "attempt", "outer"))
        trial_groups.setdefault(key, []).append(int(row["phase"]))
    commit_groups = {}
    for row in commits:
        key = tuple(int(row[k]) for k in ("step", "attempt", "outer"))
        commit_groups.setdefault(key, []).append(row)
    if set(trial_groups) != set(commit_groups):
        return "trial and commit identities differ"
    for key, phases in trial_groups.items():
        rows = commit_groups[key]
        if len(rows) != 1:
            return "identity %r has %d commits" % (key, len(rows))
        count = int(rows[0]["trials"])
        if sorted(phases) != list(range(count)):
            return "identity %r phases %r != 0..%d" % (key, phases, count - 1)
    return None


# The only operator words this gate was written against. An unknown word means
# the log came from an emitter this assessor has never seen, and its records
# cannot be scored against Stage 1's contract.
CONTENT_OPERATORS_ALLOWED = ("content", "matched")

# Cardinalities. A negative or fractional count is not a small error in a
# count, it is evidence the field is not a count.
CONTENT_COUNTERS = ("thetaCNeg", "thetaHNeg")

# Error norms. Both are built from mag(...) in the solver, so a negative value
# cannot be produced by the emitter under test.
# S6c-2: equivMax and equivL1 were retired with `c`, and emptying this tuple
# with them DELETED THE RULE instead of moving it. The rule is about keys that
# cannot be negative BY CONSTRUCTION, not about the two that happened to carry
# it: thetaCNeg and thetaHNeg are CELL COUNTS, so a negative one is an emit or
# parse fault and the record is not admissible.
#
# Deliberately NOT here: contentVolume, thetaMin, thetaMax, contentChange,
# contentSourceDt. contentChange and contentSourceDt are signed. The other
# three CAN go negative in a broken run -- that is a physics finding the
# bounds gate and thetaCNeg exist to report, and refusing the record would
# throw away the evidence rather than raise it.
CONTENT_NONNEGATIVE = ("thetaCNeg", "thetaHNeg")


def content_record_invalid(r: dict) -> str | None:
    """Schema check for one content record. None if it is admissible."""
    missing = [k for k in CONTENT_REQUIRED if k not in r]
    if missing:
        return "missing " + ",".join(missing)
    for k in CONTENT_REQUIRED:
        if not math.isfinite(r[k]):
            return "non-finite " + k
    if not r["_operator"]:
        return "no operator"
    if r["_operator"] not in CONTENT_OPERATORS_ALLOWED:
        return "unknown operator " + r["_operator"]
    for k in CONTENT_COUNTERS:
        if r[k] < 0 or r[k] != int(r[k]):
            return "counter %s = %r is not a non-negative integer" % (k, r[k])
    for k in CONTENT_NONNEGATIVE:
        if r[k] < 0:
            return "%s = %r is negative" % (k, r[k])
    return None


PLATELET_LINE = re.compile(r"PHASE_CLOT_PLATELET (?P<kv>.*)")
PLATELET_REQUIRED = ("time", "wMax", "wMin", "WpDevMax", "wDevBlood")
PLATELET_RAW_REQUIRED = (
    "WpRawQDevMax", "rawQMin", "wDevRawBlood", "rawBloodCells",
)
PLATELET_V2_REQUIRED = (
    *PLATELET_REQUIRED, "VnMagMax", "wOverBound", "qMin",
    *PLATELET_RAW_REQUIRED,
)

# WpDevMax IS AN ENVELOPE GUARD, NOT A CONSERVATION LAW, AND THE DIFFERENCE
# MATTERS.
#
# WpDevMax is an EFFECTIVE-q reference diagnostic, not necessarily raw q.
# The Batch 25 archived emitter and schema 2 use max|Wp-max(abs(q),qFloor)|.
# With q=Wp=-0.7 it reports 1.4, although the raw-q discrepancy is zero.
# Earlier unversioned emitters need their source identity to establish the
# exact reference. Do not silently apply schema 2's definition to all archives.
# Active uptake/diffusion can change either discrepancy; neither is a general
# conservation law. The historical one-velocity envelope below is retained as
# provenance for the UNCHANGED gross-screen threshold, not as a new calibration:
#
#     g40 0.0612   em_freeG40 0.0716   g30 0.0741   g20 0.0757
#     g10 0.0925   th_hold 0.1087      g04 0.1109   th_break 0.1114
#     em_free 0.1114
#
# and run 20260904-214345 child em_2vel reached 0.9926 -- w driven to ~0, i.e.
# total local platelet depletion, which is pathological on any reading.
#
# 0.25 is 2.2x the widest one-velocity observation and 4.0x below that
# pathology. It is derived from an OBSERVED ENVELOPE, not from a physical
# identity, and it is written down that way so nobody later mistakes it for
# one. It exists because this quantity was being printed every step and read
# by nothing -- the same gap that let a 4x clot growth run 9391 steps.
#
# It would NOT have caught run 20260905-054441 (WpDevMax peaked at 0.047).
# That is fine and expected: the conservation gate catches that one. Two
# instruments, two failures.
WP_DEV_LIMIT = 0.25


def parse_platelet(text: str) -> list[dict[str, float]]:
    """Read legacy or schema-2 metrics without treating missing raw q as zero.

    Schema 2 is an explicit instrument contract: every metric must be finite
    on every record. Unknown versions, truncated tokens and partial upgrades
    raise ValueError, following the existing fail-closed platelet reader.
    Unversioned archives keep their existing parsing/required-key behavior.
    """
    out: list[dict[str, float]] = []
    for m in PLATELET_LINE.finditer(text):
        pairs = re.findall(r"(\w+)=([^\s]+)", m.group("kv"))
        tokens = dict(pairs)
        if re.search(r"\bmetricSchema\b", m.group("kv")):
            try:
                schema = float(tokens.get("metricSchema", ""))
            except ValueError as exc:
                raise ValueError("PHASE_CLOT_PLATELET invalid metricSchema") from exc
            if schema != 2:
                raise ValueError(
                    f"PHASE_CLOT_PLATELET unsupported metricSchema={schema!r}"
                )
            required = ("metricSchema", *PLATELET_V2_REQUIRED)
            duplicates = [k for k in required
                          if sum(name == k for name, _ in pairs) > 1]
            if duplicates:
                raise ValueError(
                    f"PHASE_CLOT_PLATELET duplicate keys {duplicates}"
                )
            kv = _numeric_kv(m.group("kv"))
            nonfinite = [k for k in required
                         if k in kv and not math.isfinite(kv[k])]
            if nonfinite:
                raise ValueError(
                    f"PHASE_CLOT_PLATELET non-finite schema-2 metrics {nonfinite}"
                )
        else:
            if any(re.search(rf"\b{k}\b", m.group("kv"))
                   for k in PLATELET_RAW_REQUIRED):
                raise ValueError(
                    "PHASE_CLOT_PLATELET raw-q metrics require metricSchema=2"
                )
            required = PLATELET_REQUIRED
            kv = dict(
                (k, float(v))
                for k, v in re.findall(r"(\w+)=(-?[\d.eE+-]+)", m.group("kv"))
            )
        missing = [k for k in required if k not in kv]
        if missing:
            raise ValueError(
                f"PHASE_CLOT_PLATELET is missing {missing}; the platelet "
                f"envelope guard would silently pass. Line: {m.group(0)[:200]}"
            )
        if kv.get("metricSchema") == 2:
            for key in ("WpDevMax", "wDevBlood", "VnMagMax", "qMin",
                        "WpRawQDevMax", "wDevRawBlood"):
                if kv[key] < 0:
                    raise ValueError(f"PHASE_CLOT_PLATELET negative magnitude {key}")
            if kv["wOverBound"] < 0 or not kv["wOverBound"].is_integer():
                raise ValueError("PHASE_CLOT_PLATELET invalid wOverBound counter")
            cells = kv["rawBloodCells"]
            if cells < 0 or not cells.is_integer():
                raise ValueError(
                    "PHASE_CLOT_PLATELET rawBloodCells must be a nonnegative integer"
                )
            if cells == 0 and kv["wDevRawBlood"] != 0:
                raise ValueError(
                    "PHASE_CLOT_PLATELET nonzero wDevRawBlood for an empty subset"
                )
        out.append(kv)
    return out


def platelet_metric_summary(records: list[dict[str, float]]) -> dict:
    """Name references and coverage; never promote a partial raw-q history.

    Legacy WpDevMax remains the existing gate input. New raw metrics are
    reported only, with no new threshold. Mixed histories retain the legacy
    envelope while withholding a misleading whole-history raw-q peak.
    """
    v2_count = sum(r.get("metricSchema") == 2 for r in records)
    complete = bool(records) and v2_count == len(records)
    status = ("NO_RECORDS" if not records else "SCHEMA_2_COMPLETE" if complete
              else "LEGACY_UNVERSIONED" if v2_count == 0 else "MIXED_SCHEMAS")
    raw = records if complete else []
    raw_blood = [r for r in raw if r["rawBloodCells"] > 0]
    return {
        "platelet_metric_schema_status": status,
        "platelet_metric_schema": 2 if complete else None,
        "platelet_metric_records": len(records),
        "platelet_metric_schema2_records": v2_count,
        "platelet_metric_legacy_records": len(records) - v2_count,
        "platelet_raw_metrics_available": complete,
        "wp_dev_reference": (
            "max_cells(abs(Wp - max(abs(q), qFloor)))" if complete else
            "Unversioned/mixed WpDevMax: exact effective-q reference requires "
            "the archived emitter source; not certified as raw-q discrepancy."
        ),
        "platelet_q_min_reference": (
            "min_cells(max(abs(q), qFloor)); not raw q" if complete else
            "Unversioned/mixed qMin; exact reference requires emitter source."
        ),
        "platelet_w_dev_blood_reference": (
            "max(abs(w-1)) over max(abs(q),qFloor)>=0.5; empty subset gives 0"
            if complete else "Unversioned/mixed wDevBlood; subset requires emitter source."
        ),
        "wp_raw_q_dev_peak": max((r["WpRawQDevMax"] for r in raw), default=None),
        "platelet_raw_q_min": min((r["rawQMin"] for r in raw), default=None),
        "platelet_w_dev_raw_blood_peak": max(
            (r["wDevRawBlood"] for r in raw_blood), default=None
        ),
        "platelet_raw_blood_records_with_cells": len(raw_blood) if complete else None,
        "platelet_raw_blood_empty_records": (
            len(raw) - len(raw_blood) if complete else None
        ),
        "platelet_raw_blood_cells_min": min(
            (int(r["rawBloodCells"]) for r in raw), default=None
        ),
        "platelet_raw_blood_cells_max": max(
            (int(r["rawBloodCells"]) for r in raw), default=None
        ),
        "platelet_raw_blood_subset_coverage": (
            "UNAVAILABLE" if not complete else "ALL_EMPTY" if not raw_blood
            else "ALL_RECORDS_NONEMPTY" if len(raw_blood) == len(raw)
            else "SOME_EMPTY_RECORDS"
        ),
        "platelet_raw_metric_definitions": {
            "WpRawQDevMax": "max_cells(abs(Wp - q)); raw q=(c+1)/2",
            "rawQMin": "min_cells(q); raw q=(c+1)/2",
            "wDevRawBlood": "max(abs(w-1)) over raw q>=0.5; empty subset gives 0",
            "rawBloodCells": "number of cells with raw q>=0.5",
        } if complete else None,
    }


PERMEABILITY_LINE = re.compile(r"PHASE_CLOT_PERMEABILITY (?P<kv>.*)")
PERMEABILITY_REQUIRED = ("clotCells", "kClotMin", "kClotMax", "flooredCells",
                         "floor")


def parse_permeability(text: str) -> list[dict[str, float]]:
    """The realised clot permeability, from the solver rather than from k0p.

    Absent from every log written before 2026-08-14, so a missing line is not
    an error -- it returns an empty list and the report says None. A line that
    is PRESENT but has lost a field is an error, for the same reason
    parse_body_velocity refuses one: a silently disabled check reads exactly
    like a passing one.
    """
    out = []
    for m in PERMEABILITY_LINE.finditer(text):
        kv = dict(
            (k, float(v))
            for k, v in re.findall(r"(\w+)=(-?[\d.eE+-]+)", m.group("kv"))
        )
        missing = [k for k in PERMEABILITY_REQUIRED if k not in kv]
        if missing:
            raise ValueError(
                f"PHASE_CLOT_PERMEABILITY is missing {missing}; the realised "
                f"clot permeability would silently report None. Line: "
                + m.group(0)[:200]
            )
        out.append(kv)
    return out


# Key-based, for the same reason `parse_debond` is. The positional version of
# this regex stopped matching the moment `UyDiscarded` and `crossFraction` were
# inserted between `Uy` and `bandCells`, and run 20260813-084207 reported
# body_velocity_m_s = null, free_stream_m_s = null and
# body_velocity_plausible = null -- indistinguishable from "the clot never
# moved", on a run where it moved 49 cells. Adding a field must never silently
# disable a check; REMOVING one must be loud.
BODY_VELOCITY_REQUIRED = ("Ux", "bandCells", "freeStreamUx", "ratioToFreeStream")


def parse_body_velocity(text: str) -> list[dict[str, float]]:
    out = []
    for m in BODY_VELOCITY_LINE.finditer(text):
        kv = dict(
            (k, float(v))
            for k, v in re.findall(r"(\w+)=(-?[\d.eE+-]+)", m.group("kv"))
        )
        missing = [k for k in BODY_VELOCITY_REQUIRED if k not in kv]
        if missing:
            raise ValueError(
                "PHASE_CLOT_BODY_VELOCITY is missing "
                f"{missing}; the solver's log format changed and the transport "
                "plausibility check would silently report None. Line: "
                + m.group(0)[:200]
            )
        out.append(kv)
    return out
# S21 (research/designs/2026-10-05-S21-clean-solver.md section 1.2; docket 20261005-1621,
# reports 06, 09, 10): the options the solver now runs built in. Their CONFIG echoes and
# child.json members are gone (S11_CONFIG, S17B_CONFIG and PHASE_WEIGHT_CONFIG are not
# printed at all), so the assessor reads the built-in value. A log from an older binary
# that still prints the key is read as printed.
S21_BUILT_IN = {
    "networkRupture": "fibreRelease", "elasticCoupling": "implicitTangent",
    "beStretchGradient": "fluxConsistent", "beDetRenormalise": "yes",
    "fibreSlack": "v1", "fibreArithmetic": "fieldUnits", "skeletonWallFace": "zeroed",
    "frozenPendingPressure": "none", "wallPatchSplit": "split", "cplusOrder": "shared",
    "blockCoupledLumen": "eliminate", "contentTransportScheme": "fct",
    "dragOccupancyBasis": "intrinsic", "skeletonFloorBasis": "occupancy",
}
# child.json members the generator no longer writes (built in): the value each kept child had.
S21_BUILT_IN_SPEC = {
    "advect_released": True, "permeability_model": "kozenyCarmanSolid",
    "skeleton_diffuse_domain": True, "material_carrier_blend": False,
}


G3_LAMBDA_MEDIAN_MIN = 0.1     # S20 section 2.3: 1e5 above the printed resolution


def s21_config(config: dict | None, key: str):
    """A CONFIG key: as printed, else its S21 built-in value; None when the line itself is
    absent (no such line in the log, which is never a pass)."""
    if config is None:
        return None
    return config.get(key, S21_BUILT_IN[key])


def s21_spec(spec: dict, key: str):
    """A child.json member S21 removed: as written, else the built-in value."""
    return spec.get(key, S21_BUILT_IN_SPEC[key])


def s10_report(text: str, g_pa: float | None) -> dict:
    """The pre-registered S10 quantities, READ so they are not unread lines.

    Reported, NOT gated on admissibility -- EXCEPT g3_lambda, which
    assess_child ANDs into "admissible" (False gates; None is N/A) -- the gates are pre-registered in
    em51's expectation and judged in its run record. `None` means the solver
    did not emit the field (an older binary) -- never a pass.

    G1 (research/designs/2026-09-22-S10-root-causes-em49.md sec. 1): the wall
    traction and debonds at the FIRST record. em49 read peakTractionAny
    2.8996 Pa and 24 debonded faces on step 1, from the network pressure's
    tension on the seeded interface; the bound is 1e-10*G.
    G4 (sec. 4): the largest flux on a face whose U is prescribed, and the net
    through those patches. Exactly 0 when the wall constraint holds; em49's
    bottom wall carried -1.42e-14 m^3/s.
    """
    def kv(line):
        return {k: float(v) for k, v in
                re.findall(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?)", line)}

    out: dict = {"config": None}
    cfg = re.search(r"^PHASE_CLOT_S10_CONFIG (.*)$", text, re.M)
    if cfg:
        out["config"] = dict(re.findall(r"(\w+)=(\S+)", cfg.group(1)))

    first_debond = re.search(r"^PHASE_CLOT_DEBOND (.*)$", text, re.M)
    d = kv(first_debond.group(1)) if first_debond else {}
    out["step1_peak_traction_any_pa"] = d.get("peakTractionAny")
    out["step1_debonded_faces"] = d.get("debondedFaces")
    first_net = re.search(r"^PHASE_CLOT_NETWORK (.*)$", text, re.M)
    n = kv(first_net.group(1)) if first_net else {}
    out["step1_pn_min_pa"] = n.get("pnMin")
    out["step1_pn_max_pa"] = n.get("pnMax")
    bound = 1e-10 * g_pa if g_pa else None
    out["g1_bound_pa"] = bound
    out["g1_step1_traction_free"] = (
        None if bound is None or d.get("peakTractionAny") is None
        else bool(d["peakTractionAny"] < bound
                  and d.get("debondedFaces", 1) == 0))

    flux = [kv(m) for m in
            re.findall(r"^PHASE_CLOT_FLUX_DECOMPOSITION (.*)$", text, re.M)]
    fmax = [f["prescribedFluxMax"] for f in flux if "prescribedFluxMax" in f]
    fnet = [abs(f["prescribedFluxNet"]) for f in flux
            if "prescribedFluxNet" in f]
    out["prescribed_flux_max_m3_s"] = max(fmax) if fmax else None
    out["prescribed_flux_net_max_m3_s"] = max(fnet) if fnet else None
    out["prescribed_flux_records"] = len(fmax)
    out["g4_walls_impermeable"] = (
        None if not fmax
        else bool(max(fmax) <= 1e-25 and max(fnet) <= 1e-22))

    arg_max = [kv(m).get("pnArgMax") for m in
               re.findall(r"^PHASE_CLOT_NETWORK (.*)$", text, re.M)]
    arg_max = [a for a in arg_max if a is not None]
    out["pn_argument_max"] = max(arg_max) if arg_max else None
    out["body_velocity_lines"] = len(
        re.findall(r"^PHASE_CLOT_BODY_VELOCITY ", text, re.M))

    # G3' (fix 3, contentTransportScheme fct): negAfter must be 0 at EVERY
    # record -- run 20260923-012704 died because limitedLinear let it go
    # negative. lambdaFluxMean says how much second-order flux was admitted.
    fct = [kv(m) for m in
           re.findall(r"^PHASE_CLOT_CONTENT_FCT (.*)$", text, re.M)]
    out["fct_records"] = len(fct)
    out["fct_neg_after_max"] = (
        max(int(f.get("negAfter", 0)) for f in fct) if fct else None)
    lam = [f["lambdaFluxMean"] for f in fct if "lambdaFluxMean" in f]
    out["fct_lambda_flux_mean"] = (sum(lam) / len(lam)) if lam else None
    out["g3_content_nonnegative"] = (
        None if not fct else out["fct_neg_after_max"] == 0)

    # G3'lambda (research/designs/2026-10-05-S20-content-fct-neutral-zero-flux.md section
    # 2.3): over the records with activeFaces > 0, the median lambdaFluxMean must be
    # >= 0.1. It applies only to S20 logs, whose FCT lines carry zeroFluxRule=exactZero;
    # an older log (no zeroFluxRule on any FCT line) reports N/A. A failing run is
    # batch 68's fsf_G4 (lambdaFluxMean 0 on every step); the loose replay reads 0.99-1.0.
    rules = re.findall(r"^PHASE_CLOT_CONTENT_FCT .*?\bzeroFluxRule=(\S+)", text, re.M)
    out["fct_zero_flux_rule"] = sorted(set(rules)) or None
    out["g3_lambda_min_median"] = G3_LAMBDA_MEDIAN_MIN
    active_lam = [f["lambdaFluxMean"] for f in fct
                  if f.get("activeFaces", 0) > 0 and "lambdaFluxMean" in f]
    out["fct_active_records"] = len(active_lam)
    if not rules:
        out["g3_lambda_median"] = None
        out["g3_lambda"] = None
        out["g3_lambda_note"] = "N/A: no zeroFluxRule on the FCT lines (a pre-S20 log)"
    else:
        srt = sorted(active_lam)
        k = len(srt)
        med = (None if not k else srt[k // 2] if k % 2 else 0.5*(srt[k//2 - 1] + srt[k//2]))
        out["g3_lambda_median"] = med
        everywhere = len(rules) == len(fct) and set(rules) == {"exactZero"}
        out["g3_lambda"] = (None if med is None
                            else bool(everywhere and med >= G3_LAMBDA_MEDIAN_MIN))
        out["g3_lambda_note"] = (
            "no FCT record has activeFaces > 0" if med is None
            else None if everywhere
            else f"zeroFluxRule is not exactZero on every FCT line ({len(rules)} of "
                 f"{len(fct)} carry it: {sorted(set(rules))})")
    return out


def s11_report(text: str) -> dict:
    """The pre-registered S11 quantities (em53/em54), READ from the log.

    research/designs/2026-09-23-S11-root-causes-em52.md. Thresholds were
    calibrated on em52's own fields so that em52 FAILS each: voids 12 at
    2 ms; argHighPassRms 2.7e-03 at 1 ms; fringe high-pass 2.3e-03 at 1 ms;
    dilute-cell Vs 0.18 m/s at 1.53 ms. Reported, not gated on
    admissibility -- judged in the run record against the pre-registration.
    `None` means the solver did not emit the line (an older binary), which
    is never a pass.
    """
    def kv(line):
        return {k: float(v) for k, v in
                re.findall(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?)", line)}

    def series(tag):
        return [kv(m) for m in re.findall(rf"^{tag} (.*)$", text, re.M)]

    # S21 (docket 20261005-1621 report 10): PHASE_CLOT_S11_CONFIG is no longer printed;
    # every key it echoed is built in. The per-step S11 lines below are unchanged.
    out: dict = {"config": None,
                 "config_note": "S21: no S11_CONFIG line (its keys are built in)"}

    def peak(rows, key):
        vals = [r[key] for r in rows if key in r]
        return (max(vals) if vals else None), len(vals)

    st = series("PHASE_CLOT_STRIPES")
    out["stripes_records"] = len(st)
    out["void_cells_max"], _ = peak(st, "voidCells")
    out["arg_high_pass_rms_max"], _ = peak(st, "argHighPassRms")
    out["fringe_theta_high_pass_rms_max"], _ = peak(st, "fringeThetaHighPassRms")
    out["network_diffusion_number_max"], _ = peak(st, "networkDiffusionNumberMax")
    out["elastic_diffusion_number_max"], _ = peak(st, "elasticDiffusionNumberMax")
    fb = [r["fringeBeLambdaMaxMean"] for r in st if "fringeBeLambdaMaxMean" in r]
    out["fringe_be_lambda_max_mean_last"] = fb[-1] if fb else None
    fo = [r["fringeOverBodySpeed"] for r in st if "fringeOverBodySpeed" in r]
    out["fringe_over_body_speed_max"] = max(fo) if fo else None

    vsp = series("PHASE_CLOT_VS_SPLIT")
    out["vs_split_records"] = len(vsp)
    out["dilute_slip_max_m_s"], _ = peak(vsp, "slip")
    fsk = series("PHASE_CLOT_FACE_SKELETON")
    out["face_skeleton_records"] = len(fsk)
    out["max_face_vs_m_s"], _ = peak(fsk, "maxFaceVs")

    def gate(value, bound):
        return None if value is None else bool(value <= bound)

    out["g5_no_voids"] = (None if out["void_cells_max"] is None
                          else out["void_cells_max"] == 0)
    out["g6_no_compaction_stripes"] = gate(out["arg_high_pass_rms_max"], 1.0e-3)
    out["g7_fringe_smooth"] = gate(out["fringe_theta_high_pass_rms_max"], 2.0e-3)
    out["g8_no_runaway"] = (
        None if out["dilute_slip_max_m_s"] is None
        or out["max_face_vs_m_s"] is None
        else bool(out["dilute_slip_max_m_s"] <= 0.1
                  and out["max_face_vs_m_s"] <= 0.1))
    return out


def s12_report(text: str) -> dict:
    """The pre-registered S12 quantities (em55/em56), READ from the log.

    research/designs/2026-09-23-S12-intrinsic-occupancy-drag.md section 9.
    G10: PHASE_CLOT_FRINGE fringeOverBodySlipRms <= 8 at every record --
    em53/em54's own fields give 12.5-38.8 on this expression, the static
    price under intrinsic drag 1.4-4.2. G11: fringeDragOverIntrinsic = 1
    within 1e-9 at every record under the intrinsic basis (legacy reads
    ~0.099, the k1 floor against the shell's 1.008e7). A ratio printed as
    -1 means the body had no slip yet (step 1) or the intrinsic law is not
    defined in that configuration; those records are COUNTED and excluded
    from the maximum, never read as a pass. `None` means the solver did not
    emit the line (an older binary), which is never a pass either.
    """
    def kv(line):
        return {k: float(v) for k, v in
                re.findall(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?)", line)}

    out: dict = {"config": None}
    cfg = re.search(r"^PHASE_CLOT_S12_CONFIG (.*)$", text, re.M)
    if cfg:
        out["config"] = dict(re.findall(r"(\w+)=(\S+)", cfg.group(1)))
    basis = re.findall(r"^PHASE_CLOT_FRINGE .*?\bbasis=(\w+)", text, re.M)
    rows = [kv(m) for m in re.findall(r"^PHASE_CLOT_FRINGE (.*)$", text, re.M)]
    out["fringe_records"] = len(rows)
    out["basis"] = basis[-1] if basis else None

    ratio = [r["fringeOverBodySlipRms"] for r in rows
             if "fringeOverBodySlipRms" in r]
    good = [v for v in ratio if v >= 0]
    out["fringe_over_body_slip_rms_max"] = max(good) if good else None
    out["fringe_over_body_slip_rms_undefined_records"] = len(ratio) - len(good)
    fmax = [r["fringeSlipMax"] for r in rows if "fringeSlipMax" in r]
    out["fringe_slip_max_m_s"] = max(fmax) if fmax else None

    drag = [r["fringeDragOverIntrinsic"] for r in rows
            if "fringeDragOverIntrinsic" in r]
    dgood = [v for v in drag if v >= 0]
    out["fringe_drag_over_intrinsic_min"] = min(dgood) if dgood else None
    out["fringe_drag_over_intrinsic_max"] = max(dgood) if dgood else None

    out["g10_fringe_follows_body"] = (
        None if out["fringe_over_body_slip_rms_max"] is None
        else bool(out["fringe_over_body_slip_rms_max"] <= 8.0))
    out["g11_drag_is_intrinsic"] = (
        None if not dgood or out["basis"] != "intrinsic"
        else bool(all(abs(v - 1.0) <= 1e-9 for v in dgood)
                  and len(dgood) == len(drag)))
    return out


def s13_report(text: str) -> dict:
    """The S13 configuration (em57/em58), READ from the log.

    research/designs/2026-09-23-S13-grid-scale-restoring.md. S13's gates are
    the existing G6 (argHighPassRms), G7 (fringeThetaHighPassRms), G8, G9 and
    G10, reported by s11_report() and s12_report(); what this adds is the
    configuration line, so a returned run shows WHICH grid-scale fixes were
    active rather than the run record inferring it. `None` means the solver
    did not emit the line (an older binary).
    """
    cfg = re.search(r"^PHASE_CLOT_S13_CONFIG (.*)$", text, re.M)
    out: dict = {"config": dict(re.findall(r"(\w+)=(\S+)", cfg.group(1)))
                 if cfg else None}
    c = out["config"] or {}
    out["presence_weight"] = (None if not c
                              else c.get("networkPressureWeight") == "networkPresence")
    out["flux_consistent_stretch"] = (None if not c
                                      else s21_config(c, "beStretchGradient") == "fluxConsistent")
    return out


def s14_report(text: str) -> dict:
    """The S14 quantities (em59/em60), READ from the log.

    research/designs/2026-09-23-S14-network-rupture.md. PHASE_CLOT_RUPTURE
    reports each step: cells newly ruptured (all, and bulk ind >= 0.5), the
    ruptured share of the clot's content, the largest principal stretch of
    the network still intact, the largest stretch among cells rupturing that
    step, and the label's range.
    - maxIntactStretch is REPORTED, not gated: the solver collects it only
      from cells below lambda_c, so it cannot fail.
    - G12 is maxRupturedStretch <= 1.1 lambda_c at every record: how far past
      lambda_c the stretch got before the criterion acted. One step at the
      wall shear (~1e4 1/s) adds 0.5-1.5% of stretch; em58's tail went
      2.6 -> 5 -> 164 in consecutive steps (1.8x at the first).
    - G14 is 0 <= ruptured <= 1 within 1e-9 (the label solve's tolerance is
      1e-12): 1 - ruptured weights the stress, the BSD and the dt bound.
    `None` means the solver did not emit the line (an older binary, or the
    model off).
    """
    def kv(line):
        return {k: float(v) for k, v in
                re.findall(r"(\w+)=(-?[\d.]+(?:[eE][-+]?\d+)?)", line)}

    cfg = re.search(r"^PHASE_CLOT_S14_CONFIG (.*)$", text, re.M)
    out: dict = {"config": dict(re.findall(r"(\w+)=(\S+)", cfg.group(1)))
                 if cfg else None}
    rows = [kv(m) for m in re.findall(r"^PHASE_CLOT_RUPTURE (.*)$", text, re.M)]
    out["rupture_records"] = len(rows)
    if not rows:
        out.update({"ruptured_cells_total": None, "ruptured_bulk_cells_total": None,
                    "ruptured_content_fraction_last": None,
                    "ruptured_content_fraction_max": None,
                    "max_intact_stretch": None, "critical_stretch": None,
                    "intact_below_critical": None,
                    "max_ruptured_stretch": None, "ruptured_min": None,
                    "ruptured_max": None,
                    "g12_rupture_resolves_stretch": None,
                    "damage_step_max": None, "max_damage": None,
                    "elastic_number_max": None,
                    "g13_elastic_number_bounded": None,
                    "g14_label_bounded": None,
                    "slip_max_after_rupture_settles": None,
                    "max_face_vs_after_rupture_settles": None,
                    "watch_g8_outside_rupture_recoil": None})
        return out
    out["ruptured_cells_total"] = int(sum(r.get("newlyRuptured", 0) for r in rows))
    out["ruptured_bulk_cells_total"] = int(sum(r.get("newlyRupturedBulk", 0) for r in rows))
    fr = [r["rupturedContentFraction"] for r in rows if "rupturedContentFraction" in r]
    out["ruptured_content_fraction_last"] = fr[-1] if fr else None
    out["ruptured_content_fraction_max"] = max(fr) if fr else None
    ms = [r["maxIntactStretch"] for r in rows if "maxIntactStretch" in r]
    out["max_intact_stretch"] = max(ms) if ms else None
    cs = [r["criticalStretch"] for r in rows if "criticalStretch" in r]
    out["critical_stretch"] = cs[-1] if cs else None
    out["intact_below_critical"] = (
        None if not ms or not cs else bool(max(ms) < cs[-1]))
    rs = [r["maxRupturedStretch"] for r in rows if "maxRupturedStretch" in r]
    out["max_ruptured_stretch"] = max(rs) if rs else None
    lo = [r["rupturedMin"] for r in rows if "rupturedMin" in r]
    hi = [r["rupturedMax"] for r in rows if "rupturedMax" in r]
    out["ruptured_min"] = min(lo) if lo else None
    out["ruptured_max"] = max(hi) if hi else None
    # G13: the elastic stability number, in the TANGENT form once
    # elasticStabilityStiffness is tangent, against the 2-D explicit limit.
    el = [kv(m).get("elasticDiffusionNumberMax") for m in
          re.findall(r"^PHASE_CLOT_STRIPES (.*)$", text, re.M)]
    el = [v for v in el if v is not None]
    out["elastic_number_max"] = max(el) if el else None
    # S15: under progressiveDamage there is no rupture stretch to overshoot;
    # G12 is then the largest one-step rise of d, which must stay below 0.05
    # (resolved loading at the wall shear moves d by <= ~0.02 per step; a
    # stretch jump like em58's 2.6 -> 5 moves it ~0.5).
    mode = s21_config(out["config"], "networkRupture")
    ds = [r["damageStepMax"] for r in rows if "damageStepMax" in r]
    out["damage_step_max"] = max(ds) if ds else None
    out["max_damage"] = out.get("ruptured_max")
    # S16: G12p is NOT APPLICABLE under terminalRelease (14 item 3). There
    # damageStepMax is the law's release per step, <= r/(1 + r) by
    # construction, so a fixed 0.05 is met for dt <= tau/19 = 1.737e-5 s and
    # failed by a correct release above it: the score is fixed by dt, not by
    # the model. G12r (s16_report) is the release law's gate.
    if mode in ("terminalRelease", "fibreRelease"):
        out["g12_rupture_resolves_stretch"] = None
        out["g12_note"] = f"not applicable under {mode} (G12r)"
    elif mode == "progressiveDamage":
        out["g12_rupture_resolves_stretch"] = (
            None if not ds else bool(max(ds) <= 0.05))
    else:
        out["g12_rupture_resolves_stretch"] = (
            None if not rs or not cs else bool(max(rs) <= 1.1*cs[-1]))
    out["g14_label_bounded"] = (
        None if not lo or not hi
        else bool(min(lo) >= -1e-9 and max(hi) <= 1.0 + 1e-9))
    out["g13_elastic_number_bounded"] = (
        None if not el else bool(max(el) <= 0.25))
    # S17 C (design 2C): under elasticCoupling implicitTangent the elastic
    # number leaves the dt rate -- the pair carries the tangent implicitly --
    # so G13's explicit limit does not apply. Reported, not gated.
    s17cfg = re.search(r"^PHASE_CLOT_S17_CONFIG (.*)$", text, re.M)
    if s17cfg and s21_config(dict(re.findall(r"(\w+)=(\S+)", s17cfg.group(1))),
                             "elasticCoupling") == "implicitTangent":
        out["g13_elastic_number_bounded"] = None
        out["g13_note"] = ("not applicable under elasticCoupling "
                           "implicitTangent (the tangent is implicit)")
    # WATCH, NOT A GATE (batch 54, run 20260924-014332). Each rupture unloads
    # a neighbour, which recoils at the drag-limited rate, ~0.1-0.2 m/s at
    # G 400: above G8's 0.1 m/s, which was set for runaway before rupture
    # existed. G8 is kept as pre-registered. This reads the same two
    # velocities only on records >= 50 us after the last rupture event.
    # Batch 54 (label transport still unbounded): em60 0.022/0.029, em59
    # 0.0011/0.028; em58's runaway, with no rupture at all, 2.96e4/4.48e4.
    # Calibrated on the run it would judge, so it is reported, and whether to
    # adopt it as G8 for rupture children is Alireza's decision.
    events = [r["time"] for r in rows
              if r.get("newlyRuptured", 0) > 0 and "time" in r]
    quiet = 50e-6

    def settled(t):
        prev = [e for e in events if e <= t]
        return not prev or t - prev[-1] >= quiet

    def quiet_max(tag, key):
        v = [kv(m) for m in re.findall(r"^%s (.*)$" % tag, text, re.M)]
        v = [r[key] for r in v if key in r and "time" in r and settled(r["time"])]
        return max(v) if v else None

    out["slip_max_after_rupture_settles"] = quiet_max("PHASE_CLOT_VS_SPLIT", "slip")
    out["max_face_vs_after_rupture_settles"] = quiet_max(
        "PHASE_CLOT_FACE_SKELETON", "maxFaceVs")
    sq, fq = (out["slip_max_after_rupture_settles"],
              out["max_face_vs_after_rupture_settles"])
    out["watch_g8_outside_rupture_recoil"] = (
        None if sq is None or fq is None else bool(sq <= 0.1 and fq <= 0.1))
    return out


def s15_report(text: str) -> dict:
    """The S15 configuration (em61/em62), READ from the log.

    research/designs/2026-09-24-S15-composition-label-and-progressive-damage.md.
    PHASE_CLOT_S15_CONFIG carries the material-label basis and the
    progressive-damage parameters. The damage itself is reported by
    s14_report (same PHASE_CLOT_RUPTURE line, damageStepMax added). `None`
    means an older binary.
    """
    cfg = re.search(r"^PHASE_CLOT_S15_CONFIG (.*)$", text, re.M)
    out: dict = {"config": dict(re.findall(r"(\w+)=(\S+)", cfg.group(1)))
                 if cfg else None}
    c = out["config"] or {}
    out["composition_label"] = (
        None if not c else c.get("materialLabelBasis") == "contentComposition")
    out["progressive_damage"] = (
        None if not c else s21_config(c, "networkRupture") == "progressiveDamage")
    return out


# S16 G12r: the law's release increment against r/(1 + r). 0.1% above the
# bound, for the printed digits (12 significant) and nothing else; a double
# application per corrector gives 2r/(1 + r)^2 = 2.9e-3 against 1.444e-3 at
# em62's r, i.e. x2.
S16_G12R_TOLERANCE = 1.001
# r must equal the step's deltaT/tau: a wrong tau unit moves it by x1000.
S16_RATE_RTOL = 1e-6
# S16m (16 R5): the deposit mirror's max|dP1 - dtheta| over max theta.
# Round-off there is ~1e-16; a skipped mirror adds the whole deposit,
# >= 5.29e-9 in em62 (13b). ~1e4 of margin on each side.
S16M_TOLERANCE = 1e-12


def s16_report(text: str, spec: dict | None = None) -> dict:
    """The S16 quantities (em63/em64/em65), READ from the log.

    research/designs/2026-09-24-S16-timed-terminal-rupture.md section 6, as
    revised after 13, 13b, 14, 16 and 17 (C+).
    - config: PHASE_CLOT_S16_CONFIG present and equal, key by key, to the
      child's own child.json "s16" block (numbers compared as numbers).
    - G12r: at every PHASE_CLOT_S16 record lawReleaseMax <= 1.001 r/(1 + r),
      r on the same line; r equal to the step's deltaT/tau; and no missing
      release -- a record whose predecessor had pendingMax > 0 must release
      something.
    - G12p: NOT APPLICABLE under terminalRelease (its fixed 0.05 is met by
      construction for dt <= tau/19 and failed by a correct release above;
      14 item 3). Reported as None with that reason; G12r is the gate.
    - S16w, two-sided: weightAbsDevMax == 0 (max over the three sites of
      |w_used - max(ind, 1e-4)(1 - d)|) and weightSites == 7 at every record.
    - S16m: mirrorDevMax <= 1e-12 (max|dP1 - dtheta|/max theta at the deposit
      mirror) at every record; a skipped mirror adds the whole deposit.
    - S16p is WITHDRAWN as a gate (13b: no threshold window). The
      accumulated partitionSumDevMax (relative to max theta) is a WATCH.
    - WATCHES, reported and not gated: W-dd (transport-inclusive deltaDMax,
      against em62's 0.0923), H-A1n (dampedStiffMax, ~9 predicted at G 400),
      partitionSumDevMax, and 16 R4's partition diagnostics (partitionMin,
      partitionSubnormalCells, muMin, muMean, foutMax, partitionSumDevAbs),
      and the G8 cell. W-front is measured offline from the fields at return
      (field-forensics), not here.
    - wall budget: stopAtClockTime's "stopping now after writing" line, and
      the simulated time it stopped at (design: truncated after 2.5 ms -> G9
      NOT-EVALUABLE; before 2.5 ms -> a cost failure).
    `None` means the solver did not emit the line (an older binary, or no S16
    switch on), which is never a pass.
    """
    num = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"

    def kv(line):
        return {k: float(v) for k, v in re.findall(r"(\w+)=" + num, line)}

    out: dict = {"config": None}
    cfg = re.search(r"^PHASE_CLOT_S16_CONFIG (.*)$", text, re.M)
    if cfg:
        out["config"] = dict(re.findall(r"(\w+)=(\S+)", cfg.group(1)))
    want = (spec or {}).get("s16")
    out["config_expected"] = want
    if want is None:
        out["config_matches_child"] = None
    elif out["config"] is None:
        out["config_matches_child"] = False
    else:
        bad = []
        for k, v in want.items():
            got = out["config"].get(k)
            if got is None:
                bad.append(k)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    if abs(float(got) - float(v)) > 1e-12*max(1.0, abs(float(v))):
                        bad.append(k)
                except ValueError:
                    bad.append(k)
            elif str(got) != str(v):
                bad.append(k)
        out["config_mismatched_keys"] = bad
        out["config_matches_child"] = not bad

    # PHASE_CLOT_S16 records, each with the deltaT of its own step: OpenFOAM
    # prints "deltaT = X" before "Time = T", and the record follows.
    rows = []
    dt = None
    for line in text.splitlines():
        if line.startswith("deltaT = "):
            try:
                dt = float(line.split("=", 1)[1])
            except ValueError:
                dt = None
        elif line.startswith("PHASE_CLOT_S16 "):
            r = kv(line)
            r["_deltaT"] = dt
            rows.append(r)
    out["records"] = len(rows)
    keys = ("r", "lawReleaseMax", "lawReleaseBound", "pendingMax",
            "deltaDMax", "dampedStiffMax", "weightAbsDevMax", "weightSites",
            "mirrorDevMax", "partitionMin", "partitionSubnormalCells",
            "muMin", "muMean", "foutMax", "partitionSumDevAbs",
            "partitionSumDevMax")
    terminal_cfg = s21_config(out["config"], "networkRupture") in (
        "terminalRelease", "fibreRelease")
    out["g12p"] = None
    out["g12p_note"] = (
        "not applicable under terminalRelease/fibreRelease (G12r is the "
        "release law's gate)"
        if terminal_cfg else None)
    if not rows:
        out.update({k: None for k in (
            "g12r_release_bounded", "g12r_rate_consistent",
            "g12r_no_missing_release", "g12r", "s16w", "s16m",
            "law_release_max", "watch_delta_d_max", "watch_damped_stiff_max",
            "watch_partition_sum_dev_max", "partition_min",
            "partition_subnormal_cells_max", "mu_min", "mu_mean_last",
            "fout_max", "partition_sum_dev_abs_max")})
    else:
        incomplete = [i for i, r in enumerate(rows) if any(k not in r for k in keys)]
        out["records_incomplete"] = len(incomplete)
        tau = None
        if out["config"] and "networkReleaseTime" in out["config"]:
            try:
                tau = float(out["config"]["networkReleaseTime"])
            except ValueError:
                tau = None
        terminal = s21_config(out["config"], "networkRupture") in (
            "terminalRelease", "fibreRelease")
        rel = [r for r in rows if "r" in r and "lawReleaseMax" in r]
        out["law_release_max"] = max((r["lawReleaseMax"] for r in rel), default=None)
        if terminal:
            over = [r for r in rel
                    if not (r["lawReleaseMax"] <= S16_G12R_TOLERANCE*r["r"]/(1.0 + r["r"]))]
            out["g12r_release_bounded"] = bool(rel) and not over and not incomplete
            out["g12r_first_violation"] = (
                {k: over[0].get(k) for k in ("time", "r", "lawReleaseMax")} if over else None)
            badr = [r for r in rel if r["_deltaT"] is None or tau is None or not (
                abs(r["r"] - r["_deltaT"]/tau) <= S16_RATE_RTOL*abs(r["_deltaT"]/tau))]
            out["g12r_rate_consistent"] = bool(rel) and not badr
            missing = [b for a, b in zip(rel, rel[1:])
                       if a.get("pendingMax", 0) > 0 and not b["lawReleaseMax"] > 0]
            out["g12r_no_missing_release"] = bool(rel) and not missing
            out["g12r"] = bool(out["g12r_release_bounded"]
                               and out["g12r_rate_consistent"]
                               and out["g12r_no_missing_release"])
        else:
            out.update({"g12r_release_bounded": None, "g12r_rate_consistent": None,
                        "g12r_no_missing_release": None, "g12r": None})
        w = [(r.get("weightAbsDevMax"), r.get("weightSites")) for r in rows]
        out["weight_abs_dev_max"] = max((a for a, _ in w if a is not None), default=None)
        out["weight_sites_min"] = min((b for _, b in w if b is not None), default=None)
        # |.| is logged, so a NEGATIVE value is itself a defect (an old
        # one-sided binary, or a sign bug): it fails, never passes.
        out["s16w"] = bool(w) and all(
            a is not None and b is not None and a == 0.0 and int(b) == 7 for a, b in w)
        mv = [r.get("mirrorDevMax") for r in rows]
        out["mirror_dev_max"] = max((v for v in mv if v is not None), default=None)
        out["s16m"] = bool(mv) and all(
            v is not None and 0 <= v <= S16M_TOLERANCE for v in mv)
        out["watch_delta_d_max"] = max((r["deltaDMax"] for r in rows if "deltaDMax" in r), default=None)
        out["watch_delta_d_over_em62"] = (
            None if out["watch_delta_d_max"] is None else out["watch_delta_d_max"]/0.0923)
        out["watch_damped_stiff_max"] = max(
            (r["dampedStiffMax"] for r in rows if "dampedStiffMax" in r), default=None)
        sd = [r["partitionSumDevMax"] for r in rows
              if r.get("partitionSumDevMax", -1) >= 0]
        out["watch_partition_sum_dev_max"] = max(sd) if sd else None
        def col(k, f):
            v = [r[k] for r in rows if k in r and r[k] >= 0]
            return f(v) if v else None
        out["partition_min"] = col("partitionMin", min)
        out["partition_subnormal_cells_max"] = col("partitionSubnormalCells", max)
        out["mu_min"] = col("muMin", min)
        out["mu_mean_last"] = rows[-1].get("muMean")
        out["fout_max"] = col("foutMax", max)
        out["partition_sum_dev_abs_max"] = col("partitionSumDevAbs", max)

    g8 = [kv(m) for m in re.findall(r"^PHASE_CLOT_S16_G8CELL (.*)$", text, re.M)]
    out["g8cell_records"] = len(g8)
    if g8:
        worst = max((r for r in g8 if "slip" in r), key=lambda r: r["slip"], default=None)
        out["g8cell_at_max_slip"] = worst
    else:
        out["g8cell_at_max_slip"] = None
    neg = re.findall(r"^PHASE_CLOT_S16_PARTITION_NEGATIVE (.*)$", text, re.M)
    out["partition_negative"] = neg[:3] if neg else []

    stop = re.search(r"^stopAtClockTime\(timeIndex=(\d+)\): stopping now after writing",
                     text, re.M)
    times = re.findall(r"^Time = (\S+)", text, re.M)
    out["wall_budget_stop"] = bool(stop)
    out["wall_budget_stop_time_s"] = None
    out["wall_budget_verdict"] = None
    if stop:
        try:
            t_stop = float(times[-1].rstrip("s")) if times else None
        except ValueError:
            t_stop = None
        out["wall_budget_stop_time_s"] = t_stop
        if t_stop is not None:
            out["wall_budget_verdict"] = (
                "truncated after 2.5 ms: gates scored to the truncation time, "
                "G9 NOT-EVALUABLE" if t_stop >= 2.5e-3
                else "truncated before 2.5 ms: COST FAILURE (the lag dominated)")
    return out


# ---- S17 (research/designs/2026-09-26-S17-frozen-release-implicit-elastic-
#      startup.md, revision 4, section 6) ---------------------------------
S17_GTR_LIMIT = 1.05          # G-tr: transfer b1/lc^2, read before min(l1, lc)
S17_GROT_TOL = 1e-3           # G-rot: median angle/(dt omega) = 1 +- 0.001,
S17_GROT_LDT = 1e-3           #   over cells with |L|dt <= 1e-3 (redefined after
                              #   verification 17; the solver applies the cut)
S17_GEIG_LIMIT = 1e-12        # G-eig: across the rotation kernel only
S17_P2_LIMIT = 1.05           # P2: s/s_c in every pending cell, ind >= 1e-4
S17_W2_LIMIT = 0.9702         # W2 (watch): T(lc)/s_c, x(1 + 1e-12)
S17_P4_FAIL = 0.18            # P4: the cascade peak (G8-cell slip), m/s
# em66's single step gate (10): sqrt(11708 x 37327), x1.79 either side.
S17_STEP_GATES = {"em66_G400_s17": 20905}
# The discriminators' windows and predictions (10; design section 6):
# (window of step indices, 16-row, 70-row prediction, failing if >= 1).
S17_DISCRIMINATORS = {
    "D1": ((20, 40), 0.765, 2.28, False),
    "D2": ((20, 40), 0.396, 1.34, False),
    "D2b": ((20, 40), 0.416, 0.636, True),
    "D3": ((100, 200), 0.889, 1.318, False),
    "D3b": ((100, 200), 1.188, 1.838, False),
}
S17_SECONDDIFF_FLOOR = 1e-10  # read the ratio only while A_n/||phi_n|| > this


def s17_second_difference(text: str, which: str = "outlet",
                          window: tuple | None = None) -> dict:
    """The two-step amplitude ratio A_n/A_{n-2} of PHASE_CLOT_S17_SECONDDIFF
    (outlet: the mixture flux over the outlet faces; surface: the skeleton
    flux over the clot-surface faces), read only while A_n/||phi_n|| >
    1e-10 (below that the ratio is round-off; 10). Median over `window`
    (inclusive step indices) or over every readable step."""
    num = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"
    rows = {}
    for m in re.findall(r"^PHASE_CLOT_S17_SECONDDIFF (.*)$", text, re.M):
        r = {k: float(v) for k, v in re.findall(r"(\w+)=" + num, m)}
        if r.get("valid") == 1 and "step" in r:
            rows[int(r["step"])] = r
    a_key, n_key = f"{which}SecondDiff", f"{which}PhiNorm"
    ratios = []
    ordered = []
    for n in sorted(rows):
        if window and not (window[0] <= n <= window[1]):
            continue
        r, r2 = rows[n], rows.get(n - 2)
        if r2 is None:
            continue
        a, a2, nrm = r.get(a_key), r2.get(a_key), r.get(n_key)
        if a is None or a2 is None or not nrm or not a2:
            continue
        if a/nrm > S17_SECONDDIFF_FLOOR:
            ratios.append(a/a2)
            ordered.append(a/a2)
    ratios.sort()
    # "throughout": the median of every consecutive run of 20 readable
    # steps (a single two-step ratio fluctuates; 20 steps is 10's window).
    chunks = [sorted(ordered[i:i + 20]) for i in range(0, len(ordered), 20)]
    chunk_medians = [c[len(c)//2] for c in chunks if len(c) >= 10]
    return {
        "records": len(rows),
        "readable": len(ratios),
        "median_ratio": ratios[len(ratios)//2] if ratios else None,
        "chunk_median_max": max(chunk_medians) if chunk_medians else None,
        "chunks": len(chunk_medians),
        "window": window,
    }


def s17_report(text: str, spec: dict | None = None) -> dict:
    """The S17 quantities, READ from the log (design section 6).
    - config: PHASE_CLOT_S17_CONFIG equal, key by key, to child.json's "s17"
      block (networkRupture, elasticCoupling, beDetRenormalise, pendingForm).
    - G-tr: max transferB1Max <= 1.05 over every record with a transfer;
      exceeding it FAILS the run (min(l1, lc) is the event value only while
      the transfer is resolved; 11 (b)). None before any transfer.
    - G-rot (redefined after verification 17, design section 6): the
      median over records with rotCells > 0 of the per-step median
      angle/(dt omega), read by the solver only over cells with |L|dt <=
      1e-3, in 1 +- 0.001; rotLdtMedian (the candidates' median |L|dt) is
      reported beside it. NOT-EVALUABLE (None) if no record has a qualifying
      cell. A double application is caught by the guard's static call count
      (each call overwrites the median, so no runtime "reads 2" exists).
    - G-eig: eigRelMax <= 1e-12 at every record (across the rotation only).
    - G-uni: uniUp == 0 at every record (across the rotation-and-stretch
      kernel; P-LS may lower lf).
    - G-proj: projRaised == 0 at every record.
    - S17split: splitDevMax0/1/2 == 0 and splitSites == 7 at every record.
    - P2: p2Max <= 1.05 at every record (the log reads every step; the
      offline check re-reads the written fields).
    - WATCHES: W1 (w1MaxDeg, w1Over5, w1Over10: classifies a G8 exceedance),
      W2 (w2Max against 0.9702), resetCells, pendingPressureMax, projTies.
    - Steps (em66 only): <= 20905 steps to 5 ms.
    - Discriminators (s17.discriminator): the outlet two-step ratio in the
      window of 10; D2b >= 1 means B does not ship. CB6/CB4: the surface
      ratio over every readable step; CB6 >= 1 fails.
    - Start-up (em66/em67): the net outlet flux reaches 90% of its own
      plateau (mean over 0.5-1.0 ms; em67 0.3-0.8 ms) by 0.2 ms, and the
      outlet second-difference ratio stays < 1 throughout.
    - Onset (em67): the first transfer in [0.9, 1.6] ms.
    `None` means the line was not emitted (never a pass)."""
    num = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"

    def kv(line):
        return {k: float(v) for k, v in re.findall(r"(\w+)=" + num, line)}

    spec = spec or {}
    want = spec.get("s17")
    out: dict = {"config": None, "config_expected": want}
    cfg = re.search(r"^PHASE_CLOT_S17_CONFIG (.*)$", text, re.M)
    if cfg:
        out["config"] = dict(re.findall(r"(\w+)=(\S+)", cfg.group(1)))
    if want is None:
        out["config_matches_child"] = None
    elif out["config"] is None:
        out["config_matches_child"] = False
    else:
        # S21: networkRupture, elasticCoupling and beDetRenormalise are built in; the
        # line and child.json carry pendingForm only.
        bad = [k for k in ("pendingForm",)
               if str(out["config"].get(k)) != str(want.get(k))]
        out["config_mismatched_keys"] = bad
        out["config_matches_child"] = not bad
    fibre = (out["config"] or {}).get("pendingForm") == "fibre"

    rows = [kv(m) for m in re.findall(r"^PHASE_CLOT_S17 (.*)$", text, re.M)]
    out["records"] = len(rows)
    keys = ("transferB1Max", "transfers", "rotRatioMedian", "rotCells",
            "rotLdtMedian", "rotCandidates",
            "eigRelMax", "uniUp", "projRaised", "splitDevMax0",
            "splitDevMax1", "splitDevMax2", "splitSites", "p2Max", "w2Max",
            "w1MaxDeg", "w1Over5", "w1Over10")
    gates = ("g_tr", "g_rot", "g_eig", "g_uni", "g_proj", "s17split", "p2")
    out["first_transfer_time_s"] = None
    if not rows or not fibre:
        out.update({g: None for g in gates})
    else:
        incomplete = [i for i, r in enumerate(rows) if any(k not in r for k in keys)]
        out["records_incomplete"] = len(incomplete)
        tr = [r for r in rows if r.get("transfers", 0) > 0]
        out["transfer_b1_max"] = max((r["transferB1Max"] for r in tr), default=None)
        out["g_tr"] = (None if not tr else
                       bool(not incomplete and out["transfer_b1_max"] <= S17_GTR_LIMIT))
        # S17b (design section 2): G-tr becomes a WATCH on non-physical line
        # rates; min(l1, lc) is kept. Reported, not gated.
        if "s17b" in spec:
            out["g_tr_watch"] = out["g_tr"]
            out["g_tr"] = None
            out["g_tr_note"] = "WATCH under S17b"
        out["g_tr_first_violation"] = next(
            ({"time": r.get("time"), "transferB1Max": r["transferB1Max"]}
             for r in tr if r["transferB1Max"] > S17_GTR_LIMIT), None)
        rot = sorted(r["rotRatioMedian"] for r in rows if r.get("rotCells", 0) > 0)
        out["rot_ratio_median"] = rot[len(rot)//2] if rot else None
        out["rot_ratio_range"] = [rot[0], rot[-1]] if rot else None
        ldt = sorted(r["rotLdtMedian"] for r in rows
                     if r.get("rotCandidates", 0) > 0 and "rotLdtMedian" in r)
        out["rot_ldt_median"] = ldt[len(ldt)//2] if ldt else None
        out["rot_records_evaluable"] = len(rot)
        out["g_rot"] = (None if not rot else
                        bool(abs(out["rot_ratio_median"] - 1.0) <= S17_GROT_TOL))
        out["g_rot_note"] = (None if rot else
                             "NOT-EVALUABLE: no cell with |L|dt <= 1e-3 in any record")
        eig = [r["eigRelMax"] for r in rows if "eigRelMax" in r]
        out["eig_rel_max"] = max(eig) if eig else None
        out["g_eig"] = bool(eig) and not incomplete and all(
            0 <= v <= S17_GEIG_LIMIT for v in eig)
        out["uni_up_total"] = sum(int(r.get("uniUp", 0)) for r in rows)
        out["g_uni"] = not incomplete and out["uni_up_total"] == 0
        out["proj_raised_total"] = sum(int(r.get("projRaised", 0)) for r in rows)
        out["proj_count_total"] = sum(int(r.get("projCount", 0)) for r in rows)
        out["g_proj"] = not incomplete and out["proj_raised_total"] == 0
        sp = [(r.get("splitDevMax0"), r.get("splitDevMax1"),
               r.get("splitDevMax2"), r.get("splitSites")) for r in rows]
        out["split_dev_max"] = [max((v[i] for v in sp if v[i] is not None), default=None)
                                for i in range(3)]
        out["s17split"] = bool(sp) and all(
            None not in v and v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0
            and int(v[3]) == 7 for v in sp)
        p2 = [r["p2Max"] for r in rows if "p2Max" in r]
        out["p2_max"] = max(p2) if p2 else None
        out["p2"] = bool(p2) and not incomplete and all(v <= S17_P2_LIMIT for v in p2)
        w2 = [r["w2Max"] for r in rows if "w2Max" in r]
        out["watch_w2_max"] = max(w2) if w2 else None
        out["watch_w2_over"] = (None if not w2 else
                                bool(max(w2) > S17_W2_LIMIT*(1 + 1e-12)))
        out["watch_w1_max_deg"] = max((r.get("w1MaxDeg", 0) for r in rows), default=None)
        out["watch_w1_over5_max"] = max((int(r.get("w1Over5", 0)) for r in rows), default=None)
        out["watch_w1_over10_max"] = max((int(r.get("w1Over10", 0)) for r in rows), default=None)
        out["watch_reset_cells_total"] = sum(int(r.get("resetCells", 0)) for r in rows)
        out["watch_pending_pressure_max"] = max(
            (r.get("pendingPressureMax", 0) for r in rows), default=None)
        out["watch_proj_ties_total"] = sum(int(r.get("projTies", 0)) for r in rows)
        first = next((r for r in tr), None)
        out["first_transfer_time_s"] = first.get("time") if first else None

    # em66's step gate
    cid = spec.get("child_id")
    steps = re.findall(r"^Time = (\S+)", text, re.M)
    out["steps"] = len(steps)
    limit = S17_STEP_GATES.get(cid)
    out["step_gate_limit"] = limit
    if limit is None:
        out["step_gate"] = None
    else:
        try:
            t_end = float(steps[-1].rstrip("s")) if steps else None
        except ValueError:
            t_end = None
        reached = t_end is not None and t_end >= 5e-3*(1 - 1e-9)
        out["step_gate"] = (False if len(steps) > limit
                            else True if reached else None)

    # discriminators
    disc = (want or {}).get("discriminator")
    out["discriminator"] = disc
    if disc in S17_DISCRIMINATORS:
        window, p16, p70, gate = S17_DISCRIMINATORS[disc]
        r = s17_second_difference(text, "outlet", window)
        r.update({"predicted_16_rows": p16, "predicted_70_rows": p70})
        r["b_ships"] = (None if r["median_ratio"] is None
                        else bool(r["median_ratio"] < 1)) if gate else None
        out["discriminator_reading"] = r
    elif disc in ("CB6", "CB4"):
        r = s17_second_difference(text, "surface", None)
        r["passes"] = (None if r["median_ratio"] is None or disc != "CB6"
                       else bool(r["median_ratio"] < 1))
        out["discriminator_reading"] = r
    else:
        out["discriminator_reading"] = None

    # start-up (em66, em67): the net outlet flux against its own plateau
    plateau_win = {"em66_G400_s17": (0.5e-3, 1.0e-3),
                   "em67_G4_s17": (0.3e-3, 0.8e-3)}.get(cid)
    if plateau_win:
        fl = [kv(m) for m in re.findall(r"^PHASE_CLOT_S17_SECONDDIFF (.*)$", text, re.M)]
        fl = [(r["time"], abs(r["outletPhiSum"])) for r in fl
              if "time" in r and "outletPhiSum" in r]
        plat = [q for t, q in fl if plateau_win[0] <= t <= plateau_win[1]]
        plateau = sum(plat)/len(plat) if plat else None
        reach = next((t for t, q in fl if plateau and q >= 0.9*plateau), None)
        out["startup_plateau"] = plateau
        out["startup_reach_90_time_s"] = reach
        whole = s17_second_difference(text, "outlet", None)
        out["startup_second_diff_median"] = whole["median_ratio"]
        out["startup_second_diff_chunk_max"] = whole["chunk_median_max"]
        out["startup"] = (None if plateau is None else
                          bool(reach is not None and reach <= 0.2e-3
                               and whole["chunk_median_max"] is not None
                               and whole["chunk_median_max"] < 1))
    else:
        out["startup"] = None
    # P4 (em66/em68): the cascade peak -- the largest G8-cell slip at or
    # after the first transfer; predicted 0.05-0.10 m/s, FAILING > 0.18.
    t_first = out.get("first_transfer_time_s")
    g8 = [kv(m) for m in re.findall(r"^PHASE_CLOT_S16_G8CELL (.*)$", text, re.M)]
    after = [r["slip"] for r in g8 if "slip" in r and "time" in r
             and t_first is not None and r["time"] >= t_first]
    out["p4_cascade_peak"] = max(after) if after else None
    out["p4"] = (None if not fibre or out["p4_cascade_peak"] is None
                 else bool(out["p4_cascade_peak"] <= S17_P4_FAIL))
    if cid == "em67_G4_s17":
        t0 = out.get("first_transfer_time_s")
        out["onset_time_s"] = t0
        out["onset"] = None if t0 is None else bool(0.9e-3 <= t0 <= 1.6e-3)
    fib = re.findall(r"^PHASE_CLOT_S17_FIBRE_INVALID (.*)$", text, re.M)
    out["fibre_invalid"] = fib[:3]
    return out


# ---- S17b (research/designs/2026-09-27-S17b-slack-fibre-field-units-wall-
#      face.md, revision 2, sections 2 and 4) ------------------------------
S17B_W2_LIMIT = 0.9702*1.01   # W2 and P2 (12 change 3): T(lc)/s_c x (1 + 0.01)
S17B_CHUNK = 20               # the start-up clause's chunk (12 change 2)
S17B_ALT_CORR = -0.5          # a chunk alternates if its median c_n < -0.5
S17B_G8 = 0.1                 # G8, unchanged (Alireza, 2026-09-26)
S17B_TAU = 3.3e-4             # the release time, for V1's lag class


def _s17b_alternation_series(text: str, instrument: str) -> tuple[list, list, list, list]:
    """(a_n, c_n, t_n, throughflow) for one instrument:
    "face"  -- PHASE_CLOT_S17_SECONDDIFF outletSecondDiff/outletSecondDiffCorr
               (the face-wise c_n the S17b binary logs);
    "proxy" -- the outlet column's mean U_x (PHASE_CLOT_DIAGNOSTIC
               outletMeanUx, logged every step): a scalar, whose c_n is the
               sign of successive second differences."""
    num = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"

    def kv(line):
        return {k: float(v) for k, v in re.findall(r"(\w+)=" + num, line)}

    sd = [kv(m) for m in re.findall(r"^PHASE_CLOT_S17_SECONDDIFF (.*)$", text, re.M)]
    a, c, ta = [], [], []
    if instrument == "face":
        face = [r for r in sd if r.get("validCorr") == 1 and "outletSecondDiffCorr" in r]
        rows = {int(r["step"]): r for r in face}
        for n in sorted(rows):
            r, r2 = rows[n], rows.get(n - 2)
            if r2 is None or not r2.get("outletSecondDiff"):
                continue
            nrm = r.get("outletPhiNorm") or 0
            if not nrm or r["outletSecondDiff"]/nrm <= S17_SECONDDIFF_FLOOR:
                continue
            a.append(r["outletSecondDiff"]/r2["outletSecondDiff"])
            c.append(r["outletSecondDiffCorr"])
            ta.append(r.get("time", 0.0))
        flux = [(r.get("time", 0.0), r["outletPhiSum"]) for r in sd if "outletPhiSum" in r]
        return a, c, ta, flux
    diag = [kv(m) for m in re.findall(r"^PHASE_CLOT_DIAGNOSTIC (.*)$", text, re.M)]
    diag = [r for r in diag if "outletMeanUx" in r]
    u = [r["outletMeanUx"] for r in diag]
    tu = [r.get("time", 0.0) for r in diag]
    d2 = [u[i] - 2*u[i - 1] + u[i - 2] for i in range(2, len(u))]
    for i in range(3, len(d2)):
        if not d2[i - 2] or not u[i + 2]:
            continue
        if abs(d2[i])/abs(u[i + 2]) <= S17_SECONDDIFF_FLOOR:
            continue
        a.append(abs(d2[i])/abs(d2[i - 2]))
        c.append(0.0 if d2[i]*d2[i - 1] == 0 else (1.0 if d2[i]*d2[i - 1] > 0 else -1.0))
        ta.append(tu[i + 2])
    fl = [(r.get("time", 0.0), r["outletPhiSum"]) for r in sd if "outletPhiSum" in r]
    return a, c, ta, (fl or list(zip(tu, u)))


def s17b_startup_alternation(text: str) -> dict:
    """The start-up clause re-specified on sign alternation (design section
    2; 12 change 2). Per step, a_n is the outlet second difference and c_n
    the correlation of successive ones; a 20-step chunk FAILS when its
    median c_n < -0.5 AND its median two-step ratio a_n/a_{n-2} >= 1; and the
    outlet throughflow must never change sign.

    WINDOW (design section 8 h, the gate's definition): step 1 to the first
    damage event (the first PHASE_CLOT_RUPTURE record with newlyRuptured >
    0), the start-up phase. Alternation after damage onset is a WATCH,
    reported with its chunks, never the start-up gate.

    INSTRUMENTS (section 8 f): both are evaluated and reported -- the
    face-wise c_n (PHASE_CLOT_S17_SECONDDIFF outletSecondDiffCorr, the S17b
    binary) and the outletMeanUx proxy (PHASE_CLOT_DIAGNOSTIC, every step),
    with their agreement. The verdict is the face-wise one where the log
    carries it; batch 57/58 logs carry only the proxy, and the verdict is
    then the proxy's, named as such."""
    onset = None
    for m in re.finditer(r"^PHASE_CLOT_RUPTURE time=(\S+) .*?newlyRuptured=(\d+)", text, re.M):
        if int(m.group(2)) > 0:
            onset = float(m.group(1))
            break

    def evaluate(instrument):
        a, c, ta, flux = _s17b_alternation_series(text, instrument)
        fails, late = [], []
        for k in range(0, len(a), S17B_CHUNK):
            ca, cc = sorted(a[k:k + S17B_CHUNK]), sorted(c[k:k + S17B_CHUNK])
            if len(ca) < S17B_CHUNK//2:
                continue
            mr, mc = ca[len(ca)//2], cc[len(cc)//2]
            if mc < S17B_ALT_CORR and mr >= 1:
                row = {"chunk": k//S17B_CHUNK, "t_start": ta[k],
                       "median_corr": mc, "median_ratio": mr}
                (fails if onset is None or ta[k] < onset else late).append(row)
        nzw = [x for t, x in flux if x and (onset is None or t < onset)]
        changes = sum(1 for x, y in zip(nzw, nzw[1:]) if (x > 0) != (y > 0))
        nza = [x for t, x in flux if x]
        changes_all = sum(1 for x, y in zip(nza, nza[1:]) if (x > 0) != (y > 0))
        n_win = sum(1 for t in ta if onset is None or t < onset)
        return {
            "readable_steps": n_win,
            "alternating_growing_chunks": fails[:5],
            "alternating_growing_count": len(fails),
            "throughflow_sign_changes": changes,
            "passes": (None if not n_win else bool(not fails and changes == 0)),
            "watch_after_onset_alternating_growing": late[:5],
            "watch_after_onset_count": len(late),
            "watch_throughflow_sign_changes_whole_run": changes_all,
        }

    face, proxy = evaluate("face"), evaluate("proxy")
    use_face = face["readable_steps"] > 0
    out = dict(face if use_face else proxy)
    out["instrument"] = ("face-wise (outletSecondDiffCorr)" if use_face else
                         "scalar proxy (outletMeanUx; no per-face data in this log)")
    out["window"] = ("step 1 to the first damage at t = %s s" % onset
                     if onset is not None else "the whole run (no damage)")
    out["face"] = face
    out["proxy"] = proxy
    out["instruments_agree"] = (None if face["passes"] is None or proxy["passes"] is None
                                else face["passes"] == proxy["passes"])
    return out


def s17b_g8_class(g8c: list[dict]) -> tuple[str | None, dict | None]:
    """The G8 exceedance class (design section 8 c), DRIVER FIRST, at the
    record of the largest G8-cell slip. Parts: devIntact, devPending,
    pressIntact, pressFrozen (the slip-projected force parts).
    - "V1 lag": devIntact is the largest part AND a slack event in the cell
      within tau (0 <= slackAge <= tau);
    - "H-NA10 signature": the pressure parts lead (pressure > 0 and >= the
      deviatoric), theta/ref < 0.3 AND falling (below the previous same-cell
      record), and Vs.ind within +-25% over >= 3 same-cell records
      (09's criterion D); pressure-led without all three is
      "pressure-driven (not the H-NA10 signature)";
    - deviatoric-led: "neighbour-pull deviatoric";
    - otherwise "other".
    slackAge is reported as an attribute in every class. Same cell = the
    same logged (x, y)."""
    rows = [r for r in g8c if "slip" in r]
    if not rows:
        return None, None
    wi = max(range(len(rows)), key=lambda i: rows[i]["slip"])
    w = rows[wi]
    dI, dP = w.get("devIntact", 0.0), w.get("devPending", 0.0)
    pI, pF = w.get("pressIntact", 0.0), w.get("pressFrozen", 0.0)
    dev, pres = dI + dP, pI + pF
    age = w.get("slackAge", -1.0)
    same = [r for r in rows[:wi] if "x" in w and r.get("x") == w.get("x")
            and r.get("y") == w.get("y")]
    attrs = {"slackAge": age, "devIntact": dI, "devPending": dP,
             "pressIntact": pI, "pressFrozen": pF,
             "thetaByRef": w.get("thetaByRef"), "VsInd": w.get("VsInd"),
             "same_cell_prior_records": len(same)}
    if dI > 0 and dI >= max(dP, pI, pF) and 0 <= age <= S17B_TAU:
        return "V1 lag (devIntact leads; a slack event in the G8 cell within tau)", attrs
    if pres > 0 and pres >= dev:
        tb = w.get("thetaByRef", -1.0)
        falling = bool(same) and tb < same[-1].get("thetaByRef", -1.0)
        recent = same[-2:] + [w]
        vs = [r.get("VsInd", 0.0) for r in recent]
        within = (len(recent) >= 3 and vs[-1] != 0
                  and all(abs(v/vs[-1] - 1.0) <= 0.25 for v in vs))
        attrs.update({"thetaByRef_below_0.3": 0 <= tb < 0.3, "thetaByRef_falling": falling,
                      "VsInd_within_25pct_3_records": within})
        if 0 <= tb < 0.3 and falling and within:
            return "pressure-driven evacuation (H-NA10 signature)", attrs
        return "pressure-driven (not the H-NA10 signature)", attrs
    if dev > 0:
        return "neighbour-pull deviatoric", attrs
    return "other", attrs


def s17b_report(text: str, spec: dict | None = None) -> dict:
    """The S17b quantities, READ from the log (design sections 4 and 8).
    - config: PHASE_CLOT_S17B_CONFIG equal to child.json's "s17b" block.
    - THE V1 SLACK GATE (section 8 d): |dd| = 0 at every slack event,
      scored. CONFORMANCE (read back from the stores, reported, not scored):
      q_P = 0 after the event, and A1's condition "mu < 1 or lf_in = 1"
      (max mu over the events with lf_in != 1 below 1). WATCH: the
      pre-event residual tension T(lf_in)/T(1/mu) and |q_P|/(S G T(1/mu)),
      each expected <= 1. And theta_T <= theta_P exactly everywhere (scored).
    - no fibre-invalid FatalError (a PHASE_CLOT_S17_FIBRE_INVALID line, or a
      fatal naming the fibre, fails the child).
    - W2 and P2 <= 0.9702 x 1.01 (12 change 3) at every record.
    - G-tr is a WATCH (design section 2), reported with its maximum.
    - the start-up clause on sign alternation (s17b_startup_alternation).
    - G8 at 0.1 m/s: an exceedance is scored FAIL whatever its class; the
      class is reported beside it, driver first (s17b_g8_class).
    - WATCHES: the site-tagged max lf, projRaised by presence, the S < 0 and
      theta/ref < 1/2 counts, |p_P|/|min(p_n,0)|, the frozen tension's
      power, the slack content-share proxy.
    em74 against em71 (the wall-face sign rule) and G-id em74 against em66
    need two logs: scripts/s17_offline_checks.py."""
    num = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"

    def kv(line):
        return {k: float(v) for k, v in re.findall(r"(\w+)=" + num, line)}

    spec = spec or {}
    want = spec.get("s17b")
    # S21 (report 10): PHASE_CLOT_S17B_CONFIG is no longer printed and fibreSlack,
    # fibreArithmetic and skeletonWallFace are built in; a child.json s17b block would
    # be a generator defect, and it is then reported as not matching.
    out: dict = {"config": None, "config_expected": want}
    if want is None:
        out["config_matches_child"] = None
    elif out["config"] is None:
        out["config_matches_child"] = False
    else:
        bad = [k for k in ("fibreSlack", "fibreArithmetic", "skeletonWallFace")
               if str(out["config"].get(k)) != str(want.get(k))]
        out["config_mismatched_keys"] = bad
        out["config_matches_child"] = not bad
    # S21: S17B_CONFIG is gone and fibreSlack is built in at v1; child.json has no s17b
    # block, so config_matches_child is None (nothing declared) and the gates apply.
    v1 = s21_config(out["config"] or {}, "fibreSlack") == "v1"
    # S21: every child runs S17b (the block was all built-in members, so child.json
    # no longer carries it); the gates below that keyed on a declared block apply to
    # every child, or they would have gone silent on all 18 kept children.
    s17b_child = want is not None or v1
    rows = [kv(m) for m in re.findall(r"^PHASE_CLOT_S17B (.*)$", text, re.M)]
    out["records"] = len(rows)
    keys = ("slackEvents", "slackDdMax", "slackMuMax", "slackAtRest",
            "slackQpMax", "slackPreTRatioMax", "slackPreQpRatioMax",
            "tautAboveCells")
    if rows and v1:
        incomplete = [r for r in rows if any(k not in r for k in keys)]
        ev = [r for r in rows if r.get("slackEvents", 0) > 0]
        out["slack_events_total"] = int(sum(r.get("slackEvents", 0) for r in rows))
        out["slack_events_at_rest"] = int(sum(r.get("slackAtRest", 0) for r in rows))
        # THE GATE: |dd| = 0 at every slack event
        out["slack_dd"] = (None if not ev else bool(
            not incomplete and all(r["slackDdMax"] == 0 for r in ev)))
        out["slack_dd_first_violation"] = next(
            ({"time": r.get("time"), "slackDdMax": r.get("slackDdMax")}
             for r in ev if r.get("slackDdMax", 0) != 0), None)
        # CONFORMANCE (reported, not scored)
        out["conformance_slack"] = (None if not ev else {
            "qP_zero_after": all(r.get("slackQpMax", 1) == 0 for r in ev),
            "mu_below_1_or_at_rest": all(r.get("slackMuMax", 1) < 1 for r in ev),
        })
        # WATCH: the residual the event drops, against T(1/mu)
        for k in ("slackPreTMax", "slackPreTRatioMax", "slackPreQpMax",
                  "slackPreQpRatioMax"):
            vals = [r[k] for r in ev if k in r]
            out[f"watch_{k}"] = max(vals) if vals else None
        out["watch_slack_residual_above_one_step_bound"] = (
            None if not ev else any(r.get("slackPreTRatioMax", 0) > 1
                                    or r.get("slackPreQpRatioMax", 0) > 1 for r in ev))
        out["taut_le_pending"] = bool(not incomplete and all(
            r["tautAboveCells"] == 0 for r in rows))
        for k in ("maxLfTransport", "maxLfRotation", "maxLfLaw",
                  "pPRatioMax", "slackContentShareMax", "sNegCells",
                  "thetaRefHalfCells"):
            vals = [r[k] for r in rows if k in r]
            out[f"watch_{k}"] = max(vals) if vals else None
        out["watch_projRaised_present"] = int(sum(r.get("projRaisedPresent", 0) for r in rows))
        out["watch_projRaised_absent"] = int(sum(r.get("projRaisedAbsent", 0) for r in rows))
        fp = [r["frozenPower"] for r in rows if "frozenPower" in r]
        out["watch_frozen_power_min"] = min(fp) if fp else None
    else:
        out.update({"slack_dd": None, "conformance_slack": None,
                    "taut_le_pending": None})
    inv = re.findall(r"^PHASE_CLOT_S17_FIBRE_INVALID (.*)$", text, re.M)
    fatal = re.search(r"pending fibre is invalid", text)
    out["no_fibre_invalid_fatal"] = (None if not s17b_child
                                     else not (inv or fatal))
    s17 = [kv(m) for m in re.findall(r"^PHASE_CLOT_S17 (.*)$", text, re.M)]
    w2 = [r["w2Max"] for r in s17 if "w2Max" in r]
    p2 = [r["p2Max"] for r in s17 if "p2Max" in r]
    out["w2_max"] = max(w2) if w2 else None
    out["p2_max"] = max(p2) if p2 else None
    out["w2"] = None if not w2 or not s17b_child else bool(max(w2) <= S17B_W2_LIMIT)
    out["p2"] = None if not p2 or not s17b_child else bool(max(p2) <= S17B_W2_LIMIT)
    tr = [r["transferB1Max"] for r in s17 if r.get("transfers", 0) > 0]
    out["watch_g_tr_max"] = max(tr) if tr else None
    out["g_tr_note"] = "G-tr is a WATCH under S17b (design section 2)"
    out["startup_alternation"] = s17b_startup_alternation(text)
    # G8, scored FAIL on any exceedance, with the class beside it
    vsp = [kv(m) for m in re.findall(r"^PHASE_CLOT_VS_SPLIT (.*)$", text, re.M)]
    slips = [r["slip"] for r in vsp if "slip" in r]
    fsk = [kv(m) for m in re.findall(r"^PHASE_CLOT_FACE_SKELETON (.*)$", text, re.M)]
    fvs = [r["maxFaceVs"] for r in fsk if "maxFaceVs" in r]
    g8max = max(slips) if slips else None
    out["g8_slip_max"] = g8max
    out["g8_face_vs_max"] = max(fvs) if fvs else None
    out["g8"] = (None if g8max is None or not fvs else
                 bool(g8max <= S17B_G8 and max(fvs) <= S17B_G8))
    g8c = [kv(m) for m in re.findall(r"^PHASE_CLOT_S17B_G8CELL (.*)$", text, re.M)]
    cls, attrs = s17b_g8_class(g8c)
    out["g8_class"] = cls
    out["g8_class_attributes"] = attrs
    return out


# ---- S18, THE SOFT STAGE (design 2026-09-29, rev. 10, sections 2 G and 5) --
S18_G8 = 0.1                     # any VS_SPLIT slip above it, 0-5 ms, FAILS
S18_SLIP_PREDICTED = 0.002       # predicted max slip over 0-2.6 ms (a reading)
S18_SLIP_PREDICTED_T = 2.6e-3
S18_MOVED = ((1.0e-3, 6.77e-6), (2.0e-3, 15.77e-6))   # clotXbar displacement
S18_OCC_FRINGE = 0.01            # 04: 84% of em74's pF-led records sit below
S18_ANTI_SHARE = 0.0132          # the anti-restoring share watch, after 4 ms
S18_ANTI_T = 4.0e-3
S18_EM72_DEBOND = ((2.5e-3, 3), (4.0e-3, 15), (5.0e-3, 29))
S18_C6_PRE, S18_C6_POST, S18_CONV_POST = 1.6e-4, 2.4e-3, 0.99
NOT_EVALUABLE = "NOT-EVALUABLE on Colab (references em72; scored offline)"


def s18_g8_classes(g8c: list[dict], exceed: float = S18_G8) -> dict:
    """The G8 class over ALL exceedance records (design 2 G; OI-41), driver
    first, with the intact and the frozen pressure parts SEPARATE. For each
    PHASE_CLOT_S17B_G8CELL record with slip > 0.1:
    - the leading part of devIntact, devPending, pressIntact, pressFrozen;
    - "V1 lag" when devIntact leads and a slack event in the cell is within
      tau (0 <= slackAge <= tau);
    - "H-NA10 signature" when a pressure part leads, the cell's deposited
      occupancy is < 0.01 (amendment c's theta/ref test, replaced: 04), and
      Vs.ind is within +-25% over >= 3 same-cell records (09's criterion D),
      labelled with WHICH pressure part leads;
    - otherwise the leading part's name. Tallied, with slackAge carried."""
    rows = [r for r in g8c if "slip" in r]
    tally: dict = {}
    first = {}
    for i, r in enumerate(rows):
        if not r["slip"] > exceed:
            continue
        parts = {k: r.get(k, 0.0) for k in ("devIntact", "devPending",
                                             "pressIntact", "pressFrozen")}
        lead = max(parts, key=parts.get)
        age = r.get("slackAge", -1.0)
        if lead == "devIntact" and 0 <= age <= S17B_TAU:
            cls = "V1 lag (devIntact leads; slack within tau)"
        elif lead in ("pressIntact", "pressFrozen"):
            same = [q for q in rows[:i] if "x" in r and q.get("x") == r.get("x")
                    and q.get("y") == r.get("y")][-2:] + [r]
            vs = [q.get("VsInd", 0.0) for q in same]
            within = (len(same) >= 3 and vs[-1] != 0
                      and all(abs(v/vs[-1] - 1.0) <= 0.25 for v in vs))
            occ = r.get("occupancy", 1.0)
            if 0 <= occ < S18_OCC_FRINGE and within:
                cls = f"H-NA10 signature ({lead} leads)"
            else:
                cls = f"{lead}-led"
        elif parts[lead] > 0:
            cls = f"{lead}-led"
        else:
            cls = "other"
        tally[cls] = tally.get(cls, 0) + 1
        first.setdefault(cls, {"time": r.get("time"), "x": r.get("x"),
                               "y": r.get("y"), "slackAge": age,
                               "occupancy": r.get("occupancy")})
    return {"records": sum(tally.values()), "tally": tally, "first": first}


def s18_corrector_watch(text: str, n_correctors: int, onset: float | None) -> dict:
    """The corrector-convergence WATCH (22 section 3; 24 #2): corrector-6's
    initial p residual per step (the first p solve of each corrector),
    step 1 EXCLUDED; median <= 1.6e-4 before the first damage and <= 2.4e-3
    after it; the converging share (c6 < 0.1 c2) >= 99% after it."""
    rp = re.compile(r"Solving for p, Initial residual = ([^,]+),")
    steps, cur, t = [], None, None
    for line in text.splitlines():
        if line.startswith("Time = "):
            if t is not None:
                steps.append((t, cur))
            try:
                t = float(line[7:].strip().rstrip("s"))
            except ValueError:
                t = None
            cur = []
            continue
        if t is not None:
            m = rp.search(line)
            if m:
                cur.append(float(m.group(1)))
    if t is not None:
        steps.append((t, cur))
    pre, post, conv, other = [], [], [], 0
    for k, (tt, r) in enumerate(steps):
        if k == 0:
            continue                      # step 1 never converges (0.774)
        if not r or len(r) % n_correctors:
            other += 1
            continue
        per = len(r)//n_correctors
        c = [r[i*per] for i in range(n_correctors)]
        c6, c2 = c[-1], c[1] if n_correctors > 1 else c[0]
        if onset is None or tt < onset:
            pre.append(c6)
        else:
            post.append(c6)
            conv.append(c2 > 0 and c6 < 0.1*c2)
    med = lambda v: sorted(v)[len(v)//2] if v else None
    out = {"steps_read": len(pre) + len(post), "steps_unreadable": other,
           "c6_median_before_onset": med(pre), "c6_median_after_onset": med(post),
           "converging_share_after_onset": (sum(conv)/len(conv)) if conv else None}
    out["watch_exceeded"] = bool(
        (out["c6_median_before_onset"] or 0) > S18_C6_PRE
        or (out["c6_median_after_onset"] or 0) > S18_C6_POST
        or (out["converging_share_after_onset"] is not None
            and out["converging_share_after_onset"] < S18_CONV_POST))
    return out


def s18_report(text: str, spec: dict | None = None) -> dict:
    """em76's clauses, exactly as the S18 design section 5 (rev. 10), READ
    from the log. Scored:
    - config: PHASE_CLOT_S18_CONFIG equal to child.json's "s18" block;
    - G8: ANY PHASE_CLOT_VS_SPLIT slip > 0.1 m/s in 0-5 ms FAILS (never on
      maxFaceVs);
    - the moved gate (a frozen-clot catcher only): clotXbar displacement
      >= 6.77 um at 1.0 ms and >= 15.77 um at 2.0 ms;
    - dt: no skeleton-carrier-bound step (deltaT reaching the carrier limit);
    - CONFORMANCE (a breach is a build defect): frozenPower = 0 at every
      record under frozenPendingPressure none; projRaised = 0 at every step;
      S17split on the wall patch (wallSplitDevMax = 0 under split).
    Reported NOT-EVALUABLE on Colab (they reference em72; the CE scores them
    offline): the first-damage window [1.17, 1.22] ms on the RUPTURE line,
    tearing against em72, the debond watch against em72's 3/15/29. em76's
    own numbers for them are reported beside the label.
    WATCHES: the prediction (slip <= 0.002 m/s to 2.6 ms), the anti-
    restoring share after 4 ms (> 1.32%), the corrector convergence, the
    G8 class over all records (s18_g8_classes)."""
    num = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"

    def kv(line):
        return {k: float(v) for k, v in re.findall(r"(\w+)=" + num, line)}

    spec = spec or {}
    want = spec.get("s18")
    out: dict = {"config": None, "config_expected": want}
    cfg = re.search(r"^PHASE_CLOT_S18_CONFIG (.*)$", text, re.M)
    if cfg:
        out["config"] = dict(re.findall(r"(\w+)=(\S+)", cfg.group(1)))
    if want is None:
        out["config_matches_child"] = None
    elif out["config"] is None:
        out["config_matches_child"] = False
    else:
        # S21: frozenPendingPressure, wallPatchSplit and cplusOrder are built in; the
        # line and child.json carry networkPressureWeight only.
        bad = [k for k in ("networkPressureWeight",)
               if str(out["config"].get(k)) != str(want.get(k))]
        out["config_mismatched_keys"] = bad
        out["config_matches_child"] = not bad
    cf = out["config"] or {}
    rows = [kv(m) for m in re.findall(r"^PHASE_CLOT_S18 (.*)$", text, re.M)]
    out["records"] = len(rows)
    # G8 on VS_SPLIT only
    vsp = [kv(m) for m in re.findall(r"^PHASE_CLOT_VS_SPLIT (.*)$", text, re.M)]
    slips = [(r.get("time", 0.0), r["slip"]) for r in vsp if "slip" in r]
    out["g8_slip_max"] = max((v for _t, v in slips), default=None)
    out["g8"] = None if not slips else bool(out["g8_slip_max"] <= S18_G8)
    out["g8_first_exceedance_time"] = next((t for t, v in slips if v > S18_G8), None)
    early = [v for t, v in slips if t <= S18_SLIP_PREDICTED_T]
    out["prediction_slip_max_to_2p6ms"] = max(early) if early else None
    out["prediction_slip_holds"] = None if not early else bool(max(early) <= S18_SLIP_PREDICTED)
    g8c = [kv(m) for m in re.findall(r"^PHASE_CLOT_S17B_G8CELL (.*)$", text, re.M)]
    out["g8_classes"] = s18_g8_classes(g8c)
    # the moved gate
    mom = [(r["time"], r["clotXbar"]) for r in
           (kv(m) for m in re.findall(r"^PHASE_CLOT_MOMENTUM (.*)$", text, re.M))
           if "clotXbar" in r and "time" in r]
    if mom:
        x0 = mom[0][1]
        moved = {}
        ok = True
        for tt, need in S18_MOVED:
            if mom[-1][0] < tt - 1e-12:
                moved[tt] = None
                ok = None if ok is not False else False
                continue
            t_r, x_r = min(mom, key=lambda r: abs(r[0] - tt))
            moved[tt] = {"record_time": t_r, "displacement_m": x_r - x0,
                         "required_m": need, "passes": (x_r - x0) >= need}
            if not (x_r - x0) >= need:
                ok = False
        out["moved"] = moved
        out["moved_gate"] = ok
    else:
        out["moved"], out["moved_gate"] = None, None
    # dt: no carrier-bound step
    if rows:
        bound = [r for r in rows if "carrierDtLimit" in r and "deltaT" in r
                 and r["deltaT"] >= r["carrierDtLimit"]*(1 - 1e-9)]
        out["steps"] = len(rows)
        out["carrier_bound_steps"] = len(bound)
        out["dt_no_carrier_bound"] = not bound
        out["coSkel_max"] = max((r.get("coSkel", 0.0) for r in rows), default=None)
        wk = max(rows, key=lambda r: r.get("coSkel", 0.0))
        out["coSkel_cell_at_max"] = {k: wk.get(k) for k in ("time", "coSkelCell", "coSkelX", "coSkelY")}
    else:
        out.update({"steps": 0, "carrier_bound_steps": None, "dt_no_carrier_bound": None})
    # CONFORMANCE
    fp = [r["frozenPower"] for r in rows if "frozenPower" in r]
    s17 = [kv(m) for m in re.findall(r"^PHASE_CLOT_S17 (.*)$", text, re.M)]
    pr = [r["projRaised"] for r in s17 if "projRaised" in r]
    ws = [r["wallSplitDevMax"] for r in rows if "wallSplitDevMax" in r]
    out["conformance"] = {
        "frozenPower_zero": (None if s21_config(cf, "frozenPendingPressure") != "none" or not fp
                             else all(v == 0 for v in fp)),
        "frozenPower_max_abs": max((abs(v) for v in fp), default=None),
        "projRaised_zero": None if not pr else all(v == 0 for v in pr),
        "wall_patch_split_holds": (None if s21_config(cf, "wallPatchSplit") != "split" or not ws
                                   else all(v == 0 for v in ws)),
    }
    out["conformance_ok"] = (None if any(v is None for k, v in out["conformance"].items()
                                         if k != "frozenPower_max_abs")
                             else all(v for k, v in out["conformance"].items()
                                      if k != "frozenPower_max_abs"))
    # em72-referenced: NOT-EVALUABLE, with em76's own numbers
    rup = re.search(r"^PHASE_CLOT_RUPTURE time=(\S+) .*?newlyRuptured=([1-9]\d*)", text, re.M)
    out["first_damage_time_s"] = float(rup.group(1)) if rup else None
    out["onset_window"] = NOT_EVALUABLE
    out["tearing_vs_em72"] = NOT_EVALUABLE
    deb = [(r["time"], r["debondedFaces"]) for r in
           (kv(m) for m in re.findall(r"^PHASE_CLOT_DEBOND (.*)$", text, re.M))
           if "debondedFaces" in r and "time" in r]
    out["debond_watch_vs_em72"] = NOT_EVALUABLE
    out["debonded_faces_at"] = {
        tt: (min(deb, key=lambda r: abs(r[0] - tt))[1] if deb and deb[-1][0] >= tt - 1e-12 else None)
        for tt, _n in S18_EM72_DEBOND}
    out["em72_debonded_faces_reference"] = {tt: n for tt, n in S18_EM72_DEBOND}
    # WATCHES
    late = [r for r in rows if r.get("time", 0.0) >= S18_ANTI_T and r.get("netCells", 0) > 0]
    shares = [r["sNegCells"]/r["netCells"] for r in late if "sNegCells" in r]
    out["watch_anti_restoring_share_max_after_4ms"] = max(shares) if shares else None
    out["watch_anti_restoring_share_exceeded"] = (None if not shares
                                                  else max(shares) > S18_ANTI_SHARE)
    ncorr = int(((spec.get("s17") or {}).get("nCorrectors")) or 6)
    out["watch_corrector_convergence"] = s18_corrector_watch(text, ncorr, out["first_damage_time_s"])
    out["scored_ok"] = (None if out["g8"] is None or out["config_matches_child"] is None
                        else bool(out["g8"] and out["config_matches_child"]
                                  and out["moved_gate"] is not False
                                  and out["dt_no_carrier_bound"] is not False
                                  and out["conformance_ok"] is not False))
    return out


# ---- XU ET AL. 2017, FIGURE 5 (design 2026-09-29-case-xu2017-fig5, rev. 4,
# APPROVED 2026-09-30; docket 20260929-1430-case-xu2017-...) ---------------
# The pre-registration of design section 4, READ from the log and the written
# fields. Every number below is that table's. The clauses that are
# predictions of a model (P3-P9, N-rows) are scored against their own FAILING
# column; A5 (energy) needs the returned archive and is scored OFFLINE by
# scripts/s17_offline_checks.py energy-inlet (NOT-EVALUABLE here).
XU_SERIES_NAME = "xu2017_fig5"
XU_A0 = {  # A0, the config echo: (value, relative tolerance)
    "rho": (1025.0, 1e-9), "nu": (3.9024e-6, 1e-4), "G": (1.2, 1e-9),
    "Gcore": (24.0, 1e-9), "k0p": (4.07e-13, 1e-9), "k0pcore": (1e-15, 1e-9),
    "material_width_m": (3e-6, 1e-9), "adhesive_strength_pa": (1e9, 1e-9),
    "epsilonC": (3e-6, 1e-9),     # the interface width (verification 08; DECISIONS 2026-09-30)
}
XU_U_MEAN = {250.0: 2.0833e-3, 500.0: 4.1667e-3, 1000.0: 8.3333e-3, 2000.0: 1.6667e-2}
XU_DOME = {"injury_centre_x_m": 9.14e-05, "half_length_m": 4.05e-05, "height_m": 3.47e-05,
           "core_centre_x_m": 9.08e-05, "core_half_length_m": 2.0e-05,
           "core_height_m": 1.46e-05, "signed_distance_samples": 4001}
XU_A1_TOL = 1e-6
XU_DT_CAP = 2.604167e-6                      # N1
XU_N1_SHARE, XU_N1_RUN = 0.99, 10
XU_N2 = (4.28*0.75, 4.34*1.25)               # N2: 4.28-4.34 s/step +- 25%
XU_N3_UMAX, XU_N3_STEPS = 4.0, 2             # N3: G8 > 4 u_max on > 2 steps
XU_N7 = 0.2
XU_P0_Q = (0.995, 1.005)
XU_P0_TW = (1.0092 - 0.02, 1.0092 + 0.02)
XU_P1 = (1.03, 1.30)
XU_P1_PRED = 1.152
XU_NET, XU_DAMAGE, XU_BULK_G, XU_BULK_DZ = 1e-2, 0.01, 20.0, 2e-6
XU_CLOT_X = (50e-6, 135e-6)                  # P1, P9: "over the clot"
XU_BEYOND_X, XU_EDGE_REF_X = 131.9e-6, 131.90e-6
XU_P3 = {250.0: (5.83, (3.5, 9.9), 10.0, 0.0410), 500.0: (2.56, (1.54, 4.35), 6.0, 0.0929),
         1000.0: (1.18, (0.71, 2.0), 3.5, 0.153), 2000.0: (0.56, (0.34, 0.95), 2.0, 0.228)}
# P4 = N4: (first-damage band ms or None = none before 10 ms, lambda1 mark ms, band)
XU_P4 = {250.0: (None, 10.0, (1.375, 1.507)), 500.0: (None, 6.0, (1.643, 1.871)),
         1000.0: ((2.2, 3.9), None, None), 2000.0: ((0.77, 1.37), None, None)}
XU_P5 = {250.0: (2.5, 0.93), 500.0: (2.5, 1.85), 1000.0: (2.5, 3.53), 2000.0: (1.0, 3.52)}
XU_P6 = {250.0: [(2.5, 0.0175, 2.6), (10.0, 0.191, 20.3)], 500.0: [(2.5, 0.064, 8.3)],
         1000.0: [(2.5, 0.208, 21.8)], 2000.0: [(1.0, 0.150, 16.5)]}
XU_P6_T0 = 0.0041
XU_P9_XU = {250.0: 2410.0, 500.0: 4820.0, 1000.0: 9430.0, 2000.0: 17190.0}
XU_P9_FAIL = 0.5
XU_N5_EXTRA, XU_N5_REF_MS = 5, 5.0
XU_N6_RATIO, XU_N6_SQRT, XU_N6_TOL = 0.10, 25.92, {"a": 0.01, "b": 2e-6}
XU_A6_RATIO = 1e-6
# A6's crossing floor (DECISIONS 2026-09-30, the Chief Engineer's ruling): a
# step counts as a crossing only when |contentOuterFlux| > 1e-20 m3/s, x1e8
# above em72's round-off trace (1e-31 to 1e-28 m3/s on 1199 of 1200 steps,
# run 20260928-210844) -- below it the 1e-6 residual bound is below the
# residual's own round-off (em72: |contentResidual| up to 1.99e-22).
XU_A6_FLOOR = 1e-20
XU_A6_ABS_ULPS = 10           # R_abs = 10 ulps (A6 RE-RULED, DECISIONS 2026-09-30)


S18P_FIRST_RAISE_T = 1.2e-3     # em76 raised from 1.2042 ms (V2's non-empty clause)
S18P_AB_TINY, S18P_AB_RATIO = 1e-300, 1e12   # the A/B route (section 5)


def s18p_report(text: str, spec: dict | None = None) -> dict:
    """S18p (research/designs/2026-10-01-S18p-em76-attribution-probe.md rev.
    2, section 5). REPORT-ONLY: no admissibility change. V2, the instrument
    fires and adds up:
    - exactly one PHASE_CLOT_S18P record per step (the steps of the
      PHASE_CLOT_S17 records; one each, the same set);
    - T + P equals S17's logged projCount and projRaised at every step;
    - projRaisedT + projRaisedP > 0 after 1.2 ms (em78's clause; reported
      for every child, required for the control only).
    Also read: the config echo, the per-site run totals split present/absent,
    and the A/B route on the dumped transport-site cells (supported if at
    least half have Bdiv < 1e-300 or PkNew/Bdiv > 1e12, falsified if none,
    unresolved otherwise). The score, V1 and the bands are OFFLINE."""
    num = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"

    def kv(line):
        return {k: float(v) for k, v in re.findall(r"(\w+)=" + num, line)}

    spec = spec or {}
    want = (spec.get("s18p") or {}).get("projSiteLog")
    out: dict = {"config_expected": want}
    cfg = re.search(r"^PHASE_CLOT_S18P_CONFIG (.*)$", text, re.M)
    out["config"] = dict(re.findall(r"(\w+)=(\S+)", cfg.group(1))) if cfg else None
    out["config_matches_child"] = (None if want is None else
                                   bool(out["config"]) and out["config"].get("projSiteLog") == want)
    rows = [kv(m) for m in re.findall(r"^PHASE_CLOT_S18P (.*)$", text, re.M)]
    s17 = [kv(m) for m in re.findall(r"^PHASE_CLOT_S17 (.*)$", text, re.M)]
    s17_by_step = {int(r["step"]): r for r in s17 if "step" in r}
    steps = [int(r["step"]) for r in rows if "step" in r]
    out["records"] = len(rows)
    out["steps"] = len(s17_by_step)
    dup = sorted({k for k in steps if steps.count(k) > 1}) if len(set(steps)) != len(steps) else []
    missing = sorted(set(s17_by_step) - set(steps))
    stray = sorted(set(steps) - set(s17_by_step))
    out["duplicate_steps"] = dup[:5]
    out["missing_steps"] = missing[:5]
    out["missing_step_count"] = len(missing)
    out["stray_steps"] = stray[:5]
    out["one_record_per_step"] = (None if not s17_by_step else
                                  bool(len(steps) == len(rows) and not dup
                                       and not missing and not stray))
    bad = []
    for r in rows:
        ref = s17_by_step.get(int(r.get("step", -1)))
        if ref is None:
            continue
        for tot in ("projCount", "projRaised"):
            if (tot not in ref or tot + "T" not in r or tot + "P" not in r
                    or r[tot + "T"] + r[tot + "P"] != ref[tot]):
                bad.append({"step": int(r["step"]), "key": tot,
                            "T": r.get(tot + "T"), "P": r.get(tot + "P"),
                            "total": ref.get(tot)})
    out["sum_mismatches"] = len(bad)
    out["first_sum_mismatch"] = bad[:1]
    out["t_plus_p_equals_totals"] = None if not rows else not bad
    after = [r.get("projRaisedT", 0.0) + r.get("projRaisedP", 0.0)
             for r in rows if r.get("time", -1.0) > S18P_FIRST_RAISE_T]
    out["raised_after_1p2ms"] = sum(after) if after else None
    out["non_empty_after_1p2ms"] = None if not after else sum(after) > 0
    is_control = spec.get("child_id") == "em78_G4_s18p_ctl"
    out["v2"] = (None if out["one_record_per_step"] is None or out["t_plus_p_equals_totals"] is None
                 else bool(out["one_record_per_step"] and out["t_plus_p_equals_totals"]
                           and (out["non_empty_after_1p2ms"] is True or not is_control)))
    out["v2_requires_non_empty"] = is_control
    tot = {}
    for site in ("T", "P"):
        for k in ("projCount", "projRaised", "projRaisedPresent", "projRaisedAbsent"):
            # verification 10 D5: None, not 0, when no S18P record exists
            tot[k + site] = sum(r.get(k + site, 0.0) for r in rows) if rows else None
        tot["projExcessMax" + site] = max((r.get("projExcessMax" + site, 0.0) for r in rows), default=None)
    out["site_totals"] = tot
    out["transport_present_raises"] = tot["projRaisedPresentP"]
    out["transfer_raises"] = tot["projRaisedT"]
    out["drained_cells_last"] = rows[-1].get("drainedCells") if rows else None
    out["network_cells_last"] = rows[-1].get("networkCells") if rows else None
    cells = [kv(m) | {"site": st, "kernel": kn}
             for st, kn, m in ((re.search(r"site=(\w)", ln).group(1),
                                (re.search(r"kernel=(\w+)", ln) or [None, None])[1], ln)
                               for ln in re.findall(r"^PHASE_CLOT_S18P_CELL (.*)$", text, re.M))]
    out["dumped_cells"] = {"T": sum(c["site"] == "T" for c in cells),
                           "P": sum(c["site"] == "P" for c in cells)}
    ab = [c for c in cells if c["site"] == "P" and c["kernel"] == "passengersShared"
          and "Bdiv" in c and "PkNew" in c]
    hit = [c for c in ab if c["Bdiv"] < S18P_AB_TINY
           or (c["Bdiv"] > 0 and c["PkNew"]/c["Bdiv"] > S18P_AB_RATIO)]
    out["ab_route_cells"] = len(ab)
    out["ab_route_hits"] = len(hit)
    out["ab_route"] = (None if not ab else "falsified" if not hit
                       else "supported" if 2*len(hit) >= len(ab) else "unresolved")
    return out


# S19, the block-coupled step (research/designs/2026-10-03-S19-block-coupled-
# step.md rev. 4, APPROVED 2026-10-04), sections 7 and 12. Each threshold is
# the design's, fixed before the runs.
S19_CONTRES_MAX = 1e-9     # G4: max_cells |sum_f F_f| / Q0, every step
S19_FLUX_MATCH = 1e-10     # G4: qInlet/qOutlet vs logged inletFlux/outletFlux (relative)
S19_LINRES_MAX = 1e-10     # G6: linRes and linResScaled, every step
# S19a (research/designs/2026-10-04-S19a-umfpack-row-equilibration.md,
# APPROVED 2026-10-04), section 4: G6 extended to linResScaled, and G9.
S19_NUMERIC_PEAK_GB_MAX = 1.0   # G9: numericPeakGB (UMFPACK peak, GB), every step
S19_TNUMERIC_MAX = 5.0          # G9: tNumeric (numeric factorisation, s), every step
# S22 (research/designs/2026-10-07-S22-pardiso-block-solver.md rev. 2, APPROVED
# 2026-10-07), sections 3.3 and 7: under blockLinearSolver pardiso G9 also gates
# MKL's own peak; G22-4 (no perturbed pivot) and G22-5's rssGB, every step.
S22_MKL_PEAK_GB_MAX = 2.0       # G9 under pardiso: mklPeakGB (MKL's peak, GB)
S22_RSS_GB_MAX = 4.0            # G22-5: rssGB (the process's peak RSS, GB)
# S19b (research/designs/2026-10-04-S19b-lumen-elimination.md rev. 2, APPROVED
# 2026-10-04), section 4: C1 and D1 compare n with 5 x nCells, nCells read
# from the case's own mesh (the constant/polyMesh/owner header note), never
# from the record they judge.


def s19b_mesh_cells(case: Path | None) -> int | None:
    """nCells from constant/polyMesh/owner's header note ("... nCells: N
    ..."), plain or gzipped; None if the case or the note is absent."""
    if case is None:
        return None
    import gzip
    base = case / "constant" / "polyMesh"
    for name, opener in (("owner", open), ("owner.gz", gzip.open)):
        f = base / name
        if f.is_file():
            with opener(f, "rb") as fh:
                head = fh.read(4096).decode("latin-1")
            m = re.search(r"nCells:\s*(\d+)", head)
            return int(m.group(1)) if m else None
    return None


def s19_report(text: str, spec: dict | None = None, case: Path | None = None) -> dict:
    """S19 sections 7 and 12, read from PHASE_CLOT_S19 (one record per step):
    - G0 engagement: a record with mode=blockCoupled at every step (every
      "Time =" line) and no "Solving for p" line;
    - G4 mass (sanity): contRes = contResAbs/Q0 <= 1e-9 every step (Q0 = the
      free-shell series' reference flux, free_shell_timeseries.py, as G2's
      Q0), and qInlet, qOutlet within 1e-10 relative of the logged
      inletFlux/outletFlux at every step;
    - G6 (S19a section 4): linRes = ||Ax - b||/||b|| AND linResScaled (the
      row-sum-scaled residual, S19a section 2.5) each <= 1e-10 every step; a
      missing field fails (read as inf);
    - G7: kMatch = 0 and kresMatch = 0 exactly, every step;
    - G8r (networkViscosity kelvinVoigt only): r3Power <= 0 every step;
    - G9 (S19a section 4, the factor's memory and cost): numericPeakGB <= 1.0
      AND tNumeric <= 5.0 s every step; a missing field fails (read as inf);
      S22 section 3.3: under blockLinearSolver pardiso also mklPeakGB <= 2.0;
    - S22 (section 2.7; the backend is the config echo's blockLinearSolver, and
      a config line without it is a pre-S22 binary, UMFPACK): G22-4 under
      pardiso, perturbedPivots = 0 every step; G22-5 on an S22 log, rssGB <=
      4 every step; g22_backend, every record's linearSolver is the config's
      (each None where it does not apply; a missing field fails);
    - S19b section 4, C1 (blockCoupledLumen identityRows children; the control
      is the control): the config echoes blockCoupledLumen=identityRows and
      every record has lumenIdentityRows > 0 and n = 5 x nCells;
    - S19b section 4, D1 (blockCoupled children without the key: the default
      is engaged): the config echoes blockCoupledLumen=eliminate and every
      record has lumenEliminated > 0, lumenIdentityRows = 0 and n < 5 x
      nCells. C1 and D1 read nCells from the case mesh (None without it; a
      missing field fails). E1 (fsb_k13 against fsb_k13_id) is scored
      offline by scripts/s19b_e1_score.py from uMax/vsMax/pMax.
    Watches (not gated): bodySlipRms, symbolicCount, tSolve, rcond, r3Max,
    etaN; S19a: noffdiag (max; expected 0), rowSumMin (min), rowSumMax (max),
    S22: rcond and noffdiag are UMFPACK's (None under pardiso, which logs -1),
    irSteps (max, pardiso), solverThreads (the config echo),
    n and nnz of the last record (each over the records that carry the field;
    None if none does). G1, G2, G3 and V1 are scored against batch 64 OFFLINE."""
    num = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"

    def kv(line):
        return {k: float(v) for k, v in re.findall(r"(\w+)=" + num, line)}

    spec = spec or {}
    want = spec.get("s19") or {}
    out: dict = {"config_expected": want or None}
    cfg = re.search(r"^PHASE_CLOT_S19_CONFIG (.*)$", text, re.M)
    out["config"] = dict(re.findall(r"(\w+)=(\S+)", cfg.group(1))) if cfg else None
    out["config_matches_child"] = (None if not want else bool(out["config"]) and all(
        out["config"].get(k) == want.get(k) for k in ("momentumPressureCoupling", "networkViscosity")))
    # BATCH 68 (CE ruling on report 03, 2026-10-05): under R3 the echoed
    # networkViscosityTau must equal the child's tau_v_s (relative 1e-9), or a
    # wrong tau would pass V0. The solver echoes it on the config line (batch
    # 67 fsb_ctl_r3: "networkViscosityTau=0.00025"); missing or unparsable fails.
    out["config_tau_matches"] = None
    if want and want.get("networkViscosity") == "kelvinVoigt":
        try:
            got_tau = float((out["config"] or {})["networkViscosityTau"])
            ref_tau = float(want["networkViscosityTau"])
            out["config_tau_matches"] = abs(got_tau - ref_tau) <= 1e-9*abs(ref_tau)
        except (KeyError, TypeError, ValueError):
            out["config_tau_matches"] = False
        out["config_matches_child"] = bool(out["config_matches_child"]
                                           and out["config_tau_matches"])
    block = want.get("momentumPressureCoupling") == "blockCoupled"
    r3 = want.get("networkViscosity") == "kelvinVoigt"
    # S22: the backend, from the solver's own config echo.
    cfg22 = out["config"] or {}
    s22_log = "blockLinearSolver" in cfg22
    solver = (None if out["config"] is None else cfg22.get("blockLinearSolver", "umfpack"))
    pardiso = solver == "pardiso"
    out["linear_solver"] = solver
    lines = re.findall(r"^PHASE_CLOT_S19 (.*)$", text, re.M)
    rows = [kv(ln) for ln in lines]
    steps = len(re.findall(r"^Time = (\S+)", text, re.M))
    out["records"] = len(rows)
    out["steps"] = steps
    out["p_solves"] = len(re.findall(r"Solving for p,", text))
    engaged = sum("mode=blockCoupled" in ln for ln in lines)
    out["g0"] = (None if not block or steps == 0 else
                 bool(engaged == steps == len(rows) and out["p_solves"] == 0))
    out["linres_max"] = max((r.get("linRes", math.inf) for r in rows), default=None)
    out["linres_scaled_max"] = max((r.get("linResScaled", math.inf) for r in rows), default=None)
    out["g6"] = None if not rows else all(
        r.get("linRes", math.inf) <= S19_LINRES_MAX
        and r.get("linResScaled", math.inf) <= S19_LINRES_MAX for r in rows)
    out["kmatch_max"] = max((max(r.get("kMatch", math.inf), r.get("kresMatch", math.inf)) for r in rows), default=None)
    out["g7"] = None if not rows else all(r.get("kMatch") == 0.0 and r.get("kresMatch") == 0.0 for r in rows)
    out["numeric_peak_gb_max"] = max((r.get("numericPeakGB", math.inf) for r in rows), default=None)
    out["t_numeric_max"] = max((r.get("tNumeric", math.inf) for r in rows), default=None)
    out["mkl_peak_gb_max"] = (max((r.get("mklPeakGB", math.inf) for r in rows), default=None)
                              if pardiso else None)
    out["g9"] = None if not rows else all(
        r.get("numericPeakGB", math.inf) <= S19_NUMERIC_PEAK_GB_MAX
        and r.get("tNumeric", math.inf) <= S19_TNUMERIC_MAX
        and (not pardiso or r.get("mklPeakGB", math.inf) <= S22_MKL_PEAK_GB_MAX) for r in rows)
    # S22 section 7: G22-4 (pardiso) and G22-5 (any S22 log), every step.
    out["perturbed_pivots_max"] = (max((r.get("perturbedPivots", math.inf) for r in rows), default=None)
                                   if pardiso else None)
    out["g22_4"] = (None if not pardiso or not rows else
                    all(r.get("perturbedPivots", math.inf) == 0.0 for r in rows))
    out["rss_gb_max"] = (max((r.get("rssGB", math.inf) for r in rows), default=None)
                         if s22_log else None)
    out["g22_5"] = (None if not s22_log or not rows else
                    all(r.get("rssGB", math.inf) <= S22_RSS_GB_MAX for r in rows))
    rec_solvers = [m.group(1) if m else None for m in
                   (re.search(r"\blinearSolver=(\w+)", ln) for ln in lines)]
    out["g22_backend"] = (None if not s22_log or not rows else
                          all(v == solver for v in rec_solvers))
    # G4: Q0 and the logged patch fluxes come from the module G2 uses.
    q0 = None
    logged = {}
    try:
        import free_shell_timeseries as fst  # numpy; present on Colab
        if case is not None:
            m = re.search(r"inlet\s*\{[^}]*?value\s+uniform\s+([-+0-9.eE]+)",
                          (case / "0" / "p").read_text())
            dp = float(spec["inlet_pressure_m2_s2"]) if "inlet_pressure_m2_s2" in spec else float(m.group(1))
            q0 = fst.Q0_REF_M3_S * dp / fst.Q0_REF_DP
            for name in ("inletFlux", "outletFlux"):
                logged[name] = dict(fst.read_surface_field_value(case, name, "orientedSum(phi)"))
    except Exception as exc:  # the report says why rather than guessing
        out["g4_source_error"] = repr(exc)
    out["q0"] = q0
    cr = [r["contResAbs"]/q0 for r in rows if "contResAbs" in r] if q0 else []
    out["contres_max"] = max(cr, default=None)
    bad_flux = []
    for r in rows:
        for key, name in (("qInlet", "inletFlux"), ("qOutlet", "outletFlux")):
            ref = None
            for t, v in logged.get(name, {}).items():
                if abs(t - r.get("time", -1.0)) <= 1e-9*max(abs(t), 1e-30):
                    ref = v
            if ref is None or key not in r or abs(r[key] - ref) > S19_FLUX_MATCH*abs(ref):
                bad_flux.append({"time": r.get("time"), "key": key, "q": r.get(key), "logged": ref})
    out["flux_mismatches"] = len(bad_flux)
    out["first_flux_mismatch"] = bad_flux[:1]
    out["g4"] = (None if not rows or not q0 or not logged else
                 bool(len(cr) == len(rows) and max(cr) <= S19_CONTRES_MAX and not bad_flux))
    # S19b section 4: C1 on the identityRows control, D1 on the default.
    lumen = str(want.get("blockCoupledLumen", "eliminate")) if block else None
    out["lumen_expected"] = lumen
    try:
        n_cells = s19b_mesh_cells(case)
    except Exception as exc:  # the report says why rather than guessing
        n_cells = None
        out["n_cells_source_error"] = repr(exc)
    out["n_cells"] = n_cells
    cfg_lumen = s21_config(out["config"], "blockCoupledLumen")
    out["config_lumen"] = cfg_lumen
    out["c1"] = (None if lumen != "identityRows" or not rows or n_cells is None else
                 bool(cfg_lumen == "identityRows" and all(
                     r.get("lumenIdentityRows", 0.0) > 0 and r.get("n") == 5*n_cells
                     for r in rows)))
    out["d1"] = (None if lumen != "eliminate" or not rows or n_cells is None else
                 bool(cfg_lumen == "eliminate" and all(
                     r.get("lumenEliminated", 0.0) > 0 and r.get("lumenIdentityRows") == 0.0
                     and r.get("n", math.inf) < 5*n_cells for r in rows)))
    out["r3_power_max"] = max((r.get("r3Power", math.inf) for r in rows), default=None) if r3 else None
    out["g8r"] = (None if not r3 or not rows else
                  all(r.get("r3Power", math.inf) <= 0.0 for r in rows))
    out["watch"] = {
        "body_slip_rms_max": max((r.get("bodySlipRms", 0.0) for r in rows), default=None),
        "symbolic_count_last": rows[-1].get("symbolicCount") if rows else None,
        "t_solve_max": max((r.get("tSolve", 0.0) for r in rows), default=None),
        "rcond_min": (None if pardiso else
                      min((r.get("rcond", math.inf) for r in rows), default=None)),
        "lumen_identity_rows_last": rows[-1].get("lumenIdentityRows") if rows else None,
        "lumen_eliminated_last": rows[-1].get("lumenEliminated") if rows else None,
        "r3_max_max": max((r.get("r3Max", 0.0) for r in rows), default=None) if r3 else None,
        "eta_n": rows[-1].get("etaN") if (rows and r3) else None,
        "noffdiag_max": (None if pardiso else
                         max((r["noffdiag"] for r in rows if "noffdiag" in r), default=None)),
        "ir_steps_max": (max((r["irSteps"] for r in rows if "irSteps" in r), default=None)
                         if pardiso else None),
        "solver_threads": cfg22.get("solverThreads"),
        "row_sum_min_min": min((r["rowSumMin"] for r in rows if "rowSumMin" in r), default=None),
        "row_sum_max_max": max((r["rowSumMax"] for r in rows if "rowSumMax" in r), default=None),
        "n_last": rows[-1].get("n") if rows else None,
        "nnz_last": rows[-1].get("nnz") if rows else None,
    }
    return out


def _xu_eig_max(b) -> float:
    """Largest eigenvalue of the symmetric tensor (xx, xy, xz, yy, yz, zz)."""
    xx, xy, xz, yy, yz, zz = b
    if xz == 0.0 and yz == 0.0:
        m, r = 0.5*(xx + yy), math.hypot(0.5*(xx - yy), xy)
        return max(m + r, zz)
    p1 = xy*xy + xz*xz + yz*yz
    q = (xx + yy + zz)/3.0
    p = math.sqrt(((xx - q)**2 + (yy - q)**2 + (zz - q)**2 + 2*p1)/6.0)
    if p == 0.0:
        return q
    B = [[(xx - q)/p, xy/p, xz/p], [xy/p, (yy - q)/p, yz/p], [xz/p, yz/p, (zz - q)/p]]
    det = (B[0][0]*(B[1][1]*B[2][2] - B[1][2]*B[2][1])
           - B[0][1]*(B[1][0]*B[2][2] - B[1][2]*B[2][0])
           + B[0][2]*(B[1][0]*B[2][1] - B[1][1]*B[2][0]))
    phi = math.acos(max(-1.0, min(1.0, det/2.0)))/3.0
    return q + 2*p*math.cos(phi)


def _xu_read_tuples(path: Path) -> list[tuple]:
    """A vector OR tensor internalField (List<vector>, List<symmTensor>, ...):
    read_vectors() matches List<vector> only, and `be` is a symmTensor."""
    t = path.read_text(errors="replace")
    m = re.search(r"(?ms)internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;", t)
    if m:
        vals = [tuple(float(x) for x in r.split())
                for r in re.findall(r"\(([^()]+)\)", m.group(2))]
        if len(vals) != int(m.group(1)):
            raise ValueError(f"{path}: header says {m.group(1)} values, found {len(vals)}")
        return vals
    u = re.search(r"internalField\s+uniform\s+\(([^)]+)\)\s*;", t)
    if u:
        return [tuple(float(x) for x in u.group(1).split())]
    raise ValueError(f"{path}: cannot parse a vector or tensor field")


def _xu_field(d: Path, name: str, n: int, vector: bool = False):
    """One written field of the Xu case, or None if absent. Its errors name
    the TIME and the count, never an absolute path or which field happened to
    be read first: the report must read identically on a returned run and on
    any copy of it, whichever output convention wrote the fields
    (dry_run.py's theta_native_grades_identically)."""
    p = d / name
    if not p.is_file():
        return None
    try:
        v = _xu_read_tuples(p) if vector else read_scalars(p)
    except ValueError as error:
        raise ValueError(f"t = {d.name}: {name} unreadable "
                         f"({str(error).rsplit(': ', 1)[-1]})") from None
    if len(v) == 1:
        v = v*n
    if len(v) != n:
        raise ValueError(f"t = {d.name}: the written fields carry {len(v)} values "
                         f"for a {n}-cell mesh")
    return v


def _xu_band(interval, band):
    """A write-resolved onset (t_prev, t_w] against [lo, hi] ms: PASS if they
    intersect, else 'early' or 'late'."""
    lo, hi = band
    a, b = interval
    if b < lo:
        return "FAIL (early)"
    if a > hi:
        return "FAIL (late)"
    return "PASS"


def _xu_edge(ind: list, nx: int, ny: int, h: float) -> float:
    """P6 (b): the ind = 0.5 leading edge, max over rows, interpolated
    linearly between a row's LAST cell with ind >= 0.5 and its downstream
    neighbour (physics 06's instrument, probes/physics-06-xu-model.py)."""
    edge = 0.0
    for j in range(ny):
        row = ind[j*nx:(j + 1)*nx]
        on = [i for i, v in enumerate(row) if v >= 0.5]
        if not on:
            continue
        i6 = on[-1]
        x6 = (i6 + 0.5)*h
        if i6 + 1 < nx and row[i6 + 1] < 0.5:
            x6 += (row[i6] - 0.5)/(row[i6] - row[i6 + 1])*h
        edge = max(edge, x6)
    return edge


def _xu_props(case: Path) -> dict:
    props = case / "constant" / "phaseClotProperties"
    out = {}
    if props.is_file():
        for k, v in re.findall(r"(?m)^\s*(\w+)\s+(?:\[[^\]]*\]\s+)?([^;\s]+)\s*;",
                               props.read_text(errors="replace")):
            out.setdefault(k, v)
    return out


def _xu_patch_entry(path: Path, patch: str) -> str | None:
    if not path.is_file():
        return None
    t = path.read_text(errors="replace")
    i = t.find("boundaryField")
    m = re.search(rf"(?s)\b{patch}\s*\{{(.*?)\}}", t[i:]) if i >= 0 else None
    return " ".join(m.group(1).split()) if m else None


def xu2017_report(case: Path, text: str, spec: dict, result: dict) -> dict:
    """Design section 4 (A0-A7, P0, P1, N1-N7, P3-P9, B), each row
    {"verdict": PASS | FAIL | NOT-EVALUABLE | UNASSESSABLE, ...measured}."""
    num = r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|-?nan|-?inf)"

    def kv(line):
        return {k: float(v) for k, v in re.findall(r"(\w+)=" + num, line)}

    def rows_of(tag):
        return [kv(m) for m in re.findall(rf"^{tag} (.*)$", text, re.M)]

    rows: dict = {}
    gamma = float(spec["wall_shear_rate_1_s"])
    u_mean = float(spec["mean_velocity_m_s"])
    geo, clot = spec.get("geometry", {}), spec.get("clot", {})
    props = _xu_props(case)

    # ---- A0: config echo, per child (child.json, the case's dictionaries,
    # its 0/ boundary conditions, and the solver's own config lines).
    bad = []

    def pnum(k):
        try:
            return float(props[k])
        except (KeyError, ValueError):
            return None

    got = {"rho": pnum("rho"), "nu": pnum("nu"), "G": pnum("G"), "Gcore": pnum("Gcore"),
           "k0p": pnum("k0p"), "k0pcore": pnum("k0pcore"),
           "material_width_m": clot.get("material_width_m"), "epsilonC": pnum("epsilonC")}
    us_bottom = _xu_patch_entry(case / "0" / "Us", "bottom") or ""
    m = re.search(r"adhesiveStrength\s+([^;\s]+)", us_bottom)
    got["adhesive_strength_pa"] = float(m.group(1)) if m else None
    for k, (want, tol) in XU_A0.items():
        v = got.get(k)
        if v is None or not math.isclose(float(v), want, rel_tol=tol):
            bad.append(f"{k} {v} (want {want})")
    if spec.get("pressure_driven") is not False:
        bad.append(f"pressure_driven {spec.get('pressure_driven')!r}")
    u_in = _xu_patch_entry(case / "0" / "U", "inlet") or ""
    mu = re.search(r"type\s+fixedValue;\s*value\s+uniform\s+\(\s*(\S+)", u_in)
    if not mu or not math.isclose(float(mu.group(1)), u_mean, rel_tol=1e-9):
        bad.append(f"0/U inlet {u_in!r} (want fixedValue {u_mean})")
    if "zeroGradient" not in (_xu_patch_entry(case / "0" / "p", "inlet") or ""):
        bad.append("0/p inlet is not zeroGradient")
    if not math.isclose(u_mean, XU_U_MEAN.get(gamma, float("nan")), rel_tol=1e-4):
        bad.append(f"U_mean {u_mean} for {gamma} 1/s (design {XU_U_MEAN.get(gamma)})")
    if not math.isclose(u_mean, gamma*geo.get("channel_height_m", 0.0)/6.0, rel_tol=1e-9):
        bad.append("U_mean != gammaDot H / 6")
    # S21: bondHoldTaper (no) and advectReleased (yes) are built in and absent here.
    for k, want in (("bondHoldDilation", "4"),
                    ("releaseCoreExempt", "no"),
                    ("plateletTransportOnly", "yes"), ("alpha", "0.0"),
                    ("coreShell", "yes")):
        if props.get(k) != want:
            bad.append(f"{k} {props.get(k)!r} (want {want})")
    if spec.get("growth") is not False:
        bad.append("growth is not false")
    for k, want in XU_DOME.items():
        v = geo.get(k) if k == "injury_centre_x_m" else clot.get(k)
        if v is None or not math.isclose(float(v), float(want), rel_tol=1e-9):
            bad.append(f"dome {k} {v} (want {want})")
    fp_cfg = re.search(r"^PHASE_CLOT_FREEPLATELET_CONFIG .*?transportOnly=(\S+)", text, re.M)
    if text and (not fp_cfg or fp_cfg.group(1) not in ("1", "true", "yes")):
        bad.append("PHASE_CLOT_FREEPLATELET_CONFIG transportOnly not 1 in the log")
    bh = rows_of("PHASE_CLOT_BONDHOLD")
    if bh and any(r.get("dilation") != 4 or r.get("taper") != 0 for r in bh[:1]):
        bad.append("PHASE_CLOT_BONDHOLD dilation/taper differ from 4/0")
    rows["A0"] = {"verdict": "FAIL" if bad else "PASS", "mismatches": bad,
                  "echo": {**got, "U_mean": u_mean, "gamma": gamma}}

    # ---- the mesh and the written times
    nx, ny = int(geo.get("nx", 0)), int(geo.get("ny", 0))
    h = float(geo.get("h_m", 0.0))
    n = nx*ny
    ref = float(props.get("skeletonVolumeFraction", "0.45"))
    held_top = (int(spec.get("bond_hold_dilation", 4)) + 1)*h
    xs = [((k % nx) + 0.5)*h for k in range(n)]
    ys = [((k // nx) + 0.5)*h for k in range(n)]
    writes = [(t, d) for t, d in time_dirs(case) if t > 0.0]
    wt_ms = [t*1e3 for t, _ in writes]
    try:
        c0 = _xu_field(case / "0", "C", n, vector=True)
    except ValueError as e:
        c0 = None
        rows["mesh"] = {"note": f"0/C unreadable: {e}"}
    if c0 is not None:
        dev = max(max(abs(c[0] - x), abs(c[1] - y)) for c, x, y in zip(c0, xs, ys))
        rows["mesh"] = {"C_vs_structured_max_m": dev}
        if dev > 1e-12:
            rows["mesh"]["note"] = "0/C is not the structured i-fastest mesh; field rows UNASSESSABLE"
            writes = []

    def at_mark(ms):
        best = None
        for (t, d), tm in zip(writes, wt_ms):
            if abs(tm - ms) <= 0.5*float(spec["write_interval_s"])*1e3 + 1e-9:
                if best is None or abs(tm - ms) < abs(best[0]*1e3 - ms):
                    best = (t, d)
        return best

    # ---- A1: first-write fields
    a1: dict = {"verdict": "NOT-EVALUABLE"}
    try:
        mi_path = case.parents[1].parent / "AM2020_mural_geometry_gate/scripts/initialise_cases.py"
        sh0 = _xu_field(case / "0", "shell", n) if c0 is not None else None
        tC0 = _xu_field(case / "0", "thetaC", n) if sh0 else None
        tH0 = _xu_field(case / "0", "thetaH", n) if tC0 else None
        if mi_path.is_file() and sh0 and tC0 and tH0 and c0 is not None:
            ns: dict = {"__name__": "xu_initialiser"}
            exec(compile(mi_path.read_text(), str(mi_path), "exec"), ns)
            pts = list(zip(xs, ys))
            smp = int(clot["signed_distance_samples"])
            dcl = ns["half_ellipse_signed_distance"](pts, geo["injury_centre_x_m"],
                                                     clot["half_length_m"], clot["height_m"], smp)
            dco = ns["half_ellipse_signed_distance"](pts, clot["core_centre_x_m"],
                                                     clot["core_half_length_m"],
                                                     clot["core_height_m"], smp)
            eps, mw = geo["interface_width_m"], clot["material_width_m"]
            ds = max(abs(s - 0.5*(1 + math.tanh(d/mw))) for s, d in zip(sh0, dco))
            dt_ = max(abs(a + b - ref*0.5*(1 - math.tanh(d/eps)))
                      for a, b, d in zip(tC0, tH0, dcl))
            a1 = {"shell_max_abs_dev": ds, "theta_max_abs_dev": dt_}
            first = writes[0] if writes else None
            if first is not None:
                gf = _xu_field(first[1], "Gfield", n)
                shf = _xu_field(first[1], "shell", n)
                if gf and shf:
                    G, Gc = got["G"], got["Gcore"]
                    blend = [Gc*math.exp(s*math.log(G/Gc)) for s in shf]
                    a1["gfield_vs_blend_max_abs_pa"] = max(abs(a - b) for a, b in zip(gf, blend))
                    a1["first_write_s"] = first[0]
                    core_v = [abs(g - Gc) for g, s in zip(gf, shf) if s < 1e-3]
                    shel_v = [abs(g - G) for g, s in zip(gf, shf) if s > 0.999]
                    a1["literal_gfield_minus_Gcore_max_pa"] = max(core_v) if core_v else None
                    a1["literal_gfield_minus_G_max_pa"] = max(shel_v) if shel_v else None
                    a1["literal_note"] = (
                        "the literal A1 reading (Gfield 24.0 where shell < 1e-3, 1.2 where "
                        "shell > 0.999, to 1e-6) cannot hold for the solver's blend "
                        "Gcore exp(s ln(G/Gcore)) (.C L4697): at s = 1e-3 it is 0.0718 Pa "
                        "from 24. Scored: Gfield equals the blend of the written label.")
            scored = [a1["shell_max_abs_dev"], a1["theta_max_abs_dev"]]
            if "gfield_vs_blend_max_abs_pa" in a1:
                scored.append(a1["gfield_vs_blend_max_abs_pa"])
            a1["verdict"] = "PASS" if max(scored) <= XU_A1_TOL else "FAIL"
            if "gfield_vs_blend_max_abs_pa" not in a1:
                a1["verdict"] += " (0/ fields only; no Gfield at a first write)"
    except (ValueError, KeyError, OSError, ZeroDivisionError) as e:
        a1 = {"verdict": "UNASSESSABLE", "reason": str(e)}
    rows["A1"] = a1

    # ---- A2: the S18 structural clauses (s18_report's conformance and config)
    s18 = result.get("s18") or {}
    a2ok = s18.get("conformance_ok")
    rows["A2"] = {"verdict": ("NOT-EVALUABLE" if a2ok is None or s18.get("config_matches_child") is None
                              else "PASS" if a2ok and s18.get("config_matches_child") else "FAIL"),
                  "conformance": s18.get("conformance"),
                  "config_matches_child": s18.get("config_matches_child"),
                  "d_i_h_absent": "checked at build (check_s18_soft.py), not in a run"}

    # ---- A3: the carried S17b / S17 / S16 / S14 gates
    carried = {
        "no_fibre_invalid_fatal": (result.get("s17b") or {}).get("no_fibre_invalid_fatal"),
        "V1_slack_dd": (result.get("s17b") or {}).get("slack_dd"),
        "theta_T_le_theta_P": (result.get("s17b") or {}).get("taut_le_pending"),
        "W2": (result.get("s17b") or {}).get("w2"),
        "P2": (result.get("s17b") or {}).get("p2"),
        "start_up": ((result.get("s17b") or {}).get("startup_alternation") or {}).get("passes"),
        "G_proj": (result.get("s17") or {}).get("g_proj"),
        "G_rot": (result.get("s17") or {}).get("g_rot"),
        "G_eig": (result.get("s17") or {}).get("g_eig"),
        "G_uni": (result.get("s17") or {}).get("g_uni"),
        "S17split": (result.get("s17") or {}).get("s17split"),
        "G12r": (result.get("s16") or {}).get("g12r"),
        "S16w": (result.get("s16") or {}).get("s16w"),
        "S16m": (result.get("s16") or {}).get("s16m"),
        "G13": (result.get("s14") or {}).get("g13_elastic_number_bounded"),
    }
    failing = [k for k, v in carried.items() if v is False]
    unread = [k for k, v in carried.items() if v is None]
    rows["A3"] = {"verdict": "FAIL" if failing else ("PASS" if not unread else
                                                    "PASS (read) / NOT-EVALUABLE: " + ", ".join(unread)),
                  "gates": carried, "failing": failing,
                  "G_dep": "a kernel unit test (s17_fibre_kernel_test.cpp), checked at build"}

    # ---- A4: no growth, and the release line on every step
    step_times = [float(v) for v in TIME_LINE.findall(text)]
    bal = rows_of("PHASE_CLOT_CONTENT_BALANCE")
    tr = [r.get("contentTransferRate") for r in bal]
    rel_t = {float(t) for t, _c, _v in RELEASED_LINE.findall(text)}
    missing_rel = [t for t in step_times if not any(math.isclose(t, r, rel_tol=1e-9, abs_tol=1e-15)
                                                    for r in rel_t)] if len(rel_t) < len(step_times) else []
    nonzero = [(r.get("time"), r.get("contentTransferRate")) for r in bal
               if r.get("contentTransferRate") != 0.0]
    rows["A4"] = {"verdict": ("NOT-EVALUABLE" if not step_times else
                              "FAIL" if (nonzero or missing_rel or not bal or None in tr) else "PASS"),
                  "steps": len(step_times), "balance_records": len(bal),
                  "released_lines": len(rel_t), "transfer_nonzero_records": len(nonzero),
                  "first_nonzero": nonzero[:1], "steps_without_released_line": len(missing_rel)}

    rows["A5"] = {"verdict": "NOT-EVALUABLE",
                  "why": "scored OFFLINE: scripts/s17_offline_checks.py energy-inlet (phase-weighted Phi, "
                         "p_in from column 0 of the written p, rho/nu/h from the case; FAIL at R > 0.9915)"}

    # ---- A6: the outlet crossing (RE-RULED, DECISIONS 2026-09-30 "A6 RE-RULED")
    # A step with contentOuterFlux != 0 PASSES iff
    #     |contentResidual| <= 1e-6 |F| + R_abs,
    # R_abs computed from THIS run's own log: 10 x ulp(max contentVolume) / (min dt x 1e-6)
    # (m3/s). That is the ruling's stated value, ~7.6e-19 m3/s on xu5 (contentVolume ~1e-15 m3,
    # dt 2.604167e-6 s). NOTE (07b): the ruling's formula as WORDED, 10 x ulp(max cell V/dt),
    # evaluates to 2.5e-28 m3/s on this mesh, and that would FAIL the ruling's own healthy case
    # (iii), an arriving crossing whose residual sits at em72's round-off (up to 1.995e-22).
    # Both values are reported, so the Chief Engineer can overrule the reading.
    # Crossings are COUNTED at |F| > 1e-20 m3/s (XU_A6_FLOOR).
    vols = [r["contentVolume"] for r in bal if "contentVolume" in r]
    dts_a6 = [float(v) for v in re.findall(r"^deltaT = (\S+)", text, re.M)]
    dts_a6 = [v for v in dts_a6 if v > 0]
    r_abs = (XU_A6_ABS_ULPS*math.ulp(max(abs(v) for v in vols))/(min(dts_a6)*XU_A6_RATIO)
             if vols and dts_a6 else None)
    h_a6 = float(geo.get("h_m", 0.0))
    r_abs_worded = (XU_A6_ABS_ULPS*math.ulp(h_a6*h_a6*float(geo.get("depth_m", 0.0))/min(dts_a6))
                    if dts_a6 and h_a6 else None)
    scored = [r for r in bal if r.get("contentOuterFlux", 0.0) != 0.0]
    crossing = [r for r in scored if abs(r["contentOuterFlux"]) > XU_A6_FLOOR]
    trace = [r for r in scored if abs(r["contentOuterFlux"]) <= XU_A6_FLOOR]
    viol = ([] if r_abs is None else
            [r for r in scored
             if abs(r.get("contentResidual", 0.0)) > XU_A6_RATIO*abs(r["contentOuterFlux"]) + r_abs])
    fatal_outlet = bool(re.search(r"FOAM FATAL[^\n]*\n(?:[^\n]*\n){0,12}[^\n]*outlet", text))
    rows["A6"] = {"verdict": ("NOT-EVALUABLE" if not bal or (scored and r_abs is None) else
                              "FAIL" if (viol or fatal_outlet) else "PASS"),
                  "rule": "|contentResidual| <= 1e-6 |F| + R_abs on every step with F != 0",
                  "r_abs_m3_s": r_abs,
                  "r_abs_basis": "10 x ulp(max contentVolume) / (min dt x 1e-6), from this run's log",
                  "r_abs_as_worded_m3_s": r_abs_worded,
                  "crossing_floor_m3_s": XU_A6_FLOOR,
                  "records_scored": len(scored),
                  "records_with_outer_flux": len(crossing), "violations": len(viol),
                  "round_off_trace_records": len(trace),
                  "round_off_trace_max_m3_s": max((abs(r["contentOuterFlux"]) for r in trace), default=None),
                  "max_abs_outer_flux_in_violations": max((abs(r["contentOuterFlux"]) for r in viol), default=None),
                  "max_abs_residual_in_violations": max((abs(r["contentResidual"]) for r in viol), default=None),
                  "outlet_fatal": fatal_outlet}

    # ---- A7: the bond
    deb = rows_of("PHASE_CLOT_DEBOND")
    seeds = [r.get("seeds") for r in bh]
    rows["A7"] = {"verdict": ("NOT-EVALUABLE" if not deb or not seeds else
                              "PASS" if (all(r.get("debondedFaces", 1) == 0 for r in deb)
                                         and all(s == seeds[0] for s in seeds)) else "FAIL"),
                  "debonded_faces_max": max((r.get("debondedFaces", 0) for r in deb), default=None),
                  "bondhold_seeds_first": seeds[0] if seeds else None,
                  "bondhold_seeds_min": min(seeds) if seeds else None}

    # ---- the per-write field series
    series: list = []
    i270 = int(round(270e-6/h - 0.5)) if h else 0
    clot_cols = [i for i in range(nx) if XU_CLOT_X[0] < (i + 0.5)*h < XU_CLOT_X[1]]
    y1, y2 = 0.5*h, 1.5*h

    def gtop(U, i):
        u1, u2 = U[(ny - 1)*nx + i][0], U[(ny - 2)*nx + i][0]
        return (u1*y2*y2 - u2*y1*y1)/(y1*y2*(y2 - y1))

    field_error = None
    try:
        for t, d in writes:
            U = _xu_field(d, "U", n, vector=True)
            tC, tH = _xu_field(d, "thetaC", n), _xu_field(d, "thetaH", n)
            tI, tP, tB = (_xu_field(d, k, n) for k in ("thetaIntact", "thetaPending", "thetaBroken"))
            G, lab = _xu_field(d, "Gfield", n), _xu_field(d, "shell", n)
            Us, ref_net = _xu_field(d, "Us", n, vector=True), _xu_field(d, "thetaNetRef", n)
            rel = _xu_field(d, "released", n)
            if None in (U, tC, tH, tI, tP, tB, G, lab):
                raise ValueError(f"{d.name}: a field the Xu rows read is missing "
                                 f"(U, thetaC/H, thetaIntact/Pending/Broken, Gfield, shell)")
            th = [a + b for a, b in zip(tC, tH)]
            thp = [max(v, 0.0) for v in th]
            tot = sum(thp)
            ind = [v/ref for v in thp]
            dstar = [(p + b)/v if v > 0 else 0.0 for p, b, v in zip(tP, tB, th)]
            dmg = [ind[k] >= XU_NET and dstar[k] > XU_DAMAGE for k in range(n)]
            w: dict = {"t_ms": t*1e3}
            w["shell_damaged"] = sum(1 for k in range(n) if dmg[k] and lab[k] >= 0.5)
            bulk = [ind[k] >= XU_NET and G[k] >= XU_BULK_G and ys[k] - held_top >= XU_BULK_DZ
                    for k in range(n)]
            w["bulk_damaged"] = sum(1 for k in range(n) if dmg[k] and bulk[k])
            w["band_damaged"] = sum(1 for k in range(n) if dmg[k] and lab[k] < 0.5 and not bulk[k])
            w["damaged_content"] = (sum(v*s for v, s in zip(thp, dstar))/tot) if tot else None
            w["beyond"] = (sum(v for v, x in zip(thp, xs) if x > XU_BEYOND_X)/tot) if tot else None
            w["leading_edge_m"] = _xu_edge(ind, nx, ny, h)
            # content at the outlet: network in the last column (P0 turns into a watch)
            w["outlet_network"] = any(ind[j*nx + nx - 1] >= XU_NET for j in range(ny))
            zone = [k for k in range(n) if ind[k] >= XU_NET and G[k] >= XU_BULK_G]
            wz = sum(thp[k] for k in zone)
            w["D_m"] = (sum(thp[k]*Us[k][0] for k in zone)/wz) if (Us and wz) else None
            if ref_net:
                w["drained"] = sum(1 for k in range(n) if ref_net[k] >= 1e-4
                                   and th[k]/ref_net[k] < 0.1)
            if rel:
                w["released_core"] = sum(1 for k in range(n) if rel[k] >= 0.5 and lab[k] < 0.5)
            g270 = gtop(U, i270)
            gmax = max(gtop(U, i) for i in clot_cols) if clot_cols else None
            w["top_wall_270_over_gamma"] = g270/gamma
            w["top_wall_clot_max_over_gamma"] = gmax/gamma if gmax is not None else None
            w["P1_ratio"] = gmax/g270 if gmax is not None and g270 else None
            series.append(w)
    except (ValueError, OSError, ZeroDivisionError, IndexError) as e:
        field_error = str(e)
        series = []

    def first(key):
        for i, w in enumerate(series):
            if w.get(key):
                return ((series[i - 1]["t_ms"] if i else 0.0), w["t_ms"])
        return None

    def at(ms):
        for w in series:
            if abs(w["t_ms"] - ms) <= 0.5*float(spec["write_interval_s"])*1e3 + 1e-9:
                return w
        return None

    na = {"verdict": "UNASSESSABLE", "reason": field_error} if field_error else None
    if not series and not field_error:
        na = {"verdict": "NOT-EVALUABLE", "reason": "no written times beyond 0"}

    # ---- P0: the flow rate at every step until content reaches the outlet
    # (the first write with network in the outlet column; scored strictly
    # before the write preceding it, so a crossing between writes cannot leak
    # into the scored steps), then a WATCH; and the developed top-wall shear.
    diag = rows_of("PHASE_CLOT_DIAGNOSTIC")
    reach = next((i for i, w in enumerate(series) if w.get("outlet_network")), None)
    t_cross = (None if reach is None else
               (series[reach - 1]["t_ms"] if reach else 0.0)*1e-3)
    q = [(r.get("time"), r["outletMeanUx"]/u_mean) for r in diag if "outletMeanUx" in r]
    scored_q = [(t, v) for t, v in q if t_cross is None or (t is not None and t < t_cross)]
    out_q = [(t, v) for t, v in scored_q if not XU_P0_Q[0] <= v <= XU_P0_Q[1]]
    p0 = {"flow_ratio_records": len(scored_q),
          "flow_ratio_min": min((v for _t, v in scored_q), default=None),
          "flow_ratio_max": max((v for _t, v in scored_q), default=None),
          "flow_outside_band": len(out_q), "first_outside": out_q[:1],
          "watch_after_content_at_outlet_from_s": t_cross,
          "watch_flow_ratio_after": [v for t, v in q if t_cross is not None
                                     and t is not None and t >= t_cross][-1:]}
    # P0's wall-shear half, then P1
    if series:
        tw = [(w["t_ms"], w["top_wall_270_over_gamma"]) for w in series]
        bad_tw = [x for x in tw if not XU_P0_TW[0] <= x[1] <= XU_P0_TW[1]]
        p0.update({"top_wall_270_min": min(v for _t, v in tw), "top_wall_270_max": max(v for _t, v in tw),
                   "top_wall_outside_band": len(bad_tw), "first_top_wall_outside": bad_tw[:1]})
        p1r = series[0]["P1_ratio"]
        rows["P1"] = {"verdict": ("NOT-EVALUABLE" if p1r is None else
                                  "PASS" if XU_P1[0] <= p1r <= XU_P1[1] else "FAIL"),
                      "t_ms": series[0]["t_ms"], "ratio": series[0]["P1_ratio"], "predicted": XU_P1_PRED,
                      "band": XU_P1}
    else:
        rows["P1"] = dict(na)
    p0["verdict"] = ("NOT-EVALUABLE" if not scored_q else
                     "FAIL" if (out_q or p0.get("top_wall_outside_band")) else
                     "PASS" if series else "PASS (flow rate) / wall shear NOT-EVALUABLE")
    rows["P0"] = p0

    # N1: dt
    dts = [float(v) for v in re.findall(r"^deltaT = (\S+)", text, re.M)]
    if dts:
        at_cap = sum(1 for v in dts if math.isclose(v, XU_DT_CAP, rel_tol=1e-5))
        run = longest = 0
        for v in dts:
            run = run + 1 if v < 0.5*XU_DT_CAP else 0
            longest = max(longest, run)
        share = at_cap/len(dts)
        rows["N1"] = {"verdict": "FAIL" if (longest > XU_N1_RUN or share < XU_N1_SHARE) else "PASS",
                      "steps": len(dts), "share_at_cap": share, "longest_run_below_half_cap": longest,
                      "dt_min": min(dts), "dt_max": max(dts)}
    else:
        rows["N1"] = {"verdict": "NOT-EVALUABLE"}
    # N2: wall per step (ClockTime, contended at CONCURRENCY 2)
    ex = re.findall(r"^ExecutionTime = (\S+) s\s+ClockTime = (\S+) s", text, re.M)
    if ex and step_times:
        per = float(ex[-1][1])/len(step_times)
        rows["N2"] = {"verdict": "PASS" if XU_N2[0] <= per <= XU_N2[1] else "FAIL",
                      "clock_s_per_step": per, "band": XU_N2}
    else:
        rows["N2"] = {"verdict": "NOT-EVALUABLE"}
    # N3: G8 (VS_SPLIT slip) against 4 u_max
    slips = [r["slip"] for r in rows_of("PHASE_CLOT_VS_SPLIT") if "slip" in r]
    lim = XU_N3_UMAX*1.5*u_mean
    over = sum(1 for s in slips if s > lim)
    rows["N3"] = {"verdict": ("NOT-EVALUABLE" if not slips else
                              "FAIL" if over > XU_N3_STEPS else "PASS"),
                  "limit_m_s": lim, "steps_over": over, "slip_max": max(slips) if slips else None}
    # N7: nuNet per write
    s17r = [(r.get("time"), r["nuNetMax"]) for r in rows_of("PHASE_CLOT_S17") if "nuNetMax" in r]
    per_write = []
    for tw_s, _d in writes:
        prior = [v for tt, v in s17r if tt is not None and tt <= tw_s*(1 + 1e-9)]
        if prior:
            per_write.append(prior[-1])
    rows["N7"] = {"verdict": ("NOT-EVALUABLE" if not per_write else
                              "PASS" if max(per_write) < XU_N7 else "FAIL"),
                  "max_at_writes": max(per_write) if per_write else None,
                  "max_all_records": max((v for _t, v in s17r), default=None)}

    # P3: shell damage
    pred, band, mark, dc_pred = XU_P3.get(gamma, (None, None, None, None))
    if series and pred is not None:
        onset = first("shell_damaged")
        wm = at(mark)
        dc = wm["damaged_content"] if wm else None
        v1 = _xu_band(onset, band) if onset else "FAIL (none by the last write)"
        v2 = ("NOT-EVALUABLE (mark not reached)" if dc is None else
              "PASS" if 0.5*dc_pred <= dc <= 2*dc_pred else "FAIL")
        rows["P3"] = {"verdict": "FAIL" if ("FAIL" in v1 or v2 == "FAIL") else "PASS",
                      "first_shell_damage_interval_ms": onset, "predicted_ms": pred, "band_ms": band,
                      "onset_verdict": v1, "damaged_content_at_mark": dc, "mark_ms": mark,
                      "damaged_content_predicted": dc_pred, "damaged_content_verdict": v2}
    else:
        rows["P3"] = dict(na or {"verdict": "NOT-EVALUABLE"})
    # P4 = N4: the bulk core
    p4 = XU_P4.get(gamma)
    if series and p4 is not None:
        fb, lmark, lband = p4
        onset = first("bulk_damaged")
        out = {"first_bulk_damage_interval_ms": onset,
               "extendable": onset is None,
               "rule": "a child with bulk-core damage is not extended beyond 10 ms (OI-44)"}
        if fb is None:
            v = "FAIL (early: bulk damage before 10 ms)" if onset else "PASS"
            lam = None
            wm = at_mark(lmark)
            if wm is not None:
                be = _xu_field(wm[1], "be", n, vector=True)
                G = _xu_field(wm[1], "Gfield", n)
                tC, tH = _xu_field(wm[1], "thetaC", n), _xu_field(wm[1], "thetaH", n)
                if be and G and tC and tH:
                    lam = max((math.sqrt(max(_xu_eig_max(be[k]), 0.0)) for k in range(n)
                               if (tC[k] + tH[k])/ref >= XU_NET and G[k] >= XU_BULK_G
                               and ys[k] - held_top >= XU_BULK_DZ), default=None)
            out.update({"max_lambda1_bulk_at_mark": lam, "lambda1_mark_ms": lmark,
                        "lambda1_band": lband})
            vl = ("NOT-EVALUABLE (mark not reached or be absent)" if lam is None else
                  "PASS" if lband[0] <= lam <= lband[1] else "FAIL")
            out["lambda1_verdict"] = vl
            out["verdict"] = "FAIL" if ("FAIL" in v or vl == "FAIL") else "PASS"
            out["onset_verdict"] = v
        else:
            v = _xu_band(onset, fb) if onset else "FAIL (none by the last write)"
            out.update({"band_ms": fb, "onset_verdict": v,
                        "verdict": "FAIL" if "FAIL" in v else "PASS"})
        rows["P4_N4"] = out
    else:
        rows["P4_N4"] = dict(na or {"verdict": "NOT-EVALUABLE"})
    # B: reported, not scored
    rows["B"] = {"verdict": "REPORTED",
                 "first_label_core_damage_outside_bulk_ms": first("band_damaged") if series else None}
    # P5: D
    pm = XU_P5.get(gamma)
    if series and pm:
        wm = at(pm[0])
        dval = wm["D_m"] if wm else None
        cross = None
        for i, w in enumerate(series):
            if w["D_m"] is not None and w["D_m"] >= 2e-6:
                prev = series[i - 1] if i else None
                if prev and prev["D_m"] is not None and w["D_m"] != prev["D_m"]:
                    cross = prev["t_ms"] + (2e-6 - prev["D_m"])/(w["D_m"] - prev["D_m"])*(w["t_ms"] - prev["t_ms"])
                else:
                    cross = w["t_ms"]
                break
        rows["P5"] = {"verdict": ("NOT-EVALUABLE (mark not reached or Us absent)" if dval is None else
                                  "PASS" if 0.5*pm[1]*1e-6 <= dval <= 2*pm[1]*1e-6 else "FAIL"),
                      "D_um_at_mark": dval*1e6 if dval is not None else None, "mark_ms": pm[0],
                      "predicted_um": pm[1], "passes_2um_at_ms": cross}
    else:
        rows["P5"] = dict(na or {"verdict": "NOT-EVALUABLE"})
    # t = 0, from the written 0/ contents: P6's "minus its t = 0 value" and
    # the start of N6's "first 2.5 ms" (t = 0 is not a written time)
    zero: dict = {}
    try:
        tC0, tH0 = _xu_field(case / "0", "thetaC", n), _xu_field(case / "0", "thetaH", n)
        if tC0 and tH0:
            th0 = [max(a + b, 0.0) for a, b in zip(tC0, tH0)]
            s0 = sum(th0)
            if s0 > 0:
                zero = {"t_ms": 0.0,
                        "beyond": sum(v for v, x in zip(th0, xs) if x > XU_BEYOND_X)/s0,
                        "leading_edge_m": _xu_edge([v/ref for v in th0], nx, ny, h),
                        "damaged_content": 0.0}
    except ValueError:
        zero = {}
    # P6: extrusion
    if series and gamma in XU_P6:
        beyond0 = zero.get("beyond")
        marks, verdict = [], "PASS"
        for ms, pa, pb in XU_P6[gamma]:
            wm = at(ms)
            if wm is None:
                marks.append({"mark_ms": ms, "verdict": "NOT-EVALUABLE (not reached)"})
                continue
            da = wm["beyond"] - (beyond0 if beyond0 is not None else XU_P6_T0)
            db = wm["leading_edge_m"] - XU_EDGE_REF_X
            ok = (0.5*pa <= da <= 2*pa) and (0.5*pb*1e-6 <= db <= 2*pb*1e-6)
            verdict = verdict if ok else "FAIL"
            marks.append({"mark_ms": ms, "content_beyond_increment": da, "predicted": pa,
                          "leading_edge_increment_um": db*1e6, "predicted_um": pb,
                          "verdict": "PASS" if ok else "FAIL"})
        rows["P6"] = {"verdict": verdict, "t0_content_beyond": beyond0,
                      "t0_content_beyond_design": XU_P6_T0, "marks": marks}
    else:
        rows["P6"] = dict(na or {"verdict": "NOT-EVALUABLE"})
    # N5: tearing watch
    if series and all("drained" in w for w in series):
        ref5 = at(XU_N5_REF_MS)
        late = [w["drained"] for w in series if w["t_ms"] > XU_N5_REF_MS + 1e-9]
        rows["N5"] = {"verdict": ("NOT-EVALUABLE" if ref5 is None else
                                  "PASS" if not late or max(late) <= ref5["drained"] + XU_N5_EXTRA
                                  else "FAIL"),
                      "drained_at_5ms": ref5["drained"] if ref5 else None,
                      "drained_max_after": max(late) if late else None}
    else:
        rows["N5"] = dict(na or {"verdict": "NOT-EVALUABLE", "reason": "thetaNetRef not written"})
    # N6: saturation (250 and 500)
    if series and gamma in (250.0, 500.0):
        def rate(key, t0):
            a = zero if t0 == 0.0 else at(t0)
            b = at(t0 + 2.5)
            c, d = at(7.5), at(10.0)
            if None in (a, b, c, d) or not a or a.get(key) is None:
                return None, None
            early = (b[key] - a[key])/2.5
            lateval = (d[key] - c[key])/2.5
            return (lateval/early if early else None), d[key] - c[key]
        ra, dA = rate("beyond", 0.0)
        rb, dB = rate("leading_edge_m", 0.0)
        dc_first = next((w["t_ms"] for w in series if w.get("damaged_content")), None)
        rc, _dC = rate("damaged_content", dc_first) if dc_first is not None else (None, None)
        if ra is None or rb is None:
            rows["N6"] = {"verdict": "NOT-EVALUABLE (the 10 ms write is needed)"}
        else:
            stopped = (ra < XU_N6_RATIO and rb < XU_N6_RATIO
                       and dA*XU_N6_SQRT < XU_N6_TOL["a"] and dB*XU_N6_SQRT < XU_N6_TOL["b"])
            rows["N6"] = {"verdict": "FAIL" if stopped else "PASS",
                          "ratio_content_beyond": ra, "ratio_leading_edge": rb,
                          "ratio_damaged_content": rc,
                          "projection_sqrt_t_content": dA*XU_N6_SQRT,
                          "projection_sqrt_t_edge_um": dB*XU_N6_SQRT*1e6,
                          "projection_constant_edge_um": dB*76*1e6,
                          "predicted_250": "(i) 3.93, (ii) 2.43, (iii) 2.28: NOT saturated",
                          "reference_window": "0-2.5 ms (t = 0 from the 0/ contents) for (i) and (ii); "
                                              "(iii) from its first nonzero write"}
    else:
        rows["N6"] = {"verdict": "NOT-EVALUABLE" if gamma in (250.0, 500.0) else "NOT-APPLICABLE (1000 and 2000 stop at N4)"}
    # P8: released core
    if series and all("released_core" in w for w in series):
        rc_first = next((w for w in series if w["released_core"] > 0), None)
        rows["P8"] = {"verdict": ("REPORTED (OI-44 probe)" if gamma == 2000.0 else
                                  "FAIL" if rc_first else "PASS"),
                      "first_released_core": ({"t_ms": rc_first["t_ms"], "cells": rc_first["released_core"]}
                                              if rc_first else None)}
    else:
        rows["P8"] = dict(na or {"verdict": "NOT-EVALUABLE", "reason": "released not written"})
    # P9: the matched time with Xu
    if series:
        last = series[-1]
        ours = last["top_wall_clot_max_over_gamma"]*gamma if last["top_wall_clot_max_over_gamma"] else None
        ratio = ours/XU_P9_XU[gamma] if ours else None
        rows["P9"] = {"verdict": ("REPORTED (OI-44 probe)" if gamma == 2000.0 else
                                  "NOT-EVALUABLE" if ratio is None else
                                  "FAIL" if ratio >= XU_P9_FAIL else "PASS"),
                      "t_ms": last["t_ms"], "ours_over_gamma": last["top_wall_clot_max_over_gamma"],
                      "ours_over_xu": ratio, "xu_1_s": XU_P9_XU.get(gamma)}
    else:
        rows["P9"] = dict(na or {"verdict": "NOT-EVALUABLE"})

    structural = ("A0", "A1", "A2", "A3", "A4", "A6", "A7")
    out = {"rows": rows, "per_write": series,
           "structural_ok": all(str(rows[k].get("verdict", "")).startswith("PASS") for k in structural),
           "structural_failing": [k for k in structural if str(rows[k].get("verdict", "")).startswith("FAIL")],
           "prediction_failing": [k for k in ("P0", "P1", "N1", "N2", "N3", "N5", "N6", "N7",
                                              "P3", "P4_N4", "P5", "P6", "P8", "P9")
                                  if str(rows.get(k, {}).get("verdict", "")).startswith("FAIL")],
           "extendable": (rows.get("P4_N4") or {}).get("extendable")}
    return out


def xu2017_table(children: dict) -> dict:
    rows = []
    for c in sorted(children.values(), key=lambda r: r["wall_shear_rate_1_s"]):
        if c["series"] != XU_SERIES_NAME:
            continue
        x = c.get("xu2017") or {}
        rows.append({"child_id": c["child_id"], "wall_shear_rate_1_s": c["wall_shear_rate_1_s"],
                     "admissible": c["admissible"], "structural_ok": x.get("structural_ok"),
                     "structural_failing": x.get("structural_failing"),
                     "prediction_failing": x.get("prediction_failing"),
                     "extendable": x.get("extendable"),
                     "verdicts": {k: v.get("verdict") for k, v in (x.get("rows") or {}).items()
                                  if isinstance(v, dict) and "verdict" in v}})
    return {"rows": rows,
            "not_claimed": "no claim at t = 0.2 s; Figure 5's outlines are a labelled watch "
                           "(design section 5)"}


COMPILE_FAILURE = re.compile(
    r"(?m)\berror:|^make(?:\[\d+\])?: \*\*\* |^\s*Error [0-9]"
)
COMPLETED = ("End\n", "ExecutionTime =")


def find_field(d: Path, name: str) -> Path | None:
    exact = d / name
    if exact.is_file():
        return exact
    cands = [
        p for p in d.iterdir()
        if p.is_file() and p.name.startswith(name + ".") and p.name.count(".") == 1
    ]
    return cands[0] if len(cands) == 1 else None


def read_scalars(path: Path) -> list[float]:
    t = path.read_text(errors="replace")
    m = re.search(
        r"(?ms)internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;", t
    )
    if m:
        return [float(v) for v in m.group(2).split()]
    u = re.search(r"internalField\s+uniform\s+([-\d.eE+]+)\s*;", t)
    if u:
        return [float(u.group(1))]
    raise ValueError(f"{path}: cannot parse scalars")


def read_patch_scalars(path: Path, patch: str) -> list[float]:
    """The `value` list on one patch of a surfaceScalarField or volScalarField.

    Needed because the occlusion measurement is the volumetric flux through
    the OUTLET, and that lives in phi's boundaryField, not its internalField.
    Taken from phi rather than rebuilt from U and the face areas: phi is what
    the pressure equation actually conserved, and a reconstruction can differ
    from it by exactly the amount the continuity error is.
    """
    t = path.read_text(errors="replace")
    i = t.find("boundaryField")
    if i < 0:
        raise ValueError(f"{path}: no boundaryField")
    j = t.find(patch, i)
    if j < 0:
        raise ValueError(f"{path}: no patch {patch!r}")
    seg = t[j:]
    m = re.search(
        r"(?ms)value\s+nonuniform\s+List<scalar>\s*\n?\s*(\d+)\s*\n?\(\s*(.*?)\s*\)\s*;",
        seg,
    )
    if m:
        return [float(v) for v in m.group(2).split()]
    u = re.search(r"value\s+uniform\s+([-\d.eE+]+)\s*;", seg)
    if u:
        return [float(u.group(1))]
    raise ValueError(f"{path}: patch {patch!r} carries no readable value")


def open_channel_flux(d: Path, spec: dict) -> float | None:
    """Flux this channel would pass with NO clot, kinematic-pressure driven.

    Plane Poiseuille in series: dp = Q * integral(12*nu / (W * H(x)^3) dx),
    with H(x) the raised-cosine throat profile the mesh was bent to.

    READ FROM THE CASE, NOT FROM THE CONTROLS. The driving pressure comes from
    the inlet's own fixedValue in 0/p and nu from constant/phaseClotProperties,
    because those are what the solver actually ran. The child spec the assessor
    receives carries `geometry` but NOT `pressure_driven` or
    `inlet_pressure_m2_s2`, so a controls-based version of this returned None
    for every child and reported no reference at all -- silently.

    Returns None when the inlet is not a fixedValue pressure or the child has
    no throat, so an absent reference can never be read as a measured one.
    """
    g = spec.get("geometry") or {}
    hth = g.get("throat_height_m")
    if hth is None or g.get("taper_start_x_m") is None:
        return None

    pfile = d / "0" / "p"
    if not pfile.exists():
        return None
    m = re.search(
        r"inlet\s*\{[^}]*?type\s+fixedValue\s*;[^}]*?value\s+uniform\s+"
        r"([-\d.eE+]+)\s*;",
        pfile.read_text(errors="replace"),
        re.S,
    )
    if not m:
        return None
    dp = float(m.group(1))
    if dp <= 0.0:
        return None

    props = d / "constant" / "phaseClotProperties"
    nu = None
    if props.exists():
        mn = re.search(r"^\s*nu\s+\[[^\]]*\]\s+([-\d.eE+]+)\s*;",
                       props.read_text(errors="replace"), re.M)
        if mn:
            nu = float(mn.group(1))
    if not nu:
        return None

    hw = g["channel_height_m"]
    lx = g["channel_length_m"]
    x1 = g["taper_start_x_m"]
    x2 = x1 + g["taper_length_m"]
    width = g["depth_m"]

    def height(x: float) -> float:
        if x <= x1:
            return hw
        if x >= x2:
            return hth
        t = (x - x1) / (x2 - x1)
        return hw - (hw - hth) * 0.5 * (1.0 - math.cos(math.pi * t))

    n = 20000
    dx = lx / n
    resistance = 0.0
    for i in range(n):
        h = height((i + 0.5) * dx)
        resistance += 12.0 * nu / (width * h ** 3) * dx
    return dp / resistance


def occlusion_measurement(d: Path, spec: dict) -> dict:
    """Flow through a narrowing channel as the migrating shell fills it.

    THE OBSERVABLE IS THE OUTLET FLUX, which is the velocity drop Dr Li asked
    for. It is read from phi's boundaryField rather than rebuilt from U: phi is
    what the pressure equation actually conserved, and a reconstruction differs
    from it by exactly the continuity error.

    THE BLOCKAGE IS MEASURED, NOT DECLARED. The local gap is H(x), so the same
    clot blocks a different fraction at every station; the open gap is counted
    from the c field at the narrowest station the clot has reached.

    RELEASED WAS A TRIPWIRE UNTIL 2026-09-04; IT IS NOW AN OBSERVABLE.

    It used to be a tripwire because the obstruction term was
    fvm::Sp(dragCoeff*brinkmanHold, U) with brinkmanHold = max(1 - released, 0),
    so any cell that released stopped resisting the flow -- a released embolus
    was hydraulically invisible and the run silently stopped answering the
    question it was built for.

    The drag is now the relative-velocity pair
    fvm::Sp(dragCoeff, U) - dragCoeff*released*Ubody, so released material
    still resists flow THROUGH it and only stops resisting when the fluid
    travels with it. A nonzero released is therefore no longer a reason to
    distrust the flux; it is the embolus the study is about. Still reported
    every time, because which cells are free remains the thing that decides
    whether a result is about a mural clot or a travelling one.
    """
    g = spec["geometry"]
    nx, ny = g["nx"], g["ny"]
    h = g["channel_length_m"] / nx
    hw = g["channel_height_m"]
    hth = g.get("throat_height_m")
    x1 = g.get("taper_start_x_m")
    x2 = (x1 + g["taper_length_m"]) if x1 is not None else None

    def gap(x):
        if hth is None or x <= x1:
            return hw
        if x >= x2:
            return hth
        s = (x - x1) / (x2 - x1)
        return hw - (hw - hth) * 0.5 * (1.0 - math.cos(math.pi * s))

    times = sorted(
        (float(p.name), p) for p in d.iterdir()
        if p.is_dir() and re.fullmatch(r"[\d.]+(e[-+]?\d+)?", p.name)
    )
    series = []
    for t, td in times:
        phi = find_field(td, "phi")
        c = phase_c(td)
        if phi is None or c is None:
            continue
        try:
            q_out = sum(read_patch_scalars(phi, "outlet"))
        except ValueError:
            continue
        if len(c) < nx * ny:
            continue
        rel = find_field(td, "released")
        released_max = max(read_scalars(rel)) if rel is not None else None
        sh = find_field(td, "shell")
        shell = read_scalars(sh) if sh is not None else None

        # leading edge of real clot, and the gap it has reached
        lead_i = max(
            (i for i in range(nx)
             if any(c[i + nx * j] < -0.5 for j in range(ny))),
            default=None,
        )
        row = {
            "time_s": t,
            "outlet_flux_m3_s": q_out,
            "released_max": released_max,
            "clot_lead_x_m": (lead_i + 0.5) * h if lead_i is not None else None,
        }
        if lead_i is not None:
            x = (lead_i + 0.5) * h
            local = gap(x)
            # blockage at the narrowest station the clot occupies
            worst, worst_x = 0.0, None
            for i in range(nx):
                col = sum(1 for j in range(ny) if c[i + nx * j] < -0.5)
                if not col:
                    continue
                xi = (i + 0.5) * h
                b = col / ny            # fraction of the LOCAL gap, ny cells tall
                if b > worst:
                    worst, worst_x = b, xi
            row["local_gap_at_lead_m"] = local
            row["max_blockage_fraction"] = worst
            row["max_blockage_x_m"] = worst_x
        if shell is not None:
            for name, sel in (("core", lambda v: v < 0.5),
                              ("shell", lambda v: v >= 0.5)):
                idx = [i for i in range(nx * ny)
                       if c[i] < -0.5 and sel(shell[i])]
                row[f"{name}_centroid_x_m"] = (
                    sum(((i % nx) + 0.5) * h for i in idx) / len(idx)
                    if idx else None
                )
        series.append(row)

    out = {"throat_height_m": hth, "series": series}
    if series:
        flux = [r["outlet_flux_m3_s"] for r in series]
        out["initial_flux_m3_s"] = flux[0]
        out["final_flux_m3_s"] = flux[-1]
        out["min_flux_m3_s"] = min(flux)
        # AGAINST THE FIRST SAMPLE, AND THAT IS WHY IT READ 0.0.
        #
        # Run 20260903-114913 reported flux_drop_fraction = 0.0 in both
        # children while the flux ROSE 24% and then fell 11%. The clot starts
        # upstream in the wide section, which carries only 5.4% of the
        # channel's resistance against the throat's 70.7%, so the clot LEAVING
        # the wide section removes more resistance than the film arriving in
        # the throat adds. The first sample is therefore the minimum by
        # construction and a ratio against it can never be positive.
        #
        # Kept, because it is the number two run records already quote, but it
        # is no longer the only one.
        out["flux_drop_fraction"] = (
            1.0 - min(flux) / flux[0] if flux[0] else float("nan")
        )
        # THE PURELY MEASURED REFERENCE: the clearest the channel ever got in
        # this run. Well posed for a journey child, where the flux peaks
        # before the clot arrives. Meaningless for a lodged child, whose clot
        # is in the throat from t = 0 -- hence the analytic reference below.
        out["max_flux_m3_s"] = max(flux)
        out["flux_drop_from_peak"] = (
            1.0 - flux[-1] / max(flux) if max(flux) else float("nan")
        )
        # THE GEOMETRIC REFERENCE: what this channel would pass with no clot
        # at all, from plane-Poiseuille resistances in series along the
        # measured throat profile. This is the only reference that works for
        # BOTH arms, and it is what "the velocity drop before and after the
        # clot blocks the vessel" actually asks for.
        #
        # It is a LUBRICATION estimate: it neglects streamwise deceleration
        # through the taper. Calibrated, not assumed -- against run
        # 20260903-114913 it reproduced the solver's own flux to 3.3-4.0% at
        # 10, 20 and 30 ms when fed that run's written c and poroPerm fields,
        # and it matches the analytic uniform-channel result to 1.1e-7. So it
        # reads about 3.5% high and the fractions below are correspondingly
        # conservative by that much.
        q_open = open_channel_flux(d, spec)
        out["open_channel_flux_1d_m3_s"] = q_open
        out["open_channel_reference"] = (
            "1D plane-Poiseuille resistances in series along the throat "
            "profile; lubrication, ~3.5% high, calibrated against run "
            "20260903-114913"
        )
        if q_open:
            out["flux_fraction_of_open_min"] = min(flux) / q_open
            out["flux_fraction_of_open_final"] = flux[-1] / q_open
            out["occlusion_fraction_final"] = 1.0 - flux[-1] / q_open
        rel = [r["released_max"] for r in series if r["released_max"] is not None]
        out["released_max_over_run"] = max(rel) if rel else None
        # The tripwire fires on the PRIMARY child only; in th_break a nonzero
        # released is the intended outcome, so it is reported either way and
        # judged against the child's own expectation rather than here.
        out["released_became_nonzero"] = bool(rel and max(rel) > 1e-9)
    return out


def read_vectors(path: Path) -> list[tuple[float, float, float]]:
    t = path.read_text(errors="replace")
    m = re.search(
        r"(?ms)internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;", t
    )
    if m:
        return [tuple(float(x) for x in r.split())
                for r in re.findall(r"\(([^()]+)\)", m.group(2))]
    u = re.search(r"internalField\s+uniform\s+\(([^)]+)\)\s*;", t)
    if u:
        return [tuple(float(x) for x in u.group(1).split())]
    raise ValueError(f"{path}: cannot parse vectors")


def time_dirs(case: Path) -> list[tuple[float, Path]]:
    out = []
    for p in case.iterdir():
        if p.is_dir():
            try:
                out.append((float(p.name), p))
            except ValueError:
                pass
    return sorted(out)


# --- the phase field, in the c convention, from whatever the run wrote -------
#
# S6c-2 deleted `c` from the solver. Every geometry and bounds measurement in
# this file was written against it: `c < 0` is clot, `c < -0.5` is the edge
# contour, `(1-c)/2` is the material integral, and C_BOUND_LIMIT was fitted to
# a measured cUnder of 1.7788. None of those thresholds is rebased here. The
# field is RECONSTRUCTED instead, and the map is exact:
#
#     ind = (thetaC + thetaH)/theta_ref,   q = 1 - ind,   c = 1 - 2*ind = 2q - 1
#
# Measured on run 20260920-060033 / em45_s6c1_normalcontent, which wrote all
# four fields: at t = 0 the reconstruction matches the written `c` to
# 6.7e-16, and at t = 4e-04 it differs by 0.031667 -- the c-vs-theta drift
# that S6c-2 removes by making theta the only truth, and exactly twice the
# 0.0158337 that PHASE_CLOT_CONTENT reported as equivMax over the same run.
# The drift is the finding; it is not an error in this map.
#
# The convention is chosen ONCE PER CASE, from the last written time, and then
# used for every time in that case -- including 0/. Choosing per time would
# read 0/c, which create_cases.py still writes, against a reconstructed final
# and silently compare two conventions whose difference is the thing under
# test.
_C_CONVENTION: dict[Path, str] = {}


def _theta_ref(case: Path) -> float | None:
    props = case / "constant" / "phaseClotProperties"
    if not props.exists():
        return None
    m = re.search(r"^\s*skeletonVolumeFraction\s+([-\d.eE+]+)\s*;",
                  props.read_text(errors="replace"), re.M)
    if not m:
        return None
    v = float(m.group(1))
    return v if v > 0.0 else None


def _phi_s_max(case: Path) -> float | None:
    """theta_max -- close packing -- from the case's own dictionary.

    Read the same way as _theta_ref so the two cannot come from different
    places. Returns None when the entry is absent, which is the signal to
    fall back to the control's literal bound.
    """
    props = case / "constant" / "phaseClotProperties"
    if not props.exists():
        return None
    m = re.search(r"^\s*phiSMax\s+([-\d.eE+]+)\s*;",
                  props.read_text(errors="replace"), re.M)
    if not m:
        return None
    v = float(m.group(1))
    return v if v > 0.0 else None


# ALIASES, AND WHY THEY EXIST RATHER THAN A RENAME.
#
# assess_child() binds a LOCAL `_theta_ref` from the PHASE_CLOT_MASS record.
# Python decides local-vs-global per function, not per statement, so any call
# to the module-level `_theta_ref` anywhere in that function raises
# UnboundLocalError -- including on lines that run long before the
# assignment. Renaming the local would touch code this change has no business
# in, so the readers get unshadowed aliases instead.
theta_ref_of_case = _theta_ref
phi_s_max_of_case = _phi_s_max


def phase_bounds_for(case: Path, controls: dict) -> tuple[float, float, str]:
    """The admissible phase interval, DERIVED from the model where possible.

    WHY THIS IS NOT THE CONTROL'S LITERAL [-1.01, 1.01] ANY MORE.

    `c = 1 - 2*theta/theta_ref` is an affine restatement of the solid volume
    fraction, not a phase-field order parameter. The bound c >= -1.01 says
    `theta <= 1.005*theta_ref` -- that the clot may densify by at most 0.5%
    above nominal. Nothing in the model says that. The constitutive law
    declares close packing at `phiSMax` (0.6 against a nominal 0.45) and the
    network pressure is built to permit compaction up to it, so the control
    and the constitutive law disagreed by construction: every run that
    compacted at all was graded INADMISSIBLE for obeying its own material
    model. Run 20260921-072935 crossed the control bound at t = 6.735e-04 s
    with theta = 0.4523, which is 89% of the way to nothing in particular.

    THE ADMISSIBLE SET IS THE MIXTURE'S: 0 <= theta <= theta_max. In c,

        c_lo = 1 - 2*theta_max/theta_ref,    c_hi = 1 (theta = 0)

    THIS BOUND CAN STILL FAIL, which is the point and the reason it is not a
    relaxation. theta is NOT clipped anywhere in the solver -- solveContent
    Components says so in as many words -- so a run whose network pressure
    fails to hold it walks straight through theta_max and this fires. The
    companion instrument is pnCapCells: the pressure law's own 0.99*phiSMax
    clamp engaging means the clamp, not the force balance, did the bounding.

    The 1.005 margins are the control's own (its 1.01 on a c-bound is 0.5% in
    theta); they are carried over rather than re-invented so the change is
    the BASIS of the bound and not its tightness.
    """
    lo, hi = controls["bounds"]["phase"]
    tr, tmax = _theta_ref(case), _phi_s_max(case)
    if tr is None or tmax is None or tmax <= tr:
        return lo, hi, "control-literal"
    derived_lo = 1.0 - 2.0 * (1.005 * tmax) / tr
    # Never TIGHTEN what the control allowed: this replaces a legacy
    # phase-field bound, it does not start failing runs the control passed.
    return min(lo, derived_lo), hi, f"theta<=({tmax:g}) over theta_ref={tr:g}"


def c_convention(case: Path) -> str:
    """'written' if the run wrote `c` at its last output time, else 'theta'."""
    key = case.resolve()
    if key not in _C_CONVENTION:
        dirs = [(t, d) for t, d in time_dirs(case) if t > 0.0]
        _C_CONVENTION[key] = (
            "written"
            if dirs and find_field(dirs[-1][1], "c") is not None
            else "theta"
        )
    return _C_CONVENTION[key]


def phase_c(time_dir: Path) -> list[float] | None:
    """The phase field at one written time, in the c convention.

    -1 is clot, +1 is blood, whichever fields the solver wrote.
    """
    case = time_dir.parent
    if c_convention(case) == "written":
        pc = find_field(time_dir, "c")
        if pc is None:
            return None
        try:
            return read_scalars(pc)
        except ValueError:
            return None

    pq = find_field(time_dir, "q")
    if pq is not None:
        try:
            q = read_scalars(pq)
        except ValueError:
            return None
        return [2.0 * v - 1.0 for v in q]

    # 0/ carries thetaC and thetaH but no q -- create_cases.py writes the
    # initial condition and the solver has not written a step yet.
    ptc, pth = find_field(time_dir, "thetaC"), find_field(time_dir, "thetaH")
    tr = _theta_ref(case)
    if ptc is None or pth is None or tr is None:
        return None
    try:
        tc, th = read_scalars(ptc), read_scalars(pth)
    except ValueError:
        return None
    if len(tc) != len(th):
        # A uniform field parses to one value; expand it against the other.
        if len(th) == 1:
            th = th * len(tc)
        elif len(tc) == 1:
            tc = tc * len(th)
        else:
            return None
    return [1.0 - 2.0 * (tc[i] + th[i]) / tr for i in range(len(tc))]

def clot_geometry(time_dir: Path, x: list[float],
                  y: list[float] | None = None) -> dict | None:
    """Centroid and cell count of the clot (c < 0) at one written time.

    The uniform mesh makes an unweighted centroid the same as a
    volume-weighted one, which is why no Vc is read here.
    """
    c = phase_c(time_dir)
    if c is None:
        return None
    if len(c) != len(x) or not finite(c):
        return None
    inside = [i for i, v in enumerate(c) if v < 0.0]
    # c_min AND the phase integral, not just the cell count.
    #
    # Run 20260811-195034 dissolved the clot in place: c_min went from -0.997
    # to -0.145 while the cell count stayed at ~721 and the reported
    # volume_change_fraction stayed at 0.0. Counting cells below zero cannot
    # see a field draining towards zero, so the report called a dissolved clot
    # conserved. The integral of (1-c)/2 is the amount of clot MATERIAL, and it
    # falls when the field smears even though no cell crosses the contour.
    material = sum((1.0 - v) / 2.0 for v in c)
    # The LEADING and TRAILING edges, tracked separately.
    #
    # A centroid cannot tell translation from erosion. Run 20260812-041322
    # reported moved_downstream true at 4.93 cells while the trailing edge ran
    # 17 cells and the leading edge managed 9 -- the body was being eaten from
    # behind, and every metric in this file agreed it had moved. The -0.5
    # contour is used rather than 0 because it sits in the steep part of the
    # interface and is far less sensitive to smearing.
    core = [i for i, v in enumerate(c) if v < -0.5]
    if not inside:
        return {"cells": 0, "centroid_x_m": None, "c_min": min(c),
                "material": material, "leading_x_m": None,
                "trailing_x_m": None, "body_length_m": None,
                "top_y_m": None, "core_cells": 0}
    return {
        "cells": len(inside),
        "centroid_x_m": sum(x[i] for i in inside) / len(inside),
        "c_min": min(c),
        "material": material,
        "leading_x_m": max(x[i] for i in core) if core else None,
        "trailing_x_m": min(x[i] for i in core) if core else None,
        "body_length_m": (
            max(x[i] for i in core) - min(x[i] for i in core) if core else None
        ),
        # HOW TALL IS IT. Run 20260812-200349 kept its length pinned at exactly
        # 90 um (the injury length) while its HEIGHT went 20 -> 34 um and the
        # c < -0.5 core grew from 336 to 557 cells. Every length-based metric
        # in this file said the body was translating perfectly. It was being
        # advected upward into a wall it cannot pass through, and stretching
        # instead. Nothing here could see it, because nothing measured y.
        "top_y_m": max(y[i] for i in inside) if y is not None else None,
        "core_cells": len(core),
    }


def tear_measurement(case: Path, spec: dict) -> dict:
    """WHERE did the clot break, and did the wall hold while it did?

    Dr Li's 2026-08-13 objection was that the whole clot slides away, which is
    not an observed embolisation mode.  Source 33 agrees: their core keeps its
    shape and only the shell is torn off.  So "it broke" is not the
    measurement -- "it broke THROUGH THE BODY, above the wall, while the bond
    was intact" is.

    Located from the FINAL Dg field rather than from the c field.  A cell
    whose c flipped from negative to positive may have been erased by damage
    OR simply advected out of the way, and those are opposite outcomes.  Dg is
    not reset when the material is released, so `Dg > DgDetach` marks exactly
    the cells the fracture erased and nothing else.
    """
    out = {
        "available": False,
        "reason": None,
        "tore": None,
        "tear_cells": None,
        "tear_y_min_m": None,
        "tear_y_mean_m": None,
        "tear_y_max_m": None,
        "tear_height_fraction": None,
        "tear_at_wall": None,
        "tear_through_body": None,
        "first_tear_time_s": None,
        "Dg_max": None,
        "tear_mode": None,
        "intact_cells_above_damage": None,
        "material_remaining_fraction": None,
        # --- core/shell only; None everywhere else -------------------------
        "core_shell": False,
        "core_cells": None,
        "shell_cells": None,
        "core_damaged_cells": None,
        "core_survived": None,
        "first_tear_depth_cells": None,
        "first_tear_shallowest_depth_cells": None,
        "tear_started_at_the_surface": None,
        "growth_area_fraction": None,
        "grew": None,
    }
    if spec["series"] not in ("cohesive_tear", "source33_replication"):
        out["reason"] = "not a tear child"
        return out

    dirs = [(t_, d) for t_, d in time_dirs(case) if t_ > 0.0]
    if not dirs:
        out["reason"] = "no written output times beyond 0"
        return out

    zero = case / "0"
    ccy = find_field(zero, "Ccy")
    if ccy is None:
        # Without cell centres the y of the tear cannot be reported, and a
        # tear whose HEIGHT is unknown does not answer the question that was
        # asked. Refuse rather than return a bare cell count that reads like
        # a success.
        out["reason"] = (
            "0/Ccy is missing, so the tear height cannot be measured; "
            "writeCellCentres must run before the solver"
        )
        return out
    y = read_scalars(ccy)

    detach = float(spec.get("DgDetach", 0.95))
    first_time = None
    first_dir = None
    for t_, d in dirs:
        f = find_field(d, "Dg")
        if f is None:
            continue
        if max(read_scalars(f), default=0.0) > detach:
            first_time = t_
            first_dir = d
            break

    final = find_field(dirs[-1][1], "Dg")
    if final is None:
        out["reason"] = "no Dg field written"
        return out
    dg = read_scalars(final)
    if len(dg) != len(y):
        out["reason"] = f"Dg has {len(dg)} values against {len(y)} cell centres"
        return out

    torn = [y[i] for i in range(len(dg)) if dg[i] > detach]
    h = spec["geometry"]["h_m"]
    clot_height = spec["clot"]["height_m"]

    # How much material is left, against the initial condition. A body that
    # loses most of its phase integral has DISSOLVED, whatever the damage
    # field says about where.
    c0, cN = phase_c(zero), phase_c(dirs[-1][1])
    if c0 is not None and cN is not None:
        m0 = sum((1.0 - v) / 2.0 for v in c0)
        mN = sum((1.0 - v) / 2.0 for v in cN)
        out["material_remaining_fraction"] = (mN / m0) if m0 else None

    out.update({
        "available": True,
        "tore": bool(torn),
        "tear_cells": len(torn),
        "first_tear_time_s": first_time,
        "Dg_max": max(dg),
    })
    if torn:
        ymax = max(torn)
        # THE QUESTION THAT SEPARATES A TEAR FROM EROSION, and the one the
        # first version of this function did not ask.
        #
        # Run 20260814-055141 reported `tear_through_body: true` for a band of
        # 15 damaged cells sitting at 99% of the clot height -- i.e. the crest.
        # That is not a break through the body, it is the outer surface being
        # eaten. The two are told apart by what is ABOVE the damage: a real
        # through-tear leaves intact clot standing on the far side of the
        # severed band, and surface ablation leaves nothing there at all.
        # Measured on that run: ZERO intact cells above the damage, in every
        # child and at every time.
        above = 0
        if cN is not None and len(cN) == len(y):
            above = sum(1 for i in range(len(cN))
                        if cN[i] < 0.0 and y[i] > ymax)
        at_wall = bool(ymax < 2.0 * h)
        through = bool(above > 0 and not at_wall)
        out.update({
            "tear_y_min_m": min(torn),
            "tear_y_mean_m": sum(torn) / len(torn),
            "tear_y_max_m": ymax,
            "tear_height_fraction": (sum(torn) / len(torn)) / clot_height,
            "intact_cells_above_damage": above,
            # A tear confined to the first two cell rows is the bond failing
            # by another name, and would reproduce exactly the whole-body
            # release Dr Li rejected.
            "tear_at_wall": at_wall,
            "tear_through_body": through,
            "tear_mode": (
                "at_wall" if at_wall
                else "through_body" if through
                else "surface_ablation"
            ),
        })
    else:
        out["tear_mode"] = "none"

    if spec.get("core_shell"):
        out.update(core_shell_measurement(case, spec, dirs, y, detach,
                                          first_dir))
    return out


def core_shell_measurement(case: Path, spec: dict,
                           dirs: list[tuple[float, Path]],
                           y: list[float], detach: float,
                           first_dir: Path | None) -> dict:
    """Did the core survive, and did the tear start at DEPTH or at the surface?

    These are the two questions the core/shell child exists to answer, and
    neither is answerable from the numbers `tear_measurement` already reports.

    THE CORE. Its Gc is the physiological 5.547 J/m2 against the shell's
    calibrated 1e-9, so it cannot be damaged by anything this model reaches.
    That makes `core_survived` a check on the WIRING rather than on the
    physics: if it comes back false, the `shell` field or the Gc blend is
    wrong, and every other number here is describing a different experiment.

    THE DEPTH. This is the measurement that separates the tear Dr Li asked
    for from the surface erosion of run 20260814-055141 -- which Alireza
    called, exactly right, "like eating it up". Erosion begins on the outer
    surface, where a damaged cell is one cell from blood. A tear at the
    core/shell boundary begins six cells in. So: at the FIRST time any cell
    passes DgDetach, how far is the deepest of them from the nearest fluid
    cell? One cell means the surface; several means the interior.

    Distance is measured to the fluid AT THAT TIME, not to the initial
    outline, because a clot that has grown has a different surface from the
    one it started with.
    """
    out = {"core_shell": True}
    shell_path = find_field(case / "0", "shell")
    if shell_path is None:
        out["reason_core_shell"] = (
            "0/shell is missing: the child declares core_shell but the "
            "initialiser wrote no material indicator, so the solver ran ONE "
            "material and the core does not exist"
        )
        return out
    shell = read_scalars(shell_path)
    if len(shell) != len(y):
        out["reason_core_shell"] = (
            f"0/shell has {len(shell)} values against {len(y)} cell centres"
        )
        return out

    ccx = find_field(case / "0", "Ccx")
    x = read_scalars(ccx) if ccx is not None else None
    h = spec["geometry"]["h_m"]

    c_initial = phase_c(case / "0")
    c_final = phase_c(dirs[-1][1])

    if c_initial and c_final and len(c_initial) == len(shell):
        out["core_cells"] = sum(
            1 for i in range(len(shell)) if shell[i] < 0.5 and c_initial[i] < 0
        )
        out["shell_cells"] = sum(
            1 for i in range(len(shell)) if shell[i] >= 0.5 and c_initial[i] < 0
        )
        # GROWTH, measured as clot AREA rather than height. A dome that gains
        # 1.9 um of radius gains about 45% of its area but only 15% of its
        # height, so area is far the more sensitive of the two -- and it is
        # what a viewer actually sees change in ParaView.
        n0 = sum(1 for v in c_initial if v < 0)
        nN = sum(1 for v in c_final if v < 0)
        if n0:
            out["growth_area_fraction"] = nN / n0 - 1.0
            # Growth competes with the tear, which ERASES material. A clot
            # that tears hard can finish smaller than it started while
            # having grown throughout, so this is reported and not asserted.
            out["grew"] = bool(nN > n0)

    # SOURCE 33'S CENTRAL RESULT, measured rather than described: across every
    # shear rate they run, the CORE RETAINS ITS SHAPE while the SHELL is
    # pushed downstream into an elongated filament (their figure 5e-h). So the
    # measurement is not "did it tear" -- it is how the two regions moved
    # relative to each other.
    #
    # The label is advected with the material, so the shell field at the final
    # time is where the shell material has GOT to, not where it started. Both
    # ends are read from their own time.
    shell_final_path = find_field(dirs[-1][1], "shell")
    shell_final = (
        read_scalars(shell_final_path) if shell_final_path is not None else shell
    )
    if x is not None and c_initial and c_final and len(shell_final) == len(x):
        def region(cf, sf, core: bool) -> dict | None:
            idx = [
                i for i in range(len(cf))
                if cf[i] < 0.0 and ((sf[i] < 0.5) == core)
            ]
            if not idx:
                return None
            return {
                "cells": len(idx),
                "x_centroid": sum(x[i] for i in idx) / len(idx),
                "top": max(y[i] for i in idx),
                "length": max(x[i] for i in idx) - min(x[i] for i in idx),
                "downstream_edge": max(x[i] for i in idx),
            }

        for name, core in (("core", True), ("shell", False)):
            a = region(c_initial, shell, core)
            b = region(c_final, shell_final, core)
            if a is None or b is None:
                continue
            out[f"{name}_cells_initial"] = a["cells"]
            out[f"{name}_cells_final"] = b["cells"]
            out[f"{name}_centroid_shift_m"] = b["x_centroid"] - a["x_centroid"]
            out[f"{name}_downstream_edge_shift_m"] = (
                b["downstream_edge"] - a["downstream_edge"]
            )
            out[f"{name}_height_change_fraction"] = (
                b["top"] / a["top"] - 1.0 if a["top"] else None
            )
            out[f"{name}_length_change_fraction"] = (
                b["length"] / a["length"] - 1.0 if a["length"] else None
            )
        # THE SIGNATURE. Xu's shell elongates downstream and their core does
        # not. Expressed as a ratio so it is scale-free and so a run in which
        # NEITHER moved cannot pass by both being near zero.
        cs = out.get("core_centroid_shift_m")
        ss = out.get("shell_centroid_shift_m")
        if cs is not None and ss is not None:
            out["shell_over_core_shift"] = (
                ss / cs if abs(cs) > 1e-12 else (None if abs(ss) < 1e-12 else 1e9)
            )
            # TWICE AS FAR, AND AT LEAST ONE CELL. Both halves are needed.
            # Without the factor, a whole clot translating rigidly passes on
            # a floating-point difference in the last digit -- which is
            # exactly the whole-body sliding Dr Li rejected, arriving dressed
            # as Xu's mechanism. Without the one-cell floor, a run in which
            # nothing moved at all passes because the shell's numerical noise
            # happened to be twice the core's.
            out["shell_moved_more_than_core"] = bool(
                abs(ss) >= 2.0 * abs(cs) and abs(ss) >= h
            )
            out["shell_over_core_criterion"] = (
                "shell centroid shift at least 2x the core's AND at least one "
                f"cell ({h:.3g} m)"
            )

    # DEFORMATION BY REGION. Strain goes as 1/G, so the 4 Pa shell should
    # reach about ten times the 40 Pa core. If it does not, the material
    # fields are not doing what the dictionary says.
    us_path = find_field(dirs[-1][1], "Us")
    if us_path is not None and len(shell_final) == len(y):
        us = read_vectors(us_path)
        if len(us) == len(shell_final) and c_final:
            def peak(core: bool) -> float | None:
                vals = [
                    math.sqrt(sum(v * v for v in us[i]))
                    for i in range(len(us))
                    if c_final[i] < 0.0 and ((shell_final[i] < 0.5) == core)
                ]
                return max(vals) if vals else None
            out["Us_core_max_m"] = peak(True)
            out["Us_shell_max_m"] = peak(False)
            if out["Us_core_max_m"] and out["Us_shell_max_m"]:
                out["Us_shell_over_core"] = (
                    out["Us_shell_max_m"] / out["Us_core_max_m"]
                )

    dg_final_path = find_field(dirs[-1][1], "Dg")
    if dg_final_path is not None:
        dgf = read_scalars(dg_final_path)
        if len(dgf) == len(shell):
            damaged_core = sum(
                1 for i in range(len(shell))
                if shell[i] < 0.5 and dgf[i] > detach
            )
            out["core_damaged_cells"] = damaged_core
            out["core_survived"] = bool(damaged_core == 0)

    # Depth of the first damage below the surface, in cells.
    if first_dir is not None and x is not None:
        dgp = find_field(first_dir, "Dg")
        c1 = phase_c(first_dir)
        if dgp is not None and c1 is not None:
            dg1 = read_scalars(dgp)
            if len(dg1) == len(x) == len(c1):
                torn_idx = [i for i in range(len(dg1)) if dg1[i] > detach]
                fluid = [(x[i], y[i]) for i in range(len(c1)) if c1[i] > 0.0]
                if torn_idx and fluid:
                    depths = []
                    for i in torn_idx:
                        xi, yi = x[i], y[i]
                        d2 = min((xi - fx) ** 2 + (yi - fy) ** 2
                                 for fx, fy in fluid)
                        depths.append(math.sqrt(d2) / h)
                    out["first_tear_depth_cells"] = max(depths)
                    out["first_tear_shallowest_depth_cells"] = min(depths)
                    # One cell from fluid IS the surface. Anything the
                    # model erases there is being eaten, not torn.
                    out["tear_started_at_the_surface"] = bool(
                        max(depths) < 2.0
                    )
    return out


def transport_measurement(case: Path, spec: dict) -> dict:
    """Did the released clot actually MOVE, and was it conserved doing so?

    The whole point of the advection term. Run 20260810-212211 released 59 of
    60 bonded faces and the clot stayed exactly where it was, because the
    phase equation had no advective term -- so "detached" has to be measured
    as displacement, not inferred from the bond state.
    """
    out = {
        "available": False,
        "reason": None,
        "exit_code": None,
        "centroid_start_x_m": None,
        "centroid_end_x_m": None,
        "displacement_m": None,
        "clot_cells_start": None,
        "clot_cells_end": None,
        "volume_change_fraction": None,
        "moved_downstream": None,
    }
    # ONE CONTINUOUS RUN. There is no transport stage to look for any more:
    # the timestep is adaptive and the clot peels, deforms and travels in the
    # same time series. Displacement is therefore measured from the FIRST
    # written time to the last, not across a restart boundary.
    if not s21_spec(spec, "advect_released"):
        out["reason"] = "advection is off for this child"
        return out

    exit_path = case / "run-exit-code.txt"
    if not exit_path.is_file():
        out["reason"] = "the run produced no exit code"
        return out
    out["exit_code"] = int(exit_path.read_text().strip())
    if out["exit_code"] != 0:
        out["reason"] = f"the run exited {out['exit_code']}"
        return out

    centres = find_field(case / "0", "Ccx")
    if centres is None:
        out["reason"] = "no 0/Ccx; cannot locate the clot"
        return out
    x = read_scalars(centres)
    # Ccy so the body's HEIGHT can be measured, not just its length.
    ycent = find_field(case / "0", "Ccy")
    y = read_scalars(ycent) if ycent is not None else None

    # Only times > 0 count. A child that wrote nothing but its initial
    # condition would otherwise be scored against itself and look perfectly
    # conserved, which is how an empty run passed once already.
    dirs = [(t_, d) for t_, d in time_dirs(case) if t_ > 0.0]
    if len(dirs) < 2:
        out["reason"] = "fewer than two written times beyond 0"
        return out

    first = clot_geometry(dirs[0][1], x, y)

    # MEASURE THE TRANSLATION OVER THE WINDOW THE BODY IS ACTUALLY INSIDE.
    #
    # Run 20260813-013923 translated coherently from x = 75-164 to 156-238 um
    # with a stable core, then spent the rest of the run leaving through the
    # outlet. Measuring first-to-LAST spans that exit, so the report said
    # core_growth -100%, length -100%, c_min -0.997 -> -0.004: a textbook
    # dissolution signature produced by a clot that simply left. The exit is
    # not a defect and must not be scored as one.
    #
    # The endpoint is therefore the last written time at which the core's
    # leading edge is still clear of the outlet. The exit itself is reported
    # separately below.
    # A core that VANISHES is not the same as a core that LEAVES, and only the
    # second may shorten the window. Truncating on "no core" would hide a
    # dissolution behind the exit logic -- the smearing of run 20260811-195034
    # would have been scored against a window that ended before it happened.
    # So the window is only cut once the leading edge has actually reached the
    # outlet; a body that never gets there is measured over the full series.
    domain_x_m = spec["geometry"]["nx"] * spec["geometry"]["h_m"]
    exit_margin_m = 3.0 * spec["geometry"]["h_m"]
    last, last_t, exiting_from_s = None, None, None
    reached_outlet = False
    for t_, d in dirs:
        g = clot_geometry(d, x, y)
        if g is None:
            continue
        if g["leading_x_m"] is not None \
                and g["leading_x_m"] >= domain_x_m - exit_margin_m:
            reached_outlet = True
            if exiting_from_s is None:
                exiting_from_s = t_
        if not reached_outlet:
            last, last_t = g, t_
    if last is None or not reached_outlet:
        last, last_t = clot_geometry(dirs[-1][1], x, y), dirs[-1][0]
    out["reached_outlet"] = reached_outlet

    final = clot_geometry(dirs[-1][1], x, y)
    out["exit_started_s"] = exiting_from_s
    out["final_time_s"] = dirs[-1][0]
    if final is not None and first is not None and first["material"]:
        out["material_final"] = final["material"]
        out["residue_fraction"] = final["material"] / first["material"]
        out["clot_cells_final"] = final["cells"]
        # A body that left carries its material out. What stays is residue,
        # and it is the number that says whether the channel is clear.
        out["stranded_residue"] = bool(
            final["cells"] == 0 and out["residue_fraction"] >= 0.1
        )
        out["left_cleanly"] = bool(
            final["cells"] == 0 and out["residue_fraction"] < 0.1
        )
    if first is None or last is None:
        out["reason"] = "could not read c at the transport endpoints"
        return out
    if not first["cells"] or not last["cells"]:
        # "No cells below c = 0" is NOT "no clot material", and reporting it
        # that way is how run 20260812-200349 went silent about the thing that
        # was actually wrong. That run released all 60 faces, the body reached
        # the outlet and left at t = 0.1359 -- and left 83% of its material
        # behind as a frozen wedge that blocked the channel for the remaining
        # 0.21 s. Every transport field came back None and the reason string
        # said there was no material, while `material` sat at 689.9.
        #
        # The phase INTEGRAL is still meaningful with no core, so it is
        # reported here. A body that genuinely leaves carries its material out
        # with it; a body that sheds its tail does not.
        residue = (
            last["material"] / first["material"] if first["material"] else None
        )
        out.update({
            "clot_cells_start": first["cells"],
            "clot_cells_end": last["cells"],
            "material_start": first["material"],
            "material_end": last["material"],
            "residue_fraction": residue,
            "c_min_start": first["c_min"],
            "c_min_end": last["c_min"],
            # The body left AND took its material with it: a clean exit, and
            # the only version of "no core at the end" that is a success.
            "left_cleanly": bool(
                last["cells"] == 0 and residue is not None and residue < 0.1
            ),
            # The body left but the material did not. This is the failure the
            # blend-to-fluid-flux advection fix exists to prevent, and it must
            # be loud rather than absent.
            "stranded_residue": bool(
                last["cells"] == 0 and residue is not None and residue >= 0.1
            ),
        })
        if out["stranded_residue"]:
            out["reason"] = (
                f"the clot core left the domain but {residue * 100:.0f}% of "
                f"its material stayed behind as a stationary residue"
            )
        elif out["left_cleanly"]:
            out["reason"] = "the clot left the domain, carrying its material out"
        else:
            out["reason"] = "no clot core at one of the transport endpoints"
        return out

    displacement = last["centroid_x_m"] - first["centroid_x_m"]
    h = spec["geometry"]["h_m"]
    out.update({
        "available": True,
        "transport_start_time_s": dirs[0][0],
        "transport_end_time_s": last_t,
        "centroid_start_x_m": first["centroid_x_m"],
        "centroid_end_x_m": last["centroid_x_m"],
        "displacement_m": displacement,
        "displacement_cells": displacement / spec["geometry"]["h_m"],
        "clot_cells_start": first["cells"],
        "clot_cells_end": last["cells"],
        "volume_change_fraction": (
            (last["cells"] - first["cells"]) / first["cells"]
        ),
        "moved_downstream": bool(displacement > spec["geometry"]["h_m"]),
        # THE DISSOLUTION CHECK. A clot that smears away is not a clot that
        # moved, and the cell count cannot tell the difference.
        "c_min_start": first["c_min"],
        "c_min_end": last["c_min"],
        "material_change_fraction": (
            (last["material"] - first["material"]) / first["material"]
            if first["material"] else None
        ),
        # c_min drifting towards zero is the field draining, whatever the
        # centroid does. -0.9 is generous: the peel itself barely moves it.
        "dissolving": bool(last["c_min"] > -0.9),

        # DID THE BODY TRANSLATE, OR WAS IT ERODED? Both edges, separately.
        "leading_start_x_m": first["leading_x_m"],
        "leading_end_x_m": last["leading_x_m"],
        "trailing_start_x_m": first["trailing_x_m"],
        "trailing_end_x_m": last["trailing_x_m"],
        "leading_travel_cells": (
            (last["leading_x_m"] - first["leading_x_m"]) / h
            if None not in (last["leading_x_m"], first["leading_x_m"]) else None
        ),
        "trailing_travel_cells": (
            (last["trailing_x_m"] - first["trailing_x_m"]) / h
            if None not in (last["trailing_x_m"], first["trailing_x_m"]) else None
        ),
        "body_length_start_m": first["body_length_m"],
        "body_length_end_m": last["body_length_m"],

        # HEIGHT AND CORE SIZE. A body advected into a wall it cannot pass
        # through does not translate, it stretches -- and every length-based
        # metric above reports that as a clean translation. Run
        # 20260812-200349: length pinned at 90 um, height 20 -> 34 um, core
        # 336 -> 557 cells, driven by a spurious Uy that reached 17% of Ux.
        # The core GROWING is the tell that separates this from smearing,
        # which shrinks a core.
        "top_y_start_m": first["top_y_m"],
        "top_y_end_m": last["top_y_m"],
        "height_change_fraction": (
            (last["top_y_m"] - first["top_y_m"]) / first["top_y_m"]
            if None not in (last["top_y_m"], first["top_y_m"])
            and first["top_y_m"] else None
        ),
        "core_cells_start": first["core_cells"],
        "core_cells_end": last["core_cells"],
        "core_growth_fraction": (
            (last["core_cells"] - first["core_cells"]) / first["core_cells"]
            if first["core_cells"] else None
        ),
        # BOTH ends must be guarded. A clot that smears above the -0.5 contour
        # has no core left at all, so `last` is None while `first` is a number
        # -- which is a dissolved clot, not a missing measurement, and
        # subtracting them raised a TypeError that killed the whole assessor.
        "length_change_fraction": (
            (last["body_length_m"] - first["body_length_m"]) / first["body_length_m"]
            if first["body_length_m"] and last["body_length_m"] is not None
            else (-1.0 if first["body_length_m"] else None)
        ),
        # Translation means the two edges move TOGETHER. Run 20260812-041322
        # had them 8 cells apart and called it transport.
        # A translation preserves BOTH dimensions. Checking length alone
        # passed run 20260812-200349, whose length was pinned at exactly the
        # 90 um injury width while its height grew 70%.
        "translating": (
            None if None in (last["body_length_m"], first["body_length_m"],
                             last["top_y_m"], first["top_y_m"])
            or not first["body_length_m"] or not first["top_y_m"]
            else (
                abs(last["body_length_m"] - first["body_length_m"])
                / first["body_length_m"] < 0.15
                and abs(last["top_y_m"] - first["top_y_m"])
                / first["top_y_m"] < 0.15
            )
        ),
        "note": (
            "moved_downstream requires more than one cell of centroid travel, "
            "so interface diffusion cannot be mistaken for transport. "
            "`dissolving` is the separate failure: run 20260811-195034 advected "
            "at Courant 16.2 during the peel and smeared c_min from -0.997 to "
            "-0.145 while the cell count and volume_change_fraction both "
            "reported no change at all."
        ),
    })
    return out


def finite(values) -> bool:
    return all(math.isfinite(v) for v in values)


def assess_child(case: Path, spec: dict, controls: dict) -> dict:
    log = case / "log.phaseClotAxisymmetricFluid"
    text = log.read_text(errors="replace") if log.is_file() else ""
    exit_path = case / "run-exit-code.txt"
    exit_code = int(exit_path.read_text().strip()) if exit_path.is_file() else None
    wall = case / "log.wallTime"
    wall_seconds = None
    # CONCURRENCY is read alongside it. Allrun can run N children at once
    # (AM2020_CONCURRENCY), and then WALL_SECONDS is wall time with N children
    # sharing the cores -- NOT the cost of this child. Reporting it bare would
    # make a concurrent run look like the solver got slower, and would corrupt
    # any cost model built from it: the 0.418 s/step figure in the notebook
    # came from exactly this field on a SERIAL run.
    concurrency = None
    if wall.is_file():
        wall_text = wall.read_text()
        m = re.search(r"WALL_SECONDS=(\d+)", wall_text)
        wall_seconds = int(m.group(1)) if m else None
        mc = re.search(r"CONCURRENCY=(\d+)", wall_text)
        concurrency = int(mc.group(1)) if mc else None

    times = [float(v) for v in TIME_LINE.findall(text)]
    completed = bool(
        exit_code == 0 and any(k in text for k in COMPLETED) and times
        and math.isclose(max(times), spec["end_time_s"], rel_tol=1e-6)
    )

    # THE STEP-1 WATCHER (design 2026-10-02-case-free-shell-throat, rev. 3,
    # S7.3b/04 #2). watcher-stop.txt is written by clot_free_watcher.py, NOT
    # by Allrun, so it survives Allrun:155's run-exit-code.txt overwrite and
    # is read INDEPENDENTLY of exit_code. Its presence means the solver was
    # told to stop (stopAtFile, action writeNow) because V1 failed at step 1
    # -- never a clean finish, whatever exit_code and the log's own "End"
    # line say.
    watcher_path = case / "watcher-stop.txt"
    watcher_stop = watcher_path.read_text() if watcher_path.is_file() else None
    if watcher_stop is not None:
        completed = False

    # THE 5 ms WATCHER (pipeline pass 2a; runner run_main_case_colab.py watch_child):
    # runs/<child>/watcher-verdict.json. Any verdict but PASS or NOT_REACHED means the
    # runner asked the solver to stop at the 5 ms frame (stopAtFile, writeNow), and the
    # solver then exits 0 at ~5.2 ms: never a complete child, whatever the exit code or
    # the log's own "End" line say. An unreadable record is not a pass either.
    wv_path = case / "watcher-verdict.json"
    watcher_verdict = None
    if wv_path.is_file():
        try:
            watcher_verdict = json.loads(wv_path.read_text())
            if not isinstance(watcher_verdict, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as error:
            watcher_verdict = {"verdict": "UNREADABLE", "reason": repr(error)}
    stopped_by_watcher = (watcher_verdict is not None and watcher_verdict.get("verdict")
                          not in ("PASS", "NOT_REACHED"))
    if stopped_by_watcher:
        completed = False

    sweeps = [
        {"iterations": int(m[1]), "residual": float(m[3]),
         "converged": m[6].strip().lower() == "true"}
        for m in FSI_LINE.findall(text)
    ]
    settled = sweeps[int(len(sweeps) * 0.2):]
    skeleton = [(int(i), float(r)) for i, r in SKELETON_LINE.findall(text)]
    cap = 20
    at_cap = sum(1 for i, _ in skeleton if i >= cap)

    debonds = parse_debond(text)
    volumes = [(float(m[0]), float(m[1])) for m in DIAGNOSTIC_LINE.findall(text)]
    body_velocity = parse_body_velocity(text)
    driver = parse_damage_driver(text)
    permeability = parse_permeability(text)

    result = {
        "child_id": spec["child_id"],
        "series": spec["series"],
        "wall_shear_rate_1_s": spec["wall_shear_rate_1_s"],
        "mean_velocity_m_s": spec["mean_velocity_m_s"],
        "G_pa": spec["G_pa"],
        "k0p_m2": spec["k0p_m2"],
        # Per-child since the cohesive-tear series sweeps it. Everywhere else
        # it is the physiological value from material_fixed.
        "Gc_j_m2": spec["Gc_j_m2"],
        # None outside source33_replication. Carried through so the report states
        # the material it actually ran rather than the series' intent.
        "core_shell": spec.get("core_shell", False),
        "G_core_pa": spec.get("G_core_pa"),
        "Gc_core_j_m2": spec.get("Gc_core_j_m2"),
        "k0p_core_m2": spec.get("k0p_core_m2"),
        "permeability_model": s21_spec(spec, "permeability_model"),
        "skeleton_diffuse_domain": s21_spec(spec, "skeleton_diffuse_domain"),
        "growth": spec["growth"],
        "growth_rate_multiplier": spec.get("growth_rate_multiplier", 1.0),
        "alpha_m2_s": spec.get("alpha_m2_s"),
        "alpha_physiological_m2_s": spec.get("alpha_physiological_m2_s"),
        "wall_condition": spec["wall_condition"],
        "adhesive_strength_pa": spec["adhesive_strength_pa"],
        "exit_code": exit_code,
        "wall_seconds": wall_seconds,
        "concurrency": concurrency,
        "step_count": len(times),
        "final_log_time_s": max(times) if times else None,
        "completed": completed,
        "watcher_stop": watcher_stop,
        "watcher_verdict": (None if watcher_verdict is None else {
            "verdict": watcher_verdict.get("verdict"),
            "reason": watcher_verdict.get("reason"),
            "frame_read": watcher_verdict.get("frame_read"),
            "failed_clauses": [c.get("clause") for c in watcher_verdict.get("clauses") or []
                               if isinstance(c, dict) and c.get("status") in ("FAIL", "ERROR")],
            "stop": watcher_verdict.get("stop")}),
        "status": ("STOPPED_BY_WATCHER" if stopped_by_watcher
                   else "COMPLETE" if completed else "INCOMPLETE"),
        "verdict_note": ("STOPPED_BY_WATCHER" if stopped_by_watcher
                         else "WATCHER-STOPPED" if watcher_stop is not None else None),
        "ran": len(times) > 0,
        "cohesive_release_cells": sum(int(n) for n in DETACHMENT_LINE.findall(text)),
        # From the WRITTEN shearRate field, not from a log line. The solver
        # prints no such line -- an earlier draft of this script matched one
        # that does not exist, which would have reported None forever while
        # the control file claimed the enhancement threshold was checked.
        "max_shear_rate_1_s": None,
        "exchange": {
            "settled_steps": len(settled),
            "settled_unconverged_fraction": (
                sum(1 for s in settled if not s["converged"]) / len(settled)
                if settled else None
            ),
            "settled_max_residual": max((s["residual"] for s in settled), default=None),
        },
        "skeleton": {
            "sweeps": len(skeleton),
            "fraction_at_cap": (at_cap / len(skeleton)) if skeleton else None,
            "max_residual": max((r for _, r in skeleton), default=None),
        },
        "released_cells_max": max(
            (int(m[1]) for m in RELEASED_LINE.findall(text)), default=0
        ),
        "skeleton_diverged": bool(DIVERGED_LINE.search(text)),
        "skeleton_divergence": (
            {"time": float(m.group(1)), "corrector": int(m.group(2)),
             "reason": m.group(3)}
            if (m := DIVERGED_LINE.search(text)) else None
        ),
        # The rigid-body velocity the released clot is actually carried at.
        # Reported so the transport is auditable: if the clot does not move,
        # this says whether the velocity was zero or the advection was.
        #
        # THE RATIO IS THE POINT. Run 20260811-100050 carried the clot at
        # 4.06e-6 m/s against a 2.46e-3 free stream -- a ratio of 0.0016 --
        # because the ambient shell was sampled inside the diffuse interface.
        # The velocity alone looked like a small number among small numbers;
        # the ratio would have named the defect on sight.
        "body_velocity_max_m_s": max(
            (abs(v["Ux"]) for v in body_velocity), default=None
        ),
        "body_velocity_free_stream_ratio": max(
            (v["ratioToFreeStream"] for v in body_velocity), default=None
        ),
        "free_stream_m_s": max(
            (abs(v["freeStreamUx"]) for v in body_velocity), default=None
        ),
        # The measured free stream's tilt off the geometric axis. Before
        # 20260813-084207 the flow direction WAS the measured direction, so
        # this tilt was absorbed into it and the "streamwise" projection kept
        # 1-5% of cross-stream velocity, lifting the phase field into the wall.
        "axis_tilt_max": max(
            (v["axisTilt"] for v in body_velocity if "axisTilt" in v),
            default=None,
        ),
        # Ubody is projected onto a fixed geometric axis, so any residual
        # cross-stream component is a bug, not a small number.
        "body_velocity_cross_max_m_s": max(
            (abs(v.get("Uy", 0.0)) for v in body_velocity), default=None
        ),
        # THE ARTEFACT CHECK. interface/bulk near or above 1 means the
        # damage is being driven by the phase interface rather than by the
        # material, and every tear number below it is meaningless.
        "damage_driver": {
            "H_bulk_max": max((d["Hbulk"] for d in driver), default=None),
            "H_interface_max": max((d["Hinterface"] for d in driver), default=None),
            "H_blood_max": max((d["Hblood"] for d in driver), default=None),
            "interface_over_bulk": (
                max(d["Hinterface"] for d in driver)
                / max(d["Hbulk"] for d in driver)
                if driver and max(d["Hbulk"] for d in driver) > 0 else None
            ),
            # Core/shell only; None when the solver printed no such bins.
            # Above 1 the CORE is storing more driving energy than the shell,
            # which inverts the design and means the material assignment or
            # the shell field is wrong.
            "H_core_max": max(
                (d["Hcore"] for d in driver if "Hcore" in d), default=None
            ),
            "H_shell_max": max(
                (d["Hshell"] for d in driver if "Hshell" in d), default=None
            ),
            "core_over_shell": (
                max(d["Hcore"] for d in driver if "Hcore" in d)
                / max(d["Hshell"] for d in driver if "Hshell" in d)
                if any("Hcore" in d for d in driver)
                and max((d["Hshell"] for d in driver if "Hshell" in d),
                        default=0.0) > 0
                else None
            ),
        },
        # THE PERMEABILITY THE CLOT ACTUALLY RAN WITH, from the solver's own
        # line rather than from k0p. Under kozenyCarman these differ by more
        # than two orders and the difference was invisible until 2026-08-14:
        # every bulk-clot cell of run 20260814-055141 held exactly the
        # numerical floor, 2.22e-16 m2, so k0p had no effect on the clot at
        # all. floored_fraction near 1 means the porosity axis is measuring
        # the floor.
        "permeability": {
            "k_clot_min_m2": min(
                (p["kClotMin"] for p in permeability if p.get("clotCells", 0)),
                default=None,
            ),
            "k_clot_max_m2": max(
                (p["kClotMax"] for p in permeability if p.get("clotCells", 0)),
                default=None,
            ),
            "floored_fraction_max": max(
                (p["flooredCells"] / p["clotCells"]
                 for p in permeability if p.get("clotCells", 0)),
                default=None,
            ),
            "floor_m2": permeability[-1]["floor"] if permeability else None,
            "model": s21_spec(spec, "permeability_model"),
        },
        "tear": tear_measurement(case, spec),
        "transport": transport_measurement(case, spec),
        "s10": s10_report(text, spec.get("G_pa")),
        "s11": s11_report(text),
        "s12": s12_report(text),
        "s13": s13_report(text),
        "s14": s14_report(text),
        "s15": s15_report(text),
        "s16": s16_report(text, spec),
        "s17": s17_report(text, spec),
        "s17b": s17b_report(text, spec),
        "s18": s18_report(text, spec) if "s18" in spec else None,
        "s18p": s18p_report(text, spec) if "s18p" in spec else None,
        "s19": s19_report(text, spec, case) if "s19" in spec else None,
        # Only the throat series has a throat. Everything else gets
        # None rather than a zero, so an absent measurement can never
        # be read as a measured absence of blockage.
        "occlusion": (
            occlusion_measurement(case, spec)
            if spec["series"] in (
                "throat_occlusion", "throat_lodged", "throat_embolus"
            )
            else None
        ),
        "_debonds": debonds,
        "_volumes": volumes,
    }
    # XU 2017 FIGURE 5: the design's section 4 table, ONLY for that series,
    # so no other child's report changes shape. A defect in reading one
    # child's fields is reported on that child, visibly, rather than taking
    # the whole assessment down with it.
    if spec.get("series") == XU_SERIES_NAME:
        try:
            result["xu2017"] = xu2017_report(case, text, spec, result)
        except (ValueError, KeyError, OSError, ZeroDivisionError, IndexError,
                TypeError) as error:
            result["xu2017"] = {"error": f"the Xu report could not be computed: "
                                         f"{type(error).__name__}: {error}",
                                "structural_ok": False}

    # --- bounds, from the written fields ------------------------------------
    # ONLY WRITTEN OUTPUT TIMES COUNT. Every case ships a 0/ directory with a
    # complete set of initial fields, so scoring "the last time directory"
    # happily grades a child that produced nothing at all against its own
    # initial condition -- and it looks perfectly bounded, because the initial
    # condition is. The dry run's missing_fields scenario caught exactly this.
    dirs = [(t_, d) for t_, d in time_dirs(case) if t_ > 0.0]
    bounded, reason = False, "no written output times beyond 0"
    c_min = c_max = dg_max = us_max = perm_min = None
    if dirs:
        final = dirs[-1][1]
        try:
            c = phase_c(final)
            pd = find_field(final, "Dg")
            dg = read_scalars(pd) if pd else None
            pu = find_field(final, "Us")
            us = read_vectors(pu) if pu else None
            pp = find_field(final, "poroPerm")
            perm = read_scalars(pp) if pp else None
            ps = find_field(final, "shearRate")
            if ps is not None:
                s = read_scalars(ps)
                if finite(s):
                    result["max_shear_rate_1_s"] = max(s)
            if c is None or dg is None or us is None:
                reason = (
                    "missing the phase field (c, or q / thetaC+thetaH), "
                    "Dg or Us in the final time"
                )
            elif not (finite(c) and finite(dg)
                      and finite(x for v in us for x in v)):
                reason = "non-finite value in a written field"
            else:
                c_min, c_max = min(c), max(c)
                dg_max = max(dg)
                us_max = max(math.sqrt(sum(x * x for x in v)) for v in us)
                perm_min = min(perm) if perm and finite(perm) else None
                lo, hi, basis = phase_bounds_for(case, controls)
                result["phase_bound_basis"] = basis
                result["phase_bound_lo"] = lo
                dlo, dhi = controls["bounds"]["damage"]
                if not (lo <= c_min and c_max <= hi):
                    reason = (
                        f"phase out of bounds: [{c_min}, {c_max}] "
                        f"against [{lo}, {hi}] ({basis})"
                    )
                elif not (dlo <= dg_max <= dhi):
                    reason = f"damage out of bounds: {dg_max}"
                else:
                    bounded, reason = True, None
        except (OSError, ValueError) as error:
            reason = f"could not read the final fields: {error}"

    # THE TWO-VELOCITY GATE, evaluated only where it applies. A child that
    # never printed the line is not a child that passed -- it is a child that
    # did not run the two-velocity path, and it is recorded as None so the
    # report can tell those two apart.
    skel = parse_skeleton_velocity(text)
    vs_ratio_peak = max(
        (r["maxVs"] / r["maxU"] for r in skel
         if r.get("maxU", 0) > 0 and "maxVs" in r),
        default=None,
    )
    vs_exceed = skel[-1]["exceedFraction"] if skel else None
    vs_exceed_margin = skel[-1]["exceedFractionMargin"] if skel else None
    vs_mean = skel[-1]["meanVsOverU"] if skel else None
    vs_max = max((r["maxVsOverU"] for r in skel), default=None)
    vs_ok = True
    vs_reason = None
    if vs_ratio_peak is not None and vs_ratio_peak > VS_WHOLE_FIELD_LIMIT:
        vs_ok = False
        vs_reason = (
            f"whole-field max|Vs| reached {vs_ratio_peak:.1f}x max|U| "
            f"(limit {VS_WHOLE_FIELD_LIMIT:.0f}). Vs is drag-locked to U "
            f"inside the clot and driven to U by Kres outside it, so this "
            f"ratio is O(1) in every regime the model admits. Note the "
            f"clot-cell metrics can look normal while this fails -- run "
            f"20260905-191135 did exactly that, with the runaway in the "
            f"LUMEN. Check maxVsInClot / maxVsX / maxVsY."
        )
    # S18 (DECISIONS 2026-09-29, decided before batch 60 was packaged): the
    # vs-exceed-fraction gate is a WATCH for S18 children; G8 at 0.1 m/s is
    # the scored slip criterion. Batch 59's verdicts are not re-scored.
    vs_exceed_watch = None
    if ("s18" in spec and vs_exceed_margin is not None
            and vs_exceed_margin > VS_EXCEED_LIMIT):
        vs_exceed_watch = (f"WATCH (S18): {100 * vs_exceed_margin:.1f}% of clot "
                           f"cells above 1.1|U| (limit {100 * VS_EXCEED_LIMIT:.0f}%, "
                           f"not an admissibility gate for S18)")
    if ("s18" not in spec and vs_exceed_margin is not None
            and vs_exceed_margin > VS_EXCEED_LIMIT):
        vs_ok = False
        vs_reason = (
            f"the skeleton outruns the fluid by more than 10% in "
            f"{100 * vs_exceed_margin:.1f}% of clot cells "
            f"(limit {100 * VS_EXCEED_LIMIT:.0f}%; bare |Vs|>|U| count "
            f"{100 * vs_exceed:.1f}%, which was 88.3% before the coupled "
            f"solve). A free element genuinely runs ~0.5% fast, so 10% is "
            f"19x that -- this is a violated mixture balance "
            f"div(sigma_e) = grad(p), not a tolerance."
        )

    # THE CONSERVATION GATE. Applies to every child on every path -- unlike
    # the two-velocity gate, a clot whose volume is not conserved is
    # inadmissible whatever formulation produced it.
    mass_rows = parse_clot_mass(text)
    drift = clot_volume_drift(mass_rows)
    drift_fraction = drift["drift_fraction"]
    mass_ok = True
    mass_reason = None
    if drift_fraction is not None and abs(drift_fraction) > CLOT_DRIFT_LIMIT:
        mass_ok = False
        mass_reason = (
            f"clot volume is not conserved: cumulative drift "
            f"{100 * drift_fraction:+.1f}% of the initial volume "
            f"(limit {100 * CLOT_DRIFT_LIMIT:.0f}%; the discretisation floor "
            f"is 0.03%, measured identically on five children). The clot went "
            f"{drift['volume_ratio']:.2f}x its initial volume. clotResidual "
            f"is -0.5*int c*div(carrier) dV, so this is the phase-field "
            f"transport not being corrected for a carrier that is not "
            f"solenoidal -- not a tolerance."
        )

    pb = parse_phase_bounds(text)
    c_under_peak = max((r["cUnder"] for r in pb), default=None)
    c_over_peak = max((r["cOver"] for r in pb), default=None)
    # REPORTED, DELIBERATELY NOT GATED. Codex's second review (section 8.2)
    # observed that the solver emits qNegCells/qNegMin/qNegAmpMax on this very
    # line and the assessor read cUnder/cOver and threw the rest away, so the
    # `abs(q)` divisor guard was described as "gated" when nothing gated it.
    # Half of that is now fixed: these are surfaced in every result.json, so
    # an invalid state is visible in the record instead of inferred.
    #
    # No threshold is attached, and that omission is the point. q = 0.5(c+1),
    # so qNegCells > 0 is exactly cUnder > 0 -- but as a COUNT it fires on a
    # single cell at q = -1e-17, which is round-off, not an invalid state. A
    # `qNegCells == 0` gate would therefore be unpassable for a reason that
    # has nothing to do with the physics, which is the failure mode that has
    # already produced four dead gates in this project. Codex asked for the
    # distinction explicitly ("keep an exact negative-cell count distinct from
    # a materially-negative count if round-off necessitates that distinction")
    # and for thresholds derived from the instrument's resolution. That
    # derivation is open as Q18. Until it lands, this reports and does not
    # judge. Do not add a threshold here to make a run pass.
    #
    # Optional, not in PHASE_BOUNDS_REQUIRED: requiring it would raise on
    # every log written before the diagnostic existed and make the historical
    # corpus unassessable.
    #
    # COVERAGE IS REPORTED WITH EVERY SUMMARY. Codex, 2026-09-08: summarising
    # only the records that happen to carry a key means a PARTIALLY
    # instrumented history looks exactly like a fully instrumented one -- a
    # peak computed over 12 of 20,099 records prints as a peak. `None` catches
    # a wholly absent field and nothing caught a hole. The denominators below
    # make the hole visible without failing a legacy archive that never had
    # the field at all.
    q_neg_records = sum(1 for r in pb if "qNegCells" in r)
    q_neg_cells_peak = max(
        (r["qNegCells"] for r in pb if "qNegCells" in r), default=None
    )
    q_neg_min = min((r["qNegMin"] for r in pb if "qNegMin" in r), default=None)
    q_neg_amp_max = max(
        (r["qNegAmpMax"] for r in pb if "qNegAmpMax" in r), default=None
    )
    # The integral partners of the pointwise extremum, added to the solver on
    # 2026-09-08. qNegContent = sum_i V_i*max(-q_i,0) is a VIOLATION
    # FUNCTIONAL OF THE LEGACY INDICATOR -- not conserved, not negative mass.
    # See the solver comment; do not build a conservation argument on it.
    q_neg_volume_peak = max(
        (r["qNegVolume"] for r in pb if "qNegVolume" in r), default=None
    )
    q_neg_content_peak = max(
        (r["qNegContent"] for r in pb if "qNegContent" in r), default=None
    )
    q_neg_integral_records = sum(1 for r in pb if "qNegContent" in r)
    # S0.3 -- REPORTED, NOT GATED, AND THE PROVENANCE IS PART OF THE NUMBER.
    #
    # qNegContent is an absolute volume, so it means nothing without a scale.
    # Normalising by the run's OWN initial clot volume (the same v0 the drift
    # fraction uses) gives the fraction of the starting clot that has gone
    # outside the declared q domain, which is comparable across meshes.
    #
    # I chose this metric AFTER seeing Batch 26, which is exactly the move
    # Codex warned against, so it is deliberately not attached to any
    # threshold and must not become one without the resolution study in Q18.
    # It is here so the next tolerance derivation has a scale-free number to
    # work from, not so a run can be judged by it today.
    # No try/except and no getattr guard: parse_clot_mass is defined in this
    # module, so if the name is wrong the import fails loudly instead of this
    # metric silently reading None forever. (It was `parse_mass` for one
    # revision, which would have done exactly that.)
    q_neg_content_over_initial_clot = None
    if q_neg_content_peak is not None:
        # S6c-2: THE DENOMINATOR CHANGED CURRENCY AND IS CONVERTED BACK.
        # parse_clot_mass now reads PHASE_CLOT_CONTENT_BALANCE, whose
        # contentVolume is int theta dV, while q_neg_content_peak is a
        # violation functional of q -- int max(-q, 0) dV. Dividing one by
        # the other unconverted would rescale this number by theta_ref
        # (0.45) against every value on record. theta = theta_ref*(1 - q),
        # so the clot volume in q currency is contentVolume/theta_ref.
        _mass_rows = parse_clot_mass(text)
        if _mass_rows:
            _theta_ref = _mass_rows[0].get("thetaRef") or 0.0
            _v0_q = (
                _mass_rows[0]["contentVolume"]/_theta_ref
                if _theta_ref > 0 else 0.0
            )
            if _v0_q > 0:
                q_neg_content_over_initial_clot = q_neg_content_peak/_v0_q
    # ---- THE NETWORK-PRESSURE CLAMP GATE --------------------------------
    #
    # WHY A BOUND ON theta IS NOT ENOUGH ON ITS OWN. Every form of
    # networkPressure() evaluates at min(phi_s, 0.99*phiSMax), so the LAW can
    # never report a fraction above the clamp. A gate written on the law's
    # own output would be unfailable -- this project retired exactly that
    # defect as G3 of the phase-volume spec.
    #
    # What can fail is the clamp ENGAGING. If the force balance is doing the
    # bounding, no cell reaches 0.99*phiSMax and pnCapCells is 0 all run. A
    # nonzero count says the discretisation walked a cell through the pole
    # between two steps and the CLAMP held it, not the physics -- which is
    # the claim "close packing is unreachable by force balance" failing.
    #
    # ABSENT INSTRUMENT IS NOT A PASS, but it is also not a failure here: the
    # counter ships with networkPressureModel poleCubic, and one-velocity and
    # pre-2026-09-21 children have no such line. Those are recorded as None
    # and left admissible, because a gate must not retroactively fail runs
    # that predate it. A child that DOES print the line and shows a nonzero
    # count fails.
    pn_rows = [
        dict((k, v) for k, v in re.findall(r"(\w+)=([-\w.+]+)", m))
        for m in re.findall(r"^PHASE_CLOT_NETWORK (.*)$", text, re.M)
    ]
    pn_cap_peak = None
    pn_cap_first = None
    pn_model = None
    pn_ok = True
    pn_reason = None
    if pn_rows:
        pn_model = pn_rows[-1].get("pnModel")
        caps = [(r.get("time"), r.get("pnCapCells")) for r in pn_rows
                if "pnCapCells" in r]
        if caps:
            vals = [float(c) for _, c in caps]
            pn_cap_peak = max(vals)
            if pn_cap_peak > 0:
                pn_ok = False
                pn_cap_first = next(
                    float(t) for t, c in caps if float(c) > 0
                )
                pn_reason = (
                    f"the network-pressure clamp engaged in up to "
                    f"{pn_cap_peak:.0f} cells (first at t = "
                    f"{pn_cap_first:.4g} s). phi_s reached 0.99*phiSMax, so "
                    f"close packing was held by the 0.99 clamp rather than "
                    f"by the force balance. The law is "
                    f"{pn_model or 'unknown'}; either it is too soft for the "
                    f"load this case imposes, or the step is too long to "
                    f"resolve the approach to the pole."
                )

    c_bound_ok = True
    c_bound_reason = None
    worst = max(c_under_peak or 0.0, c_over_peak or 0.0)

    # THE TWO EXCURSIONS ARE NOT THE SAME PHYSICS AND MUST NOT SHARE A LIMIT.
    #
    # c = 1 - 2*theta/theta_ref, so
    #
    #   cOver  > 0  <=>  c > 1   <=>  theta < 0      -- NEGATIVE SOLID.
    #   cUnder > 0  <=>  c < -1  <=>  theta > theta_ref -- COMPACTION.
    #
    # The first is unphysical at any magnitude and keeps the tight literal
    # limit. The second is what a network under load DOES, and the model
    # declares how far it may go: theta <= phiSMax. Sharing C_BOUND_LIMIT
    # between them meant a clot that densified by 2.5% was graded the same
    # way as one with negative solid in it.
    #
    # DERIVED, NOT LOOSENED: the ceiling is 2*theta_max/theta_ref - 2, the
    # exact cUnder at close packing, with the control's own 1.005 margin. It
    # can still fail -- nothing clips theta -- and the run that motivated
    # this, 20260921-072935, reached cUnder 0.3826 while still accelerating,
    # which is a network with no equilibrium rather than one at its limit.
    _tr, _tmax = theta_ref_of_case(case), phi_s_max_of_case(case)
    if _tr and _tmax and _tmax > _tr:
        c_under_limit = 1.005 * (2.0 * _tmax / _tr - 2.0)
        c_under_basis = f"2*{_tmax:g}/{_tr:g} - 2, close packing"
    else:
        c_under_limit = C_BOUND_LIMIT
        c_under_basis = "control literal (no theta_ref/phiSMax in the case)"
    result["c_under_limit"] = c_under_limit
    result["c_under_limit_basis"] = c_under_basis
    # FAIL CLOSED ON AN ABSENT HISTORY. The failure branch below is guarded by
    # `if pb and ...`, so a log carrying NO PHASE_CLOT_PHASE_BOUNDS lines at
    # all used to leave c_bound_gate_ok True and the child admissible -- an
    # empty parse read as a pass. That is the silent-wrong-answer class: an
    # old build, a renamed diagnostic or a truncated log would all have
    # produced a clean bill of health for a run nobody had checked. Codex,
    # 2026-09-08. A run that cannot be assessed is not a run that passed.
    if not pb:
        c_bound_ok = False
        c_bound_reason = (
            "no PHASE_CLOT_PHASE_BOUNDS records in the log, so the phase "
            "field was never checked against [-1, 1]. This is a missing "
            "instrument, not a clean result: an unassessable run fails."
        )
    elif (c_over_peak or 0.0) > C_BOUND_LIMIT or \
            (c_under_peak or 0.0) > c_under_limit:
        c_bound_ok = False
        first = next(
            (r["time"] for r in pb
             if r["cOver"] > C_BOUND_LIMIT or r["cUnder"] > c_under_limit),
            None
        )
        c_bound_reason = (
            f"the phase field left [-1, 1] by {worst:.4g} "
            f"(cUnder limit {c_under_limit:.4g} = {c_under_basis}; "
            f"cOver limit {C_BOUND_LIMIT}; "
            f"every other run on disk has cUnder "
            f"exactly 0 and cOver below 1.1e-13), first at t = {first:.4g}. "
            f"q = 0.5(c+1) is the legacy accessibility indicator (NOT a "
            f"solid volume fraction -- that is theta = 0.45*chi), so below -1 "
            f"it goes negative and every closure that reads q is outside its "
            f"declared domain. The divisor now uses max(|q|, qFloor), so w "
            f"is no longer sign-flipped and only |q| < qFloor still saturates "
            f"at 1/qFloor; the field is invalid regardless."
        )

    # ---- THE SHELL SYSTEM MUST BE SOLVABLE AS ASSEMBLED ----
    #
    # THIS IS A PRECONDITION, NOT A TOLERANCE, which is why it is gated when
    # every other new measurement this week was only reported. A zero diagonal
    # is a division by zero inside symGaussSeidelSmoother and a non-finite
    # coefficient is an invalid operation: both are exactly the SIGFPE that
    # ended 13 of 14 two-velocity runs. There is no threshold to choose and no
    # instrument resolution to derive -- a matrix is either finite or it is
    # not. Codex's warning against inventing cutoffs does not apply to a
    # predicate with no cutoff in it.
    #
    # minRowDominance is deliberately NOT gated. How much diagonal dominance
    # is enough IS a tolerance question, it bounds ||A^-1|| rather than
    # deciding solvability, and Gauss-Seidel converges on plenty of
    # non-dominant systems. It is reported so the next tolerance derivation
    # has data.
    #
    # Legacy-safe: a log with no PHASE_CLOT_SHELL_MATRIX records is NOT
    # EVALUATED rather than failed -- every run before 2026-09-08 predates the
    # instrument, and failing them would make the returned corpus
    # unassessable. Once a log declares the record, a bad one fails.
    shell_mx = parse_shell_matrix(text)
    content_recs, content_ops = parse_content(text)
    content_malformed = [
        r for r in content_recs if content_record_invalid(r) is not None
    ]
    content_malformed_reasons = sorted(set(
        content_record_invalid(r) for r in content_recs
        if content_record_invalid(r) is not None
    ))
    content_operators = sorted(set(o for o in content_ops if o))
    content_not_advected = len(CONTENT_NOT_ADVECTED.findall(text))
    shell_mx_ok = True
    shell_mx_reason = None
    # S0.2 -- a present key is not a valid one. Codex, 2026-09-09: `nan` is
    # dropped by the numeric regex (so the key goes missing and the required-key
    # test catches it), but a NEGATIVE counter or an infinite margin parses
    # fine and would sail through. Counters are cardinalities; margins and
    # diagonals are finite reals. Anything else means the emitter or the log is
    # not what this gate was written against.
    SHELL_MX_COUNTERS = (
        "rowsNotDominant", "zeroDiagCells", "nonFiniteDiag",
        "nonFiniteOffDiag", "nonFiniteSource", "nonFinitePsi",
    )

    def _shell_mx_invalid(r: dict) -> str | None:
        missing = [k for k in SHELL_MATRIX_REQUIRED if k not in r]
        if missing:
            return f"missing {missing}"
        for k, v in r.items():
            if not math.isfinite(v):
                return f"{k}={v} is not finite"
        for k in SHELL_MX_COUNTERS:
            if r[k] < 0 or r[k] != int(r[k]):
                return f"{k}={r[k]} is not a non-negative integer count"
        return None

    shell_mx_faults = [
        (i, _shell_mx_invalid(r)) for i, r in enumerate(shell_mx)
    ]
    shell_mx_malformed = [i for i, why in shell_mx_faults if why]
    shell_mx_min_diag = None
    shell_mx_min_dom = None
    shell_mx_rows_not_dom = None
    if shell_mx and shell_mx_malformed:
        # Fail before computing any summary: a peak over the records that
        # happen to be complete is not a peak over the run.
        shell_mx_ok = False
        why = next(w for _, w in shell_mx_faults if w)
        shell_mx_reason = (
            f"{len(shell_mx_malformed)} of {len(shell_mx)} "
            f"PHASE_CLOT_SHELL_MATRIX records are unusable (first at index "
            f"{shell_mx_malformed[0]}: {why}). The solvability predicate reads "
            f"those fields, so an incomplete or invalid record would score as "
            f"healthy. An unassessable matrix history fails."
        )
    elif shell_mx:
        shell_mx_min_diag = min(
            r["minMagDiagFinal"] for r in shell_mx if "minMagDiagFinal" in r
        ) if any("minMagDiagFinal" in r for r in shell_mx) else None
        shell_mx_min_dom = min(
            r["minRowDominance"] for r in shell_mx if "minRowDominance" in r
        ) if any("minRowDominance" in r for r in shell_mx) else None
        shell_mx_rows_not_dom = max(
            (r.get("rowsNotDominant", 0.0) for r in shell_mx), default=None
        )
        bad = [
            r for r in shell_mx
            if r.get("zeroDiagCells", 0.0) > 0
            or r.get("nonFiniteDiag", 0.0) > 0
            or r.get("nonFiniteOffDiag", 0.0) > 0
            or r.get("nonFiniteSource", 0.0) > 0
            or r.get("nonFinitePsi", 0.0) > 0
        ]
        if bad:
            shell_mx_ok = False
            first = bad[0]
            shell_mx_reason = (
                f"the shell system was not solvable as assembled at "
                f"t = {first.get('time', float('nan')):.6g} and in "
                f"{len(bad)} of {len(shell_mx)} records: "
                f"zeroDiag={first.get('zeroDiagCells', 0):.0f}, nonFinite "
                f"diag/offdiag/source/psi = "
                f"{first.get('nonFiniteDiag', 0):.0f}/"
                f"{first.get('nonFiniteOffDiag', 0):.0f}/"
                f"{first.get('nonFiniteSource', 0):.0f}/"
                f"{first.get('nonFinitePsi', 0):.0f}. "
                f"symGaussSeidelSmoother divides by that diagonal, so this is "
                f"the SIGFPE that ended 13 of 14 two-velocity runs, caught as "
                f"a measurement instead of a crash."
            )

    # -- THE CARRIER RECORD (2026-09-14) -----------------------------------
    #
    # Standing rule: every diagnostic the solver prints must be read by a
    # gate, or deleted. PHASE_CLOT_CARRIER was emitted from 2026-09-08 and
    # read by nothing for six days, which is how the deciding numbers of the
    # carrier experiment would have had to be excavated from raw logs by hand.
    #
    # WHAT IS GATED HERE IS CONFIGURATION AGREEMENT, NOT A THRESHOLD.
    # `materialCarrierBlend` defaults to FALSE in the solver -- the run record
    # of 2026-09-08 says "default ON", which is wrong -- so a child that means
    # to run the fix and fails to pass the key runs the configuration that
    # killed 14 of 14 two-velocity runs, and its log looks entirely normal.
    # em26_blend_on versus em26_blend_off is only a discriminator if each arm
    # actually ran the carrier it declared. The numbers themselves are
    # REPORTED, not gated: the solver deliberately does not fault on a
    # non-positive diagonal, because killing the run would destroy the
    # correlation the record exists to collect.
    carrier = parse_carrier(text)
    carrier_ok = True
    carrier_reason = None
    carrier_blend_seen = None
    carrier_co_peak = None
    carrier_mag_peak = None
    carrier_min_diag = None
    carrier_nonpos_peak = None
    carrier_nonfinite_peak = None

    def _carrier_invalid(r: dict) -> str | None:
        missing = [k for k in CARRIER_REQUIRED if k not in r]
        if missing:
            return f"missing {missing}"
        if "carrierBlend" not in r:
            return "missing carrierBlend, so the configuration is unknown"
        for k in CARRIER_REQUIRED:
            if not math.isfinite(r[k]):
                return f"{k}={r[k]} is not finite"
        for k in CARRIER_COUNTERS:
            if r[k] < 0 or r[k] != int(r[k]):
                return f"{k}={r[k]} is not a non-negative integer count"
        return None

    if carrier:
        faults = [(i, _carrier_invalid(r)) for i, r in enumerate(carrier)]
        malformed = [i for i, why in faults if why]
        if malformed:
            # Same fail-before-summarising rule as the shell-matrix gate: a
            # peak over the records that happen to be complete is not a peak
            # over the run.
            carrier_ok = False
            why = next(w for _, w in faults if w)
            carrier_reason = (
                f"{len(malformed)} of {len(carrier)} PHASE_CLOT_CARRIER "
                f"records are unusable (first at index {malformed[0]}: "
                f"{why}). An unassessable carrier history fails."
            )
        else:
            blends = {r["carrierBlend"] for r in carrier}
            if len(blends) != 1:
                carrier_ok = False
                carrier_reason = (
                    f"carrierBlend changed within one run: {sorted(blends)}. "
                    f"Every other number on these records belongs to one "
                    f"configuration or the other, and this run cannot say "
                    f"which."
                )
            else:
                carrier_blend_seen = blends.pop()
            carrier_co_peak = max(r["maxCarrierCo"] for r in carrier)
            carrier_mag_peak = max(r["maxMagCarrier"] for r in carrier)
            carrier_nonpos_peak = max(r["shellDiagNonPositive"] for r in carrier)
            carrier_nonfinite_peak = max(r["shellDiagNonFinite"] for r in carrier)
            # minShellDiagOverVbyDt is computed ONLY over finite diagonals
            # (see the emitter's loop), so when every diagonal on a step is
            # non-finite the reported minimum is the `great` sentinel and
            # reads as perfectly healthy. Only summarise the records where it
            # means something, and say so when some were skipped.
            usable = [r for r in carrier if r["shellDiagNonFinite"] == 0]
            carrier_min_diag = (
                min(r["minShellDiagOverVbyDt"] for r in usable) if usable
                else None
            )

    # DECLARED VERSUS OBSERVED. The child carries its own spec on disk.
    # `spec` is the child's own entry from mainCaseControls.json -- the same
    # object create_cases.py writes to child.json -- so it is read here rather
    # than re-parsed off disk, which removes a failure path that could only
    # ever silence this comparison.
    declared_blend = s21_spec(spec, "material_carrier_blend")
    if declared_blend is not None and carrier_blend_seen is not None:
        want = "ON" if declared_blend else "OFF"
        if carrier_blend_seen != want:
            carrier_ok = False
            carrier_reason = (
                f"the child matrix declares material_carrier_blend="
                f"{declared_blend}, so the solver should report "
                f"carrierBlend={want}, but every PHASE_CLOT_CARRIER record "
                f"says {carrier_blend_seen}. The key did not reach the "
                f"solver, and this child is not the experiment it claims to "
                f"be. materialCarrierBlend defaults to FALSE."
            )

    plate = parse_platelet(text)
    result.update(platelet_metric_summary(plate))
    wp_dev_peak = max((r["WpDevMax"] for r in plate), default=None)
    wp_ok = True
    wp_reason = None
    # Same fail-closed rule as the phase-bound gate above: no platelet
    # records means the Wp/q envelope was never measured, which is a missing
    # instrument rather than a pass.
    if wp_dev_peak is None:
        wp_ok = False
        wp_reason = (
            "no PHASE_CLOT_PLATELET records in the log, so WpDevMax was "
            "never measured. An unassessable run fails."
        )
    elif wp_dev_peak > WP_DEV_LIMIT:
        wp_ok = False
        wp_reason = (
            f"WpDevMax (the emitter's effective-q reference, not certified "
            f"as raw q) reached {wp_dev_peak:.3f} (envelope limit "
            f"{WP_DEV_LIMIT}; the widest one-velocity run measured 0.111). "
            f"This fails the historical envelope screen; it does not by "
            f"itself establish depletion, conservation failure, or causation."
        )

    # G3'lambda ADMISSIBILITY (research/designs/2026-10-05-S20-content-fct-neutral-zero-flux.md
    # rev. 5 section 2.3). s10_report computes it; this is the verdict that reads it. A log
    # whose FCT lines carry zeroFluxRule=exactZero and whose median lambdaFluxMean over the
    # steps with activeFaces > 0 is below G3_LAMBDA_MEDIAN_MIN (g3_lambda False) admitted no
    # second-order content flux: batch 68's fsf_G4 read 0 on every step. None (a pre-S20
    # log, or no active step) is N/A and does not gate.
    g3l = result["s10"]["g3_lambda"]
    g3l_ok = g3l is not False
    g3l_reason = None
    if not g3l_ok:
        s10 = result["s10"]
        med = s10["g3_lambda_median"]
        g3l_reason = (
            f"G3'lambda: the median lambdaFluxMean over the "
            f"{s10['fct_active_records']} PHASE_CLOT_CONTENT_FCT records with "
            f"activeFaces > 0 is {'n/a' if med is None else f'{med:.6g}'} (must be >= "
            f"{s10['g3_lambda_min_median']}; S20 section 2.3)"
            + (f"; {s10['g3_lambda_note']}" if s10.get("g3_lambda_note") else "")
            + ". The content limiter admitted no second-order flux: inadmissible."
        )

    result.update({
        "g3_lambda_ok": g3l_ok,
        "g3_lambda_reason": g3l_reason,
        "carrier_ok": carrier_ok,
        "carrier_reason": carrier_reason,
        "carrier_records": len(carrier),
        "carrier_blend_observed": carrier_blend_seen,
        "carrier_blend_declared": declared_blend,
        "carrier_max_courant": carrier_co_peak,
        "carrier_max_magnitude": carrier_mag_peak,
        "carrier_min_shell_diag_over_VbyDt": carrier_min_diag,
        "carrier_shell_diag_non_positive_peak": carrier_nonpos_peak,
        "carrier_shell_diag_non_finite_peak": carrier_nonfinite_peak,
        "c_min": c_min, "c_max": c_max, "Dg_max": dg_max, "Us_max_m": us_max,
        "poroPerm_min_m2": perm_min,
        "bounded": bounded,
        "bounds_reason": reason,
        "vs_exceed_fraction": vs_exceed,
        "vs_exceed_fraction_margin": vs_exceed_margin,
        "vs_exceed_fraction_watch_s18": vs_exceed_watch,
        "vs_mean_over_u": vs_mean,
        "vs_max_over_u": vs_max,
        "vs_whole_field_ratio_peak": vs_ratio_peak,
        "vs_gate_ok": vs_ok,
        "vs_gate_reason": vs_reason,
        "clot_volume_initial_m3": drift["volume_initial_m3"],
        "clot_volume_final_m3": drift["volume_final_m3"],
        "clot_volume_ratio": drift["volume_ratio"],
        "clot_drift_fraction": drift_fraction,
        "mass_gate_ok": mass_ok,
        "mass_gate_reason": mass_reason,
        "wp_dev_peak": wp_dev_peak,
        "wp_gate_ok": wp_ok,
        "wp_gate_reason": wp_reason,
        "c_under_peak": c_under_peak,
        "c_over_peak": c_over_peak,
        "q_neg_cells_peak": q_neg_cells_peak,
        "q_neg_min": q_neg_min,
        "q_neg_amp_max": q_neg_amp_max,
        "q_neg_volume_peak": q_neg_volume_peak,
        "q_neg_content_peak": q_neg_content_peak,
        "q_neg_content_over_initial_clot": q_neg_content_over_initial_clot,
        "phase_bound_records": len(pb),
        "q_neg_records": q_neg_records,
        "q_neg_integral_records": q_neg_integral_records,
        "c_bound_gate_ok": c_bound_ok,
        "c_bound_gate_reason": c_bound_reason,
        # STAGE 1 CONTENT MODEL -- REPORTED, NOT GATED. See CONTENT_REQUIRED.
        "content_records": len(content_recs),
        "content_malformed_records": len(content_malformed),
        "content_operators": content_operators,
        "content_not_advected_steps": content_not_advected,
        "content_malformed_reasons": content_malformed_reasons,
        # NOT_EVALUATED means "the content path did not run". If it DID run and
        # reported that it could not advect, that is a content path which
        # executed and produced nothing scoreable -- UNASSESSABLE. Codex,
        # 2026-09-10: these were conflated, so a run whose every corrector took
        # the no-carrier branch looked like a run with the model switched off.
        "content_status": (
            "UNASSESSABLE" if (not content_recs and content_not_advected)
            else "NOT_EVALUATED" if not content_recs
            else "UNASSESSABLE" if content_malformed or content_not_advected
            else "REPORTED"
        ),
        # S6c-2: retired with c. Kept as explicit Nones rather than dropped
        # from the record, so a consumer reading an old key gets a null it
        # must handle instead of a KeyError it might swallow -- and so a
        # diff of two reports shows the field going quiet rather than the
        # schema changing shape.
        "content_equiv_max_final": None,
        "content_equiv_max_peak": None,
        "content_equiv_l1_final": None,
        "content_theta_neg_records": (
            sum(
                1 for r in content_recs
                if r["thetaCNeg"] > 0 or r["thetaHNeg"] > 0
            )
            if content_recs and not content_malformed else None
        ),
        # The inventory balance. Under `content` the source is zero and this
        # is pure drift; under `matched` the source ADDS material, so the
        # right comparison is the change against the source, not against zero.
        "content_volume_first": (
            content_recs[0]["contentVolume"]
            if content_recs and not content_malformed else None
        ),
        "content_volume_last": (
            content_recs[-1]["contentVolume"]
            if content_recs and not content_malformed else None
        ),
        # ONE RECORD PER TIME, NOT ONE PER CORRECTOR.
        #
        # solvePhaseField() runs twice per step, so there are two records per
        # time, and each recomputes from the SAME old-time state. Summing all
        # of them double-counts the source: measured on run 20260909-182243
        # child em28_content_matched, 1.28877e-18 against a realised change of
        # 6.44386e-19, a factor of 2.0000. Only the last corrector's result
        # survives the step, so only its source was actually applied. Summed
        # that way the total is 6.443864991e-19 against a realised
        # 6.443864985e-19 -- eight digits.
        "content_source_dt_sum": (
            sum(
                r["contentSourceDt"]
                for r in {
                    rec["time"]: rec for rec in content_recs
                }.values()
            )
            if content_recs and not content_malformed else None
        ),
        "shell_matrix_records": len(shell_mx),
        "shell_matrix_malformed_records": len(shell_mx_malformed),
        # S0.1 -- NOT_EVALUATED and PASS are different facts and must not
        # share a True. A legacy log predates the instrument; a complete
        # healthy history actually exercised it. Only shell_matrix_records
        # distinguished them before, and nothing was reading it.
        "shell_matrix_status": (
            "NOT_EVALUATED" if not shell_mx
            else "UNASSESSABLE" if shell_mx_malformed
            else "PASS" if shell_mx_ok
            else "FAIL"
        ),
        "carrier_status": (
            "NOT_EVALUATED" if not carrier
            else "PASS" if carrier_ok
            else "FAIL"
        ),
        "shell_matrix_gate_ok": shell_mx_ok,
        "shell_matrix_gate_reason": shell_mx_reason,
        "shell_matrix_min_mag_diag": shell_mx_min_diag,
        "shell_matrix_min_row_dominance": shell_mx_min_dom,
        "shell_matrix_rows_not_dominant": shell_mx_rows_not_dom,
        "pn_cap_cells_peak": pn_cap_peak,
        "pn_cap_first_time_s": pn_cap_first,
        "pn_model": pn_model,
        "pn_gate_ok": pn_ok,
        "pn_gate_reason": pn_reason,
        # THE gate for every table below. Named so an excluded child is
        # visible in the report rather than simply absent from a row.
        "admissible": bool(
            completed and bounded and vs_ok and mass_ok and wp_ok
            and c_bound_ok and shell_mx_ok and carrier_ok and pn_ok and g3l_ok
        ),
    })
    if not vs_ok and not result["bounds_reason"]:
        result["bounds_reason"] = vs_reason
    if not mass_ok and not result["bounds_reason"]:
        result["bounds_reason"] = mass_reason
    if not pn_ok and not result["bounds_reason"]:
        result["bounds_reason"] = pn_reason
    if not c_bound_ok and not result["bounds_reason"]:
        result["bounds_reason"] = c_bound_reason
    if not wp_ok and not result["bounds_reason"]:
        result["bounds_reason"] = wp_reason
    if not shell_mx_ok and not result["bounds_reason"]:
        result["bounds_reason"] = shell_mx_reason
    if not carrier_ok and not result["bounds_reason"]:
        result["bounds_reason"] = carrier_reason
    if not g3l_ok and not result["bounds_reason"]:
        result["bounds_reason"] = g3l_reason
    return result


def growth_table(children: dict, controls: dict) -> dict:
    rows, excluded = [], []
    for c in sorted(children.values(), key=lambda r: r["wall_shear_rate_1_s"]):
        if c["series"] != "growth":
            continue
        if not c["admissible"]:
            excluded.append({"child_id": c["child_id"],
                             "why": c["bounds_reason"] or "did not complete"})
            continue
        vol = c["_volumes"]
        entry = {
            "child_id": c["child_id"],
            "wall_shear_rate_1_s": c["wall_shear_rate_1_s"],
            "mean_velocity_m_s": c["mean_velocity_m_s"],
            "max_shear_rate_1_s": c["max_shear_rate_1_s"],
            "Vn_m_s": None,
            "reason": None,
        }
        # Discard the first fifth so the startup transient does not enter the
        # fit, exactly as the mural gate did.
        settled = vol[int(len(vol) * 0.2):]
        if len(settled) < 3:
            entry["reason"] = f"only {len(settled)} settled volume samples"
        else:
            window = settled[-1][0] - settled[0][0]
            dv = settled[-1][1] - settled[0][1]
            g = controls["_geometry"]
            area = g["injury_length_m"] * g["depth_m"]
            entry.update({
                "window_s": window,
                "dVdt_m3_s": dv / window if window > 0 else None,
                "growing_face_area_m2": area,
                "Vn_m_s": (dv / window / area) if window > 0 else None,
            })
        rows.append(entry)

    # The transport-limited prediction: Vn ~ U^(1/3). Fitted, not assumed.
    fit = {"exponent": None, "predicted_exponent": 1.0 / 3.0, "reason": None}
    usable = [r for r in rows if r.get("Vn_m_s") and r["Vn_m_s"] > 0]
    if len(usable) < 2:
        fit["reason"] = f"only {len(usable)} usable points; need two"
    else:
        lo, hi = usable[0], usable[-1]
        ratio_u = hi["mean_velocity_m_s"] / lo["mean_velocity_m_s"]
        ratio_v = hi["Vn_m_s"] / lo["Vn_m_s"]
        if ratio_u <= 1 or ratio_v <= 0:
            fit["reason"] = "degenerate flow or growth ratio"
        else:
            fit.update({
                "exponent": math.log(ratio_v) / math.log(ratio_u),
                "velocity_ratio": ratio_u,
                "growth_ratio": ratio_v,
                "note": (
                    "A concentration boundary layer of thickness "
                    "delta_c ~ (H/2)/Pe^(1/3) makes a mass-transfer-limited "
                    "front advance as U^(1/3). A measured exponent near 1/3 "
                    "says growth here is transport-limited; near 0 says it is "
                    "reaction-limited and the flow barely matters."
                ),
            })
    return {"rows": rows, "excluded": excluded, "flow_dependence": fit}


def required_strength_table(children: dict) -> dict:
    rows, excluded = [], []
    for c in sorted(
        (r for r in children.values() if r["series"] == "required_strength"),
        key=lambda r: (r["G_pa"], r["k0p_m2"], r["wall_shear_rate_1_s"]),
    ):
        if not c["admissible"]:
            excluded.append({"child_id": c["child_id"],
                             "why": c["bounds_reason"] or "did not complete"})
            continue
        peaks = [d["peak_traction_pa"] for d in c["_debonds"]]
        rows.append({
            "child_id": c["child_id"],
            "wall_shear_rate_1_s": c["wall_shear_rate_1_s"],
            "G_pa": c["G_pa"],
            "k0p_m2": c["k0p_m2"],
            "required_strength_pa": max(peaks) if peaks else None,
            "reason": None if peaks else (
                "no PHASE_CLOT_DEBOND lines: the welded condition does not "
                "report a traction, so this operating point cannot be scored"
            ),
            "Us_max_m": c["Us_max_m"],
            "poroPerm_min_m2": c["poroPerm_min_m2"],
        })

    # Is the load set by the flow rather than by the stiffness? Stated as a
    # prediction in the control file; measured here.
    prediction = {"claim": (
        "the adhesive load is set by the flow, not the stiffness, because the "
        "drag is a fluid quantity and the traction G*strain has the G cancel"
    ), "supported": None, "reason": None}
    scored = [r for r in rows if r["required_strength_pa"] is not None]
    by_flow = {}
    for r in scored:
        if r["k0p_m2"] == 3.5e-14:
            by_flow.setdefault(r["wall_shear_rate_1_s"], []).append(r)
    spreads = {
        flow: (max(x["required_strength_pa"] for x in group)
               / min(x["required_strength_pa"] for x in group))
        for flow, group in by_flow.items()
        if len(group) > 1 and min(x["required_strength_pa"] for x in group) > 0
    }
    if not spreads:
        prediction["reason"] = "no flow has two or more stiffnesses scored"
    else:
        prediction["stiffness_spread_at_fixed_flow"] = spreads
        prediction["stiffness_span"] = 11600.0 / 58.6
        prediction["supported"] = all(v < 2.0 for v in spreads.values())
        prediction["note"] = (
            "Supported means the required strength varies by less than 2x "
            "across a 198x span in stiffness. If it is NOT supported, "
            "stiffness enters the bond load directly and that is the more "
            "interesting answer."
        )
    return {"rows": rows, "excluded": excluded, "flow_not_stiffness": prediction}


def demonstration_table(children: dict) -> dict:
    rows, excluded = [], []
    for c in sorted(
        (r for r in children.values() if r["series"] == "demonstration"),
        key=lambda r: r["adhesive_strength_pa"],
    ):
        if not c["admissible"]:
            excluded.append({"child_id": c["child_id"],
                             "why": c["bounds_reason"] or "did not complete"})
            continue
        d = c["_debonds"]
        fired = [e for e in d if e["new_faces"] > 0]
        total = max((e["debonded_faces"] for e in d), default=0)
        initial_bonded = max(
            (e["bonded_clot_faces"] + e["debonded_faces"] for e in d), default=0
        )
        rows.append({
            "child_id": c["child_id"],
            "adhesive_strength_pa": c["adhesive_strength_pa"],
            "wall_shear_rate_1_s": c["wall_shear_rate_1_s"],
            "k0p_m2": c["k0p_m2"],
            "first_release_time_s": fired[0]["time"] if fired else None,
            "steps_with_releases": len(fired),
            "faces_released": total,
            "bonded_faces_initially": initial_bonded,
            "released_fraction": (
                total / initial_bonded if initial_bonded else None
            ),
            "largest_single_step_release": max(
                (e["new_faces"] for e in d), default=0
            ),
            # THE RATCHET CHECK. Run 20260812-062701 peeled all 60 faces, the
            # clot began to travel, and then the count rose to 61 and 62:
            # growth was extending the footprint onto virgin wall, each new
            # face arriving fully bonded, and one bonded face re-anchors the
            # whole body. The footprint is now latched at the injury site, so
            # this must stay flat -- if it grows, the latch has failed.
            "footprint_faces_start": d[0]["injury_faces"] if d else None,
            "footprint_faces_end": d[-1]["injury_faces"] if d else None,
            "footprint_grew": (
                None if not d
                else max(e["injury_faces"] for e in d) > d[0]["injury_faces"]
            ),
            # Re-adhesion within the injury site is legitimate physics: the
            # exposed subendothelial matrix is still there. It is counted
            # separately from failures so it can never read as the peel
            # running backwards, and a large number here means the clot is
            # settling back down rather than leaving.
            "faces_reformed": sum(e["reformed_faces"] for e in d),
            "steps_with_reformation": sum(
                1 for e in d if e["reformed_faces"] > 0
            ),
            "peak_traction_pa": max((e["peak_traction_pa"] for e in d), default=None),
            "debonded_area_m2": max((e["debonded_area_m2"] for e in d), default=None),
            "released_cells_max": c["released_cells_max"],
            # On the ROW, not just buried in the child. It was only in the
            # child dict after run 20260811-051109, so the demonstration table
            # showed "did not move" with no way to see why.
            "body_velocity_m_s": c["body_velocity_max_m_s"],
            "free_stream_m_s": c["free_stream_m_s"],
            "body_velocity_free_stream_ratio": c["body_velocity_free_stream_ratio"],
            # A freed clot in the near-wall band of a Poiseuille profile should
            # travel at an appreciable fraction of the free stream. Far below
            # that means the ambient sample is being taken somewhere the flow
            # is suppressed -- which is exactly what happened.
            "body_velocity_plausible": (
                None if c["body_velocity_free_stream_ratio"] is None
                else c["body_velocity_free_stream_ratio"] > 0.1
            ),
            "transport": c["transport"],
            "progressive": bool(len(fired) >= 3),
            # NO TOLERANCE. This read `total >= initial_bonded - 1` and so
            # reported demo_full_release as having released everything in run
            # 20260811-004647, when one face of sixty survived and that one
            # face anchored all 780 clot cells: released_cells_max was 0 and
            # the clot never moved. The last face is not a rounding error, it
            # is the difference between an embolus and a flap.
            "released_everything": bool(
                initial_bonded and total >= initial_bonded
            ),
            # An arrested peel: nearly every face gone, still attached.
            "arrested": bool(
                initial_bonded
                and 0 < initial_bonded - total <= max(1, initial_bonded // 20)
            ),
            "cohesive_release_cells": c["cohesive_release_cells"],
        })
    return {"rows": rows, "excluded": excluded}


def deformation_table(children: dict, controls: dict) -> dict:
    """Does the clot body visibly deform, and can the number be believed?

    Two columns, and the second governs the first.  A strain computed from an
    unconverged skeleton is not a measurement, and at arterial load the
    skeleton fixed point is exactly where it is expected to struggle: it
    already truncates at venous load, and the finite-strain gate sat at the
    corrector cap under only 3x load.  So every row carries its convergence,
    and `trustworthy` is false when the solve did not settle -- rather than
    reporting a large strain and leaving the caveat to prose nobody reads.
    """
    clot_height = controls["_clot"]["height_m"]
    cell = controls["_geometry"]["h_m"]
    predicted = controls["deformation_axis"]["predicted_strain_percent"]

    rows, excluded = [], []
    for c in sorted(
        (c for c in children.values() if c["series"] == "deformation"),
        key=lambda c: c["wall_shear_rate_1_s"],
    ):
        if not c["admissible"]:
            excluded.append({"child_id": c["child_id"],
                             "why": c["bounds_reason"]})
            continue
        us = c["Us_max_m"]
        strain = us / clot_height if us is not None else None
        at_cap = c["skeleton"]["fraction_at_cap"]
        residual = c["skeleton"]["max_residual"]
        # The cap fraction is the honest discriminator: a fixed point that
        # spends most of its sweeps at the iteration limit did not converge,
        # it ran out of budget.
        # Divergence overrides the cap fraction. A run that tripped the guard
        # stopped early, so it may show FEW sweeps at the cap and look healthy
        # by that measure alone -- the opposite of the truth.
        converged = bool(
            at_cap is not None and at_cap < 0.05 and not c["skeleton_diverged"]
        )
        rows.append({
            "child_id": c["child_id"],
            "wall_shear_rate_1_s": c["wall_shear_rate_1_s"],
            # Same derivation as the required-strength table: the peak is in
            # the debond lines, not on the child.
            "traction_pa": max(
                (d["peak_traction_pa"] for d in c["_debonds"]), default=None
            ),
            "Us_max_m": us,
            "Us_max_um": us * 1e6 if us is not None else None,
            "strain_percent": strain * 100 if strain is not None else None,
            "displacement_cells": us / cell if us is not None else None,
            "predicted_strain_percent": predicted.get(
                str(int(c["wall_shear_rate_1_s"]))
            ),
            "skeleton_diverged": c["skeleton_diverged"],
            "skeleton_divergence": c["skeleton_divergence"],
            "skeleton_fraction_at_cap": at_cap,
            "skeleton_max_residual": residual,
            "skeleton_converged": converged,
            # Below about a third of a cell there is nothing to see in
            # ParaView without warping, whatever the strain says.
            "visible_without_warping": bool(
                us is not None and us / cell > 0.33
            ),
            "trustworthy": converged,
        })

    believable = [r for r in rows if r["trustworthy"]]
    visible = [r for r in believable if r["visible_without_warping"]]
    return {
        "rows": rows,
        "excluded": excluded,
        "clot_height_m": clot_height,
        "venous_reference_strain_percent": 0.53,
        "deforms_visibly": bool(visible),
        "lowest_visible_shear_1_s": (
            min(r["wall_shear_rate_1_s"] for r in visible) if visible else None
        ),
        "rows_discarded_for_convergence": len(rows) - len(believable),
        "how_to_read_this": (
            "strain_percent is only a measurement where skeleton_converged is "
            "true. Where it is false the fixed point hit its iteration cap and "
            "the displacement is whatever the solver reached, not what the "
            "model predicts. Compare strain_percent against "
            "predicted_strain_percent: the prediction is a LINEAR extrapolation "
            "from the measured venous point, so a measured value below it means "
            "either genuine nonlinearity or an unconverged solve, and the "
            "convergence column says which."
        ),
        "_this_series_is_arterial": controls["deformation_axis"][
            "_this_is_deliberately_outside_the_PIs_brief"
        ],
    }


def cohesive_tear_table(children: dict, controls: dict) -> dict:
    """The toughness this operating point needs before the BODY gives way.

    Deliberately the same device as required_strength_table: it does not
    assert a Gc, it brackets the one that tears and reports how far that is
    from the measured 5.547 J/m2.  The distance IS the result, and it says
    what source 33 says -- at 250 1/s this clot does not fragment.
    """
    # S21 (docket 20261005-1621 report 07): the cohesive_tear series and its axis
    # block left the controls with the retired children; without this the whole
    # assessment died on a KeyError and wrote no result.json.
    if "cohesive_tear_axis" not in controls:
        return {"retired": "S21: the cohesive_tear series is not in the kept set",
                "rows": [], "excluded": []}
    axis = controls["cohesive_tear_axis"]
    physiological = axis["physiological_Gc_j_m2"]
    rows, excluded = [], []

    for c in sorted(children.values(), key=lambda r: r["child_id"]):
        if c["series"] != "cohesive_tear":
            continue
        if not c["admissible"]:
            excluded.append({"child_id": c["child_id"],
                             "why": c["bounds_reason"] or "inadmissible"})
            continue
        t = c["tear"]
        gc = c["Gc_j_m2"]
        # THE CONTROL. If any wall face released, the experiment did not run:
        # the clot had a second way out and "it tore" no longer means the body
        # failed rather than the bond.
        released = max(
            (e["debonded_faces"] for e in c["_debonds"]), default=0
        )
        wall_held = released == 0
        rows.append({
            "child_id": c["child_id"],
            "Gc_j_m2": gc,
            "orders_below_physiological": (
                math.log10(physiological / gc) if gc and gc > 0 else None
            ),
            "wall_held": wall_held,
            "faces_released": released,
            "tore": t.get("tore"),
            "tear_cells": t.get("tear_cells"),
            "first_tear_time_s": t.get("first_tear_time_s"),
            "tear_y_mean_m": t.get("tear_y_mean_m"),
            "tear_height_fraction": t.get("tear_height_fraction"),
            "tear_at_wall": t.get("tear_at_wall"),
            "tear_through_body": t.get("tear_through_body"),
            "tear_mode": t.get("tear_mode"),
            "intact_cells_above_damage": t.get("intact_cells_above_damage"),
            "material_remaining_fraction": t.get("material_remaining_fraction"),
            "Dg_max": t.get("Dg_max"),
            "fragment_cells_max": c["released_cells_max"],
            "fragment_moved": c["transport"].get("moved_downstream"),
            "tear_reason": t.get("reason"),
            "H_bulk_max": c["damage_driver"]["H_bulk_max"],
            "H_interface_max": c["damage_driver"]["H_interface_max"],
            "interface_over_bulk": c["damage_driver"]["interface_over_bulk"],
            # Run 20260814-055141 measured 13.5. Above 1 the phase interface
            # is carrying more driving energy than the material, so the tear
            # is an artefact of the Us pin and nothing below it can be read.
            "driver_is_the_material": (
                None if c["damage_driver"]["interface_over_bulk"] is None
                else bool(c["damage_driver"]["interface_over_bulk"] < 1.0)
            ),
            # A tear is only the picture Dr Li asked for when all of these
            # hold -- including that the material, not the interface, drove it.
            "shows_what_the_PI_asked_for": bool(
                wall_held and t.get("tear_through_body")
                and c["released_cells_max"]
                and c["transport"].get("moved_downstream")
                and c["damage_driver"]["interface_over_bulk"] is not None
                and c["damage_driver"]["interface_over_bulk"] < 1.0
            ),
        })

    # A child that ablates has still "torn" for bracketing purposes -- the
    # material did give way at that toughness -- but it has NOT produced the
    # picture asked for, which is why the two are counted separately.
    ablating = [r for r in rows if r.get("tear_mode") == "surface_ablation"]
    tearing = [r for r in rows if r["tore"] and r["wall_held"]]
    holding = [r for r in rows if r["tore"] is False and r["wall_held"]]
    # The bracket: the LARGEST toughness that still tore, and the smallest
    # that held. A ladder where everything tears or nothing does has not
    # bracketed anything, and must say so instead of reporting an endpoint as
    # though it were the answer.
    required = max((r["Gc_j_m2"] for r in tearing), default=None)
    survived = min((r["Gc_j_m2"] for r in holding), default=None)
    bracketed = required is not None and survived is not None

    return {
        "rows": rows,
        "excluded": excluded,
        "wall_held_everywhere": all(r["wall_held"] for r in rows) if rows else None,
        "required_Gc_j_m2": required,
        "lowest_Gc_that_held_j_m2": survived,
        "bracketed": bracketed,
        "physiological_Gc_j_m2": physiological,
        "orders_below_physiological": (
            math.log10(physiological / required)
            if required and required > 0 else None
        ),
        "driver_artefact": (
            None if not rows or all(r["interface_over_bulk"] is None for r in rows)
            else max(r["interface_over_bulk"] for r in rows
                     if r["interface_over_bulk"] is not None)
        ),
        "driver_artefact_note": (
            "interface_over_bulk is the ratio of peak fracture driving energy "
            "in the phase interface to that in the bulk clot. Run "
            "20260814-055141 measured 13.5, meaning the damage was driven by "
            "the numerical pin that holds Us at zero in the fluid rather than "
            "by the material -- which is why every child eroded from the "
            "outside in. Below 1 the material drives it and the tear numbers "
            "mean something; at or above 1 they do not."
        ),
        "failure_mode": (
            "surface_ablation" if ablating and not any(
                r.get("tear_mode") == "through_body" for r in rows)
            else "through_body" if any(
                r.get("tear_mode") == "through_body" for r in rows)
            else "none"
        ),
        "ablation_note": (
            "Every child that gave way did so by SURFACE ABLATION: the damage "
            "band always reached the outer surface with no intact clot beyond "
            "it, so material was removed layer by layer rather than a piece "
            "being cut off. No fragment can form this way, which is why "
            "shows_what_the_PI_asked_for is false even where the ladder "
            "brackets. A localised tear needs somewhere for the damage to "
            "concentrate -- a neck, or the stiff-core/soft-shell contrast that "
            "source 33 relies on -- and a smooth homogeneous dome has neither."
            if ablating else None
        ),
        "how_to_read_this": axis["_what_this_sweep_is"],
        "what_this_does_not_show": axis["what_this_series_does_not_show"],
        "bracket_note": (
            None if bracketed else
            "The ladder did not bracket: every rung tore, or none did. The "
            "reported required_Gc is then an ENDPOINT of the sweep, not a "
            "measurement, and the ladder needs extending in that direction."
        ),
    }


def core_shell_table(children: dict, controls: dict) -> dict:
    """One child at source 33's operating point, with source 33's materials.

    THIS TABLE DOES NOT ASK WHETHER IT TORE, and that is deliberate.  The
    previous version of this series set the shell toughness nine orders below
    the measured value so that the AT2 criterion would fire inside the run, and
    then reported the tear as a result.  That is choosing the answer.  Toughness
    here is physiological in both regions -- which is also the honest reading of
    source 33, since Xu et al. have no toughness and no damage variable at all.

    What is measured instead is what Xu actually report: across every shear rate
    they run, THE CORE RETAINS ITS SHAPE WHILE THE SHELL IS PUSHED DOWNSTREAM
    into an elongated filament (their figure 5e-h).  So the row carries the two
    regions' displacement, deformation and shape change side by side, and the
    reader can see whether this model does the same thing.  Whether anything
    separates is an outcome, not a criterion.

    The checks that remain are on the EXPERIMENT, not on the result:

      wall_held               0 faces released. The bond is unbreakable, so any
                              release means the clot had a second way out.
      driver_is_the_material  interface_over_bulk < 1. Run 20260814-055141
                              measured 13.5 -- the damage was a numerical
                              boundary layer, not the material.
      permeability_is_real    the clot ran at the declared permeability rather
                              than at the numerical floor. Source 33's whole
                              result is permeability-driven and the floor
                              silently overrode k0p in every earlier run.
      shell_moved_more_than_core
                              the paper's signature. If the core moves as much
                              as the shell, this is whole-body sliding again --
                              the thing Dr Li rejected.
    """
    rows, excluded = [], []
    physiological = controls["material_fixed"]["Gc_j_m2"]

    for c in sorted(children.values(), key=lambda r: r["child_id"]):
        if c["series"] != "source33_replication":
            continue
        if not c["admissible"]:
            excluded.append({"child_id": c["child_id"],
                             "why": c["bounds_reason"] or "inadmissible"})
            continue
        t = c["tear"]
        released = max((e["debonded_faces"] for e in c["_debonds"]), default=0)
        wall_held = released == 0
        ratio = c["damage_driver"]["interface_over_bulk"]
        driver_ok = None if ratio is None else bool(ratio < 1.0)
        depth = t.get("first_tear_depth_cells")
        deep = None if depth is None else bool(depth >= 2.0)
        rows.append({
            "child_id": c["child_id"],
            "G_shell_pa": c["G_pa"],
            "G_core_pa": c.get("G_core_pa"),
            "Gc_shell_j_m2": c["Gc_j_m2"],
            "Gc_core_j_m2": c.get("Gc_core_j_m2"),
            "k0p_shell_m2": c["k0p_m2"],
            "k0p_core_m2": c.get("k0p_core_m2"),
            # The permeability the clot ACTUALLY ran with. Must span the
            # core and shell values; if it comes back pinned at the floor,
            # the model is the old kozenyCarman one and the contrast this
            # child rests on does not exist.
            "k_clot_min_m2": c["permeability"]["k_clot_min_m2"],
            "k_clot_max_m2": c["permeability"]["k_clot_max_m2"],
            "permeability_floored_fraction": c["permeability"][
                "floored_fraction_max"
            ],
            "H_core_max": c["damage_driver"]["H_core_max"],
            "H_shell_max": c["damage_driver"]["H_shell_max"],
            "core_over_shell": c["damage_driver"]["core_over_shell"],
            "growth_rate_multiplier": c.get("growth_rate_multiplier"),
            "orders_below_physiological": (
                math.log10(physiological / c["Gc_j_m2"])
                if c["Gc_j_m2"] and c["Gc_j_m2"] > 0 else None
            ),
            "wall_held": wall_held,
            "faces_released": released,
            "interface_over_bulk": ratio,
            "driver_is_the_material": driver_ok,
            "core_cells": t.get("core_cells"),
            "shell_cells": t.get("shell_cells"),
            "core_damaged_cells": t.get("core_damaged_cells"),
            "core_survived": t.get("core_survived"),
            "tore": t.get("tore"),
            "tear_cells": t.get("tear_cells"),
            "first_tear_time_s": t.get("first_tear_time_s"),
            "first_tear_depth_cells": depth,
            "tear_started_below_the_surface": deep,
            "tear_mode": t.get("tear_mode"),
            "intact_cells_above_damage": t.get("intact_cells_above_damage"),
            "material_remaining_fraction": t.get("material_remaining_fraction"),
            "growth_area_fraction": t.get("growth_area_fraction"),
            "grew": t.get("grew"),
            "fragment_cells_max": c["released_cells_max"],
            "fragment_moved": c["transport"].get("moved_downstream"),
            "Dg_max": t.get("Dg_max"),
            "skeleton_diverged": c["skeleton_diverged"],
            # --- SOURCE 33'S OWN MEASUREMENT: core shape vs shell shape -----
            "core_centroid_shift_m": t.get("core_centroid_shift_m"),
            "shell_centroid_shift_m": t.get("shell_centroid_shift_m"),
            "shell_over_core_shift": t.get("shell_over_core_shift"),
            "shell_moved_more_than_core": t.get("shell_moved_more_than_core"),
            "core_height_change_fraction": t.get("core_height_change_fraction"),
            "shell_height_change_fraction": t.get("shell_height_change_fraction"),
            "core_length_change_fraction": t.get("core_length_change_fraction"),
            "shell_length_change_fraction": t.get("shell_length_change_fraction"),
            "core_downstream_edge_shift_m": t.get("core_downstream_edge_shift_m"),
            "shell_downstream_edge_shift_m": t.get(
                "shell_downstream_edge_shift_m"
            ),
            "Us_core_max_m": t.get("Us_core_max_m"),
            "Us_shell_max_m": t.get("Us_shell_max_m"),
            # Strain goes as 1/G, so 40/4 = 10 is what the material contrast
            # predicts. A ratio near 1 means the two regions are not actually
            # different materials in the solve.
            "Us_shell_over_core": t.get("Us_shell_over_core"),
            "Us_shell_over_core_predicted": (
                c["G_core_pa"] / c["G_pa"]
                if c.get("G_core_pa") and c.get("G_pa") else None
            ),
            "permeability_is_real": (
                None if c["permeability"]["floored_fraction_max"] is None
                else bool(c["permeability"]["floored_fraction_max"] < 0.5)
            ),
        })

    def failed(key: str) -> list[str]:
        return [r["child_id"] for r in rows if r.get(key) is not True]

    return {
        "rows": rows,
        "excluded": excluded,
        "what_this_is": (
            "One child at source 33's operating point (Xu et al. 2017): "
            "gammaDot = 1000 1/s, a 40 Pa / 1e-14 m2 core under a 4 Pa / "
            "1e-11 m2 shell, growth running throughout, and PHYSIOLOGICAL "
            "toughness in both regions. Nothing in it is set to make anything "
            "fail. Xu's result is that the core retains its shape while the "
            "shell is pushed downstream into an elongated filament; the row "
            "reports the two regions' motion and deformation side by side so "
            "that comparison can be made rather than asserted."
        ),
        "the_experiment_is_valid_only_if": {
            "wall_held": failed("wall_held"),
            "driver_is_the_material": failed("driver_is_the_material"),
            "permeability_is_real": failed("permeability_is_real"),
        },
        "did_it_look_like_source_33": {
            "shell_moved_more_than_core": failed("shell_moved_more_than_core"),
            "note": (
                "This is a DESCRIPTION of the outcome, not a pass/fail. If the "
                "shell moved and the core did not, the model reproduced Xu's "
                "qualitative picture. If both moved together, it is whole-body "
                "sliding and the material contrast did not bite. If neither "
                "moved, the deformation is below what this geometry and "
                "duration can show."
            ),
        },
        "why_no_tear_criterion": (
            "AT2 fracture cannot fire at physiological toughness here and is "
            "not expected to: sigma_c = sqrt(Gc*E/ell) is about 15600 Pa "
            "against a peak clot traction of a few Pa. Dg_max is reported so "
            "that margin stays visible, and a Dg_max at zero is the correct "
            "answer rather than a missing capability. Source 33 has no "
            "toughness either -- their material leaves by being deformed and "
            "carried, which is the mechanism this child tests."
        ),
        # S21: source33_replication is a retired series (report 07); absent, not a crash.
        "what_is_guaranteed_and_what_is_not": (controls["series"].get(
            "source33_replication") or {}).get("_what_this_can_and_cannot_promise"),
    }


def cohesive_margin(children: dict, controls: dict) -> dict:
    """How far the clot is from tearing through its own body.

    The literature gives two embolisation routes: adhesive failure at the
    wall, and cohesive failure through the clot (shear-driven fragmentation
    and pressure-gradient fissuring).  This solver has both.  Only the
    adhesive one has ever fired, and this quantifies by how much -- so the
    answer is a measured margin rather than an untested assumption.

    Reported from the ALREADY-WRITTEN Dg damage field.  It adds no children,
    no cost, and above all no lowered Gc: manufacturing a tear by weakening
    the material is the mistake this reporting exists to avoid.
    """
    strength = controls["adhesive_strength"]
    sigma_c = strength["upper_bound_pa"]
    # S21: the gate block left the controls with the retired series (report 07); with
    # no peak on record the margin is None and the interpretation says so.
    peak = (controls.get("_established_by_the_gate") or {}).get("peak_bond_traction_pa")

    scored = [c for c in children.values() if c["admissible"]
              and c.get("Dg_max") is not None]
    dg_max = max((c["Dg_max"] for c in scored), default=None)
    worst = max(scored, key=lambda c: c["Dg_max"], default=None) if scored else None

    return {
        "cohesive_strength_pa": sigma_c,
        "peak_clot_traction_pa": peak,
        "margin_factor": sigma_c / peak if peak else None,
        "max_damage_anywhere": dg_max,
        "max_damage_child": worst["child_id"] if worst else None,
        "children_scored": len(scored),
        "cohesive_failure_occurred": bool(dg_max is not None and dg_max > 0.5),
        "interpretation": (
            "The wall bond fails long before the clot body does, by a factor of "
            f"{sigma_c / peak:.0f} in stress. Embolisation here is ADHESIVE, not "
            "cohesive. That is the expected answer at venous shear -- cohesive "
            "fragmentation is reported in the SIPA regime at 15000-28000/s, two "
            "orders above the 50-200/s run here -- so a damage field that stays "
            "at zero is positive evidence the model is in the right regime, not "
            "a missing capability."
            if peak else "no peak traction on record"
        ),
    }


def main() -> int:
    case_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    controls = json.loads(
        (case_root / "constant" / "mainCaseControls.json").read_text()
    )
    runs = case_root / "runs"
    matrix = json.loads((runs / "matrix.json").read_text())
    # The cohesive-tear series carries its OWN geometry -- a finer mesh and a
    # smaller dome -- so "the geometry" is no longer a property of the case.
    # growth_table and deformation_table both divide by h and by the clot
    # height, and taking those from matrix[0] would silently score the venous
    # series against the tear mesh whenever the run order or the filter put a
    # tear child first.
    # source33_replication carries the same overridden geometry, so it is excluded
    # from the reference for the same reason.
    #
    # DERIVED, NOT LISTED. This was a hard-coded pair of names, and adding the
    # two stage-3 series broke it immediately: they declare their own
    # geometry_override AND sort first, so stage3_xu_eul became the reference
    # and every growth child was refused for "differing from the reference".
    # The condition that actually matters is "does this series override the
    # geometry", which the controls file already states, so it is read from
    # there. A new series with its own mesh can no longer silently become the
    # yardstick for the venous tables.
    own_geometry = {
        name for name, spec in controls["series"].items()
        if spec.get("geometry_override")
    }
    reference = next(
        (s for s in matrix if s["series"] not in own_geometry),
        matrix[0],
    )
    controls["_geometry"] = reference["geometry"]
    controls["_clot"] = reference["clot"]
    for s in matrix:
        if s["series"] in ("growth", "deformation") and (
            s["geometry"]["h_m"] != reference["geometry"]["h_m"]
            or s["clot"]["height_m"] != reference["clot"]["height_m"]
        ):
            raise ValueError(
                f"{s['child_id']}: its geometry differs from the reference "
                f"used to score the growth and deformation tables"
            )

    children = {
        s["child_id"]: assess_child(runs / s["child_id"], s, controls)
        for s in matrix
    }

    build_log = case_root / "log.Allwmake"
    build_text = build_log.read_text(errors="replace") if build_log.is_file() else ""
    library_compiled = bool(build_text) and not COMPILE_FAILURE.search(build_text)

    admissible = [c for c in children.values() if c["admissible"]]
    inadmissible = [
        {"child_id": c["child_id"], "series": c["series"],
         "completed": c["completed"], "bounded": c["bounded"],
         "status": c["status"],
         "why": (f"STOPPED_BY_WATCHER (watcher verdict "
                 f"{c['watcher_verdict']['verdict']}: {c['watcher_verdict']['failed_clauses']})"
                 if c["status"] == "STOPPED_BY_WATCHER"
                 else c["bounds_reason"] or f"exit code {c['exit_code']}")}
        for c in children.values() if not c["admissible"]
    ]

    result = {
        "case": "AM2020_planar_mural_detachment",
        "what_this_is": controls["_what_this_is_and_is_not"],
        "library_compiled": library_compiled,
        "children_total": len(children),
        "children_admissible": len(admissible),
        "inadmissible": inadmissible,
        "growth_vs_flow": growth_table(children, controls),
        "required_strength": required_strength_table(children),
        "deformation": deformation_table(children, controls),
        "demonstration": demonstration_table(children),
        "cohesive_tear": cohesive_tear_table(children, controls),
        "source33_replication": core_shell_table(children, controls),
        "xu2017_fig5": xu2017_table(children),
        "cohesive_release_anywhere": sum(
            c["cohesive_release_cells"] for c in children.values()
        ),
        "cohesive_margin": cohesive_margin(children, controls),
        "numerical_limitations": controls["limitations_to_state_to_the_PI"],
        "realised_cost_seconds": {
            c["child_id"]: c["wall_seconds"] for c in children.values()
        },
        # How many children shared the cores while that wall time was measured.
        # 1 (or null, on a log written before this was recorded) means the
        # number is this child's own cost and is comparable across runs.
        # Anything higher means it is contended and is NOT.
        "realised_cost_concurrency": {
            c["child_id"]: c.get("concurrency") for c in children.values()
        },
    }
    for c in children.values():
        c.pop("_debonds", None)
        c.pop("_volumes", None)
    result["children"] = children

    out = case_root / "postProcessing" / "mainCase"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "children_admissible": f"{len(admissible)}/{len(children)}",
        "growth_exponent": result["growth_vs_flow"]["flow_dependence"]["exponent"],
        "required_strength_rows": len(result["required_strength"]["rows"]),
        "demonstration": [
            {k: r[k] for k in ("child_id", "released_fraction", "progressive")}
            for r in result["demonstration"]["rows"]
        ],
    }, indent=2))

    # Exit non-zero only if the case cannot be reported on at all. A child that
    # failed is named in the table; that is a result, not a runner error.
    return 0 if (library_compiled and admissible) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"assess_case.py: {error}")
