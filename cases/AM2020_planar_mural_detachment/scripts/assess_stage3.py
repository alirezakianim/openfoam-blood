#!/usr/bin/env python3
"""Stage 3 of research/designs/AM2020_eulerian_solid_plan.md.

Four children: {Xu scale, Dr Li's dome} x {eulerianSolid on, off}. Three
questions, and they are separable:

1. COST. Is the Eulerian formulation at least 3x cheaper per step? Measured as
   an ON/OFF ratio on the SAME machine, not against the historical 30.3 s/step
   from run 20260816-031412 -- that number came from one Colab VM, and
   VM-to-VM variation would swamp a 3x threshold read across sessions. The
   historical figure is printed as a cross-check and gates nothing.

2. HEALTH. No SPD loss, and interface_over_bulk below 1.

   interface_over_bulk is where the `beRelax = q^2` hypothesis finally gets
   tested. The stage-2 slab gate could not test it: that gate initialises a
   SHARP slab, so its interface bin was empty in every child at every step and
   the zeros it reported were not evidence. These children have a real tanh
   interface. If the ratio exceeds 1, strain is accumulating at the diffuse
   interface faster than in the clot the interface belongs to, and
   re-weighting beRelax is the fix.

3. DEFORMATION REGIME, reported and NOT gated. Peak shell stretch on each
   geometry. Plan section 6(a) predicted in advance that Dr Li's smaller dome
   carries lower traction, so it deforms less and is less likely to pinch off.
   This is the number that lets stage 4's outcome be explained with a
   measurement instead of an excuse. There is no threshold because nobody
   knows what stretch pinch-off needs in this model -- inventing one here
   would make stage 4's verdict a function of where it was put.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path


CASE = Path(__file__).resolve().parents[1]
RUNS = CASE / "runs"

COST = re.compile(
    r"PHASE_CLOT_COST\s+time=(\S+)\s+step=(\S+)\s+skeleton=(\S+)"
)
EULER = re.compile(
    r"PHASE_CLOT_EULERIAN\s+time=\S+\s+beMinMinor=(\S+)\s+beMaxStretch=(\S+)"
    r"\s+beDetError=(\S+)(?:\s+beDetErrorPost=(\S+))?"
    r"\s+beSweepChange=(\S+)"
)
TRACE = re.compile(
    r"PHASE_CLOT_TRACE\s+time=\S+\s+trBulk=(\S+)\s+trInterface=(\S+)\s+"
    r"trBlood=(\S+)\s+traceFracBulk=(\S+)\s+traceFracInterface=(\S+)\s+"
    r"traceFracBlood=(\S+)\s+divPhiMax=(\S+)"
    r"(?:\s+trBulkFluxConsistent=(\S+))?"
    # THE VOLUME-WEIGHTED COMPANIONS, all optional so every pre-2026-08-23 log
    # still parses. Every field above is a MAXIMUM over cells, and a maximum
    # cannot distinguish one bad cell from a bad field -- traceFracBlood read
    # 1.41352 against the 2-D ceiling 1.41421 in run 20260823-052707, in the
    # blood, where no elastic force exists, and that one number has been
    # carried through several runs as evidence of "98% spurious dilatation".
    # These are what decide whether it was ever meaningful.
    r"(?:\s+trBulkVw=(\S+)\s+trInterfaceVw=(\S+)\s+trBloodVw=(\S+)"
    r"\s+fracBulkRms=(\S+)\s+fracInterfaceRms=(\S+)\s+fracBloodRms=(\S+)"
    r"\s+volFracBulkOver1=(\S+)\s+volFracInterfaceOver1=(\S+)"
    r"\s+volFracBloodOver1=(\S+)\s+yWorstBlood=(\S+))?"
)
# THE TERM-BY-TERM MOMENTUM BALANCE OVER THE CLOT. Added after ELEVEN
# named hypotheses were tested and none held the clot -- this stops the
# question being "which mechanism next" and makes it "which force is
# missing or wrong-signed", which is answerable.
#
# PARSED BY KEY, NOT BY POSITION, AND DELIBERATELY SO. The positional
# version cost real work twice: once when the TRACE regex gained ten groups
# and the results dict did not, so run 20260823-061943 had to be read out
# of raw logs by hand; and again here, because adding clotBeXY in the
# middle of the line would have silently shifted clotVolume into
# clotUxbar's slot with no parse error anywhere. A key scan cannot do that.
# Fields may be added to PHASE_CLOT_MOMENTUM in any order from now on, and
# logs written before any of them still parse -- a missing key reads as
# nan, which is what it is.
MOMENTUM_LINE = re.compile(r"PHASE_CLOT_MOMENTUM\s+(.+)")
BETERM_LINE = re.compile(r"PHASE_CLOT_BETERM\s+(.+)")
DEBOND_LINE = re.compile(r"PHASE_CLOT_DEBOND\s+(.+)")
WOUND_LINE = re.compile(r"PHASE_CLOT_WOUND\s+(.+)")
RELEASED_LINE = re.compile(r"PHASE_CLOT_RELEASED\s+(.+)")
BONDHOLD_LINE = re.compile(r"PHASE_CLOT_BONDHOLD\s+(.+)")
_KV = re.compile(r"(\w+)=(\S+)")


def _momentum_rows(text: str) -> "list[dict[str, float]]":
    """MOMENTUM rows, TRUNCATED AT THE FIRST BE_SPD_LOST.

    RUN 20260823-204058 PAID FOR THIS TRUNCATION AND THE GATE INVERTED ITS
    ANSWER WITHOUT IT. `be` is a left Cauchy-Green tensor; once it is not
    positive-definite it is not one, and `dev(be)` is not a stress. Reading a
    strain statistic, or a force balance, across that point is reading numbers
    produced by an object that no longer means what its name says.

    Concretely: `sat_G40` lost SPD at t = 3.044e-3 s of a 5e-3 s window. Over
    the FULL window its clot-mean be_xy "moved 52.5% across the final quarter"
    and the plateau gate failed it. Over the VALID window the same child moves
    0.33% and passes an order of magnitude inside the limit. The gate reported
    the opposite of the truth on the one child that answered the question.

    The same applies to the final-value force exports below -- accElastic and
    friends taken from the last line of a run that broke at 15% of its length
    describe the breakage, not the model.
    """
    end = len(text)
    spd = SPD_LOST.search(text)
    if spd is not None:
        end = spd.start()
    rows = []
    for match in MOMENTUM_LINE.finditer(text):
        if match.start() >= end:
            break
        row = {}
        for key, val in _KV.findall(match.group(1)):
            try:
                row[key] = float(val)
            except ValueError:
                pass          # runTime.name() can be non-numeric; skip it
        rows.append(row)
    return rows


def _beterm_rows(text: str) -> "list[dict[str, float]]":
    """PHASE_CLOT_BETERM rows. NOT truncated at SPD loss, deliberately.

    The whole point of this diagnostic is to watch the cell go bad, so the
    steps at and after the loss are the ones worth seeing. The momentum
    truncation exists because a force balance computed from an indefinite be is
    meaningless; here the indefiniteness IS the measurement.
    """
    rows = []
    for match in BETERM_LINE.finditer(text):
        row = {}
        for key, val in _KV.findall(match.group(1)):
            try:
                row[key] = float(val)
            except ValueError:
                pass
        rows.append(row)
    return rows


def _beterm_at_worst(rows: "list[dict[str, float]]") -> "dict[str, float]":
    """The step with the smallest lminWorst -- the deepest point of the failure.

    Reported alongside the final step because the two differ: the cluster can
    recover after the loss as the bad cell is smeared, which is exactly what
    run 20260823-204058 showed (min eigenvalue -0.0648 at t=3.25e-3, back to
    +0.0837 at t=3.5e-3). Reading only the final step would miss the event.
    """
    nan = float("nan")
    have = [r for r in rows if "lminWorst" in r]
    if not have:
        return {}
    return min(have, key=lambda r: r["lminWorst"])


def _dominant_det_term(row: "dict[str, float]") -> str:
    """Which term carries the most negative det rate, or the gap if it wins."""
    if not row:
        return "no BETERM lines"
    terms = {
        "advection": row.get("detRateAdvection", 0.0),
        "nonCons": row.get("detRateNonCons", 0.0),
        "stretch": row.get("detRateStretch", 0.0),
        "relax": row.get("detRateRelax", 0.0),
    }
    gap = row.get("detRateGap", 0.0)
    worst_name, worst_val = min(terms.items(), key=lambda kv: kv[1])
    if abs(gap) > abs(worst_val):
        return f"GAP ({gap:.4g}) exceeds every term; largest term {worst_name} {worst_val:.4g}"
    return f"{worst_name} ({worst_val:.4g}); gap {gap:.4g}"


def _debond_rows(text: str) -> "list[dict[str, object]]":
    rows = []
    for match in DEBOND_LINE.finditer(text):
        row = {}
        for key, val in _KV.findall(match.group(1)):
            try:
                row[key] = float(val)
            except ValueError:
                row[key] = val
        rows.append(row)
    return rows


def _traction_plateau(rows) -> float:
    """Relative change in peakTraction across the final quarter of the run."""
    have = [
        (r["time"], r["peakTraction"])
        for r in rows
        if isinstance(r.get("time"), float)
        and isinstance(r.get("peakTraction"), float)
    ]
    if len(have) < 8:
        return float("nan")
    t0, t1 = have[0][0], have[-1][0]
    cut = t1 - 0.25 * (t1 - t0)
    late = [v for t, v in have if t >= cut]
    if len(late) < 2 or abs(late[-1]) < 1e-30:
        return float("nan")
    # SIGNED, and it matters. An earlier version returned the absolute change,
    # so run 20260824-220555 -- whose traction PEAKED at 150.79 Pa and then
    # COLLAPSED to 0.17 Pa as the clot lifted off the wall -- was reported as
    # "still climbing 637%". The gate was right that the run had not saturated
    # and wrong about which way it went, which is exactly the kind of message
    # that sends the next reader after the wrong thing.
    return (late[-1] - late[0]) / abs(late[-1])


def _kv_rows(pattern, text: str) -> "list[dict[str, object]]":
    rows = []
    for match in pattern.finditer(text):
        row = {}
        for key, val in _KV.findall(match.group(1)):
            try:
                row[key] = float(val)
            except ValueError:
                row[key] = val
        rows.append(row)
    return rows


def _anchor_summary(text: str) -> "dict[str, object]":
    """Does the clot stay on the wound, and does anything free sit on a bond?

    THE QUESTION THIS ANSWERS HAD NO INSTRUMENT UNTIL 2026-08-24. Run
    20260824-220555 lost 99.5% of the clot standing over its injury -- 1093
    cells to 6 -- with every one of its forty bonds intact, and not one log
    line said so; it had to be reconstructed from written fields in a later
    session. Absence of the diagnostic is reported as None and gated
    separately from a measured value, the same way sigma_found is, because a
    run that predates the instrument must not read as a run that passed.
    """
    nan = float("nan")
    wound = _kv_rows(WOUND_LINE, text)
    rel = _kv_rows(RELEASED_LINE, text)
    hold = _kv_rows(BONDHOLD_LINE, text)

    out = {
        "has_wound": bool(wound),
        "coverage_initial": nan,
        "coverage_final": nan,
        "coverage_min": nan,
        "max_height_initial": nan,
        "max_height_final": nan,
        "cells_over_injury_initial": nan,
        "cells_over_injury_final": nan,
        "wound_x_span_final": nan,
        "wall_clot_x_max_final": nan,
        # The release side.
        "has_release_diag": bool(rel),
        "released_on_bond_max": nan,
        "release_first_time": nan,
        "release_biggest_step_jump": nan,
        "released_final": nan,
        "support_limit": nan,
        # The hold side.
        "has_bondhold": bool(hold),
        "held_cells_max": nan,
        "hold_dilation": nan,
        # ADDED 2026-08-26 with the taper and the core exemption. None (absent)
        # is distinguished from a measured value throughout: a run predating
        # the instrument must never read as a run that passed it.
        "hold_taper": nan,
        "held_weight_max": nan,
        "has_core_exempt_diag": False,
        "core_exempted_max": nan,
    }

    if wound:
        cov = [r["coverage"] for r in wound if isinstance(r.get("coverage"), float)]
        if cov:
            out["coverage_initial"] = cov[0]
            out["coverage_final"] = cov[-1]
            out["coverage_min"] = min(cov)
        mh = [r["maxHeight"] for r in wound if isinstance(r.get("maxHeight"), float)]
        if mh:
            out["max_height_initial"] = mh[0]
            out["max_height_final"] = mh[-1]
        co = [
            r["cellsOverInjury"]
            for r in wound
            if isinstance(r.get("cellsOverInjury"), float)
        ]
        if co:
            out["cells_over_injury_initial"] = co[0]
            out["cells_over_injury_final"] = co[-1]
        last = wound[-1]
        if isinstance(last.get("xMax"), float) and isinstance(last.get("xMin"), float):
            out["wound_x_span_final"] = last["xMax"] - last["xMin"]
        if isinstance(last.get("wallClotXMax"), float):
            out["wall_clot_x_max_final"] = last["wallClotXMax"]

    if rel:
        rob = [
            r["releasedOnBond"]
            for r in rel
            if isinstance(r.get("releasedOnBond"), float)
        ]
        if rob:
            out["released_on_bond_max"] = max(rob)
        cells = [
            (r.get("time"), r.get("cells"))
            for r in rel
            if isinstance(r.get("cells"), float)
            and isinstance(r.get("time"), float)
        ]
        if cells:
            out["released_final"] = cells[-1][1]
            first = [t for t, n in cells if n > 0]
            if first:
                out["release_first_time"] = first[0]
            # THE STEP FUNCTION. A physical detachment is progressive; a
            # topology test on a level set is not. Run 20260824-220555 went
            # from 0 to 1278 released cells in ONE step, which is the
            # signature that convicted the support test, so it is measured
            # rather than eyeballed off a plot.
            jump = 0.0
            for i in range(1, len(cells)):
                jump = max(jump, cells[i][1] - cells[i - 1][1])
            out["release_biggest_step_jump"] = jump
        ce = [
            r["coreExempted"] for r in rel
            if isinstance(r.get("coreExempted"), float)
        ]
        if ce:
            out["has_core_exempt_diag"] = True
            out["core_exempted_max"] = max(ce)
        sl = [
            r["supportLimit"]
            for r in rel
            if isinstance(r.get("supportLimit"), float)
        ]
        if sl:
            out["support_limit"] = sl[-1]

    if hold:
        hc = [r["heldCells"] for r in hold if isinstance(r.get("heldCells"), float)]
        if hc:
            out["held_cells_max"] = max(hc)
        hd = [r["dilation"] for r in hold if isinstance(r.get("dilation"), float)]
        if hd:
            out["hold_dilation"] = hd[-1]
        tp = [r["taper"] for r in hold if isinstance(r.get("taper"), float)]
        if tp:
            out["hold_taper"] = tp[-1]
        hw = [
            r["heldWeight"] for r in hold
            if isinstance(r.get("heldWeight"), float)
        ]
        if hw:
            out["held_weight_max"] = max(hw)
    return out


def _departure_before_debond(text: str) -> "dict[str, object]":
    """Did the clot leave before the bond broke?

    THE ORDER OF EVENTS IS THE RESULT for any child with a breakable bond. If
    coverage collapses or material is released while debondedFaces is still 0,
    the clot left for a reason that is not adhesive failure, and a run that
    ends with a detached clot would look like a success while demonstrating
    the opposite.
    """
    wound = _kv_rows(WOUND_LINE, text)
    deb = _kv_rows(DEBOND_LINE, text)
    rel = _kv_rows(RELEASED_LINE, text)
    nan = float("nan")
    out = {
        "first_debond_time": nan,
        "coverage_half_time": nan,
        "first_release_time": nan,
        "left_before_break": None,
    }
    first_deb = [
        r["time"]
        for r in deb
        if isinstance(r.get("time"), float)
        and isinstance(r.get("debondedFaces"), float)
        and r["debondedFaces"] > 0
    ]
    if first_deb:
        out["first_debond_time"] = first_deb[0]
    # DEPARTURE MEANS COVERAGE LOST, NOT COVERAGE NOT YET GAINED.
    #
    # This used to take the first sample with coverage < 0.5 anywhere in the
    # run, INCLUDING the initial condition, and run 20260827-103135 showed what
    # that costs. The clot is seeded as a body resting on the wound, not as a
    # film already spread across it: at the first step it touches 56 of 350
    # injury faces, coverage 0.16, and it reaches 1.0 by t = 4.9e-5 s and stays
    # there. Exactly ONE sample out of 11736 is below 0.5 and it is the first.
    # The gate read that single startup sample as "the clot departed", and
    # since no debond ever occurred it reported THE CLOT LEFT BEFORE THE BOND
    # BROKE for gr_p15 -- the one child whose order-of-events result the series
    # existed to produce. It would fire the same way for every breakable child
    # ever run.
    #
    # A footprint has to be ESTABLISHED before it can be LOST. So the search
    # starts only after coverage has first reached 0.5, and a run that never
    # gets there has no departure time rather than a spurious one at t=0.
    established = next(
        (
            i
            for i, r in enumerate(wound)
            if isinstance(r.get("coverage"), float) and r["coverage"] >= 0.5
        ),
        None,
    )
    half = (
        [
            r["time"]
            for r in wound[established:]
            if isinstance(r.get("time"), float)
            and isinstance(r.get("coverage"), float)
            and r["coverage"] < 0.5
        ]
        if established is not None
        else []
    )
    if half:
        out["coverage_half_time"] = half[0]
    # A QUARTER OF THE BODY, NOT ONE CELL.
    #
    # The first version of this used cells > 0 and the dry run caught it: a
    # single freed cell anywhere in the domain fired on the very first step,
    # so a run where the bond loaded, broke, and only THEN shed its clot was
    # reported as having left before the bond broke. Shedding a few cells is
    # ordinary; a quarter of the body coming free is a departure. The
    # threshold is a FRACTION of the measured clot rather than a count, so it
    # does not have to be re-tuned when the clot grows or the mesh changes.
    firstrel = [
        r["time"]
        for r in rel
        if isinstance(r.get("time"), float)
        and isinstance(r.get("cells"), float)
        and isinstance(r.get("clotCells"), float)
        and r["clotCells"] > 0
        and r["cells"] > RELEASE_DEPARTURE_FRACTION * r["clotCells"]
    ]
    if firstrel:
        out["first_release_time"] = firstrel[0]

    # Undecidable unless the wound instrument is present.
    if not wound:
        return out
    departure = min(
        [t for t in (out["coverage_half_time"], out["first_release_time"])
         if t == t],
        default=float("nan"),
    )
    if departure != departure:
        out["left_before_break"] = False
        return out
    if out["first_debond_time"] != out["first_debond_time"]:
        # Never debonded, but it left.
        out["left_before_break"] = True
        return out
    out["left_before_break"] = departure < out["first_debond_time"]
    return out


def _debond_summary(text: str) -> "dict[str, object]":
    """The bond's own account of itself.

    sigma_found is the important one. A run where the adhesion condition
    cannot see a stress reports peakTractionAny 0 and utilisation 0 -- exactly
    what an unloaded bond reports -- and run 20260824-033102 read that for
    20 ms while the criterion was inert. Older logs predate the field, so its
    absence is reported as None and gated separately from a measured 0.
    """
    rows = _debond_rows(text)
    nan = float("nan")
    if not rows:
        return {
            "has_debond": False,
            "sigma_found": None,
            "sigma_field": None,
            "peak_traction_any": nan,
            "peak_traction": nan,
            "peak_traction_normal": nan,
            "peak_traction_shear": nan,
            "peak_traction_rel_change": nan,
            "peak_traction_final": nan,
            "bonded_clot_faces_final": nan,
            "bonded_clot_faces_initial": nan,
            "debonded_faces_final": nan,
            "adhesive_strength": nan,
        }
    last = rows[-1]
    sf = last.get("sigmaFound")
    return {
        "has_debond": True,
        "sigma_found": (None if sf is None else int(sf)),
        "sigma_field": last.get("sigmaField"),
        "peak_traction_any": max(
            (r.get("peakTractionAny", 0.0) for r in rows), default=nan
        ),
        "peak_traction": max((r.get("peakTraction", 0.0) for r in rows), default=nan),
        # THE FINAL VALUE, WHICH IS NOT THE MAXIMUM. Both keys above are
        # maxima over the whole run, and the saturation message used to print
        # peak_traction_any in its "ending at" slot -- so it reported a RUN
        # MAXIMUM labelled as the final value. It read correctly only while
        # the traction rose monotonically, which every child did until
        # ad_d12 (run 20260827-051619) peaked at 22.0406 Pa at 9.5 ms and
        # decayed to 14.0514 by 24 ms -- and was reported as "COLLAPSED
        # 14.9% ... ending at 22.0406 Pa", i.e. the collapse and the value
        # contradicted each other in the same sentence.
        "peak_traction_final": last.get("peakTractionAny", nan),
        "peak_traction_normal": max(
            (r.get("peakTractionNormal", 0.0) for r in rows), default=nan
        ),
        "peak_traction_shear": max(
            (r.get("peakTractionShear", 0.0) for r in rows), default=nan
        ),
        # Is the traction still climbing at the end? Run 20260824-195730
        # reported 148.93 Pa as "the required adhesive strength" when it was
        # rising 17.8% across the final quarter -- a LOWER BOUND read as a
        # value. Reported so the distinction is on the record.
        "peak_traction_rel_change": _traction_plateau(rows),
        "bonded_clot_faces_final": last.get("bondedClotFaces", nan),
        "bonded_clot_faces_initial": rows[0].get("bondedClotFaces", nan),
        "debonded_faces_final": last.get("debondedFaces", nan),
        "adhesive_strength": last.get("strength", nan),
    }


def _mom(rows: "list[dict[str, float]]", key: str) -> float:
    """Last MOMENTUM line's value for `key`, or nan if it was never printed."""
    return rows[-1].get(key, float("nan")) if rows else float("nan")


def _plateau(rows: "list[dict[str, float]]", key: str) -> "dict[str, float]":
    """Relative change of `key` across the final quarter of the window.

    THE PASS CRITERION FOR THE SATURATION SERIES, and the reason it is not
    lockRatio. Run 20260823-102340's offline analysis showed the clot is
    anchored and SHEARING -- Ux spans a factor of 19 from wall row to crest --
    so a volume-mean velocity ratio cannot separate "held and deforming" from
    "carried along", and twelve hypotheses died against it. What is driven is
    the shear strain, so what must be shown to settle is the shear strain.

    Normalised by the LATE value, not the peak or the range: the question is
    whether the quantity has stopped moving relative to where it now sits.
    """
    have = [(r["time"], r[key]) for r in rows if "time" in r and key in r]
    nan = float("nan")
    if len(have) < 8:
        return {"start": nan, "end": nan, "rel_change": nan, "n": len(have)}
    t0, t1 = have[0][0], have[-1][0]
    cut = t1 - 0.25 * (t1 - t0)
    late = [v for t, v in have if t >= cut]
    if len(late) < 2:
        return {"start": nan, "end": nan, "rel_change": nan, "n": len(have)}
    a, b = late[0], late[-1]
    denom = max(abs(b), 1e-30)
    return {
        "start": a,
        "end": b,
        "rel_change": abs(b - a) / denom,
        "n": len(have),
    }
PHASE = re.compile(
    r"PHASE_CLOT_BE_PHASE\s+time=\S+\s+stretchBulk=(\S+)\s+"
    r"stretchInterface=(\S+)\s+stretchBlood=(\S+)"
)
SPD_LOST = re.compile(r"BE_SPD_LOST")
# The last three groups are OPTIONAL so logs written before 2026-08-22 still
# parse. A required group would have turned every archived log into "no
# PHASE_CLOT_PLATELET lines", which reads as a solver fault rather than an
# assessor that moved on.
PLATELET = re.compile(
    r"PHASE_CLOT_PLATELET\s+time=\S+\s+wMax=(\S+)\s+wMin=(\S+)\s+"
    r"VnMagMax=(\S+)\s+wOverBound=(\S+)"
    r"(?:\s+WpDevMax=(\S+)\s+qMin=(\S+)\s+wDevBlood=(\S+))?"
)
LOCK = re.compile(
    r"PHASE_CLOT_LOCK\s+time=\S+\s+clotUMean=(\S+)\s+clotUMax=(\S+)\s+"
    r"bloodUMean=(\S+)\s+lockRatio=(\S+)\s+clotCells=(\S+)"
)

# ADDED 2026-09-01 with the harmonic-diffusivity fix. These three tags all
# begin with PHASE_CLOT_PLATELET but are followed by "_", never whitespace, so
# the positional PLATELET regex above cannot match them -- verified.
#
# plateletFo is the explicit stability number of the platelet diffusion term,
# A_P = (dt/V) sum_f gamma_f |Sf| deltaCoeff_f / qEff_P. The explicit scheme is
# unstable at 2. Run 20260901-110348 child g20 died at A_P ~ 3.2 and NOTHING
# else in the log moved; this is the instrument that was missing.
PLATELET_FO = re.compile(r"PHASE_CLOT_PLATELET_FO\s+time=(\S+)\s+plateletFo=(\S+)")
# The discrete maximum principle on Wp. An OVERSHOOT is the sharp signal --
# nothing else in that equation can lift a cell above all of its neighbours.
# WpMin/WpMax are OPTIONAL so the 20260901-200203 logs, written before they
# existed, still parse.
PLATELET_LED = re.compile(
    r"PHASE_CLOT_PLATELET_LED\s+time=(\S+)"
    r"(?:\s+WpMin=(\S+)\s+WpMax=(\S+))?"
    r"\s+WpOvershoot=(\S+)\s+"
    r"WpUndershoot=(\S+)\s+violCells=(\S+)\s+overX=(\S+)\s+overY=(\S+)"
)
PLATELET_LIM = re.compile(
    r"PHASE_CLOT_PLATELET_LIM\s+time=\S+\s+facesUpwinded=(\S+)"
    r"(?:\s+facesUpwindedAtInterface=(\S+))?"
)
# THE LABEL DRIFT INSTRUMENT, 2026-09-02. shell_ is advected by its OWN vanLeer
# limiter while c now uses the shared more-upwind weights, so the material label
# and the material it labels smear at different rates. Reported, NEVER gated:
# there is no measurement yet from which to derive a threshold.
LABEL_DRIFT = re.compile(
    r"PHASE_CLOT_LABEL\s+time=(\S+)\s+labelTotal=(\S+)\s+labelInBlood=(\S+)"
)

# ADDED 2026-09-02 with the reaction growth closure.
#
# THE STOICHIOMETRY INSTRUMENT. Capek (2020) eq. (4.6) makes growth and
# platelet consumption the SAME process: clot volume appears at (alpha/D) times
# the rate platelets are absorbed. stoichNorm is that ratio divided by its
# target, so it is 1 when the model conserves platelets into clot and it is
# NOT 1 when the clot is growing at a rate unrelated to what it eats.
#
# Measured on run 20260901-200203 under the OLD gradient closure: 0.034 to
# 0.10, drifting by 3x over 30 ms. That is the defect this instrument exists
# to make impossible to miss again.
#
# VnClosureResidual is the machine-precision identity check: is the Vn the
# phase equation actually used equal to (alpha/D)*kEff*w? Under "reaction" it
# is ~1e-16; under "gradient" it is O(1).
GROWTH_STOICH = re.compile(
    r"PHASE_CLOT_GROWTH_STOICH\s+time=(\S+)\s+closure=(\S+)\s+"
    r"depletion=(\S+)(?:\s+content=(\S+))?\s+clotGrowth=(\S+)\s+plateletUptake=(\S+)\s+"
    r"stoichRatio=(\S+)\s+stoichTarget=(\S+)\s+stoichNorm=(\S+)"
    r"(?:\s+VnClosureResidual=(\S+)\s+VnClosureRef=(\S+))?"
)
# THE REGIME INSTRUMENT. Da = kEff*h_n/D. Reported, never gated: it says
# whether the reaction closure's declared limit has been reached, and that is
# a fact about the mesh and the shear, not a defect.
# WHERE THE GROWTH COMES FROM. Alireza asked whether it is legitimate to drive
# growth with `w`, given that `w` -- unlike `Wp` -- does not go to zero inside
# the clot and so does not represent the clot geometry. It is: `c` carries the
# geometry and `w` is an intensive property of the blood phase. But that is an
# argument, so this measures it instead.
GROWTH_SUPPORT = re.compile(
    r"PHASE_CLOT_GROWTH_SUPPORT\s+time=(\S+)\s+growthBelowQ005=(\S+)\s+"
    r"growthBelowQ015=(\S+)\s+growthMeanQ=(\S+)"
)
# THE PHASE-FIELD BOUNDS. c must lie in [-1, 1] because q = 0.5(c+1) is a
# volume fraction. Nothing reported this until 2026-09-02; a violation would
# have been absorbed silently by max(q, qFloor) and by chi = max(1-q, 0).
PHASE_BOUNDS = re.compile(
    r"PHASE_CLOT_PHASE_BOUNDS\s+time=(\S+)\s+cMin=(\S+)\s+cMax=(\S+)\s+"
    r"cUnder=(\S+)\s+cOver=(\S+)"
)
DAMKOHLER = re.compile(
    r"PHASE_CLOT_DAMKOHLER\s+time=(\S+)\s+DaMax=(\S+)\s+"
    r"DaInterfaceMax=(\S+)\s+layerCellsMin=(\S+)\s+interfaceCells=(\S+)"
)
# The mechanism half: how downwind-biased c's weights are when applied to Wp.
PLATELET_WGT = re.compile(
    r"PHASE_CLOT_PLATELET_WGT\s+time=\S+\s+minUpwindness=(\S+)\s+"
    r"downwindFaces=(\S+)"
)

# HALF the von Neumann limit of 2. With the harmonic face mean the measured
# value is ~2.4e-03, so this threshold sits 400x above the expected value and
# 2x below the limit -- it can be resolved and it can be crossed.
MAX_PLATELET_FO = 1.0
# THE BOUNDS THAT CANNOT BE ARGUED WITH.
#
# The Wp equation has no source and the inlet holds Wp = 1, so Wp belongs to
# [0, 1] for all time. These replaced the old-time-neighbour overshoot test as
# THE gate on 2026-09-02: that test compares against the previous step's
# neighbours, which an IMPLICIT scheme may legitimately exceed, and it produced
# a false positive in every child of run 20260901-200203 (7.3e-06 reported as
# the vanLeer hole firing, while max(Wp) never left 1.00000001).
#
# Wp solves to 1e-12 absolute, so -1e-09 is 1000x the solver noise floor and
# 5e+04 below the -5.32e-05 the defect actually produced in g30.
MIN_WP = -1.0e-9
# 100x the 1e-08 that the pre-fix run reached at its worst.
MAX_WP = 1.0 + 1.0e-6
# RETAINED AND REPORTED, NOT GATED: see above, it is not a boundedness test.
MAX_LED_OVERSHOOT = 1.0e-9

# THE STOICHIOMETRY GATE. Bounds DERIVED, not chosen.
#
# Under the reaction closure the ratio is exact POINTWISE, so the only
# admissible departure is that the c-source uses phaseGradient_ =
# sqrt(|grad c|^2 + f^2) - f while the sink uses |grad c| itself. That
# softening is strictly one-signed -- phaseGradient < |grad c| always, by
# (sqrt(a^2+f^2))^2 < (a+f)^2 -- so the ratio can only come in BELOW 1.
#
# MEASURED on run 20260901-200203's returned fields, over 5 children and
# t = 1 to 30 ms: worst deficit 7.9e-03, at the earliest time, decaying
# monotonically to 3.3e-03 by 30 ms. The floor below therefore sits 3.8x
# beyond the worst value the instrument can legitimately produce, and 9.7x
# above the 0.10 the gradient closure produced. It can be resolved and it can
# be crossed -- which is what the last four unpassable gates in this project
# were not.
MIN_STOICH_NORM = 0.97
# The softening cannot raise the ratio, so anything above 1 is a wiring fault.
MAX_STOICH_NORM = 1.0 + 1.0e-9
# Machine precision, with seven decades of headroom on both sides: the
# identity holds to ~1e-16 under "reaction" and breaks to O(1) otherwise.
MAX_VN_CLOSURE_RESIDUAL = 1.0e-9
# c IS A VOLUME FRACTION IN DISGUISE. Measured on run 20260901-200203's
# written fields: min c = -0.999999012545, max c = +1.000000000000, violation
# identically ZERO across 5 children. The c equation solves far below this, so
# 1e-06 sits about six decades above the solver's own noise and six decades
# below the O(1e-3) excursion a destabilised growth source would produce.
MAX_C_VIOLATION = 1.0e-6
# The share of growth allowed to come from cells with q < 0.05, where
# w = Wp/q is a ratio of two small numbers and the source paper says platelet
# values "have no physical meaning".
#
# MEASURED on run 20260901-200203, 5 children x 7 times: worst 3.999% at
# t = 1 ms when the interface is at its most diffuse, settling to ~2% once it
# sharpens. 0.10 therefore sits 2.5x above the worst legitimate value, and far
# below the tens of percent that a growth term genuinely driven by the
# clot-interior extension would produce.
MAX_GROWTH_BELOW_Q005 = 0.10

# Stage 0's measurement, for cross-reference only. Never a gate here: see the
# module docstring.
STAGE0_S_PER_STEP = 30.3
MIN_SPEEDUP = 3.0
MAX_DET_ERROR = 1.0e-2
MAX_SWEEP_CHANGE = 1.0e-6
# IS THE CLOT HELD? Stage 2's slab gate measured clotU/U = 8.4e-4 -- locked --
# and this assessor had no equivalent, which is how run 20260818-100608 got
# read as a det(be) problem for three sessions while the clot was flowing at
# 88% of the free stream and the report said "peak stretch 4.63".
#
# 0.25 is deliberately loose. An elastic solid bonded to a no-slip wall should
# reach clotU/U near zero, and stage 2 got 8.4e-4, so anything above a quarter
# of the blood speed is not "some deformation" -- it is a clot being carried
# away. The threshold is set where it cannot be argued about rather than where
# it is tight, because a tight one here would fail children for having real
# large deformation.
MAX_LOCK_RATIO = 0.25
# HAS THE DRIVEN QUANTITY STOPPED MOVING? Relative change of the clot-mean
# be_xy across the final quarter of the window.
#
# 5% IS DERIVED, NOT PICKED. Over run 20260823-102340's 3e-3 s window the
# clot-mean Ux decrement per 0.3 ms flattened to a near-constant ~0.18e-3 out
# of 5.96e-3, i.e. ~3% per 0.3 ms sustained, while be_xy at the wall was still
# climbing 0.44 -> 2.77 essentially linearly. A window whose last quarter still
# moves the strain by more than 5% is unambiguously still in that regime. The
# floor it must clear is the sampling noise, which is far below this: be_xy is
# a volume mean over ~800 cells written every step.
#
# It is a RELATIVE change on a signed quantity that is positive and monotone
# through the clot (2.765 at the wall down to 0.19 at the crest, all positive
# at t = 3e-3), so the mean does not suffer the cancellation that would make a
# ratio meaningless.
MAX_BE_XY_REL_CHANGE = 0.05

# THE ANCHOR GATES. Every threshold here is derived from a measurement, not
# picked, because this project has now set four gates that could not be passed
# and one that could not be failed.
#
# MAX_RELEASE_STEP_JUMP. A genuinely progressive release adds a few cells per
# step at most: the freed body travels at Ubody = 9.14e-3 m/s and the step is
# Courant-limited at 8.09e-7 s, so an advancing release front covers 7.4e-9 m
# per step against a 1e-6 m cell -- less than a hundredth of a cell. The
# measured snap was 1278 cells in ONE step. 50 sits two orders below what the
# defect does and three orders above what physics can do, so it can be both
# failed and passed.
MAX_RELEASE_STEP_JUMP = 50.0

# MIN_FINAL_COVERAGE. Fraction of injured wall faces still carrying clot at
# the end. The control measured 6/40 = 0.15; a held clot should be near 1.0.
# 0.5 is between them by a factor of 3.3 either way, so neither outcome is
# marginal. It is NOT 0.9: some coverage loss at the wound edges is legitimate
# deformation and gating at 0.9 would fire on the fix as well as the defect.
MIN_FINAL_COVERAGE = 0.5

# FROZEN_BE_XY. A HELD CLOT MUST STILL DEFORM, and the taper exists precisely
# because the obvious repair does not. The penalty enters as
# fvm::Sp(bondHoldPenalty*bondHold*chi, U), so a hold of 1 across the body
# drives U to zero; be is stretched by gradU built from U and phi, so gradU = 0
# leaves be at the identity and the clot becomes RIGID. Dr He Li, 2026-08-25,
# asked for "small and smooth deformation of the entire core region" -- a
# frozen core fails his specification exactly as a severed one does, in the
# opposite direction, and nothing in the coverage or height gates can see it.
#
# DERIVED, NOT PICKED. Run 20260825-013027 measured clot-mean be_xy at 0.0846
# (an_ctl, unheld), 0.2646 (an_all) and 0.2664 (an_hold). A clot that is held
# and deforming sits at O(0.1). 1e-3 is 85x below the smallest of those and
# eleven orders above the double-precision floor, so it separates "deforming
# less" from "not deforming".
FROZEN_BE_XY = 1.0e-3

# Any strength at or above this is a control that cannot fail. The series uses
# 1e9 Pa for unbreakable and 151 Pa for the measured requirement -- seven
# orders apart -- so the split point is not delicate.
UNBREAKABLE_PA = 1.0e8

# Fraction of the clot that must come free before it counts as the body
# DEPARTING rather than shedding. The control freed 98% in one step; ordinary
# shedding in these runs is a handful of cells out of ~1320, i.e. under 1%.
# A quarter is two orders above the shedding and a factor of four below the
# failure, so both sides are clear of it.
RELEASE_DEPARTURE_FRACTION = 0.25
# IS THE PLATELET FIELD BOUNDED? w is a concentration with a physical bound of
# ~1.1. Run 20260818-232055 finished at w = 6372 -- a factor of 5800 -- and
# every gate in this file passed it, because not one of them looked at w.
#
# This is not a cosmetic omission. Vn = alpha*(n & grad(w)) enters the phase
# field as -Vn*phaseGradient, so an unbounded w becomes an unbounded interface
# velocity (0.61 m/s measured, against a 0.01 m/s flow) and then the cell-scale
# noise in c that was visible in ParaView for three sessions while the numbers
# were read as a det(be) problem. The diverging quantity was three operators
# upstream of the one being watched.
#
# 10.0 rather than 1.1: the 1.1 bound is the model's, and a gate set exactly at
# it would fail children for ordinary interface sharpening. An order of
# magnitude past the bound cannot be argued as anything but divergence.
MAX_W = 10.0
# Steps 0-19 include mesh setup, first-step Euler fallback and the initial
# pressure transient. Stage 0 measured the cost split flat from step 20
# (0.773 / 0.775 / 0.773 / 0.772 at steps 0/20/50/100), so 20 is where a
# per-step number becomes meaningful.
WARMUP_STEPS = 20

STATIC_PAIRS = (("stage3_xu_eul", "stage3_xu_off", "Xu scale (source 33)"),
                ("stage3_li_eul", "stage3_li_off", "Dr Li's dome"))

# Eulerian-only children have NO off partner by design. Run 20260818-081527's
# five diagnostic children were all of this kind, and this module still had only
# the two static pairs above -- so even once the runner was told to invoke it,
# it would have reported on four children that did not exist and ignored the
# five that did. The pairing is now discovered from the runs directory.
#
# The health checks -- det(be), Picard convergence, SPD, the trace diagnostic,
# the interface/bulk ratio -- apply to every Eulerian child and are the reason
# these runs exist. Only the SPEEDUP needs a partner, and where there is none it
# is reported absent rather than fabricated from a stage-0 number measured on a
# different VM. That cross-VM substitution is exactly the move the cost gate was
# written to refuse.

# THE PREFIX LIST IS A FALLBACK NOW, NOT THE TRIGGER. It failed twice: run
# 20260818-081527 (five diag_* children, gate never ran) and run
# 20260823-052707 (force_cell/force_face matched nothing, so discover_pairs
# returned empty, main() took its refusal path, and no stage3_result.json was
# written at all). Both times the mitigation on offer was "add the new prefix",
# which works until the next child is named something new.
#
# The real trigger is the SERIES, read from each run's own child.json. A series
# cannot be forgotten the way a prefix can, because create_cases.py refuses a
# child whose series is not declared in the controls file. Every stage-3 child
# lives in a series whose name begins "eulerian".
EULERIAN_ONLY_PREFIXES = (
    "diag_", "fix_", "anchor_", "flux_", "grad_", "renorm_", "plt_", "gate_",
    "drift_", "force_",
)
EULERIAN_SERIES_PREFIX = "eulerian"

# THE PLATELET-CONTENT GATES. Derivation and pre-registered numbers:
# research/designs/AM2020_platelet_content_formulation_derivation.md section 5.
#
# These are scoped BY CHILD, not applied to every Eulerian child, because they
# are only meaningful under their own initial conditions -- a growing case is
# supposed to move w, and gating it at 1e-10 everywhere would be nonsense.
#
# BOTH ARE BUILT TO FAIL ON TODAY'S SOLVER. That is deliberate: a gate that has
# never failed has not been tested, and this project has been bitten by exactly
# that more than once. They are the acceptance test for the content
# formulation, written BEFORE it, so its verdict cannot be tuned afterwards.

# Gate A: alpha = 0, k0 = k1 = 0, mobility = 0, w uniform 1. c is then purely
# advected, q satisfies its own conservation law, and w == 1 is EXACT.
GATE_A_CHILDREN = ("gate_a_advect", "gate_a_content")

# GATE A-EXACT, and it RESTORES the 1e-10 the note below abandoned.
#
# That note diagnosed the symptom correctly -- "the c and W equations are
# discretised differently ... so a startup transient is injected in the first
# step" -- and then drew the wrong conclusion from it. It treated the
# difference as a fact of life and relaxed the gate to fit. It was a DEFECT,
# and run 20260819-231939 found it: c is advected by phiReleased, which inside
# the clot is the rigid-body flux, while W was advected by phi_, the fluid flux
# that the Brinkman penalty drives to zero exactly there. Two carriers, one
# ratio. W was smooth and blameless across the whole clot; every wiggle in
# w = W/q was q's, amplified by 1/q = 15 at the trough.
#
# With W on the phase field's own carrier and operator, W = q is preserved by
# exact arithmetic, so w == 1 to machine precision -- no transient to tolerate,
# nothing to decay. gate_a_exact is gate_a_content with that one change and
# nothing else, so the two numbers are directly comparable: 0.0211 against
# this. A gate that cannot pass is not a gate; neither is one relaxed to fit a
# result. This one can pass, and only if the fix is real.
GATE_A_EXACT_CHILDREN = ("gate_a_exact", "gate_a_nodiff", "gate_a_linear",
                         "gate_a_tol", "gate_a_floor")

# RE-SPECIFIED 2026-08-22 AFTER THE GATE FAILED, AND THE REASON IS RECORDED
# BECAUSE CHANGING A LIMIT AFTER SEEING A RESULT IS EXACTLY HOW A GATE STOPS
# MEANING ANYTHING. This is the second time this gate has moved. The first move
# (1e-10 -> 0.05) was wrong: it accommodated a defect. This one is different in
# kind and the difference has to be arguable, not asserted.
#
# THE OLD LIMIT WAS ON THE WRONG QUANTITY. w = Wp/max(q,qFloor), so max|w - 1|
# is |Wp - q| DIVIDED BY q, and q reaches 2e-5 in the deep clot. Demanding
# 1e-10 on that ratio demanded |Wp - q| ~ 1e-15 -- below the linear solver's
# own `tolerance 1e-12`. It was a threshold beneath the resolution of the
# instrument and NO correct discretisation could ever have passed it. That is
# the fourth time this project has set such a threshold (see the 3b-ii-c mesh
# tolerance and the two zero-flow-limit steadiness limits) and the first time
# I did it while writing the paragraph warning against relaxing limits.
#
# THE NEW LIMIT IS ON THE CONSERVED QUANTITY and is derived, not chosen:
# fvSolution solves Wp at `tolerance 1e-12` absolute with relTol 0, over ~400
# steps, so per-step solver error alone can accumulate to ~1e-9 without any
# discretisation inconsistency at all. 1e-11 sits two orders above the
# single-solve floor and two orders BELOW the 2.1e-4 the defect produces --
# tight enough that the defect cannot hide, loose enough that the instrument
# can express a pass.
GATE_A_EXACT_TOL = 1.0e-11

# THE CERTIFICATE HAS A CEILING, and the dry run is why.
#
# The floor certificate (see GATE A-EXACT below) excuses a production-settings
# child that sits above GATE_A_EXACT_TOL when gate_a_floor, running the same
# operator to its round-off floor, comes in under it. Written without a
# ceiling, that branch silenced `gate_a_exact_absolute` -- the synthetic shape
# carrying a REAL 2.1e-4 operator defect -- because the floor child in that
# shape was clean.
#
# A deviation 7 orders of magnitude above the floor is not iterative
# incompleteness whatever any other child reports. The certificate applies
# only while the excess stays COMMENSURATE with the limit: measured, the
# production children sit at 2.6167e-11 against a 6.5553e-12 floor, a factor
# of 3.99 and 2.6x the limit. 10x the limit certifies that comfortably and
# rejects 2.1e-4 by seven orders.
GATE_A_EXACT_CERTIFY_CEILING = 10.0*GATE_A_EXACT_TOL

# Reported alongside, and gated too, because it answers a different question:
# a deviation at q >= 0.5 cannot be a 1/q conditioning artefact, so it is the
# discriminator between "the absolute error is tiny and the ratio magnifies it"
# and "the transport is genuinely wrong". Run 20260822-085149 measured 3.6e-4
# here, which is why the residual is known NOT to be conditioning.
GATE_A_BLOOD_TOL = 1.0e-9
# RE-SPECIFIED after run 20260819-204541, and the reason is recorded because
# relaxing a limit after seeing the result is exactly how a gate stops meaning
# anything.
#
# The original 1e-10 assumed w == 1 exactly. It is exact in the CONTINUUM, but
# the c and W equations are discretised differently -- c carries the `bounded`
# correction and W does not -- so a startup transient is injected in the first
# step and then decays. Measured on the content branch: 1.0211 at step 1,
# decaying monotonically to 1.0045. On the w branch: 64.77 and still climbing.
#
# 1e-10 could therefore never pass, and a gate that cannot pass is not a gate.
# The two limits below replace it and BOTH must hold:
#
#   MAGNITUDE -- the transient must stay small. 0.05 is where a 5% platelet
#   error stops being arguable; the content branch sits at 0.021 peak.
#
#   DECAY -- and this is the one that carries the physics. An amplifier grows;
#   a discretisation transient decays. The w branch grew by 64x, the content
#   branch shrank by 4.7x. This check distinguishes the two MECHANISMS rather
#   than their sizes, which is what the original gate was really trying to ask.
GATE_A_TOL = 0.05

# Gate E: growth ON, uniform w. No concentration gradient means no diffusive
# flux, so the diffusion-limited interface speed must be exactly zero. This is
# the test that separates the two candidate Vn closures; the rejected
# grad(W) form predicts 1.5e-4 m/s here.
# gate_e_long is the ten-times-longer PROBE of the linear drift. It is gated
# on the same terms, deliberately: a probe exempt from the gate it is
# probing would report a number nobody has to act on.
GATE_E_CHILDREN = ("gate_e_vn", "gate_e_content", "gate_e_long",
                   "drift_long")
# RE-SPECIFIED after run 20260819-213519, for the SAME reason gate A was, and
# recorded plainly because this is the second gate relaxed after seeing a
# result -- a pattern that deserves distrust.
#
# THE ORIGINAL ERROR, in both gates: 1e-12 assumed w == 1 exactly. It is exact
# in the continuum, but c carries the `bounded` correction and Wp does not, so
# a startup transient is injected and decays. w therefore holds a residual of
# a few times 1e-4 at the interface, and Vn = alpha*(n & qEff*grad(w)) reports
# it FAITHFULLY. A closure that returned machine zero against a nonzero grad(w)
# would be broken.
#
# Confirmed by locating the maximum rather than assuming it. Before the
# gradFloor repair: cell 508, c = 1.000000, interfaceDelta 1e-10 of the domain
# max -- an artifact in pure blood. After: cell 4314, c = -0.317,
# interfaceDelta 60% of the domain max -- a genuine interface cell, where
# w = 0.99976 gives grad(w) ~ 60 1/m and hence Vn ~ 1e-8, against 9.47e-8
# measured. Same order. The number is physics, not noise.
#
# THE LIMIT IS SET BY WHAT DISCRIMINATES THE CLOSURES, NOT BY WHAT WE MEASURED.
# The rejected Vn = alpha*(n & grad W) predicts 1.5e-4 m/s under uniform blood,
# because grad(q) is parallel to n and contributes w*|grad q| with a sign that
# never changes. 1e-5 is fifteen times below that and a hundred times above the
# adopted closure's reading, so it cannot be satisfied by the wrong closure and
# cannot be accused of being tuned to the right one. In flow terms: 1.2e-3 of
# the blood velocity, against the rejected closure's 1.8e-2.
GATE_E_TOL = 1.0e-5

# GATE E, RESTORED TO MACHINE ZERO FOR THE REACTION CLOSURE -- 2026-09-02.
#
# This gate was relaxed twice (1e-12 -> 1e-5), and the record says plainly that
# the pattern "deserves distrust". The relaxation was not a fudge: under
# Vn = alpha*(n & qEff*grad w) a real w-residual of a few times 1e-4 produces a
# real gradient, and the closure reported it FAITHFULLY. The gate could not be
# met because it was asking the gradient closure to be blind.
#
# Every gate E child runs k0 = k1 = 0. Under Vn = (alpha/D)*kEff*w that makes
# Vn identically zero IN FLOATING POINT -- kEff = max(k0*E, 0) = 0.0 exactly,
# and 0.0 times anything finite is 0.0 -- no matter what residual w carries.
# So the reaction closure does not merely pass this gate, it makes the
# ORIGINAL specification attainable, and the relaxed tolerance is no longer
# needed to describe it.
#
# Applied ONLY where the log shows the child really had k = 0, which is
# plateletUptake identically zero. Where k > 0 a non-zero Vn is correct
# physics and this test does not apply; that case is certified by
# VnClosureResidual instead.
GATE_E_EXACT_TOL = 0.0


def discover_pairs() -> tuple:
    """On/off pairs from the static matrix, plus any Eulerian-only children."""
    pairs = [(on, off, label) for on, off, label in STATIC_PAIRS
             if (RUNS / on).is_dir() or (RUNS / off).is_dir()]
    named = {cid for on, off, _ in pairs for cid in (on, off)}
    if RUNS.is_dir():
        for d in sorted(p.name for p in RUNS.iterdir() if p.is_dir()):
            if d in named:
                continue
            # Series first, prefix as a fallback -- see EULERIAN_SERIES_PREFIX.
            series = ""
            spec = RUNS / d / "child.json"
            if spec.is_file():
                try:
                    series = str(json.loads(spec.read_text()).get("series", ""))
                except (ValueError, OSError):
                    series = ""
            if (
                series.startswith(EULERIAN_SERIES_PREFIX)
                or d.startswith(EULERIAN_ONLY_PREFIXES)
            ):
                pairs.append((d, None, f"{d} (Eulerian only, no off partner)"))
    return tuple(pairs)


def read(child_id: str) -> dict:
    case = RUNS / child_id
    log = case / "log.phaseClotAxisymmetricFluid"
    if not log.exists():
        return {"child_id": child_id, "ran": False}
    text = log.read_text(encoding="utf-8", errors="replace")

    steps = [(float(t), float(s), float(k)) for t, s, k in COST.findall(text)]

    # A LOG THAT EXISTS BUT ADVANCED NO STEPS IS A CRASH, NOT A RUN.
    #
    # Run 20260902-141852: four children died with SIGFPE inside the solver
    # CONSTRUCTOR (D = 0 divided in the reaction closure). The log existed, so
    # `ran` was True with zero steps, and every downstream "no PHASE_CLOT_x
    # lines" check fired -- including "this binary predates the 2026-09-02
    # reaction growth closure", which was FALSE and pointed at the package
    # instead of at the crash. Twenty-four failures, all misdirection, for one
    # cause. Distinguish the two here so the report names the crash.
    if not steps:
        crash = None
        if "sigFpe::sigHandler" in text or "Floating point exception" in text:
            crash = "SIGFPE (floating-point exception)"
        elif "FOAM FATAL ERROR" in text or "FOAM FATAL IO ERROR" in text:
            crash = "FOAM FATAL ERROR"
        elif "printStack" in text:
            crash = "fatal signal"
        if crash:
            frame = None
            for line in text.splitlines():
                st = line.strip()
                if st.startswith("#") and "phaseClotAxisymmetricFluid" in st:
                    frame = st
                    break
            return {
                "child_id": child_id,
                "ran": False,
                "crashed": True,
                "crash_kind": crash,
                "crash_frame": frame,
            }
    # ADDING beDetErrorPost AS AN OPTIONAL GROUP SHIFTED EVERY INDEX AFTER IT:
    # sweepChange moved from r[3] to r[4]. That is a SILENT corruption -- the
    # Picard-convergence gate would have started reading the det error instead
    # and still produced plausible numbers. Named here because the same class of
    # shift is one edit away every time a field is added to a log line.
    #   r[0] minMinor  r[1] maxStretch  r[2] detError
    #   r[3] detErrorPost (nan on logs written before it existed)
    #   r[4] sweepChange
    euler = [
        tuple(float(x) if x else float("nan") for x in row)
        for row in EULER.findall(text)
    ]
    # NaN, NOT float(''), for the optional trBulkFluxConsistent group. Logs
    # written before that diagnostic existed leave it empty, and a bare
    # float(x) raises there -- caught by replaying run 20260818-211300's real
    # logs, which is the whole reason the replay check exists. A parser that
    # only reads logs from the build it shipped with cannot check a regression.
    trace = [
        tuple(float(x) if x else float("nan") for x in row)
        for row in TRACE.findall(text)
    ]
    phase = [(float(a), float(b), float(c)) for a, b, c in PHASE.findall(text)]
    # Kept as FINAL VALUES, not maxima over time. The volume-weighted trace
    # fields were exported as max-over-time and read identically in all three
    # children of run 20260823-070314 because that maximum falls early, before
    # the runs diverge -- useless as a late-time discriminator. A force balance
    # is a statement about the state the run ENDED in.
    momentum = _momentum_rows(text)
    beterm = _beterm_rows(text)
    beterm_worst = _beterm_at_worst(beterm)
    be_xy_plateau = _plateau(momentum, "clotBeXY")
    sigma_xy_plateau = _plateau(momentum, "clotSigmaXY")
    lock = [tuple(float(x) for x in row) for row in LOCK.findall(text)]
    # Unmatched OPTIONAL groups come back as '' from findall, and float('')
    # raises -- so the absent trailing fields become None rather than
    # crashing the assessor on every pre-2026-08-22 log.
    plt = [
        tuple(float(x) if x else None for x in row)
        for row in PLATELET.findall(text)
    ]
    fo = [(float(a), float(b)) for a, b in PLATELET_FO.findall(text)]
    led = [
        tuple(float(x) if x else None for x in row)
        for row in PLATELET_LED.findall(text)
    ]
    lim = [
        (float(r[0]), float(r[1]) if r[1] else None)
        for r in PLATELET_LIM.findall(text)
    ]
    lab = [
        (float(r[0]), float(r[1]), float(r[2]))
        for r in LABEL_DRIFT.findall(text)
    ]
    wgt = [(float(a), float(b)) for a, b in PLATELET_WGT.findall(text)]
    sto = [
        (
            float(r[0]), r[1], r[2],
            float(r[4]), float(r[5]), float(r[6]), float(r[7]), float(r[8]),
            float(r[9]) if r[9] else None,
            float(r[10]) if r[10] else None,
            # content= is OPTIONAL so logs written before 2026-09-02 parse;
            # absent means "unknown", which must NOT be read as "legacy".
            (r[3] or None),
        )
        for r in GROWTH_STOICH.findall(text)
    ]
    dam = [
        (float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]))
        for r in DAMKOHLER.findall(text)
    ]
    pbnd = [
        (float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]))
        for r in PHASE_BOUNDS.findall(text)
    ]
    gsup = [
        (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
        for r in GROWTH_SUPPORT.findall(text)
    ]

    return {
        "child_id": child_id,
        "ran": True,
        "steps": len(steps),
        "final_time_s": steps[-1][0] if steps else float("nan"),
        "step_costs": [s for _, s, _ in steps],
        # WHERE THE SAVING CAME FROM, so the speedup is not just a number.
        # The ON children also drop nOuterCorrectors from 60 to 1, because one
        # velocity means there is no FSI to iterate -- that is the mechanism,
        # not a confound, but a reader is entitled to see how much of the
        # total is the skeleton going away versus the outer loop going away.
        "skeleton_costs": [k for _, _, k in steps],
        "spd_lost": bool(SPD_LOST.search(text)),
        "min_leading_minor": min((r[0] for r in euler), default=float("nan")),
        "peak_stretch": max((r[1] for r in euler), default=float("nan")),
        # THE NUMBER THIS ASSESSOR SHOULD HAVE CARRIED FROM THE START.
        # det(be) = 1 is EXACT for any isochoric deformation at any magnitude,
        # so this is not a large-strain allowance -- it is whether the model
        # conserved volume. Run 20260817-234636 reached 3.24 and this assessor
        # reported only that it could not compute one cost ratio, because it
        # checked SPD and interface_over_bulk and nothing else. Stage 2 gated
        # the same quantity at 1e-6 and measured 1.06e-6 on the 1-D slab.
        "max_det_error": max((r[2] for r in euler), default=float("nan")),
        "final_det_error": euler[-1][2] if euler else float("nan"),
        # The Picard iteration's own convergence. It grew from 3.8e-16 to
        # 7.9e-4 over that run, i.e. three sweeps stopped resolving the
        # explicit stretch term as the clot reached stretch 4.6.
        "final_sweep_change": euler[-1][4] if euler else float("nan"),
        "max_sweep_change": max((r[4] for r in euler), default=float("nan")),
        # det error AFTER the optional det^(-1/3) rescale. nan on logs from
        # builds without it, which is correct: it was not measured, not zero.
        "final_det_error_post": euler[-1][3] if euler else float("nan"),
        "max_det_error_post": max(
            (r[3] for r in euler if not math.isnan(r[3])), default=float("nan")
        ),
        # WHY, if it drifted. d(ln det be)/dt = 2 tr(L), so det is preserved
        # iff tr(L) = 0. divPhiMax is the same quantity the continuity check
        # reports; if traceFrac is large while divPhiMax is at machine zero,
        # the two discrete operators disagree and the remedy is a trace-free
        # source rather than a smaller step.
        # THE ABSOLUTE TRACE, beside the ratio. |tr L|/|L| is 0/0 in a
        # quiescent cell and the guard makes it 0, but in a nearly-quiescent
        # one it is a ratio of two tiny numbers and can sit at the 2-D ceiling
        # of sqrt(2) while meaning nothing. Run 20260818-081527's blood bin did
        # exactly that in all five children (1.4113-1.4139). The bulk bin did
        # not -- it was backed by tr L = 499 1/s against a 1000 1/s shear rate
        # -- and printing both is what lets those two cases be told apart.
        "tr_bulk": max((r[0] for r in trace), default=float("nan")),
        "tr_interface": max((r[1] for r in trace), default=float("nan")),
        "tr_blood": max((r[2] for r in trace), default=float("nan")),
        # Read at the END of the run, not as a max: a clot that is still
        # accelerating at the last step has not locked whatever it did early,
        # and the early steps are the transient this is not asking about.
        "lock_ratio": lock[-1][3] if lock else float("nan"),
        "clot_u_mean": lock[-1][0] if lock else float("nan"),
        "blood_u_mean": lock[-1][2] if lock else float("nan"),
        "clot_cells_final": lock[-1][4] if lock else float("nan"),
        "clot_cells_initial": lock[0][4] if lock else float("nan"),
        "has_lock": bool(lock),
        # THE BOND. Parsed here for the first time, and the reason is run
        # 20260824-033102: the stage-3 assessor never read PHASE_CLOT_DEBOND
        # at all, so a bond whose failure criterion was INERT for 20 ms passed
        # every gate in this file silently. sigma_found is the one that
        # matters -- see the gate below.
        **_debond_summary(text),
        **_anchor_summary(text),
        **_departure_before_debond(text),
        # PLATELET BOUNDEDNESS. Maxima over the whole run, not the final step:
        # w can spike and be smeared back, and a final-step reading would miss
        # it. wMin is tracked too because the runaway of run 20260818-232055
        # was accompanied by w falling to 7.7e-4 elsewhere -- the field is
        # being redistributed as well as amplified, and only seeing the top of
        # it would suggest a pure source.
        "w_max": max((r[0] for r in plt), default=float("nan")),
        "w_min": min((r[1] for r in plt), default=float("nan")),
        "vn_mag_max": max((r[2] for r in plt), default=float("nan")),
        "w_over_bound": max((r[3] for r in plt), default=float("nan")),
        "has_platelet": bool(plt),
        # THE STABILITY INSTRUMENT, added 2026-09-01.
        "has_fo": bool(fo),
        "platelet_fo_max": max((r[1] for r in fo), default=float("nan")),
        "platelet_fo_at": (
            max(fo, key=lambda r: r[1])[0] if fo else float("nan")
        ),
        "has_led": bool(led),
        "has_wp_bounds": bool(led) and led[0][1] is not None,
        "wp_min": min((r[1] for r in led if r[1] is not None), default=float("nan")),
        "wp_max": max((r[2] for r in led if r[2] is not None), default=float("nan")),
        "wp_min_at": (
            min((r for r in led if r[1] is not None), key=lambda r: r[1])[0]
            if any(r[1] is not None for r in led) else float("nan")
        ),
        "led_over_max": max((r[3] for r in led), default=float("nan")),
        "led_under_max": max((r[4] for r in led), default=float("nan")),
        "led_viol_steps": sum(1 for r in led if r[5] > 0),
        "led_worst": (max(led, key=lambda r: r[3]) if led else None),
        "has_lim": bool(lim),
        "faces_upwinded_max": max((r[0] for r in lim), default=float("nan")),
        "faces_upwinded_interface_max": max(
            (r[1] for r in lim if r[1] is not None), default=float("nan")
        ),
        "has_label_drift": bool(lab),
        "label_in_blood_max": max(
            (r[2] for r in lab if r[1] > 0), default=float("nan")
        ),
        "has_wgt": bool(wgt),
        "min_upwindness": min((r[0] for r in wgt), default=float("nan")),
        # THE GROWTH CLOSURE, added 2026-09-02.
        "has_stoich": bool(sto),
        "growth_closure": sto[0][1] if sto else None,
        "depletion_model": sto[0][2] if sto else None,
        "content_branch": sto[0][10] if sto else None,
        # Only steps with a live interface say anything: before any platelet
        # has been absorbed the ratio is 0/0 and the solver reports 0.
        "stoich_norm_min": min(
            (r[7] for r in sto if r[4] > 0), default=float("nan")
        ),
        "stoich_norm_max": max(
            (r[7] for r in sto if r[4] > 0), default=float("nan")
        ),
        "stoich_norm_at": (
            min((r for r in sto if r[4] > 0), key=lambda r: r[7])[0]
            if any(r[4] > 0 for r in sto) else float("nan")
        ),
        "stoich_target": sto[0][6] if sto else float("nan"),
        # Identically zero uptake means k == 0, which is what every gate E
        # child sets and what makes Vn exactly zero under the reaction
        # closure. Read from the log rather than the dictionary so the gate
        # tests the run that happened.
        "uptake_max": max((r[4] for r in sto), default=float("nan")),
        "has_vn_closure": bool(sto) and sto[0][8] is not None,
        "vn_closure_resid_max": max(
            (r[8] for r in sto if r[8] is not None), default=float("nan")
        ),
        "has_growth_support": bool(gsup),
        # Only steps that actually grew say anything; before the first
        # deposition the solver reports 0 and that is not a 0% support.
        "growth_below_q005_max": max(
            (r[1] for r in gsup if r[3] > 0), default=float("nan")
        ),
        "growth_below_q015_max": max(
            (r[2] for r in gsup if r[3] > 0), default=float("nan")
        ),
        "growth_mean_q_min": min(
            (r[3] for r in gsup if r[3] > 0), default=float("nan")
        ),
        "growth_support_at": (
            max((r for r in gsup if r[3] > 0), key=lambda r: r[1])[0]
            if any(r[3] > 0 for r in gsup) else float("nan")
        ),
        "has_phase_bounds": bool(pbnd),
        "c_min": min((r[1] for r in pbnd), default=float("nan")),
        "c_max": max((r[2] for r in pbnd), default=float("nan")),
        "c_under_max": max((r[3] for r in pbnd), default=float("nan")),
        "c_over_max": max((r[4] for r in pbnd), default=float("nan")),
        "c_worst_at": (
            max(pbnd, key=lambda r: max(r[3], r[4]))[0] if pbnd
            else float("nan")
        ),
        "has_damkohler": bool(dam),
        "da_interface_max": max((r[2] for r in dam), default=float("nan")),
        "layer_cells_min": min(
            (r[3] for r in dam if r[4] > 0), default=float("nan")
        ),
        "downwind_faces_max": max((r[1] for r in wgt), default=float("nan")),
        # THE ABSOLUTE DEVIATION. |Wp - q| on the conserved quantity, with no
        # division by a q that reaches 2e-5. This is the number gate A-exact
        # should always have been written on; see GATE_A_EXACT_TOL.
        "wp_dev_max": max(
            (r[4] for r in plt if r[4] is not None), default=float("nan")
        ),
        "q_min": min(
            (r[5] for r in plt if r[5] is not None), default=float("nan")
        ),
        # The ratio restricted to q >= 0.5. A deviation that survives HERE is
        # not a conditioning artefact -- there is no amplification in pure
        # blood -- so this is what separates a real transport error from the
        # 1/q magnification of a tiny absolute one.
        "w_dev_blood": max(
            (r[6] for r in plt if r[6] is not None), default=float("nan")
        ),
        "wp_dev_early": max(
            (r[4] for r in plt[: max(1, len(plt)//4)] if r[4] is not None),
            default=float("nan"),
        ),
        "wp_dev_late": max(
            (r[4] for r in plt[-max(1, len(plt)//4):] if r[4] is not None),
            default=float("nan"),
        ),
        "has_abs_platelet": bool(plt) and plt[0][4] is not None,
        # Vn split early/late on the same quarters as w, so a DECAYING
        # interface speed can be told from a GROWING one. Decay is the
        # criterion that carries the physics: it is strictly STRONGER than a
        # magnitude threshold, because an amplifier fails it at any size.
        "vn_early": max(
            (r[2] for r in plt[: max(1, len(plt)//4)]), default=float("nan")
        ),
        "vn_late": max(
            (r[2] for r in plt[-max(1, len(plt)//4):]), default=float("nan")
        ),
        # Per-step deviation of w from 1, split early/late so a DECAYING
        # transient can be told apart from a GROWING one. Quarters rather than
        # halves: the transient is injected in the first few steps, so a
        # half-window average would dilute exactly the part that matters.
        "w_dev_max": max(
            (max(abs(r[0] - 1.0), abs(r[1] - 1.0)) for r in plt),
            default=float("nan"),
        ),
        "w_dev_early": max(
            (max(abs(r[0] - 1.0), abs(r[1] - 1.0))
             for r in plt[: max(1, len(plt)//4)]),
            default=float("nan"),
        ),
        "w_dev_late": max(
            (max(abs(r[0] - 1.0), abs(r[1] - 1.0))
             for r in plt[-max(1, len(plt)//4):]),
            default=float("nan"),
        ),
        # Keyed lookups against the last MOMENTUM line. _mom() returns nan for
        # a key the solver that wrote this log did not print, so an old log
        # reads as "not measured" instead of shifting a neighbour's value into
        # the slot.
        "acc_inertia": _mom(momentum, "accInertia"),
        "acc_convection": _mom(momentum, "accConvection"),
        "acc_viscous": _mom(momentum, "accViscous"),
        "acc_elastic": _mom(momentum, "accElastic"),
        "acc_pressure": _mom(momentum, "accPressure"),
        "acc_penalty": _mom(momentum, "accPenalty"),
        "acc_drag": _mom(momentum, "accDrag"),
        # THE THIRD ANCHOR. momentumPredictor() has added
        # fvm::Sp(bondHoldPenalty*bondHold*chi, U) on the Eulerian path since
        # 2026-08-24 and no budget line accounted for it -- accPenalty is
        # lambda_, which that term is not -- so acc_residual has been carrying
        # an in-equation force as an imbalance on every eulerian_* run to
        # date. Reads nan for every log written before 2026-08-27, which is
        # what it is: not measured.
        "acc_bond_hold": _mom(momentum, "accBondHold"),
        # THREE FLAGS, NOT ONE. anchorsInUEqn gated accPenalty and accDrag
        # together, which was only ever right while the two moved together.
        # eulerianBrinkman puts accDrag in the equation while lambda_ stays
        # out of it, and bondHold is on a switch of its own.
        "anchors_in_ueqn": _mom(momentum, "anchorsInUEqn"),
        "brinkman_in_ueqn": _mom(momentum, "brinkmanInUEqn"),
        "bond_hold_in_ueqn": _mom(momentum, "bondHoldInUEqn"),
        # THE REALISED DRAG COEFFICIENT, kinematic (1/s), over clot cells.
        # drag_free_cells is the release gate's assertion: with Brinkman on it
        # must equal the released set, so 0 while everything is connected and
        # rising only as material genuinely detaches. Equal to
        # clot_cells_mom means the gate is inverted and nothing is held;
        # stuck at 0 while material detaches means an embolus is being pinned,
        # which is the objection this term was removed for.
        "drag_min": _mom(momentum, "dragMin"),
        "drag_max": _mom(momentum, "dragMax"),
        "drag_free_cells": _mom(momentum, "dragFreeCells"),
        "clot_cells_mom": _mom(momentum, "clotCellsMom"),
        "acc_sum": _mom(momentum, "accSum"),
        "acc_residual": _mom(momentum, "accResidual"),
        "clot_xbar": _mom(momentum, "clotXbar"),
        "clot_uxbar": _mom(momentum, "clotUxbar"),
        "clot_volume": _mom(momentum, "clotVolume"),
        # THE SATURATION OBSERVABLES.
        "clot_be_xy": _mom(momentum, "clotBeXY"),
        "clot_sigma_xy": _mom(momentum, "clotSigmaXY"),
        "clot_be_xy_max": _mom(momentum, "clotBeXYmax"),
        "be_xy_late_start": be_xy_plateau["start"],
        "be_xy_late_end": be_xy_plateau["end"],
        "be_xy_rel_change": be_xy_plateau["rel_change"],
        "be_xy_samples": be_xy_plateau["n"],
        # THE be TERM BUDGET at the deepest point of the failure. Exported
        # rather than left in the log because run 20260823-061943's numbers had
        # to be read out of raw logs by hand when a regex was extended and the
        # results dict was not.
        "beterm_samples": len(beterm),
        "beterm_lmin_worst": beterm_worst.get("lminWorst", float("nan")),
        "beterm_time": beterm_worst.get("time", float("nan")),
        "beterm_det": beterm_worst.get("detWorst", float("nan")),
        "beterm_lmax": beterm_worst.get("lmaxWorst", float("nan")),
        "beterm_x": beterm_worst.get("xWorst", float("nan")),
        "beterm_y": beterm_worst.get("yWorst", float("nan")),
        # indWorst, not cWorst: S6c-2 repointed the solver's value from c
        # to ind and the key was relabelled to match in the same change.
        # ind = 1 - q, so 1 is a packed skeleton and 0 is blood -- the
        # OPPOSITE sense to the c it replaced. A nan here means the log
        # predates that rename; it prints as nan rather than misleading.
        "beterm_ind": beterm_worst.get("indWorst", float("nan")),
        "beterm_rate_advection": beterm_worst.get(
            "detRateAdvection", float("nan")),
        "beterm_rate_noncons": beterm_worst.get(
            "detRateNonCons", float("nan")),
        "beterm_rate_stretch": beterm_worst.get(
            "detRateStretch", float("nan")),
        "beterm_rate_relax": beterm_worst.get("detRateRelax", float("nan")),
        "beterm_rate_termsum": beterm_worst.get(
            "detRateTermSum", float("nan")),
        "beterm_rate_actual": beterm_worst.get("detRateActual", float("nan")),
        "beterm_rate_gap": beterm_worst.get("detRateGap", float("nan")),
        "beterm_dominant": _dominant_det_term(beterm_worst),
        # Where the readable part of the run ends. Equals the final time
        # when SPD held throughout; earlier when it did not.
        "momentum_valid_until": (
            momentum[-1].get("time", float("nan"))
            if momentum else float("nan")
        ),
        "sigma_xy_rel_change": sigma_xy_plateau["rel_change"],
        "trace_frac_bulk": max((r[3] for r in trace), default=float("nan")),
        "trace_frac_interface": max((r[4] for r in trace), default=float("nan")),
        "trace_frac_blood": max((r[5] for r in trace), default=float("nan")),
        "max_div_phi": max((r[6] for r in trace), default=float("nan")),
        # THE VOLUME-WEIGHTED COMPANIONS. Parsed but NOT exported in the first
        # version of this change, so run 20260823-061943's numbers had to be
        # read out of the raw logs by hand -- the regex was extended and the
        # results dict was not, which is the same half-finished-wiring mistake
        # as the runner's prefix list. Taken as the max over time like their
        # maximum-valued siblings, so the two are directly comparable.
        **{
            key: max(
                (r[i] for r in trace if len(r) > i and not math.isnan(r[i])),
                default=float("nan"),
            )
            for key, i in (
                ("tr_bulk_vw", 8),
                ("tr_interface_vw", 9),
                ("tr_blood_vw", 10),
                ("frac_bulk_rms", 11),
                ("frac_interface_rms", 12),
                ("frac_blood_rms", 13),
                ("vol_frac_bulk_over1", 14),
                ("vol_frac_interface_over1", 15),
                ("vol_frac_blood_over1", 16),
                ("y_worst_blood", 17),
            )
        },
        # THE SAME TRACE FROM THE FLUX-CONSISTENT OPERATOR. Optional in the
        # regex: logs written before this diagnostic existed must still parse,
        # or every historical replay breaks at once.
        "tr_bulk_flux_consistent": max(
            (r[7] for r in trace if len(r) > 7 and not math.isnan(r[7])),
            default=float("nan"),
        ),
        "has_trace": bool(trace),
        "eulerian_path_ran": bool(euler),
        # Peak over the run of (interface excess strain)/(bulk excess strain).
        # Both are measured as `stretch - 1` so the ratio is of the DEPARTURES
        # from the undeformed state, not of two numbers that are both near 1
        # and would give a ratio near 1 whatever happened.
        "interface_over_bulk": max(
            (
                (i - 1.0) / (b - 1.0)
                for b, i, _ in phase
                if b - 1.0 > 1e-12
            ),
            default=float("nan"),
        ),
        "peak_bulk_stretch": max((b for b, _, _ in phase), default=float("nan")),
        "peak_interface_stretch": max((i for _, i, _ in phase), default=float("nan")),
        "has_phase_bins": bool(phase),
    }


def mean_cost(costs: list[float], lo: int, hi: int) -> float:
    window = costs[lo:hi]
    return sum(window) / len(window) if window else float("nan")


def main() -> int:
    pairs = discover_pairs()
    if not pairs:
        print("stage 3: no stage-3, diag_ or fix_ children in runs/ -- nothing "
              "to assess. This is a REFUSAL, not a pass: the runner invoked "
              "this gate, so it expected children to be here.")
        return 1

    results = {}
    for on, off, _ in pairs:
        for cid in (on, off):
            if cid is not None:
                results[cid] = read(cid)

    failures = []

    # Crashed children get ONE failure that names the crash, ahead of every
    # instrument check, so the report cannot bury the cause under a list of
    # missing-instrument messages that blame the binary.
    for cid, r in sorted(results.items()):
        if isinstance(r, dict) and r.get("crashed"):
            frame = r.get("crash_frame")
            failures.append(
                f"{cid}: THE CHILD CRASHED -- {r.get('crash_kind')}. It "
                f"advanced ZERO timesteps, so every other check on it would "
                f"report a missing instrument and blame the package. The "
                f"binary is not the issue; this child died before it ran"
                + (f", innermost solver frame: {frame}" if frame else "")
                + "."
            )
    print("stage 3 -- per-child measurements\n")
    header = (f"{'child':<18}{'ran':>5}{'steps':>7}{'s/step':>10}"
              f"{'minor':>9}{'SPD':>6}{'peakStretch':>13}{'iface/bulk':>12}")
    print(header)
    print("-" * len(header))
    for cid, r in results.items():
        if not r["ran"]:
            print(f"{cid:<18}{'NO':>5}")
            failures.append(f"{cid}: produced no solver log at all")
            continue
        cost = mean_cost(r["step_costs"], WARMUP_STEPS, len(r["step_costs"]))
        print(f"{cid:<18}{'yes':>5}{r['steps']:>7}{cost:>10.3f}"
              f"{r['min_leading_minor']:>9.5f}"
              f"{'LOST' if r['spd_lost'] else 'ok':>6}"
              f"{r['peak_stretch']:>13.5f}{r['interface_over_bulk']:>12.4f}")
        r["mean_s_per_step"] = cost

    print()
    for on, off, label in pairs:
        print(f"{label}")
        if off is None:
            # No partner, so no speedup. Said out loud, because a silently
            # omitted ratio reads the same as one that passed.
            print("  no eulerianSolid-off partner -- SPEEDUP NOT MEASURED for "
                  "this child. Health checks below still apply.\n")
            continue
        a, b = results[on], results[off]
        if not (a["ran"] and b["ran"]):
            failures.append(f"{label}: the on/off pair is incomplete, "
                            f"so no speedup can be measured")
            print("  incomplete pair -- no ratio\n")
            continue

        # MATCHED WINDOW. The OFF child runs a deliberately shorter window, so
        # comparing whole-run means would compare different parts of two
        # different curves. Both means are taken over the SAME step range.
        hi = min(len(a["step_costs"]), len(b["step_costs"]))
        if hi <= WARMUP_STEPS + 5:
            failures.append(
                f"{label}: only {hi} common steps, at or below the "
                f"{WARMUP_STEPS}-step warm-up plus five. There is no window in "
                f"which the two are comparable, so the ratio is not computed "
                f"rather than computed on noise."
            )
            print(f"  only {hi} common steps -- no ratio\n")
            continue
        on_cost = mean_cost(a["step_costs"], WARMUP_STEPS, hi)
        off_cost = mean_cost(b["step_costs"], WARMUP_STEPS, hi)
        speedup = off_cost / on_cost if on_cost > 0 else float("nan")
        print(f"  matched window       steps {WARMUP_STEPS}..{hi}")
        print(f"  eulerianSolid off    {off_cost:.3f} s/step")
        print(f"  eulerianSolid on     {on_cost:.3f} s/step")
        print(f"  speedup              {speedup:.2f}x   (threshold {MIN_SPEEDUP}x)")
        on_sk = mean_cost(a["skeleton_costs"], WARMUP_STEPS, hi)
        off_sk = mean_cost(b["skeleton_costs"], WARMUP_STEPS, hi)
        if math.isfinite(on_sk) and math.isfinite(off_sk):
            print(f"    of which skeleton  {off_sk:.3f} -> {on_sk:.3f} s/step "
                  f"({off_sk - on_sk:+.3f})")
            print(f"    everything else    {off_cost - off_sk:.3f} -> "
                  f"{on_cost - on_sk:.3f} s/step "
                  f"({(on_cost - on_sk) - (off_cost - off_sk):+.3f})")
            print(f"    The ON child also runs nOuterCorrectors 1 against 60: "
                  f"one velocity, no FSI to iterate. That is the reformulation, "
                  f"not a confound -- but the split above shows how much of the "
                  f"saving is the skeleton and how much is the outer loop.")
        print(f"  stage-0 cross-check  {STAGE0_S_PER_STEP} s/step on a different "
              f"VM; this run's OFF child measured {off_cost:.3f}")
        if math.isfinite(speedup) and speedup < MIN_SPEEDUP:
            failures.append(
                f"{label}: speedup {speedup:.2f}x is below {MIN_SPEEDUP}x. The "
                f"plan's decision point says STOP AND RE-PLAN rather than "
                f"launch stage 4 on hope."
            )
        print()

    # ---- health -----------------------------------------------------------
    for on, _, label in pairs:
        r = results[on]
        if not r["ran"]:
            continue
        if not r["eulerian_path_ran"]:
            failures.append(
                f"{on}: THE EULERIAN PATH NEVER RAN -- no PHASE_CLOT_EULERIAN "
                f"line. Every other number from this child describes the old "
                f"formulation and none of them are about stage 3."
            )
            continue
        if r["spd_lost"]:
            failures.append(f"{on}: be LOST POSITIVE-DEFINITENESS")
        if not r["has_phase_bins"]:
            failures.append(
                f"{on}: no PHASE_CLOT_BE_PHASE lines, so interface_over_bulk "
                f"could not be measured -- and that is the one measurement "
                f"this stage carries that stage 2 structurally could not make"
            )
        # VOLUME CONSERVATION. Gated at 1e-2, and the number is chosen rather
        # than inherited: stage 2 held 1e-6 on a 1-D slab where the flow was
        # exactly solenoidal and there was nothing for a collocated trace error
        # to bite on, and demanding that of a real 2-D interface flow would
        # fail every run for a reason nobody would act on. 1e-2 is a 1% volume
        # error, which is where dev(be) and the J in psi+ start to carry an
        # error a reader would care about. Run 20260817-234636 came in at 3.24
        # -- three hundred times this -- so nothing about this threshold is
        # what stands between that run and a pass.
        if math.isfinite(r["max_det_error"]) and r["max_det_error"] > MAX_DET_ERROR:
            failures.append(
                f"{on}: det(be) departed from 1 by {r['max_det_error']:.4g}, "
                f"above {MAX_DET_ERROR}. det(be) = 1 is exact for ANY isochoric "
                f"deformation at any magnitude, so this is volume being created, "
                f"not a large-strain allowance. The elastic stress is "
                f"G*dev(be), so every force this run applied is wrong by "
                f"whatever this error implies -- and nothing downstream of it, "
                f"including any apparent tearing, can be read as physical."
            )
        if math.isfinite(r["final_sweep_change"]) \
                and r["final_sweep_change"] > MAX_SWEEP_CHANGE:
            failures.append(
                f"{on}: the Picard iteration did not converge -- final "
                f"beSweepChange {r['final_sweep_change']:.3g}, above "
                f"{MAX_SWEEP_CHANGE}. The explicit stretch term is being lagged, "
                f"not resolved, so the be equation solved is not the one written."
            )
        if math.isfinite(r["interface_over_bulk"]):
            if r["interface_over_bulk"] >= 1.0:
                failures.append(
                    f"{on}: interface_over_bulk {r['interface_over_bulk']:.3f} "
                    f"is at or above 1 -- the diffuse interface is accumulating "
                    f"strain faster than the clot it belongs to. beRelax = "
                    f"q^2*beRelaxRate gives interface cells a relaxation time up "
                    f"to 100x blood's; re-weighting it is the cheap fix."
                )

    # ---- volume conservation and its cause --------------------------------
    # ---- is the clot actually held? -----------------------------------------
    # FIRST, because everything below it is meaningless if the answer is no. A
    # peak stretch of 4.63 describes an elastic deformation only if the material
    # is anchored; on a clot flowing at 88% of the free stream it is just the
    # accumulated strain of something being carried away, and run
    # 20260818-100608 was read the wrong way round for three sessions.
    print("is the clot held? clotU/U -- stage 2's slab gate measured 8.4e-4\n")
    for on, _, label in pairs:
        r = results.get(on, {})
        if not r.get("ran"):
            continue
        if not r.get("has_lock"):
            failures.append(
                f"{on}: NO PHASE_CLOT_LOCK lines. Whether the clot is anchored "
                f"is not optional information -- without it a peak stretch "
                f"cannot be told apart from a clot being washed downstream, "
                f"which is the mistake this diagnostic exists to prevent. The "
                f"solver is older than the diagnostic, or the log is truncated."
            )
            print(f"  {label}\n    NO PHASE_CLOT_LOCK lines\n")
            continue
        lost = r["clot_cells_initial"] - r["clot_cells_final"]
        frac = lost/r["clot_cells_initial"] if r["clot_cells_initial"] else 0.0
        print(f"  {label}")
        print(f"    lockRatio (final)    {r['lock_ratio']:.4f}"
              f"   (limit {MAX_LOCK_RATIO}; locked is ~0)")
        print(f"    clot |U| mean        {r['clot_u_mean']:.4e} m/s")
        print(f"    blood |U| mean       {r['blood_u_mean']:.4e} m/s")
        print(f"    bulk cells           {r['clot_cells_initial']:.0f} -> "
              f"{r['clot_cells_final']:.0f}  ({frac:+.1%})")
        # THE be TERM BUDGET, printed whenever the diagnostic ran. This is the
        # measurement that turns "which mechanism next" into "which term".
        if r.get("beterm_samples", 0):
            print(f"    be term budget at the deepest point of the failure "
                  f"(t = {r['beterm_time']:.4g} s, "
                  f"x = {r['beterm_x']*1e6:.1f} um, "
                  f"y = {r['beterm_y']*1e6:.1f} um, "
                  f"ind = {r['beterm_ind']:.3f})")
            print(f"      lmin {r['beterm_lmin_worst']:+.5f}   "
                  f"lmax {r['beterm_lmax']:.4f}   det {r['beterm_det']:.5f}")
            print(f"      d(det)/dt  advection {r['beterm_rate_advection']:+12.4g}"
                  f"   nonCons {r['beterm_rate_noncons']:+12.4g}")
            print(f"                 stretch   {r['beterm_rate_stretch']:+12.4g}"
                  f"   relax   {r['beterm_rate_relax']:+12.4g}")
            print(f"                 term sum  {r['beterm_rate_termsum']:+12.4g}"
                  f"   actual  {r['beterm_rate_actual']:+12.4g}")
            print(f"                 GAP       {r['beterm_rate_gap']:+12.4g}"
                  f"   <- if this dominates, no term is the culprit")
            print(f"      dominant: {r['beterm_dominant']}")

        # THE SATURATION READOUT. Printed for every child that has it, because
        # it is the number that reinterprets the one above.
        if not math.isnan(r.get("be_xy_rel_change", float("nan"))):
            print(f"    clot be_xy (final)   {r['clot_be_xy']:.4f}"
                  f"   sigma_xy {r['clot_sigma_xy']:.3f} Pa")
            print(f"    be_xy last quarter   {r['be_xy_late_start']:.4f} -> "
                  f"{r['be_xy_late_end']:.4f}"
                  f"   rel change {r['be_xy_rel_change']:.1%}"
                  f"   (plateau is < {MAX_BE_XY_REL_CHANGE:.0%})")

        if r["lock_ratio"] > MAX_LOCK_RATIO:
            failures.append(
                f"{on}: THE CLOT IS MOVING AT {r['lock_ratio']:.3f} OF THE BLOOD "
                f"-- clotU/U at the final step, "
                f"above {MAX_LOCK_RATIO}, where stage 2's slab gate measured "
                f"8.4e-4 on a formulation that locked. "
                f"READ THIS WITH be_xy AND NOT ALONE. Run 20260823-102340's "
                f"offline analysis -- validated against that run's own printed "
                f"accElastic to seven digits -- found the clot ANCHORED AND "
                f"SHEARING, with Ux spanning a factor of 19 from the wall row "
                f"to the crest and the wall carrying 12.32 Pa of elastic "
                f"traction against 4.62 Pa of viscous. A volume-mean over a "
                f"body reaching from the wall into the fast core CANNOT "
                f"separate 'held and deforming' from 'carried away', and "
                f"twelve hypotheses died against the assumption that it could. "
                f"This gate is sound only as a limit AT EQUILIBRIUM; if "
                f"be_xy has not plateaued the run has not reached one and this "
                f"line is not evidence of a defect."
            )
        # OUTCOME (2) OF THE PRE-REGISTERED SATURATION READING. Fires only when
        # the strain is still climbing at the END of the window, which is the
        # thing that would be a real defect: an anchored elastic solid cannot
        # accumulate shear strain without bound under a bounded load.
        rel = r.get("be_xy_rel_change", float("nan"))
        if not math.isnan(rel) and rel > MAX_BE_XY_REL_CHANGE:
            failures.append(
                f"{on}: THE SHEAR STRAIN HAS NOT SATURATED -- clot-mean be_xy "
                f"moved {rel:.1%} across the final quarter of the VALID "
                f"window (readable to t = {r['momentum_valid_until']:.4g} s) "
                f"({r['be_xy_late_start']:.4f} -> {r['be_xy_late_end']:.4f}), "
                f"above {MAX_BE_XY_REL_CHANGE:.0%}, with sigma_xy at "
                f"{r['clot_sigma_xy']:.3f} Pa. The run ended mid-transient. "
                f"Either the window is still too short -- which is a cost "
                f"question, not a physics one -- or the strain is unbounded, "
                f"which is a real defect. The bracket child at 10x G is what "
                f"separates those two; do not decide from this child alone."
            )
        if frac > 0.2:
            failures.append(
                f"{on}: the bulk clot lost {frac:.0%} of its cells "
                f"({r['clot_cells_initial']:.0f} -> {r['clot_cells_final']:.0f}). "
                f"A tear separates material, it does not delete it -- BUT "
                f"NEITHER DOES THIS, AND THE CELL COUNT CANNOT TELL YOU WHICH. "
                f"MEASURED, run 20260824-033102 offline: with growth off the "
                f"integral of (1-q)dV is conserved to SEVEN DIGITS through "
                f"t = 10 ms WHILE THIS COUNT FALLS 824 -> 699. Every one of "
                f"those cells is material that spread below the c < -0.9 "
                f"threshold, not material that was lost. The upstream face of "
                f"the clot holds a 6-8 um interface while the downstream face "
                f"reaches 49 um; an interface width is a material parameter "
                f"and has no direction, so that asymmetry is numerical "
                f"diffusion, at cell Peclet U*h/D = 83 against the ~2 needed. "
                f"IN THE UNANCHORED ERA a third loss mode took over after "
                f"t = 13 ms and was equally not a tear: the clot reached the "
                f"outlet (leading edge 249.5 um in a 250 um domain) and left. "
                f"THAT NUMBER DOES NOT APPLY TO AN ANCHORED CHILD and must "
                f"not be used to bound a window -- run 20260826-055524 "
                f"measured sd_deep's clot_xbar moving 7.5e-5 -> 8.889e-5 m, "
                f"13.9 um in 12 ms, leaving ~161 um and so about 139 ms of "
                f"headroom. Read clot_xbar, which prints every step, rather "
                f"than either figure. "
                f"TWO EARLIER ATTRIBUTIONS OF THIS MESSAGE WERE WRONG AND ARE "
                f"RECORDED SO THEY ARE NOT RE-DERIVED: it blamed 'smearing or "
                f"advection out of the bin' on the grounds that the clot "
                f"barely moves, then blamed the GROWTH SOURCE citing run "
                f"20260819-030512 -- but run 20260823-204058 falsified that "
                f"(alpha=0 lost 68%, alpha=1000x lost 74%) and the children "
                f"that trip this gate now run growth_rate_multiplier = 0. "
                f"DO NOT READ CLOT VOLUME OR CELL COUNT AS A MATERIAL MEASURE "
                f"until the phase field's cell Peclet number is addressed."
            )
        print()

    # ---- can the bond see anything at all? ----------------------------------
    #
    # THIS GATE EXISTS BECAUSE ITS ABSENCE COST THE PROJECT THE WHOLE EULERIAN
    # ADHESION STORY. clotWallAdhesion reads a registered stress field by name
    # and defaults to "sigmaD". Only solveSkeleton() writes sigmaD, and the
    # Eulerian path never calls solveSkeleton() -- it runs solveElasticStrain()
    # instead. So every Eulerian child computed a traction of exactly zero on
    # every face, no face could ever debond whatever the strength, and the log
    # said "peakTractionAny 0, utilisation 0", which is indistinguishable from
    # a bond under no load. Run 20260824-033102 read that for 20 ms.
    #
    # An inert failure criterion invalidates every statement about adhesive
    # detachment in the run, so it is a failure and not a note.
    print("can the bond see a stress?\n")
    for on, _, label in pairs:
        r = results.get(on, {})
        if not r.get("ran"):
            continue
        if not r.get("has_debond"):
            print(f"  {label}\n    NO PHASE_CLOT_DEBOND lines\n")
            continue
        found = r.get("sigma_found")
        print(f"  {label}")
        print(f"    sigma field          {r.get('sigma_field')}")
        print(f"    sigma found          {found}")
        print(f"    peak traction (any)  {r['peak_traction_any']:.6g} Pa")
        print(f"      normal (peeling)   {r['peak_traction_normal']:.6g} Pa")
        print(f"      shear              {r['peak_traction_shear']:.6g} Pa")
        print(
            f"      still rising by    {r['peak_traction_rel_change']:.1%}"
            f" across the final quarter"
        )
        print(f"    strength             {r['adhesive_strength']:.6g} Pa")
        print(
            f"    bonded clot faces    {r['bonded_clot_faces_initial']:.0f}"
            f" -> {r['bonded_clot_faces_final']:.0f}"
        )
        print(f"    debonded faces       {r['debonded_faces_final']:.0f}\n")
        if found == 0:
            failures.append(
                f"{on}: THE ADHESION CONDITION CANNOT SEE A STRESS -- "
                f"sigmaFound=0 for field '{r.get('sigma_field')}'. The traction "
                f"is identically zero on every face, so NO FACE CAN EVER "
                f"DEBOND at any strength and every adhesive statement in this "
                f"run is vacuous. On the Eulerian path the field must be "
                f"sigmaElastic, which solveElasticStrain() publishes; sigmaD "
                f"is written only by solveSkeleton(), which that path never "
                f"calls."
            )
        elif found is None:
            failures.append(
                f"{on}: PHASE_CLOT_DEBOND has no sigmaFound field. The solver "
                f"predates the diagnostic added after run 20260824-033102, so "
                f"this log CANNOT distinguish 'the bond carried no load' from "
                f"'the bond was never shown a stress'. Rebuild before reading "
                f"any adhesion result from it."
            )
        elif (
            r["peak_traction_rel_change"] == r["peak_traction_rel_change"]
            and abs(r["peak_traction_rel_change"]) > MAX_BE_XY_REL_CHANGE
        ):
            failures.append(
                f"{on}: THE WALL TRACTION HAS NOT SATURATED -- peakTraction "
                f"{'ROSE' if r['peak_traction_rel_change'] > 0 else 'COLLAPSED'}"
                f" {abs(r['peak_traction_rel_change']):.1%} across the final "
                f"quarter, above {MAX_BE_XY_REL_CHANGE:.0%}, ending at "
                f"{r['peak_traction_final']:.6g} Pa with a run maximum of "
                f"{r['peak_traction']:.6g} Pa. A COLLAPSE MEANS THE CLOT LEFT "
                f"THE WALL, not that the load eased: read bondedClotFaces, and "
                f"take the run MAXIMUM as the required strength rather than the "
                f"final value. A RISE means the window ended mid-load and the "
                f"maximum is a lower bound. This series reports the "
                f"traction an UNBREAKABLE bond carries BECAUSE THAT IS THE "
                f"REQUIRED ADHESIVE STRENGTH at the operating point -- so a "
                f"traction still climbing at the end is a LOWER BOUND and must "
                f"never be quoted as the required strength. Run "
                f"20260824-195730 ended at 148.93 Pa while rising 17.8% per "
                f"quarter. Lengthen the window, but stop before the clot "
                f"reaches the outlet. THE 'about 13 ms' THIS MESSAGE USED TO "
                f"QUOTE WAS FROM THE UNANCHORED ERA, when the clot travelled "
                f"at 1.16x U, and it is withdrawn: run 20260826-055524 "
                f"measured an ANCHORED clot_xbar moving 13.9 um in 12 ms, "
                f"which leaves ~161 um and about 139 ms of headroom. A longer "
                f"window is affordable. Bound it on clot_xbar, which prints "
                f"every step, not on a figure from a different anchor."
            )
        elif r["peak_traction_any"] <= 0 and r["bonded_clot_faces_final"] > 0:
            failures.append(
                f"{on}: the bond sees its stress field but the traction is "
                f"still exactly zero over the whole run, with "
                f"{r['bonded_clot_faces_final']:.0f} faces bonded. That is a "
                f"stress field that exists and is never written, which is the "
                f"same defect one layer further in. Do not read utilisation."
            )

    # ---- does the clot stay on the wound? -----------------------------------
    #
    # THE CENTRAL QUESTION OF THE eulerian_anchor SERIES, and the one that had
    # no instrument at all until 2026-08-24.
    print(
        "does the clot stay on the wound, and is anything free sitting on an "
        "intact bond?\n"
    )
    for on, _, label in pairs:
        r = results.get(on, {})
        if not r.get("ran"):
            continue
        if not r.get("has_wound"):
            # NOT SILENT. A run without the instrument cannot be read on this
            # question, and saying nothing would let it pass as if it had.
            print(f"  {on}: no PHASE_CLOT_WOUND lines (solver predates it)\n")
            failures.append(
                f"{on}: PHASE_CLOT_WOUND is absent, so THIS LOG CANNOT SAY "
                f"WHETHER THE CLOT STAYED ON ITS WOUND. That is the question "
                f"the series exists to answer, and run 20260824-220555 lost "
                f"99.5% of the material over its injury -- 1093 cells to 6 -- "
                f"without a single log line reporting it. Rebuild the solver "
                f"before reading any anchoring result from this run."
            )
            continue

        print(f"  {on} ({label})")
        print(
            f"    coverage             {r['coverage_initial']:.3f}"
            f" -> {r['coverage_final']:.3f}  (min {r['coverage_min']:.3f})"
        )
        print(
            f"    max height           {r['max_height_initial']:.0f}"
            f" -> {r['max_height_final']:.0f} cells"
        )
        print(
            f"    cells over injury    "
            f"{r['cells_over_injury_initial']:.0f}"
            f" -> {r['cells_over_injury_final']:.0f}"
        )
        print(
            f"    released             {r['released_final']:.0f}"
            f"  (largest one-step jump {r['release_biggest_step_jump']:.0f},"
            f" first at {r['release_first_time']:.6g} s)"
        )
        print(
            f"    releasedOnBond max   {r['released_on_bond_max']:.0f}"
            f"   supportLimit {r['support_limit']:.6g}"
        )
        if r.get("has_bondhold"):
            print(
                f"    bondHold             {r['held_cells_max']:.0f} cells max,"
                f" dilation {r['hold_dilation']:.0f}"
            )
        print()

        # IS THIS CHILD CLAIMING AN ANCHOR? Read from the run's own
        # diagnostics, not from its child_id -- a name is not a configuration,
        # and the whole sigmaD failure was a case that said one thing and did
        # another.
        sl = r["support_limit"]
        dil = r["hold_dilation"]
        anchored = (sl == sl and sl > 0) or (dil == dil and dil >= 0)

        if not anchored:
            # A CONTROL. It is SUPPOSED to reproduce the failure, so the
            # defect gates below are not applied to it -- but it is required
            # to reproduce, and that requirement is gated. A control that
            # comes back healthy means the offline reconstruction this whole
            # series is built on was wrong, and nothing downstream of it
            # survives.
            reproduced = (
                (r["released_on_bond_max"] == r["released_on_bond_max"]
                 and r["released_on_bond_max"] > 0)
                or (r["coverage_final"] == r["coverage_final"]
                    and r["coverage_final"] < MIN_FINAL_COVERAGE)
            )
            if reproduced:
                print(
                    f"    -> reproduced the failure, as the control must\n"
                )
            else:
                failures.append(
                    f"{on}: THE CONTROL DID NOT REPRODUCE THE FAILURE. It "
                    f"carries no anchor (supportLimit {sl:.6g}, "
                    f"bondHoldDilation {dil:.6g}) and yet ended with coverage "
                    f"{r['coverage_final']:.3f} and releasedOnBond "
                    f"{r['released_on_bond_max']:.0f}. Run 20260824-220555 "
                    f"with this configuration ended at coverage 0.15 with 20 "
                    f"cells released on intact bonds. If that does not "
                    f"reproduce here, the offline reconstruction the "
                    f"eulerian_anchor series is built on is WRONG and every "
                    f"other child in it is uninterpretable -- including the "
                    f"retraction of the momentum-penalty recommendation. Stop "
                    f"and re-derive before reading anything else."
                )
            continue

        # (1) THE NUMBER THAT MUST BE ZERO.
        rob = r["released_on_bond_max"]
        if rob == rob and rob > 0:
            failures.append(
                f"{on}: MATERIAL SITTING ON AN INTACT BOND WAS DECLARED FREE "
                f"-- releasedOnBond reached {rob:.0f}. A cell whose wall face "
                f"is still bonded cannot be unsupported; this is a direct "
                f"self-contradiction in the release test, not a marginal "
                f"call. Run 20260824-220555 had twenty such cells and they "
                f"were then advected at Ubody = 9.14e-3 m/s, 16x the fluid "
                f"speed in their own row, which is how the wound was vacated "
                f"while every bond held."
            )

        # (2) THE STEP FUNCTION.
        jump = r["release_biggest_step_jump"]
        if jump == jump and jump > MAX_RELEASE_STEP_JUMP:
            failures.append(
                f"{on}: THE RELEASE IS A STEP FUNCTION -- {jump:.0f} cells "
                f"went free in a single step, above the {MAX_RELEASE_STEP_JUMP} "
                f"allowed. Physical detachment is progressive. A jump of this "
                f"size is the signature of a CONNECTED-COMPONENT TEST ON A "
                f"LEVEL SET snapping: a diffuse interface has no topology, and "
                f"the c=0 contour is an arbitrary cut through a smeared field. "
                f"Run 20260824-220555 released 1278 cells -- 98% of the clot "
                f"-- in one step at t = 8.4931e-3 s, hanging until then on a "
                f"one-cell-tall filament. Raise releaseSupportLimit so the "
                f"support test conducts through material rather than through "
                f"the cut, and read an_sup05/an_sup09."
            )

        # (3) DID IT LEAVE THE WOUND AT ALL.
        #
        # GATED ON debondedFaces == 0, WHICH IS THE WHOLE POINT. Losing the
        # wound while every bond is intact is the defect; losing it after the
        # bond has failed is the physics the series is trying to produce, and
        # a gate that could not tell those apart would fail an_break for
        # succeeding. The ordering check below is what polices the other
        # direction -- leaving BEFORE the bond broke.
        cov = r["coverage_final"]
        deb = r["debonded_faces_final"]
        intact = not (deb == deb and deb > 0)
        if cov == cov and cov < MIN_FINAL_COVERAGE and intact:
            failures.append(
                f"{on}: THE CLOT LEFT THE WOUND -- coverage of the injured "
                f"wall fell to {cov:.3f} (min {r['coverage_min']:.3f}), below "
                f"{MIN_FINAL_COVERAGE}, with {r['debonded_faces_final']:.0f} "
                f"faces debonded. The injury footprint is LATCHED and does not "
                f"move, so this is material walking off a fixed wound and onto "
                f"intact endothelium where nothing can ever adhere. If "
                f"debondedFaces is 0 the bond is not the cause and the anchor "
                f"is: read cellsOverInjury and maxHeight to separate "
                f"translation from diffusion."
            )

        # (4) IS THE HOLD BOUNDARY CUTTING LIVE INTERFACE?
        mh = r["max_height_final"]
        if (
            dil == dil
            and dil >= 0
            and mh == mh
            and mh <= dil + 1
        ):
            failures.append(
                f"{on}: THE HOLD BOUNDARY IS CUTTING LIVE INTERFACE -- the "
                f"clot over the wound is {mh:.0f} cells tall against a "
                f"bondHoldDilation of {dil:.0f}. The anchored layer is sized "
                f"on the assumption that its top sits deep in bulk clot where "
                f"grad(c) is negligible; once the clot is no thicker than the "
                f"layer, the carrier discontinuity lands on the interface and "
                f"shears it, which is the failure releasedMaskDilation was "
                f"enlarged 6 -> 15 to avoid. Any holding result from this "
                f"child must be reported with that stated, not quoted plain."
            )

        # (5) HAS THE HOLD FROZEN THE CLOT?
        #
        # The failure mode this gate exists for is the one the taper was built
        # to avoid, and it looks like SUCCESS on every other gate here:
        # coverage 1.0, height unchanged, nothing released, because nothing
        # moved at all.
        bexy = r.get("clot_be_xy", float("nan"))
        if bexy == bexy and abs(bexy) < FROZEN_BE_XY:
            failures.append(
                f"{on}: THE HOLD FROZE THE CLOT -- clot-mean be_xy is "
                f"{bexy:.3g}, below {FROZEN_BE_XY}, with a hold of "
                f"{r['held_weight_max']:.4g} weight over "
                f"{r['held_cells_max']:.0f} cells. This child will pass every "
                f"coverage and height gate above, because a body that cannot "
                f"move cannot leave. It is NOT a success. Dr He Li's "
                f"specification is small and smooth deformation of the core, "
                f"and a rigid core violates it as surely as a severed one. "
                f"Reduce bondHoldPenalty or bondHoldDilation; do not report "
                f"this as the anchor working."
            )

        # (6) IS THE CORE STILL EMBOLISABLE?
        if not r.get("has_core_exempt_diag"):
            failures.append(
                f"{on}: the RELEASED line carries no coreExempted field, so "
                f"THIS SOLVER PREDATES releaseCoreExempt and a core cell can "
                f"still be handed to Ubody by a connectivity pinch. Dr He Li, "
                f"2026-08-25: the core 'does not slide or detach or rupture'. "
                f"Rebuild before reading any core result from this run."
            )

    # ---- did the clot leave before the bond broke? --------------------------
    #
    # THE ORDER OF EVENTS IS THE RESULT for any child whose bond can fail. A
    # run that ends with a detached clot looks like the physics Dr Li asked
    # for whether or not adhesion had anything to do with it.
    print("for breakable bonds: did the clot leave BEFORE the bond broke?\n")
    for on, _, label in pairs:
        r = results.get(on, {})
        if not r.get("ran") or not r.get("has_wound"):
            continue
        strength = r.get("adhesive_strength", float("nan"))
        # Unbreakable children are controls and are exempt BY CONSTRUCTION,
        # not by name: at 1e9 Pa nothing can debond, so the ordering question
        # is not defined for them.
        if not (strength == strength) or strength >= UNBREAKABLE_PA:
            continue
        lbb = r.get("left_before_break")
        print(
            f"  {on}: first debond {r['first_debond_time']:.6g} s, "
            f"coverage<0.5 {r['coverage_half_time']:.6g} s, "
            f"first release {r['first_release_time']:.6g} s"
        )
        if lbb:
            failures.append(
                f"{on}: THE CLOT LEFT BEFORE THE BOND BROKE. Departure "
                f"(coverage<0.5 at {r['coverage_half_time']:.6g} s or first "
                f"release at {r['first_release_time']:.6g} s) precedes the "
                f"first debond at {r['first_debond_time']:.6g} s. The required "
                f"sequence is LOAD, then DEBOND, then DEPARTURE. Any other "
                f"order means the anchor still leaks, and a detachment "
                f"produced this way is not adhesive failure however much it "
                f"looks like one in ParaView."
            )
    print()

    # ---- is the platelet field bounded? -------------------------------------
    print("is w bounded? the model bound is ~1.1\n")
    for on, _, label in pairs:
        r = results.get(on, {})
        if not r.get("ran"):
            continue
        if not r.get("has_platelet"):
            failures.append(
                f"{on}: NO PHASE_CLOT_PLATELET lines. w is the field that "
                f"diverged unnoticed in run 20260818-232055 while every other "
                f"gate passed; a log without it cannot be cleared. The solver "
                f"predates the diagnostic, or the log is truncated."
            )
            print(f"  {label}\n    NO PHASE_CLOT_PLATELET lines\n")
            continue
        print(f"  {label}")
        print(f"    w max (whole run)    {r['w_max']:.4g}"
              f"   (limit {MAX_W}; bound is ~1.1)")
        print(f"    w min (whole run)    {r['w_min']:.4g}")
        # |Vn| max IS A WHOLE-FIELD MAXIMUM, AND UNDER THE REACTION CLOSURE
        # IT IS NOT THE INTERFACE SPEED.
        #
        # Gradient closure: Vn = alpha*(n & grad w), so Vn fell to ~0 away
        # from the interface -- MEASURED on run 20260901-200203 g04, bulk
        # 6.98e-07 against 1.68e-05 at the interface, a ratio of 0.042.
        #
        # Reaction closure: Vn = (alpha/D)*kEff*w, and kEff and w are defined
        # EVERYWHERE, so Vn is finite in the whole domain. MEASURED on run
        # 20260902-231316 g04: bulk 3.7476e-05 against 3.5192e-05 at the
        # interface -- ratio 1.065, i.e. the maximum is now attained in the
        # FREE STREAM, not on the clot. |Vn| max is exactly (alpha/D)*k0
        # wherever w = 1, which is why it reads 3.75e-05 to five figures.
        #
        # This is faithful to Capek (4.6): the source is
        # (alpha/D) k w |grad phi_eps(c)|, and the solver applies the
        # |grad phi_eps(c)| localiser AT THE POINT OF USE (-Vn*phaseGradient).
        # Vn is a FACTOR of the source, not the source. Harmless, but do not
        # read this number as an interface velocity, and do not compare it
        # across closures -- it is a different quantity in each.
        print(f"    |Vn| max             {r['vn_mag_max']:.4e} m/s"
              f"   (WHOLE FIELD; under 'reaction' this is the BULK value"
              f" (alpha/D)*k0, not the interface speed)")
        print(f"    cells over bound     {r['w_over_bound']:.0f}")
        if r["w_max"] > MAX_W:
            failures.append(
                f"{on}: THE PLATELET FIELD DIVERGED -- w reached "
                f"{r['w_max']:.4g} against a model bound of ~1.1. Vn enters "
                f"the phase field as -Vn*phaseGradient, and w is a factor of "
                f"Vn under BOTH closures -- alpha*(n & grad w) for 'gradient', "
                f"(alpha/D)*kEff*w for 'reaction' -- so the interface is being "
                f"driven by a field that is no longer a concentration. Any "
                f"growth, tearing or detachment in this child is an artefact "
                f"of that divergence and must not be read as physics. "
                f"(|Vn| max was {r['vn_mag_max']:.3e} m/s, but that is a "
                f"WHOLE-FIELD maximum attained in the free stream under the "
                f"reaction closure -- it is not the interface speed and is "
                f"not the evidence here; w exceeding its bound is.)"
            )
        print()

    # ---- the platelet DISCRETISATION gates, added 2026-09-01 ---------------
    #
    # Run 20260901-110348 child g20 died with SIGFPE at 17.482 ms and every
    # existing gate in this file was clean until the field was already
    # destroyed. The cause was an explicit-diffusion stability violation in
    # the platelet content equation -- arithmetic face interpolation of a
    # diffusivity spanning four decades -- and the two numbers below are what
    # would have caught it on the first step it mattered.
    print("platelet discretisation: is the scheme stable, and is it bounded?\n")
    for on, _, label in pairs:
        r = results.get(on, {})
        if not r.get("ran"):
            continue
        print(f"  {label}")

        # THE LEGACY BRANCH CANNOT EMIT THESE, AND THAT IS NOT A DEFECT.
        # PHASE_CLOT_PLATELET_FO / _LED / _LIM are all written inside the
        # CONTENT branch of solvePlatelets(). gate_a_advect and the five plt_*
        # probes pin plateletContentFormulation to no ON PURPOSE -- they are
        # the w-branch controls the content formulation is measured against.
        # Run 20260902-104340 failed gate_a_advect with "this binary predates
        # the harmonic fix" for doing exactly its job.
        if r.get("content_branch") == "no":
            print("    legacy w branch (plateletContentFormulation no) -- the "
                  "content-branch instruments do not exist here, by design\n")
            continue

        if not r.get("has_fo"):
            failures.append(
                f"{on}: NO PHASE_CLOT_PLATELET_FO lines. This binary predates "
                f"the 2026-09-01 harmonic-diffusivity fix, so THIS RUN DOES "
                f"NOT TEST IT. Check that the package that was uploaded is the "
                f"one that was built -- a new notebook against an old build "
                f"produces exactly this."
            )
            print("    NO PHASE_CLOT_PLATELET_FO lines -- pre-fix binary\n")
            continue

        fo = r["platelet_fo_max"]
        print(f"    plateletFo max       {fo:.4e}"
              f"   (gate {MAX_PLATELET_FO}; unstable at 2)   <-- THE GATE")
        print(f"      reached at t =     {r['platelet_fo_at']:.6g} s")
        if fo > MAX_PLATELET_FO:
            failures.append(
                f"{on}: THE PLATELET DIFFUSION IS NOT STABLE -- plateletFo "
                f"reached {fo:.4g} at t = {r['platelet_fo_at']:.6g} s against "
                f"an explicit limit of 2. The content field is being "
                f"integrated past its own stability boundary and anything "
                f"downstream of it -- Vn, the interface, the height -- is an "
                f"artefact. With the harmonic face mean this number is "
                f"~2.4e-03 and flat; a value near 1 means the harmonic "
                f"interpolation is NOT in this binary or something else has "
                f"become stiff."
            )

        if not r.get("has_led"):
            print("    NO PHASE_CLOT_PLATELET_LED lines\n")
            continue

        if not r.get("has_wp_bounds"):
            failures.append(
                f"{on}: the PHASE_CLOT_PLATELET_LED lines carry no WpMin/WpMax "
                f"field. This binary predates the 2026-09-02 more-upwind "
                f"limiter, so it cannot be used to certify that fix."
            )
            print("    NO WpMin/WpMax fields -- pre-2026-09-02 binary\n")
            continue

        wpmin, wpmax = r["wp_min"], r["wp_max"]
        print(f"    Wp min (whole run)   {wpmin: .6e}"
              f"   (gate {MIN_WP:.0e})   <-- THE GATE")
        print(f"    Wp max (whole run)   {wpmax:.9f}"
              f"   (gate {MAX_WP:.6f})   <-- THE GATE")
        if r.get("has_lim"):
            print(f"    faces upwinded max   {r['faces_upwinded_max']:.0f}"
                  f"   (0 means the more-upwind limiter never engaged)")
            # THE QUESTION RUN 20260902-104340 LEFT OPEN. The override fired on
            # 1600 faces with Wp == q to the last bit, which the conditioning
            # of q = 0.5*(c+1) explains only if those faces are deep in the
            # clot, where both fields are flat and the weight cannot move
            # material. Near zero here confirms it; large means the override
            # must be gated on the change in face VALUE, not in the weight.
            if math.isfinite(r.get("faces_upwinded_interface_max",
                                   float("nan"))):
                print(f"      of which at interface "
                      f"{r['faces_upwinded_interface_max']:.0f}"
                      f"   [reported, not gated]")
        print(f"    [reported, not gated] LED overshoot {r['led_over_max']:.3e}"
              f", undershoot {r['led_under_max']:.3e}"
              f", steps {r['led_viol_steps']:.0f}")

        if wpmin < MIN_WP:
            failures.append(
                f"{on}: NEGATIVE PLATELET CONTENT -- Wp reached {wpmin:.4e} at "
                f"t = {r['wp_min_at']:.6g} s. The Wp equation has no source "
                f"and its adhesion term is an IMPLICIT sink with a positive "
                f"coefficient, which cannot carry a positive value through "
                f"zero; the harmonic diffusion is a convex combination and "
                f"cannot either. Only the convection can, and that is the "
                f"dispersive undershoot of a downwind-biased face value -- "
                f"c's vanLeer limiter applied to a steep Wp front in a region "
                f"where c itself is smooth. Run 20260901-200203 measured "
                f"exactly this at -5.32e-05 in g30 from 25 ms. If this fires "
                f"WITH the more-upwind limiter active (facesUpwinded > 0), "
                f"the limiter combination is not sufficient and the next step "
                f"is a Zalesak-limited antidiffusive flux, NOT a clamp."
            )
        if wpmax > MAX_WP:
            failures.append(
                f"{on}: Wp EXCEEDED ITS INLET VALUE -- reached {wpmax:.9f} "
                f"against a bound of 1. There is no source in this equation, "
                f"so the convection manufactured platelet content."
            )
        print()

    # ------------------------------------------------------------------
    # THE GROWTH CLOSURE. Added 2026-09-02.
    #
    # Capek (2020) states the growth law twice: (3.5) for the SHARP interface,
    # u_i = alpha*grad(w), and (4.6) for the DIFFUSE one,
    # alpha*D^-1*k*w*|grad phi_eps|. They agree only where the Robin condition
    # D dn(w) = k w is resolved. This solver ran (3.5) inside a diffuse-
    # interface method until 2026-09-02, and run 20260901-200203 measured the
    # consequence directly: the Robin residual, which is 1 by construction and
    # in which alpha cancels, came out at a median of 0.092 (g04) and 0.072
    # (g40) over three decades of spread. The clot was growing at a rate
    # unrelated to the platelets it consumed.
    #
    # These two checks are what make that unrepeatable.
    # ------------------------------------------------------------------
    print("growth closure: is the clot made of the platelets it absorbs?\n")
    for on, _, label in pairs:
        r = results.get(on, {})
        if not r.get("ran"):
            continue
        print(f"  {label}")

        if r.get("has_growth_support") and math.isfinite(
            r["growth_below_q005_max"]
        ):
            print(f"    growth from q<0.05   "
                  f"{100 * r['growth_below_q005_max']:.3f}%"
                  f"   (gate {100 * MAX_GROWTH_BELOW_Q005:.0f}%)"
                  f"   <-- THE GATE")
            print(f"    growth from q<0.15   "
                  f"{100 * r['growth_below_q015_max']:.3f}%   [reported]")
            print(f"    growth-weighted q    "
                  f"{r['growth_mean_q_min']:.4f}"
                  f"   (0.5 is the c = 0 surface)   [reported]")
            if r["growth_below_q005_max"] > MAX_GROWTH_BELOW_Q005:
                failures.append(
                    f"{on}: THE GROWTH IS BEING DRIVEN BY THE CLOT INTERIOR -- "
                    f"{100 * r['growth_below_q005_max']:.2f}% of it comes from "
                    f"cells with q < 0.05 at t = {r['growth_support_at']:.6g} "
                    f"s, against a limit of "
                    f"{100 * MAX_GROWTH_BELOW_Q005:.0f}%. There w = Wp/q is a "
                    f"ratio of two small numbers and the source says platelet "
                    f"values in the thrombus 'have no physical meaning'. The "
                    f"|grad c| localiser is what is supposed to hold this near "
                    f"2%; if it has risen, either the interface has smeared or "
                    f"w has drifted in the clot interior -- check wMax and the "
                    f"interface width before blaming the closure."
                )

        # Run 20260902-104340 found c and Wp assembling different matrices.
        # WpDevMax was in EVERY content-branch log at the time and nothing
        # printed it outside the gate-A block, so the drift was invisible in
        # the sweep. It is printed here for every content child now.
        #
        # Reported, NOT gated, and the reason matters: in a growing child
        # w != 1, so Wp = q*w differs from q as PHYSICS. |Wp - q| there mixes
        # the real deviation with the transport error and no threshold on it
        # is defensible. The exact test needs a child where w == 1 is the
        # true answer, which is what gate_a_exact is for. This line is a
        # tripwire to look at, not a criterion.
        if (
            r.get("content_branch") != "no"
            and r.get("has_abs_platelet")
            and math.isfinite(r.get("wp_dev_max", float("nan")))
        ):
            early, late = r.get("wp_dev_early"), r.get("wp_dev_late")
            trend = ""
            if (
                early is not None and late is not None
                and math.isfinite(early) and math.isfinite(late)
                and early > 0
            ):
                trend = f"   early->late x{late / early:.3g}"
            print(f"    |Wp - q| max         "
                  f"{r['wp_dev_max']:.3e}{trend}   [reported, not a gate:"
                  f" w != 1 here, so this is physics + error]")

        if r.get("has_phase_bounds"):
            print(f"    c in [{r['c_min']:+.9f}, {r['c_max']:+.9f}]"
                  f"   (must be [-1, 1], slack {MAX_C_VIOLATION:.0e})"
                  f"   <-- THE GATE")
            viol = max(r["c_under_max"], r["c_over_max"])
            if math.isfinite(viol) and viol > MAX_C_VIOLATION:
                failures.append(
                    f"{on}: THE PHASE FIELD LEFT [-1, 1] -- c reached "
                    f"{r['c_min']:.9f} / {r['c_max']:.9f} at "
                    f"t = {r['c_worst_at']:.6g} s, a violation of {viol:.3e} "
                    f"against {MAX_C_VIOLATION:.0e}. q = 0.5(c+1) is a VOLUME "
                    f"FRACTION: past this bound max(q, qFloor) silently clamps "
                    f"a negative q and chi = max(1-q, 0) exceeds 1, so the "
                    f"Brinkman drag is built from an impossible solid fraction "
                    f"and every mechanical number downstream is an artefact. "
                    f"Run 20260901-200203 measured ZERO violation, so this is "
                    f"new -- the growth source is 1.5-4.3x stronger under the "
                    f"reaction closure and that is the first thing to suspect."
                )

        if not r.get("has_stoich"):
            failures.append(
                f"{on}: NO PHASE_CLOT_GROWTH_STOICH lines. This binary "
                f"predates the 2026-09-02 reaction growth closure, so THIS RUN "
                f"DOES NOT TEST IT. Check that the package uploaded is the one "
                f"that was built."
            )
            print("    NO PHASE_CLOT_GROWTH_STOICH lines -- pre-fix binary\n")
            continue

        closure = r["growth_closure"]
        print(f"    growthClosure        {closure}")
        print(f"    depletion model      {r['depletion_model']}")

        if r.get("has_label_drift") and math.isfinite(
            r["label_in_blood_max"]
        ):
            print(f"    [reported] label in blood  "
                  f"{100 * r['label_in_blood_max']:.3f}%"
                  f"   (core/shell label sitting where there is no clot)")

        if r.get("has_damkohler"):
            # REPORTED, NEVER GATED. Da > 1 is a statement about the mesh and
            # the shear, not a defect: it says the reaction closure's declared
            # limit has been reached and interfaceDepletionModel is the lever.
            print(f"    [reported] Da interface max  "
                  f"{r['da_interface_max']:.4g}"
                  f"   (depletion layer {r['layer_cells_min']:.3g} cells at "
                  f"its thinnest)")
            if r["da_interface_max"] > 1.0:
                print("      NOTE: Da > 1 somewhere -- the depletion layer is "
                      "thinner than a cell there, so the reaction closure "
                      "OVER-reads growth in those cells. Not a failure; the "
                      "lever is interfaceDepletionModel seriesResistance.")

        if closure != "reaction":
            print(f"    stoichiometry not gated -- closure is '{closure}', "
                  f"and only the reaction form makes growth and uptake the "
                  f"same process\n")
            continue

        sn_min, sn_max = r["stoich_norm_min"], r["stoich_norm_max"]

        # k = 0 children absorb nothing, so the ratio is 0/0 and there is no
        # stoichiometry to check. Say that, rather than printing nan under a
        # gate marker -- an unevaluable check displayed as a passing one is
        # how a gate quietly stops meaning anything.
        if not math.isfinite(sn_min):
            print("    stoichiometry undefined -- plateletUptake is "
                  "identically zero (k = 0), so there is no absorbed "
                  "platelet to account for")
            if r.get("has_vn_closure"):
                vres0 = r["vn_closure_resid_max"]
                print(f"    VnClosureResidual    {vres0:.4e}"
                      f"   (gate {MAX_VN_CLOSURE_RESIDUAL:.0e})   <-- THE GATE")
                if (
                    math.isfinite(vres0)
                    and vres0 > MAX_VN_CLOSURE_RESIDUAL
                ):
                    failures.append(
                        f"{on}: THE Vn THE PHASE EQUATION USED IS NOT THE "
                        f"REACTION CLOSURE -- max|Vn - (alpha/D)*kEff*w|/"
                        f"max|Vn| = {vres0:.4e} against "
                        f"{MAX_VN_CLOSURE_RESIDUAL:.0e}, with k = 0. Here the "
                        f"closure must return exactly zero, so any residual "
                        f"at all names a term that should not be in Vn."
                    )
            print()
            continue

        print(f"    stoichNorm min       {sn_min:.8f}"
              f"   (gate {MIN_STOICH_NORM})   <-- THE GATE")
        print(f"    stoichNorm max       {sn_max:.8f}"
              f"   (gate {MAX_STOICH_NORM:.9f})   <-- THE GATE")
        print(f"      worst at t =       {r['stoich_norm_at']:.6g} s")

        if math.isfinite(sn_min) and sn_min < MIN_STOICH_NORM:
            failures.append(
                f"{on}: GROWTH IS NOT MADE OF THE PLATELETS IT ABSORBS -- "
                f"stoichNorm fell to {sn_min:.6f} at "
                f"t = {r['stoich_norm_at']:.6g} s, below {MIN_STOICH_NORM}. "
                f"Under the reaction closure the clot volume created per "
                f"platelet consumed is alpha/D EXACTLY and POINTWISE, so this "
                f"ratio is 1 up to the gradFloor softening of phaseGradient, "
                f"which is worth 7.9e-03 at its worst and is one-signed. A "
                f"departure larger than that means growth and the sink are no "
                f"longer reading the same kEff, delta_Gamma or w -- check that "
                f"the sink still uses adhesionSpeedEff_ and not adhesionSpeed_."
            )
        if math.isfinite(sn_max) and sn_max > MAX_STOICH_NORM:
            failures.append(
                f"{on}: stoichNorm reached {sn_max:.9f}, ABOVE 1. The only "
                f"admissible discrepancy is the gradFloor softening, and that "
                f"is strictly one-signed downwards -- phaseGradient < |grad c| "
                f"always. A ratio above 1 is a wiring fault, not a tolerance "
                f"question."
            )

        if not r.get("has_vn_closure"):
            print("    no VnClosureResidual field -- identity not certified\n")
            continue

        vres = r["vn_closure_resid_max"]
        print(f"    VnClosureResidual    {vres:.4e}"
              f"   (gate {MAX_VN_CLOSURE_RESIDUAL:.0e})   <-- THE GATE")
        if math.isfinite(vres) and vres > MAX_VN_CLOSURE_RESIDUAL:
            failures.append(
                f"{on}: THE Vn THE PHASE EQUATION USED IS NOT THE REACTION "
                f"CLOSURE -- max|Vn - (alpha/D)*kEff*w|/max|Vn| = {vres:.4e} "
                f"over the interface, against {MAX_VN_CLOSURE_RESIDUAL:.0e}. "
                f"The dictionary says growthClosure reaction, and under that "
                f"branch this identity holds to ~1e-16 by construction. A "
                f"finite residual means a stray term is being added to Vn, or "
                f"alpha/D is not what the dictionary says, or the gradient "
                f"branch is the one that actually ran."
            )
        print()

    # ---- the platelet-content acceptance gates -----------------------------
    for on, _, label in pairs:
        if on not in GATE_A_CHILDREN + GATE_A_EXACT_CHILDREN + GATE_E_CHILDREN:
            continue
        r = results.get(on, {})
        if not r.get("ran"):
            continue
        if not r.get("has_platelet"):
            failures.append(
                f"{on}: NO PHASE_CLOT_PLATELET lines -- this child exists only "
                f"to read them."
            )
            continue

        if on in GATE_A_EXACT_CHILDREN:
            if not r.get("has_abs_platelet"):
                failures.append(
                    f"{on}: the log has no WpDevMax field. This gate is "
                    f"written on the ABSOLUTE deviation |Wp - q| and cannot be "
                    f"evaluated from max|w - 1|, which divides by a q reaching "
                    f"2e-5. An old solver binary produced this log."
                )
                continue

            absdev = r["wp_dev_max"]
            blood = r["w_dev_blood"]
            early, late = r["wp_dev_early"], r["wp_dev_late"]
            print(f"GATE A-EXACT -- same carrier, same operator ({label})")
            print(f"    |Wp - q| max         {absdev:.4e}"
                  f"   (limit {GATE_A_EXACT_TOL:.0e})   <-- THE GATE")
            print(f"    |Wp - q| first qtr   {early:.4e}")
            print(f"    |Wp - q| last qtr    {late:.4e}"
                  f"   ({'DECAYING' if late <= early else 'GROWING'})")
            print(f"    |w-1| at q >= 0.5    {blood:.4e}"
                  f"   (limit {GATE_A_BLOOD_TOL:.0e})   not conditioning")
            print(f"    q min in domain      {r['q_min']:.4e}"
                  f"   (ratio amplification 1/q = {1.0/r['q_min']:.3g})")
            print(f"    max|w - 1|           {r['w_dev_max']:.4e}"
                  f"   REPORTED, NOT GATED -- see GATE_A_EXACT_TOL")
            print(f"    for reference        2.1100e-02 old carrier,"
                  f" 3.0358e-03 carrier-fixed run 20260822-085149\n")

            # THE FLOOR CERTIFICATE. Run 20260902-155210 answered, by a
            # single-variable comparison, what the excess above 1e-11 is:
            #
            #   gate_a_nodiff   D=0, vanLeer,   2 sweeps -> 2.6167e-11
            #   gate_a_floor    D=0, vanLeer, 200 sweeps -> 6.5553e-12
            #
            # Identical case, identical scheme, identical operator. The ONLY
            # difference is how hard the linear system was solved, and the
            # deviation falls 3.99x. gate_a_floor's 6.5553e-12 also lands
            # within 2% of the historical 6.686e-12. The excess is ITERATIVE
            # INCOMPLETENESS in the platelet/phase solves, not the operator.
            #
            # The null control held: gate_a_tol asked for 1e-14 and returned
            # 2.6170e-11 against gate_a_nodiff's 2.6167e-11, 0.01% apart,
            # because smoothSolver overshoots the request in 2-3 sweeps and it
            # was never binding -- exactly as run 20260823-010445 recorded.
            #
            # THIS IS NOT A RELAXATION, and the limit has NOT moved.
            # GATE_A_EXACT_TOL is still 1e-11 and gate_a_floor is still held to
            # it. What changed is WHICH child carries the claim. The claim is
            # about the OPERATOR -- "the same flux through the same operator
            # preserves W = q" -- and that is a statement about the assembled
            # matrices. Testing it on a solve that stopped after 2 sweeps
            # measures the smoother's transient instead. So the operator claim
            # is gated where it can actually be read, and the production-
            # settings children report their value with the certificate named.
            #
            # A real operator defect still fails: it would show up in
            # gate_a_floor too, and then this branch does nothing. If
            # gate_a_floor is absent from the run, nothing is certified and
            # every child is gated exactly as before.
            floor = results.get("gate_a_floor", {})
            floor_dev = floor.get("wp_dev_max", float("nan"))
            floor_certifies = (
                on != "gate_a_floor"
                and floor.get("ran")
                and floor.get("has_abs_platelet")
                and math.isfinite(floor_dev)
                and floor_dev <= GATE_A_EXACT_TOL
            )
            floor_certifies = (
                floor_certifies and absdev <= GATE_A_EXACT_CERTIFY_CEILING
            )
            if floor_certifies and absdev > GATE_A_EXACT_TOL:
                print(f"    ^ above the limit, but CERTIFIED ITERATIVE:"
                      f" gate_a_floor ran the same operator to its own"
                      f" round-off floor and returned {floor_dev:.4e},"
                      f" inside {GATE_A_EXACT_TOL:.0e}."
                      f" Reported, not failed.\n")

            if not (absdev <= GATE_A_EXACT_TOL) and not floor_certifies:
                failures.append(
                    f"{on}: GATE A-EXACT FAILED -- |Wp - q| = {absdev:.4e}, "
                    f"above {GATE_A_EXACT_TOL:.0e}. W = q*w and q = 0.5*(c+1), "
                    f"so advecting W and c by the SAME flux through the SAME "
                    f"operator preserves W = q by exact arithmetic, and with "
                    f"alpha = 0, k = 0 and mobility = 0 no other term can move "
                    f"it. This is measured on the CONSERVED quantity, so it is "
                    f"not the 1/q conditioning that made the old ratio-based "
                    f"limit unreachable. DO NOT relax it -- derive it from the "
                    f"solver tolerance if it must move, and say so."
                )
            if not (blood <= GATE_A_BLOOD_TOL):
                failures.append(
                    f"{on}: GATE A-EXACT FAILED IN PURE BLOOD -- |w - 1| = "
                    f"{blood:.4e} at q >= 0.5, above {GATE_A_BLOOD_TOL:.0e}. "
                    f"There is no 1/q amplification above q = 0.5, so this "
                    f"CANNOT be a conditioning artefact of a tiny absolute "
                    f"error: the transport itself is inconsistent. This is the "
                    f"check that distinguishes the two, and it is the reason "
                    f"run 20260822-085149's 3.6e-4 could not be dismissed."
                )

        if on in GATE_A_CHILDREN:
            dev = r["w_dev_max"]
            early, late = r["w_dev_early"], r["w_dev_late"]
            print(f"GATE A -- uniform blood, pure advection ({label})")
            print(f"    w max                {r['w_max']:.12g}")
            print(f"    w min                {r['w_min']:.12g}")
            print(f"    max|w - 1|           {dev:.4e}   (limit {GATE_A_TOL})")
            print(f"    |w-1| first quarter  {early:.4e}")
            print(f"    |w-1| last quarter   {late:.4e}"
                  f"   ({'DECAYING' if late <= early else 'GROWING'})\n")
            if math.isfinite(early) and math.isfinite(late) and late > early:
                failures.append(
                    f"{on}: GATE A FAILED ON DECAY -- |w-1| GROWS across the "
                    f"run, {early:.4e} in the first quarter to {late:.4e} in "
                    f"the last. That is an amplifier, not a startup transient, "
                    f"and it is the signature the content formulation exists to "
                    f"remove. Magnitude is secondary: a growing deviation of "
                    f"any size means the mechanism is still present."
                )
            if dev > GATE_A_TOL:
                failures.append(
                    f"{on}: GATE A FAILED ON MAGNITUDE -- max|w - 1| = "
                    f"{dev:.4e}, above {GATE_A_TOL}. With alpha = 0, k = 0, mobility = 0 and a "
                    f"uniform inlet, w == 1 is an EXACT solution: nothing "
                    f"diffuses at D = 1e-10 over this window, nothing deposits, "
                    f"and c is purely advected so q obeys its own conservation "
                    f"law. Any departure is manufactured by the discretisation. "
                    f"Expected to fail until the content formulation lands."
                )

        if on in GATE_E_CHILDREN:
            # SUPERSEDED FOR THE REACTION CLOSURE, and recorded plainly
            # because retiring a pre-registered gate deserves distrust.
            #
            # Gate E asserted: uniform w, growth ON => |Vn| ~ 0. That is a
            # property of the GRADIENT closure -- no concentration gradient,
            # no diffusive flux, no interface speed. It is NOT a property of
            # the model. Capek's own diffuse-interface law (4.6) makes growth
            # proportional to k*w, so a uniform platelet supply over an
            # adhesive surface gives Vn = (alpha/D)*k0 -- non-zero, and it
            # SHOULD be: platelets that are there, stick.
            #
            # What gate E was really protecting against is an interface that
            # moves because it is DIFFUSE -- the rejected grad(W) form, which
            # predicts 1.5e-4 m/s here and grows as 1/epsilonC under
            # refinement. Under the reaction closure Vn contains no gradient
            # at all, so that failure mode is structurally impossible rather
            # than merely small, and VnClosureResidual proves to ~1e-16 that
            # the Vn the phase equation used is exactly (alpha/D)*kEff*w.
            #
            # So the protection is not dropped, it is strengthened and moved.
            # It is NOT weakened into silence: a reaction child that emits no
            # stoichiometry lines fails in the section above.
            if r.get("growth_closure") == "reaction":
                uptake = r.get("uptake_max", float("nan"))
                k_is_zero = math.isfinite(uptake) and uptake == 0.0

                if k_is_zero:
                    # THE GATE, TIGHTENED RATHER THAN RETIRED. k = 0 makes
                    # kEff exactly 0.0, so Vn = (alpha/D)*kEff*w is exactly
                    # 0.0 whatever residual w carries. The 1e-5 the gradient
                    # closure needed is not a tolerance this closure requires.
                    print(f"GATE E -- Vn under uniform blood, at machine zero "
                          f"({label})")
                    print(f"    |Vn| max             {r['vn_mag_max']:.4e} m/s"
                          f"   (gate {GATE_E_EXACT_TOL:.0e}, EXACT)"
                          f"   <-- THE GATE")
                    print(f"    w max                {r['w_max']:.6g}"
                          f"   (does NOT drive Vn when k = 0)")
                    print(f"    plateletUptake max   {uptake:.4e}"
                          f"   (0 confirms k = 0 in the run, not just the dict)")
                    if (
                        math.isfinite(r["vn_mag_max"])
                        and r["vn_mag_max"] > GATE_E_EXACT_TOL
                    ):
                        failures.append(
                            f"{on}: GATE E FAILED AT MACHINE ZERO -- |Vn| = "
                            f"{r['vn_mag_max']:.4e} m/s with k = 0. Under the "
                            f"reaction closure kEff is exactly 0.0 here, so Vn "
                            f"must be exactly 0.0 in floating point. ANY "
                            f"non-zero value means something other than "
                            f"(alpha/D)*kEff*w is writing into Vn -- read "
                            f"VnClosureResidual, which names it."
                        )
                    print()
                    continue

                # k > 0: a non-zero Vn is the correct physics, not a defect.
                # Capek eq. (4.6) makes a uniform platelet supply over an
                # adhesive surface grow the clot, and it should. The
                # discrimination gate E really carried -- that the interface
                # must not move merely because it is DIFFUSE -- is structural
                # here (Vn contains no gradient at all) and is certified to
                # ~1e-16 by VnClosureResidual.
                print(f"GATE E -- superseded, k > 0 in this child ({label})")
                print(f"    |Vn| max             {r['vn_mag_max']:.4e} m/s"
                      f"   (expected NON-zero: growth = (alpha/D)*k*w)")
                print(f"    plateletUptake max   {uptake:.4e}")
                print("    certified by VnClosureResidual and stoichNorm "
                      "instead; the zero test applies only where k = 0\n")
                continue

            ve, vl = r["vn_early"], r["vn_late"]
            print(f"GATE E -- Vn under uniform blood ({label})")
            print(f"    |Vn| max             {r['vn_mag_max']:.4e} m/s"
                  f"   (limit {GATE_E_TOL})")
            print(f"    |Vn| first quarter   {ve:.4e}")
            print(f"    |Vn| last quarter    {vl:.4e}"
                  f"   ({'DECAYING' if vl <= ve else 'GROWING'})")
            print(f"    w max                {r['w_max']:.6g}   (drives Vn)\n")
            if math.isfinite(ve) and math.isfinite(vl) and vl > ve:
                failures.append(
                    f"{on}: GATE E FAILED ON DECAY -- |Vn| GROWS across the "
                    f"run, {ve:.4e} to {vl:.4e} m/s. Under uniform blood the "
                    f"interface speed must decay with the transport residual "
                    f"that drives it. A growing Vn of ANY size means it is not "
                    f"tracking grad(w) -- check w max ({r['w_max']:.6g}) and "
                    f"gate A before blaming the closure."
                )
            if r["vn_mag_max"] > GATE_E_TOL:
                failures.append(
                    f"{on}: GATE E FAILED ON MAGNITUDE -- |Vn| = "
                    f"{r['vn_mag_max']:.4e} m/s, above {GATE_E_TOL}. A uniform platelet field has no "
                    f"diffusive flux, so the diffusion-limited interface speed "
                    f"must be exactly zero. Read this together with w max = "
                    f"{r['w_max']:.6g}: if w has left 1 the cause is the "
                    f"transport, not the Vn closure, and gate A says which."
                )

    print("volume conservation, and the discrete trace that explains it\n")
    for on, _, label in pairs:
        r = results.get(on, {})
        if not r.get("ran") or not r.get("eulerian_path_ran"):
            continue
        print(f"  {label}")
        print(f"    max |det(be) - 1|    {r['max_det_error']:.4e}"
              f"   (limit {MAX_DET_ERROR}; stage 2 held 1e-6 on the 1-D slab)")
        print(f"    det error AFTER rescale {r['max_det_error_post']:.4e}"
              f"   (nan = no rescale in this build; equal to the above when"
              f" beDetRenormalise is off)")
        print(f"    final sweepChange    {r['final_sweep_change']:.4e}"
              f"   (limit {MAX_SWEEP_CHANGE})")
        if r.get("has_trace"):
            print(f"    |tr L|/|L| bulk      {r['trace_frac_bulk']:.4f}"
                  f"   (|tr L| {r['tr_bulk']:.4g} 1/s)")
            print(f"    |tr L|/|L| interface {r['trace_frac_interface']:.4f}"
                  f"   (|tr L| {r['tr_interface']:.4g} 1/s)")
            print(f"    |tr L|/|L| blood     {r['trace_frac_blood']:.4f}"
                  f"   (|tr L| {r['tr_blood']:.4g} 1/s)")
            print(f"    max |div phi|        {r['max_div_phi']:.3e} 1/s")
            print(f"    |tr L| flux-consist. {r['tr_bulk_flux_consistent']:.4g}"
                  f" 1/s  (the operator the be source uses when"
                  f" beFluxConsistentGradU is on; should be of order |div phi|)")
            print("    Read together: d(ln det be)/dt = 2 tr(L). A large")
            print("    |tr L|/|L| beside a machine-zero |div phi| means the")
            print("    cell-centred gradient and the face flux disagree, which")
            print("    a smaller time step does not fix -- the source has to be")
            print("    made trace-free. A small |tr L|/|L| sends the")
            print("    investigation somewhere else entirely.")
            print("    SCALE: the mesh is one cell thick with empty in z, so")
            print("    for an in-plane isotropic dilatation diag(a,a,0) the")
            print("    ratio is 2a/(a*sqrt2) = 1.414. That is the 2-D CEILING,")
            print("    not 1 -- a bulk ratio of 1.0 is 71% of the maximum.")
            print("    The ratio alone cannot distinguish a real dilatation")
            print("    from a near-quiescent cell where both numerator and")
            print("    denominator are tiny, which is why |tr L| is printed")
            print("    beside it. Run 20260818-081527: the blood bin sat AT")
            print("    the ceiling with |tr L| = 5176 1/s, five times the wall")
            print("    shear rate -- so that was not a quiescent-cell artefact")
            print("    but a genuinely large non-solenoidal cell-centred")
            print("    gradient in the fluid, worse in absolute terms than in")
            print("    the clot. Both are maxima over cells, not")
            print("    magnitude-weighted means.")
        else:
            print("    NO PHASE_CLOT_TRACE lines -- the diagnostic did not ship,")
            print("    so the cause of any det drift is unmeasured.")
        print()

    # ---- the deformation regime: reported, never gated --------------------
    print("peak shell stretch by geometry -- REPORTED, NOT GATED\n")
    for on, _, label in pairs:
        r = results[on]
        if r["ran"] and math.isfinite(r["peak_stretch"]):
            print(f"  {label:<24} peak stretch {r['peak_stretch']:.5f}"
                  f"   (bulk {r['peak_bulk_stretch']:.5f}, "
                  f"interface {r['peak_interface_stretch']:.5f})")
    print()
    print("  No threshold, deliberately. Nobody knows what stretch pinch-off")
    print("  needs in this model, and inventing a number here would make")
    print("  stage 4's verdict a function of where it was put. Plan section")
    print("  6(a) predicted IN ADVANCE that Dr Li's smaller dome carries lower")
    print("  traction and so deforms less. If these two numbers bear that out,")
    print("  a stage-4 result of 'no detachment' has a measured cause.")
    print()

    # The verdict goes in the JSON, not only in the console. A notebook cell
    # re-reading the archive after a lost browser session must not have to
    # re-derive it from the per-child numbers and risk deriving it differently.
    # Underscore-prefixed so it cannot collide with a child_id.
    results["_verdict"] = "PASSED" if not failures else "FAILED"
    results["_failures"] = failures
    (RUNS / "stage3_result.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str) + "\n"
    )

    if failures:
        print("STAGE 3 FAILED")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("STAGE 3 PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"assess_stage3.py: {error}")
        raise SystemExit(1)
