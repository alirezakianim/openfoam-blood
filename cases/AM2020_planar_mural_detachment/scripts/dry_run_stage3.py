#!/usr/bin/env python3
"""Exercise assess_stage3.py against synthetic logs, without OpenFOAM.

The stage-3 assessor decides whether stage 4 launches. It has to pass a
healthy set and reject each way the run can go wrong FOR THAT REASON -- an
assessor that fails everything is no more use than one that passes
everything.

Two failure directions are checked, not one. The obvious one is missing a
defect. The other is the one the slab gate actually got caught by twice:
reporting a conclusion the data does not support. MUST_NOT_SAY covers that.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import math
import tempfile
from pathlib import Path


CASE = Path(__file__).resolve().parents[1]
SCRIPTS = CASE / "scripts"

# s/step. The OFF children are slow, the ON children are ~5x faster: a healthy
# set, comfortably past the 3x threshold but not so far that a defect which
# halves the speedup would still pass.
OFF_COST, ON_COST = 25.0, 5.0
OFF_STEPS, ON_STEPS = 60, 215


def synth(cost, steps, *, eulerian, stretch=1.35, iface_ratio=0.4,
          spd_lost=False, phase_bins=True, det_error=4.0e-7,
          sweep=1.0e-15, trace_frac=0.004, trace=True,
          lock_ratio=0.002, lock=True, cell_loss=0.0,
          w_max=1.02, platelet=True, vn_scale=9.6e-5,
          wp_dev=0.0, q_min=0.5, w_dev_blood=0.0,
          platelet_disc=True, platelet_fo=2.4e-3, led_over=0.0,
          min_upwind=0.62, downwind_faces=0,
          wp_bounds=True, wp_min=4.4e-7, wp_max=1.0, faces_upwinded=12,
          stoich=True, growth_closure="reaction", depletion="none",
          content_branch="yes",
          faces_upwinded_interface=40, label_in_blood=0.004,
          label_drift=True,
          stoich_norm=0.9955, vn_closure_resid=3.1e-16, vn_closure=True,
          uptake_scale=1.37e-9,
          da_interface=0.33,
          c_min=-0.999999, c_max=1.0, phase_bounds=True,
          growth_support=True, growth_below_q005=0.02,
          be_xy_end=2.0, be_xy_saturates=True, be_xy_chaos_after_spd=False,
          debond=True, sigma_found=1, sigma_field="sigmaElastic",
          peak_traction=3.2, traction_saturates=True,
          wound=True, support_limit=0.5, hold_dilation=4,
          coverage_end=1.0, max_height_end=30, released_on_bond=0,
          release_snap=False, release_cells=0, debond_at=None,
          core_exempt=True, hold_taper=0, held_weight=152.0):
    out = []
    for i in range(steps):
        t = (i + 1) * 1.4e-5
        # A little structure so a mean over a window is not identical to a mean
        # over any other window -- a matched-window bug has to be able to show.
        c = cost * (2.0 if i < 5 else 1.0) * (1.0 + 0.02 * ((i % 7) - 3))
        # skeleton is 77% of the step for OFF (stage 0 measured 0.773) and a
        # small share for ON -- the whole point of the reformulation.
        sk = c * (0.10 if eulerian else 0.77)
        out.append(f"PHASE_CLOT_COST time={t:.9g} step={c:.9g} skeleton={sk:.9g} "
                   f"damage=0 phaseField=0 platelets=0 permeability=0 "
                   f"derived=0 release=0 other=0 skeletonFraction=0.7")
        if platelet:
            # w grows GEOMETRICALLY toward w_max, the shape run 20260818-232055
            # actually had (1 -> 3.9 -> 14 -> 47 -> ... -> 6372). A linear ramp
            # would let a mid-run mean pass a check that the real shape fails.
            f = (i + 1) / steps
            wm = 1.0 * (w_max / 1.0) ** f
            # The absolute deviation ramps linearly, which is the shape run
            # 20260822-085149 measured (5.1e-5 -> 1.4e-4 -> 2.1e-4). A
            # geometric ramp here would let a decay check pass that the real
            # shape fails.
            out.append(f"PHASE_CLOT_PLATELET time={t:.9g} wMax={wm:.17g} "
                       f"wMin={1.0 / wm:.17g} VnMagMax={vn_scale * wm:.9g} "
                       f"wOverBound={0 if wm <= 1.1 else 700} "
                       f"WpDevMax={wp_dev * f:.9g} qMin={q_min:.9g} "
                       f"wDevBlood={w_dev_blood * f:.9g}")
        if platelet_disc:
            # plateletFo RAMPS GEOMETRICALLY, which is the shape run
            # 20260901-110348 actually had (0.015 at 14 ms, 0.086 at 15 ms,
            # 0.449 at 16 ms, ~3.2 when g20 let go). A flat value would let a
            # max check pass that the real shape fails, and would not exercise
            # the "reached at t" reporting at all.
            g = (i + 1) / steps
            fo = 2.4e-3 * (platelet_fo / 2.4e-3) ** g
            # The overshoot appears only in the last quarter, because a
            # violation from step one reads as a bad initial condition rather
            # than as the convection weights turning downwind mid-run.
            ov = led_over if g > 0.75 else 0.0
            out.append(f"PHASE_CLOT_PLATELET_FO time={t:.9g} "
                       f"plateletFo={fo:.9g} plateletFoLimit=2")
            # Wp drifts toward its extreme over the run rather than sitting
            # at it, so a max/min taken over the whole log is exercised and a
            # first-step reading would not reproduce the gate.
            wmn = 4.4e-7 + (wp_min - 4.4e-7) * g
            wmx = 1.0 + (wp_max - 1.0) * g
            bounds = f"WpMin={wmn:.9g} WpMax={wmx:.9g} " if wp_bounds else ""
            out.append(f"PHASE_CLOT_PLATELET_LED time={t:.9g} "
                       f"{bounds}"
                       f"WpOvershoot={ov:.9g} WpUndershoot=0 "
                       f"violCells={1 if ov > 1e-9 else 0} "
                       f"overX=7.607e-05 overY=6.79e-06")
            out.append(f"PHASE_CLOT_PLATELET_LIM time={t:.9g} "
                       f"facesUpwinded={faces_upwinded} "
                       f"facesUpwindedAtInterface={faces_upwinded_interface} "
                       f"tol=1e-06")
            out.append(f"PHASE_CLOT_PLATELET_WGT time={t:.9g} "
                       f"minUpwindness={min_upwind:.9g} "
                       f"downwindFaces={downwind_faces}")
        if stoich:
            # stoichNorm DECAYS TOWARD 1 from its worst value, which is the
            # shape the returned fields of run 20260901-200203 actually have
            # (0.99210 at 1 ms rising monotonically to 0.99671 at 30 ms as the
            # interface sharpens). A flat value would not exercise the "worst
            # at t" reporting, and a value that got worse with time would let
            # a decay-shaped bug hide.
            g = (i + 1) / steps
            sn = stoich_norm + (1.0 - stoich_norm) * g * 0.45
            # The first steps have no absorbed platelets at all, so the solver
            # reports a 0/0 ratio as zero. The assessor must skip those rather
            # than read them as a total conservation failure -- exercised here.
            # uptake_scale = 0 is the gate E condition: k = 0, so no
            # platelet is ever absorbed and, under the reaction closure,
            # Vn is exactly zero. The assessor detects that from the log
            # rather than the dictionary, so the log has to be able to say it.
            uptake = 0.0 if (i < 2 or uptake_scale == 0.0) \
                else uptake_scale * (1.0 + g)
            growth = uptake * sn * 2.0 * 0.815217 if uptake else 0.0
            vnc = (f"VnClosureResidual={vn_closure_resid:.9g} "
                   f"VnClosureRef=3.75e-05" if vn_closure else "")
            out.append(f"PHASE_CLOT_GROWTH_STOICH time={t:.9g} "
                       f"closure={growth_closure} depletion={depletion} "
                       f"content={content_branch} "
                       f"clotGrowth={growth:.9g} plateletUptake={uptake:.9g} "
                       f"stoichRatio={2.0 * 0.815217 * sn:.9g} "
                       f"stoichTarget={2.0 * 0.815217:.9g} "
                       f"stoichNorm={sn if uptake else 0.0:.9g} "
                       f"{vnc}")
        if label_drift:
            g = (i + 1) / steps
            out.append(f"PHASE_CLOT_LABEL time={t:.9g} "
                       f"labelTotal={1.4e-13 * (1.0 + g):.9g} "
                       f"labelInBlood={label_in_blood * (0.5 + g):.9g}")
        if growth_support:
            # The share DECAYS from its worst value as the interface sharpens,
            # which is the shape run 20260901-200203 actually had (3.999% at
            # 1 ms falling to ~2% by 20 ms). A flat value would not exercise
            # the "worst at t" reporting.
            g = (i + 1) / steps
            sup = growth_below_q005 * (2.0 - g)
            out.append(f"PHASE_CLOT_GROWTH_SUPPORT time={t:.9g} "
                       f"growthBelowQ005={sup:.9g} "
                       f"growthBelowQ015={sup * 3.5:.9g} "
                       f"growthMeanQ={0.53 + 0.09 * g:.9g}")
        if phase_bounds:
            # c DRIFTS toward its extreme over the run rather than sitting at
            # it, so a max taken over the whole log is exercised and a
            # first-step reading would not reproduce the gate.
            g = (i + 1) / steps
            cmn = -0.999999 + (c_min - -0.999999) * g
            cmx = 1.0 + (c_max - 1.0) * g
            out.append(f"PHASE_CLOT_PHASE_BOUNDS time={t:.9g} "
                       f"cMin={cmn:.12g} cMax={cmx:.12g} "
                       f"cUnder={max(0.0, -1.0 - cmn):.9g} "
                       f"cOver={max(0.0, cmx - 1.0):.9g}")
            out.append(f"PHASE_CLOT_DAMKOHLER time={t:.9g} "
                       f"DaMax={da_interface * 1.4:.9g} "
                       f"DaInterfaceMax={da_interface:.9g} "
                       f"layerCellsMin={1.0 / max(da_interface, 1e-30):.9g} "
                       f"interfaceCells=2213")
        if eulerian:
            grow = (i + 1) / steps
            bulk = 1.0 + (stretch - 1.0) * grow
            iface = 1.0 + (bulk - 1.0) * iface_ratio
            # det error and sweepChange GROW along the run, as they did on
            # run 20260817-234636 -- a flat value would let a max-based check
            # pass that a final-value check fails, or the reverse.
            de = det_error * grow
            sc = sweep * grow
            out.append(f"PHASE_CLOT_EULERIAN time={t:.9g} beMinMinor=0.93 "
                       f"beMaxStretch={bulk:.9g} beDetError={de:.9g} "
                       f"beDetErrorPost={de:.9g} "
                       f"beSweepChange={sc:.9g}")
            if trace:
                out.append(f"PHASE_CLOT_TRACE time={t:.9g} trBulk=1.2 "
                           f"trInterface=0.4 trBlood=0.01 "
                           f"traceFracBulk={trace_frac:.9g} "
                           f"traceFracInterface={trace_frac * 0.5:.9g} "
                           f"traceFracBlood=1e-05 divPhiMax=7.5e-15 "
                           f"trBulkFluxConsistent=7.6e-15")
            if debond:
                # THE BOND'S OWN ACCOUNT. sigmaFound is the field that makes
                # "carried no load" distinguishable from "was never shown a
                # stress"; sigma_found=None reproduces a solver built before
                # it existed, which must be refused rather than read.
                sf = ("" if sigma_found is None
                      else f" sigmaFound={sigma_found}")
                # SATURATING by default -- a bond load that flattens -- so the
                # new not-saturated gate is not tripped by every clean shape.
                # traction_saturates=False climbs linearly and must trip it.
                shape = ((1.0 - math.exp(-20.0*grow)) if traction_saturates
                         else grow)
                pt = (peak_traction*shape) if sigma_found else 0.0
                # debond_at is the fraction of the run at which a BREAKABLE
                # bond starts failing. Without it the bond is the 1e9 Pa
                # control and debondedFaces stays 0, which is what every
                # child in the series except an_break runs.
                if debond_at is None:
                    strength = 1.0e9
                    dbf = 0
                else:
                    strength = 151.0
                    dbf = 12 if grow >= debond_at else 0
                out.append(f"PHASE_CLOT_DEBOND time={t:.9g}{sf} "
                           f"sigmaField={sigma_field} "
                           f"newFaces=0 reformedFaces=0 debondedFaces={dbf} "
                           f"bondedClotFaces=38 injuryFaces=40 wallFaces=250 "
                           f"debondedArea=0 peakTraction={pt:.9g} "
                           f"peakTractionAny={pt:.9g} "
                           f"peakTractionNormal={0.31*pt:.9g} "
                           f"peakTractionShear={0.95*pt:.9g} "
                           f"strength={strength:.9g} "
                           f"utilisation={pt/strength:.9g}")
            if wound:
                # THE WOUND INSTRUMENT. coverage falls linearly from 1.0 to
                # coverage_end and the height with it, so a child that holds
                # and a child that walks off are different SHAPES rather than
                # different final numbers -- a gate that only reads the last
                # step cannot tell a collapse from a slow drift.
                cov = 1.0 - (1.0 - coverage_end)*grow
                inj = 40
                onInj = max(0, int(round(cov*inj)))
                mh = max(0, int(round(30 - (30 - max_height_end)*grow)))
                out.append(f"PHASE_CLOT_WOUND time={t:.9g} "
                           f"injuryFaces={inj} clotOnInjury={onInj} "
                           f"coverage={cov:.9g} maxHeight={mh} "
                           f"cellsOverInjury={onInj*mh} "
                           f"xMin=5.15e-05 xMax=9.85e-05 "
                           f"wallClotXMin=5.15e-05 wallClotXMax=1.1e-04")

                # THE RELEASE SIDE. release_snap puts the whole body free in a
                # SINGLE step at 70% of the run, which is the signature the
                # step-function gate exists to catch; without it the count
                # ramps, which must NOT trip it.
                if release_snap:
                    cells = release_cells if grow > 0.7 else 0
                else:
                    cells = int(round(release_cells*grow))
                rob = released_on_bond if cells > 0 else 0
                out.append(f"PHASE_CLOT_RELEASED time={t:.9g} "
                           f"cells={cells} volume={cells*1e-18:.9g} "
                           f"clotCells=1320 supported={1320-cells} "
                           f"bondSeeds=20 releasedOnBond={rob} "
                           + (f"coreExempted=0 " if core_exempt else "")
                           + f"supportLimit={support_limit:.9g}")
                if hold_dilation >= 0:
                    out.append(f"PHASE_CLOT_BONDHOLD time={t:.9g} "
                               f"seeds=38 heldCells=152 "
                               f"heldWeight={held_weight:.9g} "
                               f"taper={hold_taper} "
                               f"dilation={hold_dilation} "
                               f"penalty=1e+07 carrier=1")
            if lock:
                # The clot loses cells linearly over the run when cell_loss > 0.
                cells = int(800*(1.0 - cell_loss*grow))
                blood_u = 8.33e-3
                out.append(f"PHASE_CLOT_LOCK time={t:.9g} "
                           f"clotUMean={blood_u*lock_ratio*grow:.9g} "
                           f"clotUMax={blood_u*lock_ratio*grow*2:.9g} "
                           f"bloodUMean={blood_u:.9g} "
                           f"lockRatio={lock_ratio*grow:.9g} clotCells={cells}")
                # THE SATURATION OBSERVABLE. Two shapes, and the difference
                # between them is the whole point of the gate: a SATURATING
                # be_xy approaches be_xy_end as 1 - exp(-5f) and is flat over
                # the last quarter, while a NON-SATURATING one climbs linearly
                # and is still moving there. A flat synthetic value would let
                # the gate pass for the wrong reason -- it would never see a
                # late change at all -- which is the mistake the det_error and
                # w_max ramps above already exist to avoid.
                # -20 not -5: the shape must be FLAT by the halfway point,
                # because spd_truncation_saves_plateau loses SPD there and
                # the valid window is only the first half. At -5 the last
                # quarter of that half still moves 7.7% and the case would
                # fail for a reason that has nothing to do with truncation.
                if be_xy_saturates:
                    bexy = be_xy_end * (1.0 - math.exp(-20.0 * grow))
                else:
                    bexy = be_xy_end * grow
                # AFTER be STOPS BEING SPD the strain is not a strain, and in
                # run 20260823-204058 it swung hard: sat_G40's clot-mean be_xy
                # plateaued at 0.348 and then fell to 0.186 once the tensor
                # went indefinite. Reproduced here so the truncation is tested
                # against the shape that actually occurred.
                if be_xy_chaos_after_spd and spd_lost and i > steps // 2:
                    bexy *= 0.35 + 2.0 * ((i * 7919) % 13) / 13.0
                out.append(f"PHASE_CLOT_MOMENTUM time={t:.9g} "
                           f"accInertia=-0.31 accConvection=0.25 "
                           f"accViscous=-6.11 accElastic=-306.85 "
                           f"accPressure=307.79 accPenalty=0 accDrag=0 "
                           # THE 2026-08-27 KEYS. Synthesised with the
                           # eulerianBrinkman configuration's own values --
                           # brinkmanInUEqn=1, a nonzero accDrag would follow
                           # -- would make this generator assert a physics
                           # result it has no business asserting. What the dry
                           # run must prove is that the PARSER and the EXPORT
                           # carry the keys; the values are the OFF path's,
                           # which is what every child in the file except four
                           # actually runs.
                           f"accBondHold=-0.42 "
                           f"anchorsInUEqn=0 brinkmanInUEqn=0 "
                           f"bondHoldInUEqn=1 "
                           f"dragMin=0 dragMax=0 dragFreeCells=0 "
                           f"clotCellsMom=937 "
                           f"accSum=1.08 accResidual=1.39 "
                           f"clotXbar=9.6e-05 clotUxbar=5.96e-03 "
                           f"clotBeXY={bexy:.9g} "
                           f"clotSigmaXY={4.0*bexy:.9g} "
                           f"clotBeXYmax={1.4*bexy:.9g} "
                           f"clotVolume=8.11e-16")
            if phase_bins:
                out.append(f"PHASE_CLOT_BE_PHASE time={t:.9g} "
                           f"stretchBulk={bulk:.9g} stretchInterface={iface:.9g} "
                           f"stretchBlood=1.0000001 detBulk=4e-07 "
                           f"detInterface=1e-08 detBlood=1e-10")
            if spd_lost and i == steps // 2:
                out.append(f"BE_SPD_LOST at time {t:.9g}")
    return "\n".join(out) + "\nEnd\n"


def build(defect=None):
    tmp = Path(tempfile.mkdtemp(prefix="stage3-dry-"))
    case = tmp / "AM2020_planar_mural_detachment"
    (case / "scripts").mkdir(parents=True)
    (case / "runs").mkdir()
    shutil.copy2(SCRIPTS / "assess_stage3.py", case / "scripts")

    plan = {
        "stage3_xu_eul": dict(cost=ON_COST, steps=ON_STEPS, eulerian=True,
                              stretch=1.35),
        "stage3_xu_off": dict(cost=OFF_COST, steps=OFF_STEPS, eulerian=False),
        "stage3_li_eul": dict(cost=ON_COST, steps=ON_STEPS, eulerian=True,
                              stretch=1.08),
        "stage3_li_off": dict(cost=OFF_COST, steps=OFF_STEPS, eulerian=False),
    }

    if defect == "too_slow":
        plan["stage3_xu_eul"]["cost"] = OFF_COST / 2.0
    if defect == "spd_lost":
        plan["stage3_li_eul"]["spd_lost"] = True
    # THE TRUNCATION TEST. be_xy is FLAT before SPD is lost and swings
    # wildly after. Read across the whole run the plateau gate fails; read
    # over the valid window it passes. Run 20260823-204058 had exactly this
    # shape and the gate, before truncation, reported the OPPOSITE of the
    # truth on sat_G40 -- "moved 52.5%" against an actual 0.33%.
    if defect == "spd_truncation_saves_plateau":
        plan["stage3_li_eul"]["spd_lost"] = True
        plan["stage3_li_eul"]["be_xy_chaos_after_spd"] = True
    if defect == "interface_runaway":
        plan["stage3_xu_eul"]["iface_ratio"] = 1.6
    if defect == "eulerian_path_dead":
        plan["stage3_xu_eul"]["eulerian"] = False
    if defect == "no_phase_bins":
        plan["stage3_xu_eul"]["phase_bins"] = False
    # THE DEFECT RUN 20260817-234636 ACTUALLY HAD, and which this assessor
    # missed completely: det(be) reached 3.24 and the gate reported only that
    # it could not compute one cost ratio.
    if defect == "volume_not_conserved":
        plan["stage3_xu_eul"]["det_error"] = 3.24
        plan["stage3_xu_eul"]["trace_frac"] = 0.24
    if defect == "picard_not_converged":
        plan["stage3_li_eul"]["sweep"] = 7.9e-4
    if defect == "trace_diagnostic_missing":
        plan["stage3_xu_eul"]["trace"] = False
        plan["stage3_xu_eul"]["det_error"] = 3.24
    # EULERIAN-ONLY CHILDREN, the shape run 20260818-081527 actually had. The
    # assessor never ran on it: the runner tested startswith("stage3_") and
    # these are named diag_*. Once that was fixed the assessor still knew only
    # the four static stage3_* names, so it would have reported four absent
    # children and ignored the five present ones.
    if defect in ("eulerian_only_healthy", "eulerian_only_bad_det"):
        plan = {"diag_xu_base": dict(cost=ON_COST, steps=ON_STEPS,
                                     eulerian=True, stretch=1.35),
                "fix_xu_devl": dict(cost=ON_COST, steps=ON_STEPS,
                                    eulerian=True, stretch=1.35)}
        if defect == "eulerian_only_bad_det":
            plan["diag_xu_base"]["det_error"] = 3.24
            plan["diag_xu_base"]["trace_frac"] = 0.82

    # THE PLATELET-CONTENT ACCEPTANCE GATES. These shapes exercise gates A and
    # E in both directions -- a clean pair that must PASS, and each gate
    # violated on its own so neither can be satisfied by the other's evidence.
    if defect in ("gates_clean", "gate_a_violated", "gate_e_violated",
                  "gate_e_growing", "gate_a_exact_absolute",
                  "gate_a_exact_conditioning_only",
                  "gate_a_exact_blood_only",
                  "growth_stoich_broken", "growth_stoich_above_one",
                  "growth_closure_mismatch", "growth_stoich_missing",
                  "growth_closure_uncertified", "damkohler_high",
                  "gate_e_reaction_leak", "phase_left_bounds",
                  "phase_bounds_missing", "growth_from_clot_interior",
                  "growth_support_missing", "legacy_branch_not_penalised",
                  "child_crashed_sigfpe",
                  "gate_a_certified_by_floor", "gate_a_floor_itself_fails"):
        plan = {
            # w exactly 1 and Vn exactly 0: the content formulation's target.
            "gate_a_advect": dict(cost=ON_COST, steps=ON_STEPS, eulerian=True,
                                  stretch=1.0, w_max=1.0, vn_scale=0.0),
            # k0 = k1 = 0 in every gate E child, so nothing is ever
            # absorbed and Vn is exactly zero under the reaction closure.
            "gate_e_vn": dict(cost=ON_COST, steps=ON_STEPS, eulerian=True,
                              stretch=1.0, w_max=1.0, vn_scale=0.0,
                              uptake_scale=0.0),
            "gate_a_exact": dict(cost=ON_COST, steps=ON_STEPS, eulerian=True,
                                 stretch=1.0, w_max=1.0, vn_scale=0.0),
            "gate_a_nodiff": dict(cost=ON_COST, steps=ON_STEPS, eulerian=True,
                                  stretch=1.0, w_max=1.0, vn_scale=0.0),
            "gate_a_linear": dict(cost=ON_COST, steps=ON_STEPS, eulerian=True,
                                  stretch=1.0, w_max=1.0, vn_scale=0.0),
            "gate_a_tol": dict(cost=ON_COST, steps=ON_STEPS, eulerian=True,
                               stretch=1.0, w_max=1.0, vn_scale=0.0),
            # The linear-solver floor probe. In the plan so the certificate
            # branch has something to read; run 20260902-155210 put it here.
            "gate_a_floor": dict(cost=ON_COST, steps=ON_STEPS, eulerian=True,
                                 stretch=1.0, w_max=1.0, vn_scale=0.0),
        }
        if defect == "gate_a_certified_by_floor":
            # THE SHAPE RUN 20260902-155210 ACTUALLY HAD. Production-settings
            # children sit above 1e-11 on iterative incompleteness; the floor
            # child runs the SAME operator to its round-off floor and comes in
            # under the limit. That certificate must clear the others.
            plan["gate_a_exact"].update(wp_dev=2.5841e-11, q_min=1.9e-5,
                                        w_dev_blood=3.96e-11)
            plan["gate_a_nodiff"].update(wp_dev=2.6167e-11, q_min=1.9e-5,
                                         w_dev_blood=3.96e-11)
            plan["gate_a_tol"].update(wp_dev=2.6170e-11, q_min=1.9e-5,
                                      w_dev_blood=3.96e-11)
            plan["gate_a_floor"].update(wp_dev=6.5553e-12, q_min=1.9e-5,
                                        w_dev_blood=1.0e-11)
        if defect == "gate_a_floor_itself_fails":
            # A REAL operator defect: it survives 200 sweeps, so the floor
            # child exceeds the limit too and NOTHING is certified. This is
            # what stops the certificate becoming a blanket excuse.
            plan["gate_a_exact"].update(wp_dev=4.0e-11, q_min=1.9e-5,
                                        w_dev_blood=3.96e-11)
            plan["gate_a_floor"].update(wp_dev=3.8e-11, q_min=1.9e-5,
                                        w_dev_blood=3.96e-11)
        if defect == "gate_a_exact_absolute":
            # THE SHAPE RUN 20260822-085149 ACTUALLY HAD: |Wp - q| = 2.1e-4
            # growing linearly, with the pure-blood ratio at 3.6e-4.
            plan["gate_a_exact"].update(wp_dev=2.1e-4, q_min=1.9e-5,
                                        w_dev_blood=3.6e-4, w_max=1.00304)
        if defect == "gate_a_exact_conditioning_only":
            # THE CASE THE OLD RATIO GATE COULD NOT TELL APART, and the whole
            # reason the gate was rewritten. The absolute error is 1e-13 --
            # genuinely machine level -- but q = 2e-5 magnifies the RATIO to
            # 5e-9. A limit on max|w - 1| would fail this PERFECTLY GOOD run.
            # Both new checks must stay silent.
            plan["gate_a_exact"].update(wp_dev=1e-13, q_min=2e-5,
                                        w_dev_blood=1e-13, w_max=1.000000005)
        if defect == "gate_a_exact_blood_only":
            # Absolute deviation inside the limit, but the ratio is finite in
            # PURE BLOOD where no amplification exists. That cannot be
            # conditioning, so the blood check must catch it ALONE.
            plan["gate_a_exact"].update(wp_dev=1e-13, q_min=0.5,
                                        w_dev_blood=3.6e-4)
        if defect == "gate_a_exact_violated":
            # THE NUMBER THE OLD CARRIER ACTUALLY PRODUCED. gate_a_content
            # measured 0.0211 with alpha, mobility and adhesion all off, and
            # the relaxed 0.05 gate A PASSED it -- twice. gate_a_exact must
            # fail on exactly this value, or the fix has no test.
            plan["gate_a_exact"]["w_max"] = 1.0211
        if defect == "gate_a_violated":
            # w leaves 1 while Vn stays clean -- gate A must catch this ALONE.
            plan["gate_a_advect"]["w_max"] = 50.0
        if defect == "gate_e_growing":
            plan["gate_e_vn"]["w_max"] = 50.0
            plan["gate_e_vn"]["vn_scale"] = 1.0e-9
        if defect == "gate_e_violated":
            # w stays exactly 1 and Vn is the 1.5e-4 m/s that the REJECTED
            # grad(W) closure predicts under uniform blood. Gate E must catch
            # it with gate A passing, which is precisely the discrimination
            # the closure argument turns on.
            plan["gate_e_vn"]["vn_scale"] = 1.5e-4

        # GATE E IS A GRADIENT-CLOSURE GATE. Its magnitude and decay tests
        # assert that a uniform platelet field gives no interface speed, which
        # is true of Vn = alpha*grad(w) and false of Vn = (alpha/D)*k*w. Both
        # legacy shapes must therefore declare the closure they are testing,
        # or they would be silently skipped by the supersession branch and the
        # dry run would report two gates working that no longer run.
        if defect in ("gate_e_violated", "gate_e_growing"):
            plan["gate_e_vn"]["growth_closure"] = "gradient"

        # ---- THE REACTION CLOSURE'S OWN GATES --------------------------
        # The stoichiometry shapes need a child with a LIVE interface:
        # gate_e_vn now carries uptake = 0 by construction and its stoichNorm
        # is 0/0 on every step, which the assessor correctly skips. Driving
        # them through gate_a_advect keeps each gate tested on a child where
        # the quantity it gates actually exists.
        if defect == "legacy_branch_not_penalised":
            # The w-branch control. It emits no FO/LED/LIM lines BY DESIGN and
            # must NOT be failed for it -- run 20260902-104340 did exactly that.
            # Kept HEALTHY on every other axis so the shape isolates ONE
            # question: is a legacy child failed merely for lacking the
            # content-branch instruments? gate_a_advect's real w runaway is a
            # separate, CORRECT failure and is not what this tests.
            plan["gate_a_advect"].update(content_branch="no",
                                         platelet_disc=False)
        if defect == "growth_stoich_broken":
            # THE NUMBER THE GRADIENT CLOSURE ACTUALLY PRODUCED. Measured on
            # run 20260901-200203's returned fields: the clot was created at
            # 0.034-0.10 of the rate its own platelet uptake implies. If the
            # gate cannot fail on this value it has no test.
            plan["gate_a_advect"]["stoich_norm"] = 0.10
        if defect == "growth_stoich_above_one":
            # The gradFloor softening is strictly one-signed downwards, so a
            # ratio above 1 cannot be a tolerance question -- it is a wiring
            # fault, and it must be caught SEPARATELY from the low side.
            plan["gate_a_advect"]["stoich_norm"] = 1.05
        if defect == "growth_closure_mismatch":
            # stoichNorm is HEALTHY and only the identity is broken: the ratio
            # of two integrals can be right on average while the field the
            # phase equation used is not the closure it claims. The identity
            # check must catch that ALONE.
            plan["gate_a_advect"]["vn_closure_resid"] = 0.87
        if defect == "growth_stoich_missing":
            plan["gate_a_advect"]["stoich"] = False
        if defect == "growth_closure_uncertified":
            # An older binary that emits the stoichiometry but not the
            # identity. It must be REPORTED as uncertified and must NOT fail:
            # a missing field is not a defect, and treating it as one is how
            # a working run gets rejected.
            plan["gate_a_advect"]["vn_closure"] = False
        if defect == "gate_e_reaction_leak":
            # 5e-12 would have PASSED the relaxed 1e-5 gate the gradient
            # closure needed. With k = 0 the reaction closure gives exactly
            # 0.0, so this must now fail -- which is the whole point of
            # restoring the gate to machine zero rather than retiring it.
            plan["gate_e_vn"]["vn_scale"] = 5.0e-12
        if defect == "growth_from_clot_interior":
            # Alireza's concern, given a number that must fail: growth
            # genuinely driven by w in cells that hold almost no blood.
            plan["gate_a_advect"]["growth_below_q005"] = 0.20
        if defect == "growth_support_missing":
            plan["gate_a_advect"]["growth_support"] = False
        if defect == "phase_left_bounds":
            # A growth source strong enough to push q negative. Nothing in the
            # solver reported this before 2026-09-02, and max(q, qFloor) would
            # have absorbed it in silence.
            plan["gate_a_advect"]["c_min"] = -1.004
        if defect == "phase_bounds_missing":
            plan["gate_a_advect"]["phase_bounds"] = False
        if defect == "damkohler_high":
            # Da = 6.6 is what the full 20x shear enhancement gives on this
            # mesh. It is a DECLARED LIMIT being reached, not a defect, and it
            # must be reported without failing anything.
            plan["gate_a_advect"]["da_interface"] = 6.6

    # THE DEFECT RUN 20260818-100608 ACTUALLY HAD, and which this assessor
    # missed entirely: the clot was flowing at 88% of the free stream while the
    # report said "peak stretch 4.63" and called it deformation.
    if defect == "clot_not_held":
        plan["stage3_xu_eul"]["lock_ratio"] = 0.88
    # THE GATE THAT REPLACED lockRatio AS THE CRITERION, so it needs its own
    # planted defect or it ships untested. Outcome (2) of the saturation
    # series' pre-registered reading: be_xy still climbing at the end.
    if defect == "be_xy_not_saturated":
        plan["stage3_xu_eul"]["be_xy_saturates"] = False
    if defect == "clot_smeared_away":
        plan["stage3_xu_eul"]["cell_loss"] = 0.36
    if defect == "lock_diagnostic_missing":
        plan["stage3_xu_eul"]["lock"] = False

    # THE DEFECT RUN 20260824-033102 ACTUALLY HAD, and which every gate in this
    # file passed because the assessor did not read PHASE_CLOT_DEBOND AT ALL.
    # clotWallAdhesion reads a stress field by name, defaults to "sigmaD", and
    # only solveSkeleton() writes sigmaD -- which the Eulerian path never
    # calls. The traction was identically zero for 20 ms, no face could debond
    # at any strength, and the log was indistinguishable from an unloaded bond.
    if defect == "bond_cannot_see_stress":
        plan["stage3_xu_eul"]["sigma_found"] = 0
        plan["stage3_xu_eul"]["sigma_field"] = "sigmaD"
    # A log from a solver built BEFORE the sigmaFound diagnostic. It must be
    # refused, not read: absence of the field is not evidence the bond was fine.
    if defect == "bond_diagnostic_predates_fix":
        plan["stage3_xu_eul"]["sigma_found"] = None
    # The field exists and is registered but nothing ever writes it -- the same
    # defect one layer in, and it must not pass just because sigmaFound is 1.
    if defect == "bond_sees_field_but_zero_stress":
        plan["stage3_xu_eul"]["peak_traction"] = 0.0
    # THE DEFECT RUN 20260824-195730 ACTUALLY HAD, reported as a value when it
    # was a lower bound: the traction the unbreakable bond carries IS the
    # required adhesive strength, so one still climbing at the end understates
    # it by an unknown amount.
    if defect == "traction_not_saturated":
        plan["stage3_xu_eul"]["traction_saturates"] = False

    # An anchored child that still declares material on an intact bond free.
    # The contradiction the control had twenty of.
    if defect == "anchor_released_on_bond":
        plan["stage3_xu_eul"]["release_cells"] = 400
        plan["stage3_xu_eul"]["released_on_bond"] = 20
    # 1278 cells free in ONE step -- the topology test snapping. The ramped
    # shape must NOT trip this, which is what the healthy default checks.
    if defect == "anchor_release_snap":
        plan["stage3_xu_eul"]["release_snap"] = True
        plan["stage3_xu_eul"]["release_cells"] = 1278
    # The wound vacated while the anchor claims to be holding it: 6 of 40
    # faces left, exactly what run 20260824-220555 ended with.
    if defect == "anchor_wound_vacated":
        plan["stage3_xu_eul"]["coverage_end"] = 0.15
    # The clot thinned to within a cell of the anchored layer, so the carrier
    # discontinuity is now landing on live interface rather than in bulk.
    if defect == "anchor_hold_outgrown":
        plan["stage3_xu_eul"]["max_height_end"] = 3
    # A solver built before the instrument. Must be refused, not read as a
    # clean run -- the same rule as bond_diagnostic_predates_fix.
    if defect == "anchor_diagnostic_missing":
        plan["stage3_xu_eul"]["wound"] = False
    # THE HOLD THAT WORKED TOO WELL. Coverage perfect, height untouched,
    # nothing released -- because nothing moved. Every other anchor gate reads
    # this as the best run in the matrix, which is exactly why it needs its
    # own gate: Dr He Li asked for small smooth deformation of the core, and a
    # rigid core fails that as surely as a severed one.
    if defect == "hold_froze_the_clot":
        plan["stage3_xu_eul"]["hold_taper"] = 1
        plan["stage3_xu_eul"]["be_xy_end"] = 1.0e-9
    # A solver built before releaseCoreExempt. The core can still be handed to
    # Ubody by a connectivity pinch, and the log cannot say whether it was.
    if defect == "core_exempt_missing":
        plan["stage3_xu_eul"]["core_exempt"] = False
    # A CONTROL THAT CAME BACK HEALTHY. No anchor and yet nothing went wrong,
    # which would mean the offline reconstruction the series rests on is
    # false. It must fail loudly rather than pass quietly.
    if defect == "anchor_control_healthy":
        plan["stage3_xu_eul"]["support_limit"] = 0.0
        plan["stage3_xu_eul"]["hold_dilation"] = -1
    # THE ORDER OF EVENTS IS THE RESULT. Coverage collapses at 30% of the run
    # while the breakable bond does not start failing until 80%, so the clot
    # left for a reason that is not adhesive failure -- and the run would
    # otherwise end looking exactly like the detachment Dr Li asked for.
    # MUST PASS. A control with no anchor that DOES reproduce the failure is
    # the series working as designed, not a fault -- gating it would make the
    # control unpassable and the whole matrix uninterpretable.
    if defect == "anchor_control_reproduces":
        plan["stage3_xu_eul"]["support_limit"] = 0.0
        plan["stage3_xu_eul"]["hold_dilation"] = -1
        plan["stage3_xu_eul"]["coverage_end"] = 0.15
        plan["stage3_xu_eul"]["release_cells"] = 400
        plan["stage3_xu_eul"]["released_on_bond"] = 20
    # MUST PASS. THE RESULT THE SERIES IS FOR: the bond loads, breaks at 30%
    # of the run, and only then does the clot leave. Coverage ends at 0.2 --
    # below MIN_FINAL_COVERAGE -- and that must NOT be gated, because losing
    # the wound after an adhesive failure is the physics, not the defect.
    if defect == "anchor_break_in_order":
        plan["stage3_xu_eul"]["debond_at"] = 0.3
        plan["stage3_xu_eul"]["coverage_end"] = 0.2
        plan["stage3_xu_eul"]["max_height_end"] = 20
        plan["stage3_xu_eul"]["release_cells"] = 300

    if defect == "anchor_left_before_break":
        plan["stage3_xu_eul"]["debond_at"] = 0.8
        plan["stage3_xu_eul"]["coverage_end"] = 0.1

    # THE DEFECT RUN 20260818-232055 ACTUALLY HAD, which every gate in this
    # file passed: w reached 6372 against a bound of 1.1, drove |Vn| to
    # 0.61 m/s through Vn = alpha*(n & grad(w)), and put cell-scale noise into
    # c -- visible in ParaView for three sessions while the numbers were being
    # read as a det(be) problem.
    if defect == "platelet_runaway":
        plan["stage3_xu_eul"]["w_max"] = 6372.0
    if defect == "platelet_diagnostic_missing":
        plan["stage3_xu_eul"]["platelet"] = False

    # THE DEFECT RUN 20260901-110348 ACTUALLY HAD. Child g20 died with SIGFPE
    # and every gate in this file passed until the field was already gone:
    # the explicit platelet diffusion crossed its stability limit of 2 while
    # WpDevMax, VnMagMax and beMinMinor stayed smooth to six digits.
    if defect == "platelet_fourier_exceeded":
        plan["stage3_xu_eul"]["platelet_fo"] = 3.2

    # THE DEFECT RUN 20260901-200203 ACTUALLY HAD, once the harmonic fix had
    # removed the louder one: c's vanLeer weights are TVD for c and for
    # nothing else, and in the tail -- where c is smooth to 1e-03 per cell
    # while w changes by 0.2 per cell -- they select a downwind face value for
    # a steep Wp front. The result is a smooth dispersive undershoot that goes
    # NEGATIVE at its trough. min(Wp) = -5.32e-05 in g30 from 25 ms.
    if defect == "platelet_wp_negative":
        plan["stage3_xu_eul"]["wp_min"] = -5.32e-5

    # The same scheme manufacturing content, which the offline replay of the
    # measured profile also produced (max 1.197 from an initial 0.912).
    if defect == "platelet_wp_over":
        plan["stage3_xu_eul"]["wp_max"] = 1.2

    # A binary older than the 2026-09-02 limiter, which therefore cannot
    # certify it.
    if defect == "platelet_bounds_missing":
        plan["stage3_xu_eul"]["wp_bounds"] = False

    # A NEW NOTEBOOK AGAINST AN OLD BUILD. This has already happened once in
    # this project, so the assessor must refuse to certify a fix from a log
    # that cannot contain the instrument.
    if defect == "platelet_disc_missing":
        plan["stage3_xu_eul"]["platelet_disc"] = False

    if defect == "off_child_died_early":
        # Fewer common steps than the warm-up window: the ratio must be
        # REFUSED, not computed on four samples of the ramp.
        plan["stage3_li_off"]["steps"] = 18

    for child_id, kw in plan.items():
        if defect == "child_missing" and child_id == "stage3_li_off":
            continue
        d = case / "runs" / child_id
        d.mkdir()
        if defect == "child_crashed_sigfpe" and child_id == "gate_a_nodiff":
            # THE SHAPE RUN 20260902-141852 ACTUALLY HAD. Four children set
            # D = 0; the reaction closure divided by it and trapped inside the
            # CONSTRUCTOR, so the log exists, carries a stack trace and holds
            # ZERO timesteps. The assessor must name the crash rather than
            # emit six "missing instrument" failures blaming the package.
            (d / "log.phaseClotAxisymmetricFluid").write_text(
                "Create mesh for time = 0\n"
                "Selecting solver phaseClotAxisymmetricFluid\n"
                "sigFpe : Enabling floating point exception trapping"
                " (FOAM_SIGFPE).\n"
                "#0  Foam::error::printStack(Foam::Ostream&) at ??:?\n"
                "#1  Foam::sigFpe::sigHandler(int) at ??:?\n"
                "#3  Foam::dimensioned<double> Foam::operator/<double>"
                "(Foam::dimensioned<double> const&,"
                " Foam::dimensioned<double> const&) at ??:?\n"
                "#4  Foam::solvers::phaseClotAxisymmetricFluid::"
                "updateDerivedFields() at ??:?\n"
            )
            continue
        (d / "log.phaseClotAxisymmetricFluid").write_text(synth(**kw))
    return case


def assess(case):
    p = subprocess.run(
        [sys.executable, str(case / "scripts" / "assess_stage3.py")],
        capture_output=True, text=True, cwd=str(case),
    )
    return p.returncode, p.stdout + p.stderr


DEFECTS = {
    "too_slow": r"speedup .* is below 3\.0x",
    "spd_lost": r"stage3_li_eul: be LOST POSITIVE-DEFINITENESS",
    "spd_truncation_saves_plateau": r"stage3_li_eul: be LOST POSITIVE-DEFINITENESS",
    "interface_runaway": r"interface_over_bulk .* is at or above 1",
    "eulerian_path_dead": r"THE EULERIAN PATH NEVER RAN",
    "no_phase_bins": r"no PHASE_CLOT_BE_PHASE lines",
    "child_missing": r"produced no solver log at all",
    "off_child_died_early": r"common steps, at or below",
    "volume_not_conserved": r"det\(be\) departed from 1 by 3\.24",
    "picard_not_converged": r"Picard iteration did not converge",
    "trace_diagnostic_missing": r"NO PHASE_CLOT_TRACE lines",
    "eulerian_only_bad_det": r"det\(be\) departed from 1 by 3\.24",
    # HEADLINE CHANGED 2026-08-23 AND THE OLD ONE WAS WRONG, NOT JUST
    # WORDY. "THE CLOT IS NOT HELD" asserted a conclusion the gate cannot
    # support: run 20260823-102340's offline analysis found the clot
    # ANCHORED AND SHEARING, with Ux spanning a factor of 19 from wall row
    # to crest, so a volume-mean ratio cannot tell that apart from a clot
    # being carried away. The test still pins the VALUE and the gate that
    # produced it, which is what makes it a regression test; it no longer
    # pins a claim that turned out to be false.
    "clot_not_held": r"THE CLOT IS MOVING AT 0\.880 OF THE BLOOD",
    "be_xy_not_saturated": r"THE SHEAR STRAIN HAS NOT SATURATED",
    "bond_cannot_see_stress": r"THE ADHESION CONDITION CANNOT SEE A STRESS",
    "bond_diagnostic_predates_fix": r"has no sigmaFound field",
    "bond_sees_field_but_zero_stress": r"traction is still exactly zero",
    "traction_not_saturated": r"THE WALL TRACTION HAS NOT SATURATED",

    # THE STAGE 6.30 DEFECTS. Every one of these was present in run
    # 20260824-220555 and every gate in this file passed it, because the
    # question "did the clot stay on its wound" had no instrument at all.
    "anchor_released_on_bond": r"MATERIAL SITTING ON AN INTACT BOND WAS "
                               r"DECLARED FREE",
    "anchor_release_snap": r"THE RELEASE IS A STEP FUNCTION",
    "anchor_wound_vacated": r"THE CLOT LEFT THE WOUND",
    "anchor_hold_outgrown": r"THE HOLD BOUNDARY IS CUTTING LIVE INTERFACE",
    "anchor_diagnostic_missing": r"PHASE_CLOT_WOUND is absent",
    "hold_froze_the_clot": r"THE HOLD FROZE THE CLOT",
    "core_exempt_missing": r"no coreExempted field",
    "anchor_control_healthy": r"THE CONTROL DID NOT REPRODUCE THE FAILURE",
    "anchor_left_before_break": r"THE CLOT LEFT BEFORE THE BOND BROKE",
    "clot_smeared_away": r"lost 3[0-9]% of its cells",
    "lock_diagnostic_missing": r"NO PHASE_CLOT_LOCK lines",
    "platelet_runaway": r"THE PLATELET FIELD DIVERGED -- w reached 6372",
    "platelet_diagnostic_missing": r"NO PHASE_CLOT_PLATELET lines",
    "platelet_fourier_exceeded": r"THE PLATELET DIFFUSION IS NOT STABLE",
    "platelet_wp_negative": r"NEGATIVE PLATELET CONTENT",
    "platelet_wp_over": r"Wp EXCEEDED ITS INLET VALUE",
    "platelet_bounds_missing": r"no WpMin/WpMax field",
    "platelet_disc_missing": r"NO PHASE_CLOT_PLATELET_FO lines",
    # The planted shape GROWS geometrically toward w_max, so it must trip the
    # DECAY check -- the discriminator that separates an amplifier from a
    # startup transient -- and not merely the magnitude limit.
    "gate_a_violated": r"GATE A FAILED ON DECAY",
    # The REJECTED grad(W) closure's own prediction under uniform blood.
    "gate_e_violated": r"GATE E FAILED ON MAGNITUDE -- \|Vn\| = 1\.5000e-04",
    # Vn far BELOW the magnitude limit but GROWING. Proves the decay check
    # fires on its own, which is the point of having it: an amplifier must be
    # caught at any size.
    "gate_e_growing": r"GATE E FAILED ON DECAY",
    # 0.0211 -- the value the OLD carrier produced with every source term off,
    # which the relaxed 0.05 gate A passed twice. Machine precision is the only
    # limit that can tell the fix from the defect.
    "gate_a_exact_absolute": r"GATE A-EXACT FAILED -- \|Wp - q\| = 2\.1000e-04",
    "gate_a_floor_itself_fails":
        r"gate_a_floor: GATE A-EXACT FAILED -- \|Wp - q\| = 3\.8000e-11",
    # The crash must be NAMED, and must not be reported as a stale binary.
    "child_crashed_sigfpe":
        r"gate_a_nodiff: THE CHILD CRASHED -- SIGFPE",
    "gate_a_exact_blood_only": r"GATE A-EXACT FAILED IN PURE BLOOD",
    # THE 2026-09-02 REACTION CLOSURE. 0.10 is the value the gradient closure
    # actually produced, so this is the defect the gate was written for.
    "growth_stoich_broken":
        r"GROWTH IS NOT MADE OF THE PLATELETS IT ABSORBS",
    # The synth ramps stoichNorm TOWARD 1, so the run maximum is the first
    # step that has any absorbed platelets at all -- 1.0497, not the 1.05 the
    # shape was seeded with. Matching the digits would pin the gate to the
    # ramp; matching "above 1" pins it to the physics.
    "growth_stoich_above_one": r"stoichNorm reached 1\.\d+, ABOVE 1",
    "growth_closure_mismatch":
        r"THE Vn THE PHASE EQUATION USED IS NOT THE REACTION CLOSURE",
    "growth_stoich_missing": r"NO PHASE_CLOT_GROWTH_STOICH lines",
    # A value that the OLD relaxed gate would have passed.
    "gate_e_reaction_leak": r"GATE E FAILED AT MACHINE ZERO",
    "phase_left_bounds": r"THE PHASE FIELD LEFT \[-1, 1\]",
    "growth_from_clot_interior":
        r"THE GROWTH IS BEING DRIVEN BY THE CLOT INTERIOR",
}

# Cases that must PASS, listed separately from the defects so that "the
# assessor fails everything" cannot be mistaken for "the assessor works".
HEALTHY = ("anchor_control_reproduces", "anchor_break_in_order",
           "eulerian_only_healthy", "gates_clean",
           # A machine-level ABSOLUTE error whose RATIO is 5e-9 because
           # q = 2e-5. This must PASS. It is the case the old ratio-based
           # limit would have failed, and having it in the healthy set is
           # what stops the gate drifting back to the wrong quantity.
           "gate_a_exact_conditioning_only",
           # Production-settings children above 1e-11 with gate_a_floor UNDER
           # it. Measured single-variable in run 20260902-155210: 2 sweeps
           # 2.6167e-11 against 200 sweeps 6.5553e-12, same operator, same
           # scheme. The excess is the linear solve, not the operator, so this
           # must PASS -- and it keeps the gate pointed at the claim it states.
           "gate_a_certified_by_floor",
           # A binary that predates the identity check. Missing evidence is
           # not a defect; rejecting it would fail working runs.
           "growth_closure_uncertified",
           # Da > 1 is the reaction closure's DECLARED LIMIT being reached on
           # this mesh under full shear enhancement. It must be reported and
           # must not fail anything -- a gate on it would be a gate on the
           # mesh, not on the model.
           "damkohler_high",
           # A binary predating the bounds instrument. Reported as absent, not
           # failed: rejecting a run for lacking a field is how working runs
           # get thrown away.
           "phase_bounds_missing",
           # Same rule: a binary that cannot report this is not a defective
           # run, it is an older one.
           "growth_support_missing",
           # A w-branch control emits no FO/LED/LIM lines BY DESIGN. Failing
           # it for that is how run 20260902-104340 rejected the very child
           # the content formulation is measured against.
           "legacy_branch_not_penalised")

# A speedup must never be quoted from a window that does not exist, and a
# stretch must never be turned into a verdict.
MUST_NOT_SAY = {
    # THE POINT OF THE CASE: SPD loss must be reported, and a strain
    # statistic must NOT be computed across it.
    "spd_truncation_saves_plateau": [r"THE SHEAR STRAIN HAS NOT SATURATED"],
    "gates_clean": [r"GATE A FAILED", r"GATE E FAILED"],
    # Each violated shape must fail for ITS OWN gate and leave the other clean,
    # or the two gates are not independent and neither means what it says.
    "gate_a_violated": [r"GATE E FAILED"],
    "gate_e_violated": [r"GATE A FAILED"],
    # 0.0211 sits UNDER gate A's relaxed 0.05 and decays, so gate A must stay
    # silent here. If it spoke, the two gates would not be independent and
    # gate_a_exact would be redundant rather than the discriminating test.
    "gate_a_exact_absolute": [r"GATE A FAILED"],
    # The absolute deviation is 1e-13 -- the gate must NOT cite it, and must
    # not quote max|w - 1| as a failure either.
    "gate_a_exact_blood_only": [r"\|Wp - q\| = ", r"GATE A FAILED"],
    "gate_a_exact_conditioning_only": [r"GATE A-EXACT FAILED"],
    # The certificate must NOT silence a defect that survives 200 sweeps.
    "gate_a_floor_itself_fails": [r"CERTIFIED ITERATIVE"],
    # magnitude stays 5e-8, far under the 1e-5 limit: decay alone must catch it
    "gate_e_growing": [r"GATE E FAILED ON MAGNITUDE"],
    # Each new gate must fail for ITS OWN reason and leave the others clean.
    "growth_stoich_broken": [
        r"THE Vn THE PHASE EQUATION USED IS NOT", r"ABOVE 1"],
    "growth_closure_mismatch": [
        r"GROWTH IS NOT MADE OF THE PLATELETS IT ABSORBS"],
    "growth_stoich_above_one": [
        r"GROWTH IS NOT MADE OF THE PLATELETS IT ABSORBS"],
    "damkohler_high": [r"STAGE 3 FAILED"],
    "phase_bounds_missing": [r"STAGE 3 FAILED"],
    "growth_support_missing": [r"STAGE 3 FAILED"],
    "legacy_branch_not_penalised": [
        r"NO PHASE_CLOT_PLATELET_FO lines",
        r"predates the 2026-09-01 harmonic",
        r"STAGE 3 FAILED"],
    "growth_from_clot_interior": [
        r"GROWTH IS NOT MADE OF THE PLATELETS IT ABSORBS",
        r"THE PHASE FIELD LEFT"],
    "phase_left_bounds": [
        r"GROWTH IS NOT MADE OF THE PLATELETS IT ABSORBS",
        r"THE Vn THE PHASE EQUATION USED IS NOT"],
    "growth_closure_uncertified": [r"STAGE 3 FAILED"],
    "off_child_died_early": [r"Dr Li's dome\n  matched window"],
    "child_missing": [r"Dr Li's dome\n  matched window"],
    # A broken run must never be reported as a usable speedup.
    "volume_not_conserved": [r"STAGE 3 PASSED"],
    "trace_diagnostic_missing": [r"STAGE 3 PASSED"],
    "eulerian_only_bad_det": [r"STAGE 3 PASSED"],
    # A clot being carried downstream must never be reported as a pass, and its
    # accumulated strain must never be narrated as elastic deformation.
    "clot_not_held": [r"STAGE 3 PASSED"],
    "lock_diagnostic_missing": [r"STAGE 3 PASSED"],
    # With no off partner there is NO speedup. Quoting one -- from the stage-0
    # figure measured on another VM, or from any other child -- is the failure
    # mode this pair of patterns exists to catch.
    "eulerian_only_healthy": [r"speedup\s+[0-9]", r"eulerianSolid off\s+[0-9]"],
}


def main() -> int:
    print("stage 3 dry run: the assessor, without OpenFOAM\n")

    case = build(None)
    code, out = assess(case)
    shutil.rmtree(case.parent)
    if code != 0:
        print(out)
        print("\nDRY RUN FAILED: the assessor rejects a healthy set.")
        return 1
    # The healthy set must also not invent a deformation verdict.
    for phrase in ("will pinch", "pinch-off is possible",
                   "sufficient stretch", "enough deformation"):
        if phrase in out:
            print(f"\nDRY RUN FAILED: healthy report claims {phrase!r} -- the "
                  f"peak-stretch number is reported, not adjudicated.")
            return 1
    print("healthy synthetic set -> PASS")

    print("\nplanted defects, each of which must fail FOR ITS OWN REASON:")
    bad = []
    for defect, pattern in DEFECTS.items():
        case = build(defect)
        code, out = assess(case)
        shutil.rmtree(case.parent)
        caught, right = code != 0, bool(re.search(pattern, out, re.S))
        over = [p for p in MUST_NOT_SAY.get(defect, [])
                if re.search(p, out, re.S)]
        status = ("OVERREACHED" if over else
                  "caught" if caught and right else
                  "CAUGHT BUT FOR THE WRONG REASON" if caught else "NOT CAUGHT")
        print(f"  {defect:<22} -> {status}")
        if over or not (caught and right):
            bad.append(defect)
            for line in out.splitlines():
                if line.strip().startswith("- "):
                    print(f"        reported: {line.strip()}")

    print("\nshapes that must PASS, and must not invent what they cannot "
          "measure:")
    for case_name in HEALTHY:
        case = build(case_name)
        code, out = assess(case)
        shutil.rmtree(case.parent)
        over = [p for p in MUST_NOT_SAY.get(case_name, [])
                if re.search(p, out, re.S)]
        status = ("OVERREACHED" if over else
                  "passed" if code == 0 else "WRONGLY FAILED")
        print(f"  {case_name:<22} -> {status}")
        if over or code != 0:
            bad.append(case_name)
            print("        " + "\n        ".join(out.splitlines()[-12:]))

    if bad:
        print(f"\nDRY RUN FAILED: {', '.join(bad)}")
        return 1
    print("\nDRY RUN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
