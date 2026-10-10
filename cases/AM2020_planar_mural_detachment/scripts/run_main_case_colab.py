#!/usr/bin/env python3
"""Restore OpenFOAM, run the MAIN CASE, retain evidence.

Fourteen children in three series: growth against flow rate, the bond strength
detachment would require as flow, stiffness and porosity move, and two
demonstration children that actually come off the wall.

This is the PI-facing case, not a gate. It has no pass/fail verdict - the smoke
gates tested the code, and AM2020_mural_geometry_gate returned MURAL_READY
twice with identical numbers. What this produces is a results table plus a
per-child admissibility check, so a child that failed is named rather than
averaged into a row.

It runs alone, in its own package. The finite-strain gate has cost three
sessions to harness bugs and there is no reason for it to share a session with
the deliverable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import tarfile
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path


# Copy the unpacked tree to Drive WITHOUT its time directories.
#
# The tarball already carries every byte and is the evidence of record. The
# unpacked copy exists for one reason that matters: the notebook's failure
# path globs it for run-exit-code.txt and the solver logs, and the comment
# there records run 20260814-221530, where that glob silently matched nothing
# and the FatalIOError never reached the screen. So the tree stays -- minus
# the field data, which is all of its size and none of that value.
#
# Set RETURN_UNPACKED_TIME_DIRS = True to restore the full copy.
RETURN_UNPACKED_TIME_DIRS = False


CONTENT = Path("/content")
DRIVE_ROOT = CONTENT / "drive" / "MyDrive" / "OpenFOAM_Blood"
# The default is today's Drive upload; `--package <path>` replaces it (L1: the
# notebook passes the zip it built from the verified GitHub clone).
PACKAGE_DRIVE = (
    DRIVE_ROOT / "packages" / "AM2020_main_case_colab_package.zip"
)
PROJECT_LOCAL = CONTENT / "AM2020_main_case_package"
RUNS_LOCAL = CONTENT / "am2020_main_case_runs"
SMOKE_NAME = "AM2020_straight_tube_growth_mechanics_smoke"
ARTIFACT_NAME = "main_case_v1"

# Two gates, run in order. Gate A must pass before gate B's result means
# anything: if the constitutive law is wrong, every displacement in the mural
# geometry is wrong too. B still RUNS either way, because a second Colab cycle
# to collect its evidence is expensive and its logs are informative regardless;
# the notebook reports A first and says plainly when B is not to be trusted.
# One entry, kept in the same shape as the smoke runner so the staging,
# archiving and Drive-verification machinery is shared rather than forked.
GATES = (
    {
        "name": "AM2020_planar_mural_detachment",
        "outputs": "mainCase",
        "controls": "mainCaseControls.json",
        "pass_verdict": None,
    },
)
BOOTSTRAP_MEMBER = (
    "cases/AM2020_paper_cylinder_platelet_initial_condition_gate/scripts/"
    "run_platelet_initial_condition_colab.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bootstrap():
    if not PACKAGE_DRIVE.is_file():
        raise FileNotFoundError(f"Upload the package to {PACKAGE_DRIVE}")
    local = CONTENT / "am2020_mural_smoke_phase_bootstrap.py"
    with zipfile.ZipFile(PACKAGE_DRIVE) as package:
        local.write_bytes(package.read(BOOTSTRAP_MEMBER))
    module = load_module("am2020_mural_smoke_phase_bootstrap", local)
    module.PACKAGE_DRIVE = PACKAGE_DRIVE
    module.PROJECT_LOCAL = PROJECT_LOCAL
    module.RUNS_LOCAL = RUNS_LOCAL
    return module


def stream(command: str) -> int:
    print(f"\n$ {command}", flush=True)
    process = subprocess.Popen(
        ["bash", "-lc", "set -o pipefail\n" + command],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("cannot capture process output")
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def manifest_tree(root: Path) -> dict:
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def verify_drive_copy(local: Path, drive: Path) -> dict:
    local_files = manifest_tree(local)
    drive_files = manifest_tree(drive)
    if local_files != drive_files:
        missing = sorted(set(local_files) - set(drive_files))
        changed = sorted(
            name for name in set(local_files) & set(drive_files)
            if local_files[name] != drive_files[name]
        )
        raise RuntimeError(
            f"Drive artifact verification failed; missing={missing[:5]} changed={changed[:5]}"
        )
    # Per-gate results live under <gate>/result.json now, so the root-level
    # requirement is the runner's own record plus one result per gate.
    required = ["runner_result.json", "artifact_manifest.json", "log.Allwmake"]
    required += [f"{g['name']}/result.json" for g in GATES]
    absent = [name for name in required if name not in drive_files]
    if absent:
        raise RuntimeError(f"Drive artifact verification missing required files: {absent}")
    return {"verified": True, "file_count": len(local_files), "required_files": list(required)}


# ---------------------------------------------------------------------------
# THE 5 ms WATCHER (design 2026-10-05-case-loose-40ms-r3-pair rev. 4 section 2;
# S20 rev. 4 section 3 V6; docket 20261005-1621, report 08).
#
# BATCH 69's two 40 ms children each cost ~13 h. S20's fix is judged at the 5 ms
# frame, so a child that does not pass there is stopped at ~1.5 h instead of
# running on. Only the two BATCH 69 children are watched; no other child is
# touched, and every other launch behaves exactly as before.
#
# THE RULES (rev. 4 section 2, red team 05 W1, W2, W4):
#   * a key the verdict needs and the log lacks is an ERROR clause, never PASS;
#   * the 5 ms frame is read only after the 5.2 ms frame exists;
#   * on the R3-off child the width is REPORTED, not a stop (the 4.0 um price
#     comes from the R3-on carrier); every other clause still stops it;
#   * the record is runs/<child>/watcher-verdict.json: each clause's value,
#     threshold and PASS / FAIL / ERROR;
#   * a FAIL clause outranks an ERROR clause in the overall verdict (both stop
#     the child; FAIL is the more informative), and an ERROR clause outranks a
#     PASS: the overall verdict is PASS only if every gating clause is PASS;
#   * the archive below runs after a stop exactly as after a normal end.
#
# THE STOP ROUTE (cited to the OpenFOAM-13 source; runTimeModifiable is false in
# every child's controlDict, so editing system/controlDict would NOT be re-read):
#   the child's controlDict carries the function object
#       stopWatcher { type stopAtFile; libs (...); action writeNow; }
#   (create_cases.py, the step1_watcher series). The watcher creates the file
#   <case>/stopWatcher. Then, in the solver's own loop:
#     foamRun.C:122               while (pimple.run(runTime))  -> Time::run()
#     Time.C:898-920              Time::run() calls functionObjects_.execute() at
#                                 every step; it is NOT gated by runTimeModifiable
#                                 (that flag gates only readModifiedObjects(),
#                                 Time.C:912 and TimeIO.C:200-203)
#     stopAtFile.C:65-68          condition() is isFile(stopAtFileFile_)
#     stopAtFile.C:81             stopAtFileFile_ = "$FOAM_CASE/" + the object's
#                                 name = <case>/stopWatcher (FOAM_CASE is set at
#                                 argList.C:419 and :428)
#     stopAt.C:87-137             execute(): on the condition, action writeNow
#                                 calls time_.stopAt(writeNow) (stopAt.C:112)
#     Time.C:956-969              stopAt(sa) sets stopAt_ and endTime_ = great
#     Time.C:1252-1256            at the next time increment, stopAt_ == writeNow
#                                 sets endTime_ = value() and writeTime_ = true:
#                                 a full write now, then run() is false
#     stopAtFile.C:112-116        end() removes the file
#   So foamRun writes the current state, logs "stopAtFile(timeIndex=N): stopping
#   now after writing" (stopAt.C:114-117), and exits normally. The watcher reads
#   that line as the acknowledgement. The signal route (sigStopAtWriteNow) is NOT
#   used: it needs the foamRun pid and an environment variable set before launch.
WATCHED_CHILDREN = {
    "fsf_G4_kxu_r3_40": True,     # R3 on: the width is a stop
    "fsf_G4_kxu_nor3_40": False,  # R3 off: the width is reported only
}
WATCH_TIME_S = 0.005             # the frame judged (S20 V6)
WATCH_TIME_TOL_S = 1e-7          # << dt (4.76e-6): the log prints time=0.005 exactly
WIDTH_MAX_UM = 4.0               # S20 V6: replay 3.47, batch 68 4.54, instrument agreement 0.01
LAMBDA_MEDIAN_MIN = 0.9          # S20 V6, over the steps with activeFaces > 0
# REPORTED, not a stop (report 19): batch 68, with S20 OFF, already drifts
# 1.0e-10 at 15 ms and 3.3e-8 at 25 ms, from a source outside transport (OI-52).
# At 5 ms its written value (2.2e-12) is at the write precision, so it is not
# resolved (verification 20, D2). Transport alone adds <= 1.7e-14 per 200 steps (V4),
# so a 1e-12 stop cannot be priced, and C+ under S20 is priced offline by V4.
PARTITION_SUM_B68_5MS = 2.2e-12  # batch 68's relative sum deviation at 5 ms (the reference)
WATCHER_LOG = "log.phaseClotAxisymmetricFluid"
WATCHER_VERDICT = "watcher-verdict.json"
WATCHER_STOP_FILE = "stopWatcher"   # the function object's name: stopAtFile.C:81
WATCHER_STOP_ACK = "stopAtFile(timeIndex="   # stopAt.C:114-117
WATCHER_POLL_S = 5.0
WATCHER_ACK_TIMEOUT_S = 300.0    # ~70 steps at batch 68's 4.27 s per step
_KV = re.compile(r"(\w+)=(\S+)")
_REFUSAL = re.compile(r"FOAM FATAL|\brefused\b")


def watch_frame_name(seconds: float) -> str:
    """OpenFOAM's time-directory name (timeFormat general, timePrecision 10)."""
    return format(seconds, ".10g")


def launch_includes_watched(series_filter: str) -> bool:
    tokens = series_filter.split()
    return (not tokens or "throat_free_shell" in tokens
            or any(c in tokens for c in WATCHED_CHILDREN))


def watcher_preflight(width_script: Path) -> None:
    """Refuse at the START of the session, not at 5 ms of a 13 h child: the
    width probe must be on disk and numpy must import."""
    if not width_script.is_file():
        raise RuntimeError(f"the 5 ms watcher needs {width_script}, which is not in the package")
    import importlib.util as _u
    if _u.find_spec("numpy") is None:
        raise RuntimeError("the 5 ms watcher needs numpy, which does not import")


def measure_width_um(case: Path, seconds: float, width_script: Path) -> float:
    """The interface width at `seconds`, in um, from scripts/na03_width.py (the
    docket probe, copied unchanged): the w_int column of its t = `seconds` row."""
    ms = seconds * 1e3
    proc = subprocess.run(
        [sys.executable, str(width_script), str(case), str(int(round(ms)))],
        capture_output=True, text=True, timeout=900, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"na03_width.py exit {proc.returncode}: {proc.stderr.strip()[-300:]}")
    for line in proc.stdout.splitlines():
        tok = line.split()
        if len(tok) >= 3 and re.fullmatch(r"\d+(\.\d+)?", tok[0]) and abs(float(tok[0]) - ms) < 1e-6:
            return float(tok[2])
    raise RuntimeError(f"na03_width.py printed no row for t = {ms:g} ms")


def parse_watch_window(log_path: Path, t_end: float):
    """One pass over the solver log: the PHASE_CLOT_CONTENT_FCT records up to
    t_end, and every refusal / fatal-error line met before the window closes."""
    fct, refusals = [], []
    with log_path.open("r", errors="replace") as handle:
        for line in handle:
            if line.startswith("PHASE_CLOT_CONTENT_FCT "):
                kv = dict(_KV.findall(line))
                try:
                    if float(kv["time"]) > t_end + WATCH_TIME_TOL_S:
                        break
                except (KeyError, ValueError):
                    pass            # a record with no readable time stays in: its clauses read ERROR
                fct.append(kv)
            elif _REFUSAL.search(line):
                refusals.append(line.strip()[:200])
    return fct, refusals


def _clause(name, rule, status, value=None, threshold=None, note=""):
    return {"clause": name, "rule": rule, "status": status, "value": value,
            "threshold": threshold, "note": note}


def _numbers(fct, key):
    """(values, number of records lacking a numeric `key`)."""
    values, bad = [], 0
    for kv in fct:
        try:
            values.append(float(kv[key]))
        except (KeyError, ValueError):
            values.append(None)
            bad += 1
    return values, bad


def evaluate_child(case: Path, width_stops: bool, width_fn, seconds: float = WATCH_TIME_S) -> dict:
    """The verdict at `seconds`. It reads the log and calls width_fn(case, seconds); it
    does not wait for the frame (watch_child does) and does not stop anything."""
    clauses = []
    fct, refusals, log_error = [], [], None
    try:
        fct, refusals = parse_watch_window(case / WATCHER_LOG, seconds)
    except OSError as error:
        log_error = f"{type(error).__name__}: {error}"
    if log_error or not fct:
        why = log_error or "no PHASE_CLOT_CONTENT_FCT record up to the watch time"
        clauses.append(_clause("fct_records", "at least one record, each with a time", "ERROR", 0, ">= 1", why))
        for name, rule in (("lambda_median", f"median lambdaFluxMean over activeFaces > 0 >= {LAMBDA_MEDIAN_MIN}"),
                           ("neg_after", "negAfter == 0 on every record"),
                           ("zero_flux_rule", "zeroFluxRule == exactZero on every record"),
                           ("partition_sum", f"max partitionSumDevMax, REPORTED (batch 68 at 5 ms: {PARTITION_SUM_B68_5MS:g}; OI-52)")):
            clauses.append(_clause(name, rule, "REPORTED" if name == "partition_sum" else "ERROR", note=why))
    else:
        _, bad_time = _numbers(fct, "time")
        clauses.append(_clause("fct_records", "at least one record, each with a time",
                               "ERROR" if bad_time else "PASS", len(fct), ">= 1",
                               f"{bad_time} records have no readable time" if bad_time else ""))
        act, bad_a = _numbers(fct, "activeFaces")
        lam, bad_l = _numbers(fct, "lambdaFluxMean")
        rule = f"median lambdaFluxMean over activeFaces > 0 >= {LAMBDA_MEDIAN_MIN}"
        if bad_a or bad_l:
            clauses.append(_clause("lambda_median", rule, "ERROR", note=
                                   f"{bad_a} records lack activeFaces, {bad_l} lack lambdaFluxMean"))
        else:
            sel = [l for a, l in zip(act, lam) if a > 0]
            if not sel:
                clauses.append(_clause("lambda_median", rule, "ERROR", note="no record has activeFaces > 0"))
            else:
                med = statistics.median(sel)
                clauses.append(_clause("lambda_median", rule, "PASS" if med >= LAMBDA_MEDIAN_MIN else "FAIL",
                                       med, LAMBDA_MEDIAN_MIN, f"{len(sel)} records with activeFaces > 0"))
        neg, bad_n = _numbers(fct, "negAfter")
        rule = "negAfter == 0 on every record"
        if bad_n:
            clauses.append(_clause("neg_after", rule, "ERROR", note=f"{bad_n} records lack negAfter"))
        else:
            clauses.append(_clause("neg_after", rule, "PASS" if max(neg) == 0 else "FAIL", max(neg), 0))
        rule = "zeroFluxRule == exactZero on every record"
        rules = [kv.get("zeroFluxRule") for kv in fct]
        if any(r is None for r in rules):
            clauses.append(_clause("zero_flux_rule", rule, "ERROR",
                                   note=f"{sum(r is None for r in rules)} records lack zeroFluxRule "
                                        "(the log is not from the S20 solver, or the key was dropped)"))
        else:
            odd = sorted({r for r in rules if r != "exactZero"})
            clauses.append(_clause("zero_flux_rule", rule, "FAIL" if odd else "PASS",
                                   odd[0] if odd else "exactZero", "exactZero"))
        psd, bad_p = _numbers(fct, "partitionSumDevMax")
        rule = f"max partitionSumDevMax, REPORTED (batch 68 at 5 ms: {PARTITION_SUM_B68_5MS:g}; OI-52)"
        if bad_p:
            clauses.append(_clause("partition_sum", rule, "REPORTED",
                                   note=f"{bad_p} records lack a numeric partitionSumDevMax (absent or NA)"))
        else:
            clauses.append(_clause("partition_sum", rule, "REPORTED",
                                   max(psd), PARTITION_SUM_B68_5MS))
    clauses.append(_clause("kernel_refusal", "no 'refused' or FOAM FATAL line before the watch time",
                           "FAIL" if refusals else ("ERROR" if log_error else "PASS"),
                           len(refusals), 0, "; ".join(refusals[:2]) if refusals else (log_error or "")))
    rule = f"interface width at the watch time <= {WIDTH_MAX_UM:g} um" + ("" if width_stops else " (REPORTED only on this child)")
    try:
        width = float(width_fn(case, seconds))
        status = ("PASS" if width <= WIDTH_MAX_UM else "FAIL") if width_stops else "REPORTED"
        clauses.append(_clause("width_um", rule, status, width, WIDTH_MAX_UM))
    except Exception as error:      # a width that cannot be measured is never a pass
        clauses.append(_clause("width_um", rule, "ERROR" if width_stops else "REPORTED", None, WIDTH_MAX_UM,
                               f"{type(error).__name__}: {error}"))
    gating = [c["status"] for c in clauses if c["status"] in ("PASS", "FAIL", "ERROR")]
    verdict = ("FAIL" if "FAIL" in gating else "WATCHER_ERROR" if "ERROR" in gating else "PASS")
    return {"verdict": verdict, "watch_time_s": seconds, "width_stops_child": width_stops,
            "records_in_window": len(fct), "clauses": clauses}


def request_stop(case: Path) -> dict:
    """Create <case>/stopWatcher (the stopAtFile route cited above) and wait for
    the solver's acknowledgement line in its log."""
    text = (case / "system" / "controlDict").read_text() if (case / "system" / "controlDict").is_file() else ""
    body = re.search(r"\bstopWatcher\s*\{(.*?)\}", text, re.S)
    if not body or not re.search(r"\btype\s+stopAtFile\s*;", body.group(1)) \
            or not re.search(r"\baction\s+writeNow\s*;", body.group(1)):
        return {"touched": False, "acknowledged": False,
                "reason": "the controlDict has no stopWatcher stopAtFile (writeNow) function object: "
                          "a touch would be inert, so none was made"}
    log = case / WATCHER_LOG
    offset = log.stat().st_size if log.is_file() else 0
    (case / WATCHER_STOP_FILE).touch()
    started = time.monotonic()
    while time.monotonic() - started < WATCHER_ACK_TIMEOUT_S:
        if log.is_file():
            with log.open("r", errors="replace") as handle:
                handle.seek(offset)
                for line in handle:
                    if WATCHER_STOP_ACK in line:
                        return {"touched": True, "acknowledged": True, "log_line": line.strip(),
                                "seconds_to_acknowledge": round(time.monotonic() - started, 1)}
        if (case / "run-exit-code.txt").is_file() and not (case / WATCHER_STOP_FILE).exists():
            break
        time.sleep(min(1.0, WATCHER_POLL_S))
    return {"touched": True, "acknowledged": False,
            "reason": "no 'stopAtFile(timeIndex=' line in the solver log within "
                      f"{WATCHER_ACK_TIMEOUT_S:g} s, or the child ended first"}


def _write_verdict(case: Path, verdict: dict) -> None:
    path = case / WATCHER_VERDICT
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(verdict, indent=2) + "\n")
    os.replace(tmp, path)


def frame_interval_s(case: Path) -> float:
    """The frames function object's writeInterval, from the child's own controlDict."""
    text = (case / "system" / "controlDict").read_text()
    body = re.search(r"\bframes\s*\{(.*?)\}", text, re.S)
    found = re.search(r"\bwriteInterval\s+([0-9.eE+-]+)\s*;", body.group(1)) if body else None
    if not found:
        raise RuntimeError("no frames function object with a writeInterval in the controlDict: "
                           "the 5.2 ms frame-ready rule cannot be applied")
    return float(found.group(1))


def watch_child(case: Path, width_stops: bool, stop_event, width_fn, poll_s: float = WATCHER_POLL_S,
                say=print) -> dict | None:
    """Wait until the child's 5.2 ms frame exists, then judge its 5 ms frame, record
    runs/<child>/watcher-verdict.json and stop the child unless the verdict is PASS.
    Returns the verdict, or None when the child was never started."""
    started_seen = False
    while not stop_event.is_set():
        if (case / "system" / "controlDict").is_file():
            started_seen = True
            break
        stop_event.wait(poll_s)
    if not started_seen:
        return None
    try:
        nxt = watch_frame_name(WATCH_TIME_S + frame_interval_s(case))
        now = watch_frame_name(WATCH_TIME_S)
    except Exception as error:
        verdict = {"verdict": "WATCHER_ERROR", "reason": f"{type(error).__name__}: {error}", "clauses": []}
        _write_verdict(case, verdict)
        say(f"[watcher] {case.name}: WATCHER_ERROR {verdict['reason']}", flush=True)
        verdict["stop"] = request_stop(case)
        _write_verdict(case, verdict)
        return verdict
    # THE FRAME-READY RULE (red team 05 W4b): the 5 ms frame is read only after
    # the 5.2 ms directory exists, so the solver is past writing the 5 ms one.
    while not stop_event.is_set():
        if (case / nxt).is_dir() and (case / now / "q").is_file() and (case / now / "Vs").is_file():
            break
        if (case / "run-exit-code.txt").is_file():
            break
        stop_event.wait(poll_s)
    ready = (case / nxt).is_dir() and (case / now / "q").is_file() and (case / now / "Vs").is_file()
    if not ready:
        verdict = {"verdict": "NOT_REACHED", "watch_time_s": WATCH_TIME_S,
                   "reason": f"the child ended or the launch finished before the {nxt} frame existed",
                   "run_exit_code": (case / "run-exit-code.txt").read_text().strip()
                   if (case / "run-exit-code.txt").is_file() else None}
        try:
            _, refusals = parse_watch_window(case / WATCHER_LOG, WATCH_TIME_S)
            verdict["refusal_lines"] = refusals[:3]
        except OSError:
            pass
        _write_verdict(case, verdict)
        say(f"[watcher] {case.name}: NOT_REACHED ({verdict['reason']})", flush=True)
        return verdict
    verdict = evaluate_child(case, width_stops, width_fn)
    verdict.update({"child": case.name, "frame_read": now, "frame_ready_because": f"{nxt} exists",
                    "stop_route": "stopAtFile function object stopWatcher, action writeNow "
                                  "(stopAtFile.C:65-81, stopAt.C:87-137, Time.C:898-920, :1252-1256)"})
    _write_verdict(case, verdict)               # recorded BEFORE any stop is attempted
    summary = "; ".join(f"{c['clause']}={c['status']}" + ("" if c["value"] is None else f"({c['value']:.6g})"
                        if isinstance(c["value"], (int, float)) else f"({c['value']})")
                        for c in verdict["clauses"])
    say(f"[watcher] {case.name}: {verdict['verdict']} at {now}: {summary}", flush=True)
    if verdict["verdict"] != "PASS":
        verdict["stop"] = request_stop(case)
        _write_verdict(case, verdict)
        say(f"[watcher] {case.name}: stop {verdict['stop']}", flush=True)
    return verdict


def start_watchers(gate: Path, width_script: Path, watched: dict = None, poll_s: float = WATCHER_POLL_S):
    """One daemon thread per watched child. Returns (stop_event, threads)."""
    watched = WATCHED_CHILDREN if watched is None else watched
    stop_event = threading.Event()
    threads = []

    def width_fn(case, seconds):
        return measure_width_um(case, seconds, width_script)

    def run(child_id, width_stops):
        case = gate / "runs" / child_id
        try:
            watch_child(case, width_stops, stop_event, width_fn, poll_s)
        except Exception:                # a watcher fault is recorded and is never a pass
            detail = traceback.format_exc()
            print(f"[watcher] {child_id}: WATCHER_ERROR\n{detail}", flush=True)
            if case.is_dir():
                try:
                    _write_verdict(case, {"verdict": "WATCHER_ERROR", "reason": detail[-1500:], "clauses": []})
                    request_stop(case)
                except Exception:
                    pass

    for child_id, width_stops in watched.items():
        thread = threading.Thread(target=run, args=(child_id, width_stops), daemon=True, name=f"watch-{child_id}")
        thread.start()
        threads.append(thread)
    return stop_event, threads


def run_children_and_archive(gate: Path, openfoam_bash: str, series_filter: str, archive_script: Path,
                             gate_artifact: Path, fields: str, width_script: Path, watched: dict = None,
                             poll_s: float = WATCHER_POLL_S):
    """Allrun, then the archive. The archive runs whatever Allrun returned, so a
    watcher stop (foamRun exits normally after its writeNow) is archived exactly
    as a completed child is. Returns (allrun exit, archive exit, child ids)."""
    stop_event, threads = start_watchers(gate, width_script, watched, poll_s)
    try:
        code = stream(openfoam_bash + f" && cd {shlex.quote(str(gate))} && bash ./Allrun {series_filter}")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=1800)   # a width measurement in flight is seconds; this only bounds a hang
    # CHILD LIST FROM matrix.json, not from this file. The hard-coded tuple that
    # used to live in GATES went stale the moment the finite-strain gate's
    # children were renamed, and the archiver would have quietly archived
    # nothing for it -- the same failure that cost run 20260810-075435.
    matrix_path = gate / "runs" / "matrix.json"
    if matrix_path.is_file():
        child_ids = [entry["child_id"] for entry in json.loads(matrix_path.read_text())]
    else:
        child_ids = []
        print(f"[archive] {gate.name}: no matrix.json, nothing to archive", flush=True)
    archive_code = subprocess.run(
        ["python3", str(archive_script), "--gate", str(gate), "--artifact", str(gate_artifact),
         "--children", *child_ids, "--tail-start", "0", "--fields", fields],
        check=False,
    ).returncode if child_ids else 1
    return code, archive_code, child_ids


# S22 (research/designs/2026-10-07-S22-pardiso-block-solver.md rev. 2.1,
# APPROVED by Alireza 2026-10-07), sections 2.5 and 5: the pinned MKL the
# solver dlopen's under blockLinearSolver pardiso (its default). The headers
# are not needed: the solver declares its own prototypes and never links MKL.
S22_MKL_PIN = "mkl==2026.1.0"
S22_MKL_SONAME = "libmkl_rt.so.3"


def s22_physical_cores() -> tuple[int, str]:
    """The distinct (CORE, SOCKET) pairs of `lscpu -p=CORE,SOCKET` (rev. 2.1),
    and its output. Refused (RuntimeError) when lscpu gives none: the thread
    count is never guessed."""
    r = subprocess.run(["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True)
    pairs = {tuple(f.strip() for f in line.split(","))
             for line in r.stdout.splitlines() if line.strip() and not line.startswith("#")}
    if r.returncode != 0 or not pairs or any(len(p) != 2 or "" in p for p in pairs):
        raise RuntimeError(
            f"S22: `lscpu -p=CORE,SOCKET` exited {r.returncode} with no usable "
            f"(CORE, SOCKET) pairs; MKL_NUM_THREADS cannot be set from the "
            f"physical cores (design section 2.5). Output: {(r.stdout + r.stderr)[:400]!r}")
    return len(pairs), r.stdout


# S25 section 7 (research/designs/2026-10-10-S25-arteriole-bundle.md rev. 2; Alireza 2026-10-09: only the
# basic 1-core Colab runtime exists). The 1-core batches run ONE solver thread under a PINNED MKL
# conditional-numerical-reproducibility branch, so that two arms (the baseline binary and S25, B76-OFF-A/B)
# are compared bitwise on the same branch whatever CPU model each session lands on. The solver calls
# mkl_cbwr_set(MKL_CBWR_AUTO) before any other MKL call (blockCoupledKernel.H L1610) and is not changed
# here; whether that call leaves an MKL_CBWR environment pin in force is what the probe below MEASURES on
# the session's own libmkl_rt, in the solver's own order, and a pin the call overrides refuses the launch.
S25_CBWR_PROBE = r"""
import ctypes, json, sys
lib = ctypes.CDLL(sys.argv[1])
set_ = lib.MKL_CBWR_Set
set_.argtypes = [ctypes.c_int]
set_.restype = ctypes.c_int
get_ = lib.MKL_CBWR_Get
get_.argtypes = [ctypes.c_int]
get_.restype = ctypes.c_int
out = {"set_auto_rc": None}
if sys.argv[2] == "auto":
    out["set_auto_rc"] = set_(2)      # MKL_CBWR_AUTO = 2, as blockCoupledKernel.H L1247 and L1610
out["branch"] = get_(1)               # MKL_CBWR_BRANCH = 1 (mkl_service.h)
print(json.dumps(out))
"""


def s25_probe_run(lib: str, extra: dict, mode: str) -> dict:
    """One fresh process running S25_CBWR_PROBE in `mode` (get / auto) with MKL_CBWR removed from the environment and then set to
    whatever `extra` says. Its failure is recorded as {"error": ...}, never raised."""
    env = {k: v for k, v in os.environ.items() if k != "MKL_CBWR"}
    env.update(extra)
    try:
        r = subprocess.run([sys.executable, "-c", S25_CBWR_PROBE, lib, mode], capture_output=True,
                           text=True, env=env, timeout=120)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else {
            "error": f"exit {r.returncode}: {(r.stdout + r.stderr)[-300:]}"}
    except Exception as error:   # the probe's own machinery: recorded, not raised
        return {"error": f"{type(error).__name__}: {error}"}


def s25_cpu_flags(cpuinfo_text: str):
    """The ISA flags of the first logical CPU in /proc/cpuinfo text, sorted and space-joined (None when the file has none: a hypervisor
    can hide them). S25 section 12.1: with the model, they decide whether two sessions land on the same MKL code path."""
    m = re.search(r"^flags\s*:\s*(.+)$", cpuinfo_text, re.M)
    return " ".join(sorted(m.group(1).split())) if m else None


def s25_cpu_identity(cpuinfo_text: str, lib, record_auto: bool) -> dict:
    """S25 section 12.1, what environment.json records so that two arms' sessions can be judged SAME or VOID: {"cpu_flags"} always;
    and, when `record_auto` (the 1-core batches, which request NO pin because the solver's own mkl_cbwr_set(AUTO) would override it and the
    baseline binary cannot be changed), {"auto_branch"}: the branch this session's libmkl_rt lands on, measured in a fresh process in the
    solver's own call order (an {"error": ...} when there is no library or the probe fails; never raised)."""
    out = {"cpu_flags": s25_cpu_flags(cpuinfo_text)}
    if record_auto:
        out["auto_branch"] = s25_probe_run(lib, {}, "auto") if lib else {"error": "no libmkl_rt was found to probe"}
    return out


def s25_cbwr_probe(lib: str, pin: str) -> dict:
    """Run the solver's CNR sequence against `lib` in three fresh processes (a CNR mode cannot change after
    the first MKL call): (a) MKL_CBWR=pin, no mkl_cbwr_set; (b) MKL_CBWR=pin, then mkl_cbwr_set(AUTO), as the
    solver does; (c) no MKL_CBWR, then mkl_cbwr_set(AUTO). Returns the three results and a verdict:
      "held"        (b) reports the branch (a) reports, and mkl_cbwr_set returned 0: the pin survives the
                    solver's call (on a CPU whose AUTO branch IS the pin this is true either way);
      "overridden"  (b) differs from (a): the solver's call replaced the pin, which is therefore inert;
      "set_failed"  mkl_cbwr_set(AUTO) returned non-zero under the pin: the solver would refuse at start-up;
      "not_run"     the probe itself could not run (recorded with the reason; never raised)."""
    out = {"pin": pin, "lib": lib}
    runs = (("pin_only", {"MKL_CBWR": pin}, "get"), ("pin_then_auto", {"MKL_CBWR": pin}, "auto"),
            ("auto_only", {}, "auto"))
    for name, extra, mode in runs:
        out[name] = s25_probe_run(lib, extra, mode)
    a, b = out["pin_only"], out["pin_then_auto"]
    if "branch" not in a or "branch" not in b or "branch" not in out["auto_only"]:
        out["verdict"] = "not_run"
    elif b["set_auto_rc"] != 0:
        out["verdict"] = "set_failed"
    elif a["branch"] != b["branch"]:
        out["verdict"] = "overridden"
    else:
        out["verdict"] = "held"
    return out


def s22_mkl_environment(threads_pin: int = 0, cbwr_pin: str = "") -> dict:
    """S22 sections 2.5 and 5: pip install the pinned MKL (3 attempts), export
    PHASE_CLOT_MKL_RT (the installed libmkl_rt.so.3), MKL_NUM_THREADS = max(1,
    physical cores // AM2020_CONCURRENCY) and KMP_BLOCKTIME=0, and return what
    environment.json records: the MKL version, the physical and logical CPU
    counts, the CPU model and the `lscpu -e` layout. A failed install is
    recorded, not raised: a pardiso child then refuses at start-up, naming the
    library, and never runs UMFPACK silently (V22-4).
    S25: threads_pin > 0 sets MKL_NUM_THREADS to it whatever the cores (the 1-core batches); cbwr_pin sets
    MKL_CBWR to it and runs s25_cbwr_probe on the installed library (a pin the solver's own call overrides, or
    a call that fails under it, raises). Both are recorded under "mkl_threads_pinned" and "mkl_cbwr"."""
    import importlib.metadata
    import site
    import sysconfig
    env: dict = {"mkl_pin": S22_MKL_PIN}
    attempts = []
    for attempt in (1, 2, 3):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", S22_MKL_PIN],
                           capture_output=True, text=True)
        attempts.append({"attempt": attempt, "exit": r.returncode,
                         "tail": (r.stdout + r.stderr)[-400:]})
        if r.returncode == 0:
            break
        time.sleep(10 * attempt)
    env["pip_attempts"] = attempts
    env["mkl_installed"] = attempts[-1]["exit"] == 0
    try:
        env["mkl_version"] = importlib.metadata.version("mkl")
    except importlib.metadata.PackageNotFoundError:
        env["mkl_version"] = None
    roots = [sysconfig.get_paths().get("data"), sys.prefix, "/usr/local",
             getattr(site, "USER_BASE", None)]
    lib = next((str(Path(r) / "lib" / S22_MKL_SONAME) for r in roots
                if r and (Path(r) / "lib" / S22_MKL_SONAME).is_file()), None)
    env["mkl_searched"] = [r for r in roots if r]
    if lib:
        os.environ["PHASE_CLOT_MKL_RT"] = lib
    env["phase_clot_mkl_rt"] = os.environ.get("PHASE_CLOT_MKL_RT")
    concurrency_text = os.environ.get("AM2020_CONCURRENCY", "1")
    if not concurrency_text.isdigit() or int(concurrency_text) < 1:
        raise RuntimeError(f"S22: AM2020_CONCURRENCY {concurrency_text!r} is not a positive "
                           f"integer; MKL_NUM_THREADS cannot be set (design section 2.5).")
    concurrency = int(concurrency_text)
    physical, lscpu_p = s22_physical_cores()
    logical = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
               else os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(threads_pin if threads_pin > 0 else max(1, physical // concurrency))
    os.environ["KMP_BLOCKTIME"] = "0"
    cbwr = {"requested": cbwr_pin or None, "probe": None}
    if cbwr_pin:
        os.environ["MKL_CBWR"] = cbwr_pin
        if lib:
            cbwr["probe"] = s25_cbwr_probe(lib, cbwr_pin)
        else:
            cbwr["probe"] = {"verdict": "not_run", "reason": "no libmkl_rt was found to probe"}
    cbwr["env_MKL_CBWR"] = os.environ.get("MKL_CBWR")
    model = None
    cpuinfo_text = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpuinfo_text = cpuinfo.read_text()
        m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo_text, re.M)
        model = m.group(1).strip() if m else None
    identity = s25_cpu_identity(cpuinfo_text, lib, record_auto=threads_pin > 0 and not cbwr_pin)
    flags = identity["cpu_flags"]
    if "auto_branch" in identity:
        cbwr["auto_branch"] = identity["auto_branch"]
    lscpu_e = subprocess.run(["lscpu", "-e"], capture_output=True, text=True)
    env.update({
        "am2020_concurrency": concurrency,
        "physical_cores": physical,
        "logical_cpus": logical,
        "nproc": logical,
        "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        "mkl_threads_pinned": threads_pin if threads_pin > 0 else None,
        "mkl_cbwr": cbwr,
        "kmp_blocktime": os.environ["KMP_BLOCKTIME"],
        "cpu_model": model,
        "cpu_flags": flags,
        "lscpu_p_core_socket": lscpu_p,
        "lscpu_e": lscpu_e.stdout if lscpu_e.returncode == 0 else None,
    })
    print(f"[main-case] S22: {S22_MKL_PIN} installed={env['mkl_installed']} "
          f"PHASE_CLOT_MKL_RT={env['phase_clot_mkl_rt']} physical={physical} "
          f"logical={logical} MKL_NUM_THREADS={env['mkl_num_threads']} KMP_BLOCKTIME=0")
    if cbwr_pin:
        probe = cbwr["probe"] or {}
        print(f"[main-case] S25: MKL_CBWR={cbwr_pin} probe verdict {probe.get('verdict')} "
              f"(pin_only {probe.get('pin_only')}, pin_then_auto {probe.get('pin_then_auto')}, "
              f"auto_only {probe.get('auto_only')})")
        refusal = s25_cbwr_refusal(cbwr_pin, probe)
        if refusal:
            raise RuntimeError(refusal)
    return env


def s25_cbwr_refusal(pin: str, probe: dict):
    """The launch refusal for a probe verdict, or None. Only a MEASURED inert pin refuses; a probe that could
    not run is recorded in environment.json and does not stop the session."""
    verdict = probe.get("verdict")
    if verdict not in ("overridden", "set_failed"):
        return None
    return (f"S25: MKL_CBWR={pin} is {'replaced by' if verdict == 'overridden' else 'refused after'} the solver's own "
            f"mkl_cbwr_set(MKL_CBWR_AUTO) on this session's libmkl_rt ({probe}): the pinned branch would be inert and "
            f"two arms could not be compared bitwise. Refusing the launch; the pin must live in the solver "
            f"(hand-off to the solver-engineer).")


# REFERENCE LIBRARIES (S22 section 7 arm A0, BATCH 71; S23 section 7 arm (1'), BATCH 72):
# an older release's solver, built beside the package's own. The notebook clones the tag
# (it holds the token; the runner never sees it) and passes the clone's solver directory
# here. A reference has the same library name as the package's solver, so it is compiled
# into a SEPARATE FOAM_USER_LIBBIN, and the environment variable below names that
# directory for runs/<child>/child.env (create_cases.py solver_library_env writes it for
# the child that names solver_library; Allrun sources it inside that child's subshell and
# it prepends the directory to LD_LIBRARY_PATH). name -> (the variable, the libbin
# directory under /content, the label, the verdict when its build fails, the command-line
# flag's attribute). BATCH 71 uses p2prime (tag digest-8497d46a7010); BATCH 72 uses s22
# (tag digest-442b95afaec8).
REFERENCE_LIBRARIES = {
    "p2prime": ("PHASE_CLOT_P2PRIME_LIBBIN", "p2prime_libbin", "P2-prime", "P2PRIME_COMPILE_FAILED",
                "p2prime_solver"),
    "s22": ("PHASE_CLOT_S22_LIBBIN", "s22_libbin", "S22 reference", "S22REF_COMPILE_FAILED",
            "s22_solver"),
    # S25 section 7, B76-OFF-A: the release BEFORE S25 (release-e2a57db1), the do-nothing control binary.
    "baseline": ("PHASE_CLOT_BASELINE_LIBBIN", "baseline_libbin", "baseline (release-e2a57db1)",
                 "BASELINE_COMPILE_FAILED", "baseline_solver"),
}
P2PRIME_LIB = "libphaseClotAxisymmetricFluid.so"   # the library file, the same for every reference


def selected_child_ids(series_filter: str, controls: Path) -> list:
    """The child ids this launch will build (all with no filter), selected by id or series as Allrun and
    create_cases.py select them."""
    tokens = series_filter.split()
    return [k["child_id"] for k in json.loads(controls.read_text())["children"]
            if not tokens or k["child_id"] in tokens or k.get("series") in tokens]


def stage_restart(restart_from: str, child_ids: list, driver_path: Path, dest: Path) -> dict:
    """S25 section 2C. `restart_from` is a previous session's case-after-run-*.tar.gz (its full writes and
    dictionaries are extracted into dest) or a directory laid out dest-style, <dir>/<child>/<time dirs>,
    constant/, system/. EVERY selected child must have a full intact write in it (restart_driver.choose;
    there is no latestTime and no frame): refused here, at minute 3, not by Allrun after the compile.
    Exports AM2020_RESTART_FROM for Allrun, which stages the choice per child after regenerating the case
    and refuses regenerated dictionaries that differ from the writing session's."""
    driver = load_module("restart_driver_for_runner", driver_path)
    src = Path(restart_from)
    try:
        if src.is_file():
            extracted = driver.extract_from_tarball(src, child_ids, dest)
            source = dest
        elif src.is_dir():
            extracted, source = None, src
        else:
            raise RuntimeError(f"--restart-from {restart_from!r} is neither a tarball nor a directory")
        chosen = {}
        for child in child_ids:
            if not (source / child).is_dir():
                raise RuntimeError(f"--restart-from holds no directory for child {child!r} ({source})")
            chosen[child] = driver.choose(source / child)[0].name
    except driver.RestartError as error:
        raise RuntimeError(f"S25 restart: {error}") from error
    os.environ["AM2020_RESTART_FROM"] = str(source)
    return {"restart_from": restart_from, "source": str(source), "chosen": chosen, "extracted": extracted}


def reference_libraries_needed(series_filter: str, controls: Path) -> set:
    """The solver_library names of the children this launch will build (all of them with no
    filter), read from the controls. A child is selected by its id or its series, as Allrun and
    create_cases.py select it."""
    tokens = series_filter.split()
    kids = json.loads(controls.read_text())["children"]
    return {k["solver_library"] for k in kids
            if "solver_library" in k
            and (not tokens or k["child_id"] in tokens or k.get("series") in tokens)}


def check_reference_libraries(needed: set, given: dict) -> None:
    """Refuse at the START of the session (not when child.env refuses at the child): every
    library a selected child names must have been passed, and no library is built for
    nothing. `given` is name -> the solver directory string (empty = not passed)."""
    unknown = sorted(needed - set(REFERENCE_LIBRARIES))
    if unknown:
        raise RuntimeError(f"a selected child names solver_library {unknown}, which this runner cannot build")
    missing = sorted(n for n in needed if not given.get(n))
    if missing:
        flags = ", ".join("--" + REFERENCE_LIBRARIES[n][4].replace("_", "-") for n in missing)
        raise RuntimeError(f"a selected child names solver_library {missing} but {flags} was not passed: "
                           f"its child.env would refuse it (exit 97) after the whole session was spent")
    surplus = sorted(n for n, v in given.items() if v and n not in needed)
    if surplus:
        raise RuntimeError(f"the {surplus} reference library was passed but no selected child names it "
                           f"as solver_library: it would be built for nothing")


def build_reference_library(bootstrap, name: str, solver_dir: Path, openfoam_bash: str, libbin: Path,
                            artifact: Path) -> dict:
    """Compile solver_dir (the clone's user-applications/phaseClotAxisymmetricFluid) into
    libbin and export the reference's variable. Refused (RuntimeError) when the directory is
    not a solver, the build fails, the library is missing, or it is byte-identical to the
    package's own library (the two builds would then be one)."""
    var, _dirname, label, _verdict, _attr = REFERENCE_LIBRARIES[name]
    for need in ("Allwmake", "Make/files", "phaseClotAxisymmetricFluid.C"):
        if not (solver_dir / need).is_file():
            raise RuntimeError(f"{label} solver directory {solver_dir} has no {need}")
    libbin.mkdir(parents=True, exist_ok=True)
    if any(libbin.iterdir()):
        raise RuntimeError(f"{label} libbin {libbin} is not empty; a stale library could be loaded")
    # FOAM_USER_LIBBIN is exported AFTER the bashrc: sourcing it sets the variable.
    built = bootstrap.bash(
        openfoam_bash + f" && export FOAM_USER_LIBBIN={bootstrap.quoted(libbin)}"
        f" && cd {bootstrap.quoted(solver_dir)} && bash ./Allwmake",
        check=False,
    )
    (artifact / f"log.Allwmake.{name}").write_text(built.stdout)
    lib = libbin / P2PRIME_LIB
    info = {"solver_dir": str(solver_dir), "libbin": str(libbin), "compile_exit_code": built.returncode,
            "library_exists": lib.is_file()}
    if built.returncode != 0 or not lib.is_file():
        raise RuntimeError(f"{label} build failed (exit {built.returncode}, library {lib} exists: {lib.is_file()}); "
                           f"see log.Allwmake.{name}")
    info["sha256"] = sha256(lib)
    own = bootstrap.bash(openfoam_bash + " && echo FOAM_USER_LIBBIN=$FOAM_USER_LIBBIN", check=False).stdout
    own_dir = next((l.split("=", 1)[1].strip() for l in own.splitlines() if l.startswith("FOAM_USER_LIBBIN=")), "")
    own_lib = Path(own_dir) / P2PRIME_LIB if own_dir else None
    info["own_library"] = str(own_lib) if own_lib else None
    if own_lib is not None and own_lib.is_file():
        info["own_sha256"] = sha256(own_lib)
        if info["own_sha256"] == info["sha256"]:
            raise RuntimeError(f"the {label} library is byte-identical to the package's own: the two builds are one")
    os.environ[var] = str(libbin)
    print(f"[main-case] {label} library built: {lib} sha256={info['sha256'][:12]} "
          f"{var}={libbin}", flush=True)
    return info


def build_p2prime(bootstrap, solver_dir: Path, openfoam_bash: str, libbin: Path, artifact: Path) -> dict:
    """BATCH 71's entry point, unchanged in behaviour (build_reference_library for p2prime)."""
    return build_reference_library(bootstrap, "p2prime", solver_dir, openfoam_bash, libbin, artifact)


def verify_repo_stamp(project: Path, expected: str, result: dict) -> None:
    """L1 (design 2026-10-05, change 5): RECOMPUTE the repo digest over the
    unpacked tree with the function and manifest carried in REPO_STAMP.json,
    and refuse unless it equals --repo-digest. Recomputed here, never passed
    through, so L-G2 can fail: a file changed after the notebook's verify is
    refused. A zip-path package carries no stamp and is given no digest."""
    stamp_path = project / "REPO_STAMP.json"
    if not stamp_path.is_file():
        result["source"] = "zip"
        if expected:
            raise RuntimeError(
                "--repo-digest was given but the package carries no "
                "REPO_STAMP.json: it was not built from the GitHub clone")
        return
    stamp = json.loads(stamp_path.read_text())
    result["source"] = "github"
    result["tag"] = stamp.get("tag")
    result["repo_digest"] = expected or None
    result["repo_digest_stamp"] = stamp.get("repo_digest")
    if not expected:
        raise RuntimeError(
            "the package carries REPO_STAMP.json but no --repo-digest was "
            "given: there is nothing to verify it against")
    namespace: dict = {}
    exec(compile(stamp["function"], "REPO_STAMP.json:function", "exec"), namespace)
    recomputed = namespace["repo_digest"](project, stamp["manifest"])
    result["repo_digest_recomputed"] = recomputed
    print(f"[main-case] repo digest recomputed over the unpacked package: "
          f"{recomputed}", flush=True)
    if recomputed != expected or stamp.get("repo_digest") != expected:
        raise RuntimeError(
            "REPO DIGEST MISMATCH -- refusing: the unpacked package is not the "
            f"verified clone. recomputed {recomputed}, --repo-digest {expected}, "
            f"stamp {stamp.get('repo_digest')}")


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer")
    if value < 0:
        raise argparse.ArgumentTypeError(f"{value} is negative")
    return value


def _cbwr_value(text: str) -> str:
    if not text.strip() and text != "":
        raise argparse.ArgumentTypeError("whitespace only")
    if re.search(r"\s", text):
        raise argparse.ArgumentTypeError(f"{text!r} must be one MKL_CBWR value without whitespace")
    return text


def main() -> int:
    global PACKAGE_DRIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d-%H%M%S-am2020-main-case"),
    )
    parser.add_argument(
        "--only",
        default="",
        help=(
            "series or child-id filter passed through to Allrun and "
            "create_cases.py, e.g. --only demonstration. Empty runs all 19 "
            "children. Validation always runs against the whole controls "
            "file, so a filter cannot slip an invalid child past a refusal."
        ),
    )
    parser.add_argument(
        "--package", default=str(PACKAGE_DRIVE),
        help="the package zip (default: today's Drive upload)",
    )
    parser.add_argument(
        "--repo-digest", default="",
        help=(
            "the notebook's verified EXPECTED_REPO_DIGEST. The runner "
            "recomputes the digest over the unpacked package from "
            "REPO_STAMP.json and refuses on a mismatch."
        ),
    )
    parser.add_argument(
        "--p2prime-solver", default="",
        help=(
            "S22 validation (BATCH 71) only: the P2-prime clone's "
            "user-applications/phaseClotAxisymmetricFluid directory, built into a "
            "separate FOAM_USER_LIBBIN for the child that names solver_library "
            "p2prime. Empty (the default) builds nothing."
        ),
    )
    parser.add_argument(
        "--s22-solver", default="",
        help=(
            "S23 validation (BATCH 72) only: the clone of the S22 release's "
            "user-applications/phaseClotAxisymmetricFluid directory, built into a "
            "separate FOAM_USER_LIBBIN (s22_libbin) for the child that names "
            "solver_library s22. Empty (the default) builds nothing."
        ),
    )
    parser.add_argument(
        "--baseline-solver", default="",
        help=(
            "S25 verification (BATCH 76, B76-OFF-A) only: the clone of release-e2a57db1's "
            "user-applications/phaseClotAxisymmetricFluid directory, built into a separate "
            "FOAM_USER_LIBBIN (baseline_libbin) for the child that names solver_library baseline. "
            "Empty (the default) builds nothing."
        ),
    )
    parser.add_argument(
        "--mkl-threads", type=_non_negative_int, default=0,
        help=(
            "S25 (the 1-core batches): pin MKL_NUM_THREADS to this many solver threads whatever the "
            "cores. 0 (the default) is the S22 rule, physical cores // AM2020_CONCURRENCY."
        ),
    )
    parser.add_argument(
        "--mkl-cbwr", type=_cbwr_value, default="",
        help=(
            "S25 (the 1-core batches): export MKL_CBWR=<branch> and probe, on the session's own "
            "libmkl_rt and in the solver's own call order, that the pin survives mkl_cbwr_set(AUTO). "
            "Recorded in environment.json under s22.mkl_cbwr. Empty (the default) pins nothing."
        ),
    )
    parser.add_argument(
        "--restart-from", default="",
        help=(
            "S25 (OI-65): restart every selected child from its last FULL write (restart_driver.py; never "
            "latestTime, never a frame): a previous session's case-after-run-*.tar.gz, or a directory of "
            "<child>/<time dirs>. Empty (the default) is a cold start; AM2020_RESTART_FROM is then unset."
        ),
    )
    args = parser.parse_args()
    PACKAGE_DRIVE = Path(args.package)
    series_filter = args.only.strip()
    if series_filter:
        print(f"SERIES FILTER: {series_filter!r} -- this is a PARTIAL run; "
              f"result.json will describe only the children it built")
    bootstrap = load_bootstrap()
    run_local = RUNS_LOCAL / args.run_id
    run_drive = DRIVE_ROOT / "runs" / args.run_id
    if run_local.exists() or run_drive.exists():
        raise FileExistsError(f"immutable run destination exists: {args.run_id}")
    artifact = run_local / "artifacts" / ARTIFACT_NAME
    artifact.mkdir(parents=True)
    result = {"run_id": args.run_id, "gates": [g["name"] for g in GATES],
              "series_filter": series_filter or None,
              "partial_run": bool(series_filter)}
    exit_code = 1
    try:
        bootstrap.ensure_openfoam()
        if PROJECT_LOCAL.exists():
            shutil.rmtree(PROJECT_LOCAL)
        PROJECT_LOCAL.mkdir(parents=True)
        # STAGE THE PACKAGE OFF DRIVE BEFORE UNZIPPING IT, AND VERIFY THE COPY.
        #
        # Run 20260901-102201 died on this line with "zipfile read error" on a
        # package that was INTACT: unzip -t clean on 1822 files, and cell 2 of
        # the notebook had already opened the same file and read a member out
        # of it seconds earlier. A truncated upload cannot do that -- a zip's
        # central directory lives at the END of the file, so a short file fails
        # at the first open, not at the second read.
        #
        # What does do it is the Colab Drive FUSE mount failing a long
        # sequential read. Cell 4 of the notebook already pays this toll in the
        # other direction and says so: "Zipping directly out of the mounted
        # Drive reads every byte back over the network." Unzipping straight
        # from Drive has the same exposure and had no guard.
        #
        # So: copy once, verify the COPY by CRC-checking every member, retry
        # the copy if that fails, and unzip from local disk. testzip() reads
        # /content, not the mount, so the verification cannot itself be the
        # flaky read it exists to detect -- and neither can the sha256 below,
        # which used to re-read all 3.3 MB back over FUSE for no reason.
        local_package = CONTENT / PACKAGE_DRIVE.name
        staged = False
        # THE SAME FILE IS NOT COPIED (L1, change 1). shutil.copy2(p, p) raises
        # SameFileError, an OSError, so a package passed from /content/ itself
        # would be "retried" three times and refused AFTER the OpenFOAM
        # install. Only the copy is skipped; the CRC check below still runs.
        same_file = (local_package.exists()
                     and PACKAGE_DRIVE.resolve() == local_package.resolve())
        for attempt in range(1, 4):
            # The COPY is inside the try as well: a failing FUSE read can
            # surface as an OSError out of copy2 rather than as a damaged
            # file, and that is the same fault with the same remedy.
            try:
                if not same_file:
                    shutil.copy2(PACKAGE_DRIVE, local_package)
                with zipfile.ZipFile(local_package) as archive:
                    bad = archive.testzip()
                if bad is None:
                    staged = True
                    break
                reason = f"CRC failure in member {bad}"
            except (OSError, zipfile.BadZipFile) as error:
                reason = f"{type(error).__name__}: {error}"
            tail = "; retrying" if attempt < 3 else ""
            print(
                f"[main-case] package copy attempt {attempt} failed "
                f"({reason}){tail}"
            )
        if not staged:
            raise RuntimeError(
                f"could not stage a readable copy of {PACKAGE_DRIVE} after 3 "
                f"attempts. The package on Drive is damaged, not merely slow "
                f"to read -- re-upload it and check its sha256 against the "
                f"one recorded with the build."
            )
        bootstrap.bash(
            f"unzip -q {bootstrap.quoted(local_package)} -d {bootstrap.quoted(PROJECT_LOCAL)}"
        )
        result["package_sha256"] = sha256(local_package)
        # Recorded beside package_sha256 (L1, change 9): a github package is
        # built in the notebook, so its sha is no release id; the tag and the
        # recomputed repo digest are.
        verify_repo_stamp(PROJECT_LOCAL, args.repo_digest.strip(), result)
        # The 5 ms watcher's width probe must be in the package and numpy must
        # import: refused here, at minute 3, not at 5 ms of a 13 h child.
        if launch_includes_watched(series_filter):
            watcher_preflight(PROJECT_LOCAL / "scripts" / "na03_width.py")
        repo_stamp = PROJECT_LOCAL / "REPO_STAMP.json"
        if repo_stamp.is_file():
            shutil.copy2(repo_stamp, artifact / repo_stamp.name)
        stamp = PROJECT_LOCAL / "PACKAGE_STAMP.json"
        if stamp.is_file():
            shutil.copy2(stamp, artifact / stamp.name)
            result["package_stamp"] = json.loads(stamp.read_text())

        # S22 sections 2.5 and 5: the pinned MKL and its environment, before
        # the solver is built and run (every child inherits the exports).
        # S25 (OI-65): a restart is explicit. An inherited AM2020_RESTART_FROM must never restart a cold launch.
        os.environ.pop("AM2020_RESTART_FROM", None)
        restart_record = None
        if args.restart_from:
            restart_record = stage_restart(
                args.restart_from,
                selected_child_ids(series_filter, PROJECT_LOCAL / "cases" / "AM2020_planar_mural_detachment"
                                   / "constant" / "mainCaseControls.json"),
                PROJECT_LOCAL / "cases" / "AM2020_planar_mural_detachment" / "scripts" / "restart_driver.py",
                CONTENT / "restart_source")
            result["restart"] = restart_record
            print(f"[main-case] S25: restarting from {restart_record['chosen']} ({restart_record['source']})", flush=True)
        s22_env = s22_mkl_environment(args.mkl_threads, args.mkl_cbwr.strip())
        result["s22_mkl_installed"] = s22_env["mkl_installed"]
        solver = PROJECT_LOCAL / "user-applications" / "phaseClotAxisymmetricFluid"
        openfoam_bash = f"source {bootstrap.quoted(bootstrap.OF_BASHRC)}"
        compiled = bootstrap.bash(
            openfoam_bash + f" && cd {bootstrap.quoted(solver)} && bash ./Allwmake",
            check=False,
        )
        result["compile_exit_code"] = compiled.returncode
        (artifact / "log.Allwmake").write_text(compiled.stdout)
        # The solver changed in this package. Warnings are evidence.
        result["compile_warning_count"] = compiled.stdout.lower().count("warning")
        if compiled.returncode != 0:
            result["verdict"] = "COMPILE_FAILED"
            raise RuntimeError("user library compile failed")
        # Reference libraries (S22 arm A0: p2prime; S23 arm (1'): s22), only when the notebook
        # passes the clone. Each selected child's solver_library must have been passed, and no
        # library is built for nothing; both are refused here, before the first build.
        reference_args = {"p2prime": args.p2prime_solver, "s22": args.s22_solver, "baseline": args.baseline_solver}
        check_reference_libraries(
            reference_libraries_needed(
                series_filter, PROJECT_LOCAL / "cases" / "AM2020_planar_mural_detachment"
                / "constant" / "mainCaseControls.json"),
            reference_args)
        references = {}
        for _name, _solver in reference_args.items():
            if not _solver:
                continue
            try:
                references[_name] = build_reference_library(
                    bootstrap, _name, Path(_solver), openfoam_bash,
                    CONTENT / REFERENCE_LIBRARIES[_name][1], artifact)
            except RuntimeError:
                result["verdict"] = REFERENCE_LIBRARIES[_name][3]
                raise
            result[{"p2prime": "p2prime", "s22": "s22_reference", "baseline": "baseline_reference"}[_name]] = references[_name]
        p2prime = references.get("p2prime")

        ignore = shutil.ignore_patterns(
            "__pycache__", "*.pyc", "runs", "postProcessing", "log.*"
        )
        # STAGE THE WHOLE cases/ TREE, not a hand-listed subset.
        #
        # This is the fourth time a run has died because something a generator
        # needed at runtime was not on disk: the shared field template, then
        # the mural gate that the finite-strain gate imports (run
        # 20260810-192247), then the mural gate that the MAIN CASE imports
        # (run 20260810-210833). Each fix named one more directory, and the
        # next cross-case import broke it again.
        #
        # Importing a sibling case rather than copying its geometry is
        # deliberate -- it is what stops the configurations drifting -- so the
        # dependency list is open-ended by design and must not be maintained by
        # hand. The whole tree is text: dictionaries and scripts, with runs/
        # and postProcessing/ stripped by the ignore pattern above. Copying all
        # of it costs a second and removes the failure class.
        shutil.copytree(PROJECT_LOCAL / "cases", run_local, ignore=ignore,
                        dirs_exist_ok=True)
        smoke = run_local / SMOKE_NAME
        if not smoke.is_dir():
            raise RuntimeError(f"shared template missing after staging: {smoke}")
        run_drive.mkdir(parents=True)
        environment = {
            "openfoam_revision": bootstrap.command_output(
                "git", "-C", str(bootstrap.OF_LOCAL), "rev-parse", "HEAD"
            ),
            **bootstrap.build_environment(),
            "s22": s22_env,
            "restart": restart_record,
            "p2prime": p2prime,
            "s22_reference": references.get("s22"),
            "baseline_reference": references.get("baseline"),
        }
        (artifact / "environment.json").write_text(
            json.dumps(environment, indent=2) + "\n"
        )
        shutil.copy2(artifact / "environment.json", run_drive / "environment.json")

        archive_script = PROJECT_LOCAL / "scripts" / "archive_openfoam_evidence.py"
        # sigmaElastic AND be ADDED 2026-08-24. sigmaElastic is the field the
        # wall bond reads its traction from, and be is the state it is built
        # from; both were AUTO_WRITE in the solver and NEITHER was archived, so
        # the bond's own view of the stress could not be looked at. A traction
        # that exists only as a log number is how the zero one went unexamined
        # for the whole Eulerian effort -- see run 20260824-033102.
        fields = (
            "Dg,H,c,w,q,Us,poroPerm,phaseGradient,interfaceDelta,Vn,"
            "plateletLossCoeff,shearRate,U,p,C,Vc,Ccx,Ccy,sigmaElastic,be,"
            # shell IS THE CORE. Dr He Li judged the 20260825-013027 result
            # from a core contour and rejected it; the field that defines
            # which material is where was never archived, so the core could
            # only be inferred from poroPerm. Archived from 2026-08-26.
            "released,bondHold,shell,"
            # thetaC AND thetaH ADDED 2026-09-19, same defect as the two
            # additions above. Both are AUTO_WRITE in the solver and both are
            # in case-after-run-<case>.tar.gz; NEITHER was in this list, so the
            # auditable evidence could not reconstruct
            #     ind = (thetaC + thetaH)/skeletonVolumeFraction,
            # which is the quantity the release and bond-hold masks threshold
            # after S5. Run 20260919-083453 was read against a chi proxy for
            # that reason, and the proxy was mistaken for the solver's own
            # indicator. ASCII on purpose: measured on that run's thetaC,
            # gzipped binary is 147541 B against ascii's 156185 B -- 5.5% --
            # and binary would break every ascii field reader in scripts/.
            "thetaC,thetaH,"
            # The Xu 2017 Figure 5 verdicts (design 2026-09-29-case-xu2017-
            # fig5, section 4; assess_case.py xu2017_report) read these five:
            # the stiffness contrast (Gfield: A1, the bulk-core population of
            # N4), the damage partition (d* = (thetaPending + thetaBroken)/
            # theta: P3, N4, B) and the drained-cell reference (thetaNetRef:
            # N5). Added 2026-09-30 so the evidence archive can recompute them
            # (PIPELINE.md, the evidence contract); a child that does not write
            # one simply has none archived (archive_openfoam_evidence.py L140).
            "Gfield,thetaIntact,thetaPending,thetaBroken,thetaNetRef"
        )

        # STAGE EVERY GATE BEFORE RUNNING ANY OF THEM.
        #
        # Run 20260810-192247 staged and ran each gate in turn, and the finite
        # strain gate -- which imports the mural gate's generator and control
        # file so the two configurations cannot drift -- found no mural gate on
        # disk, because it had not been copied yet. create_cases.py exited, the
        # Allrun stopped, and the gate produced no result.json at all.
        #
        # Cross-case dependencies are the point of importing rather than
        # copying, so staging is a separate pass over an ALREADY-STAGED tree:
        # the whole of cases/ was copied above, and this loop only records the
        # inputs and drops in the build log.
        for spec in GATES:
            name = spec["name"]
            gate = run_local / name
            if not gate.is_dir():
                raise RuntimeError(
                    f"{name} is not in the staged cases/ tree: {gate}"
                )
            # The assessor reads the build log for `library_compiled`, and the
            # ignore pattern strips log.*, so copy it after the tree.
            shutil.copy2(artifact / "log.Allwmake", gate / "log.Allwmake")
            shutil.copytree(gate, run_drive / f"case-input-{name}")
            print(f"staged {name}", flush=True)

        result["gates"] = {}
        all_passed = True
        for spec in GATES:
            name = spec["name"]
            print(f"\n{'='*70}\n== {name}\n{'='*70}", flush=True)
            gate = run_local / name

            gate_artifact = artifact / name
            gate_artifact.mkdir(parents=True, exist_ok=True)

            # SERIES FILTER, passed straight through to Allrun and on to
            # create_cases.py. `--only demonstration` builds and runs the three
            # demonstration children (~20 min) instead of all nineteen
            # (~3.5 h), which is what makes iterating on the transport
            # behaviour practical. matrix.json records what was actually built,
            # so a filtered run cannot be mistaken for a full one downstream.
            # Allrun, the 5 ms watcher beside it, then the archive: the
            # archive runs after a watcher stop exactly as after a normal end
            # (run_children_and_archive).
            code, archive_code, child_ids = run_children_and_archive(
                gate, openfoam_bash, series_filter, archive_script,
                gate_artifact, fields, PROJECT_LOCAL / "scripts" / "na03_width.py")
            watcher = {}
            for child_id in WATCHED_CHILDREN:
                verdict_file = gate / "runs" / child_id / WATCHER_VERDICT
                if verdict_file.is_file():
                    shutil.copy2(verdict_file, gate_artifact / f"watcher-verdict-{child_id}.json")
                    record = json.loads(verdict_file.read_text())
                    watcher[child_id] = {
                        "verdict": record.get("verdict"),
                        "stopped": bool(record.get("stop", {}).get("touched")),
                        "stop_acknowledged": bool(record.get("stop", {}).get("acknowledged")),
                    }
                    print(f"[watcher] {child_id}: {watcher[child_id]}", flush=True)

            source = gate / "postProcessing" / spec["outputs"] / "result.json"
            if source.is_file():
                shutil.copy2(source, gate_artifact / "result.json")
            for extra in (gate / "runs" / "matrix.json",
                          gate / "constant" / spec["controls"]):
                if extra.is_file():
                    shutil.copy2(extra, gate_artifact / extra.name)

            entry = {"gate_exit_code": code, "archive_exit_code": archive_code}
            if watcher:
                entry["watcher"] = watcher
            if source.is_file():
                assessed = json.loads(source.read_text())
                # The main case reports tables, not a verdict. "Complete"
                # means the library built, every child was admissible, and the
                # evidence was archived -- NOT that the physics came out any
                # particular way. Reading a result as a pass/fail here would
                # be the runner deciding what the answer should have been.
                entry["status"] = (
                    "VALID" if assessed["library_compiled"] else "INVALID"
                )
                entry["children_admissible"] = assessed["children_admissible"]
                entry["children_total"] = assessed["children_total"]
                entry["inadmissible"] = assessed["inadmissible"]
                entry["verdict"] = (
                    "ALL_CHILDREN_ADMISSIBLE"
                    if assessed["children_admissible"] == assessed["children_total"]
                    else f"{len(assessed['inadmissible'])}_CHILDREN_EXCLUDED"
                )
                entry["result"] = assessed
            else:
                entry["status"], entry["verdict"] = "INVALID", "NO_ASSESSOR_RESULT"
            entry["passed"] = bool(
                archive_code == 0
                and entry.get("status") == "VALID"
                and entry.get("children_admissible") == entry.get("children_total")
            )
            all_passed = all_passed and entry["passed"]
            result["gates"][name] = entry
            print(f"[{name}] {entry['status']}/{entry['verdict']}", flush=True)

            # ONE tarball per gate, verified. Run 20260810-010103 lost two of
            # three children to an interrupted recursive Drive copy; the tar
            # either arrives whole or does not arrive, and run 20260810-015336
            # confirmed it also carries none of the c.c / H.h / p.pas name
            # corruption the unpacked tree picks up.
            local_files = sorted(
                str(f.relative_to(gate)) for f in gate.rglob("*") if f.is_file()
            )
            tarball = run_local / f"case-after-run-{name}.tar.gz"
            with tarfile.open(tarball, "w:gz") as tf:
                tf.add(gate, arcname=f"case-after-run-{name}")
            digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
            drive_tarball = run_drive / tarball.name
            shutil.copy2(tarball, drive_tarball)
            landed = hashlib.sha256(drive_tarball.read_bytes()).hexdigest()
            with tarfile.open(drive_tarball, "r:gz") as tf:
                members = sum(1 for m in tf.getmembers() if m.isfile())
            integrity = {
                "local_file_count": len(local_files),
                "tar_member_count": members,
                "local_sha256": digest, "drive_sha256": landed,
                "intact": bool(landed == digest and members == len(local_files)),
            }
            entry["case_after_run_integrity"] = integrity
            (run_drive / f"case_after_run_integrity-{name}.json").write_text(
                json.dumps({**integrity, "files": local_files}, indent=2) + "\n"
            )
            print(
                f"[archive] {name}: "
                + ("intact" if integrity["intact"] else "TRUNCATED")
                + f", {members} of {len(local_files)} files",
                flush=True,
            )
            if not integrity["intact"]:
                all_passed = False
            # THE UNPACKED TREE IS A CONVENIENCE COPY AND IT IS NOT FREE.
            #
            # It is byte-for-byte redundant with the tarball above, which this
            # function already treats as the evidence of record -- see the
            # comment at the top of this block, and the except branch that
            # said so when the copy failed. It is also the copy that picks up
            # the c.c / H.h / p.pas case-insensitivity corruption.
            #
            # MEASURED on run 20260906-004750: unpacked 1.3 GB against a
            # 446 MB tarball, i.e. 2.9x, for the same bytes. Projected over a
            # child that survives to endTime = 0.03 at writeInterval 2.5e-05
            # -- 1200 write times at ~19.5 MB each -- that is ~23 GB per child
            # of pure duplication in the download, and Batch 20's children
            # only escaped it by dying at 4.6e-05 s.
            #
            # WHAT IS KEPT: run-exit-code.txt, every log, child.json,
            # constant/ and system/ -- everything the notebook's failure path
            # and the assessor's own reporting read. WHAT IS DROPPED: the
            # numeric time directories, which are the field data and are in
            # the tarball. Extract locally with
            #   tar xzf case-after-run-<name>.tar.gz
            def _skip_time_dirs(directory, entries):
                if RETURN_UNPACKED_TIME_DIRS:
                    return []
                skipped = []
                for entry in entries:
                    if not os.path.isdir(os.path.join(directory, entry)):
                        continue
                    try:
                        float(entry)
                    except ValueError:
                        continue
                    skipped.append(entry)
                return skipped

            try:
                shutil.copytree(gate, run_drive / f"case-after-run-{name}",
                                dirs_exist_ok=True, ignore=_skip_time_dirs)
                if not RETURN_UNPACKED_TIME_DIRS:
                    print(f"[archive] {name}: unpacked tree copied WITHOUT "
                          "time directories; fields are in the tarball "
                          "(tar xzf case-after-run-*.tar.gz)", flush=True)
            except Exception as tree_error:
                print(f"[archive] unpacked tree copy failed ({tree_error}); "
                      "the tarball is the evidence of record", flush=True)

        # The main case has no pass verdict of its own: "complete" means every
        # child ran and the assessor produced its tables, not that the physics
        # came out any particular way.
        result["verdict"] = (
            "MAIN_CASE_COMPLETE" if all_passed else "MAIN_CASE_INCOMPLETE"
        )
        exit_code = 0 if all_passed else 1
    except Exception as error:
        result["error"] = str(error)
        print(f"[main-case] ERROR: {error}", flush=True)
    finally:
        result["runner_exit_code"] = exit_code
        (artifact / "runner_result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        (artifact / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "contract": "complete logs, assessor result, controls, all written fields, verified Drive retention",
                    "files": manifest_tree(artifact),
                },
                indent=2,
            )
            + "\n"
        )
        # The artifact is small and is copied FIRST; each gate's bulk tree
        # already travelled as its own verified tarball above.
        drive_artifact = run_drive / "artifacts" / ARTIFACT_NAME
        shutil.copytree(artifact, drive_artifact, dirs_exist_ok=True)

        try:
            verification = verify_drive_copy(artifact, drive_artifact)
            (drive_artifact / "drive_copy_verification.json").write_text(
                json.dumps(verification, indent=2) + "\n"
            )
            print(f"[archive] verified {verification['file_count']} artifact files on Drive", flush=True)
        except Exception as verification_error:
            exit_code = 1
            print(f"[archive] VERIFICATION ERROR: {verification_error}", flush=True)
        print(f"[archive] {run_drive}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
