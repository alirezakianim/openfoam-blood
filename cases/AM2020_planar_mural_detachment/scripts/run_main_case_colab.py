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
import shutil
import subprocess
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


DRIVE_ROOT = Path("/content/drive/MyDrive/OpenFOAM_Blood")
PACKAGE_DRIVE = (
    DRIVE_ROOT / "packages" / "AM2020_main_case_colab_package.zip"
)
PROJECT_LOCAL = Path("/content/AM2020_main_case_package")
RUNS_LOCAL = Path("/content/am2020_main_case_runs")
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
    local = Path("/content/am2020_mural_smoke_phase_bootstrap.py")
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


def main() -> int:
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
    args = parser.parse_args()
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
        local_package = Path("/content") / PACKAGE_DRIVE.name
        staged = False
        for attempt in range(1, 4):
            # The COPY is inside the try as well: a failing FUSE read can
            # surface as an OSError out of copy2 rather than as a damaged
            # file, and that is the same fault with the same remedy.
            try:
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
        stamp = PROJECT_LOCAL / "PACKAGE_STAMP.json"
        if stamp.is_file():
            shutil.copy2(stamp, artifact / stamp.name)
            result["package_stamp"] = json.loads(stamp.read_text())

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
            code = stream(
                openfoam_bash + f" && cd {bootstrap.quoted(gate)} && "
                f"bash ./Allrun {series_filter}"
            )
            # CHILD LIST FROM matrix.json, not from this file. The hard-coded
            # tuple that used to live in GATES went stale the moment the
            # finite-strain gate's children were renamed, and the archiver
            # would have quietly archived nothing for it -- the same failure
            # that cost run 20260810-075435, one layer further out.
            matrix_path = gate / "runs" / "matrix.json"
            if matrix_path.is_file():
                child_ids = [
                    entry["child_id"]
                    for entry in json.loads(matrix_path.read_text())
                ]
            else:
                child_ids = []
                print(f"[archive] {name}: no matrix.json, nothing to archive",
                      flush=True)
            archive_code = subprocess.run(
                ["python3", str(archive_script), "--gate", str(gate),
                 "--artifact", str(gate_artifact), "--children",
                 *child_ids, "--tail-start", "0", "--fields", fields],
                check=False,
            ).returncode if child_ids else 1

            # STAGE 3 HAS ITS OWN ASSESSOR AND ITS OWN VERDICT.
            #
            # assess_case.py reports the main case's tables and deliberately
            # does not return a pass/fail. Stage 3 does return one -- it is a
            # go/no-go for stage 4, and the plan's decision point says stop and
            # re-plan below a 3x speedup. Run whenever stage-3 children are in
            # the matrix, which covers the filtered stage-3-only invocation and
            # the full run alike, and printed in full so the numbers are in the
            # console even when the exit code is all that survives.
            # THE PREFIX LIST IS THE TRIGGER AND IT HAS TO TRACK THE CHILD
            # NAMES. Run 20260818-081527 ran five `diag_*` children and this
            # line tested startswith("stage3_") alone, so assess_stage3.py never
            # ran, no stage3_result.json was written, and the det(be) gate that
            # had just been added and validated against the real logs never
            # fired. Every number from that run had to be re-extracted from the
            # raw solver logs by hand. A gate that does not run is worth less
            # than no gate, because the empty artifact reads as "nothing to
            # report".
            # THE TRIGGER IS NOW DERIVED FROM THE SERIES, NOT FROM A LITERAL
            # LIST OF PREFIXES. The list below is kept only as a union so that
            # nothing which used to run stops running.
            #
            # THE PREFIX LIST FAILED A SECOND TIME, run 20260823-052707. Two
            # new children named force_cell and force_face matched none of the
            # prefixes, so `stage3` was empty, assess_stage3.py was NEVER
            # INVOKED, no stage3_entry was recorded, and the run reported
            # gate_exit_code 0 / status VALID / passed True with the entire
            # stage-3 gate absent. Identical in kind to run 20260818-081527,
            # which is what the comment above was written about: the mitigation
            # chosen then was to extend the list, and extending a list only
            # works until the next child is named something new.
            #
            # A series cannot be forgotten the way a prefix can, because
            # create_cases.py refuses a child whose series is not declared in
            # the controls file. Every stage-3 child lives in a series whose
            # name begins "eulerian" -- eulerian_diag_xu, eulerian_diag_li,
            # eulerian_stage3_xu, eulerian_stage3_li, eulerian_drift_probe.
            stage3_series = set()
            if matrix_path.is_file():
                for entry in json.loads(matrix_path.read_text()):
                    if str(entry.get("series", "")).startswith("eulerian"):
                        stage3_series.add(entry["child_id"])

            stage3 = sorted(
                stage3_series
                | {
                    c for c in child_ids
                    if c.startswith((
                        "stage3_", "diag_", "fix_", "anchor_", "flux_",
                        "grad_", "renorm_", "plt_", "gate_", "drift_",
                        "force_",
                    ))
                }
            )
            if stage3:
                print("\n" + "=" * 72, flush=True)
                print(f"STAGE 3 -- {len(stage3)} children: "
                      f"{', '.join(sorted(stage3))}", flush=True)
                print("=" * 72, flush=True)
                s3 = subprocess.run(
                    ["python3", str(gate / "scripts" / "assess_stage3.py")],
                    cwd=str(gate), capture_output=True, text=True, check=False,
                )
                print(s3.stdout + s3.stderr, flush=True)
                s3_json = gate / "runs" / "stage3_result.json"
                if s3_json.is_file():
                    shutil.copy2(s3_json, gate_artifact / "stage3_result.json")
                stage3_entry = {
                    "children": sorted(stage3),
                    "exit_code": s3.returncode,
                    "verdict": "PASSED" if s3.returncode == 0 else "FAILED",
                }
                # A MISSING RESULT IS A FAILURE, NOT A QUIET GAP. If children
                # were selected for this gate and it produced no JSON, the
                # numbers are unassessed -- and an absent artifact reads as
                # "nothing to report" rather than "not measured", which is how
                # run 20260818-081527 lost a whole run's det(be) gate and how
                # run 20260823-052707 reported VALID with the gate absent.
                if not s3_json.is_file():
                    stage3_entry["verdict"] = "FAILED"
                    stage3_entry["no_result_json"] = (
                        "assess_stage3.py produced no stage3_result.json for "
                        f"{len(stage3)} selected children. The gate did not "
                        "run or refused; its numbers are UNASSESSED and this "
                        "run must not be read as a pass."
                    )
                    print(f"[stage3] REFUSED: {stage3_entry['no_result_json']}",
                          flush=True)

            source = gate / "postProcessing" / spec["outputs"] / "result.json"
            if source.is_file():
                shutil.copy2(source, gate_artifact / "result.json")
            for extra in (gate / "runs" / "matrix.json",
                          gate / "constant" / spec["controls"]):
                if extra.is_file():
                    shutil.copy2(extra, gate_artifact / extra.name)

            entry = {"gate_exit_code": code, "archive_exit_code": archive_code}
            if stage3:
                entry["stage3"] = stage3_entry
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
