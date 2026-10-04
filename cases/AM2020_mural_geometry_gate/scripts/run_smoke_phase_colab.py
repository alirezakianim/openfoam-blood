#!/usr/bin/env python3
"""Restore OpenFOAM, run BOTH smoke gates of the mural phase, retain evidence.

Gate A isolates the finite-strain constitutive change on the proven
axisymmetric mesh; gate B runs the new planar one-sided mural geometry. They
are separate gates on purpose - confounding a constitutive change with a
geometry change is how this project has repeatedly lost a cycle - but they
share one package and one run so the smoke phase costs a single upload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import tarfile
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path


DRIVE_ROOT = Path("/content/drive/MyDrive/OpenFOAM_Blood")
PACKAGE_DRIVE = (
    DRIVE_ROOT / "packages" / "AM2020_mural_smoke_phase_colab_package.zip"
)
PROJECT_LOCAL = Path("/content/AM2020_mural_smoke_phase_package")
RUNS_LOCAL = Path("/content/am2020_mural_smoke_phase_runs")
SMOKE_NAME = "AM2020_straight_tube_growth_mechanics_smoke"
ARTIFACT_NAME = "mural_smoke_phase_v1"

# Two gates, run in order. Gate A must pass before gate B's result means
# anything: if the constitutive law is wrong, every displacement in the mural
# geometry is wrong too. B still RUNS either way, because a second Colab cycle
# to collect its evidence is expensive and its logs are informative regardless;
# the notebook reports A first and says plainly when B is not to be trusted.
GATES = (
    {
        "name": "AM2020_finite_strain_gate",
        "outputs": "finiteStrainGate",
        "controls": "finiteStrainControls.json",
        "pass_verdict": "FINITE_STRAIN_READY",
    },
    {
        "name": "AM2020_mural_geometry_gate",
        "outputs": "muralGeometryGate",
        "controls": "muralGeometryControls.json",
        "pass_verdict": "MURAL_READY",
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
        default=datetime.now().strftime("%Y%m%d-%H%M%S-am2020-mural-smoke-phase"),
    )
    args = parser.parse_args()
    bootstrap = load_bootstrap()
    run_local = RUNS_LOCAL / args.run_id
    run_drive = DRIVE_ROOT / "runs" / args.run_id
    if run_local.exists() or run_drive.exists():
        raise FileExistsError(f"immutable run destination exists: {args.run_id}")
    artifact = run_local / "artifacts" / ARTIFACT_NAME
    artifact.mkdir(parents=True)
    result = {"run_id": args.run_id, "gates": [g["name"] for g in GATES]}
    exit_code = 1
    try:
        bootstrap.ensure_openfoam()
        if PROJECT_LOCAL.exists():
            shutil.rmtree(PROJECT_LOCAL)
        PROJECT_LOCAL.mkdir(parents=True)
        bootstrap.bash(
            f"unzip -q {bootstrap.quoted(PACKAGE_DRIVE)} -d {bootstrap.quoted(PROJECT_LOCAL)}"
        )
        result["package_sha256"] = sha256(PACKAGE_DRIVE)
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
        fields = (
            "Dg,H,c,w,q,Us,poroPerm,phaseGradient,interfaceDelta,Vn,"
            "plateletLossCoeff,shearRate,U,p,C,Vc,Ccx,Ccy"
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

            code = stream(
                openfoam_bash + f" && cd {bootstrap.quoted(gate)} && bash ./Allrun"
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

            source = gate / "postProcessing" / spec["outputs"] / "result.json"
            if source.is_file():
                shutil.copy2(source, gate_artifact / "result.json")
            for extra in (gate / "runs" / "matrix.json",
                          gate / "constant" / spec["controls"]):
                if extra.is_file():
                    shutil.copy2(extra, gate_artifact / extra.name)

            entry = {"gate_exit_code": code, "archive_exit_code": archive_code}
            if source.is_file():
                assessed = json.loads(source.read_text())
                entry["status"] = assessed["status"]
                entry["verdict"] = assessed["verdict"]
                entry["checks"] = assessed["checks"]
                entry["result"] = assessed
            else:
                entry["status"], entry["verdict"] = "INVALID", "NO_ASSESSOR_RESULT"
            entry["passed"] = bool(
                archive_code == 0
                and entry.get("status") == "VALID"
                and entry.get("verdict") == spec["pass_verdict"]
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
            try:
                shutil.copytree(gate, run_drive / f"case-after-run-{name}",
                                dirs_exist_ok=True)
            except Exception as tree_error:
                print(f"[archive] unpacked tree copy failed ({tree_error}); "
                      "the tarball is the evidence of record", flush=True)

        result["verdict"] = (
            "SMOKE_PHASE_READY" if all_passed else "SMOKE_PHASE_INCOMPLETE"
        )
        exit_code = 0 if all_passed else 1
    except Exception as error:
        result["error"] = str(error)
        print(f"[mural-smoke-phase] ERROR: {error}", flush=True)
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
