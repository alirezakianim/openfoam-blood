#!/usr/bin/env python3
"""Run the platelet initial-condition gate in a Drive-backed Colab runtime."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


DRIVE_ROOT = Path("/content/drive/MyDrive/OpenFOAM_Blood")
PACKAGE_DRIVE = (
    DRIVE_ROOT / "packages"
    / "AM2020_paper_cylinder_platelet_initial_condition_colab_package.zip"
)
SOFTWARE_DRIVE = DRIVE_ROOT / "software"
USER_LIB_DRIVE = DRIVE_ROOT / "cache" / "openfoam-user-libs-v13"
RUNS_DRIVE = DRIVE_ROOT / "runs"
PROJECT_LOCAL = Path(
    "/content/AM2020_paper_cylinder_platelet_initial_condition_colab_package"
)
OF_LOCAL = Path("/content/OpenFOAM-13")
OF_BASHRC = OF_LOCAL / "etc" / "bashrc"
RUNS_LOCAL = Path("/content/phase_clot_paper_cylinder_platelet_initial_runs")
EXPECTED_OPENFOAM_COMMIT = "18870c24d21c6b982e2cdec27b2f59738cca5f90"
OPENFOAM_ARCHIVE_FORMAT = 1
UNRELATED_APT_HOSTS = ("cloud.r-project.org", "r2u.stat.illinois.edu")


def quoted(path: Path | str) -> str:
    return shlex.quote(str(path))


def bash(command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    # flush=True on BOTH prints. Without it these sit in the block buffer that
    # Python uses when stdout is a pipe, so a caller watching the notebook sees
    # nothing until 8 KB has accumulated. Combined with the capture below - this
    # prints only after the command FINISHES - a silent 20-minute OpenFOAM
    # restore is indistinguishable from a hang. Run 20260805-090117 was reported
    # as stuck for that reason while it was progressing normally.
    # STREAMED, 2026-09-13. The capture-then-print version printed nothing
    # until the command FINISHED, so every phase that goes through here -- the
    # OpenFOAM restore and both compiles -- was silent for minutes. Run
    # 20260805-090117 was reported as stuck for that reason while it was
    # progressing normally, and run 20260913-042158 was reported as showing no
    # live log at all. The comment above described the defect accurately for
    # two runs without it being fixed.
    #
    # The CompletedProcess is still returned with its full `stdout`, because
    # callers count warnings in it and archive it. Nothing downstream changes.
    print("\n$", command, flush=True)
    process = subprocess.Popen(
        ["bash", "-lc", "set -eo pipefail\n" + command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    captured = []
    for line in process.stdout:
        captured.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    result = subprocess.CompletedProcess(
        args=command, returncode=returncode, stdout="".join(captured),
        stderr=None,
    )
    if check and returncode != 0:
        raise RuntimeError(f"Command failed with exit code {returncode}")
    return result


def command_output(*args: str) -> str:
    result = subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return (
        result.stdout.strip()
        if result.returncode == 0
        else f"unavailable: {result.stdout.strip()}"
    )


def build_environment() -> dict[str, str]:
    release = (
        Path("/etc/os-release").read_text()
        if Path("/etc/os-release").is_file()
        else "unavailable"
    )
    return {
        "machine": platform.machine(),
        "os_release": release,
        "gcc": command_output("gcc", "-dumpfullversion", "-dumpversion"),
        "openmpi": command_output("mpirun", "--version").splitlines()[0],
    }


def filtered_apt_source(text: str, deb822: bool) -> tuple[str, int]:
    """Remove only unrelated Colab R-repository entries."""
    if deb822:
        chunks = text.split("\n\n")
        kept = [
            chunk
            for chunk in chunks
            if not any(host in chunk for host in UNRELATED_APT_HOSTS)
        ]
        removed = len(chunks) - len(kept)
        filtered = "\n\n".join(chunk for chunk in kept if chunk.strip())
        return (filtered.rstrip() + "\n" if filtered else "", removed)
    lines = text.splitlines()
    kept = [
        line
        for line in lines
        if not any(host in line for host in UNRELATED_APT_HOSTS)
    ]
    removed = len(lines) - len(kept)
    return ("\n".join(kept).rstrip() + "\n" if kept else "", removed)


def prepare_apt_sources(apt_root: Path = Path("/etc/apt")) -> tuple[Path, Path]:
    """Create a temporary official-repository view without mutating Colab."""
    temporary = Path(tempfile.mkdtemp(prefix="am2020-apt-"))
    source_list = temporary / "sources.list"
    source_parts = temporary / "sources.list.d"
    source_parts.mkdir()
    removed = 0
    original_list = apt_root / "sources.list"
    if original_list.is_file():
        text, count = filtered_apt_source(original_list.read_text(), False)
        source_list.write_text(text)
        removed += count
    else:
        source_list.write_text("")
    original_parts = apt_root / "sources.list.d"
    if original_parts.is_dir():
        for source in sorted(original_parts.iterdir()):
            if not source.is_file() or source.suffix not in {".list", ".sources"}:
                continue
            text, count = filtered_apt_source(
                source.read_text(),
                source.suffix == ".sources",
            )
            (source_parts / source.name).write_text(text)
            removed += count
    active_text = source_list.read_text() + "".join(
        path.read_text() for path in source_parts.iterdir() if path.is_file()
    )
    if not (
        any(line.lstrip().startswith("deb ") for line in active_text.splitlines())
        or "URIs:" in active_text
    ):
        raise RuntimeError("Filtered APT view contains no active Ubuntu repository.")
    print(
        "Prepared temporary APT source view; omitted "
        f"{removed} unrelated CRAN/r2u entr{'y' if removed == 1 else 'ies'}."
    )
    return source_list, source_parts


def ensure_openfoam() -> None:
    for directory in (
        DRIVE_ROOT,
        PACKAGE_DRIVE.parent,
        SOFTWARE_DRIVE,
        USER_LIB_DRIVE,
        RUNS_DRIVE,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if not PACKAGE_DRIVE.is_file():
        raise FileNotFoundError(
            f"Upload the platelet initial-condition package to {PACKAGE_DRIVE}"
        )
    source_list, source_parts = prepare_apt_sources()
    apt_options = (
        f"-o Dir::Etc::sourcelist={quoted(source_list)} "
        f"-o Dir::Etc::sourceparts={quoted(source_parts)} "
        "-o Acquire::Retries=3 "
        "-o Acquire::http::Timeout=30 "
        "-o Acquire::https::Timeout=30"
    )
    bash(
        f"apt-get {apt_options} update -qq && "
        f"DEBIAN_FRONTEND=noninteractive apt-get {apt_options} install "
        "-y -qq build-essential bison flex git m4 cmake zlib1g-dev "
        "libopenmpi-dev openmpi-bin libscotch-dev libptscotch-dev "
        "libfftw3-dev libboost-system-dev libreadline-dev libncurses-dev "
        "libxt-dev rsync zstd "
        # S19 (research/designs/2026-10-03-S19-block-coupled-step.md section
        # 2, Build): UMFPACK for the block-coupled step and OpenBLAS under it
        # (reference BLAS is 2-3x slower, docket 06 section 3). On every
        # bootstrap, cache hit included; the OpenFOAM cache key
        # (build_environment) does not read this line.
        "libsuitesparse-dev libopenblas-dev"
    )
    # S19: the BLAS the solver will link through, recorded in the log.
    bash(
        "update-alternatives --display libblas.so.3 "
        "|| echo 'libblas.so.3: no alternatives entry'"
    )
    environment = build_environment()
    key = hashlib.sha256(
        json.dumps(environment, sort_keys=True).encode()
    ).hexdigest()[:12]
    archive_name = (
        f"OpenFOAM-13-{EXPECTED_OPENFOAM_COMMIT}-colab-{key}.tar.zst"
    )
    archive_drive = SOFTWARE_DRIVE / archive_name
    checksum_drive = Path(str(archive_drive) + ".sha256")
    manifest_drive = Path(str(archive_drive) + ".json")
    archive_local = Path("/content") / archive_name
    checksum_local = Path(str(archive_local) + ".sha256")
    manifest_local = Path(str(archive_local) + ".json")

    if OF_LOCAL.is_dir():
        print(f"Reusing local OpenFOAM tree: {OF_LOCAL}")
    elif archive_drive.is_file():
        if not checksum_drive.is_file() or not manifest_drive.is_file():
            raise RuntimeError(f"Incomplete OpenFOAM archive cache: {archive_drive}")
        manifest = json.loads(manifest_drive.read_text())
        if (
            manifest.get("cache_format") != OPENFOAM_ARCHIVE_FORMAT
            or manifest.get("openfoam_commit") != EXPECTED_OPENFOAM_COMMIT
            or manifest.get("environment") != environment
        ):
            raise RuntimeError("OpenFOAM cache manifest does not match this runtime.")
        bash(
            f"cp {quoted(archive_drive)} {quoted(archive_local)} && "
            f"cp {quoted(checksum_drive)} {quoted(checksum_local)}"
        )
        archive_sha = bash(
            f"cd /content && sha256sum -c {quoted(checksum_local.name)} && "
            f"sha256sum {quoted(archive_local.name)}"
        ).stdout.strip().splitlines()[-1].split()[0]
        if manifest.get("archive_sha256") != archive_sha:
            raise RuntimeError("OpenFOAM archive checksum differs from its manifest.")
        bash(f"cd /content && tar --zstd -xf {quoted(archive_local.name)}")
    else:
        bash(
            f"git clone --no-checkout "
            f"https://github.com/OpenFOAM/OpenFOAM-13.git {quoted(OF_LOCAL)} && "
            f"git -C {quoted(OF_LOCAL)} fetch --depth 1 origin "
            f"{EXPECTED_OPENFOAM_COMMIT} && git -C {quoted(OF_LOCAL)} checkout "
            f"{EXPECTED_OPENFOAM_COMMIT}"
        )

    commit = bash(f"git -C {quoted(OF_LOCAL)} rev-parse HEAD").stdout.strip().splitlines()[-1]
    if commit != EXPECTED_OPENFOAM_COMMIT:
        raise RuntimeError(f"OpenFOAM revision mismatch: {commit}")
    if not list(OF_LOCAL.glob("platforms/**/bin/foamRun")):
        bash(
            f"source {quoted(OF_BASHRC)} && export ZOLTAN_TYPE=none && "
            f"cd {quoted(OF_LOCAL)} && ./Allwmake -j$(nproc)"
        )
    if not list(OF_LOCAL.glob("platforms/**/bin/foamRun")):
        raise RuntimeError("OpenFOAM build did not produce foamRun.")
    # WRITING THE CACHE IS A SIDE-BENEFIT AND MUST NEVER ABORT THE RUN.
    #
    # It used to. Run 20260805-090117 compiled OpenFOAM from source for about
    # four hours - correctly, after a Colab image bump changed the environment
    # hash and invalidated the cache - produced foamRun, and then raised
    # "Refusing to overwrite a partial immutable cache" because a `.sha256` or
    # `.json` sidecar from an earlier interrupted write was still sitting in
    # Drive WITHOUT its archive. Nine hours of wall time bought nothing, and the
    # probe never ran, over a bookkeeping leftover discovered AFTER all the work
    # and BEFORE any of it was used.
    #
    # Two things were wrong and both are fixed here:
    #   1. the guard raised where it could have cleaned up. A sidecar with no
    #      archive beside it is USELESS BY CONSTRUCTION - nothing can restore
    #      from it - so removing it is safe, and refusing to is not "safety",
    #      it is a permanent wedge that every later run hits identically.
    #   2. the whole block was fatal. Caching is an optimisation for NEXT time;
    #      failing to store it has no bearing on whether THIS run can proceed,
    #      and the build it would have cached is already sitting in OF_LOCAL.
    #
    # The archive itself is still never overwritten once complete - that part of
    # the original intent is preserved.
    if not archive_drive.exists():
        try:
            for stale in (checksum_drive, manifest_drive, archive_local):
                if stale.exists():
                    print(
                        f"[cache] removing orphaned {stale.name}: its archive is "
                        "absent, so nothing could ever restore from it.",
                        flush=True,
                    )
                    stale.unlink()
            bash(
                f"tar --zstd -cf {quoted(archive_local)} -C /content "
                f"{quoted(OF_LOCAL.name)}"
            )
            archive_sha = bash(
                f"sha256sum {quoted(archive_local)}"
            ).stdout.strip().split()[0]
            checksum_local.write_text(f"{archive_sha}  {archive_local.name}\n")
            manifest_local.write_text(
                json.dumps(
                    {
                        "cache_format": OPENFOAM_ARCHIVE_FORMAT,
                        "openfoam_commit": commit,
                        "environment": environment,
                        "archive_sha256": archive_sha,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            # The archive LAST, so the sidecars are never the thing left behind
            # without it - the exact state that wedged run 20260805-090117.
            bash(
                f"cp {quoted(checksum_local)} {quoted(checksum_drive)} && "
                f"cp {quoted(manifest_local)} {quoted(manifest_drive)} && "
                f"cp {quoted(archive_local)} {quoted(archive_drive)}"
            )
            print(f"[cache] stored {archive_drive.name}", flush=True)
        except Exception as error:  # noqa: BLE001 - caching must not sink a run
            print(
                f"[cache] WARNING: could not store the OpenFOAM archive: {error}\n"
                "[cache] The run CONTINUES - the build is already in "
                f"{OF_LOCAL} and this only means the next run rebuilds.",
                flush=True,
            )
    bash(f"source {quoted(OF_BASHRC)} && foamVersion")


def main() -> int:
    run_id = datetime.now().strftime(
        "%Y%m%d-%H%M%S-am2020-paper-cylinder-platelet-initial-condition"
    )
    print(f"Platelet initial-condition gate: {run_id}")
    ensure_openfoam()
    if PROJECT_LOCAL.is_dir():
        raise FileExistsError(
            f"Restart the runtime before rerunning this immutable gate: "
            f"{PROJECT_LOCAL} already exists."
        )
    bash(
        f"mkdir -p {quoted(PROJECT_LOCAL)} && "
        f"unzip -q {quoted(PACKAGE_DRIVE)} -d {quoted(PROJECT_LOCAL)}"
    )
    openfoam_bash = f"source {quoted(OF_BASHRC)}"
    if USER_LIB_DRIVE.is_dir():
        bash(
            openfoam_bash
            + f' && mkdir -p "$FOAM_USER_LIBBIN" && rsync -a '
            + f'{quoted(USER_LIB_DRIVE)}/ "$FOAM_USER_LIBBIN"/'
        )
    solver = PROJECT_LOCAL / "user-applications" / "phaseClotAxisymmetricFluid"
    bash(openfoam_bash + f" && cd {quoted(solver)} && ./Allwmake")
    bash(
        openfoam_bash
        + f' && mkdir -p {quoted(USER_LIB_DRIVE)} && rsync -a '
        + f'"$FOAM_USER_LIBBIN"/ {quoted(USER_LIB_DRIVE)}/'
    )

    gate_template = (
        PROJECT_LOCAL / "cases"
        / "AM2020_paper_cylinder_platelet_initial_condition_gate"
    )
    coarse_template = (
        PROJECT_LOCAL / "cases" / "AM2020_paper_cylinder_coarse"
    )
    bundle = RUNS_LOCAL / run_id
    gate = bundle / gate_template.name
    coarse = bundle / coarse_template.name
    run_drive = RUNS_DRIVE / run_id
    if bundle.exists() or run_drive.exists():
        raise FileExistsError(f"Run ID already exists: {run_id}")
    bash(
        f"mkdir -p {quoted(bundle)} && cp -a {quoted(gate_template)} "
        f"{quoted(gate)} && cp -a {quoted(coarse_template)} {quoted(coarse)}"
    )
    bash(
        f"mkdir -p {quoted(run_drive)} && rsync -a {quoted(bundle)}/ "
        f"{quoted(run_drive)}/case-input/"
    )
    environment = gate / "environment.txt"
    bash(
        openfoam_bash
        + f" && {{ foamVersion; git -C {quoted(OF_LOCAL)} rev-parse HEAD; "
        + f"uname -a; }} > {quoted(environment)}"
    )
    bash(f"cp {quoted(environment)} {quoted(run_drive / 'environment.txt')}")

    gate_run = bash(
        openfoam_bash + f" && cd {quoted(gate)} && bash ./Allrun",
        check=False,
    )
    bash(
        f"rsync -a {quoted(bundle)}/ "
        f"{quoted(run_drive)}/case-after-gate/"
    )
    output = gate / "postProcessing" / "plateletInitialConditionGate"
    result_path = output / "result.json"
    if not result_path.is_file():
        raise RuntimeError(
            f"Gate did not create a parent result; inspect "
            f"{run_drive}/case-after-gate/"
        )
    result = json.loads(result_path.read_text())
    artifact_drive = (
        run_drive / "artifacts" / "platelet_initial_condition_gate_v1"
    )
    bash(
        f"mkdir -p {quoted(artifact_drive)} && rsync -a {quoted(output)}/ "
        f"{quoted(artifact_drive)}/"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if gate_run.returncode != 0 or result.get("collection_status") != "PASS":
        raise RuntimeError(
            "Platelet initial-condition collection failed; evidence is archived."
        )
    print(
        "PASS: initial-condition records are complete. Hypothesis classification: "
        f"{result.get('hypothesis_classification')}. Evidence is archived under "
        f"{artifact_drive}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
