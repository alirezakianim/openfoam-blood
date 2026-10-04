#!/usr/bin/env python3
"""Archive auditable OpenFOAM evidence without truncating load-bearing data."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import tarfile
from pathlib import Path



# Cap on how many written times the per-child evidence archive samples.
# The complete series is always in case-after-run-<case>.tar.gz; this is the
# auditable SAMPLE. 0 disables the cap. See run 20260906-210336.
MAX_ARCHIVED_TIMES = 60

def numeric_times(case: Path) -> list[tuple[float, Path]]:
    times = []
    for path in case.iterdir():
        if not path.is_dir():
            continue
        try:
            time = float(path.name)
        except ValueError:
            continue
        times.append((time, path))
    return sorted(times)


def read_vectors(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text(errors="replace")
    match = re.search(
        r"(?ms)internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;",
        text,
    )
    if match:
        values = [
            tuple(float(value) for value in row.split())
            for row in re.findall(r"\(([^()]+)\)", match.group(2))
        ]
        if len(values) != int(match.group(1)) or any(len(value) != 3 for value in values):
            raise ValueError(f"{path}: malformed nonuniform vector field")
        return values
    match = re.search(r"internalField\s+uniform\s+\(([^()]+)\)\s*;", text)
    if match:
        value = tuple(float(component) for component in match.group(1).split())
        if len(value) != 3:
            raise ValueError(f"{path}: malformed uniform vector field")
        return [value]
    raise ValueError(f"{path}: cannot parse vector internalField")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_child(
    case: Path,
    destination: Path,
    tail_start: float,
    fields: tuple[str, ...],
    planar_vectors: bool,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    child = case.name
    times = numeric_times(case)
    positive = [(time, path) for time, path in times if time > 0 and (path / "Us").is_file()]

    # Archive the complete solver log even when a child crashes before its
    # first write. A failure is load-bearing evidence, not a reason for the
    # archiver itself to fail and hide the log.
    log = case / "log.phaseClotAxisymmetricFluid"
    compressed_log = destination / f"{child}__log.phaseClotAxisymmetricFluid.gz"
    if log.is_file():
        with log.open("rb") as source, gzip.open(compressed_log, "wb", compresslevel=6) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    else:
        with gzip.open(compressed_log, "wb", compresslevel=6) as target:
            target.write(b"")

    tail_path = destination / f"{child}__tail_max_Us.csv"
    rows = []
    for time, folder in positive:
        if time < tail_start - 1e-14:
            continue
        values = read_vectors(folder / "Us")
        magnitude = max(
            math.hypot(value[0], value[1])
            if planar_vectors
            else math.sqrt(sum(component * component for component in value))
            for value in values
        )
        rows.append((time, magnitude))
    with tail_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "max_Us_m"))
        writer.writerows(rows)

    final_time = positive[-1][0] if positive else None
    final_path = destination / f"{child}__final_Us"
    if positive:
        shutil.copy2(positive[-1][1] / "Us", final_path)

    # SUBSAMPLED IN TIME, and the manifest says which times were taken.
    #
    # This archive is the THIRD copy of the same field data -- the unpacked
    # tree and case-after-run-<case>.tar.gz hold the other two -- and it used
    # to take every written time. Run 20260906-210336 returned 38.89 GB once a
    # child finally survived to endTime and wrote 1200 of them. The complete
    # series is still in the case tarball; what this archive is FOR is a
    # self-describing, sha256-recorded sample that can be audited without
    # unpacking 9 GB, and a sample does not need every frame.
    #
    # The first and last written times are always included -- the last is the
    # state at death or completion, which is the one most often read.
    archived_times = times
    if MAX_ARCHIVED_TIMES and len(times) > MAX_ARCHIVED_TIMES:
        step = (len(times) - 1) / (MAX_ARCHIVED_TIMES - 1)
        picks = sorted({0, len(times) - 1}
                       | {int(round(i * step)) for i in range(MAX_ARCHIVED_TIMES)})
        archived_times = [times[i] for i in picks if 0 <= i < len(times)]

    raw_archive = destination / f"{child}__written_fields.tar.gz"
    included = []
    with tarfile.open(raw_archive, "w:gz", compresslevel=6) as archive:
        for _time, folder in archived_times:
            for field in fields:
                path = folder / field
                if path.is_file():
                    archive.add(path, arcname=f"{folder.name}/{field}", recursive=False)
                    included.append(f"{folder.name}/{field}")

    return {
        "child_id": child,
        "tail_start_s": tail_start,
        "tail_rows": len(rows),
        "final_time_s": final_time,
        "planar_vector_magnitude": planar_vectors,
        "written_times_available": len(times),
        "written_times_archived": len(archived_times),
        "written_times_subsampled": len(archived_times) < len(times),
        "complete_log_gzip": compressed_log.name,
        "complete_log_sha256": sha256(compressed_log),
        "tail_series": tail_path.name,
        "tail_series_sha256": sha256(tail_path),
        "field_evidence_complete": bool(positive),
        "final_Us": final_path.name if positive else None,
        "final_Us_sha256": sha256(final_path) if positive else None,
        "written_fields_archive": raw_archive.name,
        "written_fields_sha256": sha256(raw_archive),
        "written_field_members": len(included),
        "requested_fields": list(fields),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--children", nargs="+", required=True)
    parser.add_argument("--tail-start", type=float, required=True)
    parser.add_argument("--fields", default="Us,Dg,H,p,U")
    parser.add_argument("--planar-vectors", action="store_true")
    args = parser.parse_args()

    gate = args.gate.resolve()
    artifact = args.artifact.resolve()
    evidence = artifact / "auditable_evidence"
    fields = tuple(field.strip() for field in args.fields.split(",") if field.strip())
    manifests = [
        archive_child(
            gate / "runs" / child,
            evidence,
            args.tail_start,
            fields,
            args.planar_vectors,
        )
        for child in args.children
    ]
    manifest = {
        "contract": "complete compressed solver log, tail max|Us| series, final raw Us, and the requested written fields at up to MAX_ARCHIVED_TIMES written times (first and last always included; written_times_archived vs written_times_available records the sampling). The COMPLETE series is in case-after-run-<case>.tar.gz.",
        "children": manifests,
    }
    path = evidence / "evidence_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[archive] auditable evidence manifest: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"archive_openfoam_evidence.py: {error}")
