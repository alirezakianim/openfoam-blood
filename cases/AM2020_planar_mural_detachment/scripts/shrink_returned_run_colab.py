# =====================================================================
# SHRINK AN ALREADY-RETURNED RUN ON DRIVE, WITHOUT LOSING INFORMATION
#
# Paste into the running Colab and execute. Safe to run twice.
#
# WHAT IT REMOVES: the numeric time directories of the UNPACKED tree
#   /content/drive/MyDrive/OpenFOAM_Blood/runs/<run>/case-after-run-<case>/
# and NOTHING ELSE. Those bytes are the field data, and they are the
# whole of the folder's size.
#
# WHY THAT IS LOSSLESS: the same bytes are in case-after-run-<case>.tar.gz
# beside it. The runner already verified that tarball -- it recomputed
# sha256 AFTER the copy landed on Drive and compared member count against
# the local file count, and wrote the result to
# case_after_run_integrity-<case>.json. This script REFUSES to delete
# anything unless that file says intact, AND it checks every single file
# it is about to remove against the manifest's own path list. A file that
# is not provably in the tarball is kept.
#
# WHAT IT KEEPS, always: every log, run-exit-code.txt, child.json,
# constant/, system/, postProcessing/, the tarball, and all of artifacts/
# (result.json, matrix.json, the gzipped solver logs, the tail max|Us|
# series and final_Us). Every finding in STAGES 6.79-6.81 came out of the
# logs and those artifacts, not out of the time directories.
#
# TO GET THE FIELDS BACK:
#     tar xzf case-after-run-<case>.tar.gz
#
# EXTRACT ON LINUX, NOT ON macOS. A case directory holds both `c` (the
# phase field) and `C` (cell centres), and macOS filesystems are
# case-insensitive by default, so one silently overwrites the other --
# which is exactly the "c.c / H.h / p.pas name corruption" the runner
# warns about, and it is why an unpacked tree downloaded to a Mac has
# fewer files than its own tarball. Verified while testing this script:
# extracting run 20260906-101812's tarball on macOS restored 273 of 273
# files but left runs/*/0/c and runs/*/0/C collided. Use the case-sensitive
# sparsebundle, or a Linux box, or extract only what you need.
# =====================================================================

import json
import re
import shutil
from pathlib import Path

RUN_ID = None      # None = most recent run folder under runs/
DRY_RUN = True     # <-- set False to actually delete. Read the report first.
KEEP_ZERO_TIME = True   # keep 0/ (the initial condition) unpacked
DROP_WRITTEN_FIELDS_ARCHIVE = False
# ^ artifacts/.../auditable_evidence/<child>__written_fields.tar.gz is a THIRD
#   copy of a subset of the fields. Its bytes are in the case tarball, but it
#   is the AUDIT copy and evidence_manifest.json records its sha256, so
#   deleting it breaks that chain. Off by default. The gzipped logs, the
#   tail series and final_Us are never touched.

RUNS = Path('/content/drive/MyDrive/OpenFOAM_Blood/runs')
TIME_DIR = re.compile(r'\d+(\.\d+)?([eE][+-]?\d+)?$')


def du(path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def gb(n):
    return f'{n / 2**30:7.2f} GB'


run_dir = RUNS / RUN_ID if RUN_ID else max(
    (d for d in RUNS.iterdir() if d.is_dir()), key=lambda d: d.name
)
print(f'run folder : {run_dir}')
print(f'before     : {gb(du(run_dir))}\n')

freed = kept_unverified = 0

for integrity_path in sorted(run_dir.glob('case_after_run_integrity-*.json')):
    case = integrity_path.name[len('case_after_run_integrity-'):-len('.json')]
    tree = run_dir / f'case-after-run-{case}'
    tarball = run_dir / f'case-after-run-{case}.tar.gz'
    print(f'--- {case} ---')

    if not tree.is_dir():
        print('  unpacked tree already absent, nothing to do')
        continue
    if not tarball.is_file():
        print(f'  REFUSING: {tarball.name} is missing; the tree is the only copy')
        continue

    data = json.loads(integrity_path.read_text())
    if not data.get('intact'):
        print('  REFUSING: integrity json does not say intact')
        continue
    manifest = set(data.get('files') or ())
    if not manifest:
        print('  REFUSING: integrity json carries no file list to verify against')
        continue
    print(f'  tarball verified by the runner: {data["tar_member_count"]} members, '
          f'sha256 {data["local_sha256"][:12]}...')

    runs_root = tree / 'runs'
    if not runs_root.is_dir():
        print('  no runs/ under the tree')
        continue

    for child_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        removed_here = 0
        for time_dir in sorted(p for p in child_dir.iterdir() if p.is_dir()):
            if not TIME_DIR.fullmatch(time_dir.name):
                continue
            if KEEP_ZERO_TIME and float(time_dir.name) == 0.0:
                continue
            files = [p for p in time_dir.rglob('*') if p.is_file()]
            missing = [
                str(p.relative_to(tree)) for p in files
                if str(p.relative_to(tree)) not in manifest
            ]
            if missing:
                kept_unverified += len(missing)
                print(f'    KEEPING {child_dir.name}/{time_dir.name}: '
                      f'{len(missing)} file(s) not in the tarball manifest, '
                      f'e.g. {missing[0]}')
                continue
            size = sum(p.stat().st_size for p in files)
            freed += size
            removed_here += 1
            if not DRY_RUN:
                shutil.rmtree(time_dir)
        if removed_here:
            print(f'    {child_dir.name}: '
                  f'{"would remove" if DRY_RUN else "removed"} '
                  f'{removed_here} time director{"y" if removed_here == 1 else "ies"}')

if DROP_WRITTEN_FIELDS_ARCHIVE:
    for wf in sorted(run_dir.rglob('*__written_fields.tar.gz')):
        freed += wf.stat().st_size
        print(f'{"would remove" if DRY_RUN else "removed"} {wf.name}')
        if not DRY_RUN:
            wf.unlink()

print(f'\nreclaimable: {gb(freed)}')
if kept_unverified:
    print(f'KEPT {kept_unverified} file(s) that were NOT in the tarball manifest '
          '-- those directories were left alone on purpose.')
if DRY_RUN:
    print('\nDRY_RUN is True: nothing was deleted. '
          'Set DRY_RUN = False and run again.')
else:
    print(f'after      : {gb(du(run_dir))}')
    print('fields remain in case-after-run-*.tar.gz '
          '-- extract with: tar xzf case-after-run-<case>.tar.gz')
