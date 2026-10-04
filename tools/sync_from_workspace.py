#!/usr/bin/env python3
"""Re-copy the curated file set from the research workspace, then re-run the
clean-room test.

The repository is a curated subset of a larger private research workspace.
`tools/manifest.txt` lists every file that belongs here, by its workspace-
relative path; relative paths are kept because the generators, the checks and
the Colab runner locate each other by them.

    python3 tools/sync_from_workspace.py --workspace /path/to/workspace
    python3 tools/sync_from_workspace.py --check-only          # test, no copy
    python3 tools/sync_from_workspace.py --check-only --slow   # + two slow checks
    python3 tools/sync_from_workspace.py --workspace W --dry-run
                                          # also run the main case's dry_run.py
                                          # (hours: it synthesises 219 children
                                          # per scenario)

The clean-room test copies ONLY the manifest files from this repository into an
empty temporary directory and runs everything from there. A file the generators
or checks need but the manifest lacks therefore fails here, not on Colab.

One manifest entry is derived, not copied byte for byte:
`cases/ARCHIVED_CASES.json`. The workspace registry names 38 retired template
directories that are not in this repository; the registry check requires its
names to match the library exactly, so the copy keeps only the entries whose
files exist here.

Exit status: 0 only if every step passed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tools" / "manifest.txt"
MAIN_CASE = "cases/AM2020_planar_mural_detachment"
ARCHIVE = "cases/ARCHIVED_CASES.json"

# (label, script, args, extra environment). Run from the clean-room root, in
# this order: the first step writes the 219 generated children that
# check_case_boundaries then reads.
STEPS = [
    ("generate every child", f"{MAIN_CASE}/scripts/create_cases.py", [], {}),
    ("runner staging", "scripts/check_runner_staging.py", ["AM2020_planar_mural_detachment"], {}),
    ("fvSchemes coverage self-test", "scripts/check_fvschemes_coverage.py", ["--self-test"], {}),
    ("fvSchemes coverage", "scripts/check_fvschemes_coverage.py", [MAIN_CASE], {}),
    ("refused dictionary values self-test", "scripts/check_refused_dictionary_values.py", ["--self-test"], {}),
    ("refused dictionary values", "scripts/check_refused_dictionary_values.py", [], {}),
    ("case boundaries", "scripts/check_case_boundaries.py", [MAIN_CASE], {}),
    ("throat mesh", "scripts/check_throat_mesh.py", [], {}),
    ("notebook batch selector", "scripts/check_notebook_batch_selector.py", [], {}),
    ("block-coupled step (S19)", "scripts/check_s19_block_coupled.py", [], {"SUITESPARSE_ROOT": "@suitesparse"}),
]
# Unit checks: each is one script, exit 0 = pass. (script, args)
UNIT_CHECKS = [
    ("check_b_transport_algebra", ["--self-test"]),
    ("check_b_transport_algebra", []),
    ("check_neohookean_algebra", ["--self-test"]),
    ("check_neohookean_algebra", []),
    ("check_content_record_parser", []),
    ("check_dictionary_read_forms", []),
    ("check_field_dimensions", []),
    ("check_indicator_recentred_in_mag", []),
    ("check_network_pressure_law", []),
    ("check_per_step_state_is_primed", []),
    ("check_phase_indicator_basis", []),
    ("check_phase_normal_basis", []),
    ("check_phase_weight_basis_split", []),
    ("check_platelet_bind_localisation", []),
    ("check_platelet_diagnostic_semantics", []),
    ("check_platelet_fluid_carrier", []),
    ("check_registry_lookups_resolve", []),
    ("check_runner_controls_keys", []),
    ("check_s12_intrinsic_drag", []),
    ("check_s14_network_rupture", []),
    ("check_s15_composition_label_and_damage", []),
    ("check_scheme_coverage", []),
    ("check_stage2_case_generation", []),
    ("check_stage2_content_seeding", []),
    ("check_stage3_gate_reaches_children", []),
    ("check_zero_diffusivity_safety", []),
    ("s16_offline_checks", ["self-test"]),
    ("s17_offline_checks", ["self-test"]),
]
# Slow checks (each recompiles and mutates the solver text many times and
# regenerates children; tens of minutes on a loaded machine). Run with --slow.
# check_s18_soft and check_s18p_probe were removed (2026-10-04): they import the
# research workspace's design-document checkers (check_s17b_slack_fibre.py etc.),
# which are internal process, not code. The FP-trap kernel test they wrapped runs
# on its own in the fast tier (s18_fpe_kernel_test).
SLOW_CHECKS = [
]
# Header-only kernels compiled against the solver's own text: (test source,
# extra flags). The FP-trap test must ALSO fail when run with `unfixed`.
KERNEL_TESTS = [
    ("s16_partition_kernel_test", ["-O2"]),
    ("s17_fibre_kernel_test", ["-O2"]),
    ("s18_fpe_kernel_test", ["-O3"]),
]
SOLVER_DIR = "user-applications/phaseClotAxisymmetricFluid"


def read_manifest() -> list[str]:
    out = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    if len(out) != len(set(out)):
        raise SystemExit("manifest has duplicate entries")
    return out


def filtered_archive(source: Path, root: Path) -> str:
    registry = json.loads(source.read_text())
    registry["templates"] = {
        k: v for k, v in registry["templates"].items() if (root / k).is_file()
    }
    registry["_curated_repo_note"] = (
        "Filtered by tools/sync_from_workspace.py: only the entries that "
        "exist in this repository are kept; the full registry lives in the "
        "research workspace."
    )
    return json.dumps(registry, indent=1) + "\n"


def copy_from_workspace(workspace: Path, files: list[str], prune: bool) -> None:
    for rel in files:
        src, dst = workspace / rel, REPO / rel
        if not src.is_file():
            raise SystemExit(f"manifest names a file the workspace lacks: {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel == ARCHIVE:
            dst.write_text(filtered_archive(src, REPO))
        else:
            shutil.copy2(src, dst)
    managed = ("cases", "user-applications", "scripts", "notebooks")
    keep = set(files)
    orphans = []
    for top in managed:
        for p in sorted((REPO / top).rglob("*")) if (REPO / top).is_dir() else []:
            rel = str(p.relative_to(REPO))
            if p.is_file() and rel not in keep and "__pycache__" not in p.parts:
                orphans.append(rel)
    for rel in orphans:
        if prune:
            (REPO / rel).unlink()
        print(f"{'pruned' if prune else 'ORPHAN (not in manifest)'}: {rel}")
    print(f"copied {len(files)} files from {workspace}")


def run(label: str, root: Path, script: str, args: list[str], extra: dict, results: list) -> None:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for k, v in extra.items():
        env[k] = v
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-B", script, *args], cwd=root,
                          capture_output=True, text=True, env=env)
    tail = [ln for ln in (proc.stdout + proc.stderr).strip().splitlines() if ln.strip()][-1:]
    results.append((label, proc.returncode, time.time() - t0, tail[0][:100] if tail else ""))
    print(f"{'PASS' if proc.returncode == 0 else 'FAIL'} exit={proc.returncode:<3} "
          f"{time.time() - t0:6.1f}s  {label}", flush=True)
    if proc.returncode != 0:
        print((proc.stdout + proc.stderr)[-1500:])


def kernel_test(root: Path, name: str, opt: list[str], results: list) -> None:
    cxx = shutil.which("c++")
    if not cxx:
        results.append((f"kernel test {name}", 127, 0.0, "no C++ compiler"))
        print(f"FAIL kernel test {name}: no C++ compiler")
        return
    flags = ["-std=c++14", *opt, "-ffp-contract=off"]
    if name == "s18_fpe_kernel_test" and "clang" in subprocess.run(
            [cxx, "--version"], capture_output=True, text=True).stdout:
        flags.append("-ffp-exception-behavior=maytrap")
    exe = root / f"{name}.bin"
    build = subprocess.run([cxx, *flags, "-I", SOLVER_DIR, f"scripts/{name}.cpp", "-o", str(exe)],
                           cwd=root, capture_output=True, text=True)
    if build.returncode != 0:
        results.append((f"kernel test {name}", build.returncode, 0.0, "compile failed"))
        print(f"FAIL kernel test {name}: compile\n{build.stderr[-1200:]}")
        return
    rc = subprocess.run([str(exe)], cwd=root, capture_output=True, text=True).returncode
    if name == "s18_fpe_kernel_test" and rc == 0:
        # The negative: the unfixed expression must trap, else the pass is vacuous.
        # A fatal FP trap makes macOS hand the process to its crash reporter,
        # which can stall the exit for minutes under load. A stall is NOT a
        # trap seen: it is reported as inconclusive (exit 2), never as a pass.
        neg = subprocess.Popen([str(exe), "unfixed"], cwd=root,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            neg.wait(timeout=240)
            rc = 0 if neg.returncode != 0 else 1
        except subprocess.TimeoutExpired:
            rc = 2
            print("  negative 'unfixed' did not exit within 240 s (crash-reporter "
                  "stall?): inconclusive. Run on Linux, or disable crash reporting.")
    results.append((f"kernel test {name}", rc, 0.0, ""))
    print(f"{'PASS' if rc == 0 else 'FAIL'} exit={rc:<3} kernel test {name}", flush=True)


def clean_room(files: list[str], with_dry_run: bool, slow: bool) -> int:
    results: list = []
    with tempfile.TemporaryDirectory(prefix="ofb_cleanroom_") as tmp:
        root = Path(tmp)
        for rel in files:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, dst)
        suitesparse = os.environ.get("SUITESPARSE_ROOT") or str(
            Path.home() / ".cache" / "openfoam_blood" / "suitesparse")
        for label, script, args, extra in STEPS:
            extra = {k: (suitesparse if v == "@suitesparse" else v) for k, v in extra.items()}
            run(label, root, script, args, extra, results)
        for name, args in UNIT_CHECKS:
            run(f"{name} {' '.join(args)}".strip(), root, f"scripts/{name}.py", args, {}, results)
        for name, args in (SLOW_CHECKS if slow else []):
            run(f"{name} (slow)", root, f"scripts/{name}.py", args, {}, results)
        for name, opt in KERNEL_TESTS:
            kernel_test(root, name, opt, results)
        if with_dry_run:
            run("main-case dry_run.py (all scenarios)", root,
                f"{MAIN_CASE}/scripts/dry_run.py", [], {}, results)
    failed = [r for r in results if r[1] != 0]
    print(f"\nclean room: {len(results) - len(failed)}/{len(results)} steps passed")
    for r in failed:
        print(f"  FAILED: {r[0]} (exit {r[1]})")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workspace", type=Path, default=os.environ.get("OFB_WORKSPACE"),
                    help="research workspace root (or set OFB_WORKSPACE)")
    ap.add_argument("--check-only", action="store_true", help="skip the copy")
    ap.add_argument("--prune", action="store_true", help="delete repo files not in the manifest")
    ap.add_argument("--slow", action="store_true", help="also run the slow mutation checks")
    ap.add_argument("--dry-run", action="store_true", help="also run the main case's dry_run.py (hours)")
    a = ap.parse_args()
    files = read_manifest()
    if not a.check_only:
        if not a.workspace:
            raise SystemExit("--workspace (or OFB_WORKSPACE) is required unless --check-only")
        copy_from_workspace(Path(a.workspace), files, a.prune)
    return clean_room(files, a.dry_run, a.slow)


if __name__ == "__main__":
    sys.exit(main())
