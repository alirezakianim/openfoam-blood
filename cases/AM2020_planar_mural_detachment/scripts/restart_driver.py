#!/usr/bin/env python3
"""S25 section 2C: the restart driver. Restart a child from its last FULL write, never from latestTime.

WHY (numerics 01 finding 6, S25 design section 2C). A child's time directories are of two kinds:
  * a FULL write (writeControl runTime): every AUTO_WRITE field, including w, and uniform/time;
  * a FRAME (the function-object frames the long children write between full writes): a handful of
    fields (D75: Gfield U Us Vs clotDrag p poroPerm q) and NO w. w is the solver's MUST_READ field
    (phaseClotAxisymmetricFluid.C L201), so `startFrom latestTime` after an unclean stop that left a
    frame as the newest directory dies at construction. A frame is NOT identified by a missing
    uniform/time: D75's 0.0002/ frame carries uniform/time (preflight/20261009-171539, tar listing).
    The discriminating member is w.
A session killed mid-write can also leave a FULL-looking directory whose last field is cut off, so a
directory is accepted only when it is INTACT: every file non-empty and ending in OpenFOAM's footer line.

WHAT IT DOES. `restart_driver.py CASE [--source DIR] [--apply]`
  * lists the numeric time directories of DIR (default CASE), classifies each (full / frame-or-partial /
    damaged), and picks the LATEST one that is full and intact. None -> exit 2, naming every directory and
    why it was refused. There is no fallback to latestTime, to a frame, or to a cold start.
  * --apply sets `startFrom startTime; startTime <that directory name>;` in CASE/system/controlDict (the
    generator writes startFrom startTime, and the driver never writes latestTime), copies the directory into
    CASE when DIR is another tree, or moves every NEWER directory of CASE aside to restart_discarded/ (never
    deletes), and writes CASE/restart.json (the choice, every skipped directory with its reason, and the
    sha256 of every file of the chosen directory).
  * when DIR also carries the previous session's constant/ and system/, the regenerated dictionaries must
    match them (controlDict up to its startFrom/startTime lines): a restart under another package's physics
    is a different experiment and is refused (exit 3).

  * `--index N` (S25 B76-RT, RT-1): instead of the LATEST full write, the full intact write whose uniform/time index is N.
    RT-1's A writes a full directory at every step, so "the latest" is step N0+5 and the restart must name N0. The index is
    read from the directory's own uniform/time, not from its name. None -> exit 2, naming every directory and its index.
  * `--from-child-json child.json --runs-dir RUNS` (RT-1's B): the source is the SIBLING child runs/<restart_from_child> of
    the same session and the index is child.json's restart_index. The sibling must have finished (run-exit-code.txt is 0);
    otherwise exit 4 and the child is not solved. restart.json records the sibling, the index and the chosen directory.

Standard library only (it runs under Allrun's plain python3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

# The members a full write must hold. w: MUST_READ in phaseClotAxisymmetricFluid.C L201. U and p: MUST_READ
# in OpenFOAM-13 incompressibleFluid.C L60-92. uniform/time: the index, time and deltaT the restart resumes
# from (Time.C L214-262). A frame has U and p and (D75) uniform/time, but no w. phiSkel (S25.1, OI-77): the constructor reads it on
# a restart to rebuild the stored old time of the phase weights, and REFUSES a restart whose directory lacks it (docket
# 20261013-1405, 04); a write that lacks it (a build before S25) is therefore not restartable and is refused HERE, naming it.
FULL_WRITE_REQUIRED = ("w", "U", "p", "phiSkel", "uniform/time")
FOOTER = re.compile(rb"^// \*+ //\s*$")
# The dictionaries that must be identical between the session that wrote the state and the one resuming it.
DICTIONARIES = ("constant/phaseClotProperties", "constant/physicalProperties", "constant/momentumTransport",
                "system/fvSchemes", "system/fvSolution", "system/controlDict", "system/blockMeshDict")
START_LINES = re.compile(r"(?m)^\s*(startFrom|startTime)\s+[^;]+;[ \t]*\n")


class RestartError(Exception):
    """Carries the exit code: 2 = nothing restartable, 3 = the dictionaries differ, 4 = bad input."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def time_dirs(root: Path) -> list[Path]:
    """Numeric-named directories of root, ascending by value. 0 is the cold start and is not a restart."""
    out = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        try:
            value = float(d.name)
        except ValueError:
            continue
        if value > 0:
            out.append((value, d))
    return [d for _v, d in sorted(out)]


def _read_uniform_time(path: Path) -> dict:
    text = re.sub(r"//[^\n]*", "", path.read_text(errors="replace"))
    found = {}
    for key in ("value", "name", "index", "deltaT"):
        m = re.search(rf'(?m)^\s*{key}\s+"?([^;"\s]+)"?\s*;', text)
        if m:
            found[key] = m.group(1)
    return found


def classify(d: Path) -> tuple[str, str]:
    """('full' | 'frame' | 'damaged', the reason). 'frame' covers every directory without the full
    write's members; 'damaged' is a directory that has them but is cut off or inconsistent."""
    missing = [m for m in FULL_WRITE_REQUIRED if not (d / m).is_file()]
    if missing:
        have = sorted(p.name for p in d.iterdir() if p.is_file())
        return "frame", f"no {', '.join(missing)} (holds {len(have)} fields: {' '.join(have[:10])}{' ...' if len(have) > 10 else ''})"
    bad = []
    for f in sorted(p for p in d.rglob("*") if p.is_file()):
        size = f.stat().st_size
        if size == 0:
            bad.append(f"{f.relative_to(d)} is empty")
            continue
        with open(f, "rb") as fh:
            fh.seek(max(0, size - 256))
            tail = [ln for ln in fh.read().splitlines() if ln.strip()]
        if not tail or not FOOTER.match(tail[-1]):
            bad.append(f"{f.relative_to(d)} has no closing footer line (cut off)")
    if bad:
        return "damaged", "; ".join(bad[:4]) + (f"; and {len(bad) - 4} more" if len(bad) > 4 else "")
    ut = _read_uniform_time(d / "uniform" / "time")
    try:
        value, index = float(ut["value"]), int(ut["index"])
    except (KeyError, ValueError):
        return "damaged", f"uniform/time has no readable value and index (found {ut})"
    if abs(value - float(d.name)) > 1e-9 * max(abs(value), 1e-300) or index < 1:
        return "damaged", f"uniform/time says value {ut['value']} index {ut['index']} for directory {d.name}"
    return "full", f"{len(list(d.rglob('*')))} entries, index {index}"


def choose(root: Path, index: int | None = None) -> tuple[Path, list[dict]]:
    """The latest FULL, INTACT directory of root (or, with `index`, the full intact one whose uniform/time index is `index`),
    and the verdict on every directory (ascending). With an index there is no fallback to another step."""
    verdicts, chosen = [], None
    for d in time_dirs(root):
        kind, why = classify(d)
        step = None
        if kind == "full":
            step = int(_read_uniform_time(d / "uniform" / "time")["index"])
        verdicts.append({"name": d.name, "kind": kind, "why": why, "index": step})
        if kind == "full" and (index is None or step == index):
            chosen = d
    if chosen is None:
        lines = "\n".join(f"  {v['name']}: {v['kind']} (index {v['index']}) -- {v['why']}" for v in verdicts) or "  (no time directory > 0)"
        want = "no full intact write to restart from" if index is None else f"no full intact write with uniform/time index {index}"
        raise RestartError(2, f"{root}: {want} (a full write holds {', '.join(FULL_WRITE_REQUIRED)}); there is no "
                              f"fallback to latestTime, a frame or another step:\n{lines}")
    return chosen, verdicts


def _normalised_controldict(text: str) -> str:
    return START_LINES.sub("", text)


def compare_dictionaries(case: Path, source: Path) -> list[str]:
    """The dictionaries in both trees that differ. Only those the source holds are compared."""
    differing = []
    for rel in DICTIONARIES:
        a, b = case / rel, source / rel
        if not b.is_file():
            continue
        if not a.is_file():
            differing.append(f"{rel} (absent from the regenerated case)")
            continue
        ta, tb = a.read_text(), b.read_text()
        if rel == "system/controlDict":
            ta, tb = _normalised_controldict(ta), _normalised_controldict(tb)
        if ta != tb:
            differing.append(rel)
    return differing


def _set_entry(path: Path, key: str, value: str) -> None:
    text = path.read_text()
    updated, count = re.subn(rf"(?m)^(\s*{re.escape(key)}\s+)[^;]+;", lambda m: f"{m.group(1)}{value};", text, count=1)
    if count != 1 or len(re.findall(rf"(?m)^\s*{re.escape(key)}\s", text)) != 1:
        raise RestartError(4, f"{path}: expected exactly one {key} entry")
    path.write_text(updated)


def sha256_tree(d: Path) -> dict:
    out = {}
    for f in sorted(p for p in d.rglob("*") if p.is_file()):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out[str(f.relative_to(d))] = h.hexdigest()
    return out


def restart(case: Path, source: Path | None, apply: bool, accept_differences: bool = False,
            index: int | None = None, sibling: str | None = None) -> dict:
    case = case.resolve()
    src = (source or case).resolve()
    if not src.is_dir():
        raise RestartError(4, f"restart source {src} is not a directory")
    control = case / "system" / "controlDict"
    if not control.is_file():
        raise RestartError(4, f"{control} is absent")
    ctl = control.read_text()
    if re.search(r"(?m)^\s*startFrom\s+latestTime\s*;", ctl):
        raise RestartError(4, f"{control} says startFrom latestTime: the restart names its time explicitly")
    chosen, verdicts = choose(src, index)
    end = re.search(r"(?m)^\s*endTime\s+([^;\s]+)\s*;", ctl)
    if end is None:
        raise RestartError(4, f"{control} has no endTime")
    if not float(chosen.name) < float(end.group(1)):
        raise RestartError(4, f"the last full write {chosen.name} is not before endTime {end.group(1)}: nothing to run")
    if src != case:
        differing = compare_dictionaries(case, src)
        if differing and not accept_differences:
            raise RestartError(3, f"the regenerated case differs from the session that wrote {chosen.name} in "
                                  f"{', '.join(differing)}; a restart under other dictionaries is another experiment")
    record = {"chosen": chosen.name, "chosen_index": int(_read_uniform_time(chosen / "uniform" / "time")["index"]),
              "requested_index": index, "sibling": sibling, "source": str(src), "in_place": src == case,
              "end_time": end.group(1), "directories": verdicts, "applied": apply,
              "dictionaries_differ_and_were_accepted": differing if src != case and accept_differences else []}
    if not apply:
        return record
    moved = []
    if src == case:
        aside = case / "restart_discarded"
        for d in time_dirs(case):
            if float(d.name) > float(chosen.name):
                aside.mkdir(exist_ok=True)
                target = aside / d.name
                if target.exists():
                    raise RestartError(4, f"{target} already exists; refusing to overwrite discarded evidence")
                shutil.move(str(d), str(target))
                moved.append(d.name)
    else:
        target = case / chosen.name
        if target.exists():
            raise RestartError(4, f"{target} already exists in the regenerated case; refusing to overwrite")
        shutil.copytree(chosen, target)
    _set_entry(control, "startFrom", "startTime")
    _set_entry(control, "startTime", chosen.name)
    record["moved_aside"] = moved
    record["sha256"] = sha256_tree(case / chosen.name)
    (case / "restart.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def sibling_source(case: Path, child_json: Path, runs_dir: Path) -> tuple[Path, int, str]:
    """RT-1's B: (the sibling case directory, restart_index, the sibling's id) from child.json, after checking the sibling
    exists and exited 0 (run-exit-code.txt), and that this child's stop step is after the restart index."""
    try:
        spec = json.loads(child_json.read_text())
        sib, idx, stop = spec["restart_from_child"], spec["restart_index"], spec["stop_after_step"]
    except (OSError, ValueError, KeyError) as error:
        raise RestartError(4, f"{child_json}: needs restart_from_child, restart_index and stop_after_step ({error!r})") from error
    if isinstance(idx, bool) or not isinstance(idx, int) or isinstance(stop, bool) or not isinstance(stop, int) or not 1 <= idx < stop:
        raise RestartError(4, f"{child_json}: restart_index {idx!r} must be an integer in [1, stop_after_step {stop!r})")
    if not isinstance(sib, str) or "/" in sib or sib in ("", ".", ".."):
        raise RestartError(4, f"{child_json}: restart_from_child {sib!r} is not a child id")
    src = runs_dir / sib
    if not src.is_dir():
        raise RestartError(4, f"the sibling {src} does not exist: {sib} must run before {spec.get('child_id')} in the same session")
    code = src / "run-exit-code.txt"
    if not code.is_file() or code.read_text().strip() != "0":
        raise RestartError(4, f"the sibling {sib} has not finished normally ({code} is "
                              f"{'absent' if not code.is_file() else repr(code.read_text().strip())}, not 0): it must run to its stop first")
    return src, idx, sib


def _positive(name: str) -> bool:
    try:
        return float(name) > 0
    except ValueError:
        return False


def extract_from_tarball(tarball: Path, children: list[str], dest: Path) -> dict:
    """Extract, for each child, runs/<child>/{constant,system} and every time directory that holds a `w`
    (the full writes; frames are not extracted) from a case-after-run-*.tar.gz into dest/<child>/. Two
    passes over the archive (the first reads names only). Members are written one at a time to explicit
    paths; no member may leave dest."""
    pat = re.compile(r"^[^/]+/runs/([^/]+)/(.+)$")
    seen: dict[str, dict[str, set]] = {c: {} for c in children}
    with tarfile.open(tarball, "r:*") as tf:
        for m in tf:
            hit = pat.match(m.name)
            if m.isfile() and hit and hit.group(1) in seen:
                parts = hit.group(2).split("/")
                if parts[-1] in ("w", "time") and (parts[1:] in (["w"], ["uniform", "time"])):
                    seen[hit.group(1)].setdefault(parts[0], set()).add("/".join(parts[1:]))
    # a restartable directory holds BOTH w and uniform/time: 0/ has w only, a frame has uniform/time only
    full: dict[str, set] = {}
    for c, dirs in seen.items():
        full[c] = {d for d, have in dirs.items() if have == {"w", "uniform/time"} and _positive(d)}
    missing = [c for c, s in full.items() if not s]
    if missing:
        raise RestartError(2, f"{tarball} holds no full write (a directory with w) for {', '.join(missing)}")
    wanted = 0
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as tf:
        for m in tf:
            hit = pat.match(m.name)
            if not (m.isfile() and hit and hit.group(1) in full):
                continue
            child, rel = hit.group(1), hit.group(2)
            top = rel.split("/")[0]
            if top not in ("constant", "system") and top not in full[child]:
                continue
            out = (dest / child / rel).resolve()
            if dest.resolve() not in out.parents:
                raise RestartError(4, f"archive member {m.name} would leave {dest}")
            out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            wanted += 1
    return {"tarball": str(tarball), "children": {c: sorted(s, key=float) for c, s in full.items()},
            "members_extracted": wanted}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("restart", help="choose and stage the restart of one case (the default command)")
    r.add_argument("case", type=Path)
    r.add_argument("--source", type=Path, default=None, help="a directory of time directories (default: the case)")
    r.add_argument("--apply", action="store_true", help="edit controlDict and stage the directory (default: report)")
    r.add_argument("--index", type=int, default=None,
                   help="restart from the full write whose uniform/time index is N, not the latest (S25 B76-RT, RT-1)")
    r.add_argument("--from-child-json", type=Path, default=None,
                   help="RT-1's B: read restart_from_child / restart_index from this child.json; the source is RUNS/<that child> (needs --runs-dir)")
    r.add_argument("--runs-dir", type=Path, default=None, help="the directory holding the sibling child's case (with --from-child-json)")
    r.add_argument("--accept-dictionary-differences", action="store_true",
                   help="MANUAL ONLY (Allrun never passes it): restart under dictionaries that differ from the "
                        "writing session's, e.g. a state written by an older package; the files are recorded in restart.json")
    e = sub.add_parser("extract", help="extract the full writes and dictionaries of children from a case-after-run tarball")
    e.add_argument("tarball", type=Path)
    e.add_argument("dest", type=Path)
    e.add_argument("children", nargs="+")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.cmd == "extract":
            print(json.dumps(extract_from_tarball(args.tarball, args.children, args.dest), indent=1))
            return 0
        if args.cmd != "restart":
            ap.error("a command is required: restart or extract")
        index, sibling, source = args.index, None, args.source
        if args.from_child_json is not None:
            if args.runs_dir is None or args.source is not None or args.index is not None:
                raise RestartError(4, "--from-child-json needs --runs-dir and takes neither --source nor --index")
            source, index, sibling = sibling_source(args.case, args.from_child_json, args.runs_dir)
        elif args.runs_dir is not None:
            raise RestartError(4, "--runs-dir is read only with --from-child-json (it would be silently inert)")
        rec = restart(args.case, source, args.apply, args.accept_dictionary_differences, index, sibling)
    except RestartError as error:
        print(f"restart_driver: {error}", file=sys.stderr)
        return error.code
    print(f"restart_driver: {'restarting' if rec['applied'] else 'would restart'} from {rec['chosen']} "
          f"(of {len(rec['directories'])} time directories; end {rec['end_time']})")
    for v in rec["directories"]:
        print(f"  {v['name']:<14} {v['kind']:<8} index {v['index']}  {v['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
