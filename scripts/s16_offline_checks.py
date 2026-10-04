#!/usr/bin/env python3
"""S16 offline checks, run on a RETURNED batch-57 run (not assessor gates).

research/designs/2026-09-24-S16-timed-terminal-rupture.md section 6, as
revised after 13, 13b, 14, 16 and 17 (C+).

  gid        G-id, the pre-damage identity. Every PHASE_CLOT_* line common to
             both builds must match its parent's to the printed digits, from
             step 1 to the step before the PARENT's first damage (em62:
             1.35189 ms, log L300947; em61: 2.600 ms). Excluded, and said so
             in the output: *_CONFIG lines (they carry the S16 keys, which
             differ by construction), PHASE_CLOT_COST (wall-clock seconds),
             and tags only one build prints (the new PHASE_CLOT_S16* lines).
             em65 is not gated (it differs by final relaxation from step 1).
             Under C+ theta never reads the partition and the partition is
             live from step 1, so G-id covers theta, the partition and every
             S16 term at d = 0. Its negative: a build whose partition enters
             theta's shared limiter with the shipped zero convention fails at
             step 1 (lambdaFluxMean differs first; self-test below).

  partition  A WATCH TOOL, NOT A GATE: S16p is withdrawn as a gate (13b: no
             threshold window between the solver tolerance and the smallest
             deposit). At every written time it reports max|thetaIntact +
             thetaPending + thetaBroken - (thetaC + thetaH)|/max theta, the
             accumulated deviation (the solver's partitionSumDevMax at the
             writes). --tol only marks times above it; they are reported, not
             failed. What DOES fail: a written time with thetaC but no
             partition field, a negative component, or a run with no written
             time at all -- never a skip. Reads the returned
             case-after-run-*.tar.gz as a STREAM (never extracted: the macOS
             c/C collision) or a case directory. The deposit mirror has its own
             runtime gate, S16m, in assess_case.py.

  startup    em65's start-up gate for E (UFinal/pFinal 1), design section
             6: outletMeanUx >= 0.9 U_ref = 0.01767 m/s at some record with
             t <= 0.2 ms, AND max outletMeanUx over t <= 0.5 ms <= 1.2 U_ref
             = 0.02355 m/s (U_ref = 0.0196283 m/s), AND no crash (no FOAM
             FATAL in the log; run-exit-code 0 when given; records reach
             0.5 ms). Read from PHASE_CLOT_DIAGNOSTIC.

  onset      em65's damage onset: the first PHASE_CLOT_RUPTURE record with
             newlyRuptured > 0 (under terminalRelease: the first cell whose
             d* rose, i.e. first d* > 0) lies in [0.9, 1.6] ms. No onset at
             all is outside the range.

  self-test  The negative tests of every check (each must FAIL on the defect
             it exists to catch, and PASS on the control).

Usage:
  python3 -B scripts/s16_offline_checks.py gid <child log> <parent log> [--cutoff T]
  python3 -B scripts/s16_offline_checks.py partition <tar.gz | case dir> <child_id> [--tol 1e-12]   (WATCH)
  python3 -B scripts/s16_offline_checks.py startup <em65 log> [--exit-code <run-exit-code.txt>]
  python3 -B scripts/s16_offline_checks.py onset <em65 log>
  python3 -B scripts/s16_offline_checks.py self-test [<a real parent log>]

Exit 0 = PASS, 1 = FAIL, 2 = usage / unreadable input.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import tarfile
import tempfile
from pathlib import Path

TAG = re.compile(r"^(PHASE_CLOT_[A-Z0-9_]+)\b")
TIME = re.compile(r"\btime=(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
PARTITION = ("thetaIntact", "thetaPending", "thetaBroken")
CONTENT = ("thetaC", "thetaH")


# ---------------------------------------------------------------- G-id ----

def _excluded(tag: str) -> str | None:
    if tag.endswith("_CONFIG") or tag == "PHASE_CLOT_CONFIG":
        return "config echo (carries the S16 keys)"
    if tag == "PHASE_CLOT_COST":
        return "wall-clock seconds"
    return None


def first_damage_time(path: Path) -> float | None:
    """The time of the first PHASE_CLOT_RUPTURE record with newlyRuptured > 0
    -- the parent's first damage, which ends the identity window."""
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("PHASE_CLOT_RUPTURE "):
                m = re.search(r"\bnewlyRuptured=(\d+)", line)
                if m and int(m.group(1)) > 0:
                    t = TIME.search(line)
                    return float(t.group(1)) if t else None
    return None


def window_lines(path: Path, cutoff: float) -> dict[str, list[str]]:
    """Every PHASE_CLOT_* line before the first line stamped time >= cutoff,
    grouped by tag in log order. Lines without a time= stamp before that
    point belong to the window (construction-time lines)."""
    out: dict[str, list[str]] = {}
    with open(path, errors="replace") as fh:
        for raw in fh:
            m = TAG.match(raw)
            if not m:
                continue
            t = TIME.search(raw)
            if t and float(t.group(1)) >= cutoff:
                break
            out.setdefault(m.group(1), []).append(raw.rstrip("\n"))
    return out


def gid(child: Path, parent: Path, cutoff: float | None) -> tuple[bool, str]:
    if cutoff is None:
        cutoff = first_damage_time(parent)
        if cutoff is None:
            return False, "FAIL: the parent log has no damage event; no window"
    a = window_lines(child, cutoff)
    b = window_lines(parent, cutoff)
    common = sorted(set(a) & set(b))
    report = [f"window: time < {cutoff!r}"]
    only = sorted(set(a) ^ set(b))
    if only:
        report.append("tags in one build only (not compared): " + ", ".join(only))
    compared = 0
    first_bad = None
    for tag in common:
        why = _excluded(tag)
        if why:
            report.append(f"excluded {tag}: {why}")
            continue
        la, lb = a[tag], b[tag]
        if len(la) != len(lb):
            first_bad = first_bad or (tag, min(len(la), len(lb)),
                                      f"{len(la)} lines in the child, {len(lb)} in the parent")
        for i, (x, y) in enumerate(zip(la, lb)):
            compared += 1
            if x != y and first_bad is None:
                first_bad = (tag, i, f"child:  {x}\nparent: {y}")
    report.append(f"lines compared: {compared} over {len(common)} common tags")
    if compared == 0:
        return False, "\n".join(report + ["FAIL: nothing compared -- an empty window is not identity"])
    if first_bad:
        tag, i, what = first_bad
        return False, "\n".join(report + [f"FAIL: first difference at {tag} record {i}:", what])
    return True, "\n".join(report + ["PASS: identical to the printed digits"])


# ------------------------------------------------------------- S16p -----

def parse_internal_field(text: str) -> list[float] | float:
    m = re.search(r"internalField\s+uniform\s+([^;\s]+)\s*;", text)
    if m:
        return float(m.group(1))
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(", text)
    if not m:
        raise ValueError("no ascii internalField")
    n = int(m.group(1))
    body = text[m.end():]
    vals = body.split(")", 1)[0].split()
    if len(vals) != n:
        raise ValueError(f"internalField declares {n} values, found {len(vals)}")
    return [float(v) for v in vals]


def _expand(v, n):
    return [v]*n if isinstance(v, float) else v


def check_time(fields: dict[str, str], tol: float) -> tuple[bool, float, str]:
    """ok is False only for a STRUCTURAL defect (a missing field, a negative
    component); the deviation against tol is reported, not failed."""
    missing = [f for f in CONTENT + PARTITION if f not in fields]
    if missing:
        return False, float("nan"), "missing " + ", ".join(missing)
    parsed = {k: parse_internal_field(v) for k, v in fields.items()}
    n = max(len(v) for v in parsed.values() if isinstance(v, list)) if any(
        isinstance(v, list) for v in parsed.values()) else 1
    p = {k: _expand(v, n) for k, v in parsed.items()}
    theta = [c + h for c, h in zip(p["thetaC"], p["thetaH"])]
    tmax = max(theta)
    dev = max(abs(i + (q + b) - t) for i, q, b, t in
              zip(p["thetaIntact"], p["thetaPending"], p["thetaBroken"], theta))
    rel = dev/tmax if tmax > 0 else dev
    neg = sum(1 for k in PARTITION for v in p[k] if not v >= 0)
    ok = neg == 0
    return ok, rel, (f"max|sum - theta|/max theta = {rel:.6e}"
                     f"{' (WATCH: above ' + format(tol, 'g') + ')' if rel > tol else ''}"
                     f", negative components {neg}")


def _time_value(name: str) -> float | None:
    try:
        return float(name)
    except ValueError:
        return None


def partition_from_members(items, child: str, tol: float) -> tuple[bool, str]:
    """items: iterable of (relative path parts, text). Groups by written time."""
    by_time: dict[str, dict[str, str]] = {}
    for parts, text in items:
        # .../runs/<child>/<time>/<field>
        if len(parts) < 4 or parts[-3] != child or parts[-4] != "runs":
            continue
        t, f = parts[-2], parts[-1]
        if _time_value(t) is None or _time_value(t) <= 0:
            continue
        if f in CONTENT + PARTITION:
            by_time.setdefault(t, {})[f] = text
    if not by_time:
        return False, f"FAIL: no written time > 0 for {child} -- nothing checked is not a pass"
    lines, ok_all, worst, above = [], True, 0.0, 0
    for t in sorted(by_time, key=float):
        ok, rel, msg = check_time(by_time[t], tol)
        ok_all &= ok
        if rel == rel:
            worst = max(worst, rel)
            above += rel > tol
        if not ok:
            lines.append(f"FAIL at t = {t}: {msg}")
        elif rel > tol:
            lines.append(f"WATCH at t = {t}: {msg}")
    head = (f"{child}: {len(by_time)} written times, worst relative deviation "
            f"{worst:.6e} (WATCH; {above} time(s) above {tol:g})")
    return ok_all, "\n".join([head] + lines + [
        "structure OK (WATCH reported above)" if ok_all else "FAIL"])


def partition(src: Path, child: str, tol: float) -> tuple[bool, str]:
    if src.is_dir():
        items = []
        for p in src.rglob("*"):
            if p.is_file() and p.name in CONTENT + PARTITION:
                items.append((p.relative_to(src.parent).parts, p.read_text(errors="replace")))
        return partition_from_members(items, child, tol)

    def stream():
        with tarfile.open(src, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                parts = tuple(m.name.split("/"))
                if parts[-1] not in CONTENT + PARTITION or child not in parts:
                    continue
                fh = tf.extractfile(m)
                yield parts, fh.read().decode("ascii", errors="replace")
    return partition_from_members(stream(), child, tol)


# ------------------------------------------------------- em65 gates -----

U_REF = 0.0196283           # m/s, report 10's reference (design section 6)
STARTUP_LOW = 0.9*U_REF     # 0.01767 m/s, reached by STARTUP_BY
STARTUP_HIGH = 1.2*U_REF    # 0.02355 m/s, never exceeded over [0, STARTUP_WINDOW]
STARTUP_BY = 0.2e-3
STARTUP_WINDOW = 0.5e-3
ONSET_RANGE = (0.9e-3, 1.6e-3)
FATAL = re.compile(r"FOAM FATAL (IO )?ERROR|FOAM aborting|Floating point exception")


def _records(path: Path, tag: str, key: str) -> list[tuple[float, float]]:
    out = []
    pat = re.compile(r"\b" + key + r"=(-?[0-9.]+(?:[eE][-+]?[0-9]+)?|-?nan|-?inf)")
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith(tag + " "):
                t, v = TIME.search(line), pat.search(line)
                if t and v:
                    out.append((float(t.group(1)), float(v.group(1))))
    return out


def startup(log: Path, exit_code: Path | None) -> tuple[bool, str]:
    rows = _records(log, "PHASE_CLOT_DIAGNOSTIC", "outletMeanUx")
    text_fatal = False
    with open(log, errors="replace") as fh:
        text_fatal = any(FATAL.search(line) for line in fh)
    lines = [f"U_ref {U_REF} m/s: need >= {STARTUP_LOW:.5f} by {STARTUP_BY*1e3:g} ms, "
             f"max <= {STARTUP_HIGH:.5f} over [0, {STARTUP_WINDOW*1e3:g}] ms"]
    if not rows:
        return False, "\n".join(lines + ["FAIL: no PHASE_CLOT_DIAGNOSTIC outletMeanUx records"])
    reached = [t for t, u in rows if u >= STARTUP_LOW]
    first = reached[0] if reached else None
    window = [u for t, u in rows if t <= STARTUP_WINDOW]
    peak = max(window) if window else None
    covers = rows[-1][0] >= STARTUP_WINDOW
    code_ok = True
    if exit_code is not None:
        try:
            code_ok = int(exit_code.read_text().strip()) == 0
        except (OSError, ValueError):
            code_ok = False
    finite = all(u == u and abs(u) != float("inf") for _t, u in rows)
    checks = [
        (f"reaches {STARTUP_LOW:.5f} m/s by {STARTUP_BY*1e3:g} ms "
         f"(first at {first!r} s)", first is not None and first <= STARTUP_BY),
        (f"max over [0, {STARTUP_WINDOW*1e3:g}] ms = {peak!r} <= {STARTUP_HIGH:.5f}",
         peak is not None and peak <= STARTUP_HIGH),
        (f"records reach {STARTUP_WINDOW*1e3:g} ms (last {rows[-1][0]!r} s)", covers),
        ("no crash (no FOAM FATAL, finite values, exit code 0 when given)",
         not text_fatal and finite and code_ok),
    ]
    ok = all(c for _l, c in checks)
    lines += [f"{'ok  ' if c else 'FAIL'} {l}" for l, c in checks]
    return ok, "\n".join(lines + ["PASS" if ok else "FAIL"])


def onset(log: Path) -> tuple[bool, str]:
    first = None
    with open(log, errors="replace") as fh:
        for line in fh:
            if line.startswith("PHASE_CLOT_RUPTURE "):
                m = re.search(r"\bnewlyRuptured=(\d+)", line)
                t = TIME.search(line)
                if m and t and int(m.group(1)) > 0:
                    first = float(t.group(1))
                    break
    lo, hi = ONSET_RANGE
    if first is None:
        return False, (f"FAIL: no damage onset (no newlyRuptured > 0) -- outside "
                       f"[{lo*1e3:g}, {hi*1e3:g}] ms")
    ok = lo <= first <= hi
    return ok, (f"first d* > 0 at {first!r} s against [{lo*1e3:g}, {hi*1e3:g}] ms: "
                + ("PASS" if ok else "FAIL"))


# ------------------------------------------------------------ self-test --

def _field(values) -> str:
    body = "\n".join(repr(float(v)) for v in values)
    return ("FoamFile\n{\n    class volScalarField;\n}\n"
            f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;\n")


def self_test(real_parent: Path | None) -> bool:
    results = []

    def expect(label, got, want):
        ok = got[0] is want
        results.append(ok)
        first = got[1].strip().splitlines()[-1]
        print(f"{'ok  ' if ok else 'BAD '} {label:<58} -> {'PASS' if got[0] else 'FAIL'}: {first[:90]}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = [
            "PHASE_CLOT_S14_CONFIG networkRupture=progressiveDamage",
            "Time = 1e-06",
            "PHASE_CLOT_CONTENT_FCT time=1e-06 activeFaces=5131 limitedFaces=12 lambdaMin=0.1 lambdaFluxMean=0.908731 negAfter=0",
            "PHASE_CLOT_COST time=1e-06 step=3.9 skeleton=0.08",
            "PHASE_CLOT_RUPTURE time=1e-06 criticalStretch=2.8 newlyRuptured=0 maxIntactStretch=1.0001",
            "PHASE_CLOT_CONTENT_FCT time=2e-06 activeFaces=5131 limitedFaces=13 lambdaMin=0.1 lambdaFluxMean=0.907 negAfter=0",
            "PHASE_CLOT_RUPTURE time=2e-06 criticalStretch=2.8 newlyRuptured=0 maxIntactStretch=1.0002",
            "PHASE_CLOT_RUPTURE time=3e-06 criticalStretch=2.8 newlyRuptured=1 maxIntactStretch=2.2",
            "PHASE_CLOT_CONTENT_FCT time=3e-06 activeFaces=5131 limitedFaces=13 lambdaMin=0.1 lambdaFluxMean=0.5 negAfter=0",
        ]
        par = tmp / "parent.log"
        par.write_text("\n".join(base) + "\n")
        same = [l.replace("progressiveDamage", "terminalRelease") for l in base]
        same = [l.replace("step=3.9", "step=4.4") for l in same]
        same.insert(5, "PHASE_CLOT_S16 time=1e-06 r=0 lawReleaseMax=0")
        # after the window the child differs by design:
        same[-1] = same[-1].replace("lambdaFluxMean=0.5", "lambdaFluxMean=0.4")
        ch = tmp / "child.log"
        ch.write_text("\n".join(same) + "\n")
        expect("G-id control: identical pre-damage, config/cost/new tags differ",
               gid(ch, par, None), True)
        bad = [l.replace("lambdaFluxMean=0.908731", "lambdaFluxMean=0") for l in same]
        (tmp / "noflux.log").write_text("\n".join(bad) + "\n")
        expect("G-id negative: partition in theta's shared limiter, shipped zero convention (lambda 0 at step 1)",
               gid(tmp / "noflux.log", par, None), False)
        short = [l for l in same if "time=2e-06" not in l]
        (tmp / "short.log").write_text("\n".join(short) + "\n")
        expect("G-id negative: a step missing from the child",
               gid(tmp / "short.log", par, None), False)
        (tmp / "empty.log").write_text("Time = 1e-06\n")
        expect("G-id negative: nothing to compare is not identity",
               gid(tmp / "empty.log", par, None), False)
        if real_parent is not None and real_parent.is_file():
            # The design's negative on a REAL log: the parent against itself
            # passes; the same log with the first active lambdaFluxMean set
            # to 0 (what the shipped no-flux convention prints at step 1)
            # fails, and names that line.
            head = []
            with open(real_parent, errors="replace") as fh:
                for i, line in enumerate(fh):
                    head.append(line)
                    if i > 60000:
                        break
            rp = tmp / "real.log"
            rp.write_text("".join(head))
            cut = 1e9
            expect("G-id control on a real log (itself)", gid(rp, rp, cut), True)
            mut, done = [], False
            for line in head:
                if (not done and line.startswith("PHASE_CLOT_CONTENT_FCT ")
                        and "activeFaces=0 " not in line):
                    line = re.sub(r"lambdaFluxMean=\S+", "lambdaFluxMean=0", line)
                    done = True
                mut.append(line)
            (tmp / "realmut.log").write_text("".join(mut))
            got = gid(tmp / "realmut.log", rp, cut)
            expect("G-id negative on a real log: lambdaFluxMean 0 at first active step",
                   got, False)
            if "PHASE_CLOT_CONTENT_FCT" not in got[1]:
                print("BAD  the real-log negative failed for the wrong tag")
                results.append(False)

        # S16p
        runs = tmp / "case" / "runs" / "em63_G400_s16"

        def write(t, C, H, I, P, B, skip=()):
            d = runs / t
            d.mkdir(parents=True, exist_ok=True)
            for name, v in zip(CONTENT + PARTITION, (C, H, I, P, B)):
                if name not in skip:
                    (d / name).write_text(_field(v))
        C, H = [0.1, 0.2, 0.0, 0.3], [0.05, 0.25, 0.0, 0.15]
        th = [c + h for c, h in zip(C, H)]
        write("0.001", C, H, [x*0.7 for x in th], [x*0.2 for x in th],
              [x - x*0.7 - x*0.2 for x in th])
        write("0.002", C, H, th, [0.0]*4, [0.0]*4)
        expect("partition WATCH control: the partition sums to theta",
               partition(tmp / "case", "em63_G400_s16", 1e-12), True)
        # a roll-back skipped: one deposit counted twice in thetaIntact
        dep = 1e-4
        write("0.003", C, [h + dep for h in H], [t + 2*dep for t in th],
              [0.0]*4, [0.0]*4)
        got = partition(tmp / "case", "em63_G400_s16", 1e-12)
        expect("partition WATCH: a deposit counted twice is REPORTED, not failed",
               got, True)
        if "WATCH at t = 0.003" not in got[1]:
            print("BAD  the WATCH did not report the double-counted deposit")
            results.append(False)
        # a negative component is still a structural FAIL
        write("0.0035", C, H, [t - 1e-3 for t in th], [1e-3]*4,
              [-1e-9] + [0.0]*3)
        expect("partition negative: a negative component fails",
               partition(tmp / "case", "em63_G400_s16", 1e-12), False)
        import shutil as _sh
        _sh.rmtree(runs / "0.0035")
        (runs / "0.003").rename(runs.parent / "hold")
        write("0.004", C, H, th, [0.0]*4, [0.0]*4, skip=("thetaBroken",))
        expect("partition negative: a written time without the partition",
               partition(tmp / "case", "em63_G400_s16", 1e-12), False)
        expect("partition negative: a child with no written time",
               partition(tmp / "case", "em99_absent", 1e-12), False)
        # the same through the tar.gz stream
        import shutil
        shutil.rmtree(runs / "0.004")
        tgz = tmp / "case-after-run-X.tar.gz"
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(tmp / "case", arcname="case-after-run-X")
        expect("partition WATCH control through the tar.gz stream",
               partition(tgz, "em63_G400_s16", 1e-12), True)
    # em65 gates. Synthetic control and negatives, then the real-log
    # negatives the design names: em61 (S15, no final relaxation) reaches
    # 90% of U_ref only at 3.17 ms, and its first damage is at 2.600 ms.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def diag(pairs, extra=""):
            p = tmp / f"log{len(list(tmp.iterdir()))}"
            p.write_text("".join(
                f"Time = {t}\nPHASE_CLOT_DIAGNOSTIC time={t} inventory=1 outletMeanUx={u} outletCells=70\n"
                for t, u in pairs) + extra)
            return p
        ts = [k*2.5e-5 for k in range(1, 41)]          # 25 us .. 1 ms
        good = diag([(t, min(0.0196, 0.0196*t/1.0e-4)) for t in ts])
        expect("start-up control: 0.0196 m/s by 0.1 ms, no overshoot",
               startup(good, None), True)
        slow = diag([(t, 0.0196*t/1.0e-3) for t in ts])
        expect("start-up negative: 90% only after 0.2 ms", startup(slow, None), False)
        # the overshoot at the 12th record (t = 0.3 ms), chosen by index:
        # a float compare against 3e-4 never matched 12*2.5e-5.
        over = diag([(t, 0.025 if k == 11 else 0.0196) for k, t in enumerate(ts)])
        expect("start-up negative: overshoot 0.025 > 0.02355 m/s", startup(over, None), False)
        crash = diag([(t, 0.0196) for t in ts[:30]],
                     "--> FOAM FATAL ERROR: something\n")
        expect("start-up negative: a crash (FOAM FATAL)", startup(crash, None), False)
        short = diag([(t, 0.0196) for t in ts[:12]])
        expect("start-up negative: log ends before 0.5 ms", startup(short, None), False)
        code = tmp / "run-exit-code.txt"
        code.write_text("136\n")
        expect("start-up negative: nonzero exit code", startup(good, code), False)

        ts_long = [k*2.5e-5 for k in range(1, 121)]   # 25 us .. 3 ms

        def rup(first):
            p = tmp / f"rup{len(list(tmp.iterdir()))}"
            body = ""
            for t in ts_long:
                n = 1 if first is not None and t >= first else 0
                body += f"PHASE_CLOT_RUPTURE time={t} criticalStretch=2.8 newlyRuptured={n}\n"
            p.write_text(body)
            return p
        expect("onset control: first d* > 0 at 1.2 ms", onset(rup(1.2e-3)), True)
        expect("onset negative: at 0.5 ms (too early)", onset(rup(0.5e-3)), False)
        expect("onset negative: none at all", onset(rup(None)), False)
    if real_parent is not None and real_parent.is_file():
        got = startup(real_parent, None)
        expect("start-up negative on a REAL log: em61 (90% at 3.17 ms)", got, False)
        if "first at 0.003166666667" not in got[1]:
            print("BAD  em61's start-up failed for a reason other than the 3.17 ms crossing")
            results.append(False)
        got = onset(real_parent)
        expect("onset negative on a REAL log: em61's S15 onset at 2.600 ms", got, False)
        if "0.0026" not in got[1]:
            print("BAD  em61's onset failed for a reason other than 2.600 ms")
            results.append(False)

    ok = all(results)
    print(f"self-test: {sum(results)}/{len(results)} as expected")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gid"); g.add_argument("child"); g.add_argument("parent")
    g.add_argument("--cutoff", type=float, default=None)
    p = sub.add_parser("partition"); p.add_argument("src"); p.add_argument("child_id")
    p.add_argument("--tol", type=float, default=1e-12)
    su = sub.add_parser("startup"); su.add_argument("log")
    su.add_argument("--exit-code", default=None)
    on = sub.add_parser("onset"); on.add_argument("log")
    s = sub.add_parser("self-test"); s.add_argument("real_parent", nargs="?")
    a = ap.parse_args()
    if a.cmd == "gid":
        for f in (a.child, a.parent):
            if not Path(f).is_file():
                print(f"unreadable: {f}"); return 2
        ok, msg = gid(Path(a.child), Path(a.parent), a.cutoff)
    elif a.cmd == "partition":
        if not Path(a.src).exists():
            print(f"unreadable: {a.src}"); return 2
        ok, msg = partition(Path(a.src), a.child_id, a.tol)
    elif a.cmd in ("startup", "onset"):
        if not Path(a.log).is_file():
            print(f"unreadable: {a.log}"); return 2
        if a.cmd == "startup":
            ok, msg = startup(Path(a.log), Path(a.exit_code) if a.exit_code else None)
        else:
            ok, msg = onset(Path(a.log))
    else:
        return 0 if self_test(Path(a.real_parent) if a.real_parent else None) else 1
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
