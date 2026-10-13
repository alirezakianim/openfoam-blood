#!/usr/bin/env python3
"""S25 section 2C: the restart driver picks the last FULL, INTACT write, and nothing else.

Offline, no OpenFOAM. Synthetic fixtures built to the shape of D75's returned run
(preflight/20261009-171539: a frame directory holds U p Vs ... and uniform/time but NO w; a full
write holds w and 50 fields), then every refusal, each run on its own and each seen to fire for
its stated reason. When the D75 tarball is present the driver is also run on the real archive.

A gate that cannot fail proves nothing, so the last section deletes each guard in a scratch copy of
the driver and requires the matching test to fail.

Exit 0 when every check passes. Run: python3 -B scripts/check_restart_driver.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASE_SCRIPTS = ROOT / "cases" / "AM2020_planar_mural_detachment" / "scripts"
DRIVER = CASE_SCRIPTS / "restart_driver.py"
D75 = ROOT / "preflight" / "20261009-171539-am2020-main-case" / "case-after-run-AM2020_planar_mural_detachment.tar.gz"

FOOTER = "\n\n// ************************************************************************* //\n"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def field(path: Path, body: str = "internalField uniform 0;") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("FoamFile\n{\n    format ascii;\n}\n" + body + FOOTER)


def uniform_time(d: Path, value: str, index: int) -> None:
    field(d / "uniform" / "time", f'value {value};\nname "{d.name}";\nindex {index};\ndeltaT 1e-06;')


def make_full(case: Path, name: str, index: int) -> Path:
    d = case / name
    for f in ("w", "U", "p", "phiSkel", "thetaC", "thetaH", "nP", "Wp", "phiSkelOwn"):
        field(d / f)
    uniform_time(d, name, index)
    return d


def make_frame(case: Path, name: str, with_uniform_time: bool = True) -> Path:
    d = case / name
    for f in ("Gfield", "U", "Us", "Vs", "clotDrag", "p", "poroPerm", "q"):
        field(d / f)
    if with_uniform_time:
        uniform_time(d, name, 7)
    return d


CONTROL = """FoamFile
{
    format ascii;
    object controlDict;
}
startFrom startTime;
startTime 0;
endTime 0.04;
writePrecision 17;
"""


def make_case(root: Path, *, control: str = CONTROL) -> Path:
    case = root / "case"
    (case / "system").mkdir(parents=True)
    (case / "constant").mkdir()
    (case / "system" / "controlDict").write_text(control)
    (case / "constant" / "phaseClotProperties").write_text("timeScale 1.0;\n")
    field(case / "0" / "w")
    return case


def expect(label: str, driver, want_code: int, want_text: str, case: Path, *, source: Path | None = None, apply: bool = False):
    try:
        rec = driver.restart(case, source, apply)
    except driver.RestartError as error:
        assert error.code == want_code, f"{label}: exit {error.code}, wanted {want_code}: {error}"
        assert want_text in str(error), f"{label}: refused for the WRONG reason: {error}"
        print(f"  refuses  {label:<64} exit {error.code}")
        return None
    raise AssertionError(f"{label}: the driver ACCEPTED (chose {rec['chosen']}); it must refuse")


def run_all(driver, tmp: Path) -> None:
    n = [0]

    def fresh() -> Path:
        n[0] += 1
        d = tmp / f"t{n[0]}"
        d.mkdir()
        return make_case(d)

    # 1. the D75 shape: frames between full writes, a frame NEWEST (the latestTime trap), a cut-off full write
    case = fresh()
    make_frame(case, "0.0002")
    make_frame(case, "0.0004", with_uniform_time=False)
    make_full(case, "0.01", 3000)
    make_frame(case, "0.0102")
    make_full(case, "0.02", 6000)
    make_frame(case, "0.0202")
    bad = make_full(case, "0.03", 9000)
    w = bad / "w"
    w.write_text(w.read_text()[:-90])                      # the footer cut off: killed mid-write
    make_frame(case, "0.0302")                              # the NEWEST directory is a frame
    chosen, verdicts = driver.choose(case)
    kinds = {v["name"]: v["kind"] for v in verdicts}
    assert chosen.name == "0.02", chosen.name
    assert kinds == {"0.0002": "frame", "0.0004": "frame", "0.01": "full", "0.0102": "frame", "0.02": "full",
                     "0.0202": "frame", "0.03": "damaged", "0.0302": "frame"}, kinds
    # a frame WITH uniform/time is still a frame: w is what discriminates (D75's 0.0002 has uniform/time)
    assert (case / "0.0002" / "uniform" / "time").is_file() and kinds["0.0002"] == "frame"
    print(f"  picks    {chosen.name} over a newer frame (0.0302) and a cut-off full write (0.03)")
    rec = driver.restart(case, None, True)
    ctl = (case / "system" / "controlDict").read_text()
    assert re.search(r"(?m)^startFrom\s+startTime;", ctl) and re.search(r"(?m)^startTime\s+0\.02;", ctl), ctl
    assert "latestTime" not in ctl
    assert sorted(rec["moved_aside"]) == ["0.0202", "0.03", "0.0302"], rec["moved_aside"]
    assert (case / "restart_discarded" / "0.03" / "w").is_file() and not (case / "0.03").exists()
    saved = json.loads((case / "restart.json").read_text())
    assert saved["chosen"] == "0.02" and set(saved["sha256"]) >= {"w", "U", "p", "phiSkel", "uniform/time"}
    print("  applies  startFrom startTime; startTime 0.02; newer directories moved aside, none deleted; restart.json written")

    # 2. refusals, each on its own
    case = fresh()
    make_frame(case, "0.0002")
    make_frame(case, "0.0004", with_uniform_time=False)
    expect("only frames, no full write", driver, 2, "no full intact write", case)

    case = fresh()
    make_full(case, "0.01", 1)
    (case / "0.01" / "w").unlink()
    expect("a full write without w", driver, 2, "no w", case)

    case = fresh()
    make_full(case, "0.01", 1)
    (case / "0.01" / "phiSkel").unlink()
    expect("a full write without phiSkel (S25.1: the constructor refuses a restart without it)", driver, 2, "no phiSkel", case)

    case = fresh()
    make_full(case, "0.01", 1)
    (case / "0.01" / "U").unlink()
    expect("a full write without U (MUST_READ in incompressibleFluid)", driver, 2, "no U", case)

    case = fresh()
    d = make_full(case, "0.01", 1)
    (d / "uniform" / "time").unlink()
    expect("a full write without uniform/time", driver, 2, "no uniform/time", case)

    case = fresh()
    d = make_full(case, "0.01", 1)
    (d / "thetaC").write_text("")
    expect("an empty field file", driver, 2, "thetaC is empty", case)

    case = fresh()
    d = make_full(case, "0.01", 1)
    t = (d / "p").read_text()
    (d / "p").write_text(t[: len(t) // 2])
    expect("a field cut off mid-file", driver, 2, "p has no closing footer line", case)

    case = fresh()
    d = make_full(case, "0.01", 1)
    uniform_time(d, "0.5", 1)
    expect("uniform/time value disagrees with the directory name", driver, 2, "uniform/time says value 0.5", case)

    case = fresh()
    d = make_full(case, "0.01", 0)
    expect("uniform/time index 0", driver, 2, "index 0", case)

    case = fresh()
    make_full(case, "0.01", 1)
    (case / "system" / "controlDict").write_text(CONTROL.replace("startFrom startTime;", "startFrom latestTime;"))
    expect("controlDict says startFrom latestTime", driver, 4, "startFrom latestTime", case)

    case = fresh()
    make_full(case, "0.05", 1)
    expect("the last full write is not before endTime", driver, 4, "nothing to run", case)

    case = fresh()                                  # the REGENERATED case: 0/ only, no time directory
    src = tmp / f"src{n[0]}"                        # the previous session's tree: dictionaries + a full write
    shutil.copytree(case, src)
    make_full(src, "0.01", 1)
    (src / "constant" / "phaseClotProperties").write_text("timeScale 8.33;\n")
    expect("regenerated phaseClotProperties differs from the writing session's", driver, 3, "constant/phaseClotProperties", case, source=src)
    rec = driver.restart(case, src, False, accept_differences=True)
    assert rec["dictionaries_differ_and_were_accepted"] == ["constant/phaseClotProperties"], rec
    print("  accepts  differing dictionaries ONLY with the manual flag, and records which")
    # a controlDict that differs only in startFrom/startTime (a second restart session) is accepted
    (src / "constant" / "phaseClotProperties").write_text("timeScale 1.0;\n")
    (src / "system" / "controlDict").write_text(CONTROL.replace("startTime 0;", "startTime 0.005;"))
    rec = driver.restart(case, src, True)
    assert rec["chosen"] == "0.01" and (case / "0.01" / "w").is_file()
    assert re.search(r"(?m)^startTime\s+0\.01;", (case / "system" / "controlDict").read_text())
    print("  accepts  a second restart session (controlDict differs only in startFrom/startTime); copies 0.01 in")
    expect("a second apply onto a directory that already exists", driver, 4, "already exists", case, source=src, apply=True)

    # 3. the tarball extractor writes full writes and dictionaries only, one member at a time, inside dest
    tar = tmp / "case-after-run-X.tar.gz"
    pack = tmp / "pack" / "case-after-run-X" / "runs" / "childA"
    (pack / "system").mkdir(parents=True)
    (pack / "constant").mkdir()
    (pack / "system" / "controlDict").write_text(CONTROL)
    (pack / "constant" / "phaseClotProperties").write_text("timeScale 1.0;\n")
    field(pack / "0" / "w")
    make_frame(pack, "0.0002")
    make_full(pack, "0.01", 1)
    import tarfile
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(tmp / "pack" / "case-after-run-X", arcname="case-after-run-X")
    out = tmp / "extracted"
    info = driver.extract_from_tarball(tar, ["childA"], out)
    got = sorted(str(p.relative_to(out / "childA")).split("/")[0] for p in (out / "childA").iterdir())
    assert got == ["0.01", "constant", "system"], got
    assert info["children"] == {"childA": ["0.01"]}
    try:
        driver.extract_from_tarball(tar, ["childB"], tmp / "e2")
        raise AssertionError("an archive without the child was accepted")
    except driver.RestartError as error:
        assert error.code == 2 and "childB" in str(error)
    print("  extracts full writes + dictionaries only (frames and 0/ stay in the archive); a missing child is refused")


def make_child(runs: Path, cid: str, control: str = CONTROL) -> Path:
    c = runs / cid
    (c / "system").mkdir(parents=True)
    (c / "constant").mkdir()
    (c / "system" / "controlDict").write_text(control)
    (c / "constant" / "phaseClotProperties").write_text("timeScale 1.0;\n")
    field(c / "0" / "w")
    return c


def rt1_fixture(tmp: Path, label: str, *, finished: str | None = "0", control_b: str | None = None, stop: int = 47, idx: int = 42):
    """runs/A: a cold start that wrote a full directory at every step 38..47 (names 0.0038 .. 0.0047) and exited; runs/B: its regenerated
    twin with child.json naming A and index `idx`. Returns (runs, A, B)."""
    runs = tmp / f"rt1_{label}"
    runs.mkdir()
    a = make_child(runs, "A")
    for k in range(38, 48):
        make_full(a, f"0.{k:04d}", k)
    if finished is not None:
        (a / "run-exit-code.txt").write_text(finished + "\n")
    b = make_child(runs, "B", control_b or CONTROL)
    (b / "child.json").write_text(json.dumps({"child_id": "B", "restart_from_child": "A", "restart_index": idx, "stop_after_step": stop}))
    return runs, a, b


def sibling_restart(driver, tmp: Path) -> None:
    """S25 B76-RT: RT-1's B restarts from its sibling A at A's step N0, not at A's latest step."""
    def cli(runs: Path, b: Path, *extra: str) -> int:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return driver.main(["restart", str(b), "--from-child-json", str(b / "child.json"), "--runs-dir", str(runs), "--apply", *extra])

    # the accepted case: index 42 is chosen although 0.0047 (step 47) is the latest full write
    runs, a, b = rt1_fixture(tmp, "ok")
    assert driver.choose(a)[0].name == "0.0047"
    chosen, verdicts = driver.choose(a, 42)
    assert chosen.name == "0.0042" and [v["index"] for v in verdicts] == list(range(38, 48)), verdicts
    assert cli(runs, b) == 0
    rec = json.loads((b / "restart.json").read_text())
    ctl = (b / "system" / "controlDict").read_text()
    assert rec["chosen"] == "0.0042" and rec["chosen_index"] == 42 and rec["sibling"] == "A" and rec["requested_index"] == 42, rec
    assert re.search(r"(?m)^startTime\s+0\.0042;", ctl) and (b / "0.0042" / "w").is_file() and not (b / "0.0047").exists()
    print("  restarts B from A's step 42 (not the latest 47); restart.json names the sibling and the index")

    def refuses(label: str, want_code: int, want_text: str, **kw):
        runs_, a_, b_ = rt1_fixture(tmp, label.replace(" ", "_")[:40] + str(len(list(tmp.iterdir()))), **{k: v for k, v in kw.items() if k in ("finished", "control_b", "stop", "idx")})
        mut = kw.get("mutate")
        if mut:
            mut(runs_, a_, b_)
        import contextlib, io
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = driver.main(["restart", str(b_), "--from-child-json", str(b_ / "child.json"), "--runs-dir", str(runs_), "--apply"])
        assert rc == want_code, f"{label}: exit {rc}, wanted {want_code}: {err.getvalue()}"
        assert want_text in err.getvalue(), f"{label}: refused for the WRONG reason: {err.getvalue()}"
        assert not (b_ / "restart.json").exists() and "startTime 0;" in (b_ / "system" / "controlDict").read_text(), f"{label}: B was edited"
        print(f"  refuses  {label:<64} exit {rc}")

    refuses("the sibling has not finished (no run-exit-code.txt)", 4, "has not finished normally", finished=None)
    refuses("the sibling exited 1", 4, "has not finished normally", finished="1")
    refuses("the sibling directory is absent", 4, "does not exist", mutate=lambda r, a_, b_: shutil.rmtree(a_))
    refuses("no step-42 write (purged)", 2, "index 42", mutate=lambda r, a_, b_: shutil.rmtree(a_ / "0.0042"))
    refuses("step 42 is a cut-off write", 2, "index 42",
            mutate=lambda r, a_, b_: (a_ / "0.0042" / "w").write_text((a_ / "0.0042" / "w").read_text()[:-90]))
    refuses("step 42 is a frame (no w)", 2, "index 42", mutate=lambda r, a_, b_: (a_ / "0.0042" / "w").unlink())
    refuses("B's dictionaries differ from A's", 3, "differs from the session that wrote",
            mutate=lambda r, a_, b_: (b_ / "constant" / "phaseClotProperties").write_text("timeScale 8.0;\n"))
    refuses("restart_index not before stop_after_step", 4, "must be an integer in [1, stop_after_step", idx=47)
    refuses("child.json without restart_from_child", 4, "needs restart_from_child",
            mutate=lambda r, a_, b_: (b_ / "child.json").write_text(json.dumps({"child_id": "B", "stop_after_step": 47})))
    refuses("restart_from_child naming a path", 4, "is not a child id",
            mutate=lambda r, a_, b_: (b_ / "child.json").write_text(json.dumps({"child_id": "B", "restart_from_child": "../A", "restart_index": 42, "stop_after_step": 47})))
    refuses("a boolean restart_index", 4, "must be an integer in [1, stop_after_step",
            mutate=lambda r, a_, b_: (b_ / "child.json").write_text(json.dumps({"child_id": "B", "restart_from_child": "A", "restart_index": True, "stop_after_step": 47})))
    runs, a, b = rt1_fixture(tmp, "args")
    for label, argv, text in (
        ("--runs-dir without --from-child-json", ["restart", str(b), "--runs-dir", str(runs)], "silently inert"),
        ("--from-child-json with --index", ["restart", str(b), "--from-child-json", str(b / "child.json"), "--runs-dir", str(runs), "--index", "42"], "takes neither"),
        ("--from-child-json without --runs-dir", ["restart", str(b), "--from-child-json", str(b / "child.json")], "needs --runs-dir"),
    ):
        import contextlib, io
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = driver.main(argv)
        assert rc == 4 and text in err.getvalue(), (label, rc, err.getvalue())
        print(f"  refuses  {label:<64} exit {rc}")
    # --index alone (no sibling): the plain restart of a case from its own directory at a named step
    c = make_case(tmp / "idxalone") if (tmp / "idxalone").mkdir() is None else None
    for k in (5, 6, 7):
        make_full(c, f"0.00{k}", k)
    expect_fail = None
    try:
        driver.restart(c, None, False, False, 9)
    except driver.RestartError as error:
        expect_fail = error
    assert expect_fail is not None and expect_fail.code == 2 and "index 9" in str(expect_fail)
    assert driver.restart(c, None, False, False, 6)["chosen"] == "0.006"
    print("  --index: a named step is chosen; a step that was never written is refused (exit 2), with no fallback to the latest")


def real_data(driver, tmp: Path) -> str:
    if not D75.is_file():
        return "SKIPPED (the D75 tarball is absent here; the synthetic fixtures above carry the checks)"
    info = driver.extract_from_tarball(D75, ["D75_refI"], tmp / "d75")
    assert "0.0002" not in info["children"]["D75_refI"], info
    # measured on the tarball's member list: 8 full writes (every 5 ms) among 216 time directories, the rest frames
    assert info["children"]["D75_refI"] == ["0.005", "0.01", "0.015", "0.02", "0.025", "0.03", "0.035", "0.04"], info["children"]
    # the real 0.0002 frame was not extracted (no w), but is it a frame to the driver? classify it from the tarball
    import tarfile
    frame = tmp / "d75frame" / "0.0002"
    pre = "case-after-run-AM2020_planar_mural_detachment/runs/D75_refI/0.0002/"
    with tarfile.open(D75, "r:gz") as tf:
        for m in tf:
            if m.isfile() and m.name.startswith(pre):
                dst = frame / m.name[len(pre):]
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(tf.extractfile(m).read())
    kind, why = driver.classify(frame)
    assert kind == "frame" and (frame / "uniform" / "time").is_file(), (kind, why)
    chosen, verdicts = driver.choose(tmp / "d75" / "D75_refI")
    assert chosen.name == "0.04" and verdicts[-1]["kind"] == "full", verdicts
    return f"D75_refI: real frame 0.0002 is a {kind} although it holds uniform/time ({why[:60]}...); real 0.04 is full and intact"


def guards_can_fail(tmp: Path) -> None:
    """Delete each guard in a scratch copy of the driver; the matching fixture must then be ACCEPTED."""
    text = DRIVER.read_text()
    edits = (
        ("the w requirement", 'FULL_WRITE_REQUIRED = ("w", "U", "p", "phiSkel", "uniform/time")', 'FULL_WRITE_REQUIRED = ("U", "p", "phiSkel", "uniform/time")'),
        ("the phiSkel requirement", 'FULL_WRITE_REQUIRED = ("w", "U", "p", "phiSkel", "uniform/time")', 'FULL_WRITE_REQUIRED = ("w", "U", "p", "uniform/time")'),
        ("the footer test", "if not tail or not FOOTER.match(tail[-1]):", "if False:"),
        ("the latestTime refusal", 'if re.search(r"(?m)^\\s*startFrom\\s+latestTime\\s*;", ctl):', "if False:"),
        ("the index match", "if kind == \"full\" and (index is None or step == index):", "if kind == \"full\":"),
        ("the sibling finished test", 'if not code.is_file() or code.read_text().strip() != "0":', "if False:"),
    )
    for what, old, new in edits:
        assert text.count(old) == 1, f"the guard text for {what} moved: {old!r}"
        scratch = tmp / f"driver_no_{what.replace(' ', '_')}.py"
        scratch.write_text(text.replace(old, new))
        mod = load(scratch, f"restart_driver_no_{what.replace(' ', '_')}")
        root = tmp / f"neg_{what.replace(' ', '_')}"
        root.mkdir()
        c = make_case(root)
        if what == "the w requirement":
            field(make_frame(c, "0.0002") / "phiSkel")            # a frame that holds everything but w (and uniform/time)
            rec = mod.choose(c)[0]
            assert rec.name == "0.0002"
        elif what == "the phiSkel requirement":
            d = make_full(c, "0.01", 1)
            (d / "phiSkel").unlink()                              # a write from a build before S25: no phiSkel
            assert mod.choose(c)[0].name == "0.01"
        elif what == "the footer test":
            d = make_full(c, "0.01", 1)
            t = (d / "p").read_text()
            (d / "p").write_text(t[: len(t) // 2])
            assert mod.choose(c)[0].name == "0.01"
        elif what == "the index match":
            for k in (5, 6, 7):
                make_full(c, f"0.00{k}", k)
            assert mod.choose(c, 6)[0].name == "0.007"          # asked for step 6, got the latest
        elif what == "the sibling finished test":
            runs, a_, b_ = rt1_fixture(root, "unfinished", finished=None)
            src, idx, sib = mod.sibling_source(b_, b_ / "child.json", runs)
            assert sib == "A" and idx == 42                      # an unfinished A is accepted as the source
        else:
            make_full(c, "0.01", 1)
            (c / "system" / "controlDict").write_text(CONTROL.replace("startFrom startTime;", "startFrom latestTime;"))
            assert mod.restart(c, None, False)["chosen"] == "0.01"
        print(f"  can fail: without {what}, the matching bad fixture is ACCEPTED")


def generator_guard(tmp: Path) -> None:
    gen = load(CASE_SCRIPTS / "create_cases.py", "create_cases_for_restart_check")
    ok = tmp / "ok_controlDict"
    ok.write_text(CONTROL)
    gen.restart_contract_refusal(ok, "x")
    for label, text, want in (
        ("startFrom latestTime", CONTROL.replace("startFrom startTime;", "startFrom latestTime;"), "startFrom ['latestTime']"),
        ("writePrecision 12", CONTROL.replace("writePrecision 17;", "writePrecision 12;"), "writePrecision ['12']"),
        ("no writePrecision", CONTROL.replace("writePrecision 17;\n", ""), "writePrecision []"),
    ):
        bad = tmp / "bad_controlDict"
        bad.write_text(text)
        try:
            gen.restart_contract_refusal(bad, "x")
        except ValueError as error:
            assert want in str(error), (label, str(error))
            print(f"  generator refuses a controlDict with {label}")
        else:
            raise AssertionError(f"the generator accepted a controlDict with {label}")


def main() -> int:
    driver = load(DRIVER, "restart_driver")
    with tempfile.TemporaryDirectory(prefix="restart-driver-check-") as t:
        tmp = Path(t)
        print("fixtures and refusals:")
        run_all(driver, tmp)
        print("RT-1's sibling restart (--index, --from-child-json):")
        sibling_restart(driver, tmp)
        print("the generator's controlDict guard:")
        generator_guard(tmp)
        print("the guards can fail:")
        guards_can_fail(tmp)
        print("real data: " + real_data(driver, tmp))
    print("RESTART DRIVER OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
