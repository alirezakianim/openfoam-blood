#!/usr/bin/env python3
"""Negative tests for the PHASE_CLOT_CONTENT parser in assess_case.py.

WHY THIS EXISTS. On 2026-09-08 my own shell-matrix gate scored a record that
was MISSING its predicate fields as healthy, because `.get(key, 0)` makes
"absent" and "zero" the same thing -- the identical defect I had fixed in two
other gates the day before. The content parser is new, it is the instrument
that will read the T2a identity test, and an instrument that cannot fail is
not an instrument.

Every case below is a DISCRIMINATION: the parser must refuse it, or must
classify it differently from a healthy log. A test that only feeds good input
proves nothing.

Run offline; no OpenFOAM, no network.
"""
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(
    0,
    str(HERE.parent / "cases" / "AM2020_planar_mural_detachment" / "scripts"),
)

import assess_case as A  # noqa: E402

# S6c-2: equivMax, equivL1 and chiVolume are GONE from this record.
#
# They were the Stage-1 equivalence test -- theta/theta_ref against
# chiLegacy = max(0.5*(1 - c), 0) -- and c is deleted, so the comparison has
# no second operand. They carried the non-finite and truncation tests below,
# which have been repointed at contentVolume: a numeric key that is still
# there, so the tests still fire for the reason they were written.
#
# `operator=content` rather than `matched`: contentOperator 'matched' is
# refused at construction since S6c-2, and a fixture is not the place to
# keep a value the solver would reject.
GOOD = (
    "PHASE_CLOT_CONTENT time=1.47e-06 thetaMin=0 thetaMax=0.45 thetaCNeg=0 "
    "thetaHNeg=0 contentVolume=6.0669e-16 contentChange=0 "
    "operator=content contentSourceDt=0 kinetics=off"
)


def records(text):
    recs, ops = A.parse_content(text)
    bad = [
        r for r in recs
        if [k for k in A.CONTENT_REQUIRED if k not in r]
        or not r["_operator"]
        or any(not __import__("math").isfinite(r[k]) for k in A.CONTENT_REQUIRED)
    ]
    return recs, bad, ops


failures = []


def expect(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond:
        failures.append(name)


# 1. A healthy record parses, and the operator word survives the numeric sweep.
recs, bad, ops = records(GOOD)
expect("a complete record parses", len(recs) == 1 and not bad)
expect("the operator WORD is captured", ops == ["content"], repr(ops))

# 2. THE REAL BATCH 27 LOG PREDATES BOTH NEW KEYS and must be UNASSESSABLE,
#    not silently read as the `content` operator.
legacy = GOOD.replace(" operator=content", "").replace(" contentSourceDt=0", "")
recs, bad, ops = records(legacy)
expect("a pre-2026-09-09 record is refused", len(recs) == 1 and len(bad) == 1)

# 3. The operator word alone missing -- the record must not default.
recs, bad, _ = records(GOOD.replace(" operator=content", ""))
expect("a record with no operator is refused", len(bad) == 1)

# 4. nan and inf are dropped by the numeric regex, so the key goes missing.
for tok in ("nan", "inf", "-inf"):
    recs, bad, _ = records(
        GOOD.replace("contentVolume=6.0669e-16", "contentVolume=" + tok))
    expect("contentVolume=%s is refused" % tok, len(bad) == 1)

# 5. A truncated record -- one key gone -- must be refused, not averaged over.
recs, bad, _ = records(GOOD.replace(" contentChange=0", ""))
expect("a truncated record is refused", len(bad) == 1)

# 6. The INIT line is NOT a step record and must not be counted as one.
init = (
    "PHASE_CLOT_CONTENT_INIT thetaRef=0.45 vP=1e-17 thetaMin=0 "
    "thetaMax=0.45 contentVolume=6.0669e-16"
)
recs, bad, _ = records(init)
expect("the INIT line is not a step record", len(recs) == 0, "recs=%d" % len(recs))

# 7. A log with no content records at all is NOT_EVALUATED territory, not a pass.
recs, bad, _ = records("Time = 1\nExecutionTime = 0 s\n")
expect("a log with no content records yields none", len(recs) == 0)

# 8. THE VACUOUS-TEST GUARD. Every mutation above must actually change the
#    text it was applied to; a pattern that does not match would make its
#    test pass for the wrong reason. This is the check my first negative test
#    on 2026-09-09 did not have, and it passed vacuously as a result.
for pat in (
    " operator=content", " contentSourceDt=0", "contentVolume=6.0669e-16",
    " contentChange=0",
):
    expect("mutation pattern present in the fixture: %r" % pat, pat in GOOD)

# 9. Mixed operators in one log must both be visible -- a run that changed
#    operator mid-flight is a fact the reader needs, not one to collapse.
#
#    STILL TESTED WITH `matched`, THOUGH S6c-2 REFUSES IT. This is a
#    property of the PARSER, not of the solver: archived logs from before
#    the deletion carry that word, and a parser that silently collapsed
#    two operators into one would misreport them. The refusal stops new
#    runs producing it; it does not stop old logs containing it.
recs, bad, ops = records(
    GOOD + "\n" + GOOD.replace("operator=content", "operator=matched")
)
expect(
    "both operator words are reported",
    sorted(set(ops)) == ["content", "matched"],
    repr(sorted(set(ops))),
)

# 10. THE FROZEN-theta PATH. A carrier-less step marks the whole history
#     unassessable, even when every record it did write is complete.
recs, bad, _ = records(GOOD)
na = A.CONTENT_NOT_ADVECTED.findall(
    GOOD + "\nPHASE_CLOT_CONTENT_NOT_ADVECTED time=1 operator=matched "
    "reason=noCarrier\n"
)
expect("a not-advected step is detected", len(na) == 1, "found=%d" % len(na))
status = (
    "NOT_EVALUATED" if not recs
    else "UNASSESSABLE" if bad or na
    else "REPORTED"
)
expect("one not-advected step makes the history UNASSESSABLE",
       status == "UNASSESSABLE", status)
expect("a clean log without it is REPORTED",
       ("NOT_EVALUATED" if not recs else "UNASSESSABLE"
        if bad or A.CONTENT_NOT_ADVECTED.findall(GOOD) else "REPORTED")
       == "REPORTED")

# 11. THE FIELD SIDE OF THE SAME DELETION, which is a SEPARATE gate from the
#     log parser above and was missed once already.
#
#     assess_case.py measured the clot from the written `c` field in eight
#     places: `c < 0` is clot, `c < -0.5` is the edge contour, `(1-c)/2` is the
#     material integral, and the bounds gate refuses a final time with no `c`
#     in it -- "missing c, Dg or Us" -- which after S6c-2 is EVERY child.
#     The field is reconstructed now, c = 2q - 1, and the reconstruction is
#     exact: measured against run 20260920-060033 / em45, which wrote all four
#     fields, it matches the written `c` to 6.7e-16 at t = 0.
#
#     A direct read of "c" outside the two helpers puts the dependency back
#     without anything failing, so pin that it is gone.
import re as _re  # noqa: E402

_ASSESS = (
    HERE.parent / "cases" / "AM2020_planar_mural_detachment"
    / "scripts" / "assess_case.py"
).read_text()


def _c_reads_outside_the_helpers(text: str) -> list[str]:
    """Lines reading the written `c` field from anywhere but phase_c."""
    lines = text.splitlines()
    spans = []
    for i, line in enumerate(lines):
        if _re.match(r"^def (phase_c|c_convention)\b", line):
            j = i + 1
            while j < len(lines) and not _re.match(r"^\S", lines[j]):
                j += 1
            spans.append((i, j))
    out = []
    for i, line in enumerate(lines):
        if not _re.search(r'find_field\([^)]*,\s*"c"\s*\)', line):
            continue
        if any(a <= i < b for a, b in spans):
            continue
        out.append(f"{i + 1}: {line.strip()}")
    return out


expect("phase_c exists and is the only reader of the written c",
       "def phase_c(" in _ASSESS and not _c_reads_outside_the_helpers(_ASSESS),
       "; ".join(_c_reads_outside_the_helpers(_ASSESS)))

# And the check itself must fire: put one read back, outside the helpers.
_mutated = _ASSESS.replace(
    "    c = phase_c(final)",
    '    c = read_scalars(find_field(final, "c"))',
    1,
)
expect("the c-read check is not vacuous",
       _mutated != _ASSESS and bool(_c_reads_outside_the_helpers(_mutated)),
       "nothing was mutated" if _mutated == _ASSESS else "")

# The reconstruction's arithmetic, against the numbers measured on em45.
# c = 2q - 1 and c = 1 - 2*(thetaC + thetaH)/theta_ref must be the same map.
_theta_ref = 0.45
for _c in (-1.0, -0.999, 0.0, 0.5, 1.0):
    _q = (_c + 1.0) / 2.0
    _theta = _theta_ref * (1.0 - _q)
    expect(f"c={_c:g} survives the round trip through q and theta",
           abs((2.0 * _q - 1.0) - _c) < 1e-15
           and abs((1.0 - 2.0 * (_theta + 0.0) / _theta_ref) - _c) < 1e-15)

print()
if failures:
    print("CONTENT PARSER CHECK FAILED: " + ", ".join(failures))
    raise SystemExit(1)
print("All content-record parser discriminations passed.")
