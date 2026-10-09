#!/usr/bin/env python3
"""Check versioned raw/effective-q reporting without changing legacy gates.

The returned Batch 25 emitter reports |Wp-max(abs(q),qFloor)|, not |Wp-q|.
For q=Wp=-0.7 these are 1.4 and zero. That is an algebraic instrument
discriminator, not a claim that the platelet PDE or coupled solver passes.
Schema 2 adds raw metrics; absence must remain unavailable, never zero.

Run: PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_platelet_diagnostic_semantics.py
No solver execution, package build, or edits to returned evidence.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

from check_platelet_fluid_carrier import strip_comments

ROOT = Path(__file__).resolve().parent.parent
ASSESSOR = (ROOT / "cases/AM2020_planar_mural_detachment/scripts/assess_case.py")
ARCHIVED_LOG = ROOT / (
    "preflight/20260908-061511-am2020-main-case/artifacts/main_case_v1/"
    "AM2020_planar_mural_detachment/auditable_evidence/"
    "em_2vel_carrier__log.phaseClotAxisymmetricFluid.gz"
)
LEGACY = {
    "time": 1e-6, "wMax": 1, "wMin": 1, "VnMagMax": 0,
    "wOverBound": 0, "WpDevMax": 0, "qMin": 0.7, "wDevBlood": 0,
}
VERSIONED = {
    **LEGACY, "metricSchema": 2, "WpRawQDevMax": 0,
    "rawQMin": 0.7, "wDevRawBlood": 0, "rawBloodCells": 1,
}
FIXTURE = ROOT / "scripts/fixtures/assess_child_platelet_gate.json"
CONTROLS = ROOT / "cases/AM2020_planar_mural_detachment/constant/mainCaseControls.json"
DRY0 = " dryFacesConducting=0"


def emitter_errors(source):
    """Narrow source guard, not a C++ runtime or positivity proof."""
    clean = re.sub(r"\s+", "", strip_comments(source))
    anchor = "constvolScalarFieldqEffDiag(max(mag(q_),qFloor_));"
    tag = 'Info<<"PHASE_CLOT_PLATELET"'
    if clean.count(anchor) != 1 or clean.count(tag) != 1:
        return ["missing or duplicate reporting block"]
    start, report = clean.index(anchor), clean.index(tag)
    end = clean.find("<<nl;", report)
    if not start < report < end:
        return ["malformed reporting block"]
    block = clean[start:end + len("<<nl;")]
    required = (
        "constscalarField&qIn=qEffDiag.primitiveField();",
        "constscalarField&rawQIn=q_.primitiveField();",
        "wpDevMax=max(wpDevMax,mag(WpIn[celli]-qIn[celli]));",
        "wpRawQDevMax=max(wpRawQDevMax,mag(WpIn[celli]-rawQIn[celli]));",
        "rawQMin=min(rawQMin,rawQIn[celli]);",
        "if(rawQIn[celli]>=0.5){++rawBloodCells;"
        "wDevRawBlood=max(wDevRawBlood,mag(wIn[celli]-1.0));}",
        "reduce(wpRawQDevMax,maxOp<scalar>());",
        "reduce(rawQMin,minOp<scalar>());",
        "reduce(wDevRawBlood,maxOp<scalar>());",
        "reduce(rawBloodCells,sumOp<label>());",
        '<<"metricSchema=2"',
        '<<"WpRawQDevMax="<<wpRawQDevMax',
        '<<"rawQMin="<<rawQMin',
        '<<"wDevRawBlood="<<wDevRawBlood',
        '<<"rawBloodCells="<<rawBloodCells',
    )
    return [part for part in required if part not in block]


def source_load():
    """Compile the current source; same-mtime .pyc files cannot affect a check."""
    module = types.ModuleType("platelet_semantics_assessor")
    module.__file__ = str(ASSESSOR)
    sys.modules[module.__name__] = module
    exec(compile(ASSESSOR.read_text(), str(ASSESSOR), "exec"), module.__dict__)
    return module


def line(fields):
    return "PHASE_CLOT_PLATELET " + " ".join(
        f"{key}={value}" for key, value in fields.items()
    ) + "\n"


def assess_fixture(assessor, platelet_fields, dry_suffix):
    """assess_child on the in-tree synthetic fixture with the given platelet records and FO suffix.

    The fixture child is ADMISSIBLE by construction, so a verdict that changes comes from the two platelet gates
    alone. Every placeholder must land (a template that silently kept its braces would test nothing)."""
    fx = json.loads(FIXTURE.read_text())
    log = fx["log_template"]
    for t in ("5e-4", "1e-3"):
        token = "{PLATELET_%s}" % t
        assert log.count(token) == 1, token
        log = log.replace(token, line({**platelet_fields, "time": t}).rstrip("\n"))
    assert log.count("{DRY}") == 2
    log = log.replace("{DRY}", dry_suffix)
    controls = json.loads(CONTROLS.read_text())
    with tempfile.TemporaryDirectory(prefix="platelet-fixture-") as tmp:
        case = Path(tmp)
        (case / "log.phaseClotAxisymmetricFluid").write_text(log)
        for rel, text in fx["files"].items():
            target = case / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        return assessor.assess_child(case, fx["spec"], controls)


# name, platelet record, FO suffix, expected (wp_gate_ok, dry gate ok, admissible)
FIXTURE_CASES = (
    ("baseline: schema 2, raw 0, dry faces 0", VERSIONED, DRY0, (True, True, True)),
    ("a BATCH 72 log carries no dry-face field: not failed for it", VERSIONED, "", (True, True, True)),
    ("legacy unversioned record, WpDevMax 0", LEGACY, "", (True, True, True)),
    ("raw platelet deviation 0.30 over the 0.25 screen", {**VERSIONED, "WpRawQDevMax": 0.30}, DRY0, (False, True, False)),
    ("legacy WpDevMax 1.4 over the screen", {**LEGACY, "WpDevMax": 1.4}, "", (False, True, False)),
    ("dryFacesConducting=1 (the smallest nonzero)", VERSIONED, " dryFacesConducting=1", (True, False, False)),
    ("dryFacesConducting=622", VERSIONED, " dryFacesConducting=622", (True, False, False)),
    ("a malformed dryFacesConducting fails closed", VERSIONED, " dryFacesConducting=oops", (True, False, False)),
    ("both gates fail", {**VERSIONED, "WpRawQDevMax": 0.30}, " dryFacesConducting=3", (False, False, False)),
)


def fixture_failures(assessor):
    """Every FIXTURE_CASES row whose verdicts differ from the expectation; [] when the assessor agrees with all."""
    bad = []
    for name, fields, dry, want in FIXTURE_CASES:
        try:
            result = assess_fixture(assessor, fields, dry)
            got = (result["wp_gate_ok"], result.get("platelet_dry_faces_gate_ok"), result["admissible"])
        except Exception as error:  # an assessor that crashes on the fixture disagrees with it too
            got = repr(error)
        if got != want:
            bad.append((name, got, want))
    return bad


class PlateletSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assessor = source_load()

    def summary(self, text):
        return self.assessor.platelet_metric_summary(
            self.assessor.parse_platelet(text)
        )

    def assert_refused(self, text):
        with self.assertRaises(ValueError):
            self.assessor.parse_platelet(text)

    def test_complete_schema_two_names_references(self):
        parsed = self.assessor.parse_platelet(line(VERSIONED))
        self.assertEqual(set(VERSIONED), set(parsed[0]))
        out = self.summary(line(VERSIONED))
        self.assertEqual(out["platelet_metric_schema_status"], "SCHEMA_2_COMPLETE")
        self.assertTrue(out["platelet_raw_metrics_available"])
        self.assertEqual(out["wp_raw_q_dev_peak"], 0)
        self.assertEqual(out["platelet_raw_q_min"], 0.7)
        self.assertEqual(out["platelet_w_dev_raw_blood_peak"], 0)
        self.assertIn("max(abs(q), qFloor)", out["wp_dev_reference"])
        self.assertEqual(out["platelet_raw_blood_subset_coverage"], "ALL_RECORDS_NONEMPTY")
        self.assertEqual(set(out["platelet_raw_metric_definitions"]),
                         set(self.assessor.PLATELET_RAW_REQUIRED))

    def test_negative_q_is_not_raw_disagreement(self):
        q, wp, floor = -0.7, -0.7, 1e-6
        effective = max(abs(q), floor)
        self.assertEqual(abs(wp - q), 0)
        self.assertEqual(abs(wp - effective), 1.4)
        out = self.summary(line({
            **VERSIONED, "WpDevMax": abs(wp - effective),
            "WpRawQDevMax": abs(wp - q), "rawQMin": q,
            "qMin": effective, "rawBloodCells": 0,
        }))
        self.assertEqual(out["wp_raw_q_dev_peak"], 0)
        self.assertEqual(out["platelet_raw_q_min"], -0.7)
        self.assertIsNone(out["platelet_w_dev_raw_blood_peak"])
        self.assertEqual(out["platelet_raw_blood_subset_coverage"], "ALL_EMPTY")

    def test_every_required_schema_two_key_is_required_on_every_record(self):
        for key in self.assessor.PLATELET_V2_REQUIRED:
            with self.subTest(key=key):
                partial = {k: v for k, v in VERSIONED.items() if k != key}
                self.assert_refused(line(VERSIONED) + line(partial))

    def test_nonfinite_and_partial_numeric_tokens_are_refused(self):
        for key in self.assessor.PLATELET_V2_REQUIRED:
            for bad in ("nan", "inf", "-inf", "1e999", "0oops", "1e-"):
                with self.subTest(key=key, bad=bad):
                    self.assert_refused(line({**VERSIONED, key: bad}))

    def test_unknown_or_malformed_version_never_downgrades(self):
        for value in (0, 1, 3, 2.5, "nan", "inf", "-inf", "2oops", ""):
            with self.subTest(version=value):
                self.assert_refused(line({**VERSIONED, "metricSchema": value}))
        self.assert_refused(line(LEGACY).rstrip() + " metricSchema=\n")
        self.assert_refused(line(LEGACY).rstrip() + " metricSchema 2\n")
        self.assert_refused(line({k: v for k, v in VERSIONED.items()
                                  if k != "metricSchema"}))

    def test_duplicate_required_fields_are_refused(self):
        for key in ("metricSchema", *self.assessor.PLATELET_V2_REQUIRED):
            with self.subTest(key=key):
                self.assert_refused(line(VERSIONED).rstrip() + f" {key}=0\n")

    def test_raw_blood_count_is_a_nonnegative_integer(self):
        for bad in (-1, 0.5, -0.1):
            with self.subTest(count=bad):
                self.assert_refused(line({**VERSIONED, "rawBloodCells": bad}))
        self.assert_refused(line({**VERSIONED, "rawBloodCells": 0, "wDevRawBlood": 1}))

    def test_negative_magnitudes_and_invalid_legacy_counter_in_v2_refused(self):
        for key in ("WpDevMax", "wDevBlood", "VnMagMax", "qMin",
                    "WpRawQDevMax", "wDevRawBlood", "wOverBound"):
            with self.subTest(key=key):
                self.assert_refused(line({**VERSIONED, key: -0.1}))
        self.assert_refused(line({**VERSIONED, "wOverBound": 0.5}))

    def test_actual_cpp_emitter_wiring(self):
        source = (ROOT / "user-applications/phaseClotAxisymmetricFluid/"
                  "phaseClotAxisymmetricFluid.C").read_text()
        self.assertEqual(emitter_errors(source), [])
        # C++ block comments do not nest; wrapping an already-commented
        # source would end the fake comment early and leave real code.
        self.assertTrue(emitter_errors("//" + source.replace("\n", "\n//")))

    def test_raw_reference_emitter_mutations_are_rejected(self):
        source = (ROOT / "user-applications/phaseClotAxisymmetricFluid/"
                  "phaseClotAxisymmetricFluid.C").read_text()
        mutations = (
            ("raw pointer", "rawQIn = q_.primitiveField()",
             "rawQIn = qEffDiag.primitiveField()"),
            ("raw norm", "mag(WpIn[celli] - rawQIn[celli])",
             "mag(WpIn[celli] - qIn[celli])"),
            ("blood subset", "if (rawQIn[celli] >= 0.5)", "if (qIn[celli] >= 0.5)"),
            ("blood count", "++rawBloodCells;", ""),
            ("global norm", "reduce(wpRawQDevMax, maxOp<scalar>());", ""),
            ("global minimum", "reduce(rawQMin, minOp<scalar>());", ""),
            ("global blood norm", "reduce(wDevRawBlood, maxOp<scalar>());", ""),
            ("global count", "reduce(rawBloodCells, sumOp<label>());", ""),
        )
        for name, before, after in mutations:
            with self.subTest(mutation=name):
                self.assertEqual(source.count(before), 1, "mutation must land")
                self.assertTrue(emitter_errors(source.replace(before, after, 1)))

    def test_nonempty_subset_and_empty_subset_are_distinguished(self):
        empty = {**VERSIONED, "rawBloodCells": 0, "rawQMin": -0.7}
        full = {**VERSIONED, "time": 2e-6, "rawBloodCells": 3, "wDevRawBlood": 0.2}
        out = self.summary(line(empty) + line(full))
        self.assertEqual(out["platelet_w_dev_raw_blood_peak"], 0.2)
        self.assertEqual(out["platelet_raw_blood_empty_records"], 1)
        self.assertEqual(out["platelet_raw_blood_records_with_cells"], 1)
        self.assertEqual(out["platelet_raw_blood_cells_min"], 0)
        self.assertEqual(out["platelet_raw_blood_cells_max"], 3)
        self.assertEqual(out["platelet_raw_blood_subset_coverage"], "SOME_EMPTY_RECORDS")

    def test_legacy_raw_metrics_are_unavailable_not_zero(self):
        parsed = self.assessor.parse_platelet(line(LEGACY))
        self.assertEqual(parsed, [LEGACY])
        out = self.summary(line(LEGACY))
        self.assertEqual(out["platelet_metric_schema_status"], "LEGACY_UNVERSIONED")
        self.assertFalse(out["platelet_raw_metrics_available"])
        for key in ("wp_raw_q_dev_peak", "platelet_raw_q_min",
                    "platelet_w_dev_raw_blood_peak", "platelet_raw_metric_definitions"):
            self.assertIsNone(out[key])
        self.assertIn("requires", out["wp_dev_reference"])

    def test_partial_upgrade_does_not_become_complete_raw_history(self):
        out = self.summary(line(LEGACY) + line(VERSIONED))
        self.assertEqual(out["platelet_metric_schema_status"], "MIXED_SCHEMAS")
        self.assertEqual(out["platelet_metric_schema2_records"], 1)
        self.assertEqual(out["platelet_metric_legacy_records"], 1)
        self.assertFalse(out["platelet_raw_metrics_available"])
        self.assertIsNone(out["wp_raw_q_dev_peak"])
        self.assertIsNone(out["platelet_raw_q_min"])

    def test_empty_history_is_not_zero(self):
        out = self.summary("")
        self.assertEqual(out["platelet_metric_schema_status"], "NO_RECORDS")
        self.assertIsNone(out["wp_raw_q_dev_peak"])
        self.assertFalse(out["platelet_raw_metrics_available"])

    def test_assess_child_consumes_summary_without_replacing_existing_gate(self):
        # Verification 08, hole A3: this test used to SKIP (its returned-run fixture preflight/20260906-101812 is absent),
        # so a mutant that discarded the platelet gate's verdict passed the build. It now runs on the in-tree fixture.
        self.assertTrue(FIXTURE.is_file(), f"the in-tree fixture {FIXTURE} is missing: the end-to-end test must not skip")
        for versioned in (False, True):
            fields = {**(VERSIONED if versioned else LEGACY), "WpDevMax": 1.4}
            result = assess_fixture(self.assessor, fields, DRY0 if versioned else "")
            self.assertEqual(result["wp_dev_peak"], 1.4)
            # OI-61: WpDevMax 1.4 with WpRawQDevMax 0 (the q < 0 inflation) passes on a schema-2 log;
            # the same WpDevMax on a legacy log still fails the historical screen.
            self.assertEqual(result["wp_gate_ok"], versioned)
            self.assertEqual(result["wp_gate_metric"], "WpRawQDevMax" if versioned else "WpDevMax")
            self.assertEqual(result["wp_gate_peak"], 0 if versioned else 1.4)
            self.assertEqual(result["platelet_raw_metrics_available"], versioned)
            self.assertEqual(result["wp_raw_q_dev_peak"], 0 if versioned else None)
            # the fixture child is admissible but for the platelet screen, so the verdict follows it end to end
            self.assertEqual(result["admissible"], versioned)
            if not versioned:
                self.assertEqual(result["bounds_reason"], result["wp_gate_reason"])
        self.assertEqual(self.assessor.WP_DEV_LIMIT, 0.25)
        self.assertEqual(self.assessor.C_BOUND_LIMIT, 0.05)

    def test_oi62_assess_child_verdicts_on_the_fixture(self):
        self.assertEqual(fixture_failures(self.assessor), [])

    def mutant(self, before, after):
        src = ASSESSOR.read_text()
        self.assertEqual(src.count(before), 1, f"mutation must land exactly once: {before!r}")
        module = types.ModuleType("platelet_semantics_mutant")
        module.__file__ = str(ASSESSOR)
        exec(compile(src.replace(before, after, 1), str(ASSESSOR), "exec"), module.__dict__)
        return module

    def test_oi62_discarding_a_platelet_verdict_fails_the_fixture_test(self):
        """A3 and its dry-face twins: each mutant must make fixture_failures non-empty; the no-op control must not."""
        call_tail = "    # OI-62: the dry-face conductance count"
        mutants = (
            ("A3: wp_ok = True after the gate call (verdict discarded)", call_tail, "    wp_ok = True\n" + call_tail),
            ("wp_ok dropped from admissible", "and wp_ok and dry_ok", "and dry_ok"),
            ("dry verdict discarded", 'dry_ok = dry["platelet_dry_faces_gate_ok"]', "dry_ok = True"),
            ("dry_ok dropped from admissible", "and wp_ok and dry_ok", "and wp_ok"),
            ("dry threshold n > 1 (a single dry face passes)", "if n > 0 and first_bad is None", "if n > 1 and first_bad is None"),
            ("the dryFacesConducting field is never read", 'values = [v for k, v in pairs if k == "dryFacesConducting"]', "values = []"),
            ("a malformed count is ignored", 're.fullmatch(r"\\d+", values[0])', "True"),
        )
        for name, before, after in mutants:
            with self.subTest(mutant=name):
                self.assertTrue(fixture_failures(self.mutant(before, after)), "the mutant was not caught")
        self.assertEqual(fixture_failures(self.mutant("and wp_ok and dry_ok", "and wp_ok and dry_ok")), [], "the no-op control")

    def fo(self, *suffixes):
        return "".join(
            f"PHASE_CLOT_PLATELET_FO time={1e-6 * (i + 1):g} plateletFo=0.02 plateletFoLimit=2{s}\n"
            for i, s in enumerate(suffixes)
        )

    def test_oi62_dry_face_gate_reads_the_field_and_nothing_else(self):
        gate = self.assessor.platelet_dry_face_gate
        absent = gate(self.fo("", ""))
        self.assertEqual((absent["platelet_fo_records"], absent["platelet_dry_faces_records"],
                          absent["platelet_dry_faces_gate_ok"], absent["platelet_dry_faces_conducting_max"]),
                         (2, 0, True, None))
        zeros = gate(self.fo(DRY0, DRY0, DRY0))
        self.assertEqual((zeros["platelet_dry_faces_records"], zeros["platelet_dry_faces_gate_ok"],
                          zeros["platelet_dry_faces_conducting_max"]), (3, True, 0))
        self.assertIsNone(zeros["platelet_dry_faces_gate_reason"])
        # the first nonzero line sets the time and the reason; the maximum is over every line
        some = gate(self.fo(DRY0, " dryFacesConducting=3", " dryFacesConducting=1"))
        self.assertFalse(some["platelet_dry_faces_gate_ok"])
        self.assertEqual(some["platelet_dry_faces_conducting_max"], 3)
        self.assertEqual(some["platelet_dry_faces_first_time_s"], 2e-6)
        self.assertIn("dryFacesConducting=3", some["platelet_dry_faces_gate_reason"])
        self.assertFalse(gate(self.fo(" dryFacesConducting=1"))["platelet_dry_faces_gate_ok"])
        # the nonzero line is not the last: the gate reads every record, not the final one
        self.assertFalse(gate(self.fo(" dryFacesConducting=1", DRY0, DRY0))["platelet_dry_faces_gate_ok"])
        for bad in (" dryFacesConducting=oops", " dryFacesConducting=-1", " dryFacesConducting=1.5",
                    " dryFacesConducting=nan", " dryFacesConducting=1e3", " dryFacesConducting=",
                    " dryFacesConducting", " dryFacesConducting=0 dryFacesConducting=0"):
            with self.subTest(malformed=bad):
                out = gate(self.fo(DRY0, bad))
                self.assertFalse(out["platelet_dry_faces_gate_ok"])
                self.assertIn("malformed", out["platelet_dry_faces_gate_reason"])
        # sibling records that share the prefix are not PHASE_CLOT_PLATELET_FO lines
        other = ("PHASE_CLOT_PLATELET_FO_WARNING time=1e-6 plateletFo=5 dryFacesConducting=9\n"
                 + line({**VERSIONED}).rstrip("\n") + " dryFacesConducting=9\n")
        ignored = gate(other + self.fo(DRY0))
        self.assertEqual((ignored["platelet_fo_records"], ignored["platelet_dry_faces_gate_ok"]), (1, True))
        self.assertEqual(gate("")["platelet_fo_records"], 0)

    def gate(self, *records):
        """platelet_envelope_gate over parsed records, as assess_child feeds it."""
        plate = self.assessor.parse_platelet("".join(line(r) for r in records))
        summary = self.assessor.platelet_metric_summary(plate)
        return self.assessor.platelet_envelope_gate(plate, summary["platelet_raw_metrics_available"])

    def test_oi61_schema_two_gates_on_raw_q_both_ways(self):
        limit = self.assessor.WP_DEV_LIMIT
        self.assertEqual(limit, 0.25)
        # BATCH 72's S23v_G4_r3: WpDevMax 0.419 (q < 0 inflation), raw 0.040 -> admissible on the raw metric
        metric, peak, ok, why = self.gate({**VERSIONED, "WpDevMax": 0.419, "WpRawQDevMax": 0.040})
        self.assertEqual((metric, peak, ok, why), ("WpRawQDevMax", 0.040, True, None))
        # the other way: a real raw-q discrepancy with a small WpDevMax must FAIL (it could not before)
        metric, peak, ok, why = self.gate({**VERSIONED, "WpDevMax": 0.01, "WpRawQDevMax": 0.30})
        self.assertEqual((metric, peak, ok), ("WpRawQDevMax", 0.30, False))
        self.assertIn("WpRawQDevMax", why)
        self.assertIn("0.300", why)
        # the limit is the same 0.25, inclusive of the limit itself
        self.assertTrue(self.gate({**VERSIONED, "WpRawQDevMax": 0.25})[2])
        self.assertFalse(self.gate({**VERSIONED, "WpRawQDevMax": 0.2500001})[2])
        # the peak is over every record, not the last
        self.assertFalse(self.gate({**VERSIONED, "WpRawQDevMax": 0.4}, {**VERSIONED, "WpRawQDevMax": 0.0})[2])

    def test_oi61_legacy_and_mixed_histories_keep_wpdevmax(self):
        metric, peak, ok, why = self.gate({**LEGACY, "WpDevMax": 1.4})
        self.assertEqual((metric, peak, ok), ("WpDevMax", 1.4, False))
        self.assertIn("effective-q reference", why)
        self.assertTrue(self.gate({**LEGACY, "WpDevMax": 0.25})[2])
        self.assertFalse(self.gate({**LEGACY, "WpDevMax": 0.2500001})[2])
        # a legacy record alongside a schema-2 one: the history is MIXED, the legacy metric gates every record
        metric, peak, ok, _ = self.gate({**LEGACY, "WpDevMax": 0.1}, {**VERSIONED, "WpDevMax": 1.4, "WpRawQDevMax": 0})
        self.assertEqual((metric, peak, ok), ("WpDevMax", 1.4, False))
        # no records: fail closed
        metric, peak, ok, why = self.assessor.platelet_envelope_gate([], False)
        self.assertEqual((peak, ok), (None, False))
        self.assertIn("never measured", why)

    def test_oi61_assess_child_calls_the_gate_function(self):
        src = ASSESSOR.read_text()
        self.assertEqual(src.count("platelet_envelope_gate("), 2)          # the definition and the one call in assess_child
        self.assertIn('"wp_gate_metric": wp_gate_metric', src)
        self.assertNotIn('max((r["WpDevMax"] for r in plate)', src.split("def assess_child")[1].split("G3'lambda ADMISSIBILITY")[0].replace(
            'wp_dev_peak = max((r["WpDevMax"] for r in plate), default=None)', ""))

    def test_real_legacy_archive_is_preserved(self):
        if not ARCHIVED_LOG.is_file():
            self.skipTest("optional immutable legacy archive unavailable")
        blob = ARCHIVED_LOG.read_bytes()
        self.assertEqual(hashlib.sha256(blob).hexdigest(),
                         "461e3e35b4d75ba1384ad31ed14c26dd18c2bfc5b75ef23e483fe81fd6897fe3")
        rows = self.assessor.parse_platelet(gzip.decompress(blob).decode())
        self.assertEqual(len(rows), 20099)
        self.assertTrue(math.isclose(max(r["WpDevMax"] for r in rows), 1.27030322123))
        out = self.assessor.platelet_metric_summary(rows)
        self.assertEqual(out["platelet_metric_schema_status"], "LEGACY_UNVERSIONED")
        self.assertIsNone(out["wp_raw_q_dev_peak"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
