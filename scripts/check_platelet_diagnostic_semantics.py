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
        matrix = ROOT / (
            "preflight/20260906-101812-am2020-main-case/"
            "case-after-run-AM2020_planar_mural_detachment/runs/matrix.json"
        )
        if not matrix.is_file():
            self.skipTest("returned merged child spec unavailable; parser tests still run")
        spec = json.loads(matrix.read_text())[0]
        controls = json.loads((ROOT / "cases/AM2020_planar_mural_detachment/"
                               "constant/mainCaseControls.json").read_text())
        with tempfile.TemporaryDirectory(prefix="platelet-semantics-") as tmp:
            case = Path(tmp)
            log = case / "log.phaseClotAxisymmetricFluid"
            for versioned in (False, True):
                fields = {**(VERSIONED if versioned else LEGACY), "WpDevMax": 1.4}
                log.write_text(line(fields))
                result = self.assessor.assess_child(case, spec, controls)
                self.assertEqual(result["wp_dev_peak"], 1.4)
                self.assertFalse(result["wp_gate_ok"])
                self.assertEqual(result["platelet_raw_metrics_available"], versioned)
                self.assertEqual(result["wp_raw_q_dev_peak"], 0 if versioned else None)
            self.assertEqual(self.assessor.WP_DEV_LIMIT, 0.25)
            self.assertEqual(self.assessor.C_BOUND_LIMIT, 0.05)

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
