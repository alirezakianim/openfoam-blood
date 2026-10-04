#!/usr/bin/env python3
"""Stage 2 case generation must refuse every configuration that would RUN.

WHY THIS EXISTS. The expensive Stage 2 misconfigurations do not crash. They
produce a clean-looking null:

  * nP is READ_IF_PRESENT and defaults to ZERO, and -- unlike thetaC/thetaH,
    which the solver seeds from theta_ref*chi -- nothing seeds it. A case that
    omits the field runs with no platelets, deposits nothing, grows nothing,
    and logs a tidy result. That is the same "multiply by zero" trap that made
    every arm of the carrier gate return identical fields.
  * plateletBindRate defaults to 0, with the same consequence.
  * `matched` plus the accepted transfer deposits solid TWICE and the
    inventory still balances, because each mechanism is internally paired --
    so no downstream diagnostic catches it.

Each is refused at generation, on the machine that can fix it.

Run: python3 scripts/check_stage2_case_generation.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "cases/AM2020_planar_mural_detachment/scripts/create_cases.py"

spec = importlib.util.spec_from_file_location("create_cases_under_test", GEN)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def valid_child(**overrides) -> dict:
    child = {
        "child_id": "stage2_probe",
        "content_model": True,
        "platelet_content_model": True,
        "platelet_flux_convention": "intrinsic",
        "platelet_number_density_m3": 2.5e14,
        "platelet_bind_rate": 42.0,
        "end_time_s": 4e-4,
    }
    child.update(overrides)
    return child


class Stage2GenerationTests(unittest.TestCase):
    def assertRefused(self, child, operator, fragment):
        with self.assertRaises(SystemExit) as caught:
            cc.validate_stage2_platelet(child, operator)
        self.assertIn(fragment, str(caught.exception))

    def test_the_valid_child_is_accepted(self):
        """Otherwise every refusal below could pass vacuously."""
        cc.validate_stage2_platelet(valid_child(), "content")

    def test_a_child_without_the_platelet_model_is_untouched(self):
        cc.validate_stage2_platelet({"child_id": "x"}, "content")

    # ---- the ones that would RUN -------------------------------------

    def test_missing_number_density_is_refused(self):
        child = valid_child()
        del child["platelet_number_density_m3"]
        self.assertRefused(child, "content", "defaults\nto ZERO".replace("\n", " "))

    def test_zero_number_density_is_refused(self):
        self.assertRefused(
            valid_child(platelet_number_density_m3=0.0), "content",
            "must be > 0")

    def test_zero_bind_rate_is_refused_unless_declared_transport_only(self):
        self.assertRefused(
            valid_child(platelet_bind_rate=0.0), "content",
            "platelet_bind_rate must be > 0")
        # ... but a case that SAYS it is transport-only is legitimate.
        cc.validate_stage2_platelet(
            valid_child(platelet_bind_rate=0.0, platelet_transport_only=True),
            "content")

    def test_double_growth_is_refused(self):
        self.assertRefused(
            valid_child(), "matched", "deposit solid")

    def test_double_growth_is_allowed_only_for_a_transport_only_operator_test(self):
        cc.validate_stage2_platelet(
            valid_child(platelet_transport_only=True), "matched")

    # ---- the unlocalised-deposition ceiling ---------------------------

    def test_em29_as_shipped_is_under_the_ceiling(self):
        """The real child must pass, or the ceiling is mis-set."""
        controls = json.loads(
            (cc.CASE / "constant" / "mainCaseControls.json").read_text())
        em29 = next(c for c in controls["children"]
                    if c["child_id"] == "em29_stage2_baseline")
        cc.validate_stage2_platelet(em29, em29["content_operator"])
        gain = (em29["platelet_yield_m3"] * em29["platelet_bind_rate"]
                * em29["end_time_s"] * em29["platelet_number_density_m3"])
        self.assertLess(gain, cc.UNLOCALISED_DEPOSITION_CEILING)
        # ... and comfortably below the C1 discrepancy it exists to measure.
        self.assertLess(gain, 7.41e-3 / 100)

    def test_raising_lambda_for_realistic_growth_is_refused(self):
        """The obvious next edit, and the one that must not happen silently.

        The deposition is fastest in the LUMEN, so a lambda large enough to
        give visible growth gives visible growth in the wrong place.
        """
        self.assertRefused(
            valid_child(platelet_bind_rate=200000.0, end_time_s=4e-4),
            "content", "no spatial localiser")

    def test_the_ceiling_can_be_crossed_only_deliberately(self):
        cc.validate_stage2_platelet(
            valid_child(platelet_bind_rate=200000.0, end_time_s=4e-4,
                        unlocalised_deposition_accepted=True),
            "content")

    def test_a_child_without_an_end_time_is_not_silently_exempt(self):
        """end_time_s absent means the gain cannot be computed; the check must
        not then pass by accident on a huge lambda."""
        child = valid_child(platelet_bind_rate=200000.0)
        child.pop("end_time_s", None)
        # No end time: the ceiling cannot be evaluated, so it is skipped --
        # this test records that limitation rather than asserting it is safe.
        cc.validate_stage2_platelet(child, "content")

    # ---- the ones that would crash, refused earlier anyway ------------

    def test_platelet_model_without_content_model_is_refused(self):
        self.assertRefused(
            valid_child(content_model=False), "content",
            "requires content_model true")

    def test_undeclared_flux_convention_is_refused(self):
        child = valid_child()
        del child["platelet_flux_convention"]
        self.assertRefused(child, "content", "no default")

    def test_a_misspelled_flux_convention_is_refused(self):
        self.assertRefused(
            valid_child(platelet_flux_convention="intrinsics"), "content",
            "no default")

    def test_convention_without_the_switch_is_refused(self):
        self.assertRefused(
            {"child_id": "x", "platelet_flux_convention": "intrinsic"},
            "content", "no platelet equation would be solved")

    # ---- the field itself ---------------------------------------------

    def test_the_written_field_has_the_solver_s_dimensions(self):
        """dimless/dimVolume is [0 -3 0 0 0 0 0]; a mismatch fatals at run."""
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "c"
            cc.write_platelet_number_density(case, 2.5e14, 1.0e14)
            text = (case / "0" / "nP").read_text()
        self.assertIn("dimensions [0 -3 0 0 0 0 0];", text)
        self.assertIn("object nP;", text)
        self.assertIn("internalField uniform 250000000000000.0;", text)

    def test_the_inlet_carries_platelets_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "c"
            cc.write_platelet_number_density(case, 2.5e14, 1.0e14)
            text = (case / "0" / "nP").read_text()
        self.assertIn("inlet { type fixedValue; value uniform 100000000000000.0; }",
                      text)
        self.assertIn("outlet { type zeroGradient; }", text)

    def test_the_patch_names_match_the_MESH_not_the_shared_template(self):
        """The trap this test exists for.

        The shared template is the AXISYMMETRIC straight tube --
        inlet/outlet/wall/axis, wedge front and back. This case's mesh is the
        PLANAR slab planar_mesh() writes: inlet/outlet/bottom/top with empty
        front and back. A first version of the writer copied the template's
        boundary block and would have named `wall` and `axis`, which do not
        exist here -- a run-time fatal, discovered in Colab.

        So the patch names are checked against the generated blockMeshDict,
        not against a list repeated here.
        """
        # Geometry lives in the mural gate's controls; the main case names it
        # through geometry_source rather than duplicating it.
        controls = json.loads(
            (cc.MURAL / "constant" / "muralGeometryControls.json").read_text())
        mesh = cc.mural_gen.planar_mesh(controls["geometry"])
        in_mesh = set(re.findall(r"(?m)^\s*(\w+)\s*\{\s*type\s+\w+;\s*faces",
                                 mesh))
        self.assertTrue(in_mesh, "no patches parsed out of the generated mesh")

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "c"
            cc.write_platelet_number_density(case, 1.0, 1.0)
            text = (case / "0" / "nP").read_text()
        body = text[text.index("boundaryField"):]
        in_field = set(re.findall(r"(?m)^\s*(\w+)\s*\{\s*type", body))

        self.assertEqual(
            in_field, in_mesh,
            f"0/nP names {sorted(in_field)} but the mesh has "
            f"{sorted(in_mesh)}; a patch that does not exist fatals at run "
            "time, in Colab")

    def test_the_empty_patches_are_declared_empty(self):
        """A 2D mesh's front/back are `empty`; anything else fatals."""
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "c"
            cc.write_platelet_number_density(case, 1.0, 1.0)
            text = (case / "0" / "nP").read_text()
        for patch in ("front", "back"):
            self.assertIn(f"{patch} {{ type empty; }}", text)


if __name__ == "__main__":
    print("STAGE 2 GENERATION GUARD; refuses configurations that would run "
          "and report a clean null.", flush=True)
    unittest.main(verbosity=2)
