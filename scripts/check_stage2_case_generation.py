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
  * S21 (2026-10-05, docket 20261005-1621 report 15): the `matched`
    double-growth refusal, the flux-convention refusals, the content_model
    prerequisite, the no-platelet-model early return and the unlocalised-
    deposition ceiling were deleted from the generator with the switches they
    read (release-4198bb17 create_cases.py L2135-2159, L2161, L2198, L2247, L311): the
    convention is built in at intrinsic and the localisation at interface.

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
        # S21: content_model, platelet_content_model and
        # platelet_flux_convention are built in; the localisation is built in
        # at interface, which requires an adhesion speed (every kept child
        # carries 0.006045 m/s, constant/mainCaseControls.json).
        "platelet_adhesion_speed": 0.006045,
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

    # S21: test_a_child_without_the_platelet_model_is_untouched retired with
    # the generator's early return (release-4198bb17 create_cases.py L2135-2143).

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

    # S21: test_double_growth_is_refused and its transport-only twin retired
    # with the `matched` refusal (release-4198bb17 create_cases.py L2152-2159). The four
    # unlocalised-deposition-ceiling tests retired with the ceiling (L311,
    # L2247) and, for test_em29_as_shipped_is_under_the_ceiling, with the
    # retired child em29_stage2_baseline (not among the 18 kept children).

    def test_interface_without_adhesion_speed_is_refused(self):
        child = valid_child()
        del child["platelet_adhesion_speed"]
        self.assertRefused(child, "content", "requires platelet_adhesion_speed")

    def test_zero_adhesion_speed_is_refused(self):
        self.assertRefused(
            valid_child(platelet_adhesion_speed=0.0), "content",
            "deposits nothing anywhere")

    # ---- the ones that would crash, refused earlier anyway ------------

    # S21: the content_model prerequisite and the three flux-convention
    # refusals retired with their switches (release-4198bb17 create_cases.py L2145-2150, L2135-2143,
    # L2161).

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
