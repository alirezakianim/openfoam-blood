#!/usr/bin/env python3
"""The Stage 2 content seeder must reproduce the solver's own seeding.

WHY THIS EXISTS. Run 20260915-062201 died at solver construction with
"Incomplete persistent content restart at time 0" because the case supplied
0/nP and not 0/thetaC + 0/thetaH. The restart contract is ALL-OR-NONE
(`.C:1711-1725`), and "none" means legacy seeding with ZERO free platelets --
which deposits nothing, so it is not a Stage 2 run at all.

The seeder fills the gap, and it must reproduce what the solver would have
seeded itself (`.C:1753-1770`), or the Stage 2 child starts from a different
clot than em27/em28 and nothing is comparable:

    theta0 = theta_ref * max(0.5*(1 - c), 0)      since q = 0.5*(1 + c)
    thetaH = theta0 * clamp(shell, 0, 1)
    thetaC = theta0 * (1 - clamp(shell, 0, 1))

Run: python3 scripts/check_stage2_content_seeding.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDER = ROOT / "cases/AM2020_planar_mural_detachment/scripts/seed_stage2_content.py"
spec = importlib.util.spec_from_file_location("seed_stage2", SEEDER)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

# S21: contentModel and plateletContentModel are built in, so a generated dictionary carries
# neither (the solver refuses both by name); the seeder needs only theta_ref.
PROPS = """skeletonVolumeFraction 0.45;
"""


def field(values, name="c"):
    body = "\n".join(repr(float(v)) for v in values)
    return (f'FoamFile\n{{\n    object {name};\n}}\n'
            "dimensions [0 0 0 0 0 0 0];\n"
            f"internalField nonuniform List<scalar>\n{len(values)}\n(\n{body}\n)\n;\n"
            "boundaryField\n{\n    inlet { type zeroGradient; }\n}\n")


def build(tmp, c, shell=None, props=PROPS, with_nP=True):
    case = Path(tmp) / "case"
    (case / "0").mkdir(parents=True)
    (case / "constant").mkdir(parents=True)
    (case / "constant" / "phaseClotProperties").write_text(props)
    (case / "0" / "c").write_text(field(c, "c"))
    if shell is not None:
        (case / "0" / "shell").write_text(field(shell, "shell"))
    if with_nP:
        (case / "0" / "nP").write_text(field([1.0] * len(c), "nP"))
    return case


class SeedingTests(unittest.TestCase):
    def seed(self, **kw):
        self.tmp = tempfile.TemporaryDirectory()
        case = build(self.tmp.name, **kw)
        S.main.__globals__["sys"].argv = ["seed", str(case)]
        S.main()
        return case

    def values(self, case, name):
        return S.read_internal(case / "0" / name)

    def test_it_reproduces_the_solver_seeding_formula(self):
        c = [-1.0, -0.5, 0.0, 0.5, 1.0]
        shell = [0.0, 0.25, 0.5, 1.0, 1.0]
        case = self.seed(c=c, shell=shell)
        tc, th = self.values(case, "thetaC"), self.values(case, "thetaH")
        for i, (ci, si) in enumerate(zip(c, shell)):
            theta0 = 0.45 * max(0.5 * (1 - ci), 0.0)
            self.assertAlmostEqual(th[i], theta0 * si, places=15)
            self.assertAlmostEqual(tc[i], theta0 * (1 - si), places=15)

    def test_the_two_contents_sum_to_theta0(self):
        """Only the SUM enters the content model; the split follows shell."""
        c = [-1.0, -0.9, -0.2, 0.3, 1.0]
        shell = [0.1, 0.4, 0.6, 0.9, 0.0]
        case = self.seed(c=c, shell=shell)
        tc, th = self.values(case, "thetaC"), self.values(case, "thetaH")
        for i, ci in enumerate(c):
            self.assertAlmostEqual(tc[i] + th[i],
                                   0.45 * max(0.5 * (1 - ci), 0.0), places=15)

    def test_c_above_one_gives_no_negative_content(self):
        """c > 1 is pure fluid; max(...,0) must clip, not go negative."""
        case = self.seed(c=[1.5, 2.0], shell=[0.5, 0.5])
        for name in ("thetaC", "thetaH"):
            self.assertTrue(all(v >= 0 for v in self.values(case, name)))

    def test_shell_outside_zero_one_is_clamped(self):
        case = self.seed(c=[-1.0, -1.0], shell=[-3.0, 7.0])
        tc, th = self.values(case, "thetaC"), self.values(case, "thetaH")
        self.assertAlmostEqual(th[0], 0.0, places=15)      # clamped to 0
        self.assertAlmostEqual(tc[0], 0.45, places=15)
        self.assertAlmostEqual(th[1], 0.45, places=15)     # clamped to 1
        self.assertAlmostEqual(tc[1], 0.0, places=15)

    def test_a_missing_shell_means_all_core(self):
        case = self.seed(c=[-1.0, 0.0], shell=None)
        self.assertAlmostEqual(self.values(case, "thetaH")[0], 0.0, places=15)
        self.assertAlmostEqual(self.values(case, "thetaC")[0], 0.45, places=15)

    # ---- the refusals -------------------------------------------------

    def test_an_s21_dictionary_without_the_two_switches_is_seeded(self):
        """S21: no switch in the dictionary means seeding, never a no-op (the pre-S21 no-op
        would have left the solver's all-or-none restart without thetaC/thetaH)."""
        case = self.seed(c=[-1.0], shell=[0.0], props="skeletonVolumeFraction 0.45;\n")
        self.assertTrue((case / "0" / "thetaC").exists(), "an S21 dictionary was not seeded")
        self.assertAlmostEqual(self.values(case, "thetaC")[0] + self.values(case, "thetaH")[0],
                               0.45, places=15)

    def test_missing_nP_is_refused_rather_than_half_seeded(self):
        """Writing thetaC+thetaH without nP fails the contract the other way."""
        with tempfile.TemporaryDirectory() as tmp:
            case = build(tmp, c=[-1.0], shell=[0.0], with_nP=False)
            S.main.__globals__["sys"].argv = ["seed", str(case)]
            with self.assertRaises(SystemExit) as caught:
                S.main()
            self.assertIn("0/nP is missing", str(caught.exception))
            self.assertFalse((case / "0" / "thetaC").exists())

    def test_a_uniform_c_is_refused_not_silently_mishandled(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = build(tmp, c=[-1.0], shell=[0.0])
            (case / "0" / "c").write_text(
                "FoamFile\n{\n}\ndimensions [0 0 0 0 0 0 0];\n"
                "internalField uniform -1;\nboundaryField\n{\n}\n")
            S.main.__globals__["sys"].argv = ["seed", str(case)]
            with self.assertRaises(ValueError):
                S.main()

    def test_a_shell_of_the_wrong_length_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = build(tmp, c=[-1.0, -1.0], shell=[0.0])
            S.main.__globals__["sys"].argv = ["seed", str(case)]
            with self.assertRaises(SystemExit) as caught:
                S.main()
            self.assertIn("0/shell has", str(caught.exception))

    def test_a_dictionary_without_theta_ref_is_refused(self):
        """S21 replaces the retired switch-combination refusal: theta_ref is the one input
        the seeder reads from the dictionary, and it is never defaulted."""
        with tempfile.TemporaryDirectory() as tmp:
            case = build(tmp, c=[-1.0], shell=[0.0], props="G 4;\n")
            S.main.__globals__["sys"].argv = ["seed", str(case)]
            with self.assertRaises(SystemExit) as caught:
                S.main()
            self.assertIn("skeletonVolumeFraction is not declared", str(caught.exception))
            self.assertFalse((case / "0" / "thetaC").exists())

    # ---- the written file ---------------------------------------------

    def test_the_written_patches_are_planar_and_front_back_empty(self):
        case = self.seed(c=[-1.0], shell=[0.0])
        text = (case / "0" / "thetaC").read_text()
        for patch in ("inlet", "outlet", "bottom", "top"):
            self.assertIn(f"{patch} {{ type zeroGradient; }}", text)
        for patch in ("front", "back"):
            self.assertIn(f"{patch} {{ type empty; }}", text)
        self.assertNotIn("wedge", text)
        self.assertNotIn("axis", text)

    def test_the_seeder_runs_as_a_script(self):
        """It is invoked from Allrun, not imported."""
        with tempfile.TemporaryDirectory() as tmp:
            case = build(tmp, c=[-1.0, 0.0], shell=[0.0, 1.0])
            done = subprocess.run([sys.executable, str(SEEDER), str(case)],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("stage2 content seeded", done.stdout)
            self.assertTrue((case / "0" / "thetaH").is_file())


if __name__ == "__main__":
    print("STAGE 2 SEEDING GUARD; reproduces .C:1753-1770 or the child starts "
          "from a different clot.", flush=True)
    unittest.main(verbosity=2)
