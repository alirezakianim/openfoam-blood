#!/usr/bin/env python3
"""Guard for S15 (2026-09-24): the core/shell label from the contents, and
progressive network damage.

research/designs/2026-09-24-S15-composition-label-and-progressive-damage.md.

- OI-34, root fix. Under `materialLabelBasis contentComposition`, `shell_` is
  not transported. It is the shell share of the two FCT-transported solid
  contents, `thetaH/(thetaC + thetaH)`. That is bounded in [0, 1] by
  construction (even in floating point), conserved, and moved exactly with
  the material. The label it replaces had drifted by up to 0.18–0.25 in
  ~220 core/shell-boundary cells (batch 54), moving k0p by up to 3x.
- OI-35, root fix. Under `networkRupture progressiveDamage`, the network's
  damage follows Kliuchnikov et al. 2025 (source 32) Eq. 6, the Kachanov law
  fitted to fibrin: `d = 1 - exp(-H(l1-lf)(l1-lf)/(2 dlf))`, `S = S0(1-d)`,
  irreversible. It replaces S14's all-or-nothing step, whose drag-limited
  recoil (~sigma/(h K), 0.12–0.21 m/s) exceeded G8.

Every predicate has a mutation that must make it FAIL.

Run: python3 scripts/check_s15_composition_label_and_damage.py
"""
from __future__ import annotations

import importlib.util
import math
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"
HEADER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.H"
GEN = ROOT / "cases/AM2020_planar_mural_detachment/scripts/create_cases.py"
ASSESS = ROOT / "cases/AM2020_planar_mural_detachment/scripts/assess_case.py"
CONTROLS = ROOT / "cases/AM2020_planar_mural_detachment/constant/mainCaseControls.json"

_spec = importlib.util.spec_from_file_location(
    "s12", ROOT / "scripts/check_s12_intrinsic_drag.py")
_s12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s12)
flat, body = _s12.flat, _s12.body


def label_is_the_composition(src: str) -> bool:
    b = flat(body(src, "phaseClotAxisymmetricFluid::syncMaterialLabelFromContent()"))
    return (
        'if(materialLabelBasis_!="contentComposition"){return;}' in b
        and "constscalartheta=tC[celli]+tH[celli];sI[celli]=theta>0?tH[celli]/theta:1.0;" in b
        and "if(!(sMin>=0&&sMax<=1)){FatalErrorInFunction" in b
        and b.rstrip("}").endswith("updateMaterialFields();")
    )


def label_is_synced_after_every_content_change(src: str) -> bool:
    s = flat(src)
    return (
        s.count("solvePhaseField();syncMaterialLabelFromContent();updateDerivedFields();") == 2
        and '<<"networkDamageStretchScale="<<networkDamageStretchScale_<<endl;syncMaterialLabelFromContent();' in s
    )


def transported_label_is_retired(src: str) -> bool:
    b = flat(body(src, "phaseClotAxisymmetricFluid::advectMaterialLabel"))
    return 'if(!coreShell_||materialLabelBasis_=="contentComposition"){return;}' in b


def composition_is_not_clipped(src: str) -> bool:
    b = flat(body(src, "phaseClotAxisymmetricFluid::updateMaterialFields()"))
    return ('volScalarFields("materialComposition",shell_);'
            'if(materialLabelBasis_!="contentComposition"){s=min(max(shell_,' in b)


def damage_is_kliuchnikov_eq6(src: str) -> bool:
    b = flat(body(src, "phaseClotAxisymmetricFluid::ruptureOverstretchedNetwork()"))
    # S16 (2026-09-24): the partition branch (damageTransport
    # contentPartition) comes first and has its own `else if (progressive)`;
    # the S15 loop is the one whose progressive branch starts from l1.
    i = b.find("if(progressive){constscalarl1=sqrt(b1);")
    j = b.find("elseif(b1>=b1c&&rI[celli]<1.0)")
    prog = b[i:j] if 0 <= i < j else ""
    return (
        'constboolprogressive=networkRupture_=="progressiveDamage";' in b
        and "constscalarl1=sqrt(b1);if(l1>lf){constscalard=1.0-exp(-(l1-lf)/(2.0*dlf));" in prog
        and "if(d>rI[celli]){damageStepMax=max(damageStepMax,d-rI[celli]);" in prog
        and "rI[celli]=d;" in prog
        and "symmTensor::I" not in prog          # no be reset: the intact part keeps its strain
        and "reduce(damageStepMax,maxOp<scalar>());" in b
    )


def damage_shares_every_rupture_consequence(src: str) -> bool:
    s = flat(src)
    return (
        'if(networkRupture_=="off"){returnvolScalarField::New("intactFraction",' in s
        # S16: under contentPartition d is D/theta of the carried partition,
        # so the label transport returns for that switch too.
        and flat(body(src, "phaseClotAxisymmetricFluid::advectRuptureLabel")).count(
            'if(networkRupture_=="off"||damageTransport_=="contentPartition"){return;}') == 1
        and 'if(networkRupture_!="off"){noMemory=max(' in s
        and 'scalarrIntact=r;if(networkRupture_!="off"){rIntact=' in s
        and 'if(networkRupture_=="off"){ruptured_.writeOpt()=IOobject::NO_WRITE;}' in s
    )


def switches_are_validated(src: str) -> bool:
    s = flat(src)
    return (
        'networkRupture_!="off"&&networkRupture_!="criticalStretch"&&networkRupture_!="progressiveDamage"' in s
        and 'networkRupture_=="progressiveDamage"&&!(networkDamageOnsetStretch_>1&&networkDamageStretchScale_>0)' in s
        and 'materialLabelBasis_!="transportedLabel"&&materialLabelBasis_!="contentComposition"' in s
        and 'materialLabelBasis_=="contentComposition"&&!(contentModel_&&coreShell_)' in s
        and '"networkDamageOnsetStretch",2.105' in s
        and '"networkDamageStretchScale",1.0' in s
        and '"materialLabelBasis","transportedLabel"' in s
        and '"PHASE_CLOT_S15_CONFIG"' in src
        and '" damageStepMax="' in src
    )


PREDICATES = [
    ("the label IS thetaH/theta", label_is_the_composition),
    ("synced after every content change", label_is_synced_after_every_content_change),
    ("transported label retired", transported_label_is_retired),
    ("composition not clipped", composition_is_not_clipped),
    ("damage is source 32 Eq. 6, irreversible, no be reset", damage_is_kliuchnikov_eq6),
    ("both rupture laws share every consequence", damage_shares_every_rupture_consequence),
    ("switches validated, defaults sourced, diagnostics logged", switches_are_validated),
]

MUTATIONS = [
    ("label from the core share", label_is_the_composition,
     "sI[celli] = theta > 0 ? tH[celli]/theta : 1.0;",
     "sI[celli] = theta > 0 ? tC[celli]/theta : 1.0;"),
    ("label clipped instead of checked", label_is_the_composition,
     "    if (!(sMin >= 0 && sMax <= 1))\n    {\n        FatalErrorInFunction",
     "    if (false)\n    {\n        FatalErrorInFunction"),
    ("sync missed after the transaction's second phase solve", label_is_synced_after_every_content_change,
     "        ++plateletOuter_;\n        solvePhaseField();\n        syncMaterialLabelFromContent();\n",
     "        ++plateletOuter_;\n        solvePhaseField();\n"),
    ("label still transported under composition", transported_label_is_retired,
     'if (!coreShell_ || materialLabelBasis_ == "contentComposition")',
     "if (!coreShell_)"),
    ("composition clipped again", composition_is_not_clipped,
     '    volScalarField s("materialComposition", shell_);\n    if (materialLabelBasis_ != "contentComposition")\n    {',
     '    volScalarField s("materialComposition", shell_);\n    if (true)\n    {'),
    ("damage on b1, not the stretch", damage_is_kliuchnikov_eq6,
     "                const scalar d = 1.0 - exp(-(l1 - lf)/(2.0*dlf));",
     "                const scalar d = 1.0 - exp(-(b1 - lf)/(2.0*dlf));"),
    ("damage reversible", damage_is_kliuchnikov_eq6,
     "                if (d > rI[celli])\n                {\n                    damageStepMax",
     "                if (true)\n                {\n                    damageStepMax"),
    ("progressive branch resets be", damage_is_kliuchnikov_eq6,
     "                    rI[celli] = d;\n",
     "                    rI[celli] = d;\n                    bI[celli] = symmTensor::I;\n"),
    ("relaxation only for the step law", damage_shares_every_rupture_consequence,
     '        if (networkRupture_ != "off")\n        {\n            noMemory = max',
     '        if (networkRupture_ == "criticalStretch")\n        {\n            noMemory = max'),
    ("label transport only for the step law", damage_shares_every_rupture_consequence,
     '    if (networkRupture_ == "off" || damageTransport_ == "contentPartition")\n    {\n        return;\n    }\n    fvScalarMatrix rupturedEqn',
     '    if (networkRupture_ != "criticalStretch" || damageTransport_ == "contentPartition")\n    {\n        return;\n    }\n    fvScalarMatrix rupturedEqn'),
    ("composition allowed without contents", switches_are_validated,
     '     && !(contentModel_ && coreShell_)', '     && false'),
    ("damage increment not logged", switches_are_validated,
     '<< " damageStepMax=" << damageStepMax', '<< " damageStepMx=" << damageStepMax'),
]


def kliuchnikov_d(l1, lf=2.105, dlf=1.0):
    return 0.0 if l1 <= lf else 1.0 - math.exp(-(l1 - lf)/(2.0*dlf))


class S15Properties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SOLVER.read_text()
        cls.hdr = HEADER.read_text()
        cls.gen = GEN.read_text()
        cls.assess = ASSESS.read_text()

    def test_every_property_holds(self):
        for name, pred in PREDICATES:
            with self.subTest(property=name):
                self.assertTrue(pred(self.src), f"S15 property broken: {name}")

    def test_the_declarations_carry_their_evidence(self):
        for decl in ("scalar networkDamageOnsetStretch_;", "scalar networkDamageStretchScale_;",
                     "word materialLabelBasis_;", "void syncMaterialLabelFromContent();"):
            self.assertIn(decl, self.hdr)
        self.assertIn("Kliuchnikov", self.hdr)
        self.assertIn("20260924-014332", self.hdr)

    def test_the_generator_mirrors_the_solver(self):
        g = self.gen
        self.assertIn('"off", "criticalStretch", "progressiveDamage"', g)
        self.assertIn("def s15_combination_refusal", g)
        self.assertIn('s15_combination_refusal(child, bool(spec.get("core_shell")))', g)
        self.assertIn('rupture = child.get("network_rupture", "off") != "off"', g)
        self.assertIn('"networkDamageOnsetStretch"', g)

    def test_kliuchnikov_values(self):
        """Eq. 6 with lf = 2.105, dlf = 1.0 (p. 353): no damage below the onset,
        monotone, below 1 for every finite stretch."""
        self.assertEqual(kliuchnikov_d(2.0), 0.0)
        self.assertEqual(kliuchnikov_d(2.105), 0.0)
        self.assertAlmostEqual(kliuchnikov_d(2.8), 0.29355, places=5)
        self.assertAlmostEqual(kliuchnikov_d(3.64), 0.5358, places=4)
        prev = 0.0
        for k in range(1, 4000):
            l1 = 1.0 + k*0.01
            d = kliuchnikov_d(l1)
            self.assertGreaterEqual(d, prev)
            self.assertLess(d, 1.0)
            prev = d

    def test_damaged_tangent_is_bounded_for_every_stretch(self):
        """(1 - d)(8/9)b1 <= 5.52 for all l1: the dt bound can never be
        outrun by stretching, which is what killed em58 (tangent form 1.25
        at b1 38.6)."""
        worst = max((1 - kliuchnikov_d(1 + k*0.001))*(8/9)*(1 + k*0.001)**2
                    for k in range(0, 60000))
        self.assertLess(worst, 5.52)
        self.assertGreater(worst, 5.4)

    def test_damaged_material_hardens_through_the_s14_range(self):
        """The (1 - d)-weighted uniaxial neo-Hookean nominal stress keeps
        rising until l1 ~ 4.1: no strain softening, hence no mesh-scale
        localisation, where S14 used to fire (2.8) or up to source 32's crack
        tip (3.64)."""
        def S(l1):
            return (1 - kliuchnikov_d(l1))*(l1**2 - 1/l1)
        for k in range(100, 395):
            l1 = k*0.01
            self.assertGreater(S(l1 + 0.01), S(l1))
        peak = max(range(100, 1000), key=lambda k: S(k*0.01))*0.01
        self.assertAlmostEqual(peak, 4.09, delta=0.03)

    def test_composition_is_bounded_in_floating_point(self):
        """thetaH/(thetaC + thetaH) with thetaC, thetaH >= 0 is in [0, 1]
        exactly: fl(a + b) >= b for a >= 0, so the quotient cannot exceed 1."""
        rnd = random.Random(20260924)
        pool = [0.0, 5e-324, 1e-310, 1e-300, 1e-17, 1e-3, 0.45, 1.0, 1e300]
        for _ in range(20000):
            a = rnd.choice(pool)*rnd.random() if rnd.random() < 0.5 else rnd.choice(pool)
            b = rnd.choice(pool)*rnd.random() if rnd.random() < 0.5 else rnd.choice(pool)
            if a + b > 0:
                s = b/(a + b)
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 1.0)

    def test_the_progressive_gate_fails_on_its_negative(self):
        spec = importlib.util.spec_from_file_location("assess_case", ASSESS)
        a = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(a)
        cfg = ("PHASE_CLOT_S14_CONFIG networkRupture=progressiveDamage "
               "networkCriticalStretch=2.8 elasticStabilityStiffness=tangent\n"
               "PHASE_CLOT_S15_CONFIG materialLabelBasis=contentComposition "
               "networkRupture=progressiveDamage networkDamageOnsetStretch=2.105 "
               "networkDamageStretchScale=1\n"
               "PHASE_CLOT_STRIPES elasticDiffusionNumberMax=0.2\n")
        row = ("PHASE_CLOT_RUPTURE time=0.002 criticalStretch=2.8 newlyRuptured=3 "
               "newlyRupturedBulk=0 rupturedContentFraction=1e-06 maxIntactStretch=2.9 "
               "maxRupturedStretch=2.9 rupturedMin=0 rupturedMax=0.3 damageStepMax={}\n")
        ok = a.s14_report(cfg + row.format(0.01))
        bad = a.s14_report(cfg + row.format(0.3))
        self.assertTrue(ok["g12_rupture_resolves_stretch"])
        self.assertFalse(bad["g12_rupture_resolves_stretch"])
        s15 = a.s15_report(cfg)
        self.assertTrue(s15["composition_label"])
        self.assertTrue(s15["progressive_damage"])

    def test_the_new_children_differ_only_in_s15(self):
        import json
        kids = {k["child_id"]: k for k in json.loads(CONTROLS.read_text())["children"]}
        for new, old in (("em61_G4_s15", "em59_G4_s14"), ("em62_G400_s15", "em60_G400_s14")):
            a, b = kids[old], kids[new]
            diff = sorted(k for k in set(a) | set(b)
                          if k not in ("expectation", "child_id", "_every_number_and_its_source")
                          and a.get(k) != b.get(k))
            self.assertEqual(diff, ["material_label_basis", "network_critical_stretch",
                                    "network_damage_onset_stretch",
                                    "network_damage_stretch_scale", "network_rupture"])


class S15Mutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SOLVER.read_text()

    def test_each_predicate_fires_on_its_mutation(self):
        for label, pred, old, new in MUTATIONS:
            with self.subTest(mutation=label):
                self.assertEqual(self.src.count(old), 1,
                                 f"{label}: mutation anchor not unique/absent")
                self.assertTrue(pred(self.src), f"{label}: baseline fails")
                self.assertFalse(pred(self.src.replace(old, new)),
                                 f"{label}: the guard did not catch it")


if __name__ == "__main__":
    unittest.main(verbosity=2)
