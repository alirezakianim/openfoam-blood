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

S21 (research/designs/2026-10-05-S21-clean-solver.md; docket 20261005-1621,
reports 06, 09, 11) built networkRupture in at fibreRelease, damageTransport
at contentPartition and plateletOuterTransactions at false, and deleted the
other values' code. The assertions that pinned that deleted code are removed,
each at its site, with the hunk that deleted it; what S15 still owns on the
kept code is pinned in its kept form: the label is thetaH/theta, synced after
the phase solve, never transported or clipped under contentComposition; the
damage law is source 32 Eq. 6 on the stretch; every rupture consequence is
taken unconditionally; the kept validations and defaults; the log keys.

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
        # S21: 2 -> 1. The second site was the opt-in platelet outer
        # transaction (`else if (plateletContentModel_ &&
        # plateletOuterTransactions_)`, release-4198bb17 .C:17028-17039),
        # deleted by report 06 hunk a40 (plateletOuterTransactions built in
        # at false).
        s.count("solvePhaseField();syncMaterialLabelFromContent();updateDerivedFields();") == 1
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
    # S21: S15's own progressive loop is deleted. Its `progressive` local
    # (`networkRupture_ == "progressiveDamage"`) became the constant false
    # (report 09 hunk d37), and the chain elements it guarded, `else if
    # (progressive)` and `else if (b1 >= b1c && rI[celli] < 1.0)`, were
    # deleted as never reached (report 11 hunks p11.1, p11.2; their
    # irreversibility and no-be-reset clauses went with them, p15.1). Eq. 6
    # survives as the law of the kept fibreRelease branch, pinned here.
    return (
        "constscalarl1=sqrt(b1);{if(l1>=lc){dsLaw=1.0;}"
        "elseif(l1>lf){dsLaw=1.0-exp(-(l1-lf)/(2.0*dlf));}}" in b
        and "reduce(damageStepMax,maxOp<scalar>());" in b
    )


def damage_shares_every_rupture_consequence(src: str) -> bool:
    s = flat(src)
    # S21 (networkRupture built in at fibreRelease, report 09): each
    # consequence is now taken unconditionally. Deleted with their guards:
    # intactFraction's rupture-off return (hunk a23), advectRuptureLabel(),
    # a no-op under contentPartition (b, a24, and its call b11), the
    # noMemory guard (d59), the rIntact guard (d80) and the ruptured_
    # NO_WRITE branch (a11).
    return (
        'returnvolScalarField::New("intactFraction",1.0-ruptured_);' in s
        and 'volScalarFieldnoMemory("noNetworkMemory",absent);{noMemory=max('
            '1.0-thetaNetRef_*(1.0-ruptured_)' in s
        and "scalarrIntact=r;{rIntact=r*(1.0-ruptured_[celli]);" in s
        and "ruptured_.writeOpt()=IOobject::NO_WRITE" not in s
        and "advectRuptureLabel" not in s
    )


def switches_are_validated(src: str) -> bool:
    s = flat(src)
    # S21: the networkRupture value refusal is deleted (report 09 hunk c08);
    # the onset check lost its `networkRupture_ == "progressiveDamage"` guard
    # (d24) and now runs unconditionally; the mixed guard
    # `!(contentModel_ && coreShell_)` reduced to `!coreShell_` (report 06
    # hunk d07, design section 2 (d), .C:2920).
    return (
        'if(!(networkDamageOnsetStretch_>1&&networkDamageStretchScale_>0))' in s
        and 'materialLabelBasis_!="transportedLabel"&&materialLabelBasis_!="contentComposition"' in s
        and 'if(materialLabelBasis_=="contentComposition"&&!coreShell_){FatalErrorInFunction' in s
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
    ("sync missed after the step's phase solve", label_is_synced_after_every_content_change,
     "        solvePhaseField();\n        syncMaterialLabelFromContent();\n        updateDerivedFields();\n",
     "        solvePhaseField();\n        updateDerivedFields();\n"),
    ("label still transported under composition", transported_label_is_retired,
     'if (!coreShell_ || materialLabelBasis_ == "contentComposition")',
     "if (!coreShell_)"),
    ("composition clipped again", composition_is_not_clipped,
     '    volScalarField s("materialComposition", shell_);\n    if (materialLabelBasis_ != "contentComposition")\n    {',
     '    volScalarField s("materialComposition", shell_);\n    if (true)\n    {'),
    # S21: "damage reversible" and "progressive branch resets be" retired
    # with the S15 loop they mutated (report 11 hunks p11.1, p11.2); the
    # "relaxation only for the step law" and "label transport only for the
    # step law" anchors were the deleted guards (09 d59, a24).
    ("damage on b1, not the stretch", damage_is_kliuchnikov_eq6,
     "dsLaw = 1.0 - exp(-(l1 - lf)/(2.0*dlf));",
     "dsLaw = 1.0 - exp(-(b1 - lf)/(2.0*dlf));"),
    ("intact fraction blind to rupture", damage_shares_every_rupture_consequence,
     '"intactFraction", 1.0 - ruptured_)', '"intactFraction", 1.0 + 0*ruptured_)'),
    ("relaxation blind to rupture", damage_shares_every_rupture_consequence,
     "              - thetaNetRef_*(1.0 - ruptured_)\n", "              - thetaNetRef_\n"),
    ("deposit dilution blind to rupture", damage_shares_every_rupture_consequence,
     "                        rIntact = r*(1.0 - ruptured_[celli]);\n",
     "                        rIntact = r;\n"),
    ("composition allowed without contents", switches_are_validated,
     '        materialLabelBasis_ == "contentComposition"\n     && !coreShell_\n',
     '        materialLabelBasis_ == "contentComposition"\n     && false\n'),
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
        # S21: the generator's networkRupture value list and the
        # s15_combination_refusal()/`rupture` mirror went with the key
        # (built in at fibreRelease; the case-engineer's S21 generator cut).
        # The materialLabelBasis refusal it mirrored is kept, inline.
        g = self.gen
        self.assertIn('if child.get("material_label_basis") == "contentComposition" '
                      'and not core_shell:', g)
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

    # S21: test_the_new_children_differ_only_in_s15 retired. em59-em62 are
    # not among the kept children (design section 1.1); they stay in the
    # backup's controls.


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
