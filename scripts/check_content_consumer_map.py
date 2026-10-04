#!/usr/bin/env python3
"""The consumer map must enumerate every legacy-fraction consumer in the source.

WHY THIS EXISTS. research/designs/2026-09-14-content-consumer-map.md decides,
consumer by consumer, what each reader of the legacy solid fraction must read
under the content model. A map like that is wrong the moment someone adds a
sixth call site, and wrong silently: the migration proceeds against a list
that no longer describes the code, which is exactly how
`min_advance_cells`, `permeability`/`k0p` and the cost model's missing
staggered loop got through.

So the map is not trusted as prose. The call sites are re-derived FROM THE
SOLVER SOURCE here, and a site the map does not classify fails the build.

WHAT IT DOES NOT CHECK. Whether the classifications are scientifically right.
It checks that every consumer is accounted for and that the two elastic
weights stay distinct -- nothing more.

Run: python3 scripts/check_content_consumer_map.py
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "user-applications/phaseClotAxisymmetricFluid/phaseClotAxisymmetricFluid.C"
MAP = ROOT / "research/designs/2026-09-14-content-consumer-map.md"

LEXEME = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',
                    re.S)


def code_lines(text: str) -> list[str]:
    """Source with comments and string bodies removed, line numbers preserved.

    A consumer named only in a comment is not a consumer, and the solver's
    comments name `solidFraction()` a dozen times while explaining why the
    weights differ. Counting those would make the guard pass on prose.
    """
    stripped = LEXEME.sub(
        lambda m: "\n" * m.group().count("\n")
        if m.group().startswith(("//", "/*"))
        else '""' + "\n" * m.group().count("\n"),
        text,
    )
    return stripped.split("\n")


DEFINITION = re.compile(
    r"^(?:[\w:<>,&*\s]+\s)?(Foam::solvers::phaseClotAxisymmetricFluid::\w+)\s*\(")


def enclosing_functions(lines: list[str]) -> list[str]:
    """Owning member function for each line, by brace depth at column 0."""
    current, owners, depth = "<file scope>", [], 0
    for line in lines:
        match = DEFINITION.match(line)
        if match and depth == 0:
            current = match.group(1).split("::")[-1]
        owners.append(current)
        depth = max(0, depth + line.count("{") - line.count("}"))
    return owners


def call_sites(pattern: str) -> dict[str, list[int]]:
    """{function: [line, ...]} for every CODE occurrence of `pattern`."""
    lines = code_lines(SOLVER.read_text())
    owners = enclosing_functions(lines)
    found: dict[str, list[int]] = {}
    for number, line in enumerate(lines, 1):
        if re.search(pattern, line):
            found.setdefault(owners[number - 1], []).append(number)
    return found


def mapped_consumers() -> set[str]:
    """Function names the map's consumer table classifies.

    The table's rows carry the consumer in backticks in the second column.
    Parsed rather than restated so the two cannot drift.
    """
    rows = set()
    for line in MAP.read_text().split("\n"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        if not re.fullmatch(r"\d+", cells[0]):
            continue
        name = re.fullmatch(r"`(\w+)`", cells[1])
        if name:
            rows.add(name.group(1))
    return rows


def mapped_indicator_consumers() -> set[str]:
    """Function names the section 2b table classifies.

    Its rows are numbered i1, i2, ... precisely so the two inventories cannot
    be confused: a fraction consumer and an indicator consumer need different
    decisions, and a guard that pooled them would accept one in place of the
    other.
    """
    rows = set()
    for line in MAP.read_text().split("\n"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or not re.fullmatch(r"i\d+", cells[0]):
            continue
        name = re.fullmatch(r"`(\w+)`", cells[1])
        if name:
            rows.add(name.group(1))
    return rows


def decided_analogues() -> set[str]:
    """Row ids in section 8, which decides what each indicator site becomes.

    Section 2b says what KIND of thing each site needs; section 8 says what it
    actually becomes. An inventory without decisions is a list of open
    questions, and migrating against it would be guessing site by site.
    """
    ids = set()
    seen_section = False
    for line in MAP.read_text().split("\n"):
        if line.startswith("## 8."):
            seen_section = True
        elif line.startswith("## ") and seen_section:
            break
        elif seen_section and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells and re.fullmatch(r"i\d+", cells[0]):
                ids.add(cells[0])
    return ids


def inventoried_ids() -> set[str]:
    """Row ids in section 2b."""
    ids = set()
    for line in MAP.read_text().split("\n"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and re.fullmatch(r"i\d+", cells[0]) and len(cells) >= 5:
            ids.add(cells[0])
    return ids


def table_lines(doc: str, id_pattern: str) -> dict[str, set[int]]:
    """{function: {line, ...}} from a table whose row ids match `id_pattern`.

    The line column is cell 2. Only full rows (>= 5 cells) are read, so the
    section 8 decision table -- three cells, and full of prose that contains
    digits -- cannot be mistaken for an inventory row.
    """
    out: dict[str, set[int]] = {}
    for line in doc.split("\n"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or not re.fullmatch(id_pattern, cells[0]):
            continue
        name = re.fullmatch(r"`(\w+)`", cells[1])
        if not name or not re.fullmatch(r"[\d,\s]+", cells[2]):
            continue
        out.setdefault(name.group(1), set()).update(
            int(n) for n in re.findall(r"\d+", cells[2]))
    return out


# THE INDICATOR IS NOW TWO PATTERNS, not one. Since 2026-09-15 the migrated
# sites call phaseIndicator() -- which returns max(1-q,0) on "legacy" and
# theta/theta_ref on "content" -- while the sites deliberately NOT migrated
# still compute the raw form inline. Scanning only the raw form would make
# every migrated site vanish from the inventory, and the map would go green
# by losing the rows it exists to track.
INDICATOR = r"(?:max\s*\(\s*1\.0\s*-\s*q_|\bphaseIndicator\s*\()"

# Owners that are the accessor rather than a consumer of it. solidFraction()
# and phaseIndicator() each compute the thing they return; counting them would
# make every inventory one row too long and would let a real site hide behind
# the accessor's name.
INDICATOR_ACCESSORS = {"solidFraction", "phaseIndicator"}


def line_coverage_failures(doc: str, pattern: str, id_pattern: str,
                           exclude: set[str] | None = None) -> list[str]:
    """Per-function line-set disagreement between source and a map table.

    WHY PER-LINE AND NOT PER-FUNCTION. Enumerating function NAMES cannot see a
    function losing one of several sites. `correctPressureTwoVelocity` holds
    two indicator sites and `momentumPredictor` three; deleting either of the
    former's rows left the name still classified and the whole guard still
    green. Codex demonstrated exactly that on 2026-09-14. Comparing the line
    sets is what makes a dropped row -- or a row left behind when an edit
    shifts the file -- a failure.
    """
    skip = exclude if exclude is not None else {"solidFraction"}
    sites = {f: set(v) for f, v in call_sites(pattern).items()
             if f not in skip}
    mapped = table_lines(doc, id_pattern)
    problems = []
    for function in sorted(set(sites) | set(mapped)):
        found, listed = sites.get(function, set()), mapped.get(function, set())
        if found == listed:
            continue
        problems.append(
            f"{function}: source has {sorted(found)}, map lists "
            f"{sorted(listed)} (unmapped {sorted(found - listed)}, "
            f"stale {sorted(listed - found)})")
    return problems


def elastic_weight_problems(text: str) -> list[str]:
    """What is wrong with the two elastic weights in `text`, if anything."""
    lines = code_lines(text)
    owners = enclosing_functions(lines)
    flat = [line.replace(" ", "") for line in lines]

    def in_fn(fn: str, needle: str) -> bool:
        return any(needle in flat[i] and owners[i] == fn
                   for i in range(len(lines)))

    problems = []
    for fn in ("elasticStress", "skeletonFaceStress"):
        floored = any("skeletonSolidFloor_" in lines[i] and owners[i] == fn
                      for i in range(len(lines)))
        if not floored:
            problems.append(f"{fn} no longer floors the deviatoric weight "
                            f"at skeletonSolidFloor_")
        if in_fn(fn, "forceFraction()*gD") or in_fn(fn, "solidFraction()*gD"):
            problems.append(f"{fn}: the DEVIATORIC weight reads a fraction")
    if not in_fn("skeletonFaceStress", "phaseIndicator()"):
        problems.append("skeletonFaceStress no longer builds its weight from "
                        "phaseIndicator()")
    if not in_fn("elasticStress", "-networkPartialPressure()*symmTensor::I"):
        problems.append("elasticStress's isotropic term no longer goes "
                        "through networkPartialPressure(), so the selected "
                        "law is bypassed")
    # S16 (2026-09-24): the S11 body moved, unchanged, into
    # networkPartialPressureUndamaged(); networkPartialPressure() applies the
    # tensile damage to it (the identity unless networkPressureDamage tensile).
    if not in_fn("networkPartialPressureUndamaged",
                 "forceFraction()*networkPressure()"):
        problems.append("networkPartialPressureUndamaged's legacy branch no "
                        "longer reads forceFraction()*networkPressure()")
    if not in_fn("networkPartialPressure",
                 "returnnetworkPressureDamaged(networkPartialPressureUndamaged());"):
        problems.append("networkPartialPressure() no longer routes through "
                        "networkPartialPressureUndamaged()")
    return problems


class ConsumerMapTests(unittest.TestCase):
    def setUp(self):
        # TWO ACCESSORS SINCE 2026-09-15, and the map must cover both.
        #
        # forceFraction() returns solidFraction() under the legacy basis and
        # clamp(theta) under "content". It is the funnel for the network
        # pressure and the isotropic elastic term; solidFraction() is still
        # the funnel for everything else. Scanning only the first would let a
        # site migrate to the second and silently leave the map.
        self.sites = call_sites(r"\b(?:solidFraction|forceFraction)\s*\(")
        # solidFraction() is not a consumer of itself. forceFraction() IS
        # kept: its legacy fallback genuinely calls solidFraction(), so it is
        # a real consumer as well as an accessor, and the map carries a row
        # for it (row 0) saying which.
        self.consumers = {f: v for f, v in self.sites.items()
                          if f != "solidFraction"}

    def test_the_map_parses_at_all(self):
        """A map whose table stopped matching would otherwise pass vacuously."""
        self.assertGreaterEqual(len(mapped_consumers()), 5,
                                "the consumer table parsed to almost nothing; "
                                "the guard below would pass on an empty set")

    def test_every_source_consumer_is_classified(self):
        missing = sorted(set(self.consumers) - mapped_consumers())
        self.assertEqual(
            missing, [],
            "these call solidFraction() and the consumer map does not "
            f"classify them: {missing}. Add a row deciding what each must "
            "read under the content model before migrating anything.")

    def test_the_map_lists_no_consumer_that_is_gone(self):
        """A stale row is as misleading as a missing one."""
        extra = sorted(mapped_consumers() - set(self.consumers))
        self.assertEqual(
            extra, [],
            f"the map classifies {extra}, which no longer call "
            "solidFraction(); the migration would be planned against code "
            "that is not there")

    def test_every_indicator_consumer_is_classified(self):
        """chi = max(1-q, 0) is NOT the solid fraction, and has its own map.

        Eleven sites compute it inline and none scales it by theta_ref, so a
        solidFraction()-based inventory cannot see any of them. They are where
        a global "replace chi with theta" would silently rescale by 0.45.
        """
        sites = call_sites(INDICATOR)
        consumers = {f for f in sites if f not in INDICATOR_ACCESSORS}
        missing = sorted(consumers - mapped_indicator_consumers())
        self.assertEqual(
            missing, [],
            f"these compute the raw indicator and section 2b does not "
            f"classify them: {missing}. Each needs its own analogue -- a "
            "presence test, a normalised weight, or a floored support -- "
            "decided before anything is migrated.")

    def test_the_indicator_table_parses(self):
        self.assertGreaterEqual(
            len(mapped_indicator_consumers()), 7,
            "section 2b parsed to almost nothing; the guard above would pass "
            "on an empty set")

    def test_the_two_inventories_are_not_pooled(self):
        """A fraction consumer must not be satisfied by an indicator row.

        solidFraction() and inline chi need different decisions; accepting one
        table's row for the other's site is how a half-migrated model would
        look fully classified.
        """
        # S10: the network pressure's fraction read moved into
        # networkPressureArgument(); networkPressure() itself no longer calls
        # forceFraction(). The property -- that consumer is fraction-only and
        # section 2b must not claim it -- is unchanged; only its owner moved.
        only_fraction = {"networkPressureArgument", "updatePermeability",
                         "momentumPredictor"}
        self.assertTrue(only_fraction <= mapped_consumers())
        self.assertNotIn("networkPressureArgument",
                         mapped_indicator_consumers(),
                         "networkPressureArgument does not compute the raw "
                         "indicator; section 2b should not claim it")

    def test_an_inlined_SCALED_fraction_would_be_caught(self):
        """The one thing that WOULD be a hidden solidFraction() copy.

        chi*theta_ref computed outside solidFraction() is the fraction by
        another name, and neither inventory would classify it correctly.
        """
        lines = code_lines(SOLVER.read_text())
        owners = enclosing_functions(lines)
        joined = "\n".join(lines)
        stray = []
        for match in re.finditer(
                r"max\s*\(\s*1\.0\s*-\s*q_[^;]{0,200}?skeletonVolumeFraction_",
                joined, re.S):
            line = joined[:match.start()].count("\n") + 1
            if owners[line - 1] != "solidFraction":
                stray.append(line)
        self.assertEqual(
            stray, [],
            f"the scaled legacy fraction is computed inline at {stray} rather "
            "than through solidFraction(); neither inventory can see it")

    def test_every_inventoried_site_has_a_decided_analogue(self):
        """Section 5 step 0: nothing migrates until each site has one.

        Section 2b names the KIND of analogue each site needs; section 8 names
        the substitution. A site inventoried but undecided is an open question
        that a migration would answer by accident.
        """
        undecided = sorted(inventoried_ids() - decided_analogues())
        self.assertEqual(
            undecided, [],
            f"section 2b inventories {undecided} and section 8 does not "
            "decide what they become; migrating would guess at those sites")

    def test_the_decision_table_parses(self):
        self.assertGreaterEqual(
            len(decided_analogues()), 7,
            "section 8 parsed to almost nothing; the guard above would pass "
            "on an empty set")

    def test_every_indicator_site_line_is_mapped(self):
        """Site-level coverage, not function-level -- see line_coverage_failures."""
        problems = line_coverage_failures(
            MAP.read_text(), INDICATOR, r"i\d+", INDICATOR_ACCESSORS)
        self.assertEqual(
            problems, [],
            "section 2b's line numbers disagree with the source:\n  "
            + "\n  ".join(problems))

    def test_every_fraction_site_line_is_mapped(self):
        problems = line_coverage_failures(
            MAP.read_text(), r"\b(?:solidFraction|forceFraction)\s*\(",
            r"\d+")
        self.assertEqual(
            problems, [],
            "section 2's line numbers disagree with the source:\n  "
            + "\n  ".join(problems))

    def test_dropping_one_row_of_a_multi_site_function_is_caught(self):
        """The counterexample that passed before this guard existed.

        `correctPressureTwoVelocity` has two indicator sites, 9618 and 10075.
        Deleting either row from both tables left every other test green,
        because the function name was still classified by the surviving row.
        """
        doc = MAP.read_text()
        # S17 (2026-09-26): i5 (Kres's chiReg) moved into
        # skeletonResidualDrag(), a one-site function; i21 is the multi-site
        # function's other row, so the counterexample is kept.
        for dropped in ("i21", "i6"):
            mutated = "\n".join(line for line in doc.split("\n")
                                if not line.startswith(f"| {dropped} |"))
            self.assertNotEqual(
                mutated, doc, f"row {dropped} was not found to delete")
            problems = line_coverage_failures(
                mutated, INDICATOR, r"i\d+", INDICATOR_ACCESSORS)
            self.assertTrue(
                any("correctPressureTwoVelocity" in p for p in problems),
                f"deleting row {dropped} left the map passing; the guard is "
                "still counting function names rather than sites")

    def test_a_stale_line_number_is_caught(self):
        """An edit that shifts the file must not leave the map quietly wrong."""
        # The cited line is READ from the map, not hard-coded. It was
        # hard-coded until 2026-09-15, and when a solver edit shifted the
        # file the .replace() silently matched nothing -- the mutation was
        # never applied and the test failed for the wrong reason. A negative
        # test that stops mutating is worse than no negative test.
        raw = MAP.read_text()
        # S17 (2026-09-26): i5 moved into skeletonResidualDrag(); i6 is a
        # correctPressureTwoVelocity row, as i5 was.
        m = re.search(r"\| i6 \| `correctPressureTwoVelocity` \| (\d+) \|", raw)
        self.assertIsNotNone(m, "the i6 row this test mutates is gone")
        cited = int(m.group(1))
        doc = raw.replace(
            f"| i6 | `correctPressureTwoVelocity` | {cited} |",
            f"| i6 | `correctPressureTwoVelocity` | {cited - 1} |", 1)
        self.assertNotEqual(doc, raw, "the mutation did not apply")
        problems = line_coverage_failures(
            doc, INDICATOR, r"i\d+", INDICATOR_ACCESSORS)
        self.assertTrue(
            any("correctPressureTwoVelocity" in p for p in problems),
            "a one-line drift in a cited site was not detected")

    def test_the_indicator_site_count_matches_the_maps_prose(self):
        """The count is READ FROM THE MAP, not hard-coded here.

        It was hard-coded at (13, 8) and went stale the moment S6a started
        moving reads onto the accessor -- so the guard failed for months on
        a number in a different file, which is the same staleness it exists
        to catch, one level up. The map's own sentence is the claim; this
        asserts the source agrees with it, so updating one without the other
        still fails and neither can be updated by editing this file.
        """
        prose = MAP.read_text()
        m = re.search(
            r"\*\*[^*]*?inventory to\s+([a-z-]+(?:\s+[a-z-]+)?)\s+sites\s+across\s+"
            r"([a-z-]+(?:\s+[a-z-]+)?)\s+functions\*\*",
            prose, re.S)
        self.assertTrue(m, "section 2b no longer states its own inventory "
                           "count in the form the guard reads")
        words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
            "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
            "twenty": 20, "thirty": 30, "forty": 40,
        }

        def number(phrase: str) -> int:
            parts = [w for w in re.split(r"[\s-]+", phrase.strip()) if w]
            self.assertTrue(all(w in words for w in parts),
                            f"cannot read {phrase!r} as a number")
            return sum(words[w] for w in parts)

        claimed = (number(m.group(1)), number(m.group(2)))
        sites = {f: v for f, v in call_sites(INDICATOR).items()
                 if f not in INDICATOR_ACCESSORS}
        found = (sum(len(v) for v in sites.values()), len(sites))
        self.assertEqual(
            found, claimed,
            f"section 2b claims {claimed[0]} indicator sites across "
            f"{claimed[1]} functions; source has {found[0]} across "
            f"{found[1]}. Update the prose and the table together.")

    def test_the_two_elastic_weights_stay_distinct(self):
        """The trap in section 4, and it already cost run 20260905-191135.

        The deviatoric term is weighted by max(chi, skeletonSolidFloor) -- the
        raw indicator, floored, a numerical support. The isotropic term is
        weighted by a genuine volume fraction. Using one for both made the
        term 100x too strong in the clot and put a network pressure in the
        lumen.

        S11 (2026-09-23) moved the isotropic weight into
        networkPartialPressure(), whose legacy branch is exactly
        forceFraction()*networkPressure(), and added a SECOND builder of the
        deviatoric weight, skeletonFaceStress(), for the face stress. So the
        property is now: both deviatoric builders floor the indicator; the
        isotropic term in elasticStress goes through networkPartialPressure();
        its legacy branch reads the force accessor; and neither deviatoric
        builder reads a fraction accessor. Each clause has a mutation below
        that must be caught.
        """
        self.assertEqual(elastic_weight_problems(SOLVER.read_text()), [])
        # S16 (2026-09-24): the legacy branch lives in
        # networkPartialPressureUndamaged(), which networkPartialPressure()
        # wraps (elastic_weight_problems checks the wrapper).
        self.assertIn(
            "networkPartialPressureUndamaged", self.consumers,
            "networkPartialPressureUndamaged no longer takes its legacy "
            "isotropic weight from a fraction accessor")

    def test_each_elastic_weight_mutation_is_caught(self):
        """Write the negative test before trusting the guard."""
        src = SOLVER.read_text()
        mutations = {
            "floor dropped from skeletonFaceStress":
                ("dimensionedScalar(\"solidFloor\", dimless, skeletonSolidFloor_)\n"
                 "        );\n    }\n    recordStressWeight(w.primitiveField(), 0);",
                 "dimensionedScalar(\"solidFloor\", dimless, 0)\n"
                 "        );\n    }\n    recordStressWeight(w.primitiveField(), 0);"),
            "elasticStress bypasses networkPartialPressure":
                ("- networkPartialPressure()*symmTensor::I",
                 "- forceFraction()*networkPressure()*symmTensor::I"),
            "deviatoric face weight reads the force fraction":
                # S14: the weight is the INTACT network's occupancy,
                # phaseIndicator()*intactFraction() (exactly phaseIndicator()
                # with the rupture model off).
                ('volScalarField w("elasticSolidWeight", phaseIndicator()*intactFraction());',
                 'volScalarField w("elasticSolidWeight", forceFraction()*intactFraction());'),
            "legacy isotropic branch stops reading the force accessor":
                ('        "networkPartialPressure",\n        forceFraction()*networkPressure()',
                 '        "networkPartialPressure",\n        solidFraction()*networkPressure()'),
        }
        for label, (old, new) in mutations.items():
            with self.subTest(mutation=label):
                self.assertEqual(src.count(old), 1,
                                 f"mutation anchor for {label!r} not unique")
                self.assertNotEqual(
                    elastic_weight_problems(src.replace(old, new)), [],
                    f"the guard did not catch: {label}")

if __name__ == "__main__":
    print("SOURCE-DERIVED CONSUMER ENUMERATION; no claim about whether the "
          "classifications are scientifically right.", flush=True)
    unittest.main(verbosity=2)
