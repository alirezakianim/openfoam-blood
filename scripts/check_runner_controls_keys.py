#!/usr/bin/env python3
"""Verify every controls key a Colab RUNNER reads actually exists.

WHY THIS EXISTS. Twice a runner was derived from a sibling by substitution and
kept reading keys the new case does not define:

  - the floor probe's runner printed `loose_regression["measured_amplitude"]`,
    a key removed when that check was rewritten;
  - the factorial's runner printed `controls["floor_window_s"]` and a per-child
    `delta_t_s`, neither of which the factorial controls carry.

Both were KeyErrors positioned AFTER the run and AFTER the archive - an hour of
compute losing its summary to a print statement. Both were found by reading, and
the record noted twice that "the Colab runners' reporting blocks are covered by
no dry run, because the dry run drives the assessor and not the runner."

This closes that gap for the class of defect that actually occurred. It does NOT
verify the result.json keys a runner prints - those come from the assessor at
runtime - which is why runners also read every result key with `.get`.

WHAT IT CHECKS. For each `cases/*/scripts/run_*_colab.py`, it locates the
controls file the runner loads, then requires that every `controls.get("K")` and
`controls["K"]` names a top-level key of that file, and that every `X.get("K")`
where `X` iterates `controls["children"]` names a key present in EVERY child.

A runner whose controls file cannot be located is REPORTED, not skipped. A
checker that quietly passes over what it cannot parse reports success for the
files it never opened - the defect `check_notebook_control_keys.py` was itself
caught making.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def controls_path_for(tree: ast.AST, case: Path) -> Path | None:
    """Find the constant/*.json the runner loads.

    Matched on the FILENAME appearing anywhere in the source as a string, then
    resolved against the case's own constant/ directory. The path is built at
    runtime from Path objects joined with `/`, which is not worth constant
    folding for a name that has to exist on disk anyway.
    """
    names = {n.value for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and n.value.endswith(".json")}
    for name in sorted(names):
        candidate = case / "constant" / Path(name).name
        if candidate.is_file():
            return candidate
    return None


def iterates_controls_children(node: ast.For) -> bool:
    """`for X in controls['children']` / `controls.get('children', ...)` ONLY.

    Matched STRUCTURALLY on the `controls` name, not by looking for the word
    "children" anywhere in the iterator. Every one of these runners contains a
    SECOND loop, over `result.json`'s children, which conventionally binds the
    same variable name `child` and carries entirely different keys - the
    assessor's measurements rather than the case's declarations. A textual match
    conflates the two and reports every measurement as a missing declaration.
    """
    it = node.iter
    if isinstance(it, ast.Subscript) and isinstance(it.value, ast.Name):
        return (it.value.id == "controls" and isinstance(it.slice, ast.Constant)
                and it.slice.value == "children")
    if (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute)
            and it.func.attr == "get" and isinstance(it.func.value, ast.Name)
            and it.func.value.id == "controls" and it.args
            and isinstance(it.args[0], ast.Constant)):
        return it.args[0].value == "children"
    return False


def keys_read_from(scope: ast.AST, name: str) -> set:
    """String keys read from `name` by subscript or `.get`, within `scope`."""
    keys = set()
    for node in ast.walk(scope):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == name
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == name
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
    return keys


def collect(tree: ast.AST):
    """Keys read from `controls`, and from loop variables over its children.

    The child keys are collected from the BODY OF THE BINDING LOOP ONLY. Scoping
    is the whole correctness argument here: see `iterates_controls_children`.
    """
    controls_keys = keys_read_from(tree, "controls")
    child_keys = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and iterates_controls_children(node)):
            for statement in node.body:
                child_keys |= keys_read_from(statement, node.target.id)
    return controls_keys, child_keys


def main() -> int:
    problems, checked = [], 0
    runners = sorted(ROOT.glob("cases/*/scripts/run_*_colab.py"))
    for runner in runners:
        case = runner.parent.parent
        try:
            tree = ast.parse(runner.read_text())
        except SyntaxError as error:
            problems.append(f"{runner.relative_to(ROOT)}: does not parse -- {error}")
            continue

        controls_file = controls_path_for(tree, case)
        if controls_file is None:
            # A runner with no controls file of its own is legitimate - the older
            # gates read theirs from the extracted package by a path this checker
            # does not model. Report it as unchecked rather than count it.
            if re.search(r"constant\s*/\s*[\"']", runner.read_text()):
                problems.append(
                    f"{runner.relative_to(ROOT)}: reads a constant/ file this "
                    "checker could not resolve; it is NOT being checked")
            continue

        controls = json.loads(controls_file.read_text())
        controls_keys, child_keys = collect(tree)
        checked += 1

        for key in sorted(controls_keys):
            if key not in controls:
                problems.append(
                    f"{runner.relative_to(ROOT)}: reads controls[{key!r}], absent "
                    f"from {controls_file.relative_to(ROOT)}")

        children = controls.get("children")
        if child_keys and not isinstance(children, list):  # pragma: no cover
            problems.append(
                f"{runner.relative_to(ROOT)}: iterates controls['children'] but "
                f"{controls_file.relative_to(ROOT)} has no children list")
        elif child_keys:
            for key in sorted(child_keys):
                missing = [c.get("child_id", "?") for c in children if key not in c]
                if missing:
                    problems.append(
                        f"{runner.relative_to(ROOT)}: reads child[{key!r}], absent "
                        f"from child(ren) {', '.join(missing)}")

    if problems:
        print("runner controls-key check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"runner controls-key check: {checked} runner(s) verified against their "
          f"controls files, of {len(runners)} found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
