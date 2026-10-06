#!/usr/bin/env python3
"""V21-8 (S20 V5): seed theta above theta_ref in the highest-theta cells, so q = 1 - theta/theta_ref < 0.

Runs after the initialiser and seed_stage2_content.py, only for a child whose
constant/seedTheta.json exists ({"theta", "cells", "theta_ref"}, written by create_cases.py).
The `cells` cells with the largest thetaC + thetaH are set to a total of `theta`, each keeping its
own thetaC:thetaH split. Refuses unless that many cells carry theta > 0 and every chosen cell is
strictly below `theta` (so the seed raises theta and q goes negative in exactly these cells).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_stage2_content import read_internal, write_scalar  # noqa: E402


def main() -> int:
    case = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    spec = json.loads((case / "constant" / "seedTheta.json").read_text())
    theta, n_seed, theta_ref = float(spec["theta"]), int(spec["cells"]), float(spec["theta_ref"])
    tc, th = read_internal(case / "0" / "thetaC"), read_internal(case / "0" / "thetaH")
    if len(tc) != len(th):
        raise SystemExit("thetaC and thetaH have different cell counts")
    total = [a + b for a, b in zip(tc, th)]
    ranked = sorted(range(len(total)), key=lambda i: (-total[i], i))
    chosen = ranked[:n_seed]
    if len(chosen) < n_seed or total[chosen[-1]] <= 0.0:
        raise SystemExit(f"fewer than {n_seed} cells carry theta > 0; the seed would not sit in the clot")
    if max(total[i] for i in chosen) >= theta:
        raise SystemExit(f"a chosen cell already has theta >= {theta}; the seed would not raise it")
    for i in chosen:
        f = theta / total[i]
        tc[i] *= f
        th[i] *= f
    write_scalar(case / "0" / "thetaC", "thetaC", tc)
    write_scalar(case / "0" / "thetaH", "thetaH", th)
    q = [1.0 - (a + b) / theta_ref for a, b in zip(tc, th)]
    print(f"seed_theta_peak: {n_seed} cells set to theta {theta} (q {1.0 - theta / theta_ref:.4f}); "
          f"min q now {min(q):.4f}, cells with q < 0: {sum(1 for x in q if x < 0.0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
