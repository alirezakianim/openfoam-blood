# Tests

All checks run offline, with no OpenFOAM installation. They cannot show that the
solver compiles against OpenFOAM, that `blockMesh`/`checkMesh` accept a mesh, or
that a step converges; only a real OpenFOAM build and run can. They check what
can be checked without one: the generated case files, the solver source text,
the header-only numerical kernels, and the assessment scripts.

Run everything (copies the tree into an empty temporary directory first, so a
file a check needs but the repository lacks fails here):

```
python3 tools/sync_from_workspace.py --check-only          # fast tier
python3 tools/sync_from_workspace.py --check-only --slow   # + two mutation suites
python3 tools/sync_from_workspace.py --check-only --dry-run  # + main-case dry run (hours)
```

Each step is judged by its own exit status. `SUITESPARSE_ROOT` (default
`~/.cache/openfoam_blood/suitesparse`) must hold the SuiteSparse headers and
libraries for the S19 check. A C++14 compiler is needed for the kernel tests.

## Case generation and staging

| Check | What it guards |
| --- | --- |
| `cases/AM2020_planar_mural_detachment/scripts/create_cases.py` | Generates all 219 children. The generator refuses configurations below its resolution floors. |
| `scripts/check_runner_staging.py <case>` | Copies `cases/` the way the Colab runner does and generates every child from that tree. |
| `scripts/check_case_boundaries.py <case>` | Every field in `0/` gives every mesh patch a boundary condition that exists. |
| `scripts/check_fvschemes_coverage.py [--self-test] <case>` | The solver cannot ask for a divergence scheme no case defines. |
| `scripts/check_refused_dictionary_values.py [--self-test]` | Every live case constructs on the current solver; refused values are not written; the archive registry is exact. |
| `scripts/check_throat_mesh.py` | Synthesises the points `blockMesh` would write, bends them with the throat profile, checks cell volumes, inversion, skew and dt; each negative is seen to fail. |
| `scripts/check_notebook_batch_selector.py` | The notebook's `BATCH` is the newest batch and every child it names exists. |

## Solver text, kernels and algebra

| Check | What it guards |
| --- | --- |
| `scripts/check_s19_block_coupled.py` | The block-coupled `(U, Vs, p)` step: compiles and mutates `blockCoupledKernel.H` (needs SuiteSparse); each source mutation must break its pin. |
| `scripts/s16_partition_kernel_test.cpp`, `s17_fibre_kernel_test.cpp` | Header-only damage-partition and fibre-release kernels, compiled from the solver's own headers. |
| `scripts/s18_fpe_kernel_test.cpp` | Network-pressure kernel with floating-point traps on; the `unfixed` variant must trap, otherwise the pass proves nothing. |
| `scripts/check_neohookean_algebra.py`, `check_b_transport_algebra.py` | Neo-Hookean stress and elastic-strain transport algebra, by finite difference, each with a self-test. |
| `scripts/check_zero_diffusivity_safety.py` | Every use of `D_` is safe when `D == 0`. |
| `scripts/check_network_pressure_law.py` | The network compaction law. |
| `scripts/check_s12_intrinsic_drag.py`, `check_s14_network_rupture.py`, `check_s15_composition_label_and_damage.py` | Wiring guards for intrinsic occupancy drag, network rupture, composition label and progressive damage. |
| `scripts/check_phase_indicator_basis.py`, `check_phase_normal_basis.py`, `check_phase_weight_basis_split.py`, `check_indicator_recentred_in_mag.py` | The phase-representation migration is complete and consistent. `check_content_consumer_map.py` is a helper imported by the indicator-basis check; run on its own it needs a design document this repository does not ship. |
| `scripts/check_platelet_*`, `check_stage2_*`, `check_stage3_gate_reaches_children.py` | Platelet carrier, bind localisation, content seeding, stage-3 gate wiring. |
| `scripts/check_dictionary_read_forms.py`, `check_registry_lookups_resolve.py`, `check_per_step_state_is_primed.py`, `check_field_dimensions.py`, `check_scheme_coverage.py`, `check_runner_controls_keys.py`, `check_content_record_parser.py` | Source-level consistency of dictionary reads, field names, dimensions, schemes, runner keys and the log parser. |
| `scripts/s16_offline_checks.py self-test`, `s17_offline_checks.py self-test` | Self-tests of the offline analysers. Their other modes read a returned run, which the repository does not ship. |

## Main-case dry run

`cases/AM2020_planar_mural_detachment/scripts/dry_run.py` drives `assess_case.py`
end to end on synthesised OpenFOAM-shaped output (219 children per scenario) and
checks that the assessor rejects what it should reject. It takes hours, so it
is not in the fast tier.

## Not tested here

Compilation of the solver against OpenFOAM, `blockMesh`, `checkMesh`, and any
numerical result. Those need an OpenFOAM 13 environment (see the README).
