# openfoam-blood

A two-velocity, phase-field model of platelet-driven clot growth, poroelastic
deformation and detachment, written as an OpenFOAM-13 solver module, together
with the generator for its pre-registered case series and the offline checks
that guard it.

**Status: research code.** It is not a validated blood-flow model, no
physiological parameter is claimed to be calibrated, and the interfaces change
without notice. Nothing here has been used for clinical or design decisions.

## What it is

* `phaseClotAxisymmetricFluid`: an OpenFOAM-13 `foamRun` solver module
  (it derives from `incompressibleFluid`). It carries a fluid velocity, a
  skeleton (clot network) velocity, platelet transport and content, a
  finite-strain network with damage, release and rupture, and an implicit
  block-coupled `(U, Vs, p)` step solved with UMFPACK.
* `cases/AM2020_planar_mural_detachment`: the main case. A generator
  (`scripts/create_cases.py`) writes 219 children of a planar-channel mural-clot
  geometry; `constant/mainCaseControls.json` holds each child's pre-registered
  expectation and the source of every number. 146 children are retired legacy
  children that the current solver refuses to construct, 73 are live.
* `cases/AM2020_mural_geometry_gate` and
  `cases/AM2020_straight_tube_growth_mechanics_smoke/template`: the geometry,
  mesh, boundary conditions, initialiser and dictionaries the main case imports.
* A Colab runner and notebook (`notebooks/AM2020_main_case_colab.ipynb`).

The model is summarised in [docs/MODEL.md](docs/MODEL.md), with equations,
runtime switches and sources.

## Layout

```
cases/AM2020_planar_mural_detachment/   main case: generator, assessor, runner, dry run
cases/AM2020_mural_geometry_gate/       imported geometry, BCs, initialiser
cases/AM2020_straight_tube_growth_mechanics_smoke/template/   imported dictionaries
cases/AM2020_paper_cylinder_platelet_initial_condition_gate/scripts/   Colab bootstrap only
cases/ARCHIVED_CASES.json               registry of retired cases (filtered copy)
user-applications/phaseClotAxisymmetricFluid/   the solver module (wmake libso)
user-applications/beTransport/, plateletContent/   header-only operators it includes
user-applications/phaseClotAxisymmetricFluid.rc   solver-module registration
notebooks/AM2020_main_case_colab.ipynb  Colab notebook
scripts/                                offline checks, evidence archiver, run digest
docs/MODEL.md                           equations, switches, sources
tests/README.md                         what each check proves
tools/                                  manifest and sync tool
```

Relative paths are load-bearing: the generators, the checks and the Colab runner
find each other by them. Do not move files.

## Build

Requirements: OpenFOAM 13 (the OpenFOAM Foundation release, `$WM_PROJECT_DIR`
set by sourcing its `etc/bashrc`), SuiteSparse (UMFPACK) and OpenBLAS headers and
libraries, a C++14 compiler, Python 3.

```
source <OpenFOAM-13>/etc/bashrc
cd user-applications/phaseClotAxisymmetricFluid
./Allwmake            # wmake libso -> $FOAM_USER_LIBBIN/libphaseClotAxisymmetricFluid
```

`Make/options` includes `/usr/include/suitesparse` and links `-lumfpack`. On a
system where SuiteSparse lives elsewhere, edit that include path. On Debian or
Ubuntu: `apt-get install libsuitesparse-dev libopenblas-dev`.

## Generate and run locally

```
cd cases/AM2020_planar_mural_detachment
python3 scripts/create_cases.py            # writes runs/<child>/ (refuses if runs/ exists)
python3 scripts/create_cases.py <filter>   # only matching children
```

Each child directory under `runs/` is an OpenFOAM case (`foamRun`). Run one with
the OpenFOAM environment loaded, for example `cd runs/<child> && ./Allrun`
(see the case's `Allrun` and `cases/AM2020_mural_geometry_gate/Allrun`).
`scripts/assess_case.py <case>` scores a finished run. Full-size children are
expensive (hours on Colab).

## Run on Colab

`notebooks/AM2020_main_case_colab.ipynb` mounts Google Drive, expects a package
zip at `MyDrive/OpenFOAM_Blood/packages/AM2020_main_case_colab_package.zip`,
checks it against a pinned SHA-256, installs SuiteSparse/OpenBLAS, clones
OpenFOAM-13, builds the solver and runs the children chosen by `BATCH`.

The package is a zip of this tree (same relative paths). The pinned
`EXPECTED_SHA` in the notebook is that of the author's own build; to run your
own, build the zip, upload it, and set `EXPECTED_SHA` to its SHA-256.
The notebook was written for the author's own workflow and still contains
workspace-specific text and a long batch table; treat it as a worked example.

## Tests

Everything below runs without OpenFOAM. See [tests/README.md](tests/README.md).

```
python3 tools/sync_from_workspace.py --check-only    # clean-room test of this tree
```

## Citation and licence

Cite via [CITATION.cff](CITATION.cff) and the model sources in
[docs/MODEL.md](docs/MODEL.md). The model builds on M. Capek, Applications of
Mathematics 65(4), 2020, doi:10.21136/AM.2020.0056-19.

Licensed under the GNU General Public License v3.0 ([LICENSE](LICENSE)); the
solver is a module for OpenFOAM, which is itself GPL-3.0.
