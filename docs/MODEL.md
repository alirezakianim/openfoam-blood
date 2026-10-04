# Model: equations, switches and sources

This document describes the physics the solver
`user-applications/phaseClotAxisymmetricFluid` is built to represent. It is
research code. Nothing here is a validated blood-flow model, and no
physiological parameter is claimed to be calibrated.

Two kinds of statement appear below and are labelled:

* **Design**: the equation as specified for the model before it was coded.
* **Code**: what the solver source states about what it implements. Where the
  source has moved beyond the design (the solid is now an Eulerian
  finite-strain network, see section 4) the code is authoritative.

## 1. Geometry and the case

The case is a planar channel with a half-elliptic mural clot on one wall and an
optional smooth throat, built by a single-block `blockMeshDict` and a
point-displacement script (`cases/AM2020_planar_mural_detachment/scripts/apply_throat.py`).
The mural geometry, mesh, boundary conditions and field initialiser are imported
from `cases/AM2020_mural_geometry_gate`; the dictionaries come from
`cases/AM2020_straight_tube_growth_mechanics_smoke/template`.
`constant/mainCaseControls.json` lists 219 children in named series. 146 of
them are retired legacy children that the current solver refuses to construct
(`cases/ARCHIVED_CASES.json`); 73 are live. The Colab notebook's `BATCH`
selects which live children run.

## 2. Interface and platelets (Design, v1/v2)

Fluid indicator `q = (c+1)/2`, where `c = +1` is fluid and `c = -1` is clot.
Sharp-interface target, with `n` pointing from clot into fluid:

```
div(u) = 0
d_t u + div(u u) - div(2 nu D(u)) + grad(p) = f          in the fluid
d_t w + div(u w - D grad w) = 0                          in the fluid
D d_n w = k(s) w                                         on the interface
V_n = alpha d_n w = alpha D^-1 k(s) w                    on the interface
```

`w` is the free platelet concentration, `D` its diffusivity, `k` the surface
absorption speed, `alpha` the interface-speed coefficient and `V_n` the interface
speed. The diffuse form carries the interface through a forced Cahn-Hilliard
pair with `d_t c - div(M grad mu_c) = -V_n |grad c|_delta` and the conservative
platelet equation `d_t(q w) + div(q(u w - D grad w)) + delta_Gamma (k + V_n - u.n) w = 0`.
The sign of the phase source is the corrected one, not the one printed in
Capek (2020), Eq. 4.6. In the current solver the Cahn-Hilliard pair is deleted;
the phase is carried as a content field (section 4), but the interface speed
and the platelet sink keep the same stoichiometry, and the solver reports the
growth/uptake ratio every step as a self-check.

The shear dependence of attachment is a threshold-and-saturating enhancement,
`shearEnhancementModel thresholdSaturating` with `gammaLow`, `gammaHigh` and
`enhancementMax` (template `constant/phaseClotProperties`: 3000 1/s, 8000 1/s and
20).

## 3. Poroelastic clot, damage and release (Design, mechanics v3)

Fluid with drag relative to the moving skeleton, `chi_c = 1 - q`:

```
rho (d_t U + U.grad U) = -grad p + div(2 mu D(U)) - (mu / k(phi)) chi_c (U - d_t U_s)
div(sigma_s) + (mu / k(phi)) chi_c (U - d_t U_s) = 0
sigma_s = g(D_g) [2 G eps(U_s) + lambda_L tr eps(U_s) I],   g(D_g) = (1 - D_g)^2 + eta_res
k(phi) = k0p [phi^3 / (1-phi)^2] / [phi0^3 / (1-phi0)^2]          (Kozeny-Carman)
(G_c / ell) (D_g - ell^2 laplacian D_g) = 2 (1 - D_g) H,          H = max_s psi+
```

Damage `D_g` is irreversible through the history field `H`. Material with
`D_g > DgDetach` (0.95 in the template) is released to the fluid and its
platelet content is returned to the transported field. The drag coefficient is
set by the measured permeability `k0p`, not by a numerical penalty time.

## 4. What the current solver does differently (Code)

From the headers of `phaseClotAxisymmetricFluid.H` and the kernel headers:

| Feature | Statement in the source | File |
| --- | --- | --- |
| Phase representation | `c` and `muC` were deleted; `theta = thetaC + thetaH` (platelet content) is the representation, and `c = 1 - 2 ind` is kept only as a frame for legacy diagnostics | `phaseClotAxisymmetricFluid.H` |
| Two velocities | Fluid `U` and skeleton `Vs` with an exchange coefficient `K`; lumen cells (`isSkel = 0`) carry the identity row `Vs - U = 0` | `blockCoupledKernel.H` |
| Block-coupled step | One implicit system for `(Ux, Uy, Vsx, Vsy, p)` with mixture continuity on implicit Rhie-Chow faces; UMFPACK direct LU, symbolic factorisation reused while the sparsity pattern is unchanged | `blockCoupledKernel.H` |
| Finite-strain network | Left Cauchy-Green tensor `b_e` is transported; the deviatoric stress is `G dev(b_e)` with a neo-Hookean `S = 2 dW/dC` | `beTransport.H`, solver `.C` |
| Pending share as a fibre | A broken, not yet released share carries tension along one axis only: `B_P = (1/lf) I + (lf^2 - 1/lf) n n`, `sigma = G T dev(n n)`, `T = lf^2 - 1/lf` | `fibreReleaseKernel.H` |
| Damage partition | The damage partition follows the content's limited antidiffusive flux (Zalesak-type limiter) so the partition cannot leave `[0, 1]` | `damagePartitionKernel.H` |
| Network pressure | A convex-excess network pressure whose divisions are guarded by construction (zero set of the content weight), tested with floating-point traps enabled | `networkPressureKernel.H`, `scripts/s18_fpe_kernel_test.cpp` |
| Platelet content operator | Shared transfer and drag kernels, header-only, free of OpenFOAM, so a test compiles the same text the solver includes | `plateletContent/` |
| Wall bond | `clotWallAdhesion` mixed boundary condition for the skeleton at the wall | `clotWallAdhesionFvPatchVectorField.{H,C}` |

## 5. Runtime switches

Read from `constant/phaseClotProperties` by the constructor in
`phaseClotAxisymmetricFluid.C` (default in brackets; every default keeps the
earlier behaviour):

* Switches: `clotMechanics` (false), `eulerianSolid` (false), `eulerianBrinkman`
  (false), `finiteStrain` (false), `stronglyCoupledFSI` (false), `fsiAitken`,
  `fsiSweepTrace`, `contentModel` (false), `coreShell` (false), `advectReleased`
  (false), `releaseCoreExempt` (true), `bondHoldCarrier` (false),
  `bondHoldTaper` (false), `projSiteLog`.
* Words: `contentOperator` (content), `dragDriver` (legacy), `networkRupture`
  (off), `networkViscosity` (none), `fibreSlack` (off), `wallPatchSplit`
  (shipped), `cplusOrder` (shipped).

Dictionary values the generator writes for every child, and values the solver
refuses (a value whose implementation was deleted), are checked offline by
`scripts/check_refused_dictionary_values.py`. A key that is read nowhere, or
refused nowhere, is a defect. The meaning of each switch is documented next to
its read in the solver source.

## 6. Template parameters

Values in `cases/AM2020_straight_tube_growth_mechanics_smoke/template/constant/phaseClotProperties`:
`rho` 1050 kg/m^3, `nu` 3.33e-6 m^2/s, `D` 1e-10 m^2/s, `alpha` 1e-6, `k0` 4.6e-5 m/s,
`epsilonC` 6e-6 m, `G` 1 kPa, `poissonRatio` 0.12, `k0p` 3.5e-14 m^2, `phi0` 0.5,
`Gc` 1e6, `ell` 3e-6 m, `etaRes` 1e-6, `DgDetach` 0.95. The children override these;
each number in `mainCaseControls.json` carries a key naming its source or the
decision that fixed it. `Gc` in the mechanics design is a range (roughly
1 to 16 J/m^2 across materials and methods), not a nominal value.

## 7. Sources

* M. Capek, "A phase-field method applied to interface tracking for blood clot
  formation", Applications of Mathematics 65(4), 447-481 (2020),
  doi:10.21136/AM.2020.0056-19. Used as a conceptual reference, not as an
  implementation specification; the phase-source sign is corrected as noted.
* H. G. Weller (2010), free-boundary thrombus growth: the sharp-interface
  model and the Robin condition in section 2; Weller (2013), its analysis.
* Shankar et al. (2022), multiscale thrombus growth: the multiplicative
  threshold-and-saturating shear enhancement.
* Xu et al. (2017), deformation, embolization and permeability of obstructive
  clots, and its supplementary information: the mural-dome case and the
  stiffness and permeability ranges.
* Wufsus et al. (2013), hydraulic permeability of blood clots: `k0p`.
* Miehe et al. (2016), phase-field fracture in porous plasticity, Eulerian
  setting: the damage formulation. Brown et al. (2009), multiscale mechanics of
  fibrin gel stretching.

Bibliographic details above are given from the titles of the source papers; they
should be completed (journal, volume, pages, DOI) before a public release.
