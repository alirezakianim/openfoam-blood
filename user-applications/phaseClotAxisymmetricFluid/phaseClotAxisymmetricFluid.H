/*---------------------------------------------------------------------------*\
  Runtime-selectable canonical solver for AM2020-phase-clot-v1 verification.
  This is a research prototype, not a validated blood-flow model.
\*---------------------------------------------------------------------------*/

#ifndef phaseClotAxisymmetricFluid_H
#define phaseClotAxisymmetricFluid_H

#include "incompressibleFluid.H"
#include "IOdictionary.H"
#include "damagePartitionKernel.H"  // S18p: PassengersDebug (no OpenFOAM deps)
#include "blockCoupledKernel.H"     // S19: the block step, UMFPACK (no OpenFOAM deps)

namespace Foam
{
namespace solvers
{

class phaseClotAxisymmetricFluid
:
    public incompressibleFluid
{
    IOdictionary phaseClotProperties_;

    // Controlled primary fields
    //
    // c_ AND muC_ WERE DELETED IN S6c-2 (2026-09-20). The Cahn-Hilliard pair
    // was the legacy phase representation; theta = thetaC_ + thetaH_ is the
    // representation now, and the last consumer of c_ that reached the
    // physics -- the interface normal -- moved to it in S6c-1. See
    // research/designs/2026-09-20-S6c-2-delete-c.md.
    volScalarField w_;

    // PLATELET CONTENT, W = q*w. The CONSERVED quantity: platelets per unit
    // TOTAL volume, as against w which is a concentration per unit BLOOD.
    //
    // Only meaningful when plateletContentFormulation_ is true; on the old
    // branch it is carried, written and otherwise inert.
    //
    // NOT read from disk. Cases keep their existing 0/w and the constructor
    // sets Wp_ = q_*w_, so no case file changes and no regeneration. It is
    // copy-constructed FROM w_ so it inherits w's boundary types: inlet
    // fixedValue 1 (valid because q = 1 there, asserted at construction),
    // zeroGradient elsewhere. Derivation section 3.5.
    volScalarField Wp_;

    // Derived/output fields
    volScalarField q_;
    volVectorField normal_;

    // THE SURFACE LOCALISER, AND THE FACTOR OF TWO THAT IS NOW GONE.
    //
    // Until S6c-2 there were two of these and they differed by a deliberate
    // factor of two: phaseGradient_ = sqrt(|grad c|^2 + gradFloor^2) -
    // gradFloor, which is 2*delta_Gamma, against interfaceDelta_, which is
    // delta_Gamma. The pairing was load-bearing and easy to get wrong --
    // swapping them broke no dimension check and silently halved or doubled
    // the growth rate.
    //
    // phaseGradient_ is deleted. Its only two consumers were frontSource
    // (phaseGrowthDriver 'front') and chiSource (contentOperator 'matched'),
    // and both key values are refused since S6c-2. What survives is
    // interfaceDelta_ alone, the UNIT surface delta, selected by
    // phaseDeltaBasis_ -- which now has exactly one admissible value.
    //
    // The platelet sink is the Robin condition D n.grad(w) = k w rewritten as
    // a volume term: a flux PER UNIT AREA, so it carries interfaceDelta_ and
    // consumes INT k w dA.
    //
    // Capek (4.6) prints the phase source with |grad(phi_eps(c))|, and
    // phi_eps runs 0 to 1, so read literally it is delta_Gamma. See E-10 in
    // research/literature/AM.2020.0056-19/EQUATIONS_AND_CONVENTIONS.md.
    //
    // checkStoichiometry() remains the instrument that catches a mispaired
    // delta: growth/uptake is 2*alpha/D exactly, and only with this pairing.
    //
    // BOTH FIELDS SURVIVED S6c-2, REBUILT ON theta. The scoping table said
    // phaseGradient_ had exactly two consumers, frontSource and chiSource,
    // and that deleting both key values retired it. It had a THIRD:
    // clotGrowthFront = domainIntegrate(Vn_*phaseGradient_) is read by
    // PHASE_CLOT_GROWTH_STOICH every step and is the 1.6277-against-1.6304
    // number every archived run reports. Deleting it would have silently
    // retired that comparison.
    //
    // phaseGradient_ is now sqrt(|grad c|^2 + gradFloor^2) - gradFloor with
    // |grad c| taken as |grad(-2*ind)|, which is the same quantity: c and
    // -2*ind have identical gradients by c = 1 - 2*ind. It is 2*delta_Gamma;
    // interfaceDelta_ is delta_Gamma. The factor of two between them is
    // unchanged and still load-bearing.
    volScalarField phaseGradient_;
    volScalarField interfaceDelta_;

    //- The theta-derived candidate for the SAME surface delta:
    //  |grad(thetaC_ + thetaH_)|/skeletonVolumeFraction_.
    //
    //  NO LONGER PASSIVE. Under phaseDeltaBasis 'content' this field IS
    //  interfaceDelta_ -- it localises the accepted transfer at :7496 and the
    //  bound-platelet uptake at :8458/:8472. Under 'legacy' (the default) it
    //  is still read by nothing but the instrument.
    //
    //  BECAUSE IT IS NOW CONSUMED IT IS ASSIGNED IN updateDerivedFields(),
    //  NOT IN writeBalanceDiagnostic(). It used to be assigned in the
    //  diagnostic, which runs in postSolve() AFTER the transfer has already
    //  committed; a consumer reading it there would have read the PREVIOUS
    //  step's geometry. The instrument reports deltaThetaIntPost -- the same
    //  integral recomputed at postSolve -- so the size of that relocation is
    //  measured every step rather than assumed to be small.
    //
    //  It exists so the substitution proposed in
    //  research/designs/2026-09-16-theta-native-formulation.md can be
    //  compared with deltaC_ CELL BY CELL, which run 20260916-053445
    //  proved is the only comparison that matters: the ratio of the two
    //  DOMAIN INTEGRALS passed its gate at 0.27% while the distribution of
    //  the binding source had already moved 2.2% (delta-weighted mean) and
    //  15.6% (delta-weighted p99) in 0.4 ms. An integral cannot tell a
    //  conserved total from a conserved distribution. This field can.
    //
    //  Written every write time like interfaceDelta_ so the per-cell diff is
    //  possible after the fact and not only through whatever statistic the
    //  instrument happened to choose.
    //
    //  Zero, and left zero, when there is no platelet content model or when
    //  skeletonVolumeFraction_ is not positive -- theta is then frozen at its
    //  initialisation and any figure derived from it would describe nothing.
    volScalarField deltaTheta_;
    volScalarField shearPa_;
    volScalarField shearRate_;
    volScalarField adhesionEnhancement_;
    volScalarField adhesionSpeed_;
    // k_eff: the adhesion speed ACTUALLY used by both the platelet sink
    // and the growth closure. Equal to adhesionSpeed_ unless
    // interfaceDepletionModel_ is on. It exists so that consumption and
    // growth cannot be driven by two different numbers -- that identity
    // is what makes the stoichiometry exact.
    volScalarField adhesionSpeedEff_;
    volScalarField Vn_;
    volScalarField lambda_;
    volScalarField surfaceLossCoeff_;

    // --- v3 clot mechanics: poroelastic skeleton + phase-field fracture ---
    // All are allocated always but only evolved when clotMechanics_ is true,
    // so the rigid v2 path is untouched.
    volVectorField Us_;          // skeleton displacement
    volVectorField UsPrev_;      // displacement at the start of the step;
                                 // managed explicitly rather than relying on
                                 // oldTime(), which only advances when a ddt
                                 // scheme touches the field
    volScalarField Dg_;          // damage, 0 intact -> 1 broken
    volScalarField H_;           // history field, max of psi+
    volScalarField poroPerm_;    // permeability k(phi)

    //- chi*nu/k, the Brinkman drag exactly as it enters the momentum
    //  equation. Written for ParaView because poroPerm spans nineteen
    //  decades and reads BACKWARDS: blood is the infinitely permeable
    //  limit, so it is the red end of the scale, not the blue one.
    //
    //  KINEMATIC, 1/s, THE SAME UNIT AS lambda_, AND IT USED TO BE DYNAMIC.
    //  Both fields are read by writeBalanceDiagnostic() as "the coefficient
    //  in fvm::Sp(coeff, U)", which for the kinematic momentum equation
    //  incompressibleFluid writes must be 1/s. clotDrag_ was constructed
    //  with dimDensity/dimTime and assigned chi*mu/k in solveSkeleton(),
    //  so accDrag was rho = 1050x the acceleration it was being summed with
    //  in accSum. Latent, because the only path that added it to accSum is
    //  the skeleton path and production is Eulerian; it would have become a
    //  live wrong number the moment the Brinkman term was flagged into the
    //  budget. Now 1/s on both paths, so accPenalty and accDrag are the
    //  same kind of quantity as accViscous and can legitimately be added.
    volScalarField clotDrag_;
    volScalarField released_;    // 1 where clot material is no longer
                                 // connected to an intact bond and is free to
                                 // be carried off, 0 elsewhere. WRITTEN, so
                                 // the embolus is visible in ParaView as a
                                 // field rather than inferred from a log line.

    //- THE SKELETON VELOCITY OF RELEASED MATERIAL, held for the whole step.
    //
    //  ONE VECTOR, TWO CONSUMERS, AND THAT IS THE POINT. Before this existed,
    //  releasedBodyVelocity() was called once, inside solvePhaseField(), and
    //  the embolus was ADVECTED at that vector while the momentum equation
    //  knew nothing about it. The two now read the same number by
    //  construction, because a body carried at one velocity and dragged
    //  towards another is not a body, it is two models sharing a mask.
    //
    //  Set once per time step in prePredictor(), immediately after
    //  updateReleasedIndicator() and BEFORE solvePhaseField(), so both the
    //  carrier flux and the Brinkman drag see the same value computed from
    //  the same U. Zero whenever nothing is released, which makes the
    //  relative-velocity drag collapse exactly to the lab-frame form.
    vector UbodyCached_;

    //- 1 in cells anchored to an INTACT bond, 0 elsewhere. WRITTEN.
    //
    //  WHY THIS EXISTS. Offline analysis of run 20260824-220555 found that on
    //  the Eulerian path NOTHING resists tangential motion of clot material
    //  at the bonded footprint. The source already said so at the momentum
    //  balance diagnostic -- "On the Eulerian path lambda_ and clotDrag_ are
    //  BOTH absent from UEqn -- the only thing holding the clot is
    //  div(sigma_e)" -- and clotWallAdhesion constrains Us, which under
    //  eulerianSolid is PASSIVE (D(Us)/Dt = U, used nowhere in momentum). So
    //  the bond could see its load after the sigmaElastic repair and still
    //  could not apply one. Measured consequence: with adhesiveStrength 1e9 Pa
    //  and debondedFaces = 0 at every one of ~15000 steps, the clot standing
    //  over the wound fell from 1093 cells at 1 ms to 6 at 12 ms and its
    //  height over the footprint collapsed 34 -> 2 cells between 6 and 7 ms.
    //  An unbreakable bond that holds nothing is not a weak anchor, it is no
    //  anchor.
    //
    //  Grown from the wall cells of bonded faces through clot material, by
    //  bondHoldDilation cells. A SHALLOW LAYER, deliberately: its boundary
    //  then lies inside the clot bulk where grad(c) is negligible, which is
    //  the same condition the release mask is sized by and for the same
    //  reason -- a velocity jump across live field shears it apart.
    volScalarField bondHold_;

    volSymmTensorField sigmaD_;  // deviatoric-ish stress carrier

    //- PERSISTENT copy of elasticStress(), for the wall-adhesion condition.
    //
    //  WHY THIS EXISTS. clotWallAdhesion reads a registered volSymmTensorField
    //  by name and defaults to "sigmaD". sigmaD_ is written ONLY inside
    //  solveSkeleton(), and the Eulerian path never calls solveSkeleton() --
    //  it runs solveElasticStrain() instead. So on that path the condition
    //  found sigmaD_ at its constructed value of ZERO, every step, and
    //  computed a traction of exactly zero on every face. Run 20260824-033102
    //  reported peakTractionAny = 0 and utilisation = 0 for 20 ms against a
    //  strength of 1e9 Pa: not an unloaded bond, an INERT FAILURE CRITERION.
    //  No face could ever debond, and a strength of 1e-9 Pa would have behaved
    //  identically.
    //
    //  elasticStress() returns a tmp. Pointing the condition at that name
    //  would NOT work and would fail silently in the same way, because the
    //  temporary is gone by the time the U boundary condition is evaluated and
    //  foundObject() would simply return false. A persistent member removes
    //  the question.
    //
    //  WHAT IT IS NOT. sigmaElastic is DEVIATORIC by construction -- the
    //  volumetric part lives in the pressure, which is what makes it
    //  compatible with the incompressible momentum equation. The partitioned
    //  path's sigmaD_ carries lambda*tr(E)*I as well, so the bond there sees a
    //  normal (peeling) traction and here it sees only shear. That is a real
    //  modelling difference between the two paths and it is stated, not
    //  absorbed.
    volSymmTensorField sigmaElastic_;

    // --- Eulerian elastic strain (stage 2 of the Eulerian-solid plan) ------
    //
    // The replacement for the total-Lagrangian displacement skeleton, which
    // inverted at 31% shell elongation in run 20260816-031412 for a structural
    // reason no patch removes: a fixed reference cannot follow material
    // advected into cells whose displacement is still zero. The transport
    // itself lives in ../beTransport/beTransport.H and is SHARED, unchanged,
    // with the stage-1 gate application that qualified it -- spatial order
    // 1.960, temporal 1.995, SPD held at a principal stretch of 3.271, run
    // 20260816-221135.
    //
    // BEHIND A SWITCH, DEFAULT OFF. With eulerianSolid_ false every expression
    // below is skipped and the v3 skeleton path is bit-identical to what all
    // the historical gates were run against. That is not caution for its own
    // sake: the stage-2 gate IS the comparison between the two paths on the
    // elastic slab, so both have to be runnable from one binary.
    //
    // WHAT THIS COSTS, RECORDED WHERE THE CODE IS. A single velocity field has
    // no relative solid-fluid motion, so BIOT CONSOLIDATION IS GONE. The
    // two-field skeleton could drain; this cannot. It is a real loss of
    // capability traded for kinematics that survive large strain, and it must
    // be reported as one rather than quietly dropped. See section 5 of
    // research/literature/AM.2020.0056-19/KINEMATICS_DECISION_2026-08-15.md.

    //- Put the Brinkman drag back on the Eulerian path, gated on release.
    //  DEFAULT OFF, so every case that predates this switch assembles a
    //  byte-identical matrix and no passed gate moves.
    //
    //  WHY THE TERM WAS DROPPED AND WHY IT CAN COME BACK. The comment at the
    //  eulerianSolid branch of momentumPredictor() is right that
    //  (mu/k)(U - Us_velocity) has nothing to act on when there is one
    //  velocity, and right that keeping it with Us = 0 would pin the clot in
    //  the LAB FRAME -- a clot being carried downstream, the thing Dr Li
    //  asked to see, would be held in place by a term meant to describe
    //  seepage. That objection is answered by the gate, not by the term:
    //  released_ is 1 wherever clot material has lost its connection to an
    //  intact bond, so material that has debonded feels no drag at all and
    //  leaves freely. The bond stays the CRITERION for detachment; Brinkman
    //  only supplies the FORCE that holds attached material while it is
    //  still attached.
    //
    //  WHAT IT IS FOR. Run 20260827-103135 measured 57% of the clot leaving
    //  the wound in the first 8 ms with total volume CONSERVED -- advection,
    //  not erosion -- because above the ~20-cell bondHold band nothing
    //  resists tangential motion at all. chi*nu/k was 3.3e8 1/s in the core
    //  and 3.3e5 1/s in the shell AT THAT RUN'S k (1e-14/1e-11), a 1000x
    //  core/shell contrast. THE PAIR TRACKS k0p/k0pcore -- recompute it, do
    //  not quote these. The current main case (k0p 1e-13, k0pcore 1e-15) is
    //  1.008e+07 / 1.008e+09, a 100x contrast, measured as max 1.007096e+09
    //  in run 20260909-182243. Either way it holds the body, which is
    //  Xu (2008) eq. (3.2)'s -(eta*phi3/k)u term-for-term.
    //
    //  STATED RISK, not hidden: coeff*dt = 675 in the core, so the core may
    //  go rigid -- the opposite failure from the present one. bk_soft in the
    //  eulerian_brinkman series brackets it by moving k0pcore alone.

    //- Elastic left Cauchy-Green tensor. I where unstrained.
    volSymmTensorField be_;

    //- 1/tau for the relaxation back to I: ~0 in clot, large in blood, so the
    //  same field carries a solid and a fluid without a phase-dependent
    //  branch anywhere in the transport.
    volScalarField beRelax_;

    //- The blood relaxation rate that beRelax_ ramps up to.
    scalar beRelaxRate_;

    //- Picard sweeps on the explicit stretch term. Stage 1 measured the cost
    //  of getting this wrong exactly: with one sweep the temporal order is 1
    //  whatever ddtScheme the case names, because the source is evaluated at
    //  the old be and BDF2's left side is not. Three sweeps restored 1.995.
    label beCorrectors_;

    //- Project the trace out of the stretch source, so det(be) is conserved
    //  identically. See beTransport::stretch for the derivation and for the
    //  measurement that motivates it (run 20260818-081527: |tr L|/|L| up to 1.0
    //  in the clot bulk against |div phi| = 3.3e-8, det(be) reaching 4.24 and
    //  immune to both Picard sweeps and time-step refinement).
    //
    //  Default OFF: it is a modelling choice, and the A/B against it is the
    //  evidence that it is the right one.

    //- Advect the phase field with phi_ on the Eulerian path, instead of
    //  rebuilding a chi-blended material flux from the cell-centred vs.
    //  Defaults ON: one velocity means one flux, and the blend is the
    //  partitioned formulation's construction. See the derivation and the two
    //  measured defects at the call site in solvePhaseField().

    //- Build the stretch term's velocity gradient by Gauss on a face velocity
    //  whose normal component is exactly phi_/|Sf|, so its trace is div(phi)
    //  and therefore machine zero. Defaults ON. Derivation, the measurement
    //  that motivates it, and why it is NOT the dev(gradU) mistake again, are
    //  at the call site in solveElasticStrain().

    //- Report the be equation TERM BY TERM at the cell with the smallest
    //  eigenvalue of be, every step. DEFAULT OFF: it costs four extra
    //  explicit field operations per step and is a diagnostic, not physics.
    //
    //  Exists because run 20260823-204058 lost positive-definiteness at
    //  t = 3.044e-3 s in a cluster three cells deep against the wall, while
    //  every global quantity looked ordinary -- eleven hypotheses were tested
    //  against bulk means and all eleven died. Rationale and the reading rule
    //  are at the call site in solveElasticStrain().

    //- Apply the elastic force through the FACE flux rather than as a
    //  cell-centred source inside UEqn.H(). DEFAULT OFF, because it is
    //  untested; every existing child keeps its current behaviour.
    //
    //  WHY. In the base incompressibleFluid::correctPressure() the pressure
    //  reaches the face flux as pEqn.flux(), a snGrad, while any source inside
    //  UEqn.H() reaches it only after being lumped into HbyA and interpolated.
    //  OpenFOAM tolerates that mismatch for pressure alone because the residual
    //  IS the Rhie-Chow damping that suppresses checkerboarding. It is not
    //  benign for a body force at 5x the applied load: the pressure equation
    //  cannot cancel the elastic force face by face, and the remainder drives a
    //  cell-centred velocity the solenoidal flux never sees.
    //
    //  MEASURED, across runs 20260818-100608 / -125909 / -223637 and
    //  20260819-231939: tr(grad U) in the clot bulk is 240-527 1/s while
    //  div(phi) is 6e-6 1/s -- EIGHT ORDERS. trFracBulk reaches 1.3825 of the
    //  2-D ceiling sqrt(2), i.e. 98% pure spurious dilatation. trBulk moves
    //  1400x with G while divPhiMax stays flat. The elastic stress in the clot
    //  is 17.86 Pa mean and 75.1 Pa peak against a 3.5 Pa wall shear, so the
    //  force is computed at 5x the load and the clot still creeps at
    //  lockRatio 0.70-0.89 -- and 1.283 after 42 viscous diffusion times.
    //
    //  TWO RIVAL EXPLANATIONS ARE DEAD. det(be) was pinned to 1e-14 by brute
    //  force in run 20260818-232055 and nothing else moved. Elapsed time is
    //  refuted by the 42-diffusion-time measurement above. nOuterCorrectors 5
    //  changed nothing and cost 2.4x, so it is not a splitting error either.
    //
    //  IT IS ONE CHANGE EVEN THOUGH IT TOUCHES TWO TERMS. Consistency is a
    //  property of the PAIR: the cell velocity is updated by fvc::reconstruct
    //  of the same face fields that drive the flux, for the elastic force and
    //  the pressure together. Reconstructing one and not the other would not be
    //  a partial fix, it would be a different inconsistency.
    //
    //  NOT ESTABLISHED AS THE CAUSE. Nine causes have been wrong in this
    //  project. What is established is the stencil mismatch and its dose
    //  response; that fixing it fixes the lock is what the A/B measures.

    //- Rescale be by det^(-1/3) after each solve so det(be) = 1 exactly.
    //  A BAND-AID, DEFAULT OFF: it restores the volume without undoing the
    //  advective smearing that destroyed it, and defaulting it on would make
    //  the det gate pass forever while the drift continued unmeasured. See the
    //  derivation and the Minkowski argument at the call site.

    //- Courant number for the shear wave, c_s = sqrt(G/rho).
    //
    //  The elastic stress enters the momentum equation EXPLICITLY, through
    //  fvc::div(sigmaElastic), so the shear wave it supports is an explicit
    //  stability limit and not merely an accuracy one. The fluid Courant
    //  number does not see it at all: blood moves at 0.0083 m/s mean while
    //  the shear wave runs at sqrt(4/1050) = 0.062 m/s in the shell and
    //  sqrt(40/1050) = 0.195 m/s in the core, so maxCo can be comfortably
    //  satisfied by a dt at which the elastic term is unstable.
    scalar elasticCourant_;

    //- Safety factor on the EXPLICIT biharmonic limit dt <= C h^4 / M that
    //  the Cahn-Hilliard term fvc::laplacian(mobility, muC) imposes.
    //  Default 1/16, the standard 2-D coefficient. Inactive when mobility
    //  is 0, which is every mural child.
    scalar biharmonicCourant_;

    //- Fraction of the Kelvin-Voigt relaxation time tau_kv = mu/G that one
    //  step may span.
    //
    //  The wave CFL above bounds shear-wave travel; this bounds how far the
    //  material can be asked to relax onto its own equilibrium in one step,
    //  with the elastic stress lagged. Run 20260817-052515 measured the slab
    //  gate's strain error tracking dt/tau_kv (1.35 -> 0.175, 4.3 -> 0.462,
    //  13.5 -> 0.934) with a halved-dt control confirming dt was the cause.
    //  Not binding at production settings; see the note at its use.
    scalar elasticRelaxCourant_;

    //- Diagnostics from the last solve, reported per step.
    scalar beSweepChange_;
    scalar beMinMinor_;
    scalar beMaxStretch_;
    scalar beDetError_;

    //- det error AFTER the optional rescale. Equal to beDetError_ when the
    //  rescale is off, so the two can always be printed side by side and the
    //  masked drift is never invisible.
    scalar beDetErrorPost_;

    // Numerical and physical controls
    dimensionedScalar rho_;
    dimensionedScalar nu_;
    dimensionedScalar D_;
    dimensionedScalar alpha_;
    dimensionedScalar k0_;
    dimensionedScalar etaP_;
    dimensionedScalar gradFloor_;
    dimensionedScalar qFloor_;
    scalar penaltyExponent_;

    // Shear-dependent adhesion closure.
    //   "linear"             legacy k = max(0, k0 + k1*s_Pa)
    //   "thresholdSaturating" k = k0 * E(gammaDot), Shankar et al. (2022)
    //                         eq. (9): E = 1 below gammaLow, linear ramp to
    //                         enhancementMax at gammaHigh, saturated above.

    // THE GROWTH CLOSURE. See
    // research/designs/AM2020_growth_closure_reaction_form.md
    //   "gradient"  legacy: Vn = alpha*(n & qEff*grad(Wp/qEff)).  Capek's
    //               SHARP-interface law (3.5), which needs the Robin
    //               condition D dn(w) = k w to be resolved pointwise.
    //   "reaction"  Capek's own DIFFUSE-interface law (4.6):
    //               Vn = (alpha/D)*kEff*w. No gradient, no boundary layer
    //               to resolve, and exactly stoichiometric with the sink.
    // Defaults to "gradient" so every previously passed gate reproduces
    // byte-identically without editing its dictionary.

    // Sub-grid depletion resistance folded into kEff.
    //   "none"              kEff = k                       (default)
    //   "seriesResistance"  kEff = 1/(1/k + h/(2 D))
    // Applied to the sink and the growth together, so switching it on
    // cannot break the stoichiometry.
    dimensionedScalar gammaLow_;
    dimensionedScalar gammaHigh_;
    scalar enhancementMax_;

    // --- v3 mechanics controls (AM2020-clot-mechanics-v3) ---
    dimensionedScalar G_;            // skeleton shear modulus
    scalar poissonRatio_;            // drained Poisson ratio
    dimensionedScalar lambdaLame_;   // second Lame parameter
    dimensionedScalar k0p_;          // intrinsic permeability
    dimensionedScalar Gc_;           // fracture toughness
    dimensionedScalar ell_;          // fracture regularisation length
    scalar etaRes_;                  // residual stiffness
    scalar DgDetach_;                // damage at which material is released

    // --- geometrically nonlinear skeleton ---
    // DEFAULTS TO FALSE, so every case that does not ask for it assembles the
    // small-strain system byte for byte and no passed gate moves. The
    // St Venant-Kirchhoff second Piola-Kirchhoff stress linearises EXACTLY to
    // the small-strain Cauchy stress, which is what makes that guarantee
    // structural rather than hopeful.
    //
    // Why it exists: the deformation half of the PI's question is not
    // representable in small strain. Run 20260809-113416 reached 375% strain,
    // Ramanujam et al. (2024) state that embolisation prediction "requires a
    // large-deformation, three-dimensional FE calculation", and Storti and
    // van de Vosse (2014) model the growing plug as neo-Hookean at finite
    // strain and find its deformability essential.
    //
    // St Venant-Kirchhoff rather than neo-Hookean: it is the direct
    // geometrically nonlinear extension of the existing (G, lambda)
    // parameterisation, it captures large ROTATION which is what a wall-
    // attached clot bending downstream undergoes, and it introduces no new
    // material constant. It is not a good model at large STRETCH; that is a
    // declared limit, not an oversight.

    // --- Strongly-coupled flow/skeleton exchange ---
    // DEFAULTS TO FALSE, so a case that does not ask for it gets the
    // loosely-coupled path byte for byte and every passed gate is unaffected
    // until it opts in.
    //
    // Why it exists: the loose path exchanges flow and skeleton information
    // exactly ONCE per time step - prePredictor() is guarded by
    // pimple.firstIter() and the cases run nOuterCorrectors 1 - so it is a
    // partitioned Gauss-Seidel sweep performed once and never iterated. Run
    // 20260804-022004 measured that path oscillating by 58 percent at
    // G = 1 kPa, with a PERIOD THAT SHRINKS WITH dt and therefore no continuum
    // limit, and showed the timestep is not a usable repair (382 to 11743
    // hours to reach a tolerable amplitude).
    scalar fsiRelaxation_;        // initial (or fixed) relaxation factor
                                  // exit tolerance of the SEGREGATED loop in
                                  // solveSkeleton(), distinct from the linear
                                  // solver tolerance beneath it. A dictionary
                                  // entry since 2026-08-05, defaulting to the
                                  // 1e-7 it always had, because it is a
                                  // candidate cause of the exchange floor that
                                  // run 20260805-071815 measured.
    label releaseSupportSweeps_;  // max-propagation sweeps used to decide what
                                  // is still connected to a bonded face

    //- c below which a cell counts as MATERIAL for the support test.
    //
    //  THE SUPPORT TEST IS A CONNECTED-COMPONENT QUERY ON A LEVEL SET, AND A
    //  DIFFUSE INTERFACE HAS NO TOPOLOGY. With this at its historical value of
    //  0 the query is asked of the c = 0 contour, an arbitrary cut through a
    //  field smeared over epsilonC and further by advection. Run
    //  20260824-220555: by 9 ms every cell in row 1 over the whole 40-face
    //  footprint was already blood (c = 0.4164 to 0.9014), and support
    //  survived only along a ONE-CELL-TALL filament running downstream past
    //  the wound edge onto intact endothelium and climbing into the body
    //  there. At t = 8.4931 ms that filament pinched and 1278 cells -- 98% of
    //  the clot -- flipped to released IN A SINGLE TIMESTEP, with all forty
    //  bonds intact. The release then handed the body to Ubody = 9.14e-3 m/s
    //  against a row-0 fluid Ux of 5.58e-4, and the wound was vacated at 16x
    //  the local flow speed.
    //
    //  THE INCONSISTENCY IS THE BUG. releasedMaskLimit is 0.9999 and its
    //  documented meaning is "c above which a cell carries no meaningful clot
    //  material". The support test asks the same question -- is there material
    //  here -- and answers it with a different threshold. Raising this makes
    //  the two agree.
    //
    //  DEFAULT 0, so every result established before 2026-08-24 reproduces
    //  exactly. The value is swept rather than asserted: a high limit cannot
    //  release a genuine embolus that has drifted only a few cells clear, and
    //  that loss of capability is real and is measured, not argued.
    scalar releaseSupportLimit_;

    //- Cells grown from a bonded wall face, through material, that the bond
    //  holds. NEGATIVE disables the hold entirely, which is the default.
    label bondHoldDilation_;

    //- 1/s. Coefficient of fvm::Sp(bondHoldPenalty*bondHold*chi, U) in the
    //  Eulerian momentum equation, so the anchored layer resists the FLUID as
    //  well. Default 0. UNSOURCED -- there is no measurement behind any value
    //  of it, so it is swept and reported as a numerical control, never as a
    //  material property.
    scalar bondHoldPenalty_;

    //- Zero the phase-field carrier flux inside bondHold_.
    //
    //  NECESSARY AND NOT OPTIONAL, and this is the part the first design of
    //  this repair got wrong. A momentum-equation penalty alone CANNOT hold
    //  the clot, because once the release mask covers it c is carried by
    //  Ubody, which is computed from the free stream and is INDEPENDENT of U.
    //  Slowing the fluid would leave the phase field translating at 9.14e-3
    //  m/s regardless. An anchor has to act on the CARRIER OF c, not only on
    //  the momentum equation.

    //- Ramp the hold linearly to zero over bondHoldDilation cells instead of
    //  switching it off at a cell boundary.
    //
    //  WHY IT EXISTS, AND WHY "HOLD MORE" IS THE WRONG READING OF THE DEFECT.
    //  Run 20260825-013027 held a binary four-cell band and Dr He Li rejected
    //  the resulting core: it reads as SEVERED, cut along the top of the band,
    //  because everything above the cut is unheld and shears over a base that
    //  is fully held. The natural-seeming repair -- flood the hold through the
    //  whole core -- CANNOT WORK, and the code says so directly. The penalty
    //  is fvm::Sp(bondHoldPenalty*bondHold*chi, U), so bondHold = 1 across the
    //  core drives U to zero there; be is stretched by gradU built from U and
    //  phi (beFluxConsistentGradU), so gradU = 0 leaves be at the identity and
    //  THE CORE STOPS DEFORMING ALTOGETHER. The carrier hold is stricter
    //  still: it multiplies the transporting flux by (1 - bondHold), so held
    //  material cannot move at all. Li asked for small, smooth deformation of
    //  the whole core, and a uniformly held core is a rigid one -- the same
    //  objection, in the opposite direction.
    //
    //  The band's defect is its EDGE, not its size. Tapering removes the shear
    //  plane while leaving the base held and the crest free, so the core's own
    //  stiffness still decides how much it deforms.
    //
    //  Default false, so every result before 2026-08-25 reproduces exactly.

    //- Never mark a CORE cell released. On by default, and effective only
    //  when coreShell is on, so a single-material case is untouched.
    //
    //  Dr He Li, 2026-08-25: the core "does not slide or detach or rupture",
    //  while the shell "can deform, rupture, and detach from the shell region
    //  itself". The support test knows nothing about material; it runs on the
    //  c level set, and a diffuse interface has no topology, so a numerical
    //  pinch released 1278 cells in one step in run 20260825-013027.
    bool releaseCoreExempt_;

    //- Write the skeleton momentum residual, diagnostic only.
    //
    //  WHY THIS EXISTS. The Eulerian path carries ONE velocity, so the Darcy
    //  drag -(nu/k)(U - 0) is a sink toward the LABORATORY frame. That is
    //  right for material an intact bond holds and wrong for everything else,
    //  and it is why run 20260904-093228 shows a wall layer that is frozen
    //  rather than held (research/designs/AM2020_frozen_clot_diagnosis_
    //  2026-09-04.md).
    //
    //  The quasi-static skeleton balance
    //      div(sigma_e)/rho - chi*grad(p) + dragCoeff*(U - v_s) = 0
    //  can be solved for v_s to ask what skeleton velocity the CURRENT fields
    //  already imply. Reconstructing it offline from written fields gave
    //  |v_s| > |U| -- a skeleton outrunning the fluid dragging it, which is
    //  impossible -- but that reconstruction took a tensor divergence from
    //  CELL data on a taper-bent mesh with numpy, not with the solver's own
    //  operator and the declared div(sigmaElastic) scheme. This writes the
    //  terms using those operators so the two can be told apart. It CHANGES
    //  NOTHING that is solved.

    //- div(sigma_e)/rho, the elastic force per unit mass [m/s^2]
    volVectorField skelElasticForce_;

    //- chi*grad(p), the pressure force per unit mass [m/s^2]
    volVectorField skelPressForce_;

    //- TWO-VELOCITY SKELETON. The clot gets its own velocity field.
    //
    //  WHAT IS WRONG WITHOUT IT. The Eulerian path carries ONE velocity, so
    //  the Darcy term is -(nu/k)(U - 0): a momentum sink toward the LABORATORY
    //  frame. Darcy drag acts on the RELATIVE velocity of fluid and skeleton,
    //  and with one velocity that relative velocity is identically zero, so
    //  the term has no physical referent unless something really is pinned in
    //  the lab frame. For material an intact bond holds, something is. For
    //  everything else nothing is, and run 20260904-093228 is the result: the
    //  clot is transported at the SEEPAGE velocity (measured, median
    //  |U|/(|grad p|/drag) = 1.081 over 4072 clot cells), which is fluid
    //  percolating THROUGH the skeleton rather than the skeleton moving.
    //
    //  WHAT THIS DOES. Adds the quasi-static skeleton momentum balance
    //      div(sigma_e)/rho - chi*grad(p) + dragCoeff*(U - Vs) = 0
    //  solved for Vs. The skeleton is massless and creeping, so the balance
    //  has no inertia and no convection and is ALGEBRAIC in Vs -- it is not a
    //  second Navier-Stokes equation and it costs no second linear solve.
    //  Vs then carries the phase field and be, and the fluid drag becomes
    //  dragCoeff*(U - Vs).
    //
    //  IT IS EULERIAN. Vs is a velocity field on the fixed mesh. There is no
    //  displacement, no reference configuration, and therefore none of the
    //  inversion mode that killed the total-Lagrangian skeleton at t = 0.003 s
    //  in run 20260815-231941, and none of its 77.3% wall-clock share.
    //
    //  Default OFF. It changes what the phase field is advected by, which is
    //  the most consequential switch in this solver.

    //- Blend the material carrier to the fluid flux as the clot fraction
    //  vanishes:  phiMaterial = chi_f*phiSkel + (1 - chi_f)*phi.
    //
    //  WHY THIS EXISTS. The two-velocity branch carried the phase field and
    //  the core/shell label on the RAW skeleton flux phiSkelOwn_, everywhere
    //  in the domain -- including cells that are pure blood, where there is
    //  no skeleton and that flux is the output of a 2x2 momentum block held
    //  up only by the Kres regularisation. Measured in run 20260908-010836 at
    //  7.949 ms: a lumen cell with chi = 4.03e-11 -- forty billionths of one
    //  per cent solid -- reported a skeleton velocity of 0.0386 m/s, which is
    //  110% of max|U|. That velocity carried c and shell.
    //
    //  THE EVIDENCE THAT IT IS THE CARRIER AND NOT THE PHYSICS. Across every
    //  run on record in preflight/:
    //
    //    carrier = phi_  (solenoidal, pressure-corrected, one-velocity)
    //        9 children, 9 completed.
    //    carrier = phiSkelOwn_ (raw skeleton flux, two-velocity)
    //        14 children, 14 failed -- 13 by SIGFPE inside
    //        advectMaterialLabel(), 1 (em_2vel_ps45_cub_rchi) by going inert.
    //
    //  The 13 crashes are ALL the same stack: symGaussSeidelSmoother::smooth
    //  <- advectMaterialLabel <- solvePhaseField <- prePredictor. They are
    //  scattered over 300x in time (4.4e-05 s to 1.42e-02 s), which is the
    //  signature of a POINTWISE trigger, not a threshold crossed by a smooth
    //  quantity: the shell residual is 8.08e-05 -> 5e-13 in 2 iterations on
    //  the step before the crash, and every reported diagnostic is smooth
    //  into it.
    //
    //  The 14th failure is the same defect wearing its other face. In
    //  em_2vel_ps45_cub_rchi the clot centroid clotXbar is FLAT to four
    //  digits from 14.7 ms to 30 ms (8.6281 -> 8.6340 um) while clotUxbar
    //  reports 8.96 mm/s. At that speed the centroid owed 134 um of travel
    //  and delivered 0.006 um. A carrier that is locally large but not a
    //  coherent material motion moves nothing net -- the material reports a
    //  velocity the body does not perform.
    //
    //  WHAT THE BLEND COSTS. Inside the clot chi -> 1 and the carrier is
    //  unchanged, so the clot still rides its own skeleton, which is the
    //  whole point of the two-velocity path. Only the dilute fringe and the
    //  lumen change -- exactly where the skeleton is a fiction.
    //
    //  This is NOT a new construction. The one-velocity
    //  phaseFieldMaterialVelocity_ branch a few lines below has always built
    //  its carrier as chif*(...) + (1 - chif)*phi_. The two-velocity branch
    //  simply never inherited it.
    //
    //  chi_f is clamped to [0, 1] before use. chi = 1 - q with q = (c+1)/2,
    //  and c undershoots: cUnder reached 0.187 in run 20260908-010836, which
    //  puts chi at 1.094. Unclamped, 1 - chi_f goes negative and the blend
    //  starts SUBTRACTING the blood flux.
    //
    //  DEFAULT OFF as of 2026-09-08. It shipped ON, with the argument that
    //  "shipping a known-wrong carrier behind an off-by-default flag is how
    //  the defect survives". That argument confuses two things: the old
    //  carrier being wrong does not make this one right, and a default is
    //  not the place to assert an untested repair.
    //
    //  Codex recommended OFF twice and I declined twice, on the grounds that
    //  the blend was the only plausible reason Batch 25 stopped crashing and
    //  that turning it off would discard that. Both halves were wrong. Batch
    //  25 bundled the blend with three other changes, so it was never
    //  attributable; and the returned assessor rejected BOTH children
    //  outright -- children_admissible = 0, with the c-bound, Wp and Vs gates
    //  all false. A run that completes into an inadmissible state is not
    //  evidence the carrier works.
    //
    //  Nothing is discarded by the flip: both Batch 25 children name
    //  material_carrier_blend explicitly in mainCaseControls.json, so their
    //  configuration is byte-for-byte reproducible. The switch still exists
    //  precisely so the A/B is measured rather than asserted -- which is what
    //  the next run is for, with this as the only variable.
    //
    //  The decisive argument is one neither of us made: ON-by-default broke
    //  REPRODUCTION of every earlier two-velocity child. Twelve of the
    //  fourteen never name the key, and all twelve ran before this switch
    //  existed -- run 20260908-010836's log has no carrierBlend field at all,
    //  while 20260908-061511's says carrierBlend=ON. With the default ON,
    //  re-running em_2vel_ps45_cub_df1e6 would silently apply a carrier that
    //  child never had and would not reproduce its SIGFPE. OFF restores it.

    //- Cap the growth/erosion interface speed so a step cannot carry c
    //  outside [-1, 1].  DEFAULT OFF, and it is off because it was DISPROVED.
    //
    //  It shipped in Batch 25 as a root fix. It is not one, on two
    //  independent grounds:
    //
    //  1. ALGEBRA. With chi = 1, div(v_s) = -100 1/s and G = 0 EXACTLY,
    //     implicit Euler gives chi = 1/(1 + dt*div v) = 1.000149276 and
    //     c = -1.000298552. Compression alone leaves the bound with zero
    //     growth, so no cap on the SOURCE can hold it. Forward invariance of
    //     chi <= 1 needs G - div(v_s) <= 0 at chi = 1, which constrains the
    //     CARRIER.
    //  2. MEASUREMENT, run 20260908-061511 child em_2vel_carrier. The cap
    //     fired 2,505,301 cell-steps over 80,398 samples, up to 116 cells in
    //     one step -- and cUnder still reached 1.4035, with c = -2.44 and
    //     qNegCells between 49 and 564 throughout. The pre-registered pass
    //     condition was qNegCells = 0. It failed by every sample.
    //
    //  Kept switchable rather than deleted because the counters are the only
    //  measurement of how hard the source would have to be withheld, and
    //  because deleting it would erase the ability to reproduce Batch 25.
    //
    //  CORRECTION 2026-09-08: that "measurement" claim was false as first
    //  shipped. The counters sat INSIDE the `if (capOn && ...)` test, so with
    //  the switch off they read zero however many cells would have bound.
    //  Fixed by splitting growthWouldLimit/wouldWorstFactor/wouldMaxExcessMps
    //  (computed on every exceeding cell, always) from growthLimited/
    //  worstFactor (what was actually applied). Same split on the erosion
    //  side. Renamed from growthCapEnabled_ at the same time: this ONE switch
    //  governs the erosion clamp as well, so the old name understated it.
    //
    //  The emitted counts are per CELL-EVALUATION. This block runs four times
    //  per physical timestep plus twice at startup: Batch 25's 2,505,301 is
    //  80,398 records over 20,099 steps, and dividing by four is also wrong
    //  because different evaluations see different cells.
    //
    //  It is NOT free when on: it scales adhesionSpeedEff_ as well as Vn_,
    //  which feeds surfaceLossCoeff_ and the platelet sink, and the two
    //  solves do not share one accepted transfer because prePredictor() runs
    //  updateDerivedFields() again between them.

    //- Skeleton (clot material) velocity [m/s]. Eulerian; NOT a displacement.
    //
    //  SOLVED, NOT EVALUATED, since 2026-09-04. It used to be the algebraic
    //  closure Vs = U + (fElastic - fPressure)/K, which measured |Vs| > |U| in
    //  88.3% of clot cells in run 20260904-214345 -- a solid outrunning the
    //  fluid that drags it, which the fluid momentum equation alone forbids.
    //  The residual of the violated mixture balance div(sigma_e) = grad(p) had
    //  nowhere to go but into Vs, because p was solved from the FLUID
    //  continuity and nothing could force grad(p) consistent with sigma_e.
    //  It is now the unknown of tVsEqn, coupled to U through a per-cell 2x2
    //  block inversion and a pressure equation assembled from BOTH momenta.
    //  See research/designs/AM2020_two_velocity_coupled_solve_PLAN_2026-09-04.md.
    volVectorField Vs_;

    //- The skeleton momentum equation. Parallel to tUEqn, assembled in
    //  momentumPredictor() and consumed by correctPressure(). Valid only on
    //  the two-velocity path; every other branch leaves it invalid, which is
    //  how correctPressure() selects its implementation.
    tmp<fvVectorMatrix> tVsEqn;

    //- TRUE SKELETON VOLUME FRACTION INSIDE THE CLOT, phi_s = chi*this.
    //
    //  chi = 1 - q is an INDICATOR, not a volume fraction, and the difference
    //  is the whole reason the first two-velocity attempt could not have
    //  worked. Measured in run 20260904-214345: median q over 6454 clot cells
    //  is 0.0066, and 71.7% of them have q < 0.05. Setting phi_f = q would
    //  make blood VANISH inside the clot (phi_f ~ 0.007), contradicting both
    //  the permeability model -- k = 1e-11..1e-14 m2 means blood seeps through
    //  -- and He, Kim & Ku (source 23), who give water content "more than 95%
    //  in the fibrin matrix".
    //
    //  Default 0.01, sourced: He/Kim/Ku report Wufsus et al. (2013) measuring
    //  platelet volume fraction 1% at k = 1.1e-2 um2 = 1.1e-14 m2, and our
    //  k0pcore is 1e-14 m2. It is an anchored modelling parameter, not a
    //  measurement of this clot -- sweep it, and do not present it as fitted.
    scalar skeletonVolumeFraction_;

    //- STAGE 1 OF THE CONTENT MODEL. Off by default, and that default is
    //  load-bearing: twelve returned two-velocity children never name this
    //  key, and defaulting it on would silently give them a model they never
    //  ran -- exactly the materialCarrierBlend mistake of 2026-09-08.
    //
    //  WHAT IT IS FOR. q = 0.5(c+1) currently carries TWO incompatible
    //  meanings: solid content, through theta = 0.45*max(1-q,0), and a
    //  bounded accessibility weight for the platelet system. Codex proved
    //  those cannot coexist -- an affine complement of compressible conserved
    //  content cannot also be a universally bounded indicator with nominal
    //  density at its zero endpoint -- and run 20260909-040224 measured the
    //  consequence: with alpha EXACTLY ZERO the field still leaves [-1,1] on
    //  the same schedule as with growth on. Stage 1 transports content
    //  itself, so the two meanings can be separated.
    //
    //  STAGE 1 CHANGES NO PHYSICS. Kinetics are off, and theta is REPORTED,
    //  never consumed by permeability, drag or momentum. Its only job is to
    //  answer one falsifiable question: does conservatively transported
    //  content reproduce the legacy chi to solver tolerance? If not, nothing
    //  built on top of it is trustworthy.

    //- Which operator solveContentComponents() assembles. Two words only,
    //  validated at construction -- a typo must not fall through to a
    //  default that looks live.
    //
    //  `content` (default): ddt(theta) + div(theta u_s) = 0, first-order
    //      upwind. The shipped Stage 1 model. This is a DIFFERENT equation
    //      from the legacy chi and is not expected to reproduce it.
    //
    //  `matched`: the exact conservative image of the legacy chi equation --
    //      the captured phaseWeights_ instead of upwind, plus the mapped
    //      source theta_ref*(-0.5*laplacian(mobility,muC) + 0.5*Vn*gradPhase).
    //      T2a in the plan. Its only purpose is to prove the ASSEMBLY, by
    //      making every term that separates the two equations agree; anything
    //      left over is a coding error and nothing else.
    //
    //  Run 20260909-115458 is why both exist. Under `content` equivMax grew
    //  to 1.61e-02, which I had pre-registered as a failure and which is in
    //  fact the declared model difference: 22.1% of equivL1 is the growth
    //  source (closed against the mass budget to four digits) and the
    //  sign-cancelling 77.9% is upwind against vanLeer. `matched` is the test
    //  that criterion should always have been.

    //- Declared volume per bound platelet-equivalent. DECLARED, not fitted
    //  from the failing q trajectory -- fitting it there would bake the
    //  defect being investigated into the replacement.
    scalar vP_;

    //- Vanishing-phase regularisation for the skeleton row of the 2x2 block.
    //  Where phi_s falls below this, Kres = (residualPhiS - phi_s)/deltaT is
    //  added to the skeleton row only, which drives Vs -> U smoothly in the
    //  lumen without perturbing the fluid. This is OpenFOAM's residualAlpha
    //  (momentumTransferSystem.C:611) and Du & Fogelson's "temporarily
    //  increasing theta_b by a small positive constant" -- two independent
    //  codebases, the same fix. Inside the clot Kres ~ 100 1/s against a drag
    //  of 3.3e+05 minimum at run 20260902-231316's k, and 1.008e+07 at the
    //  current main case's -- i.e. negligible, and more so now.
    scalar residualPhiS_;

    //- ANCHOR-INVARIANT SUPPORT FOR THAT REGULARISATION. Default 0 = OFF, in
    //  which case everything below behaves exactly as before.
    //
    //  residualPhiS above is an ABSOLUTE threshold on phi_s, but
    //  phi_s = chi*skeletonVolumeFraction, so the band it covers shrinks as
    //  the anchor rises. Run 20260906-101812 measured the consequence: c never
    //  reaches exactly +1, so at c = 0.99924 chi = 3.8e-04 and
    //    phi_s0 = 0.10 -> phi_s = 3.8e-05, BELOW 1e-4, Kres = 42 1/s, alive;
    //    phi_s0 = 0.45 -> phi_s = 1.71e-04, ABOVE 1e-4, Kres = 0, DEAD.
    //  All 100 of the top cells by |Vs| sat at c > 0.99 with Kres = 0 and
    //  Kd ~ 0.013 1/s, i.e. a diagonal of order 0.01-0.1 1/s against the
    //  clot's 3.9e+05. The dead band held 1.75% of lumen cells at phi_s0 =
    //  0.10 and 6.22% at 0.45.
    //
    //  Set residualChi > 0 and the support is measured on chi instead, so it
    //  does not move when the anchor moves. THE PEAK MAGNITUDE IS DELIBERATELY
    //  UNCHANGED at residualPhiS/deltaT -- the ramp is rescaled, not
    //  strengthened, so this varies the REGION and nothing else. 0.01 is the
    //  chi-support the phi_s0 = 0.01 runs already had (1e-4/0.01), which is
    //  the configuration whose lumen was measured clean.
    //
    //  CAUTION: it is NOT established that this band is the whole story. The
    //  dead band was non-empty at phi_s0 = 0.10 too (489 cells, max |Vs|
    //  0.2234) and that run still survived 25x longer. See STAGE 6.81.
    scalar residualChi_;

    //- THE NETWORK PRESSURE, as a multiple of the local shear modulus.
    //
    //  Run 20260905-113709 conserved the clot to 2.4e-05 and then drove c to
    //  -2.74 at STEP 4: chi reached 1.87, so phi_s nearly DOUBLED, and nothing
    //  in the model could object. elasticStress() is dev(be) only, and `be` is
    //  the elastic left Cauchy-Green of an INCOMPRESSIBLE neo-Hookean with
    //  det(be) = 1 enforced -- there is no bulk modulus anywhere in the
    //  constitutive model. Both-sides diffusion cannot supply one either: its
    //  halves cancel at convergence by construction, so it damps and cannot
    //  restore.
    //
    //  This is Cogan & Guy 2010's p_n (their Eqs. 12, 19) and Du & Fogelson
    //  2018's answer to the same failure in the opposite sign -- Sec. 2.3, a
    //  network velocity that is "in general, not incompressible" would
    //  otherwise "collapse an aggregate to a point".
    //
    //  K_n = networkPressureCoeff * G. A cross-linked gel's osmotic/bulk
    //  modulus is of the same order as its shear modulus, so the DEFAULT OF
    //  1.0 introduces no new number: it reuses GField_ and therefore inherits
    //  the core/shell contrast automatically. That is an anchored modelling
    //  choice, NOT a measurement -- it is a dictionary entry precisely so it
    //  can be swept, and it must not be presented as fitted.
    scalar networkPressureCoeff_;

    //- Close-packing solid fraction. p_n diverges here, which is what stops
    //  compaction rather than a clip on c.
    //  0.6 from Du & Fogelson Eq. 2.25 (theta_b^max), after Wufsus et al.
    //  2013 -- the same paper skeletonVolumeFraction_ is anchored to.
    scalar phiSMax_;

        //- Which p_n(phi_s) form to use: "poleAtMax" (the shipped law, a
        //  divergence at phiSMax) or "cubic" (Cogan & Guy 2010 Fig. 4: a
        //  double root at phi_s = 0, a simple root at phi_s0). The cubic
        //  exists because the pole form applies a FRINGE FORCE -- see
        //  networkPressure() -- and run 20260905-223740 measured it.

        //- S10 FIX 1. What p_n measures compaction AGAINST: "nominal" (the
        //  shipped behaviour, phi_s = theta against the constant phi_s0) or
        //  "deposited" (theta against thetaNetRef_, the content this material
        //  was laid down with). Under "nominal" the seeded diffuse interface
        //  reads as network dilated below phi_s0 and is in tension at t = 0:
        //  run 20260922-104613 measured 0.00725*G of normal wall traction on
        //  step 1, 24 of 56 bonds broken before the flow had loaded anything.

        //- S10 FIX 3. The face interpolation the content components are
        //  transported with: "upwind" (shipped) or "limitedLinear" (the
        //  limitedLinear 1 limiter, computed ONCE from the total theta and
        //  shared by both components, the reference content and the momentum
        //  face value). Upwind added dx*(1-Co) of variance per unit distance
        //  travelled: 39.7 um^2 measured against 40.7 predicted over 57 um.

        //- S10 FIX 2. What carries RELEASED material on the two-velocity
        //  path: "rigidBody" (shipped: the kinematic band-average Ubody) or
        //  "material" (the solved skeleton flux, the same carrier attached
        //  material rides). Ubody read 14 mm/s because its band spans the
        //  throat; the solved Vs said the released clot rolls at ~6 mm/s.

        //- S11 FIX 1. The skeleton (and fluid) momentum equation's form:
        //  "conservative" (shipped: fvm::ddt(phi, U) with a BOUNDED div,
        //  which removes only the div(F) half of the continuity error) or
        //  "phaseContinuity" (also removes the ddt(phi) half: the phase
        //  momentum form of OpenFOAM-13 multiphaseEuler,
        //  MovingPhaseModel::UEqn, fvm::SuSp(-continuityError, U)). Under
        //  the shipped form a DRAINING cell keeps the momentum of the mass
        //  that left, so Vs grows as phi_old/phi_new per step: run
        //  20260923-031942 tracked phi0/phi to 3-17% over 12 steps before
        //  its void cell ran away.

        //- S11 FIX 2. How the 2x2 block is taken to faces: "cellInverse"
        //  (shipped: interpolate the CELL inverse) or "faceAssembled"
        //  (interpolate the block entries, invert on the face: the
        //  facePressureCorrector of multiphaseEuler). A lumen cell's skeleton
        //  diagonal is only Kres ~ residualPhiS/dt, so its inverse is ~1e4x
        //  the clot's; interpolating inverses hands that mobility to every
        //  force on a clot SURFACE face.

        //- S11 FIX 3. How the deviatoric network stress crosses a face:
        //  "linear" (shipped: the average of the two PARTIAL stresses, so a
        //  void next to dense network takes half of its traction) or
        //  "series" (seriesFaceWeights() of the occupancy times the linearly
        //  interpolated intrinsic stress). Also the face coefficient of the
        //  BSD stabiliser, which carries the same stiffness.

        //- S11 FIX 4. How the isotropic network stress acts: "stressDivergence"
        //  (shipped: inside sigma, a cell divergence then INTERPOLATED to
        //  faces -- blind to a 2-cell mode, 4x weak at 3 cells) or
        //  "faceGradient" (-snGrad(Pi)*|Sf| on the face, the phase-pressure
        //  force of multiphaseEuler, momentumTransferSystem::Fs). Requires a
        //  law whose partial pressure is monotone, or the compact force is
        //  anti-diffusive: networkPressureModel convexExcess.

        //- S11 FIX 6. What decides where be relaxes to I: "blood" (shipped:
        //  rate*q^2, so the clot's own diffuse edge forgets its strain in
        //  ~1 us and is combed off by the flow) or "networkAbsent" (only
        //  where no network was ever deposited, thetaNetRef/phi_s0 below
        //  residualChi). Under networkAbsent newly deposited content is mixed
        //  into be stress-free, which is what relaxation used to do for it.

        //- S11: "off" (shipped) or "on". On, maxDeltaT() also bounds dt by
        //  the Courant number of the SKELETON carrier (0.5, as maxCo bounds
        //  the fluid's) and by the diffusion number of the explicit elastic
        //  terms (0.2, under the 2-D explicit limit 0.25). The
        //  Eulerian path had neither: run 20260923-031942's carrier Courant
        //  went to 107 while the fluid's stayed at 0.02. At that run's
        //  ordinary state the Courant and network bounds allow >= 7.8e-07
        //  and >= 2.4e-06 s against its 4.7e-07. The elastic number is the
        //  close one: the clot edge keeps its strain under networkAbsent for
        //  the first time, estimated at 0.13-0.18, i.e. a bound near 5.2e-07
        //  s -- a ~10% margin, so it may bind slightly at the edge. That is
        //  the bound doing its job, not a cost to tune away.

        //- max over cells of D/h^2, 1/s, of the two explicit elastic terms
        //  -- the compact network pressure and the deviatoric stress where
        //  network exists -- measured in the last pressure corrector for
        //  maxDeltaT(). Zero until measured, so it cannot bind on step one.
        scalar explicitDiffusionRate_;

        //- S12. How the interphase drag is weighted in a partial cell:
        //  "legacy" (shipped: Kozeny-Carman at the CELL-AVERAGE solid
        //  fraction, floored at k1*chi) or "intrinsic" (the occupancy times
        //  the network's OWN drag at its intrinsic density a,
        //  ind*(1-a)^2*nu/k_KC(a) -- the diffuse-domain weighting the stress
        //  already has; Li, Lowengrub, Raetz & Voigt 2009 Eq. 2.27, and
        //  multiphaseEuler's dispersedDragModel::K = alpha_d*Ki). Run
        //  20260923-085741: under legacy the calibrated floor bound in 100%
        //  of fringe cells, 10x (shell) to 1000x (core) below the network's
        //  own drag, and that fringe ran away in both children.

        //- S13 FIX A. What weights the convexExcess partial pressure p(a),
        //  a = phi_s0 + theta - thetaNetRef: "depositedOccupancy" (S11 as
        //  shipped: thetaNetRef/phi_s0) or "networkPresence" (1 wherever
        //  network was deposited, ramping to 0 below residualChi -- the same
        //  field beRelaxBasis networkAbsent uses). The additive excess is
        //  ALREADY a partial quantity: a partial cell of occupancy occ
        //  compacted by intrinsic strain eps holds excess occ*phi0*eps, so
        //  p(phi0 + excess) ~ occ*Kn*eps is the diffuse-domain partial
        //  pressure. Weighting it by occ again counted the occupancy twice
        //  (occ^2), which left the fringe with no grid-scale volumetric
        //  resistance: run 20260923-135545, em55 died of it.
        word networkPressureWeight_;

        //- S13 FIX B. The velocity gradient in be's stretch term on the
        //  two-velocity path: "cellGradient" (shipped: fvc::grad(Vs_)) or
        //  "fluxConsistent" (the Gauss gradient of face velocities whose
        //  normal component is the flux that transports be and the content,
        //  exactly the construction beFluxConsistentGradU gives the
        //  one-velocity path). The cell gradient is blind to the grid-scale
        //  compaction the content registers, so that mode had no elastic
        //  restoring stress; the flux-consistent one strains the network by
        //  exactly the compaction its content undergoes.

        //- S14. "off" (shipped) or "criticalStretch": network whose largest
        //  principal stretch sqrt(max eig(be)) reaches networkCriticalStretch_
        //  RUPTURES -- it keeps its content (debris, with packing resistance
        //  and drag) and loses its shear load path and its elastic memory.
        //  Tutwiler et al. 2020 (source 28): "Rupture occurs when a critical
        //  stretch is attained over a representative volume element", 2.8
        //  locally; single fibres 2.5-3.3. Run 20260923-204553 (em58): a
        //  wall tail of network reached principal stretch 3.2-6.2 and its
        //  explicit elastic term blew up.

        //- S14: the critical principal stretch, dimensionless (> 1).
        scalar networkCriticalStretch_;

        //- S14. The stiffness the explicit elastic stability number uses:
        //  "modulus" (shipped: G) or "tangent" (the incremental stiffness of
        //  G*dev(be) along its principal stretch,
        //  G*[(8/9)b1 + (2/9)(b2 + b3)], b the eigenvalues of be -- G*4/3 at
        //  rest, ~G*b1 stretched). em58's solver-form number read 0.087 while
        //  the tangent form was 1.25 at the step it blew up.

        //- S15. networkRupture progressiveDamage: Kliuchnikov et al. 2025
        //  (source 32) Eq. 6, the Kachanov isotropic damage fitted to
        //  fibre-network simulations of fibrin at 2.7 mg/mL,
        //    d(l1) = 1 - exp(-H(l1 - lf) (l1 - lf)/(2 dlf)),  S = S0 (1 - d),
        //  lf = 2.105 (the stretch at which damage begins), dlf = 1.0. d is
        //  irreversible (the running maximum, carried with the material in
        //  ruptured_) and replaces S14's all-or-nothing step, whose recoil
        //  (run 20260924-014332: 0.12-0.21 m/s) was the drag-limited
        //  response to a one-cell, one-step stress release, ~sigma/(h K).
        scalar networkDamageOnsetStretch_;
        scalar networkDamageStretchScale_;

        //- S15. What the core/shell label shell_ IS. "transportedLabel"
        //  (shipped): a separate field carried by phaseSharedConvection and
        //  clipped to [0, 1]. "contentComposition": shell_ = thetaH/theta,
        //  the shell share of the two FCT-transported solid contents --
        //  bounded in [0, 1] by construction, moving exactly with the
        //  material, and changed by growth the way material identity is.
        //  Batch 54: the clipped label had drifted from thetaH/theta by up
        //  to 0.18-0.25 in ~220 core/shell-boundary cells, i.e. k0p (x100
        //  core to shell) wrong there by up to 3x.
        word materialLabelBasis_;

        //- S16 (research/designs/2026-09-24-S16-timed-terminal-rupture.md).
        //  networkRupture terminalRelease: the network's target damage d*
        //  rises irreversibly to d*_law(l1) -- Eq. 6 of source 32 below the
        //  rupture stretch networkCriticalStretch_ (lc), 1 at and past it
        //  (source 32 p. 355: "can no longer support stress") -- and d
        //  RELAXES to d* over networkReleaseTime_ (tau, seconds): backward
        //  Euler of dd/dt = (d* - d)/tau, d_{n+1} = (d_n + r d*)/(1 + r),
        //  r = dt/tau. tau 3.3e-4 s is source 63's single-fibre recoil time
        //  (waiver D4). The release is the rate limit; the event is d*.
        scalar networkReleaseTime_;

        //- S16: whether the progressive part (Eq. 6 below lc) is on under
        //  terminalRelease. Off: d*_law is 0 below lc and 1 at it
        //  ("terminal-only"), which is the only form in which lc <= lf is
        //  allowed.

        //- S16 B. What the skeleton stress floor guards: "intactWeight"
        //  (shipped): max(ind*(1 - d), floor), so d -> 1 is overridden by
        //  the floor; "occupancy": max(ind, floor)*(1 - d), the floor guards
        //  the occupancy only and a fully damaged cell carries no deviatoric
        //  stress. Identical wherever d = 0.

        //- S16 C. How the damage d is carried: "volumeFlux" (shipped): the
        //  label ruptured_ is transported by the volume flux;
        //  "contentPartition": the non-negative partition (theta - D*,
        //  D* - D, D) of the network content, D = d theta, D* = d* theta, is
        //  carried by the content's own transport and limiter, and d, d* are
        //  its quotients. H9: the label gained 23.3% of damaged content in
        //  transport alone.

        //- S16 D (F3). Whether damage reaches the network pressure: "off"
        //  (shipped) or "tensile": p_n -> max(p_n, 0) + (1 - d) min(p_n, 0),
        //  a broken network resists no dilation, its debris still resists
        //  packing (first-principles; design section 2F).

        //- S17 C (research/designs/2026-09-26-S17-frozen-release-implicit-
        //  elastic-startup.md, section 2C). "explicit" (shipped): the BSD
        //  pair carries phi_s,intact*gD*G, the skeleton is not predicted, and
        //  the elastic tangent number bounds dt. "implicitTangent": the pair
        //  carries Gamma_t,f = series(w)_f interp(gD G k_eff)_f dt/rho, a real
        //  Vs predictor is solved with Kres on its diagonal, and the elastic
        //  number leaves the dt rate (the wave CFL and nuNet stay). Needs
        //  networkRupture fibreRelease.

        //- S17b (research/designs/2026-09-27-S17b-slack-fibre-field-units-
        //  wall-face.md, revision 2, APPROVED). Each defaults to S17.
        //  fibreSlack: "off" (S17) or "v1" (A1, Alireza's choice: a fibre
        //  carried to its rest length goes slack -- its taut content
        //  thetaTaut_, C_B and q_P go to 0 at once, the content stays in P2
        //  until the tau law releases it; q_P scales with T on every UNI
        //  contraction). Needs networkRupture fibreRelease.
        //  fibreArithmetic: "shipped" or "fieldUnits" (A2: the six
        //  partition/fibre predictors assembled in field units, rows x dt/V;
        //  the fibre read -- P-LS, W2, G-proj -- only where the carrying
        //  content >= DBL_MIN). Needs damageTransport contentPartition.
        //  skeletonWallFace: "shipped" or "zeroed" (A3: the wall-patch face
        //  term zeroed in BOTH the Vs and the U reconstructs, as
        //  phiSkelOwn_ zeroes its wall flux). Needs twoVelocitySkeleton.
        //- S18, the soft stage (research/designs/2026-09-29-S18-no-frozen-
        //  pressure-content-weight-network-viscosity.md, rev. 10, APPROVED
        //  2026-09-29). Each defaults to S17b. (C, the content weight, is
        //  networkPressureWeight contentRatio, declared with S13's switch.)
        //  frozenPendingPressure: "shipped" (S17: p_used carries q_P/S) or
        //  "none" (A, R1: p_used excludes q_P/S; q_P stays stored and
        //  enters no force). Needs networkRupture fibreRelease.
        //  wallPatchSplit: "shipped" (the split's wall patch keeps dev(be))
        //  or "split" (B, H12: the S17 intact/pending split on the wall
        //  patch). Needs networkRupture fibreRelease.
        //  cplusOrder: "shipped" (passengers() forms X - divX/Vrel) or
        //  "shared" (E, H9: passengers() in tautShare()'s A/B order).
        //  Needs networkRupture fibreRelease.
        //  projSiteLog: no (S18 as built) or yes (S18p, research/designs/
        //  2026-10-01-S18p-em76-attribution-probe.md rev. 2, section 2): the
        //  per-site P-LS counters (PHASE_CLOT_S18P, once per step) and the
        //  raised-cell dump (PHASE_CLOT_S18P_CELL). LOG-ONLY: no field, flux,
        //  solve or dt reads it. Needs networkRupture fibreRelease; serial.
        Switch projSiteLog_;
        //  S19 (research/designs/2026-10-03-S19-block-coupled-step.md rev. 4,
        //  APPROVED 2026-10-04): momentumPressureCoupling "segregated" (the
        //  shipped predictors and correctors, the default) or "blockCoupled"
        //  (one direct UMFPACK solve per step for Ux, Uy, Vsx, Vsy, p;
        //  free clots, two-velocity eulerianSolid releasedCarrier material
        //  implicitTangent, serial). networkViscosity "none" (default) or
        //  "kelvinVoigt" (R3, section 12; blockCoupled only), with
        //  networkViscosityTau tau_v [s] (default 2.5e-4, source 97).
        word momentumPressureCoupling_;
        word networkViscosity_;
        scalar networkViscosityTau_;
        //  S22 (research/designs/2026-10-07-S22-pardiso-block-solver.md rev.
        //  2, APPROVED by Alireza 2026-10-07, sections 2.4-2.6; read only
        //  under blockCoupled): blockLinearSolver "pardiso" (the default, MKL
        //  PARDISO through blockCoupled::PardisoLU) or "umfpack" (P2's
        //  UMFPACK path); blockSolverThreads (>= 0, default 0: MKL_NUM_THREADS
        //  or the physical cores under pardiso, OpenBLAS untouched under
        //  umfpack); blockSolverReproducible (MKL's CNR mode, default
        //  blockCoupled::blockSolverReproducibleDefault).
        word blockLinearSolver_;
        label blockSolverThreads_;
        Switch blockSolverReproducible_;
        //  S19b (research/designs/2026-10-04-S19b-lumen-elimination.md rev.
        //  2, APPROVED 2026-10-04, section 2.6): blockCoupledLumen
        //  "eliminate" (the default: a lumen cell's Vs is not an unknown,
        //  Vs = U there) or "identityRows" (S19a's system exactly: the
        //  lumen identity rows Vs - U = 0). Read only under blockCoupled.

    //- phi_s and phi_f, held as REGISTERED fields rather than recomputed as
    //  temporaries at each use. This is not tidiness: fvm::ddt(rho, vf) reads
    //  rho.oldTime(), and a locally-constructed volScalarField has no stored
    //  old time -- OpenFOAM would hand back a copy of the CURRENT value, so
    //  ddt(phi_s*Vs) would silently lose the d(phi_s)/dt contribution wherever
    //  the clot moves or grows. That is precisely the class of defect that
    //  costs a run and shows up in nothing. Refreshed once per step at the top
    //  of momentumPredictor(), which is after the phase field is solved and
    //  before either matrix is assembled.
    //
    //  AUTO_WRITE because phi_s is the parameter the whole two-velocity
    //  formulation is most sensitive to, and it must be visible in the
    //  returned fields rather than inferred.
    volScalarField phiSkel_;
    volScalarField phiFluid_;

    //- EACH PHASE'S OWN VOLUMETRIC FLUX, u_i & Sf, as produced by the pressure
    //  corrector. phi_ itself now carries the MIXTURE flux
    //  phi_f*u_f + phi_s*Vs -- that is the quantity the shared pressure makes
    //  divergence-free and the quantity material rides on -- so it is NOT the
    //  right flux to convect either phase's own momentum with.
    //
    //  Using phi_ for the fluid convection would convect fluid momentum with a
    //  velocity that is part skeleton. The error is O(phi_s) and confined to
    //  the clot, which is exactly where it would be least visible and most
    //  wrong. multiphaseEuler keeps the same distinction (phase.phi() against
    //  the mixture phi).
    surfaceScalarField phiFluidOwn_;
    surfaceScalarField phiSkelOwn_;

    label skeletonDivergenceRun_; // consecutive residual increases that count
                                  // as divergence of the skeleton fixed point

    // TERMINATION on sustained divergence. Until run 20260814-224138 the flag
    // above was WRITE-ONLY: the guard broke out of the corrector loop, the
    // staggered loop called solveSkeleton() again on the diverged state, and
    // UsPeak went 6.76e-6 m -> 84952 m in one sweep and then SIGFPE'd. 107
    // divergence reports were logged and the run ignored every one of them.
    // The design record claimed this guard "turns divergence into a reported
    // measurement rather than a crash"; it did not, and this is what makes
    // that true.
    //- Constitutive law for the skeleton. "stVenantKirchhoff" (default, the
    //  historical behaviour) or "neoHookean". Only meaningful with
    //  finiteStrain on; the constructor refuses the combination otherwise.

    //- The timestep that the PENDING displacement increment (Us_ - UsPrev_)
    //  actually spans.
    //
    //  UsPrev_ is refreshed once per step, in prePredictor(), and
    //  solvePhaseField() runs BEFORE that refresh. So when the phase field
    //  forms v_s = (Us_ - UsPrev_)/dt, the numerator is the increment of the
    //  PREVIOUS step while runTime.deltaT() is already the CURRENT one. The
    //  two only agree while dt is constant.
    //
    //  Under an adaptive timestep they do not, and the mismatch is a positive
    //  feedback: a smaller dt divides the same fixed increment, so v_s rises,
    //  so the material-Courant limiter cuts dt again. Run 20260815-191231 lost
    //  a factor of 229 in dt in ONE step and then locked at 1.1e-9 with an
    //  apparent v_s of 470 m/s -- 56000x the mean flow -- while the mechanics
    //  underneath was converging normally.
    scalar deltaTIncrement_;

                                  // multiplies the implicit laplacian AND the
                                  // explicit laplacian it is deferred against,
                                  // so the converged solution is unchanged and
                                  // only the contraction rate moves. Raised
                                  // above 1 to stabilise the finite-strain
                                  // correction, which diverged at 18% strain
                                  // in run 20260810-035337. Default 1.

    volVectorField fsiResidual_;  // previous fixed-point residual, for Aitken
    volScalarField HStep_;        // H at the START of the time step. The
                                  // exchange iterates a map that must not
                                  // carry memory, and solveDamage() ratchets
                                  // H = max(H, psi+); without restoring H at
                                  // the top of every sweep, H accumulates the
                                  // maximum over the ITERATES rather than
                                  // taking its value at the converged one, and
                                  // the fixed point is not a fixed point.
    scalar fsiOmega_;             // current relaxation factor
    scalar couplingResidual_;     // reported every step, never assumed
    scalar couplingResidual0_;    // residual after the FIRST sweep of the step.
                                  // With the last one it gives the observed
                                  // contraction (last/first)^(1/(n-1)) - an
                                  // estimate of the iteration's spectral
                                  // radius, which is what distinguishes SLOW
                                  // CONVERGENCE from a FLOOR the iteration
                                  // cannot pass. Run 20260804-031439 stalled
                                  // near 4e-4 and could not tell the two apart.
    label couplingIters_;
    bool couplingConverged_;
    bool mechanicsFinalised_;     // releaseDetachedMaterial runs once per step

    // --- per-phase wall-clock accounting (stage 0 of the Eulerian-solid plan)
    //
    // Why this exists. The cost case for replacing the displacement skeleton
    // rests on an INFERENCE from linear-solve counts in run 20260815-231941:
    // 464538 Us solves against 26520 pressure solves, i.e. 85% of the linear
    // algebra in the skeleton fixed point. Solve counts are not seconds. A
    // GAMG pressure solve and a smoothSolver Us solve on 12500 cells are not
    // the same unit of work, so 85% of solves need not be 85% of the time.
    //
    // The whole plan is priced off that number, so it gets measured before
    // anything is built on it. Accumulated per time step, reported once, and
    // zeroed in prePredictor().
    scalar tPhaseField_;
    scalar tPlatelets_;
    scalar tSkeleton_;
    scalar tDamage_;
    scalar tPermeability_;
    scalar tDerived_;
    scalar tRelease_;
    scalar tStepStart_;           // clock at the top of the step

    // --- discretely consistent platelet interface flux ---
    // DEFAULTS TO FALSE, so a case that does not ask for it gets the old
    // assembly byte for byte and every passed growth gate is unaffected.
    //
    // Why it exists: the surface-loss coefficient delta*(k + Vn - U.n) is not
    // a physical loss term. Expanding the conservative operators shows
    //   d(qw)/dt + div(qUw) = q*Dw/Dt + w*delta*(U.n - Vn),
    // so the (Vn - U.n) part of the coefficient is a COUNTER-TERM that exists
    // only to cancel the interface-motion flux, leaving the physical sink
    // delta*k*w, which is non-negative because adhesionSpeed_ is. That
    // cancellation is exact in the continuum but not in the discretisation:
    // the conservative half is built from face fluxes and a time difference of
    // q*w, while the counter-term is a pointwise coefficient from cell-centred
    // fvc::grad. They agree only to truncation error, and the residue enters
    // the diagonal through fvm::Sp. Run 20260808-001811 measured that residue
    // at -20.378833 1/s against a physical delta*k of about +0.23 1/s - ninety
    // times larger and of the wrong sign - which destroyed diagonal dominance
    // and amplified w until the solve failed at t = 0.0827 s.
    //
    // When true, the counter-term is formed with the SAME discrete operators
    // as the conservative terms, so it cancels to machine precision, and what
    // remains on the diagonal is delta*k >= 0. Derivation in
    // research/designs/AM2020_platelet_flux_consistency_derivation.md.

    // SOLVE FOR PLATELET CONTENT W = q*w INSTEAD OF THE CONCENTRATION w.
    // Default FALSE; false is the old branch byte for byte.
    //
    // WHY. The w equation must divide by q to advance w, and q -> 5e-5 in the
    // clot, so every discretisation error is amplified by ~2e4 per step. Run
    // 20260819-074839 measured this with EVERY source, sink and growth pathway
    // switched off (alpha = 0, k0 = k1 = 0, mobility = 0): the transport
    // operators ALONE drove w from 1 to 64.77 against a physical bound of 1.1.
    // W carries no division by q anywhere and is genuinely conserved.
    //
    // ITS CARRIER IS THE PHASE FIELD'S CARRIER, NOT phi. This is a correction
    // to the first version of this branch, which convected W by phi_ on the
    // argument that phi is solenoidal and a plain conservative scheme is
    // therefore already bounded. That argument is sound about phi and it
    // answered the wrong question.
    //
    // c is not advected by phi. Under advectReleased it is advected by
    //     phiReleased = maskf*(Ubody & Sf) + (1 - maskf)*phiCarrier
    // which inside the clot -- the whole clot, plus the dilation margin -- is
    // the RIGID-BODY flux, because the Brinkman penalty holds the fluid still
    // exactly where the clot is (2.18e-7 against 2.28e-3 m/s, run
    // 20260811-051109). So q moved with the body while W moved with the fluid,
    // and w = W/q was the ratio of two fields carried by two different
    // velocity fields over precisely the region where the clot lives.
    //
    // w == 1 is a valid FIXED POINT of these equations -- with mobility off,
    // Vn vanishes identically at w == 1 -- but the discretisation never let
    // the solution sit there. Run 20260819-231939 measured the consequence:
    // W perfectly smooth and monotone across the clot (0.047 -> 1.013, no
    // cell-to-cell noise anywhere) while ALL the structure in w came from q,
    // with max|w| = 5.36 sitting exactly at q's trough where 1/q = 15. The
    // seed appears on step ONE -- wMax = 1.02126 -- which an instability
    // cannot do and a carrier mismatch must. gate_a_content held it at 0.0211
    // with alpha AND mobility off, i.e. with every non-advective term gone.
    //
    // Physically the carrier fix is not a numerical convenience: platelet
    // content incorporated into the clot is carried by the skeleton, and free
    // platelets in blood are carried by the flow. maskf does exactly that.
    //
    // CONSEQUENCE FOR THE SCHEME, AND IT INVERTS THE OLD GUARD. phiReleased is
    // deliberately NOT divergence-free -- it jumps from zero to Ubody across
    // the mask boundary -- so the conservative form would carry a spurious
    // source and sink. `bounded` subtracts W*div(phiReleased) and leaves
    // u.grad(W), which is the same non-conservative form the c equation uses
    // and the reason c stays in bounds. The guard therefore now REQUIRES the
    // div(phiReleased,Wp) scheme to be token-identical to div(phiReleased,c):
    // same operator, same limiter, same boundedness. Anything else reintroduces
    // the mismatch in a subtler place.
    //
    // THE PRICE, STATED. Non-conservative transport means platelet content is
    // no longer conserved to machine precision. c already pays this for the
    // same unavoidable reason. The error is measured, not assumed -- see the
    // inventory diagnostic -- and it is the honest cost of the fix.
    //
    // STILL OPEN, NOT FIXED HERE. The c equation carries -Vn*phaseGradient and
    // the W equation has no counterpart. Whether growth converting blood to
    // clot should also move platelet content is a MODELLING choice, not a
    // discretisation one, and it is not guessed at here. It does not affect
    // the gate below, which runs at alpha = 0.
    //
    // Derivation, closure for Vn, and the acceptance gates:
    // research/designs/AM2020_platelet_content_formulation_derivation.md

    // The flux the phase field was ACTUALLY advected by on this step, stored by
    // solvePhaseField() and consumed by solvePlatelets() immediately after it
    // (they are adjacent in the step, so there is no lag). Invalid when the
    // phase field was not advected at all, in which case W is not advected
    // either -- consistency includes the case of no transport.
    //
    // tmp rather than autoPtr, and the choice is deliberate: this file already
    // builds tphiMaterial as a tmp<surfaceScalarField> and assigns it from a
    // `new`, so that idiom is PROVEN to compile against the OpenFOAM version
    // this is built with. autoPtr's reset/set spelling has moved between
    // Foundation releases, this solver is compiled remotely, and there is no
    // other autoPtr in the codebase to copy from -- guessing would risk a full
    // Colab round trip to discover a one-word API difference.
    tmp<surfaceScalarField> phaseCarrier_;

    // The FACE WEIGHTS the phase field was actually advected with, captured
    // from the c equation's own scheme immediately before c is solved, and
    // handed to the Wp equation so the two assemble the same matrix.
    //
    // Sharing the flux was not enough and sharing the scheme NAME was not
    // enough. vanLeer weights are a function of the transported field, so c
    // and Wp produced different matrices from identical scheme strings on one
    // shared flux. Run 20260822-103655: |Wp - q| = 2.27e-04 under vanLeer
    // against 5.85e-12 under field-independent `linear` weights, with the
    // linear result FLAT across the run rather than accumulating. That is the
    // whole defect, and this field is the fix.
    tmp<surfaceScalarField> phaseWeights_;

    // THE FACE WEIGHTS THE CONTENT EQUATION WAS ACTUALLY ADVECTED WITH.
    //
    // Same defect class as phaseWeights_ above, one equation further on, and
    // measured rather than argued: solveContentComponents() transports theta
    // with first-order upwind while momentumPredictor() rebuilt the same
    // solid fraction on the same faces with fvc::interpolate -- linear. Two
    // face values of one field, at one time level, multiplying one flux
    // (phiSkelOwn_). Run 20260919-083453, at t = 4e-04 s:
    //
    //     max|theta_f^linear - theta_f^upwind| = 2.8071e-02  (6.24% of
    //     theta_ref), flux-weighted mean relative error 1.61%, and 26.5% of
    //     the solid-carrying faces past 20% relative.
    //
    // 4.0x the C1 violation (c1Max = 6.96e-03 on that same run) that the
    // whole weight migration exists to remove.
    //
    // Rudman's condition -- the flux that advects momentum must be the flux
    // the phase equation transported with. Not the same scheme NAME; the same
    // numbers on the same faces, which is why this is a captured field and
    // not a scheme string. See
    // research/designs/2026-09-19-OI6-correctness-criterion.md.
    //
    // UPWIND WEIGHTS ARE FIELD-INDEPENDENT (pos0(faceFlux)), which is why
    // thetaC and thetaH can share them and why this capture is exact. Do NOT
    // extend the same trick to a limited scheme: that is precisely the
    // vanLeer defect phaseWeights_ above exists to repair.
    tmp<surfaceScalarField> contentFaceWeights_;

    //  S10 FIX 3 (contentTransportScheme fct): the flux-corrected-transport
    //  limiter, one value in [0,1] per internal face, SHARED by every field
    //  the content equations transport (thetaC, thetaH, their total,
    //  thetaNetRef) and inherited by Wp. The transported face value is
    //  upwind + lambda*(linear - upwind), so the effective weights are
    //  w_up + lambda*(w_lin - w_up). contentFaceWeights_ keeps its meaning:
    //  the weights the IMPLICIT matrices were assembled with, upwind under
    //  fct, which is what keeps every predictor an M-matrix. Absent unless
    //  fct ran this step; cleared beside contentFaceWeights_.
    tmp<surfaceScalarField> contentFctLambda_;

    // q AT THE OLD TIME LEVEL, captured before solvePhaseField advances c, and
    // used for the EXPLICIT diffusion source in the Wp equation.
    //
    // fvc::laplacian(gamma, w) is discretely EXACTLY zero whenever w is
    // uniform, so the diffusion term can only inject error if w departs from 1
    // at the moment it is evaluated. w = Wp/qEff, and updateDerivedFields()
    // runs between solvePhaseField() and solvePlatelets(), so that quotient
    // was Wp at n over qEff at n+1 -- off by exactly one step's change in q,
    // which is the ONLY route by which w can be non-uniform there.
    //
    // Measured, run 20260823-002216, once the limiter fix removed the larger
    // error that had been masking this one: |Wp - q| = 2.589e-11 with D = 0
    // against 4.3755e-05 with D = 1e-10, on cases differing in nothing but D.
    // Evaluating the whole explicit source at the old level restores
    // Wp^n = q^n => w == 1 exactly, and is also the consistent choice for an
    // explicit term under Euler.
    tmp<volScalarField> qOldPlatelet_;

    // Timestep-scoped accepted content state. These objects deliberately live
    // on the solver rather than inside solvePhaseField(): once the phase solve
    // is visited on multiple PIMPLE outers, a local checkpoint would capture
    // the preceding trial and turn nonlinear iteration into another physical
    // reaction step. The current call placement remains first-outer-only;
    // this is state-ownership infrastructure, not outer integration yet.
    tmp<volScalarField> thetaCStepAccepted_;
    tmp<volScalarField> thetaHStepAccepted_;
    //  S10: the deposited reference content rolls back WITH the content it
    //  references. Restored at exactly the sites thetaC_/thetaH_ are, so a
    //  repeated outer iteration cannot add one growth increment to theta
    //  once and to thetaNetRef_ twice.
    tmp<volScalarField> thetaNetRefStepAccepted_;
    tmp<volScalarField> nPStepAccepted_;
    //  S16 C: the damage partition rolls back with the content it
    //  partitions, at exactly the sites thetaC_/thetaH_ do (red-team
    //  Change 4). With nPhaseCorrectors 2 a missed roll-back would add one
    //  deposit to theta once and to the partition twice.
    tmp<volScalarField> thetaIntactStepAccepted_;
    tmp<volScalarField> thetaPendingStepAccepted_;
    tmp<volScalarField> thetaBrokenStepAccepted_;
    //  S17 A: the pending fibre rolls back with the partition that weights
    //  it (design 2A, section 5 writer sites): content C_B = thetaP B_P,
    //  q_P = thetaP p_P, and the axis and stretch the tie-break and UNI read.
    tmp<volSymmTensorField> thetaPendingBStepAccepted_;
    tmp<volScalarField> thetaPendingPStepAccepted_;
    tmp<volVectorField> pendingAxisStepAccepted_;
    tmp<volScalarField> pendingStretchStepAccepted_;
    //  S17b A1: the taut share rolls back with the partition.
    tmp<volScalarField> thetaTautStepAccepted_;

    //  S16 per-step diagnostics (design section 6), set in the constructor
    //  body and reset at the start of every step. The three stress-weight
    //  sites record what they USED, so a site that stops using the occupancy
    //  weight -- or stops recording -- shows in PHASE_CLOT_S16.
    scalar s16LawReleaseMax_;
    scalar s16LawRate_;
    scalar s16PendingMax_;
    scalar s16DeltaDMax_;
    scalar s16DampedStiffMax_;
    mutable scalar s16WeightAbsDevMax_;
    //  S16 C+ (13c): largest donor outflow fraction, mu over the step's
    //  correctors (min; |E|-weighted mean of the last), and S16m, the
    //  deposit mirror's max|dP1 - dtheta|/max theta (16 R5).
    scalar s16FoutMax_;
    scalar s16MuMin_;
    scalar s16MuMean_;
    scalar s16MirrorDevMax_;
    mutable label s16WeightSites_;

    //  S17 per-step diagnostics (design section 6), set in the constructor
    //  body and reset at the start of every step. Read by assess_case.py.
    //  G-tr: max over transferring cells of b1(be)/lc^2, read BEFORE
    //  min(l1, lc); G-rot: the median angle/(dt omega); G-eig: the largest
    //  relative eigenvalue change of B_P across the rotation; G-uni: cells
    //  whose lf rose across the rotation-and-stretch kernel; G-proj: P-LS
    //  results that raised the energy against the blend; S17split: per site
    //  max|M_used - M_split| (site 0, 1: the deviatoric tensor per unit
    //  weight; site 2: the tangent stiffness per unit weight); P2/W2: the
    //  stress measure s/s_c; W1: the misalignment of n against be's major
    //  axis in mixed cells.
    scalar s17TransferB1Max_;
    label s17Transfers_;
    scalar s17RotRatioMedian_;
    label s17RotCells_;
    //  G-rot (redefined after verification 17): the median |L|dt of every
    //  candidate cell, and their count; rotCells counts the |L|dt <= 1e-3
    //  cells the ratio is read over.
    scalar s17RotLdtMedian_;
    label s17RotCandidates_;
    scalar s17EigRelMax_;
    label s17UniUp_;
    label s17ProjCount_;
    label s17ProjRaised_;
    scalar s17ProjExcessMax_;
    label s17ProjTies_;
    mutable scalar s17SplitDevMax_[3];
    mutable label s17SplitSites_;
    scalar s17P2Max_;
    scalar s17W2Max_;
    scalar s17W1MaxDeg_;
    label s17W1Over5_;
    label s17W1Over10_;
    label s17ResetCells_;
    scalar s17PendingPressureMax_;

    //  S17b per-step diagnostics (design section 2 "New logs", section 4).
    //  V1 slack events (design section 8 d): count, content moved, THE
    //  GATE |dd|; conformance: max mu over events with lf_in != 1, the
    //  lf_in = 1 count, max |q_P| after; the WATCH on the pre-event residual
    //  T(lf_in) theta_T/S and |q_P|/S and their ratios to T(1/mu).
    label s17bSlackEvents_;
    label s17bSlackAtRest_;
    scalar s17bSlackContent_;
    scalar s17bSlackDdMax_;
    scalar s17bSlackMuMax_;
    scalar s17bSlackQpMax_;
    scalar s17bSlackPreTMax_;
    scalar s17bSlackPreTRatioMax_;
    scalar s17bSlackPreQpMax_;
    scalar s17bSlackPreQpRatioMax_;
    //  the site-tagged max lf over carrying content >= DBL_MIN (12 change 6)
    scalar s17bMaxLfTransport_;
    scalar s17bMaxLfRotation_;
    scalar s17bMaxLfLaw_;
    //  projRaised by presence (network deposited above residualChi or not)
    label s17bProjRaisedPresent_;
    label s17bProjRaisedAbsent_;
    //  per-cell accumulators of ln mu and min(0, ln mu) since the taut
    //  population was born (H-PA2), and the time of the last slack event
    scalarField s17bSumLnMu_;
    scalarField s17bSumMinLnMu_;
    scalarField s17bLastSlackTime_;

    //  S17 (design section 6, change 5): the per-step outlet-face second
    //  difference ||phi_n - 2 phi_{n-1} + phi_{n-2}|| and its clot-surface
    //  twin on the skeleton flux, from the last two steps' values.
    scalarField s17OutletPhi1_;
    scalarField s17OutletPhi2_;
    scalarField s17SurfPhi1_;
    scalarField s17SurfPhi2_;
    label s17SecondDiffSteps_;
    //  S17b (12 change 2): one step further back, for the correlation c_n
    //  of successive face-wise second differences.
    scalarField s17OutletPhi3_;
    scalarField s17SurfPhi3_;
    //  S18 F (design section 2): the skeleton-carrier Courant limit's dt as
    //  maxDeltaT() last computed it (mutable: maxDeltaT() is const), and
    //  the wall-patch split's largest departure from its adjacent cells
    //  this step (B's conformance; 0 under wallPatchSplit split).
    mutable scalar s18CarrierDtLimit_;
    mutable scalar s18WallSplitDevMax_;
    //  S18p (projSiteLog yes only): per-step P-LS counters by site, [0] the
    //  rupture-law transfer (T), [1] the transport projection (P); their
    //  sums are S17's projCount and projRaised. The dump's quotas: 5 raised
    //  cells per site per 0.25 ms window, every raised network-present cell
    //  up to 200 per run. The passengers kernel's locals of this step (E's
    //  debug out-argument; nCells = 0 where it was not filled).
    label s18pCount_[2];
    label s18pRaised_[2];
    label s18pRaisedPresent_[2];
    label s18pRaisedAbsent_[2];
    scalar s18pExcessMax_[2];
    label s18pWindow_[2];
    label s18pWindowDumped_[2];
    label s18pPresentDumped_;
    damagePartitionKernel::PassengersDebug<scalar> s18pKernelDbg_;
    //- S19: the UMFPACK factorisation, its symbolic part kept while the
    //  block's CSR pattern is unchanged (blockCoupled with blockLinearSolver
    //  umfpack only; empty otherwise).
    autoPtr<blockCoupled::UmfpackLU> blockLU_;
    //- S22: the PARDISO factorisation and its MKL, loaded in the constructor
    //  before PHASE_CLOT_S19_CONFIG (blockCoupled with blockLinearSolver
    //  pardiso only; empty otherwise).
    autoPtr<blockCoupled::PardisoLU> blockPardiso_;

    label nPhaseCorrectors_;

    scalar previousInventory_;

    //- The clot field's own balance needs its own history. previousInventory_
    //  is the PLATELET inventory (q*w) and says nothing about clot material;
    //  conflating the two is the error the 2026-08-27 audit corrected.

    //- THE GROWTH THE TRANSFER ACTUALLY APPLIED, in c currency per second:
    //  int (2/theta_ref)*(thetaNew - thetaStar) dV / dt.
    //
    //  OVERWRITTEN, NEVER ACCUMULATED. solvePlateletContent() runs once per
    //  phase corrector and fvm::ddt(c_) rebuilds c from oldTime at the top of
    //  each one, so corrector 1 DISCARDS corrector 0's increment before
    //  applying its own. Summing here would report a growth the field never
    //  received, and PHASE_CLOT_MASS would close on a number that is not the
    //  source. Reset to 0 at the start of each solvePhaseField() so a step
    //  whose transfer never ran reports zero rather than the last step's.
    //- Content removed by releaseDetachedMaterial() since the last balance
    //  report, in m^3 of theta. A CONTENT DELETION MUST BE BOOKED: zeroing
    //  thetaC/thetaH takes material out of a conserved field, and
    //  contentResidual would otherwise report it as a conservation failure
    //  rather than as the modelled release it is. Reset in
    //  writeBalanceDiagnostic() after the report, so one line covers exactly
    //  one timestep however many correctors ran.
    scalar detachedContentStep_;

    //- Domain content integral at the previous balance report, for the
    //  closing content balance that replaced PHASE_CLOT_MASS in S6c-2.
    scalar previousContentVolume_;

    scalar transferClotGrowth_;

    //- The committed transfer integral int dB dV, carried out of
    //  solvePlateletContent() so writeBalanceDiagnostic() can report the
    //  stoichiometry against the transfer that happened rather than against
    //  plateletUptake, which has no N in it and is therefore blind to the
    //  free-platelet depletion the transfer causes. OI-12: across run
    //  20260917-003709 this fell 21.76% while plateletUptake moved 0.025%
    //  and stoichNorm held to 0.054%.
    scalar dBSumCommitted_;

    //- int deltaTheta_ dV SNAPSHOTTED WHERE THE TRANSFER READS IT, which is
    //  not where the report is written. OI-11: deltaThetaIntPost was
    //  bit-identical to deltaThetaInt in all 272 records of run
    //  20260917-003709, because both are postSolve values and the value the
    //  transfer consumed was set by the earlier updateDerivedFields() call
    //  and is gone by then. Taken in solvePlateletContent() where bindRate is
    //  built, so the instrument reports the geometry the reaction used.
    scalar deltaThetaIntAtTransfer_;

    // --- THE PHASE-FIELD CLIP (a MEASUREMENT, not a fix) -----------------
    //
    // Nothing in this solver bounds c. mobility_ is 0.0 in every main-case
    // child, so clotDiffusion is identically zero and the double well is
    // inert -- c is a pure advected scalar carried by `div(phiReleased,c)
    // bounded Gauss vanLeer`, and vanLeer is TVD only for a divergence-free
    // flux. phiReleased is NOT the pressure-projected mixture flux.
    //
    // Run 20260906-004750 measured the consequence: at t = 1.225e-03,
    // cMin = -1.619 in 942 cells, of which 881 have all four neighbours also
    // below -0.9. The overshoot is 94% INTERIOR, not interfacial, so it is
    // not the one- or two-cell artefact a TVD limiter leaves at a front.
    //
    // This switch exists to decide between two readings of that, BEFORE
    // anyone rewrites the interface capture:
    //
    //   (A) a scheme overshoot that saturates -- clipping costs almost
    //       nothing, and a bounded-scheme repair on c is sufficient;
    //   (B) a genuinely convergent advecting flux -- clipping removes real
    //       material every step, and ANY limiter would silently DISCARD what
    //       it clips. Then the pressure-equation volume source has to come
    //       first and the interface rewrite second.
    //
    // NOT MULES, AND NOT AN ALPHA FIELD. STAGE 6.79 ruled that route out on
    // its PREMISE, not on taste: Cogan & Guy 2010 model the clot as
    // INTERPENETRATING continua whose individual phase velocities are
    // explicitly not divergence-free, and Wufsus 2013 measure a
    // plasma-concentration fibrin gel at fiber volume fraction ~0.005, i.e.
    // 99.5% liquid. c is an indicator inside a mixture, not a VoF fraction
    // between immiscible fluids, so MULES is a bounded answer to a question
    // this model does not ask. See
    // research/literature/AM.2020.0056-19/MISCIBILITY_AND_MODEL_CLASS.md.
    //
    // The discriminator is clipVolumeCum_ against the instantaneous excess
    // already measured, 7.10e-18 m^3 at t = 1.225e-03. (A) predicts the
    // cumulative stays within a few times that. (B) predicts it runs away.
    //
    // DEFAULT FALSE. With it false not one arithmetic operation changes and
    // every other child is bit-identical to run 20260906-004750.
    bool clipPhaseField_;

    //- Clot volume (int chi dV) REMOVED by clipping c up to -1, and ADDED by
    //  clipping c down to +1, this timestep and cumulatively. Reported, never
    //  read back: the clip is instrumented so that its cost is a measured
    //  number rather than an inference from a residual.
    scalar clipVolumeLowStep_;
    scalar clipVolumeHighStep_;
    scalar clipVolumeCum_;
    label clipCellsLowStep_;
    label clipCellsHighStep_;

    //- The extreme values of c SEEN BY THE CLIP, before it acts. Without
    //  these, PHASE_CLOT_PHASE_BOUNDS would report the clipped field and the
    //  run would look clean while the overshoot it is measuring is invisible.
    scalar cMinPreClip_;
    scalar cMaxPreClip_;

    // --- CORE/SHELL MATERIAL REGIONS (source 33, Xu et al. 2017) ---
    //
    // DEFAULTS TO FALSE, and with it false every material field below is
    // UNIFORM and equal to the dimensionedScalar it was built from, so the
    // arithmetic is a multiply by the same number in every cell and nothing
    // established so far moves.
    //
    // Why it exists: Xu et al. get their tear from a two-region clot -- a
    // dense core of highly activated platelets and fibrin (modulus 40 Pa,
    // permeability 1e-14 m2) under a loose shell of less activated platelets
    // (4 Pa, 1e-13 to 1e-11 m2). The core holds its shape at every shear rate
    // they run while the shell is torn off downstream, and that contrast IS
    // the mechanism. A single homogeneous material cannot produce it: with one
    // modulus and one toughness the only way to make anything fail is to lower
    // Gc until the whole body gives at once, which is what run 20260814-055141
    // did.
    bool coreShell_;

    //- Material region indicator: 1 shell (and everything outside the clot),
    //  0 core. Written by the initialiser, read here, AUTO_WRITE so it is
    //  visible in ParaView as the thing that decides which material is where.
    //  Fixed in time on this fixed mesh, and material ADDED BY GROWTH is
    //  shell -- which is physically right, since newly captured platelets are
    //  the less activated ones.
    volScalarField shell_;

    //- S14: the ruptured fraction of the network in a cell, 0 (intact) to 1,
    //  a MATERIAL label carried with the network exactly as shell_ is
    //  (advectRuptureLabel(), the same carrier and operator). Set to 1 where
    //  the network reaches the critical stretch; diluted by new, intact
    //  deposits; zeroed with released content. The damage field Dg_ is
    //  Eulerian (solved from a history H_ that is not transported), so a
    //  rupture written there would stay behind a moving clot.
    volScalarField ruptured_;

    //- Core and shell SOLID VOLUME PER MIXTURE VOLUME, both non-negative.
    //  Two contents rather than one content plus a clipped passive label,
    //  because core identity, growth allocation and erosion eligibility all
    //  need to survive transport. theta = thetaC_ + thetaH_ is ALGEBRAIC --
    //  there is deliberately no third equation for it, so the components
    //  cannot drift away from their own sum.
    //
    //  Composition s = thetaH_/theta is only a passive material label in the
    //  SOURCE-FREE case: the balances give theta*D_s(s) = S_h - s*S_theta, so
    //  depositing shell into a mixed cell changes s. That is why carrying an
    //  unchanged shell label while growing total content does not preserve
    //  material identity, and it is why these are transported and shell_ is
    //  not simply reused.
    bool thetaCHeaderPresent_;
    bool thetaHHeaderPresent_;
    bool nPHeaderPresent_;
    volScalarField thetaC_;
    volScalarField thetaH_;

    //  S10 FIX 1: THE DEPOSITED REFERENCE CONTENT. Transported on the same
    //  carrier, with the same face weights, as thetaC_/thetaH_, but in the
    //  NON-conservative form, so it does not take up compression: where
    //  div(carrier) = 0 it moves exactly as theta does, and where the network
    //  compacts theta rises and this does not. Every growth increment and
    //  every deletion theta receives, it receives too, so new network is laid
    //  down stress-free and removed network takes its reference with it.
    //  theta - thetaNetRef_ is therefore the content gained by compaction
    //  alone -- the quantity the network pressure is meant to resist.
    //  Initialised to theta at the start of the run when not read from disk.
    bool thetaNetRefHeaderPresent_;
    volScalarField thetaNetRef_;

    //  S16 C: THE DAMAGE PARTITION of the network content, carried only
    //  under damageTransport contentPartition:
    //    thetaIntact_  = theta - D*   (not targeted by the rupture law)
    //    thetaPending_ = D* - D       (targeted, not yet released)
    //    thetaBroken_  = D            (released: broken network, debris)
    //  All three non-negative by construction (implicit upwind M-matrix
    //  predictor + the shared Zalesak correction). d = D/theta and
    //  d* = D*/theta are quotients of non-negative sums, so 0 <= d <= d* <= 1
    //  holds in floating point; where the partition is empty d = d* = 0 (a
    //  cell with no network has no broken fibres). ruptured_ is d, derived.
    volScalarField thetaIntact_;
    volScalarField thetaPending_;
    volScalarField thetaBroken_;

    //  S17 A: THE PENDING SHARE AS A FIBRE, carried only under networkRupture
    //  fibreRelease (fibreReleaseKernel.H). Contents ride the C+ partition
    //  with thetaPending_ (docket 12 section 1; 08/10: q_P's flux is p_P of
    //  the donor times thetaP's limited flux):
    //    thetaPendingB_  = thetaP B_P  (content-weighted fibre tensor)
    //    thetaPendingP_  = thetaP p_P  (the frozen pending tension, <= 0)
    //  and the fibre itself, set by P-LS after every transport and transfer
    //  and advanced by the rotation and UNI once per step:
    //    pendingAxis_    = n (the tie-break reads the previous one)
    //    pendingStretch_ = lf (never increases between transfers)
    //  Written, and read back all-or-none: pendingAxis, pendingStretch,
    //  pendingPressure (p_P) and pendingB (B_P = fibre(n, lf), derived).
    volSymmTensorField thetaPendingB_;
    volScalarField thetaPendingP_;
    volVectorField pendingAxis_;
    volScalarField pendingStretch_;
    volSymmTensorField pendingB_;
    volScalarField pendingPressure_;
    //  S17b A1 (V1): thetaTaut_ = the TAUT share of the pending content,
    //  theta_T <= thetaPending_ exactly; C_B = theta_T B_T, q_P = theta_T p_P
    //  under fibreSlack v1. Written, and read all-or-none with the fibre.
    volScalarField thetaTaut_;

    // ---- STAGE 2: FREE PLATELET NUMBER AND THE PAIRED TRANSFER ----
    //
    // OFF BY DEFAULT AND INDEPENDENT OF contentModel_'s operator choice.
    // This is the isolated verification path; it does not feed drag,
    // permeability or momentum, and it must not be switched on in the same
    // case as the `matched` legacy growth source (both would add content --
    // see checkPlateletConfiguration()).

    //- Experimental, default-off PIMPLE-outer reconstruction path. Only the
    //  isolated platelet-content gate enables this until runtime and rollback
    //  gates pass; legacy phase/platelet ordering remains unchanged.

    //- Test-only forced rejection at the final outer. Default off and valid
    //  only with the isolated outer transaction path.

    //- Test-only transaction-attempt identity. Zero preserves every existing
    //  case; nonzero is valid only in the isolated outer transaction path.
    label plateletAttempt_;

    //- WHICH FLUX CONVENTION THE SUPPLIED CARRIER OBEYS. No default: the
    //  distinction is the defect that made the first draft of this equation
    //  wrong, so it is declared per case and validated, never inferred.
    //
    //    intrinsic   : the carrier is u_f & Sf. F_eps = interpolate(eps)*carrier.
    //    superficial : the carrier is ALREADY eps*u_f & Sf. F_eps = carrier.
    //
    //  Advecting bulk N with a superficial flux is an EXACT factor of eps
    //  too small -- 0.55x at eps = 0.55 -- and it is invisible in uniform
    //  fields because both divergences vanish. Hence nP_ (not NP_) is the
    //  transported unknown and T7 uses a non-uniform pulse.

    //- Effective pore diffusivity of free platelets, m^2/s. D_eff = 0 is a
    //  SUPPORTED transport-isolation case, not a degenerate one.
    scalar plateletDiffusivity_;

    //- DEPOSITED-SOLID VOLUME YIELD, m^3 per consumed platelet-equivalent.
    //  DELIBERATELY NOT vP_. vP_ is the material volume one platelet
    //  occupies; Y is the solid volume its binding deposits. The two are
    //  numerically equal ONLY under the declared platelet-equivalent
    //  verification specialisation Y = vP, which is a constitutive statement
    //  and not a conversion from platelet count to wet clot volume. Y is
    //  constant here: a varying Y makes Y*N + theta non-conserved, because
    //  grad(Y) and advection of Y add terms.
    scalar plateletYield_;

    //- First-order binding rate lambda, 1/s. R+ = lambda*N.
    //  ONLY consulted when plateletBindLocalisation_ == "uniform". Under
    //  "interface" the rate is a FIELD, kEff*interfaceDelta_, and this scalar
    //  is not read at all -- see plateletBindLocalisation_.
    scalar plateletBindRate_;

    //- WHERE the accepted transfer is allowed to deposit. Validated at
    //  construction against an explicit list, because lookupOrDefault returns
    //  the default for a TYPO just as readily as for a missing key, and the
    //  two modes differ by a factor of 52 in how much solid lands in the
    //  lumen (run 20260915-073208: 86.5% vs 1.7%).
    //
    //    "uniform"   -- lambda in every cell. The pre-2026-09-15 behaviour.
    //                   Retained ONLY so em29 can be reproduced exactly; it
    //                   contradicts AM2020-phase-clot-v1 eq. (P) and no
    //                   morphology claim may be made from it.
    //    "interface" -- kEff*interfaceDelta_, the specified surface-delta
    //                   localisation, matching what the legacy platelet sink
    //                   already does at :8309.

    //- Adhesion speed kEff for the interface localisation, m/s. Negative
    //  means "reuse adhesionSpeedEff_", the v2 closure the legacy sink uses,
    //  so the two platelet representations cannot silently disagree.
    scalar plateletAdhesionSpeed_;

    //- Basis for the momentum/pressure phase weights phiSkel_ and phiFluid_.
    //  This is C1 and C2 of the phase-volume coupling specification
    //  (research/designs/2026-09-14-phase-volume-coupling-spec.md): the
    //  weights that decompose the mixture flux must be the face interpolants
    //  of the SAME field whose transport equation defines phase volume.
    //
    //    "legacy"  -- solidFraction() = min(max(1-q,0)*theta_ref, 0.9), the
    //                 algebraic map of the INDICATOR. The pre-2026-09-15
    //                 behaviour; retained so em30 reproduces exactly.
    //    "content" -- theta = thetaC_ + thetaH_, the conserved content, which
    //                 is what the platelet equation already stores against
    //                 (epsP_ = 1 - theta) and what the accepted transfer
    //                 grows. Satisfies C1 and C2 together, because phiSf and
    //                 phiFf at :9706-9707 are interpolants of these two
    //                 fields and of nothing else.
    //
    //  ORDERING CONSTRAINT, MEASURED NOT ASSUMED. "content" is only safe on
    //  the LOCALISED deposition path. Migrating it on em29's unlocalised
    //  path would put theta = 4.18e-05 into 19119 lumen cells where
    //  solidFraction() is exactly 0 -- and elasticStress()'s note at :3970
    //  records a run (20260905-191135) killed by a lumen weight of 1e-04,
    //  because out there the skeleton diagonal is only Kres ~ 67 1/s against
    //  3.3e+05 inside the clot, so any surviving force is divided by
    //  something ~5000x too small. em30's localised theta in those same
    //  cells is 2.66e-13, nine decades under that precedent.

    //- Override for the ordering constraint above: run "content" weights on
    //  an unlocalised deposition path anyway. Exists so the refusal can be
    //  lifted deliberately and on the record, never by accident.

    //- Basis for the solid fraction that weights the network-pressure FORCE:
    //  networkPressure()'s argument and the ISOTROPIC elastic term.
    //  Separate from phaseWeightBasis_ on purpose -- that one sets the flux
    //  and momentum weights (C1/C2), this one sets a force, and keeping them
    //  apart is what makes each run a one-key diff.
    //
    //    "legacy"  -- solidFraction(), as before.
    //    "content" -- the same clamp(theta) the weights use.
    //
    //  APPLIES TO THE ISOTROPIC TERM ONLY. The DEVIATORIC weight stays
    //  max(chi, skeletonSolidFloor_) and must never follow this key: chi
    //  reaches 1.0022 where theta reaches 0.450977, so substituting there
    //  would rescale the deviatoric stress by theta_ref = 0.45 -- a
    //  constitutive recalibration, not a representation change -- and would
    //  drop the floor that keeps the operator non-degenerate outside the
    //  clot. That is the error run 20260905-191135 died of; see section 4 of
    //  the consumer map.

    //- Which face value of the solid fraction the SKELETON MOMENTUM equation
    //  advects with -- phiSkelFlux, and nothing else.
    //
    //    "legacy" -- fvc::interpolate(phiSkel_), i.e. linear, as before.
    //    "shared" -- the face weights solveContentComponents() actually used.
    //
    //  DEFAULT "legacy" so that every child on disk reproduces bit for bit
    //  with no case edit; create_cases.py writes the key explicitly for every
    //  new child, so the default decides nothing in practice. Retired with
    //  the other basis keys in S7.
    //
    //  APPLIES TO phiSkelFlux ONLY, and this is not an oversight.
    //  phiSf/phiFf in correctPressure() DECOMPOSE a flux, they do not advect
    //  anything, and they carry an independent requirement -- phiSf + phiFf
    //  == 1 on every face, which is what makes grad(p) cancel when the two
    //  momentum rows are added and which G4 verifies at 4.39e-16. Linear
    //  satisfies it identically because phiFluid_ = 1 - phiSkel_. Upwind
    //  would NOT: the fluid's donor cell is set by phiFluidOwn_ and the
    //  solid's by phiSkelOwn_, so on any face where those two fluxes have
    //  opposite sign the donors differ and the partition of unity fails.
    //  The criterion binds ADVECTIVE fluxes; leave the split alone.
    //
    //  phiFluidFlux is likewise untouched: the fluid volume is algebraic
    //  (eps = 1 - theta), no equation transported it, so there is no "flux it
    //  was advected with" to inherit. That is a separate question and it is
    //  not this key's.

    //- S6a. Which field DEFINES the fluid fraction q.
    //
    //    "legacy"  -- q = 0.5*(c + 1), the phase field's own definition.
    //    "content" -- q = 1 - (thetaC + thetaH)/theta_ref.
    //
    //  THIS IS THE KEY THAT ACTUALLY DELETES c, and it is separate from the
    //  deletion for a reason. The theta-native design justifies S6 with "c is
    //  a passenger", which is true of the FRONT SOURCE and false of q: q is
    //  literally defined as c at updateDerivedFields(), and q is read at 53
    //  sites in 17 functions -- updatePermeability, solveSkeleton,
    //  solvePlatelets, momentumPredictor, phaseIndicator, solidFraction.
    //  Deleting c without this key would not remove a passenger, it would
    //  leave 53 consumers with no definition.
    //
    //  NOT CLAMPED, EITHER SIDE, DELIBERATELY. q = 0.5*(1 + c) is unbounded
    //  today -- c reaches -1.00436, so q reaches -2.2e-03 -- and the content
    //  form must inherit exactly that freedom or the swap would be a
    //  representation change PLUS a silent bounding, which is two changes.
    //  phaseIndicator()'s max(.., 0) is deliberately NOT reused here: that
    //  guard exists because a negative indicator reaches an fvm::Sp on the
    //  momentum diagonal, which is a different site with a different need.
    //  The bounds are measured instead -- PHASE_CLOT_QBASIS every step.
    //
    //  PRICED BEFORE IT WAS BUILT, on run 20260919-083453's own fields: the
    //  two forms differ by at most 0.004393 at t = 2e-04 s and 0.015478 at
    //  t = 4e-04 s, with 0 and 342 cells past 1e-2. The deviation GROWS 3.5x
    //  over the second half of a 0.4 ms window and is not settling. That is
    //  S4's G4 "3.1% wider theta profile" in the units that matter.
    //
    //  WHICH PROFILE IS RIGHT IS STILL OPEN, and OI-6's criterion did not
    //  settle it -- that criterion is about the momentum flux. c is
    //  Cahn-Hilliard and its width is actively maintained; theta has no
    //  sharpening term. So this key is gated as a REPRESENTATION swap with a
    //  pre-registered offset, never as an improvement.

    //- S6b-1. WHICH FIELD phaseFieldFaceWeights() EVALUATES THE LIMITER ON.
    //
    //    "legacy"  -- c_, as since the function was written.
    //    "content" -- the UNCLAMPED ratio (thetaC_ + thetaH_)/theta_ref.
    //
    //  WHY THE KEY EXISTS. c cannot be deleted while these weights are built
    //  from it: the c equation, shell_ (phaseSharedConvection, which aborts
    //  without them) and Wp (phaseConservativeConvection) all consume them.
    //  This is the step that frees the weights from c; S6c is the deletion.
    //
    //  WHY THE RATIO IS TAKEN UNCLAMPED, AND NOT phaseIndicator(). That
    //  accessor clamps below at zero because a negative indicator reaches an
    //  fvm::Sp on the momentum and skeleton diagonals. A limiter argument is
    //  not a diagonal. The clamp buys nothing here and costs the exactness of
    //  the affine-invariance argument: after S6a q = 1 - (thetaC+thetaH)/
    //  theta_ref UNCLAMPED, so only the unclamped ratio is exactly affine in
    //  q, and vanLeer's r is invariant under an affine map of the transported
    //  field. Clamped, the invariance would hold only to ~1e-18 -- true but
    //  approximate, and this project prefers the identity.
    //
    //  WHAT IT DOES NOT DO. It does not restore Wp = q. That identity needs
    //  the same OPERATOR, not merely the same field, and after S6a the field
    //  defining q -- theta -- is transported by first-order upwind as a
    //  Stage 1 contract while Wp is transported by a limited scheme. See
    //  phasePlateletFluxBasis_ below, which is the key that addresses it.

    //- S6b-2. WHICH FACE VALUES THE PLATELET CONTENT Wp IS ADVECTED WITH.
    //  The OI-6 argument applied one equation further on.
    //
    //    "legacy"  -- phaseWeights_, the phase field's own limited weights.
    //    "content" -- contentFaceWeights_, the face weights the theta
    //                 equation actually transported with.
    //
    //  THE IDENTITY. Wp = w*q, so at saturation (w == 1) the content must
    //  equal the fluid fraction exactly. That survives discretisation only if
    //  Wp's convection operator is the operator that defines q. Before S6a,
    //  q = 0.5*(1 + c) and Wp rode c's weights, so it did. After S6a,
    //  q = 1 - theta/theta_ref and theta rides FIRST-ORDER UPWIND -- chosen
    //  as a contract in solveContentComponents(), field-independent on
    //  purpose -- while Wp still rides a limited scheme. The operators no
    //  longer agree.
    //
    //  MEASURED, not argued, on run 20260919-213132's own logs
    //  (PHASE_CLOT_PLATELET WpRawQDevMax at t = 4e-04):
    //
    //      em39/em40  q from c, same operator as Wp   1.643e-03
    //      em41       q from theta, Wp still limited  1.584e-02   (9.6x)
    //
    //  and 1.584e-02 is the c/theta divergence itself (dqMax 1.5479e-02):
    //  Wp is tracking the field q no longer comes from.
    //
    //  THE FLOOR IS NOT ZERO AND GATE A-EXACT DOES NOT APPLY HERE. The
    //  1.643e-03 residual is the adhesion sink and the explicit laplacian,
    //  not transport -- em39's two operators are identical by construction.
    //  A gate of 1e-11 on this case would be six decades under the
    //  instrument, which is the failure this project has now made five times.

    //- Basis for the INDICATOR chi -- the raw phase marker that eleven sites
    //  compute inline as max(1 - q, 0), and that is NOT the solid fraction.
    //  None of them scales it by theta_ref, so they are not inlined copies of
    //  solidFraction(); they are consumers of a different quantity with a
    //  different job (section 2b of the consumer map).
    //
    //    "legacy"   -- max(1 - q, 0), as before.
    //    "content"  -- max(thetaC_ + thetaH_, 0)/skeletonVolumeFraction_.
    //
    //  THE DIVISION BY theta_ref IS THE WHOLE POINT. theta is theta_ref times
    //  smaller than the indicator, so a global "replace chi with theta" would
    //  silently rescale every one of these sites by 0.45 -- the deviatoric
    //  elastic stress, the damage driver, the Kres ramp, the isSkel threshold
    //  and the bond hold. The fraction replaces solidFraction(); the RATIO
    //  replaces chi. Keeping the two substitutions distinct preserves every
    //  scale in the solver.
    //
    //  NO UPPER CLAMP, DECIDED 2026-09-15 AND NOT INHERITED SITE BY SITE.
    //  Both indicators already exceed 1 -- chi reaches 1.0021786 because c
    //  undershoots, theta/theta_ref reaches 1.0021686 because compression
    //  carries content past the anchor. Clipping here would not be the
    //  clipping of a conserved solution the project has ruled out (no stored
    //  field changes), but it would hide the overshoot from every consumer
    //  and from the diagnostic that measures it. So the ratio is published
    //  raw and PHASE_CLOT_PHASE_INDICATOR reports indMax and the count above
    //  one every step.
    //
    //  WHY THAT IS SAFE, SITE BY SITE, AND WHERE IT IS NOT. An overshoot of
    //  0.22% multiplies a POSITIVE coefficient: the deviatoric stress and the
    //  bond-hold fvm::Sp get 0.22% stronger (an Sp with a positive
    //  coefficient adds to the diagonal, so more of it is more dominance, not
    //  less), the damage driver 0.43% because it is squared, and Kres and
    //  isSkel saturate -- max(rchi - ind, 0) is already floored at zero and
    //  pos0 is already a step. The one site where ind > 1 is genuinely
    //  dangerous is the material-carrier BLEND, chif*phiS + (1 - chif)*phi_,
    //  where chif > 1 makes (1 - chif) negative and SUBTRACTS the blood flux.
    //  That site clamps today and keeps its clamp: this key changes the basis
    //  of the ratio, never a site's own bounds.
    //
    //  The LOWER guard at zero is not part of that decision and stays. It
    //  exists because a negative weight lands on the momentum diagonal --
    //  the source says so at the bond-hold and legacy-drag sites -- and the
    //  "matched" content operator does drive theta negative at roundoff scale
    //  in 2354 cells.

    //- Which field fills interfaceDelta_, the UNIT surface delta delta_Gamma.
    //  'legacy' -> deltaC_ = 0.5*|grad c|;
    //  'content' -> deltaTheta_ = |grad(thetaC + thetaH)|/theta_ref.
    //
    //  DEFAULT 'content'. Alireza, 2026-09-16: the new method is the default,
    //  not the exception. 'legacy' survives only so the S3 control can be
    //  stated explicitly and so the runs that established this stage stay
    //  readable; it is retired with the other three basis keys in S7.
    //
    //  The solver default is rarely what decides, because
    //  cases/AM2020_planar_mural_detachment/scripts/create_cases.py writes
    //  every basis key into constant/phaseClotProperties unconditionally. Its
    //  fallback for this key is 'content' too -- the two had to move together
    //  or the flip would have been cosmetic. Archived runs are unaffected:
    //  they reproduce from their own returned case-input tree, whose
    //  dictionary carries the value it was generated with.
    //
    //  Stage S3 of
    //  research/designs/2026-09-16-theta-native-formulation.md; priced in
    //  research/designs/2026-09-16-S3-delta-basis-pricing.md.
    //
    //  ONE KEY, TWO CONSUMERS, AND THEY MUST MOVE TOGETHER. interfaceDelta_
    //  is one physical object, and its two live consumers -- the free-platelet
    //  transfer at :7496 and the bound-platelet uptake at :8458/:8472 -- are
    //  the two halves of one reaction. Moving one and not the other would
    //  make the solver hold two different interfaces at once, which is the
    //  defect the theta-native programme exists to remove.
    //
    //  WHAT IT DOES NOT TOUCH, DELIBERATELY. phaseGradient_ stays on c: its
    //  only live consumer is the Stefan front in the c equation, which S4
    //  deletes, and moving it now would change c's own trajectory and destroy
    //  the control every later stage compares against. normal_ stays on c
    //  too: its only live path ends in adhesionEnhancement_, which the run
    //  writes as uniform 1.0 at every write time, so nothing it could move is
    //  measurable and a gate on it could not fail.
    //
    //  THE COST OF THAT SPLIT, PRE-REGISTERED. clotGrowth carries
    //  phaseGradient_ and plateletUptake carries interfaceDelta_, so under
    //  'content' the two integrals no longer share a delta and the
    //  stoichiometry identity drifts by the w-weighted ratio: stoichNorm
    //  0.992064 -> 0.992598, predicted offline before the run. That is a
    //  DIAGNOSTIC identity, not a conservation law -- the gated platelet
    //  balance NChange = -dBSum is untouched -- and S4 removes it.
    //
    //  S4 UPDATE, 2026-09-18: it does NOT remove it, and the reason is
    //  OI-12 -- plateletUptake has no N in it and is blind to the platelet
    //  depletion the transfer causes, so stoichNorm is a ratio of two
    //  front-side quantities either way. S4 leaves it alone, reports the
    //  committed transfer beside it, and does not gate on it.

    //- WHAT DEFINES THE INTERFACE NORMAL. S6c-1, and it is the LAST live
    //  consumer of `c` -- see research/designs/
    //  2026-09-20-S6c-normal-basis-and-deletion.md.
    //
    //  'legacy'  -> normal_ = grad(c)/max(|grad c|, gradFloor_)
    //  'content' -> normal_ = -2*grad((thetaC + thetaH)/theta_ref)
    //                         /max(|that|, gradFloor_)
    //
    //  THE FACTOR OF TWO IS LOAD-BEARING. c = 1 - 2*ind, so
    //  grad(c) = -2*grad(ind) EXACTLY. Dropping the 2 leaves the direction
    //  right and the floor wrong: normal_ is a unit vector only where the
    //  gradient clears gradFloor_, and below that it is grad/gradFloor_,
    //  whose magnitude carries the factor. gradFloor_ is 666.667 1/m on the
    //  main case, which is not small. This is the same trap the note on
    //  interfaceDelta_ and phaseGradient_ warns about, one field over.
    //
    //  DEFAULT 'legacy', unlike phaseDeltaBasis_. This key exists to MEASURE
    //  whether the swap is inert before it is taken: normal_ reaches the
    //  physics only through shearPa_ -> adhesionEnhancement_, and that is
    //  clamped to exactly 1 while shearRate < gammaLow. Both normals are
    //  built on either basis and PHASE_CLOT_NORMAL_BASIS reports the
    //  difference, so the legacy child says what the content child will do.
    //
    //  deltaC_ and phaseGradient_ do NOT follow this key. Moving deltaC_
    //  would make the S1' instrument compare a field against itself.

    //- WHAT GROWS THE CLOT. One key, and it changes the model rather than a
    //  discretisation.
    //
    //  'front'    -> the Stefan front: the c equation carries
    //                -Vn_*phaseGradient_, and theta grows separately through
    //                the accepted platelet transfer. TWO ACCOUNTANTS, and
    //                they disagree by 2*alpha/D = 1.61x (S1b).
    //  'transfer' -> the accepted transfer is the ONLY growth. The front term
    //                is gone from all three c branches and c receives the
    //                SAME increment theta does, converted once:
    //                    dc = -(2/theta_ref)*dtheta
    //                from chi = 0.5*(1 - c) and theta = theta_ref*chi.
    //
    //  DEFAULT 'transfer', for the same reason phaseDeltaBasis_ defaults to
    //  'content': the new method is the default, not the exception. And the
    //  same two-place rule applies -- create_cases.py writes this key
    //  unconditionally with its own 'transfer' fallback, so flipping only the
    //  solver would have been cosmetic on this case.
    //
    //  Stage S4 of
    //  research/designs/2026-09-16-theta-native-formulation.md; priced in
    //  research/designs/2026-09-18-S4-growth-driver-pricing.md.
    //
    //  THE INCREMENT IS ALGEBRAIC, AND THAT IS NOT A SHORTCUT. theta's own
    //  source is applied exactly this way -- solveContentComponents()
    //  transports theta, then solvePlateletContent() adds the transfer to the
    //  transported field. Putting c's increment in the same place, off the
    //  same array, at the same commit point, is what makes the two currencies
    //  structurally identical. Putting it inside the c matrix instead would
    //  need the previous corrector's dB (a lag) and would transport this
    //  step's source while theta's is not, reintroducing exactly the
    //  representation difference S4 exists to remove.
    //
    //  IT USES THE APPLIED INCREMENT, NOT Y*dB.
    //  plateletTransferKernel::applyIncrement clamps against phiSMax_, so a
    //  capped step taken from Y*dB would leave the two currencies
    //  disagreeing precisely when the cap bites. capClamped is 0 on every
    //  record today; that is a reason the gate will not SEE it, not a reason
    //  to get it wrong.
    //
    //  WHAT IT DOES NOT TOUCH. Vn_ is NOT demoted to a diagnostic -- the
    //  design text said so and the design text was wrong. surfaceLossCoeff_
    //  is built from Vn_ and enters the bound-platelet equation as
    //  fvm::Sp(surfaceLossCoeff_, w_) on every step, so Vn_ stops moving c
    //  and stays live physics. phaseGradient_ is NOT migrated onto theta
    //  either: once the front is gone it has no live equation consumer at
    //  all (both cap divisors log capOn=no on 1090/1090 records, the
    //  `matched` site is dead under contentOperator 'content'), so a gate on
    //  it could not fail. It is a functional of c and S6 retires it with c.

    //- Reaction off: transport and diffusion only (D0, D1, D2, D3, T7).
    Switch plateletTransportOnly_;

    //- TEST-ONLY EVIDENCE DUMP, DEFAULT OFF. Writes the fluxes and fields
    //  solvePlateletContent() ACTUALLY consumed, once per phase corrector,
    //  so an offline assembler can rebuild ddt(eps,nP) + div(Feps,nP) from
    //  the same numbers the solver used rather than from the ones a case
    //  dictionary declares.
    //
    //  WHY A DUMP AND NOT A LOG LINE. The existing PHASE_CLOT_FREEPLATELET
    //  report carries min/max/absSum of the carrier and Feps. Those are
    //  reductions: they cannot distinguish a carrier that is right
    //  everywhere from one that is wrong on half the faces and compensating
    //  on the other half, and they cannot be fed to a discrete reference.
    //  The defect this exists to measure -- a MIXTURE flux supplied where a
    //  fluid-phase flux is required -- is a per-face identity, so the
    //  evidence has to be per-face.
    //
    //  IT WRITES EVERY CORRECTOR, NOT EVERY WRITE TIME. A corrector is the
    //  granularity at which the carrier is selected, and averaging or
    //  sampling across correctors would lose exactly the trial the
    //  transport rode. Intended for fixtures of a handful of steps on a
    //  few dozen cells; it is not a production diagnostic and nothing in
    //  the solver reads it back.

    //- PORE CONCENTRATION, the SOLVED UNKNOWN, platelet-equivalents per unit
    //  PORE volume. Conservative storage ddt(eps, nP) makes the discrete
    //  conserved quantity eps*nP = NP, and fvm::laplacian(eps*Deff, nP)
    //  delivers exactly the required face flux T_f*(N_j/eps_j - N_i/eps_i)
    //  without ever forming a variable-coefficient Laplacian on NP.
    volScalarField nP_;

    //- BULK CONTENT, platelet-equivalents per unit MIXTURE volume, = eps*nP.
    //  Derived and written for the budget; never independently solved, so
    //  the two representations cannot drift apart.
    volScalarField NP_;

    //- THE POROSITY THE PLATELET TRANSPORT USES, eps = 1 - theta, from the
    //  CONTENT fields.
    //
    //  IT IS A MEMBER AND NOT A LOCAL, AND THAT IS THE WHOLE POINT.
    //  fvm::ddt(rho, psi) reads rho.oldTime(). A local
    //  `volScalarField eps(1 - theta)` has an oldTime() that is created on
    //  first access by COPYING ITSELF, so eps_old == eps_new and the time
    //  derivative silently loses the porosity change -- a wrong answer with
    //  no error and no warning. Registered here, the registry rotates the
    //  old time at each step. The constructor instantiates oldTime() so the
    //  first rotation has something to rotate.
    //
    //  This reasoning is NOT taken as proof: solvePlateletContent() closes a
    //  measured inventory budget every step, which is what would actually
    //  catch it.
    //
    //  NOT the same field as phiFluid_. phiFluid_ = 1 - solidFraction() is
    //  built from q_ and weights the CONTINUITY flux; this one is built from
    //  thetaC_ + thetaH_. They agree to ~1e-9 today and become different
    //  fields the moment deposition drives theta -- see the plan, 5.5.
    volScalarField epsP_;

    // The four material properties as FIELDS. Every use site below reads
    // these rather than the scalars; the scalars remain the SHELL (and
    // single-material) values and are what the dictionary still names G, Gc
    // and k0p, so no existing case file changes.
    volScalarField GField_;
    volScalarField lambdaField_;
    volScalarField GcField_;
    volScalarField k0pField_;

    // Core values. Default to the shell values, so a dictionary that does not
    // mention a core describes a single material exactly as before.
    dimensionedScalar Gcore_;
    dimensionedScalar Gccore_;
    dimensionedScalar k0pcore_;

    // --- DIFFUSE-DOMAIN WEIGHTING OF THE SKELETON (root-cause fix) ---
    //
    // DEFAULTS TO FALSE. With it false, phiS is identically 1 and the pin
    // weight is q, so the assembled matrix is the previous one multiplied by
    // exactly 1.0 in every term -- bit-identical, not merely equivalent.
    //
    // WHAT IT FIXES. The skeleton equation carried the FULL elastic modulus
    // everywhere, including in blood, and pinned Us to zero out there with
    // q*(2G+lambda)/ell^2. Two consequences, both measured:
    //
    //   1. The clot was an inclusion in an infinite solid OF ITS OWN
    //      STIFFNESS rather than a body with a free surface. That is a clamp
    //      on its whole outer boundary, and it is the most likely reason the
    //      measured venous displacement is 8.5x below the clamped-block
    //      estimate (the limitation already recorded in mainCaseControls).
    //   2. The pin dragged Us from its clot value to zero across the four
    //      cells of the phase interface, so grad(Us) there was spurious and
    //      psi+ ~ G|dev eps|^2 picked it straight up. Run 20260814-055141:
    //      max H in the interface 3.58e-01 against 2.64e-02 in the bulk clot,
    //      a factor of 13.5, with blood reaching 1.19e-01. Every "tear" that
    //      run produced was a numerical boundary layer failing.
    //
    // THE FIX IS THE DIFFUSE-DOMAIN METHOD (Li, Lowengrub, Ratz & Voigt 2009).
    // A sharp-interface problem div(sigma) + f = 0 in the solid with a
    // traction-free surface is approximated on the whole domain by
    //     div(phi*sigma) + phi*f = 0,
    // BOTH sides carrying the same indicator. The body force here was already
    // weighted -- dragCoeff and the pressure force both carry chi -- and the
    // stress was not. Weighting the stress by chi too is what makes the outer
    // surface traction-free instead of clamped, and it is the inconsistency,
    // not the fracture model, that produced the artefact.
    //
    // The pin cannot simply be deleted: without it the fluid-region skeleton
    // drifts with the flow, unbounded, because the drag term alone gives
    // Us -> UsPrev + U*dt every step. It is instead confined to essentially
    // pure fluid, where the solid weight is negligible, so the clot's own
    // surface is free while the far field stays anchored.

    //- Floor on the solid weight, so the elliptic operator does not vanish
    //  in blood and the linear system stays well conditioned.
    scalar skeletonSolidFloor_;

    //- Surface-erosion rate coefficient, m/(s.Pa), applied as a NEGATIVE
    //  contribution to the interface normal speed:
    //      Ve = clotErosionRate_ * max(shearPa - clotErosionTauPa_, 0)
    //  Zero (the default) is OFF and leaves every existing result identical.
    //
    //  WHY THIS EXISTS: WITHOUT IT THE CLOT CANNOT LET GO, EVER.
    //  Three routes could shorten a stretched clot and none of them can:
    //
    //  (1) BULK FRACTURE is physically unreachable here, and that is physics,
    //      not a bug. With Gc = 5.547 J/m2 (the shipped Gcfield) and the
    //      shell's G = 4 Pa, the fracture length scale Gc/G is 1.39 METRES
    //      against a 40 um clot -- a body 35000x smaller than its own fracture
    //      length deforms, it does not crack. Measured on run 20260908-010836:
    //      Dg reaches 6.4e-06 of the 1.0 needed, after 7.9 ms. The stored
    //      elastic energy G*V ~ 5.6e-15 J against Gc*A ~ 2.2e-10 J to open one
    //      crack: short by 4e+04. research/designs/AM2020_defect_audit_
    //      2026-08-27.md reached the same conclusion independently (short by
    //      8.4e5x in energy density, 302x in cohesive strength) and states it
    //      as "a clot at physiological stiffness and physiological toughness
    //      CANNOT rupture in bulk at this operating point".
    //
    //  (2) WALL PEEL is implemented, works, and ARRESTS. Run 20260908-010836:
    //      debondedFaces 0 -> 31 of 56 injury faces, utilisation
    //      (peakTraction/adhesiveStrength) 0.119 -> 0.985 at 3 ms -> 0.389 at
    //      7.9 ms. The peel advanced until the remaining bonds UNLOADED as the
    //      tail leaned downstream. And peel removes the ANCHOR, never the tail.
    //
    //  (3) THE RELEASE CRITERION IS A CONNECTIVITY FLOOD FILL. A cell is
    //      released only when it cannot be reached from a bonded wall face
    //      through connected clot (releaseDetachedMaterial, .C:4099-4160). A
    //      coherent tail stays "supported" however far it stretches, so
    //      PHASE_CLOT_RELEASED has read cells=0 in EVERY two-velocity run.
    //
    //  So the clot stretches without bound. That is the defect this closes.
    //
    //  THE MECHANISM IS THE PI'S OWN. Dr He Li, 2026-08-25: "the core region
    //  is stiff ... does not slide or detach or rupture. While the shell
    //  region is soft and permeable ... It can deform, rupture, and detach."
    //  Erosion is therefore weighted by shell_ when coreShell_ is on, exactly
    //  as releaseCoreExempt_ already exempts the core from release.
    //
    //  THE FORM IS XU'S. Source 33, Xu et al. 2017 -- the paper this case
    //  replicates and the source of k0p -- section 4: "partially activated
    //  superficial platelets located downstream, in the zone of a higher flow
    //  shear stress, are peeled off the clot ... platelet attachment to and
    //  removal from the clot surface are driven by the normal and tangential
    //  flow stress components". They identify "tearing pieces off the clot
    //  shell" as "the major contributing mechanism of embolization" and
    //  "continuous microembolization ... an indispensable mechanism that
    //  limits further clot growth".
    //
    //  THE THRESHOLD IS BRACKETED BY TWO INDEPENDENT SOURCES:
    //    - Xu (33) calibrate accumulation against detachment so the two
    //      BALANCE at a wall shear rate of 500 1/s. With mu = rho*nu =
    //      1050*3.333e-06 that is tau = 1.75 Pa.
    //    - Wu et al. 2017 (source 15) give tau_emb = 30 dyn/cm2 = 3.0 Pa,
    //      which is a wall shear rate of 865 1/s.
    //  So the threshold is sourced to 1.75-3.0 Pa. The default is Wu's 3.0,
    //  the directly quoted number; Xu's 1.75 is the other end of the bracket.
    //  THIS CASE RUNS AT 1000 1/s, above both, and measured shearPa over the
    //  clot reaches 8.2 Pa -- so the clot here SHOULD be net-eroding.
    //
    //  THE RATE COEFFICIENT IS UNSOURCED and is swept, exactly as
    //  adhesiveStrength is and for the same reason recorded in
    //  research/designs/AM2020_wall_adhesion_detachment_design.md: "No local
    //  source gives a thrombus/vessel adhesive strength. It is swept; the
    //  deliverable is the strength required." The same discipline applies
    //  here and the deliverable is the erosion rate required.
    //  Scale for the first swept value, by Xu's balance principle applied at
    //  THIS operating point rather than at 500 1/s: the model's own growth
    //  speed adhesionSpeedEff is 4.6e-05 to 1.7e-04 m/s, and the measured
    //  shear excess over 3 Pa on the clot is ~5 Pa, so a rate that makes
    //  erosion comparable to growth is ~2e-05 m/(s.Pa). That is an anchor,
    //  not a measurement, and it is labelled as one.
    scalar clotErosionRate_;

    //- Shear stress in Pa at which surface erosion begins. Sourced to
    //  1.75-3.0 Pa; see clotErosionRate_ for the bracket and both anchors.
    scalar clotErosionTauPa_;

    //- Lower bound on the skeleton drag RATE, in 1/s, imposed as
    //  K >= skeletonDiluteDragRate_*chi. Zero (the default) leaves every
    //  existing result bit-identical.
    //
    //  WHY A FLOOR PROPORTIONAL TO chi AND NOT ANOTHER RESIDUAL BAND.
    //  Kozeny-Carman is a PACKED-BED correlation and this solver evaluates it
    //  two decades below its range of validity. Measured on run
    //  20260907-111028 at t = 5.9e-03 s, binning every cell by
    //  chi = max(1-q, 0), with compliance := elastic weight / total drag rate
    //  = max(chi, skeletonSolidFloor)/(Kd + Kres):
    //
    //     chi           cells   compliance   max|Vs|/max|U|
    //     0    -1e-4    19921     1.5e-06        1.00
    //     1e-4 -0.005    4935     1.4e-05        1.00
    //     0.005-0.01      555     1.6e-04        3.43
    //     0.01 -0.02      556     3.0e-04        3.71
    //     0.02 -0.05      896     1.7e-04        3.90
    //     0.05 -0.1       998     3.1e-05        3.07
    //     0.1  -0.3       411     7.7e-06        1.76
    //     0.3  -1        2738     1.1e-07        0.60
    //
    //  |Vs| stays bounded by max|U| exactly where compliance <= ~1e-5 and
    //  runs away by a factor of four where it does not. The elastic weight
    //  grows like chi; Kd falls off far faster than chi as chi -> 0; and Kres
    //  is capped at residualPhiS/deltaT = 67 1/s however wide its support is
    //  made. So the peak cannot be removed by widening residualChi (measured:
    //  0.01 -> 0.1 buys a factor of 2.2, against a gap of 20-30x) nor by
    //  reweighting the deviatoric stress with solidFraction() (a factor
    //  1/skeletonVolumeFraction = 2.2). BOTH WERE TESTED ARITHMETICALLY
    //  AGAINST THE MEASURED FIELD BEFORE BEING REJECTED, not guessed at.
    //
    //  With K >= k1*chi the compliance is EXACTLY 1/k1 wherever the floor
    //  binds -- flat, with no band and no peak, by construction. k1 = 1e+05
    //  gives 1e-05, the measured level at which the ratio is 1.00.
    //
    //  THE FORM IS PHYSICAL, THE CONSTANT IS CALIBRATED, AND THOSE ARE NOT
    //  THE SAME CLAIM. Drag per unit mixture volume proportional to solid
    //  fraction is the dilute-suspension asymptotic -- number density times
    //  single-fibre Stokes drag -- which a packed-bed correlation does not
    //  reproduce. But k1 here is set from the table above, NOT derived from a
    //  fibre diameter, because this project does not have one. As a sanity
    //  bound only: 18*nu/d^2 on a 200 nm fibre is ~1e+09 1/s, four decades
    //  above the 1e+05 used, so this floor is conservative rather than an
    //  inflation of the physical drag.
    //
    //  Inert where the skeleton is real. At chi = 0.9 the floor is 9e+04
    //  against a measured Kd of 8.3e+06, about 1%. It binds only below
    //  chi ~ 0.35 and vanishes as chi -> 0, where Kres already holds
    //  compliance at 1.5e-06, so the two compose with no gap between them.
    //
    //  The same band is the "DEAD BAND 0.002-0.05" recorded against run
    //  20260815-201949 further down this file. It was patched then with a
    //  stiffness pin (skeletonPinExponent_) and it is STILL the failure site.
    scalar skeletonDiluteDragRate_;

    //- Fluid fraction at which the pin STARTS to act. Below it the pin is
    //  exactly zero, so no part of the clot or its interface is clamped.
    //  0.998 corresponds to chi = 1e-3, about 14 cells past the c = 0 contour
    //  at epsilonC = 3 um -- far enough out that the energy weight there is a
    //  thousandth of the bulk.
    scalar skeletonPinExponent_;       // exponent n in pinWeight = q^n.
                                       // Replaces skeletonPinStart, whose
                                       // switch-on threshold left a band of
                                       // cells with neither stiffness nor pin

    // --- THE PHASE FIELD MOVES WITH ITS OWN MATERIAL ---
    //
    // DEFAULTS TO FALSE, so every case that does not ask keeps the previous
    // advection exactly: nothing at all unless advectReleased is on, and then
    // only the released region at a rigid-body velocity.
    //
    // WHAT IS MISSING WITHOUT IT. c labels solid against blood, and it is
    // advected by the FLUID flux. Inside the clot the Brinkman penalty holds
    // the fluid still -- run 20260811-051109 measured 2.18e-7 m/s in the clot
    // against 2.28e-3 in the blood -- so the clot's own phase field cannot
    // move, however far its skeleton deforms. Us can reach any strain and c
    // will not follow it. The ONLY way material has ever been able to leave
    // the clot is the damage variable erasing cells, which is why every
    // attempt to show a tear has had to lower Gc until AT2 fired.
    //
    // Source 33 does not work that way and has no toughness at all. In Xu et
    // al. the clot phase fields are advected by the flow, the soft shell is
    // dragged downstream into an elongated filament, and pieces separate as a
    // TOPOLOGICAL event in the phase field. Reproducing that needs the phase
    // field to move with its material, not a lower Gc.
    //
    // The advecting velocity of a two-phase mixture is the volume-fraction
    // weighted material velocity,
    //     v = chi*v_s + q*u,
    // which is the skeleton velocity where the material is solid and the
    // fluid velocity where it is not, with no switch and no threshold. v_s is
    // (Us - UsPrev)/dt, the same skeleton velocity the momentum equation
    // already uses, so the two sides cannot disagree.

    //- Floor applied to the permeability before it is used as mu/k. Was the
    //  hard-coded Foam::small = 2.22e-16 m2, which is not a small number for
    //  a permeability: run 20260814-055141 realised EXACTLY 2.22e-16 m2 in
    //  every one of the 168 bulk-clot cells, so k0p had no effect on the clot
    //  at all and the whole porosity axis was measuring the floor. A
    //  dictionary entry, defaulting to what it was.
    scalar permeabilityFloor_;

    //- Which quantity drives the interphase drag: "legacy" (the shipped
    //  chi-driven path, unchanged and the default) or "content" (Stage 3).
    //
    //  A TYPO MUST NOT LOOK LIVE. Validated at construction, not at the
    //  first solve -- the same failure mode contentOperator was given a
    //  fail-closed check for, after `contentOperator matchd` would have run
    //  the shipped model while the manifest claimed the identity test.

    // Cell extent along the interface normal, h_n = 2V/sum_f|Sf.n|, exact
    // for a cube. This is the length the platelet depletion layer D/k
    // competes against, so it -- not V^(1/3) -- is what sets the cell
    // Damkohler number and the series depletion resistance. V^(1/3) would
    // be wrong on this mesh, whose z extent is not the in-plane spacing.
    // THE ONE CONVECTION OPERATOR THE PHASE FIELD AND THE PLATELET CONTENT
    // BOTH USE. Built in exactly one place so the two cannot drift apart.
    //
    // Wp = q*w and q = 0.5*(c+1), so with w == 1 the content is an AFFINE map
    // of the phase field. The bounded Gauss operator is affine-covariant, so
    // advecting both with the SAME weights and the SAME flux preserves
    // Wp == q by exact arithmetic. Run 20260823-023233 measured that at
    // |Wp - q| = 6.686e-12.
    //
    // IT BROKE ON 2026-09-02 AND THIS FUNCTION IS WHY IT CANNOT BREAK AGAIN.
    // The more-upwind limiter made phaseFieldFaceWeights() return something
    // other than c's own vanLeer weights, but the c equation still said
    // `fvm::div(phi, c_, "div(phiReleased,c)")`, which resolves to
    // `bounded Gauss vanLeer` FROM fvSchemes and rebuilds c's own weights.
    // The two fields silently assembled different matrices and gate A-exact
    // went from 6.686e-12 to 1.784e-05. Nothing else noticed.
    tmp<fvScalarMatrix> phaseSharedConvection
    (
        const surfaceScalarField& carrier,
        const volScalarField& f
    ) const;

    //- The same conservative operator with the weights named explicitly.
    //  phaseConservativeConvection() is this with phaseWeights_; the platelet
    //  content may be given contentFaceWeights_ instead, per
    //  phasePlateletFluxBasis_. One implementation, so the two cannot drift.
    tmp<fvScalarMatrix> phaseConservativeConvectionWith
    (
        const surfaceScalarField& carrier,
        const volScalarField& f,
        const surfaceScalarField& weights
    ) const;

    //- S10 FIX 3: the antidiffusive flux of f on INTERNAL faces,
    //  carrier*((w_lin - w_up)*f_owner + ((1-w_lin) - (1-w_up))*f_nei), i.e.
    //  the linear face flux minus the upwind one. Zero on boundary faces.
    tmp<surfaceScalarField> fctAntidiffusiveFlux
    (
        const surfaceScalarField& carrier,
        const volScalarField& f
    ) const;

    //- S10 FIX 3: Zalesak's face limiter for antidiffusive flux A of a field
    //  whose low-order predictor is fLow and old-time value fOld. Local
    //  bounds are the min/max of both over the cell and its face neighbours.
    //  Floating-point honest: see the definition.
    tmp<surfaceScalarField> fctLimiter
    (
        const volScalarField& fLow,
        const volScalarField& fOld,
        const surfaceScalarField& A
    ) const;

    //- S10 FIX 3: the weights the content was ACTUALLY transported with.
    //  contentFaceWeights_ (the implicit matrices' weights) plus, under fct,
    //  lambda*(w_lin - contentFaceWeights_). What the skeleton momentum flux
    //  and PHASE_CLOT_MMC must use (OI-6): the face value theta moved at.
    tmp<surfaceScalarField> contentTransportedWeights() const;

    //- S10 FIX 3: f -= deltaT/V * div(lambda*A), conservatively.
    void fctApply
    (
        volScalarField& f,
        const surfaceScalarField& lambda,
        const surfaceScalarField& A
    ) const;

    //- Convection for Wp, selecting the face weights per
    //  phasePlateletFluxBasis_. 'legacy' is phaseConservativeConvection()
    //  itself, called through, so that path is unchanged.
    tmp<fvScalarMatrix> plateletContentConvection
    (
        const surfaceScalarField& carrier,
        const volScalarField& f
    ) const;

    //- surfaceIntegrate(carrier) as a full field. The right-hand side that
    //  makes int chi dV conserved when the carrier is not solenoidal --
    //  which a per-phase flux never is. Derived and measured 2026-09-05.
    tmp<volScalarField> carrierDivergence
    (
        const surfaceScalarField& carrier
    ) const;

    //- Network (osmotic) pressure p_n(phi_s), Cogan & Guy Eq. 19. Zero at
    //  the unstressed fraction, divergent at close packing. The skeleton's
    //  volumetric response, which dev(be) cannot provide.
    tmp<volScalarField> networkPressure() const;

    //- S10 FIX 1: the solid fraction p_n is EVALUATED at. "nominal":
    //  forceFraction(), i.e. theta, measured against the constant phi_s0 --
    //  which reads a diffuse interface as dilated network. "deposited":
    //  phi_s0 + (theta - thetaNetRef_), the nominal density plus the content
    //  gained by compaction alone. Shared by networkPressure() and the
    //  pnCapCells gate so the law and its gate cannot measure different
    //  things.
    tmp<volScalarField> networkPressureArgument() const;

    //- S11: the isotropic network stress as a PARTIAL pressure, per unit
    //  mixture area -- the quantity whose gradient is a force on the
    //  skeleton. For the legacy laws it is forceFraction()*networkPressure(),
    //  exactly the expression elasticStress() always used. For convexExcess
    //  it is (thetaNetRef/phi_s0)*p(a): the dense network's volumetric
    //  Cauchy stress weighted by the occupancy that was DEPOSITED. Shared by
    //  elasticStress() and the face-gradient force so they cannot disagree.
    tmp<volScalarField> networkPartialPressure() const;

    //- S13: where network exists, min(1, thetaNetRef/(residualChi*phi_s0))
    //  -- the complement of the field beRelaxBasis networkAbsent relaxes be
    //  on, so "network present" means one thing for the stress and for the
    //  strain memory. The weight of the partial pressure under
    //  networkPressureWeight networkPresence.
    tmp<volScalarField> networkPresence() const;

    //- S14: the fraction of the cell's network that still carries shear,
    //  1 - ruptured_ under networkRupture criticalStretch, else 1.
    tmp<volScalarField> intactFraction() const;

    //- S14: carry ruptured_ with the material, on the carrier the content
    //  and shell_ use, with the same shared operator as shell_. A no-op
    //  under networkRupture off.

    //- S14: rupture the network wherever it has reached the critical
    //  stretch -- ruptured_ = 1 and be_ = I there -- and report it.
    void ruptureOverstretchedNetwork();

    //- S15: shell_ = thetaH/theta under materialLabelBasis
    //  contentComposition, then the material fields. A no-op otherwise.
    void syncMaterialLabelFromContent();

    //- S16 C: ruptured_ = D/theta and the checks that every component is
    //  non-negative (a negative component is a FatalError naming the site).
    void syncDamageFromPartition(const char* site);

    //- S16 C: restore the partition from its step checkpoint, at the sites
    //  thetaC_/thetaH_ are restored; refuses a missing checkpoint.
    void restoreDamagePartition(const char* site);

    //- S16 B: the deviatoric stress weight under skeletonFloorBasis, as
    //  skeletonFaceStress() and elasticStress() use it.
    tmp<volScalarField> skeletonStressWeight() const;

    //- S16 B: record, per site, max(w_used - max(ind, floor)*(1 - d)).
    void recordStressWeight(const scalarField& wUsed, const label site) const;

    //- S16 D: p -> max(p, 0) + (1 - d) min(p, 0) under networkPressureDamage
    //  tensile; the input unchanged otherwise.
    tmp<volScalarField> networkPressureDamaged
    (
        const tmp<volScalarField>& tp
    ) const;

    //- S11 partial pressure before S16 D; networkPartialPressure() applies
    //  networkPressureDamaged() to it.
    tmp<volScalarField> networkPartialPressureUndamaged() const;

    //- S17 A: the intact share thetaI/S of the partition (1 where S = 0);
    //  the pending share and 1 - d come with it. Under fibreRelease only.
    tmp<volScalarField> intactShare() const;

    //- S17 A: P-LS on every cell holding pending content, from the content
    //  C_B/thetaP, rewriting pendingAxis_, pendingStretch_ and C_B; the
    //  derived fields follow. site names the caller for a refusal.
    void projectPendingFibres(const char* site);

    //- S18p: whether a raised cell at site (0 T, 1 P) is dumped, by the
    //  quotas above; counts it when it is.
    bool s18pTakeCell(const label site, const bool present);

    //- S17 A: B_P, p_P from the carried state; checks every value finite.
    void syncPendingFibres(const char* site);

    //- S17 A: the material-line rotation and UNI, once per step, from the
    //  same velocity gradient the be transport used (L = grad^T).
    void rotatePendingFibres(const volTensorField& gradU);

    //- S17 A: the deviatoric stress per unit weight, per cell,
    //  [(thI/S) dev(be) + (thP/S) T dev(n n)]/(1 - d) (splitDev).
    tmp<volSymmTensorField> splitDeviatoricStress() const;

    //- S17 A: the tangent stiffness per unit weight, per cell,
    //  [(thI/S) stiff + (thP/S)(lf^2 - 1/lf)]/(1 - d) (mixedStiffness), with
    //  stiff the STRIPES form under elasticStabilityStiffness.
    tmp<volScalarField> mixedTangentStiffness() const;

    //- S17 split: max over cells of |M_used - M_ref|, the reference rebuilt
    //  HERE from the raw fields (site 0 skeletonFaceStress, 1 elasticStress).
    void recordStressSplit(const symmTensorField& Mused, const label site) const;

    //- S17 split, site 2: the tangent bound's stiffness per unit weight.
    void recordStiffnessSplit(const scalarField& kUsed) const;

    //- The skeleton-row residual drag Kres of the 2x2 block (1/s), ONE
    //  expression for the block and for S17 C's Vs predictor.
    tmp<volScalarField> skeletonResidualDrag() const;

    //- S17: the outlet-face and clot-surface second differences (change 5).
    void writeS17SecondDifferences();

    //- S17b A1: the content that carries the fibre -- thetaTaut_ under
    //  fibreSlack v1, thetaPending_ otherwise (S17, bit for bit).
    const volScalarField& fibreCarrier() const;

    //- S17b A2: the smallest carrying content at which the fibre is read
    //  (P-LS, W2, G-proj): DBL_MIN under fibreArithmetic fieldUnits, the
    //  smallest positive double otherwise (S17's "> 0").
    scalar fibreReadMin() const;

    //- S18 C (R2b): the content weight theta/thetaNetRef, 0 where no
    //  network was deposited (the presence factor is 0 there too).
    tmp<volScalarField> networkContentRatio() const;

    //- S17b log (06; design section 2 "New logs"): the four parts of the
    //  skeleton elastic force -- intact deviatoric, pending deviatoric,
    //  intact (and packing) network pressure, frozen pending tension -- as
    //  velocities iAD11 F through the block, at cell w along dir (m/s).
    void s17bForceParts
    (
        const surfaceScalarField& iAD11f, const label w, const vector& dir,
        scalar parts[4]
    ) const;

    //- S17b A2: scale an assembled predictor's rows by dt/V (field units).
    template<class Type>
    void scaleToFieldUnits(fvMatrix<Type>& m) const;

    //- S11 FIX 3: series (distance-weighted harmonic) face value of a
    //  non-negative weight. 1/w_f = lambda/w_P + (1 - lambda)/w_N, so a face
    //  with an empty side carries exactly zero -- stress crosses a face only
    //  through network that is present on BOTH sides (Patankar 1980, 4.2-3).
    //  Refuses a negative weight rather than bounding it.
    tmp<surfaceScalarField> seriesFaceWeights(const volScalarField& w) const;

    //- S11: the elastic force on the skeleton, per unit mass, as the FACE
    //  flux the 2x2 block uses and as its CELL counterpart for every
    //  diagnostic. One construction, selected by stressFaceTransmission_ and
    //  networkPressureForce_; with both at their shipped values each returns
    //  the shipped expression unchanged.
    tmp<surfaceScalarField> skeletonElasticForceFlux() const;
    tmp<volVectorField> skeletonElasticForceCell() const;

    //- S11: the elastic stress ON FACES that both of the above integrate.
    //  Deviatoric part: occupancy weight (series or linear) times the
    //  intrinsic gD*G*dev(be). Isotropic part only under stressDivergence;
    //  under faceGradient it is applied as a compact face force instead.
    tmp<surfaceSymmTensorField> skeletonFaceStress() const;

    //- S11/S12: refuse a network argument a outside (0, phiSMax), CELLS
    //  FIRST (with theta and thetaNetRef) and boundary faces after, naming
    //  the caller. Both ends are infinite barriers of convexExcess, and
    //  k_KC(a) is undefined at a <= 0: a value outside is a transport or
    //  growth defect upstream, reported, never bounded.
    void checkNetworkArgument(const volScalarField& a, const char* who) const;

    //- S12: the network's INTRINSIC permeability, Kozeny-Carman at its
    //  intrinsic density a = networkPressureArgument(), anchored at k0
    //  (the shell/core map) at a = phi_s0. No floor, no cap.
    tmp<volScalarField> intrinsicPermeability() const;

    //- S12: the interphase drag, KINEMATIC (1/s), under dragOccupancyBasis
    //  intrinsic: phaseIndicator()*(1-a)^2*nu/intrinsicPermeability(). The
    //  one expression both drag sites use (the dynamic one times rho).
    tmp<volScalarField> intrinsicDragRate() const;

    //- S12: PHASE_CLOT_FRINGE -- occupancy-weighted rms skeleton slip
    //  |Vs - U| in the fringe (1e-3 <= ind < 0.2) against the body
    //  (ind >= 0.9), and the fringe's drag over its intrinsic value. Written
    //  for every basis, so legacy runs are comparable.
    void reportFringe() const;

    tmp<volScalarField> normalCellSize() const;

    void updateDerivedFields();
    void updateMaterialFields();

    //- Carry the core/shell label with the material, on the same flux and
    //  the same scheme as the phase field itself. A no-op without a core.
    void advectMaterialLabel(const surfaceScalarField& carrier);

    //- Stage 1: conservative transport of the two solid contents on the
    //  SHARED solid face flux, kinetics off. First-order implicit upwind with
    //  field-independent weights, so the component fluxes sum exactly to the
    //  total solid flux and positivity is analysable. Reports the equivalence
    //  residual against the legacy chi and the conserved inventory.
    void solveContentComponents(const surfaceScalarField& carrier);

        //- Stage 2: transport the free platelet number and apply the one
        //  accepted paired transfer. Carrier convention per
        //  plateletFluxConvention_. Reports PHASE_CLOT_PLATELET.
        void solvePlateletContent
        (
            const surfaceScalarField& carrier,
            const label phaseCorrector
        );

        //- Fail-closed validation of the Stage 2 dictionary combination.
        //  Called once from the constructor.
        void checkPlateletConfiguration();

    void updatePermeability();

    // --- clot/wall adhesive bond, when 0/Us uses clotWallAdhesion ---
    // The bond is IRREVERSIBLE, so it is subject to exactly the hazard that
    // HStep_ exists to prevent: evaluated inside the exchange it would fail
    // faces on unconverged trial iterates and the fixed point would depend on
    // the sweep count. restore() therefore runs at the top of every sweep and
    // commit() once, after the exchange has settled - the same discipline
    // already applied to H_ and to releaseDetachedMaterial().
    //
    // Both are no-ops when no patch uses the condition, so every case that
    // keeps `wall fixedValue (0 0 0)` behaves byte for byte as before.
    void commitWallBonds();


    //- Advance be_ one step: the Eulerian replacement for solveSkeleton().
    //  Calls beTransport::solve(), the same code the stage-1 gate qualified,
    //  and records the SPD and determinant diagnostics.
    //
    //  FIRST STEP WITH EULER, AND THAT IS NOT A DETAIL. OpenFOAM's `backward`
    //  has only one old-time level at the first step, where the stencil
    //  degenerates to 1.5 (f^1 - f^0)/dt and the first increment comes out at
    //  exactly 2/3 of the correct one. Stage 1 measured that to five figures
    //  and it held the whole ladder at first order until it was fixed. The
    //  gate could seed the missing level from an exact solution; this solver
    //  cannot, so it takes one Euler step instead -- a one-off O(dt^2) local
    //  error, against the degenerate step's O(dt).
    void solveElasticStrain();

    //- Cauchy elastic stress from be_, sigma_e = G(chi) dev(be), degraded by
    //  damage and weighted by the diffuse-domain solid fraction exactly as
    //  the skeleton stress is.
    tmp<volSymmTensorField> elasticStress() const;

    //- Transport Us_ passively so the Eulerian path still reports a
    //  displacement: dUs/dt + (v.grad)Us = v, in conservative form.
    //
    //  DIAGNOSTIC ONLY, AND THAT RESTRICTION IS LOAD-BEARING. Nothing reads
    //  Us_ back into the mechanics when eulerianSolid_ is true -- not the
    //  stress, not the damage energy, not the momentum equation. The whole
    //  reason for moving to be is that a displacement field on a fixed
    //  reference inverts once the clot translates far enough; feeding this
    //  one back would reintroduce exactly that. It exists so the elastic-slab
    //  gate can compare the same quantity it compared for the v3 path, whose
    //  20260815-072849 PASS is the reference this stage is measured against.
    void solvePassiveDisplacement();

    void solveDamage();
    //- Decide, once per timestep, whether a diverged skeleton ends the run

    void releaseDetachedMaterial();

    // Mark clot material that is no longer connected to an intact bond, so
    // the phase field can be advected away by the flow. Without this the
    // model can only ever show the bond FAILING, never the embolus leaving:
    // run 20260810-212211 released 59 of 60 faces and the clot did not move a
    // micron, because the phase equation has no advective term at all.
    void updateReleasedIndicator();

        //- Report what is still standing on the injured wall. Diagnostic only.
        void reportWoundState();

        //- Build bondHold_ from the intact bonds. No-op when the hold is off.
        void updateBondHold();

    //- The face weights div(phiReleased,c) is about to use, captured from that
    //  entry itself so the Wp equation cannot be given a different operator.
    tmp<surfaceScalarField> phaseFieldFaceWeights
    (
        const surfaceScalarField& carrier
    ) const;

    void solvePhaseField();
    void beginPlateletOuterTransaction();
    void solvePlatelets();
    void writeBalanceDiagnostic();

        //- Assemble and write the skeleton momentum residual. Diagnostic only.

        //- The coupled two-velocity poroelastic pressure corrector: a
        //  per-cell 2x2 block inversion of the drag coupling and a pressure
        //  equation assembled from BOTH momentum equations. Selected by the
        //  validity of tVsEqn.
        void correctPressureTwoVelocity();

        //- S19: true under momentumPressureCoupling blockCoupled.
        bool s19BlockCoupled() const;

        //- S19: the block-coupled step, replacing the predictors and the
        //  PIMPLE correctors under blockCoupled (design section 2).
        void solveBlockCoupledStep();

        //- Report Vs_ against U_ after the coupled solve. DIAGNOSTIC ONLY --
        //  it no longer computes Vs_. The algebraic closure it used to hold is
        //  deleted, not disabled; see the note on Vs_.
        void reportSkeletonVelocity();

        //- Solid volume fraction phi_s = chi*skeletonVolumeFraction_, and the
        //  fluid fraction phi_f = 1 - phi_s. They sum to one by construction,
        //  which is what makes grad(p) cancel in the mixture momentum balance.
        tmp<volScalarField> solidFraction() const;

        //- The solid fraction the network-pressure FORCE is evaluated with,
        //  per phaseForceBasis_. Never used for the deviatoric weight.
        tmp<volScalarField> forceFraction() const;

        //- The phase INDICATOR, per phaseIndicatorBasis_. A marker in [0, ~1],
        //  NOT a volume fraction: it is theta_ref times larger than
        //  solidFraction() and the two must never be interchanged. Unclamped
        //  above; see phaseIndicatorBasis_ for why, and for the one site that
        //  keeps its own clamp.
        tmp<volScalarField> phaseIndicator() const;

        //- Face flux of Vs_, with no penetration through walls.
        tmp<surfaceScalarField> skeletonFlux() const;

public:

    TypeName("phaseClotAxisymmetricFluid");

    explicit phaseClotAxisymmetricFluid(fvMesh& mesh);

    //- Maximum time step, bounded by the MATERIAL Courant number as well as
    //  the fluid one. maxCo sizes dt from the fluid alone, and once the phase
    //  field is advected on v = chi*v_s + q*u that is the wrong velocity:
    //  run 20260815-182036 reached vsCourant 1.45 and climbing while the
    //  fluid Courant sat inside its 0.5 bound, so the clot was being
    //  transported at nearly three times the accuracy limit with nothing in
    //  the timestep control able to see it.
    virtual scalar maxDeltaT() const;

    virtual void prePredictor();
    virtual void momentumPredictor();
    virtual void pressureCorrector() override;

    //- The pressure equation. incompressibleFluid declares this PROTECTED and
    //  VIRTUAL, and pressureCorrector() is only
    //  `while (pimple.correct()) correctPressure();` -- so the face-consistent
    //  momentum force is an override of ONE function and not, as an earlier
    //  handoff estimated, a reimplementation of the PISO loop.
    //
    //  Defers to incompressibleFluid::correctPressure() verbatim whenever
    //  faceConsistentElasticForce_ is off, so the control arm of the A/B is the
    //  base code itself rather than a re-typed copy of it.
    virtual void correctPressure() override;

    virtual void postSolve();
};

} // End namespace solvers
} // End namespace Foam

#endif
