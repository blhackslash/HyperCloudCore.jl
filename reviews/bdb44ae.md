# Review: HyperCloudCore.jl — MUSCL: cache the divergence fit of the degree each particle uses (bdb44ae)
Reviewer: Claude Opus 5.5 (Anthropic), new session without the generating conversation

Independent review of the one commit `claude/review-numerics..claude/muscl-cache-degree` (d45d394..bdb44ae), which
addresses `reviews/622b9d9.md` F4 (medium) and `reviews/d92911e.md` R3-F4 (docs). Probes, outputs and logs referenced
below are in `/tmp/review_pr3/` (`probes/*.jl`, `probes/*.out`, `fullsuite.log`, `parent_schemes.log`). No file of
the worktree was edited; the scratch parent worktree (`/tmp/review_pr3/parent`, detached at `claude/review-numerics`)
has been removed again.

## Scope

Read completely: the full diff (6 files); `src/Interpolation/Interpolators.jl` (basis generation, `basis_count`,
`build_basis`, `build_scale_factors`, `_cholesky_moment`/`_pivots_ok`, `_mls_divergence`, `_derivative_rows`,
`_intercept_basis`, `MLSWeights`, `update_weights!`, `_moment_body!`, `_padded_inverse_moment`, `_pad_moment`,
`_inverse_moment`, `apply_weights`, `apply_weights_divergence`); `src/Interpolation/MUSCL.jl` (whole file:
`_divergence_index`, constructor, `prepare_geometry!`, `update_divergence_weights!`, `_weights_body!`, moment weights,
`update_content!`, the functor and `cached`, `_flux_loop`, `_slot_weight`, `_dispatch_streamed_dynamic`);
`MUSCL_hooks.jl`; `get_time_step`, `_cfl_body!`, `_cfl_weights_body!` in `src/Grid/ParticleGrids.jl`.
Skimmed: `TimeSteppers.jl` (step prologue, `compute_explicit_rhs!`, `_apply_stage_geometry!`), `GridMovers.jl`
(`refresh_geometry!`, `set_positions!`), `API.jl` (`prepare_geometry!` fallback), `MemoryBudget.jl`, `Backend.jl`
(resident-argument check), `test/unit_backends.jl` and `test/gpu_tests.jl` (callers of `get_time_step`), the docs
sections touched (`overview.md` CFL section incl. "Measured limits", `muscl.md` 75–125 and 180–230). Callers outside the
repo: `HyperCloud.jl/src/time_integration.jl` (`solve_equation`, `NoFallbackGrad`), `HyperCloud.jl/test/gpu/runtests.jl`,
the `get_time_step` uses in `/workspace/julia/Simulations` (harnesses, `experiments/*` CFL/radius settings);
PDEStudioDev has none.

## Tests run

- Full suite under the lock: `flock /tmp/hcc_test.lock timeout 60m julia -t 8 --project=test test/runtests.jl`
  → **exit 0, 1040 pass, 0 fail, 0 error** (core 460, schemes 344, pipeline 135, backends-a 25, backends-b 71, plus
  5 top-level; wall 3 min 23 s). Log: `fullsuite.log`.
- The branch's `test/unit_schemes.jl` against the parent code (scratch worktree, `HYPERCLOUDCORE_TEST_GROUP=schemes`,
  `-t 2`): **333 pass, 4 fail, 1 error** — the new testset fails 3 of its 4 configurations (passes `(3, 3.5)`, where
  nothing changes), the CFL testset fails at `dw !== nothing` (parent `get_time_step` does not prepare the cache) and
  then errors on `dw[k]` (`nothing`). Every other testset passes on the parent. Log: `parent_schemes.log`.
- Probes (`julia -t 2 --project=<tree>/test probes/<probe>.jl <tree> <tag>`, both trees where marked):

| probe | what | result |
|:--|:--|:--|
| `probe_padded.jl` | (a) graded prefix of `_fit_basis`, D = 1..3, IO ≤ 5 (3-D: 4), both intercept flags; (b) `_padded_inverse_moment(q)` vs `_inverse_moment(Val(q))` padded, every particle of jittered clouds, q = 0..IO+1, D = 1..3, F64/F32, C = false/true; (c) `@inferred`; (d) `dweights` vs per-slot weights of the dynamic fit of the degree each particle uses (12 configs: 1-D/2-D/3-D, lattice/jitter, walls, F32, `div_order = 4`, R = 1.2Δx) | (a) 0 mismatches; (b) **bitwise equal**, `ok` identical in all 12 combinations, q outside 1..IO−1 falls to the full degree; (c) inferred; (d) mixed degrees per particle (e.g. 2-D MUSCL5 walls: degrees 2/3/4 on 15/74/410 particles), max rel. difference 4e-16…2e-15 (F64, well-conditioned), 2.5e-9…2.6e-8 on regularised/near-wall stencils, 1.5e-6/1.4e-5 (F32); `ok` never differs |
| `probe_alloc.jl` | `_moment_body!` with/without degree function in a function barrier, `prepare_geometry!` after a version bump; 1-D/2-D/3-D up to IO = 5 | **0 bytes** everywhere; `_FullOrderDivDegree` is bits |
| `probe_rhs.jl` (both) | 36 fixed-dt runs (5 RK3 steps, same dt on both trees): 1-D MUSCL2–5 × R 2.5/3.5 × lattice/25 % jitter, streamed, aligned, aligned+streamed, `div_order = 3`, MUSCL1 moment correction, VK + MOODu1 (EPD1, EPD0), walls, Float32, 2-D MUSCL3 R 2.0/2.5 and MUSCL5 R 2.5 (periodic, walls+aligned+streamed, VK+MOOD, F32+aligned), 3-D MUSCL3 R 1.8 and MUSCL4 R 2.0 streamed | branch vs parent `max|Δu|/max|u|` ≤ **1.4e-15** (F64), 6e-8/1.2e-7 (F32); bitwise wherever no particle is lowered (`cmp_rhs.out`) |
| `probe_moving.jl` (both) | non-uniform `PrescribedMover`, frozen topology and full search per stage, 1-D MUSCL3, 2-D MUSCL3 R2.0 / MUSCL5 R2.5, fixed dt + CFL-driven Δt sequence | states ≤ 4e-15; Δt see F1 (the LagrangianMover case of `probe_rhs.jl` is vacuous for linear advection: zero relative flux) |
| `probe_sumc.jl` | per-particle Σ\|c\|Δx of the fit in use, of the full-degree fit the parent cached, and Δt of the cached and `cached_weights = false` paths, 16 configs | table in F1 |
| `probe_dt.jl` (both) | runs to the end with `Δt = CFL · get_time_step` every step: 1-D advection (sine, square wave with VK ± MOODu1), 1-D Sod (MUSCL3 R2.5 VK + PAD/u1, jittered; MUSCL5 R3.5), 1-D MUSCL5 R3.5 RK4, CFL 0.4/0.6/0.8/1.0; 2-D advection MUSCL5 R2.5 jit 0.2 and 2-D blast (MUSCL3/5, R2.5, jit 0.2, VK + PAD), CFL 0.4/0.8 | F1 |
| `probe_limits.jl` (both) | 1-D blow-up CFL (RK4, smooth) and overshoots (VK, no MOOD, RK3/RK2) | F1 |
| `probe_starved*.jl` (both) | particles with fewer than D + 1 neighbours (1-D R 1.2Δx jit 0.45, 2-D R 1.2Δx jit 0.4) | F2 |
| `probe_ka.jl` | KA CPU (PoCL) vs Polyester: `get_time_step` (now launching `prepare_geometry!`), `inv_moment`, `dweights`, 3 RK3 steps; 1-D MUSCL3 R2.5 jit, 1-D MUSCL5 aligned streamed walls, 2-D MUSCL3 R2.0 jit VK aligned, 2-D MUSCL5 R2.5 jit streamed, 3-D MUSCL3 R1.8 jit; F64/F32 | **10/10 pass**: states ≤ 1.9e-15 (F64) / 6.9e-7 (F32), Δt ≤ 1.1e-15 / 2.1e-6, `dweights` ≤ 3.1e-14 / 1.2e-5, `inv_moment` ≤ 1.1e-13 / 3.2e-5 relative |
| `probe_compile.jl` (both, fresh process each) | compile time of the first `prepare_geometry!`, `get_time_step`, step | F3 |
| `benchmark/step.jl` (both, under the lock, `-t 8`, 3 repetitions) | 1-D N = 2e5 MUSCL3 R2.5, `stab = both/none`; 2-D 250² | 1-D: **9.1–9.6 → 7.2–7.5 ms/step** (both), 5.7–6.0 → 3.8–4.0 (none); 2-D 7.96 → 8.08 (noise). Same relative gain as the commit message's 16.0 → 12.1 |

## Findings

### F1 (medium, CONFIRMED) — the Δt change is larger and wider than the commit message says; at CFL ≥ 0.6 limited 1-D MUSCL3 (RK3, R = 2.5Δx) now overshoots, and the docs' CFL guidance is not updated
`src/Grid/ParticleGrids.jl:392` (`prepare_geometry!` in `get_time_step`), `src/Interpolation/MUSCL.jl:179`;
docs `docs/src/particle_grids/overview.md:87-99` ("Measured limits", "in 1-D stay at or below 1").
The change is what F4 asked for and the step is now the documented bound for the weights in use. But the time step of
default configurations changes, in both directions, and the commit message only mentions the 1-D MUSCL3 case
("2-D 250² unchanged" is a timing of a lattice MUSCL3 run). Measured (`sumc.out`, `dt_*.out`, `moving_*.out`; dt·n =
Δt/Δx at unit speed):

| configuration | parent | branch |
|:--|:--|:--|
| 1-D MUSCL3 R2.5, lattice / 20 % jitter | 0.667 / 0.581 | **1.553 / 1.431** (2.33× / 2.46×) |
| 1-D Sod MUSCL3 R2.5 (first / second step) | 0.563 / 0.468 | 1.313 / 0.959 |
| 1-D MUSCL5 R3.5 (first / later steps) | 0.857 / 0.545 | 0.857 / 0.857 (first step unchanged, later 1.57×) |
| 1-D MUSCL5 R2.5 | 0.678 (cached) | 0.667 (−1.7 %) |
| 2-D MUSCL5 R2.5 lattice (first / later steps) | 0.139 / 0.170 | 0.139 / **0.139 (−18 %)**: the degree-4 fit in use has Σ\|c\|Δx 3.59, the cached degree-5 fit 2.95 |
| 2-D MUSCL5 R2.5, 20 % jitter (first / later steps) | 0.135 / **0.00018** | 0.112 / 0.112 |
| 2-D MUSCL3 R2.0, 25 % jitter | 0.0005 (Taylor, first step); cached full degree Σ\|c\|Δx up to 501 | 0.089 |
| 2-D MUSCL3/MUSCL4 R2.5, 20 % jitter; 3-D MUSCL3 R2.5 | — | unchanged (no particle lowered) |

The 2-D jittered MUSCL5 R = 2.5Δx row is a large improvement the commit does not claim: on the parent the cached
degree-5 fit is under-determined (20 neighbours, 21 unknowns, Tikhonov retry) and its weights reach Σ|c|Δx = 3414, so
after the first step Δt collapses by 750× (`probe_dt.jl`: 20001 steps without reaching t = 0.5 at CFL 0.4; the
blast at CFL 0.8: 14060 steps in 128 s vs 91 steps in 0.6 s on the branch). Worth stating in the PR.

The stability side: the physical Courant number of the affected 1-D configurations grows with the step
(MUSCL3 R2.5: λΔt/Δx = CFL/0.644, i.e. 1.24 at CFL 0.8), and the "LED bound" holds for the first-order update only.
Square wave, VK (`:hard`), no MOOD, RK3 (`limits_*.out`):
- branch MUSCL3 R2.5: range [−0.0008, 1.0008] at CFL 0.6, **[−0.0053, 1.0053] at 0.8, [−0.0136, 1.0136] at 1.0**;
  parent: [0, 1] up to CFL 1.0. MUSCL2 R2.5 (the same Σ|c|) and MUSCL3 with RK2 stay in [0, 1] up to 1.0, so it is
  the degree-2 reconstruction with RK3 beyond Courant ≈ 0.9.
- with MOODu1 the overshoot is caught, but the error grows: square wave at CFL 1.0 rel. L2 0.354 vs 0.190; Sod
  MUSCL3 R2.5 VK + MOODu1 L1(ρ) 1.255e-2 vs 1.092e-2 at CFL 0.8 (+15 %), 1.427e-2 vs 1.108e-2 at 1.0 (+29 %). With
  PAD only: +1.7 % / +6 %. No run lost positivity or blew up (1-D Sod and 2-D blast at CFL ≤ 1, min ρ/p as on the
  parent).
- smooth advection, RK4, no limiter: MUSCL3 R2.5 now runs to CFL 3.0 and blows up at 4.0 (parent ran at 4.0);
  MUSCL5 R3.5 blows up at 4.0 (parent: 2.3e-5 at 4.0). The 1-D row of the "Measured limits" table (runs at 1.5…3.0,
  fails at 2.0…4.0) does not say the radius, so it is unclear whether it still holds.
- smooth errors are unchanged at the tested CFL (1-D MUSCL3 R2.5: 3.30e-3 both); MUSCL5 R3.5 RK4 shows the time
  error (CFL 1.0: 1.84e-6 vs 1.51e-6), consistent with the commit's "EOC unchanged" for its R = 2.5Δx runs only.

Production configs checked in `/workspace/julia/Simulations`: the CFL 0.8 shock runs (DMR, Mach 3 step, Clain shock,
isentropic vortex) are 2-D, R = 2.5Δx, MUSCL2–4, jitter 0.2: unaffected (no particle lowered); the 1-D/2-D
convergence studies use R = 5.5Δx: unaffected. HyperCloud.jl's tests use R = 5.5Δx in 1-D and 2-D R = 2.5Δx with
orders ≤ 4: unaffected (not run).
- Fix: say in the PR/commit (and a NEWS/changelog line if there is one) which configurations change Δt and by how
  much (table above), including the 2-D MUSCL5 R2.5 collapse that is fixed and the −18 % on its lattice; in
  `overview.md` qualify "in 1-D stay at or below 1" (limited MUSCL3 + RK3 at R = 2.5Δx overshoots from CFL ≈ 0.6;
  RK2 or MOOD avoid it) and state the radius of the "Measured limits" rows, or re-measure the 1-D rows on the branch.

### F2 (low, CONFIRMED; pre-existing weakness, now visible in Δt) — rank-deficient starved stencils pass the pivot test; their fit is round-off and now sets the step
`src/Interpolation/Interpolators.jl:56-64` (`_pivots_ok`), `src/Interpolation/MUSCL.jl:22-32` (`_divergence_index`
stops at degree 1 whatever the neighbour count).
A particle with fewer than D + 1 neighbours gets the degree-1 intercept fit (D + 1 unknowns), which is singular. In 2-D
with R = 1.2Δx and 40 % jitter (`probe_starved*.jl`, 30 such particles) particle 126 has 2 neighbours, `eigvals(N) =
[9.2e-17, 0.41, 1.58]`, and the *plain* Cholesky is accepted (its last pivot passes `10 eps · N_kk`), so no Tikhonov
retry happens and the fit is round-off amplified by ~1e16: the cached `inv(C)` and the old `C \ E` give −17.89 vs
−10.82 for the same flux differences (65 %), and after 5 steps at a stable dt the states of branch and parent differ by
3.4e-4 (all other particles agree to ≤ 7e-8). The commit's "RHS equal to the old dynamic path to round-off" holds for
every well-conditioned stencil (F1 table, `probe_rhs.jl`), not for these. The CFL estimate now sums these weights:
Δt·n = 0.0114 vs 0.088 on the parent (7.7× smaller). A misconfiguration (radius too small for the jitter), hence low;
the same fragility exists in Central/Upwind, which instead zero the divergence below `B(D, q) + 1` neighbours.
- Fix (separate change): treat `num_nb < basis_count(D, 1) + 1` like Central (zero divergence or a first-order
  fallback), or make `_pivots_ok` relative to `tr(N)/B` rather than to the pivot's own diagonal entry; a warning at
  grid construction for particles with fewer than D + 1 neighbours (suggested in `reviews/d92911e.md` minors) would
  cover both schemes.

### F3 (low, CONFIRMED) — compile time of the generated padded inverse
`src/Interpolation/Interpolators.jl:544-552`. `_padded_inverse_moment` instantiates `_inverse_moment` (assembly,
Cholesky, `inv`) and `_pad_moment` for every degree 1..IO in every kernel it is compiled into (`probe_compile.jl`,
fresh process each, first `prepare_geometry!`): 1-D MUSCL5 1.2 → 1.6 s, 2-D MUSCL3 1.6 → 1.9 s, **2-D MUSCL5 10.3 →
13.8 s** (first step total 24.4 → 27.2 s, +11 %), 3-D MUSCL3 8.2 → 9.1 s, 3-D MUSCL4 13.2 → 14.3 s; on the KA backend it
is compiled again. CLAUDE.md names compilation as the dominant test-suite cost; the suite still ran in 3 min 23 s.
- Fix (optional): restrict the generated branches to the degrees `_divergence_index` can return for the scheme
  (`base − 1 .. full − 1` plus the floor, e.g. for MUSCL3 in 1-D/2-D only degrees 1..3 are reachable anyway, but for
  `DIV_ORDER > 0` only the starvation chain), or accept and mention it in the PR.

### F4 (low, docs/comments, CONFIRMED) — stale "full-degree" wording and one inaccurate docs sentence
- `src/Interpolation/MUSCL.jl:95` (field comment "Derivative weights c_ij of the full-degree divergence fit"),
  `:192` (`update_divergence_weights!` docstring, same), `:510` ("a particle whose fit is not the cached full-degree
  one"), `src/Interpolation/FluxFunctions.jl:86` ("from the cached full-degree fit"), `docs/src/spatial/muscl.md:188`
  ("the derivative weights c_ij of the full-degree fit per slot"): now the fit of the degree each particle uses.
- `docs/src/spatial/muscl.md:119-121`: "the weights of the dynamic fit for a particle off the cached path (a lower
  particle order with `div_order = 0`, …)". With the new `cached` condition (`MUSCL.jl:403`) a MOOD-lowered particle
  whose divergence index equals the one of its stencil at full order *is* on the cached path (e.g. MUSCL5,
  R = 2.5Δx in 1-D: an order-4 particle gets index 4 = the cached degree 3; checked to be the same fit). Suggest
  "a lower particle order whose divergence degree is lower than the cached one".
- `MUSCL.jl:91` ("They are the same object when both use the same degree") and `MemoryBudget.jl:44`: never true
  since the divergence fit has an intercept (pre-existing).

No other defects found. Checked and fine:
- **Padded inverse.** The basis is graded for every D ≤ 3 and degree (`_generate_exponents` appends degree by degree;
  the intercept is entry 1), so the zero-padded degree-q inverse applied to the full basis is the degree-q fit exactly:
  `_derivative_rows` (rows OFF+1..OFF+D are the degree-1 terms, inside every block q ≥ 1), `apply_weights_divergence`
  (full basis, padded zeros), `_weights_body!`, the CFL weights and the moment-correction weights (from `dweights`) all
  see the fit in use. `scales` are only used by `apply_weights` (reconstruction cache, no degree function); the
  divergence uses `invL` only. Same moment matrix and the same `_cholesky_moment` decision as the dynamic path, hence
  the same regularisation and `ok[i]` (verified bitwise, `probe_padded.jl`).
- **`cached`** (`MUSCL.jl:403`): for `DIV_ORDER > 0` `_divergence_index` ignores the particle order, so every particle
  with `ok[i]` is cached, as before; for `DIV_ORDER = 0` a particle is cached iff its index equals that of its stencil
  at full order, which is exactly the cached degree; MOOD-lowered particles with a lower index take the dynamic or
  streamed-dynamic path unchanged (`_dispatch_streamed_dynamic` still covers indices 2..full). `MAX_ORDER = 1/2`:
  degree 1 cannot be lowered, bitwise unchanged. Zero-weight slots of frozen-topology grids are counted the same way
  in the cache and in the flux pass.
- **Kernels/devices.** `_FullOrderDivDegree` is a singleton bits type; `_padded_inverse_moment` is a static branch
  chain, inferred, no allocation (0 bytes measured); KA CPU parity 10/10 incl. Float32 and 3-D. The device-rule
  constraints of CLAUDE.md (no static parameters inside bodies, bits arguments) hold.
- **`get_time_step` → `prepare_geometry!`.** The fallback `prepare_geometry!(::DivergenceInterpolator, pg) = nothing`
  covers Central/WENO/Upwind and HyperCloud's `NoFallbackGrad <: DivergenceInterpolator`; it is a cheap version check
  when current (`@allocated get_time_step < 8192` still passes); it touches neither gradients nor MOOD state, so a
  MUSCL that was never `update_size!`d works (the CFL testset does exactly that). Moving grids: `solve_equation` calls
  `get_time_step` after the previous step's motion/adaptivity, i.e. at the geometry of the next first stage, so the
  factorisations move from the first RHS to `get_time_step` and are not repeated (and replace the per-particle
  factorisation of the old uncached first-step estimate). Device grids: same object and kernels the stepper uses; the
  GPU test "host buffers are rejected" still gets its `ArgumentError` (now from the first `prepare_geometry!` kernel).
  Not run on a GPU here.
- **Docs of R3-F4.** The muscl.md note and the `get_time_step` docstring now describe the in-use weights; the
  first-step discrepancy of R3-F4 is gone (1-D MUSCL5 R3.5: dt0 = dt1 = 0.857).

## Addressed items

- `reviews/622b9d9.md` F4 (medium, cache and CFL degree ignore the neighbour-count safeguard) — **OK.** Every particle
  caches the degree it uses (mixed degrees per particle on jittered/walled clouds verified against the dynamic fit),
  the flux pass hits the cache (1-D MUSCL3 R2.5: 9.3 → 7.4 ms/step), `dweights` and the CFL weights are those of the fit
  in use. Consequences for Δt: F1; starved stencils: F2.
- `reviews/d92911e.md` R3-F4 (docs precision of the limitation note, first step uses the uncached fit) — **OK.**
  Note rewritten; `get_time_step` prepares the cache, so the first step uses the cached weights too.

## Test coverage

- The new testset ("the divergence cache holds the degree each particle uses") and the changed CFL testset fail on
  the parent (3 of 4 configurations; `dw !== nothing`), so they detect the old behaviour. The CFL bound
  `2 dt_i ≤ dt_c ≤ 2.6 dt_i` is meaningful (2.33 measured).
- Weak: the new testset checks particle 1 of a periodic 1-D lattice only, where all particles carry the same degree.
  Not covered: mixed degrees per particle (the reason the inverse is padded per particle), 2-D/3-D, Float32, that the
  flux pass actually takes the cached branch for lowered particles (e.g. compare `apply_weights_divergence` with
  `compute_dynamic_divergence` on a lowered particle, or the RHS of a cached vs a freshly invalidated cache), and
  `get_time_step` on a moving grid. `probe_padded.jl` part (d) on one 2-D MUSCL3 R = 2.0Δx cloud with 25–30 % jitter (degrees 2 and 3 mixed;
  cheaper to compile than 2-D MUSCL5) would cover the first two.
- The CFL testset's boundary-state and fast-interior checks (`dt_b`, `dt_i`) now run only with `cached_weights = false`;
  the default (cached) path is checked against the host sum for the fast interior state only, not for a fast
  boundary state. One more `dt_host` comparison with `pgc.rhos[b] = 1` would keep the pair-speed behaviour of the
  default path covered.
- The KA parity cases of `unit_backends.jl` already exercise the lowered path (1-D R = 2.5Δx MUSCL3, MUSCL5
  `div_order = 5`) and pass.

## Minor issues

- `cached_weights = false` now differs from the default by much more than "where the fits differ" suggests: 2.3× for
  1-D MUSCL3 R2.5, 170× for 2-D MUSCL3 R2.0 jittered (0.0005 vs 0.089), because `_cfl_body!` still uses the full
  degree whatever the stencil. Passing the per-particle safeguard degree to `dispatch_cfl` (it already dispatches on a
  runtime degree) would make the opt-out estimate the same fit up to Taylor-vs-intercept.
- `update_weights!(w, pg, degree_of)` keys the cache on `geometry_version` only; a call without `degree_of` on the same
  geometry would be taken as current with the wrong degrees. Internal API, only MUSCL calls it; a one-line comment
  would do.
- Muscl.md says "Up to 0.7.9"; the package version is still 0.7.9 on this branch (fine if the release bumps it, as for
  the other "up to 0.7.9" notes).

## Verdict

The padded inverse is exact (bitwise equal blocks, graded basis verified for every D ≤ 3), the cache now holds each
particle's fit at the degree its stencil carries, and the RHS equals the old dynamic path to ≤ 1.4e-15 in 36 fixed-dt
configurations (streamed, aligned, moment correction, MOOD, walls, moving grids, 1-D/2-D/3-D), Float32 to round-off,
allocation-free and in parity on the KA CPU backend; the 1-D MUSCL3 R2.5 step is 20–34 % cheaper. Calling
`prepare_geometry!` in `get_time_step` is safe for every caller found (HyperCloud driver, GPU tests, Simulations
harnesses). The main point is the behaviour change of Δt (F1): wider than the commit says (2-D MUSCL5 R2.5: −18 % on the
lattice, and a 750× step collapse of the parent fixed on jittered clouds), and in 1-D the 2.3× larger step lets limited
MUSCL3 with RK3 overshoot from CFL ≈ 0.6, which the docs' "1-D: CFL ≤ 1" guidance does not cover. Starved stencils
expose a pivot-test weakness (F2), the generated branches add compile time (F3), and some wording is stale (F4). Full
suite 1040/1040; the new tests fail on the parent.
Findings: 0 high, 1 medium (F1), 3 low (F2, F3, F4 docs); all confirmed.
