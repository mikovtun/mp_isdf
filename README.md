# Massively Parallel ISDF

A C++/Python prototype for **interpolative separable density fitting (ISDF)** of molecular AO products. PySCF supplies the molecule and integration grid; a pybind11 module evaluates atomic orbitals with libcint, selects interpolation anchors by pivoted QR, and builds Coulomb metrics. The Python driver fits the interpolation vectors and can compare reconstructed integrals or J/K matrices with PySCF.

The default metric is an **integral-direct, unblocked greedy Cholesky** decomposition of AO-pair electron-repulsion integrals (ERIs). An `unblocked_uniform` variant samples eligible pivots uniformly instead of always selecting the largest residual. Neither variant constructs the full AO-pair ERI matrix or calls LAPACK Cholesky. Independent libcint shell quartets and sufficiently large pair updates run on Kokkos host threads; the ordinary metric projection uses batched BLAS++ GEMMs. An approximate grid-based metric is also available. This is a research prototype, not an installed Python package.

## Density-weighted normalized anchor sketches

The density techniques `density` (raw control) and
`density_col_norm` (proposed method) under `rand_sketch` share the existing Gaussian
Khatri–Rao probes and host/device QRCP. For the normalized mode,

```text
q[g] = sum_mu AO[g,mu]**2
Y[t,g] = sqrt(m[g] / sketch_rows) * (u[t] @ AO[g]) * (v[t] @ AO[g]) / q[g]
```

The raw control omits `/ q[g]`. Both require `--axes ao`. By default `m=rho`
uses the same separate converged RKS B3LYP spin-summed density as density-CVT;
**quadrature magnitudes are not multiplied into this density-only objective**.
`--density-weight quadrature_density` instead uses `m=abs(w)*rho`, explicitly
extending the positive-weight proposal to signed grids. Signed physical weights
remain unchanged in the refit and Coulomb metric. Existing CVT's
`--cvt-weight quadrature_weights` continues to mean weights alone, not their product
with density.

```sh
OMP_NUM_THREADS=4 OMP_PROC_BIND=false python python/isdf.py \
  --anchor-algorithm rand_sketch --sketch density_col_norm --anchor-rank 2 \
  --anchor-oversampling 16 --ao-screening-threshold 1e-24 \
  --pair-space-error --compare

# One grid and density, equal ranks/sketch sizes, repeated seeds and cutoffs:
OMP_NUM_THREADS=4 OMP_PROC_BIND=false python python/compare_density_sketches.py \
  --rank 2 --seeds 7 11 23 --thresholds 0 1e-24 1e-18 \
  --output density_sketch_comparison.json
```

`build_isdf(..., point_importance=m)` accepts the **complete** nonnegative
importance aligned with the original grid, overriding generated importance for
CVT and either density sketch. `grid=` can supply a prebuilt PySCF grid for an
exactly shared comparison. At the extension API use
`RandomizedAODensityKhatriRao` or `RandomizedAODensityNormalizedKhatriRao`,
`point_importance=m`, and optionally `ao_screening_threshold`.
Zero quadrature weight, zero importance, and squared AO norms at or below the
threshold exclude sites from density-sketch selection. Original grid indices
are preserved. Screening does not remove nonzero-quadrature sites from the
physical refit. Explicit rank remains exact (an infeasible rank raises);
tolerance-only selection uses the sketch Frobenius residual, not a J/K guarantee.

All ordinary scalar refits now solve in physical pair space by SVD (relative
singular-value cutoff `1e-12`) rather than taking a pseudoinverse of a Gram matrix.
This deliberately changes rank truncation for ill-conditioned anchors. Stored
interpolation vectors and metric transforms still use the existing
anchor-weighted convention; no density/normalization factors enter the physical
fit. `rbrp_id` retains its own ID path. `fit_diagnostics` reports selected physical
column singular values, condition, retained rank, and maximum physical coefficient.

With `pair_space_error=True`, density/CVT runs also report
`pair_space_errors['density_normalized']`: the density-weighted relative-column
reconstruction residual on points above the AO cutoff, plus included/screened
importance mass. This is distinct from the ordinary `abs(w)` physical pair
residual and sketch stopping residual. Small-case `anchor_audit` reports physical
projection errors and a separate objective-matched direct-QR comparison.

The comparison runner fixes the physical refit, metric, and RHF J/K comparison
density for all variants. JSON contains separate J/K matrix and restricted-energy
errors (`E_J=Tr(DJ)/2`, `E_K=-Tr(DK)/4`), pair residuals, conditioning/coefficient
diagnostics, screening counts, timings, and anchor coordinates for spatial
inspection. Nonfinite diagnostic values are serialized as `null`. The common
importance density is RKS B3LYP; it is intentionally distinct from the common
RHF density used to measure J/K errors. Spatial separation is not enforced, and
improved normalized residuals alone are not evidence of improved J/K accuracy.
See [the proposal](density_weighted_normalized_sketch.md) for the objective.

## Build

Requirements: a C++20 compiler, CMake 3.21+, Git, a BLAS/LAPACK implementation, and Python with NumPy and PySCF. CMake fetches Kokkos, BLAS++, LAPACK++, SLATE, libcint, and pybind11; the initial configure requires network access. OpenMP is enabled by default.

For a CPU-only build without MPI or CUDA:

```sh
python -m pip install numpy pyscf
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DISDF_ENABLE_CUDA=OFF -DISDF_ENABLE_MPI=OFF \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build --target mp_isdf_python -j 4
```

The Python executable used to run the driver must match the interpreter used to build `mp_isdf`. The driver imports the extension directly from `build/`; no installation step is needed. Set `ISDF_BUILD_DIR=/absolute/path/to/build-cpu` to select a different build for the driver and comparison runner.

CUDA and MPI are **on by default**. For a CUDA build, omit `-DISDF_ENABLE_CUDA=OFF`, provide a CUDA toolkit and a suitable `CMAKE_CUDA_ARCHITECTURES` value; the build also obtains and builds MAGMA. MPI may be disabled independently. `--space host` forces host AO evaluation and anchor selection, while `--space device` requires a CUDA build. The integral-direct Cholesky metric itself runs on the host because libcint evaluates its quartets on the CPU.

## Run

```sh
# Default H2 / STO-3G example
OMP_PROC_BIND=false python python/isdf.py --anchor-rank 3

# Run the AO fit and overlap check without either Coulomb metric
OMP_PROC_BIND=false python python/isdf.py \
  --molecule python/xyz/ethanol.xyz --basis sto-3g \
  --anchor-rank 8 --no-metric

# Compare J and K with PySCF
OMP_PROC_BIND=false python python/isdf.py --anchor-rank 3 --compare

# Reconstruct and compare a mixed-derivative four-index ERI
OMP_PROC_BIND=false python python/isdf.py \
  --anchor-rank 3 --derivative-target 00xy --compare

# Sample Cholesky pivots uniformly, with a reproducible seed
OMP_PROC_BIND=false python python/isdf.py \
  --anchor-rank 3 --metric unblocked_uniform --metric-seed 17

# Log pivots, residuals, shell-quartet counts, and timings
OMP_NUM_THREADS=4 OMP_PROC_BIND=false python python/isdf.py \
  --anchor-rank 3 --debug
```

The XYZ files in `python/xyz/` can be passed with `--molecule`; omitting it selects H2. Run `python python/isdf.py --help` for all options. For a user-oriented overview of every anchor selection algorithm and when to choose it, see [docs/anchor-selection.md](docs/anchor-selection.md). Common controls:

| Option | Meaning |
| --- | --- |
| `--axes ao:1,dx:1,occ:2` | Comma-separated features for **anchor selection**, each with an optional positive relative factor (default `1`); `occ` requires a converged closed-shell RHF calculation. The default is `ao`. With multiple axes, each feature block is divided by its Frobenius norm over nonzero-quadrature-weight grid points, then multiplied by its factor, before taking the outer product. |
| `--anchor-rank N`, `--anchor-tol T` | For `rand_sketch`, rank forces exactly N pivots; tolerance mode stops by the sketch residual. They are mutually exclusive. If neither is supplied, tolerance defaults to `1e-10`. `qrcp_dense`/`qrcp_online` retain their existing tolerance and rank-cap semantics. |
| `--anchor-algorithm qrcp_dense\|qrcp_online\|rand_sketch\|cvt\|rbrp\|rbrp_id` | Dense QRCP (default), online QRCP, Khatri–Rao sketch, CVT, or AO-only RBRP. `rbrp_id` uses Algorithm 5.1's ID fit. |
| `--sketch quadrature\|density\|density_col_norm` | Optional with `rand_sketch` (default: `quadrature`). Quadrature supports all axis features; density techniques currently require `--axes ao` (other axes NYI). |
| `--anchor-batch-size N` | Grid points per batch for `qrcp_online`, `rand_sketch`, or RBRP (default `128`). |
| `--fit-batch-size N` | Grid points per interpolation-vector fit batch (default `2048`); independent of anchor-selection batching. |
| `--rbrp-block-size N`, `--rbrp-tau-block T`, `--rbrp-rank-rtol T`, `--rbrp-refresh-every N`, `--rbrp-svd-rtol T` | RBRP candidate block size (16), trailing QR-block energy filter (default `1/block_size`, or 0 for size 1), numerical rank floor (`1e-12`), residual refresh frequency (8), and Algorithm 5.1 small-matrix SVD cutoff (`1e-12`). |
| `--anchor-sketch-size E`, `--anchor-oversampling P`, `--anchor-seed S` | With explicit rank `R`, randomized sketch rows are `R + P` (`P=16` by default); `E=128` is used only when `R=0`, and then requires `E > P`. Seed defaults to `7`. |
| `--cvt-weight electron_density\|quadrature_weights`, `--cvt-max-iter N`, `--cvt-tol T` | CVT uses a separate converged RKS B3LYP spin-summed electron density by default, or absolute quadrature weights (not their product); runs up to 100 Lloyd iterations with relative centroid-shift tolerance `1e-5`. Requires `--anchor-rank > 0`. |
| `--anchor-audit` | Small-case optional comparison of randomized pivot projection error to dense QRCP at equal rank; rejects pair matrices above 2 million entries. |
| `--level N`, `--space auto\|host\|device` | PySCF grid level and AO/QR execution space. |
| `--metric unblocked_greedy\|unblocked_uniform\|grid` | Largest-residual pivoting (default), uniformly sampled eligible pivots, or the approximate spherical-cell grid metric. |
| `--metric-tol T`, `--metric-rank N` | Relative Cholesky diagonal-residual cutoff and optional rank cap (`0` means no cap). Ignored for `grid`. |
| `--metric-seed N` | Unsigned 64-bit random seed for `unblocked_uniform` (default `0`); ignored by greedy and grid modes. |
| `--metric-batch-size N` | AO projection rows per BLAS++ batch (default `64`); changes scratch use and host parallelism. |
| `--derivative-target 00xy` | Request a four-AO-slot derivative target; repeat the option for multiple targets, use `0000` for ordinary ERIs, or use `all` for all 54 two-derivative spatial targets. |
| `--no-metric` | Skip both scalar and derivative metrics but retain AO fitting and overlap validation. |
| `--pair-space-error` | Check the grid-weighted L2 reconstruction of ordinary AO products and each distinct requested derivative pair channel; works with `--no-metric`. |
| `--compare` | With explicit targets, compare reconstructed four-index ERIs against PySCF; otherwise compare ISDF J/K against PySCF. Requires a metric. |
| `--debug` | Emit Cholesky rank/ERI-quartet/residual diagnostics to stderr. Print the timing report at the **end** of stdout, after comparisons. |

The derivative-target string has four slots `(u,v,w,x)` drawn from `0xyz`. Exactly two nonzero letters denote first electronic-coordinate derivatives acting on those **individual AOs**; these are not nuclear gradients or assembled spinor integrals. `axes` does not change which AO slots are differentiated. Derivative pair channels are fitted independently using the selected anchors.

For AO-only robust blockwise random pivoting (Dong et al., `papers/2309.16002v5.pdf`, Algorithm 4.1), use `--anchor-algorithm rbrp --anchor-rank 40` to select anchors and retain the existing NumPy scalar fit, or `--anchor-algorithm rbrp_id --anchor-rank 40` to construct scalar interpolation vectors **and integral-direct metric coefficients** from the Algorithm 5.1 ID (`L` and `Q`), bypassing the NumPy scalar fit. The host backend uses C++ (LAPACK++ block QR/SVD and batched packed AO-pair rows), supplied with host AO values by the Python driver. In CUDA builds, `--space device` uses bounded-batch Kokkos CUDA AO evaluation and packed-pair generation; residual state, projections, and ID products remain on the device, with MAGMA GPU QR/BLAS and hybrid small-matrix MAGMA SVD. The small QR pivot/R diagnostics and `L1` SVD move only block-sized data to the host. `--space auto` uses the device RBRP backend in CUDA builds and the host backend otherwise; `--space host` always selects the host backend. Non-`ao` axes are rejected. `python/rbrp_id.py` remains an independent NumPy/SciPy reference. It never stores the full grid-by-AO-pair matrix but retains `O(grid*rank + packed_pairs*rank)` factors. Grid points with zero weight are excluded. Selection uses `sqrt(abs(weight))` and an absolute-weight pair-space Frobenius objective; the downstream overlap and Coulomb metric continue to use signed weights. `--anchor-tol` is a relative **norm** tolerance (`tau=anchor_tol**2` in the paper) and `--anchor-rank` is a cap, not a guarantee of exactly that rank. The result includes `rbrp_diagnostics` with achieved projection residual, stop reason, block history and (for `rbrp_id`) retained SVD rank/condition. SVD truncation may break exact least-squares equivalence. Derivative channels continue to use their existing separate fits.

For centroidal Voronoi selection, pass `--anchor-algorithm cvt --anchor-rank 40` (or `anchor_algorithm="cvt", anchor_rank=40` in `build_isdf`). The objective is the sum of `importance * squared_distance` to the nearest generator, where importance is either RKS B3LYP electron density or absolute quadrature weight (never their product). Grid sites with zero weight or zero importance are excluded before clustering and from final snapping; output is sorted, distinct indices into the original grid. A separate RKS B3LYP calculation supplies the spin-summed density unless `--cvt-weight quadrature_weights` is specified. `--anchor-seed` controls weighted k-means++ initialization; `--cvt-max-iter` and `--cvt-tol` control Lloyd relaxation. `--anchor-tol` is not used by CVT. Kokkos accelerates nearest-center assignment, weighted accumulation, and seed-distance updates on the selected host/device space; sampling, empty-cluster repair, and distinct-site snapping run on the host. Cost is `O(Ngrid * rank * iterations)` with `O(Ngrid + rank)` scratch (excluding the input grid). Floating-point atomic accumulation can produce small host/device differences near ties. At the extension API pass `algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.CentroidalVoronoi`, `max_rank=rank`, and `cvt_weights` (the nonnegative importance array aligned with the original grid), optionally `cvt_max_iter`, `cvt_tol`, and `seed`. The extension does not itself run RKS.

For Khatri–Rao sketch pivot selection, pass `--anchor-algorithm rand_sketch --anchor-rank 40 --anchor-oversampling 16` (or `anchor_algorithm="rand_sketch"` in `build_isdf`). The default sketch is `quadrature`; select another with `--sketch` or `sketch=`. `quadrature` sketches pair products of the requested `--axes` features, with one `sqrt(abs(grid_weight))` factor per column; for multiple axes each feature block is globally normalized and may be reweighted with `--axes`. This is the absolute-weight pair-space objective for AO-only input. `density` and `density_col_norm` use nonnegative density importance instead of quadrature weights, with the latter normalizing each pair column. Both density techniques are AO-only for now; non-AO axes raise an NYI error. The two Gaussian factor matrices are independent and shared across grid batches; the full pair matrix is never materialized. Sketch selection returns anchors only: the usual batched physical NumPy refit follows. `anchor_tol` is a relative **sketch Frobenius-residual tolerance**, not a physical pair-space or J/K guarantee. With an explicit rank, exactly that many pivots are requested (subject to eligibility and numerical rank); sketch rows equal `anchor_rank + anchor_oversampling`. Without an explicit rank, `anchor_sketch_size` sets the sketch rows and `anchor_sketch_size - anchor_oversampling` caps the rank. `qrcp_dense`/`qrcp_online` instead use rank as a cap. The optional `--anchor-audit` compares physical pair projection errors with dense QRCP only on small problems. Signed grid weights enter quadrature selection via their absolute values.

For bounded-batch anchor selection, pass `--anchor-algorithm qrcp_online` to the Python driver (and optionally `--anchor-batch-size N`), or use `anchor_algorithm="qrcp_online"` and `anchor_batch_size=N` in `build_isdf`. At the extension API, pass `algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.OnlineQRColumnPivoting` and optionally `batch_size=128` to `AnchorSelection.select(...)`. Dense LAPACK/MAGMA QRCP remains the default. Online QRCP makes a bounded preliminary host pass to compute global per-axis Frobenius norms for multi-axis selections, then uses Kokkos AO evaluation and pair-product generation in the requested `space="host"` or `space="device"` (CUDA build), streaming grid batches instead of building the full pair-product matrix. It keeps one Householder reflector per selected anchor, incrementally updates column norms, selectively refreshes unreliable estimates, verifies each candidate directly, and performs a full direct scan before stopping. Like `qrcp_dense` selection, its cutoff is relative to the largest initial column norm; it also stops at a floating-point numerical-rank floor. Zero-weight grid points are skipped, and signed weights are treated as `sqrt(abs(weight))`. With `M=axis_width²`, `N=nonzero_grid_points`, batch size `b`, and selected rank `k`, peak selection working storage is approximately `O(M*k + M*b + N + axis_width*b)` (plus the input grid and AO context; occupied-orbital axes also stage the `nao²` MO coefficients per batch); rank-unlimited selection can still consume substantial reflector memory. This is multipass and can be considerably slower than dense QRCP, particularly on GPU where reflector application currently processes one column per work item. With `--debug`, it reports the selected rank, dimensions and batch size, initial and final maximum residuals, full-scan count, candidate verification/reselection counts, norm-update and refresh counts, and elapsed selection time. The driver subsequently fits interpolation vectors in bounded grid batches, rather than forming the grid-wide AO-pair matrix.

The grid metric evaluates pairwise grid distances and is primarily a small-system/reference alternative; its cost grows quadratically with the number of grid points. Explicit `--compare` reconstructs a four-index tensor, so it should also be used on small systems.

## Python API

With the repository root as the working directory:

```python
import sys
sys.path.insert(0, "python")

from pyscf import gto
from isdf import build_isdf, evaluate_isdf_jk

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
result = build_isdf(mol, level=0, anchor_rank=3,
                    anchor_algorithm="qrcp_online", anchor_batch_size=128,
                    derivative_targets=("00xy",))

print(result["anchor_indices"], result["overlap_relative_error"])
print(result["coulomb_metric"].shape)
print(result["target_factors"]["00xy"]["coulomb_metric"].shape)
```

`build_isdf` returns anchor indices, coordinates and weights; interpolation vectors; an overlap reconstruction and relative error; the ordinary anchor-weighted Coulomb metric; and `timings`/`elapsed`. With `pair_space_error=True`, `result["pair_space_errors"]` contains `{"l2": ..., "relative": ...}` for `"scalar"` (ordinary AO products) and each distinct ordered channel used by `derivative_targets` (e.g. `"x0"`, `"0y"` for `"x00y"`). Otherwise it is `None`. This check fits and measures requested derivative channels even when `compute_metric=False`; without a metric no derivative ERIs or target factors are produced. `selection_only=True` cannot be combined with the check. `X1`–`X4` describe the ordinary AO fit. `target_factors[target]` contains the four target-specific anchor matrices and their anchor-weighted cross metric; `derivative_integrals[target]` contains the reconstructed four-index tensor. Pass `compute_metric=False` to omit both kinds of metric, or `selection_only=True, compute_metric=False` to stop after anchor selection. The `mp_isdf` extension also exposes `LibcintContext`, `AnchorSelection`, `cholesky_eri`, `coulomb_metric`, and `derivative_cross_metric` for lower-level use.

The pair-space diagnostic compares the fitted reconstruction `Theta_ab.T @ (W @ P_ab(anchor))` with `P_ab(grid)` across **all ordered AO pairs**. Its absolute error is `sqrt(sum_g abs(grid_weight[g]) * sum_pairs abs(exact - reconstructed)**2)`; the relative error divides by the same norm of the exact pair products. Absolute grid weights define a nonnegative discrete L2 norm even for signed quadrature weights. This diagnoses the real-space fit, not the Coulomb metric or ERI approximation. The `"scalar"` result uses the ordinary fit; a derivative target's `"00"` channel is fitted separately and can have its own error. The choice of `axes` affects anchor selection but does not change which requested channels are checked.

## How derivative fitting works

A target is split into two **ordered AO-pair channels**. For example, `x00y` pairs `(∂x φμ) φν` on the left with `φλ (∂y φσ)` on the right; `00xy` instead pairs `φμ φν` with `(∂x φλ)(∂y φσ)`. Here each derivative acts on the indicated AO only—not on the whole pair by a product rule. Reversing the two AO slots can therefore change a channel.

For each distinct channel `ab`, the driver evaluates pair products at the selected anchors and contracts grid AO values in bounded batches. Conceptually, write these products as `P_ab(grid)` and `P_ab(anchor)`, with one column for **each ordered** `(μ,ν)` pair; `P_ab(grid)` is never materialized in full. Let `W` be the diagonal matrix of anchor quadrature weights. The channel fit is computed independently as

```text
C_ab     = pinv(W @ P_ab(anchor)).T
Theta_ab = P_ab(grid) @ C_ab.T
```

`Theta_ab` contains the channel's interpolation vectors on the grid. Its returned storage is still `O(Nanchor * Ngrid)`; anchor-side coefficients needed by the integral-direct metric still occupy `O(Nanchor * nao²)`. Channels share anchor *locations* but not their fitted coefficients: even a `00` channel requested by a derivative target is fitted in this channel path rather than reusing the ordinary scalar fit. Repeated channel labels are cached within one `build_isdf` call.

For the integral-direct metric, libcint evaluates the required differentiated ERIs on demand. Joint greedy Cholesky over the two ordered pair spaces gives the cross metric `C_ab @ K_ab,cd @ C_cd.T` without forming the dense pair-ERI matrix. The grid-metric alternative contracts `Theta_ab` and `Theta_cd` against its grid Coulomb kernel. In either case, the driver applies an anchor weight on **each** side of the cross metric to produce `Z_ab,cd`, then reconstructs

```text
ERI_ab,cd ≈ P_ab(anchor).T @ Z_ab,cd @ P_cd(anchor)
```

`target_factors[target]["X1"]` through `["X4"]` hold the appropriate AO or AO-gradient values at the anchors, and `["coulomb_metric"]` is `Z_ab,cd`; `derivative_integrals[target]` holds the resulting `(nao, nao, nao, nao)` tensor. This final tensor is materialized for each requested derivative target, so requesting `all` can use substantial memory. Use `--compare --derivative-target ...` on a small system to check a target against PySCF.

## Tests and performance notes

```sh
OMP_NUM_THREADS=4 OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s python/test -q
```

Pivot iterations are sequential; work *within* each pivot is parallel. For `unblocked_uniform`, each pivot is drawn with equal probability from AO pairs whose current residual is strictly positive and above `metric_tol × initial_max_residual`. The stopping check still uses the **largest** remaining residual; a rank cap can give a worse approximation than greedy at the same rank. The seed makes repeated runs reproducible on the same C++ standard-library implementation. Debug output reports attempted shell-quartet calls, not the number of nonzero integrals, and these counts do not decrease with rank because shell screening has not been implemented. Factors and residuals remain host-resident. Factor storage grows with the Cholesky rank, and the metric projection builds only a batch of packed AO-pair coefficients at a time.

BLAS++ delegates threading to its linked BLAS implementation. If CMake selects a sequential BLAS (as in a development build using sequential MKL), Kokkos distributes independent projection batches across host threads. A threaded BLAS may benefit from different thread and batch settings; avoid oversubscribing threads. `--debug` separates pivot integral/update timings from post-pivot projection and metric-GEMM timings.

## Repository layout

- `python/isdf.py` — ISDF driver, CLI, grid metric, and PySCF comparisons.
- `python/test/` — AO evaluation, anchor, metric, and derivative tests.
- `cpp/src/cholesky_eri_greedy.cpp`, `cpp/src/cholesky_eri_uniform.cpp` — ordinary integral-direct Cholesky build and uniform-pivot entry point.
- `cpp/src/coulomb_metric.cpp`, `cpp/src/derivative_metric.cpp` — ordinary metric projection and derivative integral-direct metric.
- `cpp/include/cholesky_update.hpp` — host pair correction and deterministic next-pivot reduction.
- `cpp/src/anchor_select_qrcp.cpp` — host LAPACK++ / optional device MAGMA dense QRCP.
- `cpp/src/anchor_select_online_qr.cpp` — bounded-batch, multipass Kokkos Householder QRCP.
- `cpp/src/anchor_select_rbrp.cpp` — host AO-only blocked residual-weighted sampling and Algorithm 5.1 ID (LAPACK++ local QR/SVD).
- `cpp/src/anchor_select_rbrp_device.cpp` — CUDA AO-batch evaluation, MAGMA QR/BLAS and hybrid small-SVD RBRP/ID.
- `cpp/src/anchor_select_randomized_id.cpp` — shared general-feature and absolute-weight AO Khatri–Rao sketch, host/device sketch QRCP.
- `cpp/include/anchor_select_cvt.hpp`, `cpp/src/anchor_select_cvt.cpp` — Kokkos weighted CVT selection.
- `CMakeLists.txt` — C++ extension and dependency configuration.

`mp_isdf` (the C++ executable) is currently a placeholder; **`python/isdf.py` is the usable runner**.
