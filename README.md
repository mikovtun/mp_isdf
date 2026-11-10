# Massively Parallel ISDF

A C++/Python prototype for **interpolative separable density fitting (ISDF)** of molecular AO products. PySCF supplies the molecule and integration grid; a pybind11 module evaluates atomic orbitals with libcint, selects interpolation anchors by pivoted QR, and builds Coulomb metrics. The Python driver fits the interpolation vectors and can compare reconstructed integrals or J/K matrices with PySCF.

The default metric is an **integral-direct, unblocked greedy Cholesky** decomposition of AO-pair electron-repulsion integrals (ERIs). An `unblocked_uniform` variant samples eligible pivots uniformly instead of always selecting the largest residual. Neither variant constructs the full AO-pair ERI matrix or calls LAPACK Cholesky. Independent libcint shell quartets and sufficiently large pair updates run on Kokkos host threads; the ordinary metric projection uses batched BLAS++ GEMMs. An approximate grid-based metric is also available. This is a research prototype, not an installed Python package.

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

The Python executable used to run the driver must match the interpreter used to build `mp_isdf`. The driver imports the extension directly from `build/`; no installation step is needed.

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

The XYZ files in `python/xyz/` can be passed with `--molecule`; omitting it selects H2. Run `python python/isdf.py --help` for all options. Common controls:

| Option | Meaning |
| --- | --- |
| `--axes ao,dx,dy,dz,occ` | Comma-separated features for **anchor selection**; `occ` requires a converged closed-shell RHF calculation. The default is `ao`. |
| `--anchor-rank N`, `--anchor-tol T` | Maximum number of anchors (`0` means no rank cap) and relative QR cutoff. |
| `--anchor-algorithm dense\|online\|randomized` | Dense LAPACK/MAGMA QRCP (default), direct multipass online QRCP, or structured randomized-ID sketch QRCP. |
| `--anchor-batch-size N` | Grid points per batch for `online` or `randomized` (default `128`). |
| `--anchor-sketch-size E`, `--anchor-oversampling P`, `--anchor-seed S` | Randomized sketch rows (`128`), reserved oversampling rows (`16`), and reproducible random seed (`7`). Require `E > P` and `E >= P + anchor-rank` for an explicit rank cap. |
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

For randomized pivot selection, pass `--anchor-algorithm randomized --anchor-sketch-size 128 --anchor-rank 40` (or `anchor_algorithm="randomized"` in `build_isdf`). It uses the shared `anchor_tol` as a **relative sketch Frobenius-residual tolerance**, not a physical pair-space guarantee. With `anchor_rank=0`, selection is capped at `min(sketch_size - oversampling, nonzero_grid_points, axis_width*(axis_width+1)/2)`; `--debug` reports whether the sketch tolerance was met. A fixed seed is independent of batch size; host and device can differ near pivot ties. Host execution keeps AO evaluation, sketch construction, and LAPACK QRCP on host; device/auto in a CUDA build keeps the sketch and MAGMA QRCP on GPU. Only pivot indices and O(sketch_size) residual diagnostics return to host. The algorithm retains an `O(sketch_size * nonzero_grid_points)` sketch plus backend QR workspace, unlike the direct online method. The build configures BLAS/LAPACK and MAGMA for ILP64 integer dimensions and checks the linked LAPACK ABI at configure time. Very large sketches can still require substantial GPU memory and QRCP time; ILP64 removes the 32-bit element-offset ceiling, not those memory or computational costs. The optional `--anchor-audit` compares **actual pair-space orthogonal projection errors** at equal rank; it is neither an ID interpolation error nor an accuracy certificate. The packed sketch QR is the future hook for solving `R11 * X = R_top` in batches, once a full-ID result API replaces the present pivot-only output. The ISDF driver still performs its separate NumPy fit.

For bounded-batch anchor selection, pass `--anchor-algorithm online` to the Python driver (and optionally `--anchor-batch-size N`), or use `anchor_algorithm="online"` and `anchor_batch_size=N` in `build_isdf`. At the extension API, pass `algorithm=mp_isdf.LocalAnchorSelectionAlgorithm.OnlineQRColumnPivoting` and optionally `batch_size=128` to `AnchorSelection.select(...)`. Dense LAPACK/MAGMA QRCP remains the default. Online QRCP uses Kokkos AO evaluation and pair-product generation in the requested `space="host"` or `space="device"` (CUDA build), streaming grid batches instead of building the full pair-product matrix. It keeps one Householder reflector per selected anchor, incrementally updates column norms, selectively refreshes unreliable estimates, verifies each candidate directly, and performs a full direct scan before stopping. Like dense selection, its cutoff is relative to the largest initial column norm; it also stops at a floating-point numerical-rank floor. Zero-weight grid points are skipped, and signed weights are treated as `sqrt(abs(weight))`. With `M=axis_width²`, `N=nonzero_grid_points`, batch size `b`, and selected rank `k`, peak selection working storage is approximately `O(M*k + M*b + N + axis_width*b)` (plus the input grid and AO context; occupied-orbital axes also stage the `nao²` MO coefficients per batch); rank-unlimited selection can still consume substantial reflector memory. This is multipass and can be considerably slower than dense QRCP, particularly on GPU where reflector application currently processes one column per work item. With `--debug`, it reports the selected rank, dimensions and batch size, initial and final maximum residuals, full-scan count, candidate verification/reselection counts, norm-update and refresh counts, and elapsed selection time. It does not change the driver’s subsequent NumPy fitting allocations.

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
                    anchor_algorithm="online", anchor_batch_size=128,
                    derivative_targets=("00xy",))

print(result["anchor_indices"], result["overlap_relative_error"])
print(result["coulomb_metric"].shape)
print(result["target_factors"]["00xy"]["coulomb_metric"].shape)
```

`build_isdf` returns anchor indices, coordinates and weights; interpolation vectors; an overlap reconstruction and relative error; the ordinary anchor-weighted Coulomb metric; and `timings`/`elapsed`. With `pair_space_error=True`, `result["pair_space_errors"]` contains `{"l2": ..., "relative": ...}` for `"scalar"` (ordinary AO products) and each distinct ordered channel used by `derivative_targets` (e.g. `"x0"`, `"0y"` for `"x00y"`). Otherwise it is `None`. This check fits and measures requested derivative channels even when `compute_metric=False`; without a metric no derivative ERIs or target factors are produced. `selection_only=True` cannot be combined with the check. `X1`–`X4` describe the ordinary AO fit. `target_factors[target]` contains the four target-specific anchor matrices and their anchor-weighted cross metric; `derivative_integrals[target]` contains the reconstructed four-index tensor. Pass `compute_metric=False` to omit both kinds of metric, or `selection_only=True, compute_metric=False` to stop after anchor selection. The `mp_isdf` extension also exposes `LibcintContext`, `AnchorSelection`, `cholesky_eri`, `coulomb_metric`, and `derivative_cross_metric` for lower-level use.

The pair-space diagnostic compares the fitted reconstruction `Theta_ab.T @ (W @ P_ab(anchor))` with `P_ab(grid)` across **all ordered AO pairs**. Its absolute error is `sqrt(sum_g abs(grid_weight[g]) * sum_pairs abs(exact - reconstructed)**2)`; the relative error divides by the same norm of the exact pair products. Absolute grid weights define a nonnegative discrete L2 norm even for signed quadrature weights. This diagnoses the real-space fit, not the Coulomb metric or ERI approximation. The `"scalar"` result uses the ordinary fit; a derivative target's `"00"` channel is fitted separately and can have its own error. The choice of `axes` affects anchor selection but does not change which requested channels are checked.

## How derivative fitting works

A target is split into two **ordered AO-pair channels**. For example, `x00y` pairs `(∂x φμ) φν` on the left with `φλ (∂y φσ)` on the right; `00xy` instead pairs `φμ φν` with `(∂x φλ)(∂y φσ)`. Here each derivative acts on the indicated AO only—not on the whole pair by a product rule. Reversing the two AO slots can therefore change a channel.

For each distinct channel `ab`, the driver evaluates its pair products on the grid and at the already-selected anchors. Write these arrays as `P_ab(grid)` and `P_ab(anchor)`, with one column for **each ordered** `(μ,ν)` pair; let `W` be the diagonal matrix of anchor quadrature weights. The channel fit is computed independently as

```text
C_ab     = pinv(W @ P_ab(anchor)).T
Theta_ab = P_ab(grid) @ C_ab.T
```

`Theta_ab` contains the channel's interpolation vectors on the grid. Channels share anchor *locations* but not their fitted coefficients: even a `00` channel requested by a derivative target is fitted in this channel path rather than reusing the ordinary scalar fit. Repeated channel labels are cached within one `build_isdf` call.

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
- `cpp/src/anchor_select_randomized_id.cpp` — factored pair-space sketch and host/device sketch QRCP.
- `CMakeLists.txt` — C++ extension and dependency configuration.

`mp_isdf` (the C++ executable) is currently a placeholder; **`python/isdf.py` is the usable runner**.
