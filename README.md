# Massively Parallel ISDF

**Choose what makes a good interpolation point—and how to find it.** This project explores interpolative separable density fitting (ISDF) for molecular atomic-orbital (AO) products. Its main playground is **anchor selection**: you can search using AO values, occupied orbitals, AO derivatives, or combinations of them, then compare several deterministic and randomized ways to pick grid points.

PySCF provides molecules and integration grids; a C++/Kokkos extension evaluates orbitals and selects anchors; the Python driver fits interpolation vectors and builds Coulomb metrics. You can inspect reconstruction errors or compare integrals and J/K matrices with PySCF. This is a research prototype, not an installed Python package or a production electronic-structure solver.

## What can I select on?

Use `--axes` to choose the **features used to find anchors**. The default is `ao`.

| Feature | CLI name | What it represents |
| --- | --- | --- |
| Atomic orbitals | `ao` | AO values on the grid. |
| Occupied orbitals | `occ` | Occupied molecular-orbital values; requires a converged closed-shell RHF calculation. |
| AO derivatives | `dx`, `dy`, `dz` | Cartesian derivatives of individual AO values. |

Combine features with commas, and optionally set positive relative factors with `:factor`:

```sh
# Find anchors using both AO values and occupied orbitals
OMP_PROC_BIND=false python python/isdf.py \
  --axes ao:1,occ:2 --anchor-algorithm qrcp_online --anchor-rank 8 --no-metric

# Include a derivative channel in a randomized anchor search
OMP_PROC_BIND=false python python/isdf.py \
  --axes ao,dx --anchor-algorithm rand_sketch --anchor-rank 8 --no-metric
```

With multiple axes, each feature block is normalized over eligible grid points before applying its relative factor. These choices affect **where anchors are placed**; the ordinary fit still reconstructs AO products. `--derivative-target` is a separate option for fitting and comparing derivative integrals, not a synonym for `--axes`. Density sketches and the RBRP algorithms currently accept only `--axes ao`; CVT chooses points spatially rather than using axis features. See [anchor selection](docs/anchor-selection.md) for the objectives and restrictions.

## Pick an anchor-selection algorithm

Set `--anchor-algorithm` to choose how grid points are selected:

| Algorithm | Why try it? | Axis support |
| --- | --- | --- |
| `qrcp_dense` **(default)** | Direct pivoted QR; straightforward baseline. | AO, occupied, derivatives, and combinations. |
| `qrcp_online` | Direct QR selection streamed in grid batches instead of storing the full pair matrix. | Same as dense QRCP. |
| `rand_sketch` | Randomized Khatri–Rao pair sketch followed by pivoted QR; configurable with `--sketch`. | All axes with `quadrature`; AO only with density sketches. |
| `cvt` | Spatially distributed, density- or quadrature-weight-guided centroidal Voronoi anchors. | Spatial objective; ignores axis choices. |
| `rbrp` | Blockwise randomized residual-weighted AO-pair selection. | AO only. |
| `rbrp_id` | RBRP selection **and** its interpolation-ID fit, rather than the usual NumPy scalar refit. | AO only. |

`rand_sketch` defaults to `--sketch quadrature`, which uses the chosen axis features and absolute grid-weight magnitudes. Two AO-only alternatives, `--sketch density` and `--sketch density_col_norm`, use density importance; the latter also normalizes individual pair columns. Non-AO axes for these density techniques raise a not-yet-implemented error. These sketches change the *selection objective*, not the physical weights used by the subsequent refit. For detailed method descriptions, rank semantics, and tradeoffs, see [docs/anchor-selection.md](docs/anchor-selection.md).

## Quick start

After building the Python extension (instructions below), run from the repository root:

```sh
# H2 / STO-3G, using default AO features and dense QRCP
OMP_PROC_BIND=false python python/isdf.py --anchor-rank 3

# Select anchors on AO values plus occupied orbitals, and inspect the AO fit
OMP_PROC_BIND=false python python/isdf.py \
  --axes ao,occ --anchor-algorithm qrcp_online --anchor-rank 3 --no-metric

# Try the randomized quadrature sketch (its default sketch technique)
OMP_PROC_BIND=false python python/isdf.py \
  --axes ao,dx --anchor-algorithm rand_sketch --anchor-rank 3 --no-metric

# Compare the resulting J/K matrices to PySCF on a small system
OMP_PROC_BIND=false python python/isdf.py --anchor-rank 3 --compare
```

`python/isdf.py` uses H2 by default. Supply `--molecule python/xyz/ethanol.xyz --basis sto-3g` to try an XYZ file; examples are in `python/xyz/`. `--no-metric` skips the Coulomb metric but keeps the AO fit and overlap check. For every CLI flag, run `python python/isdf.py --help`.

> **Prototype caveat:** An anchor-selection or sketch residual is not a guarantee of J/K accuracy. Use `--pair-space-error` to inspect the physical AO-pair reconstruction and `--compare` on small molecules to check against PySCF. Randomized sketch-selection tests in the current development build have encountered an invalid LAPACK QR pivot; successful end-to-end randomized selection has not yet been verified in that environment.

## Build from source

You need a **C++20 compiler**, **CMake 3.21+**, **Git**, Python with **NumPy and PySCF**, and an installed **ILP64 BLAS/LAPACK** implementation (64-bit Fortran integers). The first configure needs network access to fetch Kokkos, BLAS++, LAPACK++, SLATE, libcint, and pybind11. The project does not automatically download OpenBLAS. OpenMP is enabled by default.

For a CPU-only build, with MPI and CUDA disabled:

```sh
python -m pip install numpy pyscf
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DISDF_ENABLE_CUDA=OFF -DISDF_ENABLE_MPI=OFF \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build --target mp_isdf_python -j 4
```

Use the **same Python interpreter** for building and running. The driver loads `mp_isdf` directly from `build/`; no package install is needed. If you build elsewhere, set `ISDF_BUILD_DIR=/absolute/path/to/build-directory` before running the driver. BLAS/LAPACK discovery must resolve to libraries with the required ILP64 ABI; a successful CMake configuration is not by itself a guarantee that the vendor library and its wrappers agree on integer sizes.

CUDA and MPI are **enabled by default**. To build with CUDA, omit `-DISDF_ENABLE_CUDA=OFF`, install a CUDA toolkit, and set an appropriate `CMAKE_CUDA_ARCHITECTURES`; the build also obtains MAGMA. MPI can be disabled independently. `--space host` forces host AO evaluation and selection, while `--space device` requires a CUDA build. The integral-direct metric evaluates libcint integrals on the CPU.

## Options at a glance

| Option | What it controls |
| --- | --- |
| `--axes ao,dx,occ:2` | Anchor-search features and optional positive relative factors. Default: `ao`. |
| `--anchor-algorithm NAME` | One of the six algorithms above. Default: `qrcp_dense`. |
| `--anchor-rank N`, `--anchor-tol T` | Anchor count or stopping criterion. For `rand_sketch`, an explicit positive rank requests exactly N eligible/numerically possible pivots; rank and tolerance cannot both be set. Other methods have different rank/cap behavior; see [the guide](docs/anchor-selection.md). |
| `--sketch quadrature\|density\|density_col_norm` | Sketch input for `rand_sketch`; default: `quadrature`. Density techniques are AO-only for now. |
| `--anchor-sketch-size E`, `--anchor-oversampling P`, `--anchor-seed S` | Random-sketch dimensions and reproducible seed. Explicit rank uses rank + oversampling rows; without it, sketch size controls the rows. |
| `--anchor-batch-size N`, `--fit-batch-size N` | Grid points processed per selection or interpolation-fit batch. |
| `--density-weight electron_density\|quadrature_density` | Importance for density sketches: electron density alone (default) or `abs(grid_weight) * density`. |
| `--ao-screening-threshold T` | Exclude low-AO-norm points from density-sketch selection; does not remove them from the physical fit. |
| `--cvt-weight electron_density\|quadrature_weights` | Importance for CVT: density (default) or absolute quadrature weight. CVT needs a positive anchor rank. |
| `--rbrp-block-size N` | RBRP candidate block size; further RBRP tuning flags are in `--help`. |
| `--level N`, `--space auto\|host\|device` | PySCF grid level and AO/selection execution space. |
| `--metric unblocked_greedy\|unblocked_uniform\|grid` | Integral-direct greedy Cholesky (default), uniformly sampled Cholesky pivots, or an approximate grid metric. |
| `--metric-tol T`, `--metric-rank N`, `--metric-seed S` | Metric stopping threshold, rank cap, and seed for the uniform variant. |
| `--derivative-target 00xy` | Fit a four-slot AO derivative integral target; repeat for multiple targets. This is separate from anchor-search `--axes`. |
| `--no-metric`, `--pair-space-error`, `--compare`, `--debug` | Skip the metric; inspect pair-space reconstruction; compare with PySCF; or print diagnostics and timings. |

`occ` triggers an RHF calculation; density-based CVT/sketch weights use a separate converged RKS B3LYP calculation by default. Signed quadrature weights contribute by magnitude to selection; the physical fit and metric retain their original weights. For explanations of sketch stopping tolerances, AO screening, audit diagnostics, and density-weighted comparisons, see [docs/anchor-selection.md](docs/anchor-selection.md), [the sketch guide](docs/anchor-selection-sketches.pdf), and [the density-sketch proposal](density_weighted_normalized_sketch.md).

## Python API

From the repository root (after building):

```python
import sys
sys.path.insert(0, "python")

from pyscf import gto
from isdf import build_isdf

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
result = build_isdf(mol, axes=("ao", "dx"), anchor_algorithm="qrcp_online",
                    anchor_rank=3, compute_metric=False)
print(result["anchor_indices"], result["overlap_relative_error"])
```

`build_isdf` also accepts `axis_weights=(1, 2)` for mixed axes, `sketch="density_col_norm"` with `anchor_algorithm="rand_sketch"` and AO-only axes, or `selection_only=True, compute_metric=False` to stop after anchor selection. Its result includes selected indices, coordinates, interpolation vectors, fit diagnostics, and (when requested) Coulomb metrics. See its docstring in [python/isdf.py](python/isdf.py) for the full return structure and [python/compare_density_sketches.py](python/compare_density_sketches.py) for shared-grid, repeated-seed density-sketch comparisons.

## Tests and project layout

```sh
OMP_NUM_THREADS=4 OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s python/test -q
```

- `python/isdf.py` — CLI, Python API, physical refit, and comparisons.
- `python/test/` — extension and driver tests.
- `cpp/src/anchor_select_*.cpp` — dense, online, randomized-sketch, CVT, and RBRP selection backends.
- `cpp/src/cholesky_eri_*.cpp`, `cpp/src/coulomb_metric.cpp` — integral-direct metric construction.
- `docs/anchor-selection.md` — method-by-method guide; `docs/anchor-selection-sketches.tex` — sketch derivations.
- `CMakeLists.txt` — build and dependency setup.

The `mp_isdf` C++ executable is a placeholder; **`python/isdf.py` is the runner**.
