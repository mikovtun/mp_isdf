# Anchor selection algorithms

This guide summarizes the anchor selectors available in `python/isdf.py`.
It focuses on their selection behavior and choosing between them; build and
low-level implementation details are intentionally omitted.

Run a selector with:

```sh
python python/isdf.py --anchor-algorithm ALGORITHM --anchor-rank N
```

In Python, use the same name with `build_isdf`:

```python
result = build_isdf(mol, anchor_algorithm="online", anchor_rank=40)
```

The selected anchors are locations on the molecular integration grid. All
methods ultimately use the ordinary physical-grid fit except `rbrp-id`, which
uses its own interpolation-ID fit. Unless noted otherwise, `--anchor-rank N`
requests/caps the number of anchors and `--axes ao` is the default selection
feature. Rank and `--anchor-tol` cannot be supplied together.

## Quick choice

| If you want… | Try… |
|---|---|
| A general-purpose deterministic baseline | `dense` |
| Less selection memory on a large grid, with a possible runtime tradeoff | `online` |
| Randomized selection based on selected orbital features | `rand_feature_kr` |
| Randomized AO-pair selection emphasizing quadrature-weighted pair magnitude | `rand_ao_kr_quadrature` |
| AO-pair selection guided by density importance | `rand_ao_kr_density_normalized` |
| A spatially distributed, density- or weight-guided set | `cvt` |
| Blockwise randomized AO-only selection | `rbrp` |
| Blockwise randomized selection with its associated interpolation ID | `rbrp-id` |

Anchor quality is not itself a guarantee of J/K or energy accuracy. Compare
methods at equal anchor counts using `--compare` (or the companion comparison
scripts), and include the physical pair residual where useful with
`--pair-space-error`.

## Algorithms

### `dense` — pivoted QR (default)

Selects grid points using direct, deterministic column-pivoted QR on the
selected pair-product matrix. Use this as the straightforward reference method
when grid and pair dimensions are manageable. It can be used with the available
`--axes` choices.

```sh
python python/isdf.py --anchor-algorithm dense --anchor-rank 40
```

An explicit rank is a maximum rank for dense selection, not a promise that all
requested pivots will be returned if the data have lower numerical rank. If no
rank is supplied, the default QR tolerance is used.

### `online` — bounded-batch pivoted QR

Selects anchors with the same direct QR objective as `dense`, while processing
the grid in batches. It is an option when the full selection matrix is too large
to hold at once. This approach can take longer because it revisits grid data;
reduced memory does not imply reduced runtime. It supports the available
`--axes` choices.

```sh
python python/isdf.py --anchor-algorithm online --anchor-rank 40 \
  --anchor-batch-size 128
```

Here the rank is a cap. With no rank, `--anchor-tol` controls stopping.

### `rand_feature_kr` — randomized feature-pair sketch

Uses a randomized sketch of the outer products of the requested orbital
features. It is suitable when feature choices such as AO values, gradients, or
occupied orbitals should guide selection. Signed quadrature weights contribute
through their absolute magnitudes.

```sh
python python/isdf.py --anchor-algorithm rand_feature_kr --axes ao,occ \
  --anchor-rank 40 --anchor-oversampling 16 --anchor-seed 7
```

With an explicit rank, exactly that many pivots are requested, subject to
candidate and numerical-rank limits. Without a rank, `--anchor-tol` controls the
sketch residual. The sketch size and oversampling are controlled by
`--anchor-sketch-size`, `--anchor-oversampling`, and `--anchor-seed`.

### `rand_ao_kr_quadrature` — quadrature-weighted AO-pair sketch

A randomized AO-only selector that emphasizes pair columns according to the
absolute quadrature weights. Use it when the selection objective should track
the grid-weighted AO-pair magnitude. It requires `--axes ao` and supports
signed grids by treating negative and positive quadrature weights by magnitude.

```sh
python python/isdf.py --anchor-algorithm rand_ao_kr_quadrature \
  --anchor-rank 40 --anchor-oversampling 16
```

Rank, tolerance, seed, and sketch-size controls follow `rand_feature_kr`.

### `rand_ao_kr_density` — raw density-weighted AO-pair sketch

An AO-only randomized selector that uses electron-density importance while
retaining the amplitudes of the AO-pair columns. It provides a useful control
for the normalized density sketch. By default its point importance is the
spin-summed density from a separate converged RKS B3LYP calculation; density
alone is used, with no extra quadrature factor. Set
`--density-weight quadrature_density` to use the product of density and
absolute quadrature weight.

```sh
python python/isdf.py --anchor-algorithm rand_ao_kr_density \
  --anchor-rank 40 --anchor-oversampling 16
```

Requires `--axes ao`. Rank and sketch controls follow the other randomized
selectors. In the Python API, `point_importance=` can supply a complete,
nonnegative importance array aligned with the original grid.

### `rand_ao_kr_density_normalized` — normalized density-weighted AO-pair sketch

The proposed density-guided selector. It weights each AO-pair direction by the
square root of its point importance, but normalizes away the pair column's
initial amplitude. This makes the initial expected sketch score depend on the
specified point importance, while pivoting still responds to new pair-space
directions. It is a function-space novelty rule, not a spatial clustering
method and not a guarantee of improved J/K accuracy.

```sh
python python/isdf.py \
  --anchor-algorithm rand_ao_kr_density_normalized --axes ao \
  --anchor-rank 40 --anchor-oversampling 16 \
  --density-weight electron_density \
  --ao-screening-threshold 1e-24
```

The default importance is density alone, matching density-weighted CVT's
default weighting convention. `--density-weight quadrature_density` instead
uses `abs(grid_weight) * density`. The threshold applies to the **squared AO
norm** and screens only anchor candidates; nonzero-weight grid points remain
available to the physical refit. In `build_isdf`, a caller may pass a complete
`point_importance` array instead of generating density internally. If comparing
against CVT, use the same importance array for both.

For both density sketch modes, an explicit rank requests exactly that many
pivots; a request exceeding the eligible or numerical rank fails. Without an
explicit rank, `--anchor-tol` is a relative sketch Frobenius residual tolerance.
Neither it nor the density-weighted normalized reconstruction residual is a
physical J/K error bound.

### `cvt` — centroidal Voronoi anchors

Groups grid sites by **spatial proximity** and places anchors near the
importance-weighted cluster centers. This is the choice when spatial coverage
is desired. Its objective differs from pair-space QR: points close in real
space may still have different pair-space directions, and CVT does not assess
those directions directly.

```sh
python python/isdf.py --anchor-algorithm cvt --anchor-rank 40 \
  --cvt-weight electron_density --anchor-seed 7
```

CVT requires a positive `--anchor-rank`; `--anchor-tol` is not used. The default
importance is the separate converged RKS B3LYP spin-summed density. Choose
`--cvt-weight quadrature_weights` to use absolute quadrature weights **instead**
of density—not their product. Control clustering with `--cvt-max-iter` and
`--cvt-tol`. A user-supplied `point_importance` is accepted by the Python
`build_isdf` API and replaces the generated CVT importance array.

### `rbrp` — blockwise randomized AO selection

An AO-only blockwise randomized selector for larger problems. It samples and
evaluates groups of candidates iteratively, using pair-space information to
grow a selected subspace. Use it when you want a randomized AO-only alternative
with block-size and sampling controls.

```sh
python python/isdf.py --anchor-algorithm rbrp --anchor-rank 40 \
  --rbrp-block-size 16 --anchor-seed 7
```

Requires `--axes ao`. Here `--anchor-rank` is a **cap**, not a promise of an
exact number of pivots; without a rank, `--anchor-tol` controls the stopping
criterion. `rbrp` uses the usual physical interpolation fit after selection.

### `rbrp-id` — blockwise randomized selection and interpolation ID

Uses the same AO-only blockwise randomized selection family as `rbrp`, and also
constructs interpolation coefficients from its selected-subspace ID. Choose it
when you want that ID-based fit instead of the usual NumPy physical refit.

```sh
python python/isdf.py --anchor-algorithm rbrp-id --anchor-rank 40 \
  --rbrp-block-size 16 --anchor-seed 7
```

Requires `--axes ao`. Its rank/tolerance behavior is the same as `rbrp` (rank is
a cap). The fit can be sensitive to the conditioning and truncation of the
selected subspace; inspect the returned `rbrp_diagnostics` when evaluating it.

## Shared randomized controls

- `--anchor-seed`: reproducible probe or initialization seed (default `7`).
- `--anchor-oversampling`: additional sketch rows beyond the requested rank
  (default `16`).
- `--anchor-sketch-size`: sketch rows for tolerance-driven randomized selection
  (default `128`; explicit rank instead uses `rank + oversampling`).
- `--anchor-batch-size`: grid points per selection batch (default `128`).
- `--anchor-audit`: small-problem randomized-selection audit; not available for
  every algorithm and not an accuracy certificate.
- `--space host|device|auto`: selection execution space; `device` requires a
  CUDA-enabled build.

See `python python/isdf.py --help` for the full option list. For the density
sketch derivation and controlled experimental setup, see
[`density_weighted_normalized_sketch.md`](../density_weighted_normalized_sketch.md).
