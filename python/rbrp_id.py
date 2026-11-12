"""Host AO-pair RBRP (Algorithm 4.1) and ID (Algorithm 5.1).

X[g, mu<=nu] = sqrt(abs(w[g])) * AO[g,mu] * AO[g,nu], with sqrt(2)
on off-diagonal pairs. Rows are streamed, never stored as a grid-wide pair
matrix. The residual tolerance is a *norm* tolerance (paper tau = rtol**2).
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr, svd


class AOPairs:
    def __init__(self, ao, weights, batch_size=512):
        self.ao = np.asarray(ao, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        if (self.ao.ndim != 2 or self.ao.shape[1] == 0 or
                self.weights.shape != (len(self.ao),) or len(self.ao) == 0 or
                not np.all(np.isfinite(self.ao)) or
                not np.all(np.isfinite(self.weights)) or
                np.any(self.weights == 0) or batch_size < 1):
            raise ValueError("expected finite real AO rows and nonzero finite weights")
        self.batch_size = batch_size
        self.mu, self.nu = np.triu_indices(self.ao.shape[1])
        self.scale = np.where(self.mu == self.nu, 1., np.sqrt(2.))
        self.shape = (len(self.ao), len(self.mu))
        self.sqrtw = np.sqrt(np.abs(self.weights))

    def rows(self, idx):
        a = self.ao[idx]
        return a[:, self.mu] * a[:, self.nu] * self.scale * self.sqrtw[idx, None]

    def batches(self):
        for start in range(0, self.shape[0], self.batch_size):
            yield slice(start, min(start + self.batch_size, self.shape[0]))

    def norm2(self):
        return np.abs(self.weights) * np.sum(self.ao * self.ao, axis=1)**2

    def multiply(self, q):
        out = np.empty((self.shape[0], q.shape[1]))
        for s in self.batches():
            out[s] = self.rows(s) @ q
        return out

    def residual_norm2(self, q, l):
        d = np.empty(self.shape[0])
        for s in self.batches():
            e = self.rows(s) - l[s] @ q.T
            d[s] = np.einsum("ij,ij->i", e, e)
        return d


@dataclass
class Result:
    indices: np.ndarray
    L: np.ndarray
    Q: np.ndarray
    residual_norm2: np.ndarray
    relative_residual: float
    status: str
    history: list


def select(op, *, block_size=16, rtol=1e-10, max_rank=0, seed=7,
           tau_block=None, rank_rtol=1e-12, refresh_every=8):
    """Sample by residual squared norm; locally pivot/filter each block."""
    n, p = op.shape
    if (not isinstance(block_size, (int, np.integer)) or block_size < 1 or
            not np.isfinite(rtol) or rtol < 0 or
            not np.isfinite(rank_rtol) or rank_rtol <= 0 or
            not isinstance(refresh_every, (int, np.integer)) or refresh_every < 0 or
            not isinstance(max_rank, (int, np.integer)) or max_rank < 0):
        raise ValueError("invalid RBRP block size, rank or tolerances")
    tb = (1 / block_size if block_size > 1 else 0.) if tau_block is None else tau_block
    if not np.isfinite(tb) or not 0 <= tb < 1:
        raise ValueError("RBRP tau_block must be in [0,1)")
    cap = min(n, p, max_rank if max_rank else min(n, p))
    rng = np.random.default_rng(seed)
    q = np.empty((p, 0))
    l = np.empty((n, 0))
    selected = np.zeros(n, dtype=bool)
    anchors = []
    d = op.norm2()
    total = float(d.sum())
    target = rtol**2 * total
    history = []
    status = "rank_cap"
    iteration = 0
    while True:
        if q.shape[1] and (d.sum() <= target or
                           (refresh_every and iteration % refresh_every == 0)):
            d = op.residual_norm2(q, l)
            d[selected] = 0.
        if d.sum() <= target:
            status = "tolerance"
            break
        if len(anchors) >= cap:
            break
        active = np.flatnonzero((d > 0) & ~selected)
        if not len(active):
            status = "no_candidates"
            break
        b = min(block_size, cap - len(anchors), len(active))
        candidates = rng.choice(active, size=b, replace=False, p=d[active] / d[active].sum())
        v = op.rows(candidates).T
        for _ in range(2):
            v -= q @ (q.T @ v)
        qb, r, piv = qr(v, mode="economic", pivoting=True)
        energy = np.sum(r * r)
        if energy == 0:
            status = "numerical_rank"
            break
        tails = np.cumsum(np.sum(r * r, axis=1)[::-1])[::-1]
        keep = int(np.flatnonzero(tails >= tb * energy)[-1]) + 1
        diag = np.abs(np.diag(r))
        bad = np.flatnonzero(diag <= rank_rtol * np.linalg.norm(op.rows(candidates), "fro"))
        if len(bad):
            keep = min(keep, int(bad[0]))
        if keep == 0:
            status = "numerical_rank"
            break
        chosen = candidates[piv[:keep]]
        qb = qb[:, :keep]
        for _ in range(2):
            qb -= q @ (q.T @ qb)
        qb, _ = qr(qb, mode="economic")
        lb = op.multiply(qb)
        lb[selected] = 0.
        q = np.column_stack((q, qb))
        l = np.column_stack((l, lb))
        anchors.extend(chosen.tolist())
        selected[chosen] = True
        d = np.maximum(d - np.sum(lb * lb, axis=1), 0.)
        d[selected] = 0.
        history.append((len(anchors), b, keep, np.sqrt(d.sum() / total)))
        iteration += 1
    d = op.residual_norm2(q, l)
    d[selected] = 0.
    relative = float(np.sqrt(d.sum() / total)) if total else 0.
    if relative <= rtol:
        status = "tolerance"
    elif status == "tolerance":
        status = "roundoff_limited"
    return Result(np.asarray(anchors, dtype=np.intp), l, q, d, relative, status, history)


def interpolate(op, result, svd_rtol=1e-12):
    """Return (driver theta, ordered-pair metric transform, SVD diagnostics).

    theta.T @ (diag(signed anchor weights) @ B[J]) reconstructs B.
    transform = pinv(diag(signed anchor weights) @ B[J]).T, using L1 and Q.
    """
    if not np.isfinite(svd_rtol) or svd_rtol < 0:
        raise ValueError("svd_rtol must be finite and nonnegative")
    j, l, q = result.indices, result.L, result.Q
    if len(j) == 0:
        raise ValueError("cannot interpolate without anchors")
    u, s, vh = svd(l[j], full_matrices=False)
    take = s > svd_rtol * s[0]
    if not np.any(take):
        raise ValueError("RBRP anchor block has zero numerical rank")
    inv = (vh[take].T / s[take]) @ u[:, take].T
    h = l @ inv
    h[j] = np.eye(len(j))
    scale = op.sqrtw[j] / op.weights[j]
    theta = (h * scale[None, :] / op.sqrtw[:, None]).T
    # Convert the packed isometric pair basis back to ordered AO pairs.
    packed = (scale[:, None] * (inv.T @ q.T))
    nao = op.ao.shape[1]
    transform = np.zeros((len(j), nao, nao), dtype=np.float64)
    transform[:, op.mu, op.nu] = packed / op.scale
    transform[:, op.nu, op.mu] = packed / op.scale
    return theta, transform.reshape(len(j), nao * nao), {
        "retained_rank": int(take.sum()),
        "condition": float(s[0] / s[-1]) if s[-1] else float("inf"),
        "relative_residual": result.relative_residual,
        "status": result.status,
    }
