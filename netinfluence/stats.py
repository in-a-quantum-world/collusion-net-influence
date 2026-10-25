"""
Pairwise net influence (Bonjour, Aggarwal and Bhargava, UAI 2022) for
simultaneous, fully observable repeated games, plus baselines and a
permutation-based bias correction.

Notation (paper eq. numbers in brackets):
  gamma(i; j)  = sum_s p(s) sum_{a_i,a_j} p(a_i,a_j|s) log p(a_i,a_j|s) / (p(a_i|s) p(a_j|s))   [3, 6-7]
               = I(A_i ; A_j | S), conditional mutual information (symmetric in the
                 simultaneous-move case).
  Gamma(i; j)  = gamma(i; j) - max_{k not in {i,j}} gamma(k; j)                                 [4]
  colluding    iff Gamma(i; j) >= alpha and Gamma(j; i) >= alpha                                [5]

Inputs are integer action arrays of shape (T, N): T rounds, N agents.
States are integer labels of shape (T,), computed from history by a
state function (see STATE_FUNCS). Probabilities are plug-in counts
[12-13]. Logs are natural, matching alpha = 0.05 in the paper.
"""
from __future__ import annotations

import itertools
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# State functions: map the action history to an observable state per round.
# ---------------------------------------------------------------------------

def state_none(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """Memoryless: a single state. gamma reduces to plain mutual information."""
    return np.zeros(actions.shape[0], dtype=np.int64)


def state_prev_profile(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """State = the full joint action profile of the previous round (all agents).
    Round 0 gets its own 'no history' state."""
    T, N = actions.shape
    s = np.zeros(T, dtype=np.int64)
    if T > 1:
        weights = n_actions ** np.arange(N)
        s[1:] = 1 + actions[:-1] @ weights
    return s


def state_prev_majority(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """State = the most common action in the previous round (ties -> lowest)."""
    T, N = actions.shape
    s = np.zeros(T, dtype=np.int64)
    for t in range(1, T):
        counts = np.bincount(actions[t - 1], minlength=n_actions)
        s[t] = 1 + int(np.argmax(counts))
    return s


STATE_FUNCS = {
    "none": state_none,
    "prev_profile": state_prev_profile,
    "prev_majority": state_prev_majority,
}


# ---------------------------------------------------------------------------
# Core statistic
# ---------------------------------------------------------------------------

def conditional_mi(a: np.ndarray, b: np.ndarray, s: np.ndarray, n_actions: int) -> float:
    """Plug-in I(A; B | S) in nats. a, b, s are integer arrays of length T."""
    T = a.shape[0]
    if T == 0:
        return 0.0
    n_s = int(s.max()) + 1
    # joint counts N(a, b, s)
    idx = (s * n_actions + a) * n_actions + b
    joint = np.bincount(idx, minlength=n_s * n_actions * n_actions).reshape(n_s, n_actions, n_actions).astype(float)
    n_state = joint.sum(axis=(1, 2))               # N(s)
    p_s = n_state / T
    mi = 0.0
    for si in range(n_s):
        if n_state[si] == 0:
            continue
        pab = joint[si] / n_state[si]              # p(a,b|s)
        pa = pab.sum(axis=1, keepdims=True)        # p(a|s)
        pb = pab.sum(axis=0, keepdims=True)        # p(b|s)
        nz = pab > 0
        mi += p_s[si] * np.sum(pab[nz] * np.log(pab[nz] / (pa @ pb)[nz] + EPS))
    return float(max(mi, 0.0))


def gamma_matrix(actions: np.ndarray, s: np.ndarray, n_actions: int) -> np.ndarray:
    """gamma[i, j] = I(A_i ; A_j | S). Symmetric, zero diagonal."""
    N = actions.shape[1]
    g = np.zeros((N, N))
    for i, j in itertools.combinations(range(N), 2):
        g[i, j] = g[j, i] = conditional_mi(actions[:, i], actions[:, j], s, n_actions)
    return g


def net_influence(gamma: np.ndarray) -> np.ndarray:
    """Gamma[i, j] = gamma[i, j] - max_{k != i, j} gamma[k, j]   (eq. 4). Not symmetric."""
    N = gamma.shape[0]
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            others = [k for k in range(N) if k not in (i, j)]
            G[i, j] = gamma[i, j] - (max(gamma[k, j] for k in others) if others else 0.0)
    return G


def pair_scores(G: np.ndarray) -> dict[tuple[int, int], float]:
    """Score each unordered pair by min(Gamma(i;j), Gamma(j;i)); the criterion in
    eq. 5 is 'this score >= alpha'."""
    N = G.shape[0]
    return {(i, j): float(min(G[i, j], G[j, i])) for i, j in itertools.combinations(range(N), 2)}


def detect_pair(G: np.ndarray, alpha: float = 0.05):
    """Bonjour criterion: report the pair iff exactly one pair has both
    directions >= alpha. Returns (pair or None, scores)."""
    scores = pair_scores(G)
    flagged = [p for p, v in scores.items() if v >= alpha]
    return (flagged[0] if len(flagged) == 1 else None), scores


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def same_action_rate_matrix(actions: np.ndarray) -> np.ndarray:
    """Naive 'agreement' statistic: fraction of rounds the two agents chose the
    same action. Symmetric."""
    N = actions.shape[1]
    m = np.zeros((N, N))
    for i, j in itertools.combinations(range(N), 2):
        m[i, j] = m[j, i] = float(np.mean(actions[:, i] == actions[:, j]))
    return m


def cramers_v_matrix(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """Cramer's V between action sequences (unconditional association)."""
    N = actions.shape[1]
    T = actions.shape[0]
    m = np.zeros((N, N))
    for i, j in itertools.combinations(range(N), 2):
        tab = np.zeros((n_actions, n_actions))
        np.add.at(tab, (actions[:, i], actions[:, j]), 1)
        expected = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / T
        nz = expected > 0
        chi2 = float(np.sum((tab[nz] - expected[nz]) ** 2 / expected[nz]))
        m[i, j] = m[j, i] = np.sqrt(chi2 / (T * (n_actions - 1)))
    return m


def symmetric_top_pair(mat: np.ndarray):
    N = mat.shape[0]
    pairs = list(itertools.combinations(range(N), 2))
    return max(pairs, key=lambda p: mat[p])


# ---------------------------------------------------------------------------
# Permutation null / bias correction
# ---------------------------------------------------------------------------

def permutation_corrected_gamma(actions: np.ndarray, s: np.ndarray, n_actions: int,
                                n_perm: int = 200, rng: np.random.Generator | None = None):
    """For every pair, estimate the null distribution of gamma under
    'A_j independent of A_i given S' by shuffling A_j within each state
    stratum (marginals p(a_j|s) preserved). Returns
      gamma_corrected = gamma - mean(null), clipped at 0,
      pvals[i, j]     = P(null >= observed).
    Plug-in MI is positively biased by about (K-1)^2 / (2 T) nats per state
    stratum, which at T ~ 25 rounds is of the same order as alpha."""
    rng = rng or np.random.default_rng(0)
    T, N = actions.shape
    g = gamma_matrix(actions, s, n_actions)
    null_mean = np.zeros((N, N))
    pvals = np.ones((N, N))
    strata = [np.where(s == v)[0] for v in np.unique(s)]
    for i, j in itertools.combinations(range(N), 2):
        null = np.empty(n_perm)
        aj = actions[:, j].copy()
        for p in range(n_perm):
            shuffled = aj.copy()
            for idx in strata:
                shuffled[idx] = aj[rng.permutation(idx)]
            null[p] = conditional_mi(actions[:, i], shuffled, s, n_actions)
        null_mean[i, j] = null_mean[j, i] = null.mean()
        pv = (np.sum(null >= g[i, j]) + 1) / (n_perm + 1)
        pvals[i, j] = pvals[j, i] = pv
    g_corr = np.clip(g - null_mean, 0.0, None)
    return g, g_corr, pvals
