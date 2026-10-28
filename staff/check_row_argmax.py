#!/usr/bin/env python3
"""Course-staff check: does the fitted HMM use its dynamics?

Loads the authors' saved parameters (data/hmm_model_march11.pkl) with an unpickler
that admits only NumPy types, prints the transition matrix, the expected dwell time
in each state and the state means next to where a raw zero lands after the authors'
log transform, then simulates sequences from the fitted HMM and measures how often
the forward filter and Viterbi agree with a row-by-row classification that ignores
the transition matrix altogether. For contrast it repeats the simulation with the
same emissions and a sticky transition matrix.

Needs only NumPy:  python staff/check_row_argmax.py [--params PATH] [--seqs N] [--length T]
"""
import argparse
import importlib
import math
import pathlib
import pickle

import numpy as np

ALLOWED = {
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "scalar"),
}


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def load_params(path):
    with open(path, "rb") as f:
        return NumpyOnlyUnpickler(f).load()


def log_emission(X, mu, sigma):
    """log p(x_t | z_t = k) for every row and state, (T, K); the authors' log_normal_diag."""
    x = X[:, None, :]
    m = mu[None]
    s = sigma[None]
    return (-0.5 * (((x - m) / s) ** 2 + 2 * np.log(s) + np.log(2 * np.pi))).sum(-1)


def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True)), axis=axis)


def forward_filter(logB, logpi, logA):
    """argmax of the filtered state distribution P(z_t | x_1..t) at every t."""
    T, K = logB.shape
    la = np.empty((T, K))
    la[0] = logpi + logB[0]
    la[0] -= logsumexp(la[0], 0)
    for t in range(1, T):
        la[t] = logB[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)
        la[t] -= logsumexp(la[t], 0)
    return la.argmax(1)


def viterbi(logB, logpi, logA):
    T, K = logB.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = logpi + logB[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + logA  # (from, to)
        psi[t] = scores.argmax(0)
        delta[t] = scores.max(0) + logB[t]
    path = np.empty(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def simulate(rng, pi, A, mu, sigma, T):
    K = len(pi)
    z = np.empty(T, dtype=int)
    z[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        z[t] = rng.choice(K, p=A[z[t - 1]])
    return z, rng.normal(mu[z], sigma[z])


def agreement(pi, A, mu, sigma, n_seq, T, seed):
    rng = np.random.default_rng(seed)
    logpi, logA = np.log(pi + 1e-12), np.log(A + 1e-12)
    counts = dict(filter_vs_row=0, viterbi_vs_row=0, row_correct=0, filter_correct=0, viterbi_correct=0)
    for _ in range(n_seq):
        z, X = simulate(rng, pi, A, mu, sigma, T)
        logB = log_emission(X, mu, sigma)
        row = logB.argmax(1)
        f = forward_filter(logB, logpi, logA)
        v = viterbi(logB, logpi, logA)
        counts["filter_vs_row"] += (f == row).sum()
        counts["viterbi_vs_row"] += (v == row).sum()
        counts["row_correct"] += (row == z).sum()
        counts["filter_correct"] += (f == z).sum()
        counts["viterbi_correct"] += (v == z).sum()
    return {k: c / (n_seq * T) for k, c in counts.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = pathlib.Path(__file__).resolve().parent.parent
    ap.add_argument("--params", default=str(repo / "data" / "hmm_model_march11.pkl"))
    ap.add_argument("--seqs", type=int, default=300, help="simulated sequences (default 300)")
    ap.add_argument("--length", type=int, default=200, help="windows per sequence (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load_params(args.params)
    pi, A, mu, sigma = (np.asarray(d[k], dtype=float) for k in ("pi", "A", "mu", "sigma"))
    # the saved values are averages of guide draws in float32, so they do not sum to one exactly
    pi = pi / pi.sum()
    A = A / A.sum(axis=1, keepdims=True)
    feats = [f.replace("_logged_normed", "") for f in d["features"]]
    K = len(pi)
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    print("features:", feats)
    print("\ninitial distribution pi:", pi)
    print("transition matrix A (row = from, column = to):")
    print(A)
    print("expected dwell time per state, 1 / (1 - A_kk), in windows:", 1 / (1 - np.diag(A)))
    w, v = np.linalg.eig(A.T)
    st = np.real(v[:, np.argmin(abs(w - 1))])
    print("stationary distribution of A:", st / st.sum())

    zero = np.array(
        [(math.log(d["norm_params"][f]["eps"]) - float(d["norm_params"][f]["mean"])) / float(d["norm_params"][f]["std"]) for f in feats]
    )
    print("\nstate means, then where a raw zero lands after log(x + eps) and standardization, then state scales:")
    print("            " + "".join(f"{f:>22s}" for f in feats))
    for k in range(K):
        print(f"  mean   {k}  " + "".join(f"{m:22.3f}" for m in mu[k]))
    print("  raw zero  " + "".join(f"{z:22.3f}" for z in zero))
    for k in range(K):
        print(f"  scale  {k}  " + "".join(f"{s:22.3f}" for s in sigma[k]))

    print("\nlog-likelihood of each state's mean vector under each state (row = observation, column = state):")
    print(np.array([[log_emission(mu[i][None], mu, sigma)[0, j] for j in range(K)] for i in range(K)]))
    print("log transition probabilities, log A:")
    print(np.log(A))

    res = agreement(pi, A, mu, sigma, args.seqs, args.length, args.seed)
    print(f"\nsimulated {args.seqs} sequences of {args.length} windows from the fitted HMM:")
    print(f"  forward-filter state equals the row-wise emission argmax on {res['filter_vs_row']:.1%} of rows")
    print(f"  Viterbi state equals the row-wise emission argmax on {res['viterbi_vs_row']:.1%} of rows")
    print(
        "  accuracy against the simulated true states: "
        f"row-wise {res['row_correct']:.1%}, forward filter {res['filter_correct']:.1%}, Viterbi {res['viterbi_correct']:.1%}"
    )

    sticky = 0.95 * np.eye(K) + (0.05 / (K - 1)) * (1 - np.eye(K))
    res2 = agreement(pi, sticky, mu, sigma, args.seqs, args.length, args.seed)
    print("\nfor contrast, the same emissions with a sticky transition matrix (0.95 on the diagonal):")
    print(f"  forward-filter state equals the row-wise emission argmax on {res2['filter_vs_row']:.1%} of rows")
    print(f"  Viterbi state equals the row-wise emission argmax on {res2['viterbi_vs_row']:.1%} of rows")
    print(
        "  accuracy against the simulated true states: "
        f"row-wise {res2['row_correct']:.1%}, forward filter {res2['filter_correct']:.1%}, Viterbi {res2['viterbi_correct']:.1%}"
    )


if __name__ == "__main__":
    main()
