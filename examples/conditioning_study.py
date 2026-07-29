"""Where does the from-scratch solver diverge from LAPACK?

Run:  python examples/conditioning_study.py

Prints a per-decomposition error table for matrices of escalating
difficulty:
  - random well-conditioned
  - Hilbert (classic, intrinsic ill-conditioning, no growth)
  - Vandermonde (geometric ill-conditioning)
  - random with controlled condition number (10^k for k = 2..14)
  - non-symmetric with a complex eigenvalue pair (where single-shift QR
    must force-deflate)

The point is not to beat LAPACK; it is to *characterize* the failure
modes that the four algorithms exhibit when stripped of blocking,
iterative refinement, and double-shift tricks.
"""
from __future__ import annotations
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import scipy.linalg as sla

import linalg_bedrock as lb
from linalg_bedrock._core import matmul, frobenius


# ---- test-matrix generators -------------------------------------------------

def hilbert(n):
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def vandermonde(n):
    # nodes chosen in (0, 1); Vandermonde is famously ill-conditioned
    x = [(i + 1) / (n + 1) for i in range(n)]
    return [[xi ** k for k in range(n)] for xi in x]


def random_with_cond(n, cond, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    U, _, Vt = np.linalg.svd(A)
    s = np.geomspace(1.0, 1.0 / cond, n)
    return (U * s) @ Vt


def complex_spectrum_4x4():
    theta = 0.7
    R2 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
    A = np.zeros((4, 4))
    A[0:2, 0:2] = R2
    A[2, 2] = 5.0; A[3, 3] = -3.0
    A[2, 3] = 0.5; A[0, 2] = 0.01; A[1, 2] = 0.01
    return A


# ---- comparison routines ----------------------------------------------------

def _rel(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b))
                 / max(np.linalg.norm(np.array(b)), 1e-300))


def compare_lu(A_py):
    A = np.array(A_py)
    LU = lb.lu_decompose([row[:] for row in A_py])
    rec_err = _rel(LU.reconstruct(), A)
    P, L, U = sla.lu(A)
    lap_err = _rel(P @ L @ U, A)
    return rec_err, lap_err, LU.growth, LU.singular


def compare_qr(A_py):
    A = np.array(A_py)
    QR = lb.qr_decompose([row[:] for row in A_py])
    rec = matmul(QR.Q, QR.R)
    rec_err = _rel(rec, A_py)
    Q_lap, R_lap = np.linalg.qr(A)
    lap_err = _rel(Q_lap @ R_lap, A)
    ortho = float(np.linalg.norm(
        np.array(QR.Q).T @ np.array(QR.Q) - np.eye(QR.m)))
    return rec_err, lap_err, ortho


def compare_svd(A_py):
    A = np.array(A_py)
    SVD = lb.svd_decompose([row[:] for row in A_py])
    rec = SVD.reconstruct()
    rec_err = _rel(rec, A_py)
    U_l, S_l, Vt_l = np.linalg.svd(A, full_matrices=False)
    lap_err = _rel((U_l * S_l) @ Vt_l, A)
    s_err = float(np.linalg.norm(np.array(SVD.S) - S_l)
                  / max(np.linalg.norm(S_l), 1e-300))
    return rec_err, lap_err, s_err, SVD.iters, SVD.converged


def compare_eig(A_py):
    A = np.array(A_py)
    R = lb.eig([row[:] for row in A_py])
    ev_ours = np.array(sorted(zip(R.eigenvalues_real, R.eigenvalues_imag),
                              key=lambda z: (z[0], z[1])))
    ev_lap = np.array(sorted(
        [(complex(z).real, complex(z).imag) for z in np.linalg.eigvals(A)],
        key=lambda z: (z[0], z[1])))
    # compare eigenvalue multisets
    err = float(np.linalg.norm(ev_ours - ev_lap)
                / max(np.linalg.norm(ev_lap), 1e-300))
    return err, R.iters, R.force_deflations, R.symmetric


# ---- report -----------------------------------------------------------------

def row(cols, widths):
    return "  ".join(f"{str(c):>{w}}" for c, w in zip(cols, widths))


def main():
    print("=" * 90)
    print(" Conditioning study: from-scratch vs LAPACK")
    print("=" * 90)

    cases = []
    rng_seed = 0
    cases.append(("rand well-cond n=6", np.random.default_rng(rng_seed).standard_normal((6, 6))))
    for n in [4, 6, 8, 10]:
        cases.append((f"Hilbert({n})", np.array(hilbert(n))))
    for n in [4, 6, 8]:
        cases.append((f"Vandermonde({n})", np.array(vandermonde(n))))
    for k in [2, 4, 6, 8, 10, 12, 14]:
        cases.append((f"rand cond=1e{k}", random_with_cond(8, 10 ** k, seed=k)))
    cases.append(("complex 4x4", complex_spectrum_4x4()))

    W = [22, 10, 10, 10, 10, 10, 10, 8, 8, 6]
    header = ["case", "cond", "LU rec", "LAPACK", "growth",
              "QR rec", "SVD rec", "SVD s.err", "eig err", "fd"]
    print(row(header, W))
    print("-" * (sum(W) + 2 * len(W)))

    for name, A in cases:
        A_py = A.tolist()
        cond = float(np.linalg.cond(A))
        lu_r, lu_l, growth, sing = compare_lu(A_py)
        qr_r, qr_l, ortho = compare_qr(A_py) if A.shape[0] == A.shape[1] else (float('nan'), float('nan'), float('nan'))
        svd_r, svd_l, svd_s, sweeps, conv = compare_svd(A_py)
        eig_e, eig_it, fd, sym = compare_eig(A_py)
        print(row([name, f"{cond:.1e}", f"{lu_r:.1e}", f"{lu_l:.1e}",
                   f"{growth:.2f}", f"{qr_r:.1e}", f"{svd_r:.1e}",
                   f"{svd_s:.1e}", f"{eig_e:.1e}", f"{fd}"], W))

    print()
    print("Legend:")
    print("  LU rec / QR rec / SVD rec : ||A - reconstruct|| / ||A||")
    print("  LAPACK                    : same quantity from scipy.linalg")
    print("  growth                    : ||U||_inf / ||A||_inf (LU element growth)")
    print("  SVD s.err                 : ||sigma_ours - sigma_lapack|| / ||sigma_lapack||")
    print("  eig err                   : ||sorted(ev_ours) - sorted(ev_lapack)|| / ||ev_lapack||")
    print("  fd                        : #force-deflations in eigen (single-shift QR")
    print("                              can't deflate complex blocks; non-zero => distrust)")


if __name__ == "__main__":
    main()