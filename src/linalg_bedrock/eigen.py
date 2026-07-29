"""Eigenvalues via Hessenberg reduction + shifted QR iteration.

Pipeline:
    A  ->  H  (upper Hessenberg, via Householder similarity)
    H  ->  T  (quasi-upper-triangular Schur form, via Wilkinson-shift QR)
    eigenvalues are read off the diagonal (1x1 blocks) or the eigenvalues
    of trailing 2x2 blocks (real or complex-conjugate pairs).

For symmetric A, H is symmetric tridiagonal and T is diagonal; columns
of the accumulated orthogonal Z are then the eigenvectors.

For nonsymmetric A with complex eigenvalues, single-shift QR with a
real shift cannot fully deflate complex 2x2 blocks (this needs the
Francis double shift).  When we hit `max_iter_per_eig` without
convergence we force-deflate the trailing 2x2 block and read its
eigenvalues analytically; the result is a *Schur-approximation* rather
than a converged Schur form.  The `force_deflations` counter in
EigenResult exposes how often this happened — it is the diagnostic
that tells you the answer is not to be trusted for that matrix.
"""
from __future__ import annotations
import math
from ._core import (
    Matrix, zeros, identity, copy_matrix, matmul, transpose,
    householder_vector,
)


def _is_symmetric(A: Matrix, tol: float = 1e-10) -> bool:
    n = len(A)
    for i in range(n):
        Ai = A[i]
        for j in range(i + 1, n):
            if abs(Ai[j] - A[j][i]) > tol * (1.0 + abs(Ai[j]) + abs(A[j][i])):
                return False
    return True


def _hessenberg(A: Matrix) -> tuple[Matrix, Matrix]:
    """H = Q^T A Q  with H upper Hessenberg, Q orthogonal."""
    n = len(A)
    H = copy_matrix(A)
    Q = identity(n)
    for k in range(n - 2):
        x = [H[i][k] for i in range(k + 1, n)]
        v, beta = householder_vector(x)
        if beta == 0.0:
            continue
        L = n - k - 1
        # left:  H[k+1:, :] -= beta v (v^T H[k+1:, :])
        for j in range(n):
            s = 0.0
            for i in range(L):
                s += v[i] * H[k + 1 + i][j]
            s *= beta
            if s != 0.0:
                for i in range(L):
                    H[k + 1 + i][j] -= s * v[i]
        # right: H[:, k+1:] -= beta (H[:, k+1:] v) v^T
        for i in range(n):
            s = 0.0
            Hi = H[i]
            for j in range(L):
                s += Hi[k + 1 + j] * v[j]
            s *= beta
            if s != 0.0:
                for j in range(L):
                    Hi[k + 1 + j] -= s * v[j]
        # accumulate Q := Q (I - beta v v^T)
        for i in range(n):
            s = 0.0
            Qi = Q[i]
            for j in range(L):
                s += Qi[k + 1 + j] * v[j]
            s *= beta
            if s != 0.0:
                for j in range(L):
                    Qi[k + 1 + j] -= s * v[j]
    return H, Q


def _wilkinson_shift(a: float, b: float, c: float, d: float) -> float:
    """Eigenvalue of [[a,b],[c,d]] closer to d (real part if complex)."""
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    if disc < 0.0:
        return tr / 2.0
    s = math.sqrt(disc)
    l1 = (tr + s) / 2.0
    l2 = (tr - s) / 2.0
    return l1 if abs(l1 - d) < abs(l2 - d) else l2


def _qr_step(H: Matrix, lo: int, hi: int, Z: Matrix | None) -> None:
    """One explicit single-shift QR sweep on H[lo:hi+1, lo:hi+1]."""
    a = H[hi - 1][hi - 1]; b = H[hi - 1][hi]
    c = H[hi][hi - 1];     d = H[hi][hi]
    shift = _wilkinson_shift(a, b, c, d)
    for i in range(lo, hi + 1):
        H[i][i] -= shift
    # QR via Givens; store rotations to apply from the right afterwards
    cs = []
    for k in range(lo, hi):
        a_ = H[k][k]; b_ = H[k + 1][k]
        r = math.hypot(a_, b_)
        if r == 0.0:
            c_, s_ = 1.0, 0.0
        else:
            c_, s_ = a_ / r, b_ / r
        cs.append((c_, s_))
        for j in range(lo, hi + 1):
            t1 = H[k][j]; t2 = H[k + 1][j]
            H[k][j]     =  c_ * t1 + s_ * t2
            H[k + 1][j] = -s_ * t1 + c_ * t2
    for k in range(lo, hi):
        c_, s_ = cs[k - lo]
        for i in range(lo, hi + 1):
            t1 = H[i][k]; t2 = H[i][k + 1]
            H[i][k]     =  c_ * t1 + s_ * t2
            H[i][k + 1] = -s_ * t1 + c_ * t2
        if Z is not None:
            for i in range(len(Z)):
                t1 = Z[i][k]; t2 = Z[i][k + 1]
                Z[i][k]     =  c_ * t1 + s_ * t2
                Z[i][k + 1] = -s_ * t1 + c_ * t2
    for i in range(lo, hi + 1):
        H[i][i] += shift


class EigenResult:
    __slots__ = ("eigenvalues_real", "eigenvalues_imag", "eigenvectors",
                 "n", "symmetric", "iters", "force_deflations")

    def __init__(self, evr, evi, V, n, symmetric, iters, force_deflations):
        self.eigenvalues_real = evr
        self.eigenvalues_imag = evi
        self.eigenvectors = V      # n x n with eigenvectors as columns, or None
        self.n = n
        self.symmetric = symmetric
        self.iters = iters
        self.force_deflations = force_deflations


def eig(A: Matrix, max_iter_per_eig: int = 100, tol: float = 1e-12,
        compute_vectors: bool = True) -> EigenResult:
    n = len(A)
    if n == 0 or any(len(r) != n for r in A):
        raise ValueError("eig requires a square matrix")
    symmetric = _is_symmetric(A)
    H, Q = _hessenberg(A)
    Z = [row[:] for row in Q] if compute_vectors else None

    hi = n - 1
    total_iters = 0
    force_deflations = 0
    iters_here = 0
    max_total = max_iter_per_eig * n + 100

    while hi > 0:
        if total_iters > max_total:
            break
        # find smallest lo such that H[lo:hi+1, lo:hi+1] is unreduced
        lo = hi
        while lo > 0 and abs(H[lo][lo - 1]) > tol * (
                abs(H[lo - 1][lo - 1]) + abs(H[lo][lo]) + 1e-300):
            lo -= 1

        if lo == hi:
            hi -= 1
            iters_here = 0
            continue
        if lo == hi - 1:
            hi -= 2
            iters_here = 0
            continue

        _qr_step(H, lo, hi, Z)
        total_iters += 1
        iters_here += 1

        if iters_here >= max_iter_per_eig:
            # 2x2 block refused to converge (typical: complex eigenvalues
            # under a real shift).  Force-deflate and read eigenvalues
            # from the current 2x2 block; flag for the caller.
            force_deflations += 1
            hi -= 2
            iters_here = 0

    # extract eigenvalues from quasi-triangular H
    evr = [0.0] * n
    evi = [0.0] * n
    i = 0
    while i < n:
        if i == n - 1 or abs(H[i + 1][i]) <= tol * (
                abs(H[i][i]) + abs(H[i + 1][i + 1]) + 1e-300):
            evr[i] = H[i][i]
            evi[i] = 0.0
            i += 1
        else:
            a = H[i][i]; b = H[i][i + 1]
            c = H[i + 1][i]; d = H[i + 1][i + 1]
            tr = a + d
            det = a * d - b * c
            disc = tr * tr - 4 * det
            if disc >= 0:
                s = math.sqrt(disc)
                evr[i] = (tr + s) / 2.0
                evr[i + 1] = (tr - s) / 2.0
                evi[i] = evi[i + 1] = 0.0
            else:
                s = math.sqrt(-disc)
                evr[i] = evr[i + 1] = tr / 2.0
                evi[i] = s / 2.0
                evi[i + 1] = -s / 2.0
            i += 2

    # eigenvectors: for symmetric A, columns of Z are eigenvectors
    # (Z is orthogonal and A = Z D Z^T with D diagonal at convergence).
    V = Z if (compute_vectors and symmetric) else None
    return EigenResult(evr, evi, V, n, symmetric, total_iters, force_deflations)


__all__ = ["eig", "EigenResult"]