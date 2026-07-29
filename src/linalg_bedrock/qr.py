"""QR decomposition via Householder reflectors.

    A = Q R

Q is m x m orthogonal, R is m x n upper-trapezoidal (zeros below the
main diagonal).  Householder QR is backward stable: there exists a
nearby A+E with ||E|| = O(eps ||A||) such that the computed (Q,R)
exactly factor A+E.  This is the same guarantee LAPACK's DGEQRF gives
(modulo blocking).
"""
from __future__ import annotations
import math
from ._core import (
    Matrix, Vector, zeros, identity, copy_matrix, matmul,
    householder_vector,
)


class QRDecomposition:
    __slots__ = ("Q", "R", "m", "n", "rank", "rdiag_max")

    def __init__(self, Q, R, m, n, rank, rdiag_max):
        self.Q = Q
        self.R = R
        self.m = m
        self.n = n
        self.rank = rank
        self.rdiag_max = rdiag_max

    def reconstruct(self) -> Matrix:
        return matmul(self.Q, self.R)

    def solve(self, b: Vector) -> Vector:
        """Least-squares solve.  For square full-rank A this is A x = b."""
        m, n = self.m, self.n
        if len(b) != m:
            raise ValueError("dimension mismatch")
        Qtb = [0.0] * m
        for i in range(m):
            s = 0.0
            for j in range(m):
                s += self.Q[j][i] * b[j]
            Qtb[i] = s
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = Qtb[i]
            for j in range(i + 1, n):
                s -= self.R[i][j] * x[j]
            d = self.R[i][i]
            if abs(d) < 1e-300:
                raise ZeroDivisionError(f"zero R[{i}][{i}]")
            x[i] = s / d
        return x


def qr_decompose(A: Matrix, tol: float = 1e-12) -> QRDecomposition:
    m = len(A)
    if m == 0:
        raise ValueError("empty matrix")
    n = len(A[0])
    if m < n:
        raise ValueError("qr_decompose requires m >= n; transpose first")
    R = copy_matrix(A)
    Q = identity(m)
    r = min(m - 1, n)
    for k in range(r):
        x = [R[i][k] for i in range(k, m)]
        v, beta = householder_vector(x)
        if beta == 0.0:
            continue
        # apply H = I - beta v v^T  to R[k:, k:]
        for j in range(k, n):
            s = 0.0
            for i in range(m - k):
                s += v[i] * R[k + i][j]
            s *= beta
            for i in range(m - k):
                R[k + i][j] -= s * v[i]
        # accumulate Q := Q H_k  (apply H_k from the right to Q[:, k:])
        for j in range(m):
            s = 0.0
            for i in range(m - k):
                s += v[i] * Q[j][k + i]
            s *= beta
            for i in range(m - k):
                Q[j][k + i] -= s * v[i]

    diag = [abs(R[i][i]) for i in range(min(m, n))]
    dmax = max(diag) if diag else 0.0
    rank = sum(1 for d in diag if d > tol * (dmax + 1e-300)) if dmax > 0 else 0
    return QRDecomposition(Q=Q, R=R, m=m, n=n, rank=rank, rdiag_max=dmax)


def qr_solve(A: Matrix, b: Vector) -> Vector:
    return qr_decompose(A).solve(b)


__all__ = ["QRDecomposition", "qr_decompose", "qr_solve"]