"""SVD via one-sided Jacobi.

    A = U S V^T

One-sided Jacobi orthogonalizes pairs of columns of A by 2x2 Jacobi
rotations; after convergence the column norms of the rotated A are the
singular values and the normalized columns are U.  The accumulated
rotations form V.

Why this algorithm (and not the cheaper "eig of A^T A"): forming A^T A
squares the condition number, so singular values below sqrt(eps) ~ 1e-8
are lost in noise.  One-sided Jacobi works on A directly and is the
algorithm that, per Demmel-Veselic (1989), reaches high relative
accuracy on matrices with well-scaled rows.
"""
from __future__ import annotations
import math
from ._core import (
    Matrix, Vector, zeros, identity, copy_matrix, transpose, matmul,
)


class SVDDecomposition:
    __slots__ = ("U", "S", "Vt", "m", "n", "iters", "converged")

    def __init__(self, U, S, Vt, m, n, iters, converged):
        self.U = U         # m x min(m,n), columns are left singular vectors
        self.S = S         # length min(m,n), descending
        self.Vt = Vt       # min(m,n) x n, rows are right singular vectors
        self.m = m
        self.n = n
        self.iters = iters
        self.converged = converged

    def reconstruct(self, k: int | None = None) -> Matrix:
        m, n = self.m, self.n
        r = len(self.S) if k is None else min(k, len(self.S))
        A = zeros(m, n)
        for t in range(r):
            s = self.S[t]
            if s == 0.0:
                continue
            for i in range(m):
                ui = self.U[i][t]
                if ui == 0.0:
                    continue
                Vt_t = self.Vt[t]
                for j in range(n):
                    A[i][j] += ui * s * Vt_t[j]
        return A


def _rotate(A: Matrix, V: Matrix, p: int, q: int,
            m: int, n: int) -> None:
    """One Jacobi rotation that orthogonalizes columns p, q of A.

    Uses the rotation  J = [[c, -s], [s, c]]  applied from the right
    (A <- A J, V <- V J).  The angle is chosen so that the (p,q) entry
    of  (A J)^T (A J)  is zero.  The smaller-magnitude root of the
    diagonalizing quadratic is

        t = -sign(tau) / (|tau| + sqrt(1 + tau^2)),
        tau = (a_qq - a_pp) / (2 a_pq).

    (Sign verified by hand on M=[[2,1],[1,3]]: with +sign the
    off-diagonal does NOT vanish; with -sign it does.)
    """
    app = aqq = apq = 0.0
    for i in range(m):
        aip = A[i][p]; aiq = A[i][q]
        app += aip * aip
        aqq += aiq * aiq
        apq += aip * aiq
    if apq == 0.0:
        return

    tau = (aqq - app) / (2.0 * apq)
    if tau >= 0.0:
        t = -1.0 / (tau + math.sqrt(1.0 + tau * tau))
    else:
        t = 1.0 / (-tau + math.sqrt(1.0 + tau * tau))
    c = 1.0 / math.sqrt(1.0 + t * t)
    s = t * c

    for i in range(m):
        aip = A[i][p]; aiq = A[i][q]
        A[i][p] = c * aip - s * aiq
        A[i][q] = s * aip + c * aiq
    for i in range(n):
        vip = V[i][p]; viq = V[i][q]
        V[i][p] = c * vip - s * viq
        V[i][q] = s * vip + c * viq


def svd_decompose(A: Matrix, max_sweeps: int = 60,
                  tol: float = 1e-14) -> SVDDecomposition:
    m = len(A)
    if m == 0:
        raise ValueError("empty matrix")
    n = len(A[0])

    # For wide matrices, work with A^T and swap roles at the end.
    if m < n:
        sub = svd_decompose(transpose(A), max_sweeps, tol)
        # A^T = U' S' V'^T   =>   A = V' S' U'^T
        U_A = transpose(sub.Vt)   # m x m   (rows of V'  -> cols of V'  = U of A)
        Vt_A = transpose(sub.U)   # m x n   (cols of U'^T -> rows of U'^T = V^T of A)
        return SVDDecomposition(U_A, sub.S, Vt_A, m, n,
                                sub.iters, sub.converged)

    W = copy_matrix(A)
    V = identity(n)
    converged = False
    sweeps = 0
    for sweep in range(max_sweeps):
        sweeps += 1
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                app = aqq = apq = 0.0
                for i in range(m):
                    aip = W[i][p]; aiq = W[i][q]
                    app += aip * aip
                    aqq += aiq * aiq
                    apq += aip * aiq
                if apq == 0.0:
                    continue
                # Demmel-Veselic relative convergence test
                if abs(apq) <= tol * math.sqrt(app * aqq) + 1e-300:
                    continue
                _rotate(W, V, p, q, m, n)
                rotated = True
        if not rotated:
            converged = True
            break

    S = [0.0] * n
    U = zeros(m, n)
    for j in range(n):
        col_norm = 0.0
        for i in range(m):
            col_norm += W[i][j] * W[i][j]
        col_norm = math.sqrt(col_norm)
        S[j] = col_norm
        if col_norm > 0.0:
            inv = 1.0 / col_norm
            for i in range(m):
                U[i][j] = W[i][j] * inv
        # else: leave U[:,j] = 0 (rank-deficient case)

    order = sorted(range(n), key=lambda i: -S[i])
    S_sorted = [S[i] for i in order]
    U_sorted = [[U[i][order[j]] for j in range(n)] for i in range(m)]
    V_sorted = [[V[i][order[j]] for j in range(n)] for i in range(n)]
    Vt = transpose(V_sorted)   # rows of Vt = right singular vectors

    return SVDDecomposition(
        U=U_sorted, S=S_sorted, Vt=Vt, m=m, n=n,
        iters=sweeps, converged=converged,
    )


__all__ = ["SVDDecomposition", "svd_decompose"]