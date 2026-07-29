"""LU decomposition with partial (row) pivoting.

    P A = L U

L is unit lower triangular, U is upper triangular, P is a permutation.
The permutation is stored as an index list `perm` (row i of P A is row
`perm[i]` of A) rather than as a dense matrix; `sign` is ±1 for det(P).

Diagnostics exposed for the conditioning study:
    singular   : True if a pivot fell below `eps`
    pivot_min  : smallest |pivot| actually used (inf if singular)
    pivot_max  : largest |pivot| used
    growth     : ||U||_inf / ||A||_inf  — the element-growth factor
                 that, together with cond(A), bounds the backward error.
"""
from __future__ import annotations
import math
from ._core import (
    Matrix, Vector, zeros, copy_matrix, swap_rows,
    matmul, norm_inf_matrix,
)


class LUDecomposition:
    __slots__ = ("L", "U", "perm", "sign", "singular",
                 "pivot_min", "pivot_max", "growth", "n")

    def __init__(self, L, U, perm, sign, singular,
                 pivot_min, pivot_max, growth, n):
        self.L = L
        self.U = U
        self.perm = perm
        self.sign = sign
        self.singular = singular
        self.pivot_min = pivot_min
        self.pivot_max = pivot_max
        self.growth = growth
        self.n = n

    # -- materialize P, or reconstruct A from the factors -------------------

    def P(self) -> Matrix:
        n = self.n
        Pm = zeros(n, n)
        for i, p in enumerate(self.perm):
            Pm[i][p] = 1.0
        return Pm

    def reconstruct(self) -> Matrix:
        """Return P^T (L U), which equals the original A."""
        LU = matmul(self.L, self.U)
        n = self.n
        A = zeros(n, n)
        for i in range(n):
            A[self.perm[i]] = LU[i][:]
        return A

    # -- linear solve -------------------------------------------------------

    def solve(self, b: Vector) -> Vector:
        n = self.n
        if len(b) != n:
            raise ValueError("dimension mismatch")
        pb = [b[self.perm[i]] for i in range(n)]    # forward + back subst.
        y = [0.0] * n
        for i in range(n):
            s = pb[i]
            Li = self.L[i]
            for j in range(i):
                s -= Li[j] * y[j]
            y[i] = s
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = y[i]
            Ui = self.U[i]
            for j in range(i + 1, n):
                s -= Ui[j] * x[j]
            d = Ui[i]
            if abs(d) < 1e-300:
                raise ZeroDivisionError(f"zero pivot at index {i}")
            x[i] = s / d
        return x


def lu_decompose(A: Matrix, eps: float = 1e-14) -> LUDecomposition:
    n = len(A)
    if n == 0 or any(len(r) != n for r in A):
        raise ValueError("LU requires a square matrix")
    a_inf = norm_inf_matrix(A)
    U = copy_matrix(A)             # overwrite: upper tri <- U, strict lower <- L
    perm = list(range(n))
    sign = 1
    singular = False
    pivot_min = math.inf
    pivot_max = 0.0

    for k in range(n):
        # --- partial pivot: max |U[i][k]| for i >= k ---
        pivot_row = k
        pivot_val = abs(U[k][k])
        for i in range(k + 1, n):
            v = abs(U[i][k])
            if v > pivot_val:
                pivot_val, pivot_row = v, i
        if pivot_val < eps:
            singular = True
            for i in range(k + 1, n):
                U[i][k] = 0.0      # L[i][k] stays 0
            continue
        if pivot_row != k:
            swap_rows(U, k, pivot_row)
            perm[k], perm[pivot_row] = perm[pivot_row], perm[k]
            sign = -sign

        pivot = U[k][k]
        ap = abs(pivot)
        if ap < pivot_min: pivot_min = ap
        if ap > pivot_max: pivot_max = ap

        inv = 1.0 / pivot
        Uk = U[k]
        for i in range(k + 1, n):
            Ui = U[i]
            m = Ui[k] * inv
            Ui[k] = m              # store multiplier in strict-lower part
            for j in range(k + 1, n):
                Ui[j] -= m * Uk[j]

    # split U-storage into L and U
    L = zeros(n, n)
    Uu = zeros(n, n)
    for i in range(n):
        L[i][i] = 1.0
        for j in range(i):
            L[i][j] = U[i][j]
        for j in range(i, n):
            Uu[i][j] = U[i][j]

    growth = (norm_inf_matrix(Uu) / a_inf) if a_inf > 0 else 0.0

    return LUDecomposition(
        L=L, U=Uu, perm=perm, sign=sign, singular=singular,
        pivot_min=(0.0 if singular else pivot_min),
        pivot_max=pivot_max, growth=growth, n=n,
    )


def lu_solve(A: Matrix, b: Vector) -> Vector:
    return lu_decompose(A).solve(b)


__all__ = ["LUDecomposition", "lu_decompose", "lu_solve"]         
          

            


