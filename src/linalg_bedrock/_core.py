"""Raw-array primitives shared by all decompositions.

Everything here is O(n^3) or below, plain Python.  No numpy.
"""
from __future__ import annotations
import math

Matrix = list   # list[list[float]]
Vector = list   # list[float]


# ---------- constructors ----------------------------------------------------

def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> Matrix:
    M = zeros(n, n)
    for i in range(n):
        M[i][i] = 1.0
    return M


def copy_matrix(A: Matrix) -> Matrix:
    return [row[:] for row in A]


def copy_vector(v: Vector) -> Vector:
    return v[:]


# ---------- basic linear algebra -------------------------------------------

def matmul(A: Matrix, B: Matrix) -> Matrix:
    n, m, p = len(A), len(A[0]), len(B[0])
    C = zeros(n, p)
    for i in range(n):
        Ai, Ci = A[i], C[i]
        for k in range(m):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(p):
                Ci[j] += a * Bk[j]
    return C


def transpose(A: Matrix) -> Matrix:
    n, m = len(A), len(A[0])
    T = zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T


def matvec(A: Matrix, x: Vector) -> Vector:
    n, m = len(A), len(A[0])
    return [sum(A[i][j] * x[j] for j in range(m)) for i in range(n)]


def dot(x: Vector, y: Vector) -> float:
    s = 0.0
    for i in range(len(x)):
        s += x[i] * y[i]
    return s


def norm2(x: Vector) -> float:
    return math.sqrt(dot(x, x))


def frobenius(A: Matrix) -> float:
    s = 0.0
    for row in A:
        for a in row:
            s += a * a
    return math.sqrt(s)


def norm_inf_matrix(A: Matrix) -> float:
    return max((sum(abs(a) for a in row) for row in A), default=0.0)


def swap_rows(A: Matrix, i: int, j: int) -> None:
    A[i], A[j] = A[j], A[i]


# ---------- householder helper (shared by QR and eigen) ---------------------

def householder_vector(x: Vector) -> tuple[Vector, float]:
    """Return (v, beta) with v[0] == 1 such that  H = I - beta v v^T
    sends x to  alpha * e_1, where alpha = -sign(x[0]) * ||x||.

    The sign choice avoids cancellation when x[0] is large and positive.
    beta == 0 means x is already a multiple of e_1 (no reflector needed).
    """
    n = len(x)
    sigma = 0.0
    for i in range(1, n):
        sigma += x[i] * x[i]
    v = x[:]
    if sigma == 0.0:
        v[0] = 1.0
        return v, (0.0 if x[0] >= 0.0 else -2.0)
    mu = math.sqrt(x[0] * x[0] + sigma)
    alpha = mu if x[0] <= 0.0 else -mu
    v[0] = x[0] - alpha
    vnorm2 = v[0] * v[0] + sigma
    if vnorm2 == 0.0:
        return v, 0.0
    return v, 2.0 / vnorm2


def fmt_matrix(A: Matrix, ndigits: int = 6) -> str:
    return "\n".join(
        "  ".join(f"{v:+.{ndigits}f}" for v in row) for row in A
    )


__all__ = [
    "Matrix", "Vector",
    "zeros", "identity", "copy_matrix", "copy_vector",
    "matmul", "transpose", "matvec", "dot", "norm2",
    "frobenius", "norm_inf_matrix", "swap_rows",
    "householder_vector", "fmt_matrix",
]