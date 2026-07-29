"""Validate LU against scipy.linalg.lu (LAPACK)."""
import math
import numpy as np
import scipy.linalg as sla
import pytest

import linalg_bedrock as lb
from linalg_bedrock._core import frobenius


def _to_py(M):
    return [[float(x) for x in row] for row in M]


def _reconstruction_error(A, LU):
    rec = LU.reconstruct()
    a = np.array(A, dtype=float)
    r = np.array(rec, dtype=float)
    return np.linalg.norm(a - r) / np.linalg.norm(a)


def test_lu_identity():
    A = [[1.0, 0.0], [0.0, 1.0]]
    LU = lb.lu_decompose(A)
    assert LU.singular is False
    assert _reconstruction_error(A, LU) < 1e-14


def test_lu_well_conditioned():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    LU = lb.lu_decompose(_to_py(A))
    assert _reconstruction_error(_to_py(A), LU) < 1e-12
    # cross-check against LAPACK
    P, L, U = sla.lu(A)
    assert np.allclose(np.array(LU.L), L, atol=1e-10)
    assert np.allclose(np.array(LU.U), U, atol=1e-10)


def test_lu_pivoting_required():
    # 0 in (0,0) forces a row swap
    A = [[0.0, 2.0], [1.0, 1.0]]
    LU = lb.lu_decompose(A)
    assert LU.perm[0] == 1
    assert _reconstruction_error(A, LU) < 1e-14
    x = LU.solve([2.0, 2.0])
    assert np.allclose(np.array(A) @ np.array(x), [2.0, 2.0])


def test_lu_solve_matches_lapack():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((8, 8))
    b = rng.standard_normal(8)
    x_ours = lb.lu_solve(_to_py(A), list(b))
    x_lap = np.linalg.solve(A, b)
    assert np.allclose(x_ours, x_lap, atol=1e-10)


def test_lu_singular_flagged():
    A = [[1.0, 2.0], [2.0, 4.0]]
    LU = lb.lu_decompose(A)
    assert LU.singular is True


def test_lu_hilbert_grows():
    # Hilbert(6) is famously ill-conditioned; reconstruction should
    # still be accurate to ~ eps * cond, NOT to eps.
    n = 6
    H = [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]
    LU = lb.lu_decompose(H)
    err = _reconstruction_error(H, LU)
    cond = np.linalg.cond(np.array(H))
    # backward error should be O(eps); reconstruction error O(eps * cond)
    assert err < 10 * cond * 1e-16
    assert LU.growth < 5.0   # Hilbert doesn't trigger growth; condition is intrinsic