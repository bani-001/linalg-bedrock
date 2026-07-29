import numpy as np
import scipy.linalg as sla
import pytest

import linalg_bedrock as lb
from linalg_bedrock._core import matmul, transpose


def _err_reconstruct(A, QR):
    rec = matmul(QR.Q, QR.R)
    a = np.array(A, dtype=float)
    r = np.array(rec, dtype=float)
    return np.linalg.norm(a - r) / np.linalg.norm(a)


def _err_orthogonality(QR):
    Q = np.array(QR.Q, dtype=float)
    return np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0]))


def test_qr_square():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    QR = lb.qr_decompose([[float(x) for x in row] for row in A])
    assert _err_reconstruct(A.tolist(), QR) < 1e-12
    assert _err_orthogonality(QR) < 1e-12
    # cross-check R against LAPACK (Q may differ by signs per row)
    _, R_lap = np.linalg.qr(A)
    R_ours = np.array(QR.R, dtype=float)
    assert np.allclose(np.abs(R_ours), np.abs(R_lap), atol=1e-10)


def test_qr_tall():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((8, 4))
    QR = lb.qr_decompose([[float(x) for x in row] for row in A])
    assert _err_reconstruct(A.tolist(), QR) < 1e-12
    assert _err_orthogonality(QR) < 1e-12


def test_qr_solve_least_squares():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((10, 3))
    b = rng.standard_normal(10)
    x_ours = lb.qr_solve([[float(x) for x in row] for row in A], list(b))
    x_lap, *_ = np.linalg.lstsq(A, b, rcond=None)
    assert np.allclose(x_ours, x_lap, atol=1e-10)


def test_qr_rank_revealed():
    # rank-2 matrix in 4x3
    A = np.array([[1., 2., 4.],
                  [2., 4., 8.],
                  [3., 6., 12.],
                  [1., 1., 1.]])
    QR = lb.qr_decompose(A.tolist())
    assert QR.rank == 2