import numpy as np
import pytest

import linalg_bedrock as lb
from linalg_bedrock._core import matmul


def _reconstruct(U, S, Vt):
    m = len(U); n = len(Vt[0])
    A = [[0.0] * n for _ in range(m)]
    for t in range(len(S)):
        s = S[t]
        for i in range(m):
            for j in range(n):
                A[i][j] += U[i][t] * s * Vt[t][j]
    return A


def _check_svd(A):
    A = np.array(A, dtype=float)
    SVD = lb.svd_decompose(A.tolist())
    rec = np.array(_reconstruct(SVD.U, SVD.S, SVD.Vt))
    rec_err = np.linalg.norm(A - rec) / np.linalg.norm(A)
    U = np.array(SVD.U); Vt = np.array(SVD.Vt)
    ortho_U = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))
    ortho_V = np.linalg.norm(Vt @ Vt.T - np.eye(Vt.shape[0]))
    s_lap = np.linalg.svd(A, compute_uv=False)
    return rec_err, ortho_U, ortho_V, np.array(SVD.S), s_lap


def test_svd_square():
    rng = np.random.default_rng(10)
    A = rng.standard_normal((5, 5))
    rec_err, oU, oV, s_ours, s_lap = _check_svd(A)
    assert rec_err < 1e-12
    assert oU < 1e-11
    assert oV < 1e-11
    assert np.allclose(s_ours, s_lap, atol=1e-10)


def test_svd_tall():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((8, 3))
    rec_err, oU, oV, s_ours, s_lap = _check_svd(A)
    assert rec_err < 1e-12
    assert oU < 1e-11
    assert oV < 1e-11
    assert np.allclose(s_ours, s_lap, atol=1e-10)


def test_svd_wide():
    rng = np.random.default_rng(12)
    A = rng.standard_normal((3, 8))
    SVD = lb.svd_decompose(A.tolist())
    rec = np.array(_reconstruct(SVD.U, SVD.S, SVD.Vt))
    assert np.linalg.norm(A - rec) / np.linalg.norm(A) < 1e-12
    s_lap = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(np.array(SVD.S), s_lap, atol=1e-10)


def test_svd_rank_deficient():
    # rank-2 in 5x4
    A = np.array([[1., 2., 3., 4.],
                  [2., 4., 6., 8.],
                  [1., 0., 1., 0.],
                  [0., 1., 0., 1.],
                  [1., 1., 1., 1.]])
    A = A * 1.0  # rank 2 (rows 0 and 1 dependent; rows 2 and 3 span)
    # Actually constructed so that there are exactly 2 nonzero singular values
    # in the row-rank-2 case (the matrix above has rank 3).  We instead
    # build a guaranteed rank-2 matrix:
    B = np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])
    A = B @ np.array([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    SVD = lb.svd_decompose(A.tolist())
    s = np.array(SVD.S)
    assert np.sum(s > 1e-10) == 2
    rec = np.array(_reconstruct(SVD.U, SVD.S, SVD.Vt))
    assert np.linalg.norm(A - rec) / np.linalg.norm(A) < 1e-12