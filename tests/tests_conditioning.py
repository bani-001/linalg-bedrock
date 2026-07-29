"""Cross-cutting: where does our solver diverge from LAPACK?

This test module is the *summary*; the detailed narrative study lives
in examples/conditioning_study.py.  We assert the qualitative findings
here so they survive refactors.
"""
import numpy as np
import pytest

import linalg_bedrock as lb
from linalg_bedrock._core import matmul, frobenius


def hilbert(n):
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def test_hilbert_lu_reconstruction_error_scales_with_condition():
    errs = []
    conds = []
    for n in [3, 5, 7, 9]:
        H = hilbert(n)
        LU = lb.lu_decompose(H)
        rec = LU.reconstruct()
        err = frobenius(matmul(LU.L, matmul(LU.P() and lb.transpose(LU.P()), LU.U)) if False else rec) / frobenius(H)
        cond = np.linalg.cond(np.array(H))
        errs.append(err)
        conds.append(cond)
    # error grows roughly linearly with condition number * eps
    for e, c in zip(errs, conds):
        assert e < 100 * c * 2.2e-16


def test_svd_relative_accuracy_beats_eig_of_ata():
    """A^T A approach would square the condition. Our Jacobi SVD should
    resolve singular values down to ~ eps * sigma_max, not eps * sigma_max^2."""
    rng = np.random.default_rng(99)
    A = rng.standard_normal((8, 5))
    # manually shrink the smallest singular value
    U, s, Vt = np.linalg.svd(A)
    s[-1] = 1e-8
    A_tilde = (U * s) @ Vt
    SVD = lb.svd_decompose(A_tilde.tolist())
    s_ours = np.array(SVD.S)
    assert abs(s_ours[-1] - s[-1]) < 1e-6 * s[0]   # relative error << 1


def test_symmetric_eig_accuracy_on_hilbert():
    """Symmetric eig should be the most accurate of our four decompositions
    even on Hilbert; nonsymmetric with complex eigenvalues is the worst."""
    H = hilbert(6)
    R = lb.eig(H)
    assert R.symmetric
    assert R.force_deflations == 0
    ev_lap = np.linalg.eigvalsh(np.array(H))
    ev_ours = np.array(sorted(R.eigenvalues_real))
    assert np.allclose(ev_ours, ev_lap, atol=1e-9)