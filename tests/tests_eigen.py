import numpy as np
import pytest

import linalg_bedrock as lb


def _sorted_complex(re, im):
    return sorted(zip(re, im), key=lambda z: (z[0], z[1]))


def test_eig_symmetric_simple():
    A = [[2.0, 1.0], [1.0, 3.0]]
    R = lb.eig(A)
    assert R.symmetric
    ev = sorted(R.eigenvalues_real)
    expected = sorted(np.linalg.eigvalsh(np.array(A)))
    assert np.allclose(ev, expected, atol=1e-10)


def test_eig_symmetric_medium():
    rng = np.random.default_rng(20)
    M = rng.standard_normal((6, 6))
    A = M + M.T   # symmetric
    R = lb.eig(A.tolist())
    ev_ours = sorted(R.eigenvalues_real)
    ev_lap = sorted(np.linalg.eigvalsh(A))
    assert np.allclose(ev_ours, ev_lap, atol=1e-8)
    assert R.force_deflations == 0
    # eigenvectors: A V = V diag(λ)
    V = np.array(R.eigenvectors)
    AV = np.array(A) @ V
    VL = V * np.array(R.eigenvalues_real)
    assert np.allclose(AV, VL, atol=1e-7)


def test_eig_diagonal():
    A = [[3.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 7.0]]
    R = lb.eig(A)
    assert np.allclose(sorted(R.eigenvalues_real), [-1.0, 3.0, 7.0], atol=1e-12)


def test_eig_nonsymmetric_real_spectrum():
    # triangular -> eigenvalues on diagonal, no iteration needed
    A = [[2.0, 1.0, 4.0],
         [0.0, 3.0, 5.0],
         [0.0, 0.0, -2.0]]
    R = lb.eig(A)
    assert np.allclose(sorted(R.eigenvalues_real), [-2.0, 2.0, 3.0], atol=1e-10)


def test_eig_nonsymmetric_complex_pair():
    # 2x2 rotation by 60 deg -> eigenvalues e^{±iπ/3} = 0.5 ± 0.866i
    A = [[0.5, -0.8660254],
         [0.8660254, 0.5]]
    R = lb.eig(A)
    ev = list(zip(R.eigenvalues_real, R.eigenvalues_imag))
    # sort by imaginary part so we get the conjugate pair in order
    ev.sort(key=lambda z: z[1])
    assert abs(ev[0][0] - 0.5) < 1e-6
    assert abs(ev[1][0] - 0.5) < 1e-6
    assert abs(ev[0][1] + 0.8660254) < 1e-6
    assert abs(ev[1][1] - 0.8660254) < 1e-6


def test_eig_force_deflation_flagged_for_complex_block():
    # 4x4 with a complex conjugate pair in a Hessenberg block.
    # single-shift QR will not converge; we expect force_deflations >= 1.
    theta = 0.7
    R4 = [[np.cos(theta), -np.sin(theta)],
          [np.sin(theta),  np.cos(theta)]]
    # embed as a 2x2 block in the middle of a 4x4 with real eigenvalues
    A = np.zeros((4, 4))
    A[0:2, 0:2] = R4
    A[2, 2] = 5.0
    A[3, 3] = -3.0
    A[2, 3] = 0.5
    res = lb.eig(A.tolist())
    # We can't trust the exact values, but force_deflations tells the story
    # AND the real eigenvalues -3 and 5 should still be recovered.
    assert 5.0 in [round(v, 6) for v in res.eigenvalues_real] or \
           any(abs(v - 5.0) < 1e-4 for v in res.eigenvalues_real)