"""linalg_bedrock — linear algebra from raw arrays.

The decomposition routines in this package operate on plain Python
lists of lists.  numpy / scipy are *not* imported by the solver; they
appear only in tests (as the LAPACK reference) and in the
conditioning-study script.
"""
from ._core import (
    zeros, identity, copy_matrix, copy_vector,
    matmul, transpose, matvec, dot, norm2,
    frobenius, norm_inf_matrix, swap_rows, fmt_matrix,
)
from .lu import lu_decompose, lu_solve, LUDecomposition
from .qr import qr_decompose, qr_solve, QRDecomposition
from .svd import svd_decompose, SVDDecomposition
from .eigen import eig, EigenResult

__all__ = [
    # core
    "zeros", "identity", "copy_matrix", "copy_vector",
    "matmul", "transpose", "matvec", "dot", "norm2",
    "frobenius", "norm_inf_matrix", "swap_rows", "fmt_matrix",
    # decompositions
    "lu_decompose", "lu_solve", "LUDecomposition",
    "qr_decompose", "qr_solve", "QRDecomposition",
    "svd_decompose", "SVDDecomposition",
    "eig", "EigenResult",
]