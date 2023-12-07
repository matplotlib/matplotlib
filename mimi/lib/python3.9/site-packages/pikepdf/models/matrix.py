# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""PDF content matrix support."""

from __future__ import annotations

from math import cos, pi, sin

from deprecated import deprecated


@deprecated('use pikepdf.Matrix instead')
class PdfMatrix:
    """Support class for PDF content stream matrices.

    PDF content stream matrices are 3x3 matrices summarized by a shorthand
    ``(a, b, c, d, e, f)``, where the first column vector is ``(a, c, e)``
    and the second column vector is ``(b, d, f)``. The final column vector
    is always ``(0, 0, 1)`` since PDF uses
    `homogenous coordinates <https://en.wikipedia.org/wiki/Homogeneous_coordinates>`_.

    ``a`` is the horizontal scaling factor.
    ``b`` is horizontal skewing.
    ``c`` is vertical skewing.
    ``d`` is the vertical scaling factor.
    ``e`` is the horizontal translation.
    ``f`` is the vertical translation.

    For scaling, ``a`` and ``d`` are the scaling factors in the horizontal and vertical
    directions, respectively; for pure scaling, ``b`` and ``c`` are zero.

    PDF uses row vectors.  That is, ``vr @ A'`` gives the effect of transforming
    a row vector ``vr=(x, y, 1)`` by the matrix ``A'``.  Most textbook
    treatments use ``A @ vc`` where the column vector ``vc=(x, y, 1)'``.

    Matrices should be **premultipled** with other matrices to concatenate
    transformations.

    (``@`` is the Python matrix multiplication operator.)

    Addition and other operations are not implemented because they're not that
    meaningful in a PDF context (they can be defined and are mathematically
    meaningful in general).

    PdfMatrix objects are immutable. All transformations on them produce a new
    matrix.

    .. deprecated:: 8.7
        Use :class:`pikepdf.Matrix` instead.
    """

    def __init__(self, *args):
        """Initialize a PdfMatrix."""
        # fmt: off
        if not args:
            self.values = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        elif len(args) == 6:
            a, b, c, d, e, f = map(float, args)
            self.values = ((a, b, 0),
                           (c, d, 0),
                           (e, f, 1))
        elif isinstance(args[0], PdfMatrix):
            self.values = args[0].values
        elif len(args[0]) == 6:
            a, b, c, d, e, f = map(float, args[0])
            self.values = ((a, b, 0),
                           (c, d, 0),
                           (e, f, 1))
        elif len(args[0]) == 3 and len(args[0][0]) == 3:
            self.values = (tuple(args[0][0]),
                           tuple(args[0][1]),
                           tuple(args[0][2]))
        else:
            try:
                import numpy as np
                if isinstance(args[0], (np.ndarray, np.generic)):
                    self.values = tuple(map(tuple, args[0]))
            except ImportError:
                pass
            raise ValueError('invalid arguments: ' + repr(args))
        # fmt: on

    @staticmethod
    def identity():
        """Return an identity matrix."""
        return PdfMatrix()

    def __matmul__(self, other):
        """Multiply this matrix by another matrix.

        Can be used to concatenate transformations. Transformations should be composed
        by pre-multiplying matrices.
        """
        a = self.values
        b = other.values
        return PdfMatrix(
            [
                [sum(float(i) * float(j) for i, j in zip(row, col)) for col in zip(*b)]
                for row in a
            ]
        )

    def __array__(self):
        """Return a numpy array of the matrix.

        This function requires numpy, which is an optional dependency of pikepdf.
        If numpy is not installed, an ImportError will be raised.
        """
        import numpy as np

        return np.array(self.values)

    def inverse(self):
        """Return the inverse of this matrix.

        The inverse matrix reverses the transformation of the original matrix.

        This function requires numpy, which is an optional dependency of pikepdf.
        If numpy is not installed, an ImportError will be raised.
        """
        import numpy as np

        return PdfMatrix(np.linalg.inv(self.__array__()))

    def scaled(self, x, y):
        """Concatenate a scaling matrix to this matrix.

        .. warning::
            This function is subtly incorrect, because it post-multiplies by the
            scaling matrix instead of pre-multiplying. It is assumed that any users
            of the code may have noticed this and corrected it by compensating
            for it, so correcting the error would be a breaking change.
        """
        return self @ PdfMatrix((x, 0, 0, y, 0, 0))

    def rotated(self, angle_degrees_ccw):
        """Concatenate a rotation matrix to this matrix.

        .. warning::
            This function is subtly incorrect, because it post-multiplies by the
            scaling matrix instead of pre-multiplying. It is assumed that any users
            of the code may have noticed this and corrected it by compensating
            for it, so correcting the error would be a breaking change.
        """
        angle = angle_degrees_ccw / 180.0 * pi
        c, s = cos(angle), sin(angle)
        return self @ PdfMatrix((c, s, -s, c, 0, 0))

    def translated(self, x, y):
        """Translate this matrix.

        .. warning::
            This function is subtly incorrect, because it post-multiplies by the
            scaling matrix instead of pre-multiplying. It is assumed that any users
            of the code may have noticed this and corrected it by compensating
            for it, so correcting the error would be a breaking change.
        """
        return self @ PdfMatrix((1, 0, 0, 1, x, y))

    @property
    def shorthand(self):
        """Return the 6-tuple (a,b,c,d,e,f) that describes this matrix."""
        return (self.a, self.b, self.c, self.d, self.e, self.f)

    @property
    def a(self):
        """Return the horizontal scaling factor."""
        return self.values[0][0]

    @property
    def b(self):
        """Return horizontal skew."""
        return self.values[0][1]

    @property
    def c(self):
        """Return vertical skew."""
        return self.values[1][0]

    @property
    def d(self):
        """Return the vertical scaling factor."""
        return self.values[1][1]

    @property
    def e(self):
        """Return the horizontal translation.

        Typically corresponds to translation on the x-axis.
        """
        return self.values[2][0]

    @property
    def f(self):
        """Return the vertical translation.

        Typically corresponds to translation on the y-axis.
        """
        return self.values[2][1]

    def __eq__(self, other):
        if isinstance(other, PdfMatrix):
            return self.shorthand == other.shorthand
        return False

    def encode(self):
        """Encode this matrix in binary suitable for including in a PDF."""
        return '{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
            self.a, self.b, self.c, self.d, self.e, self.f
        ).encode()

    def __repr__(self):
        return f'pikepdf.PdfMatrix({repr(self.values)})'
