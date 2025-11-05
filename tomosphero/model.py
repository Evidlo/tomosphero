#!/usr/bin/env python3

"""
Models are mappings from a (usually) low dimensional space of coefficients to a 3D/4D object space.
They are useful as a tool in tomographic retrievals to enforce certain properties in reconstructed objects.

Models need not be linear, but they should be differentiable and implemented using PyTorch functions.
"""

import torch as t

from tomosphero.geometry import SphericalGrid

class Model:
    """A parameterized model for an object.  Subclass this class and provide
    implementations for `__init__`, `__call__` and `coeffs_shape`, to make custom models

    Properties:
        coeffs_shape (tuple): Shape of input coeffs

    Usage:
    ``` python
        g = SphericalGrid(...)
        m = MyCustomModel(g)
        coeffs = t.rand(m.coeffs_shape)
        x = m(coeffs)
    ```
    """

    def __init__(self, grid: SphericalGrid):
        """@private
        Do any model setup you need here.
        You may add args/kwargs as necessary (e.g. model params)
        Be sure to store `grid` with `self.grid = grid`.
        """
        raise NotImplementedError

    def __call__(self, coeffs):
        """@private
        Generate object from parameters.

        This function should produce a 3D/4D tensor given inputs `coeffs`.
        Be sure that `device` on the returned tensor matches `coeffs.device`.

        Args:
            coeffs (ndarray or tensor): array of shape self.coeffs_shape

        Returns:
            object (tensor):
        """
        raise NotImplementedError

    @property
    def coeffs_shape(self):
        """tuple[int]: Shape of coeffs"""
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}({tuple(self.grid.shape)})'


class FullyDenseModel(Model):
    """Basic model which is an identity mapping from coefficients→object"""

    def __init__(self, grid: SphericalGrid):
        self.grid = grid

    def __call__(self, coeffs):

        # return coefficients directly
        return coeffs

    @property
    def coeffs_shape(self):
        return self.grid.shape


class CubesModel(Model):
    """Test model with two overlapping boxes in spherical coordinates"""

    def __init__(self, grid: SphericalGrid):
        self.grid = grid
        self.obj = t.zeros(grid.shape)
        r0, r1 = grid.shape.r * t.tensor((.333, .666))
        e00, e01 = grid.shape.e * t.tensor((.2, .3))
        e10, e11 = grid.shape.e * t.tensor((.7, .9))
        a0, a1 = grid.shape.a * t.tensor((.4, .6))

        r0, r1 = int(r0), int(r1)
        e00, e01, e10, e11 = int(e00), int(e01), int(e10), int(e11)
        a0, a1 = int(a0), int(a1)

        # self.obj[int(r0):int(r1), int(e00):int(e01), int(a0):int(a1)] = 1
        # self.obj[int(r0):int(r1), int(e10):int(e11), int(a0):int(a1)] = 1
        self.obj[r0:r1, e00:e01, a0:a1] = 1
        self.obj[r0:r1, e10:e11, a0:a1] = 1

        self.r0, self.r1 = r0, r1
        self.e00, self.e01, self.e10, self.e11 = e00, e01, e10, e11
        self.a0, self.a1 = a0, a1

    def __call__(self, coeffs):
        return self.obj

    @property
    def coeffs_shape(self):
        return ()


class AxisAlignmentModel(Model):
    """Test Model to verify orientation of tomographic operator.  Useful for
    detecting mirroring issues


    ```
    Z
    |
    |
    |   Y
    |  /
    | /
    |/
    .--X
    ```

    """
    def __init__(self, grid: SphericalGrid):
        self.grid = grid
        self.obj = t.zeros(grid.shape)

        # Z axis is longest, followed by Y then X

        # X axis
        self.obj[:grid.shape.r//3, grid.shape.e//2, 0] = 1
        # Y axis
        self.obj[:grid.shape.r//2, grid.shape.e//2, (grid.shape.a*3)//4] = 1
        # Z axis
        self.obj[:, 0, :] = 1

    def __call__(self, coeffs):
        return self.obj

    @property
    def coeffs_shape(self):
        """@private"""
        return ()