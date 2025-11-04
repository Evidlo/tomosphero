#!/usr/bin/env python3

from tomosphero import *
from tomosphero import SphericalGrid
import torch as t

def test_instantiate_models():
    """Check that models can be instantiated and generate object"""
    g = SphericalGrid()

    for model in (FullyDenseModel, CubesModel, AxisAlignmentModel):
        m = model(g)
        c = t.rand(m.coeffs_shape)
        result = m(c)
        assert result.shape == g.shape, f"Invalid object shape returned by model {m}"