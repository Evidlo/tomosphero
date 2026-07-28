#!/usr/bin/env python3

import torch as tr

from .test_raytracer import check
from .geometry import *

def test_sphericalgrid_static():
    grid = SphericalGrid(shape=(10, 11, 12))
    assert not grid.dynamic
    assert (len(grid.r_b), len(grid.e_b), len(grid.a_b)) == (11, 12, 13)
    grid = SphericalGrid(r_b=[1, 2], e_b=[1, 2, 3], a_b=[1, 2, 3, 4])
    assert grid.shape == (1, 2, 3)
    assert not grid.dynamic
    # check grid boundaries and centers
    def check_bounds(grid):
        assert len(grid.r) == len(grid.r_b) - 1
        assert len(grid.e) == len(grid.e_b) - 1
        assert len(grid.a) == len(grid.a_b) - 1
        assert all(grid.r > grid.r_b[:-1])
        assert all(grid.e > grid.e_b[:-1])
        assert all(grid.a > grid.a_b[:-1])
        assert all(grid.r < grid.r_b[1:])
        assert all(grid.e < grid.e_b[1:])
        assert all(grid.a < grid.a_b[1:])

    check_bounds(grid)
    check_bounds(
        SphericalGrid(
            shape=(10, 11, 12),
            size_r=(1, 10), size_e=(0, tr.pi), size_a=(0, 2*tr.pi),
            spacing='log',

        )
    )

    for x in (grid.r, grid.e, grid.a):
        assert type(x) is tr.Tensor

    assert grid.mesh.ndim == 4, "Invalid mesh dimensions"

def test_sphericalgrid_dynamic():
    grid = SphericalGrid(shape=(9, 10, 11, 12))
    assert grid.dynamic
    assert (len(grid.t), len(grid.r_b), len(grid.e_b), len(grid.a_b)) == (9, 11, 12, 13)
    grid = SphericalGrid(t=[1], r_b=[1, 2], e_b=[1, 2, 3], a_b=[1, 2, 3, 4])
    assert grid.shape == (1, 1, 2, 3)
    assert grid.dynamic

    assert len(grid.nptime) == grid.shape.t, "Incorrect time shape"

    for x in (grid.t, grid.r, grid.e, grid.a):
        assert type(x) is tr.Tensor

    assert grid.mesh.ndim == 5, "Invalid mesh dimensions"

def test_conerectgeom():
    g = ConeRectGeom((11, 11), (4, 0, 1), fov=(23, 45))

    # check fov angles
    assert check(tr.dot(g.rays[5, 0], g.rays[5, -1]), tr.cos(tr.deg2rad(g.fov[1])))
    assert check(tr.dot(g.rays[0, 5], g.rays[-1, 5]), tr.cos(tr.deg2rad(g.fov[0])))
    # check lookdir
    assert check(g.rays[5, 5], g.lookdir)

    # single pixel detector
    g = ConeRectGeom((1, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), fov=(23, 45))
    # check lookdir
    assert check(g.rays[0, 0], g.lookdir)
    # generate wireframe
    g._wireframe


def test_conecircgeom():
    g = ConeCircGeom((11, 11), (1, 0, 0), (-1, 0, 0), (0, 1, 0), fov=(0, 45))

    # check fov angles
    assert check(tr.dot(g.rays[-1, 0], g.rays[-1, 5]), tr.cos(tr.deg2rad(g.fov[1])))
    # check look dir
    assert check(g.rays[0, 0], g.lookdir)

    # single pixel detector
    g = ConeCircGeom((1, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), fov=(0, 45))
    # check lookdir
    assert check(g.rays[0, 0], g.lookdir)
    # generate wireframe
    g._wireframe

def test_parallelgeom():
    g = ParallelGeom((11, 11), (4, 0, 1), size=(2, 3))

    # check ray separation
    assert check(
        tr.linalg.norm(g.ray_starts[5, 0] - g.ray_starts[5, -1]),
        g.size[1]
    )
    assert check(
        tr.linalg.norm(g.ray_starts[0, 5] - g.ray_starts[-1, 5]),
        g.size[0]
    )
    # check lookdir
    assert all((g.rays == g.lookdir).flatten())

    # single pixel detector
    g = ParallelGeom((1, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0))
    # check lookdir
    assert check(g.rays[0, 0], g.lookdir)
    # generate wireframe
    g._wireframe

def test_viewgeom():
    # not much to test here.  just instantiate a ViewGeom with random LOS's
    rays = tr.rand((4, 4, 3))
    ray_starts=tr.tensor((10., 0, 0)).broadcast_to(rays.shape)
    g = ViewGeom(
        rays=rays,
        ray_starts=ray_starts
    )
    # generate wireframe
    g._wireframe

    g = ViewGeom(
        ray_starts=ray_starts,
        ray_ends=tr.rand_like(ray_starts)
    )
    # generate wireframe
    g._wireframe
