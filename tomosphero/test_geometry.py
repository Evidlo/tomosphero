#!/usr/bin/env python3

import numpy as np
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

def test_sphericalgrid_resample():
    # static: rebin, then reclip
    grid = SphericalGrid(
        shape=(10, 11, 12), size_r=(3, 25), spacing='log',
    )
    g = grid.resample(shape=(20, 11, 12))
    assert g.shape == (20, 11, 12) and not g.dynamic
    assert g.size == grid.size and g.spacing == grid.spacing
    g = grid.resample(size_r=(3, 15))
    assert g.shape == grid.shape and g.size.r == (3, 15)

    # dynamic: a spatial-only resample keeps the sample times
    dates = np.arange('2026-03-01', '2026-03-05', dtype='datetime64[h]')
    grid = SphericalGrid(
        t=dates, timeunit='h',
        r_b=tr.linspace(3, 25, 11),
        e_b=tr.linspace(0, tr.pi, 12),
        a_b=tr.linspace(-tr.pi, tr.pi, 13),
    )
    g = grid.resample(shape=(len(dates), 20, 11, 12))
    assert g.shape == (len(dates), 20, 11, 12)
    assert all(g.t == grid.t), "sample times not carried over"
    assert g.timeunit == grid.timeunit

    # a new time bin count drops them, respacing over the same extent
    g = grid.resample(shape=(5, 10, 11, 12))
    assert g.shape == (5, 10, 11, 12)
    assert g.size.t == grid.size.t
    assert g.t[0] == grid.t[0] and g.t[-1] == grid.t[-1]

    # explicit times set the time axis and its length
    g = grid.resample(t=dates[::2])
    assert g.shape == (len(dates) // 2, 10, 11, 12)
    assert all(g.t == tr.asarray(dates[::2].astype('float64')))

    # an explicit t wins over shape's time bin count
    g = grid.resample(t=dates[::2], shape=(99, 5, 6, 7))
    assert g.shape == (len(dates) // 2, 5, 6, 7)

    # times can be added to a static grid, and dropped from a dynamic one
    assert SphericalGrid(
        shape=(10, 11, 12),
    ).resample(t=dates).shape == (len(dates), 10, 11, 12)
    assert not grid.static.resample(shape=(10, 11, 12)).dynamic

    # explicit boundaries supersede shape
    g = grid.resample(r_b=[1, 2, 3], e_b=[1, 2, 3], a_b=[1, 2, 3, 4])
    assert g.shape == (len(dates), 2, 2, 3)

    # datetime samples on the shape path set size_t, so a temporal resample spans them
    grid = SphericalGrid(shape=(len(dates), 10, 11, 12), t=dates, timeunit='h')
    assert grid.size.t == (float(grid.t[0]), float(grid.t[-1]))
    g = grid.resample(shape=(2, 10, 11, 12))
    assert (g.t[0], g.t[-1]) == (grid.t[0], grid.t[-1])


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
