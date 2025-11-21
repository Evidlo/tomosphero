#!/usr/bin/env python3
# Evan Widloski - 2024-04-01
# Demonstration of forward raytracer operator from a single vantage point

import torch as t
import matplotlib.pyplot as plt

from tomosphero import SphericalGrid, ConeRectGeom, Operator

device = "cuda:0" if t.cuda.is_available() else "cpu"

# define spherical grid and viewing geometry vantage
grid = SphericalGrid(shape=(50, 50, 50))
# rectilinear detector with 45° FOV (pointed at origin by default)
geom = ConeRectGeom(
    shape=(256, 256),
    pos=(5, 0, 0),
    fov=(45, 45)
)

# define forward operator
op = Operator(grid, geom, device=device)

# generate a simple static test object with two nested shells
x = t.zeros(grid.shape, device=device)
x[-1, :, :] += 1
x[-10, :, :] += 1

# raytrace over the object
result = op(x)

# ----- Plotting -----

plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.set_title('Nested Shells')
ax1.imshow(result.detach().cpu())

ax2.set_title('View Geometry')
# operator has a .plot() method which returns a matplotlib animation
ani = op.plot(ax2)

# plt.show()
