#!/usr/bin/env python3

import numpy as np
from numpy import pi
import torch as t
from tomosphero import SphericalGrid, Operator
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')

# %% grid

plt.close('all')
fig, ax = plt.subplots(figsize=(2, 2), subplot_kw={'projection': '3d'})
grid = SphericalGrid(
    shape=(30, 30, 30),
    size_r=(0, 1)
)
grid.plot(ax)
ax.set_aspect('equal')
plt.tick_params(axis='x', which='major', pad=-2)
plt.tick_params(axis='y', which='major', pad=-2)
plt.tick_params(axis='z', which='major', pad=-2)
ax.set_xlabel('X', labelpad=-5)
ax.set_ylabel('Y', labelpad=-5)
ax.set_zlabel('Z', labelpad=-8)
plt.subplots_adjust(bottom=.15, right=.85)
plt.savefig('api_grid.png')

# %% wedge

plt.close('all')
fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection':'3d'})
grid = SphericalGrid(
    # voxels (radius, elevation, azimuth)
    shape=(10, 10, 10),
    size_r=(5, 10),  # inner/outer grid radius
    size_e=(0.4 * pi, 0.6 * pi), # (radians)
    size_a=(-.1 * pi,  .1 * pi), # (radians)
)
grid.plot(ax)
ax.set_aspect('equal')
plt.tick_params(axis='x', which='major', pad=-2)
plt.tick_params(axis='y', which='major', pad=-2)
plt.tick_params(axis='z', which='major', pad=-2)
ax.set_xlabel('X', labelpad=-5)
ax.set_ylabel('Y', labelpad=-5)
ax.set_zlabel('Z', labelpad=-8)
plt.subplots_adjust(bottom=.15, right=.85)
plt.savefig('api_grid_wedge.png')

plt.close('all')
fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection':'3d'})
grid = SphericalGrid(
    r_b=[8, 9, 10],
    e_b=[0.4*pi, 0.5*pi, 0.6*pi],
    a_b=[-0.1*pi, 0.0*pi, 0.1*pi],
)
grid.plot(ax)
ax.set_aspect('equal')
plt.tick_params(axis='x', which='major', pad=-2)
plt.tick_params(axis='y', which='major', pad=-2)
plt.tick_params(axis='z', which='major', pad=-2)
ax.set_xlabel('X', labelpad=-5)
ax.set_ylabel('Y', labelpad=-5)
ax.set_zlabel('Z', labelpad=-8)
plt.subplots_adjust(bottom=.15, right=.85)
plt.savefig('api_grid_boundary.png')


from tomosphero import ConeRectGeom
fig, ax = plt.subplots(figsize=(2, 2), subplot_kw={'projection': '3d'})
geom = None
# sweep 360° around origin
for w in np.linspace(0, 2*np.pi, 50):
    w += np.pi / 2
    # sum primitive geometries
    geom += ConeRectGeom(
        shape=(64, 64),
        pos=(5 * np.cos(w), 5 * np.sin(w), 2),
        fov=(25, 25)
    )
anim = geom.plot(ax)
ax.set_aspect('equal')
plt.tick_params(axis='x', which='major', pad=-2)
plt.tick_params(axis='y', which='major', pad=-2)
plt.tick_params(axis='z', which='major', pad=-2)
ax.set_xlabel('X', labelpad=-5)
ax.set_ylabel('Y', labelpad=-5)
ax.set_zlabel('Z', labelpad=-8)
plt.subplots_adjust(bottom=.15, right=.85)
anim.save('api_collection.gif')

# %% obj

plt.close()
fig, ax = plt.subplots(figsize=(2, 2))
grid = SphericalGrid((30, 30, 30), size_r=(0, 1))
op = Operator(grid, geom)
x = t.zeros(grid.shape)
# x[12:15, 12:18, 0:26] = 1
x[:, :, 3:] = 1
y = op(x)
from tomosphero.plotting import image_stack, preview3d
ax.set_aspect('equal')
fig.tight_layout()
anim = image_stack(y, geom, ax=ax)
anim.save('api_meas.gif', fps=15)

# %% recon

x_hat = t.zeros(grid.shape, requires_grad=True)
optim = t.optim.Adam([x_hat], lr=1e-1)

for i in (bar:=tqdm(range(100))):
    optim.zero_grad()
    loss = t.sum((y - op(x_hat))**2)
    loss.backward()
    bar.set_description(str(loss))
    optim.step()

# %% foo

plt.close()
fig, ax = plt.subplots(figsize=(2, 2))
anim = image_stack(preview3d(x, grid), ax=ax)
plt.title('Ground Truth')
anim.save('truth.gif')

plt.close()
fig, ax = plt.subplots(figsize=(2, 2))
anim = image_stack(preview3d(x_hat, grid), ax=ax)
plt.title('Reconstruction')
anim.save('recon.gif')