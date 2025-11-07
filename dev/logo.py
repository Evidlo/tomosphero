import svg3d
from tomosphero import SphericalGrid
from tomosphero.plotting import sph2cart
import numpy as np

grid = SphericalGrid(
    (2, 15, 20),
    # (1, 1, 1),
    size_r=(.25, 1),
    size_a=(-np.pi, np.pi* (6/10))
)

orig_vertices = sph2cart(grid.mesh_b)

outer_vertices = orig_vertices[-1, :, :]
inner1_vertices = orig_vertices[:, :, 0]
inner2_vertices = orig_vertices[:, :, -1]

def grid2faces(shape):
    """Generate list of faces for a rectangular 2D grid

    Args:
        shape (tuple[int]): shape of 2D grid

    Returns:
        list[list[int]]

    Derivation (2x5 grid):

    ·  ·  ·  ·  ·
     0  1  2  3

    ·  ·  ·  ·  ·
     6  7  8  9
    ------------

    ·  ·  ·  ·  ·
     01 12 23 34
     67 78 89 9X
    ·  ·  ·  ·  ·
    """
    r, c = shape

    x = np.arange(r * c).reshape(r, c)[:-1, :-1]
    # print(x)
    x = x.flatten()
    # print(x)
    x = np.stack((x, x + 1, x + c + 1, x + c)).T
    # print(x)
    return x

vertices = np.vstack([
    o:=outer_vertices.reshape((-1, 3)),
    i:=inner1_vertices.reshape((-1, 3)),
    inner2_vertices.reshape((-1, 3)),
])

faces = np.vstack([
    grid2faces(outer_vertices.shape[:-1]),
    grid2faces(inner1_vertices.shape[:-1]) + len(o),
    grid2faces(inner2_vertices.shape[:-1]) + len(o) + len(i),
])


# Set up our rendering style - transparent white gives a nice wireframe appearance
style = {
    "fill": "#FFFFFF",
    "fill_opacity": "0.95",
    "stroke": "black",
    "stroke_linejoin": "round",
    "stroke_width": "0.005",
}

empty_shader=svg3d.shaders.DiffuseShader.from_style_dict(style)

pos_object = [0.0, 0.0, 0.0]  # "at" position
pos_camera = [-40, 40, 20]  # "eye" position
vec_up = [0.0, 0.0, 1.0]  # "up" vector of camera. This is the default value.

z_near, z_far = 1.0, 200.0
aspect = 1.0  # Aspect ratio of the view cone
fov_y = 2.0  # Opening angle of the view cone. fov_x is equal to fov_y * aspect

look_at = svg3d.get_lookat_matrix(pos_object, pos_camera, vec_up=vec_up)
projection = svg3d.get_projection_matrix(
    z_near=z_near, z_far=z_far, fov_y=fov_y, aspect=aspect
)

# A "scene" is a list of Mesh objects, which can be easily generated from raw data
scene = [
    svg3d.Mesh.from_vertices_and_faces(vertices, faces, shader=empty_shader)
]

view = svg3d.View.from_look_at_and_projection(
    look_at=look_at,
    projection=projection,
    scene=scene,
)

svg3d.Engine([view]).render("wireframe.svg")