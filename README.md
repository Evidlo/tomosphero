<img src="logo.svg" width="200px"/>

# TomoSphero

A differentiable 3D/4D volumetric tomographic projector in spherical coordinates.

Check the [tutorial](https://evidlo.github.io/tomosphero/tomosphero.html#tutorial) for instruction on using this library or [examples](https://github.com/evidlo/tomosphero/tree/master/examples) for complete samples demonstrating forward raytracing and retrieval.

## Features

TomoSphero was originally created for tomographic retrieval of planetary atmospheres, but is designed to work for any problem defined on a spherical grid.  Some of its features are

- 3D spherical raytracing with optional support for dynamic volume (4D)
- implemented purely in PyTorch for easy integration with PyTorch's optimization and machine learning capabilities
- support for square/circular detectors or other custom detector shapes
- retrieval framework for easily defining loss functions and parametric models (currently supports only static 3D volumes)

Note that TomoSphero is a *volumetric* raytracer (i.e. not occlusions, shading, etc.).

## Installation and Quickstart

    pip install tomosphero
    git clone https://github.com/evidlo/tomosphero && cd tomosphero
    python examples/single_vantage.py

<img src="example.png" height=250/>

    python examples/static_retrieval.py

<p>
<img src="static_retrieval2.gif" height=200/>
<img src="static_retrieval1.gif" height=200/>
</p>

    python examples/dynamic_measurements.py
    
<img src="dynamic.gif" height=250/>

## Memory Usage

This library uses only PyTorch array operations for implementation simplicity and speed at the expense of memory consumption.  The peak memory usage in GB can be approximated with `examples/memory_usage.py`

``` bash
$ python examples/memory_usage.py

--- Parameters ---

(50, 50, 50) object
50 observations, 1 channels, (50, 100) sensor

--- Memory Usage ---

Ray coordinates memory: 4.25 GB
Object memory: 0.05 GB
```

## Architecture

Below is a list of modules in this package and their purpose:

Forward Raytracing

- `raytracer.py` - computation of voxel indices for intersecting rays, raytracing Operator
- `geometry.py` - viewing geometry (detector) definitions, spherical grid definition

Retrieval

- `model.py` - parameterized models for representing an object.  used in retrieval
- `loss.py` - some loss functions to be used in retrieval
- `retrieval.py` - retrieval algorithms
- `plotting.py` - functions for plotting stacks of images, retrieval losses

## Running Tests

    pytest tomosphero
    
## See Also

[tomosipo](https://github.com/ahendriksen/tomosipo), which inspired parts of this library's interface.
