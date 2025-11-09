---
title: 'TomoSphero: Fast Differentiable Tomographic Projector in Spherical Coordinates'
tags:
  - Python
  - astronomy
  - tomography
  - PyTorch
  - autograd
  - inverse problems
  - differentiable rendering
authors:
  - name: Evan Widloski
    orcid: 0000-0001-8549-991X
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: University of Illinois Urbana-Champaign
   index: 1
   ror: 047426m28
date: 5 November 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/TBD
aas-journal: Astrophysical Journal
---

# Summary

Computational tomography is a tool for determining the internal structure of objects from a set of projections, typically taken along some regular path (e.g. circular, helical). In recent years, methods and GPU-accelerated libraries have emerged that allow for fast reconstruction from projections along more complicated paths. Most of these libraries rely on a Cartesian discretization of the object.  In planetary and solar tomography, projections are taken along an irregular spacecraft orbit of spherical bodies not well-suited to Cartesian grids.

# Statement of need

We present `TomoSphero`, a differentiable tomographic projector over spherical grids which are often used in planetary and solar tomography. `TomoSphero` is designed to be used as a building block in reconstruction algorithms and includes common projection types such as cone-beam and parallel-beam, but is flexible enough to accommodate arbitrary projections. `TomoSphero` is implemented in PyTorch, which allows for fast projection computation on GPUs, easy integration into machine learning algorithms, and automatic differentiation for reconstruction algorithms which require access to gradients.

# Tomographic Inversion

Tomography is a method for determining the internal structure of objects from a set of measurements that penetrate into the object being measured.  These measurements (sometimes called projections or sinograms) are usually captured from a variety of locations and times which are collectively referred to as the _view geometry_. Measurements are typically modeled as

$$y = F x + \epsilon$$

where $y$ is a collection of measurements,  $F$ is a linear projection operator, $x$ is the object under study, and $\epsilon$ is noise.
])

Tomography has found application in a vast number of domains such as medical imaging, crystallography, and remote sensing, utilizing modalities like X-ray, ultraviolet (UV), ultrasound, seismic waves, and many more.  In this paper we discuss TomoSphero, a Python library for planetary and solar tomography.

Fast tomographic reconstruction algorithms that implement explicit inversion formulas typically work only for specific view geometries (such as circular or helical view geometry) and are referred to as _filtered back projection_ (FBP) algorithms [@fbp].  However, some situations (like an orbiting spacecraft) necessitate more complicated measurement paths than are allowed by FBP-type algorithms.  For these situations requiring more flexible view geometries where an exact inverse solution is not available, _iterative reconstruction_ (IR) algorithms prevail, usually solving an optimization problem of the form

$$hat(bold(rho)) = arg min_bold(rho) ||bold(y) - F bold(rho)||_2^2 + ...$$

Examples include SIRT [@sirt], TV-MIN [@tvmin], ART [@art], CGLS [@cgls], Plug-and-play [@plugandplay] and many others.
  These algorithms obtain synthetic projections of a candidate object using a tomographic operator (sometimes called a _raytracer_) that simulates waves traveling through the object medium.  They produce a reconstruction by repeatedly tweaking the candidate object to minimize discrepancy between synthetic and actual projections, and they stand to benefit the most from a fast operator implementation. 
  
  TomoSphero is parallelized and GPU-enabled, and its speed has been benchmarked as described in the companion paper.
  In cases where a simultaneous computation for every pixel of every measurement would consume more memory than is available, some algorithms operate _out-of-core_, where they parallelize as many tasks as will fit into available memory, then serially queue the remaining tasks for processing after current tasks are complete.  TomoSphero is not capable of out-of-core operation.

  Another consideration in tomographic reconstruction is the choice of grid type for discretization of the reconstructed object.  Most publications consider a regular rectilinear grid, which is a reasonable choice when the underlying structure of the object is completely unknown or the scale of features is uniform throughout the object.  The primary focus of TomoSphero is in the domain of atmospheric tomography, where regular spherical grids are well-suited for modeling solar and planetary atmospheres that exhibit spherical symmetries [@solartomography1] [@solartomography2].

  Many reconstruction algorithms
rely on gradient-based optimization to solve for an object whose structure corresponds to measurement data.
Automatic differentiation (_autograd_) is a class of techniques that convert an arbitrary expression into a computational graph of simpler functions, then compute the overall derivative by applying chain rule at each node.  Modern machine learning libraries such as PyTorch [@pytorch] and Jax [@jax] provide such capabilities for building this computational graph.  TomoSphero is implemented on top of PyTorch and its autograd capabilities enable rapid prototyping of different parametric models and regularizations.

TomoSphero development was motivated by the [Carruthers Geocorona Observatory](https://science.nasa.gov/mission/carruthers-geocorona-observatory/), a spacecraft containing UV imagers which will survey the Earth's exosphere.

A non-exhaustive comparison of TomoSphero's capabilities against other popular libraries is shown below:

| Name                 | Grid Type | GPU Support | Autograd | Visualization | Out-of-Core |
|----------------------|-----------|-------------|----------|---------------|-------------|
| TIGRE [@tigre]       | Cartesian | Yes         | No       | No            | Yes         |
| LEAP [@leap]         | Cartesian | Yes         | Yes      | No            | Yes         |
| ASTRA [@astra1]      | Cartesian | Yes         | Yes      | No            | Yes         |
| mbirjax [@mbirjax]   | Cartesian | Yes         | No       | No            | Yes         |
| ToMoBAR [@tomobar]   | Cartesian | yes         | No       | No            | Yes         |
| CIL [@cil]           | Cartesian | Yes         | No       | Yes           | Yes         |
| Tomosipo [@tomosipo] | Cartesian | Yes         | Yes      | Yes           | Yes         |
|                      |           |             |          |               |             |