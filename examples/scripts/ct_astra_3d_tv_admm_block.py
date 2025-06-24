#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
3D TV-Regularized Sparse-View CT Reconstruction (ADMM Solver)
=============================================================

This example demonstrates solution of a sparse-view, 3D CT
reconstruction problem with isotropic total variation (TV)
regularization

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - C \mathbf{x}
  \|_2^2 + \lambda \| D \mathbf{x} \|_{2,1} \;,$$

where $C$ is the X-ray transform (the CT forward projection operator),
$\mathbf{y}$ is the sinogram, $D$ is a 3D finite difference operator,
and $\mathbf{x}$ is the reconstructed image.

In this example the problem is solved via ADMM, while proximal
ADMM is used in a [companion example](ct_astra_3d_tv_padmm.rst).
"""

import argparse
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
# from scico import numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom
from scico.linop.xray.astra import XRayTransform3D
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description="3D TV-Regularized CT Reconstruction")
parser.add_argument(
    "--extract_individual", 
    action="store_true",
    help="Extract individual Radon matrices for each angle"
)
args = parser.parse_args()


"""
Create a ground truth image and projector.
"""
Nx = 128
Ny = 256
Nz = 64

tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))

n_projection = 10  # number of projections
angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
C = XRayTransform3D(
    tangle.shape, det_count=[Nz, max(Nx, Ny)], det_spacing=[1.0, 1.0], angles=angles
)  # CT projection operator
y = C @ tangle  # sinogram





print("C is: ", C)
print("Shape of y is: ", y.shape)




"""
Extract individual Radon transform matrices for each angle (optional).
"""
if args.extract_individual:
    print("\n=== Extracting Individual Radon Transform Matrices ===")
    
    # Method 1: Create individual operators for each angle
    individual_operators = []
    for i, angle in enumerate(angles):
        single_angle_op = XRayTransform3D(
            tangle.shape, 
            det_count=[Nz, max(Nx, Ny)], 
            det_spacing=[1.0, 1.0], 
            angles=np.array([angle]),
        )
        individual_operators.append(single_angle_op)
        print(f"Created individual Radon transform for angle {i}: {angle:.3f} rad ({angle*180/np.pi:.1f}°)")
    
    # Method 2: Extract individual projections from full sinogram
    print("\n=== Extracting Individual Projections from Sinogram ===")
    individual_projections = {}
    for i in range(n_projection):
        # Extract projection for angle i from full sinogram
        projection = y[:, i, :]  # Shape: (Nz, det_count_y)
        individual_projections[i] = {
            'angle': angles[i],
            'angle_degrees': angles[i] * 180 / np.pi,
            'projection': projection,
            'shape': projection.shape
        }
        print(f"Extracted projection for angle {i}: {angles[i]*180/np.pi:.1f}° -> shape {projection.shape}")
    
    # Verify that individual projections match
    print("\n=== Verifying Individual Projections ===")
    for i in range(n_projection):
        # Get projection from individual operator
        single_proj = individual_operators[i] @ tangle  # Shape: (Nz, 1, det_count_y)
        single_proj_2d = single_proj[:, 0, :]  # Remove angle dimension
        
        # Get projection from full sinogram
        full_proj = individual_projections[i]['projection']
        
        # Compare
        diff = np.abs(single_proj_2d - full_proj).max()
        print(f"Angle {i}: max difference = {diff:.2e}")
    
    print("\nIndividual Radon matrices extracted successfully!")

"""
Set up problem and solver.
"""
λ = 2e0  # ℓ2,1 norm regularization parameter
ρ = 5e0  # ADMM penalty parameter
maxiter = 25  # number of ADMM iterations
cg_tol = 1e-4  # CG relative tolerance
cg_maxiter = 25  # maximum CG iterations per ADMM iteration

# The append=0 option makes the results of horizontal and vertical
# finite differences the same shape, which is required for the L21Norm,
# which is used so that g(Ax) corresponds to isotropic TV.
D = linop.FiniteDifference(input_shape=tangle.shape, append=0)
g = λ * functional.L21Norm()
f = loss.SquaredL2Loss(y=y, A=C)

"""
Check memory usage of C and D objects.
"""
import sys
import psutil
import gc

def get_object_memory_usage(obj, obj_name):
    """Get memory usage of an object in bytes."""
    # Force garbage collection to get accurate measurement
    gc.collect()
    
    # Get memory before creating reference
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    # Create a reference to the object
    obj_ref = obj
    
    # Get memory after
    mem_after = process.memory_info().rss
    mem_used = mem_after - mem_before
    
    # Get object size using sys.getsizeof (approximate)
    obj_size = sys.getsizeof(obj)
    
    print(f"\n=== Memory Usage for {obj_name} ===")
    print(f"Object type: {type(obj)}")
    print(f"sys.getsizeof: {obj_size:,} bytes ({obj_size/1024/1024:.2f} MB)")
    print(f"Process memory change: {mem_used:,} bytes ({mem_used/1024/1024:.2f} MB)")
    
    # Try to get more detailed info if available
    if hasattr(obj, 'input_shape'):
        print(f"Input shape: {obj.input_shape}")
    if hasattr(obj, 'output_shape'):
        print(f"Output shape: {obj.output_shape}")
    if hasattr(obj, 'input_dtype'):
        print(f"Input dtype: {obj.input_dtype}")
    if hasattr(obj, 'output_dtype'):
        print(f"Output dtype: {obj.output_dtype}")
    
    return obj_size, mem_used

# Check memory usage of C (XRayTransform3D)
print("\n" + "="*50)
print("MEMORY USAGE ANALYSIS")
print("="*50)

c_size, c_mem = get_object_memory_usage(C, "C (XRayTransform3D)")

# Check memory usage of D (FiniteDifference)
d_size, d_mem = get_object_memory_usage(D, "D (FiniteDifference)")

# Check memory usage of other objects
y_size, y_mem = get_object_memory_usage(y, "y (sinogram)")
tangle_size, tangle_mem = get_object_memory_usage(tangle, "tangle (ground truth)")

# Summary
print(f"\n=== Memory Usage Summary ===")
print(f"C (XRayTransform3D): {c_size:,} bytes ({c_size/1024/1024:.2f} MB)")
print(f"D (FiniteDifference): {d_size:,} bytes ({d_size/1024/1024:.2f} MB)")
print(f"y (sinogram): {y_size:,} bytes ({y_size/1024/1024:.2f} MB)")
print(f"tangle (ground truth): {tangle_size:,} bytes ({tangle_size/1024/1024:.2f} MB)")

total_size = c_size + d_size + y_size + tangle_size
print(f"Total object memory: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")


print("="*50)





x_initial = C.T(y)
print("x_initial shape: ", x_initial.shape)
print("Type of x_initial: ", type(x_initial))
print("dtype of x_initial: ", x_initial.dtype)


solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[D],
    rho_list=[ρ],
    x0=C.T(y),
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 5},
)


"""
Run the solver.
"""

print("CUDA availablility: ", torch.cuda.is_available())



print(f"Solving on {device_info()}\n")
tangle_recon = solver.solve()
hist = solver.itstat_object.history(transpose=True)

print(
    "TV Restruction\nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(tangle, tangle_recon), metric.mae(tangle, tangle_recon))
)


"""
Show the recovered image.
"""
fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
plot.imview(
    tangle[32],
    title="Ground truth (central slice)", 
    cmap=plot.cm.Blues,
    cbar=None,
    fig=fig,
    ax=ax[0],
)
plot.imview(
    tangle_recon[32],
    title="TV Reconstruction (central slice)\nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(tangle, tangle_recon), metric.mae(tangle, tangle_recon)),
    cmap=plot.cm.Blues,
    fig=fig,
    ax=ax[1],
)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.2)
fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
fig.show()


# Save the figure
import os
my_path = os.path.dirname(os.path.abspath(__file__))  # Figures out the absolute path for you in case your working directory moves around.
plot.savefig(os.path.join(my_path, "results/ct_astra_3d_tv_admm.png"))

input("\nWaiting for input to close figures and exit")
