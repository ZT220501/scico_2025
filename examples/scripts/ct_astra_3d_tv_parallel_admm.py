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


"""
Create a ground truth image and projector.
"""
Nx = 128
Ny = 256
Nz = 64

tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))

n_projection = 20  # number of projections
angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles

'''
Instead of creating a single XRayTransform3D operator, we create one for each angle.
This is more memory efficient and allows for parallel computation.
'''
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

print("Input shape of single operator is: ", individual_operators[0].input_shape)
print("Output shape of single operator is: ", individual_operators[0].output_shape)


y_individuals = []
for i in range(n_projection):
    y_individuals.append(individual_operators[i] @ tangle)
    # print(f"Sinogram for angle {i} is: ", y_individuals[i])
    print(f"Shape of sinogram for angle {i} is: ", y_individuals[i].shape)

# """
# Extract individual Radon transform matrices for each angle (optional).
# """
# if args.extract_individual:
#     print("\n=== Extracting Individual Radon Transform Matrices ===")
    
#     # Method 1: Create individual operators for each angle
#     individual_operators = []
#     for i, angle in enumerate(angles):
#         single_angle_op = XRayTransform3D(
#             tangle.shape, 
#             det_count=[Nz, max(Nx, Ny)], 
#             det_spacing=[1.0, 1.0], 
#             angles=np.array([angle]),
#         )
#         individual_operators.append(single_angle_op)
#         print(f"Created individual Radon transform for angle {i}: {angle:.3f} rad ({angle*180/np.pi:.1f}°)")
    
#     # Method 2: Extract individual projections from full sinogram
#     print("\n=== Extracting Individual Projections from Sinogram ===")
#     individual_projections = {}
#     for i in range(n_projection):
#         # Extract projection for angle i from full sinogram
#         projection = y[:, i, :]  # Shape: (Nz, det_count_y)
#         individual_projections[i] = {
#             'angle': angles[i],
#             'angle_degrees': angles[i] * 180 / np.pi,
#             'projection': projection,
#             'shape': projection.shape
#         }
#         print(f"Extracted projection for angle {i}: {angles[i]*180/np.pi:.1f}° -> shape {projection.shape}")
    
#     # Verify that individual projections match
#     print("\n=== Verifying Individual Projections ===")
#     for i in range(n_projection):
#         # Get projection from individual operator
#         single_proj = individual_operators[i] @ tangle  # Shape: (Nz, 1, det_count_y)
#         single_proj_2d = single_proj[:, 0, :]  # Remove angle dimension
        
#         # Get projection from full sinogram
#         full_proj = individual_projections[i]['projection']
        
#         # Compare
#         diff = np.abs(single_proj_2d - full_proj).max()
#         print(f"Angle {i}: max difference = {diff:.2e}")
    
#     print("\nIndividual Radon matrices extracted successfully!")

"""
Set up problem and solver.
"""
λ = 2e0  # ℓ2,1 norm regularization parameter
ρ = 5e0  # ADMM penalty parameter
maxiter = 200  # number of ADMM iterations
cg_tol = 1e-4  # CG relative tolerance
cg_maxiter = 25  # maximum CG iterations per ADMM iteration

# The append=0 option makes the results of horizontal and vertical
# finite differences the same shape, which is required for the L21Norm,
# which is used so that g(Ax) corresponds to isotropic TV.
D = linop.FiniteDifference(input_shape=tangle.shape, append=0)
g = λ * functional.L21Norm()
f = loss.SquaredL2Loss(y=y, A=C)

solver = ADMM(
    f=f,
    g_list=[g],
    C_list=[D],
    rho_list=[ρ],
    x0=C.T(y),
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 25},
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

input("\nWaiting for input to close figures and exit")
