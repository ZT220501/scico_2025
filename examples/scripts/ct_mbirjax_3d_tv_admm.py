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

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

import os




def full_recon_test():
    """
    Create a ground truth image and projector.
    """
    Nx = 128
    Ny = 256
    Nz = 64

    tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))

    n_projection = 10  # number of projections
    angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
    C = XRayTransformParallel(
        output_shape=(Nz, n_projection, max(Nx, Ny)), 
        angles=angles,
        recon_shape=(Nx, Ny, Nz)
    )  # CT projection operator
    y = C @ tangle  # sinogram

    print(f"Sinogram shape: {y.shape}")

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
    print(f"Solving on {device_info()}\n")
    tangle_recon = solver.solve()
    hist = solver.itstat_object.history(transpose=True)

    print(
        "MBIRJAX TV Restruction\nSNR: %.2f (dB), MAE: %.3f"
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

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'ct_mbirjax_3d_tv_admm.png')
    fig.savefig(save_path)   # save the figure to file

def partial_recon_test():
    """
    Create a ground truth image and projector.
    """
    Nx = 128
    Ny = 256
    Nz = 64

    tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))

    n_projection = 10  # number of projections
    angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles

    # Define ROI (Region of Interest)
    roi_start_row, roi_end_row = 20, 100  # Selected rows
    roi_start_col, roi_end_col = 10, 150  # Selected columns
    
    # Create ROI indices
    roi_rows = np.arange(roi_start_row, roi_end_row)
    roi_cols = np.arange(roi_start_col, roi_end_col)
    roi_row_grid, roi_col_grid = np.meshgrid(roi_rows, roi_cols, indexing='ij')
    roi_indices = np.ravel_multi_index((roi_row_grid.flatten(), roi_col_grid.flatten()), (Nx, Ny))
    
    print(f"ROI indices shape: {roi_indices.shape}")
    print(f"ROI coverage: {len(roi_indices)} / {Nx * Ny} = {len(roi_indices) / (Nx * Ny) * 100:.1f}%")
    
    # Create sinogram shape
    sinogram_shape = (Nz, n_projection, max(Nx, Ny))
    
    # Method 1: Direct ROI projection using XRayTransformParallel with partial reconstruction
    print("\n1. Creating direct ROI projection using partial reconstruction...")
    C = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        partial_reconstruction=True,
        roi_indices=roi_indices,
        roi_recon_shape=(len(roi_rows), len(roi_cols), Nz),
        recon_shape=(Nx, Ny, Nz)
    )

    tangle_roi = tangle[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row]
    y = C @ tangle_roi  # sinogram

    print(f"Sinogram shape: {y.shape}")

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
    D = linop.FiniteDifference(input_shape=tangle_roi.shape, append=0)
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
        itstat_options={"display": True, "period": 5},
    )


    """
    Run the solver.
    """
    print(f"Solving on {device_info()}\n")
    tangle_recon_roi = solver.solve()
    hist = solver.itstat_object.history(transpose=True)

    print(
        "MBIRJAX TV Restruction (ROI)\nSNR: %.2f (dB), MAE: %.3f"
        % (metric.snr(tangle_roi, tangle_recon_roi), metric.mae(tangle_roi, tangle_recon_roi))
    )


    """
    Show the recovered image.
    """
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    test_slice = 32
    plot.imview(
        tangle_roi[test_slice],
        title="Ground truth (central slice)",
        cmap=plot.cm.Blues,
        cbar=None,
        fig=fig,
        ax=ax[0],
    )
    plot.imview(
        tangle_recon_roi[test_slice],
        title="TV Reconstruction (central slice)\nSNR: %.2f (dB), MAE: %.3f"
        % (metric.snr(tangle_roi, tangle_recon_roi), metric.mae(tangle_roi, tangle_recon_roi)),
        cmap=plot.cm.Blues,
        fig=fig,
        ax=ax[1],
    )
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
    fig.show()

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'ct_mbirjax_3d_tv_admm_roi.png')
    fig.savefig(save_path)   # save the figure to file







if __name__ == "__main__":
    print("Starting XRayTransformParallel ADMM 3D TV tests...")

    # print("\n" + "="*80)
    # print("TEST 1: Full reconstruction forward projection test")
    # print("="*80)
    # test_results_1 = full_recon_test()
    
    # print("\n✅ Test 1 completed successfully!")



    print("\n" + "="*80)
    print("TEST 2: Partial reconstruction forward projection test")
    print("="*80)
    test_results_2 = partial_recon_test()
    
    print("\n✅ Test 2 completed successfully!")
    


    input("\nWaiting for input to close figures and exit")

    
    