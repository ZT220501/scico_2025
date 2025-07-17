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
ADMM is used in a [companion example](ct_mbirjax_3d_tv_padmm.rst).
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
import scipy.io
import argparse

def create_roi_indices(Nx, Ny, roi_start_row, roi_end_row, roi_start_col, roi_end_col, display=False):
    # Check the validity of the ROI indices
    if roi_start_row < 0 or roi_end_row > Nx or roi_start_col < 0 or roi_end_col > Ny:
        raise ValueError("ROI start and end indices must be within the image dimensions")
    if roi_start_row >= roi_end_row or roi_start_col >= roi_end_col:
        raise ValueError("ROI start indices must be strictly less than end indices")

    # Create ROI indices
    roi_rows = np.arange(roi_start_row, roi_end_row)
    roi_cols = np.arange(roi_start_col, roi_end_col)
    roi_row_grid, roi_col_grid = np.meshgrid(roi_rows, roi_cols, indexing='ij')
    roi_indices = np.ravel_multi_index((roi_row_grid.flatten(), roi_col_grid.flatten()), (Nx, Ny))
    
    if display:
        print(f"ROI indices shape: {roi_indices.shape}")
        print(f"ROI coverage: {len(roi_indices)} / {Nx * Ny} = {len(roi_indices) / (Nx * Ny) * 100:.1f}%")
    
    return roi_indices

'''
Test for reconstruction for full 3D CT image, using MBIRJAX and ADMM.
'''
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

'''
Test for reconstruction for a region of interest (ROI) 3D CT image, using MBIRJAX and ADMM.
'''
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


'''
Test for reconstruction for full 3D CT image with 
naive way of dividing the image into blocks and reconstructing each block separately.
ADMM and MBIRJAX are used to reconstruct each block.
'''
def simple_block_admm_test(Nx=128, Ny=256, Nz=64, row_division_num=4, col_division_num=8, do_block_recon=True):
    '''
    Create a ground truth image and projector.
    
    Args:
        do_block_recon: Whether to perform block reconstruction
        Nx: Image width
        Ny: Image height  
        Nz: Image depth
        row_division_num: Number of row divisions
        col_division_num: Number of column divisions
    '''
    # Create a full 3D CT image phantom
    tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))

    n_projection = 10  # number of projections
    angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles

    # Define each ROI (Region of Interest) and its corresponding mbirjax projector
    C_list = [] # List of mbirjax projectors for each ROI
    y_list = [] # List of sinograms for each ROI
    tangle_roi_list = [] # List of ground truth ROI images for each ROI

    if do_block_recon:
        for i in range(row_division_num):
            for j in range(col_division_num):
                roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
                roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
                roi_indices = create_roi_indices(Nx, Ny, roi_start_row, roi_end_row, roi_start_col, roi_end_col)

                # Create sinogram shape
                sinogram_shape = (Nz, n_projection, max(Nx, Ny))

                C = XRayTransformParallel(
                    output_shape=sinogram_shape,
                    angles=angles,
                    partial_reconstruction=True,
                    roi_indices=roi_indices,
                    roi_recon_shape=(roi_end_row - roi_start_row, roi_end_col - roi_start_col, Nz),
                    recon_shape=(Nx, Ny, Nz)
                )

                tangle_roi = tangle[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row]
                y = C @ tangle_roi  # sinogram
                print(f"ROI shape: {tangle_roi.shape}; sinogram shape: {y.shape}; current number of ROI: {len(C_list)}")

                # Append the mbirjax projector and sinogram to the list
                C_list.append(C)
                y_list.append(y)
                tangle_roi_list.append(tangle_roi)

        """
        Set up problems and solvers.
        """
        print("I can reach here.")
        # Generic parameters for all sub-block solvers
        λ = 2e0  # ℓ2,1 norm regularization parameter
        ρ = 5e0  # ADMM penalty parameter
        maxiter = 25  # number of ADMM iterations
        cg_tol = 1e-4  # CG relative tolerance
        cg_maxiter = 25  # maximum CG iterations per ADMM iteration
        g = λ * functional.L21Norm()

        # Specific parameters for each sub-block solver
        # The append=0 option makes the results of horizontal and vertical
        # finite differences the same shape, which is required for the L21Norm,
        # which is used so that g(Ax) corresponds to isotropic TV.
        for i in range(len(C_list)):
            C = C_list[i]
            y = y_list[i]
            D = linop.FiniteDifference(input_shape=C.input_shape, append=0)
            f = loss.SquaredL2Loss(y=y, A=C)

            print("Initial value shape: ", (C.T(y)).shape)
            print("The finite difference input shape is: ", D.input_shape)

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
                f"MBIRJAX TV Restruction block {i}\nSNR: %.2f (dB), MAE: %.3f"
                % (metric.snr(tangle_roi_list[i], tangle_recon_roi), metric.mae(tangle_roi_list[i], tangle_recon_roi))
            )

            """
            Show the recovered ROI image.
            """
            fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
            test_slice = 32
            plot.imview(
                tangle_roi_list[i][test_slice],
                title=f"Ground truth (central slice) block {i}",
                cmap=plot.cm.Blues,
                cbar=None,
                fig=fig,
                ax=ax[0],
            )
            plot.imview(
                tangle_recon_roi[test_slice],
                title=f"TV Reconstruction (central slice) block {i}\nSNR: %.2f (dB), MAE: %.3f"
                % (metric.snr(tangle_roi_list[i], tangle_recon_roi), metric.mae(tangle_roi_list[i], tangle_recon_roi)),
                cmap=plot.cm.Blues,
                fig=fig,
                ax=ax[1],
            )
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.2)
            fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
            fig.show()
            plot.close(fig)

            results_dir = os.path.join(os.path.dirname(__file__), f'results/naive_block_admm_{row_division_num}_{col_division_num}')
            os.makedirs(results_dir, exist_ok=True)
            save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_admm_naive_block_{i}_{n_projection}views.png')
            fig.savefig(save_path)   # save the figure to file

            # Save the reconstructed block to a .mat file in case of future full reconstruction
            scipy.io.savemat(os.path.join(results_dir, f'ct_mbirjax_3d_tv_admm_naive_block_{i}_{n_projection}views.mat'), {'array': tangle_recon_roi})


    # Manually read the reconstructed blocks from the .mat files
    block_recon_dir = os.path.join(os.path.dirname(__file__), f'results/naive_block_admm_{row_division_num}_{col_division_num}')

    tangle_recon_list = []
    for idx in range(row_division_num * col_division_num):
        block_recon_image = scipy.io.loadmat(os.path.join(block_recon_dir, f'ct_mbirjax_3d_tv_admm_naive_block_{idx}_{n_projection}views.mat'))['array']
        tangle_recon_list.append(block_recon_image)
        print("Shape of block ", idx, ": ", block_recon_image.shape)

    # Reconstruct the full image
    tangle_recon = np.zeros((Nz, Ny, Nx))
    for i in range(row_division_num):
        for j in range(col_division_num):
            roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
            roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
            tangle_recon[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row] = tangle_recon_list[i * col_division_num + j]

    tangle_recon = snp.array(tangle_recon)

    """
    Show the recovered image.
    """
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    test_slice = 32
    plot.imview(
        tangle[test_slice],
        title="Ground truth (central slice)",
        cmap=plot.cm.Blues,
        cbar=None,
        fig=fig,
        ax=ax[0],
    )
    plot.imview(
        tangle_recon[test_slice],
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

    results_dir = os.path.join(os.path.dirname(__file__), f'results/naive_block_admm_{row_division_num}_{col_division_num}')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_admm_naive_recon_{n_projection}views.png')
    fig.savefig(save_path)   # save the figure to file





if __name__ == "__main__":

    # Preliminary tests for full reconstruction and region of interest reconstruction, using MBIRJAX and ADMM.
    
    # print("\n" + "="*80)
    # print("TEST 1: Full reconstruction ADMM test")
    # print("="*80)
    # test_results_1 = full_recon_test()
    # print("\n✅ Test 1 completed successfully!")

    # print("\n" + "="*80)
    # print("TEST 2: Region of interest reconstruction ADMM test")
    # print("="*80)
    # test_results_2 = partial_recon_test()
    # print("\n✅ Test 2 completed successfully!")

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="3D TV-Regularized Sparse-View CT Reconstruction with ADMM using MBIRJAX",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('-x', '--Nx', type=int, default=128,
                       help='Image width (default: 128)')
    parser.add_argument('-y', '--Ny', type=int, default=256,
                       help='Image height (default: 256)')
    parser.add_argument('-z', '--Nz', type=int, default=64,
                       help='Image depth (default: 64)')
    parser.add_argument('--row_division', type=int, default=4,
                       help='Number of row divisions (default: 4)')
    parser.add_argument('--col_division', type=int, default=8,
                       help='Number of column divisions (default: 8)')
    parser.add_argument('--do-block-recon', action='store_false',
                       help='Perform block reconstruction (default: enabled)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate parameters
    if args.Nx <= 0 or args.Ny <= 0 or args.Nz <= 0:
        parser.error("Image dimensions must be positive integers")
    
    if args.row_division <= 0 or args.col_division <= 0:
        parser.error("Division numbers must be positive integers")
    
    # # Check divisibility
    # if args.nx % args.row_division != 0:
    #     print(f"Warning: NX ({args.nx}) is not evenly divisible by row_division ({args.row_division})")
    
    # if args.ny % args.col_division != 0:
    #     print(f"Warning: NY ({args.ny}) is not evenly divisible by col_division ({args.col_division})")
    
    # Display configuration
    print("="*80)
    print("3D TV-Regularized Sparse-View CT Reconstruction (ADMM Solver) using MBIRJAX")
    print("="*80)
    print(f"Configuration:")
    print(f"  Image dimensions: {args.Nx}x{args.Ny}x{args.Nz}")
    print(f"  Block division: {args.row_division}x{args.col_division}")
    print(f"  Block reconstruction: {args.do_block_recon}")
    print(f"  Block size: {args.Nx // args.row_division}x{args.Ny // args.col_division}x{args.Nz}")
    print(f"  Total blocks: {args.row_division * args.col_division}")
    print("="*80)
    
    # Run the test
    print("\n" + "="*80)
    print("TEST: Simple Block ADMM test")
    print("="*80)
    
    test_results = simple_block_admm_test(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        row_division_num=args.row_division,
        col_division_num=args.col_division,
        do_block_recon=args.do_block_recon
    )
    
    print("\n✅ Test completed successfully!")

    
    