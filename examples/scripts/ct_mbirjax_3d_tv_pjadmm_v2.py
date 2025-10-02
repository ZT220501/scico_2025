import numpy as np
import jax
import os

# Force GPU usage
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
jax.config.update('jax_platform_name', 'gpu')

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom, create_3d_foam_phantom
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.optimize import ProxJacobiADMMv2
from scico.util import device_info, create_roi_indices
from scico.functional import IsotropicTVNorm, L1Norm

import scipy.io
import argparse
import sys
from datetime import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm


'''
Test for reconstruction for full 3D CT image with 
naive way of dividing the image into blocks and reconstructing each block separately.
ADMM and MBIRJAX are used to reconstruct each block.
'''
def pjadmm_test(
    Nx=128, Ny=256, Nz=64, row_division_num=2, col_division_num=2, rho=1e-3, tau_factor=1.05, gamma=1, tv_weight=10, n_projection=30, maxiter=1000
):
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
    gpu_devices = jax.devices('gpu')
    print(f"Using {gpu_devices[0]} GPU")

    # # Create a full 3D CT image phantom
    # tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))

    # TODO: Maybe times this by a large factor like 10^8, see how it works.
    tangle = snp.array(create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=100))
    tangle = snp.array(jax.device_put(tangle, gpu_devices[0]))

    angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles

    # Define each ROI (Region of Interest) and its corresponding mbirjax projector
    A_list = [] # List of mbirjax projectors for each ROI
    g_list = [] # List of TV regularizers for each ROI

    sinogram_shape = (Nz, n_projection, max(Nx, Ny))

    # Create the full sinogram
    # NOTE: In the real-world application, only the full sinogram is available. We can not get the sinogram for each ROI.
    A_full = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        recon_shape=(Nx, Ny, Nz)
    )
    y = A_full @ tangle

    row_start_indices = []
    col_start_indices = []
    row_end_indices = []
    col_end_indices = []


    for i in range(row_division_num):
        for j in range(col_division_num):
            roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
            roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns

            row_start_indices.append(roi_start_row)
            col_start_indices.append(roi_start_col)
            row_end_indices.append(roi_end_row)
            col_end_indices.append(roi_end_col)

            assert roi_start_row >= 0 and roi_start_col >= 0 and roi_end_row <= Nx and roi_end_col <= Ny

            roi_indices = create_roi_indices(Nx, Ny, roi_start_row, roi_end_row, roi_start_col, roi_end_col)

            # Create the mbirjax projector for the current ROI
            A = XRayTransformParallel(
                output_shape=sinogram_shape,
                angles=angles,
                partial_reconstruction=True,
                roi_indices=roi_indices,
                roi_recon_shape=(roi_end_row - roi_start_row, roi_end_col - roi_start_col, Nz),
                recon_shape=(Nx, Ny, Nz)
            )

            # g = IsotropicTVNorm(input_shape=A.input_shape, input_dtype=A.input_dtype)
            # g = L1Norm()

            # Append the mbirjax projector and sinogram to the list
            A_list.append(A)
            # g_list.append(g)





    """
    Set up problems and solvers for tv regularized solver.
    In the second version, only one stage is used; the parameters are uniformly chosen by estimating the operator norm.
    """
    ρ = rho
    τ = [ProxJacobiADMMv2.estimate_parameter(A_list[i], rho=ρ, maxiter=100, factor=tau_factor) for i in tqdm(range(len(A_list)))]
    tv_weight = tv_weight

    γ = gamma  # Damping parameter
    λ = snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype)  # Dual variable
    correction = False
    α = 0.8 if correction else None
    maxiter = 1000  # number of decentralized ADMM iterations

    print(f"ρ: {ρ}, τ factor: {tau_factor}, regularization: {tv_weight}, γ: {γ}, correction: {correction}, α: {α}, maxiter: {maxiter}")
    print("τ: ", τ)

    test_mode = True

    # Set up the tv regularizer for each ROI as the new g_list.
    g_list = [IsotropicTVNorm(input_shape=A_list[i].input_shape, input_dtype=A_list[i].input_dtype) for i in range(len(A_list))]

    # Use the result of l1 regularized solver as the initial guess for tv regularized solver.
    tv_solver = ProxJacobiADMMv2(
        A_list=A_list,
        g_list=g_list,
        ρ=ρ,
        y=y,
        τ=τ,
        γ=γ,
        λ=λ,
        # TODO: THIS IS NOT FBP!!!!
        # Try filter back projection as the initial condition, in constrast to the current one.
        # Or: Try using a FULL ADMM solver (with less iterations) as the initial condition!
        x0_list=[snp.array(jax.device_put(A_list[i].T(y), gpu_devices[0])) for i in range(len(A_list))],
        display_period = 1,
        device = gpu_devices[0],
        with_correction = correction,
        α = α,
        maxiter = maxiter,
        itstat_options={"display": True, "period": 10},
        ground_truth = tangle,
        test_mode = test_mode,
        row_division_num = row_division_num,
        col_division_num = col_division_num,
        tv_weight = tv_weight
    )

    """
    Run the tv regularized solver.
    """
    print(f"Solving on {device_info()}\n")
    tangle_recon_list= tv_solver.solve()


    '''
    Reconstruct the full image
    '''
    Nz, Ny, Nx = tangle.shape
    tangle_recon = snp.zeros(tangle.shape)

    for i in range(row_division_num):
        for j in range(col_division_num):
            roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
            roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
            tangle_recon = tangle_recon.at[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row].set(tangle_recon_list[i * col_division_num + j])



    """
    Show the recovered image.
    """
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    test_slice = Nz // 2
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

    results_dir = os.path.join(os.path.dirname(__file__), f'results/pjadmm_tv_v2_adaptive_τ_{row_division_num}_{col_division_num}')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_pjadmm_v2_adaptive_τ_recon_{n_projection}views_{Nx}x{Ny}x{Nz}_foam_ρ{ρ}_τfactor{tau_factor}_γ{γ}_tv_weight{tv_weight}_maxiter{maxiter}.png')
    fig.savefig(save_path)   # save the figure to file

    return True



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="3D TV-Regularized Sparse-View CT Reconstruction with Proximal Jacobi ADMM v2 using MBIRJAX",
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
    parser.add_argument('--rho', type=float, default=1e-3,
                       help='Regularization parameter (default: 1e-3)')
    parser.add_argument('--tau_factor', type=float, default=1.05,
                       help='Step size parameter (default: 1.05)')
    parser.add_argument('--gamma', type=float, default=1,
                       help='Damping parameter (default: 1)')
    parser.add_argument('--tv_weight', type=float, default=10,
                       help='TV regularization weight (default: 10)')
    parser.add_argument('--n_projection', type=int, default=30,
                       help='Number of projections (default: 30)')
    parser.add_argument('--maxiter', type=int, default=1000,
                       help='Number of iterations (default: 1000)')
    # Parse arguments
    args = parser.parse_args()
    
    # Validate parameters
    if args.Nx <= 0 or args.Ny <= 0 or args.Nz <= 0:
        parser.error("Image dimensions must be positive integers")
    
    if args.row_division <= 0 or args.col_division <= 0:
        parser.error("Division numbers must be positive integers")
    
    # Display configuration
    print("="*100)
    print("3D TV-Regularized Sparse-View CT Reconstruction (Proximal Jacobi ADMM v2 Solver) using MBIRJAX")
    print("="*100)
    print(f"Configuration:")
    print(f"  Image dimensions: {args.Nx}x{args.Ny}x{args.Nz}")
    print(f"  Block division: {args.row_division}x{args.col_division}")
    print(f"  Block size: {args.Nx // args.row_division}x{args.Ny // args.col_division}x{args.Nz}")
    print(f"  Total blocks: {args.row_division * args.col_division}")
    print("="*80)
    
    # Run the test
    print("\n" + "="*80)
    print("TEST: Block Proximal Jacobi ADMM test")
    print("="*80)
    

    test_results = pjadmm_test(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        row_division_num=args.row_division,
        col_division_num=args.col_division,
        rho=args.rho,
        tau_factor=args.tau_factor,
        tv_weight=args.tv_weight,
        n_projection=args.n_projection,
        maxiter=args.maxiter
    )
    print("\n✅ Test completed!")