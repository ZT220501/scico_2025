import numpy as np
import jax
import os
import jax.numpy as jnp
# # Force GPU usage
# os.environ['JAX_PLATFORM_NAME'] = 'gpu'
# jax.config.update('jax_platform_name', 'gpu')

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom, create_3d_foam_phantom
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.optimize import ParallelSemiProxJacobiADMML2PlusReg
from scico.util import device_info, create_roi_indices
from scico.functional import IsotropicTVNorm, L1Norm

import scipy.io
import argparse
import sys
from datetime import datetime
import gc

import matplotlib.pyplot as plt


def cleanup_memory():
    """
    Comprehensive memory cleanup function.
    Clears Python variables, JAX cache, and forces garbage collection.
    """
    # Close all matplotlib figures
    plt.close('all')
    
    # Clear JAX backend caches (important for GPU memory)
    try:
        jax.clear_backends()
        jax.clear_caches()
    except AttributeError:
        # Fallback for older JAX versions
        pass
    
    # Force garbage collection
    gc.collect()
    gc.collect()  # Call twice to handle circular references


def noisy_sinogram(sinogram, snr_db=30, use_variance=True, save_path=None, seed=42):
    """Add Gaussian noise to the sinogram, so that SNR is around snr_db dB."""
    # Set the seed for reproducibility.
    np.random.seed(seed)

    if use_variance:
        P_signal = np.mean((sinogram - sinogram.mean())**2)
    else:
        P_signal = np.mean(sinogram**2)

    sigma_n = np.sqrt(P_signal / (10**(snr_db/10.0)))
    noise = np.random.normal(0.0, sigma_n, size=sinogram.shape).astype(np.float32)
    sinogram_noisy = sinogram + noise
    if save_path is not None:
        save_recon_comparision(sinogram, sinogram_noisy, save_path)
    return sinogram_noisy, noise, sigma_n


def save_recon_comparision(x_gt, x_recon, save_path):
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    assert x_gt.shape == x_recon.shape
    Nz = x_gt.shape[0]
    test_slice = Nz // 2
    plot.imview(
        x_gt[test_slice],
        title="Ground truth (central slice)",
        cmap=plot.cm.Blues,
        cbar=None,
        fig=fig,
        ax=ax[0],
    )
    plot.imview(
        x_recon[test_slice],
        title="FBP Reconstruction (central slice)\nSNR: %.2f (dB), MAE: %.3f"
        % (metric.snr(x_gt, x_recon), metric.mae(x_gt, x_recon)),
        cmap=plot.cm.Blues,
        fig=fig,
        ax=ax[1],
    )
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
    fig.show()

    # Save the figure
    fig.savefig(save_path)   # save the figure to file



'''
Test for reconstruction for full 3D CT image with 
naive way of dividing the image into blocks and reconstructing each block separately.
Proximal Jacobi ADMM is used to reconstruct the full image, with the initial guess 
to be reconstructed using FBP on each of the GPUs.
'''
def pjadmm_parallel_fbp_parallel_noisy_test(
    Nx=128, Ny=256, Nz=64, row_division_num=2, col_division_num=2,
    rho=1e-3, tau=0.1, regularization=1e-2, n_projection=30, maxiter=1000,
    N_sphere=100, sinogram_snr="30", regularization_type="l1"
):
    '''
    Create a ground truth image and projector.
    
    Args:
        do_block_recon: Whether to perform block reconstruction
        Nx: Image width
        Ny: Image height  
        Nz: Image depth
        row_division_num: Number of row divisions number
        col_division_num: Number of column divisions number
    '''
    gpu_devices = jax.devices('gpu')
    print("GPU devices: ", gpu_devices)
    # # Create a full 3D CT image phantom
    # x_gt = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    x_gt = create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=N_sphere, default_device='gpu')

    angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
    sinogram_shape = (Nz, n_projection, max(Nx, Ny))

    # Create the full sinogram
    # NOTE: In the real-world application, only the full sinogram is available. We can not get the sinogram for each ROI.
    A_full = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        recon_shape=(Nx, Ny, Nz)
    )
    print("Creating the full sinogram...")
    y = A_full @ x_gt
    if np.isinf(sinogram_snr):
        print("No noise is added to the sinogram.")
        y_noisy = y
        noise = snp.zeros(y.shape, dtype=y.dtype)
        sigma = 0.0
    else:
        print("Creating the noisy sinogram...")
        sinogram_snr = int(sinogram_snr)
        y_noisy, noise, sigma = noisy_sinogram(y, snr_db=sinogram_snr, use_variance=True, save_path=os.path.join("/home/zhengtan/repos/scico/examples/scripts/results", "noisy_sinogram.png"))

    '''
    Set up problems and solvers for TV regularized solver, block reconstruction using proximal Jacobi ADMM.
    '''
    # Define each ROI (Region of Interest) and its corresponding mbirjax projector
    A_list = [] # List of mbirjax projectors for each ROI
    # Cut the initial guess into corresponding blocks.
    x0_list = []

    # Set up the ROI indices for each ROI.
    row_start_indices = []
    col_start_indices = []
    row_end_indices = []
    col_end_indices = []

    print("Creating mbirjax projectors for each ROI...")
    device_idx = 0
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
            # Append the mbirjax projector to the list
            A_list.append(A)
            # Do the FBP on each of the GPUs by using the ROI FBP.
            print("FBP on GPU %d..." % device_idx)
            x0_block = A.fbp_recon(y_noisy, device="gpu")
            device_idx += 1         # Update the GPU device index to put the block on.
            x0_list.append(x0_block)
    
    # Define the TV regularizer for each ROI in the later block reconstruction.
    if regularization_type == "tv":
        g_list = [IsotropicTVNorm(input_shape=A_list[i].input_shape, input_dtype=A_list[i].input_dtype) for i in range(len(A_list))]
    elif regularization_type == "l1":
        g_list = [L1Norm() for i in range(len(A_list))]
    else:
        raise ValueError(f"Regularization type {regularization_type} not supported.")
        exit()

    # Define the algorithm parameters.
    ρ = rho
    τ = tau
    regularization = regularization
    
    γ = 1  # Damping parameter in the proximal Jacobi ADMM solver, in the update of the dual variable.
    λ = snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype)  # Dual variable
    correction = False
    α = 0.8 if correction else None      # Relaxation parameter in the proximal Jacobi ADMM solver, in the correction step. Currently not used.
    maxiter = maxiter  # number of decentralized ADMM iterations

    print(f"ρ: {ρ}, τ: {τ}, regularization: {regularization}, γ: {γ}, correction: {correction}, α: {α}, maxiter: {maxiter}, regularization_type: {regularization_type}")

    test_mode = True

    tv_solver = ParallelSemiProxJacobiADMML2PlusReg(
        A_list=A_list,
        g_list=g_list,
        ρ=ρ,
        y=y_noisy,          # Note we only have the noisy sinogram available instead of the full sinogram.
        τ=τ,
        γ=γ,
        regularization = regularization,
        λ=λ,
        x0_list=x0_list,
        display_period = 1,
        with_correction = correction,
        α = α,
        test_mode = test_mode,
        ground_truth = x_gt,
        row_division_num = row_division_num,
        col_division_num = col_division_num,
        device_list = gpu_devices,
        num_processes = len(gpu_devices),
        maxiter = maxiter,
        itstat_options={"display": True, "period": 10}
    )

    """
    Run the TV regularized solver, block reconstruction.
    """
    print(f"Solving on {device_info()}\n")
    x_gt_recon_list = tv_solver.solve()
    
    '''
    Reconstruct the full image
    '''
    Nz, Ny, Nx = x_gt.shape
    x_gt_recon = snp.zeros(x_gt.shape)

    for i in range(row_division_num):
        for j in range(col_division_num):
            roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
            roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
            x_gt_recon = x_gt_recon.at[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row].set(jax.device_put(x_gt_recon_list[i * col_division_num + j], gpu_devices[0]))

    """
    Show the recovered image.
    """
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    test_slice = Nz // 2
    plot.imview(
        x_gt[test_slice],
        title="Ground truth (central slice)",
        cmap=plot.cm.Blues,
        cbar=None,
        fig=fig,
        ax=ax[0],
    )
    plot.imview(
        x_gt_recon[test_slice],
        title="TV Reconstruction (central slice)\nSNR: %.2f (dB), MAE: %.3f"
        % (metric.snr(x_gt, x_gt_recon), metric.mae(x_gt, x_gt_recon)),
        cmap=plot.cm.Blues,
        fig=fig,
        ax=ax[1],
    )
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
    fig.show()

    # Save the figure
    # results_dir = os.path.join(os.path.dirname(__file__), f'results/semi_pjadmm_parallel_{regularization_type}_fbp_noisy_sinogram_snr{sinogram_snr}_{row_division_num}_{col_division_num}_N_sphere{N_sphere}_l2_plus_reg')
    # results_dir = os.path.join(os.path.dirname(__file__), f'results/semi_pjadmm_parallel_{regularization_type}_fbp_noisy_sinogram_snr{sinogram_snr}_{row_division_num}_{col_division_num}_N_sphere{N_sphere}_l2_plus_reg_no_tau_decrease')
    # results_dir = os.path.join(os.path.dirname(__file__), f'results/semi_pjadmm_parallel_{regularization_type}_fbp_noisy_sinogram_snr{sinogram_snr}_{row_division_num}_{col_division_num}_N_sphere{N_sphere}_l2_plus_reg_parameter_tuning')
    results_dir = os.path.join(os.path.dirname(__file__), f'results/semi_pjadmm_parallel_{regularization_type}_fbp_noisy_sinogram_snr{sinogram_snr}_{row_division_num}_{col_division_num}_N_sphere{N_sphere}_l2_plus_reg_corrected_residuals')
    os.makedirs(results_dir, exist_ok=True)
    # save_path = os.path.join(results_dir, f'ct_mbirjax_3d_{regularization_type}_semi_pjadmm_l2_plus_reg_parallel_fbp_recon_noisy_{n_projection}views_{Nx}x{Ny}x{Nz}_foam_ρ{ρ}_τ{τ}_regularization{regularization}_gamma{γ}_maxiter{maxiter}.png')
    # save_path = os.path.join(results_dir, f'ct_mbirjax_3d_{regularization_type}_semi_pjadmm_l2_plus_reg_parallel_fbp_recon_noisy_{n_projection}views_{Nx}x{Ny}x{Nz}_foam_ρ{ρ}_τ{τ}_regularization{regularization}_gamma{γ}_maxiter{maxiter}_no_tau_decrease.png')
    # save_path = os.path.join(results_dir, f'ct_mbirjax_3d_{regularization_type}_semi_pjadmm_l2_plus_reg_parallel_fbp_recon_noisy_{n_projection}views_{Nx}x{Ny}x{Nz}_foam_ρ{ρ}_τ{τ}_regularization{regularization}_gamma{γ}_maxiter{maxiter}_parameter_tuning.png')
    save_path = os.path.join(results_dir, f'ct_mbirjax_3d_{regularization_type}_semi_pjadmm_l2_plus_reg_parallel_fbp_recon_noisy_{n_projection}views_{Nx}x{Ny}x{Nz}_foam_ρ{ρ}_τ{τ}_regularization{regularization}_gamma{γ}_maxiter{maxiter}_corrected_residuals.png')
    fig.savefig(save_path)   # save the figure to file

    # Save the result to a txt file
    results_file = os.path.join(results_dir, f'results.txt')
    with open(results_file, 'a') as f:
        f.write(f"Test configuration: Nx: {Nx}, Ny: {Ny}, Nz: {Nz}, row_division_num: {row_division_num}, col_division_num: {col_division_num}, rho: {rho}, tau: {tau}, regularization: {regularization}, n_projection: {n_projection}, maxiter: {maxiter}, N_sphere: {N_sphere}, sinogram_snr: {sinogram_snr}, regularization_type: {regularization_type}\n")
        f.write(f"Final SNR: {round(metric.snr(x_gt, x_gt_recon), 2)} (dB), Final MAE: {round(metric.mae(x_gt, x_gt_recon), 3)}\n")
        f.write("---------------------------------------------------------\n")
    print(f"Final SNR: {round(metric.snr(x_gt, x_gt_recon), 2)} (dB), Final MAE: {round(metric.mae(x_gt, x_gt_recon), 3)}")
    
    # # Save the reconstruction result for replotting
    # raw_data_dir = os.path.join(results_dir, "raw_data")
    # os.makedirs(raw_data_dir, exist_ok=True)
    # snp.save(os.path.join(raw_data_dir, f"x_gt_recon_parallel_noisy_{n_projection}views_snr{sinogram_snr}_ρ{ρ}_τ{τ}_regularization{regularization}_gamma{γ}_maxiter{maxiter}.npy"), x_gt_recon)
    
    return x_gt_recon



if __name__ == "__main__":    

    cleanup_memory()

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="3D TV-Regularized Sparse-View CT Reconstruction with Proximal Jacobi L2+TV ADMM using MBIRJAX",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('-x', '--Nx', type=int, default=128,
                       help='Image width (default: 128)')
    parser.add_argument('-y', '--Ny', type=int, default=256,
                       help='Image height (default: 256)')
    parser.add_argument('-z', '--Nz', type=int, default=64,
                       help='Image depth (default: 64)')
    parser.add_argument('--row_division_num', type=int, default=2,
                       help='Number of row divisions (default: 2)')
    parser.add_argument('--col_division_num', type=int, default=2,
                       help='Number of column divisions (default: 2)')
    parser.add_argument('--rho', type=float, default=1e-3,
                       help='Constraint parameter (default: 1e-3)')
    parser.add_argument('--tau', type=float, default=0.1,
                       help='Step size parameter (default: 0.1)')
    parser.add_argument('--regularization', type=float, default=1e-2,
                       help='Regularization parameter (default: 1e-2)')
    parser.add_argument('--n_projection', type=int, default=30,
                       help='Number of projections (default: 30)')
    parser.add_argument('--maxiter', type=int, default=1000,
                       help='Number of iterations for block reconstruction (default: 1000)')
    parser.add_argument('--N_sphere', type=int, default=100,
                       help='Number of spheres in the foam phantom (default: 100)')
    parser.add_argument('--sinogram_snr', type=str, default='30',
                       help='SNR of the sinogram in dB (default: 30). Use "inf" for no noise.')
    parser.add_argument('--regularization_type', type=str, default="tv",
                       help='Regularization type (default: tv)')
    # Parse arguments
    args = parser.parse_args()
    
    # Convert sinogram_snr to float, handling "inf" case
    if args.sinogram_snr.lower() in ['inf', 'infinity']:
        args.sinogram_snr = float('inf')
    else:
        try:
            args.sinogram_snr = int(args.sinogram_snr)
        except ValueError:
            parser.error(f"sinogram_snr must be a number or 'inf', got: {args.sinogram_snr}")
    
    # Validate parameters
    if args.Nx <= 0 or args.Ny <= 0 or args.Nz <= 0:
        parser.error("Image dimensions must be positive integers")
    
    if args.row_division_num <= 0 or args.col_division_num <= 0:
        parser.error("Division numbers must be positive integers")

    
    # Display configuration
    print("="*100)
    print("3D TV-Regularized Sparse-View CT Reconstruction (Proximal Jacobi L2+TV ADMM Solver) using MBIRJAX")
    print("="*100)
    print(f"Configuration:")
    print(f"  Image dimensions: {args.Nx}x{args.Ny}x{args.Nz}")
    print(f"  Block division number: {args.row_division_num}x{args.col_division_num}")
    print(f"  Block size: {args.Nx // args.row_division_num}x{args.Ny // args.col_division_num}x{args.Nz}")
    print(f"  Total blocks: {args.row_division_num * args.col_division_num}")
    print(f"  Number of projections: {args.n_projection}")
    print(f"  Number of iterations for block reconstruction: {args.maxiter}")
    print(f"  Regularization type: {args.regularization_type}")
    print("="*80)
    
    # Run the test
    print("\n" + "="*80)
    print("TEST: Block Proximal Jacobi ADMM test")
    print("="*80)
    

    test_results = pjadmm_parallel_fbp_parallel_noisy_test(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        row_division_num=args.row_division_num,
        col_division_num=args.col_division_num,
        rho=args.rho,
        tau=args.tau,
        regularization=args.regularization,
        n_projection=args.n_projection,
        maxiter=args.maxiter,
        N_sphere=args.N_sphere,
        sinogram_snr=args.sinogram_snr,
        regularization_type=args.regularization_type
    )
    print("\n✅ Test completed!")
    print("="*80)