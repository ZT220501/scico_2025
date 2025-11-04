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
from scico.optimize import ProxJacobiADMM, ParallelProxJacobiADMM
from scico.util import device_info, create_roi_indices
from scico.functional import IsotropicTVNorm, L1Norm

import scipy.io
import argparse
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import tifffile as tiff



def preprocess_nanolaminography_image(img_path="/home/zhengtan/datasets/TIFF_delta_regularized_ram-lak_freqscl_1.00"):
    """Load the full 3D image from the TIFF files."""
    # The id runs from 0001 to 0768 to form a 3D image with shape (3840, 3840, 768)
    full_img = []
    for i in range(1, 769):
        img_slice = tiff.imread(os.path.join(img_path, f"tomo_delta_S00974_to_S03875_ram-lak_freqscl_1.00_{i:04d}.tif"))
        full_img.append(img_slice)
    with jax.default_device(jax.devices('cpu')[0]):
        full_img = snp.array(full_img)    # Shape of the image is automatically (768, 3840, 3840), which fits the scico convention.

    return full_img

def save_recon_comparision(x_gt, x_recon, save_path):
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    assert x_gt.shape == x_recon.shape
    Nz = x_gt.shape[0]
    # test_slice = Nz // 2
    test_slice = 439
    plot.imview(
        x_gt[test_slice],
        title="Ground truth (central slice)",
        cmap="gray",
        cbar=None,
        fig=fig,
        ax=ax[0],
    )
    plot.imview(
        x_recon[test_slice],
        title="FBP Reconstruction (central slice)\nSNR: %.2f (dB), MAE: %.3f"
        % (metric.snr(x_gt, x_recon), metric.mae(x_gt, x_recon)),
        cmap="gray",
        fig=fig,
        ax=ax[1],
    )
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
    fig.show()

    # Save the figure
    fig.savefig(save_path)   # save the figure to file
    plt.close(fig)

def pjadmm_parallel_fbp_nanolaminography_test(
    Nx=3840, Ny=3840, Nz=768, row_division_num=2, col_division_num=2, 
    rho=1e-4, tau=0.1, tv_weight=1e-2, n_projection=30, maxiter=1000,
    img_path="/home/zhengtan/datasets/TIFF_delta_regularized_ram-lak_freqscl_1.00",
    save_initial_guess=True):

    gpu_devices = jax.devices('gpu')

    # Load the full image, with shape (768, 3840, 3840)
    print("Loading the full image...")
    x_gt = preprocess_nanolaminography_image(img_path=img_path)

    # Create the projector and the full sinogram
    angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
    sinogram_shape = (Nz, n_projection, max(Nx, Ny))
    A_full = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        recon_shape=(Nx, Ny, Nz)
    )
    print("Creating the full sinogram...")
    sinogram = A_full @ x_gt        # The sinogram is stored onto the first GPU.

    '''
    Partition the initial guess into corresponding blocks.
    Notice that this part is not wrapped as a function, since we want to delete the initial guess to free up memory.
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
            # Blockwise FBP on each of the GPUs.
            sinogram = jax.device_put(sinogram, gpu_devices[device_idx])
            print(f"FBP on GPU {device_idx}...")
            x0_block = A.fbp_recon(sinogram, device="gpu")
            device_idx += 1         # Update the GPU device index to put the block on.
            x0_list.append(x0_block)
    # Define the TV regularizer for each ROI in the later block reconstruction.
    g_list = [IsotropicTVNorm(input_shape=A_list[i].input_shape, input_dtype=A_list[i].input_dtype) for i in range(len(A_list))]

    # # Reconstruct the full image of the initial guess.
    # Nz, Ny, Nx = x_gt.shape
    # with jax.default_device(jax.devices('cpu')[0]):
    #     x0_recon = snp.zeros(x_gt.shape)

    # for i in range(row_division_num):
    #     for j in range(col_division_num):
    #         roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
    #         roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
    #         x0_recon = x0_recon.at[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row].set(jax.device_put(x0_list[i * col_division_num + j], jax.devices('cpu')[0]))

    # save_recon_comparision(x_gt, x0_recon, os.path.join(os.path.dirname(__file__), f'results/ct_mbirjax_3d_tv_pjadmm_parallel_fbp_recon_nanolaminography_initial_guess.png'))
    # exit()

    '''
    Reconstruction using the proximal Jacobi ADMM solver.
    '''
    ρ = rho
    τ = tau
    tv_weight = tv_weight
    γ = 1  # Damping parameter in the proximal Jacobi ADMM solver, in the update of the dual variable.
    λ = snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype)  # Dual variable
    maxiter = maxiter  # number of decentralized ADMM iterations

    correction = False
    α = 0.8 if correction else None      # Relaxation parameter in the proximal Jacobi ADMM solver, in the correction step. Currently not used.
    print(f"ρ: {ρ}, τ: {τ}, regularization: {tv_weight}, γ: {γ}, correction: {correction}, α: {α}, maxiter: {maxiter}")

    test_mode = True
    tv_solver = ParallelProxJacobiADMM(
        A_list=A_list,
        g_list=g_list,
        ρ=ρ,
        y=sinogram,
        τ=τ,
        γ=γ,
        tv_weight = tv_weight,
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

    print(f"Solving block reconstruction using parallel proximal Jacobi ADMM solver...\n")
    x_recon_list = tv_solver.solve()

    '''
    Reconstruct the full image
    '''
    Nz, Ny, Nx = x_gt.shape
    x_recon = snp.zeros(x_gt.shape)

    for i in range(row_division_num):
        for j in range(col_division_num):
            roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
            roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
            x_recon = x_recon.at[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row].set(jax.device_put(x_recon_list[i * col_division_num + j], gpu_devices[0]))

    save_recon_comparision(x_gt, x_recon, os.path.join(os.path.dirname(__file__), f'results/ct_mbirjax_3d_tv_pjadmm_parallel_fbp_recon_nanolaminography.png'))
    # Save the raw reconstruction result.
    results_dir = os.path.join(os.path.dirname(__file__), f'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.imshow(x_recon[test_slice], cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, f"ct_mbirjax_3d_tv_pjadmm_parallel_fbp_recon_nanolaminography_raw.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    return x_recon
    


if __name__ == "__main__":

    # img_path = "/home/zhengtan/datasets/TIFF_delta_regularized_ram-lak_freqscl_1.00"
    # save_path = "/home/zhengtan/repos/scico/examples/scripts/results/nanolaminography_ground_truth"
    # os.makedirs(save_path, exist_ok=True)

    # for i in range(1, 769):
    #     if i % 50 == 0:
    #         print(f"Processing slice {i}...")
    #     img_slice = tiff.imread(os.path.join(img_path, f"tomo_delta_S00974_to_S03875_ram-lak_freqscl_1.00_{i:04d}.tif"))
    #     plt.imshow(img_slice, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(os.path.join(save_path, f"tomo_delta_S00974_to_S03875_ram-lak_freqscl_1.00_{i:04d}.png"), bbox_inches='tight', pad_inches=0)
    #     plt.close()

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="3D TV-Regularized Sparse-View CT Reconstruction with Proximal Jacobi Overlapped ADMM using MBIRJAX",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('-x', '--Nx', type=int, default=3840,
                       help='Image width (default: 3840)')
    parser.add_argument('-y', '--Ny', type=int, default=3840,
                       help='Image height (default: 3840)')
    parser.add_argument('-z', '--Nz', type=int, default=768,
                       help='Image depth (default: 768)')
    parser.add_argument('--row_division_num', type=int, default=2,
                       help='Number of row divisions (default: 2)')
    parser.add_argument('--col_division_num', type=int, default=2,
                       help='Number of column divisions (default: 2)')
    parser.add_argument('--rho', type=float, default=1e-4,
                       help='Regularization parameter (default: 1e-4)')
    parser.add_argument('--tau', type=float, default=0.1,
                       help='Step size parameter (default: 0.1)')
    parser.add_argument('--tv_weight', type=float, default=1e-2,
                       help='TV regularization weight (default: 1e-2)')
    parser.add_argument('--n_projection', type=int, default=30,
                       help='Number of projections (default: 30)')
    parser.add_argument('--maxiter', type=int, default=1000,
                       help='Number of iterations for block reconstruction (default: 1000)')

    # Parse arguments
    args = parser.parse_args()
    
    # Validate parameters
    if args.Nx <= 0 or args.Ny <= 0 or args.Nz <= 0:
        parser.error("Image dimensions must be positive integers")
    
    if args.row_division_num <= 0 or args.col_division_num <= 0:
        parser.error("Division numbers must be positive integers")
    
    # Display configuration
    print("="*100)
    print("3D TV-Regularized Sparse-View CT Reconstruction (Proximal Jacobi Overlapped ADMM Solver) using MBIRJAX")
    print("="*100)
    print(f"Configuration:")
    print(f"  Image dimensions: {args.Nx}x{args.Ny}x{args.Nz}")
    print(f"  Block division number: {args.row_division_num}x{args.col_division_num}")
    print(f"  Block size: {args.Nx // args.row_division_num}x{args.Ny // args.col_division_num}x{args.Nz}")
    print(f"  Total blocks: {args.row_division_num * args.col_division_num}")
    print(f"  Number of projections: {args.n_projection}")
    print(f"  Number of iterations for block reconstruction: {args.maxiter}")
    print("="*80)
    
    # Run the test
    print("\n" + "="*80)
    print("TEST: Parallel Proximal Jacobi ADMM test with FBP initial guess on Real Data")
    print("="*80)
    

    test_results = pjadmm_parallel_fbp_nanolaminography_test(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        row_division_num=args.row_division_num,
        col_division_num=args.col_division_num,
        rho=args.rho,
        tau=args.tau,
        tv_weight=args.tv_weight,
        n_projection=args.n_projection,
        maxiter=args.maxiter,
    )
    print("\n✅ Test completed!")