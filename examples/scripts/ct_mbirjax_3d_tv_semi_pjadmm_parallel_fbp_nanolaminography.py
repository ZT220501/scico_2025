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
from scico.optimize import ParallelSemiProxJacobiADMML2PlusReg
from scico.util import device_info, create_roi_indices
from scico.functional import IsotropicTVNorm, L1Norm

import scipy.io
import argparse
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.ndimage import zoom

# This funtion loads the full 3D image from the TIFF files from the original paper.
# The image size is (768, 3840, 3840), which is of the full size of the image.
def load_nanolaminography_image_from_tiff(img_path="/home/zhengtan/datasets/TIFF_delta_regularized_ram-lak_freqscl_1.00"):
    """Load the full 3D image from the TIFF files."""
    # The id runs from 0001 to 0768 to form a 3D image with shape (3840, 3840, 768)
    full_img = []
    for i in range(1, 769):
        img_slice = tiff.imread(os.path.join(img_path, f"tomo_delta_S00974_to_S03875_ram-lak_freqscl_1.00_{i:04d}.tif"))
        full_img.append(img_slice)
    with jax.default_device(jax.devices('cpu')[0]):
        full_img = snp.array(full_img)    # Shape of the image is automatically (768, 3840, 3840), which fits the scico convention.

    return full_img

# This function downsamples the image by a factor of downsampling_factor in the frontal dimensions.
# The expected shape of the original image is (768, 3840, 3840)
# The expected shape of the downsampled image is (768, 1920, 1920) if downsampling_factor is 2.
def image_downsampling(img, downsampling_factor=2):
    """Downsample the image by a factor of downsampling_factor in the frontal dimensions."""
    return zoom(img, (1, 1/downsampling_factor, 1/downsampling_factor), order=1)

# This function loads the saved images from the PNG files, which are the saved images in the scico repo.
# The image size is (768, 1920, 1920) if downsampled or (3840, 3840, 768) which is the full size image.
def load_saved_nanolaminography_image(downsample=True):
    """Load the saved downsampled image from the PNG files."""
    if downsample:
        img_path = "/home/zhengtan/repos/scico/examples/scripts/results/nanolaminography_ground_truth_downsampled"
        return snp.load(os.path.join(img_path, 'tomo_volume_downsampled.npy'))
    else:
        img_path = "/home/zhengtan/repos/scico/examples/scripts/results/nanolaminography_ground_truth"
        return snp.load(os.path.join(img_path, 'tomo_volume.npy'))

def save_recon_comparision(x_gt, x_recon, save_path, test_slice=None):
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    assert x_gt.shape == x_recon.shape
    Nz = x_gt.shape[0]
    if test_slice is None:
        test_slice = Nz // 2
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


def pjadmm_parallel_fbp_nanolaminography_test(
    row_division_num=2, col_division_num=2, 
    rho=4e-2, tau=0.1, regularization=2e0, n_projection=100, maxiter=1000,
    downsample=True, regularization_type="tv", sinogram_snr=float('inf')):
    # Get the GPU devices used.
    gpu_devices = jax.devices('gpu')
    # Load the saved ground truth image, with or without downsampling.
    print("Loading the ground truth image...")
    x_gt = load_saved_nanolaminography_image(downsample=downsample)
    x_gt = x_gt.astype(snp.float16)     # Convert to float32 to avoid overflow when plotting uint16 images (Now try float16 to save memory)
    # Get the shape of the ground truth image.
    Nz, Ny, Nx = x_gt.shape

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
    if np.isinf(sinogram_snr):
        print("No noise is added to the sinogram.")
        y_noisy = sinogram
        noise = snp.zeros(sinogram.shape, dtype=sinogram.dtype)
        sigma = 0.0
    else:
        print("Creating the noisy sinogram...")
        sinogram_snr = int(sinogram_snr)
        y_noisy, noise, sigma = noisy_sinogram(sinogram, snr_db=sinogram_snr, use_variance=True, save_path=os.path.join("/home/zhengtan/repos/scico/examples/scripts/results", "noisy_sinogram_nanolaminography.png"))

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
                recon_shape=(Nx, Ny, Nz),
            )
            # Append the mbirjax projector to the list
            A_list.append(A)
            # Blockwise FBP on each of the GPUs.
            sinogram = jax.device_put(sinogram, gpu_devices[device_idx])
            print(f"FBP on GPU {device_idx}...")
            x0_block = A.fbp_recon(sinogram, device="gpu")
            device_idx += 1         # Update the GPU device index to put the block on.
            x0_list.append(x0_block)

    print("Type of A is: ", type(A_list[0]))

    # Define the TV regularizer for each ROI in the later block reconstruction.
    if regularization_type == "tv":
        g_list = [IsotropicTVNorm(input_shape=A_list[i].input_shape, input_dtype=A_list[i].input_dtype) for i in range(len(A_list))]
    elif regularization_type == "l1":
        g_list = [L1Norm() for i in range(len(A_list))]
    else:
        raise ValueError(f"Regularization type {regularization_type} not supported.")
        exit()

    '''
    Reconstruction using the proximal Jacobi ADMM solver.
    '''
    ρ = rho
    τ = tau
    regularization = regularization
    γ = 1  # Damping parameter in the proximal Jacobi ADMM solver, in the update of the dual variable.
    λ = snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype)  # Dual variable
    maxiter = maxiter  # number of decentralized ADMM iterations

    correction = False
    α = 0.8 if correction else None      # Relaxation parameter in the proximal Jacobi ADMM solver, in the correction step. Currently not used.
    print(f"ρ: {ρ}, τ: {τ}, regularization: {regularization}, γ: {γ}, correction: {correction}, α: {α}, maxiter: {maxiter}")

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
        display_period = 100,
        with_correction = correction,
        α = α,
        test_mode = test_mode,
        ground_truth = x_gt,
        row_division_num = row_division_num,
        col_division_num = col_division_num,
        device_list = gpu_devices,
        num_processes = len(gpu_devices),
        maxiter = maxiter,
        itstat_options={"display": True, "period": 100}
    )

    print(f"Solving block reconstruction using parallel proximal semi-Jacobi ADMM solver...\n")
    x_recon_list = tv_solver.solve()

    '''
    Reconstruct the full image
    '''
    Nz, Ny, Nx = x_gt.shape
    x_recon = snp.zeros(x_gt.shape)
    # Use slice 439 for visual quality testing, which is a representative slice with fine details.
    # test_slices = [359, 439, 529, 639]
    for i in range(row_division_num):
        for j in range(col_division_num):
            roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
            roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
            x_recon = x_recon.at[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row].set(jax.device_put(x_recon_list[i * col_division_num + j], gpu_devices[0]))

    # Save the reconstruction results.
    print("Saving the reconstruction results...")
    if downsample:
        results_dir = os.path.join(os.path.dirname(__file__), f"results/semi_pjadmm_parallel_{regularization_type}_fbp_recon_nanolaminography_sinogram_snr{sinogram_snr}_{row_division_num}_{col_division_num}_l2_plus_reg_downsampled")
    else:
        results_dir = os.path.join(os.path.dirname(__file__), f"results/semi_pjadmm_parallel_{regularization_type}_fbp_recon_nanolaminography_sinogram_snr{sinogram_snr}_{row_division_num}_{col_division_num}_l2_plus_reg")
    # for test_slice in test_slices:
    for test_slice in range(1, Nz+1):
        slice_dir = os.path.join(results_dir, f'slice{test_slice}')
        os.makedirs(slice_dir, exist_ok=True)
        # Save the reconstruction comparison figure.
        save_recon_comparision(x_gt, x_recon, os.path.join(slice_dir, f'ct_mbirjax_3d_{regularization_type}_semi_pjadmm_parallel_fbp_recon_nanolaminography_{n_projection}views_ρ{ρ}_τ{τ}_regularization{regularization}_maxiter{maxiter}.png'), test_slice=test_slice)
        # Save the raw reconstruction result.
        plt.imshow(x_recon[test_slice], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(slice_dir, f"ct_mbirjax_3d_{regularization_type}_semi_pjadmm_parallel_fbp_recon_nanolaminography_{n_projection}views_ρ{ρ}_τ{τ}_regularization{regularization}_maxiter{maxiter}_raw.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
    # Save the raw full reconstruction result so that it can be examined for various metrics in the future.
    snp.save(os.path.join(results_dir, f'ct_mbirjax_3d_{regularization_type}_semi_pjadmm_parallel_fbp_recon_nanolaminography_{n_projection}views_ρ{ρ}_τ{τ}_regularization{regularization}_maxiter{maxiter}.npy'), x_recon)

    # Save the result to a txt file
    results_file = os.path.join(results_dir, f'results.txt')
    with open(results_file, 'a') as f:
        f.write(f"Test configuration: Nx: {Nx}, Ny: {Ny}, Nz: {Nz}, row_division_num: {row_division_num}, col_division_num: {col_division_num}, rho: {rho}, tau: {tau}, regularization: {regularization}, n_projection: {n_projection}, maxiter: {maxiter}, sinogram_snr: {sinogram_snr}, regularization_type: {regularization_type}\n")
        f.write(f"Final SNR: {round(metric.snr(x_gt, x_recon), 2)} (dB), Final MAE: {round(metric.mae(x_gt, x_recon), 3)}\n")
        f.write("---------------------------------------------------------\n")
    print(f"Final SNR: {round(metric.snr(x_gt, x_recon), 2)} (dB), Final MAE: {round(metric.mae(x_gt, x_recon), 3)}")
    return x_recon
    


if __name__ == "__main__":
    # img_path = "/home/zhengtan/datasets/TIFF_delta_regularized_ram-lak_freqscl_1.00"
    # save_path = "/home/zhengtan/repos/scico/examples/scripts/results/nanolaminography_ground_truth"
    # # Load the full image, with shape (768, 3840, 3840)
    # print("Loading the full image...")
    # x_gt = load_nanolaminography_image_from_tiff(img_path=img_path)
    # snp.save(os.path.join(save_path, 'tomo_volume.npy'), x_gt)
    # exit()
    # # Save the downsampled image
    # print("Downsampling the image...")
    # x_gt_downsampled = image_downsampling(x_gt)
    # print("Shape of the downsampled image: ", x_gt_downsampled.shape)
    # save_path = "/home/zhengtan/repos/scico/examples/scripts/results/nanolaminography_ground_truth_downsampled"
    # snp.save(os.path.join(save_path, 'tomo_volume_downsampled.npy'), x_gt_downsampled)
    # exit()
    # os.makedirs(save_path, exist_ok=True)    
    # for i in range(1, 769):
    #     if i % 50 == 0:
    #         print(f"Processing slice {i}...")
    #     img_slice = x_gt_downsampled[i-1]
    #     plt.imshow(img_slice, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(os.path.join(save_path, f"tomo_delta_S00974_to_S03875_ram-lak_freqscl_1.00_downsampled_{i:04d}.png"), bbox_inches='tight', pad_inches=0)
    #     plt.close()

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="3D TV-Regularized Sparse-View CT Reconstruction with Proximal Semi-Jacobi ADMM using MBIRJAX",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('--row_division_num', type=int, default=2,
                       help='Number of row divisions (default: 2)')
    parser.add_argument('--col_division_num', type=int, default=2,
                       help='Number of column divisions (default: 2)')
    parser.add_argument('--rho', type=float, default=4e-2,
                       help='Regularization parameter (default: 1e-4)')
    parser.add_argument('--tau', type=float, default=0.1,
                       help='Step size parameter (default: 0.1)')
    parser.add_argument('--regularization', type=float, default=2e0,
                       help='TV regularization weight (default: 2e0)')
    parser.add_argument('--n_projection', type=int, default=100,
                       help='Number of projections (default: 100)')
    parser.add_argument('--maxiter', type=int, default=1000,
                       help='Number of iterations for block reconstruction (default: 1000)')
    parser.add_argument('--downsample', type=bool, default=True,
                       help='Whether to downsample the image (default: True)')
    parser.add_argument('--regularization_type', type=str, default="tv",
                       help='Regularization type (default: tv)')
    parser.add_argument('--sinogram_snr', type=str, default='30',
                       help='SNR of the sinogram in dB (default: 30). Use "inf" for no noise.')

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
    if args.row_division_num <= 0 or args.col_division_num <= 0:
        parser.error("Division numbers must be positive integers")
    
    # Display configuration
    print("="*100)
    print("3D TV-Regularized Sparse-View CT Reconstruction (Proximal Jacobi Overlapped ADMM Solver) using MBIRJAX")
    print("="*100)
    print(f"Configuration:")
    print(f"  Block division number: {args.row_division_num}x{args.col_division_num}")
    print(f"  Total blocks: {args.row_division_num * args.col_division_num}")
    print(f"  Number of projections: {args.n_projection}")
    print(f"  Number of iterations for block reconstruction: {args.maxiter}")
    print(f"  Regularization type: {args.regularization_type}")
    print(f"  SNR of the sinogram: {args.sinogram_snr}")
    print(f"  Downsample: {args.downsample}")
    print("="*80)
    
    # Run the test
    print("\n" + "="*80)
    print("TEST: Parallel Proximal Jacobi ADMM test with FBP initial guess on Real Data")
    print("="*80)
    

    test_results = pjadmm_parallel_fbp_nanolaminography_test(
        row_division_num=args.row_division_num,
        col_division_num=args.col_division_num,
        rho=args.rho,
        tau=args.tau,
        regularization=args.regularization,
        n_projection=args.n_projection,
        maxiter=args.maxiter,
        downsample=args.downsample,
        regularization_type=args.regularization_type,
        sinogram_snr=args.sinogram_snr,
    )
    print("\n✅ Test completed!")