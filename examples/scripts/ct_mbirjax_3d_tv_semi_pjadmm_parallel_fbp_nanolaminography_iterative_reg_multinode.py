import numpy as np
import jax
import os
import argparse
import sys
from datetime import datetime
from distutils.util import strtobool
from jax.experimental import multihost_utils

# Force GPU usage
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
jax.config.update('jax_platform_name', 'gpu')

from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io

import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.ndimage import zoom

snp = None
linop = None
metric = None
plot = None
XRayTransformParallel = None
ParallelSemiProxJacobiADMML2PlusRegIterativeRegEstimatedPDHGMultinode = None
create_roi_indices = None
L1Norm = None
L21Norm = None


def load_runtime_modules():
    """Import JAX-dependent runtime modules after distributed init."""
    global snp
    global linop
    global metric
    global plot
    global XRayTransformParallel
    global ParallelSemiProxJacobiADMML2PlusRegIterativeRegEstimatedPDHGMultinode
    global create_roi_indices
    global L1Norm
    global L21Norm

    import scico.numpy as _snp
    from scico import linop as _linop, metric as _metric, plot as _plot
    from scico.linop.xray.mbirjax import XRayTransformParallel as _XRayTransformParallel
    from scico.optimize._semipjadmm_parallel_l2_plus_reg_iterative_reg_estimated_pdhg_multinode import (
        ParallelSemiProxJacobiADMML2PlusRegIterativeRegEstimatedPDHGMultinode as _Solver,
    )
    from scico.util import create_roi_indices as _create_roi_indices
    from scico.functional import L1Norm as _L1Norm, L21Norm as _L21Norm

    snp = _snp
    linop = _linop
    metric = _metric
    plot = _plot
    XRayTransformParallel = _XRayTransformParallel
    ParallelSemiProxJacobiADMML2PlusRegIterativeRegEstimatedPDHGMultinode = _Solver
    create_roi_indices = _create_roi_indices
    L1Norm = _L1Norm
    L21Norm = _L21Norm


def initialize_jax_distributed_if_needed(num_processes, process_id, coordinator_address, local_device_ids):
    """Initialize JAX distributed mode when running with multiple processes."""
    if num_processes <= 1:
        return
    if coordinator_address is None:
        raise ValueError("coordinator_address is required when num_processes > 1.")
    parsed_local_device_ids = [int(device_id) for device_id in local_device_ids.split(",") if device_id != ""]
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=parsed_local_device_ids,
    )


def is_root_process():
    return jax.process_index() == 0


def rank_print(*args, **kwargs):
    print(f"[rank {jax.process_index()}]", *args, **kwargs)

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
def load_saved_nanolaminography_image(downsample_factor=2, data_root=None):
    """Load the saved downsampled image from the PNG files."""
    if not data_root:
        data_root = os.environ.get(
            "NANOLAMINOGRAPHY_DATA_ROOT",
            os.path.join(os.path.dirname(__file__), "results"),
        )
    if downsample_factor > 1:
        img_path = os.path.join(
            data_root,
            f"nanolaminography_ground_truth_downsampled_factor_{downsample_factor}",
        )
        volume_path = os.path.join(img_path, 'tomo_volume_downsampled.npy')
    else:
        img_path = os.path.join(data_root, "nanolaminography_ground_truth")
        volume_path = os.path.join(img_path, 'tomo_volume.npy')
    if not os.path.exists(volume_path):
        available_downsampled = sorted(
            entry
            for entry in os.listdir(data_root)
            if entry.startswith("nanolaminography_ground_truth_downsampled_factor_")
        ) if os.path.isdir(data_root) else []
        raise FileNotFoundError(
            f"Nanolaminography volume file not found: {volume_path}. "
            f"Set --data_root or NANOLAMINOGRAPHY_DATA_ROOT to the directory containing "
            f"'nanolaminography_ground_truth[_downsampled_factor_N]'. "
            f"Available downsampled datasets under {data_root}: {available_downsampled or 'none found'}."
        )
    return snp.load(volume_path)

def save_recon_comparision(x_gt, x_recon, save_path, test_slice=None):
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    assert x_gt.shape == x_recon.shape
    Nz = x_gt.shape[0]
    if test_slice is None:
        test_slice = Nz // 2
    plot.imview(
        x_gt[test_slice].astype(snp.float16),
        title=f"Ground truth slice {test_slice}",
        cmap="gray",
        cbar=None,
        fig=fig,
        ax=ax[0],
    )
    plot.imview(
        x_recon[test_slice],
        title=f"TV-Regularized Reconstruction slice {test_slice}\nSNR: %.2f (dB), MAE: %.3f"
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
    rho=5e-1, tau=0.1, regularization=2e0, n_projection=50, maxiter=5000,
    downsample_factor=2, regularization_type="tv", sinogram_snr=float('inf'), 
    maxiter_pdhg=50, tau_decrease=True, data_root=None):
    global_gpu_devices = jax.devices('gpu')
    local_gpu_devices = jax.local_devices(backend='gpu')
    if len(local_gpu_devices) == 0:
        raise ValueError("No local GPU devices are available on this process.")
    total_blocks = row_division_num * col_division_num
    if total_blocks != len(global_gpu_devices):
        raise ValueError(
            f"Number of blocks ({total_blocks}) must equal total GPU count ({len(global_gpu_devices)}) for the multinode layout."
        )
    if len(local_gpu_devices) * jax.process_count() != total_blocks:
        raise ValueError(
            "Expected a uniform block-to-GPU mapping with one block per GPU across all processes."
        )
    local_block_count = len(local_gpu_devices)
    local_block_start = jax.process_index() * local_block_count
    local_block_indices = list(range(local_block_start, local_block_start + local_block_count))

    # Load the saved ground truth image, with or without downsampling.
    rank_print("Loading the ground truth image...")
    x_gt = load_saved_nanolaminography_image(
        downsample_factor=downsample_factor,
        data_root=data_root,
    )
    # x_gt = x_gt.astype(snp.float16)     # Convert to float32 to avoid overflow when plotting uint16 images (Now try float16 to save memory)
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
    rank_print("Creating the full sinogram...")
    sinogram = A_full @ x_gt        # The sinogram is stored onto the first GPU.
    if np.isinf(sinogram_snr):
        rank_print("No noise is added to the sinogram.")
        y_noisy = sinogram
        noise = snp.zeros(sinogram.shape, dtype=sinogram.dtype)
        sigma = 0.0
    else:
        rank_print("Creating the noisy sinogram...")
        sinogram_snr = int(sinogram_snr)
        intermediate_results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(intermediate_results_dir, exist_ok=True)
        y_noisy, noise, sigma = noisy_sinogram(
            sinogram,
            snr_db=sinogram_snr,
            use_variance=True,
            save_path=os.path.join(intermediate_results_dir, "noisy_sinogram_nanolaminography.png"),
        )

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

    rank_print(
        f"Creating local mbirjax projectors for global block indices {local_block_indices} on local devices {[device.id for device in local_gpu_devices]}..."
    )
    for local_device_idx, global_block_idx in enumerate(local_block_indices):
        i = global_block_idx // col_division_num
        j = global_block_idx % col_division_num
        roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num
        roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num

        row_start_indices.append(roi_start_row)
        col_start_indices.append(roi_start_col)
        row_end_indices.append(roi_end_row)
        col_end_indices.append(roi_end_col)
        assert roi_start_row >= 0 and roi_start_col >= 0 and roi_end_row <= Nx and roi_end_col <= Ny
        roi_indices = create_roi_indices(Nx, Ny, roi_start_row, roi_end_row, roi_start_col, roi_end_col)

        A = XRayTransformParallel(
            output_shape=sinogram_shape,
            angles=angles,
            partial_reconstruction=True,
            roi_indices=roi_indices,
            roi_recon_shape=(roi_end_row - roi_start_row, roi_end_col - roi_start_col, Nz),
            recon_shape=(Nx, Ny, Nz),
        )
        A_list.append(A)
        sinogram_on_device = jax.device_put(sinogram, local_gpu_devices[local_device_idx])
        rank_print(f"FBP for global block {global_block_idx} on local GPU {local_device_idx}...")
        x0_block = A.fbp_recon(sinogram_on_device, device="gpu")
        x0_list.append(x0_block)

    # Define the TV regularizer for each ROI in the later block reconstruction.
    if regularization_type == "tv":
        g_list = [L21Norm() for i in range(len(A_list))]
        C_list = [linop.FiniteDifference(input_shape=A_list[i].input_shape, input_dtype=A_list[i].input_dtype, append=0) for i in range(len(A_list))]
    elif regularization_type == "l1":
        g_list = [L1Norm() for i in range(len(A_list))]
        C_list = [linop.Identity(input_shape=A_list[i].input_shape, input_dtype=A_list[i].input_dtype) for i in range(len(A_list))]
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
    rank_print(
        f"Nx: {Nx}, Ny: {Ny}, Nz: {Nz}, row_division_num: {row_division_num}, col_division_num: {col_division_num}, "
        f"ρ: {ρ}, τ: {τ}, regularization: {regularization}, γ: {γ}, correction: {correction}, α: {α}, "
        f"maxiter: {maxiter}, regularization_type: {regularization_type}, maxiter_pdhg: {maxiter_pdhg}"
    )

    test_mode = True
    tv_solver = ParallelSemiProxJacobiADMML2PlusRegIterativeRegEstimatedPDHGMultinode(
        A_list=A_list,
        g_list=g_list,
        C_list=C_list,
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
        full_reconstruction_shape = x_gt.shape,
        row_division_num = row_division_num,
        col_division_num = col_division_num,
        device_list = local_gpu_devices,
        num_processes = jax.process_count(),
        global_block_indices = local_block_indices,
        maxiter_pdhg = maxiter_pdhg,
        tau_decrease = tau_decrease,
        maxiter = maxiter,
        itstat_options={"display": is_root_process(), "period": 10}
    )

    rank_print("Solving block reconstruction using multinode parallel proximal semi-Jacobi ADMM solver...\n")
    x_recon_list = tv_solver.solve()

    # All ranks must participate in the final cross-host gather before any
    # process exits, otherwise JAX distributed shutdown can deadlock.
    ordered_blocks = tv_solver._gather_global_blocks()
    multihost_utils.sync_global_devices("post_solve_block_gather")

    if is_root_process():
        x_recon = tv_solver._assemble_global_volume_from_blocks(ordered_blocks)

        final_snr = metric.snr(x_gt, x_recon)
        final_mae = metric.mae(x_gt, x_recon)
        print(f"Final SNR: {round(final_snr, 2)} (dB), Final MAE: {round(final_mae, 3)}")
        # Convert the ground truth image to float16 in order to do the plotting.

        # Save the reconstruction results.
        print("Saving the reconstruction results...")
        results_dir = os.path.join(os.path.dirname(__file__), f"results/semi_pjadmm_parallel_{regularization_type}_fbp_recon_nanolaminography_sinogram_snr{sinogram_snr}_{row_division_num}_{col_division_num}_l2_plus_reg_iterative_reg_multinode_downsampled_factor_{downsample_factor}")
        os.makedirs(results_dir, exist_ok=True)
        # for test_slice in test_slices:
        for test_slice in range(1, Nz+1):
            slice_dir = os.path.join(results_dir, f'slice{test_slice}')
            os.makedirs(slice_dir, exist_ok=True)
            # Save the reconstruction comparison figure.
            save_recon_comparision(x_gt, x_recon, os.path.join(slice_dir, f'ct_mbirjax_3d_{regularization_type}_semi_pjadmm_parallel_fbp_recon_nanolaminography_{n_projection}views_ρ{ρ}_τ{τ}_regularization{regularization}_maxiter{maxiter}_maxiter_pdhg{maxiter_pdhg}.png'), test_slice=test_slice)
            # Save the raw reconstruction result.
            plt.imshow(x_recon[test_slice], cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(slice_dir, f"ct_mbirjax_3d_{regularization_type}_semi_pjadmm_parallel_fbp_recon_nanolaminography_{n_projection}views_ρ{ρ}_τ{τ}_regularization{regularization}_maxiter{maxiter}_maxiter_pdhg{maxiter_pdhg}_raw.png"), bbox_inches='tight', pad_inches=0)
            plt.close()
        # Save the raw full reconstruction result so that it can be examined for various metrics in the future.
        snp.save(os.path.join(results_dir, f'ct_mbirjax_3d_{regularization_type}_semi_pjadmm_parallel_fbp_recon_nanolaminography_{n_projection}views_ρ{ρ}_τ{τ}_regularization{regularization}_maxiter{maxiter}_maxiter_pdhg{maxiter_pdhg}_iterative_reg_estimated_pdhg.npy'), x_recon)

        # Save the result to a txt file
        results_file = os.path.join(results_dir, f'results.txt')
        with open(results_file, 'a') as f:
            f.write(f"Test configuration: Nx: {Nx}, Ny: {Ny}, Nz: {Nz}, row_division_num: {row_division_num}, col_division_num: {col_division_num}, rho: {rho}, tau: {tau}, regularization: {regularization}, n_projection: {n_projection}, maxiter: {maxiter}, sinogram_snr: {sinogram_snr}, regularization_type: {regularization_type}, maxiter_pdhg: {maxiter_pdhg}\n")
            f.write(f"Final SNR: {round(final_snr, 2)} (dB), Final MAE: {round(final_mae, 3)}\n")
            f.write("---------------------------------------------------------\n")
    multihost_utils.sync_global_devices("post_results_write")
    if not is_root_process():
        rank_print("Local solve finished.")
        return None
    return x_recon
    
if __name__ == "__main__":

    # # Uncomment the following code to downsample the full image and save the downsampled image.
    # import scico.numpy as snp
    # import os
    # img_path = "/global/cfs/projectdirs/m5278/repos/scico_2025/TIFF_delta_regularized_ram-lak_freqscl_1.00"
    # results_dir = "/global/cfs/projectdirs/m5278/repos/scico_2025/examples/scripts/results"
    # save_path = os.path.join(results_dir, "nanolaminography_ground_truth")
    # os.makedirs(save_path, exist_ok=True)
    # # Load the full image, with shape (768, 3840, 3840)
    # print("Loading the full image...")
    # x_gt = load_nanolaminography_image_from_tiff(img_path=img_path)
    # snp.save(os.path.join(save_path, 'tomo_volume.npy'), x_gt)
    # # Save the downsampled image
    # print("Downsampling the image...")
    # downsampling_factor = 2
    # x_gt_downsampled = image_downsampling(x_gt, downsampling_factor=downsampling_factor)
    # print("Shape of the downsampled image: ", x_gt_downsampled.shape)
    # save_path = os.path.join(results_dir, f"nanolaminography_ground_truth_downsampled_factor_{downsampling_factor}")
    # os.makedirs(save_path, exist_ok=True)
    # snp.save(os.path.join(save_path, 'tomo_volume_downsampled.npy'), x_gt_downsampled)
    # for i in range(1, 769):
    #     if i % 50 == 0:
    #         print(f"Processing slice {i}...")
    #     img_slice = x_gt_downsampled[i-1]
    #     plt.imshow(img_slice, cmap='gray')
    #     plt.axis('off')
    #     plt.savefig(os.path.join(save_path, f"tomo_delta_S00974_to_S03875_ram-lak_freqscl_1.00_downsampled_{i:04d}.png"), bbox_inches='tight', pad_inches=0)
    #     plt.close()
    # exit()

    # x_gt = load_saved_nanolaminography_image(downsample_factor=4)
    # x_gt = x_gt.astype(snp.float16)     # Convert to float32 to avoid overflow when plotting uint16 images (Now try float16 to save memory)
    # folder_name = "results/semi_pjadmm_parallel_tv_fbp_recon_nanolaminography_sinogram_snrinf_2_2_l2_plus_reg_iterative_reg_downsampled_factor_4"
    # file_name = "ct_mbirjax_3d_tv_semi_pjadmm_parallel_fbp_recon_nanolaminography_50views_ρ0.5_τ0.1_regularization2.0_maxiter50_maxiter_pdhg50_iterative_reg_estimated_pdhg"
    # x_recon = snp.load(os.path.join(os.path.dirname(__file__), f"{folder_name}/{file_name}.npy"))
    # print(f"Final SNR: {round(metric.snr(x_gt, x_recon), 2)} (dB), Final MAE: {round(metric.mae(x_gt, x_recon), 3)}")
    # exit()

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="3D TV-Regularized Sparse-View CT Reconstruction on NanoLaminography dataset, semi-proximal Jacobi ADMM solver, iterative regularization, estimated PDHG parameters",
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
    parser.add_argument('--downsample_factor', type=int, default=2,
                       help='Factor to downsample the image (default: 2)')
    parser.add_argument('--regularization_type', type=str, default="tv",
                       help='Regularization type (default: tv)')
    parser.add_argument('--sinogram_snr', type=str, default='30',
                       help='SNR of the sinogram in dB (default: 30). Use "inf" for no noise.')
    parser.add_argument('--maxiter_pdhg', type=int, default=50,
                       help='Number of iterations for PDHG (default: 50)')
    parser.add_argument('--tau_decrease', type=lambda x: bool(strtobool(x)), default=True,
                       help='Whether to decrease tau (default: True)')
    parser.add_argument('--data_root', type=str, default=os.environ.get("NANOLAMINOGRAPHY_DATA_ROOT"),
                       help='Directory containing nanolaminography_ground_truth[_downsampled_factor_N] data folders')
    parser.add_argument('--coordinator_address', type=str, default=os.environ.get("JAX_COORDINATOR_ADDRESS"),
                       help='Coordinator host:port for JAX distributed initialization')
    parser.add_argument('--num_processes', type=int, default=int(os.environ.get("SLURM_NTASKS", "1")),
                       help='Total number of JAX processes')
    parser.add_argument('--process_id', type=int, default=int(os.environ.get("SLURM_PROCID", "0")),
                       help='This process rank id')
    parser.add_argument('--local_device_ids', type=str, default=os.environ.get("JAX_LOCAL_DEVICE_IDS", "0,1,2,3"),
                       help='Comma-separated local GPU ids visible to this process')
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

    initialize_jax_distributed_if_needed(
        num_processes=args.num_processes,
        process_id=args.process_id,
        coordinator_address=args.coordinator_address,
        local_device_ids=args.local_device_ids,
    )
    load_runtime_modules()
    
    # Display configuration
    if is_root_process():
        print("="*100)
        print("3D TV-Regularized Sparse-View CT Reconstruction on NanoLaminography dataset, multi-node semi-proximal Jacobi ADMM solver, iterative regularization, estimated PDHG parameters")
        print("="*100)
        print(f"Configuration:")
        print(f"  Block division number: {args.row_division_num}x{args.col_division_num}")
        print(f"  Total blocks: {args.row_division_num * args.col_division_num}")
        print(f"  Number of projections: {args.n_projection}")
        print(f"  Number of iterations for block reconstruction: {args.maxiter}")
        print(f"  Regularization type: {args.regularization_type}")
        print(f"  SNR of the sinogram: {args.sinogram_snr}")
        print(f"  Downsample factor: {args.downsample_factor}")
        print(f"  JAX processes: {jax.process_count()}")
        print(f"  Local GPUs per process: {len(jax.local_devices(backend='gpu'))}")
        print("="*80)
    
    # Run the test
    if is_root_process():
        print("\n" + "="*80)
        print("TEST: Multi-node Parallel Semi-Proximal Jacobi ADMM solver with iterative regularization, estimated PDHG parameters test with FBP initial guess on Real Data")
        print("="*80)

    test_results = pjadmm_parallel_fbp_nanolaminography_test(
        row_division_num=args.row_division_num,
        col_division_num=args.col_division_num,
        rho=args.rho,
        tau=args.tau,
        regularization=args.regularization,
        n_projection=args.n_projection,
        maxiter=args.maxiter,
        downsample_factor=args.downsample_factor,
        regularization_type=args.regularization_type,
        maxiter_pdhg=args.maxiter_pdhg,
        tau_decrease=args.tau_decrease,
        sinogram_snr=args.sinogram_snr,
        data_root=args.data_root,
    )
    if is_root_process():
        print("\n✅ Test completed!")
