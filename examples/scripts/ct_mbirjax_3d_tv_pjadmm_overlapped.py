import numpy as np
import jax
import os

# Force GPU usage
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
jax.config.update('jax_platform_name', 'gpu')

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.optimize import ProxJacobiOverlappedADMM
from scico.util import device_info, create_roi_indices
from scico.functional import IsotropicTVNorm

import scipy.io
import argparse
import sys
from datetime import datetime

import matplotlib.pyplot as plt


'''
Test for reconstruction for full 3D CT image with 
naive way of dividing the image into blocks and reconstructing each block separately.
ADMM and MBIRJAX are used to reconstruct each block.
'''
def pjadmm_test(Nx=128, Ny=256, Nz=64, row_division_num=4, col_division_num=8, do_block_recon=True):
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
    tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    tangle = snp.array(jax.device_put(tangle, gpu_devices[0]))

    n_projection = 10  # number of projections
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

    row_start_indices_original = []
    col_start_indices_original = []
    row_end_indices_original = []
    col_end_indices_original = []

    overlap_size = 20

    for i in range(row_division_num):
        for j in range(col_division_num):
            roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
            roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns

            row_start_indices_original.append(roi_start_row)
            col_start_indices_original.append(roi_start_col)
            row_end_indices_original.append(roi_end_row)
            col_end_indices_original.append(roi_end_col)

            if i != 0:
                roi_start_row -= overlap_size
            if j != 0:
                roi_start_col -= overlap_size
            if i != row_division_num - 1:
                roi_end_row += overlap_size
            if j != col_division_num - 1:
                roi_end_col += overlap_size

            assert roi_start_row >= 0 and roi_start_col >= 0 and roi_end_row <= Nx and roi_end_col <= Ny

            row_start_indices.append(roi_start_row)
            col_start_indices.append(roi_start_col)
            row_end_indices.append(roi_end_row)
            col_end_indices.append(roi_end_col)

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

            g = IsotropicTVNorm(input_shape=A.input_shape, input_dtype=A.input_dtype)

            # Append the mbirjax projector and sinogram to the list
            A_list.append(A)
            g_list.append(g)

    print("row_start_indices: ", row_start_indices)
    print("col_start_indices: ", col_start_indices)
    print("row_end_indices: ", row_end_indices)
    print("col_end_indices: ", col_end_indices)

    if do_block_recon:
        """
        Set up problems and solvers.
        """
        # Generic parameters for all sub-block solvers
        # ρ = 10 / snp.linalg.norm(y.reshape(-1), ord=1)  # ADMM penalty parameter
        # τ = 0.1 * row_division_num * col_division_num * ρ * 2**16   # Proximal parameter

        ρ = 1e-6
        τ = 0.03
        tv_weight = 5
        γ = 1  # Damping parameter
        λ = snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype)  # Dual variable
        correction = False
        α = 0.8 if correction else None
        maxiter = 500  # number of decentralized ADMM iterations

        print(f"ρ: {ρ}, τ: {τ}, γ: {γ}, correction: {correction}, α: {α}, maxiter: {maxiter}")

        test_mode = True

        solver = ProxJacobiOverlappedADMM(
            A_list=A_list,
            g_list=g_list,
            ρ=ρ,
            y=y,
            τ=τ,
            γ=γ,
            λ=λ,
            x0_list=[snp.array(jax.device_put(A_list[i].T(y), gpu_devices[0])) for i in range(len(A_list))],
            display_period = 1,
            device = gpu_devices[0],
            with_correction = correction,
            α = α,
            maxiter = maxiter,
            itstat_options={"display": True, "period": 10},
            ground_truth = tangle,
            test_mode = test_mode,
            row_start_indices = row_start_indices,
            col_start_indices = col_start_indices,
            row_end_indices = row_end_indices,
            col_end_indices = col_end_indices,
            overlap_size = overlap_size,
            row_start_indices_original = row_start_indices_original,
            col_start_indices_original = col_start_indices_original,
            row_end_indices_original = row_end_indices_original,
            col_end_indices_original = col_end_indices_original,
            tv_weight = tv_weight
        )

        """
        Run the solver.
        """
        print(f"Solving on {device_info()}\n")
        tangle_recon_list, x_all, res_all = solver.solve()

        save_path = os.path.join(os.path.dirname(__file__), f'results/pjadmm_{row_division_num}_{col_division_num}')
        os.makedirs(save_path, exist_ok=True)
        snp.save(os.path.join(save_path, 'x_all.npy'), x_all)
        snp.save(os.path.join(save_path, 'res_all.npy'), res_all)

        iters = np.arange(len(res_all))
        plt.plot(iters, res_all, label='Residual', marker='o')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(save_path, 'res_all.pdf'))
    else:
        print("Loading the results from the file...")
        tangle_recon = np.load(os.path.join(os.path.dirname(__file__), f'results/pjadmm_{row_division_num}_{col_division_num}/x_all.npy'))
        print(tangle_recon.shape)
        tangle_recon = tangle_recon[100]
        
        tangle_recon_list = [tangle_recon[i] for i in range(row_division_num * col_division_num)]

        tangle_recon = np.zeros((Nz, Ny, Nx))

        for i in range(row_division_num):
            for j in range(col_division_num):
                roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
                roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
                tangle_recon[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row] = tangle_recon_list[i * col_division_num + j]

        print("residual: ", snp.linalg.norm((A_full @ tangle_recon - y).reshape(-1), ord=2))
    

    '''
    Reconstruct the full image
    '''
    tangle_recon = snp.zeros(tangle.shape)

    idx = 0
    for i in range(len(row_start_indices)):
        starting_row_ind = overlap_size if row_start_indices_original[i] != row_start_indices[i] else 0
        starting_col_ind = overlap_size if col_start_indices_original[i] != col_start_indices[i] else 0
        ending_row_ind = -overlap_size if row_end_indices_original[i] != row_end_indices[i] else tangle.shape[2]
        ending_col_ind = -overlap_size if col_end_indices_original[i] != col_end_indices[i] else tangle.shape[1]

        tangle_recon = tangle_recon.at[:, col_start_indices_original[i]:col_end_indices_original[i], row_start_indices_original[i]:row_end_indices_original[i]].set(tangle_recon_list[idx][:, starting_col_ind:ending_col_ind, starting_row_ind:ending_row_ind])
        idx += 1


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

    results_dir = os.path.join(os.path.dirname(__file__), f'results/pjadmm_overlapped_{row_division_num}_{col_division_num}')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_pjadmm_overlapped_recon_{n_projection}views_overlap{overlap_size}.png')
    fig.savefig(save_path)   # save the figure to file

    return tangle_recon



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="3D TV-Regularized Sparse-View CT Reconstruction with Proximal Jacobi Overlapped ADMM using MBIRJAX",
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
    parser.add_argument('--do_block_recon', action='store_true', default=True,
                    help='Perform block reconstruction (default: enabled)')
    parser.add_argument('--no_block_recon', action='store_false', dest='do_block_recon',
                    help='Disable block reconstruction')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate parameters
    if args.Nx <= 0 or args.Ny <= 0 or args.Nz <= 0:
        parser.error("Image dimensions must be positive integers")
    
    if args.row_division <= 0 or args.col_division <= 0:
        parser.error("Division numbers must be positive integers")
    
    # Display configuration
    print("="*100)
    print("3D TV-Regularized Sparse-View CT Reconstruction (Proximal Jacobi Overlapped ADMM Solver) using MBIRJAX")
    print("="*100)
    print(f"Configuration:")
    print(f"  Image dimensions: {args.Nx}x{args.Ny}x{args.Nz}")
    print(f"  Block division: {args.row_division}x{args.col_division}")
    print(f"  Block reconstruction: {args.do_block_recon}")
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
        do_block_recon=args.do_block_recon
    )
    
    print("\n✅ Test completed!")