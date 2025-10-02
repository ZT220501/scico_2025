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
from scico.optimize import ProxJacobiADMM
from scico.util import device_info, create_roi_indices
from scico.functional import IsotropicTVNorm, L1Norm

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
def pjadmm_test(
    Nx=128, Ny=256, Nz=64, row_division_num=2, col_division_num=2, do_block_recon=True, rho=1e-3, tau=0.1, tv_weight_second_stage=8e-3,
    tv_weight_first_stage=1e-4, n_projection=30
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

    print("Creating mbirjax projectors for each ROI...")
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
    Set up problems and solvers for tv regularized solver, first stage.
    """
    ρ = 1e-4
    τ = 0.03
    tv_weight = 1

    γ = 1  # Damping parameter
    λ = snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype)  # Dual variable
    correction = False
    α = 0.8 if correction else None
    maxiter = 100  # number of decentralized ADMM iterations

    print(f"ρ: {ρ}, τ: {τ}, regularization: {tv_weight}, γ: {γ}, correction: {correction}, α: {α}, maxiter: {maxiter}")

    test_mode = True

    # Set up the tv regularizer for each ROI as the new g_list.
    g_list = [IsotropicTVNorm(input_shape=A_list[i].input_shape, input_dtype=A_list[i].input_dtype) for i in range(len(A_list))]

    # Use the result of l1 regularized solver as the initial guess for tv regularized solver.
    tv_solver = ProxJacobiADMM(
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
        row_division_num = row_division_num,
        col_division_num = col_division_num,
        tv_weight = tv_weight
    )

    """
    Run the tv regularized solver.
    """
    print(f"Solving on {device_info()}\n")
    tangle_recon_list = tv_solver.solve()



    """
    Set up problems and solvers for tv regularized solver, second stage.
    """
    ################################################################################################################
    # # This is the best setting for the problem of size 128*256*64.
    # In the Proximal Jacobi ADMM solver, tau is updated as 
    # if lower_bound < 0:
    #     print("τ is doubled at iteration ", self.itnum)
    #     self.τ = self.τ * 2

    #     # Revert back the variables.
    #     self.x_list = self.x_list_prev
    #     self.λ = self.λ + self.γ * self.ρ * (sum(self.A_list[i](self.x_list[i]) for i in range(self.N)) - self.y)
    #     self.λ = jax.device_put(self.λ, self.device)
    #     # Revert back the stored variables.
    #     self.x_list = self.x_list_prev.copy()
    #     self.res = self.res_prev.copy()
    #     self.x_list_prev = self.x_list_two_prev.copy()
    #     self.res_prev = self.res_two_prev.copy()
    # elif self.itnum % 10 == 0:
    #     # Decrase τ after every a pre-defined number of iterations.
    #     self.τ = self.τ / 1.2

    # ρ = 1e-3
    # τ = 0.1
    # tv_weight = 8e-3
    ################################################################################################################


    # TODO: Try Parameter Tuning for TV-Regularized Abel Inversion in scico.
    ρ = rho
    τ = tau
    tv_weight = tv_weight_second_stage

    
    γ = 1  # Damping parameter
    λ = snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype)  # Dual variable
    correction = False
    α = 0.8 if correction else None
    maxiter = 1900  # number of decentralized ADMM iterations

    print(f"ρ: {ρ}, τ: {τ}, regularization: {tv_weight}, γ: {γ}, correction: {correction}, α: {α}, maxiter: {maxiter}")

    test_mode = True

    tv_solver_2 = ProxJacobiADMM(
        A_list=A_list,
        g_list=g_list,
        ρ=ρ,
        y=y,
        τ=τ,
        γ=γ,
        λ=λ,
        x0_list=tangle_recon_list,
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
    Run the l1 regularized solver.
    """
    print(f"Solving on {device_info()}\n")
    tangle_recon_list = tv_solver_2.solve()

    # save_path = os.path.join(os.path.dirname(__file__), f'results/pjadmm_tv_2stages_{row_division_num}_{col_division_num}')
    # os.makedirs(save_path, exist_ok=True)
    # snp.save(os.path.join(save_path, 'x_all.npy'), x_all)
    # snp.save(os.path.join(save_path, 'res_all.npy'), res_all)

    # iters = np.arange(len(res_all))
    # plt.plot(iters, res_all, label='Residual', marker='o')
    # plt.legend()
    # plt.show()
    # plt.savefig(os.path.join(save_path, 'res_all.pdf'))

    




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

    results_dir = os.path.join(os.path.dirname(__file__), f'results/pjadmm_tv_2stages_{row_division_num}_{col_division_num}')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_pjadmm_2stages_recon_{n_projection}views_{Nx}x{Ny}x{Nz}_foam_ρ{ρ}_τ{τ}_tv_weight{tv_weight}.png')
    fig.savefig(save_path)   # save the figure to file


    print(f"Final SNR: {round(metric.snr(tangle, tangle_recon), 2)} (dB), Final MAE: {round(metric.mae(tangle, tangle_recon), 3)}")

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
    parser.add_argument('--rho', type=float, default=1e-3,
                       help='Regularization parameter (default: 1e-3)')
    parser.add_argument('--tau', type=float, default=0.1,
                       help='Step size parameter (default: 0.1)')
    parser.add_argument('--tv_weight_second_stage', type=float, default=8e-3,
                       help='TV regularization weight (default: 8e-3)')
    parser.add_argument('--n_projection', type=int, default=30,
                       help='Number of projections (default: 30)')
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
    print(f"  Number of projections: {args.n_projection}")
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
        do_block_recon=args.do_block_recon,
        rho=args.rho,
        tau=args.tau,
        tv_weight_second_stage=args.tv_weight_second_stage,
        n_projection=args.n_projection
    )
    print("\n✅ Test completed!")