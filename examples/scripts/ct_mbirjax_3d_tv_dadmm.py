import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.optimize import DecentralizedADMM
from scico.util import device_info, create_roi_indices

import os
import scipy.io
import argparse


'''
Test for reconstruction for full 3D CT image with 
naive way of dividing the image into blocks and reconstructing each block separately.
ADMM and MBIRJAX are used to reconstruct each block.
'''
def dadmm_test(Nx=128, Ny=256, Nz=64, row_division_num=4, col_division_num=8, do_block_recon=True):
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
    A_list = [] # List of mbirjax projectors for each ROI
    D_list = [] # List of finite difference operators for each ROI

    if do_block_recon:
        for i in range(row_division_num):
            for j in range(col_division_num):
                roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
                roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
                roi_indices = create_roi_indices(Nx, Ny, roi_start_row, roi_end_row, roi_start_col, roi_end_col)

                # Create sinogram shape
                sinogram_shape = (Nz, n_projection, max(Nx, Ny))

                # Create the mbirjax projector for the current ROI
                A = XRayTransformParallel(
                    output_shape=sinogram_shape,
                    angles=angles,
                    partial_reconstruction=True,
                    roi_indices=roi_indices,
                    roi_recon_shape=(roi_end_row - roi_start_row, roi_end_col - roi_start_col, Nz),
                    recon_shape=(Nx, Ny, Nz)
                )

                D = linop.FiniteDifference(input_shape=A.input_shape, append=0)

                # Append the mbirjax projector and sinogram to the list
                A_list.append(A)
                D_list.append(D)

        # Create the full sinogram
        # NOTE: In the real-world application, only the full sinogram is available. We can not get the sinogram for each ROI.
        A_full = XRayTransformParallel(
            output_shape=sinogram_shape,
            angles=angles,
            recon_shape=(Nx, Ny, Nz)
        )
        y = A_full @ tangle

        """
        Set up problems and solvers.
        """
        # Generic parameters for all sub-block solvers
        λ = 2e0  # ℓ2,1 norm regularization parameter
        ρ = 5e0  # ADMM penalty parameter
        maxiter = 1  # number of decentralized ADMM iterations
        maxiter_per_block = 1  # number of ADMM iterations for each block
        cg_tol = 1e-4  # CG relative tolerance
        cg_maxiter = 25  # maximum CG iterations per ADMM iteration
        g = λ * functional.L21Norm()

        solver = DecentralizedADMM(
            A_list=A_list,
            g_list=[g],
            D_list=D_list,
            rho_list=[ρ],
            y=y,
            x0_list=[A_list[i].T(y) for i in range(len(A_list))],
            subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
            # itstat_options={"display": True, "period": 5}
            maxiter_per_block = maxiter_per_block,
            maxiter = maxiter
        )

        """
        Run the solver.
        """
        print(f"Solving on {device_info()}\n")
        tangle_recon_list = solver.solve()
        # hist = solver.itstat_object.history(transpose=True)

        # Reconstruct the full image
        tangle_recon = np.zeros((Nz, Ny, Nx))
        print("tangle_recon_list first and second element are equal: ", np.allclose(tangle_recon_list[0], tangle_recon_list[1]))
        print("tangle_recon_list first and third element are equal: ", np.allclose(tangle_recon_list[0], tangle_recon_list[2]))
        print("max difference between first and second element: ", np.max(np.abs(tangle_recon_list[0] - tangle_recon_list[1])))
        print("max difference between first and third element: ", np.max(np.abs(tangle_recon_list[0] - tangle_recon_list[2])))
        for i in range(row_division_num):
            for j in range(col_division_num):
                roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
                roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns
                tangle_recon[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row] = tangle_recon_list[i * col_division_num + j]

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

        results_dir = os.path.join(os.path.dirname(__file__), f'results/dadmm_{row_division_num}_{col_division_num}')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_dadmm_recon_{n_projection}views.png')
        fig.savefig(save_path)   # save the figure to file

        return tangle_recon



# TODO: Test the decentralized ADMM solver
if __name__ == "__main__":
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
    
    test_results = dadmm_test(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        row_division_num=args.row_division,
        col_division_num=args.col_division,
        do_block_recon=args.do_block_recon
    )
    
    print("\n✅ Test completed!")