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

In this example the problem is solved via proximal ADMM, while
ADMM is used in a [companion example](ct_mbirjax_3d_tv_admm.rst).
"""

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom, create_3d_foam_phantom
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize import ProximalADMM
from scico.util import device_info

from concurrent.futures import ProcessPoolExecutor
import jax

import os
import scipy.io
import argparse
import psutil

# Potential useful function for monitoring memory usage in multi-GPU reconstruction
def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent()     # Memory usage as percentage
    }

def log_memory_usage(block_id, device_id, stage="start"):
    """Log memory usage at different stages."""
    memory = get_memory_usage()
    print(f"Block {block_id} (GPU {device_id}) - {stage}: "
          f"RSS: {memory['rss']:.1f}MB, "
          f"VMS: {memory['vms']:.1f}MB, "
          f"Percent: {memory['percent']:.1f}%")
    return memory

# Potential useful function for multi-GPU reconstruction
def process_block_on_device(block_data, device_id):
    """
    Process a single block on a specific device.
    
    Args:
        block_data: Dictionary containing block information
        device_id: Device ID to use
    
    Returns:
        Dictionary with reconstruction results
    """
    # Set the device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    # Log initial memory usage
    initial_memory = log_memory_usage(block_data['block_id'], device_id, "start")
    
    try:
        # Your block processing logic here
        # This is a placeholder - replace with actual reconstruction code
        
        # Log memory after processing
        final_memory = log_memory_usage(block_data['block_id'], device_id, "end")
        
        result = {
            'block_id': block_data['block_id'],
            'device_id': device_id,
            'status': 'completed',
            'memory_usage': {
                'initial': initial_memory,
                'final': final_memory,
                'peak': get_memory_usage()  # Current peak usage
            }
        }
        return result
    except Exception as e:
        error_memory = log_memory_usage(block_data['block_id'], device_id, "error")
        return {
            'block_id': block_data['block_id'],
            'device_id': device_id,
            'status': 'failed',
            'error': str(e),
            'memory_usage': {
                'initial': initial_memory,
                'error': error_memory
            }
        }

# Potential useful function for multi-GPU reconstruction
def multi_gpu_block_reconstruction(Nx=128, Ny=256, Nz=64, row_division_num=4, col_division_num=8, 
                                  gpu_ids=[0, 1, 2, 3], max_workers=None):
    """
    Distribute block reconstruction across multiple GPUs.
    
    Args:
        Nx, Ny, Nz: Image dimensions
        row_division_num, col_division_num: Block divisions
        gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3]), default is [0, 1, 2, 3]
        max_workers: Maximum number of parallel workers
    """
    if gpu_ids is None:
        gpu_ids = list(range(len(jax.devices())))
    
    if max_workers is None:
        max_workers = len(gpu_ids)
    
    # Create block data
    blocks = []
    for i in range(row_division_num):
        for j in range(col_division_num):
            block_id = i * col_division_num + j
            roi_start_row = i * Nx // row_division_num
            roi_end_row = (i + 1) * Nx // row_division_num
            roi_start_col = j * Ny // col_division_num
            roi_end_col = (j + 1) * Ny // col_division_num
            
            blocks.append({
                'block_id': block_id,
                'i': i, 'j': j,
                'roi_start_row': roi_start_row,
                'roi_end_row': roi_end_row,
                'roi_start_col': roi_start_col,
                'roi_end_col': roi_end_col
            })
    
    print(f"Processing {len(blocks)} blocks on {len(gpu_ids)} GPUs")
    print(f"Available devices: {jax.devices()}")
    
    # Distribute blocks across GPUs
    results = []
    
    # Start memory monitoring
    print("\n" + "="*60)
    print("MEMORY MONITORING")
    print("="*60)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_block = {}
        for block in blocks:
            device_id = block['block_id'] % len(gpu_ids)
            future = executor.submit(process_block_on_device, block, gpu_ids[device_id])
            future_to_block[future] = block
        
        # Monitor and collect results
        completed_blocks = 0
        for future in future_to_block:
            try:
                result = future.result()
                results.append(result)
                completed_blocks += 1
                
                # Display memory usage for completed block
                if 'memory_usage' in result:
                    mem = result['memory_usage']
                    print(f"\nBlock {result['block_id']} (GPU {result['device_id']}) completed:")
                    print(f"  Initial: {mem['initial']['rss']:.1f}MB RSS, {mem['initial']['vms']:.1f}MB VMS")
                    print(f"  Final:   {mem['final']['rss']:.1f}MB RSS, {mem['final']['vms']:.1f}MB VMS")
                    print(f"  Peak:    {mem['peak']['rss']:.1f}MB RSS, {mem['peak']['vms']:.1f}MB VMS")
                
                # Show progress
                print(f"Progress: {completed_blocks}/{len(blocks)} blocks completed")
                
            except Exception as e:
                print(f"Block processing failed: {e}")
    
    # Summary of memory usage
    print("\n" + "="*60)
    print("MEMORY USAGE SUMMARY")
    print("="*60)
    
    total_initial_rss = sum(r['memory_usage']['initial']['rss'] for r in results if 'memory_usage' in r)
    total_final_rss = sum(r['memory_usage']['final']['rss'] for r in results if 'memory_usage' in r)
    total_peak_rss = sum(r['memory_usage']['peak']['rss'] for r in results if 'memory_usage' in r)
    
    print(f"Total Initial RSS: {total_initial_rss:.1f}MB")
    print(f"Total Final RSS:   {total_final_rss:.1f}MB")
    print(f"Total Peak RSS:    {total_peak_rss:.1f}MB")
    
    return results

def create_roi_indices(Nx, Ny, roi_start_row, roi_end_row, roi_start_col, roi_end_col, display=False):
    '''
    Create ROI indices for a row and column range.
    
    Args:
        Nx: Image width
        Ny: Image height
        roi_start_row: Start row index of the ROI
        roi_end_row: End row index of the ROI
        roi_start_col: Start column index of the ROI
        roi_end_col: End column index of the ROI
        display: Whether to display the ROI indices, default is False
    
    Returns:
        roi_indices: ROI indices
    '''
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
Test for reconstruction for full 3D CT image with 
naive way of dividing the image into blocks and reconstructing each block separately.
Proximal ADMM and MBIRJAX are used to reconstruct each block.
'''
def simple_block_padmm_test(Nx=128, Ny=256, Nz=64, row_division_num=4, col_division_num=8, n_projection=10, do_block_recon=True, device_ids=None):
    '''
    Create a ground truth image and projector.
    
    Args:
        do_block_recon: Whether to perform block reconstruction
        Nx: Image width
        Ny: Image height  
        Nz: Image depth
        row_division_num: Number of row divisions
        col_division_num: Number of column divisions
        device_ids: List of device IDs to use for each block (e.g., [0, 1, 2, 3])
                   If None, uses default device for all blocks
    '''
    # Create a full 3D CT image phantom
    # tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    print("Creating 3D foam phantom...")
    tangle = snp.array(create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=100)) # 3D foam phantom; notice that the order of the dimensions follows the scico convention (Nz, Ny, Nx)
    print("3D foam phantom created.")


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
        𝛼 = 1e2  # improve problem conditioning by balancing C and D components of A
        λ = 2e0 / 𝛼  # ℓ2,1 norm regularization parameter
        ρ = 5e-3  # ADMM penalty parameter
        maxiter = 1000  # number of ADMM iterations
        f = functional.ZeroFunctional()
        g1 = λ * functional.L21Norm()

        # Specific parameters for each sub-block solver
        # The append=0 option makes the results of horizontal and vertical
        # finite differences the same shape, which is required for the L21Norm,
        # which is used so that g(Ax) corresponds to isotropic TV.
        for i in range(len(C_list)):
            C = C_list[i]
            # y = y_list[i]
            C_full = XRayTransformParallel(
                output_shape=C.output_shape,
                angles=angles,
                recon_shape=(Nx, Ny, Nz)
            )
            y = C_full @ tangle

            g0 = loss.SquaredL2Loss(y=y)
            g = functional.SeparableFunctional((g0, g1))
            D = linop.FiniteDifference(input_shape=C.input_shape, append=0)
            A = linop.VerticalStack((C, 𝛼 * D))

            mu, nu = ProximalADMM.estimate_parameters(A)

            solver = ProximalADMM(
                f=f,
                g=g,
                A=A,
                B=None,
                rho=ρ,
                mu=mu,
                nu=nu,
                maxiter=maxiter,
                itstat_options={"display": True, "period": 50},
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

            results_dir = os.path.join(os.path.dirname(__file__), f'results/naive_block_padmm_{row_division_num}_{col_division_num}')
            os.makedirs(results_dir, exist_ok=True)
            # save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_padmm_naive_block_{i}_{n_projection}views.png')
            save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_padmm_foam_block_{i}_{n_projection}views.png')
            fig.savefig(save_path)   # save the figure to file

            # Save the reconstructed block to a .mat file in case of future full reconstruction
            # scipy.io.savemat(os.path.join(results_dir, f'ct_mbirjax_3d_tv_padmm_naive_block_{i}_{n_projection}views.mat'), {'array': tangle_recon_roi})
            scipy.io.savemat(os.path.join(results_dir, f'ct_mbirjax_3d_tv_padmm_foam_block_{i}_{n_projection}views.mat'), {'array': tangle_recon_roi})


    # Manually read the reconstructed blocks from the .mat files
    block_recon_dir = os.path.join(os.path.dirname(__file__), f'results/naive_block_padmm_{row_division_num}_{col_division_num}')

    tangle_recon_list = []
    for idx in range(row_division_num * col_division_num):
        # block_recon_image = scipy.io.loadmat(os.path.join(block_recon_dir, f'ct_mbirjax_3d_tv_padmm_naive_block_{idx}_{n_projection}views.mat'))['array']
        block_recon_image = scipy.io.loadmat(os.path.join(block_recon_dir, f'ct_mbirjax_3d_tv_padmm_foam_block_{idx}_{n_projection}views.mat'))['array']
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

    results_dir = os.path.join(os.path.dirname(__file__), f'results/naive_block_padmm_{row_division_num}_{col_division_num}')
    os.makedirs(results_dir, exist_ok=True)
    # save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_admm_naive_recon_{n_projection}views.png')
    save_path = os.path.join(results_dir, f'ct_mbirjax_3d_tv_padmm_foam_recon_{n_projection}views.png')
    fig.savefig(save_path)   # save the figure to file





if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="3D TV-Regularized Sparse-View CT Reconstruction with Proximal ADMM using MBIRJAX",
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
    parser.add_argument('--n_projection', type=int, default=10,
                       help='Number of projections (default: 10)')
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
    print("3D TV-Regularized Sparse-View CT Reconstruction (Proximal ADMM Solver) using MBIRJAX")
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
    print("TEST: Simple Block Proximal ADMM test")
    print("="*80)
    
    test_results = simple_block_padmm_test(
        Nx=args.Nx,
        Ny=args.Ny,
        Nz=args.Nz,
        row_division_num=args.row_division,
        col_division_num=args.col_division,
        n_projection=args.n_projection,
        do_block_recon=args.do_block_recon
    )
    
    print("\n✅ Test completed successfully!")