#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
3D TV-Regularized Sparse-View CT Reconstruction (Block ADMM Solver)
===================================================================

This example demonstrates solution of a sparse-view, 3D CT
reconstruction problem with isotropic total variation (TV)
regularization using block ADMM

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - C \mathbf{x}
  \|_2^2 + \lambda \| D \mathbf{x} \|_{2,1} \;,$$

where $C$ is the X-ray transform (the CT forward projection operator),
$\mathbf{y}$ is the sinogram, $D$ is a 3D finite difference operator,
and $\mathbf{x}$ is the reconstructed image.

The problem is solved via block ADMM by dividing the image into
overlapping blocks and solving subproblems on each block.
"""

import argparse
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
# from scico import numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom, create_3d_foam_phantom
from scico.linop.xray.astra import XRayTransform3D, SparseXRayTransform3D
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

from jax.experimental.sparse import BCOO
import torch
from mbirjax.parallel_beam import ParallelBeamModel
import mbirjax

import os

def test_xray_transform_parallel_proj():
    """
    Test the _proj function in XRayTransformParallel to ensure it's identical 
    to mbirjax's forward projection, with pixel_indices corresponding to the 
    correct pixels in the original phantom.
    """
    print("=" * 60)
    print("Testing XRayTransformParallel._proj function")
    print("=" * 60)
    
    # Setup parameters
    Nx = 256
    Ny = 256
    Nz = 256
    num_angles = 30
    
    # Create angles
    angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)
    
    # Create test phantom
    # test_phantom = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    test_phantom = snp.array(create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=10))
    print(f"Test phantom shape: {test_phantom.shape}")
    
    # Create sinogram shape
    sinogram_shape = (Nz, num_angles, max(Nx, Ny))
    print(f"Sinogram shape: {sinogram_shape}")
    
    # Step 1: Create XRayTransformParallel instance
    print("\n1. Creating XRayTransformParallel instance...")
    from scico.linop.xray.mbirjax import XRayTransformParallel
    
    xray_transform = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        recon_shape=(Nx, Ny, Nz)  # Set reconstruction shape, with mbirjax convention
    )
    print(f"XRayTransformParallel input shape: {xray_transform.input_shape}")
    print(f"XRayTransformParallel output shape: {xray_transform.output_shape}")
    
    # Step 2: Get mbirjax model and parameters
    print("\n2. Getting mbirjax model and parameters...")
    mbirjax_model = xray_transform.model
    projector_params = xray_transform.get_params()
    print(f"Projector params: {projector_params}")
    
    # Step 3: Generate full indices (equivalent to all pixels)
    print("\n3. Generating full indices...")
    full_indices = mbirjax.gen_full_indices(mbirjax_model.get_params("recon_shape"), use_ror_mask=False)
    print(f"Full indices shape: {full_indices.shape}")
    print(f"Number of indices: {len(full_indices)}")
    print(f"Expected number: {Nx * Ny}")  # Should be Nx * Ny
    
    # Step 4: Reshape phantom to match expected format
    print("\n4. Reshaping phantom for _proj function...")
    # The _proj function expects x to be reshaped to (-1, x.shape[-1])
    # This means (Nx, Ny, Nz) -> (Nx*Ny, Nz)
    # phantom_reshaped = test_phantom_transposed.reshape(-1, test_phantom_transposed.shape[-1])
    # print(f"Reshaped phantom shape: {phantom_reshaped.shape}")
    print(f"Test phantom shape: {test_phantom.shape}")
    print(f"Expected shape: ({Nx * Ny}, {Nz})")
    
    # Step 5: Test _proj function
    print("\n5. Testing _proj function...")
    try:
        proj_sinogram = xray_transform @ test_phantom
        print(f"_proj sinogram shape: {proj_sinogram.shape}")
        print(f"Expected shape: {sinogram_shape}")

        print("\n6. Comparing with scico forward projection...")
        C = XRayTransform3D(
            test_phantom.shape, det_count=[Nz, max(Nx, Ny)], det_spacing=[1.0, 1.0], angles=angles
        )
        scico_sinogram = C @ test_phantom
        print(f"SCICO sinogram shape: {scico_sinogram.shape}")

        # Step 7: Calculate differences
        print("\n7. Calculating differences...")

        diff = np.abs(proj_sinogram - scico_sinogram)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_diff = max_diff / (np.max(np.abs(scico_sinogram)) + 1e-10)
        mse = np.mean(diff**2) * (1 / (Nx * Ny * Nz))
        
        print(f"Max value of proj_sinogram in absolute value: {np.max(np.abs(proj_sinogram))}")
        print(f"Maximum absolute difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        print(f"Relative difference: {rel_diff:.2e}")
        print(f"MSE: {mse:.2e}")

        # Step 8: Verify pixel correspondence
        print("\n8. Verifying pixel correspondence...")
        
        # Check that full_indices covers all pixels
        expected_indices = snp.arange(Nx * Ny)
        actual_indices = full_indices
        
        if snp.all(expected_indices == actual_indices):
            print("✅ Pixel indices correctly cover all pixels")
        else:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            print(f"❌ Pixel index mismatch:")
            print(f"   Missing indices: {len(missing)}")
            print(f"   Extra indices: {len(extra)}")
        
        # Step 9: Overall assessment
        print("\n9. Overall assessment...")
        if rel_diff < 0.01:
            print("✅  Test succeeded: Results are similar")
            print("   This might be due to numerical precision differences")
        else:
            print("❌ Test FAILED: Results differ significantly")

        # Step 10: Visualize the results
        print("\n10. Visualizing the results...")
        fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 2))
        plot.imview(
            proj_sinogram[::-1, 10, ::-1],
            title="Sinogram with wrapper mbirjax @ operator",
            cmap=plot.cm.Blues,
            cbar=None,
            fig=fig,
            ax=ax[0],
        )
        plot.imview(
            scico_sinogram[:, 10, :],
            title="Sinogram with scico @ operator",
            cmap=plot.cm.Blues,
            cbar=None,
            fig=fig,
            ax=ax[1],
        )
        fig.show()

        results_dir = os.path.join(os.path.dirname(__file__), f'results/XRayTransformParallel_proj_test')
        os.makedirs(results_dir, exist_ok=True)
        # save_path = os.path.join(results_dir, f'XRayTransformParallel_proj_test_{Nx}x{Ny}x{Nz}_foam_100_spheres.png')
        save_path = os.path.join(results_dir, f'XRayTransformParallel_proj_test_{Nx}x{Ny}x{Nz}_tangle_phantom.png')
        fig.savefig(save_path)   # save the figure to file

        
        return {
            'xray_transform': xray_transform,
            'mbirjax_model': mbirjax_model,
            'test_phantom': test_phantom,
            'full_indices': full_indices,
            'proj_sinogram': proj_sinogram,
            'scico_sinogram': scico_sinogram,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'rel_diff': rel_diff
        }
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_xray_transform_parallel_proj_partial():
    """
    Test the _proj function with partial reconstruction (ROI).
    """
    print("=" * 60)
    print("Testing XRayTransformParallel._proj with partial reconstruction")
    print("=" * 60)
    
    # Setup parameters
    Nx = 64
    Ny = 128
    Nz = 32
    num_angles = 30
    
    # Create angles
    angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)
    
    # Create test phantom
    # test_phantom = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    test_phantom = snp.array(create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=10))
    
    # Define ROI (Region of Interest)
    roi_start_row, roi_end_row = 24, 40  # Middle 16 rows
    roi_start_col, roi_end_col = 32, 96  # Middle 64 columns
    
    # Create ROI indices
    roi_rows = np.arange(roi_start_row, roi_end_row)
    roi_cols = np.arange(roi_start_col, roi_end_col)
    roi_row_grid, roi_col_grid = np.meshgrid(roi_rows, roi_cols, indexing='ij')
    roi_indices = np.ravel_multi_index((roi_row_grid.flatten(), roi_col_grid.flatten()), (Nx, Ny))
    
    print(f"ROI indices shape: {roi_indices.shape}")
    print(f"ROI coverage: {len(roi_indices)} / {Nx * Ny} = {len(roi_indices) / (Nx * Ny) * 100:.1f}%")
    
    # Create sinogram shape
    sinogram_shape = (Nz, num_angles, max(Nx, Ny))
    
    # Method 1: Direct ROI projection using XRayTransformParallel with partial reconstruction
    print("\n1. Creating direct ROI projection using partial reconstruction...")
    xray_transform_roi = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        partial_reconstruction=True,
        roi_indices=roi_indices,
        roi_recon_shape=(len(roi_rows), len(roi_cols), Nz),
        recon_shape=(Nx, Ny, Nz)
    )
    
    # Extract ROI phantom
    roi_phantom = test_phantom[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row]
    print(f"ROI phantom shape: {roi_phantom.shape}")
    
    # Project ROI directly
    roi_proj_sinogram = xray_transform_roi @ roi_phantom
    print(f"Direct ROI projection shape: {roi_proj_sinogram.shape}")
    
    # Method 2: Zero-padded ROI projection (should give same result)
    print("\n2. Creating zero-padded ROI projection for comparison...")
    
    # Create zero-padded phantom with ROI in correct position
    zero_padded_phantom = np.zeros_like(test_phantom)
    zero_padded_phantom[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row] = roi_phantom
    
    # Create full projection operator
    C_full = XRayTransform3D(
        test_phantom.shape, 
        det_count=[Nz, max(Nx, Ny)], 
        det_spacing=[1.0, 1.0], 
        angles=angles
    )
    
    # Project zero-padded phantom
    zero_padded_sinogram = C_full @ zero_padded_phantom
    print(f"Zero-padded ROI projection shape: {zero_padded_sinogram.shape}")
    
    # Method 3: Full phantom projection for reference
    print("\n3. Creating full phantom projection for reference...")
    full_sinogram = C_full @ test_phantom
    print(f"Full phantom projection shape: {full_sinogram.shape}")
    
    # Calculate differences
    print("\n4. Calculating differences...")
    # Direct ROI vs Zero-padded ROI (should be very similar)
    roi_vs_zero_diff = np.abs(roi_proj_sinogram - zero_padded_sinogram)
    roi_vs_zero_max = np.max(roi_vs_zero_diff)
    roi_vs_zero_mean = np.mean(roi_vs_zero_diff)
    roi_vs_zero_rel = roi_vs_zero_max / (np.max(np.abs(zero_padded_sinogram)) + 1e-10)
    
    print(f"Direct ROI vs Zero-padded ROI:")
    print(f"  Max difference: {roi_vs_zero_max:.2e}")
    print(f"  Mean difference: {roi_vs_zero_mean:.2e}")
    print(f"  Relative difference: {roi_vs_zero_rel:.2e}")
    
    # Assessment
    print("\n5. Assessment...")
    
    # Check if direct ROI and zero-padded ROI projections are the same
    if roi_vs_zero_rel < 0.05:  # 5% relative difference threshold
        print("✅ Direct ROI and zero-padded ROI projections are the same (as expected)")
        print("   This confirms that ROI projection is working correctly!")
    else:
        print("❌ Direct ROI and zero-padded ROI projections differ significantly")
        print("   This indicates an issue with the ROI projection implementation")
    
    # Visual comparison
    print("\n6. Visual comparison...")
    for testing_slice in range(Nz):
        fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 2))
        plot.imview(
            roi_proj_sinogram[:, testing_slice, :],
            title="Sinogram, mbirjax ROI projection",
            cmap=plot.cm.Blues,
            cbar=None,
            fig=fig,
            ax=ax[0],
        )
        plot.imview(
            zero_padded_sinogram[:, testing_slice, :],
            title="Sinogram, zero-padded scico",
            cmap=plot.cm.Blues,
            cbar=None,
            fig=fig,
            ax=ax[1],
        )
        fig.show()

        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f'sinogram_comparison_proj_roi_{testing_slice}.png')
        fig.savefig(save_path)   # save the figure to file
    
    return {
        'xray_transform_roi': xray_transform_roi,
        'roi_indices': roi_indices,
        'roi_phantom': roi_phantom,
        'zero_padded_phantom': zero_padded_phantom,
        'roi_proj_sinogram': roi_proj_sinogram,
        'zero_padded_sinogram': zero_padded_sinogram,
        'roi_vs_zero_rel': roi_vs_zero_rel,
    }

def test_backward_projection_full():
    """
    Test the backward projection (adjoint) functionality using .T operator.
    This tests the full reconstruction case.
    """
    print("=" * 60)
    print("Testing XRayTransformParallel Backward Projection (Full)")
    print("=" * 60)
    
    # Setup parameters
    Nx = 64
    Ny = 128
    Nz = 32
    num_angles = 20
    
    # Create angles
    angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)
    
    # Create test phantom
    # test_phantom = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    test_phantom = snp.array(create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=10))
    print(f"Test phantom shape: {test_phantom.shape}")
    
    # Create sinogram shape
    sinogram_shape = (Nz, num_angles, max(Nx, Ny))
    print(f"Sinogram shape: {sinogram_shape}")
    
    # Step 1: Create XRayTransformParallel instance
    print("\n1. Creating XRayTransformParallel instance...")
    xray_transform = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        recon_shape=(Nx, Ny, Nz)
    )
    print(f"XRayTransformParallel input shape: {xray_transform.input_shape}")
    print(f"XRayTransformParallel output shape: {xray_transform.output_shape}")
    
    # Step 2: Perform forward projection to create test sinogram
    print("\n2. Performing forward projection to create test sinogram...")
    forward_sinogram = xray_transform @ test_phantom
    print(f"Forward sinogram shape: {forward_sinogram.shape}")
    
    # Step 3: Test backward projection using .T operator
    print("\n3. Testing backward projection using .T operator...")
    try:
        backward_projection = xray_transform.T @ forward_sinogram
        print(f"Backward projection shape: {backward_projection.shape}")
        print(f"Expected shape: {xray_transform.input_shape}")
        
        # Step 4: Compare with SCICO backward projection
        print("\n4. Comparing with SCICO backward projection...")
        C_scico = XRayTransform3D(
            test_phantom.shape, 
            det_count=[Nz, max(Nx, Ny)], 
            det_spacing=[1.0, 1.0], 
            angles=angles
        )
        
        # Forward project with SCICO to get sinogram in same format
        scico_backward = C_scico.T @ forward_sinogram
        print(f"SCICO backward projection shape: {scico_backward.shape}")
        
        # Step 5: Calculate differences
        print("\n5. Calculating differences...")
        diff = np.abs(backward_projection - scico_backward)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_diff = max_diff / (np.max(np.abs(scico_backward)) + 1e-10)
        
        print(f"Maximum absolute difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        print(f"Relative difference: {rel_diff:.2e}")
        
        # Step 6: Assessment
        print("\n6. Assessment...")
        if rel_diff < 0.01:
            print("✅ Backward projection test PASSED: Results are similar to SCICO")
        else:
            print("❌ Backward projection test FAILED: Results differ significantly")
        
        # Step 7: Visual comparison
        print("\n7. Visual comparison...")
        fig, ax = plot.subplots(nrows=2, ncols=2, figsize=(10, 8))
        
        # Middle slice for visualization
        slice_idx = Nz // 2
        
        plot.imview(
            test_phantom[:, :, slice_idx],
            title="Original Phantom (middle slice)",
            cmap=plot.cm.Blues,
            fig=fig,
            ax=ax[0, 0],
        )
        
        plot.imview(
            backward_projection[:, :, slice_idx],
            title="MBIRJAX Backward Projection",
            cmap=plot.cm.Blues,
            fig=fig,
            ax=ax[0, 1],
        )
        
        plot.imview(
            scico_backward[:, :, slice_idx],
            title="SCICO Backward Projection",
            cmap=plot.cm.Blues,
            fig=fig,
            ax=ax[1, 0],
        )
        
        plot.imview(
            diff[:, :, slice_idx],
            title="Difference (MBIRJAX - SCICO)",
            cmap=plot.cm.Reds,
            fig=fig,
            ax=ax[1, 1],
        )
        
        fig.tight_layout()
        fig.show()
        
        # Save results
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'backward_projection_test.png')
        fig.savefig(save_path)
        
        return {
            'xray_transform': xray_transform,
            'test_phantom': test_phantom,
            'forward_sinogram': forward_sinogram,
            'backward_projection': backward_projection,
            'scico_backward': scico_backward,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'rel_diff': rel_diff,
        }
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_backward_projection_roi():
    """
    Test the backward projection (adjoint) functionality for ROI reconstruction.
    This tests the partial reconstruction case.
    """
    print("=" * 60)
    print("Testing XRayTransformParallel Backward Projection (ROI)")
    print("=" * 60)
    
    # Setup parameters
    Nx = 64
    Ny = 128
    Nz = 32
    num_angles = 20
    
    # Create angles
    angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)
    
    # Create test phantom
    # test_phantom = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    test_phantom = snp.array(create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=10))
    
    # Define ROI (Region of Interest)
    roi_start_row, roi_end_row = 24, 40  # Middle 16 rows
    roi_start_col, roi_end_col = 32, 96  # Middle 64 columns
    
    # Create ROI indices
    roi_rows = np.arange(roi_start_row, roi_end_row)
    roi_cols = np.arange(roi_start_col, roi_end_col)
    roi_row_grid, roi_col_grid = np.meshgrid(roi_rows, roi_cols, indexing='ij')
    roi_indices = np.ravel_multi_index((roi_row_grid.flatten(), roi_col_grid.flatten()), (Nx, Ny))
    
    print(f"ROI indices shape: {roi_indices.shape}")
    print(f"ROI coverage: {len(roi_indices)} / {Nx * Ny} = {len(roi_indices) / (Nx * Ny) * 100:.1f}%")
    
    # Create sinogram shape
    sinogram_shape = (Nz, num_angles, max(Nx, Ny))
    
    # Step 1: Create ROI forward projection operator
    print("\n1. Creating ROI forward projection operator...")
    xray_transform_roi = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        partial_reconstruction=True,
        roi_indices=roi_indices,
        roi_recon_shape=(len(roi_rows), len(roi_cols), Nz),
        recon_shape=(Nx, Ny, Nz)
    )
    
    # Extract ROI phantom
    roi_phantom = test_phantom[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row]
    print(f"ROI phantom shape: {roi_phantom.shape}")
    
    # Step 2: Perform forward projection to create test sinogram
    print("\n2. Performing ROI forward projection...")
    roi_forward_sinogram = xray_transform_roi @ roi_phantom
    print(f"ROI forward sinogram shape: {roi_forward_sinogram.shape}")
    
    # Step 3: Test backward projection using .T operator
    print("\n3. Testing ROI backward projection using .T operator...")
    try:
        roi_backward_projection = xray_transform_roi.T @ roi_forward_sinogram
        print(f"ROI backward projection shape: {roi_backward_projection.shape}")
        print(f"Expected shape: {xray_transform_roi.input_shape}")
        
        # Step 4: Create zero-padded comparison
        print("\n4. Creating zero-padded comparison...")
        
        # Create full projection operator for comparison
        C_full = XRayTransform3D(
            test_phantom.shape, 
            det_count=[Nz, max(Nx, Ny)], 
            det_spacing=[1.0, 1.0], 
            angles=angles
        )
        
        # Forward and backward project with full operator
        full_backward = C_full.T @ roi_forward_sinogram
        print(f"Full backward projection shape: {full_backward.shape}")
        
        # Step 5: Calculate differences
        print("\n5. Calculating differences...")
        
        # ROI backward vs Full backward (should be similar in ROI region)
        roi_vs_full_diff = np.abs(roi_backward_projection - full_backward[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row])
        roi_vs_full_max = np.max(roi_vs_full_diff)
        roi_vs_full_mean = np.mean(roi_vs_full_diff)
        roi_vs_full_rel = roi_vs_full_max / (np.max(np.abs(full_backward)) + 1e-10)
        
        print(f"ROI vs Full backward projection:")
        print(f"  Max difference: {roi_vs_full_max:.2e}")
        print(f"  Mean difference: {roi_vs_full_mean:.2e}")
        print(f"  Relative difference: {roi_vs_full_rel:.2e}")

        
        # Step 6: Assessment
        print("\n6. Assessment...")
        if roi_vs_full_rel < 0.05:  # 5% relative difference threshold
            print("✅ ROI backward projection test PASSED: Results are similar to full projection")
        else:
            print("❌ ROI backward projection test FAILED: Results differ significantly")
        
        # Step 7: Visual comparison
        print("\n7. Visual comparison...")
        fig, ax = plot.subplots(nrows=2, ncols=2, figsize=(10, 8))
        
        # Middle slice for visualization
        slice_idx = Nz // 2
        
        plot.imview(
            roi_phantom[slice_idx, :, :],
            title="Original ROI Phantom (middle slice)",
            cmap=plot.cm.Blues,
            fig=fig,
            ax=ax[0, 0],
        )
        
        plot.imview(
            roi_backward_projection[slice_idx, :, :],
            title="ROI Backward Projection",
            cmap=plot.cm.Blues,
            fig=fig,
            ax=ax[0, 1],
        )
        
        plot.imview(
            full_backward[slice_idx, roi_start_col:roi_end_col, roi_start_row:roi_end_row],
            title="ROI From Full Backward Projection",
            cmap=plot.cm.Blues,
            fig=fig,
            ax=ax[1, 0],
        )
        
        plot.imview(
            roi_vs_full_diff[slice_idx, :, :],
            title="Difference (ROI - Full)",
            cmap=plot.cm.Reds,
            fig=fig,
            ax=ax[1, 1],
        )
        
        fig.tight_layout()
        fig.show()
        
        # Save results
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'roi_backward_projection_test.png')
        fig.savefig(save_path)
        
        return {
            'xray_transform_roi': xray_transform_roi,
            'roi_phantom': roi_phantom,
            'roi_forward_sinogram': roi_forward_sinogram,
            'roi_backward_projection': roi_backward_projection,
            'full_backward': full_backward,
            'roi_vs_full_rel': roi_vs_full_rel,
        }
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# Main execution block
if __name__ == "__main__":
    print("Starting XRayTransformParallel tests...")
    
    # Run the main forward projection test
    print("\n" + "="*80)
    print("TEST 1: Full reconstruction forward projection test")
    print("="*80)
    test_results_1 = test_xray_transform_parallel_proj()
    
    if test_results_1 is not None:
        print("\n✅ Test 1 completed successfully!")
    else:
        print("\n❌ Test 1 failed!")
    
    # Run the partial reconstruction forward projection test
    print("\n" + "="*80)
    print("TEST 2: Partial reconstruction forward projection test")
    print("="*80)
    test_results_2 = test_xray_transform_parallel_proj_partial()
    
    if test_results_2 is not None:
        print("\n✅ Test 2 completed successfully!")
        print(f"Summary: roi_vs_zero_rel={test_results_2['roi_vs_zero_rel']:.2e}")
    else:
        print("\n❌ Test 2 failed!")
    
    # Run the full backward projection test
    print("\n" + "="*80)
    print("TEST 3: Full reconstruction backward projection test")
    print("="*80)
    test_results_3 = test_backward_projection_full()
    
    if test_results_3 is not None:
        print("\n✅ Test 3 completed successfully!")
    else:
        print("\n❌ Test 3 failed!")
    
    # Run the ROI backward projection test
    print("\n" + "="*80)
    print("TEST 4: ROI backward projection test")
    print("="*80)
    test_results_4 = test_backward_projection_roi()
    
    if test_results_4 is not None:
        print("\n✅ Test 4 completed successfully!")
    else:
        print("\n❌ Test 4 failed!")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)





    # '''
    # Simple test for checking the mbirjax sinogram angle convention
    # '''

    # import scico.numpy as snp
    # from scico.linop.xray.svmbir import XRayTransform
    # import svmbir

    # # Setup parameters
    # Nx = 512
    # Ny = 256
    # Nz = 64

    # num_angles = 50
    # angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)

    # x_gt = snp.array(create_tangle_phantom(Nx, Ny, Nz))

    # # Create scico's svmbir projector. Angle is shifted to match the svmbir convention.
    # A_svmbir = XRayTransform(x_gt.shape, 0.5 * snp.pi - angles, max(Nx, Ny))
    # # A_svmbir = XRayTransform(x_gt.shape, angles, max(Nx, Ny))
    # A_scico = XRayTransform3D(
    #     x_gt.shape, det_count=[Nz, max(Nx, Ny)], det_spacing=[1.0, 1.0], angles=angles
    # )  # CT projection operator

    # # Generate sinogram using SCICO's interface
    # svmbir_sinogram = A_svmbir @ x_gt
    # scico_sinogram = A_scico @ x_gt


    # print("scico_sinogram shape: ", scico_sinogram.shape)
    # print("svmbir_sinogram shape: ", svmbir_sinogram.shape)


    # ct_model = ParallelBeamModel(svmbir_sinogram.shape, 0.5 * snp.pi - angles)
    # mbirjax_phantom = ct_model.gen_modified_3d_sl_phantom()
    # print(f"mbirjax phantom shape: {mbirjax_phantom.shape}")
    # # ct_model = ParallelBeamModel(svmbir_sinogram.shape, angles)
    # mbirjax_sinogram = ct_model.forward_project(x_gt)

    # print("mbirjax_sinogram shape: ", mbirjax_sinogram.shape)

    # diff = np.abs(mbirjax_sinogram - svmbir_sinogram)
    # print("max diff between mbirjax and svmbir: ", np.max(diff))

    # diff = np.abs(scico_sinogram[:,32,:] - svmbir_sinogram[32, :, :])
    # print("max diff between scico and svmbir: ", np.max(diff))


    # """
    # Show the recovered image.
    # """
    # fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 2))
    # # plot.imview(
    # #     scico_sinogram[:,32,:],
    # #     title="scico sinogram",
    # #     cmap=plot.cm.Blues,
    # #     cbar=None,
    # #     fig=fig,
    # #     ax=ax[0],
    # # )
    # plot.imview(
    #     svmbir_sinogram[32],
    #     title="svmbir sinogram",
    #     cmap=plot.cm.Blues,
    #     cbar=None,
    #     fig=fig,
    #     ax=ax[0],
    # )
    # plot.imview(
    #     mbirjax_sinogram[32],
    #     title="mbirjax sinogram",
    #     cmap=plot.cm.Blues,
    #     cbar=None,
    #     fig=fig,
    #     ax=ax[1],
    # )

    # # divider = make_axes_locatable(ax[1])
    # # cax = divider.append_axes("right", size="5%", pad=0.2)
    # # fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
    # # fig.colorbar(ax[1].get_images()[0], label="arbitrary units")
    # fig.show()

    # results_dir = os.path.join(os.path.dirname(__file__), 'results')
    # os.makedirs(results_dir, exist_ok=True)
    # save_path = os.path.join(results_dir, 'sinogram_comparison.png')
    # fig.savefig(save_path)   # save the figure to file
