#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Example: Using sparse_forward_project in MBIRJAX

This script provides a simple demonstration of how to use the 
sparse_forward_project function for tomographic reconstruction.
"""

import numpy as np
import jax.numpy as jnp
from mbirjax import ParallelBeamModel


def main():
    """
    Simple demonstration of sparse_forward_project usage
    """
    print("MBIRJAX sparse_forward_project Simple Example")
    print("=" * 50)
    
    # Step 1: Setup the tomography model
    print("1. Setting up tomography model...")
    num_views = 180
    sinogram_shape = (num_views, 20, 256)  # (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    
    ct_model = ParallelBeamModel(sinogram_shape, angles)
    # ct_model.set_params(recon_shape=(256, 256, 20))  # (num_rows, num_cols, num_slices)
    
    print(f"   Sinogram shape: {sinogram_shape}")
    print(f"   Reconstruction shape: {ct_model.get_params('recon_shape')}")
    print(f"   Number of angles: {len(angles)}")
    
    # Step 2: Generate test data
    print("\n2. Generating test phantom...")
    phantom = ct_model.gen_modified_3d_sl_phantom()
    print(f"   Phantom shape: {phantom.shape}")
    
    # Step 3: Demonstrate full forward projection
    print("\n3. Full forward projection (for comparison)...")
    full_sinogram = ct_model.forward_project(phantom)
    print(f"   Full sinogram shape: {full_sinogram.shape}")
    
    # Step 4: Demonstrate sparse forward projection
    print("\n4. Sparse forward projection...")
    
    # Get reconstruction shape
    recon_shape = ct_model.get_params('recon_shape')
    
    # Create indices for all voxels (equivalent to full projection)
    # Note: indices are into flattened 2D array (num_rows * num_cols)
    print(f"   Reconstruction shape: {recon_shape}")
    full_indices = np.arange(recon_shape[0] * recon_shape[1])
    print(f"   Number of pixel indices: {len(full_indices)}")
    
    # Get voxel values at these indices
    # This extracts all slices for each 2D pixel position
    voxel_values = ct_model.get_voxels_at_indices(phantom, full_indices)
    print(f"   Voxel values shape: {voxel_values.shape}")
    print(f"   Expected shape: ({len(full_indices)}, {recon_shape[2]})")
    
    # Perform sparse forward projection
    sparse_sinogram = ct_model.sparse_forward_project(
        voxel_values, 
        full_indices, 
        output_device=ct_model.main_device
    )
    print(f"   Sparse sinogram shape: {sparse_sinogram.shape}")
    
    # Step 5: Compare results
    print("\n5. Comparing results...")
    max_diff = np.max(np.abs(full_sinogram - sparse_sinogram))
    rel_diff = max_diff / np.max(np.abs(full_sinogram))
    print(f"   Maximum absolute difference: {max_diff:.2e}")
    print(f"   Relative difference: {rel_diff:.2e}")
    print(f"   Results are {'identical' if max_diff < 1e-10 else 'different'}")
    
    # Step 6: Demonstrate ROI projection
    print("\n6. ROI (Region of Interest) projection...")
    
    # Define a circular ROI in the center
    center_row, center_col = recon_shape[0] // 2, recon_shape[1] // 2
    radius = 50
    
    # Create coordinate grids
    rows, cols = np.meshgrid(
        np.arange(recon_shape[0]), 
        np.arange(recon_shape[1]), 
        indexing='ij'
    )
    
    # Calculate distance from center
    distance = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
    roi_mask = distance <= radius
    
    # Get indices where ROI mask is True
    roi_indices = np.where(roi_mask.flatten())[0]
    print(f"   ROI center: ({center_row}, {center_col})")
    print(f"   ROI radius: {radius}")
    print(f"   Number of ROI voxels: {len(roi_indices)}")
    print(f"   ROI coverage: {len(roi_indices) / (recon_shape[0] * recon_shape[1]) * 100:.1f}%")
    
    # Get voxel values for ROI only
    roi_voxel_values = ct_model.get_voxels_at_indices(phantom, roi_indices)
    print(f"   ROI voxel values shape: {roi_voxel_values.shape}")
    
    # Perform sparse projection for ROI only
    roi_sinogram = ct_model.sparse_forward_project(
        roi_voxel_values, 
        roi_indices, 
        output_device=ct_model.main_device
    )
    print(f"   ROI sinogram shape: {roi_sinogram.shape}")
    
    # Compare ROI projection with full projection
    roi_diff = full_sinogram - roi_sinogram
    print(f"   Mean difference (ROI vs Full): {np.mean(np.abs(roi_diff)):.2e}")
    
    # Step 7: Demonstrate partial view projection
    print("\n7. Partial view projection...")
    
    # Select every 3rd view
    view_indices = np.arange(0, num_views, 3)
    print(f"   Total views: {num_views}")
    print(f"   Selected views: {len(view_indices)}")
    print(f"   View indices: {view_indices[:10]}...")  # Show first 10
    
    # Perform sparse projection with partial views
    partial_sinogram = ct_model.sparse_forward_project(
        voxel_values, 
        full_indices, 
        view_indices=view_indices,
        output_device=ct_model.main_device
    )
    print(f"   Partial sinogram shape: {partial_sinogram.shape}")
    
    # Compare with subset of full sinogram
    full_subset = full_sinogram[view_indices]
    partial_diff = full_subset - partial_sinogram
    print(f"   Maximum difference (Partial vs Full subset): {np.max(np.abs(partial_diff)):.2e}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("=" * 50)
    
    return {
        'ct_model': ct_model,
        'phantom': phantom,
        'full_sinogram': full_sinogram,
        'sparse_sinogram': sparse_sinogram,
        'roi_sinogram': roi_sinogram,
        'partial_sinogram': partial_sinogram,
        'roi_indices': roi_indices,
        'view_indices': view_indices
    }


if __name__ == "__main__":
    # Run the main example
    results = main()