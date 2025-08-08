#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of forward_project_pixel_batch_to_one_view function in MBIRJAX

This script demonstrates how to use the forward_project_pixel_batch_to_one_view 
function from the ParallelBeamModel class in mbirjax. This function performs
parallel beam forward projection on a batch of voxel cylinders to produce
a single sinogram view.

The function is useful for:
- Sparse forward projections (only projecting specific voxels)
- Custom projection implementations
- Understanding the projection process step-by-step
"""

import numpy as np
import jax.numpy as jnp
from collections import namedtuple
import mbirjax
from mbirjax import ParallelBeamModel


def create_projector_params(model):
    """
    Create the projector_params structure required by forward_project_pixel_batch_to_one_view.
    
    Args:
        model: ParallelBeamModel instance
        
    Returns:
        namedtuple: Projector parameters containing sinogram_shape, recon_shape, and geometry_params
    """
    # Get geometry parameters from the model
    geometry_params = model.get_geometry_parameters()
    
    # Get sinogram and reconstruction shapes
    sinogram_shape = model.get_params('sinogram_shape')
    recon_shape = model.get_params('recon_shape')
    
    # Create projector parameters structure
    ProjectorParams = namedtuple('ProjectorParams', [
        'sinogram_shape', 'recon_shape', 'geometry_params'
    ])
    
    projector_params = ProjectorParams(
        sinogram_shape=sinogram_shape,
        recon_shape=recon_shape,
        geometry_params=geometry_params
    )
    
    return projector_params


def extract_voxel_values_at_indices(phantom, pixel_indices):
    """
    Extract voxel values at specific pixel indices from a 3D phantom.
    
    Args:
        phantom: 3D array of shape (num_rows, num_cols, num_slices)
        pixel_indices: 1D array of indices into flattened (num_rows, num_cols) array
        
    Returns:
        2D array of shape (num_pixels, num_slices) containing voxel values
    """
    num_rows, num_cols, num_slices = phantom.shape
    num_pixels = len(pixel_indices)
    
    # Initialize voxel values array
    voxel_values = np.zeros((num_pixels, num_slices))
    
    # Extract values for each pixel index
    for i, pixel_idx in enumerate(pixel_indices):
        row_idx, col_idx = np.unravel_index(pixel_idx, (num_rows, num_cols))
        voxel_values[i, :] = phantom[row_idx, col_idx, :]
    
    return voxel_values


def demonstrate_sparse_projection():
    """Demonstrate sparse forward projection (only specific voxels)."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 2: Sparse Projection (Selected Voxels)")
    print("=" * 60)
    
    # Step 1: Setup the tomography model
    print("\n1. Setting up tomography model...")
    num_views = 90
    sinogram_shape = (num_views, 16, 128)  # Smaller for faster execution
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    
    ct_model = ParallelBeamModel(sinogram_shape, angles)
    ct_model.auto_set_recon_size(sinogram_shape)
    
    print(f"   Sinogram shape: {sinogram_shape}")
    print(f"   Reconstruction shape: {ct_model.get_params('recon_shape')}")
    
    # Step 2: Generate test phantom
    print("\n2. Generating test phantom...")
    phantom = ct_model.gen_modified_3d_sl_phantom()
    print(f"   Phantom shape: {phantom.shape}")
    
    # Step 3: Create projector parameters
    print("\n3. Creating projector parameters...")
    projector_params = create_projector_params(ct_model)
    
    # Step 4: Select specific voxels for sparse projection
    print("\n4. Selecting voxels for sparse projection...")
    recon_shape = ct_model.get_params('recon_shape')
    num_rows, num_cols, num_slices = recon_shape
    
    # Create a sparse pattern - select every 4th voxel in a checkerboard pattern
    rows, cols = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
    sparse_mask = ((rows + cols) % 4 == 0)  # Checkerboard pattern
    sparse_pixel_indices = np.where(sparse_mask.flatten())[0]
    
    print(f"   Total voxels: {num_rows * num_cols}")
    print(f"   Selected voxels: {len(sparse_pixel_indices)}")
    print(f"   Sparsity ratio: {len(sparse_pixel_indices) / (num_rows * num_cols):.2%}")
    
    # Step 5: Create sparse phantom and extract voxel values
    print("\n5. Creating sparse phantom and extracting voxel values...")
    
    # Create a sparse phantom where only selected voxels have non-zero values
    sparse_phantom = np.zeros_like(phantom)
    for i, pixel_idx in enumerate(sparse_pixel_indices):
        row_idx, col_idx = np.unravel_index(pixel_idx, (num_rows, num_cols))
        sparse_phantom[row_idx, col_idx, :] = phantom[row_idx, col_idx, :]
    
    # Extract voxel values for selected indices (should be the same as original phantom values)
    voxel_values = extract_voxel_values_at_indices(phantom, sparse_pixel_indices)
    print(f"   Voxel values shape: {voxel_values.shape}")
    
    # Step 6: Project multiple views using sparse voxels
    print("\n6. Projecting multiple views using sparse voxels...")
    num_test_views = 5
    test_angles = angles[:num_test_views]
    
    # Convert to JAX arrays
    voxel_values_jax = jnp.array(voxel_values)
    pixel_indices_jax = jnp.array(sparse_pixel_indices)

    print(f"   Voxel values shape: {voxel_values_jax.shape}")
    print(f"   Pixel indices shape: {pixel_indices_jax.shape}")
    
    # Project each view
    sparse_sinogram_views = []
    for i, angle in enumerate(test_angles):
        print(f"   Projecting view {i+1}/{num_test_views} at angle {np.degrees(angle):.1f}°")
        sinogram_view = ParallelBeamModel.forward_project_pixel_batch_to_one_view(
            voxel_values_jax, pixel_indices_jax, angle, projector_params
        )
        sparse_sinogram_views.append(sinogram_view)
    
    sparse_sinogram = np.stack(sparse_sinogram_views, axis=0)
    print(f"   Sparse sinogram shape: {sparse_sinogram.shape}")
    
    # Step 7: Compare with sparse phantom projection
    print("\n7. Comparing with sparse phantom projection...")
    
    # Project the sparse phantom using the model's forward_project method
    sparse_phantom_sinogram = ct_model.forward_project(sparse_phantom)
    sparse_phantom_subset = sparse_phantom_sinogram[:num_test_views, :, :]
    
    print(f"   Sparse phantom sinogram subset shape: {sparse_phantom_subset.shape}")
    
    # Calculate differences between sparse projection and sparse phantom projection
    diff = np.abs(sparse_sinogram - sparse_phantom_subset)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"   Maximum difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    print(f"   Are sparse projections identical? {np.allclose(sparse_sinogram, sparse_phantom_subset, atol=1e-10)}")


def demonstrate_custom_voxel_pattern():
    """Demonstrate projection with a custom voxel pattern (e.g., circular region)."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 3: Custom Voxel Pattern (Circular Region)")
    print("=" * 60)
    
    # Step 1: Setup the tomography model
    print("\n1. Setting up tomography model...")
    num_views = 60
    sinogram_shape = (num_views, 12, 64)  # Small for demonstration
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    
    ct_model = ParallelBeamModel(sinogram_shape, angles)
    ct_model.auto_set_recon_size(sinogram_shape)
    
    print(f"   Sinogram shape: {sinogram_shape}")
    print(f"   Reconstruction shape: {ct_model.get_params('recon_shape')}")
    
    # Step 2: Generate test phantom
    print("\n2. Generating test phantom...")
    phantom = ct_model.gen_modified_3d_sl_phantom()
    print(f"   Phantom shape: {phantom.shape}")
    
    # Step 3: Create projector parameters
    print("\n3. Creating projector parameters...")
    projector_params = create_projector_params(ct_model)
    
    # Step 4: Create circular voxel pattern
    print("\n4. Creating circular voxel pattern...")
    recon_shape = ct_model.get_params('recon_shape')
    num_rows, num_cols, num_slices = recon_shape
    
    # Create circular mask
    center_row, center_col = num_rows // 2, num_cols // 2
    radius = min(num_rows, num_cols) // 4
    
    rows, cols = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
    distance_from_center = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
    circular_mask = distance_from_center <= radius

    print(f"   Circular mask shape: {circular_mask.shape}")
    
    circular_pixel_indices = np.where(circular_mask.flatten())[0]
    
    print(f"   Total voxels: {num_rows * num_cols}")
    print(f"   Circular region voxels: {len(circular_pixel_indices)}")
    print(f"   Coverage ratio: {len(circular_pixel_indices) / (num_rows * num_cols):.2%}")
    
    # Step 5: Create circular phantom and extract voxel values
    print("\n5. Creating circular phantom and extracting voxel values...")
    
    # Create a circular phantom where only voxels in the circular region have non-zero values
    circular_phantom = np.zeros_like(phantom)
    for i, pixel_idx in enumerate(circular_pixel_indices):
        row_idx, col_idx = np.unravel_index(pixel_idx, (num_rows, num_cols))
        circular_phantom[row_idx, col_idx, :] = phantom[row_idx, col_idx, :]
    
    # Extract voxel values for circular region
    voxel_values = extract_voxel_values_at_indices(phantom, circular_pixel_indices)
    print(f"   Voxel values shape: {voxel_values.shape}")
    
    # Step 6: Project a single view
    print("\n6. Projecting single view of circular region...")
    test_angle = angles[len(angles)//2]  # Middle angle
    print(f"   Test angle: {test_angle:.3f} radians ({np.degrees(test_angle):.1f} degrees)")
    
    # Convert to JAX arrays
    voxel_values_jax = jnp.array(voxel_values)
    pixel_indices_jax = jnp.array(circular_pixel_indices)
    
    # Perform the projection
    circular_sinogram_view = ParallelBeamModel.forward_project_pixel_batch_to_one_view(
        voxel_values_jax, pixel_indices_jax, test_angle, projector_params
    )
    
    print(f"   Circular projection shape: {circular_sinogram_view.shape}")
    
    # Step 7: Compare with circular phantom projection
    print("\n7. Comparing with circular phantom projection...")
    circular_phantom_sinogram = ct_model.forward_project(circular_phantom)
    circular_phantom_view = circular_phantom_sinogram[len(angles)//2, :, :]  # Same view
    
    print(f"   Circular phantom projection view shape: {circular_phantom_view.shape}")
    
    # Calculate differences between circular projection and circular phantom projection
    diff = np.abs(circular_sinogram_view - circular_phantom_view)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"   Maximum difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    print(f"   Are circular projections identical? {np.allclose(circular_sinogram_view, circular_phantom_view, atol=1e-10)}")
    
    # Step 8: Compare with full projection to show the difference
    print("\n8. Comparing circular projection with full projection...")
    full_sinogram = ct_model.forward_project(phantom)
    full_sinogram_view = full_sinogram[len(angles)//2, :, :]  # Same view
    
    # Calculate differences between circular and full projections
    diff_circular_vs_full = np.abs(circular_sinogram_view - full_sinogram_view)
    max_diff_circular_vs_full = np.max(diff_circular_vs_full)
    mean_diff_circular_vs_full = np.mean(diff_circular_vs_full)
    
    print(f"   Maximum difference (circular vs full): {max_diff_circular_vs_full:.6f}")
    print(f"   Mean difference (circular vs full): {mean_diff_circular_vs_full:.6f}")
    print(f"   Are circular and full projections identical? {np.allclose(circular_sinogram_view, full_sinogram_view, atol=1e-10)}")
    print(f"   Note: Differences are expected since circular projection only includes voxels in the circular region")
    
    # Step 9: Demonstrate masking the full sinogram view with circular mask
    print("\n9. Demonstrating masking of full sinogram view with circular mask...")
    
    # Create a binary phantom from the circular mask
    circular_binary_phantom = np.zeros_like(phantom)
    for i, j in enumerate(circular_mask.flatten()):
        circular_binary_phantom[i, j, :] = 1.0  # Set circular region to 1
    
    # Project the binary phantom to create a sinogram mask
    circular_binary_sinogram = ct_model.forward_project(circular_binary_phantom)
    sinogram_mask = circular_binary_sinogram[len(angles)//2, :, :]  # Same view
    
    print(f"   Sinogram mask shape: {sinogram_mask.shape}")
    print(f"   Sinogram mask range: [{np.min(sinogram_mask):.3f}, {np.max(sinogram_mask):.3f}]")
    
    # Apply the mask to the full sinogram view
    masked_full_sinogram_view = full_sinogram_view * sinogram_mask
    
    print(f"   Masked full sinogram view shape: {masked_full_sinogram_view.shape}")
    
    # Compare masked full sinogram with circular projection
    diff_masked_vs_circular = np.abs(masked_full_sinogram_view - circular_sinogram_view)
    max_diff_masked = np.max(diff_masked_vs_circular)
    mean_diff_masked = np.mean(diff_masked_vs_circular)
    
    print(f"   Maximum difference (masked full vs circular): {max_diff_masked:.6f}")
    print(f"   Mean difference (masked full vs circular): {mean_diff_masked:.6f}")
    print(f"   Are masked full and circular projections identical? {np.allclose(masked_full_sinogram_view, circular_sinogram_view, atol=1e-10)}")


def main():
    """Run all demonstrations."""
    print("MBIRJAX forward_project_pixel_batch_to_one_view Demonstration")
    print("=" * 80)
    print("\nThis script demonstrates how to use the forward_project_pixel_batch_to_one_view")
    print("function from the ParallelBeamModel class in mbirjax.")
    print("\nThe function signature is:")
    print("forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, angle, projector_params)")
    print("\nWhere:")
    print("- voxel_values: 2D array of shape (num_pixels, num_recon_slices)")
    print("- pixel_indices: 1D array of indices into flattened (num_rows, num_cols) array")
    print("- angle: projection angle in radians")
    print("- projector_params: namedtuple with sinogram_shape, recon_shape, and geometry_params")
    
    # Run demonstrations
    # demonstrate_full_projection()
    demonstrate_sparse_projection()
    demonstrate_custom_voxel_pattern()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey takeaways:")
    print("1. The function projects voxel cylinders to a single sinogram view")
    print("2. It's useful for sparse projections and custom implementations")
    print("3. The projector_params structure contains all necessary geometry information")
    print("4. The function is JIT-compiled for efficient execution")
    print("5. It can be used to project any subset of voxels, not just full volumes")


if __name__ == "__main__":
    main() 