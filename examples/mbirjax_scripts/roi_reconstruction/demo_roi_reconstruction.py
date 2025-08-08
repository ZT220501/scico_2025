#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of ROI Reconstruction with MBIRJAX

This script demonstrates how to use the ROIExtendedParallelBeamModel
for off-center ROI reconstructions, similar to ASTRA's move_vol_geom functionality.
"""

import numpy as np
import jax.numpy as jnp
import mbirjax
import time
from mbirjax_roi_extension import (
    ROIExtendedParallelBeamModel, 
    create_roi_reconstruction_model,
    compare_roi_reconstruction_methods
)


def create_test_data():
    """Create test sinogram data for demonstration."""
    print("Creating test data...")
    
    # Setup parameters
    sinogram_shape = (400, 20, 400)
    full_shape = (400, 400, 20)
    angles = np.linspace(0, np.pi, 400, endpoint=False)
    
    # Create phantom using MBIRJAX
    ct_model_full = mbirjax.ParallelBeamModel(sinogram_shape, angles)
    ct_model_full.set_params(recon_shape=full_shape)
    
    # Generate phantom and sinogram
    phantom = ct_model_full.gen_modified_3d_sl_phantom()
    sinogram = ct_model_full.forward_project(phantom)
    
    print(f"Phantom shape: {phantom.shape}")
    print(f"Sinogram shape: {sinogram.shape}")
    
    return sinogram, phantom, angles, full_shape


def demonstrate_center_roi():
    """Demonstrate center ROI reconstruction (baseline)."""
    print("\n" + "="*60)
    print("DEMONSTRATION 1: CENTER ROI RECONSTRUCTION")
    print("="*60)
    
    sinogram, phantom, angles, full_shape = create_test_data()
    
    # Define center ROI
    roi_shape = (100, 100, 20)
    roi_offset = (0, 0, 0)  # Center
    
    # Create ROI model
    roi_model = create_roi_reconstruction_model(
        sinogram.shape, angles, roi_offset, roi_shape
    )
    roi_model.print_roi_info()
    
    # Perform ROI reconstruction
    print("\nPerforming center ROI reconstruction...")
    start_time = time.time()
    roi_recon, roi_dict = roi_model.recon_roi(sinogram)
    elapsed_time = time.time() - start_time
    
    print(f"Center ROI reconstruction completed in {elapsed_time:.2f} seconds")
    
    # Extract corresponding region from full phantom for comparison
    roi_phantom = roi_model.extract_roi_from_full_reconstruction(phantom)
    
    # Calculate metrics
    mse = np.mean((roi_recon - roi_phantom)**2)
    print(f"Mean Squared Error: {mse:.6f}")
    
    return roi_recon, roi_phantom


def demonstrate_off_center_roi():
    """Demonstrate off-center ROI reconstruction."""
    print("\n" + "="*60)
    print("DEMONSTRATION 2: OFF-CENTER ROI RECONSTRUCTION")
    print("="*60)
    
    sinogram, phantom, angles, full_shape = create_test_data()
    
    # Define off-center ROI
    roi_shape = (100, 100, 20)
    roi_offset = (100, 100, 0)  # Off-center
    
    # Create ROI model
    roi_model = create_roi_reconstruction_model(
        sinogram.shape, angles, roi_offset, roi_shape
    )
    roi_model.print_roi_info()
    
    # Perform ROI reconstruction
    print("\nPerforming off-center ROI reconstruction...")
    start_time = time.time()
    roi_recon, roi_dict = roi_model.recon_roi(sinogram)
    elapsed_time = time.time() - start_time
    
    print(f"Off-center ROI reconstruction completed in {elapsed_time:.2f} seconds")
    
    # Extract corresponding region from full phantom for comparison
    roi_phantom = roi_model.extract_roi_from_full_reconstruction(phantom)
    
    # Calculate metrics
    mse = np.mean((roi_recon - roi_phantom)**2)
    print(f"Mean Squared Error: {mse:.6f}")
    
    return roi_recon, roi_phantom


def demonstrate_roi_movement():
    """Demonstrate moving ROI geometry."""
    print("\n" + "="*60)
    print("DEMONSTRATION 3: MOVING ROI GEOMETRY")
    print("="*60)
    
    sinogram, phantom, angles, full_shape = create_test_data()
    
    # Create ROI model with initial position
    roi_shape = (80, 80, 20)
    initial_offset = (50, 50, 0)
    
    roi_model = create_roi_reconstruction_model(
        sinogram.shape, angles, initial_offset, roi_shape
    )
    
    print("Initial ROI configuration:")
    roi_model.print_roi_info()
    
    # Move ROI to different positions
    positions = [
        (0, 0, 0),      # Center
        (100, 0, 0),    # Right
        (0, 100, 0),    # Down
        (100, 100, 0),  # Corner
        (-50, -50, 0),  # Left-up
    ]
    
    results = {}
    
    for i, offset in enumerate(positions):
        print(f"\n--- Position {i+1}: {offset} ---")
        
        # Move ROI geometry
        roi_model.move_roi_geometry(offset, is_relative=False)
        roi_model.print_roi_info()
        
        # Perform reconstruction
        start_time = time.time()
        roi_recon, roi_dict = roi_model.recon_roi(sinogram)
        elapsed_time = time.time() - start_time
        
        print(f"Reconstruction time: {elapsed_time:.2f} seconds")
        
        # Extract ground truth
        roi_phantom = roi_model.extract_roi_from_full_reconstruction(phantom)
        
        # Calculate metrics
        mse = np.mean((roi_recon - roi_phantom)**2)
        print(f"Mean Squared Error: {mse:.6f}")
        
        results[f"position_{i+1}"] = {
            'offset': offset,
            'recon': roi_recon,
            'phantom': roi_phantom,
            'mse': mse,
            'time': elapsed_time
        }
    
    return results


def compare_methods():
    """Compare different ROI reconstruction methods."""
    print("\n" + "="*60)
    print("DEMONSTRATION 4: COMPARING RECONSTRUCTION METHODS")
    print("="*60)
    
    sinogram, phantom, angles, full_shape = create_test_data()
    
    # Define ROI parameters
    roi_shape = (100, 100, 20)
    roi_offset = (75, 75, 0)
    
    print(f"Comparing methods for ROI: shape={roi_shape}, offset={roi_offset}")
    
    # Compare different methods
    results = compare_roi_reconstruction_methods(
        sinogram, full_shape, roi_shape, roi_offset, angles
    )
    
    # Extract ground truth for comparison
    roi_center = np.array(full_shape) // 2 + np.array(roi_offset)
    roi_start = roi_center - np.array(roi_shape) // 2
    roi_end = roi_start + np.array(roi_shape)
    
    roi_phantom = phantom[
        roi_start[0]:roi_end[0],
        roi_start[1]:roi_end[1],
        roi_start[2]:roi_end[2]
    ]
    
    # Calculate metrics for each method
    print("\nMethod Comparison Results:")
    print("-" * 50)
    
    for method_name, recon in results.items():
        mse = np.mean((recon - roi_phantom)**2)
        print(f"{method_name:20s}: MSE = {mse:.6f}")
    
    return results, roi_phantom


def demonstrate_edge_roi():
    """Demonstrate reconstruction of ROI at the edge (challenging case)."""
    print("\n" + "="*60)
    print("DEMONSTRATION 5: EDGE ROI RECONSTRUCTION (CHALLENGING)")
    print("="*60)
    
    sinogram, phantom, angles, full_shape = create_test_data()
    
    # Define edge ROI (challenging case)
    roi_shape = (80, 80, 20)
    roi_offset = (150, 150, 0)  # Near edge
    
    print(f"Edge ROI: shape={roi_shape}, offset={roi_offset}")
    print("This is a challenging case with significant missing data.")
    
    # Create ROI model
    roi_model = create_roi_reconstruction_model(
        sinogram.shape, angles, roi_offset, roi_shape
    )
    roi_model.print_roi_info()
    
    # Use stronger regularization for edge ROI
    roi_model.set_params(sharpness=-1.5)  # More regularization
    
    # Perform reconstruction
    print("\nPerforming edge ROI reconstruction...")
    start_time = time.time()
    roi_recon, roi_dict = roi_model.recon_roi(sinogram)
    elapsed_time = time.time() - start_time
    
    print(f"Edge ROI reconstruction completed in {elapsed_time:.2f} seconds")
    
    # Extract ground truth
    roi_phantom = roi_model.extract_roi_from_full_reconstruction(phantom)
    
    # Calculate metrics
    mse = np.mean((roi_recon - roi_phantom)**2)
    print(f"Mean Squared Error: {mse:.6f}")
    
    # Note about expected artifacts
    print("\nNote: Edge ROI reconstructions typically have:")
    print("- Intensity non-uniformity due to incomplete ray coverage")
    print("- Boundary artifacts")
    print("- Reduced image quality compared to center ROI")
    
    return roi_recon, roi_phantom


def main():
    """Run all demonstrations."""
    print("MBIRJAX ROI RECONSTRUCTION DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows how to use the ROIExtendedParallelBeamModel")
    print("for off-center ROI reconstructions, similar to ASTRA's move_vol_geom.")
    print("=" * 60)
    
    try:
        # Demonstration 1: Center ROI
        center_recon, center_phantom = demonstrate_center_roi()
        
        # Demonstration 2: Off-center ROI
        off_center_recon, off_center_phantom = demonstrate_off_center_roi()
        
        # Demonstration 3: Moving ROI
        movement_results = demonstrate_roi_movement()
        
        # Demonstration 4: Method comparison
        comparison_results, comparison_phantom = compare_methods()
        
        # Demonstration 5: Edge ROI
        edge_recon, edge_phantom = demonstrate_edge_roi()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nSummary:")
        print("- Center ROI reconstruction provides the best quality")
        print("- Off-center ROI reconstruction is possible but may have artifacts")
        print("- Edge ROI reconstruction is challenging and requires more regularization")
        print("- The ROIExtendedParallelBeamModel provides ASTRA-like functionality")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 