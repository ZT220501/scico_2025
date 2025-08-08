#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: ROI Reconstruction with MBIRJAX ROI Extension

This script demonstrates how to use the recon_roi() method to perform
region-of-interest reconstructions with automatic sinogram shifting.
"""

import numpy as np
# import mbirjax
from mbirjax import ParallelBeamModel, slice_viewer
from examples.mbirjax_scripts.roi_reconstruction.ROIParallelBeamModel import create_roi_reconstruction_model


def main():
    """Demonstrate ROI reconstruction."""
    print("ROI Reconstruction Example")
    print("=" * 50)
    
    # Step 1: Setup parameters
    num_views = 400
    sinogram_shape = (num_views, 20, 400)  # (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(0, np.pi, num_views, endpoint=False)

    
    # Step 2: Create test data (phantom and sinogram)
    print("Creating test data...")
    ct_model_full = ParallelBeamModel(sinogram_shape, angles)
    sharpness = -0.5
    ct_model_full.set_params(sharpness=sharpness)
    
    # Generate phantom
    phantom = ct_model_full.gen_modified_3d_sl_phantom()
    print("Phantom shape: ", phantom.shape)
    
    # Generate sinogram
    sinogram = ct_model_full.forward_project(phantom)
    
    print(f"Phantom shape: {phantom.shape}")
    print(f"Sinogram shape: {sinogram.shape}")
    
    # Step 3: Define ROI parameters
    roi_shape = (200, 150, 20)
    roi_offset = (30, -50, 0)
    # roi_offset = (0, 50, 0)
    # roi_offset = (0, 0, 0)
    
    print(f"\nROI Parameters:")
    print(f"  ROI Shape: {roi_shape}")
    print(f"  ROI Offset: {roi_offset}")
    
    # Step 4: Create ROI reconstruction model
    print("\nCreating ROI reconstruction model...")
    roi_model = create_roi_reconstruction_model(
        sinogram_shape, angles, roi_offset, roi_shape
    )
    
    # Print ROI information
    roi_model.print_roi_info()
    
    # Step 5: Perform ROI reconstruction
    print("\nPerforming ROI reconstruction...")
    max_iterations = 50
    roi_recon, recon_dict = roi_model.recon_roi(sinogram, max_iterations=max_iterations)
    
    print(f"ROI reconstruction completed!")
    print(f"ROI reconstruction shape: {roi_recon.shape}")
    
    # Step 6: Extract corresponding region from full phantom for comparison
    roi_phantom = roi_model.extract_roi_from_full_reconstruction(phantom)

    print(f"ROI phantom shape: {roi_phantom.shape}")
    print(f"ROI recon shape: {roi_recon.shape}")
    
    # Step 7: Calculate quality metrics
    mse = np.mean((roi_recon - roi_phantom)**2)
    print(f"\nQuality Metrics:")
    print(f"  Mean Squared Error: {mse:.6f}")
    
    # Step 8: Save and visualize the ROI reconstruction
    title = 'Cropped center recon with sharpness = {:.1f}: Phantom (left) vs VCD Recon (right)'.format(sharpness)
    title += '\nThis recon does not include all pixels used to generate the sinogram.'
    title += '\nThe missing pixels lead to an intensity shift (adjust intensity to [0, 1]) and a bright outer ring.'



    save_path = './examples/mbirjax_scripts/results/roi_recon.png'
    slice_viewer(roi_phantom, roi_recon, title=title, vmin=0.0, vmax=2.0, save=True, save_path=save_path)
    # slice_viewer(roi_phantom, roi_phantom_check, title=title, vmin=0.0, vmax=2.0, save=True, save_path=save_path)
    # slice_viewer(roi_phantom_check, roi_recon, title=title, vmin=0.0, vmax=2.0, save=True, save_path=save_path)

    
    return roi_recon, roi_phantom


if __name__ == "__main__":
    main() 