#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MBIRJAX ROI Extension - Adding ASTRA-like move_vol_geom functionality

This module extends MBIRJAX's ParallelBeamModel class with functionality
similar to ASTRA's move_vol_geom function for handling off-center ROI reconstructions.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional, Union
import mbirjax

from mbirjax.parallel_beam import ParallelBeamModel


class ROIExtendedParallelBeamModel(ParallelBeamModel):
    """
    Extended ParallelBeamModel with ROI functionality similar to ASTRA's move_vol_geom.
    
    This class adds methods to handle off-center ROI reconstructions by managing
    coordinate system transformations and geometry adjustments.
    """
    
    def __init__(self, sinogram_shape, angles, roi_offset=(0, 0, 0), roi_shape=None):
        """
        Initialize the extended model with ROI support.
        
        Args:
            sinogram_shape: Shape of the sinogram (num_views, num_det_rows, num_det_channels)
            angles: Array of projection angles in radians
            roi_offset: Offset of ROI center from original image center (x, y, z)
            roi_shape: Shape of the ROI reconstruction (num_rows, num_cols, num_slices)
                      If None, uses the default recon_shape
        """
        super().__init__(sinogram_shape, angles)
        
        # Store ROI parameters
        self.roi_offset = np.array(roi_offset, dtype=np.float32)
        self.original_recon_shape = self.get_params('recon_shape')
        print("Original recon shape: ", self.original_recon_shape)
        
        # Set ROI reconstruction shape if provided
        if roi_shape is not None:
            self.set_params(recon_shape=roi_shape)
        
        # Calculate ROI bounds in original coordinate system
        self._calculate_roi_bounds()
    
    def _calculate_roi_bounds(self):
        """Calculate ROI bounds in the original coordinate system."""
        recon_shape = self.get_params('recon_shape')
        original_shape = self.original_recon_shape
        
        # Calculate ROI center in original coordinate system
        roi_center = np.array(original_shape) // 2 + self.roi_offset
        print("Original center: ", np.array(original_shape) // 2)
        
        # Calculate ROI bounds
        roi_start = roi_center - np.array(recon_shape) // 2
        roi_end = roi_start + np.array(recon_shape)
        
        # Ensure bounds are within original image
        roi_start = np.maximum(roi_start, 0)
        roi_end = np.minimum(roi_end, original_shape)
        
        # Calculate actual ROI shape after clipping
        actual_roi_shape = roi_end - roi_start
        
        self.roi_bounds = {
            'start': roi_start.astype(int),
            'end': roi_end.astype(int),
            'center': roi_center.astype(int),
            'shape': recon_shape,
            'actual_shape': actual_roi_shape.astype(int)
        }
    
    def move_roi_geometry(self, new_offset, is_relative=True):
        """
        Move the ROI geometry to a new location, similar to ASTRA's move_vol_geom.
        
        Args:
            new_offset: New offset (x, y, z) for the ROI center
            is_relative: If True, new_offset is relative to current position
                        If False, new_offset is absolute from original center
        
        Returns:
            self: Returns self for method chaining
        """
        if is_relative:
            self.roi_offset += np.array(new_offset, dtype=np.float32)
        else:
            self.roi_offset = np.array(new_offset, dtype=np.float32)
        
        # Recalculate ROI bounds
        self._calculate_roi_bounds()
        
        return self
    
    def set_roi_shape(self, roi_shape):
        """
        Set the ROI reconstruction shape.
        
        Args:
            roi_shape: New ROI shape (num_rows, num_cols, num_slices)
        """
        self.set_params(recon_shape=roi_shape)
        self._calculate_roi_bounds()
        return self
    
    def get_roi_bounds(self):
        """
        Get the current ROI bounds in the original coordinate system.
        
        Returns:
            dict: Dictionary containing ROI bounds information
        """
        return self.roi_bounds.copy()
    
    def extract_roi_from_full_reconstruction(self, full_recon):
        """
        Extract the ROI region from a full reconstruction.
        
        Args:
            full_recon: Full reconstruction array
            
        Returns:
            ROI region as a numpy array
        """
        bounds = self.roi_bounds
        print("Bounds start: ", bounds['start'])
        print("Bounds end: ", bounds['end'])
        print("Shape of full_recon: ", full_recon.shape)

        roi_recon = full_recon[
            bounds['start'][0]:bounds['end'][0],
            bounds['start'][1]:bounds['end'][1],
            bounds['start'][2]:bounds['end'][2]
        ]
        return roi_recon
    
    def insert_roi_into_full_reconstruction(self, roi_recon, full_recon=None):
        """
        Insert ROI reconstruction into a full reconstruction array.
        
        Args:
            roi_recon: ROI reconstruction array
            full_recon: Full reconstruction array (if None, creates new array)
            
        Returns:
            Full reconstruction with ROI inserted
        """
        bounds = self.roi_bounds
        
        if full_recon is None:
            full_recon = np.zeros(self.original_recon_shape, dtype=roi_recon.dtype)

        # Handle shape mismatch by cropping or padding the ROI data
        actual_shape = bounds['actual_shape']
        roi_shape = np.array(roi_recon.shape)
        
        if np.array_equal(roi_shape, actual_shape):
            # Shapes match, insert directly
            full_recon[
                bounds['start'][0]:bounds['end'][0],
                bounds['start'][1]:bounds['end'][1],
                bounds['start'][2]:bounds['end'][2]
            ] = roi_recon
        else:
            # Shapes don't match, crop or pad the ROI data
            # Calculate crop/pad parameters
            crop_start = np.maximum(0, roi_shape - actual_shape) // 2
            crop_end = crop_start + np.minimum(roi_shape, actual_shape)
            
            # Crop the ROI data to fit the actual ROI region
            cropped_roi = roi_recon[
                crop_start[0]:crop_end[0],
                crop_start[1]:crop_end[1],
                crop_start[2]:crop_end[2]
            ]
            
            # Insert the cropped ROI
            full_recon[
                bounds['start'][0]:bounds['end'][0],
                bounds['start'][1]:bounds['end'][1],
                bounds['start'][2]:bounds['end'][2]
            ] = cropped_roi
        
        return full_recon
    
    def calculate_detector_shift(self, angles):
        """
        Calculate the detector shift needed to account for ROI offset.
        
        Args:
            angles: Array of projection angles
            
        Returns:
            Array of detector shifts for each angle
        """
        # For parallel beam, the detector shift is calculated as:
        # shift = roi_offset[0] * cos(angle) + roi_offset[1] * sin(angle)
        cos_angles = jnp.cos(angles)
        sin_angles = jnp.sin(angles)
        
        detector_shift = (-self.roi_offset[0] * sin_angles + 
                         self.roi_offset[1] * cos_angles)
        
        return detector_shift
    
    def shift_sinogram_for_roi(self, sinogram):
        """
        Shift the sinogram to account for ROI offset.
        
        Args:
            sinogram: Original sinogram
            
        Returns:
            Shifted sinogram
        """
        angles = self.get_params('angles')
        detector_shift = self.calculate_detector_shift(angles)
        
        # Apply shift to sinogram
        shifted_sinogram = self._apply_sinogram_shift(sinogram, detector_shift)
        
        return shifted_sinogram
    
    def _apply_sinogram_shift(self, sinogram, detector_shift):
        """
        Apply detector shift to sinogram using interpolation.
        
        Args:
            sinogram: Original sinogram
            detector_shift: Detector shift for each angle
            
        Returns:
            Shifted sinogram
        """
        # Convert to numpy for interpolation
        sinogram_np = np.array(sinogram)
        shifted_sinogram = np.zeros_like(sinogram_np)
        
        num_views, num_det_rows, num_det_channels = sinogram_np.shape
        
        for view_idx in range(num_views):
            shift = detector_shift[view_idx]
            
            # Create coordinate arrays
            x_coords = np.arange(num_det_channels)
            shifted_x_coords = x_coords - shift
            
            # Interpolate each row
            for row_idx in range(num_det_rows):
                shifted_sinogram[view_idx, row_idx, :] = np.interp(
                    x_coords, shifted_x_coords, sinogram_np[view_idx, row_idx, :],
                    left=0, right=0
                )
        
        return jnp.array(shifted_sinogram)
    
    def recon_roi(self, sinogram, weights=None, **kwargs):
        """
        Reconstruct ROI directly using the current ROI geometry.
        
        Args:
            sinogram: Sinogram data
            weights: Optional weights for reconstruction
            **kwargs: Additional arguments passed to recon()
            
        Returns:
            ROI reconstruction
        """
        # Shift sinogram to account for ROI offset
        shifted_sinogram = self.shift_sinogram_for_roi(sinogram)
        print(f"Shifted sinogram shape: {shifted_sinogram.shape}")
        
        # Perform reconstruction
        roi_recon, recon_dict = self.recon(shifted_sinogram, weights=weights, **kwargs)
        
        return roi_recon, recon_dict
    
    def print_roi_info(self):
        """Print information about the current ROI configuration."""
        print("ROI Configuration:")
        print(f"  ROI Offset: {self.roi_offset}")
        print(f"  ROI Shape: {self.get_params('recon_shape')}")
        print(f"  ROI Bounds: {self.roi_bounds}")
        print(f"  Original Shape: {self.original_recon_shape}")


def create_roi_reconstruction_model(sinogram_shape, angles, roi_offset, roi_shape, **kwargs):
    """
    Convenience function to create an ROI reconstruction model.
    
    Args:
        sinogram_shape: Shape of the sinogram
        angles: Array of projection angles
        roi_offset: Offset of ROI from center (x, y, z)
        roi_shape: Shape of ROI reconstruction
        **kwargs: Additional arguments for model initialization
        
    Returns:
        ROIExtendedParallelBeamModel instance
    """
    return ROIExtendedParallelBeamModel(
        sinogram_shape=sinogram_shape,
        angles=angles,
        roi_offset=roi_offset,
        roi_shape=roi_shape,
        **kwargs
    )


def compare_roi_reconstruction_methods(sinogram, full_shape, roi_shape, roi_offset, angles):
    """
    Compare different ROI reconstruction methods.
    
    Args:
        sinogram: Full sinogram data
        full_shape: Full reconstruction shape
        roi_shape: ROI reconstruction shape
        roi_offset: ROI offset from center
        angles: Projection angles
        
    Returns:
        Dictionary containing results from different methods
    """
    results = {}
    
    # Method 1: Full reconstruction + cropping
    print("Method 1: Full reconstruction + cropping")
    ct_model_full = mbirjax.ParallelBeamModel(sinogram.shape, angles)
    ct_model_full.set_params(recon_shape=full_shape)
    full_recon, full_dict = ct_model_full.recon(sinogram)
    
    # Extract ROI from full reconstruction
    roi_center = np.array(full_shape) // 2 + np.array(roi_offset)
    roi_start = roi_center - np.array(roi_shape) // 2
    roi_end = roi_start + np.array(roi_shape)
    
    roi_from_full = full_recon[
        roi_start[0]:roi_end[0],
        roi_start[1]:roi_end[1],
        roi_start[2]:roi_end[2]
    ]
    results['full_recon_cropped'] = roi_from_full
    
    # Method 2: ROI-specific reconstruction
    print("Method 2: ROI-specific reconstruction")
    roi_model = create_roi_reconstruction_model(
        sinogram.shape, angles, roi_offset, roi_shape
    )
    roi_recon, roi_dict = roi_model.recon_roi(sinogram)
    results['roi_specific'] = roi_recon
    
    # Method 3: Standard MBIRJAX with scaled recon shape
    print("Method 3: Standard MBIRJAX with scaled recon shape")
    ct_model_scaled = mbirjax.ParallelBeamModel(sinogram.shape, angles)
    ct_model_scaled.set_params(recon_shape=roi_shape)
    scaled_recon, scaled_dict = ct_model_scaled.recon(sinogram)
    results['scaled_recon'] = scaled_recon
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Example: Create a test case
    sinogram_shape = (400, 20, 400)
    angles = np.linspace(0, np.pi, 400, endpoint=False)
    full_shape = (400, 400, 20)
    roi_shape = (100, 100, 20)
    roi_offset = (50, 50, 0)
    
    print("Creating ROI reconstruction model...")
    roi_model = create_roi_reconstruction_model(
        sinogram_shape, angles, roi_offset, roi_shape
    )
    
    roi_model.print_roi_info()
    
    print("\nROI reconstruction model created successfully!")
    print("Use roi_model.recon_roi(sinogram) to perform ROI reconstruction.") 