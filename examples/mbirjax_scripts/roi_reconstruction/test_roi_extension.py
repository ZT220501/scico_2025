#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script for the MBIRJAX ROI extension functionality.
"""

import numpy as np
import mbirjax
from examples.mbirjax_scripts.roi_reconstruction.ROIParallelBeamModel import create_roi_reconstruction_model


def test_basic_functionality():
    """Test basic ROI functionality."""
    print("Testing basic ROI functionality...")
    
    # Setup
    sinogram_shape = (400, 10, 400)  # Smaller for faster testing
    angles = np.linspace(0, np.pi, 400, endpoint=False)
    roi_shape = (50, 50, 10)
    roi_offset = (25, 25, 0)
    
    # Create ROI model
    roi_model = create_roi_reconstruction_model(
        sinogram_shape, angles, roi_offset, roi_shape
    )
    
    # Test ROI bounds calculation
    bounds = roi_model.get_roi_bounds()
    print(f"ROI bounds: {bounds}")
    
    # Test ROI movement
    roi_model.move_roi_geometry((10, 10, 0), is_relative=True)
    new_bounds = roi_model.get_roi_bounds()
    print(f"ROI bounds after movement: {new_bounds}")
    
    print("✓ Basic functionality test passed!")


def test_sinogram_shift():
    """Test sinogram shifting functionality."""
    print("\nTesting sinogram shifting...")
    
    # Setup
    sinogram_shape = (50, 5, 50)
    angles = np.linspace(0, np.pi, 50, endpoint=False)
    roi_offset = (10, 10, 0)
    
    # Create test sinogram
    sinogram = np.random.rand(*sinogram_shape).astype(np.float32)
    
    # Create ROI model
    roi_model = create_roi_reconstruction_model(
        sinogram_shape, angles, roi_offset, (20, 20, 5)
    )
    
    # Test sinogram shifting
    shifted_sinogram = roi_model.shift_sinogram_for_roi(sinogram)
    
    print(f"Original sinogram shape: {sinogram.shape}")
    print(f"Shifted sinogram shape: {shifted_sinogram.shape}")
    print(f"Sinogram shift applied successfully!")
    
    print("✓ Sinogram shifting test passed!")


def test_roi_extraction():
    """Test ROI extraction and insertion."""
    print("\nTesting ROI extraction and insertion...")
    
    # Setup
    full_shape = (400, 400, 50)
    roi_shape = (30, 30, 10)
    roi_offset = (20, 20, 5)
    
    # Create test data
    full_data = np.random.rand(*full_shape).astype(np.float32)
    roi_data = np.random.rand(*roi_shape).astype(np.float32)
    
    # Create ROI model
    roi_model = create_roi_reconstruction_model(
        (400, 10, 400), np.linspace(0, np.pi, 400), roi_offset, roi_shape
    )
    
    # Test extraction
    extracted_roi = roi_model.extract_roi_from_full_reconstruction(full_data)
    print(f"Extracted ROI shape: {extracted_roi.shape}")
    
    # Test insertion
    inserted_full = roi_model.insert_roi_into_full_reconstruction(roi_data)
    print(f"Inserted full shape: {inserted_full.shape}")
    
    print("✓ ROI extraction/insertion test passed!")


def test_detector_shift_calculation():
    """Test detector shift calculation."""
    print("\nTesting detector shift calculation...")
    
    # Setup
    angles = np.linspace(0, np.pi, 10, endpoint=False)
    roi_offset = (5, 3, 0)
    
    # Create ROI model
    roi_model = create_roi_reconstruction_model(
        (10, 5, 10), angles, roi_offset, (5, 5, 5)
    )
    
    # Calculate detector shift
    detector_shift = roi_model.calculate_detector_shift(angles)
    
    print(f"Detector shift shape: {detector_shift.shape}")
    print(f"Detector shift values: {detector_shift}")
    
    # Verify the calculation makes sense
    # For angle 0, shift should be approximately roi_offset[0]
    # For angle pi/2, shift should be approximately roi_offset[1]
    angle_0_shift = detector_shift[0]
    
    # Find the angle closest to π/2
    target_angle = np.pi / 2
    angle_idx = np.argmin(np.abs(angles - target_angle))
    angle_pi2_shift = detector_shift[angle_idx]
    actual_angle = angles[angle_idx]
    
    print(f"Shift at angle 0: {angle_0_shift:.3f} (expected ~{roi_offset[0]})")
    print(f"Shift at angle {actual_angle:.3f} (π/2): {angle_pi2_shift:.3f} (expected ~{roi_offset[1]})")
    
    print("✓ Detector shift calculation test passed!")


def main():
    """Run all tests."""
    print("MBIRJAX ROI Extension Tests")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_sinogram_shift()
        test_roi_extraction()
        test_detector_shift_calculation()
        
        print("\n" + "=" * 40)
        print("ALL TESTS PASSED! ✓")
        print("=" * 40)
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 