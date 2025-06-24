#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Demonstration of Radon projection matrix structure

import numpy as np
import scico.numpy as snp
from scico.examples import create_tangle_phantom
from scico.linop.xray.astra import XRayTransform3D
from scico import plot

def analyze_radon_matrix_structure():
    """Analyze and visualize the structure of a Radon projection matrix."""
    
    print("="*60)
    print("RADON PROJECTION MATRIX STRUCTURE")
    print("="*60)
    
    # 1. Create a small test case
    print("\n1. Creating test case...")
    Nx, Ny, Nz = 8, 8, 4  # Small volume for visualization
    volume_shape = (Nz, Ny, Nx)
    n_angles = 4  # Few angles for clarity
    
    print(f"Volume shape: {volume_shape}")
    print(f"Number of angles: {n_angles}")
    print(f"Total voxels: {Nz * Ny * Nx}")
    
    # 2. Create projection operator
    print("\n2. Creating projection operator...")
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    det_count = [Nz, max(Nx, Ny)]
    
    C = XRayTransform3D(
        volume_shape, 
        det_count=det_count, 
        det_spacing=[1.0, 1.0], 
        angles=angles
    )
    
    print(f"Sinogram shape: {C.output_shape}")
    print(f"Total detector measurements: {np.prod(C.output_shape)}")
    
    # 3. Analyze matrix dimensions
    print("\n3. Matrix dimensions:")
    N_voxels = Nz * Ny * Nx
    N_measurements = np.prod(C.output_shape)
    
    print(f"Volume vector length: {N_voxels}")
    print(f"Sinogram vector length: {N_measurements}")
    print(f"Projection matrix shape: {N_measurements} × {N_voxels}")
    print(f"Matrix size: {N_measurements * N_voxels:,} elements")
    
    # 4. Create test volume and project
    print("\n4. Creating test volume and projection...")
    x = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    y = C @ x
    
    print(f"Test volume shape: {x.shape}")
    print(f"Projected sinogram shape: {y.shape}")
    
    # 5. Analyze sparsity pattern
    print("\n5. Analyzing sparsity pattern...")
    
    # Create a simple test to understand the pattern
    # Test with unit vectors (one voxel at a time)
    sparsity_pattern = np.zeros((N_measurements, N_voxels))
    
    for i in range(min(10, N_voxels)):  # Test first 10 voxels
        # Create unit vector for voxel i
        x_unit = snp.zeros(N_voxels)
        x_unit = x_unit.at[i].set(1.0)
        x_unit_vol = x_unit.reshape(volume_shape)
        
        # Project this unit vector
        y_unit = C @ x_unit_vol
        sparsity_pattern[:, i] = y_unit.flatten()
    
    # Count non-zero elements
    nnz = np.count_nonzero(sparsity_pattern)
    sparsity = 1.0 - (nnz / sparsity_pattern.size)
    
    print(f"Non-zero elements in sample: {nnz}")
    print(f"Sparsity: {sparsity:.3f} ({sparsity*100:.1f}% zero elements)")
    
    
    
    # 6. Mathematical properties
    print("\n7. Mathematical properties:")
    print(f"• Matrix H maps: R^{N_voxels} → R^{N_measurements}")
    print(f"• Each row represents one detector measurement")
    print(f"• Each column represents one voxel's contribution")
    print(f"• Element H[i,j] = contribution of voxel j to measurement i")
    print(f"• Matrix is typically underdetermined: {N_measurements} < {N_voxels}")
    print(f"• Sparsity: ~{sparsity*100:.1f}% of elements are zero")
    
    # 7. Block structure
    print("\n8. Block structure:")
    print(f"• H = [H₁; H₂; H₃; ...; H_{n_angles}]")
    print(f"• Each Hᵢ has shape: {C.output_shape[1:]} × {N_voxels}")
    print(f"• Hᵢ represents projection at angle θᵢ")
    print(f"• Blocks are independent (no overlap in measurements)")
    
    print("\n" + "="*60)
    print("MATHEMATICAL FORMULATION")
    print("="*60)
    
    print("""
For a volume x ∈ R^(Nz×Ny×Nx) and sinogram y ∈ R^(N_angles×N_detectors):

1. FLATTENED REPRESENTATION:
   x_vec = x.flatten() ∈ R^(Nz×Ny×Nx)
   y_vec = y.flatten() ∈ R^(N_angles×N_detectors)

2. PROJECTION EQUATION:
   y_vec = H @ x_vec

3. MATRIX STRUCTURE:
   H = [H₁]
       [H₂]
       [H₃]
       [...]
       [H_N_angles]

4. ELEMENT MEANING:
   H[i,j] = intersection length of ray i with voxel j
           = weight of voxel j's contribution to measurement i

5. SPARSITY:
   - Most elements are zero
   - Non-zero pattern depends on projection geometry
   - Each ray intersects only a small fraction of voxels

6. BLOCK DECOMPOSITION:
   - Hᵢ = projection matrix for angle θᵢ
   - yᵢ = Hᵢ @ x_vec (sinogram for angle θᵢ)
   - y = [y₁; y₂; y₃; ...; y_N_angles]
""")
    
    print("="*60)

if __name__ == "__main__":
    analyze_radon_matrix_structure() 