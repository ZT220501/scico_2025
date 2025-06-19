#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory Profiler for SCICO Objects
=================================

This script provides detailed memory usage analysis for SCICO objects,
particularly useful for analyzing large operators like XRayTransform3D.
"""

import sys
import gc
import psutil
import numpy as np
import scico.numpy as snp
from scico.linop.xray.astra import XRayTransform3D
from scico.linop import FiniteDifference
from scico.examples import create_tangle_phantom

def get_detailed_memory_usage(obj, obj_name):
    """Get detailed memory usage of an object."""
    gc.collect()
    
    # Get process memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    # Create reference
    obj_ref = obj
    
    # Get process memory after
    mem_after = process.memory_info().rss
    mem_used = mem_after - mem_before
    
    # Get object size
    obj_size = sys.getsizeof(obj)
    
    print(f"\n{'='*60}")
    print(f"MEMORY ANALYSIS: {obj_name}")
    print(f"{'='*60}")
    print(f"Object type: {type(obj)}")
    print(f"Object size (sys.getsizeof): {obj_size:,} bytes ({obj_size/1024/1024:.2f} MB)")
    print(f"Process memory change: {mem_used:,} bytes ({mem_used/1024/1024:.2f} MB)")
    
    # Get object attributes
    print(f"\nObject attributes:")
    for attr in ['input_shape', 'output_shape', 'input_dtype', 'output_dtype']:
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            print(f"  {attr}: {value}")
    
    # Special analysis for XRayTransform3D
    if isinstance(obj, XRayTransform3D):
        print(f"\nXRayTransform3D specific info:")
        if hasattr(obj, 'angles'):
            print(f"  Number of angles: {len(obj.angles)}")
            print(f"  Angle range: {obj.angles.min():.3f} to {obj.angles.max():.3f} radians")
        if hasattr(obj, 'det_count'):
            print(f"  Detector count: {obj.det_count}")
        if hasattr(obj, 'device'):
            print(f"  Device: {obj.device}")
    
    # Special analysis for FiniteDifference
    if isinstance(obj, FiniteDifference):
        print(f"\nFiniteDifference specific info:")
        if hasattr(obj, 'input_shape'):
            print(f"  Input shape: {obj.input_shape}")
        if hasattr(obj, 'output_shape'):
            print(f"  Output shape: {obj.output_shape}")
    
    return obj_size, mem_used

def analyze_array_memory(arr, arr_name):
    """Analyze memory usage of arrays."""
    if hasattr(arr, 'nbytes'):
        # NumPy array
        size_bytes = arr.nbytes
        dtype = arr.dtype
    elif hasattr(arr, 'shape'):
        # JAX array or similar
        size_bytes = np.prod(arr.shape) * arr.dtype.itemsize
        dtype = arr.dtype
    else:
        size_bytes = sys.getsizeof(arr)
        dtype = "unknown"
    
    print(f"\n{'='*60}")
    print(f"ARRAY ANALYSIS: {arr_name}")
    print(f"{'='*60}")
    print(f"Array type: {type(arr)}")
    print(f"Shape: {arr.shape}")
    print(f"Data type: {dtype}")
    print(f"Memory usage: {size_bytes:,} bytes ({size_bytes/1024/1024:.2f} MB)")
    
    if hasattr(arr, 'device'):
        print(f"Device: {arr.device()}")
    
    return size_bytes

def main():
    """Main function to demonstrate memory profiling."""
    
    print("SCICO Memory Profiler")
    print("="*60)
    
    # Create test objects (similar to your CT script)
    Nx, Ny, Nz = 128, 256, 64
    n_projection = 10
    angles = np.linspace(0, np.pi, n_projection, endpoint=False)
    
    print(f"Test configuration:")
    print(f"  Volume size: {Nx} × {Ny} × {Nz}")
    print(f"  Number of projections: {n_projection}")
    print(f"  Angles: {angles.min():.3f} to {angles.max():.3f} radians")
    
    # Create objects
    print(f"\nCreating test objects...")
    
    # Ground truth
    tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    
    # XRayTransform3D operator
    C = XRayTransform3D(
        tangle.shape, 
        det_count=[Nz, max(Nx, Ny)], 
        det_spacing=[1.0, 1.0], 
        angles=angles
    )
    
    # FiniteDifference operator
    D = FiniteDifference(input_shape=tangle.shape, append=0)
    
    # Sinogram
    y = C @ tangle
    
    # Analyze memory usage
    print(f"\nAnalyzing memory usage...")
    
    # Analyze operators
    c_size, c_mem = get_detailed_memory_usage(C, "C (XRayTransform3D)")
    d_size, d_mem = get_detailed_memory_usage(D, "D (FiniteDifference)")
    
    # Analyze arrays
    tangle_size = analyze_array_memory(tangle, "tangle (ground truth)")
    y_size = analyze_array_memory(y, "y (sinogram)")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"MEMORY USAGE SUMMARY")
    print(f"{'='*60}")
    print(f"C (XRayTransform3D): {c_size:,} bytes ({c_size/1024/1024:.2f} MB)")
    print(f"D (FiniteDifference): {d_size:,} bytes ({d_size/1024/1024:.2f} MB)")
    print(f"tangle (ground truth): {tangle_size:,} bytes ({tangle_size/1024/1024:.2f} MB)")
    print(f"y (sinogram): {y_size:,} bytes ({y_size/1024/1024:.2f} MB)")
    
    total_size = c_size + d_size + tangle_size + y_size
    print(f"\nTotal memory: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    
    # Memory efficiency analysis
    print(f"\nMemory efficiency analysis:")
    print(f"  Volume elements: {Nx * Ny * Nz:,}")
    print(f"  Sinogram elements: {y.size:,}")
    print(f"  Memory per volume element: {tangle_size / (Nx * Ny * Nz):.2f} bytes")
    print(f"  Memory per sinogram element: {y_size / y.size:.2f} bytes")
    
    # GPU memory check
    try:
        import jax
        print(f"\nGPU Memory Information:")
        print(f"  Available devices: {jax.device_count()}")
        for i, device in enumerate(jax.devices()):
            print(f"  Device {i}: {device}")
    except ImportError:
        print("JAX not available for GPU memory checking")

if __name__ == "__main__":
    main() 