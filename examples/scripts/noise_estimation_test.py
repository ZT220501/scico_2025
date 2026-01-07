import numpy as np
import jax
import os
import jax.numpy as jnp
# Force GPU usage
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
jax.config.update('jax_platform_name', 'gpu')

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom, create_3d_foam_phantom
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.optimize import ProxJacobiADMM, ParallelProxJacobiADMM, ParallelProxJacobiADMMv2, ParallelProxJacobiADMMUnconstrained, ParallelProxJacobiADMMIndicator
from scico.util import device_info, create_roi_indices
from scico.functional import IsotropicTVNorm, L1Norm

import scipy.io
import argparse
import sys
from datetime import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm


def noisy_sinogram(sinogram, snr_db=30, use_variance=True, seed=42):
    """Add Poisson noise to the sinogram, so that SNR is around snr_db dB."""
    # Set the seed for reproducibility.
    np.random.seed(seed)

    if use_variance:
        P_signal = np.mean((sinogram - sinogram.mean())**2)
    else:
        P_signal = np.mean(sinogram**2)

    sigma_n = np.sqrt(P_signal / (10**(snr_db/10.0)))
    noise = np.random.normal(0.0, sigma_n, size=sinogram.shape).astype(np.float32)
    sinogram_noisy = sinogram + noise
    return sinogram_noisy, noise, sigma_n

if __name__ == "__main__":
    gpu_devices = jax.devices('gpu')
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Noise estimation test",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Add arguments
    parser.add_argument('-x', '--Nx', type=int, default=128,
                       help='Image width (default: 128)')
    parser.add_argument('-y', '--Ny', type=int, default=256,
                       help='Image height (default: 256)')
    parser.add_argument('-z', '--Nz', type=int, default=64,
                       help='Image depth (default: 64)')
    parser.add_argument('--n_projection', type=int, default=30,
                        help='Number of projections (default: 30)')
    parser.add_argument('--N_sphere', type=int, default=100,
                        help='Number of spheres in the foam phantom (default: 100)')
    parser.add_argument('--sinogram_snr', type=float, default=30,
                       help='SNR of the sinogram in dB (default: 30)')
    # Parse arguments
    args = parser.parse_args()
    Nx = args.Nx
    Ny = args.Ny
    Nz = args.Nz
    N_sphere = args.N_sphere
    n_projection = args.n_projection
    sinogram_snr = args.sinogram_snr
    x_gt = create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=N_sphere, default_device='gpu')

    angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
    sinogram_shape = (Nz, n_projection, max(Nx, Ny))

    # Create the full sinogram
    # NOTE: In the real-world application, only the full sinogram is available. We can not get the sinogram for each ROI.
    A_full = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        recon_shape=(Nx, Ny, Nz)
    )
    print("Creating the full sinogram...")
    y = A_full @ x_gt
    print("Creating the noisy sinogram...")
    sinogram_snr = int(sinogram_snr)


    L2_norm_errors = []
    y_noisy, noise, sigma_n = noisy_sinogram(y, snr_db=sinogram_snr, use_variance=True, seed=1234)
    for seed in tqdm(range(100)):
        y_noisy_2, noise_2, _ = noisy_sinogram(y, snr_db=sinogram_snr, use_variance=True, seed=seed)
        diff = y_noisy - y_noisy_2
        sigma_n_estimated = snp.linalg.norm(diff) / snp.sqrt(2*max(Nx, Ny)*n_projection*Nz)
        # print("True sigma_n: ", sigma_n)
        # print("Estimated sigma_n: ", sigma_n_estimated)

        true_noise = noise
        true_norm = snp.linalg.norm(true_noise)
        estimated_noise = np.random.normal(0.0, sigma_n_estimated, size=y_noisy.shape).astype(np.float32)
        estimated_norm = snp.linalg.norm(estimated_noise)
        # print("True L2 norm is: ", true_norm)
        # print("Estimated L2 norm is: ", estimated_norm)
        L2_norm_errors.append(abs(true_norm - estimated_norm))

    print("Average L2 norm: ", np.mean(L2_norm_errors))
    print("Standard deviation of L2 norm: ", np.std(L2_norm_errors))

    with open("noise_estimation_test_results.txt", "a") as f:
        f.write("Nx: {}, Ny: {}, Nz: {}, N_sphere: {}, n_projection: {}, sinogram_snr: {}\n".format(Nx, Ny, Nz, N_sphere, n_projection, sinogram_snr))
        f.write("Average L2 norm error: {}\n".format(np.mean(L2_norm_errors)))
        f.write("Standard deviation of L2 norm error: {}\n".format(np.std(L2_norm_errors)))
        f.write("----------------------------------------\n")
