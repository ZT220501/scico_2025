#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
3D TV-Regularized Sparse-View CT Reconstruction (Proximal ADMM Solver)
======================================================================

This example demonstrates solution of a sparse-view, 3D CT
reconstruction problem with isotropic total variation (TV)
regularization

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - C \mathbf{x}
  \|_2^2 + \lambda \| D \mathbf{x} \|_{2,1} \;,$$

where $C$ is the X-ray transform (the CT forward projection operator),
$\mathbf{y}$ is the sinogram, $D$ is a 3D finite difference operator,
and $\mathbf{x}$ is the reconstructed image.

In this example the problem is solved via proximal ADMM, while standard
ADMM is used in a [companion example](ct_astra_3d_tv_admm.rst).
"""

import argparse
import os

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom, create_3d_foam_phantom
from scico.linop.xray.astra import XRayTransform3D, angle_to_vector
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize import ProximalADMM
from scico.util import device_info


def noisy_sinogram(sinogram, snr_db=30, use_variance=True, save_path=None):
    """Add Poisson noise to the sinogram, so that SNR is around snr_db dB."""
    # Set the seed for reproducibility.
    seed = 42
    np.random.seed(seed)

    if use_variance:
        P_signal = np.mean((sinogram - sinogram.mean())**2)
    else:
        P_signal = np.mean(sinogram**2)

    sigma_n = np.sqrt(P_signal / (10**(snr_db/10.0)))
    noise = np.random.normal(0.0, sigma_n, size=sinogram.shape).astype(np.float32)
    sinogram_noisy = sinogram + noise
    if save_path is not None:
        save_recon_comparision(sinogram, sinogram_noisy, save_path)
    return sinogram_noisy, noise


def parse_args():
    """Parse command-line arguments for parameter tuning."""
    parser = argparse.ArgumentParser(
        description="3D TV/L1-regularized sparse-view CT reconstruction (Proximal ADMM)."
    )
    # Volume and phantom
    parser.add_argument("--Nx", type=int, default=512, help="Volume size in x")
    parser.add_argument("--Ny", type=int, default=512, help="Volume size in y")
    parser.add_argument("--Nz", type=int, default=512, help="Volume size in z")
    parser.add_argument(
        "--N_sphere",
        type=int,
        default=100,
        help="Number of spheres for 3D foam phantom",
    )
    # Acquisition
    parser.add_argument(
        "--n_projection",
        type=int,
        default=100,
        help="Number of projection angles",
    )
    parser.add_argument(
        "--sinogram_snr",
        type=float,
        default=30.0,
        help="Sinogram SNR in dB",
    )
    # Solver
    parser.add_argument(
        "--regularization",
        type=float,
        default=2.0,
        dest="lambda_reg",
        help="Regularization strength (lambda)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.5,
        help="ADMM penalty parameter",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum ADMM iterations",
    )
    parser.add_argument(
        "--regularization_type",
        type=str,
        default="l1",
        choices=["l1", "tv"],
        help="Regularization type: l1 or tv",
    )
    parser.add_argument(
        "--itstat_period",
        type=int,
        default=50,
        help="Iteration statistics display period",
    )
    # Output
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory for results (default: script_dir/results/...)",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not show figure (for batch/headless runs)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save figure and skip creating results dir",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    Nx = args.Nx
    Ny = args.Ny
    Nz = args.Nz
    n_projection = args.n_projection
    snr_db = args.sinogram_snr
    lambda_reg = args.lambda_reg
    rho = args.rho
    maxiter = args.maxiter
    regularization_type = args.regularization_type
    N_sphere = args.N_sphere

    print(f"Running with configuration: Nx: {Nx}, Ny: {Ny}, Nz: {Nz}, n_projection: {n_projection}, snr_db: {snr_db}, lambda_reg: {lambda_reg}, rho: {rho}, maxiter: {maxiter}, regularization_type: {regularization_type}, N_sphere: {N_sphere}")

    r"""
    Create a ground truth image and projector.
    """
    # tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))
    tangle = create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=N_sphere)

    angles = np.linspace(0, np.pi, n_projection, endpoint=False)
    det_spacing = [1.0, 1.0]
    det_count = [Nz, max(Nx, Ny)]
    vectors = angle_to_vector(det_spacing, angles)

    C = XRayTransform3D(tangle.shape, det_count=det_count, vectors=vectors)

    print("shape of tangle is: ", tangle.shape)
    print("shape of C is: ", C.shape)

    y = C @ tangle
    print(f"SNR of sinogram: {snr_db} dB")
    y_noisy, noise = noisy_sinogram(y, snr_db=snr_db, use_variance=True, save_path=None)

    r"""
    Set up problem and solver.
    """
    print(f"Regularization type: {regularization_type}")

    g = loss.SquaredL2Loss(y=y_noisy)
    if regularization_type == "tv":
        f = lambda_reg * functional.IsotropicTVNorm(input_shape=tangle.shape)
    elif regularization_type == "l1":
        f = lambda_reg * functional.L1Norm()
    else:
        raise ValueError(f"Regularization type {regularization_type} not supported.")
    A = C

    # FBP initial guess
    sinogram_shape = (Nz, n_projection, max(Nx, Ny))
    A_full = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=angles,
        recon_shape=(Nx, Ny, Nz),
    )
    x0 = A_full.fbp_recon(y_noisy)

    mu, nu = ProximalADMM.estimate_parameters(A)
    print(f"mu: {mu}, nu: {nu}")

    if not args.no_save:
        results_dir = args.results_dir
        if results_dir is None:
            results_dir = os.path.join(
                os.path.dirname(__file__),
                f"results/ct_astra_3d_{regularization_type}_padmm_fbp_initial_l2_plus_reg",
            )
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = None

    print("Start of the L2+reg reconstruction")
    solver = ProximalADMM(
        f=f,
        g=g,
        A=A,
        B=None,
        rho=rho,
        mu=mu,
        nu=nu,
        maxiter=maxiter,
        itstat_options={"display": True, "period": args.itstat_period},
        x0=x0,
        x_gt=tangle,
    )

    print(f"Solving on {device_info()}\n")
    tangle_recon = solver.solve()
    hist = solver.itstat_object.history(transpose=True)

    print(
        "TV Restruction\nSNR: %.2f (dB), MAE: %.3f"
        % (metric.snr(tangle, tangle_recon), metric.mae(tangle, tangle_recon))
    )

    # Plot and save
    test_slice = Nz // 2
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    plot.imview(
        tangle[test_slice],
        title="Ground truth (central slice)",
        cmap=plot.cm.Blues,
        cbar=None,
        fig=fig,
        ax=ax[0],
    )
    plot.imview(
        tangle_recon[test_slice],
        title="TV Reconstruction (central slice)\nSNR: %.2f (dB), MAE: %.3f"
        % (metric.snr(tangle, tangle_recon), metric.mae(tangle, tangle_recon)),
        cmap=plot.cm.Blues,
        fig=fig,
        ax=ax[1],
    )
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")

    if not args.no_show:
        fig.show()

    if not args.no_save and results_dir is not None:
        save_path = os.path.join(
            results_dir,
            f"ct_astra_3d_{regularization_type}_padmm_l2_plus_reg_recon_"
            f"{n_projection}views_{Nx}x{Ny}x{Nz}_regularization{lambda_reg}_"
            f"ρ{rho}_snr{snr_db}_maxiter{maxiter}.png",
        )
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    main()
