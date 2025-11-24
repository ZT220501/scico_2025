#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

r"""
3D TV-Regularized Sparse-View CT Reconstruction (ADMM Solver)
=============================================================

This example demonstrates solution of a sparse-view, 3D CT
reconstruction problem with isotropic total variation (TV)
regularization

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - C \mathbf{x}
  \|_2^2 + \lambda \| D \mathbf{x} \|_{2,1} \;,$$

where $C$ is the X-ray transform (the CT forward projection operator),
$\mathbf{y}$ is the sinogram, $D$ is a 3D finite difference operator,
and $\mathbf{x}$ is the reconstructed image.

In this example the problem is solved via ADMM, while proximal
ADMM is used in a [companion example](ct_astra_3d_tv_padmm.rst).
"""

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom, create_3d_foam_phantom
from scico.linop.xray.astra import XRayTransform3D
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.linop.xray.svmbir import XRayTransform
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.util import device_info

from astra.functions import move_vol_geom
import os


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

def save_recon_comparision(x_gt, x_recon, save_path):
    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(7, 6))
    assert x_gt.shape == x_recon.shape
    Nz = x_gt.shape[0]
    test_slice = Nz // 2
    plot.imview(
        x_gt[test_slice],
        title="Ground truth (central slice)",
        cmap=plot.cm.Blues,
        cbar=None,
        fig=fig,
        ax=ax[0],
    )
    plot.imview(
        x_recon[test_slice],
        title="FBP Reconstruction (central slice)\nSNR: %.2f (dB), MAE: %.3f"
        % (metric.snr(x_gt, x_recon), metric.mae(x_gt, x_recon)),
        cmap=plot.cm.Blues,
        fig=fig,
        ax=ax[1],
    )
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(ax[1].get_images()[0], cax=cax, label="arbitrary units")
    fig.show()

    # Save the figure
    fig.savefig(save_path)   # save the figure to file

"""
Create a ground truth image and projector.
"""
Nx = 512    
Ny = 512
Nz = 512

# tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))
tangle = create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=100, default_device='gpu')

n_projection = 100  # number of projections
angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
C = XRayTransform3D(
    tangle.shape, det_count=[Nz, max(Nx, Ny)], det_spacing=[1.0, 1.0], angles=angles
)  # CT projection operator
y = C @ tangle  # sinogram
snr_db = int(40)
y_noisy, noise = noisy_sinogram(y, snr_db=snr_db, use_variance=True, save_path=None)



"""
Set up problem and solver.
"""
λ = 2e0  # ℓ2,1 norm regularization parameter
ρ = 5e0  # ADMM penalty parameter
maxiter = 100  # number of ADMM iterations
cg_tol = 1e-4  # CG relative tolerance
cg_maxiter = 25  # maximum CG iterations per ADMM iteration

########################################################################
# The original formulation.
# The append=0 option makes the results of horizontal and vertical
# finite differences the same shape, which is required for the L21Norm,
# which is used so that g(Ax) corresponds to isotropic TV.
# D = linop.FiniteDifference(input_shape=tangle.shape, append=0)
D = linop.Identity(input_shape=tangle.shape)
# g = λ * functional.L21Norm()
g = λ * functional.L1Norm()
# f = loss.SquaredL2Loss(y=y, A=C)
f = loss.SquaredL2Loss(y=y_noisy, A=C)
########################################################################

print("D is ", D)
print("g is ", g)
print("f is ", f)

# FBP initial guess
sinogram_shape = (Nz, n_projection, max(Nx, Ny))
A_full = XRayTransformParallel(
    output_shape=sinogram_shape,
    angles=angles,
    recon_shape=(Nx, Ny, Nz)
)
x0 = A_full.fbp_recon(y_noisy)


# TODO: Do the ADMM formulation to be the same as the PJADMM solver.
solver = ADMM(
    f=f,        # replace by the TV term
    g_list=[g], # zero functional
    C_list=[D],
    rho_list=[ρ],
    x0=x0,
    maxiter=maxiter,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
    itstat_options={"display": True, "period": 5},
)

# solver = ADMM(
#     f=f,
#     g_list=[g],
#     C_list=[D],
#     rho_list=[ρ],
#     x0=x0,
#     maxiter=maxiter,
#     subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": cg_tol, "maxiter": cg_maxiter}),
#     itstat_options={"display": True, "period": 5},
# )


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
tangle_recon = solver.solve()
hist = solver.itstat_object.history(transpose=True)

print(
    "TV Restruction\nSNR: %.2f (dB), MAE: %.3f"
    % (metric.snr(tangle, tangle_recon), metric.mae(tangle, tangle_recon))
)


"""
Show the recovered image.
"""
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
fig.show()


# Save the figure
results_dir = os.path.join(os.path.dirname(__file__), f'results/ct_astra_3d_tv_admm')
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, f'ct_astra_3d_tv_admm_recon_{n_projection}views_{Nx}x{Ny}x{Nz}_snr{snr_db}_maxiter{maxiter}_l1norm.png')
fig.savefig(save_path)   # save the figure to file

input("\nWaiting for input to close figures and exit")



# import scico.numpy as snp
# from scico.linop.xray.svmbir import XRayTransform
# from scico.linop.xray.astra import XRayTransform3D
# from scico.linop.xray.mbirjax import XRayTransformParallel
# from scico.examples import create_tangle_phantom

# # Setup parameters
# Nx = 128
# Ny = 256
# Nz = 64

# num_angles = 50
# angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)

# x_gt = snp.array(create_tangle_phantom(Nx, Ny, Nz))

# # Create scico's astra projector.
# A_astra = XRayTransform3D(
#     x_gt.shape, det_count=[Nz, max(Nx, Ny)], det_spacing=[1.0, 1.0], angles=angles
# )

# # Create scico's svmbir projector. Angles are shifted to match the svmbir convention.
# A_svmbir = XRayTransform(x_gt.shape, 0.5 * snp.pi - angles, max(Nx, Ny))

# # Create mbirjax projector, with the same angles as svmbir projector.
# sinogram_shape = (Nz, num_angles, max(Nx, Ny))
# A_mbirjax = XRayTransformParallel(
#     output_shape=sinogram_shape,
#     angles=angles,
#     recon_shape=(Nx, Ny, Nz)
# )


# # Generate sinogram using SCICO's interface
# astra_sinogram = A_astra @ x_gt
# svmbir_sinogram = A_svmbir @ x_gt
# svmbir_sinogram = svmbir_sinogram.transpose(1, 0, 2)
# mbirjax_sinogram = A_mbirjax @ x_gt

# print("Max difference between astra and mbirjax sinograms: ", snp.max(snp.abs(astra_sinogram - mbirjax_sinogram)))
# print("Max difference between astra and svmbir sinograms: ", snp.max(snp.abs(astra_sinogram - svmbir_sinogram)))
# print("Max difference between mbirjax and svmbir sinograms: ", snp.max(snp.abs(mbirjax_sinogram - svmbir_sinogram)))

# backproject_astra = A_astra.T(astra_sinogram)
# backproject_svmbir = A_svmbir.T(astra_sinogram.transpose(1, 0, 2))
# backproject_mbirjax = A_mbirjax.T(astra_sinogram)

# print("Max difference between astra and mbirjax backproject: ", snp.max(snp.abs(backproject_astra - backproject_mbirjax)))
# print("Max difference between astra and svmbir backproject: ", snp.max(snp.abs(backproject_astra - backproject_svmbir)))
# print("Max difference between mbirjax and svmbir backproject: ", snp.max(snp.abs(backproject_mbirjax - backproject_svmbir)))




