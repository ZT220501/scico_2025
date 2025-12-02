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

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_tangle_phantom, create_3d_foam_phantom
from scico.linop.xray.astra import XRayTransform3D, angle_to_vector
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.optimize import ProximalADMM
from scico.util import device_info

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


"""
Create a ground truth image and projector.
"""
Nx = 512
Ny = 512
Nz = 512

# tangle = snp.array(create_tangle_phantom(Nx, Ny, Nz))
tangle = create_3d_foam_phantom(im_shape=(Nz, Ny, Nx), N_sphere=100)

n_projection = 100  # number of projections
angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
det_spacing = [1.0, 1.0]
det_count = [Nz, max(Nx, Ny)]
vectors = angle_to_vector(det_spacing, angles)

# It would have been more straightforward to use the det_spacing and angles keywords
# in this case (since vectors is just computed directly from these two quantities), but
# the more general form is used here as a demonstration.
C = XRayTransform3D(tangle.shape, det_count=det_count, vectors=vectors)  # CT projection operator

print("shape of tangle is: ", tangle.shape)
print("shape of C is: ", C.shape)

y = C @ tangle  # sinogram
snr_db = int(30)
print(f"SNR of sinogram: {snr_db} dB")
y_noisy, noise = noisy_sinogram(y, snr_db=snr_db, use_variance=True, save_path=None)

r"""
Set up problem and solver. We want to minimize the functional

  $$\mathrm{argmin}_{\mathbf{x}} \; (1/2) \| \mathbf{y} - C \mathbf{x}
  \|_2^2 + \lambda \| D \mathbf{x} \|_{2,1} \;,$$

where $C$ is the X-ray transform and $D$ is a finite difference
operator. This problem can be expressed as

  $$\mathrm{argmin}_{\mathbf{x}, \mathbf{z}} \; (1/2) \| \mathbf{y} -
  \mathbf{z}_0 \|_2^2 + \lambda \| \mathbf{z}_1 \|_{2,1} \;\;
  \text{such that} \;\; \mathbf{z}_0 = C \mathbf{x} \;\; \text{and} \;\;
  \mathbf{z}_1 = D \mathbf{x} \;,$$

which can be written in the form of a standard ADMM problem

  $$\mathrm{argmin}_{\mathbf{x}, \mathbf{z}} \; f(\mathbf{x}) + g(\mathbf{z})
  \;\; \text{such that} \;\; A \mathbf{x} + B \mathbf{z} = \mathbf{c}$$

with

  $$f = 0 \qquad g = g_0 + g_1$$
  $$g_0(\mathbf{z}_0) = (1/2) \| \mathbf{y} - \mathbf{z}_0 \|_2^2 \qquad
  g_1(\mathbf{z}_1) = \lambda \| \mathbf{z}_1 \|_{2,1}$$
  $$A = \left( \begin{array}{c} C \\ D \end{array} \right) \qquad
  B = \left( \begin{array}{cc} -I & 0 \\ 0 & -I \end{array} \right) \qquad
  \mathbf{c} = \left( \begin{array}{c} 0 \\ 0 \end{array} \right) \;.$$

This is a more complex splitting than that used in the
[companion example](ct_astra_3d_tv_admm.rst), but it allows the use of a
proximal ADMM solver in a way that avoids the need for the conjugate
gradient sub-iterations used by the ADMM solver in the
[companion example](ct_astra_3d_tv_admm.rst).
"""
𝛼 = 1e2  # improve problem conditioning by balancing C and D components of A
λ = 2e0 / 𝛼  # ℓ2,1 norm regularization parameter
ρ = 5e-3  # ADMM penalty parameter
maxiter = 500  # number of ADMM iterations
# maxiter = 1

g = functional.ZeroFunctional()
c = y_noisy
f = λ * functional.L1Norm()
D = linop.Identity(input_shape=tangle.shape)
A = C
B = linop.ScaledIdentity(input_shape=A.output_shape, scalar=0.0)      # Zero operator


# FBP initial guess
sinogram_shape = (Nz, n_projection, max(Nx, Ny))
A_full = XRayTransformParallel(
    output_shape=sinogram_shape,
    angles=angles,
    recon_shape=(Nx, Ny, Nz)
)
x0 = A_full.fbp_recon(y_noisy)

mu, nu = ProximalADMM.estimate_parameters(A)
print(f"mu: {mu}, nu: {nu}")

solver = ProximalADMM(
    f=f,
    g=g,
    A=A,
    B=B,
    c=c,
    rho=ρ,
    mu=mu,
    nu=nu,
    maxiter=maxiter,
    itstat_options={"display": True, "period": 50},
    x0=x0
)

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
results_dir = os.path.join(os.path.dirname(__file__), f'results/ct_astra_3d_l1_basis_pursuit_padmm_fbp_initial')
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, f'ct_astra_3d_l1_basis_pursuit_padmm_recon_{n_projection}views_{Nx}x{Ny}x{Nz}_snr{snr_db}_maxiter{maxiter}.png')
fig.savefig(save_path)   # save the figure to file

# input("\nWaiting for input to close figures and exit")
