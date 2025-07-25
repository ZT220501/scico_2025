#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SCICO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
PPP (with BM3D) CT Reconstruction (ADMM with CG Subproblem Solver)
==================================================================

This example demonstrates solution of a tomographic reconstruction problem
using the Plug-and-Play Priors framework
:cite:`venkatakrishnan-2013-plugandplay2`, using BM3D
:cite:`dabov-2008-image` as a denoiser and SVMBIR :cite:`svmbir-2020` for
tomographic projection.

There are two versions of this example, solving the same problem in two
different ways. This version uses the data fidelity term as the ADMM $f$,
and thus the optimization with respect to the data fidelity uses CG rather
than the prox of the `SVMBIRSquaredL2Loss` functional, as in the
[other version](ct_svmbir_ppp_bm3d_admm_prox.rst).
"""

import numpy as np

import matplotlib.pyplot as plt
import svmbir
from xdesign import Foam, discrete_phantom

import scico.numpy as snp
from scico import metric, plot
from scico.functional import BM3D, NonNegativeIndicator
from scico.linop import Diagonal, Identity
from scico.linop.xray.svmbir import SVMBIRSquaredL2Loss, XRayTransform
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.util import device_info

"""
Generate a ground truth image.
"""
N = 256  # image size
density = 0.025  # attenuation density of the image
np.random.seed(1234)
x_gt = discrete_phantom(Foam(size_range=[0.075, 0.005], gap=2e-3, porosity=1.0), size=N - 10)
x_gt = x_gt / np.max(x_gt) * density
x_gt = np.pad(x_gt, 5)
x_gt[x_gt < 0] = 0


"""
Generate tomographic projector and sinogram.
"""
num_angles = int(N / 2)
num_channels = N
angles = snp.linspace(0, snp.pi, num_angles, endpoint=False, dtype=snp.float32)
A = XRayTransform(x_gt.shape, angles, num_channels)
sino = A @ x_gt


"""
Impose Poisson noise on sinogram. Higher max_intensity means less noise.
"""
max_intensity = 2000
expected_counts = max_intensity * np.exp(-sino)
noisy_counts = np.random.poisson(expected_counts).astype(np.float32)
noisy_counts[noisy_counts == 0] = 1  # deal with 0s
y = -np.log(noisy_counts / max_intensity)


"""
Reconstruct using default prior of SVMBIR :cite:`svmbir-2020`.
"""
weights = svmbir.calc_weights(y, weight_type="transmission")

x_mrf = svmbir.recon(
    np.array(y[:, np.newaxis]),
    np.array(angles),
    weights=weights[:, np.newaxis],
    num_rows=N,
    num_cols=N,
    positivity=True,
    verbose=0,
)[0]


"""
Set up an ADMM solver.
"""
y = snp.array(y)
x0 = snp.array(x_mrf)
weights = snp.array(weights)

ρ = 15  # ADMM penalty parameter
σ = density * 0.18  # denoiser sigma

f = SVMBIRSquaredL2Loss(y=y, A=A, W=Diagonal(weights), scale=0.5)
g0 = σ * ρ * BM3D()
g1 = NonNegativeIndicator()

solver = ADMM(
    f=f,
    g_list=[g0, g1],
    C_list=[Identity(x_mrf.shape), Identity(x_mrf.shape)],
    rho_list=[ρ, ρ],
    x0=x0,
    maxiter=20,
    subproblem_solver=LinearSubproblemSolver(cg_kwargs={"tol": 1e-4, "maxiter": 100}),
    itstat_options={"display": True, "period": 5},
)


"""
Run the solver.
"""
print(f"Solving on {device_info()}\n")
x_bm3d = solver.solve()
hist = solver.itstat_object.history(transpose=True)


"""
Show the recovered image.
"""
norm = plot.matplotlib.colors.Normalize(vmin=-0.1 * density, vmax=1.2 * density)
fig, ax = plt.subplots(1, 3, figsize=[15, 5])
plot.imview(img=x_gt, title="Ground Truth Image", cbar=True, fig=fig, ax=ax[0], norm=norm)
plot.imview(
    img=x_mrf,
    title=f"MRF (PSNR: {metric.psnr(x_gt, x_mrf):.2f} dB)",
    cbar=True,
    fig=fig,
    ax=ax[1],
    norm=norm,
)
plot.imview(
    img=x_bm3d,
    title=f"BM3D (PSNR: {metric.psnr(x_gt, x_bm3d):.2f} dB)",
    cbar=True,
    fig=fig,
    ax=ax[2],
    norm=norm,
)
fig.show()

import os
my_path = os.path.dirname(os.path.abspath(__file__))  # Figures out the absolute path for you in case your working directory moves around.

plot.savefig(os.path.join(my_path, "results/ct_svmbir_ppp_bm3d_admm_cg.png"))


"""
Plot convergence statistics.
"""
plot.plot(
    snp.vstack((hist.Prml_Rsdl, hist.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
)

plot.savefig(os.path.join(my_path, "results/ct_svmbir_ppp_bm3d_admm_cg_residuals.png"))


input("\nWaiting for input to close figures and exit")
