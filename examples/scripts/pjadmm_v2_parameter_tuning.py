# isort: off
import os
import jax

os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["JAX_PLATFORMS"] = "gpu"
jax.config.update('jax_platform_name', 'gpu')


import numpy as np

import logging
import ray

ray.init(logging_level=logging.ERROR)  # need to call init before jax import: ray-project/ray#44087

import scico.numpy as snp
from scico import functional, linop, loss, metric, plot
from scico.examples import create_circular_phantom
from scico.linop.abel import AbelTransform
from scico.optimize.admm import ADMM, LinearSubproblemSolver
from scico.ray import tune
plot.config_notebook_plotting()

import argparse
from scico.examples import create_3d_foam_phantom
from scico.optimize import ProxJacobiADMMv2
from scico.util import device_info, create_roi_indices
from scico.linop.xray.mbirjax import XRayTransformParallel
from scico.functional import IsotropicTVNorm
from tqdm import tqdm

# Performance evaluation class for Proximal Jacobi ADMM v2
class Trainable(tune.Trainable):
    """Parameter evaluation class."""

    def setup(self, config, x_gt, y, row_division_num, col_division_num, n_projection=30):
        """This method initializes a new parameter evaluation object. It
        is called once when a new parameter evaluation object is created.
        The `config` parameter is a dict of specific parameters for
        evaluation of a single parameter set (a pair of parameters in
        this case). The remaining parameters are objects that are passed
        to the evaluation function via the ray object store.
        """
        # Get arrays passed by tune call.
        self.x_gt, self.y = snp.array(x_gt), snp.array(y)
        Nz, Ny, Nx = self.x_gt.shape
        # Set up problem to be solved.
        self.row_division_num = row_division_num
        self.col_division_num = col_division_num
        self.row_start_indices = []
        self.col_start_indices = []
        self.row_end_indices = []
        self.col_end_indices = []

        self.A_list = []
        angles = np.linspace(0, np.pi, n_projection, endpoint=False)  # evenly spaced projection angles
        for i in range(row_division_num):
            for j in range(col_division_num):
                roi_start_row, roi_end_row = i * Nx // row_division_num, (i + 1) * Nx // row_division_num  # Selected rows
                roi_start_col, roi_end_col = j * Ny // col_division_num, (j + 1) * Ny // col_division_num  # Selected columns

                self.row_start_indices.append(roi_start_row)
                self.col_start_indices.append(roi_start_col)
                self.row_end_indices.append(roi_end_row)
                self.col_end_indices.append(roi_end_col)

                assert roi_start_row >= 0 and roi_start_col >= 0 and roi_end_row <= Nx and roi_end_col <= Ny

                roi_indices = create_roi_indices(Nx, Ny, roi_start_row, roi_end_row, roi_start_col, roi_end_col)

                # Create the mbirjax projector for the current ROI
                A = XRayTransformParallel(
                    output_shape=y.shape,
                    angles=angles,
                    partial_reconstruction=True,
                    roi_indices=roi_indices,
                    roi_recon_shape=(roi_end_row - roi_start_row, roi_end_col - roi_start_col, Nz),
                    recon_shape=(Nx, Ny, Nz)
                )

                # g = IsotropicTVNorm(input_shape=A.input_shape, input_dtype=A.input_dtype)
                # g = L1Norm()

                # Append the mbirjax projector and sinogram to the list
                self.A_list.append(A)
        self.g_list = [IsotropicTVNorm(input_shape=self.A_list[i].input_shape, input_dtype=self.A_list[i].input_dtype) for i in range(len(self.A_list))]
        self.reset_config(config)

    def reset_config(self, config):
        """This method is only required when `scico.ray.tune.Tuner` is
        initialized with `reuse_actors` set to ``True`` (the default). In
        this case, a set of parameter evaluation processes and
        corresponding objects are created once (including initialization
        via a call to the `setup` method), and this method is called when
        switching to evaluation of a different parameter configuration.
        If `reuse_actors` is set to ``False``, then a new process and
        object are created for each parameter configuration, and this
        method is not used.
        """
        gpu_devices = jax.devices('gpu')
        # Extract solver parameters from config dict.
        tv_weight, ρ = config["tv_weight"], config["rho"]
        # Set up parameter-dependent functional.
        τ = [ProxJacobiADMMv2.estimate_parameter(self.A_list[i], maxiter=100, factor=1.01) for i in tqdm(range(len(self.A_list)))]
        self.solver = ProxJacobiADMMv2(
            A_list=self.A_list,
            g_list=self.g_list,
            ρ=ρ,
            y=self.y,
            τ=τ,
            γ=1,
            λ=snp.zeros(self.A_list[0].output_shape, dtype=self.A_list[0].output_dtype),
            x0_list=[snp.array(jax.device_put(self.A_list[i].T(self.y), gpu_devices[0])) for i in range(len(self.A_list))],
            display_period = 1,
            device = gpu_devices[0],
            maxiter = 500,
            itstat_options={"display": True, "period": 10},
            ground_truth = self.x_gt,
            test_mode = True,
            row_division_num = self.row_division_num,
            col_division_num = self.col_division_num,
            tv_weight = tv_weight
        )
        return True

    def step(self):
        """This method is called for each step in the evaluation of a
        single parameter configuration. The maximum number of times it
        can be called is controlled by the `num_iterations` parameter
        in the initialization of a `scico.ray.tune.Tuner` object.
        """
        # Perform 10 solver steps for every ray.tune step
        tangle_recon_list, x_all, res_all = self.solver.solve()
        Nz, Ny, Nx = self.x_gt.shape
        x_tv = snp.zeros(self.x_gt.shape)

        for i in range(self.row_division_num):
            for j in range(self.col_division_num):
                roi_start_row, roi_end_row = i * Nx // self.row_division_num, (i + 1) * Nx // self.row_division_num  # Selected rows
                roi_start_col, roi_end_col = j * Ny // self.col_division_num, (j + 1) * Ny // self.col_division_num  # Selected columns
                tangle_recon = tangle_recon.at[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row].set(tangle_recon_list[i * col_division_num + j])

        return {"psnr": float(metric.psnr(self.x_gt, x_tv))}
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho_lower", type=float, default=1e-3)
    parser.add_argument("--rho_upper", type=float, default=1e-1)
    parser.add_argument("--tv_weight_lower", type=float, default=1e-4)
    parser.add_argument("--tv_weight_upper", type=float, default=1e-2)
    parser.add_argument("--Nx", type=int, default=512)
    parser.add_argument("--Ny", type=int, default=512)
    parser.add_argument("--Nz", type=int, default=64)
    parser.add_argument("--row_division_num", type=int, default=8)
    parser.add_argument("--col_division_num", type=int, default=8)
    parser.add_argument("--n_projection", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--gpu_num", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="pjadmm_v2_parameter_tuning")
    args = parser.parse_args()
    
    # Create a test phantom image, with default size 512x512x64
    gpu_devices = jax.devices('gpu')
    tangle = snp.array(create_3d_foam_phantom(im_shape=(args.Nz, args.Ny, args.Nx), N_sphere=100))
    tangle = snp.array(jax.device_put(tangle, gpu_devices[0]))

    sinogram_shape = (args.Nz, args.n_projection, max(args.Nx, args.Ny))
    A_full = XRayTransformParallel(
        output_shape=sinogram_shape,
        angles=np.linspace(0, np.pi, args.n_projection, endpoint=False),
        recon_shape=(args.Nx, args.Ny, args.Nz)
    )
    y = A_full @ tangle

    # Setup configs for parameter tuning
    # Fix tau and tune the rho and tv_weight parameters.
    config = {"tv_weight": tune.loguniform(args.tv_weight_lower, args.tv_weight_upper), "rho": tune.loguniform(args.rho_lower, args.rho_upper)}
    resources = {"gpu": args.gpu_num}

    tuner = tune.Tuner(
        tune.with_parameters(Trainable, x_gt=tangle, y=y, row_division_num=args.row_division_num, col_division_num=args.col_division_num, n_projection=args.n_projection),
        param_space=config,
        resources=resources,
        metric="psnr",
        mode="max",
        num_samples=args.num_samples,  # perform 100 parameter evaluations
        num_iterations=args.num_iterations,  # perform at most 10 steps for each parameter evaluation
    )
    results = tuner.fit()
    print("results", results)
    ray.shutdown()


    best_result = results.get_best_result()
    best_config = best_result.config
    print(f"Best PSNR: {best_result.metrics['psnr']:.2f} dB")
    print("Best config: " + ", ".join([f"{k}: {v:.2e}" for k, v in best_config.items()]))