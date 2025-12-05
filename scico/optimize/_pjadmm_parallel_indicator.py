# -*- coding: utf-8 -*-
# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Parallel Proximal Jacobi ADMM solver."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import List, Optional, Tuple, Union, Callable
import logging
import os
from datetime import datetime
from functools import partial

import scico.numpy as snp
from scico.functional import Functional
from scico.linop import LinearOperator
from scico.numpy import Array, BlockArray
from scico.numpy.linalg import norm
from scico.optimize.admm import ADMM
from scico import functional, linop, loss, metric, plot
from scico.util import Timer
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import multiprocessing, threading
import time

from ._admmaux import (
    FBlockCircularConvolveSolver,
    G0BlockCircularConvolveSolver,
    GenericSubproblemSolver,
    LinearSubproblemSolver,
    MatrixSubproblemSolver,
    SubproblemSolver,
)
from ._common import Optimizer


# TODO: Finish writing the actual multi-GPU version of proximal Jacobi ADMM.
# Think more about where/how to store the sinogram acrossGPUs.
class ParallelProxJacobiADMMIndicator(Optimizer):
    r"""Proximal Jacobi Alternating Direction Method of Multipliers (ADMM) algorithm.
    For reference, see https://link.springer.com/article/10.1007/s10915-016-0318-2.

    This is a parallelized version of the ProxJacobiADMM algorithm using multiprocessing.
    We assume that the block-wise PJADMM is executed on different GPUs, and for now
    the sum_{i=1}^N A_ix_i and the dual variable λ are stored on the main CPU.

    TODO: Write the documentation more comprehensively.

    """
    def __init__(
        self,
        A_list: List[LinearOperator],
        g_list: List[Functional],
        ρ: float,
        y: Array,
        τ: float,
        γ: float,
        regularization: float,
        ε: float,
        λ: Optional[Array] = None,
        x0_list: Optional[List[Union[Array, BlockArray]]] = None,
        display_period: int = 5,
        with_correction: bool = False,
        α: float = None,
        test_mode: bool = False,
        ground_truth: Optional[Array] = None,
        row_division_num: int = 4,
        col_division_num: int = 8,
        device_list: List[str] = None,
        num_processes: int = None,
        **kwargs,
    ):
        r"""Initialize an :class:`ProxJacobiADMM` object.

        Args:
            A_list: List of :math:`A_i` operators, represents the partial
                forward sinogram projection operators of each block.
            g_list: List of :math:`g_i` functionals. Must be same length
                 as :code:`D_list` and :code:`rho_list`.
            D_list: List of :math:`C_i` operators.
            rho_list: List of :math:`\rho_i` penalty parameters.
                Must be same length as :code:`D_list` and :code:`g_list`.
            alpha: Relaxation parameter. No relaxation for default 1.0.
            b: Array that represents the ground truth full sinogram.
            x0: Initial value for :math:`\mb{x}`. If ``None``, defaults
                to an array of zeros.
            subproblem_solver: Solver for :math:`\mb{x}`-update step.
                Defaults to ``None``, which implies use of an instance of
                :class:`GenericSubproblemSolver`.
            **kwargs: Additional optional parameters handled by
                initializer of base class :class:`.Optimizer`.
        """
        # Currently we only support the case that both the number of GPUs and the number of processes are provided.
        if num_processes is None:
            raise ValueError("num_processes is required.")
        if device_list is None:
            raise ValueError("device_list is required.")
        self.num_processes = num_processes
        self.device_list = device_list
        # We need the process to be equal to the number of GPUs.
        if self.num_processes != len(self.device_list):
            raise ValueError(f"num_processes={self.num_processes} is not equal to len(device_list)={len(self.device_list)}.")
        if len(self.device_list) != row_division_num * col_division_num:
            raise ValueError(f"len(device_list)={len(self.device_list)} is not equal to number of blocks={row_division_num * col_division_num}.")

        self.N = len(A_list)
        if len(g_list) != self.N:
            raise ValueError(f"len(g_list)={len(g_list)} not equal to len(A_list)={self.N}.")
        if len(x0_list) != self.N:
            raise ValueError(f"len(x0_list)={len(x0_list)} not equal to len(A_list)={self.N}.")
        # Notice that the number of blocks must be less than or equal to the number of GPUs.
        # If the multiple blocks are on the same GPU, they can just be merged into a single block!
        if self.N != len(self.device_list):
            raise ValueError(f"N={self.N} is greater than len(device_list)={len(self.device_list)}.")
        
        if not with_correction and α is not None:
            raise ValueError("alpha is only used when with_correction is True.")
        if with_correction and α is None:
            self.α = 1 - snp.sqrt(self.N / (self.N + 1))
        else:
            self.α: float = α

        if test_mode and ground_truth is None:
            raise ValueError("ground_truth is required in test mode.")
        if not test_mode and ground_truth is not None:
            raise ValueError("ground_truth is only used in test mode.")
        self.test_mode: bool = test_mode
        self.ground_truth: Optional[Array] = ground_truth

        self.A_list: List[LinearOperator] = A_list      # List of partial forward sinogram projection operators typically.
        self.g_list: List[Functional] = g_list          # List of TV regularizers typically.

        self.ρ: float = ρ                               # ADMM penalty parameter.
        self.y_list: List[Array] = [jax.device_put(y, device) for device in self.device_list]  # Ground truth full sinogram on each of the GPUs.
        self.γ: float = γ                                # Damping parameter.
        self.τ: float = τ                                # Proximal weight.
        self.regularization: float = regularization      # Regularization weight.
        self.ε: float = ε                                # Epsilon for the ||Ax-y||_2 <= ε.
        self.display_period: int = display_period        # Display period for the ADMM solver.
        self.with_correction: bool = with_correction    # Whether to use the correction term.

        # Initialize the dual variable λ, and store it on each of the GPUs.
        if λ is None:
            self.λ_list = [jax.device_put(snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype), device) for device in self.device_list]
        else:
            self.λ_list = [jax.device_put(λ, device) for device in self.device_list]
        # Put the λ_list to be sharded on each of the GPUs for convenience of later parallel computation.
        self.λ_list = jax.device_put_sharded(self.λ_list, self.device_list)
        self.λ_prev_list = self.λ_list.copy()
        self.λ_prev_prev_list = self.λ_prev_list.copy()

        # Cut the full image x into blocks x_i, and store each block x_i on each of the GPUs.
        if x0_list is None:
            input_shape = A_list[0].input_shape
            dtype = A_list[0].input_dtype
            self.x_list = [snp.zeros(input_shape, dtype=dtype) for _ in range(self.N)]
        else:
            self.x_list = x0_list
        # Manually put the x_list elements on the corresponding GPUs.
        self.x_list = [jax.device_put(x, device) for x, device in zip(self.x_list, self.device_list)]
        self.x_list_prev = self.x_list.copy()
        self.x_list_prev_prev = self.x_list.copy()

        # Initialize global variable Ax = \sum_{i=1}^N A_i(x_i). Make copies on each of the GPUs.
        # The sparse view sinogram Ax will not be memory costly.
        # In the update iterations, all the sinograms will be updated simultaneously.
        Ax = self.A_list[0](self.x_list[0])
        for i in range(1, self.N):
            Ax = jax.device_put(Ax, self.device_list[i])
            Ax += self.A_list[i](self.x_list[i])
        self.Ax_list = [jax.device_put(Ax, device) for device in self.device_list]

        self.res = self.Ax_list[0] - self.y_list[0]
        self.res_prev = self.res.copy()
        self.res_prev_prev = self.res.copy()
        
        # Initialize the auxiliary indicator variable z, and store it on each of the GPUs.
        self.z_list = [jax.device_put(self.res, device) for device in self.device_list]
        # Put the z_list to be sharded on each of the GPUs for convenience of later parallel computation.
        self.z_list = jax.device_put_sharded(self.z_list, self.device_list)
        self.z_prev_list = self.z_list.copy()
        self.z_prev_prev_list = self.z_prev_list.copy()

        self.row_division_num: int = row_division_num
        self.col_division_num: int = col_division_num

        # Auxiliary functions for the parallelization of the Ax update.
        def _make_A_apply_fn(A_i):
            @jax.jit
            def A_apply_single(x_i):
                return A_i(x_i)
            return A_apply_single
        self._A_apply_fns = [_make_A_apply_fn(Ai) for Ai in self.A_list]

        super().__init__(**kwargs)

    # Parallel x-update step for x on different GPUs.
    def x_update(self, worker_index: int):
        # x-update step.
        grad = self.ρ * self.A_list[worker_index].T(self.Ax_list[worker_index] - self.z_list[worker_index] - self.y_list[worker_index] - self.λ_list[worker_index] / self.ρ)
        self.x_list[worker_index] = self.g_list[worker_index].prox(self.x_list[worker_index] - 1 / self.τ * grad, self.regularization / self.τ)
            
    # Parallel Ax update step for Ax on different GPUs.
    def Ax_update(self):
        """Update the predicted sinogram, by summing over A_ix_i across all GPUs."""
        # Launch A_i(x_i) on each GPU using JIT compilation.
        # After the first time compilation, the Ax_list update are fully parallelized.
        Ax_list = [None] * self.N
        for i in range(self.N):
            Ax_list[i] = self._A_apply_fns[i](self.x_list[i])
        for Ax in Ax_list:
            Ax.block_until_ready()
        # Convert the Ax_list to a sharded array in order to use jax.lax.psum.
        Ax_list = jax.device_put_sharded(Ax_list, self.device_list)
        # Use jax.lax.psum to simultaneously update all the Ax on each of the GPUs.
        self.Ax_list = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(Ax_list)

    # Parallel z-update step for z on different GPUs.
    def z_update(self):
        """Update the auxilliary indicator variable z, by using the proximal operator of the ."""
        # Compute the gradient.
        grad = -self.ρ * (self.Ax_list[0] - self.z_list[0] - self.y_list[0] - self.λ_list[0] / self.ρ)
        z = self.z_list[0] - 1 / self.τ * grad
        # Update the auxilliary indicator variable z by L2 projection onto the feasible set.
        if snp.linalg.norm(z) > self.ε:
            z = z / snp.linalg.norm(z) * self.ε
        else:
            z = z
        self.z_list = [jax.device_put(z, device) for device in self.device_list]

    # Parallel residual update step for residual on the first GPU.
    def residual_update(self):
        """Update the residual, by first updating the predicted sinogram, and then subtracting the ground truth sinogram."""
        # Update the global variable Ax = \sum_{i=1}^N A_ix_i.
        self.Ax_update()
        self.res = self.Ax_list[0] - self.z_list[0] - self.y_list[0]
        # print("The maximum of the indicator variable z is: ", snp.max(abs(self.z_list[0])))

    def get_residual_on_device(self, device_index: int = 0):
        """Get the residual value from a specific GPU device.
        
        Args:
            device_index: Index of the device in device_list (default: 0)
            
        Returns:
            Residual array on the specified device
        """
        if device_index >= len(self.device_list):
            raise ValueError(f"device_index {device_index} out of range for {len(self.device_list)} devices")
        return self.res_gpu_list[device_index]

    def get_residual_sum_on_device(self, device_index: int = 0):
        """Get the sum A_list[i](x_list[i]) for i in range(N) on a specific GPU device.
        
        Args:
            device_index: Index of the device in device_list (default: 0)
            
        Returns:
            Sum of A_list[i](x_list[i]) on the specified device
        """
        if device_index >= len(self.device_list):
            raise ValueError(f"device_index {device_index} out of range for {len(self.device_list)} devices")
        return self.res_gpu_list[device_index] + self.y

    def _working_vars_finite(self) -> bool:
        """Determine where ``NaN`` of ``Inf`` encountered in solve.

        Return ``False`` if a ``NaN`` or ``Inf`` value is encountered in
        a solver working variable.
        """
        for v in (
            [
                self.x,
            ]
            + self.z_list
            + self.u_list
        ):
            if not snp.all(snp.isfinite(v)):
                return False
        return True

    def _objective_evaluatable(self):
        """Determine whether the objective function can be evaluated."""
        # return all([_.has_eval for _ in self.A_list]) and all([_.has_eval for _ in self.g_list])
        return True

    def _itstat_extra_fields(self):
        """Define ADMM-specific iteration statistics fields."""
        # itstat_fields = {"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e", "SNR": "%9.3e", "Constraint": "%9.3e", "Regularization": "%9.3e"}
        # itstat_attrib = ["norm_primal_residual()", "norm_dual_residual()", "snr()", "constraint()", "tv()"]
        itstat_fields = {"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e", "Constraint": "%9.3e", "Regularization": "%9.3e"}
        itstat_attrib = ["norm_primal_residual()", "norm_dual_residual()", "constraint()", "tv()"]

        return itstat_fields, itstat_attrib

    def _state_variable_names(self) -> List[str]:
        # While x is in the most abstract sense not part of the algorithm
        # state, it does form part of the state in pratice due to its use
        # as an initializer for iterative solvers for the x step of the
        # ADMM algorithm.
        return ["x", "z_list", "z_list_old", "u_list"]

    def minimizer(self) -> Union[Array, BlockArray]:
        return self.x_list

    def snr(self) -> float:
        Nz, Ny, Nx = self.ground_truth.shape
        tangle_recon = snp.zeros(self.ground_truth.shape)

        for i in range(self.row_division_num):
            for j in range(self.col_division_num):
                roi_start_row, roi_end_row = i * Nx // self.row_division_num, (i + 1) * Nx // self.row_division_num  # Selected rows
                roi_start_col, roi_end_col = j * Ny // self.col_division_num, (j + 1) * Ny // self.col_division_num  # Selected columns
                tangle_recon = tangle_recon.at[:, roi_start_col:roi_end_col, roi_start_row:roi_end_row].set(self.x_list[i * self.col_division_num + j])

        return metric.snr(self.ground_truth, tangle_recon)

    def constraint(self) -> float:
        return snp.linalg.norm(self.Ax_list[0] - self.y_list[0]) ** 2 / 2

    def tv(self) -> float:
        out = 0.0
        for i in range(self.N):
            out = jax.device_put(out, self.device_list[i])
            out += self.regularization * self.g_list[i](self.x_list[i])
        out = jax.device_put(out, self.device_list[0])
        return out

    def objective(self) -> float:
        r"""Evaluate the objective function.

        Evaluate the objective function

        .. math::
            f(\mb{x}) + \sum_{i=1}^N g_i(\mb{z}_i) \;.

        Note that this form is cheaper to compute, but may have very poor
        accuracy compared with the "true" objective function

        .. math::
            f(\mb{x}) + \sum_{i=1}^N g_i(C_i \mb{x}) \;.

        when the primal residual is large.

        Args:
            x: Point at which to evaluate objective function. If ``None``,
                the objective is  evaluated at the current iterate
                :code:`self.x`.
            z_list: Point at which to evaluate objective function. If
                ``None``, the objective is evaluated at the current iterate
                :code:`self.z_list`.

        Returns:
            Value of the objective function.
        """
        if snp.linalg.norm(self.z_list[0]) <= self.ε * 1.1:
            return self.tv()
        else:
            return snp.inf

    def norm_primal_residual(self, x_list: Optional[Union[Array, BlockArray]] = None) -> float:
        r"""Compute the :math:`\ell_2` norm of the primal residual.

        Compute the :math:`\ell_2` norm of the primal residual

        .. math::
            \left( \sum_{i=1}^N \rho_i \left\| C_i \mb{x} -
            \mb{z}_i^{(k)} \right\|_2^2\right)^{1/2} \;.

        Args:
            x: Point at which to evaluate primal residual. If ``None``,
                the primal residual is evaluated at the current iterate
                :code:`self.x`.

        Returns:
            Norm of primal residual.
        """
        residual = self.ρ * snp.linalg.norm(self.Ax_list[0] - self.z_list[0] - self.y_list[0]) ** 2
        
        return snp.sqrt(residual)

    # TODO: This doesn't seem to be correct. Fix this later.
    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual

        .. math::
            \left\| \sum_{i=1}^N \rho_i C_i^T \left( \mb{z}^{(k)}_i -
            \mb{z}^{(k-1)}_i \right) \right\|_2 \;.

        Returns:
            Norm of dual residual.

        """
        dual_residual = 0.0
        for i in range(self.N):
            dual_residual += self.ρ * self.A_list[i].T(self.λ_list[0] - self.λ_prev_list[0])
        return snp.linalg.norm(dual_residual)

    def step(self):
        r"""Perform a single proximal Jacobi ADMM iteration.

        The primary variable :math:`\mb{x}` is updated by solving the the
        optimization problem

        .. math::
            \mb{x}^{(k+1)} = \argmin_{x_i} \|x_i\|_1 + 
            \left\langle \rho A_i^T \left( Ax^k - y - \frac{\lambda^k}{\rho} \right), x_i \right\rangle 
            + \frac{\tau_i}{2} \|x_i - x^k_i\|_2^2\;.

        Update the scaled Lagrange multipliers :math:`\mb{\lambda}_i` according to

        .. math::
            \mb{\lambda}_i^{(k+1)} =  \mb{\lambda}_i^{(k)} - \gamma \rho_i (\sum_{i=1}^N A_i x^k - y)\;.
        """
        # Store the previous two iterations' x, dual variable λ, and residual.
        self.x_list_prev_prev = self.x_list_prev.copy()
        self.x_list_prev = self.x_list.copy()
        self.res_prev = self.res.copy()
        self.res_prev_prev = self.res_prev.copy()
        self.λ_prev_prev_list = self.λ_prev_list.copy()
        self.λ_prev_list = self.λ_list.copy()

        # Update the predicted sinogram \sum_{i=1}^N A_ix_i for proximal Jacobi ADMM update.
        self.Ax_update()

        # Update each of the x_i in parallel.
        for i in range(self.N):
            self.x_update(i)
        jax.block_until_ready(self.x_list)
        # Update the auxilliary indicator variable z
        self.z_update()
        
        # Update the residual \sum_{i=1}^N A_ix_i - y.
        self.residual_update()

        # Update dual variable λ
        @jax.pmap
        def update_lambda(λ, res):
            return λ - self.γ * self.ρ * res
        res_replicated = jax.device_put_replicated(self.res, self.device_list)
        self.λ_list = update_lambda(self.λ_list, res_replicated)

        # Here the with_correction is not used yet in the parallel version.
        if self.with_correction:
            # Compute the correction step using parallel execution.
            self._parallel_correction_step()
            self.λ = self.λ_prev - self.α * (self.λ_prev - self.λ)

        # Compute 2/gamma*(lambda^k-lambda^{k+1})'*A(x^k-x^{k+1})
        cross_term = 2 * self.ρ * snp.dot(self.res.reshape(-1), (self.res_prev - self.res).reshape(-1))
        # Compute ||x^k-x^{k+1}||^2_G
        dx_norm = 0
        for i in range(self.N):
            diff = (self.x_list[i] - self.x_list_prev[i]).reshape(-1)
            # Move the norm computation result to device 0 before adding to dx_norm
            norm_squared = snp.linalg.norm(diff)**2 * self.τ
            norm_squared = jax.device_put(norm_squared, self.device_list[0])
            dx_norm += norm_squared
        # Compute (2-gamma)/(rho*gamma^2)*||lambda^k-lambda^{k+1}||^2
        d_lambda_norm = (2 - self.γ) * self.ρ * snp.linalg.norm(self.res)**2
        # Compute the lower bound of error decrease: h(u^k,u^{k+1})
        lower_bound = dx_norm + d_lambda_norm + cross_term

        # if lower_bound < 0, double τ to ensure convergence:
        if lower_bound < 0:
            print("τ is doubled at iteration ", self.itnum)
            self.τ = self.τ * 2

            # Revert back the variables
            @jax.pmap
            def update_lambda_back(λ, res):
                return λ + self.γ * self.ρ * res
            res_replicated = jax.device_put_replicated(self.res, self.device_list)
            self.λ_list = update_lambda_back(self.λ_list, res_replicated)
            self.x_list = self.x_list_prev.copy()
            self.x_list_prev = self.x_list_prev_prev.copy()
            self.res = self.res_prev.copy()
            self.res_prev = self.res_prev_prev.copy()
            self.λ_prev_list = self.λ_prev_prev_list.copy()
        elif self.itnum % 10 == 0:
            # Decrase τ after every a pre-defined number of iterations.
            self.τ = self.τ / 1.2

    def solve(
        self,
        callback: Optional[Callable[[Optimizer], None]] = None,
    ) -> Union[Array, BlockArray]:
        r"""Initialize and run the optimization algorithm.

        Initialize and run the opimization algorithm for a total of
        `self.maxiter` iterations.

        Args:
            callback: An optional callback function, taking an a single
              argument of type :class:`Optimizer`, that is called
              at the end of every iteration.

        Returns:
            Computed solution.
        """
        self.timer.start()
        for self.itnum in range(self.itnum, self.itnum + self.maxiter):
            self.step()
            if self.nanstop and not self._working_vars_finite():
                raise ValueError(
                    f"NaN or Inf value encountered in working variable in iteration {self.itnum}."
                    ""
                )
            self.itstat_object.insert(self.itstat_insert_func(self))
            if callback:
                self.timer.stop()
                callback(self)
                self.timer.start()

        self.timer.stop()
        self.itstat_object.end()
        
        return self.minimizer()
