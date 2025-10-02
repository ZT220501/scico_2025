# -*- coding: utf-8 -*-
# Copyright (C) 2020-2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""ADMM solver."""

# Needed to annotate a class method that returns the encapsulating class;
# see https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

from typing import List, Optional, Tuple, Union, Callable
from scico.typing import BlockShape, DType, PRNGKey, Shape
import logging
import os
import sys
from datetime import datetime
from io import StringIO

import scico.numpy as snp
from scico.functional import Functional
from scico.linop import LinearOperator
from scico.numpy import Array, BlockArray
from scico.numpy.linalg import norm
from scico.linop import Identity, LinearOperator, operator_norm
from scico.optimize.admm import ADMM
from scico import functional, linop, loss, metric, plot
from scico.util import Timer
import jax

from ._admmaux import (
    FBlockCircularConvolveSolver,
    G0BlockCircularConvolveSolver,
    GenericSubproblemSolver,
    LinearSubproblemSolver,
    MatrixSubproblemSolver,
    SubproblemSolver,
)
from ._common import Optimizer



class ProxJacobiADMMv2(Optimizer):
    r"""Proximal Jacobi Alternating Direction Method of Multipliers (ADMM) algorithm, version 2.
    For reference, see https://link.springer.com/article/10.1007/s10915-016-0318-2.
    In this version, the parameter τ is not updated as in the original proximal Jacobi ADMM, but instead  
    approximated using the matrix norm of the block-wise operator A

    |

    Solve an optimization problem of the form

    .. math::
        \argmin_{\mb{x}} \; f(\mb{x}) + \sum_{i=1}^N g_i(C_i \mb{x}) \;,

    where :math:`f` and the :math:`g_i` are instances of
    :class:`.Functional`, and the :math:`C_i` are
    :class:`.LinearOperator`.

    The optimization problem is solved by introducing the splitting
    :math:`\mb{z}_i = C_i \mb{x}` and solving

    .. math::
        \argmin_{\mb{x}, \mb{z}_i} \; f(\mb{x}) + \sum_{i=1}^N
        g_i(\mb{z}_i) \; \text{such that}\; C_i \mb{x} = \mb{z}_i \;,

    via an ADMM algorithm :cite:`glowinski-1975-approximation`
    :cite:`gabay-1976-dual` :cite:`boyd-2010-distributed` consisting of
    the iterations (see :meth:`step`)

    .. math::
       \begin{aligned}
       \mb{x}^{(k+1)} &= \argmin_{\mb{x}} \; f(\mb{x}) + \sum_i
       \frac{\rho_i}{2} \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i
       \mb{x}}_2^2 \\
       \mb{z}_i^{(k+1)} &= \argmin_{\mb{z}_i} \; g_i(\mb{z}_i) +
       \frac{\rho_i}{2}
       \norm{\mb{z}_i - \mb{u}^{(k)}_i - C_i \mb{x}^{(k+1)}}_2^2  \\
       \mb{u}_i^{(k+1)} &=  \mb{u}_i^{(k)} + C_i \mb{x}^{(k+1)} -
       \mb{z}^{(k+1)}_i  \; .
       \end{aligned}


    Attributes:
        f_list (list of :class:`.Functional`): List of :math:`f_i` 
            functionals. Must be same length as :code:`D_list` and
            :code:`rho_list`.
        g_list (list of :class:`.Functional`): List of :math:`g_i`
            functionals. Must be same length as :code:`D_list` and
            :code:`rho_list`.
        D_list (list of :class:`.LinearOperator`): List of :math:`C_i`
            operators.
        rho_list (list of scalars): List of :math:`\rho_i` penalty
            parameters. Must be same length as :code:`D_list` and
            :code:`g_list`.
        alpha (float): Relaxation parameter.
        u_list (list of array-like): List of scaled Lagrange multipliers
            :math:`\mb{u}_i` at current iteration.
        x (array-like): Solution.
        subproblem_solver (:class:`.SubproblemSolver`): Solver for
            :math:`\mb{x}`-update step.
        z_list (list of array-like): List of auxiliary variables
            :math:`\mb{z}_i` at current iteration.
        z_list_old (list of array-like): List of auxiliary variables
            :math:`\mb{z}_i` at previous iteration.
    """

    def __init__(
        self,
        A_list: List[LinearOperator],
        g_list: List[Functional],
        ρ: float,
        y: Array,
        τ: List[float],
        γ: float,
        tv_weight: float,
        λ: Optional[Array] = None,
        x0_list: Optional[List[Union[Array, BlockArray]]] = None,
        cg_tol: float = 1e-4,
        cg_maxiter: int = 25,
        display_period: int = 5,
        device: str = "gpu",
        with_correction: bool = False,
        α: float = None,
        test_mode: bool = False,
        ground_truth: Optional[Array] = None,
        row_division_num: int = 4,
        col_division_num: int = 8,
        **kwargs,
    ):
        r"""Initialize an :class:`ADMM` object.

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
        self.device = device
        self.N = len(A_list)
        if len(g_list) != self.N:
            raise ValueError(f"len(g_list)={len(g_list)} not equal to len(A_list)={self.N}.")
        
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

        self.ρ: float = ρ                                # ADMM penalty parameter.
        self.y: Array = y                                # Ground truth full sinogram.
        self.γ: float = γ                                # Damping parameter.
        self.τ: List[float] = τ                          # Proximal weight for each block, estimated by the operator norm.
                                                         # Notice that τ is a list of length N, where N is the number of blocks.
        self.tv_weight: float = tv_weight                # TV weight.
        self.display_period: int = display_period        # Display period for the ADMM solver.
        self.with_correction: bool = with_correction     # Whether to use the correction term.

        if λ is None:
            self.λ = snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype)
        else:
            self.λ = λ
        self.λ = jax.device_put(self.λ, self.device)
        self.λ_prev = self.λ.copy()

        self.cg_tol: float = cg_tol
        self.cg_maxiter: int = cg_maxiter

        if x0_list is None:
            input_shape = A_list[0].input_shape
            dtype = A_list[0].input_dtype
            self.x_list = [snp.zeros(input_shape, dtype=dtype) for _ in range(self.N)]
        else:
            self.x_list = x0_list
        self.x_list = [snp.array(jax.device_put(x, self.device)) for x in self.x_list]
        self.x_list_two_prev = self.x_list.copy()
        self.x_list_prev = self.x_list.copy()

        self.res = sum(self.A_list[i](self.x_list[i]) for i in range(self.N)) - self.y
        self.res = jax.device_put(self.res, self.device)
        self.res_two_prev = self.res.copy()
        self.res_prev = self.res.copy()

        # self.res_all = [norm(self.res.reshape(-1), ord=2)]
        # self.x_all = [self.x_list.copy()]

        self.row_division_num: int = row_division_num
        self.col_division_num: int = col_division_num

        super().__init__(**kwargs)



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
        itstat_fields = {"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e", "SNR": "%9.3e", "Constraint": "%9.3e", "Regularization": "%9.3e"}
        itstat_attrib = ["norm_primal_residual()", "norm_dual_residual()", "snr()", "constraint()", "tv()"]

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
        out = 0.0

        Aix_list = []
        for i, x_i in enumerate(self.x_list):
            Aix = self.A_list[i](x_i)
            Aix = jax.device_put(Aix, self.device)
            Aix_list.append(Aix)

        out += self.ρ * snp.linalg.norm(sum(Aix_list) - self.y) ** 2 / 2

        return out

    def tv(self) -> float:
        out = 0.0
        for i in range(self.N):
            out += self.g_list[i](self.x_list[i])
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
        return self.constraint() + self.tv()

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
        if x_list is None:
            x_list = self.x_list

        Aix_list = []
        for i, x_i in enumerate(x_list):
            Aix = self.A_list[i](x_i)
            Aix = jax.device_put(Aix, self.device)
            Aix_list.append(Aix)
        
        residual = self.ρ * snp.linalg.norm(sum(Aix_list) - self.y) ** 2
        
        return snp.sqrt(residual)

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
            dual_residual += self.ρ * self.A_list[i].T(self.λ - self.λ_prev)
        return snp.linalg.norm(dual_residual)

    def step(self):
        r"""Perform a single proximal Jacobi ADMM iteration.

        The primary variable :math:`\mb{x}` is updated by solving the the
        optimization problem

        .. math::
            \mb{x}^{(k+1)} = \argmin_{\mb{x}} \; f(\mb{x}) + \sum_i
            \frac{\rho_i}{2} \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i -
            C_i \mb{x}}_2^2 \;.

        Update auxiliary variables :math:`\mb{z}_i` and scaled Lagrange
        multipliers :math:`\mb{u}_i`. The auxiliary variables are updated
        according to

        .. math::
            \begin{aligned}
            \mb{z}_i^{(k+1)} &= \argmin_{\mb{z}_i} \; g_i(\mb{z}_i) +
            \frac{\rho_i}{2} \norm{\mb{z}_i - \mb{u}^{(k)}_i - C_i
            \mb{x}^{(k+1)}}_2^2  \\
            &= \mathrm{prox}_{g_i}(C_i \mb{x} + \mb{u}_i, 1 / \rho_i) \;,
            \end{aligned}

        and the scaled Lagrange multipliers are updated according to

        .. math::
            \mb{u}_i^{(k+1)} =  \mb{u}_i^{(k)} + C_i \mb{x}^{(k+1)} -
            \mb{z}^{(k+1)}_i \;.
        """
        # Store the previous two iterations' x, dual variable λ, and residual.
        self.x_list_two_prev = self.x_list_prev.copy()
        self.x_list_prev = self.x_list.copy()

        self.res_two_prev = self.res_prev.copy()
        self.res_prev = self.res.copy()

        self.λ_prev = self.λ.copy()

        # x-update for all the subproblem blocks. 
        # Each of the x-update is a proximal operator update.
        Ax_k = sum(self.A_list[i](self.x_list[i]) for i in range(self.N))
        for i in range(self.N):
            grad = self.ρ * self.A_list[i].T(Ax_k - self.y - self.λ / self.ρ)
            # Proximal operator update of the TV norm.
            # Notice that the τ[i] is the proximal weight for the i-th block, instead of using a uniform τ.
            self.x_list[i] = self.g_list[i].prox(self.x_list[i] - 1 / self.τ[i] * grad, self.tv_weight / self.τ[i])
            self.x_list[i] = snp.array(jax.device_put(self.x_list[i], self.device))
        
        # Update the residual.
        # Notice that this in fact can be done parallelly in practice, on self.N GPUs!
        self.res = sum(self.A_list[i](self.x_list[i]) for i in range(self.N)) - self.y
        self.res = jax.device_put(self.res, self.device)

        # Update dual variable λ
        # Notice that in practice, only A_ix_i-y needs to be distributed, which has much smaller size than the full image.
        # This part might be accelerated further.
        self.λ = self.λ - self.γ * self.ρ * (sum(self.A_list[i](self.x_list[i]) for i in range(self.N)) - self.y)
        self.λ = jax.device_put(self.λ, self.device)

        if self.with_correction:
            # Compute the correction step.
            for i in range(self.N):
                self.x_list[i] = self.x_list_prev[i] - self.α * (self.x_list_prev[i] - self.x_list[i])
            self.λ = self.λ_prev - self.α * (self.λ_prev - self.λ)


        #################################################################################################
        # The code below, which was used in the first version, is deprecated.
        # tau is now estimated by the operator norm once initially, instead of dynamically updated.
        #################################################################################################

        # Compute 2/gamma*(lambda^k-lambda^{k+1})'*A(x^k-x^{k+1})
        cross_term = 2 * self.ρ * snp.dot(self.res.reshape(-1), (self.res_prev - self.res).reshape(-1))
        # Compute ||x^k-x^{k+1}||^2_G
        dx_norm = 0
        for i in range(self.N):
            diff = (self.x_list[i] - self.x_list_prev[i]).reshape(-1)
            dx_norm += snp.linalg.norm(diff)**2 * self.τ[i]
        # Compute (2-gamma)/(rho*gamma^2)*||lambda^k-lambda^{k+1}||^2
        d_lambda_norm = (2 - self.γ) * self.ρ * snp.linalg.norm(self.res)**2
        # Compute the lower bound of error decrease: h(u^k,u^{k+1})
        lower_bound = dx_norm + d_lambda_norm + cross_term


        # For tau, set it to 1.05 operator norm might be able to work.
        if lower_bound < 0:
            print("τ is doubled at iteration ", self.itnum)

            self.τ = [τ * 2 for τ in self.τ]

            # Revert back the variables.
            self.x_list = self.x_list_prev
            self.λ = self.λ + self.γ * self.ρ * (sum(self.A_list[i](self.x_list[i]) for i in range(self.N)) - self.y)
            self.λ = jax.device_put(self.λ, self.device)
            # Revert back the stored variables.
            self.x_list = self.x_list_prev.copy()
            self.res = self.res_prev.copy()
            self.x_list_prev = self.x_list_two_prev.copy()
            self.res_prev = self.res_two_prev.copy()
        # # Note: This is a version that currently works, but stuck at a bad SNR. Don't change this.
        elif self.itnum % 100 == 0:
            # Decrase τ after every a pre-defined number of iterations.
            self.τ = [τ / 1.2 for τ in self.τ]
        # elif self.itnum % 20 == 0:
        #     # Decrase τ after every a pre-defined number of iterations.
        #     self.τ = [τ / 2 for τ in self.τ]

        #################################################################################################
        # Proximal parameters update ends here.
        #################################################################################################
            



        

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
            if self.itnum == 100:
                self.ρ = self.ρ * 10
                self.tv_weight = self.tv_weight / 10
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



    # This method is directly copied from _padmm.py, for estimating the parameters of the proximal Jacobi ADMM.
    @staticmethod
    def estimate_parameter(
        A: LinearOperator,
        rho: float,
        factor: Optional[float] = 1.01,
        maxiter: int = 100,
        key: Optional[PRNGKey] = None,
    ) -> float:
        r"""Estimate `tau` parameter of :class:`ProxJacobiADMMv2`.

        Find values of the `tau` parameter of :class:`ProxJacobiADMMv2`
        that respect the constraints

        .. math::
           \tau > \norm{ A }_2^2

        Args:
            A: Linear operator :math:`A`.
            B: Linear operator :math:`B` (if ``None``, :math:`B = -I`
               where :math:`I` is the identity operator).
            factor: Safety factor with which to multiply estimated
               operator norms to ensure strict inequality compliance. If
               ``None``, return the estimated squared operator norms.
            maxiter: Maximum number of power iterations to use in operator
               norm estimation (see :func:`.operator_norm`). Default: 100.
            key: Jax PRNG key to use in operator norm estimation (see
               :func:`.operator_norm`). Defaults to ``None``, in which
               case a new key is created.

        Returns:
            'tau' representing the estimated parameter
            values or corresponding squared operator norm values,
            depending on the value of the `factor` parameter.
        """
        tau = operator_norm(A, maxiter=maxiter, key=key) ** 2
        if factor is None:
            return tau
        else:
            return factor * tau * rho