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

import scico.numpy as snp
from scico.functional import Functional
from scico.linop import LinearOperator
from scico.numpy import Array, BlockArray
from scico.numpy.linalg import norm
from scico.optimize.admm import ADMM
from scico import functional, linop, loss, metric, plot
from scico.util import Timer

from ._admmaux import (
    FBlockCircularConvolveSolver,
    G0BlockCircularConvolveSolver,
    GenericSubproblemSolver,
    LinearSubproblemSolver,
    MatrixSubproblemSolver,
    SubproblemSolver,
)
from ._common import Optimizer


class DecentralizedADMM:
    r"""Decentralized Alternating Direction Method of Multipliers (ADMM) algorithm.

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
        D_list: List[LinearOperator],
        rho_list: List[float],
        y: Array,
        alpha: float = 1.0,
        x0_list: Optional[List[Union[Array, BlockArray]]] = None,
        subproblem_solver: Optional[SubproblemSolver] = None,
        display_period: int = 5,
        maxiter_per_block: int = 25,
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
        self.N = len(A_list)
        if len(A_list) != self.N:
            raise ValueError(f"len(A_list)={len(A_list)} not equal to len(g_list)={self.N}.")
        if len(D_list) != self.N:
            raise ValueError(f"len(D_list)={len(D_list)} not equal to len(g_list)={self.N}.")
        # if len(rho_list) != self.N:
        #     raise ValueError(f"len(rho_list)={len(rho_list)} not equal to len(g_list)={self.N}.")

        iter0 = kwargs.pop("iter0", 0)
        self.maxiter: int = kwargs.pop("maxiter", 0)
        self.nanstop: bool = kwargs.pop("nanstop", False)
        itstat_options = kwargs.pop("itstat_options", None)

        if kwargs:
            raise TypeError(f"Unrecognized keyword argument(s) {', '.join([k for k in kwargs])}")

        self.itnum: int = iter0
        self.timer: Timer = Timer()

        self.A_list: List[LinearOperator] = A_list      # List of partial forward sinogram projection operators typically.
        self.g_list: List[Functional] = g_list          # List of L21 regularizers typically.
        self.D_list: List[LinearOperator] = D_list      # List of finite difference operators typically.
        self.rho_list: List[float] = rho_list            # List of ADMM penalty parameters typically. Currently, all the penalty parameters are the SAME.
        self.alpha: float = alpha                        # Relaxation parameter typically.
        self.y: Array = y                                # Ground truth full sinogram.
        self.alpha: float = alpha                        # Relaxation parameter typically.
        self.maxiter_per_block: int = maxiter_per_block  # Maximum number of iterations for each block.
        self.display_period: int = display_period        # Display period for the ADMM solver.

        if subproblem_solver is None:
            subproblem_solver = GenericSubproblemSolver()
        self.subproblem_solver: SubproblemSolver = subproblem_solver
        print("My subproblem solver is: ", self.subproblem_solver)

        if x0_list is None:
            input_shape = D_list[0].input_shape
            dtype = D_list[0].input_dtype
            x0_list = [snp.zeros(input_shape, dtype=dtype) for _ in range(self.N)]

        # Important notice: Here the x-update is the proximal operator update,
        # while the z_avg-update is the CG update.
        self.x_list = x0_list
        self.Ax_avg = self.calculate_Ax_avg(x0_list)
        self.z_avg = sum(A_list[i](x0_list[i]) for i in range(len(A_list))) / len(A_list)           # Notice that the initial value of z_avg is not important, since in the following-up steps,
                                                                                                    # there will be a closed-form solution for z_avg that doesn't depend on the previous value of z_avg.
        self.u_avg = snp.zeros(A_list[0].output_shape, dtype=A_list[0].output_dtype)

        print("Initializing ADMM solvers for each block...")
        self.ADMM_list = []
        for i in range(self.N):
            f = loss.SquaredL2Loss(y=self.A_list[i](x0_list[i])+self.z_avg-self.Ax_avg-self.u_avg, A=self.A_list[i])
            block_solver = ADMM(
                f = f, 
                g_list = [self.g_list[0]],                  # Assume for now that g_list only contains one element, which is λ * functional.L21Norm()
                C_list = [self.D_list[i]],                  # Contains the finite difference operators for each block.
                rho_list = [self.rho_list[0]],              # Assume for now that rho_list only contains one element, which is ρ.
                alpha = alpha, 
                x0 = x0_list[i], 
                maxiter = self.maxiter_per_block,
                subproblem_solver = subproblem_solver,
                itstat_options={"display": True, "period": self.display_period}
            )
            self.ADMM_list.append(block_solver)

        print("DecentralizedADMM initialized successfully.")

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
        return (not self.f or self.f.has_eval) and all([_.has_eval for _ in self.g_list])

    def _itstat_extra_fields(self):
        """Define ADMM-specific iteration statistics fields."""
        itstat_fields = {"Prml Rsdl": "%9.3e", "Dual Rsdl": "%9.3e"}
        itstat_attrib = ["norm_primal_residual()", "norm_dual_residual()"]

        # subproblem solver info when available
        if isinstance(self.subproblem_solver, GenericSubproblemSolver):
            itstat_fields.update({"Num FEv": "%6d", "Num It": "%6d"})
            itstat_attrib.extend(
                ["subproblem_solver.info['nfev']", "subproblem_solver.info['nit']"]
            )
        elif (
            type(self.subproblem_solver) == LinearSubproblemSolver
            and self.subproblem_solver.cg_function == "scico"
        ):
            itstat_fields.update({"CG It": "%5d", "CG Res": "%9.3e"})
            itstat_attrib.extend(
                ["subproblem_solver.info['num_iter']", "subproblem_solver.info['rel_res']"]
            )
        elif (
            type(self.subproblem_solver)
            in [MatrixSubproblemSolver, FBlockCircularConvolveSolver, G0BlockCircularConvolveSolver]
            and self.subproblem_solver.check_solve
        ):
            itstat_fields.update({"Slv Res": "%9.3e"})
            itstat_attrib.extend(["subproblem_solver.accuracy"])

        return itstat_fields, itstat_attrib

    def _state_variable_names(self) -> List[str]:
        # While x is in the most abstract sense not part of the algorithm
        # state, it does form part of the state in pratice due to its use
        # as an initializer for iterative solvers for the x step of the
        # ADMM algorithm.
        return ["x", "z_list", "z_list_old", "u_list"]

    def minimizer(self) -> Union[Array, BlockArray]:
        return self.x_list

    def objective(
        self,
        x: Optional[Union[Array, BlockArray]] = None,
        z_list: Optional[List[Union[Array, BlockArray]]] = None,
    ) -> float:
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
        if (x is None) != (z_list is None):
            raise ValueError("Both or neither of x and z_list must be supplied.")
        if x is None:
            x = self.x
            z_list = self.z_list
        assert z_list is not None
        out = 0.0
        if self.f:
            out += self.f(x)
        for g, z in zip(self.g_list, z_list):
            out += g(z)
        return out

    def norm_primal_residual(self, x: Optional[Union[Array, BlockArray]] = None) -> float:
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
        if x is None:
            x = self.x

        sum = 0.0
        for rhoi, Ci, zi in zip(self.rho_list, self.D_list, self.z_list):
            sum += rhoi * norm(Ci(self.x) - zi) ** 2
        return snp.sqrt(sum)

    def norm_dual_residual(self) -> float:
        r"""Compute the :math:`\ell_2` norm of the dual residual.

        Compute the :math:`\ell_2` norm of the dual residual

        .. math::
            \left\| \sum_{i=1}^N \rho_i C_i^T \left( \mb{z}^{(k)}_i -
            \mb{z}^{(k-1)}_i \right) \right\|_2 \;.

        Returns:
            Norm of dual residual.

        """
        sum = 0.0
        for rhoi, zi, ziold, Ci in zip(self.rho_list, self.z_list, self.z_list_old, self.D_list):
            sum += rhoi * Ci.adj(zi - ziold)
        return norm(sum)

    def step(self):
        r"""Perform a single ADMM iteration.

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
        print(f"Iteration {self.itnum} starts.")

        # x-update for all the subproblem blocks. 
        # Notice that each of the x-update is a ADMM problem itself, but they can be solved in parallel on different GPUs.
        for i in range(len(self.ADMM_list)):
            self.x_list[i] = (self.ADMM_list[i]).solve()

        Ax_avg = sum(self.A_list[i](self.x_list[i]) for i in range(len(self.A_list))) / len(self.A_list)

        # Closed-form solution for z_avg update.
        self.z_avg = 1 / (self.N + self.rho_list[0]) * (self.y + self.rho_list[0] * Ax_avg + self.rho_list[0] * self.u_avg)
        # Update for the u_avg.
        self.u_avg = self.u_avg + Ax_avg - self.z_avg

        # Update the ADMM solvers for each block.
        for i in range(len(self.ADMM_list)):
            new_solver = ADMM(
                f = loss.SquaredL2Loss(y=self.A_list[i](self.x_list[i])+self.z_avg-self.Ax_avg-self.u_avg, A=self.A_list[i]),
                g_list = [self.g_list[0]],
                C_list = [self.D_list[i]],
                rho_list = [self.rho_list[0]],
                alpha = self.alpha,
                x0 = self.x_list[i],
                maxiter = self.maxiter_per_block,
                subproblem_solver = self.subproblem_solver,
                itstat_options={"display": True, "period": self.display_period}
            )
            self.ADMM_list[i] = new_solver

        

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
            # self.itstat_object.insert(self.itstat_insert_func(self))
            if callback:
                self.timer.stop()
                callback(self)
                self.timer.start()
        self.timer.stop()
        self.itnum += 1
        # self.itstat_object.end()
        return self.minimizer()

    def calculate_Ax_avg(self, x_list: List[Union[Array, BlockArray]]):
        Ax_list = [self.A_list[i](x_list[i]) for i in range(len(self.A_list))]
        return sum(Ax_list) / len(Ax_list)

