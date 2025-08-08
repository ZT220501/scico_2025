#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MBIRJAX ROI Extension - Adding ASTRA-like move_vol_geom functionality

This module extends MBIRJAX's ParallelBeamModel class with functionality
similar to ASTRA's move_vol_geom function for handling off-center ROI reconstructions.
"""

import numpy as np
import jax.numpy as jnp
from typing import Literal, Union, overload, Any, Optional, Tuple, Dict
import mbirjax
from jax import jit, grad, vmap, lax
from jax.scipy.optimize import minimize
from jax.scipy.sparse.linalg import cg
from jax.scipy.signal import convolve

from mbirjax.parallel_beam import ParallelBeamModel
from scico.linop.xray.svmbir import XRayTransform

BlockADMMBeamParamNames = mbirjax.ParamNames | Literal['angles']

class BlockADMMBeamModel(ParallelBeamModel):
    """
    BlockADMM ParallelBeamModel with MBIRJAX implementation
    
    This class implements ADMM with TV regularization using MBIRJAX functions.
    """
    
    def __init__(self, sinogram_shape, angles, roi_offset=(0, 0, 0), roi_shape=None):
        """
        Initialize the extended model with ROI support.
        
        Args:
            sinogram_shape: Shape of the sinogram (num_views, num_det_rows, num_det_channels)
            angles: Array of projection angles in radians
        """
        super().__init__(sinogram_shape, angles)
        
        # ADMM parameters
        self.admm_solver = None
        self.tv_weight = 0.1
        self.rho = 1.0
        self.max_iter = 100
        self.tol = 1e-4
        
        # Problem setup
        self.sinogram_data = None
        self.weights = None

        recon_shape = self.get_params('recon_shape')
        self.C = XRayTransform((recon_shape[2], recon_shape[0], recon_shape[1]), angles, num_channels=sinogram_shape[2], center_offset=0.0, is_masked=False, geometry="parallel")
        print("C: ", self.C)

    def forward_project_scico(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward project using SCICO.
        """
        return self.C @ x
        
    def setup_admm_tv_problem(self, sinogram: jnp.ndarray, weights: Optional[jnp.ndarray] = None,
                             tv_weight: float = 0.1, rho: float = 1.0, 
                             max_iter: int = 100, tol: float = 1e-4) -> None:
        """
        Setup ADMM problem with TV regularization using pure MBIRJAX.
        
        The problem is:
            min_x (1/2) ||W^(1/2) * (A*x - y)||^2 + tv_weight * TV(x)
        
        where TV(x) is the total variation of x, A is the forward projection operator,
        y is the sinogram data, and W are the weights.
        
        Args:
            sinogram: Observed sinogram data
            weights: Optional weights for the data fidelity term
            tv_weight: Weight for TV regularization term
            rho: ADMM penalty parameter
            max_iter: Maximum number of ADMM iterations
            tol: Convergence tolerance
        """
        self.tv_weight = tv_weight
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.sinogram_data = sinogram
        
        # Set weights (default to ones if not provided)
        if weights is None:
            self.weights = jnp.ones_like(sinogram)
        else:
            self.weights = weights
    
    def _compute_tv_gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the gradient of the TV functional.
        
        Args:
            x: 3D image array
            
        Returns:
            Gradient of TV functional
        """
        # Compute gradients in all directions
        grad_x = jnp.gradient(x, axis=0)
        grad_y = jnp.gradient(x, axis=1)
        grad_z = jnp.gradient(x, axis=2)
        
        # Compute magnitude of gradient
        grad_mag = jnp.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-10)
        
        # Compute TV gradient (simplified isotropic TV)
        tv_grad_x = jnp.gradient(grad_x / grad_mag, axis=0)
        tv_grad_y = jnp.gradient(grad_y / grad_mag, axis=1)
        tv_grad_z = jnp.gradient(grad_z / grad_mag, axis=2)
        
        return tv_grad_x + tv_grad_y + tv_grad_z
    
    def _solve_x_subproblem(self, x: jnp.ndarray, z: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """
        Solve the x-subproblem in ADMM.
        
        The x-subproblem is:
            min_x (1/2) ||W^(1/2) * (A*x - y)||^2 + (rho/2) ||D*x - z + u||^2
        
        where D is the finite difference operator.
        
        Args:
            x: Current x iterate
            z: Current z iterate
            u: Current u iterate
            
        Returns:
            Updated x iterate
        """
        # Use CG to solve the linear system
        def A_op(x_vec):
            x_reshaped = x_vec.reshape(self.get_params('recon_shape'))
            
            # Data fidelity term: A^T * W * A * x
            # TODO: Replace forward_project and back_project with their sparse versions
            proj = self.forward_project(x_reshaped)
            weighted_proj = self.weights * proj
            backproj = self.back_project(weighted_proj)
            
            # Regularization term: D^T * D * x
            tv_grad = self._compute_tv_gradient(x_reshaped)
            
            result = backproj + self.rho * tv_grad
            return result.flatten()
        
        def b_vec():
            # Right-hand side: A^T * W * y + rho * D^T * (z - u)
            backproj_data = self.back_project(self.weights * self.sinogram_data)
            
            # For simplicity, we'll use a simplified approach
            # In practice, you might want to implement D^T * (z - u) more carefully
            z_minus_u = z - u
            reg_term = self.rho * z_minus_u
            
            result = backproj_data + reg_term
            return result.flatten()
        
        # Solve using CG
        x0 = x.flatten()
        b = b_vec()
        
        # Use a simplified solver for now (in practice, you might want to use CG)
        # For demonstration, we'll use a simple gradient descent step
        def solve_step(x_flat, b_flat):
            # Simple gradient descent step
            residual = A_op(x_flat) - b_flat
            step_size = 0.01
            return x_flat - step_size * residual
        
        # Apply multiple steps
        x_new = x0
        for _ in range(10):  # Simple fixed number of steps
            x_new = solve_step(x_new, b)
        
        return x_new.reshape(self.get_params('recon_shape'))
    
    def _solve_z_subproblem(self, Dx_plus_u: jnp.ndarray) -> jnp.ndarray:
        """
        Solve the z-subproblem in ADMM.
        
        The z-subproblem is:
            min_z tv_weight * ||z||_1 + (rho/2) ||z - (D*x + u)||^2
        
        This is equivalent to soft thresholding.
        
        Args:
            Dx_plus_u: D*x + u
            
        Returns:
            Updated z iterate
        """
        # Soft thresholding
        threshold = self.tv_weight / self.rho
        return jnp.sign(Dx_plus_u) * jnp.maximum(0, jnp.abs(Dx_plus_u) - threshold)
    
    def _compute_finite_differences(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute finite differences of x in all directions.
        
        Args:
            x: 3D image array
            
        Returns:
            Finite differences
        """
        # Compute gradients in all directions
        grad_x = jnp.gradient(x, axis=0)
        grad_y = jnp.gradient(x, axis=1)
        grad_z = jnp.gradient(x, axis=2)
        
        # Stack gradients
        return jnp.stack([grad_x, grad_y, grad_z], axis=-1)
    
    def _compute_objective(self, x: jnp.ndarray, z: jnp.ndarray) -> float:
        """
        Compute the objective function value.
        
        Args:
            x: Current x iterate
            z: Current z iterate
            
        Returns:
            Objective function value
        """
        # Data fidelity term
        proj = self.forward_project(x)
        data_term = 0.5 * jnp.sum(self.weights * (proj - self.sinogram_data)**2)
        
        # TV regularization term
        tv_term = self.tv_weight * jnp.sum(jnp.sqrt(jnp.sum(self._compute_finite_differences(x)**2, axis=-1)))
        
        return data_term + tv_term
    
    def solve_admm_tv(self, x0: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, Dict]:
        """
        Solve the ADMM problem with TV regularization using pure MBIRJAX.
        
        Args:
            x0: Initial guess for reconstruction (optional)
            
        Returns:
            Tuple of (reconstructed_image, solver_info)
        """
        if self.sinogram_data is None:
            raise ValueError("ADMM problem not set up. Call setup_admm_tv_problem() first.")
        
        # Initialize variables
        if x0 is None:
            x0 = jnp.zeros(self.get_params('recon_shape'))
        
        x = x0.copy()
        z = jnp.zeros_like(self._compute_finite_differences(x))
        u = jnp.zeros_like(z)
        
        # ADMM iterations
        objective_history = []
        primal_residual_history = []
        dual_residual_history = []
        
        print(f"Starting ADMM iterations (max_iter={self.max_iter})...")
        
        for iter_idx in range(self.max_iter):
            # Save previous iterates
            x_prev = x.copy()
            z_prev = z.copy()
            
            # x-subproblem
            x = self._solve_x_subproblem(x, z, u)
            
            # z-subproblem
            Dx = self._compute_finite_differences(x)
            z = self._solve_z_subproblem(Dx + u)
            
            # u-update
            u = u + Dx - z
            
            # Compute residuals and objective
            primal_residual = jnp.linalg.norm(Dx - z)
            dual_residual = self.rho * jnp.linalg.norm(z - z_prev)
            objective = self._compute_objective(x, z)
            
            objective_history.append(objective)
            primal_residual_history.append(primal_residual)
            dual_residual_history.append(dual_residual)
            
            # Print progress
            if (iter_idx + 1) % 10 == 0:
                print(f"Iteration {iter_idx + 1}: Objective = {objective:.6f}, "
                      f"Primal = {primal_residual:.6f}, Dual = {dual_residual:.6f}")
            
            # Check convergence
            if primal_residual < self.tol and dual_residual < self.tol:
                print(f"Converged at iteration {iter_idx + 1}")
                break
        
        # Extract solver information
        solver_info = {
            'iterations': iter_idx + 1,
            'objective': objective,
            'primal_residual': primal_residual,
            'dual_residual': dual_residual,
            'converged': iter_idx + 1 < self.max_iter,
            'objective_history': objective_history,
            'primal_residual_history': primal_residual_history,
            'dual_residual_history': dual_residual_history
        }
        
        return x, solver_info
    
    def recon_admm_tv(self, sinogram: jnp.ndarray, weights: Optional[jnp.ndarray] = None,
                     tv_weight: float = 0.1, rho: float = 1.0, 
                     max_iter: int = 100, tol: float = 1e-4,
                     x0: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, Dict]:
        """
        Complete ADMM reconstruction with TV regularization using pure MBIRJAX.
        
        Args:
            sinogram: Observed sinogram data
            weights: Optional weights for the data fidelity term
            tv_weight: Weight for TV regularization term
            rho: ADMM penalty parameter
            max_iter: Maximum number of ADMM iterations
            tol: Convergence tolerance
            x0: Initial guess for reconstruction (optional)
            
        Returns:
            Tuple of (reconstructed_image, solver_info)
        """
        # Setup the problem
        self.setup_admm_tv_problem(sinogram, weights, tv_weight, rho, max_iter, tol)
        
        # Solve the problem
        return self.solve_admm_tv(x0)
    
    def recon_admm_tv_simple(self, sinogram: jnp.ndarray, tv_weight: float = 0.1,
                           max_iter: int = 100, step_size: float = 0.01) -> Tuple[jnp.ndarray, Dict]:
        """
        Simplified ADMM reconstruction using gradient descent.
        
        This is a simpler implementation that uses gradient descent instead of
        solving the exact subproblems. It's faster but may be less accurate.
        
        Args:
            sinogram: Observed sinogram data
            tv_weight: Weight for TV regularization term
            max_iter: Maximum number of iterations
            step_size: Step size for gradient descent
            
        Returns:
            Tuple of (reconstructed_image, solver_info)
        """
        print(f"Starting simplified ADMM reconstruction (max_iter={max_iter})...")
        
        # Initialize
        x = jnp.zeros(self.get_params('recon_shape'))
        objective_history = []
        
        for iter_idx in range(max_iter):
            # Compute gradient of data fidelity term
            proj = self.forward_project(x)
            data_grad = self.back_project(proj - sinogram)
            
            # Compute gradient of TV term
            tv_grad = self._compute_tv_gradient(x)
            
            # Total gradient
            total_grad = data_grad + tv_weight * tv_grad
            
            # Gradient descent step
            x = x - step_size * total_grad
            
            # Compute objective
            objective = 0.5 * jnp.sum((proj - sinogram)**2) + tv_weight * jnp.sum(jnp.sqrt(jnp.sum(self._compute_finite_differences(x)**2, axis=-1)))
            objective_history.append(objective)
            
            # Print progress
            if (iter_idx + 1) % 20 == 0:
                print(f"Iteration {iter_idx + 1}: Objective = {objective:.6f}")
        
        solver_info = {
            'iterations': max_iter,
            'objective': objective,
            'objective_history': objective_history,
            'method': 'simplified_gradient_descent'
        }
        
        return x, solver_info
    

    
    @overload
    def get_params(self, parameter_names: Union[BlockADMMBeamParamNames, list[BlockADMMBeamParamNames]]) -> Any: ...

    def get_params(self, parameter_names) -> Any:
        return super().get_params(parameter_names)