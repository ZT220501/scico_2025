# -*- coding: utf-8 -*-
# Copyright (C) 2025 by SCICO Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the SCICO package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""X-ray transform LinearOperator wrapping the mbirjax package.

X-ray transform :class:`.LinearOperator` wrapping the
`mbirjax <https://github.com/cabouman/mbirjax>`_ package..
"""

from typing import Optional, Tuple, Union, Any
import numpy as np
import jax.numpy as jnp
from functools import partial
import jax

import scico.numpy as snp
from scico.loss import Loss, SquaredL2Loss
from scico.typing import Shape

from .._diag import Diagonal, Identity
from .._linop import LinearOperator
from collections import namedtuple


try:
    import mbirjax
except ImportError:
    raise ImportError("Could not import mbirjax; please install it.")

class XRayTransformParallel(LinearOperator):
    r"""Parallel beam X-ray transform based on mbirjax.

    Perform parallel beam tomographic projection of an image at specified
    angles, using the `mbirjax <https://github.com/cabouman/mbirjax>`_
    package.
    """

    def __init__(
        self,
        output_shape: Shape,
        angles: snp.Array,
        jit: bool = False,
        partial_reconstruction: bool = False,
        roi_indices: Optional[snp.Array] = None,
        roi_recon_shape: Optional[Shape] = None,
        **kwargs,
    ):
        """
        Args:
            output_shape: Shape of the output array (sinogram).
            angles: Array of projection angles in radians, should be
                increasing.
            jit: If ``True``, call :meth:`.jit()` on this
                :class:`LinearOperator` to jit the forward, adjoint, and
                gram functions. Same as calling :meth:`.jit` after the
                :class:`LinearOperator` is created.
        """
        # Convert angles to svmbir/mbirjax convention.
        # mbirjax_angles = 0.5 * snp.pi - angles
        mbirjax_angles = angles - 0.5 * snp.pi      # Fixed: This should be the correct conversion!
        # Convert the output shape to match the mbirjax convention.
        output_shape_mbirjax = (output_shape[1], output_shape[0], output_shape[2])
        self.model = mbirjax.ParallelBeamModel(output_shape_mbirjax, mbirjax_angles)
        self.model.set_params(no_warning=False, no_compile=False, **kwargs)

        if partial_reconstruction != (roi_indices is not None):
            raise ValueError("Only one of partial_reconstruction and roi_indices can be provided.")
        
        if roi_recon_shape is not None:
            if roi_indices is None:
                raise ValueError("roi_indices must be provided if roi_recon_shape is provided.")
            if (roi_recon_shape[0] * roi_recon_shape[1]) != len(roi_indices):
                raise ValueError("roi_recon_shape must match the shape of the ROI indices.")
        
        # Change the input shape to the ROI reconstruction shape if partial reconstruction is enabled.
        # Otherwise, use the full input shape.
        if partial_reconstruction:
            # Change the input shape to match the scico convention.
            # scico expects (slices, cols, rows), while parameters expect (rows, cols, slices).
            input_shape = (roi_recon_shape[2], roi_recon_shape[1], roi_recon_shape[0])
        else:
            input_shape = self.model.get_params("recon_shape")        
            # Change the input shape to match the scico convention.
            input_shape = (input_shape[2], input_shape[1], input_shape[0])

        # Only consider the indices in the block region for the reconstruction if block_region_indices is provided.
        # Otherwise, use the full indices.
        if partial_reconstruction and roi_indices is not None:
            self.indices = roi_indices
        else:
            self.indices = mbirjax.gen_full_indices(self.model.get_params("recon_shape"), use_ror_mask=False)

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            eval_fn=self.project,
            adj_fn=self.back_project,
            jit=jit,
        )

        self.projector_params = self.create_projector_params()

    def project(self, x: snp.Array) -> snp.Array:
        return self._proj(x, self.indices, self.model.get_params("angles"), self.projector_params)
    
    def back_project(self, y: snp.Array) -> snp.Array:
        return self._bproj(y, self.indices, self.model.get_params("angles"), self.projector_params, coeff_power=1)

    def fbp_recon(self, y: snp.Array) -> snp.Array:
        y = y.transpose(1, 0, 2)
        view_batch_size = len(self.model.get_params("angles"))

        filtered_sinogram = self.model.fbp_filter(y, filter_name="ramp", view_batch_size=view_batch_size)
        filtered_sinogram = filtered_sinogram.transpose(1, 0, 2)        # Transpose to match the scico convention.
        filtered_sinogram = jax.device_put(filtered_sinogram, jax.devices('cpu')[0])    # Put the filtered sinogram on the CPU to avoid memory constraints.
        # By default, the filtered back projection is performed on the CPU instead of the GPU to avoid memory constraints.
        # This will be served as the initial guess for the block proximal Jacobi ADMM solver.
        recon = self._bproj(filtered_sinogram, self.indices, self.model.get_params("angles"), self.projector_params, coeff_power=1, device='cpu')

        return recon

    @staticmethod
    def _proj(
        x: snp.Array,
        pixel_indices: snp.Array,
        angles: snp.Array,
        projector_params: dict,
    ) -> snp.Array:
        
        x = x.transpose(2, 1, 0)
        # Turn the 3D image into voxel values.
        voxel_values = x.copy()
        voxel_values = voxel_values.reshape(-1, x.shape[-1])
        
        sinogram = snp.zeros(projector_params.sinogram_shape)
        # Generate the sinogram for each angle, with the desired pixel indices.
        # Stack all the views into a single sinogram.
        for i in range(len(angles)):
            sinogram_view = mbirjax.ParallelBeamModel.forward_project_pixel_batch_to_one_view(
                voxel_values,
                pixel_indices,
                angles[i],
                projector_params,
            )
            sinogram = sinogram.at[i, :, :].set(sinogram_view)

        # Transpose the sinogram to match the scico convention.
        sinogram = sinogram.transpose(1, 0, 2)
        return snp.array(sinogram)


    @staticmethod
    def _bproj(
        sinogram: snp.Array,
        pixel_indices: snp.Array,
        angles: snp.Array,
        projector_params: dict,
        coeff_power: int,
        device: str = 'gpu',
    ) -> snp.Array:
        """
        Backward projection function for ROI reconstruction.
        
        Args:
            sinogram: Input sinogram with shape (det_rows, views, det_cols)
            pixel_indices: Indices of pixels to reconstruct
            angles: Projection angles
            projector_params: Projector parameters
            coeff_power: Power for coefficient calculation
            device: Device to use for the reconstruction

        Returns:
            Reconstructed image with shape matching input_shape
        """
        # Get the ROI reconstruction shape from projector params
        input_shape = projector_params.recon_roi_shape
        input_shape = (input_shape[2], input_shape[1], input_shape[0])
        if device == 'cpu':
            with jax.default_device(jax.devices('cpu')[0]):
                recon = snp.zeros((input_shape[0] * input_shape[1], input_shape[2]))
        else:
            recon = snp.zeros((input_shape[0] * input_shape[1], input_shape[2]))

        # Generate the sinogram for each angle, with the desired pixel indices.
        # Stack all the views into a single sinogram.
        if device == 'cpu':
            with jax.default_device(jax.devices('cpu')[0]):
                for i in range(len(angles)):
                    recon_cylinder = XRayTransformParallel.back_project_one_view_to_pixel_batch(
                        sinogram[:, i, :],
                        pixel_indices,
                        angles[i],
                        projector_params,
                        coeff_power=coeff_power,
                    )
                    recon += jax.device_put(recon_cylinder, jax.devices('cpu')[0])
                # Transpose the reconstructed 3D image to match the scico convention.
                recon = recon.reshape(input_shape)
                recon = recon.transpose(2, 1, 0)
                result = snp.array(recon)
        else:
            for i in range(len(angles)):
                recon_cylinder = mbirjax.ParallelBeamModel.back_project_one_view_to_pixel_batch(
                    sinogram[:, i, :],
                    pixel_indices,
                    angles[i],
                    projector_params,
                    coeff_power=coeff_power,
                )
                recon += recon_cylinder
            # Transpose the reconstructed 3D image to match the scico convention.
            recon = recon.reshape(input_shape)
            recon = recon.transpose(2, 1, 0)
            result = snp.array(recon)

        return result

    def get_params(self, parameter_names=None) -> Any:
        if parameter_names is None:
            return self.projector_params
        else:
            return self.model.get_params(parameter_names)

    def create_projector_params(self):
        """
        Create the projector_params structure required by forward_project_pixel_batch_to_one_view.
        
        Args:
            model: ParallelBeamModel instance
            
        Returns:
            namedtuple: Projector parameters containing sinogram_shape, recon_shape, and geometry_params
        """
        # Get geometry parameters from the model
        geometry_params = self.model.get_geometry_parameters()
        
        # Get sinogram and reconstruction shapes
        sinogram_shape = self.model.get_params('sinogram_shape')
        recon_shape = self.model.get_params('recon_shape')
        recon_roi_shape = self.input_shape
        
        # Create projector parameters structure
        ProjectorParams = namedtuple('ProjectorParams', [
            'sinogram_shape', 'recon_shape', 'geometry_params', 'recon_roi_shape'
        ])
        
        projector_params = ProjectorParams(
            sinogram_shape=sinogram_shape,
            recon_shape=recon_shape,
            geometry_params=geometry_params,
            recon_roi_shape=recon_roi_shape,
        )
    
        return projector_params

    # Copied from mbirjax/parallel_beam.py and modified to operate on the CPU by default.
    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, angle, projector_params, coeff_power=1):
        """
        Apply parallel back projection to a single sinogram view and return the resulting voxel cylinders.

        Args:
            sinogram_view (2D jax array): one view of the sinogram to be back projected.
                2D jax array of shape (num_det_rows)x(num_det_channels)
            pixel_indices (1D jax array of int):  indices into flattened array of size num_rows x num_cols.
            angle (float): The projection angle in radians for this view.
            projector_params (namedtuple): tuple of (sinogram_shape, recon_shape, get_geometry_params()).
            coeff_power (int): backproject using the coefficients of (A_ij ** coeff_power).
                Normally 1, but should be 2 when computing Hessian diagonal.
        Returns:
            jax array of shape (len(pixel_indices), num_det_rows)
        """
        # Get all the geometry parameters - we use gp since geometry parameters is a named tuple and we'll access
        # elements using, for example, gp.delta_det_channel, so a longer name would be clumsy.
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape

        num_pixels = pixel_indices.shape[0]

        # Get the data needed for horizontal projection, and put everything on the CPU.
        n_p, n_p_center, W_p_c, cos_alpha_p_xy = mbirjax.ParallelBeamModel.compute_proj_data(pixel_indices, angle, projector_params)
        n_p = jax.device_put(n_p, jax.devices('cpu')[0])
        n_p_center = jax.device_put(n_p_center, jax.devices('cpu')[0])
        W_p_c = jax.device_put(W_p_c, jax.devices('cpu')[0])
        cos_alpha_p_xy = jax.device_put(cos_alpha_p_xy, jax.devices('cpu')[0])

        L_max = jnp.minimum(1.0, W_p_c)
        L_max = jax.device_put(L_max, jax.devices('cpu')[0])

        with jax.default_device(jax.devices('cpu')[0]):
            # Allocate the voxel cylinder array
            det_voxel_cylinder = jnp.zeros((num_pixels, num_det_rows))
            # jax.debug.breakpoint(num_frames=1)
            # Do the horizontal projection
            for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
                n = n_p_center + n_offset
                abs_delta_p_c_n = jnp.abs(n_p - n)
                L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
                A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
                A_chan_n *= (n >= 0) * (n < num_det_channels)
                A_chan_n = A_chan_n ** coeff_power
                det_voxel_cylinder = jnp.add(det_voxel_cylinder, A_chan_n.reshape((-1, 1)) * sinogram_view[:, n].T)

        return det_voxel_cylinder


# TODO: Implement the scico wrapper of the cone beam X-ray transform, with block reconstruction.
class XRayTransformCone(LinearOperator):
    r"""Cone beam X-ray transform based on mbirjax.

    Perform cone beam tomographic projection of an image at specified
    angles, using the `mbirjax <https://github.com/cabouman/mbirjax>`_
    package.
    """

    def __init__(
        self,
        output_shape: Shape,
        angles: snp.Array,
        iso_dist: float,
        det_dist: float,
        jit: bool = False,
        **kwargs,
    ):
        """
        Args:
            output_shape: Shape of the output array (sinogram).
            angles: Array of projection angles in radians, should be
                increasing.
            iso_dist: Distance in arbitrary length units (ALU) from
                source to imaging isocenter.
            det_dist: Distance in arbitrary length units (ALU) from
                source to detector.
            jit: If ``True``, call :meth:`.jit()` on this
                :class:`LinearOperator` to jit the forward, adjoint, and
                gram functions. Same as calling :meth:`.jit` after the
                :class:`LinearOperator` is created.
        """
        self.model = mbirjax.ConeBeamModel(output_shape, angles, det_dist, iso_dist)
        self.model.set_params(no_warning=False, no_compile=False, **kwargs)
        input_shape = self.model.get_params("recon_shape")
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            input_dtype=np.float32,
            output_dtype=np.float32,
            eval_fn=self.model.forward_project,
            adj_fn=self.model.back_project,
            jit=jit,
        )