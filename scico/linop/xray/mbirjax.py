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
        mbirjax_angles = 0.5 * snp.pi - angles
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
            # eval_fn=self._eval,
            # adj_fn=self._adj,
            eval_fn=self.project,
            adj_fn=self.back_project,
            jit=jit,
        )

        self.projector_params = self.create_projector_params()

    def project(self, x: snp.Array) -> snp.Array:
        return self._proj(x, self.indices, self.model.get_params("angles"), self.projector_params)
    
    def back_project(self, y: snp.Array) -> snp.Array:
        return self._bproj(y, self.indices, self.model.get_params("angles"), self.projector_params, coeff_power=1)

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

    # def _proj_hcb(self, x: snp.Array) -> snp.Array:
    #     """
    #     Host callback wrapper for forward projection.
    #     """
    #     # Transpose the input to match the mbirjax convention.
    #     x = x.transpose(2, 1, 0)
    #     # This put the stuff out of the GPU and to the CPU.
    #     # TODO: Fix this; maybe not use pure_callback?
    #     y = jax.pure_callback(
    #         lambda x: self._proj(
    #             np.array(x),
    #             self.indices,
    #             self.model.get_params("angles"),
    #             self.projector_params,
    #         ),
    #         jax.ShapeDtypeStruct(self.output_shape, self.output_dtype),
    #         x,
    #     )

    #     return y

    @staticmethod
    def _bproj(
        sinogram: snp.Array,
        pixel_indices: snp.Array,
        angles: snp.Array,
        projector_params: dict,
        coeff_power: int,
    ) -> snp.Array:
        """
        Backward projection function for ROI reconstruction.
        
        Args:
            sinogram: Input sinogram with shape (det_rows, views, det_cols)
            pixel_indices: Indices of pixels to reconstruct
            angles: Projection angles
            projector_params: Projector parameters
            coeff_power: Power for coefficient calculation
            
        Returns:
            Reconstructed image with shape matching input_shape
        """
        # Get the ROI reconstruction shape from projector params
        input_shape = projector_params.recon_roi_shape
        input_shape = (input_shape[2], input_shape[1], input_shape[0])
        recon = np.zeros((input_shape[0] * input_shape[1], input_shape[2]))

        # Generate the sinogram for each angle, with the desired pixel indices.
        # Stack all the views into a single sinogram.
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
        return snp.array(recon)
    
    # def _bproj_hcb(self, y: snp.Array) -> snp.Array:
    #     """
    #     Host callback wrapper for backward projection.
    #     """
    #     x = jax.pure_callback(
    #         lambda y: self._bproj(
    #             np.array(y),
    #             self.indices,
    #             self.model.get_params("angles"),
    #             self.projector_params,
    #             coeff_power=1,
    #         ),
    #         jax.ShapeDtypeStruct(self.input_shape, self.input_dtype),
    #         y,
    #     )
    #     return x
    
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