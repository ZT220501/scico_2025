from examples.mbirjax_scripts.BlockADMMBeamModel import BlockADMMBeamModel
import numpy as np

num_angles = 20
sinogram_shape = (num_angles, 128, 256)  # (num_views, num_det_rows, num_det_channels)
angles = np.linspace(0, np.pi, num_angles, endpoint=False)

ct_model = BlockADMMBeamModel(sinogram_shape, angles)


phantom = ct_model.gen_modified_3d_sl_phantom()

transposed_phantom = np.transpose(phantom, (2, 0, 1))
for i in range(num_angles):
    print("Compare slice ", i, ": ", np.all(transposed_phantom[i, :, :]==phantom[:, :, i]))



# Forward project using both MBIRJAX and SCICO
sinogram_mbirjax = ct_model.forward_project(phantom)
sinogram_scico = ct_model.forward_project_scico(np.transpose(phantom, (2, 0, 1)))  # Uses helper method for consistent ordering

print("MBIRJAX sinogram shape: ", sinogram_mbirjax.shape)
print("SCICO sinogram shape: ", sinogram_scico.shape)




# Verify that the shapes are now consistent
print("Shapes match:", sinogram_mbirjax.shape == sinogram_scico.shape)

# Compare the actual values (they should be very close)
if sinogram_mbirjax.shape == sinogram_scico.shape:
    diff = np.abs(sinogram_mbirjax - sinogram_scico)
    print("Max difference between MBIRJAX and SCICO:", np.max(diff))
    print("Mean difference between MBIRJAX and SCICO:", np.mean(diff))
    print("Relative difference (max):", np.max(diff) / (np.max(sinogram_mbirjax) + 1e-10))
else:
    raise ValueError("Shapes do not match")






