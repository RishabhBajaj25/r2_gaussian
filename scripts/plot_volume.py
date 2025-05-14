# This is a demo script to plot 3D volume images in Fig. 1 of our paper. A monitor is required.

import numpy as np
import pyvista as pv

# Path to *.npy volume file
# volume_path = "output/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/point_cloud/iteration_30000/vol_gt.npy"
volume_path = '/home/rishabh/projects/r2_gaussian/output/head/point_cloud/iteration_30000/vol_pred.npy'

# Image save path
save_path = "complete_full_head_volume.png"

# Pyvista settings. You may need to tune them.
cpos = [
    (-458.0015547298666, -207.26124611865254, 324.4699978427509),
    (129.02644270914504, 111.50694084289574, 98.55158287937994),
    (0.0, 0.0, 79.59633400474613),
]
window_size = [800, 1000]
colormap = "viridis"

volume = np.load(volume_path)

# create a threshold to "peel" the outer layers off and expose the inner structure
volume[volume < 0.38] = 0.0
half_size = volume.shape[0] // 2
# volume[:half_size, :, :] = 0  # Set half to zero to show the inner structure
clim = [0.0, 1.0]

plotter = pv.Plotter(window_size=window_size, line_smoothing=True, off_screen=True)
plotter.add_volume(volume, cmap=colormap, opacity="linear", clim=clim)
plotter.camera_position = cpos
plotter.show(screenshot=save_path)
