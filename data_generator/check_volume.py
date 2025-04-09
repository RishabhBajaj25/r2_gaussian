import numpy as np
import pyvista as pv

#vol_path = "data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/vol_gt.npy"
vol_path = "/home/rishabh/projects/r2_gaussian/output/seashell/point_cloud/iteration_30000/vol_pred.npy"
# vol_path = "/home/rishabh/projects/r2_gaussian/data/real_dataset/pine/vol_gt.npy"
vol = np.load(vol_path)

plotter = pv.Plotter(window_size=[800, 800], line_smoothing=True, off_screen=False)
plotter.add_volume(vol, cmap="viridis", opacity="linear")
plotter.show()
