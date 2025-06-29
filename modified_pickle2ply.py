# This file is for modifying the colors of the gaussian splat as a function of its distance from a camera position

import pickle

import numpy as np
import matplotlib.cm as cm
from plyfile import PlyData, PlyElement
import os.path as osp

def pickle_to_ply(pickle_path, ply_path):
    # Load the pickle file
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Extract necessary attributes
    xyz = data["xyz"]  # Shape (N, 3)
    scale = data["scale"]   # Shape (N, 3)
    rotation = data["rotation"]   # Shape (N, 4)
    opacities = data["density"]   # Shape (N, 1)

    # Ensure opacities is a 1D array
    opacities = opacities.flatten()
    op_percentile = 85  # This means you’ll remove the lowest 85% opacities
    # Compute the threshold value at the given percentile
    op_threshold = np.percentile(opacities, op_percentile)

    scale_percentile = 80  # Retain only smaller 80% of Gaussians (i.e., remove outer blurry blobs)
    # Compute the threshold value at the given percentile
    scale_x_threshold = np.percentile(scale[:,0], scale_percentile)
    scale_y_threshold = np.percentile(scale[:,1], scale_percentile)
    scale_z_threshold = np.percentile(scale[:,2], scale_percentile)


    mask = (opacities.flatten() > op_threshold) & (scale[:, 0] < scale_x_threshold) & (scale[:, 1] < scale_y_threshold) & (scale[:, 2] < scale_z_threshold)

    # Filter all attributes using the mask
    xyz = xyz[mask]
    scale = scale[mask]
    rotation = rotation[mask]
    opacities = opacities[mask]
    opacities = opacities.reshape((-1, 1))

    txt_file_path = '/media/rishabh/SSD_1/Data/Blender_Renders/Watanabe/kiri_3dgs_plugin/torso/central_cam_center.txt'
    cam_center = np.loadtxt(txt_file_path)
    distances = np.linalg.norm(xyz - cam_center, axis=1)

    norm_distances = (distances - distances.min()) / (distances.max() - distances.min())  # Normalize distances to [0,1]
    factor_distances = 1 - norm_distances  # Invert to make closer points brighter

    # opacities*= factor_distances.reshape((-1, 1))  # Scale opacities by distance factor

    N = xyz.shape[0]  # Update N

    # Normalize opacities
    opacity_min, opacity_max = opacities.min(), opacities.max()
    opacity_shifted = opacities - opacity_min  # Shift to positive range
    opacities_norm = opacity_shifted / opacity_shifted.max()  # Normalize to [0,1]

    opacities_max = 5
    opacities_min = -5

    opacities = (opacities_norm + (opacities_min/(opacities_max-opacities_min)) )*(opacities_max-opacities_min)
    opacity_min, opacity_max = opacities.min(), opacities.max()

    # opacities*= factor_distances.reshape((-1, 1))  # Scale opacities by distance factor
    #
    # opacities = (opacities + (opacities_min/(opacities_max-opacities_min)) )*(opacities_max-opacities_min)


    # Map to colors using plasma colormap
    colormap = cm.get_cmap("jet")

    colors = colormap(opacities_norm.flatten())[:, :3]

    # Assign colors to _features_dc
    f_dc = colors.astype(np.float32)
    f_rest = np.zeros((N, 45), dtype=np.float32)  # Placeholder for _features_rest

    # opacities = np.ones((N, 1), dtype=np.float32)*-5 # Placeholder for opacity, set to -5 for visibility in PLY

    normals = np.zeros_like(xyz)  # (N, 3)
    # Define PLY attributes format
    dtype_full = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    # Add fields for f_rest (0 to 44)
    dtype_full.extend([(f"f_rest_{i}", "f4") for i in range(45)])
    # Add remaining attributes
    dtype_full.extend([
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")
    ])
    # Combine attributes
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    # Convert to structured array
    elements = np.empty(N, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    # Save as PLY
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(ply_path)

# Example usage
input_pickle = "/home/rishabh/projects/r2_gaussian/output/head/point_cloud/iteration_30000/point_cloud.pickle"
output_ply = input_pickle.split(".")[0] + "_test.ply"
pickle_to_ply(input_pickle, output_ply)