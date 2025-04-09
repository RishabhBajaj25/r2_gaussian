import pickle
import numpy as np
from plyfile import PlyData

def ply_to_pickle(ply_path, pickle_path):
    # Load the PLY file
    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data["vertex"]

    # Extract required fields
    xyz = np.vstack((vertex_data["x"], vertex_data["y"], vertex_data["z"])).T
    scale = np.vstack((vertex_data["scale_x"], vertex_data["scale_y"], vertex_data["scale_z"])).T
    rotation = np.vstack((
        vertex_data["rotation_w"], vertex_data["rotation_x"], vertex_data["rotation_y"], vertex_data["rotation_z"]
    )).T

    # Construct the dictionary for Pickle
    out = {
        "xyz": xyz,
        "density": None,  # No density data in PLY, so set it to None
        "scale": scale,
        "rotation": rotation,
        "scale_bound": None  # Not available in PLY, setting to None
    }

    # Save to pickle
    with open(pickle_path, "wb") as f:
        pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

# Example usage
# ply_to_pickle("/home/rishabh/projects/gaussian-splatting/output/table_2/point_cloud/iteration_30000/point_cloud.ply", "/home/rishabh/projects/gaussian-splatting/output/table_2/point_cloud/iteration_30000/test.ply")
ply_to_pickle("/home/rishabh/projects/gaussian-splatting/output/foot/point_cloud/iteration_30000/point_cloud.ply", "/home/rishabh/projects/r2_gaussian/output/foot/point_cloud/iteration_30000/test2.ply")
