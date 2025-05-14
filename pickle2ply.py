import pickle
import numpy as np
from plyfile import PlyData, PlyElement



import matplotlib
matplotlib.use("TkAgg")  # Force it to use the Qt5 GUI backend

print(matplotlib.get_backend())
import matplotlib.cm as cm
import matplotlib.pyplot as plt
def pickle_to_ply(pickle_path, ply_path):
    # Load the pickle file
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Extract necessary attributes
    xyz = data["xyz"]  # Shape (N, 3)
    scale = data["scale"]  # Shape (N, 3)
    rotation = data["rotation"]  # Shape (N, 4)
    opacities = data["density"]  # Shape (N, 1)
    N = xyz.shape[0]
    print(opacities.min(), opacities.max())
    # Plot the histogram
    plt.hist(opacities.flatten(), bins=1000)
    plt.title("Opacity Histogram")
    plt.xlabel("Opacity")
    plt.ylabel("Frequency")
    plt.show()

    # # Normalize opacities
    # opacity_min, opacity_max = opacities.min(), opacities.max()
    # opacity_shifted = opacities - opacity_min  # Shift to positive range
    # opacities_norm = opacity_shifted / opacity_shifted.max()  # Normalize to [0,1]
    #
    # # Map to colors using plasma colormap
    colormap = cm.get_cmap("inferno")  # You can change this to any colormap you prefer
    # colors = colormap(opacities_norm.flatten())[:, :3]  # Extract RGB, ignore alpha

    p_min, p_max = np.percentile(opacities, [5, 95])  # Ignore extreme outliers
    opacities_clipped = np.clip(opacities, p_min, p_max)
    opacities_norm = (opacities_clipped - p_min) / (p_max - p_min)
    colors = colormap(opacities_norm.flatten())[:, :3]

    plt.hist(opacities_norm.flatten(), bins=1000)
    plt.title("opacities_norm Histogram")
    plt.xlabel("opacities_norm")
    plt.ylabel("Frequency")
    plt.show()

    opacities_norm_clipped = np.clip(opacities_norm, 0.4, 0.8)
    plt.hist (opacities_norm_clipped.flatten(), bins=1000)
    plt.title("opacities_norm_clipped Histogram")
    plt.xlabel("opacities_norm_clipped")
    plt.ylabel("Frequency")
    plt.show()

    # Assign colors to _features_dc
    f_dc = colors.astype(np.float32)
    f_rest = np.zeros((N, 45), dtype=np.float32)  # Placeholder for _features_rest
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


input_pickle = "/home/rishabh/projects/r2_gaussian/output/chest/point_cloud/iteration_30000/point_cloud.pickle"
output_ply = input_pickle.split(".")[0] + ".ply"

pickle_to_ply(input_pickle, output_ply)
