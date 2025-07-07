import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # GUI backend for matplotlib
from skimage.transform import resize
import os

# --- Configuration ---
fname = "/media/rishabh/SSD_1/Data/UTokyo/CT_20250618_170040_reconstruction/CT_20250618_170040.VOX"
header_size = 466              # Skip file header
dtype = np.int16               # CT data type
depth, height, width = 1024, 800, 1024

# --- Load raw binary file ---
with open(fname, "rb") as f:
    f.seek(header_size)
    data = np.fromfile(f, dtype=dtype)

# --- Check expected size ---
expected = depth * height * width
if data.size != expected:
    raise ValueError(f"要素数ミスマッチ: 読み込み {data.size} vs 期待 {expected}")

# --- Reshape into 3D volume ---
# volume = data.reshape((depth, height, width))
volume = data.reshape((height, depth, width))

# Target size
new_shape = (256, 256, 256)

# Resize with anti-aliasing
volume_resized = resize(
    volume,
    output_shape=new_shape,
    anti_aliasing=True
)

print("Original shape:", volume.shape)
print("Resized shape:", volume_resized.shape)

vol_save_path = "/home/rishabh/Downloads/vol_reshaped_gt.npy"
np.save(vol_save_path, volume_resized)

# # --- (Optional) Visual check of one slice ---
# plt.imshow(volume[depth // 2], cmap='gray')
# plt.axis('off')
# plt.title("Middle Slice")
# plt.show()

# --- Convert to N x 4 array: [x, y, z, density] ---
zz, yy, xx = np.meshgrid(
    np.arange(height),
    np.arange(depth),
    np.arange(width),
    indexing='ij'
)

# Use Open3D convention: x, y, z as spatial coordinates
points_4d1 = np.stack((xx.ravel(), yy.ravel(), zz.ravel(), volume.ravel()), axis=1)



import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator


# Assuming your data is in format: [[x1,y1,z1,density1], [x2,y2,z2,density2], ...]
def reshape_3d_volume(data, new_shape):
    # Extract coordinates and density values
    coords = data[:, :3]  # x, y, z coordinates
    densities = data[:, 3]  # density values

    # Create regular grid from your data
    # First, determine the original grid dimensions
    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])
    z_unique = np.unique(coords[:, 2])

    # Reshape density data into 3D grid
    original_shape = (len(x_unique), len(y_unique), len(z_unique))
    density_grid = densities.reshape(original_shape)

    # Calculate zoom factors for each dimension
    zoom_factors = (new_shape[0] / original_shape[0],
                    new_shape[1] / original_shape[1],
                    new_shape[2] / original_shape[2])

    # Resize using zoom (trilinear interpolation)
    resized_density = zoom(density_grid, zoom_factors, order=1)

    return resized_density

new_shape = (256, 256, 200)  # Desired shape for the new volume
new_vol = reshape_3d_volume(volume, new_shape)

vol_save_path = "/home/rishabh/Downloads/resized_vol_gt.npy"
np.save(vol_save_path, new_vol)

# save every 100th point for testing
# points_4d1 = points_4d1[::100, :]  # Downsample to every 100th point

threshold = np.percentile(points_4d1[:, 3], 95)  # Keep only points with density above 95th percentile

points_4d1 = points_4d1[points_4d1[:, 3] > threshold]
# --- Save to .npy file ---
save_path = "/home/rishabh/Downloads/all_CT_points_4d1.npy"
np.save(save_path, points_4d1)