import open3d as o3d
import numpy as np
import matplotlib as plt
import os.path as osp

# Load the .npy file containing the point cloud
npy_file = '/home/rishabh/projects/r2_gaussian/data/synthetic_dataset/cone_ntrain_75_angle_360/0_foot_cone/init_0_foot_cone.npy'
point_cloud_data = np.load(npy_file)

# Extract the 3D coordinates (assuming the first three columns are x, y, z)
points = point_cloud_data[:, :3]

# Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Optionally, if the point cloud has density or color information (in the 4th column), you can assign it
# Example: Assign densities as colors
densities = point_cloud_data[:, 3]
colors = plt.cm.viridis(densities / densities.max())[:, :3]  # Normalize densities to [0, 1]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud(npy_file.split('.')[0] + '.ply', pcd)