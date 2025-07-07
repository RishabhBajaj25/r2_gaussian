import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def convert_npy_to_ply(npy_path):
    # Load the .npy file
    point_cloud_data = np.load(npy_path)

    # Extract the 3D coordinates (assume first 3 columns are x, y, z)
    points = point_cloud_data[:, :3]

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Use 4th column (e.g., density) as color if it exists
    if point_cloud_data.shape[1] > 3:
        densities = point_cloud_data[:, 3]
        colors = plt.cm.viridis(densities / densities.max())[:, :3]  # Normalize and map to RGB
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd])

    # Save to .ply
    ply_path = os.path.splitext(npy_path)[0] + '.ply'
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved PLY to: {ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy point cloud to .ply using Open3D.")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to .npy point cloud file.")
    args = parser.parse_args()

    convert_npy_to_ply(args.npy_path)
