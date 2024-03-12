import numpy as np
import open3d as o3d

# Function to load point cloud from a text file
def load_point_cloud_from_txt(file_path, delimiter=' '):
    data = np.loadtxt(file_path, delimiter=delimiter)
    points = data[:, :3]  # Only the first three columns are considered (x, y, z coordinates)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray for the original points
    return point_cloud

# Function to extract ISS keypoints with adjusted parameters
def extract_iss_keypoints(point_cloud):
    radius = 0.0015
    gamma_21 = 0.975
    gamma_32 = 0.975
    min_neighbors = 3
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(point_cloud, salient_radius=radius, non_max_radius=radius*2, gamma_21=gamma_21, gamma_32=gamma_32, min_neighbors=min_neighbors)
    return keypoints

# Function to highlight keypoints in the original point cloud and return their indices
def highlight_keypoints(point_cloud, keypoints):
    colors = np.asarray(point_cloud.colors)  # Access existing colors
    keypoint_indices = []
    for keypoint in keypoints.points:
        # Find the index of the keypoint in the original point cloud
        index = np.where(np.all(np.asarray(point_cloud.points) == keypoint, axis=1))[0]
        if len(index) > 0:
            keypoint_indices.append(index[0])
            colors[index[0]] = [1, 0, 0]  # Assign a red color to the keypoint
    point_cloud.colors = o3d.utility.Vector3dVector(colors)  # Update the point cloud colors
    return keypoint_indices

# Function to pick points and display their indices
def pick_points(pcd):
    print("Please pick points using [shift + left click]")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # User picks points here using the visualization window
    vis.destroy_window()
    print("Point indices picked:", vis.get_picked_points())

# Load the point cloud data
file_path = 'C:/algorithms/new/test_wire/point_cloud/target_uni.txt'  # Update this path to where your file is located
point_cloud = load_point_cloud_from_txt(file_path)

# Extract ISS keypoints with adjusted parameters
keypoints = extract_iss_keypoints(point_cloud)

# Highlight the keypoints within the original point cloud and get their indices
keypoint_indices = highlight_keypoints(point_cloud, keypoints)
print("Keypoint Indices:", keypoint_indices)

# Visualize the point cloud with highlighted keypoints and enable picking
pick_points(point_cloud)
