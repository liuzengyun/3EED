import cv2
import json
import numpy as np
import open3d as o3d
import glob
import os

def save_colored_bin(rgbd_points, bin_path, suffix="_rgb"):
    """
    rgbd_points: (M,7) [x,y,z,reflectance,r,g,b] float32
    bin_path: 原始 lidar.bin 路径
    """

    out_dir = os.path.dirname(bin_path)
    base = os.path.basename(bin_path).replace(".bin", "")
    out_path = os.path.join(out_dir, base + suffix + ".bin")

    rgbd_points.astype(np.float32).tofile(out_path)
    print(f"[INFO] colored bin saved to: {out_path}")

    return out_path

def project_lidar_bin_to_image(
    bin_path,
    image_path,
    json_path,
    return_depth=False,
    return_uv=False
):
    """
    Project LiDAR .bin point cloud to image plane and colorize points.

    Args:
        bin_path (str): path to LiDAR .bin (x,y,z,reflectance) float32
        image_path (str): path to RGB image
        json_path (str): path to JSON metadata file
        return_depth (bool): whether to return camera depth Z
        return_uv (bool): whether to return pixel coordinates

    Returns:
        rgbd_points (M,7): [X,Y,Z,R,G,B,reflectance] in world coords
        (optional) depth_cam (M,)
        (optional) uv (M,2)
    """

    # ---------- load LiDAR bin ----------
    lidar = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pts_world = lidar[:, :3]
    reflectance = lidar[:, 3]

    # ---------- load camera images and metadata ----------
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    with open(json_path, 'r') as f:
        meta = json.load(f)
        T_wc = np.array(meta['image_extrinsic'], dtype=np.float32)  # world to camera
        K = np.array(meta['image_intrinsic'], dtype=np.float32)        # camera intrinsics

    # ---------- world -> camera ----------
    N = pts_world.shape[0]
    pts_h = np.hstack([pts_world, np.ones((N, 1), dtype=np.float32)])
    pts_cam = (T_wc @ pts_h.T).T

    Xc, Yc, Zc = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # keep points in front of camera
    front = Zc > 1e-6
    Xc, Yc, Zc = Xc[front], Yc[front], Zc[front]
    pts_world = pts_world[front]
    reflectance = reflectance[front]

    # ---------- camera -> image ----------
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * Xc / Zc + cx
    v = fy * Yc / Zc + cy

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    # keep points inside image
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[inside], v[inside]
    Zc = Zc[inside]
    pts_world = pts_world[inside]
    reflectance = reflectance[inside]

    # ---------- sample RGB ----------
    colors = image[v, u]   # RGB

    # ---------- assemble RGB-D ----------
    rgbd_points = np.hstack([
        pts_world,
        reflectance.reshape(-1, 1),
        colors.astype(np.float32)
    ])

    # save_colored_bin(rgbd_points, bin_path)

    outputs = [rgbd_points]

    if return_depth:
        outputs.append(Zc)

    if return_uv:
        outputs.append(np.stack([u, v], axis=1))

    return outputs if len(outputs) > 1 else rgbd_points


root_dir = "data/3eed"
# lidar_path = "data/3eed/drone/Outdoor_Day_penno_parking_1/002492/lidar.bin"
# img_path = "data/3eed/drone/Outdoor_Day_penno_parking_1/002492/image.jpg"
# json_path = "data/3eed/drone/Outdoor_Day_penno_parking_1/002492/meta_info.json"
search_pattern = os.path.join(root_dir, "drone", "*", "*", "lidar.bin")
lidar_path = sorted(glob.glob(search_pattern))
for lidar_path in lidar_path:
    img_path = lidar_path.replace("lidar.bin", "image.jpg")
    json_path = lidar_path.replace("lidar.bin", "meta_info.json")
    results = project_lidar_bin_to_image(
        lidar_path,
        img_path,
        json_path,
        return_depth=True,
        return_uv=True
    )

    # p, depth, uv = results
    # points = p[:, :3]              # XYZ
    # colors = p[:, 4:7] / 255.0     # RGB -> [0,1]

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    # print("Point cloud with colors visualized.")