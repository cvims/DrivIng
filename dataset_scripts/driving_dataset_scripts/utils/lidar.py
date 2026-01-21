import numpy as np
import cv2


def read_extended_point_cloud_format_npz(input_path):
    """
    Loads a .npz file saved by `save_point_cloud_npz` and reconstructs the (N, 5) numpy array.
    """
    data = np.load(input_path)
    x = data['x']
    y = data['y']
    z = data['z']
    intensity = data['intensity']
    timestamp = data['timestamp']

    # Stack back into original shape (N, 5)
    point_cloud = np.stack([x, y, z, intensity, timestamp], axis=-1)
    return point_cloud


def project_point_cloud_to_image_plane(pcd, K, cTv, vTl, image_size, D=None):
    """
    Projects LiDAR points to the image plane.

    Args:
        pcd: (N, 5) numpy array of LiDAR points [x, y, z, intensity, timestamp].
        K: (3, 3) camera intrinsic matrix.
        cTv: (4, 4) camera extrinsics (vehicle -> camera).
        vTl: (4, 4) LiDAR extrinsics (LiDAR -> vehicle).
        image_size: (w, h) tuple of image size.
        Distortion coefficients.

    Returns:
        points_img: (M, 2) array of 2D points on the image plane.
        mask: Boolean array of shape (N,), True for points inside image & in front of camera.
    """
    N = pcd.shape[0]
    # Homogeneous coordinates
    lidar_points = np.ones((N, 4))
    lidar_points[:, :3] = pcd[:, :3]

    # LiDAR -> camera transformation
    cTl = cTv @ vTl
    lidar_in_camera = (cTl @ lidar_points.T).T  # (N, 4)

    # Points in front of camera
    in_front = lidar_in_camera[:, 2] > 0

    # Perspective projection
    pts_camera_frame = lidar_in_camera[:, :3]

    if D is not None:
        pts_camera_cv = pts_camera_frame.reshape(-1, 1, 3).astype(np.float32)
        rvec = np.zeros((3, 1))  # no extra rotation, already in camera frame
        tvec = np.zeros((3, 1))  # no extra translation, already in camera frame
        if D.shape[0] == 5:
            # pinhole
            pts_image_plane, _ = cv2.projectPoints(pts_camera_cv, rvec, tvec, K, D)
            pts_image_plane = pts_image_plane.reshape(-1, 2)
        elif D.shape[0] == 4:
            # fisheye
            pts_image_plane, _ = cv2.fisheye.projectPoints(
                pts_camera_cv, rvec=rvec, tvec=tvec, K=K, D=D
            )
            pts_image_plane = pts_image_plane.reshape(-1, 2)
        else:
            raise Exception('Dont know which distortion model to apply.')
    else:
        pts_image_plane = (K @ pts_camera_frame.T).T
        pts_image_plane = pts_image_plane[:, :2] / pts_camera_frame[:, 2:3]
        pts_image_plane = pts_image_plane[:, :2]

    # Inside image bounds
    w, h = image_size
    u, v = pts_image_plane[:, 0], pts_image_plane[:, 1]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # Combine masks
    mask = in_front & inside

    return pts_image_plane[mask], mask


def project_point_cloud_to_image(img: np.ndarray, pcd, K, cTv, vTl, D=None, color=None):
    _img = img.copy()
    h, w, *_ = _img.shape

    # Project points
    points_img, mask = project_point_cloud_to_image_plane(pcd, K, cTv, vTl, (w, h), D)

    colors = None
    if color is None:
        # Get intensity of valid points
        intensities = pcd[mask][:, 3]

        # Normalize intensity to 0-255
        intensities = intensities.astype(np.uint8)

        # Map to colors using COLORMAP_TURBO (or any OpenCV colormap)
        colors = cv2.applyColorMap(intensities, cv2.COLORMAP_TURBO)[:, 0, :].tolist()

    # Draw on image
    for i, pt in enumerate(points_img.astype(int)):
        if colors is not None:
            cv2.circle(_img, tuple(pt), 2, colors[i], -1)
        else:
            cv2.circle(_img, tuple(pt), 2, color, -1)

    return _img
