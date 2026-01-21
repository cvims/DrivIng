import os
import cv2
import numpy as np
from typing import List
import math
from driving_dataset_scripts.utils.codes import CLASS_COLOR_CODES


def load_image(img_path):
    if img_path is not None and os.path.exists(img_path):
        return cv2.imread(img_path)

    return None


def annotation_to_image(img: np.ndarray, annotations: dict, vTl: np.ndarray, cTv: np.ndarray, K: np.ndarray, D: np.ndarray=None):
    if annotations is None:
        return img

    box_img = img.copy()
    img_h, img_w = img.shape[:2]

    n_boxes = len(annotations)
    if n_boxes == 0:
        return box_img

    # Box edges
    edges = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ])

    # Local corners template (8x4)
    corners = np.array([
        [-0.5, -0.5, -0.5, 1],
        [-0.5,  0.5, -0.5, 1],
        [ 0.5,  0.5, -0.5, 1],
        [ 0.5, -0.5, -0.5, 1],
        [-0.5, -0.5,  0.5, 1],
        [-0.5,  0.5,  0.5, 1],
        [ 0.5,  0.5,  0.5, 1],
        [ 0.5, -0.5,  0.5, 1]
    ], dtype=np.float64)

    # Stack all boxes
    all_corners = []
    texts = []
    for ann in annotations:
        pos = np.array(ann['position'], dtype=np.float64)
        dims = np.array(ann['dimension'], dtype=np.float64)  # l,w,h
        yaw = ann['orientation']
        obj_type = ann['object_type']

        # scale template by box dims
        c_scaled = corners.copy()
        c_scaled[:,0] *= dims[0]
        c_scaled[:,1] *= dims[1]
        c_scaled[:,2] *= dims[2]

        # 4x4 rotation matrix around Z
        R = np.eye(4)
        R[:2,:2] = [[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]]

        # translate
        T = np.eye(4)
        T[:3,3] = pos

        # combined local->lidar transform
        M = T @ R

        # apply to corners
        c_transformed = (M @ c_scaled.T).T  # 8x4
        all_corners.append(c_transformed)
        texts.append(obj_type)

    all_corners = np.vstack(all_corners)  # (n_boxes*8,4)

    # Transform all boxes to camera frame
    cam_corners = (cTv @ vTl @ all_corners.T).T  # (n_boxes*8,4)

    # Keep boxes where all Z > 0
    cam_corners_reshaped = cam_corners.reshape(n_boxes, 8, 4)
    z_positive = np.all(cam_corners_reshaped[:,:,2] > 0, axis=1)

    if not np.any(z_positive):
        return box_img

    cam_corners_reshaped = cam_corners_reshaped[z_positive]
    texts = [t for i,t in enumerate(texts) if z_positive[i]]
    n_valid = len(texts)

    pts_3d = cam_corners_reshaped[:,:,:3].reshape(-1,3).astype(np.float32)

    if D is not None:
        if D.shape[0] == 5:  # standard distortion model
            pts_2d, _ = cv2.projectPoints(
                pts_3d.reshape(-1,1,3), rvec=np.zeros(3), tvec=np.zeros(3), cameraMatrix=K, distCoeffs=D
            )
            pts_2d = pts_2d.reshape(-1,2)
        elif D.shape[0] == 4:  # fisheye distortion
            pts_2d, _ = cv2.fisheye.projectPoints(
                pts_3d.reshape(-1,1,3), rvec=np.zeros(3), tvec=np.zeros(3), K=K, D=D
            )
            pts_2d = pts_2d.reshape(-1,2)
    else:  # pinhole projection
        proj = (K @ pts_3d.T).T
        pts_2d = proj[:,:2] / proj[:,2:3]

    img_pts = np.round(pts_2d).astype(np.int32).reshape(n_valid,8,2)

    # Clip boxes fully outside image
    for i in range(n_valid):
        box = img_pts[i]
        if np.all((box[:,0] < 0) | (box[:,0] >= img_w) |
                  (box[:,1] < 0) | (box[:,1] >= img_h)):
            continue

        # Draw edges
        for e in edges:
            cv2.line(box_img, tuple(box[e[0]]), tuple(box[e[1]]), CLASS_COLOR_CODES[texts[i]], 2)

        # Draw text above top face
        top_center = np.mean(box[4:8], axis=0)
        text_pos = (int(top_center[0]), max(int(top_center[1])-5,0))
        cv2.putText(box_img, texts[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, CLASS_COLOR_CODES[texts[i]], 2, cv2.LINE_AA)

    return box_img


def plot_images_combined(images: List[np.ndarray], rows: int, out_path: str, img_size=(1080, 1920)):
    if isinstance(images, np.ndarray):
        assert len(images.shape) == 4, f'Invalid shape of images input. Got {images.shape} but (N, H, W, C) is required.'
    else:
        images = np.array(images)

    len_imgs = images.shape[0]
    cols = math.ceil(len_imgs / rows)

    rel_img_h, rel_img_w = img_size[0] // rows, img_size[1] // cols

    full_img = np.zeros(shape=(*img_size, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        row_index = i // cols
        col_index = i % cols

        full_img[
            row_index*rel_img_h:row_index*rel_img_h+rel_img_h,
            col_index*rel_img_w:col_index*rel_img_w+rel_img_w
            :] = cv2.resize(img, (rel_img_w, rel_img_h))
    
    if out_path:
        cv2.imwrite(out_path, full_img)
    
    return full_img
