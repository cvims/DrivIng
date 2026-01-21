# Copyright (c) OpenMMLab. All rights reserved.
import os

from argparse import ArgumentParser

import torch

import mmcv
import mmengine
from mmengine.structures import InstanceData

from mmdet3d.apis import inference_multi_modality_detector, inference_detector, init_model
from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import LiDARInstance3DBoxes
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data-root', default=r'data/nuscenes-driving', help='Point cloud file')
    # parser.add_argument('--ann', default=r'data/driving/di_day/nuscenes-driving_infos_val.pkl', help='ann file')
    parser.add_argument('--ann', default=r'data/nuscenes-driving/nuscenes-driving_infos_val.pkl', help='ann file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('--config', default=r'configs_driving/BEVFusion/eval_bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-carla.py', help='Config file')
    parser.add_argument('--config', default=r'configs_driving/CenterPoint/eval_centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-carla.py', help='Config file')

    # parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--checkpoint', default=r'work_dirs/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-carla/epoch_10.pth', help='Checkpoint file')

    # parser.add_argument('--checkpoint', default=r'work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-carla/epoch_5.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default=r'tmp/demo/', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    return args

    # 3. Plot Ground Truth Bounding Boxes
def convert_mmdet_to_bev_bboxes(bboxes_3d, scores_3d, score_thr, pc_range):
    """
    Converts mmdet3d 'LiDARInstance3DBoxes' to BEV representation for plotting.

    Args:
        bboxes_3d (BaseInstance3DBoxes): 3D bounding boxes (e.g., LiDARInstance3DBoxes).
        scores_3d (torch.Tensor): (N,) scores for each box.
        score_thr (float): Threshold to filter boxes.
        pc_range (list or np.array): The point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
                                     Used to filter boxes outside the BEV plot area.

    Returns:
        tuple:
            - bev_bboxes (list): A list of (5, 2) numpy arrays. Each array represents
                                 the 4 corners + 1 closing point of a box for plotting.
            - headings (list): A list of (2, 2) numpy arrays. Each array represents
                               the [start_point, end_point] of the box's heading line.
    """
    # Filter boxes based on score
    mask = scores_3d > score_thr
    bboxes_3d = bboxes_3d[mask]
    
    # Get tensor data (N, 7) -> (x, y, z, w, l, h, yaw)
    # Note: mmdet3d uses (w, l, h) for dims, which maps to (x_size, y_size, z_size)
    box_tensor = bboxes_3d.tensor.cpu().numpy()
    bev_bboxes = []
    
    for box in box_tensor:
        center_x = box[0]
        center_y = box[1]
        
        # Filter boxes outside the BEV x-y plane
        if center_x < pc_range[0] or center_x > pc_range[3] \
            or center_y < pc_range[1] or center_y > pc_range[4]:
                continue
        
        x_size = box[3] # This is 'w'
        y_size = box[4] # This is 'l'
        yaw = box[6]

        # 4. Define 4 corners in box's local coordinate system (at origin)
        half_x = x_size / 2.0
        half_y = y_size / 2.0
        local_corners = np.array([
            [half_x,  half_y],  # Top-right
            [half_x, -half_y],  # Bottom-right
            [-half_x, -half_y], # Bottom-left
            [-half_x,  half_y]  # Top-left
        ])

        # 5. Create 2D rotation matrix
        c = np.cos(yaw)
        s = np.sin(yaw)
        rotation_matrix = np.array([
            [c, -s],
            [s,  c]
        ])

        # 6. Rotate corners
        rotated_corners = np.dot(local_corners, rotation_matrix.T)

        # 7. Translate corners to final position
        final_corners = rotated_corners + np.array([center_x, center_y])
        
        # 8. Close the loop for plotting (add first corner to the end)
        corners_to_plot = np.vstack([final_corners, final_corners[0]])
            
        bev_bboxes.append(corners_to_plot)
        
        # Calculate the end point of the heading line
        end_x = center_x + half_x * c * 2
        end_y = center_y + half_x * s * 2
        
        heading_line = np.array([
            [center_x, center_y], # Start point (center)
            [end_x, end_y]        # End point (front midpoint)
        ])
        bev_bboxes.append(heading_line)
        # --- End of Added Section ---
        
    return bev_bboxes

def visualize_bev_and_save(points, pred_bboxes, gt_bboxes, output_path, subsample_points=50000):
    """
    Uses Matplotlib to render a 2D Bird's Eye View (BEV) scene and save it to a file.
    Points are colored by their Z-value (height).
    
    Args:
        pcd (o3d.geometry.PointCloud): The main point cloud.
        pred_bboxes (list[o3d.geometry.OrientedBoundingBox]): List of predicted boxes.
        gt_bboxes (list[o3d.geometry.OrientedBoundingBox]): List of ground truth boxes.
        output_path (str): Where to save the resulting image (e.g., "scene_bev.png").
        subsample_points (int): Max number of points to draw for performance.
    """
    print(f"Initializing Matplotlib BEV renderer...")
    
    # Create a figure and a 2D axis
    fig, ax = plt.subplots(figsize=(12, 12)) # BEV maps are often square
    
    if subsample_points and points.shape[0] > subsample_points:
        print(f"Subsampling {points.shape[0]} points to {subsample_points}...")
        indices = np.random.choice(points.shape[0], subsample_points, replace=False)
        points = points[indices]
    
    # Get x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2] # We'll use this for color
    
    # Plot as a 2D scatter plot, colored by height (z)
    # Using a colormap (like 'viridis' or 'jet') is common
    scatter = ax.scatter(x, y, s=1, c=z, cmap='viridis', alpha=0.7)

    # 2. Plot Predicted Bounding Boxes
    print(f"Adding {len(pred_bboxes)} predicted boxes (blue)...")
    for box in pred_bboxes:
        ax.plot(box[:, 0], box[:, 1], color='blue')
        
    # 3. Plot Ground Truth Bounding Boxes
    print(f"Adding {len(gt_bboxes)} ground truth boxes (red)...")
    for box in gt_bboxes:
        ax.plot(box[:, 0], box[:, 1], color='red')

    # 4. Set up the view and labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_title("Bird's Eye View (BEV) Visualization")
    
    # CRITICAL: Set aspect ratio to 'equal'
    # This ensures that a 1-meter-by-1-meter box looks square, not rectangular.
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 5. Save the image
    print(f"Saving image to {output_path}...")
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    
    # Close the figure to free up memory
    plt.close(fig)

def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # init visualizer
    pc_range = model.cfg.point_cloud_range
    print(f"point cloud range: ", pc_range)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    pcd_dir = os.path.join(args.data_root, 'samples', 'middle_lidar')
    # img_dir = os.path.join(args.data_root, 'samples', 'front_left_camera')
    
    pcds = sorted([os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir)])[:100]
    # imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])[:100]
    ann_content = mmengine.load(args.ann)

    # for i, (_pcd, _img) in enumerate(zip(pcds, imgs)):
    for i, (curr_frame_content) in enumerate(ann_content['data_list']):
        if i % 20 != 0:
            continue
        
        _ann_content = dict(data_list=ann_content['data_list'][i:i+1])
        _pcd = _ann_content['lidar_points']['lidar']['lidar_path']
        # test a single image and point cloud sample
        # result, data = inference_multi_modality_detector(model, _pcd, _img,
        #                                                 _ann_content, 'front_left_camera')
        result, data = inference_detector(model, _pcd)
        # add instances to result
        # Convert to tensors
        bboxes_3d = []
        labels_3d = []
        velocities = []

        for obj in _ann_content['data_list'][0]['instances']:
            bboxes_3d.append(obj['bbox_3d'])      # each should be a list/array of 7 params (x,y,z,w,l,h,yaw)
            labels_3d.append(obj['bbox_label'])   # class index
            velocities.append(obj['velocity'])  # [vx, vy]

        # Stack into tensors
        bboxes_3d = torch.tensor(bboxes_3d, dtype=torch.float32) if bboxes_3d else torch.zeros((0, 7))
        labels_3d = torch.tensor(labels_3d, dtype=torch.long) if labels_3d else torch.zeros((0,), dtype=torch.long)
        velocities = torch.tensor(velocities, dtype=torch.float32) if velocities else torch.zeros((0, 2))

        # Wrap into LiDARInstance3DBoxes
        bboxes_3d = LiDARInstance3DBoxes(bboxes_3d, origin=(0.5,0.5,0.5))

        # Create InstanceData
        gt_instances_3d = InstanceData(
            bboxes_3d=bboxes_3d,
            labels_3d=labels_3d,
            velocities=velocities
        )
        result.gt_instances_3d = gt_instances_3d
        
        # --- Start of headless visualization ---
        
        # 1. Load the point cloud
        print("Loading point cloud from inference result...")
        points_tensor = data['inputs']['points'] # Get batch index 0
        points = points_tensor.cpu().numpy()
        pcd = points[:, :3]

        # 2. Extract and convert PREDICTION boxes
        print("Extracting and filtering predictions...")
        pred_instances = result.pred_instances_3d
        pred_bboxes_3d = pred_instances.bboxes_3d
        pred_scores_3d = pred_instances.scores_3d
        
        o3d_pred_bboxes = convert_mmdet_to_bev_bboxes(
            pred_bboxes_3d, pred_scores_3d, 0.3, pc_range
        )
        print(f"Found {len(o3d_pred_bboxes)} prediction boxes with score > {0.3}")

        # 3. Extract and convert GROUND TRUTH boxes
        print("Extracting ground truth boxes...")
        gt_instances = result.gt_instances_3d
        gt_bboxes_3d = gt_instances.bboxes_3d
        # Create dummy scores (all 1.0) to pass the filter (score_thr=0.0)
        gt_scores_dummy = torch.ones_like(gt_bboxes_3d.tensor[:, 0]).to(gt_bboxes_3d.tensor.device)
        
        o3d_gt_bboxes = convert_mmdet_to_bev_bboxes(
            gt_bboxes_3d, gt_scores_dummy, 0.0, pc_range
        )
        print(f"Found {len(o3d_gt_bboxes)} ground truth boxes.")

        # 4. Visualize and save the result
        out_filename = f"sample_{i:04d}_visualization.png"
        out_filepath = os.path.join(args.out_dir, out_filename)
        
        visualize_bev_and_save(pcd, o3d_pred_bboxes, o3d_gt_bboxes, out_filepath)
        # --- End of headless visualization ---
        if (i // 200) > 1:
            break


if __name__ == '__main__':
    args = parse_args()
    main(args)
