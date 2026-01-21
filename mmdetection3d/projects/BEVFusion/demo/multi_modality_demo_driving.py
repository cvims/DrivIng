# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

import torch

import mmcv
import mmengine
from mmengine.structures import InstanceData

from mmdet3d.apis import inference_multi_modality_detector, init_model
from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import LiDARInstance3DBoxes



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data-root', default=r'data/nuscenes-driving', help='Point cloud file')
    parser.add_argument('--ann', default=r'data/nuscenes-driving/nuscenes-driving_infos_train.pkl', help='ann file')
    # parser.add_argument('config', help='Config file')
    parser.add_argument('--config', default=r'work_dirs/bev_fusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-3d.py', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--checkpoint', default=r'work_dirs/bev_fusion/epoch_10.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
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


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    pcd_dir = os.path.join(args.data_root, 'samples', 'middle_lidar')
    img_dir = os.path.join(args.data_root, 'samples', 'front_left_camera')

    pcds = sorted([os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir)])[:100]
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])[:100]

    ann_content = mmengine.load(args.ann)

    for i, (_pcd, _img) in enumerate(zip(pcds, imgs)):
        _ann_content = dict(data_list=ann_content['data_list'][i:i+1])
        # test a single image and point cloud sample
        result, data = inference_multi_modality_detector(model, _pcd, _img,
                                                        _ann_content, 'front_left_camera')
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
        points = data['inputs']['points']
        if isinstance(result.img_path, list):
            img = []
            for img_path in result.img_path:
                single_img = mmcv.imread(img_path)
                single_img = mmcv.imconvert(single_img, 'bgr', 'rgb')
                img.append(single_img)
        else:
            img = mmcv.imread(result.img_path)
            img = mmcv.imconvert(img, 'bgr', 'rgb')
        data_input = dict(points=points, img=img)

        # show the results
        visualizer.add_datasample(
            'result',
            data_input,
            data_sample=result,
            draw_gt=True,
            show=args.show,
            wait_time=-1,
            out_file=args.out_dir,
            pred_score_thr=args.score_thr,
            vis_task='multi-modality_det')


if __name__ == '__main__':
    args = parse_args()
    main(args)
