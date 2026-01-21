_base_ = [
    '../../../configs/_base_/datasets/nus-driving-3d.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/cyclic-20e.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
custom_imports = dict(imports=['projects.PETR.petr'])

# Driving day norm values (DONT USE THEM WITH PRETRAINED BACKBONE)
# mean=[133.39681933, 134.46777097, 133.59565155]
# std=[52.97550372, 39.42423722, 41.34866047]

# ORG normalization values
mean=[103.53, 116.28, 123.675]
std=[57.375, 57.12, 58.395]

randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=mean,
    std=std,
    to_rgb=False)
# For nuScenesDrivIng we usually do 8-class detection
class_names = [
    'car', 'truck', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian'
]
metainfo = dict(classes=class_names)

input_modality = dict(use_camera=True)
model = dict(
    type='PETR',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=mean,
        std=std,
        bgr_to_rgb=False,
        pad_size_divisor=32),
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNetCP',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=(
            'stage4',
            'stage5',
        )),
    img_neck=dict(
        type='CPFPN', in_channels=[768, 1024], out_channels=256, num_outs=2),
    pts_bbox_head=dict(
        type='PETRHead',
        num_classes=len(class_names),
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names)),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(
                    type='IoUCost', weight=0.0
                ),  # Fake cost. Just to be compatible with DETR head.
                pc_range=point_cloud_range))))

dataset_type = 'NuScenesDrivIngDataset'
data_root = 'data/nuscenes-driving/day/'
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes-driving_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args),
    backend_args=backend_args)

ida_aug_conf = {
    'resize_lim': (0.5, 0.65),
    'final_dim': (384, 960),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1080,
    'W': 1920,
    'rand_flip': True,
}

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='PhotoMetricDistortionMultiView',
        brightness_delta=32,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=18
    ),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=False,
        training=True),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
            'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths'
        ])
]
test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        backend_args=backend_args),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf,
        training=False),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            pts='samples/middle_lidar',
            front_left_camera='samples/front_left_camera',
            front_right_camera='samples/front_right_camera',
            back_left_camera='samples/back_left_camera',
            back_right_camera='samples/back_right_camera',
            left_camera='samples/left_camera',
            right_camera='samples/right_camera'),
        pipeline=train_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=False,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))
test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            pts='samples/middle_lidar',
            front_left_camera='samples/front_left_camera',
            front_right_camera='samples/front_right_camera',
            back_left_camera='samples/back_left_camera',
            back_right_camera='samples/back_right_camera',
            left_camera='samples/left_camera',
            right_camera='samples/right_camera'),
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_prefix=dict(
            pts='samples/middle_lidar',
            front_left_camera='samples/front_left_camera',
            front_right_camera='samples/front_right_camera',
            back_left_camera='samples/back_left_camera',
            back_right_camera='samples/back_right_camera',
            left_camera='samples/left_camera',
            right_camera='samples/right_camera'),
        pipeline=test_pipeline,
        box_type_3d='LiDAR',
        metainfo=metainfo,
        test_mode=True,
        modality=input_modality,
        use_valid_flag=True,
        backend_args=backend_args))


num_epochs_training = 100
val_epoch_interval = 3

lr = 1e-5
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)  # lower LR for backbone
        }
    ),
    clip_grad=dict(max_norm=5.0, norm_type=2)  # tighter clipping for stability
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        T_max=num_epochs_training,
        by_epoch=True,
    )
]

train_cfg = dict(by_epoch=True, max_epochs=num_epochs_training, val_interval=val_epoch_interval)

find_unused_parameters = False

# pretrain_path can be found here:
# https://drive.google.com/file/d/1ABI5BoQCkCkP4B0pO5KBJ3Ni0tei0gZi/view
load_from = 'work_dirs/pretrained/imgbackbone_fcos3d_vovnet/fcos3d_vovnet_imgbackbone-remapped.pth'
resume = False

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=3,
        by_epoch=True,
        max_keep_ckpts=3,
        rule='greater',
        save_best=
        'NuScenesDrivIng metric/pred_instances_3d_NuScenes_DrivIng/NDS',
    )
)
