_base_ = [
    '../../../projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-3d.py'
]

data_root = 'data/nuscenes-driving/day/'

mean=[133.39681933, 134.46777097, 133.59565155]
std=[52.97550372, 39.42423722, 41.34866047]


model = dict(
    data_preprocessor=dict(
        mean=mean,
        std=std,
        bgr_to_rgb=False
    )
)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes-driving_dbinfos_train.pkl'
)

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            data_root=data_root)))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root))
test_dataloader = val_dataloader

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_val.pkl')

test_evaluator = val_evaluator
test_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_test.pkl')
