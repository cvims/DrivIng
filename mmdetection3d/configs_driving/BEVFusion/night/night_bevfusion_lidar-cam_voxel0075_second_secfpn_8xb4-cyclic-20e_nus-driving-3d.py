_base_ = [
    '../../../projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-3d.py'
]

data_root = 'data/nuscenes-driving/night/'

mean=[34.02669461, 40.88143143, 43.71425076]
std=[21.27352702, 23.91500977, 24.8232915]


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
