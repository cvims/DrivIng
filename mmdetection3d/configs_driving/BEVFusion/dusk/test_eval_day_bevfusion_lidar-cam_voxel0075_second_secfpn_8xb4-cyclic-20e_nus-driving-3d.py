_base_ = [
    'dusk_bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-3d.py'
]

data_root = 'data/nuscenes-driving/day/'
metainfo = dict(version='v1.0-test')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes-driving_dbinfos_test.pkl'
)


val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes-driving_infos_test.pkl',
        metainfo=metainfo)
    )

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes-driving_infos_test.pkl',
        metainfo=metainfo)
    )

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_test.pkl')

test_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_test.pkl')
