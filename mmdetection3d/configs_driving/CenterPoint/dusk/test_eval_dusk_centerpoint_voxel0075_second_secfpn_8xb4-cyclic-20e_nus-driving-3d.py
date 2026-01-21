_base_ = ['../../../configs/centerpoint/centerpoint_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-driving-3d.py']

class_names = [
    'car', 'truck', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian'
]

data_root = 'data/nuscenes-driving/dusk/'
metainfo = dict(version='v1.0-test')

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo))
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
