_base_ = [
    'dusk_petr_vovnet_gridmask_p4_960x384-nus-driving.py'
]

data_root = 'data/nuscenes-driving/day/'
metainfo = dict(version='v1.0-test')


db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes-driving_dbinfos_test.pkl'
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes-driving_infos_test.pkl',
        metainfo=metainfo
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes-driving_infos_test.pkl',
        metainfo=metainfo
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes-driving_infos_test.pkl',
        metainfo=metainfo
    )
)

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_test.pkl'
)

test_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_test.pkl'
)
