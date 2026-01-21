_base_ = [
    'dusk_petr_vovnet_gridmask_p4_960x384-nus-driving.py'
]

data_root = 'data/nuscenes-driving/dusk/'

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes-driving_dbinfos_val.pkl'
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes-driving_infos_val.pkl'
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes-driving_infos_val.pkl'
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='nuscenes-driving_infos_val.pkl'
    )
)

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_val.pkl'
)

test_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_val.pkl'
)
