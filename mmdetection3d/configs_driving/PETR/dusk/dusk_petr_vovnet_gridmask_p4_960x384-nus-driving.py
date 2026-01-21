_base_ = [
    '../../../projects/PETR/configs/petr_vovnet_gridmask_p4_960x384-nus-driving.py'
]

data_root = 'data/nuscenes-driving/dusk/'

mean=[56.48122024, 55.63517632, 56.9717724]
std=[28.05802552, 26.72225286, 32.99747801]

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=mean,
        std=std,
        bgr_to_rgb=False,
        pad_size_divisor=32),
)

img_norm_cfg = dict(
    mean=mean,
    std=std,
    to_rgb=False)


db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes-driving_dbinfos_train.pkl'
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
    )
)

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_val.pkl'
)

test_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes-driving_infos_test.pkl'
)
