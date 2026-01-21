_base_ = [
    '../_base_/models/pointpillars_hv_nofpn_nus-driving.py',
    '../_base_/datasets/nus-driving-3d.py', '../_base_/schedules/schedule-2x_nuscenes-driving.py',
    '../_base_/default_runtime.py'
]

# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
# Check '../_base_/schedules/schedule-2x.py' (if set in _base_ varialbe) for available parameters to change.
train_cfg = dict(val_interval=250)


# test
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    # logger=dict(type='LoggerHook', interval=50),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)
