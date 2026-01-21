#!/bin/bash

DATASET_ROOT="/home/xujun/DrivIng_carla"
SPLIT_NAME="v1_train_val.json"
NUM_WORKERS=32

echo "creating train val split: $SPLIT_NAME"
python carla_conversion/split_train_val.py \
    --dataset_root $DATASET_ROOT \
    --split_name $SPLIT_NAME

echo "converting carla dataset to driving format"
python carla_conversion/convert_carla2driving.py \
    --dataset_root $DATASET_ROOT \
    --split_name $SPLIT_NAME \
    --num_workers $NUM_WORKERS

echo "creating nuscenes format dataset"
python carla_conversion/create_nuscenes_format_carla.py \
    --dataset_root $DATASET_ROOT \
    --split_name $SPLIT_NAME \
    --num_workers $NUM_WORKERS