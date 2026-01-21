#!/usr/bin/env bash

# NCCL debugging & safety
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1          # disable InfiniBand if not used
export NCCL_P2P_LEVEL=SYS         # use system-level peer-to-peer
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo,docker0,eth0  # restrict network interfaces
export OMP_NUM_THREADS=4          # avoid CPU oversubscription
export MKL_NUM_THREADS=4


CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
