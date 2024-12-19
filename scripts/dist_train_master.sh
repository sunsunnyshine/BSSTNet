#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-30033}

# usage
if [ $# -ne 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_test.sh [number of gpu] [path to option file]"
    exit
fi

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=2 --node_rank=0 --master_addr="115.156.212.64" --master_port=$PORT \
    basicsr/train.py -opt $CONFIG --launcher pytorch
