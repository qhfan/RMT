#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
/usr/local/bin/TORCHRUN $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
