#!/bin/bash

trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT

python -m visdom.server -logging_level WARNING &
pids="$pids $!"
sleep 2

python $1 &
pids="$pids $!"

wait -n $pids
