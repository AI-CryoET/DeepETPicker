#!/bin/bash

bsub -P scicompsoft -n 12 -gpu "num=1" -q gpu_a100 -W 1:00 \
  -o /groups/scicompsoft/home/raridenm/projects/cryoet/au_np/DeepETPicker/logs/train_aunp.%J.log \
  -e /groups/scicompsoft/home/raridenm/projects/cryoet/au_np/DeepETPicker/logs/train_aunp.%J.err \
  ./.venv/bin/python train_AuNPs.py
