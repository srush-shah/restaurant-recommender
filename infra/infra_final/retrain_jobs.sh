#!/bin/bash

# Submit ALS training job
ray job submit \
  --runtime-env /home/jovyan/work/scripts/runtime.json \
  --entrypoint-num-cpus 2 \
  --verbose \
  --working-dir ./scripts \
  -- python als_train.py

# Submit MLP training job
ray job submit \
  --runtime-env /home/jovyan/work/scripts/runtime.json \
  --entrypoint-num-cpus 2 \
  --verbose \
  --working-dir ./scripts \
  -- python mlp_train.py
