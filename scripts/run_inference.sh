#!/bin/bash

python3 ~/projects/master-thesis/src/inference.py \
  --model_id="$1" \
  --prompt="$2" \
  --number_of_samples="$3"
