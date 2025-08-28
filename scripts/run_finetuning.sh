#!/bin/bash

accelerate config default

export MODEL_NAME="runwayml/stable-diffusion-v1-5"

accelerate launch --mixed_precision="fp16" ~/projects/diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name="$1" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_warmup_steps=0 \
  --output_dir="$2" \
  --push_to_hub
