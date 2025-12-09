#!/bin/bash

# Script to train WAN model on fold_towel dataset
# This script uses LoRA fine-tuning for the Wan2.1-T2V-1.3B model

# Step 1: Prepare the dataset metadata
echo "🔧 Preparing dataset metadata..."
python prepare_dataset.py

# Step 2: Run LoRA training
echo "🚀 Starting LoRA training for Wan2.1-T2V-1.3B..."

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /nyx-storage1/hanliu/fold_towel_new \
  --dataset_metadata_path data/fold_towel_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 100 \
  --model_paths '["/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors", "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/fold_towel_Wan2.1-T2V-1.3B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --save_steps 100 \
  --use_gradient_checkpointing_offload

echo "✅ Training complete! Model saved to /nyx-storage1/hanliu/world_model_ckpt/Wan-AI/fold_towel_Wan2.1-T2V-1.3B_lora"