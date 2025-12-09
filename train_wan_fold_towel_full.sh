#!/bin/bash

# Script to perform full fine-tuning of WAN model on fold_towel dataset
# Note: Full fine-tuning requires significant GPU memory (at least 40GB VRAM)

# Step 1: Prepare the dataset metadata
echo "🔧 Preparing dataset metadata..."
python prepare_dataset.py

# Step 2: Run full fine-tuning
echo "🚀 Starting full fine-tuning for Wan2.1-T2V-1.3B..."

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /nyx-storage1/hanliu/fold_towel_new \
  --dataset_metadata_path data/fold_towel_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --dataset_repeat 50 \
  --model_paths '["/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors", "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --trainable_models "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/fold_towel_Wan2.1-T2V-1.3B_full" \
  --save_steps 200 \
  --use_gradient_checkpointing_offload

echo "✅ Training complete! Model saved to /nyx-storage1/hanliu/world_model_ckpt/Wan-AI/fold_towel_Wan2.1-T2V-1.3B_full"