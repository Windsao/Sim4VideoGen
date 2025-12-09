#!/bin/bash

# Multi-node training script for WAN model on fold_towel dataset - WORKER NODE
# This script joins the multi-node LoRA fine-tuning for the Wan2.1-T2V-1.3B model

echo "🌐 Starting multi-node LoRA training for Wan2.1-T2V-1.3B (WORKER NODE)..."

# Check if main process IP is set
if grep -q "REPLACE_WITH_MAIN_NODE_IP" multi_node_config_worker.yaml; then
    echo "❌ ERROR: Please replace 'REPLACE_WITH_MAIN_NODE_IP' in multi_node_config_worker.yaml with the actual IP address of the main node"
    exit 1
fi

# Verify shared storage access
if [ ! -d "/nyx-storage1/hanliu/fold_towel_new" ]; then
    echo "❌ ERROR: Cannot access dataset at /nyx-storage1/hanliu/fold_towel_new"
    echo "   Make sure shared storage is mounted on this worker node"
    exit 1
fi

if [ ! -f "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" ]; then
    echo "❌ ERROR: Cannot access model checkpoint at /nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-T2V-1.3B/"
    echo "   Make sure shared storage is mounted on this worker node"
    exit 1
fi

# Wait for main node to prepare metadata
echo "⏳ Waiting for main node to prepare dataset metadata..."
while [ ! -f "data/fold_towel_dataset/metadata.csv" ]; do
    sleep 5
    echo "   Still waiting for metadata file..."
done
echo "✅ Metadata file found, proceeding with training..."

# Launch worker node training
echo "🚀 Launching multi-node training on WORKER NODE..."
echo "📊 Configuration: Worker node (rank 1), connecting to main node"
echo "💾 Using local checkpoint files to avoid downloads"

accelerate launch --config_file multi_node_config_worker.yaml examples/wanvideo/model_training/train.py \
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
  --output_path "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/fold_towel_Wan2.1-T2V-1.3B_lora_multinode" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --save_steps 100 \
  --use_gradient_checkpointing_offload \
  --gradient_accumulation_steps 2

echo "✅ Worker node training complete!"