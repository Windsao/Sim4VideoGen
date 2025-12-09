#!/bin/bash

# Multi-node training script for WAN model on fold_towel dataset - MAIN NODE
# This script uses LoRA fine-tuning for the Wan2.1-T2V-1.3B model across multiple nodes

echo "🌐 Starting multi-node LoRA training for Wan2.1-T2V-1.3B (MAIN NODE)..."

# Check if main process IP is set
if grep -q "REPLACE_WITH_MAIN_NODE_IP" multi_node_config_main.yaml; then
    echo "❌ ERROR: Please replace 'REPLACE_WITH_MAIN_NODE_IP' in multi_node_config_main.yaml with the actual IP address of this main node"
    exit 1
fi

# Step 1: Prepare the dataset metadata (only on main node)
echo "🔧 Preparing dataset metadata..."
python prepare_dataset.py

# Step 2: Launch multi-node training
echo "🚀 Launching multi-node training on MAIN NODE..."
echo "📊 Configuration: 2 machines, 16 processes total (8 per machine)"
echo "💾 Using local checkpoint files to avoid downloads"

accelerate launch --config_file multi_node_config_main.yaml examples/wanvideo/model_training/train.py \
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

echo "✅ Multi-node training complete! Model saved to /nyx-storage1/hanliu/world_model_ckpt/Wan-AI/fold_towel_Wan2.1-T2V-1.3B_lora_multinode"