#!/usr/bin/env python3
"""
Inference script for the trained WAN model on fold towel dataset.
This script loads the fine-tuned model and generates videos.
"""

import torch
from diffsynth import save_video, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

def inference_lora_model():
    """Load LoRA fine-tuned model and generate videos."""

    # Load base model
    print("Loading base Wan2.1-T2V-1.3B model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", local_model_path='/nyx-storage1/hanliu/world_model_ckpt'),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", local_model_path='/nyx-storage1/hanliu/world_model_ckpt'),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", local_model_path='/nyx-storage1/hanliu/world_model_ckpt'),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*", local_model_path='/nyx-storage1/hanliu/world_model_ckpt'),
    )
    pipe.enable_vram_management()

    # Load LoRA weights
    print("Loading LoRA weights...")
    lora_path = "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/fold_towel_Wan2.1-T2V-1.3B_lora/step-10000.safetensors"
    lora_state_dict = load_state_dict(lora_path)

    # Use GeneralLoRALoader to load LoRA weights
    from diffsynth.lora import GeneralLoRALoader
    lora_loader = GeneralLoRALoader(device="cuda", torch_dtype=torch.bfloat16)
    lora_loader.load(pipe.dit, lora_state_dict, alpha=1.0)

    # Generate videos with different prompts
    prompts = [
        "A person carefully folding a white towel on a clean surface, step by step demonstration",
        "Hands demonstrating the proper technique to fold a bath towel neatly",
        "A video showing the process of folding a towel, close-up view of hands",
        "Tutorial showing how to fold a colorful beach towel into a compact square",
        "Professional hotel-style towel folding demonstration with precise movements",
    ]

    negative_prompt = "blurry, low quality, distorted, unclear movements, static image, cartoon, animation"

    for i, prompt in enumerate(prompts, 1):
        print(f"\nGenerating video {i}/{len(prompts)}...")
        print(f"Prompt: {prompt}")

        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=81,
            cfg_scale=7.0,
            num_inference_steps=50,
            seed=42 + i,  # Different seed for each video
            tiled=True,
        )

        output_path = f"output_fold_towel_{i}.mp4"
        save_video(video, output_path, fps=15, quality=5)
        print(f"Saved video to: {output_path}")

    print("\n✅ All videos generated successfully!")

def inference_full_model():
    """Load fully fine-tuned model and generate videos."""

    # Load fine-tuned model
    print("Loading fully fine-tuned Wan2.1-T2V-1.3B model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            # Load the fine-tuned DiT model
            ModelConfig(path="./models/train/fold_towel_Wan2.1-T2V-1.3B_full/model_latest.safetensors", offload_device="cpu"),
            # Load the original text encoder and VAE
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()

    # Generate videos
    prompts = [
        "A person carefully folding a white towel on a clean surface, step by step demonstration",
        "Hands demonstrating the proper technique to fold a bath towel neatly",
        "A video showing the process of folding a towel, close-up view of hands",
    ]

    negative_prompt = "blurry, low quality, distorted, unclear movements, static image"

    for i, prompt in enumerate(prompts, 1):
        print(f"\nGenerating video {i}/{len(prompts)}...")
        print(f"Prompt: {prompt}")

        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=81,
            cfg_scale=7.0,
            num_inference_steps=50,
            seed=42 + i,
            tiled=True,
        )

        output_path = f"output_fold_towel_full_{i}.mp4"
        save_video(video, output_path, fps=15, quality=5)
        print(f"Saved video to: {output_path}")

    print("\n✅ All videos generated successfully!")

if __name__ == "__main__":
    import os

    # Check which model exists and run appropriate inference
    lora_model_path = "/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/fold_towel_Wan2.1-T2V-1.3B_lora/step-10000.safetensors"
    full_model_path = "./models/train/fold_towel_Wan2.1-T2V-1.3B_full/model_latest.safetensors"

    if os.path.exists(lora_model_path):
        print("Found LoRA model, running LoRA inference...")
        inference_lora_model()
    elif os.path.exists(full_model_path):
        print("Found fully fine-tuned model, running full model inference...")
        inference_full_model()
    else:
        print("❌ No trained model found. Please run training first using:")
        print("   bash train_wan_fold_towel.sh  (for LoRA training)")
        print("   bash train_wan_fold_towel_full.sh  (for full fine-tuning)")