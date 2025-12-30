"""
Stage 2 fine-tuning for WAN2.2 TI2V with motion/depth heads.

Supports:
- Full DiT fine-tuning (no LoRA).
- Selective layer fine-tuning using name patterns.
"""

import argparse
import os
import re
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from train_wan22_ti2v_motion_depth import (
    WAN22MotionDepthTrainingModule,
    collate_keep_pil,
    create_motion_data_operator,
    normalize_model_paths_arg,
    summarize_model_paths,
)
from diffsynth.trainers.unified_dataset import UnifiedDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _split_comma_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern '{pattern}': {exc}") from exc
    return compiled


def _apply_trainable_patterns(
    model: WAN22MotionDepthTrainingModule,
    patterns: List[str],
    scope: str,
) -> Tuple[int, int]:
    scope_prefix = "pipe.dit." if scope == "dit" else "pipe."
    compiled = _compile_patterns(patterns)
    total = 0
    matched = 0
    for name, param in model.named_parameters():
        if not name.startswith(scope_prefix):
            continue
        total += 1
        if any(pattern.search(name) for pattern in compiled):
            param.requires_grad = True
            matched += 1
        else:
            param.requires_grad = False
    return total, matched


def _log_trainable_params(model: WAN22MotionDepthTrainingModule) -> None:
    if os.environ.get("RANK", "0") != "0":
        return
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"[INFO] Trainable params: {len(trainable)} tensors")
    if trainable:
        print(f"[INFO] Example trainable params: {trainable[:5]}")


def _save_module_state(
    module: torch.nn.Module,
    path: str,
    trainable_only: bool,
) -> None:
    if not trainable_only:
        torch.save(module.state_dict(), path)
        return
    state_dict = {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if param.requires_grad
    }
    torch.save(state_dict, path)


def main() -> None:
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--dataset_metadata_path", type=str, required=True)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)

    # Model parameters (provide either model_paths or model_id_with_origin_paths)
    parser.add_argument("--model_paths", type=str, default=None,
                        help="JSON array of model paths, e.g., '[\"path1\", \"path2\"]'")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Comma-separated model_id:pattern pairs")
    parser.add_argument("--extra_inputs", type=str, default="input_image",
                        help="Additional model inputs, comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Local path to UMT5 tokenizer files (avoids online download).")
    parser.add_argument("--output_path", type=str, required=True)

    # Fine-tuning configuration
    parser.add_argument("--finetune_mode", type=str, default="full",
                        choices=["full", "custom"])
    parser.add_argument("--trainable_models", type=str, default="dit",
                        help="Comma list for pipe.freeze_except (e.g., 'dit,text_encoder').")
    parser.add_argument("--trainable_param_patterns", type=str, default=None,
                        help="Comma-separated regex patterns for trainable params.")
    parser.add_argument("--trainable_scope", type=str, default="dit",
                        choices=["dit", "pipe"])

    # LoRA parameters (unused for full finetune but kept for compatibility)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--lora_checkpoint", type=str, default=None)

    # Checkpoint loading
    parser.add_argument("--motion_head_checkpoint", type=str, default=None)
    parser.add_argument("--depth_head_checkpoint", type=str, default=None)

    # Loss weights
    parser.add_argument("--motion_channels", type=int, default=4)
    parser.add_argument("--motion_loss_weight", type=float, default=0.1)
    parser.add_argument("--depth_loss_weight", type=float, default=0.1)
    parser.add_argument("--motion_loss_type", type=str, default="mse",
                        choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--depth_loss_type", type=str, default="mse",
                        choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--use_warp_loss", action="store_true")
    parser.add_argument("--warp_loss_weight", type=float, default=0.1)
    parser.add_argument("--warp_loss_type", type=str, default="mse",
                        choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--use_rgb_warp_loss", action="store_true")
    parser.add_argument("--rgb_warp_loss_weight", type=float, default=0.1)
    parser.add_argument("--rgb_warp_loss_type", type=str, default="l1",
                        choices=["mse", "l1", "smooth_l1"])
    parser.add_argument("--rgb_warp_use_ssim", action="store_true")
    parser.add_argument("--rgb_warp_ssim_weight", type=float, default=0.85)

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)

    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="wan22-ti2v-stage2-finetune")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Optional run name for Weights & Biases.")

    # Spatio-temporal depth
    parser.add_argument("--use_spatiotemporal_depth", action="store_true")
    parser.add_argument("--spatiotemporal_depth_type", type=str, default="simple",
                        choices=["simple", "full"])

    # Saving
    parser.add_argument("--save_modules", type=str, default="dit",
                        help="Comma list of pipe submodules to save (e.g., 'dit,text_encoder').")
    parser.add_argument("--save_trainable_only", action="store_true",
                        help="Save only requires_grad=True params for selected modules.")

    args = parser.parse_args()

    args.model_paths = normalize_model_paths_arg(args.model_paths)
    if os.environ.get("RANK", "0") == "0":
        print(f"[INFO] Resolved --model_paths: {summarize_model_paths(args.model_paths)}")

    if args.finetune_mode == "custom" and not args.trainable_param_patterns:
        raise ValueError("--trainable_param_patterns is required when --finetune_mode=custom")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )

    if args.use_wandb and accelerator.is_main_process:
        if not WANDB_AVAILABLE:
            raise RuntimeError("wandb is not installed, but --use_wandb was provided.")
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=["video"],
        main_data_operator=create_motion_data_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.height * args.width,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            motion_channels=args.motion_channels,
            normalize_motion=False,
            motion_scale=1.0,
            depth_scale=1.0,
            normalize_depth=False,
            load_depth=True,
        ),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_keep_pil,
    )

    model = WAN22MotionDepthTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        training_mode="lora",
        trainable_models=args.trainable_models,
        lora_base_model=None,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        extra_inputs=args.extra_inputs,
        tokenizer_path=args.tokenizer_path,
        motion_channels=args.motion_channels,
        motion_loss_weight=args.motion_loss_weight,
        motion_loss_type=args.motion_loss_type,
        depth_loss_weight=args.depth_loss_weight,
        depth_loss_type=args.depth_loss_type,
        use_warp_loss=args.use_warp_loss,
        warp_loss_weight=args.warp_loss_weight,
        warp_loss_type=args.warp_loss_type,
        use_rgb_warp_loss=args.use_rgb_warp_loss,
        rgb_warp_loss_weight=args.rgb_warp_loss_weight,
        rgb_warp_loss_type=args.rgb_warp_loss_type,
        rgb_warp_use_ssim=args.rgb_warp_use_ssim,
        rgb_warp_ssim_weight=args.rgb_warp_ssim_weight,
        motion_head_checkpoint=args.motion_head_checkpoint,
        depth_head_checkpoint=args.depth_head_checkpoint,
        use_spatiotemporal_depth=args.use_spatiotemporal_depth,
        spatiotemporal_depth_type=args.spatiotemporal_depth_type,
    )

    if args.finetune_mode == "custom":
        patterns = _split_comma_list(args.trainable_param_patterns)
        total, matched = _apply_trainable_patterns(model, patterns, args.trainable_scope)
        if os.environ.get("RANK", "0") == "0":
            print(f"[INFO] Custom finetune scope: {args.trainable_scope}")
            print(f"[INFO] Matched {matched}/{total} parameters in scope")
        if matched == 0:
            raise ValueError("No parameters matched --trainable_param_patterns")
        if args.trainable_scope == "dit":
            model.pipe.dit.train()

    _log_trainable_params(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if os.environ.get("RANK", "0") == "0":
        print(f"[INFO] Optimizer params: {len(trainable_params)} tensors (requires_grad=True)")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    global_step = 0
    os.makedirs(args.output_path, exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress_bar:
            with accelerator.accumulate(model):
                if isinstance(batch, list):
                    losses = []
                    loss_dict_sum = {}
                    for sample in batch:
                        loss_i, loss_dict_i = model(sample)
                        losses.append(loss_i)
                        for key, value in loss_dict_i.items():
                            loss_dict_sum[key] = loss_dict_sum.get(key, 0.0) + float(value)
                    loss = torch.stack(losses).mean()
                    loss_dict = {k: v / len(batch) for k, v in loss_dict_sum.items()}
                else:
                    loss, loss_dict = model(batch)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                progress_bar.set_postfix({
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "noise": f"{loss_dict['noise_loss']:.4f}",
                    "motion": f"{loss_dict.get('motion_loss', 0):.4f}",
                    "depth": f"{loss_dict.get('depth_loss', 0):.4f}",
                    "warp": f"{loss_dict.get('warp_loss', 0):.4f}",
                    "rgb_warp": f"{loss_dict.get('rgb_warp_loss', 0):.4f}",
                })

                if args.use_wandb:
                    wandb.log({
                        "train/total_loss": loss_dict["total_loss"],
                        "train/noise_loss": loss_dict["noise_loss"],
                        "train/motion_loss": loss_dict.get("motion_loss", 0),
                        "train/depth_loss": loss_dict.get("depth_loss", 0),
                        "train/warp_loss": loss_dict.get("warp_loss", 0),
                        "train/warp_loss_weighted": loss_dict.get("warp_loss_weighted", 0),
                        "train/rgb_warp_loss": loss_dict.get("rgb_warp_loss", 0),
                        "train/rgb_warp_loss_weighted": loss_dict.get("rgb_warp_loss_weighted", 0),
                        "global_step": global_step,
                    })

            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(args.output_path, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)

                if unwrapped.enable_motion:
                    torch.save(
                        unwrapped.motion_head.state_dict(),
                        os.path.join(save_path, "motion_head.pth"),
                    )

                if unwrapped.enable_depth:
                    torch.save(
                        unwrapped.depth_head.state_dict(),
                        os.path.join(save_path, "depth_head.pth"),
                    )

                for module_name in _split_comma_list(args.save_modules):
                    if not hasattr(unwrapped.pipe, module_name):
                        if os.environ.get("RANK", "0") == "0":
                            print(f"[WARN] pipe has no submodule '{module_name}', skipping save")
                        continue
                    module = getattr(unwrapped.pipe, module_name)
                    save_file = os.path.join(save_path, f"{module_name}.pth")
                    _save_module_state(module, save_file, args.save_trainable_only)

                print(f"Checkpoint saved to {save_path}")

    if accelerator.is_main_process:
        final_path = os.path.join(args.output_path, "final")
        os.makedirs(final_path, exist_ok=True)

        unwrapped = accelerator.unwrap_model(model)

        if unwrapped.enable_motion:
            torch.save(
                unwrapped.motion_head.state_dict(),
                os.path.join(final_path, "motion_head.pth"),
            )

        if unwrapped.enable_depth:
            torch.save(
                unwrapped.depth_head.state_dict(),
                os.path.join(final_path, "depth_head.pth"),
            )

        for module_name in _split_comma_list(args.save_modules):
            if not hasattr(unwrapped.pipe, module_name):
                if os.environ.get("RANK", "0") == "0":
                    print(f"[WARN] pipe has no submodule '{module_name}', skipping save")
                continue
            module = getattr(unwrapped.pipe, module_name)
            save_file = os.path.join(final_path, f"{module_name}.pth")
            _save_module_state(module, save_file, args.save_trainable_only)

        print(f"Final checkpoint saved to {final_path}")

    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
