# Sim_Physics Dataset Setup for WAN2.2 Training

## Dataset Structure

Your dataset at `/nyx-storage1/hanliu/Sim_Physics/TestOutput` has been analyzed and configured for training.

### Format
- **Total samples**: 21 physics simulation scenes
- **Frames per sample**: 131 frames
- **Image resolution**: 480x480 pixels
- **Motion vectors**: 4 channels (H, W, 4) - float32
- **Depth maps**: Single channel (H, W) - float32

### Directory Structure
```
TestOutput/
├── test_ball_and_block_fall/
│   └── env_0/0/0/
│       ├── rgb/               # RGB frames (step_XXXX.png)
│       ├── motion_vectors/    # Motion flow (step_XXXX.npy)
│       └── distance_to_camera/ # Depth maps (step_XXXX.npy)
├── test_ball_collide/
│   └── env_0/0/0/
│       ├── rgb/
│       ├── motion_vectors/
│       └── distance_to_camera/
└── ... (19 more scenes)
```

## Changes Made

### 1. Created Metadata CSV
- **Location**: `/home/mzh1800/DiffSynth-Studio/data/sim_physics_metadata.csv`
- **Format**: CSV with columns `video` (relative path) and `prompt` (description)
- **Samples**: All 21 physics simulation scenes

### 2. Updated Training Script
- **File**: `train_wan22_ti2v_motion_depth.py`
- **Added**: Data loading functions for directory-based datasets
  - `LoadImageSequenceWithMotionWrapper` - Handles rgb/, motion_vectors/, distance_to_camera/ structure
  - `create_motion_data_operator` - Creates proper data pipeline
- **Fixed**: Import errors (removed unused imports)

### 3. Updated Training Bash Script
- **File**: `train_wan22_stage1_heads.sh`
- **Updated paths**:
  - `DATASET_BASE_PATH="/nyx-storage1/hanliu/Sim_Physics/TestOutput"`
  - `DATASET_METADATA_PATH="/home/mzh1800/DiffSynth-Studio/data/sim_physics_metadata.csv"`
- **Updated dimensions**:
  - `HEIGHT=480` (matches your data)
  - `WIDTH=480` (matches your data)
  - `NUM_FRAMES=49` (will sample from 131 available frames)

## How to Run Training

### Stage 1: Train Motion and Depth Heads Only

```bash
bash train_wan22_stage1_heads.sh
```

This will:
1. Load WAN2.2-5B model from local checkpoints
2. Train motion and depth prediction heads
3. Keep the DiT backbone frozen
4. Save checkpoints every 500 steps
5. Log to wandb (if enabled)

### Important Configuration

**Training Parameters** (in `train_wan22_stage1_heads.sh`):
- Learning rate: `1e-4` (higher for heads-only training)
- Batch size: `1`
- Gradient accumulation: `8` (effective batch size = 8)
- Epochs: `10`
- Loss weights:
  - Motion loss: `1.0`
  - Depth loss: `1.0`

**GPU Configuration**:
- Currently set to use GPUs 2,3: `CUDA_VISIBLE_DEVICES=2,3`
- Using 2 processes with accelerate

**Output Location**:
- Checkpoints: `/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/wan22_ti2v_stage1/`

## Dataset Statistics

- **Total samples**: 21
- **With dataset_repeat=100**: 2,100 training iterations per epoch
- **Total iterations** (10 epochs): 21,000 iterations
- **Checkpoint frequency**: Every 500 steps (~2.4% of training)

## Verification

The metadata CSV has been created and verified:
```
Number of samples: 21
Columns: ['video', 'prompt']
First sample: test_ball_and_block_fall/env_0/0/0
```

## Next Steps

After Stage 1 completes:

1. **Check results**: Motion and depth heads saved to `${OUTPUT_PATH}/final/`
2. **Run Stage 2**: Use `train_wan22_stage2_lora.sh` to fine-tune the backbone with LoRA
3. **Evaluate**: Test the model's physics understanding on new scenes

## Troubleshooting

If you encounter issues:

1. **Memory errors**: Reduce batch_size or num_frames
2. **Data loading errors**: Check that all samples have rgb/, motion_vectors/, distance_to_camera/ subdirs
3. **Training instability**: Reduce learning rate or adjust loss weights

## Reference

The data loading implementation follows the pattern from `train_wan_with_motion.py`:
- Uses `LoadImageSequenceWithMotion` for efficient frame loading
- Handles motion vector and depth map loading
- Supports frame subsampling (49 frames from 131)
- Applies proper image preprocessing (resize, crop)
