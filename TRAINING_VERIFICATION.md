# Training Data Verification

## ✅ YES, ALL 21 SCENES WILL BE USED IN TRAINING

I've verified the complete data pipeline. Here's the proof:

## Verification Results

### 1. Metadata Generation ✓
```
Found 21 image sequence directories
Total samples: 21

All 21 test scenarios included:
1. test_ball_and_block_fall
2. test_ball_collide
3. test_ball_hits_duck
4. test_ball_hits_nothing
5. test_ball_in_basket
6. test_ball_ramp
7. test_ball_rolls_off
8. test_ball_rolls_on_glass
9. test_ball_train
10. test_block_domino
11. test_domino_in_juice
12. test_domino_with_space
13. test_duck_and_domino
14. test_duck_falls_in_box
15. test_duck_static
16. test_light_on_block
17. test_light_on_mug
18. test_light_on_mug_block
19. test_light_on_sculpture
20. test_roll_behind_box
21. test_roll_front_box
```

### 2. Dataloader Test ✓
```
✓ All 21 samples are accessible!
✓ Found 21 unique prompts
✓ Each sample loads 81 frames at 832x480 resolution
```

### 3. Dataset Repeat Mechanism ✓

**How it works:**

The `UnifiedDataset` class (diffsynth/trainers/unified_dataset.py:312, 325) uses:

```python
def __getitem__(self, data_id):
    data = self.data[data_id % len(self.data)].copy()  # Line 312: Modulo operation!
    # ... load the data

def __len__(self):
    return len(self.data) * self.repeat  # Line 325: Multiply by repeat
```

**With your training configuration:**
- Base dataset: 21 scenes
- Repeat factor: 100
- Total dataset length: 21 × 100 = **2,100 samples per epoch**

**What this means:**
- PyTorch DataLoader will iterate through indices 0 to 2099
- Index 0 → Scene 0 (test_ball_and_block_fall)
- Index 1 → Scene 1 (test_ball_collide)
- ...
- Index 20 → Scene 20 (test_roll_front_box)
- Index 21 → Scene 0 (test_ball_and_block_fall) ← Cycles back!
- Index 22 → Scene 1 (test_ball_collide)
- ...

**Each of the 21 scenes appears exactly 100 times per epoch**

### 4. Training Over 5 Epochs

With `NUM_EPOCHS=5`:
- Total training samples: 2,100 × 5 = **10,500 samples**
- Each scene seen: 100 × 5 = **500 times**

## How Data Flows Through Training

```
1. prepare_image_dataset.py
   → Scans /nyx-storage1/hanliu/Sim_Physics/TestOutput
   → Finds all test_*/env_0/0/0/rgb directories
   → Creates metadata.csv with 21 rows

2. train_wan_image_sequences.py
   → UnifiedDataset loads metadata.csv
   → Sets repeat=100
   → Dataset length becomes 21 × 100 = 2100

3. PyTorch DataLoader (in training loop)
   → Iterates through indices 0 to 2099
   → Each index maps to: scene_id = index % 21
   → All 21 scenes are used cyclically

4. Training loop (5 epochs)
   → Each epoch processes 2100 samples
   → All 21 scenes seen 100 times per epoch
   → Total: 500 views per scene across training
```

## Verification Code Ran

I tested this with:

### Test 1: Metadata Generation
```bash
python prepare_image_dataset.py \
  --source_dir /nyx-storage1/hanliu/Sim_Physics/TestOutput \
  --output_dir data/sim_physics_dataset_test \
  --pattern "rgb"

Result: ✓ Found 21 scenes, created metadata.csv
```

### Test 2: Dataloader Functionality
```bash
python test_dataloader.py

Result: ✓ All 21 samples load correctly with 81 frames each
```

### Test 3: Dataset Repeat Behavior
```python
dataset = UnifiedDataset(..., repeat=100)
len(dataset)  # Returns 2100
dataset[0]    # Returns scene 0
dataset[21]   # Returns scene 0 (cycles)
dataset[42]   # Returns scene 0 (cycles)
```

Result: ✓ Confirmed cyclic access pattern

## Summary

**Your training WILL use all 21 test scenarios from the TestOutput directory.**

The modulo operation (`data_id % len(self.data)`) in UnifiedDataset ensures that:
1. All 21 scenes are accessed
2. They are cycled through repeatedly
3. With repeat=100, each scene appears 100 times per epoch
4. Training is balanced across all physics scenarios

You can confidently run `./train_wan_sim_physics.sh` knowing that all your physics test scenarios will be included in the fine-tuning!
