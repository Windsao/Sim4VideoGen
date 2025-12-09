#!/bin/bash

# Organize I2V model files into a flat structure like the T2V model
# This makes it easier to manage and matches the existing T2V structure

SOURCE_DIR="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-I2V-14B-480P"
TARGET_DIR="/nyx-storage1/hanliu/world_model_ckpt/Wan-AI/Wan2.1-I2V-14B-480P"

echo "================================================"
echo "Organizing Wan2.1-I2V-14B-480P Model Files"
echo "================================================"
echo ""

# Find the actual model directory (ModelScope downloads to nested structure)
MODEL_SOURCE=$(find "$SOURCE_DIR" -type d -name "Wan2*1-I2V-14B-480P" 2>/dev/null | grep -v "._____temp" | head -1)

if [ -z "$MODEL_SOURCE" ]; then
    echo "❌ Could not find model source directory"
    echo "Download may still be in progress. Check:"
    echo "  tail -f download_i2v_model.log"
    exit 1
fi

echo "Source directory: $MODEL_SOURCE"
echo "Target directory: $TARGET_DIR"
echo ""

# Check if files are already in the target directory
if [ -f "$TARGET_DIR/diffusion_pytorch_model-00001-of-00007.safetensors" ] && \
   [ ! -f "$TARGET_DIR/._____temp" ]; then
    echo "✅ Model files are already organized in target directory"
    exit 0
fi

echo "Moving model files to flat structure..."
echo ""

# Move all model files to the target directory
for file in "$MODEL_SOURCE"/*; do
    filename=$(basename "$file")
    if [ -f "$file" ]; then
        echo "  Moving: $filename"
        mv "$file" "$TARGET_DIR/"
    elif [ -d "$file" ] && [ "$filename" = "google" ]; then
        # Move the google tokenizer directory
        echo "  Moving directory: $filename"
        if [ -d "$TARGET_DIR/google" ]; then
            rm -rf "$TARGET_DIR/google"
        fi
        mv "$file" "$TARGET_DIR/"
    fi
done

# Clean up the nested directory structure
echo ""
echo "Cleaning up temporary directories..."
rm -rf "$SOURCE_DIR/Wan-AI"
rm -rf "$SOURCE_DIR/._____temp"

echo ""
echo "================================================"
echo "✅ Organization complete!"
echo "================================================"
echo ""
echo "Model files are now in:"
echo "  $TARGET_DIR"
echo ""
echo "You can verify with:"
echo "  ls -lh $TARGET_DIR/*.safetensors"
echo "  ls -lh $TARGET_DIR/*.pth"
