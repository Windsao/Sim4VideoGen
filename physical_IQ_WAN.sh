python examples/wanvideo/image_to_video_inference.py \
  /nyx-storage1/hanliu/physics-IQ-benchmark/physics-IQ-benchmark/switch-frames \
  --description-csv /nyx-storage1/hanliu/physics-IQ-benchmark/descriptions/descriptions.csv \
  --variant wan2.2-i2v-a14b \
  --local-model-root /nyx-storage1/hanliu/world_model_ckpt \
  --output-dir /nyx-storage1/hanliu/physics-IQ_WAN \
  --model-tag wan2.2-i2v-a14b \
  --resize-image --tiled --skip-existing
