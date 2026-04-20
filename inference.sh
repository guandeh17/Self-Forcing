pip install lmdb easydict

# add "from loguru import logger" to
# vim /usr/local/lib/python3.12/dist-packages/diffusers/quantizers/torchao/torchao_quantizer.py, line 93

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --output_folder videos/self_forcing_dmd \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/MovieGenVideoBench_extended.txt \
  --use_ema
