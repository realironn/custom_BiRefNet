## Quick Test (Inference)

```bash
# GPU 0번 사용, config.size 해상도, 전체 testsets 대상으로 추론
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
  --ckpt BiRefNet/ckpt/matting-no-freeze/epoch_110.pth \
  --pred_root e_preds \
  --resolution config.size
