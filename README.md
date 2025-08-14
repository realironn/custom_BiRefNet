## Quick Test (Inference)

```bash
# test data root -> config.py 9~10 line
self.sys_home_dir = [os.path.expanduser('~'), 'your root'][1]   # Default, custom
self.data_root_dir = os.path.join(self.sys_home_dir, 'datasets')

# data folder
${data_root_dir}/${task}/${testset}/
├─ im/   # 입력 이미지
│   ├─ 0001.jpg (또는 .png)
│   └─ ...
└─ gt/   # GT 마스크(흑백)
    ├─ 0001.png
    └─ ...

# GPU 0번 사용, config.size 해상도, 전체 testsets 대상으로 추론
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
  --ckpt BiRefNet/ckpt/matting-no-freeze/epoch_110.pth \
  --pred_root e_preds \
  --resolution config.size


# 실행 스크립트 예시 (test.sh)
# GPU 번호, 예측 결과 저장 경로, 해상도 설정 가능
# 기본값: GPU 0, pred_root=e_preds, resolution=config.size
devices=${1:-0}
pred_root=${2:-e_preds}
resolutions=${3:-"config.size"}