CUDA_VISIBLE_DEVICES=1 python train.py --data yahoo_ans --min_freq 10 --epochs 100 --sigma 5e5 --gamma 0.3 && \
CUDA_VISIBLE_DEVICES=1 python train.py --data yahoo_ans --min_freq 10 --epochs 100 --sigma 0.0 --gamma 0.3