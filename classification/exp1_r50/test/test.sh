CUDA_VISIBLE_DEVICES=0 \
python -u test.py \
--print_freq 5 \
-op /DATA5_DB8/data/zdcheng/hyperthyreosis_eye/output \
--suffix exp1_r50 \
--batch_size 32 \
--resume /DATA5_DB8/data/zdcheng/hyperthyreosis_eye/checkpoint/exp1_r50/model005.pth \
--threshold 0.5 \
