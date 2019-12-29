CUDA_VISIBLE_DEVICES=0 python -u train.py \
--print_freq 5 --sum_freq 5 \
--save_freq 5 \
-sp /DATA5_DB8/data/zdcheng/hyperthyreosis_eye/summary \
-cp /DATA5_DB8/data/zdcheng/hyperthyreosis_eye/checkpoint \
-op /DATA5_DB8/data/zdcheng/hyperthyreosis_eye/output \
--suffix exp1_r50 \
--lr_path lr_exp1.txt \
--threshold 0.5 \
--valfold 0 \

