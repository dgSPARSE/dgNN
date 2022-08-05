output=edgeconv_result.csv

CUDA_VISIBLE_DEVICES=1 python perf_test/train_edgeconv_dgl.py --output=$output
# CUDA_VISIBLE_DEVICES=1 python perf_test/train_edgeconv_pyg.py --output=$output
# CUDA_VISIBLE_DEVICES=1 python ../dgNN/script/train/train_edgeconv.py --output=$output