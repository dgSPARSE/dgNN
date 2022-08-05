output="gmm_result.csv"

# rm $output

# for dataset in "cora" "pubmed" "citeseer" "reddit"
# do
#     CUDA_VISIBLE_DEVICES=1 python perf_test/train_gmm_pyg.py --dataset=$dataset --n-hidden=64 --pseudo-dim=2 --n-kernels=3 --n-epochs=100 --gpu=0 --output=$output --dropout=0.5
#     CUDA_VISIBLE_DEVICES=1 python perf_test/train_gmm_dgl.py --dataset=$dataset --n-hidden=64 --pseudo-dim=2 --n-kernels=3 --n-epochs=100 --gpu=0 --output=$output --dropout=0.5
#     CUDA_VISIBLE_DEVICES=1 python ../dgNN/script/train/train_gmmconv.py --dataset=$dataset --n-hidden=64 --pseudo-dim=2 --n-kernels=3 --n-epochs=100 --gpu=0 --output=$output --dropout=0.5
# done

CUDA_VISIBLE_DEVICES=1 python perf_test/train_gmm_pyg_cpu.py --dataset=reddit --n-hidden=64 --pseudo-dim=2 --n-kernels=2 --n-epochs=1 --gpu=0 --output=$output --dropout=0.5
CUDA_VISIBLE_DEVICES=1 python perf_test/train_gmm_dgl.py --dataset=reddit --n-hidden=64 --pseudo-dim=2 --n-kernels=2 --n-epochs=100 --gpu=0 --output=$output --dropout=0.5
CUDA_VISIBLE_DEVICES=1 python ../dgNN/script/train/train_gmmconv.py --dataset=reddit --n-hidden=64 --pseudo-dim=2 --n-kernels=2 --n-epochs=100 --gpu=0 --output=$output --dropout=0.5