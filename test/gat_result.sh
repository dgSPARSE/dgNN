# rm gat_result.csv

# CUDA_VISIBLE_DEVICES=1 python ../dgNN/script/train/train_gatconv.py --dataset=reddit --n-hidden=64 --n-heads=4 --n-epochs=100 --gpu=0 --output=gat_result.csv --attn-drop=0. --dropout=0.5
# python perf_test/train_gat_pyg_cpu.py --dataset=reddit --n-hidden=64 --n-heads=1 --n-epochs=100 --gpu=1 --output=gat_result.csv --attn-drop=0.5 --dropout=0.5

# python perf_test/train_gat_pyg.py --dataset=reddit --n-hidden=64 --n-heads=4 --n-epochs=10 --gpu=1 --output=gat_result.csv --attn-drop=0.5 --dropout=0.5

for dataset in  "cora" #"pubmed" "citeseer" 
do
    python perf_test/train_gat_pyg.py --dataset=$dataset --n-hidden=64 --n-heads=4 --n-epochs=100 --gpu=1 --output=gat_result.csv --attn-drop=0.5 --dropout=0.5
    CUDA_VISIBLE_DEVICES=1 python perf_test/train_gat_dgl.py --dataset=$dataset --n-hidden=64 --n-heads=1 --n-epochs=100 --gpu=0 --output=gat_result.csv --attn-drop=0.5 --dropout=0.5
    CUDA_VISIBLE_DEVICES=1 python ../dgNN/script/train/train_gatconv.py --dataset=$dataset --n-hidden=64 --n-heads=1 --n-epochs=100 --gpu=0 --output=gat_result.csv --attn-drop=0.5 --dropout=0.5
done
