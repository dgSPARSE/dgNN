for dataset in  "cora" "pubmed" "citeseer" "reddit" 
do
    python func_test/func_gat_dgl.py --dataset=$dataset
    python func_test/func_gat_pyg.py --dataset=$dataset
done