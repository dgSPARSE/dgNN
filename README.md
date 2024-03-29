## dgNN

> **Note**: We'd like to inform our users that dgNN will no longer be actively maintained. Our development efforts will focus on integrating its features and functionalities into [dgSPARSE-Lib](https://github.com/dgSPARSE/dgSPARSE-Lib). We believe this consolidation will provide a more streamlined and efficient experience for our user community. Thank you for your continued support.

dgNN is a high-performance backend for GNN layers with DFG (Data Flow Graph) level optimization. dgNN project is based on [PyTorch](https://github.com/pytorch/pytorch).

### How to install

**through pip**

```
pip install dgNN
```

If pip couldn't build dgNN, we recommend you to build dgNN from source.

```shell
git clone git@github.com:dgSPARSE/dgNN.git
cd dgNN
bash install.sh
```

### Requirement

```
CUDA toolkit >= 10.0
pytorch >= 1.7.0
scipy
dgl >= 0.7 (We use dgl's dataset)
ninja
```

We prepare a docker to run our implementation. You could run our dgNN in a docker container.

```shell
cd docker
docker build -t dgNN:v1 -f Dockerfile .
docker run -it dgNN:v1 /bin/bash
```

### Examples

Our training script is modified from [DGL](https://github.com/dmlc/dgl). Now we implements three popular GNN models.

**Run GAT**

[DGL Code](https://github.com/dmlc/dgl/tree/master/examples/pytorch/gat)

```python
cd dgNN/script/train
python train_gatconv.py --num-hidden=64 --num-heads=4 --dataset cora --gpu 0
```

**Run Monet**

[DGL Code](https://github.com/dmlc/dgl/tree/master/examples/pytorch/monet)

```python
cd dgNN/script/train
python train_gmmconv.py --n-kernels 3 --pseudo-dim 2 --dataset cora --gpu 0
```

**Run PointCloud**

We use modelnet40-sampled-2048 data in our PointNet. [DGL Code](https://github.com/dmlc/dgl/tree/master/examples/pytorch/pointcloud)

```python
cd dgNN/script/train
python train_edgeconv.py
```

### Collaborative Projects

[CogDL](https://github.com/THUDM/cogdl) is a flexible and efficient graph-learning framework that uses GE-SpMM to accelerate GNN algorithms. This repo is implemented in CogDL as a submodule.

### Citation

If you use our dgNN project in your research, please cite the following bib:

This project also implements part of algorithms from [GNN-computing](https://github.com/xxcclong/GNN-Computing), especially method of neighbor grouping in SpMM. If you use our dgNN project in your research, please also cite the following bib:

```bibtex
@inproceedings{huang2021understanding,
  title={Understanding and bridging the gaps in current GNN performance optimizations},
  author={Huang, Kezhao and Zhai, Jidong and Zheng, Zhen and Yi, Youngmin and Shen, Xipeng},
  booktitle={Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming},
  pages={119--132},
  year={2021}
}
```

---

If you meet any problems in this repo, fill free to write issues or contact us by e-mail (team@dgsparse.org).
