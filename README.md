## dgNN

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

### LICENSE

This project is projected by [Apache-2.0](https://github.com/dgSPARSE/dgNN/blob/main/LICENSE) License.
If you use our dgNN project in your research, please cite the following bib:

```bibtex
@misc{zhang2021understanding,
    title={Understanding GNN Computational Graph: A Coordinated Computation, IO, and Memory Perspective},
    author={Hengrui Zhang and Zhongming Yu and Guohao Dai and Guyue Huang and Yufei Ding and Yuan Xie and Yu Wang},
    year={2021},
    eprint={2110.09524},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
