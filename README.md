### How to run

```shell
cd src
bash install.sh
cd ..
cd script/train
python train_gatconv_our.py
```

### Requirement
dgl
scipy
pytorch

Our file tree:

```shell
.
├── data
│   └── Reddit
│       ├── processed
│       └── raw
├── layers
│   ├── edgeconv_layer.py
│   ├── gatconv_layer.py
│   ├── gmmconv_layer.py
│   └── __pycache__
│       ├── edgeconv_layer.cpython-38.pyc
│       ├── gatconv_layer.cpython-38.pyc
│       ├── gmmconv_layer.cpython-38.pyc
│       └── __init__.cpython-38.pyc
├── operators
│   ├── csr2csc.py
│   ├── edge_softmax.py
│   ├── fused_edgeconv.py
│   ├── fused_gat.py
│   ├── fused_gmmconv.py
│   ├── mhspmm.py
│   ├── __pycache__
│   │   ├── edge_softmax.cpython-38.pyc
│   │   ├── fused_edgeconv.cpython-38.pyc
│   │   ├── fused_gat.cpython-38.pyc
│   │   ├── fused_gmmconv.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── mhspmm.cpython-38.pyc
│   │   ├── spmm.cpython-38.pyc
│   │   └── u_add_v.cpython-38.pyc
│   └── u_add_v.py
├── README.md
├── script
│   ├── inference
│   │   ├── edgeconv_our_forward.py
│   │   ├── gatconv_our_forward.py
│   │   └── gmmconv_our_forward.py
│   └── train
│       ├── train_edgeconv_our.py
│       ├── train_gatconv_our.py
│       └── train_gmmconv_our.py
└── src
    ├── csr2csc
    │   ├── csr2csc.cc
    │   ├── csr2csc.cu
    │   ├── mhtranspose.cc
    │   └── mhtranspose.cu
    ├── edge_softmax
    │   ├── edge_softmax.cc
    │   └── edge_softmax.cu
    ├── fused_edgeconv
    │   ├── fused_edgeconv.cpp
    │   └── fused_edgeconv.cu
    ├── fused_gat
    │   ├── fused_gat.cpp
    │   └── fused_gat.cu
    ├── fused_gmmconv
    │   ├── gmmconv.cc
    │   └── gmmconv.cu
    ├── sddmm
    │   ├── mhsddmm.cc
    │   ├── mhsddmm.cu
    │   ├── sddmm.cpp
    │   └── sddmm.cu
    ├── spmm
    │   ├── mhspmm.cc
    │   └── mhspmm.cu
    └── util
        ├── computeUtil.h
        ├── indicator.cc
        └── indicator.cu

```
