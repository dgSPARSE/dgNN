import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dgNN",
    version="0.1",
    author="HenryChang fishming",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "dgNN"},
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    ext_modules=[
        CUDAExtension('fused_gatconv', ['dgNN/src/fused_gatconv/fused_gatconv.cpp', 'dgNN/src/fused_gatconv/fused_gatconv_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70']}),
        CUDAExtension('fused_edgeconv',['dgNN/src/fused_edgeconv/fused_edgeconv.cpp','dgNN/src/fused_edgeconv/fused_edgeconv_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70']}),
        CUDAExtension('fused_gmmconv',['dgNN/src/fused_gmmconv/fused_gmmconv.cpp','dgNN/src/fused_gmmconv/fused_gmmconv_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70']}),
        CUDAExtension('mhsddmm',['dgNN/src/sddmm/mhsddmm.cc','dgNN/src/sddmm/mhsddmm_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70']}),
        CUDAExtension('mhtranspose',['dgNN/src/csr2csc/mhtranspose.cc','dgNN/src/csr2csc/mhtranspose_kernel.cu'], extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70']})
        ],
    
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)