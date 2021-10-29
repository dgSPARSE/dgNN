import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dgNN",
    version="0.0.2",
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
        CUDAExtension('fused_gat', ['dgNN/src/fused_gat/fused_gat.cpp', 'dgNN/src/fused_gat/fused_gat_kernel.cu']),
        CUDAExtension('fused_edgeconv',['dgNN/src/fused_edgeconv/fused_edgeconv.cpp','dgNN/src/fused_edgeconv/fused_edgeconv_kernel.cu']),
        CUDAExtension('fused_gmm',['dgNN/src/fused_gmm/fused_gmm.cpp','dgNN/src/fused_gmm/fused_gmm_kernel.cu']),
        CUDAExtension('mhsddmm',['dgNN/src/sddmm/mhsddmm.cc','dgNN/src/sddmm/mhsddmm_kernel.cu']),
        CUDAExtension('mhtranspose',['dgNN/src/csr2csc/mhtranspose.cc','dgNN/src/csr2csc/mhtranspose_kernel.cu'])
        ],
    
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)