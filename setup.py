"""
setup.py — Build script for GNN-EADD CUDA kernels (PyTorch C++ Extension)

[Lec 4] GPGPU Computing: This setup script compiles our custom CUDA kernels
into a loadable Python module using PyTorch's CUDAExtension. After
installation, the kernels are callable as normal Python functions while
executing on the GPU.

Usage:
    pip install -e .                    # editable install (recommended)
    python setup.py install             # standard install
    python setup.py build_ext --inplace # build in-place (alternative)
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gnn_cuda_kernels',
    version='2.0.0',
    description='Custom CUDA kernels for GNN-EADD Phase 2',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='gnn_cuda_kernels',
            sources=[
                'csrc/binding.cpp',
                'csrc/kernels.cu',
            ],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    # Target Ampere architecture (RTX 2050 = sm_86, compute 8.6)
                    '-gencode=arch=compute_86,code=sm_86',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
