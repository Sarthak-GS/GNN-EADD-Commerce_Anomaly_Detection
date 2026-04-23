import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup_cuda.py
# Compiles the raw CUDA kernels for GNN-EADD.
# Usage: python kernels/setup_cuda.py install

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='gnn_cuda_ext',
    ext_modules=[
        CUDAExtension(
            name='gnn_cuda_ext',
            sources=[
                os.path.join(here, 'cuda_extension.cpp'),
                os.path.join(here, 'cuda_kernels.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)
