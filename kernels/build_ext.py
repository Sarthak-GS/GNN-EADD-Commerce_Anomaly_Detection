"""
build_ext.py  —  Compiles the OpenMP C extension for GNN-EADD kernels.

Usage:
    python kernels/build_ext.py build_ext --inplace

The output is kernels/openmp_ext.so (Linux) / openmp_ext.pyd (Windows).
If gcc + OpenMP are not available the script exits cleanly with a warning.
"""

import os
import sys
import subprocess
import warnings


def compile_openmp_extension():
    here    = os.path.dirname(os.path.abspath(__file__))
    c_src   = os.path.join(here, 'openmp_kernels.c')
    out_so  = os.path.join(here, 'openmp_ext.so')

    cmd = [
        'gcc', '-O3', '-fopenmp', '-shared', '-fPIC',
        '-o', out_so, c_src, '-lm',
    ]

    print(f"[build_ext] Compiling: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[build_ext] Success: {out_so}")
        else:
            print(f"[build_ext] Compilation failed:")
            print(result.stderr)
            sys.exit(1)
    except FileNotFoundError:
        warnings.warn(
            "[build_ext] gcc not found. OpenMP extension will not be available. "
            "Training will fall back to PyTorch sequential kernels.",
            RuntimeWarning,
        )


if __name__ == '__main__':
    compile_openmp_extension()
