from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='horder_cuda',
    ext_modules=[
        CUDAExtension('horder_cuda', [
            'horder_cuda.cpp',
            'horder_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
