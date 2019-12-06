from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='horder_cpp',
    ext_modules=[
        CppExtension('horder_cpp', ['horder.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
