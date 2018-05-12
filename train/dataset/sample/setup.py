#!/usr/bin/env python
#-*- coding:utf8 -*-

# Created by Li Bo 2018-04-23 14:58:05
from distutils.core import setup, Extension
import numpy as np
import os


extra_link_args = ['-fopenmp']
extra_objects = [os.path.abspath('resize.o')]
# extra_link_args=['-std=c++11', '-Wall', '-msse', '-msse2', '-pthread', '-shared', ] #'-DKALDI_DOUBLEPRECISION=0', '-DHAVE_EXECINFO_H=1', '-DHAVE_CXXABI_H', '-DHAVE_ATLAS', '-O2', '-fPIC'],
ext_modules = [ Extension('_sample',
    sources = ['_sample.cpp'],
    language='c++11',
    extra_compile_args=['-std=c++11', '-O3', '-fPIC', '-DNDEBUG', '-fopenmp'],
    extra_link_args=extra_link_args,
    extra_objects=extra_objects,
    swig_opts=['-c++']
    )]

setup(
    name = '_sample',
    version = '1.0',
    include_dirs = [np.get_include()], #Add Include path of numpy
    ext_modules = ext_modules,
)
