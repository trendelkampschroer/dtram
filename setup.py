#!/usr/bin/env python
from setuptools import setup, find_packages, Extension

ext_objective_sparse = Extension("dtram.objective_sparse",
                                 sources=["dtram/objective_sparse.pyx",],
                                 libraries=["m",])

setup(name="dtram",
      version = "0.0.1",
      description = "Python DTRAM solver",
      author = "Benjamin Trendelkamp-Schroer",
      author_email = "Benjamin.Trendelkamp-Schroer@fu-berlin.de",
      packages = find_packages(),
      ext_modules=[ext_objective_sparse,],
      install_requires = ['numpy>=1.7.1', 
                          'scipy>=0.11']
      )
    
