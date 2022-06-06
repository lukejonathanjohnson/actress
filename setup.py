#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:05:25 2020
@author: lukejohnson1
"""


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
      name='actress',
      url='https://github.com/lukejonathanjohnson/actress',
      author='Luke J. Johnson',
      author_email='l.johnson17@imperial.ac.uk',
      packages=['actress'],
      include_package_data=True,
      install_requires=['numpy', 'scipy', 'healpy', 'pandas', 'spectres',
                        'astropy', 'photutils', 'joblib', 'uncertainties'],
      version='1.0',
      license='Imperial College London',
      description='Open source program for simulating stellar variability due to spots and faculae.',
      long_description=long_description,
      python_requires='>=3.7')
