#! /usr/bin/env python
# coding: utf-8

from setuptools import find_packages, setup
import os

setup(name='eTraGo',
      author='NEXT ENERGY, ZNES Flensburg',
      author_email='',
      description='electrical Transmission Grid Optimization of flexibility options for transmission grids based on PyPSA ',
      version='0.0.1',
      url='https://github.com/openego/eTraGo',
      packages=find_packages(),
      license='GNU Affero General Public License v3.0',
      install_requires=[
          'geoalchemy2 >= 0.3.0, <=0.4.0',
          'sqlalchemy >= 1.0.15, <= 1.1.9',
          'numpy >= 1.11.3, <= 1.12.1',
          'egoio >= 0.2.0, <= 0.2.0',
          'ego.powerflow == 0.0.5',
          'psycopg2'],
      extras_require={
          "sqlalchemy": 'postgresql'}
)
