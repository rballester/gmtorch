#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

setup(
    name='gmtorch',
    version='0.1.0',
    description="Graphical Models in PyTorch",
    long_description="gmtorch is a PyTorch-powered library to model Bayesian networks, Markov random fields, and tensor decompositions.",
    url='https://github.com/rballester/gmtorch',
    author="Rafael Ballester-Ripoll",
    author_email='rafael.ballester@ie.edu',
    packages=[
        'gmtorch',
    ],
    include_package_data=True,
    install_requires=[
        'numpy',
        'torch'
    ],
    license="BSD",
    zip_safe=False,
    keywords='gmtorch',
    classifiers=[
        'License :: OSI Approved',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require='pytest'
)
