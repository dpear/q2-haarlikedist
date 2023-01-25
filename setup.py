#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2022--, convex-hull development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages

setup(
    name="q2-haarlikedist",
    packages=find_packages(),
    version='0.0.1',
    author="Daniela Perry",
    author_email="dsperry@ucsd.edu",
    description="compute haar-like distance",
    license='BSD-3',
    entry_points={
        'qiime2.plugins': ['q2-haarlikedist=q2_haarlikedist.plugin_setup:plugin']
    },
    package_data={
        "q2_haarlikedist": ['citations.bib'],
    },
    zip_safe=False,
    install_requires=['scipy',
                      'scikit-bio',
                      'pandas',
                      'biom-format']
)
