# q2-haarlikedist

![](https://github.com/dpear/q2-haarlikedist/actions/workflows/main.yml/badge.svg)


Qiime2 plugin for implementing the haar-like wavelet distance metric originally defined in Gorman et. al 2022.
For more information on Qiime2 visit: https://qiime2.org/

## Installation
Clone this git repo, navigate to the directory, and run:
```
pip install -e .
```

## About

The haar-like wavelet distance is a phylogenetically aware distance metric that projects distances onto a haar-like basis and returns additional information on which edges in a phylogeny tree contribute most to the difference between samples.

A tree is represented first as two sparsified matrices. A sparse-haar-like matrix represents each internal node in the tree as a haar-like wavelet. Another represents the internal nodes in the tree as weighted sums of the path to each tip. These matrices are then used to compute the distance between two samples using a phylogenetically aware projection onto a haar-like basis. These distances can be computed efficiently since the time consuming sparsification step need only be performed once.
