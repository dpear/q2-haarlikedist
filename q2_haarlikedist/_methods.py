# ----------------------------------------------------------------------------
# Copyright (c) 2022--, convex-hull development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

# import pandas as pd
# from scipy.spatial import ConvexHull
# from skbio import OrdinationResults
# from q2_convexhull._defaults import (DEFAULT_N_DIMENSIONS)
# from warnings import warn
# from qiime2 import Metadata
from ete3 import Tree
import pkg_resources
import q2templates
import os


def haar_like_dist(output_dir: str) -> None:

    """ Computes Haar Like Distance between two
        samples by projecting their phylogenies
        onto a Haar-like wavelet space.

    Parameters
    ----------
    table: FeatureTable 

    phylogeny: UM

    Returns
    -------
    output.qzv:
        visualization which contains a downloadable
        distance matrix as well as vectors
        which correspond to importance of 
        difference.

    Raises
    ------
    TypeError, ValueError
        If inputs are of incorrect type. If column ID not
        found in metadata.
    """

    t = Tree()

    TEMPLATES = pkg_resources.resource_filename(
        'q2_haarlikedist', 'haar_like_dist_assets')
    index = os.path.join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir)