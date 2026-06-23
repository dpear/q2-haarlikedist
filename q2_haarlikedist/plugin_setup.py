import importlib

from qiime2.plugin import (Plugin, Citations, Str, Int, Bool, Metadata)


# from skbio.stats.distance import DistanceMatrix
from q2_types.ordination import PCoAResults
from q2_types.feature_table import (FeatureTable,
                                    Frequency,
                                    RelativeFrequency)

from q2_types.distance_matrix import DistanceMatrix
from q2_types.tree import (Phylogeny, Rooted)
from q2_types.feature_data import (FeatureData, Taxonomy)


from q2_haarlikedist._methods import (haar_like_dist,
                                      adaptive_distance,
                                      adaptive_visual)

from q2_haarlikedist._type import Modmags
from q2_haarlikedist._format import ModmagsFormat, ModmagsDirFormat

citations = Citations.load('citations.bib', package='q2_haarlikedist')

plugin = Plugin(
    name='haarlikedist',
    version='0.1.1',
    website='https://github.com/dpear/q2-haarlikedist',
    package='q2_haarlikedist',
    description=('This QIIME 2 plugin implements haar-like '
                 'distance calculation as described by '
                 'E et. al 2022.'),
    short_description='Plugin for haar-like distance.',
)

plugin.register_formats(ModmagsFormat, ModmagsDirFormat)
plugin.register_semantic_types(Modmags)
plugin.register_semantic_type_to_format(
    Modmags,
    artifact_format=ModmagsDirFormat)

plugin.methods.register_function(
    function=haar_like_dist,
    inputs={
        'tree': Phylogeny[Rooted],
        'table': FeatureTable[Frequency | RelativeFrequency]
    },
    parameters={},
    outputs=[
        ('distance_matrix', DistanceMatrix),
        ('annotated_tree', Phylogeny[Rooted]),
        ('modmags', Modmags),
        ('pcoa', PCoAResults)
    ],
    input_descriptions={
        'tree': (
            'Phylogeny tree associated with table.'
        ),
        'table': (
            'Biom table with samples and matching OTU IDs.'
        )
    },
    parameter_descriptions={},
    output_descriptions={
        'distance_matrix':
            ('Resulting pairwise distance matrix computed from '
             'modmags.'),
        'annotated_tree':
            ('Resulting tree with annotated number of times '
             'the edge is most significant.'),
        'modmags':
            ('A feature table which can be seen as '
             'a differential encoding. Distances can be '
             'calculated from this matrix in several different '
             'ways.'),
        'pcoa':
            ('PCoA plot of the distance matrix.')
    },
    name='haarlikedist',
    description='Computes haar-like-distance between samples.',
    citations=[
        citations['Gorman2022'],
    ]
)

plugin.visualizers.register_function(
    function=adaptive_visual,
    inputs={
        'tree': Phylogeny[Rooted],
        'biom_table': FeatureTable[Frequency | RelativeFrequency]
    },
    parameters={
        'label': Str,
        'metadata': Metadata,
        'taxonomy': Metadata,
        's': Int,
        'k': Int,
        'n': Int,
        'cluster_affinity': Bool,
        'num_clstr': Int,
        'tune': Bool
    },
    input_descriptions={
        'tree': (
            'Phylogeny tree associated with table.'
        ),
        'biom_table': (
            'Biom table with samples and OTU IDs (features)'
            'that match tree tip names.'
        )
    },
    parameter_descriptions={
        'label': ('Name of metadata column to use '
                  'for group comparisons. Variable of interest'),
        'metadata': 'Associated metadata that is a superset of samples.',
        'taxonomy': ('A qiime2 taxonomy file mapping tree tip '
                     'names to species names.'),
        's': ('Number of important nodes to find'),
        'k': ('for PCoA reconstruction'),  # TODO: update with more info
        'n': ('for PCoA reconstruction'),  # TODO: update with more info
        'cluster_affinity': ('Subsampling with clustering of tree representation affinity when sample size is big '
                             '(default: True).'),
        'num_clstr': ('Number of clusters for clustering step '
                      '(default: 2000).'),
        'tune': ('Tune RF hyperparameters - slows down the pipeline (default: False)'),
    },
    name='adaptive-visual',
    description='Computes haar-like-distance between samples using new'
                'supervised method',
    citations=[
        citations['Gorman2022'],
    ]
)

plugin.methods.register_function(
    function=adaptive_distance,
    inputs={
        'tree': Phylogeny[Rooted],
        'table': FeatureTable[Frequency | RelativeFrequency]
    },
    parameters={
        'label': Str,
        'metadata': Metadata,
        'taxonomy': Metadata,
        's': Int
    },
    outputs=[
        ('distance_matrix', DistanceMatrix),
        ('annotated_tree', Phylogeny[Rooted]),
        ('modmags', Modmags),
        ('pcoa', PCoAResults),
        ('feature_metadata', FeatureData[Taxonomy])
    ],
    input_descriptions={
        'tree': ('Phylogeny tree associated with table.'),
        'table': ('Biom table with samples and matching OTU IDs.')
    },
    parameter_descriptions={
        'label': ('Name of metadata column to use '
                  'for group comparisons. Variable of interest'),
        'metadata': 'Associated metadata that is a superset of samples.',
        'taxonomy': ('A qiime2 taxonomy file mapping tree tip '
                     'names to species names.'),
        's': ('Number of important nodes to find'),
    },
    output_descriptions={
        'distance_matrix':
            ('Resulting pairwise distance '
             'matrix computed from modmags.'),
        'annotated_tree':
            ('Resulting tree with trimmed tips used in analysis. '
             'Has annotated internal nodes which match up with '
             'feature metadata.'),
        'modmags':
            ('A feature table which can be seen as '
             'a differential encoding. Distances can be '
             'calculated from this matrix '
             'in several different ways.'),
        'pcoa':
            ('PCoA plot of the distance matrix.'),
        'feature_metadata':
            ('Metadata file with information on tree tips '
             'and internal nodes.')
    },
    name='adaptive-distance',
    description='Computes haar-like-distance between samples.',
    citations=[
        citations['Gorman2022'],
    ]
)

import q2_haarlikedist._transformer
