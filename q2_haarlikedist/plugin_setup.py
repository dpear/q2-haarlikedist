from qiime2.plugin import (Plugin, Citations,
                           MetadataColumn, Str,
                           Categorical)

# from skbio.stats.distance import DistanceMatrix
from q2_types.feature_table import (FeatureTable,
                                    Frequency,
                                    RelativeFrequency)
from q2_types.distance_matrix import DistanceMatrix
from q2_types.tree import (Phylogeny, Rooted)

from q2_haarlikedist._methods import haar_like_dist


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


plugin.methods.register_function(
    function=haar_like_dist,
    inputs={
        'phylogeny': Phylogeny[Rooted],
        'table': FeatureTable[Frequency | RelativeFrequency]
    },
    parameters={
        'comparative_column': MetadataColumn[Categorical],
        'comparative_value': Str
    },
    outputs=[
        ('distance_matrix', DistanceMatrix),
        ('annotated_tree', Phylogeny[Rooted])
    ],
    input_descriptions={
        'phylogeny': (
            'Phylogeny tree associated with table.'
        ),
        'table': (
            'Biom table with samples and matching OTU IDs.'
        )
    },
    parameter_descriptions={
        'group_column': ('Name of metadata column to use '
                         'for group comparisons.'),
        'group_value': ('Name of group to compare against '
                        'all others.')
    },
    output_descriptions={
        'distance_matrix':
            'Resulting distance matrix.',
        'annotated_tree':
            ('Resulting tree with annotated number of times '
             'the edge is most significant.')
    },
    name='haarlikedist',
    description='Computes haar-like-distance between samples.',
    citations=[
        citations['Gorman2022'],
    ]
)
