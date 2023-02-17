from qiime2.plugin import Plugin, Citations

# from skbio.stats.distance import DistanceMatrix
from q2_types.feature_table import (FeatureTable, Frequency)
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
        'table': FeatureTable[Frequency]
    },
    parameters={},
    outputs=[
        ('distance_matrix', DistanceMatrix),
    ],
    input_descriptions={
        'phylogeny': (
            'Phylogeny tree associated with table.'
        ),
        'table': (
            'Biom table with samples and matching OTU IDs.'
        )
    },
    parameter_descriptions={},
    output_descriptions={
        'distance_matrix':
            'Resulting distance matrix.'
    },
    name='haarlikedist',
    description='Computes haar-like-distance between samples.',
    citations=[
        citations['Gorman2022'],
    ]
)
