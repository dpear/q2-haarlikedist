import importlib
from qiime2.plugin import (Plugin, Int, Citations,
                           Str, Range, Metadata)


from q2_types.sample_data import SampleData
from q2_types.ordination import PCoAResults

import q2_haarlikedist
from . import haar_like_dist
from q2_types.feature_table import (FeatureTable, Frequency)


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

plugin.visualizers.register_function(
    function=haar_like_dist,
    inputs={},
    parameters={},
    input_descriptions={},
    parameter_descriptions={},
    name='haarlikedist',
    description='Computes the haar-like distance matrix.',
    citations=[
        citations['Gorman2022'],
    ]
)

