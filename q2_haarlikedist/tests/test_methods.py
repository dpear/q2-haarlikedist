import tempfile
import os

from unittest import TestCase, main
from os.path import dirname, abspath, join
from inspect import currentframe, getfile
from q2_haarlikedist._methods import haar_like_dist


class TestHaarLikeDist(TestCase):

    def test_haar_like_dist(self):

        with tempfile.TemporaryDirectory() as output_dir:

            index_fp = os.path.join(output_dir, 'index.html')

            haar_like_dist(output_dir)
            self.assertTrue(os.path.exists(index_fp))



