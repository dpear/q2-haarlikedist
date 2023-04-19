from unittest import TestCase
import qiime2
import numpy as np
import scipy
import skbio
import biom
from os.path import dirname, abspath, join
from inspect import currentframe, getfile

from q2_haarlikedist._methods import (initiate_values,
                                      get_case,
                                      get_nontip_index,
                                      get_tip_indeces,
                                      get_lstar,
                                      get_L,
                                      get_haarvec,
                                      get_lilmat_and_shl,
                                      handle_neither,
                                      handle_left,
                                      handle_right,
                                      handle_both,
                                      # create_branching_tree,
                                      sparsify,
                                      get_lambda,
                                      match_to_tree,
                                      compute_haar_dist,
                                      # haar_like_dist,
                                      get_lambdas)


class TestSparsify(TestCase):

    def setUp(self):

        path = dirname(abspath(getfile(currentframe())))

        # Initialize blank matrices
        self.ntips = 7
        self.lilmat = scipy.sparse.lil_matrix((self.ntips, self.ntips))
        self.shl = scipy.sparse.lil_matrix((self.ntips, self.ntips))

        # Test data table (has one OTU not present in the tree)
        testdata = 'test-data.qza'
        testdatafile = join(path, testdata)
        table = qiime2.Artifact.load(testdatafile)
        self.table = table.view(view_type=biom.Table)

        # Test tree (has seven tips as described in the diagram)
        testtree = 'small-tree.qza'
        testtreefile = join(path, testtree)
        tree = qiime2.Artifact.load(testtreefile)
        self.tree = tree.view(view_type=skbio.TreeNode)

    def test_initiate_values(self):

        expected = scipy.sparse.lil_matrix((self.ntips, self.ntips))

        _, t2, _ = match_to_tree(self.table, self.tree)
        t2, lilmat, shl = initiate_values(t2)

        assert (lilmat != expected).nnz == 0
        assert (shl != expected).nnz == 0

        for i, node in enumerate(t2.non_tips(include_self=True)):
            assert node.postorder_pos == i

        for i, node in enumerate(t2.postorder(include_self=True)):

            assert node.ntips_in_tree == 7

            if i == 0:
                assert node.tip_names == [0]
            if i == 1:
                assert node.tip_names == [1]
            if i == 2:
                assert node.tip_names == [2]
            if i == 3:
                assert node.tip_names == [1, 2]
            if i == 4:
                assert node.tip_names == [0, 1, 2]
            if i == 11:
                assert node.tip_names == [3, 4, 5, 6]
            if i == 12:
                assert node.tip_names == [0, 1, 2, 3, 4, 5, 6]

    def test_get_case(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            if i == 0:
                assert get_case(node) == 'both'
            if i == 1:
                assert get_case(node) == 'left'
            if i == 2:
                assert get_case(node) == 'both'
            if i == 3:
                assert get_case(node) == 'right'
            if i == 4:
                assert get_case(node) == 'left'
            if i == 5:
                assert get_case(node) == 'neither'

    def test_get_nontip_index(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            if i == 0:  # case = both, neither children nontips
                continue
            if i == 1:  # case = left, right child is nontip
                assert get_nontip_index(node, 'right') == 0
            if i == 2:  # case = both
                continue
            if i == 3:  # case = right
                assert get_nontip_index(node, 'left') == 2
            if i == 4:  # case = left
                assert get_nontip_index(node, 'right') == 3
            if i == 5:  # case = neither
                assert get_nontip_index(node, 'left') == 1
                assert get_nontip_index(node, 'right') == 4

    def test_get_tip_indeces(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            if i == 0:
                assert get_tip_indeces(node, 'left') == [1]
                assert get_tip_indeces(node, 'right') == [2]
            if i == 1:
                assert get_tip_indeces(node, 'left') == [0]
                assert get_tip_indeces(node, 'right') == [1, 2]
            if i == 2:
                assert get_tip_indeces(node, 'left') == [4]
                assert get_tip_indeces(node, 'right') == [5]
            if i == 3:
                assert get_tip_indeces(node, 'left') == [4, 5]
                assert get_tip_indeces(node, 'right') == [6]
            if i == 4:
                assert get_tip_indeces(node, 'left') == [3]
                assert get_tip_indeces(node, 'right') == [4, 5, 6]
            if i == 5:
                assert get_tip_indeces(node, 'left') == [0, 1, 2]
                assert get_tip_indeces(node, 'right') == [3, 4, 5, 6]

    def test_get_lstar(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            if i == 0:
                expected = np.matrix([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                tip_inds0 = get_tip_indeces(node, 'left')
                nontip_inds0 = get_tip_indeces(node, 'left')
                lstar = get_lstar(node.children[0], tip_inds0,
                                  nontip_inds0, lilmat)
                assert (lstar.T == expected).all()

                expected = np.matrix([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
                tip_inds1 = get_tip_indeces(node, 'right')
                nontip_inds1 = get_tip_indeces(node, 'right')
                lstar = get_lstar(node.children[1], tip_inds1,
                                  nontip_inds1, lilmat)
                assert (lstar.T == expected).all()

            if i == 4:

                expected = np.matrix([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
                tip_inds0 = get_tip_indeces(node, 'left')
                nontip_inds0 = get_tip_indeces(node, 'left')
                lstar = get_lstar(node.children[0], tip_inds0,
                                  nontip_inds0, lilmat)
                assert np.isclose(lstar.T, expected).all()

                expected = [[0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0],
                            [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0],
                            [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0]]
                tip_inds1 = get_tip_indeces(node, 'right')
                nontip_inds1 = get_tip_indeces(node, 'right')
                lstar = get_lstar(node.children[1], tip_inds1,
                                  nontip_inds1, lilmat)
                assert np.isclose(lstar.T, expected).all()

    def test_get_L(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            if i == 0:
                tip_inds0 = get_tip_indeces(node, 'left')
                tip_inds1 = get_tip_indeces(node, 'right')
                left, right = get_L(tip_inds0, tip_inds1)
                assert np.isclose(left, np.sqrt(1/2))
                assert np.isclose(right, -np.sqrt(1/2))

            if i == 3:
                tip_inds0 = get_tip_indeces(node, 'left')
                tip_inds1 = get_tip_indeces(node, 'right')
                left, right = get_L(tip_inds0, tip_inds1)
                assert np.isclose(left, np.sqrt(1/6))
                assert np.isclose(right, -np.sqrt(2/3))

    def test_get_haarvec(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            tip_inds0 = get_tip_indeces(node, 'left')
            tip_inds1 = get_tip_indeces(node, 'right')
            left, right = get_L(tip_inds0, tip_inds1)
            ntips = node.ntips_in_tree
            h = get_haarvec(tip_inds0, tip_inds1, left, right, ntips)
            assert np.isclose(sum(h), 0.0)  # Sum of haar vectors = 0

            if i == 0:
                expected = np.array([[0.0,
                                      0.70710678,
                                      -0.70710678,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0]])
                assert np.isclose(h.T, expected).all()

            if i == 5:
                expected = np.array([[0.43643578,
                                    0.43643578,
                                    0.43643578,
                                    -0.32732684,
                                    -0.32732684,
                                    -0.32732684,
                                    -0.32732684]])
                assert np.isclose(h.T, expected).all()

    def test_handle_case(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            if i == 0:
                lilmat, shl = handle_both(node, lilmat, shl, i)
                val = 1/np.sqrt(2)
                expected_shl = [[0, val, -val, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]]

                expected_lilmat = [[0, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]]

                expected_shl = scipy.sparse.lil_matrix(expected_shl)
                expected_shl = expected_shl.todense()
                expected_lilmat = scipy.sparse.lil_matrix(expected_lilmat)
                expected_lilmat = expected_lilmat.todense()

                assert np.isclose(expected_shl, shl.todense()).all()
                assert np.isclose(expected_lilmat, lilmat.todense()).all()

            if i == 1:  # left
                lilmat, shl = handle_left(node, lilmat, shl, i)
                val1 = np.sqrt(2/3)
                expected_shl = [[0, val, -val, 0, 0, 0, 0],
                                [val1, -val1/2, -val1/2, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]]

                expected_lilmat = [[0, 1, 1, 0, 0, 0, 0],
                                   [1, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]]

                expected_shl = scipy.sparse.lil_matrix(expected_shl)
                expected_shl = expected_shl.todense()
                expected_lilmat = scipy.sparse.lil_matrix(expected_lilmat)
                expected_lilmat = expected_lilmat.todense()

                assert np.isclose(expected_shl, shl.todense()).all()
                assert np.isclose(expected_lilmat, lilmat.todense()).all()

            if i == 2:
                lilmat, shl = handle_both(node, lilmat, shl, i)

            if i == 3:  # right
                lilmat, shl = handle_right(node, lilmat, shl, i)
                expected_shl = [[0, val, -val, 0, 0, 0, 0],
                                [val1, -val1/2, -val1/2, 0, 0, 0, 0],
                                [0, 0, 0, 0, val, -val, 0],
                                [0, 0, 0, 0, val1/2, val1/2, -val1],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]]

                expected_lilmat = [[0, 1, 1, 0, 0, 0, 0],
                                   [1, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1, 0],
                                   [0, 0, 0, 0, 3, 3, 1],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]]

                expected_shl = scipy.sparse.lil_matrix(expected_shl)
                expected_shl = expected_shl.todense()
                expected_lilmat = scipy.sparse.lil_matrix(expected_lilmat)
                expected_lilmat = expected_lilmat.todense()

                assert np.isclose(expected_shl, shl.todense()).all()
                assert np.isclose(expected_lilmat, lilmat.todense()).all()

            if i == 4:
                lilmat, shl = handle_left(node, lilmat, shl, i)

            if i == 5:
                lilmat, shl = handle_neither(node, lilmat, shl, i)
                val3 = np.sqrt(3/4)  # 0.8660254038
                val4 = -np.sqrt(1/12)  # 0.2886751346
                val5 = np.sqrt(4/21)  # 0.4364357805
                val6 = -np.sqrt(3/28)  # 0.3273268354

                expected_shl = [[0, val, -val, 0, 0, 0, 0],
                                [val1, -val1/2, -val1/2, 0, 0, 0, 0],
                                [0, 0, 0, 0, val, -val, 0],
                                [0, 0, 0, 0, val1/2, val1/2, -val1],
                                [0, 0, 0, val3, val4, val4, val4],
                                [val5, val5, val5, val6, val6, val6, val6],
                                [0, 0, 0, 0, 0, 0, 0]]

                expected_lilmat = [[0, 1, 1, 0, 0, 0, 0],
                                   [1, 3, 3, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1, 0],
                                   [0, 0, 0, 0, 3, 3, 1],
                                   [0, 0, 0, 1, 6, 6, 4],
                                   [4, 6, 6, 5, 10, 10, 8],
                                   [0, 0, 0, 0, 0, 0, 0]]

                expected_shl = scipy.sparse.lil_matrix(expected_shl)
                expected_shl = expected_shl.todense()
                expected_lilmat = scipy.sparse.lil_matrix(expected_lilmat)
                expected_lilmat = expected_lilmat.todense()

                assert np.isclose(expected_shl, shl.todense()).all()
                assert np.isclose(expected_lilmat, lilmat.todense()).all()

    def test_get_lilmat_and_shl(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):
            if i == 0:
                nontip_inds0 = None
                nontip_inds1 = None
                tip_inds0 = get_tip_indeces(node, 'left')
                tip_inds1 = get_tip_indeces(node, 'right')
                left, right = get_L(tip_inds0, tip_inds1)
                ntips = node.ntips_in_tree
                h = get_haarvec(tip_inds0, tip_inds1, left, right, ntips)
                assert (np.isclose(sum(h), 0.0)).all()  # Sum of haar = 0

                lilmat, shl = get_lilmat_and_shl(node, nontip_inds0,
                                                 nontip_inds1,
                                                 lilmat, shl, i)

                val = 1/np.sqrt(2)
                expected_shl = [[0, val, -val, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]]

                expected_lilmat = [[0, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]]

                expected_shl = scipy.sparse.lil_matrix(expected_shl)
                expected_shl = expected_shl.todense()
                expected_lilmat = scipy.sparse.lil_matrix(expected_lilmat)
                expected_lilmat = expected_lilmat.todense()
                shl = shl.todense()
                lilmat = lilmat.todense()

                assert np.isclose(expected_shl, shl).all()
                assert np.isclose(expected_lilmat, lilmat).all()

    def test_sparsify(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        lilmat, shl = sparsify(t2)

        expected_lilmat = [[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                           [1.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 1.0],
                           [0.0, 0.0, 0.0, 1.0, 6.0, 6.0, 4.0],
                           [4.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 8.0]]

        x1 = [0.0, 0.70710678, -0.70710678, 0.0, 0.0, 0.0, 0.0]
        x2 = [0.81649658, -0.40824829, -0.40824829, 0.0,  0.0, 0.0, 0.0]
        x3 = [0.0, 0.0, 0.0, 0.0, 0.70710678, -0.70710678, 0.0]
        x4 = [0.0, 0.0, 0.0, 0.0, 0.40824829, 0.40824829, -0.81649658]
        x5 = [0.0, 0.0, 0.0,
              0.8660254, -0.28867513, -0.28867513, -0.28867513]
        x6 = [0.57735027,  0.57735027,  0.57735027,  0.0, 0.0, 0.0, 0.0]
        x7 = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        expected_shl = [x1, x2, x3, x4, x5, x6, x7]

        expected_lilmat = scipy.sparse.lil_matrix(expected_lilmat)
        expected_shl = scipy.sparse.lil_matrix(expected_shl)

        assert np.isclose(expected_lilmat.todense(), lilmat.todense()).all()
        assert np.isclose(expected_shl.todense(), shl.todense()).all()

        assert True
        # TODO

    def test_get_lambda(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        lilmat, shl = sparsify(t2)

        for i in range(lilmat.shape[0]):

            if i == 0:
                x = get_lambda(lilmat, shl, i)
                assert np.isclose(x[0, 0], 1)
            if i == 1:
                x = get_lambda(lilmat, shl, i)
                assert np.isclose(x[0, 0], 5/3)
            if i == 2:
                x = get_lambda(lilmat, shl, i)
                assert np.isclose(x[0, 0], 1)
            if i == 3:
                x = get_lambda(lilmat, shl, i)
                assert np.isclose(x[0, 0], 5/3)
            if i == 4:
                x = get_lambda(lilmat, shl, i)
                assert np.isclose(x[0, 0], 25/12)
            if i == 5:
                x = get_lambda(lilmat, shl, i)
                assert np.isclose(x[0, 0], 16/3)
            if i == 6:
                x = get_lambda(lilmat, shl, i)
                assert np.isclose(x[0, 0], 33/4)

    def test_get_lambdas(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        lilmat, shl = sparsify(t2)

        diagonal = get_lambdas(lilmat, shl)
        diagonal_expected = [1.0, 5/3, 1, 5/3, 25/12, 16/3, 33/4]

        assert np.isclose(diagonal, diagonal_expected).all()

    def test_match_to_tree(self):

        before_exp = ['o1', 'o2', 'o7', 'o3', 'o4', 'o5', 'o6', 'o8']
        before_exp = np.array(before_exp)
        before_obs = self.table.ids(axis='observation')
        assert (before_obs == before_exp).all()

        table, _, _ = match_to_tree(self.table, self.tree)
        table = table.todense()

        after_exp = np.array(
            [[0.0, 0.07142857, 0.0, 0.44444444, 0.05555556],
             [0.0, 0.07142857, 0.0, 0.05555556, 0.05555556],
             [0.18181818, 0.14285714, 0.2, 0.05555556, 0.44444444],
             [0.27272727, 0.21428571, 0.350, 0.0, 0.0],
             [0.36363636, 0.35714286, 0.05, 0.11111111, 0.11111111],
             [0.0, 0.07142857, 0.1, 0.16666667, 0.16666667],
             [0.0, 0.0, 0.3, 0.16666667, 0.16666667]]
        )

        assert np.isclose(table, after_exp).all()

    def test_create_branching_tree(self):

        _, t2, _ = match_to_tree(self.table, self.tree)
        lilmat, shl = sparsify(t2)

        lilmat_exp = np.array([[4, 6, 6, 0, 0, 0, 0],
                               [0, 0, 0, 5, 10, 10, 8]])
        val = np.sqrt(1/3)
        shl_exp = np.array([[val, val, val, 0, 0, 0, 0],
                            [0, 0, 0, .5, .5, .5, .5]])

        assert np.isclose(lilmat_exp, lilmat.todense()[-2:]).all()
        assert np.isclose(shl_exp, shl.todense()[-2:]).all()

    def test_compute_haar_dist(self):

        exp_D = [
            [0.0, 0.20732524, 0.57374449, 0.93465657, 0.7966299],
            [0.20732524, 0.0, 0.61110055, 0.78242334, 0.6983557],
            [0.57374449, 0.61110055, 0.0, 1.00183645, 0.86363033],
            [0.93465657, 0.78242334, 1.00183645, 0.0, 0.67357531],
            [0.7966299, 0.6983557, 0.86363033, 0.67357531, 0.0]
        ]

        exp_modmags = [
            [-0.12856487, -0.0958266, 0.25712974, 0.19165319] +
            [0.18939394, 0.24242424, 0.91390769],
            [-0.05050763, -0.03764616, 0.20203051, 0.22587698] +
            [0.08928571, 0.38095238, 0.92323328],
            [-0.14142136, -0.10540926, -0.03535534, -0.23717082] +
            [0.25, 0.26666667, 1.14891253],
            [0.0, 0.40992488, -0.03928371, -0.02928035] +
            [-0.18518519, 0.74074074, 0.63828474],
            [-0.27498597, -0.20496244, -0.03928371, -0.02928035] +
            [-0.18518519, 0.74074074, 0.63828474]
        ]

        exp_D = np.array(exp_D)
        exp_modmags = np.array(exp_modmags)

        table, tree, ids = match_to_tree(self.table, self.tree)
        lilmat, shl = sparsify(tree)
        diagonal = get_lambdas(lilmat, shl)
        D, modmags = compute_haar_dist(table, shl, diagonal)
        modmags = modmags.todense()

        assert np.isclose(exp_D, D).all()
        assert np.isclose(exp_modmags, modmags).all()

    def test_format_tree(self):

        # TODO
        assert True
