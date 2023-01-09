import tempfile
import os

from unittest import TestCase, main
from os.path import dirname, abspath, join
from inspect import currentframe, getfile
from q2_haarlikedist._methods import *


class TestSparsify(TestCase):

    def setUp(self):
        self.ntips = 7
        self.tree_file = 'small-tree.tree'

        self.lilmat = scipy.sparse.lil_matrix((self.ntips, self.ntips))
        self.shl = scipy.sparse.lil_matrix((self.ntips, self.ntips))


    def test_get_tree_from_file(self):

        t2 = get_tree_from_file(self.tree_file)
        assert(t2.count() == 13)


    def test_initiate_values(self):

        expected = scipy.sparse.lil_matrix((self.ntips, self.ntips))

        t2 = get_tree_from_file(self.tree_file)
        t2, lilmat, shl = initiate_values(t2)

        assert((lilmat != expected).nnz == 0)
        assert((shl != expected).nnz == 0)

        for i, node in enumerate(t2.non_tips(include_self=True)):
            assert(node.postorder_pos == i)

        for i, node in enumerate(t2.postorder(include_self=True)):

            assert(node.ntips_in_tree == 7)
            
            if i == 0:
                assert(node.tip_names == [0])
            if i == 1:
                assert(node.tip_names == [1])
            if i == 2:
                assert(node.tip_names == [2])
            if i == 3:
                assert(node.tip_names == [1, 2])
            if i == 4:
                assert(node.tip_names == [0, 1, 2])
            if i == 11:
                assert(node.tip_names == [3, 4, 5, 6])
            if i == 12:
                assert(node.tip_names == [0, 1, 2, 3, 4, 5, 6])

    def test_get_case(self):

        t2 = get_tree_from_file(self.tree_file)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            if i == 0:
                assert(get_case(node) == 'both')
            if i == 1:
                assert(get_case(node) == 'left')
            if i == 2:
                assert(get_case(node) == 'both')
            if i == 3:
                assert(get_case(node) == 'right')
            if i == 4:
                assert(get_case(node) == 'left')
            if i == 5:
                assert(get_case(node) == 'neither')


    def test_get_nontip_index(self):

        t2 = get_tree_from_file(self.tree_file)
        t2, lilmat, shl = initiate_values(t2)
        
        for i, node in enumerate(t2.non_tips(include_self=True)):
            
            if i == 0: # case = both, neither children nontips
                continue 
            if i == 1: # case = left, right child is nontipi
                assert(get_nontip_index(node, 'right') == 0)
            if i == 2: # case = both
                continue
            if i == 3: # case = right
                assert(get_nontip_index(node, 'left') == 2)
            if i == 4: # case = left
                assert(get_nontip_index(node, 'right') == 3)
            if i == 5: # case = neither
                assert(get_nontip_index(node, 'left') == 1)
                assert(get_nontip_index(node, 'right') == 4)

    def test_get_tip_indeces(self):

        t2 = get_tree_from_file(self.tree_file)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            if i == 0:
                assert(get_tip_indeces(node, 'left') == [1])
                assert(get_tip_indeces(node, 'right') == [2])
            if i == 1:
                assert(get_tip_indeces(node, 'left') == [0])
                assert(get_tip_indeces(node, 'right') == [1, 2])
            if i == 2:
                assert(get_tip_indeces(node, 'left') == [4])
                assert(get_tip_indeces(node, 'right') == [5])
            if i == 3:
                assert(get_tip_indeces(node, 'left') == [4, 5])
                assert(get_tip_indeces(node, 'right') == [6])
            if i == 4:
                assert(get_tip_indeces(node, 'left') == [3])
                assert(get_tip_indeces(node, 'right') == [4, 5, 6])
            if i == 5:
                assert(get_tip_indeces(node, 'left') == [0, 1, 2])
                assert(get_tip_indeces(node, 'right') == [3, 4, 5, 6])

    def test_get_lstar(self):

        t2 = get_tree_from_file(self.tree_file)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):
            
            if i == 0:
                expected = np.matrix([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                tip_inds0 = get_tip_indeces(node, 'left')
                nontip_inds0 = get_tip_indeces(node, 'left')
                l = get_lstar(node.children[0], tip_inds0, nontip_inds0, lilmat)
                assert((l.T == expected).all())

                expected = np.matrix([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
                tip_inds1 = get_tip_indeces(node, 'right')
                nontip_inds1 = get_tip_indeces(node, 'right')
                l = get_lstar(node.children[1], tip_inds1, nontip_inds1, lilmat)
                assert((l.T == expected).all())

            if i == 4:

                expected = np.matrix([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
                tip_inds0 = get_tip_indeces(node, 'left')
                nontip_inds0 = get_tip_indeces(node, 'left')
                l = get_lstar(node.children[0], tip_inds0, nontip_inds0, lilmat)
                assert(np.isclose(l.T, expected).all())

                expected = [[0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0],
                            [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0],
                            [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0]]
                tip_inds1 = get_tip_indeces(node, 'right')
                nontip_inds1 = get_tip_indeces(node, 'right')
                l = get_lstar(node.children[1], tip_inds1, nontip_inds1, lilmat)
                assert(np.isclose(l.T, expected).all())

    def test_get_L(self):

        t2 = get_tree_from_file(self.tree_file)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):

            if i == 0:
                tip_inds0 = get_tip_indeces(node, 'left')
                tip_inds1 = get_tip_indeces(node, 'right')
                left, right = get_L(tip_inds0, tip_inds1)
                assert(np.isclose(left, np.sqrt(1/2)))
                assert(np.isclose(right, -np.sqrt(1/2)))

            if i == 3:
                tip_inds0 = get_tip_indeces(node, 'left')
                tip_inds1 = get_tip_indeces(node, 'right')
                left, right = get_L(tip_inds0, tip_inds1)
                assert(np.isclose(left, np.sqrt(1/6)))
                assert(np.isclose(right, -np.sqrt(2/3)))

    def test_get_haarvec(self):

        t2 = get_tree_from_file(self.tree_file)
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):
            
            tip_inds0 = get_tip_indeces(node, 'left')
            tip_inds1 = get_tip_indeces(node, 'right')
            left, right = get_L(tip_inds0, tip_inds1)
            h = get_haarvec(tip_inds0, tip_inds1, left, right, node.ntips_in_tree)
            assert(np.isclose(sum(h), 0.0)) # The sum of haar vectors should be 0

            if i == 0:
                expected = np.array([[0.0, 
                                    0.70710678,
                                    -0.70710678,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0]])
                assert(np.isclose(h.T, expected).all())

            if i == 5:
                expected = np.array([[0.43643578,
                                    0.43643578,
                                    0.43643578,
                                    -0.32732684,
                                    -0.32732684,
                                    -0.32732684,
                                    -0.32732684]])
                assert(np.isclose(h.T, expected).all())

    def test_get_lilmat_and_shl(self):

        t2 = get_tree_from_file('small-tree.tree')
        t2, lilmat, shl = initiate_values(t2)

        for i, node in enumerate(t2.non_tips(include_self=True)):
            if i == 0:        
                nontip_inds0 = None
                nontip_inds1 = None
                tip_inds0 = get_tip_indeces(node, 'left')
                tip_inds1 = get_tip_indeces(node, 'right')
                left, right = get_L(tip_inds0, tip_inds1)
                h = get_haarvec(tip_inds0, tip_inds1, left, right, node.ntips_in_tree)
                assert((np.isclose(sum(h), 0.0)).all()) # The sum of haar vectors should be 0

                lilmat, shl = get_lilmat_and_shl(node, nontip_inds0, nontip_inds1, lilmat, shl, i)

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
                expected_lilmat = scipy.sparse.lil_matrix(expected_lilmat)  

                assert(np.isclose(expected_shl.todense(), shl.todense()).all())
                assert(np.isclose(expected_lilmat.todense(), lilmat.todense()).all())

    def test_handle_case(self):

        t2 = get_tree_from_file('small-tree.tree')
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
                expected_lilmat = scipy.sparse.lil_matrix(expected_lilmat)  

                assert(np.isclose(expected_shl.todense(), shl.todense()).all())
                assert(np.isclose(expected_lilmat.todense(), lilmat.todense()).all())

    def test_sparsify(self):

        lilmat, shl = sparsify(self.tree_file)

        expected_lilmat = [[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                           [1.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 1.0],
                           [0.0, 0.0, 0.0, 1.0, 6.0, 6.0, 4.0],
                           [4.0, 6.0, 6.0, 5.0, 10.0, 10.0, 8.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        expected_shl = [[0.0, 0.70710678, -0.70710678, 0.0, 0.0, 0.0, 0.0],
                        [0.81649658, -0.40824829, -0.40824829, 0.0,  0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.70710678,-0.70710678, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.40824829, 0.40824829, -0.81649658],
                        [0.0, 0.0, 0.0, 0.8660254, -0.28867513, -0.28867513, -0.28867513],
                        [0.43643578, 0.43643578, 0.43643578, -0.32732684, -0.32732684, -0.32732684, -0.32732684],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        expected_lilmat = scipy.sparse.lil_matrix(expected_lilmat)  
        expected_shl = scipy.sparse.lil_matrix(expected_shl)  

        assert(np.isclose(expected_lilmat.todense(), lilmat.todense()).all())
        assert(np.isclose(expected_shl.todense(), shl.todense()).all())