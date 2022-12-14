# ----------------------------------------------------------------------------
# Copyright (c) 2022--, convex-hull development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import scipy
import skbio
import numpy as np 
from io import StringIO
from skbio import read

def get_tree_from_file(tree_file):

    f = open(tree_file, 'r')
    t2 = read(f, format='newick', into=TreeNode)
    f.close()
    return t2


def initiate_values(t2):
    """ Returns t2, shl, lilmat.
    shl: Matrix that tracks all haar-like vectors
    SHL = SPARSE HAAR LIKE 
    Rows = internal nodes; Columns = tips with values of :
       -- 0 if tip is not a descendant from internal node
       -- c_L if tip is a left descendant from internal node
       -- c_R if tip is a right descendant from internal node
    c_L and c_R are calculated in the get_L function.

    lilmat: Matrix that tracks l-star values. n tips x n nontips (so square)
    Entry i, j represents weighted path distance from internal node i to tip j
    This is why it is computed iteratively for internal nodes 
    with non-tip descendants. """

    ntips = len([x for x in t2.tips()])
    shl = scipy.sparse.lil_matrix((ntips, ntips))
    lilmat = scipy.sparse.lil_matrix((ntips, ntips))

    tip_index = {t: i for i, t in enumerate(t2.tips())}

    for i, node in enumerate(t2.non_tips(include_self=True)):
        node.postorder_pos = i

    for node in t2.postorder(include_self=True):
        node.ntips_in_tree = ntips
        node.tip_names = [tip_index[x] for x in node.tips()]

        if node.is_tip():
            node.tip_names = [tip_index[node]]

    return t2, shl, lilmat


def get_case(node):
    """ Handle different types of children differently. 
        4 possible child layouts; returns string describing the case.
        Neither => neither child is a tip etc. """

    left_is_tip = not node.children[0].has_children()
    right_is_tip = not node.children[1].has_children()

    if not left_is_tip and not right_is_tip:
        return 'neither'

    if left_is_tip and right_is_tip:
        return 'both'

    if left_is_tip and not right_is_tip:
        return 'left'

    if not left_is_tip and right_is_tip:
        return 'right'
        

def get_nontip_index(node, side):
    """ Returns a single nontip index of the node's left or right child. 
        Only valid for a nontip node and nontip child. """

    ind = 0 if side == 'left' else 1
    return node.children[ind].postorder_pos


def get_tip_indeces(node, side):
    """ Returns all left or right tip children. """

    ind = 0 if side == 'left' else 1
    return node.children[ind].tip_names


def get_lstar(child, tip_inds, nontip_inds, lilmat):
    """ Returns lstar, which is the cumulative modified branch lengths
        from each internal node to each tip node (entries in lstar). """

    if nontip_inds == None:
        ntips = child.ntips_in_tree
        lstar = np.zeros((ntips, 1))
    else:
        lstar = lilmat[nontip_inds].todense().T

    lstar[tip_inds] = lstar[tip_inds] + len(tip_inds) * child.length

    return lstar


def get_L(tip_inds0, tip_inds1):
    """ Returns the values that each left and right side of the haarvec
        should be set to. These are the heights of the wavelets """

    L0 = len(tip_inds0)
    L1 = len(tip_inds1)

    left = np.sqrt(L1/(L0*(L0+L1)))
    right = - np.sqrt(L0/(L1*(L0+L1)))

    return left, right


def get_haarvec(tip_inds0, tip_inds1, left, right, ntips):
    """ Returns the haarlike wavelets. """

    haarvec = np.zeros((ntips, 1))

    haarvec[tip_inds0] = left
    haarvec[tip_inds1] = right

    return haarvec


def get_lilmat_and_shl(node, nontip_inds0, nontip_inds1, lilmat, shl, i):
    """ Processes the nontip indeces and updates lilmat and shl
        matrices. Is performed in each of the 4 cases. """

    tip_inds0 = get_tip_indeces(node, 'left')
    tip_inds1 = get_tip_indeces(node, 'right')

    lstar0 = get_lstar(node.children[0], tip_inds0, nontip_inds0, lilmat)
    lstar1 = get_lstar(node.children[1], tip_inds1, nontip_inds1, lilmat)

    lilmat[i] = lstar0.T + lstar1.T

    ntips = node.ntips_in_tree
    left, right = get_L(tip_inds0, tip_inds1)
    haarvec = get_haarvec(tip_inds0, tip_inds1, left, right, ntips)
    shl[i] = haarvec.T

    return lilmat, shl


def handle_neither(node, lilmat, shl, i):
    """ Case where neither child is a tip. """

    # rows of lilmat indexed by nontips - select row to modify
    nontip_inds0 = get_nontip_index(node, 'left')
    nontip_inds1 = get_nontip_index(node, 'right')

    return get_lilmat_and_shl(node, nontip_inds0, nontip_inds1, lilmat, shl, i)


def handle_left(node, lilmat, shl, i):
    """ Case where neither left is a tip. """

    nontip_inds0 = None # Doesn't have a row index in lilmat
    nontip_inds1 = get_nontip_index(node, 'right')

    return get_lilmat_and_shl(node, nontip_inds0, nontip_inds1, lilmat, shl, i)


def handle_right(node, lilmat, shl, i):
    """ Case where right child is a tip. """

    nontip_inds0 = get_nontip_index(node, 'left')
    nontip_inds1 = None

    return get_lilmat_and_shl(node, nontip_inds0, nontip_inds1, lilmat, shl, i)


def handle_both(node, lilmat, shl, i):
    """ Case where both children are tips. """

    nontip_inds0 = None
    nontip_inds1 = None

    return get_lilmat_and_shl(node, nontip_inds0, nontip_inds1, lilmat, shl, i)


def sparsify(tree_file):

    t2 = get_tree_from_file(tree_file)
    t2, shl, lilmat = initiate_values(t2)

    traversal = t2.non_tips(include_self=True)
    for i, node in enumerate(traversal):
        case = get_case(node)

        if case == 'neither':
            lilmat, shl = handle_neither(node, lilmat, shl, i)

        elif case == 'both':
            lilmat, shl = handle_both(node, lilmat, shl, i)
            
        elif case == 'left':
            lilmat, shl = handle_left(node, lilmat, shl, i)

        elif case == 'right':
            lilmat, shl = handle_right(node, lilmat, shl, i)

    return lilmat, shl