# ----------------------------------------------------------------------------
# Copyright (c) 2022--, haar-like-dist development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import skbio
from skbio import read
from skbio.tree import TreeNode
from skbio.stats.distance import DistanceMatrix
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import biom
from qiime2 import CategoricalMetadataColumn as Column


def get_tree_from_file(tree_file):
    """ Used only for testing and development. """

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
    shl = lil_matrix((ntips, ntips))
    lilmat = lil_matrix((ntips, ntips))

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
    """ Returns all left or right tip children indeces. """

    ind = 0 if side == 'left' else 1
    return node.children[ind].tip_names


def get_lstar(child, tip_inds, nontip_inds, lilmat):
    """ Returns lstar, which is the cumulative modified branch lengths
        from each internal node to each tip node (entries in lstar). """

    if nontip_inds is None:
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

    nontip_inds0 = None  # Doesn't have a row index in lilmat
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


def create_branching_tree(t2, lilmat, shl):
    """ Returns lilmat, shl represented as two branching trees. """
    mastersplit = t2.children

    ntips0 = len([x for x in mastersplit[0].tips()])
    ntips1 = len([x for x in mastersplit[1].tips()])

    values0 = np.repeat(1/np.sqrt(ntips0), ntips0)
    zeros0 = np.repeat(0, ntips1)

    values1 = np.repeat(1/np.sqrt(ntips1), ntips1)
    zeros1 = np.repeat(0, ntips0)

    shl[-2] = np.hstack((values0, zeros0))
    shl[-1] = np.hstack((zeros1, values1))

    lilmat[-1] = np.copy(lilmat[-2].todense())
    lilmat[-2, ntips0:] = 0
    lilmat[-1, :ntips0] = 0

    return lilmat, shl


def sparsify(t2):

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

    lilmat, shl = create_branching_tree(t2, lilmat, shl)

    return lilmat, shl


def get_lambda(lilmat, shl, i):
    """ Computes lambda for each internal node.
        Lambda is the lilmat entry for i * shl entry ^2. """

    lstar = lilmat[i].todense().T
    phi = shl[i].todense()
    phi2 = np.multiply(phi, phi)
    lambd = np.dot(phi2, lstar)
    return lambd


def get_lambdas(lilmat, shl):
    """ Computes all lambdas. """

    n = lilmat.shape[0]
    data = [get_lambda(lilmat, shl, i) for i in range(n)]
    data = np.array(data).T
    diagonal = data[0][0]
    return diagonal


def match_to_tree(table, tree):
    """ Returns aligned data in biom format.
        data_file must be a biom table. """

    table = table.norm()
    table, tree = table.align_tree(tree)
    ids = table.ids()
    table_matrix = table.matrix_data.tocsr()
    return table, tree, ids, table_matrix


def compute_haar_dist(table, shl, diagonal):

    # columns are samples
    nsamples = table.shape[1]
    diagonal_mat = csr_matrix([diagonal] * nsamples)
    diagonal_mat_sqrt = np.sqrt(diagonal_mat)
    mags = shl @ table
    modmags = mags.T.multiply(diagonal_mat_sqrt)
    ones = csr_matrix(np.ones((nsamples, 1)))  # 1's csr_matrix for casting

    D = np.zeros((nsamples, nsamples))
    for i in range(nsamples):
        a = modmags - modmags[i, :].multiply(ones)
        b = csr_matrix.power(a, 2)
        c = csr_matrix.sum(b, axis=1)
        d = np.sqrt(c)
        D[i, :] = csr_matrix(d.T)

    D = D + D.T
    return D, modmags


def format_tree(tree, modmags):
    """ Formats tree for output.
        Saves number of times node is most significant
        as a node name and returns tree. """

    nontips = [node for node in tree.non_tips(include_self=True)]
    n = np.shape(modmags)[0]
    m = np.shape(modmags)[1]
    node_weights = np.zeros(m)
    for i in range(n):
        for j in range(i):
            if i != j:
                diff = modmags[i] - modmags[j]
                diff = [np.abs(d) for d in diff.todense()]
                d = np.array(diff)[0][0]
                node_weights += d

    # node_weights = [np.log(x) for x in node_weights]
    node_weights = node_weights / sum(node_weights)
    for i, node in enumerate(nontips):
        node.length = np.round(node_weights[i], 4)
        node.name = str(node.length)

    nontips[-1].name = '0'

    return tree


def format_tree_meta(tree, modmags, metadata_col, metadata_val):
    """ Formats tree for output.
        Saves number of times node is most significant
        as a node name and returns tree. """

    nontips = [node for node in tree.non_tips(include_self=True)]
    for node in nontips:
        node.max_modmag = 0

    m = np.shape(modmags)[1]
    node_weights = np.zeros(m)

    # Construct the indices of the two groups to prepare
    col = metadata_col.to_series()
    vals = [x[0] for x in col.values]
    i_ind = [i for i, x in enumerate(vals) if x == metadata_val]
    j_ind = [x for x in range(m) if x not in i_ind]

    for i in i_ind:
        for j in j_ind:
            diff = modmags[i] - modmags[j]
            diff = [np.abs(d) for d in diff.todense()]
            d = np.array(diff)[0][0]
            node_weights += d

    for i, node in enumerate(nontips):
        node.length = np.round(node_weights[i], 4)
        node.name = str(node.length)

    nontips[-1].name = '0'

    return tree


def haar_like_dist(table: biom.Table,
                   phylogeny: skbio.TreeNode,
                   group_column: Column = None,
                   group_value: str = None) \
                   -> (DistanceMatrix, skbio.TreeNode,
                       biom.Table):
    """ Returns D, tree, mm. Distance matrix and significance.
        Returns distance matrix and formatted tree.
        This now returns modmags as a biom table, which
        can be thought of as a differentially encoded
        feature table. """

    table, tree, ids, table_matrix = match_to_tree(table, phylogeny)
    lilmat, shl = sparsify(tree)
    diagonal = get_lambdas(lilmat, shl)
    D, modmags = compute_haar_dist(table_matrix, shl, diagonal)
    D = DistanceMatrix(D, ids)
    if group_column is not None and group_value is not None:
        tree = format_tree_meta(tree, modmags, group_column, group_value)
    else:
        tree = format_tree(tree, modmags)
    mm = biom.Table(modmags, observation_ids=[], sample_ids=[])

    return D, tree, mm
