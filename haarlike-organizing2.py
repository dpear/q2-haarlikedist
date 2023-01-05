import scipy
import skbio
import numpy as np 
from io import StringIO
from skbio import read

tree_string = '((1:1.0,(2:1.0,3:1.0):1.0):1.0,(4:1.0,((5:1.0,6:1.0):1.0,7:1.0):1.0):1.0);'
tree_string_io = StringIO(tree_string)
t2 = read(tree_string_io, format="newick", into=TreeNode)
all_leaves = t2.tips()
numleaves = len([x for x in t2.tips()])

# Matrix that tracks all haar-like vectors
##### SHL = SPARSE HAAR LIKE #########
# Rows = internal nodes; Columns = tips with values of :
#    -- 0 if tip is not a descendant from internal node
#    -- c_L if tip is a left descendant from internal node
#    -- c_R if tip is a right descendant from internal node
# c_L and c_R are calculated in the get_L function.
shl = scipy.sparse.lil_matrix((numleaves, numleaves))

# Matrix that tracks l-star values. n tips x n nontips (so square)
# Entry i, j represents weighted path distance from internal node i to tip j
# This is why it is computed iteratively for internal nodes 
#       with non-tip descendants.
lilmat = scipy.sparse.lil_matrix((numleaves, numleaves))


###############################################  

def get_case(node):
    """ Handle different types of children differently. 
        4 possible child layouts; returns string describing the case.
        Neither => neither child is a tip etc.
    """

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


###############################################
tip_index = {t: i for i, t in enumerate(t2.tips())}
tip_children = {node: [x for x in node.tips()] for node in traversal}

###############################################

def get_nontip_index(node, side):
    """ Returns a single nontip index 
        of the node's left or right child.
    """

    ind = 0 if side == 'left' else 1
    return nontip_index[child[ind]]


def get_tip_indeces(node, side):
    """ Returns all left or right tip children. """

    ind = 0 if side == 'left' else 1
    child = node.children[ind]

    return tip_children[child]


def get_lstar(child, tip_inds, nontip_inds, lilmat):
    """ Returns lstar, which is the cumulative modified branch lengths
        from each internal node to each tip node (entries in lstar). """

    if nontip_inds == None:
        lstar = np.zeros((numleaves, 1))
    else:
        lstar = lilmat[nontip_inds].todense()

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


def get_haarvec(tip_inds0, tip_inds1, left, right):
    """ Returns the haarlike wavelets. """

    haarvec = np.zeros((numleaves, 1))

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

    left, right = get_L(tip_inds0, tip_inds1)
    haarvec = get_haarvec(tip_inds0, tip_inds1, left, right)
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
    """ Case where both children is are tips. """

    nontip_inds0 = None
    nontip_inds1 = None

    return get_lilmat_and_shl(node, nontip_inds0, nontip_inds1, lilmat, shl, i)


###############################################
# Traverse non_tips in postorder
traversal = t2.non_tips(include_self=True)
for i,node in enumerate(traversal):

    case = get_case(node)

    if case == 'both_child_tip':
        lilmat, shl = handle_both(node, lilmat, shl, i)
        
    elif case == 'left_child_tip':
        lilmat, shl = handle_left(node, lilmat, shl, i)

    elif case == 'right_child_tip':
        lilmat, shl = handle_right(node, lilmat, shl, i)

    elif case == 'neither_child_tip':
        lilmat, shl = handle_neither(node, lilmat, shl, i)


###############################################
# RANDOM THINGS FOR TESTING

def get_node_from_ind(t2, node_ind):
""" Helper function for testing.
    Returns <node_ind>'th non-tip node. """    
    for i,n in enumerate(t2.non_tips(include_self=True)):
        if i == node_ind:
            return n

# Test get_case on tree from diagrams
def test_get_case():

    for i,node in enumerate(t2.non_tips(include_self=True)):

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

# Test handle_both:
def test_handle_both():
    shl = scipy.sparse.lil_matrix((numleaves, numleaves))
    lilmat = scipy.sparse.lil_matrix((numleaves, numleaves))
    shl, lilmat = handle_both(node, lilmat, shl, i)

    # shl ---> sparse haar-like
    val = 1/np.sqrt(2)
    expected_shl = [[0, val, -val, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]
    expected_shl = scipy.sparse.lil_matrix(expected_shl)       

    # use numpy method to see if they are all close
    np.allclose(expected_shl.A, shl.A)