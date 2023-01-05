import qiime2
import pandas as pd
from skbio import OrdinationResults
from warnings import warn
from qiime2 import Metadata
import scipy
import skbio
import qiime2
import itertools
import numpy as np 
import scipy.stats as ss
from qiime2.plugins import diversity, feature_table
try:
    from q2_types.tree import NewickFormat
except ImportError:
    NewickFormat = str
from skbio.tree import TreeNode

from io import StringIO
from skbio import read
# # Format = 1: flexible with internal node names


tree_string = '((1:1.0,(2:1.0,3:1.0):1.0):1.0,(4:1.0,((5:1.0,6:1.0):1.0,7:1.0):1.0):1.0);'
tree_string_io = StringIO(tree_string)
t2 = read(tree_string_io, format="newick", into=TreeNode)
all_leaves = t2.tips()
numleaves = len([x for x in t2.tips()])

# Matrix that tracks all haar-like vectors
# Has n-leaves columns and n-internal nodes rows
# Each row represents an internal node
# Each column represents a leaf: 
#    0 if leaf is not a descendant from internal node
#    c_L if leaf is a left descendant from internal node
#    c_R if leaf is a right descendant from internal node
sparsehaarlike = scipy.sparse.lil_matrix((numleaves, numleaves))

# Matrix that tracks l-star values
# Has n-internal-nodes rows and n-tips columns
# Entry i, j represents weighted path distance from internal node i to tip j
# This is why it is computed iteratively for internal nodes 
#       with non-tip descendants.
lilmat = scipy.sparse.lil_matrix((numleaves, numleaves)) # 2d vector of lstar0


###############################################  
def get_case(node):
    """ Handle different types of children
        differently. 4 possible child layouts and 
        returns string describing the case.
        Neither => neither child tip etc.
    """

    if not node.has_children():
        return 'null'

    if node.children[0].has_children():
        if node.children[1].has_children():
            return 'neither'
        else:
            return 'right'
    else:
        if node.children[1].has_children():
            return 'left'
        else:
            return 'both'


###############################################
tip_index = {t: i for i, t in enumerate(t2.tips())}
nontip_index = {t: i for i, t in enumerate(t2.non_tips())}

fn_tip = lambda n: [tip_index[n]] if n.is_tip() else []
fn_case = lambda n: [get_case(n)] if not n.is_tip() else []

# Cache attr stores something for each child I think
t2.cache_attr(fn_tip, 'tip_names')
t2.cache_attr(fn_case, 'case')

###############################################
def handle_both(node, lilmat, sparsehaarlike, i):
    """ Case that both a node's children are leaves.
        First get the indeces of each child's leaf children. 
        Then assign correct index of lstar to length of edge.
        This is actually length * 1 since there is 1 descendant 
        from a leaf, which is itself.

        Assign ith row in lilmat.

        Then assign the correct values to the ith row of
        sparsehaarlike matrix.
    """

    child = node.children

    index0 = child[0].tip_names
    index1 = child[1].tip_names
    lstar = np.zeros((numleaves, 1))

    lstar[index0] = child[0].length
    lstar[index1] = child[1].length
    lilmat[i] = np.transpose(lstar)
    haarvec = np.zeros((numleaves,1))
    haarvec[index0] = 1/np.sqrt(2)
    haarvec[index1] = -1/np.sqrt(2)
    sparsehaarlike[i] = np.transpose(haarvec)

    return lilmat, sparsehaarlike


def handle_left(node, lilmat, sparsehaarlike, i):
    """ Case that a node's left child is a leaf.
        Right child is not a leaf.
    """

    child = node.children

    index = nontip_index[child[1]]

    lstar0 = np.zeros((numleaves, 1))
    lstar1 = np.transpose(lilmat[index].todense())


    index0 = child[0].tip_names
    index1 = child[1].tip_names

    # lstar0[index0] = child[0].length
    lstar0[index0] = lstar0[index0] + len(child[0].tip_names) * child[0].length
    lstar1[index1] = lstar1[index1] + len(child[1].tip_names) * child[1].length # child[1].dist
    
    lilmat[i] = np.transpose(lstar0) + np.transpose(lstar1)
    L1 = np.count_nonzero(lstar1)
    haarvec = np.zeros((numleaves, 1))
    
    haarvec[index0] = np.sqrt(L1/(L1+1))
    haarvec[index1] = - np.sqrt(1/(L1*(L1+1)))
    sparsehaarlike[i] = np.transpose(haarvec)

    return lilmat, sparsehaarlike

def handle_right(node, lilmat, sparsehaarlike, i):
    """ Case that a node's right child is a leaf.
        Left child is not a leaf.
    """

    child = node.children
    index = nontip_index[child[0]]

    lstar0 = np.transpose(lilmat[index].todense())
    lstar1 = np.zeros((numleaves, 1))

    index0 = child[0].tip_names
    index1 = child[1].tip_names

    lstar0[index0] = lstar0[index0] + len(child[0].tip_names) * child[0].length
    lstar1[index1] = lstar1[index1] + len(child[1].tip_names) * child[1].length
    
    lilmat[i] = np.transpose(lstar0) + np.transpose(lstar1)

    L0 = np.count_nonzero(lstar0)
    haarvec = np.zeros((numleaves, 1))
    
    haarvec[index0] = np.sqrt(1/(L0*(L0+1)))
    haarvec[index1] = -np.sqrt(L0/((L0+1)))
    sparsehaarlike[i] = np.transpose(haarvec)

    return lilmat, sparsehaarlike

def handle_neither(node, lilmat, sparsehaarlike, i):
    """ Case that a node's children both have 
        descendants.
    """

    child = node.children

    # index of the internal node children, for 
    # indexing rows of matrices
    index0 = nontip_index[child[0]]
    index1 = nontip_index[child[1]]

    lstar0 = np.transpose(lilmat[index0].todense())
    lstar1 = np.transpose(lilmat[index1].todense())

    # index of the leaf children, for
    # indexing column of lstar vector
    index00 = child[0].tip_names
    index11 = child[1].tip_names

    lstar0[index00] = lstar0[index00] + len(child[0].tip_names) * child[0].length
    lstar1[index11] = lstar1[index11] + len(child[1].tip_names) * child[1].length
    
    lilmat[i] = np.transpose(lstar0) + np.transpose(lstar1)

    L0 = np.count_nonzero(lstar0)
    L1 = np.count_nonzero(lstar1)

    haarvec = np.zeros((numleaves, 1))
    
    haarvec[index00] = np.sqrt(L1/(L0*(L0+L1)))
    haarvec[index11] = - np.sqrt(L0/(L1*(L0+L1)))

    sparsehaarlike[i] = np.transpose(haarvec)

    return lilmat, sparsehaarlike

###############################################
# Traverse non_tips in postorder
traversal = t2.non_tips(include_self=True)
for i,node in enumerate(traversal):

    case = get_case(node)

    if case == 'both_child_tip':
        lilmat, sparsehaarlike = handle_both(node, lilmat, sparsehaarlike, i)
        
    elif case == 'left_child_tip':
        lilmat, sparsehaarlike = handle_left(node, lilmat, sparsehaarlike, i)

    elif case == 'right_child_tip':
        lilmat, sparsehaarlike = handle_right(node, lilmat, sparsehaarlike, i)

    elif case == 'neither_child_tip':
        lilmat, sparsehaarlike = handle_neither(node, lilmat, sparsehaarlike, i)


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
    sparsehaarlike = scipy.sparse.lil_matrix((numleaves, numleaves))
    lilmat = scipy.sparse.lil_matrix((numleaves, numleaves))
    sparsehaarlike, lilmat = handle_both(node, lilmat, sparsehaarlike, i)

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
    np.allclose(expected_shl.A, sparsehaarlike.A)