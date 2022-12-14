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
numleaves = len(all_leaves)

# Matrix that tracks all haar-like vectors
# Has n-leaves rows and n-internal nodes columns
# Each column represents an internal node
# Each row represents a leaf: 
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

    child = node.children
    lstar = np.zeros((numleaves, 1))
    index0 = child[0].tip_names
    index1 = child[1].tip_names

    lstar[index0] = child[0].length
    lstar[index1] = child[1].length
    lilmat[i] = np.transpose(lstar)
    haarvec = np.zeros((numleaves,1))
    haarvec[index0] = 1/np.sqrt(2)
    haarvec[index1] = -1/np.sqrt(2)
    sparsehaarlike[i] = np.transpose(haarvec)

    return lilmat, sparsehaarlike


###############################################
# RANDOM THINGS FOR TESTING

def get_node_from_ind(t2, node_ind):
""" Helper function for testing.
    Returns i'th non-tip node. """    
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

    # shl = sparse haar-like
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


###############################################
# Traverse non_tips in postorder
traversal = t2.non_tips(include_self=True)
for i,node in enumerate(traversal):

    case = get_case(node)

    if case == 'both_child_tip':

        lilmat, sparsehaarlike = handle_both(node, lilmat, 
                                             sparsehaarlike, i)
        
    elif case == 'left_child_tip':
        
        def handle_left(node, lilmat, sparsehaarlike, i):
            # NOT FINISHED

            child = node.children
            index = nontip_index[child[1]] # not sure about this

            lstar0 = np.zeros((numleaves, 1))
            lstar1 = np.transpose(lilmat[index].todense())


            index0 = child[0].tip_names
            index1 = child[1].tip_names


            # lstar0[index0] = child[0].length
            lstar0[index0] = lstar0[index0] + len(child[1].tip_names) * child[0].length
            lstar1[index1] = lstar1[index1] + len(child[1].tip_names) * child[1].length # child[1].dist
            
            lilmat[i] = np.transpose(lstar0) + np.transpose(lstar1)
            L1 = np.count_nonzero(lstar1)
            haarvec = np.zeros((numleaves, 1))
            
            haarvec[index0] = np.sqrt(L1/(L1+1))
            haarvec[index1] = -np.sqrt(1/(L1*(L1+1)))
            sparsehaarlike[i] = np.transpose(haarvec)

            return lilmat, sparsehaarlike

        lilmat, sparsehaarlike = handle_left(node, lilmat, sparsehaarlike, i)

    # TODO:
    #     # Right child is a leaf
    #     elif len(node2leaves[child[1]]) == 1:
    #         lstar1 = np.zeros((numleaves, 1))
    #         index1 = child[1].pos
    #         lstar1[index1] = child[1].dist
    #         index = child[0].loc
    #         lstar0 = np.transpose(lilmat[index].todense())
    #         index0 = child[0].pos
    #         lstar0[index0] = lstar0[index0] + len(child[0])*child[0].dist
    #         lilmat[i] = np.transpose(lstar1) + np.transpose(lstar0)
    #         L0 = np.count_nonzero(lstar0)
            
    #         haarvec = np.zeros((numleaves,1))
    #         haarvec[index0] = np.sqrt(1/(L0*(L0+1)))
    #         haarvec[index1] = -np.sqrt(L0/((L0+1)))
    #         sparsehaarlike[i] = np.transpose(haarvec)
        
    #     # Both children are leaves?
    #     else:
    #         index0 = child[0].loc
    #         index1 = child[1].loc
            
    #         lstar0 = np.transpose(lilmat[index0].todense())
    #         lstar1 = np.transpose(lilmat[index1].todense())
            
    #         index00 = child[0].pos
    #         lstar0[index00] = lstar0[index00] + len(child[0])*child[0].dist
    #         index11 = child[1].pos
    #         lstar1[index11] = lstar1[index11]+len(child[1])*child[1].dist
    #         lilmat[i] = np.transpose(lstar0)+np.transpose(lstar1)
    #         L0 = np.count_nonzero(lstar0)
    #         L1 = np.count_nonzero(lstar1)
            
    #         haarvec = np.zeros((numleaves, 1))
    #         haarvec[index00] = np.sqrt(L1/(L0*(L0+L1)))
    #         haarvec[index11] = -np.sqrt(L0/(L1*(L0+L1)))
            
    #         sparsehaarlike[i] = np.transpose(haarvec)
    #     i+=1


all_leaves = t.get_leaves()

node_index = {l.name:i for i,l in enumerate(all_leaves)}
# return dictionary of node instances, 
# allows quick access to node attributes without traversing the tree
node2leaves = t.get_cached_content()
numleaves = len(t) 

# Initialize row-based list of lists sparse matrix to store Haar-like vectors
# List of lists format
sparsehaarlike = scipy.sparse.lil_matrix((numleaves, numleaves))
all_leaves = t.get_leaves()
mastersplit = t.children
lilmat = scipy.sparse.lil_matrix((numleaves, numleaves))
node2leaves
def is_leaf_fn(node):
    print(node, node.is_leaf())
    return node.is_leaf()


traversal = t.traverse("postorder", is_leaf_fn)
for node in traversal:
    print('xxxxxxxx')
traversal = t.traverse("postorder")
for node in traversal:

    pos = find_matching_index(node, all_leaves)
    print(pos)
    print('   ')
for i, node in enumerate(node):
    print(i, node)
# I had an error of "TreeNode object has no pos"
# Resolution: I had added the continue break too early on in code
# ordering of nodes in post order traversal

# Error of list index out of range for lilmat
# Move i+= to the end of the code
i = 0
traversal = t.traverse("postorder")
for node in traversal:

    print(i,node)
    pos = find_matching_index(node, all_leaves)
    
    node.add_features(pos=pos) # store indices of leaves under each internal node
    node.add_features(loc=i) # Add node index to node features
    
    if node.is_leaf(): continue # only iterate over internal nodes

    # Both children are leaves
    if len(node2leaves[node]) == 2:
        child = node.children
        lstar = np.zeros((numleaves, 1))
        index0 = child[0].pos
        index1 = child[1].pos
        lstar[index0] = child[0].dist
        lstar[index1] = child[1].dist
        lilmat[i] = np.transpose(lstar)
        haarvec = np.zeros((numleaves,1))
        haarvec[index0] = 1/np.sqrt(2)
        haarvec[index1] = -1/np.sqrt(2)
        sparsehaarlike[i] = np.transpose(haarvec)

    else:
        
        child = node.children

        # Left child is a leaf
        if len(node2leaves[child[0]]) == 1:

            lstar0 = np.zeros((numleaves, 1))
            index0 = child[0].pos
            lstar0[index0] = child[0].dist
            index = child[1].loc
            lstar1 = np.transpose(lilmat[index].todense())
            index1 = child[1].pos                
            lstar1[index1] = lstar1[index1] + len(child[1])*child[1].dist
            lilmat[i] = np.transpose(lstar0) + np.transpose(lstar1)
            L1 = np.count_nonzero(lstar1)
            haarvec = np.zeros((numleaves, 1))
            
            haarvec[index0] = np.sqrt(L1/(L1+1))
            haarvec[index1] = -np.sqrt(1/(L1*(L1+1)))
            sparsehaarlike[i] = np.transpose(haarvec)

        # Right child is a leaf
        elif len(node2leaves[child[1]]) == 1:
            lstar1 = np.zeros((numleaves, 1))
            index1 = child[1].pos
            lstar1[index1] = child[1].dist
            index = child[0].loc
            lstar0 = np.transpose(lilmat[index].todense())
            index0 = child[0].pos
            lstar0[index0] = lstar0[index0] + len(child[0])*child[0].dist
            lilmat[i] = np.transpose(lstar1) + np.transpose(lstar0)
            L0 = np.count_nonzero(lstar0)
            
            haarvec = np.zeros((numleaves,1))
            haarvec[index0] = np.sqrt(1/(L0*(L0+1)))
            haarvec[index1] = -np.sqrt(L0/((L0+1)))
            sparsehaarlike[i] = np.transpose(haarvec)
        
        # Both children are leaves?
        else:
            index0 = child[0].loc
            index1 = child[1].loc
            
            lstar0 = np.transpose(lilmat[index0].todense())
            lstar1 = np.transpose(lilmat[index1].todense())
            
            index00 = child[0].pos
            lstar0[index00] = lstar0[index00] + len(child[0])*child[0].dist
            index11 = child[1].pos
            lstar1[index11] = lstar1[index11]+len(child[1])*child[1].dist
            lilmat[i] = np.transpose(lstar0)+np.transpose(lstar1)
            L0 = np.count_nonzero(lstar0)
            L1 = np.count_nonzero(lstar1)
            
            haarvec = np.zeros((numleaves, 1))
            haarvec[index00] = np.sqrt(L1/(L0*(L0+L1)))
            haarvec[index11] = -np.sqrt(L0/(L1*(L0+L1)))
            
            sparsehaarlike[i] = np.transpose(haarvec)
        i+=1
lilmat.data
i=0 #ordering of nodes in post order traversal
for node in t.traverse("postorder", is_leaf_fn):
    
    node.add_features(pos=find_matching_index(node,all_leaves)) # store indices of leaves under each internal node
    veclen=len(node2leaves[node])
    print(i)
    
    if not node.is_leaf():
        node.add_features(loc=i) #add node index to node features
        if veclen==2:
            child=node.children
            lstar=np.zeros((numleaves,1))
            index0=child[0].pos
            index1=child[1].pos
            lstar[index0]=child[0].dist
            lstar[index1]=child[1].dist
            lilmat[i]=np.transpose(lstar)
            haarvec=np.zeros((numleaves,1))
            haarvec[index0]=1/np.sqrt(2)
            haarvec[index1]=-1/np.sqrt(2)
            sparsehaarlike[i]=np.transpose(haarvec)
            i=i+1
        else:
            child=node.children
            if len(node2leaves[child[0]])==1:

                lstar0=np.zeros((numleaves,1))
                index0=child[0].pos
                lstar0[index0]=child[0].dist

                index=child[1].loc
                lstar1=np.transpose(lilmat[index].todense())
                index1=child[1].pos                
                lstar1[index1]=lstar1[index1]+len(child[1])*child[1].dist
                lilmat[i]=np.transpose(lstar0)+np.transpose(lstar1)
                L1=np.count_nonzero(lstar1)
                haarvec=np.zeros((numleaves,1))
                haarvec[index0]=np.sqrt(L1/(L1+1))
                haarvec[index1]=-np.sqrt(1/(L1*(L1+1)))
                sparsehaarlike[i]=np.transpose(haarvec)
                i=i+1
            elif len(node2leaves[child[1]])==1:
                lstar1=np.zeros((numleaves,1))
                index1=child[1].pos
                lstar1[index1]=child[1].dist
                index=child[0].loc
                lstar0=np.transpose(lilmat[index].todense())
                index0=child[0].pos
                lstar0[index0]=lstar0[index0]+len(child[0])*child[0].dist
                lilmat[i]=np.transpose(lstar1)+np.transpose(lstar0)
                L0=np.count_nonzero(lstar0)
                haarvec=np.zeros((numleaves,1))
                haarvec[index0]=np.sqrt(1/(L0*(L0+1)))
                haarvec[index1]=-np.sqrt(L0/((L0+1)))
                sparsehaarlike[i]=np.transpose(haarvec)
                i=i+1
            else:
                index0=child[0].loc
                index1=child[1].loc
                lstar0=np.transpose(lilmat[index0].todense())
                lstar1=np.transpose(lilmat[index1].todense())
                index00=child[0].pos
                lstar0[index00]=lstar0[index00]+len(child[0])*child[0].dist
                index11=child[1].pos
                lstar1[index11]=lstar1[index11]+len(child[1])*child[1].dist
                lilmat[i]=np.transpose(lstar0)+np.transpose(lstar1)
                L0=np.count_nonzero(lstar0)
                L1=np.count_nonzero(lstar1)
                haarvec=np.zeros((numleaves,1))
                haarvec[index00]=np.sqrt(L1/(L0*(L0+L1)))
                haarvec[index11]=-np.sqrt(L0/(L1*(L0+L1)))
                sparsehaarlike[i]=np.transpose(haarvec)
                i=i+1
# CODE GRAVEYARD



t2 = read(StringIO(tree_string), 
          format="newick", into=TreeNode)

node_index = {l: i for i, l in enumerate(t2.tips())}
f = lambda n: [node_index[n]] if n.is_tip() else []
f2 = lambda n: [n] 


t2.cache_attr(f, 'tip_names')

traversal = t2.postorder(include_self=True)
for n in traversal:
    print("Node name: %s, cache: %r" % (n.name, n.tip_names))
    print('   ')




treefile = "Sparsify-Ultrametric/tree.qza"

tree = qiime2.Artifact.load(treefile)

try:
    from q2_types.tree import NewickFormat
except ImportError:
    NewickFormat = str



