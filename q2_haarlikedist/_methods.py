# ----------------------------------------------------------------------------
# Copyright (c) 2022--, haar-like-dist development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from collections import defaultdict
import skbio
import qiime2
import biom
import numpy as np
import pandas as pd
import os
import hashlib
import time
import pickle

from skbio import read
from skbio.tree import TreeNode
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import OrdinationResults
from skbio.stats.ordination import pcoa

from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz

from sklearn.metrics import silhouette_samples

import q2templates
from qiime2 import Metadata, Artifact

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from qiime2.plugins.emperor.visualizers import plot as emperor_plot

from pkg_resources import resource_filename

from ._adaptive import *







#####################################################
import os
import json
import numpy as np
from scipy.sparse import issparse, save_npz


def save_plot_checkpoint(path, mags, Y, coordinates, coefs, dic):
    """
    Save everything needed to reproduce the balance boxplots to disk.

    Parameters
    ----------
    path        : str  – directory to write files into (created if absent)
    mags        : scipy sparse matrix (n_nodes x n_samples)
    Y           : pd.Series or array-like of integer class labels
    coordinates : list[int]  – selected node indices from matching pursuit
    coefs       : list[float] – matching pursuit coefficients
    dic         : dict  – {class_name: int_label}  e.g. {'positive':1,'negative':2}
    """
    os.makedirs(path, exist_ok=True)

    # 1. mags  (keep sparse to avoid blowing up disk for large trees)
    mags_path = os.path.join(path, "checkpoint_mags.npz")
    if issparse(mags):
        save_npz(mags_path, mags.tocsr())
    else:
        # fall back: dense -> compressed npz
        np.savez_compressed(mags_path.replace(".npz", "_dense.npz"),
                            mags=np.asarray(mags))

    # 2. Y labels
    np.save(os.path.join(path, "checkpoint_Y.npy"), np.asarray(Y))

    # 3. coordinates + coefs  (small arrays – plain npy is fine)
    np.save(os.path.join(path, "checkpoint_coordinates.npy"),
            np.asarray(coordinates, dtype=np.int64))
    np.save(os.path.join(path, "checkpoint_coefs.npy"),
            np.asarray(coefs, dtype=np.float64))

    # 4. dic  (string keys -> json)
    with open(os.path.join(path, "checkpoint_dic.json"), "w") as f:
        json.dump(dic, f, indent=2)

    print(f"[checkpoint] saved to {path}")
    print(f"  mags      : {mags.shape}")
    print(f"  Y         : {len(Y)} samples")
    print(f"  coordinates: {coordinates}")
    print(f"  coefs     : {[round(float(c), 6) for c in coefs]}")
    print(f"  dic       : {dic}")
#####################################################









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

    # DEBUG
    t2.bifurcate(0)
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

    l_is_tip = not node.children[0].has_children()
    r_is_tip = not node.children[1].has_children()

    case = {
        (False, False): "neither",
        (False, True): "right",
        (True, False): "left",
        (True, True): "both"
    }

    return case[(l_is_tip, r_is_tip)]


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

    if child.length is None:
        child.length = 0

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
    """ Returns lilmat, shl represented as two branching trees. 
        This will allow the number of internal nodes to 
        match the number of tips. 

        NOTE: ntips0 and ntips1 are defined as 1 in the case
        that a root's child is a tip. Otherwise this would
        assign 0 causing division by 0 errors for values0,
        or values 1
        """

    child0, child1 = t2.children

    ntips0 = len([x for x in child0.tips()])
    ntips1 = len([x for x in child1.tips()])
    ntips0 = max(ntips0, 1)
    ntips1 = max(ntips1, 1)

    if ntips0 + ntips1 != shl[0].shape[1]:
        m = 'Number of tips inconsistent between tree and shl matrix'
        raise ValueError(m)

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
    """ Sparsifies a tree and returns lilmat, shl.
        Represents tree as matrices. """

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


def fast_tree_hash(tree):
    """Efficiently generate a unique hash for a tree structure."""
    def hash_node(node):
        node_id = node.name if node.name is not None else ""
        child_hashes = tuple(sorted(hash_node(child)
                             for child in node.children)) if node.children else ()
        return hashlib.sha256((node_id + str(child_hashes)).encode()).hexdigest()

    return hash_node(tree)


def get_haar_basis(tree, cache_dir="cache/"):
    """Returns cached Haar basis if available, otherwise computes and saves it."""

    os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists
    tree_id = fast_tree_hash(tree)  # Fast unique tree identifier
    cache_path = os.path.join(cache_dir, f"haar_basis_{tree_id}.npz")
    print('cache path:', os.getcwd(), cache_path)

    if os.path.exists(cache_path):
        print(f"Loading cached Haar basis for tree {tree_id}...")
        return load_npz(cache_path).tolil()  # Load from cache

    print(f"Computing Haar basis for new tree {tree_id}...")
    _, haar_basis = sparsify(tree)
    save_npz(cache_path, haar_basis.tocsr())
    return haar_basis


def get_lambda(lilmat, shl, i):
    """ Computes lambda for each internal node.
        Lambda is the lilmat entry for i * shl entry ^2. """

    lstar = lilmat[i].todense().T
    phi = shl[i].todense()
    phi2 = np.multiply(phi, phi)
    lambd = phi2.dot(lstar)
    return lambd


def get_lambdas(lilmat, shl):
    """ Computes all lambdas. """

    n = lilmat.shape[0]
    data = [get_lambda(lilmat, shl, i) for i in range(n)]
    data = np.array(data).T
    diagonal = data[0][0]
    return diagonal


def match_to_tree(biom_table, tree):
    """ Returns aligned data in biom format.
        biom_table: biom.Table.
        tree: skbio.TreeNode. """

    biom_table = biom_table.norm(inplace=False)
    biom_table, tree = biom_table.align_tree(tree)

    return tree, biom_table


def compute_haar_dist(mags, diagonal):

    # columns are samples
    nsamples = mags.shape[1]
    diagonal_clipped = np.maximum(diagonal, 0)
    diagonal_mat = csr_matrix([diagonal_clipped] * nsamples)
    diagonal_mat_sqrt = np.sqrt(diagonal_mat)

    modmags = mags.T.multiply(diagonal_mat_sqrt)

    # compute the distance matrix
    D = lil_matrix((nsamples, nsamples))  # Blank dist matrix
    mat = csr_matrix(np.ones((nsamples, 1)))  # Blank csr_matrix for broadcast
    for i in range(nsamples):
        a = modmags - modmags[i, :].multiply(mat)
        b = csr_matrix.power(a, 2)
        c = csr_matrix.sum(b, axis=1)
        d = np.sqrt(c)
        D[i, :] = csr_matrix(d.T)

    D = D + D.T

    assert (D != D.T).nnz == 0

    return D, modmags


def left_children(node):
    if not node.children[0].is_tip():
        return [x.name for x in node.children[0].tips()]
    else:
        return [node.children[0].name]


def right_children(node):
    if not node.children[1].is_tip():
        return [x.name for x in node.children[1].tips()]
    else:
        return [node.children[1].name]


def get_taxa(node_name_list, taxonomy, make_set=True):
    """ node_name_list, taxonomy in dict form, make_set=True """
    taxa = []
    for name in node_name_list:
        try:
            t = taxonomy[name]
        except KeyError:
            t = 'not found'
        taxa.append(t)

    if make_set:
        taxa = set(taxa)

    return taxa


def get_important_taxa(tree, idx, taxonomy):
    nontips = list(tree.non_tips(include_self=True))
    node = nontips[idx]
    l = left_children(node)
    r = right_children(node)
    l_taxa = get_taxa(l, taxonomy)
    r_taxa = get_taxa(r, taxonomy)
    return l_taxa, r_taxa


def get_species(tree, coords, taxonomy):
    species = defaultdict(dict)
    nontips = list(tree.non_tips(include_self=True))
    for i, c in enumerate(coords):
        l, r = get_important_taxa(tree, c, taxonomy)
        label = f'{nontips[c].name}'
        keyname = f'coord {i}: {label}'
        species[keyname]['left'], species[keyname]['right'] = l, r
    return species


def find_common_clade(node, taxonomy):
    """ Taxonomy must be a dict mapping tips to taxonomies """

    # Get the taxonomy paths for all descendant tips
    taxonomies = [taxonomy[tip.name].split(';')
                  for tip in node.tips()
                  if tip.name in taxonomy]

    if not taxonomies:  # This should only be true for the root or unfound taxa
        return None

    # Find the common clade by comparing taxonomy paths
    common_clade = []
    for clade_parts in zip(*taxonomies):
        if all(part == clade_parts[0] for part in clade_parts):
            common_clade.append(clade_parts[0])
        else:
            break

    # Return the common clade as a semicolon-separated string
    clade = ';'.join(common_clade) if common_clade else None

    return clade


def annotate_tree(tree, taxonomy):
    """ Adds the most common taxonomy descendant and postorder position 
        as names to the tree. Returns tree, taxonomy. """

    nontips = [x for x in tree.postorder(include_self=True) if not x.is_tip()]

    for i, node in enumerate(nontips):

        node.__setattr__('original_name', node.name)
        node.__setattr__('name', i)

    # Traverse the tree and label internal nodes
    taxonomy = taxonomy.to_dataframe().reset_index()
    taxonomy_map = dict(zip(taxonomy['Feature ID'], taxonomy['Taxon']))
    for node in nontips:
        common_clade = find_common_clade(node, taxonomy_map)
        if common_clade:
            node.name = f'{node.name}: {common_clade}'

    return tree, taxonomy_map


def save_species(species, output_dir):

    s = ''
    for k, v in species.items():
        s += f'\n{k}\n'

        # Check if `species[k]` is a dictionary and contains 'left' and 'right' keys
        left_values = species[k].get(
            'left', []) if isinstance(species[k], dict) else []
        right_values = species[k].get(
            'right', []) if isinstance(species[k], dict) else []

        # Add left values to the output string
        for x in left_values:
            s += f'\t\t{x}\n'

        # Add separator
        s += f'\n\t\t~~~~\n'

        # Add right values to the output string
        for x in right_values:
            s += f'\t\t{x}\n'

    fname = os.path.join(output_dir, 'species.txt')
    with open(fname, 'w') as f:
        f.write(s)

    return s


def _save_silhouettes(output_dir: str, D_sparse, y_series: pd.Series, variable: str):
    """
    Compute per-sample silhouette scores from a *precomputed* distance matrix
    and save them to <output_dir>/silhouttes.pickle.

    Parameters
    ----------
    output_dir : str
        Folder for this variable's outputs.
    D_sparse : scipy.sparse matrix (nsamples x nsamples)
        Haar-like *distance* matrix (symmetric, zero diagonal).
    y_series : pd.Series
        Labels for the samples in the same order as D_sparse.
    variable : str
        Variable name (label) for bookkeeping.
    """
    y = np.asarray(list(y_series))
    classes, counts = np.unique(y, return_counts=True)

    payload = {
        'variable': variable,
        'classes': classes.tolist(),
        'class_counts': counts.tolist(),
        'scores': None,
        'mean': np.nan,
        'metric': 'precomputed',
        'note': ''
    }

    # Silhouette requires at least two classes with at least 2 samples each
    if len(classes) < 2 or np.any(counts < 2):
        payload['note'] = 'insufficient class structure for silhouette'
        with open(os.path.join(output_dir, 'silhouttes.pickle'), 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    # sklearn expects dense for metric='precomputed'
    D = np.asarray(D_sparse.todense())
    np.fill_diagonal(D, 0.0)

    scores = silhouette_samples(D, y, metric='precomputed')
    payload['scores'] = scores.astype(np.float32)
    payload['mean'] = float(np.nanmean(scores))

    with open(os.path.join(output_dir, 'silhouttes.pickle'), 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_tax_heatmap_info(label: str,
                           output_dir: str,
                           tree: skbio.TreeNode,
                           coordinates: list[int],
                           species_dict: dict):
    """
    Collect per-coordinate info for taxonomic heatmaps and save to
    <output_dir>/heatmap_info.pickle.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ---------- helpers ----------
    def _dist_to_root(n: skbio.TreeNode) -> float:
        d = 0.0
        cur = n
        while cur is not None:
            if getattr(cur, 'length', None):
                d += float(cur.length)
            cur = cur.parent
        return d

    def _species_set_from_taxa_strings(taxa_strings):
        """Extract species names (after 's__'), drop empties."""
        spp = set()
        for t in taxa_strings:
            if not isinstance(t, str):
                continue
            parts = [p.strip() for p in t.split(';')]
            s_parts = [p for p in parts if p.startswith('s__')]
            if not s_parts:
                continue
            s_val = s_parts[-1][3:].strip()  # after 's__'
            if s_val:
                spp.add(s_val)
        return spp

    # Use the SAME traversal order as sparsify()/haar basis construction
    nontips = list(tree.non_tips(include_self=True))

    # scale: use internal nodes only (optional, but tighter)
    try:
        max_root_distance = max((_dist_to_root(n)
                                for n in nontips), default=0.0)
    except ValueError:
        max_root_distance = 0.0

    per_coord = []
    for i, c in enumerate(coordinates):
        # robust node lookup
        if 0 <= c < len(nontips):
            node = nontips[c]
        else:
            # fallback: try attribute set during sparsify()
            node = next((n for n in nontips if getattr(
                n, 'postorder_pos', None) == c), None)
            if node is None:
                # cannot map this coordinate; skip gracefully
                per_coord.append({
                    'coord_index': int(c),
                    'node_label': f'<unmapped:{c}>',
                    'lca': None,
                    'dist_to_root': float('nan'),
                    'nonoverlap_species_count': 0,
                    'species_left': [],
                    'species_right': [],
                })
                continue

        node_label = getattr(node, 'name', str(c))
        lca_text = node_label.split(':', 1)[1].strip() if isinstance(
            node_label, str) and ':' in node_label else None

        # safer species lookup: match only on the "coord i:" prefix
        key_prefix = f'coord {i}:'
        key = next((k for k in species_dict.keys()
                   if str(k).startswith(key_prefix)), None)

        left_taxa = species_dict.get(key, {}).get(
            'left', set()) if key else set()
        right_taxa = species_dict.get(key, {}).get(
            'right', set()) if key else set()

        left_spp = _species_set_from_taxa_strings(left_taxa)
        right_spp = _species_set_from_taxa_strings(right_taxa)
        nonoverlap = len(left_spp.symmetric_difference(right_spp))

        per_coord.append({
            'coord_index': int(c),
            'node_label': node_label,
            'lca': lca_text,
            'dist_to_root': float(_dist_to_root(node)),
            'nonoverlap_species_count': int(nonoverlap),
            'species_left': sorted(left_spp),
            'species_right': sorted(right_spp),
        })

    payload = {
        'variable': label,
        'max_root_distance': float(max_root_distance),
        'per_coord': per_coord,
    }

    with open(os.path.join(output_dir, 'heatmap_info.pickle'), 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_emperor_pca_qzv_from_biplot(coefs,
                                     coordinates,
                                     mags,
                                     s,
                                     y_series,
                                     dic,
                                     out_dir,
                                     filename="interactive-pca.qzv",
                                     standardize=True,
                                     sample_ids=None):

    Z = np.transpose(reconstruct_coord(
        coefs, coordinates, mags, s))
    Z = np.asarray(Z, dtype=float)

    if standardize:
        Z = StandardScaler().fit_transform(Z)

    pca = PCA()
    scores = pca.fit_transform(Z)
    eigvals = pca.explained_variance_
    ratios = pca.explained_variance_ratio_

    n, p = scores.shape
    pc_labels = [f"PC{i+1}" for i in range(p)]

    # ----- sample IDs & metadata -----
    if sample_ids is not None:
        ids = [str(x) for x in sample_ids]
    elif getattr(y_series, "index", None) is not None and len(y_series.index) == n:
        ids = [str(x) for x in y_series.index]
    else:
        ids = [f"S{i}" for i in range(n)]

    inv = {v: k for k, v in dic.items()}  # code -> readable name
    md_df = pd.DataFrame({"group": [inv[int(v)] for v in y_series]}, index=ids)
    md_df.index.name = "sample-id"  # required by QIIME 2
    md = Metadata(md_df)

    # ----- ordination (PCA wrapped as a "PCoAResults"-compatible object) -----
    samples_df = pd.DataFrame(scores, index=ids, columns=pc_labels)
    ord_res = OrdinationResults(
        short_method_name="PCA",
        long_method_name="Principal Components Analysis",
        eigvals=pd.Series(eigvals, index=pc_labels),
        samples=samples_df,
        proportion_explained=pd.Series(ratios, index=pc_labels)
    )

    # Package as QIIME2 artifact and plot with Emperor
    ord_art = Artifact.import_data("PCoAResults", ord_res)
    viz = emperor_plot(ord_art, md)   # positional args for broad compatibility

    out_path = os.path.join(out_dir, filename)
    # robust save across versions
    if hasattr(viz, "save"):
        viz.save(out_path)
    elif hasattr(viz, "visualization") and hasattr(viz.visualization, "save"):
        viz.visualization.save(out_path)
    else:
        raise RuntimeError(
            "Unexpected return type from emperor.plot; cannot save.")
    return out_path


def haar_like_dist(table: biom.Table,
                   tree: skbio.TreeNode) -> (skbio.DistanceMatrix, skbio.TreeNode, csr_matrix, OrdinationResults):
    """ Returns D, tree, mm. Distance matrix and significance.
        Returns distance matrix and formatted tree.
        This now returns modmags as a biom table, which
        can be thought of as a differentially encoded
        feature table. """

    lilmat, haar_basis = sparsify(tree)
    abund_vec = get_otu_abundances(table, tree)

    diagonal = get_lambdas(lilmat, haar_basis)
    mags = calc_haar_mags(haar_basis, abund_vec)

    D, modmags = compute_haar_dist(mags, diagonal)
    ids = table.ids(axis='sample')
    D = DistanceMatrix(np.asarray(D.todense()), ids)
    mm = csr_matrix(modmags)  # Going to see if this works w new format
    p = pcoa(D)

    return D, tree, mm, p


def adaptive_visual(
    output_dir: str,
    biom_table: biom.Table,
    tree: skbio.TreeNode,
    label: str,
    metadata: Metadata,
    taxonomy: Metadata = None,
    s: int = 5,  # Number of important nodes
    k: int = 5,
    n: int = 5,
    cluster_affinity: bool = True,
    num_clstr: int = 6000,
    tune: bool = False
) -> None:

    starttime = time.time()

    print('tree tips before align:', len(list(tree.tips())))
    tree, biom_table = match_to_tree(biom_table, tree)
    print('tree tips after align:', len(list(tree.tips())))

    haar_basis = get_haar_basis(tree)
    meta = metadata.to_dataframe()

    adhld_results = adaptive(haar_basis, biom_table, label,
                             tree, meta, s, cluster_affinity, num_clstr, tune)

    _, coordinates, coefs, Y, dic, diagonal, mags = adhld_results

    Distances, modmags = compute_haar_dist(mags, diagonal)
    modmags = modmags.T

    # Save silhouettes for this variable (radar plot source)
    _save_silhouettes(output_dir, Distances, Y, label)

    # Interactive PCA from HLD biplot features
    try:
        save_emperor_pca_qzv_from_biplot(coefs, coordinates, mags, s, Y, dic, output_dir,
                                         filename="interactive-pca.qzv", standardize=True, sample_ids=None)
        print("Saved Emperor PCA (feature-based) to interactive-pca.qzv")
    except Exception as e:
        print(f"[WARN] Unable to write Emperor PCA visualization: {e}")

    # Get context to send to HTML file
    annotated_tree = tree
    if taxonomy:
        annotated_tree, taxonomy = annotate_tree(tree, taxonomy)
        species = get_species(annotated_tree, coordinates, taxonomy)

        # heatmap info (per-coordinate detail saved; aggregation happens later)
        _save_tax_heatmap_info(
            label, output_dir, annotated_tree, coordinates, species)
    else:
        species = {'coord 1': 'No taxonomy provided'}






    #####################################################
    # ── checkpoint for fast plot iteration ──
    save_plot_checkpoint(output_dir, mags, Y, coordinates, coefs, dic)
    #####################################################






    make_plots(adhld_results, modmags, output_dir, s, k, n)

    s = save_species(species, output_dir)
    coords = ' '.join([str(x) for x in coordinates])
    context = {'coordinates': coords,
               'label': label,
               's': s
               }

    # Visualizer
    TEMPLATES = resource_filename(
        'q2_haarlikedist', 'adhld_assets')
    index = os.path.join(TEMPLATES, 'index.html')
    q2templates.render(index, output_dir, context=context)

    print(f'adaptive visual took {time.time() - starttime} second to finish.')


def adaptive_distance(
        table: biom.Table,
        tree: skbio.TreeNode,
        label: str,
        metadata: Metadata,
        taxonomy: pd.DataFrame = None,
        s: int = 5  # Number of important nodes
    ) -> (skbio.DistanceMatrix,
          skbio.TreeNode,
          csr_matrix,
          OrdinationResults,
          pd.DataFrame):  # type: ignore

    haar_basis = get_haar_basis(tree)
    meta = metadata.to_dataframe()

    adhld_results = adaptive(haar_basis, table, label, tree, meta, s)

    _, _, _, _, _, _, diagonal, mags = adhld_results

    D, modmags = compute_haar_dist(mags, diagonal)

    ids = table.ids()
    D = skbio.DistanceMatrix(np.array(D.todense()), ids)
    mm = csr_matrix(modmags)
    p = pcoa(D)

    feature_metadata = taxonomy
    feature_metadata = feature_metadata.to_dataframe()
    return D, tree, mm, p, feature_metadata
