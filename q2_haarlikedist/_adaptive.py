from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from numpy.typing import DTypeLike
from sklearn_extra.cluster import KMedoids
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix, csc_matrix
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import optuna
import time
import numpy as np
import pandas as pd

# --- headless rendering: must come before any matplotlib/ete3 import ---
import os
os.environ.setdefault('MPLBACKEND', 'Agg')          # Matplotlib: no GUI
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')  # Qt: no X server


matplotlib.use('Agg', force=True)


__all__ = [
    'calc_haar_mags',
    'get_otu_abundances',
    'preprocess',
    'proximity_from_leaves_parallel',
    'mds_full_randomized',
    'convert_least_squares',
    'matching_pursuit_sequential_parallel',
    'diag_impo',
    'adaptive',
    'reconstruct_coord',
    'reconstruct',
    'new_biplot3d',
    'new_biplot3dnormalized',
    '_get_correlation',
    'rfgram_plot',
    'make_plots',
    'boxplot_plotter'
]


def calc_haar_mags(haar_basis, abund_vec):
    # Convert to sparse form
    abund_vec = csr_matrix(abund_vec)
    # Normalize abundances
    abund_vec = abund_vec / abund_vec.sum(axis=0)
    # Apply Haar transformation
    mags = haar_basis @ abund_vec
    mags = csr_matrix(mags)
    return mags


def get_otu_abundances(table, tree):
    """ Maps the given OTU abundances onto a reference tree.
        table: pd.DataFrame
        tree: skbio.TreeNode (Newick tree structure)

        Returns:
        - abundvec: np.ndarray (OTU abundances mapped onto the tree)
    """

    return table.reindex(columns=[leaf.name for leaf in tree.tips()], fill_value=0).T.values


def preprocess(label, biom_table, haar_basis, metadata, tree):
    """ For adaptive hld. Converts to necessary types for
        Evan's HLD code to work.

        label: column name in metadata,
        biomtab: biom.Table,
        shl: result from sparsification,
        meta: metadata with `label` not as index,
        returns:  X, Y, mags, dic
        """

    labels = list(metadata[label].unique())
    dic = dict(zip(labels, list(range(1, len(labels)+1))))

    mapped_labels = metadata[label].map(dic).rename("labels")
    X = biom_table.to_dataframe().T.join(
        mapped_labels, how="inner").sort_values("labels")
    Y = X["labels"]
    X = X.drop(columns="labels")

    abundance_vector = get_otu_abundances(X, tree)
    mags = calc_haar_mags(haar_basis, abundance_vector)

    # Save outputs for comparison
    print("X shape:", X.shape)
    print(type(X))
    print(X)
    print("Y shape:", Y.shape)
    print(type(Y))
    print("abundance_vector shape:", abundance_vector.shape)
    print("mags shape:", mags.shape)
    print("haar_basis shape:", haar_basis.shape)

    # Normalize X
    sums = X.to_numpy().sum(axis=1)
    X = X.div(sums, axis=0)
    X = np.array(X)

    return X, Y, mags, dic


# ---------- parallel proximity (tree-wise) ----------
def _prox_sum_for_tree_column(col: np.ndarray, dtype: DTypeLike = np.float32):
    """Return the nxn coincidence matrix for one tree (float32)."""
    # equality outer product -> 1 if same leaf, else 0
    eq = (col[:, None] == col[None, :])
    # cast once; returning ndarray to be summed by caller
    return eq.astype(dtype, copy=False)


def proximity_from_leaves_parallel(terminals: np.ndarray,
                                   n_jobs: int = None,
                                   return_distance: bool = True,
                                   dtype=np.float32):
    """
    terminals: (n_samples, n_trees) leaf indices from RF.apply(X)
    Parallelizes across trees; returns:
      - distance matrix D = 1 - proximity if return_distance=True
      - otherwise the proximity matrix in [0,1].
    """
    n, T = terminals.shape
    if n_jobs is None:
        n_jobs = min(cpu_count(), T)

    # chunk trees into roughly equal parts for threads
    splits = np.array_split(np.arange(T), n_jobs)
    prox_parts = [None] * len(splits)

    def worker(chunk_idx):
        cols = splits[chunk_idx]
        if len(cols) == 0:
            return np.zeros((n, n), dtype=dtype)
        acc = np.zeros((n, n), dtype=dtype)
        # accumulate chunk's trees
        for t in cols:
            acc += _prox_sum_for_tree_column(terminals[:, t], dtype=dtype)
        return acc

    with ThreadPoolExecutor(max_workers=len(splits)) as ex:
        futures = {ex.submit(worker, i): i for i in range(len(splits))}
        for fut in futures:
            idx = futures[fut]
            prox_parts[idx] = fut.result()

    prox = np.zeros((n, n), dtype=dtype)
    for part in prox_parts:
        prox += part

    prox /= terminals.shape[1]  # normalize by #trees
    if return_distance:
        return (1.0 - prox).astype(dtype, copy=False)
    return prox


def mds_full_randomized(D, dim=50, n_oversamples=10, random_state=None):
    """
    Memory-efficient full MDS using randomized SVD.

    Parameters
    ----------
    D : ndarray of shape (n_samples, n_samples)
        Full distance matrix (symmetric, zero diagonal).
    dim : int
        Target embedding dimension.
    n_oversamples : int
        Extra dimensions to improve accuracy in randomized SVD.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, dim)
        Low-dimensional embedding.
    """
    # Squared distances
    D2 = D ** 2

    # Double centering
    row_mean = np.mean(D2, axis=1, keepdims=True)
    col_mean = np.mean(D2, axis=0, keepdims=True)
    total_mean = np.mean(D2)
    B = -0.5 * (D2 - row_mean - col_mean + total_mean)

    # Randomized truncated SVD
    U, S, _ = randomized_svd(
        B,
        n_components=dim,
        n_oversamples=n_oversamples,
        random_state=random_state
    )

    # Convert eigenvalues to coordinates
    pos_mask = S > 1e-12
    U = U[:, pos_mask]
    S = S[pos_mask]
    X = U * np.sqrt(S)

    return X


def center_gram(K: np.ndarray) -> np.ndarray:
    if not isinstance(K, np.ndarray):
        K = K.toarray()

    """Double-center a Gram/similarity matrix K."""
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


def make_label_kernel(y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Construct the ideal label kernel L_ij = 1[y_i == y_j],
    center it, and return (L_c, ||L_c||_F).
    """
    _, inv = np.unique(y, return_inverse=True)
    L = (inv[:, None] == inv[None, :]).astype(np.float64)
    Lc = center_gram(L)
    Lc_norm = np.linalg.norm(Lc)
    return Lc, Lc_norm


def kta_score(K: np.ndarray, Lc: np.ndarray, Lc_norm: float) -> float:
    """
    1 - centered kernel target alignment between K and the label kernel.
    Lower is better, 0 is perfect alignment, 1 is poor alignment.
    """
    Kc = center_gram(K)
    num = np.sum(Kc * Lc)
    denom = (np.linalg.norm(Kc) * Lc_norm) + 1e-12
    alignment = num / denom
    return 1.0 - alignment


# ---------- sparsegram builders ----------
def compute_mds_embedding_from_distance(D: np.ndarray,
                                        size_embedding: int = 50,
                                        random_state=None):
    return mds_full_randomized(D, dim=size_embedding, random_state=random_state)


def build_sparsegram_from_embedding(embedding: np.ndarray):
    # K = X X^T ; return as float32 to save RAM
    X = np.asarray(embedding, dtype=np.float32)
    return X @ X.T


# ---------- stratified subsample ----------
def stratified_cap_indices(y: np.ndarray, cap: int, rng=None):
    """
    Balanced subsample across classes (without replacement).
    Returns sorted indices into original array.
    """
    rng = check_random_state(rng)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    # allocate roughly proportional, at least 1 per class
    weights = counts / counts.sum()
    take = np.maximum(1, np.floor(weights * cap).astype(int))
    # make sure we hit the cap exactly
    while take.sum() < min(cap, len(y)):
        # add one to the class with largest fractional remainder
        remainders = weights * cap - np.floor(weights * cap)
        j = np.argmax(remainders)
        take[j] += 1
    # sample
    idxs = []
    for c, k in zip(classes, take):
        cand = np.where(y == c)[0]
        if k >= len(cand):
            idxs.extend(cand.tolist())
        else:
            idxs.extend(rng.choice(cand, size=int(k), replace=False).tolist())
    return np.array(sorted(idxs), dtype=int)

# ---------- main objective used by Optuna ----------


def rf_kta_objective_sklearn(trial,
                             X: np.ndarray,
                             y: np.ndarray,
                             *,
                             subsample_cap: int = 6000,
                             mds_dim: int = 50,
                             n_jobs_prox: int = None,
                             random_state: int = 42):
    """
    Optuna objective: fit RF on a stratified subsample, build RF distance,
    do randomized-MDS -> sparsegram, compute KTA loss.
    Minimize KTA loss (0 is perfect).
    """

    # --- subsample (stratified) for speed ---
    idx = stratified_cap_indices(y, cap=min(subsample_cap, len(y)),
                                 rng=random_state)
    Xs = X[idx]
    ys = y[idx]

    # --- search space (keeping it narrow for speed/robustness) ---
    n_estimators = trial.suggest_int("n_estimators", 120, 400)
    max_depth = trial.suggest_int("max_depth", 12, 40)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
    # mix of categorical + numeric options keeps trees cheap on 175k features
    max_features_choice = trial.suggest_categorical(
        "max_features_choice", ["sqrt", "log2", 0.05, 0.1, 0.2, 0.4, 0.8]
    )
    if isinstance(max_features_choice, float):
        max_features = max_features_choice
    else:
        max_features = max_features_choice

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        oob_score=True,   # we’ll use this for lightweight pruning
        n_jobs=-1,
        random_state=random_state,
        warm_start=False
    )

    t0 = time.time()
    rf.fit(Xs, ys)
    t_fit = time.time() - t0

    # lightweight pruning: if oob accuracy is really bad, prune
    if rf.oob_score_ is not None:
        # higher oob -> lower loss; report an intermediate value
        trial.report(1.0 - float(rf.oob_score_), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # leaves -> RF distance (parallel over trees)
    t1 = time.time()
    leaves = rf.apply(Xs)  # (n_sub, n_trees)
    D = proximity_from_leaves_parallel(leaves,
                                       n_jobs=n_jobs_prox,
                                       return_distance=True,
                                       dtype=np.float32)
    t_prox = time.time() - t1

    # randomized MDS -> sparsegram
    t2 = time.time()
    emb = compute_mds_embedding_from_distance(D, size_embedding=mds_dim,
                                              random_state=random_state)
    K = build_sparsegram_from_embedding(emb)  # float32
    t_mds = time.time() - t2

    # KTA loss
    Lc, Lc_norm = make_label_kernel(ys)
    loss = kta_score(K, Lc, Lc_norm)

    # record component times for logs
    trial.set_user_attr("t_fit", t_fit)
    trial.set_user_attr("t_prox", t_prox)
    trial.set_user_attr("t_mds", t_mds)
    return float(loss)


def tune_rf_params_sklearn(X: np.ndarray,
                           y: np.ndarray,
                           *,
                           n_trials: int = 25,
                           subsample_cap: int = 6000,
                           mds_dim: int = 50,
                           random_state: int = 42,
                           n_jobs_prox: int = None,
                           study_name: str = "RF-KTA-Tuning"):
    """
    Run Optuna on the KTA objective above. Returns (best_params, study).
    """
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=random_state, n_startup_trials=5)

    def _obj(trial):
        return rf_kta_objective_sklearn(
            trial, X, y,
            subsample_cap=subsample_cap,
            mds_dim=mds_dim,
            n_jobs_prox=n_jobs_prox,
            random_state=random_state
        )

    study = optuna.create_study(direction="minimize",
                                pruner=pruner,
                                sampler=sampler,
                                study_name=study_name)
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)

    # build params dict for sklearn RF
    p = study.best_trial.params
    best_params = dict(
        n_estimators=int(p["n_estimators"]),
        max_depth=int(p["max_depth"]),
        min_samples_leaf=int(p["min_samples_leaf"]),
        max_features=p["max_features_choice"],  # may be 'sqrt'/'log2'/float
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state
    )
    return best_params, study


def select_hybrid_balanced_medoids(affinity, Y, target_total, random_state=0, verbose=True):
    labels = np.asarray(Y)
    unique_labels = np.unique(labels)
    n_samples = labels.shape[0]
    if target_total > n_samples:
        raise ValueError(
            f"target_total ({target_total}) exceeds number of samples ({n_samples}).")
    if verbose:
        print('Y', labels)
        print('unique', unique_labels)

    label_to_indices = {label: np.flatnonzero(
        labels == label) for label in unique_labels}
    n_labels = len(unique_labels)
    target_per_label = target_total // n_labels

    assignment_dict = {label: (len(label_to_indices[label]), min(
        len(label_to_indices[label]), target_per_label)) for label in unique_labels}  # {label: (available, assigned)}
    print(assignment_dict)

    small_labels = set()

    def update_small_labels():
        nonlocal small_labels
        for label, tupl in assignment_dict.items():
            if tupl[0] == tupl[1]:
                small_labels.add(label)
    update_small_labels()

    total_assigned = 0

    def update_assigned():
        nonlocal total_assigned
        total_assigned = sum([_[1] for _ in assignment_dict.values()])
    update_assigned()

    increment = False
    while total_assigned < target_total:
        available_labels = set(
            [_ for _ in unique_labels if _ not in small_labels])
        remaining_per_label = (
            target_total - total_assigned) // (len(available_labels))

        if remaining_per_label == 0:
            increment = True
            break

        for label in available_labels:
            assert label in assignment_dict
            label_avail, label_assigned = assignment_dict[label]
            assignment_dict[label] = (label_avail, min(
                label_avail, label_assigned + remaining_per_label))

        previously_assigned = total_assigned
        update_assigned()
        if previously_assigned == total_assigned:
            raise ValueError(
                '[select_hybrid_balanced_medoids] Critical error: not updating assignments')
        update_small_labels()

    if increment:
        for label in available_labels:
            assert label in assignment_dict
            assignment_dict[label] = (
                assignment_dict[label][0], assignment_dict[label][1] + 1)
            update_assigned()
            if total_assigned == target_total:
                break

    if verbose:
        print('assignments:\n', assignment_dict)

    # Now selecting indices
    selected_indices = []
    for label in small_labels:
        assert label in label_to_indices
        assert label in assignment_dict
        assert assignment_dict[label][0] == assignment_dict[label][1]
        selected_indices.extend(label_to_indices[label].tolist())

    remaining_labels = set([_ for _ in unique_labels if _ not in small_labels])
    for label in remaining_labels:
        assert label in label_to_indices
        assert label in assignment_dict
        assert assignment_dict[label][0] > assignment_dict[label][1]
        idx = label_to_indices[label]
        final_assigned = assignment_dict[label][1]
        # square RF distance block
        D_sub = affinity[np.ix_(idx, idx)]
        k = max(1, min(final_assigned, len(idx)))      # safety
        km = KMedoids(n_clusters=k, metric='precomputed',
                      init='heuristic',
                      random_state=random_state,
                      max_iter=300).fit(D_sub)
        selected_indices.extend(idx[km.medoid_indices_].tolist())

    if verbose:
        print('\n~#$   Medoids   $#~:\n', selected_indices)
    # Preserve original order
    ordered = np.sort(np.asarray(selected_indices))
    return ordered


def convert_least_squares(distance, Y, clstr, num_clstr, size_embedding=50):
    print('convert least squares:')
    st = time.time()

    print('doing randomized MDS')
    embedding = mds_full_randomized(distance, size_embedding)
    affinity_transformed = csr_matrix(embedding)
    print(f'MDS time: {time.time() - st}')

    if clstr and affinity_transformed.shape[0] > num_clstr:
        print('clustering samples based on tree-leaf-predictions representation')
        medoid_inds = select_hybrid_balanced_medoids(distance, Y, num_clstr)
        affinity_transformed = affinity_transformed[medoid_inds]
        print('medoid inds shape:', medoid_inds.shape)
        print('affinity shape:', end=' ')
        print(affinity_transformed.shape)
    else:
        medoid_inds = np.arange(affinity_transformed.shape[0])

    affinity_transformed = csr_matrix(affinity_transformed)
    sparsegram = affinity_transformed @ affinity_transformed.T
    print('sparsegram shape:', end=' ')
    print(sparsegram.shape)

    signal = csr_matrix(sparsegram.reshape(
        (sparsegram.shape[0]**2, 1), order='F'))
    signal = csc_matrix(signal)

    print('signal created.')
    print('signal shape:', signal.shape)

    return signal, sparsegram, medoid_inds


def _score_chunk_return_arrays(chunk_indices, sub_chunk, M, row_sq_norms):
    """
    Compute scores for a chunk and return arrays aligned with chunk_indices.
    Returns (chunk_indices, nums_array, denoms_array, scores_array)
    All arrays use float64 and preserve the same index order as chunk_indices.
    """
    n_local = len(chunk_indices)
    nums = np.zeros(n_local, dtype=np.float64)
    denoms = np.zeros(n_local, dtype=np.float64)
    scores = np.full(n_local, -np.inf, dtype=np.float64)

    for local_i, global_i in enumerate(chunk_indices):
        u = sub_chunk.getrow(local_i).T  # column (n_samples, 1), sparse
        if u.nnz == 0:
            # leave score = -inf
            continue

        # exact same matvec order as serial: tmp = M @ u; then num = u.T @ tmp
        tmp = M.dot(u)

        # get scalar robustly from 1x1 sparse result
        tmpmat = u.T.dot(tmp)
        if getattr(tmpmat, "nnz", 0):
            num = float(tmpmat.data[0])
        else:
            num = 0.0

        denom = float(row_sq_norms[global_i])  # ensure float64

        if denom == 0.0:
            # avoid division by zero
            scores[local_i] = -np.inf
        else:
            scores[local_i] = num / denom

        nums[local_i] = num
        denoms[local_i] = denom

    return chunk_indices, nums, denoms, scores


def matching_pursuit_sequential_parallel(signal, sub_mags, s, n_jobs=None):
    """
    Parallel matching pursuit that is exact-equivalent to the serial implementation.
    - signal: sparse vector (n_samples**2, 1), Fortran-order vec(M)
    - sub_mags: csr_matrix, shape (n_bases, n_samples)  <-- each row is u_i
    - s: number of atoms to select
    - n_jobs: number of threads (defaults to min(cpu_count(), n_bases))

    Returns: indices, coefs.
    """
    if not isinstance(sub_mags, csr_matrix):
        sub_mags = csr_matrix(sub_mags)

    # ensure float64 to avoid dtype differences
    if sub_mags.dtype != np.float64:
        sub_mags = sub_mags.astype(np.float64)

    n_bases, n_samples = sub_mags.shape

    # reshape signal -> residual matrix M (CSR)
    R_vec = signal.toarray().ravel()
    M = csr_matrix(R_vec.reshape(
        (n_samples, n_samples), order='F'), dtype=np.float64)

    # precompute denom (||u||^2)
    row_sq_norms = np.asarray(sub_mags.power(2).sum(
        axis=1)).reshape(-1).astype(np.float64)

    # chunking strategy (deterministic)
    if n_jobs is None:
        n_jobs = min(cpu_count(), n_bases)
    else:
        n_jobs = min(n_jobs, n_bases)
    all_indices = np.arange(n_bases)
    chunks = [c for c in np.array_split(all_indices, n_jobs) if len(c) > 0]
    sub_chunks = [sub_mags[chunk] for chunk in chunks]

    indices = []
    coefs = []

    # reuse thread pool across iterations for speed
    with ThreadPoolExecutor(max_workers=n_jobs) as exc:
        for _ in range(s):
            # submit scoring jobs for each chunk
            futures = [
                exc.submit(_score_chunk_return_arrays,
                           chunks[i], sub_chunks[i], M, row_sq_norms)
                for i in range(len(chunks))
            ]
            # collect chunk outputs and assemble full arrays
            # create arrays to hold full results in global order
            nums_full = np.zeros(n_bases, dtype=np.float64)
            denoms_full = np.zeros(n_bases, dtype=np.float64)
            scores_full = np.full(n_bases, -np.inf, dtype=np.float64)

            for f in futures:
                chunk_indices, nums, denoms, scores = f.result()
                # chunk_indices is a numpy array of global indices; we place results accordingly
                scores_full[chunk_indices] = scores
                nums_full[chunk_indices] = nums
                denoms_full[chunk_indices] = denoms

            # now pick global best (np.argmax picks first occurrence on ties)
            best_idx = int(np.argmax(scores_full))
            best_score = float(scores_full[best_idx])

            # if best_score is -inf then no candidate left
            if not np.isfinite(best_score) or best_score == -np.inf:
                break

            best_denom = float(denoms_full[best_idx])

            # reconstruct the chosen atom and update residual exactly as serial
            u_chosen = sub_mags.getrow(best_idx).toarray().ravel()
            atom_vec = np.kron(u_chosen, u_chosen)
            atom_norm = best_denom
            maxproj = best_score

            coef = maxproj / atom_norm
            coefs.append(coef)
            indices.append(best_idx)

            # Update residual vector and M for next iteration
            R_vec = R_vec - (maxproj / atom_norm) * atom_vec
            M = csr_matrix(R_vec.reshape(
                (n_samples, n_samples), order='F'), dtype=np.float64)

    return indices, coefs


def diag_impo(mags, coordinates, coefficients):
    """ Reconstruct the diagonal arrray
        from the adaptive haarlike model.

        Parameters:
        mags
        coordinates
        coefficients

        Returns:
        coefs
    """

    assigned = set()
    coefs = np.zeros(mags.shape[0])

    for i in range(len(coordinates)):

        coord = coordinates[i]
        coef = coefficients[i]
        if abs(coef) > 1e10:
            print(
                f"Warning: suspiciously large coefficient {coef} at coord {coord}")

        if not coord in assigned:
            assigned.add(coord)
            coefs[coord] = coef

    return coefs


def adaptive(haar_basis, biom_table, label, tree, meta, s, cluster_affinity, num_clstr, tune):
    """ shl: sparse haar like coordinates
        biom_table_data: biom_table, 
        label: str, column in meta to use 
        biom_table: skbio.TreeNode
        meta: pd.dataframe
        s=5: Number of important nodes to find

        returns dic, rfgram, coordinates, coefs, Y, dic, new_diag
    """

    # Control Vars
    print('running with cluster_affinity=', cluster_affinity, sep='')

    X, Y, mags, dic = preprocess(label, biom_table, haar_basis, meta, tree)
    print('preprocessing done.')

    # Tune RF hyperparams fast on a stratified subset - 25 trials
    if tune:
        st = time.time()
        print('tuning RF hyperparams...')
        best_params, study = tune_rf_params_sklearn(
            X, np.asarray(Y),
            n_trials=35,
            subsample_cap=min(num_clstr, len(Y)),
            mds_dim=50,
            random_state=42,
            n_jobs_prox=None
        )
        print(f'tuning done in {time.time() - st} seconds. Best study:',
              study.best_trial, 'Best params:', best_params, sep='\n')
        avg_fit = np.mean([t.user_attrs["t_fit"]
                           for t in study.trials if t.value is not None])
        avg_prox = np.mean([t.user_attrs["t_prox"]
                            for t in study.trials if t.value is not None])
        avg_mds = np.mean([t.user_attrs["t_mds"]
                           for t in study.trials if t.value is not None])
        print(
            f"Avg per-trial: fit={avg_fit:.2f}s, prox={avg_prox:.2f}s, mds={avg_mds:.2f}s")
        print(
            f"Estimated 25-trial wall-time: {(25*(avg_fit+avg_prox+avg_mds))/60:.1f} minutes")

        clf = RandomForestClassifier(**best_params)
        clf.fit(X, Y)
    else:
        clf = RandomForestClassifier(n_estimators=500,
                                     bootstrap=True,
                                     min_samples_leaf=1)
        clf.fit(X, Y)
    print('RF training done.')
    

    from sklearn.metrics import classification_report

    # 1. Quick accuracy score
    train_accuracy = clf.score(X, Y)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    # 2. Detailed training stats
    print("Training Classification Report:\n", classification_report(Y, clf.predict(X)))


    leaves = clf.apply(X)
    rf_distance = proximity_from_leaves_parallel(
        leaves, n_jobs=None, return_distance=True, dtype=np.float32)
    print('affinity generated.')

    signal, rfgram, medoid_indices = convert_least_squares(
        rf_distance,
        Y,
        cluster_affinity, num_clstr)

    Y = Y.iloc[medoid_indices].reset_index(drop=True)
    mags = mags[:, medoid_indices]

    # Drop synthetic “root-split” rows so rows map 1:1 to non-tips
    nontips_count = sum(1 for _ in tree.postorder() if not _.is_tip())
    if mags.shape[0] > nontips_count:
        mags = mags[:nontips_count, :]

    st = time.time()
    coordinates, coefs = matching_pursuit_sequential_parallel(signal, mags, s)
    print(coordinates)
    print(coefs)
    print(f'signal estimated with Haar coefs in {time.time() - st} seconds.')

    new_diag = diag_impo(mags, coordinates, coefs)
    print('diag created.')

    return rfgram, coordinates, coefs, Y, dic, new_diag, mags


def reconstruct_coord(coefs, coordinates, mags, s):
    """ This function is for reconstructing and obtaining a vector
        used in plotting the biplots. The resulting coord is of shape
        n_internal_nodes_found (default 5) x n_samples (?) """

    coord = np.sqrt(coefs[0]) * mags[coordinates[0], :].todense()
    for i in range(1, s):
        nodei = np.sqrt(coefs[i]) * mags[coordinates[i], :].todense()
        coord = np.vstack((coord, nodei))
    return coord


def reconstruct(coordinates, mags, s, coefs):
    """ This function reconstructs the matrix using weights of the
        top coordinates and the modmags matrix, which is the projection
        onto the shl matrix. 

        coordinates: 
        mags: csr_matrix
        s: int - number of coordinates to find (default 5)
        coefs:

        returns 
        outer: numpy.matrix of shape s x n_samples
    """
    d, n = mags.get_shape()  # d=n internal nodes; n=n samples
    outer = np.zeros((n, n))
    for i in range(s):
        temp = mags[coordinates[i], :].todense()
        out = np.outer(temp, temp)
        outer = outer + coefs[i]*out
    return outer


# PLOTTERS

def new_biplot3d(s, coefs, coordinates, mags, y,
                 labeltype, dic, k, n, save, path):

    Z = np.transpose(reconstruct_coord(coefs, coordinates, mags, s))
    pca = PCA()
    x_new = pca.fit_transform(np.asarray(Z))
    score = x_new[:, 0:3]
    coeff = np.transpose(pca.components_[0:3, :])
    xs = score[:, 0]
    ys = score[:, 1]
    zs = score[:, 2]
    scalex = 1.0
    scaley = 1.0
    scalez = 1.0
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(projection='3d')

    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels = [inv_map[i] for i in y]

    if labeltype == 'classification':
        n_classes = len(np.unique(y))
        if n_classes == 2:
            colors = cm.tab10(y/(2*max(y)))
        else:
            colors = cm.tab10(y/max(y))

        for xs, ys, zs, c, label in zip(xs, ys, zs, colors, labels):
            ax.scatter(xs*scalex, ys*scaley, zs*scalez,
                       facecolors="None", edgecolors=c, label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 7})

    elif labeltype == 'regression':
        c = cm.viridis(y/max(y))
        p = ax.scatter(xs*scalex, ys*scaley, zs*scalez, c=y, cmap='viridis')
        plt.colorbar(p, pad=0.15)

    for i in range(n):
        ax.quiver(
            0, 0, 0,  # starting point of vector
            1.15*coeff[i, 0], 1.15*coeff[i, 1], 1.15 *
            coeff[i, 2],  # vector directi
            color='black', alpha=.7, lw=2
        )
        ax.text(
            coeff[i, 0] * 1.25, coeff[i, 1] * 1.25, coeff[i, 2] * 1.25,
            coordinates[i],
            color='black', ha='center', va='center'
        )

    rat0 = np.around(pca.explained_variance_ratio_[0] * 100, 2)
    rat1 = np.around(pca.explained_variance_ratio_[1] * 100, 2)
    rat2 = np.around(pca.explained_variance_ratio_[2] * 100, 2)
    xlab = "PC1" + ' ' + str(rat0) + '%'
    ylab = "PC2" + ' ' + str(rat1) + '%'
    zlab = "PC3" + ' ' + str(rat2) + '%'
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    ax.set_zlabel(zlab)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 6, box.height])

    plt.title('PCA Biplot')
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor('white')
    ax.dist = 12
    fig.tight_layout(pad=2)

    if save == True:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def new_biplot3dnormalized(s, coefs, coordinates, mags, y,
                           labeltype, dic, k, n, save, path):

    Z = np.transpose(reconstruct_coord(coefs, coordinates, mags, s))
    scaler = StandardScaler()
    scaler.fit(np.asarray(Z))
    Z = scaler.transform(np.asarray(Z))
    pca = PCA()
    x_new = pca.fit_transform(Z)
    score = x_new[:, 0:3]
    coeff = np.transpose(pca.components_[0:3, :])
    print('~#$ coeffs for the black arrows:\n', coeff)
    xs = score[:, 0]
    ys = score[:, 1]
    zs = score[:, 2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    scalez = 1.0 / (zs.max() - zs.min())
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(projection='3d')

    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels = [inv_map[i] for i in y]
    if labeltype == 'classification':
        n_classes = len(np.unique(y))
        if n_classes == 2:
            colors = cm.tab10(y / (2*max(y)))
        else:
            colors = cm.tab10(y / max(y))
        for xs, ys, zs, c, label in zip(xs, ys, zs, colors, labels):

            ax.scatter(xs * scalex, ys * scaley, zs * scalez,
                       facecolors="None",
                       edgecolors=c, label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 7})

    elif labeltype == 'regression':
        c = plt.cm.viridis(y / max(y))
        p = ax.scatter(xs*scalex, ys*scaley, zs*scalez,
                       c=y, cmap='viridis')
        plt.colorbar(p, pad=0.15)

    for i in range(n):
        ax.quiver(
            0, 0, 0,  # starting point of vector
            1.15*coeff[i, 0], 1.15*coeff[i, 1], 1.15 *
            coeff[i, 2],  # vector directi
            color='black', alpha=.7, lw=2
        )
        ax.text(coeff[i, 0] * 1.25, coeff[i, 1] * 1.25, coeff[i, 2] * 1.25,
                coordinates[i], color='black', ha='center', va='center')

    rat0 = np.around(pca.explained_variance_ratio_[0]*100, 2)
    rat1 = np.around(pca.explained_variance_ratio_[1]*100, 2)
    rat2 = np.around(pca.explained_variance_ratio_[2]*100, 2)
    plt.xlabel("PC1" + ' ' + str(rat0) + '%')
    plt.ylabel("PC2" + ' ' + str(rat1) + '%')
    ax.set_zlabel("PC3" + ' ' + str(rat2) + '%')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 6, box.height])
    plt.title('PCA Biplot Normalized')
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor('white')
    ax.dist = 12
    fig.tight_layout(pad=2)

    if save == True:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def _get_correlation(sorted_rfgram, sorted_reconstructed):

    matrix1 = np.array(sorted_rfgram.todense())
    matrix2 = sorted_reconstructed
    corr = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
    return corr


def rfgram_plot(rfgram, coordinates, modmags, s, coefs, Y, dic,
                save, path):
    """ rfgram is the representation of the dataset using a trained rf 
        model, which works by training an RF on all data poins then forming
        an rf gram matrix. this matrix represents how similar two points are 
        to eachother by seeing how similarly they are grouped by the decision 
        trees in the random forest. the reconstructed plot takes our learned
        haar-like weight vector, which only uses 5 internal nodes to re-
        represent the data set. if the two plots look similar, this means
        that our learned weight vector w does a good job of recapitulating the
        original random forest in a far smaller feature space."""

    groups = list(Y)
    inv_dic = {v: k for k, v in dic.items()}

    reconstructed = reconstruct(coordinates, modmags, s, coefs)
    sorted_indices = np.argsort(groups)
    sorted_groups = np.array(groups)[sorted_indices]
    group_names = [inv_dic[x] for x in sorted_groups]  # Name for each group

    sorted_rfgram = rfgram[:, sorted_indices][sorted_indices, :]

    sorted_reconstructed = reconstructed[:, sorted_indices][sorted_indices, :]
    # reconstructed = reconstruct(coordinates, modmags, s, coefs)

    # Create figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Heatmap 1
    vmin, vmax = np.quantile(sorted_rfgram.data, [0.01, 0.99])
    axes[0].imshow(sorted_rfgram.todense(), vmin=vmin,
                   vmax=vmax, cmap='binary')
    axes[0].xaxis.set_visible(False)

    # Heatmap 2
    vmin, vmax = np.quantile(sorted_reconstructed, [0.01, 0.99])
    axes[1].imshow(sorted_reconstructed, vmin=vmin, vmax=vmax, cmap='binary')
    axes[1].xaxis.set_visible(False)

    # Create new axes below each heatmap for the group bars
    # Adjust the position and size to fit the bars
    group_bar_height = 0.05  # Height of the group bars
    label_padding = 0.1  # Space between the bar and the label
    ax_group_bar1 = fig.add_axes([axes[0].get_position().x0, axes[0].get_position(
    ).y0, axes[0].get_position().width, group_bar_height], frameon=False)
    ax_group_bar2 = fig.add_axes([axes[1].get_position().x0, axes[1].get_position(
    ).y0, axes[1].get_position().width, group_bar_height], frameon=False)

    # Plot the group bars
    ax_group_bar1.imshow([sorted_groups], aspect='auto', cmap='Set1')
    ax_group_bar2.imshow([sorted_groups], aspect='auto', cmap='Set1')

    # Hide y-axis and ticks for the group bars
    ax_group_bar1.set_yticks([])
    ax_group_bar1.set_xticks([])
    ax_group_bar2.set_yticks([])
    ax_group_bar2.set_xticks([])

    # Determine where each group starts and ends
    unique_groups, start_idx = np.unique(sorted_groups, return_index=True)
    end_idx = np.append(start_idx[1:], len(sorted_groups))

    # Add vertical group labels centered below each group segment
    for i, group in enumerate(unique_groups):

        # Center position for the group label
        center = (start_idx[i] + end_idx[i] - 1) / 2

        # Adjust the vertical position of the labels to avoid overlap
        label_y = 1
        ax_group_bar1.text(
            center, label_y, group_names[start_idx[i]], ha='center', va='top', rotation=90, fontsize=10)
        ax_group_bar2.text(
            center, label_y, group_names[start_idx[i]], ha='center', va='top', rotation=90, fontsize=10)

    # Adjust layout to fit the labels
    # Increase bottom margin to accommodate labels
    plt.subplots_adjust(bottom=0.2)

    corr_coeff = _get_correlation(sorted_rfgram, sorted_reconstructed)
    axes[1].text(1.1, 0.5, f'Pearson Correlation: {corr_coeff:.2f}',
                 fontsize=12, va='center', ha='left',
                 transform=axes[1].transAxes)

    if save == True:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def compute_ilr_for_nodes(tree, table_df, node_indices, pseudocount=1e-6, scale='ilr'):
    """
    table_df: pd.DataFrame of counts/abund (samples x tips), columns are tip names (OTU ids)
    node_indices: iterable of internal-node row indices (same indexing as your 'mags' rows)
    scale: 'ilr' for sqrt(|L||R|/(|L|+|R|)) scaling, or 'none' for plain log-ratio
    returns: array of shape (len(node_indices), n_samples)
    """
    # reorder columns to tree tip order -> tips x samples
    tip_names = [t.name for t in tree.tips()]
    X = table_df.reindex(columns=tip_names, fill_value=0).T.values
    # closure to compositional
    X = X / (X.sum(axis=0, keepdims=True) + 1e-15)
    X = X + pseudocount

    # index map for tips
    tip_idx = {name: i for i, name in enumerate(tip_names)}

    # internal nodes in postorder (index must match your mags row order)
    internal_nodes = [n for n in tree.postorder() if not n.is_tip()]
    out = []

    for node_id in node_indices:
        node = internal_nodes[node_id]
        kids = list(node.children)
        if len(kids) != 2:
            raise ValueError(
                'Tree must be strictly bifurcating for ILR balances.')

        L_leaves = [tip_idx[t.name] for t in kids[0].tips()]
        R_leaves = [tip_idx[t.name] for t in kids[1].tips()]

        L = X[L_leaves, :]
        R = X[R_leaves, :]

        # mean log (i.e., log geometric mean)
        mlogL = np.mean(np.log(L), axis=0)
        mlogR = np.mean(np.log(R), axis=0)
        bal = mlogL - mlogR

        if scale == 'ilr':
            g = np.sqrt(len(L_leaves) * len(R_leaves) /
                        (len(L_leaves) + len(R_leaves)))
            bal = g * bal

        out.append(bal)

    return np.vstack(out)  # (len(node_indices), n_samples)



def boxplot_plotter(mags, y, indices, dic, xlabels, save, path, coefs=None, scale='sqrt'):
    """
    Produces a two-column figure for each selected node:
      Left column  — prevalence bar chart (fraction of samples with node present)
      Right column — conditional boxplot of wavelet values for present samples only
 
    Shared column headers are printed once at the top.
    'Node XXXXX' appears as the y-axis label of each left panel.
 
    scale: 'none' | 'sqrt' | 'coef'  (kept for API compatibility; no longer
           applied to the conditional boxplot — raw wavelet values are shown)
    """
    import matplotlib.gridspec as gridspec
    from scipy.stats import mannwhitneyu
    from scipy.sparse import issparse
 
    n_panels = len(indices)
 
    # ── class ordering and colours (matches original pipeline) ───────────
    # dic: {class_name: int_code}
    # y:   array of int codes
    y = np.asarray(y)
    class_names = sorted(dic.keys(), key=lambda k: dic[k])
    class_codes = np.array([dic[k] for k in class_names], dtype=float)
    denom       = 2 * class_codes.max() if len(class_codes) == 2 else class_codes.max()
    colors      = cm.tab10(class_codes / denom)
 
    # ── figure geometry ───────────────────────────────────────────────────
    panel_h  = 2.8
    header_h = 0.45
    fig_w    = 13
    fig_h    = header_h + n_panels * panel_h
 
    fig = plt.figure(figsize=(fig_w, fig_h))
 
    outer = gridspec.GridSpec(
        n_panels + 1, 1,
        figure        = fig,
        height_ratios = [header_h] + [panel_h] * n_panels,
        hspace        = 0.35,
        top           = 0.97,
        bottom        = 0.06,
        left          = 0.07,
        right         = 0.97,
    )
 
    # ── shared column headers ─────────────────────────────────────────────
    header_grid = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec = outer[0],
        width_ratios = [1, 2.2],
        wspace       = 0.35,
    )
    ax_hl = fig.add_subplot(header_grid[0])
    ax_hr = fig.add_subplot(header_grid[1])
    for ax in (ax_hl, ax_hr):
        ax.set_axis_off()
 
    ax_hl.text(
        0.5, 0.5,
        "Prevalence\n(fraction of samples with node present)",
        transform=ax_hl.transAxes,
        ha="center", va="center",
        fontsize=10, fontweight="bold",
    )
    ax_hr.text(
        0.5, 0.5,
        "Wavelet value  (present samples only)\n"
        ">0: left-child clade dominates     "
        "<0: right-child clade dominates",
        transform=ax_hr.transAxes,
        ha="center", va="center",
        fontsize=10, fontweight="bold",
    )
 
    # ── helper: presence mask from sparse or dense row ───────────────────
    def _presence_mask(mags, node_idx):
        row = mags[node_idx, :]
        if issparse(row):
            row_csr = row.tocsr()
            present = np.zeros(mags.shape[1], dtype=bool)
            present[row_csr.indices] = True
        else:
            arr     = np.asarray(row).ravel()
            present = arr != 0.0
        return present
 
    # ── helper: extract dense row ─────────────────────────────────────────
    def _get_vals(mags, node_idx):
        row = mags[node_idx, :]
        if hasattr(row, 'A1'):
            return row.A1.astype(np.float64)
        elif hasattr(row, 'todense'):
            return np.asarray(row.todense()).ravel().astype(np.float64)
        else:
            return np.asarray(row).ravel().astype(np.float64)
 
    # ── data rows ─────────────────────────────────────────────────────────
    for k in range(n_panels):
        node_idx     = indices[k]
        wavelet_vals = _get_vals(mags, node_idx)
        present      = _presence_mask(mags, node_idx)
 
        prevalences         = []
        cond_vals           = []
        n_per_class         = []
        n_present_per_class = []
 
        for name in class_names:
            code             = dic[name]
            class_mask       = y == code
            present_in_class = class_mask & present
            n_class          = int(class_mask.sum())
            n_present        = int(present_in_class.sum())
 
            prevalences.append(n_present / n_class if n_class > 0 else 0.0)
            cond_vals.append(wavelet_vals[present_in_class])
            n_per_class.append(n_class)
            n_present_per_class.append(n_present)
 
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2,
            subplot_spec = outer[k + 1],
            width_ratios = [1, 2.2],
            wspace       = 0.35,
        )
 
        # ── left: prevalence bar ─────────────────────────────────────────
        ax_prev = fig.add_subplot(inner[0])
        bars = ax_prev.bar(class_names, prevalences,
                           color=colors, alpha=0.85, width=0.5)
        ax_prev.set_ylim(0, 1.18)
        ax_prev.tick_params(labelsize=8)
        ax_prev.axhline(0.5, color="grey", lw=0.7, ls=":", zorder=0)
        ax_prev.set_ylabel(f"Node {node_idx}", fontsize=9,
                           fontweight="bold", labelpad=6)
 
        for bar, pv, n_c, n_p in zip(bars, prevalences,
                                      n_per_class, n_present_per_class):
            ax_prev.text(
                bar.get_x() + bar.get_width() / 2,
                pv + 0.05,
                f"{pv:.0%}\n({n_p}/{n_c})",
                ha="center", va="bottom", fontsize=7,
            )
 
        # ── right: conditional boxplot ───────────────────────────────────
        ax_box = fig.add_subplot(inner[1])
 
        if any(len(v) > 0 for v in cond_vals):
            bp = ax_box.boxplot(
                cond_vals,
                vert=True,
                patch_artist=True,
                labels=class_names,
                medianprops  = dict(color="black", linewidth=1.5),
                whiskerprops = dict(color="black"),
                capprops     = dict(color="black"),
                flierprops   = dict(marker=".", markersize=3,
                                   color="black", alpha=0.4),
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
 
        ax_box.axhline(0, color="grey", lw=0.8, ls="--", zorder=0)
        ax_box.tick_params(labelsize=8)
 
        # Mann-Whitney annotation
        if len(class_names) == 2 and len(cond_vals[0]) > 1 and len(cond_vals[1]) > 1:
            _, p     = mannwhitneyu(cond_vals[0], cond_vals[1],
                                    alternative="two-sided")
            med_diff = np.median(cond_vals[0]) - np.median(cond_vals[1])
            p_str    = f"p = {p:.3g}" if p >= 0.001 else "p < 0.001"
            ax_box.text(
                0.98, 0.97,
                f"Δmedian = {med_diff:.4f}  |  {p_str}",
                transform=ax_box.transAxes,
                ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="none", alpha=0.8),
            )
 
    if save:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.close(fig)




def _boxplot_plotter(mags, y, indices, dic, xlabels, save, path, coefs=None, scale='sqrt'):
    """
    scale: 'none' | 'sqrt' | 'coef'
      - 'sqrt' -> multiply node k by sqrt(coefs[k])  (matches biplot)
      - 'coef' -> multiply by coefs[k]
      - 'none' -> raw mags (prev/default behavior)
    """

    n_panels = len(indices)
    fig, ax = plt.subplots(n_panels, sharex=True, figsize=(14, 2.5 * n_panels))
    if n_panels == 1:
        ax = np.array([ax])

    boxlabels = list(dic.keys())
    inv_map = {v: k for k, v in dic.items()}
    datalabels = [inv_map[i] for i in y]

    for k in range(n_panels):
        alldata = []
        for i, lbl in enumerate(boxlabels):
            dataindices = [j for j, x in enumerate(datalabels) if x == lbl]
            if not dataindices:
                alldata.append(np.array([]))
                continue

            vals = mags[indices[k], dataindices]
            vals = vals.A1 if hasattr(vals, 'A1') else (np.asarray(vals.todense()).ravel()
                                                        if hasattr(vals, 'todense') else np.asarray(vals).ravel())

            if coefs is not None and scale != 'none':
                w = np.sqrt(coefs[k]) if scale == 'sqrt' else coefs[k]
                vals = w * vals

            alldata.append(vals)

        temp = ax[k].boxplot(alldata, vert=True,
                             patch_artist=True, labels=xlabels)
        ax[k].set_title('Haar-like Node ' + str(indices[k]))
        plt.setp(temp['whiskers'], color='black')
        plt.setp(temp['medians'], color='black')
        plt.setp(temp['fliers'], color='black', marker='')

        nums = np.array(list(dic.values()))
        denom = 2 * nums.max() if len(np.unique(y)) == 2 else nums.max()
        colors = cm.tab10(nums / denom)
        for patch, color in zip(temp['boxes'], colors):
            patch.set_facecolor(color)

    if save:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def make_plots(adhld_results, modmags, path, s, k, n):

    # unpack reslts
    rfgram, coordinates, coefs, Y, dic, _, _ = adhld_results

    # define paths to save all 4 plots
    path1 = os.path.join(path, 'rfgram.svg')
    path2 = os.path.join(path, 'boxplot.svg')
    path3 = os.path.join(path, 'biplot.svg')
    path4 = os.path.join(path, 'biplotn.svg')

    # RF GRAM MATRIX
    rfgram_plot(rfgram, coordinates, modmags, s, coefs, Y, dic,
                save=True, path=path1)

    # BOXPLOTS OF NODES
    boxplot_plotter(modmags, Y.values, coordinates[0:s], dic,
                    dic.keys(), True, path2, coefs)

    # BIPLOT
    new_biplot3d(s, coefs, coordinates, modmags, Y,
                 'classification', dic, k, n,
                 save=True, path=path3)

    new_biplot3dnormalized(s, coefs, coordinates, modmags, Y,
                           'classification', dic, k, n,
                           save=True, path=path4)
