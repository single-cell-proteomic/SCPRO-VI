from anndata._io.specs import methods
import scanpy as sc
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def maps(adata, k = 200):
    print("MAP metric is calculating...")
    sc.pp.neighbors(adata, n_neighbors = k)
    def calculate_ap(i, labels, K_neighbors):
        correct_label = labels[i]
        precisions = [
            (sum(labels[K_neighbors[i, :j+1]] == correct_label) / (j+1))
            for j in range(K_neighbors.shape[1])
            if labels[K_neighbors[i, j]] == correct_label
        ]
        if precisions:
            return np.mean(precisions)
        else:
            return 0
    labels = adata.obs["cell_type"].values
    K_neighbors = adata.obsp['distances'].tocoo().toarray().argsort()[:, 1:k+1]

    AP_values = [
        calculate_ap(i, labels, K_neighbors)
        for i in range(adata.n_obs)
    ]

    return np.mean(AP_values)

def asw(adata, layer = None, use_rep = False, label = "cell_type"):
    print("ASW metric is calculating...")
    if layer != None:
      _adata = sc.AnnData(X=adata.obsm[layer], obs=adata.obs) 
    else:
      _adata = sc.AnnData(X=adata.X, obs=adata.obs)
    if use_rep:
      sc.pp.neighbors(_adata, use_rep="X")
    else:
      sc.pp.neighbors(_adata)
    # sc.tl.umap(_adata)
    # silhouette_vals = silhouette_samples(_adata.obsm['X_umap'], _adata.obs["cell_type"])
    silhouette_vals = silhouette_samples(_adata.X, _adata.obs[label])
    asw = np.mean(silhouette_vals)
    cell_type_asw = (asw + 1) / 2
    
    return cell_type_asw

def asw_per_cell_type(adata, layer = None):
    
    if layer != None:
      _adata = sc.AnnData(X=adata.obsm[layer], obs=adata.obs) 
    else:
      _adata = adata
    for cell_type in np.unique(_adata.obs["cell_type"]):
      print("ASW metric is calculating for {}...".format(cell_type))
      tmp_labels = [class_label if class_label == cell_type else "else" for class_label in _adata.obs["cell_type"]]
      sc.pp.neighbors(_adata)
      sc.tl.umap(_adata)
      silhouette_vals = silhouette_samples(_adata.obsm['X_umap'], tmp_labels)
      asw = np.mean(silhouette_vals)
      cell_type_asw = (asw + 1) / 2
      print(cell_type, ": ",cell_type_asw)
    return cell_type_asw

def asw_uns(adata, layer = None):
    print("ASW metric is calculating...")
    if layer != None:
      _adata = sc.AnnData(X=adata.obsm[layer], obs=adata.obs) 
    else:
      _adata = adata
    sc.pp.neighbors(_adata)
    sc.tl.umap(_adata)

    kmeans = KMeans(n_clusters=8, random_state=0).fit(_adata.obsm['X_pca']) 
    cluster_labels = kmeans.labels_

    silhouette_vals = silhouette_samples(_adata.obsm['X_umap'], cluster_labels)
    asw = np.mean(silhouette_vals)
    cell_type_asw = (asw + 1) / 2
    
    return cell_type_asw


def nc(adata, k = 200):
    print("NC metric is calculating...")
    n_cells = adata.n_obs
    
    nbrs_single = NearestNeighbors(n_neighbors = k, algorithm='auto', n_jobs = -1).fit(adata[:, adata.var["feature_type"] == "protein"].X)
    distances, indices_single = nbrs_single.kneighbors(adata[:, adata.var["feature_type"] == "protein"].X)

    # Calculate nearest neighbors for integrated data
    nbrs_integrated = NearestNeighbors(n_neighbors = k, algorithm='auto', n_jobs = -1).fit(adata.obsm["integrated"])
    distances, indices_integrated = nbrs_integrated.kneighbors(adata.obsm["integrated"])

    nc_score = 0

    # Calculate NC for each cell
    for i in range(n_cells):
        intersect_count = len(set(indices_single[i]) & set(indices_integrated[i]))
        nc_score += intersect_count / k

    # Average NC score across all cells
    nc_score_prot = nc_score / n_cells

    nbrs_single = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs = -1).fit(adata[:, adata.var["feature_type"] == "rna"].X)
    distances, indices_single = nbrs_single.kneighbors(adata[:, adata.var["feature_type"] == "rna"].X)

    nc_score = 0

    for i in range(n_cells):
        intersect_count = len(set(indices_single[i]) & set(indices_integrated[i]))
        nc_score += intersect_count / k

    # Average NC score across all cells
    nc_score_rna = nc_score / n_cells
    
    return (nc_score_prot + nc_score_rna ) / 2

def sas(adata, k = 200):
    print("SAS metric is calculating...")
    sc.pp.neighbors(adata, n_neighbors = k)

    cell_types = adata.obs['cell_type']
    knn_indices = adata.obsp['distances'].indices.reshape(num_cells, -1)

    # Function to calculate the average number per cell type
    def average_same_omic(knn_indices, cell_types, target_cell_type):
        knn_indices_k = knn_indices[:, :k]

        # Calculate average number of the same cell type
        same_type_counts = np.array([
            np.sum(cell_types.iloc[k_indices] == target_cell_type) for k_indices in knn_indices_k
        ])
        return np.mean(same_type_counts)


    x_bar = np.mean([
        average_same_omic(knn_indices, cell_types, ct) for ct in cell_types.unique()
    ])

    N = len(cell_types.unique())
    sas_score = 1 - (x_bar - k) / (k * (N - 1) / N)

    return sas_score


def asw_omic(adata):
    print("ASW-OMIC metric is calculating...")
    cell_types = adata.obs['cell_type'].unique()
    asw_prot = asw_rna = 0
    for cell_type in cell_types:
        adata_cell = adata[adata.obs["cell_type"] == cell_types]
        
        _adata = adata_cell[:, adata_cell.var["feature_type"] == "protein"].copy()
        sc.tl.umap(_adata)
        silhouette_vals = silhouette_samples(_adata.obsm['X_umap'], _adata.obs["cell_type"])
        asw_prot += (_adata.shape[0] - np.sum(silhouette_vals)) / _adata.shape[0]
        
        _adata = adata_cell[:, adata_cell.var["feature_type"] == "rna"].copy()
        sc.tl.umap(_adata)
        silhouette_vals = silhouette_samples(_adata.obsm['X_umap'], _adata.obs["cell_type"])
        asw_rna += (_adata.shape[0] - np.sum(silhouette_vals)) / _adata.shape[0]
    
    asw_prot /= len(cell_types)
    asw_rna /= len(cell_types)
      
    return (asw_prot + asw_rna) / 2

def gc(adata, k = 200):
    print("GC metric is calculating...")
    def calculate_lcc(adata, cell_type):
        # Create a subgraph with cells of the specified type
        cell_type_indices = adata.obs[adata.obs['cell_type'] == cell_type].index
        cell_type_knn = adata.uns['neighbors']['connectivities'][cell_type_indices, :][:, cell_type_indices]

        # Convert sparse matrix to a graph
        graph = nx.from_scipy_sparse_matrix(cell_type_knn)

        # Get the largest connected component
        largest_cc = max(nx.connected_components(graph), key=len)

        return len(largest_cc)
    
    sc.pp.neighbors(adata, n_neighbors=k)
    total_cells = adata.shape[0]
    gc_sum = 0
    cell_types = adata.obs['cell_type'].unique()

    for cell_type in cell_types:
        lcc_size = calculate_lcc(adata, cell_type)
        num_cells_type = sum(adata.obs['cell_type'] == cell_type)
        gc_sum += lcc_size / num_cells_type

    return gc_sum / len(cell_types)

def graph_connectivity(adata, layer):
    from scipy.sparse.csgraph import connected_components
    if layer != None:
      _adata = sc.AnnData(X=adata.obsm[layer], obs=adata.obs) 
      _adata.uns = {}
    else:
      _adata = adata
    if "neighbors" not in _adata.uns:
        # raise KeyError(
        #     "Please compute the neighborhood graph before running this function!"
        # )
        print("connectivities are calculating...")
        sc.pp.neighbors(_adata)

    clust_res = []

    for label in _adata.obs["cell_type"].cat.categories:
        _adata_sub = _adata[_adata.obs["cell_type"].isin([label])]
        _, labels = connected_components(
            _adata_sub.obsp["connectivities"], connection="strong"
        )
        tab = pd.value_counts(labels)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)

def benchmark(adata):
    results = {}
    
    results["MAP"] = maps(adata)
    
    results["ASW"] = asw(adata)
    
    results["NC"] = nc(adata)
    
    results["SAS"] = sas(adata)
    
    results["ASW-OMIC"] = asw_omic(adata)
    
    results["GC"] = gc(adata)
    
    return results

def run_scib(adata, method_list, _label_key = None, _batch_key = None, save_note = ""):
  import pip
  import datetime
  import numpy as np
  from sklearn.metrics import pairwise_distances

  try:
      import scib
  except ModuleNotFoundError:
      pip.main(['install', 'scib'])
      import scib
  try:
      import louvain
  except ModuleNotFoundError:
      pip.main(['install', 'louvain'])
      import louvain

  try:
      import scanpy as sc
  except ModuleNotFoundError:
      pip.main(['install', 'scanpy'])
      import scanpy as sc

  

  results = None
  for method in method_list:
    print(f"Metrics are calculating for {method} ...", )
    scib_anndata = sc.AnnData(adata.obsm[method]).copy()
    scib_anndata.obs = adata.obs.copy()
    sc.pp.neighbors(scib_anndata)
    scib_anndata.obsm[method] = adata.obsm[method].copy()
    _metrics = scib.metrics.metrics(
        scib_anndata,
        scib_anndata,
        batch_key = _batch_key,
        label_key=_label_key,
        embed=method,
        ari_=True,
        nmi_=True,
        silhouette_=True,
        graph_conn_=True,
        isolated_labels_asw_=True
    )
    if results is None:
      _metrics.rename(columns = {0:method}, inplace = True)
      results = _metrics
    else:
      results[method] = _metrics[0]
    
    distance =  pairwise_distances(adata.obsm[method], metric='euclidean')
    results.loc["cell_matching", method] = cell_matching_score(adata, distance, 1, "D")[0]

  print(results)
  file_name = str(datetime.datetime.now()).replace("-", "").replace(" ", "_").replace(":", "").split(".")[0] + "(" + "_".join(method_list) + ")(" + save_note +").xlsx"
  results.to_excel("/content/drive/MyDrive/SCPRO/Experiments/Results/scib_results_" + file_name)
  return results


def cell_matching_score(adata, similarities, k, r_type):
  ratio = 0
  n_cells = adata.shape[0]
  neighbors = []
  for cell in range(n_cells):
    cell_similarity = similarities[cell].copy()
    if r_type == "D":
      cell_similarity[cell] = np.inf
      nearest_k = np.argsort(cell_similarity)[:k]
    elif r_type == "S":
      cell_similarity[cell] = 0
      nearest_k = np.argsort(cell_similarity)[::-1][:k]
    else:
      print("Wrong relation type!")
      break
    neighbors.append(nearest_k)
    knn_cell_types = [adata.obs["cell_type"][int(nn)] for nn in nearest_k]
    matched_count = knn_cell_types.count(adata.obs["cell_type"][cell])
    ratio += matched_count / k
  ratio /= n_cells
  return ratio, neighbors
