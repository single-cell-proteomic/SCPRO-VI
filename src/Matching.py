import cupy as cp
import numpy as np
import networkx as nx
import pandas as pd
import gc
import torch
from sklearn.metrics import pairwise_distances

def filter_weights(weights, protein_list, feature_count):
  filtered_df = weights[weights['subs1'].isin(protein_list) & weights['subs2'].isin(protein_list)]
  filtered_weights = np.zeros((feature_count, feature_count))
  
  if filtered_df.shape[0] != 0:
    print("# of considered PPIs: ", len(filtered_df['subs1'].unique()))
    for row_index, row in filtered_df.iterrows():
      i = np.where(protein_list == row['subs1'])[0]
      j = np.where(protein_list == row['subs2'])[0]
      for _i in i:
        for _j in j:
          if _i < _j:
            filtered_weights[_i, _j] = row['combined_score']
    for i in range(filtered_weights.shape[0]):
      filtered_weights[i, i] = 1
  else:
    filtered_weights = np.ones((feature_count, feature_count))
  print(np.sum(filtered_weights > 0))
  return filtered_weights

def fast_cell_similarity(adata, weights, batch_size=300):
    eps = 0
    cp._default_memory_pool.free_all_blocks()
    torch.cuda.empty_cache()
    n_cells, n_features = adata.shape
    if isinstance(adata.X, np.ndarray):
      vals = adata.X
    else:
      vals = adata.X.toarray()
    protein_expressions = cp.asarray(vals)
    protein_expressions_n= cp.sum(protein_expressions, axis=1)
    protein_expressions_n += eps
    weights = cp.asarray(weights)
    protein_similarities = cp.zeros((n_cells, n_features, n_features))
    
    for i in range(n_cells):
      cell_i = protein_expressions[i]
      outer_product = cp.outer(cell_i, cell_i)
      weighted_outer_product = outer_product * weights
      protein_similarities[i] = weighted_outer_product
      torch.cuda.empty_cache()
      cp._default_memory_pool.free_all_blocks()
    
    filtered_protein_similarities = protein_similarities[:, weights > 0]
    print(filtered_protein_similarities.shape)
    
    del protein_similarities, weights, protein_expressions, outer_product, weighted_outer_product
    gc.collect()


    cell_similarities = cp.zeros((n_cells, n_cells))

    for i in range(n_cells):
      if i % 100 == 0:
        print("# of processed cells: ", i)
      diffs_scalars = cp.sum(cp.abs(filtered_protein_similarities[i+1:] - filtered_protein_similarities[i]), axis=(1))
      cell_similarities[i, i+1:] = diffs_scalars / ((protein_expressions_n[i] + eps) * protein_expressions_n[i+1:])
    
    del diffs_scalars, filtered_protein_similarities
    gc.collect()
    cell_similarities = cell_similarities + cell_similarities.T
    cell_similarities_np = cell_similarities.get()
    inf_mask = np.isinf(cell_similarities_np)
    max_value = np.max(cell_similarities_np[~inf_mask])
    cell_similarities_np[inf_mask] = max_value
    cell_similarities_np[np.isnan(cell_similarities_np)] = 0
    np.fill_diagonal(cell_similarities_np, np.inf)

    del cell_similarities
    gc.collect()
    
    return cell_similarities_np

def positive_edges(distances, k, th):
  adj_matrix = np.zeros((distances.shape[0], distances.shape[0]))
  edges = []
  ctr_included = 0
  for i in range(distances.shape[0]):
    row = distances[i]
    below_threshold_indices = np.where(row < th)[0]
    if below_threshold_indices.size > 0:
        sorted_indices = below_threshold_indices[np.argsort(row[below_threshold_indices])]
        if sorted_indices.size > k:
          sorted_indices = sorted_indices[:k]
        ctr_included += 1
    else:
      sorted_indices = np.argsort(row)[:1]
    for j in range(len(sorted_indices)):
      edges.append((i, sorted_indices[j]))
    adj_matrix[i, sorted_indices] = 1
  np.fill_diagonal(adj_matrix, 1)
  print("# of included cells: ", ctr_included)
  return edges, adj_matrix




def knn_neighbors(similarities, k):
  neighbors = []
  for cell in range(similarities.shape[0]):
    cell_similarity = similarities[cell].copy()
    cell_similarity[cell] = np.inf
    nearest_k = np.argsort(cell_similarity)[:k]
    neighbors.append(nearest_k)
  return neighbors


def cutoff_th(_dist, dist_order, bin_size = 50):
  for j in range(len(_dist) // bin_size):
    curr_mean = np.mean(_dist[dist_order[:(j + 1) * bin_size]])
    curr_neighbors = np.sum(_dist < curr_mean)
    if curr_neighbors > (j+1) * bin_size * 0.50: # 40 for GSE166895, 50 for GSE164378
      if j != 0:
        return _dist[dist_order[j * bin_size]]
      else:
        return _dist[dist_order[bin_size // 2]]
  return _dist[dist_order[len(_dist) // 2]]

def cutoff_th_new(_dist, dist_order, bin_size = 50):
  for j in range(len(_dist) // bin_size):
    selected = _dist[dist_order[:(j + 1) * bin_size]]
    curr_mean = np.mean(selected)
    curr_std = np.std(selected)
    if curr_std > 0.20 * curr_mean:
      if j != 0:
        return _dist[dist_order[j * bin_size]]
      else:
        return _dist[dist_order[2]]
  return _dist[dist_order[len(_dist) // 2]]

def built_graphs(exp_data):
  print("RNA similarities are calculating...")
  if 1 or "cell_similarity" not in exp_data.rna.obsm:
    if "embeddings" in exp_data.rna.obsm:
      rna_distances = pairwise_distances(exp_data.rna.obsm["embeddings"], metric = 'cosine')
    else:
      rna_distances = pairwise_distances(exp_data.rna.X, metric = 'cosine')
    rna_neighbors = knn_neighbors(rna_distances, 40)
    rna_neighbors_sum = np.mean(np.array([row[cols] for row, cols in zip(rna_distances, rna_neighbors)]), axis = 1)
    for i in range(rna_distances.shape[0]):
      rna_distances[i] = rna_distances[i]/ rna_neighbors_sum[i]
    np.fill_diagonal(rna_distances, np.inf)
    exp_data.rna.obsm["cell_similarity"] = rna_distances

  print("Protein similarities are calculating...")
  if 0 or "cell_similarity" not in exp_data.prot.obsm:
    ppi_weights = pd.read_csv("/content/drive/MyDrive/SCPRO/Data/ppi_weights.csv")
    prot_list = exp_data.prot.var.feature_name.values
    ppi_weights = filter_weights(ppi_weights, prot_list, exp_data.prot.shape[1]) 
    prot_distances = fast_cell_similarity(exp_data.prot, ppi_weights)
    prot_neighbors = knn_neighbors(prot_distances, 100)
    prot_neighbors_sum = np.mean(np.array([row[cols] for row, cols in zip(prot_distances, prot_neighbors)]), axis = 1)
    for i in range(prot_distances.shape[0]):
      prot_distances[i] = prot_distances[i]/ prot_neighbors_sum[i]
    exp_data.prot.obsm["cell_similarity"] =  prot_distances

  print("Graphs are building...")
  if 1 or "common_edges" not in exp_data.whole.uns:
    n_cells = exp_data.prot.shape[0]

    distances_pr = exp_data.prot.obsm["cell_similarity"] * exp_data.rna.obsm["cell_similarity"]
    min_val = distances_pr[distances_pr != np.inf].min()
    max_val = distances_pr[distances_pr != np.inf].max()
    distances_pr = (distances_pr - min_val) / (max_val - min_val)

    prot_edges, prot_adj = positive_edges(exp_data.prot.obsm["cell_similarity"], 100, 1)
    print("Number of prots:", len(prot_edges))

    rna_edges, rna_adj = positive_edges(exp_data.rna.obsm["cell_similarity"], 100, 1) 
    print("Number of rnas:", len(rna_edges))

    node_labels = exp_data.rna.obs.cell_type.tolist()
    
    ctr = 0
    for edge in prot_edges:
      if node_labels[edge[0]] != node_labels[edge[1]]:
        ctr += 1
    print("Mismatch count: ", ctr, " all prot edges: ", len(prot_edges))

    ctr = 0
    for edge in rna_edges:
      if node_labels[edge[0]] != node_labels[edge[1]]:
        ctr += 1
    print("Mismatch count: ", ctr, " all rna edges: ", len(rna_edges))


    exp_data.whole.uns["prot_edges"] = list(zip(*prot_edges))
    exp_data.whole.uns["rna_edges"] = list(zip(*rna_edges))
    
    orders = np.argsort(distances_pr, axis=1)
    np.fill_diagonal(distances_pr, 0)
    elbo_vect = [cutoff_th_new(distances_pr[i], orders[i]) for i in range(distances_pr.shape[0])]
    non_filtered_mask = (distances_pr < np.array(elbo_vect)[:, None])
    filter_mask = (distances_pr >= np.array(elbo_vect)[:, None])
    np.fill_diagonal(distances_pr, 0)
    filtered_distances = (distances_pr * non_filtered_mask)
    filtered_distances += filter_mask
    np.fill_diagonal(filtered_distances, 0)
    distances_pr = np.minimum(filtered_distances, filtered_distances.T)

    print(np.sum(distances_pr > 0.3), np.sum(distances_pr < 0.02))
    exp_data.whole.obsm["cell_similarity"] = distances_pr
    exp_data.rna.obsm["cell_similarity_top"] = rna_adj
    exp_data.prot.obsm["cell_similarity_top"] = prot_adj


