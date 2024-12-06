import cupy as cp
import numpy as np
import networkx as nx
import pandas as pd
import gc
import torch
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.preprocessing import RobustScaler

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
    if curr_neighbors > (j+1) * bin_size * 0.40: # 40 for GSE166895, 50 for GSE164378
      if j != 0:
        return _dist[dist_order[j * bin_size]]
      else:
        return _dist[dist_order[bin_size // 2]]
  return _dist[dist_order[len(_dist) // 2]]

def filter_by_pathogen(pathogen, prot_list):
  df = pd.read_excel("/content/drive/MyDrive/SCPRO/Data/PHIs.xlsx")
  df_fitered = df[df["Pathogen"].str.contains(pathogen)]
  return list(set(df_fitered["Uniprot ID.1"]) & set(prot_list))


def built_graphs(exp_data):
  print("RNA similarities are calculating...")
  if 1 or "cell_similarity" not in exp_data.rna.obsm:
    rna_distances = pairwise_distances(exp_data.rna.obsm["embeddings"], metric = 'cosine')
    rna_neighbors = knn_neighbors(rna_distances, 20)
    rna_neighbors_sum = np.mean(np.array([row[cols] for row, cols in zip(rna_distances, rna_neighbors)]), axis = 1)
    for i in range(rna_distances.shape[0]):
      rna_distances[i] = rna_distances[i]/ rna_neighbors_sum[i]
    np.fill_diagonal(rna_distances, np.inf)
    exp_data.rna.obsm["cell_similarity"] = rna_distances

  print("Protein similarities are calculating...")
  if 1 or "cell_similarity" not in exp_data.prot.obsm:
    ppi_weights = pd.read_csv("/content/drive/MyDrive/SCPRO/Data/ppi_weights.csv")
    prot_list = exp_data.prot.var.feature_name.values
    # prot_list = filter_by_pathogen("HIV1", prot_list)
    ppi_weights = filter_weights(ppi_weights, prot_list, exp_data.prot.shape[1]) 
    prot_distances = fast_cell_similarity(exp_data.prot, ppi_weights)
    prot_neighbors = knn_neighbors(prot_distances, 20)
    prot_neighbors_sum = np.mean(np.array([row[cols] for row, cols in zip(prot_distances, prot_neighbors)]), axis = 1)
    for i in range(prot_distances.shape[0]):
      prot_distances[i] = prot_distances[i]/ prot_neighbors_sum[i]
    exp_data.prot.obsm["cell_similarity"] =  prot_distances

  print("Graphs are building...")
  if 1 or "common_edges" not in exp_data.whole.uns:
    n_cells = exp_data.prot.shape[0]
    prot_edges = []
    rna_edges = []
    # distances_p = exp_data.prot.obsm["cell_similarity"].copy()
    # distances_r = exp_data.rna.obsm["cell_similarity"].copy()
    distances_pr = exp_data.prot.obsm["cell_similarity"] * exp_data.rna.obsm["cell_similarity"]
    min_val = distances_pr[distances_pr != np.inf].min()
    max_val = distances_pr[distances_pr != np.inf].max()
    distances_pr = (distances_pr - min_val) / (max_val - min_val)
    # distances_pr = exp_data.whole.obsm["cell_similarity"]
    
    # distances_pr = clean_supervised_distance(distances_pr, exp_data.rna.obs.cell_type.tolist())
    # distances_pr = exponential_normalize(distances_pr)


    # matrix_without_inf = np.where(distances_pr == np.inf, 10, distances_pr)
    # exp_data.whole.obsm["distance_pr"] = matrix_without_inf
    # exp_data.plot_umap("distance_pr", "cell_type")
    # common_edges = list(set(prot_edges) & set(rna_edges))
    # prune_th = 0.2 #P2_0 0.04 #P2_3 0.1 #P1_7 0.03 # GSE166895 0.35 
    # pos_prune_th = 0.06 #P2_0 0.008 #P2_3 0.03 #P1_7 0.007 # GSE166895 0.01
    # common_edges = positive_edges(distances_pr, 10, pos_prune_th, np.array(exp_data.whole.obs.cell_type.tolist()))
    # print("Number of commons:", len(common_edges))

    # labels = np.array(exp_data.whole.obs.cell_type.tolist())
    # for i in range(5000):
    #   for j in range(5000):
    #     if labels[i] != labels[j]:
    #       distances_pr[i][j] = 1

    # print("# of values greather than ", prune_th, ": ", np.sum(distances_pr > prune_th))
    # distances_pr[distances_pr > prune_th] = 1 # 0.2 
    # distances_pr[distances_pr < pos_prune_th] = 0 # 0.03

    prot_edges, prot_adj = positive_edges(exp_data.prot.obsm["cell_similarity"], 100, 1)
    print("Number of prots:", len(prot_edges))

    rna_edges, rna_adj = positive_edges(exp_data.rna.obsm["cell_similarity"], 100, 1) 
    print("Number of rnas:", len(rna_edges))

    # component_labels, component_counts, centers_p, centers_r = connected_components(common_edges, exp_data)

    # cell_cc_labels = []
    # prot_data = exp_data.prot.X.toarray()
    # rna_data = exp_data.rna.X.toarray()
    # for cell in range(exp_data.rna.shape[0]):
    #   cell_cc_labels.append(component_labels.get(cell, -1))
    # exp_data.whole.obs["cc_label"] = cell_cc_labels
    # exp_data.whole.obs["cc_label"] = exp_data.whole.obs["cc_label"].astype(str)
    # exp_data.plot_umap("all", "cc_label")


    # reversed_list = []
    # filtered_common = set()
    # temp_common = set()
    # for i in range(len(common_edges)):
    #   temp_common.add(common_edges[i])
    #   temp_common.add((common_edges[i][1], common_edges[i][0]))
    #   if len(temp_common) == len(filtered_common) + 2:
    #     filtered_common.add(common_edges[i])
    #     temp_common.remove((common_edges[i][1], common_edges[i][0]))
    #   else:
    #     temp_common.remove(common_edges[i])
    # common_edges = list(filtered_common)
    
    
    # common_edges = clean_weaks(common_edges, distances_pr)
    # print("filtered commons: ", len(common_edges))

    # plot_new_umap(exp_data, common_edges, distances_pr)

    # node_labels = {}
    # for edge in common_edges:
    #   node_labels[edge[0]] = exp_data.rna.obs["cell_type"][edge[0]]
    #   node_labels[edge[1]] = exp_data.rna.obs["cell_type"][edge[1]]
    # includeds = list(node_labels.keys())
    # # exp_data.whole.obs["included"] = "0"
    # # exp_data.whole.obs["included"][includeds] = exp_data.whole.obs["cell_type"][includeds]
    # # exp_data.plot_umap("all", coloring = "included")
    # print(np.unique(exp_data.whole.obs["cell_type"], return_counts = True))
    # print(np.unique(exp_data.whole.obs["cell_type"][includeds], return_counts = True))
    # # draw_colored_graph(common_edges, node_labels)


    # component_labels, component_counts, centers_p, centers_r = connected_components(common_edges, exp_data)
    # cell_cc_labels = []
    # soles = []
    # prot_data = exp_data.prot.X.toarray()
    # rna_data = exp_data.rna.X.toarray()
    # for cell in range(exp_data.rna.shape[0]):
    #   cell_cc_labels.append(component_labels.get(cell, -1))
    #   if cell_cc_labels[-1] == -1:
    #     soles.append(cell)
    #     sorted_neighbors = np.argsort(distances_pr[cell])
    #     for i in range(sorted_neighbors.shape[0]):
    #       if distances_pr[cell, sorted_neighbors[i]] < 0.1:
    #         closest_ = component_labels.get(sorted_neighbors[i], -1)
    #         if closest_ != -1:
    #           cell_cc_labels[-1] = closest_
    #           component_labels[cell] = closest_
    #           common_edges.append((cell, sorted_neighbors[i]))
    #       else:
    #         break


        # min_dist = np.inf
        # min_cc = 0
        # for cc in centers_p:
        #   distances_p = cosine(prot_data[cell], centers_p[cc])
        #   distances_r = cosine(rna_data[cell], centers_r[cc])
        #   dist_ = distances_p * distances_r
        #   if min_dist > dist_:
        #     min_dist = dist_
        #     min_cc = cc

        # cell_cc_labels[-1] = min_cc
        # centers_p[min_cc] = ((centers_p[min_cc] * component_counts[min_cc]) + prot_data[cell] ) / (component_counts[min_cc] + 1)
        # centers_r[min_cc] = ((centers_r[min_cc] * component_counts[min_cc]) + rna_data[cell] ) / (component_counts[min_cc] + 1)
        # component_counts[min_cc] += 1
    
    # common_edges = clean_supervised(common_edges, exp_data.whole.obs.cell_type.tolist())

    # print(np.unique(exp_data.whole.obs["cell_type"][soles], return_counts = True))
    # exp_data.whole.obs["cc_label"] = cell_cc_labels
    # exp_data.whole.obs["cc_label"] = exp_data.whole.obs["cell_type"]
    # exp_data.whole.obs["cc_label"] = exp_data.whole.obs["cc_label"].astype(str)
    # print(np.unique(exp_data.whole.obs["cc_label"], return_counts = True))
    # exp_data.plot_umap(data_type = "all", coloring = "cc_label")
    # exp_data.plot_umap(data_type = "all", coloring = "cell_type")

    # subplot_umap(exp_data, common_edges, "cell_type")
    # subplot_umap(exp_data, common_edges, "cc_label")

    # neg_edges = negative_edge_sampling_dist_4(distances_pr, len(common_edges), 0.25) -----> current one
    # filtered_negs = set()
    # temp_neg = set()
    # for i in range(len(neg_edges)):
    #   temp_neg.add(neg_edges[i])
    #   temp_neg.add((neg_edges[i][1], neg_edges[i][0]))
    #   if len(temp_neg) == len(filtered_negs) + 2:
    #     filtered_negs.add(neg_edges[i])
    #     temp_neg.remove((neg_edges[i][1], neg_edges[i][0]))
    #   else:
    #     temp_neg.remove(neg_edges[i])
    # neg_edges = list(filtered_negs)

    # neg_edges = clean_supervised_neg(neg_edges, exp_data.whole.obs.cell_type.tolist())
    # print("filtered negs: ", len(neg_edges))

    # common_same = 0
    # diff_map_c = {}
    # same_map_c = {}
    # for edge in neg_edges:
    #   if exp_data.prot.obs.cell_type[edge[0]] == exp_data.prot.obs.cell_type[edge[1]]:
    #     common_same += 1
    #     # same_map_c[exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]]] = same_map_c.get(exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]], 0) + 1
    #   # else:
    #   #   diff_map_c[exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]]] = diff_map_c.get(exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]], 0) + 1
    # print("neg same:", common_same / len(neg_edges))
    # print(same_map_c)
    # print(diff_map_c)
    
    # common_same = 0
    # diff_map_c = {}
    # same_map_c = {}
    # for edge in common_edges:
    #   if exp_data.prot.obs.cell_type[edge[0]] == exp_data.prot.obs.cell_type[edge[1]]:
    #     common_same += 1
    #   #   same_map_c[exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]]] = same_map_c.get(exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]], 0) + 1
    #   # else:
    #   #   diff_map_c[exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]]] = diff_map_c.get(exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]], 0) + 1
    # print("comon same:", common_same / len(common_edges))
    # print(same_map_c)
    # print(diff_map_c)


    
    # prot_edges = MNN_cell_matching(exp_data.prot.obsm["cell_similarity"], 20 )#, exp_data.whole.obs["cc_label"])
    # rna_edges = MNN_cell_matching(exp_data.rna.obsm["cell_similarity"], 20 )#, exp_data.whole.obs["cc_label"])
    # rna_same = 0
    # diff_map_p = {}
    # diff_map_r = {}
    # for edge in rna_edges:
    #   if exp_data.rna.obs.cell_type[edge[0]] == exp_data.rna.obs.cell_type[edge[1]]:
    #     rna_same += 1
    #   else:
    #     diff_map_r[exp_data.rna.obs.cell_type[edge[0]] +"-"+exp_data.rna.obs.cell_type[edge[1]]] = diff_map_r.get(exp_data.rna.obs.cell_type[edge[0]] +"-"+exp_data.rna.obs.cell_type[edge[1]], 0) + 1
    # print("rna same :", rna_same / len(rna_edges))
    # prot_same = 0
    # for edge in prot_edges:
    #   if exp_data.prot.obs.cell_type[edge[0]] == exp_data.prot.obs.cell_type[edge[1]]:
    #     prot_same += 1
    #   else:
    #     diff_map_p[exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]]] = diff_map_p.get(exp_data.prot.obs.cell_type[edge[0]] +"-"+exp_data.prot.obs.cell_type[edge[1]], 0) + 1
    # print("prot same :", prot_same / len(prot_edges))


    node_labels = exp_data.rna.obs.cell_type.tolist()
    # ctr = 0
    # for edge in common_edges:
    #   if node_labels[edge[0]] != node_labels[edge[1]]:
    #     ctr += 1
    # print("Mismatch count: ", ctr, " all positive edges: ", len(common_edges))
    
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

    # graph_statistics(exp_data, common_edges)
    # component_labels, centers_p, centers_r = connected_components(common_edges, exp_data)
    # print(component_labels)
    
    # neg_edges = negative_edge_sampling(exp_data, len(common_edges))

    # exp_data.plot_umap(data_type = "all", coloring = "cc_label")
    # exp_data.plot_umap(data_type = "all")


    # prot_edges = common_edges #  list(set(prot_edges) & set(common_edges))
    # rna_edges =  common_edges # list(set(rna_edges) & set(common_edges))
    # common_edges = list(zip(*common_edges))

    exp_data.whole.uns["prot_edges"] = list(zip(*prot_edges))
    exp_data.whole.uns["rna_edges"] = list(zip(*rna_edges))
    
    # np.fill_diagonal(distances_pr, 0)
    # dists_backup = distances_pr.copy()
    # orders = np.argsort(distances_pr, axis = 1)
    # closests = orders[:, :200]
    # to_be_filtered = orders[:, :5]
    # rows = np.arange(distances_pr.shape[0])[:, np.newaxis]
    # distances_pr = np.ones_like(distances_pr)
    # distances_pr[rows, closests] = dists_backup[rows, closests]
    # distances_pr[rows, to_be_filtered] = 0
    # np.fill_diagonal(distances_pr, 0)
    # exp_data.whole.obsm["cell_similarity"] = distances_pr

    # orders = np.argsort(exp_data.rna.obsm["cell_similarity"], axis = 1)
    # closests = orders[:, :200]
    # to_be_filtered = orders[:, :5]
    # rows = np.arange(exp_data.rna.obsm["cell_similarity"].shape[0])[:, np.newaxis]
    # rna_distances = np.ones_like(exp_data.rna.obsm["cell_similarity"])
    # rna_distances[rows, closests] = exp_data.rna.obsm["cell_similarity"][rows, closests]
    # rna_distances[rows, to_be_filtered] = 0
    # np.fill_diagonal(rna_distances, 0)
    # exp_data.rna.obsm["cell_similarity_top"] = rna_distances

    # orders = np.argsort(exp_data.prot.obsm["cell_similarity"], axis = 1)
    # closests = orders[:, :200]
    # to_be_filtered = orders[:, :5]
    # rows = np.arange(exp_data.prot.obsm["cell_similarity"].shape[0])[:, np.newaxis]
    # prot_distances = np.ones_like(exp_data.prot.obsm["cell_similarity"])
    # prot_distances[rows, closests] = exp_data.prot.obsm["cell_similarity"][rows, closests]
    # prot_distances[rows, to_be_filtered] = 0
    # np.fill_diagonal(prot_distances, 0)
    # exp_data.prot.obsm["cell_similarity_top"] = prot_distances

    # np.fill_diagonal(distances_pr, 0)
    # orders = np.argsort(distances_pr, axis = 1)
    # to_be_filtered = orders[:, :5]
    # rows = np.arange(distances_pr.shape[0])[:, np.newaxis]
    # closest_vars = np.var(distances_pr[rows, to_be_filtered], axis = 1)
    # cell_to_filter = np.argsort(closest_vars)[-500:]
    # filtered_orders = np.argsort(distances_pr[:, ~cell_to_filter], axis = 1)
    # closest_cells = filtered_orders[cell_to_filter, 1]
    # distances_pr[rows, cell_to_filter] = distances_pr[:, ~cell_to_filter][rows, closest_cells]
    # distances_pr[cell_to_filter] = distances_pr[:, cell_to_filter].T

    rows = np.arange(distances_pr.shape[0])[:, np.newaxis]
    dists_backup = distances_pr.copy()
    # row_wise_dist = np.sort(distances_pr, axis = 1)[:, 3]
    # closest_th = np.sort(row_wise_dist)[int(exp_data.whole.shape[0] * 0.9)]
    # # row_wise_dist = np.sort(distances_pr, axis = 1)[:, 20]
    # # nearest_th = np.sort(row_wise_dist)[int(exp_data.whole.shape[0] * 0.4)]
    # # row_wise_dist = np.sort(distances_pr, axis = 1)[:, 200]
    # # counting_th = np.sort(row_wise_dist)[int(exp_data.whole.shape[0] * 0.4)]
    # row_wise_dist = np.sort(distances_pr, axis = 1)[:, 2500]
    # cutting_th = np.sort(row_wise_dist)[int(exp_data.whole.shape[0] * 0.9)]
    # distances_pr = np.ones_like(distances_pr)
    # # distances_pr[dists_backup < counting_th] = 0.5 
    # distances_pr[dists_backup < cutting_th] = dists_backup[dists_backup < cutting_th]
    # distances_pr[dists_backup < closest_th] = 0

    # orders = np.argsort(distances_pr, axis = 1)
    # to_be_filtered = orders[:, :3]
    # rows = np.arange(distances_pr.shape[0])[:, np.newaxis]
    # distances_pr[rows, to_be_filtered] = 0
    # np.fill_diagonal(distances_pr, 0)
    # distances_pr[distances_pr > 0.07] = 1
    # exp_data.whole.obsm["cell_similarity"] = distances_pr

    # edge_cells = np.where(cell_orders > exp_data.whole.shape[0] * 0.9)[0]
    # filtered_orders = np.argsort(distances_pr[:, ~edge_cells], axis = 1)
    # closest_cells = filtered_orders[edge_cells, 1]
    # distances_pr[rows, edge_cells] = np.ones_like(distances_pr[rows, edge_cells])
    # for i in range(len(edge_cells)):
    #   distances_pr[closest_cells[i], edge_cells[i]] = 0
    # distances_pr[edge_cells] = distances_pr[:, edge_cells].T
    # np.fill_diagonal(distances_pr, 0)
    
    # orders = np.argsort(distances_pr, axis = 1)
    # closests = orders[:, :200]
    # middlest = orders[:, 200:1000]
    # to_be_filtered = orders[:, :10]
    # distances_pr = np.ones_like(distances_pr)
    # distances_pr[rows, closests] = dists_backup[rows, closests]
    # distances_pr[rows, to_be_filtered] = 0
    # distances_pr[rows, middlest] = 0.5
    # np.fill_diagonal(distances_pr, 0)

    # for c_type in supervised_ths:
    #   tmp = distances_pr[exp_data.whole.obs.cell_type == c_type]
    #   tmp[tmp >= supervised_up_ths[c_type]] = 1
    #   distances_pr[exp_data.whole.obs.cell_type == c_type] = tmp
    #   distances_pr[:, exp_data.whole.obs.cell_type == c_type] = tmp.T
    # for c_type in supervised_ths:
    #   tmp = distances_pr[exp_data.whole.obs.cell_type == c_type]
    #   tmp[tmp < supervised_ths[c_type]] = 0
    #   distances_pr[exp_data.whole.obs.cell_type == c_type] = tmp
    #   distances_pr[:, exp_data.whole.obs.cell_type == c_type] = tmp.T


    # for c_type in supervised_ths:
    #   tmp = distances_pr[exp_data.whole.obs.cell_type == c_type]
    #   tmp[tmp >= supervised_up_ths[c_type]] = 1
    #   distances_pr[exp_data.whole.obs.cell_type == c_type] = tmp
    #   distances_pr[:, exp_data.whole.obs.cell_type == c_type] = tmp.T
    # for c_type in supervised_ths:
    #   tmp = distances_pr[exp_data.whole.obs.cell_type == c_type]
    #   tmp[tmp < supervised_ths[c_type]] = 0
    #   distances_pr[exp_data.whole.obs.cell_type == c_type] = tmp
    #   distances_pr[:, exp_data.whole.obs.cell_type == c_type] = tmp.T
    elbo_vect = []
    np.fill_diagonal(distances_pr, 0)
    orders = np.argsort(distances_pr, axis = 1)
    for i in range(distances_pr.shape[0]):
      elbo_th = cutoff_th(distances_pr[i], orders[i])
      elbo_vect.append(elbo_th)
    # print(elbo_vect)
    for i in range(distances_pr.shape[0]):
      tmp = distances_pr[i]
      tmp[tmp > (elbo_vect[i])] = 1
      distances_pr[i] = tmp
      distances_pr[:, i] = tmp.T
    for i in range(distances_pr.shape[0]):
      tmp = distances_pr[i]
      tmp[tmp <= elbo_vect[i]] = 0
      distances_pr[i] = tmp
      distances_pr[:, i] = tmp.T
    
    print(np.sum(distances_pr == 0), np.sum(distances_pr == 1))
    exp_data.whole.obsm["cell_similarity"] = distances_pr


    # rna_dists = exp_data.rna.obsm["cell_similarity"].copy()
    # np.fill_diagonal(rna_dists, 0)
    # orders = np.argsort(rna_dists, axis = 1)
    # elbo_vect = []

    # for i in range(rna_dists.shape[0]):
    #   elbo_th = cutoff_th(rna_dists[i], orders[i])
    #   elbo_vect.append(elbo_th)

    # for i in range(rna_dists.shape[0]):
    #   tmp = rna_dists[i]
    #   tmp[tmp > (elbo_vect[i])] = 1
    #   rna_dists[i] = tmp
    #   rna_dists[:, i] = tmp.T
    # for i in range(rna_dists.shape[0]):
    #   tmp = rna_dists[i]
    #   tmp[tmp <= elbo_vect[i]] = 0
    #   rna_dists[i] = tmp
    #   rna_dists[:, i] = tmp.T

    # print(np.sum(rna_dists == 0), np.sum(rna_dists == 1))
    # exp_data.rna.obsm["cell_similarity_top"] = rna_dists
    
    # prot_dists = exp_data.prot.obsm["cell_similarity"].copy()
    # np.fill_diagonal(prot_dists, 0)
    # orders = np.argsort(prot_dists, axis = 1)
    # elbo_vect = []

    # for i in range(prot_dists.shape[0]):
    #   elbo_th = cutoff_th(prot_dists[i], orders[i])
    #   elbo_vect.append(elbo_th)

    # for i in range(prot_dists.shape[0]):
    #   tmp = prot_dists[i]
    #   tmp[tmp > (elbo_vect[i])] = 1
    #   prot_dists[i] = tmp
    #   prot_dists[:, i] = tmp.T
    # for i in range(prot_dists.shape[0]):
    #   tmp = prot_dists[i]
    #   tmp[tmp <= elbo_vect[i]] = 0
    #   prot_dists[i] = tmp
    #   prot_dists[:, i] = tmp.T

    # print(np.sum(prot_dists == 0), np.sum(prot_dists == 1))
    # exp_data.rna.obsm["cell_similarity_top"] = prot_dists

    # orders = np.argsort(exp_data.rna.obsm["cell_similarity"], axis = 1)
    # closests = orders[:, :200]
    # middlest = orders[:, 200:1000]
    # to_be_filtered = orders[:, :5]
    # rows = np.arange(exp_data.rna.obsm["cell_similarity"].shape[0])[:, np.newaxis]
    # rna_distances = np.ones_like(exp_data.rna.obsm["cell_similarity"])
    # rna_distances[rows, closests] = exp_data.rna.obsm["cell_similarity"][rows, closests]
    # rna_distances[rows, to_be_filtered] = 0
    # rna_distances[rows, middlest] = 0.5
    # np.fill_diagonal(rna_distances, 0)
    # exp_data.rna.obsm["cell_similarity_top"] = rna_distances
    exp_data.rna.obsm["cell_similarity_top"] = 1 - rna_adj


    # orders = np.argsort(exp_data.prot.obsm["cell_similarity"], axis = 1)
    # closests = orders[:, :200]
    # middlest = orders[:, 200:1000]
    # to_be_filtered = orders[:, :5]
    # rows = np.arange(exp_data.prot.obsm["cell_similarity"].shape[0])[:, np.newaxis]
    # prot_distances = np.ones_like(exp_data.prot.obsm["cell_similarity"])
    # prot_distances[rows, closests] = exp_data.prot.obsm["cell_similarity"][rows, closests]
    # prot_distances[rows, to_be_filtered] = 0
    # prot_distances[rows, middlest] = 0.5
    # np.fill_diagonal(prot_distances, 0)
    # exp_data.prot.obsm["cell_similarity_top"] = prot_distances
    exp_data.prot.obsm["cell_similarity_top"] = 1 - prot_adj


    
  # return None, None, None 

  # exp_data.prot.obsm["cell_similarity_2"] = cell_similarity_pairwise(exp_data.prot, ppi_weights)


