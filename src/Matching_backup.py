import cupy as cp
import numpy as np
import networkx as nx
import pandas as pd
import gc
import torch
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.preprocessing import RobustScaler

# from kneed import KneeLocator

def cell_matching_score(adata, similarities, k, r_type):
  ratio = 0
  n_cells = adata.shape[0]
  neighbors = []
  for cell in range(n_cells):
    cell_similarity = similarities[cell]
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

def filter_weights(weights, protein_list):
  filtered_df = weights[weights['subs1'].isin(protein_list) & weights['subs2'].isin(protein_list)]
  n = protein_list.shape[0]
  filtered_weights = np.ones((n, n))

  if filtered_df.shape[0] != 0:
    for row_index, row in filtered_df.iterrows():
      i = np.where(protein_list == row['subs1'])[0]
      j = np.where(protein_list == row['subs2'])[0]
      filtered_weights[i, j] += row['combined_score']
    # for i in range(filtered_weights.shape[0]):
    #   filtered_weights[i, i] = 1
    filtered_weights /= 2
  else:
    filtered_weights = np.ones((n, n))
  return filtered_weights

def filter_weights_legacy(weights, protein_list):
  filtered_df = weights[weights['subs1'].isin(protein_list) & weights['subs2'].isin(protein_list)]
  n = protein_list.shape[0]
  filtered_weights = np.zeros((n, n))
  
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
    filtered_weights = np.ones((n, n))
  print(np.sum(filtered_weights > 0))
  return filtered_weights

# def cell_similarity_cpu(self, adata, weights):
#   n_cells = adata.shape[0]
#   protein_expressions = adata.X.toarray()
#   filtered_weights = self.filter_weights(weights, adata.var_names)
#   cell_similarities = np.zeros((n_cells, n_cells))
#   for i in range(n_cells - 1):
#     for j in range(i + 1, n_cells):
#       cell1 = protein_expressions[i]
#       cell2 = protein_expressions[j]
#       similarity_score = np.outer(cell1, cell2)
#       similarity_score = similarity_score * filtered_weights
#       similarity_score = np.sum(similarity_score)
#       cell_similarities[i, j] = similarity_score
#   return cell_similarities

# def cell_similarity_gpu(self, adata, weights, batch_size=300):
#     n_cells = adata.shape[0]
#     protein_expressions = cp.asarray(adata.X.toarray())
#     filtered_weights = cp.asarray(self.filter_weights(weights, adata.var_names))
    
#     cell_similarities = cp.zeros((n_cells, n_cells))
    
#     for i in range(n_cells):
#         cell_i = protein_expressions[i].reshape(1, -1)
#         if i % 100 == 0:
#           print(i)
#         for j in range(i + 1, n_cells, batch_size):
#             batch_end_j = min(j + batch_size, n_cells)
#             batch_protein_expressions_j = protein_expressions[j:batch_end_j,:]
#             outer_product = cp.outer(cell_i, batch_protein_expressions_j)
#             outer_product = cp.asarray(cp.hsplit(outer_product, batch_end_j - j))
#             weighted_outer_product = outer_product * filtered_weights[None, :, :]
            
#             cell_similarities[i, j:batch_end_j] += cp.sum(weighted_outer_product, axis=(1, 2))
    
#     return cell_similarities

def save_graph(edge_weights, node_names, name):
  import numpy as np
  import matplotlib.pyplot as plt
  import networkx as nx


  G = nx.Graph()

  num_nodes = len(node_names)
  for i in range(num_nodes):
      for j in range(i + 1, num_nodes):
          weight = edge_weights[i, j]
          if weight > 0:
              G.add_edge(i, j, weight=weight)


  edge_colors = [G[i][j]['weight'] for i, j in G.edges()]

  nx.draw(
      G,
      with_labels=True,
      node_color='skyblue',
      node_size=800,
      edge_cmap=plt.cm.Blues,  # Set edge colormap
      edge_color=edge_colors,  # Set edge colors based on weights
  )

  sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
  sm.set_array(edge_colors)
  cbar = plt.colorbar(sm)

  plt.title('Graph with Edges Colored by Weights')
  plt.savefig(name + ".png")
  plt.show()

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
    # for i in range(5):
    #   print(adata.obs.cell_type[i])
    #   self.save_graph(protein_similarities[i].get(), np.array(adata.var_names), "graph" + str(i))


    cell_similarities = cp.zeros((n_cells, n_cells))

    print(filtered_protein_similarities.nbytes)
    # print(filtered_protein_similarities.shape, filtered_protein_similarities)
    for i in range(n_cells):
      if i % 100 == 0:
        print(i)
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
    # print(np.isnan(cell_similarities_np).any())

    del cell_similarities
    gc.collect()
    
    return cell_similarities_np

def fast_cell_similarity2(adata, weights, batch_size=300):
    eps = 0
    cp._default_memory_pool.free_all_blocks()
    torch.cuda.empty_cache()
    n_cells, n_features = adata.shape
    if isinstance(adata.X, np.ndarray):
      vals = adata.X
    else:
      vals = adata.X.toarray()
    protein_expressions = cp.asarray(vals)
    protein_expressions_n= cp.sum(protein_expressions, axis=1, keepdims=True)
    protein_expressions /= protein_expressions_n

    # max_expressions= cp.max(protein_expressions, axis=0)
    # max_expressions = cp.outer(max_expressions, max_expressions)
    weights = cp.asarray(weights)
    weights = weights #/ (max_expressions)
    protein_similarities = cp.zeros((n_cells, n_features, n_features))
    
    for i in range(n_cells):
      cell_i = protein_expressions[i]
      outer_product = cp.outer(cell_i, cell_i)
      weighted_outer_product = outer_product * weights
      protein_similarities[i] = weighted_outer_product
      torch.cuda.empty_cache()
      cp._default_memory_pool.free_all_blocks()
    
    del weights
    del protein_expressions
    del outer_product
    del weighted_outer_product
    gc.collect()
    # for i in range(5):
    #   print(adata.obs.cell_type[i])
    #   self.save_graph(protein_similarities[i].get(), np.array(adata.var_names), "graph" + str(i))


    cell_similarities = cp.zeros((n_cells, n_cells))
    total_sum_count = n_features * n_features
    
    for i in range(n_cells):
      if i % 100 == 0:
        print(i)
      diffs = cp.abs(protein_similarities - protein_similarities[i])
      diffs_scalars = cp.sum(diffs, axis=(1, 2))
      # sums_scalars = cp.sum(sums, axis=(1, 2))
      cell_similarities[i] = diffs_scalars / 2 # / total_sum_count
    # cell_similarities = (cell_similarities - cell_similarities.min()) / (cell_similarities.max() - cell_similarities.min())
    cell_similarities_np = cell_similarities.get()
    np.fill_diagonal(cell_similarities_np, np.inf)
    del cell_similarities
    del diffs
    gc.collect()
    
    return cell_similarities_np

def fast_cell_similarity_n(adata, weights, batch_size=500):
    cp._default_memory_pool.free_all_blocks()
    torch.cuda.empty_cache()
    n_cells, n_features = adata.shape
    protein_expressions = cp.asarray(adata.X.toarray())
    protein_expressions_n= cp.sum(protein_expressions, axis=1)
    weights = cp.asarray(weights)
    cell_similarities = cp.zeros((n_cells, n_cells))
    
    for i in range(0, n_features, batch_size):
      batch_end = min(n_features, i + batch_size)
      if i % 10 == 0:
        print(i)
      protein_similarities = cp.zeros((n_cells,batch_end - i,n_features))
      for j in range(n_cells):
        protein_similarities[i] = cp.outer(protein_expressions[j][i:batch_end], protein_expressions[j]) * weights[i:batch_end]

      # cell_similarities = cp.asarray(cell_similarities)
      for j in range(n_cells):
        cell_similarities[j] += cp.sum(cp.abs(protein_similarities - protein_similarities[j]), axis=(1, 2))

    for i in range(n_cells):
      cell_similarities[i] /= (protein_expressions_n[i] * protein_expressions_n)
      
    
    
    # for i in range(5):
    #   print(adata.obs.cell_type[i])
    #   self.save_graph(protein_similarities[i].get(), np.array(adata.var_names), "graph" + str(i))


    
    
    del weights
    del protein_expressions


    cell_similarities_np = cell_similarities.get()
    del cell_similarities
    gc.collect()
    return cell_similarities_np

def cell_similarity_pairwise(adata, weights, batch_size=300):
  n_cells = adata.shape[0]
  protein_expressions = cp.asarray(adata.X.toarray())
  weights = cp.asarray(weights)
  
  cell_similarities = cp.zeros((n_cells, n_cells))
  
  for i in range(n_cells):
      cell_i = protein_expressions[i]
      cell_i_n = cp.sum(cell_i)
      if i % 100 == 0:
        print(i)
      for j in range(i + 1, n_cells, batch_size):
          batch_end_j = min(j + batch_size, n_cells)
          batch_protein_expressions_j = protein_expressions[j:batch_end_j,:]
          batch_protein_expressions_j_n = cp.sum(batch_protein_expressions_j, axis=1)
          batch_protein_expressions_j = cp.abs(batch_protein_expressions_j - cell_i)
          reshaped_matrix = batch_protein_expressions_j[:, :, cp.newaxis]
          outer_product = reshaped_matrix * reshaped_matrix.transpose(0, 2, 1)
          weighted_outer_product = outer_product * weights[None, :, :]
          
          cell_similarities[i, j:batch_end_j] += cp.sum(weighted_outer_product, axis=(1, 2))
          cell_similarities[i, j:batch_end_j] /= cell_i_n * batch_protein_expressions_j_n
  return cell_similarities + cell_similarities.T

def euclidean_similarity(adata):
    cell_data = adata.X
    return pairwise_distances(cell_data, metric='euclidean')

# def cell_similarity_pathway(adata, batch_size = 3):
#   n_cells = adata.shape[0]
#   df_pathways = pd.read_csv("/content/drive/MyDrive/SCPRO/Data/Pathways.txt", delimiter = "	", header = None)
#   df_pathways = df_pathways[df_pathways[0].isin(adata.var_names)]
#   grouped_data = df_pathways.groupby(1)[0].apply(list)
#   grouped_dict = grouped_data.to_dict()
#   concat_arr = np.concatenate([arr for arr in grouped_dict.values()])

#   unique_items, item_counts = np.unique(concat_arr, return_counts=True)
#   item_counts = cp.array(item_counts)
#   n_features = len(concat_arr)
#   cell_similarities = cp.zeros((n_cells, n_cells))
#   gene_expressions = cp.asarray(adata[:,unique_items].X.toarray())[cp.newaxis, :, :]
#   for i in range(0, n_cells, batch_size):
#     if i % 100 == 0:
#       print(i)
#     batch_end_i = min(i + batch_size, n_cells)
#     cell_expressions = cp.array(adata[i:batch_end_i,unique_items].X.toarray())
#     cell_expressions = cell_expressions[:, cp.newaxis, :]
#     # cell_similarities[i:batch_end_i] = cp.sum((cp.minimum(cell_expressions, gene_expressions) / (cell_expressions + gene_expressions + 0.00001)) * item_counts, axis=-1)
#     cell_similarities[i:batch_end_i] = cp.sum(cp.minimum(cell_expressions, gene_expressions) / (cell_expressions + gene_expressions + 0.00001), axis=-1)
#     # cell_similarities[i:batch_end_i] = cp.sqrt(cp.sum(cp.power(cp.abs((cell_expressions - gene_expressions)*item_counts), 2), axis=-1))
    
#   del cell_expressions
#   del gene_expressions
#   cell_similarities_np = cell_similarities.get()
#   del cell_similarities
#   gc.collect()
#   return cell_similarities_np

def weighted_cosine_distance(adata):
    df_pathways = pd.read_csv("/content/drive/MyDrive/SCPRO/Data/Pathways.txt", delimiter = "	", header = None)
    # df_pathways_U = pd.read_csv("/content/drive/MyDrive/SCPRO/Data/Pathways-Uniprot.txt", delimiter = "	", header = None)
    # df_pathways = pd.concat([df_pathways, df_pathways_U], ignore_index=True)
    df_pathways = df_pathways[df_pathways[0].isin(adata.var_names)]
    grouped_data = df_pathways.groupby(1)[0].apply(list)
    grouped_dict = grouped_data.to_dict()
    
    var_order_dict = {var_name: idx for idx, var_name in enumerate(adata.var_names)}
    weights = np.ones(adata.shape[1])
    for pathway in grouped_dict:
      pathway_len = len(grouped_dict[pathway])
      if pathway_len < 6:
        for gene in grouped_dict[pathway]:
          weights[var_order_dict[gene]] += 1 
    weights = np.log(weights)
    weights += 1

    gene_expression = adata.X.toarray()
    gene_expression = cp.array(gene_expression)
    weights = cp.array(weights)
    # Normalize the gene expression matrix
    norm = cp.linalg.norm(gene_expression * weights, axis=1, keepdims=True)
    normalized_expression = (gene_expression * weights) / norm

    # Compute cosine similarity matrix
    similarity_matrix = cp.matmul(normalized_expression, normalized_expression.T)
    similarity_matrix = similarity_matrix.get()
    np.fill_diagonal(similarity_matrix, 0)
    return 1 - similarity_matrix

def MNN_cell_matching(distances, k, knn = []):
  closest_neighbors = np.argsort(distances, axis=1)[:, :k]
  mnn_pairs = []
  knn_size = len(knn)
  for cell in range(closest_neighbors.shape[0]):
    cell_neighbors = closest_neighbors[cell]
    for neighbor in cell_neighbors:
      if cell in closest_neighbors[neighbor]:
        if knn_size != 0:
          if knn[cell] == knn[neighbor]:
            mnn_pairs.append((cell, neighbor))
        else:
          mnn_pairs.append((cell, neighbor))
  return mnn_pairs

def plot_distribution(array, name):
  import matplotlib.pyplot as plt
  plt.hist(array, bins=len(set(array)))
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.title(name + " Distribution")
  plt.show()

def graph_statistics(exp_data, edges):
  import networkx as nx
  import matplotlib.pyplot as plt
  import scanpy as sc
  from scipy.spatial.distance import cosine
 
  G = nx.Graph()
  G.add_edges_from(edges)
  edge_betweenness = nx.degree_centrality(G)
  # print(edge_betweenness)

  decent_edges = []
  decent_distances = []
  faulty_edges = []
  faulty_distances = []
  node_labels = exp_data.whole.obs.cell_type.tolist()
  ctr = 0
  for edge in edges:
    if node_labels[edge[0]] != node_labels[edge[1]]:
      faulty_edges.append(edge)
      faulty_distances.append(cosine(exp_data.whole.X[edge[0]].toarray().flatten(), exp_data.whole.X[edge[1]].toarray().flatten()))
    else:
      decent_edges.append(edge)
      decent_distances.append(cosine(exp_data.whole.X[edge[0]].toarray().flatten(), exp_data.whole.X[edge[1]].toarray().flatten()))
  print("Decent: ", len(decent_edges), " Faulty:", len(faulty_edges))
  # decent_betweenness = [edge_betweenness[edge] if edge in edge_betweenness else edge_betweenness[(edge[1], edge[0])] for edge in decent_edges]
  # faulty_betweenness = [edge_betweenness[edge] if edge in edge_betweenness else edge_betweenness[(edge[1], edge[0])] for edge in faulty_edges]
  decent_betweenness = [edge_betweenness[edge[0]] + edge_betweenness[edge[1]] for edge in decent_edges]
  faulty_betweenness = [edge_betweenness[edge[0]] + edge_betweenness[edge[1]] for edge in faulty_edges]
  # Calculate mean betweenness centrality for decent and faulty edges
  mean_decent_betweenness = sum(decent_betweenness) / len(decent_betweenness) if decent_betweenness else 0
  mean_faulty_betweenness = sum(faulty_betweenness) / len(faulty_betweenness) if faulty_betweenness else 0
  # print(np.unique(decent_betweenness, return_counts= True))
  # print(np.unique(faulty_betweenness, return_counts= True))

  print("Mean betweenness centrality for decent edges:", mean_decent_betweenness)
  print("Mean betweenness centrality for faulty edges:", mean_faulty_betweenness)
  print("max decent", max(decent_betweenness))
  print("max faulty", max(faulty_betweenness))

  plot_distribution(decent_betweenness,  "decent")
  plot_distribution(faulty_betweenness, "faulty")



def draw_colored_graph(edges, node_labels):
  import networkx as nx
  import matplotlib.pyplot as plt
  # Create a graph
  G = nx.Graph()
  
  # Add edges to the graph
  G.add_edges_from(edges)
  
  # Create a list of unique labels
  unique_labels = list(set(node_labels.values()))
  # Assign a color to each unique label
  color_map = {}
  for i, label in enumerate(unique_labels):
      color_map[label] = plt.cm.tab10(i)  # You can change the colormap here
  
  # Create a list of colors for each node based on its label
  colors = [color_map[node_labels[node]] for node in G.nodes()]
  
  # Draw the graph
  pos = nx.spring_layout(G)  # You can use different layout algorithms
  nx.draw(G, pos, with_labels = False, node_color=colors, cmap=plt.cm.tab10)
  
  # Draw color legend
  for label, color in color_map.items():
      plt.scatter([], [], c=color, label=label)
  
  plt.legend()
  plt.show()


def connected_components(edge_list, exp_data):
    import community.community_louvain as community_louvain
    import matplotlib.pyplot as plt
    G = nx.Graph()
    G.add_edges_from(edge_list)

    num_nodes = G.number_of_nodes()
    print("Number of nodes:", num_nodes)

    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    print("Average degree of nodes:", avg_degree)

    # degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    # degree_count = nx.degree_histogram(G)

    # # Plot degree distribution
    # plt.bar(range(len(degree_count)), degree_count, width=0.8, color='b')
    # plt.xlabel('Degree')
    # plt.ylabel('Number of Nodes')
    # plt.title('Degree Distribution')
    # plt.show()

    # components = nx.connected_components(G)
    components = community_louvain.best_partition(G)
    communities = {}
    for node, community_id in components.items():
        if community_id not in communities:
            communities[community_id] = [node]
        else:
            communities[community_id].append(node)
    components = communities

    node_labels = {}
    centers_p = {}
    centers_r = {}
    counts = {}
    for label, component in components.items():
    # for label, component in enumerate(components):
      if len(component) > 1:
        counts[label] = len(component)
        centers_p[label] = np.mean(exp_data.prot.X.toarray()[list(component)], axis = 0)
        centers_r[label] = np.mean(exp_data.rna.X.toarray()[list(component)], axis = 0)
        for node in component:
            node_labels[node] = label
        # print(label, len(component))

    return node_labels, counts, centers_p, centers_r

def negative_edge_sampling(exp_data, pos_edge_count):
  cc_list, cell_counts = np.unique(exp_data.whole.obs.cc_label, return_counts= True)
  negative_edge_index = []
  for i in range(len(cc_list)):
    sample_size = int(cell_counts[i] / exp_data.whole.shape[0] * pos_edge_count * 3)
    cell_mask = exp_data.whole.obs.cc_label == cc_list[i]
    cells = np.where(cell_mask)[0]
    target_mask = exp_data.whole.obs.cc_label != cc_list[i]
    targets = np.where(target_mask)[0]
    selected_cells = np.random.choice(cells, size= sample_size, replace=True)
    selected_targets = np.random.choice(targets, size= sample_size, replace=True)
    # print(len(selected_cells), len(selected_targets))
    random_pairs = list(zip(selected_cells, selected_targets))
    negative_edge_index.extend(random_pairs)
  return negative_edge_index

def negative_edge_sampling_dist(distances, pos_edge_count):
  neg_edge_count = pos_edge_count * 10 * 2
  flattened_indices = np.argsort(-distances, axis=None)[:neg_edge_count]
  row_indices, col_indices = np.unravel_index(flattened_indices, distances.shape)
  return list(zip(row_indices, col_indices))

def negative_edge_sampling_dist_2(distances, pos_edge_count):
  neg_edge_per_cell = int(pos_edge_count / distances.shape[0]) * 10
  flattened_indices = np.argsort(-distances, axis=1)[:, :neg_edge_per_cell]
  edges = []
  for i in range(distances.shape[0]):
      for j in range(neg_edge_per_cell):
          edges.append((i, flattened_indices[i][j]))

  return edges

def negative_edge_sampling_dist_3(distances, pos_edge_count, th):
    k = int(pos_edge_count / distances.shape[0])
    if k == 0 :
      k = 1
    k *= 3
    n = distances.shape[0]  # Number of samples
    edges = []
    for i in range(n):
        # Create a mask to filter distances greater than or equal to m
        mask = distances[i] >= th
        
        # Exclude self-distances
        mask[i] = False
        
        # Get indices of valid neighbors
        valid_indices = np.where(mask)[0]
        
        # Select k random neighbors from valid neighbors
        if len(valid_indices) < k:
            # If there are less than k valid neighbors, select them all
            selected_indices = valid_indices
        else:
            # Otherwise, randomly select k neighbors
            selected_indices = np.random.choice(valid_indices, k, replace=False)
        for j in range(len(selected_indices)):
          edges.append((i, selected_indices[j]))
    
    return edges

def negative_edge_sampling_dist_4(distances, pos_edge_count, th):

    n = distances.shape[0]  # Number of samples
    edges = []
    for i in range(n):

        mask = distances[i] >= th
        
        mask[i] = False

        valid_indices = np.where(mask)[0]
        for j in range(len(valid_indices)):
          edges.append((i, valid_indices[j]))
    
    return edges

def kmeans_clustering(adata, n_clusters):
    from sklearn.cluster import KMeans
    X = adata.X.toarray() 
    kmeans_model = KMeans(n_clusters=n_clusters)
    kmeans_model.fit(X)
    cluster_labels = [str(x) for x in kmeans_model.labels_]
    return cluster_labels

import umap
from sklearn.cluster import KMeans

def kmeans_with_umap(anndata, n_clusters=10):
    umap_embeddings = umap.UMAP().fit_transform(anndata.X)

    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(umap_embeddings)
    
    return kmeans.labels_

def MNN_analysis(exp_data, edges):
  common_same = 0
  diff_map_c = {}
  same_map = {}
  for edge in edges:
    cell_type_0 = exp_data.prot.obs.cell_type[edge[0]]
    cell_type_1 = exp_data.prot.obs.cell_type[edge[1]]
    if cell_type_0 == cell_type_1:
      common_same += 1
      same_map[cell_type_0] = same_map.get(cell_type_0, 0) + 1
    else:
      diff_map_c[cell_type_0 + "-" + cell_type_1] = diff_map_c.get(cell_type_0 + "-" + cell_type_1, 0) + 1
  print("# of correct:", common_same, " # of all: ", len(edges))
  print("Correctness:", common_same / len(edges))
  print("Correct mappings: ", same_map)
  print("Faulty mappings: ", diff_map_c)

def subplot_umap(exp_data, edges, coloring):
  import scanpy as sc
  flat_list = [cell for edge in edges for cell in edge]
  unique_cells = list(set(flat_list))
  filtered_adata = exp_data.whole[unique_cells, :]
  _adata = sc.AnnData(X = filtered_adata.X, obs = filtered_adata.obs, var = filtered_adata.var)
  sc.pp.neighbors(_adata)
  sc.tl.umap(_adata)
  sc.pl.umap(_adata, color = [coloring], cmap='viridis', use_raw=False)

def clean_weaks(edges, distances):
  cleaned_edges = []
  for edge in edges:
    if distances[edge[0], edge[1]] < 0.08:
      cleaned_edges.append(edge)
  return cleaned_edges

from scipy.spatial.distance import cdist

# def closest_pairing(distances, centers, communities):
#   saturated = 0
#   center_distances = cdist(centers, centers, metric='cosine')
#   np.fill_diagonal(center_distances, np.inf)
#   while (not saturated):
#     min_index = np.argmin(center_distances)
#     if center_distances.flat[min_index] < 0.3:
#       min_row, min_col = np.unravel_index(min_index, center_distances.shape)





def plot_new_umap(exp_data, edges, distances):
  import scanpy as sc
  from scipy.sparse import csr_matrix
  adata = exp_data.prot
  if "X_umap" in adata.obsm:
    del adata.obsm["X_umap"]
  adata.uns["neighbors"] = {}
  adata.uns["neighbors"]["connectivities_key"] = "connectivities"
  adata.uns["neighbors"]["distances_key"] = "distances"
  adata.obsp["connectivities"] = np.zeros((adata.shape[0], adata.shape[0]), dtype=bool)
  for edge in edges:
    adata.obsp['connectivities'][edge[0], edge[1]] = True
    adata.obsp['connectivities'][edge[1], edge[0]] = True
  adata.obsp['connectivities'] = csr_matrix(adata.obsp['connectivities'])
  adata.obsp["distances"] = distances
  adata.uns["neighbors"]["params"] = {'random_state': 0}

  sc.tl.umap(adata)
  print("printing...")
  sc.pl.umap(adata, color = "cell_type", cmap='viridis')

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

def positive_edges_by_count(distances, k):
  adj_matrix = np.zeros((distances.shape[0], distances.shape[0]))
  edges = []
  ctr_included = 0
  for i in range(distances.shape[0]):
    row = distances[i]
    sorted_indices = np.argsort(row)
    sorted_indices = sorted_indices[:k]
    for j in range(len(sorted_indices)):
      edges.append((i, sorted_indices[j]))
    adj_matrix[i, sorted_indices] = 1
  np.fill_diagonal(adj_matrix, 1)
  return edges, adj_matrix

def clean_supervised(edges, cell_type_list):
  _edges = []
  for edge in edges:
    if cell_type_list[edge[0]] == cell_type_list[edge[1]]:
      _edges.append(edge)
  return _edges

def clean_supervised_neg(edges, cell_type_list):
  _edges = []
  for edge in edges:
    if cell_type_list[edge[0]] != cell_type_list[edge[1]]:
      _edges.append(edge)
  return _edges

def clean_supervised_distance(distances, cell_type_list):
  for i in range(5000):
    for j in range(5000):
      if (cell_type_list[i] == "CD4 T" and  cell_type_list[j] == "CD8 T") or (cell_type_list[i] == "CD8 T" and  cell_type_list[j] == "CD4 T"):
        distances[i, j] = 1
  return distances

def exponential_normalize(matrix, exp_factor = 3):
    # Filter out inf values for normalization
    new_matrix = matrix.copy()
    # Calculate the median value
    finite_values = matrix[np.isfinite(matrix)]
    median = np.median(finite_values)
    # Apply exponential normalization
    new_matrix[np.isfinite(new_matrix)] = 1 - (1 / np.exp(exp_factor * new_matrix[np.isfinite(new_matrix)]))
    new_matrix[new_matrix < 0] = 0
    new_matrix[new_matrix > 1] = 1
    # print(np.median(new_matrix))
    np.fill_diagonal(new_matrix, np.inf)
    return new_matrix

def calculate_mnn_distance(adata, k=20):
  from scipy.spatial import cKDTree
  tree = cKDTree(adata.X)
  distances, indices = tree.query(adata.X, k=k+1) 
  mnn_pairs = set()
  for i in range(adata.shape[0]):
      for j in indices[i, 1:]:
          if i in indices[j, 1:]:
              mnn_pairs.add(tuple(sorted((i, j)))) 
  
  distance_map = np.ones((adata.shape[0], adata.shape[0]))
  for pair in mnn_pairs:
    distance_map[pair[0], pair[1]] = 0
    distance_map[pair[1], pair[0]] = 0
  return distance_map

def knn_neighbors(similarities, k):
  neighbors = []
  for cell in range(similarities.shape[0]):
    cell_similarity = similarities[cell].copy()
    cell_similarity[cell] = np.inf
    nearest_k = np.argsort(cell_similarity)[:k]
    neighbors.append(nearest_k)
  return neighbors

def find_elbo(distance_vector):
  x = np.arange(len(distance_vector))
  # first_point = (x[0], distance_vector[0])
  # last_point = (x[-1], distance_vector[-1])
  # m = (last_point[1] - first_point[1]) / (last_point[0] - first_point[0])
  # c = first_point[1] - m * first_point[0]
  # distances = np.abs(m * x - distance_vector + c) / np.sqrt(m**2 + 1)
  # return distance_vector[np.argmax(distances)]

  kneedle = KneeLocator(x, distance_vector, curve='convex', direction='decreasing')
  return distance_vector[kneedle.knee]

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

def built_graphs(exp_data):
  print("RNA similarities are calculating...")
  if 0 or "cell_similarity" not in exp_data.rna.obsm:
    rna_distances = pairwise_distances(exp_data.rna.obsm["embeddings"], metric = 'cosine')
    rna_neighbors = knn_neighbors(rna_distances, 20)
    rna_neighbors_sum = np.mean(np.array([row[cols] for row, cols in zip(rna_distances, rna_neighbors)]), axis = 1)
    for i in range(rna_distances.shape[0]):
      rna_distances[i] = rna_distances[i]/ rna_neighbors_sum[i]
    np.fill_diagonal(rna_distances, np.inf)
    exp_data.rna.obsm["cell_similarity"] = rna_distances
    # exp_data.rna.obsm["cell_similarity"] = pairwise_distances(exp_data.rna.X, metric='cosine')
    # exp_data.rna.obsm["cell_similarity"] = pairwise_distances(exp_data.rna[:, exp_data.rna.var["highly_variable"] == True].X.toarray(), metric='cosine')
    # hvgs = exp_data.rna[:, exp_data.rna.var["highly_variable"]!= False]
    # num_hvgs = hvgs.shape[1]
    # exp_data.rna.obsm["cell_similarity"] = fast_cell_similarity2(hvgs, np.ones((num_hvgs, num_hvgs)))
  print("Protein similarities are calculating...")
  if 0 or "cell_similarity" not in exp_data.prot.obsm:
    ppi_weights = pd.read_csv("/content/drive/MyDrive/SCPRO/Data/ppi_weights.csv")
    ppi_weights = filter_weights_legacy(ppi_weights, exp_data.prot.var.feature_name.values) 
    prot_distances = fast_cell_similarity(exp_data.prot, ppi_weights)
    prot_neighbors = knn_neighbors(prot_distances, 20)
    prot_neighbors_sum = np.mean(np.array([row[cols] for row, cols in zip(prot_distances, prot_neighbors)]), axis = 1)
    for i in range(prot_distances.shape[0]):
      prot_distances[i] = prot_distances[i]/ prot_neighbors_sum[i]
    exp_data.prot.obsm["cell_similarity"] =  prot_distances
    # print("Prot cell matching score:", cell_matching_score(exp_data.whole, exp_data.prot.obsm["cell_similarity"], 1, "D")[0])
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

    prot_edges, prot_adj = positive_edges(exp_data.prot.obsm["cell_similarity"], 20, 1) #GSE166895 P1_7 0.26 P2_3 0.22 P2_0 0.24
    # prot_edges, prot_adj = positive_edges_by_count(exp_data.prot.obsm["cell_similarity"], 250) #GSE166895 P1_7 0.26 P2_3 0.22 P2_0 0.24
    print("Number of prots:", len(prot_edges))

    rna_edges, rna_adj = positive_edges(exp_data.rna.obsm["cell_similarity"], 20, 1) # #GSE166895 P1_7 0.28 P2_3 0.1 P2_0 0.24
    # rna_edges, rna_adj = positive_edges_by_count(exp_data.rna.obsm["cell_similarity"], 250)
    print("Number of rnas:", len(rna_edges))

    combine_edges, combine_adj = positive_edges_by_count(distances_pr, 30)

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


