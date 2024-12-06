import pip
try:
    import scanpy as sc
except ModuleNotFoundError:
    pip.main(['install', 'scanpy'])
    import scanpy as sc

pip.main(['install', 'scikit-misc'])

import gc


import datetime
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
import seaborn as sns
from sklearn.decomposition import PCA

import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.utils.data import DataLoader

from . import Models, Matching
importlib.reload(Models)
importlib.reload(Matching)




class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class scMO():
    def __init__(self, path, sub_sample = True, load = False):
      if load:
        self.whole = sc.read_h5ad(path+"whole.h5ad")
        self.prot = sc.read_h5ad(path+"prot.h5ad")
        self.rna = sc.read_h5ad(path+"rna.h5ad")
      else:
        whole = sc.read_h5ad(path)
        # sc.pp.scale(whole)
        # row_sums = whole.X.sum(axis=1)
        # non_zero_indices = (row_sums != 0).A1
        # whole = whole[non_zero_indices, :]
        whole.uns = {}
        if sub_sample:
          rnd_cell_list = np.random.choice(whole.shape[0], 10000, replace=False)
          self.whole = whole[rnd_cell_list].copy()
          del whole
          gc.collect()
        else:
          self.whole = whole
        # self.whole.write("/content/drive/MyDrive/SCPRO/Data/GSE164378_sub.h5ad")
        

        # self.id_mapping()
        self.prot = self.whole[:, self.whole.var["feature_type"] == "protein"]
        self.rna = self.whole[:, self.whole.var["feature_type"] == "rna"]
        # self.min_max_scale()

        # measurements = self.prot.X.toarray()
        # min_val = np.min(measurements)
        # max_val = np.max(measurements)
        # normalized_measurements = (measurements - min_val) / (max_val - min_val)
        # self.prot.X = normalized_measurements

    def min_max_scale(self):

        # Ensure adata.X is a dense matrix for min-max scaling
        if not isinstance(self.prot.X, np.ndarray):
            vals = self.prot.X.toarray()
        else:
            vals = self.prot.X

        # Perform min-max scaling
        min_vals = vals.min(axis=0)
        max_vals = vals.max(axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        scaled_vals = (vals - min_vals) / range_vals

        # Update the adata.X matrix in-place
        self.prot.X = scaled_vals
    
    def plot_umap(self, data_type = "all", coloring = "cell_type", save = True, save_note =""):
        if isinstance(data_type, str):
          data_type = [data_type]
        for d_type in data_type:
          print(d_type, " is plotting ...")
          fig, ax = plt.subplots(figsize=(10, 8))
          if d_type == "all":
              if "neighbors" in self.whole.uns:
                  del self.whole.uns["neighbors"]
              if "X_umap" in self.whole.obsm:
                  del self.whole.obsm["X_umap"]
                  
              _adata = sc.AnnData(X = self.whole.X, obs = self.whole.obs, var = self.whole.var)
              
          elif d_type == "prot":
              _adata = sc.AnnData(X = self.prot.X, obs = self.prot.obs, var = self.prot.var)
              
          elif d_type == "rna":
            _adata = sc.AnnData(X = self.rna.X, obs = self.rna.obs, var = self.rna.var)

          elif d_type in self.whole.obsm:
            _adata = sc.AnnData(X = self.whole.obsm[d_type], obs = self.whole.obs, var = pd.DataFrame(pd.DataFrame(self.whole.obsm[d_type]).columns))
          else:
            print(d_type, " is not in .obsm!")
            break
          plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
          sc.pp.neighbors(_adata)
          sc.tl.umap(_adata)
          sc.pl.umap(_adata, color = [coloring], cmap='viridis', use_raw=False, show = not save, ax=ax)
          if save:
            file_name = str(datetime.datetime.now()).replace("-", "").replace(" ", "_").replace(":", "").split(".")[0] + "_" + d_type + "(" + save_note + ").pdf"
            plt.savefig("/content/drive/MyDrive/SCPRO/figures/" + file_name, format='pdf', dpi=300, bbox_inches='tight')
            plt.show()
          else:
            return plt
    def plot_gene_to_cell_frequency(self, data_type = "all", save = True):
      if data_type == "all":
        adata = self.whole
      elif data_type == "prot":
        adata = self.prot
      elif data_type == "rna":
        adata = self.rna
      else:
        print("Wrong data type! Please use one of all, prot or rna.")
        return
      # Calculate molecule counts per gene
      molecule_counts_per_gene = (adata.X > 0).sum(axis=0).A1
      zero_vals = molecule_counts_per_gene[molecule_counts_per_gene == 0].shape[0]
      # Filter out zero counts
      molecule_counts_per_gene_nonzero = molecule_counts_per_gene[molecule_counts_per_gene != 0]
      # Take logarithm of non-zero counts
      log_molecule_counts = np.log10(molecule_counts_per_gene_nonzero)
      # Define bins for molecule counts
      molecule_min = np.min(log_molecule_counts)
      molecule_max = np.max(log_molecule_counts)
      molecule_bins = np.arange(int(molecule_max) + 2)
      # Count number of cells in each bin
      cell_counts_in_bins = []
      cell_counts_in_bins.append(zero_vals)
      prev_bin_end = molecule_bins[0]
      for bin_end in molecule_bins[1:]:
          count_in_bin = np.sum((log_molecule_counts >= prev_bin_end) & (log_molecule_counts < bin_end))
          cell_counts_in_bins.append(count_in_bin)
          prev_bin_end = bin_end
      # Remove duplicate labels
      unique_labels = ["10^"+ str(molecule_bins[i]) +" - 10^"+ str(molecule_bins[i + 1]) for i in range(len(molecule_bins) - 1)]
      unique_labels.insert(0, "0")
      # Plot
      plt.figure(figsize=(10, 6))
      plt.bar(np.arange(len(cell_counts_in_bins)), cell_counts_in_bins, width=0.8, align='center', color='skyblue')
      plt.xlabel('log(Number of Cells)')
      plt.ylabel('Frequency of genes')
      plt.title('Frequency of Genes for Each Range of Sequenced Cell Numbers')
      plt.xticks(np.arange(len(unique_labels)), unique_labels)
      intervals = 10 ** max(0, (int(np.log10(adata.shape[1])) - 1))
      plt.yticks(np.arange(min(cell_counts_in_bins), max(cell_counts_in_bins) + intervals, intervals))
      plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
      if save:
        file_name = str(datetime.datetime.now()).replace("-", "").replace(" ", "_").replace(":", "").split(".")[0] + "_cell_to_genes.pdf"
        plt.savefig("/content/drive/MyDrive/SCPRO/figures/" + file_name, dpi=300, bbox_inches='tight')
      plt.show()
      print("Cell counts in bins: ", cell_counts_in_bins)
    def filter_cells_by_feature_count(self, data_type = "all", k = 500):
      if data_type == "all":
        adata = self.whole
      elif data_type == "prot":
        adata = self.prot
      elif data_type == "rna":
        adata = self.rna
      else:
        print("Wrong data type! Please use one of all, prot or rna.")
        return
      if isinstance(adata.X, np.ndarray):
        vals = adata.X
      else:
        vals = adata.X.toarray()
      non_zero_gene_counts = np.count_nonzero(vals, axis=1)
      mask = non_zero_gene_counts >= k

      self.whole = self.whole[mask, :]
      self.prot = self.prot[mask, :]
      self.rna = self.rna[mask, :]

    def filter_features_by_cell_count(self, data_type = "all", k = 500):
      if data_type == "all":
        adata = self.whole
      elif data_type == "prot":
        adata = self.prot
      elif data_type == "rna":
        adata = self.rna
      else:
        print("Wrong data type! Please use one of all, prot or rna.")
        return
      if isinstance(adata.X, np.ndarray):
        vals = adata.X
      else:
        vals = adata.X.toarray()

      non_zero_gene_counts = np.count_nonzero(vals, axis=0)
      mask = non_zero_gene_counts >= k
      adata = adata[:, mask]
      if data_type == "all":
        self.whole = adata
      elif data_type == "prot":
        self.prot = adata
      elif data_type == "rna":
        self.rna = adata

    def load_embedding(self, key, data_path, cols):
      embs = pd.read_csv(data_path)
      self.whole.obsm[key] = np.array(embs.iloc[:, cols])

    def id_mapping(self):
      mappings = pd.read_excel("/content/drive/MyDrive/SCPRO/Data/prot_map.xlsx")
      map_dict = dict(zip(mappings['From'], mappings['Entry']))
      new_ids = np.array(self.whole.var.feature_name)
      for i in  range(new_ids.shape[0]):
        new_ids[i] = map_dict.get(new_ids[i], new_ids[i])
      self.whole.var["feature_name"] = new_ids


    def write(self, path):

      
      import anndata as ad

      if issparse(self.prot.X):
        dense_X = self.prot.X.toarray()  # Convert to a dense numpy array
      else:
        dense_X = self.prot.X  # If already dense, keep it as is
      new_adata = ad.AnnData(X=dense_X, 
                       obs=self.prot.obs.copy(), 
                       var=self.prot.var.copy(), 
                       uns=self.prot.uns.copy(), 
                       obsm=self.prot.obsm.copy(), 
                       varm=self.prot.varm.copy(), 
                       layers=self.prot.layers.copy())
      self.prot = new_adata
      
      if issparse(self.rna.X):
        dense_X = self.rna.X.toarray()  # Convert to a dense numpy array
      else:
        dense_X = self.rna.X  # If already dense, keep it as is
      new_adata = ad.AnnData(X=dense_X, 
                       obs=self.rna.obs.copy(), 
                       var=self.rna.var.copy(), 
                       uns=self.rna.uns.copy(), 
                       obsm=self.rna.obsm.copy(), 
                       varm=self.rna.varm.copy(), 
                       layers=self.rna.layers.copy())
      self.rna = new_adata

      self.whole.write(path+"whole.h5ad")
      self.prot.write(path+"prot.h5ad")
      self.rna.write(path+"rna.h5ad")

def load_data(data_path, sub_sample, load):
  return scMO(data_path, sub_sample, load)

def rna_embedding(adata):
  if isinstance(adata.X, np.ndarray):
    data_matrix = adata.X
  else:
    data_matrix = adata.X.toarray()
  data_size = data_matrix.shape[1]

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data_loader = DataLoader(data_matrix, batch_size=32, shuffle=True)
  epochs=10
  lr=1e-3
  latent_size =100
  dense_layer_size = 256
  vae = Models.VAE_torch(data_size, dense_layer_size, latent_size).to(device)
  optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
  
  for epoch in range(epochs):
      vae.train()
      train_loss = 0
      for batch in data_loader:
          batch = batch.view(batch.size(0), -1).to(device)
          optimizer.zero_grad()
          x_hat, mean, logvar = vae(batch)
          loss = vae.loss_function(batch, x_hat, mean, logvar)
          loss.backward()
          train_loss += loss.item()
          optimizer.step()
      print(f'Epoch {epoch+1}, Loss: {train_loss/len(data_loader.dataset)}')
  all_data = torch.tensor(data_matrix, device = device)
  x_hat, mean, logvar = vae(all_data)
  return mean.detach().cpu().numpy()



def rna_embedding_pca(adata):
  print("PCA components are calculating for RNAs...")
  # pca = PCA(n_components=25)
  pca = PCA(n_components=25)
  if issparse(adata.X):
    pca_embeddings = pca.fit_transform(adata.X.toarray())
  else:
    pca_embeddings = pca.fit_transform(adata.X)
  return pca_embeddings

def cell_embedding(scmo, vae_type):
  k = 2000
  sc.pp.highly_variable_genes(scmo.prot, flavor='seurat_v3', n_top_genes=scmo.prot.shape[1])
  hvg_scores = scmo.prot.var['highly_variable_rank']
  prot_importance = (hvg_scores - np.min(hvg_scores)) / (np.max(hvg_scores) - np.min(hvg_scores)) 

  sc.pp.highly_variable_genes(scmo.rna, flavor='seurat_v3', n_top_genes=k)
  hvg_scores = scmo.rna.var['highly_variable_rank']
  rna_importance = (hvg_scores - np.min(hvg_scores)) / (np.max(hvg_scores) - np.min(hvg_scores)) 
  
  top_indices = np.argsort(rna_importance)[-k:]
  rna_importance = rna_importance[top_indices]

  prot_matrix = scmo.prot.X.toarray()
  rna_matrix = scmo.rna[:, scmo.rna.var["highly_variable"]!= False].X.toarray()
  
  if vae_type == "VAE-CONCAT":
    prot_size = rna_matrix.shape[1] + prot_matrix.shape[1]
  else:
    prot_size = prot_matrix.shape[1]

  args = Namespace(vtype = vae_type,
                    inp_prot_size = prot_size, 
                    inp_rna_size = rna_matrix.shape[1], 
                    dense_layer_size = 256, 
                    latent_size = 100, 
                    dropout = 0.2, 
                    beta = 1, 
                    epochs = 10, 
                    batch_size = 16,
                    prot_importances = prot_importance,
                    rna_importances = rna_importance,
                    save_model = False)
  vae_model = Models.VAE(args)
  if vae_type == "VAE-CONCAT":
    vae_model.train(np.concatenate((prot_matrix, rna_matrix), axis=1))
    # vae_model.train(prot_matrix)
  else:
    vae_model.train(prot_matrix, rna_matrix)

  if vae_type == "VAE-CONCAT":
    scmo.whole.obsm["VAE_" + vae_type] = np.array(vae_model.predict(np.concatenate((prot_matrix, rna_matrix), axis=1)))
    # scmo.whole.obsm["integrated_" + vae_type] = np.array(vae_model.predict(prot_matrix))
  else:
    scmo.whole.obsm["VAE_" + vae_type] = np.array(vae_model.predict(prot_matrix, rna_matrix))
  print("VAE integration is done. The embeddings are saved in .obsm['VAE_" + vae_type +"'].")

def edge_list_to_adj_matrix(edge_index_p, num_nodes=None, directed=False):

    rows = torch.tensor([edge[0] for edge in edge_index_p])
    cols = torch.tensor([edge[1] for edge in edge_index_p])

    if num_nodes is None:
        num_nodes = torch.max(edge_index_p) + 1
    
    # Initialize an empty adjacency matrix (num_nodes x num_nodes)
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
    adj_matrix[rows, cols] = 1
    return adj_matrix


def cell_embedding_svgae(scmo, args):
  # torch.manual_seed(42)
  torch.cuda.empty_cache()
  
  if args.use_embeddings:
    if "embeddings" not in scmo.rna.obsm:
      scmo.rna.obsm["embeddings"] = rna_embedding_pca(scmo.rna) # rna_embedding(scmo.rna)
    x_data_rna = scmo.rna.obsm["embeddings"]
  else:
    if args.num_hvgs != -1:
      # if "highly_variable" not in scmo.rna.var:
      sc.pp.highly_variable_genes(scmo.rna, flavor='seurat_v3', n_top_genes=args.num_hvgs)
      x_data_rna = scmo.rna[:, scmo.rna.var["highly_variable"]!= False].X.toarray()
    else:
      if isinstance(scmo.rna.X, np.ndarray):
        vals = scmo.rna.X
      else:
        vals = scmo.rna.X.toarray()
      x_data_rna = vals
  num_nodes, rna_dim = x_data_rna.shape
  x_data_rna = torch.tensor(x_data_rna, dtype=torch.float32)

  num_nodes, prot_dim = scmo.prot.shape
  if isinstance(scmo.prot.X, np.ndarray):
    vals = scmo.prot.X
  else:
    vals = scmo.prot.X.toarray()
  x_data_prot = vals
  x_data_prot = torch.tensor(x_data_prot, dtype=torch.float32)

  if "cell_similarity" not in scmo.whole.obsm:
    Matching.built_graphs(scmo)
  edge_index_p = torch.tensor(scmo.whole.uns["prot_edges"]) 
  edge_index_r = torch.tensor(scmo.whole.uns["rna_edges"])

  adj_matrix_p = torch.tensor(1 - scmo.prot.obsm["cell_similarity_top"])
  adj_matrix_p.fill_diagonal_(1)

  adj_matrix_r = torch.tensor(1 - scmo.rna.obsm["cell_similarity_top"])
  adj_matrix_r.fill_diagonal_(1)

  model_p = Models.SGVAE(prot_dim, args.hidden_dim, args.latent_dim)
  model_r = Models.SGVAE(rna_dim, args.hidden_dim, args.latent_dim)
  optimizer_p = model_p.set_optimizer(model_p.parameters())
  optimizer_r = model_p.set_optimizer(model_r.parameters())
  writer_p = Models.log_writer("SGVAE_prot_dim_" + str(args.latent_dim))
  writer_r = Models.log_writer("SGVAE_rna_dim_" + str(args.latent_dim))

  model = Models.SEncoder(args.latent_dim, int(args.latent_dim / 2))
  optimizer = model.set_optimizer(model.parameters())
  writer = Models.log_writer("SGVAE_encoder_dim_" + str(args.latent_dim))

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data_p = Data(x=x_data_prot, edge_index=edge_index_p)
  data_r = Data(x=x_data_rna, edge_index=edge_index_r)

  # prot_adj_matrix = edge_list_to_adj_matrix(edge_index_p)
  # prot_adj_matrix.to(device)
  # rna_adj_matrix = edge_list_to_adj_matrix(edge_index_r)
  # rna_adj_matrix.to(device)
  
  train_loader_p = NeighborLoader(data_p, num_neighbors=args.num_neighbors, batch_size=num_nodes, shuffle=False)
  train_loader_r = NeighborLoader(data_r, num_neighbors=args.num_neighbors, batch_size=num_nodes, shuffle=False)

  data_p = data_p.to(device)
  data_r = data_r.to(device)
  model_p = model_p.to(device)
  model_r = model_r.to(device)
  adj_matrix_p = adj_matrix_p.to(device)
  # print(adj_matrix_r)
  adj_matrix_r = adj_matrix_r.to(device)
  # model_p.train()
  # optimizer_p.zero_grad()
  # recon_x_p_all = model_p(data_p.x, data_p.edge_index)[0].detach()

  # model_r.train()
  # optimizer_r.zero_grad()
  # recon_x_r_all = model_r(data_r.x, data_r.edge_index)[0].detach()
  progress_bar = tqdm(total=args.num_epochs, desc='Training', unit='epoch', position=0)
  for epoch in range(args.num_epochs):
    model_p.train()
    for batch in train_loader_p:
        optimizer_p.zero_grad()
        # recon_x_p, mu_p, logvar_p, z_p = model_p(x_data_prot, edge_index_p)
        recon_x_p, mu_p, logvar_p, z_p = model_p(batch.x, batch.edge_index)
        # recon_x_p_all[batch.n_id[:, None], batch.n_id] = recon_x_p.detach()
        recon_loss_p = Models.recon_loss(recon_x_p, adj_matrix_p)
        # recon_loss_p = Models.recon_loss(recon_x_p_all, adj_matrix_p)
        KL_p = Models.kl_loss(mu=mu_p, logvar=logvar_p, n_nodes=num_nodes)
        loss_p = recon_loss_p + KL_p 
        loss_p.backward()
        optimizer_p.step()
        # writer_p.write("prot loss",loss_p.item(),epoch)


    model_r.train()
    
    for batch in train_loader_r:
      optimizer_r.zero_grad()
      # recon_x_r, mu_r, logvar_r, z_r = model_r(x_data_rna, edge_index_r)
      recon_x_r, mu_r, logvar_r, z_r = model_r(batch.x, batch.edge_index)
      recon_loss_r = Models.recon_loss(recon_x_r, adj_matrix_r)
      # recon_x_r_all[batch.n_id[:, None], batch.n_id] = recon_x_r.detach()
      # recon_loss_r = Models.recon_loss(recon_x_r_all, adj_matrix_r)
      
      KL_r = Models.kl_loss(mu=mu_r, logvar=logvar_r, n_nodes=num_nodes)
      # print(recon_loss_r)
      # print(np.isnan(mu_r.cpu().detach().numpy()).any(), np.isnan(logvar_r.cpu().detach().numpy()).any(), KL_r, recon_loss_r)
      loss_r = recon_loss_r + KL_r
      loss_r.backward()
      optimizer_r.step()  
      
    progress_bar.set_postfix({'Prot Loss': loss_p.item(), 'RNA Loss': loss_r.item()})
    progress_bar.update()
        # writer_p.write("rna loss",loss_r.item(),epoch)
      # print(f"Epoch [{epoch+1}/{args.num_epochs}] Prot Loss: {loss_p.item():.4f}", f", rna Loss: {loss_r.item():.4f}")
  progress_bar.close()
  if args.pretrained:
    return model_p, model_r
  else:
    num_epoch_emb = args.num_epochs * 1
    adj_similarity = torch.tensor(1 - scmo.whole.obsm["cell_similarity"])
    adj_similarity.fill_diagonal_(1)
    adj_similarity.to(device)
    for epoch in range(num_epoch_emb):
        model.train()
        optimizer.zero_grad()
        recon_x, z = model(z_p, z_r)
        loss = Models.recon_loss(recon_x, adj_similarity)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epoch_emb}] embedding Loss: {loss.item():.4f}")
        writer.write("embedding loss",loss.item(),epoch)


    scmo.whole.obsm["SCPRO-VI-SEQ"] = z.detach().numpy()
    print("SCPRO-VI-SEQ integration is done. The embeddings are saved in .obsm['SCPRO-VI'].")

def cell_embedding_vgae(scmo, args):
  # torch.manual_seed(42)
  torch.cuda.empty_cache()
  import gc

  if args.pretrained:
    model_p, model_r = cell_embedding_svgae(scmo, args)
  else:
    model_p = model_r = None
  gc.collect()
  
  # if args.num_hvgs != -1:
  #   if "highly_variable" not in scmo.rna.var:
  #     sc.pp.highly_variable_genes(scmo.rna, flavor='seurat_v3', n_top_genes=args.num_hvgs)
  #   x_data_rna = scmo.rna[:, scmo.rna.var["highly_variable"]!= False].X.toarray()
  #   rna_dim = args.num_hvgs
  # else:
  #   if isinstance(scmo.rna.X, np.ndarray):
  #     vals = scmo.rna.X
  #   else:
  #     vals = scmo.rna.X.toarray()
  #   x_data_rna = vals
  #   num_nodes, rna_dim = scmo.rna.shape
  if args.use_embeddings:
    if "embeddings" not in scmo.rna.obsm:
      scmo.rna.obsm["embeddings"] = rna_embedding(scmo.rna)
    x_data_rna = scmo.rna.obsm["embeddings"]
  else:
    if args.num_hvgs != -1:
      # if "highly_variable" not in scmo.rna.var:
      sc.pp.highly_variable_genes(scmo.rna, flavor='seurat_v3', n_top_genes=args.num_hvgs)
      x_data_rna = scmo.rna[:, scmo.rna.var["highly_variable"]!= False].X.toarray()
    else:
      if isinstance(scmo.rna.X, np.ndarray):
        vals = scmo.rna.X
      else:
        vals = scmo.rna.X.toarray()
      x_data_rna = vals
  num_nodes, rna_dim = x_data_rna.shape
  x_data_rna = torch.tensor(x_data_rna, dtype=torch.float32)

  num_nodes, prot_dim = scmo.prot.shape
  if isinstance(scmo.prot.X, np.ndarray):
    vals = scmo.prot.X
  else:
    vals = scmo.prot.X.toarray()
  x_data_prot = vals
  x_data_prot = torch.tensor(x_data_prot, dtype=torch.float32)

  if "cell_similarity" not in scmo.whole.obsm:
    Matching.built_graphs(scmo)
  edge_index_p = torch.tensor(scmo.whole.uns["prot_edges"]) 
  edge_index_r = torch.tensor(scmo.whole.uns["rna_edges"])
  # adj_similarity = torch.tensor(exp_data.whole.uns["distances_pr"])
  adj_similarity = torch.tensor(1 - scmo.whole.obsm["cell_similarity"])
  adj_similarity.fill_diagonal_(1)

  model = Models.GraphVAE(prot_dim, rna_dim, args.hidden_dim, args.latent_dim, args.pretrained, model_p, model_r)

  optimizer = model.set_optimizer(model.parameters())

  writer = Models.log_writer("GVAE_dim_" + str(args.latent_dim))

  # adj_matrix = torch.zeros(5000, 5000, dtype=torch.float32)
  # adj_matrix[edge_index_u[0], edge_index_u[1]] = 1
  # adj_matrix.fill_diagonal_(1)

  # pruning_thp = 1 #0.40
  # pruning_thr = 0.55 #0.55
  # prot_dist = scmo.prot.obsm["cell_similarity"].copy()
  # prot_dist[prot_dist > pruning_thp] = 1 

  adj_matrix_p = torch.tensor(1- scmo.prot.obsm["cell_similarity_top"])
  adj_matrix_p.fill_diagonal_(1)

  adj_matrix_r = torch.tensor(1 - scmo.rna.obsm["cell_similarity_top"])
  adj_matrix_r.fill_diagonal_(1)



  print(torch.min(adj_similarity), torch.max(adj_similarity))
  # print("# of 0 distances: ", np.sum(adj_similarity.detach().numpy() == 0))
  # edge_weights = torch.tensor(adj_similarity[pos_mask].view(-1, 1), dtype = torch.float32)



  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data_p = Data(x=x_data_prot, edge_index=edge_index_p)
  data_r = Data(x=x_data_rna, edge_index=edge_index_r)
  
  train_loader_p = NeighborLoader(data_p, num_neighbors=args.num_neighbors, batch_size=num_nodes, shuffle=False)
  train_loader_r = NeighborLoader(data_r, num_neighbors=args.num_neighbors, batch_size=num_nodes, shuffle=False)

  data_p = data_p.to(device)
  data_r = data_r.to(device)
  model = model.to(device)

  adj_matrix_p = adj_matrix_p.to(device)
  adj_matrix_r = adj_matrix_r.to(device)
  adj_similarity = adj_similarity.to(device)


  # progress_bar = tqdm(total=args.num_epochs, desc='Multi-view', unit='epoch', position=1, leave = 1)
  for epoch in range(args.num_epochs * 2):
    model.train()
    optimizer.zero_grad()
    for batch_p in train_loader_p:
      for batch_r in train_loader_r:
        # recon_x_p, recon_x_r, recon_x_u, mu_p, mu_r, logvar_p, logvar_r, z = model(x_data_prot, x_data_rna, edge_index_p, edge_index_r, edge_index_u) # for generating two graphs individually.
        recon_x_p, recon_x_r, recon_x_u, mu_p, mu_r, logvar_p, logvar_r, z = model(batch_p.x, batch_r.x, batch_p.edge_index, batch_r.edge_index) # for generating two graphs individually.
        
        # pos_loss_p = Models.recon_loss(recon_x_p, adj_matrix_p)
        # pos_loss_r = Models.recon_loss(recon_x_r, adj_matrix_r)
        pos_loss_u = Models.recon_loss(recon_x_u, adj_similarity)

        KL_p = Models.kl_loss(mu=mu_p, logvar=logvar_p, n_nodes=num_nodes)
        KL_r = Models.kl_loss(mu=mu_r, logvar=logvar_r, n_nodes=num_nodes)
        # adversarial_loss = Models.adversarial_loss(cc_preds, cc_labels)
        # print(pos_loss_p, pos_loss_r, pos_loss_u, neg_loss_p, neg_loss_r, neg_loss_u)
        # print(adversarial_loss)
        # print(pos_loss_u, KL_p, KL_r)
        # loss = 10 * pos_loss_u + neg_loss_u + KL_p + KL_r 
        # loss = pos_loss_p + pos_loss_r + pos_loss_u + KL_p + KL_r 
        loss = pos_loss_u + KL_p + KL_r 
        # loss = pos_loss_p + pos_loss_r + (2 * pos_loss_u) + KL_p + KL_r 
        # loss = pos_loss_p +  neg_loss_p  + Models.kl_loss(mu=mu, logvar=logvar, n_nodes=num_nodes)
        loss.backward()
        optimizer.step()
    # progress_bar.set_postfix({'Loss': loss.item()})
    # progress_bar.update()
        # writer_p.write("rna loss",loss_r.item(),epoch)
      # print(f"Epoch [{epoch+1}/{args.num_epochs}] Prot Loss: {loss_p.item():.4f}", f", rna Loss: {loss_r.item():.4f}")
  # progress_bar.close()
      print(f"Epoch [{epoch+1}/{args.num_epochs * 2}] Loss: {loss.item():.4f}")
      # writer.write("loss",loss.item(),epoch)
  if torch.cuda.is_available():
    scmo.whole.obsm["SCPRO-VI"] = z.detach().cpu().numpy()
  else:
    scmo.whole.obsm["SCPRO-VI"] = z.detach().numpy()
  print("SCPRO-VI integration is done. The embeddings are saved in .obsm['SCPRO-VI'].")

def run_mofa(scmo, args):
  import pip
  try:
      import mudata as md
  except ModuleNotFoundError:
      pip.main(['install', 'mudata'])
      import mudata as md

  try:
      import muon as mu
  except ModuleNotFoundError:
      pip.main(['install', 'muon'])
      import muon as mu

  try:
      import mofapy2
  except ModuleNotFoundError:
      pip.main(['install', 'mofapy2'])
      import mofapy2

  _mudata = mu.MuData({"rna": scmo.rna, "adt": scmo.prot})
  if hasattr(args, "batch"):
    _mudata.obs["batch"] = _mudata["rna"].obs[args.batch]
    _mudata.obs["batch"].astype("category")
    mu.tl.mofa(_mudata, groups_label="batch",  gpu_mode=True)
  else:
    mu.tl.mofa(_mudata, gpu_mode=True)
  scmo.whole.obsm["mofa"] = _mudata.obsm['X_mofa']
  print("Mofa integration is done. The embeddings are saved in .obsm['mofa'].")

def run_mowgli(scmo, args):
  import pip
  try:
      import mudata as md
  except ModuleNotFoundError:
      pip.main(['install', 'mudata'])
      import mudata as md

  try:
      import muon as mu
  except ModuleNotFoundError:
      pip.main(['install', 'muon'])
      import muon as mu

  try:
      import mowgli
  except ModuleNotFoundError:
      pip.main(['install', 'mowgli'])
      import mowgli
  
  sc.pp.highly_variable_genes(scmo.rna, flavor='seurat_v3', n_top_genes=200)
  scmo.prot.var["highly_variable"] = True
  mdata = mu.MuData({"rna": scmo.rna, "adt": scmo.prot})
  model = mowgli.models.MowgliModel(latent_dim=15)
  model.train(mdata, device= "cuda")
  scmo.whole.obsm["Mowgli"] = mdata.obsm["W_OT"]
  print("Mowgli integration is done. The embeddings are saved in .obsm['Mowgli'].")


def run_totalVI(scmo, args):
  try:
      import  scvi
  except ModuleNotFoundError:
      pip.main(['install', ' scvi-tools'])
      import  scvi

  if hasattr(args, "batch"):
    _batch = args.batch
  else:
    _batch = None
  if hasattr(args, "layer"):
    _layer = args.layer
  else:
    _layer = None
  if "raw_counts" in scmo.prot.obsm:
    scmo.rna.uns["prot_names"] = scmo.prot.var.feature_name.values
    scmo.rna.obsm["prot_raw_counts"] = scmo.prot.obsm["raw_counts"]
  else:
    print("TotalVI needs raw protein counts to operate. Please save unnormalized protein counts in adata rna with 'prot_raw_counts' keyword.")
    return

  scvi.model.TOTALVI.setup_anndata(scmo.rna, protein_expression_obsm_key = "prot_raw_counts", protein_names_uns_key = "prot_names")
  model = scvi.model.TOTALVI(scmo.rna)
  model.to_device('cuda:0')
  model.train()
  scmo.whole.obsm["TotalVI"] = model.get_latent_representation()
  print("TotalVI integration is done. The embeddings are saved in .obsm['TotalVI'].")

def enhancing(scmo):
  args = Namespace(l1_size = 15)
  gcn_model = Models.GCN(args)
  gcn_model.built_graphs(scmo)

class VI():
  # vae_models = ["VAE-CONCAT", "VAE-EARLY", "VAE-INT","VAE-LATE"]
  # for model in vae_models:
  #   print(model, " is under process...")
  #   cell_embedding(scmo, model)
  def scpro_vi(data, args = None):
    cell_embedding_vgae(data, args)
  def vae(data, args = None):
    cell_embedding(data, args.model)
  def mofa(data, args = None):
    run_mofa(data,args)
  def totalVI(data, args = None):
    run_totalVI(data,args)
  def Mowgli(data, args = None):
    run_mowgli(data,args)
  def scpro_vi_sequential(data, args = None):
    cell_embedding_svgae(data, args)
  # enhancing(scmo)

