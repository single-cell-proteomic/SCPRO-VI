import pandas as pd
import numpy as np

def scib_to_tabular(analysis_name):
  res_all = pd.read_excel(r"/content/drive/MyDrive/SCPRO/Experiments/Results/"+ analysis_name +".xlsx",index_col=0)
  res_all = res_all.T

  # res_all["Donor"] = "None"
  res_all["Time"] = "None"
  res_all["Algorithm"] = "None"
  i = 0
  for row, index in res_all.iterrows():
    res_all.iloc[i, 15] = "1"
    res_all.iloc[i, 16] = index.name
    # res_all.iloc[i, 16] = index.name[3]
    # res_all.iloc[i, 17] = index.name[6:]
    i += 1
  res_all.to_excel("/content/drive/MyDrive/SCPRO/Experiments/Results/"+ analysis_name +"_tabular.xlsx")

def cell_matching_dict(edges, cell_type_list):
  diff_map_c = {}
  same_map_c = {}
  common_same = 0

  cell_types = cell_type_list.unique()
  for cell_type in cell_types:
    for cell_type2 in cell_types:
      same_map_c[(cell_type,cell_type2)] = 0
      same_map_c[(cell_type2,cell_type)] = 0

  for i in range(len(edges[0])):
    edge = (edges[0][i], edges[1][i])
    same_map_c[(cell_type_list[edge[0]], cell_type_list[edge[1]])] += 1
  print("# of True Matchings:", common_same / len(edges))
  print(same_map_c)
  print(diff_map_c)

def process_new_datasets():
  import scanpy as sc
  import numpy as np
  new_dataset = sc.read_h5ad("/content/drive/MyDrive/SCPRO/Data/GSE166895_rna.h5ad")
  new_dataset_prot = sc.read_h5ad("/content/drive/MyDrive/SCPRO/Data/GSE166895_Prot.h5ad")

  # check some props
  new_dataset.obs.keys()
  new_dataset_prot.obs
  new_dataset.shape
  new_dataset.uns

  sorted_new_dataset = new_dataset[new_dataset.obs.sort_index().index]
  sorted_new_dataset_prot = new_dataset_prot[new_dataset_prot.obs.sort_index().index]
  new_dataset.var["feature_type"] = "rna"
  new_dataset_prot.var["feature_type"] = "protein"
  new_dataset_prot.var["feature_name"] = new_dataset_prot.var.index.values
  index_new = np.concatenate([new_dataset.var.index.values, new_dataset_prot.var.index.values])

  feature_types = np.concatenate([new_dataset.var["feature_type"].values, new_dataset_prot.var["feature_type"].values])
  feature_names = np.concatenate([new_dataset.var["feature_name"].values, new_dataset_prot.var["feature_name"].values])
  new_var = pd.DataFrame({"feature_type": feature_types, "feature_name": feature_names}, index=index_new)



  new_X = np.concatenate((new_dataset.X.toarray(), new_dataset_prot.X), axis=1)
  new_adata = sc.AnnData(new_X, obs=new_dataset.obs, var=new_var)
  new_adata.write("/content/drive/MyDrive/SCPRO/Data/GSE166895.h5ad")

def uniprot_map(adata, df_map):
  map_ = {}
  for index, row in df_map.iterrows():
    map_[row[0]] = row[1]
    if map_[row[0]] == None:
      map_[row[0]] = row[0]
  old_ids = np.array(adata.var.feature_name.values)
  new_ids = []
  for id_ in old_ids:
    new_ids.append(map_[id_])
  adata.var.index = np.array(new_ids)

def reset_gpu_memory():
  import pip
  pip.main(["install", "numba"])
  from numba import cuda
  device = cuda.get_current_device()
  device.reset()



































