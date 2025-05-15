# SCPRO-VI

**SCPRO-VI: Structure-aware Cell Embedding for Protein and RNA-based Multi-Omics Data Integration**

> **Preprint**: [bioRxiv, 2024.12.06.627151v1](https://www.biorxiv.org/content/10.1101/2024.12.06.627151v1)

---

## 🧬 Abstract

Integration of single-cell multi-omics data, particularly transcriptomics and proteomics, remains a central challenge due to modality heterogeneity and scale imbalances. **SCPRO-VI** is a structure-aware variational inference model that introduces a biologically motivated similarity calculation using protein-protein interaction (PPI) networks and combines RNA and protein modalities via graph-based variational autoencoders. Our method builds graph topologies reflecting intra-modality consistency and inter-modality agreement to yield an informative latent embedding. SCPRO-VI improves biological interpretability and enhances cell type resolution, especially in sparse or imbalanced modalities.

---

## 🧰 Installation

Clone the repository and install dependencies using:

```bash
git clone https://github.com/YOUR_USERNAME/SCPRO-VI.git
cd SCPRO-VI
pip install -r requirements.txt
```

You will need Python ≥ 3.8 with GPU support (for PyTorch and CuPy). Dependencies include:

- `scanpy`, `scikit-learn`, `cupy`, `torch`, `torch_geometric`
- Optional: `muon`, `mofapy2`, `mowgli`, `scvi-tools` (for comparing with MOFA, Mowgli, TotalVI)

---

## 📁 Repository Structure

```bash
├── SCPRO_VI_Github.ipynb        # Main usage notebook
├── Matching.py                  # Cell-cell graph construction
├── scPROMO.py                   # Main training and integration pipeline
├── Models/                      # (Expected) neural model definitions
├── Data/                        # Place for raw input data and PPI info
└── README.md                    # This file
```

---

## 📦 Input Format

SCPRO-VI expects `.h5ad` files with RNA and protein features annotated as:

- `feature_type` column in `.var` with values `rna` or `protein`
- Cell type annotations in `.obs["cell_type"]`
- PPI network: CSV file `ppi_weights.csv` with columns `subs1`, `subs2`, `combined_score`

---

## 🚀 Getting Started

1. **Prepare Input Data**  
Place the `.h5ad` file (combined RNA+protein) in the `Data/` directory, and ensure a PPI file `ppi_weights.csv` exists.

2. **Load and Preprocess**

```python
from scPROMO import load_data

data_path = "Data/dataset_name.h5ad"
scmo = load_data(data_path, sub_sample=True, load=False)

# Optional cleaning
scmo.filter_cells_by_feature_count("all", k=500)
scmo.filter_features_by_cell_count("all", k=500)
scmo.min_max_scale()
```

3. **Graph Construction**

```python
from Matching import built_graphs
built_graphs(scmo)
```

4. **Run SCPRO-VI**

```python
from scPROMO import VI
from types import SimpleNamespace

args = SimpleNamespace(
    latent_dim=100,
    hidden_dim=256,
    num_neighbors=[15],
    num_epochs=20,
    num_hvgs=1000,
    pretrained=False,
    use_embeddings=True
)

VI.scpro_vi(scmo, args)
```

5. **Visualization**

```python
scmo.plot_umap(data_type="SCPRO-VI", coloring="cell_type")
```

---

## 📌 Other Supported Methods

The `scPROMO.py` file also supports:

```python
VI.vae(scmo, args)        # Standard VAE
VI.scpro_vi_sequential(scmo, args)  # SGVAE sequential variant
VI.mofa(scmo, args)       # MOFA integration
VI.Mowgli(scmo, args)     # Mowgli integration
VI.totalVI(scmo, args)    # TotalVI integration
```

---

## 📊 Output

- Cell embeddings: `adata.obsm["SCPRO-VI"]`
- Graphs: `adata.uns["rna_edges"]`, `adata.uns["prot_edges"]`, and `.obsm["cell_similarity"]`

These can be used for downstream clustering, trajectory inference, and visualization.

---

## 📃 Citation

If you use SCPRO-VI in your research, please cite:

```bibtex
@article{SCPROVI2024,
  title={SCPRO-VI: Structure-aware Cell Embedding for Protein and RNA-based Multi-Omics Data Integration},
  author={Your Name and Others},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.12.06.627151}
}
```

---

## 📮 Contact

For questions or collaboration, feel free to contact **[Your Name]** at `your_email@domain.com`.
