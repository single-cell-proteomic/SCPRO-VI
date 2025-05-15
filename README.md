# SCPRO-VI

**Explainable Graph Learning for Multimodal Single-Cell Data Integration**

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
```

You will need Python ≥ 3.8 with GPU support (for PyTorch and CuPy). Dependencies include:

- `scanpy`, `scikit-learn`, `cupy`, `torch`, `torch_geometric`
- Optional: `muon`, `mofapy2`, `mowgli`, `scvi-tools` (for comparing with MOFA, Mowgli, TotalVI)

---



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
scmo = load_data(data_path, sub_sample=False, load=False)

# Optional cleaning
scmo.filter_cells_by_feature_count("all", k=10)
scmo.filter_features_by_cell_count("all", k=250)
scmo.min_max_scale()
```


4. **Run SCPRO-VI**

```python
from scPROMO import VI
from scPROMO import Namespace

args = Namespace(
    latent_dim=100,
    hidden_dim=256,
    num_neighbors=[15],
    num_epochs=20,
    num_hvgs=1000,
    pretrained=True,
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

- Cell embeddings: `scmo.whole.obsm["SCPRO-VI"]`
- Graphs: `scmo.whole.uns["rna_edges"]`, `scmo.whole.uns["prot_edges"]`, and `scmo.whole.obsm["cell_similarity"]`

These can be used for downstream clustering, trajectory inference, and visualization.

---

## 📃 Citation

If you use SCPRO-VI in your research, please cite:

```bibtex
@article{SCPROVI2024,
  title={Explainable Graph Learning for Multimodal Single-Cell Data Integration},
  author={Mehmet Burak Koca, Fatih Erdoğan Sevilgen},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.12.06.627151}
}
```

---

## 📮 Contact

For questions or collaboration, feel free to contact with me at `b.koca@gtu.edu.tr`.
