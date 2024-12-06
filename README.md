This repo is prepared for deposit of SCPRO-VI implementation. You can run the method using the following code block:

```python
from src import scPROMO, Metrics, Utils, Models
importlib.reload(scPROMO)
importlib.reload(Metrics)
importlib.reload(Utils)
importlib.reload(Models)
```

```python
data_path = "/content/drive/MyDrive/SCPRO/Data/GSE164378_Sub_"
exp_data = scPROMO.load_data(data_path, sub_sample = False, load = True)
```

```python
args = scPROMO.Namespace(
hidden_dim = 256,
latent_dim = 128,
num_hvgs = 200,
use_embeddings = True,
pretrained = True,
num_neighbors = [20, 10],
num_epochs = 500) # 100
exp_data.prot
scPROMO.VI.scpro_vi(exp_data, args)
```
