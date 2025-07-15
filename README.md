# NAM-NeuralAdjointMaps
![graphical_abstract_1494](https://github.com/user-attachments/assets/f7ecd67d-4a1d-4aa2-8d5d-412534c86b8b)

Official Repository for "NAM: Neural Adjoint Maps for refining shape correspondences"  
Authors: Giulio Viganò, Maks Ovsjanikov, Simone Melzi.

## Overview

In this Repo, we implement and show Neural Adjoint Maps, a new functional representation of correspondences between shapes. This Repository does not replicate the results of the papers; stay tuned for additional material.

You can find Nam implemented also in Geomfum package at https://github.com/DiG-AIR/geomfum.

## Directory Structure
```
NAM-NeuralAdjointMaps/
├── model/
│   ├── __init__.py
│   ├── neural_adjoint_map.py
│   ├── nueral_zoomout.py
├── notebooks/
│   ├── fmap_vs_NAM.ipynb
│   ├── zoomout_vs_NZO.py  
├── data/
│   └── (sample data files)
├── README.md
├── requirements.txt
└── LICENSE
```

## Getting Started

Go to notebooks to see how to implement nam, compared to standard fmaps approaches.

Install dependencies with:

```sh
pip install -r requirements.txt
```


### Usage

You can start by running the example notebooks in [`notebooks/`](notebooks/):

```sh
jupyter notebook notebooks/fmap_vs_NAM.ipynb
```

Or use the NAM model and losses directly in your code:

```python
from model.neural_adjoint_map import NeuralAdjointMap


emb1 = torch.tensor(eigvecs1).to(torch.float32).cuda() #embedding1
emb2 = torch.tensor(eigvecs2).to(torch.float32).cuda() #embedding2

#optimize NAM from p2p
nam = NeuralAdjointMap(emb1, emb2)
nam.optimize_from_p2p(p2p_gt)


#compute p2p from nam
emb2_nn = nam(emb2)
knn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(eigvecs1)
distances, indices = knn.kneighbors(emb2_nn.detach().cpu().numpy())

p2p = indices.flatten()
```

### Note
This repo provides a simple implementation of NAM and Neural ZoomOut, it does not exactly replicate results from the original paper. In particular:
- The nearest search here is computed in CPU via nearest neighbor, while in the paper, we used torch_cluster.nearest.
- The zoomOut version used here is taken from the geomfum library (https://github.com/luisfpereira/geomfum/tree/main/geomfum), and it is not  GPU-accelerated.

For further issues, please contact the authors.



## Citation

If you use this code, please cite:

```

@article{vigano2025nam,
title = {NAM: Neural Adjoint Maps for refining shape correspondences},
journal = {Transactions On Graphics},
volume = {122},
pages = {103985},
year = {2024},
issn = {0097-8493},
doi = {https://doi.org/10.1145/3730943},
url = {https://www.lix.polytechnique.fr/~maks/papers/NAM_SIGGRAPH2025.pdf},
author = {Viganò, Giulio, Ovsjanikov, Maks and Melzi, Simone},
keywords = {Shape correspondence, Functional maps, Point clouds, Machine learning}}

---

For questions or issues, please open an issue on GitHub.
