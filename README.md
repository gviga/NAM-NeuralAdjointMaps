# NAM-NeuralAdjointMaps
![graphical_abstract_1494](https://github.com/user-attachments/assets/f7ecd67d-4a1d-4aa2-8d5d-412534c86b8b)

Official Repository for "NAM: Neural Adjoint Maps for refining shape correspondences"  
Authors: Giulio Viganò, Maks Ovsjanikov, Simone Melzi.
`paper <https://dl.acm.org/doi/10.1145/3730943>`_
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
author = {Vigan\`{o}, Giulio and Ovsjanikov, Maks and Melzi, Simone},
title = {NAM: Neural Adjoint Maps for refining shape correspondences},
year = {2025},
issue_date = {August 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {44},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3730943},
doi = {10.1145/3730943},
abstract = {In this paper, we propose a novel approach to refine 3D shape correspondences by leveraging multi-layer perceptions within the framework of functional maps. Central to our contribution is the concept of Neural Adjoint Maps, a novel neural representation that generalizes the traditional solution of functional maps for estimating correspondence between manifolds. Fostering our neural representation, we propose an iterative algorithm explicitly designed to enhance the precision and robustness of shape correspondence across diverse modalities such as meshes and point clouds. By harnessing the expressive power of non-linear solutions, our method captures intricate geometric details and feature correspondences that conventional linear approaches often overlook. Extensive evaluations on standard benchmarks and challenging datasets demonstrate that our approach achieves state-of-the-art accuracy for both isometric and non-isometric meshes and for point clouds where traditional methods frequently struggle. Moreover, we show the versatility of our method in tasks such as signal and neural field transfer, highlighting its broad applicability to domains including computer graphics, medical imaging, and other fields demanding precise transfer of information among 3D shapes. Our work sets a new standard for shape correspondence refinement, offering robust tools across various applications.},
journal = {ACM Trans. Graph.},
month = jul,
articleno = {60},
numpages = {15},
keywords = {shape matching, point cloud, machine learning}
}



---

For questions or issues, please open an issue on GitHub.
