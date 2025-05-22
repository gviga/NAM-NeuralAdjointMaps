# NAM-NeuralAdjointMaps
![graphical_abstract_1494](https://github.com/user-attachments/assets/f7ecd67d-4a1d-4aa2-8d5d-412534c86b8b)

Official Repository for "NAM: Neural Adjoint Maps for refining shape correspondences"  
Authors: Giulio Viganò, Maks Ovsjanikov, Simone Melzi.

## Overview

In this Repo, we imlement and show Neural Adjoint Maps, a new functional representation of correspondences between shapes.

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

Go to notebooks to see how to simply implements nam, compared to standard ZoomOut.

Install dependencies with:

```sh
pip install pipreqs
pipreqs /path/to/your/project
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

## Citation

If you use this code, please cite:

```
@article{vigano2024nam,
  title={NAM: Neural Adjoint Maps for refining shape correspondences},
  author={Viganò, Giulio and Ovsjanikov, Maks and Melzi, Simone},
  year={2024}
}

---

For questions or issues, please open an issue on GitHub.
