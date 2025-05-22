# NAM-NeuralAdjointMaps

Official Repository for "NAM: Neural Adjoint Maps for refining shape correspondences"  
Authors: Giulio Viganò, Maks Ovsjanikov, Simone Melzi

## Overview

This repository provides an implementation of Neural Adjoint Maps (NAM) for refining shape correspondences using deep learning. The codebase includes neural network models, optimization routines, and utilities for spectral shape matching.

## Directory Structure

- `model/`  
  Contains core model definitions, including the neural adjoint map ([`model/neural_adjoint_map.py`](model/neural_adjoint_map.py)), loss functions, and optimizers.
- `methods/`  
  Reference and baseline methods for shape matching.
- `nam_utils/`  
  Utility functions, e.g., Sinkhorn algorithm implementations.
- `notebooks/`  
  Example Jupyter notebooks demonstrating usage and experiments.
- `README.md`  
  This file.

## Getting Started

### Requirements

- Python 3.8+
- PyTorch
- numpy
- pyyaml

Install dependencies with:

```sh
pip install torch numpy pyyaml
```

### Usage

You can start by running the example notebooks in [`notebooks/`](notebooks/):

```sh
jupyter notebook notebooks/notebook.ipynb
```

Or use the [`Neural_Adjoint_Map`](model/neural_adjoint_map.py) model directly in your code:

```python
from model.neural_adjoint_map import Neural_Adjoint_Map

model = Neural_Adjoint_Map(input_dim=128)
output = model(input_tensor)
```

## Citation

If you use this code, please cite:

```
@article{vigano2024nam,
  title={NAM: Neural Adjoint Maps for refining shape correspondences},
  author={Viganò, Giulio and Ovsjanikov, Maks and Melzi, Simone},
  year={2024}
}
```

## License

[Specify your license here, e.g., MIT, GPL, etc.]

---

For questions or issues, please open an issue on GitHub.