# AM220

# Graph Attention for Heterogeneous Graphs with Positional Encoding

This repository contains implementations of various Graph Neural Network (GNN) architectures enhanced with positional encoding techniques derived from the SAN paper. The projects include RGAT, Graph Transformer Network, and pyHGT, each tailored for heterogeneous graph data and enhanced with positional encoding for improved performance on tasks like node classification and link prediction.

## Overview

The repository is structured into four main directories:
- `Graph Transformer Networks`: Implementation of the Graph Transformer Networks with enhancements for handling heterogeneous graphs using positional encoding.
- `pyHGT`: The Heterogeneous Graph Transformer (HGT) architecture implemented with positional encoding.
- `RGAT`: Relational Graph Attention Networks with support for positional encoding to improve the attention mechanism across different node and relation types.
- `SAN`: Source implementation of Spectral Attention Networks used as a reference for positional encoding methods.

## Key Features

- **Positional Encoding**: Enhancements using the SAN paper's method are applied to Graph Transformer Networks, pyHGT, and RGAT to improve their efficacy on heterogeneous graphs.
- **Heterogeneous Graphs**: Specialized attention mechanisms to handle different types of nodes and edges effectively.
- **Multiple Datasets**: Supports various datasets including ACM, IMDB, Tox21, and AIFB, ideal for tasks like node classification and link prediction.

## Getting Started

To run any of the projects with the implemented positional encoding, follow the installation instructions provided in each project's respective directory.

### Prerequisites

- Python 3.x
- PyTorch
- DGL (For Graph Transformer Network and RGAT)
- PyTorch Geometric (For pyHGT)
- Additional requirements are listed in the `requirements.txt` files within each project directory.

### Installation

Navigate to the project directory and install dependencies:

```bash
cd <project-directory>
pip install -r requirements.txt
```

### Running the Models

Each project has specific scripts to train and evaluate the models. For example, to run the Graph Transformer Network on the ACM dataset:

```bash
cd Graph_Transformer_Networks
python main.py --dataset ACM --model GTN --num_layers 1 --epoch 50 --lr 0.02 --num_channels 2
```

### Citation
If you use the enhancements or the positional encoding techniques in your research, please consider citing the relevant papers listed in each project's README.md.

### License
This project is licensed under the MIT License.