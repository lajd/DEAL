# DEAL

### Overview

This is the implementation of "[Inductive Link Prediction for Nodes Having Only Attribute Information
](https://www.ijcai.org/Proceedings/2020/168)" published at IJCAI 2020.

This work has been adapted from the original implementation by `working-yuhao`, found at https://github.com/working-yuhao/DEAL. 

In particular, this work is a refactoring of the original implementation to be more minimal and composable. This
refactoring work focuses solely on the inductive link prediction task.

### Major changes
- Datasets are not the same as in the original paper
    - Graphs are undirected (twice the number of edges)
    - Different train/validation/test splits
    - Different method for negative sampling, negative sampling fraction

### Requirements
- Linux
- Nvidia GPU
- Cuda version 11.0
- Anaconda

### Installation
The installation below uses anaconda.

```shell
#!/bin/bash

CONDA_ENV_NAME=DEAL

#conda clean --all -y
conda env list | grep ${CONDA_ENV_NAME}
if [ $? -eq 0 ]; then
    echo "DEAL environment already exists; skipping creation"
else
    echo "Creating ${CONDA_ENV_NAME} environment"
    conda create -n ${CONDA_ENV_NAME} python=3.9 -y
fi

conda activate ${CONDA_ENV_NAME}

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c nvidia -y
conda install -q -y numpy pyyaml scipy ipython mkl mkl-include conda-build
conda install pyg -c pyg -c conda-forge -y

pip install -e .
```

### Original Paper DEAL Inductive Results

** Note: The below table is not a fair comparison (due to the changes to the dataset mentioned above)

| Implementation | Cora        | Cora-full      | CiteSeer       |   CS          |  PubMed       | Computers     | Photos   |
| -----------  | -----------   | -----------    | -----------    | -----------   | -----------   | -----------   | ----------- |
|              | AUC/AP        | AUC/AP         |  AUC/AP        | AUC/AP        | AUC/AP        | AUC/AP        | AUC/AP      |
| Original     | 0.864/0.804   | N/A            | 0.937/0.907    | 0.977/0.959   | 0.966/0.931   | 0.953/0.899   | 0.965/0.922  |
| Refactored   | 0.8757/0.8753 | 0.9124/0.9095  | 0.9114/0.9160  | 0.9485/0.9412 | 0.9226/0.9140 | 0.8494/0.8373 | 0.8920/0.8804  |



### Reproduction

#### Cora (small)

```shell
make train-cora-small
```

```yaml
 Total Load data time: 0.02 s
 Total Train/val time: 6.39 s
 Test time: 0.00 s
 Total time: 6.40 s
 ROC-AUC: 0.8757 
 AP: 0.8753
```


### Cora (full)

```shell
make train-cora-full
```

```yaml
 Total Load data time: 0.57 s
 Total Train/val time: 33.38 s
 Test time: 0.00 s
 Total time: 33.95 s
 ROC-AUC: 0.9124
 AP: 0.9095
```
#### CiteSeer

```shell
make train-citeseer
```

```yaml
 Total Load data time: 0.04 s
 Total Train/val time: 9.83 s
 Test time: 0.00 s
 Total time: 9.87 s
 ROC-AUC: 0.9114 
 AP: 0.9160
```


#### Co-authorship (CS)

```shell
make train-coauthor-cs
```

```yaml
 Total Load data time: 0.48 s
 Total Train/val time: 41.70 s
 Test time: 0.00 s
 Total time: 42.18 s
 ROC-AUC: 0.9485 
 AP: 0.9412
```


#### Pubmed

```shell
make train-pubmed
```

```yaml
 Total Load data time: 120.92 s
 Total Train/val time: 20.63 s
 Test time: 0.00 s
 Total time: 141.55 s
 ROC-AUC: 0.9226
 AP: 0.9140
```

#### Amazon Computers
```shell
make train-computers
```

```yaml
 Total Load data time: 0.28 s
 Total Train/val time: 69.72 s
 Test time: 0.00 s
 Total time: 70.00 s
 ROC-AUC: 0.8494 
 AP: 0.8373
```


#### Amazon Photos
```shell
make train-photos
```

```yaml
 Total Load data time: 0.12 s
 Total Train/val time: 30.83 s
 Test time: 0.00 s
 Total time: 30.95 s
 ROC-AUC: 0.8920
 AP: 0.8804
```

### Citing the original work

Please cite the original IJCAI 2020 paper:

```
@inproceedings{ijcai2020-168,
  title     = {Inductive Link Prediction for Nodes Having Only Attribute Information},
  author    = {Hao, Yu and Cao, Xin and Fang, Yixiang and Xie, Xike and Wang, Sibo},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {1209--1215},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/168},
  url       = {https://doi.org/10.24963/ijcai.2020/168},
}
```
