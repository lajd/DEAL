# DEAL

This is the implementation of "[Inductive Link Prediction for Nodes Having Only Attribute Information
](https://www.ijcai.org/Proceedings/2020/168)" published at IJCAI 2020.


### Requirements
- Linux
- Cuda version 11.0

### Installation

```shell
#!/bin/bash

CONDA_ENV_NAME=DEAL

#conda clean --all -y
source ./config.sh
conda env list | grep ${CONDA_ENV_NAME}
if [ $? -eq 0 ]; then
    echo "DEAL environment already exists; skipping creation"
else
    echo "Creating ${CONDA_ENV_NAME} environment"
    conda create -n ${CONDA_ENV_NAME} python=3.9 -y
fi

conda activate ${CONDA_ENV_NAME}

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y
conda install -q -y numpy pyyaml scipy ipython mkl mkl-include conda-build
conda install pyg -c pyg -c conda-forge -y

pip install -e .

# Download CiteCeer `dists`
gdown https://drive.google.com/uc?id=1LqSKvzsDThMzqZWBQqP9b-k3KeEtaUto
mv dists-1.dat data/CiteSeer/
```
### Train DEAL on the CiteSeer dataset
`make train`

### Datasets
More datasets can be found at https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html.

### Cite

Please cite our IJCAI 2020 paper:

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

