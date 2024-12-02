# MSF-CPMP

# introduction

MSF-CPMP is a novel multi-source feature fusion model for the permeability of cyclic peptide membranes   
# datasets

data_deep_learning   
* The results of ten fold cross validation of cyclic peptide data are used for training deep neural networks
data_machine_learning

* The results of ten fold cross validation of cyclic peptide data are used for training traditional machine learning networks

# datasets_process

* CycPeptMPDB_Peptide_PAMPA.csv-6491 cyclic peptide data, including various characteristics and permeability sizes of cyclic peptides. The download address is http://cycpeptmpdb.com/peptides/type_PAMPA/
* data_deeplearning_process.py：Obtain the ten fold cross validation data required for deep learning models
* data_machinelearning_process.py：Obtain the ten fold cross validation data required for machine learning models   
# Environmental requirements

This code has been tested in Python 3.8. Quickly set up using environment. yaml
```bash
conda env create -f environment.yaml
```
```bash
conda activate MSF-CPMP
conda install -c rmg descriptastorus
pip install dgllife==0.3.2
```
# src

deep_learning

Detailed information on all deep learning models, including data preprocessing, model construction, training, validation, testing, and result preservation

machine_learning

Detailed information on all machine learning models, including data preprocessing, model construction, training, validation, testing, and result preservation

##Distribution Description

1.Install dependencies, including torch, sklearn, rdkit, and dgllife 

2.Run data_deeplearning process. py and data_machinelearning process. py to obtain the data used for training

3.Run data_pretrofit. py to obtain preprocessed data, which can be directly used for network training, validation, and testing

4.Run model_comcat-py for training, validation, prediction, and result saving
