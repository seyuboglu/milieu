# Mutual Interactors: A graph-based machine learning model with applications in molecular phenotype prediction

*Sabri Eyuboglu, Marinka Zitnik and Jure Leskovec*

This repository includes our PyTorch implementation of *Mutual Interactors*, a machine learning algorithm for node set expansion in large networks. The algorithm is motivated by the structure of disease-associated proteins, drug targets and protein functions in molecular networks, and can be used to  predicrt molecular phenotypes *in silico*. For a detailed description of the algorithm, please see our [paper](TODO).  

We include software for easily reproducing *all* the experiments described in the paper. Each experiment has a designated class in the `milieu/experiment' module. We also provide implementations of several baseline network-based disease protein prediction methods, including DIAMOnD, Random Walks and Graph Neural Networks.


<p align="center">
<img src="https://github.com/seyuboglu/milieu/blob/master/data/images/model.png" width="800" align="center">
</p>

## Setup

Clone the repository

```bash
git clone https://github.com/seyuboglu/milieu.git
cd milieu
```

Create a virtual environment and activate it
```
python3 -m venv ./env
source env/bin/activate
```

Install package (`-e` for development mode)

```
pip install -e .
```

## Using *Mutual Interactors*
To get started checkout `mutual_interactors.ipynb` a Jupyter Notebook that will walk you
through the process of training a *Mutual Interactors* model.

It will also show you how to visualize the predictions of a trained *Mutual Interactors*
model. 

<p align="center">
<img src="https://github.com/seyuboglu/milieu/blob/master/data/images/notebook.png" width="800" align="center">
</p>



## Replicating Experiments

### disease-significance

### disease_subgraph

### dpp_evaluate

### dpp_predict

### go_enrichment

### lci_anlaysis

### protein_significance




## Directory

- *data* - all of the data for the project
  - *associations* - datasets of disease protein associations each stored in a *.csv* 
  - *disease_classes* - datasets 
  - *drug* - drug target datasets
  - embeddings - assorted protein embeddings
  - networks - protein-protein interaction networks
  - protein - assorted protein data
- *experiments* - each experiment we ran has a directory here with parameters, results, figures and notebooks 
- *milieu* - all of the source code 
  - `data` - modules including utility classes and functions for preprocessing and loading data
  - `experiments` - modules including experiment harness classes
  - `figures` - modules implementing figure generating classes and functions
  - `methods` - modules including implementations of disease protein prediction methods
- *notebooks* - assorted notebooks for exploring data and experiments

