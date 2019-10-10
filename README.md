# Network-based Disease Protein Prediction

*Sabri Eyuboglu, Marinka Zitnik and Jure Leskovec*

This repository includes our PyTorch implementation of *Milieu*, a disease protein discovery algorithm that uncovers novel disease-protein associations *in silico* by leveraging the mutal interators between proteins already known to be associated with the disease. For a detailed description of the algorithm, please see our [paper](TODO).  

We also include software for replicating the experiments referenced in the paper. Each experiment has a designated class in the `milieu/experiment' module. We also provide implementations of several baseline network-based disease protein prediction methods, including DIAMOnD, Random Walks and Graph Neural Networks.


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

## Using *Milieu*
To 


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

## Experiments 

### disease-significance

### disease_subgraph

### dpp_evaluate

### dpp_predict

### go_enrichment

### lci_anlaysis

### protein_significance
