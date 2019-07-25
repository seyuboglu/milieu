# Network-based Disease Protein Prediction

*Sabri Eyuboglu, Marinka Zitnik and Jure Leskovec*

Implementations of several network-based disease protein prediction methods including Learned Common Interactors (LCI). Includes harnesses for disease protein prediction experiments as well as Jupyter notebooks detailing the experiments described in our paper. 



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

### Jupyter Notebooks





#### Directory

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
