# Mutual Interactors: A graph-based machine learning model with applications in molecular phenotype prediction

*Sabri Eyuboglu, Marinka Zitnik and Jure Leskovec*

This repository includes our PyTorch implementation of *Mutual Interactors*, a machine learning algorithm for node set expansion in large networks. The algorithm is motivated by the structure of disease-associated proteins, drug targets and protein functions in molecular networks, and can be used to  predict molecular phenotypes *in silico*. For a detailed description of the algorithm, please see our [paper](TODO).  

We include software for easily reproducing *all* the experiments described in the paper. Each experiment has a designated class in the `milieu/experiment` module. We also provide implementations of several baseline network-based disease protein prediction methods, including DIAMOnD, Random Walks and Graph Neural Networks.


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
To get started checkout `notebooks/mutual_interactors.ipynb` a Jupyter Notebook that will walk you
through the process of training a *Mutual Interactors* model.

It will also show you how to visualize the predictions of a trained *Mutual Interactors*
model. 

<p align="center">
<img src="https://github.com/seyuboglu/milieu/blob/master/data/images/notebook.png" width="800" align="center">
</p>



## Reproducing Experiments
We provide the code and data to reproduce *all* of the experiments described in our manuscript. 

### `Experiment` Classes
Each experiment has a designated class in the `milieu.paper.experiments` module. For example, the `EvaluateMethod` class can be used to evaluate a method on any node-set expansion task (e.g. disease protein prediction on the human PPI network).  All experiment classes subclass `milieu.paper.experiments.experiment.Experiment`. 

To run an experiment, we first construct an instance of an experiment class (such as `EvaluateMethod`). The constructor accepts an experiment directory (`dir`) and a parameter dictionary (`params`). The **experiment directory** is a directory where the experiment parameters, results, and logs are stored. The **experiment parameters** should be represented by a nested dictionary that specify which experiment class to use (e.g.  `"process": "evaluate_method"`) and the parameters for that experiment class (e.g. `"process_params": {...}`). Below we've included the parameters we used to evaluate the performance of Random Walks on the task of protein function prediction with the human PPI Network:
```
{
    "process": "evaluate_method",
    "process_params": {
        "n_processes": 20,
        "ppi_network": "data/networks/species_9606/bio-pathways/network.txt",
        "associations_path": "data/associations/gene_ontology/species_9606/go_function/associations.csv",
        
        "n_folds": 10,
        
        "method_class": "RandomWalk",
        "method_params": { 
            "alpha": 0.25
        }
    }
}
```
Once we've created an experiment object, we can run the experiment using the objects `run()` method. 

### Running from command line
We can also easily run any experiment from the command line with the `run` command. The `run` command accepts one argument: an experiment directory containing a params file `params.json`. This JSON file should follow the structure shown above. 
To run a new experiment: 
1) Create a directory with an descriptive name:
```
mkdir new_experiment
```
2) Add a params file and fill it in with the desired params.
```
vi new_experiment/params.json
```
3) Run it!
```
run new_experiment
```

This repository includes the experiment directories for all of the experiments we ran in our study. For example, the parameters we used to evaluate the performance of Random Walks on the task of protein function prediction with the human PPI Network can be found at `experiments/3_go_evaluate/species_9606/function/random_walk/params.json`. The results from this experiment can be found at `experiments/3_go_evaluate/species_9606/function/random_walk/metrics.csv`. 

We can easily re-run any of these experiments from the command line:
```
run experiments/3_go_evaluate/species_9606/function/random_walk
```

Below we walk through the various `Experiment` classes we've implemented for our study. 


### 1) `EvaluateMethod`
Provides an experimental harness for evaluating a methods 

### 2) `NodeSignificance`

### 3) `SetSignificance` 

### 4) `GOEnrichment`

### 5) Network Robustness 

### 6) MilieuAnalysis



## Directory
S

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

