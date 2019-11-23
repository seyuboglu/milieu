# Mutual Interactors: A graph-based machine learning model with applications in molecular phenotype prediction

*Sabri Eyuboglu, Marinka Zitnik and Jure Leskovec*

This repository includes our PyTorch implementation of *Mutual Interactors*, a machine learning algorithm for node set expansion in large networks. The algorithm is motivated by the structure of disease-associated proteins, drug targets and protein functions in molecular networks, and can be used to  predict molecular phenotypes *in silico*. For a detailed description of the algorithm, please see our [paper](TODO).  

We include software that makes it easy to **reproduce** *all* the experiments described in the paper. Each experiment has a designated class in the `milieu.paper.experiment` module. We also provide implementations of several **baseline** network-based disease protein prediction methods, including DIAMOnD, Random Walks and GCN.


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

Once you've trained a model you can use it to discover new nodes that might belong to a set of nodes you're interested in. For example, below we use it to predict what other nodes might also be associated tracheomalacia. You can visualize the model's predictions in the context of known associated nodes (red), predicted nodes (orange), and the mutual interactors between them (blue). In the screenshot below, we show a visualization for tracheomalacia.

<p align="center">
<img src="https://github.com/seyuboglu/milieu/blob/master/data/images/notebook.png" width="800" align="center">
</p>



## Reproducing Experiments
We provide the code and data to reproduce *all* of the experiments described in our manuscript. 

### `Experiment` Classes
Each experiment has a designated class in the `milieu.paper.experiments` module. For example, the `EvaluateMethod` class can be used to evaluate a method on a node-set expansion task (e.g. disease protein prediction on the human PPI network).  All experiment classes subclass `milieu.paper.experiments.experiment.Experiment`. 

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

Below we walk through a selection of the `Experiment` classes we've implemented for our study. 

### 1) `EvaluateMethod`
Uses node-wise cross-validation to evaluate a method's capacity to accurately expand node sets in a network. 
<p align="center">
<img src="https://github.com/seyuboglu/milieu/blob/master/data/images/recall_curve.png" width="300" align="center">
</p>

*Required parameters*: `network`, `associations_path`, `n_folds`, `method_class`, `method_params`

*Experiment directories*: the directories containing the parameters we used when running experiments for our study
- `experiments/1_dpp_evaluate/**`: evaluate method performance on disease protein prediction (*Used for*: Fig. 2a, Fig. 2c-e)
- `experiments/2_dti_evaluate/**`: evaluate method performance on drug-target interaction prediction (*Used for*: Fig. 3b, Fig. 3e)
- `experiments/3_go_evaluate/**`: evaluate method performance on gene ontology molecular function and biological process prediction (*Used for*: Extended Fig. 1b, Extended Fig. 2b) 

*Example*: 
```
run experiments/3_go_evaluate/species_9606/function/random_walk
```


### 2) `NodeSignificance`
Evaluate the statistical significance of a network structure in a dataset of node sets (e.g. disease pathways). Uses permutation tests to compute $p$-values (See **Evaluating the statistical significance of network structures** in Methods.)
<p align="center">
<img src="https://github.com/seyuboglu/milieu/blob/master/data/images/node_significance.png" width="300" align="center">
</p>

*Required parameters*: `network`, `associations_path`, `metric_fns`, `network_matrices`, `n_random_nodes`, `min_bucket_len`

*Experiment directories*: the directories containing the parameters we used when running experiments for our study
- `experiments/4_protein_significance/disease`: evaluate statistical significance of mutual interactor scores and direct interactor scores for proteins in the same disease pathway (*Used for*: Fig. 1d) 
- `experiments/4_protein_significance/drug`: evaluate statistical significance of mutual interactor scores and direct interactor scores for proteins targetted by the same drug (*Used for*: Fig. 3a) 
- `experiments/4_protein_significance/go`: evaluate statistical significance of mutual interactor scores and direct interactor scores for proteins with the same function (*Used for*: Extended Fig. 1a, Extended Fig. 2a) 

*Example*: 
```
run experiments/4_protein_significance/disease
```

### 3) `Network Robustness`
Evaluate a method's robustness to incomplete networks. 
<p align="center">
<img src="https://github.com/seyuboglu/milieu/blob/master/data/images/robustness.png" width="300" align="center">
</p>

*Required parameters:* `configs`, `experiment_class`, `experiment_params`

*Experiment directories*: the directories containing the parameters we used when running experiments for our study
- `experiments/5_milieu_robustness`: evaluate the robustness of *Mutual Interactors* to incomplete networks. (*Used for*: Fig. 2b)

*Example*:
```
run experiments/5_milieu_robustness
```

### Others
`milieu_analysis.DrugTargets`: (*Used for*: Fig 2f), Note: cannot run from command line, see `experiments/6_milieu_analysis/milieu_analysis.ipynb`

`GoEnrichment`: (*Used for*: Supplementary Fig. 11)

`SetSignificance`: (*Used for*: Fig. 1e-f)


