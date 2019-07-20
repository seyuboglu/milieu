# Manuscript Outline

Outline, TODOs and notes for the disease pathway 

## Abstract



## Introduction



## Results

### Overview of approach.

> - [x] Draft

### Disease proteins tend to share common interactors.

> - [x] Draft

### Disease protein discovery.

> - [x] Draft

In the top 20 diseases that LCI performs better 

use other performance metrics, and check 

### Functional enrichment analysis. 

> - [x] Draft

### LCI weights in the context of network and protein properties. 

> - [x] Draft

## Discussion

##Methods

### Datasets and Preprocessing

#### PPI Networks.

> - [x] Draft

#### Protein-Disease Association Data.

> - [x] Draft

#### Disease Categorization Data

> - [x] Draft

#### Drug Target Data

> - [x] Draft



degree distribution of common interactors vs. 

### Disease Protein Discovery and Experimental Setup

#### Protein-wise Cross Validation.

> - [x] Draft

#### Disease-wise Cross Validation .

> - [x] Draft





### Learned Common Interactors (LCI)

#### The LCI Model. 

> - [x] Draft

#### Training the LCI Model.

> - [x] Draft





### Baseline Methods

#### Random Walks.

> - [x] Draft

#### Graph Neural Networks.

> - [x] Draft

#### Shallow Network Embeddings.

> - [x] Draft

#### DIAMOnD.

> - [x] Draft





### Network Analysis

#### Network Metrics

> - [x] Draft
> - [ ] Check what normalization strategy we actually used.

#### Statistical Tests

> - [x] Draft

#### Functional Enrichment Analysis

> - [x] Draft







##  Figures

Scaling in seaborn, set font-scale  to 2.0 or 1.5, relevant for the histograms for distributions 

Change figsize to 4, 2

### Figure 1

Increase number of bars

Chan



### Figure 2

Disease size 



### Figure 3

Learned weights, w_k and  

Increase number of bars 

Get the table, with disease to spearman correlation

move disease classes to supplementary 

Confirm all final numbers 



### Figure 4

Resize only intermediate nodes by weight

Pathway density, cut-ratio 

NUmber of disease proteins 





## Supplementary Material

### Notes

1) Performance on Two other PPI networks 

Todo:

- [x] STRING
- [x] Bio-Grid
- [x] Reformat to combine into one and match main figure
- [x] Write caption
- [x] Write note 
- [x] Reference

2) Performance on more challenging split

- [x] Reformat to match main figure
- [x] Write caption
- [x] Write note
- [x] Reference

2) Performance by disease class, create two rows 

- [x] Make figure
- [x] Upload figure
- [x] Write caption
- [x] Write note 
- [x] Reference

Questions:

Was moving methods section to this note a good idea?

3) Weights as a function of degree 

- [x] Make Figure
- [x] Write Caption
- [x] Reference

NO NOTE? 

4) Enrichment table

- [x] Make table
- [x] Write Caption
- [x] Reference 

NO NOTE? 

5) Embedding based approach

- [x] Create embeddings figure
- [x] Write caption
- [x] Note
- [x] Reference

Questions: Is Parameteric LCI a good name? 



### Figures

Supplementary Note 1:

Document outline for the supplementary information



for each suppplmentary note that I finish, throw in a sentencein the main paper referring to it 



can reference supplementary note from the caption of the figure, 



make sure to mention the two other networks i the main paper 



TODO:

- [ ] Change names in figures 



### Supplementary Data

Disease Protein Prediction Results

Network

Associations



## Code

- [ ] Next step 



Move LCI method to root of directory, and rename methods to baseline_methods 

One ipython notebook for how to use LCI 

README, look at decagon, graphsage for examples 

As simple as possible 

- [ ] Make repo public

- [ ] Add sentence paper for the code 

- [ ] name of the repo should be name of method 





# Title

Method name: 

Learned Shared Interactors

Learning Interactors Shared in Disease Pathways



Short list:

IMPALA (letter order incorrectr)

aLMIghty (it is what it is)

MILieu  (pronounce)







Notes: 

Milieu's prediction model is tranferable to new diseases:

- expand on the generalizability and emphasize 







Add panel C: 

Data robustness experiment - 

Goal: bar plot, x axis dropped PPI edges, y axis 





Do what is most relevant to meeting tomorrow:

1. Finish figure 3
2. Captions
3. Rename LCI
4. Make small changes, don't obsess about english 



discussed 

representation learning for patient data plus knowledge graph biomedical 

main project 

knowledge graph 



1) Selling it as a geneal network science principle 

2) Selling it as general biological network principle 

- Predicted GO labels for proteins (very standard, canonical straightforward, perhaps more interesting problem)
- Marinka is going to think about this for a day or two
- Predicting drug targets (already tons of work done)

Get GitHub ready for submissions 

Printed copy with notes is on marinka's desk, ask adrijan if you can't get in 

We'll talk on monday 