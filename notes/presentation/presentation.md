# Presentation - April 22

# Sections

### Disease Associated Proteins

Let's motivate the discussion with a simple example: 

- Meckel Gruber Syndrome is a rare autosomal recessive disorder.  It is thought to be caused by dysfunction of cilia and flagella in human cells.
- Currently, there are twelve proteins known to be associated with Meckel Gruber Syndrome. Amongst these proteins, the three that seem to be most salient in manifestation of the disease are B9D1 B9D2 and MSK1 – these proteins form a protein complex that critical ciliogenesis. Disruption of one or more of these proteins – through genetic mutation – causes Meckel Gruber Syndrome. 



**Disease-Gene Association**

Human diseases can often be traced back to specific genes responsible for the manifestation of the disease.

We're going to call such a gene a **disease gene** and say that the disease and gene are associated. 

Specifically, a disease gene is one where an alteration in the gene’s code, expression or protein product underpins the disease. 

Note that I'm probably going to use the terms gene and protein interchangeably, because many genes act as instructions to make proteins we'll say that a protein is associated with a gene when its gene is associated and vice versa. 



**Disease Pathway**

A disease pathway is simply the full set of proteins associated with a disease.

**DisGeNet**

For our work we use DisGeNet, a huge database of  disease pathways. We use the 1,811diseases in DisGeNet with at least ten associated proteins. In total our dataset consists of some 75k associations. 

In many cases we can identify disease proteins through genome wide association studies.  In other cases, disease proteins can be uncovered through targetted studies that aim to understand the molecular mechanics behind a disease.



Slides:

- [x] Theoretical explanation
- [x] Example: Meckel-Gruber Syndrome

### Problem Formulation

Ideas: 

- Explain what disease protein prediction is 
- Why it is an important 
- Why it is challenging
- What datasets are typically used:
  - Genomic data
  - Ontology
  - Interactome

Notes:

So clearly this notion of disease-



Slides:

- [ ] problem formulation

### Human Interactome

Ideas:

- What isthe interactome?
- What is an interaction? What types of interactions are included?
- Where do our interactomes come from (STRING, bio-pathways etc.)
- Key network metrics

### Disease Pathways in the Interactome

Ideas:

- Projecting disease pathways onto the interactome
- Disease pathway example

Slides: 

- [ ] Disease pathways 
- [ ] Example: well-connected disease pathway example 1 

### Disease Module Hypothesis

Ideas:

- Theoretical idea: diseases show modular behavior in networks
- Stats to support this idea
- Another real example



### Existing DPP Methods 

Ideas: 

- Introduce idea of using interactome for disease protein prediction (consider idea of moving DPP to here)
- Review existing DPP methods
  - Random walks 
  - DIAMOND
  - direct-neighbor scoring
  - representation learning (node2vec, gcn)
- Existing methods do not perform that well

Slides:

- [ ] Overview of existing methods
- [ ] Method performance

### Disease Pathways are poorly well-connected

Ideas:

- Suggest that poor performance can be explained by poor modularity 
- Show examples of poor connection
- Show statistics on poor connection 
- Present some potential reasons for the poor connections

Slides:

- [ ] Hypothesis slide
- [ ] Example: poorly-connected 1
- [ ] Example: poorly-connected 2
- [ ] Example: poorly-connected 3
- [ ] Observation: stats slide
- [ ] Transition: what could explain the poor connections? 

### Mutual Neighborhoods

Ideas: 

- Are there more powerful ways of characterizing disease pathways in the interactome? 
- Perhaps looking at the common interactions could yield some insight
- Examples: poorly-connected -> common interactors
- Observation: disease proteins associated with the same disease tend to share a significant number of common interactors with other proteins associated with the disease when we normalize for degree
- Normalizing for degree of intermediate node is important

Slides:

- [ ] Question slide
- [ ] Example: common interactors 1
- [ ] Example: common interactors 2
- [ ] Example: common interactors 3
- [ ] Observation: stats



### Learned Common Interactors

Ideas:

- Can we leverage this new observation to design new methods of disease protein prediction? 
- Introduce idea that disease proteins are shared across diseases (TODO)
- Explain model theoretically
- Explain training 
- Explain justification 

Slides: 

- [ ] Question slide
- [ ] Model explanation theoretically
- [ ] Training explanation



### Results

Ideas:

- LCI outperforms existing methods 
  - Recall curve
  - Disease by disease comparison
  - Disease class comparison
- These results are consistent even when we split diseases against similarity

- Show examples on a few diseases 

Slides:

- [ ] Recall curve
- [ ] method comparison
- [ ] disease class comparison

### Enrichment Analysis

Ideas:

- How can we validate novel predictions that the model is making?
- Explain enrichment analysis procedure
- Compare enriched functions in disease proteins compared with enriched functions in predicted proteins

Slides:

- [ ] Enrichment analysis explanation
- [ ] Key examples 
- [ ] Analysis (TODO)

### Future work

Incorporating other forms of datahttps://science-sciencemag-org.stanford.idm.oclc.org/content/363/6433/1287



### Meeting with Marinka 

Start with a motivation of the problem: our high level goal is to discover disease protein – potentially move the how to discover disease genes slide to the 



Current paradigm: disease module hypothesis 

Specify 

Include empty slide with old paradigm: 

Include empty slide with New paradigm: 

Send slides to Jure next week 



On slide about current methods: explicitly state that they are based on the disease module hypothesis – potentially include a pop-up slide, as little text as possible



Is the disease module hypothesis valid – one slide with one word 



Weight intermediate proteins, we define a simple optimization task 



Don't add any additional content – what's their suffices. What's lacking are empty slides or pop-ups that have the takeaway messages

Make an outline of the talk: super simple 3 points 



Outline:

1) Current paradigm: disease module hypothesis. Is the disease module hypothesis valid? Disease module hypothesis is not a complete characterization. More nuance to the disease module hypothesis. 

2) Evaluating:

3) New Paradigm: Common Interactors 



Add summary and conclusions slide at the end

around 25 mins 



Be sure to explain the network. 

Explain the plots clearly "on the x-axis is" on the y-ax is blah 