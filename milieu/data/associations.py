import os
import csv

import numpy as np 
import pandas as pd 
import networkx as nx
from goatools.obo_parser import GODag

from milieu.util.util import load_mapping

class NodeSet:
    """
    Abstract class for representing a set of nodes in a network
    """
    def __init__(self, id, name, split="none"):
        self.id = id
        self.name = name
        self.split = split
    
    def to_node_array(self, network):
        raise NotImplementedError

class ProteinSet(NodeSet): 
    """
    Represents a set of proteins associated with an entity. 
    """
    def __init__(self, id, name, entrez_ids, split="none"):
        """ Initialize a disease. 
        Args:
            id (string) 
            name (string)
            proteins (list of ints) entrez ids
            valdiation_proteins (list of ints)
        """
        super().__init__(id=id, name=name, split=split)
        self.proteins = entrez_ids

        self.doids = []
        self.parents = []
        self.class_doid = None 
        self.class_name = None
    
    def to_node_array(self, network):
        """ Translates the diseases protein list to an array of node ids using
        the protein_to_node dictionary.
        Args: 
            protein_to_node (dictionary)
        """
        return network.get_nodes(self.proteins)
    
    def to_names(self, entrez_to_name):
        """ Translates the disease protein list from entrez ids to gene names
        Args:
            entrez_to_name  (dict)  dictionary mapping entrez_ids to name
        """
        return [entrez_to_name[entrez_id] for entrez_id in self.proteins]
    
    def __len__(self):
        return len(self.proteins)


def load_node_sets(file_path, 
                   id_subset=[],
                   exclude_splits=[],
                   gene_names_path=None):
    return load_diseases(file_path, id_subset, exclude_splits, gene_names_path)


def load_diseases(associations_path, 
                  diseases_subset=[],
                  exclude_splits=[],
                  gene_names_path=None): 
    """ Load a set of disease-protein associations
    Args:
        associations_path (string)
        diseases_subset (set) 
    Returns:
        diseases (dict)
    """
    diseases = {}
    total = 0
    with open(associations_path) as associations_file:
        reader = csv.DictReader(associations_file)

        for row in reader:
            disease_id = row["disease_id"]
            if(diseases_subset and disease_id not in diseases_subset):
                continue  

            split = row.get("splits", None)
            if split in exclude_splits:
                continue

            disease_name = row["disease_name"]

            entrez_ids = set([int(a.strip()) 
                                    for a in row["gene_entrez_ids"].split(",")])

            total += len(entrez_ids)
            diseases[disease_id] = ProteinSet(disease_id, disease_name, 
                                              entrez_ids, split)

    return diseases 


def build_disease_matrix(diseases_dict, network, exclude_splits=[]):
    """
    Builds a matrix with one row for each disease in diseases_dict. Each row is a binary
    vector encoding the diseases associations. Protein to index mapping is the same as 
    that in the network.
    args:
        diseases_dict   (dict)  dictionary of diseases
        network     (PPINetwork) 
        exclude_splites     (list)
    """
    split_diseases = [disease for disease in diseases_dict.values() 
                      if disease.split not in exclude_splits]
    m = len(split_diseases)
    n = len(network)
    diseases = np.zeros((m, n), dtype=int)
    idx_to_disease = []
    for i, disease in enumerate(split_diseases):
        disease_nodes = disease.to_node_array(network)
        diseases[i, disease_nodes] = 1
        idx_to_disease.append(disease)
    return diseases, idx_to_disease


def write_associations(diseases, associations_path, threshold=10): 
    """ Write a set of disease-protein associations to a csv
    Args:
        diseases
        assoications_path (string)
        diseases_subset (set) 
    """
    disease_list = [{"Disease ID": disease.id,
                     "Disease Name": disease.name,
                     "Associated Gene IDs": ",".join(map(str, disease.proteins))} 
                    for _, disease in diseases.items() if len(disease.proteins) >= threshold] 

    with open(associations_path, 'w') as associations_file:
        writer = csv.DictWriter(associations_file, fieldnames=["Disease ID", 
                                                               "Disease Name", 
                                                               "Associated Gene IDs"])
        writer.writeheader()
        for disease in disease_list:
            writer.writerow(disease)


def is_disease_id(str):
    """ Returns bool indicating whether or not the passed in string is 
    a valid disease id. 
    Args: 
        str (string)
    """
    return len(str) == 8 and str[0] == 'C' and str[1:].isdigit()


def load_doid(diseases,
              hdo_path='data/raw/disease_ontology.obo'):
    """
    Adds a doids attribute to disease objects.
    Allows for mapping between mesh ID (e.g. C003548), like those
    used in the dassociation files, and DOID (e.g. 0001816). 
    args:
        hdo_path    (string)
        diseases    (dict)  meshID to disease object 
    """
    with open(hdo_path) as hdo_file:
        for line in hdo_file:
            if line.startswith('id:'):
                doid = line.strip().split('id:')[1].strip()
            elif line.startswith('xref: UMLS_CUI:'):
                mesh_id = line.strip().split('xref: UMLS_CUI:')[1].strip()
                if mesh_id in diseases:
                    diseases[mesh_id].doids.append(doid)


def load_disease_classes(diseases, 
                         hdo_path='data/raw/disease_ontology.obo',
                         level=2, 
                         min_size=10):
    """
    Adds a classes attribute to disease objects.
    """
    obo = GODag(hdo_path)
    load_doid(diseases, hdo_path)

    class_doid_to_diseases = {}
    num_classified = 0
    for disease in diseases.values():
        if not disease.doids: 
            continue
        doid = disease.doids[0]
        for parent in obo[doid].get_all_parents():
            if obo[parent].level == level:
                disease.class_name = obo[parent].name.replace("disease of ", "").replace("disease", "")
                class_doid_to_diseases.setdefault(parent, set()).add(disease.id)
        num_classified += 1

    for class_doid, class_diseases in class_doid_to_diseases.items():
        if len(class_diseases) < min_size:
            for disease_id in class_diseases:
                disease = diseases[disease_id]
                disease.class_name = None
                num_classified -= 1
    
    print("Classified {:.2f}% ({}/{}) of diseases".format(
          100.0 * num_classified / len(diseases),
          num_classified,
          len(diseases)))


def output_diseases(diseases, output_path):
    """
    Output disease objects to csv. 
    args:
        diseases    (dict)
        output_path (string)
    """
    df = pd.DataFrame([{"name": disease.name,
                        "class": "" if  disease.class_name is None 
                                 else disease.class_name,
                        "size": len(disease.proteins)} 
                       for disease in diseases.values()],
                      index=[disease.id for disease in diseases.values()],
                      columns=["name", "class", "size"])
    df.index.name = "id"
    df.to_csv(output_path, index=False)   


def build_disease_matrix(id_to_disease, protein_to_idx):
    """
    """
    n = len(protein_to_idx)
    m = len(id_to_disease)
    diseases = np.zeros((m, n), dtype=int)
    index_to_disease = []
    for i, disease in enumerate(id_to_disease.values()):
        disease_nodes = disease.to_node_array(protein_to_idx)
        diseases[i, disease_nodes] = 1
        index_to_disease.append(disease)
    return diseases, index_to_disease 


def remove_duplicate_diseases(diseases_path, network_path, threshold=0.9):
    """
    """
    print("Loading diseases and network...")
    id_to_disease = load_diseases(diseases_path)
    _, _, protein_to_node = load_network(network_path)

    print("Building disease matrix...")
    disease_matrix, idx_to_disease = build_disease_matrix(id_to_disease, protein_to_node)

    # compute common associations
    common_associations = np.matmul(disease_matrix, disease_matrix.T)
    np.fill_diagonal(common_associations, 0)

    # compute jaccard similarity 
    disease_sizes = np.sum(disease_matrix, axis=1, keepdims=True)
    intersection_size = common_associations
    union_size = np.add(disease_sizes, disease_sizes.T)
    jaccard_sim = 1.0*common_associations / (union_size - common_associations)

    dupicates = np.where(jaccard_sim >= threshold)

    to_remove = set()
    df = pd.read_csv(diseases_path, index_col="Disease ID")
    for a, b in zip(*dupicates):
        if a in to_remove or b in to_remove:
            continue
        idx_remove = a if idx_to_disease[a].size < idx_to_disease[a].size else b
        df['splits'][idx_to_disease[idx_remove].id] = 'none'
        to_remove.add(idx_remove)
        print("Found duplicate {}-{}".format(idx_to_disease[a].name, 
                                             idx_to_disease[b].name))
        print("Removing {}".format(idx_to_disease[idx_remove].name))
    print("Removed {} duplicates.".format(len(to_remove)))
    df.to_csv(diseases_path[:-4] + "_nodup.csv", index=True)        


def split_diseases_random(split_fractions, path):
    """ Splits a set of disease assocation into data sets i.e. train, test, and dev sets 
    Args: 
        split_fractions (dictionary) dictionary mapping split name to fraction. fractions 
                                     should sum to 1.0
        path (string) 
    """
    df = pd.read_csv(path)

    # randomly shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    num_diseases = len(df)
    splits = np.empty(num_diseases, dtype=object)
    curr_start = 0
    for name, fraction in split_fractions.items():
        curr_end = curr_start + int(num_diseases * fraction)
        splits[curr_start : curr_end] = name
        curr_start = curr_end
    
    df['splits'] = splits
    df.to_csv(path, index=False)


def split_diseases_cc(split_sizes, disease_path, network_path, threshold=0.3):
    """
    """
    ppi_networkx, ppi_network_adj, protein_to_node = load_network(network_path)
    diseases_dict = load_diseases(disease_path)
    diseases_dict = {did : disease for did, disease in diseases_dict.items() 
                     if disease.split != "none"}
    m = len(diseases_dict)
    n = ppi_network_adj.shape[0]

    # build disease matrix
    print("Building disease matrix...")
    diseases = np.zeros((m, n), dtype=int)
    index_to_disease = []
    for i, disease in enumerate(diseases_dict.values()):
        disease_nodes = disease.to_node_array(protein_to_node)
        diseases[i, disease_nodes] = 1
        index_to_disease.append(disease)

    # compute jaccard similarity
    print("Computing jaccard similarity...")
    intersection_size = np.matmul(diseases, diseases.T)
    np.fill_diagonal(intersection_size, 0)
    N = np.sum(diseases, axis=1, keepdims=True)
    union_size = np.add(N, N.T)
    jaccard_sim = 1.0*intersection_size / (union_size - intersection_size)

    splits = {key: set() for key in split_sizes.keys()}

    # build splits
    print("Splitting dataset...")
    jaccard_network = nx.from_numpy_matrix(jaccard_sim > threshold)
    connected_components = list(nx.connected_components(jaccard_network))
    splits = {}
    for split, total_size in split_sizes.items():
        if total_size == -1:
            overflow_split = split
            continue 
        splits[split] = []
        while len(splits[split]) < total_size:
            idx = random.randint(0, len(connected_components)-1)
            cc = connected_components[idx]
            if total_size >= len(splits[split]) + len(cc):
                connected_components.pop(idx)
                splits[split].extend(cc)
                print(total_size)

    splits[overflow_split] = [idx for cc in connected_components for idx in cc]
    print(splits)

    df = pd.read_csv(disease_path)
    row_splits = ["none"] * len(df)
    for split, idxs in splits.items():
        for idx in idxs:
            #print(split)
            disease = index_to_disease[idx]
            row = df.index[df["Disease ID"] == disease.id][0]
            row_splits[row] = split

    df['splits'] = row_splits
    directory, filename = os.path.split(disease_path)
    print(threshold)
    df.to_csv(os.path.join(directory, filename[:-4] + "_cc.csv"), index=False)
    return df 
