
import os
import json
import logging
from collections import defaultdict

import numpy as np
import networkx as nx
import torch 
from torch.utils.data import DataLoader
from torch.optim import Adam 
from tqdm import tqdm

from milieu.util.util import place_on_cpu, place_on_gpu
from milieu.paper.methods.method import DPPMethod


class MilieuMethod(DPPMethod):
    """ GCN method class
    """
    def __init__(self, network, diseases, params):
        super().__init__(network, diseases, params)

        self.dir = params["dir"]
        self.adjacency = self.network.adj_matrix
        self.diseases = diseases
        self.params = params
        print(self.params)
        if self.params.get("load", False):
            self.load_method()
        else:
            self.train_method(diseases)
        
        self.curr_fold = None
    
    def load_method(self):
        """
        """
        logging.info("Loading Params...")
        with open(os.path.join(self.dir, "params.json")) as f:
            params = json.load(f)["process_params"]["method_params"]
        params.update(self.params)
        self.params = params
        
        logging.info("Loading Models...")
        self.folds_to_models = {}
        for model_file in os.listdir(os.path.join(self.dir, "models")):
            split = parse.parse("model_{}.tar", model_file)[0]
            self.folds_to_models[split] = os.path.join(self.dir, 
                                                       "models", 
                                                       model_file)


    def train_method(self, diseases):
        """
        """
        logging.info("Training Models...")
        folds_to_diseases = defaultdict(set)
        for disease in diseases.values():
            if disease.split == "none":
                continue
            folds_to_diseases[disease.split].add(disease)
        
        self.folds_to_models = {}
        if not(os.path.exists(os.path.join(self.dir, "models"))):
            os.mkdir(os.path.join(self.dir, "models"))
        
        for test_fold in folds_to_diseases.keys(): 
            logging.info("Training model for test {}".format(test_fold))
            val_fold = str((int(test_fold) - 1) % len(folds_to_diseases))
            test_dataset = DiseaseDataset([disease 
                                           for disease in folds_to_diseases[test_fold]],
                                           self.network)
            val_dataset = DiseaseDataset([disease 
                                          for disease in folds_to_diseases[val_fold]],
                                          self.network) 
            train_dataset = DiseaseDataset([disease  
                                             for fold, diseases in folds_to_diseases.items()
                                             if fold != test_fold and fold != val_fold
                                             for disease in diseases], 
                                            self.network)
            
            # ensure no data leakage
            assert(not set.intersection(*[test_dataset.get_ids(), 
                                          train_dataset.get_ids()]))
            assert(not set.intersection(*[val_dataset.get_ids(),
                                          train_dataset.get_ids()]))

            model = self.train_model(train_dataset, val_dataset)
            path = os.path.join(self.dir, "models/model_{}.tar".format(test_fold))
            torch.save(model.state_dict(), path)
            self.folds_to_models[test_fold] = path
    
    def train_model(self, train_dataset, val_dataset):
        """ Trains the underlying model
        """
        train_dl = DataLoader(train_dataset, 
                              batch_size=self.params["batch_size"], 
                              shuffle=True,
                              num_workers=self.params["num_workers"],
                              pin_memory=self.params["cuda"])
    
        dev_dl = DataLoader(val_dataset, 
                            batch_size=self.params["batch_size"], 
                            shuffle=True,
                            num_workers=self.params["num_workers"],
                            pin_memory=self.params["cuda"])

        if self.params["model_class"] == "LCIEmbModule":
            model = LCIEmbModule(self.params["model_args"], self.network)
        else:
            model = LCIModule(self.params, self.adjacency)

        if self.params["cuda"]:
            model = model.cuda()
        optimizer = Adam(model.parameters(), lr=self.params["learning_rate"], 
                         weight_decay=self.params["weight_decay"])
        
        logging.info("Starting training for {} epoch(s)".format(self.params["num_epochs"]))
        model.train()
        train_and_evaluate(
            model,
            train_dl,
            dev_dl,
            optimizer,
            bce_loss,
            metrics,
            self.params,
            self.dir
        )
        model.eval()
        return model.cpu()


    def compute_scores(self, train_pos, disease):
        """ Compute the scores predicted by GCN.
        Args: 
            
        """
        val_pos = None
        # Adjacency: Get sparse representation of ppi_adj
        N, _ = self.adjacency.shape
        X = torch.zeros(1, N)
        X[0, train_pos] = 1
        if self.params["cuda"]:
            X = X.cuda()
            
        if disease.split != self.curr_fold:
            if self.params["model_class"] == "LCIEmbModule":
                model = LCIEmbModule(self.params["model_args"], self.network)
            else:
                model = LCIModule(self.params, self.adjacency)
            model.load_state_dict(torch.load(self.folds_to_models[disease.split]))          
            model.eval()
            model.cuda()  
            self.curr_model = model 
            self.curr_fold = disease.split
        Y = self.curr_model(X)
        scores = Y.cpu().detach().numpy().squeeze()
        return scores
