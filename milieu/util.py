"""General utility functions"""
import os
import json
import logging
import smtplib
from re import sub
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch 

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f,  object_pairs_hook=OrderedDict)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__
    
def load_params(experiment_dir):
    """
    loads the params at the experiment dir
    """
    with open(os.path.join(experiment_dir, "params.json")) as f:
        params = json.load(f,  object_pairs_hook=OrderedDict)
    return params
    

def set_logger(log_path, level=logging.INFO, console=True):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        if console: 
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)

def parse_id_rank_pair(str):
    """ Parses a rank entry. Since protein labels are sometimes not included
    return -1 for protein id.  
    """
    if "=" not in str:
        return -1, float(str)
    
    id, rank = str.split("=")
    return int(id), float(rank)


def prepare_sns(sns, params={}, kwargs={}):
    """ Prepares seaborn for plotting according to the plot settings specified in
    params. 
    Args: 
        params (object) dictionary containing settings for each of the seaborn plots
    """
    sns.set_context('paper', font_scale=1) 
    sns.set(palette=tuple(params.get("plot_palette", ["#E03C3F", "lightgrey"])),
            font=params.get("plot_font", "Times New Roman"), **kwargs)
    sns.set_style(params.get("plot_style", "ticks"),  
                  {'xtick.major.size': 5.0, 'xtick.minor.size': 5.0, 
                   'ytick.major.size': 5.0, 'ytick.minor.size': 5.0})


def send_email(subject, message, to_addr="eyuboglu@stanford.edu"):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("sabriexperiments", "experiments")

    msg = MIMEMultipart()       # create a message

    # add in the actual person name to the message template

    # setup the parameters of the message
    msg['From'] = "sabriexperiments@gmail.com"
    msg['To'] = to_addr
    msg['Subject'] = subject

    # add in the message body
    msg.attach(MIMEText(message, 'plain'))

    problems = server.sendmail("sabriexperiments@gmail.com", 
                               to_addr, 
                               msg.as_string())
    server.quit()


def string_to_list(string, type=str, delimiter=" "):
    """
    Converts a string to a list. 
    """
    string = string.strip("[] ")
    string = string.replace("\n", "")
    string = sub(" +", " ", string)
    return list(map(type, string.split(delimiter)))


def list_to_string(list, delimiter=" "):
    """
    Converts a string to a list. 
    """
    string = "[" + delimiter.join(map(str, list)) + "]"
    return string


def fraction_nonzero(array):
    """
    Computes fraction of elements in array that are nonzero
    """
    return 1 - np.mean(np.isclose(0, array))


def print_title(title="Experiment", subtitle=None):
    print(title)
    if subtitle is not None: 
        print(subtitle)
    print("Sabri Eyuboglu, Marinka Zitnik and Jure Leskovec -- SNAP Group")
    print("==============================================================")


def torch_all_close(a, b, tolerance=1e-12):
    """
    Check if all elements in two tensors are equal within a tolerance.
    """
    return torch.all(torch.lt(torch.abs(a - b), tolerance))


def compute_pvalue(result, null_results):
    """
    """
    null_results = np.array(null_results)
    return np.logical_or((null_results > result), 
                          np.isclose(null_results, result)).mean()


def load_mapping(path, delimiter=" ", reverse=False, 
                 a_transform=lambda x: x,
                 b_transform=lambda x: x):
    """
    Loads a str-str mapping from a text file of the form:     
        ' a b
          c d
          ...'
    such that, mapping[a] = b and mapping[c] = d.
    Ignores any lines with missing input. For example:
        ' a 
          c d
          ...'
    Args:
        reverse (bool) reverse the order of the mapping so that mapping[b] = a etc.
        delimiter (str) the string separating the two elements on each line
    """
    mapping = {}
    with open(path) as f:
        for line in f:
            if line[0] == '#':
                continue
            a, b = line.split(delimiter)
            b = b.strip("\n")

            if not a or not b:
                continue
                
            a = a_transform(a)
            b = b_transform(b)

            if reverse:
                mapping[b] = a
            else:
                mapping[a] = b

    return mapping


def build_degree_buckets(network, min_len=500):
    """
    Buckets nodes by degree such that no bucket has less than min_size nodes.
    args:
        network (PPINetwork)
        min_len (int)  minimum bucket size
    return:
        degree_to_bucket  (dict)    map from degree to the corresponding bucket
    """
    network = network.nx
    degrees = np.array(network.degree())[:, 1]

    # build degree to buckets
    degree_to_buckets = defaultdict(list)
    max_degree = np.max(degrees)
    for node, degree in enumerate(degrees):
        degree_to_buckets[degree].append(node)

    # enforce min_len
    curr_bucket = None
    prev_bucket = None
    curr_degrees = []
    for degree in range(max_degree + 1):
        # skip nonexistant degrees
        if degree not in degree_to_buckets:
            continue
        
        curr_degrees.append(degree)

        # extend current bucket if necessary
        if curr_bucket is not None:
            curr_bucket.extend(degree_to_buckets[degree])
            degree_to_buckets[degree] = curr_bucket
        else: 
            curr_bucket = degree_to_buckets[degree]
            
        if(len(curr_bucket) >= min_len):
            prev_bucket = curr_bucket
            curr_bucket = None
            curr_degrees = []

    if curr_bucket is not None and prev_bucket is not None and len(curr_bucket) < min_len:
        prev_bucket.extend(curr_bucket)
        for degree in curr_degrees:
            degree_to_buckets[degree] = prev_bucket

    return degree_to_buckets

def place_on_gpu(data, device=0):
    """
    Recursively places all 'torch.Tensor's in data on gpu and detaches.
    If elements are lists or tuples, recurses on the elements. Otherwise it
    ignores it.
    source: inspired by place_on_gpu from Snorkel Metal
    https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    """
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_gpu(data[i], device) for i in range(len(data))]
        data = data_type(data)
        return data
    elif data_type is dict:
        data = {key: place_on_gpu(val, device) for key, val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def place_on_cpu(data):
    """
    Recursively places all 'torch.Tensor's in data on cpu and detaches from computation
    graph. If elements are lists or tuples, recurses on the elements. Otherwise it
    ignores it.
    source: inspired by place_on_gpu from Snorkel Metal
    https://github.com/HazyResearch/metal/blob/master/metal/utils.py
    """
    data_type = type(data)
    if data_type in (list, tuple):
        data = [place_on_cpu(data[i]) for i in range(len(data))]
        data = data_type(data)
        return data
    elif data_type is dict:
        data = {key: place_on_cpu(val) for key,val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach()
    else:
        return data

