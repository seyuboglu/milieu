"""General utility functions"""

import json
import logging
import smtplib
from re import sub
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from collections import OrderedDict

import numpy as np
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


def prepare_sns(sns, params):
    """ Prepares seaborn for plotting according to the plot settings specified in
    params. 
    Args: 
        params (object) dictionary containing settings for each of the seaborn plots
    """
    sns.set_context('paper', font_scale=1) 
    sns.set(palette=tuple(getattr(params, "plot_palette", ["#E03C3F", "#FF9300", 
                                                           "#F8BA00", "#CB297B", 
                                                           "#6178A8", "#56C1FF"])),
            font=getattr(params, "plot_font", "Times New Roman"))
    sns.set_style(getattr(params, "plot_style", "ticks"),  
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
    return map(type, string.split(delimiter))


def fraction_nonzero(array):
    """
    Computes fraction of elements in array that are nonzero
    """
    return 1 - np.mean(np.isclose(0, array))


def print_title(title="Experiment", subtitle=None):
    print(title)
    if subtitle is not None: 
        print(subtitle)
    print("Sabri Eyuboglu  -- SNAP Group -- Stanford University")
    print("====================================================")

def torch_all_close(a, b, tolerance=1e-12):
    """
    Check if all elements in two tensors are equal within a tolerance.
    """
    return torch.all(torch.lt(torch.abs(a - b), tolerance))
