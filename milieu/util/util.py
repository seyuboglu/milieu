"""General utility functions"""

import os, sys
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


class Process(object):
    """ 
    Base class for all disease protein prediction processes.
    """

    def __init__(self, dir, params):
        """ Initialize the 
        Args: 
            dir (string) The directory where the experiment should be run
            params  (dict)
        """
        self.dir = dir
        # ensure dir exists
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
            with open(os.path.join(self.dir, "params.json"), "w") as f:
                json.dump(params, f, indent=4)

        self.params = params
        set_logger(
            os.path.join(self.dir, "process.log"), level=logging.INFO, console=True
        )

    def is_completed(self):
        """
        Checks if the experiment has already been run. 
        """
        return os.path.isfile(os.path.join(self.dir, "results.csv"))

    def _run(self):
        raise NotImplementedError

    def run(self, overwrite=False):
        if os.path.isfile(os.path.join(self.dir, "results.csv")) and not overwrite:
            print("Experiment already run.")
            return False

        if hasattr(self.params, "notify") and self.params.notify:
            try:
                self._run()
            except:
                tb = traceback.format_exc()
                self.notify_user(error=tb)
                return False
            else:
                self.notify_user()
                return True
        self._run()
        return True

    def notify_user(self, error=None):
        # read params
        with open(os.path.join(self.dir, "params.json"), "r") as file:
            params_string = file.readlines()
        if error is None:
            subject = "Experiment Completed: " + self.dir
            message = (
                "Yo!\n",
                "Good news, your experiment just finished.",
                "You were running the experiment on: {}".format(socket.gethostname()),
                "---------------------------------------------",
                "See the results here: {}".format(self.dir),
                "---------------------------------------------",
                "The parameters you fed to this experiment were: {}".format(
                    params_string
                ),
                "---------------------------------------------",
                "Thanks!",
            )
        else:
            subject = "Experiment Error: " + self.dir
            message = (
                "Uh Oh!\n",
                "Your experiment encountered an error.",
                "You were running the experiment found at: {}".format(self.dir),
                "You were running the experiment on: {}".format(socket.gethostname()),
                "---------------------------------------------",
                "Check out the error message: \n{}".format(error),
                "---------------------------------------------",
                "The parameters you fed to this experiment were: {}".format(
                    params_string
                ),
                "---------------------------------------------",
                "Thanks!",
            )

        message = "\n".join(message)
        send_email(subject, message)

    def __call__(self):
        return self.run()

    def summarize_results(self):
        """
        Returns the summary of the dataframe 
        return:
            summary_df (DataFrame) 
        """
        return self.results.describe()


def load_params(experiment_dir):
    """
    loads the params at the experiment dir
    """
    with open(os.path.join(experiment_dir, "params.json")) as f:
        params = json.load(f, object_pairs_hook=OrderedDict)
    return params


def set_logger(log_path=None, level=logging.INFO, console=True):
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
        if log_path is not None:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
            )
            logger.addHandler(file_handler)

        # Logging to console
        if console:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter("%(message)s"))
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
    sns.set_context("paper", font_scale=1)
    sns.set(
        palette=tuple(params.get("plot_palette", ["#E03C3F", "lightgrey"])),
        font=params.get("plot_font", "Times New Roman"),
        **kwargs,
    )
    sns.set_style(
        params.get("plot_style", "ticks"),
        {
            "xtick.major.size": 5.0,
            "xtick.minor.size": 5.0,
            "ytick.major.size": 5.0,
            "ytick.minor.size": 5.0,
        },
    )


def send_email(subject, message, to_addr=None, login=None, pw=None):
    """
    Send emails notifying the completion of experiments. 
    """
    if to_addr is None or login is None or pw is None:
        return

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(login, pw)

    msg = MIMEMultipart()  # create a message

    # add in the actual person name to the message template

    # setup the parameters of the message
    msg["From"] = f"{login}@gmail.com"
    msg["To"] = to_addr
    msg["Subject"] = subject

    # add in the message body
    msg.attach(MIMEText(message, "plain"))

    problems = server.sendmail(f"{login}@gmail.com", to_addr, msg.as_string())
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
    return np.logical_or(
        (null_results > result), np.isclose(null_results, result)
    ).mean()


def load_mapping(
    path, delimiter=" ", reverse=False, a_transform=lambda x: x, b_transform=lambda x: x
):
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
            if line[0] == "#":
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
        network (Network)
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

        if len(curr_bucket) >= min_len:
            prev_bucket = curr_bucket
            curr_bucket = None
            curr_degrees = []

    if (
        curr_bucket is not None
        and prev_bucket is not None
        and len(curr_bucket) < min_len
    ):
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
        data = {key: place_on_cpu(val) for key, val in data.items()}
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach()
    else:
        return data


def ensure_dir_exists(dir):
    """
    Ensures that a directory exists. Creates it if it does not.
    args:
        dir     (str)   directory to be created
    """
    if not (os.path.exists(dir)):
        parent_dir = os.path.dirname(dir)
        if parent_dir:
            ensure_dir_exists(parent_dir)
        os.mkdir(dir)

