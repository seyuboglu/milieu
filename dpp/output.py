import numpy as np
import sklearn.metrics
import csv

def write_dict_to_csv(filename, dict): 
    with open(filename, "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dict.keys())
        writer.writerows(zip(*dict.values()))

class ExperimentReader():
    def __init__(self, filename):
        """ Reads metrics csv into an array which can be
        indexed using the header_to_col and disease_to_row 
        dictionaries.
        """
        # Read metrics into dictionaries
        with open(filename, 'r') as metrics_file:
            metrics_reader = csv.DictReader(metrics_file)
            disease_to_metrics = {row["Disease ID"]: row for row in metrics_reader}
            self.header_to_col = {name: i for i, name in enumerate(metrics_reader.fieldnames[2:])}
            self.disease_to_row = {id: i for i, id in enumerate(disease_to_metrics.keys())}
        
        # Build metrics array
        self.metrics = np.zeros((len(disease_to_metrics), len(self.header_to_col)))
        for disease, metrics in disease_to_metrics.items():
            for header, metric in metrics.items():
                if (header in self.header_to_col):
                    self.metrics[self.disease_to_row[disease], self.header_to_col[header]] =  metric 

    def get_col(self, header):
        """ Get one column of the experiment metrics, output as a 
        an ndarray.
        """
        return self.metrics[:, self.header_to_col[header]]

        
class ExperimentResults():
    def __init__(self):
        self.result_grid = []
        self.rg_col_map = {"Disease ID": 0, "Disease Name": 1}
        self.rg_col_labels = ["Disease ID", "Disease Name"]
        self.n_cols = 2
        self.rg_row_map = {}

    def add_disease_row(self, disease_id, disease_name):
        result_row = ["" for i in range(self.n_cols)]
        result_row[0] = disease_id
        result_row[1] = disease_name
        self.rg_row_map[disease_id] = len(self.result_grid)
        self.result_grid.append(result_row)

    def add_disease_row_multiple(self, diseases):
        for disease_id, disease in diseases.items():
            self.add_disease_row(disease_id, disease.name)

    def add_data_col_def(self, label, def_value = None):
        col = self.n_cols
        self.n_cols += 1
        self.rg_col_map[label] = col
        self.rg_col_labels.append(label)
        for row in self.result_grid:
            row.append(def_value if(def_value) else "")
        return col

    def add_data_col_def_multiple(self, label_def_value_map):
        for label, def_value in label_def_value_map.items():
            self.add_data_col(label, def_value)
    
    def add_data_col(self, col_label, disease_value_map):
        col = self.n_cols
        self.n_cols += 1
        self.rg_col_map[col_label] = col
        self.rg_col_labels.append(col_label)
        for row in self.result_grid:
            row.append("")
        for disease_id, value in disease_value_map.items():
            self.add_data_row(disease_id, col_label, value)


    def add_data_row(self, disease_id, col_label, value):
        if(disease_id not in self.rg_row_map):
            print("Error: disease_id does not exist")
        row = self.rg_row_map[disease_id]
        col = None
        if (col_label in self.rg_col_map):
            col = self.rg_col_map[col_label]
        else:
            col = self.add_data_col(col_label)
        self.result_grid[row][col] = str(value)

    def add_data_row_multiple(self, disease_id, label_value_map):
        if(disease_id not in self.rg_row_map):
            print("Error: disease_id does not exist")
        row = self.rg_row_map[disease_id]
        for col_label, value in label_value_map.items():
            col = None
            if (col_label in self.rg_col_map):
                col = self.rg_col_map[col_label]
            else:
                col = self.add_data_col_def(col_label)
            self.result_grid[row][col] = str(value)
    
    def compute_statistic(self, fn, nd_result_grid):
        stats = {}
        for label, index in self.rg_col_map.items():
            if (index - 2 < 0): continue
            stat = fn(nd_result_grid[:,index-2])
            stats[label] = stat 
        return stats
    
    def add_statistics(self): 
        nd_result_grid = np.array(self.result_grid)
        nd_result_grid = nd_result_grid[:,2:].astype(float)
        means = self.compute_statistic(np.mean, nd_result_grid)
        medians = self.compute_statistic(np.median, nd_result_grid)
        stdevs = self.compute_statistic(np.std, nd_result_grid) 
        self.add_disease_row("Mean", "")
        self.add_data_row_multiple("Mean", means)
        self.add_disease_row("Median", "")
        self.add_data_row_multiple("Median", medians)
        self.add_disease_row("STD", "")
        self.add_data_row_multiple("STD", stdevs)

    def output_to_csv(self, filename):
        with open(filename, 'a') as csvfile:
            table_writer = csv.writer(csvfile, delimiter=',')
            table_writer.writerow(self.rg_col_labels)
            for row in self.result_grid:
                table_writer.writerow(row)

