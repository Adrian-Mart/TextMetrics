import json
import os

import numpy as np

from scripts.data_processing import analyzer
from ..data_processing import distance_matrix_generator as dmg


class SetPicker:
    def __init__(self, base_path, a_base_path, b_base_path):
        self.a_base_path = a_base_path
        self.b_base_path = b_base_path

        self.a_set = None
        self.b_set = None
        self.c_set = None
        self.z_set = None

        self.labels = None

        self.matrices_data = None
        with open(os.path.join(base_path, 'distance_matrices.json'), 'r') as f:
            self.matrices_data = json.load(f)
            self.matrices = {
                'infoboxes': self.matrices_data['infoboxes_distance_matrix'],
                'links': self.matrices_data['links_distance_matrix'],
                'texts': self.matrices_data['texts_distance_matrix']
            }

        self.get_labels()

    def set_z_set(self, z_set):
        self.z_set = z_set

    def get_labels(self):
        if self.labels is None:
            if self.a_set is None:
                self.get_a_set()
            if self.b_set is None:
                self.get_b_set()
            self.labels = {}
            for name in self.a_set:
                self.labels[name] = 'a'
            for name in self.b_set:
                self.labels[name] = 'b'
        return self.labels

    def get_a_set(self):
        if self.a_set is None:
            self.a_set = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(self.a_base_path, 'words', 'Infoboxes'))] 
        return self.a_set

    def get_b_set(self):
        if self.b_set is None:
            self.b_set = [os.path.splitext(file)[0] for file in os.listdir(os.path.join(self.b_base_path, 'words', 'Infoboxes'))]
        return self.b_set

    def get_c_set(self):
        if self.c_set is None:
            if self.a_set is None:
                self.get_a_set()
            if self.b_set is None:
                self.get_b_set()
            self.c_set = []
            self.c_set.extend(self.a_set)
            self.c_set.extend(self.b_set)
            self.c_set = list(set(self.c_set))
        return self.c_set
        
    def get_predictions(self, alpha, beta, gamma, threshold = 50):
        dictionary = self.matrices_data['dictionary']
        matrices = [self.matrices['infoboxes'], self.matrices['links'], self.matrices['texts']]
        general_distance_data = analyzer.get_distance_data(alpha, beta, gamma, threshold, matrices)
        z_submatrices = [SetPicker.get_submatrix(matrix, self.matrices_data['dictionary'], self.z_set) for matrix in matrices]
        z_distance_data = analyzer.get_distance_data(alpha, beta, gamma, threshold, z_submatrices)

        z_min = z_distance_data['min_value']

        predictions = {}
        for i in dictionary.keys():
            xi_name = i
            if xi_name in self.z_set:
                continue
            xi_submatrix, index = SetPicker.get_submatrix(general_distance_data['raw_distance_matrix'], dictionary, [xi_name] + self.z_set, xi_name)
            x_min = min([dist for dist in xi_submatrix[index] if dist > 0])
            predictions[xi_name]= x_min < z_min

        return predictions
    
    def get_submatrix(matrix, index_dict, item_names, xi_name=None):
        # Get the indexes of the items in the matrix
        indexes = [index_dict[name] for name in item_names]
        submatrix = [[matrix[i][j] for j in indexes] for i in indexes]
        if xi_name:
            xi_index = indexes.index(index_dict[xi_name])
            return submatrix, xi_index
        else:
            return submatrix


