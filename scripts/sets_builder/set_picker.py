import json
import os

import numpy as np
from ..data_processing import distance_matrix_generator as dmg


class SetPicker:
    def __init__(self, a_base_path, b_base_path):
        self.a_base_path = a_base_path
        self.b_base_path = b_base_path

        self.a_set = None
        self.b_set = None
        self.c_set = None

        self.labels = None

        self.get_labels()

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
            with open(os.path.join(self.a_base_path, 'distances', 'Infoboxes_distances.json'), 'r', encoding='utf-8') as file_a:
                # Get "filenames" from the Json
                self.a_set = json.load(file_a)["filenames"]
                for i in range(len(self.a_set)):
                    self.a_set[i] = self.a_set[i].replace('_words.json', '') 
                self.a_set = list(set(self.a_set))   
        return self.a_set

    def get_b_set(self):
        if self.b_set is None:
            with open(os.path.join(self.b_base_path, 'distances', 'Infoboxes_distances.json'), 'r', encoding='utf-8') as file_b:
                # Get "filenames" from the Json
                self.b_set = json.load(file_b)["filenames"]
                for i in range(len(self.b_set)):
                    self.b_set[i] = self.b_set[i].replace('_words.json', '')
                self.b_set = list(set(self.b_set))
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
            # print duplicates
            for name in self.a_set:
                if name in self.b_set:
                    print(name)
            self.c_set = list(set(self.c_set))
        return self.c_set
        
    def get_xi_matrix(self, z_train, xi_train):
        names = []
        names.extend(z_train)
        names.append(xi_train)
        return self.get_matrix(names)    
    
    def get_z_matrices(self, z_train):
        return self.get_matrix(z_train)

    def get_matrix(self, names):
        # Get paths
        info_paths = []
        link_paths = []
        text_paths = []
        for name in names:
            if self.labels[name] == 'a':
                info_paths.append(os.path.join(self.a_base_path, 'words', 'Infoboxes', f'{name}_words.json'))
                link_paths.append(os.path.join(self.a_base_path, 'words', 'Links', f'{name}_words.json'))
                text_paths.append(os.path.join(self.a_base_path, 'words', 'Texts', f'{name}_words.json'))
            else:
                info_paths.append(os.path.join(self.b_base_path, 'words', 'Infoboxes', f'{name}_words.json'))
                link_paths.append(os.path.join(self.b_base_path, 'words', 'Links', f'{name}_words.json'))
                text_paths.append(os.path.join(self.b_base_path, 'words', 'Texts', f'{name}_words.json'))
        return dmg.calculate_distance_matrices(info_paths, link_paths, text_paths)


