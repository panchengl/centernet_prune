import numpy as np
import torch
import torch.nn.functional as F
import os

def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[name.strip('\n')] = ID
    return list(names.keys())

def get_classes_standard_dict(classes):
    dict = {}
    for i, cl in enumerate(classes):
        dict[cl] = i
    return dict