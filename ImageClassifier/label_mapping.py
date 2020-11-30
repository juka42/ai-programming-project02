#label mapping
import json

def label_mapping(label_filename, class_to_idx):
    ###############
    #label mapping
    # this defines a cat to name dictionary
    with open(label_filename, 'r') as f:
        cat_to_name = json.load(f)
    #cat_to_name

    #this defines a index to category dictionary
    idx_to_class={}
    for key, value in class_to_idx.items():
        idx_to_class[value]=key
    # idx_to_class

    # this defines a index to name dictionary
    idx_to_label = {}
    for key, value in idx_to_class.items():
        idx_to_label[key] = cat_to_name[value]

    return cat_to_name, idx_to_class, idx_to_label
