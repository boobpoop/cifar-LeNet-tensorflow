import numpy as np
import pandas as pd
import os
import pickle

def load_a_file(data_path):
    with open(data_path, "rb") as file:
        dict1 = pickle.load(file, encoding = "latin1")
        image_data = dict1["data"]
        image_labels = dict1["labels"]
    return (image_data, image_labels)    

#for cifar_train.py
def get_train_data(ROOT):
    data = []
    labels = []
    for index in range(1, 6):
        data_path = os.path.join(ROOT, "data_batch_%d" %(index))
        (image_data, image_labels) = load_a_file(data_path)
        data.append(image_data)
        labels.append(image_labels)
    
    data_train = np.concatenate(data).reshape([50000, 3, 32, 32]).transpose([0, 2, 3, 1]) / 255.0
    labels_train_pro = np.array(pd.get_dummies(np.concatenate(labels)))
    return (data_train, labels_train_pro)

#for cifar_validate.py
def get_test_data():
    with open("cifar_dataset/test_batch", "rb") as file:
        dict2 = pickle.load(file, encoding = "latin1")
    data_test = np.array(dict2["data"]).reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1]) / 255.0
    labels_test = np.array(dict2["labels"]).reshape([10000])
    labels_test_pro = np.array(pd.get_dummies(labels_test))
    return (data_test, labels_test_pro)