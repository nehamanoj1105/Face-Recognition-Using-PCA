import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

def load_lfw(min_faces_per_person=50, resize=0.4):
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
    X = lfw.images
    y = lfw.target
    target_names = lfw.target_names
    h, w = X.shape[1:]
    return X, y, target_names, (h, w)

def flatten_images(X):
    n_samples, h, w = X.shape
    return X.reshape(n_samples, h * w)

def split_data(X, y, test_size=0.25, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
