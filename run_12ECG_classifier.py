#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
from algorithm import Model


def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    label = model.predict(data, header_data)
    score = model.scores_final

    keys = sorted(label.keys(), reverse=True)

    for key in keys:
        current_label = np.append(current_label, label[key])
        current_score = np.append(current_score, score[key])

    return list(current_label), list(current_score)


def load_12ECG_model():
    # load the model from disk 
    loaded_model = Model()
    loaded_model.load()

    return loaded_model
