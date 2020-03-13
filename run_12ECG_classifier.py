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
    return list(label.values()), list(score.values())


def load_12ECG_model():
    # load the model from disk 
    loaded_model = Model()
    loaded_model.load()

    return loaded_model
