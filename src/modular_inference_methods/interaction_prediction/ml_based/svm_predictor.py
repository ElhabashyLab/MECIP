import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.modular_inference_methods.interaction_prediction.ml_based.ml_predictor_interface import MlPredictorInterface
from src.utils.protein_protein_interaction import PPI
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.cluster_detection import calculate_clusters
from sklearn.svm import SVC
from src.analysis_tools.interface import get_ap_interaction
from src.analysis_tools.feature_calculation import *
import numpy as np
import json
pd.options.mode.chained_assignment = None  # default='warn'


class SVMPredictor(MlPredictorInterface):
    def learn(self, features, labels, params) -> None:
        # no parameter grid
        if params['interaction_predictor_parameter_grid'] is None:
            # ml model
            self.clf = SVC(probability=True)
            # fit to data:
            self.clf.fit(features, labels)

        # default parameter space for ml model
        elif params['interaction_predictor_parameter_grid'] == 'default':
            parameter_space = {'C': [1, .1, 10 ],
                               'gamma': ['scale', 'auto'],
                               'kernel': ['rbf', 'poly', 'sigmoid']}
            # ml model
            self.clf = GridSearchCV(SVC(probability=True), parameter_space, n_jobs=params['n_jobs'],
                                    verbose=1, cv=3)
            # fit to data:
            self.clf.fit(features, labels)

        # read parameter space for ml model
        else:
            try:
                with open(params['interaction_predictor_parameter_grid']) as f:
                    data = f.read()
                parameter_space = json.loads(data)
                # ml model
                self.clf = GridSearchCV(SVC(probability=True), parameter_space, n_jobs=params['n_jobs'],
                                        verbose=10, cv=3)
                # fit to data:
                self.clf.fit(features, labels)
            except Exception:
                grid_location = params['interaction_predictor_parameter_grid']
                print(f'parameter grid at {grid_location} was not readable. No parameter grid will be used.')
                # ml model
                self.clf = SVC(probability=True)
                # fit to data:
                self.clf.fit(features, labels)

        # print(self.clf.score(features, labels))
        # print(self.clf.get_params())

