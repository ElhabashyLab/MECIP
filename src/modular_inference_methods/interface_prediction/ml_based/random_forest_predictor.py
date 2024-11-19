import pandas as pd
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
from src.modular_inference_methods.interface_prediction.ml_based.ml_interface_predictor_interface import MlInterfacePredictorInterface
from src.utils.protein_protein_interaction import PPI
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.cluster_detection import calculate_clusters
from sklearn.ensemble import RandomForestClassifier
from src.modular_inference_methods.interface_prediction.ml_based.ml_interface_predictor_interface import MlInterfacePredictorInterface
from src.analysis_tools.feature_calculation import *
import json
pd.options.mode.chained_assignment = None  # default='warn'


class RandomForestPredictor(MlInterfacePredictorInterface):

    def learn(self, features, labels, params) -> None:

        # no parameter grid
        if params['interface_predictor_parameter_grid'] is None:
            # ml model
            self.clf = RandomForestClassifier()
            # fit to data:
            self.clf.fit(features, labels)

        # default parameter space for ml model
        elif params['interface_predictor_parameter_grid'] == 'default':
            #TODO
            parameter_space = {'n_estimators': [100, 50, 500],
                               'min_samples_split': [2, 5],
                               'max_features': ['sqrt', 'log2']}
            # ml model
            self.clf = GridSearchCV(RandomForestClassifier(), parameter_space, n_jobs=params['n_jobs'],
                                    verbose=1, cv=3)
            # fit to data:
            self.clf.fit(features, labels)

        # read parameter space for ml model
        else:
            try:
                with open(params['interface_predictor_parameter_grid']) as f:
                    data = f.read()
                parameter_space = json.loads(data)
                # ml model
                self.clf = GridSearchCV(RandomForestClassifier(), parameter_space, n_jobs=params['n_jobs'],
                                        verbose=10, cv=3)
                # fit to data:
                self.clf.fit(features, labels)
            except Exception:
                grid_location = params['interface_predictor_parameter_grid']
                print(f'parameter grid at {grid_location} was not readable. No parameter grid will be used.')
                # ml model
                self.clf = RandomForestClassifier()
                # fit to data:
                self.clf.fit(features, labels)






        #print(self.clf.score(features, labels))
        #print(self.clf.get_params())
