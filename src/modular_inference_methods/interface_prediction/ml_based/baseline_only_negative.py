import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.modular_inference_methods.interface_prediction.predictor_interface import PredictorInterface
from src.utils.protein_protein_interaction import PPI
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.cluster_detection import calculate_clusters
from sklearn.ensemble import GradientBoostingClassifier
from src.modular_inference_methods.interface_prediction.ml_based.ml_interface_predictor_interface import MlInterfacePredictorInterface
from src.analysis_tools.feature_calculation import *
import json
pd.options.mode.chained_assignment = None  # default='warn'



# DISCLAIMER
# this is no machine learning model but rather a basic function which is used as  a comparison baseline for machine learning models


class BaselineOnlyNegative(MlInterfacePredictorInterface):

    def predict_single_ppi(self, ppp, params) -> PPI:
        ppi = PPI(ppp)
        df_top_ecs = get_top_ecs(ppp, params)
        df_top_ecs['prediction_confidence'] = 0
        ppi.df_ecs = df_top_ecs
        return ppi

    def predict_multiple_feature_list(self, features, params):
        prediction = []
        for f in features:
            prediction.append(False)
        return prediction

    def learn(self, features, labels, params) -> None:

        #substitute code
        self.clf = GradientBoostingClassifier()
        self.clf.fit(features, labels)



    def ppp_to_features(self, ppp, df_top_ecs, params, with_labels=False):

        # substitute code
        features = []
        labels = []
        feature_names = []

        for i, (idx, row) in enumerate(df_top_ecs.iterrows()):
            curr_features = []

            curr_features.extend([ppp.name])
            if i == 0: feature_names.extend(['prefix'])

            curr_features.extend([int(row.name)])
            if i == 0: feature_names.extend(['ec_row'])

            curr_features.extend([i])
            if i == 0: feature_names.extend(['i'])


            # add curr_feature to features and actual label to labels
            features.append(curr_features)
            if with_labels: labels.append(int(row['true_ec']))

        if with_labels:
            return feature_names, features, labels
        else:
            return feature_names, features