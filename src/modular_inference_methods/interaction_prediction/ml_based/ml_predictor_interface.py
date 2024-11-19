import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.modular_inference_methods.interaction_prediction.predictor_interface import PredictorInterface
from src.utils.protein_protein_interaction import PPI
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.cluster_detection import calculate_clusters
from sklearn.ensemble import GradientBoostingClassifier
from src.analysis_tools.interface import get_ap_interaction
from src.analysis_tools.feature_calculation import *

pd.options.mode.chained_assignment = None  # default='warn'


class MlPredictorInterface(PredictorInterface):
    def __init__(self):
        self.ml = True
        self.feature_names=[]
    def predict_single_ppi(self, ppp, params) -> PPI:
        ppi = PPI(ppp)
        features = self.ppp_to_features(ppp, params)
        interaction_prediction = self.predict_single_feature_list(features,params)
        ppi.interaction_confidence = interaction_prediction
        return ppi

    def predict_single_feature_list(self, feature, params):
        prediction = self.clf.predict_proba([feature])
        if len(prediction[0]) == 2:
            return prediction[0][1]
        else:
            return prediction[0]

    def predict_multiple_feature_list(self, features, params):
        prediction = self.clf.predict_proba(features)
        if len(prediction[0]) == 2:
            return [i[1] for i in prediction]
        else:
            return prediction


    def learn(self, features, labels, params) -> None:
        print('no learning method is define yet')

    def ppp_to_features(self, ppp, df_top_ecs, params, with_labels=False):
        feature_names = []
        cn_dist_metrics_tuple = cn_dist_metrics(ppp, params)
        clusters = calculate_clusters(ppp, params, df_top_ecs)
        from src.utils.pdb_parser import parse_pdb_file
        models = [parse_pdb_file(x) for x in ppp.haddock_results]
        features = []



        wanted = params['include_features_interaction']

        features.extend([ppp.name])
        feature_names.extend(
            ['prefix'])

        cn_dist_metrics_names = ['kurtosis', 'skewness', 'max(cn)', 'median(cn)', 'iqr(cn)',
                                 'jarque_bera_test_statistic']
        for j, n in enumerate(cn_dist_metrics_names):
            if n in wanted:
                features.extend([list(cn_dist_metrics_tuple)[j]])
                feature_names.extend([cn_dist_metrics_names[j]])

        if 'num_top_ecs' in wanted:
            features.extend([n_ecs(df_top_ecs)])
            feature_names.extend(
                ['num_top_ecs'])

        if 'num_clusters' in wanted:
            features.extend([n_clusters(clusters)])
            feature_names.extend(
                ['num_clusters'])

        if 'size_clusters' in wanted:
            features.extend([size_clusters(clusters)])
            feature_names.extend(
                ['size_clusters'])

        if 'n_seq' in wanted:
            features.extend([n_seq(ppp)])
            feature_names.extend(
                ['n_seq'])

        if 'n_eff' in wanted:
            features.extend([n_eff(ppp)])
            feature_names.extend(
                ['n_eff'])

        if 'sequence_length' in wanted:
            features.extend([sequence_length(ppp)])
            feature_names.extend(
                ['sequence_length'])

        if 'haddock_score_best' in wanted:
            features.extend([haddock_score_best(ppp)])
            feature_names.extend(
                ['haddock_score_best'])

        if 'haddock_score_average' in wanted:
            features.extend([haddock_score_average(ppp)])
            feature_names.extend(
                ['haddock_score_average'])

        if 'haddock_num_contacts' in wanted:
            features.extend([haddock_num_contacts(models, params)])
            feature_names.extend(
                ['haddock_num_contacts'])

        if 'haddock_ecs_in_all' in wanted or 'haddock_ecs_in_half' in  wanted:
            all, half = haddock_ecs_in_all(models, df_top_ecs, params)
        if 'haddock_ecs_in_all' in wanted:
            features.extend([all])
            feature_names.extend(
                ['haddock_ecs_in_all'])

        if 'haddock_ecs_in_half' in wanted:
            features.extend([half])
            feature_names.extend(
                ['haddock_ecs_in_half'])


        if with_labels:
            return feature_names, [features], [get_ap_interaction(ppp, params)]
        else:
            return feature_names, [features]