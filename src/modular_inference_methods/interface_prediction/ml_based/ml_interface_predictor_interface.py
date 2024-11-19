import copy

import pandas as pd
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
from src.modular_inference_methods.interface_prediction.predictor_interface import PredictorInterface
from src.utils.protein_protein_interaction import PPI
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.cluster_detection import calculate_clusters
from sklearn.ensemble import RandomForestClassifier
import src.analysis_tools.feature_calculation as feat
from src.analysis_tools.feature_calculation import *


pd.options.mode.chained_assignment = None  # default='warn'


class MlInterfacePredictorInterface(PredictorInterface):
    def __init__(self):
        self.ml = True
    def predict_single_ppi(self, ppp, params) -> PPI:
        ppi = PPI(ppp)
        df_top_ecs = get_top_ecs(ppp, params)
        features = self.ppp_to_features(ppp, df_top_ecs, params)

        df_top_ecs['prediction_confidence'] = self.predict_multiple_feature_list(features,params)
        ppi.df_ecs = df_top_ecs

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

        features = []
        labels = []
        feature_names = []
        cn_dist_metrics_tuple = cn_dist_metrics(ppp, params)
        clusters = calculate_clusters(ppp, params, df_top_ecs)
        length_of_first_protein = len(ppp.protein1.get_sequence())
        from src.utils.pdb_parser import parse_pdb_file
        if ppp.haddock_result_best is not None: model = parse_pdb_file(ppp.haddock_result_best)
        models = [parse_pdb_file(x) for x in ppp.haddock_results]
        wanted = params['include_features_interface']
        if 'heavy_atom_distance_in_best_top5' in wanted: top5_model = get_topx_model(models, df_top_ecs, 5, params)
        if 'heavy_atom_distance_in_best_top10' in wanted: top10_model = get_topx_model(models, df_top_ecs, 10, params)


        curr_features = []

        curr_features.extend([ppp.name])
        feature_names.extend(['prefix'])

        curr_features.extend([''])
        feature_names.extend(['ec_row'])

        cn_dist_metrics_names = ['kurtosis', 'skewness', 'max(cn)', 'median(cn)', 'iqr(cn)',
                                 'jarque_bera_test_statistic']
        for j, n in enumerate(cn_dist_metrics_names):
            if n in wanted:
                curr_features.extend([list(cn_dist_metrics_tuple)[j]])
                feature_names.extend([cn_dist_metrics_names[j]])

        if 'num_top_ecs' in wanted:
            curr_features.extend([n_ecs(df_top_ecs)])
            feature_names.extend(['num_top_ecs'])

        if 'num_clusters' in wanted:
            curr_features.extend([n_clusters(clusters)])
            feature_names.extend(['num_clusters'])

        if 'size_clusters' in wanted:
            curr_features.extend([size_clusters(clusters)])
            feature_names.extend(['size_clusters'])

        if 'n_seq' in wanted:
            curr_features.extend([n_seq(ppp)])
            feature_names.extend(['n_seq'])

        if 'n_eff' in wanted:
            curr_features.extend([n_eff(ppp)])
            feature_names.extend(['n_eff'])

        if 'sequence_length' in wanted:
            curr_features.extend([sequence_length(ppp)])
            feature_names.extend(['sequence_length'])

        if 'haddock_score_best' in wanted:
            curr_features.extend([haddock_score_best(ppp)])
            feature_names.extend(
                ['haddock_score_best'])

        if 'haddock_score_average' in wanted:
            curr_features.extend([haddock_score_average(ppp)])
            feature_names.extend(
                ['haddock_score_average'])

        if 'haddock_num_contacts' in wanted:
            curr_features.extend([haddock_num_contacts(models,params)])
            feature_names.extend(
                ['haddock_num_contacts'])

        if 'haddock_ecs_in_all' in wanted or 'haddock_ecs_in_half' in wanted:
            all, half = haddock_ecs_in_all(models, df_top_ecs, params)
        if 'haddock_ecs_in_all' in wanted:
            curr_features.extend([all])
            feature_names.extend(
                ['haddock_ecs_in_all'])

        if 'haddock_ecs_in_half' in wanted:
            curr_features.extend([half])
            feature_names.extend(
                ['haddock_ecs_in_half'])

        curr_features2 = curr_features

        for i, (idx, row) in enumerate(df_top_ecs.iterrows()):
            curr_features = copy.deepcopy(curr_features2)
            curr_features[1] = int(row.name)
            # local
            if 'cn' in wanted:
                curr_features.extend([cn(row)])
                if i == 0: feature_names.extend(['cn'])

            if 'rel_rank_ec' in wanted:
                curr_features.extend([rel_rank_ec(i, df_top_ecs)])
                if i == 0: feature_names.extend(['rel_rank_ec'])

            if 'dist_to_higher_cn' in wanted:
                curr_features.extend([dist_to_higher_cn(i, df_top_ecs)])
                if i == 0: feature_names.extend(['dist_to_higher_cn'])

            if 'dist_to_lower_cn' in wanted:
                curr_features.extend([dist_to_lower_cn(i, df_top_ecs)])
                if i == 0: feature_names.extend(['dist_to_lower_cn'])

            if 'cn_density' in wanted:
                curr_features.extend([cn_density(i, df_top_ecs, bar_size=0)])
                if i == 0: feature_names.extend(['cn_density'])

            calc_conservations = False
            conservation_measure_names = ['added_conservation', 'min_conservation', 'max_conservation']
            for n in conservation_measure_names:
                if n in wanted:
                    calc_conservations = True
                    break
            if calc_conservations:
                conservation_measures = list(
                    conservation(row['i'], length_of_first_protein + row['j'], ppp.frequencies_file_in))
                for j, n in enumerate(conservation_measure_names):
                    if n in wanted:
                        curr_features.extend([conservation_measures[j]])
                        if i == 0: feature_names.extend([conservation_measure_names[j]])

            if 'is_in_cluster' in wanted:
                curr_features.extend([is_in_cluster(clusters, row)])
                if i == 0: feature_names.extend(['is_in_cluster'])

            if 'cluster_size' in wanted:
                curr_features.extend([cluster_size(clusters, row)])
                if i == 0: feature_names.extend(['cluster_size'])

            if 'rsa_ij' in wanted:
                curr_features.extend([rsa_ij(row, ppp, params)])
                if i == 0: feature_names.extend(['rsa_ij'])

            if 'rsa_min' in wanted:
                curr_features.extend([rsa_min(row, ppp, params)])
                if i == 0: feature_names.extend(['rsa_min'])

            if 'heavy_atom_distance_in_top_model' in wanted:
                curr_features.extend([heavy_atom_distance_in_top_model(model, row)])
                if i == 0: feature_names.extend(
                    ['heavy_atom_distance_in_top_model'])

            if 'heavy_atom_distance_in_models' in wanted or 'num_models_satisfied' in wanted:
                dist, num = heavy_atom_distance_in_models(models, row, params)
            if 'heavy_atom_distance_in_models' in wanted:
                curr_features.extend([dist])
                if i == 0: feature_names.extend(
                    ['heavy_atom_distance_in_models'])

            if 'num_models_satisfied' in wanted:
                curr_features.extend([num])
                if i == 0: feature_names.extend(
                    ['num_models_satisfied'])

            if 'heavy_atom_distance_in_best_top5' in wanted:
                curr_features.extend([heavy_atom_distance_in_top_model(top5_model, row)])
                if i == 0: feature_names.extend(
                    ['heavy_atom_distance_in_best_top5'])

            if 'heavy_atom_distance_in_best_top10' in wanted:
                curr_features.extend([heavy_atom_distance_in_top_model(top10_model, row)])
                if i == 0: feature_names.extend(
                    ['heavy_atom_distance_in_best_top10'])


            # add curr_feature to features and actual label to labels
            features.append(curr_features)
            if with_labels: labels.append(int(row['true_ec']))

        if with_labels:
            return feature_names, features, labels
        else:
            return feature_names, features
