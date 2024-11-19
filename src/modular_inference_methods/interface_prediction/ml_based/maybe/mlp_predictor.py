import pandas as pd
import sklearn.svm as svm
from sklearn.model_selection import GridSearchCV
from src.modular_inference_methods.interface_prediction.predictor_interface import PredictorInterface
from src.utils.protein_protein_interaction import PPI
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.cluster_detection import calculate_clusters
from sklearn.neural_network import MLPClassifier

pd.options.mode.chained_assignment = None  # default='warn'


class MLPPredictor(PredictorInterface):
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
        # parameter space for SVM
        parameter_space = {'kernel': ['linear', 'rbf'], 'C': [0.1, 10, 1]}

        # SVM
        self.clf = GridSearchCV(svm.SVC(probability=True), parameter_space, n_jobs=1,
                           verbose=10, cv=3)
        self.clf = MLPClassifier()
        # work on feature space:
        self.clf.fit(features,labels)


        print(self.clf.score(features, labels))
        print(self.clf.get_params())

    def ppp_to_features(self, ppp, df_top_ecs, params, with_labels=False):
        features=[]
        labels=[]
        kurt, skew = ppp.calc_kurtosis_and_skewness(exclude_outside_pdb=params['remove_ecs_outside_of_monomer_pdb'])
        clusters = calculate_clusters(ppp, params, df_top_ecs)
        for idx, row in df_top_ecs.iterrows():
            in_cluster=0
            cluster_size=0
            for cluster in clusters:
                if row.name in cluster:
                    in_cluster=1
                    cluster_size = len(cluster)


            curr_features = []
            curr_features.append(row['cn'])
            curr_features.append(kurt)
            curr_features.append(skew)
            if True:
                curr_features.append(len(df_top_ecs.index))
                curr_features.append(int(row.name))
                curr_features.append(max(df_top_ecs['cn']))
                curr_features.append(in_cluster)
                curr_features.append(cluster_size)
                curr_features.append(len(clusters))





            # add curr_feature to features and actual label to labels
            features.append(curr_features)
            if with_labels: labels.append(int(row['true_ec']))
        if with_labels: return features, labels
        else: return features