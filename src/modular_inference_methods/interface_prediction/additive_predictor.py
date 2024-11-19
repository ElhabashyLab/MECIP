import pandas as pd

from src.modular_inference_methods.interface_prediction.predictor_interface import PredictorInterface
from src.utils.protein_protein_interaction import PPI
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.cluster_detection import calculate_clusters
from src.modular_inference_methods.utils import normalize_predictions
pd.options.mode.chained_assignment = None  # default='warn'


class AdditivePredictor(PredictorInterface):
    def __init__(self):
        self.ml = False

    def predict_single_ppi(self, ppp, params) -> PPI:
        ppi = PPI(ppp)
        df_top_ecs = get_top_ecs(ppp, params)
        df_top_ecs['prediction_confidence'] = df_top_ecs['cn']

        #values for clusters
        clusters = calculate_clusters(ppp, params, df_top_ecs)
        for cluster in clusters:
            df_top_ecs.loc[cluster, 'prediction_confidence'] += (1-3/len(cluster))*.1


        normalize_predictions(df_top_ecs)
        ppi.df_ecs = df_top_ecs
        return ppi
