import pandas as pd

from src.modular_inference_methods.interface_prediction.predictor_interface import PredictorInterface
from src.utils.protein_protein_interaction import PPI
from src.analysis_tools.ecs import get_top_ecs
from src.modular_inference_methods.utils import normalize_predictions
pd.options.mode.chained_assignment = None  # default='warn'


class CNPredictor(PredictorInterface):
    def __init__(self):
        self.ml = False

    def predict_single_ppi(self, ppp, params) -> PPI:
        ppi = PPI(ppp)
        df_top_ecs = get_top_ecs(ppp, params)
        df_top_ecs['prediction_confidence'] = df_top_ecs['cn']



        normalize_predictions(df_top_ecs)
        ppi.df_ecs = df_top_ecs
        return ppi
