import pandas as pd

from src.modular_inference_methods.interface_prediction.predictor_interface import PredictorInterface
from src.utils.protein_protein_interaction import PPI
from src.analysis_tools.ecs import get_top_ecs
pd.options.mode.chained_assignment = None  # default='warn'


class Top5Predictor(PredictorInterface):
    def __init__(self):
        self.ml = False

    def predict_single_ppi(self, ppp, params) -> PPI:
        ppi = PPI(ppp)
        df_top_ecs = get_top_ecs(ppp, params)
        df_top_ecs['prediction_confidence'] = 0
        df_top_ecs['prediction_confidence'].iloc[:5] = 1
        ppi.df_ecs = df_top_ecs
        return ppi
