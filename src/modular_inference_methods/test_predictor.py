from src.modular_inference_methods.interface_prediction.predictor_interface import PredictorInterface
from src.utils.protein_protein_interaction import PPI

class TestPredictor(PredictorInterface):
    def predict_single_ppi(self, ppp, params, x) -> PPI:
        print('test')
