import src.modular_inference_methods
from src import modular_inference_methods as mim

def normalize_predictions(df_ecs):
    #TODO is actual normalisation needed? so far it only sets all values above 1 to 1
    if len(df_ecs)>0:
        df_ecs.loc[df_ecs['prediction_confidence'] > 1, 'prediction_confidence'] = 1

def create_interface_predictor(params):
    #returns predictor object dependent on 'predictor_type' of params
    pt = params['interface_predictor_type']
    if pt == 'top5': return mim.BaselineTop5Predictor()
    elif pt == 'additive': return mim.additive_predictor.AdditivePredictor()
    elif pt == 'ml_test': return mim.MlTestPredictor()
    elif pt == 'random_forest': return mim.RandomForestPredictor()
    elif pt == 'mlp': return mim.MLPPredictor()
    elif pt == 'calibrated': return mim.CalibratedPredictor()
    elif pt == 'logistic_regression': return mim.LogisticRegressionPredictor()
    elif pt == 'linear_discriminant': return mim.LinearDiscriminantPredictor()
    elif pt == 'gradient_boosting': return mim.GradientBoostingPredictor()
    elif pt == 'cn': return mim.BaselineCnPredictor()
    elif pt == 'svm': return mim.SVMPredictor()
    elif pt == 'only_negative': return mim.BaselineOnlyNegative()
    else:
        print(f'the predictor {pt} does not exist')
        return None

def create_interaction_predictor(params):
    #returns predictor object dependent on 'predictor_type' of params
    pt = params['interaction_predictor_type']
    if pt == 'gradient_boosting': return mim.interaction_prediction.GradientBoostingPredictor()
    elif pt == 'svm': return mim.interaction_prediction.SVMPredictor()
    elif pt == 'random_forest': return mim.interaction_prediction.RandomForestPredictor()
    elif pt == 'logistic_regression': return mim.interaction_prediction.LogisticRegressionPredictor()
    else:
        print(f'the predictor {pt} does not exist')
        return None