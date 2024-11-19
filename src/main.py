from src.utils import timestamp, give_info_on_known_data
from src.modular_inference_methods.utils import create_interaction_predictor, create_interface_predictor
from src.analysis_tools.prediction_evaluation import full_evaluation, full_prediction
from src.dataset.check_dataset_completion import precompute_expensive_files
import multiprocessing as mp
from src.dataset.read_dataset import read_training_dataset
import json
import os
import sys








def main():
    #read input:
    args = sys.argv
    if not len(args)==2:
        print(f'WARNING: only 1 input needed but {len(args)-1} received. Input only absolute or relative filepath to \'params_file.txt\'')
        exit()
    params_file_path = str(args[1])
    if not os.path.isfile(params_file_path):
        print(f'WARNING: input {params_file_path} is no filepath. Input only absolute or relative filepath to \'params_file.txt\'')
        exit()




    # read params file
    with open(params_file_path) as f:
        data = f.readlines()
    json_string=''
    for d in data:
        d = d.strip()
        if not d.startswith('#') and len(d)>0:
            json_string+=d+'\n'
    params = json.loads(json_string)
    if params['n_jobs']=='max': params['n_jobs']=mp.cpu_count()

    predict=params['predict']
    evaluate=params['evaluate']
    print(f'\n\n===============================Read In Datasets; Start: {timestamp.get_timestamp_seconds()}===================================\n\n')
    training_ppps = read_training_dataset(params)
    precompute_expensive_files(training_ppps, params, True)
    print('\ntraining data:')
    _ = give_info_on_known_data(training_ppps)
    if predict:
        test_ppps = read_training_dataset(params)
        precompute_expensive_files(test_ppps, params, False)
        print('\ntest data:')
        _ = give_info_on_known_data(test_ppps)
    print(f'\n\n===============================Read In Datasets; End: {timestamp.get_timestamp_seconds()}===================================\n\n')





    pred_interaction = create_interaction_predictor(params)

    pred_interface = create_interface_predictor(params)

    if evaluate:
        print(f'\n\n===============================Evaluation; Start: {timestamp.get_timestamp_seconds()}===================================\n\n')
        pred_interaction = create_interaction_predictor(params)
        pred_interface = create_interface_predictor(params)
        full_evaluation(pred_interaction,pred_interface, training_ppps, params, out_dir=params['out_dir'])
        print(f'\n\n===============================Evaluation; End: {timestamp.get_timestamp_seconds()}===================================\n\n')

    if predict:
        print(f'\n\n===============================Prediction; Start: {timestamp.get_timestamp_seconds()}===================================\n\n')
        pred_interaction = create_interaction_predictor(params)
        pred_interface = create_interface_predictor(params)
        full_prediction(pred_interaction,pred_interface, training_ppps, test_ppps, params, out_dir=params['out_dir'])
        print(f'\n\n===============================Prediction; End: {timestamp.get_timestamp_seconds()}===================================\n\n')






if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()