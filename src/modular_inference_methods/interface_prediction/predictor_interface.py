from src.utils.protein_protein_interaction import PPI
import pickle
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.ecs import mark_true_ecs
import pandas as pd
import multiprocessing as mp
import time
from pathos.multiprocessing import ProcessingPool as Pool
import os
import src.utils.timestamp as timestamp

def ppp_to_feature_parallelized(ppp, params, with_labels, predictor):
    df_top_ecs = get_top_ecs(ppp, params)
    if df_top_ecs is None:
        print(f'WARNING: still no top ecs found for {ppp.name}. It will be ignored for further predictions')
        return None
    if with_labels:
        df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(), params['TP_EC_distance_threshold_heavy_atoms'], ppp, params)
        feature_names, add_features, add_labels = predictor.ppp_to_features(ppp, df_top_ecs, params, with_labels=True)

        # make ppi object
        ppi = PPI(ppp, df_ecs=df_top_ecs, ap_ecs=add_labels)

        return [feature_names, add_features, add_labels, ppi]
    else:
        feature_names, add_features = predictor.ppp_to_features(ppp, df_top_ecs, params, with_labels=False)
        ppi = PPI(ppp, df_ecs=df_top_ecs)
        return [feature_names, add_features, [],  ppi]

class PredictorInterface:
    def predict_single_ppi(self,ppp, params) -> PPI:
        print('no prediction function made yet')
        pass

    def predict_single_feature_list(self, feature, params) -> int:
        print('this classifier does not work with features')
        pass

    def predict_multiple_ppi(self, ppps, params) -> [PPI]:
        ppis=[]
        for ppp in  ppps:
            ppi = self.predict_single_ppi(ppp, params)
            if ppi: ppis.append(ppi)
        return ppis


    def predict_multiple_feature_list(self, features, params) -> [int]:
        labels=[]
        for feature in features:
            label = self.predict_single_feature_list(feature,params)
            if label: labels.append(label)
        return labels

    def learn(self, features, labels, params) -> None:
        print('no learning method for this predictor needed')
        pass

    def ppp_to_features(self, ppp, df_top_ecs, params, with_labels) -> [int]:
        print('this classifier does not work with features')
        pass

    def ppps_to_features(self, ppps, params, with_labels = False):
        features = []
        labels = []
        ppis = []

        n_ppps = len(ppps)
        if params['use_saved_features'] is None:
            before  = time.time()
            with Pool(params['n_jobs']) as pool:
                results = pool.imap(ppp_to_feature_parallelized,ppps, [params]*n_ppps, [with_labels]*n_ppps, [self]*n_ppps)
            results = list(results)
            print(f'time spent in pool: {time.time() - before}')



        else:
            file_to_read = open(params['use_saved_features']+'/interface_features.pkl',"rb")
            results = pickle.load(file_to_read)
            file_to_read.close()
            print(f'loaded {len(results)} complexes')



        count1=0
        count2=0
        for result, ppp in zip(results, ppps):
            if result is not None:
                if len(result[0])==24: count2+=1
                else: count1+=1
                feature_names = result[0]
                features.extend(result[1])
                labels.extend(result[2])
                ppis.append(result[3])

                # add additional info to ppp object
                ap_results = []
                an_results = []
                for l, (idx, row) in zip(result[2], result[3].df_ecs.iterrows()):
                    if l == 1:
                        ap_results.append([row['i'], row['A_i'], row['j'], row['A_j']])
                    else:
                        an_results.append([row['i'], row['A_i'], row['j'], row['A_j']])
                ppp.ap_ec_ij = ap_results
                ppp.an_ec_ij = an_results
                ppp.num_ap_ecs = sum(result[2])






        from src.analysis_tools.feature_calculation import impute_missing_data
        feature_names, features = impute_missing_data(features, feature_names, params)

        print('features used for interface prediction:')
        print(feature_names)


        if params['save_features'] is not None:
            os.makedirs(params['save_features'], exist_ok=True)
            for result in results: result[0]=feature_names
            file_to_store = open(params['save_features'] + '/interface_features.pkl', "wb")
            pickle.dump(results, file_to_store)
            file_to_store.close()
            combined = [feature + [label] for (feature, label) in zip(features, labels)]
            pd.DataFrame(combined, columns=feature_names + ['labels']).to_csv(params['save_features'] + '/interface_features.csv')
            save_dir=params['save_features']
            print(f'features have been saved to {save_dir}/interface_features.pkl. {timestamp.get_timestamp_seconds()}')



        combined = [feature + [label] for (feature, label) in zip(features, labels)]


        if isinstance(params['save_full_table'], str):
            #explanation here:
            combined = [feature + [label] for (feature,label) in zip(features, labels)]
            combined_feature_names = feature_names+['label']
            max_ecs = max([a[1] for a in combined])
            new_names = ['prefix']
            for i in range(max_ecs+1):
                for n in combined_feature_names[2:]:
                    new_names.append(n + f'_ec_number_{i}')
            complexes = list(set([a[0] for a in combined]))
            num_features = len(combined[0]) - 2
            d = dict([(key, [None] * (num_features * max_ecs)) for key in complexes])
            for f in combined:
                complex = f[0]
                num_ec = f[1] - 1
                feats = f[2:]
                d.get(complex)[num_ec * num_features:num_ec * num_features + num_features] = feats
            d_list = [[k] + v for k, v in d.items()]
            old_df = pd.read_csv(params['save_full_table']).set_index('prefix')
            df = pd.DataFrame(d_list, columns=new_names).set_index('prefix')
            result = pd.concat([old_df, df], axis=1).reset_index()

            result.to_csv(params['save_full_table'])
            save_dir = params['save_full_table']
            print(f'full table has been saved to {save_dir}. {timestamp.get_timestamp_seconds()}')

        # remove name from features (prefix)
        features = [feature[1:] for feature in features]
        feature_names = feature_names[1:]


        #for ppp in ppps:
        #    df_top_ecs = get_top_ecs(ppp, params)
        #    if df_top_ecs is None: continue
        #    if with_labels:
        #        df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(), params['TP_EC_distance_threshold_heavy_atoms'])
        #        add_features, add_labels = self.ppp_to_features(ppp, df_top_ecs, params, with_labels=True)

        #        #make ppi object
        #        ppis.append(PPI(ppp, df_ecs=df_top_ecs, ap_ecs=add_labels))

        #        # add additional info to ppp object
        #        ap_results = []
        #        an_results = []
        #        for l, (idx, row) in zip(add_labels, df_top_ecs.iterrows()):
        #            if l == 1:
        #                ap_results.append([row['i'], row['A_i'], row['j'], row['A_j']])
        #            else:
        #                an_results.append([row['i'], row['A_i'], row['j'], row['A_j']])
        #        ppp.ap_ec_ij = ap_results
        #        ppp.an_ec_ij = an_results
        #        ppp.num_ap_ecs = sum(add_labels)

        #        labels.extend(add_labels)
        #    else:
        #        add_features = self.ppp_to_features(ppp, df_top_ecs, params, with_labels=False)
        #    features.extend(add_features)
        if with_labels: return feature_names, features, labels, ppis
        else: return feature_names, features, ppis

    def save_classifier(self, out_file) -> None:
        if self.ml:
            if hasattr(self, 'clf'):
                with open(out_file, 'wb') as f:
                    pickle.dump(self.clf, f)
            else:
                print('this ml-classifier was not constructed yet. Try to first use the \'learn\' function.')
        else:
            print('this classifier can not be saved')


    def load_classifier(self, in_file) -> None:
        if self.ml:
            with open(in_file) as f:
                self.clf = pickle.load(f)
            if not isinstance(self.clf, PredictorInterface):
                print(f'something went wrong with loading file {in_file}')
        else:
            print('this classifier can not be loaded')

