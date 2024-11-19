import copy

import pandas as pd
from src.dataset.test_set import add_ap_ec_information
import numpy as np
from src.visualising.prediction_evaluation import draw_precision_recall_curve, draw_roc_curve, plot_num_ap_ec_per_ppp, plot_num_ap_ec_per_ppi, plot_involved_aa, plot_tp_ecs_per_complex, plot_interaction_prediction, plot_feature_importances, plot_ppp_removal_history, plot_EC_removal_history, plot_num_of_interface_predictions
import os
from src.utils import timestamp, give_info_on_known_data
from src.utils.misc import give_info_on_params
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from src.visualising.relation_plots import *
import pickle
from src.visualising.skew_kurt import draw_kurt_skew_plot
from src.dataset.test_set import print_info_on_training_set

def evaluate_interface_prediction(ppi, params):
    # returns 4 values: TP, FP, FN, TN
    ap_ec_df = ppi.get_ap_ec_df()
    if not isinstance(ap_ec_df, pd.DataFrame):
        print(f'No evaluation possible, since no TP are known for complex {ppi.ppp.name}')
        return 0,0,0, 0
    if not 'prediction_confidence' in ppi.df_ecs.columns:
        print(f'No evaluation possible, since no prediction was done for complex {ppi.ppp.name}')
        return 0, 0, 0, 0
    predicted = list(get_predicted_ECs(ppi, params).index)
    not_predicted = list(ppi.df_ecs[ppi.df_ecs['prediction_confidence'] < params['confidence_threshold_residue_level']].index)
    ap = list(ap_ec_df.index)

    tp = list(set(ap).intersection(predicted))
    fp = list(set(predicted) - set(tp)) + list(set(tp) - set(predicted))
    fn = list(set(ap) - set(tp)) + list(set(tp) - set(ap))
    tn = [i for i in not_predicted if i not in ap]
    tp.sort()
    fp.sort()
    fn.sort()
    tn.sort()


    #print(tp)
    #print(fp)
    #print(fn)
    return len(tp), len(fp), len(fn), len(tn)











def evaluate_learning_predictor_interface(predictor, ppps, params, out_dir=None):
    #evaluates the performance of the machine learning prediction of interfaces
    # returns :
    #   a Dataframe containing all metrics as well as the set of predicted ppis

    #define kfold
    n_outer_cv = params['n_outer_cv_interface']
    kfold = StratifiedKFold(n_outer_cv, shuffle=True)
    #kfold = KFold(n_outer_cv, shuffle=True)

    confidence_threshold_residue_levels = np.arange(-.01, 1.01, .01)

    condition_dict = {}

    metrics = []

    out = f'number of checked PPI: {len(ppps)} (times {n_outer_cv} due to outer cross validation)\n\n'
    out += f'starting time: {timestamp.get_timestamp_seconds()}\n\n'
    print('converting input files into usable features...')
    feature_names, features, labels, ppis = predictor.ppps_to_features(ppps,params,with_labels=True)

    # create pairplot
    if params['plot_feature_pairplot']:
        print(f'\tfeature pairplot... ({timestamp.get_timestamp_seconds()})')
        plot_pairplot(features, labels, feature_names, params, filepath_out=out_dir + '/feature_pairplot.png')

    num_ap = labels.count(1)
    num_an = labels.count(0)
    out += f'number of actual positive labels in dataset: {num_ap} (number of considered residue pairs that have a heavy atom distance smaller than the given threshold)\n'
    out += f'number of actual negative labels in dataset: {num_an}\n\n'
    if num_an>num_ap*1.5:
        out += 'WARNING: the number of actual negative labels is significantly (more than 150%) bigger than the number of actual positive labels\n'
        out += '         This can lead to severe performance problems when training a machine learning model on this data\n\n'
    if num_ap > num_an * 1.5:
        out += 'WARNING: the number of actual positive labels is significantly (more than 150%) bigger than the number of actual negative labels\n'
        out += '         This can lead to severe performance problems when training a machine learning model on this data\n\n'

    final_predictions = [0] * len(features)
    summed_feature_importances = [0]*len(feature_names)
    best_hyperparameters = []
    for i, (train, test) in enumerate(kfold.split(features, labels)):
        print(f'start of cross validation loop {i+1}/{n_outer_cv}: {timestamp.get_timestamp_seconds()}')
        train_features = [features[i] for i in train]
        test_features = [features[i] for i in test]
        train_labels = [labels[i] for i in train]
        test_labels = [labels[i] for i in test]

        #random sampling:
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import RandomOverSampler
        resampler = params['resample_interface_prediction_data']
        if resampler is not None:
            no_resampling=False
            if  resampler == 'RandomUnderSampler' : rs = RandomUnderSampler(random_state=42)
            elif resampler == 'RandomOverSampler' : rs = RandomOverSampler(random_state=42)
            else:
                print(f'no random sub-sampler available by the name {resampler} so none was used')
                features_res, labels_res = train_features, train_labels
                no_resampling=True
            if not no_resampling:
                features_res, labels_res = rs.fit_resample(train_features, train_labels)
                print(f'{type(rs)} changes training set from {train_labels.count(True)}(true), {train_labels.count(False)}(false), to '
                      f'{labels_res.count(True)}(true), {labels_res.count(False)}(false)')
        else:
            features_res, labels_res = train_features, train_labels


        predictor.learn(features_res,labels_res,params)

        # save best hyperparameters
        from sklearn.model_selection import GridSearchCV
        if isinstance(predictor.clf, GridSearchCV):
            best_hyperparameters.append((predictor.clf.best_params_, predictor.clf.best_score_))


        #look at feature importances:
        #first only single iterations
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        if isinstance(predictor.clf, GridSearchCV):
            clf = predictor.clf.best_estimator_
        else:
            clf = predictor.clf
        if not isinstance(clf,SVC) and not isinstance(clf,LogisticRegression):
            feature_importances = clf.feature_importances_
            title = f'feature importance analysis for interface prediction, iteration {i}'
            filepath_out = out_dir+f'/feature_importances/feature_importance_{i}.png'
            os.makedirs(out_dir + '/feature_importances/', exist_ok=True)
            plot_feature_importances(feature_importances, feature_names,title, filepath_out=filepath_out)

            #sum up all feature importances for average at the end
            summed_feature_importances = [x1 + y1 for x1, y1 in zip(summed_feature_importances, feature_importances)]

        import random
        predictions = predictor.predict_multiple_feature_list(test_features, params)
        for p, index in zip(predictions, test): final_predictions[index] = p
        for confidence_threshold_residue_level in confidence_threshold_residue_levels:
            binary_predictions = list(map(int,predictions>confidence_threshold_residue_level))
            zipped = list(zip(binary_predictions, test_labels))

            tp = condition_dict.get(confidence_threshold_residue_level, [0, 0, 0, 0])[0] + zipped.count((1,1))
            fp = condition_dict.get(confidence_threshold_residue_level, [0, 0, 0, 0])[1] + zipped.count((1,0))
            fn = condition_dict.get(confidence_threshold_residue_level, [0, 0, 0, 0])[2] + zipped.count((0,1))
            tn = condition_dict.get(confidence_threshold_residue_level, [0, 0, 0, 0])[3] + zipped.count((0,0))

            condition_dict[confidence_threshold_residue_level] = [tp, fp, fn, tn]

    # output best hyperparameters:
    if len(best_hyperparameters)>0:
        hyper_out = f'best hyperparameters for all splits: ml model: {type(predictor.clf.best_estimator_)}\n\n'
        for i,best_hyper in enumerate(best_hyperparameters):
            hyper_out+=f'split {i}\n\thyperparameters: {best_hyper[0]}\n\tscore:{best_hyper[1]}\n\n'
        with open(out_dir+'/best_hyperparameters.txt', 'w') as f:
            f.write(hyper_out)


    # look at feature importances:
    # average over all splits
    if sum(summed_feature_importances)>0:
        feature_importances = [a/n_outer_cv for a in summed_feature_importances]
        title = f'feature importance analysis for interface prediction'
        filepath_out = out_dir + f'/feature_importances/feature_importance_average.png'
        plot_feature_importances(feature_importances, feature_names, title, filepath_out=filepath_out)

    #update ppis with predictions:
    print(f'update ppi predictions... ({timestamp.get_timestamp_seconds()})')
    curr_pointer = 0
    for ppi in ppis:
        df_size = len(ppi.df_ecs)
        curr_predictions = final_predictions[curr_pointer:curr_pointer+df_size]
        curr_pointer += df_size
        ppi.df_ecs['prediction_confidence'] = curr_predictions
    max_accuracy = 0

    #set confidence_threshold_residue_level to None if confidence_threshold_residue_level_recall is defined
    if params['confidence_threshold_residue_level_recall'] is not None:
        params['confidence_threshold_residue_level'] = None

    for confidence_threshold_residue_level in confidence_threshold_residue_levels:
        tp = condition_dict.get(confidence_threshold_residue_level, [0, 0, 0, 0])[0]
        fp = condition_dict.get(confidence_threshold_residue_level, [0, 0, 0, 0])[1]
        fn = condition_dict.get(confidence_threshold_residue_level, [0, 0, 0, 0])[2]
        tn = condition_dict.get(confidence_threshold_residue_level, [0, 0, 0, 0])[3]
        # print information
        pre = precision(tp,fp)
        rec = recall(tp,fn)
        fal = false_positive_rate(fp,tn)
        acc = accuracy(tp, fp, tn, fn)

        #set new threshold if confidence_threshold_residue_level_recall is defined:
        if params['confidence_threshold_residue_level_recall'] is not None and params['confidence_threshold_residue_level'] is None and rec<=params['confidence_threshold_residue_level_recall']:
            params['confidence_threshold_residue_level']=confidence_threshold_residue_level


        out += f'\tconfidence_threshold_residue_level: {confidence_threshold_residue_level}\n'
        out += f'\t\ttp: {tp}\n'
        out += f'\t\tfp: {fp}\n'
        out += f'\t\tfn: {fn}\n'
        out += f'\t\ttn: {tn}\n\n'
        out += f'\t\tprecision:           {pre}\n'
        out += f'\t\trecall:              {rec}\n'
        out += f'\t\tfalse_positive_rate: {fal}\n'
        out += f'\t\taccuracy:            {acc}\n'
        if params['confidence_threshold_residue_level'] is not None and round(confidence_threshold_residue_level*100)/100 == round(params['confidence_threshold_residue_level']*100)/100:
            actual_confidence_metrics = [pre, rec, fal, acc]
        if acc>max_accuracy:
            max_acc_confidence_threshold_residue_level = confidence_threshold_residue_level
            max_accuracy=acc
            max_acc_metrics = [pre, rec, fal, acc]



        metrics.append([confidence_threshold_residue_level, pre, rec, fal, acc])

    out += f'end time: {timestamp.get_timestamp_seconds()}\n\n'

    #add ap info
    #TODO is this even needed?
    #add_ap_ec_information(ppis, params)


    #create a file showing comparison between model and baseline
    if params['interface_with_all_negative_baseline'] and params['confidence_threshold_residue_level']:
        baseline_metrics = [None, 0, 0, accuracy(0,num_ap,num_an,0)]
        conf_thresh = round(params['confidence_threshold_residue_level']*100)/100
        baseline_out=f'used baseline: predict everything to be no interaction interface\nbaseline metrics:\n' \
                     f'\tprecision:           {baseline_metrics[0]}\n' \
                     f'\trecall:              {baseline_metrics[1]}\n' \
                     f'\tfalse_positive_rate: {baseline_metrics[2]}\n' \
                     f'\taccuracy:            {baseline_metrics[3]}\n\n' \
                     f'metrics of confidence threshold with the highest accuracy: {max_acc_confidence_threshold_residue_level}:\n' \
                     f'\tprecision:           {max_acc_metrics[0]}\n' \
                     f'\trecall:              {max_acc_metrics[1]}\n' \
                     f'\tfalse_positive_rate: {max_acc_metrics[2]}\n' \
                     f'\taccuracy:            {max_acc_metrics[3]}\n\n'
        try:
            baseline_out+=f'metrics of given confidence threshold (rounded): {conf_thresh}:\n' \
                         f'\tprecision:           {actual_confidence_metrics[0]}\n' \
                         f'\trecall:              {actual_confidence_metrics[1]}\n' \
                         f'\tfalse_positive_rate: {actual_confidence_metrics[2]}\n' \
                         f'\taccuracy:            {actual_confidence_metrics[3]}\n\n'
        except UnboundLocalError:
            baseline_out+=f'metrics of given confidence threshold: {conf_thresh}:\n' \
                          f'\t this confidence threshold can not be computed. It has to be a multiple of 0.01'

        with open(out_dir+'/baseline_comparison.txt', 'w') as f:
            f.write(baseline_out)

    if params['confidence_threshold_residue_level'] is None:
        params['confidence_threshold_residue_level']=.5


    if out_dir:
        with open(out_dir+'/predictor_eval.txt', 'w') as f:
            f.write(out)
    else:
        print(out)

    return pd.DataFrame(metrics, columns=['threshold', 'precision','recall','false_positive_rate', 'accuracy']).set_index('threshold'), ppis


def evaluate_learning_predictor_interaction(predictor, ppps, params, out_dir=None):
    # evaluates the performance of the machine learning prediction of interactions
    # returns a Dataframe containing all metrics as well as the final_predictions, labels, and features

    # define kfold
    n_outer_cv = params['n_outer_cv_interaction']
    kfold = StratifiedKFold(n_outer_cv, shuffle=True)
    #kfold = KFold(n_outer_cv, shuffle=True)

    confidence_threshold_ppis = np.arange(-.01, 1.01, .01)

    condition_dict = {}

    metrics = []

    out = f'number of checked PPI: {len(ppps)} (times {n_outer_cv} due to outer cross validation)\n\n'
    out += f'starting time: {timestamp.get_timestamp_seconds()}\n\n'
    print('converting input files into usable features...')
    feature_names, features, labels = predictor.ppps_to_features(ppps, params, with_labels=True)
    length_before_removal = len(ppps)
    features2 = [x for (x,y) in zip(features,labels) if x and isinstance(y, bool)]
    labels = [y for (x,y) in zip(features,labels) if x and isinstance(y, bool)]
    features = features2

    #create pairplot
    if params['plot_feature_pairplot']:
        print(f'\tfeature pairplot... ({timestamp.get_timestamp_seconds()})')
        plot_pairplot(features,labels, feature_names, params, filepath_out=out_dir + '/feature_pairplot.png')


    num_removed = length_before_removal - len(labels)
    out += f'number of removed complexes: {num_removed}\n\t(removal due to several reasons: no interface file present, ' \
           f'no ec file present, no points in ec file present (after removing all that are located outside of the ' \
           f'residue ranges given by the inputted pdb files) This removal can lead to errors in the following prediction\n\n'
    num_ap = labels.count(1)
    num_an = labels.count(0)
    out += f'number of actual positive labels in dataset: {num_ap} (number of complexes that have enough interacting residue pairs (see params))\n'
    out += f'number of actual negative labels in dataset: {num_an}\n\n'
    if num_an > num_ap * 1.5:
        out += 'WARNING: the number of actual negative labels is significantly (more than 150%) bigger than the number of actual positive labels\n'
        out += '         This can lead to severe performance problems when training a machine learning model on this data\n\n'
    if num_ap > num_an * 1.5:
        out += 'WARNING: the number of actual positive labels is significantly (more than 150%) bigger than the number of actual negative labels\n'
        out += '         This can lead to severe performance problems when training a machine learning model on this data\n\n'
    final_predictions = [0]*len(features)
    summed_feature_importances = [0]*len(feature_names)
    best_hyperparameters = []
    for i, (train, test) in enumerate(kfold.split(features, labels)):
        print(f'start of cross validation loop {i + 1}/{n_outer_cv}: {timestamp.get_timestamp_seconds()}')
        train_features = [features[i] for i in train]
        test_features = [features[i] for i in test]
        train_labels = [labels[i] for i in train]
        test_labels = [labels[i] for i in test]


        #random sampling:
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import RandomOverSampler
        resampler = params['resample_interaction_prediction_data']
        if resampler is not None:
            no_resampling=False
            if  resampler == 'RandomUnderSampler' : rs = RandomUnderSampler(random_state=42)
            elif resampler == 'RandomOverSampler' : rs = RandomOverSampler(random_state=42)
            else:
                print(f'no random sub-sampler available by the name {resampler} so none was used')
                features_res, labels_res = train_features, train_labels
                no_resampling=True
            if not no_resampling:
                features_res, labels_res = rs.fit_resample(train_features, train_labels)
                print(f'{type(rs)} changes training set from {train_labels.count(True)}(true), {train_labels.count(False)}(false), to '
                      f'{labels_res.count(True)}(true), {labels_res.count(False)}(false)')
        else:
            features_res, labels_res = train_features, train_labels


        predictor.learn(features_res, labels_res, params)


        #save best hyperparameters
        from sklearn.model_selection import GridSearchCV
        if isinstance(predictor.clf, GridSearchCV):
            best_hyperparameters.append((predictor.clf.best_params_,predictor.clf.best_score_))


        #look at feature importances:
        #first only single iterations
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        if isinstance(predictor.clf, GridSearchCV):
            clf = predictor.clf.best_estimator_
        else:
            clf = predictor.clf
        if not isinstance(clf,SVC) and not isinstance(clf,LogisticRegression):
            feature_importances = clf.feature_importances_
            title = f'feature importance analysis for interaction prediction, iteration {i}'
            filepath_out = out_dir+f'/feature_importances/feature_importance_{i}.png'
            os.makedirs(out_dir + '/feature_importances/', exist_ok=True)
            plot_feature_importances(feature_importances, feature_names,title, filepath_out=filepath_out)

            #sum up all feature importances for average at the end
            summed_feature_importances = [x1 + y1 for x1, y1 in zip(summed_feature_importances, feature_importances)]



        predictions = predictor.predict_multiple_feature_list(test_features, params)
        for p,index in zip(predictions,test): final_predictions[index]=p
        for confidence_threshold_ppi in confidence_threshold_ppis:
            binary_predictions = list(map(int, predictions > confidence_threshold_ppi))
            zipped = list(zip(binary_predictions, test_labels))

            tp = condition_dict.get(confidence_threshold_ppi, [0, 0, 0, 0])[0] + zipped.count((1, 1))
            fp = condition_dict.get(confidence_threshold_ppi, [0, 0, 0, 0])[1] + zipped.count((1, 0))
            fn = condition_dict.get(confidence_threshold_ppi, [0, 0, 0, 0])[2] + zipped.count((0, 1))
            tn = condition_dict.get(confidence_threshold_ppi, [0, 0, 0, 0])[3] + zipped.count((0, 0))

            condition_dict[confidence_threshold_ppi] = [tp, fp, fn, tn]

    # output best hyperparameters:
    if len(best_hyperparameters)>0:
        hyper_out = f'best hyperparameters for all splits: ml model: {type(predictor.clf.best_estimator_)}\n\n'
        for i,best_hyper in enumerate(best_hyperparameters):
            hyper_out+=f'split {i}\n\thyperparameters: {best_hyper[0]}\n\tscore:{best_hyper[1]}\n\n'
        with open(out_dir+'/best_hyperparameters.txt', 'w') as f:
            f.write(hyper_out)


    # look at feature importances:
    # average over all splits
    if sum(summed_feature_importances) > 0:
        feature_importances = [a/n_outer_cv for a in summed_feature_importances]
        title = f'feature importance analysis for interaction prediction'
        filepath_out = out_dir + f'/feature_importances/feature_importance_average.png'
        plot_feature_importances(feature_importances, feature_names, title, filepath_out=filepath_out)

    """
    #testing
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    testy = [int(x) for x in labels]
    ns_probs = [0 for _ in range(len(testy))]
    lr_probs = final_predictions
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    from sklearn.metrics import precision_recall_curve
    # predict class values
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    # summarize scores
    print('Logistic: auc=%.3f' % (lr_auc))
    # plot the precision-recall curves
    #no_skill = len(testy[testy == 1]) / len(testy)
    #plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    """

    # set confidence_threshold_residue_level to None if confidence_threshold_residue_level_recall is defined
    if params['confidence_threshold_ppi_recall'] is not None:
        params['confidence_threshold_ppi'] = None

    for confidence_threshold_ppi in confidence_threshold_ppis:
        tp = condition_dict.get(confidence_threshold_ppi, [0, 0, 0, 0])[0]
        fp = condition_dict.get(confidence_threshold_ppi, [0, 0, 0, 0])[1]
        fn = condition_dict.get(confidence_threshold_ppi, [0, 0, 0, 0])[2]
        tn = condition_dict.get(confidence_threshold_ppi, [0, 0, 0, 0])[3]
        # print information
        pre = precision(tp, fp)
        rec = recall(tp, fn)
        fal = false_positive_rate(fp, tn)
        acc = accuracy(tp, fp, tn, fn)

        # set new threshold if confidence_threshold_residue_level_recall is defined:
        if params['confidence_threshold_ppi_recall'] is not None and params[
            'confidence_threshold_ppi'] is None and rec <= params[
            'confidence_threshold_ppi_recall']:
            params['confidence_threshold_ppi'] = confidence_threshold_ppi

        out += f'\tconfidence_threshold_residue_level: {confidence_threshold_ppi}\n'
        out += f'\t\ttp: {tp}\n'
        out += f'\t\tfp: {fp}\n'
        out += f'\t\tfn: {fn}\n'
        out += f'\t\ttn: {tn}\n\n'
        out += f'\t\tprecision:           {pre}\n'
        out += f'\t\trecall:              {rec}\n'
        out += f'\t\tfalse_positive_rate: {fal}\n'
        out += f'\t\taccuracy:            {acc}\n'

        metrics.append(
            [confidence_threshold_ppi, precision(tp, fp), recall(tp, fn), false_positive_rate(fp, tn),
             accuracy(tp, fp, tn, fn)])
    out += f'end time: {timestamp.get_timestamp_seconds()}\n\n'
    if out_dir:
        with open(out_dir+'/predictor_eval.txt', 'w') as f:
            f.write(out)
    else:
        print(out)

    return pd.DataFrame(metrics,
                        columns=['threshold', 'precision', 'recall', 'false_positive_rate', 'accuracy']).set_index(
        'threshold'), final_predictions, labels, features


def evaluate_predictor_interface(predictor, ppps, params, out_file=None):
    # evaluates the performance of the basic prediction of interfaces
    # returns a Dataframe containing all metrics

    ppis = predictor.predict_multiple_ppi(ppps, params)
    out = f'number of checked PPI: {len(ppis)}\n\n'
    out += f'starting time: {timestamp.get_timestamp_seconds()}\n\n'
    confidence_threshold_residue_levels=np.arange(-.01,1.01,.01)
    metrics=[]
    add_ap_ec_information(ppis, params)
    out += 'interface prediction:\n'
    for confidence_threshold_residue_level in confidence_threshold_residue_levels:
        params['confidence_threshold_residue_level'] = confidence_threshold_residue_level
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for ppi in ppis:
            tp_add, fp_add, fn_add, tn_add = evaluate_interface_prediction(ppi, params)
            tp += tp_add
            fp += fp_add
            fn += fn_add
            tn += tn_add
        # print information
        out += f'\tconfidence_threshold_residue_level: {confidence_threshold_residue_level}\n'
        out += f'\t\ttp: {tp}\n'
        out += f'\t\tfp: {fp}\n'
        out += f'\t\tfn: {fn}\n'
        out += f'\t\ttn: {tn}\n\n'
        out += f'\t\tprecision:           {precision(tp,fp)}\n'
        out += f'\t\trecall:              {recall(tp,fn)}\n'
        out += f'\t\tfalse_positive_rate: {false_positive_rate(fp,tn)}\n'
        out += f'\t\taccuracy:            {accuracy(tp, fp, tn, fn)}\n'



        metrics.append([confidence_threshold_residue_level, precision(tp,fp), recall(tp,fn), false_positive_rate(fp,tn),
                        accuracy(tp, fp, tn, fn)])

    out += f'end time: {timestamp.get_timestamp_seconds()}\n\n'
    if out_file:
        with open(out_file, 'w') as f:
            f.write(out)
    else:
        print(out)

    return pd.DataFrame(metrics, columns=['threshold', 'precision','recall','false_positive_rate', 'accuracy']).set_index('threshold')


def evaluate_predictor_interaction(predictor, ppps, params, out_file=None):
    # evaluates the performance of the basic prediction of interactions
    # returns a Dataframe containing all metrics
    #TODO
    pass


def precision(tp, fp):
    if (tp+fp)==0: return 1
    return tp/(tp+fp)

def recall(tp, fn):
    # true positive rate, sensitivity
    if (tp + fn) == 0: return 1
    return tp/(tp+fn)

def false_positive_rate(fp, tn):
    if (tn + fp) == 0: return 1
    return fp/(fp+tn)

def accuracy(tp, fp, tn, fn):
    if (tn + fp) == 0: return 1
    return (tp + tn)  / (fp + tn + tp + fn)


def get_predicted_ECs(ppi, params):
    #returns df with all predicted Ecs
    return ppi.df_ecs[ppi.df_ecs['prediction_confidence'] >= params['confidence_threshold_residue_level']]

def full_evaluation_interface(predictor, ppps, params, out_dir):
    #creates folder in out_dir with all information: on dataset, on predictor, with plots, ...
    # returns predicted ppis
    # use this function to make a full evaluation of only the interface prediction with any predictor
    if not out_dir.endswith('/'): out_dir+='/'
    folder_name = params['interface_predictor_type']+'_'+timestamp.get_timestamp()
    os.makedirs(out_dir+folder_name, exist_ok=True)

    give_info_on_known_data(ppps, out_file=out_dir+folder_name+'/dataset_info.txt')
    give_info_on_params(params, out_file=out_dir+folder_name+'/params_info.txt')
    if predictor.ml:
        metrics,  ppis= evaluate_learning_predictor_interface(predictor, ppps, params, out_dir=out_dir+folder_name)
    else:
        metrics = evaluate_predictor_interface(predictor, ppps, params, out_file=out_dir+folder_name+'/predictor_eval.txt')
    print(f'\tdraw_precision_recall_curve... ({timestamp.get_timestamp_seconds()})')
    draw_precision_recall_curve(metrics, params['confidence_threshold_residue_level'], filepath_out=out_dir+folder_name+'/precision_recall_curve.png')
    print(f'\tdraw_roc_curve... ({timestamp.get_timestamp_seconds()})')
    draw_roc_curve(metrics,params['confidence_threshold_residue_level'], filepath_out=out_dir+folder_name+'/roc_curve.png')





    return ppis


def full_evaluation_interaction(predictor, ppps, params, out_dir):
    #creates folder in out_dir with all information: on dataset, on predictor
    # returns predictions as well as labels and features for that given prediction
    # use this function to make a full evaluation of only the interaction prediction with any predictor
    if not out_dir.endswith('/'): out_dir+='/'
    folder_name = params['interaction_predictor_type']+'_'+timestamp.get_timestamp()
    os.makedirs(out_dir+folder_name, exist_ok=True)

    give_info_on_known_data(ppps, out_file=out_dir+folder_name+'/dataset_info.txt')
    give_info_on_params(params, out_file=out_dir+folder_name+'/params_info.txt')
    if predictor.ml:
        metrics, predictions, labels, features = evaluate_learning_predictor_interaction(predictor, ppps, params, out_dir=out_dir+folder_name)
    else:
        metrics = evaluate_predictor_interaction(predictor, ppps, params, out_file=out_dir+folder_name+'/predictor_eval.txt')
    print(f'\tdraw_precision_recall_curve... ({timestamp.get_timestamp_seconds()})')
    draw_precision_recall_curve(metrics, params['confidence_threshold_ppi'], filepath_out=out_dir+folder_name+'/precision_recall_curve.png')
    print(f'\tdraw_roc_curve... ({timestamp.get_timestamp_seconds()})')
    draw_roc_curve(metrics,params['confidence_threshold_ppi'], filepath_out=out_dir+folder_name+'/roc_curve.png')

    return predictions, labels, features





def full_evaluation(interaction_predictor, interface_predictor, ppps, params, out_dir):

    # creates folder in out_dir with all information
    # computes all plots possible, runs evalluation of interface as well as interaction prediction
    # runs full pipeline with iterations as specified in params
    # use this to fully evaluate the performance of your predictors on your dataset
    # given all params the out_dir that is created can get quite memory expensive



    if not out_dir.endswith('/'): out_dir+='/'
    folder_name = timestamp.get_timestamp()
    os.makedirs(out_dir+folder_name, exist_ok=True)
    out_dir = out_dir+folder_name+'/'

    if params['save_full_table']: params['save_full_table']= out_dir+'full_info.csv'

    if params['plot_expensive_plots']:
        print(f'print_info_on_training_set... ({timestamp.get_timestamp_seconds()})')
        print_info_on_training_set(ppps, params, out_path=out_dir+'training_set_info.txt')

    interaction_prediction, interaction_labels, interaction_features = full_evaluation_interaction(interaction_predictor, ppps, params, out_dir+'interaction_prediction')

    #remove ppps beneath certain predction value:
    import copy
    all_ppps = copy.deepcopy(ppps)
    interaction_predictions_before_removal = (copy.deepcopy(interaction_prediction), copy.deepcopy(interaction_labels))
    ppps = [ppp for (ppp,pred) in zip(ppps, interaction_prediction) if pred>params['confidence_threshold_ppi']]
    interaction_labels = [lab for (lab,pred) in zip(interaction_labels, interaction_prediction) if pred>params['confidence_threshold_ppi']]
    interaction_prediction = [pred for pred in interaction_prediction if pred>params['confidence_threshold_ppi']]
    ppis = full_evaluation_interface(interface_predictor, ppps, params, out_dir+'interface_prediction')
    # finish ppis
    print(f'finish interactions... ({timestamp.get_timestamp_seconds()})')
    for pred, label, ppi in zip (interaction_prediction, interaction_labels, ppis):
        ppi.interaction_confidence = pred
        ppi.is_interacting = bool(label)

    #delete later
    plot_ppp_removal_history(interaction_predictions_before_removal, ppis, params)
    if params['plot_expensive_plots']: plot_EC_removal_history(interaction_predictions_before_removal, all_ppps, ppis, params)



    #general plots
    print(f'generating plots... ({timestamp.get_timestamp_seconds()})')
    out_dir_general_plots = out_dir + f'general_plots_{timestamp.get_timestamp_seconds()}/'
    os.makedirs(out_dir_general_plots, exist_ok=True)
    plot_general_plots(out_dir_general_plots, ppis, ppps, all_ppps, interaction_predictions_before_removal, params)



    #complex_specific_plots
    if params['plot_complex_details']:
        print(f'generating complex specific plots... ({timestamp.get_timestamp_seconds()})')
        out_dir_complex_plots = out_dir + f'complex_plots_{timestamp.get_timestamp_seconds()}/'
        os.makedirs(out_dir_complex_plots, exist_ok=True)
        plot_complex_details(out_dir_complex_plots, ppis, params)

    #write predictions so far
    print(f'writing predictions... ({timestamp.get_timestamp_seconds()})')
    out_dir_predictions = out_dir + f'predictions/'
    os.makedirs(out_dir_predictions, exist_ok=True)
    out_dir_predictions = out_dir_predictions + f'{timestamp.get_timestamp_seconds()}/'
    os.makedirs(out_dir_predictions, exist_ok=True)
    write_predictions(ppis, out_dir_predictions)

    #write restraint files for possible docking step based on predictions
    from src.haddock_code.haddock_generation import write_dist_restraints_file, run_haddock_with_params
    if params['generate_distance_restraints'] or params['run_haddock']:
        print(f'writing restraint files... ({timestamp.get_timestamp_seconds()})')
        dist_restraint_path = out_dir+'distance_restraints/'
        os.makedirs(dist_restraint_path, exist_ok=True)

        # check for complexes with 1 or 2 EC predictions and give informations on them
        write_info_on_1_and_2_predicted_EC_complexes(ppis, params, out_file_path=dist_restraint_path+'complex_info.txt')

        for ppi in ppis:
            ca_s = [[row['i'], row['j']] for (idx,row) in ppi.df_ecs.iterrows() if row['prediction_confidence']>params['confidence_threshold_residue_level']]
            if len(ca_s)>0:




                if params['distance_restraints_with_monomer_pdbs']:
                    dirpath_complicated = dist_restraint_path + ppi.ppp.name + '_' + ppi.ppp.protein1.uid.replace('_HUMAN','')+ '_' + ppi.ppp.protein2.uid.replace('_HUMAN','') + '/haddock_ecs/'
                    os.makedirs(dirpath_complicated, exist_ok=True)
                write_dist_restraints_file(ca_s, f'{dirpath_complicated}{ppi.ppp.name}_restraints.tbl', params, ppi)

    #run haddock
    if params['run_haddock']:
        print(f'starting HADDOCK... ({timestamp.get_timestamp_seconds()})')
        run_haddock_with_params(ppis, params, out_dir)




def plot_general_plots(out_dir, ppis, ppps, all_ppps, interaction_predictions_before_removal, params):
    #plots all plots that can be done for the dataset in general.
    # no plots here refer to single complexes, but rather only the whole dataset and the made predictions

    # skew_kurt plot preparations
    x = []
    for ppi in ppis:
        x.append([ppi.ppp.name, ppi.is_interacting])
    hue = pd.DataFrame(x, columns=['ppp_name', 'is_interacting']).set_index('ppp_name')
    print(f'\tkurt_skew plot... ({timestamp.get_timestamp_seconds()})')
    draw_kurt_skew_plot([ppi.ppp for ppi in ppis], params, additional_hue=hue,
                        filepath_out=out_dir + 'skew_kurt_plot_interacting.png')

    x = []
    for ppi in ppis:
        text = 'not interacting'
        if sum(ppi.ap_ecs) > 0: text = 'interacting'
        if sum(ppi.ap_ecs) > 2: text = 'interacting and detectable'
        x.append([ppi.ppp.name, text])
    hue = pd.DataFrame(x, columns=['ppp_name', 'interaction:']).set_index('ppp_name')
    print(f'\tkurt_skew_detectable plot... ({timestamp.get_timestamp_seconds()})')
    draw_kurt_skew_plot([ppi.ppp for ppi in ppis], params, additional_hue=hue,
                        filepath_out=out_dir + 'skew_kurt_plot_interacting_detectable.png')
    print(f'\tremoval history plot of complexes... ({timestamp.get_timestamp_seconds()})')
    plot_ppp_removal_history(interaction_predictions_before_removal,ppis, params, filepath_out=out_dir + 'removal_history_ppps.png')
    if params['plot_expensive_plots']:
        print(f'\tremoval history plot of ECs... ({timestamp.get_timestamp_seconds()})')
        plot_EC_removal_history(interaction_predictions_before_removal,all_ppps,ppis, params, filepath_out=out_dir + 'removal_history_ECs.png')
    if params['plot_expensive_plots']:
        print(f'\tinvolved_aa plot... ({timestamp.get_timestamp_seconds()})')
        plot_involved_aa(all_ppps, params, calculate_ap_ecs=True,
                         filepath_out=[out_dir + 'amino_acids_involved_in_actual_postitive_ECs.png',
                                       out_dir + 'amino_acids_involved_in_actual_negative_ECs.png',
                                       out_dir + 'amino_acids_involved_in_actual_postitive_ECs_normalised.png',
                                       out_dir + 'amino_acids_involved_in_actual_negative_ECs_normalised.png'])
    print(f'\tnum_ap_per_ppi after interaction prediction plot... ({timestamp.get_timestamp_seconds()})')
    plot_num_ap_ec_per_ppi(ppis, params, filepath_out=[out_dir + 'number_of_actual_positive_ECs_per_PPI_after_interaction_prediction.png',
                           out_dir + 'number_of_actual_positive_ECs_per_PPI_with_cutoff_after_interaction_prediction.png'])
    if params['plot_expensive_plots']:
        print(f'\tnum_ap_per_ppp plot... ({timestamp.get_timestamp_seconds()})')
        plot_num_ap_ec_per_ppp(all_ppps, params, filepath_out=[out_dir + 'number_of_actual_positive_ECs_per_PPP.png',
                               out_dir + 'number_of_actual_positive_ECs_per_PPP_with_cutoff.png'])
    #print(f'\ttp_vs_ap_ecs plot... ({timestamp.get_timestamp_seconds()})')
    #plot_tp_ecs_per_complex(ppis, params, filepath_out=out_dir + 'correlation_predicted_vs_ap_ecs_per_ppp.png')
    print(f'\tnum_predicted_ecs plot... ({timestamp.get_timestamp_seconds()})')
    plot_num_of_interface_predictions(ppis, params, filepath_out=out_dir + 'num_predicted_ecs.png')
    print(f'\trsa_comparison_an plot... ({timestamp.get_timestamp_seconds()})')
    plot_rsa_comparison_an(ppis, params, filepath_out=out_dir + 'rsa_comparison_an.png')
    print(f'\trsa_comparison_ap plot... ({timestamp.get_timestamp_seconds()})')
    plot_rsa_comparison_ap(ppis, params, filepath_out=out_dir + 'rsa_comparison_ap.png')
    print(f'\tkurt_vs_ap plot... ({timestamp.get_timestamp_seconds()})')
    plot_kurtosis_vs_AP(ppis, params, filepath_out=out_dir + 'kurtosis_vs_actual_positive_ECs.png')
    if params['plot_expensive_plots']:
        print(f'\tkurt_vs_num_contacts plot... ({timestamp.get_timestamp_seconds()})')
        plot_kurtosis_vs_num_contacts(all_ppps, params, filepath_out=out_dir + 'kurtosis_vs_num_contacts.png')
    print(f'\tinteraction_prediction plot... ({timestamp.get_timestamp_seconds()})')
    plot_interaction_prediction(interaction_predictions_before_removal, params, filepath_out=out_dir + 'interaction_prediction_distribution.png')
    if params['plot_expensive_plots']:
        print(f'\tconservation_vs_cn plot... ({timestamp.get_timestamp_seconds()})')
        plot_conservation_vs_cn(ppis, params, filepath_out=[out_dir + 'total_conservation_vs_cn.png', out_dir + 'single_conservation_vs_cn.png'])


from src.visualising.contact_maps import draw_contact_map, draw_ec_dots
from src.visualising.skew_kurt import plot_ec_distribution
from src.haddock_code.haddock_analysis import visualise_restraints
def plot_complex_details(out_dir, ppis, params):
    #plots all plots that refer to single complexes.
    # therefore each complex gets its own directory in which the respective plots are saved

    for i,ppi in enumerate(ppis):
        print(f'\t{i}/{len(ppis)} ppis: {ppi.ppp.name}... ({timestamp.get_timestamp_seconds()})')
        curr_dir = out_dir + ppi.ppp.name + '_' + ppi.ppp.complex_description + '/'
        os.makedirs(curr_dir, exist_ok=True)
        draw_contact_map(ppi.ppp,params,filepath_out=curr_dir+'contact_map.png')
        draw_ec_dots(ppi, params, title=f'Contact map of ECs for {ppi.ppp.name}', filepath_out=curr_dir+'EC_contact_map.png')
        plot_ec_distribution(ppi.ppp, params, filepath_out=curr_dir+'ec_distribution.png', with_norm_dist=False)
        plot_ec_distribution(ppi.ppp, params, filepath_out=curr_dir+'ec_distribution_with_norm.png', with_norm_dist=True)
        plot_conservation_vs_cn_single_ppi(ppi,params,filepath_out=[curr_dir + 'total_conservation_vs_cn.png', curr_dir + 'single_conservation_vs_cn.png'])
        #visualise restraints
        if ppi.ppp.pdb_complex_file_in is not None:
            # calculate restraints
            restraints = [[row['i'], row['j']] for (idx, row) in ppi.df_ecs.iterrows() if
                    row['prediction_confidence'] > params['confidence_threshold_residue_level']]
            visualise_restraints(ppi.ppp.pdb_complex_file_in,restraints, pml_file_out= curr_dir+'visualise_predicted_restraints_on_complex_pdb.pml', complex_identifier=ppi.ppp.complex_description, index=ppi.ppp.name)


def write_predictions(ppis, out_dir):
    #writes all predictions so far in out_dir: 1 file containing all interaction predictions, and 1 folder containing 1 interface prediction file for each complex
    if not out_dir.endswith('/'): out_dir += '/'
    os.makedirs(out_dir, exist_ok=True)
    out_dir_interface = out_dir + 'interface/'
    os.makedirs(out_dir_interface, exist_ok=True)
    out_interaction_text='complex\tinteraction_prediction\n'
    interactions=[]
    for ppi in ppis:
        ppi.df_ecs.to_csv(out_dir_interface+ppi.ppp.name+'_interface_prediction.csv')
        interactions.append((ppi.ppp.name, ppi.interaction_confidence))
    #sort interaction prediction by confidence
    interactions.sort(key=lambda x: x[1], reverse=False)
    for (complex_name, interaction_prediction) in interactions:
        out_interaction_text += f'{complex_name}\t{interaction_prediction}\n'
    with open(out_dir+'interaction_predictions.csv', 'w') as f:
        f.write(out_interaction_text)


def write_info_on_1_and_2_predicted_EC_complexes(ppis,params, out_file_path=None):
    # some complexes only have 1 or 2 predicted ecs. For modelling these predictions are stocked up to 3, to ensure enough contacts for modelling without too many degrees of freedom
    # this function gives information on the complexes in question as well as the added ECs
    # if no out_file_path is given, the info will be printed on console, otherwise it will be saved as a txt file

    ppis_with_1_ca = [0, 0, 0, 0, 0]
    ppis_with_2_ca = [0, 0, 0, 0, 0]
    ppis_with_3_or_more_ca = [0, 0, 0]
    list_of_small_ppis_1 = []
    list_of_small_ppis_2 = []
    for ppi in ppis:
        ca_s = [[row['i'], row['j']] for (idx, row) in ppi.df_ecs.iterrows() if
                row['prediction_confidence'] > params['confidence_threshold_residue_level']]
        if len(ca_s) == 1:
            list_of_small_ppis_1.append(ppi.ppp.name)
            ppis_with_1_ca[0] += 1
            if ppi.is_interacting:
                ppis_with_1_ca[1] += 1
            else:
                ppis_with_1_ca[2] += 1
            df2 = ppi.df_ecs.sort_values('prediction_confidence', ascending=False)
            df2 = df2.reset_index(drop=True)
            if df2.loc[1]['true_ec']:
                ppis_with_1_ca[3] += 1
            else:
                ppis_with_1_ca[4] += 1
            if df2.loc[2]['true_ec']:
                ppis_with_1_ca[3] += 1
            else:
                ppis_with_1_ca[4] += 1
        if len(ca_s) == 2:
            list_of_small_ppis_2.append(ppi.ppp.name)
            ppis_with_2_ca[0] += 1
            if ppi.is_interacting:
                ppis_with_2_ca[1] += 1
            else:
                ppis_with_2_ca[2] += 1
            df2 = ppi.df_ecs.sort_values('prediction_confidence', ascending=False)
            df2 = df2.reset_index(drop=True)
            if df2.loc[2]['true_ec']:
                ppis_with_2_ca[3] += 1
            else:
                ppis_with_2_ca[4] += 1
        if len(ca_s) > 2:
            ppis_with_3_or_more_ca[0] += 1
            if ppi.is_interacting:
                ppis_with_3_or_more_ca[1] += 1
            else:
                ppis_with_3_or_more_ca[2] += 1
    out = f'''1 predicted EC:
    	{ppis_with_1_ca[0]} complexes
    		{ppis_with_1_ca[1]} interacting complexes
    		{ppis_with_1_ca[2]} non-interacting complexes
    	if we would add the next 2 best scoring ECs (so that we have 3 predicted ECs):
    		{ppis_with_1_ca[3]}   ECs added that are at an interaction site (therefore true)
    		{ppis_with_1_ca[4]} ECs added that are not at an interaction site (therefore false)

    2 predicted EC:
    	{ppis_with_2_ca[0]} complexes
    		{ppis_with_2_ca[1]} interacting complexes
    		{ppis_with_2_ca[2]}  non-interacting complexes
    	if we would add the next 1 best scoring ECs (so that we have 3 predicted ECs):
    		{ppis_with_2_ca[3]} ECs added that are at an interaction site (therefore true)
    		{ppis_with_2_ca[4]} ECs added that are not at an interaction site (therefore false)

    3 or more predicted EC:
    	{ppis_with_3_or_more_ca[0]} complexes
    		{ppis_with_3_or_more_ca[1]} interacting complexes
    		{ppis_with_3_or_more_ca[2]}  non-interacting complexes
    
    list of all ppis that have been altered:
    {list_of_small_ppis_1}
    {list_of_small_ppis_2}'''

    if out_file_path is None:
        print(out)
    else:
        with open(out_file_path,'w') as f:
            f.write(out)

























def full_prediction(pred_interaction,pred_interface, training_ppps, test_ppps, params, out_dir):
    # creates folder in out_dir with all information
    # uses the whole of training data to train the 2 machine learning models and optimally predict interaction interfaces
    # in the test data



    if not out_dir.endswith('/'): out_dir+='/'
    folder_name = timestamp.get_timestamp()
    os.makedirs(out_dir+folder_name, exist_ok=True)
    out_dir = out_dir+folder_name+'/'



    if params['plot_expensive_plots']:
        print(f'print_info_on_training_set... ({timestamp.get_timestamp_seconds()})')
        print_info_on_training_set(training_ppps, params, out_path=out_dir+'training_set_info.txt')
        print(f'print_info_on_test_set... ({timestamp.get_timestamp_seconds()})')
        print_info_on_training_set(test_ppps, params, out_path=out_dir + 'test_set_info.txt')


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~interaction~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print()
    print(f'interaction prediction... ({timestamp.get_timestamp_seconds()})')
    #create features
    print(f'\tcreate features... ({timestamp.get_timestamp_seconds()})')
    print(f'\t\tcreate training features... ({timestamp.get_timestamp_seconds()})')
    feature_names, train_features, train_labels = pred_interaction.ppps_to_features(training_ppps, params, with_labels=True)
    print(f'\t\tcreate testing features... ({timestamp.get_timestamp_seconds()})')
    feature_names, test_features = pred_interaction.ppps_to_features(training_ppps, params, with_labels=False)

    # random sampling:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    resampler = params['resample_interaction_prediction_data']
    if resampler is not None:
        no_resampling = False
        if resampler == 'RandomUnderSampler':
            rs = RandomUnderSampler(random_state=42)
        elif resampler == 'RandomOverSampler':
            rs = RandomOverSampler(random_state=42)
        else:
            print(f'no random sub-sampler available by the name {resampler} so none was used')
            features_res, labels_res = train_features, train_labels
            no_resampling = True
        if not no_resampling:
            features_res, labels_res = rs.fit_resample(train_features, train_labels)
            print(
                f'{type(rs)} changes training set from {train_labels.count(True)}(true), {train_labels.count(False)}(false), to '
                f'{labels_res.count(True)}(true), {labels_res.count(False)}(false)')
    else:
        features_res, labels_res = train_features, train_labels


    # train predictor
    print(f'\ttrain interaction predictor... ({timestamp.get_timestamp_seconds()})')
    pred_interaction.learn(features_res, labels_res, params)

    #predict interaction
    print(f'\tpredict interaction... ({timestamp.get_timestamp_seconds()})')
    predictions = pred_interaction.predict_multiple_feature_list(test_features, params)


    #sort predictions
    print(f'\tsort predictions... ({timestamp.get_timestamp_seconds()})')
    keep_test_ppps=[]
    keep_predictions=[]
    for (test_ppp, test_pred) in zip(test_ppps, predictions):
        if test_pred > params['confidence_threshold_ppi']:
            keep_test_ppps.append(test_ppp)
            keep_predictions.append(test_pred)




    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~interface~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'interface prediction... ({timestamp.get_timestamp_seconds()})')

    # create features
    print(f'\tcreate features... ({timestamp.get_timestamp_seconds()})')
    print(f'\t\tcreate training features... ({timestamp.get_timestamp_seconds()})')
    train_feature_names, train_features, train_labels, train_ppis = pred_interface.ppps_to_features(training_ppps, params, with_labels=True)
    print(f'\t\tcreate testing features... ({timestamp.get_timestamp_seconds()})')
    test_feature_names, test_features, test_ppis = pred_interface.ppps_to_features(keep_test_ppps, params, with_labels=False)





    #chance for less features in one of (train, test) due to removing them in imputation
    # thus this hs to be fixed by removing it in the other one as well
    if not train_feature_names == test_feature_names:
        print(f'\t\t WARNING: the features used by test and training are different and will be adjusted accordingly')
        combined_feature_names = []
        for feature_name in test_feature_names:
            if feature_name in train_feature_names: combined_feature_names.append(feature_name)

        #delete in training
        delete_columns = []
        for i, col_name in enumerate(train_feature_names):
            if not col_name in combined_feature_names:
                delete_columns.append(i)
        delete_columns.sort(reverse=True)
        for d in delete_columns:
            del train_feature_names[d]
            for j in train_features:
                del j[d]

        # delete in testing
        delete_columns = []
        for i, col_name in enumerate(test_feature_names):
            if not col_name in combined_feature_names:
                delete_columns.append(i)
        delete_columns.sort(reverse=True)
        for d in delete_columns:
            del test_feature_names[d]
            for j in test_features:
                del j[d]

        print(f'\t\tnew train features: {train_feature_names}')
        print(f'\t\tnew test features: {test_feature_names}')






    # add interaction_confidence to test_ppis

    for ppi, i_conf in zip(test_ppis,keep_predictions):
        ppi.interaction_confidence = i_conf



    # random sampling:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    resampler = params['resample_interface_prediction_data']
    if resampler is not None:
        no_resampling = False
        if resampler == 'RandomUnderSampler':
            rs = RandomUnderSampler(random_state=42)
        elif resampler == 'RandomOverSampler':
            rs = RandomOverSampler(random_state=42)
        else:
            print(f'no random sub-sampler available by the name {resampler} so none was used')
            features_res, labels_res = train_features, train_labels
            no_resampling = True
        if not no_resampling:
            features_res, labels_res = rs.fit_resample(train_features, train_labels)
            print(
                f'{type(rs)} changes training set from {train_labels.count(True)}(true), {train_labels.count(False)}(false), to '
                f'{labels_res.count(True)}(true), {labels_res.count(False)}(false)')
    else:
        features_res, labels_res = train_features, train_labels

    # train predictor
    print(f'\ttrain interface predictor... ({timestamp.get_timestamp_seconds()})')
    pred_interface.learn(features_res, labels_res, params)

    # predict interface
    print(f'\tpredict interface... ({timestamp.get_timestamp_seconds()})')
    predictions = pred_interface.predict_multiple_feature_list(test_features, params)

    #sort predictions
    print(f'\tsort predictions... ({timestamp.get_timestamp_seconds()})')
    curr_pointer = 0
    for ppi in test_ppis:
        df_size = len(ppi.df_ecs)
        curr_predictions = predictions[curr_pointer:curr_pointer + df_size]
        curr_pointer += df_size
        ppi.df_ecs['prediction_confidence'] = curr_predictions



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~wrap-up~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



    #write predictions so far
    print(f'writing predictions... ({timestamp.get_timestamp_seconds()})')
    out_dir_predictions = out_dir + f'predictions/'
    os.makedirs(out_dir_predictions, exist_ok=True)
    out_dir_predictions = out_dir_predictions + f'{timestamp.get_timestamp_seconds()}/'
    os.makedirs(out_dir_predictions, exist_ok=True)
    write_predictions(test_ppis, out_dir_predictions)

    # write restraint files for possible docking step based on predictions
    from src.haddock_code.haddock_generation import write_dist_restraints_file, run_haddock_with_params
    if params['generate_distance_restraints'] or params['run_haddock']:
        print(f'writing restraint files... ({timestamp.get_timestamp_seconds()})')
        dist_restraint_path = out_dir + 'distance_restraints/'
        os.makedirs(dist_restraint_path, exist_ok=True)


        for ppi in test_ppis:
            ca_s = [[row['i'], row['j']] for (idx, row) in ppi.df_ecs.iterrows() if
                    row['prediction_confidence'] > params['confidence_threshold_residue_level']]
            if len(ca_s) > 0:

                if params['distance_restraints_with_monomer_pdbs']:
                    dirpath_complicated = dist_restraint_path + ppi.ppp.name + '_' + ppi.ppp.protein1.uid.replace(
                        '_HUMAN', '') + '_' + ppi.ppp.protein2.uid.replace('_HUMAN', '') + '/haddock_ecs/'
                    os.makedirs(dirpath_complicated, exist_ok=True)
                write_dist_restraints_file(ca_s, f'{dirpath_complicated}{ppi.ppp.name}_restraints.tbl', params, ppi)

    # run haddock
    if params['run_haddock']:
        print(f'starting HADDOCK... ({timestamp.get_timestamp_seconds()})')
        run_haddock_with_params(test_ppis, params, out_dir)