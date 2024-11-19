import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
import sklearn.metrics as metrics
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.ecs import mark_true_ecs

def draw_precision_recall_curve(df_metrics, confidence_threshold, filepath_out=None):
    # draws a precision recall curve for a given prediction metric dataframe
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath


    sns.lineplot(data=df_metrics, x='recall', y='precision')
    plt.plot(df_metrics.loc[confidence_threshold]['recall'],
             df_metrics.loc[confidence_threshold]['precision'],'gX',  markersize=8)

    # Compute the area under curve using the composite Simpson's rule.

    area = metrics.auc(df_metrics['recall'], df_metrics['precision'])
    plt.title(f'precision_recall_curve, AUC={area}')
    print(f'AUPRC: {area}')



    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()

def draw_roc_curve(df_metrics,confidence_threshold, filepath_out=None):
    # draws a receiver operator curve for a given prediction metric dataframe
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath


    sns.lineplot(data=df_metrics, x='false_positive_rate', y='recall')
    plt.plot(df_metrics.loc[confidence_threshold]['false_positive_rate'],
             df_metrics.loc[confidence_threshold]['recall'],'gX',  markersize=8)

    # Compute the area under curve using the composite Simpson's rule.
    area = metrics.auc(df_metrics['false_positive_rate'], df_metrics['recall'])
    print(f'AUROC: {area}')
    plt.title(f'roc_curve, AUC={area}')

    #dotted line at minimum
    x = [0,1]
    plt.plot(x,x,'b--')



    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()

def plot_num_ap_ec_per_ppi(ppis, params, calculate_ap_ecs=False, filepath_out=None):
    # plots how many actual positive ecs are in each ppp
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath

    values = []
    for ppi in ppis:
        ppp = ppi.ppp
        if calculate_ap_ecs:
            df_top_ecs = get_top_ecs(ppp, params)
            df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(), params['TP_EC_distance_threshold_heavy_atoms'], ppp, params)
            labels = []
            for idx, row in df_top_ecs.iterrows():
                labels.append(int(row['true_ec']))
            ppp.num_ap_ecs = sum(labels)
        if hasattr(ppp, 'num_ap_ecs'):
            values.append([ppp.num_ap_ecs, ppi.is_interacting])
        else:
            values.append([sum(ppi.ap_ecs), ppi.is_interacting])
    if len(values)==0:
        print('first all actual positive ecs have to be computed')
        return

    values_df = pd.DataFrame(values, columns=['num_ecs', 'is_interacting'])
    sns.histplot(values_df, x='num_ecs', hue='is_interacting', multiple='dodge',bins=40)
    plt.title('histogram of number of actual positive ECs (after first prediction)')
    plt.xlabel('num_ap_ecs')
    plt.ylabel('ppp_count')
    plt.yscale('log')

    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out[0])
    else:
        plt.show()
    plt.clf()

    # cut off at num_ap_ecs = 20
    for v in values:
        if v[0]>20: v[0]=20
    values_df = pd.DataFrame(values, columns=['num_ecs', 'is_interacting'])
    sns.histplot(values_df, x='num_ecs', hue='is_interacting', multiple='dodge', bins=21)
    plt.title('histogram of number of actual positive ECs (summed bin for 20+)(after first prediction)')
    plt.xlabel('num_ap_ecs')
    plt.ylabel('ppp_count')
    plt.yscale('log')

    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out[1])
    else:
        plt.show()
    plt.clf()


def plot_num_ap_ec_per_ppp(ppps, params, filepath_out=None):
    # plots how many actual positive ecs are in each ppp
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath
    values = []
    from src.analysis_tools.interface import get_ap_interaction
    for ppp in ppps:
        df_top_ecs = get_top_ecs(ppp, params)
        df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(), params['TP_EC_distance_threshold_heavy_atoms'], ppp, params)
        labels = []
        for idx, row in df_top_ecs.iterrows():
            labels.append(int(row['true_ec']))
        ppp.num_ap_ecs = sum(labels)
        values.append([ppp.num_ap_ecs, get_ap_interaction(ppp, params)])
    if len(values)==0:
        print('first all actual positive ecs have to be computed')
        return
    values_df = pd.DataFrame(values, columns=['num_ecs', 'is_interacting'])
    sns.histplot(values_df, x='num_ecs', hue='is_interacting', multiple='dodge',bins=40)
    plt.title('histogram of number of actual positive ECs')
    plt.xlabel('num_ap_ecs')
    plt.ylabel('ppp_count')
    plt.yscale('log')

    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out[0])
    else:
        plt.show()
    plt.clf()

    # cut off at num_ap_ecs = 20
    for v in values:
        if v[0]>20: v[0]=20
    values_df = pd.DataFrame(values, columns=['num_ecs', 'is_interacting'])
    sns.histplot(values_df, x='num_ecs', hue='is_interacting', multiple='dodge', bins=21)
    plt.title('histogram of number of actual positive ECs (summed bin for 20+)')
    plt.xlabel('num_ap_ecs')
    plt.ylabel('ppp_count')
    plt.yscale('log')

    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out[1])
    else:
        plt.show()
    plt.clf()



def plot_tp_ecs_per_complex(ppis, params, filepath_out=None):
    # plots how many predicted true positive ecs are in each ppp
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath

    values = []
    for ppi in ppis:
        values.append([sum(ppi.ap_ecs), sum(ppi.df_ecs['prediction_confidence']>params['confidence_threshold_residue_level'])])
    values = pd.DataFrame(values, columns=['num_ecs', 'num_predicted_ecs'])
    sns.histplot(values, x='num_ecs', multiple='stack', palette='tab10')
    plt.title('number of ap ecs per complex')
    plt.xlabel('num_ap_ecs')
    plt.ylabel('ppp_count')

    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()

def plot_involved_aa(ppps, params, calculate_ap_ecs=False, filepath_out=None):
    #filepath_out should be a list of 4 paths (actual positive ECs, actual negative ECs, ap ECs normalised, an ECs normalised)
    # normalisation happens over all observed ec residues
    #plots the involved amino acids for each actual positive ec
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath

    values_ap = []
    values_an = []
    for ppp in ppps:
        if calculate_ap_ecs:
            if hasattr(ppp, 'ap_ec_ij') and hasattr(ppp, 'an_ec_ij'):
                values_ap.extend(ppp.ap_ec_ij)
                values_an.extend(ppp.an_ec_ij)
            else:
                df_top_ecs = get_top_ecs(ppp, params)
                if df_top_ecs is None: continue
                df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(),
                                           params['TP_EC_distance_threshold_heavy_atoms'], ppp, params)
                labels = []
                for idx, row in df_top_ecs.iterrows():
                    labels.append(int(row['true_ec']))
                ap_results = []
                an_results = []
                for l, (idx, row) in zip(labels, df_top_ecs.iterrows()):
                    if l == 1:
                        ap_results.append([row['i'], row['A_i'], row['j'], row['A_j']])
                    else:
                        an_results.append([row['i'], row['A_i'], row['j'], row['A_j']])
                ppp.ap_ec_ij = ap_results
                ppp.an_ec_ij = an_results
                values_ap.extend(ppp.ap_ec_ij)
                values_an.extend(ppp.an_ec_ij)
    if len(values_ap) == 0:
        print('first all actual positive ecs have to be computed')
        return
    if len(values_an)==0:
        print('first all actual negative ecs have to be computed')
        return
    #sort values into nice df
    for idx, values in enumerate([values_ap, values_an]):
        if idx<2:
            xs = [sorted([i[1], i[3]], reverse=True) for i in values]
            ys = [xs[0] + [int(xs.count(xs[0]))]]
            for x in xs:
                if x not in [y[:2] for y in ys] and not (x[0]=='-' or x[1]=='-') :
                    ys.append(x + [int(xs.count(x))])
            df = pd.DataFrame(ys, columns=['residue_1', 'residue_2', 'count'])
            annot = df['count'].max()<99
            df = df.pivot('residue_1', 'residue_2', 'count')
            if idx==0:
                ap_df = df
            if idx==1:
                an_df = df
        elif idx==2:
            df = ap_df/(ap_df+an_df)
            annot = True
        elif idx==2:
            df = an_df/(ap_df+an_df)
            annot = True
        ax = sns.heatmap(df, annot=annot, cmap="YlGnBu")
        if idx==0:
            plt.title('Amino Acid count for all interacting top scoring ECs')
        elif idx==1:
            plt.title('Amino Acid count for all non-interacting top scoring ECs')
        elif idx==2:
            plt.title('Amino Acid count for all interacting top scoring ECs (normalised)')
        elif idx==3:
            plt.title('Amino Acid count for all non-interacting top scoring ECs (normalised)')
        plt.tight_layout()
        if filepath_out:
            plt.savefig(filepath_out[idx])
        else:
            plt.show()
        plt.clf()

def plot_interaction_prediction(interaction_predictions_before_removal, params, filepath_out=None):
    #plots a histogram of the predicted interaction confidence, and its threshold as well as the number of excluded complexes
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath

    predictions = interaction_predictions_before_removal[0]
    label = interaction_predictions_before_removal[1]
    l = [predictions, label]
    l = list(map(list, zip(*l)))
    x = pd.DataFrame(l, columns=['predictions', 'is_interacting'])

    threshold = params['confidence_threshold_ppi']
    plt.axvline(threshold, linestyle='--', color='red', label='confidence_threshold for interactions')
    plt.text(threshold, 0, 'confidence threshold', rotation=90, verticalalignment='bottom')
    sns.histplot(data=x, x='predictions', hue='is_interacting', bins=30, multiple="stack")
    plt.xlabel('interaction_confidence')
    num_excluded = sum(i <= threshold for i in predictions)
    plt.title(f'distribution of interaction confidences, excluded compexes: {num_excluded}')
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()



def plot_ppp_removal_history(interaction_predictions_before_removal, ppis, params, filepath_out=None):
    # plots a bar plot showing how many ppp are excluded in which step and coloring the actual labels
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath

    ppi_threshold = params['confidence_threshold_ppi']
    ap_removed = [0,0,0]
    an_removed = [0,0,0]
    num_predicted_and_ap_ecs_more_2=0
    num_predicted_and_tp_ecs_more_2=0
    # removed by first prediction
    for (pred, label) in zip(interaction_predictions_before_removal[0],interaction_predictions_before_removal[1]):
        if pred<=ppi_threshold:
            if label:
                ap_removed[0]+=1
            else:
                an_removed[0]+=1

    #no interface prediction done
    for ppi in ppis:
        interface_predictions = [1 for (idx, row) in ppi.df_ecs.iterrows() if
                row['prediction_confidence'] > params['confidence_threshold_residue_level']]
        if len(interface_predictions)==0:
            if ppi.is_interacting:
                ap_removed[1]+=1
            else:
                an_removed[1]+=1

        #further considered
        else:
            if ppi.is_interacting:
                ap_removed[2]+=1

                ap_ECs = sum(ppi.ap_ecs)
                tp_ECs = len(ppi.df_ecs[(ppi.df_ecs['true_ec']) & (ppi.df_ecs['prediction_confidence']>params['confidence_threshold_residue_level'])])
                if ap_ECs>2:num_predicted_and_ap_ecs_more_2+=1
                if tp_ECs>2:num_predicted_and_tp_ecs_more_2+=1
            else:
                an_removed[2]+=1
    #print(f'ap_removed: {ap_removed}')
    #print(f'an_removed: {an_removed}')
    print(f'num_predicted_and_ap_ecs_more_2: {num_predicted_and_ap_ecs_more_2}')
    print(f'num_predicted_and_tp_ecs_more_2: {num_predicted_and_tp_ecs_more_2}')
    is_interacting = [True, True, True, False, False, False]
    removal_history = ['no interaction predicted', 'no interface predicted', 'still considered',
                       'no interaction predicted', 'no interface predicted', 'still considered']
    l = [ap_removed + an_removed, is_interacting, removal_history]
    l = list(map(list, zip(*l)))
    data = pd.DataFrame(l, columns=['number of complexes', 'is interacting', 'removal history'])
    ax = sns.barplot(x="removal history", y="number of complexes", hue="is interacting", data=data)
    for container in ax.containers:
        ax.bar_label(container)

    plt.title(f'removal history of all complexes')
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()



def plot_EC_removal_history(interaction_predictions_before_removal,ppps, ppis, params, filepath_out=None):
    # plots a bar plot showing how many ECs are excluded in which step and coloring the actual labels
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath

    ppi_threshold = params['confidence_threshold_ppi']
    ap_removed = [0,0,0,0]
    an_removed = [0,0,0,0]
    # removed by first prediction
    for (pred, ppp) in zip(interaction_predictions_before_removal[0],ppps):
        if pred<=ppi_threshold:
            df_top_ecs = get_top_ecs(ppp, params)
            df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(), params['TP_EC_distance_threshold_heavy_atoms'],
                                       ppp, params)
            for idx,row in df_top_ecs.iterrows():
                if row['true_ec']:
                    ap_removed[0] += 1
                else:
                    an_removed[0] += 1


    #no interface prediction done
    for ppi in ppis:
        interface_predictions = [1 for (idx, row) in ppi.df_ecs.iterrows() if
                row['prediction_confidence'] > params['confidence_threshold_residue_level']]
        if len(interface_predictions)==0:
            for i in ppi.ap_ecs:
                if i==1:
                    ap_removed[1]+=1
                else:
                    an_removed[1]+=1

        #further considered
        else:
            for idx,row in ppi.df_ecs.iterrows():
                if row['prediction_confidence'] <= params['confidence_threshold_residue_level']:
                    if row['true_ec']:
                        ap_removed[2] += 1
                    else:
                        an_removed[2] += 1
                else:
                    if row['true_ec']:
                        ap_removed[3] += 1
                    else:
                        an_removed[3] += 1
    #print(f'ap_removed: {ap_removed}')
    #print(f'an_removed: {an_removed}')
    is_interacting = [True, True, True,True, False, False, False, False]
    removal_history = ['no interaction predicted', 'no interface predicted', 'considered complex, not predicted', 'considered complex, predicted',
                       'no interaction predicted', 'no interface predicted', 'considered complex, not predicted', 'considered complex, predicted']
    l = [ap_removed + an_removed, is_interacting, removal_history]
    l = list(map(list, zip(*l)))
    data = pd.DataFrame(l, columns=['number of ECs', 'is interacting', 'removal history'])
    ax = sns.barplot(x="removal history", y="number of ECs", hue="is interacting", data=data)
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.title(f'removal history of all ECs')
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()



def plot_feature_importances(importances, names,title, filepath_out=None):
    #plots a barplot showing the most important features of a machine learning model
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath
    importances_df = pd.DataFrame([importances, names]).T.sort_values(0, ascending=False)
    importances_df.columns = ['feature importance', 'feature']
    chart = sns.barplot(data=importances_df, x='feature', y='feature importance')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title(title)
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()


def plot_num_of_interface_predictions(ppis, params, filepath_out=None):
    # plots as a histogram the number of ECs predicted for each complex, where at least 1 EC was predicted
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath
    num_predictions_list = []
    for ppi in ppis:
        num_predictions = sum([1 for (idx, row) in ppi.df_ecs.iterrows() if
                row['prediction_confidence'] > params['confidence_threshold_residue_level']])
        if num_predictions > 0: num_predictions_list.append(num_predictions)

    z = []
    for each_elem in list(set(num_predictions_list)):
        z.append([each_elem, num_predictions_list.count(each_elem)])
    df = pd.DataFrame(z, columns=['num_ecs', 'count'])
    chart = sns.barplot(data=df, x='num_ecs', y='count')
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')
    plt.title('number of predicted ecs per complex')

    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()