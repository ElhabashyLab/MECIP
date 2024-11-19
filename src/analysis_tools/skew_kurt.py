import pandas as pd

# ###########################   WARNING   #####################
#This class is out of order so far

def get_s_k_df(ppps, params):
    results=[]
    for ppp in ppps:
        k,s = ppp.calc_kurtosis_and_skewness(exclude_outside_pdb=params['remove_ecs_outside_of_monomer_pdb'])
        results.append([ppp.name, k, s])
    return pd.DataFrame(results, columns=['filename', 'excess_kurtosis', 'skewness']).set_index('filename')


def get_ecs_above_decision_boundary(ppps, params):
    #returns s_k_df of ecs located above the decision boundary
    s_k_df = get_s_k_df(ppps, params)

    result_indices = []
    for idx, row in s_k_df.iterrows():
        if row['excess_kurtosis'] >= eval(params['ks_plot_decision_boundary'])(row['skewness']):
            result_indices.append(row.name)
    return s_k_df.loc[result_indices]

def get_fraction_of_samples_above_boundary(ppps, params):
    s_k_df = get_s_k_df(ppps, params)
    result_indices = []
    for idx, row in s_k_df.iterrows():
        if row['excess_kurtosis'] >= eval(params['ks_plot_decision_boundary'])(row['skewness']):
            result_indices.append(row.name)
    s_k_df_above = s_k_df.loc[result_indices]
    return len(s_k_df_above.index)/(len(s_k_df.index))