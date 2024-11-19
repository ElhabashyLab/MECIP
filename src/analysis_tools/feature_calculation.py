import copy

import pandas as pd
from scipy.stats import kurtosis, skew, iqr
import statistics
import re
from Bio.Emboss.Applications import NeedleCommandline
from Bio.Align import substitution_matrices
from Bio import pairwise2


#helper functions



def get_topx_model(models, df_top_ecs, x, params):
    # gives the model that has the most satisfied ECs in the top x ECs
    best_model = None
    best_frac = -1
    for model in models:
        dists=[]
        for idx, row in df_top_ecs.head(x).iterrows():
            try:
                dist = model.chains[0].get_res(row['i']).calc_heavy_atom_distance_to(model.chains[1].get_res(row['j']))
            except (IndexError, AttributeError):
                dist = None
            if dist is not None: dists.append(dist < params['TP_interaction_distance_threshold'])
        curr_frac = sum(dists) / len(dists) if len(dists)>0 else -1
        if curr_frac>best_frac:
            best_model = model
            best_frac = curr_frac
    return best_model


def get_rsa_dfs(ppp, params):
    return ppp.protein1.get_rsa_df(), ppp.protein1.get_rsa_df()
    # old code
    #if ppp.protein1.naccess_file_in and ppp.protein1.dssp_file_in:
    #    rsa_preferred = params['rsa_preferred_method']
    #    if rsa_preferred == 'dssp':
    #        rsa_df_1 = ppp.protein1.get_dssp_df()
    #        rsa_df_2 = ppp.protein2.get_dssp_df()
    #    elif rsa_preferred == 'naccess':
    #        rsa_df_1 = ppp.protein1.get_naccess_df()
    #        rsa_df_2 = ppp.protein2.get_naccess_df()
    #    else:
    #        print(f'the preferred rsa method {rsa_preferred} does not exist')
    #        return None,None
    #elif ppp.protein1.naccess_file_in:
    #        rsa_df_1 = ppp.protein1.get_naccess_df()
    #        rsa_df_2 = ppp.protein2.get_naccess_df()
    #elif ppp.protein1.dssp_file_in:
    #        rsa_df_1 = ppp.protein1.get_dssp_df()
    #        rsa_df_2 = ppp.protein2.get_dssp_df()
    #else:
    #    return None, None
    #return rsa_df_1, rsa_df_2

def pairwise_align_subst_matr(query_seq, target_seq):
    # aligns 2 sequences with the help of the BLOSUM62 substitution matrix
    #not done yet
    matrix = substitution_matrices.load("BLOSUM62")
    curr=0
    for a in pairwise2.align.globaldx(query_seq, target_seq, matrix):
        curr +=1
        print(curr)
        print(a)



def pairwise_align(query_seq, target_seq):
    # aligns 2 sequences with the help of the Biopython package and a global alignment method

    global_align = pairwise2.align.globalxx(query_seq, target_seq)

    seq_length = min(len(global_align[0][0]), len(global_align[0][1]))
    matches = global_align[0][2]

    return (matches / seq_length) * 100

def needle_align_code(query_seq, target_seq):
    # aligns 2 sequences with the help of the needle command line
    needle_cline = NeedleCommandline(asequence="asis:" + query_seq,
                                     bsequence="asis:" + target_seq,
                                     aformat="simple",
                                     gapopen=10,
                                     gapextend=0.5,
                                     outfile='stdout'
                                     )
    out_data, err = needle_cline()
    out_split = out_data.split("\n")
    p = re.compile("\((.*)\)")
    return p.search(out_split[25]).group(1).replace("%", "")





#local features

def cn(row):
    #returns cn value of a row of a ec df
    return row['cn']

def rel_rank_ec(i, df_ec):
    #i  :   int :   row number of curr ec (can be obtained by 'enumerate(df.iterrows())')
    #returns the relative rank of a certain ec (value between 0 and 1 with 0 being the top)
    return i/len(df_ec)-1

def dist_to_higher_cn(i, df_ec):
    #i  :   int :   row number of curr ec (can be obtained by 'enumerate(df.iterrows())')
    #returns the difference between the current cn value and the cn value of the next best scoring ec
    if i==0: return 0
    try:
        return df_ec.iloc[i-1]['cn'] - df_ec.iloc[i]['cn']
    except IndexError:
        return None

def dist_to_lower_cn(i, df_ec):
    #i  :   int :   row number of curr ec (can be obtained by 'enumerate(df.iterrows())')
    #returns the difference between the current cn value and the cn value of the next worse scoring ec
    if i==len(df_ec)-1: return df_ec.iloc[i]['cn']
    try:
        return df_ec.iloc[i]['cn'] - df_ec.iloc[i+1]['cn']
    except IndexError:
        return None

def cn_density(i, df_ec, bar_size=0):
    #i  :   int :   row number of curr ec (can be obtained by 'enumerate(df.iterrows())')
    #bar_size   :   int :   can be determined by hand, if 0: bar size is 1/n of total range (with n=number of ECs)
    #returns the number of ecs within the cn-window of [cn(i)-bar_size/2, cn(i)+bar_size/2]
    curr_cn =  df_ec.iloc[i]['cn']
    all_cn = list(df_ec['cn'])
    if bar_size<=0: bar_size=(max(all_cn)-min(all_cn))/len(all_cn)
    return len([i for i in all_cn if i<curr_cn+bar_size/2 and i>curr_cn-bar_size/2])




def conservation(rel_i, rel_j, frequencies_file_in):
    #rel_i, rel_j  :   int :   relative i: starts at 1 (not 0) and also is different for second monomer (rel_j) in a complex, since it keeps on counting (if first monomer is 100 res long, rel_j of the first residue of monomer 2 is 101)
    # returns the added, min and max conservation for a specific residue pair based on the frequencies file
    if frequencies_file_in is not None:
        frequencies_df = pd.read_csv(frequencies_file_in)
        try:
            c1 = float(frequencies_df[frequencies_df['i'] == rel_i]['conservation'])
            c2 = float(frequencies_df[frequencies_df['i'] == rel_j]['conservation'])
            return c1+c2, min(c1,c2), max(c1,c2)
        except Exception as e:
            return None, None, None

    else:
        return None, None, None

def is_in_cluster(clusters, row):
    #clusters   :   [[int]] :   return value from 'calculate_clusters'
    #returns 0 if EC is not in any cluster, 1 if EC is in cluster
    for cluster in clusters:
        if row.name in cluster:
            return 1
    return 0

def cluster_size(clusters, row):
    #clusters   :   [[int]] :   return value from 'calculate_clusters'
    #returns 0 if EC is not in any cluster, len(cluster) if EC is in cluster
    for cluster in clusters:
        if row.name in cluster:
            return len(cluster)
    return 0

def rsa_ij(row, ppp, params):
    #returns the sum of both rsa's from the ec is question
    rsa_df_1, rsa_df_2 = get_rsa_dfs(ppp, params)
    if not isinstance(rsa_df_1, pd.DataFrame) or not isinstance(rsa_df_2, pd.DataFrame):
        return None
    try:
        return rsa_df_1.loc[row['i']]['All-atoms_rel'] + rsa_df_2.loc[row['j']]['All-atoms_rel']
    except KeyError:
        #print('nope')
        return None

def rsa_min(row, ppp, params):
    #returns the smaller rsa from the 2 residue rsa's from the ec in question
    rsa_df_1, rsa_df_2 = get_rsa_dfs(ppp, params)
    if not isinstance(rsa_df_1, pd.DataFrame) or not isinstance(rsa_df_2, pd.DataFrame):
        return None
    try:
        return min(rsa_df_1.loc[row['i']]['All-atoms_rel'],rsa_df_2.loc[row['j']]['All-atoms_rel'])
    except KeyError:
        #print('nope')
        return None

def heavy_atom_distance_in_top_model(model, row):
    # returns 0 or 1 indicating if the specific ec is validated by the top scoring model returned by haddock
    try:
        dist = model.chains[0].get_res(row['i']).calc_heavy_atom_distance_to(model.chains[1].get_res(row['j']))
    except (IndexError, AttributeError):
        dist = None
    return dist


def heavy_atom_distance_in_models(models, row, params):
    # returns value between 0 or 1 indicating if the specific ec is validated by the top scoring models returned by haddock
    # also returns values for num_models_satisfied: number of models in which the ec is satisfied
    dists = []
    for model in models:
        try:
            dist = model.chains[0].get_res(row['i']).calc_heavy_atom_distance_to(model.chains[1].get_res(row['j']))
        except (IndexError, AttributeError):
            dist = None
        dists.append(dist)
    dists = [dist for dist in dists if dist is not None]
    if len(dists)==0: return None, None
    return np.mean(dists), len([x for x in dists if x<params['TP_interaction_distance_threshold']])




def n_eff_ij(msa_file_path, rel_i, rel_j):
    #rel_i, rel_j  :   int :   relative i: starts at 1 (not 0) and also is different for second monomer(rel_j) in a complex, since it keeps on counting (if first monomer is 100 res long, rel_j of the first residue of monomer 2 is 101)
    #TODO: dont know yet where to get this info (maybe from msas?)
    pass











#global features

def cn_dist_metrics(ppp, params):
    #returns kurtosis, skewness, max_cn, median_cn, iqr_cn, and jarque_bera_test_statistic
    ecs_df = ppp.get_ecs(exclude_outside_pdb=params['remove_ecs_outside_of_monomer_pdb'])
    if not isinstance(ecs_df, pd.DataFrame):
        return None, None, None, None, None, None
    if ecs_df.empty:
        return None, None, None, None, None, None
    ecs_cn = ecs_df['cn']
    kurt = kurtosis(ecs_cn)
    ske = skew(ecs_cn)
    #n_obs = params['num_observations']
    dof = 2 #(degrees of freedom)
    jarque_bera_test_statistic = (dof/6)*(ske*ske+((kurt)*(kurt)/4))
    return kurt, ske, max(ecs_cn), statistics.median(ecs_cn), iqr(ecs_cn), jarque_bera_test_statistic

def sequence_id(ppp):
    # uses global alignment search algorithm to determine sequence identity
    return pairwise_align(ppp.protein1.get_sequence(), ppp.protein2.get_sequence())

def n_xlink(model, ppp):
    #returns number of xlinks fulfilled by model
    #TODO
    pass


def f_xlink(model, ppp):
    #returns fraction of xlinks fulfilled by model (form 0 to 1)
    #TODO
    pass


def n_ecs(df_top_ecs):
    #returns number of ecs that are being considered
    return len(df_top_ecs)

def n_clusters(clusters):
    #clusters   :   [[int]] :   return value from 'calculate_clusters'
    #returns number of clusters found
    return len(clusters)

def size_clusters(clusters):
    #clusters   :   [[int]] :   return value from 'calculate_clusters'
    #returns total size of all clusters found
    return sum([len(cluster) for cluster in clusters])


def n_seq(ppp):
    #returns number of sequences used in msa used for ec calculations
    return ppp.N_seq

def n_eff(ppp):
    #returns number of effective sequences used in msa used for ec calculations (similar sequences are downweighted)
    return ppp.N_eff

def sequence_length(ppp):
    # returns length of concatinated msa used for ec calculations
    return ppp.seq_length

def haddock_score_best(ppp):
    #returns the haddock score of the best scoring docked complex
    return ppp.haddock_score_best

def haddock_score_average(ppp):
    #returns the mean of all haddock scores of the best scoring docked complexes
    return ppp.haddock_score_average

def haddock_num_contacts(models, params):
    # dont use this. runtime way too long
    contacts = []
    for model in models:
        try:
            contact = 0
            for res1 in model.chains[0].residues:
                for res2 in model.chains[1].residues:
                    if res1.calc_heavy_atom_distance_to(res2) > params['TP_interaction_distance_threshold']: contact+=1
        except (IndexError, AttributeError):
            contact = None
        contacts.append(contact)
    contacts = [contact for contact in contacts if contact is not None]
    if len(contacts) == 0: return None
    return np.mean(contacts)


def haddock_ecs_in_all(models, df_top_ecs, params):
    # returns the number of ECs that are satisfied in all, and in at least half of the models
    frac = []
    for idx, row in df_top_ecs.iterrows():
        dists = []
        for model in models:
            try:
                dist = model.chains[0].get_res(row['i']).calc_heavy_atom_distance_to(model.chains[1].get_res(row['j']))
            except (IndexError, AttributeError):
                dist = None
            if dist is not None: dists.append(dist<params['TP_interaction_distance_threshold'])
        if len(dists)>0: frac.append(sum(dists)/len(dists))
    return len([x for x in frac if x==1]), len([x for x in frac if x>=.5])

    

def n_val_ecs(model, df_top_ecs, params):
    #returns number of validated ecs by the constructed model
    #TODO
    pass

def f_val_ecs(model, df_top_ecs, params):
    #returns fraction of validated ecs by the constructed model
    #TODO
    pass

def blddt(alphafold_model):
    #returns BLDDT value from given alphafold model
    #TODO
    pass

def btm(alphafold_model):
    #returns BTM value from given alphafold model
    #TODO
    pass







import numpy as np
import copy
def impute_missing_data(features, feature_names, params):
    #impute missing data
    # removes all Nones from features and feature names and returns modified lists
    # removes whole features if they are

    #remove all columns with more than 50% Nones
    delete_columns=[]
    for i,col_name in enumerate(feature_names):
        col = [x[i] for x in features]
        if sum(x is None for x in col) > len(features)/2:
            print(f'feature {col_name} is removed due to more than 50% None')
            delete_columns.append(i)
    delete_columns.sort(reverse=True)
    for d in delete_columns:
        del feature_names[d]
        for j in features:
            del j[d]


    #impute by using mean
    if params['imputation_method'] == 'mean':
        num_Nones = sum([1 for b in features for a in b if a is None])
        print(f'number of Nones in features that have been replaced after column deletion: {num_Nones}')
        #precompue all means for all features:
        num_features = len(features[0])
        means = [0]*num_features
        #print(features)
        for i in range(num_features):
            if not isinstance(features[0][i],str): means[i] = np.mean([row[i] for row in features if not row[i] is None])


        for feature in features:
            if None in feature:
                for i,f in enumerate(feature):
                    if f is None:
                        curr_feature_name = feature_names[i]
                        feature[i] = means[i]
        return feature_names, features

    #k-nearest neighbour approach
    elif params['imputation_method'] == 'knn':
        import sys
        from impyute.imputation.cs import fast_knn
        num_Nones = sum([1 for b in features for a in b if a is None])
        print(f'number of Nones in features that have been replaced after column deletion: {num_Nones}')
        sys.setrecursionlimit(100000)  # Increase the recursion limit of the OS

        # start the KNN training
        features_first = [x[0] for x in features]
        for j in features:
            del j[0]
        features_np = np.array([np.array([float(xii) if xii is not None else np.nan for xii in xi]) for xi in features])
        imputed_features_np = fast_knn(features_np, k=30)
        full_imputed_features_np = [[a]+b.tolist() for (a,b) in zip(features_first, imputed_features_np)]
        return feature_names, full_imputed_features_np
    else:
        print('WARNING: unknown imputation method. Mean is used.')
        num_Nones = sum([1 for b in features for a in b if a is None])
        print(f'number of Nones in features that have been replaced after column deletion: {num_Nones}')
        # precompue all means for all features:
        num_features = len(features[0])
        means = [0] * num_features
        # print(features)
        for i in range(num_features):
            if not isinstance(features[0][i], str): means[i] = np.mean(
                [row[i] for row in features if not row[i] is None])

        for feature in features:
            if None in feature:
                for i, f in enumerate(feature):
                    if f is None:
                        curr_feature_name = feature_names[i]
                        feature[i] = means[i]
        return feature_names, features
