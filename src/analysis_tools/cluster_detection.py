import numpy as np
import pandas as pd

from .ecs import get_top_ecs, get_percentile_or_value_seeds
from itertools import combinations
import pickle

def get_clustered_ecs(ppp, params):
    # returns dataframes with all ecs in and all ecs not in clusters
    df_top_ecs = get_top_ecs(ppp, params)

    clusters = calculate_clusters(ppp, params, df_top_ecs)
    if clusters: in_cluster=list(set([item for sublist in clusters for item in sublist]))
    else: in_cluster=[]
    not_in_cluster= [i for i in list(df_top_ecs.index) + in_cluster if i not in list(df_top_ecs.index) or i not in in_cluster]
    try:
        return df_top_ecs.loc[in_cluster], df_top_ecs.loc[not_in_cluster]
    except KeyError:
        return None, df_top_ecs


def calculate_clusters(ppp, params, df_top_ecs):
    # ppp   :   PPP :   protein protein pair that should be considered
    # params    :   dict    :   dictionary containing all needed parameters:
    # cluster_ecs_consider_top_x, cluster_ecs_consider_percentile, cluster_ecs_seed_top_x, cluster_ecs_seed_percentile

    #if clusters are precomputed and saved in a file just take it from there:
    if ppp.cluster_file is not None and ppp.cluster_file.endswith('/'+generate_cluster_file_name(params)):
        with open(ppp.cluster_file, 'rb') as fp:
            clusters = pickle.load(fp)
        return clusters

    if params['cluster_method']=='pdb':
        clusters = calculate_clusters_pdb(ppp, params,df_top_ecs)
    elif params['cluster_method']=='old':
        clusters = calculate_clusters_old(ppp, params,df_top_ecs)
    return clusters

def generate_cluster_file_name(params):
    #generates the filename that the current cluster file should have with respect to the current params
    cluster_file_name = 'cluster_'
    cluster_file_name += str(params['cluster_method']) + '_'
    cluster_file_name += str(params['cluster_pre_compute_inner_contacts']) + '_'
    cluster_file_name += str(params['ecs_consider_top_x']) + '_'
    cluster_file_name += str(params['ecs_consider_percentile']) + '_'
    cluster_file_name += str(params['cluster_ecs_seed_top_x']) + '_'
    cluster_file_name += str(params['cluster_ecs_seed_percentile']) + '_'
    cluster_file_name += str(params['cluster_min_size']) + '_'
    cluster_file_name += str(params['cluster_cross_search']) + '_'
    cluster_file_name += str(params['cluster_circle_search']) + '_'
    cluster_file_name += str(params['cluster_circle_search_pdb']) + '_'
    cluster_file_name += str(params['rsa_exclude']) + '_'
    cluster_file_name += str(params['rsa_measurement']) + '_'
    cluster_file_name += str(params['rsa_threshold']) + '.pkl'
    return cluster_file_name


import os
def write_cluster_file(ppp, params):
    #writes the cluster file for a single ppp given the parameters in params
    # file is generated in same location as ec file
    # file name follows conventions from function: 'generate_cluster_file_name'
    filepath = ppp.ec_file_in
    if not filepath:
        print(f'Warning: no cluster file can be generated for {ppp.name}_{ppp.complex_description}, since no ec file exists')
        return
    filepath = '/'.join(filepath.split('/')[:-1])
    filepath += '/'+generate_cluster_file_name(params)
    if os.path.isfile(filepath): return
    df_top_ecs = get_top_ecs(ppp, params)
    clusters = calculate_clusters(ppp, params, df_top_ecs)
    with open(filepath, 'wb') as fp:
        pickle.dump(clusters, fp)
    ppp.cluster_file = filepath
    print(f'cluster file created for: {ppp.name}')




def calculate_clusters_pdb(ppp, params, df_top_ecs):
    # ppp   :   PPP :   protein protein pair that should be considered
    # params    :   dict    :   dictionary containing all needed parameters:
    # cluster_ecs_consider_top_x, cluster_ecs_consider_percentile, cluster_ecs_seed_top_x, cluster_ecs_seed_percentile
    # calculates clusters if pdb is present
    # uses monomer pdb to identify sets of ecs where the first residue (on protein 1) all are close to each other in 3d space
    # as well as the second residues (on protein 2) follow the same behaviour

    pdb1 = ppp.protein1.get_pdb_file()
    pdb2 = ppp.protein2.get_pdb_file()


    

    seeds = get_percentile_or_value_seeds(df_top_ecs, params)
    #print(len(df_top_ecs.index))
    #print(len(seeds.index))
    #print(seeds)
    #print(df_top_ecs.to_string())
    #pdb1=None
    if pdb1 and pdb2:
        inner_contact_map1=None
        inner_contact_map2=None
        if params['cluster_pre_compute_inner_contacts']:
            if hasattr(ppp.protein1, 'inner_contact_map'):
                inner_contact_map1 = ppp.protein1.inner_contact_map
            else:
                inner_contact_map1 = ppp.protein1.calc_inner_contact_map()
            if hasattr(ppp.protein2, 'inner_contact_map'):
                inner_contact_map2 = ppp.protein2.inner_contact_map
            else:
                inner_contact_map2 = ppp.protein2.calc_inner_contact_map()
        clusters=[]
        for idx, seed_ec in seeds.iterrows():

            if not seed_ec.name in [item for sublist in clusters for item in sublist]:
                #print('seed taken')
                curr_cluster = list(set(find_with_pdb(seed_ec, df_top_ecs.drop([seed_ec.name]),pdb1, pdb2, inner_contact_map1, inner_contact_map2, 0, params)))
                if len(curr_cluster) > params['cluster_min_size']: clusters.append(curr_cluster)
            else:
                pass
                #print('seed skipped')
    else:
        first_clusters=[]
        clusters=[]
        for idx,seed_ec in seeds.iterrows():
            curr_cluster=list(set(find_in_circle(seed_ec,df_top_ecs.drop([seed_ec.name]), params)))
            if len(curr_cluster)>1: first_clusters.append(curr_cluster)
        for i,c1 in enumerate(first_clusters):
            found=False
            for j,c2 in enumerate(first_clusters):
                if i<j:
                    if sorted(c1)==sorted(c2):
                        found=True
            if not found: clusters.append(c1)
        #print(first_clusters)
        #print(clusters)

        for i,curr_cluster in enumerate(clusters):
            curr_cluster2 = curr_cluster.copy()
            if len(curr_cluster)>1:
                for idx2,ec in df_top_ecs.iterrows():
                    for idx3,cluster_ec in df_top_ecs.loc[curr_cluster].iterrows():
                        if np.abs(cluster_ec['i'] - ec['i'])<params['cluster_cross_search']/2 or np.abs(cluster_ec['j'] - ec['j'])<params['cluster_cross_search']/2:
                            curr_cluster2.append(ec.name)
            clusters[i]=sorted(list(set(curr_cluster2)))
    #print(clusters)
    return clusters
            #print(list(set(curr_cluster)))


def find_in_circle(seed_ec, df_top_ecs, params):
    #helper function for calculate_clusters_pdb

    #old code
    #print(f'looking at: {seed_ec.name}')
    #curr_cluster=[seed_ec.name]
    #curr_ecs=df_top_ecs
    #for idx,ec in curr_ecs.iterrows():
    #    if np.sqrt((seed_ec['i'] - ec['i']) ** 2 + (seed_ec['j'] - ec['j']) ** 2) < params['cluster_circle_search'] and not seed_ec.name == ec.name\
    #            and ec.name in curr_ecs.index:
    #        curr_ecs = curr_ecs.drop([ec.name])
    #        curr_cluster.extend(find_in_circle(ec,curr_ecs, params))
    #return curr_cluster

    curr_cluster = [seed_ec.name]
    curr_ecs = df_top_ecs
    list_of_close = []
    for idx, ec in curr_ecs.iterrows():
        if np.sqrt((seed_ec['i'] - ec['i']) ** 2 + (seed_ec['j'] - ec['j']) ** 2) < params['cluster_circle_search'] and not seed_ec.name == ec.name:
            list_of_close.append(ec.name)

    df_of_close = curr_ecs.loc[list_of_close]
    df_without_close = curr_ecs.drop(list_of_close)

    for idx, close_ec in df_of_close.iterrows():
        found = find_in_circle(close_ec, df_without_close,params)
        df_without_close = df_without_close.drop(found, errors='ignore')
        curr_cluster.extend(found)
    return curr_cluster




def find_with_pdb(seed_ec, df_top_ecs,pdb1, pdb2,inner_contact_map1, inner_contact_map2, depth,params):
    # helper function for calculate_clusters_pdb

    #print(f'looking at: {seed_ec.name}')
    curr_cluster=[seed_ec.name]
    curr_ecs=df_top_ecs
    list_of_close=[]
    for idx,ec in curr_ecs.iterrows():
        if close_ecs_in_pdb(ec,seed_ec,pdb1, pdb2,inner_contact_map1, inner_contact_map2, params) and not seed_ec.name == ec.name:
            list_of_close.append(ec.name)
    #print(list(curr_ecs.index))
    #print(list_of_close)
    #print(curr_ecs.to_string())
    #print()
    df_of_close=curr_ecs.loc[list_of_close]
    df_without_close = curr_ecs.drop(list_of_close)
    #print(f'depth: {depth}')
    #print(f'remaining ecs: {len(df_without_close.index)}')
    for idx,close_ec in df_of_close.iterrows():
        found=find_with_pdb(close_ec, df_without_close, pdb1, pdb2, inner_contact_map1, inner_contact_map2, depth+1,params)
        df_without_close = df_without_close.drop(found, errors='ignore')
        curr_cluster.extend(found)
    return curr_cluster

def close_ecs_in_pdb(ec1, ec2, pdb1, pdb2, inner_contact_map1, inner_contact_map2, params):
    #helper function for find_with_pdb and therefore for calculate_clusters_pdb

    i1=ec1['i']
    i2=ec2['i']
    j1=ec1['j']
    j2=ec2['j']
    if not len(pdb2.get_chain_names())==1:
        print(f'pdb file {pdb2.name} does not consist of 1 chain')
    if not len(pdb1.get_chain_names())==1:
        print(f'pdb file {pdb1.name} does not consist of 1 chain')

    if params['cluster_pre_compute_inner_contacts']:
        try:
            distance_i = inner_contact_map1[i1][i2]
        except IndexError:
            distance_i = None
        try:
            distance_j = inner_contact_map2[j1][j2]
        except IndexError:
            distance_j = None
    else:
        distance_i = pdb1.chains[0].calc_distance_btw_res_heavy_atoms(i1, i2)
        distance_j = pdb2.chains[0].calc_distance_btw_res_heavy_atoms(j1, j2)
    if distance_i and distance_j: return distance_i<params['cluster_circle_search_pdb'] and distance_j<params['cluster_circle_search_pdb']
    else: return False




def calculate_clusters_old(ppp, params, df_top_ecs):
    #function to calculate clusters only based on ec data

    #if len(df_percentile.index) < 100: distance = 5
    #if len(df_percentile.index) < 80: distance = 6
    #if len(df_percentile.index) < 60: distance = 7
    #if len(df_percentile.index) < 40: distance = 8
    #if len(df_percentile.index) < 20: distance = 10
    #if len(df_percentile.index) < 10: distance = 15

    distance = params['cluster_circle_search']
    #calc all pairs within certain distance
    pairs=[]
    for idx,row in df_top_ecs.iterrows():
        for idx_2, row_2 in df_top_ecs.iterrows():
            if idx_2>idx:
                #if calc_dist(row,row_2)<distance*((row['probability']+row_2['probability'])/1.5):
                if calc_dist(row,row_2)<distance:
                    pairs.append([idx,idx_2])

    #find clusters in pairs
    #print(pairs)
    flattened_pairs = [item for sublist in pairs for item in sublist]

    #find all that occur more than once
    more_than_once = []
    for elem in flattened_pairs:
        if not elem in more_than_once:
            if flattened_pairs.count(elem)>1:more_than_once.append(elem)
    three_combinations=[]
    #print(more_than_once)
    for x, y, z in combinations(more_than_once, 3):
        if ([x, y] in pairs or [y, x] in pairs) and ([x, z] in pairs or [z, x] in pairs) and (
                [y, z] in pairs or [z, y] in pairs):
            three_combinations.append([x,y,z])

    #print(three_combinations)
    # combine all three combinations into clusters
    combined = []
    if len(three_combinations)<2:
        combined=three_combinations
    else:
        while len(three_combinations) > 0:
            first, *rest = three_combinations
            first = set(first)

            lf = -1
            while len(first) > lf:
                lf = len(first)

                rest2 = []
                for r in rest:
                    if len(first.intersection(set(r))) > 0:
                        first |= set(r)
                    else:
                        rest2.append(r)
                rest = rest2

            combined.append(list(first))
            three_combinations = rest
    return combined


def calc_dist(row1, row2):
    #helper function
    #rowx   :   row of a dataframe containing EC information (most importantly 'i' and 'j')
    #returns the distance between the 2 residue pairs
    return np.sqrt((row1['i']-row2['i'])**2+(row1['j']-row2['j'])**2)