import pandas as pd
import collections
from src.analysis_tools.ecs import get_top_ecs, mark_true_ecs
from src.analysis_tools.contact_map import calculate_contact_map
import numpy as np



def print_info_on_training_set(ppps, params, out_path = None):
    # check which constraints are met in the pdb models
    # prints information on the number of ecs, the number of true ecs and so on
    # if out_path = None it will be printed in the console, otherwise it will be written in the given file

    num_ecs = 0
    num_true_ecs=0
    num_complexes=0
    num_true_ecs_per_complex={}
    out ='Info on the given training dataset:\n\n'
    for ppp in ppps:
        df_top_ecs = get_top_ecs(ppp, params)
        if not isinstance(df_top_ecs, pd.DataFrame): continue
        df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(), params['TP_EC_distance_threshold_heavy_atoms'], ppp, params)
        df_true_ecs = df_top_ecs[df_top_ecs['true_ec']]
        num_ecs+=len(df_top_ecs.index)
        num_true_ecs+=len(df_true_ecs.index)
        num_complexes+=1
        if len(df_true_ecs.index) in  num_true_ecs_per_complex: num_true_ecs_per_complex[len(df_true_ecs.index)]+=1
        else: num_true_ecs_per_complex[len(df_true_ecs.index)]=1
    out+=f'number of complexes considered: {num_complexes}\n'
    out+=f'number of ecs considered: {num_ecs}\n'
    out+=f'number of true ecs: {num_true_ecs}\n'
    out+=f'number of complexes without any true ecs: {num_true_ecs_per_complex[0]}\n'
    out+=f'number of complexes with less than 3 true ecs: {num_true_ecs_per_complex[0]+num_true_ecs_per_complex[1]+num_true_ecs_per_complex[2]}\n'
    out+=f'number of complexes with at least one ecs: {num_complexes - num_true_ecs_per_complex[0]}\n'
    out+=f'number of complexes with 3 or more true ecs: {num_complexes - (num_true_ecs_per_complex[0]+num_true_ecs_per_complex[1]+num_true_ecs_per_complex[2])}\n'
    out+=str(collections.OrderedDict(sorted(num_true_ecs_per_complex.items())))
    if out_path is None:
        print(out)
    else:
        with open(out_path,'w') as f:
            f.write(out)

def get_AP_ECs(ppp, params):
    # returns a list of 0s and 1s indicating the actual positive ECs (same indices)
    df_top_ecs = get_top_ecs(ppp, params)
    df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(), params['TP_EC_distance_threshold_heavy_atoms'], ppp, params)
    return [int(x) for x in list(df_top_ecs['true_ec'])]

def add_ap_ec_information(ppis, params):
    #adds the calculated info about ap ecs to all ppi objects
    for ppi in ppis:
        ppi.ap_ecs = get_AP_ECs(ppi.ppp, params)

def write_dataset_info(ppps, params):
    #writes in the dataset location for each complex a csv file containing basic information like skewness kurtosis heavy atom contacts, ...
    for ppp in ppps:
        columns=['name']
        data = [ppp.name]

        ca_map, heavy_atom_map = calculate_contact_map(ppp)
        if not isinstance(heavy_atom_map, np.ndarray): continue
        heavy_atom_map = [i for sublist in heavy_atom_map for i in sublist]
        dist_threshold = params['contact_map_heavy_atom_dist_threshold']
        num_contacts = sum(1 for x in heavy_atom_map if x > 0 and x < dist_threshold)
        columns.append(f'num_contacts_heavy_atoms_{dist_threshold}A')
        data.append(num_contacts)
        k, s = ppp.calc_kurtosis_and_skewness(exclude_outside_pdb=params['remove_ecs_outside_of_monomer_pdb'])
        columns.append('excess_kurtosis')
        data.append(k)
        columns.append('skewness')
        data.append(s)



        df_info = pd.DataFrame([data], columns=columns).set_index('name')
        ppp.write_info_df(df_info)