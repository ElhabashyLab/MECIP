import os
import pandas as pd

import src.analysis_tools.ecs
def get_rsa_dfs(protein, params):
    # returns dataframe with at least 'res_num' (as index) and 'All-atoms_rel' columns
    return protein.get_rsa_df()

    #old code
    #if protein.naccess_file_in and protein.dssp_file_in:
    #    rsa_preferred = params['rsa_preferred_method']
    #    if rsa_preferred == 'dssp':
    #        rsa_df_1 = protein.get_dssp_df()
    #    elif rsa_preferred == 'naccess':
    #        rsa_df_1 = protein.get_naccess_df()
    #    else:
    #        print(f'the preferred rsa method {rsa_preferred} does not exist')
    #        return None
    #elif protein.naccess_file_in:
    #        rsa_df_1 = protein.get_naccess_df()
    #elif protein.dssp_file_in:
    #        rsa_df_1 = protein.get_dssp_df()
    #else:
    #    return None
    #return rsa_df_1

def get_accessable_res_for_protein(protein, params):
    # filepath      :   String  :   filepath to the rsa file generated by naccess
    # measurement   :   String  :   type of measurement used (see colnames in 'read_naccess_file) (e.g. 'All-atoms_abs')
    # threshold     :   int     :   threshold for that measurement to be considered accessible
    # returns_ list of residue numbers that are accessible
    acc_res = []
    df = get_rsa_dfs(protein, params)
    if df is None: return None
    return df[df[params['rsa_measurement']]>params['rsa_threshold']]['res_num'].tolist()

def get_accessable_res_for_complex(ppp, params):
    # filepaths      :   [String]  :   filepaths to the rsa file generated by naccess
    # measurement   :   String  :   type of measurement used (see colnames in 'read_naccess_file) (e.g. 'All-atoms_abs')
    # threshold     :   int     :   threshold for that measurement to be considered accessible
    # returns_ 2 lists of residue numbers that are accessible
    return [get_accessable_res_for_protein(ppp.protein1, params), get_accessable_res_for_protein(ppp.protein2, params)]



def is_accessible(res_indices, complex_access, params):
    #res_indices    :   [int,int]   :   residue indices of 1 single event (e.g. 1 EC)
    #complex_access :   [[int],[int]]   :   return value from get_accessable_res_for_complex
    #returns boolean to show if given residue  is accessable
    if complex_access:
        rsa_e = params['rsa_exclude']
        if rsa_e == 'and':
            return res_indices[0] in complex_access[0] and res_indices[1] in complex_access[1]
        elif rsa_e == 'or':
            return res_indices[0] in complex_access[0] or res_indices[1] in complex_access[1]
        else:
            print(f'The parameter \'{rsa_e}\' given for \'rsa_exclude\' is not possible')
            return True #if the given 'rsa_exclude' parameter is defined wrongly, we just assume everything is accessible
    else:
        return True #if no data is available, just imagine everything is accessible

def check_accessability_of_ec_df(df_ecs, complex_access, params):
    #complex_access :   [[int],[int]]   :   return value from get_accessable_res_for_complex
    # returns df of ecs with only accessible entries
    result_ecs=[]
    for idx,row in df_ecs.iterrows():
        if is_accessible([row['i'],row['j']],complex_access, params): result_ecs.append(idx)
    return df_ecs.iloc[result_ecs]

def get_fraction_of_accessible_ecs(ppp, params):
    # returns the fraction of accessible ecs of a given complex
    complex_access = get_accessable_res_for_complex(ppp, params)
    df_top_ecs = src.analysis_tools.ecs.get_top_ecs(ppp, params)
    df_top_accessible_ecs = check_accessability_of_ec_df(df_top_ecs,complex_access, params)
    return len(df_top_accessible_ecs.index)/len(df_top_ecs)

def get_multiple_fractions_of_accessible_ecs(ppps, params):
    # returns dataframe with results
    fractions=[]
    for ppp in ppps:
        if ppp.naccess_complex_file_in and ppp.ec_file_in:
            fractions.append([ppp.name, get_fraction_of_accessible_ecs(ppp, params)])
        else :
            fractions.append([ppp.name, -1])
    return pd.DataFrame(fractions, columns=['filename', 'fractions']).set_index('filename')



