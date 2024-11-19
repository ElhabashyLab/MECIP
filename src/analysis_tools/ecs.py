import pandas as pd
from src.analysis_tools.solvent_access_ability import check_accessability_of_ec_df, get_accessable_res_for_complex

def get_top_ecs(ppp, params):
    # returns a dataframe containing either the top percentile, or the top_x of the dataframe, whichever is bigger
    # it also can exclude ecs if the rsa is too small
    # it can also exclude ecs if they are not within the range of the monomer pdbs (see params)

    df_ecs = ppp.get_ecs(exclude_outside_pdb=params['remove_ecs_outside_of_monomer_pdb'])
    if not isinstance(df_ecs, pd.DataFrame) or df_ecs.empty:
        print(f'WARNING: mandatory EC input for complex {ppp.name} not present. This can likely cause errors.')
        return None
    percentile = params['ecs_consider_percentile']
    top_x = params['ecs_consider_top_x']
    rsa_exclude = params['rsa_exclude']

    if rsa_exclude: df_ecs = check_accessability_of_ec_df(df_ecs, get_accessable_res_for_complex(ppp, params), params)

    df_ecs = df_ecs.sort_values(by=['cn'], ascending=False)
    top_x_percentile = df_ecs.head(int(len(df_ecs.index)*(1-percentile)))
    if len(top_x_percentile)>=top_x:
        return top_x_percentile
    else:
        return df_ecs.head(top_x)


def get_percentile_or_value_seeds(df, params):
    # returns a dataframe containing either the top percentile, or the top_x of the dataframe, whichever is bigger (this time looking at the seed parameters)
    # it also can exclude ecs if the rsa is too small
    percentile = params['cluster_ecs_seed_percentile']
    top_x = params['cluster_ecs_seed_top_x']
    rsa_exclude = params['rsa_exclude']

    if not isinstance(df, pd.DataFrame):
        return None
    df = df.sort_values(by=['cn'], ascending=False)
    top_x_percentile = df.head(int(len(df.index) * (1 - percentile)))
    if len(top_x_percentile) >= top_x:
        return top_x_percentile
    else:
        return df.head(top_x)

from src.analysis_tools.interface import read_interface_file

def mark_true_ecs(df_ecs, pdb_file, distance_threshold, ppp, params):
    # returns df with additional column ('true_ec') that states if the ec is a close connection according to the 'dist' value (if a 'dist' value is present, otherwise based on 'calc_dist')
    # returns df with additional column ('calc_dist') that states the calculated distance from the given pdb file (if no 'dist' column was present)
    if 'true_ec' in list(df_ecs.columns):
        return df_ecs
    if 'dist' in list(df_ecs.columns):
        df_ecs['true_ec'] = df_ecs.dist<distance_threshold
    else:
        results = []
        if not isinstance(df_ecs, pd.DataFrame):
            return None
        for idx, row in df_ecs.iterrows():
            if not pdb_file:
                interface_df = read_interface_file(ppp, params)
                #TODO: check if this works
                if not (interface_df[(interface_df['resno1']==row['i']) & (interface_df['resno2']==row['j'])]).empty:
                    results.append([row.name, float(interface_df['na_dist'][(interface_df['resno1']==row['i']) & (interface_df['resno2']==row['j'])]), False])
                else:
                    results.append([row.name, None, False])
            else:
                res_i = pdb_file.chains[0].get_res(row['i'])
                res_j = pdb_file.chains[1].get_res(row['j'])
                if res_i and res_j:
                    distance = res_i.calc_heavy_atom_distance_to(res_j)
                else:
                    distance = None
                if not distance:
                    results.append([row.name, distance, False])
                else:
                    results.append([row.name, distance, distance < distance_threshold])
        df_ecs = pd.concat([df_ecs, pd.DataFrame(results, columns=['id', 'calc_dist', 'true_ec']).set_index('id')],
                           axis=1)
        #df_ecs = df_ecs.rename(columns={'calc_true_ec' : 'true_ec'})
    return df_ecs
