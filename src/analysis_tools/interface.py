import pandas as pd
from src.analysis_tools.contact_map import calculate_contact_map
import os
import numpy as np

def read_interface_file(ppp, params):
    #params needed:
    #TP_interaction_atom_type  :   String  :   which kind of atom should be analysed? (ca or na)
    #TP_interaction_distance_threshold   :   int     :   distance threshold
    # returns the read interface file as a Dataframe
    atom_type = params['TP_interaction_atom_type']
    distance = params['TP_interaction_distance_threshold']

    if not isinstance(ppp.interface_files_dict, dict):
        print(f'For complex {ppp.name} no interaction files are present')
        return None
    filepath = ppp.interface_files_dict.get(f'{atom_type}_interface_{distance}A')
    if not filepath:
        print(f'For complex {ppp.name} no interface file for the parameters {atom_type}_interface_{distance}A is present')
        print(f'The following parameters are possible: {ppp.interface_files_dict.keys()}')
        print('For this complex no validation based on its interface can be done. It is advised to first calculate all interface files.')
        return None
    return pd.read_csv(filepath, sep='\t')

def get_ap_interaction(ppp, params):
    # returns boolean weather given interaction is actual positive interactions
    #params needed:
    #TP_interaction_atom_type  :   String  :   which kind of atom should be analysed? (ca or na)
    #TP_interaction_distance_threshold   :   int     :   distance threshold
    #TP_interaction_num_interface_needed_threshold   :   int     :   how many interactions of the properties above are needed so that the complex is called interacting
    df_interface = read_interface_file(ppp, params)
    if isinstance(df_interface, pd.DataFrame): return (len(df_interface)>=params['TP_interaction_num_interface_needed_threshold'])
    return None


def get_ap_interactions(ppps, params):
    # returns list of booleans indicating if ppps are actual positive interactions
    #params needed:
    #TP_interaction_atom_type  :   String  :   which kind of atom should be analysed? (ca or na)
    #TP_interaction_distance_threshold   :   int     :   distance threshold
    #TP_interaction_num_interface_needed_threshold   :   int     :   how many interactions of the properties above are needed so that the complex is called interacting
    results = []
    for ppp in ppps:
        results.append(get_ap_interaction(ppp,params))
    return results

def write_ap_interface_file(ppps, params):
    # writes a interface file and saves it directly in ppp object for given parameters if it doesnt exist yet
    # loaction of interface file: <location of ec-file>/complex_pdb/<interface_file_name>

    atom_type = params['TP_interaction_atom_type']
    if not (atom_type=='na' or atom_type=='ca'):
        print(f'unknown atom type: {atom_type}')
        return
    distance = params['TP_interaction_distance_threshold']
    columns = 'chain1	resno1	resid1	chain2	resno2	resid2	na_atom1	na_atom2	na_dist	ca_dist	cb_dist'
    for ppp in ppps:
        results=[]
        results.append(columns)
        filepath = ppp.ec_file_in
        if not filepath:
            print(f'Warning: no interface file can be generated for {ppp.name}, since no ec file exists')
            continue
        filepath = '/'.join(filepath.split('/')[:-1])+f'/complex_pdb/'
        os.makedirs(filepath, exist_ok=True)
        existing_filepath=''
        for file_name in os.listdir(filepath):
            if file_name.startswith(f'{atom_type}_{distance}A_interface_') and file_name.endswith('.txt'):
                existing_filepath= filepath + file_name
                break
        if os.path.isfile(existing_filepath): continue
        # if file does not exist yet:
        complex_description = ppp.complex_description
        # if no complex_pdb exists:
        if complex_description == '':
            filepath += f'/{atom_type}_{distance}A_interface_None.txt'
            with open(filepath, 'w') as f:
                for x in results:
                    f.write(x + '\n')
            ppp.interface_files_dict[f'{atom_type}_interface_{distance}'] = filepath
            print(f'interface generated (negative): {ppp.name}')
        # if complex pdb exists:
        else:
            chain1 = complex_description.split('_')[1]
            chain2 = complex_description.split('_')[2]
            filepath += f'/{atom_type}_{distance}A_interface_{complex_description}.txt'
            dist, close_dist = calculate_contact_map(ppp)
            if not isinstance(dist, np.ndarray):
                print(f'Warning: no interface file can be generated for {ppp.name}_{complex_description}, since no pdb complex file exists')
                continue
            for i,(ca_row, na_row) in enumerate(zip(dist, close_dist)):
                for j,(ca_cell, na_cell) in enumerate(zip(ca_row, na_row)):
                    if atom_type=='na':
                        curr_cell = na_cell
                    if atom_type=='ca':
                        curr_cell = ca_cell
                    if not curr_cell==0 and curr_cell<=distance:
                        #TODO include missing values
                        resid1='?'
                        resid2='?'
                        na_atom1='?'
                        na_atom2='?'
                        cb_cell='?'
                        results.append(f'{chain1}\t{i}\t{resid1}\t{chain2}\t{j}\t{resid2}\t{na_atom1}\t{na_atom2}\t{na_cell}\t{ca_cell}\t{cb_cell}')
            with open(filepath, 'w') as f:
                for x in results:
                    f.write(x+'\n')
            ppp.interface_files_dict[f'{atom_type}_interface_{distance}']=filepath


            print(f'interface generated: {ppp.name}_{complex_description}')

