
import os
import pandas as pd


def check_proteins(proteins_dir_path, out_file_path):
    #returns df with info on protein dataset and which files are present or missing
    columns = ['uid', 'num_pdb_files', 'has_chain_pdb', 'has_complex_pdb', 'has_AF_pdb', 'has_hm_pdb', 'has_fasta']
    results = []
    if not proteins_dir_path.endswith('/'): proteins_dir_path +='/'
    for dir in os.listdir(proteins_dir_path):
        prot_dir = proteins_dir_path + dir
        curr_results = [dir, 0, False, False, False, False, False]
        for file in os.listdir(prot_dir):
            if file.endswith('.pdb'):
                curr_results[1]+=1
            if file.endswith('.pdb') and file.count('_')==0:
                curr_results[3]=True
            if file.endswith('.pdb') and file.count('_')==1:
                curr_results[2]=True
            if file.endswith('_AF.pdb'):
                curr_results[4]=True
            if file.endswith('_hm.pdb'):
                curr_results[5]=True
            if file.endswith('.fasta'):
                curr_results[6]=True
        results.append(curr_results)
    df = pd.DataFrame(results, columns=columns).set_index('uid')
    df.to_csv(out_file_path, sep='\t')


from src.analysis_tools.interface import write_ap_interface_file
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.cluster_detection import write_cluster_file
from src.analysis_tools.modify_pdb_files import modify_all_monomer_pdb_files

def precompute_expensive_files(ppps, params, not_test_dataset):
    #precomputes the following files, if they are not present yet:
    #   interface file
    #   cluster calculation
    #   modified pdbs

    print('precompute time expensive files: ...')

    # modified pdbs + dssp files
    modify_all_monomer_pdb_files(ppps, params)
    print('monomer pdbs modified')

    #interface file only if not test dataset
    if not_test_dataset:
        write_ap_interface_file(ppps, params)
        print('interface files created')

    #cluster calculation
    from src.utils.timestamp import get_timestamp_seconds

    #no need to calculate clusters if they are not needed:
    clusters_needed=False
    for feat in params['include_features_interaction'] + params['include_features_interface']:
        if 'cluster' in feat:
            clusters_needed=True
    if clusters_needed:
        for ppp in ppps:
            write_cluster_file(ppp,params)
    print('cluster files created')

    # info for user:
    # delete all cluster files with 'find . -name \*.pkl -type f -delete'

    # info for user:
    # delete all modified files with 'find . -name \*_modified* -type f -delete'



from src.analysis_tools.ecs import get_top_ecs
def check_mandatory_inputs(ppps,params):
    #checks if all mandatory inputs are given for all ppps
    #mandatory inputs: valid ec file, protein sequence file
    #if one or more are missing for a complex it is not further considered and a warning is displayed
    # returns set of ppps that have all mandatory inputs
    new_ppps = []
    for ppp in ppps:
        if ppp.ec_file_in is None:
            print(f'WARNING: complex {ppp.name} has no readable EC file and can therefore not be considered')
        elif get_top_ecs(ppp,params) is None:
            print(f'WARNING: complex {ppp.name} has no ECs within the ranges of the monomer pdb files and can therefore not be considered')
        elif ppp.protein2.sequence_file_in is None:
            print(f'WARNING: complex {ppp.name} has no sequence file of the second monomer {ppp.protein2.uid} and can therefore not be considered')
        elif ppp.protein1.sequence_file_in is None:
            print(f'WARNING: complex {ppp.name} has no sequence file of the first monomer {ppp.protein1.uid} and can therefore not be considered')
        else:
            new_ppps.append(ppp)
    return new_ppps
