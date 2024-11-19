import numpy
import pandas as pd
import os
from src.utils.protein import Protein
from src.utils.protein_protein_pairs import PPP
import re
from src.utils.pdb_parser import parse_pdb_file
from src.dataset.check_dataset_completion import check_mandatory_inputs
import numpy as np


def convert_haddock_output_to_normal_pdb(pdb_file, chain1, chain2):
    with open(pdb_file,'r') as file:
        x = file.readlines()
    results = ''
    for y in x:
        curr_line = y
        if y.startswith('ATOM'):
            chain = chain1 if y[72] == 'A' else chain2
            l = list(curr_line)
            l[21] = chain
            curr_line = ''.join(l)
        results += curr_line
    with open(pdb_file.replace('.pdb','_modified.pdb'),'w') as file:
        file.write(results)
    return pdb_file.replace('.pdb','_modified.pdb')

def remove(strObj, index):
    if len(strObj) > index:
        strObj = strObj[0: index:] + strObj[index + 1::]
    return strObj

def get_from_dict(dict, seq1, seq2):
    if seq1 == seq2: return 100
    elif (seq2,seq1) in dict.keys(): return dict[(seq2,seq1)]
    else: return dict[(seq1,seq2)]

from Bio import pairwise2

def pairwise_align(query_seq, target_seq):
    # aligns 2 sequences with the help of the Biopython package and a global alignment method

    global_align = pairwise2.align.globalxx(query_seq, target_seq)

    seq_length = min(len(global_align[0][0]), len(global_align[0][1]))
    matches = global_align[0][2]

    return (matches / seq_length) * 100

def calc_seq_similarity(sequences):
    (_, prot1), (_, prot2) = sequences
    seq1_1 = prot1.get_sequence()
    seq1_2 = prot2.get_sequence()
    if seq1_2 is None or seq1_1 is None:
        seq_sim = 0
    else:
        seq_sim = pairwise_align(seq1_1, seq1_2)
    return ((prot1.uid, prot2.uid),seq_sim)


def read_training_dataset(params):
    ppp_dir_path = params['training_input_complexes']
    proteins_dir_path = params['training_input_proteins']
    complex_info_csv_file_path = params['training_input_complexes_csv']
    protein_info_csv_file_path = params['training_input_proteins_csv']
    return read_dataset(params, ppp_dir_path, proteins_dir_path, complex_info_csv_file_path, protein_info_csv_file_path)

def read_test_dataset(params):
    ppp_dir_path = params['test_input_complexes']
    proteins_dir_path = params['test_input_proteins']
    complex_info_csv_file_path = params['test_input_complexes_csv']
    protein_info_csv_file_path = params['test_input_proteins_csv']
    return read_dataset(params, ppp_dir_path, proteins_dir_path, complex_info_csv_file_path, protein_info_csv_file_path)

def read_dataset(params, ppp_dir_path, proteins_dir_path, complex_info_csv_file_path, protein_info_csv_file_path):
    # ppp_dir_path   :   String  :   path to directory containing all protein protein pair directories
    #                                   can contain: directory named: 'docking_results' which contains:
    #                                       a directory for each complex that was docked on, naming convention: <prefix>_<uid_1_first_part>_<uid_2_first_part> (e.g.: sepmito_00000007_DHB8_CBR4)
    #                                       contains:
    #                                           top scoring modelling results in pdb format with original naming by haddock: (example: haddock_ecs_15w.pdb)
    #                                           list of top scoring models with scores created by haddock named: structures_haddock-sorted.stat
    #                                   has to contain a protein protein pair directory for each complex:
    #                                   protein protein pair directory : naming convention: <prefix>_<uid_1_first_part>_<uid_2_first_part> (e.g.: sepmito_00000007_DHB8_CBR4)
    #                                   can contain the following:
    #                                       '*_CouplingScores_inter.csv' file, or '*_CouplingScoresCompared_inter.csv': conventionally: '<prefix>_CouplingScores_inter.csv' (e.g. 'sepmito_00000007_CouplingScores_inter.csv')
    #                                       'complex_pdb' directory containing the following:
    #                                           '<complex_pdb>_<complex_chain_1>_<complex_chain_2>.pdb' file
    #                                           '*_interface_<complex_pdb>_<complex_chain_1>_<complex_chain_2>.txt' file:
    #                                               any number of those files are possible, * defines the sort of interface analysed and must contain one '_'
    #                                               e.g. na_5A_interface_4CQL_A_B.txt for all interactions of any heavy atom closer than 5A
    #                                       '*.a2m' file: conventionally: '<prefix>.a2m' (e.g. 'sepmito_00000007.a2m'): MSA file that was used for creating the ec file
    #                                       '*_alignment_statistics.csv' file: conventionally: '<prefix>_alignment_statistics.csv' (e.g. 'sepmito_00000007_alignment_statistics.csv'): csv containing: 'num_seqs', 'seqlen', 'N_eff'
    #                                       '*_frequencies.csv' file: conventionally: '<prefix>_frequencies.csv' (e.g. 'sepmito_00000007_frequencies.csv'): csv file with columns: 'i', 'A_i', 'conservation', and frequencies for all amino acids
    #                                       TODO: x_links
    #                                       TODO: initial_model
    #                               also contains directory named 'docking_results' containing
    #                                   protein protein pair directory : naming convention: <prefix>_<uid_1_first_part>_<uid_2_first_part> (e.g.: sepmito_00000007_DHB8_CBR4)
    #                                   contains any number of files:
    #                                       '*.pdb' files that each are results from a docking run and contain 2 chains
    # proteins_dir_path :   String  :   path to directory containing all protein dirs
    #                                       protein dir: naming convention: <uid_name> (if uid name contains '_' (e.g. ZNT9_HUMAN) then <uid_first_part> = 'ZNT9', which is used later on)
    #                                       can contain the following:
    #                                           '*.fasta' file: conventionally <uid>.fasta (e.g. ZNT9_HUMAN.fasta)
    #                                           '<pdb_name>_<chain>.pdb' file: (e.g. 2ENK_A.pdb)
    #                                           '*_hm.pdb' file: conventionally <uid>_hm.pdb (e.g. ZNT9_HUMAN_hm.pdb)
    #                                           '<uid>_AF.pdb' file: (e.g. ZNT9_HUMAN_AF.pdb)
    #                                           '<pdb_name>_<chain>_dssp.out' file: dssp file for the pdb file (e.g. 2ENK_A_dssp.out)
    #                                           '<uid>_AF_dssp.out' file: dssp file for the Alphafold pdb file (e.g. ZNT9_HUMAN_AF_dssp.out)
    #                                           TODO: naccess
    # complex_info_csv_file_path    :   String  :   path to csv file containing a row for each protein protein complex with the following columns:
    #                                           'uid1'  :   uniprot id of first protein (order of proteins is important, based on order used in EC file)
    #                                           'uid2'  :   uniprot id of second protein (order of proteins is important, based on order used in EC file)
    #                                           'prefix'    :   a unique string for each protein protein complex
    #                                           'complex_pdb'   :   the pdb name for the complex containing both monomers
    #                                           'complex_chain_1'   :   the chain name of the first monomer in the complex_pdb
    #                                           'complex_chain_2'   :   the chain name of the second monomer in the complex_pdb
    #                                           'N_seq' :   the number of sequences in the used MSA TODO
    #                                           'N_eff' :   the number of effective sequences in the used MSA (similar sequences are down weighted) TODO
    #                                           'seq_length'    :   the max length of the sequences in the MSA TODO
    # protein_info_csv_file_path    :   String  :   path to csv file containing a row for each protein with the following columns:
    #                                           'uid'  :   uniprot id of protein
    #                                           '%identity'  :   identity score when obtaining pdb file based on uniprot sequence (if not present, pdb file is taken no matter what, if present it can be used as threshold
    #                                           '%align_len'  :   alignment coverage score when obtaining pdb file based on uniprot sequence (if not present, pdb file is taken no matter what, if present it can be used as threshold



    #optimise path inputs:
    if not ppp_dir_path.endswith('/'): ppp_dir_path+='/'
    if not proteins_dir_path.endswith('/'): proteins_dir_path+='/'

    #read info.csv:
    complex_info_df = pd.read_csv(complex_info_csv_file_path)

    #collect all proteins needed:
    prot_uids=[]
    ppp_prefixes = []
    ppp_dir_names = []
    for idx,row in complex_info_df.iterrows():
        prot_uids.append(row['uid1'])
        prot_uids.append(row['uid2'])
        ppp_prefixes.append(row['prefix'])
        ppp_dir_names.append(row['prefix']+'_'+row['uid1'].split('_')[0]+'_'+row['uid2'].split('_')[0])
    prot_uids = list(set(prot_uids))

    #read in protein info dataframe:
    prot_info_df = pd.read_csv(protein_info_csv_file_path)
    #read in proteins and put in dict
    prots = {}
    for prot_uid in prot_uids:
        curr_prot_path = proteins_dir_path+prot_uid+'/'
        if not os.path.isdir(curr_prot_path):
            print(f'WARNING: the protein {prot_uid} is needed for a complex, but not present as a protein directory')
            exit()

        #exclude pdbs due to thresholds:
        exclude_pdb=False
        try:
            if '%identity' in prot_info_df.columns:
                if int(prot_info_df[prot_info_df['uid']==prot_uid]['%identity'])<params['pdb_threshold_identity']:
                    #print(f'pdb excluded due to small identity:  {prot_uid}')
                    exclude_pdb=True
            if '%align_len' in prot_info_df.columns:
                if int(prot_info_df[prot_info_df['uid']==prot_uid]['%align_len'])<params['pdb_threshold_align_coverage']:
                    #print(f'pdb excluded due to small alignment coverage:  {prot_uid}')
                    exclude_pdb=True
        except ValueError:
            #print(f'pdb excluded due to no identity and alignment coverage:  {prot_uid}')
            exclude_pdb=True
        p = Protein(prot_uid)
        import math
        p.chain = prot_info_df[prot_info_df['uid']==prot_uid]['chain'].values[0]
        if (not isinstance(p.chain, str)) and math.isnan(p.chain): p.chain=None
        p.pdb_name = prot_info_df[prot_info_df['uid']==prot_uid]['pdb'].values[0]
        if (not isinstance(p.pdb_name, str)) and math.isnan(p.pdb_name): p.pdb_name = None


        # add pdb file (only if we dont exclude it)
        if not exclude_pdb:
            if os.path.isfile(f'{curr_prot_path}{p.pdb_name}_{p.chain}.pdb'):
                check_for_multiple(p, 'pdb_file_in')
                p.pdb_file_in = f'{curr_prot_path}{p.pdb_name}_{p.chain}.pdb'

                try:
                    pdb_obj = p.get_original_pdb_file()
                    # remove if less than 30 residues
                    if len(pdb_obj.chains[0].get_sequence()[0])<30:
                        print(f'pdb file {p.pdb_name}_{p.chain}.pdb from protein {p.uid} contains less than 30 residues and will not be considered')
                        p.pdb_file_in = None
                except Exception as e:
                    # rewrite pdbs with errors
                    if False:
                        chain = p.chain
                        pdb_in_file = p.pdb_file_in
                        with open(pdb_in_file) as fp:
                            Lines = fp.readlines()
                        results = ''
                        pdb_name = f'{p.pdb_name}_{p.chain}.pdb'
                        for x in Lines:
                            if x.startswith('ATOM'):
                                if x[17] == ' ':
                                    x = remove(x, 11)
                                if len(chain) == 2:
                                    x = remove(x, 20)
                                results += x
                        with open(
                                '/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/datasets/new_pdbs/' + pdb_name,
                                'w') as f:
                            f.write(results)
                    print(f'error during parsing of pdb file of {p.uid}. {p.pdb_name}_{p.chain}.pdb will not be used ')
                    p.pdb_file_in = None
            else:
                print(f'no pdb file for: {p.uid}')
        p.dir_path=curr_prot_path

        # add dssp
        if not exclude_pdb:
            if os.path.isfile(f'{curr_prot_path}{p.pdb_name}_{p.chain}_dssp.out'):
                check_for_multiple(p, 'dssp_file_in')
                p.dssp_file_in = f'{curr_prot_path}{p.pdb_name}_{p.chain}_dssp.out'
            else:
                print(f'NO DSSP: {p.uid}')


        # add Alphafold pdb file
        if os.path.isfile(f'{curr_prot_path}{prot_uid}_AF.pdb'):
            check_for_multiple(p, 'AF_pdb_file_in')
            p.AF_pdb_file_in = f'{curr_prot_path}{prot_uid}_AF.pdb'

        # add Alphafold dssp file
        if os.path.isfile(f'{curr_prot_path}{prot_uid}_AF_dssp.out'):
            check_for_multiple(p, 'AF_dssp_file_in')
            p.AF_dssp_file_in = f'{curr_prot_path}{prot_uid}_AF_dssp.out'
        elif p.AF_pdb_file_in is not None:
            print(f'alphafold dssp file missing, even though AF_pdb file exists for protein {p.uid}')

        #add files
        for file_name in os.listdir(curr_prot_path):
            # add fasta file
            if os.path.isfile(f'{curr_prot_path}{file_name}') and file_name.endswith('.fasta'):
                check_for_multiple(p,'sequence_file_in')
                p.sequence_file_in = f'{curr_prot_path}{file_name}'

            # add homology pdb file
            if os.path.isfile(f'{curr_prot_path}{file_name}') and file_name.endswith('_hm.pdb'):
                check_for_multiple(p, 'hm_pdb_file_in')
                p.hm_pdb_file_in = f'{curr_prot_path}{file_name}'

            # add modified pdb file
            if os.path.isfile(f'{curr_prot_path}{file_name}') and file_name.endswith('_modified.pdb'):
                check_for_multiple(p, 'modified_pdb_file_in')
                p.modified_pdb_file_in = f'{curr_prot_path}{file_name}'

            # add modified rsa file
            if os.path.isfile(f'{curr_prot_path}{file_name}') and file_name.endswith('_modified_rsa.csv'):
                check_for_multiple(p, 'modified_rsa_file_in')
                p.modified_rsa_file_in = f'{curr_prot_path}{file_name}'

            #TODO: naccess




        #save protein in dict
        prots[prot_uid] = p



    import itertools
    import time
    from pathos.multiprocessing import ProcessingPool as Pool
    import src.utils.timestamp as timestamp
    # compare sequence similarity
    if params['check_pairwise_sequence_similarity']:
        seq_sims={}
        seq_sims2 = []

        before = time.time()
        x = list(itertools.combinations(prots.items(), 2))
        with Pool(params['n_jobs']) as pool:
            results = pool.imap(calc_seq_similarity, x)
        results = list(results)
        print(f'time spent in pool: {time.time() - before}')
        for result in results:
            (key, seq_sim) = result
            seq_sims[key]=seq_sim
            if seq_sim>30:seq_sims2.append(key)
        print(seq_sims2)

        this_worked='''
        for (i, ((_,prot1), (_,prot2))) in enumerate(itertools.combinations(prots.items(), 2)):
            seq1_1 = prot1.get_sequence()
            seq1_2 = prot2.get_sequence()
            if seq1_2 is None or seq1_1 is None:
                seq_sim=0
            else:
                seq_sim = pairwise_align(seq1_1, seq1_2)
            seq_sims[(prot1.uid, prot2.uid)] = seq_sim'''




    # create ppps
    not_created = 0
    ppps = []
    b_list = []

    remove_list = ['sepmito_00149028', 'sepmito_01253939', 'sepmito_00001665', 'sepmito_00001639', 'sepmito_01253948',
                   'sepmito_01019523', 'sepmito_01019511', 'sepmito_01179062', 'sepmito_00292651']
    #far homology detected:
    remove_list.extend(['sepmito_00003400','sepmito_00002858','sepmito_00127067','sepmito_00131065','sepmito_00000007','sepmito_00827869'])


    for prefix,ppp_dir_name, (idx, row) in zip(ppp_prefixes, ppp_dir_names, complex_info_df.iterrows()):
        if prefix in remove_list and params['fast_read_in']:
            continue
        #positive = not math.isnan(row.X)
        a_list=[]
        curr_ppp_path = ppp_dir_path + ppp_dir_name + '/'
        if not os.path.isdir(curr_ppp_path):
            print(f'WARNING: the protein-protein pair {prefix} is one of the listed complexes, but not present as a complex directory')
            exit()
        if row['uid1'] not in prots.keys() or row['uid2'] not in prots.keys():
            not_created+=1
            continue
        ppp = PPP(prots[row['uid1']],prots[row['uid2']],prefix)

        # add files

        # add cluster file
        from src.analysis_tools.cluster_detection import generate_cluster_file_name
        cluster_file_path = f'{curr_ppp_path}/{generate_cluster_file_name(params)}'
        if os.path.isfile(cluster_file_path):
            check_for_multiple(ppp, 'cluster_file')
            ppp.cluster_file = cluster_file_path



        for file_name in os.listdir(curr_ppp_path):
            # add ec file
            if os.path.isfile(f'{curr_ppp_path}{file_name}') and (file_name.endswith('_CouplingScores_inter.csv') or file_name.endswith('_CouplingScoresCompared_inter.csv')):
            #if os.path.isfile(f'{curr_ppp_path}{file_name}') and file_name.endswith('_CouplingScoresCompared_inter.csv'):
                check_for_multiple(ppp, 'ec_file_in')
                ppp.ec_file_in = f'{curr_ppp_path}{file_name}'

            #look for complex_pdb directory
            if os.path.isdir(f'{curr_ppp_path}{file_name}') and file_name == 'complex_pdb':
                complex_dir = f'{curr_ppp_path}{file_name}/'
                complex_description = str(row['complex_pdb']) + '_' + str(row['complex_chain1']) + '_' + str(row['complex_chain2'])
                if complex_description == 'nan_nan_nan': complex_description = ''
                ppp.complex_description = complex_description
                # add complex_pdb_file
                if os.path.isfile(f'{complex_dir}{complex_description}.pdb'):
                    check_for_multiple(ppp, 'pdb_complex_file_in')
                    ppp.pdb_complex_file_in = f'{complex_dir}{complex_description}.pdb'
                # add interface files
                for filename in os.listdir(complex_dir):
                    if filename.endswith('_interface_' + complex_description + '.txt') or filename.endswith('_interface_None.txt'):
                        atom_type = filename.split('_')[0]
                        dist = filename.split('_')[1]
                        ppp.interface_files_dict[f'{atom_type}_interface_{dist}'] = f'{complex_dir}/{filename}'
            # add MSA file
            if os.path.isfile(f'{curr_ppp_path}{file_name}') and file_name.endswith('.a2m'):
                check_for_multiple(ppp, 'msa_concat_file_in')
                ppp.msa_concat_file_in = f'{curr_ppp_path}{file_name}'

            # add alignment statistics file
            if os.path.isfile(f'{curr_ppp_path}{file_name}') and file_name.endswith('_alignment_statistics.csv'):
                check_for_multiple(ppp, 'alignment_statistics_file_in')
                ppp.alignment_statistics_file_in = f'{curr_ppp_path}{file_name}'

            # add frequencies file
            if os.path.isfile(f'{curr_ppp_path}{file_name}') and file_name.endswith('_frequencies.csv'):
                check_for_multiple(ppp, 'frequencies_file_in')
                ppp.frequencies_file_in = f'{curr_ppp_path}{file_name}'

            # add couplings_standard_plmc file
            if os.path.isfile(f'{curr_ppp_path}{file_name}') and file_name.endswith('.couplings_standard_plmc.outcfg'):
                check_for_multiple(ppp, 'couplings_standard_plmc_file_in')
                ppp.couplings_standard_plmc_file_in = f'{curr_ppp_path}{file_name}'







            #TODO: xlinks
            #TODO: initial model


        # add single values: 'num_seqs', 'seqlen', 'N_eff' 'N_seq' :   the number of sequences in the used MSA TODO
        #     #                                           'N_eff' :   the number of effective sequences in the used MSA (similar sequences are down weighted) TODO
        #     #                                           'seq_length'
        import math
        if ppp.alignment_statistics_file_in is not None:
            alignment_statistics_df = pd.read_csv(ppp.alignment_statistics_file_in)
            if not math.isnan(alignment_statistics_df['seqlen'].values[0]): ppp.seq_length = int(alignment_statistics_df['seqlen'])
            if not math.isnan(alignment_statistics_df['N_eff'].values[0]): ppp.N_eff = int(alignment_statistics_df['N_eff'])
            if not math.isnan(alignment_statistics_df['num_seqs'].values[0]): ppp.N_seq = int(alignment_statistics_df['num_seqs'])
        else:
            if 'N_seq' in complex_info_df.columns: ppp.N_seq = row['N_seq']
            if 'N_eff' in complex_info_df.columns: ppp.N_eff = row['N_eff']
            if 'seq_length' in complex_info_df.columns: ppp.seq_length = row['seq_length']
        #read N_eff from plmc.outcfg file as effective samples: (3rd option to get N_eff)
        if ppp.couplings_standard_plmc_file_in is not None and ppp.N_eff is None:
            with open(ppp.couplings_standard_plmc_file_in) as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('effective_samples: '):
                    ppp.N_eff = float(line.replace('effective_samples: ', '').strip())


        a_list.append(ppp.name)
        #if ppp.ec_file_in is None and positive:
        if ppp.ec_file_in is None:
            print(f'no ec_file: {ppp.name}')
            a_list.append('x')
        else:
            a_list.append('')
        if ppp.msa_concat_file_in is None :
            print(f'no msa file: {ppp.name}')
            a_list.append('x')
        else:
            a_list.append('')
        if ppp.alignment_statistics_file_in is None :
            print(f'no alignment statistics file: {ppp.name}')
            a_list.append('x')
        else:
            a_list.append('')
        if ppp.frequencies_file_in is None :
            print(f'no frequencies file: {ppp.name}')
            a_list.append('x')
        else:
            a_list.append('')
        if ppp.N_seq is None :
            print(f'no N_seq: {ppp.name}')
            a_list.append('x')
        else:
            a_list.append('')
        if ppp.N_eff is None :
            print(f'no N_eff: {ppp.name}')
            a_list.append('x')
        else:
            a_list.append('')
        if ppp.seq_length is None :
            print(f'no seq_length: {ppp.name}')
            a_list.append('x')
        else:
            a_list.append('')


        #read in docking results:
        curr_docking_path = ppp_dir_path+'docking_results/'+ppp_dir_name+'/'
        if os.path.exists(curr_docking_path):
            curr_filepaths = []
            for file in os.listdir(curr_docking_path):
                if file.endswith('_modified.pdb'):
                    curr_filepaths.append(curr_docking_path+file)
                elif file.endswith('.pdb'):
                    if not os.path.isfile(curr_docking_path+file.replace('.pdb', '_modified.pdb')):
                        curr_filepaths.append(convert_haddock_output_to_normal_pdb(curr_docking_path+file, 'A', 'B'))
                elif file == 'structures_haddock-sorted.stat':
                    df = pd.read_csv(curr_docking_path+file, sep=' ')
                    ppp.haddock_result_best = curr_docking_path + (df.head(1)['#struc']).to_string().split(' ')[-1]
                    ppp.haddock_score_best = float(df.head(1)['haddock-score'])
                    ppp.haddock_score_average = float(np.mean(df.head(10)['haddock-score']))
            ppp.haddock_results = list(set(curr_filepaths))
        if params['only_use_complexes_with_docking_results']:
            if len(ppp.haddock_results)>0:
                ppps.append(ppp)
        else:
            ppps.append(ppp)
        b_list.append(a_list)

    pd.DataFrame(b_list, columns=['prefix','ec_file_compared', 'msa_file','alignment_statistics_file','frequencies_file','N_seq','N_eff','seq_length']).sort_values(by=['prefix']).set_index('prefix').to_csv('still_missing.csv')





    if not params['fast_read_in']:
        ppps = check_mandatory_inputs(ppps, params)
        print(f'not created: {not_created}')

    # check for far homology
    from src.analysis_tools.far_homology import detect_multiple_far_homology
    from src.analysis_tools.ecs import get_top_ecs
    if params['check_for_far_homology']:
        new_ppps = []
        for ppp in ppps:
            (frac, modes) = detect_multiple_far_homology(get_top_ecs(ppp, params))
            if frac>.7:
                print(f'Possible far homology detected in complex {ppp.name}. It will be removed in further computations.')
            else:
                new_ppps.append(ppp)
        ppps = new_ppps



    #update params['num_observations']
    params['num_observations'] = len(ppps)

    import itertools
    import src.utils.timestamp as timestamp
    # compare sequence similarity
    if params['check_pairwise_sequence_similarity']:
        for (i,(ppp1, ppp2)) in enumerate(itertools.combinations(ppps,2)):

            seq1_1=ppp1.protein1.uid
            seq1_2=ppp1.protein2.uid
            seq2_1=ppp2.protein1.uid
            seq2_2=ppp2.protein2.uid

            seq_sim_1 = get_from_dict(seq_sims, seq1_1, seq2_1)
            seq_sim_2 = get_from_dict(seq_sims, seq1_2, seq2_1)
            seq_sim_3 = get_from_dict(seq_sims, seq1_1, seq2_2)
            seq_sim_4 = get_from_dict(seq_sims, seq1_2, seq2_2)
            seq_sim_threshold = params['pairwise_sequence_similarity_percent_threshold']
            if (seq_sim_1>seq_sim_threshold and seq_sim_4>seq_sim_threshold):
                print(f'WARNING: might be similar: {ppp1.name}, {ppp2.name}: ({seq1_1},{seq2_1}):{seq_sim_1}%; ({seq1_2},{seq2_2}):{seq_sim_4}%')
            if (seq_sim_2>seq_sim_threshold and seq_sim_3>seq_sim_threshold):
                print(f'WARNING: might be similar: {ppp1.name}, {ppp2.name}: ({seq1_2},{seq2_1}):{seq_sim_2}%; ({seq1_1},{seq2_2}):{seq_sim_3}%')


    return ppps








def check_for_multiple(obj, attr):
    # checks if an object already has a certain attribute and prints results
    if hasattr(obj,attr) and getattr(obj,attr) is not None:
        name = str(obj)
        if hasattr(obj, 'name'): name=obj.name
        if hasattr(obj, 'uid'): name=obj.uid
        print(f'for object {name} multiple inputs have been detected for: {attr}')



