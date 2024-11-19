import copy



def modify_all_monomer_pdb_files(ppps, params):
    # generate and add modified pdb file if not present:
    # modified pdb files are optimal pdb files for the ec analysis:
    # - the residue are numbered the same as in the uniprot sequence, meaning that they match the residue numbers from ec files
    # - the pdb file is optimised, by taking either the original or alphafold, deciding on their quality measures
    for ppp in ppps:
        for p in [ppp.protein1, ppp.protein2]:
            if not hasattr(p, 'modified_pdb_file_in') or p.modified_pdb_file_in is None:
                modified_pdb_filepath = p.dir_path+p.uid+'_modified.pdb'
                modified_rsa_filepath = p.dir_path+p.uid+'_modified_rsa.csv'
                if not p.pdb_file_in is None:
                    modified_pdb, modified_rsa = modify_pdb(p.get_original_pdb_file(), p.get_sequence(),p.get_original_rsa_df(params['rsa_preferred_method']))
                    # doesnt work yet so maybe TODO?
                    # if p.AF_pdb_file_in is not None:
                    #    modified_pdb.add_missing_res(p.get_AF_pdb_file(params['pdb_af_threshold_plddt']))
                    modified_pdb.write_pdb_file(modified_pdb_filepath)
                    p.modified_pdb_file_in = modified_pdb_filepath
                    if not modified_rsa.empty: modified_rsa.to_csv(modified_rsa_filepath)
                    p.modified_rsa_file_in = modified_rsa_filepath
                    print(f'pdb file modified: {p.uid}')
                elif p.AF_pdb_file_in is not None:
                    modified_pdb = p.get_AF_pdb_file(params['pdb_af_threshold_plddt'])
                    # check size of resulting pdb file (might be small due to params['pdb_af_threshold_plddt'] )
                    # if no alphafold residue has a high enough confidence score:
                    if len(modified_pdb.chains)==0:
                        p.modified_pdb_file_in = None
                        p.modified_rsa_file_in = None
                        print(f'no modified pdb file possible (no AF residues with good plddt score): {p.uid}')
                    # if less than 10% of original remain -> ignore alphafold reuslt
                    elif len(modified_pdb.chains[0].residues) < (params['pdb_af_threshold_%_confident_res']/100) * len(p.get_sequence()):
                        p.modified_pdb_file_in = None
                        p.modified_rsa_file_in = None
                        print(f'no modified pdb file possible (too little AF residues with good plddt score): {p.uid}')
                    else:
                        modified_pdb.write_pdb_file(modified_pdb_filepath)
                        p.modified_pdb_file_in = modified_pdb_filepath
                        rsa_df = p.get_original_AF_rsa_df(params['rsa_preferred_method'])
                        if rsa_df is not None:
                            rsa_df.to_csv(modified_rsa_filepath)
                            p.modified_rsa_file_in = modified_rsa_filepath
                        print(f'Alphafold pdb file modified: {p.uid}')
                else:
                    p.modified_pdb_file_in = None
                    p.modified_rsa_file_in = None
                    print(f'no modified pdb file possible: {p.uid}')

        # if p.hm_pdb_file_in is None:
        #    print(f'no homology pdb file for {p.uid}')
        # else:
        #    modified_pdb = modify_pdb(p.get_hm_pdb_file(), p.get_sequence())
        #    p.modified_hm_pdb = modified_pdb
    # else: pairwise_align2(p.get_sequence(), p.get_pdb_file().chains[0].get_sequence()[0], p.pdb_name, p.chain)


import pandas as pd
import numpy
def pairwise_align2(query_seq, target_seq, pdb_name, chain):
    #currently unused
    pdb_2_uniprot = pd.read_csv('/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/important_files/pdb_chain_uniprot.tsv', sep='\t', skiprows=1, index_col='PDB')
    try:
        curr_df = pdb_2_uniprot.loc[pdb_name.lower()]
    except KeyError:
        print(f'skipped: {pdb_name} {chain}')
        return
    curr_df = curr_df[curr_df['CHAIN']==chain] if not isinstance(curr_df, pd.Series) else curr_df
    print(pdb_name, chain)
    try:
        pdb_beg = curr_df['PDB_BEG']
        res_beg = curr_df['RES_BEG']
        pdb_beg = pdb_beg if isinstance(pdb_beg, numpy.int64) else int(list(pdb_beg)[0])
        res_beg = res_beg if isinstance(res_beg, numpy.int64) else int(list(res_beg)[0])
    except ValueError:
        print(f'skipped: {pdb_name} {chain}')
        return
    except IndexError:
        print(f'skipped: {pdb_name} {chain}')
        return
    print(pdb_beg, res_beg)
    if pdb_beg<res_beg:
        query_seq = '-'*(res_beg-pdb_beg)+query_seq
    else:
        target_seq = '-'*(pdb_beg-res_beg)+target_seq

    print(query_seq)
    middle=''
    for x,y in zip(target_seq,query_seq):
        if x==y:
            middle+='|'
        else:
            middle+=' '
    print(middle)
    print(target_seq)
    print()


def modify_pdb(pdb_obj, uniprot_seq, rsa_df):
    # aligns the uniprot sequence to the pdb sequence and changes the residue numbers according to the original uniprot numbering

    pdb_seq, curr_res_pdb_seq = pdb_obj.chains[0].get_sequence()
    score,alignment = pairwise_align(uniprot_seq, pdb_seq)
    seq_1 = alignment[0]
    seq_2 = alignment[1]
    print(pdb_obj.name)
    # fix error of single isolated residues at the end of alignments:
    seq_2 = remove_isolated_res(seq_2)
    seq_2 = remove_isolated_res(seq_2[::-1])[::-1]
    seq_1 = remove_isolated_res(seq_1)
    seq_1 = remove_isolated_res(seq_1[::-1])[::-1]


    #print aligngment
    if True:
        middle = ''
        for x, y in zip(seq_1, seq_2):
            if x==y: middle+='|'
            elif x=='-' or y=='-': middle+=' '
            else: middle+='.'
        print(seq_1)
        print(middle)
        print(seq_2)

    #go over the alignment of both sequences res by res
    columns = ['res_num', 'All-atoms_rel']
    new_rsa = []
    curr_res_uniprot_seq=1
    list_of_res=[]
    for x,y in zip(seq_1, seq_2):
        if not y=='-':
            if not x=='-' and not y=='*':
                # update pdb file residues
                curr_res = copy.deepcopy(pdb_obj.chains[0].get_res(curr_res_pdb_seq))
                curr_res.number = curr_res_uniprot_seq
                list_of_res.append(curr_res)
                # update rsa dataframe

                #try except catches cases, where the rsa file does not have residues that are in the pdb file and sets their value to 0
                try:
                    curr_row = rsa_df.loc[curr_res_pdb_seq]
                    new_rsa.append([curr_res_uniprot_seq, curr_row['All-atoms_rel']])
                except KeyError:
                    new_rsa.append([curr_res_uniprot_seq, 0])
                except AttributeError:
                    pass
            curr_res_pdb_seq += 1
        if not x == '-':
         curr_res_uniprot_seq+=1
    pdb_obj.chains[0].residues = list_of_res
    return pdb_obj, pd.DataFrame(new_rsa, columns=columns).set_index('res_num')
import re
def remove_isolated_res(seq):
    # removes issues where single residues are added at the end or beginning
    if re.match('.+-[A-Z]{1,3}$',seq):
        new_alignment_1 = ''
        mem = ''
        skip=True
        added=False
        for c in reversed(seq):
            if skip and not c=='-':
                mem += c
            elif skip and c=='-':
                skip=False
                new_alignment_1+=c
            elif not skip and c=='-':
                new_alignment_1+=c
            elif not added:
                added=True
                new_alignment_1+=mem
                new_alignment_1+=c
            else:
                new_alignment_1+=c
        return new_alignment_1[::-1]
    else: return seq

from Bio import pairwise2
from Bio.Align import substitution_matrices

def pairwise_align(query_seq, target_seq):
    # returns global pairwise alignment of 2 sequences as well as their score.
    # the blosum62 substitution matrix is used here, as well as following gap penalties:
    # opening gap: -15, extending gap: -4

    matrix = substitution_matrices.load("BLOSUM62")
    try:
        global_align = pairwise2.align.globalds(query_seq.replace('-','*').replace('U','C'), target_seq.replace('-','*').replace('U','C'), matrix, -15, -4)
    except SystemError:
        global_align = pairwise2.align.globalds(query_seq.replace('-', '*'), target_seq.replace('-', '*'), -15, -4)
    #global_align = pairwise2.align.globalds(query_seq, target_seq, matrix, -10, -4)

    seq_length = min(len(global_align[0][0]), len(global_align[0][1]))
    matches = global_align[0][2]


    #print(format_alignment(*global_align[0]))

    return (matches / seq_length) * 100, global_align[0]