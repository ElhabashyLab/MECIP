import os
from .protein import Protein
import pandas as pd
from.pdb_parser import parse_pdb_file
from scipy.stats import kurtosis, skew

class PPP():
    def __init__(self, protein1 : Protein, protein2 : Protein, name, pdb_complex_file_in=None, ec_file_in=None,
                 msa_concat_file_in=None, xlms_file_in=None, haddock_results=[], alphafold_results_dict={},
                 rosetta_results_dict = None, naccess_complex_file_in=None, fasta_complex_file_in=None,
                 dssp_complex_file_in=None, interface_files_dict={}):
        # protein1  :   Protein :   first protein of this protein pair (for all further analysis data it is assumed,
        #                           that this is the first protein of this interaction)
        # protein2  :   Protein :   second protein of this protein pair (for all further analysis data it is assumed,
        #                           that this is the second protein of this interaction)
        # name      :   String  :   name for the given interaction (sometimes also referred to as index)
        # pdb_complex_file_in   :   String  :   absolute filepath to the pdb file containing the structure of the
        #                                       protein pair complex in question
        # ec_file_in   :   String  :   absolute filepath to the evolutionary coupling file created by EVComplex
        # msa_concat_file_in   :   String  :   absolute filepath to the file containing the concatenated multiple
        #                                      sequence alignment of the two proteins and orthologs from different
        #                                      proteomes
        # xlms_file_in   :   String  :   absolute filepath to the cross linking MS results file
        # haddock_results   :   [String]  :   list of filepaths containing all results of HADDOCK
        # alphafold_results_dict   :   dict  :   dictionary containing all results of alphafold multimer
        # rosetta_results_dict   :   dict  :   dictionary containing all results of RosettaFold
        # naccess_complex_file_in   :   String  :   absolute filepath to the Naccess file of the complex
        # fasta_complex_file_in   :   String  :   absolute filepath to the fasta file of the complex
        # dssp_complex_file_in  :   String  :   absolute filepath to the file containing the dssp results of the complex
        # interface_files_dict  :   dict    :   dict containing filepaths to interface files with different params
        # complex_description   :   String  :   <complex_pdb_name>_<complex_chain_1>_<complex_chain_2>
        # cluster_file   :   String  :   filepath to cluster file, containing cluster information:

        self.protein1 = protein1
        self.protein2 = protein2
        self.name = name
        self.pdb_complex_file_in = pdb_complex_file_in
        self.ec_file_in = ec_file_in
        self.msa_concat_file_in = msa_concat_file_in
        self.xlms_file_in = xlms_file_in
        self.haddock_results = []
        self.haddock_result_best = None
        self.haddock_score_best = None
        self.haddock_score_average = None
        self.alphafold_results_dict = {}
        self.rosetta_results_dict = rosetta_results_dict
        self.naccess_complex_file_in = naccess_complex_file_in
        self.fasta_complex_file_in = fasta_complex_file_in
        self.dssp_complex_file_in = dssp_complex_file_in
        self.interface_files_dict = {}
        self.computed_info_df = None
        self.complex_description = ''
        self.cluster_file = None
        self.alignment_statistics_file_in = None
        self.frequencies_file_in = None
        self.couplings_standard_plmc_file_in = None
        self.seq_length = None
        self.N_eff = None
        self.N_seq = None

    def get_ecs(self, exclude_outside_pdb=True,  print_number_of_dropped=False):
        # returns df of the ecs known
        # if pdb files of the single proteins are given and 'exclude_outside_pdb' is True, the residue ranges are cropped accordingly.
        # This can result in less ECs since some might be located outside of the residue ranges given by the pdbs
        if not exclude_outside_pdb: return self.get_all_ecs()
        else:
            if not self.ec_file_in: return None
            pdb1 = self.protein1.get_pdb_file()
            pdb2 = self.protein2.get_pdb_file()
            if pdb1 and pdb2:
                pdb1_res_list = pdb1.chains[0].get_res_number_list()
                pdb2_res_list = pdb2.chains[0].get_res_number_list()
                df = pd.read_csv(self.ec_file_in)
                before = len(df.index)
                df = df.drop(df[~df['i'].isin(pdb1_res_list)].index)
                df = df.drop(df[~df['j'].isin(pdb2_res_list)].index)
                if print_number_of_dropped: print(f'number of dropped ECs for {self.name}: {before-len(df.index)}')
                return df
            else:
                return pd.read_csv(self.ec_file_in)

    def get_all_ecs(self):
        # returns all ecs (not only the fitting and top ones)
        if self.ec_file_in: return pd.read_csv(self.ec_file_in)
        else: return None

    def get_pdb_file(self):
        # returns the complex pdb file in PdbFile object format
        if self.pdb_complex_file_in: return parse_pdb_file(self.pdb_complex_file_in)
        else: return None

    def get_naccess_df(self):
        # filepath  :   String  :   filepath to the rsa file generated by naccess
        # returns: dataframe containing all information
        colnames = ['type', 'residue_name', 'chain', 'res_num', 'All-atoms_abs', 'All-atoms_rel', 'Total-Side_abs',
                    'Total-Side_rel', 'Main-Chain_abs', 'Main-Chain_rel', 'Non-polar_abs', 'Non-polar_rel',
                    'All-polar_abs', 'All polar_rel']
        df = pd.read_csv(self.naccess_complex_file_in, skiprows=4, delimiter=r"\s+", header=None, skipfooter=4, names=colnames)
        return df

    def calc_skewness(self, exclude_outside_pdb):
        # calculates skewness (and first gets ecs, therefore if skewness and kurtosis are needed, use 'calc_kurtosis_and_skewness' instead)
        ecs = self.get_ecs(exclude_outside_pdb=exclude_outside_pdb)
        if not isinstance(ecs, pd.DataFrame):
            print(f'index {self.name} has no valid EC file')
            return None
        if ecs.empty:
            print(f'index {self.name} has no ECs computed (that are also present in both pdb files)')
            return None
        self.skew = skew(ecs['cn'])
        return self.skew

    def calc_kurtosis(self, exclude_outside_pdb):
        # calculates kurtotis (and first gets ecs, therefore if skewness and kurtosis are needed, use 'calc_kurtosis_and_skewness' instead)
        # automatically computes excess kurtosis (meaning a normalised version, which is 3 less than the normal variant)
        ecs = self.get_ecs(exclude_outside_pdb = exclude_outside_pdb)
        if not isinstance(ecs, pd.DataFrame):
            print(f'index {self.name} has no valid EC file')
            return None
        if ecs.empty:
            print(f'index {self.name} has no ECs computed (that are also present in both pdb files)')
            return None
        self.kurt = kurtosis(ecs['cn'])
        return self.kurt

    def calc_kurtosis_and_skewness(self, exclude_outside_pdb):
        # returns skewness and kurtotsis
        # automatically computes excess kurtosis (meaning a normalised version, which is 3 less than the normal variant)
        if hasattr(self,'kurt') and hasattr(self,'skew'):
            return self.kurt,self.skew

        ecs = self.get_ecs(exclude_outside_pdb = exclude_outside_pdb)
        if not isinstance(ecs, pd.DataFrame):
            print(f'index {self.name} has no valid EC file')
            return None,None
        if ecs.empty:
            print(f'index {self.name} has no ECs computed (that are also present in both pdb files)')
            return None, None
        ecs_cn = ecs['cn']
        self.kurt = kurtosis(ecs_cn)
        self.skew = skew(ecs_cn)
        return self.kurt,self.skew
    def give_info_on_known_data(self):
        #returns information, which data is known for this complex
        col_names = []
        values = []
        ppp_attributes = self.__dict__
        p1_attributes = self.protein1.__dict__
        p2_attributes = self.protein2.__dict__
        for key in ppp_attributes:
            col_names.append(key)
            values.append(int(bool(ppp_attributes[key])))
        for key in p1_attributes:
            col_names.append('protein1 '+key)
            values.append(int(bool(p1_attributes[key])))
        for key in p2_attributes:
            col_names.append('protein2 '+key)
            values.append(int(bool(p2_attributes[key])))
        return pd.DataFrame([values], columns=col_names).set_index('name')

        #old code
        col_names = ['name', 'protein1_pdb_name', 'protein1_chain', 'protein1_sequence_file', 'protein1_pdb_file', 'protein1_AF_pdb_file', 'protein1_hm_pdb_file',
                     'protein1_naccess_file', 'protein1_msa_file', 'protein1_dssp_file',
                     'protein2_pdb_name', 'protein2_chain', 'protein2_sequence_file', 'protein2_pdb_file', 'protein2_AF_pdb_file', 'protein2_hm_pdb_file',
                     'protein2_naccess_file', 'protein2_msa_file', 'protein2_dssp_file',
                     'complex_name', 'complex_pdb_file', 'complex_ec_file', 'complex_msa_file', 'complex_xlms_file',
                     'complex_haddock_results', 'complex_alphafold_results', 'complex_rosetta_results',
                     'complex_naccess_file', 'complex_fasta_file', 'complex_dssp_file', 'complex_interface_file', 'complex_cluster_file'
                     ]
        results=[]
        results.append(self.name)
        if self.protein1.pdb_name: results.append(1)
        else: results.append(0)
        if self.protein1.chain: results.append(1)
        else: results.append(0)
        if self.protein1.sequence_file_in: results.append(1)
        else: results.append(0)
        if self.protein1.pdb_file_in: results.append(1)
        else: results.append(0)
        if self.protein1.AF_pdb_file_in: results.append(1)
        else: results.append(0)
        if self.protein1.hm_pdb_file_in: results.append(1)
        else: results.append(0)
        if self.protein1.naccess_file_in: results.append(1)
        else: results.append(0)
        if self.protein1.msa_file_in: results.append(1)
        else: results.append(0)
        if self.protein1.dssp_file_in: results.append(1)
        else: results.append(0)
        if self.protein2.pdb_name: results.append(1)
        else: results.append(0)
        if self.protein2.chain: results.append(1)
        else: results.append(0)
        if self.protein2.sequence_file_in: results.append(1)
        else: results.append(0)
        if self.protein2.pdb_file_in: results.append(1)
        else: results.append(0)
        if self.protein2.AF_pdb_file_in: results.append(1)
        else: results.append(0)
        if self.protein2.hm_pdb_file_in: results.append(1)
        else: results.append(0)
        if self.protein2.naccess_file_in: results.append(1)
        else: results.append(0)
        if self.protein2.msa_file_in: results.append(1)
        else: results.append(0)
        if self.protein2.dssp_file_in: results.append(1)
        else: results.append(0)
        if self.name: results.append(1)
        else: results.append(0)
        if self.pdb_complex_file_in: results.append(1)
        else: results.append(0)
        if self.ec_file_in: results.append(1)
        else: results.append(0)
        if self.msa_concat_file_in: results.append(1)
        else: results.append(0)
        if self.xlms_file_in: results.append(1)
        else: results.append(0)
        if self.haddock_results: results.append(1)
        else: results.append(0)
        if self.alphafold_results_dict: results.append(1)
        else: results.append(0)
        if self.rosetta_results_dict: results.append(1)
        else: results.append(0)
        if self.naccess_complex_file_in: results.append(1)
        else: results.append(0)
        if self.fasta_complex_file_in: results.append(1)
        else: results.append(0)
        if self.dssp_complex_file_in: results.append(1)
        else: results.append(0)
        if self.interface_files_dict: results.append(1)
        else: results.append(0)
        if self.cluster_file: results.append(1)
        else: results.append(0)

        return pd.DataFrame([results], columns=col_names).set_index('name')
    def write_info_df(self, df):
        path = '/'.join(self.ec_file_in.split('/')[:-2])+'/info.csv'
        df.to_csv(path)

def give_info_on_known_data(ppps, out_file=None):
    # returns dataframe with combined info on all ppps and prints sum over all
    curr_df=None
    for ppp in ppps:
        if isinstance(curr_df, pd.DataFrame):
            curr_df = pd.concat([curr_df, ppp.give_info_on_known_data()])
        else:
            curr_df = ppp.give_info_on_known_data()
    if out_file:
        with open(out_file, 'w') as f:
            f.write(f'number of complexes observed: {len(ppps)}\n\n')
            f.write(str(curr_df.sum(axis=0)))
    else:
        # print summed info
        print(f'number of complexes observed: {len(ppps)}')
        print(curr_df.sum(axis=0))



    return curr_df




############################  WARNING   ###########################################
# The following code block will be replaced soon
# No further changes will be made here and it only works for old datasets
# No further support guaranteed




def read_multiple_ppp(pair_dir_in):
    #TODO:  msa_file_in from protein, msa_concat_file_in, xlms_file_in, haddock_results_dict, alphafold_results_dict,
    #       rosetta_results_dict from PPP

    # pair_dir_in   :   String  :   absolute path to directory containing the following:
    #                                   x folders each dedicated to one protein-protein pair each
    #                                   naming convention: <name>_<pdb_id>_<chain1>_<chain2> (e.g.: all0001_4BV4_R_M)
    #                                   each dir can contain the following:
    #                                       fasta and pdb file named <pdb_id>_<chain>.<fasta|pdb> for each monomer
    #                                           example: 4BV4_M.fasta, 4BV4_M.pdb, 4BV4_R.fasta, 4BV4_R.pdb
    #                                       fasta and pdb file named <pdb_id>_<chain1>_<chain2>.<fasta|pdb> for complex
    #                                           example: 4BV4_R_M.fasta, 4BV4_R_M.pdb
    #                                       dir containing 'naccess' (example: naccess_4BV4_R_M) containing:
    #                                           rsa file named <pdb_id>_<chain>.rsa for each monomer
    #                                           rsa file named <pdb_id>_<chain1>_<chain2>.rsa for complex
    #                                       dir containing 'dssp' (example: dssp_4BV4_R_M) containing:
    #                                           dssp file named <pdb_id>_<chain>.dssp for each monomer
    #                                           dssp file named <pdb_id>_<chain1>_<chain2>.dssp for complex
    #                                       dir containing 'interface' (example: interface_4BV4_R_M) containing:
    #                                           txt file named na_interface_6A_<pdb_id>_<chain1>_<chain2>.txt
    #                                           txt file named ca_interface_15A_<pdb_id>_<chain1>_<chain2>.txt
    #                                       dir containing 'evcomplex' (example: evcomplex_4BV4_R_M) containing:
    #                                           csv named <pdb_id>_<chain1>_<chain2>_CouplingsScoresCompared_inter.csv
    #                                               (instead of <pdb_id>_<chain1>_<chain2>, <name> is also possible)

    if not pair_dir_in.endswith('/'):pair_dir_in+='/'
    ppps=[]
    # iterate over all ppp directories
    for ppp_dir_path in os.listdir(pair_dir_in):

        ppp = read_single_ppp(pair_dir_in+ppp_dir_path)

        if ppp: ppps.append(ppp)
    return ppps


def read_single_ppp(ppp_dir_path):
    #TODO:  msa_file_in from protein, msa_concat_file_in, xlms_file_in, haddock_results_dict, alphafold_results_dict,
    #       rosetta_results_dict from PPP

    # ppp_dir_path   :   String  :   path to directory
    #                                   naming convention: <name>_<pdb_id>_<chain1>_<chain2> (e.g.: all0001_4BV4_R_M)
    #                                   dir can contain the following:
    #                                       fasta and pdb file named <pdb_id>_<chain>.<fasta|pdb> for each monomer
    #                                           example: 4BV4_M.fasta, 4BV4_M.pdb, 4BV4_R.fasta, 4BV4_R.pdb
    #                                       fasta and pdb file named <pdb_id>_<chain1>_<chain2>.<fasta|pdb> for complex
    #                                           example: 4BV4_R_M.fasta, 4BV4_R_M.pdb
    #                                       dir containing 'naccess' (example: naccess_4BV4_R_M) containing:
    #                                           rsa file named <pdb_id>_<chain>.rsa for each monomer
    #                                           rsa file named <pdb_id>_<chain1>_<chain2>.rsa for complex
    #                                       dir containing 'dssp' (example: dssp_4BV4_R_M) containing:
    #                                           dssp file named <pdb_id>_<chain>.dssp for each monomer
    #                                           dssp file named <pdb_id>_<chain1>_<chain2>.dssp for complex
    #                                       dir containing 'interface' (example: interface_4BV4_R_M) containing:
    #                                           txt file named na_interface_6A_<pdb_id>_<chain1>_<chain2>.txt
    #                                           txt file named ca_interface_15A_<pdb_id>_<chain1>_<chain2>.txt
    #                                       dir containing 'evcomplex' (example: evcomplex_4BV4_R_M) containing:
    #                                           csv named <pdb_id>_<chain1>_<chain2>_CouplingsScoresCompared_inter.csv
    #                                               (instead of <pdb_id>_<chain1>_<chain2>, <name> is also possible)

    #get folder name
    ppp_dir_name=ppp_dir_path.split('/')[-1]

    # check if naming conventions of folders are met
    if ppp_dir_name.count('_') < 3:
        print(f'directory {ppp_dir_name} does not have at least three \"_\" in its name and does therefore not confine to '
              f'naming conventions')
        return None

    # analyse folder name
    # TODO: what if no pdb files are given? then we caan not have this folder name and no id, chains are known
    chain2 = ppp_dir_name.split('_')[-1]
    chain1 = ppp_dir_name.split('_')[-2]
    id = ppp_dir_name.split('_')[-3]
    name = '_'.join(ppp_dir_name.split('_')[:-3])

    # create protein objects and ppp object
    protein1 = Protein(id, chain=chain1)
    protein2 = Protein(id, chain=chain2)
    ppp = PPP(protein1, protein2, name)

    # add all existing info into ppp object
    if os.path.isfile(
        f'{ppp_dir_path}/{id}_{chain1}.pdb'): ppp.protein1.pdb_file_in = f'{ppp_dir_path}/{id}_{chain1}.pdb'
    if os.path.isfile(
        f'{ppp_dir_path}/{id}_{chain1}.fasta'): ppp.protein1.sequence_file_in = f'{ppp_dir_path}/{id}_{chain1}.fasta'
    if os.path.isfile(
        f'{ppp_dir_path}/{id}_{chain2}.pdb'): ppp.protein2.pdb_file_in = f'{ppp_dir_path}/{id}_{chain2}.pdb'
    if os.path.isfile(
        f'{ppp_dir_path}/{id}_{chain2}.fasta'): ppp.protein2.sequence_file_in = f'{ppp_dir_path}/{id}_{chain2}.fasta'
    if os.path.isfile(
        f'{ppp_dir_path}/{id}_{chain1}_{chain2}.pdb'): ppp.pdb_complex_file_in = f'{ppp_dir_path}/{id}_{chain1}_{chain2}.pdb'
    if os.path.isfile(
        f'{ppp_dir_path}/{id}_{chain1}_{chain2}.fasta'): ppp.fasta_complex_file_in = f'{ppp_dir_path}/{id}_{chain1}_{chain2}.fasta'
    if os.path.isfile(
        f'{ppp_dir_path}/info.csv'): ppp.computed_info_df = pd.read_csv(f'{ppp_dir_path}/info.csv').set_index('name')

    # check for dir containing 'naccess', 'dssp
    found_naccess = False
    found_dssp = False
    found_interface = False
    found_ev_complex = False
    for dirname in os.listdir(ppp_dir_path):
        if os.path.isdir(f'{ppp_dir_path}/{dirname}'):
            #ignore upper and lower case
            lower_dirname = str.lower(dirname)
            if 'naccess' in lower_dirname:
                if found_naccess: print(f'for directory {ppp_dir_path} multiple directories for naccess data were found')
                found_naccess = True
                if os.path.isfile(
                    f'{ppp_dir_path}/{dirname}/{id}_{chain1}_{chain2}.rsa'): ppp.naccess_complex_file_in = f'{ppp_dir_path}/{dirname}/{id}_{chain1}_{chain2}.rsa'
                if os.path.isfile(
                    f'{ppp_dir_path}/{dirname}/{id}_{chain1}.rsa'): ppp.protein1.naccess_file_in = f'{ppp_dir_path}/{dirname}/{id}_{chain1}.rsa'
                if os.path.isfile(
                    f'{ppp_dir_path}/{dirname}/{id}_{chain2}.rsa'): ppp.protein2.naccess_file_in = f'{ppp_dir_path}/{dirname}/{id}_{chain2}.rsa'
            if 'dssp' in lower_dirname:
                if found_dssp: print(
                    f'for directory {ppp_dir_path} multiple directories for dssp data were found')
                found_dssp = True
                if os.path.isfile(
                        f'{ppp_dir_path}/{dirname}/{id}_{chain1}_{chain2}.dssp'): ppp.dssp_complex_file_in = f'{ppp_dir_path}/{dirname}/{id}_{chain1}_{chain2}.dssp'
                if os.path.isfile(
                        f'{ppp_dir_path}/{dirname}/{id}_{chain1}.dssp'): ppp.protein1.dssp_file_in = f'{ppp_dir_path}/{dirname}/{id}_{chain1}.dssp'
                if os.path.isfile(
                        f'{ppp_dir_path}/{dirname}/{id}_{chain2}.dssp'): ppp.protein2.dssp_file_in = f'{ppp_dir_path}/{dirname}/{id}_{chain2}.dssp'
            if 'interface' in lower_dirname:
                if found_interface: print(
                    f'for directory {ppp_dir_path} multiple directories for interface data were found')
                found_interface = True
                for filename in os.listdir(f'{ppp_dir_path}/{dirname}'):
                    if 'interface' in filename and filename.count('_') > 1 and filename.endswith('.txt'):
                        atom_type = filename.split('_')[0]
                        dist = filename.split('_')[2]
                        ppp.interface_files_dict[f'{atom_type}_interface_{dist}'] = f'{ppp_dir_path}/{dirname}/{filename}'
                #if os.path.isfile(f'{ppp_dir_path}/{dirname}/na_interface_6A_{id}_{chain1}_{chain2}.txt'):
                #    ppp.interface_files_dict[
                #        'na_interface_6A'] = f'{ppp_dir_path}/{dirname}/na_interface_6A_{id}_{chain1}_{chain2}.txt'
                #if os.path.isfile(f'{ppp_dir_path}/{dirname}/ca_interface_15A_{id}_{chain1}_{chain2}.txt'):
                #    ppp.interface_files_dict[
                #        'ca_interface_15A'] = f'{ppp_dir_path}/{dirname}/ca_interface_15A_{id}_{chain1}_{chain2}.txt'
            if 'evcomplex' in lower_dirname:
                if found_ev_complex: print(
                    f'for directory {ppp_dir_path} multiple directories for evcomplex data were found')
                found_ev_complex = True
                if os.path.isfile(f'{ppp_dir_path}/{dirname}/{id}_{chain1}_{chain2}_CouplingsScoresCompared_inter.csv'):
                    ppp.ec_file_in = f'{ppp_dir_path}/{dirname}/{id}_{chain1}_{chain2}_CouplingsScoresCompared_inter.csv'
                if os.path.isfile(f'{ppp_dir_path}/{dirname}/{name}_CouplingsScoresCompared_inter.csv'):
                    ppp.ec_file_in = f'{ppp_dir_path}/{dirname}/{name}_CouplingsScoresCompared_inter.csv'
    return ppp

