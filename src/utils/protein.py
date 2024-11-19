
from.pdb_parser import parse_pdb_file
import pandas as pd
from Bio import SeqIO
import numpy as np

class Protein():
    def __init__(self, uid, pdb_name = None, chain = None, sequence_file_in = None, pdb_file_in = None, AF_pdb_file_in = None, hm_pdb_file_in = None, naccess_file_in = None,
                 msa_file_in = None, dssp_file_in=None):
        # uid  :   String  :   name of the protein (uniprot id)
        # pdb_name  :   String  :   name of the protein or protein complex in question (pdb identifier) (if complex, a chain name is needed)
        # chain :   String  :   name of the chain that identifies the protein in a complex
        # sequence_file_in  :   String  :   absolute filepath to the aa sequence file (mostly a fasta file)
        # pdb_file_in   :   String  :   absolute filepath to the pdb file containing the structure of the protein in question
        # AF_pdb_file_in   :   String  :   absolute filepath to the Alphafold pdb file containing the structure of the protein in question
        # hm_pdb_file_in   :   String  :   absolute filepath to the homologous pdb file containing the structure of the protein in question
        # naccess_file_in   :   String  :   absolute filepath to the Naccess file
        # msa_file_in   :   String  :   absolute filepath to the file containing the multiple sequence alignment of the protein and orthologs from different proteomes
        # dssp_file_in   :   String  :   absolute filepath to the file containing the dssp results
        # dir_path   :   String  :   absolute filepath to the directory containing all info on the protein (ends with '/')

        self.uid = uid
        self.pdb_name = pdb_name
        self.chain = chain
        self.sequence_file_in = sequence_file_in
        self.pdb_file_in = pdb_file_in
        self.AF_pdb_file_in = AF_pdb_file_in
        self.hm_pdb_file_in = hm_pdb_file_in
        self.naccess_file_in = naccess_file_in
        self.msa_file_in = msa_file_in
        self.dssp_file_in = dssp_file_in
        self.dir_path=''
        self.AF_dssp_file_in=None
        self.AF_naccess_file_in=None
        self.modified_pdb_file_in = None
        self.modified_rsa_file_in = None

    def get_sequence(self):
        # returns string containing the exact sequence in single letter aa code
        if self.sequence_file_in is None: return None
        fasta_sequences = SeqIO.parse(open(self.sequence_file_in),'fasta')
        # if multiple fasta sequences are given, only the first one is considered
        for fasta_sequence in fasta_sequences:
            return str(fasta_sequence.seq)


    def get_pdb_file(self):
        #returns the modified version of the pdb (if the unmodified is wanted use 'get_original_pdb_file')
        # modified means, that it was aligned against the uniprot sequence, to adjust the residue numbering regarding the uniprot sequence, so that it fits to the EC data
        if hasattr(self, 'modified_pdb_file_in') and self.modified_pdb_file_in is not None:
            return parse_pdb_file(self.modified_pdb_file_in)
        else:
            return None

    def get_original_pdb_file(self):
        #returns the original version of the pdb used as input
        if self.pdb_file_in:
            return parse_pdb_file(self.pdb_file_in)
        else:
            return None



    def get_AF_pdb_file(self, plddt_threshold):
        # returns the alphafold pdb file as a PdbFile object
        if self.AF_pdb_file_in:
            return parse_pdb_file(self.AF_pdb_file_in, plddt_threshold)
        else:
            return None

    def get_hm_pdb_file(self):
        # returns the homology pdb file as a PdbFile object
        #if pdb file was aligned to uniprot sequence during read in, a modified pdb exists
        if hasattr(self, 'modified_hm_pdb_file_in'):
            return parse_pdb_file(self.modified_hm_pdb_file_in)
        if self.hm_pdb_file_in:
            return parse_pdb_file(self.hm_pdb_file_in)
        else:
            return None

    def get_naccess_df(self):
        # returns: dataframe containing all information
        if self.naccess_file_in:
            colnames = ['type', 'residue_name', 'chain', 'res_num', 'All-atoms_abs', 'All-atoms_rel', 'Total-Side_abs',
                        'Total-Side_rel', 'Main-Chain_abs', 'Main-Chain_rel', 'Non-polar_abs', 'Non-polar_rel',
                        'All-polar_abs', 'All polar_rel']
            df = pd.read_csv(self.naccess_file_in, skiprows=4, delimiter=r"\s+", header=None, skipfooter=4, names=colnames, index_col='res_num')
        else:
            df=None
        return df

    def get_dssp_df(self):
        #returns dataframe with the column 'All-atoms_rel' and the indices 'res_num'
        if self.dssp_file_in:
            columns = ['res_num', 'All-atoms_rel']
            results = []
            # Using readlines()
            file1 = open(self.dssp_file_in, 'r')
            lines = file1.readlines()

            start=False
            # Strips the newline character
            for line in lines:
                if start:
                    results.append([mk_int(line[5:11].strip()), mk_int(line[34:40].strip())])
                if line.startswith('  #  RESIDUE AA STRUCTURE BP1 BP2  ACC'):
                    start = True
            return pd.DataFrame(results, columns=columns).set_index('res_num')
        else:
            return None


    def get_AF_naccess_df(self):
        # returns: dataframe containing all information
        if self.AF_naccess_file_in:
            colnames = ['type', 'residue_name', 'chain', 'res_num', 'All-atoms_abs', 'All-atoms_rel', 'Total-Side_abs',
                        'Total-Side_rel', 'Main-Chain_abs', 'Main-Chain_rel', 'Non-polar_abs', 'Non-polar_rel',
                        'All-polar_abs', 'All polar_rel']
            df = pd.read_csv(self.AF_naccess_file_in, skiprows=4, delimiter=r"\s+", header=None, skipfooter=4, names=colnames, index_col='res_num')
        else:
            df=None
        return df

    def get_AF_dssp_df(self):
        #returns dataframe with the column 'All-atoms_rel' and the indices 'res_num'
        if self.AF_dssp_file_in:
            columns = ['res_num', 'All-atoms_rel']
            results = []
            # Using readlines()
            file1 = open(self.AF_dssp_file_in, 'r')
            lines = file1.readlines()

            start=False
            # Strips the newline character
            for line in lines:
                if start:
                    results.append([mk_int(line[5:11].strip()), mk_int(line[34:40].strip())])
                if line.startswith('  #  RESIDUE AA STRUCTURE BP1 BP2  ACC'):
                    start = True
            return pd.DataFrame(results, columns=columns).set_index('res_num')
        else:
            return None

    def get_original_rsa_df(self, rsa_preferred_method):
        #returns dataframe containing at least the column 'All-atoms_rel' with the index column 'res_num'
        if self.dssp_file_in and self.naccess_file_in:
            if rsa_preferred_method=='dssp':
                return self.get_dssp_df()
            elif rsa_preferred_method=='naccess':
                return self.get_naccess_df()
            else:
                print(f'preferred method for rsa ({rsa_preferred_method}) does not exist')
                return None
        elif self.dssp_file_in:
            return self.get_dssp_df()
        elif self.naccess_file_in:
            return self.get_naccess_df()
        else:
            print(f'missing rsa file for protein {self.uid}')
            return None


    def get_original_AF_rsa_df(self, rsa_preferred_method):
        #returns dataframe containing at least the column 'All-atoms_rel' with the index column 'res_num'
        if self.AF_dssp_file_in and self.AF_naccess_file_in:
            if rsa_preferred_method=='dssp':
                return self.get_AF_dssp_df()
            elif rsa_preferred_method=='naccess':
                return self.get_AF_naccess_df()
            else:
                print(f'preferred method for rsa ({rsa_preferred_method}) does not exist')
                return None
        elif self.AF_dssp_file_in:
            return self.get_AF_dssp_df()
        elif self.AF_naccess_file_in:
            return self.get_AF_naccess_df()
        else:
            print(f'missing rsa file for protein {self.uid}')
            return None

    def get_rsa_df(self):
        # returns the modified rsa file (with matching indices to the modified pdb file and therefor the uniprot sequence)
        if self.modified_rsa_file_in is not None:
            return pd.read_csv(self.modified_rsa_file_in)
        else:
            return None


        
    def calc_inner_contact_map(self):
        # returns a contact map showing all heavy atom distances between each residue pair
        pdb = self.get_pdb_file()
        if pdb is None:
            print(f'pdb file for protein {self.pdb_name} does not exist')
            return None
        if not len(pdb.get_chain_names()) == 1:
            print(f'pdb file {pdb.name} does not consist of 1 chain')
        _,max_res_number = pdb.chains[0].get_res_range()
        c_map = [ [ None for i in range(max_res_number) ] for j in range(max_res_number) ]

        for i, res1 in enumerate(pdb.chains[0].residues):
            for j, res2 in enumerate(pdb.chains[0].residues):
                c_map[i][j] = res1.calc_heavy_atom_distance_to(res2)
        self.inner_contact_map = c_map
        return c_map
def mk_int(s):
    # helper function: strips string and converts into int, -1 if not possible
    s = s.strip()
    return int(s) if s else -1


