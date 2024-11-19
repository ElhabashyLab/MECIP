import numpy as np
from Bio.SeqUtils import seq1
from src.utils.timestamp import get_timestamp_seconds

class PdbFile:
    def __init__(self, name, remark, chains):
        # remark    :   [String]
        # chains    :   [chain]
        # name      :   String
        self.remark = remark
        self.chains = chains
        self.name = name

    def add_chain(self, chain):
        # adds a chain
        self.chains.append(chain)

    def get_chain_names(self):
        # returns chain names
        return [chain.name for chain in self.chains]


    def add_missing_res(self, pdb):
        # adds all residues of pdb into self, that are not already present (only checks for residue number)
        for res in pdb.chains[0].residues:
            if not self.chains[0].has_res(res.number):
                self.chains[0].add_residue(res)
        self.chains[0].residues.sort(key=lambda x: x.number)

    def write_pdb_file(self,outfile):
        #outfile    :   String  :   path to file where the final pdb file should be saved
        out = ''
        out+=f'REMARK    This file was automatically added for the protein {self.name}\n'
        out+=f'REMARK    Time of creation: '+get_timestamp_seconds()+'\n'
        out+=f'REMARK    Original Remark if present:\n'
        for r in self.remark:
            out+=r
        out+='\n'

        for chain in self.chains:
            for res in chain.residues:
                for atom in res.atoms:
                    out+='ATOM  '

                    #atom serial number
                    out+=fill_str_with_space(str(atom.number), 5, False)

                    out+=' '

                    #atom name
                    out += fill_str_with_space(str(atom.name), 4, True)

                    out += ' '

                    # res name
                    out += fill_str_with_space(str(res.name), 3, False)

                    # if length of chain is 2, no space between res name and chain name is made
                    if len(chain.name)==1 : out += ' '

                    # chain identifier
                    out += chain.name

                    # res sequence number
                    out += fill_str_with_space(str(res.number), 4, False)

                    out += '    '

                    # x coord
                    out += fill_str_with_space(str(atom.coords[0]), 8, False)

                    # y coord
                    out += fill_str_with_space(str(atom.coords[1]), 8, False)

                    # z coord
                    out += fill_str_with_space(str(atom.coords[2]), 8, False)

                    out += '  1.00'

                    # af_plddt
                    out += fill_str_with_space(str(res.af_plddt), 6, False)

                    out += '          '

                    # atom element
                    out += fill_str_with_space(str(atom.elem), 2, False)



                    out += '\n'
        with open(outfile, 'w') as f:
            f.write(out)


    def write_pdb_file_for_haddock(self,outfile):
        #outfile    :   String  :   path to file where the final pdb file should be saved

        #this creates a pdb file that can directly be used for haddock computations. Therefore it must meet some conditions:
        #    1. All non atoms remarks must be removed
        #    2. All pdb files must ends with "END" statement.
        #    3. At chain breaks, a "TER" card must be located.
        #    4. All pdb files, "SEGID" statment must be removed.
        #    5. I also remove all hydrogen to decrease the computational cost
        out = ''
        last_number=False

        for chain in self.chains:
            for res in chain.residues:
                curr_number = res.number
                if not last_number or curr_number == last_number+1:
                    pass
                else:
                    out += 'TER\n'
                last_number=curr_number
                for atom in res.atoms:
                    if str(atom.elem) =='H': continue

                    out+='ATOM  '

                    #atom serial number
                    out+=fill_str_with_space(str(atom.number), 5, False)

                    out+=' '

                    #atom name
                    out += fill_str_with_space(str(atom.name), 4, True)

                    out += ' '

                    # res name
                    out += fill_str_with_space(str(res.name), 3, False)

                    # if length of chain is 2, no space between res name and chain name is made
                    if len(chain.name)==1 : out += ' '

                    # chain identifier
                    out += chain.name

                    # res sequence number
                    out += fill_str_with_space(str(res.number), 4, False)

                    out += '    '

                    # x coord
                    out += fill_str_with_space(str(atom.coords[0]), 8, False)

                    # y coord
                    out += fill_str_with_space(str(atom.coords[1]), 8, False)

                    # z coord
                    out += fill_str_with_space(str(atom.coords[2]), 8, False)

                    out += '  1.00'

                    # af_plddt
                    out += fill_str_with_space(str(res.af_plddt), 6, False)

                    out += '          '



                    out += '\n'
        out += 'END'
        with open(outfile, 'w') as f:
            f.write(out)


class Chain:
    def __init__(self, name, residues):
        # residues    :   [residue]
        # name      :   String
        self.name = name
        self.residues = residues

    def get_res(self, number):
        # get the number-th residue out of the list of residues
        for residue in self.residues:
            if residue.number==number: return residue
        #print(f'residue number {number} not found in chain {self.name}')
        return None

    def has_res(self, number):
        #returns if a residue with a certain residue number is present in this chain
        return self.get_res(number) is not None

    def add_residue(self, residue):
        # adds a residue
        self.residues.append(residue)

    def calc_distance_btw_res_ca(self, res1_number, res2_number):
        # calculates euclidian distance between C alpha atoms of residue 1 and residue 2
        res1 = self.get_res(res1_number)
        res2 = self.get_res(res2_number)
        if res1 and res2: return res1.calc_distance_to(res2)
        else: return None

    def calc_distance_btw_res_heavy_atoms(self, res1_number, res2_number):
        # calculates euclidian distance between heavy atoms of residue 1 and residue 2
        res1 = self.get_res(res1_number)
        res2 = self.get_res(res2_number)
        if res1 and res2: return res1.calc_heavy_atom_distance_to(res2)
        else: return None

    def get_res_range(self):
        #returns min and max number of all residues within, even if some inside are missing
        res_list = self.get_res_number_list()
        return min(res_list), max(res_list)
    def get_res_number_list(self):
        #returns list of int referring to the number of all residues within
        res_list = []
        for residue in self.residues:
            res_list.append(residue.number)
        return sorted(res_list)

    def get_sequence(self):
        #returns sequence as a 1-letter encoding of amino acids, and the starting residue
        seq_min, seq_max = self.get_res_range()
        # * indicates breaks in the sequence
        seq = '*'*(seq_max-seq_min+1)
        for residue in self.residues:
            try :
                l = list(seq)
                l[residue.number - seq_min] = seq1(residue.name)
                seq = ''.join(l)
            except IndexError:
                pass
        return seq, seq_min






class Residue:
    def __init__(self, name, number, atoms, af_plddt=0):
        # name      :   String (3 letter src)
        # number      :   int
        # atoms      :   [atom]
        self.name = name
        self.number = number
        self.atoms = atoms
        self.af_plddt = af_plddt

    def add_atom(self, atom):
        # adds an atom
        self.atoms.append(atom)

    def get_CA_atom(self):
        # returns the CA atom of this residue (returns None if none is there)
        for atom in self.atoms:
            if atom.name=='CA': return atom
        print(f'no CA atom found in residue {self.name}, {self.number}')
        return None

    def calc_distance_to(self, res):
        # res   :   Residue :   other residue for distance calculation
        # calculates euclidian distance between C alpha atoms this residue and res
        if not res: return None
        diff_vector = np.array(self.get_CA_atom().coords) - np.array(res.get_CA_atom().coords)
        return np.sqrt(np.sum(diff_vector * diff_vector))
    def calc_heavy_atom_distance_to(self, res):
        # res   :   Residue :   other residue for distance calculation
        # calculates min euclidian distance between heavy atoms in this residue and res
        if not res: return None
        min_dist=999999
        for a1 in self.atoms:
            for a2 in res.atoms:
                if not 'H' in a1.name and not 'H' in a2.name:
                    diff_vector = np.array(a1.coords) - np.array(a2.coords)
                    min_dist = min(min_dist, np.sqrt(np.sum(diff_vector * diff_vector)))
        return min_dist

class Atom:
    def __init__(self, name, coords, number, elem):
        # name      :   String
        # number      :   int
        # coords      :   [float, float, float]
        # elem      :   String
        self.name = name
        self.number = number
        self.coords = coords
        self.elem = elem


def parse_pdb_file(filepath_in, plddt_threshold=-1):
    # if plddt_threshold is -1 it will be ignored, otherwise only residues with a high enough score will be saved in the pdb model
    # parses a pdb file into the PdbFile object format and returns it
    data = []
    with open(filepath_in, "r") as inp:  # Read phase
        data = inp.readlines()  # Reads all lines into data at the same time
    result=PdbFile(filepath_in.split('/')[-1], [], [])
    curr_chain = Chain(None, None)
    curr_res = Residue(None, None, None)
    for line in data:
        if line.startswith('REMARK'):
            result.remark.append(line)
        if line.startswith('ATOM'):
            if plddt_threshold == -1 or float(line[60:66].strip()) >= plddt_threshold:
                # chain identifier is increased to look at the blank space before it for chain ids that are longer than 1 character
                line_chain = line[20:22].strip()
                if line_chain==' ':
                    line_chain = line[72:76].strip() #segment identifier
                if not curr_chain.name==line_chain:
                    curr_chain=Chain(line_chain, [])
                    result.add_chain(curr_chain)
                if not curr_res.number==int(line[22:26].strip()):
                    try:
                        curr_res=Residue(line[17:20].strip(), int(line[22:26].strip()), [], float(line[60:66].strip()))
                    except ValueError:
                        curr_res = Residue(line[17:20].strip(), int(line[22:26].strip()), [])
                    curr_chain.add_residue(curr_res)
                try:
                    curr_res.add_atom(Atom(line[12:16].strip(), [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())], int(line[6:11].strip()), line[76:78].strip()))
                except ValueError:
                    pass
    return result


def fill_str_with_space(string, length, left_justified):
    #fills a string up to a certain length with spaces and checks if the string should be at the lefft or right
    num_space = length - len(string)
    if left_justified: return string+(' ' * num_space)
    else: return (' ' * num_space)+string