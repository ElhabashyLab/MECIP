import os
import numpy as np
import Bio.PDB

def calculate_contact_map(ppp):
    # returns 2 distnce matrices for cantact maps:
    # first dist matrix contains all c-alpha distances
    # second dist matrix contains all heavy atom distances (only if c-alpha distance is smaller than 20)
    # all missing values are set to 0
    chain_1=ppp.protein1.chain
    chain_2=ppp.protein2.chain
    pdb_file = ppp.pdb_complex_file_in
    if not pdb_file: return None,None
    structure = Bio.PDB.PDBParser().get_structure('', pdb_file)
    model = structure[0]
    # get chains directly from model
    chains = []
    for c in model.get_chains():
        chains.append(c.get_id())
    if not len(chains) == 2:
        print('instead of 2 distinct chains ' + str(len(chains)) + ' were found')
        return None,None
    if chain_1 not in chains or chain_2 not in chains:
        print('expected chains ' + chain_1 + ' and ' + chain_2 + ' but instead got chains ' + chains[0] + ' and ' +
              chains[1])
        chain_1 = chains[0]
        chain_2 = chains[1]
    return calc_dist_matrix(model[chain_1], model[chain_2])


def calc_residue_dist(residue_one, residue_two):
    """Returns the C-alpha distance between two residues"""
    if not 'CA' in residue_one or not 'CA' in residue_two: return 0
    diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_heavy_residue_dist(residue_one, residue_two, c_alpha_dist):
    """Returns the minimal distance of any heavy atoms between two residues"""
    minimal_dist = c_alpha_dist
    for atom1 in residue_one:
        if not 'H' in atom1.name:
            for atom2 in residue_two:
                if not 'H' in atom2.name:
                    diff_vector = atom1.coord - atom2.coord
                    dist = np.sqrt(np.sum(diff_vector * diff_vector))
                    if dist < minimal_dist: minimal_dist = dist
    return minimal_dist


def calc_dist_matrix(chain_one, chain_two):
    """Returns a matrix of C-alpha distances between two chains as well as their heavy atom distance if C-alpha distnace is smaller than 20"""
    # get length of chains (chain names differ from actual lengths, since they can start at any number and skip numbers)
    max_segid_chain_1 = 0
    max_segid_chain_2 = 0
    for residue_one in chain_one:
        if int(residue_one.get_full_id()[3][1]) > max_segid_chain_1: max_segid_chain_1 = int(
            residue_one.get_full_id()[3][1])
    for residue_two in chain_two:
        if int(residue_two.get_full_id()[3][1]) > max_segid_chain_2: max_segid_chain_2 = int(
            residue_two.get_full_id()[3][1])
    result1 = np.zeros((max_segid_chain_1 + 1, max_segid_chain_2 + 1))
    result2 = np.zeros((max_segid_chain_1 + 1, max_segid_chain_2 + 1))
    for residue_one in chain_one:
        row = residue_one.get_full_id()[3][1]
        for residue_two in chain_two:
            col = residue_two.get_full_id()[3][1]
            result1[row, col] = calc_residue_dist(residue_one, residue_two)
            if result1[row, col] <= 20: result2[row, col] = calc_heavy_residue_dist(residue_one, residue_two,
                                                                                    result1[row, col])
    return result1, result2
