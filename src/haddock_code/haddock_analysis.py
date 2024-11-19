import numpy as np
from src.utils.pdb_parser import parse_pdb_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




#TODO: kind of old code: write new when needed with params and so on



def check_pdb_for_restraints(pdb_file_in, restraints, verbose=False):
    #pdb_file_in    :   String  :   path to a pdb file containing the docked complex pdb
    #restraints     :   [[int,int]]   :   the known restraints, that should be met (in form of residue pairs)
    #returns    :   list of distances (one for each restraint)
    structure = parse_pdb_file(pdb_file_in)
    if not structure.chains:
        print(f'the file {pdb_file_in} contains errors (contains no model)')
    if not len(structure.chains) == 2:
        print(f'instead of 2 distinct chains {len(structure.chains)} were found')
    if verbose: print(f'chains: {structure.get_chain_names()}')
    distances=[]
    for restraint in restraints:
        #check if residues from restraint are close enough:
        res1=structure.chains[0].get_res(restraint[0])
        res2=structure.chains[1].get_res(restraint[1])

        distance = calc_residue_dist(res1, res2)
        if verbose: print(f'{restraint}: distance = {distance}')
        distances.append(distance)
    return distances

def compare_restraints(pdb_file_in, ec_file_in, distance_threshold = 15, verbose=False, num_top=5, percentile=.999):
    restraints = get_restraints_from_ec_file(ec_file_in, num_top=num_top, percentile=percentile)
    distances = check_pdb_for_restraints(pdb_file_in, restraints, verbose=verbose)
    bool_distances = [dist < distance_threshold for dist in distances]
    return bool_distances

def compare_restraints_iteratively(pdb_file_in, ec_file_in, pprint=False):
    result=[]
    for dist_t in [8,10,15]:
        for top_x in [5,10,50]:
            result.append([dist_t, top_x, compare_restraints(pdb_file_in, ec_file_in, distance_threshold=dist_t,
                                                             num_top=top_x, percentile=1)])
    if pprint:
        print('distance threshold \ttop ecs \tnum of correct restraints\tpercentage of correct restraints')
        for x,y,z in result:
            print(f'{x}\t\t\t{y}\t\t\t{sum(z)}\t\t\t\t{"{:.2f}".format(sum(z)/y)}')
    return result


def calc_residue_dist(residue_one, residue_two):
    """Returns the C-alpha distance between two residues"""
    # if one or more is None, return a infinite high distance:
    if not (residue_one and residue_two): return np.infty
    diff_vector = np.array(residue_one.get_CA_atom().coords) - np.array(residue_two.get_CA_atom().coords)
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
                    if dist< minimal_dist: minimal_dist=dist
    return minimal_dist

def visualise_restraints(pdb_file_in, restraints, pml_file_out, complex_identifier, index):
    #pdb_file_in    :   String  :   path to a pdb file containing the docked complex pdb (has to be absolute path)
    #restraints     :   [[int,int]]   :   the known restraints, that should be met (in form of residue pairs)
    #pml_dir_out   :   String  :   path to output directory
    #index  :   String  :   name of protein_pair (for filenames of generated files)
    #returns    :   pml file that shows restraints once executed
    chain1=complex_identifier.split('_')[1]
    chain2=complex_identifier.split('_')[2]


    lines=[]
    lines.append('#How to run ')
    lines.append('#{Pathtopymol}/pymol -c script.pml')
    lines.append(f'load {pdb_file_in}')
    lines.append('hide all')
    lines.append('bg_color white')
    lines.append(f'show cartoon, chain {chain1}')
    lines.append(f'#show surface, chain {chain1}')
    lines.append(f'color palegreen, chain {chain1}')
    lines.append(f'show cartoon, chain {chain2}')
    lines.append(f'#show surface, chain {chain2}')
    lines.append(f'color lightblue, chain {chain2}')
    lines.append('set transparency, 0.8')#?
    lines.append('show cartoon, all')#?
    for idx,restraint in enumerate(restraints):
        lines.append(f'dist ecs_{idx}_{restraint[0]}_{restraint[1]} , resid {restraint[0]} and chain {chain1} and name ca, resid {restraint[1]} and chain {chain2} and name ca')
    lines.append('show dashes')
    lines.append('set dash_gap, 0.1')
    lines.append('color deeppurple, ecs*')
    lines.append('set dash_width, 5')
    png_file_out= pml_file_out.replace('.pml', '.png')
    pse_file_out= pml_file_out.replace('.pml', '.pse')
    lines.append(f'png {png_file_out}')
    lines.append(f'save {pse_file_out}')
    lines = [line+'\n' for line in lines]
    with open(pml_file_out, "w") as document1:  # write back phase
        document1.writelines(lines)


def get_restraints_from_ec_file(ec_filepath_in, num_top=5, percentile=.999):
    df_ecs = pd.read_csv(ec_filepath_in)
    df_percentile = get_percentile_or_value(df_ecs.sort_values(by=['cn'], ascending=False), percentile, num_top)
    return df_percentile[['i','j']].to_numpy()


def get_percentile(df, percentile):
    return(df.head(int(len(df.index)*(1-percentile))))


def get_percentile_or_value(df, percentile, top_x):
    # returns a dataframe containing either the top percentile, or the top_x of the dataframe, whichever is bigger
    df = df.sort_values(by=['cn'], ascending=False)
    top_x_percentile = get_percentile(df, percentile)
    if len(top_x_percentile)>=top_x:
        return top_x_percentile
    else:
        return df.head(top_x)


def read_structures_haddock_sorted_stat(filepath_in):
    df = pd.read_csv(filepath_in, sep=' ')
    df = df.dropna()
    return df

def plot_rmsd_haddock_scores(filepath_in, index, save_dir=None):
    #filepath_in    :   String  :   path to structures_haddock_sorted_stat file
    #index  :   String  :   index of complex (used for title of plot)
    #save   :   String  :   if None, plot is shown; if not None, plot is saved at this location
    df = read_structures_haddock_sorted_stat(filepath_in)
    sns.scatterplot(data=df, y='haddock-score', x='rmsd_all')
    plt.xlabel('RMSD to the lowest energy structure')
    plt.xlabel('HADDOCK score')
    plt.title(index)
    if save_dir:
        plt.savefig(save_dir + index + '_haddock_rmsd_plot.png')
    else:
        plt.show()

#visualise_restraints('/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_in/haddock_test_results/haddock_ecs_top1_100w.pdb',[[10,10]],
#                     '/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_out/','test_index')
#check_pdb_for_restraints('../../data/test_data_in/test_complex_2/5Z62_H_N.pdb',[[10,10]])


#test_pdb = '/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_in/haddock_test_results/haddock_ecs_top1_100w.pdb'
#test_ec_file = '../../data/test_data_in/ECs/xlmusmito_00000003_CouplingScores_inter.csv'
#print(compare_restraints_iteratively(test_pdb,test_ec_file,pprint=True))


#visualise_restraints(test_pdb, test_restraints, '/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_out/','test_index')
#read_structures_haddock_sorted_stat('../../data/test_data_in/haddock_test_results/structures_haddock-sorted.stat')
