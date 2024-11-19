import pandas as pd
from pathlib import Path
from subprocess import call
import sys
import os











def write_run_param(filepath_out,HADDOCK_DIR, PROJECT_DIR, pdbs, UNAMBIG_TBL=None, AMBIG_TBL=None):
    #filepath_out   :   String      :   rel or absolute filepath to the dir where the run.param file is created
    #HADDOCK_DIR    :   String      :   rel or absolute filepath to the location of HADDOCK2.4 on disk
    #PROJECT_DIR    :   String      :   rel or absolute filepath to the location where the docking run will be performed
    #pdbs           :   [String]    :   list of rel or absolute filepaths to the pdb files that should be used for docking (max 20 pdb files)
    #UNAMBIG_TBL    :   String      :   rel or absolute filepath to the restraint file (will always be used)
    #AMBIG_TBL      :   String      :   rel or absolute filepath to the restraint file (50% randomly selected)

    # writes the run.param file that is used to initiate the HADDOCK run

    if not filepath_out.endswith('/'): filepath_out += '/'
    with open(filepath_out + 'run.param', 'w') as f:
        f.write('HADDOCK_DIR='+HADDOCK_DIR)
        f.write('\n')

        f.write('PROJECT_DIR='+PROJECT_DIR)
        f.write('\n')

        f.write('RUN_NUMBER='+'_0')
        f.write('\n')

        f.write('N_COMP='+str(len(pdbs)))
        f.write('\n')

        for idx,pdb in enumerate(pdbs):
            f.write('PDB_FILE'+str(idx+1)+'='+pdb)
            f.write('\n')

        if UNAMBIG_TBL:
            f.write('UNAMBIG_TBL='+UNAMBIG_TBL)
            f.write('\n')

        if AMBIG_TBL:
            f.write('AMBIG_TBL='+AMBIG_TBL)
            f.write('\n')

def edit_run_cns(filepath, cmrest = True, sym_on = False, num_struc_rigid = 100, num_struc_refine = 50, num_struc_analyse = 50, cpu_number=28,
                 ana_type='cluster', clust_meth='RMSD', clust_cutoff=5, clust_size=4):
    # filepath  :   String  :   filepath to the run.cns file, created by the first haddock call on the run.param file
    # edits the run.cns file according to the given inputs


    with open(filepath, "r") as inp:  # Read phase
        data = inp.readlines()  # Reads all lines into data at the same time

    for index, item in enumerate(data):  # Modify phase, modification happens at once in memory
        if item.startswith('{===>} structures_0'):
            data[index] = '{===>} structures_0='+str(num_struc_rigid)+';\n'
        if item.startswith('{===>} structures_1'):
            data[index] = '{===>} structures_1='+str(num_struc_refine)+';\n'
        if item.startswith('{===>} anastruc_1'):
            data[index] = '{===>} anastruc_1='+str(num_struc_analyse)+';\n'
        if item.startswith('{===>} cmrest'):
            if cmrest: data[index] = '{===>} cmrest=true;\n'
            else: data[index] = '{===>} cmrest=false;\n'
        if item.startswith('{===>} sym_on'):
            if sym_on: data[index] = '{===>} sym_on=true;\n'
            else: data[index] = '{===>} sym_on=false;\n'
        if item.startswith('{===>} cpunumber_1='):
            data[index] = '{===>} cpunumber_1='+str(cpu_number)+';\n'
        if item.startswith('{===>} runana'):
            data[index] = '{===>} runana='+str(ana_type)+';\n'
        if item.startswith('{===>} clust_meth'):
            data[index] = '{===>} clust_meth='+str(clust_meth)+';\n'
        if item.startswith('{===>} clust_cutoff'):
            data[index] = '{===>} clust_cutoff='+str(clust_cutoff)+';\n'
        if item.startswith('{===>} clust_size'):
            data[index] = '{===>} clust_size='+str(clust_size)+';\n'



    with open(filepath, "w") as document1:  # write back phase
        document1.writelines(data)

def write_dist_restraints_file_for_command_line_tool (ca_s, filepath_out):
    output = ''
    for ca in ca_s:
        if ca[2] and ca[3]:
            output += f'assign (name ca and segid A and resi {ca[0]}) (name ca and segid B and resi {ca[1]}) 5 5 2\n'
        elif ca[2]:
            output += f'assign (name ca and segid A and resi {ca[0]}) (name cb and segid B and resi {ca[1]}) 5 5 2\n'
        elif ca[3]:
            output += f'assign (name cb and segid A and resi {ca[0]}) (name ca and segid B and resi {ca[1]}) 5 5 2\n'
        else:
            output += f'assign (name cb and segid A and resi {ca[0]}) (name cb and segid B and resi {ca[1]}) 5 5 2\n'

    # save results
    with open(filepath_out, 'w') as f:
        f.write(output)
def write_dist_restraints_file(ca_s, filepath_out, params, ppi):
    #Distance restraints for use in HADDOCK are defined as (following the CNS syntax):
    #assign (selection1) (selection2) distance, lower-bound correction, upper-bound correction
    # example: assign (name ca and segid A and resi 128) (name ca and segid B and resi 21) 13.999 0.736 0.709
    # if params say so (distance_restraints_with_monomer_pdbs): the two pdb files are also created at the location of filepath_out (therefore it is advised to make a new directory for each complex)
    # the created pdb files meet the haddock confinements and criteria (see function: pdb_parser.write_pdb_file_for_haddock)
    #ca_s   :   [[int,int, bool, bool]] :   list of lists containing all residues that are suspected to be connected (first 2 int) as well as two indicators, if there is a glycine present -> no C-beta assignment possible
    # filepath_out  :   String  :   filepath where the distance restaint file will be saved (in the same directory of this filepath, the pdbs will be saved as well)

    # if only 1 or 2 ECs are predicted, it is stocked up to 3 predictions, by taking the next 2, or 1 most likely ECs respectively by looking at the predicted confidence.
    if len(ca_s)<3:
        #print(ppi.ppp.name)
        df2 = ppi.df_ecs.sort_values('prediction_confidence', ascending=False)
        df2 = df2.reset_index(drop=True)
        df2 = df2.head(3)
        ca_s = [[row['i'], row['j']] for (idx, row) in df2.iterrows()]


    # check for gycine in ca_s and add boolean identifier
    g_identifiers = []
    seq1 = ppi.ppp.protein1.get_sequence()
    seq2 = ppi.ppp.protein2.get_sequence()
    for ca in ca_s:
        curr_identifiers = []
        curr_identifiers.append(True if seq1[ca[0] - 1] == 'G' else False)
        curr_identifiers.append(True if seq2[ca[1] - 1] == 'G' else False)
        g_identifiers.append(curr_identifiers)
    ca_s = [ca + g_i for (ca, g_i) in zip(ca_s, g_identifiers)]

    output=''
    for ca in ca_s:
        if ca[2] and ca[3]:
            output+= f'assign (name ca and segid A and resi {ca[0]}) (name ca and segid B and resi {ca[1]}) 5 5 2\n'
        elif ca[2]:
            output+= f'assign (name ca and segid A and resi {ca[0]}) (name cb and segid B and resi {ca[1]}) 5 5 2\n'
        elif ca[3]:
            output+= f'assign (name cb and segid A and resi {ca[0]}) (name ca and segid B and resi {ca[1]}) 5 5 2\n'
        else:
            output+= f'assign (name cb and segid A and resi {ca[0]}) (name cb and segid B and resi {ca[1]}) 5 5 2\n'

    #save results
    with open(filepath_out, 'w') as f:
        f.write(output)

    if params['distance_restraints_with_monomer_pdbs']:
        #import shutil
        try:
            filepath_out_pdb1 = '/'.join(filepath_out.split('/')[:-1])+'/'+ppi.ppp.protein1.modified_pdb_file_in.split('/')[-1].replace('_modified.pdb', '.pdb')
            filepath_out_pdb2 = '/'.join(filepath_out.split('/')[:-1])+'/'+ppi.ppp.protein2.modified_pdb_file_in.split('/')[-1].replace('_modified.pdb', '.pdb')
            ppi.ppp.protein1.get_pdb_file().write_pdb_file_for_haddock(filepath_out_pdb1)
            ppi.ppp.protein2.get_pdb_file().write_pdb_file_for_haddock(filepath_out_pdb2)
        except AttributeError:
            pass
        #shutil.copy(ppi.ppp.protein1.modified_pdb_file_in, filepath_out_pdb1)
        #shutil.copy(ppi.ppp.protein2.modified_pdb_file_in, filepath_out_pdb2)

def get_restraints_from_ec_file(ec_filepath_in, num_top=5, percentile=.999):
    # basic way of getting all top ECs and using them as restraints
    # returns top performing ecs in a way to directly use them as restraints
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


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)



def run_haddock(filepath_out,HADDOCK_DIR, ec_filepath_in, pdb1, pdb2, UNAMBIG_TBL=None, AMBIG_TBL=None,
                cmrest=True, sym_on=False, num_struc_rigid=100, num_struc_refine=50, num_struc_analyse=50,
                cpu_number=28,
                ana_type='cluster', clust_meth='RMSD', clust_cutoff=5, clust_size=4, num_top=5, percentile=.999):
    #filepath_out   :   String      :   rel or absolute filepath to the location, where everything happens (a 'haddock_ecs' folder will be created there)
    #HADDOCK_DIR    :   String      :   rel or absolute filepath to the location of HADDOCK2.4 on disk
    #pdbs           :   [String]    :   list of rel or absolute filepaths to the pdb files that should be used for docking (max 20 pdb files)
    #UNAMBIG_TBL    :   String      :   rel or absolute filepath to the restraint file (50% randomly selected)
    #AMBIG_TBL      :   String      :   rel or absolute filepath to the restraint file (will always be used)

    # runs haddock by first generating all necessary files and executing them


    #define directory and create it if not exists yet
    filepath_out = filepath_out+'haddock_ecs/' if filepath_out.endswith('/') else filepath_out+'/haddock_ecs/'
    Path(filepath_out).mkdir(parents=True, exist_ok=True)

    # write restraint.tbl file
    ca_s = get_restraints_from_ec_file(ec_filepath_in, num_top, percentile)
    ca_s = [x+[True,True] for x in ca_s]
    write_dist_restraints_file_for_command_line_tool(ca_s,filepath_out)

    #write run.param file
    write_run_param(filepath_out, HADDOCK_DIR, filepath_out, [pdb1, pdb2], UNAMBIG_TBL=filepath_out+'restraints.tbl')

    #go to said run.param file and run haddock there to create run_0 directory
    with cd(filepath_out):
        call(['/home/centos/anaconda3/envs/haddock/bin/python', HADDOCK_DIR+'/Haddock/RunHaddock.py', filepath_out+'run.params'])

    #edit run.cns file
    edit_run_cns(filepath_out+'run_0/run.cns', cmrest, sym_on, num_struc_rigid, num_struc_refine, num_struc_analyse, cpu_number, ana_type, clust_meth, clust_cutoff, clust_size)

    #run the final haddock run
    with cd(filepath_out+'run_0/'):
        call(['/home/centos/anaconda3/envs/haddock/bin/python', HADDOCK_DIR+'/Haddock/RunHaddock.py', '>&haddock2.4.out', '&'])

    #additional analysis of clusters
    cluster_analysis(HADDOCK_DIR, filepath_out)



def run_haddock_with_params(ppis, params, out_dir):

    # runs HADDOCK with a set of parameters defined from the user
    # used after the prediction step in the overall pipeline

    dist_restraints_dir = out_dir+'distance_restraints/'

    # define directory and create it if not exists yet
    filepath_out = out_dir + 'haddock_ecs/'
    Path(filepath_out).mkdir(parents=True, exist_ok=True)
    HADDOCK_DIR = params['HADDOCK_DIR']


    for ppi in ppis:
        curr_filepath_out = filepath_out + ppi.ppp.name + '/'
        Path(curr_filepath_out).mkdir(parents=True, exist_ok=True)

        # write run.param file
        write_run_param(curr_filepath_out, HADDOCK_DIR, curr_filepath_out, [ppi.ppp.protein1.pdb_file_in, ppi.ppp.protein2.pdb_file_in], UNAMBIG_TBL=f'{dist_restraints_dir}{ppi.ppp.name}_restraints.tbl')

        # go to said run.param file and run haddock there to create run_0 directory
        with cd(curr_filepath_out):
            call(['/home/centos/anaconda3/envs/haddock/bin/python', HADDOCK_DIR + '/Haddock/RunHaddock.py',
                  curr_filepath_out + 'run.params'])

        # edit run.cns file
        edit_run_cns(curr_filepath_out + 'run_0/run.cns', params['haddock_cmrest'], params['haddock_sym_on'], params['haddock_num_struc_rigid'], params['haddock_num_struc_refine'], params['haddock_num_struc_analyse'],
                     params['n_jobs'], params['haddock_ana_type'], params['haddock_clust_meth'], params['haddock_clust_cutoff'], params['haddock_clust_size'])

        # run the final haddock run
        with cd(curr_filepath_out + 'run_0/'):
            call(['/home/centos/anaconda3/envs/haddock/bin/python', HADDOCK_DIR + '/Haddock/RunHaddock.py',
                  '>&haddock2.4.out', '&'])

        # additional analysis of clusters
        cluster_analysis(HADDOCK_DIR, curr_filepath_out)


def cluster_analysis(HADDOCK_DIR, filepath_out):
    # filepath_out   :   String      :   rel or absolute filepath to the location of the 'haddock_ecs' folder
    # HADDOCK_DIR    :   String      :   rel or absolute filepath to the location of HADDOCK2.4 on disk

    # performs HADDOCKs cluster analysis after a successful run

    HADDOCK_TOOLS_DIR = HADDOCK_DIR+'/tools'

    #call cluster_struc
    with cd(filepath_out+'run_0/structures/it1/analysis'):
        call([HADDOCK_TOOLS_DIR+'/cluster_struc.exe', '-f', 'haddock_ecs_rmsd.disp', '5', '4'], stdout=open('cluster.out','w'))
    print(f'successfully written cluster.out file in '+filepath_out+'run_0/structures/it1/analysis')
    # call ana_clusters
    with cd(filepath_out+'run_0/structures/it1'):
        call([HADDOCK_TOOLS_DIR+'/ana_clusters.csh', '-best', '4', filepath_out+'run_0/structures/it1/analysis/cluster.out'])



