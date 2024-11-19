import sys
from src.haddock_code.haddock_generation import run_haddock





# This is a command line tool to run haddock
# by inputting wrong parameters the correct way how to use it is displayed

# example run:
# nohup python /vol/ben_data/xlms/scripts/co-evolution_mitochondria/src/haddock_code_2/haddock_generation.py
# /vol/ben_data/xlms/tmp
# /vol/ben_data/xlms/haddock2.4-2021-05
# /vol/ben_data/xlms/evcomplex/xlmusmito_00000055_CouplingScores_inter.csv
# /vol/ben_data/xlms/dataset/xlmusmito_00000055_ETFA_ETFB/haddock_com/ETFA_hm_1efv_A.pdb
# /vol/ben_data/xlms/dataset/xlmusmito_00000055_ETFA_ETFB/haddock_com/ETFB_hm_1efv_B.pdb
# num_struc_rigid 1000 num_struc_refine 500 num_struc_analyse 500 clust_cutoff 7.5 ana_type full
# > test.out 2> test.err & disown





if len(sys.argv) <6 :
    print('input parameters: \nfilepath_out,HADDOCK_DIR, ec_filepath_in, pdb1, pdb2, [additional params]')
    print(f'number of found input parameters: {len(sys.argv)}')
    print(sys.argv)
    exit()

#additional params:
add_params = {'UNAMBIG_TBL':None,
'AMBIG_TBL':None,
'cmrest':True,
'sym_on':False,
'num_struc_rigid':100,
'num_struc_refine':50,
'num_struc_analyse':50,
'cpu_number':28,
'ana_type':'cluster',
'clust_meth':'RMSD',
'clust_cutoff':5,
'clust_size':4,
'num_top':5,
'percentile':.999}
add = sys.argv[6:]
if add:
    if len(add)%2==1:
        print(f'number of found additional input parameters has to be even and not {len(add)}')
        print(add)
        exit()
    for idx,(var,val) in enumerate(zip(add,add[1:])):
        if idx%2==0:
            if var == 'ana_type' or var == 'clust_meth': add_params[var]=val
            elif var == 'cmrest' or var == 'sym_on' : add_params[var]=(val=='True')
            elif var in ['num_struc_rigid','num_struc_refine','num_struc_analyse',
                       'cpu_number','clust_size','num_top']: add_params[var]=int(val)
            else: add_params[var]=float(val)

print(f'used parameters:\n{sys.argv[1:]}')
print(f'used additional parameters:\n{add_params}')

run_haddock(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],UNAMBIG_TBL=add_params['UNAMBIG_TBL'], AMBIG_TBL=add_params['AMBIG_TBL'],
                cmrest=add_params['cmrest'], sym_on=add_params['sym_on'], num_struc_rigid=add_params['num_struc_rigid'], num_struc_refine=add_params['num_struc_refine'], num_struc_analyse=add_params['num_struc_analyse'],
                cpu_number=add_params['cpu_number'],
                ana_type=add_params['ana_type'], clust_meth=add_params['clust_meth'], clust_cutoff=add_params['clust_cutoff'], clust_size=add_params['clust_size'], num_top=add_params['num_top'], percentile=add_params['percentile'])








if False:
    write_run_param('../../data/test_data_out',
                    HADDOCK_DIR='/home/ben/haddock2.4-2021-05',
                    PROJECT_DIR='/data/test_data_out',
                    pdbs=['/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_in/haddock_in/ATP5E_hm_6tt7_Q.pdb',
                          '/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_in/haddock_in/ATPA_hm_2w6e_C.pdb'])