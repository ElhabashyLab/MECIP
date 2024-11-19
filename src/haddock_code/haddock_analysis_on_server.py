from haddock_analysis import compare_restraints_iteratively
import os




####################### WARNING ################################
# This class is used to analyse a specific dataset and is therefore not part of the tool






def check_hadeers_results(dataset_dir):
    if not dataset_dir.endswith('/'): dataset_dir=dataset_dir+'/'
    ec_file_dir = dataset_dir.replace('dataset/','evcomplex/')
    for dir_name in os.listdir(dataset_dir):
        if dir_name.startswith('xlmusmito_0'):
            curr_dir = dataset_dir+dir_name+'/models/haddock_ecs/'

            #get top structure
            if not os.path.exists(curr_dir): continue
            top_struc = [x for x in os.listdir(curr_dir) if x.startswith('haddock_ecs_top1_')]
            if top_struc: top_struc = curr_dir+top_struc[0]
            else:
                #print(f'no top 1 structure for {dir_name}')
                continue

            #get ec_file
            ec_file=''
            for file_name in os.listdir(ec_file_dir):
                if file_name == '_'.join(dir_name.split('_')[:2])+'_CouplingScores_inter.csv':
                    ec_file=ec_file_dir+file_name

            #print(dir_name)
            #print(ec_file)
            #print(top_struc)

            #check if files exist
            if not os.path.exists(ec_file): continue
            if not os.path.exists(top_struc): continue


            #perform analysis for top structure
            print(dir_name)
            compare_restraints_iteratively(top_struc, ec_file, pprint=True)
            print()

check_hadeers_results('/vol/ben_data/xlms/dataset')