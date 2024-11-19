from src.analysis_tools.ecs import get_top_ecs
import statistics
import os
from src.visualising.contact_maps import draw_ec_dots
from src.utils.protein_protein_pairs import PPP
from src.utils.protein import Protein






def detect_far_homology(df_top_ecs):
    # df_top_ecs    :   DataFrame   :   pd.DataFrame with the top scoring ecs that should be analysed for far homology
    # returns fraction of points that are located within a single diagonal
    epsilon=15
    diff=[]
    for idx,row in df_top_ecs.iterrows():
        diff.append(row['i'] - row['j'])
    mode = statistics.mode(diff)
    return sum([1 for x in diff if x>mode-epsilon and x< mode+epsilon])/len(df_top_ecs)


def detect_multiple_far_homology(df_top_ecs, epsilon_max=20):
    # df_top_ecs    :   DataFrame   :   pd.DataFrame with the top scoring ecs that should be analysed for far homology
    # returns fraction of points that are located within either a single or two diagonals (whichever is the best)

    #calculate epsilon
    min_range = min(max(df_top_ecs.i)-min(df_top_ecs.i), max(df_top_ecs.j) - min(df_top_ecs.j))
    epsilon = min(epsilon_max, min_range/10)

    diff=[]
    for idx,row in df_top_ecs.iterrows():
        diff.append(row['i'] - row['j'])
    modes = []
    for i in range(2):
        if len(diff)==0: continue
        mode = statistics.mode(diff)
        modes.append(mode)
        diff = [x for x in diff if x<mode-epsilon or x>mode+epsilon]
    frac=(len(df_top_ecs)-len(diff))/len(df_top_ecs)
    # check for dense clusters instead of diagonals
    if frac>.6:
        range_i = max(df_top_ecs.i)-min(df_top_ecs.i)
        min_i=-1
        max_i=-1
        range_j = max(df_top_ecs.j) - min(df_top_ecs.j)
        min_j = -1
        max_j = -1
        right_half=0
        left_half=0

        num_first=0
        first_1=0
        first_2=0
        first_3=0
        first_4=0
        middle_point= (max(df_top_ecs.i)-min(df_top_ecs.i))/2+min(df_top_ecs.i)
        quart_point= (max(df_top_ecs.i)-min(df_top_ecs.i))/4+min(df_top_ecs.i)
        third_quart_point= (max(df_top_ecs.i)-min(df_top_ecs.i))/4+middle_point
        for idx, row in df_top_ecs.iterrows():
            if modes[0]-epsilon<=(row['i'] - row['j'])<=modes[0]+epsilon or modes[1]-epsilon<=(row['i'] - row['j'])<=modes[1]+epsilon:
                if row['i'] > max_i: max_i = row['i']
                if row['j'] > max_j: max_j = row['j']
                if row['i'] < min_i or min_i==-1: min_i = row['i']
                if row['j'] < min_j or min_j==-1: min_j = row['j']

                if row['i'] > middle_point: right_half+=1
                else: left_half+=1

                num_first+=1
                if row['i'] < quart_point:
                    first_1 += 1
                elif row['i'] < quart_point:
                    first_2 += 1
                elif row['i'] < third_quart_point:
                    first_3 += 1
                else:
                    first_4 += 1
        #print(num_first, first_1, first_2, first_3, first_4)
        if frac<.95:
            if max_i-min_i < .7*range_i or max_j-min_j < .7*range_j:
                frac=-1
            if left_half>right_half*4 or right_half>left_half*4:
                frac=-2
            if first_1*1.5>num_first or first_4*1.5>num_first :
                frac=-3

    return (frac, modes)

def check_for_far_homology(ec_dirpath, out_path, params):
    # ec_dirpath    :   String  :   path to directory containing all csv files containing ec information
    # out_path      :   String  :   path to directory, where 5 txt files will be generated containing results
    # writes 5 text files that indicate possible far homology candidates with different certainties
    # it also prints different special cases that could falsely be detected as far homology (frac < 0)
    header = 'ec_file_name\tfraction\tmodes\n'
    out_60 = ''
    out_70 = ''
    out_80 = ''
    out_90 = ''
    out_95 = ''
    num_files = len(os.listdir(ec_dirpath))
    for i, ec_file_name in enumerate(os.listdir(ec_dirpath)):
        if not ec_file_name.endswith('CouplingScores_inter.csv'): continue
        df_top_ecs = get_top_ecs(PPP(Protein(None), Protein(None), ec_file_name, ec_file_in=f'{ec_dirpath}/{ec_file_name}'), params)
        (frac, modes) = detect_multiple_far_homology(df_top_ecs)
        if frac==-1:
            print(f'{ec_file_name} was removed from far-homology due to a short diagonal')
            continue
        if frac==-2:
            print(f'{ec_file_name} was removed from far-homology due to a imbalanced diagonal')
            continue
        if frac==-3:
            print(f'{ec_file_name} was removed from far-homology due to a cluster at either end')
            continue
        if frac>.95: out_95+=f'{ec_file_name}\t{frac}\t{modes}\n'
        elif frac>.9: out_90+=f'{ec_file_name}\t{frac}\t{modes}\n'
        elif frac>.8: out_80+=f'{ec_file_name}\t{frac}\t{modes}\n'
        elif frac>.7: out_70+=f'{ec_file_name}\t{frac}\t{modes}\n'
        elif frac>.6: out_60+=f'{ec_file_name}\t{frac}\t{modes}\n'
        with open(f'{out_path}/still_runnning.txt', 'w') as f: f.write(f'{i}/{num_files}')
    with open(f'{out_path}/far_homology_60_percent.txt', 'w') as f: f.write(header+out_95+out_90+out_80+out_70+out_60)
    with open(f'{out_path}/far_homology_70_percent.txt', 'w') as f: f.write(header+out_95+out_90+out_80+out_70)
    with open(f'{out_path}/far_homology_80_percent.txt', 'w') as f: f.write(header+out_95+out_90+out_80)
    with open(f'{out_path}/far_homology_90_percent.txt', 'w') as f: f.write(header+out_95+out_90)
    with open(f'{out_path}/far_homology_95_percent.txt', 'w') as f: f.write(header+out_95)

#usage:
# check_for_far_homology('<path_to_ev_complex_files>', '<path_where_results_should_be_stored>',params)