import matplotlib.pyplot as plt

from src.analysis_tools.far_homology import detect_multiple_far_homology
from src.analysis_tools.ecs import get_top_ecs
import statistics
import os
from src.visualising.contact_maps import draw_ec_dots
from src.utils.protein_protein_pairs import PPP
from src.utils.protein import Protein


####################### WARNING ################################
# This class is used to analyse specific datasets and is therefore not part of the tool

params={'interface_predictor_type': 'random_forest',                   # top5, cn, additive, ml_test, random_forest, mlp, logistic_regression, linear_discriminant, gradient_boosting, calibrated
        'interaction_predictor_type': 'gradient_boosting',
        'cluster_method': 'old',                        # old or pdb
        'cluster_ecs_consider_top_x': 50,
        'cluster_ecs_consider_percentile': .999,
        'cluster_ecs_seed_top_x': 10,
        'cluster_ecs_seed_percentile': .5,              #comment: this is the percentile of the top considered ecs, not the whole set of ecs
        'cluster_min_size': 5,
        'cluster_cross_search': 5,                      #for cluster methods 'pdb'(when no pdb present)
        'cluster_circle_search': 30,                    #for cluster methods 'pdb'(when no pdb present), and 'old'
        'cluster_circle_search_pdb': 8,                    #for cluster methods 'pdb'
        'contact_map_ca_dist_threshold': 12,
        'contact_map_heavy_atom_dist_threshold': 8,
        'rsa_measurement':'All-atoms_rel',
        'rsa_threshold': 20,
        'ks_plot_decision_boundary': lambda x:-1*x+2.5,
        'TP_EC_distance_threshold_heavy_atoms' : 5,
        'TP_interaction_distance_threshold' : 5,
        'TP_interaction_num_interface_needed_threshold' : 20,
        'TP_interaction_atom_type' : 'na',
        'confidence_threshold_residue_level': .5,
        'confidence_threshold_ppi': 0,
        'rsa_preferred_method': 'dssp',                    # possible: 'dssp', 'naccess'
        'rsa_exclude' : None,                            #possible: None, 'and', 'or'
        'n_jobs' : 6,                                   #possible: mp.cpu_count()
        'save_features' : None,                         # '/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_in/features'
        'use_saved_features' : '/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_in/features',
        'n_outer_cv_interaction': 10,
        'n_outer_cv_interface': 50,
        }

df_top_ecs = get_top_ecs(PPP(Protein(None), Protein(None), '', ec_file_in='/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_in/sepmito_00016767_CouplingScores_inter.csv'), params)
(frac, modes) = detect_multiple_far_homology(df_top_ecs)

print(frac)
print(modes)


#df_top_ecs = get_top_ecs(PPP(Protein(None), Protein(None), '', ec_file_in='/media/ben/Volume/Ben/uni_stuff/Master_Thesis/co-evolution_mitochondria/data/test_data_in/xlmusmito_00000209_CouplingScores_inter.csv'), params)
#(frac, modes) = detect_multiple_far_homology(df_top_ecs)

#print(frac)
#print(modes)

