import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.analysis_tools.contact_map import calculate_contact_map
from src.analysis_tools.ecs import get_top_ecs
from src.analysis_tools.cluster_detection import get_clustered_ecs
import pandas as pd
from src.utils.protein_protein_pairs import PPP
from src.utils.protein import Protein

def draw_contact_map(ppp, params, filepath_out=None):
    #draws the contact map for a given ppp
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath


    ecs_cluster, ecs_not_cluster = get_clustered_ecs(ppp, params)
    #ecs=get_percentile_or_value(ppp.get_ecs(), params['cluster_ecs_consider_percentile'],params['cluster_ecs_consider_top_x'])
    dist, close_dist = calculate_contact_map(ppp)
    if dist is None:
        print(f' complex {ppp.name} has no complex pdb file and therefore no contact map can be computed')
        return
    fig = plt.figure()
    plt.xlabel(ppp.protein1.chain)
    plt.ylabel(ppp.protein2.chain)
    ax = plt.subplot(111)
    dots_close = []
    dots_very_close = []
    for x in range(len(dist)):
        for y in range(len(dist[0])):
            if dist[x][y] <= params['contact_map_ca_dist_threshold'] and not dist[x][y] == 0: dots_close.append([x, y])
            if close_dist[x][y] <= params['contact_map_heavy_atom_dist_threshold'] and not close_dist[x][y] == 0: dots_very_close.append(
                [x, y])
    df_close = pd.DataFrame(dots_close, columns=['chain_1', 'chain_2'])
    df_very_close = pd.DataFrame(dots_very_close, columns=['chain_1', 'chain_2'])
    sns.scatterplot(x='chain_1', y='chain_2', data=df_close,
                    label='C-alpha atom distance < ' + str(params['contact_map_ca_dist_threshold']), s=30, alpha=.8, c=['lightblue'])
    sns.scatterplot(x='chain_1', y='chain_2', data=df_very_close,
                    label='heavy alpha atom distance < ' + str(params['contact_map_heavy_atom_dist_threshold']), s=20, alpha=.8,
                    c=['blue'])
    if ecs_cluster is not None:
        sns.scatterplot(x='i', y='j', data = ecs_cluster, label = 'top scoring ECs in clusters', s=20, alpha=.8, c=['red'])
        sns.scatterplot(x='i', y='j', data = ecs_not_cluster, label = 'top scoring ECs not in clusters', s=20, alpha=.8, c=['orange'])
    else:
        sns.scatterplot(x='i', y='j', data = ecs_not_cluster, s=20, alpha=.8, label = 'top scoring ECs')

    plt.title('contact map of ' + ppp.name)
    box = ax.get_position()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=2)

    # save or draw the plot
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()



def draw_ec_dots(ppi, params, title = None, filepath_out=None):
    # draws a contact map containing only the ECs as dots
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath

    df_top_ecs = ppi.df_ecs
    sns.scatterplot(x='i', y='j', data=df_top_ecs, hue='true_ec', alpha=.8)
    plt.legend()

    if title:
        plt.title(title)

    # save or draw the plot
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()