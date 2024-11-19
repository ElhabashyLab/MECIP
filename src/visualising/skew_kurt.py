import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.analysis_tools.skew_kurt import get_s_k_df
from scipy.stats import norm
from statistics import NormalDist

def draw_kurt_skew_plot(ppps, params, additional_hue=None, with_boundary = False, filepath_out=None):
    # additional_hue    :   Dataframe : dataframe containing the same indices as the ppps, that is plotted as hue
    # with_boundary   :   boolean    :   if True: plot that decision boundary as well (see params)
    # plots the kurtosis against the skewness of all complexes
    # can include specific coloring or a decision boundary
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath


    log_scale = True

    s_k_df = get_s_k_df(ppps, params)





    if params['ks_plot_decision_boundary'] and with_boundary:
        decision_boundary_vector = np.vectorize(eval(params['ks_plot_decision_boundary']))
        x_pts = np.linspace(0.2, 2, 1000)
        plt.plot(x_pts, decision_boundary_vector(x_pts), label='decision boundary')

    if isinstance(additional_hue, pd.DataFrame):
        hue_name=additional_hue.columns[0]
        s_k_h_df = pd.concat([s_k_df,additional_hue], axis=1)
        splot = sns.scatterplot(data=s_k_h_df, x='skewness', y='excess_kurtosis', hue=hue_name, alpha=.8, s=8)
    else: splot = sns.scatterplot(data = s_k_df, x='skewness', y='excess_kurtosis', alpha=.8, s=8)
    if log_scale: splot.set(xscale="log", yscale="log")
    plt.title('skewness-kurtosis plot for each complex of the dataset')
    # save or draw the plot
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()

def plot_ec_distribution(ppp, params, filepath_out=None, with_norm_dist=True):
    # plots the distribution curve for a given set of coupling scores
    # if with_norm_dist is True, a normal distribution will be fitted to the existing distribution
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath

    data = ppp.get_ecs(exclude_outside_pdb=params['remove_ecs_outside_of_monomer_pdb'])['cn']
    # Plot the histogram
    ax = sns.histplot(data = data, bins=50, kde=True, alpha=.3)
    # Get the kde curve details
    if with_norm_dist:
        kde_curve = ax.lines[0]
        x = kde_curve.get_xdata()
        y = kde_curve.get_ydata()
        kde_argmax = x[np.argmax(y)]
        kde_max = np.max(y)

        # Fit a normal distribution to the left side of the data:
        mu, std = norm.fit(data.to_numpy())
        nor = NormalDist.from_samples(data)
        m = nor.mean
        s = nor.stdev

        # Plot the PDF.
        xmin = np.min(data)
        xmax = np.max(data)
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, kde_argmax, s-.0015)
        plt.plot(x, p * (kde_max/np.max(p)), 'k--')


    #save or draw the plot
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()