import os
from get_tiago_data import *
from clean_tiago_data import *
from shallow_ml_approaches import *
from shallow_ml_plotting import *
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import tensorflow.keras
from tensorflow.keras.optimizers import Adam, Adadelta
import h5py
import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import sys

all_data_from_matlab, mat_paths = get_data(data_directory=[r"/Users/ankushgupta/Documents/tiago_data/3143_1", r'/Users/ankushgupta/Documents/tiago_data/3143_2',
                  r'/Users/ankushgupta/Documents/tiago_data/6742'])

global_slopes = []
name_of_exp_array = []
for x in range(6):
    # just for simplicity specify single experiment file
    specific_exp = all_data_from_matlab[x]
    mat_paths = np.array(mat_paths).flatten()
    name_of_exp = string = str(mat_paths[x]).split('/')[-1]
    name_of_exp = name_of_exp.split('.')[0]
    name_of_exp_array.append(name_of_exp)

    avg_activity_per_roi, cleaned_full_df, norm_split_results = clean_the_data(data=specific_exp, threshold_percentile=0.4, splits=[0.8, 0.2], threshold_for_minimum_activity=0.01,
                                                                               validation=True, type_of_scaling='StandardScaler')

    # --------------------------------------------- DIMENSIONALITY REDUCTION ------------------------------------------------------

    # ----> Vanilla PCA
    # plot pca plots for different temporal domains (parts of recording) + corresponding variance vs # of pca components
    x_data = cleaned_full_df[cleaned_full_df.columns[:-1]].to_numpy()
    y_data = cleaned_full_df[cleaned_full_df.columns[-1]].to_numpy()
    mod_data, mod_data3, mod_data4, y_data_for_on, y_data_for_100, y_data_for_500, slopes = run_PCA(x_data=x_data, y_data=y_data,
                                  num_of_neurons=np.shape(cleaned_full_df)[1], pca_rank_coarse_graining=1,
                                                                                                exp_name=name_of_exp)
    global_slopes.append(slopes)

    # ----> UMAP
    UMAP_param_neural_net(full_x=cleaned_full_df[cleaned_full_df.columns[:-1]].to_numpy(), full_y=cleaned_full_df[cleaned_full_df.columns[-1]].to_numpy(),
                          hundred_x=mod_data3, hundred_y=y_data_for_100, five_hundred_x=mod_data4, five_hundred_y=y_data_for_500,
                          on_x=mod_data, on_y=y_data_for_on, name_of_exp=name_of_exp, num_of_neurons=np.shape(cleaned_full_df.to_numpy())[1])

    do_UMAP(full_x=cleaned_full_df[cleaned_full_df.columns[:-1]].to_numpy(), full_y=cleaned_full_df[cleaned_full_df.columns[-1]].to_numpy(),
            hundred_x=mod_data3, hundred_y=y_data_for_100, five_hundred_x=mod_data4, five_hundred_y=y_data_for_500,
            on_x=mod_data, on_y=y_data_for_on, name_of_exp=name_of_exp, num_of_neurons=np.shape(cleaned_full_df.to_numpy())[1])

    # ----> dPCA [Demixed PCA]
    # for running dPCA on data --> compare to vanilla PCA and LSTM VAE
    # run_dPCA(data=cleaned_data)

    #   ----------------------------------- CLASSIFIER TO TEST CHANGE IN LABEL ENCODING  ----------------------------------------------

    # --> Multinomial Logistic Regression
    multinomial_log_regression(train=norm_split_results[0], test=norm_split_results[1], num_of_neurons=np.shape(cleaned_full_df)[1], data=cleaned_full_df,
                          hundred_ms_data=mod_data3, five_hundred_ms_data=mod_data4, on_data=mod_data, y_on=y_data_for_on,
                          y_100=y_data_for_100, y_500=y_data_for_500, exp_name=name_of_exp)


# plot all of the slopes for the first 10 points (or PCA components) vs total variance captured
plot_slopes_per_exp(global_slopes, name_of_exp_array)


# ------------ doing some initial analysis on the data (PCA + Multinomial Log Regression Classifier) -------------------
# todo: figure out a way to implement halving_grid_search_CV for hyperparameter optimization
# todo: try out demixed PCA as well --> PCA taking into consideration task labels
# todo: debug the pi-VAE package and run on data in similar way --> VAE taking into consideration task labels
# todo: do avalanche analysis and analyze performance or compare coding dynamics during these frames + coarse graining
# todo: maybe incorporate t-sne for visualizing higher dimensional latent space (not currently an issue)
