import numpy as np
import pandas as pd
from shallow_ml_plotting import initialize_subplot_axes, specify_fontsize_dimensions_subplot
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.linear_model import PoissonRegressor
from scipy.stats import poisson
# from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn.metrics
import seaborn as sns
import stats
import scipy
from numpy.random import rand, randn, randint
from sklearn.manifold import TSNE
from sklearn import *
import math
import sys
import umap
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP
import plotly.express as px
sys.path.append('/Users/ankushgupta/Documents/Python/VAE_for_gabor/shallow_ml/dPCA/python')
from dPCA import *
from dPCA import dPCA
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from shallow_ml_plotting import *


def run_PCA(dataset_for_diff_recording_periods, pca_rank_coarse_graining, experiment_name):
    # Get 2 and 3 PCA components ------------------------------------------------------------>
    pca_data = {}
    for x in range(2, 4):
        pca_model = PCA(n_components=x)
        pca_values_for_entire = pca_model.fit_transform(dataset_for_diff_recording_periods['Full'][0])
        pca_values_for_on = pca_model.fit_transform(dataset_for_diff_recording_periods['On'][0])
        pca_values_for_100 = pca_model.fit_transform(dataset_for_diff_recording_periods['Hundred'][0])
        pca_values_for_500 = pca_model.fit_transform(dataset_for_diff_recording_periods['Five Hundred'][0])
        pca_data[str(x)] = pca_values_for_entire, pca_values_for_on, pca_values_for_100, pca_values_for_500

    labels_for_entire = dataset_for_diff_recording_periods['Full'][1]
    labels_for_100 = dataset_for_diff_recording_periods['Hundred'][1]
    labels_for_on = dataset_for_diff_recording_periods['On'][1]
    labels_for_500 = dataset_for_diff_recording_periods['Five Hundred'][1]
    labels = [labels_for_entire, labels_for_on, labels_for_100, labels_for_500]

    # plot 2d and 3d scatter for PCA
    plot_pca_scatter(pca_data, labels, experiment_name)

    # plot histogram for PCA values for first component and second
    plot_pca_histogram(pca_data, experiment_name)

    # plot variance vs PCA component plot
    # slopes = variance_vs_pca_components(dataset_for_diff_recording_periods, pca_rank_coarse_graining=1, experiment_name=experiment_name)

    # run multinomial log regression
    # multinomial_log_regression(dataset_for_diff_recording_periods, experiment_name)

    # run UMAP
    do_UMAP(dataset_for_diff_recording_periods, experiment_name)

    return mod_data, mod_data3, mod_data4, y_data_for_on, y_data_for_100, y_data_for_500, slopes


def variance_vs_pca_components(dataset_for_diff_recording_periods, pca_rank_coarse_graining, experiment_name):
    # plot variance encoded vs increasing number of PCA components on different sets of frames
    num_of_neurons = np.shape(dataset_for_diff_recording_periods['Full'][0])[1]
    total_iterations = np.floor(num_of_neurons / pca_rank_coarse_graining)
    total_var_array = np.empty([10, 10000])
    for x in range(int(total_iterations)):
        pca_model = PCA(n_components=x * pca_rank_coarse_graining)
        pca_values_for_entire = pca_model.fit_transform(dataset_for_diff_recording_periods['Full'][0])
        total_var_array[0, x] = pca_model.explained_variance_ratio_.sum() * 100
        pca_values_for_on = pca_model.fit_transform(dataset_for_diff_recording_periods['On'][0])
        total_var_array[1, x] = pca_model.explained_variance_ratio_.sum() * 100
        pca_values_for_100 = pca_model.fit_transform(dataset_for_diff_recording_periods['Hundred'][0])
        total_var_array[2, x] = pca_model.explained_variance_ratio_.sum() * 100
        pca_values_for_500 = pca_model.fit_transform(dataset_for_diff_recording_periods['Five Hundred'][0])
        total_var_array[3, x] = pca_model.explained_variance_ratio_.sum() * 100

    slopes = plot_variance_vs_pca_component(total_var_array, total_iterations, experiment_name)
    return slopes


def do_UMAP(dataset_for_diff_recording_periods, experiment_name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    axes = [ax1, ax2, ax3, ax4]
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2)
    axes2 = [ax5, ax6, ax7, ax8]
    titles = ['On and Off Stimulus Periods', 'On Stimulus Periods Only', 'First 100ms', 'First 500ms']
    colors = ['red', 'blue', 'green', 'orange', 'yellow', 'gray', 'black', 'pink']

    # plot each time period within recording with standard UMAP (default input parameters)
    for i in range(4):
        key_list = [x for x in dataset_for_diff_recording_periods.keys()]
        key = key_list[i]
        if i == 0:
            label_indices = dataset_for_diff_recording_periods[key][1] != 8
            x_data = dataset_for_diff_recording_periods[key][0][label_indices]
            y_data = dataset_for_diff_recording_periods[key][1][label_indices]
        else:
            x_data = dataset_for_diff_recording_periods[key][0]
            y_data = dataset_for_diff_recording_periods[key][1]

        # initialize the model
        embedding_non_param = umap.UMAP(min_dist=0, random_state=42).fit_transform(x_data)
        embedding_param = ParametricUMAP(n_epochs=50, verbose=True).fit_transform(x_data)

        # plot the 2D UMAP latent space
        for x in range(8):
            orientation_indices = y_data == x
            axes[i].scatter(embedding_non_param[orientation_indices, 0], embedding_non_param[orientation_indices, 1], s=5, color=colors[x], alpha=0.4, cmap='Spectral', label='x')
            axes2[i].scatter(embedding_param[orientation_indices, 0], embedding_param[orientation_indices, 1], s=5, color=colors[x], alpha=0.4, cmap='Spectral', label='x')
        axes[i].set_title(titles[i]), axes2[i].set_title(titles[i])
        # axes[i].set_xlim([-1, 17]), axes[i].set_ylim([-2, 15])
        axes[i].set_xlabel('Latent 1')
        axes[i].set_ylabel('Latent 2')
    fig.suptitle('Non-Parameterized UMAP For Different Recording Periods [' + experiment_name + ']', fontweight='bold')
    fig2.suptitle('Parameterized UMAP For Different Recording Periods [' + experiment_name + ']', fontweight='bold')
    plt.legend()
    plt.show()

    # plot other UMAP to visualize
    umap_data = umap.UMAP(n_components=2, n_neighbors=optimal_n_neighbor, min_dist=0, random_state=42).fit_transform(datasets_for_diff_recording_periods['Full'][0])


def multinomial_log_regression(dataset_for_diff_recording_periods, experiment_name):

    # prep the portion of entire df for inputs and outputs respectively
    num_of_neurons = np.shape(dataset_for_diff_recording_periods['Full'][0])[1]

    train_accs = np.empty((500, 500))
    test_accs = np.empty((500, 500))
    mse_array = np.empty((500, 500))

    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    for i in range(4):
        key_list = [x for x in dataset_for_diff_recording_periods.keys()]
        key = key_list[i]
        x_data = dataset_for_diff_recording_periods[key][0]
        y_data = dataset_for_diff_recording_periods[key][1]

        for x in range(1, num_of_neurons-1):
            # specify pca model + translate entire dataframe into PCA space for further parsing
            pca_model = PCA(n_components=x)
            values = pca_model.fit_transform(x_data)
            transformed_data = pd.DataFrame(values)
            # split the x_pca dataset
            x_train_pca, x_test_pca, y_train, y_test = train_test_split(transformed_data, y_data, test_size=0.20)
            # initialize logistic regression model with L2 penalty
            logistic = LogisticRegression(multi_class='multinomial', max_iter=10000, penalty='l2', solver='lbfgs')
            logistic.fit(x_train_pca, y_train)
            # get predictions
            pca_transformed_predictions = logistic.predict(x_test_pca)
            probabilities_on_test = logistic.predict_proba(x_test_pca)
            mse = metrics.mean_squared_error(y_true=y_test, y_pred=pca_transformed_predictions)
            mse_array[i, x] = mse
            train_accs[i, x] = accuracy_score(y_train, logistic.predict(x_train_pca))
            test_accs[i, x] = accuracy_score(y_test, pca_transformed_predictions)
            # print(str(x) + ' --> MSE = ' + str(mse) + ' --> ' + 'Training Accuracy = ' + str(train_accs[i, x]) + ' --> Test Accuracy = ' + str(test_accs[i, x]))

    plot_multinomial_log_regression_results(num_of_neurons, train_accs, test_accs, mse_array, experiment_name)


def run_dPCA(data):
    # initialize the dPCA model
    dpca = dPCA.dPCA(labels='st', regularizer='auto')
    dpca.protect = ['t']  # this essentially does not shuffle the frames that correspond to certain trials
    # load / determine some necessary data stuff
    num_of_frames, num_of_neurons = np.shape(data)
    N, T, S = num_of_neurons, num_of_frames, 8  # I am using 8 classes (0-7 for frames during actual stimuli
                                                # presentations and 8 = off stimulus periods)... to clean up a little
                                                # since we know that most of the stimulus is encoded in beginning
    # build two latent factors
    zt = (np.arange(T) / float(T))
    zs = (np.arange(S) / float(S))

    # build trial-by trial data
    noise, n_samples = 0.2, 20  # second value corresponds to the number of presentations per stimuli type
    trialR = noise * randn(n_samples, N, S, T)
    trialR += randn(N)[None, :, None, None] * zt[None, None, None, :]
    trialR += randn(N)[None, :, None, None] * zs[None, None, :, None]

    Z = dpca.fit_transform(data.to_numpy(), trialR)

    # do some plotting of the compressed space (generated latent data or Z from demixed PCA)
    time = np.arange(T)
    plt.figure(figsize=(16, 7))
    plt.subplot(131)
    for s in range(S):
        plt.plot(time, Z['t'][0, s])
    plt.title('1st time component')
    plt.subplot(132)
    for s in range(S):
        plt.plot(time, Z['s'][0, s])
    plt.title('1st stimulus component')
    plt.subplot(133)
    for s in range(S):
        plt.plot(time, Z['st'][0, s])
    plt.title('1st mixing component')
    plt.show()


def UMAP_param_neural_net(full_x, full_y, hundred_x, hundred_y, five_hundred_x, five_hundred_y, on_x, on_y, name_of_exp, num_of_neurons):
    """
    parameterize via simple neural network instead and pass into UMAP
    """
    datasets = [[full_x, full_y], [hundred_x, hundred_y], [five_hundred_x, five_hundred_y], [on_x, on_y]]
    final_figs = initialize_subplot_axes(number_of_subplots=1, title='Trained Neural Network for UMAP Parameterization [' + str(name_of_exp) + ']', dimensions=[2, 2],
                                      fontsize=10)

    axes = final_figs.axes
    titles = ['On and Off Stimulus Periods', 'First 100ms of Stimulus Period', 'First 500ms of Stimulus Period', 'On Periods Only']
    for x in range(4):
        x_data = datasets[x][0]
        y_data = datasets[x][1]

        # UTILIZE A SIMPLE MLP MODEL --> 3-layer 100-neuron fully-connected neural network
        embedding = ParametricUMAP(n_epochs=100, verbose=True).fit_transform(x_data)
        targets = [0, 1, 2, 3, 4, 5, 6, 7]

        # plot the results
        axes[x].scatter(embedding[:, 0], embedding[:, 1], c=y_data, cmap="tab10",
                        s=1, alpha=0.5, rasterized=True)
        axes[x].axis('equal')
        # axes[x].xlim([-10, 12]), axes[x].ylim([-13, 10])
        if x == 2:
            axes[x].set_xlabel('Latent 1'), axes[x].set_ylabel('Latent 2')
        elif x == 0:
            axes[x].set_ylabel('Latent 2')
        elif x == 3:
            axes[x].set_xlabel('Latent 1')
        axes[x].set_title(titles[x])
    plt.show()
    """
    # CONVOLUTIONAL NEURAL NETWORK + UMAP --------------------------------->
    # specify the encoder structure
    batch_size = 100
    for x in range(4):

        x_data = datasets[x][0]
        y_data = datasets[x][1]
        dims = (1, np.shape(x_data)[1], 1)
        print(dims)
        n_components = 2

        # place the created encoder onto the UMAP model
        embedder = ParametricUMAP(
            encoder=encoder,
            decoder=decoder,
            dims=dims,
            parametric_reconstruction=True,
            autoencoder_loss=True,
            verbose=True,
        )

        x_data_sub = x_data[0:2000, :]
        x_data_reshaped = np.reshape(x_data_sub, (2000, np.shape(x_data)[1]))

        embedding = embedder.fit_transform(x_data_reshaped)
        # plot the results
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y_data.astype(int), cmap="tab10",
                        s=1, alpha=0.5, rasterized=True)
        plt.axis('equal')
        plt.title(titles[x], fontsize=10)
        plt.colorbar(sc, ax=ax)
    """
