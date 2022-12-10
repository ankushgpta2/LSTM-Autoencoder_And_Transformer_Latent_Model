import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import wandb
from datetime import date
import random


def plot_latent(recon_latent_holder, input_labels, autoencoder_parameters, general_parameters, mlp_parameters, dataset, latent_dim, experiment_name, loss_type, epochs, vae_reg_flag, batch_layers_flag, ax):
    if loss_type == 'only_reconstruction':
        loss_type = 'MSE(recon)'
    elif loss_type == 'kl_with_post_and_embed':
        loss_type = 'KL(post,embed)+MSE(recon,original)'
    title2 = '[' + 'batch=' + str(autoencoder_parameters['autoencoder_batch_size']) + ', ' + 'dataset=' + dataset + ', ae_reg=' + \
             str(autoencoder_parameters['autoencoder_reg']) + ', loss type= ' + str(loss_type) + ', lr=' + \
             str(autoencoder_parameters['autoencoder_learning_rate']) + ', latent=' + str(latent_dim) + ', vae reg=' + str(vae_reg_flag) + ', epochs=' + str(epochs) + ', noise=' + \
             str(general_parameters['threshold_percentile']*100) + '% lowest activity' + ', scaling=' + str(general_parameters['scaling']) + ', batch norm=' + str(batch_layers_flag) + ']\n'
    title3 = 'Experiment Name= ' + experiment_name + '\n'
    title4 = 'Date Generated= ' + str(date.today())
    colors = ['blue', 'red', 'green', 'orange', 'yellow', 'brown', 'gray', 'purple', 'black']
    subset = np.array([x for x in input_labels[:np.where(np.isnan(recon_latent_holder))[0][0]]])
    for x in np.unique(subset):
        indices = np.argwhere(subset == x)
        if latent_dim == 2:
            data1 = recon_latent_holder[indices[:, 0], 0]
            data2 = recon_latent_holder[indices[:, 0], 1]
            plt.scatter(data1, data2, color=colors[int(x)], alpha=0.8, s=50, label='Class = ' + str(x))
            title1 = '2D LATENT SPACE\n'
            plt.title(title1 + title2 + title3 + title4, fontweight='bold', fontsize=10)
            plt.xlabel('Latent 1'), plt.ylabel('Latent 2'), plt.legend()
        elif latent_dim == 3:
            data1 = recon_latent_holder[indices, 0]
            data2 = recon_latent_holder[indices, 1]
            data3 = recon_latent_holder[indices, 2]
            ax.scatter3D(data1, data2, data3, color=colors[int(x)], alpha=0.2, s=20, label='Class = ' + str(x))
            title1 = '3D LATENT SPACE\n'
            ax.set_title(title1 + title2 + title3 + title4, fontweight='bold', fontsize=10)
            ax.set_xlabel('Latent 1'), ax.set_ylabel('Latent 2'), ax.legend()


def plot_custom_loss_metrics(i, acc_holder, loss_holder, class_acc_holder, class_loss_holder, global_time_holder, time_holder, axes, previous, autoencoder_parameters, general_parameters, mlp_parameters, dataset, latent_dim, experiment_name,
                             loss_type, add_final_stuff, vae_reg_flag, batch_layers_flag):
    if loss_type == 'only_reconstruction':
        loss_type = 'MSE(recon)'
    elif loss_type == 'kl_with_post_and_embed':
        loss_type = 'KL(post,embed)+MSE(recon,original)'
    if i == 0:
        axes[0].set_title('Loss Across Epochs [' + loss_type + ']'), axes[1].set_title('Accuracy Across Epochs [Input Recon]')
        axes[2].set_title('Max Accuracy Across Epochs [Classification]')
        axes[3].set_title('Loss Across Epochs [Classification]')
        axes[0].set_xlabel('Epochs'), axes[1].set_xlabel('Epochs'), axes[0].set_ylabel('Mean Loss'), axes[1].set_ylabel('Accuracy'), axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Accuracy'), axes[3].set_xlabel('Epochs'), axes[3].set_ylabel('Mean Loss')

    if i == 1:
        axes[0].plot([i - 1, i], [previous[0][0], np.mean(loss_holder)], color='green', label='Mean Loss in Mini Batch', linewidth=2)
        axes[1].plot([i - 1, i], [previous[1][0], np.max(acc_holder)], color='red', label='Max Accuracy in Mini Batch', linewidth=2)
        axes[1].plot([i - 1, i], [previous[1][1], np.average(acc_holder)], color='dodgerblue', label='Mean Accuracy in Mini Batch', linewidth=2)
        axes[1].plot([i - 1, i], [previous[1][2], np.median(acc_holder)], color='black', label='Median Accuracy in Mini Batch', linewidth=2)
        axes[2].plot([i - 1, i], [previous[2][0], np.max(class_acc_holder)], color='red', label='Max Accuracy in Mini Batch', linewidth=2)
        axes[2].plot([i - 1, i], [previous[2][2], np.median(class_acc_holder)], color='black', label='Median Accuracy in Mini Batch', linewidth=2)
        axes[1].legend(), axes[2].legend(), axes[0].legend()

    if add_final_stuff is True:
        title1 = 'LOSS AND ACCURACY METRICS\n'
        title2 = '[' + 'batch=' + str(autoencoder_parameters['autoencoder_batch_size']) + ', ' + 'dataset=' + dataset + ', ae_reg=' + \
                 str(autoencoder_parameters['autoencoder_reg']) + ', loss type = ' + str(loss_type) + ', lr=' + \
                 str(autoencoder_parameters['autoencoder_learning_rate']) + ', latent=' + str(latent_dim) + ', vae reg=' + str(vae_reg_flag) + ', noise=' + \
                 str(general_parameters['threshold_percentile'] * 100) + '% lowest activity' + ', scaling=' + str(general_parameters['scaling']) + ', total time=' + str(np.round(np.sum(global_time_holder), 2)) + \
                 ', time/epoch=' + str(np.round((np.sum(global_time_holder) / i), 2)) + ', batch norm=' + str(batch_layers_flag) + ']\n'
        title3 = 'Experiment Name= ' + experiment_name + '\n'
        title4 = 'Date Generated= ' + str(date.today())
        plt.suptitle(title1 + title2 + title3 + title4, fontweight='bold', fontsize=9)
        axes[3].plot([i - 1, i], [previous[3][0], np.mean(class_loss_holder)], color='dodgerblue', label='Mean Loss in Mini Batch', linewidth=2)
        axes[3].legend()

    if i > 1 and add_final_stuff is False:
        axes[1].plot([i - 1, i], [previous[1][0], np.max(acc_holder)], color='red', linewidth=2)
        axes[1].plot([i - 1, i], [previous[1][1], np.average(acc_holder)], color='dodgerblue', linewidth=2)
        axes[1].plot([i - 1, i], [previous[1][2], np.median(acc_holder)], color='black', linewidth=2)
        axes[2].plot([i - 1, i], [previous[2][0], np.max(class_acc_holder)], color='red', linewidth=2)
        axes[2].plot([i - 1, i], [previous[2][2], np.median(class_acc_holder)], color='black', linewidth=2)
        axes[3].plot([i-1, i], [previous[3][0], np.mean(class_loss_holder)], color='dodgerblue', linewidth=2)
        axes[0].plot([i-1, i], [previous[0][0], np.mean(loss_holder)], color='green', linewidth=2)
    previous = [[np.mean(loss_holder)], [np.max(acc_holder), np.average(acc_holder), np.median(acc_holder)], [np.max(class_acc_holder), np.average(class_acc_holder), np.median(class_acc_holder)],
                [np.mean(class_loss_holder)]]
    return previous, axes


def plot_latent_trajectory(previous, counter, latent_points, class_num):
    """
    simply take the dataset that was completely encoded and plot the trajectories for each class through time
    """
    colors = ['blue', 'red', 'green', 'orange', 'yellow', 'brown', 'gray', 'purple', 'black']
    if counter != 0:
        plt.plot([previous[0], np.mean(latent_points[:, 0])], [previous[1], np.mean(latent_points[:, 1])], color=colors[class_num])
    previous = [np.mean(latent_points[:, 0]), np.mean(latent_points[:, 1])]
    print(previous)
    return previous
