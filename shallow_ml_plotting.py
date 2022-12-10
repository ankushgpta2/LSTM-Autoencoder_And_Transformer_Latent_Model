import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


def initialize_subplot_axes(number_of_subplots, title, dimensions, fontsize):
    if number_of_subplots == 1:
        if sum(dimensions) == 3:
            fig1, (ax1, ax2) = plt.subplots(1, 2)
        elif sum(dimensions) == 4:
            fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            plt.suptitle(title, fontsize=fontsize, fontweight='bold')
            figure = specify_fontsize_dimensions_subplot(figure=fig1)
            final_figs = figure
    elif number_of_subplots == 2:
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        plt.suptitle(title[0], fontsize=fontsize, fontweight='bold')
        figure1 = figure = specify_fontsize_dimensions_subplot(figure=fig1)
        fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2)
        plt.suptitle(title[1], fontsize=fontsize, fontweight='bold')
        figure2 = figure = specify_fontsize_dimensions_subplot(figure=fig2)
        final_figs = [figure1, figure2]
    elif number_of_subplots == 3:
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        plt.suptitle(title[0], fontsize=fontsize, fontweight='bold')
        figure1 = figure = specify_fontsize_dimensions_subplot(figure=fig1)

        fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2)
        plt.suptitle(title[1], fontsize=fontsize, fontweight='bold')
        figure2 = figure = specify_fontsize_dimensions_subplot(figure=fig2)

        fig3, ((ax9, ax10), (ax11, ax12)) = plt.subplots(2, 2)
        plt.suptitle(title[2], fontsize=fontsize, fontweight='bold')
        figure3 = figure = specify_fontsize_dimensions_subplot(figure=fig3)

        fig4, ((ax13, ax14), (ax15, ax16)) = plt.subplots(2, 2)
        plt.suptitle(title[3], fontsize=fontsize, fontweight='bold')
        figure4 = figure = specify_fontsize_dimensions_subplot(figure=fig4)

        final_figs = [figure1, figure2, figure3, figure4]
    return final_figs


def specify_fontsize_dimensions_subplot(figure):
    axes = figure.axes
    axes[0].tick_params(labelsize=7), axes[1].tick_params(labelsize=7), axes[2].tick_params(labelsize=7), axes[3].tick_params(
        labelsize=7)
    box, box2, box3, box4 = figure.axes[0].get_position(), figure.axes[1].get_position(), figure.axes[2].get_position(), figure.axes[3].get_position()
    box2.x0, box2.x1, box4.x0, box4.x1, box3.y0, box3.y1, box4.y0, box4.y1 = box2.x0 + 0.05, box2.x1 + 0.05, \
                                                                             box4.x0 + 0.05, box4.x1 + 0.05, \
                                                                             box3.y0 - 0.02, box3.y1 - 0.02, \
                                                                             box4.y0 - 0.02, box4.y1 - 0.02
    figure.axes[0].set_position(box)
    figure.axes[1].set_position(box2)
    figure.axes[2].set_position(box3)
    figure.axes[3].set_position(box4)
    return figure


def plot_slopes_per_exp(global_slopes, name_of_exp_array):
    labels = ['On and Off Periods', 'On Periods', '100ms', '500ms']
    colors = ['green', 'red', 'black', 'orange', 'deepskyblue', 'hotpink']
    global_slopes = np.array(global_slopes)
    for x in range(np.shape(global_slopes)[0]):
        subset = global_slopes[x, :]  # for the specific experiment
        for i in range(3):
            plt.scatter(i, subset[i], color=colors[x], s=40)
            if i == 2:
                plt.scatter(3, subset[3], color=colors[x], label=name_of_exp_array[x], s=40)
        plt.plot(range(4), subset, color='grey')
    plt.legend()
    plt.title('Slope For First 10 PCA Components Across Experiments', fontsize=12, fontweight='bold')
    plt.xlabel('Portion of Recording'), plt.ylabel('Slope in Log-Log')
    plt.xticks([0, 1, 2, 3], labels=[labels[0], labels[1], labels[2], labels[3]])
    plt.show()


def get_pca_scatter_information():
    targets = [0, 1, 2, 3, 4, 5, 6, 7]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey']
    # for 2D
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    axes = [ax1, ax2, ax3, ax4]
    # for 3D
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(221, projection='3d')
    ax2 = fig2.add_subplot(222, projection='3d')
    ax3 = fig2.add_subplot(223, projection='3d')
    ax4 = fig2.add_subplot(224, projection='3d')
    axes2 = [ax1, ax2, ax3, ax4]

    titles = ['On and Off Stimulus Periods', 'On Stimulus Periods Only', 'First 100ms', 'First 500ms']
    counter = 0
    for x in range(len(axes)):
        axes[x].set_title(titles[counter], fontweight='bold'), axes2[x].set_title(titles[counter], fontweight='bold')
        axes[x].set_ylabel('PC 2'), axes[x].set_xlabel('PC 1'), axes2[x].set_ylabel('PC 2'), axes2[x].set_xlabel('PC 1'), axes2[x].set_zlabel('PC 3')
        axes[x].set_ylim([-10, 10]), axes2[x].set_ylim([-10, 10])
        counter += 1
    return fig, fig2, axes, axes2, targets, colors


def plot_pca_scatter(pca_data, labels, experiment_name):
    fig, fig2, axes, axes2, targets, colors = get_pca_scatter_information()
    for y in pca_data.keys():
        counter = 0
        for x in targets:
            indices1 = np.where(labels[0] == x)
            indices2 = np.where(labels[1] == x)
            indices3 = np.where(labels[2] == x)
            indices4 = np.where(labels[3] == x)
            indices = [indices1, indices2, indices3, indices4]
            for z in range(4):
                if y == '2':
                    axes[z].scatter(pca_data[y][z][indices[z], 0], pca_data[y][z][indices[z], 1], s=20, alpha=0.3, color=colors[counter], label=x)
                elif y == '3':
                    axes2[z].scatter(pca_data[y][z][indices[z], 0], pca_data[y][z][indices[z], 1], pca_data[y][z][indices[z], 2], s=20, alpha=0.3, color=colors[counter], label=x)
            counter += 1
        for z in range(4):
            if y == '2':
                axes[z].legend(ncol=4, loc='upper right')
            elif y == '3':
                axes2[z].legend(ncol=4, loc='upper right', fontsize=8)
    fig.suptitle('2D PCA [' + experiment_name + ']', fontweight='bold', fontsize=15)
    fig2.suptitle('3D PCA [' + experiment_name + ']', fontweight='bold', fontsize=15)
    # ax.grid()
    plt.show()


def plot_pca_histogram(pca_data, experiment_name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2)
    axes = [ax1, ax2, ax3, ax4]
    axes2 = [ax5, ax6, ax7, ax8]
    titles = ['On and Off Stimulus Periods', 'On Stimulus Periods Only', 'First 100ms', 'First 500ms']
    counter = 0
    for x in pca_data['2']:
        sns.histplot(x[:, 0], ax=axes[counter], alpha=0.5, color='blue', label='Latent 1', stat='probability', binwidth=0.5)
        sns.histplot(x[:, 1], ax=axes[counter], alpha=0.5, color='red', label='Latent 2', stat='probability', binwidth=0.5)
        axes[counter].plot([], [], ' ', label='Bin Width = ' + str(0.5))
        axes[counter].set_title(titles[counter])
        axes[counter].set_xlabel('Value in 2D Latent Space'), axes[counter].set_ylabel('Probability')
        axes[counter].set_xlim([-15, 15])
        axes[counter].set_ylim([0, 0.5])
        axes[counter].legend()
        counter += 1

    counter = 0
    for y in pca_data['3']:
        sns.histplot(y[:, 0], ax=axes2[counter], alpha=0.5, color='blue', label='Latent 1', stat='probability', binwidth=0.5)
        sns.histplot(y[:, 1], ax=axes2[counter], alpha=0.5, color='red', label='Latent 2', stat='probability', binwidth=0.5)
        sns.histplot(y[:, 1], ax=axes2[counter], alpha=0.5, color='green', label='Latent 3', stat='probability', binwidth=0.5)
        axes2[counter].plot([], [], ' ', label='Bin Width = ' + str(0.5))
        axes2[counter].set_title(titles[counter])
        axes2[counter].set_xlabel('Value in 2D Latent Space'), axes2[counter].set_ylabel('Probability')
        axes2[counter].set_xlim([-15, 15])
        axes2[counter].set_ylim([0, 0.5])
        axes2[counter].legend()
        counter += 1
    fig.suptitle('Histogram in 2D PCA Space [' + experiment_name + ']', fontweight='bold')
    fig2.suptitle('Histogram in 3D PCA Space [' + experiment_name + ']', fontweight='bold')
    plt.show()


def plot_variance_vs_pca_component(total_var_array, total_iterations, experiment_name):
    titles = ['On and Off Stimulus Periods', 'On Stimulus Periods Only', 'First 100ms', 'First 500ms']
    colors = ['red', 'blue', 'green', 'black']
    slopes = []
    for x in range(4):
        plt.plot(range(int(total_iterations)), total_var_array[x, range(int(total_iterations))], linewidth=1,
                 label=titles[x], color=colors[x])
        plt.scatter(range(int(total_iterations)), total_var_array[x, range(int(total_iterations))], color='red', s=12, marker='|')
        # fit the first ten points and get the slope of each line
        slope, intercept = np.polyfit(range(10), total_var_array[x, range(10)], 1)
        slopes.append(math.log(slope))
    plt.legend(loc='upper left', ncol=2), plt.xlabel('# of PCA Components (Ranked/Summed)'), plt.ylabel('Amount of Variance Explained (%)')
    plt.title('Total Variance Vs # of PCA Components [' + str(experiment_name) + ']', fontweight='bold', fontsize=12)
    plt.ylim([3, 105])
    # plt.xscale('log'), plt.yscale('log')
    plt.show()
    return slopes


def plot_multinomial_log_regression_results(num_of_neurons, train_accs, test_accs, mse_array, experiment_name):
    x_values_axis = range(1, num_of_neurons - 1)
    legend_labels = ['Train Accuracy', 'Test Accuracy']
    titles = ['Accuracy Vs PCA Component [Full]', 'MSE For Predictions On Test Set [Full]', 'Accuracy Vs PCA Component [On Periods]',
                  'MSE For Predictions On Test Set [On Periods]', 'Accuracy Vs PCA Component [100ms On Periods]', 'MSE For Predictions On Test Set [100ms On Periods]',
                  'Accuracy Vs PCA Component [500ms On Periods]', 'MSE For Predictions On Test Set [500ms On Periods]']
    y_labels = ['Accuracy Predicting Orientation Labels', 'Mean Square Error (MSE)']
    x_labels = ['Principal Components']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2)
    axes_list1 = [ax1, ax2, ax3, ax4]
    axes_list2 = [ax5, ax6, ax7, ax8]
    final_list = ([axes_list1[0], axes_list1[1]], [axes_list1[2], axes_list1[3]], [axes_list2[0], axes_list2[1]],
                  [axes_list2[2], axes_list2[3]])
    i = 0
    for x in range(4):
        axis1 = final_list[x][0]
        axis2 = final_list[x][1]

        axis1.scatter(x_values_axis, train_accs[x, x_values_axis], color='blue', s=10, label=legend_labels[0])
        m, b = np.polyfit(x_values_axis, train_accs[x, x_values_axis], deg=1)
        axis1.plot(x_values_axis, m * x_values_axis + b, color='green')
        axis1.scatter(x_values_axis, test_accs[x, x_values_axis], color='red', s=10, label=legend_labels[1])
        m, b = np.polyfit(x_values_axis, test_accs[x, x_values_axis], deg=1)
        axis1.plot(x_values_axis, m * x_values_axis + b, color='green')
        axis1.legend(), axis1.set_title(titles[i]), axis1.set_ylabel(y_labels[0])

        axis2.scatter(x_values_axis, mse_array[x, x_values_axis], s=10)
        m, b = np.polyfit(x_values_axis, mse_array[x, x_values_axis], deg=1)
        axis2.plot(x_values_axis, m * x_values_axis + b, color='green')
        axis2.set_ylabel(y_labels[1]), axis2.set_title(titles[i+1])
        i += 2

        if x == 1 or x == 3:
            axis1.set_xlabel(x_labels[0])
            axis2.set_xlabel(x_labels[0])
    fig.suptitle('Accuracy and MSE For Different Recording Periods [' + experiment_name + ']', fontweight='bold')
    fig2.suptitle('Accuracy and MSE For Different Recording Periods [' + experiment_name + ']', fontweight='bold')
    plt.show()
