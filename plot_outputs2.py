import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_loss(results_holder, history_holder, general_parameters, name, exp_name, vae_parameters, mlp_parameters):
    # for training loss and accuracy
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    axes = [ax1, ax2, ax3, ax4]
    # for validation loss and accuracy
    fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    axes3 = [ax1, ax2, ax3, ax4]
    fig4, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    axes4 = [ax1, ax2, ax3, ax4]
    # for prediction metrics on test set + validation and training means
    fig2, ((ax3, ax4), (ax1, ax2)) = plt.subplots(2, 2)
    axes2 = [ax3, ax4, ax1, ax2]
    if general_parameters['run_vae']:
        models = ['VAE']
    else:
        models = ['MLP']

    colors = ['red', 'blue', 'green', 'black', 'orange', 'yellow', 'purple', 'dodgerblue', 'darkgray',  'pink']
    i = 0
    name_counter = 0
    color_counter = 0
    latent_dim_counter = 0
    for y in models:
        keys_for_model_history = [x for x in history_holder.keys() if y in x]
        latent_list = [x for x in vae_parameters['vae_latent_dim']]
        for x in keys_for_model_history:
            # get entire data for the particular VAE model with specific latent dimension
            data_for_certain_dnn = history_holder[x]
            # first plot loss and accuracy from training -------------------------------------------->
            train_loss = [x for x in data_for_certain_dnn[0][1]]
            train_accs = [x * 100 for x in data_for_certain_dnn[1][1]]
            val_loss = [x for x in data_for_certain_dnn[2][1]]
            val_accs = [x * 100 for x in data_for_certain_dnn[3][1]]
            if latent_dim_counter == 0 or latent_dim_counter == 1 or latent_dim_counter == 2 or latent_dim_counter == 3:
                axes[i].plot(range(1, len(train_loss) + 1), train_loss, color=colors[color_counter], linewidth=2, label='Training [' + str(x) + ']')
                axes[i+1].plot(range(1, len(train_accs) + 1), train_accs, color=colors[color_counter], linewidth=2, label='Training [' + str(x) + ']')
            elif latent_dim_counter == 4 or latent_dim_counter == 5 or latent_dim_counter == 6 or latent_dim_counter == 7 or latent_dim_counter == 8:
                axes[i+2].plot(range(1, len(train_loss) + 1), train_loss, color=colors[color_counter], linewidth=2, label='Training [' + str(x) + ']')
                axes[i+3].plot(range(1, len(train_accs) + 1), train_accs, color=colors[color_counter], linewidth=2, label='Training [' + str(x) + ']')
            # Now plot validation plots
            if latent_dim_counter == 0 or latent_dim_counter == 1:
                axes3[i].plot(range(1, len(val_loss) + 1), val_loss, color=colors[color_counter], linewidth=1, label='Validation [' + str(x) + ']')
                axes3[i + 1].plot(range(1, len(val_accs) + 1), val_accs, color=colors[color_counter], linewidth=1, label='Validation [' + str(x) + ']')
            elif latent_dim_counter == 2 or latent_dim_counter == 3:
                axes3[i+2].plot(range(1, len(val_loss) + 1), val_loss, color=colors[color_counter], linewidth=1, label='Validation [' + str(x) + ']')
                axes3[i+3].plot(range(1, len(val_accs) + 1), val_accs, color=colors[color_counter], linewidth=1, label='Validation [' + str(x) + ']')
            elif latent_dim_counter == 4 or latent_dim_counter == 5:
                axes4[i].plot(range(1, len(val_loss) + 1), val_loss, color=colors[color_counter], linewidth=1, label='Validation [' + str(x) + ']')
                axes4[i+1].plot(range(1, len(val_accs) + 1), val_accs, color=colors[color_counter], linewidth=1, label='Validation [' + str(x) + ']')
            elif latent_dim_counter == 6 or latent_dim_counter == 7 or latent_dim_counter == 8:
                axes4[i+2].plot(range(1, len(val_loss) + 1), val_loss, color=colors[color_counter], linewidth=1, label='Validation [' + str(x) + ']')
                axes4[i+3].plot(range(1, len(val_accs) + 1), val_accs, color=colors[color_counter], linewidth=1, label='Validation [' + str(x) + ']')
            # plot the mean training and validation metrics
            if latent_dim_counter == 0:
                axes2[i + 2].scatter(latent_list[latent_dim_counter], np.mean(train_loss), s=40, label='Training', color='red')
                axes2[i + 3].scatter(latent_list[latent_dim_counter], np.mean(train_accs), s=40, label='Training', color='red')
                axes2[i + 2].scatter(latent_list[latent_dim_counter], np.mean(val_loss), s=40, label='Validation', color='blue')
                axes2[i + 3].scatter(latent_list[latent_dim_counter], np.mean(val_accs), s=40, label='Validation', color='blue')
            else:
                axes2[i + 2].scatter(latent_list[latent_dim_counter], np.mean(train_loss), s=40, color='red')
                axes2[i + 3].scatter(latent_list[latent_dim_counter], np.mean(train_accs), s=40, color='red')
                axes2[i + 2].scatter(latent_list[latent_dim_counter], np.mean(val_loss), s=40, color='blue')
                axes2[i + 3].scatter(latent_list[latent_dim_counter], np.mean(val_accs), s=40, color='blue')
            color_counter += 1
            latent_dim_counter += 1
        # specify labels for training plots ----->
        axes[i].set_title('Training Loss'), axes[i].set_xlabel('Epochs'), axes[i].set_ylabel('Loss')
        axes[i].legend(ncol=2, loc='upper right')
        axes[i+1].set_title('Training Accuracy'), axes[i+1].set_xlabel('Epochs'), axes[i+1].set_ylabel('Accuracy')
        axes[i+1].legend(ncol=2, loc='lower right')
        axes[i+2].set_title('Training Loss'), axes[i+2].set_xlabel('Epochs'), axes[i+2].set_ylabel('Loss')
        axes[i+2].legend(ncol=2, loc='upper right')
        axes[i+3].set_title('Training Accuracy'), axes[i+3].set_xlabel('Epochs'), axes[i+3].set_ylabel('Accuracy')
        axes[i+3].legend(ncol=2, loc='lower right')
        fig.suptitle('Training Loss and Accuracy for ' + name, fontweight='bold')
        # specify labels for validation plots ----->
        axes3[i].set_title('Validation Loss'), axes3[i].set_xlabel('Epochs'), axes3[i].set_ylabel('Validation Loss')
        axes3[i].legend(ncol=2, loc='upper right')
        axes3[i + 1].set_title('Validation Accuracy'), axes3[i + 1].set_xlabel('Epochs'), axes3[i + 1].set_ylabel('Accuracy')
        axes3[i + 1].legend(ncol=2, loc='lower right')
        axes3[i + 2].set_title('Validation Loss'), axes3[i + 2].set_xlabel('Epochs'), axes3[i + 2].set_ylabel('Loss')
        axes3[i + 2].legend(ncol=2, loc='upper right')
        axes3[i + 3].set_title('Validation Accuracy'), axes3[i + 3].set_xlabel('Epochs'), axes3[i + 3].set_ylabel('Accuracy')
        axes3[i + 3].legend(ncol=2, loc='lower right')
        fig3.suptitle('Validation Loss and Accuracy for ' + name, fontweight='bold')
        axes4[i].set_title('Validation Loss'), axes4[i].set_xlabel('Epochs'), axes4[i].set_ylabel('Loss')
        axes4[i].legend(ncol=2, loc='upper right')
        axes4[i + 1].set_title('Validation Accuracy'), axes4[i + 1].set_xlabel('Epochs'), axes4[i + 1].set_ylabel('Accuracy')
        axes4[i + 1].legend(ncol=2, loc='lower right')
        axes4[i + 2].set_title('Validation Loss'), axes4[i + 2].set_xlabel('Epochs'), axes4[i + 2].set_ylabel('Loss')
        axes4[i + 2].legend(ncol=2, loc='upper right')
        axes4[i + 3].set_title('Validation Accuracy'), axes4[i + 3].set_xlabel('Epochs'), axes4[i + 3].set_ylabel('Accuracy')
        axes4[i + 3].legend(ncol=2, loc='lower right')
        fig4.suptitle('Validation Loss and Accuracy for ' + name, fontweight='bold')

        # now plot the predictions for the latent dimensions  -------------------------------------------->
        keys_for_model_results = [x for x in results_holder.keys() if y in x]
        latent_dim_counter = 0
        for x in keys_for_model_results:
            prediction_on_test_data = results_holder[x]
            axes2[i].scatter(latent_list[latent_dim_counter], prediction_on_test_data[0], color='black', s=40)  # loss I think
            axes2[i+1].scatter(latent_list[latent_dim_counter], prediction_on_test_data[1], color='black', s=40)  # accuracy I think
            latent_dim_counter += 1
        axes2[i].set_title('Prediction Loss On Test Set'), axes2[i].set_xlabel('Size of Latent Dimension'), axes2[i].set_ylabel('Prediction Loss'), axes2[i].set_xticks([x for x in latent_list])
        axes2[i+1].set_title('Prediction Accuracy On Test Set'), axes2[i+1].set_xlabel('Size of Latent Dimension'), axes2[i+1].set_ylabel('Prediction Accuracy'), axes2[i+1].set_xticks([x for x in latent_list])
        axes2[i+2].set_title('Mean Training and Validation Loss'), axes2[i+2].set_xlabel('Size of Latent Dimension'), axes2[i+2].set_ylabel('Average Training/Validation Loss')
        axes2[i+2].set_xticks([x for x in latent_list])
        axes2[i+3].set_title('Mean Training and Validation Accuracy'), axes2[i+3].set_xlabel('Size of Latent Dimension'), axes2[i+3].set_ylabel('Average Training/Validation Accuracy')
        axes2[i+3].set_xticks([x for x in latent_list]), axes2[i+2].legend(), axes2[i+3].legend()
        fig2.suptitle('Prediction, Validation, and Training Metrics For ' + name, fontweight='bold')
        i += 2
        name_counter += 1
    plt.show()


def plot_latent_2d(z_latent_test, test_labels, average, title, num_classes):
    color = ['blue', 'red', 'orange', 'brown', 'darkgray', 'black', 'yellow', 'green']
    # plot the 2D learned latent space --------------------------------------->
    plt.figure(figsize=(6, 6))
    # remove the points in the 8th class or not in any of the orientation gratings
    dictionary = {'latent1': z_latent_test[:, 0], 'latent2': z_latent_test[:, 1], 'orientations': test_labels}
    df = pd.DataFrame(dictionary)
    df = df[df['orientations'] != 8]
    # plot normal (without averaging)
    for x in range(num_classes):
        indices_for_class_subset = df.index[df['orientations'] == x].to_list()
        indices_for_class_subset.sort()
        first_latent = df['latent1'].loc[indices_for_class_subset].to_list()
        second_latent = df['latent2'].loc[indices_for_class_subset].to_list()
        plt.scatter(first_latent, second_latent, color=color[x], alpha=.85, s=20, label='Grating Class = ' + str(x))
    plt.title(title, fontsize=15, fontweight='bold'), plt.xlabel('Latent Dimension 1'), plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.show()
    # plot latent points in order ------------------------------------->
    df = df.sort_index(axis=0)
    plt.figure(figsize=(6, 6))
    # plot with averaging
    for x in range(num_classes):
        indices_for_class_subset = df.index[df['orientations'] == x].to_list()
        indices_for_class_subset.sort()
        first_latent = df['latent1'].loc[indices_for_class_subset].to_list()
        second_latent = df['latent2'].loc[indices_for_class_subset].to_list()
        i = 0
        while i < len(first_latent):
            first_point = np.mean(first_latent[i:i+average])
            second_point = np.mean(second_latent[i:i+average])
            if i == 0:
                plt.scatter(first_point, second_point, color=color[x], s=20, label='Grating Class = ' + str(x))
            else:
                plt.scatter(first_point, second_point, color=color[x], s=20)
            #if i == 0:
                #pass
            #else:
                #plt.plot([previous1, first_point], [previous2, second_point], '--', color=color[x], linewidth=1)
            i += average
            #previous1 = first_point
            #previous2 = second_point
    plt.title(title, fontsize=15, fontweight='bold'), plt.xlabel('Latent Dimension 1'), plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.show()


def plot_latent_3d(z_latent_test, test_labels, title, num_classes, average):
    color = ['blue', 'red', 'orange', 'brown', 'darkgray', 'black', 'yellow', 'green']
    # plot 3d plot of learned latent space ------------------------------->
    # remove the points in the 8th class or not in any of the orientation gratings
    dictionary = {'latent1': z_latent_test[:, 0], 'latent2': z_latent_test[:, 1], 'latent3': z_latent_test[:, 2], 'orientations': test_labels}
    df = pd.DataFrame(dictionary)
    df = df[df['orientations'] != 8]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for x in range(num_classes):
        indices_for_class_subset = df.index[df['orientations'] == x].to_list()
        indices_for_class_subset.sort()
        first_latent = df['latent1'].loc[indices_for_class_subset].to_list()
        second_latent = df['latent2'].loc[indices_for_class_subset].to_list()
        third_latent = df['latent3'].loc[indices_for_class_subset].to_list()
        ax.scatter(first_latent, second_latent, third_latent, alpha=.85, s=20, color=color[x])
    plt.title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Latent Dimension 1'), ax.set_ylabel('Latent Dimension 2'), ax.set_zlabel('Latent Dimension 3')
    plt.legend()
    plt.show()
    # not plot averages
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for x in range(num_classes):
        indices_for_class_subset = df.index[df['orientations'] == x].to_list()
        indices_for_class_subset.sort()
        first_latent = df['latent1'].loc[indices_for_class_subset].to_list()
        second_latent = df['latent2'].loc[indices_for_class_subset].to_list()
        third_latent = df['latent3'].loc[indices_for_class_subset].to_list()
        i = 0
        while i < len(first_latent):
            first_point = np.mean(first_latent[i:i+average])
            second_point = np.mean(second_latent[i:i+average])
            third_point = np.mean(third_latent[i:i+average])
            if i == 0:
                ax.scatter(first_point, second_point, third_point, color=color[x], s=20, label='Grating Class = ' + str(x))
            else:
                ax.scatter(first_point, second_point, third_point, color=color[x], s=20)
            #if i == 0:
                #pass
            #else:
                #plt.plot([previous1, first_point], [previous2, second_point], '--', color=color[x], linewidth=1)
            i += average
            #previous1 = first_point
            #previous2 = second_point
    plt.title(title, fontsize=15, fontweight='bold'), plt.xlabel('Latent Dimension 1'), plt.ylabel('Latent Dimension 2')
    ax.set_xlabel('Latent Dimension 1'), ax.set_ylabel('Latent Dimension 2'), ax.set_zlabel('Latent Dimension 3')
    plt.legend()
    plt.show()
