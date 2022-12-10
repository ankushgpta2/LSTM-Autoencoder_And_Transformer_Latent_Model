from dnn_structures import *
from dnn_modules_help_functions import *
import wandb
wandb.login()
run = wandb.init(project='deep_learning_plenz',
                 config={'learning_rate': 0.0005, 'epochs': 1000, 'reg': 0.001})
config = wandb.config
from wandb.keras import WandbCallback
# tf.keras.backend.clear_session()
from sklearn.model_selection import train_test_split
import random


def run_model(mlp_parameters, autoencoder_parameters, artificial_parameters, general_parameters, data_for_different_recording_periods, experiment_name):
    """
    CYCLES THROUGH LATENT DIMENSIONS AS HIGHEST LEVEL LOOP + THEN CYCLES THROUGH DIFFERENT DATASET PERIODS [FULL, ON, HUNDRED, FIVE HUNDRED]
    """
    for v in range(len(autoencoder_parameters['autoencoder_latent_dim'])):
        counter = 0
        input_dict = {'general_parameters': general_parameters, 'autoencoder_parameters': autoencoder_parameters, 'mlp_parameters': mlp_parameters, 'experiment_name': experiment_name, 'latent_dim': autoencoder_parameters['autoencoder_latent_dim'][v]}
        if experiment_name != 'Artificial Data':
            for x in general_parameters['dataset']:
                input_data = data_for_different_recording_periods[x][0]
                run_custom_loss_model(input_data, input_labels=data_for_different_recording_periods[x][1], dataset=x, epochs=autoencoder_parameters['autoencoder_epochs'][counter], batch_size=autoencoder_parameters['autoencoder_batch_size'], input_dict=input_dict)
        elif experiment_name == 'Artificial Data':
            input_data = data_for_different_recording_periods[data_for_different_recording_periods.columns[:-1]].to_numpy()
            run_custom_loss_model(input_data=input_data, input_labels=data_for_different_recording_periods[data_for_different_recording_periods.columns[-1]].to_numpy(), dataset='Full', epochs=autoencoder_parameters['autoencoder_epochs'][counter],
                                  batch_size=autoencoder_parameters['autoencoder_batch_size'], input_dict=input_dict)
        counter += 1


def run_custom_loss_model(input_data, input_labels, dataset, epochs, batch_size, input_dict):
    general_parameters, autoencoder_parameters, mlp_parameters, experiment_name, latent_dim, ax, axes = get_info(input_dict)
    for num in range(8):
        models, num_classes = get_model_information(latent_dim, general_parameters, autoencoder_parameters, mlp_parameters, input_data=input_data, x=dataset)
        previous, i = handle_iteration_counter(i=0, start=0, dataset=dataset, autoencoder_parameters=autoencoder_parameters,  epochs=epochs, loss_holder=0, initialize_flag='global_counters', in_epoch=0, frame_counter=0, label_counter=0)
        class_input_data, class_input_labels = get_individual_class(input_data, input_labels, class_num=num)
        if autoencoder_parameters['run_LSTM'] is True:
            class_input_data, class_input_labels = reformat_data_for_transformer(class_input_data, class_input_labels, autoencoder_parameters)
        # CYCLE THROUGH EPOCHS
        while i < epochs:
            loss_holder, acc_holder, class_acc_holder, class_loss_holder, time_holder, recon_latent_holder = get_structs(latent_dim, global_time_holder_flag=False)
            start, frame_counter, label_counter = handle_iteration_counter(i, start=0, dataset=dataset, autoencoder_parameters=autoencoder_parameters, epochs=epochs, loss_holder=loss_holder, initialize_flag='local_counters', in_epoch=0, frame_counter=0, label_counter=0)
            # CYCLE THROUGH BATCHES
            while frame_counter <= np.shape(class_input_data)[0]-1:  # number of times neural network is ran through with mini batch
                if autoencoder_parameters['run_LSTM'] is True:
                    mini_batch = class_input_data[frame_counter:frame_counter + batch_size, :, :]   # essentially the number of sequences while keeping the number of neurons the same
                    # labels = class_input_labels[label_counter:label_counter + batch_size*autoencoder_parameters['lstm_sequence_length']]
                else:
                    mini_batch = class_input_data[frame_counter:frame_counter + batch_size, :]
                    # labels = class_input_labels[frame_counter:frame_counter + batch_size]
                if 'autoencoder' in models.keys():
                        class_acc, class_loss, loss, acc, encoded = models['autoencoder'].network_learn(mlp_model=models['mlp'], num_classes=num_classes, input_data=mini_batch, true_values=mini_batch,
                                                                             autoencoder_parameters=autoencoder_parameters)

                loss_holder, acc_holder, class_acc_holder, class_loss_holder, recon_latent_holder = dnn_output_before_epoch(i, latent_dim, loss, acc, class_acc, class_loss, encoded, loss_holder,
                                                                                                                            acc_holder, class_acc_holder, class_loss_holder, recon_latent_holder,
                                                                                                                            autoencoder_parameters, frame_counter, epochs, class_input_data, label_counter)
                frame_counter, label_counter = handle_iteration_counter(i, start, dataset, autoencoder_parameters, epochs, loss_holder, initialize_flag=0, in_epoch=True, frame_counter=frame_counter, label_counter=label_counter)
            # print and plot necessary information
            i = handle_iteration_counter(i, start, dataset, autoencoder_parameters=autoencoder_parameters, epochs=epochs, loss_holder=loss_holder, initialize_flag=0, in_epoch=False, frame_counter=frame_counter, label_counter=label_counter)
            previous, axes = dnn_output_after_epoch(i, autoencoder_parameters, general_parameters, mlp_parameters, dataset, acc_holder, class_acc_holder, loss_holder, class_loss_holder, axes, previous, latent_dim, experiment_name, add_final_stuff=False)

        # now plot everything
        dnn_output_after_epoch(i, autoencoder_parameters, general_parameters, mlp_parameters, dataset, acc_holder, class_acc_holder, loss_holder, class_loss_holder, axes, previous, latent_dim, experiment_name,
                               add_final_stuff=False)
        # plt.show()
        plot_latent(recon_latent_holder, class_input_labels, autoencoder_parameters, general_parameters, mlp_parameters, dataset, latent_dim, experiment_name, loss_type=autoencoder_parameters['autoencoder_type_of_loss'],
                epochs=epochs, vae_reg_flag=autoencoder_parameters['vae_latent_reg_flag'], batch_layers_flag=autoencoder_parameters['autoencoder_batch_layers_flag'], ax=ax)
    plt.show()

