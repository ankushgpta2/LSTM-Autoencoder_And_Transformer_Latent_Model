from dnn_structures import *


# ------------------------------------------------------------ DNN MODULES SUPPORT FUNCTIONS ----------------------------------------------------------------------------------------------------------------
def get_info(input_dict):
    general_parameters = input_dict['general_parameters']
    autoencoder_parameters = input_dict['autoencoder_parameters']
    mlp_parameters = input_dict['mlp_parameters']
    experiment_name = input_dict['experiment_name']
    latent_dim = input_dict['latent_dim']
    if latent_dim == 3:
        ax = plt.axes(projection='3d')
    elif latent_dim == 2:
        ax = 0
    axes = 0
    return general_parameters, autoencoder_parameters, mlp_parameters, experiment_name, latent_dim, ax, axes


def dnn_output_before_epoch(i, latent_dim, loss, acc, class_acc, class_loss, encoded, loss_holder, acc_holder, class_acc_holder, class_loss_holder, recon_latent_holder, autoencoder_parameters,
                                frame_counter, epochs, input_data, label_counter):
    if autoencoder_parameters['run_LSTM'] is True:
        print(str(frame_counter / autoencoder_parameters['autoencoder_batch_size']) + '/' + str(round(input_data.shape[0] / autoencoder_parameters['autoencoder_batch_size'])), end="\r")
    else:
        print(str(frame_counter / autoencoder_parameters['autoencoder_batch_size']) + '/' + str(round(input_data.shape[0] / autoencoder_parameters['autoencoder_batch_size'])), end="\r")
    # place necessary information into structures for plotting + further analysis
    loss_holder.append(np.mean(loss[~ np.isnan(np.array(loss))]))
    acc_holder.append(acc)
    class_acc_holder.append(class_acc)
    class_loss_holder.append(class_loss)
    if i == epochs-1:
        if autoencoder_parameters['run_LSTM'] is True:
            recon_latent_holder[label_counter:label_counter + len(encoded[:, 0]), 0] = encoded[:, 0]
            recon_latent_holder[label_counter:label_counter + len(encoded[:, 1]), 1] = encoded[:, 1]
        else:
            recon_latent_holder[frame_counter:frame_counter + len(encoded[:, 0]), 0] = encoded[:, 0]
            recon_latent_holder[frame_counter:frame_counter + len(encoded[:, 1]), 1] = encoded[:, 1]
        if latent_dim == 3:
            if autoencoder_parameters['run_LSTM'] is True:
                recon_latent_holder[label_counter:label_counter + len(encoded[:, 2]), 2] = encoded[:, 2]
            else:
                recon_latent_holder[frame_counter:frame_counter + len(encoded[:, 2]), 2] = encoded[:, 2]
    return loss_holder, acc_holder, class_acc_holder, class_loss_holder, recon_latent_holder


def dnn_output_after_epoch(i, autoencoder_parameters, general_parameters, mlp_parameters, dataset, acc_holder, class_acc_holder, loss_holder, class_loss_holder, axes, previous, latent_dim, experiment_name, add_final_stuff):
    previous = plot_custom_loss_metrics(i, acc_holder, loss_holder, class_acc_holder, class_loss_holder, axes, previous, autoencoder_parameters, general_parameters, mlp_parameters, dataset,
                                        latent_dim, experiment_name,
                                        loss_type=autoencoder_parameters['autoencoder_type_of_loss'],
                                        add_final_stuff=add_final_stuff,
                                        vae_reg_flag=autoencoder_parameters['vae_latent_reg_flag'],
                                        batch_layers_flag=autoencoder_parameters['autoencoder_batch_layers_flag'],
                                        experiment_name=experiment_name,
                                        latent_dim=latent_dim)
    return previous


def handle_iteration_counter(i, start, dataset, autoencoder_parameters, epochs, loss_holder, initialize_flag, in_epoch, frame_counter, label_counter):
    # potentially intialize counters
    if initialize_flag == 'global_counters':
        previous = 0
        i = 0
        print('---------------------------- DATASET = ' + dataset + ' , ' + ' LOSS = ' + str(autoencoder_parameters['autoencoder_type_of_loss']) + ' ----------------------------------')
        return previous, i
    elif initialize_flag == 'local_counters':
        start = time.time()
        frame_counter = 0
        label_counter = 0
        return start, frame_counter, label_counter
    # keep track of counters within epoch
    if in_epoch is True:
        frame_counter += autoencoder_parameters['autoencoder_batch_size']
        label_counter += autoencoder_parameters['autoencoder_batch_size'] * autoencoder_parameters['lstm_sequence_length']
        return frame_counter, label_counter
    elif in_epoch is False:
        total_time = round(time.time() - start, 2)
        print(str(i + 1) + '/' + str(epochs) + ' --> Time For Epoch = ' + str(total_time) + ' Seconds' + ' --> Loss = ' + str(np.mean(loss_holder)))
        i += 1
        return i


def reformat_data_for_transformer(input_data, input_labels, autoencoder_parameters):
    sequence_length = autoencoder_parameters['lstm_sequence_length']
    rounded_data = input_data[:round(np.floor(input_data.shape[0] / sequence_length) * sequence_length), :]
    rounded_labels = input_labels[:round(np.floor(input_data.shape[0] / sequence_length) * sequence_length)]
    reformatted_data = rounded_data.reshape(round(rounded_data.shape[0]/sequence_length), sequence_length, rounded_data.shape[1])
    return reformatted_data, rounded_labels


def get_individual_class(input_data, input_labels, class_num):
    indices = np.argwhere(input_labels == class_num)
    input_data = input_data[indices]
    input_labels = input_labels[indices]
    input_data = input_data.reshape(input_data.shape[0], input_data.shape[-1])
    return input_data, input_labels


def get_structs(latent_dim, global_time_holder_flag):
    # get necessary structure information for storing outputs from the neural nets
    if global_time_holder_flag is True:
        global_time_holder = []
        return global_time_holder
    else:
        loss_holder = []
        acc_holder = []
        class_acc_holder = []
        class_loss_holder = []
        time_holder = []
        recon_latent_holder = np.empty((100000, latent_dim))
        recon_latent_holder[:, :] = np.NaN
        return loss_holder, acc_holder, class_acc_holder, class_loss_holder, time_holder, recon_latent_holder


def get_model_information(latent_dim, general_parameters, autoencoder_parameters, mlp_parameters, input_data, x):
    num_classes = 8
    models = {}
    if general_parameters['run_autoencoder'] is True:
        autoencoder_model = get_autoencoder_struct(autoencoder_parameters,
                                                    latent_dim=latent_dim,
                                                    reg=autoencoder_parameters['autoencoder_reg'],
                                                    learning_rate=autoencoder_parameters['autoencoder_learning_rate'],
                                                    input_data=input_data,
                                                    num_classes=num_classes)
        models['autoencoder'] = autoencoder_model
    if autoencoder_parameters['run_mlp'] is True:
        models['mlp'] = get_mlp_struct(mlp_parameters, autoencoder_parameters, num_classes)
    else:
        models['mlp'] = 0
    return models, num_classes


def get_autoencoder_struct(autoencoder_parameters, latent_dim, reg, learning_rate, input_data, num_classes):
    autoencoder_model = autoencoder(original_dim=np.shape(input_data)[1],
                                               autoencoder_optimizer=autoencoder_parameters['autoencoder_optimizer'],
                                               learning_rate=learning_rate, latent_dim=latent_dim,
                                               default_tf_reg=reg,
                                               num_classes=num_classes,
                                    autoencoder_parameters=autoencoder_parameters)
    return autoencoder_model


def get_mlp_struct(mlp_parameters, autoencoder_parameters, num_classes):
    return MLP(mlp_parameters, autoencoder_parameters, mlp_optimizer=mlp_parameters['mlp_optimizer'], mlp_reg=mlp_parameters['mlp_reg'], num_classes=num_classes)


def shuffle_data(input_data, input_labels):
    # whether or not to shuffle the dataset
    idx = np.random.permutation(input_data.shape[0])
    input_data = input_data[idx, :]
    input_labels = input_labels[idx]
    return input_data, input_labels

