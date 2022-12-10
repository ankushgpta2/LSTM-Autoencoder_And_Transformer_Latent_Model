from artificial_data import *
import matplotlib.pyplot as plt
from handle_inputs import *
from plot_dnn_outputs import *
from vae_structure import *


def prep_data_for_dnn(list_of_cells, artificial_parameters, list_of_gratings, frames_per_orientation, on_set, frames_per_trial, random_gratings):
    data_split = artificial_parameters['data_split']
    num_time_bins = artificial_parameters['num_time_bins']
    different_classes = artificial_parameters['different_classes']

    # convert the data array into DF with the stimulus information across frames as last column
    artificial_df = pd.DataFrame(list_of_cells, columns=['ROI_' + str(x) for x in range(np.shape(list_of_cells)[1])])
    orientation_classes_column = np.array([8 for x in range(num_time_bins)])
    for x in list_of_gratings:
        orientation_classes_column[frames_per_orientation[x]] = x
    artificial_df['orientation_classes'] = orientation_classes_column

    # now split the data into training, validation, and test sets --------------------------------------->
    original_num_of_rows = np.shape(artificial_df.to_numpy())[0]
    activity_for_train_rows = round(data_split[0 ] *original_num_of_rows)
    indices_for_valid = [x for x in range(activity_for_train_rows, activity_for_train_rows + (round((original_num_of_rows - activity_for_train_rows) / 2)))]
    indices_for_test = [x for x in range(len(indices_for_valid) + activity_for_train_rows, original_num_of_rows)]
    # get training information
    train_split = artificial_df.loc[[x for x in range(activity_for_train_rows)]]
    input_data = train_split[train_split.columns[:-1]]
    input_labels = train_split[train_split.columns[-1]]
    # get validation and test information
    valid_split = artificial_df.loc[indices_for_valid]
    validation_data = valid_split[valid_split.columns[:-1]]
    validation_labels = valid_split[valid_split.columns[-1]]
    test_split = artificial_df.loc[indices_for_test]
    test_data = test_split[test_split.columns[:-1]]
    test_labels = test_split[test_split.columns[-1]]

    # now split the data based on each trial... dictionary containing the trial number as key, orientation class as first list in key, and frames as second list in key ----------------->
    counter = 0
    data_per_trial = {}
    for x in range(len(random_gratings ) -1):
        frames_for_trial = [x for x in range(counter, counter + frames_per_trial)]
        data_per_trial[x] = [[random_gratings[x]], artificial_df.loc[frames_for_trial]]
        counter += frames_per_trial

    # get hyperparams and other parameters for running the neural nets
    general_parameters, vae_parameters, artificial_info = get_hyperparams()
    general_parameters['validate_flag'] = 'True'
    general_parameters['num_classes'] = different_classes + 1
    return general_parameters, vae_parameters, validation_data, validation_labels, test_labels, test_data, input_labels, input_data, artificial_df, data_per_trial


def run_actual_neural_nets(general_parameters, vae_parameters, artificial_parameters, validation_data, validation_labels, test_labels, test_data, input_labels, input_data,
                           data_per_trial, random_gratings):
    history_holder = {}
    results_holder = {}
    name = []
    if general_parameters['run_vae'] == 'True':
        # get the other necessary hyperparameters
        reg = vae_parameters['vae_reg'][0]
        learning_rate = vae_parameters['vae_learning_rate'][0]
        batch_size = vae_parameters['vae_batch_size'][0]
        epochs = vae_parameters['vae_epochs'][0]
        # get the name for VAE
        name_of_vae = 'VAE [4,4,2,' + 'lr=' + str(learning_rate) + ',reg=' + str(reg) + ',batch=' + str(batch_size) + ',epochs=' + str(epochs) + ',cells=' + \
                      str(artificial_parameters['num_cells']) + ',bins=' + str(artificial_parameters['num_time_bins']) + ',classes=' + str(artificial_parameters['different_classes']) + ']'
        name.append(name_of_vae)
        # NOW ACTUALLY RUNNING THE NEURAL NETS
        for v in range(len(vae_parameters['vae_latent_dim'])):
            # FOR DIFFERENT LATENT DIMENSIONS --> RUNNING VAE ON FULL DATASET AND PLOTTING LATENT + FEEDING TO MLP FOR ACC AND PREDICTION  ---------------------------------->
            latent_dim = vae_parameters['vae_latent_dim'][v]
            vae, encoder = create_vanilla_VAE(input_data, vae_parameters, reg, latent_dim)
            # now run the VAE on data to get encoded representation and feed this encoder
            vae, history_for_vae, results = run_vae(vae, encoder, general_parameters, vae_parameters, test_data, test_labels, input_data, input_labels, validation_data, validation_labels,
                                                batch_size=batch_size, keep_vae_per_latent_dim=False, learning_rate=learning_rate, reg=reg, latent_dim=latent_dim, epochs=epochs)


            results_holder['VAE: ' + str(v)] = results[0], results[1]
            history_holder['VAE: ' + str(v)] = list(history_for_vae.history.items())
            if vae_parameters['vae_latent_dim'][v] == 2:
                z_test = vae.predict(test_data, batch_size=vae_parameters['vae_batch_size'][0])
                plot_latent_2d(z_latent_test=z_test, test_labels=test_labels, average=20, title='2D ' + name_of_vae + ' Latent Space For Artificial Data',
                        num_classes=artificial_parameters['different_classes'])
            elif vae_parameters['vae_latent_dim'][v] == 3:
                z_test = vae.predict(test_data, batch_size=vae_parameters['vae_batch_size'][0])
                plot_latent_3d(z_latent_test=z_test, test_labels=test_labels, title='3D ' + name_of_vae + ' Latent Space For Artificial Data',
                        num_classes=artificial_parameters['different_classes'], average=20)
            
  
