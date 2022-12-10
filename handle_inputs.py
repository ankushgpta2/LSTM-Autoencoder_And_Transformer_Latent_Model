import wandb
import argparse
import tensorflow as tf

def get_hyperparams():
    args = get_args().parse_args()
    # general parameters for data
    data_directory = args.data_directory
    validate_flag = args.validate
    splits = args.splits
    scaling = args.scaling
    dataset = args.dataset
    exp_num = args.specific_exp_num
    run_autoencoder = args.run_autoencoder
    run_single_trials = args.run_single_trials
    run_artificial = args.run_artificial
    run_tiago_data = args.run_tiago_data
    run_PCA = args.run_PCA
    threshold_percentile = args.threshold_percentile
    threshold_for_minimum_activity = args.threshold_for_minimum_activity
    drop_cells = args.drop_cells
    general_parameters = {'data_directory': data_directory, 'validate_flag': validate_flag, 'splits': splits, 'scaling': scaling, 'dataset': dataset, 'exp_num': exp_num,
                          'run_autoencoder': run_autoencoder, 'run_single_trials': run_single_trials,
                          'run_artificial': run_artificial, 'threshold_percentile': threshold_percentile, 'threshold_for_minimum_activity': threshold_for_minimum_activity,
                          'drop_cells': drop_cells, 'run_tiago_data': run_tiago_data, 'run_PCA': run_PCA}

    # for MLP ----------------------------------------------------->
    mlp_epochs = args.mlp_epochs
    mlp_batch_size = args.mlp_batch_size
    mlp_learning_rate = args.mlp_learning_rate
    mlp_reg = args.mlp_reg
    mlp_optimizer = args.mlp_optimizer
    mlp_batch_norm = args.mlp_batch_norm
    if mlp_optimizer == 'SGD':
        mlp_optimizer = tf.keras.optimizers.SGD(learning_rate=mlp_learning_rate, momentum=0.9)
    elif mlp_optimizer == 'Adam':
        mlp_optimizer = tf.keras.optimizers.Adam(learning_rate=mlp_learning_rate)
    elif mlp_optimizer == 'Adagrad':
        mlp_optimizer = tf.keras.optimizers.Adagrad(learning_rate=mlp_learning_rate)
    mlp_parameters = {'mlp_epochs': mlp_epochs, 'mlp_batch_size': mlp_batch_size, 'mlp_reg': mlp_reg, 'mlp_learning_rate': mlp_learning_rate, 'mlp_optimizer': mlp_optimizer,
                      'mlp_batch_norm': mlp_batch_norm}

    # for VAE --------------------------------------------------------->
    autoencoder_latent_dim = args.autoencoder_latent_dim
    autoencoder_batch_size = args.autoencoder_batch_size
    autoencoder_reg = args.autoencoder_reg
    autoencoder_learning_rate = args.autoencoder_learning_rate
    autoencoder_epochs = args.autoencoder_epochs
    autoencoder_batch_layers_flag = args.autoencoder_batch_layers_flag
    autoencoder_run_embedding = args.autoencoder_run_embedding
    single_trials_batch_size = args.single_trials_batch_size
    single_trials_reg = args.single_trials_reg
    single_trials_learning_rate = args.single_trials_learning_rate
    single_trials_epochs = args.single_trials_epochs
    autoencoder_optimizer = args.autoencoder_optimizer
    vae_latent_reg_flag = args.vae_latent_reg_flag
    autoencoder_type_of_loss = args.autoencoder_type_of_loss
    lstm_sequence_length = args.lstm_sequence_length
    run_LSTM = args.run_LSTM
    run_mlp = args.run_mlp
    autoencoder_parameters = {'autoencoder_optimizer': autoencoder_optimizer, 'autoencoder_latent_dim': autoencoder_latent_dim,
                      'autoencoder_batch_size': autoencoder_batch_size, 'autoencoder_reg': autoencoder_reg, 'autoencoder_learning_rate': autoencoder_learning_rate,
                      'autoencoder_epochs': autoencoder_epochs, 'single_trials_batch_size': single_trials_batch_size, 'single_trials_reg': single_trials_reg,
                      'single_trials_learning_rate': single_trials_learning_rate, 'single_trials_epochs': single_trials_epochs, 'vae_latent_reg_flag': vae_latent_reg_flag,
                              'autoencoder_type_of_loss': autoencoder_type_of_loss, 'autoencoder_batch_layers_flag': autoencoder_batch_layers_flag, 'autoencoder_run_embedding': autoencoder_run_embedding,
                              'lstm_sequence_length': lstm_sequence_length, 'run_LSTM': run_LSTM, 'run_mlp': run_mlp}

    # for artificial simulation
    num_cells = args.num_cells
    num_time_bins = args.num_time_bins
    stim_duration = args.stim_duration
    gap_until_next = args.gap_until_next
    different_classes = args.different_classes
    tuned_cell_range = args.tuned_cell_range
    delay_after_stim = args.delay_after_stim
    imaging_rate = args.imaging_rate
    synaptic_delay = args.synaptic_delay
    data_split = args.data_split
    single_gen_branching = args.single_gen_branching
    random_branching = args.random_branching
    num_of_initial_cell_activations = args.num_of_initial_cell_activations
    branching_after_how_many_gen = args.branching_after_how_many_gen
    ensemble_level_temporal_profile = args.ensemble_level_temporal_profile
    spike_rate_of_neurons = args.spike_rate_of_neurons
    vary_spike_rate = args.vary_spike_rate
    random_event_prob = args.random_event_prob
    spatial_spread_of_random_events = args.spatial_spread_of_random_events
    add_random_events = args.add_random_events
    artificial_parameters = {'num_cells': num_cells, 'num_time_bins': num_time_bins, 'stim_duration': stim_duration, 'gap_until_next': gap_until_next, 'different_classes': different_classes,
                         'tuned_cell_range': tuned_cell_range, 'delay_after_stim': delay_after_stim, 'imaging_rate': imaging_rate, 'synaptic_delay': synaptic_delay, 'data_split': data_split,
                         'single_gen_branching': single_gen_branching, 'random_branching': random_branching, 'num_of_initial_cell_activations': num_of_initial_cell_activations,
                         'branching_after_how_many_gen': branching_after_how_many_gen, 'ensemble_level_temporal_profile': ensemble_level_temporal_profile, 'spike_rate_of_neurons': spike_rate_of_neurons,
                         'vary_spike_rate': vary_spike_rate, 'random_event_prob': random_event_prob, 'spatial_spread_of_random_events': spatial_spread_of_random_events,
                             'add_random_events': add_random_events}

    return general_parameters, mlp_parameters, autoencoder_parameters, artificial_parameters


def get_args():
    parser = argparse.ArgumentParser(description="Parameters For Neural Nets")

    # general parameters for data
    parser.add_argument('--data_directory', type=list, default=[r"/Users/ankushgupta/Documents/tiago_data/3143_1", r'/Users/ankushgupta/Documents/tiago_data/3143_2',
                                                                r'/Users/ankushgupta/Documents/tiago_data/6742'], help='directories for data')
    parser.add_argument('--validate', type=bool, default=False, help='whether to split into validate or not')
    parser.add_argument('--splits', nargs='+', default=[0.9, 0.1], help='actual splitting data')
    parser.add_argument('--scaling', type=str, default='MaxAbsScaler', help='how to normalize data')
    parser.add_argument('--dataset', nargs='+', default=['Full'], help='which portion of recording to run neural net on')
    parser.add_argument('--specific_exp_num', type=int, default=0, help='# for experiment within array')
    parser.add_argument('--threshold_percentile', type=float, default=0, help='will make the bottom percentage specified = to 0... considers it noise')
    parser.add_argument('--threshold_for_minimum_activity', type=float, default=0.01, help='will take average activity throughout entire recording')
    parser.add_argument('--drop_cells', type=bool, default=False, help='whether or not to drop least active cells based on threshold above')

    # which structures to run
    parser.add_argument('--run_autoencoder', type=bool, default=True, help='Whether or not to run VAE')
    parser.add_argument('--run_single_trials', type=bool, default=False, help='whether or not to run set of functions for single trial passes on full sim data')
    parser.add_argument('--run_artificial', type=bool, default=False, help='whether or not to rnu the artificial data')
    parser.add_argument('--run_tiago_data', type=bool, default=True, help='whether or not to run experimental data')
    parser.add_argument('--run_PCA', type=bool, default=False, help='whether or not to run experimental data')

    # for MLP
    parser.add_argument('--mlp_params', nargs='+', default=['learning_rate'], help='whether or not to do hyperparam tuning for VAE')
    parser.add_argument('--mlp_batch_size', nargs='+', default=1000, help='batch size')
    parser.add_argument('--mlp_reg', nargs='+', default=0.001, help='regularization lambda')
    parser.add_argument('--mlp_learning_rate', type=float, default=0.0008, help='learn rate')
    parser.add_argument('--mlp_epochs', nargs='+', default=500, help='number of epochs to train')
    parser.add_argument('--mlp_optimizer', type=str, default='Adam', help='optimizer for loss function')
    parser.add_argument('--mlp_batch_norm', type=str, default='False', help='whether or not to add batchnorm layers')

    # for Autoencoder
    parser.add_argument('--autoencoder_latent_dim', type=int, default=[2, 3, 10, 20, 50, 100, 200], help='whether or not to plot latent space')
    parser.add_argument('--autoencoder_optimizer', type=str, default='Adam', help='optimizer for loss function')
    parser.add_argument('--autoencoder_batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--autoencoder_learning_rate', type=int, default=0.0005, help='learning rate')
    parser.add_argument('--autoencoder_reg', type=int, default=0.001, help='traditional activity regularization applied to each layer')
    parser.add_argument('--autoencoder_epochs', nargs='+', default=[200, 5000, 100], help='epochs --> if more than one, then the different epochs are for the different recording periods, if float, then early stop at that classification accuracy')
    parser.add_argument('--autoencoder_type_of_loss', type=str, default='only_reconstruction', help='type of loss for autoencoder optimization')
    parser.add_argument('--autoencoder_batch_layers_flag', type=bool, default=False, help='whether or not to have batch normalization layers')
    parser.add_argument('--vae_latent_reg_flag', type=bool, default=False, help='whether or not to do latent regularization for variational autoencoder, otherwise normal autoencoder ran')
    parser.add_argument('--autoencoder_run_embedding', type=bool, default=False, help='whether or not to do embedding for the class labels')
    parser.add_argument('--run_mlp', type=bool, default=False, help='whether or not to run MLP for classification')
    parser.add_argument('--run_LSTM', type=bool, default=True, help='Whether or not to run LSTM')
    parser.add_argument('--lstm_sequence_length', type=int, default=15, help='the length of the sequence for LSTM')
    # for single trials module --> parameters for doing the single trials trajectories
    parser.add_argument('--single_trials_batch_size', type=int, default=2, help='batch size for single trial passes')
    parser.add_argument('--single_trials_learning_rate', type=int, default=0.00005, help='learning rate for the single trial passes')
    parser.add_argument('--single_trials_reg', type=int, default=0.01, help='reg strength for single trial passes')
    parser.add_argument('--single_trials_epochs', type=int, default=20, help='epochs for single trial passes')

    # for simulation
    parser.add_argument('--num_cells', type=int, default=200, help='number of cells for simulated network')
    parser.add_argument('--num_time_bins', type=int, default=70000, help='number of separate timebins in recording')
    parser.add_argument('--stim_duration', type=int, default=1, help='duration in seconds for stimulus presentation')
    parser.add_argument('--gap_until_next', type=int, default=7, help='duration in seconds between stimulus presentation and next trial')
    parser.add_argument('--different_classes', type=int, default=8, help='number of different stimulus classes')
    parser.add_argument('--tuned_cell_range', nargs='+', default=[0.05, 0.3], help='range for proportion of total cells in network that will be tuned towards each stimulus class')
    parser.add_argument('--delay_after_stim', type=float, default=0.1, help='delay between first network response and stimulus presentation')
    parser.add_argument('--imaging_rate', type=float, default=45.0, help='imaging rate or mapping between frames/time bins and seconds or real time')
    parser.add_argument('--synaptic_delay', type=float, default=0.1, help='delay for information transfer between nodes in network')
    parser.add_argument('--data_split', nargs='+', default=[0.6, 0.4], help='how to split the simulated dataset into train / validation / test splits')
    parser.add_argument('--single_gen_branching', type=bool, default=True, help='whether initially activated cells branch or coactivate two neurons to create two branches that are feedforward')
    parser.add_argument('--random_branching', type=bool, default=False, help='whether or not there is random branching in the network')
    parser.add_argument('--num_of_initial_cell_activations', nargs='+', default=[0.05, 0.2], help='proportion of initial activations of determined tuned cells')
    parser.add_argument('--branching_after_how_many_gen', type=int, default=1, help='branching occurs at a fixed interval')
    parser.add_argument('--ensemble_level_temporal_profile', type=str, default='hold', help='specifies higher level temporal relationship')
    parser.add_argument('--spike_rate_of_neurons', type=int, default=10, help='number of action potentials / stim duration')
    parser.add_argument('--vary_spike_rate', type=str, default='hold', help='vary spike rate')
    parser.add_argument('--random_event_prob', type=float, default=0.01, help='specifies random event probability or lambda for poisson')
    parser.add_argument('--spatial_spread_of_random_events', nargs='+', default=[0.1, 0.5], help='spatial spread of random events or how many neurons are effected by the event')
    parser.add_argument('--add_random_events', type=bool, default=True, help='add poisson distributed random events or spikes ')
    return parser
