from artificial_data import *
from handle_dnn_inputs import *
from tiago_data import *
from dnn_modules import *

# get hyperparameters
general_parameters, mlp_parameters, autoencoder_parameters, artificial_parameters = get_hyperparams()

# ----------------------------------------------- ARTIFICIAL DATA THROUGH VAE ------------------------------------------------
# whether or not to run on artificial data
if general_parameters['run_artificial'] is True:
    artificial_main(artificial_parameters)

# ------------------------------------------------------- TIAGO DATA THROUGH VAE ---------------------------------------------------------
# whether or not to run experimental data
if general_parameters['run_tiago_data'] is True:
    analyze_tiago_data_main(general_parameters, mlp_parameters, autoencoder_parameters, artificial_parameters)

