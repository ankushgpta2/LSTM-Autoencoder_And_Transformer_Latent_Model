import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras
import tensorflow.keras.utils
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential, Model, regularizers
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU, Flatten, GaussianDropout, Lambda, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import wandb
from tensorflow.keras.losses import SparseCategoricalCrossentropy, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------------------- VAE -----------------------------------------------------------------------------------------------------


def create_vanilla_VAE(input_data, vae_parameters, reg, latent_dim):
    # get relevant inputs
    batch_or_no = vae_parameters['vae_batch_norm']

    original_dim = input_data.shape[1]
    intermediate_dim = original_dim / 2
    intermediate_dim2 = original_dim / 4
    intermediate_dim3 = original_dim / 8
    intermediate_dim4 = original_dim / 10
    original_dim = input_data.shape[1]

    # define encoder part ---------------------------------------------------->
    # input layer
    x = Input(batch_shape=(None, original_dim), name='encoder_input')
    # first encoder hidden layer
    first_encoder_layer_output = Dense(intermediate_dim, bias_initializer='zeros', kernel_initializer='glorot_uniform', kernel_regularizer=l2(reg))(x)
    if batch_or_no == 'True':
        first_batch = BatchNormalization()(first_encoder_layer_output)
        leaky_1 = LeakyReLU(alpha=0.3)(first_batch)
    else:
        leaky_1 = LeakyReLU(alpha=0.3)(first_encoder_layer_output)
    # second encoder hidden layer
    second_encoder_layer_output = Dense(intermediate_dim2, bias_initializer='zeros', kernel_initializer='glorot_uniform', kernel_regularizer=l2(reg))(leaky_1)
    if batch_or_no == 'True':
        second_batch = BatchNormalization()(second_encoder_layer_output)
        leaky_2 = LeakyReLU(alpha=0.3)(second_batch)
    else:
        leaky_2 = LeakyReLU(alpha=0.3)(second_encoder_layer_output)
    # third encoder hidden layer
    third_encoder_layer_output = Dense(intermediate_dim3, bias_initializer='zeros', kernel_initializer='glorot_uniform', kernel_regularizer=l2(reg))(leaky_2)
    if batch_or_no == 'True':
        third_batch = BatchNormalization()(third_encoder_layer_output)
        leaky_3 = LeakyReLU(alpha=0.3)(third_batch)
    else:
        leaky_3 = LeakyReLU(alpha=0.3)(third_encoder_layer_output)
    # fourth encoder hidden layer
    fourth_encoder_layer_output = Dense(intermediate_dim4, bias_initializer='zeros', kernel_initializer='glorot_uniform', kernel_regularizer=l2(reg))(leaky_3)
    if batch_or_no == 'True':
        fourth_batch = BatchNormalization()(fourth_encoder_layer_output)
        leaky_4 = LeakyReLU(alpha=0.3)(fourth_batch)
    else:
        leaky_4 = LeakyReLU(alpha=0.3)(fourth_encoder_layer_output)

    # layers for the mean and variance sampling ---------------------------------------------------->
    z_mean = Dense(latent_dim)(leaky_4)
    z_log_var = Dense(latent_dim)(leaky_4)
    # lambda layer with the latent space sampling function
    z = Lambda(sample_from_latent, name='z')([z_mean, z_log_var])

    # define decoder part ---------------------------------------------------->
    # first decoder hidden layer
    first_decoder_layer_output = Dense(intermediate_dim4, bias_initializer='zeros', kernel_initializer='glorot_uniform', kernel_regularizer=l2(reg))(z)
    if batch_or_no == 'True':
        fifth_batch = BatchNormalization()(first_decoder_layer_output)
        leaky_5 = LeakyReLU(alpha=0.3)(fifth_batch)
    else:
        leaky_5 = LeakyReLU(alpha=0.3)(first_decoder_layer_output)
    # second decoder hidden layer
    second_decoder_layer_output = Dense(intermediate_dim3, bias_initializer='zeros', kernel_initializer='glorot_uniform', kernel_regularizer=l2(reg))(leaky_5)
    if batch_or_no == 'True':
        sixth_batch = BatchNormalization()(second_decoder_layer_output)
        leaky_6 = LeakyReLU(alpha=0.3)(sixth_batch)
    else:
        leaky_6 = LeakyReLU(alpha=0.3)(second_decoder_layer_output)
    # third decoder hidden layer
    third_decoder_layer_output = Dense(intermediate_dim2, bias_initializer='zeros', kernel_initializer='glorot_uniform', kernel_regularizer=l2(reg))(leaky_6)
    if batch_or_no == 'True':
        seventh_batch = BatchNormalization()(third_decoder_layer_output)
        leaky_7 = LeakyReLU(alpha=0.3)(seventh_batch)
    else:
        leaky_7 = LeakyReLU(alpha=0.3)(third_decoder_layer_output)
    # fourth decoder hidden layer
    fourth_decoder_layer_output = Dense(intermediate_dim, bias_initializer='zeros', kernel_initializer='glorot_uniform', kernel_regularizer=l2(reg))(leaky_7)
    if batch_or_no == 'True':
        eighth_batch = BatchNormalization()(fourth_decoder_layer_output)
        leaky_8 = LeakyReLU(alpha=0.3)(eighth_batch)
    else:
        leaky_8 = LeakyReLU(alpha=0.3)(fourth_decoder_layer_output)
    # specify output layer
    decoder_mean = Dense(original_dim, activation='softmax')(leaky_8)

    # initialize this structure built above ---------------------------------------------------->
    encoder = Model(x, z)
    vae = Model(x, decoder_mean)
    return vae, encoder


def run_vae(vae, encoder, general_parameters, vae_parameters, test_data, test_labels, input_data, input_labels, validation_data, validation_labels, batch_size, keep_vae_per_latent_dim,
            learning_rate, reg, latent_dim, epochs):

    if vae_parameters['vae_optimizer'] == 'SGD':
        vae_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif vae_parameters['vae_optimizer'] == 'Adam':
        vae_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # put everything together
    if keep_vae_per_latent_dim is False:
        vae, encoder = create_vanilla_VAE(input_data, vae_parameters, reg, latent_dim)

    vae.compile(optimizer=vae_optimizer, loss=vae_parameters['vae_loss'], metrics=general_parameters['metrics'])
    vae.summary()

    if general_parameters['validate_flag'] == 'True':
        history_for_vae = vae.fit(input_data, input_labels, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(validation_data, validation_labels),
                                  callbacks=[EarlyStopping(monitor='val_accuracy', patience=50)], shuffle=False)
    elif general_parameters['validate_flag'] == 'False':
        history_for_vae = vae.fit(input_data, input_labels, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False)

    # metrics such as prediction loss and accuracies
    results = vae.evaluate(test_data, test_labels, batch_size=batch_size)
    # actual latent predictions
    latent_pred = encoder.predict(test_data, batch_size=batch_size)
    # actual label predictions
    activity_pred = vae.predict(test_data, batch_size=batch_size)

    return vae, history_for_vae, results


def sample_from_latent(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    Input --> args (tensor): mean and log of variance of Q(z|X)
    Output --> z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

"""
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = (-1)*0.5 * K.sum(1 + z_log_var — K.square(z_mean) — K.exp(z_log_var), axis = -1)
    return xent_loss + kl_loss
"""
