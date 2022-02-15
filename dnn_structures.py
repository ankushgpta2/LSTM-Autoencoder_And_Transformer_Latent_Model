import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras import Model, Sequential
import tensorflow.keras.utils
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU, Flatten, GaussianDropout, Lambda, LSTM, TimeDistributed, RepeatVector, Embedding, LayerNormalization, Dropout, MultiHeadAttention, Conv1D
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
from plot_dnn_outputs import *
import time


# ---------------------------------------------------------------------------------------- VAE -----------------------------------------------------------------------------------------------------
# FOR OOP AND CUSTOM LOSS FUNCTION
class autoencoder():
    def __init__(self, original_dim, autoencoder_optimizer, learning_rate, latent_dim, default_tf_reg, num_classes, autoencoder_parameters):
        # for parameterizing latent
        self.var_embed = Embedding(num_classes, latent_dim, embeddings_initializer='glorot_uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=None)
        self.mean_embed = Embedding(num_classes, latent_dim, embeddings_initializer='glorot_uniform',  embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=None)
        self.posterior = Lambda(self.compute_post)
        # encoder
        self.input = Dense(original_dim, input_shape=(None, None), name='encoder_input')
        self.hidden_1 = Dense(round(original_dim * 0.9), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_2 = Dense(round(original_dim * 0.7), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_3 = Dense(round(original_dim * 0.5), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_4 = Dense(round(original_dim * 0.4), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_5 = Dense(round(original_dim * 0.3), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_6 = Dense(round(original_dim * 0.2), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_7 = Dense(10, bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        # decoder
        self.hidden_8 = Dense(10, bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_9 = Dense(round(original_dim * 0.2), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_10 = Dense(round(original_dim * 0.3), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_11 = Dense(round(original_dim * 0.4), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_12 = Dense(round(original_dim * 0.5), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_13 = Dense(round(original_dim * 0.7), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.hidden_14 = Dense(round(original_dim * 0.9), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l1(default_tf_reg))
        self.output_decoder = Dense(original_dim, activation='softmax')
        # lstm
        self.lstm_1 = LSTM(original_dim, activation='tanh', input_shape=(autoencoder_parameters['autoencoder_batch_size'], autoencoder_parameters['lstm_sequence_length'], original_dim), return_sequences=True)
        self.lstm_2 = LSTM(round(original_dim/2), activation='tanh', return_sequences=True)
        self.lstm_3 = LSTM(round(original_dim / 4), activation='tanh', return_sequences=True)
        self.lstm_4 = LSTM(round(original_dim / 8), activation='tanh', return_sequences=True)
        self.lstm_5 = LSTM(round(original_dim/16), activation='tanh', return_sequences=True)
        self.lstm_6 = LSTM(round(original_dim/16), activation='tanh', input_shape=(autoencoder_parameters['lstm_sequence_length'], latent_dim), return_sequences=True)
        self.lstm_7 = LSTM(round(original_dim / 8), activation='tanh', return_sequences=True)
        self.lstm_8 = LSTM(round(original_dim / 4), activation='tanh', return_sequences=True)
        self.lstm_9 = LSTM(round(original_dim/2), activation='tanh', return_sequences=True)
        self.lstm_10 = LSTM(original_dim, activation='tanh', return_sequences=True)
        self.lstm_bottleneck_1 = LSTM(latent_dim, return_sequences=False)
        self.lstm_bottleneck_2 = LSTM(latent_dim, return_sequences=False)
        self.time_dist_1 = TimeDistributed(Dense(original_dim))
        self.lstm_repeat_vec = RepeatVector(autoencoder_parameters['lstm_sequence_length'])
        self.dropout = Dropout(0.2)
        # latent
        self.z_mean = Dense(latent_dim)
        self.z_log_var = Dense(latent_dim)
        self.z = Lambda(self.sample_from_latent, name='z')
        # general
        self.batch1 = BatchNormalization()
        self.batch2 = BatchNormalization()
        self.batch3 = BatchNormalization()
        self.batch4 = BatchNormalization()
        self.batch5 = BatchNormalization()
        self.batch6 = BatchNormalization()
        self.batch7 = BatchNormalization()
        self.batch8 = BatchNormalization()
        self.batch9 = BatchNormalization()
        self.batch10 = BatchNormalization()
        self.batch11 = BatchNormalization()
        self.batch12 = BatchNormalization()
        self.batch13 = BatchNormalization()
        self.batch14 = BatchNormalization()
        self.leaky = LeakyReLU(alpha=0.3)
        # autoencoder_optimizer
        if autoencoder_optimizer == 'SGD':
            autoencoder_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif autoencoder_optimizer == 'Adam':
            autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif autoencoder_optimizer == 'Adagrad':
            autoencoder_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        self.ae_op = autoencoder_optimizer
        # transformer
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.multihead = MultiHeadAttention(key_dim=32, num_heads=8, dropout=0)

    def run_lstm_transformer_encoder(self, input_data):
        layers = [self.layernorm, self.multihead, self.dropout, self.layernorm]
        output = input_data
        for x in range(len(layers)):
            if layers[x] == self.dropout:
                output = layers[x](output)
                res = output + input_data
            elif layers[x] == self.multihead:
                output = layers[x](output, output)
            else:
                output = layers[x](output)
        latent_mean_vec, latent_var_vec = self.run_lstm_encoder(output)
        return latent_mean_vec, latent_var_vec

    def run_lstm_encoder(self, input_data):
        layers = [self.lstm_1, self.dropout, self.lstm_2, self.dropout, self.lstm_3, self.dropout, self.lstm_4, self.dropout, self.lstm_5, self.lstm_bottleneck_1, self.lstm_bottleneck_2]
        output = input_data
        for x in range(len(layers)):
            if layers[x] == self.lstm_bottleneck_1:
                latent_mean_vec = layers[x](output)
                latent_mean_vec = self.lstm_repeat_vec(latent_mean_vec)
            elif layers[x] == self.lstm_bottleneck_2:
                latent_var_vec = layers[x](output)
                latent_var_vec = self.lstm_repeat_vec(latent_var_vec)
            else:
                output = layers[x](output)
        return latent_mean_vec, latent_var_vec

    def run_lstm_decoder(self, encoded):
        layers = [self.lstm_6, self.dropout, self.lstm_7, self.dropout, self.lstm_8, self.dropout, self.lstm_9, self.dropout, self.lstm_10, self.time_dist_1]
        output = encoded
        for x in range(len(layers)):
            output = layers[x](output)
        return output

    def run_encoder(self, input_data, batch_layers_flag):
        if batch_layers_flag is True:
            layers = [self.input, self.hidden_1, self.batch1, self.leaky, self.hidden_2, self.batch2, self.leaky, self.hidden_3, self.batch3, self.leaky, self.hidden_4, self.batch4, self.leaky,
                  self.hidden_5, self.batch5, self.leaky, self.hidden_6, self.batch6, self.leaky, self.hidden_7, self.batch7, self.leaky, self.z_mean, self.z_log_var]
        else:
            layers = [self.input, self.hidden_1, self.leaky, self.hidden_2, self.leaky, self.hidden_3, self.leaky, self.hidden_4, self.leaky, self.hidden_5, self.leaky, self.hidden_6, self.leaky, self.hidden_7, self.leaky, self.z_mean, self.z_log_var]
        output = input_data
        for x in range(len(layers)):
            if layers[x] == self.z_mean:
                latent_mean_vec = layers[x](output)
            elif layers[x] == self.z_log_var:
                latent_var_vec = layers[x](output)
            output = layers[x](output)
        return latent_mean_vec, latent_var_vec

    def run_decoder(self, encoded, batch_layers_flag):
        if batch_layers_flag is True:
            layers = [self.hidden_8, self.batch8, self.leaky, self.hidden_9, self.batch9, self.leaky, self.hidden_10, self.batch10, self.leaky, self.hidden_11, self.batch11, self.leaky,
                  self.hidden_12, self.batch12, self.leaky, self.hidden_13, self.batch13, self.leaky, self.hidden_14, self.batch14, self.leaky, self.output_decoder]
        else:
            layers = [self.hidden_8, self.leaky, self.hidden_9, self.leaky, self.hidden_10, self.leaky, self.hidden_11, self.leaky, self.hidden_12, self.leaky, self.hidden_13, self.leaky, self.hidden_14, self.leaky, self.output_decoder]
        output = encoded
        for x in range(len(layers)):
            output = layers[x](output)
        return output

    @staticmethod
    def compute_post(args):
        z_mean, z_log_var, lam_mean, lam_log_var = args
        post_mean = (z_mean / (1 + K.exp(z_log_var - lam_log_var))) + (lam_mean / (1 + K.exp(lam_log_var - z_log_var)))
        post_log_var = z_log_var + lam_log_var - K.log(K.exp(z_log_var) + K.exp(lam_log_var))
        return [post_mean, post_log_var]

    @staticmethod
    def get_mean_embed(args):
        num_classes, latent_dim_for_embedding, input_labels = args
        return Embedding(num_classes, latent_dim_for_embedding, input_length=1)(K.constant(input_labels))

    @staticmethod
    def get_var_embed(args):
        num_classes, latent_dim_for_embedding, input_labels = args
        return Embedding(num_classes, latent_dim_for_embedding, input_length=1)(K.constant(input_labels))

    @staticmethod
    def sample_from_latent(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def run_embedding_lstm_encoder(self, input_data):
        latent_mean_vec, latent_var_vec = self.run_lstm_encoder(input_data)
        mean_for_embedding = self.mean_embed(K.constant(input_labels))
        var_for_embedding = self.var_embed(K.constant(input_labels))
        latent_mean_vec = latent_mean_vec.numpy().reshape(latent_mean_vec.numpy().shape[0] * latent_mean_vec.numpy().shape[1], latent_mean_vec.numpy().shape[-1])
        latent_var_vec = latent_var_vec.numpy().reshape(latent_var_vec.numpy().shape[0] * latent_var_vec.numpy().shape[1], latent_var_vec.numpy().shape[-1])
        posterior = self.compute_post([latent_mean_vec, latent_var_vec, mean_for_embedding, var_for_embedding])
        return posterior, mean_for_embedding, var_for_embedding

    def run_lstm_autoencoder(self, input_data, autoencoder_parameters):
        latent_mean_vec, latent_var_vec = self.run_lstm_encoder(input_data)
        # latent_mean_vec, latent_var_vec = self.run_lstm_transformer_encoder(input_data)
        if autoencoder_parameters['autoencoder_run_embedding'] is True:
            posterior, mean_for_embedding, var_for_embedding = self.run_embedding_lstm_encoder(input_data)
        else:
            posterior = [latent_mean_vec, latent_var_vec]
            mean_for_embedding, var_for_embedding = 0, 0
        if autoencoder_parameters['vae_latent_reg_flag'] is True:
            encoded = self.z([posterior[0], posterior[1]])
        else:
            encoded = posterior[0]
        recon_output = self.run_lstm_decoder(encoded)
        encoded = encoded.numpy().reshape(encoded.numpy().shape[0] * encoded.numpy().shape[1], encoded.numpy().shape[2])
        return recon_output, encoded, mean_for_embedding, var_for_embedding, posterior

    def run_vanilla_autoencoder(self, input_data, autoencoder_parameters):
        # get information for distribution from encoder
        latent_mean_vec, latent_var_vec = self.run_encoder(input_data, batch_layers_flag=autoencoder_parameters['autoencoder_batch_layers_flag'])
        # get embedding and posterior information
        if autoencoder_parameters['autoencoder_run_embedding'] is True:
            mean_for_embedding = self.mean_embed(K.constant(input_labels))
            var_for_embedding = self.var_embed(K.constant(input_labels))
            posterior = self.compute_post([latent_mean_vec, latent_var_vec, mean_for_embedding, var_for_embedding])
        else:
            posterior = [latent_mean_vec, latent_var_vec]
            mean_for_embedding, var_for_embedding = 0, 0
        # get potential latent sampling (VAE)
        if autoencoder_parameters['vae_latent_reg_flag'] is True:
            encoded = self.z([posterior[0], posterior[1]])
        else:
            encoded = posterior[0]
        # run the encoded representation into decoder
        recon_output = self.run_decoder(encoded, batch_layers_flag=autoencoder_parameters['autoencoder_batch_layers_flag'])
        return recon_output, encoded, mean_for_embedding, var_for_embedding, posterior

    def get_loss(self, mlp_model, num_classes, input_data, true_values, autoencoder_parameters):
        if autoencoder_parameters['run_LSTM'] is True:
            recon_output, encoded, mean_for_embedding, var_for_embedding, posterior = self.run_lstm_autoencoder(input_data, autoencoder_parameters)
        else:
            recon_output, encoded, mean_for_embedding, var_for_embedding, posterior = self.run_vanilla_autoencoder(input_data, autoencoder_parameters)
        if autoencoder_parameters['run_mlp'] is True:
            class_acc, class_loss = mlp_model.mlp_learn(true_labels=input_labels, reconstructed_latent=encoded, num_classes=num_classes)
        else:
            class_acc = 0
            class_loss = 0
            acc = 0
        # TAKING AVERAGE MSE LOSS ACROSS LATENT DIMENSIONS TO GET [BATCH SIZE,] TENSOR
        recon_loss = tf.reduce_sum(tf.math.square(recon_output - true_values), axis=-1) / autoencoder_parameters['lstm_sequence_length']
        if autoencoder_parameters['autoencoder_run_embedding'] is True:
            # KL LOSS BETWEEN THE DISTRIBUTIONS OF POSTERIOR (LABELS AND FIRING RATE COMBINED DIST) + DISTRIBUTION OF THE LABEL EMBEDDING THEMSELVES... WANT TO MINIMIZE THE DISTANCE BETWEEN THE TWO
            post_mean = posterior[0]
            post_log_var = posterior[1]
            kl_loss = 1 + post_log_var - var_for_embedding - ((tf.math.square(post_mean - mean_for_embedding) + tf.math.exp(post_log_var)) / tf.math.exp(var_for_embedding))
            kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=-1)
            if autoencoder_parameters['run_LSTM'] is True:
                kl_loss = kl_loss.numpy().reshape(round(kl_loss.numpy().shape[0]/autoencoder_parameters['lstm_sequence_length']), autoencoder_parameters['lstm_sequence_length'])
            loss = recon_loss + kl_loss
        else:
            loss = recon_loss
        if autoencoder_parameters['run_LSTM'] is True:
            acc = (np.sum(np.equal(np.around(true_values, 4), np.around(recon_output, 4))) / (np.shape(true_values)[0] * np.shape(true_values)[1] * np.shape(true_values)[2])) * 100
        else:
            acc = (np.sum(np.equal(np.around(true_values, 4), np.around(recon_output, 4))) / (np.shape(true_values)[0] * np.shape(true_values)[1])) * 100
        return class_acc, class_loss, loss, acc, encoded

    def get_gradient(self, mlp_model, num_classes, input_data, true_values, autoencoder_parameters):
        with tf.GradientTape() as tape:
            layer_vars = [self.input.variables, self.mean_embed.variables, self.var_embed.variables, self.hidden_1.variables, self.hidden_2.variables, self.hidden_3.variables, self.hidden_4.variables,
                          self.hidden_5.variables, self.hidden_6.variables, self.hidden_7.variables, self.batch1.variables, self.batch2.variables, self.batch3.variables, self.batch4.variables,
                          self.batch5.variables, self.batch6.variables, self.batch7.variables, self.batch8.variables, self.batch9.variables, self.batch10.variables, self.batch11.variables,
                          self.batch12.variables, self.batch13.variables, self.batch14.variables, self.hidden_8.variables, self.hidden_9.variables, self.hidden_10.variables, self.hidden_11.variables,
                          self.hidden_12.variables, self.hidden_13.variables, self.hidden_14.variables, self.output_decoder.variables, self.lstm_1.variables, self.lstm_2.variables, self.lstm_3.variables,
                          self.lstm_4.variables, self.lstm_5.variables, self.lstm_6.variables, self.lstm_7.variables, self.lstm_8.variables, self.lstm_9.variables, self.lstm_10.variables,
                          self.lstm_bottleneck_1.variables, self.lstm_bottleneck_2.variables]
            for x in range(len(layer_vars)):
                tape.watch(layer_vars[x])
            class_acc, class_loss, loss, acc, encoded = self.get_loss(mlp_model, num_classes, input_data, true_values,
                                                                      autoencoder_parameters=autoencoder_parameters)
            if autoencoder_parameters['run_LSTM'] is True:
                layer_vars2 = [self.lstm_1.variables[0], self.lstm_2.variables[0], self.lstm_3.variables[0], self.lstm_4.variables[0], self.lstm_5.variables[0], self.lstm_6.variables[0], self.lstm_7.variables[0],
                               self.lstm_8.variables[0], self.lstm_9.variables[0], self.lstm_10.variables[0], self.lstm_bottleneck_1.variables[0]]
                if autoencoder_parameters['vae_latent_reg_flag'] is True:
                    layer_vars2.extend( self.lstm_bottleneck_2.variables[0])
            else:
                layer_vars2 = [self.input.variables[0], self.input.variables[1], self.hidden_1.variables[0], self.hidden_1.variables[1], self.hidden_2.variables[0],
                           self.hidden_2.variables[1], self.hidden_3.variables[0], self.hidden_3.variables[1], self.hidden_4.variables[0], self.hidden_4.variables[1], self.hidden_5.variables[0], self.hidden_5.variables[1],
                           self.hidden_6.variables[0], self.hidden_6.variables[1], self.hidden_7.variables[0], self.hidden_7.variables[1], self.hidden_8.variables[0], self.hidden_8.variables[1], self.hidden_9.variables[0],
                           self.hidden_9.variables[1], self.hidden_10.variables[0], self.hidden_10.variables[1], self.hidden_11.variables[0], self.hidden_11.variables[1], self.hidden_12.variables[0], self.hidden_12.variables[1],
                           self.hidden_13.variables[0], self.hidden_13.variables[1], self.hidden_14.variables[0], self.hidden_14.variables[1], self.output_decoder.variables[0], self.output_decoder.variables[1]]
                if autoencoder_parameters['autoencoder_batch_layers_flag'] is True:
                    layer_vars2.extend([self.batch1.variables[0], self.batch1.variables[1], self.batch2.variables[0], self.batch2.variables[1], self.batch3.variables[0], self.batch3.variables[1], self.batch4.variables[0],
                           self.batch4.variables[1], self.batch5.variables[0], self.batch5.variables[1], self.batch6.variables[0], self.batch6.variables[1], self.batch7.variables[0], self.batch7.variables[1],
                           self.batch8.variables[0], self.batch8.variables[1], self.batch9.variables[0], self.batch9.variables[1], self.batch10.variables[0], self.batch10.variables[1], self.batch11.variables[0],
                           self.batch11.variables[1], self.batch12.variables[0], self.batch12.variables[1], self.batch13.variables[0], self.batch13.variables[1], self.batch14.variables[0], self.batch14.variables[1]])

            if autoencoder_parameters['autoencoder_run_embedding'] is True:
                layer_vars2.extend([self.mean_embed.variables[0], self.var_embed.variables[0]])
            autoencoder_gradient = tape.gradient(loss, layer_vars2)
            return class_acc, class_loss, autoencoder_gradient, loss, acc, encoded, layer_vars2

    def network_learn(self, mlp_model, num_classes, input_data, true_values, autoencoder_parameters):
        class_acc, class_loss, autoencoder_gradient, loss, acc, encoded, layer_vars2 = self.get_gradient(mlp_model, num_classes, input_data, true_values, autoencoder_parameters)
        self.ae_op.apply_gradients(zip(autoencoder_gradient, layer_vars2))
        return class_acc, class_loss, loss, acc, encoded


class MLP():
    def __init__(self, mlp_parameters, autoencoder_parameters, mlp_optimizer, mlp_reg, num_classes):
        # mlp
        self.mlp_input = Dense(autoencoder_parameters['autoencoder_batch_size'], input_shape=(None, None))
        self.mlp_layer_1 = Dense(round(autoencoder_parameters['autoencoder_batch_size'] * 0.9), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l2(mlp_reg))
        self.mlp_layer_2 = Dense(round(autoencoder_parameters['autoencoder_batch_size'] * 0.7), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l2(mlp_reg))
        self.mlp_layer_3 = Dense(round(autoencoder_parameters['autoencoder_batch_size'] * 0.5), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l2(mlp_reg))
        self.mlp_layer_4 = Dense(round(autoencoder_parameters['autoencoder_batch_size'] * 0.3), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l2(mlp_reg))
        self.mlp_layer_5 = Dense(round(autoencoder_parameters['autoencoder_batch_size'] * 0.1), bias_initializer='zeros', kernel_initializer='glorot_uniform', activity_regularizer=l2(mlp_reg))
        self.mlp_output_sigmoid = Dense(num_classes, activation='sigmoid')
        self.mlp_output_softmax = Dense(num_classes, activation='softmax')
        # mlp_optimizer
        self.mlp_op = mlp_optimizer
        # leaky
        self.mlp_leaky = LeakyReLU(alpha=0.3)

    def run_mlp(self, reconstructed_latent):
        """
        maybe think about having separate gradient for this + optimizer and looop through this
        need to add leaky relu activation
        """
        mlp_input = self.mlp_input(reconstructed_latent)
        mlp1 = self.mlp_layer_1(mlp_input)
        mlp_leaky1 = self.mlp_leaky(mlp1)
        mlp2 = self.mlp_layer_2(mlp_leaky1)
        mlp_leaky2 = self.mlp_leaky(mlp2)
        mlp3 = self.mlp_layer_3(mlp_leaky2)
        mlp_leaky3 = self.mlp_leaky(mlp3)
        mlp4 = self.mlp_layer_4(mlp_leaky3)
        mlp_leaky4 = self.mlp_leaky(mlp4)
        mlp5 = self.mlp_layer_5(mlp_leaky4)
        mlp_leaky5 = self.mlp_leaky(mlp5)
        mlp_output_softmax = self.mlp_output_softmax(mlp_leaky5)
        return mlp_output_softmax

    def mlp_gradient(self, true_labels, reconstructed_latent, num_classes):
        with tf.GradientTape() as tape2:
            # mlp
            tape2.watch(self.mlp_input.variables)
            tape2.watch(self.mlp_layer_1.variables)
            tape2.watch(self.mlp_layer_2.variables)
            tape2.watch(self.mlp_layer_3.variables)
            tape2.watch(self.mlp_layer_4.variables)
            tape2.watch(self.mlp_layer_5.variables)
            tape2.watch(self.mlp_output_softmax.variables)
            class_acc, class_loss = self.get_mlp_loss(true_labels, reconstructed_latent, num_classes)
            mlp_gradient = tape2.gradient(class_loss, [self.mlp_input.variables[0], self.mlp_input.variables[1],
                                                self.mlp_layer_1.variables[0], self.mlp_layer_1.variables[1], self.mlp_layer_2.variables[0], self.mlp_layer_2.variables[1],
                                                self.mlp_layer_3.variables[0], self.mlp_layer_3.variables[1], self.mlp_layer_4.variables[0], self.mlp_layer_4.variables[1],
                                                self.mlp_layer_5.variables[0], self.mlp_layer_5.variables[1], self.mlp_output_softmax.variables[0], self.mlp_output_softmax.variables[1]])
        return mlp_gradient, class_acc, class_loss

    def mlp_learn(self, true_labels, reconstructed_latent, num_classes):
        mlp_gradient, class_acc, class_loss = self.mlp_gradient(true_labels, reconstructed_latent, num_classes)
        self.mlp_op.apply_gradients(zip(mlp_gradient, [self.mlp_input.variables[0], self.mlp_input.variables[1], self.mlp_layer_1.variables[0], self.mlp_layer_1.variables[1],
                                                       self.mlp_layer_2.variables[0], self.mlp_layer_2.variables[1], self.mlp_layer_3.variables[0], self.mlp_layer_3.variables[1],
                                                       self.mlp_layer_4.variables[0], self.mlp_layer_4.variables[1], self.mlp_layer_5.variables[0], self.mlp_layer_5.variables[1],
                                                       self.mlp_output_softmax.variables[0], self.mlp_output_softmax.variables[1]]))
        return class_acc, class_loss

    def get_mlp_loss(self, true_labels, reconstructed_latent, num_classes):
        mlp_output_softmax = self.run_mlp(reconstructed_latent)
        # mlp_output = [200, 8] array of probability value for each feature or neuron belonging to a particular class --> find max prob for class
        # mlp_output_softmax = mlp_output_softmax.numpy()
        # mlp_output_softmax = mlp_output_softmax.reshape(mlp_output_softmax.shape[0]*mlp_output_softmax.shape[1], mlp_output_softmax.shape[2])
        class_acc = (np.sum(np.equal(np.argmax(mlp_output_softmax, axis=1), true_labels)) / len(true_labels)) * 100
        encoded_labels = self.one_hot_encoding(labels=true_labels, num_classes=num_classes, convert_back=False)
        class_loss = -tf.reduce_sum(tf.math.log(mlp_output_softmax) * encoded_labels)
        return class_acc, class_loss

    @staticmethod
    def one_hot_encoding(labels, num_classes, convert_back):
        if convert_back is False:
            hot_encoded_labels = np.zeros((len(labels), num_classes))
            for x in range(len(labels)):
                hot_encoded_labels[x, int(labels[x])] = 1
            return hot_encoded_labels
        else:
            return [x for x in np.argwhere(labels == 1)[:, 1]]
