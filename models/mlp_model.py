import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras import Model, Sequential
import tensorflow.keras.utils
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU, Flatten, GaussianDropout, Lambda, LSTM, TimeDistributed, RepeatVector, Embedding, LayerNormalization, Dropout, MultiHeadAttention, Conv1D
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
from plot_outputs import *
import time


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
