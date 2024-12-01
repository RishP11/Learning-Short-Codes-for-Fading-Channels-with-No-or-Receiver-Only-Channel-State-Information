# Dependencies :
import time 
start_time = time.time()

import numpy as np 
import tensorflow as tf 

import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 12
plt.figure(dpi=500)

print(tf.__version__)
tf.config.list_physical_devices('GPU')

# System Specifications
k = 1           # Input bits to be encoded at a time
n = 2           # Number or output branches of the autoencoder 
M = 2 ** k      # Size of the alphabet 
R = k / n       # Information rate
E_b = 1         # Fixed energy per bit
SNR_TRAIN = float(input('Enter Training SNR (in dB) : '))

# Generation of training data  
training_set_size = 10 ** 6 

# Generating random labels :
sample_indices = np.random.randint(0, M, training_set_size)

# Converting to 1hot encoded vectors
x_train = np.zeros((training_set_size, M))
x_train[np.arange(training_set_size), sample_indices] = 1

# Since we wish to reproduce the inputs at the outputs:
y_train = x_train

print(f'Sample Input :{x_train[0]}')
print(f'Input shape :{np.shape(x_train)}')

# Generating the random fading taps for training == CSI @ Tx and Rx
fade_mean = 0
fade_std = np.sqrt(0.5)
# fading_taps = np.random.normal(fade_mean, fade_std, (training_set_size, n)) + 1j*np.random.normal(fade_mean, fade_std, (training_set_size,n))
fading_taps = np.random.normal(fade_mean, fade_std, (training_set_size, n, n)) + 1j*np.random.normal(fade_mean, fade_std, (training_set_size,n, n))

print(f'Sample Channel Matrices : {fading_taps[:5]}')
print(f'Sample Channel Matrices : {np.real(fading_taps[:5])}')

class MatMulLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MatMulLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Ensure inputs are a list of two tensors
        if isinstance(inputs, list) and len(inputs) == 2:
            x, y = inputs

            # Reshape y to be a 2D tensor with shape (2, 1)
            y = tf.reshape(y, [-1, 1])

            # Perform matrix multiplication
            result = tf.matmul(x, y)

            return result
        else:
            raise ValueError("Input must be a list of two tensors.")

# Define tensors
tensor_b = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
tensor_a_1d = tf.constant([1, 2], dtype=tf.float32)

# Apply the layer
result_1d = MatMulLayer()([tensor_b, tensor_a_1d])

print(result_1d)

# Encoder(== transmitter) part :
enc_input_layer_bits = tf.keras.Input(shape=(M, ), name="Input_Layer")
enc_layer_01 = tf.keras.layers.Dense(M, activation='relu', name="Enc_Hidden_Layer_01")(enc_input_layer_bits)
enc_layer_02 = tf.keras.layers.Dense(n, activation='linear', name="Enc_Hidden_Layer_02")(enc_layer_01)
enc_layer_normalized = tf.keras.layers.Lambda((lambda x: np.sqrt(E_b * n) * tf.keras.backend.l2_normalize(x, axis=-1)), name='Power_Constraint')(enc_layer_02)

# Channel :
# y = fx + w  model(flat + fast fading):
fading_layer_I = tf.keras.Input(shape=(n, n), name='fading_taps_I')
fading_layer_Q = tf.keras.Input(shape=(n, n), name='fading_taps_Q')
faded_signal_I = MatMulLayer()([fading_layer_I, enc_layer_normalized])
faded_signal_Q = MatMulLayer()([fading_layer_Q, enc_layer_normalized])

# fading_layer_I = tf.keras.Input(shape=(n,), name='fading_taps_I')
# fading_layer_Q = tf.keras.Input(shape=(n,), name='fading_taps_Q')
# faded_signal_I = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_I])
# faded_signal_Q = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_Q])

# Noise :
SNR_lin = 10 ** (SNR_TRAIN / 10)
rx_noisy_signal_I = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_I')(faded_signal_I)
rx_noisy_signal_Q = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_Q')(faded_signal_Q)

# Decoder (==Receiver) part :
rx_signal = tf.keras.layers.Concatenate(axis=-1)([rx_noisy_signal_I, rx_noisy_signal_Q])
dec_layer_01 = tf.keras.layers.Dense(8*n, activation='relu', name='Decoder_Hidden_01')(rx_signal)
dec_layer_03 = tf.keras.layers.Dense(4*M, activation='relu', name='Decoder_Hidden_03')(dec_layer_01)
dec_layer_04 = tf.keras.layers.Dense(2*M, activation='relu', name='Decoder_Hidden_04')(dec_layer_03)
dec_output_layer = tf.keras.layers.Dense(M, activation='softmax', name='Output_Layer')(dec_layer_04)

autoencoder = tf.keras.Model(
                    inputs = [enc_input_layer_bits, fading_layer_I, fading_layer_Q], 
                    outputs = [dec_output_layer])

# Compiling the model :
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
autoencoder.summary()

# # Uncomment the following to view the block diagram of the neural network ; Requires GraphViz
# tf.keras.utils.plot_model(
#     autoencoder,
#     to_file=f'deepVariable_{k}_{n}.png',
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=True,
#     dpi=200,
#     show_layer_activations=True,
#     show_trainable=True,
# )

autoencoder.fit([x_train, np.real(fading_taps), np.imag(fading_taps)], y_train, batch_size=1000, epochs=500)
autoencoder.save('autoencoder.keras')

