# Dependencies :
import time 
start_time = time.time()
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 12
plt.figure(dpi=500)

import tensorflow as tf 
print(tf.__version__)
tf.config.list_physical_devices('GPU')

# System Parameters 
k = 4       # Number of bits to be encoded
n = 7       # Size of the codeword 
M = 2 ** k  # Size of the alphabet
R = k / n   # Information rate
SNR_TRAIN = float(input('Enter training SNR (in dB): '))

# Generation of training data
training_set_size = 10 ** 6

# Random indices :
sample_indices = np.random.randint(0, M, training_set_size)

# Converting to 1-hot encoded vectors 
x_train = np.zeros((training_set_size, M))
x_train[np.arange(training_set_size), sample_indices] = 1 
y_train = x_train

# Generate random fading taps for training :: CSI @ Rx
fade_mean = 0 
fade_std = np.sqrt(0.5)
fading_taps_real = np.random.normal(fade_mean, fade_std, (training_set_size, 2*n))
fading_taps_imag = np.random.normal(fade_mean, fade_std, (training_set_size, 2*n))

###################################### Creating the Autoencoder #########################################
# Custom Layer Definition
class DuplicateLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DuplicateLayer, self).__init__()

    def call(self, inputs):
        return tf.concat([inputs, inputs], axis=-1)

# Test the custom layer
input_data = tf.constant([[1, 2, 3]], dtype=tf.float32)
custom_layer = DuplicateLayer()
output_data = custom_layer(input_data)
print(output_data)

# Encoder(== transmitter) part :
enc_input_layer_bits = tf.keras.Input(shape=(M, ), name="Input_Layer")
enc_layer_01 = tf.keras.layers.Dense(M, activation='relu', name="Enc_Hidden_Layer_01")(enc_input_layer_bits)
enc_layer_02 = tf.keras.layers.Dense(n, activation='relu', name="Enc_Hidden_Layer_02")(enc_layer_01)
enc_layer_03 = tf.keras.layers.Dense(n, activation='linear', name="Enc_Hidden_Layer_03")(enc_layer_02)
enc_layer_normalized = tf.keras.layers.Lambda((lambda x: np.sqrt(n) * tf.keras.backend.l2_normalize(x, axis=-1)), name='Power_Constraint')(enc_layer_03)

# Two paths of antenna 01 and antenna 02 :: by duplicating the output of the encoder 
# || Antenna 01 | Antenna 02 ||
signal_01_02 = DuplicateLayer()(enc_layer_normalized)

# Channel 
fading_layer_I = tf.keras.Input(shape=(2*n, ), name='fading_taps_I')
fading_layer_Q = tf.keras.Input(shape=(2*n, ), name='fading_taps_Q')
faded_signal_I = tf.keras.layers.Multiply()([signal_01_02, fading_layer_I])
faded_signal_Q = tf.keras.layers.Multiply()([signal_01_02, fading_layer_Q])
# AWGN
SNR_lin = 10 ** (SNR_TRAIN / 10)
rx_noisy_signal_I = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_I')(faded_signal_I)
rx_noisy_signal_Q = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_Q')(faded_signal_Q)

# Decoder (==Receiver) part :
rx_signal = tf.keras.layers.Concatenate()([rx_noisy_signal_I, rx_noisy_signal_Q])
dec_layer_01 = tf.keras.layers.Dense(4*n, activation='relu', name='Decoder_Hidden_01')(rx_signal)
dec_layer_02 = tf.keras.layers.Dense(4*M, activation='relu', name='Decoder_Hidden_02')(dec_layer_01)
dec_layer_03 = tf.keras.layers.Dense(4*M, activation='relu', name='Decoder_Hidden_03')(dec_layer_02)
dec_layer_04 = tf.keras.layers.Dense(2*M, activation='relu', name='Decoder_Hidden_04')(dec_layer_03)
dec_output_layer = tf.keras.layers.Dense(M, activation='softmax', name='Output_Layer')(dec_layer_04)

autoencoder = tf.keras.Model(
                    inputs = [enc_input_layer_bits, fading_layer_I, fading_layer_Q], 
                    outputs = [dec_output_layer],
                )

# Compiling the model :
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
autoencoder.summary()

## Uncomment to view the block diagram of the Neural network ; Requires GraphViz
# tf.keras.utils.plot_model(
#     autoencoder,
#     to_file='deepSIMO.png',
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=True,
#     dpi=200,
#     show_layer_activations=True,
#     show_trainable=True,
# )

##### Training the autoencoder
autoencoder.fit([x_train, fading_taps_real, fading_taps_imag], y_train, batch_size=1000, epochs=500)

######################################### Testing the above learnt system ###############################
# Generation of testing/ validation data 
testing_set_size = 10 ** 6

# Random samples :
y_test = np.random.randint(0, M, testing_set_size)

# 1 hot encoded vectors 
x_test = np.zeros((testing_set_size, M))
x_test[np.arange(testing_set_size), y_test] = 1
print(x_test)

# Abstracting the encoder and decoder for separate operations
# Encoder 
encoder_model = tf.keras.Model(enc_input_layer_bits, signal_01_02)

# Decoder  
signal_rx_real = tf.keras.Input(shape=(2*n, ))
signal_rx_imag = tf.keras.Input(shape=(2*n, ))

decoder_output = autoencoder.layers[-6]([signal_rx_real, signal_rx_imag])
decoder_output = autoencoder.layers[-5](decoder_output)
decoder_output = autoencoder.layers[-4](decoder_output)
decoder_output = autoencoder.layers[-3](decoder_output)
decoder_output = autoencoder.layers[-2](decoder_output)
decoder_output = autoencoder.layers[-1](decoder_output)
decoder_model = tf.keras.Model([signal_rx_real, signal_rx_imag], decoder_output)

autoencoder.save('deepSIMO.keras')
encoder_model.save('deepSIMO_enc.keras')
decoder_model.save('deepSIMO_dec.keras')
# Validation Routine
# Range of Signal to Noise Ratio :
SNR_dB = np.linspace(-2, 20, 20)
SNR_lin = 10 ** (SNR_dB / 10)
# Fixing energy per bit :
E_b = 1 
# Range of noise variance accordingly :
noise_var = 1 / (2 * R * SNR_lin) 

# Encoding 
encoded_signal = encoder_model.predict(x_test)
# Energy Verification :
with open(f'results_deepSIMO_{int(SNR_TRAIN)}.txt', mode='w') as file_id:
    file_id.write(f'Number of bits = {testing_set_size * k}\n')
    file_id.write(f'Energy of the signal = {np.linalg.norm(encoded_signal) ** 2}\n')
    file_id.write(f'-----------------------------------------------------\n')

count = 0 
BLER_learned_simo_csir = []
for noise in noise_var :
    # Fading 
    fade_taps = np.random.normal(fade_mean, fade_std, encoded_signal.shape) + 1j* np.random.normal(0, fade_std, encoded_signal.shape)
    rx_signal = fade_taps * encoded_signal
    # Noise
    noise_samples = np.random.normal(0, np.sqrt(noise), rx_signal.shape) + 1j* np.random.normal(0, np.sqrt(noise), rx_signal.shape)
    rx_signal += noise_samples 
    # Decoding 
    decoded_signal = decoder_model.predict([np.real(rx_signal), np.imag(rx_signal)])
    estimated_vectors = np.argmax(decoded_signal, axis=-1)
    BLER_learned_simo_csir.append(np.sum(estimated_vectors != y_test) / testing_set_size)
    # Progress Update 
    count += 1 
    print(f'Progress : {100 * count // len(noise_var)}')

fig, axes = plt.subplots()
BLER_uncoded_mrc = [0.325984, 0.257417, 0.197093, 0.145022, 0.10335, 0.070514, 0.046663, 0.031214, 0.019661, 0.012328, 0.007551, 0.004637, 0.002844, 0.00165, 0.000972, 0.000609, 0.000397, 0.000218, 0.000129, 8e-05]
BLER_coded_mrc_hard = [0.278303, 0.203482, 0.139147, 0.089367, 0.052874, 0.029171, 0.014854, 0.00708, 0.003166, 0.00132, 0.000566, 0.000221, 8.1e-05, 3.5e-05, 3e-06, 2e-06, 1e-06, 0.0, 0.0, 0.0]
BLER_coded_mrc_mld = [0.17427, 0.110159, 0.06344, 0.032253, 0.014692, 0.005904, 0.002147, 0.000744, 0.000209, 5.5e-05, 1e-05, 3e-06, 3e-06, 2e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

axes.semilogy(SNR_dB, BLER_uncoded_mrc, label='Uncoded BPSK', color='black', marker='o')
axes.semilogy(SNR_dB, BLER_coded_mrc_hard, label='Hamming (7, 4) Hard', color='blue', marker='s')
axes.semilogy(SNR_dB, BLER_coded_mrc_mld, label='Hamming (7, 4) MLD', color='green', marker='>')
axes.semilogy(SNR_dB, BLER_learned_simo_csir, label='Learned', color='red', marker="o", markerfacecolor="none")
axes.set_xlabel('$SNR (in dB)$')
axes.set_ylabel('$BLER$')
axes.set_title('$1 x 2$ SIMO + CSIR')
axes.legend()
axes.grid()

# Saving the results for future reference :
# Plots :
fig.savefig(f'results_deepSIMO_{int(SNR_TRAIN)}.png')
# BLER values :
with open(file=f'results_deepSIMO_{int(SNR_TRAIN)}.txt', mode='a') as file_id:
    file_id.write(f'BLER_uncoded_mrc = {BLER_uncoded_mrc}\n')
    file_id.write(f'BLER_coded_mrc_hard = {BLER_coded_mrc_hard}\n')
    file_id.write(f'BLER_coded_mrc_mld = {BLER_coded_mrc_mld}\n')
    file_id.write(f'BLER_learned_simo_csir = {BLER_learned_simo_csir}\n')
    file_id.write(f'------------------------------------------------------\n')
    file_id.write(f'Total execution time = {(time.time() - start_time) / 60} mins')