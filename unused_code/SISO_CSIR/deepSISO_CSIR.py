# Dependencies :
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 12
plt.figure(dpi=500)

import time 
t0 = time.time()

import tensorflow as tf
print(tf.__version__)
tf.config.list_physical_devices('GPU')


# System Parameters
M = 2 ** 4                  # Size of alphabet
k = 4                       # Number of bits required
n = 7                       # Size of coded vector 
R = k / n                   # Information/ Communication rate 
SNR_TRAIN = float(input('Enter training SNR (in dB): ')) 

# Generation of the training data 
training_set_size = 10 ** 6 
sample_indices = np.random.randint(0, M, training_set_size)

# Converting the indices to 1-hot vectors
x_train = np.zeros((training_set_size, M))  
x_train[np.arange(training_set_size), sample_indices] = 1  # One-hot encoding
print(f'Samples = {x_train}')
# Since we are reproducing the input at the output :
y_train = x_train 

# Generate random fading taps for training == CSI @ Rx
fade_mean = 0 
fade_std = np.sqrt(0.5)
fade_taps_real = np.random.normal(fade_mean, fade_std, (training_set_size, n)) 
fade_taps_imag = np.random.normal(fade_mean, fade_std, (training_set_size, n))

########################################### Creating the Autoencoder ####################################
# Encoder(transmitter) part :
enc_input_layer = tf.keras.Input(shape=(M,), name="Input_Layer")
enc_layer_01 = tf.keras.layers.Dense(M, activation='relu', name="Encoder_Hidden_01")(enc_input_layer)
enc_layer_02 = tf.keras.layers.Dense(n, activation='linear', name="Encoder_Hidden_02")(enc_layer_01)
enc_layer_normalized = tf.keras.layers.Lambda((lambda x: np.sqrt(n) * tf.keras.backend.l2_normalize(x, axis=-1)), name='Power_Constraint')(enc_layer_02)

# AWGN Channel with Rayleigh fading :
fading_layer_real = tf.keras.Input(shape=(n,), name='Fading_layer_real')
fading_layer_imag = tf.keras.Input(shape=(n,), name='Fading_layer_imag')
# y = fx + w  model(flat + fast fading):
rx_signal_real = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_real])
rx_signal_imag = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_imag])

# Gaussian noise :
SNR_lin = 10 ** (SNR_TRAIN / 10)
rx_noisy_signal_real = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_real')(rx_signal_real)
rx_noisy_signal_imag = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_imag')(rx_signal_imag)

# Decoder(receiver) part :
rx_signal = tf.keras.layers.Concatenate()([rx_noisy_signal_real, rx_noisy_signal_imag, fading_layer_real, fading_layer_imag])
dec_layer_01 = tf.keras.layers.Dense(4*M, activation='relu', name='Decoder_Hidden_01')(rx_signal)
dec_layer_02 = tf.keras.layers.Dense(M, activation='relu', name='Decoder_Hidden_02')(dec_layer_01)
dec_output_layer = tf.keras.layers.Dense(M, activation='softmax', name='Output_Layer')(dec_layer_02)

autoencoder = tf.keras.Model(
                    inputs = [enc_input_layer, fading_layer_real, fading_layer_imag], 
                    outputs = [dec_output_layer],
                )

# Compiling the model :
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
autoencoder.summary()

# # Uncomment the code block below to view the block diagram of the neural network defined above : (Requires graphviz module)
# tf.keras.utils.plot_model(
#     autoencoder,
#     to_file='deepSISOCSIR.png',
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=True,
#     dpi=200,
#     show_layer_activations=True,
#     show_trainable=True,
# )

# Training the above autoencoder
autoencoder.fit([x_train, fade_taps_real, fade_taps_imag], y_train, batch_size=1000, epochs=500)
# Save the model for reproduction purpose
autoencoder.save('modelSISOcsir.keras')

############################### Testing the above encoder at various SNRs ###############################
# Generation of testing/ validation data
test_data_size = 10 ** 6
y_test = np.random.randint(0, M, test_data_size)
x_test = []
for idx in y_test:
    temp = np.zeros(M)
    temp[idx] = 1
    x_test.append(temp)

x_test = tf.constant(x_test)
x_test

# Abstracting the learned encoder and decoder 
# Encoder :
encoder_model = tf.keras.Model(enc_input_layer, enc_layer_normalized)

# Decoder :
rx_noisy_signal_I = tf.keras.Input(shape=(n,))
rx_noisy_signal_Q = tf.keras.Input(shape=(n,))
fading_I = tf.keras.Input(shape=(n,))
fading_Q = tf.keras.Input(shape=(n,))
decoder_output = autoencoder.layers[-4]([rx_noisy_signal_I, rx_noisy_signal_Q, fading_I, fading_Q])
decoder_output = autoencoder.layers[-3](decoder_output)
decoder_output = autoencoder.layers[-2](decoder_output)
decoder_output = autoencoder.layers[-1](decoder_output)

decoder_model = tf.keras.Model([rx_noisy_signal_I, rx_noisy_signal_Q, fading_I, fading_Q], decoder_output)

################## Validation Routine
# Range of Signal to Noise Ratio :
n_points = 20
SNR_dB = np.linspace(-2, 20, n_points)
SNR_lin = 10 ** (SNR_dB / 10)
# Fixing energy per bit :
E_b = 1 
# Range of noise variance accordingly :
noise_var = 1 / (2 * R * SNR_lin) 

encoded_signal = encoder_model.predict(x_test)
### Signal Energy test ####
with open(f'results_deepSISOcsir_{int(SNR_TRAIN)}.txt', mode='w') as file_id:
    file_id.write(f'Number of bits: {test_data_size * k}\n')
    file_id.write(f'Total Signal Energy: {np.linalg.norm(encoded_signal) ** 2}\n')
    file_id.write(f'---------------------------------------\n')
###########################

BLER_learned_coded_csir = []
for noise in noise_var :
    # Fading 
    fade_taps = np.random.normal(fade_mean, fade_std, encoded_signal.shape) + 1j* np.random.normal(0, fade_std, encoded_signal.shape)
    faded_signal_rx = fade_taps * encoded_signal
    # Noise
    noise_samples = np.random.normal(0, np.sqrt(noise), faded_signal_rx.shape) + 1j* np.random.normal(0, np.sqrt(noise), faded_signal_rx.shape)
    rx_noisy_signal = faded_signal_rx + noise_samples 
    # Decoding 
    decoded_signal = decoder_model.predict([np.real(rx_noisy_signal), np.imag(rx_noisy_signal), np.real(fade_taps), np.imag(fade_taps)])
    estimated_vectors = np.argmax(decoded_signal, axis=-1)
    BLER_learned_coded_csir.append(np.sum(estimated_vectors != y_test) / test_data_size)

# Conventional Data :
BLER_uncoded_coherent = [0.568744, 0.510474, 0.453514, 0.393655, 0.337754, 0.283992, 0.234753, 0.192111, 0.155179, 0.124392, 0.098039, 0.077155, 0.059857, 0.046734, 0.036071, 0.028182, 0.021548, 0.016597, 0.012735, 0.009868] 
BLER_coded_coherent_hard = [0.536971, 0.469589, 0.400025, 0.329686, 0.262367, 0.201499, 0.148375, 0.106653, 0.073774, 0.049709, 0.032504, 0.020712, 0.013085, 0.008012, 0.004965, 0.003048, 0.00181, 0.001019, 0.000619, 0.000381] 
BLER_coded_coherent_mld = [0.464027, 0.389022, 0.312285, 0.240074, 0.174105, 0.119859, 0.078081, 0.047754, 0.027741, 0.015409, 0.008192, 0.004143, 0.002078, 0.001025, 0.000494, 0.000233, 0.000105, 4.3e-05, 2.8e-05, 9e-06] 

fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_coherent, label='Uncoded', color='black', marker='o')
axes.semilogy(SNR_dB, BLER_coded_coherent_hard, label='Hamming (7, 4) Hard', color='blue', marker='s')
axes.semilogy(SNR_dB, BLER_coded_coherent_mld, label='Hamming (7, 4) MLD', color='green', marker='*')
axes.semilogy(SNR_dB, BLER_learned_coded_csir, label='Learned', color='red', marker=">", markerfacecolor='none')
axes.set_xlabel(f'SNR (in dB)')
axes.set_ylabel(f'BLER')
axes.set_title(f'SISO CSIR - Coherent BPSK')
axes.legend()
axes.grid()

# Saving the results :
fig.savefig(f'results_deepSISOcsir_{int(SNR_TRAIN)}.png')
with open(f'results_deepSISOcsir_{int(SNR_TRAIN)}.txt', mode='a') as file_id :
    file_id.write(f'BLER_uncoded_coherent = {BLER_uncoded_coherent}\n')
    file_id.write(f'BLER_coded_coherent_hard = {BLER_coded_coherent_hard}\n')
    file_id.write(f'BLER_coded_coherent_mld = {BLER_coded_coherent_mld}\n')
    file_id.write(f'BLER_learned_coded_csir = {BLER_learned_coded_csir}\n')
    file_id.write(f'----------------------------------------\n')
    file_id.write(f'Total Execution time = {(time.time()-t0)/60} mins')