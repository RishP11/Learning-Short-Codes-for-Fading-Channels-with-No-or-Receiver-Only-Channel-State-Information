# Dependencies : 
import tensorflow as tf
import numpy as np 

import time 
t0 = time.time()

import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 12
plt.figure(dpi=500)

print(tf.__version__)
tf.config.list_physical_devices('GPU')

# System Parameters
k = 4                       # Number of bits required
M = 2 ** k                  # Size of alphabet
n = 7                       # Size of coded vector 
R = k / n                   # Information/ Communication rate 
SNR_TRAIN = float(input("Enter Training SNR (in dB) : "))

# Generation of the training data
training_set_size = 10 ** 6  
sample_indices = np.random.randint(0, M, training_set_size)

# Corresponding 1-hot vectors
x_train = np.zeros((training_set_size, M))
x_train[np.arange(training_set_size), sample_indices] = 1  # One-hot encoding
print(f'Random Bits = {sample_indices}')
print(f'One Hot encoded vectors :{x_train}')
print(f'Input Shape :{np.shape(x_train)}')
# Since we are reproducing the input at the output :
y_train = x_train 

# Generate random fading tap samples for training
fade_mean = 0 
fade_std = np.sqrt(0.5)
fade_taps_I = np.random.normal(fade_mean, fade_std, (training_set_size, 2*n)) 
fade_taps_Q = np.random.normal(fade_mean, fade_std, (training_set_size, 2*n))

##################################### End to End Autoencoder ###########################################
# Encoder(transmitter) part 
enc_input_layer = tf.keras.Input(shape=(M,), name="Input_Layer")
enc_layer_01 = tf.keras.layers.Dense(2*M, activation='relu', name="Encoder_Hidden_01")(enc_input_layer)
enc_layer_02 = tf.keras.layers.Dense(2*n, activation='linear', name="Encoder_Hidden_02")(enc_layer_01)
enc_layer_normalized = tf.keras.layers.Lambda((lambda x: np.sqrt(n) * tf.keras.backend.l2_normalize(x, axis=-1)), name='Power_Constraint')(enc_layer_02)

# Fading 
fading_layer_real = tf.keras.Input(shape=(2*n, ), name='fading_real_part')
fading_layer_imag = tf.keras.Input(shape=(2*n,), name='fading_taps_imag')
rx_signal_real = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_real])
rx_signal_imag = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_imag])

# Noise 
SNR_lin = 10 ** (SNR_TRAIN / 10)
rx_noisy_signal_real = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_I')(rx_signal_real)
rx_noisy_signal_imag = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_Q')(rx_signal_imag)

# Decoder(Receiver) 
# Concatenating the real and imag noisy signal at the decoder part :
rx_signal = tf.keras.layers.Concatenate()([rx_noisy_signal_real, rx_noisy_signal_imag])
dec_layer_01 = tf.keras.layers.Dense(16*M, activation='relu', name='Decoder_Hidden_01')(rx_signal)
dec_layer_02 = tf.keras.layers.Dense(16*M, activation='relu', name='Decoder_Hidden_02')(dec_layer_01)
dec_layer_03 = tf.keras.layers.Dense(8*M, activation='relu', name='Decoder_Hidden_03')(dec_layer_02)
dec_layer_04 = tf.keras.layers.Dense(M, activation='relu', name='Decoder_Hidden_04')(dec_layer_03)
dec_output_layer = tf.keras.layers.Dense(M, activation='softmax', name='Output_Layer')(dec_layer_04)

autoencoder = tf.keras.Model(
                    inputs = [enc_input_layer, fading_layer_real, fading_layer_imag], 
                    outputs = [dec_output_layer],
                )

# Compiling the model :
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
autoencoder.summary()

# Fitting the model by using the training set :
autoencoder.fit([x_train, fade_taps_I, fade_taps_Q], y_train, batch_size=1000, epochs=500)
# Save this model 
autoencoder.save(f'modelSISOnoCSI_coded_{int(SNR_TRAIN)}.keras')

############################### Analysing the Encoder and Decoder individually ##########################
# Abstracting out the decoder :
encoder_model = tf.keras.Model(enc_input_layer, enc_layer_normalized)

# Codeword Received at the receiver :
rx_noisy_signal_I = tf.keras.Input(shape=(2*n,))
rx_noisy_signal_Q = tf.keras.Input(shape=(2*n,))
dec_out = autoencoder.layers[-6]([rx_noisy_signal_I, rx_noisy_signal_Q])
dec_out = autoencoder.layers[-5](dec_out)
dec_out = autoencoder.layers[-4](dec_out)
dec_out = autoencoder.layers[-3](dec_out)
dec_out = autoencoder.layers[-2](dec_out)
dec_out = autoencoder.layers[-1](dec_out)

# Abstracting out the decoder model :
decoder_model = tf.keras.Model([rx_noisy_signal_I, rx_noisy_signal_Q], dec_out)

######################################## Testing for BLER performance ###################################
testing_set_size = 10 ** 6 
y_test = np.random.randint(0, M, testing_set_size)
x_test = np.zeros((testing_set_size, M))  
x_test[np.arange(testing_set_size), y_test] = 1  # One-hot encoding

print(f'Test bits = {y_test}')
print(f'Test inputs = {x_test}')
# Range of Signal to Noise Ratio :
n_points = 20 
SNR_dB = np.linspace(-2, 20, n_points)
SNR_lin = 10 ** (SNR_dB / 10)
# Fixing energy per bit :
E_b = 1 
# Range of total noise variance accordingly :
noise_var = 1 / (2 * SNR_lin) 

# Encoding : 
encoded_signal = encoder_model.predict(x_test)
# Signal energy verification :
with open(f'results_deepSISOnoCSI_coded_{int(SNR_TRAIN)}.txt', mode='w') as file_id:
    file_id.write(f'Number of bits = {testing_set_size * k}\n')
    file_id.write(f'Energy of the entire signal = {np.linalg.norm(encoded_signal) ** 2}\n')
    file_id.write(f'------------------------------------------\n')

BLER_learned_coded = []
count = 0 
for noise in noise_var:
    # Fading 
    fade_taps = np.random.normal(fade_mean, fade_std, encoded_signal.shape) + 1j * np.random.normal(fade_mean, fade_std, encoded_signal.shape)
    faded_signal = fade_taps * encoded_signal
    # Noise
    noise_samples = np.random.normal(0, np.sqrt(noise), encoded_signal.shape) + 1j * np.random.normal(0, np.sqrt(noise), encoded_signal.shape)
    noisy_signal = faded_signal + noise_samples
    # Decoding
    decoded_signal = decoder_model.predict([np.real(noisy_signal), np.imag(noisy_signal)])
    # Analysis
    estimated_vectors = np.argmax(decoded_signal, axis=-1)
    BLER_learned_coded.append(np.sum(estimated_vectors != y_test) / testing_set_size)
    count += 1 
    print(f'Progress : {100 * count // n_points} %')

# Comparing the performance
BLER_uncoded_ortho = [0.37974, 0.354447, 0.325193, 0.293239, 0.260487, 0.22836, 0.195176, 0.164309, 0.136198, 0.111775, 0.090265, 0.072551, 0.056927, 0.045165, 0.035206, 0.027456, 0.021321, 0.016435, 0.012747, 0.009633]
BLER_coded_ortho_hard = [0.441423, 0.424523, 0.405017, 0.382224, 0.352968, 0.319068, 0.279605, 0.237627, 0.194115, 0.151925, 0.114517, 0.083964, 0.058103, 0.03885, 0.025569, 0.016817, 0.010509, 0.006524, 0.004138, 0.002472]

fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_ortho, label='Orthogonal Uncoded', color='black', marker='o')
axes.semilogy(SNR_dB, BLER_coded_ortho_hard, label='Orthogonal Coded', color='blue', marker='s')
axes.semilogy(SNR_dB, BLER_learned_coded, label='Learned E2E', color='red', marker='>')

axes.set_xlabel(f'SNR(in dB)')
axes.set_ylabel(f'BLER')
axes.set_title(f'Uncoded SISO No CSI')
axes.legend()
axes.grid()

# Writing all the results to a file
fig.savefig(f'results_deepSISO_noCSI_coded_{int(SNR_TRAIN)}.png')
with open(f'results_deepSISOnoCSI_coded_{int(SNR_TRAIN)}.txt', mode='a') as file_id:
    file_id.write(f'BLER_uncoded_ortho = {BLER_uncoded_ortho}\n')
    file_id.write(f'BLER_coded_ortho_hard = {BLER_coded_ortho_hard}\n')
    file_id.write(f'BLER_learned_uncoded = {BLER_learned_coded}\n')
    file_id.write(f'----------------------------------------------------\n')
    file_id.write(f'Total time of execution = {(time.time()-t0)/60} mins\n')
