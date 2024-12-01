# Dependencies : 
import tensorflow as tf
import numpy as np 

import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 12
plt.figure(dpi=500)

print(tf.__version__)
tf.config.list_physical_devices('GPU')

# System Specifications/ Parameters/ Definitions 
k = 4           # Uncoded block length       
n = 7           # Codeword length 
M = 2 ** 4      # Size of the alphabet 
R = k / n       # Information rate 
SNR_TRAIN = float(input('Enter the training SNR (in dB): '))

# Generating the training data 
training_set_size = 10 ** 6
samples_indices = np.random.randint(0, M, training_set_size)

# Converting the indices to 1-hot vectors
x_train = np.zeros((training_set_size, M))
x_train[np.arange(training_set_size), samples_indices] = 1

# We wish to reconstruct the input at the output
y_train = x_train

# Generate random fading taps for training == CSI @ Tx + Rx
fade_mean = 0 
fade_std = np.sqrt(0.5)
fade_taps_real = np.random.normal(fade_mean, fade_std, (training_set_size, n))
fade_taps_imag = np.random.normal(fade_mean, fade_std, (training_set_size, n))

##################################### Creating the Autoencoder #########################################
# Encoder Layers :: (Transmitter)
enc_bits_input_layer = tf.keras.Input(shape=(M, ), name="Bits_Input_Layer")
enc_csi_input_I = tf.keras.Input(shape=(n, ), name='Fading_Input_Layer_real')
enc_csi_input_Q = tf.keras.Input(shape=(n, ), name='Fading_Input_Layer_imag')
enc_combined_input = tf.keras.layers.Concatenate()([enc_bits_input_layer, enc_csi_input_I, enc_csi_input_Q])

enc_layer_01 = tf.keras.layers.Dense(M + 2*n, activation='relu', name="Encoder_Hidden_01")(enc_combined_input)
enc_layer_02 = tf.keras.layers.Dense(n, activation='relu', name="Encoder_Hidden_02")(enc_layer_01)
enc_layer_03 = tf.keras.layers.Dense(n, activation='linear', name="Encoder_Hidden_03")(enc_layer_02)
enc_layer_normalized = tf.keras.layers.Lambda((lambda x : np.sqrt(n) * tf.keras.backend.l2_normalize(x, axis=-1)), name='Power_Constraint')(enc_layer_03)

# Channel Layers :: Rayleigh Fading + AWGN 
faded_signal_I = tf.keras.layers.Multiply()([enc_csi_input_I, enc_layer_normalized])
faded_signal_Q = tf.keras.layers.Multiply()([enc_csi_input_Q, enc_layer_normalized])
# Adding the noise 
SNR_lin = 10 ** (SNR_TRAIN / 10)
noisy_signal_I = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_I')(faded_signal_I)
noisy_signal_Q = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_Q')(faded_signal_Q)

# Decoder Layers :: (Receiver)
rx_signal = tf.keras.layers.Concatenate()([noisy_signal_I, noisy_signal_Q, enc_csi_input_I, enc_csi_input_Q])
dec_layer_01 = tf.keras.layers.Dense(4*M, activation='relu', name="Decoder_layer_01")(rx_signal)
dec_layer_02 = tf.keras.layers.Dense(2*M, activation='relu', name="Decoder_layer_02")(dec_layer_01)
dec_output_layer = tf.keras.layers.Dense(M, activation='softmax', name="Output_layer")(dec_layer_02)

autoencoder = tf.keras.Model(
                    inputs = [enc_bits_input_layer, enc_csi_input_I, enc_csi_input_Q],
                    outputs = [dec_output_layer]
                )

# Compiling the model :
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
autoencoder.summary()

# # Uncomment this code to save the block diagram of the above neural network ; GraphViz is required
# tf.keras.utils.plot_model(
#     autoencoder,
#     to_file='deepSISOCSIT.png',
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
autoencoder.fit([x_train, fade_taps_real, fade_taps_imag], y_train, batch_size=1000, epochs=250)

##################### Testing the above trained autoencoder at varying SNR levels #######################

# Generating the testing/ validation data
testing_data_size = 10 ** 6
y_test = np.random.randint(0, M, testing_data_size)
x_test = np.zeros((testing_data_size, M))  
x_test[np.arange(testing_data_size), y_test] = 1  # One-hot encoding

# Abstracting the encoder :
encoder_model = tf.keras.Model([enc_bits_input_layer, enc_csi_input_I, enc_csi_input_Q], enc_layer_normalized)

# Abstracting the decoder :
signal_at_rx_I = tf.keras.Input(shape=(n,))
signal_at_rx_Q = tf.keras.Input(shape=(n,))
csi_at_rx_I = tf.keras.Input(shape=(n,))
csi_at_rx_Q = tf.keras.Input(shape=(n,))
decoder_output = autoencoder.layers[-4]([signal_at_rx_I, signal_at_rx_Q, csi_at_rx_I, csi_at_rx_Q])
decoder_output = autoencoder.layers[-3](decoder_output)
decoder_output = autoencoder.layers[-2](decoder_output)
decoder_output = autoencoder.layers[-1](decoder_output)
decoder_model = tf.keras.Model([signal_at_rx_I, signal_at_rx_Q, csi_at_rx_I, csi_at_rx_Q], decoder_output)

###### Validation Routine
# Range of Signal to Noise Ratio :
SNR_dB = np.linspace(-2, 20, 30)
SNR_lin = 10 ** (SNR_dB / 10)
# Fixing energy per bit :
E_b = 1 
# Range of noise variance accordingly :
noise_var = 1 / (2 * R * SNR_lin)

count = 0 
BLER_learned_coded_csit = []
for noise in noise_var:
    # CSI @ tx
    fade_taps = np.random.normal(fade_mean, fade_std, (testing_data_size, n)) + 1j * np.random.normal(fade_mean, fade_std, (testing_data_size, n))
    # Encoding
    encoded_signal = encoder_model.predict([x_test, np.real(fade_taps), np.imag(fade_taps)])
    # Fading 
    faded_signal = fade_taps * encoded_signal
    # Noise
    noise_samples = np.random.normal(0, np.sqrt(noise), (testing_data_size, n)) + 1j * np.random.normal(0, np.sqrt(noise), (testing_data_size, n))
    noisy_signal = faded_signal + noise_samples
    # Decoding
    decoded_signal = decoder_model.predict([np.real(noisy_signal), np.imag(noisy_signal), np.real(fade_taps), np.imag(fade_taps)])
    estimated_vectors = np.argmax(decoded_signal, axis=-1)
    BLER_learned_coded_csit.append(np.sum(estimated_vectors != y_test) / testing_data_size)
    # Progress 
    count += 1 
    print(f'Progress : {100 * count // len(noise_var)} %')

fig, axes = plt.subplots()
BLER_uncoded_precombining = [0.5684428, 0.5311236, 0.493762, 0.4558404, 0.4171428, 0.378762, 0.3408908, 0.3059336, 0.271428, 0.2400552, 0.210732, 0.183804, 0.1598504, 0.1384896, 0.11907, 0.1020932, 0.0874284, 0.0742084, 0.0635188, 0.0540476, 0.0457924, 0.0387388, 0.0325744, 0.0276616, 0.023314, 0.0196752, 0.0166252, 0.0139868, 0.011788, 0.009902] 
BLER_coded_precombining_hard = [0.5365056, 0.4934964, 0.4483044, 0.4026564, 0.35597, 0.310084, 0.2662452, 0.2260272, 0.1878712, 0.1536716, 0.1244648, 0.098852, 0.0779424, 0.059974, 0.0460384, 0.0352164, 0.026156, 0.0195808, 0.0144304, 0.0106992, 0.007686, 0.0055644, 0.0040268, 0.0028976, 0.0020868, 0.0015252, 0.001048, 0.00076, 0.0005628, 0.0003944] 
BLER_coded_precombining_mld = [0.464462, 0.4150208, 0.3656132, 0.3149372, 0.2664712, 0.2206332, 0.1786068, 0.1408384, 0.1090332, 0.081956, 0.060166, 0.0430664, 0.0300664, 0.0207288, 0.0139748, 0.0091792, 0.0060196, 0.003918, 0.0024428, 0.0015752, 0.0009784, 0.000578, 0.0003552, 0.0002084, 0.0001328, 7.08e-05, 4.32e-05, 2.92e-05, 1.32e-05, 5.2e-06] 
axes.semilogy(SNR_dB, BLER_learned_coded_csit, label='Learned', color='red', marker="o", ls=" ")
axes.semilogy(SNR_dB, BLER_uncoded_precombining, label='Uncoded BPSK', color='black')
axes.semilogy(SNR_dB, BLER_coded_precombining_hard, label='Hamming (7, 4) Hard', color='blue')
axes.semilogy(SNR_dB, BLER_coded_precombining_mld, label='Hamming (7, 4) MLD', color='blue', ls="-.")
axes.set_xlabel(r'$SNR\ (dB)$')
axes.set_ylabel(r'$BLER$')
axes.set_ylim(10**-5, 10**0)
axes.set_title(f'SISO CSIT')
axes.legend()
axes.grid(which='both')

# Saving the results for future reference :
# Plot : 
fig.savefig(f'deepSISOCSIT_{int(SNR_TRAIN)}.svg', transparent=True)
fig.savefig(f'deepSISOCSIT_{int(SNR_TRAIN)}.png')
# BLER points :
with open(file=f'deepSISOCSIT_{int(SNR_TRAIN)}.txt', mode='w') as file_id:
    file_id.write(f'BLER_uncoded_precombining = {BLER_uncoded_precombining}\n')
    file_id.write(f'BLER_coded_precombining_hard = {BLER_coded_precombining_hard}\n')
    file_id.write(f'BLER_coded_precombining_mld = {BLER_coded_precombining_mld}\n')
    file_id.write(f'BLER_learned_coded_csit = {BLER_learned_coded_csit}\n')