# Dependencies :
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
plt.figure(dpi=500)

import tensorflow as tf 
print(tf.__version__)

# System Parameters
M = 2 ** 4        # Size of alphabet                     
k = np.log2(M)    # Number of bits required           
n = 7             # Block length of coded vector    
R = k / n         # Communication rate

######################################## Generation of training data
training_data_size = 10 ** 6 
sample_indices = np.random.randint(0, M, training_data_size)

# Set of One Hot Encoded Vectors :
x_train = []
for idx in sample_indices:
    temp = np.zeros(M)
    temp[idx] = 1
    x_train.append(temp)
x_train = tf.constant(x_train)

# Labels for the data :
# Since we want to reproduce the input at the output :
y_train = x_train

####################################### Creating the Auto-Encoder Model 
# Describing the encoder layers :
enc_input_layer = tf.keras.Input(shape=(M,), name='Input_Layer')
enc_layer_01 = tf.keras.layers.Dense(M, activation='relu', name='Encoder_Hidden_01')(enc_input_layer)
enc_layer_02 = tf.keras.layers.Dense(n, activation='linear', name='Encoder_Hidden_02')(enc_layer_01)
enc_layer_normalized = tf.keras.layers.Lambda((lambda x: np.sqrt(n) * tf.keras.backend.l2_normalize(x, axis=-1)))(enc_layer_02)

# Describing the channel layers :
# Training SNR
SNR_TRAIN = float(input('Enter training SNR in dB: '))        
SNR_lin = 10 ** (SNR_TRAIN / 10)                   
# AWGN Channel
ch_noise_layer = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel')(enc_layer_normalized)

# Describing the decoder layers :
dec_layer_01 =  tf.keras.layers.Dense(M, activation='relu', name='Decoder_Hidden_01')(ch_noise_layer)
dec_output_layer = tf.keras.layers.Dense(M, activation='softmax', name='Output_Layer')(dec_layer_01)

autoencoder = tf.keras.Model(enc_input_layer,dec_output_layer)

# Compiling the model :
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
autoencoder.summary()

# Uncomment to view the block diagram of the autoencoder :
# tf.keras.utils.plot_model(
#     autoencoder,
#     to_file='deepAWGN.png',
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=True,
#     dpi=200,
#     show_layer_activations=True,
#     show_trainable=True,
# )

# Fitting the model by using training set :
autoencoder.fit(x_train, y_train, batch_size=1000, epochs=250) 

# Testing the above learned system for various SNRs
# Encoder :
encoder_model = tf.keras.Model(enc_input_layer, enc_layer_normalized)

# Supposed received codeword at the receiver
encoded_input = tf.keras.Input(shape=(n,))
decoder_output = autoencoder.layers[-2](encoded_input)
decoder_output = autoencoder.layers[-1](decoder_output)

# Decoder :
decoder_model = tf.keras.Model(encoded_input, decoder_output)

# Saving the models for future :
autoencoder.save(f'awgn_autoencoder_{int(SNR_TRAIN)}.keras')
encoder_model.save(f'awgn_encoder_{int(SNR_TRAIN)}.keras')
decoder_model.save(f'awgn_decoder_{int(SNR_TRAIN)}.keras')

# Using this same model for performing manual coherent detection : 
# Here we take the output from encoder of the above learned system, then we simulate the fading and noise by ourselves. After that we do coherent detection. Then, the resultant signal is passed on to the decoder of the above learned system. 

######################################## Generation of validation data
test_data_size = 10 ** 6 # Number of Blocks 
y_test = np.random.randint(0, M, test_data_size)
x_test = []
for idx in y_test:
    temp = np.zeros(M)
    temp[idx] = 1
    x_test.append(temp)

x_test = tf.constant(x_test)

# Validation Routine
# Range of Signal to Noise Ratio a
# in dB :
n_points = 20
SNR_dB = np.linspace(-2, 20, n_points)
# in Linear Scale :
SNR_lin = 10 ** (SNR_dB / 10)
# Fixing energy per bit :
E_b = 1 
# Range of noise variance accordingly :
noise_var = 1 / (2 * R * SNR_lin) # This is the real noise only.

###################################### Fading Test Routine
# Encoding using our model :
encoded_signal = encoder_model.predict(x_test)
BLER_manual_coherent = []
count = 0 
for noise in noise_var :
    # Fast fading effect :
    fade_taps = np.random.normal(0, np.sqrt(0.5), encoded_signal.shape) + 1j* np.random.normal(0, np.sqrt(0.5), encoded_signal.shape)
    rx_faded_signal = fade_taps * encoded_signal
    # Generating AWGN samples :
    noise_samples = np.random.normal(0, np.sqrt(noise), rx_faded_signal.shape) + 1j* np.random.normal(0, np.sqrt(noise), rx_faded_signal.shape)
    rx_noisy_signal = rx_faded_signal + noise_samples 
    # Decoding using our model :
    decoded_signal = decoder_model.predict(np.real(np.conjugate(fade_taps) * rx_noisy_signal / np.absolute(fade_taps)))
    estimated_vectors = np.argmax(decoded_signal, axis=-1)
    BLER_manual_coherent.append(np.sum(estimated_vectors != y_test) / test_data_size)
    # Progress Update :
    count += 1
    print(f'Progress : {100 * count // len(noise_var)}')

# Comparison curves : These have been obtained by classical method simulations
BLER_uncoded_coherent = [0.568744, 0.510474, 0.453514, 0.393655, 0.337754, 0.283992, 0.234753, 0.192111, 0.155179, 0.124392, 0.098039, 0.077155, 0.059857, 0.046734, 0.036071, 0.028182, 0.021548, 0.016597, 0.012735, 0.009868] 
BLER_coded_coherent_hard = [0.536971, 0.469589, 0.400025, 0.329686, 0.262367, 0.201499, 0.148375, 0.106653, 0.073774, 0.049709, 0.032504, 0.020712, 0.013085, 0.008012, 0.004965, 0.003048, 0.00181, 0.001019, 0.000619, 0.000381] 
BLER_coded_coherent_mld = [0.464027, 0.389022, 0.312285, 0.240074, 0.174105, 0.119859, 0.078081, 0.047754, 0.027741, 0.015409, 0.008192, 0.004143, 0.002078, 0.001025, 0.000494, 0.000233, 0.000105, 4.3e-05, 2.8e-05, 9e-06] 

# Plotting :
fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_coherent, label="Uncoded", c='black')
axes.semilogy(SNR_dB, BLER_coded_coherent_hard, label="Hamming (7, 4) (Hard)", c="blue", ls="-.")
axes.semilogy(SNR_dB, BLER_coded_coherent_mld, label="Hamming (7, 4) (MLD)", c="blue", ls="--")
axes.semilogy(SNR_dB, BLER_manual_coherent, label="Learned", c='red', marker='o', ls=" ")
axes.set_xlabel(f'SNR (in dB)')
axes.set_ylabel(f'BLER')
axes.legend()
axes.grid()

# Saving the results for future reference
# Plots :
fig.savefig('awgnCodeComparison.png')
# Values :
with open("awgnCodeComparison.txt", mode='w') as file_id :
    file_id.write(f'Number of bits = {test_data_size * k}\n')
    file_id.write(f'Energy of the total signal = {np.linalg.norm(encoded_signal) ** 2}\n')
    file_id.write(f'-----------------------------------------------------------------')
    file_id.write(f'BLER_uncoded_coherent = {BLER_uncoded_coherent}\n')
    file_id.write(f'BLER_coded_coherent_hard = {BLER_coded_coherent_hard}\n')
    file_id.write(f'BLER_coded_coherent_mld = {BLER_coded_coherent_mld}\n')
    file_id.write(f'BLER_manual_coherent = {BLER_manual_coherent}\n')