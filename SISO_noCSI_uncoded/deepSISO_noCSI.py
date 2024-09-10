# Dependencies : 
import tensorflow as tf
import numpy as np 

import time 
start = time.time()

import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 12
plt.figure(dpi=500)

print(tf.__version__)
tf.config.list_physical_devices('GPU')

# System Parameters
k = 1                       # Number of bits required
M = 2                       # Size of alphabet
n = 2                       # Size of coded vector 
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
fade_taps_I = np.random.normal(fade_mean, fade_std, (training_set_size, n)) 
fade_taps_Q = np.random.normal(fade_mean, fade_std, (training_set_size, n))

##################################### End to End Autoencoder ###########################################
# Encoder(transmitter) part 
enc_input_layer = tf.keras.Input(shape=(M,), name="Input_Layer")
enc_layer_01 = tf.keras.layers.Dense(M, activation='relu', name="Encoder_Hidden_01")(enc_input_layer)
enc_layer_02 = tf.keras.layers.Dense(n, activation='linear', name="Encoder_Hidden_02")(enc_layer_01)
enc_layer_normalized = tf.keras.layers.Lambda((lambda x: np.sqrt(n / 2) * tf.keras.backend.l2_normalize(x, axis=-1)), name='Power_Constraint')(enc_layer_02)

# Fading 
fading_layer_real = tf.keras.Input(shape=(n, ), name='fading_real_part')
fading_layer_imag = tf.keras.Input(shape=(n,), name='fading_taps_imag')
rx_signal_real = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_real])
rx_signal_imag = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_imag])

# Noise 
SNR_lin = 10 ** (SNR_TRAIN / 10)
rx_noisy_signal_real = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * SNR_lin)), name='AWGN_channel_I')(rx_signal_real)
rx_noisy_signal_imag = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * SNR_lin)), name='AWGN_channel_Q')(rx_signal_imag)

# Decoder(Receiver) 
# Concatenating the real and imag noisy signal at the decoder part :
rx_signal = tf.keras.layers.Concatenate()([rx_noisy_signal_real, rx_noisy_signal_imag])
dec_layer_01 = tf.keras.layers.Dense(8*M, activation='relu', name='Decoder_Hidden_01')(rx_signal)
dec_layer_02 = tf.keras.layers.Dense(4*M, activation='relu', name='Decoder_Hidden_02')(dec_layer_01)
dec_layer_03 = tf.keras.layers.Dense(2*M, activation='relu', name='Decoder_Hidden_03')(dec_layer_02)
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
autoencoder.fit([x_train, fade_taps_I, fade_taps_Q], y_train, batch_size=1000, epochs=250)

############################### Analysing the Encoder and Decoder individually ##########################
# Abstracting out the decoder :
encoder_model = tf.keras.Model(enc_input_layer, enc_layer_normalized)

# Codeword Received at the receiver :
rx_noisy_signal_I = tf.keras.Input(shape=(n,))
rx_noisy_signal_Q = tf.keras.Input(shape=(n,))
dec_out = autoencoder.layers[-6]([rx_noisy_signal_I, rx_noisy_signal_Q])
dec_out = autoencoder.layers[-5](dec_out)
dec_out = autoencoder.layers[-4](dec_out)
dec_out = autoencoder.layers[-3](dec_out)
dec_out = autoencoder.layers[-2](dec_out)
dec_out = autoencoder.layers[-1](dec_out)

# Abstracting out the decoder model :
decoder_model = tf.keras.Model([rx_noisy_signal_I, rx_noisy_signal_Q], dec_out)

# Save the models for reproduction 
autoencoder.save(f'modelSISOnoCSI_uncoded_{int(SNR_TRAIN)}.keras')
encoder_model.save(f'encoder_model_uncoded{int(SNR_TRAIN)}.keras')
decoder_model.save(f'decoder_model_uncoded{int(SNR_TRAIN)}.keras')

######################################## Testing for BLER performance ###################################
testing_set_size = 10 ** 6 
y_test = np.random.randint(0, M, testing_set_size)
x_test = np.zeros((testing_set_size, M))  
x_test[np.arange(testing_set_size), y_test] = 1  # One-hot encoding

# Range of Signal to Noise Ratio :
n_points = 20 
SNR_dB = np.linspace(-2, 20, n_points)
SNR_lin = 10 ** (SNR_dB / 10)
# Fixing energy per bit :
E_b = 1 
# Range of noise variance for I/ Q component only :
noise_var = 1 / (2 * SNR_lin) 

# Encoding : 
encoded_signal = encoder_model.predict(x_test)

# Signal energy verification :
with open(f'ber_SISOnoCSI_uncoded_{int(SNR_TRAIN)}.txt', mode='w') as file_id:
    file_id.write(f'Number of bits = {testing_set_size * k}\n')
    file_id.write(f'Energy of the entire signal = {np.linalg.norm(encoded_signal) ** 2}\n')
    file_id.write(f'------------------------------------------\n')

# Fading 
fade_taps = np.random.normal(fade_mean, fade_std, encoded_signal.shape) + 1j * np.random.normal(fade_mean, fade_std, encoded_signal.shape)
faded_signal = fade_taps * encoded_signal

BLER_learned_uncoded = []
count = 0 
for noise in noise_var:
    # Noise
    noise_samples = np.random.normal(0, np.sqrt(noise), encoded_signal.shape) + 1j * np.random.normal(0, np.sqrt(noise), encoded_signal.shape)
    noisy_signal = faded_signal + noise_samples
    # Decoding
    decoded_signal = decoder_model.predict([np.real(noisy_signal), np.imag(noisy_signal)])
    # Analysis
    estimated_vectors = np.argmax(decoded_signal, axis=-1)
    BLER_learned_uncoded.append(np.sum(estimated_vectors != y_test) / testing_set_size)
    count += 1 
    print(f'Progress : {100 * count // n_points} %')

# Comparing the performance
BLER_uncoded_ortho = [0.379885, 0.353048, 0.32461, 0.29466, 0.261365, 0.22714, 0.195263, 0.164991, 0.136231, 0.111861, 0.090276, 0.072357, 0.056944, 0.044984, 0.035383, 0.027273, 0.021171, 0.016492, 0.012801, 0.00973]
BLER_coded_ortho_hard = [0.441056, 0.423869, 0.405249, 0.381611, 0.351994, 0.318561, 0.279774, 0.237635, 0.193919, 0.151703, 0.113621, 0.08356, 0.058248, 0.039873, 0.025686, 0.016703, 0.010626, 0.006438, 0.003953, 0.002417]

fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_ortho, label='Orthogonal Uncoded', color='black', marker='o')
axes.semilogy(SNR_dB, BLER_coded_ortho_hard, label='Orthogonal Coded', color='blue', marker='s')
axes.semilogy(SNR_dB, BLER_learned_uncoded, label='Learned E2E', color='red', marker='>')

axes.set_xlabel(f'SNR (in dB)')
axes.set_ylabel(f'BLER')
axes.set_title(f'Uncoded SISO No CSI')
axes.legend()
axes.grid()
fig.savefig(f'ber_siso_noCSI_uncoded_{int(SNR_TRAIN)}.png')

########################################### Visualizing the mappings ################################### 
x0 = tf.Variable([[1, 0]])    
x1 = tf.Variable([[0, 1]])

# Predictions i.e., their mappings :
y0 = encoder_model.predict(x0)
y1 = encoder_model.predict(x1)

with open(f'constellation_mapping_{int(SNR_TRAIN)}.txt', mode='w') as file_id:
    file_id.write(f'BIT 0 = {y0}\n')
    file_id.write(f'BIT 1 = {y1}\n')

fig, axes = plt.subplots()
axes.scatter(y0[0][0], y0[0][1], label='Bit 0', color='black', marker='D')
axes.scatter(y1[0][0], y1[0][1], label='Bit 1', color='black', marker='D')
axes.set_xlabel('$X_0$')
axes.set_ylabel('$X_1$')
axes.set_title('Constellation Diagram')
axes.legend()
axes.grid()
fig.savefig(f'constellation_mapping_{int(SNR_TRAIN)}.png')
######################################## Decision Boundary Analysis
num_points = 10 ** 4
limit = 100  # Exploration Region Boundary

# Generate random points
complex_points = np.random.uniform(-limit, limit, num_points * 2) + 1j * np.random.uniform(-limit, limit, num_points * 2)
complex_points = (complex_points).reshape((-1, 2))
input_real = np.real(complex_points)
input_imag = np.imag(complex_points)

# Predict in batch
pred = decoder_model.predict([input_real, input_imag])

test_points = []
# Compare probabilities and decide scatter plot color
for i in range(num_points):
    t0 = complex_points[i][0]
    t1 = complex_points[i][1]
    
    if pred[i][0] > pred[i][1]:
        test_points.append([t0, t1, 0])
    else:
        test_points.append([t0, t1, 1])
# Assuming test_points is a numpy array or can be converted to one
test_points = np.array(test_points)

# Store this numpy array for datViz:
np.save('decision_reg_points.npy', test_points)

# Extract data from test_points for clarity
points0_real = test_points[:, 0].real
points0_imag = test_points[:, 0].imag
points1_real = test_points[:, 1].real
points1_imag = test_points[:, 1].imag
labels = np.array(test_points[:, 2], dtype=int)

# Create subplots with shared x-axis
fig, (axes0, axes1) = plt.subplots(2, sharex=True)

# Define colors based on condition
colors = np.where(labels == 0, 'goldenrod', 'navy')

# Scatter plot on both axes
axes0.scatter(points0_real, points0_imag, color=colors, alpha=0.5)
axes1.scatter(points1_real, points1_imag, color=colors, alpha=0.5)

axes0.set_title('$Y_0$')
axes0.set_ylabel('$\mathcal{Im}$')

axes1.set_title('$Y_1$')
axes1.set_xlabel('$\mathcal{Re}$')
axes1.set_ylabel('$\mathcal{Im}$')

fig.savefig(f'decision_region_{int(SNR_TRAIN)}.png')

################################ Absolute value decision
# Number of points
n_points = 10 ** 4
limit = 10000

# Generate random complex numbers in a batch
real_parts_0 = np.random.uniform(-limit, limit, n_points)
imag_parts_0 = np.random.uniform(-limit, limit, n_points)
real_parts_1 = np.random.uniform(-limit, limit, n_points)
imag_parts_1 = np.random.uniform(-limit, limit, n_points)

t0_batch = real_parts_0 + 1j * imag_parts_0
t1_batch = real_parts_1 + 1j * imag_parts_1

np.save('t0_batch.npy', t0_batch)
np.save('t1_batch.npy', t1_batch)

# Prepare inputs for prediction
real_parts = np.column_stack((np.real(t0_batch), np.real(t1_batch)))
imag_parts = np.column_stack((np.imag(t0_batch), np.imag(t1_batch)))

# Predictions in batch
pred_batch = decoder_model.predict([real_parts, imag_parts])

# Scatter plot with different colors based on predictions
colors = ['purple' if pred[0] > pred[1] else 'tomato' for pred in pred_batch]
np.save('colors.npy', colors)

fig, axes = plt.subplots()
axes.scatter(np.abs(t0_batch), np.abs(t1_batch), c=colors, alpha=0.2)
axes.plot(np.linspace(0, limit * np.sqrt(2)), np.linspace(0, limit * np.sqrt(2)), c='black', ls='--')
axes.set_xlabel('$|Y_0|$')
axes.set_ylabel('$|Y_1|$')
axes.set_title('Learned Classification')
fig.savefig(f'absolute_learned_{int(SNR_TRAIN)}.png')

# # Theoretical Decision boundary
# # Plot the absolute values
# colors = ['tomato' if np.abs(t0_batch[i]) > np.abs(t1_batch[i]) else 'purple' for i in range(10**4)]
# fig, axes = plt.subplots()
# axes.scatter(np.abs(t0_batch), np.abs(t1_batch), c=colors, alpha=0.2)
# axes.plot(np.linspace(0, limit * np.sqrt(2)), np.linspace(0, limit * np.sqrt(2)), c='black', ls='--')
# axes.set_xlabel('$|Y_0|$')
# axes.set_ylabel('$|Y_1|$')
# axes.set_title('Conventional Classification')
# fig.savefig(f'absolute_conventional_{int(SNR_TRAIN)}.png')

# Writing all the results to a file
fig.savefig(f'ber_SISO_noCSI_uncoded_{int(SNR_TRAIN)}.png')
with open(f'results_SISOnoCSI_uncoded_{int(SNR_TRAIN)}.txt', mode='a') as file_id:
    file_id.write(f'BLER_uncoded_ortho = {BLER_uncoded_ortho}\n')
    file_id.write(f'BLER_coded_ortho_hard = {BLER_coded_ortho_hard}\n')
    file_id.write(f'BLER_learned_uncoded = {BLER_learned_uncoded}\n')
    file_id.write(f'----------------------------------------------------\n')
    file_id.write(f'Total time of execution = {(time.time()-start)/60} mins\n')
