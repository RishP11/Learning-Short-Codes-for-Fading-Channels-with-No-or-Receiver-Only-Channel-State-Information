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
k, n = 4, 7     # Uncoded and Coded block lengths
M = 2 ** k      # Size of the alphabet 
R = k / n       # Information rate
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
fade_taps = np.random.normal(fade_mean, fade_std, (training_set_size, 2*n)) + 1j*np.random.normal(fade_mean, fade_std, (training_set_size, 2*n))

######################################### Defining the Model ############################################ 
# 1. We will use it to emulate the superposition of the two signal streams that have been independenly faded and subsequently arrive at the receiver. How does this make sense ? 
# 2. We will generate a tensor of length 14 corresponding to a input tensor representing a block of 4 bits. These 14 symbols we will interpret as pair of 7 coded bits for ever antenna ; The first 7 elements of the output of the encoder are from the first antenna and remaining from the other antenna. 
# 3. What we want implement to simulate a flat fading channel, is that multiply the elements in the 14 dimensional tensor with the channel taps. After that, we will split the vector in the middle, thereby getting 2 halves corresponding to the two antennas. Following this we get a 7 dimensional tensor which can be peturbed by AWGN simulating the imparity at the receiver.
# Custom Layer Class definition :
class SuperPoseLayer(tf.keras.layers.Layer):
    """ 
    Custom tensorflow layer that takes in an input (parent) tensor, splits it into halves and then performs elementwise
    addition along the last axis.(child tensors) and that is the output.
    """
    def __init__(self):
        super(SuperPoseLayer, self).__init__()

    def call(self, inputs):
        # Get the shape of the input tensor
        input_shape = tf.shape(inputs)
        
        # Split the input tensor into two halves along the last dimension
        split_size = input_shape[-1] // 2
        split1 = inputs[..., :split_size]
        split2 = inputs[..., split_size:]
        
        # Add the elements of each half
        added_splits = tf.math.add(split1, split2)
        
        return added_splits
# Demo 
demo_layer = SuperPoseLayer()
demo_input = tf.Variable([1, 2, 3, 4, 5, 6])
print(f'Demo Output :{demo_layer(demo_input)}')

# Encoder(== transmitter) part :
enc_input_layer_bits = tf.keras.Input(shape=(M, ), name="Bits_Input_Layer")
fading_layer_real = tf.keras.Input(shape=(2*n, ), name='fading_taps_real')
fading_layer_imag = tf.keras.Input(shape=(2*n, ), name='fading_taps_imag')

enc_layer_01 = tf.keras.layers.Dense(M, activation='relu', name="Enc_Hidden_Layer_01")(enc_input_layer_bits)
enc_layer_02 = tf.keras.layers.Dense(2*n, activation='relu', name="Enc_Hidden_Layer_02")(enc_layer_01)
enc_layer_03 = tf.keras.layers.Dense(2*n, activation='linear', name="Enc_Hidden_Layer_03")(enc_layer_02)
enc_layer_normalized = tf.keras.layers.Lambda((lambda x: np.sqrt(n) * tf.keras.backend.l2_normalize(x, axis=-1)), name='Power_Constraint')(enc_layer_03)

# Channel :
# y = fx + w  model(flat + fast fading):
faded_signal_real = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_real])
faded_signal_imag = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_imag])

# Superposition of the signals of both the antennas at the receiver  :
rx_signal_real = SuperPoseLayer()(faded_signal_real)
rx_signal_imag = SuperPoseLayer()(faded_signal_imag)

# Noise :
SNR_lin = 10 ** (SNR_TRAIN / 10)
rx_noisy_signal_real = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_real')(rx_signal_real)
rx_noisy_signal_imag = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_imag')(rx_signal_imag)

# Decoder (==Receiver) part :
rx_signal = tf.keras.layers.Concatenate()([rx_noisy_signal_real, rx_noisy_signal_imag, fading_layer_real, fading_layer_imag])
dec_layer_01 = tf.keras.layers.Dense(4*n, activation='relu', name='Decoder_Hidden_01')(rx_signal)
dec_layer_02 = tf.keras.layers.Dense(2*n, activation='relu', name='Decoder_Hidden_02')(dec_layer_01)
dec_layer_03 = tf.keras.layers.Dense(4*M, activation='relu', name='Decoder_Hidden_03')(dec_layer_02)
dec_layer_04 = tf.keras.layers.Dense(2*M, activation='relu', name='Decoder_Hidden_04')(dec_layer_03)
dec_layer_05 = tf.keras.layers.Dense(1*M, activation='relu', name='Decoder_Hidden_05')(dec_layer_04)
dec_output_layer = tf.keras.layers.Dense(M, activation='softmax', name='Output_Layer')(dec_layer_05)

autoencoder = tf.keras.Model(
                    inputs = [enc_input_layer_bits, fading_layer_real, fading_layer_imag], 
                    outputs = [dec_output_layer],
                )

# Compiling the model :
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
autoencoder.summary()

# # Uncomment the following to view the block diagram of the neural network ; Requires GraphViz
# tf.keras.utils.plot_model(
#     autoencoder,
#     to_file='deepMISO.png',
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir='TB',
#     expand_nested=True,
#     dpi=200,
#     show_layer_activations=True,
#     show_trainable=True,
# )

autoencoder.fit([x_train, np.real(fade_taps), np.imag(fade_taps)], y_train, batch_size=1000, epochs=500)
del x_train
del fade_taps
del y_train
autoencoder.save('autoencoder.keras')
################################## Testing the above learned system #####################################
# Abstracting Encoder :
encoder_model = tf.keras.Model(enc_input_layer_bits, enc_layer_normalized)

# Abstracting Decoder :
rx_noisy_signal_I = tf.keras.Input(shape=(n,))
rx_noisy_signal_Q = tf.keras.Input(shape=(n,))
fade_I = tf.keras.Input(shape=(2*n,))
fade_Q = tf.keras.Input(shape=(2*n,))
decoder_output = autoencoder.layers[-7]([rx_noisy_signal_I, rx_noisy_signal_Q, fade_I, fade_Q])
decoder_output = autoencoder.layers[-6](decoder_output)
decoder_output = autoencoder.layers[-5](decoder_output)
decoder_output = autoencoder.layers[-4](decoder_output)
decoder_output = autoencoder.layers[-3](decoder_output)
decoder_output = autoencoder.layers[-2](decoder_output)
decoder_output = autoencoder.layers[-1](decoder_output)

decoder_model = tf.keras.Model([rx_noisy_signal_I, rx_noisy_signal_Q, fade_I, fade_Q], decoder_output)

# Generation of validation data :
test_data_size = 10 ** 6
y_test = np.random.randint(0, M, test_data_size)
x_test = []
for idx in y_test:
    temp = np.zeros(M)
    temp[idx] = 1
    x_test.append(temp)

x_test = tf.constant(x_test)
x_test

# Validation Routine
# Range of Signal to Noise Ratio :
SNR_dB = np.linspace(-2, 20, 20)
SNR_lin = 10 ** (SNR_dB / 10)
# Fixing energy per bit :
E_b = 1 
# Range of noise variance accordingly :
noise_var = 1 / (2 * R * SNR_lin) 

def split_and_add(array):
    # Check if the last dimension can be split into two equal parts
    if array.shape[-1] % 2 != 0:
        raise ValueError("Last dimension size must be even for splitting")

    # Split the last dimension into two halves
    half_size = array.shape[-1] // 2
    first_half = array[..., :half_size]
    second_half = array[..., half_size:]

    # Perform elementwise addition along the last dimension
    result = np.add(first_half, second_half)

    return result

# Encoding using our model :
BLER_learned_alamouti = []
encoded_signal = encoder_model.predict(x_test)
with open(f'results_deepMISO_{int(SNR_TRAIN)}.txt', mode='w') as file_id:
    file_id.write(f'Number of bits = {k * test_data_size}\n')
    file_id.write(f'Energy of the signal = {np.linalg.norm(encoded_signal) ** 2}\n')
    file_id.write(f'-------------------------------------------------------------------\n')
    
count = 0 
for noise in noise_var :  
    # Fading :
    fade_taps = np.random.normal(fade_mean, fade_std, encoded_signal.shape) + 1j * np.random.normal(fade_mean, fade_std, encoded_signal.shape)
    # Fast fading effect :
    rx_signal = fade_taps * encoded_signal
    # Superposition : 
    rx_signal = split_and_add(rx_signal)
    noise_samples = np.random.normal(0, np.sqrt(noise), rx_signal.shape) + 1j * np.random.normal(0, np.sqrt(noise), rx_signal.shape)
    rx_signal += noise_samples 
    # Decoding using our model :
    decoded_signal = decoder_model.predict([np.real(rx_signal), np.imag(rx_signal), np.real(fade_taps), np.imag(fade_taps)])
    estimated_vectors = np.argmax(decoded_signal, axis=-1)
    BLER_learned_alamouti.append(np.sum(estimated_vectors != y_test) / test_data_size)
    # Progress Update :
    count += 1
    print(f'Progress : {100 * count // 20} %')

# Performance Benchmarks 
BLER_uncoded_alamouti = [0.3155876, 0.248934, 0.1894568, 0.139134, 0.098632, 0.0679476, 0.0453796, 0.0295844, 0.0187456, 0.0118964, 0.0073864, 0.0045564, 0.0027136, 0.0016224, 0.0010068, 0.000586, 0.0003516, 0.0002012, 0.000122, 5.8e-05]
BLER_coded_alamouti_hard = [0.2789248, 0.2074372, 0.1450088, 0.0959216, 0.0596248, 0.0352576, 0.0199676, 0.011044, 0.0059796, 0.0032528, 0.0017732, 0.0009824, 0.000552, 0.0003392, 0.0001864, 0.0001096, 6.04e-05, 4.4e-05, 2.48e-05, 1.08e-05]
BLER_coded_alamouti_mld = [0.1977316, 0.1332216, 0.0828972, 0.0471944, 0.0248644, 0.0122352, 0.0055452, 0.0023956, 0.0009988, 0.0003956, 0.0001424, 5.88e-05, 2.52e-05, 8e-06, 2.8e-06, 8e-07, 8e-07, 4e-07, 0.0, 0.0]

fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_alamouti, label='Uncoded', color='black', marker='o')
axes.semilogy(SNR_dB, BLER_coded_alamouti_hard, label='Hamming (7, 4) Hard', color='blue', marker='s')
axes.semilogy(SNR_dB, BLER_coded_alamouti_mld, label='Hamming (7, 4) MLD', color='green', marker='D')
axes.semilogy(SNR_dB, BLER_learned_alamouti, label='Learned', color='red', marker="x")

axes.set_xlabel('SNR (in dB)')
axes.set_ylabel('BLER')
axes.set_title('2 x 1 MISO Alamouti Scheme')
axes.legend()
axes.grid()

# Recording the results for future use :
# Plots
fig.savefig(f'results_deepMISO_{int(SNR_TRAIN)}.png')
# BLER values :
with open(f'results_deepMISO_{int(SNR_TRAIN)}.txt', mode='a') as file_id:
    file_id.write(f'BLER_uncoded_alamouti = {BLER_uncoded_alamouti}\n')
    file_id.write(f'BLER_coded_alamouti_hard = {BLER_coded_alamouti_hard}\n')
    file_id.write(f'BLER_coded_alamouti_mld = {BLER_coded_alamouti_mld}\n')
    file_id.write(f'BLER_learned_alamouti = {BLER_learned_alamouti}\n')
    file_id.write(f'---------------------------------------------------------------\n')
    file_id.write(f'Total Execution time = {(time.time() - start_time) / 60} mins')