# Dependencies : 
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import sys
plt.rcParams['font.size'] = 12
plt.figure(dpi=500)

print(tf.__version__)
tf.config.list_physical_devices('GPU')

class RestartTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, threshold=0.6):
        super(RestartTrainingCallback, self).__init__()
        self.patience = patience
        self.threshold = threshold
        self.count = 0
        self.initial_weights = None

    def on_train_begin(self, logs=None):
        # Save the initial weights at the beginning of training
        self.initial_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        
        # Check if the accuracy is stuck at or below the threshold
        if accuracy is not None and accuracy <= self.threshold:
            self.count += 1
        else:
            self.count = 0

        # If the condition is met for 'patience' epochs, reset the model
        if self.count >= self.patience:
            print(f"\nAccuracy stuck at {accuracy * 100:.2f}% for {self.patience} epochs. Restarting training...")
            self.model.set_weights(self.initial_weights)
            self.count = 0

def branch_diversity_routine(k, n, SNR_TRAIN, channelDist='rayleigh'):
    # System Parameters
    # k                         # Number of bits required
    # n                         # Size of coded vector 
    E_b = 1                     # Energy per coded bit
    M = 2 ** k                  # Size of alphabet
    R = k / n                   # Information/ Communication rate 
    # SNR_TRAIN = float(input("Enter Training SNR (in dB) : "))

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

    # Generate random channel filter samples for training
    fade_mean = 0 
    fade_std = np.sqrt(0.5)
    if channelDist == 'rayleigh':
        fade_taps_I = np.random.normal(fade_mean, fade_std, (training_set_size, n)) 
        fade_taps_Q = np.random.normal(fade_mean, fade_std, (training_set_size, n))
    elif channelDist == 'gamma':
        mean_gamma = 0.75
        var_gamma = 0.5
        k_param = ((mean_gamma) ** 2) / var_gamma
        theta_param = mean_gamma / k_param
        fade_taps_I = np.random.gamma(k_param, theta_param, (training_set_size, n))
        fade_taps_Q = np.random.gamma(k_param, theta_param, (training_set_size, n))
    elif channelDist == 'gumbel':
        mean_gumbel = 0 
        var_gumbel = 0.5
        beta_param = np.sqrt(var_gumbel * 6 / (np.pi ** 2))
        mu_param = mean_gumbel - (beta_param * np.euler_gamma)
        fade_taps_I = np.random.gumbel(mu_param, beta_param, (training_set_size, n))
        fade_taps_Q = np.random.gumbel(mu_param, beta_param, (training_set_size, n))
    elif channelDist == 'custom':
        mean_custom = 0 
        var_custom = 0.5
        lambda_param = np.sqrt(2 / var_custom) 
        fade_taps_I = np.random.exponential(1 / lambda_param, (training_set_size, n)) -  np.random.exponential(1 / lambda_param, (training_set_size, n))
        fade_taps_Q = np.random.exponential(1 / lambda_param, (training_set_size, n)) -  np.random.exponential(1 / lambda_param, (training_set_size, n))
    else:
        raise ValueError('Invalid Channel Distribution. Available channel distributions are \n 1. Rayleigh\n 2. Gamma Distribution\n 3. Gumbel Distribution\n 4. Custom')
        

    ##################################### End to End Autoencoder ###########################################
    # Encoder(transmitter) part 
    enc_input_layer = tf.keras.Input(shape=(M,), name="Input_Layer")
    enc_layer_01 = tf.keras.layers.Dense(M, activation='relu', name="Encoder_Hidden_01")(enc_input_layer)
    enc_layer_02 = tf.keras.layers.Dense(n, activation='linear', name="Encoder_Hidden_02")(enc_layer_01)
    enc_layer_normalized = tf.keras.layers.Lambda((lambda x: np.sqrt(E_b * n) * tf.keras.backend.l2_normalize(x, axis=-1)), name='Power_Constraint')(enc_layer_02)

    # Fading 
    fading_layer_real = tf.keras.Input(shape=(n, ), name='fading_real_part')
    fading_layer_imag = tf.keras.Input(shape=(n,), name='fading_taps_imag')
    rx_signal_real = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_real])
    rx_signal_imag = tf.keras.layers.Multiply()([enc_layer_normalized, fading_layer_imag])

    # Noise 
    SNR_lin = 10 ** (SNR_TRAIN / 10)
    rx_noisy_signal_real = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_I')(rx_signal_real)
    rx_noisy_signal_imag = tf.keras.layers.GaussianNoise(stddev=np.sqrt(1 / (2 * R * SNR_lin)), name='AWGN_channel_Q')(rx_signal_imag)

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

    # Callback incase of ""beaching occurrence""
    restart_callback = RestartTrainingCallback(patience=5, threshold=0.5)

    # Fitting the model by using the training set :
    history = autoencoder.fit([x_train, fade_taps_I, fade_taps_Q], y_train, batch_size=1000, epochs=250, callbacks=[restart_callback])

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
    autoencoder.save(f'models/modelSISOnoCSI_uncoded_{k}_{n}_{SNR_TRAIN}_{channelDist}.keras')
    encoder_model.save(f'models/encoder_model_uncoded_{k}_{n}_{SNR_TRAIN}_{channelDist}.keras')
    decoder_model.save(f'models/decoder_model_uncoded_{k}_{n}_{SNR_TRAIN}_{channelDist}.keras')

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
    noise_var = 1 / (2 * R * SNR_lin) 

    # Encoding : 
    encoded_signal = encoder_model.predict(x_test)

    # Signal energy verification :
    with open(f'numerical_results/results_SISOnoCSI_uncoded_{k}_{n}_{SNR_TRAIN}_{channelDist}.txt', mode='w') as file_id:
        file_id.write(f'Number of bits = {testing_set_size * k}\n')
        file_id.write(f'Energy of the entire signal = {np.linalg.norm(encoded_signal) ** 2}\n')
        file_id.write(f'------------------------------------------\n')
    
    # Fading  
    fade_mean = 0 
    fade_std = np.sqrt(0.5)
    if channelDist == 'rayleigh':
        fade_taps_I = np.random.normal(fade_mean, fade_std, (testing_set_size, n)) 
        fade_taps_Q = np.random.normal(fade_mean, fade_std, (testing_set_size, n))
    elif channelDist == 'gamma':
        mean_gamma = 0.75
        var_gamma = 0.5
        k_param = ((mean_gamma) ** 2) / var_gamma
        theta_param = mean_gamma / k_param
        fade_taps_I = np.random.gamma(k_param, theta_param, (testing_set_size, n))
        fade_taps_Q = np.random.gamma(k_param, theta_param, (testing_set_size, n))
    elif channelDist == 'gumbel':
        mean_gumbel = 0 
        var_gumbel = 0.5
        beta_param = np.sqrt(var_gumbel * 6 / (np.pi ** 2))
        mu_param = mean_gumbel - (beta_param * np.euler_gamma)
        fade_taps_I = np.random.gumbel(mu_param, beta_param, (testing_set_size, n))
        fade_taps_Q = np.random.gumbel(mu_param, beta_param, (testing_set_size, n))
    elif channelDist == 'custom':
        mean_custom = 0 
        var_custom = 0.5
        lambda_param = np.sqrt(2 / var_custom) 
        fade_taps_I = np.random.exponential(1 / lambda_param, (testing_set_size, n)) -  np.random.exponential(1 / lambda_param, (testing_set_size, n))
        fade_taps_I = np.random.exponential(1 / lambda_param, (testing_set_size, n)) -  np.random.exponential(1 / lambda_param, (testing_set_size, n))

    fade_taps = fade_taps_I + 1j * fade_taps_Q
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

    # Writing all the results to a file
    fig.savefig(f'figures/ber_SISO_noCSI_uncoded_{k}_{n}_{SNR_TRAIN}_{channelDist}.png')
    with open(f'numerical_results/results_SISOnoCSI_uncoded_{k}_{n}_{SNR_TRAIN}_{channelDist}.txt', mode='a') as file_id:
        file_id.write(f'BLER_uncoded_ortho = {BLER_uncoded_ortho}\n')
        file_id.write(f'BLER_coded_ortho_hard = {BLER_coded_ortho_hard}\n')
        file_id.write(f'BLER_learned_uncoded_{k}_{n}_{SNR_TRAIN}_{channelDist} = {BLER_learned_uncoded}\n')
        file_id.write(f'----------------------------------------------------\n')
    
    # Mappings learnt:
    all_possible_inputs = np.eye(M)
    codebook = encoder_model.predict(all_possible_inputs)
    with open(f'numerical_results/encodings_{k}_{n}_{SNR_TRAIN}_{channelDist}.txt', mode='w') as file_id:
            file_id.write(f'codebook_{k}_{n}_{channelDist} = {codebook}\n')