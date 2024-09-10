# Dependencies :
import numpy as np 
import matplotlib.pyplot as plt 
import time
t0 = time.time()
plt.rcParams['font.size'] = 11
plt.figure(dpi=500)

# Function Definitions 
# Importing from custom made module : CommSysLib
import CommSysLib as csl

# System Parameters
k, n = 4, 7     # Uncoded and Coded block lengths 
R = k / n       # Information Rate
E_b = 1         # Energy per bit

# SNR range in dB and linear scale :
n_points = 20
SNR_dB = np.linspace(-2, 20, n_points)
SNR_lin = 10 ** (SNR_dB / 10)

# Variance of only real(imag) component (I(Q)) noise for a particular SNR
noise_var_uncoded = 1 / (2 * SNR_lin)
noise_var_coded = 1 / (2 * R * SNR_lin)
# Fading model (Rayleigh) parameters : Only real or imag component
fade_mean = 0 
fade_std = np.sqrt(0.5)

# Data Generation (Random Binary data)
num_blocks = 10 ** 6                          
num_bits = k * num_blocks
num_bits_c = num_bits * n // k 
binary_stream_tx = np.random.randint(0, 2, num_bits)

print(f'Samples of binary stream: {binary_stream_tx[:10]}')

####################################### Without channel coding 
# Constellation Mapping
signal_stream_tx = csl.BPSK_mapper(binary_stream_tx, E_b)

print(f'Samples of BPSK symbols stream: {signal_stream_tx[:10]}')

with open('results_tradSISOcsir.txt', mode='w') as file_id:
    file_id.write(f'Number of bits: {num_bits_c}\n')
    file_id.write(f'Energy of the uncoded stream: {np.linalg.norm(signal_stream_tx) ** 2}\n')
    file_id.write(f'------------------\n')

# Simulating the channel and the decoding 
BLER_uncoded_coherent = []
count = 0 
for noise in noise_var_uncoded:
    # Fading 
    fade_taps = np.random.normal(fade_mean, fade_std, signal_stream_tx.shape) + 1j * np.random.normal(fade_mean, fade_std, signal_stream_tx.shape)
    signal_stream_rx = fade_taps * signal_stream_tx
    # Noise 
    noise_samples = np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape)
    signal_stream_rx = signal_stream_rx + noise_samples
    # Coherent Combining 
    rectified_stream_rx = np.conjugate(fade_taps) * signal_stream_rx / np.abs(fade_taps)
    # Demapping
    binary_stream_rx = csl.BPSK_demapper(rectified_stream_rx)
    # Analysis
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, 4)
    BLER_uncoded_coherent.append(BLER)
    # Progress Update 
    count += 1 
    print(f'Progress : {100 * count // n_points} %', end='\r')

#################################### With Channel Coding 
# Channel coding
# Generator matrix for (7, 4) Hamming Code :
G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)
# Parity Check Matrix for (7, 4) Hamming Code :
H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
    ], dtype=int)

channel_coded_stream_tx = csl.hamming_encoder(binary_stream_tx, G)
print(f'Samples of the channel coded bits = {channel_coded_stream_tx[:10]}')

# Constellation Mapping
signal_stream_tx = csl.BPSK_mapper(channel_coded_stream_tx, E_b)
print(f'Samples of the BPSK symbol stream = {signal_stream_tx[:10]}')

with open('results_tradSISOcsir.txt', mode='a') as file_id:
    file_id.write(f'Number of bits: {num_bits_c}\n')
    file_id.write(f'Energy of the coded stream: {np.linalg.norm(signal_stream_tx) ** 2}\n')
    file_id.write(f'------------------\n')

# Simulating the channel and the receiver
################### 1. Hard Decision + Syndrome-based Correction
BLER_coded_coherent_hard = []
count = 0 
for noise in noise_var_coded:
    # Fading 
    fade_taps = np.random.normal(fade_mean, fade_std, signal_stream_tx.shape) + 1j * np.random.normal(fade_mean, fade_std, signal_stream_tx.shape)
    signal_stream_rx = fade_taps * signal_stream_tx
    # Noise 
    noise_samples = np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape)
    signal_stream_rx = signal_stream_rx + noise_samples 
    # Coherent Combining 
    rectified_stream_rx = np.conjugate(fade_taps) * signal_stream_rx / np.absolute(fade_taps)
    # BPSK Demapping 
    channel_coded_stream_rx = csl.BPSK_demapper(rectified_stream_rx)
    # Syndrome Correction 
    binary_stream_rx = csl.hamming_decoder(channel_coded_stream_rx, H)
    # Analysis
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, k)
    BLER_coded_coherent_hard.append(BLER) 
    # Update the progress
    count += 1
    print(f'Progress : {100 * count // n_points} %', end='\r')

############################ 2. Maximum Likelihood Decoding 
BLER_coded_coherent_mld = []
count = 0 
for noise in noise_var_coded:
    # Fading 
    fade_taps = np.random.normal(fade_mean, fade_std, signal_stream_tx.shape) + 1j * np.random.normal(fade_mean, fade_std, signal_stream_tx.shape)
    signal_stream_rx = fade_taps * signal_stream_tx
    # Noise 
    noise_samples = np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape)
    signal_stream_rx = signal_stream_rx + noise_samples 
    # Coherent Combining 
    rectified_stream_rx = np.conjugate(fade_taps) * signal_stream_rx / np.absolute(fade_taps)
    binary_stream_rx = csl.ML_Detection(rectified_stream_rx, G, E_b, n, k)
    # Analysis
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, k)
    BLER_coded_coherent_mld.append(BLER) 
    # Update the progress
    count += 1
    print(f'Progress : {100 * count // n_points} %', end='\r')

# Analysis 
fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_coherent, label='Uncoded', color='black', marker='o')
axes.semilogy(SNR_dB, BLER_coded_coherent_hard, label='Hamming (7, 4) Hard', color='blue', marker='s')
axes.semilogy(SNR_dB, BLER_coded_coherent_mld, label='Hamming (7, 4) MLD', color='green', marker='*')
axes.set_xlabel('SNR (in dB)')
axes.set_ylabel('BLER')
axes.legend()
axes.set_title(f'Coherent Schemes')
axes.grid(True, which="both")

# Saving the results for future reference
fig.savefig('results_tradSISOcsir.png')
with open("results_tradSISOcsir.txt", mode='a') as file_id :
    file_id.write(f'BLER_uncoded_coherent = {BLER_uncoded_coherent} \n')
    file_id.write(f'BLER_coded_coherent_hard = {BLER_coded_coherent_hard} \n')
    file_id.write(f'BLER_coded_coherent_mld = {BLER_coded_coherent_mld} \n')
    file_id.write(f'------------------------------\n')
    file_id.write(f'Total execution time = {(time.time() - t0) / 60} mins')