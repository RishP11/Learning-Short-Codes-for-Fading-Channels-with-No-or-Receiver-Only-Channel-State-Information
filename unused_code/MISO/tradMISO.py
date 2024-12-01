# Dependencies :
import numpy as np 
import matplotlib.pyplot as plt
plt.figure(dpi=500)

import time 
start_time = time.time()

# Function Defintions
# Importing from the custom library CommSysLib :
import CommSysLib as csl

# System Parameters
k, n = 4, 7         # Uncoded and Coded block lengths 
R = k / n           # Information Rate 
E_b = 1             # Energy per bit per antenna per symbol duration (BPSK)

# SNR range in dB and linear scale :
n_points = 20 
SNR_dB = np.linspace(-2, 20, n_points)
SNR_lin = 10 ** (SNR_dB / 10)

# Variance of only real(imag) component (I(Q)) noise for a particular SNR
noise_var_coded = 1 / (2 * R * SNR_lin)
noise_var_uncoded = 1 / (2 * SNR_lin)
# Fading model (Rayleigh) parameters : Only real or imag component
fade_mean = 0 
fade_std = np.sqrt(0.5)

# Data Generation (Random Binary Data)
n_bits = 10 ** 7
n_bits_c = n * n_bits // k

binary_stream_tx = np.random.randint(0, 2, n_bits)
print(f'Samples : {binary_stream_tx[:10]}')

######################################## Without Coding
# Alamouti encoding 
signal_stream_tx = csl.AlamoutiEncoder(binary_stream_tx, E_b)
print(f'Antenna 01 : {signal_stream_tx[0][:20]}')
print(f'Antenna 02 : {signal_stream_tx[1][:20]}')
with open(f'results_tradMISO.txt', mode='w') as file_id:
    file_id.write(f'Number of bits = {n_bits}\n')
    file_id.write(f'Energy spent at the transmitter = {np.linalg.norm(signal_stream_tx[0])**2 + np.linalg.norm(signal_stream_tx[1]) ** 2}\n')
    file_id.write(f'---------------------------------------------------------------------\n')

# Simulating the channel and the decoding 
BLER_uncoded_alamouti = []
count = 0 
# Fading 
fade_taps_01 = np.random.normal(fade_mean, fade_std, n_bits//2) + 1j * np.random.normal(fade_mean, fade_std, n_bits//2)
fade_taps_02 = np.random.normal(fade_mean, fade_std, n_bits//2) + 1j * np.random.normal(fade_mean, fade_std, n_bits//2)
faded_signal_01 = np.repeat(fade_taps_01, 2) * signal_stream_tx[0]
faded_signal_02 = np.repeat(fade_taps_02, 2) * signal_stream_tx[1]
for noise in noise_var_uncoded:
    # Noise @ receiver
    noise_samples = np.random.normal(0, np.sqrt(noise), faded_signal_01.shape) + 1j * np.random.normal(0, np.sqrt(noise), faded_signal_01.shape)
    signal_stream_rx = faded_signal_01 + faded_signal_02 + noise_samples
    # Decoding
    binary_stream_rx = csl.AlamoutiDecoder(signal_stream_rx, fade_taps_01, fade_taps_02)
    # Analysis 
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, k)
    BLER_uncoded_alamouti.append(BLER)
    # Progress Update
    count += 1 
    print(f'Progress : {100 * count // n_points} %', end='\r')

################################################# With Coding 
# Channel coding 
# Matrices for Hamming (7, 4) code:
# Generator :
G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=int)

# Parity Check 
H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
], dtype=int)

channel_coded_stream_tx = csl.hamming_encoder(binary_stream_tx, G)

# Alamouti Encoding 
signal_stream_tx = csl.AlamoutiEncoder(channel_coded_stream_tx, E_b)
print(f'Samples = {signal_stream_tx}')
print(f'{np.linalg.norm(signal_stream_tx[0][:7]) ** 2}')
with open(f'results_tradMISO.txt', mode='a') as file_id:
    file_id.write(f'Number of bits = {n_bits_c}\n')
    file_id.write(f'Energy spent at the transmitter = {np.linalg.norm(signal_stream_tx[0]) ** 2 + np.linalg.norm(signal_stream_tx[1]) ** 2}\n')
    file_id.write(f'---------------------------------------------------------------------\n')

# Simulating the channel and the decoding 
# 1. Hard Decoding + Syndrome Correction 
BLER_coded_alamouti_hard = []
count = 0 
# Fading 
fade_taps_01 = np.random.normal(fade_mean, fade_std, n_bits_c // 2) + 1j * np.random.normal(fade_mean, fade_std, n_bits_c // 2)
fade_taps_02 = np.random.normal(fade_mean, fade_std, n_bits_c // 2) + 1j * np.random.normal(fade_mean, fade_std, n_bits_c // 2)
faded_signal_01 = np.repeat(fade_taps_01, 2) * signal_stream_tx[0]
faded_signal_02 = np.repeat(fade_taps_02, 2) * signal_stream_tx[1]
for noise in noise_var_coded:
    # Noise @ receiver
    noise_samples = np.random.normal(0, np.sqrt(noise), n_bits_c) + 1j * np.random.normal(0, np.sqrt(noise), n_bits_c)
    signal_stream_rx = faded_signal_01 + faded_signal_02 + noise_samples
    # Decoding
    binary_stream_rx = csl.AlamoutiDecoder(signal_stream_rx, fade_taps_01, fade_taps_02)
    binary_stream_rx = csl.hamming_decoder(binary_stream_rx, H)
    # Analysis 
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, k)
    BLER_coded_alamouti_hard.append(BLER)
    # Progress Update
    count += 1 
    print(f'Progress : {100 * count // n_points} %', end='\r')

# 2. Maximum Likelihood Decoding (MLD)
BLER_coded_alamouti_mld = []
count = 0 
# Fading 
fade_taps_01 = np.random.normal(fade_mean, fade_std, n_bits_c // 2) + 1j * np.random.normal(fade_mean, fade_std, n_bits_c // 2)
fade_taps_02 = np.random.normal(fade_mean, fade_std, n_bits_c // 2) + 1j * np.random.normal(fade_mean, fade_std, n_bits_c // 2)
faded_signal_01 = np.repeat(fade_taps_01, 2) * signal_stream_tx[0]
faded_signal_02 = np.repeat(fade_taps_02, 2) * signal_stream_tx[1]

for noise in noise_var_coded:
    # Noise @ receiver
    noise_samples = np.random.normal(0, np.sqrt(noise), n_bits_c) + 1j * np.random.normal(0, np.sqrt(noise), n_bits_c)
    signal_stream_rx = faded_signal_01 + faded_signal_02 + noise_samples
    # Decoding
    binary_stream_rx = csl.AlamoutiDecoder(signal_stream_rx, fade_taps_01, fade_taps_02, False)
    binary_stream_rx = csl.ML_Detection(binary_stream_rx, G, E_b, n, k)
    # Analysis 
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, k)
    BLER_coded_alamouti_mld.append(BLER)
    # Progress Update
    count += 1 
    print(f'Progress : {100 * count // n_points} %', end='\r')

# Visualization of results
fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_alamouti, label='Uncoded Alamouti', color='black', marker='o')
axes.semilogy(SNR_dB, BLER_coded_alamouti_hard, label='Hamming (7, 4) Hard', color='blue', marker='s')
axes.semilogy(SNR_dB, BLER_coded_alamouti_mld, label='Hamming (7, 4) MLD', color='green', marker='>')
axes.set_xlabel('SNR (in dB)')
axes.set_ylabel('BLER')
axes.legend()
axes.set_title('2 x 1 Alamouti')
axes.grid(True, which="both")

# Saving the above obtained results for future reference
# Plot(s) :
fig.savefig("results_tradMISO.png")
# BLER values :
with open("results_tradMISO.txt", mode='a') as file_id:
    file_id.write(f'BLER_uncoded_alamouti = {BLER_uncoded_alamouti}\n')
    file_id.write(f'BLER_coded_alamouti_hard = {BLER_coded_alamouti_hard}\n')
    file_id.write(f'BLER_coded_alamouti_mld = {BLER_coded_alamouti_mld}\n')
    file_id.write(f'---------------------------------------------------------------\n')
    file_id.write(f'Total execution time : {(time.time() - start_time) / 60} mins')


