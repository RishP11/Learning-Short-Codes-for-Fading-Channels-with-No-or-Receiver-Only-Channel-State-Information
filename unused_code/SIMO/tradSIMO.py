# Dependencies :
import time 
start_time = time.time()
import numpy as np 
import matplotlib.pyplot as plt 

# Function Defintions
# Importing from custom made module : CommSysLib.py
import CommSysLib as csl 

# System Parameters
k, n = 4, 7     # Uncoded and Coded block lengths  
R = k / n       # Information rate
E_b = 1         # Energy per bit

# SNR range in dB and linear scale 
n_points = 20 
SNR_dB = np.linspace(-2, 20, n_points)
SNR_lin = 10 ** (SNR_dB / 10)

# Variance of only the real(imag) component of the noise 
noise_var_uncoded = 1 / (2 * SNR_lin)
noise_var_coded = 1 / (2 * R * SNR_lin)
# Fading model parameters :: Rayleigh fading 
fade_mean = 0 
fade_std = np.sqrt(0.5) 

####################################### Data Generation (Random Binary Data) 
n_bits = 4 * (10 ** 6)
n_bits_c = n_bits * n // k
binary_stream_tx = np.random.randint(0, 2, n_bits)

print(f'Completed Data Generation. Samples : {binary_stream_tx[:10]}')

######################################## Without channel coding
# Constellation mapping
signal_stream_tx = csl.BPSK_mapper(binary_stream_tx, 1)
print(f'Completed Constellation mapping. Samples : {signal_stream_tx}')
with open("results_tradSIMO.txt", mode='w') as file_id:
    file_id.write(f'Number of bits = {n_bits}\n')
    file_id.write(f'Energy of the signal = {np.linalg.norm(signal_stream_tx) ** 2}\n')
    file_id.write(f'---------------------------------------------------------------------\n')

# Simulating the channel and the decoding 
BLER_uncoded_mrc = []
count = 0
for noise in noise_var_uncoded:
    # Fading
    fade_taps_01 = np.random.normal(fade_mean, fade_std, signal_stream_tx.shape) + 1j * np.random.normal(fade_mean, fade_std, signal_stream_tx.shape)
    fade_taps_02 = np.random.normal(fade_mean, fade_std, signal_stream_tx.shape) + 1j * np.random.normal(fade_mean, fade_std, signal_stream_tx.shape)
    signal_stream_rx_01 = fade_taps_01 * signal_stream_tx
    signal_stream_rx_02 = fade_taps_02 * signal_stream_tx 
    # Noise 
    noise_samples_01 = np.random.normal(0, np.sqrt(noise),  signal_stream_rx_01.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx_01.shape)
    noise_samples_02 = np.random.normal(0, np.sqrt(noise), signal_stream_rx_02.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx_02.shape)
    signal_stream_rx_01 += noise_samples_01
    signal_stream_rx_02 += noise_samples_02
    # Decoding
    binary_stream_rx = csl.mrc_decoding([signal_stream_rx_01, signal_stream_rx_02], [fade_taps_01, fade_taps_02])
    # Analysis 
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, k)
    BLER_uncoded_mrc.append(BLER)
    # Progress Update 
    count += 1
    print(f'Progress : {100 * count // n_points} %', end='\r')

print("Uncoded system simulation complete.")
################################################ With Channel Coding
# Channel Coding
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

print(f'Samples = {channel_coded_stream_tx[:10]}')
print(f'Length = {len(channel_coded_stream_tx)}')

# Constellation mapping
signal_stream_tx = csl.BPSK_mapper(channel_coded_stream_tx, E_b)

print(f'Samples = {signal_stream_tx}')
with open("results_tradSIMO.txt", mode='a') as file_id:
    file_id.write(f'Number of bits = {n_bits}\n')
    file_id.write(f'Energy of the signal = {np.linalg.norm(signal_stream_tx) ** 2}\n')
    file_id.write(f'---------------------------------------------------------------------\n')

# Simulating the channel and the decoding
# 1.  Hard decision and Syndrome Decoding 
BLER_coded_mrc_hard = []
count = 0 
for noise in noise_var_coded:
    # Fading 
    fade_taps_01 = np.random.normal(fade_mean, fade_std, signal_stream_tx.shape) + 1j * np.random.normal(fade_mean, fade_std, signal_stream_tx.shape)
    fade_taps_02 = np.random.normal(fade_mean, fade_std, signal_stream_tx.shape) + 1j * np.random.normal(fade_mean, fade_std, signal_stream_tx.shape)
    signal_stream_rx_01 = fade_taps_01 * signal_stream_tx
    signal_stream_rx_02 = fade_taps_02 * signal_stream_tx
    # Noise 
    noise_samples_01 = np.random.normal(0, np.sqrt(noise), signal_stream_rx_01.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx_01.shape)
    noise_samples_02 = np.random.normal(0, np.sqrt(noise), signal_stream_rx_02.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx_02.shape)
    signal_stream_rx_01 += noise_samples_01
    signal_stream_rx_02 += noise_samples_02
    # Decoding
    binary_stream_rx = csl.hamming_decoder(csl.mrc_decoding([signal_stream_rx_01, signal_stream_rx_02], [fade_taps_01, fade_taps_02]), H)
    # Analysis
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, k)
    BLER_coded_mrc_hard.append(BLER)
    # Progress Update
    count += 1
    print(f'Progress : {100 * count // n_points} %', end='\r')

print("Coded system with Hard Decoding complete.")
# 2. Maximum Likelihood Decoding
BLER_coded_mrc_mld = []
count = 0 
for noise in noise_var_coded:
    # Fading 
    fade_taps_01 = np.random.normal(fade_mean, fade_std, n_bits_c) + 1j * np.random.normal(fade_mean, fade_std, n_bits_c)
    fade_taps_02 = np.random.normal(fade_mean, fade_std, n_bits_c) + 1j * np.random.normal(fade_mean, fade_std, n_bits_c)
    signal_stream_rx_01 = fade_taps_01 * signal_stream_tx
    signal_stream_rx_02 = fade_taps_02 * signal_stream_tx
    # Noise 
    noise_samples_01 = np.random.normal(0, np.sqrt(noise), n_bits_c) + 1j * np.random.normal(0, np.sqrt(noise), n_bits_c)
    noise_samples_02 = np.random.normal(0, np.sqrt(noise), n_bits_c) + 1j * np.random.normal(0, np.sqrt(noise), n_bits_c)
    signal_stream_rx_01 = signal_stream_rx_01 + noise_samples_01
    signal_stream_rx_02 = signal_stream_rx_02 + noise_samples_02
    # Decoding
    binary_stream_rx = csl.mrc_decoding([signal_stream_rx_01, signal_stream_rx_02], [fade_taps_01, fade_taps_02], False) 
    binary_stream_rx = csl.ML_Detection(binary_stream_rx, G, E_b, n, k)
    # Analysis
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, k)
    BLER_coded_mrc_mld.append(BLER)
    # Progress Update
    count += 1
    print(f'Progress : {100 * count // n_points} %', end='\r')
print("Coded system with MLD decoding complete.")

fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_mrc, label='Uncoded', color='black', marker='o')
axes.semilogy(SNR_dB, BLER_coded_mrc_hard, label='Hamming (7, 4) Hard', color='blue', marker='>')
axes.semilogy(SNR_dB, BLER_coded_mrc_mld, label='Hamming (7, 4) MLD', color='green', marker='s')
axes.set_xlabel('$SNR (in dB)$')
axes.set_ylabel('$BLER$')
axes.legend()
axes.set_title(f'SIMO')
axes.grid(True, which="both")

# Saving the results for future reference :
# Plot(s):
fig.savefig("results_tradSIMO.png")
# BLER values :
with open("results_tradSIMO.txt", mode='a') as file_id:
    file_id.write(f'BLER_uncoded_mrc = {BLER_uncoded_mrc}\n')
    file_id.write(f'BLER_coded_mrc_hard = {BLER_coded_mrc_hard}\n')
    file_id.write(f'BLER_coded_mrc_mld = {BLER_coded_mrc_mld}\n')
    file_id.write(f'------------------------------------------------\n')
    file_id.write(f'Total execution time = {(time.time() - start_time) / 60} mins\n')
    file_id.write(f'------------------------------------------------\n')