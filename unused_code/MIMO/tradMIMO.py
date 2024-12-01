# Dependencies :
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 12
plt.figure(dpi=500)

# Function Definitions
# Importing from custom made module CommSysLib
import CommSysLib as csl

# System Parameters
k, n = 4, 7         # Uncoded and coded block lengths
R = k / n           # Information rate
E_b = 1             # Energy per bit

# SNR range in dB and linear scale
n_points = 30
SNR_dB = np.linspace(-2, 20, n_points)
SNR_lin = 10 ** (SNR_dB / 10)
noise_var_uncoded = 1 / (2 * SNR_lin)
noise_var_coded = 1 / (2 * R * SNR_lin)

# Fading model : Rayleigh Fading 
fade_mean, fade_std = 0, np.sqrt(0.5)

# Data generation (Random binary data)
n_bits = 10 ** 7
n_bits_c = n * n_bits // k
binary_stream_tx = np.random.randint(0, 2, n_bits)

print(f'Samples : {binary_stream_tx[:10]}')

########################################### Without channel coding
# Transmission
# Fading taps
fade_taps = np.random.normal(fade_mean, fade_std, (n_bits//2, 2, 2)) + 1j * np.random.normal(fade_mean, fade_std, (n_bits//2, 2, 2))

signal_stream_tx = csl.svdMIMOencoder(binary_stream_tx, fade_taps, 1)

print(f'Energy of signal: {np.linalg.norm(signal_stream_tx) ** 2}')

# Simulating the channel and the decoding
count = 0 
BLER_uncoded_svd = []

for noise in noise_var_uncoded:
    # Fading
    signal_stream_rx = []
    for i in range(len(fade_taps)):
        signal_stream_rx.append(np.matmul(fade_taps[i], signal_stream_tx[i]))
    
    signal_stream_rx = np.array(signal_stream_rx)

    # Noise 
    noise_samples = np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape)
    signal_stream_rx += noise_samples 

    # Decoding 
    binary_stream_rx = csl.svdMIMOdecoder(signal_stream_rx, fade_taps)

    # Analysis
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, 4)
    BLER_uncoded_svd.append(BLER)

    # Progress 
    count += 1 
    print(f'Progress : {100 * count // len(noise_var_uncoded)} %', end='\r')

######################################### With Channel Coding
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

print(len(channel_coded_stream_tx))

# Transmit Beamforming 
fade_taps = np.random.normal(fade_mean, fade_std, (n_bits_c//2, 2, 2)) + 1j * np.random.normal(fade_mean, fade_std, (n_bits_c//2, 2, 2))

signal_stream_tx = csl.svdMIMOencoder(channel_coded_stream_tx, fade_taps, 1)
print(f'Energy of the coded signal = {np.linalg.norm(signal_stream_tx) ** 2}')

# Simulating the channel and the decoding
# Hard Decision and Syndrome-based correction
count = 0 
BLER_coded_svd = []

for noise in noise_var_coded:
    # Fading
    signal_stream_rx = []
    for i in range(len(fade_taps)):
        signal_stream_rx.append(np.matmul(fade_taps[i], signal_stream_tx[i]))
    
    signal_stream_rx = np.array(signal_stream_rx)

    # Noise 
    noise_samples = np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape)
    signal_stream_rx += noise_samples 

    # Decoding 
    binary_stream_rx = csl.svdMIMOdecoder(signal_stream_rx, fade_taps)
    
    # Syndrome Correction 
    binary_stream_rx = csl.hamming_decoder(binary_stream_rx, H)

    # Analysis
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, 4)
    BLER_coded_svd.append(BLER)

    # Progress 
    count += 1 
    print(f'Progress : {100 * count // len(noise_var_uncoded)} %', end='\r')

fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_svd, color='black', label='Uncoded')
axes.semilogy(SNR_dB, BLER_coded_svd, color='blue', label='Hamming (7, 4) Hard')

axes.set_xlabel('SNR')
axes.set_ylabel('BLER')
axes.grid(which='both')
axes.legend()
axes.set_title(r'$2 \times 2$ SVD-based MIMO')

# Save the results :
fig.savefig('tradMIMO.svg', transparent=True)
fig.savefig('tradMIMO.png')
with open('results_tradMIMO.txt', mode='w') as file_id:
    file_id.write(f'# of data bits = {n_bits}\n')
    file_id.write(f'BLER_uncoded_svd = {BLER_uncoded_svd}\n')
    file_id.write(f'BLER_coded_svd = {BLER_coded_svd}\n')