# Dependencies :
import numpy as np 
import matplotlib.pyplot as plt 
import time
t0 = time.time()
plt.rcParams['font.size'] = 11
plt.figure(dpi=500)

# Function definitions
# Importing from custom made module : CommSysLib
import CommSysLib as csl

# System Parameters
k, n = 4, 7         # Uncoded and coded block length  
R = k / n           # Information rate (R) 
E_b = 1             # Energy per bit

# Signal-to-noise ratio (SNR) range in dB scale and linear scale:
n_points = 20
SNR_dB = np.linspace(-2, 20, n_points)
SNR_lin = 10 ** (SNR_dB / 10)
# AWGN noise variance : this is either I or Q noise only
noise_var_uncoded =  1 / (2 * SNR_lin)          # For uncoded
noise_var_coded =  1 / (2 * R * SNR_lin)        # For coded

# Fading model (Rayleigh) parameters : Only real or imag component
fade_mean = 0
fade_std = np.sqrt(0.5)

# Data Generation (Random binary data)
n_bits = 10 ** 6          
n_bits_c = n_bits * n // k      
binary_stream_tx = np.random.randint(0, 2, n_bits)

print(f'Samples : {binary_stream_tx[:10]}')

######################################## Without channel coding
# Orthogonal Signalling 
signal_stream_tx = csl.pulse_pos_modulation(binary_stream_tx, E_b)
print(f'Samples = {signal_stream_tx[:20]}')

with open('results_tradSISOnoCSI.txt', mode='w') as file_id:
    file_id.write(f'Number of bits = {n_bits}\n')
    file_id.write(f'Energy of the uncoded signal = {np.linalg.norm(signal_stream_tx) ** 2}\n')
    file_id.write(f'-----------------------------\n')

# Simulating the channel and the decoding
BLER_uncoded_ortho = []
count = 0 
for noise in noise_var_uncoded:
    # Fading 
    fade_taps = np.random.normal(fade_mean, fade_std, signal_stream_tx.shape) + 1j * np.random.normal(fade_mean, fade_std, signal_stream_tx.shape)
    signal_stream_rx = fade_taps * signal_stream_tx
    # Noise
    noise_samples = np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape)
    signal_stream_rx += noise_samples
    # Decoding at the receiver
    binary_stream_rx = csl.square_law_detector(signal_stream_rx)
    # Analysis 
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, 1)
    BLER_uncoded_ortho.append(BLER)
    # Progress update 
    count += 1 
    print(f'Progress : {100 * count // n_points} %', end='\r')

######################################## With Channel Coding
# Channel Coding
#  (7, 4) Hamming Code :
# Generator matrix 
G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)
# Parity Check matrix
H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1]
    ], dtype=int)

channel_coded_stream_tx = csl.hamming_encoder(binary_stream_tx, G)

# Orthogonal Signalling
signal_stream_tx = csl.pulse_pos_modulation(channel_coded_stream_tx, E_b)
print(f'Samples : {signal_stream_tx[:20]}')

with open('results_tradSISOnoCSI.txt', mode='a') as file_id:
    file_id.write(f'Number of bits = {n_bits_c}\n')
    file_id.write(f'Energy of the coded signal = {np.linalg.norm(signal_stream_tx) ** 2}\n')
    file_id.write(f'-----------------------------\n')

###################### Hard Decoding + Syndrome-based correction   
# Simulating the channel and the receiver 

BLER_coded_ortho_hard = []
count = 0 
for noise in noise_var_coded:
    # Fading 
    fade_taps = np.random.normal(fade_mean, fade_std, signal_stream_tx.shape) + 1j * np.random.normal(fade_mean, fade_std, signal_stream_tx.shape)
    signal_stream_rx = fade_taps * signal_stream_tx    
    # Noise
    noise_samples = np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape) + 1j * np.random.normal(0, np.sqrt(noise), signal_stream_rx.shape) 
    signal_stream_rx = signal_stream_rx + noise_samples 
    # Decoding 
    binary_coded_stream_rx = csl.square_law_detector(signal_stream_rx)
    # Correction 
    binary_stream_rx = csl.hamming_decoder(binary_coded_stream_rx, H)
    # Analysis 
    _, BLER = csl.calcBLER(binary_stream_tx, binary_stream_rx, 1)
    BLER_coded_ortho_hard.append(BLER)
    # Progress Update 
    count += 1 
    print(f'Progress : {100 * count // n_points} %', end='\r') 

# Plot the results :
fig, axes = plt.subplots()
axes.semilogy(SNR_dB, BLER_uncoded_ortho, label='Uncoded', color='black', marker='o')
axes.semilogy(SNR_dB, BLER_coded_ortho_hard, label='Hamming (7, 4) Hard', color='blue', marker='s')
axes.set_xlabel('SNR (in dB)')
axes.set_ylabel('Bit Error Rate')
axes.set_title('No Channel State Information')
axes.legend()
axes.grid(which='major')

# Saving the figure for future reference 
fig.savefig(f'results_tradSISOnoCSI.png')
with open(f'results_tradSISOnoCSI_{k}.txt', mode='a') as file_id:
    file_id.write(f'BLER_uncoded_ortho = {BLER_uncoded_ortho}\n')
    file_id.write(f'BLER_coded_ortho_hard = {BLER_coded_ortho_hard}\n')
    file_id.write(f'-----------------------------\n')
    file_id.write(f'Total execution time = {(time.time() - t0) / 60} mins') 