# Required dependencies 
import numpy as np
import itertools

"""
---------------------------------------------------------------------------------------------------------------------
Created by RISHABH POMAJE for the paper #__Learning to communicate over fading channels__#
Alias used while importing is "csl".

This script defines various functions which are to be used as black boxes to simulate a chain of actual digital 
communication system (wired/ wireless) and analyse its various aspects such as 
1. error rates, 
2. studying impacts of various modulation and coding schemes. 
The most used data structure is np.ndarray since there is a lot of efficient calculations available from NumPy
and is the preferred choice of data structure. 
---------------------------------------------------------------------------------------------------------------------
"""
#
# Functions to be used at the transmitter
#

def hamming_encoder(input_stream, gnrtr_matrix) :
    """
    Function that first generates a codebook depending on the generator matrix 
    (provided as a numpy multi-dimensional ndarray). 
    Using that codebook, it performs the channel coding using Hamming codes.
    
    Args:
    - input_stream (array-like): The input bit stream to be encoded.
    - gnrtr_matrix (ndarray): The generator matrix used for encoding.
    
    Returns:
    - ndarray: The channel coded bit stream generated as per the Generator Matrix specified.
    
    Note:
    - The length of the input bit stream should be compatible with the generator matrix.
    - The generator matrix should be properly constructed for the desired Hamming code.
    """
    input_size = len(input_stream)
    # k = length of the input packet ; n = length of the output packet
    k, n = np.shape(gnrtr_matrix)
    # Generating the codebook :
    code_book = np.array([], dtype=int)
    possible_inputs = [list(i) for i in itertools.product([0, 1], repeat=k)]
    for i in range(2 ** k) :
        code_word = np.array(np.matmul(possible_inputs[i], gnrtr_matrix) % 2)
        code_book = np.append(code_book, code_word) 
    code_book = code_book.reshape(2 ** k, n)
    # Encoding the stream using the codebook :
    index = 0
    channel_coded_stream = []
    for i in range(input_size // k) :
        temp = input_stream[index:index+k]
        asgn_id = np.sum(temp * 2 ** np.arange(k - 1, -1, -1))
        channel_coded_stream.append(code_book[asgn_id])
        index = index + k
        
    channel_coded_stream = np.array(channel_coded_stream)

    return channel_coded_stream.flatten()

def pulse_pos_modulation(input_stream, energy_per_symbol) :
    """
    Function to perform Pulse Position Modulation with amplitude energy_per_symbol in one symbol duration of 
    the two slots available per bit depending on the bit.
    
    Args:
    - input_stream (array-like): The input bit stream to be modulated.
    - energy_per_symbol (float): The energy per symbol for Pulse Position Modulation.
    
    Returns:
    - ndarray: An array representing the modulated signal, where each index corresponds to the discrete 
    symbol time unit corresponding to the two time slots per bit. Packaged as [].
    
    Note:
    - Each bit in the input stream is mapped to two symbol durations, with the energy_per_symbol divided 
    between the two time slots depending on the bit value (0 or 1).
    """
    output_stream = np.zeros(len(input_stream) * 2)
    coord = np.sqrt(energy_per_symbol)
    index = 0
    for i in range(len(input_stream)) :
        output_stream[index] = coord * (1 - input_stream[i])
        output_stream[index+1] = coord * input_stream[i]
        index = index + 2

    return output_stream

def BPSK_mapper(input_stream, energy_per_symbol) :
    """
    Function to perform constellation mapping on the input stream as per Binary Phase Shift Keying (BPSK) 
    with the given energy per symbol.
    
    Args:
    - input_stream (array-like): The input bit stream to be mapped.
    - energy_per_symbol (float): The energy per symbol for BPSK modulation.
    
    Returns:
    - ndarray: An array consisting of possibly complex symbols, each with magnitude sqrt(energy_per_symbol), 
    where positive values represent bit 0 and negative values represent bit 1.
    """
    output_stream = np.where(input_stream == 0, np.sqrt(energy_per_symbol), -1 * np.sqrt(energy_per_symbol))

    return output_stream

def AlamoutiEncoder(input_stream, E_b=1):
    antenna_01_signal = []
    antenna_02_signal = []
    idx = 0 
    s0 = np.sqrt(E_b / 2)
    s1 = -1 * s0
    for _ in range(0, len(input_stream), 2):
        if input_stream[idx] == 0:
            x1 = s0
        else:
            x1 = s1
        if input_stream[idx+1] == 0:
            x2 = s0
        else:
            x2 = s1
        antenna_01_signal.append(x1)
        antenna_02_signal.append(x2)
        antenna_01_signal.append(-1 * np.conjugate(x2))
        antenna_02_signal.append(np.conjugate(x1))
        idx = idx + 2
    return np.array([antenna_01_signal, antenna_02_signal])

def AlamoutiDecoder(input_stream, csi_01, csi_02, return_bin=True):
    output_stream = []
    idx = 0 
    for m in range(len(csi_01)):
        y = [input_stream[idx], np.conjugate(input_stream[idx + 1])]
        c1 = [csi_01[m], np.conjugate(csi_02[m])]
        c2 = [csi_02[m], -1 * np.conjugate(csi_01[m])]
        # Estimate 
        r1 = np.matmul(np.conjugate(c1), y) / np.linalg.norm(c1)
        r2 = np.matmul(np.conjugate(c2), y) / np.linalg.norm(c2)
        if return_bin == False:
            output_stream.append(r1)
            output_stream.append(r2)
        else:
            # BPSK Demapping 
            output_stream.append(BPSK_demapper(r1))
            output_stream.append(BPSK_demapper(r2))
        idx = idx + 2

    return np.array(output_stream).flatten()

def BPSK_demapper(rx_stream) :
    """
    Function to perform Binary Phase Shift Keying (BPSK) constellation demapping.

    Args:
    - rx_stream (ndarray): The received signal.

    Returns:
    - ndarray: The bit stream demodulated using the Minimum Distance method.

    Note:
    - This function assumes the received signal is possibly complex, but the demapping is performed using 
    the real part as the sufficient statistic.
    - Demapping is done using the Minimum Distance method, where a bit is decoded based on whether the 
    real part of the received signal is greater than 0.
    """
    rx_bits = np.where(np.real(rx_stream) > 0 , 0, 1)
    return rx_bits

def square_law_detector(rx_signal):
    """
    Function to decide the bit directly depending on the received signal by using the square law.
    (Non-coherent detection)

    Args:
    - rx_signal (ndarray): The received signal, represented as two lists, one for the time 0 slot and 
    one for the time 1 slot.

    Returns:
    - ndarray: The bit stream as an ndarray, obtained by performing square law detection.

    Note:
    - This function assumes non-coherent detection, where the bit decision is made directly based on 
    the received signal without needing phase synchronization.
    """
    estimated_message = np.ones(len(rx_signal) // 2, dtype=int)

    # Perform vectorized comparison
    rx_signal_reshaped = rx_signal.reshape(-1, 2)
    estimated_message = np.where(np.abs(rx_signal_reshaped[:, 0]) > np.abs(rx_signal_reshaped[:, 1]), 0, 1)

    return estimated_message

def mrc_decoding(input_stream, csi, return_bin=True):
    """
    Perform Maximal Ratio Combining (MRC) on received complex symbols.

    MRC is a diversity combining technique used in wireless communication systems
    to maximize the received signal-to-noise ratio (SNR) by weighting and combining
    signals received from multiple antennas.

    Args:
        input_stream (tuple): A tuple containing two arrays of received complex symbols.
            Each array corresponds to the received symbols from one of the antennas.
        csi (tuple): A tuple containing two arrays of channel state information (CSI).
            Each array represents the fading coefficients for one of the antennas.

    Returns:
        np.ndarray: An array of demodulated symbols after MRC.

    Notes:
        This implementation assumes a 1x2 MIMO (Multiple-Input Multiple-Output) system,
        where there are two antennas at the receiver.
    """
    signal_01, signal_02 = input_stream
    fade_taps_01, fade_taps_02 = csi

    combined_symbols = []
    # Combine fading taps and signals
    for i in range(len(signal_01)):
        combined_symbols.append(np.conjugate(np.array([fade_taps_01[i], fade_taps_02[i]])) @ np.array([signal_01[i], signal_02[i]]))

    if return_bin == False:
        return np.array(combined_symbols).flatten()
    else:
        return BPSK_demapper(np.array(combined_symbols).flatten())

def hamming_decoder(rx_stream, parity_chk_matrix):
    """
    Function to perform hard decision syndrome decoding for Hamming codes, given a bit stream and parity check matrix 
    as input.
    
    Args:
    - rx_stream (ndarray): The received bit stream to be decoded.
    - parity_chk_matrix (ndarray): The parity check matrix for the Hamming code.

    Returns:
    - ndarray: The 't-error' corrected bit stream.
    
    Note:
    - The matrices provided as inputs to this function must be ndarray with dtype = int.
    - The current implementation is matrix based and hence inefficient.
    """
    # Get the size of the input stream and dimensions of the parity check matrix
    input_size = len(rx_stream)
    k_1, n = parity_chk_matrix.shape
    k = n - k_1
    decoded_stream = np.zeros((input_size // n, k), dtype=int)
    # Generating the syndrome book :
    possible_single_errors = np.eye(n, dtype=int) 
    syndrome_book = parity_chk_matrix.T
    # Performing the syndrome decoding :
    num_blocks = input_size // n 
    for idx in range(num_blocks):
        rx_packet = rx_stream[idx * n: (idx + 1) * n]  # Extract packet for processing
        syndrome = np.matmul(rx_packet, parity_chk_matrix.T) % 2
        if np.sum(syndrome) != 0 :
        # Correcting 1-bit errors :
            for j in range(n):
                if np.array_equal(syndrome, syndrome_book[j]):
                    rx_packet = np.bitwise_xor(rx_packet, possible_single_errors[j])
        decoded_stream[idx] = rx_packet[:k]  # Append the corrected packet
        
    return decoded_stream.flatten()

def ML_Detection(rx_signal, G, E_sym, n, k) :
    """ 
    Function to perform Maximum Likelihood Decoding for Hamming Code.
    
    Args:
    - rx_signal (ndarray): The received signal.
    - G (ndarray): The generator matrix for the Hamming code.
    - E_sym (float): The energy per symbol.
    - n (int): Length of the codeword.
    - k (int): Length of the message.
    
    Returns:
    - ndarray: The Maximum Likelihood Decoding estimates.
    """
    # Matrix of all possible codewords :
    code_book = np.array([], dtype=int)
    possible_inputs = [list(i) for i in itertools.product([0, 1], repeat=k)]
    for i in range(2 ** k) :
        code_word = np.array(np.matmul(possible_inputs[i], G) % 2)
        code_book = np.append(code_book, code_word)
    C = BPSK_mapper(code_book, E_sym)  
    code_book = code_book.reshape(2 ** k, n)
    C = C.reshape(2 ** k, n)
    
    # Finding the MLD Estimates :
    n_blocks = len(rx_signal) // n
    estimates = []
    for i in range(n_blocks):
        start_idx = i * n
        end_idx = (i + 1) * n
        received_segment = rx_signal[start_idx:end_idx]
        distances = [np.linalg.norm(C[j] - received_segment) for j in range(2**k)]
        nearest_codeword_idx = np.argmin(distances)
        nearest_codeword = code_book[nearest_codeword_idx]  
        estimates.append(np.array(nearest_codeword[0:k])) # Extracting the first 4 bits
    
    return np.array(estimates).flatten()

def calcBLER(original_message, estimated_message, block_length) :
    """ 
    Function to calculate the block error rate.
    
    Args:
    - original_message (ndarray): The original bit stream.
    - estimated_message (ndarray): The estimated bit stream.
    - block_length (int): The length of each block.
    
    Returns:
    - list: A list containing the number of block errors and the block error rate.
    """
    n_errors = 0
    n_blocks = len(original_message) // block_length
    idx = 0
    for _ in range(n_blocks) :
        if not np.all(original_message[idx:idx+block_length] == estimated_message[idx:idx+block_length]) :
            n_errors += 1 
        idx += block_length  
    return [n_errors, n_errors / n_blocks]      