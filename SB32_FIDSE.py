# Comparison of SE and FID from Leonids measurements

import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os, re

def calculate_frequency_scale(Time):
    numberp = len(Time)

    dt = Time[1] - Time[0]
    f_range = 1 / dt
    f_nyquist = f_range / 2
    df = 2 * (f_nyquist / numberp)
    Freq = np.arange(-f_nyquist, f_nyquist + df, df)
    Freq = Freq[:-1]

    return Freq

def time_domain_phase(Real, Imaginary):
    delta = np.zeros(360)
    
    for phi in range(360):
        Re_phased = Real * np.cos(np.deg2rad(phi)) - Imaginary * np.sin(np.deg2rad(phi))
        Im_phased = Real * np.sin(np.deg2rad(phi)) + Imaginary * np.cos(np.deg2rad(phi))
        Magnitude_phased = calculate_amplitude(Re_phased, Im_phased)
        
        Re_cut = Re_phased[:5]
        Ma_cut = Magnitude_phased[:5]
        
        delta[phi] = np.mean(Ma_cut - Re_cut)
    
    idx = np.argmin(delta)
    #print(idx)

    Re = Real * np.cos(np.deg2rad(idx)) - Imaginary * np.sin(np.deg2rad(idx))
    Im = Real * np.sin(np.deg2rad(idx)) + Imaginary * np.cos(np.deg2rad(idx))

    # # For debug
    # Amp = calculate_amplitude(Re, Im)

    # plt.plot(Real, 'k', label='Re original')
    # plt.plot(Imaginary, 'b', label='Im original')
    # plt.plot(Re, 'm--', label='Re phased')
    # plt.plot(Im, 'c--', label='Im phased')
    # plt.plot(Amp, 'r', label='Amp phased')
    # plt.xlabel('Time, μs')
    # plt.ylabel('Amplitude, a.u.')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return Re, Im

def adjust_frequency(Frequency, Re, Im):
    # Create complex FID
    Fid_unshifted = np.array(Re + 1j * Im)

    # FFT
    FFT = np.fft.fftshift(np.fft.fft(Fid_unshifted))

    # Check the length of FFT and Frequency (it is always the same, this is just in case)
    if len(Frequency) != len(FFT):
        Frequency = np.linspace(Frequency[0], Frequency[-1], len(FFT))

    # Find index of max spectrum (amplitude)
    index_max = np.argmax(FFT)

    # Find index of zero (frequency)
    index_zero = find_nearest(Frequency, 0)

    # Find difference
    delta_index = index_max - index_zero

    if delta_index == 0:
        return Re, Im

    # Shift the spectra (amplitude) by the difference in indices
    FFT_shifted = np.concatenate((FFT[delta_index:], FFT[:delta_index]))

    # iFFT
    Fid_shifted = np.fft.ifft(np.fft.fftshift(FFT_shifted))

    # Define Real, Imaginary and Amplitude
    Re_shifted = np.real(Fid_shifted)
    Im_shifted = np.imag(Fid_shifted)

    # # For debug
    # Re_freq_original = np.real(FFT)
    # Im_freq_original = np.imag(FFT)

    # Re_freq_shifted = np.real(FFT_shifted)
    # Im_freq_shifted = np.imag(FFT_shifted)

    # plt.plot(Fid_unshifted, 'k', label='Fid original')
    # plt.plot(Fid_shifted, 'b', label='Fid shifted')
    # plt.xlabel('Frequency, MHz')
    # plt.ylabel('Amplitude, a.u.')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plt.plot(Frequency, Re_freq_original, 'k', label='Re original')
    # plt.plot(Frequency, Im_freq_original, 'b', label='Im original')
    # plt.plot(Frequency, Re_freq_shifted, 'm--', label='Re shifted')
    # plt.plot(Frequency, Im_freq_shifted, 'c--', label='Im shifted')
    # plt.plot(Frequency, FFT_shifted, 'r', label='Amp shifted')
    # plt.plot(Frequency, FFT, 'k--', label='Amp original')
    # plt.xlabel('Frequency, MHz')
    # plt.ylabel('Amplitude, a.u.')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return Re_shifted, Im_shifted

# Find nearest value in array
def find_nearest(array, value):
    idx = (np.abs(np.asarray(array) - value)).argmin()
    return idx

# Gaussian functions
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def gauss1(x, A, sigma, y0):
    return A * np.exp(-x**2 / (2 * sigma**2)) + y0

def gauss2(amplitude):
    return lambda x, sigma, y0: gauss1(x, amplitude, sigma, y0)

def poly1(x, A, c, g):
    return A +  c * x**2 + g * x**4

def poly2(amplitude):
    return lambda x, c, g: poly1(x, amplitude, c, g)

def decaying_exponential(x, a, b, c):
    return a * np.exp(-x/b) + c

# Calculate amplitude
def calculate_amplitude(real, imaginary):
    return np.sqrt(real**2 + imaginary**2)

def frequency_domain_analysis(FFT, Frequency):

    # 8. Simple baseline
    _, Re, _ = simple_baseline_correction(FFT)

    # 9. Apodization
    Real_apod = calculate_apodization(Re, Frequency)

    # 10. M2 & T2
    M2, T2 = calculate_M2(Real_apod, Frequency)

    return M2, T2

def calculate_M2(FFT_real, Frequency):
    # Take the integral of the REAL PART OF FFT by counts
    Integral = np.trapz(np.real(FFT_real))
    
    # Normalize FFT to the Integral value
    Fur_normalized = np.real(FFT_real) / Integral
    
    # Calculate the integral of normalized FFT to receive 1
    Integral_one = np.trapz(Fur_normalized)
    
    # Multiplication (the power ^n will give the nth moment (here it is n=2)
    Multiplication = (Frequency ** 2) * Fur_normalized
    
    # Calculate the integral of multiplication - the nth moment
    # The (2pi)^2 are the units to transform from rad/sec to Hz
    # ppbly it should be (2pi)^n for generalized moment calculation
    M2 = (np.trapz(Multiplication)) * 4 * np.pi ** 2
    
    # Check the validity
    if np.abs(np.mean(Multiplication[0:10])) > 10 ** (-6):
        print('Apodization is wrong!')

    if M2 < 0:
        M2 = 0
        T2 = 0
    else:
        T2 = np.sqrt(2/M2)
    
    return M2, T2

def simple_baseline_correction(FFT):
    twentyperc = int(round(len(FFT) * 0.02))
    Baseline = np.mean(np.real(FFT[:twentyperc]))
    FFT_corrected = FFT - Baseline
    Re = np.real(FFT_corrected)
    Im = np.imag(FFT_corrected)
    Amp = calculate_amplitude(Re, Im)
    return Amp, Re, Im

def calculate_apodization(Real, Freq):
    # Find sigma at 0.1% from the max amplitude of the spectra
    Maximum = np.max(np.abs(Real))
    idx_max = np.argmax(np.abs(Real))
    ten_percent = Maximum * 0.001

    b = np.argmin(np.abs(Real[idx_max:] - ten_percent))
    sigma_ap = Freq[idx_max + b]

    apodization_function_s = np.exp(-(Freq / sigma_ap) ** 6)

    Real_apod = Real * apodization_function_s

    plt.plot(Freq, Real, 'r', label = 'Original')
    plt.plot(Freq, apodization_function_s, 'k--', label = 'Apodization')
    plt.plot(Freq, Real_apod, 'b', label = 'Apodized')
    plt.legend()
    plt.show
    
    return Real_apod

def calculate_frequency_scale(Time):
    numberp = len(Time)

    dt = Time[1] - Time[0]
    f_range = 1 / dt
    f_nyquist = f_range / 2
    df = 2 * (f_nyquist / numberp)
    Freq = np.arange(-f_nyquist, f_nyquist + df, df)
    Freq = Freq[:-1]

    return Freq

def add_zeros(Time, Real, Imaginary, number_of_points):
    length_diff = number_of_points - len(Time)
    amount_to_add = np.zeros(length_diff+1)

    Re_zero = np.concatenate((Real, amount_to_add))
    Im_zero = np.concatenate((np.zeros(len(Real)), amount_to_add))

    dt = Time[1] - Time[0]
    Time_to_add = Time[-1] + np.arange(1, length_diff + 1) * dt

    Time = np.concatenate((Time, Time_to_add))
    Fid = np.array(Re_zero + 1j * Im_zero)
    Fid = Fid[:-1]

    return Time, Fid

def apodization(Time, Real, Imaginary):
    Amplitude = calculate_amplitude(Real, Imaginary)
    sigma = 90

    apodization_function = np.exp(-(Time / sigma) ** 4)
    Re_ap = Real * apodization_function
    Im_ap = Imaginary * apodization_function

    # plt.plot(Time, apodization_function, 'r--', label='apodization')
    # plt.plot(Time, Real, 'k', label='Re')
    # plt.plot(Time, Re_ap, 'k--', label='Re ap')
    # plt.xlim([-5,80])
    # plt.xlabel('Time, μs')
    # plt.ylabel('Amplitude, a.u.')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return Re_ap, Im_ap

# Math procedures
def FFT_handmade(Fid, Time, Freq):
    N = len(Freq)
    Fur = np.zeros(N, dtype=complex)

    cos_values = np.cos(2 * np.pi * Time[:, None] * Freq)
    sin_values = np.sin(2 * np.pi * Time[:, None] * Freq)

    for i in range(N):
        Fur[i] = np.sum(Fid * (cos_values[:, i] - 1j * sin_values[:, i]))
    return Fur
    
def create_spectrum(Time, Real, Imaginary, correct):
    number_of_points = 2**16
    # 5. Apodize the time-domain
    Re_ap, Im_ap = apodization(Time, Real, Imaginary)

    if correct == True:
        Fr = calculate_frequency_scale(Time)
        Re_ph, Im_ph = time_domain_phase(Re_ap, Im_ap)
        Re_ad, Im_ad = adjust_frequency(Fr, Re_ph, Im_ph)
    else:
        Re_ad = Re_ap
        Im_ad = Im_ap

    # 6. Add zeros
    Time_zero, Fid_zero = add_zeros(Time, Re_ad, Im_ad, number_of_points)

    Frequency = calculate_frequency_scale(Time_zero)

    # 7. FFT
    FFT = np.fft.fftshift(np.fft.fft(Fid_zero))
    # FFT = FFT_handmade(Fid_zero, Time_zero, Frequency)

    # 8. Simple baseline
    _, Re, Im = simple_baseline_correction(FFT)

    # 9. Apodization
    #Real_apod = calculate_apodization(Re, Frequency)

    return Frequency, Re, Im

# Read data from file
def read_data(file_path):
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1], data[:, 2]

def prepare_data(parent_directory, filename, correction):
        file_path = os.path.join(parent_directory, filename)
        Time, Re_original, Im_original = read_data(file_path)

        if correction == True:
            Frequency = calculate_frequency_scale(Time)
            #Correct phase
            R_phased, I_phased = time_domain_phase(Re_original, Im_original)
            #Adjust frequency
            Re, Im = adjust_frequency(Frequency, R_phased, I_phased)
            Amp = calculate_amplitude(Re, Im)

            # For debug
            # plt.plot(Time, Re_original, 'k--', label='Re original')
            # plt.plot(Time, Im_original, 'b--', label='Im original')
            # plt.plot(Time, R_phased, 'k.', label='Re phased')
            # plt.plot(Time, I_phased, 'b.', label='Im phased')
            # plt.plot(Time, Re, 'k', label='Re fr')
            # plt.plot(Time, Im, 'b', label='Im fr')
            # plt.plot(Time, Amp, 'r', label='Amp')
            # plt.xlim([-5,80])
            # plt.xlabel('Time, μs')
            # plt.ylabel('Amplitude, a.u.')
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

        else:
            Re = Re_original
            Im = Im_original
            Amp = calculate_amplitude(Re_original, Im_original)

        return Time, Re, Im, Amp

def reference_long_component(Time, Component_n, end):
    # 3. Cut the ranges for fitting
    minimum = find_nearest(Time, end)

    Time_range = Time[minimum:]
    Component_n_range = Component_n[minimum:]

    p = [5, 30, 0.5]
    # 7. Fit data to exponential decay
    popt, _      = curve_fit(decaying_exponential, Time_range, Component_n_range, p0 =p)
    
    # 9. Calculate the curves fitted to data within the desired range
    Component_f = decaying_exponential(Time, *popt)

    # 10. Subtract
    Component_sub = Component_n - Component_f

    # # For Debug
    # plt.plot(Time, Component_n, 'r', label='Original')
    # plt.plot(Time, Component_sub, 'b', label='Subtracted')
    # plt.plot(Time, Component_f, 'k--', label='Fitted')
    # plt.xlabel('Time, μs')
    # plt.ylabel('Amplitude, a.u.')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return Component_sub


# Read data
parent_directory = r'C:\Mega\NMR\003_Temperature\2024_09_12_SE and FID experiment\SB32'

SE_file = r'SE_Cellulose_ 6_c.dat'
FID_file = r'FID_Cellulose_ 6_c.dat'

Time_SE1, Re_SE1, Im_SE1 = read_data(r'C:\Mega\NMR\003_Temperature\2024_09_12_SE and FID experiment\SB32\SE_Cellulose_ 6_c.dat')
Time_SE, Re_SE, Im_SE, Amp_SE = prepare_data(parent_directory, SE_file, False)

Time_FID1, Re_FID1, Im_FID1 = read_data(r'C:\Mega\NMR\003_Temperature\2024_09_12_SE and FID experiment\SB32\FID_Cellulose_ 6_c.dat')
Time_FID, Re_FID, Im_FID, Amp_FID = prepare_data(parent_directory, FID_file, False)

# Subtract the long component for FID, SE and MSE REAL & IMAG
Re_td_fid_short    = reference_long_component(Time_FID, Re_FID, end= 60)
Re_td_se_short     = reference_long_component(Time_SE, Re_SE, end= 60)

## PLOT FID, SE together
Fr_FID, Re_FID, Im_FID = create_spectrum(Time_FID, Re_td_fid_short, 0, False)
Fr_SE, Re_SE, Im_SE = create_spectrum(Time_SE, Re_td_se_short, 0, False)

Amp_FID = calculate_amplitude(Re_FID, Im_FID)
Amp_SE = calculate_amplitude(Re_SE, Im_SE)





# #plot FID, MSE and SE REAL
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5))
# ax1.plot(Time_SE, Re_td_se_short, 'b', label='SE')
# ax1.plot(Time_FID, Re_td_fid_short, 'r', label='FID')
# ax1.set_xlim(-5, 80)
# ax1.set_title('FID and SE decays', loc='left')
# ax1.set_xlabel('Time, μs')
# ax1.set_ylabel('Amplitude, a.u.')
# ax1.legend()

# ax2.plot(Fr_FID, Re_FID, 'b', label='Re FID')
# ax2.plot(Fr_SE, Re_SE, 'r', label='Re SE')
# ax2.set_xlim(-0.2, 0.2)
# ax2.set_title('FFT FID and SE', loc='left')
# ax2.set_xlabel('Frequency, MHz')
# ax2.set_ylabel('Amplitude, a.u.')
# ax2.legend()


# ax3.plot(Fr_FID, Re_FID, 'r', label='FID Re')
# ax3.plot(Fr_FID, Im_FID, 'b', label='FID Im')
# ax3.plot(Fr_FID, Amp_FID, 'k', label='FID Amp')
# ax3.set_xlim(-0.4,0.4)
# ax3.set_title('FFT FID', loc='left')
# ax2.set_xlabel('Frequency, MHz')
# ax3.set_ylabel('Intensity, a.u.')
# ax3.legend()

# ax4.plot(Fr_SE, Re_SE, 'r', label='SE Re')
# ax4.plot(Fr_SE, Im_SE, 'b', label='SE Im')
# ax4.plot(Fr_SE, Amp_SE, 'k', label='SE Amp')
# ax4.set_xlim(-0.4,0.4)
# ax4.set_title('FFT SE', loc='left')
# ax4.set_xlabel('Frequency, MHz')
# ax4.set_ylabel('Intensity, a.u.')
# ax4.legend()
# plt.tight_layout()
# plt.show()
