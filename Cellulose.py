# This Python file uses the following encoding: utf-8
import numpy as np
import pandas as pd 
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os, re


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

#Polynom functions
def poly1(x, A, c, g):
    return A +  c * x**2 + g * x**4

def poly2(amplitude):
    return lambda x, c, g: poly1(x, amplitude, c, g)

# Exp function
def decaying_exponential(x, a, b, c):
    return a * np.exp(-x/b) + c

# Read data from file
def read_data(file_path):
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1], data[:, 2]

# Calculate frequency from time
def calculate_frequency_scale(Time):
    numberp = len(Time)

    dt = Time[1] - Time[0]
    f_range = 1 / dt
    f_nyquist = f_range / 2
    df = 2 * (f_nyquist / numberp)
    Freq = np.arange(-f_nyquist, f_nyquist + df, df)
    Freq = Freq[:-1]

    return Freq

# Calculate amplitude
def calculate_amplitude(real, imaginary):
    return np.sqrt(real**2 + imaginary**2)

# Adjust phase
def adjust_phase(Real, Imaginary):
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

    return Re, Im

# Adjust frequency
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

    # Shift the spectra (amplitude) by the difference in indices
    FFT_shifted = np.concatenate((FFT[delta_index:], FFT[:delta_index]))

    # iFFT
    Fid_shifted = np.fft.ifft(np.fft.fftshift(FFT_shifted))

    # Define Real, Imaginary and Amplitude
    Re_shifted = np.real(Fid_shifted)
    Im_shifted = np.imag(Fid_shifted)

    return Re_shifted, Im_shifted

# Correct spectra with baseline
def simple_baseline_correction(FFT):
    twentyperc = int(round(len(FFT) * 0.02))
    Baseline = np.mean(np.real(FFT[:twentyperc]))
    FFT_corrected = FFT - Baseline
    Re = np.real(FFT_corrected)
    Im = np.imag(FFT_corrected)
    Amp = calculate_amplitude(Re, Im)
    return Amp, Re, Im

# Apodize spectra
def apodization_fft(Real, Freq):
    # Find sigma at 0.1% from the max amplitude of the spectra
    Maximum = np.max(np.abs(Real))
    idx_max = np.argmax(np.abs(Real))
    ten_percent = Maximum * 0.001

    b = np.argmin(np.abs(Real[idx_max:] - ten_percent))
    sigma_ap = Freq[idx_max + b]

    apodization_function_s = np.exp(-(Freq / sigma_ap) ** 6)

    Real_apod = Real * apodization_function_s

    # plt.plot(Freq, Real, 'r', label = 'Original')
    # plt.plot(Freq, apodization_function_s, 'k--', label = 'Apodization')
    # plt.plot(Freq, Real_apod, 'b', label = 'Apodized')
    # plt.legend()
    # plt.show()
    
    return Real_apod

# Zero filling procedure
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

# Apodization of time Domain
def apodization(Time, Real, Imaginary):
    Amplitude = calculate_amplitude(Real, Imaginary)
    sigma = 60

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

# Cut the first part of the FID
def cut_beginning(Time, Data):
    Time_plot = Time[np.argmax(Data):]
    Data_plot = Data[np.argmax(Data):]
    return Time_plot, Data_plot

# Normalize the data to FID at long times
def normalize_to_fid(Fid, Data, Time_fid, Time_data):
    Mean_amp_fid_long = np.mean(Fid[find_nearest(Time_fid, 60):find_nearest(Time_fid, 70)])
    Mean_amp_dat_long = np.mean(Data[find_nearest(Time_data, 60):find_nearest(Time_data, 70)])

    difference = Mean_amp_dat_long-Mean_amp_fid_long
    Normalized_amp = Data -difference

    # the division doesn't work very good
    # proportionality_coefficient = Mean_amp_fid_long /Mean_amp_dat_long
    # Normalized_amp = Data * proportionality_coefficient
    return Normalized_amp

# reference the long component
def reference_long_component(Time, Component_n, end):
    # 3. Cut the ranges for fitting
    minimum = find_nearest(Time, end)

    Time_range = Time[minimum:]
    Component_n_range = Component_n[minimum:]

    # Smooth data
    Smooth = savgol_filter(Component_n_range, 40, 0)

    p = [5, 30, 0.5]
    # 7. Fit data to exponential decay
    popt, _      = curve_fit(decaying_exponential, Time_range, Smooth, p0 =p)
    
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

# Calculate M2
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

# Global functions
# Create spectra with option for corrections
def freq_domain_correction(Time, Real, Imaginary, correct):
    number_of_points = 2**14
    # 5. Apodize the time-domain
    Re_ap, Im_ap = apodization(Time, Real, Imaginary)

    if correct == True:
        Fr = calculate_frequency_scale(Time)
        Re_ph, Im_ph = adjust_phase(Re_ap, Im_ap)
        Re_ad, Im_ad = adjust_frequency(Fr, Re_ph, Im_ph)
    else:
        Re_ad = Re_ap
        Im_ad = Im_ap

    # 6. Add zeros
    Time_zero, Fid_zero = add_zeros(Time, Re_ad, Im_ad, number_of_points)

    Frequency = calculate_frequency_scale(Time_zero)

    # 7. FFT
    FFT = np.fft.fftshift(np.fft.fft(Fid_zero))

    # 8. Simple baseline
    _, Re, Im = simple_baseline_correction(FFT)

    # 9. Apodization
    Real_apod = apodization_fft(Re, Frequency)

    return Frequency, Real_apod, Im

# Create NMR signal with option for corrections
def time_domain_correction(parent_directory, filename, correction):
        file_path = os.path.join(parent_directory, filename)
        Time, Re_original, Im_original = read_data(file_path)

        if correction == True:
            Frequency = calculate_frequency_scale(Time)
            #Correct phase
            R_phased, I_phased = adjust_phase(Re_original, Im_original)
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
            # plt.title(filename)
            # plt.tight_layout()
            # plt.show()

        else:
            Re = Re_original
            Im = Im_original
            Amp = calculate_amplitude(Re_original, Im_original)

        return Time, Re, Im, Amp

# General analysis of SE
def analysis_SE(measurement_files, baseline, filename1, filename2, type, type_save, Re_td_fid_cut, Time_fid):
    # read data
    Time = np.array(measurement_files[filename1]['Time'])
    data_measurement_files = np.array(measurement_files[filename1][type])
    data_baseline = np.array(baseline[filename2][type])
    # compare length
    if len(data_measurement_files) != len(data_baseline):
        data_baseline = data_baseline[:len(data_measurement_files)]
    # subtract
    difference = data_measurement_files - data_baseline
    # save subtracted data and maximum
    measurement_files[filename1][type_save] = difference

    Time_cut, Data_cut = cut_beginning(Time, difference)
    Data_norm = normalize_to_fid(Re_td_fid_cut, Data_cut, Time_fid, Time_cut)
    Data_short    = reference_long_component(Time_cut, Data_cut, end= 55)

    maximum.append(max(Data_short))

    return maximum, Time_cut, Data_short

# general analysis for FID, MSE and SE



# Directory and pattern setup
parent_directory = os.path.join(os.getcwd(), 'SE_Cycle')
pattern = re.compile(r'Cellulose.*.dat$')
pattern2 = re.compile(r'Empty.*.dat$')
pattern3 = re.compile(r'Cellulose.*_\s*(\d+)_c\.dat')
pattern_FID = re.compile(r'FID_C.*.dat$')
pattern_FID_empty = re.compile(r'FID_Empty.*.dat$')
pattern_FID_water = re.compile(r'FID_Water.*.dat$')
pattern_MSE = re.compile(r'MSE.*.dat$')

measurement_files = {}
baseline = {}
maximum = []
time_shift = 0

echo_time = np.arange(9, 26)
echo_time_fit = np.arange(0, 25, 0.001)

#### Read and prepare data
# FID, MSE, Empty, Water part, SE
for filename in os.listdir(parent_directory):

# CHECK ZERO!!!
    # FID
    if pattern_FID.match(filename):
        correction = True
        Time_fid, Re_td_fid, Im_td_fid, Amp_td_fid = time_domain_correction(parent_directory, filename, correction)
        Time_fid = Time_fid - 2

    # MSE
    elif pattern_MSE.match(filename):
        correction = True
        Time_mse, Re_td_mse, Im_td_mse, Amp_td_mse = time_domain_correction(parent_directory, filename, correction)
        # UNFORTUNATELY I recorded a very long MSE, my bad
        length_to_cut = len(Time_fid)
        Time_mse      = Time_mse[:length_to_cut]
        Re_td_mse     = Re_td_mse[:length_to_cut]
        Im_td_mse     = Im_td_mse[:length_to_cut]
        Amp_td_mse    = Amp_td_mse[:length_to_cut]

    # Empty
    elif pattern_FID_empty.match(filename):
        correction = False
        Time_fid_empty, Re_td_fid_empty, Im_td_fid_empty, Amp_td_fid_empty = time_domain_correction(parent_directory, filename, correction)
    
    # Water
    elif pattern_FID_water.match(filename):
        correction = True
        Time_water, Re_td_water, Im_td_water, Amp_td_water = time_domain_correction(parent_directory, filename, correction)
        # UNFORTUNATELY I recorded a very long FID for water, my bad
        length_to_cut = len(Time_fid)
        Time_water      = Time_water[:length_to_cut]
        Re_td_water     = Re_td_water[:length_to_cut]
        Im_td_water     = Im_td_water[:length_to_cut]
        Amp_td_water    = Amp_td_water[:length_to_cut]

    # SE
    elif pattern.match(filename):
        correction = True
        Time, Re, Im, Amp = time_domain_correction(parent_directory, filename, correction)
        measurement_files[filename] = {'Time': Time, 'Amp': Amp, 'Re': Re, 'Im': Im}

    # SE empty
    elif pattern2.match(filename):
        Time, Re, Im, Amp = time_domain_correction(parent_directory, filename, correction)
        baseline[filename] = {'Time': Time, 'Amp': Amp, 'Re': Re, 'Im': Im}

# Water
Re_td_water_sub    = Re_td_water - Re_td_fid_empty

# Set names for empty and RE from SE 
filename_for_plot_se = 'Cellulose_500scnas_12gain_ 9_c.dat'
filename_for_plot_empty = 'Empty_500scnas_12gain_ 9_c.dat'
Time_se = np.array(measurement_files[filename_for_plot_se]['Time'])

# Subtract empty from FID cellulose, FID water and MSE for REAL && IMAG
Re_td_fid_sub      = Re_td_fid - Re_td_fid_empty
Re_td_mse_sub      = Re_td_mse - Re_td_fid_empty
Re_td_se_sub = np.array(measurement_files[filename_for_plot_se]['Re']) - np.array(baseline[filename_for_plot_empty]['Re'])

# Cut the beginning for amplitudes and real for FID and MSE
Time_fid, Re_td_fid_cut     = cut_beginning(Time_fid, Re_td_fid_sub)
Time_mse, Re_td_mse_cut     = cut_beginning(Time_mse, Re_td_mse_sub)
Time_se,  Re_td_se_cut      = cut_beginning(Time_se, Re_td_se_sub)

# Normalize RE of SE and MSE to FID's 60-70 microsec
Re_td_fid_norm = normalize_to_fid(Re_td_fid_cut, Re_td_fid_cut, Time_fid, Time_fid)
Re_td_se_norm = normalize_to_fid(Re_td_fid_cut, Re_td_se_cut, Time_fid, Time_se)
Re_td_mse_norm = normalize_to_fid(Re_td_fid_cut, Re_td_mse_cut, Time_fid, Time_mse)

# Subtract the long component for FID, SE and MSE REAL & IMAG
Re_td_fid_short    = reference_long_component(Time_fid, Re_td_fid_norm, end= 55)
Re_td_mse_short    = reference_long_component(Time_mse, Re_td_mse_norm, end= 55)
Re_td_se_short     = reference_long_component(Time_se, Re_td_se_norm, end= 55)

## PLOT FID, SE and MSE all together
Fr_FID, Re_FID,_ = freq_domain_correction(Time_fid, Re_td_fid_short, 0, False)
Fr_MSE, Re_MSE,_ = freq_domain_correction(Time_mse, Re_td_mse_short , 0, False)
Fr_SE, Re_SE,_ = freq_domain_correction(Time_se, Re_td_se_short, 0, False)

# 10. M2 & T2
M2_FID, T2_FID = calculate_M2(Re_FID, Fr_FID)
print(f'FID\nM2: {M2_FID}\nT2: {T2_FID}')
M2_FID, T2_FID = calculate_M2(Re_SE, Fr_SE)
print(f'SE\nM2: {M2_FID}\nT2: {T2_FID}')
M2_FID, T2_FID = calculate_M2(Re_MSE, Fr_MSE)
print(f'MSE\nM2: {M2_FID}\nT2: {T2_FID}')

#### SE part
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7))
cmap = plt.get_cmap('winter')

# Processing SE data
num_files = len(measurement_files)

# Amplitudes SE
for filename1, filename2 in zip(measurement_files, baseline):
    maximum, Time_cut, Amp_re_cut = analysis_SE(measurement_files, baseline, filename1, filename2, 'Re', 'Re_diff', Re_td_fid_cut, Time_fid)

    color = cmap(time_shift / num_files)
    time_shift += 1

    # Define label for legend
    match = pattern3.search(filename1)
    file_key = match.group(1)

#     ax1.plot(Time_cut, Amp_re_cut, label=file_key, color=color)
# ax1.plot(Time_fid, Re_td_fid_short, 'r', label='FID')
# ax1.set_xlim(-5, 80)
# ax1.legend(title="Echo time in μs")
# ax1.set_title('a)', loc='left')
# ax1.set_xlabel('Time, μs')
# ax1.set_ylabel('Amplitude')

# Gaussian fit for SE maximum amplitude
p1 = [10, 6, 1] # Initial guess
popt1, _ = curve_fit(gauss1, echo_time, maximum, p0=p1)
fitting_line = gauss1(echo_time_fit, *popt1)
extrapolation = fitting_line[0]

# Show SE decays and maxima fitting

# ax2.plot(echo_time, maximum, 'o', label='Max SE Amplitude')
# ax2.plot(echo_time_fit, fitting_line, 'r--', label='Gaussian Fit')
# ax2.plot(0, extrapolation, 'ro', label='Exrapolated to time=0')
# ax2.set_xlabel('Echo time, μs')
# ax2.set_ylabel('Amplitude max')
# ax2.set_title('b)', loc='left')
# plt.tight_layout()
# plt.show()


def build_up_fid(Time, Data, A):
    # Normalize FID to the amplitude A
    # 1. Cut the FID between 10 and 18 microsec
    start = 10
    finish = 20

    Time_cut    = Time[find_nearest(Time, start):find_nearest(Time, finish)]
    Data_cut    = Data[find_nearest(Time, start):find_nearest(Time, finish)]

    # # Fit the small part of the FID with gauss function with restricted amplitude
    # popt2, _ = curve_fit(gauss2(A), Time_cut, Data_cut, p0=[8, 0])
    # Data_built = gauss1(Time, A, *popt2)

    # # Fit the small part of the FID with polynom (4 degree)
    # popt2, _ = curve_fit(poly2(A), Time_cut, Data_cut, p0=[10, 0.005])  # Начальные приближения для остальных коэффициентов
    # Data_built = poly1(Time, A, *popt2)  # Восстановление данных

    popt2, _ = curve_fit(poly1, Time_cut, Data_cut)
    Data_built = poly1(Time, *popt2)  # Восстановление данных

    # Build-up the FID from time 0 to the first interception
    dif_t = Time_cut[1]-Time_cut[0]
    Time_build_from_zero = np.arange(0, start, dif_t)
    # Data_build_from_zero = gauss1(Time_build_from_zero, A, *popt2)
    Data_build_from_zero = gauss1(Time_build_from_zero, *popt2)


    # For polynomial fitting
    # Data_build_from_zero =  poly1(Time_build_from_zero, A, *popt) 

    # Build the data from 1 interception to 2d interception
    # Make an weighted average, where weight depends on X
    # So, in the beginning, no FID, all built ->0
    # In the end, only FID, no built ->1
    # Begin - where the start, end where is the finish
    start_idx = find_nearest(Time, start)
    finish_idx = find_nearest(Time, finish)
    Time_build_middle = Time[start_idx+1:finish_idx]
    length = len(Time_build_middle)
    data_fid = Data[start_idx+1:finish_idx]
    data_built = Data_built[start_idx+1:finish_idx]
    weight = np.linspace(0, 1, length)

    Data_build_middle = weight * data_fid + (1 - weight) * data_built

    # Build the data from 2d interception until the end
    Time_build_end  = Time[finish_idx+1:]
    Data_build_end   = Data[finish_idx+1:]

    Time_build_full = np.concatenate((Time_build_from_zero,Time_build_middle, Time_build_end))
    Data_build_full  = np.concatenate((Data_build_from_zero,Data_build_middle, Data_build_end))

    plt.plot(Time, Data_built, 'b')
    plt.plot(Time, Data, 'r')
    plt.show()

    return Time_build_full, Data_build_full, Data_built

# Build up FIDS from SE/MSE for Real
A_se = extrapolation
A_mse = np.max(Re_td_mse_norm)

Time_build_full_se_r, Re_build_full_se, Re_build_se = build_up_fid(Time_fid, Re_td_fid_short, A_se)
Time_build_full_mse_r, Re_build_full_mse, Re_build_mse = build_up_fid(Time_fid, Re_td_fid_short, A_mse)

# Calculate M2 of build-up Fids real
Frequency_buildupfid_SE_r, Real_buildupfid_SE_r, _      = freq_domain_correction(Time_build_full_se_r, Re_build_full_se, 0, False)
Frequency_buildupfid_MSE_r, Real_buildupfid_MSE_r, _    = freq_domain_correction(Time_build_full_mse_r, Re_build_full_mse, 0, True)

M2_FID_SE_r, T2_FID_SE_r    = calculate_M2(Real_buildupfid_SE_r, Frequency_buildupfid_SE_r)
M2_FID_MSE_r, T2_FID_MSE_r  = calculate_M2(Real_buildupfid_MSE_r, Frequency_buildupfid_MSE_r)

print(f'FID build-up with real SE:\nM2: {M2_FID_SE_r}\nT2: {T2_FID_SE_r}')
print(f'FID build-up with real MSE:\nM2: {M2_FID_MSE_r}\nT2: {T2_FID_MSE_r}')

# PLOT all figures here

# #plot FID, MSE and SE REAL
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.plot(Time_mse, Re_td_mse_short, 'k', label='MSE')
# ax1.plot(Time_se, Re_td_se_short, 'b', label='SE')
# ax1.plot(Time_fid, Re_td_fid_short, 'r', label='FID')
# ax1.set_xlim(-5, 80)
# ax1.set_title('a) NMR Signal', loc='left')
# ax1.set_xlabel('Time, μs')
# ax1.set_ylabel('Amplitude, a.u.')
# ax1.legend()

# ax2.plot(Fr_MSE, Re_MSE, 'k', label='MSE')
# ax2.plot(Fr_SE, Re_SE, 'b', label='SE')
# ax2.plot(Fr_FID, Re_FID, 'r', label='FID')
# ax2.set_xlim(-0.15,0.15)
# ax2.set_title('b) FFT spectra', loc='left')
# ax2.set_xlabel('Frequency, MHz')
# ax2.set_ylabel('Intensity, a.u.')
# ax2.legend()
# plt.tight_layout()
# plt.show()


## Build-up decys and spectra
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(Time_build_full_mse_r, Re_build_full_mse, 'k', label='FID from MSE')
ax1.plot(Time_build_full_se_r, Re_build_full_se, 'b', label='FID from SE')
ax1.plot(Time_fid, Re_td_fid_short, 'r', label='FID original')
ax1.set_xlim(-5, 80)
ax1.set_title('a) NMR Signal', loc='left')
ax1.set_xlabel('Time, μs')
ax1.set_ylabel('Amplitude, a.u.')
ax1.legend()


ax2.plot(Frequency_buildupfid_MSE_r, Real_buildupfid_MSE_r, 'k', label='MSE')
ax2.plot(Frequency_buildupfid_SE_r, Real_buildupfid_SE_r, 'b', label='SE')
ax2.plot(Fr_FID, Re_FID, 'r', label='FID')
ax2.set_xlim(-0.3,0.3)
ax2.set_title('b) FFT spectra', loc='left')
ax2.set_xlabel('Frequency, MHz')
ax2.set_ylabel('Intensity, a.u.')
ax2.legend()

plt.tight_layout()
plt.show()


## Save build-up data for export
df = pd.DataFrame({'Time' : Time_build_full_mse_r, 'Re': Re_build_full_mse})
df.to_csv("MSE_build_up.dat", sep ='\t', index = 'none')

df = pd.DataFrame({'Time' : Time_build_full_se_r, 'Re': Re_build_full_se})
df.to_csv("SE_build_up.dat", sep ='\t', index = 'none')



# ## Difference FID and SE real components

# # Setting time arrays to be the same length
# Time_to_start_from = Time_se[find_nearest(Time_fid[0], Time_se):]
# Time_to_finish_at = Time_fid[:find_nearest(Time_se[-1], Time_fid)+1]

# # Setting Re of FID and SE to be the same length
# # I need to cut the beginning of SE and the end of FID
# SE_re_difference = Re_td_se_short[find_nearest(Time_fid[0], Time_se):]
# FID_re_difference = Re_td_fid_short[:find_nearest(Time_se[-1], Time_fid)+1]
# Difference = FID_re_difference - SE_re_difference


# plt.plot(Time_to_start_from, Difference)
# plt.show()

# # Water
# water_cut_1 = find_nearest(Time_w, 10)
# water_cut_2 = find_nearest(Time_w, 30)
# Time_water_cut = Time_w[water_cut_1:water_cut_2]
# Amp_water_cut = Amp_w[water_cut_1:water_cut_2]
# Amp_water = np.mean(Amp_water_cut)
# Amp_cellu = popt1[0]

# plt.plot(Time_build_full, Amp_build_full, 'r', label='FID Built')
# plt.plot(Time_w, Amp_w, 'b', label='FID water')
# plt.plot(Time_water_cut, Amp_water_cut, 'c--', label='Mean')
# plt.xlabel('Time, μs')
# plt.ylabel('Amplitude, a.u.')
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Constants
# mass_water = 0.0963
# mass_cellu = 0.1334
# Avogadro_number= 6.022*(10**23)
# molar_mass_water = 18.01528
# molar_mass_cellu = 162.1406

# protons_water = (mass_water/molar_mass_water)*Avogadro_number*2
# protons_cellu = (mass_cellu/molar_mass_cellu)*Avogadro_number*10

# proton_density_water = Amp_water/protons_water
# proton_density_cellu = Amp_cellu/protons_cellu

# Amp_cellu_from_protondensity_water = proton_density_water*protons_cellu

# # Print results
# print(f'The amplitude of cellulose calculated from water is {Amp_cellu_from_protondensity_water}')

# print(f'Maximum amplitude from SE: {popt1[0]}')
# print(f'Maximum amplitude from MSE: {np.max(Amp_MSE)}')
# print(f'Maximum amplitude from FID: {popt[0]}')
print('done')
