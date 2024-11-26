# This Python file uses the following encoding: utf-8
import numpy as np
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

    # Shift the spectra (amplitude) by the difference in indices
    FFT_shifted = np.concatenate((FFT[delta_index:], FFT[:delta_index]))

    # iFFT
    Fid_shifted = np.fft.ifft(np.fft.fftshift(FFT_shifted))

    # Define Real, Imaginary and Amplitude
    Re_shifted = np.real(Fid_shifted)
    Im_shifted = np.imag(Fid_shifted)

    return Re_shifted, Im_shifted

# Find nearest value in array
def find_nearest(array, value):
    idx = (np.abs(np.asarray(array) - value)).argmin()
    return idx

# Gaussian functions
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def gauss1(x, A, sigma):
    return A * np.exp(-x**2 / (2 * sigma**2))

def gauss2(A_const):
    def gauss_const(x, sigma):
        return gauss1(x, A_const, sigma)
    return gauss_const

def decaying_exponential(x, a, b, c):
    return a * np.exp(-x/b) + c

# Read data from file
def read_data(file_path):
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1], data[:, 2]

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
    Im_zero = np.concatenate((Imaginary, amount_to_add))

    dt = Time[1] - Time[0]
    Time_to_add = Time[-1] + np.arange(1, length_diff + 1) * dt

    Time = np.concatenate((Time, Time_to_add))
    Fid = np.array(Re_zero + 1j * Im_zero)
    Fid = Fid[:-1]

    return Time, Fid

def apodization(Time, Real, Imaginary):
    Amplitude = calculate_amplitude(Real, Imaginary)
    sigma = 70
    # coeffs = np.polyfit(Time, Amplitude, 1)  # Fit an exponential decay function
    # c = np.polyval(coeffs, Time)
    # d = np.argmin(np.abs(c - 3e-5))
    # sigma = Time[d]
    # if sigma == 0:
    #     sigma = 1000
    apodization_function = np.exp(-(Time / sigma) ** 6)
    Re_ap = Real * apodization_function
    Im_ap = Imaginary * apodization_function

    # plt.plot(Time, apodization_function, 'r--', label='apodization')
    # plt.plot(Time, Imaginary, 'b', label='Im')
    # plt.plot(Time, Real, 'k', label='Re')
    # plt.plot(Time, Re_ap, 'k--', label='Re ap')
    # plt.plot(Time, Im_ap, 'b--', label='Im ap')
    # plt.xlim([-5,80])
    # plt.xlabel('Time, μs')
    # plt.ylabel('Amplitude, a.u.')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    return Re_ap, Im_ap

def create_spectrum(Time, Real, Imaginary):
    number_of_points = 2**16

    # 5. Apodize the time-domain
    Re_ap, Im_ap = apodization(Time, Real, Imaginary)
    Fr = calculate_frequency_scale(Time)

    Re_ph, Im_ph = time_domain_phase(Re_ap, Im_ap)
    Re_ad, Im_ad = adjust_frequency(Fr, Re_ph, Im_ph)
    
    # 6. Add zeros
    Time_zero, Fid_zero = add_zeros(Time, Re_ad, Im_ad, number_of_points)

    Frequency = calculate_frequency_scale(Time_zero)

    # 7. FFT
    FFT = np.fft.fftshift(np.fft.fft(Fid_zero))

    # 8. Simple baseline
    _, Re, Im = simple_baseline_correction(FFT)

    # 9. Apodization
    Real_apod = calculate_apodization(Re, Frequency)

    return Frequency, Real_apod, Im

def adjust_spectrum (Time, Re, Im):
    Frequency, Real, _ = create_spectrum(Time, Re, Im)
    # shift = Frequency[np.argmax(Real)]
    # Frequency = Frequency - shift
    return Frequency, Real

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
            # plt.title(filename)
            # plt.tight_layout()
            # plt.show()

        else:
            Re = Re_original
            Im = Im_original
            Amp = calculate_amplitude(Re_original, Im_original)

        return Time, Re, Im, Amp

def cut_beginning(Time, Data):
    Time_plot = Time[np.argmax(Data):]
    Data_plot = Data[np.argmax(Data):]
    return Time_plot, Data_plot

def analysis_SE(measurement_files, baseline, filename1, filename2, type, type_save):
    # read data
    Time = np.array(measurement_files[filename1]['Time'])
    data_measurement_files = np.array(measurement_files[filename1][type])
    data_baseline = np.array(baseline[filename2][type])
    # compare length
    if len(data_measurement_files) != len(amp_baseline):
        amp_baseline = amp_baseline[:len(data_measurement_files)]
    # subtract
    difference = data_measurement_files - data_baseline
    # save subtracted data and maximum
    measurement_files[filename1][type_save] = difference
    maximum.append(max(difference))
    Time_cut, Data_cut = cut_beginning(Time, difference)

    return maximum, Time_cut, Data_cut

def normalize_to_fid(Fid, Data, Time_fid, Time_data):
    Mean_amp_fid_long = np.mean(Fid[find_nearest(Time_fid, 60):find_nearest(Time_fid, 70)])
    Mean_amp_dat_long = np.mean(Data[find_nearest(Time_data, 60):find_nearest(Time_data, 70)])

    difference = Mean_amp_dat_long-Mean_amp_fid_long
    Normalized_amp = Data -difference

    # the division doesn't work very good
    # proportionality_coefficient = Mean_amp_fid_long /Mean_amp_dat_long
    # Normalized_amp = Data * proportionality_coefficient
    return Normalized_amp

def reference_long_component(Time, Component_n, end):
    # 3. Cut the ranges for fitting
    minimum = find_nearest(Time, end)

    Time_range = Time[minimum:]
    Component_n_range = Component_n[minimum:]

    # Smooth data
    Smooth = savgol_filter(Component_n_range, 30, 0)

    p = [5, 30, 0.5]
    # 7. Fit data to exponential decay
    popt, _      = curve_fit(decaying_exponential, Time_range, Smooth, p0 =p)
    
    # 9. Calculate the curves fitted to data within the desired range
    Component_f = decaying_exponential(Time, *popt)

    # 10. Subtract
    Component_sub = Component_n - Component_f


    # For Debug
    plt.plot(Time, Component_n, 'r', label='Original')
    plt.plot(Time, Component_sub, 'b', label='Subtracted')
    plt.plot(Time, Component_f, 'k--', label='Fitted')
    plt.xlabel('Time, μs')
    plt.ylabel('Amplitude, a.u.')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return Component_sub

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

    # FID
    if pattern_FID.match(filename):
        correction = False
        Time_fid, Re_td_fid, Im_td_fid, Amp_td_fid = prepare_data(parent_directory, filename, correction)

    # MSE
    elif pattern_MSE.match(filename):
        correction = True
        Time_mse, Re_td_mse, Im_td_mse, Amp_td_mse = prepare_data(parent_directory, filename, correction)
        # UNFORTUNATELY I recorded a very long MSE, my bad
        length_to_cut = len(Time_fid)
        Time_mse      = Time_mse[:length_to_cut]
        Re_td_mse     = Re_td_mse[:length_to_cut]
        Im_td_mse     = Im_td_mse[:length_to_cut]
        Amp_td_mse    = Amp_td_mse[:length_to_cut]

    # Empty
    elif pattern_FID_empty.match(filename):
        correction = False
        Time_fid_empty, Re_td_fid_empty, Im_td_fid_empty, Amp_td_fid_empty = prepare_data(parent_directory, filename, correction)
    
    # Water
    elif pattern_FID_water.match(filename):
        correction = True
        Time_water, Re_td_water, Im_td_water, Amp_td_water = prepare_data(parent_directory, filename, correction)
        # UNFORTUNATELY I recorded a very long FID for water, my bad
        length_to_cut = len(Time_fid)
        Time_water      = Time_water[:length_to_cut]
        Re_td_water     = Re_td_water[:length_to_cut]
        Im_td_water     = Im_td_water[:length_to_cut]
        Amp_td_water    = Amp_td_water[:length_to_cut]

    # SE
    elif pattern.match(filename):
        correction = False
        Time, Re, Im, Amp = prepare_data(parent_directory, filename, correction)
        measurement_files[filename] = {'Time': Time, 'Amp': Amp, 'Re': Re, 'Im': Im}

    # SE empty
    elif pattern2.match(filename):
        Time, Re, Im, Amp = prepare_data(parent_directory, filename, correction)
        baseline[filename] = {'Time': Time, 'Amp': Amp, 'Re': Re, 'Im': Im}

# Subtract empty from FID cellulose, FID water and MSE for AMPLITUDES
Amp_td_fid_sub      = Amp_td_fid - Amp_td_fid_empty
Amp_td_water_sub    = Amp_td_water - Amp_td_fid_empty
Amp_td_mse_sub      = Amp_td_mse - Amp_td_fid_empty

# Subtract empty from FID cellulose, FID water and MSE for REAL && IMAG
Re_td_fid_sub      = Re_td_fid - Re_td_fid_empty
Re_td_water_sub    = Re_td_water - Re_td_fid_empty
Re_td_mse_sub      = Re_td_mse - Re_td_fid_empty

Im_td_fid_sub      = Im_td_fid - Im_td_fid_empty
Im_td_water_sub    = Im_td_water - Im_td_fid_empty
Im_td_mse_sub      = Im_td_mse - Im_td_fid_empty

# Subtract empty from SE & cut the beginning for AMPLITUDES 
filename_for_plot_se = 'Cellulose_500scnas_12gain_ 9_c.dat'
filename_for_plot_empty = 'Empty_500scnas_12gain_ 9_c.dat'
Amp_td_se_sub = np.array(measurement_files[filename_for_plot_se]['Amp']) - np.array(baseline[filename_for_plot_empty]['Amp'])
Time_se = np.array(measurement_files[filename_for_plot_se]['Time'])

# Subtract empty from SE & cut the beginning for Real
Re_td_se_sub = np.array(measurement_files[filename_for_plot_se]['Re']) - np.array(baseline[filename_for_plot_empty]['Re'])
Im_td_se_sub = np.array(measurement_files[filename_for_plot_se]['Im']) - np.array(baseline[filename_for_plot_empty]['Im'])

# Normalize amplitudes of SE and MSE to FID's 60-70 microsec
Amp_td_se_norm = normalize_to_fid(Amp_td_fid_sub, Amp_td_se_sub, Time_fid, Time_se)
Amp_td_mse_norm = normalize_to_fid(Amp_td_fid_sub, Amp_td_mse_sub, Time_fid, Time_mse)

Re_td_se_norm = normalize_to_fid(Re_td_fid_sub, Re_td_se_sub, Time_fid, Time_se)
Re_td_mse_norm = normalize_to_fid(Re_td_fid_sub, Re_td_mse_sub, Time_fid, Time_mse)
Im_td_se_norm = normalize_to_fid(Im_td_fid_sub, Im_td_se_sub, Time_fid, Time_se)
Im_td_mse_norm = normalize_to_fid(Im_td_fid_sub, Im_td_mse_sub, Time_fid, Time_mse)

# Subtract the long component for FID, SE and MSE AMplitudese
Amp_fid_td_short    = reference_long_component(Time_fid, Amp_td_fid_sub, end= 50)
Amp_mse_td_short    = reference_long_component(Time_mse, Amp_td_mse_norm, end= 50)
Amp_se_td_short     = reference_long_component(Time_se, Amp_td_se_norm, end= 80)

# Subtract the long component for FID, SE and MSE REAL & IMAG
Re_fid_td_short    = reference_long_component(Time_fid, Re_td_fid_sub, end= 50)
Re_mse_td_short    = reference_long_component(Time_mse, Re_td_mse_norm, end= 50)
Re_se_td_short     = reference_long_component(Time_se, Re_td_se_norm, end= 80)

# Im_fid_td_short    = reference_long_component(Time_fid, Im_td_fid_sub)
# Im_mse_td_short    = reference_long_component(Time_mse, Im_td_mse_norm)
# Im_se_td_short     = reference_long_component(Time_se, Im_td_se_norm)

# Cut the beginning for amplitudes and real for FID and MSE
Time_FID_plot, Amp_FID_plot = cut_beginning(Time_fid, Amp_fid_td_short)
_, Re_FID_plot = cut_beginning(Time_fid, Re_fid_td_short)

Time_MSE_plot, Amp_MSE_plot = cut_beginning(Time_mse, Amp_mse_td_short)
Time_MSE_plot_Re, Re_MSE_plot = cut_beginning(Time_mse, Re_mse_td_short)

Time_SE_plot, Amp_SE_plot = cut_beginning(Time_se, Amp_se_td_short)
Time_SE_plot_Re, Re_SE_plot = cut_beginning(Time_se, Re_se_td_short)

## PLOT FID, SE and MSE all together
# comparison of FID, SE and MSE
Fr_FID, Re_FID = adjust_spectrum(Time_fid, Re_fid_td_short, 0)
Fr_SE, Re_SE = adjust_spectrum(Time_se, Re_se_td_short, 0)
Fr_MSE, Re_MSE = adjust_spectrum(Time_mse, Re_mse_td_short , 0)

# 10. M2 & T2
M2_FID, T2_FID = calculate_M2(Re_FID, Fr_FID)
print(f'FID\nM2: {M2_FID}\nT2: {T2_FID}')
M2_FID, T2_FID = calculate_M2(Re_SE, Fr_SE)
print(f'SE\nM2: {M2_FID}\nT2: {T2_FID}')
M2_FID, T2_FID = calculate_M2(Re_MSE, Fr_MSE)
print(f'MSE\nM2: {M2_FID}\nT2: {T2_FID}')

#plot FID, MSE and SE AMPLITUDES
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(Time_FID_plot, Amp_FID_plot, 'r', label='FID')
ax1.plot(Time_MSE_plot, Amp_MSE_plot, 'k', label='MSE')
ax1.plot(Time_SE_plot, Amp_SE_plot, 'b', label='SE')
ax1.set_xlim(-5, 80)
ax1.set_title('a)', loc='left')
ax1.set_xlabel('Time, μs')
ax1.set_ylabel('Amplitude, a.u.')

ax2.plot(Fr_FID, Re_FID, 'r', label='FID')
ax2.plot(Fr_MSE, Re_MSE, 'k', label='MSE')
ax2.plot(Fr_SE, Re_SE, 'b', label='SE')
ax2.set_xlim(-0.07,0.070)
ax2.set_title('b)', loc='left')
ax2.set_xlabel('Frequency, MHz')
ax2.set_ylabel('Intensity, a.u.')
plt.tight_layout()
plt.show()

#plot FID, MSE and SE REAL
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(Time_FID_plot, Re_FID_plot, 'r', label='FID')
ax1.plot(Time_MSE_plot_Re, Re_MSE_plot, 'k', label='MSE')
ax1.plot(Time_SE_plot_Re, Re_SE_plot, 'b', label='SE')
ax1.set_xlim(-5, 80)
ax1.set_title('a)', loc='left')
ax1.set_xlabel('Time, μs')
ax1.set_ylabel('Amplitude, a.u.')
ax1.legend()

ax2.plot(Fr_FID, Re_FID, 'r', label='FID')
ax2.plot(Fr_MSE, Re_MSE, 'k', label='MSE')
ax2.plot(Fr_SE, Re_SE, 'b', label='SE')
ax2.set_xlim(-0.07,0.070)
ax2.set_title('b)', loc='left')
ax2.set_xlabel('Frequency, MHz')
ax2.set_ylabel('Intensity, a.u.')
ax2.legend()
plt.tight_layout()
plt.show()


# SE part
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 7))

# Processing SE data
cmap = plt.get_cmap('winter')
num_files = len(measurement_files)

# Amplitudes SE
for filename1, filename2 in zip(measurement_files, baseline):
    maximum, Time_cut, Amp_cut = analysis_SE(measurement_files, baseline, filename1, filename2, 'Amp', 'Amp_diff')

    color = cmap(time_shift / num_files)
    time_shift += 1

    # Define label for legend
    match = pattern3.search(filename1)
    file_key = match.group(1)

    ax1.plot(Time_cut, Amp_cut, label=file_key, color=color)

ax1.plot(Time_FID_plot, Amp_FID_plot, 'r', label='FID')
ax1.set_xlim(-5, 80)
ax1.legend(title="Echo time in μs")
ax1.set_title('a)', loc='left')
ax1.set_xlabel('Time, μs')
ax1.set_ylabel('Amplitude')

# Gaussian fit for SE maximum amplitude
p1 = [10, 6] # Initial guess
popt1, _ = curve_fit(gauss1, echo_time, maximum, p0=p1)
fitting_line = gauss1(echo_time_fit, *popt1)
extrapolation = fitting_line[0]

ax2.plot(echo_time, maximum, 'o', label='Max SE Amplitude')
ax2.plot(echo_time_fit, fitting_line, 'r--', label='Gaussian Fit')
ax2.plot(0, extrapolation, 'ro', label='Exrapolated to time=0')
ax2.set_xlabel('Echo time, μs')
ax2.set_ylabel('Amplitude max')
ax2.set_title('b)', loc='left')
plt.tight_layout()
plt.show()



# # Normalize FID to the amplitude SE at t=0
# time_shift = 0
# for filename1 in measurement_files:
#     amp_difference = np.array(measurement_files[filename1]['Amp_diff'])
#     shifted_time = np.array(measurement_files[filename1]['Time'])

#     match = pattern3.search(filename1)
#     file_key = match.group(1)

#     color = cmap(time_shift / num_files)
#     cut_max_idx = np.argmax(amp_difference)

#     # plt.plot(shifted_time[:-cut_max_idx] + abs(min(shifted_time)), amp_difference[cut_max_idx:], '--', label=file_key, color=color)
#     # plt.legend(title="Echo time in μs")
#     # plt.xlabel('Time, μs')
#     # plt.ylabel('Amplitude, a.u.')

#     time_shift += 1

# # FID fitting and building
# cut_idx = find_nearest(Time_f, 10)
# cut2_idx = find_nearest(Time_f, 18)
# Time_cut = Time_f[cut_idx:cut2_idx]
# Amp_cut = Amp_FID[cut_idx:cut2_idx]

# p = [max(Amp_cut), 18]
# popt, _ = curve_fit(gauss1, Time_cut, Amp_cut, p0=p)

# Time_fit = np.arange(0, 100, 0.1)
# AMP_fit = gauss1(Time_fit, *popt)
# coeff = extrapolation / max(AMP_fit)
# Amp_n = coeff * AMP_fit

# A_const = extrapolation
# popt2, _ = curve_fit(gauss2(A_const), Time_cut, Amp_cut, p0=[8])
# A_built = gauss1(Time_f, A_const, popt2[0])

# # Find intersections
# diff = A_built - Amp_FID
# sign_changes = np.where(np.diff(np.sign(diff)))[0]
# intersection_times = []
# intersection_amps = []
# intersection_idxs = []

# for idx in sign_changes:
#     t1, t2 = Time_f[idx], Time_f[idx + 1]
#     y1, y2 = diff[idx], diff[idx + 1]
#     t_intersection = t1 - y1 * (t2 - t1) / (y2 - y1)
#     intersection_times.append(t_intersection)
#     amp_intersection = A_built[idx] + (A_built[idx + 1] - A_built[idx]) * (t_intersection - t1) / (t2 - t1)
#     intersection_amps.append(amp_intersection)
#     intersection_idxs.append(idx)

# # # plot the fitted and original FID
# # plt.plot(Time_f, A_built, 'r', label='FID Built')
# # plt.plot(Time_FID_plot, Amp_FID_plot, 'm', label='Original')
# # plt.scatter(intersection_times, intersection_amps, color='blue', zorder=5, label='Intersections')
# # plt.xlabel('Time, μs')
# # plt.ylabel('Amplitude, a.u.')
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

# # Build-up the FID
# Time_build_from_zero = np.arange(0, intersection_times[0], 0.1)
# Amp_build_from_zero = gauss1(Time_build_from_zero, A_const, popt2[0])

# Time_build_end = Time_f[intersection_idxs[0]:]
# Amp_build_end = Amp_FID[intersection_idxs[0]:]

# Time_build_full = np.concatenate((Time_build_from_zero, Time_build_end))
# Amp_build_full = np.concatenate((Amp_build_from_zero, Amp_build_end))

# # plt.plot(Time_build_full, Amp_build_full, 'r', label='FID Built')
# # plt.plot(Time_FID_plot, Amp_FID_plot, 'm--', label='Original')
# # plt.xlabel('Time, μs')
# # plt.ylabel('Amplitude, a.u.')
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

# # Build-up the FID from MSE
# # FID fitting and building
# A_mse = np.max(Amp_MSE)
# popt2, _ = curve_fit(gauss2(A_mse), Time_cut, Amp_cut, p0=[8])
# A_built = gauss1(Time_f, A_mse, popt2[0])

# # Find intersections
# diff = A_built - Amp_FID
# sign_changes = np.where(np.diff(np.sign(diff)))[0]
# intersection_times = []
# intersection_amps = []
# intersection_idxs = []

# for idx in sign_changes:
#     t1, t2 = Time_f[idx], Time_f[idx + 1]
#     y1, y2 = diff[idx], diff[idx + 1]
#     t_intersection = t1 - y1 * (t2 - t1) / (y2 - y1)
#     intersection_times.append(t_intersection)
#     amp_intersection = A_built[idx] + (A_built[idx + 1] - A_built[idx]) * (t_intersection - t1) / (t2 - t1)
#     intersection_amps.append(amp_intersection)
#     intersection_idxs.append(idx)

# # plot the fitted and original FID
# plt.plot(Time_f, A_built, 'r', label='FID Built')
# plt.plot(Time_FID_plot, Amp_FID_plot, 'm', label='Original')
# plt.scatter(intersection_times, intersection_amps, color='blue', zorder=5, label='Intersections')
# plt.xlabel('Time, μs')
# plt.ylabel('Amplitude, a.u.')
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Build-up the FID
# Time_build_from_zero = np.arange(0, intersection_times[0], 0.1)
# Amp_build_from_zero = gauss1(Time_build_from_zero, A_mse, popt2[0])

# Time_build_end = Time_f[intersection_idxs[0]:]
# Amp_build_end = Amp_FID[intersection_idxs[0]:]

# Time_build_full_mse = np.concatenate((Time_build_from_zero, Time_build_end))
# Amp_build_full_mse = np.concatenate((Amp_build_from_zero, Amp_build_end))

# plt.plot(Time_build_full_mse, Amp_build_full_mse, 'r', label='FID Built')
# plt.plot(Time_FID_plot, Amp_FID_plot, 'm--', label='Original')
# plt.xlabel('Time, μs')
# plt.ylabel('Amplitude, a.u.')
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Calculate M2 of build-up Fids
# Frequency_buildupfid_SE, Real_buildupfid_SE, _ = create_spectrum(Time_build_full, Amp_build_full, 0)
# Frequency_buildupfid_MSE, Real_buildupfid_MSE, _ = create_spectrum(Time_build_full_mse, Amp_build_full_mse, 0)

# M2_FID_SE, T2_FID_SE = calculate_M2(Real_buildupfid_SE, Frequency_buildupfid_SE)
# M2_FID_MSE, T2_FID_MSE = calculate_M2(Real_buildupfid_MSE, Frequency_buildupfid_MSE)

# print(f'FID build-up with SE:\nM2: {M2_FID_SE}\nT2: {T2_FID_SE}')
# print(f'FID build-up with MSE:\nM2: {M2_FID_MSE}\nT2: {T2_FID_MSE}')

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
# print('done')
