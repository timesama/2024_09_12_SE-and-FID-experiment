# This Python file uses the following encoding: utf-8
import numpy as np
from scipy.optimize import curve_fit
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

# Read data from file
def read_data(file_path):
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1], data[:, 2]

# Calculate amplitude
def calculate_amplitude(real, imaginary):
    return np.sqrt(real**2 + imaginary**2)

# Directory and pattern setup
parent_directory = os.path.join(os.getcwd(), 'FID_Cycle')
pattern = re.compile(r'FID_Cellulose.*.dat$')
pattern2 = re.compile(r'FID_Empty.*.dat$')
pattern3 = re.compile(r'FID_Cellulose_.*_\s*(\d+)_c\.dat')

measurement_files = {}
baseline = {}

# Read part
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

for filename in os.listdir(parent_directory):
    if pattern.match(filename):
        file_path = os.path.join(parent_directory, filename)
        Time, Re_original, Im_original = read_data(file_path)
        #Correct phase
        R_phased, I_phased = time_domain_phase(Re_original, Im_original)
        #Adjust frequency
        Frequency = calculate_frequency_scale(Time)
        Re, Im = adjust_frequency(Frequency, R_phased, I_phased)
        Amp = calculate_amplitude(Re, Im)
        measurement_files[filename] = {'Time': Time, 'Amp': Amp}
    elif pattern2.match(filename):
        file_path = os.path.join(parent_directory, filename)
        Time, Re_original, Im_original = read_data(file_path)
        #Correct phase
        R_phased, I_phased = time_domain_phase(Re_original, Im_original)
        #Adjust frequency
        Frequency = calculate_frequency_scale(Time)
        Re, Im = adjust_frequency(Frequency, R_phased, I_phased)
        Amp = calculate_amplitude(Re, Im)
        baseline[filename] = {'Time': Time, 'Amp': Amp}

# Processing FID
time_shift = 0
echo_time = np.arange(7, 26)
maximum = []
cmap = plt.get_cmap('winter')
num_files = len(measurement_files)

for filename1, filename2 in zip(measurement_files, baseline):
    amp_measurement_files = np.array(measurement_files[filename1]['Amp'])
    amp_baseline = np.array(baseline[filename2]['Amp'])

    if len(amp_measurement_files) != len(amp_baseline):
        amp_baseline = amp_baseline[:len(amp_measurement_files)]

    amp_difference = amp_measurement_files - amp_baseline
    measurement_files[filename1]['Amp_diff'] = amp_difference

    shifted_time = np.array(measurement_files[filename1]['Time'])

    cut_idx = find_nearest(shifted_time, 20)
    maximum.append(max(amp_difference[:cut_idx]))

    match = pattern3.search(filename1)
    file_key = match.group(1)


    color = cmap(time_shift / num_files)
    time_shift += 1
    ax1.plot(shifted_time, amp_difference, label=file_key, color=color)

ax1.legend(title="Ringing time in μs")
ax1.set_xlabel('Time, μs')
ax1.set_ylabel('Amplitude')


ax2.plot(echo_time, maximum, 'o', label='Max FID Amplitude')
ax2.grid()
ax2.set_xlabel('Ringing time, μs')
ax2.set_ylabel('Amplitude max')
ax2.set_title('Max Amplitudes of FID')
plt.show()


print('done')
