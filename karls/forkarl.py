# This Python file uses the following encoding: utf-8
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def read_data(file_path):
    data = np.loadtxt(file_path)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    return x, y, z

def calculate_amplitude(Real, Imaginary):
    Amp = np.sqrt(Real ** 2 + Imaginary ** 2)
    return Amp

parent_directory = os.getcwd()
#user should choose it
pattern = re.compile(r'FID_(\d+).*.dat$')
Empty = r'FID_empty.dat'
Glycerol = r'FID_000_Glycerol.dat'

measurement_files = {}
baseline = {}

# read files, create dictionary
for filename in os.listdir(parent_directory):
    Time = []
    Re = []
    Im = []
    Amp = []
    try:
        #Read files
        file_path = os.path.join(parent_directory, filename)
        Time, Re_original, Im_original = read_data(file_path)

        #Correct phase
        R_phased, I_phased = time_domain_phase(Re_original, Im_original)
        #Adjust frequency
        Frequency = calculate_frequency_scale(Time)
        Re_shifted_1, Im_shifted_1 = adjust_frequency(Frequency, R_phased, I_phased)

        #Just to make sure do it again
        # 5 Again Phase
        Re_phased_2, Im_phased_2 = time_domain_phase(Re_shifted_1, Im_shifted_1)

        # Again frequency
        Re, Im = adjust_frequency(Frequency, Re_phased_2, Im_phased_2)
        Amp = calculate_amplitude(Re, Im)

        if pattern.match(filename):
            measurement_files[filename] = {
            'Time': Time,  
            'Amp' : Amp,
            'Re'  : Re,
            'Im'  : Im  
            }

        elif filename == Empty:
            baseline[filename] = {
            'Time': Time,  
            'Amp' : Amp,
            'Re'  : Re,
            'Im'  : Im  
            }
    except:
        pass

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

cmap = plt.get_cmap('winter')
i = 0
num_files = len(measurement_files)

# Plot the corrected signals
for filename in measurement_files:
    Time = np.array(measurement_files[filename]['Time'])
    Amplitude = np.array(measurement_files[filename]['Amp'])
    Re = np.array(measurement_files[filename]['Re'])
    Im = np.array(measurement_files[filename]['Im'])
    
    match = pattern.search(filename)
    file_key = match.group(1)

    color = cmap(i/num_files)
    i += 1

    ax1.plot(Time, Amplitude, label=file_key, color=color)
    ax1.set_xlabel('Time, μs')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Corrected Signals')
ax1.plot(baseline[Empty]['Time'], baseline[Empty]['Amp'], 'r--', label='Baseline')
ax1.legend()
i = 0

# Subtract baseline
for filename in measurement_files:
    Time = np.array(measurement_files[filename]['Time'])
    Amplitude = np.array(measurement_files[filename]['Amp'])
    Re = np.array(measurement_files[filename]['Re'])
    Im = np.array(measurement_files[filename]['Im'])
    
    Amplitude_baseline = np.array(baseline[Empty]['Amp'])
    
    if len(Amplitude) != len(Amplitude_baseline):
        Amplitude_baseline = Amplitude_baseline[:len(Amplitude)]

    match = pattern.search(filename)
    file_key = match.group(1)

    amp_difference = Amplitude - Amplitude_baseline
    measurement_files[filename]['Amp_diff'] = amp_difference

    color = cmap(i/num_files)
    i += 1

    ax2.plot(Time, amp_difference, label=file_key, color=color)
    ax2.legend()
    ax2.set_xlabel('Time, μs')
    ax2.set_ylabel('Amplitude Difference')
    ax2.set_title('Baseline Subtracted Signals')

plt.tight_layout()
plt.show()

export_data = {}
for filename, data in measurement_files.items():
    match = pattern.search(filename)
    file_key = match.group(1)
    
    export_data[file_key + ' Time'] = data['Time']
    export_data[file_key + ' Amplitude'] = data['Amp']
    export_data[file_key + ' Real'] = data['Re']
    export_data[file_key + ' Imaginary'] = data['Im']
    export_data[file_key + ' Amplitude_diff'] = data['Amp_diff']

export_data['Empty Time'] = baseline[Empty]['Time']
export_data['Empty Amplitude'] = baseline[Empty]['Amp']
export_data['Empty Real'] = baseline[Empty]['Re']
export_data['Empty Imaginary'] = baseline[Empty]['Im']

# Save to excel
df = pd.DataFrame(export_data)
output_file = os.path.join(parent_directory, 'Resulting_table.xlsx')

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Results', index=False)

print('done')
