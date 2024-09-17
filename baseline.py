# This Python file uses the following encoding: utf-8
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import os, re

def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss1(x, A, sigma):
    return A*np.exp(-x**2/(2.*sigma**2))

def read_data(file_path):
    data = np.loadtxt(file_path)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    return x, y, z

def calculate_amplitude(Real, Imaginary):
    Amp = np.sqrt(Real ** 2 + Imaginary ** 2)
    return Amp

parent_directory = r'C:\Mega\NMR\003_Temperature\2021_09_12_SE and FId experiment\5_19'
pattern = re.compile(r'Ibuprofen.*.dat$')
pattern2 = re.compile(r'Empty.*.dat$')
pattern3 = re.compile(r'Ibuprofen_\s*(\d+)_c\.dat')
pattern_FID = re.compile(r'FID_Ib.*.dat$')
pattern_FID_empty = re.compile(r'FID_Empty.*.dat$')

# parent_directory = r'C:\Mega\NMR\003_Temperature\2021_09_12_SE and FID experiment\cellulose\200 scans'
# pattern = re.compile(r'Cellulose.*.dat$')
# pattern2 = re.compile(r'Empty.*.dat$')
# pattern3 = re.compile(r'Cellulose_\s*(\d+)_c\.dat')

# pattern_FID = re.compile(r'FID_C.*.dat$')
# pattern_FID_empty = re.compile(r'FID_Empty.*.dat$')


measurement_files = {}
baseline = {}

#FID part
for filename in os.listdir(parent_directory):
    Time = []
    Signal = []
    if pattern_FID.match(filename):
        file_path_f = os.path.join(parent_directory, filename)
        Time_f, Re_f, Im_f = read_data(file_path_f)
        Amp_f = calculate_amplitude(Re_f, Im_f)

    elif pattern_FID_empty.match(filename):
        file_path_fe = os.path.join(parent_directory, filename)
        Time_fe, Re_fe, Im_fe = read_data(file_path_fe)
        Amp_fe = calculate_amplitude(Re_fe, Im_fe)

Amp_FID = Amp_f-Amp_fe

# SE part
plt.figure()
for filename in os.listdir(parent_directory):
    Time = []
    Signal = []
    if pattern.match(filename):
        file_path = os.path.join(parent_directory, filename)
        Time, Re, Im = read_data(file_path)
        Amp = calculate_amplitude(Re, Im)

        file_key = filename.replace('_c.dat', '')

        measurement_files[filename] = {
        'Time': Time,  
        'Amp' : Amp    
        }

    elif pattern2.match(filename):
        file_path = os.path.join(parent_directory, filename)
        Time, Re, Im = read_data(file_path)
        Amp = calculate_amplitude(Re, Im)

        file_key = filename.replace('_c.dat', '')

        baseline[filename] = {
        'Time': Time,  
        'Amp' : Amp    
        }

time_shift = 0
echo_time = np.arange(5,20)
maximum = []

cmap = plt.get_cmap('winter')
num_files = len(measurement_files)

for (filename1, filename2) in zip(measurement_files, baseline):
    amp_measurement_files = np.array(measurement_files[filename1]['Amp'])
    amp_baseline =  np.array(baseline[filename2]['Amp'])
    
    amp_difference = amp_measurement_files - amp_baseline
    
    measurement_files[filename1]['Amp_diff'] = amp_difference
    maximum.append(max(amp_difference))
    match = pattern3.search(filename1)
    file_key = match.group(1)

    shifted_time = np.array(measurement_files[filename1]['Time']) #+ time_shift

    color = cmap(time_shift / num_files)
    time_shift+=1

    # plt.plot(shifted_time, amp_difference, label = file_key, color=color)
    # plt.legend(title="Echo time in μs")
    # plt.xlabel('Time, μs')
    # plt.ylabel('Amplitude')
# plt.plot(Time_f, Amp_FID, 'r', label='FID')
# plt.legend()
# plt.show()

# Extrapolating to the time zero to find the true amplitude of the FID or so
# Gaus fittin in the whole range
p1 = [55, 18]
popt1, _= curve_fit(gauss1, echo_time, maximum, p0 = p1)
extrapolation = popt1[0]
echo_time_fit = np.arange(0,19,0.001)
fitting_line = gauss1(echo_time_fit, *popt1)

# Linear fitting in the linear range
# fitting = np.polyfit(echo_time[0:4], maximum[0:4], 1)
# extrapolation = fitting[1]
# fitting_line = echo_time[0:4]*fitting[0] + fitting[1] 

plt.plot(echo_time, maximum, 'o')
plt.plot(0, extrapolation, 'ro')
plt.plot(echo_time_fit, fitting_line, 'r--')
plt.xlabel('Echo time, μs')
plt.ylabel('Amplitude max')
plt.show()

# Normalize FID to the amplitude SE at t=0
time_shift = 0
for filename1 in measurement_files:
    amp_difference = np.array(measurement_files[filename1]['Amp_diff'])
    shifted_time = np.array(measurement_files[filename1]['Time']) #+ time_shift

    match = pattern3.search(filename1)
    file_key = match.group(1)

    color = cmap(time_shift / num_files)

    # Shift SE to the max amlitude = zero time
    cut_max_idx = np.argmax(amp_difference)

    plt.plot(shifted_time[:-cut_max_idx], amp_difference[cut_max_idx:], '--', label = file_key, color=color)
    plt.legend(title="Echo time in μs")
    plt.xlabel('Time, μs')
    plt.ylabel('Amplitude')

    time_shift+=1

# Cut everything left than max in FID
cut_idx = np.argmax(Amp_FID)
Time_cut = Time_f[cut_idx:]
Amp_cut = Amp_FID[cut_idx:]

# fit the FID with gauss
p = [max(Amp_cut), min(Time_cut), 18]
popt, _= curve_fit(gauss, Time_cut, Amp_cut, p0 = p)

Time_fit = np.arange(0,100,0.1)
AMP_fit = gauss(Time_fit, *popt)

coeff = extrapolation/max(AMP_fit)
Amp_n = coeff * AMP_fit
plt.plot(Time_fit, Amp_n, 'r', label='FID normalized')
plt.plot(Time_fit, AMP_fit, 'k', label='fitting')
plt.plot(Time_f, Amp_FID, 'm', label='Original')
plt.legend(title="Echo time in μs")
plt.show()

print(r'Maximum amplitude from SE ', popt1[0])
print(r'Maximum amplitude from FID ', popt[0])

print('donedone')    
print('done')
