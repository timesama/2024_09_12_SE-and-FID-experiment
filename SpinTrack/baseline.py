# This Python file uses the following encoding: utf-8
import numpy as np
from scipy.optimize import curve_fit, fsolve
import matplotlib.pyplot as plt
import os, re

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def gauss(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def gauss1(x, A, sigma):
    return A*np.exp(-x**2/(2.*sigma**2))

def gauss2(A_const):
    def gauss_const(x, sigma):
        return gauss1(x, A_const, sigma)
    return gauss_const

def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x : fun1(x) - fun2(x),x0)

# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

# def wrapperfunc(a_test):
#     def tempfunc(x, b, c, a=a_test):
#         return func(x, a, b, c)
#     return tempfunc


def read_data(file_path):
    data = np.loadtxt(file_path)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    return x, y, z

def calculate_amplitude(Real, Imaginary):
    Amp = np.sqrt(Real ** 2 + Imaginary ** 2)
    return Amp

# parent_directory = r'C:\Mega\NMR\003_Temperature\2021_09_12_SE and FId experiment\Ibuprofen\5_19'
# pattern = re.compile(r'Ibuprofen.*.dat$')
# pattern2 = re.compile(r'Empty.*.dat$')
# pattern3 = re.compile(r'Ibuprofen_\s*(\d+)_c\.dat')
# pattern_FID = re.compile(r'FID_Ib.*.dat$')
# pattern_FID_empty = re.compile(r'FID_Empty.*.dat$')

parent_directory = r'C:\Mega\NMR\003_Temperature\2021_09_12_SE and FID experiment\SPINMATE\silence 120'
pattern = re.compile(r'Cellulose.*.dat$')
pattern2 = re.compile(r'Empty.*.dat$')
pattern3 = re.compile(r'Cellulose.*_\s*(\d+)_c\.dat')

pattern_FID = re.compile(r'FID_C.*.dat$')
pattern_FID_empty = re.compile(r'FID_Empty.*.dat$')


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
echo_time = np.arange(9,26)
maximum = []

cmap = plt.get_cmap('winter')
num_files = len(measurement_files)

for (filename1, filename2) in zip(measurement_files, baseline):
    amp_measurement_files = np.array(measurement_files[filename1]['Amp'])
    amp_baseline =  np.array(baseline[filename2]['Amp'])
    
    if len(amp_measurement_files)!=len(amp_baseline):
       amp_baseline =  amp_baseline[:len(amp_measurement_files)]

    amp_difference = amp_measurement_files - amp_baseline
    
    measurement_files[filename1]['Amp_diff'] = amp_difference
    maximum.append(max(amp_difference))
    match = pattern3.search(filename1)
    file_key = match.group(1)

    shifted_time = np.array(measurement_files[filename1]['Time']) #+ time_shift

    color = cmap(time_shift / num_files)
    time_shift+=1

    plt.plot(shifted_time, amp_difference, label = file_key, color=color)
    plt.legend(title="Echo time in μs")
    plt.xlabel('Time, μs')
    plt.ylabel('Amplitude')
plt.plot(Time_f, Amp_FID, 'r', label='FID')
plt.legend()
plt.show()

# plt.plot(echo_time, maximum, 'or')
# plt.legend()
# plt.show()

# Extrapolating to the time zero to find the true amplitude of the FID or so
# Gaus fittin in the whole range
p1 = [10, 6]
# popt1, _= curve_fit(gauss, echo_time, maximum, p0 = p1)
popt1, _= curve_fit(gauss1, echo_time, maximum, p0=p1)
# extrapolation = popt1[0]
echo_time_fit = np.arange(0,25,0.001)
fitting_line = gauss1(echo_time_fit, *popt1)

# The maximum SE is at time = 0, not the coefficient
extrapolation = fitting_line[0]

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

    plt.plot(shifted_time[:-cut_max_idx]+ abs(min(shifted_time)), amp_difference[cut_max_idx:], '--', label = file_key, color=color)
    plt.legend(title="Echo time in μs")
    plt.xlabel('Time, μs')
    plt.ylabel('Amplitude')

    time_shift+=1

Time_f = Time_f +0
# Cut everything left than max in FID
cut_idx = find_nearest(Time_f, 10)
cut2_idx = find_nearest(Time_f, 18)
Time_cut = Time_f[cut_idx:cut2_idx]
Amp_cut = Amp_FID[cut_idx:cut2_idx]

# fit the FID with gauss
# p = [max(Amp_cut), 0, 18]
# popt, _= curve_fit(gauss, Time_cut, Amp_cut, p0 = p)
p = [max(Amp_cut),18]
popt, _= curve_fit(gauss1, Time_cut, Amp_cut, p0=p)

Time_fit = np.arange(0,100,0.1)

AMP_fit = gauss1(Time_fit, *popt)
coeff = extrapolation/max(AMP_fit)
Amp_n = coeff * AMP_fit

A_const = extrapolation
popt2, _= curve_fit(gauss2(A_const), Time_cut, Amp_cut, p0 = [8])
A_built = gauss1(Time_f, A_const, popt2[0])

plt.plot(Time_f, A_built, 'r', label='FID Built')
plt.plot(Time_f, Amp_FID, 'm', label='Original')
plt.legend(title="Echo time in μs")
plt.show()

# Find the intersection
diff = A_built - Amp_FID
sign_changes = np.where(np.diff(np.sign(diff)))[0]
intersection_times = []
intersection_amps = []
intersection_idxs = []

for idx in sign_changes:
    t1, t2 = Time_f[idx], Time_f[idx + 1]
    y1, y2 = diff[idx], diff[idx + 1]
    t_intersection = t1 - y1 * (t2 - t1) / (y2 - y1)
    intersection_times.append(t_intersection)
    amp_intersection = A_built[idx] + (A_built[idx + 1] - A_built[idx]) * (t_intersection - t1) / (t2 - t1)
    intersection_amps.append(amp_intersection)
    intersection_idxs.append(idx)

plt.plot(Time_f, A_built, 'r', label='FID Built')
plt.plot(Time_f, Amp_FID, 'm', label='Original')
plt.scatter(intersection_times, intersection_amps, color='blue', zorder=5, label='Intersections')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Build-up the FID
Time_build_from_zero = np.arange(0,intersection_times[0],0.1)
Amp_build_from_zero = gauss1(Time_build_from_zero, A_const, popt2[0])

Time_build_end = Time_f[intersection_idxs[0]:]
Amp_build_end = Amp_FID[intersection_idxs[0]:]

Time_build_full = np.concatenate((Time_build_from_zero,Time_build_end))
Amp_build_full = np.concatenate((Amp_build_from_zero,Amp_build_end))

plt.plot(Time_build_full, Amp_build_full, 'r', label='FID Built')
plt.plot(Time_f, Amp_FID, 'm--', label='Original')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

print(r'Maximum amplitude from SE ', popt1[0])
print(r'Maximum amplitude from FID ', popt[0])

print(r'Popt from SE ', popt1)
print(r'Popt from FID ', popt)

print('donedone')    
print('done')
