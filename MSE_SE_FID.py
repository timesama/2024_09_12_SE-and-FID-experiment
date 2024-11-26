# This Python file uses the following encoding: utf-8
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os, re


# ------------------------------
# Utility Functions
# ------------------------------

def calculate_frequency_scale(Time):
    """
    Calculate the frequency scale for a given time array.
    """
    dt = Time[1] - Time[0]
    numberp = len(Time)
    f_nyquist = 1 / (2 * dt)
    df = 2 * f_nyquist / numberp
    Freq = np.arange(-f_nyquist, f_nyquist, df)
    return Freq

def calculate_amplitude(real, imaginary):
    """
    Compute the magnitude (amplitude) of a complex signal.
    """
    return np.sqrt(real**2 + imaginary**2)

def find_nearest(array, value):
    """
    Find the index of the nearest value in an array.
    """
    return (np.abs(np.asarray(array) - value)).argmin()

def time_domain_phase(Real, Imaginary):
    """
    Perform phase correction by minimizing baseline offset.
    """
    delta = np.zeros(360)

    for phi in range(360):
        Re_phased = Real * np.cos(np.deg2rad(phi)) - Imaginary * np.sin(np.deg2rad(phi))
        Im_phased = Real * np.sin(np.deg2rad(phi)) + Imaginary * np.cos(np.deg2rad(phi))
        Magnitude_phased = calculate_amplitude(Re_phased, Im_phased)

        delta[phi] = np.mean(Magnitude_phased[:5] - Re_phased[:5])

    optimal_phi = np.argmin(delta)

    Re_corrected = Real * np.cos(np.deg2rad(optimal_phi)) - Imaginary * np.sin(np.deg2rad(optimal_phi))
    Im_corrected = Real * np.sin(np.deg2rad(optimal_phi)) + Imaginary * np.cos(np.deg2rad(optimal_phi))

    return Re_corrected, Im_corrected

def adjust_frequency(Frequency, Re, Im):
    """
    Adjust the frequency spectrum to center the peak at zero frequency.
    """
    Fid_unshifted = Re + 1j * Im
    FFT = np.fft.fftshift(np.fft.fft(Fid_unshifted))

    if len(Frequency) != len(FFT):
        Frequency = np.linspace(Frequency[0], Frequency[-1], len(FFT))

    index_max = np.argmax(np.abs(FFT))
    index_zero = find_nearest(Frequency, 0)

    delta_index = index_max - index_zero
    FFT_shifted = np.roll(FFT, -delta_index)

    Fid_shifted = np.fft.ifft(np.fft.ifftshift(FFT_shifted))
    return np.real(Fid_shifted), np.imag(Fid_shifted)

def gauss1(x, A, sigma):
    """
    Gaussian function centered at zero.
    """
    return A * np.exp(-x**2 / (2 * sigma**2))


# ------------------------------
# Data Loading and Processing
# ------------------------------

def read_data(file_path):
    """
    Load time, real, and imaginary data from a file.
    """
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1], data[:, 2]

def process_fid_files(directory, patterns):
    """
    Process FID files: phase correction, frequency adjustment, and amplitude calculation.
    """
    results = {}
    for filename in os.listdir(directory):
        for label, pattern in patterns.items():
            if pattern.match(filename):
                file_path = os.path.join(directory, filename)
                Time, Re_original, Im_original = read_data(file_path)

                # Phase correction
                R_phased, I_phased = time_domain_phase(Re_original, Im_original)

                # Frequency adjustment
                Frequency = calculate_frequency_scale(Time)
                Re_final, Im_final = adjust_frequency(Frequency, R_phased, I_phased)

                # Store results
                results[label] = {
                    "Time": Time,
                    "Amplitude": calculate_amplitude(Re_final, Im_final),
                }
    return results

# ------------------------------
# Main Script
# ------------------------------

# Directory and file patterns
parent_directory = os.path.join(os.getcwd(), 'SE_Cycle')
file_patterns = {
    "FID": re.compile(r'FID_C.*\.dat$'),
    "Empty": re.compile(r'FID_Empty.*\.dat$'),
    "Water": re.compile(r'FID_Water.*\.dat$'),
    "MSE": re.compile(r'MSE.*\.dat$'),
}

# Process FID files
fid_results = process_fid_files(parent_directory, file_patterns)

# Example: FID Analysis
Time_fid = fid_results["FID"]["Time"]
Amp_fid = fid_results["FID"]["Amplitude"]
Amp_empty = fid_results["Empty"]["Amplitude"]
Amp_water = fid_results["Water"]["Amplitude"]

Amp_fid_corrected = Amp_fid - Amp_empty

# Plot FID corrected signal
plt.plot(Time_fid, Amp_fid_corrected, label="FID Corrected")
plt.xlabel("Time (μs)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Extrapolation and normalization
cut_idx = find_nearest(Time_fid, 10)
Time_cut = Time_fid[cut_idx:]
Amp_cut = Amp_fid_corrected[cut_idx:]

popt, _ = curve_fit(gauss1, Time_cut, Amp_cut, p0=[max(Amp_cut), 10])
Time_fit = np.linspace(0, max(Time_fid), 500)
Amp_fit = gauss1(Time_fit, *popt)

plt.plot(Time_fit, Amp_fit, label="Gaussian Fit")
plt.plot(Time_fid, Amp_fid_corrected, 'o', label="FID Corrected")
plt.xlabel("Time (μs)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
