# mar 2024
# linearity data analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy import signal
from scipy import optimize
from scipy.signal import butter, sosfilt, sosfreqz
import time as time_
#%% initalize dict for holding data to be save

results = {'Voltage p2p (V)':[],
           'Magnetic Field p2p (nT)':[],
           'Magnetic Uncertainty (nT)':[]}

#%% read data and find peak to peak with simple peak finding method

# axis parallel to applied B
axis = 'z'
# driving frequency of applied B
freq = 3
# sampling frequency
sr = 10000

# bandpass
bandpass = False
 
def butter_bandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


for file in glob.glob('../data/mar13/zaxis_{}Hz/*.csv'.format(freq)):
    df = pd.read_csv(file)
    # Vin
    V = float(file[-22:-18]) 
    results['Voltage p2p (V)'].append(V)
    
    # time
    t = df['Epoch Time'] - df['Epoch Time'].iloc[0]
    # B along axis 
    B = df[axis].to_numpy()
    if bandpass:
        B = butter_bandpass_filter(B, freq-1, freq+1, sr)[sr:]
    # find index of maximums and minimums
    max_idx = signal.find_peaks(B, distance=int(0.9*sr/freq))
    min_idx = signal.find_peaks(-B, distance=int(0.9*sr/freq))
    
    # ensure both arrays have same number of elements
    if len(max_idx) == len(min_idx):
        pass
    elif len(max_idx) == len(min_idx) + 1:
        max_idx = max_idx[:-1]
    elif len(max_idx) == len(min_idx) - 1:
        min_idx = min_idx[:-1]
    
    # get interweaved peak values
    maximas = B[max_idx[0]]
    minimas = B[min_idx[0]]
    extremas = np.asarray([val for pair in zip(maximas, minimas) for val in pair])
    
    plt.figure()
    plt.plot(B)
    plt.plot(max_idx[0], B[max_idx[0]], "x")
    plt.plot(min_idx[0], B[min_idx[0]], "x")
    plt.show()
    
    time_.sleep(10)
    # calculate peak to peak 
    pp = []
    for i in range(len(extremas)-1):
        pp.append(abs(extremas[i]-extremas[i+1]))
    results['Magnetic Field p2p (nT)'].append(np.mean(pp))
    results['Magnetic Uncertainty (nT)'].append(np.std(pp))
    
#%%
plt.figure()
plt.errorbar(x=results['Voltage p2p (V)'], 
             y=results['Magnetic Field p2p (nT)'], 
             yerr=results['Magnetic Uncertainty (nT)'],
             capsize=3,
             fmt='o--',
             markersize=2)
plt.xlabel('Input Peak-to-Peak Voltage (V)')
plt.ylabel('Measured Magnetic Peak-to-Peak Amplitude (nT)')
plt.grid()
plt.tight_layout()
#plt.gca().set_aspect("equal")
plt.show()

#%% curve fit to determine proportionality constant k between input voltage and input B

# fitting function
def fitfunc(V, a, b):
    return a * V + b

results = pd.read_csv('..//results//zaxis_003Hz.csv')
popt, pcov = optimize.curve_fit(fitfunc, results['Voltage p2p (V)'][:10],
                       results['Magnetic Field p2p (nT)'][:10], sigma=results['Magnetic Uncertainty (nT)'][:10],
                       absolute_sigma=True)

k = popt[0]
print(k)
Bin = np.multiply(k, results['Voltage p2p (V)'])
Bmeas = results['Magnetic Field p2p (nT)']



plt.figure()
plt.errorbar(Bin, Bmeas, results['Magnetic Uncertainty (nT)'],
             capsize=5,
             fmt='o--',
             markersize=2)
plt.plot(Bin, Bin, 'k--')
plt.xlabel('162 Hz Reference Magnetic Field [nT peak-to-peak]')
plt.ylabel('162 Hz Measured Magnetic Field [nT peak-to-peak]')
plt.grid()
plt.show()


#%% save data
results_df = pd.DataFrame(results)
input('Sure you want to save? If not careful, WILL overwrite existing file')
results_df.to_csv('../results/zaxis_010Hz.csv')


#%%
# theory according to Biot Savart and Ohms law
mu0 = 4 * np.pi * 1e-7
res = 3.39e3
voltages = np.linspace(0, 10, 100)
rad = 85e-3
Btheory = mu0 * (voltages / res) / 4 / rad * 1e9




plt.figure()
for file in glob.glob('..//results//*.csv'):
    freq = file[18:21]
    df = pd.read_csv(file)
    plt.errorbar(x=np.multiply(k, df['Voltage p2p (V)']), 
                 y=df['Magnetic Field p2p (nT)'], 
                 yerr=df['Magnetic Uncertainty (nT)'],
                 capsize=1,
                 fmt='o--',
                 markersize=2,
                 label='{} Hz'.format(freq))
plt.plot(Bin, Bin, 'k--')
#plt.plot(voltages, Btheory, 'k--', label='Biot Savart Law')
plt.xlabel('Input Magnetic Peak-to-Peak Voltage (nT)')
plt.ylabel('Measured Magnetic Peak-to-Peak Amplitude (nT)')
plt.grid()
plt.tight_layout()
plt.legend()
#plt.gca().set_aspect("equal")
plt.show()






