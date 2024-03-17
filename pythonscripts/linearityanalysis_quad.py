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
axis = 'y'
# driving frequency of applied B
freq = 35
# sampling frequency
sr = 10000

# bandpass
bandpass = False
 
def butter_bandpass(lowcut, highcut, fs, order=1):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y
    
def quad_fit(x, a, b, c):
    return a*x**2 + b*x + c

for file in glob.glob('../data/mar15/{}axis_{}Hz/*.csv'.format(axis, freq)):
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
    
    period = sr / freq
    
    maximas = []
    minimas = []
    for idx in max_idx[0]:
        if idx > period // 4:
            try:
                y = B[int(idx-period//4):int(idx+period//4)]
                x = np.arange(len(y))
                p = np.polyfit(x, y, 4)
                yn = np.poly1d(p)
                
                # plt.figure()
                # plt.plot(x, y)
                # plt.plot(x, yn(x))
                # plt.show()
                # time_.sleep(0.2)
                
                #print(max(yn(x)))
                maximas.append(max(yn(x)))
            except IndexError:
                continue
    for idx in min_idx[0]:
        if idx > period // 4:
            try:
                y = B[int(idx-period//4):int(idx+period//4)]
                x = np.arange(len(y))
                p = np.polyfit(x, y, 4)
                yn = np.poly1d(p)
                
                # plt.figure()
                # plt.plot(x, y)
                # plt.plot(x, yn(x))
                # plt.show()
                # time_.sleep(0.2)
                
                minimas.append(min(yn(x)))
            except IndexError:
                continue
    # get interweaved peak values
    # maximas = B[max_idx[0]]
    # minimas = B[min_idx[0]]
    extremas = np.asarray([val for pair in zip(maximas, minimas) for val in pair])
  
    # plt.figure()
    # plt.plot(B)
    # plt.plot(max_idx[0], B[max_idx[0]], "x")
    # plt.plot(min_idx[0], B[min_idx[0]], "x")
    # plt.show()
    
    #time_.sleep(4)
    # calculate peak to peak 
    pp = []
    for i in range(len(extremas)-1):
        pp.append(abs(extremas[i]-extremas[i+1]))
    results['Magnetic Field p2p (nT)'].append(np.mean(pp))
    results['Magnetic Uncertainty (nT)'].append(np.std(pp))

#%% save data
results_df = pd.DataFrame(results)
input('Sure you want to save? If not careful, WILL overwrite existing file')
results_df.to_csv('../results/yaxis_035Hz_quadfit_2.csv')

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

# function to get convert factor from input voltage axis to input magnetic field 
def conv_factor_v2b(df, cutoff_idx):
    popt, pcov = optimize.curve_fit(fitfunc, df['Voltage p2p (V)'][:cutoff_idx],
                           df['Magnetic Field p2p (nT)'][:cutoff_idx], sigma=df['Magnetic Uncertainty (nT)'][:cutoff_idx],
                           absolute_sigma=True)
    return popt[0]

#%%
plt.figure()
plt.errorbar(x=np.multiply(results['Voltage p2p (V)'], conv_factor_v2b(results, 7)), 
             y=results['Magnetic Field p2p (nT)'], 
             yerr=results['Magnetic Uncertainty (nT)'],
             capsize=3,
             fmt='o',
             markersize=2)
plt.xlabel('Input Peak-to-Peak Voltage (V)')
plt.ylabel('Measured Magnetic Peak-to-Peak Amplitude (nT)')
plt.grid()
plt.tight_layout()
#plt.gca().set_aspect("equal")
plt.show()




#%%
# theory according to Biot Savart and Ohms law
mu0 = 4 * np.pi * 1e-7
res = 3.39e3
voltages = np.linspace(0, 10, 100)
rad = 85e-3
Btheory = mu0 * (voltages / res) / 4 / rad * 1e9


plt.figure()
for file in glob.glob('..//results//*035Hz*final*quad*.csv'):
    freq = file[18:21]
    axis = file[12]
    df = pd.read_csv(file)
    plt.errorbar(x=np.multiply(df['Voltage p2p (V)'], conv_factor_v2b(df, 7)), 
                 y=df['Magnetic Field p2p (nT)'], 
                 yerr=df['Magnetic Uncertainty (nT)'],
                 capsize=1,
                 fmt='o',
                 markersize=2,
                 label='{} Hz {} axis'.format(freq, axis))
plt.plot(np.multiply(df['Voltage p2p (V)'], conv_factor_v2b(df, 5)), np.multiply(df['Voltage p2p (V)'], conv_factor_v2b(df, 5)), 'k--')
#plt.plot(voltages, Btheory, 'k--', label='Biot Savart Law')
plt.xlabel('Input Magnetic Peak-to-Peak Voltage (nT)')
plt.ylabel('Measured Magnetic Peak-to-Peak Amplitude (nT)')
plt.grid()
plt.tight_layout()
plt.legend()
#plt.gca().set_aspect("equal")
plt.show()

#%% residual graph
cutoff_idx = 10
df = pd.read_csv('..//results//xaxis_035Hz_final_quadfit.csv')
popt, pcov = optimize.curve_fit(fitfunc, df['Voltage p2p (V)'][:cutoff_idx],
                       df['Magnetic Field p2p (nT)'][:cutoff_idx], sigma=df['Magnetic Uncertainty (nT)'][:cutoff_idx],
                       absolute_sigma=True)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].errorbar(df['Voltage p2p (V)'], df['Magnetic Field p2p (nT)'], 
                yerr=df['Magnetic Uncertainty (nT)'], fmt='o', markersize=2, label='')
axs[0].plot(df['Voltage p2p (V)'], fitfunc(df['Voltage p2p (V)'], *popt), 'k--', label='Fit based on first 10 points')
axs[0].set_xlim(right=11)
axs[0].set_ylim(top=10)
axs[0].set_ylabel('Measured Magnetic Peak-to-Peak Amplitude (nT)')
axs[0].set_xlabel('Input Magnetic Peak-to-Peak Voltage (nT)')
axs[0].legend()
axs[0].grid()

residuals = (fitfunc(df['Voltage p2p (V)'], *popt) - df['Magnetic Field p2p (nT)'])[:cutoff_idx]
axs[1].errorbar(x=df['Voltage p2p (V)'][:cutoff_idx], y=residuals, 
                yerr=df['Magnetic Uncertainty (nT)'][:cutoff_idx], fmt='o', 
                capsize=3, color='C0')
axs[1].axhline(0, -1, 11, linewidth=2, color='Black')
axs[1].set_xlabel('Input Magnetic Peak-to-Peak Voltage in the Linear Regime (nT)')
axs[1].set_ylim(-0.025, 0.025)
axs[1].grid()
plt.tight_layout()
fig.show()
