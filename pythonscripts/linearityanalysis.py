# mar 2024
# linearity data analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy import signal
from scipy import optimize
#%% initalize dict for holding data to be save

results = {'Voltage p2p (V)':[],
           'Magnetic Field p2p (nT)':[],
           'Magnetic Uncertainty (nT)':[]}

#%% read data and find peak to peak with simple peak finding method

# axis parallel to applied B
axis = 'z'
# driving frequency of applied B
freq = 50 
# sampling frequency
sr = 10000

# notch filtering
notch = True
 

for file in glob.glob('../data/mar11/zaxis_50Hz/*.csv'):
    df = pd.read_csv(file)
    # Vin
    V = float(file[-22:-18]) 
    results['Voltage p2p (V)'].append(V)
    
    # time
    t = df['Epoch Time'] - df['Epoch Time'].iloc[0]
    # B along axis 
    B = df[axis].to_numpy()
    if notch == True:
        f0 = 60.0  # Frequency to be removed from signal (Hz)
        Q = 300.0  # Quality factor
        w0 = f0 / (sr / 2 )  # Normalized Frequency
        b, a = signal.iirnotch(w0, Q)
        B = signal.filtfilt(b, a, B)
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

popt, pcov = optimize.curve_fit(fitfunc, results['Voltage p2p (V)'][:10],
                       results['Magnetic Field p2p (nT)'][:10], sigma=results['Magnetic Uncertainty (nT)'][:10],
                       absolute_sigma=True)

k = popt[0]
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
results_df.to_csv('../results/mar11_zaxis_50Hz.csv')


#%%
df_162Hz = pd.read_csv('../results/mar11_zaxis_162Hz.csv')
df_50Hz = pd.read_csv('../results/mar11_zaxis_50Hz.csv') 

# theory according to Biot Savart and Ohms law
mu0 = 4 * np.pi * 1e-7
res = 3.39e3
voltages = np.linspace(0, 5, 100)
rad = 85e-3
Btheory = mu0 * (voltages / res) / 2 / rad * 1e9

plt.figure()
plt.errorbar(x=df_50Hz['Voltage p2p (V)'], 
             y=df_50Hz['Magnetic Field p2p (nT)'], 
             yerr=df_50Hz['Magnetic Uncertainty (nT)'],
             capsize=1,
             fmt='o--',
             markersize=2,
             label='50 Hz')
plt.errorbar(x=df_162Hz['Voltage p2p (V)'], 
             y=df_162Hz['Magnetic Field p2p (nT)'], 
             yerr=df_162Hz['Magnetic Uncertainty (nT)'],
             capsize=1,
             fmt='o--',
             markersize=2,
             label='162 Hz')
plt.plot(voltages, Btheory, 'k--', label='Biot Savart Law')
plt.xlabel('Input Peak-to-Peak Voltage (V)')
plt.ylabel('Measured Magnetic Peak-to-Peak Amplitude (nT)')
plt.grid()
plt.tight_layout()
plt.legend()
#plt.gca().set_aspect("equal")
plt.show()









