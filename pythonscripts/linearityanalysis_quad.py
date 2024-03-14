# # mar 2024
# linearity data analysis with quadratic curvefit 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from scipy import signal
from scipy import optimize
from scipy.signal import butter, sosfilt, sosfreqz
#%% initalize dict for holding data to be save

results = {'Voltage p2p (V)':[],
           'Magnetic Field p2p (nT)':[],
           'Magnetic Uncertainty (nT)':[]}

#%% read data and find peak to peak with simple peak finding method

# axis parallel to applied B
axis = 'z'
# driving frequency of applied B
freq = 10
# sampling frequency
sr = 10000
def quad(x, a, b, c):
    return a*x**2 + b*x + c

for file in glob.glob('../data/mar13/zaxis_10Hz/*.csv'):
    df = pd.read_csv(file)
    # Vin
    V = float(file[-22:-18]) 
    results['Voltage p2p (V)'].append(V)
    
    # time
    t = df['Epoch Time'] - df['Epoch Time'].iloc[0]
    # B along axis 
    B = df[axis].to_numpy()
    # find index of maximums and minimums
    max_idx = signal.find_peaks(B, distance=int(0.9*sr/freq))
    min_idx = signal.find_peaks(-B, distance=int(0.9*sr/freq))

    # calculate number of samples for 1/8 of period
    period = sr/freq # period in samples
    
    maximas = []
    minimas = []
    # iterate through maxima and minima indexes, fit quadratic 
    for idx in max_idx[0]:
        if idx-int(period/8) < 0:
            segment = B[:idx+int(period/8)]
        else:
            segment = B[idx-int(period/8):idx+int(period/8)]
        x = np.linspace(0, 1, len(segment))
        fit_ok = False
        itr = 0
        last_init_guess = (0.01, 0.6, 1, 0.75)
        while not fit_ok:
            if itr == 0:
                p0 = (1, 1, 1)
                popt, pcov = optimize.curve_fit(quad, x, segment, p0=p0, maxfev=2000)
                # show fit result
                plt.figure(figsize=(6.4*1.5, 4.8*1.5))
                plt.plot(x, segment, label="Raw")
                plt.plot(x, quad(x, *popt), 'r--',
                         label=r'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
                plt.ylim(np.min(segment), np.max(segment))
                plt.legend() 
                plt.pause(0.1)
                plt.show()
                print('Currently on {}'.format(file))
                fit_ok = input('Fit is OK? T or F)\n')
                if fit_ok == 'T' or fit_ok == 't':
                    fit_ok = True
                else:
                    fit_ok = False
                plt.close()
                itr += 1
                continue
            else:
                p0 = (1, 1, 1)
                popt, pcov = optimize.curve_fit(quad, x, segment, p0=p0, maxfev=2000)
                # show fit result
                plt.figure(figsize=(6.4*1.5, 4.8*1.5))
                plt.plot(x, segment, label="Raw")
                plt.plot(x, quad(x, *popt), 'r--',
                     label=r'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
                plt.ylim(np.min(segment), np.max(segment))
                plt.legend() 
                plt.pause(0.1)
                plt.show()
                print('Currently on {}'.format(file))
                fit_ok = 't' #input('Fit is OK? T or F)\n')
                if fit_ok == 'T' or fit_ok == 't':
                    fit_ok = True
                else:
                    fit_ok = False
                plt.close()
                last_init_guess = p0
                continue
        maximas.append(popt[2])
    for idx in min_idx[0]:
        if idx-int(period/8) < 0:
            segment = B[:idx+int(period/8)]
            x = np.linspace(0, 1, len(segment))
            popt, pcov = optimize.curve_fit(quad, x, segment)
        else:
            segment = B[idx-int(period/8):idx+int(period/8)]
            x = np.linspace(0, 1, len(segment))
            popt, pcov = optimize.curve_fit(quad, x, segment)
        minimas.append(popt[2])
    # get interweaved peak values
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


















