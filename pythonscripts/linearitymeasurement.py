# mar 2024
# pure labjack measure

from QZFM import QZFM
import os, glob, time, sys
import matplotlib.pyplot as plt
import queue
from threading import Thread
from labjack import ljm 
from datetime import datetime
import ctypes
import numpy as np
from time import time 
import time as time_
import pandas as pd
from tqdm import tqdm
from scipy import signal
from scipy.signal import butter, lfilter
#%%
q = QZFM('COM3')
#%%
q.auto_start(zero_calibrate=False)

#%% labjack function
def labjack_measure(measurement_length, scanRate, aScanListNames, convfactors=None, ranges=[1.0, 1.0, 1.0], queue=None):
    """
    Reads data from labjack daq (T7), specfically adapted for reading analog output from Barington

    Parameters
    ----------
    measurement_length : float
        Measurement time length, up to  0.5 seconds precision.
    scanRate : int
        Sampling rate in Hz.
    aScanListNames : list of string 
        Name of channels to read, for example "AIN0" or "AIN3".
    convfactors : list of floats or None
        The unit conversions in V/A.U. in a list, one element for each AIN channel. Or None, then defaults to 1 V/A.U.
    ranges : [10.0, 1.0, 0.1]
        Three possible values for the voltage range of each channel. Pick to get best resolution
    Returns
    -------
    output : array
        1-D list of measured data in interweaved format
    """
    # Open first found LabJack
    handle = ljm.openS("ANY", "ANY", "ANY")  # Any device, Any connection, Any identifier

    info = ljm.getHandleInfo(handle)
    # print("Opened a LabJack with Device type: %i, Connection type: %i,\n"
    #       "Serial number: %i, IP address: %s, Port: %i,\nMax bytes per MB: %i" %
    #       (info[0], info[1], info[2], ljm.numberToIP(info[3]), info[4], info[5]))
    deviceType = info[0]

    # Stream Configuration
    numAddresses = len(aScanListNames)
    aScanList = ljm.namesToAddresses(numAddresses, aScanListNames)[0]
    scansPerRead = int(scanRate / 2)

    MAX_REQUESTS = int(measurement_length * scanRate / scansPerRead)
    # The number of eStreamRead calls that will be performed.

    output = []
    # initialize empty list for appending labjack readings
    
    try:
        # When streaming, negative channels and ranges can be configured for
        # individual analog inputs, but the stream has only one settling time and
        # resolution.

        # LabJack T7 and other devices configuration

        # Ensure triggered stream is disabled.
        ljm.eWriteName(handle, "STREAM_TRIGGER_INDEX", 0)

        # Enabling internally-clocked stream.
        ljm.eWriteName(handle, "STREAM_CLOCK_SOURCE", 0)

        # All negative channels are single-ended, AIN0 and AIN1 ranges are
        # +/-10 V, stream settling is 0 (default) and stream resolution index
        # is 0 (default).
        range_list = ["AIN{}_RANGE".format(i)  for i in range(len(ranges))]
            
        aNames = ["AIN_ALL_NEGATIVE_CH"] + range_list + ["STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
        aValues = [ljm.constants.GND]+ranges+[0, 0]
        # Write the analog inputs' negative channels (when applicable), ranges,
        # stream settling time and stream resolution configuration.
        numFrames = len(aNames)
        ljm.eWriteNames(handle, numFrames, aNames, aValues)

        # Configure and start stream
        scanRate = ljm.eStreamStart(handle, scansPerRead, numAddresses, aScanList, scanRate)
        # print("\nStream started with a scan rate of %0.0f Hz." % scanRate)

        # print("\nPerforming %i stream reads." % MAX_REQUESTS)
        start = datetime.now()
        time_start = time() # epoch time at start
        totScans = 0
        totSkip = 0  # Total skipped samples
        
        i = 1
        #initialize progress bar
        progress = tqdm(leave=False, total=MAX_REQUESTS, desc='Measuring Cell')
        
        while i <= MAX_REQUESTS:
            i_start = i
            ret = ljm.eStreamRead(handle)

            aData = ret[0]
            output = output + aData
            scans = len(aData) / numAddresses
            totScans += scans

            # Count the skipped samples which are indicated by -9999 values. Missed
            # samples occur after a device's stream buffer overflows and are
            # reported after auto-recover mode ends.
            curSkip = aData.count(-9999.0)
            totSkip += curSkip

            # print("\neStreamRead %i" % i)
            ainStr = ""
            for j in range(0, numAddresses):
                ainStr += "%s = %0.5f, " % (aScanListNames[j], aData[j])
            # print("  1st scan out of %i: %s" % (scans, ainStr))
            # print("  Scans Skipped = %0.0f, Scan Backlogs: Device = %i, LJM = "
            #       "%i" % (curSkip/numAddresses, ret[1], ret[2]))            
            i += 1
            progress.update(i-i_start)

        end = datetime.now()
        
        # print("\nTotal scans = %i" % (totScans))
        tt = (end - start).seconds + float((end - start).microseconds) / 1000000
        # times = tt*np.arange(measurement_length*scanRate)/(measurement_length*scanRate) + time_start
        
        # print("Time taken = %f seconds" % (tt))
        # print("LJM Scan Rate = %f scans/second" % (scanRate))
        # print("Timed Scan Rate = %f scans/second" % (totScans / tt))
        # print("Timed Sample Rate = %f samples/second" % (totScans * numAddresses / tt))
        # print("Skipped scans = %0.0f" % (totSkip / numAddresses))
        
        actual_length = len(output[::numAddresses])
        times = np.linspace(0, tt, actual_length) + time_start
        output_ar = np.empty(shape=(numAddresses+1, actual_length))
        output_ar[0, :] = times
        for i in range(numAddresses):
            output_ar[i+1, :] = np.asarray(output[i::numAddresses])

        def clean_array(array):
            """
            Cleans -9999.0 (labjack error value) from array

            Parameters
            ----------
            array : numpy array
                Array to be cleaned and converted
            conversion_factor : float
                conversion factor in the format * V / AU

            Returns
            -------
            cleaned and converted array

            """
            
            # remove junk values from labjack internal error
            for i in range(numAddresses):
                for j in range(actual_length):
                    if array[i+1,j] < -9998.0:
                        try:
                            array[i+1, j] = array[i+1, j-1]
                        except IndexError:
                            array[i+1, j] = array[i+1, j+1]
            return array

        output_ar = clean_array(output_ar)
        
        # perform unit conversion
        if convfactors == None:
            pass
        else:
            for i in range(numAddresses):
                output_ar[i+1, :] /= convfactors[i]
        # queue.put(output_ar)
    except ljm.LJMError:
        ljme = sys.exc_info()[1]
        print(ljme)
    except Exception:
        e = sys.exc_info()[1]
        print(e)

    try:
        print("\nStop Stream")
        ljm.eStreamStop(handle)
    except ljm.LJMError:
        ljme = sys.exc_info()[1]
        print(ljme)
    except Exception:
        e = sys.exc_info()[1]
        print(e)

    # Close handle
    ljm.close(handle)

    return output_ar

#%% main measuring script
if __name__ == '__main__':
    # measuring length
    t = 3
    
    
    gain = '0.33x' # possible gains are 0.1x|0.33x|1x|3x
    save = False # save as csv if True
    fp = '..//data//mar11//zaxis_35Hz//'
    
    # corresponding conversion V/nT for each gain
    gain_dict = {'0.1x':0.27,
                 '0.33x':0.9,
                 '1x':2.7,
                 '3x':8.1}
    current_time = datetime.now().strftime('%y%m%dT%H%M%S')
    # set gain
    # q.set_gain(gain)

    # zero 
    q.field_zero(True, show=False)
    # sleep 10 seconds
    for i in range(10):
        time_.sleep(1)
        print(i)
    q.field_zero(False)
    
    Vrms = float(input('Check voltage with Multimeter, enter result in units of V\n'))
    V = round(np.sqrt(2)*Vrms, 2) # convert Vrms to Vpp
    strV = str(V)
    if len(strV) == 4:
        pass
    else:
        strV = strV + '0'
    # measure labjack
    out = labjack_measure(t, 10000, ["AIN2"], [gain_dict[gain]], [10.0])
    
    # save data
    out_df = pd.DataFrame(out.T)
    out_df.columns = ['Epoch Time', 'z']
    out_df.set_index("Epoch Time", inplace=True)
    if save:
        out_df.to_csv(fp+strV+'-'+datetime.now().strftime('%y%m%dT%H%M%S')+'.csv')

    # last is 3.3 Vrms on the AWG
#%%
from scipy.signal import butter, sosfilt, sosfreqz
def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

plt.figure()
t = out[0, :]-out[0, 0]
plt.plot(t, out[1, :])
plt.plot(t, butter_bandpass_filter(out[1, :], 30, 40, 10000))
plt.show()



#%%




plt.figure()
a, b = signal.periodogram(out[3,:], 1000)
plt.semilogy(a, np.sqrt(b))
plt.show()



