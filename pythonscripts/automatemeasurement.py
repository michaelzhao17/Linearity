# mar 2024
# to fully automate linearity measurements

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
import pyvisa 
from SiglentDevices import DG1032Z
import pathlib
#%% initialize instruments as objects
rm = pyvisa.ResourceManager()
#print(rm.list_resources())

# Digital Multimeter
DMM = rm.open_resource('USB0::0x1334::0x0204::262800066::INSTR')
DMM.read_termination = '\n'
DMM.write_termination = '\n'
DMM.query('*IDN?')

# Digital Oscilloscope
awg = DG1032Z(hostname='USB0::0x1AB1::0x0642::DG1ZA232603182::INSTR')
awg.query('*IDN?')

#%%
# QZFM
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
#%% initial configuration
# AWG settings
ch = 1
freq = 200
offset = 0
vpp = 1
waveform = 'SIN'
res = 2030
cap = 8.67e-6
imp = np.sqrt(res**2+1/(2*np.pi*freq*cap)**2)

awg.set_impedance(ch, imp)
awg.set_wave(ch, waveform, freq, vpp, offset, phase=0)

# QZFM settings
gain = '0.33x' # possible gains are 0.1x|0.33x|1x|3x
# corresponding conversion V/nT for each gain
gain_dict = {'0.1x':0.27,
             '0.33x':0.9,
             '1x':2.7,
             '3x':8.1}
q.set_gain(gain)
zero_t = 10
t = 3 # number of seconds to record
axis = 'z' # axis being measured

# save file settings
save = True # save as csv if True
fp = '..//data//mar13//'
#%% auto folder creation function
def make_folder(fp, axis, freq):
    '''
    Parameters
    ----------
    fp : str
        file path to location where folder is to be created
    axis : str, x|y|z
        axis being measured.
    freq : float
        frequency being measured.

    Returns
    -------
    folder_name : str
        name of folder
    
    creates folder and then returns folder name
    '''
    folder_name = '{}axis_{}Hz'.format(axis, freq)
    # check if folder already exists, if not create it
    pathlib.Path(fp+folder_name).mkdir(parents=True, exist_ok=True) 
    return folder_name

#%% main script
if __name__ == '__main__':
    # iterate over frequencies
    for freq in [20,  80, 400]:
        # iterable of Vpp values to output
        vpps = np.linspace(0.1, 10, 99)
        i = 0
        progress = tqdm(leave=False, total=99, desc='Experiment Running')
        # make folder and get folder name 
        folder_name = make_folder('..//data//mar13//', axis, freq)
        
        for idx, vpp in enumerate(vpps):
            i_start = i
            awg.set_wave(ch, waveform, freq, vpp, offset, phase=0)
            
            # zero 
            q.field_zero(True, show=False)
            # sleep 10 seconds
            for i in range(zero_t):
                time_.sleep(1)
            q.field_zero(False)
            
            # turn on AWG
            awg.set_ch_state(ch, state=True)
            time_.sleep(2)
            # measure rms voltage from DMM
            v_rms_list = []
            DMM.write('F2')
            j = 0
            while j < 50:
                ret = DMM.query("*TRG")
                v_rms_list.append(float(ret[:-1]))
                j += 1
            # calculate average vrms measured and convert to vpp
            v_rms_meas = np.mean(v_rms_list)
            v_pp_meas = round(v_rms_meas * 2 * np.sqrt(2), 2)
            strV = str(v_pp_meas)
            print('DMM measures {} Vpp'.format(strV))
            if len(strV) == 4:
                pass
            else:
                strV = strV + '0'
                
            # labjack measure
            out = labjack_measure(t, 10000, ["AIN2"], [gain_dict[gain]], [10.0])
            
            # turn off AWG
            awg.set_ch_state(ch, state=False)
            # save data
            out_df = pd.DataFrame(out.T)
            out_df.columns = ['Epoch Time', 'z']
            out_df.set_index("Epoch Time", inplace=True)
            if save:
                current_time = datetime.now().strftime('%y%m%dT%H%M%S')
                out_df.to_csv(fp+folder_name+'//'+strV+'-'+datetime.now().strftime('%y%m%dT%H%M%S')+'.csv')
            i += 1
            progress.update(i-i_start)
    
    












