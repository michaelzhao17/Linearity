# Jan 2024
# To measure from Labjack and QZFM module serial roughly simultaneously 

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
#%%
q = QZFM("COM3")

#%%
q.auto_start(zero_calibrate=False)

#%%
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
        queue.put(output_ar)
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

#%%
q.field_zero(True, True, False)
#%%
if __name__ == '__main__':
    t = 30
    save = False
    # initialize queue object for saving outputs of functions
    output_queue = queue.Queue()
     
    # turn on field zeroing
    t0 = Thread(target=q.field_zero, args=(True, True, False))
    t0.start()
    t0.join()
    
    # initialize thread of QZFM module for coil offset reading
    t1 = Thread(target=q.read_offsets_custom_test, args=(int(t*7.5), output_queue))
    print('QZFM thread defined')
    
    # initialize thread of labjack for cell reading
    t2 = Thread(target=labjack_measure, args=(t, 1000, ["AIN0", "AIN1", "AIN2"], [0.0027, 0.0027, 0.0027], [10.0, 10.0, 10.0], output_queue))
    print("Labjack thread defined")
                                                               
    # start the threads
    t1.start()
    print('QZFM thread started')
    t2.start()
    print('Labjack thread started')


    
    # join the threads
    t1.join()
    print("p1 joined")
    t2.join()
    print("p2 joined")
    
    t3 = Thread(target=q.field_zero, args=(False, True, False))
    t3.start()
    t3.join()
    print("p3 joined")
        
    cnt = 0
    out = []
    while cnt < 2:
        try:
            out.append(output_queue.get())
            cnt +=1
        except Exception:
            break
        
    for item in out:
        if isinstance(item, pd.DataFrame):
            coil_offsets = item
        elif isinstance(item, np.ndarray):
            cell_readings = item 

    cell_readings = pd.DataFrame(cell_readings.T)
    cell_readings.columns = ['Epoch Time', 'x', 'y', 'z']
    cell_readings.set_index("Epoch Time", inplace=True)
    
    if save:
        file_name = "" #input("Name of the file is?\n")
        coil_offsets.to_csv("flip_data//feb20//"+file_name+datetime.now().strftime('%y%m%dT%H%M%S')+"coiloffsets"+".csv")
        cell_readings.to_csv("flip_data//feb20//"+file_name+datetime.now().strftime('%y%m%dT%H%M%S')+"cellreadings"+".csv")


#%%

plt.figure()
for axis in ['x', 'y', 'z']: 
    
    plt.plot(cell_readings.index-cell_readings.index[0], cell_readings[axis], label="Cell Reading {}".format(axis))
    #plt.plot((coil_offsets.index-coil_offsets.index[0])[:], coil_offsets[axis].iloc[:], label="Coil Reading {}".format(axis))
    #plt.plot((coil_offsets.index-cell_readings.index[0])[25*8:], coil_offsets[axis].iloc[25*8:]-coil_offsets[axis].iloc[25*8:].mean(), label="Coil Reading {}".format(axis))
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("B field [pT]")
    plt.title("Simultaneous coil offset field and cell field in {} direction".format(axis))
#plt.plot((coil_offsets.index-cell_readings.index[0])[25*8:], coil_offsets['temp'].iloc[25*8:], label="temperature voltage")
plt.grid()
plt.show()
#plt.savefig('report_pictures//driftover10mins.pdf')

#%%
plt.figure()
for axis in ['x', 'y', 'z']: 
    a, b = signal.periodogram(cell_readings[axis], 1000)
    plt.semilogy(a, np.sqrt(b), label="Coil Reading {} PSD".format(axis))
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("B field [pT]")
plt.grid()
plt.show()





