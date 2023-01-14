import threading
import random
import time
import queue
import psutil

import neurokit2 as nk

from threading import Thread, Event

GAIN = 2/3
COLLECTION_TIME = 15
SAMPLE_RATE = 1000

# Used to exchange data between Prcoessing and Analyzer Threads
my_list = []

# Used to exchange data betwen Collection and Processing Threads 
q = queue.Queue()

# Lock to synchronize access to the list
lock = threading.Lock()

# Flag used to signal list is ready to analyze
process = Event()
process.clear()



# Collects data from the sensors
def CollectionThread():
    while True:
        # num = adc.read_adc(0, gain=GAIN)
        num = random.randint(0,10)
        q.put(num)      # ---- automatically synchronized 
        time.sleep(0.5)

# Recieves data from the collection thread and implements a circular buffer 
# that enables a moving rectangular time window 
def ProcessingThread():
    global readings
    l1 = []
    l2 = []
    l3 = []

    buckets = [l1,l2,l3]
    for bucket in buckets:
        start = time.time()
        while (time.time() - start <= COLLECTION_TIME):
            num = q.get(True) # block if work queue empty ------
            bucket.append(num)
            # print(bucket , q.qsize())

    # combine and send list
    with lock:
        readings = l1 + l2 + l3
        print(readings)
    print(q.qsize())
    #add flag to consume readings
    process.set()
    # readings.clear()

    while True:

        for x in range(0,3):
            print("Filling bucket", x)
            buckets[x].clear()
            start = time.time()
            while (time.time() - start <= COLLECTION_TIME):
                num = q.get(True) # block if work queue empty ------
                buckets[x].append(num)
                # print(buckets[x], q.qsize())
            #combine and send list 
            
            with lock:
                if x == 0:
                    readings = l2 + l3 + l1
                elif x == 1:
                    readings = l3 + l1 + l2
                else:
                    readings = l1 + l2 + l3
                print(readings)

            print(time.time()-start)
            process.set()
            # readings.clear()

def AnalyzerThread():
    print("Started analysis")
    global readings
    ppg_signal = []
    while True:
        process.wait()
        print("Analyzer", readings)
        for reading in readings:
            ppg_signal.append(reading)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        start = time.time()
        # arr = nk.ppg_simulate(duration=180, sampling_rate=1000, heart_rate=70)
        # ppg_signal = list(arr)
        ppg_cleaned = nk.ppg_clean(ppg_signal)
        ppg_signals, info = nk.ppg_process(ppg_cleaned, sampling_rate = SAMPLE_RATE)
        f_hrv = nk.hrv(ppg_signals)
        t_hrv = nk.hrv_time(ppg_signals)
        print("frequency domain hrv", f_hrv)
        print("time domain hrv", t_hrv)
        print("elaspsed time:", time.time()-start)
        # ppg_signal.clear()
        process.clear()
        readings.clear()

producer = threading.Thread(target=CollectionThread)
consumer = threading.Thread(target=ProcessingThread)
analyzer = threading.Thread(target=AnalyzerThread)
producer.start()
consumer.start()
analyzer.start()

# Wait for the threads to finish
producer.join()
consumer.join()
analyzer.join()