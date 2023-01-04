import queue
import time
import random
from threading import Thread, Event

from collections import deque

import Adafruit_ADS1x15
import neurokit2 as nk

import psutil


GAIN = 2/3
COLLECTION_TIME = 10
SAMPLE_RATE = 1000


# Create an ADS1115 ADC (16-bit) instance.
adc = Adafruit_ADS1x15.ADS1115()

q = queue.Queue()  # ----- new data structure ----

readings = []

process = Event()
process.clear()

class AnalyzerThread(Thread):
    def run(self):
        global readings
        while True:
            process.wait()
            ppg_signal = readings
            print('RAM memory % used:', psutil.virtual_memory()[2])
            # start = time.time()
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

class ProducerThread(Thread):
    def run(self):
        while True:
            num = adc.read_adc(0, gain=GAIN)
            q.put(num)      # ---- automatically synchronized 
            time.sleep(0.0011627906977)

class ConsumerThread(Thread):
    def run(self):
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
        readings = l1 + l2 + l3
        # print(len(readings))
        print(q.qsize())
        #add flag to consume readings
        process.set()
        readings.clear()

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
                
                if x == 0:
                    readings = l2 + l3 + l1
                elif x == 1:
                    readings = l3 + l1 + l2
                else:
                    readings = l1 + l2 + l3
                # print(len(readings))
                print(time.time()-start)
                process.set()
                readings.clear()

producer = ProducerThread().start()
consumer = ConsumerThread().start()
analyzer = AnalyzerThread().start()