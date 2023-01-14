import queue
import time
from threading import Thread, Event

import Adafruit_ADS1x15
import neurokit2 as nk

import copy


import psutil


GAIN = 2/3
COLLECTION_TIME = 60
SAMPLE_RATE = 860
ADC_CONVERSION_RATE = 1/SAMPLE_RATE


# Create an ADS1115 ADC (16-bit) instance.
adc = Adafruit_ADS1x15.ADS1115()

ppgSensorDataQueue = queue.Queue()  # ----- new data structure ----

AnalyzerDataBuff = []

isDataReady = Event()
isDataReady.clear()

class ppgAlgoThread(Thread):
    def run(self):
        global AnalyzerDataBuff
        while True:
            isDataReady.wait()
            algoPpgSensorBuff = copy.deepcopy(AnalyzerDataBuff)
            isDataReady.clear()
            
            # print(algoPpgSensorBuff)
            print(len(algoPpgSensorBuff))
            
            print('RAM memory % used:', psutil.virtual_memory()[2])
            start = time.time()
            
            #arr = nk.ppg_simulate(duration=180, sampling_rate=1000, heart_rate=70)
            # algoPpgSensorBuff = list(arr)
            
            # ppg_cleaned = nk.ppg_clean(algoPpgSensorBuff)
            ppg_signals, info = nk.ppg_process(algoPpgSensorBuff, sampling_rate = SAMPLE_RATE)
            f_hrv = nk.hrv(ppg_signals)
            t_hrv = nk.hrv_time(ppg_signals)
            print("frequency domain hrv", f_hrv)
            print("time domain hrv", t_hrv)
            print("elaspsed time:", time.time()-start)
            

class SenReaderThread(Thread):
    def run(self):
        while True:
            # Read current ADC conversion
            num = adc.read_adc(0, gain=GAIN) 
            ppgSensorDataQueue.put(num)      # ---- automatically synchronized 
            time.sleep(ADC_CONVERSION_RATE)

class SenDataCollectorThread(Thread):
    def run(self):
        global AnalyzerDataBuff
        senDataBuff1 = []
        senDataBuff2 = []
        senDataBuff3 = []
        
        # Each element in the list store a minute of sensor data 
        ThreeMinSenDataList = [senDataBuff1, senDataBuff2, senDataBuff3]
        
        
        #Fills the first three minutes sensor data
        for OneMinSenData in ThreeMinSenDataList:
            start = time.time()
            while (time.time() - start <= COLLECTION_TIME):
                lastSendData = ppgSensorDataQueue.get(True) # block if work queue empty ------
                OneMinSenData.append(lastSendData)
                # print(OneMinSenData , ppgSensorDataQueue.qsize())

        # combine and send list
        AnalyzerDataBuff = ThreeMinSenDataList[0] + ThreeMinSenDataList[1] + ThreeMinSenDataList[2]
        
        # print(len(AnalyzerDataBuff))
        print(ppgSensorDataQueue.qsize())
        #add flag to consume AnalyzerDataBuff
        isDataReady.set()

        while True:
            for x in range(0,3):
                print("Filling bucket", x)
                ThreeMinSenDataList[x].clear()
                start = time.time()
                while (time.time() - start <= COLLECTION_TIME):
                    num = ppgSensorDataQueue.get(True) # block if work queue empty ------
                    ThreeMinSenDataList[x].append(num)
                    # print(ThreeMinSenDataList[x], ppgSensorDataQueue.qsize())
                #combine and send list 
                
                if x == 0:
                    AnalyzerDataBuff = senDataBuff2 + senDataBuff3 + senDataBuff1
                elif x == 1:
                    AnalyzerDataBuff = senDataBuff3 + senDataBuff1 + senDataBuff2
                else:
                    AnalyzerDataBuff = senDataBuff1 + senDataBuff2 + senDataBuff3
                    
                # print(len(AnalyzerDataBuff))
                print(time.time()-start)
                isDataReady.set()

producer = SenReaderThread().start()
consumer = SenDataCollectorThread().start()
analyzer = ppgAlgoThread().start()