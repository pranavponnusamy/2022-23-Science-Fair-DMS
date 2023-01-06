import threading
import random
import time
import queue
import psutil

from threading import Thread, Event

GAIN = 2/3
COLLECTION_TIME = 5
SAMPLE_RATE = 1000

# Global list
my_list = []

# Lock to synchronize access to the list
lock = threading.Lock()

q = queue.Queue()

process = Event()
process.clear()




def ProducerThread():
    while True:
        # num = adc.read_adc(0, gain=GAIN)
        num = random.randint(0,10)
        q.put(num)      # ---- automatically synchronized 
        time.sleep(0.5)

        
def ConsumerThread():
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
    while True:
        process.wait()
        print("Analyzer", readings)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        start = time.time()
        # arr = nk.ppg_simulate(duration=180, sampling_rate=1000, heart_rate=70)
        # ppg_signal = list(arr)
        ppg_cleaned = nk.ppg_clean(readings)
        ppg_signals, info = nk.ppg_process(ppg_cleaned, sampling_rate = SAMPLE_RATE)
        f_hrv = nk.hrv(ppg_signals)
        t_hrv = nk.hrv_time(ppg_signals)
        print("frequency domain hrv", f_hrv)
        print("time domain hrv", t_hrv)
        print("elaspsed time:", time.time()-start)
        # ppg_signal.clear()
        process.clear()
        readings.clear()


def thread_function1():
    # Acquire the lock to synchronize access to the list
    with lock:
        # Append to the list
        my_list.append(1)

def thread_function2():
    # Acquire the lock to synchronize access to the list
    with lock:
        # Append to the list
        my_list.append(2)

# Create and start the threads
thread1 = threading.Thread(target=thread_function1)
thread2 = threading.Thread(target=thread_function2)
thread1.start()
thread2.start()

producer = threading.Thread(target=ProducerThread)
consumer = threading.Thread(target=ConsumerThread)
analyzer = threading.Thread(target=AnalyzerThread)
producer.start()
consumer.start()
analyzer.start()

# Wait for the threads to finish
thread1.join()
thread2.join()

# Print the final state of the list
print(my_list)