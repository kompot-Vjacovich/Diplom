import time
import continuous_threading


time_list = []

def save_time():
    time_list.append(time.time())
    print(time_list)

th = continuous_threading.PeriodicThread(1.0, save_time)
th.start()

time.sleep(4)
th.join()