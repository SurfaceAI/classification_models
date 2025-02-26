import sys
sys.path.append('.')

from src.models import prediction
from experiments.config import predict_config
import time
import os
import psutil
import threading

# def monitor_cpu():
#     pid = os.getpid()
#     process = psutil.Process(pid)
    
#     while True:
#         cpu_usage = process.cpu_percent(interval=1)
#         print(f"CPU usage: {cpu_usage}%")
#         # Anzahl der logischen Kerne
#         num_cpus = psutil.cpu_count()
#         # Anzahl der tatsächlich genutzten Kerne berechnen
#         cores_used = (cpu_usage / 100) * num_cpus
#         print(f"Number of CPU cores used: {cores_used}")
#         # Anzahl der Threads, die der Prozess verwendet
#         num_threads = process.num_threads()
#         print(f"Number of threads: {num_threads}")
#         time.sleep(1)  # 1 Sekunde Pause, um nicht ständig zu aktualisieren

# # Monitoring in einem separaten Thread starten
# monitor_thread = threading.Thread(target=monitor_cpu)
# monitor_thread.daemon = True  # Thread im Hintergrund laufen lassen
# monitor_thread.start()

start = time.time()
prediction.run_dataset_predict_csv(predict_config.CC_V1_0)
end = time.time()
print(end-start)