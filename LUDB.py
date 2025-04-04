import wfdb
import os

import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from pyedflib import highlevel 
from scipy.signal import butter, filtfilt, find_peaks
import json



def vizualization_signal(signal, fig, ax):
    
    ax.plot(signal, '-k', alpha = 0.25)
    
    ax.plot(signal, 'ok', markersize=2)
    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.grid(which='major', axis='both', linestyle='-', alpha=0.75)

def read_LUDB():
    
        
    base_filename = '198'  
    
    data_dir = '../LUDB'  
    
    record_path = os.path.join(data_dir, f"{base_filename}")
    
    record = wfdb.rdrecord(record_path, physical=False)
   
    # Получаем названия отведений
    lead_names = record.sig_name

    ecg_data = {}
    
    for i, lead in enumerate(lead_names):
        
        adc_signal = record.d_signal[:, i]  # Берём один канал
       
        # Преобразуем в физические единицы (мВ)
        gain = record.adc_gain[i]  # Коэффициент усиления (например, 1000)
        baseline = record.baseline[i]  # Смещение (обычно 0)
        ecg_mv = (adc_signal - baseline) / gain
        
        
        # Чтение аннотаций
        annotation = wfdb.rdann(record_path, lead)
                
        # Сохраняем данные
        ecg_data[lead] = {
            'signal': ecg_mv,
            'annotation':annotation.sample
        }
        


    return ecg_data


data = read_LUDB()

print(data.keys())

lead = data["i"]



signal = lead["signal"]
annotation = lead["annotation"]

print(signal.shape)
print(signal)

print(annotation)

fig, ax = plt.subplots()

vizualization_signal(signal, fig, ax)

# ax.plot(annotation, [signal[i] for i in annotation], 'or')




for i in range(0, len(annotation) - 3, 9):

    for k in range(annotation[i], annotation[i+2]):
        
        ax.plot(k, signal[k], 'or')
        
for i in range(3, len(annotation) - 3, 9):

    for k in range(annotation[i], annotation[i+2]):
        
        ax.plot(k, signal[k], 'og')
        
        
for i in range(6, len(annotation) - 3, 9):

    for k in range(annotation[i], annotation[i+2]):
        
        ax.plot(k, signal[k], 'ob')

   
plt.show()
 
    
