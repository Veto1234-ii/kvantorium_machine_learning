import numpy as np
from scipy.signal import medfilt, firwin, lfilter

def preprocess_ecg(record, lead = "i"):
    
    # Получаем названия отведений
    lead_names = record.sig_name
    
    idx_lead = lead_names.index(lead)
    
    adc_signal = record.d_signal[:, idx_lead]  # Берём один канал
    
    fs = record.fs
   
    # Преобразуем в физические единицы (мВ)
    gain = record.adc_gain[idx_lead]  # Коэффициент усиления (например, 1000)
    baseline = record.baseline[idx_lead]  # Смещение (обычно 0)
    ecg_mv = (adc_signal - baseline) / gain
    
    # 1. Применим медианный фильтр шириной 200 мс
    kernel_200ms = int(fs * 0.2)
    if kernel_200ms % 2 == 0:  # ядро медианного фильтра должно быть нечетным
        kernel_200ms += 1
    ecg_200 = medfilt(ecg_mv, kernel_size=kernel_200ms)

    # 2. Применим медианный фильтр шириной 600 мс
    kernel_600ms = int(fs * 0.6)
    if kernel_600ms % 2 == 0:
        kernel_600ms += 1
    baseline = medfilt(ecg_200, kernel_size=kernel_600ms)

    # 3. Вычитаем базовую линию
    ecg_baseline_corrected = ecg_mv - baseline

    # 4. Создаём КИХ low-pass фильтр (FIR) на 35 Гц
    nyquist = fs / 2
    cutoff_hz = 35
    numtaps = 12
    fir_coeff = firwin(numtaps, cutoff_hz / nyquist)

    # 5. Применяем фильтр к сигналу
    ecg_pre_processed = lfilter(fir_coeff, 1.0, ecg_baseline_corrected)

    return ecg_pre_processed



