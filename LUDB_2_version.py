import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import wfdb
import os

import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from Preprocess import preprocess_ecg

from UNet_ECG import UNet

def vizualization_signal(signal, fig, ax):
    
    ax.plot(signal, '-k', alpha = 0.25)
    
    ax.plot(signal, 'ok', markersize=2)
    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.grid(which='major', axis='both', linestyle='-', alpha=0.75)

class ECGDataset(Dataset):
    def __init__(self, ecg_data, labels):
        self.ecg_data = ecg_data
        self.labels = labels
        
    def __len__(self):
        return len(self.ecg_data)
    
    def __getitem__(self, idx):
        return self.ecg_data[idx], self.labels[idx]
    
    
    

def process_patient(base_filename, data_dir, lead = "i", start_index = 1000, end_index = 4001):
    
    
    record_path = os.path.join(data_dir, f"{base_filename}")
    
    record = wfdb.rdrecord(record_path, physical=False)
    
    # Получаем названия отведений
    lead_names = record.sig_name
    
    idx_lead = lead_names.index(lead)
    
    # adc_signal = record.d_signal[:, idx_lead]  # Берём один канал
       
    # # Преобразуем в физические единицы (мВ)
    # gain = record.adc_gain[idx_lead]  # Коэффициент усиления (например, 1000)
    # baseline = record.baseline[idx_lead]  # Смещение (обычно 0)
    # ecg_mv = (adc_signal - baseline) / gain
    
    ecg_mv = record.d_signal[:, idx_lead]
      
    # Чтение аннотаций
    annotation = wfdb.rdann(record_path, lead).sample
    
    background = [1]*len(ecg_mv)

    qrs = [0]*len(ecg_mv)

    for i in range(0, len(annotation), 9):
        for k in range(annotation[i], annotation[i+2]+1):
            qrs[k] = 1
            background[k] = 0
            
            
    t = [0]*len(ecg_mv)
    for i in range(3, len(annotation), 9):
        for k in range(annotation[i], annotation[i+2]+1):
            t[k] = 1
            background[k] = 0

            
            
    p = [0]*len(ecg_mv) 
    for i in range(6, len(annotation), 9):
        for k in range(annotation[i], annotation[i+2]+1):
            p[k] = 1
            background[k] = 0

            

    label = torch.stack([torch.FloatTensor(qrs[start_index:end_index]), # 1-ый класс
                         torch.FloatTensor(t[start_index:end_index]),   # 2-ой класс
                         torch.FloatTensor(p[start_index:end_index]),   # 3-ий класс
                         torch.FloatTensor(background[start_index:end_index])], dim=0)  # 4-ый класс
    
    
    ecg_tensor = torch.FloatTensor(ecg_mv[start_index:end_index])
    
        
    return ecg_tensor, label
        
        
           
def create_and_save_dataset(data_dir, lead, test_size=0.2, random_seed=42):
        
    ecg_signals = []
    
    labels = []
    
    for i in range(1, 201):
        
        ecg_mv, label = process_patient(str(i), data_dir, lead = lead)
        
        ecg_signals.append(ecg_mv)
        labels.append(label)
        
    # Создание и сохранение датасета
    dataset = ECGDataset(ecg_signals, labels)

    # Разделение на train/test
    test_count = int(len(dataset) * test_size)
    train_count = len(dataset) - test_count
    
    # Фиксируем random seed для воспроизводимости
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_count, test_count],
        generator=generator
    )
        
    # Сохранение датасетов
    torch.save(train_dataset, f'LUDB_{lead}_train_dataset_ADC.pt')
    torch.save(test_dataset, f'LUDB_{lead}_test_dataset_ADC.pt')
    
    
def train_model(model, train_loader,lead, epochs=50, lr=0.001):
   
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for multi-label classification
    
    optimizer = optim.Adam(model.parameters(), lr=lr)    
    
    # цикл по количеству эпох обучения
    for epoch in range(epochs):
        
        model.train()
        
        # цикл по батчам даталоадера
        for signals, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            
            # Обнулим сохраненные у оптимизатора значения градиентов
            # перед следующим шагом обучения
            optimizer.zero_grad()
            
            # Вычислим предсказания нашей модели
            outputs = model(signals.unsqueeze(1))
            
            # Посчитаем значение функции потерь на полученном предсказании
            # ошибка
            loss = criterion(outputs, labels)
            
            # Выполним подсчёт новых градиентов
            loss.backward()
            
            # Выполним шаг градиентного спуска
            optimizer.step()
            
    # сохраним обученную модель
    torch.save(model, f"UNet_{lead}_{epochs}_ADC.pth")
    

if __name__ == "__main__":
    
    data_dir = '../LUDB'
    lead = "i"
    
  
        
    # create_and_save_dataset(data_dir, lead)
    
    
    
    train_dataset = torch.load(f'LUDB_{lead}_train_dataset_ADC.pt', weights_only=False)
    test_dataset = torch.load(f'LUDB_{lead}_test_dataset_ADC.pt', weights_only=False)

    # # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Создание модели
    model = UNet(n_channels=1, n_classes=4)
    
    # Обучение и сохранение модели
    epochs=50
    # train_model(model, train_loader, lead, epochs=50, lr=0.001)
    
    trained_model = torch.load(f"UNet_{lead}_{epochs}_ADC.pth", weights_only=False)
    
    signals, labels = next(iter(test_loader))
    
    k = 1
    
    for signals, labels in test_loader:
        k+=1
        if k == 5:
            break
        
        num = 0
        signal = signals[num]
        # label_true = labels[num]
        test_signal = signal.unsqueeze(0).unsqueeze(0)
        
        
            
        with torch.no_grad():
            output = trained_model(test_signal)
        label_pred = output[0]
        
        print(f"Input shape: {test_signal.shape}")
        print(f"Output shape: {output.shape}")  # Should be (1, 4, 1000)
        
        
        label = torch.argmax(label_pred, dim=0).numpy()
        
        print(set(label))
        
        fig, ax = plt.subplots()
        
        vizualization_signal(signal, fig, ax)
        
        colors = {
            0: 'r',    # QRS
            1: 'b',   # T
            2: 'g',  # P
            3: 'k'    # Background
            }
        
        for i in range(len(label)):
            ax.plot(i, signal[i], f'o{colors[label[i]]}', markersize=4)
            
        plt.show()
        
        
    
    
    
    
    
    
    
    
    
    
    
    




    
    


