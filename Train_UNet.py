import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import wfdb
import os

from UNet_ECG import UNet1D

# Assuming UNet1D class is defined as provided

# 1. Custom Dataset Class
class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.FloatTensor(self.signals[idx]).unsqueeze(0)  # (1, L)
        label = torch.FloatTensor(self.labels[idx])  # (4, L)
        return signal, label

# 2. Data Preparation (example with synthetic data)
def read_LUDB():
    
    # Список отведений (leads) из вашего файла
    leads = ['avf', 'avl', 'avr', 'i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    
    # Базовое имя файла (без расширения)
    base_filename = '1'  # соответствует "1.avf", "1.avl" и т.д.
    
    # Путь к директории с файлами (замените на ваш путь)
    data_dir = '../LUDB'  # текущая директория
    


    # Прочитаем сигналы и метаданные

    """
    Чтение ЭКГ-сигналов и метаданных для указанных отведений.
    
    Параметры:
        base_name (str): Базовое имя файла (например, '1')
        leads (list): Список отведений (например, ['avf', 'avl', ...])
        directory (str): Путь к директории с файлами
        
    Возвращает:
        dict: Словарь с сигналами и метаданными для каждого отведения
    """
    ecg_data = {}
    
    for lead in leads:
        try:
            # Формируем полный путь к файлу (без расширения)
            record_path = os.path.join(data_dir, f"{base_filename}")
            
            # Читаем сигнал и метаданные
            signals, fields = wfdb.rdsamp(record_path)
            
            annotation_extension = 'i'  # Расширение файла аннотаций

            # Чтение аннотаций
            annotation = wfdb.rdann(record_path, lead)
            
            print(signals)
            print(fields)
            
            print(len(annotation.sample))
                    
            # Сохраняем данные
            ecg_data[lead.upper()] = {
                'signal': signals,
                'fields': fields,
                'annotation':annotation.sample
            }
            print(f"Успешно прочитано отведение {lead.upper()}")
            
        except Exception as e:
            print(f"Ошибка при чтении {lead.upper()}: {str(e)}")
    

    # Пример доступа к данным
    if 'AVF' in ecg_data:
        print(f"\nПример данных для AVF:")
        print(f"Форма сигнала: {ecg_data['AVF']['signal'].shape}")
        print(f"Частота дискретизации: {ecg_data['AVF']['fields']['fs']} Гц")
        print(f"Количество samples: {ecg_data['AVF']['fields']['sig_len']}")
    

# 3. Training Setup
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        
        for signals, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
        
        # Calculate average losses
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()
    
    return model

# 4. Main Training Pipeline
def main():
    
    read_LUDB()
    
    # # Split into train/val
    # X_train, X_val, y_train, y_val = train_test_split(signals, labels, test_size=0.2, random_state=42)
    
    # # Create datasets and dataloaders
    # train_dataset = ECGDataset(X_train, y_train)
    # val_dataset = ECGDataset(X_val, y_val)
    
    # batch_size = 8
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # # Initialize model
    # model = UNet1D(n_channels=1, n_classes=4)
    
    # # Train
    # trained_model = train_model(model, train_loader, val_loader, epochs=50, lr=0.001)
    
    # # Save final model
    # torch.save(trained_model.state_dict(), 'final_model.pth')

if __name__ == "__main__":
    main()
