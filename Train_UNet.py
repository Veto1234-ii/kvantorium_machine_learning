import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
    

