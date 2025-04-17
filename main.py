import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.data import DataLoader

import glob
import numpy as np

import torch
import torch.nn as nn

from Preprocess import preprocess_ecg

from UNet_ECG import UNet
    
from Visualization import visualize_ecg_segments

from Creating_dataset_training_testing import ECGDataset

from Сalculation_metrics import Calculating_metrics_test
    
if __name__ == "__main__":
    
    data_dir = '../LUDB'
    lead = "i"
    
  
        
    # create_and_save_dataset(data_dir, lead)
    
    train_dataset = torch.load(f'LUDB_{lead}_train_dataset_ADC.pt', weights_only=False)
    test_dataset = torch.load(f'LUDB_{lead}_test_dataset_ADC.pt', weights_only=False)

    # # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Создание модели
    # model = UNet(n_channels=1, n_classes=4)
    
    # Обучение и сохранение модели
    epochs=50
    # train_model(model, train_loader, lead, epochs=50, lr=0.001)
    
    trained_model = torch.load(f"UNet_{lead}_{epochs}_ADC.pth", weights_only=False)
    
    # signal, label = next(iter(test_loader))

    k = 0
    for signal_batch, label_batch in test_loader:
        k+=1
        if k == 5:
            break
        
        with torch.no_grad():
            output = trained_model(signal_batch.unsqueeze(0))
  
        
        signal = signal_batch[0]
        label = label_batch[0]
        label_pred = output[0]
    
        visualize_ecg_segments(signal, label, label_pred)    
    
    # Calculating_metrics_test(test_loader, 1, trained_model)
    
    
    
    
    # fig, ax = plt.subplots()
    # vizualization_signal(signal, fig, ax)
    # colors = {
    #     0: 'r',    # QRS
    #     1: 'b',   # T
    #     2: 'g',  # P
    #     3: 'k'    # Background
    #     }
    # for i in range(len(label)):
    #     ax.plot(i, signal[i], f'o{colors[label[i]]}', markersize=4)  
    # plt.show()
    
    








        
        
            
        
    
    
    
    




    
    


