from torch.utils.data import Dataset, random_split
import wfdb
import os
import torch


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
