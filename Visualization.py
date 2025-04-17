import matplotlib.pyplot as plt
import torch
from Сalculation_metrics import find_onsets_and_offsets_segments

def vizualization_signal(signal, fig, ax):
    
    ax.plot(signal, '-k', alpha = 0.25)
    
    ax.plot(signal, 'ok', markersize=2)
    
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.grid(which='major', axis='both', linestyle='-', alpha=0.75)
    


    
def visualize_ecg_segments(signal, label, label_pred):

    # Получаем метки классов (argmax по классам)
    # Карта сегментации - 0, 1, 2, 3
    label_segment_mask = torch.argmax(label, dim=0).numpy()
    label_pred_segment_mask = torch.argmax(label_pred, dim=0).numpy()
    
    print(label_segment_mask)
    print(label_pred_segment_mask)
    
    fig, ax = plt.subplots()
    
    vizualization_signal(signal, fig, ax)
    
    """
    0 - QRS
    1 - T
    2 - P
    3 - Background
    """

    # Цвета для разных сегментов
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    labels = {0: 'QRS', 1: 'T', 2: 'P'}

    # Отрисовка ground truth сегментов
    for class_id in [0]:
        segments_onsets, segments_offsets = find_onsets_and_offsets_segments(label_segment_mask, class_id)
        
        for start, end in zip(segments_onsets, segments_offsets):
            
            # plt.scatter(start, signal[start], color=colors[class_id], marker='o',label=f'GT {labels[class_id]}', s=50)
            plt.scatter(end, signal[end], color=colors[class_id], marker='o', label=f'GT {labels[class_id]}',s=50)
            
            
    # Отрисовка предсказанных сегментов
    for class_id in [0]:
        segments_onsets, segments_offsets = find_onsets_and_offsets_segments(label_pred_segment_mask, class_id)
        for start, end in zip(segments_onsets, segments_offsets):
            ax.axvspan(end, end, 
                       color=colors[class_id], 
                       # alpha = 0.2, 
                       label=f'Pred {labels[class_id]}')
        
        
    
    # Настройка легенды
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Удаляем дубликаты
    ax.legend(by_label.values(), by_label.keys(), 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left')
    

    plt.tight_layout()
    plt.show()

