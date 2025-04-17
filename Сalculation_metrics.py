import torch

# Функция для нахождения границ сегментов
def find_onsets_and_offsets_segments(mask, class_id):
    
    segments_onsets = []
    segments_offsets = []
    
    start = None
    
    for i in range(len(mask)):
        if mask[i] == class_id and start is None:
            start = i
        elif mask[i] != class_id and start is not None:
            if start != i-1:
                segments_onsets.append(start)
                segments_offsets.append(i-1)
                start = None
            
    # Добавляем последний сегмент, если он есть
    if start is not None:
        segments_onsets.append(start)
        segments_offsets.append(len(mask)-1)
        
    return segments_onsets, segments_offsets



def Calculating_metrics_test(test_loader, class_id, trained_model):
    
    true_delinations_onset = []
    our_delinations_onset  = []
    
    true_delinations_offset = []
    our_delinations_offset  = []
    
    
    for signal_batch, label_batch in test_loader:
        
        with torch.no_grad():
            output = trained_model(signal_batch.unsqueeze(0))
                
        
        label = label_batch[0]
        label_pred = output[0]
        
        label_segment_mask = torch.argmax(label, dim=0).numpy()
        label_pred_segment_mask = torch.argmax(label_pred, dim=0).numpy()
        
        segments_onsets, segments_offsets = find_onsets_and_offsets_segments(label_segment_mask, class_id)
        true_delinations_onset.append(segments_onsets)
        true_delinations_offset.append(segments_offsets)
        
        
        segments_onsets_pred, segments_offsets_pred = find_onsets_and_offsets_segments(label_pred_segment_mask, class_id)
        our_delinations_onset.append(segments_onsets_pred)
        our_delinations_offset.append(segments_offsets_pred)
        
    
    recall_onset, precision_onset, F1_onset, mean_err_onset = get_F1(true_delinations_onset,
                                                                     our_delinations_onset)
    
    
    recall_offset, precision_offset, F1_offset, mean_err_offset = get_F1(true_delinations_offset,
                                                                     our_delinations_offset)
    
    labels = {0: 'QRS', 1: 'T', 2: 'P'}
    
    print(f"\t   {labels[class_id]} onset {labels[class_id]} offset")
    print(f"Se (%)  {round(recall_onset*100, 2)}   {round(recall_offset*100, 2)}")
    print(f"PPV (%) {round(precision_onset*100, 2)}   {round(precision_offset*100, 2)}")
    print(f"F1 (%)  {round(F1_onset*100, 2)}   {round(F1_offset*100, 2)}")
    print(f"m       {round(mean_err_onset, 2)}     {round(mean_err_offset, 2)}")


    

def get_F1(true_delinations, our_delinations, TOLERANCE = 75):
    
    pairs = []

    TP = 0
    FP = 0
    FN = 0


    for i in range(len(true_delinations)):


        doctor_labels = true_delinations[i]
        program_labels = our_delinations[i]

       
        if len(doctor_labels) == 0:
            continue

        TP_ = 0  # True Positives
        FP_ = 0  # False Positives
        FN_ = 0  # False Negatives

        # Копируем программную разметку, чтобы удалять использованные точки
        available_program_labels = program_labels

        # Проверяем каждую точку докторской разметки
        for doctor_point in doctor_labels:
            # Находим ближайшую точку программной разметки
            # в пределах толерантного окна
            min_distance = float('inf')
            closest_point = None
            for program_point in available_program_labels:
                distance = abs(program_point - doctor_point)
                if distance <= TOLERANCE and distance < min_distance:
                    min_distance = distance
                    closest_point = program_point
            # Если найдена ближайшая точка, 
            # фиксируем её как TP и удаляем из доступных
            if closest_point is not None:
                TP_ += 1
                pairs.append((doctor_point, closest_point))
                available_program_labels.remove(closest_point)
            else:
                FN_ += 1
        # Оставшиеся точки программной разметки считаем как FP
        FP_ = len(available_program_labels)

        TP += TP_
        FP += FP_
        FN += FN_

    # F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # mean_err
    total_distance = 0
    for doctor_point, program_point in pairs:
        distance = abs(doctor_point - program_point)
        total_distance += distance

    if len(pairs) != 0:
        mean_err = total_distance / len(pairs)
    else:
        mean_err = None

    return recall, precision, F1, mean_err


