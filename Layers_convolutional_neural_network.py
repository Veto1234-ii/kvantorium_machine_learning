import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Создаем сверточный слой
conv = nn.Conv2d(in_channels=1,  # Количество каналов входного изображения
                 out_channels=6, # Количество каналов, полученных в результате свертки
                 kernel_size=5,  # Размер сверточного ядра
                 stride=1)       # Шаг свертки


# Создаем функцию активации
act = nn.ReLU()

# Создаем слой подвыборки (MaxPooling)
pool = nn.MaxPool2d(kernel_size=2, stride=1)

# Создаем изображение
image = np.zeros((32, 32))
image[:, 16] = 1
image[10, :17] = 1

# Преобразуем изображение в тензор и добавляем размерности для батча и каналов
image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)  # [1, 1, 32, 32]

# Применяем свертку
image_conv = conv(image_tensor)  # Выход будет размерности [1, 6, 28, 28]

# Применяем функцию активации
image_act = act(image_conv)

# Применяем MaxPooling
image_pool = pool(image_act)

print(image_pool.shape)


# # так как ожидается одномерный вектор
# x = image_pool.view(-1, 16 * 5 * 5)

# linear_layer = nn.Linear(in_features = 16 * 5 * 5, out_features = 100)

# res = linear_layer(x)

# print(res)

# Создаем окно с четырьмя субплогами
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Визуализируем исходное изображение
axes[0].imshow(image, cmap='gray')
axes[0].set_title(f"Исходное изображение\nРазмер: {image.shape}")
axes[0].axis('off')

# Визуализируем результат свертки (первый канал)
axes[1].imshow(image_conv[0, 0].detach().numpy(), cmap='gray')
axes[1].set_title(f"Результат свертки\nРазмер: {image_conv[0, 0].shape}")
axes[1].axis('off')

# Визуализируем результат применения функции активации (первый канал)
axes[2].imshow(image_act[0, 0].detach().numpy(), cmap='gray')
axes[2].set_title(f"Результат активации\nРазмер: {image_act[0, 0].shape}")
axes[2].axis('off')

# Визуализируем результат MaxPooling (первый канал)
axes[3].imshow(image_pool[0, 0].detach().numpy(), cmap='gray')
axes[3].set_title(f"Результат MaxPooling\nРазмер: {image_pool[0, 0].shape}")
axes[3].axis('off')

# Показываем окно с графиками
plt.tight_layout()
plt.show()






