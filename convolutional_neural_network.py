import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchvision
from torchvision import transforms
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False)

classes = tuple(str(i) for i in range(10))

print(trainloader.dataset.train_data.shape)

print(testloader.dataset.test_data.shape)

# print(trainloader.dataset.train_data[0])

# преобразовать тензор в np.array
numpy_img = trainloader.dataset.train_data[0].numpy()

print(numpy_img.shape)

plt.imshow(numpy_img, cmap='gray')
  

import torch
import numpy as np
import torch.nn as nn


# ЗАМЕТЬТЕ: КЛАСС НАСЛЕДУЕТСЯ ОТ nn.Module

class SimpleConvNet(nn.Module):
    def __init__(self):
        # вызов конструктора предка
        super(SimpleConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(4 * 4 * 16, 120)  # !!!
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.act = nn.ReLU()

    def forward(self, x):
        
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        
        #print(x.shape)
        x = x.view(-1, 4 * 4 * 16)  # !!!
        
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        
        x = self.fc3(x)
        
        return x




from tqdm import tqdm_notebook

model = SimpleConvNet()

# criterion = torch.nn.CrossEntropyLoss()


# for X_batch, y_batch in trainloader:
        
#     print(X_batch.shape)
#     print(y_batch)
        
#     y_pred = model(X_batch)
    
#     print(y_pred)

#     _, predicted = torch.max(y_pred, 1)
    
#     c = (predicted.detach() == y_batch)
    
#     print(c)
    
#     loss = criterion(y_pred, y_batch)
    
#     print(loss)

    # break


def Train(model, trainloader):

    
    # выбираем функцию потерь
    criterion = torch.nn.CrossEntropyLoss()

    # выбираем алгоритм оптимизации и learning_rate
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # итерируемся
    for epoch in tqdm_notebook(range(2)):
    
        epoch_loss = []

        # цикл по батчам даталоадера
        for X_batch, y_batch in trainloader:

            # Вычислим предсказания нашей модели
            y_pred = model(X_batch)
            
            # Посчитаем значение функции потерь  на полученном предсказании
            loss = criterion(y_pred, y_batch)
            epoch_loss.append(loss.item())
            print(loss.item())

            # Выполним подсчёт новых градиентов
            loss.backward()
            # Выполним шаг градиентного спуска
            optimizer.step()
            # Обнулим сохраненные у оптимизатора значения градиентов
            # перед следующим шагом обучения
            optimizer.zero_grad()
            
        print("curr_loss", np.mean(epoch_loss))
    
    print('Обучение закончено')
    
    return model

def Test(trained_model, testloader):

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in testloader:
            y_pred = model(images)
            _, predicted = torch.max(y_pred, 1)
            
            c = (predicted.detach() == labels)
                    
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        
        

trained_model = Train(model, trainloader)

Test(trained_model, testloader)
        
# img = np.random.randint(0, 255, (28, 28))
# out = model(torch.FloatTensor(img).unsqueeze(0).unsqueeze(0))
# _, predicted = torch.max(out, 1)
# print(predicted)
