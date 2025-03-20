# import torch
# import torchvision
# from torchvision import transforms
# from torchsummary import summary

# import numpy as np
# import matplotlib.pyplot as plt  # для отрисовки картиночек

# # Проверяем, доступны ли GPU
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transform = transforms.Compose(
#     [transforms.ToTensor()])

# trainset = torchvision.datasets.MNIST(root='./data', train=True, 
#                                       download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                      download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

# classes = tuple(str(i) for i in range(10))

# print(trainloader.dataset.train_data.shape)

# print(testloader.dataset.test_data.shape)

# print(trainloader.dataset.train_data[0])

# # преобразовать тензор в np.array
# numpy_img = trainloader.dataset.train_data[0].numpy()

# print(numpy_img.shape)

# plt.imshow(numpy_img, cmap='gray')








import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F  # Functional


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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 4 * 4 * 16)  # !!!
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleConvNet()

img = np.random.randint(0, 255, (28, 28))

out = model(torch.FloatTensor(img).unsqueeze(0).unsqueeze(0))

_, predicted = torch.max(out, 1)


print(predicted)
