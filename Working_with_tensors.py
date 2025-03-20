import torch

"""1. Тензоры можно создавать из листов, массивов и других контейнеров Python."""

x_list = [1, 2, 3]

x_tensor = torch.tensor(x_list)

print(x_tensor.size())

"""2. Тензоры можно создавать при помощи инициализирующих функций, как в numpy."""

# тензор из нулей заданного размера
zeros_tensor = torch.zeros(2, 3)
# print(zeros_tensor)

# тензор из единиц заданного размера
ones_tensor = torch.ones(2, 3)
# print(ones_tensor)

"""3. Вообще практически все методы, которые есть у Numpy массивов, есть и у torch.Tensor. С тензорами также можно производить операции:"""

x_tensor = torch.tensor([
    [1, 2],
    [3, 4]
])

y_tensor = torch.tensor([
    [-10, 3],
    [5, -4]
])

print(y_tensor.size())

# сложение
print(x_tensor + y_tensor)

# вычитание
print(x_tensor - y_tensor)

# аналог np.concatenate([x_tensor, y_tensor], axis=1)
print(torch.cat([x_tensor, y_tensor], dim=0))

"""4. Тензоры можно переводить обратно в формат Numpy или питоновских значений"""

x_tensor = torch.tensor([
    [1, 2],
    [3, 4]
])

x_numpy = x_tensor.numpy()
print(type(x_numpy))



# многомерный тензор
x_tensor = torch.tensor([
    [1, 2],
    [3, 4]
])

print(x_tensor.tolist())


# одномерный тензор
x_tensor = torch.tensor([3])

print(x_tensor.item())
