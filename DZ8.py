import torch

# Создайте тензор размерности (5, 5), заполненный числами от 0 до 24
tensor = torch.arange(25).reshape(5, 5)
print("Исходный тензор:")
print(tensor)

# Выведите второй столбец тензора
second_column = tensor[:, 1]
print("\nВторой столбец:")
print(second_column)

# Выведите первую строку тензора
first_row = tensor[0, :]
print("\nПервая строка:")
print(first_row)

# Выведите подматрицу размерности (2, 2) из центра тензора
center_submatrix = tensor[2:4, 2:4]
print("\nПодматрица из центра:")
print(center_submatrix)

# Создайте тензор размерности (4, 4), заполненный случайными числами от 0 до 9
random_tensor = torch.randint(0, 10, (4, 4))
print("\nСлучайный тензор:")
print(random_tensor)

# Найдите индексы всех элементов, которые больше 5
indices = random_tensor > 5
print("\nИндексы элементов, которые больше 5:")
print(indices)

# Используя найденные индексы, измените значения этих элементов на -1
random_tensor[indices] = -1
print("\nТензор после изменения:")
print(random_tensor)
