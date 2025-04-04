

def find_subarrays_with_sum(numbers, target_sum):
    result = []
    
    n = len(numbers)
    
    for start in range(n):
        cur_sum = 0
        
        for end in range(start, n):
            
            cur_sum +=numbers[end]
            
            if cur_sum == target_sum:
                
                result.append(numbers[start:end+1])
            
    return result


arr = [1, 3, 4, 0, 5, -1]

out = find_subarrays_with_sum(arr, 4)

# print(out)


def max_unique_subarray_length(numbers):
    
    max_length = 0
    
    n = len(numbers)
    
    for start in range(n):
        
        unique_elements = set()
        
        current_length = 0
        
        for end in range(start, n):
            
            if numbers[end] in unique_elements:
                break  # Прерываем, если элемент уже есть в подмассиве
                
            unique_elements.add(numbers[end])
            current_length += 1
            
            if current_length > max_length:
                max_length = current_length
    
    return max_length


ml = max_unique_subarray_length([1, 1])

print(ml)


            
                        
           
    
    
    
    