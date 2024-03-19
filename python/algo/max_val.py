def max_value(num):
    list_num = [int(i) for i in str(num)]
    list_num_with_index = zip(range(list_num), list_num)
    sorted_num = sorted(list_num_with_index)
    tmp = 0
    for i in range(len(list_num)-1):
        if list_num[i] != sorted_num[i][1] and sorted_num[i][1]!=sorted_num[i+1][1]:
            tmp = list_num[i]
            list_num[i] = sorted_num[i][1]
            list_num[sorted_num[i][0]] = tmp
        
            break

    str_num = ''.join(list_num)
    return int(str_num)
