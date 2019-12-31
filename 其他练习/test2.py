f = open('test.txt', 'r')
total_list = []
while True:
    temp = f.readline()
    temp = temp[:-1]
    total_list.append(temp)
    if not temp:
        break
f_temp = open()
total_list = total_list[100:]

flag = 0
read_index = 0
while True:
    temp_list = []

    # 读取列表中二十个
    for i in range(20):
        try:
            temp  = total_list.pop(0)
        except Exception as e:
            flag = 1
            break

        # todo：拆包

        temp_list.append(temp)

    if flag == 1: break

    # todo:存储
    pass

    read_index += 20
    f_temp = open()
    f_temp.close()


f.close()