import multiprocessing
import time


def sub_fuc(sub_dict:dict):
    sub_dict["ss"] = "tt"


def func(mydict:dict, mylist):
    mydict["index1"] = {"a": "b"}  # 子进程改变dict,主进程跟着改变
    local_dict =  mydict["index1"]
    sub_fuc(mydict)
    mydict["index2"] = "bbbbbb"
    # mydict.ex
    # mylist.append(11)  # 子进程改变List,主进程跟着改变
    # mylist.append(22)
    # mylist.append(33)


def func2(mydict, mylist:multiprocessing.Array):
    mydict["index3"] = "cccccc"  # 子进程改变dict,主进程跟着改变
    mydict["index4"] = "dddddd"
    mylist[0] = 44  # 子进程改变List,主进程跟着改变
    # mylist.append(55)
    # mylist.append(66)
    time.sleep(2)


if __name__ == "__main__":
    mydict = multiprocessing.Manager().dict()  # 主进程与子进程共享这个字典
    num_arr = multiprocessing.Array('i', [1, 2, 3])  # 主进程与子进程共享这个List

    print(num_arr)
    print(mydict)

    p = multiprocessing.Process(target=func, args=(mydict, num_arr))
    p.start()
    # p.join()

    q = multiprocessing.Process(target=func2, args=(mydict, num_arr))
    q.start()
    # q.join()

    print("split:")
    print(num_arr)
    print(mydict)

    time.sleep(5)
    print("split:")
    print(num_arr)
    print(mydict)

