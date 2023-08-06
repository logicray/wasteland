/**
 * @author logicpw
 * @brief a simple heap sort
 * @version 0.1
 * @date 2021-12-11
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <iostream>

using namespace std;

/**
 * @brief 分为两步:
 * 1.将数据构造成一个大顶堆或者小顶堆，
 * 2.重复（逐步抽出最大（最小元素）后重新构造堆）
 * 
 * @param arr 输入的数组
 * @param n 数组长度
 */

void swap(int arr[], int i, int j)
{
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}

/**
 * @brief 根据输入的参数，修复大顶堆
 * 
 * @param arr 数组
 * @param i 以第i个元素为根节点，
 * @param len 只考虑arr中 0->len 的元素，
 */
void fix_heap(int arr[], int i, int len)
{
    int tmp = arr[i];
    for (int k = 2 * i + 1; k < len; k = 2 * k + 1) //依次遍历左子节点
    {
        //如果右子节点比左大，则改为处理右
        if (k+1 < len && arr[k] < arr[k+1])
        {
            k++;
        }
        if (arr[k] > tmp)
        {
            arr[i] = arr[k];
            i = k; //
        }else{
            break;
        }
    }
    arr[i] = tmp;
}

void heap_sort(int arr[], int n)
{
    //原始数据输入可以想象为一个二叉树，根节点index=0，左节点为1，右为2，
    //构建大顶堆
    for (int i = n / 2 - 1; i >= 0; i--)
    {
        fix_heap(arr, i, n);
    }
    //此时第一个元素已经是最大值
    //将最大的元素放到数组末尾，然后继续构建大顶堆，不断重复
    for(int j=n-1; j>0; j--){
        swap(arr, 0, j);
        fix_heap(arr, 0, j);
    }
}

int main(int argc, char *argv[])
{
    int tmp_arr[] = {1, 4, 3, 6, 2, 7, 9, 5, 10};
    heap_sort(tmp_arr, 9);
    for (size_t i = 0; i < 9; i++)
    {
        cout << tmp_arr[i] << "\t";
    }
    cout << endl;
    cout << 9 /2 << endl;
    return 0;
}