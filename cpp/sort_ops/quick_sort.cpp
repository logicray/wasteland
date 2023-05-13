//
// Created by page on 2018/9/8.
//
#include <iostream>

using namespace std;

void swap(int v[], int i, int j)
{
    int temp;
    temp = v[i];
    v[i] = v[j];
    v[j] = temp;
}


/*quick sort */
void my_quick_sort(int v[], int n)
{
    int i, last;
    if (n <= 1)
        return;
    //保证第一个元素是随机的
    swap(v, 0, rand() % n);
    last = 0;
    //将小于v[0]的元素全部移动到前半部分
    for (i = 0; i < n; i++)
        if (v[i] < v[0])
            swap(v, ++last, i);
    swap(v, 0, last);
    my_quick_sort(v, last);
    my_quick_sort(v + last + 1, n - last - 1);
}

int main(int argc, char* argv[]){
    int tmp_arr[] = {1,4,3,6,2};
      my_quick_sort(tmp_arr, 5);
    for (size_t i = 0; i < 5; i++)
    {
        cout <<  tmp_arr[i] << "\t";
    }
    
    cout << "\n" << *(tmp_arr + 1) << "\n";
  
    return 0;
}
