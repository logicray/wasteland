//
// Created by page on 2018/9/8.
//

#include <iostream>

using namespace std;

void swap(int arr[], int i, int j)
{
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}

void bubble_sort(int arr[], int n)
{
    for (int i =0;i<n ;i++){
        for (int j=i; j <n; j++){
            if (arr[i] < arr[j])
            {
                swap(arr, i, j);
            }
        }
    }
}


int main(int argc, char* argv[]){
    int tmp_arr[] = {1, 4, 3, 6, 2, 6, 7, 9};
    int n = sizeof(tmp_arr)/sizeof(tmp_arr[0]);
    bubble_sort(tmp_arr, n);
    for (size_t i = 0; i < n; i++)
    {
        cout << tmp_arr[i] << "\t";
    }
    
    
    cout << "hello," << endl;
    return 0;
}

