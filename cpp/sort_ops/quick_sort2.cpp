/* 
  simple quick
*/
#include<iostream>

using namespace std;

void swap(int arr[], int i, int j)
{
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}

void qucik_sort(int arr[], int low, int high)
{
    if (low >= high)
        return;

    int i = low;
    int j = high + 1;
    int pivot = arr[i];

    while (1)
    {
        while (arr[++i] < pivot)
        {
            // ++i;
            if (i==high)break;
            cout << "1_";
        }

        while (arr[--j] > pivot)
        {
            // --j;
             if (j==low)break;
        }

        if (i >= j)
            break;

        swap(arr, i, j);
    }
    //
    arr[low] = arr[j];
    arr[j] = pivot;
    qucik_sort(arr, low, j);
    qucik_sort(arr, j + 1, high);
}

int main(int argc, char* argv[])
{
    int arr[] = {3,6,1,9,4,2,8};
    // swap(arr, 0,2);
    qucik_sort(arr, 0, 6);
    cout << endl;
    for(int a:arr){
        cout << a << " ";
    }
}