/**
 * @file lis.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-04-18
 * 
 * @copyright Copyright (c) 2022
 * longest incremental sequence
 */

#include<iostream>

using namespace std;

int lis_v1(int[], int);
int lis_v2(int[], int);
int binary_search(int arr[], int start, int end, int value);

int main(int argc, char* argv[]){
    int arr[] = {1, 3, 5, 4, 7, 5, 6, 9, 8};
    int size = sizeof(arr)/sizeof(arr[0]);
    int res = lis_v1(arr, size);
    cout << "res: " << res << endl;

    int res3 = lis_v2(arr, size);
    cout << "res3: " << res << endl;

    int arr2[] = {1,3,5,9};
    int b_res = binary_search(arr2, 0, 3, 6);
    cout << "b res:" << b_res << endl;
    cout << "=================" << endl;
    int arr3[] = {1, 3, 5,7, 9};
    int c_res = binary_search(arr2, 0, 4, 6);
    cout << "c res:" << c_res << endl;
    return 0;
}

int lis_v1(int arr[], int n){
    //dp[i] 表示以i结尾的数组的最长递增子序列
    int *dp = new int[n];
    for(int i=0; i< n; i++){
        dp[i] = 1;
    }


    for(int i=0; i< n; i++){
        for (int j=0; j < i; j++){
            if (arr[j] < arr[i]){
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }

    int res = 0;
    for(int i = 0; i < n; i++){
        res = max(res, dp[i]);
    }
    return res;
}

int lis_v2(int arr[], int n){
    int *tmp = new int[n];
    
    tmp[0] = arr[0];
    int tmp_end_idx = 0;
    for (int i=1; i< n; i++){
        if (arr[i] > tmp[tmp_end_idx]){
          tmp_end_idx++;
          tmp[tmp_end_idx] = arr[i];
        }
        else{
            int index = binary_search(tmp, 0, tmp_end_idx, arr[i]);
            tmp[index] = arr[i];
        }
    }

    return tmp_end_idx+1;
}

int binary_search(int arr[], int start, int end, int value){
    //if find value,return index of value,
    //else return minimum value larger than input value 
   int l = start;
   int r = end;
   int m;
   while( l < r -1 ){
    m = (l + r)/2;
    // cout << "l " <<  r << endl;
    cout << m << endl;

    if (arr[m] == value){
        return m;
    }
    else if (arr[m] < value)
    {
        l = m;
    }else{
        r = m;
    }
   }
   return (l+r)/2 + 1;
}