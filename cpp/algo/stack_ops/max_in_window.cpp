/**
 * @file max_in_window.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include<iostream>
#include<deque>

using namespace std;

int* get_max_values(int a[], int a_size, int w){
    int res_size = a_size - w +1;
    // cout << "res size:" << res_size << endl;
    int *res = new int[res_size];
    // int res[res_size]
    for (int i = 0; i < res_size; i++)
    {
        res[i] = a[i];
        for (int j = 1; j < w; j++)
        {
            if (res[i] < a[i+j])
            {
                res[i] = a[i+j];
            }
        }
        // cout << res[i] << endl;
    }
    
    return res;
}


int* get_max_values2(int a[], int a_size, int w){
    int res_size = a_size - w +1;
    // cout << "res size:" << res_size << endl;
    int *res = new int[res_size];
    // int res[res_size]
    deque<int> my_queue;
    int start = 0;
    for (int i = 0; i < a_size; i++)
    {
        while (!my_queue.empty() &&  a[my_queue.back()] <= a[i] )
        {
            my_queue.pop_back();
        }
        my_queue.push_back(i);

        if (i-w == my_queue.front())
        {
             my_queue.pop_front();
        }
        
        // cout << res[i] << endl;
        if (i >= w -1)
        {
            res[start++] = a[my_queue.front()];
            cout << "front:" <<my_queue.front()<< endl;
            // my_queue.pop_front();
        }
    }
    
    return res;
}



int main(int argc, char* argv[])
{
    int a[] = {4,3,5,4,3,3,6,7};
    int a_size = 8;
    int w = 3;
    int* res = get_max_values(a, a_size, w);
    
    for (int i = 0; i < a_size - w +1; i++)
    {
        cout << *(res+i) << endl;
    }

    delete res;
}