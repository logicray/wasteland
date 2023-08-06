/**
 * @file near_less.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include<iostream>
#include<stack>
#include<vector>

using namespace std;

int (*get_near_less_no_repeat(int[], int))[2];
int (*get_near_less(int[], int))[2];

int main(int argc, char* argv[])
{
    int input[] = {3,4,1,5,6,2,7};
    int size = 7;

    int (*res)[2];
    res = get_near_less_no_repeat(input, size);
    for (int i = 0; i < size; i++)
    {
        for (int  j = 0; j < 2; j++)
        {
            cout << res[i][j] << " ";
        }
        cout << endl;
        /* code */
    }

    cout << "====split line====" << endl;
     int input2[] = {3,1,3,4,3,5,3,2,2};
     int size2 = 9;

    int (*res2)[2];
    res2 = get_near_less(input2, size2);
    for (int i = 0; i < size2; i++)
    {
        for (int  j = 0; j < 2; j++)
        {
            cout << res2[i][j] << " ";
        }
        cout << endl;
    }
}


int (*get_near_less(int arr[], int size))[2]
{
    int (*res)[2] = new int[size][2];
    stack<vector<int> > stk;

    for(int i = 0;i < size; i++){
        while (!stk.empty() && arr[stk.top()[0]] > arr[i]){
            vector<int> pop_index_vec = stk.top();
            stk.pop();

            int left_less = stk.empty() ? -1:stk.top()[stk.top().size() - 1];

            for (int j = 0; j < pop_index_vec.size(); j++)
            {
                res[pop_index_vec[j]][0] = left_less;
                res[pop_index_vec[j]][1] = i;
            }
        }

        if (!stk.empty() && arr[stk.top()[0]] == arr[i])
        {
            stk.top().push_back(i);
        }else{
            vector<int> tmp_vec;
            tmp_vec.push_back(i);
            stk.push(tmp_vec);
        }
    }

    while(!stk.empty()){
        vector<int> pop_index_vec = stk.top();
        stk.pop();
        int left_less = stk.empty() ? -1:stk.top()[stk.top().size() - 1];
        for (int i = 0; i < pop_index_vec.size(); i++)
        {
            res[pop_index_vec[i]][0] = left_less;
            res[pop_index_vec[i]][1] = -1;
        }
    }
    return res;
}


int (*get_near_less_no_repeat(int arr[], int size))[2]
{
    int (*res)[2] = new int[size][2];
    stack<int> stk;

    for(int i=0;i< size; i++){
        while (!stk.empty() && arr[stk.top()] > arr[i]){
            int pop_index = stk.top();
            stk.pop();
            int left_less = stk.empty() ? -1:stk.top();
            res[pop_index][0] = left_less;
            res[pop_index][1] = i;
        }
        stk.push(i);
    }

    while(!stk.empty()){
        int pop_index = stk.top();
        stk.pop();
        int left_less = stk.empty() ? -1:stk.top();
        res[pop_index][0] = left_less;
        res[pop_index][1] = -1;
    }
    return res;
}