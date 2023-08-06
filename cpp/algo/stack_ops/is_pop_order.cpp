/**
 * @file is_pop_order.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-02-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include<iostream>
#include<vector>
#include<stack>

using namespace std;

bool is_pop_order(vector<int> push_arr, vector<int> pop_arr)
{
    stack<int> tmp_stack;
    int push_cnt = 0;
    for (int i = 0; i < pop_arr.size(); i++)
    {
        while (tmp_stack.empty() || tmp_stack.top() != pop_arr[i])
        {
            if (push_cnt > push_arr.size())
            {
                return false;
            }
            
            tmp_stack.push(push_arr[push_cnt++]);
        }
        if (!tmp_stack.empty() && tmp_stack.top() == pop_arr[i])
        {
            tmp_stack.pop();
        }
    }
    return true;
}

bool is_pop_order_hash(vector<int> push_arr, vector<int> pop_arr)
{
    
}

int main(int argc, char const *argv[])
{
    vector<int> push_arr = {1,2,3,4,5};
    vector<int> pop_arr = {4, 5, 3,2,1};
    /* code */
    cout << is_pop_order(push_arr, pop_arr) << endl;
    return 0;
}
