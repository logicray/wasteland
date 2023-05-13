/**
 * @file stack_sort.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-02
 * 
 * @copyright Copyright (c) 2022
 * sort a stack with help of another stack
 */

#include<iostream>
#include<stack>

using namespace std;

void sort_stack(stack<int> &stk)
{
    stack<int> tmp_stk;
    tmp_stk.push(stk.top());
    stk.pop();

    while (!stk.empty())
    {
        if (stk.top() <= tmp_stk.top())
        {
            tmp_stk.push(stk.top());
            stk.pop();
        }else{
            int tmp = stk.top();
            stk.pop();
            int tmp_size = tmp_stk.size();
            for (int i = 0; i < tmp_size; i++)
            {
                stk.push(tmp_stk.top());
                tmp_stk.pop();
            }
            tmp_stk.push(tmp);
        }
    }    

    int tmp_size = tmp_stk.size();
    for (int i = 0; i < tmp_size; i++)
    {
        stk.push(tmp_stk.top());
        tmp_stk.pop();
    }
}

int main(int argc, char* argv[])
{
    stack<int> my_stack;
    my_stack.push(12);
    my_stack.push(10);
    my_stack.push(16);
    my_stack.push(12);
    my_stack.push(7);
    my_stack.push(19);
    sort_stack(my_stack);
    int size = my_stack.size();
    cout << size << endl;

    for (int i = 0; i < size; i++)
    {
        cout << my_stack.top() << endl;
        my_stack.pop();
    }
}