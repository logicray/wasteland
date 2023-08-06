/**
 * @file stack_reverse.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-01
 * 
 * @copyright Copyright (c) 2022
 *  reverse a stack only use 
 */

#include<iostream>
#include<stack>

using namespace std;

/**
 * @brief get and remove fisrt element of stack
 * 
 * @param stk 
 * @return int 
 */
int pop_bottom(stack<int> &stk)
{
    int result = stk.top();
    stk.pop();
    if (stk.empty())
    {
        return result;
    }

    int last = pop_bottom(stk);
    stk.push(result);
    return last;
}

void reverse_stack(stack<int> &stk)
{
    if (stk.empty())
    {
        return;
    }
    int last = pop_bottom(stk);
    cout << "--:"<< last << endl;
    reverse_stack(stk);
    stk.push(last);
}

int main(int argc, char* argv[])
{
    stack<int> my_stack;
    my_stack.push(1);
    my_stack.push(3);
    my_stack.push(6);
    my_stack.push(9);
    // cout << pop_bottom(my_stack) << "-=-" << endl;
    // cout << pop_bottom(my_stack) << "-=-" << endl;
    reverse_stack(my_stack);
    while (!my_stack.empty())
    {
        cout << my_stack.top() << endl;
        my_stack.pop();
    }
    
}