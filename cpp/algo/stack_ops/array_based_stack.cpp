/**
 * @file array_based_stack.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-02-15
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include<iostream>
#include<stack>

using namespace std;

int main(int argc, char* argv[])
{
    stack<int> my_stack;
    for (int i =0;i<10; i++)
    {
        my_stack.push(i);
    }
    cout << "stack size:" << my_stack.size() << "\n";

    my_stack.pop();
    cout << "top:" << my_stack.top()<< endl;
    cout << "stack size:" << my_stack.size() << "\n";


}