/**
 * @file stack_get_min.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-02-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include<iostream>
#include<stack>

using namespace std;

class StackWithMin
{
    private:
    stack<int> data;
    stack<int> min;

    public:
    void push(int ele)
    {
        data.push(ele);
        if (min.empty())
        {
           min.push(ele);
        }
        
        if (!min.empty() && ele <= min.top())
        {
            min.push(ele);
        }
    }

    void pop()
    {
        
        int top = data.top();
        if (top == min.top())
        {
            min.pop();
        }
        data.pop();
    }

    int get_min()
    {
        return min.top();
    }

};

int main(int argc, char* argv[])
{
    StackWithMin s;
    s.push(5);
    s.push(10);
    s.push(3);
    cout << s.get_min() << endl;
    s.pop();
    cout << s.get_min() << endl;
    s.push(2);
     cout << s.get_min() << endl;

         s.pop();
    cout << s.get_min() << endl;
    cout << "hello";
    return 0;
}

