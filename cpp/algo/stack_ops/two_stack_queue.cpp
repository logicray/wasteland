/**
 * @file two_stack_queue.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-02-27
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <stack>

using namespace std;

class TwoStackQueue
{
private:
    stack<int> stack_push;
    stack<int> stack_pop;
    void push2pop()
    {
        if (stack_pop.empty())
        {
            /* code */

            while (!stack_push.empty())
            {
                int top = stack_push.top();
                stack_pop.push(top);
                stack_push.pop();
            }
        }
    }

public:
    void add(int ele)
    {
        stack_push.push(ele);
        push2pop();
    }

    void poll()
    {
        if (stack_push.empty() && stack_pop.empty())
        {
            throw "error";
        }
        push2pop();
        stack_pop.pop();
    }

    int peek()
    {
        if (stack_push.empty() && stack_pop.empty())
        {
            throw "error";
        }
        push2pop();
        return stack_pop.top();
    }
};

int main(int argc, char *argv[])
{
    TwoStackQueue queue;
    queue.add(12);
    queue.add(14);
    queue.add(16);
    cout << queue.peek() << endl;
    queue.poll();
    cout << queue.peek() << endl;
    queue.poll();
    cout << queue.peek() << endl;
    return 0;
}
