/**
 * @file hanoi_base.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-06
 * 
 * @copyright Copyright (c) 2022
 * base honoi implementation
 * A B C
 */

#include <iostream>
#include <stack>
#include <assert.h>

using namespace std;

int move_cnt = 0;
// move disk from src to dest
void move(int disk_num, char src, char dest)
{
    cout << "step: " << ++move_cnt << " move No." << disk_num << " disk from " << src << " to " << dest << endl;
}

void hanoi_recursive(int disk_num, char a, char b, char c)
{
    if (disk_num == 1)
    {
        move(disk_num, a, c);
        return;
    }

    // move top n-1 disks from A to B
    hanoi_recursive(disk_num - 1, a, c, b);
    // move 1 disk from a to c,
    move(disk_num, a, c);
    // move n-1 disks in b to c
    hanoi_recursive(disk_num - 1, b, a, c);
}

int step_move()
{
    return 0;
}

class MoveStep
{
private:
    int n;
    char a, b, c;

public:
    MoveStep(int n, char a, char b, char c)
    {
        this->n = n;
        this->a = a;
        this->b = b;
        this->c = c;
    }
    int get_n()
    {
        return n;
    }

    char get_a()
    {
        return a;
    }

    char get_b()
    {
        return b;
    }

    char get_c()
    {
        return c;
    }
};

void hanoi_stack(int disk_num, char src, char inter, char dest)
{
    move_cnt = 0;
    stack<MoveStep> my_stack;
    my_stack.push(MoveStep(disk_num, src, inter, dest));
    while (!my_stack.empty())
    {
        MoveStep step_status = my_stack.top();
        my_stack.pop();

        int n = step_status.get_n();
        if (n == 1)
        {
            move(n, step_status.get_a(), step_status.get_c()); //n is always 1, need to fix
        }
        else
        {
            char a = step_status.get_a();
            char b = step_status.get_b();
            char c = step_status.get_c();

            my_stack.push(MoveStep(n - 1, b, a, c));
            my_stack.push(MoveStep(1, a, b, c));
            my_stack.push(MoveStep(n - 1, a, c, b));
        }
    }
}

int main(int argc, char *argv[])
{
    int disk_num = 2;

    cout << "it's a hanoi tower problem, the task is move all disks in tower A to tower C with the help of tower B, disks are in order in tower a, must be kept in order at any step.\n"
         << endl;

    char src = 'A';
    char inter = 'B';
    char dest = 'C';
    // hanoi_recursive(disk_num, src, inter, dest);

    hanoi_stack(disk_num, src, inter, dest);
    return 0;
}