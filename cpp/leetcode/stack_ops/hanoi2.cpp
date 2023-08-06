/**
 * @file hanoi2.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-06
 * 
 * @copyright Copyright (c) 2022
 * special hanoi tower, can not move from a to c,
 */

#include<iostream>
#include<stack>
#include<string>

using namespace std;

int move_cnt = 0;

void move(int disk_num, char src, char dest)
{
    // if (src == 'A' && dest == 'C' )
    // {
    //     move(disk_num, src, 'B');
    //     move(disk_num, 'B', dest);
    // }else if (src == 'C' && dest == 'A')
    // {
    //     move(disk_num, src, 'B');
    //     move(disk_num, 'B', dest);
    // }else{
        cout << "step: "<< ++move_cnt <<" move No." << disk_num <<" disk from " << src << " to "<< dest << endl;
    // }    
}

void hanoi_recursive(int disk_num, char &a, char &b, char &c)
{
    if (disk_num == 1)
    {
        if (a == 'A' && c == 'C' )
        {
            move(disk_num, a, 'B');
            move(disk_num, 'B', c);
        }else if (a == 'C' && c == 'A')
        {
            move(disk_num, c, 'B');
            move(disk_num, 'B', a);
        }else{
            move(disk_num, a, c);
        }
        return;
    }

    //move n-1 from src to middle, 
    hanoi_recursive(disk_num-1, a, b, c);
    //move n-1 from middle to dest
    // hanoi_recursive(disk_num-1, b, a, c);
    //move n from src to middle
    move(disk_num, a, b);
    //move n-1 from dest to middle
    hanoi_recursive(disk_num-1, c, b, a);
    // move n-1 from middle to src
    // hanoi_recursive(disk_num-1, b, c, a);
    //move n from middle to dest,
    move(disk_num, b, c);
    //move n-1 from src to middle
    hanoi_recursive(disk_num-1, a, b, c);
    //move n-1 from middle to dest
    // hanoi_recursive(disk_num-1, b, a, c);
}

enum action_set {no, l2m, m2l, m2r, r2m};

action_set inverse_action(action_set a)
{
    if (a == l2m)
    {
        return m2l;
    }else if (a==m2l)
    {
       return l2m;
    }else if (a==m2r)
    {
        return r2m;
    }else if (a==r2m)
    {
       return m2r;
    }else{return no;}   
}

void print_action(string desc, action_set a)
{
    cout << desc;
    switch (a)
    {
    case no:
        cout << "no" << endl;
        break;
    case l2m:
        cout << "l2m" << endl;
        break;
    case m2l:
        cout << "m2l" << endl;
        break;
    case m2r:
        cout << "m2r" << endl;
        break;
    case r2m:
        cout << "r2m" << endl;
        break;
    default:
        break;
    }
}
int one_step(action_set &last, action_set now, stack<int> &from, stack<int> &to)
{
    if (from.empty())
    {
       return 0;
    }
    action_set inv_last = inverse_action(last);
    if (inv_last != now && (to.empty() || from.top() < to.top() ))
    {
        int tmp = from.top();
        to.push(tmp);
        from.pop();
        last = now;
        cout << "move No." << tmp << " disk from " << endl;
        return 1;
    }else{
        return 0;
    }
}

void hanoi_stack(int disk_num, char &a, char &b, char &c)
{
    stack<int> left;
    stack<int> middle;
    stack<int> right;
    for (int i = disk_num; i > 0; i--)
    {
        left.push(i);
    }
    
    
    int step = 0;
    action_set last_action = no;
    print_action("last_action1:", last_action);
    while (right.size() != disk_num)
    {
        step += one_step(last_action, l2m, left, middle);
        // print_action("last_action1:", last_action);
        // cout << middle.size();
        step += one_step(last_action, m2l, middle, left);
        // print_action("last_action2:", last_action);
        step += one_step(last_action, m2r, middle, right);
        step += one_step(last_action, r2m, right, middle);
    }
    cout << "all steps: " << step << endl;
}



int main(int argc, char* argv[])
{
    int disk_num = 3;

    cout << "it's a hanoi tower problem, the task is move all disks in tower A to tower C with the help of tower B, disks are in order in tower a, must be kept in order at any step. and: can not direct move from src to dest\n" << endl;
    
    char src = 'A';
    char inter = 'B';
    char dest = 'C';
    hanoi_recursive(disk_num, src, inter, dest);
    hanoi_stack(disk_num, src, inter, dest);
}