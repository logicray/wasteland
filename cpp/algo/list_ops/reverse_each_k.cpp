/**
 * @file reverse_each_k.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-04-28
 *
 * @copyright Copyright (c) 2022
 * 单向链表每k个节点反转
 */

#include <iostream>
#include <stack>

using namespace std;

class Node
{
public:
    int value;
    Node *next;
    Node(int data)
    {
        value = data;
    }
};
Node *reverse1(Node *h, int k);
Node *reverse2(Node *h, int k);
void print_list(Node *h);

int main(int argc, char *argv[])
{
    Node *n1 = new Node(6);
    Node *n2 = new Node(9);
    Node *n3 = new Node(3);
    Node *n3_d = new Node(4);
    Node *n4 = new Node(1);
    Node *n5 = new Node(2);
    Node *n6 = new Node(16);
    Node *n7 = new Node(26);

    n1->next = n2;
    n2->next = n3;
    n3->next = n3_d;
    n3_d->next = n4;
    n4->next = n5;
    n5->next = n6;
    n6->next = n7;
    n7->next = nullptr;

    print_list(n1);
    Node *res = reverse2(n1, 5);
    print_list(res);
}

Node *reverse1(Node *h, int k)
{
    stack<Node *> k_stk;

    Node *curr = h;

    Node *new_head = nullptr;
    Node *new_curr = nullptr;
    Node *sub_head = nullptr;

    while (curr != nullptr)
    {
        sub_head = curr;
        while (k_stk.size() < k && curr != nullptr)
        {
            k_stk.push(curr);
            curr = curr->next;
        }

        cout << "k stk size:" << k_stk.size() << endl;
        if (k_stk.size() < k)
        {
            new_curr->next = sub_head;
            while (new_curr->next != nullptr)
            {
                new_curr = new_curr->next;
            }
            break;
        }

        while (!k_stk.empty())
        {
            if (new_head == nullptr)
            {
                // cout << "1" << endl;
                // cout << k_stk.top()->value << endl;;
                new_curr = k_stk.top();
                k_stk.pop();
                new_head = new_curr;
                // cout << "new head:" << new_head->value << endl;
            }
            else
            {
                // cout << "2" << endl;
                new_curr->next = k_stk.top();
                k_stk.pop();
                // cout << "new head2:" << new_head->value << endl;
                new_curr = new_curr->next;
            }
        }
    }
    //将最后剩下的小于k个节点保持原状

    new_curr->next = nullptr;

    cout << new_curr->value << endl;
    cout << new_head->value << endl;
    return new_head;
}

Node *reverse2(Node *h, int k)
{
    Node *curr = h;

    int cnt = 0;
    Node *new_head = nullptr;

    Node *sub_start = nullptr;
    Node *sub_end = nullptr;

    Node *next = nullptr;
    Node *pre = nullptr;

    while (curr != nullptr){
            if (sub_end == nullptr){
                sub_end = curr;
            }

            next = curr->next;
            curr->next = pre;
            pre = curr;
            curr = next;
            cnt++;

            if (cnt == k){
                if (new_head == nullptr){
                    new_head = pre;
                    sub_start = sub_end;
                     sub_end = nullptr;
                }else{
                    sub_start->next = pre;

                    sub_start = sub_end;
                    sub_end = nullptr;
                }
                pre = nullptr;
                cnt = 0;
            }
    }
    //剩余的不足k个已反转，此刻 pre为头，sub_end 为尾，需要再次反转为原顺序
    Node* pre_pre = nullptr;
    while(pre != nullptr){
        next = pre->next;
        pre->next = pre_pre;

        pre_pre = pre;
        pre = next;
    }
    sub_start->next = pre_pre;
    return new_head;
}

void print_list(Node *head)
{
    if (head == nullptr)
    {
        cout << "list is empty" << endl;
        return;
    }
    int max_cnt = 0;
    cout << head->value << "->";
    while (head->next != nullptr)
    {
        cout << head->next->value << "->";
        head = head->next;
        max_cnt++;
        if (max_cnt == 30)
        {
            break;
        }
    }

    cout << "null |" << endl;
}