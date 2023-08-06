/**
 * @file josephus.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-30
 * 
 * @copyright Copyright (c) 2022
 * josephus problem in circled list
 */

#include<iostream>

using namespace std;

class Node;
Node* solve1(Node* ,int);
Node* solve2(Node* ,int);
int find_pos(int, int);
void print_list(Node*);

class Node{
    public:
    int value;
    Node* next;
    Node(int data){
        value = data;
    }
};

int main(int argc, char *argv[]){
    Node* head = new Node(1);
    Node* node2 = new Node(2);
    Node* node3 = new Node(3);
    Node* node4 = new Node(4);
    Node* node5 = new Node(5);

    head->next = node2;
    node2->next = node3;
    node3->next = node4;
    node4->next = node5;
    node5->next = head;

    print_list(head);
    // Node* res1 = solve1(head, 3);
    // print_list(head);
    // cout << res1->value << endl;

    Node* res2 = solve2(head, 3);
     cout << res2->value << endl;
}

/**
 * @brief 
 * 
 * @param head 
 * @param m 
 * @return Node* 
 * simple solve, O(n*m)
 */
Node* solve1(Node* head,int m){
    if (head == nullptr || head->next == head || m < 1){
        return head;
    }
    Node* pre_head = head;
    while (pre_head->next !=head)
    {
        pre_head = pre_head->next;
    }
    // cout << pre_head->value<<endl;

    int cnt = 1;
    while (pre_head != pre_head->next){
        if (cnt % m == 0){
            cnt = 1;
            //delete current
            pre_head->next = pre_head->next->next;
            // cout << pre_head->value << "," <<  pre_head->next->value<<endl;
        }
        cnt ++ ;
        pre_head = pre_head->next;
        // head = head->next;
        cout << "curr:" << pre_head->value << endl;
        // cout << cnt<<endl;
    }
    return pre_head;
}

/**
 * @brief 经过推导可得
 * 老编号 = （新编号+m-1）% n + 1
 * 
 * @param head 
 * @param m 
 * @return Node* 
 */
Node* solve2(Node* head,int m){
    if (head == nullptr || head->next == nullptr || m < 1){
        return head;
    }
    
    int n = 1;
    Node* curr = head;
    while (curr->next != head){
        cout << "curr next:" << curr->next->value << endl;
        curr = curr->next;
        n++;
    }
    cout << "n is:" << n << endl;
    int final_pos = find_pos(n, m);

    while (-- final_pos != 0)
    {
        head = head->next;
    }
    head->next = head;
    return head;
}

int find_pos(int i, int m){
    if (i == 1){
        return 1;
    }
    return (find_pos(i-1, m) + m -1)%i + 1;
}


void print_list(Node* head)
{
    if(head == nullptr){
        cout << "list is empty" << endl;
    }
    Node* curr = head;
    cout << curr->value << "->" ;
    while (curr->next != head)
    {
        cout << curr->next->value << "->" ;
        curr = curr->next;
    }

    cout <<  head->value << " |" << endl;
}
