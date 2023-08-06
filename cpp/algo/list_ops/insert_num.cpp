/**
 * @file insert_num.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include<iostream>

using namespace std;

class Node{
    public:
    int value;
    Node* next;
    Node(int data){
        value = data;
    }
};

void print_list(Node*);
Node* insert_node(Node*, int);

int main(int argc, char* argv[]){
    Node *n1 = new Node(3);
    Node *n2 = new Node(4);
    Node *n3 = new Node(7);
    Node *n3_d = new Node(9);
    Node *n4 = new Node(12);
    Node *n5 = new Node(14);
    Node *n6 = new Node(16);
    Node *n7 = new Node(19);

    n1->next = n2;
    n2->next = n3;
    n3->next = n3_d;
    n3_d->next = n4;
    n4->next = n5;
    n5->next = n6;
    n6->next = n7;
    n7->next = n1;
    print_list(n1);
    Node* res = insert_node(n1, 2);
    print_list(res);
}

Node* insert_node(Node* head, int num){
    if(head == nullptr || head->next == nullptr){
        return head;
    }

    Node* new_node = new Node(num);
    Node* pre = head;
    Node* curr = head->next;

    bool cond = pre->value <= num && num <= curr->value;
    while(!cond && curr != head){
        pre = pre->next;
        curr = curr->next;

        cond = pre->value <= num && num <= curr->value;
    }
    
    cout << "pre->value :" << pre->value << endl;

    pre->next = new_node;
    new_node->next = curr;
    if(cond || num > head->value){
        return head;
    }else {
        return new_node;
    }
}


void print_list(Node* head){
    if(head == nullptr){
        cout << "list is empty" << endl;
        return;
    }
    int max_cnt = 0;
    cout << head->value << "->" ;
    while (head->next != nullptr)
    {
        cout << head->next->value << "->" ;
        head = head->next;
        max_cnt ++ ;
        if (max_cnt == 10){
            break;
        }
    }

    cout << "null |" << endl;
}