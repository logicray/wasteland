/**
 * @file merge_list.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-14
 * 
 * @copyright Copyright (c) 2022
 * merge two sroted list
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


Node* merge_list(Node* node, Node* h2);
void print_list(Node*);

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
    n7->next = nullptr;
    print_list(n1);

    Node* m1 = new Node(5);
    Node* m2 = new Node(8);
    Node* m3 = new Node(13);
    Node* m3_d = new Node(15);
    Node* m4 = new Node(18);
    Node* m5 = new Node(22);

    m1->next = m2;
    m2->next = m3;
    m3->next = m3_d;
    m3_d->next = m4;
    m4->next = m5;
    m5->next = nullptr;

    print_list(m1);
    Node* res = merge_list(n1, m1);
    print_list(res);

    int a = 12;
    cout << "a/2: " << a/2 << endl;
}

Node* merge_list(Node* h1, Node* h2){
    if(h1 == nullptr){
        return h2;
    }

    if(h2 == nullptr){
        return h1;
    }

cout << h1->value << " | " << h2->value << endl;
    Node* new_head = nullptr;
    if(h1->value <= h2->value){
        new_head = h1;
        h1 = h1->next;
    }else{
        new_head = h2;
        h2 = h2->next;
    }
    
    Node* new_curr = new_head;
    while(h1 != nullptr && h2 != nullptr){
        cout << h1->value << " | " << h2->value << endl;
        if(h1->value <= h2->value){
            new_curr->next = h1; 
            new_curr = new_curr->next;
            h1 = h1->next;
        }else{
            new_curr->next = h2;

            new_curr = new_curr->next;
            h2 = h2->next;
        }
    }
    print_list(new_head);
    if(h1 == nullptr){
        new_curr->next = h2;
    }

    if(h2 == nullptr){
        new_curr->next = h1;
    }

    
    return new_head;
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
        if (max_cnt == 30){
            break;
        }
    }

    cout << "null |" << endl;
}