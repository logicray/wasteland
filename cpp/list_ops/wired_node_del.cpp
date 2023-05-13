/**
 * @file wired_node_del.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-10
 * 
 * @copyright Copyright (c) 2022
 * delete node in linked list by
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


void delete_node(Node* node);
void print_list(Node*);

int main(int argc, char* argv[]){
    Node *n1 = new Node(6);
    Node *n2 = new Node(6);
    Node *n3 = new Node(3);
    Node *n3_d = new Node(3);
    Node *n4 = new Node(1);
    Node *n5 = new Node(16);
    Node *n6 = new Node(16);
    Node *n7 = new Node(16);

    n1->next = n2;
    n2->next = n3;
    n3->next = n3_d;
    n3_d->next = n4;
    n4->next = n5;
    n5->next = n6;
    n6->next = n7;
    n7->next = nullptr;
    print_list(n1);
    delete_node(n4);
    print_list(n1);
}

void delete_node(Node* node){
    if(node == nullptr || node->next == nullptr){
        cout << "can not delete," << endl;
        return;
    }
    node->value = node->next->value;
    node->next = node->next->next;
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