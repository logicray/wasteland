/**
 * @file list_partition.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-15
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


Node* re_partition(Node* head);
void print_list(Node*);

int main(int argc, char* argv[]){
    Node *n1 = new Node(1);
    Node *n2 = new Node(2);
    Node *n3 = new Node(3);
    Node *n3_d = new Node(4);
    Node *n4 = new Node(5);
    Node *n5 = new Node(6);
    Node *n6 = new Node(7);
    Node *n7 = new Node(8);

    n1->next = n2;
    n2->next = n3;
    n3->next = n3_d;
    n3_d->next = n4;
    n4->next = n5;
    n5->next = n6;
    n6->next = nullptr;
    // n7->next = nullptr;
    print_list(n1);
    Node* res = re_partition(n1);
    print_list(res);
}

Node* re_partition(Node* head){
    //
    if(head == nullptr || head->next == nullptr){
        return head;
    }

    //split to left and right
    int n = 0;
    Node* curr = head;
    while(curr != nullptr){
        curr = curr->next;
        n++;
    }

    int mid = n/2;
    Node* left_start = head;
    curr = head;
    while (--mid > 0){
        curr = curr->next;
    }
    
    Node* pre_right_start = curr;
    mid = n/2;
    curr = head;
    cout << pre_right_start->value << endl;
    // curr->next = right_start;
    while(mid > 0){
        Node* right_curr =  pre_right_start->next;
        pre_right_start->next = pre_right_start->next->next;

        Node* next = curr->next;
        curr->next = right_curr;
        right_curr->next = next;
        
        print_list(head);
        curr = curr->next->next;
        // pre_right_start = pre_right_start->next;

        mid-- ;
    }

return head;
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