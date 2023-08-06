/**
 * @file reverse_part_list.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-03-29
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

Node* reverse_part(Node* head, int start, int end);
void print_list(Node* head);


int main(int argc, char* argv[])
{
    Node* head = new Node(1);
    Node* node2 = new Node(2);
    Node* node3 = new Node(3);
    Node* node4 = new Node(4);
    Node* node5 = new Node(5);

    head->next = node2;
    node2->next = node3;
    node3->next = node4;
    node4->next = node5;

    print_list(head);
    Node* res = reverse_part(head, 1, 5);
    print_list(res);
}


void print_list(Node* head)
{
    if(head == nullptr){
        cout << "list is empty" << endl;
    }
    cout << head->value << "->" ;
    while (head->next != nullptr)
    {
        cout << head->next->value << "->" ;
        head = head->next;
    }

    cout << "null |" << endl;
}


Node* reverse_part(Node* head, int start, int end){
     int k=0;
     Node* curr = head;
     Node* pre_start = nullptr;
     Node* post_end = nullptr;
    //  Node* start_node = nullptr;
  
     while (curr != nullptr){
        k++;
        if (k == start-1){
            pre_start = curr;
        }

        if (k == end+1){
            post_end = curr;
        }
        curr = curr->next;
    }

    if(start > end || start < 0 || end > k){
         return head;
    }

    Node* start_node;
     if (pre_start == nullptr){
         start_node = head;
     }else{
         start_node = pre_start->next;
     }
     
    //  cout << start_node->value << "," << post_end->value << endl;


    Node* next = nullptr;
    Node* pre = nullptr;
    Node* start_node2 = start_node;
    while (start_node2 != post_end){
        next = start_node2->next;
        start_node2->next = pre;

        pre = start_node2;
        start_node2 = next;
    }
    start_node->next = post_end;
    
    // print_list(pre);

     if (pre_start == nullptr)
     {
         return pre;
     }else{
         pre_start->next = pre;
        return pre_start;
     }
}

