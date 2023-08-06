/**
 * @file select_sort.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-05
 * 
 * @copyright Copyright (c) 2022
 * select sort on linked list
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
Node* sort_list(Node* head);

int main(int argc, char* argv[]){
    Node *n1 = new Node(6);
    Node *n2 = new Node(5);
    Node *n3 = new Node(1);
    Node *n3_d = new Node(9);
    Node *n4 = new Node(3);
    Node *n5 = new Node(13);
    Node *n6 = new Node(19);
    Node *n7 = new Node(7);

    n1->next = n2;
    n2->next = n3;
    n3->next = n3_d;
    n3_d->next = n4;
    n4->next = n5;
    n5->next = n6;
    n6->next = n7;
    n7->next = nullptr;

    print_list(n1);
    Node* res = sort_list(n1);
    print_list(res);
}

Node* sort_list(Node* head){
    if(head == nullptr || head->next == nullptr){
        return head;
    }

    Node* curr = head;
    Node* pre_curr = nullptr;
    while(curr != nullptr){
        Node* tmp = curr;
        int max_value = tmp->value; //
        Node* pre_max_node = pre_curr;
        while(tmp->next != nullptr){
            if(tmp->next->value > max_value){
                pre_max_node = tmp;
                max_value = tmp->next->value;
            }
            tmp = tmp->next;
        }
        if(pre_max_node == pre_curr){
            pre_curr = curr;
            curr = curr->next;
            continue;
        }
        //cut and reconnect
        // pre_curr = curr;
        if(curr == head){
            Node* max_node = pre_max_node->next;
            pre_max_node->next = max_node->next;
            max_node->next = head;
            head = max_node;
            pre_curr = head;
            // curr = head;
        }else{
            Node* max_node = pre_max_node->next;
            pre_max_node->next = max_node->next;

            cout << "max_node->value: "<<max_node->value << endl;
            cout << "pre_curr->value: "<<pre_curr->value << endl;
            print_list(head);
            
            pre_curr->next = max_node;
            max_node->next = curr;
            // curr = max_node;
            pre_curr = max_node;
        }
        
        // print_list(head);
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