/**
 * @file remove_repeat.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-04-30
 * 
 * @copyright Copyright (c) 2022
 * remove repeat num in list
 */

#include<iostream>
#include<unordered_set>

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
Node* remove1(Node* head);
Node* remove2(Node* head);

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
    Node* res = remove2(n1);
    print_list(res);

    return 0;
}

Node* remove1(Node* head){
    if(head == nullptr){
        return head;
    }

    unordered_set<int> set;
    
    Node* pre = head;
    set.insert(pre->value);
    Node *curr = pre->next;
    while(curr != nullptr){
        //contain 
        if(set.find(curr->value) != set.end()){
            pre->next = curr->next;
            // curr->next = curr->next->next;
        }else{
            set.insert(curr->value);
            pre = curr;
        }
        curr = curr->next;
    }
    return head;
}

Node* remove2(Node* head){
    if(head == nullptr){
        return head;
    }

    //使用类似选择排序，
    Node* curr = head;
    while(curr != nullptr){
        int curr_val = curr->value;
        Node* sub_pre = curr;
        Node* sub_curr = curr->next;
        while(sub_curr != nullptr){
            if(sub_curr->value == curr_val){
                sub_pre->next = sub_curr->next;
            }else{
                sub_pre = sub_curr;
            }
            sub_curr = sub_curr->next;
        }
        curr = curr->next;
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