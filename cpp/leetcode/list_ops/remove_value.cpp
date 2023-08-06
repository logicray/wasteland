/**
 * @file remove_value.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-01
 * 
 * @copyright Copyright (c) 2022
 * remove list node which value equal to input value
 */

#include<iostream>
#include<stack>

using namespace std;

class Node{
    public:
    int value;
    Node* next;

    Node(int value) {
        this->value = value;
    }
};
Node* remove_num(Node* head, int num);
Node* remove_num2(Node* head, int num);
void print_list(Node *head);


int main(int argc, char* argv[]){
    Node *n1 = new Node(4);
    Node *n2 = new Node(9);
    Node *n3 = new Node(3);
    Node *n3_d = new Node(4);
    Node *n4 = new Node(1);
    Node *n5 = new Node(2);
    Node *n6 = new Node(16);
    Node *n7 = new Node(4);

    n1->next = n2;
    n2->next = n3;
    n3->next = n3_d;
    n3_d->next = n4;
    n4->next = n5;
    n5->next = n6;
    n6->next = n7;
    n7->next = nullptr;

    print_list(n1);
    Node* res = remove_num2(n1, 4);
    print_list(res);
    return 0;
}

Node* remove_num2(Node* head, int num){
    //use stack
    stack<Node*> stk;

    while(head != nullptr ){
        if(head->value != num){
            stk.push(head);
        }
        head = head->next;
    }

    cout << "push end:"<< stk.size() <<endl;
    // Node* tail = stk.top();
    // stk.pop();
    // tail->next = nullptr;
    
    while(!stk.empty()){
        Node* tmp = stk.top();
        tmp->next = head;
        stk.pop();
        // tmp->next = tail;
        head = tmp;
    }
    return head;
}


Node* remove_num(Node* head, int num){
    while(head != nullptr && head->value == num){
        head = head->next;
    }


    Node* curr = head;
    while(curr!=nullptr && curr->next != nullptr){
        if(curr->next->value == num){
            curr->next = curr->next->next;
        }
        curr = curr->next;
    }

    return head;
}


void print_list(Node *head){
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