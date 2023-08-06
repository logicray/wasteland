/**
 * @file list_as_num.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-04-09
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include<iostream>
#include<stack>
#include<cmath>

using namespace std;

class Node{
    public:
    int value;
    Node* next;
    Node(int data){
        value = data;
    }
};
Node* list_add(Node*, Node*);
Node *list_add2(Node*, Node*);
void print_list(Node*);

int main(int argc, char * argv[]){
    Node* head = new Node(6);
    Node* node2 = new Node(9);
    Node* node3 = new Node(3);
    Node* node3_d = new Node(3);
    Node* node4 = new Node(2);
    Node* node5 = new Node(2);

    head->next = node2;
    node2->next = node3;
    node3->next = node3_d;
    node3_d->next = node4;
    node4->next = node5;
    // print_list(head);

    Node* head2 = new Node(3);
    Node* node22 = new Node(2);
    Node* node23 = new Node(3);
    Node* node23_d = new Node(3);
    Node* node24 = new Node(2);
    // Node* node25 = new Node(2);

    head2->next = node22;
    node22->next = node23;
    node23->next = node23_d;
    node23_d->next = node24;
    // node24->next = node25;
    print_list(head);
    print_list(head2);
    Node *res1 = list_add2(head, head2);
    print_list(res1);
    return 0;
}

Node* list_add(Node* h1, Node* h2){
    stack<int> stk1, stk2;

    while(h1 != nullptr){
        stk1.push(h1->value);
        h1 = h1->next;
    }

    while(h2 != nullptr){
        stk2.push(h2->value);
        h2 = h2->next;
    }

    int ca = 0, d = 0;
    Node* sum_list = nullptr;
    Node* curr = sum_list;
    while(!stk1.empty() || !stk2.empty()){
        int sum = 0;
        if (!stk1.empty()){
            sum +=  stk1.top();
            stk1.pop();
        }

        if (!stk2.empty()){
            sum +=  stk2.top();
            stk2.pop();
        }

        sum += ca;
        if( sum >= 10){
            ca = 1;
            sum = sum %10;
        }else{
            ca = 0;
        }
        // cout << "sum:" << sum << endl;
        Node* tmp = new Node(sum);
        if (sum_list == nullptr){
            cout << "sum list null:" << sum<<endl;
            sum_list = tmp;
            curr = sum_list;
        }else{
            cout << "sum list non null:" << sum<<endl;
            curr->next = tmp;
            curr = curr->next;
        }
    }
    if(ca == 1){
        curr->next = new Node(ca);
    }
    print_list(sum_list);
    //reverse

    Node* next = nullptr;
    Node* pre= nullptr;
    while( sum_list != nullptr){
        next = sum_list->next;
        sum_list->next = pre;

        pre = sum_list;
        sum_list = next;
    }
    return pre;
}


Node* list_add2(Node* h1, Node* h2){
    //reverse list
    Node *next = nullptr, * pre = nullptr;
    while(h1 != nullptr){
        next = h1->next;
        h1->next = pre;

        pre = h1;
        h1 = next;
    }
    h1 = pre;
    print_list(h1);

    next = nullptr;
    pre = nullptr;
    while( h2 != nullptr){
        next = h2->next;
        h2->next = pre;

        pre = h2;
        h2 = next;
    }
    h2 = pre;
    print_list(h2);

    int ca = 0;
    Node *sum_list = nullptr;
    Node * curr = nullptr;
    int d = 0;
    while(h1 != nullptr || h2!=nullptr){
        int sum = 0;
        if (h1 !=nullptr){
            sum+= h1->value;
            h1 = h1->next;
        }

        if(h2 != nullptr){
            sum += h2->value;
            h2 = h2->next;
        }
        sum += ca;

        if(sum >= 10){
            sum = sum %10;
            ca =1;
        }else{
            ca =0;
        }
        Node *tmp = new Node(sum);
        if (sum_list == nullptr){
            sum_list = tmp;
            curr = sum_list;
        }else{
            curr->next = tmp;
            curr = curr->next;
        }
    }
    if (ca == 1){
       Node *tmp = new Node(ca);
       curr->next = tmp;
    }
    
    next = nullptr;
    pre= nullptr;
    while( sum_list != nullptr){
        next = sum_list->next;
        sum_list->next = pre;

        pre = sum_list;
        sum_list = next;
    }
    return pre;
}



void print_list(Node* head)
{
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