/**
 * @file palindrome.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-04-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include<iostream>
#include<stack>

using namespace std;

class Node{
    public:
    int value;
    Node* next;
    Node(int data){
        value = data;
    }
};
bool is_palingdrome1(Node*);
bool is_palingdrome2(Node*);
bool is_palingdrome3(Node*);
void print_list(Node*);

int main(int argc, char* argv[]){
    Node* head = new Node(1);
    Node* node2 = new Node(2);
    Node* node3 = new Node(3);
    Node* node3_d = new Node(3);
    Node* node4 = new Node(2);
    Node* node5 = new Node(1);

    head->next = node2;
    node2->next = node3;
    node3->next = node3_d;
    node3_d->next = node4;
    node4->next = node5;
    print_list(head);
    bool res1 = is_palingdrome3(head);
    
    cout << "is palingdrome:" << res1 << endl;
    cout << "true is:" << true << endl;
    print_list(head);
    return 0;
}

bool is_palingdrome1(Node* head){
    if (head == nullptr || head->next == nullptr){
        return true;
    }

    int n = 0;
    Node* curr = head;
    while (curr != nullptr){
        n++;
        curr = curr->next;
    }

    bool is_even = n%2==0;
    int k = is_even? n/2: (n-1)/2;
    stack<int> stk;
    cout << k<< endl;
    while (k-- != 0){
        cout << "1-- " << head->value << endl;
        stk.push(head->value);
        head = head->next;
    }

    if (!is_even){
        head = head->next;
    }

    while (head != nullptr){
        if (head->value == stk.top()){
           stk.pop();
           head = head->next;
        }else{
            return false;
        }
    }
    return true;
}


bool is_palingdrome2(Node* head){
    if (head == nullptr || head->next == nullptr){
        return true;
    }

    stack<int> stk;
    Node* curr = head;
    while (curr != nullptr){
        stk.push(curr->value);
        curr = curr->next;
    }
    curr = head;
    while (curr != nullptr){
        if(curr->value == stk.top()){
            curr = curr->next;
            stk.pop();
        }else{
            return false;
        }
    }
    return true;
}


bool is_palingdrome3(Node* head){
    if (head == nullptr || head->next == nullptr){
        return true;
    }

    int n = 0;
    Node* curr = head;
    while (curr != nullptr){
        n++;
        curr = curr->next;
    }

    bool is_even = n%2==0;
    int k = is_even? n/2: (n-1)/2;

    curr = head;
    int tmp = k;
    while (--tmp != 0){
        curr = curr->next;
    }
    
    if (!is_even){
        curr = curr->next;
    }
    cout << "curr->value" <<curr->next->value << endl;
    //反转右半部分, 如果总节点个数为奇数，则中间节点留在左边
    Node* left_end = curr;

    curr = curr->next;
    Node* right_start = curr;

    Node* next = nullptr;
    Node* pre = nullptr;
    while (curr!= nullptr){
        next = curr->next;
        curr->next = pre;

        pre = curr;
        curr = next;
    }
    left_end->next = nullptr;
    Node* new_start = pre;
    print_list(new_start);
    // cout << pre->value << " -- " << pre->next->value << " -- " << pre->next->next->value << endl;
    //
    
    curr = head;
    bool res = true;
    while (k != 0){
        if (curr !=nullptr && pre !=nullptr && curr->value == pre->value){
            curr = curr->next;
            pre = pre->next;
            k--;
        }else{
            res = false;
            break;
        }
    }
    cout << head << endl;
    print_list(head);
    print_list(new_start);
    //恢复右半部分
    pre = nullptr;
    while (new_start != nullptr){
        next = new_start->next;
        new_start->next = pre;

        pre = new_start;
        new_start = next;
    }
    left_end->next = pre;
    print_list(head);
    return res;
}


void print_list(Node* head)
{
    if(head == nullptr){
        cout << "list is empty" << endl;
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