/**
 * @file copy_list.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-04-07
 * 
 * @copyright Copyright (c) 2022
 * 带有随机数的列表复制
 */

#include<iostream>
#include<unordered_map>

using namespace std;

class Node{
    public:
    int value;
    Node* random;
    Node* next;
    Node(int data){
        this->value = data;
        
    }
};
Node* copy_random_list(Node* head);
Node* copy_random_list2(Node* head);
void print_list(Node*);

int main(int argc, char *argv[]){
    Node* head = new Node(1);
    Node* node2 = new Node(4);
    Node* node3 = new Node(3);
    Node* node3_d = new Node(2);
    Node* node4 = new Node(5);
    Node* node5 = new Node(7);

    head->next = node2;
    head->random = node3;

    node2->next = node3;
    node2->random = node4;

    node3->next = node3_d;
    node3->random = node5;

    node3_d->next = node4;
    node3_d->random = head;

    node4->next = node5;
    node4->random = node3;

    node5->next = nullptr;
    node5->random = node2;

    print_list(head);
    Node* res1 = copy_random_list2(head);
    cout << "before second print" <<endl;
    print_list(res1);
    return 0;
}


Node* copy_random_list(Node *head){
    if (head == nullptr){
        return head;
    }
    
    unordered_map<Node*, Node*> node_map;
    Node *curr = head;
    while (curr != nullptr){
        node_map[curr] =  new Node(curr->value);
        curr = curr->next;
    }

    curr = head;
    while (curr != nullptr){
        node_map.at(curr)->next = curr->next == nullptr ? nullptr : node_map.at(curr->next);
        node_map.at(curr)->random = node_map.at(curr->random);
       
        curr = curr->next;
    }
    
    return  node_map.at(head);
}


Node* copy_random_list2(Node *head){
    if(head == nullptr){
        return head;
    }

    Node *curr = head;
    // 在旧节点中间插入新节点
    while (curr != nullptr){
       Node * old_next = curr->next;
       Node *new_next = new Node(curr->value);
       curr->next = new_next;
       new_next->next = old_next;

       curr = old_next;
    }
    //更新random
    curr = head;
    while(curr != nullptr){
        curr->next->random = curr->random->next;

        curr = curr->next->next;
    }
    print_list(head);
    //分离出新节点
    Node* new_head = head->next;
   
    curr = head;
    Node* new_curr = new_head;

    while(curr->next->next != nullptr){
        cout << curr->value << endl;
        Node* next_curr = curr->next->next;
        Node* next_new_curr = new_curr->next->next;

        curr->next = curr->next->next;
        new_curr->next = new_curr->next->next;

        curr = next_curr;
        new_curr = next_new_curr;
    }
    curr->next = nullptr;
    cout << head->value << endl;
    print_list(head);
    return new_head;
}



void print_list(Node* head)
{
    if(head == nullptr){
        cout << "list is empty" << endl;
        return;
    }
    int max_cnt = 0;
    cout << head->value << "|" << head->random->value << " -> " ;
    while (head->next != nullptr)
    {
        cout << head->next->value << "|" << head->next->random->value << " -> " ;
        head = head->next;
        max_cnt ++ ;
        if (max_cnt == 30){
            break;
        }
        
    }
    cout << "null |" << endl;
}

