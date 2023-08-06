/**
 * @file del_middle_node.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-27
 * 
 * @copyright Copyright (c) 2022
 * remove moddile node in linked list, or remove node in a/b ratio
 */

#include<iostream>


using namespace std;

class Node;
Node* remove_middle_node(Node*);
Node* remove_node_by_ratio(Node*, int, int);
void print_lint(Node*);


class Node{
    public:
    int value;
    Node* next;
    Node(int data){
        value = data;
    }
};

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
    cout << " |" << endl;
}

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
    Node* res = remove_middle_node(head);
    print_list(res);
    return 0;
}


Node* remove_middle_node(Node* head)
{
    //先求出n， 需要删除的 节点index为 (n-1)/2
    int n = 1;
    Node* curr = head;
    while (curr->next != nullptr)
    {
        curr = curr->next;
        n++;
    }
    cout << "n is:" << n << endl;
    int k = (n-1)/2 ;
    cout << "k is:" << k << endl;;

    
    curr = head;
    while (--k > 0)
    {
        curr = curr->next;
    }
    if(k < 0){
        head = curr->next;
    }else{
        curr->next = curr->next->next;
    }
    return head;   
}


Node* remove_node_by_ratio(Node* head, int a, int b)
{
    //先求出n， 需要删除的 节点index为 (n-1)/2
    int n = 1;
    Node* curr = head;
    while (curr->next != nullptr)
    {
        curr = curr->next;
        n++;
    }
    cout << "n is:" << n << endl;
    int k = (n-1)/2 ;
    cout << "k is:" << k << endl;;

    
    curr = head;
    while (--k > 0)
    {
        curr = curr->next;
    }
    if(k < 0){
        head = curr->next;
    }else{
        curr->next = curr->next->next;
    }
    return head;
    
}