/**
 * @file del_last_k.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-18
 * 
 * @copyright Copyright (c) 2022
 * 在单链表和双链表中删除倒数第K个节点
 */
#include<iostream>


using namespace std;


class Node;
void print_list(Node*);

class Node{
    private:
    int value;
    Node *next;

    public:
    Node(int data){
        this->value = data;
    }

    void set_next(Node *next){
        this->next = next;
    }

    int get_value(){
        return value;
    }

    Node* get_next(){
        return next;
    }
};


class Node2{
    private:
    

    public:
    int value;
    Node2 *next;
    Node2 *last;

    Node2(int data){
        this->value = data;
    }
};


void print_list(Node* head)
{
    if (head == nullptr)
    {
        cout << "empty list"<< endl;
        return;
    }
    
    cout << head->get_value() << "->" ;
    while (head->get_next() != nullptr)
    {
        Node* next = head->get_next();
        cout << next->get_value() << "->";
        head = next;
    }
    cout << " ||" << endl;
}

void print_list2(Node2* head)
{
    if (head == nullptr)
    {
        cout << "empty list"<< endl;
        return;
    }
    
    cout << head->value << "->" ;
    while (head->next != nullptr)
    {
        Node2* next = head->next;
        cout << next->value << "->";
        head = next;
    }
    cout << " ||" << endl;
}


Node* del_last_k(Node head, int k){
    Node cur = head;
    k--;
    while (cur.get_next() != nullptr)
    {
        cur = *cur.get_next();
        k--;
    }
    cout << cur.get_next() << " k is: " << k<<endl;
    if (k == 0)
    {
        cout << "remove first:" << endl;
        return head.get_next();
    }

    if (k > 0)
    {
        cout <<  "k is large than length of list, do not remove" << endl;
        return nullptr;
    }
    // process k < 0
    Node* cur2 = &head;
    while (++k < 0)
    {
        cur2 = cur2->get_next();
    }
    if (cur2->get_next()->get_next() != nullptr){
        Node* tmp = cur2->get_next()->get_next();
        cur2->set_next(tmp);
    }else{
         cur2->set_next(nullptr);
    }
    return new Node(head);
}

Node2 * del_last_k_b(Node2 *head, int k)
{   
    if (k<1){
        return head;
    }

    Node2* curr = head;
    k--;
    while (curr->next != nullptr)
    {
        curr = curr->next;
        k--;
    }
    cout << "k is:" << k << endl;
    if (k > 0)
    {
        cout << "k large than length of list" << endl;
        return head;
    }

    if (k==0)
    {
        // head = head->next;
        cout << head->next->value<<endl;
        return head->next;
    }

    curr = head;
    while (++k < 0)
    {
        curr = curr->next;
    }

    if (curr->next->next != nullptr)
    {
        curr->next->next->last = curr;
        curr->next = curr->next->next;
    }else{
         curr->next = nullptr;
    }
    
    
    
    
    return head;

}


int main(int argc, char* argv[])
{
    Node headA(2);
    Node nodeA2(4);
    headA.set_next(&nodeA2);
    Node nodeA3(6);
    nodeA2.set_next(&nodeA3);
    Node nodeA4(8);
    nodeA3.set_next(&nodeA4);
    nodeA4.set_next(nullptr);

    // print_list(&headA);
    // Node* head = del_last_k(headA, 4);
    // print_list(head);      


    Node2* headB = new Node2(2) ;
    Node2* nodeB2 = new Node2(4) ;
    headB->next = nodeB2;
    nodeB2->last = headB;

    Node2* nodeB3 = new Node2(6) ;
    nodeB2->next = nodeB3;
    nodeB3->last = nodeB2;

    Node2* nodeB4 = new Node2(8) ;
    nodeB3->next = nodeB4;
    nodeB4->last = nodeB3;
    print_list2(headB);
    Node2* res =  del_last_k_b(headB, 4);
    print_list2(res);
    return 0;
}

