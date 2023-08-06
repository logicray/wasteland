/**
 * @file partition.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-04-02
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

    Node(){
        value = 0;
    }
};
Node* list_partition1(Node* , int );
Node* list_partition2(Node* , int );
void swap(Node*, int, int);
void print_list(Node*);

int main(int argc, char* argv[]){
    Node* head = new Node(1);
    Node* node2 = new Node(4);
    Node* node3 = new Node(3);
    Node* node3_d = new Node(2);
    Node* node4 = new Node(5);
    Node* node5 = new Node(7);

    head->next = node2;
    node2->next = node3;
    node3->next = node3_d;
    node3_d->next = node4;
    node4->next = node5;

    print_list(head);
    Node* res1 = list_partition2(head, 5);
    print_list(res1);
    return 0;
}


Node* list_partition1(Node* head, int pivot){
    if (head == nullptr || head->next == nullptr){
        return head;
    }
    // 求出list 长度 n
    int n = 1;
    Node* curr = head;
    while (curr->next != nullptr){
        n++;
        curr = curr->next;
    }

    cout << n << endl;
    //list 转为array
    Node* arr = new Node[n];
    n = 0;
    while (head != nullptr)
    {
        arr[n++] = *head;
        head = head->next;
    }
    cout << arr[0].value << " -- " << arr[5].value << endl;
    

    int less_end = -1, pre_large = n;
    int index = 0;
    
    while (index != pre_large){
       if (arr[index].value < pivot){
         swap(arr, ++less_end, index++);
       }else if (arr[index].value > pivot){
        swap(arr, --pre_large, index);
       }else{
           index ++;
       }
    }

    cout << "array content: ";
    for (int i = 0; i < n; i++){
         cout << arr[i].value << "->";
    }
    cout << endl;
    
    head = &arr[0];
    int i= 0;
    curr = head;
    while (++i<n){
        curr->next = &arr[i];
        curr = curr->next;
    }
    curr->next = nullptr;
    return head;
}


Node* list_partition2(Node* head, int pivot){
    if (head == nullptr || head->next == nullptr){
        return head;
    }
    
    Node *less_start = nullptr, *eq_start = nullptr, *large_start = nullptr;
    Node *less_curr = nullptr, *eq_curr = nullptr, *large_curr = nullptr;

    Node* curr = head;
    while (curr != nullptr){
        cout << curr->value << "   ||" << endl;
        Node *next = curr->next;
       if (curr->value < pivot){
          if (less_start == nullptr){
              less_start = curr;
              less_curr = less_start;
              less_curr->next = nullptr;
          }else{
              Node *tmp = curr;
              tmp->next = nullptr;
              less_curr->next = tmp;
              less_curr = tmp;
          }
       }else if (curr->value > pivot){
           if (large_start == nullptr){
              large_start = curr;
              large_curr = large_start;
              large_curr->next = nullptr;
          }else{
              large_curr->next = curr;
               Node *tmp = curr->next;
              large_curr = tmp;
              large_curr->next = nullptr;
          }
       }else{
           if (eq_start == nullptr){
              eq_start = curr;
              eq_curr = eq_start;
              eq_curr->next = nullptr;
          }else{
              eq_curr->next = curr;
               Node *tmp = curr->next;
              eq_curr = tmp;
              eq_curr->next = nullptr;
          }
       }
       curr = next;
    }
    cout << "start print:" << endl;
    print_list(less_start);
    print_list(eq_start);
    print_list(large_start);

    //concate less_start, eq_start, large_start
    head = less_start;
    curr = head;
    while(curr->next != nullptr){
        curr = curr->next;
    }
    curr->next = eq_start;
     while(curr->next != nullptr){
        curr = curr->next;
    }
    curr->next = large_start;
    print_list(head);
    return head;
}


void swap(Node arr[], int a, int b){
    Node tmp = arr[a];
    arr[a] = arr[b];
    arr[b] = tmp;
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
