/**
 * @file intersection.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-04-10
 * 
 * @copyright Copyright (c) 2022
 * 返回两个链表相交的第一个节点
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

Node* has_circle(Node*);
Node* has_intersetion(Node*, Node*);
Node* intersection_no_circle(Node *h1, Node *h2);
Node* intersection_with_circle(Node*, Node*, Node*, Node*);
void swap(Node *h1, Node *h2);
void print_list(Node*);
void print_node(string, Node*);

int main(int argc, char *argv[]){
    Node* n1 = new Node(6);
    Node* n2 = new Node(9);
    Node* n3 = new Node(3);
    Node* n3_d = new Node(4);
    Node* n4 = new Node(1);
    Node* n5 = new Node(2);

    n1->next = n2;
    n2->next = n3;
    n3->next = n3_d;
    n3_d->next = n4;
    n4->next = n5;
    n5->next = n3_d;


    Node* m1 = new Node(6);
    Node* m2 = new Node(9);
    Node* m3 = new Node(3);
    Node* m3_d = new Node(4);
    Node* m4 = new Node(1);
    Node* m5 = new Node(2);

    m1->next = m2;
    m2->next = n4;
    // m3->next = m3_d;
    // m3_d->next = m4;
    // m4->next = m5;
    // n5->next = n3_d;

    // print_node("node n1 " ,has_circle(n1));
    // print_node("m1", has_circle(m1));
    // cout << "has circle n:" <<  has_circle(n1)->value <<  endl;
    // cout << "has circle m:" <<  has_circle(m1) <<  endl;
    Node* res = has_intersetion(n1, m1);
    print_node("res:", res);
    return 0;
}

/**
 * @brief 
 * 
 * @param head 链表头节点
 * @return Node* 
 */
Node* has_circle(Node* head){
    if (head->next == head){
        return head;
    }
    
    Node *slow = head;
    Node *fast = head;
    while(slow != nullptr && fast->next != nullptr){
        slow = slow->next;
        fast = fast->next->next;
        // cout << "55" << endl;
        if(slow==nullptr || fast==nullptr){
            return nullptr;
        }
        if(slow == fast){
            fast = head; //fast
            break;
        }
    }
    cout << "first end;" << endl;
    while(fast->next!=nullptr && slow->next!=nullptr){
        fast = fast->next;
        slow = slow->next;
        if (fast == slow){
            return fast;
        }
    }
    return nullptr;
}

Node* intersection_with_circle(Node *h1, Node *h2, Node *e1, Node *e2){
    //如果入环的点是同一个，则相交，只需查找相交的点
    if (e1 == e2){
        cout << "same entry point" << endl;
        /* code */
        int len1 = 0, len2 = 0;
        Node *curr1 = h1;
        while(curr1 != e1){
            curr1 = curr1->next;
            len1++;
        }

        Node *curr2 = h2;
        while(curr2 != e1){
            curr2 = curr2->next;
            len2++;
        }
        
        int step_gap = len1 > len2 ? len1-len2: len2-len1;

        curr1 = h1; curr2= h2;
        while(step_gap > 0){
            if(len1 > len2){
                curr1 = curr1->next;
            }else{
                curr2 = curr2->next;
            }
            step_gap--;
        }

        while(curr1 != e1 && curr2 != e1){
            if(curr1 == curr2){
                return curr1;
            }
            curr1 = curr1->next;
            curr2 = curr2->next;
        }
        
        return nullptr;
    }

    //如果不是同一个，则以其中一个点为起点，在环上查找一圈，看是否与另外一个点相遇，
    //如果相遇返回任意一个入环点，不相遇则两个链表不相交，
    Node* curr1 = e1;
    while(curr1 != nullptr){
        curr1 = curr1->next;
        if(curr1 == e2){
            return e1;
        }
    }
    return nullptr;
}

Node* intersection_no_circle(Node *h1, Node *h2){
    int len1 = 0,len2 = 0;
    Node* curr1 = h1;
    while(curr1 != nullptr){
        curr1 = curr1->next;
        len1++;
    }

    Node* curr2 = h2;
    while(curr2 != nullptr){
        curr2 = curr2->next;
        len2++;
    }
    cout << "len1: " << len1 << "   len2:" << len2 << endl;

    bool len1_first = len1 > len2 ? true:false;
    int start_before = len1 > len2 ? len1-len2 : len2-len1;

    cout << "start before:" << start_before << endl;
    curr1 = h1;
    curr2 = h2;
    while(start_before > 0){
        if (len1_first){
           curr1 = curr1->next;
        }else{
            curr2 = curr2->next;
        }
        start_before--;
    }
    cout << "curr1->value"<<curr1->value << endl;
    cout << "curr2->value"<<curr2->value << endl;

    while (curr1 != nullptr && curr2 != nullptr){
        if(curr1 == curr2){
            return curr1;
        }
        curr1 = curr1->next;
        curr2 = curr2->next;
    }
    return nullptr;
}


Node* has_intersetion(Node *h1, Node *h2){
    print_list(h1);
    print_list(h2);
    Node* h1_cirle = has_circle(h1);
    Node* h2_circle = has_circle(h2);

    print_list(h1);
    print_list(h2);

    //如果都没有环，
    if (h1_cirle == nullptr && h2_circle == nullptr){
        cout << "both no circle" << endl;
        return intersection_no_circle(h1, h2);
    }

    //都有环
    if (h1_cirle != nullptr && h2_circle != nullptr){
        cout << "both have circle" << endl;
        return intersection_with_circle(h1,h2, h1_cirle, h2_circle);
    }

    //其他情况，一个有环，一个没环，不可能相交
    return nullptr;
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


void print_node(string desc, Node* n){
    if(n == nullptr){
        cout << desc << " nullptr" << endl;
        return;
    }else{
        cout << desc << " " << n->value << endl;
    }
}

void swap( Node* a,  Node* b){
     Node* tmp = a;
     a = b;
     b = tmp;
}