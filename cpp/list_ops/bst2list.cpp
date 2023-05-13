/**
 * @file bst2list.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-05
 * 
 * @copyright Copyright (c) 2022
 * trans binary search tree to a sorted linked list
 */

#include<iostream>
#include<queue>

using namespace std;

class Node{
    public:
    Node* left;
    Node* right;
    // Node* next = right;
    int value;

    Node(int v){
        value = v;
    }
};

class ReturnType{
    public:
    Node* start;
    Node* end;

    ReturnType(Node* s, Node* e){
        start = s;
        end = e;
    }
};

Node* tree2list(Node* root);
void trverse(Node* root, queue<int>&);
Node* tree2list2(Node* root);
ReturnType trverse2(Node* root);
void print_list(Node*);

int main(int argc, char* argv[]){
    Node* root = new Node(6);

    Node* l1 = new Node(3);
    Node* r1 = new Node(8);

    root->left = l1;
    root->right = r1;


    Node* ll1 = new Node(2);
    Node* lr1 = new Node(4);
    l1->left = ll1;
    l1->right = lr1;

    Node* rl1 = new Node(7);
    Node* rr1 = new Node(9);
    r1->left = rl1;
    r1->right = rr1;


    Node* res = tree2list2(root);
    print_list(res);
    return 0;
}

Node* tree2list(Node* root){
    queue<int> q;
    trverse(root, q);
    cout << q.size() << endl;

    Node* head = nullptr;
    Node* curr = head;
    while(!q.empty()){
        // cout << "444" << endl;
        if(head == nullptr){
            cout << q.front() << " ->" << endl;
            curr = new Node(q.front());
            head = curr;
            // cout << q.front() << " ->";
            q.pop();
        }else{
            Node* node = new Node(q.front());;
            curr->right = node;
            node->left = curr;
            curr = curr->right;
            q.pop();
        }
        
       
    }
    cout << endl;

    return head;
}

void trverse(Node* node, queue<int> &q){
    if(node == nullptr){
        return;
    }
    trverse(node->left, q);
    // cout << " =>"<< node->value;
    q.push(node->value);
    trverse(node->right, q);
}

Node* tree2list2(Node* root){
    ReturnType ret = trverse2(root);
    return ret.start;
}

ReturnType trverse2(Node* node){
    if(node == nullptr){
        return ReturnType(nullptr, nullptr);
    }

    // if(node->left == nullptr && node->right==nullptr){
    //     return ReturnType(node, node);
    // }

    Node* start = nullptr;
    Node* end = nullptr;

    ReturnType left =  trverse2(node->left);
    if(left.start != nullptr){
        left.end->right = node;
        node->left = left.end;
        start = left.start;
    }
    start = left.start !=nullptr ? left.start:node;
    end = node;

    ReturnType right = trverse2(node->right);
    if(right.end != nullptr ){
        end->right = right.start;
        right.start->left = end;

        end = right.end;
    }
    
    return ReturnType(start, end);
}


void print_list(Node* head){
    if(head == nullptr){
        cout << "list is empty" << endl;
        return;
    }
    int max_cnt = 0;
    cout << head->value << "->" ;
    while (head->right != nullptr)
    {
        cout << head->right->value << "->" ;
        head = head->right;
        max_cnt ++ ;
        if (max_cnt == 30){
            break;
        }
    }

    cout << "null |" << endl;
}