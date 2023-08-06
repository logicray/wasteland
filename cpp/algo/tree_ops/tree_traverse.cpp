/**
 * @file tree_traverse.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-05-21
 * 
 * @copyright Copyright (c) 2022
 * preorder, inorder postorder binary tree traverse
 */

#include<iostream>
#include<queue>
#include<stack>

using namespace std;


class Node{
    public:
    int value;
    Node* left;
    Node* right;

    Node(int v){
        value = v;
    }
};

void pre_order(Node* root);
void in_order(Node* root);
void post_order(Node* root);

void pre_order2(Node* root);
void in_order2(Node* root);
void post_order2(Node* root);
void post_order3(Node* root);


int main(int argc, char* argv[]){
    Node* root = new Node(12);
    
    Node* left1 = new Node(10);
    Node* right1 = new Node(19);
    root->left = left1;
    root->right = right1;

    Node* left2 = new Node(8);
    Node* right2 = new Node(13);

    left1->left = left2;
    left1->right = right2;

    post_order(root);
    post_order2(root);
    post_order3(root);

    return 0;
}

void pre_order(Node* root){
    if (root == nullptr){
        return;
    }
    
    cout << root->value << " ";
    pre_order(root->left);
    pre_order(root->right);
}

void pre_order2(Node* root){
    if (root == nullptr){
        return;
    }
    
    stack<Node*> q;
    q.push(root);

    cout << "1-" << endl;
    while (!q.empty()){
        Node* tmp = q.top();
        q.pop();
        cout << tmp->value << " ";
        
        if(tmp->right != nullptr){
            q.push(tmp->right);
        }
        if(tmp->left != nullptr){
            q.push(tmp->left);
        }
    }
}


void in_order(Node* root){
    if(root == nullptr){
        return ;
    }
    in_order(root->left);
    cout << root->value << " ";
    in_order(root->right);
}

void in_order2(Node* root){
    if(root == nullptr){
        return ;
    }
    stack<Node*> q;
    
    Node* curr = root;
    while(curr != nullptr){
        q.push(curr);
        curr = curr->left;
    }

    while(!q.empty()){
        Node* tmp = q.top();
        q.pop();

        cout << tmp->value << " ";
        if(tmp->right != nullptr){
            curr = tmp->right;
            while(curr != nullptr){
                q.push(curr);
                curr = curr->left;
            }
        }
    }
}

void post_order(Node* root){
    if(root == nullptr){
        return;
    }
    post_order(root->left);
    post_order(root->right);
    cout << root->value << " ";
}

void post_order2(Node* root){
    if(root == nullptr){
        return;
    }
    stack<Node*> p;
    stack<Node*> q;
    
    p.push(root);
    while(!p.empty()){
        Node* tmp = p.top();
        p.pop();

        if(tmp->left != nullptr){
            p.push(tmp->left);
        }

        if (tmp->right != nullptr){
            p.push(tmp->right);
        }

        q.push(tmp);
    }
    cout << "1--=" << endl;
    while(!q.empty()){
        cout << q.top()->value << " ";
        q.pop();
    }
}

void post_order3(Node* root){
    cout << " post order 3 " << endl;
    if(root == nullptr){
        return;
    }
    stack<Node*> q;
    q.push(root);

    // t is current top node in stack q, c is last poped node
    Node* t = q.top();
    Node* c = nullptr;

    while(!q.empty()){
        t = q.top();
        if(t->left != nullptr && t->left != c && t->right != c){
            q.push(t->left);
        }else if(t->right != nullptr && t->right != c){
            q.push(t->right);
        }else{
            cout << t->value << " ";
            q.pop();
            c = t;
        }
        
        
    }
}
