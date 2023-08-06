/**
 * @file traverse_by_level.cpp
 * @author logic-pw
 * @brief 
 * 1. binary tree traverse by level
 * 2. n -ary tree traverse by level
 * @version 0.1
 * @date 2022-05-21
 * 
 * @copyright Copyright (c) 2022
 */
#include<iostream>
#include<queue>
#include<vector>
#include<set>

using  namespace std;

class Node{
    public:
    int value;
    Node* left;
    Node* right;

    Node(int v){
        value = v;
    }
};
class NaryNode{
    public:
    int value;
    vector<NaryNode*> children;

    NaryNode(int v){
        value = v;
    }
};

class StateNode{
    public:
    int depth;
    Node* node;

    StateNode(Node* n, int d){
        node = n;
        depth = d;
    }
};


void level_traverse(Node*);
void level_traverse_v2(Node*);
void level_traverse2(NaryNode*);


int main(int argc, char const *argv[])
{
    Node* root = new Node(12);
    
    Node* left1 = new Node(10);
    Node* right1 = new Node(19);
    root->left = left1;
    root->right = right1;

    Node* left12 = new Node(8);
    Node* right12 = new Node(13);

    left1->left = left12;
    left1->right = right12;
    
    Node* left22 = new Node(8);
    Node* right22 = new Node(13);
    right1->left = left22;
    right1->right = right22;


    level_traverse(root);
    cout << "-----level traverse v2-----" << endl;
    level_traverse_v2(root);

    cout << "-----n-ary tree traverse-----" << endl;
    NaryNode* nroot = new NaryNode(12);

    NaryNode* child[3];
    child[0] = new NaryNode(10);
    child[1] = new NaryNode(19);
    child[2] = new NaryNode(21);
    nroot->children.push_back(child[0]);
    nroot->children.push_back(child[1]);
    nroot->children.push_back(child[2]);

    level_traverse2(nroot);

    
    return 0;
}


void level_traverse(Node* root){
    if (root == nullptr){
        return;
    }
    queue<Node*> q;
    q.push(root);
    int level = 0;
    while (!q.empty())
    {
        int size = q.size();
        for (int i=0; i< size; i++){
            Node* node = q.front();
            q.pop();
            cout << "level" << level << ", index "<< i << ", node " << node->value << endl;

            if (node->left != nullptr){
                q.push(node->left);
            }

            if (node->right != nullptr){
                q.push(node->right);
            }
        }
        level++;
    }
}

void level_traverse_v2(Node* root){
    if (root == nullptr){
        return;
    }

    queue<StateNode*> q;
    q.push(new StateNode(root, 0));
    while(!q.empty()){
        StateNode* state_node = q.front();
        q.pop();
        Node* node = state_node->node;
        int depth = state_node->depth;
        cout << "level:" << depth << " node:" << node->value << endl;
        if (node->left != nullptr){
            q.push(new StateNode(node->left, depth+1));
        } 

        if (node->right != nullptr){
            q.push(new StateNode(node->right, depth+1));
        }
    }
}

void level_traverse2(NaryNode* root){
    if (root == nullptr){
        return;
    }

    queue<NaryNode*> q;
    q.push(root);
    while(!q.empty()){
        int size = q.size();
        for (int i=0; i< size; i++){
            NaryNode* cur = q.front();
            // auto x = cur->children;
            cout << cur->value << endl;
            vector<NaryNode*> chilren = cur->children;
            int children_num = chilren.size();
            for (int j=0; j<children_num; j++){
                NaryNode* node = chilren[j];
                q.push(node);
            }
            q.pop();
        }
    }
}


