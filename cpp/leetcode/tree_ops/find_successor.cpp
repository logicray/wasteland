/**
 * @file find_successor.cpp
 * @author logic-pw (you@domain.com)
 * @brief find the successor node in pre order binary search tree 
 * @version 0.1
 * @date 2023-02-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include<stack>
#include<iostream>

using namespace std;

class TreeNode{
    public:
      int val;
      TreeNode* left;
      TreeNode* right;

    
      TreeNode(int value){
        val = value;
      }
};

TreeNode* find_successor(TreeNode* root, TreeNode* p);
TreeNode* find_successor2(TreeNode* root, TreeNode* p);

int main(int argc, char* argv[]){
    TreeNode* root = new TreeNode(8);
    TreeNode* left1 = new TreeNode(6);
    TreeNode* right1 = new TreeNode(10);
    root->left = left1;
    root->right = right1;

     TreeNode* left1left = new TreeNode(4);
     TreeNode* left1right = new TreeNode(7);
     left1->left = left1left;
     left1->right = left1right;

    TreeNode* right1left = new TreeNode(9);
    TreeNode* right1right = new TreeNode(12);
    right1->left = right1left;
    right1->right = right1right;

    TreeNode* res1 = find_successor(root, right1);
    cout << "res1: " << res1->val << endl;
    cout << "---------- split line ----------" << endl;
    TreeNode* res2 = find_successor2(root, right1);
    cout << "res2: " << res2->val << endl;
    return 0;
}



TreeNode* find_successor(TreeNode* root, TreeNode* p){
    //
    TreeNode* successor = nullptr;

    if(root == nullptr){
        return successor;
    }

    stack<TreeNode*> q;
    TreeNode* curr = root;
    while(curr != nullptr){
        q.push(curr);
        curr = curr->left;
    }

    TreeNode* last = nullptr;
    
    while(!q.empty()){
        TreeNode* tmp = q.top();
        q.pop();

        if(last == p){
            successor = tmp;
            break;
        }
        if(tmp == p){
            last = p;
        }
        if(tmp->right != nullptr){
            curr = tmp->right;
            while(curr != nullptr){
                q.push(curr);
                curr = curr->left;
            }
        }
    }
    return successor;
}

TreeNode* find_successor2(TreeNode* root, TreeNode* p){
    TreeNode* successor = nullptr;
    if(p->right !=nullptr){
        successor = p->right;
        while(successor->left != nullptr){
            successor = successor->left;
        }
        return successor;
    }

    TreeNode* node = root;
    while (node != nullptr){
        if (node->val > p->val){
            successor = node;
            node = node->left;
        }else{
            node = node->right;
        }
    }
    return successor;
    
}