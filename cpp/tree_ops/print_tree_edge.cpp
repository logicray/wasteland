/**
 * @file print_tree_edge.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-06-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include<iostream>

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
void print_edge1(Node* root);
void print_edge2(Node* root);
int get_height(Node* root, int h);
void set_edge_map(Node*, int, Node* e_map[][2]);
void print_leaf_node(Node*, int, Node* e_map[][2]);
void print_left_edge(Node* root, bool print);
void print_right_edge(Node* root, bool print);
Node* tree_define();

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

    Node* right3 = new Node(9);
    left2->left = nullptr;
    left2->right = right3;

    Node* left4 = new Node(11);
    right3->left = left4;
    
    Node* root2 = tree_define();
    int res = get_height(root2, 0);
    cout << "tree height is: " << res << endl;
    print_edge2(root2);
}

Node* tree_define(){
    Node* root = new Node(1);
    // level 1
    Node* left1 = new Node(2);
    Node* right1 = new Node(3);
    root->left = left1;
    root->right = right1;

    //level 2
    Node* left2 = new Node(4);
    Node* right21 = new Node(5);
    Node* right22 = new Node(6);

    left1->right = left2;
    right1->left = right21;
    right1->right = right22;

    // level 3
    Node* left31 = new Node(7);
    Node* left32 = new Node(8);

    Node* right31 = new Node(9);
    Node* right32 = new Node(10);

    left2->left = left31;
    left2->right = left32;
    right21->left = right31;
    right22->right = right32;


    Node* left41 = new Node(11);
    Node* right41 = new Node(12);
    left32->right = left41;
    right31->left = right41;

    // level 5
    // Node* left51 = new Node(13);
    Node* left52 = new Node(14);

    Node* right51 = new Node(15);
    Node* right52 = new Node(16);

    // left41->left = left51;
    left41->right = left52;
    right41->left = right51;
    right41->right = right52;
    return root;
}

void print_edge1(Node* root){
    int height = get_height(root, 0);
    Node* edge_map[height][2];
    for(int i =0;i<height; i++){
        edge_map[i][0] = nullptr;
        edge_map[i][1] = nullptr;
        cout << edge_map[i][0] << "," <<  edge_map[i][1] << " ";
    }
    cout << endl;
    // cout << (edge_map[0][0] == nullptr) << endl;
    set_edge_map(root, 0, edge_map);
    
    //print left edge
    for(int i=0; i< height; i++){
        cout << edge_map[i][0]->value<< " ";
    }
    //print leaf node which not in left or right edge
    print_leaf_node(root, 0, edge_map);
    cout << endl;
    // print right edge
    for(int i=height-1; i >= 0; i--){
        if(edge_map[i][1] != edge_map[i][0]){
            cout << edge_map[i][1]->value << " ";
        }
    }
}

void set_edge_map(Node* root, int h, Node *e_map[][2]){
    if(root == nullptr){
        return;
    }
    if(e_map[h][0] == nullptr){
         e_map[h][0] =  root;
    }
   
    e_map[h][1] = root;
    set_edge_map(root->left, h+1, e_map);
    set_edge_map(root->right, h+1, e_map);
}

void print_leaf_node(Node* root , int h, Node* e_map[][2]){
    if(root == nullptr){
        return;
    }
    //此处if在大多数情况下不成立，可以反过来写
    if(root->left == nullptr && root->right == nullptr && root!=e_map[h][0] && root!=e_map[h][1]){
        cout << root->value << " ";
    }

    print_leaf_node(root->left, h+1, e_map);
    print_leaf_node(root->right, h+1, e_map);
}

int get_height(Node* root, int h){
    if(root == nullptr){
        return h;
    }
    return max(get_height(root->left, h+1),  get_height(root->right, h+1));
}


void print_edge2(Node* root){
    if(root == nullptr){
        return;
    }

    cout << root->value << " ";
    if(root->left != nullptr && root->right !=nullptr){
        print_left_edge(root->left, true);
        print_right_edge(root->right, true);
    }else {
        print_edge2(root->left != nullptr ? root->left: root->right);
    }
    cout << endl;
}


void print_left_edge(Node* root, bool print){
    if(root == nullptr){
        return;
    }
    if(print || (root->left == nullptr && root->right == nullptr)){
        cout << root->value << " ";
    }

    print_left_edge(root->left, print);
    print_left_edge(root->right, print && root->left == nullptr ? true : false);
}


void print_right_edge(Node* root, bool print){
    if(root == nullptr){
        return;
    }
    print_right_edge(root->left, print && root->right == nullptr ? true : false);
    print_right_edge(root->right, true);

    if(print || root->left == nullptr && root->right == nullptr){
        cout << root->value << " ";
    }
}