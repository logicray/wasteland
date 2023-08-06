/**
 * @file bfs.cpp
 * @author your name (you@domain.com)
 * @brief graph breadth first search
 * @version 0.1
 * @date 2023-01-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include<iostream>
#include<queue>
#include<vector>
#include<set>

using namespace std;

class GraphNode{
    public:
    int value;
    vector<GraphNode*> adj;

    GraphNode(int v){
        value = v;
    }
};
void bfs(GraphNode*);


int main(int argc , char* argv[]){
    cout << "bfs traverse" << endl;
    GraphNode* groot = new GraphNode(12);
    GraphNode* adj0 = new GraphNode(10);
    GraphNode* adj01 =  new GraphNode(8);
    vector<GraphNode*> adj1 = {adj0, adj01, new GraphNode(4) };
    groot->adj = adj1;

    vector<GraphNode*> adj2 = {new GraphNode(5), adj01, new GraphNode(7) };
    adj0->adj = adj2;
    bfs(groot);
    return 0;

}


void bfs(GraphNode* root){
    if (root == nullptr){
        return;
    }

    queue<GraphNode*> q;
    set<GraphNode*> visited;

    q.push(root);
    visited.insert(root);
    int step = 0;
    while (!q.empty())
    {
        int size = q.size();
        for (int i=0; i< size; i++){
            GraphNode* node = q.front();
            cout << "step:" << step << "node: " << node->value << endl;
            for (GraphNode* n : node->adj){
                if (visited.count(n) != 1){
                    q.push(n);
                    visited.insert(n);
                }
            }
            q.pop();
        }
        step++;
    }
}