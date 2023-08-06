/**
 * @file dijkstra.cpp
 * @author logic-pw
 * @brief dijkstra algo implement
 * @version 0.1
 * @date 2023-01-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include<vector>
#include<set>
#include<unordered_map>
#include<iostream>
#include<limits.h>

#define max_dist 10000
using namespace std;

vector<int> dijkstra(int, vector<vector<int>>);
int dijkstra2(int start, int end, unordered_map<int, unordered_map<int, int>> graph);

int main(int argc, char* argv[]){
    vector<int> v0 = {0, 2, max_dist, 2};
    vector<int> v1 = {max_dist, 0, max_dist, 4};
    vector<int> v2 = {max_dist, max_dist, 0, max_dist};
    vector<int> v3 = {max_dist, max_dist, 3, 0};
    vector<vector<int>> inp = {v0, v1, v2, v3};

    vector<int> res1 = dijkstra(0, inp);
    for(auto i: res1){
        cout << i << endl;
    }
    cout << "---------- split line ----------" << endl;
    unordered_map<int,int> start({ {2,5}, {3, 0} });
    unordered_map<int,int> b({ {4, 15}, {5,20}});
    unordered_map<int,int> c({{4, 30}, {5,35}});
    unordered_map<int,int> d({ {6,20}});
    unordered_map<int,int> e({ {6,10}});
    unordered_map<int,int> end({});
    unordered_map<int, unordered_map<int, int>> graph({
        {1, start},
        {2, b},
        {3, c},
        {4, d},
        {5, e},
        {6, end},
    });
    int res2 = dijkstra2(1,6,graph);
    cout << "res2: " << res2 << endl;
    return 0;
}


vector<int> dijkstra(int src_idx, vector<vector<int>> graph){
    int size = graph.size();
    int* dist = new int[size];
    
    for (int i=0; i< size; i++){
        dist[i] = graph[src_idx][i];
    }

    set<int> s = {0};
    set<int> unprocessed = {1,2,3};
    while (!unprocessed.empty()){
        //find min path from 
        int min_path_node = 0;
        int min_value = max_dist;
        cout << "unprocessed: " << unprocessed.size() << endl;
        for(int vi : unprocessed){
            if(dist[vi] < min_value){
                min_path_node = vi;
                min_value = dist[vi];
            }
        }
        cout << "== " << min_path_node << endl;
        unprocessed.erase(min_path_node);
        s.insert(min_path_node);

        // update distance in 
        for (auto vj : unprocessed){
            if(dist[min_path_node] + graph[min_path_node][vj] < dist[vj]){
                dist[vj] = dist[min_path_node] + graph[min_path_node][vj];
            }
        }

        cout << "after update" << endl;
        for(int i=0;i<size; i++){
            cout << dist[i] << endl;
    }
    }
    
    vector<int> res;
    for(int i=0;i<size; i++){
        res.push_back(dist[i]);
    }
    return res;
}


int dijkstra2(int start, int end, unordered_map<int, unordered_map<int,int>> graph){
    unordered_map<int, int> distance_map(graph[start]);
    set<int> unprocessed_nodes;
    for(auto i : distance_map){
        unprocessed_nodes.insert(i.first);
    }
    while(!unprocessed_nodes.empty()){
        //find nearest node in unprocessed set
        int tmp_node = start;
        int tmp_dist = max_dist;
        for(int node : unprocessed_nodes){
            int d = distance_map[node];
            if(d < tmp_dist){
                tmp_node = node;
                tmp_dist = d;
            }
        }
        if (tmp_node == end){
            return distance_map[tmp_node];
        }
        for (auto i :unprocessed_nodes){
            cout << "content" << i << "  ";
        }
         
        unprocessed_nodes.erase(tmp_node);
        cout << "size:" << unprocessed_nodes.size() << " tmp node:" << tmp_node << endl;
        //update the node's neighbor in 
        for(auto j: graph[tmp_node]){
            unprocessed_nodes.insert(j.first);
            if (distance_map.find(j.first) == distance_map.end()){
                distance_map[j.first] = max_dist;
            }

            if (distance_map[tmp_node] + graph[tmp_node][j.first] < distance_map[j.first]){
                distance_map[j.first] = distance_map[tmp_node] + graph[tmp_node][j.first];
            }
        }
    }
    return -1;
}