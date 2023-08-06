/**
 * @file max_water.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2023-01-18
 * 
 * @copyright Copyright (c) 2023
 * max water in trap
 */

#include<iostream>
#include<stack>

using namespace std;

int trap1(int* arr, int size);
int trap2(int* arr, int size);
int trap3(int* arr, int size);

int main(int argc, char* argv[]){
    int inp[] = {0,1,0,2,1,0,1,3,2,1,2,1};
    int size = sizeof(inp)/sizeof(inp[0]);
    cout << "szie inp: " << size << endl;
    int res1 = trap1(inp, size);
    cout << "res1: " << res1 << endl;

    cout << "------- split line -------" << endl;
    int res2 = trap2(inp, size);
    cout << "res2: " << res2 << endl;

    cout << "------- split line -------" << endl;
    // int inp2[] = {3,2,5,4,6};
    int res3 = trap3(inp, size);
    cout << "res3: " << res3 << endl;
    return 0;
}

int trap1(int *arr, int size){
    // 
    int n = size;
    if(n <= 2){
        return 0;
    }

    int h_left[n];
    int h_right[n];
    h_left[0] = 0; // h_left[i] 表示i左边比最高的
    h_right[n-1] = 0;  // h_right[i] 表示 i右边最高的
    for(int i = 1; i<n; i++){
        h_left[i] = max(arr[i-1], h_left[i-1]);
        h_right[n- i - 1] = max(arr[n - i], h_right[n - i]);
    }

    int res = 0;
    for(int j = 0; j<n; j++){
        res += max(0, min(h_left[j],h_right[j]) - arr[j]);
    }
    return res;
}

int trap2(int* arr, int size){
    int n = size;
    if(n <= 2){
        return 0;
    }
    //遍历的左侧和右侧起始点
    int l_index = 0;
    int r_index = n-1;
    //当前点的左侧和右侧最高点
    int l_height = 0;
    int r_height = 0;

    int res = 0;
    while (l_index <= r_index){
        if (l_height < r_height){
            res += max(0,(l_height - arr[l_index]));
            l_height = max(l_height, arr[l_index]);
            l_index++;
        }else{
            res += max(0,(r_height - arr[r_index]));
            r_height = max(r_height, arr[r_index]);
            r_index--;
        }
    }
    return res;
}

int trap3(int* arr, int size){
    if(size <= 2){
        return 0;
    }
    int res = 0;
    stack<int> st; //decrease mono stack
    for (int i=0; i< size; i++){

        if(st.empty()){
            st.push(i);
        }else{
            if(arr[i] <= arr[st.top()]){
                st.push(i);
            }else{
                
                while(!st.empty() && arr[i] > arr[st.top()]){
                    int idx = st.top();
                    st.pop();
                    // equal, pop
                    while(!st.empty() && arr[st.top()] == arr[idx]){
                        idx = st.top();
                        st.pop();
                    }
                    if (!st.empty()){
                        int next = st.top();
                        res += max((min(arr[i], arr[next]) - arr[idx]),0) * (i-next-1);
                    }else{
                        break;
                    }
                    cout << "res::: " << res << "::   ";
                    
                }
                st.push(i);
            }
        }
        
        cout << "stack info << " << arr[st.top()] << "--- " << endl;;
        
    }
    return res;
}