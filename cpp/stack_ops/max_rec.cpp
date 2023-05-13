/**
 * @file max_rec.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-15
 * 
 * @copyright Copyright (c) 2022
 * max rectangle full with 1 in a matrix  
 */

#include<iostream>
#include<stack>

using namespace std;

int get_max_rec(int, int, int*);
int get_max_area(int[], int);

int main(int argc, char* argv[])
{
    int mat[3][4] = {
        {1, 0, 0, 1}, 
        {1, 1, 1, 1}, 
        {1, 1, 1, 0}
    };
    int max_area = get_max_rec(3, 4, (int*)mat);
    cout << "max area:" << max_area << endl;
}

int get_max_rec(int row, int col, int* mat)
{
    int height[4]{};
    int max_area = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j=0; j< col; j++)
        {
            int v = *(mat + i * col + j);
            height[j] = *(mat + i * col + j) == 0 ? 0:height[j]+*(mat + i * col + j);
            cout << height[j] << " ";
        }
        // cout << endl;
        max_area = max(max_area, get_max_area(height, col));
        cout << "max area:" << max_area << endl;
    }
    return max_area;
}

int get_max_area(int height[], int size)
{
    stack<int> stk;
    int max_area = 0;
    for(int i=0; i< size; i++)
    {
        while(!stk.empty() && height[stk.top()] > height[i])
        {
            int index = stk.top();
            stk.pop();
            int k  = stk.empty() ? -1 : stk.top();
            int curr_area = (i-k-1) * height[index];
            cout << "curr_area:" << curr_area << endl;
            max_area = max(max_area, curr_area);
        }
        stk.push(i);
    }

    while (!stk.empty())
    {
        int index = stk.top();
        stk.pop();
        int k = stk.empty() ? -1 : stk.top();
        int curr_area = (size-k-1) * height[index];
        // cout << "curr_area2:" << curr_area << endl;
        max_area = max(max_area, curr_area);
    }
    return max_area;
}

