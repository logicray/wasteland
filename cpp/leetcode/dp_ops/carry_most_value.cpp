/**
 * @file carry_most_value.cpp
 * @author logic-pw (you@domain.com)
 * @brief each item has a weight and value, given a package can load n kg, return the largest value by items combination
 * @version 0.1
 * @date 2023-02-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include<iostream>

using namespace std;

int carry_most_value(int[][2], int item_num, int const capacity);

int main(int argc, char* argv[]){
    int inp[][2] = {{3000,4}, {2000,3}, {1500,1}, {2000,1}};
    int res = carry_most_value(inp, 4, 4);
    cout << "res: " << res << endl;

    return 0;
}

int carry_most_value(int arr[][2], int item_num, int const capacity){
    int dp[item_num+1][capacity+1] = {0}; //initial to 0
    for(int i=0;i<=item_num;i++){
        dp[i][0] =0;
    }
    for(int j=0;j<=capacity;j++){
        dp[0][j] =0;
    }
    cout << dp[0][1] << endl;
    for (int i=1; i<=item_num; i++){
        for(int j = 1; j<= capacity; j++){
            if (j < arr[i-1][1] ){ //special case
                dp[i][j] = dp[i-1][j];
            }else{
                dp[i][j] = max(dp[i-1][j], arr[i-1][0] + dp[i-1][j-arr[i-1][1]]);
            }
        }
    }
    return dp[item_num][capacity];
}



