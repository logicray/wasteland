/**
 * @file lcs.cpp
 * @author logic-pw (you@domain.com)
 * @brief loongest common string and loongest common sequence
 * @version 0.1
 * @date 2023-02-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include<iostream>
#include<cstring>

using namespace std;


int lc_str(char arr1[], char arr2[]);
int lc_seq(char arr1[], char arr2[]);

int main(int argc, char* argv[]){
    char arr1[] = "fosh";
    char arr2[] = "fish";
    int lc_str_res = lc_str(arr1, arr2);
    cout << "res1: " << lc_str_res << endl;
    cout << "----- split line ----- " << endl;
    int lc_seq_res = lc_seq(arr1, arr2);
    cout << "res2: " << lc_seq_res << endl;

    return 0;
}

int lc_str(char arr1[], char arr2[]){
    int len1 = strlen(arr1);
    int len2 = strlen(arr2);
    int dp[len1][len2] = {0};
    int res = 0;
    for (int i = 0; i<len1; i++){
        for(int j=0; j<len2; j++){
            if(arr1[i] != arr2[j]) {
                dp[i][j]=0;
            }
            else{
                if (i >=1 && j >=1){
                  dp[i][i] = dp[i-1][j-1] + 1;
                }else{
                     dp[i][i] = 1;
                }
            }
            if (dp[i][i] > res){
                res = dp[i][j];
            }

        }
    }
    return res;
}

int lc_seq(char arr1[], char arr2[]){
    int len1 = strlen(arr1);
    int len2 = strlen(arr2);
    int dp[len1][len2] = {0};
    for (int i = 0; i<len1; i++){
        for(int j=0; j<len2; j++){
            if(arr1[i] != arr2[j]) {
                if (i == 0 && j==0){
                    dp[i][j]=0;
                }else if( i==0 && j!=0){
                    dp[i][j] = dp[i][j-1];
                }else if(i!=0 && j==0){
                    dp[i][j] = dp[i-1][j];
                }else{
                    dp[i][j]=dp[i-1][j] > dp[i][j-1] ? dp[i-1][j]:dp[i][j-1];
                }
            }
            else{
                if (i !=0 && j !=0){
                  dp[i][i] = dp[i-1][j-1] + 1;
                }else{
                     dp[i][i] = 1;
                }
            }

        }
    }
    return dp[len1-1][len2-1];
}

