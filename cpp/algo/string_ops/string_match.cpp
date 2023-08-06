/**
 * @file string_match.cpp
 * @author logic-pw
 * @brief 字符串匹配, brute force, kmp
 * TODO add bm, sunday
 * @version 0.1
 * @date 2021-01-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include<cstring>
#include<iostream>

using namespace std;

int brute_force(char* s, char* p);
int kmp(char* s, char* p);
int* get_next(char* s);

int main(int argc, char* argv[]){

    char s[6] = "abdeg"; //注意最后有个终止符号
    char p[4] = "deg";
    int loc = kmp(s, p);
    cout << "index location:" << loc << endl;

    char pp[5] = "abab";
    int* next = get_next(pp);
    for (int i=0; i<4; i++){
      cout << "next:" << *(next + i) << endl;
    }
    return 0;
}


int brute_force(char* s, char* p){
    int ls = strlen(s);
    int lp = strlen(p);

    int i=0, j=0;
    while (i < ls && j < lp){
        if (s[i] == p[j]){
            i++;
            j++;
        }else{
            i = i - j + 1;
            j = 0;
        }
    }
    if (j == lp){
        return i - j;
    }else{
        return -1;
    }
}

int kmp(char* s, char* p){
    int ls = strlen(s);
    int lp = strlen(p);
    int* next = get_next(p);
    int i=0, j=0;
    while (i < ls && j < lp){
        if (j == -1 || s[i] == p[j]){
            i++;
            j++;
        }else{
            j = next[j];
        }
    }
    if (j == lp){
        return i - j;
    }else{
        return -1;
    }
}

int* get_next(char* p){
    const int len_p = strlen(p);
    cout << "len p:" << len_p << endl;
    int* next = new int[len_p];
    next[0] = -1;
    int j = 0;
    int k = -1;
    while(j < len_p -1){
        if (k == -1 || p[j] == p[k]){
            ++j;
            ++k;
            if (p[j] == p[k]){
              next[j] = next[k];
            }else{
              next[j] = k;
            }
        }else{
            k = next[k];
        }
    }
    
    return next;
}