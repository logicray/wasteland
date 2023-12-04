/**
 * @file sieve.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <cmath>
#include <iostream>
#include<cstring>

using namespace std;

int sieve(int);

int main(int argc, char* argv[]){
    int n = 100;
    sieve(n);
    return 0;
}


int sieve(int n)
{
    // const int n = 100;
    bool a[101];
    int i, j;

    memset(a, 1, sizeof(a));
    a[1] = 0;
    for(i = 2; i <= sqrt(n); i ++)
    {
        if(a[i])
        {
            for(j = 2; j <= n/i; j ++)
            {
                a[i*j] = 0;
            }
        }
    }
    for(i = 2; i <= n; i ++)
    {
        if(a[i])
            cout << i << " ";
    }
    return 0;
}
