/**
 * @file bracket_pair_check.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-02-16
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include<iostream>
#include<vector>
#include<stack>

using namespace std;

bool check_by_stack(vector<char> &chars)
{
    if (chars.size()<=1 || chars[0]==')' || chars.size()%2!=0)
    {
        return false;
    }
    stack<char> s;
    // cout << "AA" ;
    for (int i = 0; i < chars.size(); i++)
    {
        if (chars[i] == '(')
        {
            s.push('(');
        }else if(chars[i] == ')')
        {
            s.pop();
        }
    }
    // cout << s.empty()<< endl;
    return s.empty();
}

bool check_without_stack(vector<char> &chars)
{
    if (chars.size()<=1 || chars[0]==')' || chars.size()%2!=0)
    {
        return false;
    }

    int sum = 0;
    for (int i = 0; i < chars.size(); i++)
    {
        if (chars[i]=='(')
        {
            sum+=1;
        }else if(chars[i] == ')'){
            sum-=1;
            if (sum <= -1)
            {
                break;
            }
        }
    }
    return sum==0;
}


int main(int argc, char const *argv[])
{
    vector<char> chars = {'(',  '(', ')', '(', ')', ')', '(', ')', ')'};
    cout << check_without_stack(chars) << endl;
    cout << check_by_stack(chars) << endl;
    return 0;
}
