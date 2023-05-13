/* 
    some basic string operation in cpp
*/

#include<iostream>

using namespace std;

char* c_str()
{
    static char str[4] = "ccc";
    return str;
}

void read_str()
{
    char str[20];
    cin.get(str, 20);
    cout << str << endl;
}

int main(int argc, char* argv[])
{
    string a = "hello";
    cout << "input a word: \n"; 
    getline(cin, a);

    string b = ",world";
    cout<<a+b<< endl;
    cout<< a.append(",--").append(b)<<endl;
    cout<< "size of a:" << a.length()<< endl;
    cout<< a[0] << endl;

    b[0] = ':';
    cout << b <<endl;

    // char c_str[] = {'c', 'p','\0'};
    cout << c_str() << endl;

    read_str();
    return 0;
}
