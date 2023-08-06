/**
 * if/else/for/while
 * 
**/
#include <iostream>

using namespace std;
int main(int argc, char *argv[])
{
    int time = 13;
    if (time < 10)
    {
        cout << "good moring\n";
    }
    else
    {
        cout << "good afternoon \n";
    }
    string result = time > 10 ? "good moring" : "good evening";
    cout << result << "\n";

    int t2 = 110;
    switch (t2)
    {
    case 7:
        cout << "hello 7";
        break;
    case 8:
        cout << "hello 8";
        break;
    case 10:
        cout << "hello 10";
        break;
    default:
        cout << "not match";
        break;
    }

    int w_i = 7;
    do
    {
        cout<< "i is:" << w_i << "\n";
        w_i --;
    } while (w_i > 0);
    
    return 0;
}