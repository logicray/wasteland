/**
 * cpp exception demo
 */

#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    try
    {
        int age = 15;
        if (age > 18)
        {
            cout << "the age is satisfy\n";
        }
        else
        {
            throw 100;
        }
    }
    catch (int num)
    {
        std::cerr << "catch int" << num<< '\n';
    }
}