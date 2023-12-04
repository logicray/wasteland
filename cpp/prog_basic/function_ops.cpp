/**
 * function practice
 * */

#include <iostream>

using namespace std;
void func1();
// void func2();


void func1() 
{
    cout << "hello ,cpp" << endl;
}

void func2(string country="china")
{
    cout << country << endl;

}

void func3(string fname, int age)
{
    cout << "name is: " << fname << " age is: " << age << "\n";
}

/**
 *  return sum of two parameters
 */ 
int func4(int a, int b)
{
    return a + b;
}

void func5(int &x, int &y)
{
    cout << x << "  " << &x << "\n";
    int z =  x;
    x = y;
    y = z;
}

int num = 5;
int& func6(){
    return num;
}

int main(int argc, char *argv[])
{
    func1();
    func2("america");
    func2();
    func3("xiaoming", 18);
    int sum1 = func4(15, 17);
    cout << sum1;
    int x = 5, y = 7;
    func5(x, y);
    cout << "\n" << x << ", " << y << endl;

    func6()++;
    cout << "num:" << num << endl;
    return 0;
}


