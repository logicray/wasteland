/* array operation in cpp
*/

#include <iostream>
#include <vector>

using namespace std;

void func_loop()
{
    int a[] = {3, 6, 9, 1};
    for (const int &n : a)
    {
        cout << n << "\t";
    }
}

void func_loop2()
{
    int a[] = {3, 6, 9, 1};
    for (const int n : a)
    {
        cout << n << "\t";
    }
}

void multi_array()
{
    int arr[2][3];

    cout << "please enter 6 numbers" << endl;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
            cin >> arr[i][j];
    }

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
            cout << "arr[" << i << "][" << j << "]:" << arr[i][j] << "\n";
    }
}

void multi_array2()
{
    int arr[2][3][2] = {
        {{1, 3}, {2, 4}, {3, 6}},
        {{4, 5}, {5, 6}, {6, 7}}

    };

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 2; ++k)
                cout << arr[i][j][k] << " ";
        }
    }
}


int total(int marks[5])
{
    int sum = 0;
    for(int i=0; i<5; ++i)
    {
        sum+=marks[i];
    }
    
    return sum;
}

void func_loop_vec()
{
    vector<int> vec = {2, 5, 7, 0};
    for (int a : vec)
    {
        cout << a << " ";
    }
    cout << "\n";
}

int main(int argc, char *argv[])
{
    string a[4] = {"hh", "22", "lo", "90"};
    a[1] = "new22";
    // cout << a.length() << endl;
    cout << a[1] << "\n";
    int num[4];
    num[0] = 1;

    float x = 11.8;
    float &y = x;
    cout << y << "\n"
         << &y << endl;

    string food = "beaf";
    string *meal = &food;
    cout << "food:" << food << " reference of food is: " << &food << " pointer is: " << meal << "\n";

    func_loop();
    cout << "\n";
    func_loop2();
    cout << "\n";
    func_loop_vec();
    multi_array2();

    int marks[5] = {1,2,3,4,5};
    cout << total(marks)<< endl;
    return 0;
}
