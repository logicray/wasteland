/**
 * pointer and reference practice
 * 
 */

#include<iostream>

using namespace std;

int plusA(int x, int y);
float plus(float x, float y);

int plusA(int x, int y)
{
    return x+y;
}
float plusA(float x, float y)
{
    return x+y;
}

void swapA(int& a, int& b)
{
    int tmp = a;
    a = b;
    b = tmp;
}

void swapB(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void dispaly(int a, int b)
{
    cout << "a is: " << a << ", b is: " << b << "\n";
}

void memory_allocation()
{
    int* ptr;
    ptr = new int;
    *ptr = 45;
    cout << *ptr<< endl;
    delete ptr;
}

void memory_allocation_array()
{
    int num;
    cout << "please input the number of students:";
    cin >> num;
    int* gpa = new int[num]; 
    for(int i=0; i< num; i++)
    {
        cin >> *(gpa + i);
    }

    for(int j=0; j<num; j++)
    {
        cout << *(gpa + j) << "\t";
    }

    delete[] gpa;
}

class Student
{
    private:
    int age;

    public:
    Student():age(10){}
    void displayAge()
    {
        cout << "the age of student is: " << age << endl;
    }
};


void memory_allocation_obj()
{
    Student* s = new Student();
    s->displayAge();
    delete s;
}

int main(int argc, char* argv[])
{
    string food = "beef";
    string &dinner = food;
    cout << dinner << "," << &dinner << "\n";
    string* meal = &food;
    cout << meal << "," << *meal << "\n";
    cout << plusA(1,3) << ", " << plusA(1.1f, 2.2f) << "\n";
    int a = 6, b = 9;
    dispaly(a, b);
    swapA(a, b);
    dispaly(a, b);

    int c=10, d= 12;
    dispaly(c, d);
    swapB(&c, &d);
    dispaly(c, d);
    memory_allocation();
    memory_allocation_obj();
    return 0;
}