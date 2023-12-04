/**
 * simple structure operation
 */

#include <iostream>

using namespace std;

struct Person
{
    char name[50];
    int age;
    float salary;
};

struct Distance
{
    int km;
    int meter;
};

void display_person(Person p)
{
    cout << "the detail of the person: \n name is:" << p.name << ", age is: " << p.age << ", salary is:" << p.salary << endl;
}

int main(int argc, char *argv[])
{
    Person p;
    cout << "enter name:";
    cin.get(p.name, 50);
    cout << "enter age:";
    cin >> p.age;
    cout << "enter salary:";
    cin >> p.salary;

    display_person(p);

    Distance d, *ptr_d;
    ptr_d = &d;
    d.km = 104;
    d.meter = 12;

    cout << "distance:" << (*ptr_d).km << "," << (*ptr_d).meter << endl;
}
