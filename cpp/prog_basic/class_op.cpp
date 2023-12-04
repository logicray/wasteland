/**
 * simple class operation
 */

#include<iostream>

using namespace std;

class MyClass
{
private:
    /* data */
public:
    int num;
    string s;
    MyClass()
    {

    }
    MyClass(int num, string s);
    ~MyClass();
    /*function implement out of class*/
    void printNum()
    {
        cout << num << "\n";
    }

    void printNumAdd(int num2)
    {
        cout << num + num2 << "\n";
    }

    void printStr();
};

MyClass::MyClass(int n, string s)
{
    this->num = n;
    this->s = s;
}

MyClass::~MyClass()
{
}

void MyClass::printStr()
{
    cout << s << "\n";
}


class Room
{
    private:
    float length;
    float breadth;
    float height;

    public:

    void init(float l, float b, float h)
    {
        length = l;
        breadth = b;
        height = h;
    }

    float calculateArea()
    {
        return length * breadth;
    }

    float calculateVolumn()
    {
        return length * breadth * height;
    }

};


class Wall
{
    private:
    float length;
    float height;

    public:
    Wall(float len, float hgt)
    {
        length = len;
        height = hgt;
    }

    Wall(Wall &wall)
    {
        length = wall.length;
        height = wall.height;
    }

    float calculateArea()
    {   
        return length * height;
    }
};

class Student
{
    private:
    int math_mark;
    int history_mark;

    public:
    // Student(int m1, int m2)
    // {
    //     math_mark = m1;
    //     history_mark = m2;
    // }

    float avg_math_mark(Student s1)
    {
        return (s1.math_mark + math_mark)/2.0;
    }

    Student buildStudent()
    {
        Student stu;
        stu.math_mark = 12;
        stu.history_mark = 14;
        return stu;
    }
};

class Count
{
    private:
    int value;

    public:
    Count():value(5){}

    void operator ++ ()
    {
        ++value;
    }

    Count operator ++ (int)
    {
        Count tmp;
        tmp.value = value++;
        return tmp;
    }

    void display()
    {
        cout << "the value is:" << value << endl;
    }

};


class Complex
{
    private:
    float real;
    float imag;

    public:
    Complex():real(1.2),imag(1.3){}

    Complex operator + (const Complex &another)
    {
        Complex tmp;
        tmp.real = real + another.real;
        tmp.imag = imag + another.imag;
        return tmp;
    }

    void display()
    {
        cout << "the real part is:" << real << " ,the imagine part is:" << imag << endl;
    }

};

int main(int argc, char* argv[])
{
    MyClass myClass;
    myClass.num = 10;
    myClass.s = "sss";
    myClass.printNum();
    myClass.printStr();
    myClass.printNumAdd(4);

    MyClass class2(2999, "bb");
    class2.printNum();
    class2.printStr();


    Room room;
    room.init(2.1, 3.5, 2.6);
    cout << "area:" << room.calculateArea() <<endl;
    cout << "volumn: " << room.calculateVolumn() << endl;

    Wall wall(12.0, 9.9);

    Wall wall2 = wall;

    Wall *wall3 = new Wall(12.0, 9.9);
    Wall *wall4 = wall3;

    cout << "area of wall:" << wall2.calculateArea() << endl;
    cout << &wall << "," << &wall2 << endl;

    Student stu;
    stu = stu.buildStudent();

    Student stu2;
    stu2 = stu2.buildStudent();

    cout << "avg math:" << stu.avg_math_mark(stu2) << endl;

    Count count;
    ++count;
    count.display();
    count++;
    count.display();

    Count cnt2= count++;
    cnt2.display();
    count.display();

    Complex com;
    Complex com2 = com + com;
    com2.display();

    return 0;
}
