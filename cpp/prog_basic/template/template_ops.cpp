/**
 * simple template use demo
 */

#include<iostream>

using namespace std;

template<typename T>
T add(T num1, T num2)
{
    return num1 + num2;
}

template<class T>
class Calculator
{
    private:
    T num1, num2;
    public:
    Calculator(T n1, T n2)
    {
        num1 = n1;
        num2 = n2;
    }

    T add()
    {
        return num1 + num2;
    }

    T substract()
    {
        return num1 - num2;
    }

    T multiply()
    {
        return num1 * num2;
    }

    T division()
    {
        return num1 / num2;
    }

    void dispaly()
    {
        cout << "a add b is:" << add() << endl;
        cout << "a substract b is:" << substract() << endl;
    }
};

template<class T, class U, class W = char>
class ClassTemplate
{
    private:
    T t;
    U u;
    W w;

    public:
    ClassTemplate(T t1, U u1, W v1):t(t1),u(u1),w(v1){}

    void display()
    {
        cout << t << ", \t, " << u << ",\t " << w<< endl;
    }
};

int main(int argc, char* argv[])
{
    int res1 = add<int>(2, 5);
    cout << res1 << endl;
    double res2 = add<double>(1.4, 5.6);
    cout << res2 << endl;

    Calculator<int> *cal = new Calculator<int>(3,5);
    cal->dispaly();

    Calculator<double> *cal2 = new Calculator<double>(2.3, 4.5);
    cal2->dispaly();

    ClassTemplate<int, float> *template2 = new ClassTemplate<int,float>(1, 2.2, 'c');
    template2->display();

    ClassTemplate<float,int,bool> *template3 = new ClassTemplate<float, int, bool>(2.2, 4, true);
    template3->display();

}