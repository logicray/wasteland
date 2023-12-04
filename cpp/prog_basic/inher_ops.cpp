/**
 * inheritance practice 
 */

#include<iostream>

using namespace std;

class Vehicle{
    public:
        string brand = "Ford";
        void honk(){
            cout << "tuut" << "\n";
        }
};

class Car : public Vehicle{
    public:
        string model = "mustang";
            void printAge(){
        cout << "age" << "\n";
    }
};


class MyClass{
    protected:
    int age = 100;
};

class MyChild :public MyClass{

};

class MyGrandChild: public MyChild{
    public:
    void printAge(){
        cout << age << "\n";
    }

};


class MultiInher : public Car, public MyGrandChild{

};


class Base
{
    public:
    void print()
    {
        cout << "in base class\n";
    }
};

class Derived: public Base
{
    public:
    void print()
    {
        cout << "in derived class " << endl;
        Base::print();
    }
};


class Animal
{
    private:
    string type;
    public:
    Animal():type("animal"){}
    virtual string getType()
    {
        return type;
    }

};

class Dog:public Animal
{
    private:
    string type;
    public:
    Dog():type("dog"){}

    string  getType() override
    {
        return type;
    }
};

class Cat: public Animal
{
    private:
    string type;
    public:
    Cat():type("cat"){}

    string getType() override
    {
        return type;
    }
};

void print_animal_type(Animal* ani)
{
    cout << ani->getType() << endl;
}

int main(int argc, char* argv[])
{
    Car car;
    cout<< car.brand << ", " << car.model << "\n";
    
    MyGrandChild grandChild;
    // grandChild.age = 10;
    grandChild.printAge();
    MultiInher multiInher;
    multiInher.Car::printAge();
    // cout << multiInher.age;
    Derived derived, derived2;
    derived.print();
    derived2.Base::print();

    Animal *animal = new Animal();
    print_animal_type(animal);
    Animal *dog = new Dog();
    print_animal_type(dog);
    Animal *cat = new Cat();
    print_animal_type(cat);


    return 0;
}