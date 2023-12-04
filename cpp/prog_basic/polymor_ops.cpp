/**
 *  a simple polymorphism demo
 */

#include<iostream>

using namespace std;

class Animal{
    public:
    void animalSound(){
        cout << "this is a animal sound" << "\n";
    }

};

class Dog: public Animal{
    public:
    void animalSound(){
        cout << "wang wang!" << "\n";
    }
};


class Cat : public Animal{
    public:
    void animalSound(){
        cout << "meow meow!" << "\n";
    }
};

int main(int argc, char* argv[])
{
    Dog dog;
    dog.animalSound();

    Cat cat;
    cat.animalSound();
    return 0;
}