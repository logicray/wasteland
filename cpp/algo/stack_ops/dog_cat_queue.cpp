/**
 * @file dog_cat_queue.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-01
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include<iostream>
#include<string>
#include<queue>

using namespace std;

class Pet
{
    private:
    string type;

    public:
    Pet(){}
    Pet (string type){
        this->type = type;
    }

    string get_type(){
        return type;
    }
};

class Dog:public Pet
{
    private:
    int count;
    public:
    Dog():Pet("dog"){
    }
};

class Cat:public Pet
{
    public:
    Cat():Pet("cat"){
    }
};

class PriPet
{
    private:
    long count;
    Pet pet;

    public:
    PriPet(long count, string type){
        this->count = count;
        this->pet = Pet(type);
    }

    PriPet(long count, Pet pet){
        this->count = count;
        this->pet = pet;
    }

    long get_count()
    {
        return count;
    }

    Pet get_pet()
    {
        return pet;
    }

    
};

class DogCatQueue
{
    private:
    queue<PriPet> dog_queue;
    queue<PriPet> cat_queue;
    int count =0 ;
    public:
    void add(Pet pet)
    {
        if (pet.get_type()=="dog")
        {
            dog_queue.push(PriPet(count++ ,pet));
        }else if(pet.get_type()=="cat")
        {
            cat_queue.push(PriPet(count++, pet));
        }else{
            throw "no matched pet type";
        }
    }

    Pet pollAll()
    {
        if (cat_queue.empty() && dog_queue.empty())
        {
            throw "empty";
        }
        

        if (cat_queue.empty())
        {
            Pet tmp = dog_queue.front().get_pet();
            dog_queue.pop();
            return tmp;
        }

        if (dog_queue.empty())
        {
             Pet tmp = cat_queue.front().get_pet();
            cat_queue.pop();
            return tmp;
        }
        
        
        if (dog_queue.front().get_count() < cat_queue.front().get_count())
        {
            Pet tmp = dog_queue.front().get_pet();
            dog_queue.pop();
            return tmp;
        }else if (dog_queue.front().get_count() > cat_queue.front().get_count()){
            Pet tmp = cat_queue.front().get_pet();
            cat_queue.pop();
            return tmp;
        }else{
             throw "system error";
        }
        
    }

    Pet pollDog()
    {
        if (!dog_queue.empty())
        {
            Pet tmp = dog_queue.front().get_pet();
            dog_queue.pop();
           return tmp;
        }else{
            throw "dog queue is empty";
        }
        
    }

    Pet pollCat()
    {
        if (!cat_queue.empty())
        {
            Pet tmp = cat_queue.front().get_pet();
            cat_queue.pop();
            return tmp;
        }else{
            throw "cat queue is empty";
        }
        
    }

    bool is_empty()
    {
        return dog_queue.empty() && cat_queue.empty();
    }

    bool isDogEmpty()
    {
        return dog_queue.empty();
    }

    bool is_cat_queue()
    {
        return cat_queue.empty();
    }

    int size(){
        return cat_queue.size() + dog_queue.size();
    }

    
};


int main(int argc, char* argv[])
{
    DogCatQueue my_queue;
    Cat cat1;
    my_queue.add(cat1);
    Dog dog1;
    my_queue.add(dog1);
    Dog dog2;
    my_queue.add(dog2);
    cout << my_queue.size() << endl;
    cout << my_queue.pollAll().get_type() << endl;
    cout << my_queue.size() << endl;
    cout << my_queue.pollAll().get_type() << endl;
    // cout << my_queue.pollDog().get_type() << endl;
    // cout << my_queue.pollDog().get_type() << endl;
    

}