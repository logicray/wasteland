/**
 * @file common_linked_list.cpp
 * @author logic-pw
 * @brief 
 * @version 0.1
 * @date 2022-03-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include<iostream>
#include<string>

using namespace std;

class Node;
void print_common(Node , Node);

class Node{
    private:
    int value;
    Node *next;

    public:
    Node(int data){
        this->value = data;
    }

    void set_next(Node *next){
        this->next = next;
    }

    int get_value(){
        return value;
    }

    Node* get_next(){
        return next;
    }
};

int main(int argc, char *argv[])
{
    Node head1(2);
    Node node12(4);
    head1.set_next(&node12);
    Node node13(6);
    node12.set_next(&node13);

    Node head2(3);
    Node node22(4);
    head2.set_next(&node22);
    Node node23(5);
    node22.set_next(&node23);

    print_common(head1, head2);
}

void print_common(Node head1, Node head2)
{
    while (head1.get_next()!=NULL &&  head2.get_next()!=NULL)
    {
        if (head1.get_value() > head2.get_value())
        {
            head2 = *(head2.get_next());
        }else if (head1.get_value() < head2.get_value())
        {
            head1 = *(head1.get_next());
        }else
        {
            cout << head1.get_value() << endl;
            head1 = *(head1.get_next());
            head2 = *(head2.get_next());
        }
    }
}