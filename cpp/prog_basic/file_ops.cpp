/**
 * it's a cpp file operation
 */

#include<iostream>
#include<fstream>

using namespace std;

int main(int argc, char* argv[])
{
    ofstream my_file("./tmp.txt");
    my_file << "abcdefg,1234567";
    my_file.flush();
    my_file.close();


    ifstream my_reader("./tmp.txt");
    string content;

    while (getline(my_reader, content))
    {
         cout << content << "\n";
    }
    my_reader.close();
   
}