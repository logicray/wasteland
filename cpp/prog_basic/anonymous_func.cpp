#include<iostream>

using namespace std;

/**
[]   //不截取任何变量,试图在Lambda内使用任何外部变量都是错误的(全局变量除外).
[&]  //截取外部作用域中所有变量，并作为引用在函数体中使用
[=]  //截取外部作用域中所有变量，并拷⻉一份在函数体中使用
[=, &foo]  //截取外部作用域中所有变量，并拷⻉一份在函数体中使用，但是对foo变量使用引用
[bar]  //截取bar变量并且拷⻉一份在函数体重使用，同时不截取其他变量
[this]  截取当前类中的this指针
[x, &y]  //x 按值捕获, y 按引用捕获.
[&, x]    //x显式地按值捕获. 其它变量按引用捕获
[=, &z]   //z按引用捕获. 其它变量按值捕获
 */

void anony1(){
	auto func = [](){
          cout << "hello, world" << endl;
      };
     func();
}

void anony2(){
   auto func = [](int x, int y){
	   return x+y;
   };

   cout << func(2, 4) << endl;
}

void anony3(){
	auto func = [](int& x){
		++x;
	};
	int a = 20;
	func(a);
	cout << a << endl;
}


void anony4(){
	auto func = [](int x, int y) -> int {
        return x * y;
	};
	cout << func(7, 8) << endl;
}

int main(int argc, char* argv[]){
	anony1();
	anony2();
	anony3();
	anony4();
}

