package person.me.galaxy.design.cygni;

/**
 *
 * Created by page on 30/11/2016.
 */

public class Food<T> {

  void show_1(T t){
    System.out.println("show_1  "+ t.toString());
  }

  <E> void show_2(E e){
    System.out.println("show_2  "+e.toString());
  }

  <T> void show_3(T t){
    System.out.println("show_3  "+t.toString());
  }

  public static void main(String[] args) {
    Food<Fruit> o = new Food<Fruit>();
    Fruit f = new Fruit();
    Apple a = new Apple();
    Person p = new Person();
    System.out.println("show_1 演示________________________");
    o.show_1( f );
    o.show_1( a );
//		o.show_1( p );  楼主把这行代码去掉注释看一下，是不能编译通过的。因为在
//		ClassName<Fruit>中已经限定了全局的T为Fruit，所以不能再加入Person;
    System.out.println("show_2 演示________________________");
    o.show_2( f );
    o.show_2( a );
    o.show_2( p );
    System.out.println("show_3 演示________________________");
    o.show_3( f );
    o.show_3( a );
    o.show_3( p );

  }
}


class Fruit { 	public String toString() {	return "Fruit"; } }

class Apple extends Fruit {	public String toString(){ return "Apple";	} }

class Person {	public String toString(){	return "Person";	} }


