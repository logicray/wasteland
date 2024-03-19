package pack

/*   */
import scala.collection.mutable.Map
import Practise

import scala.collection.mutable


object Simple {
  def main(args: Array[String]): Unit = {
    println(factorial(30))

  }
  //Practise p ;
  //use of array
  def use_of_array() = {
    val greetStrings:Array[String] = new Array[String](3)
    greetStrings(0) = "Hello"
    greetStrings(1) = ", "
    greetStrings(2) = "world!\n"

    for(i<-0 to 2)
      print(greetStrings(i))
  }

  //use of map
  def use_of_map() = {
    println("Hello, world!")
    print(0.to(2))

    val treasureMap = mutable.Map[Int, String]()
    treasureMap += (1 -> "Go to island.")
    treasureMap += (2 -> "Find big X on ground.")
    treasureMap += (3 -> "Dig.")
    println(treasureMap(2))
  }

  //
  def max(x: Int, y: Int): Int = {
    if (x > y) x
    else y
  }

  //
  def max2(x: Int, y:Int) :Int = if (x > y) x else y

  //
  def factorial(x: BigInt): BigInt =
    if (x == 0) 1 else x * factorial(x - 1)

}
