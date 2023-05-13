//print args by foreach

class Prac {
  def main(args: Array[String]): Unit = {
    args.foreach(println)
    //print args split by space
    var i = 0
    while (i < args.length) {
      if (i != 0) {
        print(" ");
      }
      print(args(i))
      i += 1
    }
    println()
    println("Hello, world, from a script!" + args(0))

  }

}