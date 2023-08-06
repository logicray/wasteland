package person.me.galaxy.design.sidus;

/**
 * this program display hello
 */

public class Hello {
  public static void main(String[] args) {

//    if(args.length == 1)
//      if(args[0].equals("test1"))
//        GetProperty.outer();


    String[] greeting = new String[3];
    greeting[0] = "fwr";
    greeting[1] = "hello";
    greeting[2] = "tt";

    for (String g : greeting) {
      //assert g.equals("fwr"):g;
      System.out.println(g);
    }
  }
}
