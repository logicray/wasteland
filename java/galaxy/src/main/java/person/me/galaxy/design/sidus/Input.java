package person.me.galaxy.design.sidus;

import java.util.Scanner;

/**
 * @version 1.0
 * @author logic Wang
 * 2016.3.28
 */

public class Input {
  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);

    System.out.print("What is your name?");
    String name = in.nextLine();

    System.out.print("How old are you?");
    int age = in.nextInt();

    //display
    System.out.println("Hello, " + name + "next year, you will be" + (age + 1));
  }
}
