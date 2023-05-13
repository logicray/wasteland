package person.me.galaxy.design.sidus;

import java.util.Scanner;

/**
 *  This program demonstrates a do--while loop
 *  @version 1.0
 *  @author Cay Horstmann
 */

public class R2 {
  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);

    System.out.print("How much money will u contribute every year?");
    double payment = in.nextDouble();

    System.out.print("Interest rate in %:");
    double interestRate = in.nextDouble();

    double balance = 0;
    int years = 0;

    String input;

    do {
      balance += payment;
      double interest = balance * interestRate / 100;
      balance += interest;

      years++;

      System.out.printf("After year %d, your balance is %,.2f%n", years, balance);

      System.out.print("Ready to retire? (Y/N) ");
      input = in.next();
    }
    while (input.equalsIgnoreCase("N"));
  }
}
