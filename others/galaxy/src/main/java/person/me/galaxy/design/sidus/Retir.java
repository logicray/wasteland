package person.me.galaxy.design.sidus;

import java.util.Scanner;

/**
 * while
 * @version 1.0
 * @author Logic Wang
 */

public class Retir {
  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);

    System.out.print("How much money do you need to retire?");
    double goal = in.nextDouble();

    System.out.print("How much money will u contribute every year?");
    double payment = in.nextDouble();

    System.out.print("Interest rate in %: ");
    double interestRate = in.nextDouble();

    double balance = 0;
    int years = 0;

    //
    while (balance < goal) {
      balance += payment;
      double interest = balance * interestRate / 100;
      balance += interest;
      years++;
    }
    System.out.println("u can retire in " + years + " years.");
  }
}
