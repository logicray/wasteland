package person.me.galaxy.design.sidus;

import java.util.Scanner;

/**
 * demonstrate for loop
 * @version 1.0
 * @author Logic_Wang
 * 
 */

public class Lottery 
{
    public static void main(String[] args)
    {
        Scanner in = new Scanner(System.in);

        System.out.print("How many numbers do you need to draw?");
        int k = in.nextInt();

        System.out.print("What's the highest number you can draw?");
        int n = in.nextInt();

        //binomial coefficent
        int lotteryOdds = 1;
        for (int i = 1; i <= k; i++)
            lotteryOdds = lotteryOdds * (n-i+1) / i;

        System.out.println("your odds are 1 in " + lotteryOdds + ". Good luck!");
    }
}
