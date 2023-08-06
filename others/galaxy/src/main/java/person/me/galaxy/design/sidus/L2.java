package person.me.galaxy.design.sidus;

import java.util.Arrays;
import java.util.Scanner;

/**
 * demonstrates array 
 * @version 1.0 2016.03.29
 * @author page Wang
 */

public class L2
{
    public static void main(String[] args)
    {
        Scanner in = new Scanner(System.in);

        System.out.print("How many numbers do u need to draw?");
        int k = in.nextInt();

        System.out.print("what's the highest number you can draw?");
        int n = in.nextInt();

        int[] numbers = new int[n];
        for (int i = 0; i < numbers.length; i++)
            numbers[i] = i + 1;

        int[] result = new int[k];
        for (int i = 0; i < result.length; i++)
        {
            int r = (int) (Math.random() * n);

            result[i] = numbers[r];

            numbers[r] = numbers[n-1];
            n--;
        }
        Arrays.sort(result);
        System.out.println("Bet the following combination. It'll make u rich!");
        for(int r : result)
            System.out.println(r);
    }
}
