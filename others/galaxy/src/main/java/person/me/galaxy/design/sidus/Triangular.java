package person.me.galaxy.design.sidus;

/**
 * triangular array 
 * @version 1.0
 * @author Logic Wang
 */

public class Triangular
{
    public static void main(String[] args)
    {
        final int NMAX = 10;
        
        int[][] odds = new int[NMAX+1][];
        for (int n = 0; n <= NMAX; n++)
            odds[n] = new int[n+1];

        for (int n = 0; n < odds.length; n++)
            for (int k = 0; k < odds[n].length;k++)
              ;
    }
}