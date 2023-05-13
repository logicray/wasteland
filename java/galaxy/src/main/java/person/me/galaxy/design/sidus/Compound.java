package person.me.galaxy.design.sidus;
/**
 *
 * 2 dimension array
 * @version 1.0
 * @author Logic Wang
 */

public class Compound
{
    public static void main(String[] args)
    {
        final double STARTRATE = 10;
        final int NRATES = 6;
        final int NYEARS =10;

        double[] interestRate = new double[NRATES];
        for (int j = 0; j < interestRate.length; j++)
            interestRate[j] = (STARTRATE + j) /100.0;

        double[][] balances = new double[NYEARS][NRATES];

        for (int j = 0; j < balances[0].length; j++)
            balances[0][j] = 10000;

        for (int i = 1; i < balances.length; i++)
        {
            for (int j = 0; j < balances[i].length; j++)
            {
                double oldBalance = balances[i-1][j];

                double interest = oldBalance * interestRate[j];

                balances[i][j] = oldBalance + interest;
            }   
        }

        //print interest rates
        for (int j = 0; j < interestRate.length; j++)
            System.out.printf("%9.0f%%",100 * interestRate[j]);

        System.out.println();
        //print balance table
        for (double[] row : balances)
        {
            for (double b : row)
                System.out.printf("%10.3f", b);

            System.out.println();
        }
    }
}
