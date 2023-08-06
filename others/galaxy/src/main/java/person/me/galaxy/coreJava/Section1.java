package person.me.galaxy.coreJava;

import java.math.BigInteger;
import java.util.Scanner;

public class Section1 {
    public static void main(String[] args) {
        bigIntegerUse();
    }

    /**
     *
     */
    public static void bigIntegerUse(){
        Scanner in = new Scanner(System.in);
        System.out.println("How many numbers do you ");
        int k = 9;
        int n = 50;
        BigInteger lotteryOdds =  BigInteger.ONE;
        for (int i = 1; i<=k;i++){
            lotteryOdds = lotteryOdds.multiply(BigInteger.valueOf(n-i-1).divide(BigInteger.valueOf(k)));
        }
        System.out.println("1 in " + lotteryOdds);
    }
}
