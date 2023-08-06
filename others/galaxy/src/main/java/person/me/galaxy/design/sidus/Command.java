package person.me.galaxy.design.sidus;
/*
 * practice  of Command parameters
 * @version 1.0
 * @author Logic Wang
 */

public class Command
{
    public static void main(String[] args)
    {
        if (args[0].equals("-h"))
            System.out.print("Hello, ");
        else if (args[0].equals("-g"))
            System.out.print("Goodbye,");
        //print the other command-line arguments
        for (int i = 1; i<args.length; i++)
            System.out.print(" " + args[i]);
        System.out.println("!");
    }
}
