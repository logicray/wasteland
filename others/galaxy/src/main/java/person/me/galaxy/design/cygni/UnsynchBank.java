package person.me.galaxy.design.cygni;

/**
 *
 * Created by page on 19/12/2016.
 */
public class UnsynchBank {
  //100 account total
  public static final int NACCOUNTS = 100;
  //initial balance od each account is 1000
  public static final double INITIAL_BALANCE = 1000;

  public static void main(String[] args) {
    Bank b = new Bank(NACCOUNTS, INITIAL_BALANCE);
    int i;
    for(i = 0; i < NACCOUNTS; i++){
      TransferRunnable r = new TransferRunnable(b, i ,INITIAL_BALANCE);
      Thread t = new Thread(r);
      t.start();
    }
  }
}
