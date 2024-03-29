package person.me.galaxy.design.cygni;

/**
 * Created by page on 19/12/2016.
 */
public class TransferRunnable implements  Runnable{
  private Bank bank;
  private int fromAccount;
  private double maxAmount;
  //delay 10 times a random num
  private int DELAY = 10;

  public TransferRunnable(Bank b, int from, double max){
    bank = b;
    fromAccount = from;
    maxAmount = max;
  }

  public void run(){
    try{
      while (true){
        int toAccount = (int) (bank.size() * Math.random());
        double amount = maxAmount * Math.random();
        bank.transfer(fromAccount, toAccount ,amount);
        Thread.sleep((int)(DELAY * Math.random()));
      }
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }
}
