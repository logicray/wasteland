package person.me.galaxy.design.strategy;

/**
 *
 * Created by page on 30/12/2016.
 */
public class MiniDuckSimulator {
    public static void main(String[] args) {
        Duck mallard = new MallardDuck();
        mallard.performQuack();
        mallard.performFly();
    }
}
