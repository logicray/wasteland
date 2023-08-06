package person.me.galaxy.design.strategy;

/**
 *
 * Created by page on 30/12/2016.
 */
public class MallardDuck extends Duck {
    public MallardDuck() {
        quackBehavior = new Quack();
        flyBehavior = new FlyWithWings();
    }

    public void display() {
        System.out.println("I am a real mallard duck");
    }
}
