package person.me.galaxy.design.strategy;

/**
 * Created by page on 30/12/2016.
 */
public class MuteQuack implements QuackBehavior{
    public void quack(){
        System.out.println("<<silence>>");
    }
}
