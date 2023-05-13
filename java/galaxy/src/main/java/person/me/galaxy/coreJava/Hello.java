package person.me.galaxy.coreJava;

import person.me.galaxy.coreJava.inherit.Employee;

import javax.swing.*;
import java.util.Arrays;
import java.util.Date;
import java.util.List;


/**
 * @author page
 */
public class Hello {
    public static void main(String[] args) {
        String x= "朢䖯㊡♋\uD83C\uDF19";
        System.out.println(x);
        System.out.println(x.length());
        System.out.println(x.charAt(0));
        System.out.println(x.codePointAt(0));
        System.out.println(x.codePointCount(0, x.length()));
        System.out.println("we will not 'adfe'");

        List<String> x2 = Arrays.asList("1","2");
        Employee[] employees = x2.stream().map(Employee::new).toArray(Employee[]::new);

        Timer t = new Timer(1000, event -> System.out.println("The time is " + new Date()));
        t.start() ;


    }
}
