package person.me.galaxy.coreJava.inherit;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Map;
import java.util.Scanner;

public class SuperSubClass {
    public static void main(String[] args) {
//        Manager[] managers = new Manager[10];
//        Employee[] staff = managers;
//        staff[0] = new Employee();
//
//        try (Scanner in = new Scanner(new FileInputStream("usr/share/dict/words"),"UTF-8")) {
//            while (in.hasNext()) System.out.println(in.next());
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        }

        Map<Thread, StackTraceElement[]> map = Thread.getAllStackTraces();
        for (Thread t : map.keySet()) {
            StackTraceElement[] frames = map.get(t);
            for (StackTraceElement element:frames) {
                System.out.println(element);
            }
        }

    }

}
