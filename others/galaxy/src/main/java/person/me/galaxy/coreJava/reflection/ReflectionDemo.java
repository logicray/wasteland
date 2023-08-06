package person.me.galaxy.coreJava.reflection;

import java.lang.reflect.Modifier;
import java.util.Scanner;

public class ReflectionDemo {
    public static void main(String[] args) {
        String name = "";
        if (args.length > 0) {
            name = args[0];
        } else {
            Scanner in = new Scanner(System.in);
            System.out.println("enter class name (e.g. java.util.Date): ");
            name = in.next();
        }
        try {
            Class cls = Class.forName(name);
            Class superCls = cls.getSuperclass();
            String modifiers  = Modifier.toString(cls.getModifiers());
            if (modifiers.length() > 0){
                System.out.print(modifiers + " ");
            }
            System.out.print("class " + name);
            if (superCls != null && superCls != Object.class){
                System.out.print(" extends " + superCls.getName());
            }
            System.out.println("\n{\n");
            //
            System.out.println();

        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        System.exit(0);


    }
}
