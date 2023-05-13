package person.me.galaxy.coreJava;

import java.lang.reflect.Field;

public class Ls {
//    public static void main(String[] args) {
//        if (args == null || new Ls(){{Ls.main(null);}}.equals("123")){
//            System.out.println("a");
//        }else {
//            System.out.println("b");
//        }
//    }


    public static void main(String[] args) throws Exception {
        Class integer = Integer.class.getDeclaredClasses()[0];
        Field field = integer.getDeclaredField("cache");
        field.setAccessible(true);
        Integer[] array =  (Integer[]) field.get(integer);
        array[130] = array[131];
        System.out.printf("%d", 1 + 1);
    }


    public static void test1(){
        Integer a = -128;
        Integer b = -128;
        Integer c = -129;
        Integer d = -129;
        System.out.println(a == b);
//true
        System.out.println(c == d);//false

    }


}
