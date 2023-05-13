package person.me.galaxy.design.sidus;

/**
 *
 * Created by page on 28/11/2016.
 * 静态变量会按照声明的顺序先依次声明并设置为该类型的默认值，但不赋值为初始化的值。
 * 声明完毕后,再按声明的顺序依次设置为初始化的值，如果没有初始化的值就跳过。
 */
public class StaticTest {
  public static Test1 t = new Test1();
  public static int a = 0;
  public static int b;

  public static void main(String[] args) {
    System.out.println(StaticTest.a);
    System.out.println(StaticTest.b);
  }
}

class Test1 {
  public Test1() {
    StaticTest.a ++;
    StaticTest.b ++;
  }

}
