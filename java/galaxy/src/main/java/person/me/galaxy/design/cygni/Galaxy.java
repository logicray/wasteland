package person.me.galaxy.design.cygni;


import java.io.Serializable;

/**
 * Created by page on
 * 05/12/2016.
 */
public class Galaxy implements Serializable {
  public int a;
  public int b;
  Galaxy(int x, int y){
    this.a = x;
    this.b =y;
  }

  Galaxy(){
  }

  public int getA() {
    return a;
  }

  public int getB() {
    return b;
  }

  public void setA(int a) {
    this.a = a;
  }

  public void setB(int b) {
    this.b = b;
  }
}
