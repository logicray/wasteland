package person.me.galaxy.design.cygni;


import java.io.*;

/**
 *
 * Created by page on 05/12/2016.
 */
public class Obj2byte {

  /**
   * 对象转数组
   * @param obj
   * @return
   */
  public static byte[] toByteArray (Object obj) {
    byte[] bytes = null;
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    try {
      ObjectOutputStream oos = new ObjectOutputStream(bos);
      oos.writeObject(obj);
      oos.flush();
      bytes = bos.toByteArray ();
      oos.close();
      bos.close();
    } catch (IOException ex) {
      ex.printStackTrace();
    }
    return bytes;
  }

  /**
   * 数组转对象
   * @param bytes
   * @return
   */
  public  static Object toObject (byte[] bytes) {
    Object obj = null;
    try {
      ByteArrayInputStream bis = new ByteArrayInputStream (bytes);
      ObjectInputStream ois = new ObjectInputStream (bis);
      obj = ois.readObject();
      ois.close();
      bis.close();
    } catch (IOException ex) {
      ex.printStackTrace();
    } catch (ClassNotFoundException ex) {
      ex.printStackTrace();
    }
    return obj;
  }

  public static void main(String[] args) {
    Galaxy galaxy = new Galaxy();
    galaxy.setA(1);
    galaxy.setB(2);

    byte[] b = Obj2byte.toByteArray(galaxy);
    System.out.println(new String(b));

    System.out.println("=======================================");

    Galaxy teb = (Galaxy) Obj2byte.toObject(b);
    System.out.println(teb.getA());
    System.out.println(teb.getB());
  }

}