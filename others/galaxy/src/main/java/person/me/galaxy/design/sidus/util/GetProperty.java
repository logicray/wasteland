package person.me.galaxy.design.sidus.util;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * 从配置文件获取参数,
 * Created by page on 17/11/2016.
 */

public class GetProperty {

  public static void inner1() {
    Properties prop = new Properties();
    try {
      //读取jar 内属性文件a.properties
      //String filePath =  GetProperty.class.getResourceAsStream("/project.properties").toString();
      //InputStream in = new BufferedInputStream(new FileInputStream(filePath));
      InputStream in = GetProperty.class.getResourceAsStream("/project.properties");
      prop.load(in);     ///加载属性列表

      System.out.println(GetProperty.class.getResourceAsStream("/config/project.properties").toString());
      for (String key : prop.stringPropertyNames()) {
        System.out.println(key + ":" + prop.getProperty(key));
      }
      in.close();

      ///保存属性到b.properties文件
      //FileOutputStream oFile = new FileOutputStream("b.properties", true);//true表示追加打开
      //prop.setProperty("phone", "10086");
      //prop.store(oFile, "The New properties file");
      //oFile.close();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void inner2() throws IOException {
    Properties p = new Properties();
    InputStream in = GetProperty.class.getClassLoader().getResourceAsStream("config/project.properties");
    p.load(in);
    System.out.println(p.getProperty("username").trim());
  }


  public static Properties outer() {
    Properties prop = new Properties();
    try {
      //读取属性文件a.properties
      String filePath = System.getProperty("user.dir") + "/project.properties";
      InputStream in = new BufferedInputStream(new FileInputStream(filePath));
      prop.load(in);     ///加载属性列表

      //for (String key : prop.stringPropertyNames()) {
        //System.out.println(key + ":" + prop.getProperty(key));
      //}
      in.close();

      ///保存属性到b.properties文件
      //FileOutputStream oFile = new FileOutputStream("b.properties", true);//true表示追加打开
      //prop.setProperty("phone", "10086");
      //prop.store(oFile, "The New properties file");
      //oFile.close();


    } catch (Exception e) {
      e.printStackTrace();
    }

    return prop;
  }

  /**
   * 自动加载配置文件机制，可在修改配置文件后，不用重启服务也能得到配置文件的
   */

  public static void main(String[] args) throws Exception {
    if(args.length == 1)
      if(args[0].equals("test1"))
        System.out.println(999);
    Properties prop = outer();
    while(true) {
      System.out.println(prop.get("days"));
      Thread.sleep(3000);
    }
    //inner2();
  }
}