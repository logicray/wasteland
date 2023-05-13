package person.me.galaxy.design.cygni;

import java.util.Random;

/**
 * a simple Bloom filter, data type can be String, Integer ,Double
 * Created by page on 08/12/2016.
 */
public class BloomFilter {
  private int[] bTable;
  private int maxLen = 0;

  public BloomFilter(int[] data,int tableLen){
    this.maxLen = tableLen;
    this.bTable = new int[tableLen];
    for(int d:data){
      int x = hash(d, maxLen);
      //System.out.println("x=" + x);
      bTable[x] = 1;
    }
  }

  private int hash(int d, int max){
    return d%max;
  }

  public boolean get(int number){
    int x = hash(number, maxLen);
    return bTable[x] == 1;
  }

  public static void main(String[] args) {
    //构造一个长度为1000的数组,然后用随机数初始化
    Random random = new Random();
    int[] a = new int[50];
    for(int i= 0; i<a.length; i++)
      a[i] = random.nextInt(50);

    for(int i:a)
      System.out.println(i);

    //用上述数组初始化一个长度为10的bloom filter
    BloomFilter bf = new BloomFilter(a,100);
    System.out.println(bf.get(12));
  }
}
