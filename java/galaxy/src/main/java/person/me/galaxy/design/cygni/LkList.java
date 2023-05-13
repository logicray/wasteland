package person.me.galaxy.design.cygni;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

/**
 * practice of linked list
 * Created by page on 06/12/2016.
 */
public class LkList {
  public static void main(String[] args){
    List<String> a = new LinkedList<String>();
    a.add("alpha");
    a.add("beta");
    a.add("gamma");

    List<String> b = new LinkedList<String>();
    b.add("I");
    b.add("V");
    b.add("X");
    b.add("M");

    //merge
    ListIterator<String> aIter = a.listIterator();
    Iterator<String> bIter = b.iterator();

    while (bIter.hasNext())
    {
      if (aIter.hasNext())
        aIter.next();
      aIter.add(bIter.next());
    }

    System.out.println(a);

    bIter = b.iterator();
    while (bIter.hasNext()) {
      bIter.next(); //skip
      if(bIter.hasNext()){
        bIter.next();
        bIter.remove();
      }
    }

    System.out.println(b);

    a.removeAll(b);
    System.out.println(a);
  }
}
