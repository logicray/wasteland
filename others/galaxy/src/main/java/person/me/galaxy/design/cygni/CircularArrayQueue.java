package person.me.galaxy.design.cygni;

/**
 *
 * Created by page on 30/11/2016.
 */
public class CircularArrayQueue<E> implements Queue<E>{
  private E[] elements;
  private int head;
  private int tail;

  CircularArrayQueue(int capacity){
    elements = (E[]) new Object[capacity];

  }

  public void add(E element) {

  }

  public E remove(){
    return null;
  }

  public int size() {
    return 0;
  }


}
