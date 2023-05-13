package person.me.galaxy.design.cygni;

/**
 *
 * a simplified form of queue
 * Created by page on 30/11/2016.
 */
public interface Queue<E> {
  void add(E element);
  E remove();
  int size();
}
