package person.me.galaxy.ml.word2vec;

import java.io.File;
import java.io.IOException;

public class TestLearn {
    public static void main(String[] args)  {
       testExpTable();
    }


    public static void test1() throws IOException{
        Learn learn = new Learn();
        long start = System.currentTimeMillis();
        learn.learnFile(new File("library/xh.txt"));
        System.out.println("use time " + (System.currentTimeMillis() - start));
        learn.saveModel(new File("library/javaVector"));
    }

    public static void testExpTable(){
        Learn learn = new Learn();
        System.out.println(learn.expTable.length);
        for (int i =0;i<10;i++){
            System.out.println(learn.expTable[i]);
        }
    }
}
