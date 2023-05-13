package person.me.galaxy.lambda;

import java.util.Arrays;
import java.util.IntSummaryStatistics;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class LambdaUse {

    public void intSummary(){
        List<String>  aList = Arrays.asList("a", "bb", "ccc");
        IntSummaryStatistics statistics = aList.stream().collect(Collectors.summarizingInt(String::length));
        System.out.println(statistics.getMax());
        System.out.println(statistics.getAverage());
    }

    public void iterate(){
        Object[] numbers = Stream.iterate(1, n->n+2).limit(10).toArray();
        for (int i =0;i< numbers.length; i++) {
            System.out.println(numbers[i]);
        }

    }

    public static void main(String[] args) {
        LambdaUse lambdaUse = new LambdaUse();
        lambdaUse.intSummary();
        lambdaUse.iterate();
    }


}
