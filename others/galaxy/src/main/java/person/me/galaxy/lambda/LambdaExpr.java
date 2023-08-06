package person.me.galaxy.lambda;

import person.me.galaxy.lambda.entity.Album;
import person.me.galaxy.lambda.entity.Artist;

import java.util.*;
import java.util.function.BinaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.util.stream.Collectors.*;

/**
 * Java8+ Lambda 表达式功能实践
 */
public class LambdaExpr {

    private int x;

    public LambdaExpr() {
    }


    public static void func1() {
        Set<String> entities = new HashSet<>();
        entities.add("我们");
        entities.add("你们");
        entities.add("吃饭");
        entities.add("厕所");
        List<String> tokens = new ArrayList<>();
        tokens.add("我们");
        tokens.add("都有");
        tokens.add("一个");
        tokens.add("家");

        List<String> x = tokens.stream().filter(entities::contains)
                .map(token -> token = token + "1").collect(toList());
    }

    public <T> Map<Boolean, List<T>> split(List<T> list, int n) {
        return IntStream.range(0, list.size())
                .mapToObj(i -> new AbstractMap.SimpleEntry<>(i, list.get(i)))
                .collect(
                        partitioningBy(entry -> entry.getKey() < n, mapping(AbstractMap.SimpleEntry::getValue, Collectors.toList()))
                );
    }

    public static void t() {
        List<User> users = new ArrayList<>();
        Map<Integer, List<String>> collect = users.stream().collect(
                Collectors.groupingBy(User::getAge,
                        Collectors.mapping(item -> {
                            //当然你这里也可以构建一个新的对象，进行返回
                            return item.getName();
                        }, Collectors.toList())
                ));
        System.out.println(collect);
    }


    public static void flatMapT() {
        List<Integer> together = Stream.of(Arrays.asList(1, 2), Arrays.asList(3, 4))
                .flatMap(numbers -> numbers.stream())
                .collect(toList());
        System.out.println((Arrays.asList(1, 2, 3, 4).equals(together)));
    }


    public static void reduceT() {
        int count = Stream.of(1, 2, 3)
                .reduce(2, (acc, element) -> acc + element);
        System.out.println(count);
    }

    private static void reduceR() {
        BinaryOperator<Integer> accumulator = (acc, element) -> acc + element;
        int count = accumulator.apply(
                accumulator.apply(
                        accumulator.apply(0, 1),
                        2), 3);
        System.out.println(count);
    }


    public static void reduceSum() {
        List<Integer> aList = new ArrayList<>();
        aList.add(1);
        aList.add(3);
        aList.add(5);
        int n = aList.stream().reduce((a, b) -> a + b).get();
        System.out.println(n);
    }


    public Map<Artist, Long> numberOfAlbums(Stream<Album> albums) {
        return albums.collect(groupingBy(album -> album.getMainMusician(), counting()));
    }

    public Map<Artist, List<String>> nameOfAlbums(Stream<Album> albums) {
        return albums.collect(groupingBy(Album::getMainMusician,
                mapping(Album::getName, toList())));
    }

    private int addIntegers(List<Integer> values) { return values.parallelStream().mapToInt(i -> i).sum();
    }



    public static void main(String[] args) {
        reduceR();
    }
}
