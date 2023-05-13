package person.me.galaxy.lambda;

import org.testng.annotations.BeforeClass;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.*;

/**
 * Java8+ Lambda 表达式功能实践
 */
public class LambdaExprTest {

    @BeforeClass
    public static void setup(){
        Set<String> entities=new HashSet<>();
        entities.add("我们");
        entities.add("你们");
        entities.add("吃饭");
        entities.add("厕所");
        List<String> tokens= new ArrayList<>();
        tokens.add("我们");
        tokens.add("都有");
        tokens.add("一个");
        tokens.add("家");

        List<String> x = tokens.stream().filter(entities::contains)
                .map(token -> token=token+"1" ).collect(toList());
    }

    public <T> Map<Boolean, List<T>> split(List<T> list, int n){
        return IntStream.range(0, list.size())
                .mapToObj(i ->  new AbstractMap.SimpleEntry<>(i, list.get(i)))
                .collect(
                        partitioningBy(entry -> entry.getKey() < n, mapping(AbstractMap.SimpleEntry::getValue, Collectors.toList()) )
                );
    }

    public static void main1() {
        List<User> users = new ArrayList<>();
        Map<Integer, List<String>> collect = users.stream().collect(
                Collectors.groupingBy(User::getAge,
                        Collectors.mapping( item ->{
                            //当然你这里也可以构建一个新的对象，进行返回
                            return item.getName();
                        }, Collectors.toList())
                ));
    }
}
