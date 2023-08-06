package person.me.galaxy.algorithm;

import org.testng.Assert;
import org.testng.annotations.Test;


public class C1Test {

    @Test
    public void testFindAnyRepeat() {
        int[] input = new int[]{0, 1, 2, 3, 4, 3, 4};
        int firstRepeat = C1.findAnyRepeat(input);
        Assert.assertEquals(firstRepeat, 3);
        System.out.println(firstRepeat);
    }
}