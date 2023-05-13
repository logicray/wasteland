package person.me.galaxy.algorithm;

import org.testng.Assert;
import org.testng.annotations.Test;

public class BloomFilterTest {

    @Test
    public void testBloomFilter() {
        BloomFilter bloomFilter = new BloomFilter(1000, 5);
        bloomFilter.add("a");

        Assert.assertFalse(bloomFilter.contain("b"));
        Assert.assertTrue(bloomFilter.contain("a"));
    }

    @Test
    public void testBloomFilter2() {
        BloomFilter bloomFilter = new BloomFilter(1000, 5);
        int l = 90;
        for (int i = 0; i < l; i++) {
            bloomFilter.add("a" + i);
        }

        for (int i = 0; i < l; i++) {
            Assert.assertFalse(bloomFilter.contain("b" + i));
        }

        for (int i=0;i<l;i++) {
            Assert.assertTrue(bloomFilter.contain("a" + i));
        }
    }
}
