package person.me.galaxy.algorithm;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.BitSet;

/**
 * a simple bloom filter, reference to unit test,
 */
public class BloomFilter {
    private final int size;
    private final int hashNumber;
    private final BitSet bitSet;

    private final static MessageDigest digestFunction;

    static {
        MessageDigest local;
        try {
            local = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            local = null;
            e.printStackTrace();
        }
        digestFunction = local;
    }


    public BloomFilter(int size, int hashNumber) {
        this.size = size;
        this.hashNumber = hashNumber;
        this.bitSet = new BitSet(size);
    }

    /**
     * @param element 需要新增的元素
     * @return 是否成功
     */
    public boolean add(String element) {
        int[] hashes = createHashes(element.getBytes(), hashNumber);
        for (int hash : hashes) {
            bitSet.set(Math.abs(hash % size), true);
        }
        return true;
    }

    /**
     * 判断元素是否在过滤器中
     *
     * @param element 需要判断的元素
     * @return
     */
    public boolean contain(String element) {
        int[] hashes = createHashes(element.getBytes(), hashNumber);
        for (int hash : hashes) {
            if (!bitSet.get(Math.abs(hash % size))) {
                return false;
            }
        }
        return true;
    }

    public int[] createHashes(byte[] data, int hashes) {
        int[] result = new int[hashes];

        byte salt = 0;
        for (int i = 0; i < hashes; i++) {
            result[i] = createHash(data, salt);
            salt++;
        }
        return result;
    }

    public int createHash(byte[] data, byte salt) {
        int result = 0;

        digestFunction.update(salt);
        byte[] digest = digestFunction.digest(data);

        //使用digest的结果按照 byte 拆分出result
        int i = digest.length / 4 -2;
//        int i = 0;
        for (int j = (i * 4); j < (i * 4) + 4; j++) {
            result <<= 8;
            result |= ((int) digest[j]) & 0xFF;
        }
        return result;
    }
}
