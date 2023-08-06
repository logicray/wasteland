package person.me.galaxy.util;

public class ArrayUtil {

    public static void  swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
