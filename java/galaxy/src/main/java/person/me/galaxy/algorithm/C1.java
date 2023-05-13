package person.me.galaxy.algorithm;

import person.me.galaxy.util.ArrayUtil;

public class C1 {

    /**
     * 输入由1-n的数字组成的数组，输出第一个重复的数字
     * @param nums 由整数1-n组成的数组
     * @return 第一个重复的数字
     */
    public static int findAnyRepeat(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] != i) {
                if (nums[i] == nums[nums[i]]) {
                    return nums[i];
                }
                ArrayUtil.swap(nums, i, nums[i]);
            }
        }
        return -1;
    }


    public static void main(String[] args) {
        int[] input = new int[]{0, 1, 2, 3, 4, 3, 4};
        System.out.println(findAnyRepeat(input));
    }


}
