package person.me.galaxy.algorithm.leetcode;

import java.util.Stack;


/**
 *
 */
public class GetMin {
    private Stack<Integer> data;
    private Stack<Integer> min;

    public GetMin() {
        this.data = new Stack<>();
        this.min = new Stack<>();
    }

    public void push(int num) {
        data.push(num);
        if (min.isEmpty() || num <= min.peek()) {
            min.push(num);
        }
    }

    public int pop() {
        int res = data.pop();
        if (res == getMin()) {
            min.pop();
        }
        return res;
    }

    public int getMin() {
        if (min.isEmpty()) {
            throw new RuntimeException("");
        }
        return min.peek();
    }


}
