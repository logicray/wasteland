package person.me.galaxy.algorithm.leetcode;

import java.util.Stack;

public class RecursiveInverseStack {

    public static int getAndRemoveLast(Stack<Integer> stack){
        int res = stack.pop();
        if (stack.isEmpty()){
            return res;
        }else {
            int ele = getAndRemoveLast(stack);
            stack.push(res);
            return ele;
        }
    }

    public static void reverse(Stack<Integer> stack){
        if (stack.isEmpty()){
            return;
        }

        int num = getAndRemoveLast(stack);
        reverse(stack);
        stack.push(num);
    }

    public static void main(String[] args) {
        Stack<Integer> stack = new Stack<>();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        reverse(stack);
        while (!stack.isEmpty()){
            System.out.println(stack.pop());
        }

    }
}
