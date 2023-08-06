package person.me.galaxy.algorithm.tree;

import java.util.Arrays;

/**
 * Serialize binary tree which each node is integer
 */
public class SerializeBinaryTree {

    /**
     *
     * @param root root node of binary tree
     * @return serialized tree
     */
    public static String serialize(Node<Integer> root){
        if (root == null){
            return "#!";
        }
        String res = "";
        if (root.getValue() == null){
            return "#!";
        }
        res += root.getValue() + "!";
//        if (root.getLeft() == null){
//            res +=
//        }
        res += serialize(root.getLeft());
        res += serialize((root.getRight()));
        return res;
    }


    public static Node<Integer> deserialize(String str){
        String[] strNodes = str.split("!");

        if (strNodes.length == 0){
            return null;
        }
        if (strNodes[0].equals("#")){
            return null;
        }


        return deserialize(strNodes);
    }

    public static Node<Integer> deserialize(String[] strNodes){
        if (strNodes.length == 0 || strNodes[0].equals("#")){
            return null;
        }
        if (strNodes.length == 1){
            return new Node<>(Integer.parseInt(strNodes[0]));
        }

        Node<Integer> root = new Node<>(Integer.parseInt(strNodes[0]));
        int l = (strNodes.length-1)/2;
        Node<Integer> left1 = deserialize(Arrays.copyOfRange(strNodes,1,l+1));
        Node<Integer> right1 = deserialize(Arrays.copyOfRange(strNodes,l+1,strNodes.length-1));
        root.setLeft(left1);
        root.setRight(right1);
        return root;
    }


    public static void main(String[] args) {
        Node<Integer> root = new Node<>(10);
        Node<Integer> left1 = new Node<>(9);
        Node<Integer> right1 = new Node<>(7);
        root.setLeft(left1);
        root.setRight(right1);

        String strTree = serialize(root);
        System.out.println(strTree);

        Node<Integer> recoverRoot = deserialize(strTree);
        System.out.println(recoverRoot.getValue());
    }

}
